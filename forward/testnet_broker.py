from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Literal, Optional

import requests
from urllib.parse import urlencode


class TestnetAuthError(RuntimeError):
    """Raised when API key/secret env vars are missing."""


class TestnetAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class OrderValidationError(ValueError):
    """Raised when local order quantization/validation rejects an order before API submission."""

    def __init__(self, message: str, *, meta: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.meta = meta or {}


class AmbiguousOrderError(RuntimeError):
    """Raised when exchange-side execution may have happened but could not be proven locally."""

    def __init__(self, message: str, *, client_order_id: str, context: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.client_order_id = str(client_order_id)
        self.context = context or {}


@dataclass
class RateLimitConfig:
    max_retries: int = 6
    base_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 20.0


@dataclass
class TestnetConfig:
    # Binance USD-M Futures *demo/testnet* REST base URL.
    # Ref: Binance docs ("Testnet API Information")
    base_url: str = "https://demo-fapi.binance.com"
    recv_window_ms: int = 5000
    exchange_info_ttl_seconds: int = 900
    # IMPORTANT: use default_factory (mutable dataclass default).
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sign(secret: str, query: str) -> str:
    return hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        try:
            return json.loads(resp.text)
        except Exception:
            return {"raw": resp.text}


def _to_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    s = str(value).strip()
    if s == "":
        raise ValueError("empty numeric value")
    return Decimal(s)


def floor_to_step(value: Any, step: Any) -> Decimal:
    value_d = _to_decimal(value)
    step_d = _to_decimal(step)
    if step_d <= 0:
        raise ValueError("step must be > 0")
    units = (value_d / step_d).to_integral_value(rounding=ROUND_DOWN)
    return units * step_d


def format_decimal(d: Decimal) -> str:
    if not isinstance(d, Decimal):
        d = _to_decimal(d)
    if not d.is_finite():
        raise ValueError("numeric value must be finite")
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("", "-0"):
        return "0"
    return s


def _safe_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    try:
        return _to_decimal(value)
    except Exception:
        return default


def _payload_code(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    code = payload.get("code")
    if code is None:
        return None
    try:
        return int(code)
    except Exception:
        return None


SubmitErrorKind = Literal["ambiguous", "definitive_reject", "transient_system"]


def classify_submit_error(error: TestnetAPIError) -> SubmitErrorKind:
    if BinanceFuturesTestnetBroker._is_ambiguous_submit_error(error):
        return "ambiguous"

    code = _payload_code(getattr(error, "payload", None))
    try:
        status_code = int(error.status_code) if error.status_code is not None else None
    except Exception:
        status_code = None

    if code in (-1003, -1021, -1022, -2014, -2015):
        return "transient_system"
    if status_code in (401, 403, 418, 429):
        return "transient_system"

    msg = str(error).lower()
    for marker in (
        "rate limit",
        "too many requests",
        "api-key",
        "signature",
        "recvwindow",
        "timestamp for this request",
    ):
        if marker in msg:
            return "transient_system"

    if status_code == 400 and code == -2010:
        return "definitive_reject"
    if status_code == 400:
        return "ambiguous"
    return "transient_system"


@dataclass(frozen=True)
class _QtyFilter:
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal
    filter_type: str


@dataclass(frozen=True)
class _PriceFilter:
    min_price: Decimal
    max_price: Decimal
    tick_size: Decimal


@dataclass(frozen=True)
class _SymbolFilters:
    symbol: str
    lot_size: _QtyFilter
    market_lot_size: _QtyFilter
    price_filter: _PriceFilter
    min_notional: Optional[Decimal]
    min_notional_filter_type: Optional[str]


class BinanceFuturesTestnetBroker:
    """Minimal Binance USD-M Futures TESTNET broker (REST).

    Secrets are read from env vars (never config):
      - BINANCE_TESTNET_API_KEY
      - BINANCE_TESTNET_API_SECRET

    Notes:
      - Market data ingestion remains on mainnet via BinanceLiveKlineSource.
      - Conditional order types (STOP_MARKET / TAKE_PROFIT_MARKET / etc.) must be routed
        via POST /fapi/v1/algoOrder as per Binance API change (Dec 2025).
    """

    def __init__(
        self,
        *,
        api_key_env: str = "BINANCE_TESTNET_API_KEY",
        api_secret_env: str = "BINANCE_TESTNET_API_SECRET",
        cfg: Optional[TestnetConfig] = None,
    ) -> None:
        self.api_key = (os.getenv(api_key_env) or "").strip()
        self.api_secret = (os.getenv(api_secret_env) or "").strip()
        if not self.api_key or not self.api_secret:
            raise TestnetAuthError(
                f"Missing testnet secrets. Set env vars {api_key_env} and {api_secret_env}."
            )

        self.cfg = cfg or TestnetConfig()
        if not str(self.cfg.base_url).lower().startswith("https://"): 
            raise ValueError("TestnetConfig.base_url must start with https://")
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        self._exchange_info_cache: Optional[dict[str, Any]] = None
        self._exchange_info_cached_at: float = 0.0
        self._symbol_filters_cache: dict[str, _SymbolFilters] = {}
        self._last_quantization: dict[str, Any] = {}
        self._client_order_seq: int = 0

    # -------------------------
    # Core request machinery
    # -------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        timeout: float = 10.0,
    ) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        base_params = dict(params or {})

        rl = self.cfg.rate_limit
        attempt = 0
        backoff = float(rl.base_backoff_seconds)

        while True:
            attempt += 1

            # Rebuild params each attempt so signed requests refresh timestamp/signature on retries.
            p = dict(base_params)
            if signed:
                p["timestamp"] = _now_ms()
                p["recvWindow"] = int(self.cfg.recv_window_ms)

            # Build encoded query string for signature using the *same* ordering and URL encoding that is sent.
            pairs: list[tuple[str, str]] = []
            for k, v in p.items():  # preserve insertion order
                if v is None:
                    continue
                pairs.append((str(k), str(v)))

            if signed:
                query = urlencode(pairs, doseq=True)
                sig = _sign(self.api_secret, query)
                pairs.append(("signature", sig))

            try:
                resp = self.session.request(method, url, params=pairs, timeout=timeout)
            except requests.RequestException as e:
                if attempt > rl.max_retries:
                    raise TestnetAPIError(f"Network error after retries: {e}") from e
                self._sleep_backoff(backoff)
                backoff = min(float(rl.max_backoff_seconds), backoff * 2.0)
                continue

            if resp.status_code in (418, 429):
                if attempt > rl.max_retries:
                    payload = _safe_json(resp)
                    raise TestnetAPIError(
                        f"Rate limit after retries (HTTP {resp.status_code})",
                        status_code=resp.status_code,
                        payload=payload if isinstance(payload, dict) else {"raw": str(payload)},
                    )
                ra = resp.headers.get("Retry-After")
                if ra is not None:
                    try:
                        self._sleep_backoff(float(ra))
                    except Exception:
                        self._sleep_backoff(backoff)
                else:
                    self._sleep_backoff(backoff)
                backoff = min(float(rl.max_backoff_seconds), backoff * 2.0)
                continue

            if resp.status_code >= 400:
                payload = _safe_json(resp)
                code = payload.get("code") if isinstance(payload, dict) else None
                # -1003 is a common "too many requests" style error.
                if code in (-1003,) and attempt <= rl.max_retries:
                    self._sleep_backoff(backoff)
                    backoff = min(float(rl.max_backoff_seconds), backoff * 2.0)
                    continue

                err = f"Binance API error HTTP {resp.status_code}"
                if isinstance(payload, dict):
                    if payload.get("code") is not None:
                        err += f" code={payload.get('code')}"
                    if payload.get("msg") is not None:
                        err += f" msg={payload.get('msg')}"
                raise TestnetAPIError(
                    err,
                    status_code=resp.status_code,
                    payload=payload if isinstance(payload, dict) else {"raw": str(payload)},
                )

            return _safe_json(resp)

    @staticmethod
    def _sleep_backoff(seconds: float) -> None:
        jitter = random.random() * 0.1
        time.sleep(max(0.0, float(seconds) + jitter))

    def _make_client_id(self, prefix: str) -> str:
        clean = "".join(ch for ch in str(prefix or "").lower() if ch.isalnum())[:8] or "ord"
        self._client_order_seq = int(getattr(self, "_client_order_seq", 0)) + 1
        token = f"{clean}_{_now_ms():x}_{self._client_order_seq:x}_{random.getrandbits(16):04x}"
        return token[:36]

    @staticmethod
    def _is_ambiguous_submit_error(error: TestnetAPIError) -> bool:
        code = _payload_code(getattr(error, "payload", None))
        if code in (-1006, -1007, -4116):
            return True
        try:
            if error.status_code is not None and int(error.status_code) >= 500:
                return True
        except Exception:
            pass
        msg = str(error).lower()
        if "network error after retries" in msg:
            return True
        if "execution status unknown" in msg:
            return True
        if "unknown error" in msg and "check your request" in msg:
            return True
        return False

    def _exchange_info(self) -> dict[str, Any]:
        ttl = max(1, int(getattr(self.cfg, "exchange_info_ttl_seconds", 900) or 900))
        now = time.time()
        if self._exchange_info_cache is not None and (now - float(self._exchange_info_cached_at)) < float(ttl):
            return self._exchange_info_cache

        info = self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
        if not isinstance(info, dict):
            raise TestnetAPIError("Invalid exchangeInfo payload", payload={"raw": str(info)})

        self._exchange_info_cache = info
        self._exchange_info_cached_at = now
        self._symbol_filters_cache = {}
        return info

    def _symbol_filters(self, symbol: str) -> _SymbolFilters:
        symbol_u = str(symbol).upper()
        cached = self._symbol_filters_cache.get(symbol_u)
        if cached is not None:
            return cached

        info = self._exchange_info()
        symbols = info.get("symbols")
        if not isinstance(symbols, list):
            raise TestnetAPIError("exchangeInfo missing symbols list", payload={"symbol": symbol_u})

        row: Optional[dict[str, Any]] = None
        for item in symbols:
            if isinstance(item, dict) and str(item.get("symbol", "")).upper() == symbol_u:
                row = item
                break
        if row is None:
            raise TestnetAPIError("Symbol not found in exchangeInfo", payload={"symbol": symbol_u})

        raw_filters = row.get("filters")
        if not isinstance(raw_filters, list):
            raise TestnetAPIError("exchangeInfo symbol missing filters", payload={"symbol": symbol_u})

        by_type: dict[str, dict[str, Any]] = {}
        for f in raw_filters:
            if isinstance(f, dict):
                ft = str(f.get("filterType", "")).upper()
                if ft:
                    by_type[ft] = f

        lot_raw = by_type.get("LOT_SIZE")
        if not isinstance(lot_raw, dict):
            raise TestnetAPIError("LOT_SIZE filter missing", payload={"symbol": symbol_u})
        mlot_raw = by_type.get("MARKET_LOT_SIZE")
        if not isinstance(mlot_raw, dict):
            mlot_raw = lot_raw
        price_raw = by_type.get("PRICE_FILTER")
        if not isinstance(price_raw, dict):
            raise TestnetAPIError("PRICE_FILTER filter missing", payload={"symbol": symbol_u})

        lot = _QtyFilter(
            min_qty=_safe_decimal(lot_raw.get("minQty"), Decimal("0")),
            max_qty=_safe_decimal(lot_raw.get("maxQty"), Decimal("0")),
            step_size=_safe_decimal(lot_raw.get("stepSize"), Decimal("0")),
            filter_type="LOT_SIZE",
        )
        mlot = _QtyFilter(
            min_qty=_safe_decimal(mlot_raw.get("minQty"), lot.min_qty),
            max_qty=_safe_decimal(mlot_raw.get("maxQty"), lot.max_qty),
            step_size=_safe_decimal(mlot_raw.get("stepSize"), lot.step_size),
            filter_type="MARKET_LOT_SIZE" if by_type.get("MARKET_LOT_SIZE") else "LOT_SIZE",
        )
        price_filter = _PriceFilter(
            min_price=_safe_decimal(price_raw.get("minPrice"), Decimal("0")),
            max_price=_safe_decimal(price_raw.get("maxPrice"), Decimal("0")),
            tick_size=_safe_decimal(price_raw.get("tickSize"), Decimal("0")),
        )

        min_notional: Optional[Decimal] = None
        min_notional_filter_type: Optional[str] = None
        min_notional_raw = by_type.get("MIN_NOTIONAL")
        if isinstance(min_notional_raw, dict):
            val = min_notional_raw.get("notional")
            if val is None:
                val = min_notional_raw.get("minNotional")
            if val is not None:
                notional_val = _safe_decimal(val, Decimal("0"))
                if notional_val > 0:
                    min_notional = notional_val
                    min_notional_filter_type = "MIN_NOTIONAL"
        if min_notional is None:
            notional_raw = by_type.get("NOTIONAL")
            if isinstance(notional_raw, dict):
                val = notional_raw.get("minNotional")
                if val is None:
                    val = notional_raw.get("notional")
                if val is not None:
                    notional_val = _safe_decimal(val, Decimal("0"))
                    if notional_val > 0:
                        min_notional = notional_val
                        min_notional_filter_type = "NOTIONAL"

        if lot.step_size <= 0 or mlot.step_size <= 0 or price_filter.tick_size <= 0:
            raise TestnetAPIError(
                "Invalid exchangeInfo filters",
                payload={
                    "symbol": symbol_u,
                    "lot_step": format_decimal(lot.step_size),
                    "market_lot_step": format_decimal(mlot.step_size),
                    "tick_size": format_decimal(price_filter.tick_size),
                },
            )

        parsed: _SymbolFilters = _SymbolFilters(
            symbol=symbol_u,
            lot_size=lot,
            market_lot_size=mlot,
            price_filter=price_filter,
            min_notional=min_notional,
            min_notional_filter_type=min_notional_filter_type,
        )
        self._symbol_filters_cache[symbol_u] = parsed
        return parsed

    def get_last_quantization(self) -> dict[str, Any]:
        return dict(self._last_quantization)

    def _last_price(self, symbol: str) -> Optional[Decimal]:
        try:
            out = self._request("GET", "/fapi/v1/ticker/price", params={"symbol": str(symbol).upper()}, signed=False)
        except Exception:
            return None
        if isinstance(out, dict):
            p = out.get("price")
            if p not in (None, ""):
                d = _safe_decimal(p, Decimal("0"))
                if d > 0:
                    return d
        return None

    def quantize_qty(
        self,
        *,
        symbol: str,
        qty: Any,
        is_market: bool,
        reference_price: Optional[Any] = None,
        enforce_min_notional: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        sf = self._symbol_filters(symbol)
        qf = sf.market_lot_size if bool(is_market) else sf.lot_size
        qty_raw = _to_decimal(qty)
        qty_sent = floor_to_step(qty_raw, qf.step_size)
        meta: dict[str, Any] = {
            "symbol": sf.symbol,
            "field": "quantity",
            "raw": format_decimal(qty_raw),
            "sent": format_decimal(qty_sent),
            "stepSize": format_decimal(qf.step_size),
            "minQty": format_decimal(qf.min_qty),
            "maxQty": format_decimal(qf.max_qty),
            "qtyFilterType": qf.filter_type,
            "isMarket": bool(is_market),
        }

        if qty_sent <= 0:
            raise OrderValidationError("quantity floored to 0 by stepSize", meta=meta)
        if qf.min_qty > 0 and qty_sent < qf.min_qty:
            raise OrderValidationError("quantity below minQty after quantization", meta=meta)
        if qf.max_qty > 0 and qty_sent > qf.max_qty:
            raise OrderValidationError("quantity above maxQty after quantization", meta=meta)

        if bool(enforce_min_notional) and sf.min_notional is not None:
            rp: Optional[Decimal] = None
            if reference_price is not None:
                rp = _to_decimal(reference_price)
                if rp <= 0:
                    rp = None
            if rp is not None:
                notional = qty_sent * rp
                meta["referencePrice"] = format_decimal(rp)
                meta["notional"] = format_decimal(notional)
                meta["minNotional"] = format_decimal(sf.min_notional)
                meta["minNotionalFilterType"] = str(sf.min_notional_filter_type or "")
                if notional < sf.min_notional:
                    raise OrderValidationError("order notional below minNotional after quantization", meta=meta)

        return format_decimal(qty_sent), meta

    def quantize_price(self, *, symbol: str, price: Any, field_name: str = "price") -> tuple[str, dict[str, Any]]:
        sf = self._symbol_filters(symbol)
        pf = sf.price_filter
        raw = _to_decimal(price)
        sent = floor_to_step(raw, pf.tick_size)
        meta: dict[str, Any] = {
            "symbol": sf.symbol,
            "field": str(field_name),
            "raw": format_decimal(raw),
            "sent": format_decimal(sent),
            "tickSize": format_decimal(pf.tick_size),
            "minPrice": format_decimal(pf.min_price),
            "maxPrice": format_decimal(pf.max_price),
        }
        if sent <= 0:
            raise OrderValidationError(f"{field_name} floored to 0 by tickSize", meta=meta)
        if pf.min_price > 0 and sent < pf.min_price:
            raise OrderValidationError(f"{field_name} below minPrice after quantization", meta=meta)
        if pf.max_price > 0 and sent > pf.max_price:
            raise OrderValidationError(f"{field_name} above maxPrice after quantization", meta=meta)
        return format_decimal(sent), meta

    # -------------------------
    # Public API
    # -------------------------
    def ping(self) -> Any:
        return self._request("GET", "/fapi/v1/ping", signed=False)

    def server_time(self) -> Any:
        return self._request("GET", "/fapi/v1/time", signed=False)

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": int(leverage)},
            signed=True,
        )

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        reference_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Any:
        ref_price: Optional[Any] = reference_price
        if ref_price is None and not bool(reduce_only):
            ref_price = self._last_price(symbol)
        qty_sent, qty_meta = self.quantize_qty(
            symbol=symbol,
            qty=quantity,
            is_market=True,
            reference_price=ref_price,
            enforce_min_notional=not bool(reduce_only),
        )
        self._last_quantization = {
            "symbol": str(symbol).upper(),
            "orderType": "MARKET",
            "fields": {"quantity": qty_meta},
        }
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": qty_sent,
            "newOrderRespType": "RESULT",
            "newClientOrderId": str(client_order_id or self._make_client_id("mkt")),
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        try:
            return self._request("POST", "/fapi/v1/order", params=params, signed=True)
        except TestnetAPIError as e:
            if classify_submit_error(e) != "ambiguous":
                raise
            lookup_error: Optional[str] = None
            try:
                recovered = self.get_order(symbol=symbol, client_order_id=str(params["newClientOrderId"]))
            except Exception as lookup_exc:
                lookup_error = str(lookup_exc)
            else:
                if isinstance(recovered, dict) and (
                    recovered.get("orderId") is not None
                    or recovered.get("clientOrderId") is not None
                    or recovered.get("origClientOrderId") is not None
                ):
                    return recovered
            raise AmbiguousOrderError(
                "Market order submission result is ambiguous",
                client_order_id=str(params["newClientOrderId"]),
                context={
                    "symbol": str(symbol).upper(),
                    "side": str(side).upper(),
                    "quantity": str(qty_sent),
                    "reduce_only": bool(reduce_only),
                    "submit_error": str(e),
                    "submit_status_code": getattr(e, "status_code", None),
                    "submit_payload": dict(getattr(e, "payload", {}) or {}),
                    "lookup_error": lookup_error,
                },
            ) from e
    # ---- Conditional (Algo) Orders ----

    def place_stop_market(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
        reference_price: Optional[float] = None,
        client_algo_id: Optional[str] = None,
    ) -> Any:
        # NOTE: Binance routes STOP_MARKET via algoOrder endpoint.
        # In our single-position runner, safest is Close-All for protective orders.
        stop_sent, stop_meta = self.quantize_price(symbol=symbol, price=stop_price, field_name="triggerPrice")
        fields: dict[str, Any] = {"triggerPrice": stop_meta}
        params: Dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "triggerPrice": stop_sent,
            "workingType": "MARK_PRICE",
            "newOrderRespType": "RESULT",
            "clientAlgoId": str(client_algo_id or self._make_client_id("sl")),
        }
        if reduce_only:
            params["closePosition"] = "true"
        else:
            qty_sent, qty_meta = self.quantize_qty(
                symbol=symbol,
                qty=quantity,
                is_market=False,
                reference_price=reference_price if reference_price is not None else stop_sent,
                enforce_min_notional=True,
            )
            fields["quantity"] = qty_meta
            params["quantity"] = qty_sent
            params["reduceOnly"] = "false"
        self._last_quantization = {
            "symbol": str(symbol).upper(),
            "orderType": "STOP_MARKET",
            "fields": fields,
        }
        return self._request("POST", "/fapi/v1/algoOrder", params=params, signed=True)

    def place_take_profit_market(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
        reference_price: Optional[float] = None,
        client_algo_id: Optional[str] = None,
    ) -> Any:
        stop_sent, stop_meta = self.quantize_price(symbol=symbol, price=stop_price, field_name="triggerPrice")
        fields: dict[str, Any] = {"triggerPrice": stop_meta}
        params: Dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "triggerPrice": stop_sent,
            "workingType": "MARK_PRICE",
            "newOrderRespType": "RESULT",
            "clientAlgoId": str(client_algo_id or self._make_client_id("tp")),
        }
        if reduce_only:
            params["closePosition"] = "true"
        else:
            qty_sent, qty_meta = self.quantize_qty(
                symbol=symbol,
                qty=quantity,
                is_market=False,
                reference_price=reference_price if reference_price is not None else stop_sent,
                enforce_min_notional=True,
            )
            fields["quantity"] = qty_meta
            params["quantity"] = qty_sent
            params["reduceOnly"] = "false"
        self._last_quantization = {
            "symbol": str(symbol).upper(),
            "orderType": "TAKE_PROFIT_MARKET",
            "fields": fields,
        }
        return self._request("POST", "/fapi/v1/algoOrder", params=params, signed=True)

    def get_algo_order(self, *, symbol: str, algo_id: int) -> Any:
        # Symbol is required by Binance for many algo order queries.
        return self._request(
            "GET",
            "/fapi/v1/algoOrder",
            params={"symbol": symbol, "algoId": int(algo_id)},
            signed=True,
        )

    def get_algo_open_orders(self, *, symbol: str) -> Any:
        return self._request(
            "GET",
            "/fapi/v1/algoOpenOrders",
            params={"symbol": symbol},
            signed=True,
        )

    def cancel_algo_order(self, *, algo_id: int, symbol: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {"algoId": int(algo_id)}
        # Some endpoints accept symbol optional; keep if supplied.
        if symbol:
            params["symbol"] = symbol
        return self._request("DELETE", "/fapi/v1/algoOrder", params=params, signed=True)

    def cancel_all_algo_open_orders(self, *, symbol: str) -> Any:
        return self._request(
            "DELETE",
            "/fapi/v1/algoOpenOrders",
            params={"symbol": symbol},
            signed=True,
        )

    # ---- Regular Orders ----

    def get_order(self, *, symbol: str, order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> Any:
        if order_id is None and not client_order_id:
            raise ValueError("get_order requires order_id or client_order_id")
        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = int(order_id)
        if client_order_id:
            params["origClientOrderId"] = str(client_order_id)
        return self._request(
            "GET",
            "/fapi/v1/order",
            params=params,
            signed=True,
        )

    def cancel_order(self, *, symbol: str, order_id: int) -> Any:
        return self._request(
            "DELETE",
            "/fapi/v1/order",
            params={"symbol": symbol, "orderId": int(order_id)},
            signed=True,
        )

    def cancel_all_open_orders(self, *, symbol: str) -> Any:
        # Cancel both regular and algo open orders.
        out: Dict[str, Any] = {"symbol": symbol, "regular": None, "algo": None}
        out["regular"] = self._request(
            "DELETE",
            "/fapi/v1/allOpenOrders",
            params={"symbol": symbol},
            signed=True,
        )
        try:
            out["algo"] = self.cancel_all_algo_open_orders(symbol=symbol)
        except Exception as e:
            out["algo"] = {"error": str(e)}
        return out


    def position_risk(self, *, symbol: str) -> Dict[str, Any]:
        """Return position info for `symbol`.

        Binance introduced `/fapi/v3/positionRisk` as a replacement for `/fapi/v2/positionRisk`.
        We try v3 first for forward-compatibility, then fallback to v2.
        """
        last_err: Optional[Exception] = None
        for path in ("/fapi/v3/positionRisk", "/fapi/v2/positionRisk"):
            try:
                rows = self._request("GET", path, params={"symbol": symbol}, signed=True)
            except Exception as e:
                last_err = e
                continue

            if isinstance(rows, dict) and "data" in rows:
                rows = rows["data"]

            if isinstance(rows, dict) and str(rows.get("symbol")) == symbol:
                return rows

            if isinstance(rows, list):
                for r in rows:
                    if str(r.get("symbol")) == symbol:
                        return r
                return {"symbol": symbol, "positionAmt": "0", "entryPrice": "0", "unRealizedProfit": "0"}

            return {"symbol": symbol, "positionAmt": "0", "entryPrice": "0", "unRealizedProfit": "0"}

        if last_err is not None:
            raise last_err
        return {"symbol": symbol, "positionAmt": "0", "entryPrice": "0", "unRealizedProfit": "0"}

    def account(self) -> Any:
        """Returns futures account snapshot (wallet/margin/maintenance)."""
        return self._request("GET", "/fapi/v2/account", signed=True)
