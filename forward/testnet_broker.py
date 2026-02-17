from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests
from urllib.parse import urlencode


class TestnetAuthError(RuntimeError):
    """Raised when API key/secret env vars are missing."""


class TestnetAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


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


def _fmt_qty(q: float) -> str:
    return f"{float(q):.8f}".rstrip("0").rstrip(".")


def _fmt_price(p: float) -> str:
    return f"{float(p):.8f}".rstrip("0").rstrip(".")


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
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

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

    def place_market_order(self, *, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> Any:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": _fmt_qty(quantity),
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    # ---- Conditional (Algo) Orders ----

    def place_stop_market(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> Any:
        # NOTE: Binance routes STOP_MARKET via algoOrder endpoint.
        # In our single-position runner, safest is Close-All for protective orders.
        params: Dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "triggerPrice": _fmt_price(stop_price),
            "workingType": "MARK_PRICE",
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["closePosition"] = "true"
        else:
            params["quantity"] = _fmt_qty(quantity)
            params["reduceOnly"] = "false"
        return self._request("POST", "/fapi/v1/algoOrder", params=params, signed=True)

    def place_take_profit_market(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> Any:
        params: Dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "triggerPrice": _fmt_price(stop_price),
            "workingType": "MARK_PRICE",
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["closePosition"] = "true"
        else:
            params["quantity"] = _fmt_qty(quantity)
            params["reduceOnly"] = "false"
        return self._request("POST", "/fapi/v1/algoOrder", params=params, signed=True)

    def get_algo_order(self, *, symbol: str, algo_id: int) -> Any:
        # Symbol is required by Binance for many algo order queries.
        return self._request(
            "GET",
            "/fapi/v1/algoOrder",
            params={"symbol": symbol, "algoId": int(algo_id)},
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

    def get_order(self, *, symbol: str, order_id: int) -> Any:
        return self._request(
            "GET",
            "/fapi/v1/order",
            params={"symbol": symbol, "orderId": int(order_id)},
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
