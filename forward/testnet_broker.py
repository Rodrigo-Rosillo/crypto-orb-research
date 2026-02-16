from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


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
    base_url: str = "https://testnet.binancefuture.com"
    recv_window_ms: int = 5000
    rate_limit: RateLimitConfig = RateLimitConfig()


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
      - This is intentionally minimal for Phase 5 Step 4.
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
        params = dict(params or {})

        if signed:
            params["timestamp"] = _now_ms()
            params["recvWindow"] = int(self.cfg.recv_window_ms)

        # Deterministic query string for signature
        def build_query(p: Dict[str, Any]) -> str:
            items = []
            for k in sorted(p.keys()):
                v = p[k]
                if v is None:
                    continue
                items.append(f"{k}={v}")
            return "&".join(items)

        if signed:
            query = build_query(params)
            params["signature"] = _sign(self.api_secret, query)

        rl = self.cfg.rate_limit
        attempt = 0
        backoff = float(rl.base_backoff_seconds)

        while True:
            attempt += 1
            try:
                resp = self.session.request(method, url, params=params, timeout=timeout)
            except requests.RequestException as e:
                if attempt > rl.max_retries:
                    raise TestnetAPIError(f"Network error after retries: {e}") from e
                self._sleep_backoff(backoff)
                backoff = min(float(rl.max_backoff_seconds), backoff * 2.0)
                continue

            if resp.status_code in (418, 429):
                if attempt > rl.max_retries:
                    raise TestnetAPIError(
                        f"Rate limit after retries (HTTP {resp.status_code})",
                        status_code=resp.status_code,
                        payload=_safe_json(resp) if isinstance(_safe_json(resp), dict) else {"raw": str(_safe_json(resp))},
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
                raise TestnetAPIError(
                    f"Binance API error HTTP {resp.status_code}",
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

    def place_stop_market(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> Any:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": _fmt_price(stop_price),
            "quantity": _fmt_qty(quantity),
            "workingType": "MARK_PRICE",
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

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
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": _fmt_price(stop_price),
            "quantity": _fmt_qty(quantity),
            "workingType": "MARK_PRICE",
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

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
        return self._request(
            "DELETE",
            "/fapi/v1/allOpenOrders",
            params={"symbol": symbol},
            signed=True,
        )

    def position_risk(self, *, symbol: str) -> Dict[str, Any]:
        rows = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        if isinstance(rows, dict) and "data" in rows:
            rows = rows["data"]
        if not isinstance(rows, list):
            return {"symbol": symbol, "positionAmt": "0", "entryPrice": "0", "unRealizedProfit": "0"}
        for r in rows:
            if str(r.get("symbol")) == symbol:
                return r
        return {"symbol": symbol, "positionAmt": "0", "entryPrice": "0", "unRealizedProfit": "0"}

    def account(self) -> Any:
        """Returns futures account snapshot (wallet/margin/maintenance)."""
        return self._request("GET", "/fapi/v2/account", signed=True)
