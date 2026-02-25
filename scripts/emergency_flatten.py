#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, NoReturn
from urllib.parse import urlencode

import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from forward.state_store_sqlite import SQLiteStateStore


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _format_decimal(value: Decimal) -> str:
    s = format(value, "f")
    return s.rstrip("0").rstrip(".") or "0"


def _as_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [x for x in payload["data"] if isinstance(x, dict)]
    return []


def _sign(secret: str, params: Dict[str, Any]) -> str:
    pairs: List[tuple[str, str]] = []
    for k, v in params.items():
        if v is None:
            continue
        pairs.append((str(k), str(v)))
    query = urlencode(pairs, doseq=True)
    return hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()


class BinanceClient:
    def __init__(self, *, base_url: str, api_key: str, api_secret: str, recv_window_ms: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window_ms = int(recv_window_ms)
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def signed_request(self, method: str, path: str, params: Dict[str, Any] | None = None) -> Any:
        req_params = dict(params or {})
        req_params["timestamp"] = _now_ms()
        req_params["recvWindow"] = self.recv_window_ms
        req_params["signature"] = _sign(self.api_secret, req_params)

        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method=method, url=url, params=req_params, timeout=20)
        except requests.RequestException as exc:
            _fail(f"{method} {path} request failed: {exc}")

        if response.status_code >= 400:
            _fail(f"{method} {path} failed HTTP {response.status_code}: {response.text}")

        try:
            return response.json()
        except ValueError as exc:
            _fail(f"{method} {path} returned non-JSON response: {exc}")

    def get_positions(self) -> List[Dict[str, Any]]:
        try:
            payload = self.signed_request("GET", "/fapi/v2/positionRisk")
            return _as_list(payload)
        except Exception:
            # Fallback only for version/API incompatibility scenarios.
            pass

        payload = self.signed_request("GET", "/fapi/v3/positionRisk")
        return _as_list(payload)

    def cancel_open_orders_best_effort(self, symbol: str) -> None:
        try:
            self.signed_request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol})
        except Exception as exc:
            print(f"WARNING: cancel open orders failed for {symbol}: {exc}")

    def close_position(self, row: Dict[str, Any]) -> None:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            _fail("position row missing symbol")

        try:
            amount = Decimal(str(row.get("positionAmt", "0")))
        except Exception as exc:
            _fail(f"invalid positionAmt for {symbol}: {exc}")

        if amount == 0:
            return

        side = "SELL" if amount > 0 else "BUY"
        direction = "LONG" if amount > 0 else "SHORT"
        qty = _format_decimal(abs(amount))

        self.cancel_open_orders_best_effort(symbol)

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": qty,
            "reduceOnly": "true",
            "newOrderRespType": "RESULT",
        }
        position_side = row.get("positionSide")
        if isinstance(position_side, str) and position_side and position_side != "BOTH":
            params["positionSide"] = position_side

        response = self.signed_request("POST", "/fapi/v1/order", params=params)
        if not isinstance(response, dict):
            _fail(f"close order for {symbol} returned invalid response: {response}")
        order_id = response.get("orderId")
        if order_id is None:
            _fail(f"close order for {symbol} missing orderId: {response}")
        print(f"CLOSED {symbol} {direction} qty={qty} orderId={order_id}")


def _get_credentials(use_testnet: bool) -> tuple[str, str]:
    if use_testnet:
        key = (os.getenv("BINANCE_TESTNET_API_KEY") or "").strip()
        secret = (os.getenv("BINANCE_TESTNET_API_SECRET") or "").strip()
        if not key or not secret:
            _fail("missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_API_SECRET")
        return key, secret

    key = (os.getenv("BINANCE_API_KEY") or "").strip()
    secret = (os.getenv("BINANCE_API_SECRET") or "").strip()
    if not key or not secret:
        _fail("missing BINANCE_API_KEY or BINANCE_API_SECRET")
    return key, secret


def _select_base_url(use_testnet: bool) -> str:
    override = (os.getenv("BINANCE_FAPI_BASE_URL") or "").strip()
    if override and not override.lower().startswith("https://"): 
        _fail("BINANCE_FAPI_BASE_URL must start with https://")
    if override:
        return override
    if use_testnet:
        return "https://demo-fapi.binance.com"
    return "https://fapi.binance.com"


def _get_recv_window_ms() -> int:
    raw = (os.getenv("BINANCE_RECV_WINDOW_MS") or "5000").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        _fail(f"invalid BINANCE_RECV_WINDOW_MS: {exc}")
    if value <= 0:
        _fail("BINANCE_RECV_WINDOW_MS must be > 0")
    return value


def _open_positions(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            amt = float(row.get("positionAmt", "0"))
        except Exception as exc:
            _fail(f"invalid positionAmt in row {row}: {exc}")
        if abs(amt) > 0:
            out.append(row)
    return out


def _state_db_path() -> Path:
    raw = (os.getenv("STATE_DB_PATH") or "").strip()
    if raw:
        return Path(raw)
    return Path("/data/state.db")


def _clear_open_position_state(db_path: Path) -> None:
    with SQLiteStateStore(db_path=db_path) as store:
        state = store.load_state()
        state.open_position = None
        store.save_state(state)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emergency flatten for Binance USD-M Futures (close all open positions at market)."
    )
    parser.add_argument("--testnet", action="store_true", help="Use Binance USD-M demo/testnet.")
    parser.add_argument(
        "--clear-state",
        dest="clear_state",
        action="store_true",
        help="Clear persisted SQLite open_position state after exchange is confirmed flat.",
    )
    parser.add_argument(
        "--no-clear-state",
        dest="clear_state",
        action="store_false",
        help="Do not clear persisted SQLite open_position state.",
    )
    parser.set_defaults(clear_state=True)
    args = parser.parse_args()

    api_key, api_secret = _get_credentials(args.testnet)
    client = BinanceClient(
        base_url=_select_base_url(args.testnet),
        api_key=api_key,
        api_secret=api_secret,
        recv_window_ms=_get_recv_window_ms(),
    )

    open_rows = _open_positions(client.get_positions())

    if not open_rows:
        print("No open positions.")
    else:
        for row in list(open_rows):
            client.close_position(row)

    final_open_rows = _open_positions(client.get_positions())
    if final_open_rows:
        print("STATE CLEAR SKIPPED (exchange not flat)")
        print("WARNING: Exchange still reports open positions; refusing to clear persisted state.")
        return 1

    if not args.clear_state:
        print("STATE CLEAR SKIPPED (--no-clear-state)")
        return 0

    db_path = _state_db_path()
    try:
        _clear_open_position_state(db_path)
    except Exception as exc:
        print(f"STATE CLEAR FAILED in {db_path}: {exc}")
    else:
        print(f"STATE CLEARED in {db_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
