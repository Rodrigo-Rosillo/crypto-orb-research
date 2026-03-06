from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from forward.testnet_broker import (
    AmbiguousOrderError,
    BinanceFuturesTestnetBroker,
    OrderValidationError,
    TestnetAPIError,
    classify_submit_error,
    floor_to_step,
    format_decimal,
)


def _stub_broker(
    *,
    lot_step: str = "0.01",
    market_step: str = "0.01",
    tick_size: str = "0.001",
    min_qty: str = "0.001",
    min_notional: str | None = "5",
) -> BinanceFuturesTestnetBroker:
    broker = object.__new__(BinanceFuturesTestnetBroker)
    lot = SimpleNamespace(
        min_qty=Decimal(min_qty),
        max_qty=Decimal("1000000"),
        step_size=Decimal(lot_step),
        filter_type="LOT_SIZE",
    )
    market = SimpleNamespace(
        min_qty=Decimal(min_qty),
        max_qty=Decimal("1000000"),
        step_size=Decimal(market_step),
        filter_type="MARKET_LOT_SIZE",
    )
    price = SimpleNamespace(
        min_price=Decimal("0.0001"),
        max_price=Decimal("1000000"),
        tick_size=Decimal(tick_size),
    )
    broker._symbol_filters = lambda _symbol: SimpleNamespace(  # type: ignore[attr-defined]
        symbol="SOLUSDT",
        lot_size=lot,
        market_lot_size=market,
        price_filter=price,
        min_notional=Decimal(min_notional) if min_notional is not None else None,
        min_notional_filter_type="MIN_NOTIONAL" if min_notional is not None else None,
    )
    broker._last_quantization = {}
    broker._client_order_seq = 0
    return broker


def test_floor_to_step_and_format_decimal() -> None:
    floored = floor_to_step("24.087679152113694", "0.001")
    assert floored == Decimal("24.087")
    assert format_decimal(floored) == "24.087"
    assert format_decimal(Decimal("1E-8")) == "0.00000001"


def test_classify_submit_error_unknown_400_is_ambiguous() -> None:
    err = TestnetAPIError(
        "Binance API error HTTP 400 code=-9999 msg=unexpected submit failure",
        status_code=400,
        payload={"code": -9999, "msg": "unexpected submit failure"},
    )
    assert classify_submit_error(err) == "ambiguous"


def test_classify_submit_error_new_order_rejected_is_definitive_reject() -> None:
    err = TestnetAPIError(
        "Binance API error HTTP 400 code=-2010 msg=NEW_ORDER_REJECTED",
        status_code=400,
        payload={"code": -2010, "msg": "NEW_ORDER_REJECTED"},
    )
    assert classify_submit_error(err) == "definitive_reject"


def test_classify_submit_error_rate_limit_is_transient_system() -> None:
    err = TestnetAPIError(
        "Binance API error HTTP 429 code=-1003 msg=Too many requests",
        status_code=429,
        payload={"code": -1003, "msg": "Too many requests"},
    )
    assert classify_submit_error(err) == "transient_system"


def test_classify_submit_error_http_500_is_ambiguous() -> None:
    err = TestnetAPIError("Binance API error HTTP 500", status_code=500, payload={})
    assert classify_submit_error(err) == "ambiguous"


def test_quantize_qty_uses_market_lot_size_for_market_orders() -> None:
    broker = _stub_broker(lot_step="0.01", market_step="0.1")
    sent, meta = broker.quantize_qty(symbol="SOLUSDT", qty="24.087679152113694", is_market=True, reference_price="150")
    assert sent == "24"
    assert meta["stepSize"] == "0.1"
    assert meta["qtyFilterType"] == "MARKET_LOT_SIZE"


def test_quantize_qty_uses_lot_size_for_non_market_orders() -> None:
    broker = _stub_broker(lot_step="0.01", market_step="0.1")
    sent, meta = broker.quantize_qty(symbol="SOLUSDT", qty="24.087679152113694", is_market=False, reference_price="150")
    assert sent == "24.08"
    assert meta["stepSize"] == "0.01"
    assert meta["qtyFilterType"] == "LOT_SIZE"


def test_regression_high_precision_qty_becomes_valid_step_multiple() -> None:
    broker = _stub_broker(lot_step="0.001", market_step="0.001", min_notional=None)
    sent, _ = broker.quantize_qty(symbol="SOLUSDT", qty="24.087679152113694", is_market=True, reference_price="100")
    sent_d = Decimal(sent)
    step = Decimal("0.001")
    assert sent_d == Decimal("24.087")
    assert (sent_d / step).to_integral_value() == (sent_d / step)


def test_quantize_price_uses_tick_size() -> None:
    broker = _stub_broker(tick_size="0.01")
    sent, meta = broker.quantize_price(symbol="SOLUSDT", price="154.189")
    assert sent == "154.18"
    assert meta["tickSize"] == "0.01"


def test_quantize_qty_rejects_when_below_min_notional() -> None:
    broker = _stub_broker(lot_step="0.001", market_step="0.001", min_notional="5")
    with pytest.raises(OrderValidationError, match="minNotional"):
        broker.quantize_qty(symbol="SOLUSDT", qty="24.087679152113694", is_market=True, reference_price="0.1")


def test_quantize_qty_rejects_when_floored_to_zero() -> None:
    broker = _stub_broker(lot_step="0.001", market_step="0.001", min_notional=None)
    with pytest.raises(OrderValidationError, match="floored to 0"):
        broker.quantize_qty(symbol="SOLUSDT", qty="0.0004", is_market=True, reference_price="100")


def test_exchange_info_is_cached_in_memory() -> None:
    broker = object.__new__(BinanceFuturesTestnetBroker)
    broker.cfg = SimpleNamespace(exchange_info_ttl_seconds=900)
    broker._exchange_info_cache = None
    broker._exchange_info_cached_at = 0.0
    broker._symbol_filters_cache = {}
    broker._last_quantization = {}

    calls = {"n": 0}

    def _fake_request(method: str, path: str, *, params=None, signed=False, timeout: float = 10.0):
        calls["n"] += 1
        assert method == "GET"
        assert path == "/fapi/v1/exchangeInfo"
        return {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "minPrice": "0.001", "maxPrice": "1000000", "tickSize": "0.001"},
                        {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000000", "stepSize": "0.001"},
                        {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "maxQty": "1000000", "stepSize": "0.001"},
                    ],
                }
            ]
        }

    broker._request = _fake_request  # type: ignore[method-assign]
    broker._exchange_info()
    broker._exchange_info()
    assert calls["n"] == 1

def test_get_algo_open_orders_wires_expected_endpoint() -> None:
    broker = object.__new__(BinanceFuturesTestnetBroker)

    observed: dict[str, Any] = {}

    def _fake_request(method: str, path: str, *, params=None, signed=False, timeout: float = 10.0):
        observed["method"] = method
        observed["path"] = path
        observed["params"] = params
        observed["signed"] = signed
        return [{"algoId": 123, "symbol": "SOLUSDT", "status": "NEW"}]

    broker._request = _fake_request  # type: ignore[method-assign]

    out = broker.get_algo_open_orders(symbol="SOLUSDT")
    assert isinstance(out, list)
    assert observed["method"] == "GET"
    assert observed["path"] == "/fapi/v1/algoOpenOrders"
    assert observed["params"] == {"symbol": "SOLUSDT"}
    assert observed["signed"] is True


def test_place_market_order_sends_new_client_order_id() -> None:
    broker = _stub_broker(min_notional=None)

    observed: dict[str, Any] = {}

    def _fake_request(method: str, path: str, *, params=None, signed=False, timeout: float = 10.0):
        observed["method"] = method
        observed["path"] = path
        observed["params"] = dict(params or {})
        observed["signed"] = signed
        return {"orderId": 321, "status": "FILLED", "avgPrice": "100", "executedQty": "1"}

    broker._request = _fake_request  # type: ignore[method-assign]

    out = broker.place_market_order(symbol="SOLUSDT", side="BUY", quantity=1.0, reference_price=100.0)
    assert out["orderId"] == 321
    assert observed["method"] == "POST"
    assert observed["path"] == "/fapi/v1/order"
    assert observed["signed"] is True
    assert observed["params"]["newOrderRespType"] == "RESULT"
    assert isinstance(observed["params"]["newClientOrderId"], str)
    assert 1 <= len(observed["params"]["newClientOrderId"]) <= 36


def test_place_algo_orders_send_client_algo_ids() -> None:
    broker = _stub_broker(min_notional=None)
    calls: list[dict[str, Any]] = []

    def _fake_request(method: str, path: str, *, params=None, signed=False, timeout: float = 10.0):
        calls.append(
            {
                "method": method,
                "path": path,
                "params": dict(params or {}),
                "signed": signed,
            }
        )
        return {"algoId": 123, "status": "NEW"}

    broker._request = _fake_request  # type: ignore[method-assign]

    broker.place_take_profit_market(symbol="SOLUSDT", side="SELL", quantity=1.0, stop_price=101.0)
    broker.place_stop_market(symbol="SOLUSDT", side="SELL", quantity=1.0, stop_price=99.0)

    assert len(calls) == 2
    assert all(call["path"] == "/fapi/v1/algoOrder" for call in calls)
    assert all(call["signed"] is True for call in calls)
    assert all(isinstance(call["params"].get("clientAlgoId"), str) for call in calls)
    assert all(1 <= len(str(call["params"]["clientAlgoId"])) <= 36 for call in calls)


def test_get_order_accepts_orig_client_order_id() -> None:
    broker = object.__new__(BinanceFuturesTestnetBroker)

    observed: dict[str, Any] = {}

    def _fake_request(method: str, path: str, *, params=None, signed=False, timeout: float = 10.0):
        observed["method"] = method
        observed["path"] = path
        observed["params"] = dict(params or {})
        observed["signed"] = signed
        return {"orderId": 4321, "status": "FILLED"}

    broker._request = _fake_request  # type: ignore[method-assign]

    out = broker.get_order(symbol="SOLUSDT", client_order_id="cid-1")
    assert out["orderId"] == 4321
    assert observed["method"] == "GET"
    assert observed["path"] == "/fapi/v1/order"
    assert observed["params"] == {"symbol": "SOLUSDT", "origClientOrderId": "cid-1"}
    assert observed["signed"] is True


def test_place_market_order_recovers_duplicate_client_order_id_via_lookup() -> None:
    broker = _stub_broker(min_notional=None)
    calls: list[dict[str, Any]] = []

    def _fake_request(method: str, path: str, *, params=None, signed=False, timeout: float = 10.0):
        calls.append({"method": method, "path": path, "params": dict(params or {}), "signed": signed})
        if method == "POST":
            raise TestnetAPIError(
                "Binance API error HTTP 400 code=-4116 msg=client order id is duplicated",
                status_code=400,
                payload={"code": -4116, "msg": "client order id is duplicated"},
            )
        return {"orderId": 777, "status": "FILLED", "avgPrice": "100", "executedQty": "1"}

    broker._request = _fake_request  # type: ignore[method-assign]

    out = broker.place_market_order(
        symbol="SOLUSDT",
        side="BUY",
        quantity=1.0,
        reference_price=100.0,
        client_order_id="cid-lookup",
    )
    assert out["orderId"] == 777
    assert calls[0]["params"]["newClientOrderId"] == "cid-lookup"
    assert calls[1]["params"]["origClientOrderId"] == "cid-lookup"


def test_place_market_order_raises_ambiguous_when_lookup_fails() -> None:
    broker = _stub_broker(min_notional=None)
    calls: list[dict[str, Any]] = []

    def _fake_request(method: str, path: str, *, params=None, signed=False, timeout: float = 10.0):
        calls.append({"method": method, "path": path, "params": dict(params or {}), "signed": signed})
        if method == "POST":
            raise TestnetAPIError("Network error after retries: timeout")
        raise TestnetAPIError(
            "Binance API error HTTP 400 code=-2013 msg=Order does not exist.",
            status_code=400,
            payload={"code": -2013, "msg": "Order does not exist."},
        )

    broker._request = _fake_request  # type: ignore[method-assign]

    with pytest.raises(AmbiguousOrderError) as excinfo:
        broker.place_market_order(
            symbol="SOLUSDT",
            side="BUY",
            quantity=1.0,
            reference_price=100.0,
            client_order_id="cid-missing",
        )

    err = excinfo.value
    assert err.client_order_id == "cid-missing"
    assert err.context["lookup_error"] != ""
    assert calls[0]["params"]["newClientOrderId"] == "cid-missing"
    assert calls[1]["params"]["origClientOrderId"] == "cid-missing"
