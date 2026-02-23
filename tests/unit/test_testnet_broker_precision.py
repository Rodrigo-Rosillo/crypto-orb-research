from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from forward.testnet_broker import (
    BinanceFuturesTestnetBroker,
    OrderValidationError,
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
    return broker


def test_floor_to_step_and_format_decimal() -> None:
    floored = floor_to_step("24.087679152113694", "0.001")
    assert floored == Decimal("24.087")
    assert format_decimal(floored) == "24.087"
    assert format_decimal(Decimal("1E-8")) == "0.00000001"


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
