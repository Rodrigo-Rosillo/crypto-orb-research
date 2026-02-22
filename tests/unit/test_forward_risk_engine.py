from __future__ import annotations

from forward.risk_engine import (
    RiskDecision,
    check_data_staleness,
    check_margin_ratio,
    check_order_rejects,
)


def test_check_data_staleness_uses_strict_greater_than() -> None:
    equal = check_data_staleness(since_seconds=120.0, allowed_seconds=120.0)
    greater = check_data_staleness(since_seconds=120.1, allowed_seconds=120.0)

    assert equal.decision == RiskDecision.ALLOW
    assert greater.decision == RiskDecision.KILL_SWITCH
    assert greater.reason == "KILL_SWITCH_DATA_STALE"


def test_check_order_rejects_uses_strict_greater_than() -> None:
    equal = check_order_rejects(rejects_today=3, max_rejects=3)
    greater = check_order_rejects(rejects_today=4, max_rejects=3)

    assert equal.decision == RiskDecision.ALLOW
    assert greater.decision == RiskDecision.KILL_SWITCH
    assert greater.reason == "KILL_SWITCH_ORDER_REJECTS"


def test_check_margin_ratio_triggers_on_greater_or_equal_threshold() -> None:
    equal = check_margin_ratio(maint=85.0, balance=100.0, threshold=0.85)
    below = check_margin_ratio(maint=84.9, balance=100.0, threshold=0.85)

    assert equal.decision == RiskDecision.KILL_SWITCH
    assert equal.reason == "KILL_SWITCH_MARGIN_RATIO"
    assert below.decision == RiskDecision.ALLOW
