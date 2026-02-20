from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskDecision(Enum):
    ALLOW = "allow"
    KILL_SWITCH = "kill_switch"


@dataclass
class RiskCheckResult:
    decision: RiskDecision
    reason: Optional[str] = None


def check_margin_ratio(maint: float, balance: float, threshold: float) -> RiskCheckResult:
    """Reproduce live_testnet margin-ratio kill-switch behavior."""
    try:
        maint_f = float(maint)
        balance_f = float(balance)
        threshold_f = float(threshold)
    except Exception:
        return RiskCheckResult(decision=RiskDecision.ALLOW)
    if balance_f <= 0:
        return RiskCheckResult(decision=RiskDecision.ALLOW)
    ratio = maint_f / balance_f
    if ratio >= threshold_f:
        return RiskCheckResult(decision=RiskDecision.KILL_SWITCH, reason="KILL_SWITCH_MARGIN_RATIO")
    return RiskCheckResult(decision=RiskDecision.ALLOW)


def check_data_staleness(since_seconds: float, allowed_seconds: float) -> RiskCheckResult:
    """Data stale if elapsed seconds strictly exceeds allowed seconds."""
    try:
        since_f = float(since_seconds)
        allowed_f = float(allowed_seconds)
    except Exception:
        return RiskCheckResult(decision=RiskDecision.ALLOW)
    if since_f > allowed_f:
        return RiskCheckResult(decision=RiskDecision.KILL_SWITCH, reason="KILL_SWITCH_DATA_STALE")
    return RiskCheckResult(decision=RiskDecision.ALLOW)


def check_order_rejects(rejects_today: int, max_rejects: int) -> RiskCheckResult:
    """Order-reject kill-switch uses strict > to preserve current behavior."""
    try:
        rejects_i = int(rejects_today)
        max_i = int(max_rejects)
    except Exception:
        return RiskCheckResult(decision=RiskDecision.ALLOW)
    if rejects_i > max_i:
        return RiskCheckResult(decision=RiskDecision.KILL_SWITCH, reason="KILL_SWITCH_ORDER_REJECTS")
    return RiskCheckResult(decision=RiskDecision.ALLOW)

