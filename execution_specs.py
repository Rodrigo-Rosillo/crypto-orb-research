from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["long", "short"]
TargetKind = Literal["orb_high", "orb_low", "entry_pct"]
StopKind = Literal["orb_high", "orb_low", "symmetric_to_target"]


@dataclass(frozen=True)
class ExecutionSpec:
    side: Side
    target_kind: TargetKind
    target_pct: float | None = None
    stop_kind: StopKind = "symmetric_to_target"

    def __post_init__(self) -> None:
        if self.target_kind == "entry_pct":
            if self.target_pct is None or float(self.target_pct) <= 0:
                raise ValueError("entry_pct execution specs require a positive target_pct")
        elif self.target_pct is not None:
            raise ValueError("target_pct is supported only when target_kind='entry_pct'")


@dataclass(frozen=True)
class ResolvedExecutionPlan:
    side: Side
    target_price: float
    stop_loss: float


_EXECUTION_SPECS: dict[str, ExecutionSpec] = {
    "uptrend_reversion": ExecutionSpec(
        side="long",
        target_kind="orb_high",
        stop_kind="symmetric_to_target",
    ),
    "uptrend_continuation": ExecutionSpec(
        side="long",
        target_kind="entry_pct",
        target_pct=0.02,
        stop_kind="orb_low",
    ),
    "downtrend_breakdown": ExecutionSpec(
        side="short",
        target_kind="entry_pct",
        target_pct=0.02,
        stop_kind="orb_high",
    ),
    "downtrend_reversion": ExecutionSpec(
        side="short",
        target_kind="orb_low",
        stop_kind="symmetric_to_target",
    ),
}


def get_execution_spec(signal_type: str) -> ExecutionSpec:
    try:
        return _EXECUTION_SPECS[str(signal_type)]
    except KeyError as exc:
        raise ValueError(f"Unsupported signal_type for execution: {signal_type!r}") from exc


def required_orb_fields(signal_type: str) -> tuple[str, ...]:
    spec = get_execution_spec(signal_type)
    fields: list[str] = []
    if spec.target_kind == "orb_high":
        fields.append("orb_high")
    elif spec.target_kind == "orb_low":
        fields.append("orb_low")

    if spec.stop_kind == "orb_high":
        fields.append("orb_high")
    elif spec.stop_kind == "orb_low":
        fields.append("orb_low")

    return tuple(dict.fromkeys(fields))


def resolve_execution_plan(
    *,
    signal_type: str,
    entry_price: float,
    orb_high: float,
    orb_low: float,
) -> ResolvedExecutionPlan:
    spec = get_execution_spec(signal_type)
    entry = float(entry_price)
    high = float(orb_high)
    low = float(orb_low)

    if spec.target_kind == "orb_high":
        target_price = high
    elif spec.target_kind == "orb_low":
        target_price = low
    else:
        assert spec.target_pct is not None
        if spec.side == "long":
            target_price = entry * (1.0 + float(spec.target_pct))
        else:
            target_price = entry * (1.0 - float(spec.target_pct))

    if spec.stop_kind == "orb_high":
        stop_loss = high
    elif spec.stop_kind == "orb_low":
        stop_loss = low
    else:
        if spec.side == "long":
            distance = target_price - entry
            stop_loss = entry - distance
        else:
            distance = entry - target_price
            stop_loss = entry + distance

    return ResolvedExecutionPlan(
        side=spec.side,
        target_price=float(target_price),
        stop_loss=float(stop_loss),
    )
