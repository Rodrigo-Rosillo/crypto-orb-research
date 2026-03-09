from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from strategy import build_rule_orb_ranges, build_signals_from_ruleset, generate_signals_from_rules, load_signal_rules_from_config


def multi_rule_cfg() -> dict:
    return {
        "adx": {"period": 14},
        "signals": {
            "rules": [
                {
                    "signal_type": "uptrend_reversion",
                    "signal": 1,
                    "trend": "uptrend",
                    "trigger": "close_below_orb_low",
                    "adx_threshold": 35,
                    "orb": {"start": "12:30", "end": "13:00", "cutoff": "13:00"},
                },
                {
                    "signal_type": "downtrend_reversion",
                    "signal": -2,
                    "trend": "downtrend",
                    "trigger": "close_above_orb_high",
                    "adx_threshold": 44,
                    "orb": {"start": "13:00", "end": "13:30", "cutoff": "13:30"},
                },
                {
                    "signal_type": "downtrend_breakdown",
                    "signal": -1,
                    "trend": "downtrend",
                    "trigger": "close_below_orb_low",
                    "adx_threshold": 43,
                    "orb": {"start": "13:30", "end": "14:00", "cutoff": "14:00"},
                },
                {
                    "signal_type": "uptrend_continuation",
                    "signal": 2,
                    "trend": "uptrend",
                    "trigger": "close_above_orb_high",
                    "adx_threshold": 29,
                    "orb": {"start": "14:00", "end": "14:30", "cutoff": "14:30"},
                },
            ]
        },
    }


def legacy_cfg() -> dict:
    return {
        "adx": {"period": 14, "threshold": 43},
        "orb": {"start": "13:30", "end": "14:00", "cutoff": "14:00"},
    }


def single_rule_signals_cfg() -> dict:
    return {
        "adx": {"period": 14},
        "signals": {
            "rules": [
                {
                    "signal_type": "uptrend_reversion",
                    "signal": 1,
                    "trend": "uptrend",
                    "trigger": "close_below_orb_low",
                    "adx_threshold": 35,
                    "orb": {"start": "12:30", "end": "13:00", "cutoff": "13:00"},
                }
            ]
        },
    }


def make_df(rows: list[dict]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(row["ts"], tz="UTC") for row in rows])
    payload = [{k: v for k, v in row.items() if k != "ts"} for row in rows]
    return pd.DataFrame(payload, index=idx)


def run_multi_rule_signals(df: pd.DataFrame, rules=None) -> pd.DataFrame:
    rules = rules or load_signal_rules_from_config(multi_rule_cfg())
    rule_orb_ranges = build_rule_orb_ranges(df, rules)
    return generate_signals_from_rules(df, rules, rule_orb_ranges)


def test_uptrend_reversion_uses_own_orb_and_threshold() -> None:
    df = make_df(
        [
            {"ts": "2024-01-10 12:30", "open": 105, "high": 110, "low": 100, "close": 105, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-10 13:00", "open": 106, "high": 108, "low": 101, "close": 106, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-10 13:30", "open": 100, "high": 101, "low": 98, "close": 99, "trend": "uptrend", "adx": 35},
            {"ts": "2024-01-10 14:00", "open": 104, "high": 109, "low": 103, "close": 109, "trend": "downtrend", "adx": 50},
            {"ts": "2024-01-10 14:30", "open": 90, "high": 92, "low": 88, "close": 89, "trend": "downtrend", "adx": 50},
        ]
    )

    out = run_multi_rule_signals(df)
    ts = pd.Timestamp("2024-01-10 13:30", tz="UTC")

    assert out.at[ts, "signal"] == 1
    assert out.at[ts, "signal_type"] == "uptrend_reversion"
    assert out.at[ts, "orb_high"] == pytest.approx(110.0)
    assert out.at[ts, "orb_low"] == pytest.approx(100.0)
    assert int((out["signal"] != 0).sum()) == 1


def test_downtrend_reversion_uses_own_orb_and_threshold() -> None:
    df = make_df(
        [
            {"ts": "2024-01-11 12:30", "open": 105, "high": 110, "low": 100, "close": 105, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-11 13:00", "open": 104, "high": 108, "low": 101, "close": 104, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-11 13:30", "open": 107, "high": 108, "low": 102, "close": 107, "trend": "downtrend", "adx": 50},
            {"ts": "2024-01-11 14:00", "open": 109, "high": 111, "low": 106, "close": 109, "trend": "downtrend", "adx": 44},
            {"ts": "2024-01-11 14:30", "open": 108, "high": 109, "low": 100, "close": 104, "trend": "downtrend", "adx": 50},
        ]
    )

    out = run_multi_rule_signals(df)
    ts = pd.Timestamp("2024-01-11 14:00", tz="UTC")

    assert out.at[ts, "signal"] == -2
    assert out.at[ts, "signal_type"] == "downtrend_reversion"
    assert out.at[ts, "orb_high"] == pytest.approx(108.0)
    assert out.at[ts, "orb_low"] == pytest.approx(101.0)
    assert int((out["signal"] != 0).sum()) == 1


def test_downtrend_breakdown_uses_own_orb_and_threshold() -> None:
    df = make_df(
        [
            {"ts": "2024-01-12 12:30", "open": 105, "high": 110, "low": 100, "close": 105, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-12 13:00", "open": 104, "high": 107, "low": 103, "close": 104, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-12 13:30", "open": 100, "high": 106, "low": 97, "close": 100, "trend": "downtrend", "adx": 50},
            {"ts": "2024-01-12 14:00", "open": 99, "high": 105, "low": 96, "close": 99, "trend": "downtrend", "adx": 50},
            {"ts": "2024-01-12 14:30", "open": 95, "high": 100, "low": 94, "close": 95, "trend": "downtrend", "adx": 43},
        ]
    )

    out = run_multi_rule_signals(df)
    ts = pd.Timestamp("2024-01-12 14:30", tz="UTC")

    assert out.at[ts, "signal"] == -1
    assert out.at[ts, "signal_type"] == "downtrend_breakdown"
    assert out.at[ts, "orb_high"] == pytest.approx(106.0)
    assert out.at[ts, "orb_low"] == pytest.approx(96.0)
    assert int((out["signal"] != 0).sum()) == 1


def test_uptrend_continuation_uses_own_orb_and_threshold() -> None:
    df = make_df(
        [
            {"ts": "2024-01-20 12:30", "open": 101, "high": 105, "low": 100, "close": 102, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-20 13:00", "open": 102, "high": 104, "low": 101, "close": 103, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-20 13:30", "open": 103, "high": 104, "low": 102, "close": 103, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-20 14:00", "open": 100, "high": 103, "low": 99, "close": 100, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-20 14:30", "open": 101, "high": 104, "low": 101, "close": 103, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-20 15:00", "open": 105, "high": 107, "low": 104, "close": 105, "trend": "uptrend", "adx": 29},
        ]
    )

    out = run_multi_rule_signals(df)
    ts = pd.Timestamp("2024-01-20 15:00", tz="UTC")

    assert out.at[ts, "signal"] == 2
    assert out.at[ts, "signal_type"] == "uptrend_continuation"
    assert out.at[ts, "orb_high"] == pytest.approx(104.0)
    assert out.at[ts, "orb_low"] == pytest.approx(99.0)
    assert int((out["signal"] != 0).sum()) == 1


def test_earliest_qualifying_bar_wins_across_rules() -> None:
    cfg = multi_rule_cfg()
    rules = list(reversed(load_signal_rules_from_config(cfg)))
    df = make_df(
        [
            {"ts": "2024-01-13 12:30", "open": 105, "high": 110, "low": 100, "close": 105, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-13 13:00", "open": 106, "high": 108, "low": 101, "close": 106, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-13 13:30", "open": 100, "high": 101, "low": 98, "close": 99, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-13 14:00", "open": 109, "high": 111, "low": 106, "close": 109, "trend": "downtrend", "adx": 50},
            {"ts": "2024-01-13 14:30", "open": 90, "high": 92, "low": 88, "close": 89, "trend": "downtrend", "adx": 50},
        ]
    )

    out = run_multi_rule_signals(df, rules=rules)
    first_ts = pd.Timestamp("2024-01-13 13:30", tz="UTC")
    later_ts = pd.Timestamp("2024-01-13 14:30", tz="UTC")

    assert out.at[first_ts, "signal"] == 1
    assert out.at[first_ts, "signal_type"] == "uptrend_reversion"
    assert out.at[later_ts, "signal"] == 0
    assert int((out["signal"] != 0).sum()) == 1


@pytest.mark.parametrize(
    ("rows", "label"),
    [
        pytest.param(
            [
                {"ts": "2024-01-14 12:30", "open": 105, "high": 110, "low": 100, "close": 105, "trend": "uptrend", "adx": 50},
                {"ts": "2024-01-14 13:00", "open": 104, "high": 107, "low": 103, "close": 104, "trend": "uptrend", "adx": 50},
                {"ts": "2024-01-14 13:30", "open": 100, "high": 106, "low": 97, "close": 100, "trend": "downtrend", "adx": 50},
                {"ts": "2024-01-14 14:00", "open": 99, "high": 105, "low": 96, "close": 99, "trend": "downtrend", "adx": 50},
                {"ts": "2024-01-14 14:30", "open": 95, "high": 100, "low": 94, "close": 95, "trend": "downtrend", "adx": 42.9},
            ],
            "adx_below_threshold",
            id="adx_below_threshold",
        ),
        pytest.param(
            [
                {"ts": "2024-01-15 12:30", "open": 105, "high": 110, "low": 100, "close": 105, "trend": "uptrend", "adx": 50},
                {"ts": "2024-01-15 13:00", "open": 104, "high": 107, "low": 103, "close": 104, "trend": "uptrend", "adx": 50},
                {"ts": "2024-01-15 13:30", "open": 100, "high": 106, "low": 97, "close": 100, "trend": "downtrend", "adx": 50},
                {"ts": "2024-01-15 14:00", "open": 99, "high": 105, "low": 96, "close": 99, "trend": "downtrend", "adx": 50},
                {"ts": "2024-01-15 14:30", "open": 95, "high": 100, "low": 94, "close": 95, "trend": "sideways", "adx": 50},
            ],
            "wrong_trend",
            id="wrong_trend",
        ),
        pytest.param(
            [
                {"ts": "2024-01-16 12:30", "open": 105, "high": 110, "low": 100, "close": 105, "trend": "uptrend", "adx": 50},
                {"ts": "2024-01-16 13:00", "open": 104, "high": 107, "low": 103, "close": 104, "trend": "uptrend", "adx": 50},
                {"ts": "2024-01-16 13:30", "open": 100, "high": 106, "low": 97, "close": 100, "trend": "downtrend", "adx": 50},
                {"ts": "2024-01-16 14:00", "open": 99, "high": 105, "low": 96, "close": 99, "trend": "downtrend", "adx": 50},
                {"ts": "2024-01-16 14:30", "open": 96, "high": 100, "low": 94, "close": 96, "trend": "downtrend", "adx": 50},
            ],
            "equal_orb_low",
            id="equal_orb_low",
        ),
        pytest.param(
            [
                {"ts": "2024-01-17 12:30", "open": 105, "high": 110, "low": 100, "close": 105, "trend": "uptrend", "adx": 50},
                {"ts": "2024-01-17 13:00", "open": 104, "high": 108, "low": 101, "close": 104, "trend": "uptrend", "adx": 50},
                {"ts": "2024-01-17 13:30", "open": 107, "high": 108, "low": 102, "close": 107, "trend": "downtrend", "adx": 50},
                {"ts": "2024-01-17 14:00", "open": 108, "high": 109, "low": 106, "close": 108, "trend": "downtrend", "adx": 50},
            ],
            "equal_orb_high",
            id="equal_orb_high",
        ),
        pytest.param(
            [
                {"ts": "2024-01-18 14:30", "open": 95, "high": 100, "low": 94, "close": 95, "trend": "downtrend", "adx": 50},
            ],
            "missing_orb_day",
            id="missing_orb_day",
        ),
    ],
)
def test_partial_conditions_do_not_fire(rows: list[dict], label: str) -> None:
    del label
    df = make_df(rows)
    out = run_multi_rule_signals(df)

    assert int((out["signal"] != 0).sum()) == 0
    assert (out["signal"] == 0).all()
    assert (out["signal_type"] == "").all()


def test_legacy_single_rule_fallback_still_works() -> None:
    rules = load_signal_rules_from_config(legacy_cfg())
    assert len(rules) == 1
    assert rules[0].signal_type == "downtrend_breakdown"
    assert rules[0].signal == -1

    df = make_df(
        [
            {"ts": "2024-01-19 13:30", "open": 100, "high": 106, "low": 97, "close": 100, "trend": "downtrend", "adx": 50},
            {"ts": "2024-01-19 14:00", "open": 99, "high": 105, "low": 96, "close": 99, "trend": "downtrend", "adx": 50},
            {"ts": "2024-01-19 14:30", "open": 95, "high": 100, "low": 94, "close": 95, "trend": "downtrend", "adx": 43},
        ]
    )

    rule_orb_ranges = build_rule_orb_ranges(df, rules)
    out = generate_signals_from_rules(df, rules, rule_orb_ranges)
    ts = pd.Timestamp("2024-01-19 14:30", tz="UTC")

    assert out.at[ts, "signal"] == -1
    assert out.at[ts, "signal_type"] == "downtrend_breakdown"


def test_build_signals_from_ruleset_supports_explicit_single_rule_variants() -> None:
    base_rule = replace(load_signal_rules_from_config(single_rule_signals_cfg())[0], adx_threshold=40)
    tuned_rule = replace(base_rule, adx_threshold=35)
    df = make_df(
        [
            {"ts": "2024-01-21 12:30", "open": 105, "high": 110, "low": 100, "close": 105, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-21 13:00", "open": 106, "high": 108, "low": 101, "close": 106, "trend": "uptrend", "adx": 50},
            {"ts": "2024-01-21 13:30", "open": 99, "high": 100, "low": 98, "close": 99, "trend": "uptrend", "adx": 37},
        ]
    )
    valid_days = {pd.Timestamp("2024-01-21", tz="UTC").date()}

    base_out, _, base_rules = build_signals_from_ruleset(df, [base_rule], valid_days)
    tuned_out, _, tuned_rules = build_signals_from_ruleset(df, [tuned_rule], valid_days)
    ts = pd.Timestamp("2024-01-21 13:30", tz="UTC")

    assert base_rules[0].signal_type == tuned_rules[0].signal_type == "uptrend_reversion"
    assert base_rules[0].signal == tuned_rules[0].signal == 1
    assert base_rules[0].trend == tuned_rules[0].trend == "uptrend"
    assert base_rules[0].trigger == tuned_rules[0].trigger == "close_below_orb_low"
    assert base_out.at[ts, "signal"] == 0
    assert tuned_out.at[ts, "signal"] == 1
    assert tuned_out.at[ts, "signal_type"] == "uptrend_reversion"
