from __future__ import annotations

import pytest

from strategy import load_signal_rules_from_config


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


def test_valid_multi_rule_config_parses() -> None:
    rules = load_signal_rules_from_config(multi_rule_cfg())

    assert len(rules) == 4
    assert rules[0].signal_type == "uptrend_reversion"
    assert rules[1].signal == -2
    assert rules[2].adx_threshold == 43
    assert rules[2].orb_cutoff.strftime("%H:%M") == "14:00"
    assert rules[3].signal_type == "uptrend_continuation"
    assert rules[3].signal == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("trend", "sideways", id="invalid_trend"),
        pytest.param("trigger", "unsupported", id="invalid_trigger"),
    ],
)
def test_invalid_trend_or_trigger_fails(field: str, value: str) -> None:
    cfg = multi_rule_cfg()
    cfg["signals"]["rules"][0][field] = value

    with pytest.raises(ValueError):
        load_signal_rules_from_config(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("signal_type", "uptrend_reversion", id="duplicate_signal_type"),
        pytest.param("signal", 1, id="duplicate_signal"),
    ],
)
def test_duplicate_signal_type_or_signal_fails(field: str, value) -> None:
    cfg = multi_rule_cfg()
    cfg["signals"]["rules"][1][field] = value

    with pytest.raises(ValueError):
        load_signal_rules_from_config(cfg)


def test_legacy_fallback_config_parses() -> None:
    rules = load_signal_rules_from_config(legacy_cfg())

    assert len(rules) == 1
    assert rules[0].signal_type == "downtrend_breakdown"
    assert rules[0].signal == -1
    assert rules[0].orb_start.strftime("%H:%M") == "13:30"


def test_explicit_single_rule_signals_config_parses() -> None:
    rules = load_signal_rules_from_config(single_rule_signals_cfg())

    assert len(rules) == 1
    assert rules[0].signal_type == "uptrend_reversion"
    assert rules[0].signal == 1
    assert rules[0].trend == "uptrend"
    assert rules[0].trigger == "close_below_orb_low"
    assert rules[0].adx_threshold == 35
    assert rules[0].orb_start.strftime("%H:%M") == "12:30"


def test_signals_rules_take_precedence_over_conflicting_legacy_fields() -> None:
    cfg = single_rule_signals_cfg()
    cfg["adx"]["threshold"] = 99
    cfg["orb"] = {"start": "23:00", "end": "23:30", "cutoff": "23:30"}

    rules = load_signal_rules_from_config(cfg)

    assert len(rules) == 1
    assert rules[0].signal_type == "uptrend_reversion"
    assert rules[0].adx_threshold == 35
    assert rules[0].orb_start.strftime("%H:%M") == "12:30"
    assert rules[0].orb_end.strftime("%H:%M") == "13:00"
