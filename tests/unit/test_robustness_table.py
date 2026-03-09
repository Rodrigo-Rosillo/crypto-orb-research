from __future__ import annotations

from scripts.robustness_table import build_robustness_scenarios, multi_rule_orb_start_offsets_for_timeframe
from strategy import load_signal_rules_from_config, serialize_signal_rules


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


def test_multi_rule_orb_start_offsets_for_30m_use_bar_aligned_steps() -> None:
    assert multi_rule_orb_start_offsets_for_timeframe("30m") == [-30, 0, 30]
    assert multi_rule_orb_start_offsets_for_timeframe("1h") == [-15, 0, 15]


def test_multi_rule_robustness_scenarios_perturb_one_rule_at_a_time() -> None:
    cfg = multi_rule_cfg()
    rules = load_signal_rules_from_config(cfg)
    base_rules = serialize_signal_rules(rules)

    scenarios = build_robustness_scenarios(
        cfg,
        rules,
        adx_threshold_grid=[35.0],
        orb_start_grid=["13:30"],
        orb_window_min=30,
        cutoff_offset_min=0,
        multi_rule_orb_start_offsets_min=multi_rule_orb_start_offsets_for_timeframe("30m"),
    )

    assert len(scenarios) == 33
    assert scenarios[0].scenario_id == "baseline"
    assert scenarios[0].perturbed_rule_signal_type is None
    assert scenarios[0].strategy_rules == base_rules
    scenario_ids = {scenario.scenario_id for scenario in scenarios}

    assert "uptrend_reversion_adx35_orb1200" in scenario_ids
    assert "uptrend_reversion_adx35_orb1300" in scenario_ids
    assert "uptrend_reversion_adx35_orb1215" not in scenario_ids
    assert "uptrend_reversion_adx35_orb1245" not in scenario_ids

    for scenario in scenarios[1:]:
        diffs = [
            idx
            for idx, (base_rule, scenario_rule) in enumerate(zip(base_rules, scenario.strategy_rules))
            if base_rule != scenario_rule
        ]

        assert len(diffs) == 1
        diff_rule = scenario.strategy_rules[diffs[0]]
        assert diff_rule["signal_type"] == scenario.perturbed_rule_signal_type
        assert scenario.perturbed_adx_threshold is not None
        assert scenario.perturbed_orb_start is not None
