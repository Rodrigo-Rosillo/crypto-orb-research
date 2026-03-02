from __future__ import annotations

from forward.live_testnet import _should_cancel_on_exit


def test_should_cancel_on_exit_truth_table() -> None:
    assert _should_cancel_on_exit(config_flag=True, runtime_skip=False) is True
    assert _should_cancel_on_exit(config_flag=True, runtime_skip=True) is False
    assert _should_cancel_on_exit(config_flag=False, runtime_skip=False) is False
    assert _should_cancel_on_exit(config_flag=False, runtime_skip=True) is False
