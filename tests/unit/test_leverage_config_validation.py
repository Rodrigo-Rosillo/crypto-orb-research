from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable, cast

import pytest


def _load_parse_leverage() -> Callable[[dict[str, Any]], float]:
    path = Path(__file__).resolve().parents[2] / "scripts" / "forward_test.py"
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)

    func_src = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_leverage":
            func_src = ast.get_source_segment(source, node)
            break

    if not isinstance(func_src, str) or not func_src.strip():
        raise AssertionError("_parse_leverage not found in scripts/forward_test.py")

    namespace: dict[str, Any] = {}
    exec("from typing import Any\n" + func_src, namespace)
    fn = namespace.get("_parse_leverage")
    if not callable(fn):
        raise AssertionError("_parse_leverage failed to load")
    return cast(Callable[[dict[str, Any]], float], fn)


_PARSE_LEVERAGE = _load_parse_leverage()


@pytest.mark.parametrize("value", [1.9, 2.5])
def test_parse_leverage_rejects_fractional_values(value: float) -> None:
    cfg = {"leverage": {"enabled": True, "max_leverage": value}}

    with pytest.raises(ValueError, match=r"leverage\.max_leverage must be a whole number"):
        _PARSE_LEVERAGE(cfg)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, 1.0),
        (2.0, 2.0),
        (1, 1.0),
        (3, 3.0),
    ],
)
def test_parse_leverage_accepts_whole_number_values(value: float | int, expected: float) -> None:
    cfg = {"leverage": {"enabled": True, "max_leverage": value}}

    assert _PARSE_LEVERAGE(cfg) == expected
