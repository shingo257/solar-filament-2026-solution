"""Legacy-cache probes must not silently execute or claim scientific adoption."""

import ast
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scratch" / "v68_magnetogram_probe.py"


def function(name, namespace):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace[name]


def test_default_main_blocks_before_any_cache_or_network_work():
    messages = []
    main = function("main", {"print": lambda *args, **kwargs: messages.append(args)})
    # No MAG_DIR/log/fetch/model names exist: reaching legacy work must fail this test.
    assert main() == 2
    assert "blocked" in str(messages).lower()


@pytest.mark.parametrize("delta", [0.1, 0.001, -0.1])
def test_any_legacy_delta_remains_scientifically_blocked(delta):
    status = function("legacy_exploratory_status", {})(delta)
    assert status["scientific_acceptance"] is False
    assert status["verdict"] == "blocked-unverified-provenance"
    assert "legacy" in status["kind"]
