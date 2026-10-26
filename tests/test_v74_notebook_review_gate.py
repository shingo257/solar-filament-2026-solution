"""Notebook entry points must require independently reviewed provenance before IO."""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_requires_reviewed_pin_before_input_io(monkeypatch):
    monkeypatch.delenv("V73_PROVENANCE_SHA256", raising=False)
    monkeypatch.setattr(Path, "rglob", lambda *args: pytest.fail("input IO before review pin"))
    notebook = json.loads((ROOT / "solar_filament_v74_ssl_confirm_colab.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code")
    with pytest.raises(RuntimeError, match="independent.*review"):
        exec(compile(source, "nb", "exec"), {})  # noqa: S102 - run repository preflight with IO blocked


def test_generated_notebook_matches_builder():
    tree = ast.parse((ROOT / "scratch/build_v74_ssl_confirm_colab.py").read_text(encoding="utf-8"))
    cells = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == "CELLS" for t in n.targets)))
    notebook = json.loads((ROOT / "solar_filament_v74_ssl_confirm_colab.ipynb").read_text(encoding="utf-8"))
    assert [c["source"] for c in cells] == [c["source"] for c in notebook["cells"]]
