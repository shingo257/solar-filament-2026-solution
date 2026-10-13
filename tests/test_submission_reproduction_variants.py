"""Run the generated refiner dispatch with synthetic checkpoints only."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE = Path(__file__).resolve().parents[1]


def _notebook():
    return json.loads((MODULE / "solar_filament_submission_reproduction.ipynb")
                      .read_text(encoding="utf-8"))


def _dispatch(variant, attachment):
    source = "".join(_notebook()["cells"][12]["source"])
    nodes = ast.parse(source).body
    dispatch = next(node for node in nodes if isinstance(node, ast.If)
                    and "SUBMISSION_VARIANT" in ast.unparse(node.test))
    loaded = []
    patterns_seen = []

    class Model:
        def __init__(self, kind):
            self.kind = kind

        def cuda(self):
            return self

        def eval(self):
            return self

    def resolve(patterns):
        patterns_seen.extend(patterns)
        return attachment

    scope = {
        "CFG": SimpleNamespace(SUBMISSION_VARIANT=variant),
        "ALLDATA_CKPT": Path("all-data.pt"), "resolve_pretrained": resolve,
        "MediSegPlainMHSAUNet": lambda *args: Model("plain"),
        "MediSegEdgeAttUNet": lambda *args: Model("edge"),
        "torch": SimpleNamespace(load=lambda path, **kwargs: loaded.append(path)),
    }
    tree = ast.fix_missing_locations(ast.Module(body=[dispatch], type_ignores=[]))
    exec(compile(tree, "generated-variant-dispatch", "exec"), scope)
    return scope["refiner"].kind, loaded, patterns_seen


def test_prod_dispatch_loads_all_data_edge_model():
    kind, loaded, patterns = _dispatch("prod", None)
    assert kind == "edge"
    assert loaded == [Path("all-data.pt")]
    assert not patterns


def test_plain_dispatch_loads_exact_fold2_seed8407_model():
    checkpoint = Path("ablation_plain_mhsa_seed8407_fold2.pt")
    kind, loaded, patterns = _dispatch("plain_mhsa_f2", checkpoint)
    assert kind == "plain"
    assert loaded == [checkpoint]
    assert patterns == [checkpoint.name]


def test_missing_plain_attachment_stops_before_model_creation():
    with pytest.raises(FileNotFoundError):
        _dispatch("plain_mhsa_f2", None)


def test_unknown_variant_cannot_fall_back_to_prod():
    with pytest.raises(ValueError, match="Unknown submission variant"):
        _dispatch("plain_mhsa_typo", None)


def test_notebook_contains_plain_model_definition():
    defined = set()
    for cell in _notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(line for line in "".join(cell["source"]).splitlines()
                           if not line.lstrip().startswith("%"))
        defined.update(node.name for node in ast.parse(source).body
                       if isinstance(node, ast.ClassDef))
    assert {"MediSegPlainMHSAUNet", "PlainMHSA", "MediSegEdgeAttUNet"} <= defined
