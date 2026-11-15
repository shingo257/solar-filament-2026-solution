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


def _variant_if(marker):
    source = "".join(_notebook()["cells"][12]["source"])
    return next(node for node in ast.parse(source).body if isinstance(node, ast.If)
                and "SUBMISSION_VARIANT" in ast.unparse(node.test)
                and marker in ast.unparse(node))


def _dispatch(variant, attachment):
    dispatch = _variant_if("refiner =")
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
    assert patterns == [checkpoint.name,
                        "refiner_plain_mhsa_fold2_s8407.pt"]


def test_missing_plain_attachment_stops_before_model_creation():
    with pytest.raises(FileNotFoundError):
        _dispatch("plain_mhsa_f2", None)


def test_unknown_variant_cannot_fall_back_to_prod():
    with pytest.raises(ValueError, match="Unknown submission variant"):
        _dispatch("plain_mhsa_typo", None)


def _detector(variant, attachment):
    dispatch = _variant_if("detector_ckpt =")
    patterns_seen = []

    def resolve(patterns):
        patterns_seen.extend(patterns)
        return attachment

    scope = {"CFG": SimpleNamespace(SUBMISSION_VARIANT=variant, SINGLE_FOLD=2),
             "resolve_pretrained": resolve,
             "yolo_weight_paths": {2: Path("yolo_fold_2/best.pt")}}
    tree = ast.fix_missing_locations(ast.Module(body=[dispatch], type_ignores=[]))
    exec(compile(tree, "generated-detector-dispatch", "exec"), scope)
    return scope["detector_ckpt"], patterns_seen


def test_prod_detector_is_fold2_yolo():
    ckpt, patterns = _detector("prod", None)
    assert ckpt == Path("yolo_fold_2/best.pt")
    assert not patterns


def test_v96full_detector_resolves_exact_full_data_weights():
    full = Path("weights/yolo_full_fold2_best.pt")
    ckpt, patterns = _detector("prod_v96full", full)
    assert ckpt == full
    assert patterns == ["yolo_full_fold2_best.pt"]


def test_missing_v96full_weights_never_fall_back_to_fold2():
    with pytest.raises(FileNotFoundError):
        _detector("prod_v96full", None)


def test_v96full_dispatch_uses_prod_all_data_refiner():
    kind, loaded, patterns = _dispatch("prod_v96full", None)
    assert kind == "edge"
    assert loaded == [Path("all-data.pt")]
    assert not patterns


def test_v96full_is_a_disjoint_audited_variant():
    nb = _notebook()
    setup = "".join(nb["cells"][2]["source"])
    audit = "".join(nb["cells"][14]["source"])
    assert '"prod_v96full"' in setup
    assert '"prod_v96full"' in audit.split("assert overlap_n == 0")[0]


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


def _detector_mapping(variant, mapping):
    dispatch = _variant_if("detector_ckpt =")
    patterns_seen = []

    def resolve(patterns):
        patterns_seen.extend(patterns)
        for p in patterns:
            if p in mapping:
                return mapping[p]
        return None

    scope = {"CFG": SimpleNamespace(SUBMISSION_VARIANT=variant, SINGLE_FOLD=2),
             "resolve_pretrained": resolve,
             "yolo_weight_paths": {2: Path("yolo_fold_2/best.pt")}}
    tree = ast.fix_missing_locations(ast.Module(body=[dispatch], type_ignores=[]))
    exec(compile(tree, "generated-detector-dispatch", "exec"), scope)
    return scope, patterns_seen


def test_sptal_trio_detector_resolves_all_three_all_data_weights():
    mapping = {
        "yolo_full_fold2_best.pt": Path("weights/yolo_full_fold2_best.pt"),
        "yolo26m_all_fold0_best.pt": Path("weights/yolo26m_all_fold0_best.pt"),
        "yolo_spine_small_tal_all_fold2_best.pt": Path("weights/yolo_spine_small_tal_all_fold2_best.pt"),
    }
    scope, patterns = _detector_mapping("cons2of3_sptal_maj3", mapping)
    assert scope["detector_ckpt"] == mapping["yolo_full_fold2_best.pt"]
    assert scope["yolo26_ckpt"] == mapping["yolo26m_all_fold0_best.pt"]
    assert scope["sptal_ckpt"] == mapping["yolo_spine_small_tal_all_fold2_best.pt"]
    assert patterns == [
        "yolo_full_fold2_best.pt",
        "yolo26m_all_fold0_best.pt",
        "yolo_spine_small_tal_all_fold2_best.pt",
    ]


@pytest.mark.parametrize(
    "missing_key",
    [
        "yolo_full_fold2_best.pt",
        "yolo26m_all_fold0_best.pt",
        "yolo_spine_small_tal_all_fold2_best.pt",
    ],
)
def test_missing_any_sptal_trio_weight_never_falls_back_to_fold2(missing_key):
    mapping = {
        "yolo_full_fold2_best.pt": Path("weights/yolo_full_fold2_best.pt"),
        "yolo26m_all_fold0_best.pt": Path("weights/yolo26m_all_fold0_best.pt"),
        "yolo_spine_small_tal_all_fold2_best.pt": Path("weights/yolo_spine_small_tal_all_fold2_best.pt"),
    }
    mapping.pop(missing_key)
    with pytest.raises(FileNotFoundError):
        _detector_mapping("cons2of3_sptal_maj3", mapping)


def test_sptal_trio_dispatch_uses_prod_all_data_refiner():
    kind, loaded, patterns = _dispatch("cons2of3_sptal_maj3", None)
    assert kind == "edge"
    assert loaded == [Path("all-data.pt")]
    assert not patterns


def test_sptal_trio_is_disjoint_audited_and_embeds_1to1_consensus_helpers():
    nb = _notebook()
    setup = "".join(nb["cells"][2]["source"])
    cell12 = "".join(nb["cells"][12]["source"])
    audit = "".join(nb["cells"][14]["source"])
    assert '"cons2of3_sptal_maj3"' in setup
    assert '"cons2of3_sptal_maj3"' in audit.split("assert overlap_n == 0")[0]
    parsed12 = ast.parse(cell12)
    classes12 = {n.name for n in parsed12.body if isinstance(n, ast.ClassDef)}
    funcs12 = {n.name for n in parsed12.body if isinstance(n, ast.FunctionDef)}
    assert {"DetectorProposal", "MatchedCluster"} <= classes12
    assert {
        "match_detector_proposals_1to1",
        "fuse_refined_crop_specs",
        "fast_apply_own_specs",
        "refine_batch_proposals",
    } <= funcs12
