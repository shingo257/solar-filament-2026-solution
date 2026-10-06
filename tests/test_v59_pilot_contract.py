"""CPU contract tests for v59; no Colab mount or GPU execution."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v59", ROOT / "scratch/build_v59_boundary_dt_pilot.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def functions(tmp_path):
    text = "\n".join(c["source"] for c in BUILDER.build_cells(0) if c["cell_type"] == "code")
    # Notebook pip magics are never evaluated by CPU contract tests.
    tree = ast.parse("\n".join(l for l in text.splitlines() if not l.lstrip().startswith("!")))
    names = {"read_pilot_manifest", "validate_run_row", "reuse_baselines", "validate_checkpoint", "file_sha256", "atomic_save", "validate_paired_rows"}
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in found} == names, "v59 lacks fail-closed manifest/checkpoint contracts"
    cfg = SimpleNamespace(FOLD=0, CONFIGS=("baseline", "dtb"), SEEDS=(3407, 8407),
                          MANIFEST=tmp_path / "manifest.json", REUSE_V54_BASELINE=False)
    namespace = {"CFG": cfg, "PILOT_CONTRACT": {"fold": 0, "hash": "a"},
                 "json": json, "math": __import__("math"), "Path": Path}
    exec(compile(ast.Module(body=found, type_ignores=[]), "v59-contract", "exec"), namespace)  # noqa: S102 - execute trusted generated functions in CPU tests
    return namespace


def row(tmp_path):
    import hashlib
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"validated-checkpoint")
    return {"config": "baseline", "seed": 3407, "fold": 0, "pq": .4, "sq": .5, "rq": .8, "tp": 4, "fp": 1, "fn": 1,
            "checkpoint": str(checkpoint),
            "input_contract": {"fold": 0, "hash": "a"}, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "initial_state_sha256": "b" * 64, "checkpoint_selection": "best_model", "batch_size": 4}


@pytest.mark.parametrize("mutation", ["fold", "contract", "duplicate", "nan", "selection", "hash"])
def test_old_or_mixed_manifest_is_rejected_without_rewrite(tmp_path, mutation):
    ns = functions(tmp_path)
    r = row(tmp_path)
    if mutation == "fold": r["fold"] = 1
    if mutation == "contract": r.pop("input_contract")
    if mutation == "nan": r["pq"] = float("nan")
    if mutation == "selection": r["checkpoint_selection"] = "last"
    if mutation == "hash": r.pop("checkpoint_sha256")
    rows = [r, dict(r)] if mutation == "duplicate" else [r]
    p = ns["CFG"].MANIFEST
    p.write_text(json.dumps({"experiment_id": "v59_boundary_dt_fold0", "input_contract": ns["PILOT_CONTRACT"], "runs": rows}))
    before = p.read_bytes()
    with pytest.raises(ValueError): ns["read_pilot_manifest"]()
    assert p.read_bytes() == before


def test_v54_legacy_reuse_is_off_by_default(tmp_path):
    ns = functions(tmp_path)
    assert ns["reuse_baselines"]({"runs": [{"config": "baseline", "seed": 3407, "fold": 2, "pq": .9}]}) == []


def test_explicit_reuse_rejects_wrong_fold_without_relabelling(tmp_path):
    ns = functions(tmp_path)
    ns["CFG"].REUSE_V54_BASELINE = True
    r = row(tmp_path); r["fold"] = 2
    with pytest.raises(ValueError): ns["reuse_baselines"]({"runs": [r]})
    assert r["fold"] == 2


@pytest.mark.parametrize("status", ["partial", "complete_wrong_contract", "missing_best"])
def test_incomplete_checkpoint_is_not_eval_only(tmp_path, status):
    ns = functions(tmp_path)
    ckpt = {"training_status": "complete", "input_contract": ns["PILOT_CONTRACT"],
            "config": "dtb", "seed": 3407, "fold": 0, "best_model": {"w": 1},
            "model": {"w": 2}, "initial_state_sha256": "b" * 64, "training_summary": {"minutes": 2}}
    if status == "partial": ckpt["training_status"] = "partial"
    if status == "complete_wrong_contract": ckpt["input_contract"] = {}
    if status == "missing_best": ckpt.pop("best_model")
    with pytest.raises(ValueError): ns["validate_checkpoint"](ckpt, "dtb", 3407)


def test_valid_manifest_and_explicit_equivalent_baseline_roundtrip(tmp_path):
    ns = functions(tmp_path)
    r = row(tmp_path)
    manifest = {"experiment_id": "v59_boundary_dt_fold0", "input_contract": ns["PILOT_CONTRACT"], "runs": [r]}
    ns["CFG"].MANIFEST.write_text(json.dumps(manifest))
    assert ns["read_pilot_manifest"]() == manifest
    ns["CFG"].REUSE_V54_BASELINE = True
    reused = ns["reuse_baselines"](manifest)
    assert reused[0]["fold"] == r["fold"] and r == manifest["runs"][0]


def test_checkpoint_bytes_change_rejects_resume(tmp_path):
    ns = functions(tmp_path)
    r = row(tmp_path)
    Path(r["checkpoint"]).write_bytes(b"replacement-checkpoint")
    with pytest.raises(ValueError, match="bytes changed"):
        ns["validate_run_row"](r)


@pytest.mark.parametrize("field", ["sq", "rq", "tp", "fp", "fn"])
def test_incomplete_metrics_cannot_be_treated_as_completed(tmp_path, field):
    ns = functions(tmp_path)
    r = row(tmp_path)
    r.pop(field)
    with pytest.raises(ValueError):
        ns["validate_run_row"](r)


@pytest.mark.parametrize("fold", [0, 1, 2])
def test_generated_notebook_parity_and_code_syntax(fold):
    notebook = json.loads((ROOT / f"solar_filament_v59_boundary_dt_fold{fold}_colab.ipynb").read_text(encoding="utf-8"))
    expected = BUILDER.build_cells(fold)
    assert len(notebook["cells"]) == len(expected)
    for saved, generated in zip(notebook["cells"], expected):
        assert saved["source"] == generated["source"]
        if generated["cell_type"] == "code":
            ast.parse("\n".join(l for l in generated["source"].splitlines() if not l.lstrip().startswith(("!", "%"))))


@pytest.mark.parametrize("changed", [None, "annotation", "folds", "detector", "pixels", "protocol", "recipe"])
def test_actual_input_contract_is_json_stable_and_fingerprints_consumed_inputs(tmp_path, changed):
    import hashlib
    import sys
    setup = BUILDER.build_cells(0)[1]["source"]
    tree = ast.parse("\n".join(l for l in setup.splitlines() if not l.lstrip().startswith(("!", "%"))))
    ns = {"Path": Path}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]), "v59-config", "exec"), ns)  # noqa: S102 - trusted generated config
    cfg = ns["CFG"]
    cfg.DRIVE_ROOT = cfg.ARTIFACT_ROOT = tmp_path
    detector = tmp_path / "runs/yolo_fold_0/weights/best.pt"
    detector.parent.mkdir(parents=True)
    paths = {"annotation": tmp_path / "annotation.json", "folds": tmp_path / "grouped_folds.csv", "detector": detector, "pixels": tmp_path / "pixel.jpeg"}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    ns.update({"hashlib": hashlib, "json": json, "sys": sys, "TRAIN_JSON": paths["annotation"],
               "preprocessed_hashes": {"pixel.jpeg": hashlib.sha256(paths["pixels"].read_bytes()).hexdigest()},
               "torch": SimpleNamespace(__version__="CPU-test"), "np": SimpleNamespace(__version__="CPU-test"),
               "cv2": SimpleNamespace(__version__="CPU-test"), "version": lambda package: "CPU-test"})
    data = next(c["source"] for c in BUILDER.build_cells(0) if c["cell_type"] == "code" and "# [3/8]" in c["source"])
    tree = ast.parse("\n".join(l for l in data.splitlines() if not l.lstrip().startswith(("!", "%"))))
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "file_sha256")
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PILOT_CONTRACT" for t in n.targets))]
    code = compile(ast.Module(body=nodes, type_ignores=[]), "v59-input-contract", "exec")
    exec(code, ns)  # noqa: S102 - evaluate trusted generated contract only
    before = ns["PILOT_CONTRACT"]
    assert json.loads(json.dumps(before, allow_nan=False)) == before
    if changed is None:
        return
    if changed == "protocol":
        cfg.PROTOCOL_SHA256 = "changed-protocol"
    elif changed == "recipe":
        cfg.BDT_WEIGHT = .2
    else:
        paths[changed].write_bytes(b"changed bytes")
        if changed == "pixels":
            ns["preprocessed_hashes"]["pixel.jpeg"] = hashlib.sha256(paths[changed].read_bytes()).hexdigest()
    exec(code, ns)  # noqa: S102 - evaluate trusted generated contract only
    assert ns["PILOT_CONTRACT"] != before


@pytest.mark.parametrize('different', ['batch_size', 'initial_state_sha256'])
def test_paired_rows_reject_other_training_differences(tmp_path, different):
    ns = functions(tmp_path)
    first, second = row(tmp_path), row(tmp_path)
    first['batch_size'] = second['batch_size'] = 4
    second['config'] = 'dtb'
    second[different] = 2 if different == 'batch_size' else 'c' * 64
    with pytest.raises(ValueError, match='Paired'):
        ns['validate_paired_rows']([first, second])


def test_choose_common_batch_probes_both_losses():
    text = '\n'.join(c['source'] for c in BUILDER.build_cells(0) if c['cell_type'] == 'code')
    tree = ast.parse('\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('!')))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'choose_batch_size')
    calls = []
    def probe(dataset, candidate, loss):
        calls.append((candidate, loss))
        if candidate == 8 and loss == 'dtb':
            raise MemoryError('dtb uses more memory')
    ns = {'CFG': SimpleNamespace(REFINER_BATCH_CANDIDATES=(8, 4)),
          'probe_batch_size': probe, 'refiner_loss': 'baseline', 'refiner_loss_dtb': 'dtb',
          'torch': SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'common-batch', 'exec'), ns)  # noqa: S102
    assert ns['choose_batch_size'](object()) == 4
    assert calls == [(8, 'baseline'), (8, 'dtb'), (4, 'baseline'), (4, 'dtb')]


@pytest.mark.parametrize('checkpoint', [False, True])
def test_atomic_save_failure_preserves_previous_artifact(tmp_path, monkeypatch, checkpoint):
    ns = functions(tmp_path)
    destination = tmp_path / ('best.pt' if checkpoint else 'manifest.json')
    destination.write_bytes(b'previous-complete-artifact')
    def save(value, handle):
        handle.write(b'partial-new-checkpoint')
        raise RuntimeError('serialization interrupted')
    ns['torch'] = SimpleNamespace(save=save)
    def replace(source, target):
        raise RuntimeError('replacement interrupted')
    monkeypatch.setattr(__import__('os'), 'replace', replace)
    with pytest.raises(RuntimeError, match='interrupted'):
        ns['atomic_save'](destination, {'runs': []}, is_checkpoint=checkpoint)
    assert destination.read_bytes() == b'previous-complete-artifact'
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('checkpoint', [False, True])
def test_atomic_save_publishes_only_complete_artifact(tmp_path, checkpoint):
    ns = functions(tmp_path)
    destination = tmp_path / ('best.pt' if checkpoint else 'manifest.json')
    destination.write_bytes(b'old')
    ns['torch'] = SimpleNamespace(save=lambda value, handle: handle.write(b'complete-checkpoint'))
    ns['atomic_save'](destination, {'runs': []}, is_checkpoint=checkpoint)
    if checkpoint:
        assert destination.read_bytes() == b'complete-checkpoint'
    else:
        assert json.loads(destination.read_text()) == {'runs': []}
    assert not list(tmp_path.glob('*.tmp'))


def test_read_manifest_rejects_mismatched_paired_batch_sizes(tmp_path):
    ns = functions(tmp_path)
    first, second = row(tmp_path), row(tmp_path)
    second.update(config='dtb', batch_size=2)
    ns['CFG'].MANIFEST.write_text(json.dumps({'experiment_id': 'v59_boundary_dt_fold0',
        'input_contract': ns['PILOT_CONTRACT'], 'runs': [first, second]}))
    with pytest.raises(ValueError, match='Paired'):
        ns['read_pilot_manifest']()
