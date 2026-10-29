from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from solar_filament.metrics.metrics_official import PanopticQuality as OfficialPQ
from solar_filament.pipeline.dense_oversample import (
    dense_cut_value,
    dense_replication_counts,
)

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / 'scratch' / 'build_v57_colab_dense_oversample_pilot.py'
spec = importlib.util.spec_from_file_location('v57_builder_test', BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def _execute(source, namespace):
    exec(source, namespace)  # noqa: S102 - trusted generated cells run with CPU-only fakes


@pytest.fixture
def dataset(tmp_path):
    preproc = tmp_path / 'preproc'
    preproc.mkdir()
    for name in ['a.jpeg', 'b.jpeg', 'val.jpeg']:
        (preproc / name).write_bytes(name.encode())
    cfg = SimpleNamespace(YOLO_ROOT=tmp_path / 'datasets', PREPROC_DIR=preproc,
                          RUNS_ROOT=tmp_path / 'runs', FOLD=0, DENSE_QUANTILE=0.75,
                          DENSE_EXTRA=1, MANIFEST=tmp_path / 'manifest.json',
                          CONFIGS=('baseline', 'dense2x'), SEEDS=(3407, 8407))
    ns = {'Path': Path, 'np': np, 'json': json, 'shutil': shutil, 'CFG': cfg,
          'tqdm': lambda items, **kwargs: items,
          'dense_replication_counts': dense_replication_counts, 'dense_cut_value': dense_cut_value,
          'sample_counts': [1, 5], 'DENSE_CUT': dense_cut_value([1, 5]),
          'train_samples': [('a.jpeg', 1), ('b.jpeg', 2)],
          'valid_files_fold': ['val.jpeg'], 'file_to_reference_id': {'val.jpeg': 3},
          'images': {i: {'width': 64, 'height': 64} for i in [1, 2, 3]},
          'annotations_by_image': {i: [{'segmentation': [[0, 0, 8, 0, 8, 8]]}] for i in [1, 2, 3]},
          'PILOT_CONTRACT': {'fold': 0, 'inputs_sha256': 'original'}}
    _execute(builder.build_cells(0)[4]['source'], ns)
    return ns


def test_dataset_reuses_only_matching_input_contract(dataset):
    build = dataset['build_fold_dataset']
    yaml, initial = build('dense2x')
    assert initial['n_train_samples'] == 3
    assert len(list((yaml.parent / 'images/train').glob('*.jpeg'))) == 3
    assert len(list((yaml.parent / 'images/val').glob('*.jpeg'))) == 1
    assert build('dense2x')[1]['reused'] is True
    dataset['PILOT_CONTRACT'] = {'fold': 0, 'inputs_sha256': 'changed'}
    with pytest.raises(ValueError, match='input contract'):
        build('dense2x')


def test_dataset_refuses_legacy_reuse_without_contract(dataset):
    root = dataset['CFG'].YOLO_ROOT / 'baseline' / 'fold_0'
    root.mkdir(parents=True)
    (root / 'data.yaml').write_text('stale')
    with pytest.raises(ValueError, match='input contract'):
        dataset['build_fold_dataset']('baseline')


@pytest.mark.parametrize('checkpoint', ['best', 'last'])
def test_training_refuses_changed_inputs_before_resume(dataset, checkpoint):
    cfg = dataset['CFG']
    run = cfg.RUNS_ROOT / 'yolo_v57_baseline_s3407_fold0'
    (run / 'weights').mkdir(parents=True)
    (run / 'weights' / f'{checkpoint}.pt').write_bytes(b'fake-checkpoint')
    if checkpoint == 'best':
        (run / 'TRAINING_COMPLETE').write_text('complete')
    (run / 'INPUT_CONTRACT.json').write_text(json.dumps({'inputs_sha256': 'other'}))
    dataset['build_fold_dataset'] = lambda config: (Path('fake.yaml'), {})
    dataset['seed_everything'] = lambda seed: None
    dataset['YOLO'] = lambda *args: pytest.fail('must refuse stale checkpoint before loading YOLO')
    source = '\n'.join(line for line in builder.build_cells(0)[5]['source'].splitlines()
                       if line != 'from ultralytics import YOLO')
    _execute(source, dataset)
    with pytest.raises(ValueError, match='input contract'):
        dataset['train_yolo']('baseline', 3407)


def test_manifest_refuses_unverified_done_keys_before_training(dataset):
    dataset['CFG'].MANIFEST.write_text(json.dumps({'runs': [{'config': 'baseline', 'seed': 3407}]}))
    dataset['build_fold_dataset'] = lambda config: pytest.fail('must reject legacy manifest first')
    with pytest.raises(ValueError, match='input contract'):
        _execute(builder.build_cells(0)[8]['source'], dataset)


@pytest.mark.parametrize('change', ['replace', 'remove'])
def test_training_refuses_changed_initial_weights_before_loading_an_arm(dataset, tmp_path, change):
    initial = tmp_path / 'yolo11m-seg.pt'
    initial.write_bytes(b'original-initial-weights')
    dataset.update({'INITIAL_YOLO_PATH': initial, 'hashlib': hashlib})
    _execute(compile(_cell_definitions(3, {'file_sha256'}), 'v57_hash', 'exec'), dataset)
    dataset['PILOT_CONTRACT']['initial_yolo_sha256'] = dataset['file_sha256'](initial)
    dataset['CFG'].YOLO_MODEL = initial.name
    # A pending arm must not silently start from a same-named checkpoint
    # with different bytes, including within the current notebook session.
    dataset['build_fold_dataset'] = lambda config: (Path('fake.yaml'), {})
    dataset['seed_everything'] = lambda seed: None
    dataset['YOLO'] = lambda *args: pytest.fail('must reject initial weights before loading YOLO')
    if change == 'replace':
        initial.write_bytes(b'replacement-initial-weights')
    else:
        initial.unlink()
    source = '\n'.join(line for line in builder.build_cells(0)[5]['source'].splitlines()
                       if line != 'from ultralytics import YOLO')
    _execute(source, dataset)
    with pytest.raises(ValueError, match='initial YOLO'):
        dataset['train_yolo']('dense2x', 8407)


def _cell_definitions(index, names):
    tree = ast.parse(builder.build_cells(0)[index]['source'])
    return ast.Module(body=[node for node in tree.body
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names],
                      type_ignores=[])


@pytest.mark.parametrize('fold', [0, 1, 2])
def test_generated_notebook_and_code_cells_match_source(fold):
    notebook = json.loads((ROOT / f'solar_filament_v57_dense_oversample_fold{fold}_colab.ipynb').read_text(encoding="utf-8"))
    expected = builder.build_cells(fold)
    for i, cell in enumerate(expected):
        cell['id'] = f'v57-dense-f{fold}-{i:02d}'
        if cell['cell_type'] == 'code':
            source = '\n'.join(line for line in cell['source'].splitlines() if not line.startswith('!'))
            ast.parse(source)
    assert notebook['cells'] == expected


def test_detector_recipe_is_preserved_from_v30():
    # The v30 payload strings moved to the cells module; scan both sources.
    original = ''.join(
        (ROOT / f'scratch/{name}').read_text(encoding='utf-8')
        for name in ('build_v30_prize_colab.py', 'build_v30_prize_colab_cells.py')
    )
    module_tree = ast.parse(original)
    strings = [n.value for n in ast.walk(module_tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    old_source = next(s for s in strings if 'def train_yolo_fold' in s)

    def train_kwargs(source):
        calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'train'
                 and any(k.arg == 'optimizer' for k in n.keywords)]
        assert len(calls) == 1
        return {k.arg: ast.dump(k.value) for k in calls[0].keywords
                if k.arg not in {'seed', 'data', 'project', 'name'}}

    assert train_kwargs(builder.build_cells(0)[5]['source']) == train_kwargs(old_source)


@pytest.mark.parametrize('matrix,n_gt,n_pred', [
    (np.empty((0, 2)), 0, 2), (np.empty((2, 0)), 2, 0),
    (np.array([[0.5, 0.6], [0.7, 0.0]]), 2, 2),
    (np.array([[0.8], [0.8]]), 2, 1),
])
def test_pilot_pq_matches_current_official_semantics(matrix, n_gt, n_pred):
    ns = {'np': np}
    _execute(compile(_cell_definitions(7, {'PanopticQuality'}), 'v57_score', 'exec'), ns)
    pilot, official = ns['PanopticQuality'](), OfficialPQ()
    pilot.add_matrix(matrix, n_gt, n_pred)
    official.add_matrix(matrix, n_gt, n_pred)
    for key, value in pilot.summary().items():
        assert value == pytest.approx(official.summary()[key])


def test_four_fake_runs_resume_and_refuse_changed_checkpoint(dataset):
    ns = dataset
    calls = []
    cfg = ns['CFG']
    cfg.BASELINE_PQ_REFINED = 0.4393

    def fake_train(config, seed):
        calls.append((config, seed))
        path = cfg.RUNS_ROOT / f'{config}_{seed}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f'{config}/{seed}'.encode())
        return path

    cuda = SimpleNamespace(reset_peak_memory_stats=lambda: None,
                           get_device_name=lambda index: 'FAKE_CPU_TEST',
                           max_memory_allocated=lambda: 0)
    ns.update({'hashlib': hashlib, 'time': time, 'datetime': datetime, 'timezone': timezone,
               'torch': SimpleNamespace(cuda=cuda), 'train_yolo': fake_train,
               'evaluate_checkpoint': lambda *args: {'raw': {'pq': 0.4}, 'refined': {'pq': 0.5},
                                                      'fn_dense_val': 2, 'fn_sparse_val': 3}})
    _execute(compile(_cell_definitions(3, {'file_sha256'}), 'v57_hash', 'exec'), ns)
    source = builder.build_cells(0)[8]['source']
    _execute(source, ns)
    assert len(calls) == 4
    manifest = json.loads(cfg.MANIFEST.read_text(encoding="utf-8"))
    assert manifest['input_contract'] == ns['PILOT_CONTRACT']
    assert len(manifest['runs']) == 4
    assert all(row['execution']['gpu_name'] == 'FAKE_CPU_TEST' for row in manifest['runs'])
    _execute(source, ns)
    assert len(calls) == 4
    Path(manifest['runs'][0]['checkpoint']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint/arm'):
        _execute(source, ns)
    assert len(calls) == 4


@pytest.mark.parametrize('state,scope', [
    ('complete', 'evaluation_only'),
    ('partial', 'resumed_training_and_evaluation'),
    ('fresh', 'training_and_evaluation'),
])
def test_checkpoint_recovery_marks_current_invocation_telemetry(dataset, state, scope):
    ns, cfg = dataset, dataset['CFG']
    cfg.CONFIGS, cfg.SEEDS = ('baseline',), (3407,)
    cfg.BASELINE_PQ_REFINED = 0.4393
    run = cfg.RUNS_ROOT / 'yolo_v57_baseline_s3407_fold0'
    (run / 'weights').mkdir(parents=True)
    checkpoint = run / 'weights/best.pt'
    checkpoint.write_bytes(b'already-trained')
    if state == 'complete':
        (run / 'TRAINING_COMPLETE').write_text('complete')
    elif state == 'partial':
        (run / 'weights/last.pt').write_bytes(b'partial-training')
    cuda = SimpleNamespace(reset_peak_memory_stats=lambda: None,
                           get_device_name=lambda index: 'EVALUATION_GPU',
                           max_memory_allocated=lambda: 0)
    ns.update({'hashlib': hashlib, 'time': time, 'datetime': datetime, 'timezone': timezone,
               'torch': SimpleNamespace(cuda=cuda), 'train_yolo': lambda *args: checkpoint,
               'evaluate_checkpoint': lambda *args: {'raw': {'pq': 0.4}, 'refined': {'pq': 0.5},
                                                      'fn_dense_val': 2, 'fn_sparse_val': 3}})
    _execute(compile(_cell_definitions(3, {'file_sha256'}), 'v57_hash', 'exec'), ns)
    _execute(builder.build_cells(0)[8]['source'], ns)
    row = json.loads(cfg.MANIFEST.read_text(encoding='utf-8'))['runs'][0]
    assert row['execution']['measurement_scope'] == scope
    expected_prior = None if state == 'fresh' else 'unavailable: not recorded in this invocation'
    assert row['execution']['prior_training_telemetry'] == expected_prior
    assert row['execution']['gpu_name'] == 'EVALUATION_GPU'


def test_density_cut_uses_training_observations_and_fingerprints_validation_inputs(dataset, tmp_path):
    ns, cfg = dataset, dataset['CFG']
    cfg.DATA_CACHE = tmp_path / 'data'
    ann_dir = cfg.DATA_CACHE / 'train' / 'annotations'
    ann_dir.mkdir(parents=True)
    ann_path = ann_dir / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    image_records = [{'id': i, 'file_name': name, 'width': 64, 'height': 64}
                     for i, name in enumerate(['a.jpeg', 'b.jpeg', 'val.jpeg'], 1)]
    annotations = [{'image_id': i, 'iscrowd': 0} for i, count in [(1, 1), (2, 5), (3, 99)]
                   for _ in range(count)]
    ann_path.write_text(json.dumps({'images': image_records, 'annotations': annotations}), encoding='utf-8')
    cfg.ARTIFACT_ROOT = tmp_path / 'artifacts'
    cfg.ARTIFACT_ROOT.mkdir()
    (cfg.ARTIFACT_ROOT / 'grouped_folds.csv').write_text('file_name,fold\na.jpeg,1\nb.jpeg,2\nval.jpeg,0\n')
    (cfg.ARTIFACT_ROOT / 'mediseg_refiner_fold0_resume.pt').write_bytes(b'fake-refiner')
    cfg.PREPROC_CACHE = tmp_path / 'preproc_cache'
    cfg.PREPROC_CACHE.mkdir()
    for row in image_records:
        (cfg.PREPROC_CACHE / row['file_name']).write_bytes(b'current-pixels')
    config_source = '\n'.join(line for line in builder.build_cells(0)[1]['source'].splitlines()
                              if not line.startswith('!'))
    config_class = next(n for n in ast.parse(config_source).body if isinstance(n, ast.ClassDef) and n.name == 'CFG')
    defaults = {'Path': Path}
    _execute(compile(ast.Module(body=[config_class], type_ignores=[]), 'v57_config', 'exec'), defaults)
    for name, value in vars(defaults['CFG']).items():
        if name.isupper() and isinstance(value, (str, int, float, tuple)):
            setattr(cfg, name, value)
    ns.update({'pd': pd, 'defaultdict': defaultdict, 'hashlib': hashlib, 'sys': sys,
               'torch': SimpleNamespace(__version__='FAKE_CPU_TEST'),
               'cv2': SimpleNamespace(__version__='FAKE_CPU_TEST'), 'version': lambda name: 'FAKE_CPU_TEST'})
    initial = tmp_path / 'yolo11m-seg.pt'
    initial.write_bytes(b'original-initial-weights')
    ns['YOLO'] = lambda name: SimpleNamespace(ckpt_path=str(initial))
    source = builder.build_cells(0)[3]['source']
    _execute(source, ns)
    assert ns['DENSE_CUT'] == 4.0
    assert ns['train_samples'] == [('a.jpeg', 1), ('b.jpeg', 2)]
    contract = ns['PILOT_CONTRACT'].copy()
    annotations += [{'image_id': 3, 'iscrowd': 0}] * 100
    ann_path.write_text(json.dumps({'images': image_records, 'annotations': annotations}), encoding='utf-8')
    _execute(source, ns)
    assert ns['DENSE_CUT'] == 4.0
    assert ns['PILOT_CONTRACT'] != contract
    contract = ns['PILOT_CONTRACT'].copy()
    initial.write_bytes(b'replacement-initial-weights')
    _execute(source, ns)
    assert ns['INITIAL_YOLO_PATH'] == initial.resolve()
    assert ns['PILOT_CONTRACT']['initial_yolo_sha256'] == ns['file_sha256'](initial)
    assert ns['PILOT_CONTRACT'] != contract
    assert ns['PILOT_CONTRACT']['annotations_sha256'] == ns['file_sha256'](ann_path)

    contract = ns['PILOT_CONTRACT'].copy()
    cfg.CONF = 0.3
    _execute(source, ns)
    assert ns['PILOT_CONTRACT']['recipe']['CONF'] == 0.3
    assert ns['PILOT_CONTRACT'] != contract
