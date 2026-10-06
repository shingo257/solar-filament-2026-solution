from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v58_builder_test', ROOT / 'scratch/build_v58_conf_sweep_probe.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def execute(source, namespace):
    exec(source, namespace)  # noqa: S102 - trusted generated cells with CPU fakes


@pytest.fixture
def probe(tmp_path):
    checkpoint = tmp_path / 'prod.pt'
    checkpoint.write_bytes(b'original-weights')
    missing = tmp_path / 'missing.pt'
    cfg = SimpleNamespace(FOLD=0, MANIFEST=tmp_path / 'manifest.json',
                          CONF_SWEEP=(0.05, 0.08), PREDICT_CONF_FLOOR=0.03)
    calls = []
    ns = {'CFG': cfg, 'json': json, 'time': time, 'Path': Path,
          'PILOT_CONTRACT': {'fold': 0, 'checkpoint_sha256': {'prod': 'original'}, 'protocol_sha256': 'old'},
          'checkpoint_map': lambda fold: {'prod': checkpoint, 'v57baseline_s3407': missing},
          'collect_cache': lambda path, label: calls.append(label) or [],
          'score_conf': lambda cache, conf: {'raw': {'pq': 0.2}, 'refined': {'pq': 0.3, 'fp': 1, 'fn': 2},
                                            'fn_dense_val': 1, 'fn_sparse_val': 1},
          'torch': SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None, max_memory_allocated=lambda: 123,
                                                       reset_peak_memory_stats=lambda: None)),
          'RUNTIME_INFO': {'GPU': 'CPU-test-fake'}, 'calls': calls}
    return ns


def manifest_cell(probe):
    tree = ast.parse(builder.build_cells(0)[2]['source'])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'file_sha256', 'require_checkpoint', 'read_probe_manifest', 'save_probe_manifest'}]
    probe['math'] = __import__('math')
    probe['hashlib'] = hashlib
    execute(ast.unparse(ast.Module(body=functions, type_ignores=[])), probe)
    execute(builder.build_cells(0)[6]['source'], probe)


@pytest.mark.parametrize('change', ['legacy', 'checkpoint', 'protocol', 'fold', 'data'])
def test_stale_manifest_refused_before_inference_or_write(probe, change):
    contract = dict(probe['PILOT_CONTRACT'])
    if change == 'legacy':
        old = {'runs': []}
    else:
        if change == 'checkpoint':
            contract['checkpoint_sha256'] = {'prod': 'different'}
        elif change == 'protocol':
            contract['protocol_sha256'] = 'different'
        elif change == 'fold':
            contract['fold'] = 2
        else:
            contract['annotations_sha256'] = 'other-data'
        old = {'experiment_id': 'v58_conf_sweep_fold0', 'input_contract': contract, 'runs': []}
    path = probe['CFG'].MANIFEST
    original = json.dumps(old).encode()
    path.write_bytes(original)
    with pytest.raises(ValueError, match='contract'):
        manifest_cell(probe)
    assert not probe['calls']
    assert path.read_bytes() == original


@pytest.mark.parametrize('problem', ['duplicate', 'unknown_checkpoint', 'unknown_conf', 'wrong_fold'])
def test_invalid_rows_refused_before_inference(probe, problem):
    row = {'ckpt': 'prod', 'conf': 0.05, 'fold': 0}
    rows = [row.copy()]
    if problem == 'duplicate':
        rows.append(row.copy())
    elif problem == 'unknown_checkpoint':
        rows[0]['ckpt'] = 'foreign'
    elif problem == 'unknown_conf':
        rows[0]['conf'] = 0.99
    else:
        rows[0]['fold'] = 2
    old = {'experiment_id': 'v58_conf_sweep_fold0', 'input_contract': probe['PILOT_CONTRACT'], 'runs': rows}
    path = probe['CFG'].MANIFEST
    path.write_text(json.dumps(old))
    original = path.read_bytes()
    with pytest.raises(ValueError, match='row'):
        manifest_cell(probe)
    assert not probe['calls']
    assert path.read_bytes() == original


@pytest.mark.parametrize('problem', ['missing_count', 'negative_count', 'fractional_count', 'nan_count', 'wrong_path'])
def test_incomplete_or_misaligned_completed_rows_are_refused(probe, problem):
    row = {'ckpt': 'prod', 'conf': 0.05, 'fold': 0,
           'checkpoint': str(probe['checkpoint_map'](0)['prod']),
           'raw': {'pq': 0.2}, 'refined': {'pq': 0.3, 'fp': 1, 'fn': 2},
           'fn_dense_val': 1, 'fn_sparse_val': 1}
    if problem == 'missing_count':
        del row['fn_sparse_val']
    elif problem == 'negative_count':
        row['refined']['fn'] = -1
    elif problem == 'fractional_count':
        row['fn_dense_val'] = 0.5
    elif problem == 'nan_count':
        row['refined']['fp'] = float('nan')
    else:
        row['checkpoint'] = 'foreign.pt'
    manifest = {'experiment_id': 'v58_conf_sweep_fold0', 'input_contract': probe['PILOT_CONTRACT'], 'runs': [row]}
    path = probe['CFG'].MANIFEST
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    with pytest.raises(ValueError, match='row'):
        manifest_cell(probe)
    assert not probe['calls']
    assert path.read_bytes() == original


def test_checkpoint_changed_after_contract_is_refused_before_yolo_load(probe):
    checkpoint = probe['checkpoint_map'](0)['prod']
    probe['PILOT_CONTRACT']['checkpoint_sha256']['prod'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.write_bytes(b'changed-after-contract')
    tree = ast.parse(builder.build_cells(0)[2]['source'])
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in {'file_sha256', 'require_checkpoint'}]
    probe['hashlib'] = hashlib
    execute(ast.unparse(ast.Module(body=helpers, type_ignores=[])), probe)
    tree = ast.parse(builder.build_cells(0)[5]['source'])
    collect = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'collect_cache')
    probe['YOLO'] = lambda path: pytest.fail('changed checkpoint must be refused before YOLO loads it')
    execute(ast.unparse(collect), probe)
    with pytest.raises(ValueError, match='checkpoint'):
        probe['collect_cache'](checkpoint, 'prod')


def test_partial_checkpoint_set_is_explicit_and_matching_resume_skips(probe):
    manifest_cell(probe)
    result = json.loads(probe['CFG'].MANIFEST.read_text())
    assert result['status'] == 'partial_checkpoint_set'
    assert result['missing_checkpoints'] == ['v57baseline_s3407']
    assert result['input_contract'] == probe['PILOT_CONTRACT']
    assert len(result['runs']) == 2
    assert result['runtime']['GPU'] == 'CPU-test-fake'
    assert result['runtime']['execution_scope'] == 'inference_only'
    assert result['runtime']['peak_vram_bytes'] == 123
    probe['calls'].clear()
    manifest_cell(probe)
    assert not probe['calls']


def test_all_checkpoints_absent_refused_without_creating_manifest(probe):
    next(iter(probe['checkpoint_map'](0).values())).unlink()
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        manifest_cell(probe)
    assert not probe['CFG'].MANIFEST.exists()


def test_failed_inference_is_recorded_without_fabricated_metrics(probe):
    def fail(path, label):
        raise RuntimeError('GPU inference failed')
    probe['collect_cache'] = fail
    with pytest.raises(RuntimeError, match='GPU inference failed'):
        manifest_cell(probe)
    result = json.loads(probe['CFG'].MANIFEST.read_text())
    assert result['runs'] == []
    assert result['status'] == 'failed'
    assert result['runtime']['completion'] == 'failed'
    assert result['runtime']['peak_vram_bytes'] == 123
    assert result['runtime']['duration_scope'] == 'current_inference_session'


def test_authoritative_preprocessing_refreshes_stale_local_pixels(tmp_path):
    source = tmp_path / 'source'
    local = tmp_path / 'local'
    source.mkdir()
    local.mkdir()
    (source / 'a.jpeg').write_bytes(b'authoritative')
    (local / 'a.jpeg').write_bytes(b'stale')
    tree = ast.parse(builder.build_cells(0)[2]['source'])
    chosen = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'file_sha256']
    chosen += [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'preprocessed_hashes' for t in node.targets)]
    chosen += [node for node in tree.body if isinstance(node, ast.For) and '前処理画像' in ast.unparse(node)]
    ns = {'CFG': SimpleNamespace(PREPROC_DIR=local, PREPROC_CACHE=source),
          'valid_files_fold': ['a.jpeg'], 'shutil': shutil, 'Path': Path, 'hashlib': hashlib,
          'tqdm': lambda values, **kwargs: values}
    execute(ast.unparse(ast.Module(body=chosen, type_ignores=[])), ns)
    assert (local / 'a.jpeg').read_bytes() == b'authoritative'
    assert ns['preprocessed_hashes']['a.jpeg'] == hashlib.sha256(b'authoritative').hexdigest()


@pytest.mark.parametrize('changed', ['annotations', 'folds', 'refiner', 'checkpoint', 'pixels', 'protocol'])
def test_contract_fingerprints_actual_consumed_inputs(tmp_path, changed):
    cfg_tree = ast.parse('\n'.join(line for line in builder.build_cells(0)[1]['source'].splitlines()
                                  if not line.lstrip().startswith('!')))
    ns = {'Path': Path}
    execute(ast.unparse(next(node for node in cfg_tree.body if isinstance(node, ast.ClassDef))), ns)
    cfg = ns['CFG']
    cfg.ARTIFACT_ROOT = tmp_path
    paths = {name: tmp_path / filename for name, filename in {
        'annotations': 'annotations.json', 'folds': 'grouped_folds.csv',
        'refiner': 'mediseg_refiner_fold0_resume.pt', 'checkpoint': 'prod.pt', 'pixels': 'val.jpeg'}.items()}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    ns.update({'hashlib': hashlib, 'json': json, 'sys': __import__('sys'), 'TRAIN_JSON': paths['annotations'],
               'checkpoint_map': lambda fold: {'prod': paths['checkpoint']},
               'preprocessed_hashes': {'val.jpeg': hashlib.sha256(paths['pixels'].read_bytes()).hexdigest()},
               'DENSE_CUT': 2.0, 'torch': SimpleNamespace(__version__='CPU-test'),
               'np': SimpleNamespace(__version__='CPU-test'), 'cv2': SimpleNamespace(__version__='CPU-test'),
               'version': lambda name: 'CPU-test'})
    tree = ast.parse(builder.build_cells(0)[2]['source'])
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'file_sha256']
    selected += [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'PILOT_CONTRACT' for t in node.targets)]
    source = ast.unparse(ast.Module(body=selected, type_ignores=[]))
    execute(source, ns)
    before = ns['PILOT_CONTRACT']
    if changed == 'protocol':
        cfg.PROTOCOL_SHA256 = 'updated-protocol'
    else:
        paths[changed].write_bytes(b'updated input')
        if changed == 'pixels':
            ns['preprocessed_hashes']['val.jpeg'] = hashlib.sha256(paths[changed].read_bytes()).hexdigest()
    execute(source, ns)
    assert ns['PILOT_CONTRACT'] != before
    assert ns['PILOT_CONTRACT']['schema_version'] == 1


@pytest.mark.parametrize('fold', [0, 1, 2])
def test_generated_notebook_parity_and_python_syntax(fold):
    notebook = json.loads((ROOT / f'solar_filament_v58_conf_sweep_fold{fold}_colab.ipynb').read_text(encoding='utf-8'))
    expected = builder.build_cells(fold)
    assert len(notebook['cells']) == len(expected)
    for index, (saved, generated) in enumerate(zip(notebook['cells'], expected)):
        assert saved['source'] == generated['source']
        assert saved['id'] == f'v58-confsweep-f{fold}-{index:02d}'
        if saved['cell_type'] == 'code':
            assert saved['outputs'] == [] and saved['execution_count'] is None
            ast.parse('\n'.join(line for line in saved['source'].splitlines() if not line.lstrip().startswith(('!', '%'))))


@pytest.mark.parametrize("fold", [1, 2])
def test_confirmation_requires_single_explicit_fold0_choice(fold):
    tree = ast.parse("\n".join(line for line in builder.build_cells(fold)[1]['source'].splitlines()
                              if not line.lstrip().startswith('!')))
    ns = {'Path': Path, 'math': __import__('math')}
    execute(ast.unparse(next(n for n in tree.body if isinstance(n, ast.ClassDef))), ns)
    assert ns['CFG'].CONF_SWEEP == ()
    execute(ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'confirmation_confidences')), ns)
    with pytest.raises(ValueError, match='fold0'):
        ns['confirmation_confidences']()
    ns['CFG'].SELECTED_CONF = 0.26
    assert ns['confirmation_confidences']() == (0.26,)
    ns['CFG'].SELECTED_CONF = float('nan')
    with pytest.raises(ValueError):
        ns['confirmation_confidences']()


def test_successful_retry_has_only_current_runtime_fields(probe):
    old = {'experiment_id': 'v58_conf_sweep_fold0', 'input_contract': probe['PILOT_CONTRACT'],
           'runs': [], 'runtime': {'error_type': 'RuntimeError',
                                  'peak_vram_unavailable_reason': 'old failure'}}
    probe['CFG'].MANIFEST.write_text(json.dumps(old))
    manifest_cell(probe)
    result = json.loads(probe['CFG'].MANIFEST.read_text())
    assert 'error_type' not in result['runtime']
    assert not result['runtime'].get('peak_vram_unavailable_reason')


def test_atomic_save_failure_preserves_completed_manifest(probe, monkeypatch):
    manifest_cell(probe)
    path = probe['CFG'].MANIFEST
    original = path.read_bytes()
    def fail_replace(source, destination):
        raise OSError('simulated interrupted replacement')
    monkeypatch.setattr(__import__('os'), 'replace', fail_replace)
    with pytest.raises(OSError):
        probe['save_probe_manifest']({'runs': []})
    assert path.read_bytes() == original
    assert not list(path.parent.glob('*.tmp'))
