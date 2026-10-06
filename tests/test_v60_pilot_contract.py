"""CPU acceptance of the copy-paste pilot; no fitting or downloads."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v60", ROOT / "scratch/build_v60_copypaste_pilot.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def function(name):
    source = "\n".join(c["source"] for c in BUILDER.build_cells(0) if c["cell_type"] == "code")
    tree = ast.parse("\n".join(l for l in source.splitlines() if not l.lstrip().startswith("!")))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def helpers(ns):
    ns.update(math=__import__("math"), hashlib=__import__("hashlib"))
    execute([function(name) for name in ["atomic_save", "file_sha256", "require_input_contract",
        "validate_run_row", "read_pilot_manifest", "reuse_baselines"]], ns)


def execute(nodes, ns):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "v60-contract", "exec"), ns)  # noqa: S102 - trusted generated CPU functions


@pytest.mark.parametrize("artifact", ["complete", "partial"])
def test_uncontracted_checkpoints_refused_before_model_load(tmp_path, artifact):
    cfg = SimpleNamespace(FOLD=0, RUNS_ROOT=tmp_path, CONFIGS=("baseline", "copypaste"), SEEDS=(3407,8407))
    run = tmp_path / "yolo_v60_baseline_s3407_fold0"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / ("best.pt" if artifact == "complete" else "last.pt")).write_bytes(b"legacy")
    if artifact == "complete":
        (run / "TRAINING_COMPLETE").write_text("old marker")
    ns = {"CFG": cfg, "Path": Path, "json": json, "PILOT_CONTRACT": {"fold":0},
          "build_fold_dataset": lambda: pytest.fail("legacy checkpoint must fail before dataset/model access"),
          "YOLO": lambda path: pytest.fail("must not load legacy model")}
    helpers(ns)
    execute([function("train_yolo")], ns)
    with pytest.raises(ValueError, match="contract"):
        ns["train_yolo"]("baseline", 3407)


def test_legacy_dataset_cannot_be_reused(tmp_path):
    cfg = SimpleNamespace(FOLD=0, YOLO_ROOT=tmp_path)
    root = tmp_path / "shared/fold_0"
    root.mkdir(parents=True)
    (root / "data.yaml").write_text("legacy dataset")
    ns = {"CFG": cfg, "Path": Path, "PILOT_CONTRACT": {"fold":0}, "json": json}
    helpers(ns)
    execute([function("build_fold_dataset")], ns)
    with pytest.raises(ValueError, match="contract"):
        ns["build_fold_dataset"]()


def test_legacy_baseline_is_not_adopted_or_relabelled(tmp_path):
    cfg = SimpleNamespace(FOLD=0, MANIFEST=tmp_path/'manifest.json', VERIFIED_BASELINE_MANIFEST=tmp_path/'v57.json',
                          REUSE_VERIFIED_BASELINE=False)
    old = {"config":"baseline", "seed":3407, "fold":2, "refined":{"pq":.9}}
    cfg.VERIFIED_BASELINE_MANIFEST.write_text(json.dumps({"runs":[old]}))
    source = next(c['source'] for c in BUILDER.build_cells(0) if '# [7/8]' in c['source'])
    prefix = source.split('manifest.pop')[0]
    ns = {"CFG":cfg, "json":json, "PILOT_CONTRACT":{"fold":0}}
    helpers(ns)
    exec(prefix, ns)  # noqa: S102 - trusted generated preflight without GPU loop
    assert ns['rows'] == []
    assert json.loads(cfg.VERIFIED_BASELINE_MANIFEST.read_text())['runs'][0] == old


def configuration():
    source = BUILDER.build_cells(0)[1]['source']
    tree = ast.parse('\n'.join(l for l in source.splitlines() if not l.lstrip().startswith('!')))
    ns = {'Path':Path}
    execute([next(n for n in tree.body if isinstance(n,ast.ClassDef))],ns)
    return ns['CFG']


def test_paired_training_arguments_differ_only_in_copy_paste():
    cfg = configuration()
    ns = {'CFG':cfg}
    execute([function('training_kwargs')],ns)
    baseline = ns['training_kwargs']('baseline',3407,Path('shared.yaml'),'same-run')
    treatment = ns['training_kwargs']('copypaste',3407,Path('shared.yaml'),'same-run')
    assert baseline['copy_paste'] == 0.0 and treatment['copy_paste'] == 0.2
    assert {k for k in baseline if baseline[k] != treatment[k]} == {'copy_paste'}
    assert baseline['data'] == treatment['data'] == 'shared.yaml'


@pytest.mark.parametrize('change',['annotations','folds','initial','refiner','pixels','protocol','recipe','runtime'])
def test_contract_fingerprints_actual_consumed_inputs(tmp_path, change):
    cfg = configuration()
    cfg.ARTIFACT_ROOT = cfg.PREPROC_DIR = tmp_path
    paths = {name:tmp_path/file for name,file in {'annotations':'annotations.json','folds':'grouped_folds.csv',
        'initial':'initial.pt','refiner':'mediseg_refiner_fold0_resume.pt','pixels':'pixel.jpeg'}.items()}
    for name,path in paths.items(): path.write_bytes(name.encode())
    ns = {'CFG':cfg,'json':json,'Path':Path,'PILOT_CONTRACT':{}, 'sys':__import__('sys'),
          'TRAIN_JSON':paths['annotations'],'INITIAL_YOLO_PATH':paths['initial'],
          'train_files_fold':['pixel.jpeg'],'valid_files_fold':[],
          'torch':SimpleNamespace(__version__='cpu-test'),'np':SimpleNamespace(__version__='cpu-test'),
          'cv2':SimpleNamespace(__version__='cpu-test'),'version':lambda name:'cpu-test'}
    helpers(ns)
    tree = ast.parse(BUILDER.build_cells(0)[2]['source'])
    nodes = [n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PILOT_CONTRACT' for t in n.targets)]
    execute(nodes,ns)
    before=ns['PILOT_CONTRACT']
    assert json.loads(json.dumps(before)) == before
    if change == 'protocol': cfg.PROTOCOL_SHA256='changed'
    elif change == 'recipe': cfg.COPY_PASTE=.3
    elif change == 'runtime': ns['version']=lambda name:'changed'
    else: paths[change].write_bytes(b'changed bytes')
    execute(nodes,ns)
    assert ns['PILOT_CONTRACT'] != before


@pytest.mark.parametrize('problem',['missing','changed','valid'])
def test_completion_marker_pins_actual_best_bytes(tmp_path,problem):
    import hashlib
    cfg=SimpleNamespace(FOLD=0,RUNS_ROOT=tmp_path,CONFIGS=('baseline','copypaste'),SEEDS=(3407,8407))
    contract={'fold':0,'config':'baseline','seed':3407}
    run=tmp_path/'yolo_v60_baseline_s3407_fold0'
    (run/'weights').mkdir(parents=True)
    best=run/'weights/best.pt'
    best.write_bytes(b'complete')
    (run/'INPUT_CONTRACT.json').write_text(json.dumps(contract))
    (run/'TRAINING_COMPLETE').write_text(json.dumps({'input_contract':contract,
        'checkpoint_sha256':hashlib.sha256(best.read_bytes()).hexdigest()}))
    if problem=='missing': best.unlink()
    if problem=='changed': best.write_bytes(b'changed')
    ns={'CFG':cfg,'Path':Path,'json':json,'PILOT_CONTRACT':{'fold':0}}
    helpers(ns)
    execute([function('train_yolo')],ns)
    if problem=='valid': assert ns['train_yolo']('baseline',3407)==best
    else:
        with pytest.raises(ValueError,match='checkpoint'):
            ns['train_yolo']('baseline',3407)


@pytest.mark.parametrize('fold',[0,1,2])
def test_generated_notebook_parity_and_syntax(fold):
    path=ROOT/f'solar_filament_v60_copypaste_fold{fold}_colab.ipynb'
    saved=json.loads(path.read_text(encoding='utf-8'))['cells']
    expected=BUILDER.build_cells(fold)
    assert len(saved)==len(expected)
    for actual,generated in zip(saved,expected):
        assert actual['source']==generated['source']
        if generated['cell_type']=='code':
            assert actual['outputs']==[] and actual['execution_count'] is None
            ast.parse('\n'.join(l for l in generated['source'].splitlines() if not l.lstrip().startswith('!')))


def test_atomic_manifest_interruption_preserves_previous_rows(tmp_path,monkeypatch):
    ns={'Path':Path,'json':json}
    helpers(ns)
    path=tmp_path/'manifest.json'
    path.write_bytes(b'previous completed rows')
    def fail_replace(source,target): raise OSError('interrupted')
    monkeypatch.setattr(__import__('os'),'replace',fail_replace)
    with pytest.raises(OSError,match='interrupted'): ns['atomic_save'](path,{'runs':[]})
    assert path.read_bytes()==b'previous completed rows'
    assert not list(tmp_path.glob('*.tmp'))


def valid_row_namespace(tmp_path):
    import hashlib
    cfg=SimpleNamespace(FOLD=0,CONFIGS=('baseline','copypaste'),SEEDS=(3407,8407),COPY_PASTE=.2,
        MANIFEST=tmp_path/'manifest.json',REUSE_VERIFIED_BASELINE=False)
    checkpoint=tmp_path/'best.pt'
    checkpoint.write_bytes(b'complete evaluated bytes')
    contract={'fold':0,'protocol':'current'}
    metrics={'pq':.4,'sq':.5,'rq':.8,'tp':4,'fp':1,'fn':2}
    row={'config':'baseline','seed':3407,'fold':0,'copy_paste':0.,'input_contract':contract,
        'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'raw':dict(metrics),'refined':dict(metrics),'fn_dense_val':1,'fn_sparse_val':1}
    ns={'CFG':cfg,'PILOT_CONTRACT':contract,'Path':Path,'json':json}
    helpers(ns)
    return ns,row


@pytest.mark.parametrize('problem',['legacy','wrong_fold','wrong_arm','wrong_seed','copy_paste','missing_metric',
    'nan','negative_count','fractional_count','missing_density','changed_checkpoint','duplicate'])
def test_unqualified_result_manifest_is_rejected_without_overwrite(tmp_path,problem):
    ns,row=valid_row_namespace(tmp_path)
    if problem=='legacy': row.pop('input_contract')
    if problem=='wrong_fold': row['fold']=2
    if problem=='wrong_arm': row['config']='dense2x'
    if problem=='wrong_seed': row['seed']=999
    if problem=='copy_paste': row['copy_paste']=.2
    if problem=='missing_metric': row['refined'].pop('sq')
    if problem=='nan': row['raw']['pq']=float('nan')
    if problem=='negative_count': row['refined']['fp']=-1
    if problem=='fractional_count': row['raw']['fn']=1.5
    if problem=='missing_density': row.pop('fn_dense_val')
    if problem=='changed_checkpoint': Path(row['checkpoint']).write_bytes(b'changed')
    rows=[row,dict(row)] if problem=='duplicate' else [row]
    manifest={'experiment_id':'v60_copypaste_fold0','input_contract':ns['PILOT_CONTRACT'],'runs':rows}
    path=ns['CFG'].MANIFEST
    path.write_text(json.dumps(manifest))
    before=path.read_bytes()
    with pytest.raises(ValueError): ns['read_pilot_manifest']()
    assert path.read_bytes()==before


def test_valid_manifest_and_explicit_baseline_roundtrip(tmp_path):
    ns,row=valid_row_namespace(tmp_path)
    manifest={'experiment_id':'v60_copypaste_fold0','input_contract':ns['PILOT_CONTRACT'],'runs':[row]}
    ns['CFG'].MANIFEST.write_text(json.dumps(manifest))
    assert ns['read_pilot_manifest']()==manifest
    ns['CFG'].REUSE_VERIFIED_BASELINE=True
    reused=ns['reuse_baselines'](manifest)
    assert reused[0]['fold']==0 and reused[0]['checkpoint_sha256']==row['checkpoint_sha256']
    assert row==manifest['runs'][0]
    with pytest.raises(ValueError,match='equivalence'): ns['reuse_baselines']({'runs':[row]})
