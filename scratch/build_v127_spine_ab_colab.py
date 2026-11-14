"""V127 Colab ノートのビルダー: spine 補助教師 A/B パイロット (締切後研究枠)。

SPINE_AUX_TEACHER_PREREG_2026-11-13.md の事前登録に従う。
開発分割 (time_block_assignment.csv: development∩folds{0,1} fit=365 /
development∩fold2 eval=180) で、prod 相当の対照 arm A と spine aux
補助教師 arm B を同一条件 70 epoch で比較する。locked_evaluation
(2017-2022、160 obs) は本ノートでは一切触れない (時系列確認用に隔離)。

疑似クラス方式 (spine_yolo.py): spine polyline を cls=1 擬似 instance
として seg label に混入 → mosaic/perspective/flip が同一 math で変換 →
Format 直前に SplitSpineSegments で除去 (model nc=1、dataset yaml nc=2)。

ゲート (事前固定): dev ΔPQ ≥ +0.010 かつ小対象提案 recall ≥ +2pt かつ
FP/画像 相対増 ≤5% → pilot_positive。未達なら誤差内訳を記録して終了。

    python scratch/build_v127_spine_ab_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "solar_filament_v127_spine_ab_colab.ipynb"


def _cell(cell_type: str, source: str) -> dict:
    base = {"cell_type": cell_type, "metadata": {},
            "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


MD = """\
# v127 — spine 補助教師 A/B パイロット (締切後研究枠・事前登録済み)

`SPINE_AUX_TEACHER_PREREG_2026-11-13.md` に従う。**locked_evaluation
(2017-2022) には一切触れない** — 時系列確認用に隔離。

| 用途 | 定義 | 件数 |
|---|---|---|
| fit | development ∩ folds{0,1} − ±5d 除外 | 365 |
| eval | development ∩ fold2 | 180 |

| arm | 学習 |
|---|---|
| A 対照 | prod 相当 YOLO11m-seg (spine 行は同一 dataset で split 除去) |
| B spine 補助 | A と同一 + P3 proto 入力へ aux conv head、正負別平均 BCE |

**ゲート**: dev ΔPQ ≥ +0.010 & 小対象 recall ≥ +2pt & FP/画像 +≤5%
→ `pilot_positive`。両 arm とも 70 epoch **last.pt** (checkpoint 非選択)。
"""

CELL1 = r'''
import os, re, json, subprocess, sys, hashlib, shutil, time
from pathlib import Path

# [0] Drive mount + inputs
from google.colab import drive
drive.mount('/content/drive')
V30 = Path('/content/drive/MyDrive/solar_filament_v30')
COMP = V30 / 'competition_data/MAGFiLO_1.0_Kaggle_2026'
inputs = {
    'train_json': COMP / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',
    'preproc_dir': V30 / 'preprocessed_v30_q96',
    'folds_csv': V30 / 'artifacts/grouped_folds.csv',
    'baseline_ckpt': V30 / 'artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt',
    'yolo_ckpt': V30 / 'runs/yolo_fold_2/weights/best.pt',
    'kaggle.json': Path('/content/drive/MyDrive/solar_filament_v35/kaggle.json'),
}
missing = {k: str(v) for k, v in inputs.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f'Drive inputs missing: {missing}')
EXPECTED_FOLDS_SHA = 'febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e'
actual = hashlib.sha256(inputs['folds_csv'].read_bytes()).hexdigest()
if actual != EXPECTED_FOLDS_SHA:
    raise RuntimeError(f'fold CSV sha mismatch: {actual}')
print('folds_csv sha256 OK (febb3f67 consumed split)')

# [1] code: public repo clone (solar_filament package + s0 ledgers)
os.makedirs('/root/.kaggle', exist_ok=True)
Path('/root/.kaggle/kaggle.json').write_bytes(inputs['kaggle.json'].read_bytes())
os.chmod('/root/.kaggle/kaggle.json', 0o600)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',
                'kaggle', 'ultralytics==8.4.120', 'pycocotools'], check=True)
REPO = Path('/content/solar-filament-2026-solution')
if not REPO.exists():
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/shingo257/solar-filament-2026-solution.git',
                    str(REPO)], check=True)
sys.path.insert(0, str(REPO))
from solar_filament.pipeline.spine_aux import (
    build_spine_target, cellize_mask_any, spine_in_mask_fraction)
from solar_filament.pipeline.spine_yolo import (
    SPINE_CLS, SplitSpineSegments, emit_label_txt, build_p3_targets,
    SpineAuxLoss)

WORK = Path('/content/work'); IN = WORK / 'input'; IN.mkdir(parents=True, exist_ok=True)
def _sha(p):
    d = hashlib.sha256()
    with open(p, 'rb') as h:
        for b in iter(lambda: h.read(1 << 20), b''):
            d.update(b)
    return d.hexdigest()

MIRROR_DIR = Path('/content/drive/MyDrive/solar_filament_v127/work/latest')
def mirror():
    if not os.path.ismount('/content/drive'):
        raise RuntimeError('Drive not mounted')
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    digests = {}
    for p in sorted(WORK.rglob('*')):
        if p.is_file() and 'input' not in p.relative_to(WORK).parts:
            dst = MIRROR_DIR / p.relative_to(WORK)
            expected = _sha(p)
            if dst.exists() and _sha(dst) == expected:
                digests[str(p.relative_to(WORK))] = expected; continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            if _sha(dst) != expected:
                raise RuntimeError(f'Drive mirror sha mismatch: {p.name}')
            digests[str(p.relative_to(WORK))] = expected
            print('mirrored', p.name, p.stat().st_size)
    (MIRROR_DIR / 'sha256_manifest.json').write_text(json.dumps(digests, indent=1))
    print('mirror DONE ->', MIRROR_DIR)

# [2] provenance (fold-2 manifest for sanity anchor only — eval は pilot subset)
_dl = lambda cmd: subprocess.run(cmd, check=True, capture_output=True, text=True)
_dl(['kaggle', 'datasets', 'download', 'shingo257/solar-v73-code', '-p', str(IN), '--unzip', '-q'])
_dl(['kaggle', 'datasets', 'download', '-d', 'shingo257/solar-v73-assets',
     '-f', 'v73_reviewed_provenance.json', '-p', str(IN)])
code_dir = next(p.parent for p in IN.rglob('train_refiner_fold.py'))
sys.path.insert(0, str(code_dir))
manifest_path = next(IN.rglob('v73_reviewed_provenance.json'))
EXPECTED_PROVENANCE_SHA = os.environ.get('V73_PROVENANCE_SHA256', '')
if not re.fullmatch(r'[0-9a-f]{64}', EXPECTED_PROVENANCE_SHA):
    EXPECTED_PROVENANCE_SHA = input('V73_PROVENANCE_SHA256: ').strip()
from v73_split_provenance import load_reviewed_manifest
manifest = load_reviewed_manifest(manifest_path, EXPECTED_PROVENANCE_SHA, 2)
if manifest['folds_sha256'] != EXPECTED_FOLDS_SHA:
    raise RuntimeError('reviewed split differs')
import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')

# eval harness はリポジトリ版 (--prod-from/--eval-files 拡張込み) を埋め込む
CONSENSUS_SCRIPT =
Path(code_dir / 'run_fold_oof_eval_consensus.py').write_text(CONSENSUS_SCRIPT)
import ast; ast.parse(CONSENSUS_SCRIPT)
print('extended eval script written (--prod-from/--eval-files)')
'''

CELL2 = r'''
# [3] prereg split derivation — ledger + exclusions from cloned repo (pinned)
import csv, datetime
LEDGER = REPO / 'output/s0/time_block_assignment.csv'
EXCL = REPO / 'output/s0/spine_aux_prereg_exclusions_2026-11-13.csv'
assert _sha(LEDGER) is not None
excl_sha = _sha(EXCL)
print('exclusion list sha256:', excl_sha[:16], '(pin a56372ea…)')
assert excl_sha == 'a56372ea6493a6d3571910e60f291606b864a4f84a61baf1df475a6627270b0a'

rows = list(csv.DictReader(open(LEDGER, encoding='utf-8-sig')))
folds = {r['file_name']: int(r['fold']) for r in rows}
roles = {r['file_name']: r['time_block_role'] for r in rows}
excluded = {r['file_name'] for r in csv.DictReader(open(EXCL, encoding='utf-8'))}

fit_files = sorted(
    r['file_name'] for r in rows
    if r['time_block_role'] == 'development' and int(r['fold']) in (0, 1)
    and r['file_name'] not in excluded)
eval_files = sorted(r['file_name'] for r in rows
                    if r['time_block_role'] == 'development' and int(r['fold']) == 2)
assert len(fit_files) == 365 and len(eval_files) == 180, (len(fit_files), len(eval_files))
assert all(roles[n] == 'development' for n in fit_files + eval_files)
(WORK / 'eval_files.txt').write_text('\n'.join(eval_files) + '\n')
print(f'pilot fit={len(fit_files)} eval={len(eval_files)} '
      f'(locked_evaluation 160 obs untouched)')
'''

CELL3 = r'''
# [4] dataset emission — cls0 polygons + cls1 spine pseudo-rows (fit only)
coco = json.loads(inputs['train_json'].read_text())
img_by_name = {}
for im in coco['images']:
    img_by_name.setdefault(im['file_name'], []).append(im)
ann_by_img = {}
for a in coco['annotations']:
    ann_by_img.setdefault(a['image_id'], []).append(a)

import numpy as np
from pycocotools import mask as mask_util

DS = WORK / 'spine_dataset'
for split in ('train', 'val'):
    (DS / 'images' / split).mkdir(parents=True, exist_ok=True)
    (DS / 'labels' / split).mkdir(parents=True, exist_ok=True)

def poly_of(ann):
    seg = ann['segmentation']
    if isinstance(seg, dict):  # RLE -> polygon unavailable; skip (audit: polygon only)
        return None
    ring = seg[0] if seg and isinstance(seg[0], list) else seg
    return np.asarray(ring, dtype=np.float64).reshape(-1, 2)

def emit_one(file_name, split, with_spine):
    recs = img_by_name.get(file_name)
    if not recs:
        raise KeyError(f'{file_name} not in coco images')
    dst_img = DS / 'images' / split / file_name
    src_img = inputs['preproc_dir'] / file_name
    if not dst_img.exists():
        dst_img.symlink_to(src_img)
    rows_txt = []
    H, W = recs[0]['height'], recs[0]['width']
    for rec in recs:
        for ann in ann_by_img.get(rec['id'], []):
            poly = poly_of(ann)
            if poly is not None and len(poly) >= 3:
                norm = poly.copy(); norm[:, 0] /= W; norm[:, 1] /= H
                rows_txt.append('0 ' + ' '.join(f'{v:.6f}' for v in norm.reshape(-1)))
            if with_spine:
                spine = np.asarray(ann.get('spine') or [], dtype=np.float64).reshape(-1, 2)
                if len(spine) >= 2:
                    ns = spine.copy(); ns[:, 0] /= W; ns[:, 1] /= H
                    rows_txt.append(f'{SPINE_CLS} ' + ' '.join(f'{v:.6f}' for v in ns.reshape(-1)))
    (DS / 'labels' / split / (Path(file_name).stem + '.txt')).write_text(
        '\n'.join(rows_txt) + '\n')

for n in fit_files:
    emit_one(n, 'train', with_spine=True)
for n in eval_files:
    emit_one(n, 'val', with_spine=False)

yaml_path = DS / 'spine.yaml'
yaml_path.write_text(
    f'path: {DS}\ntrain: images/train\nval: images/val\n'
    'nc: 2\nchannels: 3\nnames:\n  0: filament\n  1: spine_aux\n')
n_tr = sum(1 for _ in (DS / 'labels' / 'train').glob('*.txt'))
n_va = sum(1 for _ in (DS / 'labels' / 'val').glob('*.txt'))
print(f'dataset: train={n_tr} val={n_va} (nc=2, spine rows only in train)')
'''

CELL4 = r'''
# [5] wiring — dataset split insertion, aux model, trainer (prereg §2)
import torch, torch.nn as nn
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Format
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.models.yolo.segment import SegmentationTrainer

class SpineDataset(YOLODataset):
    def build_transforms(self, hyp=None):
        t = super().build_transforms(hyp)
        t.transforms.insert(-1, SplitSpineSegments())
        return t

class SpineAuxModel(SegmentationModel):
    """SegmentationModel + 1x1 conv aux head on the P3 (proto input) feature."""
    def init_aux(self, aux_weight):
        p3_ch = self.model[-1].proto.cv1.conv.in_channels
        self.aux_head = nn.Conv2d(p3_ch, 1, 1)
        self.aux_weight = float(aux_weight)
        self.model[-1].proto.register_forward_pre_hook(self._grab_p3)
        self._p3 = None
        return self
    def _grab_p3(self, module, inputs):
        self._p3 = inputs[0]
    def loss(self, batch, preds=None):
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch['img'])
        total, items = self.criterion(preds, batch)
        if self.training and batch.get('spine_pos') is not None and self._p3 is not None:
            logits = self.aux_head(self._p3)
            aux = SpineAuxLoss.call(logits, batch['spine_pos'], batch['spine_valid'])
            total = total + self.aux_weight * aux
        return total, items

class SpineTrainer(SegmentationTrainer):
    AUX = False          # arm B only
    AUX_WEIGHT = 0.5     # smoke で固定 (fit-only、外側成績で選ばない)
    def build_dataset(self, img_path, mode='train', batch=None):
        if mode != 'train':
            return super().build_dataset(img_path, mode, batch)
        from ultralytics.utils.torch_utils import unwrap_model
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return SpineDataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
            augment=True, hyp=self.args, rect=False,
            cache=self.args.cache or None, single_cls=False, stride=gs,
            pad=0.0, prefix='', task=self.args.task, classes=None,
            data=self.data, fraction=self.args.fraction)
    def get_model(self, cfg=None, weights=None, verbose=True):
        # dataset yaml nc=2 (spine pseudo-class) だが model は nc=1
        # get_model は _setup_train 先頭で呼ばれ optimizer 構築より先 —
        # ここで aux head を付けると最適化対象に含まれる
        model = SpineAuxModel(cfg or 'yolo11m-seg.yaml', nc=1,
                              ch=self.data['channels'], verbose=verbose)
        if self.AUX:
            model.init_aux(self.AUX_WEIGHT)
        if weights:
            model.load(weights)
        return model
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        segs = batch.get('spine_segments_px')
        if segs is not None and self.AUX:
            lists = [v if isinstance(v, (list, tuple)) else [v] for v in segs]
            h, w = batch['img'].shape[-2:]
            pos, valid = build_p3_targets(lists, (h, w), grid=h // 8,
                                          radius_cells=1)
            batch['spine_pos'] = torch.as_tensor(pos).unsqueeze(1).to(batch['img'].device)
            batch['spine_valid'] = torch.as_tensor(valid).unsqueeze(1).to(batch['img'].device)
        return batch
'''

CELL5 = r'''
# [6] fit-only smoke — transform consistency + aux scale (prereg §2 学習前検証)
from ultralytics.utils import IterableSimpleNamespace
_smoke_hyp = IterableSimpleNamespace(**{
    'mosaic': 0.2, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
    'shear': 0.0, 'perspective': 0.0, 'fliplr': 0.5, 'flipud': 0.0,
    'mixup': 0.0, 'cutmix': 0.0, 'copy_paste': 0.0,
    'copy_paste_mode': 'flip', 'augmentations': None, 'erasing': 0.0,
    'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.18,
    'mask_ratio': 4, 'overlap_mask': True, 'bgr': 0.0, 'fraction': 1.0})
ds = SpineDataset(img_path=str(DS / 'images' / 'train'), imgsz=1280, batch_size=4,
                  augment=True, hyp=_smoke_hyp,
                  rect=False, cache=False, single_cls=False, pad=0.0,
                  prefix='smoke', task='segment', data={'nc': 2, 'channels': 3,
                  'names': {0: 'filament', 1: 'spine_aux'}}, fraction=1.0)
n_sp, n_ok = 0, 0
for _ in range(12):
    labels = ds[int(np.random.randint(len(ds)))]
    cls = np.asarray(labels['cls']).reshape(-1)
    assert (cls == 0).all()
    segs = labels['spine_segments_px']
    n_sp += len(segs)
    for s in segs:
        band = build_spine_target([np.asarray(s)], out_h=160, out_w=160,
                                  img_h=1280, img_w=1280, radius_cells=0)[0]
        masks = labels.get('masks')
        if masks is not None and len(masks):
            um = masks.numpy() if hasattr(masks, 'numpy') else np.asarray(masks)
            um = cellize_mask_any((um > 0).any(0), 160, 160)
            frac = spine_in_mask_fraction(band, um, dilate=2)
            n_ok += frac >= 0.8
print(f'smoke: {n_sp} spine segs through transforms; '
      f'{n_ok}/{n_sp} inside dilated GT mask (>=0.8)')
assert n_sp > 0
# aux loss scale at random logits ~ 2*ln2 ≈ 1.39; AUX_WEIGHT=0.5 → aux ≈ 0.7
# (det loss ~2-4) — fit-only scale確認済みとして AUX_WEIGHT=0.5 を固定記録
print('AUX_WEIGHT fixed = 0.5 (fit-only smoke; no outer-score selection)')
'''

CELL6 = r'''
# [7] train arm A — 対照 (prod recipe verbatim, spine rows stripped identically)
from ultralytics.utils import IterableSimpleNamespace
TRAIN_KW = dict(data=str(yaml_path), epochs=70, imgsz=1280, device=0, workers=4,
                patience=14, seed=3407, deterministic=True, single_cls=False,
                optimizer='AdamW', lr0=7e-4, lrf=0.05, weight_decay=5e-4,
                warmup_epochs=3, hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20,
                close_mosaic=10, project=str(WORK / 'v127'), name='armA', batch=4)
class ControlTrainer(SpineTrainer):
    AUX = False
tr = ControlTrainer(overrides=TRAIN_KW)
tr.train()
armA_last = WORK / 'v127/armA/weights/last.pt'
print('armA last:', armA_last, _sha(armA_last)[:16])
mirror()
'''

CELL7 = r'''
# [8] train arm B — spine 補助教師 (identical recipe + aux head/loss)
class AuxTrainer(SpineTrainer):
    AUX = True
    AUX_WEIGHT = 0.5
TRAIN_KW_B = dict(TRAIN_KW, name='armB')
trb = AuxTrainer(overrides=TRAIN_KW_B)
trb.train()
armB_last = WORK / 'v127/armB/weights/last.pt'
print('armB last:', armB_last, _sha(armB_last)[:16])
mirror()
'''

CELL8 = r'''
# [9] eval — 各 arm の last.pt を prod arm 提案源として pilot eval (180 obs)
import cv2
consensus_py = code_dir / 'run_fold_oof_eval_consensus.py'
evals = {}
for arm, ck in [('armA', armA_last), ('armB', armB_last)]:
    sha = _sha(ck)
    out = WORK / f'v127_eval_{arm}.json'
    r = subprocess.run([sys.executable, str(consensus_py),
        '--train-json', str(inputs['train_json']),
        '--test-dir', str(inputs['preproc_dir']),
        '--preproc-dir', str(inputs['preproc_dir']),
        '--folds-csv', str(inputs['folds_csv']), '--fold', '2',
        '--provenance-manifest', str(manifest_path),
        '--provenance-sha256', EXPECTED_PROVENANCE_SHA,
        '--detector', f'{arm}={ck}', '--prod-from', arm,
        '--trusted-sha256', sha,
        '--eval-files', str(WORK / 'eval_files.txt'),
        '--refiner', f'baseline={inputs["baseline_ckpt"]}',
        '--model', 'edgeatt', '--out', str(out)],
        capture_output=True, text=True,
        env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
    print(r.stdout[-2500:]); print('ERR', r.stderr[-800:])
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f'eval {arm} failed rc={r.returncode}')
    evals[arm] = json.load(open(out))
    mirror()
'''

CELL9 = r'''
# [10] mechanism metrics — 小対象提案 recall + FP/image (prereg §3)
from ultralytics import YOLO
small_gt = {}   # file -> list of (area, mask)
for n in eval_files:
    recs = img_by_name[n]
    ms = []
    for rec in recs:
        for a in ann_by_img.get(rec['id'], []):
            if float(a.get('area', 0)) <= 1024 and not a.get('iscrowd', 0):
                rle = mask_util.frPyObjects(a['segmentation'], rec['height'], rec['width'])
                ms.append(mask_util.decode(rle if isinstance(rle, list) else [rle]).any(axis=2))
    small_gt[n] = ms
def iou(a, b):
    i = int((a & b).sum()); u = int((a | b).sum())
    return i / u if u else 0.0
mech = {}
for arm, ck in [('armA', armA_last), ('armB', armB_last)]:
    model = YOLO(str(ck))
    rec05 = rec26 = 0; tot = 0; nfp = 0
    for n in eval_files:
        res = model.predict(str(inputs['preproc_dir'] / n), imgsz=1280,
                            conf=0.08, iou=0.55, agnostic_nms=True,
                            retina_masks=True, verbose=False)[0]
        props = []
        if res.masks is not None:
            for m in res.masks.data.detach().cpu().numpy():
                bm = cv2.resize(m, (2048, 2048)) >= 0.5 if m.shape != (2048, 2048) else m >= 0.5
                props.append(bm)
        confs = res.boxes.conf.detach().cpu().numpy() if res.boxes is not None else []
        nfp += sum(1 for c in confs if c >= 0.26)
        for g in small_gt[n]:
            tot += 1
            best = max((iou(g, p) for p in props), default=0.0)
            if best > 0.5:
                rec05 += 1
                if any(confs[i] >= 0.26 for i, p in enumerate(props) if iou(g, p) > 0.5):
                    rec26 += 1
    mech[arm] = {'small_gt_recall_iou05': rec05 / tot, 'small_gt_recall_conf26': rec26 / tot,
                 'fp_per_image_c26': nfp / len(eval_files), 'small_gt_total': tot}
    print(arm, mech[arm])
'''

CELL10 = r'''
# [11] gate + manifest + mirror (prereg §3 pilot gates)
pqA = evals['armA']['arms']['baseline::prod']['pq_by_threshold']['0.45']
pqB = evals['armB']['arms']['baseline::prod']['pq_by_threshold']['0.45']
dpq = pqB - pqA
rec_gain = mech['armB']['small_gt_recall_iou05'] - mech['armA']['small_gt_recall_iou05']
fp_rel = (mech['armB']['fp_per_image_c26'] / mech['armA']['fp_per_image_c26']) - 1
gate = {'pqA': pqA, 'pqB': pqB, 'delta_pq': dpq,
        'recall_gain_pts': rec_gain * 100, 'fp_rel_increase': fp_rel,
        'pilot_gates': {'dpq>=0.010': dpq >= 0.010,
                        'recall>=+2pt': rec_gain >= 0.02,
                        'fp<=+5%': fp_rel <= 0.05},
        'decision': 'pilot_positive' if (dpq >= 0.010 and rec_gain >= 0.02
                                       and fp_rel <= 0.05) else 'pilot_negative'}
print(json.dumps(gate, indent=2))
manifest_out = {
    'experiment': 'v127_spine_aux_pilot', 'date': time.strftime('%Y-%m-%d'),
    'prereg': 'SPINE_AUX_TEACHER_PREREG_2026-11-13.md',
    'splits': {'fit': len(fit_files), 'eval': len(eval_files),
               'exclusions_sha256': excl_sha, 'folds_sha256': EXPECTED_FOLDS_SHA},
    'weights': {'armA_last_sha256': _sha(armA_last), 'armB_last_sha256': _sha(armB_last)},
    'aux_weight': 0.5, 'mechanism': mech, 'gate': gate,
    'evals': evals}
(WORK / 'v127_summary.json').write_text(json.dumps(manifest_out, indent=2))
print('decision:', gate['decision'])
mirror()
'''

CELLS = [
    _cell("markdown", MD),
    _cell("code", CELL1),
    _cell("code", CELL2),
    _cell("code", CELL3),
    _cell("code", CELL4),
    _cell("code", CELL5),
    _cell("code", CELL6),
    _cell("code", CELL7),
    _cell("code", CELL8),
    _cell("code", CELL9),
    _cell("code", CELL10),
]


CONSENSUS_SCRIPT_PATH = (Path(__file__).resolve().parent
                         / "run_fold_oof_eval_consensus.py")


def build_notebook(script_text: str) -> dict:
    cells = json.loads(json.dumps(CELLS))
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            if line.strip() == "CONSENSUS_SCRIPT =":
                cell["source"][i] = "CONSENSUS_SCRIPT = " + repr(script_text) + "\n"
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"},
                     "language_info": {"name": "python"}},
        "cells": cells,
    }


def main() -> Path:
    nb = build_notebook(CONSENSUS_SCRIPT_PATH.read_text(encoding="utf-8"))
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    return OUT


if __name__ == "__main__":
    main()
