"""Embedded cell payloads for the build_v92_cellpose_flow notebook builder.

Derived from build_v90_flowprop_cells.py: v92 は Cellpose 型 flow-field
head (fg logit + 各画素→インスタンス重心への unit flow) を持つ
semantic U-Net (smp ConvNeXt-T, 1024px crop, (file,annotator)
エントリ単位) を学習し、flow 追従→端点クラスタリングで個体提案を
生成する。v90 の connected-components blob と違い、各画素が 1 つの
中心に束縛されるため merged instance の分離を原理で解く。
arm `flowprop` = flow 提案のみ、`union` = prod ∪ flow。
採点は prod と同一経路 (refine→ownership→公式PQ)、診断に
「prod の primary FN を flow 提案が救済した数」を記録する。
"""


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


MEDISEG = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


cells = [
    markdown(
        """# v92 — Cellpose flow-field 提案源: 個体分離 + blind-FN 救済

arm: `union` (prod ∪ flow) + `flowprop` (flow のみ) + paired `prod`

**仮説** (§2.51): v90/v91 で free-blob pixel 提案は FP 爆発だったが、
Cellpose 型 flow-field は各画素を必ず 1 つのインスタンス中心に束縛
する。merged instance の分離を box-NMS と別原理で解き、かつ
blob より厳格な提案を生成できるはず。

**flow モデル**: smp U-Net (`tu-convnext_tiny`, ImageNet)、
1024px crop・(file,annotator) エントリ単位、出力 3ch
(fg logit + flow_y + flow_x)。flow 教師 = 各画素→所属 instance
重心への unit ベクトル。4-flip TTA (flow は符号反転して反転解除)。
推論: fg>=閾値 の画素を flow で Euler 追従 (~120 step)→端点を
DBSCAN (eps=6px) でクラスタ → 各クラスタの画素集合を 1 instance。

**統合ルール**: det 提案 (conf>=閾値) + flow 提案 (conf>=閾値 かつ
mask-IoU>0.5 の det 提案に重ならないもの) → 同一 refine→ownership→PQ。

**事前固定ゲート** (RESEARCH_LOG §2.52):

- `PQ(union,0.26) − PQ(prod,0.26) >= +0.005`: fold0/1 に昇格検討
- `[0,+0.005)`: reject_noise、`<0`: reject
- 診断 (判定不使用): prod の primary FN が flow 提案で救済された数

GPU必須。A100推奨 (~1.5h 学習 + flow 追従 ~15分 + 評価)。
"""
    ),
    code(
        """# [1/6] 依存・Drive・設定
%pip install -q "ultralytics>=8.3,<9" pycocotools segmentation-models-pytorch
import hashlib, json, math, os, random, shutil, sys, tempfile, time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from google.colab import drive
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from ultralytics import YOLO

drive.mount("/content/drive")

class CFG:
    SEED = 3407
    FOLD = 2
    ARMS = ('flowprop', 'union')
    # Drive artifacts/grouped_folds.csv (707 obs, file_name/fold) の凍結 SHA256
    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v92_cellpose_flow"
    YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v92_cellpose_flow"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v92")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    PROD_YOLO = DRIVE_ROOT / "runs" / f"yolo_fold_{FOLD}" / "weights" / "best.pt"
    REFINER_CKPT = ARTIFACT_ROOT / f"mediseg_edgeattnet_refiner_fold{FOLD}_resume.pt"
    MANIFEST = ARTIFACT_ROOT / f"v92_cellpose_flow_fold{FOLD}_manifest.json"
    FLOW_CKPT = RUNS_ROOT / f"flowseg_fold{FOLD}.pt"
    FLOW_MAPS = RUNS_ROOT / f"maps_fold{FOLD}"
    # --- Cellpose flow U-Net (v90 改造: §2.52) ---
    FLOW_ENCODER = "tu-convnext_tiny"
    FLOW_CROP = 1024
    FLOW_BATCH = 8          # A100 40GB 向け
    FLOW_LR = 2e-4
    FLOW_TRAIN_HOURS = 1.2  # 時間予算 (v90 と同条件)
    FLOW_LOSS_W = 1.0       # flow L1 損失の重み (fg 損失に対する比)
    FLOW_PROP_THRESH = 0.5  # fg 確率の提案化閾値
    FLOW_DEDUP_IOU = 0.5    # union で det 提案との重複判定
    FLOW_NITER = 200        # Euler 追従の反復回数 (長大 filament の重心まで ~150px あり得る)
    FLOW_STEP = 1.0         # 1 step の移動量 (px)
    FLOW_EPS = 6.0          # 端点クラスタ DBSCAN の eps (px)
    FLOW_MIN_SAMPLES = 8    # DBSCAN min_samples
    # --- YOLO (prod det 評価のみ。再学習はしない) ---
    YOLO_IMGSZ = 1280
    # --- 評価 (run_fold_oof_eval.py / v87 と同一) ---
    PREDICT_CONF_FLOOR = 0.08   # refine 対象の下限 (prod 経路と同一)
    DIAG_CONF_FLOOR = 0.02      # 診断のみ: 生提案をこの下限まで記録
    PREDICT_IOU = 0.55
    CONF_SWEEP = (0.15, 0.2, 0.26, 0.32)
    PRIMARY_CONF = 0.26
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ = 0.444238
    GATE_DELTA = 0.005
    SPARSE_R = 0.77

for path in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.RUNS_ROOT, CFG.YOLO_ROOT, CFG.FLOW_MAPS):
    path.mkdir(parents=True, exist_ok=True)

# fold 別 refiner ckpt 名が予定名と違う場合に備え、glob でフォールバック解決
if not CFG.REFINER_CKPT.exists():
    _cands = sorted(CFG.ARTIFACT_ROOT.glob(f"*refiner*fold{CFG.FOLD}*.pt"))
    if _cands:
        CFG.REFINER_CKPT = _cands[0]
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
required = {"competition data": CFG.DATA_CACHE, "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
            "前処理画像": CFG.PREPROC_CACHE, "prod YOLO": CFG.PROD_YOLO, "EdgeAtt refiner": CFG.REFINER_CKPT}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提 artifact が見つからない: {missing}")
assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
RUNTIME_INFO = {"started_utc": datetime.now(timezone.utc).isoformat(), "GPU": torch.cuda.get_device_name(0),
                "python": sys.version, "torch": torch.__version__, "ultralytics": version("ultralytics"),
                "pycocotools": version("pycocotools"), "opencv": cv2.__version__, "numpy": np.__version__}
print(RUNTIME_INFO)
"""
    ),
    code(
        """# [2/6] COCO・凍結 fold (SHA256 検証)・前処理画像
import csv, io, re

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

TRAIN_JSON = next(Path(CFG.DATA_CACHE).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
coco = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)
train_files = sorted(file_to_ids)

folds_path = CFG.ARTIFACT_ROOT / "grouped_folds.csv"
folds_sha256 = file_sha256(folds_path)
if folds_sha256 != CFG.GROUPED_FOLDS_SHA256:
    raise ValueError(f"grouped_folds.csv の SHA256 が不一致: {folds_sha256}")
fold_frame = pd.read_csv(folds_path)
if fold_frame["file_name"].duplicated().any() or set(fold_frame["file_name"]) != set(train_files):
    raise ValueError("fold CSV が全観測を一意にカバーしていない")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
valid_files = [n for n in train_files if fold_of_file[n] == CFG.FOLD]
fit_files = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
# semantic 学習は (file, annotator) エントリ単位: 同一観測を複数注釈者が
# 別サンプルとして扱う (biconcavelens と同じ)。val は観測全体で採点するので
# 早期停止監視用 val は使わず、時間予算で全 fit_entries を回す。
fit_entries = [(f, iid) for f in fit_files for iid in file_to_ids[f]
               if annotations[iid]]

for name in tqdm(train_files, desc="前処理画像をローカルへ"):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)

def to_rle(annotation, height, width):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_utils.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

gt_rles = {image_id: [to_rle(a, int(info["height"]), int(info["width"]))
                      for a in annotations[image_id] if not a.get("iscrowd", 0)]
           for image_id, info in images.items()}
print({"観測": len(train_files), "fit": len(fit_files), "val(OOF)": len(valid_files),
       "folds_sha256": folds_sha256[:16]})
"""
    ),
    code(
        """# [3/6] flow-field U-Net 学習 (§2.52: fg + 重心方向 unit flow, 1024px crop・(file,annotator) 単位)
def render_instances(image_id):
    \"\"\"image_id (file,annotator) の annotation を instance label map (uint16) へ。
    各 annotation を一意 id で fill する (1 annotation = 1 instance)。\"\"\"
    info = images[image_id]
    height, width = int(info["height"]), int(info["width"])
    inst = np.zeros((height, width), np.uint16)
    iid = 0
    for ann in annotations[image_id]:
        if ann.get("iscrowd", 0):
            continue
        iid += 1
        for poly in ann.get("segmentation", []):
            if len(poly) >= 6:
                cv2.fillPoly(inst, [np.asarray(poly, np.int32).reshape(-1, 2)], iid)
    return inst

def instance_centroids(inst):
    \"\"\"label map → {inst_id: (cy, cx)}。重心はフル画像の instance 全体で計算
    するため crop 位置に依らず flow 教師が安定する。\"\"\"
    ys, xs = np.nonzero(inst)
    ids, inv = np.unique(inst[ys, xs], return_inverse=True)
    n = np.bincount(inv)
    cy = np.bincount(inv, weights=ys) / n
    cx = np.bincount(inv, weights=xs) / n
    return {int(i): (float(a), float(b)) for i, a, b in zip(ids, cy, cx)}

class FlowSegDataset(torch.utils.data.Dataset):
    \"\"\"fit_entries 各エントリから 1024px crop をオンデマンド生成。
    75%: filament 画素中心 + jitter、25%: 完全ランダム。
    出力 (x[1,c,c], y[3,c,c]) — y = (fg, flow_y, flow_x)。
    flow は所属 instance 全体重心への unit ベクトル (bg=0)。
    rot90/flip は flow 場にも空間変換 + ベクトル成分変換を掛ける。\"\"\"
    def __init__(self, entries, crop):
        self.entries, self.crop, self._cache = entries, crop, {}

    def __len__(self):
        return len(self.entries)

    def _load(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        file_name, image_id = self.entries[idx]
        image = cv2.imread(str(CFG.PREPROC_DIR / file_name), cv2.IMREAD_GRAYSCALE)
        inst = render_instances(image_id)
        cents = instance_centroids(inst)
        if len(self._cache) < 512:  # 画像 ~4MB + inst ~8MB
            self._cache[idx] = (image, inst, cents)
        return image, inst, cents

    def __getitem__(self, idx):
        rng = np.random.default_rng((CFG.SEED, idx, os.getpid()))
        image, inst, cents = self._load(idx)
        height, width = image.shape
        crop = self.crop
        ys, xs = np.where(inst)
        if len(xs) and rng.random() < 0.75:
            i = rng.integers(len(xs))
            jitter = crop // 4
            cx = int(xs[i]) + rng.integers(-jitter, jitter + 1)
            cy = int(ys[i]) + rng.integers(-jitter, jitter + 1)
        else:
            cx, cy = rng.integers(0, width), rng.integers(0, height)
        x0, y0 = int(cx) - crop // 2, int(cy) - crop // 2
        pad = [(max(0, -y0), max(0, y0 + crop - height)),
               (max(0, -x0), max(0, x0 + crop - width))]
        sl = np.s_[max(0, y0):min(height, y0 + crop), max(0, x0):min(width, x0 + crop)]
        x, ic = image[sl], inst[sl]
        if any(pad[0]) or any(pad[1]):
            x = np.pad(x, pad, mode="edge"); ic = np.pad(ic, pad, mode="constant")
        # flow 教師: padded 行 p は画像行 y0+p に対応 → 重心座標も y0 分ずらす
        fy = np.zeros((crop, crop), np.float32); fx = np.zeros((crop, crop), np.float32)
        pys, pxs = np.nonzero(ic)
        if len(pys):
            lab = ic[pys, pxs]
            cyv = np.array([cents[int(l)][0] for l in lab], np.float32) - y0
            cxv = np.array([cents[int(l)][1] for l in lab], np.float32) - x0
            dy, dx = cyv - pys, cxv - pxs
            d = np.hypot(dy, dx)
            ok = d > 1e-3
            fy[pys[ok], pxs[ok]] = dy[ok] / d[ok]
            fx[pys[ok], pxs[ok]] = dx[ok] / d[ok]
        fg = (ic > 0).astype(np.float32)
        k = rng.integers(4)
        x, fg = np.rot90(x, k), np.rot90(fg, k)
        fy, fx = np.rot90(fy, k), np.rot90(fx, k)
        if k == 1: fy, fx = -fx, fy
        elif k == 2: fy, fx = -fy, -fx
        elif k == 3: fy, fx = fx, -fy
        if rng.random() < 0.5:
            x, fg, fy, fx = np.fliplr(x), np.fliplr(fg), np.fliplr(fy), np.fliplr(fx)
            fx = -fx
        xf = x.astype(np.float32) / 255.0
        gamma = rng.uniform(0.7, 1.4)
        xf = np.clip(xf ** gamma * rng.uniform(0.8, 1.2) + rng.uniform(-0.05, 0.05), 0, 1)
        y = np.stack([fg, fy, fx])
        return torch.from_numpy(xf[None].copy()), torch.from_numpy(y.copy())

def soft_dice_loss(logit, target, eps=1e-6):
    prob = torch.sigmoid(logit).flatten(1)
    t = target.flatten(1)
    inter = (prob * t).sum(1)
    return (1 - (2 * inter + eps) / (prob.sum(1) + t.sum(1) + eps)).mean()

_FOCAL = None

def flowseg_loss(pred, y):
    \"\"\"pred (B,3,H,W) = (fg logit, fy, fx)、y = (fg, fy_t, fx_t)。
    fg: 10*focal+dice (v90 と同式)。flow: fg 画素上の smooth-L1 x FLOW_LOSS_W。\"\"\"
    global _FOCAL
    if _FOCAL is None:
        _FOCAL = smp.losses.FocalLoss(mode="binary", gamma=2.0, alpha=0.25)
    loss = 10.0 * _FOCAL(pred[:, :1], y[:, :1]) + soft_dice_loss(pred[:, :1], y[:, :1])
    m = y[:, 0] > 0.5
    if m.any():
        fl = (torch.nn.functional.smooth_l1_loss(pred[:, 1][m], y[:, 1][m]) +
              torch.nn.functional.smooth_l1_loss(pred[:, 2][m], y[:, 2][m]))
        loss = loss + CFG.FLOW_LOSS_W * fl
    return loss

def train_flowseg():
    done_path = CFG.FLOW_CKPT.parent / f"TRAINING_COMPLETE_fold{CFG.FOLD}"
    stats = {"train_entries": len(fit_entries), "encoder": CFG.FLOW_ENCODER,
             "crop": CFG.FLOW_CROP, "batch": CFG.FLOW_BATCH, "classes": 3,
             "flow_loss_w": CFG.FLOW_LOSS_W}
    if done_path.exists() and CFG.FLOW_CKPT.exists():
        print(f"flowseg: 学習済み checkpoint を再利用 {CFG.FLOW_CKPT}")
        return CFG.FLOW_CKPT, stats
    dataset = FlowSegDataset(fit_entries, CFG.FLOW_CROP)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=CFG.FLOW_BATCH, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True)
    net = smp.Unet(CFG.FLOW_ENCODER, encoder_weights="imagenet",
                   in_channels=1, classes=3).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=CFG.FLOW_LR, weight_decay=1e-4)
    budget = CFG.FLOW_TRAIN_HOURS * 3600
    started, epoch, epoch_secs = time.time(), 0, []
    bar = tqdm(desc=f"flowseg fold{CFG.FOLD} 学習")
    while True:
        epoch += 1
        ep_start = time.time()
        running, seen = 0.0, 0
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = flowseg_loss(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            running += loss.item() * len(x); seen += len(x)
            elapsed = time.time() - started
            bar.update(1); bar.set_postfix(ep=epoch, loss=round(loss.item(), 4),
                                          elapsed=f"{elapsed / 60:.0f}m")
            if elapsed > budget:
                break
        epoch_secs.append(time.time() - ep_start)
        torch.save({"model": net.state_dict(), "epoch": epoch},
                   CFG.FLOW_CKPT)  # 毎 epoch 上書き (last.ckpt 運用)
        if time.time() - started > budget:
            break
    bar.close()
    stats["epochs"] = epoch
    stats["sec_per_epoch"] = round(float(np.mean(epoch_secs)), 1)
    stats["train_minutes"] = round((time.time() - started) / 60, 1)
    done_path.write_text("ok", encoding="utf-8")
    print({"flowseg_epochs": epoch, "sec_per_epoch": stats["sec_per_epoch"],
           "train_min": stats["train_minutes"], "last_loss": round(running / max(seen, 1), 4)})
    return CFG.FLOW_CKPT, stats
"""
    ),
    code(
        """# [4/6] refiner モデル定義 (solar_filament/models/mediseg_models.py と同一)
"""
        + MEDISEG
    ),
    code(
        """# [5/6] 評価関数 (run_fold_oof_eval.py / v87 と同一経路 + 低conf診断)
def sparse_label_pq(sum_iou, tp, fp, fn, r=0.77):
    \"\"\"dariushafshar の sparse-label 換算 (RESEARCH_LOG §2.28)。r=1 で通常の PQ。\"\"\"
    denominator = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return float(r * sum_iou / denominator) if denominator > 0 else 0.0

refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
_ckpt = torch.load(CFG.REFINER_CKPT, map_location="cuda", weights_only=False)
refiner.load_state_dict(_ckpt.get("best_model") or _ckpt["model"], strict=True)
del _ckpt
REFINER_SHA256 = file_sha256(CFG.REFINER_CKPT)

def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    out = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        out.append(mask >= 0.5)
    conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)
    return out, conf

def mask_square_box(mask, context=CFG.CROP_CONTEXT, minimum=96):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, width, height
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > width: x0 -= x1 - width; x1 = width
    if y1 > height: y0 -= y1 - height; y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)

@torch.no_grad()
def refine_proposal(image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal

def drop_contained_indices(masks, threshold=CFG.DROP_THRESHOLD):
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = mask_utils.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))
    areas = np.asarray(mask_utils.area(rles), dtype=np.float64)
    self_iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float64)
    dropped = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in dropped or j in dropped or self_iou[i, j] <= 0:
                continue
            inter = self_iou[i, j] * (areas[i] + areas[j]) / (1.0 + self_iou[i, j])
            smaller = i if areas[i] <= areas[j] else j
            if inter / max(areas[smaller], 1.0) >= threshold:
                dropped.add(smaller)
    return [i for i in range(len(masks)) if i not in dropped]

def apply_ownership(masks):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    occupied = np.zeros(masks[0].shape, dtype=bool)
    kept = []
    for index in sorted(drop_contained_indices(masks), key=lambda i: areas[i]):
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < CFG.MIN_AREA:
            continue
        kept.append(exclusive); occupied |= exclusive
    return kept

class PanopticQuality:  # self-evaluation notebook 準拠 (閾値越えを全て数える)
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold, self.tp_ious, self.fp_count, self.fn_count = iou_threshold, [], 0, 0
    def add_matrix(self, matrix, n_gt, n_pred):
        if n_gt == 0:
            self.fp_count += n_pred; return
        if n_pred == 0:
            self.fn_count += n_gt; return
        hits = matrix > self.iou_threshold
        self.tp_ious.extend(matrix[hits].tolist())
        self.fp_count += int((hits.sum(axis=0) == 0).sum())
        self.fn_count += int((hits.sum(axis=1) == 0).sum())
    @property
    def score(self):
        d = len(self.tp_ious) + 0.5 * self.fp_count + 0.5 * self.fn_count
        return float(sum(self.tp_ious) / d) if d > 0 else 0.0
    def summary(self):
        tp, s = len(self.tp_ious), float(sum(self.tp_ious))
        return {"pq": self.score, "sq": s / tp if tp else 0.0, "tp": tp, "fp": self.fp_count,
                "fn": self.fn_count, "sum_iou": s,
                "pq_sparse_r077": sparse_label_pq(s, tp, self.fp_count, self.fn_count, CFG.SPARSE_R)}

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

def lowconf_coverage(raw_rles, raw_conf, truth_rles):
    \"\"\"primary 未満の生提案が truth を IoU>0.5 で覆う数 (診断のみ)。\"\"\"
    if not truth_rles:
        return 0
    band = [r for r, c in zip(raw_rles, raw_conf) if CFG.DIAG_CONF_FLOOR <= c < CFG.PRIMARY_CONF]
    if not band:
        return 0
    hits = np.asarray(mask_utils.iou(band, truth_rles, [0] * len(truth_rles)), dtype=np.float64).T
    return int((hits > 0.5).any(axis=1).sum())

def evaluate_yolo(weights_path, label):
    \"\"\"floor 推論 1 回 → conf>=0.08 の提案のみ refine (prod 経路と同一) →
    conf sweep はメモリ上で採点。conf [0.02,0.26) の生提案は診断用に保持。\"\"\"
    model = YOLO(str(weights_path))
    cache = []
    for name in tqdm(valid_files, desc=f"{label} 推論+refine"):
        path = CFG.PREPROC_DIR / name
        result = model.predict(str(path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.DIAG_CONF_FLOOR,
                               iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, conf = result_masks(result)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        raw_rles = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
        refined = [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8)))
                   if conf[i] >= CFG.PREDICT_CONF_FLOOR else None for i, m in enumerate(masks)]
        cache.append((name, conf, [float(m.sum()) for m in masks], raw_rles, refined))
    del model; torch.cuda.empty_cache()
    out, diag = {}, {"fn_at_primary": 0, "fn_covered_by_lowconf_prop": 0}
    for c in CFG.CONF_SWEEP:
        meter = PanopticQuality()
        n_pred = 0
        for name, conf, areas, raw_rles, refined in cache:
            keep = [i for i in range(len(conf)) if conf[i] >= c and areas[i] >= CFG.MIN_AREA]
            preds = apply_ownership([mask_utils.decode(refined[i]) != 0 for i in keep])
            pred_rles = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
            n_pred += len(pred_rles)
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
        out[str(c)] = {**meter.summary(), "preds_per_image": n_pred / max(len(valid_files), 1)}
        print(label, c, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[str(c)].items()})
    # 診断: primary conf で FN となった GT のうち低conf生提案で覆える数
    for name, conf, areas, raw_rles, refined in cache:
        keep = [i for i in range(len(conf)) if conf[i] >= CFG.PRIMARY_CONF and areas[i] >= CFG.MIN_AREA]
        pred_masks = [mask_utils.decode(refined[i]) != 0 for i in keep]
        pred_rles = mask_utils.encode(np.asfortranarray(np.stack(pred_masks, -1).astype(np.uint8))) if pred_masks else []
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            if not truth:
                continue
            hits = iou_matrix(truth, pred_rles) > 0.5 if pred_rles else np.zeros((len(truth), 0))
            fn_idx = np.where(hits.sum(axis=1) == 0)[0] if pred_rles else np.arange(len(truth))
            if len(fn_idx):
                diag["fn_at_primary"] += len(fn_idx)
                diag["fn_covered_by_lowconf_prop"] += lowconf_coverage(
                    raw_rles, conf, [truth[i] for i in fn_idx])
    out["diag_lowconf"] = diag
    print(label, "diag:", diag)
    return out

# --- Cellpose flow 提案源 (v92 追加分) ---
from sklearn.cluster import DBSCAN

_FLOW_NET = None

def flowseg_net():
    global _FLOW_NET
    if _FLOW_NET is None:
        net = smp.Unet(CFG.FLOW_ENCODER, encoder_weights=None, in_channels=1, classes=3)
        ckpt = torch.load(CFG.FLOW_CKPT, map_location="cuda", weights_only=False)
        net.load_state_dict(ckpt["model"], strict=True)
        _FLOW_NET = net.cuda().eval()
    return _FLOW_NET

@torch.no_grad()
def flowseg_map(image_path):
    \"\"\"full-res 4-flip TTA → (prob[H,W] float32, flow[2,H,W] float32)。
    flip 解除時に flow 成分の符号を反転 (dim2=H→flow_y、dim3=W→flow_x)。\"\"\"
    x = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy(x[None, None].astype(np.float32) / 255.0).cuda()
    pa = torch.zeros((1, 1, *x.shape[2:]), device="cuda")
    fa = torch.zeros((1, 2, *x.shape[2:]), device="cuda")
    for flip in [(), (3,), (2,), (2, 3)]:
        xi = torch.flip(x, flip) if flip else x
        with torch.autocast("cuda", torch.bfloat16):
            out = flowseg_net()(xi).float()
        p, f = out[:, :1].sigmoid(), out[:, 1:3]
        p = torch.flip(p, flip) if flip else p
        f = torch.flip(f, flip) if flip else f
        s = torch.tensor([-1.0 if 2 in flip else 1.0,
                          -1.0 if 3 in flip else 1.0], device="cuda")[None, :, None, None]
        pa += p; fa += f * s
    return (pa / 4)[0, 0].cpu().numpy(), (fa / 4)[0].cpu().numpy()

def flowseg_maps_for_valid():
    \"\"\"valid_files 全件の (prob, flow) を生成し Drive npz に cache (冪等)。\"\"\"
    maps = {}
    for name in tqdm(valid_files, desc="flow マップ"):
        cache = CFG.FLOW_MAPS / (Path(name).stem + ".npz")
        if cache.exists():
            z = np.load(cache)
            maps[name] = (z["prob"].astype(np.float32), z["flow"].astype(np.float32))
            continue
        prob, flow = flowseg_map(CFG.PREPROC_DIR / name)
        np.savez_compressed(cache, prob=prob.astype(np.float16), flow=flow.astype(np.float16))
        maps[name] = (prob, flow)
    return maps

@torch.no_grad()
def follow_flow(fg, fy, fx):
    \"\"\"fg 画素を予測 flow で Euler 追従 → 端点 (N,2) float32 (y,x)。
    全 fg 画素を GPU バッチで FLOW_NITER step 反復 (bilinear 補間)。\"\"\"
    height, width = fg.shape
    ys, xs = np.nonzero(fg)
    if not len(ys):
        return np.zeros((0, 2), np.float32)
    pos = torch.from_numpy(np.stack([ys, xs], 1).astype(np.float32)).cuda()
    field = torch.from_numpy(np.stack([fy, fx])[None]).cuda()  # (1,2,H,W)
    for _ in range(CFG.FLOW_NITER):
        gy = pos[:, 0] / max(height - 1, 1) * 2 - 1
        gx = pos[:, 1] / max(width - 1, 1) * 2 - 1
        g = torch.stack([gx, gy], 1)[None, :, None, :]          # (1,N,1,2) (x,y)
        step = torch.nn.functional.grid_sample(
            field, g, mode="bilinear", align_corners=True,
            padding_mode="border")[0, :, :, 0].T                # (N,2)=(fy,fx)
        pos += step * CFG.FLOW_STEP
        pos[:, 0].clamp_(0, height - 1); pos[:, 1].clamp_(0, width - 1)
    return pos.cpu().numpy()

def flow_proposals(prob, flow, threshold=CFG.FLOW_PROP_THRESH):
    \"\"\"fg>=閾値 の画素を flow 追従 → 端点 DBSCAN → クラスタ毎に個体マスク。
    conf=クラスタ内平均 fg 確率。noise 点・min_area 未満は捨てる。\"\"\"
    fg = prob >= threshold
    if not fg.any():
        return []
    ends = follow_flow(fg, flow[0], flow[1])
    if not len(ends):
        return []
    labels = DBSCAN(eps=CFG.FLOW_EPS, min_samples=CFG.FLOW_MIN_SAMPLES).fit_predict(ends)
    ys, xs = np.nonzero(fg)
    probs_fg = prob[ys, xs]
    out = []
    for lab in np.unique(labels):
        if lab < 0:
            continue
        sel = labels == lab
        if int(sel.sum()) < CFG.MIN_AREA:
            continue
        m = np.zeros(fg.shape, bool)
        m[ys[sel], xs[sel]] = True
        out.append((float(probs_fg[sel].mean()), m))
    return out

def evaluate_flowprop(maps, label):
    \"\"\"flow 提案のみ (det 不使用) → prod と同一 refine→ownership→PQ。\"\"\"
    cache = []
    for name in tqdm(valid_files, desc=f"{label} 提案+refine"):
        image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
        prob, flow = maps[name]
        entries = flow_proposals(prob, flow)
        raw_rles = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for _, m in entries]
        refined = [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8)))
                   for _, m in entries]
        confs = np.array([c for c, _ in entries], np.float32)
        cache.append((name, confs, raw_rles, refined))
    return _score_union(cache, label, union=False)

def evaluate_union(maps, label):
    \"\"\"prod det 提案 ∪ flow 提案。flow 提案は conf>=c の det 提案と
    mask-IoU > FLOW_DEDUP_IOU で重複する場合落とす。両方を conf>=c で
    フィルタ後、同一 refine→ownership→PQ (prod 経路と同一)。\"\"\"
    model = YOLO(str(CFG.PROD_YOLO))
    cache = []
    for name in tqdm(valid_files, desc=f"{label} 推論+提案+refine"):
        path = CFG.PREPROC_DIR / name
        result = model.predict(str(path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.DIAG_CONF_FLOOR,
                               iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False)[0]
        det_masks, det_conf = result_masks(result)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        det_raw = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in det_masks]
        det_ref = [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8)))
                   if det_conf[i] >= CFG.PREDICT_CONF_FLOOR else None for i, m in enumerate(det_masks)]
        prob, flow = maps[name]
        props = flow_proposals(prob, flow)
        flow_raw = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for _, m in props]
        flow_ref = [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8)))
                   for _, m in props]
        flow_conf = np.array([c for c, _ in props], np.float32)
        cache.append((name, det_conf, det_raw, det_ref, flow_conf, flow_raw, flow_ref))
    del model; torch.cuda.empty_cache()
    return _score_union(cache, label, union=True)

def _dedup_flow(flow_rles, flow_conf, det_rles, det_conf, conf_level):
    \"\"\"conf>=c の flow 提案のうち、conf>=c の det 提案と mask-IoU>閾値
    で重複しない index を返す。\"\"\"
    det_idx = [i for i in range(len(det_conf)) if det_conf[i] >= conf_level]
    det_kept = [det_rles[i] for i in det_idx]
    keep = []
    for i in range(len(flow_conf)):
        if flow_conf[i] < conf_level:
            continue
        if det_kept:
            ov = mask_utils.iou([flow_rles[i]], det_kept, [0] * len(det_kept))[0]
            if ov.max() > CFG.FLOW_DEDUP_IOU:
                continue
        keep.append(i)
    return keep, det_idx

def _score_union(cache, label, union=False):
    out = {}
    for c in CFG.CONF_SWEEP:
        meter = PanopticQuality()
        n_pred = 0
        for entry in cache:
            if union:
                name, det_conf, det_raw, det_ref, flow_conf, flow_raw, flow_ref = entry
                keep_flow, keep_det = _dedup_flow(flow_raw, flow_conf, det_raw, det_conf, c)
                pred_masks = [mask_utils.decode(det_ref[i]) != 0 for i in keep_det]
                pred_masks += [mask_utils.decode(flow_ref[i]) != 0 for i in keep_flow]
            else:
                name, flow_conf, flow_raw, flow_ref = entry
                keep_flow = [i for i in range(len(flow_conf)) if flow_conf[i] >= c]
                pred_masks = [mask_utils.decode(flow_ref[i]) != 0 for i in keep_flow]
            preds = apply_ownership(pred_masks)
            pred_rles = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
            n_pred += len(pred_rles)
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
        out[str(c)] = {**meter.summary(), "preds_per_image": n_pred / max(len(valid_files), 1)}
        print(label, c, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[str(c)].items()})
    if union:
        # 診断: prod det の primary FN のうち flow 提案が IoU>0.5 で覆う数
        diag = {"prod_fn_at_primary": 0, "fn_covered_by_flow_prop": 0, "flow_props_per_image": 0}
        for entry in cache:
            name, det_conf, det_raw, det_ref, flow_conf, flow_raw, flow_ref = entry
            diag["flow_props_per_image"] += len(flow_conf)
            keep_det = [i for i in range(len(det_conf)) if det_conf[i] >= CFG.PRIMARY_CONF]
            pred_rles = [det_ref[i] for i in keep_det]
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                if not truth:
                    continue
                hits = iou_matrix(truth, pred_rles) > 0.5 if pred_rles else np.zeros((len(truth), 0))
                fn_idx = np.where(hits.sum(axis=1) == 0)[0] if pred_rles else np.arange(len(truth))
                if not len(fn_idx):
                    continue
                diag["prod_fn_at_primary"] += len(fn_idx)
                if flow_raw:
                    covered = np.asarray(mask_utils.iou(flow_raw, [truth[i] for i in fn_idx],
                                                      [0] * len(fn_idx)), dtype=np.float64).T
                    diag["fn_covered_by_flow_prop"] += int((covered > 0.5).any(axis=1).sum())
        diag["flow_props_per_image"] /= max(len(valid_files), 1)
        out["diag_flowrescue"] = diag
        print(label, "diag:", diag)
    return out
"""
    ),
    code(
        """# [6/6] 実行: prod baseline → flowseg 学習 → 確率map → flowprop/union 評価 → ゲート判定
def save_manifest(manifest):
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

CONTRACT = {"experiment_id": "v92_cellpose_flow", "fold": CFG.FOLD, "arms": list(CFG.ARMS),
            "protocol": "RESEARCH_LOG §2.52 (Cellpose flow-field 提案源による blind-FN 救済+個体分離)",
            "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO),
            "flowseg": {"encoder": CFG.FLOW_ENCODER, "crop": CFG.FLOW_CROP,
                       "batch": CFG.FLOW_BATCH, "lr": CFG.FLOW_LR,
                       "train_hours": CFG.FLOW_TRAIN_HOURS, "classes": 3,
                       "flow_loss_w": CFG.FLOW_LOSS_W,
                       "prop_thresh": CFG.FLOW_PROP_THRESH, "dedup_iou": CFG.FLOW_DEDUP_IOU,
                       "niter": CFG.FLOW_NITER, "step": CFG.FLOW_STEP,
                       "dbscan_eps": CFG.FLOW_EPS, "dbscan_min_samples": CFG.FLOW_MIN_SAMPLES},
            "conf_sweep": list(CFG.CONF_SWEEP), "primary_conf": CFG.PRIMARY_CONF,
            "gate": ("union@0.26: PQ(union,0.26)-PQ(prod,0.26)>=+0.005 で fold0/1 昇格検討; "
                     "[0,+0.005) は reject_noise、<0 は reject; "
                     "flowprop は診断 (提案源単独の性能) で採用候補にしない")}
manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8")) if CFG.MANIFEST.exists() else {}
if manifest and manifest.get("contract") != CONTRACT:
    raise ValueError("既存 manifest と入力契約が違う。履歴保持のため別パスを指定してください")
manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}

if "prod" not in manifest["results"]:
    manifest["results"]["prod"] = {"weights": str(CFG.PROD_YOLO), "by_conf": evaluate_yolo(CFG.PROD_YOLO, "prod")}
    save_manifest(manifest)
prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]
print({"prod@primary": round(prod_pq, 6), "既知値": CFG.BASELINE_PQ,
       "差": round(prod_pq - CFG.BASELINE_PQ, 6)})
if abs(prod_pq - CFG.BASELINE_PQ) > 0.002:
    print("⚠️ prod の再現値が既知値から 0.002 以上ずれている — 配線/入力の差を確認 (判定は paired Δ で行う)")

if "flowseg" not in manifest["datasets"]:
    ckpt, stats = train_flowseg()
    manifest["datasets"]["flowseg"] = {**stats, "ckpt": str(ckpt),
                                     "ckpt_sha256": file_sha256(ckpt)}
    save_manifest(manifest)

_maps_flag = CFG.FLOW_MAPS / "maps_cached.flag"
if not _maps_flag.exists():
    maps = flowseg_maps_for_valid()
    _maps_flag.write_text("ok", encoding="utf-8")
else:
    maps = {}
    for n in valid_files:
        z = np.load(CFG.FLOW_MAPS / (Path(n).stem + ".npz"))
        maps[n] = (z["prob"].astype(np.float32), z["flow"].astype(np.float32))

for arm in CFG.ARMS:
    if arm in manifest["results"]:
        continue
    by_conf = evaluate_flowprop(maps, arm) if arm == "flowprop" else evaluate_union(maps, arm)
    delta_same = by_conf[str(CFG.PRIMARY_CONF)]["pq"] - prod_pq
    best_conf = max((c for c in by_conf if not c.startswith("diag")), key=lambda c: by_conf[c]["pq"])

    def _verdict(d):
        return "promote_candidate" if d >= CFG.GATE_DELTA else ("reject_noise" if d >= 0 else "reject")

    manifest["results"][arm] = {"by_conf": by_conf,
                                "delta_vs_prod026": delta_same,
                                "delta_sparse_primary_vs_prod": by_conf[str(CFG.PRIMARY_CONF)]["pq_sparse_r077"]
                                    - manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq_sparse_r077"],
                                "exploratory_best_conf": best_conf,
                                "exploratory_best_delta": by_conf[best_conf]["pq"] - prod_pq,
                                "verdict": _verdict(delta_same)}
    manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    print(arm, {"Δ@0.26": round(delta_same, 4), "verdict": _verdict(delta_same),
                "expl_best": (best_conf, round(by_conf[best_conf]["pq"] - prod_pq, 4))})

print(json.dumps({a: {c: round(r["by_conf"][c]["pq"], 4) for c in r["by_conf"] if not c.startswith("diag")}
                  for a, r in manifest["results"].items()}, indent=2))
print("manifest:", CFG.MANIFEST)
"""
    ),
]
