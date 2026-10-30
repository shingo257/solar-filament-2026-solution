"""Embedded cell payloads for the build_v90_semprop notebook builder.

Derived from build_v89_cls15_fold01_cells.py: v90 は semantic U-Net
(smp ConvNeXt-T, 1024px crop, (file,annotator) エントリ単位) を独立
提案源として学習し、prod detector の提案に dedup 統合する。
arm `semprop` = semantic のみ、`union` = prod ∪ semantic。
採点は prod と同一経路 (refine→ownership→公式PQ)、低conf診断に加えて
「prod の primary FN を semantic 提案が救済した数」を記録する。
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
        """# v90 — semantic U-Net 提案源: detector-blind FN の直接救済

arm: `union` (prod ∪ semantic) + `semprop` (semantic のみ) + paired `prod`

**仮説**: detector が any-conf で提案を出さない blind-FN (primary FN の
~43%) は、検出ではなく画素分類でしか拾えない構造のはず。biconcavelens
(§2.39) は semantic ConvNeXt-T U-Net で detector-missed filament を
~30個追加獲得した。同機構を我々の eval 経路に提案源として統合する。

**semantic モデル**: smp U-Net (`tu-convnext_tiny`, ImageNet)、
1024px crop・(file,annotator) エントリ単位の binary 学習 (focal+dice)、
4-flip TTA full-res 推論 → connected components → 提案 (conf=最大確率)。
fold2 pilot。学習は時間予算 ~1.2h cosine。

**統合ルール**: det 提案 (conf>=閾値) + sem 提案 (conf>=閾値 かつ
mask-IoU>0.5 の det 提案に重ならないもの) → 同一 refine→ownership→PQ。

**事前固定ゲート** (RESEARCH_LOG §2.42):

- `PQ(union,0.26) − PQ(prod,0.26) >= +0.005`: fold0/1 に昇格検討
- `[0,+0.005)`: reject_noise、`<0`: reject
- 診断 (判定不使用): prod の primary FN が semantic 提案で救済された数

GPU必須。A100推奨 (~1.5h 学習 + 評価)。
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
    ARMS = ('semprop', 'union')
    # Drive artifacts/grouped_folds.csv (707 obs, file_name/fold) の凍結 SHA256
    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v90_semprop"
    YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v90_semprop"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v90")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    PROD_YOLO = DRIVE_ROOT / "runs" / f"yolo_fold_{FOLD}" / "weights" / "best.pt"
    REFINER_CKPT = ARTIFACT_ROOT / f"mediseg_edgeattnet_refiner_fold{FOLD}_resume.pt"
    MANIFEST = ARTIFACT_ROOT / f"v90_semprop_fold{FOLD}_manifest.json"
    SEM_CKPT = RUNS_ROOT / f"semseg_fold{FOLD}.pt"
    SEM_PROBS = RUNS_ROOT / f"probs_fold{FOLD}"
    # --- semantic U-Net (biconcavelens §2.39 準拠) ---
    SEM_ENCODER = "tu-convnext_tiny"
    SEM_CROP = 1024
    SEM_BATCH = 8           # A100 40GB 向け (biconcavelens は T4 で batch 2)
    SEM_LR = 2e-4
    SEM_TRAIN_HOURS = 1.2   # 時間予算 (彼らは T4 で 3h 規模)
    SEM_PROP_THRESH = 0.5   # 提案化の確率閾値
    SEM_DEDUP_IOU = 0.5     # union で det 提案との重複判定
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

for path in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.RUNS_ROOT, CFG.YOLO_ROOT, CFG.SEM_PROBS):
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
        """# [3/6] semantic U-Net 学習 (biconcavelens §2.39 準拠: 1024px crop・(file,annotator) 単位)
def render_fg(image_id):
    \"\"\"image_id (file,annotator) の annotation を fg=1 binary mask へ rasterize。\"\"\"
    info = images[image_id]
    height, width = int(info["height"]), int(info["width"])
    fg = np.zeros((height, width), np.uint8)
    for ann in annotations[image_id]:
        if ann.get("iscrowd", 0):
            continue
        for poly in ann.get("segmentation", []):
            if len(poly) >= 6:
                cv2.fillPoly(fg, [np.asarray(poly, np.int32).reshape(-1, 2)], 1)
    return fg

class SemSegDataset(torch.utils.data.Dataset):
    \"\"\"fit_entries 各エントリから 1024px crop をオンデマンド生成。
    75%: filament 画素中心 + jitter、25%: 完全ランダム。isotropic な
    rot90/hflip と gamma/brightness/contrast jitter を crop 後に掛ける。\"\"\"
    def __init__(self, entries, crop):
        self.entries, self.crop, self._cache = entries, crop, {}

    def __len__(self):
        return len(self.entries)

    def _load(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        file_name, image_id = self.entries[idx]
        image = cv2.imread(str(CFG.PREPROC_DIR / file_name), cv2.IMREAD_GRAYSCALE)
        fg = render_fg(image_id)
        if len(self._cache) < 512:  # 全fit画像をRAMに持たない (画像 ~4MB x2)
            self._cache[idx] = (image, fg)
        return image, fg

    def __getitem__(self, idx):
        rng = np.random.default_rng((CFG.SEED, idx, os.getpid()))
        image, fg = self._load(idx)
        height, width = image.shape
        crop = self.crop
        ys, xs = np.where(fg)
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
        x = image[max(0, y0):min(height, y0 + crop), max(0, x0):min(width, x0 + crop)]
        y = fg[max(0, y0):min(height, y0 + crop), max(0, x0):min(width, x0 + crop)]
        if any(pad[0]) or any(pad[1]):
            x = np.pad(x, pad, mode="edge"); y = np.pad(y, pad, mode="constant")
        k = rng.integers(4)
        x, y = np.rot90(x, k), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = np.fliplr(x), np.fliplr(y)
        xf = x.astype(np.float32) / 255.0
        gamma = rng.uniform(0.7, 1.4)
        xf = np.clip(xf ** gamma * rng.uniform(0.8, 1.2) + rng.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(xf[None].copy()), torch.from_numpy(y[None].astype(np.float32).copy())

def soft_dice_loss(logit, target, eps=1e-6):
    prob = torch.sigmoid(logit).flatten(1)
    t = target.flatten(1)
    inter = (prob * t).sum(1)
    return (1 - (2 * inter + eps) / (prob.sum(1) + t.sum(1) + eps)).mean()

_FOCAL = None

def semseg_loss(logit, target):
    global _FOCAL
    if _FOCAL is None:
        _FOCAL = smp.losses.FocalLoss(mode="binary", gamma=2.0, alpha=0.25)
    return 10.0 * _FOCAL(logit, target) + soft_dice_loss(logit, target)

def train_semseg():
    done_path = CFG.SEM_CKPT.parent / f"TRAINING_COMPLETE_fold{CFG.FOLD}"
    stats = {"train_entries": len(fit_entries), "encoder": CFG.SEM_ENCODER,
             "crop": CFG.SEM_CROP, "batch": CFG.SEM_BATCH}
    if done_path.exists() and CFG.SEM_CKPT.exists():
        print(f"semseg: 学習済み checkpoint を再利用 {CFG.SEM_CKPT}")
        return CFG.SEM_CKPT, stats
    dataset = SemSegDataset(fit_entries, CFG.SEM_CROP)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=CFG.SEM_BATCH, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True)
    net = smp.Unet(CFG.SEM_ENCODER, encoder_weights="imagenet",
                   in_channels=1, classes=1).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=CFG.SEM_LR, weight_decay=1e-4)
    budget = CFG.SEM_TRAIN_HOURS * 3600
    started, epoch, epoch_secs = time.time(), 0, []
    bar = tqdm(desc=f"semseg fold{CFG.FOLD} 学習")
    while True:
        epoch += 1
        ep_start = time.time()
        running, seen = 0.0, 0
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = semseg_loss(net(x), y)
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
                   CFG.SEM_CKPT)  # 毎 epoch 上書き (last.ckpt 運用)
        if time.time() - started > budget:
            break
    bar.close()
    stats["epochs"] = epoch
    stats["sec_per_epoch"] = round(float(np.mean(epoch_secs)), 1)
    stats["train_minutes"] = round((time.time() - started) / 60, 1)
    done_path.write_text("ok", encoding="utf-8")
    print({"semseg_epochs": epoch, "sec_per_epoch": stats["sec_per_epoch"],
           "train_min": stats["train_minutes"], "last_loss": round(running / max(seen, 1), 4)})
    return CFG.SEM_CKPT, stats
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

# --- semantic 提案源 (v90 追加分) ---
_SEM_NET = None

def semseg_net():
    global _SEM_NET
    if _SEM_NET is None:
        net = smp.Unet(CFG.SEM_ENCODER, encoder_weights=None, in_channels=1, classes=1)
        ckpt = torch.load(CFG.SEM_CKPT, map_location="cuda", weights_only=False)
        net.load_state_dict(ckpt["model"], strict=True)
        _SEM_NET = net.cuda().eval()
    return _SEM_NET

@torch.no_grad()
def semseg_prob(image_path):
    \"\"\"full-res 4-flip TTA の fg 確率マップ [0,1] float32。\"\"\"
    x = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy(x[None, None].astype(np.float32) / 255.0).cuda()
    acc = torch.zeros((x.shape[2], x.shape[3]), dtype=torch.float32, device="cuda")
    for flip in [(), (3,), (2,), (2, 3)]:
        xi = torch.flip(x, flip) if flip else x
        with torch.autocast("cuda", torch.bfloat16):
            p = semseg_net()(xi).float().sigmoid()[0, 0]
        acc += torch.flip(p, flip) if flip else p
    return (acc / 4).cpu().numpy()

def semseg_probs_for_valid():
    \"\"\"valid_files 全件の確率マップを生成し Drive npz に cache (冪等)。\"\"\"
    probs = {}
    for name in tqdm(valid_files, desc="semantic 確率マップ"):
        cache = CFG.SEM_PROBS / (Path(name).stem + ".npz")
        if cache.exists():
            probs[name] = np.load(cache)["prob"].astype(np.float32)
            continue
        prob = semseg_prob(CFG.PREPROC_DIR / name)
        np.savez_compressed(cache, prob=prob.astype(np.float16))
        probs[name] = prob
    return probs

def sem_proposals(prob, threshold=CFG.SEM_PROP_THRESH):
    \"\"\"fg 確率 → 閾値 → connected components。conf=成分内最大確率。\"\"\"
    binary = (prob >= threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for lab in range(1, n_labels):
        if stats[lab, cv2.CC_STAT_AREA] < CFG.MIN_AREA:
            continue
        m = labels == lab
        out.append((float(prob[m].max()), m))
    return out

def evaluate_semprop(probs, label):
    \"\"\"semantic 提案のみ (det 不使用) → prod と同一 refine→ownership→PQ。\"\"\"
    cache = []
    for name in tqdm(valid_files, desc=f"{label} 提案+refine"):
        image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
        entries = sem_proposals(probs[name])
        raw_rles = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for _, m in entries]
        refined = [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8)))
                   for _, m in entries]
        confs = np.array([c for c, _ in entries], np.float32)
        cache.append((name, confs, raw_rles, refined))
    return _score_union(cache, label, union=False)

def evaluate_union(probs, label):
    \"\"\"prod det 提案 ∪ semantic 提案。sem 提案は conf>=c の det 提案と
    mask-IoU > SEM_DEDUP_IOU で重複する場合落とす。両方を conf>=c で
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
        sem = sem_proposals(probs[name])
        sem_raw = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for _, m in sem]
        sem_ref = [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8)))
                   for _, m in sem]
        sem_conf = np.array([c for c, _ in sem], np.float32)
        cache.append((name, det_conf, det_raw, det_ref, sem_conf, sem_raw, sem_ref))
    del model; torch.cuda.empty_cache()
    return _score_union(cache, label, union=True)

def _dedup_sem(sem_rles, sem_conf, det_rles, det_conf, conf_level):
    \"\"\"conf>=c の sem 提案のうち、conf>=c の det 提案と mask-IoU>閾値
    で重複しない index を返す。\"\"\"
    det_idx = [i for i in range(len(det_conf)) if det_conf[i] >= conf_level]
    det_kept = [det_rles[i] for i in det_idx]
    keep = []
    for i in range(len(sem_conf)):
        if sem_conf[i] < conf_level:
            continue
        if det_kept:
            ov = mask_utils.iou([sem_rles[i]], det_kept, [0] * len(det_kept))[0]
            if ov.max() > CFG.SEM_DEDUP_IOU:
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
                name, det_conf, det_raw, det_ref, sem_conf, sem_raw, sem_ref = entry
                keep_sem, keep_det = _dedup_sem(sem_raw, sem_conf, det_raw, det_conf, c)
                pred_masks = [mask_utils.decode(det_ref[i]) != 0 for i in keep_det]
                pred_masks += [mask_utils.decode(sem_ref[i]) != 0 for i in keep_sem]
            else:
                name, sem_conf, sem_raw, sem_ref = entry
                keep_sem = [i for i in range(len(sem_conf)) if sem_conf[i] >= c]
                pred_masks = [mask_utils.decode(sem_ref[i]) != 0 for i in keep_sem]
            preds = apply_ownership(pred_masks)
            pred_rles = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
            n_pred += len(pred_rles)
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
        out[str(c)] = {**meter.summary(), "preds_per_image": n_pred / max(len(valid_files), 1)}
        print(label, c, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[str(c)].items()})
    if union:
        # 診断: prod det の primary FN のうち semantic 提案が IoU>0.5 で覆う数
        diag = {"prod_fn_at_primary": 0, "fn_covered_by_sem_prop": 0, "sem_props_per_image": 0}
        for entry in cache:
            name, det_conf, det_raw, det_ref, sem_conf, sem_raw, sem_ref = entry
            diag["sem_props_per_image"] += len(sem_conf)
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
                if sem_raw:
                    covered = np.asarray(mask_utils.iou(sem_raw, [truth[i] for i in fn_idx],
                                                      [0] * len(fn_idx)), dtype=np.float64).T
                    diag["fn_covered_by_sem_prop"] += int((covered > 0.5).any(axis=1).sum())
        diag["sem_props_per_image"] /= max(len(valid_files), 1)
        out["diag_semrescue"] = diag
        print(label, "diag:", diag)
    return out
"""
    ),
    code(
        """# [6/6] 実行: prod baseline → semseg 学習 → 確率map → semprop/union 評価 → ゲート判定
def save_manifest(manifest):
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

CONTRACT = {"experiment_id": "v90_semprop", "fold": CFG.FOLD, "arms": list(CFG.ARMS),
            "protocol": "RESEARCH_LOG §2.42 (semantic U-Net 提案源による blind-FN 救済)",
            "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO),
            "semseg": {"encoder": CFG.SEM_ENCODER, "crop": CFG.SEM_CROP,
                       "batch": CFG.SEM_BATCH, "lr": CFG.SEM_LR,
                       "train_hours": CFG.SEM_TRAIN_HOURS,
                       "prop_thresh": CFG.SEM_PROP_THRESH, "dedup_iou": CFG.SEM_DEDUP_IOU},
            "conf_sweep": list(CFG.CONF_SWEEP), "primary_conf": CFG.PRIMARY_CONF,
            "gate": ("union@0.26: PQ(union,0.26)-PQ(prod,0.26)>=+0.005 で fold0/1 昇格検討; "
                     "[0,+0.005) は reject_noise、<0 は reject; "
                     "semprop は診断 (提案源単独の性能) で採用候補にしない")}
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

if "semseg" not in manifest["datasets"]:
    ckpt, stats = train_semseg()
    manifest["datasets"]["semseg"] = {**stats, "ckpt": str(ckpt),
                                     "ckpt_sha256": file_sha256(ckpt)}
    save_manifest(manifest)

_probs_flag = CFG.SEM_PROBS / "probs_cached.flag"
if not _probs_flag.exists():
    probs = semseg_probs_for_valid()
    _probs_flag.write_text("ok", encoding="utf-8")
else:
    probs = {n: np.load(CFG.SEM_PROBS / (Path(n).stem + ".npz"))["prob"].astype(np.float32)
             for n in valid_files}

for arm in CFG.ARMS:
    if arm in manifest["results"]:
        continue
    by_conf = evaluate_semprop(probs, arm) if arm == "semprop" else evaluate_union(probs, arm)
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
