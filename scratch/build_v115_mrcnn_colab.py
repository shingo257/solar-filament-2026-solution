"""V115 Colab ノートのビルダー: Mask R-CNN 異種検出器 + cross-detector union (§2.93)。

v96 third_f0 Colab ノートの凍結セル (COCO/fold・refiner 定義・評価関数) を
verbatim 再利用し、torchvision MaskRCNN-ResNet50-FPN-v2 の学習・提案キャッシュ・
union dedup 評価を新規セルで構成する。

- `mrcnn_raw`: MRCNN mask → ownership → PQ (refiner 不使用)
- `mrcnn_ref`: MRCNN 提案 → refiner → ownership → PQ
- `union_yolo_mrcnn`: YOLO ∪ MRCNN、conf 降順 mask-IoU>0.30 dedup → refiner → PQ
- min_size=1280 (prod imgsz に揃える)、aug = dihedral のみ、box は非空 mask と対で再計算

凍結セル (index 2,4,5) は SOURCE から verbatim コピーされるため採点経路は同一。

    python scratch/build_v115_mrcnn_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v115_mrcnn_colab.ipynb"


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cell(cell_type: str, source: str) -> dict:
    base = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


TITLE = """# v115 — Mask R-CNN 異種検出器 + cross-detector union (Colab 版)

RESEARCH_LOG §2.93 の事前登録実験。全 ~60 arm が YOLO 系 + 独自 refiner
の枠内だったため、**異種検出器ファミリー** (torchvision Mask R-CNN
ResNet50-FPN-v2, COCO pretrained) を導入して 2 問を分離する。

| arm | 内容 |
|---|---|
| `mrcnn_raw` | MRCNN mask → ownership → PQ (refiner 不使用) |
| `mrcnn_ref` | MRCNN 提案 → refiner → ownership → PQ |
| `union_yolo_mrcnn` | YOLO ∪ MRCNN、conf 降順 mask-IoU>0.30 dedup → refiner → PQ |

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`。

**caveat**: 両 detector の score は未校正 — isotonic 無しの単純 conf
順位 pool (限定的試行)。学習に best-epoch 選択なし (固定 10ep の
last 重み、prod の patience 選択とは非対称 — probe として開示)。

**ゲート (§2.93、結果取得前に固定)**: いずれかの arm で
Δ@0.26 >= **+0.010** → fold0 honest 複製 (folds(1,2) で再学習)。
fold2 単独では提出を変更しない。

GPU 必須 (~4h: MRCNN 学習 ~2h + 推論/評価 ~1.5h)。上から順に全セル実行。
結果は `artifacts/v115_mrcnn_colab_manifest.json` に逐次保存。
"""

CELL1 = '''# [1/6] 依存・Drive・設定
%pip install -q "ultralytics==8.4.118" pycocotools  # prod と同一版
import hashlib, json, math, os, random, shutil, sys, tempfile, time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from google.colab import drive
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from ultralytics import YOLO

drive.mount("/content/drive")

class CFG:
    SEED = 3407
    FOLD = 2  # v115: fold2 で評価 (prod runs/yolo_fold_2 と paired)
    ARMS = ('mrcnn_raw', 'mrcnn_ref', 'union_yolo_mrcnn')
    TRAIN_FOLDS = (0, 1)
    # Drive artifacts/grouped_folds.csv (707 obs, file_name/fold) の凍結 SHA256
    GROUPED_FOLDS_SHA256 = "febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e"
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v115_mrcnn"
    YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v115_mrcnn"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v115")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    PROD_YOLO = DRIVE_ROOT / "runs" / f"yolo_fold_{FOLD}" / "weights" / "best.pt"
    REFINER_CKPT = ARTIFACT_ROOT / f"mediseg_edgeattnet_refiner_fold{FOLD}_resume.pt"
    MRCNN_CKPT = RUNS_ROOT / f"mrcnn_fpnv2_fold{FOLD}_last.pt"
    MANIFEST = ARTIFACT_ROOT / "v115_mrcnn_colab_manifest.json"
    # --- Mask R-CNN (torchvision、COCO pretrained) ---
    MRCNN_EPOCHS = 10
    MRCNN_BATCH = 2
    MRCNN_LR = 1e-4
    MRCNN_WD = 1e-4
    MRCNN_MIN_SIZE = 1280  # model.transform で縮小 (prod YOLO_IMGSZ に揃える)
    MRCNN_NCLASS = 1
    YOLO_IMGSZ = 1280  # prod YOLO 推論解像度 (baseline 評価・union arm で使用)
    # --- 評価 (v87/prod と同一) ---
    PREDICT_CONF_FLOOR = 0.08
    DIAG_CONF_FLOOR = 0.02
    PREDICT_IOU = 0.55
    CONF_SWEEP = (0.15, 0.2, 0.26, 0.32)
    PRIMARY_CONF = 0.26
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    UNION_MERGE_IOU = 0.30  # cross-detector dedup (conf 降順 greedy)
    BASELINE_PQ = 0.444238  # fold2 prod 既知値 (§2.48、warn-only)
    GATE_DELTA = 0.010
    SPARSE_R = 0.77

for path in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.RUNS_ROOT, CFG.YOLO_ROOT):
    path.mkdir(parents=True, exist_ok=True)

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
                "python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
                "ultralytics": version("ultralytics"),
                "pycocotools": version("pycocotools"), "opencv": cv2.__version__, "numpy": np.__version__}
print(RUNTIME_INFO)
'''

CELL3 = '''# [3/6] Mask R-CNN データセット構築 + 学習 (torchvision、§2.93 固定レシピ)
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

rng = np.random.default_rng(CFG.SEED + CFG.FOLD)

def build_mrcnn():
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    num_classes = CFG.MRCNN_NCLASS + 1  # background(0) + filament(1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
    model.roi_heads.score_thresh = CFG.DIAG_CONF_FLOOR
    model.transform.min_size = (CFG.MRCNN_MIN_SIZE,)
    model.transform.max_size = CFG.MRCNN_MIN_SIZE
    return model.cuda()

def _boxes_of(masks, width, height):
    """aug 後の mask から非空個体の (boxes, valid_masks) を対で返す。

    torchvision ops.masks_to_boxes と同じ inclusive 画素座標 + 画像境界内
    clamp。境界に接する個体で x2/y2 が画像サイズに一致する排他座標を避け、
    roi_align 系カーネルの device-side assert を防止する。
    """
    boxes, valid_masks = [], []
    for m in masks:
        ys, xs = np.where(m)
        if not len(xs):
            continue
        x0, y0 = float(xs.min()), float(ys.min())
        x1 = min(float(xs.max()), float(width) - 1.0)
        y1 = min(float(ys.max()), float(height) - 1.0)
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append([x0, y0, x1, y1])
        valid_masks.append(m)
    return boxes, valid_masks

def mrcnn_sample(file_name, image_id):
    """前処理画像 + annotation → (image_tensor, target)。aug は dihedral のみ。"""
    info = images[image_id]
    height, width = int(info["height"]), int(info["width"])
    image = cv2.imread(str(CFG.PREPROC_DIR / file_name), cv2.IMREAD_COLOR)
    masks = [mask_utils.decode(to_rle(a, height, width)) != 0
             for a in annotations[image_id] if not a.get("iscrowd", 0)]
    k = int(rng.integers(0, 4))
    if k:
        image = np.ascontiguousarray(np.rot90(image, k))
        masks = [np.ascontiguousarray(np.rot90(m, k)) for m in masks]
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        masks = [np.ascontiguousarray(m[:, ::-1]) for m in masks]
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[::-1])
        masks = [np.ascontiguousarray(m[::-1]) for m in masks]
    ih, iw = image.shape[:2]
    boxes, masks = _boxes_of(masks, iw, ih)
    if not boxes:  # 空 target は MRCNN 学習で不安定なので skip
        return None
    tensor = torch.from_numpy(image[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255.0
    target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
              "labels": torch.ones(len(boxes), dtype=torch.int64),
              "masks": torch.from_numpy(np.stack(masks).astype(np.uint8))}
    assert target["masks"].shape == (len(boxes), ih, iw)
    assert bool((target["boxes"][:, 2] > target["boxes"][:, 0]).all())
    assert bool((target["boxes"][:, 3] > target["boxes"][:, 1]).all())
    assert bool((target["boxes"][:, 2] <= iw - 1).all())  # inclusive clamp 確認
    return tensor, target

def train_mrcnn(prev_stats=None):
    done_path = CFG.MRCNN_CKPT.parent / "TRAINING_COMPLETE"
    if done_path.exists() and CFG.MRCNN_CKPT.exists():
        print(f"学習済み checkpoint を再利用 {CFG.MRCNN_CKPT}")
        return CFG.MRCNN_CKPT, dict(prev_stats or {}) | {"reused": True}
    samples = [(name, image_id) for name in fit_files for image_id in file_to_ids[name]]
    def _fit(batch_size):
        global rng
        seed = CFG.SEED + CFG.FOLD
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        ep_rng = np.random.default_rng(seed)
        model = build_mrcnn()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.MRCNN_LR,
                                      weight_decay=CFG.MRCNN_WD)
        started = time.time()
        for epoch in range(CFG.MRCNN_EPOCHS):
            order = ep_rng.permutation(len(samples))
            losses_run, used = 0.0, 0
            for b0 in tqdm(range(0, len(order), batch_size), desc=f"mrcnn ep{epoch + 1}/{CFG.MRCNN_EPOCHS}"):
                batch, names = [], []
                for idx in order[b0:b0 + batch_size]:
                    sample = mrcnn_sample(*samples[idx])
                    if sample is not None:
                        batch.append(sample); names.append(samples[idx][0])
                if not batch:
                    continue
                imgs = [t.to("cuda") for t, _ in batch]
                tgts = [{k: v.to("cuda") for k, v in t.items()} for _, t in batch]
                try:
                    loss = sum(model(imgs, tgts).values())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except RuntimeError:
                    print({"failed_batch": names,
                           "boxes_max": [float(t["boxes"].max()) for _, t in batch],
                           "boxes_min": [float(t["boxes"].min()) for _, t in batch],
                           "n_gt": [int(t["labels"].numel()) for _, t in batch]})
                    raise
                losses_run += float(loss.detach()); used += 1
            print({"epoch": epoch + 1, "loss": round(losses_run / max(used, 1), 4)})
        return model, round((time.time() - started) / 60, 1)

    batch_used, oom_fallback = CFG.MRCNN_BATCH, False
    try:
        model, train_minutes = _fit(batch_used)
    except torch.cuda.OutOfMemoryError:
        if batch_used <= 1:
            raise
        torch.cuda.empty_cache()
        batch_used, oom_fallback = 1, True
        print("⚠️ CUDA OOM: §2.93 規定に従い batch=1 へ 1 回 fallback して再学習")
        model, train_minutes = _fit(batch_used)
    stats = {"train_samples": len(samples), "train_minutes": train_minutes,
             "batch_used": batch_used, "oom_fallback": oom_fallback}
    CFG.MRCNN_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": {"epochs": CFG.MRCNN_EPOCHS,
                                                     "min_size": CFG.MRCNN_MIN_SIZE,
                                                     "lr": CFG.MRCNN_LR,
                                                     "batch_used": batch_used,
                                                     "oom_fallback": oom_fallback}},
               CFG.MRCNN_CKPT)
    done_path.write_text("ok", encoding="utf-8")
    return CFG.MRCNN_CKPT, stats

@torch.no_grad()
def predict_mrcnn(model, path):
    """MRCNN 推論 → (masks list, scores)。出力は入力解像度 (2048)。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    tensor = torch.from_numpy(image[:, :, ::-1].transpose(2, 0, 1).copy()).float().cuda() / 255.0
    out = model([tensor])[0]
    scores = out["scores"].detach().cpu().numpy().astype(np.float32)
    masks = []
    height, width = image.shape[:2]
    for m in out["masks"][:, 0].detach().cpu().numpy():
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
        masks.append(m >= 0.5)
    return masks, scores
'''

CELL6 = '''# [6/6] 実行: prod baseline → MRCNN 学習 → 提案キャッシュ → 3 arm 評価、manifest 逐次保存 (判定は §2.93)
def save_manifest(manifest):
    CFG.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

def build_cache(predict_fn, label):
    """evaluate_yolo と同じ cache 構造: (name, conf, areas, raw_rles, refined)。"""
    cache = []
    for name in tqdm(valid_files, desc=f"{label} 推論+refine"):
        path = CFG.PREPROC_DIR / name
        masks, conf = predict_fn(path)
        keep = [i for i in range(len(conf)) if conf[i] >= CFG.DIAG_CONF_FLOOR]
        masks = [masks[i] for i in keep]
        conf = np.asarray([conf[i] for i in keep], np.float32)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        raw_rles = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
        refined = [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8)))
                   if conf[i] >= CFG.PREDICT_CONF_FLOOR else None for i, m in enumerate(masks)]
        cache.append((name, conf, [float(m.sum()) for m in masks], raw_rles, refined))
    return cache

def merge_union_caches(cache_a, cache_b, label):
    """2 detector の提案を conf 降順 mask-IoU>0.30 で dedup (RLE IoU + 決定論的 refine 再利用)。"""
    cache = []
    for (name, ca, aa, ra, fa), (name_b, cb, ab, rb, fb) in tqdm(
            list(zip(cache_a, cache_b)), desc=f"{label} union merge"):
        assert name == name_b
        pool = [(float(c), float(a), r, f)
                for c, a, r, f in list(zip(ca, aa, ra, fa)) + list(zip(cb, ab, rb, fb))]
        pool.sort(key=lambda t: -t[0])
        kept, kept_rles = [], []
        for conf, area, rle, ref_rle in pool:
            if kept_rles:
                ious = np.asarray(
                    mask_utils.iou([rle], kept_rles, [0] * len(kept_rles)), dtype=np.float64)[0]
                if (ious > CFG.UNION_MERGE_IOU).any():
                    continue
            kept.append((conf, area, rle, ref_rle))
            kept_rles.append(rle)
        conf = np.asarray([c for c, _, _, _ in kept], np.float32)
        areas = [a for _, a, _, _ in kept]
        raw_rles = [r for _, _, r, _ in kept]
        refined = [f for _, _, _, f in kept]
        cache.append((name, conf, areas, raw_rles, refined))
    return cache

def eval_cached(cache, label, use_refined):
    """conf sweep → keep conf>=c & MIN_AREA → ownership → pooled PQ (evaluate_yolo と同一採点)。"""
    out = {}
    for c in CFG.CONF_SWEEP:
        meter = PanopticQuality()
        n_pred = 0
        for name, conf, areas, raw_rles, refined in cache:
            keep = [i for i in range(len(conf)) if conf[i] >= c and areas[i] >= CFG.MIN_AREA]
            src = refined if use_refined else raw_rles
            preds = apply_ownership([mask_utils.decode(src[i]) != 0 for i in keep])
            pred_rles = (mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8)))
                       if preds else [])
            n_pred += len(pred_rles)
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
        out[str(c)] = {**meter.summary(), "preds_per_image": n_pred / max(len(cache), 1)}
        print(label, c, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[str(c)].items()})
    return out

CONTRACT = {"experiment_id": "v115_mrcnn", "fold": CFG.FOLD, "arms": list(CFG.ARMS),
            "protocol": "RESEARCH_LOG §2.93 (Mask R-CNN family + cross-detector union)",
            "train_folds": list(CFG.TRAIN_FOLDS),
            "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO),
            "mrcnn": {"model": "maskrcnn_resnet50_fpn_v2", "epochs": CFG.MRCNN_EPOCHS,
                      "batch": CFG.MRCNN_BATCH, "lr": CFG.MRCNN_LR, "wd": CFG.MRCNN_WD,
                      "min_size": CFG.MRCNN_MIN_SIZE},
            "union_merge_iou": CFG.UNION_MERGE_IOU,
            "conf_sweep": list(CFG.CONF_SWEEP), "primary_conf": CFG.PRIMARY_CONF,
            "gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.93)"}
manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8")) if CFG.MANIFEST.exists() else {}
if manifest and manifest.get("contract") != CONTRACT:
    raise ValueError("既存 manifest と入力契約が違う。履歴保持のため別パスを指定してください")
manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}
CACHES = {}  # 提案 cache は numpy/RLE を含むため manifest 外 (resume 時は推論をやり直す)

if "prod" not in manifest["results"]:
    manifest["results"]["prod"] = {"weights": str(CFG.PROD_YOLO), "by_conf": evaluate_yolo(CFG.PROD_YOLO, "prod")}
    save_manifest(manifest)
prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]
print({"prod@primary": round(prod_pq, 6), "既知値": CFG.BASELINE_PQ, "差": round(prod_pq - CFG.BASELINE_PQ, 6)})

weights, stats = train_mrcnn(manifest["datasets"].get("mrcnn_train"))
manifest["datasets"]["mrcnn_train"] = stats
manifest["results"].setdefault("_meta", {})["mrcnn_ckpt_sha256"] = file_sha256(weights)
save_manifest(manifest)
mrcnn = build_mrcnn()
mrcnn.load_state_dict(torch.load(weights, map_location="cuda", weights_only=False)["model"])
mrcnn.eval()

if "mrcnn" not in CACHES:
    CACHES["mrcnn"] = build_cache(lambda p: predict_mrcnn(mrcnn, p), "mrcnn")

MODE_OF_ARM = {"mrcnn_raw": False, "mrcnn_ref": True, "union_yolo_mrcnn": True}
for arm in CFG.ARMS:
    if arm in manifest["results"]:
        continue
    if arm == "union_yolo_mrcnn":
        if "yolo" not in CACHES:
            yolo_model = YOLO(str(CFG.PROD_YOLO))
            def _yolo_props(p):
                r = yolo_model.predict(str(p), imgsz=CFG.YOLO_IMGSZ, conf=CFG.DIAG_CONF_FLOOR,
                                       iou=CFG.PREDICT_IOU, agnostic_nms=True,
                                       retina_masks=True, verbose=False)[0]
                return result_masks(r)
            CACHES["yolo"] = build_cache(_yolo_props, "yolo_union")
            del yolo_model; torch.cuda.empty_cache()
        cache = merge_union_caches(CACHES["mrcnn"], CACHES["yolo"], arm)
    else:
        cache = CACHES["mrcnn"]
    by_conf = eval_cached(cache, arm, MODE_OF_ARM[arm])
    delta = by_conf[str(CFG.PRIMARY_CONF)]["pq"] - prod_pq
    manifest["results"][arm] = {"by_conf": by_conf, "delta_primary_vs_prod": delta,
                                "verdict": ("promote_fold0_replication" if delta >= CFG.GATE_DELTA
                                            else "reject_v115_fold2")}
    manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    print(arm, {"Δ@primary": round(delta, 4)})

print(json.dumps({a: {c: round(v["pq"], 4) for c, v in r["by_conf"].items()
                      if isinstance(v, dict) and "pq" in v}  # diag_lowconf を除外
                  for a, r in manifest["results"].items()
                  if a != "_meta" and isinstance(r.get("by_conf"), dict)}, indent=2))
print("manifest:", CFG.MANIFEST)
'''


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = _cell("markdown", TITLE)
    cells[1] = _cell("code", CELL1)
    # cell2 (COCO/fold/前処理)・cell4 (refiner 定義)・cell5 (評価関数) は無改変
    cells[3] = _cell("code", CELL3)
    cells[6] = _cell("code", CELL6)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {OUT.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
