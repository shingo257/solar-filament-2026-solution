"""V93 探針ノートのビルダー: consensus-class YOLO (annotator 合意数の明示学習)

## 背景 (§2.54)

annotator 一致解析で: 一致 IoU 0.634 < 我々の SQ 0.680 (SQ 天井超え)、
per-record recall 0.563 (~44% が単独 annotator 物体)、単独物体は
予測すると per-record 採点で 1TP vs ~1.44FP = 数学的にマイナス。
prod の conf は暗黙に consensus-likeness を学習しているはずだが、
明示的な class 化は未試行 — これが v87 union-label (label SET 変更)
とは異なる最後の構造レバー。

## 設計 (ラベルの class のみ変更。集合構造は prod と完全一致)

- prod と同じ per-record 学習サンプル ((file,annotator) の各 record が
  それ自身の polygon ラベルを持つ `stem__c{j}` コピー)
- 各 instance の class = **1 (consensus)** if その instance が属する
  IoU>0.5 union-find cluster の distinct annotator 数 >= 2、else **0 (weak)**
- cluster は file 内の全 record を横断して計算 (v87 の cluster_from_iou
  流用、単体テスト済み規則)
- `yolo11m-seg` を 2-class (names: 0=weak, 1=strong)、`single_cls=False`
- それ以外の学習引数 18 項目・fold・seed は prod と完全一致

## 評価 (fold2 OOF、prod と同一経路)

- predict 各提案に (cls, conf)。effective score = conf if cls==1 else conf*w0
- w0 grid {1.0, 0.7, 0.5, 0.3, 0.0} × conf sweep {0.15,0.20,0.26,0.32}
- w0=1.0 は sanity arm (class 情報を捨てた場合 = prod 同等 ranking)
- w0=0.0 は consensus-only (singleton 提案を全棄却)

## 事前固定ゲート (RESEARCH_LOG §2.55)

- `max_{w0} [PQ(arm,conf=0.26) − PQ(prod,conf=0.26)] >= +0.005`
  → fold0/1 に昇格検討 (w0 は 5 点の小さな calibration grid として
  事前登録。採用時は fold0/1 で同じ w0 grid を回して頑健性を確認)
- 全 w0 で [0,+0.005) は reject_noise、<0 は reject

## 使い方

    python scratch/build_v93_consensus_class.py

出力: solar_filament_v93_consensus_class_fold2_colab.ipynb (Colab GPU、A100 推奨)
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLD = 2
BASELINE_PQ = 0.444238  # FINAL_PHASE_PLAN A5 / v74 baseline arm (fold2 OOF, thr0.45, conf0.26)
CONF_SWEEP = (0.15, 0.20, 0.26, 0.32)
PRIMARY_CONF = 0.26
GATE_DELTA = 0.005
SPARSE_R = 0.77
CLUSTER_IOU = 0.5
ARMS = ("consensus2",)


def _load_models_source():
    """mediseg_models.py の実体。repo 配置 → 平置き → 既存 master notebook の埋め込みセルの順で解決。"""
    for path in (ROOT / "solar_filament" / "models" / "mediseg_models.py",
                 Path(__file__).resolve().parent / "mediseg_models.py"):
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if "class MediSegEdgeAttUNet" in text:
                return text, str(path)
    for nb_path in (ROOT / "solar_filament_master_pipeline_colab.ipynb",
                    Path(__file__).resolve().parent / "solar_filament_master_pipeline_colab.ipynb"):
        if nb_path.exists():
            for cell in json.loads(nb_path.read_text(encoding="utf-8"))["cells"]:
                text = "".join(cell["source"])
                if "class MediSegEdgeAttUNet" in text:
                    return text, f"{nb_path}#cell"
    raise FileNotFoundError("MediSegEdgeAttUNet のソースが見つからない (solar_filament/models/mediseg_models.py)")


# ---------------------------------------------------------------------------
# annotator 合意数の純関数群 (v87 から流用)。notebook に埋め込む。
# cluster_from_iou: soft_target.cluster_gt_instances と同じ規則 (union-find、同一 record
# 同士は直接・間接とも併合しない) を、pycocotools の RLE IoU 行列に対して適用する。
# v93 では union mask は作らず、クラスタの distinct annotator 数だけを使う。
# ---------------------------------------------------------------------------
UNION_SRC = r'''
def cluster_from_iou(record_of, iou, threshold=0.5):
    """record_of[k]: 個体 k の annotator record 番号、iou: (K,K) 対称行列。
    IoU > threshold の異 record 対を union-find で連結。同一 record を含むクラスタ同士は併合しない。
    返り値: クラスタ (個体 index の昇順リスト) のリスト。"""
    n = len(record_of)
    parent = list(range(n))
    records = [{record_of[k]} for k in range(n)]

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    pairs = sorted(((float(iou[a][b]), a, b) for a in range(n) for b in range(a + 1, n)
                    if record_of[a] != record_of[b] and float(iou[a][b]) > threshold),
                   reverse=True)  # 高 IoU の対から結ぶ (順序依存を決定的にする)
    for _, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb and records[ra].isdisjoint(records[rb]):
            parent[rb] = ra
            records[ra] |= records[rb]
    clusters = {}
    for k in range(n):
        clusters.setdefault(find(k), []).append(k)
    return sorted(clusters.values(), key=lambda c: c[0])


def sparse_label_pq(sum_iou, tp, fp, fn, r=0.77):
    """dariushafshar の sparse-label 換算 (RESEARCH_LOG §2.28)。r=1 で通常の PQ。"""
    denominator = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return float(r * sum_iou / denominator) if denominator > 0 else 0.0
'''


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells(arms, models_src, protocol_sha256, grouped_folds_sha256):
    arm_label = "_".join(arms) if len(arms) > 1 else arms[0]
    cells = [
        markdown(f"""# v93 — consensus-class YOLO 再学習 (fold{FOLD} パイロット)

arm: `consensus2` (2-class: 0=weak/単独annotator, 1=strong/合意>=2) + paired baseline `prod`。

**仮説** (§2.54): 単独 annotator 物体は per-record 採点で 1TP vs ~1.44FP
= 数学的にマイナス。consensus 尤度を class として明示学習すれば、
conf に依存しない「合意らしさ」ランキングが得られる可能性。

**変えるもの**: 学習ラベルの class のみ (0=単独, 1=合意>=2)。
サンプル構造 ((file,annotator) per-record コピー)・polygon・
ハイパラ・分割・seed は prod と完全一致。`single_cls=False`、
data.yaml は 2 names。

**評価**: 各提案に (cls,conf)。score = conf if cls==1 else conf*w0、
w0 grid {{1.0,0.7,0.5,0.3,0.0}} × conf sweep。w0=1.0 は sanity arm。

**事前登録ゲート** (RESEARCH_LOG §2.55):
`max_w0 [PQ(arm,0.26) − PQ(prod,0.26)] >= +{GATE_DELTA}` で fold0/1 昇格検討。
全 w0 で [0,+{GATE_DELTA}) は reject_noise、<0 は reject。

学習済み arm は `TRAINING_COMPLETE` マーカーでスキップ、結果は manifest に逐次追記。
GPU 目安: A100 で約 2〜3h + 評価 約 20〜30 分。
"""),
        code(f"""# [1/7] 依存・Drive・設定
%pip install -q "ultralytics>=8.3,<9" pycocotools
import hashlib, json, math, os, random, shutil, sys, tempfile, time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from ultralytics import YOLO

drive.mount("/content/drive")

class CFG:
    SEED = 3407
    FOLD = {FOLD}
    ARMS = {tuple(arms)!r}
    PROTOCOL_SHA256 = {protocol_sha256!r}
    GROUPED_FOLDS_SHA256 = {grouped_folds_sha256!r}  # 外部で凍結した grouped_folds.csv の SHA256 (master と同じ運用。空なら停止)
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v93_consensus_class"
    YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v93_consensus_class"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v93")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    PROD_YOLO = DRIVE_ROOT / "runs" / f"yolo_fold_{FOLD}" / "weights" / "best.pt"
    REFINER_CKPT = ARTIFACT_ROOT / f"mediseg_edgeattnet_refiner_fold{FOLD}_resume.pt"
    MANIFEST = ARTIFACT_ROOT / "v93_consensus_class_fold{FOLD}_{arm_label}_manifest.json"
    # --- YOLO (build_v30_prize_colab.py / master §B と同一。変更禁止) ---
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    # --- consensus class ---
    CLUSTER_IOU = {CLUSTER_IOU}      # union-find の IoU 閾値 (eval と同じ 0.5)
    W0_SWEEP = (1.0, 0.7, 0.5, 0.3, 0.0)  # weak class の conf 乗数。0.0=weak 全棄却
    # --- 評価 (run_fold_oof_eval.py / v58 と同一) ---
    PREDICT_CONF_FLOOR = 0.08
    PREDICT_IOU = 0.55
    CONF_SWEEP = {CONF_SWEEP!r}
    PRIMARY_CONF = {PRIMARY_CONF}
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ = {BASELINE_PQ}
    GATE_DELTA = {GATE_DELTA}
    SPARSE_R = {SPARSE_R}

for path in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.RUNS_ROOT, CFG.YOLO_ROOT):
    path.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
required = {{"competition data": CFG.DATA_CACHE, "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
            "前処理画像": CFG.PREPROC_CACHE, "prod YOLO": CFG.PROD_YOLO, "EdgeAtt refiner": CFG.REFINER_CKPT}}
missing = {{k: str(v) for k, v in required.items() if not v.exists()}}
if missing:
    raise FileNotFoundError(f"前提 artifact が見つからない: {{missing}}")
assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
RUNTIME_INFO = {{"started_utc": datetime.now(timezone.utc).isoformat(), "GPU": torch.cuda.get_device_name(0),
                "python": sys.version, "torch": torch.__version__, "ultralytics": version("ultralytics"),
                "pycocotools": version("pycocotools"), "opencv": cv2.__version__, "numpy": np.__version__}}
print(RUNTIME_INFO)
"""),
        code("# [2/7] annotator 合意数の純関数 (scratch/build_v93_consensus_class.py の UNION_SRC、単体テスト済み)\n" + UNION_SRC),
        code("""# [3/7] COCO・凍結 fold (SHA256 検証)・前処理画像
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
if not re.fullmatch(r"[0-9a-f]{64}", CFG.GROUPED_FOLDS_SHA256 or ""):
    raise ValueError("CFG.GROUPED_FOLDS_SHA256 に凍結 fold CSV の SHA256 を設定してください (prod と同一分割の保証)")
if folds_sha256 != CFG.GROUPED_FOLDS_SHA256:
    raise ValueError(f"grouped_folds.csv の SHA256 が不一致: {folds_sha256}")
fold_frame = pd.read_csv(folds_path)
if fold_frame["file_name"].duplicated().any() or set(fold_frame["file_name"]) != set(train_files):
    raise ValueError("fold CSV が全観測を一意にカバーしていない")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
valid_files = [n for n in train_files if fold_of_file[n] == CFG.FOLD]
fit_files = [n for n in train_files if fold_of_file[n] != CFG.FOLD]

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
"""),
        code("""# [4/7] consensus-class ラベルデータセット構築 + YOLO 学習 (prod と同一ハイパラ・seed)
def write_image(root, split, stem, file_name):
    out = root / "images" / split / f"{stem}.jpeg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        shutil.copy2((CFG.PREPROC_DIR / file_name).resolve(), out)

def write_label(root, split, stem, lines):
    out = root / "labels" / split / f"{stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\\n".join(lines), encoding="utf-8")

def prod_polygon_line(ann, width, height):  # master §B polygon_line と同一
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    points = np.asarray(max(polygons, key=len), dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{v:.6f}" for v in points.reshape(-1))

def file_clusters(file_name):
    \"\"\"file の全 record 個体を IoU>CLUSTER_IOU union-find で連結。
    返り値: (flat_index→合意数 distinct annotator 数, n_records)。
    flat index は record 順・record 内の非 iscrowd annotation 順。\"\"\"
    ids = file_to_ids[file_name]
    rles, record_of = [], []
    for r, image_id in enumerate(ids):
        for rle in gt_rles[image_id]:
            rles.append(rle); record_of.append(r)
    if not rles:
        return {}, len(ids)
    iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float64)
    clusters = cluster_from_iou(record_of, iou, CFG.CLUSTER_IOU)
    agree = {}
    for cluster in clusters:
        a = len({record_of[k] for k in cluster})
        for k in cluster:
            agree[k] = a
    return agree, len(ids)

def classed_polygon_line(ann, agree_val, width, height):
    \"\"\"prod_polygon_line と同一の polygon だが class=1 (consensus>=2) or 0 (weak)。\"\"\"
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    points = np.asarray(max(polygons, key=len), dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return f"{1 if agree_val >= 2 else 0} " + " ".join(f"{v:.6f}" for v in points.reshape(-1))

def build_arm_dataset(arm):
    \"\"\"per-record 学習サンプル (prod 構造) のまま class=合意数 を付与。
    (file,annotator) の各 record が自分の polygon を持ち、instance の
    class はその union cluster の distinct annotator 数で決める。\"\"\"
    root = CFG.YOLO_ROOT / f"{arm}_fold{CFG.FOLD}"
    stats = {"files": 0, "records": 0, "instances": 0,
             "cls_hist": {"0": 0, "1": 0}, "agree_hist": defaultdict(int),
             "train_samples": 0}
    for file_name in tqdm(train_files, desc=f"{arm} データセット構築"):
        info = images[file_to_ids[file_name][0]]
        stem = Path(file_name).stem
        width, height = info["width"], info["height"]
        ids = file_to_ids[file_name]
        agree, n_records = file_clusters(file_name)
        flat = 0
        rec_lines = []
        for image_id in ids:
            lines = []
            for ann in annotations[image_id]:
                if ann.get("iscrowd", 0):
                    continue
                a = agree.get(flat, 1)
                flat += 1
                stats["agree_hist"][int(a)] += 1
                line = classed_polygon_line(ann, a, width, height)
                if line:
                    lines.append(line)
                    stats["cls_hist"]["1" if a >= 2 else "0"] += 1
                    stats["instances"] += 1
            rec_lines.append(lines)
        if fold_of_file[file_name] == CFG.FOLD:
            # val は prod と同一 (先頭 record のみ) だが class は合意数を反映
            write_image(root, "val", stem, file_name)
            write_label(root, "val", stem, rec_lines[0])
            continue
        for j, lines in enumerate(rec_lines):
            copy_stem = f"{stem}__c{j}"
            write_image(root, "train", copy_stem, file_name)
            write_label(root, "train", copy_stem, lines)
            stats["train_samples"] += 1
        stats["files"] += 1; stats["records"] += n_records
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: weak\\n  1: strong\\n",
        encoding="utf-8")
    stats["agree_hist"] = dict(sorted(stats["agree_hist"].items()))
    return yaml_path, stats

def train_arm(arm):
    run_name = f"yolo_{arm}_fold{CFG.FOLD}"
    run_dir = CFG.RUNS_ROOT / run_name
    best_path, done_path = run_dir / "weights" / "best.pt", run_dir / "TRAINING_COMPLETE"
    yaml_path, stats = build_arm_dataset(arm)  # 再開時も統計を manifest に残すため毎回構築 (既存画像は skip)
    if done_path.exists() and best_path.exists():
        print(f"{arm}: 学習済み checkpoint を再利用 {best_path}")
        return best_path, stats
    started = time.time()
    YOLO(CFG.YOLO_MODEL).train(
        data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
        batch=CFG.YOLO_BATCH, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=False,
        mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
        optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
        degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
        project=str(CFG.RUNS_ROOT), name=run_name, exist_ok=True,
    )
    stats["train_minutes"] = round((time.time() - started) / 60, 1)
    done_path.write_text("ok", encoding="utf-8")
    return best_path, stats
"""),
        code("# [5/7] refiner モデル定義 (solar_filament/models/mediseg_models.py と同一)\n" + models_src),
        code("""# [6/7] 評価関数 (run_fold_oof_eval.py / build_v58 と同一経路)
refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
_ckpt = torch.load(CFG.REFINER_CKPT, map_location="cuda", weights_only=False)
refiner.load_state_dict(_ckpt.get("best_model") or _ckpt["model"], strict=True)
del _ckpt
REFINER_SHA256 = file_sha256(CFG.REFINER_CKPT)

def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32), np.zeros(0, np.int64)
    height, width = result.orig_shape
    out = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        out.append(mask >= 0.5)
    conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)
    cls = result.boxes.cls.detach().cpu().numpy().astype(np.int64) if result.boxes is not None else np.zeros(0, np.int64)
    return out, conf, cls

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

def evaluate_yolo(weights_path, label, use_cls=False):
    \"\"\"floor 推論 1 回 → 全提案を refine して保持 → (w0 × conf) sweep はメモリ上で採点。
    use_cls=True のとき score = conf if cls==1(strong) else conf*w0。
    prod は use_cls=False で conf のみ (w0 grid を潰す)。\"\"\"
    model = YOLO(str(weights_path))
    cache = []
    for name in tqdm(valid_files, desc=f"{label} 推論+refine"):
        path = CFG.PREPROC_DIR / name
        result = model.predict(str(path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.PREDICT_CONF_FLOOR,
                               iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, conf, cls = result_masks(result)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        cache.append((name, conf, cls, [float(m.sum()) for m in masks],
                      [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8))) for m in masks]))
    del model; torch.cuda.empty_cache()
    out = {}
    w0_grid = CFG.W0_SWEEP if use_cls else (1.0,)
    for w0 in w0_grid:
        out[f"w0={w0}"] = {}
        for c in CFG.CONF_SWEEP:
            meter = PanopticQuality()
            n_pred = 0
            n_weak = 0
            for name, conf, cls, areas, refined in cache:
                score = conf * np.where(cls == 1, 1.0, w0)
                keep = [i for i in range(len(score)) if score[i] >= c and areas[i] >= CFG.MIN_AREA]
                n_weak += int(sum(1 for i in keep if cls[i] == 0))
                preds = apply_ownership([mask_utils.decode(refined[i]) != 0 for i in keep])
                pred_rles = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
                n_pred += len(pred_rles)
                for image_id in file_to_ids[name]:
                    truth = gt_rles[image_id]
                    meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
            out[f"w0={w0}"][str(c)] = {**meter.summary(), "preds_per_image": n_pred / max(len(valid_files), 1),
                                       "weak_kept_per_image": n_weak / max(len(valid_files), 1)}
            print(label, f"w0={w0}", c,
                  {k: round(v, 4) if isinstance(v, float) else v for k, v in out[f"w0={w0}"][str(c)].items()})
    return out if use_cls else out["w0=1.0"]
"""),
        code("""# [7/7] 実行: prod baseline → 各 arm 学習 → 評価 → ゲート判定、manifest を逐次保存
def save_manifest(manifest):
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

CONTRACT = {"experiment_id": "v93_consensus_class", "fold": CFG.FOLD, "arms": list(CFG.ARMS),
            "protocol_sha256": CFG.PROTOCOL_SHA256, "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "cluster_iou": CFG.CLUSTER_IOU,
            "conf_sweep": list(CFG.CONF_SWEEP), "primary_conf": CFG.PRIMARY_CONF,
            "w0_sweep": list(CFG.W0_SWEEP),
            "gate": f"max_w0[PQ(arm,{CFG.PRIMARY_CONF}) - PQ(prod,{CFG.PRIMARY_CONF})] >= +{CFG.GATE_DELTA}"}
manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8")) if CFG.MANIFEST.exists() else {}
if manifest and manifest.get("contract") != CONTRACT:
    raise ValueError("既存 manifest と入力契約が違う。履歴保持のため別パスを指定してください")
manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}

if "prod" not in manifest["results"]:
    manifest["results"]["prod"] = {"weights": str(CFG.PROD_YOLO), "by_conf": evaluate_yolo(CFG.PROD_YOLO, "prod")}
    save_manifest(manifest)
prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]
print({"prod@primary": round(prod_pq, 6), "既知値": CFG.BASELINE_PQ, "差": round(prod_pq - CFG.BASELINE_PQ, 6)})
if abs(prod_pq - CFG.BASELINE_PQ) > 0.002:
    print("⚠️ prod の再現値が既知値から 0.002 以上ずれている — 配線/入力の差を確認 (判定は paired Δ で行う)")

for arm in CFG.ARMS:
    if arm in manifest["results"]:
        continue
    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_w0 = evaluate_yolo(weights, arm, use_cls=True)  # {"w0=..": {conf: metrics}}
    primary = str(CFG.PRIMARY_CONF)
    deltas = {w0: by_w0[w0][primary]["pq"] - prod_pq for w0 in by_w0}
    best_w0 = max(deltas, key=deltas.get)
    delta = deltas[best_w0]
    flat = {(w0, c): by_w0[w0][c]["pq"] for w0 in by_w0 for c in by_w0[w0]}
    best_pair = max(flat, key=flat.get)  # 探索値のみ。判定には使わない
    verdict = "promote_to_fold01" if delta >= CFG.GATE_DELTA else ("reject_noise" if delta >= 0 else "reject")
    manifest["results"][arm] = {"weights": str(weights), "weights_sha256": file_sha256(weights),
                                "by_w0": by_w0,
                                "delta_primary_vs_prod_by_w0": deltas,
                                "best_w0": best_w0, "delta_primary_vs_prod": delta,
                                "delta_sparse_primary_vs_prod": by_w0[best_w0][primary]["pq_sparse_r077"]
                                    - manifest["results"]["prod"]["by_conf"][primary]["pq_sparse_r077"],
                                "exploratory_best_w0_conf": list(best_pair),
                                "exploratory_best_delta": flat[best_pair] - prod_pq,
                                "verdict": verdict}
    manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    print(arm, {"best_w0": best_w0, "Δ@primary": round(delta, 4), "verdict": verdict})

summary = {}
for a, r in manifest["results"].items():
    if "by_conf" in r:
        summary[a] = {c: round(m["pq"], 4) for c, m in r["by_conf"].items()}
    else:
        summary[a] = {f"{w0}@{c}": round(m["pq"], 4)
                      for w0, m0 in r["by_w0"].items() for c, m in m0.items()}
print(json.dumps(summary, indent=2))
print("manifest:", CFG.MANIFEST)
"""),
    ]
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=(*ARMS, "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--folds-sha256", default="",
                        help="凍結 grouped_folds.csv の SHA256 (既定空=notebook 内で手入力)")
    args = parser.parse_args()
    arms = list(ARMS) if args.arm == "both" else [args.arm]
    models_src, models_origin = _load_models_source()
    protocol = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")
                              + models_src.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    notebook = {"cells": build_cells(arms, models_src, protocol,
                                     args.folds_sha256),
                "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                             "kernelspec": {"display_name": "Python 3", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    label = "both" if len(arms) > 1 else arms[0]
    out = args.out_dir / f"solar_filament_v93_consensus_class_fold{FOLD}_colab.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print({"notebook": str(out), "models_source": models_origin, "protocol_sha256": protocol[:16]})


if __name__ == "__main__":
    main()
