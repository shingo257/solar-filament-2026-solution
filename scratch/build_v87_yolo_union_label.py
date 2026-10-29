"""v87 — YOLO 検出器を「アノテータ union ラベル + 合意度重み」で再学習する fold2 パイロット (PLAN D3)。

## 仮説

prod (V30/V52) の YOLO は `build_fold_dataset` で **(file, annotator record) を独立サンプル**
として複製学習している (`build_master_pipeline_notebook.py` §B)。この方式は

* 合意数 a のフィラメントが a 回正例になる (= 暗黙の合意度重み) 一方で、
* 1-of-R のフィラメントは残り R-1 枚のコピーで **背景(負例)として明示的に教えられる**、
* 境界はアノテータごとに揺れる (同じ物体に矛盾した polygon が付く)

という性質を持つ。v87 は「同一物理フィラメントを IoU>0.5 でクラスタ化し、クラスタ union を
1 個体ラベルにする」ことで境界の矛盾と「描かれなかったコピーでの負例化」を除き、
detector-blind FN (FN の約 68%, v62) を減らせるかを検証する。

## arm

| arm | ラベル | 1 file あたりのサンプル数 | 合意度の扱い |
|---|---|---|---|
| `union_hard` | 全クラスタの union polygon (singleton 含む) | R 枚の同一コピー | 重みなし (全クラスタ等重み) |
| `union_agree` | コピー j (j=1..R) には合意数 >= j のクラスタだけ | R 枚 | 合意数 a のクラスタが a 回正例 |

* どちらも 1 file = R サンプルにして **prod と 1 epoch の反復数を揃える** (変更点をラベルだけに限定)。
* `union_agree` は Ultralytics に per-instance loss weight が無いための近似 (サンプリング重み)。
  per-instance 重みを入れるには `v8SegmentationLoss` の subclass が必要だが、Ultralytics の
  内部 API は 8.3 系のマイナー更新で頻繁に変わり (TaskAlignedAssigner の戻り値・
  loss の reduction など)、Colab 上で pin しないと壊れる。コピー複製方式は
  学習コードに一切手を入れないため prod との差分が「ラベル集合」だけになり、paired 比較が明快。
  代償として、合意数 < j のクラスタはコピー j で背景扱いになる (prod と同じ負例化が残る)。
  `union_hard` との差で「合意度重み」の効果を、prod との差で「union 化」の効果を分離する。

## 評価 (fold2 OOF、prod と同一経路)

`run_fold_oof_eval.py` / `build_v58_conf_sweep_probe.py` と同じ後処理:
YOLO predict(imgsz1280, conf=0.08 floor, iou0.55, agnostic_nms, retina_masks)
→ conf>=c & area>=20 → EdgeAtt refiner fold2 (`mediseg_edgeattnet_refiner_fold2_resume.pt`)
320px crop / context1.65 → shrink@0.45 → drop_contained(0.90) → 面積昇順 greedy_disjoint(20)
→ 公式 PQ (annotator record × image 毎、IoU>0.5 の閾値越えを全て数える self-eval 準拠)。

* 同 notebook 内で **prod YOLO (`runs/yolo_fold_2/weights/best.pt`) を同じ refiner で採点** し
  paired baseline とする (conf 0.26 で 0.444238 が再現されれば配線 OK)。
* conf sweep {0.15, 0.20, 0.26, 0.32}。**事前登録の主判定は conf=0.26 (prod 操作点) の Δ**。
  sweep の max は探索値で、採用には fold0/1 での独立確認が必要 (v58 の選択バイアス教訓)。
* 副指標: sparse-label PQ' (r=0.77, dariushafshar 式, RESEARCH_LOG §2.28)
  `PQ' = r·ΣIoU / (r·TP + ½FP + ½(1−r)TP + ½r·FN)`。repo に実装が無いためここで新規実装。

## ゲート (結果を見る前に固定)

`PQ(arm, conf=0.26) − PQ(prod, conf=0.26) >= +0.005` (fold2 OOF) → fold0/1 確認へ昇格。
`[0, +0.005)` はノイズ棄却、負は棄却。

## 使い方

    python scratch/build_v87_yolo_union_label.py --arm both      # 2 arm を 1 notebook で順番に
    python scratch/build_v87_yolo_union_label.py --arm union_agree

出力: solar_filament_v87_yolo_union_label_fold2_{arm}_colab.ipynb (Colab GPU、A100/L4 推奨)
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
ARMS = ("union_hard", "union_agree")


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
# ラベル union の純関数群。notebook に埋め込み、tests/test_v87_union_labels.py で単体テストする。
# soft_target.cluster_gt_instances と同じ規則 (union-find、同一 record 同士は直接・間接とも
# 併合しない) を、pycocotools の RLE IoU 行列に対して適用する (2048² bool の総当たりは遅いため)。
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


def union_cluster_masks(masks, record_of, clusters):
    """各クラスタの union mask と合意数 (クラスタに含まれる distinct record 数) を返す。"""
    import numpy as np
    out = []
    for cluster in clusters:
        union = np.logical_or.reduce([masks[k] for k in cluster])
        out.append((union, len({record_of[k] for k in cluster})))
    return out


def mask_to_yolo_line(mask, min_area=20):
    """union mask → YOLO-seg 1 行。外輪郭の最大成分のみ (YOLO-seg は 1 行 1 polygon)。
    返り値: (line or None, n_components)。"""
    import cv2
    import numpy as np
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > 0 or len(c) >= 3]
    if not contours or int(mask.sum()) < min_area:
        return None, len(contours)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < 3:
        return None, len(contours)
    contour[:, 0] = np.clip(contour[:, 0] / width, 0, 1)
    contour[:, 1] = np.clip(contour[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{v:.6f}" for v in contour.reshape(-1)), len(contours)


def copies_for_arm(unions, n_records, arm):
    """arm ごとの R 枚のラベル集合 (各要素 = そのコピーに入れるクラスタ index のリスト)。
    union_hard : 全コピーに全クラスタ
    union_agree: コピー j (1..R) に合意数 >= j のクラスタ → 合意数 a のクラスタは a 回正例"""
    n_records = max(int(n_records), 1)
    if arm == "union_hard":
        return [list(range(len(unions))) for _ in range(n_records)]
    if arm == "union_agree":
        return [[k for k, (_, agree) in enumerate(unions) if agree >= j]
                for j in range(1, n_records + 1)]
    raise ValueError(f"unknown arm: {arm}")


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
        markdown(f"""# v87 — YOLO union ラベル + 合意度重み 再学習 (fold{FOLD} パイロット)

arm: {', '.join(f'`{a}`' for a in arms)} + paired baseline `prod` (既存 `runs/yolo_fold_{FOLD}`)。
ハイパラ・分割・seed は prod (`build_master_pipeline_notebook.py` §B / `build_v30_prize_colab.py`) と同一、
変えるのは **YOLO の学習ラベル集合だけ**。評価は prod 後処理 + EdgeAtt refiner fold{FOLD}。

**事前登録ゲート**: PQ(arm, conf={PRIMARY_CONF}) − PQ(prod, conf={PRIMARY_CONF}) >= +{GATE_DELTA}
(参考: prod の既知値 {BASELINE_PQ})。conf sweep {list(CONF_SWEEP)} の最大値は探索値として記録のみ。

学習済み arm は `TRAINING_COMPLETE` マーカーでスキップ、評価結果は manifest に逐次追記 (途中再開可)。
GPU 目安: A100 で 1 arm 約 2〜3h + 評価 約 20〜30 分/YOLO (T4 は 2〜3 倍)。
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
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v87_yolo_union_label"
    YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v87_yolo_union_label"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v87")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    PROD_YOLO = DRIVE_ROOT / "runs" / f"yolo_fold_{FOLD}" / "weights" / "best.pt"
    REFINER_CKPT = ARTIFACT_ROOT / f"mediseg_edgeattnet_refiner_fold{FOLD}_resume.pt"
    MANIFEST = ARTIFACT_ROOT / "v87_yolo_union_label_fold{FOLD}_{arm_label}_manifest.json"
    # --- YOLO (build_v30_prize_colab.py / master §B と同一。変更禁止) ---
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    # --- union ラベル ---
    CLUSTER_IOU = {CLUSTER_IOU}
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
        code("# [2/7] union ラベル純関数 (scratch/build_v87_yolo_union_label.py の UNION_SRC、単体テスト済み)\n" + UNION_SRC),
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
        code("""# [4/7] union ラベルデータセット構築 + YOLO 学習 (prod と同一ハイパラ・seed)
def write_image(root, split, stem, file_name):
    out = root / "images" / split / f"{stem}.jpeg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        shutil.copy2((CFG.PREPROC_DIR / file_name).resolve(), out)

def write_label(root, split, stem, lines):
    out = root / "labels" / split / f"{stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\\n".join(lines), encoding="utf-8")

def prod_polygon_line(ann, width, height):  # master §B polygon_line と同一 (val ラベル用)
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    points = np.asarray(max(polygons, key=len), dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{v:.6f}" for v in points.reshape(-1))

def file_unions(file_name):
    ids = file_to_ids[file_name]
    rles, record_of = [], []
    for r, image_id in enumerate(ids):
        for rle in gt_rles[image_id]:
            rles.append(rle); record_of.append(r)
    if not rles:
        return [], len(ids), 0
    iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float64)
    clusters = cluster_from_iou(record_of, iou, CFG.CLUSTER_IOU)
    masks = [mask_utils.decode(rle).astype(bool) for rle in rles]
    return union_cluster_masks(masks, record_of, clusters), len(ids), len(rles)

def build_arm_dataset(arm):
    root = CFG.YOLO_ROOT / f"{arm}_fold{CFG.FOLD}"
    stats = {"files": 0, "instances_in": 0, "clusters": 0, "agree_hist": defaultdict(int),
             "multi_component_unions": 0, "dropped_unions": 0, "train_samples": 0, "train_instances": 0}
    for file_name in tqdm(train_files, desc=f"{arm} データセット構築"):
        info = images[file_to_ids[file_name][0]]
        stem = Path(file_name).stem
        if fold_of_file[file_name] == CFG.FOLD:
            # val は prod と同一 (先頭 record の生ラベル)。early stopping の基準を揃えるため変えない
            write_image(root, "val", stem, file_name)
            write_label(root, "val", stem, [l for l in (prod_polygon_line(a, info["width"], info["height"])
                        for a in annotations[file_to_ids[file_name][0]] if not a.get("iscrowd", 0)) if l])
            continue
        unions, n_records, n_in = file_unions(file_name)
        lines = []
        for mask, agree in unions:
            line, n_comp = mask_to_yolo_line(mask, CFG.MIN_AREA)
            stats["multi_component_unions"] += int(n_comp > 1)
            stats["dropped_unions"] += int(line is None)
            stats["agree_hist"][int(agree)] += 1
            lines.append(line)
        stats["files"] += 1; stats["instances_in"] += n_in; stats["clusters"] += len(unions)
        for j, keep in enumerate(copies_for_arm(unions, n_records, arm)):
            copy_stem = f"{stem}__{arm}_c{j}"
            write_image(root, "train", copy_stem, file_name)
            copy_lines = [lines[k] for k in keep if lines[k]]
            write_label(root, "train", copy_stem, copy_lines)
            stats["train_samples"] += 1; stats["train_instances"] += len(copy_lines)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
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
        seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
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

def evaluate_yolo(weights_path, label):
    \"\"\"floor 推論 1 回 → 全提案を refine して保持 → conf sweep はメモリ上で採点 (v58 と同型)。\"\"\"
    model = YOLO(str(weights_path))
    cache = []
    for name in tqdm(valid_files, desc=f"{label} 推論+refine"):
        path = CFG.PREPROC_DIR / name
        result = model.predict(str(path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.PREDICT_CONF_FLOOR,
                               iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, conf = result_masks(result)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        cache.append((name, conf, [float(m.sum()) for m in masks],
                      [mask_utils.encode(np.asfortranarray(refine_proposal(image, m).astype(np.uint8))) for m in masks]))
    del model; torch.cuda.empty_cache()
    out = {}
    for c in CFG.CONF_SWEEP:
        meter = PanopticQuality()
        n_pred = 0
        for name, conf, areas, refined in cache:
            keep = [i for i in range(len(conf)) if conf[i] >= c and areas[i] >= CFG.MIN_AREA]
            preds = apply_ownership([mask_utils.decode(refined[i]) != 0 for i in keep])
            pred_rles = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8))) if preds else []
            n_pred += len(pred_rles)
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
        out[str(c)] = {**meter.summary(), "preds_per_image": n_pred / max(len(valid_files), 1)}
        print(label, c, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[str(c)].items()})
    return out
"""),
        code("""# [7/7] 実行: prod baseline → 各 arm 学習 → 評価 → ゲート判定、manifest を逐次保存
def save_manifest(manifest):
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

CONTRACT = {"experiment_id": "v87_yolo_union_label", "fold": CFG.FOLD, "arms": list(CFG.ARMS),
            "protocol_sha256": CFG.PROTOCOL_SHA256, "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "cluster_iou": CFG.CLUSTER_IOU,
            "conf_sweep": list(CFG.CONF_SWEEP), "primary_conf": CFG.PRIMARY_CONF,
            "gate": f"PQ(arm,{CFG.PRIMARY_CONF}) - PQ(prod,{CFG.PRIMARY_CONF}) >= +{CFG.GATE_DELTA}"}
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
    by_conf = evaluate_yolo(weights, arm)
    delta = by_conf[str(CFG.PRIMARY_CONF)]["pq"] - prod_pq
    best_conf = max(by_conf, key=lambda c: by_conf[c]["pq"])
    verdict = "promote_to_fold01" if delta >= CFG.GATE_DELTA else ("reject_noise" if delta >= 0 else "reject")
    manifest["results"][arm] = {"weights": str(weights), "weights_sha256": file_sha256(weights),
                                "by_conf": by_conf, "delta_primary_vs_prod": delta,
                                "delta_sparse_primary_vs_prod": by_conf[str(CFG.PRIMARY_CONF)]["pq_sparse_r077"]
                                    - manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq_sparse_r077"],
                                "exploratory_best_conf": best_conf,
                                "exploratory_best_delta": by_conf[best_conf]["pq"] - prod_pq,
                                "verdict": verdict}
    manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    print(arm, {"Δ@primary": round(delta, 4), "verdict": verdict})

print(json.dumps({a: {c: round(r["by_conf"][c]["pq"], 4) for c in r["by_conf"]}
                  for a, r in manifest["results"].items()}, indent=2))
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
    out = args.out_dir / f"solar_filament_v87_yolo_union_label_fold{FOLD}_{label}_colab.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print({"notebook": str(out), "models_source": models_origin, "protocol_sha256": protocol[:16]})


if __name__ == "__main__":
    main()
