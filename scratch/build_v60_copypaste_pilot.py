"""copy_paste augmentation パイロット notebook builder — v57(§6.16)の代替機構検証。

## 目的

v57 の dense2x(密集サンプル複製)は dense FN を減らしたが、シーン全体の複製が
「そのシーンで過剰提案する」 prior を学ばせて FP 爆増(+229/+146)し refined PQ
が悪化した。**シーン複製ではなくインスタンス単位の貼付**(ultralytics 組込みの
copy_paste augmentation)なら、密集の「インスタンスが多い」という信号だけを
増やせて、シーンごとの複製が持つ副作用を避けられる可能性がある。

学習レシピはV30と完全同一(yolo11m-seg、imgsz 1280、70epoch、batch 4、
AdamW lr0=7e-4)。変数は `copy_paste` ハイパラだけ:

  1. `baseline`  — V30のまま(copy_paste=0)。旧 v57 baseline は再利用しない。
                   この v60 契約で実行した baseline のみ、検証済み記録の明示 opt-in に対応
  2. `copypaste` — copy_paste=0.2(デフォルトmode=flip)

各構成を seed (3407, 8407) で学習し、対象foldの OOF で評価する。
評価はv57と同一の2段(raw PQ / refined PQ + dense・sparseスライス別FN)。
判定もv57と同じ規約: 2seedで符号一致 + dense FN が減ること。

## 使い方

    python scratch/build_v60_copypaste_pilot.py --fold 0

出力: solar_filament_v60_copypaste_fold{N}_colab.ipynb (Colab A100推奨)
CPU 構造検証は GPU 学習・独立 OOF 受入を代替しない。
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")

PROTOCOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n") + MODELS.replace("\r\n", "\n").encode()).hexdigest()
GUARDS = 'def atomic_save(path, value, is_checkpoint=False):\n    import os\n    import tempfile\n    path = Path(path)\n    payload = None if is_checkpoint else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode(\'utf-8\')\n    temporary = None\n    try:\n        with tempfile.NamedTemporaryFile(mode=\'wb\', dir=path.parent, suffix=\'.tmp\', delete=False) as handle:\n            temporary = Path(handle.name)\n            if is_checkpoint:\n                torch.save(value, handle)\n            else:\n                handle.write(payload)\n            handle.flush()\n            os.fsync(handle.fileno())\n        os.replace(temporary, path)\n    finally:\n        if temporary is not None and temporary.exists():\n            temporary.unlink()\n\ndef file_sha256(path):\n    digest = hashlib.sha256()\n    with Path(path).open("rb") as handle:\n        for chunk in iter(lambda: handle.read(1024 * 1024), b""):\n            digest.update(chunk)\n    return digest.hexdigest()\n\ndef require_input_contract(path, expected):\n    if not Path(path).is_file() or json.loads(Path(path).read_text(encoding="utf-8")) != expected:\n        raise ValueError("Input contract differs or is absent; preserve artifacts and choose a fresh path")\n\ndef runtime_measurement(scope, started):\n    try:\n        peak, peak_note = int(torch.cuda.max_memory_allocated()), None\n    except Exception as error:\n        peak, peak_note = None, f"CUDA statistics unavailable: {type(error).__name__}"\n    return {"service": "Google Colab", "project": str(CFG.DRIVE_ROOT),\n            "notebook": f"solar_filament_v60_copypaste_fold{CFG.FOLD}_colab.ipynb",\n            "gpu_name": torch.cuda.get_device_name(0), "completed_at_utc": datetime.now(timezone.utc).isoformat(),\n            "measurement_scope": scope, "duration_seconds": round(time.time() - started, 3),\n            "peak_vram_bytes": peak, "peak_vram_unavailable_reason": peak_note,\n            "compute_units_used": None, "compute_units_note": "Unavailable from runtime API; record from Colab UI."}\n\ndef validate_run_row(row):\n    if (not isinstance(row, dict) or row.get("config") not in CFG.CONFIGS or row.get("seed") not in CFG.SEEDS\n            or row.get("fold") != CFG.FOLD or row.get("input_contract") != PILOT_CONTRACT\n            or row.get("copy_paste") != (CFG.COPY_PASTE if row["config"] == "copypaste" else 0.0)):\n        raise ValueError("Run arm/seed/fold/input contract differs")\n    for mode in ("raw", "refined"):\n        metrics = row.get(mode)\n        if not isinstance(metrics, dict):\n            raise ValueError("Missing quality metrics")\n        for name in ("pq", "sq", "rq"):\n            value = metrics.get(name)\n            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:\n                raise ValueError("Invalid quality metric")\n        for name in ("tp", "fp", "fn"):\n            if type(metrics.get(name)) is not int or metrics[name] < 0:\n                raise ValueError("Invalid match count")\n    for name in ("fn_dense_val", "fn_sparse_val"):\n        if type(row.get(name)) is not int or row[name] < 0:\n            raise ValueError("Invalid density count")\n    checkpoint = Path(row.get("checkpoint", ""))\n    if not checkpoint.is_file() or row.get("checkpoint_sha256") != file_sha256(checkpoint):\n        raise ValueError("Checkpoint bytes differ or are absent")\n    return row\n\ndef read_pilot_manifest():\n    if not CFG.MANIFEST.exists():\n        return {"experiment_id": f"v60_copypaste_fold{CFG.FOLD}", "input_contract": PILOT_CONTRACT, "runs": []}\n    manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8"))\n    if (not isinstance(manifest, dict) or manifest.get("experiment_id") != f"v60_copypaste_fold{CFG.FOLD}"\n            or manifest.get("input_contract") != PILOT_CONTRACT or not isinstance(manifest.get("runs"), list)):\n        raise ValueError("Manifest input contract differs or is absent; retain historical files")\n    seen = set()\n    for row in manifest["runs"]:\n        validate_run_row(row)\n        key = (row["config"], row["seed"])\n        if key in seen:\n            raise ValueError("Duplicate pilot run")\n        seen.add(key)\n    return manifest\n\ndef reuse_baselines(source):\n    if not CFG.REUSE_VERIFIED_BASELINE:\n        return []\n    if not isinstance(source, dict) or source.get("input_contract") != PILOT_CONTRACT:\n        raise ValueError("Baseline equivalence is unverified; rerun paired baseline")\n    result, seen = [], set()\n    for row in source.get("runs", []):\n        if row.get("config") != "baseline":\n            continue\n        validate_run_row(row)\n        key = (row["config"], row["seed"])\n        if key in seen:\n            raise ValueError("Duplicate reused baseline")\n        seen.add(key)\n        result.append({**row, "reused_from": "explicit_contract_checked_baseline"})\n    return result\n\nfrom ultralytics import YOLO\ninitial_yolo = YOLO(CFG.YOLO_MODEL)\nINITIAL_YOLO_PATH = Path(initial_yolo.ckpt_path).resolve(strict=True)\nif initial_yolo.task != "segment":\n    raise ValueError("Copy-paste pilot requires the segmentation pretrained model")\ndel initial_yolo\nPILOT_CONTRACT = {\n    "schema_version": 1, "fold": CFG.FOLD, "protocol_sha256": CFG.PROTOCOL_SHA256,\n    "annotations_sha256": file_sha256(TRAIN_JSON),\n    "folds_sha256": file_sha256(CFG.ARTIFACT_ROOT / "grouped_folds.csv"),\n    "initial_yolo_sha256": file_sha256(INITIAL_YOLO_PATH),\n    "refiner_sha256": file_sha256(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt"),\n    "preprocessed_sha256": hashlib.sha256(json.dumps({name: file_sha256(CFG.PREPROC_DIR / name)\n        for name in sorted(train_files_fold + valid_files_fold)}, sort_keys=True).encode()).hexdigest(),\n    "recipe": {name: getattr(CFG, name) for name in ("YOLO_MODEL", "YOLO_EPOCHS", "YOLO_IMGSZ", "YOLO_BATCH",\n        "YOLO_PATIENCE", "YOLO_MASK_RATIO", "CROP_SIZE", "CROP_CONTEXT", "CONF", "PREDICT_CONF_FLOOR",\n        "PREDICT_IOU", "REFINER_THRESHOLD", "DROP_THRESHOLD", "MIN_AREA", "DENSE_QUANTILE", "COPY_PASTE")},\n    "seeds": list(CFG.SEEDS), "configs": list(CFG.CONFIGS),\n    "runtime_versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,\n        "opencv": cv2.__version__, "ultralytics": version("ultralytics"), "pycocotools": version("pycocotools")},\n}\nPILOT_CONTRACT = json.loads(json.dumps(PILOT_CONTRACT, sort_keys=True, allow_nan=False))\nread_pilot_manifest()  # No legacy/mixed results may reach GPU evaluation.\n'

# 標準refinerパイプラインの全fold OOF基準(README「現在地」、v57と同一)
BASELINE_PQ_REFINED = {0: 0.4393, 1: 0.4415, 2: 0.4471}
# 生YOLO基準(V44計測、fold0のみ既知)
BASELINE_PQ_RAW = {0: 0.3977}
DENSE_QUANTILE = 0.75
COPY_PASTE = 0.2
SEEDS = (3407, 8407)
CONFIGS = ("baseline", "copypaste")


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells(fold):
    baseline_refined = BASELINE_PQ_REFINED[fold]
    baseline_raw = BASELINE_PQ_RAW.get(fold)

    cells = [
        markdown(
            """# copy_paste augmentation パイロット — fold{fold}、検出器側レバー(v57の代替機構)

v57 で密集サンプル複製(dense2x)は dense FN を減らしたが FP 爆増で不採用。
シーン複製の代わりに **インスタンス貼付(ultralytics copy_paste)** で密集
シーンのリコールを上げられるか検証する。

| 構成 | 内容 |
|---|---|
| `baseline` | V30完全同一(copy_paste=0)。旧 baseline の自動再利用なし |
| `copypaste` | copy_paste={cp} のみ追加(他のハイパラはV30不変) |

学習レシピはV30と完全同一(yolo11m-seg/imgsz1280/70epoch/AdamW lr0=7e-4)。
fold{fold}基準: refined PQ {baseline_refined} / raw YOLO PQ {baseline_raw}。
GPU必須、A100推奨。manifestはDriveへ逐次追記(中断再開可)。
""".format(
                fold=fold,
                cp=COPY_PASTE,
                baseline_refined=baseline_refined,
                baseline_raw=(baseline_raw if baseline_raw is not None else "未計測(paired baselineが参照)"),
            )
        ),
        code(
            """# [1/8] Drive接続、設定、前提の確認
!pip -q install ultralytics pycocotools
import hashlib, json, math, random, shutil, sys, time, warnings
from datetime import datetime, timezone
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from google.colab import drive
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
drive.mount("/content/drive")

class CFG:
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v60_copypaste")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    YOLO_ROOT = LOCAL_ROOT / "yolo_datasets"
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v60_copypaste"
    FOLD = %d
    PROTOCOL_SHA256 = %r
    REUSE_VERIFIED_BASELINE = False
    # --- build_v30_prize_colab.py と完全同一。触らないこと ---
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    # --- OOF評価は確定構成(build_v56/v57系と同一。refinerは既存ckptを使い回す) ---
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    CONF = 0.26
    PREDICT_CONF_FLOOR = 0.08
    PREDICT_IOU = 0.55
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    BASELINE_PQ_REFINED = %r
    DENSE_QUANTILE = %r
    COPY_PASTE = %r
    SEEDS = %r
    CONFIGS = %r
    VERIFIED_BASELINE_MANIFEST = DRIVE_ROOT / "artifacts" / "v60_verified_baseline_fold%d_manifest.json"
    MANIFEST = DRIVE_ROOT / "artifacts" / "v60_copypaste_fold%d_manifest.json"

for path in [CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.YOLO_ROOT, CFG.RUNS_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

required = {
    "competition data": CFG.DATA_CACHE,
    "grouped_folds.csv": CFG.ARTIFACT_ROOT / "grouped_folds.csv",
    "前処理画像キャッシュ": CFG.PREPROC_CACHE,
    f"fold{CFG.FOLD} 標準refiner": CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
}
missing = {k: str(v) for k, v in required.items() if not v.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({k: "OK" for k in required})
print({"GPU": torch.cuda.get_device_name(0)})
"""  # noqa: UP031 - preserve embedded dictionary braces
            % (fold, PROTOCOL_SHA256, baseline_refined, DENSE_QUANTILE, COPY_PASTE, SEEDS, CONFIGS, fold, fold)
        ),
        code(
            """# [2/8] COCOとfoldを読み、密度スライス用のcutを計算し、前処理画像をローカルSSDへ展開
def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"学習annotation JSONが見つかりません: {base}")
    return matches[0].parents[2]

TRAIN_JSON = next(find_dataset_root(CFG.DATA_CACHE).rglob(
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

images = {item["id"]: item for item in coco["images"]}
annotations_by_image = defaultdict(list)
for ann in coco["annotations"]:
    annotations_by_image[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

def annotation_rank(image_id):
    anns = annotations_by_image[image_id]
    return len(anns), sum(float(a.get("area", 0)) for a in anns)

file_to_reference_id = {
    name: max(ids, key=annotation_rank) for name, ids in file_to_ids.items()
}
train_files = sorted(file_to_ids)

fold_frame = pd.read_csv(CFG.ARTIFACT_ROOT / "grouped_folds.csv")
if fold_frame["file_name"].duplicated().any() or not fold_frame["fold"].isin([0, 1, 2]).all():
    raise ValueError("Invalid or duplicate fold assignments")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
if set(train_files) != set(fold_of_file):
    raise ValueError("Fold assignments do not cover exactly the consumed files")
train_files_fold = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
valid_files_fold = [n for n in train_files if fold_of_file[n] == CFG.FOLD]
if not train_files_fold or not valid_files_fold:
    raise ValueError("Both training and validation partitions must be nonempty")
assert not (set(train_files_fold) & set(valid_files_fold)), "fold分割が壊れている"

def noncrowd_count(image_id):
    return sum(1 for a in annotations_by_image[image_id] if not a.get("iscrowd", 0))

# v57と同じ定義: cutは学習サンプル(file,annotator)のGT数q75(dense/sparseスライス用)
train_sample_counts = [
    noncrowd_count(image_id)
    for name in train_files_fold
    for image_id in file_to_ids[name]
]
DENSE_CUT = float(np.quantile(train_sample_counts, CFG.DENSE_QUANTILE))
val_density = {
    name: float(np.mean([noncrowd_count(i) for i in file_to_ids[name]]))
    for name in valid_files_fold
}

for name in tqdm(train_files_fold + valid_files_fold, desc="前処理画像をSSDへ"):
    local = CFG.PREPROC_DIR / name
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG.PREPROC_CACHE / name, local)

print({"観測": len(train_files), "除外train": len(train_files_fold),
       "検証(OOF)": len(valid_files_fold),
       "dense_cut": DENSE_CUT,
       "dense検証画像数": int(sum(v >= DENSE_CUT for v in val_density.values()))})
""" + GUARDS
        ),
        code(
            """# [3/8] YOLOデータセット構築(V30と完全同一 — copy_pasteはデータセットではなく学習kwargなので1構成のみ)
def polygon_line(ann, width, height):
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    # MAGFiLO specifies one polygon per filament. Largest is a safe fallback for
    # malformed/multipart records without turning one filament into many labels.
    polygon = max(polygons, key=len)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.reshape(-1))

def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)

def write_variant(fold_root, split, file_name, image_id, suffix):
    info = images[image_id]
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{stem}.jpeg"
    label_out = fold_root / "labels" / split / f"{stem}.txt"
    link_or_copy((CFG.PREPROC_DIR / file_name).resolve(), image_out)
    lines = []
    for ann in annotations_by_image[image_id]:
        if ann.get("iscrowd", 0):
            continue
        line = polygon_line(ann, info["width"], info["height"])
        if line:
            lines.append(line)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(lines), encoding="utf-8")

def build_fold_dataset():
    # データセットはV30と完全同一の1種のみ(baseline/copypaste共用)
    root = CFG.YOLO_ROOT / "shared" / f"fold_{CFG.FOLD}"
    contract_path = root / "INPUT_CONTRACT.json"
    if (root / "data.yaml").exists():
        require_input_contract(contract_path, PILOT_CONTRACT)
        print(f"既存データセットを再利用 {root}")
        return root / "data.yaml"
    if root.exists() and any(root.iterdir()):
        raise ValueError("Incomplete dataset: preserve it and choose a fresh pilot path")
    for name in tqdm(train_files_fold, desc=f"build fold{CFG.FOLD} train"):
        for image_id in file_to_ids[name]:
            write_variant(root, "train", name, image_id, f"__labeler_{image_id}")
    for file_name in tqdm(valid_files_fold, desc=f"build fold{CFG.FOLD} val"):
        image_id = file_to_reference_id[file_name]
        write_variant(root, "val", file_name, image_id, "")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
        encoding="utf-8",
    )
    atomic_save(contract_path, PILOT_CONTRACT)
    return yaml_path
"""
        ),
        code(
            """# [4/8] 学習(V30と同一ハイパラ、copypaste構成のみcopy_paste追加) — runごとに再開対応
from ultralytics import YOLO

def training_kwargs(config, seed, yaml_path, run_name):
    return dict(
            data=str(yaml_path),
            epochs=CFG.YOLO_EPOCHS,
            imgsz=CFG.YOLO_IMGSZ,
            batch=CFG.YOLO_BATCH,
            device=0,
            workers=4,
            patience=CFG.YOLO_PATIENCE,
            seed=seed,
            deterministic=True,
            single_cls=True,
            mask_ratio=CFG.YOLO_MASK_RATIO,
            overlap_mask=False,
            amp=True,
            cache="disk",
            optimizer="AdamW",
            lr0=7e-4,
            lrf=0.05,
            weight_decay=5e-4,
            warmup_epochs=3,
            degrees=180,
            translate=0.04,
            scale=0.25,
            fliplr=0.5,
            flipud=0.5,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.18,
            mosaic=0.20,
            close_mosaic=10,
            project=str(CFG.RUNS_ROOT),
            name=run_name,
            exist_ok=True,
            plots=True,
            save=True,
            save_period=5,
            verbose=True,
            copy_paste=CFG.COPY_PASTE if config == "copypaste" else 0.0,
        )

def train_yolo(config, seed):
    run_name = f"yolo_v60_{config}_s{seed}_fold{CFG.FOLD}"
    run_dir = CFG.RUNS_ROOT / run_name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    done_path = run_dir / "TRAINING_COMPLETE"
    if config not in CFG.CONFIGS or seed not in CFG.SEEDS:
        raise ValueError("Unknown pilot arm or seed")
    contract = {**PILOT_CONTRACT, "config": config, "seed": seed}
    contract_path = run_dir / "INPUT_CONTRACT.json"
    if contract_path.exists():
        require_input_contract(contract_path, contract)
    elif best_path.exists() or last_path.exists() or done_path.exists():
        raise ValueError("Checkpoint has no input contract; retain it and choose a fresh path")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_save(contract_path, contract)
    if done_path.exists():
        completion = json.loads(done_path.read_text(encoding="utf-8"))
        if (completion.get("input_contract") != contract or not best_path.is_file()
                or completion.get("checkpoint_sha256") != file_sha256(best_path)):
            raise ValueError("Completed checkpoint contract/bytes differ; retain artifacts")
        return best_path
    if file_sha256(INITIAL_YOLO_PATH) != PILOT_CONTRACT["initial_yolo_sha256"]:
        raise ValueError("Initial YOLO bytes changed after contract construction")

    yaml_path = build_fold_dataset()
    seed_everything(seed)
    if last_path.exists():
        print(f"{run_name}: resuming {last_path}")
        model = YOLO(str(last_path))
        model.train(resume=True)
    else:
        model = YOLO(str(INITIAL_YOLO_PATH))
        model.train(**training_kwargs(config, seed, yaml_path, run_name))
    if not best_path.exists():
        raise FileNotFoundError(f"Training ended without {best_path}")
    atomic_save(done_path, {"input_contract": contract, "checkpoint_sha256": file_sha256(best_path)})
    return best_path
"""
        ),
        code(
            "# [5/8] MediSegUNet (solar_filament/models/mediseg_models.py と同一。標準refinerの評価に使用)\n"
            + MODELS
        ),
        code(
            """# [6/8] 推論・後処理・採点関数の定義(build_v57と同一経路: raw + refined + 密度スライスFN)
from pycocotools import mask as mask_utils

if file_sha256(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt") != PILOT_CONTRACT["refiner_sha256"]:
    raise ValueError("Fixed refiner bytes changed")
refiner = MediSegUNet(3, 1, False).cuda().eval()
_ckpt = torch.load(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt",
                    map_location="cuda", weights_only=False)
refiner.load_state_dict(_ckpt.get("best_model") or _ckpt["model"], strict=True)
del _ckpt

def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)
    return output, confidences

def mask_square_box(mask, context=1.65, minimum=96):
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
    return refined & proposal  # shrink(確定構成)

def greedy_disjoint(masks, order, min_area=CFG.MIN_AREA):
    occupied = np.zeros((2048, 2048), dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept

def drop_contained_indices(masks, threshold=CFG.DROP_THRESHOLD):
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = mask_utils.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))
    areas = np.asarray(mask_utils.area(rles), dtype=np.float64)
    self_iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float64)
    dropped = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in dropped or j in dropped:
                continue
            value = self_iou[i, j]
            if value <= 0:
                continue
            intersection = value * (areas[i] + areas[j]) / (1.0 + value)
            smaller = i if areas[i] <= areas[j] else j
            if intersection / max(areas[smaller], 1.0) >= threshold:
                dropped.add(smaller)
    return [i for i in range(len(masks)) if i not in dropped]

def apply_ownership(masks, scores):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    subset = drop_contained_indices(masks)
    return greedy_disjoint(masks, sorted(subset, key=lambda i: areas[i]))

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

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

class PanopticQuality:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.tp_ious = []
        self.fp_count = 0
        self.fn_count = 0
    def add_matrix(self, matrix, n_gt, n_pred):
        if n_gt == 0:
            self.fp_count += n_pred; return
        if n_pred == 0:
            self.fn_count += n_gt; return
        hits = matrix > self.iou_threshold
        scores = matrix[hits].tolist()
        self.tp_ious.extend(scores)
        self.fp_count += int((hits.sum(axis=0) == 0).sum())
        self.fn_count += int((hits.sum(axis=1) == 0).sum())
    @property
    def denominator(self):
        return len(self.tp_ious) + 0.5 * self.fp_count + 0.5 * self.fn_count
    @property
    def score(self):
        d = self.denominator
        return float(sum(self.tp_ious) / d) if d > 0 else 0.0
    def sq_rq(self):
        tp = len(self.tp_ious)
        sq = float(sum(self.tp_ious) / tp) if tp > 0 else 0.0
        rq = float(tp / (tp + 0.5 * self.fp_count + 0.5 * self.fn_count)) if (tp + self.fp_count + self.fn_count) > 0 else 0.0
        return sq, rq
    def summary(self):
        sq, rq = self.sq_rq()
        return {"pq": self.score, "sq": sq, "rq": rq, "tp": len(self.tp_ious),
                "fp": self.fp_count, "fn": self.fn_count, "denominator": self.denominator}

gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in annotations_by_image[image_id] if not ann.get("iscrowd", 0)]

def evaluate_checkpoint(best_path, config, seed):
    \"\"\"raw(生YOLO)とrefined(確定構成)の両方でPQ採点 + 密度スライス別FNを記録。\"\"\"
    yolo_model = YOLO(str(best_path))
    meters = {"raw": PanopticQuality(), "refined": PanopticQuality()}
    fn_dense = fn_sparse = 0
    for name in tqdm(valid_files_fold, desc=f"{config}/s{seed} fold{CFG.FOLD} OOF採点"):
        image_path = CFG.PREPROC_DIR / name
        result = yolo_model.predict(str(image_path), imgsz=CFG.YOLO_IMGSZ,
                                     conf=CFG.PREDICT_CONF_FLOOR, iou=CFG.PREDICT_IOU,
                                     agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, confidences = result_masks(result)
        keep = [i for i, (m, s) in enumerate(zip(masks, confidences)) if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
        proposals = [masks[i] for i in keep]
        scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        refined = [refine_proposal(image, m) for m in proposals]
        predictions = {"raw": apply_ownership(proposals, scores),
                       "refined": apply_ownership(refined, scores)}
        pred_rles_by_mode = {}
        for mode, preds in predictions.items():
            pred_rles = []
            if preds:
                encoded = mask_utils.encode(np.asfortranarray(np.stack(preds, axis=-1).astype(np.uint8)))
                pred_rles = encoded if isinstance(encoded, list) else [encoded]
            pred_rles_by_mode[mode] = pred_rles
        file_fn = 0
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            for mode in ("raw", "refined"):
                matrix = iou_matrix(truth, pred_rles_by_mode[mode])
                meters[mode].add_matrix(matrix, len(truth), len(pred_rles_by_mode[mode]))
                if mode == "refined":
                    # FN(PQのIoU>0.5判定と同じ)をGT側で数え、annotator×imageで加算
                    if matrix.size:
                        file_fn += int((matrix.max(axis=1) <= 0.5).sum())
                    else:
                        file_fn += len(truth)
        if val_density[name] >= DENSE_CUT:
            fn_dense += file_fn
        else:
            fn_sparse += file_fn
    del yolo_model
    torch.cuda.empty_cache()
    return {"raw": meters["raw"].summary(), "refined": meters["refined"].summary(),
            "fn_dense_val": fn_dense, "fn_sparse_val": fn_sparse,
            "n_dense_val_files": int(sum(v >= DENSE_CUT for v in val_density.values()))}
"""
        ),
        code(
            """# [7/8] runを直列実行し、runごとにmanifestをDriveへ逐次追記(途中再開対応)
manifest = read_pilot_manifest()
rows = manifest["runs"]
done_keys = {(r["config"], r["seed"]) for r in rows}
if CFG.REUSE_VERIFIED_BASELINE and CFG.VERIFIED_BASELINE_MANIFEST.exists():
    for row in reuse_baselines(json.loads(CFG.VERIFIED_BASELINE_MANIFEST.read_text(encoding="utf-8"))):
        if (row["config"], row["seed"]) not in done_keys:
            rows.append(row)
            done_keys.add((row["config"], row["seed"]))
manifest.pop("failure", None)
manifest["status"] = "in_progress"
started = time.time()
scope = "pilot_preflight"
try:
    for config in CFG.CONFIGS:
        for seed in CFG.SEEDS:
            if (config, seed) in done_keys:
                continue
            run_dir = CFG.RUNS_ROOT / f"yolo_v60_{config}_s{seed}_fold{CFG.FOLD}"
            scope = "evaluation_only" if (run_dir / "TRAINING_COMPLETE").exists() else (
                "resumed_training_and_evaluation" if (run_dir / "weights/last.pt").exists() else "training_and_evaluation")
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            best_path = train_yolo(config, seed)
            checkpoint_sha = file_sha256(best_path)
            metrics = evaluate_checkpoint(best_path, config, seed)
            if file_sha256(best_path) != checkpoint_sha:
                raise ValueError("Evaluation checkpoint bytes changed during evaluation")
            row = {"config": config, "seed": seed, "fold": CFG.FOLD,
                   "checkpoint": str(best_path), "checkpoint_sha256": checkpoint_sha,
                   "input_contract": PILOT_CONTRACT,
                   "copy_paste": CFG.COPY_PASTE if config == "copypaste" else 0.0,
                   "execution": runtime_measurement(scope, started), **metrics}
            validate_run_row(row)
            rows.append(row)
            done_keys.add((config, seed))
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_save(CFG.MANIFEST, manifest)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False))
except (Exception, KeyboardInterrupt) as error:
    manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    manifest["failure"] = {"error_type": type(error).__name__, **runtime_measurement(scope, started)}
    atomic_save(CFG.MANIFEST, manifest)
    raise
manifest["status"] = "complete"
atomic_save(CFG.MANIFEST, manifest)

# --- 集計と採否の目安 ---
print("\\n=== 集計 ===")
for row in rows:
    delta_ref = row["refined"]["pq"] - CFG.BASELINE_PQ_REFINED
    print(f"{row['config']}/s{row['seed']}: raw PQ {row['raw']['pq']:.4f} | "
          f"refined PQ {row['refined']['pq']:.4f} (Δ{delta_ref:+.4f} vs fold{CFG.FOLD}基準) | "
          f"FN dense/sparse {row['fn_dense_val']}/{row['fn_sparse_val']}")

pair = {}
for row in rows:
    pair.setdefault(row["seed"], {})[row["config"]] = row
print("\\n=== seed別 paired Δ(copypaste - baseline) ===")
for seed, arms in sorted(pair.items()):
    if "baseline" in arms and "copypaste" in arms:
        d_raw = arms["copypaste"]["raw"]["pq"] - arms["baseline"]["raw"]["pq"]
        d_ref = arms["copypaste"]["refined"]["pq"] - arms["baseline"]["refined"]["pq"]
        d_fn = arms["copypaste"]["fn_dense_val"] - arms["baseline"]["fn_dense_val"]
        d_fp = arms["copypaste"]["refined"]["fp"] - arms["baseline"]["refined"]["fp"]
        print(f"seed {seed}: Δraw {d_raw:+.4f} | Δrefined {d_ref:+.4f} | "
              f"ΔFN(dense) {d_fn:+d} | ΔFP {d_fp:+d}")
print("\\n採否の目安: 2seedで符号一致かつdense FNが減ればfold1/2で独立確認。"
      "v57同様「FN改善があってもFP爆増でPQが下がる」場合は不採用(conf選択・独立確認は別途必要)。"
      "copypasteでもdense FNが減らなければ、検出器側の密集リコール系レバーは総括して不採用方向。")
"""
        ),
    ]
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["0", "1", "2"], help="評価対象fold")
    args = parser.parse_args()
    fold = int(args.fold)

    cells = build_cells(fold)
    for index, cell in enumerate(cells):
        cell["id"] = f"v60-copypaste-f{fold}-{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output = ROOT / f"solar_filament_v60_copypaste_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
