"""密集シーン oversampling パイロット notebook builder — WINNING_STRATEGY_2026-08-29.md §6.12優先度6。

## 目的

診断済みの最大ロス源は密集シーン(gt_count_mean 10-21)での検出漏れ
(hard FN 2138件、hard FP 1561件を37%上回る、2026-08-16診断)。
検出器側のレバーとしてまだ試していない「学習サンプリング重み付け」——
GTインスタンス数が上位の観測の学習サンプルを複製し、密集シーンの
見かけ上の出現頻度を上げる——を検証する。

refinerではなく**検出器(YOLO)側**のレバーである。hard FNはproposal自体が
存在しない GT を指し、refinerは既存proposalの形状修正しかできないため、
この失敗バケツに効くには検出の再学習が必要。

## 設計

学習レシピはV30と完全同一(yolo11m-seg、imgsz 1280、70epoch、batch 4、
AdamW lr0=7e-4、augmentation不変)。変数は学習データへの複製だけ:

  1. `baseline` — V30のデータセット構築と完全同一(複製なし)
  2. `dense2x`  — 各 train サンプル(file, annotator)の非crowd GT数が
                  全サンプルの75分位点(cut)以上なら、そのサンプルを
                  `__labeler_{id}_r2` サフィックスでもう1枚書き込む
                  (=dense サンプルは実質2倍の出現頻度)

各構成を seed (3407, 8407) で学習し、対象foldの OOF で評価する(計4本)。

評価は2段で行う:
  - **raw**: YOLO proposals → conf 0.26 / MIN_AREA 20 → ownership → PQ。
    refinerを通さないので、検出器の改善がそのまま見える
    (fold0の生YOLO基準: 0.3977、V44で計測)
  - **refined**: 上記に加えて既存の標準refiner(`mediseg_refiner_fold{N}_resume.pt`)
    を適用してから ownership → PQ(確定構成の評価経路)

加えてメカニズム検証として、検証画像を同じcutで dense/sparse に分け、
各runの FN(未マッチGT)数をスライス別に記録する。dense2xが効くなら
denseスライスのFNが減るはず。

## 使い方

    python scratch/build_v57_colab_dense_oversample_pilot.py --fold 0
    python scratch/build_v57_colab_dense_oversample_pilot.py --fold 1
    python scratch/build_v57_colab_dense_oversample_pilot.py --fold 2

出力: solar_filament_v57_dense_oversample_fold{N}_colab.ipynb (Colab A100推奨)
YOLO 4本の再学習(70epoch×imgsz1280)を直列に回すため、数時間〜半日規模。
各runのmanifestはDriveへ逐次追記され、中断しても完了済みrunを飛ばして再開する。
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (ROOT / "solar_filament" / "models" / "mediseg_models.py").read_text(encoding="utf-8")
OVERSAMPLE = (ROOT / "solar_filament" / "pipeline" / "dense_oversample.py").read_text(encoding="utf-8")

# 標準refinerパイプラインの全fold OOF基準(README「現在地」、v56と同一)
BASELINE_PQ_REFINED = {0: 0.4393, 1: 0.4415, 2: 0.4471}
# 生YOLO基準(V44計測、fold0のみ既知。fold1/2は未計測のためpaired baseline runが参照)
BASELINE_PQ_RAW = {0: 0.3977}
DENSE_QUANTILE = 0.75
DENSE_EXTRA = 1
SEEDS = (3407, 8407)
CONFIGS = ("baseline", "dense2x")
PROTOCOL_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    + MODELS.replace("\r\n", "\n").encode("utf-8")
    + OVERSAMPLE.replace("\r\n", "\n").encode("utf-8")
).hexdigest()


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build_cells(fold):
    baseline_refined = BASELINE_PQ_REFINED[fold]
    baseline_raw = BASELINE_PQ_RAW.get(fold)

    cells = [
        markdown(
            """# 密集シーン oversampling パイロット — fold{fold}、検出器側レバー

WINNING_STRATEGY_2026-08-29.md §6.12優先度6。診断済みの最大ロス源
(密集シーンでの検出漏れ、hard FN 2138件)に対し、GTインスタンス数が
上位{q:.0%}の観測の学習サンプルを複製して出現頻度を2倍にする。

| 構成 | 内容 |
|---|---|
| `baseline` | V30のYOLO学習データ構築と完全同一 |
| `dense2x` | 各(file,annotator)サンプルのGT数 >= cut で1枚複製して2倍重み付け |

学習レシピはV30と完全同一(yolo11m-seg/imgsz1280/70epoch/AdamW lr0=7e-4)。
fold{fold}で2構成 × seed 2本 = 4本を学習・OOF評価する。

fold{fold}基準: refined PQ {baseline_refined}(標準refiner) / raw YOLO PQ {baseline_raw}。
元 PR #239 の fold0 は不採用として記録済み。今回の補修版は CPU 契約のみ検証し、
GPU 受入は未実施。fold1/2 の追加実験と提出構成の変更は行っていない。
旧入力契約のない dataset/checkpoint/manifest は再開せず、既存の実験記録を保持する。

GPU必須、A100推奨(4本直列で数時間〜半日見込み。各runのmanifestはDriveへ
逐次追記され、中断後は完了済みrunを飛ばして再開する)。
""".format(
                fold=fold,
                q=DENSE_QUANTILE,
                baseline_refined=baseline_refined,
                baseline_raw=(baseline_raw if baseline_raw is not None else "未計測(paired baselineが参照)"),
            )
        ),
        code(
            """# [1/9] Drive接続、設定、前提の確認
!pip -q install ultralytics pycocotools
import hashlib, json, math, random, shutil, sys, time, warnings
from importlib.metadata import version
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from google.colab import drive
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
drive.mount("/content/drive")

class CFG:
    DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    LOCAL_ROOT = Path("/content/solar_v57_dense")
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    YOLO_ROOT = LOCAL_ROOT / "yolo_datasets"
    RUNS_ROOT = DRIVE_ROOT / "runs" / "v57_dense_oversample"
    FOLD = %d
    PROTOCOL_SHA256 = %r
    # --- build_v30_prize_colab.py と完全同一。触らないこと ---
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    # --- OOF評価は確定構成(build_v56系と同一。refinerは既存ckptを使い回す) ---
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
    DENSE_EXTRA = %d
    SEEDS = %r
    CONFIGS = %r
    MANIFEST = DRIVE_ROOT / "artifacts" / "v57_dense_oversample_fold%d_manifest.json"

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
"""  # noqa: UP031 - preserve braces in embedded Python source
            % (fold, PROTOCOL_SHA256, baseline_refined, DENSE_QUANTILE, DENSE_EXTRA, SEEDS, CONFIGS, fold)
        ),
        code(
            "# [2/9] dense_oversample (solar_filament/pipeline/dense_oversample.py と同一、テスト済み)\n"
            + OVERSAMPLE
        ),
        code(
            """# [3/9] COCOとfoldを読み、密度分布を確認し、前処理画像をローカルSSDへ展開
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
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
train_files_fold = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
valid_files_fold = [n for n in train_files if fold_of_file[n] == CFG.FOLD]
assert not (set(train_files_fold) & set(valid_files_fold)), "fold分割が壊れている"

def noncrowd_count(image_id):
    return sum(1 for a in annotations_by_image[image_id] if not a.get("iscrowd", 0))

# 学習サンプルは (file, annotator) 単位 — V30と同じく全annotatorを使う。
# 密度はサンプル自身のGT数で測る。
train_samples = [
    (name, image_id)
    for name in train_files_fold
    for image_id in file_to_ids[name]
]
if not train_samples or not valid_files_fold:
    raise ValueError("Both training and validation observations are required")
sample_counts = [noncrowd_count(image_id) for _, image_id in train_samples]
DENSE_CUT = dense_cut_value(sample_counts, CFG.DENSE_QUANTILE)

# 検証画像の密度(run_case_diagnostics.pyのgt_count_meanと同じ定義)。
# dense/sparseスライスのFN集計に使う。
val_density = {
    name: float(np.mean([noncrowd_count(i) for i in file_to_ids[name]]))
    for name in valid_files_fold
}
n_dense_val = sum(v >= DENSE_CUT for v in val_density.values())

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# Resolve the actual checkpoint (including Ultralytics' cached/downloaded path),
# rather than treating a same-named local file as the same initialization.
initial_yolo = YOLO(CFG.YOLO_MODEL)
INITIAL_YOLO_PATH = Path(initial_yolo.ckpt_path).resolve(strict=True)
del initial_yolo

preprocessed_hashes = {}
for name in tqdm(train_files_fold + valid_files_fold, desc="前処理画像をSSDへ・SHA256"):
    local = CFG.PREPROC_DIR / name
    # Refresh from the authoritative cache rather than silently reuse old pixels.
    shutil.copy2(CFG.PREPROC_CACHE / name, local)
    preprocessed_hashes[name] = file_sha256(local)

PILOT_CONTRACT = {
    "schema_version": 1, "fold": CFG.FOLD, "protocol_sha256": CFG.PROTOCOL_SHA256,
    "annotations_sha256": file_sha256(TRAIN_JSON),
    "folds_sha256": file_sha256(CFG.ARTIFACT_ROOT / "grouped_folds.csv"),
    "refiner_sha256": file_sha256(CFG.ARTIFACT_ROOT / f"mediseg_refiner_fold{CFG.FOLD}_resume.pt"),
    "initial_yolo_sha256": file_sha256(INITIAL_YOLO_PATH),
    "preprocessed_sha256": hashlib.sha256(
        json.dumps(preprocessed_hashes, sort_keys=True).encode("utf-8")).hexdigest(),
    "dense_quantile": CFG.DENSE_QUANTILE, "dense_extra": CFG.DENSE_EXTRA,
    "dense_cut": DENSE_CUT,
    "recipe": {name: getattr(CFG, name) for name in (
        "YOLO_MODEL", "YOLO_IMGSZ", "YOLO_BATCH", "YOLO_EPOCHS", "YOLO_PATIENCE",
        "YOLO_MASK_RATIO", "CROP_SIZE", "CROP_CONTEXT", "CONF", "PREDICT_CONF_FLOOR",
        "PREDICT_IOU", "REFINER_THRESHOLD", "DROP_THRESHOLD", "MIN_AREA")},
    "seeds": list(CFG.SEEDS), "configs": list(CFG.CONFIGS),
    "runtime_versions": {"python": sys.version, "torch": torch.__version__,
                         "ultralytics": version("ultralytics"), "numpy": np.__version__,
                         "opencv": cv2.__version__, "pycocotools": version("pycocotools")},
}

print({
    "観測": len(train_files), "除外train": len(train_files_fold),
    "検証(OOF)": len(valid_files_fold),
    "trainサンプル数": len(train_samples),
    "gt_count分布": {
        "min": int(np.min(sample_counts)), "median": float(np.median(sample_counts)),
        "q75": float(np.quantile(sample_counts, 0.75)),
        "max": int(np.max(sample_counts)),
    },
    "dense_cut": DENSE_CUT,
    "dense対象サンプル数": int(sum(c >= DENSE_CUT for c in sample_counts)),
    "dense検証画像数": int(n_dense_val),
})
"""
        ),
        code(
            """# [4/9] YOLOデータセット構築(V30と同一 + dense構成のみ複製書き込み)
def require_input_contract(path, expected):
    if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"{path}: input contract differs or is absent; use a fresh pilot directory")


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

def build_fold_dataset(config):
    # configごとに独立したrootにする(ultralyticsのlabels.cache混線を防ぐ)
    root = CFG.YOLO_ROOT / config / f"fold_{CFG.FOLD}"
    extra = CFG.DENSE_EXTRA if config == "dense2x" else 0
    cut, reps = dense_replication_counts(sample_counts, CFG.DENSE_QUANTILE, extra)
    assert cut == DENSE_CUT
    stats = {
        "n_train_samples": int(sum(reps)),
        "n_base_samples": len(train_samples),
        "n_dense_samples": int(sum(1 for r in reps if r > 1)),
        "dense_cut": cut,
    }
    contract = {**PILOT_CONTRACT, "config": config}
    contract_path = root / "INPUT_CONTRACT.json"
    if (root / "data.yaml").exists():
        require_input_contract(contract_path, contract)
        print(f"{config}: 既存データセットを再利用 {root}")
        return root / "data.yaml", {**stats, "reused": True}
    if root.exists():
        shutil.rmtree(root)
    for (file_name, image_id), rep in tqdm(
            list(zip(train_samples, reps)), desc=f"build {config} fold{CFG.FOLD} train"):
        write_variant(root, "train", file_name, image_id, f"__labeler_{image_id}")
        for r in range(2, rep + 1):
            write_variant(root, "train", file_name, image_id, f"__labeler_{image_id}_r{r}")
    for file_name in tqdm(valid_files_fold, desc=f"build {config} fold{CFG.FOLD} val"):
        image_id = file_to_reference_id[file_name]
        write_variant(root, "val", file_name, image_id, "")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
        encoding="utf-8",
    )
    contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    print(f"{config}: {stats}")
    return yaml_path, stats
"""
        ),
        code(
            """# [5/9] 学習(V30と同一ハイパラ、seedのみ変数) — runごとに再開対応
from ultralytics import YOLO

def train_yolo(config, seed):
    run_name = f"yolo_v57_{config}_s{seed}_fold{CFG.FOLD}"
    run_dir = CFG.RUNS_ROOT / run_name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    done_path = run_dir / "TRAINING_COMPLETE"
    contract_path = run_dir / "INPUT_CONTRACT.json"
    contract = {**PILOT_CONTRACT, "config": config, "seed": seed}
    if contract_path.exists():
        require_input_contract(contract_path, contract)
    elif best_path.exists() or last_path.exists() or done_path.exists():
        raise ValueError(f"{run_dir}: checkpoint has no input contract; use a fresh pilot directory")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    if done_path.exists() and best_path.exists():
        print(f"{run_name}: using completed checkpoint {best_path}")
        return best_path

    if (not INITIAL_YOLO_PATH.is_file()
            or file_sha256(INITIAL_YOLO_PATH) != PILOT_CONTRACT["initial_yolo_sha256"]):
        raise ValueError("initial YOLO checkpoint changed; use a fresh pilot directory")
    yaml_path, _ = build_fold_dataset(config)
    seed_everything(seed)
    if last_path.exists():
        print(f"{run_name}: resuming {last_path}")
        model = YOLO(str(last_path))
        model.train(resume=True)
    else:
        model = YOLO(str(INITIAL_YOLO_PATH))
        model.train(
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
        )
    if not best_path.exists():
        raise FileNotFoundError(f"Training ended without {best_path}")
    done_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return best_path
"""
        ),
        code(
            "# [6/9] MediSegUNet (solar_filament/models/mediseg_models.py と同一。標準refinerの評価に使用)\n"
            + MODELS
        ),
        code(
            """# [7/9] 推論・後処理・採点関数の定義(build_v56系と同一経路 + raw評価 + 密度スライスFN)
from pycocotools import mask as mask_utils

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
            """# [8/9] 4本を直列実行し、runごとにmanifestをDriveへ逐次追記(途中再開対応)
if CFG.MANIFEST.exists():
    manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("input_contract") != PILOT_CONTRACT:
        raise ValueError("manifest input contract differs or is absent; use a fresh pilot directory")
    for row in manifest["runs"]:
        checkpoint = Path(row["checkpoint"])
        if (row["fold"] != CFG.FOLD or row["config"] not in CFG.CONFIGS
                or row["seed"] not in CFG.SEEDS or not checkpoint.is_file()
                or file_sha256(checkpoint) != row.get("checkpoint_sha256")):
            raise ValueError("manifest checkpoint/arm does not match its input contract")
else:
    manifest = {"experiment_id": f"v57_dense_oversample_fold{CFG.FOLD}",
                "input_contract": PILOT_CONTRACT, "runs": []}
done_keys = {(r["config"], r["seed"]) for r in manifest["runs"]}

for config in CFG.CONFIGS:
    yaml_path, train_stats = build_fold_dataset(config)
    for seed in CFG.SEEDS:
        if (config, seed) in done_keys:
            print(f"{config}/s{seed}: manifest済み、skip")
            continue
        started = time.time()
        run_dir = CFG.RUNS_ROOT / f"yolo_v57_{config}_s{seed}_fold{CFG.FOLD}"
        if (run_dir / "TRAINING_COMPLETE").exists() and (run_dir / "weights/best.pt").exists():
            measurement_scope = "evaluation_only"
        elif (run_dir / "weights/last.pt").exists():
            measurement_scope = "resumed_training_and_evaluation"
        else:
            measurement_scope = "training_and_evaluation"
        torch.cuda.reset_peak_memory_stats()
        best_path = train_yolo(config, seed)
        metrics = evaluate_checkpoint(best_path, config, seed)
        minutes = round((time.time() - started) / 60, 1)
        row = {
            "config": config, "seed": seed, "fold": CFG.FOLD,
            "checkpoint": str(best_path), "checkpoint_sha256": file_sha256(best_path),
            "minutes_train_eval": minutes,
            "execution": {"service": "Google Colab", "gpu_name": torch.cuda.get_device_name(0),
                          "measurement_scope": measurement_scope,
                          "prior_training_telemetry": (
                              None if measurement_scope == "training_and_evaluation"
                              else "unavailable: not recorded in this invocation"),
                          "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                          "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                          "compute_units_used": None,
                          "compute_units_note": "Record in Colab UI; runtime API does not expose consumption."},
            **train_stats, **metrics,
        }
        manifest["runs"].append(row)
        manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        temporary = CFG.MANIFEST.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(CFG.MANIFEST)
        print(json.dumps(row, ensure_ascii=False, indent=2))

print(f"\\n保存: {CFG.MANIFEST}")

# --- 集計と採否の目安 ---
rows = manifest["runs"]
print("\\n=== 集計 ===")
for row in rows:
    delta_ref = row["refined"]["pq"] - CFG.BASELINE_PQ_REFINED
    print(f"{row['config']}/s{row['seed']}: raw PQ {row['raw']['pq']:.4f} | "
          f"refined PQ {row['refined']['pq']:.4f} (Δ{delta_ref:+.4f} vs fold{CFG.FOLD}基準) | "
          f"FN dense/sparse {row['fn_dense_val']}/{row['fn_sparse_val']}")

pair = {}
for row in rows:
    pair.setdefault(row["seed"], {})[row["config"]] = row
print("\\n=== seed別 paired Δ(dense2x - baseline) ===")
for seed, arms in sorted(pair.items()):
    if "baseline" in arms and "dense2x" in arms:
        d_raw = arms["dense2x"]["raw"]["pq"] - arms["baseline"]["raw"]["pq"]
        d_ref = arms["dense2x"]["refined"]["pq"] - arms["baseline"]["refined"]["pq"]
        d_fn = arms["dense2x"]["fn_dense_val"] - arms["baseline"]["fn_dense_val"]
        print(f"seed {seed}: Δraw {d_raw:+.4f} | Δrefined {d_ref:+.4f} | ΔFN(dense) {d_fn:+d}")
print("\\n採否の目安: プロジェクト規律(§2.5)に従い、2seedで符号一致するか確認。有望なら"
      "fold1・fold2で同じパイロットを独立確認してから採用判断。"
      "dense2xでFN(dense)が減らずPQも変わらない場合は不採用。")
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
        cell["id"] = f"v57-dense-f{fold}-{index:02d}"

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

    output = ROOT / f"solar_filament_v57_dense_oversample_fold{fold}_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
