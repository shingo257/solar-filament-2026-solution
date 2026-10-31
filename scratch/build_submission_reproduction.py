"""提出用再現notebookを master pipeline notebook から派生生成する。

審査委員が Kaggle または Colab で「すべて実行」するだけで
train → inference → submission.csv まで通る構成にする:

- 環境検出: /kaggle/input があれば Kaggle モード、なければ Colab+Drive
- 重み解決: 添付 Kaggle dataset → Drive artifacts → フル再学習 の優先順
- 前処理キャッシュが無い環境では solar_three_channel を inline で計算
- 末尾に pip freeze + 重み SHA256 を記録する環境固定セルを追加
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
MASTER = MODULE / "solar_filament_master_pipeline_colab.ipynb"
OUTPUT = MODULE / "solar_filament_submission_reproduction.ipynb"
KAGGLE_OUTPUT = MODULE / "output" / "kaggle_kernel" / OUTPUT.name


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


HEADER_MD = """\
# Solar Filament Segmentation Challenge 2026 — Submission Reproduction

**これは提出物の公式再現 notebook です。** 「すべて実行」で前処理 → 検出器学習 →
refiner 学習 → test 推論 → `submission.csv` 生成まで一通り通ります。
凍結 fold 分割(`grouped_folds.csv`)の SHA256 は `CFG.GROUPED_FOLDS_SHA256` に
同梱済みで、CSV 本体は重み dataset(Kaggle)または Drive artifacts(Colab)から
自動解決されます。欠落・不整合なら前処理前に停止し、新しい分割は生成しません。

## 実行環境

| 環境 | 手順 |
|---|---|
| Kaggle | 本 notebook に競技データ(`filament-segmentation-2026`)と重み dataset をアタッチして Run。GPU T4x2 以上推奨 |
| Colab | Google Drive に `solar_filament_v30` ルート(競技データ + 成果物)を配置して Run。A100/L4 推奨 |

## 重みの解決順序

1. **repo 同梱 `weights/`**(本 notebook と同じリポジトリに Git LFS で格納。
   `weights_manifest.json` の SHA256 で検証。審査要件の第一経路)
2. Kaggle にアタッチされた重み dataset(`*yolo*fold{N}*best*.pt`, `*refiner*all*data*.pt`)
3. Drive artifacts(`solar_filament_v30/runs`, `artifacts/`)内の既存 checkpoint
4. どれも無ければ **フル再学習**(YOLO 3-fold + refiner 3-fold + all-data fine-tune、A100 で約6-8時間)

`SOLAR_WEIGHTS_DIR` 環境変数でも repo weights の場所を明示できます。

## 確定構成(V50/V52 提出と同一)

- YOLO11m-seg @1280, conf=0.26, iou=0.55, retina_masks
- EdgeAttNet風 refiner(crop 320px, context 1.65, threshold 0.45)
- 包含率0.90でのdrop → 面積昇順 ownership(排他化)→ min_area 20
- fold2 単一モデル(OOF PQ最良、アンサンブルより計算1/3で同等以上)

## 選択した2つの提出の再現

最終選択は `prod`(上記構成)と `plain_mhsa_f2`(refiner を ablation の
plain-MHSA fold2 重みに差し替えた構成、アーキテクチャ差による private ヘッジ)。
`prod` は本 notebook をそのまま実行すれば再現できる。
`plain_mhsa_f2` は setup セルの `CFG.SUBMISSION_VARIANT` を `"plain_mhsa_f2"`
に変更して再実行する(重み dataset の `ablation_plain_mhsa_seed8407_fold2.pt`
を使用、出力は `submission_plain_mhsa_f2.csv`)。

## 探索用 conf バリアント (sparse-label 仮説の A/B、§2.54)

`CFG.SUBMISSION_CONF` は `SOLAR_SUBMISSION_CONF` 環境変数で上書きできる
(既定 0.26)。0.26 以外では出力名に `_confNNN` が付き、公式提出 CSV を
上書きしない(例: `SOLAR_SUBMISSION_CONF=0.32` → `submission_conf032.csv` /
`submission_master_pipeline_conf032.csv`)。これは探索用候補であり、
最終選択 (0.26) を置き換えるものではない。
"""

SETUP_CELL = """\
# [1/6] 環境検出・依存関係・設定
%pip install -q "ultralytics>=8.3,<9" pycocotools

import json, math, os, random, shutil, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from ultralytics import YOLO

IS_KAGGLE = Path("/kaggle/input").exists()
if IS_KAGGLE:
    WORK_ROOT = Path("/kaggle/working")
    DATA_SEARCH_ROOTS = [Path("/kaggle/input")]
else:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK_ROOT = Path("/content")

class CFG:
    SEED = 3407
    N_FOLDS = 3
    # 凍結済み grouped_folds.csv(707 observations, file_name/fold)の SHA256。
    # CSV 本体は重み dataset または Drive artifacts から自動解決される。
    GROUPED_FOLDS_SHA256 = "24367120279820997396f7063b1ae5b235a442fd1c92617e1e8941d472504c3b"
    if IS_KAGGLE:
        DRIVE_ROOT = WORK_ROOT / "solar_filament_v30"  # 成果物の書き込み先(working)
        PREPROC_CACHE = None  # Kaggle には前処理キャッシュ無し → inline計算
    else:
        DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
        PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"
    DATA_CACHE = DRIVE_ROOT / "competition_data"
    ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
    RUNS_ROOT = DRIVE_ROOT / "runs"
    YOLO_ROOT = DRIVE_ROOT / "yolo_datasets"
    LOCAL_ROOT = WORK_ROOT / "solar_master"
    PREPROC_DIR = LOCAL_ROOT / "preprocessed"
    # --- YOLO検出器(build_v30_prize_colab.pyと同一) ---
    YOLO_MODEL = "yolo11m-seg.pt"
    YOLO_IMGSZ = 1280
    YOLO_BATCH = 4
    YOLO_EPOCHS = 70
    YOLO_PATIENCE = 14
    YOLO_MASK_RATIO = 2
    # --- refiner(build_v35_refiner_folds12.py / build_v51_edgeattnet_refiner.pyと同一) ---
    CROP_SIZE = 320
    CROP_CONTEXT = 1.65
    REFINER_EPOCHS = 14
    REFINER_ALLDATA_EPOCHS = 5
    REFINER_LR = 2e-4
    REFINER_BATCH_CANDIDATES = (8, 6, 4, 2)
    # --- 確定後処理(README.md「現在地」節、V52提出構成) ---
    # "prod"(既定: EdgeAtt all-data refiner) / "plain_mhsa_f2"(第2提出:
    # ablation plain-MHSA fold2 refiner、重み dataset の
    # ablation_plain_mhsa_seed8407_fold2.pt が必要)
    SUBMISSION_VARIANT = "prod"
    SUBMISSION_CONF = float(os.environ.get("SOLAR_SUBMISSION_CONF", "0.26"))
    REFINER_THRESHOLD = 0.45
    DROP_THRESHOLD = 0.90
    MIN_AREA = 20
    PREDICT_IOU = 0.55
    SINGLE_FOLD = 2  # V50/V52で確認済み

if CFG.SUBMISSION_VARIANT not in ("prod", "plain_mhsa_f2"):
    raise ValueError(f"Unknown submission variant: {CFG.SUBMISSION_VARIANT}")

for path in (CFG.LOCAL_ROOT, CFG.PREPROC_DIR, CFG.ARTIFACT_ROOT, CFG.YOLO_ROOT, CFG.RUNS_ROOT):
    path.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)


def find_dataset_root():
    # Kaggle: /kaggle/input 以下、Colab: Drive competition_data 以下を探す
    for base in DATA_SEARCH_ROOTS if IS_KAGGLE else [CFG.DATA_CACHE]:
        matches = list(base.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
        if matches:
            return matches[0].parents[2]
    raise FileNotFoundError("競技データ(MAGFiLO annotations JSON)が見つかりません。"
                            "Kaggleでは競技データをアタッチ、ColabではDriveに配置してください")


def repo_weights_roots():
    \"\"\"repo 同梱 weights/ の候補 (審査要件: weights stored in the repository)。

    notebook が clone 済み repo 内で実行される標準ケースを cwd 起点で探す。
    SOLAR_WEIGHTS_DIR 環境変数でも明示指定できる。
    \"\"\"
    cands = []
    env = os.environ.get("SOLAR_WEIGHTS_DIR")
    if env:
        cands.append(Path(env))
    cwd = Path.cwd()
    cands += [cwd / "weights", cwd.parent / "weights",
              cwd / "Solar_Filament_Segmentation" / "weights"]
    return [c for c in cands if c.is_dir()]


REPO_WEIGHTS = repo_weights_roots()


def verify_repo_weight(path):
    \"\"\"repo weights/ 由来のファイルを同梱 manifest の SHA256 で検証する。\"\"\"
    manifest = path.parent / "weights_manifest.json"
    if not manifest.exists():
        return
    entries = {w["file_name"]: w["sha256"] for w in
               json.loads(manifest.read_text(encoding="utf-8"))["weights"]}
    expected = entries.get(path.name)
    if expected is None:
        return
    import hashlib
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"repo weight sha256 mismatch: {path.name} "
            f"(manifest={expected[:16]}..., actual={actual[:16]}...)")


def resolve_pretrained(patterns):
    \"\"\"事前学習済み重みを探す。解決順序:

    1. repo 同梱 weights/ (manifest sha256 検証付き — 審査要件の第一経路)
    2. 添付 Kaggle dataset / Drive artifacts
    3. 見つからなければ None (呼び出し側がフル再学習へフォールバック)
    \"\"\"
    for base in REPO_WEIGHTS:
        for pattern in patterns:
            hits = sorted(base.glob(pattern))
            hits = [h for h in hits if h.is_file() and h.stat().st_size > 1_000_000]
            if hits:
                verify_repo_weight(hits[0])
                return hits[0]
    for base in DATA_SEARCH_ROOTS if IS_KAGGLE else [CFG.RUNS_ROOT, CFG.ARTIFACT_ROOT]:
        for pattern in patterns:
            hits = sorted(base.rglob(pattern))
            hits = [h for h in hits if h.is_file() and h.stat().st_size > 1_000_000]
            if hits:
                return hits[0]
    return None


assert torch.cuda.is_available(), "GPUランタイムを選択してください"
print({"env": "kaggle" if IS_KAGGLE else "colab",
       "GPU": torch.cuda.get_device_name(0)})
"""

PREPROC_CELL = """\
# [2/6] データ読み込み・前処理・fold分割
def find_train_dir(data_root):
    matches = list(data_root.rglob("train_images"))
    if matches:
        return matches[0]
    # レイアウト差への耐性: annotation JSON と同階層から jpeg を探す
    return next(data_root.rglob("*train*"))


def robust_uint8(array, mask=None, low=0.5, high=99.5):
    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)


def solar_three_channel(gray):
    # EdgeAttNet着想: 径方向/背景勾配を抑えて暗いフィラメントを浮かせつつ元観測も残す
    gray = gray.astype(np.float32)
    disk = gray > max(4.0, np.percentile(gray, 8))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=80, sigmaY=80)
    flattened = gray / np.maximum(background, 1.0)
    flat_u8 = robust_uint8(flattened, disk, 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(flat_u8)
    raw_u8 = robust_uint8(gray, disk, 0.5, 99.5)
    dark_detail = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    dark_detail = robust_uint8(dark_detail, disk, 0.0, 99.5)
    out = np.dstack([raw_u8, clahe, dark_detail])
    out[~disk] = 0
    return out


def prepare_preprocessed(file_name, raw_dir):
    \"\"\"Drive キャッシュがあればコピー、無ければ raw JPEG から inline で前処理を計算。\"\"\"
    local = CFG.PREPROC_DIR / file_name
    if local.exists():
        return
    if CFG.PREPROC_CACHE is not None and (CFG.PREPROC_CACHE / file_name).exists():
        shutil.copy2(CFG.PREPROC_CACHE / file_name, local)
        return
    gray = cv2.imread(str(raw_dir / file_name), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(raw_dir / file_name)
    cv2.imwrite(str(local), solar_three_channel(gray), [cv2.IMWRITE_JPEG_QUALITY, 96])


"""

YOLO_PRETRAINED_HOOK = """
def train_yolo_fold(fold):
    run_dir = CFG.RUNS_ROOT / f"yolo_fold_{fold}"
    best_path = run_dir / "weights" / "best.pt"
    done_path = run_dir / "TRAINING_COMPLETE"
    if done_path.exists() and best_path.exists():
        print(f"fold {fold}: 学習済みcheckpointを再利用 {best_path}")
        return best_path
    pretrained = resolve_pretrained([f"*yolo*fold{fold}*best*.pt", f"*fold{fold}*best*.pt",
                                     f"*yolo_fold_{fold}*best*.pt"])
    if pretrained is not None:
        best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pretrained, best_path)
        done_path.write_text("ok", encoding="utf-8")
        print(f"fold {fold}: 添付済み重みを使用 {pretrained}")
        return best_path
    yaml_path = build_fold_dataset(fold)"""

REFINER_PRETRAINED_BLOCK = """\
# all-data fine-tune — 添付重み → Drive ckpt → 再学習 の順で解決
pretrained_refiner = resolve_pretrained(["*refiner*all*data*.pt", "*edgeattnet*all*.pt"])
if pretrained_refiner is not None and not ALLDATA_CKPT.exists():
    ALLDATA_CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pretrained_refiner, ALLDATA_CKPT)
    print(f"添付済みrefiner重みを使用 {pretrained_refiner}")
if ALLDATA_CKPT.exists():
    print(f"学習済みcheckpointを再利用 {ALLDATA_CKPT}")
else:
    all_ds = InstanceCropDataset(train_files, augment=True)"""

FOLD_CSV_STAGING = """\
grouped_folds_path = CFG.ARTIFACT_ROOT / "grouped_folds.csv"
if not grouped_folds_path.exists():
    # repo data/ → 添付 dataset(Kaggle) / Drive artifacts(Colab) の順に
    # 凍結 fold CSV を解決。SHA256 が一致するコピーのみ採用し、
    # 不一致候補は永続領域に書き込まない。
    import hashlib as _hashlib
    cwd = Path.cwd()
    repo_data = [cwd / "data", cwd.parent / "data",
                 cwd / "Solar_Filament_Segmentation" / "data"]
    fold_bases = ([d for d in repo_data if d.is_dir()] +
                  (DATA_SEARCH_ROOTS if IS_KAGGLE
                   else [CFG.ARTIFACT_ROOT, CFG.DATA_CACHE]))
    fold_hits = sorted(h for base in fold_bases
                       for h in base.rglob("grouped_folds.csv") if h.is_file())
    chosen = next(
        (h for h in fold_hits
         if _hashlib.sha256(h.read_bytes()).hexdigest() == CFG.GROUPED_FOLDS_SHA256),
        None)
    if chosen is None:
        raise FileNotFoundError(
            "凍結 fold CSV(grouped_folds.csv)が見つかりません。"
            "repo の data/、Kaggle の重み dataset 同梱分、または "
            "Colab の Drive artifacts に配置してください")
    shutil.copy2(chosen, grouped_folds_path)
    print({"fold_csv_resolved": str(chosen)})

"""

ENV_RECORD_CELL = """\
# [6/6] 環境固定記録 — pip freeze + 重みSHA256
import hashlib, subprocess, sys

freeze_path = CFG.ARTIFACT_ROOT / "submission_reproduction_pip_freeze.txt"
freeze_path.write_text(
    subprocess.run([sys.executable, "-m", "pip", "freeze"],
                    capture_output=True, text=True).stdout,
    encoding="utf-8")
print(f"pip freeze -> {freeze_path}")

refiner_label, refiner_path = (
    ("refiner_all_data", ALLDATA_CKPT) if CFG.SUBMISSION_VARIANT == "prod"
    else ("refiner_" + CFG.SUBMISSION_VARIANT, plain_ckpt))
for label, path in [
    ("yolo_fold2", yolo_weight_paths.get(CFG.SINGLE_FOLD)),
    (refiner_label, refiner_path),
]:
    if path and Path(path).exists():
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        print(f"SHA256 {label}: {digest}")
        (CFG.ARTIFACT_ROOT / f"sha256_{label}.txt").write_text(digest + "\\n", encoding="utf-8")
"""


def main() -> None:
    nb = json.loads(MASTER.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # 1. タイトル markdown を差し替え
    cells[0] = md(HEADER_MD)

    # 2. setup セル(cell 2)を環境検出版に差し替え
    cells[2] = code(SETUP_CELL)

    # 3. main の凍結fold検証・COCO契約を保ち、前処理だけfallback化
    src3 = "".join(cells[3]["source"])
    loader_start = src3.index("def load_frozen_folds(")
    loader_end = src3.index("def find_dataset_root(base):")
    data_start = src3.index("DATA_ROOT = find_dataset_root(CFG.DATA_CACHE)")
    frozen_loader = src3[loader_start:loader_end]
    data_loading = src3[data_start:].replace(
        "find_dataset_root(CFG.DATA_CACHE)", "find_dataset_root()", 1)
    old_preproc = '''for name in tqdm(train_files, desc="前処理画像をローカルへ"):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)'''
    assert old_preproc in data_loading, "cell3 preprocessing anchor mismatch"
    data_loading = data_loading.replace(old_preproc, '''TRAIN_DIR = find_train_dir(DATA_ROOT)
for name in tqdm(train_files, desc="前処理画像をローカルへ"):
    prepare_preprocessed(name, TRAIN_DIR)''')
    # 凍結 fold CSV を ARTIFACT_ROOT へ自動ステージ(添付 dataset / Drive から解決)。
    # loader は固定パスの SHA256 を厳格検証するため、解決はコピーのみで検証は変えない。
    old_folds = 'grouped_folds_path = CFG.ARTIFACT_ROOT / "grouped_folds.csv"\n'
    assert old_folds in data_loading, "cell3 fold path anchor mismatch"
    data_loading = data_loading.replace(old_folds, FOLD_CSV_STAGING, 1)
    cells[3] = code(PREPROC_CELL + frozen_loader + data_loading)

    # 4. YOLO学習セル(cell 5): train_yolo_fold 冒頭に添付重み解決を挿入
    src5 = "".join(cells[5]["source"])
    assert "def train_yolo_fold(fold):" in src5
    old_head = """def train_yolo_fold(fold):
    run_dir = CFG.RUNS_ROOT / f"yolo_fold_{fold}"
    best_path = run_dir / "weights" / "best.pt"
    done_path = run_dir / "TRAINING_COMPLETE"
    if done_path.exists() and best_path.exists():
        print(f"fold {fold}: 学習済みcheckpointを再利用 {best_path}")
        return best_path
    yaml_path = build_fold_dataset(fold)"""
    assert old_head in src5, "cell5 train_yolo_fold head anchor mismatch"
    src5 = src5.replace(old_head, YOLO_PRETRAINED_HOOK.strip())
    cells[5] = code(src5)

    # 5a. refiner fold 学習セル(cell 9): 添付重みがあれば fold 学習をスキップ
    src9 = "".join(cells[9]["source"])
    anchor9 = """    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if state.get("epoch", -1) + 1 >= epochs:
            print(f"fold {fold}: 学習済みcheckpointを再利用 {checkpoint_path}")
            return checkpoint_path
"""
    assert anchor9 in src9, "cell9 anchor mismatch"
    src9 = src9.replace(anchor9, anchor9 + """
    pretrained = resolve_pretrained([f"*refiner*fold{fold}*resume*.pt",
                                     f"*refiner*fold{fold}*.pt"])
    if pretrained is not None and pretrained.resolve() != checkpoint_path.resolve():
        state = torch.load(pretrained, map_location="cpu")
        if state.get("epoch", -1) + 1 >= epochs:
            shutil.copy2(pretrained, checkpoint_path)
            print(f"fold {fold}: 添付済み重みを使用 {pretrained}")
            return checkpoint_path
""")
    cells[9] = code(src9)

    # 5b. all-data refiner セル(cell 10): ckpt存在チェック前に添付重み解決を挿入
    src10 = "".join(cells[10]["source"])
    anchor10 = """if ALLDATA_CKPT.exists():
    print(f"学習済みcheckpointを再利用 {ALLDATA_CKPT}")
else:
    all_ds = InstanceCropDataset(train_files, augment=True)"""
    assert anchor10 in src10, "cell10 anchor mismatch"
    src10 = src10.replace(anchor10, REFINER_PRETRAINED_BLOCK.strip())
    cells[10] = code(src10)

    # 6. submission セル(cell 12): 出力先を WORK_ROOT ベースにし、
    #    test画像のコピー経路も prepare_preprocessed 経由に(Kaggle で PREPROC_CACHE=None)
    src12 = "".join(cells[12]["source"])
    anchor12 = 'OUTPUT_PATH = Path("/content/submission.csv")'
    assert anchor12 in src12, "cell12 output anchor mismatch"
    src12 = src12.replace(anchor12,
        '''_conf_tag = ("" if CFG.SUBMISSION_CONF == 0.26
               else f"_conf{int(round(CFG.SUBMISSION_CONF * 100)):03d}")
OUTPUT_PATH = (WORK_ROOT / f"submission{_conf_tag}.csv"
               if CFG.SUBMISSION_VARIANT == "prod"
               else WORK_ROOT / f"submission_{CFG.SUBMISSION_VARIANT}{_conf_tag}.csv")''')
    # refiner を提出バリアントに応じて解決(prod=all-data EdgeAtt、
    # plain_mhsa_f2=添付 dataset/Drive の ablation plain-MHSA fold2 重み)
    old_refiner = """refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
refiner_ckpt = torch.load(ALLDATA_CKPT, map_location="cuda", weights_only=False)"""
    assert old_refiner in src12, "cell12 refiner anchor mismatch"
    src12 = src12.replace(old_refiner, """if CFG.SUBMISSION_VARIANT == "plain_mhsa_f2":
    plain_ckpt = resolve_pretrained(
        ["ablation_plain_mhsa_seed8407_fold2.pt",
         "refiner_plain_mhsa_fold2_s8407.pt"])
    if plain_ckpt is None:
        raise FileNotFoundError(
            "plain_mhsa_f2 の refiner 重みが見つかりません。"
            "repo weights/、重み dataset、または Drive artifacts に配置してください")
    refiner = MediSegPlainMHSAUNet(3, 1, False).cuda().eval()
    refiner_ckpt = torch.load(plain_ckpt, map_location="cuda", weights_only=False)
elif CFG.SUBMISSION_VARIANT == "prod":
    refiner = MediSegEdgeAttUNet(3, 1, False).cuda().eval()
    refiner_ckpt = torch.load(ALLDATA_CKPT, map_location="cuda", weights_only=False)
else:
    raise ValueError(f"Unknown submission variant: {CFG.SUBMISSION_VARIANT}")""")
    old_test_copy = """for file_name in tqdm(test_files, desc="test画像をローカルへ"):
    local = CFG.PREPROC_DIR / file_name
    if not local.exists():
        cached = CFG.PREPROC_CACHE / file_name
        shutil.copy2(cached, local) if cached.exists() else shutil.copy2(TEST_DIR / file_name, local)"""
    assert old_test_copy in src12, "cell12 test copy anchor mismatch"
    src12 = src12.replace(old_test_copy, """for file_name in tqdm(test_files, desc="test画像をローカルへ"):
    prepare_preprocessed(file_name, TEST_DIR)""")
    # master由来の artifact copy は prod 名固定なので variant で分離する
    # (2本目の実行が第1提出の成果物を上書きしないように)
    old_artifact_copy = 'shutil.copy2(OUTPUT_PATH, CFG.ARTIFACT_ROOT / "submission_master_pipeline.csv")'
    assert old_artifact_copy in src12, "cell12 artifact copy anchor mismatch"
    src12 = src12.replace(old_artifact_copy,
        '''shutil.copy2(OUTPUT_PATH, CFG.ARTIFACT_ROOT /
               (f"submission_master_pipeline_{CFG.SUBMISSION_VARIANT}{_conf_tag}.csv"
                if CFG.SUBMISSION_VARIANT != "prod"
                else f"submission_master_pipeline{_conf_tag}.csv"))''')
    # 計算効率(ルーブリック明記項目): 画像あたり推論wall-clockを計測して記録
    old_loop = """for file_name in tqdm(test_files, desc="test推論"):
    image_path = CFG.PREPROC_DIR / file_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    proposals = single_fold_proposals(image_path)
    scores = np.array([c for c, _ in proposals], dtype=np.float32)
    refined = [refine_proposal(image, m) for _, m in proposals]
    final_masks = apply_ownership(refined, scores)"""
    assert old_loop in src12, "cell12 inference loop anchor mismatch"
    src12 = src12.replace(old_loop, """inference_start = time.time()
for file_name in tqdm(test_files, desc="test推論"):
    image_path = CFG.PREPROC_DIR / file_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    proposals = single_fold_proposals(image_path)
    scores = np.array([c for c, _ in proposals], dtype=np.float32)
    refined = [refine_proposal(image, m) for _, m in proposals]
    final_masks = apply_ownership(refined, scores)""")

    # 最終 print の直前に計算効率サマリを挿入
    old_tail = """print({"rows": len(submission), "images_with_predictions":
       len(set(a["file_name"] for a in audit if a["submitted"] > 0))})"""
    assert old_tail in src12, "cell12 tail anchor mismatch"
    src12 = src12.replace(old_tail, """total_s = time.time() - inference_start
efficiency = {"test_images": len(test_files),
              "total_wall_s": round(total_s, 1),
              "s_per_image": round(total_s / len(test_files), 3),
              "gpu": torch.cuda.get_device_name(0)}
(CFG.ARTIFACT_ROOT / "submission_efficiency.json").write_text(
    json.dumps(efficiency, indent=2), encoding="utf-8")
print(efficiency)
print({"rows": len(submission), "images_with_predictions":
       len(set(a["file_name"] for a in audit if a["submitted"] > 0))})""")
    cells[12] = code(src12)

    # 7. 末尾に環境固定セルを追加
    cells.append(md("## (F) 環境固定記録"))
    cells.append(code(ENV_RECORD_CELL))

    # markdown のセクション番号を再掃き替えはしない(元の A-E ラベルは内部参照無し)
    nb["cells"] = cells
    serialized = json.dumps(nb, ensure_ascii=False, indent=1)
    OUTPUT.write_text(serialized, encoding="utf-8", newline="\n")
    KAGGLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    KAGGLE_OUTPUT.write_text(serialized, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT} ({len(cells)} cells)")

    # AST 検証
    import ast
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            src_no_magics = "\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith("%"))
            try:
                ast.parse(src_no_magics)
            except SyntaxError as error:
                raise SystemExit(f"cell {i} syntax error: {error}")
    print("AST validation passed")


if __name__ == "__main__":
    main()
