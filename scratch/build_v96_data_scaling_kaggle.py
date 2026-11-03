"""V96 Kaggle ノートのビルダー: 検出器のデータ量スケーリング (RESEARCH_LOG §2.67)。

v88 Kaggle 版ノート (`solar_filament_v88_yolo_cls15_fold2_kaggle.ipynb`、§2.48 で
Colab と deterministic 一致を確認済み) の凍結セルに最小置換だけを適用する。
評価経路 (refine・ownership・公式 PQ・低 conf 診断) は無改変で再利用し、
変えるのは (1) 学習に使う fold 集合、(2) cls を prod 既定 0.5 に戻すこと、
(3) ultralytics を prod 学習時の 8.4.118 に固定すること、(4) 出力パス。
置換はすべて「ちょうど 1 回」一致を assert するので、元ノートが変わると
ビルドが失敗する。

arm:
- third_f0 / third_f1: train = fold0 / fold1 のみ (1/3 データ)、fold2 で評価
- full: train = fold0+1+2 (全データ)。fold2 は学習集合に含まれるため評価は
  しない (重みと SHA256 のみ保存、ゲート判定前に学習だけ先行)

Colab 版 (`*_colab.ipynb`) は Drive 解決・前処理画像コピー (セル 1-2) を v89 Colab
ノートから取り、セル 3-6 は Kaggle 版と同一文字列を使う。

    python scratch/build_v96_data_scaling_kaggle.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v88_yolo_cls15_fold2_kaggle.ipynb"
# Colab 版: Drive レイアウト (セル 1-2) は v89 Colab ノート、評価経路 (3-6) は v88 Kaggle と共通
COLAB_SOURCE = ROOT / "solar_filament_v89_cls15_fold01_colab.ipynb"
KERNEL_ROOT = ROOT / "output" / "kaggle_kernel_v96"
ARMS = {"third_f0": (0,), "third_f1": (1,), "full": (0, 1, 2)}
DATASETS = ["shingo257/solar-filament-v35-artifacts",
            "shingo257/solar-filament-2026-weights",
            "shingo257/solar-v73-assets"]


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one match, found {count}: {old[:60]!r}")
    return text.replace(old, new)


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def patch_config(src: str, arm: str, train_folds: tuple[int, ...]) -> str:
    src = replace_once(src, '%pip install -q "ultralytics>=8.3,<9"',
                       '%pip install -q "ultralytics==8.4.118"  # prod 学習時の版に固定')
    src = replace_once(src, "    ARMS = ('cls15',)",
                       f"    ARMS = ('{arm}',)\n    TRAIN_FOLDS = {train_folds!r}")
    src = replace_once(src, 'WORK = Path("/kaggle/working/v88_yolo_cls15")',
                       f'WORK = Path("/kaggle/working/v96_{arm}")')
    src = replace_once(src, 'MANIFEST = WORK / "v88_yolo_cls15_fold2_manifest.json"',
                       f'MANIFEST = WORK / "v96_{arm}_manifest.json"')
    src = replace_once(src, "# --- YOLO (master §B と同一。cls 以外変更禁止) ---",
                       "# --- YOLO (prod と完全一致。v96 は学習 fold 集合のみ変更) ---")
    return replace_once(src, "    YOLO_CLS = 1.5  # prod=0.5 (ultralytics default)。biconcavelens 検証値",
                        "    YOLO_CLS = 0.5  # prod と同一 (ultralytics default)")


def patch_colab_config(src: str, arm: str, train_folds: tuple[int, ...]) -> str:
    src = replace_once(src, '%pip install -q "ultralytics>=8.3,<9" pycocotools',
                       '%pip install -q "ultralytics==8.4.118" pycocotools  # prod 学習時の版に固定')
    src = replace_once(src, "    FOLD = 0  # ← fold1 を評価する場合は 1 に変更して再実行",
                       "    FOLD = 2  # v96: fold2 で評価 (prod runs/yolo_fold_2 と paired)")
    src = replace_once(src, "    ARMS = ('cls15',)",
                       f"    ARMS = ('{arm}',)\n    TRAIN_FOLDS = {train_folds!r}")
    for old in ('"runs" / "v89_yolo_cls15"', '"yolo_datasets" / "v89_yolo_cls15"'):
        src = replace_once(src, old, old.replace("v89_yolo_cls15", "v96_data_scaling"))
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v89")',
                       'LOCAL_ROOT = Path("/content/solar_v96")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / f"v89_yolo_cls15_fold{FOLD}_manifest.json"',
                       f'MANIFEST = ARTIFACT_ROOT / "v96_{arm}_colab_manifest.json"')
    src = replace_once(src, "# --- YOLO (master §B と同一。cls 以外変更禁止) ---",
                       "# --- YOLO (prod と完全一致。v96 は学習 fold 集合のみ変更) ---")
    src = replace_once(src, "    YOLO_CLS = 1.5  # prod=0.5 (ultralytics default)。biconcavelens 検証値",
                       "    YOLO_CLS = 0.5  # prod と同一 (ultralytics default)")
    return replace_once(
        src, "    BASELINE_PQ = None  # fold0/1 の prod 既知値は v80 系評価のみ (経路差あり)。warn-only",
        "    BASELINE_PQ = 0.444238  # fold2 prod 既知値 (§2.48、warn-only)")


def patch_folds(src: str) -> str:
    return replace_once(
        src, "fit_files = [n for n in train_files if fold_of_file[n] != CFG.FOLD]",
        "fit_files = [n for n in train_files if fold_of_file[n] in CFG.TRAIN_FOLDS]")


def patch_dataset(src: str) -> str:
    src = replace_once(
        src, '    root = CFG.YOLO_ROOT / f"fold_{fold}"',
        '    root = CFG.YOLO_ROOT / f"val{fold}_train{\'\'.join(map(str, CFG.TRAIN_FOLDS))}"')
    src = replace_once(
        src, "        else:\n            for image_id in file_to_ids[file_name]:",
        "        if fold_of_file[file_name] in CFG.TRAIN_FOLDS:\n"
        "            for image_id in file_to_ids[file_name]:")
    return replace_once(src, "cls=CFG.YOLO_CLS,  # ★ prod との唯一の差分 (default 0.5 → 1.5)",
                        "cls=CFG.YOLO_CLS,  # prod 既定 0.5 (v96 の差分は学習 fold 集合のみ)")


def patch_gate_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → cls15 学習 → 評価 → ゲート判定、manifest を逐次保存",
                       "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、"
                       "manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)")
    src = replace_once(src, '"experiment_id": "v88_yolo_cls15"', '"experiment_id": "v96_data_scaling"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.40 (cls=1.5 retrain, biconcavelens 診断の検証)",',
                       '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)",\n'
                       '            "train_folds": list(CFG.TRAIN_FOLDS), "ultralytics": version("ultralytics"),')
    src = replace_once(src, '"gate": f"PQ(arm,{CFG.PRIMARY_CONF}) - PQ(prod,{CFG.PRIMARY_CONF}) >= +{CFG.GATE_DELTA}"}',
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"}')
    src = replace_once(src, '    verdict = "promote_to_fold01" if delta >= CFG.GATE_DELTA else '
                            '("reject_noise" if delta >= 0 else "reject")',
                       '    verdict = "cross_arm_gate_§2.67"')
    return replace_once(
        src, '    manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()\n    save_manifest(manifest)\n    print(arm,',
        '    last = Path(weights).parent / "last.pt"\n'
        '    if last.exists():  # 診断 (判定不使用): 全データ学習の best/last 選択方針の根拠\n'
        '        manifest["results"][arm]["last_pt"] = {"weights_sha256": file_sha256(last),\n'
        '                                               "by_conf": evaluate_yolo(last, arm + "_last")}\n'
        '    manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()\n'
        '    save_manifest(manifest)\n    print(arm,')


FULL_RUN = '''# [6/6] 実行 (full arm): 全データ学習のみ。fold2 は学習集合に含まれるため評価しない
def save_manifest(manifest):
    CFG.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

arm = CFG.ARMS[0]
weights, stats = train_arm(arm)
last = Path(weights).parent / "last.pt"
manifest = {
    "contract": {"experiment_id": "v96_data_scaling", "arm": arm,
                 "protocol": "RESEARCH_LOG §2.67 (full-data detector; no OOF evaluation)",
                 "train_folds": list(CFG.TRAIN_FOLDS), "ultralytics": version("ultralytics"),
                 "annotations_sha256": file_sha256(TRAIN_JSON), "folds_sha256": folds_sha256,
                 "yolo_cls": CFG.YOLO_CLS},
    "runtime": {**RUNTIME_INFO, "finished_utc": datetime.now(timezone.utc).isoformat()},
    "datasets": {arm: stats},
    "weights": {"best.pt": {"path": str(weights), "sha256": file_sha256(weights)},
                "last.pt": {"path": str(last), "sha256": file_sha256(last)} if last.exists() else None},
    "note": "val=fold2 is inside the training set; best.pt selection is not held-out",
}
save_manifest(manifest)
print(json.dumps(manifest["weights"], indent=2))
'''


def build(arm: str) -> dict:
    train_folds = ARMS[arm]
    nb = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    cells = nb["cells"]
    cells[0]["source"] = (
        f"# v96 — 検出器データ量スケーリング: `{arm}` (train folds {train_folds}, Kaggle 版)\n\n"
        "RESEARCH_LOG §2.67 の事前登録実験。学習レシピ・評価経路は prod / v88 Kaggle 版と同一、"
        "変更は学習 fold 集合のみ。\n\n"
        "**Add Input 必須**: competition `filament-segmentation-2026` + "
        + " + ".join(f"`{d}`" for d in DATASETS) + "。Internet ON。\n")
    cells[1]["source"] = patch_config(_source(cells[1]), arm, train_folds)
    cells[2]["source"] = patch_folds(_source(cells[2]))
    cells[3]["source"] = patch_dataset(_source(cells[3]))
    cells[6]["source"] = FULL_RUN if arm == "full" else patch_gate_run(_source(cells[6]))
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    return nb


def build_colab(arm: str) -> dict:
    train_folds = ARMS[arm]
    kaggle = build(arm)["cells"]
    nb = copy.deepcopy(json.loads(COLAB_SOURCE.read_text(encoding="utf-8")))
    cells = nb["cells"]
    cells[0]["source"] = (
        f"# v96 — 検出器データ量スケーリング: `{arm}` (train folds {train_folds}, Colab 版)\n\n"
        "RESEARCH_LOG §2.67 の事前登録実験。学習レシピ・評価経路は prod / v88 と同一、"
        "変更は学習 fold 集合のみ。\n\n"
        "**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・"
        "`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・"
        "`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`。\n\n"
        "GPU 必須 (A100 推奨)。上から順に全セル実行。結果は "
        f"`artifacts/v96_{arm}_colab_manifest.json`、重みは "
        "`runs/v96_data_scaling/` に保存される。\n")
    cells[1]["source"] = patch_colab_config(_source(cells[1]), arm, train_folds)
    cells[2]["source"] = patch_folds(_source(cells[2]))
    for index in (3, 4, 5, 6):
        cells[index]["source"] = _source(kaggle[index])
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    return nb


def kernel_metadata(arm: str, code_file: str) -> dict:
    return {"id": f"shingo257/solar-v96-data-scaling-{arm.replace('_', '-')}",
            "title": f"solar-v96-data-scaling-{arm.replace('_', '-')}",
            "code_file": code_file, "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": DATASETS,
            "competition_sources": ["filament-segmentation-2026"],
            "kernel_sources": [], "category_ids": []}


def main() -> None:
    for arm in ARMS:
        name = f"solar_filament_v96_data_scaling_{arm}_kaggle.ipynb"
        payload = json.dumps(build(arm), ensure_ascii=False, indent=1)
        (ROOT / name).write_text(payload, encoding="utf-8")
        kernel_dir = KERNEL_ROOT / arm
        kernel_dir.mkdir(parents=True, exist_ok=True)
        (kernel_dir / name).write_text(payload, encoding="utf-8")
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(kernel_metadata(arm, name), indent=2), encoding="utf-8")
        colab = f"solar_filament_v96_data_scaling_{arm}_colab.ipynb"
        (ROOT / colab).write_text(json.dumps(build_colab(arm), ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"wrote {name} + {kernel_dir} + {colab}")


if __name__ == "__main__":
    main()
