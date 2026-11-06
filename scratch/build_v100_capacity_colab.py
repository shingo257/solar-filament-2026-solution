"""V100 Colab ノートのビルダー: 検出器容量スケーリング yolo11l/x-seg (RESEARCH_LOG §2.76)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・評価関数)
を無改変で再利用し、CFG の arms/paths/model と実行セルの experiment_id/verdict
だけを差し替える。変更はモデルスケールのみ (m→l/x)、学習 fold 集合・学習レシピ・
評価経路は prod と完全一致。OOM 時のみ batch=2 へ一度だけ fallback し
manifest に batch_fallback を記録する (§2.76 誠実留保 c)。
置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v100_capacity_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v100_capacity_colab.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one match, found {count}: {old[:60]!r}")
    return text.replace(old, new)


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cell(cell_type: str, source: str) -> dict:
    base = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


TITLE = """# v100 — 検出器容量スケーリング: yolo11l/x-seg (train folds (0,1), Colab 版)

RESEARCH_LOG §2.76 の事前登録実験。V36–V41 の大規模検出器は fold0 単独判定で
「判定不能」と訂正済み (§2.66) — **fold2 paired で一度も honest 評価されていない
最後の上流レバー**。学習レシピ・評価経路は prod / v96 と同一、変更は
モデルスケールのみ (yolo11m → yolo11l / yolo11x)。

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`。

**ゲート (§2.76、結果取得前に固定)**: arm 毎に
Δ = PQ(arm,0.26) − PQ(prod,0.26) が >= +0.010 なら fold0 honest 複製へ昇格。
それ以外は容量スケーリングを閉鎖。fold2 単独では提出を変更しない。

GPU 必須 (A100 推奨 — yolo11x-seg/imgsz1280/batch4 は T4 では OOM の可能性。
OOM の場合のみ batch=2 へ一度だけ fallback し manifest に記録する)。
上から順に全セル実行。結果は `artifacts/v100_capacity_colab_manifest.json`、
重みは `runs/v100_capacity/` に保存される。
"""


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('yolo11l', 'yolo11x')\n    TRAIN_FOLDS = (0, 1)")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v100_capacity"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v100_capacity"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v100")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v100_capacity_colab_manifest.json"')
    src = replace_once(src,
                       '    # --- YOLO (prod と完全一致。v96 は学習 fold 集合のみ変更) ---\n'
                       '    YOLO_MODEL = "yolo11m-seg.pt"',
                       '    # --- YOLO (prod と完全一致。v100 は model scale のみ変更) ---\n'
                       '    YOLO_MODELS = {"yolo11l": "yolo11l-seg.pt", "yolo11x": "yolo11x-seg.pt"}')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # §2.76: LB 判別可能幅をゲートに使う")
    return src


def patch_train(src: str) -> str:
    src = replace_once(src, "# [3/6] prod all-labeler データセット構築 + YOLO 学習 (cls=1.5 のみ変更)",
                       "# [3/6] prod all-labeler データセット構築 + YOLO 学習 (model scale のみ変更)")
    src = replace_once(
        src,
        """    started = time.time()
    YOLO(CFG.YOLO_MODEL).train(
        data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
        batch=CFG.YOLO_BATCH, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
        mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
        cls=CFG.YOLO_CLS,  # prod 既定 0.5 (v96 の差分は学習 fold 集合のみ)
        optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
        degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
        project=str(CFG.RUNS_ROOT), name=run_name, exist_ok=True,
    )""",
        """    started = time.time()
    for batch in (CFG.YOLO_BATCH, 2):  # §2.76: OOM 時のみ batch2 へ一度だけ fallback (manifest に記録)
        try:
            YOLO(CFG.YOLO_MODELS[arm]).train(
                data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
                batch=batch, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
                seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
                mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
                cls=CFG.YOLO_CLS,  # prod 既定 0.5 (v100 の差分は model scale のみ)
                optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
                degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
                project=str(CFG.RUNS_ROOT), name=run_name, exist_ok=True,
            )
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch == 2:
                raise
            stats["batch_fallback"] = True""")
    return src


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習",
                       "# [6/6] 実行: prod baseline → 大容量検出器 arm 学習")
    src = replace_once(src, "(判定は §2.67 の arm 横断ゲートで行う)",
                       "(判定は §2.76 の per-arm ゲートで行う)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v100_detector_capacity"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.76 (detector capacity scaling)"')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.76)"')
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = ("promote_fold0_replication" if delta >= CFG.GATE_DELTA\n'
                       '               else "reject_capacity_fold2")')
    return src


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = _cell("markdown", TITLE)
    cells[1]["source"] = patch_config(_source(cells[1])).splitlines(keepends=True)
    # cell2 (COCO/fold/前処理) と cell4 (refiner 定義) と cell5 (評価関数) は無改変
    cells[3]["source"] = patch_train(_source(cells[3])).splitlines(keepends=True)
    cells[6]["source"] = patch_run(_source(cells[6])).splitlines(keepends=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {OUT.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
