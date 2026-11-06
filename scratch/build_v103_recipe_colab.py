"""V103 Colab ノートのビルダー: 学習レシピ残差 (RESEARCH_LOG §2.81)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・
評価関数) を無改変で再利用し、CFG の arms/paths、train セルの copy_paste
引数追加、実行セルの per-arm パラメータ差替えだけを行う。変更は各 arm の
単一学習パラメータのみ、モデルは全 arm yolo11m-seg、fold 集合・評価経路は
prod と完全一致。

- `prod_infer960`: 学習なし診断 (prod 重みを imgsz960 で推論のみ)
- `imgsz960`: 解像度曲線の下方向完結 (train/infer 960)
- `epochs120`: epochs 120 / patience 20 (prod best_epoch=70 終端 = 未収束)
- `mask_ratio1`: YOLO_MASK_RATIO 2→1 (mask proto 640²→1280²)
- `copypaste01`: copy_paste 0.0→0.1 (v60 未実行 aug)

OOM 時のみ batch=2 へ一度だけ fallback し manifest に batch_fallback を
記録する (§2.81、mask_ratio1 の proto メモリ ~4x 対策)。置換は
「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v103_recipe_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v103_recipe_colab.ipynb"


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


TITLE = """# v103 — 学習レシピ残差: 未収束・マスク分解能・解像度下方向・copy-paste (Colab 版)

RESEARCH_LOG §2.81 の事前登録実験。prod は best_epoch=70 (終端) で
**未収束**、§2.78 で境界側に残余余地、v101 で解像度の用量反応が単調負
(1280 がピークとは限らない) — 学習レシピの単一パラメータを fold2 paired
で評価する。全 arm yolo11m-seg、train folds (0,1)、評価 fold2。

| arm | 変更 (prod → arm) |
|---|---|
| `prod_infer960` (診断) | 学習なし、prod 重みを imgsz960 で推論 |
| `imgsz960` | imgsz 1280 → 960 (train/infer) |
| `epochs120` | epochs 70→120 / patience 14→20 |
| `mask_ratio1` | YOLO_MASK_RATIO 2→1 (proto 640²→1280²) |
| `copypaste01` | copy_paste 0.0→0.1 |

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`。

**ゲート (§2.81、結果取得前に固定)**: 学習 arm 毎に
Δ = PQ(arm,0.26) − PQ(prod,0.26) が >= +0.010 なら fold0 honest 複製へ。
`prod_infer960` は診断のみでゲート対象外。fold2 単独では提出を変更しない。

GPU 必須 (~11–13h 見込み。mask_ratio1 は proto メモリ ~4x で OOM の可能性、
OOM 時のみ batch=2 へ一度だけ fallback し manifest に記録)。上から順に
全セル実行。結果は `artifacts/v103_recipe_colab_manifest.json`、
重みは `runs/v103_recipe/` に保存される。
"""


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('prod_infer960', 'imgsz960', 'epochs120', 'mask_ratio1', 'copypaste01')\n"
                       "    TRAIN_FOLDS = (0, 1)\n"
                       "    SKIP_TRAIN_ARMS = ('prod_infer960',)\n"
                       "    IMGSZ_OF_ARM = {'prod_infer960': 960, 'imgsz960': 960, 'epochs120': 1280,\n"
                       "                    'mask_ratio1': 1280, 'copypaste01': 1280}\n"
                       "    PROD_TRAIN_PARAMS = {'imgsz': 1280, 'epochs': 70, 'patience': 14,\n"
                       "                         'mask_ratio': 2, 'copy_paste': 0.0}\n"
                       "    TRAIN_PARAMS_OF_ARM = {'prod_infer960': {'imgsz': 960},\n"
                       "                           'imgsz960': {'imgsz': 960},\n"
                       "                           'epochs120': {'epochs': 120, 'patience': 20},\n"
                       "                           'mask_ratio1': {'mask_ratio': 1},\n"
                       "                           'copypaste01': {'copy_paste': 0.1}}")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v103_recipe"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v103_recipe"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v103")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v103_recipe_colab_manifest.json"')
    src = replace_once(src,
                       '    # --- YOLO (prod と完全一致。v96 は学習 fold 集合のみ変更) ---\n'
                       '    YOLO_MODEL = "yolo11m-seg.pt"',
                       '    # --- YOLO (prod と完全一致。v103 は arm 毎に単一パラメータのみ変更) ---\n'
                       '    YOLO_MODEL = "yolo11m-seg.pt"\n'
                       '    YOLO_COPY_PASTE = 0.0  # prod 既定 (v103: arm のみ 0.1)')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # §2.81: LB 判別可能幅をゲートに使う")
    return src


def patch_train(src: str) -> str:
    src = replace_once(src, "# [3/6] prod all-labeler データセット構築 + YOLO 学習 (cls=1.5 のみ変更)",
                       "# [3/6] prod all-labeler データセット構築 + YOLO 学習 (arm 毎の単一パラメータのみ変更)")
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
    for batch in (CFG.YOLO_BATCH, 2):  # §2.81: OOM 時のみ batch2 へ一度だけ fallback (manifest に記録)
        try:
            YOLO(CFG.YOLO_MODEL).train(  # model 同一・arm 値は実行セルで CFG に設定済み
                data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
                batch=batch, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
                seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
                mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
                cls=CFG.YOLO_CLS,  # prod 既定 0.5 (v103 の差分は arm 単一パラメータのみ)
                copy_paste=CFG.YOLO_COPY_PASTE,  # prod 既定 0.0
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
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       "# [6/6] 実行: prod baseline → 学習レシピ残差 arm (prod_infer960 は学習なし診断、manifest 逐次保存、判定は §2.81 の per-arm ゲート)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v103_recipe_residuals"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.81 (training-recipe residuals)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "imgsz_of_arm": dict(CFG.IMGSZ_OF_ARM),\n'
                       '            "train_params_of_arm": {k: dict(v) for k, v in CFG.TRAIN_PARAMS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.81)"')
    src = replace_once(
        src,
        """    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""",
        """    p = dict(CFG.PROD_TRAIN_PARAMS); p.update(CFG.TRAIN_PARAMS_OF_ARM.get(arm, {}))
    CFG.YOLO_IMGSZ = p["imgsz"]; CFG.YOLO_EPOCHS = p["epochs"]
    CFG.YOLO_PATIENCE = p["patience"]; CFG.YOLO_MASK_RATIO = p["mask_ratio"]
    CFG.YOLO_COPY_PASTE = p["copy_paste"]
    if arm in CFG.SKIP_TRAIN_ARMS:
        weights, stats = CFG.PROD_YOLO, {"infer_only": True}
    else:
        weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats | {"params": p}
    by_conf = evaluate_yolo(weights, arm)""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = ("diagnostic_no_train" if arm in CFG.SKIP_TRAIN_ARMS\n'
                       '               else "promote_fold0_replication" if delta >= CFG.GATE_DELTA\n'
                       '               else "reject_recipe_fold2")')
    src = replace_once(src, "    if last.exists():  # 診断 (判定不使用): 全データ学習の best/last 選択方針の根拠",
                       "    if last.exists() and arm not in CFG.SKIP_TRAIN_ARMS:  # 診断 (判定不使用): best/last 選択方針の根拠")
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
