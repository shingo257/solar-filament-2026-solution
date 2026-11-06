"""V108 Colab ノートのビルダー: mask_ratio1 の fold-0 honest 複製 (RESEARCH_LOG §2.86)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・
評価関数) を無改変で再利用し、評価 fold を 0 に、train folds を (1,2) に
差し替える。paired baseline は既存 `runs/yolo_fold_0` (prod recipe・
mask_ratio=2・folds(1,2) 学習済み・同 seed 3407) を同一経路で再評価し、
arm は mask_ratio=1 のみ変更した新規学習。

- prod_f0 (baseline): 既存 yolo_fold_0 best.pt を fold-0 OOF で再評価
  (既知値 0.4443 との sanity 確認)
- mask_ratio1_f0: mask proto 2→1 のみ変更して学習 → fold-0 OOF 評価

判定基準は §2.86 で結果取得前に固定: Δ@0.26 >= +0.003 → cross-fold
検証済み (prod レシピへの mr1 適用を検討対象化)、未満なら棄却。

置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v108_mr1f0_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v108_mr1f0_colab.ipynb"


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


TITLE = """# v108 — mask_ratio1 の fold-0 honest 複製 (Colab 版)

RESEARCH_LOG §2.86 の事前登録実験。v103 `mask_ratio1` が fold2 で
**Δ+0.0065** (全史上最大・ゲート +0.010 未達で棄却判定済み)。
この節は sub-gate arm への複製追加という**ゲート拡張**であり、
判定基準は fold-0 結果を見る前に固定済み。

| arm | 内容 |
|---|---|
| prod_f0 (baseline) | 既存 `runs/yolo_fold_0` (prod recipe, folds(1,2)) を fold-0 OOF で再評価 — 既知値 0.4443 sanity |
| `mask_ratio1_f0` | mask_ratio 2→1 のみ変更して folds(1,2) 学習 → fold-0 OOF |

**判定基準 (§2.86 固定)**: Δ@0.26 >= **+0.003** なら cross-fold 検証済み
(prod レシピへの mr1 適用を検討対象化)。未満なら棄却・提出不変。

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold0_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_0/weights/best.pt`。

GPU 必須 (~6.5h: mask_ratio=1 は proto 4x で ~5.8h 学習 + 評価 2 回)。
上から順に全セル実行。結果は
`artifacts/v108_mr1_f0_colab_manifest.json`、重みは
`runs/v108_mr1_f0/` に保存される。
"""


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('mask_ratio1_f0',)\n"
                       "    TRAIN_FOLDS = (1, 2)\n"
                       "    PROD_TRAIN_PARAMS = {'imgsz': 1280, 'epochs': 70, 'patience': 14,\n"
                       "                         'mask_ratio': 2, 'copy_paste': 0.0}\n"
                       "    TRAIN_PARAMS_OF_ARM = {'mask_ratio1_f0': {'mask_ratio': 1}}")
    src = replace_once(src, "    FOLD = 2  # v96: fold2 で評価 (prod runs/yolo_fold_2 と paired)",
                       "    FOLD = 0  # v108: fold0 で複製 (prod runs/yolo_fold_0 と paired)")
    src = replace_once(src, "    BASELINE_PQ = 0.444238  # fold2 prod 既知値 (§2.48、warn-only)",
                       "    BASELINE_PQ = 0.4443  # fold-0 prod OOF 既知値 (warn-only)")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v108_mr1_f0"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v108_mr1_f0"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v108")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v108_mr1_f0_colab_manifest.json"')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.003  # §2.86: cross-fold 複製の検証幅 (fold0 結果取得前に固定)")
    return src


def patch_train(src: str) -> str:
    src = replace_once(src, "# [3/6] prod all-labeler データセット構築 + YOLO 学習 (cls=1.5 のみ変更)",
                       "# [3/6] prod all-labeler データセット構築 + YOLO 学習 (mask_ratio のみ変更)")
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
    for batch in (CFG.YOLO_BATCH, 2):  # §2.86: OOM 時のみ batch2 へ一度だけ fallback (manifest に記録)
        try:
            YOLO(CFG.YOLO_MODEL).train(  # model 同一・mask_ratio は実行セルで CFG に設定済み
                data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
                batch=batch, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
                seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
                mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
                cls=CFG.YOLO_CLS,  # prod 既定 0.5 (v108 の差分は mask_ratio のみ)
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
                       "# [6/6] 実行: prod fold-0 baseline 再評価 → mask_ratio1_f0 学習 → fold-0 paired 評価、manifest 逐次保存 (判定は §2.86 の固定基準)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v108_mr1_fold0"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.86 (mask_ratio1 fold-0 honest replication)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "train_params_of_arm": {k: dict(v) for k, v in CFG.TRAIN_PARAMS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(mr1_f0,0.26)-PQ(prod_f0,0.26) >= +0.003 → cross-fold verified (§2.86)"')
    src = replace_once(
        src,
        """    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""",
        """    p = dict(CFG.PROD_TRAIN_PARAMS); p.update(CFG.TRAIN_PARAMS_OF_ARM.get(arm, {}))
    CFG.YOLO_IMGSZ = p["imgsz"]; CFG.YOLO_EPOCHS = p["epochs"]
    CFG.YOLO_PATIENCE = p["patience"]; CFG.YOLO_MASK_RATIO = p["mask_ratio"]
    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats | {"params": p}
    by_conf = evaluate_yolo(weights, arm)""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = "verified_cross_fold" if delta >= CFG.GATE_DELTA else "unverified_fold0"')
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
