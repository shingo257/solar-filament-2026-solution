"""V101 Colab ノートのビルダー: 入力解像度スケーリング imgsz {1536, 2048} (RESEARCH_LOG §2.79)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・評価関数)
を無改変で再利用し、CFG の arms/imgsz/paths と実行セルの experiment_id/verdict
だけを差し替える。変更は imgsz のみ、モデルは全 arm yolo11m-seg、学習 fold
集合・学習レシピ・評価経路は prod と完全一致。推論 imgsz も arm 値を使い、
凍結 eval セル (`model.predict(imgsz=CFG.YOLO_IMGSZ)`) は無改変のまま
arm ループ冒頭で `CFG.YOLO_IMGSZ` を差し替える。

- `prod_infer1536`: 学習なし診断 arm (prod 重みを imgsz1536 で推論のみ)
- `imgsz1536` / `imgsz2048`: yolo11m を imgsz だけ変えて fold0+1 で学習

OOM 時のみ batch=2 へ一度だけ fallback し manifest に batch_fallback を
記録する (§2.79 誠実留保 c)。置換は「ちょうど 1 回」一致を assert するので
元ノートが変わると失敗する。

    python scratch/build_v101_imgsz_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v101_imgsz_colab.ipynb"


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


TITLE = """# v101 — 入力解像度スケーリング: imgsz {1536, 2048} (train folds (0,1), Colab 版)

RESEARCH_LOG §2.79 の事前登録実験。画像はネイティブ 2048² だが imgsz 1280 は
1.6 倍のダウンサンプルで細い構造の情報を物理的に捨てている (§2.78 監査)。
v36 の imgsz1536 は fold0 単独判定で「判定不能」(§2.66)、入力解像度は
**一度も honest 評価されていない最後の上流レバー**。学習レシピ・評価経路は
prod / v96 / v100 と同一、変更は imgsz のみ (全 arm yolo11m-seg)。

- `prod_infer1536`: prod 重みを imgsz1536 で推論するだけの診断 arm (学習なし)
- `imgsz1536`: v36 の honest 複製 (imgsz 1536 で fold0+1 学習)
- `imgsz2048`: ネイティブ解像度 = ダウンサンプル完全解消

**fold0 の事前値 (V30-era、README §P2)**: imgsz1536 再学習 Δ−0.0098・
推論のみ1536 Δ−0.0073 と fold0 では両方負方向 — fold0 単独 CI ~±0.035 で
±0.01 判別不能として残置 (§2.66)。本実験はその負方向が真か fold0
ノイズかを fold2 paired で判別する。

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`。

**ゲート (§2.79、結果取得前に固定)**: 学習 arm 毎に
Δ = PQ(arm,0.26) − PQ(prod,0.26) が >= +0.010 なら fold0 honest 複製へ昇格。
それ以外は解像度スケーリングを閉鎖。fold2 単独では提出を変更しない。
`prod_infer1536` は診断のみでゲート判定対象外。

GPU 必須 (A100 推奨 — imgsz2048/batch4 は OOM の可能性。
OOM の場合のみ batch=2 へ一度だけ fallback し manifest に記録する)。
上から順に全セル実行。結果は `artifacts/v101_imgsz_colab_manifest.json`、
重みは `runs/v101_imgsz/` に保存される。
"""


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('prod_infer1536', 'imgsz1536', 'imgsz2048')\n"
                       "    TRAIN_FOLDS = (0, 1)\n"
                       "    SKIP_TRAIN_ARMS = ('prod_infer1536',)\n"
                       "    IMGSZ_OF_ARM = {'prod_infer1536': 1536, 'imgsz1536': 1536, 'imgsz2048': 2048}")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v101_imgsz"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v101_imgsz"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v101")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v101_imgsz_colab_manifest.json"')
    src = replace_once(src,
                       '    # --- YOLO (prod と完全一致。v96 は学習 fold 集合のみ変更) ---\n'
                       '    YOLO_MODEL = "yolo11m-seg.pt"',
                       '    # --- YOLO (prod と完全一致。v101 は imgsz のみ変更、model 同一) ---\n'
                       '    YOLO_MODEL = "yolo11m-seg.pt"')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # §2.79: LB 判別可能幅をゲートに使う")
    return src


def patch_train(src: str) -> str:
    src = replace_once(src, "# [3/6] prod all-labeler データセット構築 + YOLO 学習 (cls=1.5 のみ変更)",
                       "# [3/6] prod all-labeler データセット構築 + YOLO 学習 (imgsz のみ変更)")
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
    for batch in (CFG.YOLO_BATCH, 2):  # §2.79: OOM 時のみ batch2 へ一度だけ fallback (manifest に記録)
        try:
            YOLO(CFG.YOLO_MODEL).train(  # model 同一・imgsz は CFG.YOLO_IMGSZ (arm 値)
                data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
                batch=batch, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
                seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
                mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
                cls=CFG.YOLO_CLS,  # prod 既定 0.5 (v101 の差分は imgsz のみ)
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
                       "# [6/6] 実行: prod baseline → imgsz arm (prod_infer1536 は学習なし診断)")
    src = replace_once(src, "(判定は §2.67 の arm 横断ゲートで行う)",
                       "(判定は §2.79 の per-arm ゲートで行う)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v101_imgsz_scaling"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.79 (input resolution scaling)"')
    src = replace_once(src, '"yolo_cls": CFG.YOLO_CLS,',
                       '"yolo_cls": CFG.YOLO_CLS, "imgsz_of_arm": dict(CFG.IMGSZ_OF_ARM),')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.79)"')
    src = replace_once(
        src,
        """    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""",
        """    CFG.YOLO_IMGSZ = CFG.IMGSZ_OF_ARM[arm]  # train/infer で解像度を揃える (凍結 eval は CFG を読む)
    if arm in CFG.SKIP_TRAIN_ARMS:
        weights, stats = CFG.PROD_YOLO, {"infer_only": True}
    else:
        weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = ("diagnostic_no_train" if arm in CFG.SKIP_TRAIN_ARMS\n'
                       '               else "promote_fold0_replication" if delta >= CFG.GATE_DELTA\n'
                       '               else "reject_imgsz_fold2")')
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
