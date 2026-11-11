"""V123 Colab ノートのビルダー: yolo26m detector の all-data 学習 (RESEARCH_LOG §2.99)。

v121c fold-0 ゲート (cons2of3_c26 Δ≥+0.005) 通過時の v122 提出バリアント用に、
all-data detector trio の欠員 (yolo26m) を **事前登録ゲート判定と並行して**
投機的に学習するノートブックを生成する:

- trio 既存メンバ: yolo11m-all = runs/v96_data_scaling/yolo_full_fold2、
  crop-all = runs/v116_croptrain (全 707 file で学習済み = 実質 all-data、
  test-time 用途では漏洩問題なし — §2.98/§2.99 開示参照)
- 本ノート: TRAIN_FOLDS=(0,1,2) で全 707 file を学習 (v96-full det の
  `val2_train012` と同型。FOLD=0 は val/best-epoch 選択 + prod sanity 専用)
- **fold-0 での det eval は in-sample** (学習データを含む) → informational
  sanity のみ、OOF 証拠としては一切使用しない
- ゲート未通過なら artifact は不使用 (投機的計算、RESEARCH_LOG に記録)

v121b ビルダーの fold ミラー: ソースノート・置換機構は同一。
置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v123_yolo26_all_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v123_yolo26_all_colab.ipynb"


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


TITLE = """# v123 — yolo26m detector all-data 学習 (Colab 版・投機的)

RESEARCH_LOG §2.99 の補助実験。v121c fold-0 ゲート通過時に検討する
**all-data detector trio 提出バリアント** の欠員 (yolo26m-all) を
ゲート判定と並行して投機的に学習する。

- train: folds(0,1,2) = 全 707 file (`val0_train012`、v96-full det の
  `val2_train012` と同型。val=fold-0 は best-epoch 選択専用)
- 既存 trio メンバ: yolo11m-all=`yolo_full_fold2`、crop-all=`v116_croptrain`
- **fold-0 det eval は in-sample → informational sanity のみ。
  OOF 証拠・ゲート証拠としては一切使用しない**
- ゲート未通過なら重みは不使用 (投機的計算としてログ記録)

**prod sanity**: fold-0 prod = 0.44413694566994816 (v108 §2.86 確定値)。

GPU 必須 (~4h)。上から順に全セル実行。v121a (crop honest) と
並行実行可。結果は `artifacts/v123_yolo26_all_colab_manifest.json` に逐次保存。
"""


def patch_config(src: str) -> str:
    src = replace_once(src, "    FOLD = 2  # v96: fold2 で評価 (prod runs/yolo_fold_2 と paired)",
                       "    FOLD = 0  # v123: fold0 を val/best-epoch 選択・prod sanity に使用\n"
                       "    # (train は全 fold を含む → fold-0 det eval は in-sample informational のみ)")
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('yolo26m_all',)\n"
                       "    TRAIN_FOLDS = (0, 1, 2)  # all-data: 提出用 det のため全 file 学習\n"
                       "    PARAMS_OF_ARM = {\n"
                       "        'yolo26m_all': {'yolo_model': 'yolo26m-seg.pt'}}")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v123_yolo26_all"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v123_yolo26_all"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v123")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v123_yolo26_all_colab_manifest.json"')
    src = replace_once(src, "    BASELINE_PQ = 0.444238  # fold2 prod 既知値 (§2.48、warn-only)",
                       "    BASELINE_PQ = 0.44413694566994816  # fold0 prod 確定値 (v108 §2.86、warn-only)")
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # informational (in-sample eval; 真のゲートは v121c §2.99)")
    return src


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       "# [6/6] 実行: prod baseline → yolo26m all-data 学習 → 評価 (in-sample informational)、manifest 逐次保存")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v123_yolo26_all"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.99 (all-data det trio 欠員、v121c ゲート通過時の提出バリアント用・投機的)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "params_of_arm": {k: dict(v) for k, v in CFG.PARAMS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "informational in-sample sanity only — 真の gate は v121c consensus eval (§2.99)"')
    src = replace_once(src,
                       """for arm in CFG.ARMS:
    if arm in manifest["results"]:
        continue
    weights, stats = train_arm(arm)""",
                       """for arm in CFG.ARMS:
    if arm in manifest["results"]:
        continue
    p = {"yolo_model": "yolo11m-seg.pt"}  # prod リテラル (CFG 持ち越さない)
    p.update(CFG.PARAMS_OF_ARM.get(arm, {}))
    CFG.YOLO_MODEL = p["yolo_model"]
    # 中断再開 (運用): last.pt 残存 & TRAINING_COMPLETE 無し → 同一 run dir で
    # ultralytics resume (optimizer/scheduler 状態ごと継続・中断なし実行と等価)
    _rd = CFG.RUNS_ROOT / f"yolo_{arm}_fold{CFG.FOLD}"
    _last, _done = _rd / "weights" / "last.pt", _rd / "TRAINING_COMPLETE"
    if _last.exists() and not _done.exists():
        YOLO(str(_last)).train(resume=True)
        _done.write_text("ok", encoding="utf-8")
        weights = _rd / "weights" / "best.pt"
        stats = {"train_samples": None, "resumed_from_last_pt": True}
    else:
        weights, stats = train_arm(arm)""")
    src = replace_once(src, "    manifest[\"datasets\"][arm] = stats\n",
                       "    manifest[\"datasets\"][arm] = stats | {\"params\": p}\n")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = ("informational_positive" if delta >= CFG.GATE_DELTA\n'
                       '               else "informational")  # in-sample のため証拠不使用 (真の判定は v121c §2.99)')
    return src


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = _cell("markdown", TITLE)
    cells[1] = _cell("code", patch_config(_source(cells[1])))
    # cell2 (COCO/fold/前処理)・cell3 (dataset構築+train_arm)・cell4 (refiner)・cell5 (評価関数) は無改変
    cells[6] = _cell("code", patch_run(_source(cells[6])))
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {OUT.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
