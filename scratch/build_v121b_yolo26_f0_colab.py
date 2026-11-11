"""V121b Colab ノートのビルダー: yolo26m detector の fold-0 学習 (RESEARCH_LOG §2.99)。

v120 `cons2of3_c26` (Δ@0.45=+0.01155、§2.98 ゲート通過) の fold-0 honest
複製 Phase A: v117 yolo26m-seg レシピ verbatim を folds(1,2) で学習し
fold-0 OOF 用重みを作る。v117 ビルダーの fold ミラー: FOLD=0、
TRAIN_FOLDS=(1,2)。生成重みの weights_sha256 は manifest
`results.yolo26m` に記録 → v121c が trusted-pin として読み取る。

置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v121b_yolo26_f0_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v121b_yolo26_f0_colab.ipynb"


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


TITLE = """# v121b — yolo26m detector fold-0 学習 (Colab 版)

RESEARCH_LOG §2.99 の事前登録実験 (v120 cons2of3_c26 +0.01155 gate 通過の
fold-0 複製 Phase A)。v117 yolo26m-seg を prod レシピ verbatim・
folds(1,2) で学習。fold-0 は dataset 構築・学習・val 全て非接触。

| arm | モデル | その他 |
|---|---|---|
| `yolo26m` | yolo26m-seg.pt | 全て prod verbatim |

生成重み (`runs/v121_yolo26_f0/yolo_yolo26m_fold0/weights/best.pt`) と
`results.yolo26m.weights_sha256` が v121c consensus eval の `yolo26`
detector 入力・trusted byte-pin になる。fold-0 での yolo26m 単独 PQ は
informational paired sanity として併記。

**prod sanity**: fold-0 prod = 0.44413694566994816 (v108 §2.86 確定値)。

GPU 必須 (~3.5h)。上から順に全セル実行。
結果は `artifacts/v121b_yolo26_f0_colab_manifest.json` に逐次保存。
"""


def patch_config(src: str) -> str:
    src = replace_once(src, "    FOLD = 2  # v96: fold2 で評価 (prod runs/yolo_fold_2 と paired)",
                       "    FOLD = 0  # v121b: fold0 で評価 (prod runs/yolo_fold_0 と paired)")
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('yolo26m',)\n"
                       "    TRAIN_FOLDS = (1, 2)  # prod fold-0 recipe (fold-0 非接触)\n"
                       "    PARAMS_OF_ARM = {\n"
                       "        'yolo26m': {'yolo_model': 'yolo26m-seg.pt'}}")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v121_yolo26_f0"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v121_yolo26_f0"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v121b")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v121b_yolo26_f0_colab_manifest.json"')
    src = replace_once(src, "    BASELINE_PQ = 0.444238  # fold2 prod 既知値 (§2.48、warn-only)",
                       "    BASELINE_PQ = 0.44413694566994816  # fold0 prod 確定値 (v108 §2.86、warn-only)")
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # informational (真のゲートは v121c §2.99)")
    return src


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       "# [6/6] 実行: prod baseline → yolo26m fold-0 学習 → 評価、manifest 逐次保存 (informational、真の判定は v121c §2.99)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v121b_yolo26_f0"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.99 (cons2of3_c26 fold-0 replication, v117 yolo26m recipe verbatim)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "params_of_arm": {k: dict(v) for k, v in CFG.PARAMS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "informational paired sanity only — 真の gate は v121c consensus eval (§2.99)"')
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
                       '               else "informational")  # 真の判定は v121c (§2.99)')
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
