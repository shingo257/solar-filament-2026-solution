"""V124a Colab ノートのビルダー: crop detector の fold-2 honest 学習 (RESEARCH_LOG §2.99)。

v120 `cons2of3_c26` の fold-2 証拠は v116 crop det の漏洩で撤回済み。
fold-0 (v121a honest + v121b) が clean 複製の第一 fold で、v121c ゲートを
通過した場合に限り **第二 fold として clean な fold-2 複製が必要**になる
(片 fold 陽性だけでは cross-fold verified と呼べない — 調査計画 2026-11-10
P1 の指摘に対応)。

本ノートは v121a honest ビルダーの fold ミラー:
- FOLD=2 (eval/valid)、TRAIN_FOLDS=(0,1) → fit_files=472 file のみで
  crop dataset 構築 (fold-2 の crop は train に混入しない)
- 出力先 `v124_croptrain_f2_honest` (v116 汚染 run/dataset とは完全分離)
- v116 croptrain レシピ verbatim (896² tile×9 crops・全 record・imgsz=896)
- prod sanity: fold-2 prod = 0.444238 (§2.48 既知値、warn-only)

**実行条件**: v121c が fold-0 で `cons2of3_c26_delta >= +0.005` を返した
場合のみ投入する (事前登録ゲートの第二 fold 確認)。ゲート未通過なら
実行しない (fold-0 棄却なら consensus 線は閉鎖のため)。

生成重み (`runs/v124_croptrain_f2_honest/yolo_croptrain_fold2/weights/best.pt`)
と `datasets.croptrain.weights_sha256` が v124c fold-2 consensus eval の
`crop` detector 入力・trusted byte-pin になる。

    python scratch/build_v124a_croptrain_f2_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

try:  # pytest では scratch パッケージ経由、直接実行では同ディレクトリ経由
    from scratch.build_v121a_croptrain_f0_colab import (
        CELL3 as _CELL3,
        CELL6 as _CELL6,
        patch_config as _patch_config_v121a,
        replace_once,
    )
except ImportError:  # pragma: no cover - 直接 python scratch/xxx.py で実行時
    from build_v121a_croptrain_f0_colab import (
        CELL3 as _CELL3,
        CELL6 as _CELL6,
        patch_config as _patch_config_v121a,
        replace_once,
    )

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v124a_croptrain_f2_colab.ipynb"


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cell(cell_type: str, source: str) -> dict:
    base = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


TITLE = """# v124a — crop detector fold-2 honest 学習 (Colab 版、条件付き)

RESEARCH_LOG §2.99 補完。**v121c (fold-0) ゲート通過時のみ実行** —
v120 の fold-2 +0.01155 は v116 crop det の漏洩で撤回済みのため、
fold-2 側の clean 複製が「第二 fold」の証拠になる。

v116 croptrain レシピ verbatim — folds(0,1) の全 annotator レコードを
**896² タイル crop (stride 448、3×3=9/画像)** に展開し yolo11m-seg
学習 (imgsz=896、他は prod verbatim)。fold-2 の crop は train に混入
しない (`fit_files`=472 file のみ)。fold-2 は val/best-epoch 選択に使用
(prod yolo_fold_2・fold-2 refiner と同じ eval-fold-as-val 構成のため
paired 対称 — §2.99 監査注記)。

| arm | 推論 | 目的 |
|---|---|---|
| `croptrain_full` | full-image imgsz=1280 | fold-2 上での crop 重み paired sanity |

生成重み (`runs/v124_croptrain_f2_honest/yolo_croptrain_fold2/weights/best.pt`)
と `datasets.croptrain.weights_sha256` が v124c fold-2 consensus eval の
`crop` detector 入力・trusted byte-pin になる。

**prod sanity**: fold-2 prod = 0.444238 (§2.48 既知値) を paired 再測定。

GPU 必須 (~13h: dataset ~25min + 学習 ~12h + 評価 ~15min)。manifest 逐次保存で
resume 可能。結果は `artifacts/v124a_croptrain_f2_colab_manifest.json`。
"""


def patch_config(src: str) -> str:
    """v121a のパッチを流用し、fold-0 → fold-2 の差分だけ上書きする。"""
    src = _patch_config_v121a(src)
    src = replace_once(src, "    FOLD = 0  # v121a: fold0 で評価 (prod runs/yolo_fold_0 と paired)",
                       "    FOLD = 2  # v124a: fold2 で評価 (prod runs/yolo_fold_2 と paired、honest 複製)")
    src = replace_once(src, "    TRAIN_FOLDS = (1, 2)  # prod fold-0 recipe (fold-0 非接触)",
                       "    TRAIN_FOLDS = (0, 1)  # prod fold-2 recipe (fold-2 crop 非接触)")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v121_croptrain_f0_honest"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v124_croptrain_f2_honest"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v121_croptrain_f0_honest"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v124_croptrain_f2_honest"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v121a")',
                       'LOCAL_ROOT = Path("/content/solar_v124a")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v121a_croptrain_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v124a_croptrain_f2_colab_manifest.json"')
    src = replace_once(src, "    BASELINE_PQ = 0.44413694566994816  # fold0 prod 確定値 (v108 §2.86、warn-only)",
                       "    BASELINE_PQ = 0.444238  # fold2 prod 既知値 (§2.48、warn-only)")
    return src


CELL3 = (_CELL3
         .replace("v121a crop データセット構築 + YOLO 学習 (v116 §2.94 レシピ verbatim、folds(1,2))",
                  "v124a crop データセット構築 + YOLO 学習 (v116 §2.94 レシピ verbatim、folds(0,1))")
         .replace("folds(1,2) 全 record × 9 crop + fold-0 reference record の val crop",
                  "folds(0,1) 全 record × 9 crop + fold-2 reference record の val crop"))

CELL6 = (_CELL6
         .replace('"experiment_id": "v121a_croptrain_f0"',
                  '"experiment_id": "v124a_croptrain_f2"')
         .replace("cons2of3_c26 fold-0 replication, v116 crop recipe verbatim",
                  "cons2of3_c26 fold-2 honest replication, v116 crop recipe verbatim")
         .replace("真の判定は v121c §2.99", "真の判定は v124c §2.99")
         .replace("真の gate は v121c consensus eval (§2.99)", "真の gate は v124c consensus eval (§2.99)"))


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = _cell("markdown", TITLE)
    cells[1] = _cell("code", patch_config(_source(cells[1])))
    # cell2 (COCO/fold/前処理)・cell4 (refiner 定義)・cell5 (評価関数) は無改変
    cells[3] = _cell("code", CELL3)
    cells[6] = _cell("code", CELL6)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {OUT.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
