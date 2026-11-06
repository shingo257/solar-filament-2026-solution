"""V102 Colab ノートのビルダー: 推論パラメータ掃引 (RESEARCH_LOG §2.80)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・
評価関数) を無改変で再利用する。v101 と違い学習は一切行わない — 全 arm が
prod fold2 YOLO 重み + all-data EdgeAtt refiner の推論経路で、宣言済み掃引
(conf/閾値/drop/min_area/重み選択) に列挙されなかった残存 CFG だけを変える:

- `nms_iou_035` / `nms_iou_075`: PREDICT_IOU 0.55 → {0.35, 0.75}
- `crop_ctx_130` / `crop_ctx_200`: CROP_CONTEXT 1.65 → {1.30, 2.00}
- `crop_size_448`: CROP_SIZE 320 → 448 (off-train-res リスク明示)
- `prop_dilate_2` / `prop_dilate_4`: refine 前の proposal disk 膨張 0 → {2, 4}

凍結 eval セルの `mask_square_box` は `context=CFG.CROP_CONTEXT` を
default-arg に bind するため CFG 差替が効かない。実行セル冒頭で call 時に
CFG を読む同等 wrapper に差し替える (prod 既定値では元関数と同一経路、
prod sanity ゲートが実測で担保)。PROPOSAL_DILATE_PX も同じ wrapper で実装。

置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v102_infer_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v102_infer_colab.ipynb"


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


TITLE = """# v102 — 推論パラメータ掃引: 宣言済み掃引に列挙されなかった残存 CFG (Colab 版)

RESEARCH_LOG §2.80 の事前登録実験。§2.23 末で「残存パラメータなし」と
宣言した掃引リスト (conf/閾値/drop/min_area/重み選択) に `PREDICT_IOU`・
`CROP_CONTEXT`・`CROP_SIZE`・proposal の refine 前カバー率は含まれて
いなかった。§2.78 監査で残余余地は存在の不一致ではなく**境界側**
(matched-pair IoU 0.63) にあると定量されたため、これらを fold2 paired で
評価する。**全 arm 学習なし** (prod fold2 重み + all-data refiner の推論のみ)。

| arm | 変更 (prod → arm) |
|---|---|
| `nms_iou_035` / `nms_iou_075` | PREDICT_IOU 0.55 → 0.35 / 0.75 |
| `crop_ctx_130` / `crop_ctx_200` | CROP_CONTEXT 1.65 → 1.30 / 2.00 |
| `crop_size_448` | CROP_SIZE 320 → 448 |
| `prop_dilate_2` / `prop_dilate_4` | refine 前 proposal 膨張 0 → 2 / 4 px |

`prop_dilate_*` は `refined & proposal` のクリッピング緩和 (B2 §2.37 の
refine 後盲目膨縮とは機構が異なる)。

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`。

**ゲート (§2.80、結果取得前に固定)**: arm 毎に
Δ = PQ(arm,0.26) − PQ(prod,0.26) が >= +0.010 なら fold0 honest 複製へ昇格。
7 cell の固定 grid なので偶然超過の可能性は fold0 複製で制御する。
fold2 単独では提出を変更しない。

GPU 必須 (学習なし、全体 ~3h 見込み)。上から順に全セル実行。
結果は `artifacts/v102_infer_colab_manifest.json` に逐次保存される。
"""


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('nms_iou_035', 'nms_iou_075', 'crop_ctx_130', 'crop_ctx_200',\n"
                       "            'crop_size_448', 'prop_dilate_2', 'prop_dilate_4')\n"
                       "    TRAIN_FOLDS = (0, 1)  # 記録用 (v102 は学習しない)\n"
                       "    PROD_INFER_PARAMS = {'predict_iou': 0.55, 'crop_context': 1.65,\n"
                       "                         'crop_size': 320, 'proposal_dilate_px': 0}\n"
                       "    PARAMS_OF_ARM = {'nms_iou_035': {'predict_iou': 0.35},\n"
                       "                     'nms_iou_075': {'predict_iou': 0.75},\n"
                       "                     'crop_ctx_130': {'crop_context': 1.30},\n"
                       "                     'crop_ctx_200': {'crop_context': 2.00},\n"
                       "                     'crop_size_448': {'crop_size': 448},\n"
                       "                     'prop_dilate_2': {'proposal_dilate_px': 2},\n"
                       "                     'prop_dilate_4': {'proposal_dilate_px': 4}}\n"
                       "    PROPOSAL_DILATE_PX = 0  # refine 前 proposal 膨張幅 (prod=0)")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v102_infer"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v102_infer"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v102")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v102_infer_colab_manifest.json"')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # §2.80: LB 判別可能幅をゲートに使う")
    return src


RUN_HEAD = """# [6/6] 実行: prod baseline → 推論パラメータ掃引 (全 arm 学習なし、manifest 逐次保存、判定は §2.80 の per-arm ゲート)
"""


WRAPPERS = '''
# --- v102: 凍結 eval セルの `mask_square_box`/`refine_proposal` は CFG 値を
# default-arg に bind するため、arm 毎の変更が効くよう call 時に CFG を読む
# 同等 wrapper に差し替える (prod 既定値では元関数と同一経路)。
_orig_mask_square_box = mask_square_box
def mask_square_box(mask, context=None, minimum=96):  # noqa: F811
    return _orig_mask_square_box(mask, CFG.CROP_CONTEXT if context is None else context, minimum)

_orig_refine_proposal = refine_proposal
def refine_proposal(image, proposal):  # noqa: F811
    if CFG.PROPOSAL_DILATE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CFG.PROPOSAL_DILATE_PX + 1,) * 2)
        proposal = cv2.dilate(proposal.astype(np.uint8), k) != 0
    return _orig_refine_proposal(image, proposal)

'''


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       RUN_HEAD.rstrip("\n"))
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v102_infer_params"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.80 (inference-parameter sweep)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "params_of_arm": {k: dict(v) for k, v in CFG.PARAMS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.80)"')
    src = replace_once(src, 'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}',
                       'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}\n'
                       + WRAPPERS)
    src = replace_once(
        src,
        """    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""",
        """    p = dict(CFG.PROD_INFER_PARAMS); p.update(CFG.PARAMS_OF_ARM.get(arm, {}))
    CFG.PREDICT_IOU, CFG.CROP_CONTEXT = p["predict_iou"], p["crop_context"]
    CFG.CROP_SIZE, CFG.PROPOSAL_DILATE_PX = p["crop_size"], p["proposal_dilate_px"]
    weights = CFG.PROD_YOLO  # v102: 全 arm 学習なし (prod 重みの推論のみ)
    manifest["datasets"][arm] = {"infer_only": True, "params": p}
    by_conf = evaluate_yolo(weights, arm)""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = "promote_fold0_replication" if delta >= CFG.GATE_DELTA else "reject_param_fold2"')
    src = replace_once(src, "    if last.exists():  # 診断 (判定不使用): 全データ学習の best/last 選択方針の根拠",
                       "    if last.exists() and weights != CFG.PROD_YOLO:  # 診断: 学習 arm のみ (v102 は該当なし)")
    return src


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = _cell("markdown", TITLE)
    cells[1]["source"] = patch_config(_source(cells[1])).splitlines(keepends=True)
    # cell2 (COCO/fold/前処理)・cell3 (dataset構築+train_arm 定義のみ・未使用)・
    # cell4 (refiner 定義)・cell5 (評価関数) は無改変
    cells[6]["source"] = patch_run(_source(cells[6])).splitlines(keepends=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {OUT.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
