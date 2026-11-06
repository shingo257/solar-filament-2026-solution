"""V106 Colab ノートのビルダー: existence 救済 + 未 sweep 後処理パラメータ (RESEARCH_LOG §2.84)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・
評価関数) を無改変で再利用し、CFG の arms/paths、実行セルの per-arm
パラメータ差替え + 同等 override の差し込みだけを行う (v104 §2.82 と
同一のパッチ構造)。全 arm 学習なし・推論のみ。

- `rescue_mean050` / `rescue_mean065`: det_conf<0.26 かつ refiner 平均
  確率>=τ の提案を conf=0.2601 にブースト (primary 採点にのみ入る)
- `demote_mean030`: det_conf>=0.26 かつ refiner 平均確率<0.30 の提案を
  conf=0 に (全 sweep から除去)
- `refiner_th040`/`refiner_th050`/`drop080`/`drop097`/`minarea10`/`minarea40`:
  v102 で sweep されなかった残りの後処理パラメータ
- `combo_960_nms35`: imgsz960 重み + predict_iou 0.35 (sub-gate 信号の加算)

凍結関数の default-arg 束縛 (drop_contained_indices の threshold) は
CFG 経由の同等 override で回避し、rescue/demote は result_masks を
ラップして refiner 平均確率で conf を再スコアする (採点経路は無改変)。
置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v106_rescue_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v106_rescue_colab.ipynb"


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


TITLE = """# v106 — existence 救済 + 未 sweep 後処理パラメータ (Colab 版)

RESEARCH_LOG §2.84 の事前登録実験。primary conf の FN の 57% が低conf
生提案でカバーされている実測から、**refiner 平均確率を独立した実在性
信号として低conf提案を選別**する機構を試す。併せて v102 で未 sweep
だった後処理パラメータ (refiner 閾値・包含閾値・最小面積) を網羅する。

| arm | 変更 (prod → arm) |
|---|---|
| `rescue_mean050` | det_conf<0.26 かつ refiner 平均>=0.50 → conf=0.2601 ブースト |
| `rescue_mean065` | 同上 τ=0.65 |
| `demote_mean030` | det_conf>=0.26 かつ refiner 平均<0.30 → conf=0 除去 |
| `refiner_th040` / `refiner_th050` | REFINER_THRESHOLD 0.45 → 0.40 / 0.50 |
| `drop080` / `drop097` | DROP_THRESHOLD 0.90 → 0.80 / 0.97 |
| `minarea10` / `minarea40` | MIN_AREA 20 → 10 / 40 |
| `combo_960_nms35` | imgsz960 重み + predict_iou 0.35 |

**refiner スコア固定式 (§2.84)**: prod crop 窓 (context 1.65) 内の提案
画素上の sigmoid 平均確率。

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`・
`runs/v103_recipe/yolo_imgsz960_fold2/weights/best.pt`。

**ゲート (§2.84、結果取得前に固定)**: arm 毎に
Δ = PQ(arm,0.26) − PQ(prod,0.26) が >= +0.010 なら fold0 honest 複製へ。
fold2 単独では提出を変更しない (探索 arm が多いため単発 fold2 プラスは
絶対に昇格しない)。

GPU 必須 (~3-4h 見込み、全 arm 推論のみ)。上から順に全セル実行。
結果は `artifacts/v106_rescue_colab_manifest.json` に逐次保存。
"""

RESCUE_AND_WRAPPERS = '''

# --- v106: 凍結 eval の default-arg 束縛/スコア挿入を CFG から読む同等
# override (prod 既定値では元関数と同一経路)。
@torch.no_grad()
def _refiner_mean_prob(image, proposal):
    """prod refine と同じ crop 窓・前処理の sigmoid 確率マップに対し、
    crop 窓内の提案画素上の平均確率を返す (§2.84 固定式)。"""
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    pcrop = proposal[y0:y1, x0:x1]
    return float(probability[pcrop].mean()) if pcrop.any() else 0.0

_orig_result_masks = result_masks
def result_masks(result):  # noqa: F811
    masks, conf = _orig_result_masks(result)
    if (CFG.RESCUE_TAU is None and CFG.DEMOTE_TAU is None) or not conf.size:
        return masks, conf
    for i, m in enumerate(masks):
        score = _refiner_mean_prob(result.orig_img, m)
        c = float(conf[i])
        if c < CFG.PRIMARY_CONF and CFG.RESCUE_TAU is not None and score >= CFG.RESCUE_TAU:
            conf[i] = CFG.RESCUE_BOOST
        elif c >= CFG.PRIMARY_CONF and CFG.DEMOTE_TAU is not None and score < CFG.DEMOTE_TAU:
            conf[i] = 0.0
    return masks, conf

_orig_apply_ownership = apply_ownership
def apply_ownership(masks):  # noqa: F811
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    occupied = np.zeros(masks[0].shape, dtype=bool)
    kept = []
    for index in sorted(drop_contained_indices(masks, CFG.DROP_THRESHOLD), key=lambda i: areas[i]):
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < CFG.MIN_AREA:
            continue
        kept.append(exclusive); occupied |= exclusive
    return kept

'''


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('rescue_mean050', 'rescue_mean065', 'demote_mean030',\n"
                       "            'refiner_th040', 'refiner_th050', 'drop080', 'drop097',\n"
                       "            'minarea10', 'minarea40', 'combo_960_nms35')\n"
                       "    TRAIN_FOLDS = (0, 1)  # 記録用 (v106 は学習しない)\n"
                       "    RESCUE_TAU = None      # prod 既定 (救済なし)\n"
                       "    DEMOTE_TAU = None      # prod 既定 (demote なし)\n"
                       "    RESCUE_BOOST = 0.2601  # primary 直上 (0.32 sweep には出ない)\n"
                       "    PARAMS_OF_ARM = {\n"
                       "        'rescue_mean050': {'rescue_tau': 0.50},\n"
                       "        'rescue_mean065': {'rescue_tau': 0.65},\n"
                       "        'demote_mean030': {'demote_tau': 0.30},\n"
                       "        'refiner_th040': {'refiner_threshold': 0.40},\n"
                       "        'refiner_th050': {'refiner_threshold': 0.50},\n"
                       "        'drop080': {'drop_threshold': 0.80},\n"
                       "        'drop097': {'drop_threshold': 0.97},\n"
                       "        'minarea10': {'min_area': 10},\n"
                       "        'minarea40': {'min_area': 40},\n"
                       "        'combo_960_nms35': {'predict_iou': 0.35}}\n"
                       "    WEIGHTS_OF_ARM = {'combo_960_nms35':\n"
                       "                      'runs/v103_recipe/yolo_imgsz960_fold2/weights/best.pt'}")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v106_rescue"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v106_rescue"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v106")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v106_rescue_colab_manifest.json"')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # §2.84: LB 判別可能幅をゲートに使う")
    return src


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       "# [6/6] 実行: prod baseline → rescue/demote/後処理 arm (全て学習なし、manifest 逐次保存、判定は §2.84 の per-arm ゲート)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v106_rescue_postproc"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.84 (existence rescue + postproc params)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "rescue_boost": CFG.RESCUE_BOOST,\n'
                       '            "params_of_arm": {k: dict(v) for k, v in CFG.PARAMS_OF_ARM.items()},\n'
                       '            "weights_of_arm": {k: str(v) for k, v in CFG.WEIGHTS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.84)"')
    src = replace_once(src, 'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}',
                       'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}\n'
                       + RESCUE_AND_WRAPPERS)
    src = replace_once(
        src,
        """    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""",
        """    p = {"rescue_tau": None, "demote_tau": None, "refiner_threshold": 0.45,
         "drop_threshold": 0.90, "min_area": 20, "predict_iou": 0.55}
    p.update(CFG.PARAMS_OF_ARM.get(arm, {}))
    CFG.RESCUE_TAU = p["rescue_tau"]; CFG.DEMOTE_TAU = p["demote_tau"]
    CFG.REFINER_THRESHOLD = p["refiner_threshold"]; CFG.DROP_THRESHOLD = p["drop_threshold"]
    CFG.MIN_AREA = p["min_area"]; CFG.PREDICT_IOU = p["predict_iou"]
    if arm in CFG.WEIGHTS_OF_ARM:
        weights = CFG.DRIVE_ROOT / CFG.WEIGHTS_OF_ARM[arm]
        stats = {"infer_only": True, "weights_override": CFG.WEIGHTS_OF_ARM[arm]}
    else:
        weights, stats = CFG.PROD_YOLO, {"infer_only": True}
    manifest["datasets"][arm] = stats | {"params": p}
    by_conf = evaluate_yolo(weights, arm)""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = "promote_fold0_replication" if delta >= CFG.GATE_DELTA else "reject_v106_fold2"')
    src = replace_once(src, "    if last.exists():  # 診断 (判定不使用): 全データ学習の best/last 選択方針の根拠",
                       "    if last.exists() and weights != CFG.PROD_YOLO and arm not in CFG.WEIGHTS_OF_ARM:  # 診断 (v106 は該当なし)")
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
