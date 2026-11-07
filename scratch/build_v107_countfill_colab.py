"""V107 Colab ノートのビルダー: rank/count 補正 + 別系統 demote + multi-context refine (RESEARCH_LOG §2.85)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・
評価関数) を無改変で再利用し、CFG の arms/paths、実行セルの per-arm
パラメータ差替え + 同等 override の差し込みだけを行う (v104/v106 と
同一のパッチ構造)。全 arm 学習なし・推論のみ。

- `countfill_k7` / `countfill_k9`: 画像毎に conf<0.26 の上位提案を
  合計 K 個まで 0.2601 へブースト (rank ベースの under-count 補正)
- `demote_ratio030`: conf>=0.26 かつ refined/proposal<0.30 → conf=0
- `demote_aspect15`: conf>=0.26 かつ bbox aspect<1.5 → conf=0
- `ctx_union` / `ctx_intersect`: refine を ctx{1.45,1.85} の 2窓で
  行い binary union/intersect (マスク品質向き)

rescue/demote/countfill は result_masks をラップして conf を再スコア
(採点経路は無改変)。ctx ensemble は refine_proposal をラップして
mask_square_box(context=c) を明示指定 (prod 既定では元関数と同一経路)。
置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v107_countfill_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v107_countfill_colab.ipynb"


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


TITLE = """# v107 — rank/count 補正 + 別系統 demote + multi-context refine (Colab 版)

RESEARCH_LOG §2.85 の事前登録実験。v106 で「refiner 平均確率は existence
選別として機能しない」ことが確定したため、スコア不要の別機構を試す:
**rank ベースの under-count 補正**(per-record GT ≈7.1 vs preds 6.38)、
**別系統 demote**(refined/proposal 比・mask aspect)、**refine の
context ensemble**(2窓 union/intersect)。

| arm | 変更 (prod → arm) |
|---|---|
| `countfill_k7` / `countfill_k9` | conf<0.26 の上位を合計 K/img まで 0.2601 ブースト |
| `demote_ratio030` | conf>=0.26 かつ refined/proposal<0.30 → conf=0 |
| `demote_aspect15` | conf>=0.26 かつ bbox aspect<1.5 → conf=0 |
| `ctx_union` / `ctx_intersect` | refine を ctx{1.45,1.85} 2窓で union/intersect |

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`。

**ゲート (§2.85、結果取得前に固定)**: arm 毎に
Δ = PQ(arm,0.26) − PQ(prod,0.26) が >= +0.010 なら fold0 honest 複製へ。
fold2 単独では提出を変更しない。

GPU 必須 (~2.5h 見込み、全 arm 推論のみ)。上から順に全セル実行。
結果は `artifacts/v107_countfill_colab_manifest.json` に逐次保存。
"""

V107_WRAPPERS = '''

# --- v107: 凍結 eval のスコア挿入/context 差替えを CFG から読む同等
# override (prod 既定値では元関数と同一経路)。
_orig_result_masks = result_masks
def result_masks(result):  # noqa: F811
    masks, conf = _orig_result_masks(result)
    if not conf.size or (CFG.COUNTFILL_K is None and CFG.DEMOTE_MODE is None):
        return masks, conf
    if CFG.COUNTFILL_K is not None:  # rank 補正: primary 未満の上位を合計 K までブースト
        n_primary = int((conf >= CFG.PRIMARY_CONF).sum())
        if n_primary < CFG.COUNTFILL_K:
            low = np.argsort(-conf, kind="stable")
            need = CFG.COUNTFILL_K - n_primary
            for i in low:
                if need <= 0:
                    break
                if conf[i] < CFG.PRIMARY_CONF:
                    conf[i] = CFG.RESCUE_BOOST; need -= 1
    if CFG.DEMOTE_MODE is not None:
        for i, m in enumerate(masks):
            if conf[i] < CFG.PRIMARY_CONF:
                continue
            if CFG.DEMOTE_MODE == "ratio":
                prop = max(float(m.sum()), 1.0)
                score = float(refine_proposal(result.orig_img, m).sum()) / prop
            else:  # "aspect": bbox の細長さ (filament 形状事前・反証テスト)
                x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
                score = max(w, h) / max(min(w, h), 1)
            if score < CFG.DEMOTE_TAU:
                conf[i] = 0.0
    return masks, conf

_orig_refine_proposal = refine_proposal
def _refine_at_context(image, proposal, context):
    """refine_proposal と同一手順で context だけを変えた精製 (§2.85 固定)。"""
    x0, y0, x1, y1 = mask_square_box(proposal, context=context)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    with torch.no_grad():
        probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal

def refine_proposal(image, proposal):  # noqa: F811
    if CFG.CTX_ENSEMBLE is None:
        return _orig_refine_proposal(image, proposal)
    masks = [_refine_at_context(image, proposal, c) for c in CFG.CTX_ENSEMBLE[0]]
    return (np.logical_or.reduce(masks) if CFG.CTX_ENSEMBLE[1] == "union"
            else np.logical_and.reduce(masks))

'''


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('countfill_k7', 'countfill_k9', 'demote_ratio030',\n"
                       "            'demote_aspect15', 'ctx_union', 'ctx_intersect')\n"
                       "    TRAIN_FOLDS = (0, 1)  # 記録用 (v107 は学習しない)\n"
                       "    COUNTFILL_K = None     # prod 既定 (rank 補正なし)\n"
                       "    DEMOTE_MODE = None     # prod 既定 (demote なし)\n"
                       "    DEMOTE_TAU = None\n"
                       "    RESCUE_BOOST = 0.2601  # primary 直上\n"
                       "    CTX_ENSEMBLE = None    # prod 既定 (単一 ctx 1.65)\n"
                       "    PARAMS_OF_ARM = {\n"
                       "        'countfill_k7': {'countfill_k': 7},\n"
                       "        'countfill_k9': {'countfill_k': 9},\n"
                       "        'demote_ratio030': {'demote_mode': 'ratio', 'demote_tau': 0.30},\n"
                       "        'demote_aspect15': {'demote_mode': 'aspect', 'demote_tau': 1.5},\n"
                       "        'ctx_union': {'ctx_ensemble': [[1.45, 1.85], 'union']},\n"
                       "        'ctx_intersect': {'ctx_ensemble': [[1.45, 1.85], 'intersect']}}")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v107_countfill"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v107_countfill"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v107")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v107_countfill_colab_manifest.json"')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # §2.85: LB 判別可能幅をゲートに使う")
    return src


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       "# [6/6] 実行: prod baseline → countfill/demote/ctx-ensemble arm (全て学習なし、manifest 逐次保存、判定は §2.85 の per-arm ゲート)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v107_countfill_ctx"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.85 (rank/count correction + alt demote + multi-context refine)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "rescue_boost": CFG.RESCUE_BOOST,\n'
                       '            "params_of_arm": {k: dict(v) for k, v in CFG.PARAMS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.85)"')
    src = replace_once(src, 'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}',
                       'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}\n'
                       + V107_WRAPPERS)
    src = replace_once(
        src,
        """    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""",
        """    p = {"countfill_k": None, "demote_mode": None, "demote_tau": None,
         "ctx_ensemble": None}
    p.update(CFG.PARAMS_OF_ARM.get(arm, {}))
    CFG.COUNTFILL_K = p["countfill_k"]; CFG.DEMOTE_MODE = p["demote_mode"]
    CFG.DEMOTE_TAU = p["demote_tau"]; CFG.CTX_ENSEMBLE = p["ctx_ensemble"]
    weights, stats = CFG.PROD_YOLO, {"infer_only": True}
    manifest["datasets"][arm] = stats | {"params": p}
    by_conf = evaluate_yolo(weights, arm)""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = "promote_fold0_replication" if delta >= CFG.GATE_DELTA else "reject_v107_fold2"')
    src = replace_once(src, "    if last.exists():  # 診断 (判定不使用): 全データ学習の best/last 選択方針の根拠",
                       "    if last.exists() and arm not in CFG.ARMS:  # 診断 (v107 は全て prod 重み・該当なし)")
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
