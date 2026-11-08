"""V114 Colab ノートのビルダー: SAHI 型タイリング推論 (RESEARCH_LOG §2.92)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・
評価関数) を無改変で再利用し、YOLO.predict をラップして重複タイル推論 +
global canvas 併合を差し込む (v109 の ms_union と同型の最小互換 namespace)。
全 arm 学習なし・推論のみ (§2.21 の 1152² 2×2 タイルより細かい拡大領域)。

- `sahi640o128`: tile 640 / overlap 128 → 4×4=16 タイル (原寸比 2.00× / prod 比 3.20×)
- `sahi896o192`: tile 896 / overlap 192 → 3×3=9 タイル (原寸比 1.43× / prod 比 2.29×)
- 併合: conf 降順 greedy、bbox 短絡付き mask-IoU > 0.30 で重複除去
- per-arm 既定は prod リテラル (v109 の CFG 持ち越し欠陥の教訓)

置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v114_sahi_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v114_sahi_colab.ipynb"


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


TITLE = """# v114 — SAHI 型タイリング推論 (Colab 版)

RESEARCH_LOG §2.92 の事前登録実験。2048×2048 画像を prod 検出器が全画面
imgsz=1280 (原寸比 0.625×) で見ている**解像度飢餓仮説**に対し、§2.21
(1152² 2×2 タイル、原寸比 1.11×) で未試行だった 1280 未満小タイルの
拡大推論領域を検証する。全 arm 学習なし・推論のみ。

| arm | タイル (imgsz=1280) | 原寸比 | prod 1280 比 |
|---|---|---|---|
| `sahi640o128` | 640²/overlap128 → 4×4=16枚 | 2.00× | 3.20× |
| `sahi896o192` | 896²/overlap192 → 3×3=9枚 | 1.43× | 2.29× |

併合: タイル局所 mask → bbox 短絡付き conf 降順 greedy で
mask-IoU > 0.30 重複除去 → 生存提案のみ global canvas へ配置。
残りは prod 評価経路そのまま (refine/ownership/PQ 無改変)。

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`。

**caveat**: §2.21 と同様、細長フィラメントのタイル境界切断および tile-FP
増加リスクがある — 負方向なら切断・tile-FP 説と整合 (併合 IoU=0.30 で
部分/全体の重複は吸収する設計)。

**ゲート (§2.92、結果取得前に固定)**: Δ@0.26 >= **+0.010** →
fold0 honest 複製へ。fold2 単独では提出を変更しない。

GPU 必須 (~2h: 9-16 タイル ×235枚で推論が重い)。上から順に全セル実行。
結果は `artifacts/v114_sahi_colab_manifest.json` に逐次保存。
"""

V114_WRAPPERS = '''

# --- v114: SAHI 重複タイル推論 (§2.92 固定)。prod 既定では元関数経路。
from types import SimpleNamespace

def _tile_starts(length, tile, overlap):
    stride = tile - overlap
    last = max(length - tile, 0)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts

def _tight_crop(mask, y_off=0, x_off=0):
    ys = np.flatnonzero(mask.any(axis=1))
    if not len(ys):
        return (0, 0, 0, 0), mask[:0, :0]
    xs = np.flatnonzero(mask.any(axis=0))
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    return (y_off + y0, x_off + x0, y_off + y1, x_off + x1), mask[y0:y1, x0:x1]

def _merge_sahi_boxes(crops, confs, boxes, iou_thr=0.30):
    """bbox 非交差を O(1) で短絡し、交差領域スライスのみで厳密 mask-IoU を計算。"""
    order = sorted(range(len(crops)), key=lambda i: -confs[i])
    areas = [int(c.sum()) for c in crops]
    kept = []
    for i in order:
        yi0, xi0, yi1, xi1 = boxes[i]
        ai = areas[i]
        dup = False
        if ai > 0:
            for j in kept:
                aj = areas[j]
                if aj == 0:
                    continue
                yj0, xj0, yj1, xj1 = boxes[j]
                oy0, ox0 = max(yi0, yj0), max(xi0, xj0)
                oy1, ox1 = min(yi1, yj1), min(xi1, xj1)
                if oy0 >= oy1 or ox0 >= ox1:
                    continue
                inter = int(np.count_nonzero(
                    crops[i][oy0 - yi0:oy1 - yi0, ox0 - xi0:ox1 - xi0]
                    & crops[j][oy0 - yj0:oy1 - yj0, ox0 - xj0:ox1 - xj0]))
                union = ai + aj - inter
                if union > 0 and inter / union > iou_thr:
                    dup = True
                    break
        if not dup:
            kept.append(i)
    return kept

def _merge_sahi(masks, confs, iou_thr=0.30):
    """conf 降順 greedy mask-IoU 重複除去 (global canvas 互換 API)。"""
    tight = [_tight_crop(m) for m in masks]
    boxes = [b for b, _ in tight]
    crops = [c for _, c in tight]
    kept = _merge_sahi_boxes(crops, confs, boxes, iou_thr=iou_thr)
    return [masks[i] for i in kept], np.asarray([confs[i] for i in kept], np.float32)

_orig_YOLO = YOLO
def YOLO(path, *args, **kwargs):  # noqa: F811
    m = _orig_YOLO(path, *args, **kwargs)
    _predict = m.predict
    def predict(source, **kw):
        if CFG.SAHI_TILE is None:
            return _predict(source, **kw)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        crops, boxes, confs = [], [], []
        for y0 in _tile_starts(height, CFG.SAHI_TILE, CFG.SAHI_OVERLAP):
            for x0 in _tile_starts(width, CFG.SAHI_TILE, CFG.SAHI_OVERLAP):
                tile = image[y0:y0 + CFG.SAHI_TILE, x0:x0 + CFG.SAHI_TILE]
                result = _predict(tile, **kw)[0]
                tmasks, tconf = result_masks(result)
                for i, tm in enumerate(tmasks):
                    box, crop = _tight_crop(tm, y0, x0)
                    boxes.append(box)
                    crops.append(crop)
                    confs.append(float(tconf[i]))
        kept = _merge_sahi_boxes(crops, confs, boxes) if crops else []
        merged_m = np.zeros((len(kept), height, width), dtype=bool)
        for out_idx, k in enumerate(kept):
            by0, bx0, by1, bx1 = boxes[k]
            if by1 > by0 and bx1 > bx0:
                merged_m[out_idx, by0:by1, bx0:bx1] = crops[k]
        merged_c = np.asarray([confs[k] for k in kept], np.float32)
        return [SimpleNamespace(orig_shape=(height, width), orig_img=image,
                                masks=SimpleNamespace(data=torch.from_numpy(merged_m)),
                                boxes=SimpleNamespace(conf=torch.from_numpy(merged_c)))]
    m.predict = predict
    return m
'''


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('sahi640o128', 'sahi896o192')\n"
                       "    TRAIN_FOLDS = (0, 1)  # 記録用 (v114 は学習しない)\n"
                       "    SAHI_TILE = None      # prod 既定 (単一画像推論)\n"
                       "    SAHI_OVERLAP = 0      # prod 既定\n"
                       "    PARAMS_OF_ARM = {\n"
                       "        'sahi640o128': {'sahi_tile': 640, 'sahi_overlap': 128},\n"
                       "        'sahi896o192': {'sahi_tile': 896, 'sahi_overlap': 192}}")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v114_sahi"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v114_sahi"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v114")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v114_sahi_colab_manifest.json"')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # §2.92: fold0 複製ゲート")
    return src


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       "# [6/6] 実行: prod baseline → SAHI タイル arm (全て学習なし、manifest 逐次保存、判定は §2.92)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v114_sahi"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.92 (SAHI tiled inference resolution hypothesis)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "params_of_arm": {k: dict(v) for k, v in CFG.PARAMS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.92)"')
    src = replace_once(src, 'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}',
                       'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}\n'
                       + V114_WRAPPERS)
    src = replace_once(src, 'if "prod" not in manifest["results"]:',
                       'CFG.SAHI_TILE = None; CFG.SAHI_OVERLAP = 0\n'
                       'if "prod" not in manifest["results"]:')
    src = replace_once(
        src,
        """    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""",
        """    p = {"sahi_tile": None, "sahi_overlap": 0}  # prod リテラル (CFG 持ち越さない)
    p.update(CFG.PARAMS_OF_ARM.get(arm, {}))
    CFG.SAHI_TILE = p["sahi_tile"]; CFG.SAHI_OVERLAP = p["sahi_overlap"]
    weights, stats = CFG.PROD_YOLO, {"infer_only": True}
    manifest["datasets"][arm] = stats | {"params": p}
    by_conf = evaluate_yolo(weights, arm)""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = ("promote_fold0_replication" if delta >= CFG.GATE_DELTA\n'
                       '               else "reject_v114_fold2")')
    src = replace_once(src, "    if last.exists():  # 診断 (判定不使用): 全データ学習の best/last 選択方針の根拠",
                       "    if last.exists() and arm not in CFG.ARMS:  # 診断 (v114 は全て prod 重み・該当なし)")
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
