"""V109 Colab ノートのビルダー: refiner TTA + ms-union + crop/ctx sweep (RESEARCH_LOG §2.87)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・
評価関数) を無改変で再利用し、CFG の arms/paths、実行セルの per-arm
パラメータ差替え + 同等 override の差し込みだけを行う (v107 と同一の
パッチ構造)。全 arm 学習なし・推論のみ。

- `refine_tta4`: refiner 確率マップを二面体4変換で平均 (未試行の refiner TTA)
- `ms_union960`: 同一 prod 重みを imgsz 1280+960 で2回推論し提案を併合
- `crop384` / `crop480`: CROP_SIZE 320→{384,480} (未 sweep パラメータ)
- `ctx145` / `ctx185`: CROP_CONTEXT 1.65→{1.45,1.85} 単値
- `conf_fine`: CONF_SWEEP 細粒度診断 {0.15..0.32 by 0.02} (verdict=diagnostic)

TTA/ctx は refine_proposal を同等手順でラップ、ms_union は YOLO
コンストラクタをラップして predict を2スケール併合 (最小互換 namespace)、
crop/conf_fine は CFG 実行時参照パラメータの差替えのみ。採点経路は無改変。
置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v109_tta_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v109_tta_colab.ipynb"


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


TITLE = """# v109 — refiner TTA + ms-union + crop/ctx sweep (Colab 版)

RESEARCH_LOG §2.87 の事前登録実験。v108 で全実験系が閉じた後の
棚卸しで見つかった**推論のみの未試行機構**を試す。

| arm | 変更 (prod → arm) |
|---|---|
| `refine_tta4` | refiner 確率マップを二面体4変換 (id/hflip/vflip/hvflip) で平均 |
| `ms_union960` | 同一 prod 重みを imgsz 1280+960 の2回推論で提案併合 |
| `crop384` / `crop480` | CROP_SIZE 320→{384,480} |
| `ctx145` / `ctx185` | CROP_CONTEXT 1.65→{1.45,1.85} 単値 |
| `conf_fine` | CONF_SWEEP {0.15,0.2,0.22,0.24,0.26,0.28,0.30,0.32} 診断 |

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`。

**ゲート (§2.87、結果取得前に固定)**: Δ@0.26 >= **+0.010** →
fold0 honest 複製へ。fold2 単独では提出を変更しない。
`conf_fine` は診断 arm (verdict=diagnostic_conf_grid)。細粒度点で
高値が出ても fold0 paired 複製を経ない限り採用しない。

GPU 必須 (~1.5h 見込み、全 arm 推論のみ)。上から順に全セル実行。
結果は `artifacts/v109_tta_colab_manifest.json` に逐次保存。
"""

V109_WRAPPERS = '''

# --- v109: refine TTA / ctx 単値 / ms-union の CFG 駆動 override
# (prod 既定では元関数と同一経路)。
from types import SimpleNamespace

_orig_refine_proposal = refine_proposal
def _refine_impl(image, proposal, context):
    """refine_proposal と同一手順で context/TTA だけを変えた精製 (§2.87 固定)。"""
    x0, y0, x1, y1 = mask_square_box(proposal, context=context)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    flips = [None] if not CFG.REFINE_TTA else [None, "h", "v", "hv"]
    dims = {"h": [3], "v": [2], "hv": [2, 3]}
    probs = []
    with torch.no_grad():
        for f in flips:
            t = tensor if f is None else torch.flip(tensor, dims=dims[f])
            p = refiner(t).sigmoid()
            if f is not None:
                p = torch.flip(p, dims=dims[f])
            probs.append(p)
    probability = torch.stack(probs).mean(0)[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal

def refine_proposal(image, proposal):  # noqa: F811
    if CFG.CTX_VALUE is None and not CFG.REFINE_TTA:
        return _orig_refine_proposal(image, proposal)
    return _refine_impl(image, proposal, CFG.CTX_VALUE or CFG.CROP_CONTEXT)

_orig_YOLO = YOLO
def YOLO(path, *args, **kwargs):  # noqa: F811
    m = _orig_YOLO(path, *args, **kwargs)
    if CFG.MS_UNION_IMGSZ is None:
        return m
    _predict = m.predict
    def predict(source, **kw):
        r1 = _predict(source, **kw)[0]
        kw2 = dict(kw); kw2["imgsz"] = CFG.MS_UNION_IMGSZ
        r2 = _predict(source, **kw2)[0]
        c1 = r1.boxes.conf if r1.boxes is not None else None
        c2 = r2.boxes.conf if r2.boxes is not None else None
        if c2 is None or not len(c2):
            return [r1]
        if c1 is None or not len(c1):
            merged_m, merged_c = r2.masks.data, c2
        else:
            merged_m = torch.cat([r1.masks.data, r2.masks.data.to(r1.masks.data.device)])
            merged_c = torch.cat([c1, c2.to(c1.device)])
        return [SimpleNamespace(orig_shape=r1.orig_shape, orig_img=r1.orig_img,
                                masks=SimpleNamespace(data=merged_m),
                                boxes=SimpleNamespace(conf=merged_c))]
    m.predict = predict
    return m

'''


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('refine_tta4', 'ms_union960', 'crop384', 'crop480',\n"
                       "            'ctx145', 'ctx185', 'conf_fine')\n"
                       "    TRAIN_FOLDS = (0, 1)  # 記録用 (v109 は学習しない)\n"
                       "    MS_UNION_IMGSZ = None  # prod 既定 (単一スケール)\n"
                       "    REFINE_TTA = 0         # prod 既定 (TTA なし)\n"
                       "    CTX_VALUE = None       # prod 既定 (context=1.65)\n"
                       "    PARAMS_OF_ARM = {\n"
                       "        'refine_tta4': {'refine_tta': 4},\n"
                       "        'ms_union960': {'ms_union_imgsz': 960},\n"
                       "        'crop384': {'crop_size': 384},\n"
                       "        'crop480': {'crop_size': 480},\n"
                       "        'ctx145': {'ctx_value': 1.45},\n"
                       "        'ctx185': {'ctx_value': 1.85},\n"
                       "        'conf_fine': {'conf_sweep': [0.15, 0.2, 0.22, 0.24,\n"
                       "                                     0.26, 0.28, 0.30, 0.32]}}")
    # 注: conf_sweep は tuple ではなく list — contract が JSON round-trip 後も
    # 一致するよう (manifest resume 互換)。run 側で tuple() 化済み。
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v109_tta"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v109_tta"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v109")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v109_tta_colab_manifest.json"')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # §2.87: fold0 複製ゲート")
    return src


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       "# [6/6] 実行: prod baseline → TTA/ms-union/crop/ctx/conf-fine arm (全て学習なし、manifest 逐次保存、判定は §2.87 の per-arm ゲート)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v109_tta_msunion_crop"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.87 (refiner TTA + ms-union + crop/ctx sweep + fine conf)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "params_of_arm": {k: dict(v) for k, v in CFG.PARAMS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.87)"')
    src = replace_once(src, 'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}',
                       'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}\n'
                       + V109_WRAPPERS)
    src = replace_once(
        src,
        """    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""",
        """    p = {"ms_union_imgsz": None, "refine_tta": 0, "ctx_value": None,
         "crop_size": 320, "conf_sweep": (0.15, 0.2, 0.26, 0.32)}  # prod リテラル
    # (前 arm で CFG を書き換えた場合でも持ち越さない — 既定は常に prod 値)
    p.update(CFG.PARAMS_OF_ARM.get(arm, {}))
    CFG.MS_UNION_IMGSZ = p["ms_union_imgsz"]; CFG.REFINE_TTA = p["refine_tta"]
    CFG.CTX_VALUE = p["ctx_value"]; CFG.CROP_SIZE = p["crop_size"]
    CFG.CONF_SWEEP = tuple(p["conf_sweep"])
    weights, stats = CFG.PROD_YOLO, {"infer_only": True}
    manifest["datasets"][arm] = stats | {"params": p}
    by_conf = evaluate_yolo(weights, arm)""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = ("diagnostic_conf_grid" if arm == "conf_fine" else\n'
                       '               "promote_fold0_replication" if delta >= CFG.GATE_DELTA else "reject_v109_fold2")')
    src = replace_once(src, "    if last.exists():  # 診断 (判定不使用): 全データ学習の best/last 選択方針の根拠",
                       "    if last.exists() and arm not in CFG.ARMS:  # 診断 (v109 は全て prod 重み・該当なし)")
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
