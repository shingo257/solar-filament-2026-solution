"""V104 Colab ノートのビルダー: weight soup + ownership 順序 + refine 2回目 (§2.82)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・
評価関数) を無改変で再利用する。v102/v103 と同じく学習は一切行わない —
全 arm が prod fold2 評価経路で、未試行の残存機構だけを変える:

- `soup_tail3` / `soup_tail7`: prod fold2 の周期 checkpoint 末尾平均 (SWA)
- `soup_res3`: prod best + v101 imgsz{1536,2048} best の3系統重み平均
- `own_desc`: greedy ownership を area 昇順→降順
- `refine_twice`: refiner 出力を proposal として再 refine

soup ckpt は実行セルで組み立て ('ema' があれば評価重みとして拾い、
soup 後は 'ema' を外して predict が soup 'model' を使う形で保存)、
members・sha256 を manifest に記録する。own_desc/refine_twice は
凍結 eval セル無改変のまま実行セルの wrapper で実現 (prod 既定では
元関数と同一経路)。置換は「ちょうど 1 回」一致を assert するので
元ノートが変わると失敗する。

    python scratch/build_v104_soup_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v104_soup_colab.ipynb"


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


TITLE = """# v104 — weight soup + ownership 順序 + refine 2回目 (Colab 版)

RESEARCH_LOG §2.82 の事前登録実験。予測空間アンサンブル/重複除去は
v97–v99 で閉じたが、**重み空間の平均 (weight soup / tail SWA) は
未試行**。prod の best_epoch=70 (終端・未収束) なら末尾 checkpoint の
平均が有効な可能性。全 arm 学習なし (推論のみ、~2.5h)。

| arm | 内容 |
|---|---|
| `soup_tail3` | prod epoch60/65/best の3重み平均 |
| `soup_tail7` | prod epoch40..65+best の7重み平均 |
| `soup_res3` | prod best + v101 imgsz1536 best + imgsz2048 best 平均 |
| `own_desc` | greedy ownership を area 昇順→降順 (prod は昇順) |
| `refine_twice` | refiner 出力を proposal として再 refine (収束テスト) |

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/` (epoch*.pt + best.pt)・
`runs/v101_imgsz/yolo_imgsz{1536,2048}_fold2/weights/best.pt`。

**ゲート (§2.82、結果取得前に固定)**: arm 毎に
Δ = PQ(arm,0.26) − PQ(prod,0.26) が >= +0.010 なら fold0 honest 複製へ。
fold2 単独では提出を変更しない。

GPU 必須。上から順に全セル実行。結果は
`artifacts/v104_soup_colab_manifest.json` に逐次保存、soup ckpt は
`artifacts/v104_*_soup.pt` に保存される。
"""


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('soup_tail3', 'soup_tail7', 'soup_res3', 'own_desc', 'refine_twice')\n"
                       "    TRAIN_FOLDS = (0, 1)  # 記録用 (v104 は学習しない)\n"
                       "    OWNERSHIP_ORDER = 'asc'   # prod 既定\n"
                       "    REFINE_TWICE = False      # prod 既定\n"
                       "    SOUP_OF_ARM = {\n"
                       "        'soup_tail3': ['runs/yolo_fold_2/weights/epoch60.pt',\n"
                       "                       'runs/yolo_fold_2/weights/epoch65.pt',\n"
                       "                       'runs/yolo_fold_2/weights/best.pt'],\n"
                       "        'soup_tail7': ['runs/yolo_fold_2/weights/epoch40.pt',\n"
                       "                       'runs/yolo_fold_2/weights/epoch45.pt',\n"
                       "                       'runs/yolo_fold_2/weights/epoch50.pt',\n"
                       "                       'runs/yolo_fold_2/weights/epoch55.pt',\n"
                       "                       'runs/yolo_fold_2/weights/epoch60.pt',\n"
                       "                       'runs/yolo_fold_2/weights/epoch65.pt',\n"
                       "                       'runs/yolo_fold_2/weights/best.pt'],\n"
                       "        'soup_res3': ['runs/yolo_fold_2/weights/best.pt',\n"
                       "                      'runs/v101_imgsz/yolo_imgsz1536_fold2/weights/best.pt',\n"
                       "                      'runs/v101_imgsz/yolo_imgsz2048_fold2/weights/best.pt']}\n"
                       "    PARAMS_OF_ARM = {'own_desc': {'ownership_order': 'desc'},\n"
                       "                     'refine_twice': {'refine_twice': True}}")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v104_soup"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v104_soup"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v104")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v104_soup_colab_manifest.json"')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # §2.82: LB 判別可能幅をゲートに使う")
    return src


SOUP_AND_WRAPPERS = '''

# --- v104: soup ckpt 組み立て + 凍結 eval の default-arg/順序を CFG から
# 読む同等 override (prod 既定値では元関数と同一経路)。
def _eval_state_dict(path):
    """ckpt の評価用重みを返す (周期 ckpt は 'ema'、stripped ckpt は 'model')。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt.get("ema") if ckpt.get("ema") is not None else ckpt["model"]
    sd = src.state_dict() if hasattr(src, "state_dict") else src
    return ckpt, {k: v.float() for k, v in sd.items()}, ("ema" if ckpt.get("ema") is not None else "model")

def build_soup(members, out_path):
    """members の state_dict を算術平均した ckpt を保存 ('ema' は外して predict が soup 'model' を使う)。"""
    out_path = Path(out_path)
    if out_path.exists():  # resume
        return out_path
    base, avg, used = _eval_state_dict(members[0])
    member_keys = [used]
    for p in members[1:]:
        _, sd, used = _eval_state_dict(p)
        member_keys.append(used)
        assert set(sd) == set(avg), "soup member の key 不一致"
        for k in avg:
            avg[k] += sd[k]
    for k in avg:
        avg[k] /= len(members)
    # 周期 ckpt は model=None で ema に重み、best.pt は model に重み → ある方へ書き戻し
    target = base.get("model") if base.get("model") is not None else base.get("ema")
    if target is None or not hasattr(target, "load_state_dict"):
        raise ValueError("soup base ckpt に書き戻し可能な module がない")
    target.load_state_dict(avg)
    base["model"] = target
    for k in ("ema", "optimizer", "updates", "train_results", "best_fitness"):
        base.pop(k, None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out_path)
    return out_path

_orig_apply_ownership = apply_ownership
def apply_ownership(masks):  # noqa: F811
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    occupied = np.zeros(masks[0].shape, dtype=bool)
    kept = []
    key = (lambda i: areas[i]) if CFG.OWNERSHIP_ORDER == "asc" else (lambda i: -areas[i])
    for index in sorted(drop_contained_indices(masks), key=key):
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < CFG.MIN_AREA:
            continue
        kept.append(exclusive); occupied |= exclusive
    return kept

_orig_refine_proposal = refine_proposal
def refine_proposal(image, proposal):  # noqa: F811
    first = _orig_refine_proposal(image, proposal)
    if CFG.REFINE_TWICE and first.any():
        return _orig_refine_proposal(image, first)
    return first

'''


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       "# [6/6] 実行: prod baseline → soup/ownership/refine arm (全て学習なし、manifest 逐次保存、判定は §2.82 の per-arm ゲート)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v104_weight_soup"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.82 (weight soup + ownership/refine residuals)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "soup_of_arm": {k: list(v) for k, v in CFG.SOUP_OF_ARM.items()},\n'
                       '            "params_of_arm": {k: dict(v) for k, v in CFG.PARAMS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.82)"')
    src = replace_once(src, 'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}',
                       'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}\n'
                       + SOUP_AND_WRAPPERS)
    src = replace_once(
        src,
        """    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""",
        """    p = {"ownership_order": "asc", "refine_twice": False}
    p.update(CFG.PARAMS_OF_ARM.get(arm, {}))
    CFG.OWNERSHIP_ORDER = p["ownership_order"]; CFG.REFINE_TWICE = p["refine_twice"]
    if arm in CFG.SOUP_OF_ARM:
        weights = build_soup([CFG.DRIVE_ROOT / m for m in CFG.SOUP_OF_ARM[arm]],
                             CFG.ARTIFACT_ROOT / f"{arm}_soup.pt")
        stats = {"infer_only": True, "soup_members": CFG.SOUP_OF_ARM[arm]}
    else:
        weights, stats = CFG.PROD_YOLO, {"infer_only": True}
    manifest["datasets"][arm] = stats | {"params": p}
    by_conf = evaluate_yolo(weights, arm)""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = "promote_fold0_replication" if delta >= CFG.GATE_DELTA else "reject_v104_fold2"')
    src = replace_once(src, "    if last.exists():  # 診断 (判定不使用): 全データ学習の best/last 選択方針の根拠",
                       "    if last.exists() and weights != CFG.PROD_YOLO and arm not in CFG.SOUP_OF_ARM:  # 診断 (v104 は該当なし)")
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
