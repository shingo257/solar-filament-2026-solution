"""V121a Colab ノートのビルダー: crop detector の fold-0 学習 (RESEARCH_LOG §2.99)。

v120 `cons2of3_c26` (Δ@0.45=+0.01155、§2.98 ゲート通過) の fold-0 honest
複製 Phase A: v116 croptrain レシピ verbatim (896² tile×9 crops・全 record・
imgsz=896) を folds(1,2) で学習し fold-0 OOF 用重みを作る。
v116 ビルダーの fold ミラー: FOLD=0、TRAIN_FOLDS=(1,2)、arm は
`croptrain_full` のみ (consensus 評価は v121c が担う、学習重みの
weights_sha256 は manifest `datasets.croptrain` に記録 → v121c が
trusted-pin として読み取る)。

**漏洩修正 (2026-11-10)**: v116 原本の `build_crop_dataset` は `train_files`
(全 707 file) を回しており、eval fold の crop が train split に混入していた
(v116 fold-2: 235 file 混入 → v116/v120 の fold-2 OOF 値は漏洩済み)。
本ノートは `fit_files` (folds(1,2)=471 file のみ) に修正し、出力先を
`v121_croptrain_f0_honest` に分離して汚染 artifact との混同を防ぐ。

    python scratch/build_v121a_croptrain_f0_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

try:  # pytest では scratch パッケージ経由、直接実行では同ディレクトリ経由
    from scratch.build_v114_sahi_colab import V114_WRAPPERS
except ImportError:  # pragma: no cover - 直接 python scratch/xxx.py で実行時
    from build_v114_sahi_colab import V114_WRAPPERS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v121a_croptrain_f0_colab.ipynb"


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


TITLE = """# v121a — crop detector fold-0 学習 (Colab 版、honest 修正版)

RESEARCH_LOG §2.99 の事前登録実験 (v120 cons2of3_c26 +0.01155 gate 通過の
fold-0 複製 Phase A)。v116 croptrain レシピ — folds(1,2) の全
annotator レコードを **896² タイル crop (stride 448、3×3=9/画像)** に
展開し yolo11m-seg を学習 (imgsz=896、他は prod verbatim)。
fold-0 は dataset 構築・学習・val 全て非接触。

> ⚠️ **漏洩修正**: v116 原本は crop train に全 707 file (eval fold 含む) を
> 使っていた。本ノートは `fit_files` (471 file) のみに修正済み —
> 旧 run/dataset (`v121_croptrain_f0`) は汚染のため `_honest` パスに分離。

| arm | 推論 | 目的 |
|---|---|---|
| `croptrain_full` | full-image imgsz=1280 | fold-0 上での crop 重み paired sanity |

生成重み (`runs/v121_croptrain_f0_honest/yolo_croptrain_fold0/weights/best.pt`)
と `datasets.croptrain.weights_sha256` が v121c consensus eval の
`crop` detector 入力・trusted byte-pin になる。

**prod sanity**: fold-0 prod = 0.44413694566994816 (v108 §2.86 確定値) を
paired 再測定して乖離を検知。

GPU 必須 (~6h: dataset ~25min + 学習 ~5h + 評価 ~15min)。manifest 逐次保存で
resume 可能。結果は `artifacts/v121a_croptrain_f0_colab_manifest.json`。
"""

CELL3 = '''# [3/6] v121a crop データセット構築 + YOLO 学習 (v116 §2.94 レシピ verbatim、folds(1,2))

def _crop_label_lines(image_id, y0, x0):
    """crop 窓内の instance を最大 contour polygon 化 (crop 内面積>=MIN_AREA のみ)。"""
    info = images[image_id]
    height, width = int(info["height"]), int(info["width"])
    tile = CFG.CROP_TILE
    lines = []
    for ann in annotations[image_id]:
        if ann.get("iscrowd", 0):
            continue
        m = mask_utils.decode(to_rle(ann, height, width))
        c = m[y0:y0 + tile, x0:x0 + tile]
        if c.shape[0] < tile or c.shape[1] < tile:  # 設計上全覆盖だが端の保険
            pad = np.zeros((tile, tile), dtype=m.dtype)
            pad[:c.shape[0], :c.shape[1]] = c
            c = pad
        if int(c.sum()) < CFG.MIN_AREA:
            continue
        contours, _ = cv2.findContours(np.ascontiguousarray(c, dtype=np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        poly = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        if len(poly) < 3:  # prod の 1個体1polygon 規約 (最大 contour)
            continue
        points = np.clip(poly / tile, 0.0, 1.0)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in points.reshape(-1)))
    return lines

def _write_crop_split(root, split, file_name, image_ids, image):
    written = 0
    for image_id in image_ids:
        for gy, y0 in enumerate(CFG.CROP_STARTS):
            for gx, x0 in enumerate(CFG.CROP_STARTS):
                stem = f"{Path(file_name).stem}__l{image_id}_c{gy}{gx}"
                img_out = root / "images" / split / f"{stem}.jpg"
                lbl_out = root / "labels" / split / f"{stem}.txt"
                img_out.parent.mkdir(parents=True, exist_ok=True)
                lbl_out.parent.mkdir(parents=True, exist_ok=True)
                if not img_out.exists():
                    crop = image[y0:y0 + CFG.CROP_TILE, x0:x0 + CFG.CROP_TILE]
                    cv2.imwrite(str(img_out), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                lines = _crop_label_lines(image_id, y0, x0)
                lbl_out.write_text("\\n".join(lines), encoding="utf-8")
                written += 1
    return written

def build_crop_dataset():
    """folds(1,2) 全 record × 9 crop + fold-0 reference record の val crop。冪等。"""
    root = CFG.YOLO_ROOT
    done = root / "DATASET_COMPLETE"
    stats = {"train_samples": 0, "val_samples": 0}
    if done.exists():
        for split, key in (("train", "train_samples"), ("val", "val_samples")):
            stats[key] = len(list((root / "labels" / split).glob("*.txt")))
        return root / "data.yaml", stats
    for file_name in tqdm(fit_files, desc="crop dataset (train)"):
        image = cv2.imread(str(CFG.PREPROC_DIR / file_name), cv2.IMREAD_COLOR)
        stats["train_samples"] += _write_crop_split(
            root, "train", file_name, file_to_ids[file_name], image)
    for file_name in tqdm(valid_files, desc="crop dataset (val)"):
        image = cv2.imread(str(CFG.PREPROC_DIR / file_name), cv2.IMREAD_COLOR)
        stats["val_samples"] += _write_crop_split(
            root, "val", file_name, [file_to_ids[file_name][0]], image)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
                         encoding="utf-8")
    done.write_text("ok", encoding="utf-8")
    return yaml_path, stats

def train_crop_yolo():
    """prod レシピ同一・imgsz=896 のみ変更 (§2.94)。TRAINING_COMPLETE で resume。"""
    run_name = f"yolo_croptrain_fold{CFG.FOLD}"
    run_dir = CFG.RUNS_ROOT / run_name
    best_path, done_path = run_dir / "weights" / "best.pt", run_dir / "TRAINING_COMPLETE"
    yaml_path, stats = build_crop_dataset()
    if done_path.exists() and best_path.exists():
        print(f"学習済み checkpoint を再利用 {best_path}")
        return best_path, stats
    last_path = run_dir / "weights" / "last.pt"
    if last_path.exists():  # 中断検出: 同一 run dir で resume (中断なし実行と等価)
        print(f"中断検出: {last_path} から resume")
        started = time.time()
        YOLO(str(last_path)).train(resume=True)
        stats["train_minutes_resumed"] = round((time.time() - started) / 60, 1)
        stats["resumed_from_last_pt"] = True
        done_path.write_text("ok", encoding="utf-8")
        return best_path, stats
    started = time.time()
    YOLO(CFG.YOLO_MODEL).train(
        data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.TRAIN_IMGSZ,
        batch=CFG.YOLO_BATCH, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
        mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
        cls=CFG.YOLO_CLS,  # prod 既定 0.5
        optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
        degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
        project=str(CFG.RUNS_ROOT), name=run_name, exist_ok=True)
    stats["train_minutes"] = round((time.time() - started) / 60, 1)
    done_path.write_text("ok", encoding="utf-8")
    return best_path, stats
'''

CELL6 = '''# [6/6] 実行: prod baseline → croptrain 学習 (resume 可) → croptrain_full 評価、manifest 逐次保存 (informational、真の判定は v121c §2.99)
import json

def save_manifest(manifest):
    tmp = CFG.MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, CFG.MANIFEST)
''' + V114_WRAPPERS + '''

CONTRACT = {"experiment_id": "v121a_croptrain_f0", "fold": CFG.FOLD,
            "protocol": "RESEARCH_LOG §2.99 (cons2of3_c26 fold-0 replication, v116 crop recipe verbatim)",
            "train_folds": list(CFG.TRAIN_FOLDS),
            "crop_tile": CFG.CROP_TILE, "crop_stride": CFG.CROP_STRIDE,
            "crop_starts": list(CFG.CROP_STARTS), "train_imgsz": CFG.TRAIN_IMGSZ,
            "min_crop_area": CFG.MIN_AREA, "jpeg_quality": 95,
            "yolo_recipe": "prod verbatim except train imgsz=896 (§2.94)",
            "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,
            "arms": list(CFG.ARMS),
            "params_of_arm": {k: dict(v) for k, v in CFG.PARAMS_OF_ARM.items()},
            "conf_sweep": list(CFG.CONF_SWEEP), "primary_conf": CFG.PRIMARY_CONF,
            "refiner_threshold": CFG.REFINER_THRESHOLD, "drop_threshold": CFG.DROP_THRESHOLD,
            "min_area": CFG.MIN_AREA,
            "gate": "informational paired sanity only — 真の gate は v121c consensus eval (§2.99)"}
manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8")) if CFG.MANIFEST.exists() else {}
if manifest and manifest.get("contract") != CONTRACT:
    raise ValueError("既存 manifest と入力契約が違う。履歴保持のため別パスを指定してください")
manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}

if "prod" not in manifest["results"]:
    manifest["results"]["prod"] = {"weights": str(CFG.PROD_YOLO), "by_conf": evaluate_yolo(CFG.PROD_YOLO, "prod")}
    save_manifest(manifest)
prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]
print({"prod@primary": round(prod_pq, 6), "既知値": CFG.BASELINE_PQ, "差": round(prod_pq - CFG.BASELINE_PQ, 6)})

if "croptrain" not in manifest["datasets"]:  # 両 arm は同一重みを共有 (1回学習)
    weights, stats = train_crop_yolo()
    manifest["datasets"]["croptrain"] = stats | {"weights": str(weights),
                                                "weights_sha256": file_sha256(weights)}
    save_manifest(manifest)
CROP_WEIGHTS = manifest["datasets"]["croptrain"]["weights"]

for arm in CFG.ARMS:
    if arm in manifest["results"]:
        continue
    p = {"sahi_tile": None, "sahi_overlap": 0}  # prod リテラル (CFG 持ち越さない)
    p.update(CFG.PARAMS_OF_ARM[arm])
    CFG.SAHI_TILE, CFG.SAHI_OVERLAP = p["sahi_tile"], p["sahi_overlap"]
    by_conf = evaluate_yolo(Path(CROP_WEIGHTS), arm)
    delta = by_conf[str(CFG.PRIMARY_CONF)]["pq"] - prod_pq
    verdict = ("informational_positive" if delta >= CFG.GATE_DELTA
               else "informational")  # 真の判定は v121c (§2.99)
    manifest["results"][arm] = {"weights": CROP_WEIGHTS, "by_conf": by_conf,
                                "delta_primary_vs_prod": delta, "params": p,
                                "verdict": verdict}
    manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    print(arm, {"Δ@primary": round(delta, 4), "verdict": verdict})

print(json.dumps({a: {c: round(v["pq"], 4) for c, v in r["by_conf"].items()
                      if isinstance(v, dict) and "pq" in v}  # diag_lowconf を除外
                  for a, r in manifest["results"].items()
                  if isinstance(r, dict) and isinstance(r.get("by_conf"), dict)}, indent=2))
print("manifest:", CFG.MANIFEST)
'''


def patch_config(src: str) -> str:
    src = replace_once(src, "    FOLD = 2  # v96: fold2 で評価 (prod runs/yolo_fold_2 と paired)",
                       "    FOLD = 0  # v121a: fold0 で評価 (prod runs/yolo_fold_0 と paired)")
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('croptrain_full',)\n"
                       "    TRAIN_FOLDS = (1, 2)  # prod fold-0 recipe (fold-0 非接触)\n"
                       "    # --- v116 crop dataset (§2.94 verbatim、fold のみ変更) ---\n"
                       "    CROP_TILE = 896\n"
                       "    CROP_STRIDE = 448\n"
                       "    CROP_STARTS = [0, 448, 1152]  # 3x3=9 crop 全覆盖 (JSON-safe list)\n"
                       "    TRAIN_IMGSZ = 896             # crop ネイティブ = 縮小なし\n"
                       "    SAHI_TILE = None              # prod 既定 (単一画像推論)\n"
                       "    SAHI_OVERLAP = 0              # prod 既定\n"
                       "    _SAHI_CUR = None              # refine キャッシュの画像キー (predict で設定)\n"
                       "    PARAMS_OF_ARM = {\n"
                       "        'croptrain_full': {'sahi_tile': None, 'sahi_overlap': 0}}")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v121_croptrain_f0_honest"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v121_croptrain_f0_honest"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v121a")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v121a_croptrain_f0_colab_manifest.json"')
    src = replace_once(src, "    BASELINE_PQ = 0.444238  # fold2 prod 既知値 (§2.48、warn-only)",
                       "    BASELINE_PQ = 0.44413694566994816  # fold0 prod 確定値 (v108 §2.86、warn-only)")
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # informational (真のゲートは v121c §2.99)")
    return src


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
