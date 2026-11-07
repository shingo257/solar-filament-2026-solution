"""V112 Colab ノートのビルダー: pseudotest の fold-0 honest 複製 (RESEARCH_LOG §2.90)。

v111 (§2.89) の fold-2 Δ=+0.0070 はゲート未達で公式棄却だが、
キャンペーン最大の正値のため fold-0 複製を実施する (v108 と同型)。
v111 ビルダーの fold ミラー: FOLD=0、教師 = fold-0 prod + fold-0 refiner、
TRAIN_FOLDS=(1,2)、判定 Δ>=+0.003 → cross_fold_verified。

- `pseudotest_f0`: folds(1,2) 実ラベル + test 疑似ラベル (fold-0 教師)。
  fold-0 は疑似ラベル生成・学習・val 全て非接触 (transductive 汚染なし)。
- build_fold_dataset に pseudo マージ + batch4→2 OOM fallback (v108 同型)。
- 採点セルは無改変。置換は「ちょうど 1 回」一致を assert。

    python scratch/build_v112_pseudo_f0_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v112_pseudo_f0_colab.ipynb"


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


TITLE = """# v112 — pseudotest の fold-0 honest 複製 (Colab 版)

RESEARCH_LOG §2.90 の事前登録実験。v111 の fold-2 +0.0070
(キャンペーン最大の正値、ゲート未達で公式棄却) を fold-0 で複製検証。

| arm | 内容 |
|---|---|
| `pseudotest_f0` | folds(1,2) 実ラベル + test 疑似ラベル (fold-0 prod 教師、conf>=0.26) |

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`
(test_images 180 枚含む)・`artifacts/grouped_folds.csv`・
`artifacts/mediseg_edgeattnet_refiner_fold0_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_0/weights/best.pt`。

**対照**: prod fold-0 baseline = 0.444137 (v108 §2.86 確定値、
同一セッションでも paired 再測定)。

**判定 (§2.90、結果取得前に固定)**: Δ@0.26 >= **+0.003** →
`cross_fold_verified` → 採用検討へ (MAGFiLO 規約での test 半教師
許容確認が必須条件)。未満 → `unverified_fold0` で疑似ラベル系閉鎖。

GPU 必須 (~4h: 疑似ラベル ~30分 + 学習 ~2h + 評価 ~45分)。
上から順に全セル実行。結果は
`artifacts/v112_pseudo_f0_colab_manifest.json` に逐次保存。
"""

V112_FUNCS = '''

# --- v112: test 画像疑似ラベル生成 (§2.90、fold-0 prod パイプラインを教師に使用)
def robust_uint8(array, mask=None, low=0.5, high=99.5):
    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)

def solar_three_channel(gray):  # submission_reproduction.ipynb と同一式
    # EdgeAttNet着想: 径方向/背景勾配を抑えて暗いフィラメントを浮かせつつ元観測も残す
    gray = gray.astype(np.float32)
    disk = gray > max(4.0, np.percentile(gray, 8))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=80, sigmaY=80)
    flattened = gray / np.maximum(background, 1.0)
    flat_u8 = robust_uint8(flattened, disk, 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(flat_u8)
    raw_u8 = robust_uint8(gray, disk, 0.5, 99.5)
    dark_detail = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    dark_detail = robust_uint8(dark_detail, disk, 0.0, 99.5)
    out = np.dstack([raw_u8, clahe, dark_detail])
    out[~disk] = 0
    return out

def pseudo_mask_to_line(mask, width, height):
    """disjoint bool mask → YOLO polygon line (最大輪郭のみ、MAGFiLO 1個体1polygon 規約)。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        return None
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.reshape(-1))

def ensure_pseudolabels():
    """fold-0 prod (folds(1,2) 学習済み) で test 180 枚を疑似ラベル化。
    推論経路は採点と同一 (floor 0.08 → refine → conf>=PSEUDO_CONF & MIN_AREA
    → ownership/disjoint)。画像単位の逐次書き込み + COMPLETE マーカーで冪等。"""
    done = CFG.PSEUDO_DIR / "PSEUDOLABELS_COMPLETE"
    stats_path = CFG.PSEUDO_DIR / "pseudolabel_stats.json"
    if done.exists() and stats_path.exists():
        return json.loads(stats_path.read_text(encoding="utf-8"))
    test_dir = next(CFG.DATA_CACHE.rglob("test_images"))
    test_files = sorted(p.name for p in test_dir.glob("*.jpeg"))
    assert len(test_files) == 180, f"test 画像が 180 枚ではない: {len(test_files)}"
    for name in tqdm(test_files, desc="test 画像を前処理"):
        local = CFG.PREPROC_DIR / name
        if local.exists():
            continue
        if (CFG.PREPROC_CACHE / name).exists():
            shutil.copy2(CFG.PREPROC_CACHE / name, local)
            continue
        gray = cv2.imread(str(test_dir / name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(test_dir / name)
        cv2.imwrite(str(local), solar_three_channel(gray), [cv2.IMWRITE_JPEG_QUALITY, 96])
    img_dir, lbl_dir = CFG.PSEUDO_DIR / "images", CFG.PSEUDO_DIR / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(CFG.PROD_YOLO))
    n_img, n_inst, n_prop = 0, 0, 0
    for name in tqdm(test_files, desc="pseudolabel 生成 (fold-0 prod 経路)"):
        stem = "pseudotest__" + Path(name).stem
        if (lbl_dir / f"{stem}.txt").exists():  # 途中再開: 書き込み済みは集計のみ
            n_img += 1
            n_inst += sum(1 for line in (lbl_dir / f"{stem}.txt").read_text().splitlines() if line.strip())
            continue
        path = CFG.PREPROC_DIR / name
        result = model.predict(str(path), imgsz=CFG.YOLO_IMGSZ, conf=CFG.PREDICT_CONF_FLOOR,
                               iou=CFG.PREDICT_IOU, agnostic_nms=True, retina_masks=True,
                               verbose=False)[0]
        masks, conf = result_masks(result)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        keep = [i for i in range(len(conf))
                if conf[i] >= CFG.PSEUDO_CONF and masks[i].sum() >= CFG.MIN_AREA]
        n_prop += len(masks)
        preds = apply_ownership([refine_proposal(image, masks[i]) for i in keep])
        lines = [line for line in (pseudo_mask_to_line(m, image.shape[1], image.shape[0])
                                   for m in preds) if line]
        shutil.copy2(path, img_dir / f"{stem}.jpeg")
        (lbl_dir / f"{stem}.txt").write_text("\\n".join(lines), encoding="utf-8")
        n_img += 1
        n_inst += len(lines)
    del model
    torch.cuda.empty_cache()
    stats = {"test_images": n_img, "pseudo_instances": n_inst,
             "raw_proposals_ge_floor": n_prop,
             "teacher": "prod fold0 detector + fold0 EdgeAtt refiner + disjoint",
             "conf_keep": CFG.PSEUDO_CONF, "pseudo_dir": str(CFG.PSEUDO_DIR)}
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    done.write_text("ok", encoding="utf-8")
    return stats
'''


def patch_config(src: str) -> str:
    src = replace_once(src, "    FOLD = 2  # v96: fold2 で評価 (prod runs/yolo_fold_2 と paired)",
                       "    FOLD = 0  # v112: fold0 で評価 (prod runs/yolo_fold_0 と paired)")
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('pseudotest_f0',)\n"
                       "    TRAIN_FOLDS = (1, 2)  # prod fold-0 recipe (fold-0 非接触)\n"
                       "    PSEUDO_CONF = 0.26   # = PRIMARY_CONF (prod 最終出力を教師に)\n"
                       "    PSEUDO_DIR = None    # 実行セルで YOLO_ROOT から設定")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v112_pseudo_f0"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v112_pseudo_f0"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v112")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v112_pseudo_f0_colab_manifest.json"')
    src = replace_once(src, "    BASELINE_PQ = 0.444238  # fold2 prod 既知値 (§2.48、warn-only)",
                       "    BASELINE_PQ = 0.44413694566994816  # fold0 prod 確定値 (v108 §2.86、warn-only)")
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.003  # §2.90: cross_fold_verified 複製ゲート (v108 同型)")
    return src


def patch_dataset(src: str) -> str:
    src = replace_once(src, '    root = CFG.YOLO_ROOT / f"val{fold}_train{\'\'.join(map(str, CFG.TRAIN_FOLDS))}"',
                       '    root = CFG.YOLO_ROOT / f"val{fold}_train{\'\'.join(map(str, CFG.TRAIN_FOLDS))}_pseudotest"')
    src = replace_once(src,
                       """    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
                         encoding="utf-8")""",
                       """    if CFG.PSEUDO_DIR is not None:  # v112: test 疑似ラベルを train に併合
        pseudo_labels = sorted((CFG.PSEUDO_DIR / "labels").glob("*.txt"))
        for label_path in pseudo_labels:
            image_src = CFG.PSEUDO_DIR / "images" / f"{label_path.stem}.jpeg"
            if not image_src.exists():
                raise FileNotFoundError(f"疑似ラベルに対応する画像がない: {image_src}")
            shutil.copy2(image_src, root / "images" / "train" / f"{label_path.stem}.jpeg")
            shutil.copy2(label_path, root / "labels" / "train" / label_path.name)
            stats["train_samples"] += 1
            stats["pseudo_instances"] = stats.get("pseudo_instances", 0) + sum(
                1 for line in label_path.read_text().splitlines() if line.strip())
        stats["pseudo_images"] = len(pseudo_labels)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
                         encoding="utf-8")""")
    src = replace_once(src,
                       """    YOLO(CFG.YOLO_MODEL).train(
        data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
        batch=CFG.YOLO_BATCH, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
        mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
        cls=CFG.YOLO_CLS,  # prod 既定 0.5 (v96 の差分は学習 fold 集合のみ)
        optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
        degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
        project=str(CFG.RUNS_ROOT), name=run_name, exist_ok=True,
    )""",
                       """    _train_kwargs = dict(
        data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
        device=0, workers=4, patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True,
        mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
        cls=CFG.YOLO_CLS,  # prod 既定 0.5
        optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
        degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
        project=str(CFG.RUNS_ROOT), name=run_name, exist_ok=True,
    )
    for _batch in (CFG.YOLO_BATCH, 2):  # v108 と同型の OOM fallback (1回のみ、記録)
        try:
            YOLO(CFG.YOLO_MODEL).train(batch=_batch, **_train_kwargs)
            break
        except torch.cuda.OutOfMemoryError:
            stats["batch_fallback"] = True
            torch.cuda.empty_cache()
    else:
        raise RuntimeError("YOLO training OOM at batch<=2")""")
    return src


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       "# [6/6] 実行: prod baseline → test 疑似ラベル生成 → pseudotest_f0 arm 学習 → 評価、manifest 逐次保存 (判定は §2.90)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v112_pseudo_f0"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.90 (test-image pseudolabel fold-0 replication)"')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.003 → cross_fold_verified + rules check (§2.90)",\n'
                       '            "pseudo_conf": CFG.PSEUDO_CONF')
    src = replace_once(src, 'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}',
                       'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}\n'
                       + V112_FUNCS)
    src = replace_once(src,
                       """prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]""",
                       """CFG.PSEUDO_DIR = CFG.YOLO_ROOT / "pseudotest_teacher_f0"
manifest["datasets"]["pseudolabels"] = manifest["datasets"].get(
    "pseudolabels") or ensure_pseudolabels()
save_manifest(manifest)
prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = ("cross_fold_verified" if delta >= CFG.GATE_DELTA\n'
                       '               else "unverified_fold0")')
    return src


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = _cell("markdown", TITLE)
    cells[1]["source"] = patch_config(_source(cells[1])).splitlines(keepends=True)
    # cell2 (COCO/fold/前処理)・cell4 (refiner 定義)・cell5 (評価関数) は無改変
    cells[3]["source"] = patch_dataset(_source(cells[3])).splitlines(keepends=True)
    cells[6]["source"] = patch_run(_source(cells[6])).splitlines(keepends=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {OUT.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
