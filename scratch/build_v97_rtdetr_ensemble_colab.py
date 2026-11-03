"""V97 Colab ノートのビルダー: RT-DETR 第2検出器 + 校正済み true-dedup ensemble (RESEARCH_LOG §2.71)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・評価関数)
を無改変で再利用し、検出器学習セルと実行セルだけを v97 用に差し替える。
新規セルは (a) bbox 形式データセット + RT-DETR 学習、(b) 候補収集・isotonic
校正・dedup-before-refine・ensemble 評価、(c) 実行/manifest。
置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v97_rtdetr_ensemble_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v97_rtdetr_ensemble_colab.ipynb"


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


TITLE = """# v97 — RT-DETR 第2検出器 + 校正済み true-dedup ensemble (Colab 版)

RESEARCH_LOG §2.71 の事前登録実験。残存する唯一の未試行上流レバー:
biconcavelens が real LB +0.01 を得た「isotonic 校正 + dedup-before-paint
(mask IoU>0.05 一括破棄)」merge 機構を、prod YOLO + 新学習 RT-DETR-l で
fold2 paired 評価する。

**前提 artifacts** (Drive `solar_filament_v30/`):

- `competition_data/` の COCO 訓練 JSON
- `artifacts/grouped_folds.csv` (SHA256 凍結)
- `preprocessed_v30_q96/` 前処理画像
- `runs/yolo_fold_2/weights/best.pt` (prod YOLO)
- `artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt` (fold2 refiner)

**ゲート (§2.71、結果取得前に固定)**: Δ = PQ(ens, cross-fit 校正,
dedup=0.05, λ=0.5·PQ_prod) − PQ(prod,0.26) が >= +0.005 なら両検出器を
全データ再学習 → LB 確認。それ以外は ensemble 機構を閉鎖。
"""

CELL3 = '''# [3/7] bbox データセット構築 (all-labeler・polygon外接矩形) + RT-DETR 学習
def bbox_line(ann, width, height):
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    polygon = max(polygons, key=len)  # prod polygon_line と同じ選択規則
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
    w, h = (x1 - x0) / width, (y1 - y0) / height
    if w <= 0 or h <= 0:
        return None
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def write_det_variant(fold_root, split, file_name, image_id, suffix):
    info = images[image_id]
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{stem}.jpeg"
    label_out = fold_root / "labels" / split / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    if not image_out.exists():
        shutil.copy2((CFG.PREPROC_DIR / file_name).resolve(), image_out)
    lines = [bbox_line(ann, info["width"], info["height"])
             for ann in annotations[image_id] if not ann.get("iscrowd", 0)]
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(line for line in lines if line), encoding="utf-8")

def build_det_dataset(fold):
    root = CFG.YOLO_ROOT / f"det_val{fold}_train{''.join(map(str, CFG.TRAIN_FOLDS))}"
    reference_id = {name: file_to_ids[name][0] for name in train_files}
    stats = {"train_samples": 0, "train_instances": 0, "val_samples": 0, "val_instances": 0}
    for file_name in tqdm(train_files, desc=f"fold {fold} bbox データセット構築"):
        if fold_of_file[file_name] == fold:
            write_det_variant(root, "val", file_name, reference_id[file_name], "")
            stats["val_samples"] += 1
            stats["val_instances"] += len(annotations[reference_id[file_name]])
        if fold_of_file[file_name] in CFG.TRAIN_FOLDS:
            for image_id in file_to_ids[file_name]:
                write_det_variant(root, "train", file_name, image_id, f"__labeler_{image_id}")
                stats["train_samples"] += 1
                stats["train_instances"] += len(annotations[image_id])
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
                         encoding="utf-8")
    return yaml_path, stats

def train_rtdetr():
    run_name = f"rtdetr_fold{CFG.FOLD}"
    run_dir = CFG.RUNS_ROOT / run_name
    best_path, done_path = run_dir / "weights" / "best.pt", run_dir / "TRAINING_COMPLETE"
    yaml_path, stats = build_det_dataset(CFG.FOLD)  # 既存ファイルは skip (冪等)
    if done_path.exists() and best_path.exists():
        print(f"rtdetr: 学習済み checkpoint を再利用 {best_path}")
        return best_path, stats
    started = time.time()
    RTDETR(CFG.RTDETR_MODEL).train(
        data=str(yaml_path), epochs=CFG.RTDETR_EPOCHS, imgsz=CFG.RTDETR_IMGSZ,
        batch=CFG.RTDETR_BATCH, device=0, workers=4, patience=CFG.RTDETR_PATIENCE,
        seed=CFG.SEED + CFG.FOLD, deterministic=True, single_cls=True, amp=True,
        cls=CFG.RTDETR_CLS,  # biconcavelens 検証値 2.0 (confidence head undertraining 対策)
        cache="disk",
        degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
        project=str(CFG.RUNS_ROOT), name=run_name, exist_ok=True,
    )
    stats["train_minutes"] = round((time.time() - started) / 60, 1)
    done_path.write_text("ok", encoding="utf-8")
    return best_path, stats

'''

CELL_ENS = '''# [6/7] v97 ensemble 機構: 候補収集・isotonic 校正・dedup-before-refine・ensemble 評価
@torch.no_grad()
def gather_candidates(weights_path, det, files):
    """det='yolo' → seg mask 提案、det='rtdetr' → box 充填 mask 提案。
    {file_name: [(conf, mask_rle)]} を DIAG_CONF_FLOOR で収集。"""
    model = YOLO(str(weights_path)) if det == "yolo" else RTDETR(str(weights_path))
    imgsz = CFG.YOLO_IMGSZ if det == "yolo" else CFG.RTDETR_IMGSZ
    out = {}
    for name in tqdm(files, desc=f"{det} 候補収集"):
        path = str(CFG.PREPROC_DIR / name)
        if det == "yolo":
            result = model.predict(path, imgsz=imgsz, conf=CFG.DIAG_CONF_FLOOR,
                                   iou=CFG.PREDICT_IOU, agnostic_nms=True,
                                   retina_masks=True, verbose=False)[0]
        else:
            result = model.predict(path, imgsz=imgsz, conf=CFG.DIAG_CONF_FLOOR,
                                   verbose=False)[0]
        height, width = result.orig_shape
        confs = (result.boxes.conf.detach().cpu().numpy() if result.boxes is not None
                 else np.zeros(0, np.float32))
        cands = []
        if det == "yolo":
            masks, _ = result_masks(result)
            for m, c in zip(masks, confs):
                cands.append((float(c), mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))))
        else:
            boxes = (result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None
                     else np.zeros((0, 4), np.float32))
            for box, c in zip(boxes, confs):
                m = np.zeros((height, width), dtype=np.uint8)
                x0, y0, x1, y1 = [int(round(v)) for v in box]
                m[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 1
                if m.sum():
                    cands.append((float(c), mask_utils.encode(np.asfortranarray(m))))
        out[name] = cands
    del model; torch.cuda.empty_cache()
    return out

def expected_hit_iou(cand_rle, name):
    """isotonic 目的変数: file の全 annotator record で
    (その record の GT と IoU>0.5 ならその IoU、否 0) の平均。"""
    vals = []
    for image_id in file_to_ids[name]:
        truth = gt_rles[image_id]
        if not truth:
            vals.append(0.0); continue
        ious = mask_utils.iou([cand_rle], truth, [0] * len(truth))[0]
        hit = float(ious.max()) if len(ious) else 0.0
        vals.append(hit if hit > 0.5 else 0.0)
    return float(np.mean(vals)) if vals else 0.0

def fit_isotonic(cand_by_file, files):
    """files 上の全候補で conf→E[hit-IoU] の isotonic を fit。"""
    xs, ys = [], []
    for name in files:
        for conf, rle in cand_by_file[name]:
            xs.append(conf); ys.append(expected_hit_iou(rle, name))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(xs), np.asarray(ys))
    return iso

def dedup_accept(pool, dedup_iou):
    """cal 降順に貪欲受理 — 受理済みと mask-IoU>dedup_iou の候補は破棄。"""
    accepted = []
    for entry in sorted(pool, key=lambda t: -t[0]):
        cal, det, conf, rle = entry
        if accepted:
            ious = mask_utils.iou([rle], [a[3] for a in accepted], [0] * len(accepted))[0]
            if (ious > dedup_iou).any():
                continue
        accepted.append(entry)
    return accepted

@torch.no_grad()
def ensemble_eval(cand_pools, cal_maps, dedup_iou, accept_lam, files):
    """受理候補 (cal>λ) を refine → ownership → 公式 PQ (record 単位)。
    cal_maps: {det: {file_name: fitted isotonic}} — cross-fit では file 毎に
    反対 half で fit した写像を使う。"""
    meter = PanopticQuality()
    n_pred, n_cand = 0, 0
    for name in files:
        pool = []
        for det, by_file in cand_pools.items():
            iso = cal_maps[det][name]
            for conf, rle in by_file.get(name, []):
                pool.append((float(iso.predict([conf])[0]), det, conf, rle))
        n_cand += len(pool)
        survivors = [e for e in dedup_accept(pool, dedup_iou) if e[0] > accept_lam]
        image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
        refined = [refine_proposal(image, mask_utils.decode(e[3]).astype(bool)) for e in survivors]
        preds = apply_ownership([m for m in refined if m.any()])
        pred_rles = (mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8)))
                     if preds else [])
        n_pred += len(pred_rles)
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
    s = meter.summary()
    s["preds_per_image"] = n_pred / max(len(files), 1)
    s["cands_per_image"] = n_cand / max(len(files), 1)
    return s

def solo_eval(cand_by_file, conf_sweep, files):
    """detector 単体: conf>=c の候補を refine → ownership → 公式 PQ。"""
    out = {}
    for c in conf_sweep:
        meter = PanopticQuality()
        n_pred = 0
        for name in files:
            keep = [rle for conf, rle in cand_by_file.get(name, []) if conf >= c]
            image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
            refined = [refine_proposal(image, mask_utils.decode(r).astype(bool)) for r in keep]
            preds = apply_ownership([m for m in refined if m.any()])
            pred_rles = (mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8)))
                         if preds else [])
            n_pred += len(pred_rles)
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
        out[str(c)] = {**meter.summary(), "preds_per_image": n_pred / max(len(files), 1)}
        print("solo", c, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[str(c)].items()})
    return out

def crossfit_halves(files):
    """deterministic 2 分割 (sorted index parity)。"""
    srt = sorted(files)
    return srt[0::2], srt[1::2]

def _part_path(tag):
    d = CFG.ARTIFACT_ROOT / "v97_eval_parts"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}.pt"

def _part_save(path, done):
    tmp = path.with_suffix(".tmp")
    torch.save(done, tmp)
    os.replace(tmp, path)

@torch.no_grad()
def ensemble_eval_cached(cand_pools, cal_maps, dedup_iou, accept_lam, files, tag):
    """ensemble_eval の file 単位チェックポイント版。
    file 毎の受理後 pred_rles を Drive の v97_eval_parts/{tag}.pt に逐次保存し、
    中断時は未処理 file のみ再計算。tag は構成を一意に表す文字列とする。"""
    part = _part_path(f"ens_{tag}")
    done = _part_load(part)
    n_cand = 0
    for name in tqdm(files, desc=f"ens[{tag}]"):
        for det, by_file in cand_pools.items():
            n_cand += len(by_file.get(name, []))
        if name in done:
            continue
        pool = []
        for det, by_file in cand_pools.items():
            iso = cal_maps[det][name]
            for conf, rle in by_file.get(name, []):
                pool.append((float(iso.predict([conf])[0]), det, conf, rle))
        survivors = [e for e in dedup_accept(pool, dedup_iou) if e[0] > accept_lam]
        image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
        refined = [refine_proposal(image, mask_utils.decode(e[3]).astype(bool)) for e in survivors]
        preds = apply_ownership([m for m in refined if m.any()])
        pred_rles = (mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8)))
                     if preds else [])
        done[name] = pred_rles
        _part_save(part, done)
    meter = PanopticQuality()
    n_pred = 0
    for name in files:
        pred_rles = done[name]
        n_pred += len(pred_rles)
        for image_id in file_to_ids[name]:
            truth = gt_rles[image_id]
            meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
    s = meter.summary()
    s["preds_per_image"] = n_pred / max(len(files), 1)
    s["cands_per_image"] = n_cand / max(len(files), 1)
    return s

def _part_load(path):
    return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else {}

@torch.no_grad()
def solo_eval_cached(cand_by_file, conf_sweep, files, tag):
    """solo_eval の file 単位チェックポイント版 ({conf: {file: pred_rles}})。"""
    part = _part_path(f"solo_{tag}")
    done = _part_load(part)
    out = {}
    for c in conf_sweep:
        key = str(c)
        per_file = done.setdefault(key, {})
        n_pred = 0
        for name in tqdm(files, desc=f"solo[{tag}]@{c}"):
            if name in per_file:
                continue
            keep = [rle for conf, rle in cand_by_file.get(name, []) if conf >= c]
            image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
            refined = [refine_proposal(image, mask_utils.decode(r).astype(bool)) for r in keep]
            preds = apply_ownership([m for m in refined if m.any()])
            per_file[name] = (mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8)))
                              if preds else [])
            _part_save(part, done)
        meter = PanopticQuality()
        for name in files:
            pred_rles = per_file[name]
            n_pred += len(pred_rles)
            for image_id in file_to_ids[name]:
                truth = gt_rles[image_id]
                meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
        out[key] = {**meter.summary(), "preds_per_image": n_pred / max(len(files), 1)}
        print("solo", c, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[key].items()})
    return out

'''

CELL_RUN = '''# [7/7] 実行: prod baseline → RT-DETR 学習 → 候補収集 → cross-fit 校正 ensemble → manifest
def save_manifest(manifest):
    CFG.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

CONTRACT = {"experiment_id": "v97_rtdetr_ensemble", "fold": CFG.FOLD,
            "protocol": "RESEARCH_LOG §2.71 (calibrated true-dedup ensemble gate)",
            "train_folds": list(CFG.TRAIN_FOLDS), "ultralytics": version("ultralytics"),
            "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO),
            "rtdetr": {"model": CFG.RTDETR_MODEL, "imgsz": CFG.RTDETR_IMGSZ,
                       "cls": CFG.RTDETR_CLS, "epochs": CFG.RTDETR_EPOCHS,
                       "batch": CFG.RTDETR_BATCH, "seed": CFG.SEED + CFG.FOLD},
            "ensemble": {"dedup_iou": CFG.ENS_DEDUP_IOU, "accept": "0.5*PQ(prod,0.26)",
                         "calibration": "isotonic conf->E[hit-IoU], cross-fit halves",
                         "conf_floor": CFG.DIAG_CONF_FLOOR},
            "gate": "PQ(ens,crossfit,dedup0.05,λ) - PQ(prod,0.26) >= +0.005 (§2.71)"}
manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8")) if CFG.MANIFEST.exists() else {}
if manifest and manifest.get("contract") != CONTRACT:
    raise ValueError("既存 manifest と入力契約が違う。履歴保持のため別パスを指定してください")
manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}

if "prod" not in manifest["results"]:
    manifest["results"]["prod"] = {"weights": str(CFG.PROD_YOLO), "by_conf": evaluate_yolo(CFG.PROD_YOLO, "prod")}
    save_manifest(manifest)
prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]
print({"prod@primary": round(prod_pq, 6), "既知値": CFG.BASELINE_PQ, "差": round(prod_pq - CFG.BASELINE_PQ, 6)})
if abs(prod_pq - CFG.BASELINE_PQ) > 0.002:
    print("⚠️ prod の再現値が既知値から 0.002 以上ずれている — 配線/入力の差を確認 (判定は paired Δ で行う)")

if "rtdetr" not in manifest.get("datasets", {}):
    weights, stats = train_rtdetr()
    manifest["datasets"]["rtdetr"] = {**stats, "weights": str(weights), "weights_sha256": file_sha256(weights)}
    save_manifest(manifest)
rtdetr_weights = manifest["datasets"]["rtdetr"]["weights"]

yolo_cands = rtdetr_cands = None
if CFG.CAND_CACHE.exists():
    try:
        _cache = torch.load(CFG.CAND_CACHE, map_location="cpu", weights_only=False)
        yolo_cands, rtdetr_cands = _cache["yolo"], _cache["rtdetr"]
    except Exception:
        CFG.CAND_CACHE.unlink(missing_ok=True)  # 中断で破損した cache を除去
if yolo_cands is None or "candidates" not in manifest["results"]:
    yolo_cands = gather_candidates(CFG.PROD_YOLO, "yolo", valid_files)
    rtdetr_cands = gather_candidates(rtdetr_weights, "rtdetr", valid_files)
    _part_save(CFG.CAND_CACHE, {"yolo": yolo_cands, "rtdetr": rtdetr_cands})
    # RLE は manifest に入らないので再現性は seed/重みで担保。候補数だけ記録。
    manifest["results"]["candidates"] = {
        "yolo": {n: len(c) for n, c in yolo_cands.items()},
        "rtdetr": {n: len(c) for n, c in rtdetr_cands.items()}}
    save_manifest(manifest)

cand_pools = {"yolo": yolo_cands, "rtdetr": rtdetr_cands}
half_a, half_b = crossfit_halves(valid_files)

cal_maps_cross = cal_maps_ins = None
if CFG.CAL_CACHE.exists():
    try:
        _cm = torch.load(CFG.CAL_CACHE, map_location="cpu", weights_only=False)
        cal_maps_cross, cal_maps_ins = _cm["cross"], _cm["ins"]
        print("calibration maps: Drive cache から復元 (fit skip)")
    except Exception:
        CFG.CAL_CACHE.unlink(missing_ok=True)  # 中断で破損した cache を除去
if cal_maps_cross is None:
    # isotonic cross-fit (honest): A→B, B→A
    cal_maps_cross = {"yolo": {}, "rtdetr": {}}
    for det in cand_pools:
        iso_a = fit_isotonic(cand_pools[det], half_a)
        iso_b = fit_isotonic(cand_pools[det], half_b)
        for n in half_b: cal_maps_cross[det][n] = iso_a
        for n in half_a: cal_maps_cross[det][n] = iso_b
    # in-sample (診断のみ — 全 file 同一写像)
    cal_maps_ins = {}
    for det in cand_pools:
        iso = fit_isotonic(cand_pools[det], valid_files)
        cal_maps_ins[det] = {n: iso for n in valid_files}
    _part_save(CFG.CAL_CACHE, {"cross": cal_maps_cross, "ins": cal_maps_ins})

lam = 0.5 * prod_pq
if "ensemble" in manifest["results"]:
    ens_primary = {k: v for k, v in manifest["results"]["ensemble"].items()
                   if k not in ("dedup_iou", "accept_lambda", "calibration", "delta_vs_prod", "verdict")}
    delta = manifest["results"]["ensemble"]["delta_vs_prod"]
    verdict = manifest["results"]["ensemble"]["verdict"]
else:
    ens_primary = ensemble_eval_cached(cand_pools, cal_maps_cross, CFG.ENS_DEDUP_IOU, lam, valid_files,
                                       tag=f"cross_d{CFG.ENS_DEDUP_IOU}_l{lam:.4f}")
    delta = ens_primary["pq"] - prod_pq
    verdict = "promote_to_alldata_lb" if delta >= CFG.GATE_DELTA else ("reject_noise" if delta >= 0 else "reject")
    manifest["results"]["ensemble"] = {
        "dedup_iou": CFG.ENS_DEDUP_IOU, "accept_lambda": lam, "calibration": "cross-fit isotonic",
        **ens_primary, "delta_vs_prod": delta, "verdict": verdict}
    save_manifest(manifest)
print("ENSEMBLE PRIMARY", {k: round(v, 4) if isinstance(v, float) else v for k, v in ens_primary.items()}, "Δ", round(delta, 4), verdict)

# ---- 判定不使用の診断 ----
diag = manifest["results"].setdefault("diagnostics", {})
if "solo_rtdetr" not in diag:
    diag["solo_rtdetr"] = solo_eval_cached(rtdetr_cands, CFG.CONF_SWEEP, valid_files, tag="rtdetr")
    save_manifest(manifest)
if "ens_insample" not in diag:
    diag["ens_insample"] = ensemble_eval_cached(cand_pools, cal_maps_ins, CFG.ENS_DEDUP_IOU, lam, valid_files,
                                              tag=f"ins_d{CFG.ENS_DEDUP_IOU}_l{lam:.4f}")
    save_manifest(manifest)
diag.setdefault("dedup_sweep", {})
for d in (0.03, 0.05, 0.10):
    if str(d) in diag["dedup_sweep"]:
        continue
    diag["dedup_sweep"][str(d)] = ensemble_eval_cached(cand_pools, cal_maps_cross, d, lam, valid_files,
                                                     tag=f"cross_d{d}_l{lam:.4f}")
    save_manifest(manifest)
diag.setdefault("lambda_sweep", {})
for m in (0.8, 1.0, 1.2):
    if str(round(m * lam, 4)) in diag["lambda_sweep"]:
        continue
    diag["lambda_sweep"][str(round(m * lam, 4))] = ensemble_eval_cached(
        cand_pools, cal_maps_cross, CFG.ENS_DEDUP_IOU, m * lam, valid_files,
        tag=f"cross_d{CFG.ENS_DEDUP_IOU}_l{m * lam:.4f}")
    save_manifest(manifest)
if "yolo_only_cal_dedup" not in diag:
    diag["yolo_only_cal_dedup"] = ensemble_eval_cached(
        {"yolo": yolo_cands}, {"yolo": cal_maps_cross["yolo"]},
        CFG.ENS_DEDUP_IOU, lam, valid_files,
        tag=f"yolo_cross_d{CFG.ENS_DEDUP_IOU}_l{lam:.4f}")
    save_manifest(manifest)
if "rtdetr_only_cal_dedup" not in diag:
    diag["rtdetr_only_cal_dedup"] = ensemble_eval_cached(
        {"rtdetr": rtdetr_cands}, {"rtdetr": cal_maps_cross["rtdetr"]},
        CFG.ENS_DEDUP_IOU, lam, valid_files,
        tag=f"rtdetr_cross_d{CFG.ENS_DEDUP_IOU}_l{lam:.4f}")
    save_manifest(manifest)
manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
save_manifest(manifest)
print(json.dumps({"verdict": verdict, "delta": round(delta, 4),
                  "ens": {k: round(v, 4) if isinstance(v, float) else v for k, v in ens_primary.items()},
                  "prod": round(prod_pq, 4)}, indent=2))
print("manifest:", CFG.MANIFEST)

'''

def patch_config(src: str) -> str:
    src = replace_once(src, "from ultralytics import YOLO",
                       "from ultralytics import RTDETR, YOLO")
    src = replace_once(src, "from tqdm.auto import tqdm",
                       "from tqdm.auto import tqdm\nfrom sklearn.isotonic import IsotonicRegression")
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('rtdetr_ens',)\n    TRAIN_FOLDS = (0, 1)")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v97_rtdetr_ensemble"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v97_rtdetr_ensemble"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v97")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v97_rtdetr_ensemble_colab_manifest.json"\n'
                       '    CAND_CACHE = ARTIFACT_ROOT / "v97_candidates.pt"\n'
                       '    CAL_CACHE = ARTIFACT_ROOT / "v97_cal_maps.pt"')
    src = replace_once(src, "    # --- YOLO (prod と完全一致。v96 は学習 fold 集合のみ変更) ---",
                       "    # --- RT-DETR 第2検出器 (biconcavelens 検証値に基づく) ---\n"
                       "    RTDETR_MODEL = 'rtdetr-l.pt'\n"
                       "    RTDETR_IMGSZ = 1280\n"
                       "    RTDETR_BATCH = 2   # A100 40GB 安全側。余裕があれば 4 へ\n"
                       "    RTDETR_EPOCHS = 70\n"
                       "    RTDETR_PATIENCE = 14\n"
                       "    RTDETR_CLS = 2.0   # biconcavelens 検証値 (RT-DETR は 2.0 が最適)\n"
                       "    # --- ensemble (§2.71) ---\n"
                       "    ENS_DEDUP_IOU = 0.05\n"
                       "    # --- YOLO (prod と完全一致) ---")
    return src


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = _cell("markdown", TITLE)
    cells[1]["source"] = patch_config(_source(cells[1])).splitlines(keepends=True)
    # cell2 (COCO/fold/前処理) と cell4 (refiner 定義) と cell5 (評価関数) は無改変
    cells[3] = _cell("code", CELL3)
    cells.insert(6, _cell("code", CELL_ENS))
    cells[7] = _cell("code", CELL_RUN)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {OUT.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
