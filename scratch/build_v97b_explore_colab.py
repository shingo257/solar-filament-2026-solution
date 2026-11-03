# v97b 探索ノート生成 — v97 ipynb を複製し、manifest 差替え + cell7 を探索ロジックに置換
# 使い方: python scratch/build_v97b_explore_colab.py
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "solar_filament_v97_rtdetr_ensemble_colab.ipynb"
DST = ROOT / "solar_filament_v97b_explore_colab.ipynb"

CELL_EXPLORE = '''# [7/7] v97b 探索: prod per-file preds + rescue policy + λグリッド + per-file Δ
# NOTE: 本セルは探索であり §2.71 の採用ゲートではない。
def save_manifest(manifest):
    CFG.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

CONTRACT = {"experiment_id": "v97b_explore", "fold": CFG.FOLD,
            "protocol": "exploratory neighborhood of §2.71 (post-gate, non-adoption)",
            "train_folds": list(CFG.TRAIN_FOLDS), "ultralytics": version("ultralytics"),
            "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO)}
manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8")) if CFG.MANIFEST.exists() else {}
if manifest and manifest.get("contract") != CONTRACT:
    raise ValueError("既存 manifest と入力契約が違う。履歴保持のため別パスを指定してください")
manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}
save_manifest(manifest)

if "prod" not in manifest["results"]:
    manifest["results"]["prod"] = {"weights": str(CFG.PROD_YOLO), "by_conf": evaluate_yolo(CFG.PROD_YOLO, "prod")}
    save_manifest(manifest)
prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]

rtdetr_weights = CFG.RUNS_ROOT / "rtdetr_fold2" / "weights" / "best.pt"
assert rtdetr_weights.exists(), rtdetr_weights

_cache = torch.load(CFG.CAND_CACHE, map_location="cpu", weights_only=False)
yolo_cands, rtdetr_cands = _cache["yolo"], _cache["rtdetr"]
cand_pools = {"yolo": yolo_cands, "rtdetr": rtdetr_cands}
half_a, half_b = crossfit_halves(valid_files)
_cm = torch.load(CFG.CAL_CACHE, map_location="cpu", weights_only=False)
cal_maps_cross, cal_maps_ins = _cm["cross"], _cm["ins"]

lam = 0.5 * prod_pq
res = manifest["results"].setdefault("explore", {})

@torch.no_grad()
def prod_pred_rles(files, tag="prod_primary"):
    """prod emission の per-file pred_rles を parts キャッシュで構築。"""
    part = _part_path(tag)
    done = _part_load(part)
    model = YOLO(str(CFG.PROD_YOLO))
    for name in tqdm(files, desc="prod preds"):
        if name in done:
            continue
        result = model.predict(str(CFG.PREPROC_DIR / name), imgsz=CFG.YOLO_IMGSZ,
                               conf=max(0.08, CFG.PRIMARY_CONF * 0.75), iou=CFG.PREDICT_IOU,
                               agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, _ = result_masks(result)
        confs = result.boxes.conf.detach().cpu().numpy().tolist() if result.boxes is not None else []
        proposals = [(s, m) for s, m in zip(confs, masks) if s >= CFG.PRIMARY_CONF]
        image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
        refined = [refine_proposal(image, m) for _, m in proposals]
        preds = apply_ownership([m for m in refined if m.any()])
        done[name] = (mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8)))
                      if preds else [])
        _part_save(part, done)
    del model; torch.cuda.empty_cache()
    return done

@torch.no_grad()
def rescue_eval(lam_hi, dedup_iou, files, prod_preds, tag):
    """prod emission を土台に RT-DETR の cal>lam_hi 候補のみ追加救済 (prod 侵食なし)。"""
    part = _part_path(f"rescue_{tag}")
    done = _part_load(part)
    for name in tqdm(files, desc=f"rescue[{tag}]"):
        if name in done:
            continue
        base_rles = prod_preds[name]
        base_masks = [mask_utils.decode(r).astype(bool) for r in base_rles]
        image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
        iso = cal_maps_cross["rtdetr"][name]
        kept = list(base_masks)
        kept_rles = list(base_rles)
        for conf, rle in sorted(rtdetr_cands.get(name, []), key=lambda t: -t[0]):
            cal = float(iso.predict([conf])[0])
            if cal <= lam_hi:
                continue
            if kept_rles:
                ious = mask_utils.iou([rle], kept_rles, [0] * len(kept_rles))[0]
                if (ious > dedup_iou).any():
                    continue
            rm = refine_proposal(image, mask_utils.decode(rle).astype(bool))
            if not rm.any():
                continue
            occ = np.zeros(rm.shape, dtype=bool)
            for bm in kept:
                occ |= bm
            rm2 = rm & ~occ
            if int(rm2.sum()) < CFG.MIN_AREA:
                continue
            kept.append(rm2)
            kept_rles.append(mask_utils.encode(np.asfortranarray(rm2.astype(np.uint8))))
        done[name] = kept_rles
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
    return s

if "prod_preds" not in res:
    res["prod_preds"] = "building"
    save_manifest(manifest)
prod_preds = prod_pred_rles(valid_files)
res["prod_preds"] = {n: len(v) for n, v in prod_preds.items()}
save_manifest(manifest)

# 1) rescue policy sweep (prod 土台 + RT-DETR 救済)
res.setdefault("rescue", {})
for lh in (0.30, 0.35, 0.40):
    key = f"lam{lh}_d0.05"
    if key in res["rescue"]:
        continue
    s = rescue_eval(lh, CFG.ENS_DEDUP_IOU, valid_files, prod_preds, tag=f"{lh}_d{CFG.ENS_DEDUP_IOU}")
    s["delta_vs_prod"] = s["pq"] - prod_pq
    res["rescue"][key] = s
    print("RESCUE", key, {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})
    save_manifest(manifest)

# 2) ensemble λ グリッド (cross-fit 校正・dedup0.05 固定)
res.setdefault("lambda_grid", {})
for m in (0.65, 0.8, 0.9, 1.1, 1.25):
    key = str(round(m * lam, 4))
    if key in res["lambda_grid"]:
        continue
    s = ensemble_eval_cached(cand_pools, cal_maps_cross, CFG.ENS_DEDUP_IOU, m * lam,
                             valid_files, tag=f"cross_d{CFG.ENS_DEDUP_IOU}_l{m * lam:.4f}")
    s["delta_vs_prod"] = s["pq"] - prod_pq
    res["lambda_grid"][key] = s
    print("LAM", key, "Δ", round(s["delta_vs_prod"], 4))
    save_manifest(manifest)

# 3) per-file Δ 分析 (ens@primary vs prod)
def perfile_pq(name, pred_rles):
    meter = PanopticQuality()
    for image_id in file_to_ids[name]:
        truth = gt_rles[image_id]
        meter.add_matrix(iou_matrix(truth, pred_rles), len(truth), len(pred_rles))
    return meter.summary()["pq"]

if "perfile_delta" not in res:
    ens_primary = _part_load(_part_path(f"ens_cross_d{CFG.ENS_DEDUP_IOU}_l{lam:.4f}"))
    deltas = []
    for name in valid_files:
        ep = perfile_pq(name, ens_primary.get(name, []))
        pp = perfile_pq(name, prod_preds.get(name, []))
        deltas.append({"file": name, "ens": round(ep, 4), "prod": round(pp, 4),
                       "delta": round(ep - pp, 4)})
    deltas.sort(key=lambda d: -d["delta"])
    res["perfile_delta"] = {"top_gain": deltas[:15], "top_loss": deltas[-15:],
                            "n_gain": sum(1 for d in deltas if d["delta"] > 0.001),
                            "n_loss": sum(1 for d in deltas if d["delta"] < -0.001)}
    save_manifest(manifest)

manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
save_manifest(manifest)
print("manifest:", CFG.MANIFEST)
print(json.dumps({k: (list(v.keys()) if isinstance(v, dict) else v) for k, v in res.items()}, ensure_ascii=False, indent=1)[:2000])
'''


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    s1 = nb["cells"][1]["source"]
    s1 = "".join(s1) if isinstance(s1, list) else s1
    assert "v97_rtdetr_ensemble_colab_manifest.json" in s1
    nb["cells"][1]["source"] = s1.replace(
        "v97_rtdetr_ensemble_colab_manifest.json",
        "v97b_explore_colab_manifest.json")
    s0 = nb["cells"][0]["source"]
    s0 = "".join(s0) if isinstance(s0, list) else s0
    nb["cells"][0]["source"] = (
        "# v97b 探索 — RT-DETR ensemble 近傍探索 (rescue policy・λグリッド・per-file Δ)\n\n"
        "§2.71 の採用ゲート (reject_noise) 後の**探索用**ノート。採用判定には使わない。\n\n" + s0)
    nb["cells"][7]["source"] = CELL_EXPLORE
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        ast.parse("\n".join(l for l in s.split("\n") if not l.lstrip().startswith("%")) if i == 1 else s)
    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"built {DST.name}: all cells compile")


if __name__ == "__main__":
    main()
