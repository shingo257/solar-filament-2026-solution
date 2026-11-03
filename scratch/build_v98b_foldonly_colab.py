# v98b 探索ノート生成 — fold0-only / fold1-only YOLO を新規学習して honest pooled 評価
# v98 の前提誤認: yolo_fold_k は fold_k 以外で学習済み → fold0/1 は fold2 を学習済みで
# pooled 評価は汚染。honest 検証には fold2 を一切見ないモデルが必要 → 各 fold 単独
# (train の ~1/3 データ) で学習し、val も fold 内分割にして fold2 完全非接触を保つ。
# 使い方: python scratch/build_v98b_foldonly_colab.py
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "solar_filament_v97_rtdetr_ensemble_colab.ipynb"
DST = ROOT / "solar_filament_v98_foldensemble_honest_colab.ipynb"

CELL_EXPLORE = '''# [7/7] v98b: fold0-only / fold1-only YOLO 学習 → honest pooled 評価 (探索)
# ⚠️ v98 の pooled 結果は汚染 (yolo_fold_0/1 は fold2 を学習済み)。このノートの
#    onlyfold モデルだけが fold2 完全非接触 = honest 検証に使える唯一の証拠。
def save_manifest(manifest):
    CFG.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

V98_RUNS = CFG.DRIVE_ROOT / "runs" / "v98b_foldonly"
V98_YOLO_ROOT = CFG.DRIVE_ROOT / "yolo_datasets" / "v98b_foldonly"
V98_CAND_CACHE = CFG.ARTIFACT_ROOT / "v98b_candidates.pt"

CONTRACT = {"experiment_id": "v98b_foldonly", "fold": CFG.FOLD,
            "protocol": "exploratory — HONEST multi-model pooling via fold-only detectors (fold2-blind training AND val)",
            "train_folds": "fold0-only + fold1-only (val = 20% of same fold; fold2 never seen)",
            "ultralytics": version("ultralytics"),
            "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO),
            "WARNING": "v98 results are contaminated — fold0/1 weights saw fold2. Do NOT cite them as honest evidence."}
manifest = json.loads(CFG.MANIFEST.read_text(encoding="utf-8")) if CFG.MANIFEST.exists() else {}
if manifest and manifest.get("contract") != CONTRACT:
    raise ValueError("既存 manifest と入力契約が違う。履歴保持のため別パスを指定してください")
manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}
save_manifest(manifest)

if "prod" not in manifest["results"]:
    manifest["results"]["prod"] = {"weights": str(CFG.PROD_YOLO), "by_conf": evaluate_yolo(CFG.PROD_YOLO, "prod")}
    save_manifest(manifest)
prod_pq = manifest["results"]["prod"]["by_conf"][str(CFG.PRIMARY_CONF)]["pq"]

# 1) fold 単独データセット: train = fold 内ファイルの全 labeler variant、
#    val = 同 fold の sorted 5 番目毎 (20%) — fold2 は train/val のどちらにも入らない
def polygon_line(ann, width, height):
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    polygon = max(polygons, key=len)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.reshape(-1))

def write_variant(fold_root, split, file_name, image_id, suffix):
    info = images[image_id]
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{stem}.jpeg"
    label_out = fold_root / "labels" / split / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    if not image_out.exists():
        shutil.copy2((CFG.PREPROC_DIR / file_name).resolve(), image_out)
    lines = [polygon_line(ann, info["width"], info["height"])
             for ann in annotations[image_id] if not ann.get("iscrowd", 0)]
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\\n".join(line for line in lines if line), encoding="utf-8")

def build_onlyfold_dataset(only_fold):
    root = V98_YOLO_ROOT / f"onlyfold_{only_fold}"
    members = sorted(n for n in train_files if fold_of_file[n] == only_fold)
    val_members = set(members[4::5])  # deterministic 20% val — fold2 非接触を保つ
    stats = {"train_samples": 0, "train_instances": 0, "val_samples": 0, "val_instances": 0}
    for file_name in tqdm(members, desc=f"onlyfold{only_fold} データセット構築"):
        if file_name in val_members:
            write_variant(root, "val", file_name, file_to_ids[file_name][0], "")
            stats["val_samples"] += 1
            stats["val_instances"] += len(annotations[file_to_ids[file_name][0]])
        else:
            for image_id in file_to_ids[file_name]:
                write_variant(root, "train", file_name, image_id, f"__labeler_{image_id}")
                stats["train_samples"] += 1
                stats["train_instances"] += len(annotations[image_id])
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: filament\\n",
                         encoding="utf-8")
    return yaml_path, stats

def train_onlyfold(only_fold):
    run_name = f"yolo_onlyfold_{only_fold}"
    run_dir = V98_RUNS / run_name
    best_path, done_path = run_dir / "weights" / "best.pt", run_dir / "TRAINING_COMPLETE"
    yaml_path, stats = build_onlyfold_dataset(only_fold)
    if done_path.exists() and best_path.exists():
        print(f"{run_name}: 学習済み checkpoint を再利用 {best_path}")
        return best_path, stats
    started = time.time()
    YOLO(CFG.YOLO_MODEL).train(
        data=str(yaml_path), epochs=CFG.YOLO_EPOCHS, imgsz=CFG.YOLO_IMGSZ,
        batch=CFG.YOLO_BATCH, device=0, workers=4, patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED + only_fold, deterministic=True, single_cls=True,
        mask_ratio=CFG.YOLO_MASK_RATIO, overlap_mask=False, amp=True, cache="disk",
        cls=CFG.YOLO_CLS,  # prod と同一 (0.5)
        optimizer="AdamW", lr0=7e-4, lrf=0.05, weight_decay=5e-4, warmup_epochs=3,
        degrees=180, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.18, mosaic=0.20, close_mosaic=10,
        project=str(V98_RUNS), name=run_name, exist_ok=True,
    )
    stats["train_minutes"] = round((time.time() - started) / 60, 1)
    done_path.write_text("ok", encoding="utf-8")
    return best_path, stats

ds = manifest["datasets"]
for of in (0, 1):
    key = f"onlyfold_{of}"
    if key not in ds:
        w, stats = train_onlyfold(of)
        ds[key] = {**stats, "weights": str(w), "weights_sha256": file_sha256(w)}
        save_manifest(manifest)
W0, W1 = ds["onlyfold_0"]["weights"], ds["onlyfold_1"]["weights"]

# 2) 候補収集 (conf floor 0.02 — prod/dedup030 と同条件)
_cache = torch.load(CFG.CAND_CACHE, map_location="cpu", weights_only=False)
yolo_cands = _cache["yolo"]
f0_cands = f1_cands = None
if V98_CAND_CACHE.exists():
    try:
        _v = torch.load(V98_CAND_CACHE, map_location="cpu", weights_only=False)
        f0_cands, f1_cands = _v["f0"], _v["f1"]
        print("onlyfold candidates: Drive cache から復元")
    except Exception:
        V98_CAND_CACHE.unlink(missing_ok=True)
if f0_cands is None:
    f0_cands = gather_candidates(W0, "yolo", valid_files)
    f1_cands = gather_candidates(W1, "yolo", valid_files)
    _part_save(V98_CAND_CACHE, {"f0": f0_cands, "f1": f1_cands})
    manifest["results"]["onlyfold_candidates"] = {
        "f0": {n: len(c) for n, c in f0_cands.items()},
        "f1": {n: len(c) for n, c in f1_cands.items()}}
    save_manifest(manifest)

# 3) solo sanity — 1/3 データ学習なので prod (2/3) より弱い想定 (~0.40-0.43 前後)。
#    prod 越えなら依然として汚染か記録不備の兆候 = fail-closed で中止する。
res = manifest["results"].setdefault("solo", {})
for tag, cands in (("f0only", f0_cands), ("f1only", f1_cands)):
    if tag in res:
        continue
    res[tag] = solo_eval_cached(cands, (CFG.PRIMARY_CONF,), valid_files, tag=f"v98b_solo_{tag}")
    res[tag][f"{CFG.PRIMARY_CONF}"]["delta_vs_prod"] = res[tag][f"{CFG.PRIMARY_CONF}"]["pq"] - prod_pq
    print("SOLO", tag, "Δ", round(res[tag][f"{CFG.PRIMARY_CONF}"]["delta_vs_prod"], 4), flush=True)
    save_manifest(manifest)
for tag in ("f0only", "f1only"):
    assert res[tag][f"{CFG.PRIMARY_CONF}"]["pq"] < prod_pq + 0.01, \\
        f"{tag} が prod を不審に上回る — fold2 接触の兆候。結果を信用せず中止"

# 4) pooled 評価 — half-cross 共有 isotonic + dedup + λ (v98 と同型、唯一の差は honest モデル)
def merge_pools(*pools):
    return {n: [c for p in pools for c in p.get(n, [])] for n in valid_files}

half_a, half_b = crossfit_halves(valid_files)
lam = 0.5 * prod_pq
res = manifest["results"].setdefault("pooled", {})
POOLS = {"f2": [yolo_cands], "f2f0": [yolo_cands, f0_cands],
         "f2f1": [yolo_cands, f1_cands], "f0f1": [f0_cands, f1_cands],
         "f2f0f1": [yolo_cands, f0_cands, f1_cands]}
for pname, plist in POOLS.items():
    pooled = merge_pools(*plist)
    iso_a = fit_isotonic(pooled, half_a)
    iso_b = fit_isotonic(pooled, half_b)
    cal = {"yolo": {}}
    for n in half_b: cal["yolo"][n] = iso_a
    for n in half_a: cal["yolo"][n] = iso_b
    for d in (0.10, 0.30):
        key = f"{pname}_d{d}"
        if key in res:
            continue
        s = ensemble_eval_cached({"yolo": pooled}, cal, d, lam, valid_files,
                                 tag=f"v98b_{pname}_d{d}_l{lam:.4f}")
        s["delta_vs_prod"] = s["pq"] - prod_pq
        s["pool"] = pname
        res[key] = s
        print("POOL", key, "Δ", round(s["delta_vs_prod"], 4), "pq", round(s["pq"], 4), flush=True)
        save_manifest(manifest)

manifest["runtime"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
save_manifest(manifest)
print("manifest:", CFG.MANIFEST)
print(json.dumps({k: round(v.get("delta_vs_prod", 0), 4) for k, v in res.items() if isinstance(v, dict)}, ensure_ascii=False, indent=1)[:1500])
'''


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    s1 = nb["cells"][1]["source"]
    s1 = "".join(s1) if isinstance(s1, list) else s1
    assert "v97_rtdetr_ensemble_colab_manifest.json" in s1
    nb["cells"][1]["source"] = s1.replace(
        "v97_rtdetr_ensemble_colab_manifest.json",
        "v98b_foldonly_colab_manifest.json")
    s0 = nb["cells"][0]["source"]
    s0 = "".join(s0) if isinstance(s0, list) else s0
    nb["cells"][0]["source"] = (
        "# v98b — fold0-only/fold1-only YOLO 学習 → honest pooled 評価 (探索)\n\n"
        "⚠️ v98 の pooled 結果は汚染: yolo_fold_0/1 は fold2 を学習済みであり、"
        "fold2 上の評価は in-sample (solo で +0.02〜+0.04 の記憶プレミアム実測)。"
        "v98 の数値は提出判断の証拠に使わないこと。\n\n"
        "このノートの onlyfold モデルは fold 内のみで学習・val も fold 内 20% で、"
        "fold2 に train/val 両面で完全非接触 — pooled 評価が honest。\n\n" + s0)
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
