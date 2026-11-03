# v97d 探索ノート生成 — 出荷グローバル isotonic 写像の fold2 検証
# dedup030 カーネルと同じ写像 (fold2 全候補 fit) を insample / half-cross で評価。
# 使い方: python scratch/build_v97d_globalmap_colab.py
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "solar_filament_v97_rtdetr_ensemble_colab.ipynb"
DST = ROOT / "solar_filament_v97d_globalmap_colab.ipynb"

CELL_EXPLORE = '''# [7/7] v97d: グローバル isotonic 写像 (dedup030 カーネル出荷構成) の fold2 検証
# NOTE: 探索であり採用ゲートではない。LB probe の局所見積もりを正直に出す。
def save_manifest(manifest):
    CFG.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

CONTRACT = {"experiment_id": "v97d_globalmap", "fold": CFG.FOLD,
            "protocol": "validation of shipped global isotonic map (dedup030 kernel)",
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

_cache = torch.load(CFG.CAND_CACHE, map_location="cpu", weights_only=False)
yolo_cands = _cache["yolo"]
half_a, half_b = crossfit_halves(valid_files)

# 1) グローバル写像を fold2 全候補で再 fit (Drive artifacts/v97_global_iso.pt と同一の手順)
iso_global = fit_isotonic(yolo_cands, valid_files)
xt, yt = iso_global.X_thresholds_, iso_global.y_thresholds_
lam = 0.5 * prod_pq
conf_star = float(xt[yt > lam].min()) if (yt > lam).any() else float("inf")
print(f"global iso: n_thresholds={len(xt)} lam={lam:.4f} 実効conf*={conf_star:.4f}")

# Drive 側の保存済み写像と一致するか sanity (出荷 npz の再現性証明)
_saved = torch.load(CFG.ARTIFACT_ROOT / "v97_global_iso.pt", map_location="cpu", weights_only=False)
_same = (len(_saved["yolo"].X_thresholds_) == len(xt)
         and bool((abs(_saved["yolo"].X_thresholds_ - xt) < 1e-9).all())
         and bool((abs(_saved["yolo"].y_thresholds_ - yt) < 1e-9).all()))
print("saved iso 一致:", _same)
assert _same

# 2) 写像の適用方式2種: insample (fit と同じ file 群 = 楽観側) /
#    half-cross (反対 half で fit した写像を適用 = test 状況に近い honest 側)
half_map = {"yolo": {}}
iso_a = fit_isotonic(yolo_cands, half_a)
iso_b = fit_isotonic(yolo_cands, half_b)
for n in half_b: half_map["yolo"][n] = iso_a
for n in half_a: half_map["yolo"][n] = iso_b
maps = {"insample": {"yolo": {n: iso_global for n in valid_files}},
        "halfcross": half_map}

res = manifest["results"].setdefault("globalmap", {})
pools = {"yolo": yolo_cands}
for mode, cal in maps.items():
    for d in (0.10, 0.20, 0.30, 0.50):
        key = f"{mode}_d{d}"
        if key in res:
            continue
        s = ensemble_eval_cached(pools, cal, d, lam, valid_files,
                                 tag=f"glob_{mode}_d{d}_l{lam:.4f}")
        s["delta_vs_prod"] = s["pq"] - prod_pq
        s["dedup_iou"] = d
        res[key] = s
        print("GLOB", key, "Δ", round(s["delta_vs_prod"], 4), "pq", round(s["pq"], 4), flush=True)
        save_manifest(manifest)

# 3) 参照点: 出荷構成と同じ写像・同じ受理での単純 conf 閾値との差を確認
#    (dedup030 ≈ dedup0.30 + conf>conf* — dedup が効いているかの分離)
res.setdefault("conf_only", {})
for c in (0.238, 0.20):
    key = f"conf{c}"
    if key in res["conf_only"]:
        continue
    s = solo_eval(yolo_cands, (c,), valid_files)[f"{c}"]
    s["delta_vs_prod"] = s["pq"] - prod_pq
    res["conf_only"][key] = s
    print("CONF_ONLY", key, "Δ", round(s["delta_vs_prod"], 4), flush=True)
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
        "v97d_globalmap_colab_manifest.json")
    s0 = nb["cells"][0]["source"]
    s0 = "".join(s0) if isinstance(s0, list) else s0
    nb["cells"][0]["source"] = (
        "# v97d — グローバル isotonic 写像 (dedup030 出荷構成) の fold2 検証\n\n"
        "探索用。LB probe に出した写像の honest/insample 両側の局所見積もり。\n\n" + s0)
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
