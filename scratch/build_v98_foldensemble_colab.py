# v98 探索ノート生成 — fold0/1/2 YOLO 候補プールの calibrated acceptance + dedup 統合
# v97 で RT-DETR は校正設計差で寄与ゼロだったが、fold モデルは同一アーキテクチャ・
# 同分布のため多様性が活きる可能性。fold0/1 は fold2 を未学習 → fold2 で honest 評価可。
# 使い方: python scratch/build_v98_foldensemble_colab.py
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "solar_filament_v97_rtdetr_ensemble_colab.ipynb"
DST = ROOT / "solar_filament_v98_foldensemble_colab.ipynb"

CELL_EXPLORE = '''# [7/7] v98: fold0+fold1+fold2 YOLO 候補プールの統合 (探索 — 採用ゲートではない)
# NOTE: dedup030 の局所 +0.0075 が LB で実証されなかったため、受理パイプラインの
# 多様性軸を fold モデルで試す。fold0/1 の重みは fold2 を一切見ていないため
# fold2 上の評価は honest。RT-DETR と違い同一アーキテクチャ・同一画像分布なので
# 共有 isotonic 写像がそのまま使える想定。
def save_manifest(manifest):
    CFG.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CFG.MANIFEST.parent, suffix=".tmp", delete=False) as h:
        h.write(payload); h.flush(); os.fsync(h.fileno()); tmp = h.name
    os.replace(tmp, CFG.MANIFEST)

YOLO_F0 = CFG.DRIVE_ROOT / "runs" / "yolo_fold_0" / "weights" / "best.pt"
YOLO_F1 = CFG.DRIVE_ROOT / "runs" / "yolo_fold_1" / "weights" / "best.pt"
V98_CAND_CACHE = CFG.ARTIFACT_ROOT / "v98_candidates.pt"

CONTRACT = {"experiment_id": "v98_foldensemble", "fold": CFG.FOLD,
            "protocol": "exploratory multi-fold candidate pooling (calibrated acceptance + dedup)",
            "train_folds": list(CFG.TRAIN_FOLDS), "ultralytics": version("ultralytics"),
            "annotations_sha256": file_sha256(TRAIN_JSON),
            "folds_sha256": folds_sha256, "refiner_sha256": REFINER_SHA256,
            "prod_yolo_sha256": file_sha256(CFG.PROD_YOLO),
            "yolo_f0_sha256": file_sha256(YOLO_F0), "yolo_f1_sha256": file_sha256(YOLO_F1)}
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

# 1) fold0/fold1 の候補収集 (conf floor 0.02、f2 と同じ収集条件)。破損 cache は除去して再生成。
f0_cands = f1_cands = None
if V98_CAND_CACHE.exists():
    try:
        _v98 = torch.load(V98_CAND_CACHE, map_location="cpu", weights_only=False)
        f0_cands, f1_cands = _v98["f0"], _v98["f1"]
        print("fold0/1 candidates: Drive cache から復元")
    except Exception:
        V98_CAND_CACHE.unlink(missing_ok=True)
if f0_cands is None:
    f0_cands = gather_candidates(YOLO_F0, "yolo", valid_files)
    f1_cands = gather_candidates(YOLO_F1, "yolo", valid_files)
    _part_save(V98_CAND_CACHE, {"f0": f0_cands, "f1": f1_cands})
    manifest["results"]["fold_candidates"] = {
        "f0": {n: len(c) for n, c in f0_cands.items()},
        "f1": {n: len(c) for n, c in f1_cands.items()}}
    save_manifest(manifest)

# 2) solo sanity: fold0/fold1 単体の fold2 性能 (同世代モデルの素の差)
res = manifest["results"].setdefault("solo", {})
for tag, cands in (("f0", f0_cands), ("f1", f1_cands)):
    if tag in res:
        continue
    res[tag] = solo_eval_cached(cands, (CFG.PRIMARY_CONF,), valid_files, tag=f"v98_solo_{tag}")
    res[tag][f"{CFG.PRIMARY_CONF}"]["delta_vs_prod"] = res[tag][f"{CFG.PRIMARY_CONF}"]["pq"] - prod_pq
    print("SOLO", tag, "Δ", round(res[tag][f"{CFG.PRIMARY_CONF}"]["delta_vs_prod"], 4), flush=True)
    save_manifest(manifest)

# 3) プール統合評価 — 同じ候補辞書に det 違いでなく同系統の候補を concat して
#    共有 global isotonic (half-cross honest) + dedup + λ 受理
def merge_pools(*pools):
    merged = {}
    for n in valid_files:
        merged[n] = [c for p in pools for c in p.get(n, [])]
    return merged

lam = 0.5 * prod_pq
res = manifest["results"].setdefault("pooled", {})
POOLS = {"f2": [yolo_cands], "f2f0": [yolo_cands, f0_cands],
         "f2f1": [yolo_cands, f1_cands], "f0f1": [f0_cands, f1_cands],
         "f2f0f1": [yolo_cands, f0_cands, f1_cands]}
for pname, plist in POOLS.items():
    pooled = merge_pools(*plist)
    # half-cross isotonic (pooled 候補全体で fit — 同系統なので共有写像)
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
                                 tag=f"v98_{pname}_d{d}_l{lam:.4f}")
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
        "v98_foldensemble_colab_manifest.json")
    s0 = nb["cells"][0]["source"]
    s0 = "".join(s0) if isinstance(s0, list) else s0
    nb["cells"][0]["source"] = (
        "# v98 — fold0+fold1+fold2 YOLO 候補プール統合 (探索)\n\n"
        "⚠️ **汚染警告**: `yolo_fold_0/1` は fold_k 以外で学習する慣例のため **fold2 を学習済み**。\n"
        "fold2 上の評価は in-sample で、solo で +0.02〜+0.04 の記憶プレミアムを実測済み。\n"
        "**このノートの pooled 結果は提出判断の証拠に使わない** (RESEARCH_LOG §2.74)。\n"
        "honest 評価は v98b (fold-only 学習モデル) を参照。\n\n" + s0)
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
