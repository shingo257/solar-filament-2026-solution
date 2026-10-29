"""v68 磁気図補助特徴プローブ — selector-v3 として v64 ckpt を再利用して評価。

v64 で静的特徴selectorは棄却されたが、原因は「特徴がconfの再表現に留まった」
ことだった。磁気図は**パイプラインに無い物理情報**(filamentは分極反転線=PIL
上に存在する)を持ち込む。v64 の収集済み提案(fold2 val235/trainA236/trainB236)
に磁気図由来特徴を追加して selector-v3 を同じプロトコルで評価する。

## 手順

1. 観測時刻+サイトから最近傍の GONG `bqr` 磁気図(180x180, ~10分cadence)をDL
2. H-α画像のdisk(中心+半径)に合わせて磁気図を 2048x2048 へアフィン写像
   (bqrはdisk内接の標準製品: 中心≡画像中心、半径≈縁までの推定半径)
3. sanity check: **GT filamentの重心→PIL距離がランダム画素より小さいこと**
   (レジストレーションが合っていればfilamentはPIL近傍に乗るはず)
4. 提案ごとに磁気図特徴(mag_mean/abs, pil_dist, pil_cover, mag_grad)を追加
5. v64と同一プロトコル: train-AでHistGBM学習 → train-Bでgrid選定 →
   val適用。旧しきい値(+0.005/0.002)は履歴診断に限り、科学的受入は保留

現在のv64は凍結契約で検証した提案をメモリ内だけに保持し、旧pickleを出力しない。
この旧キャッシュ用プローブは既定で実行を拒否する。互換性・入力来歴は未検証で、
明示的に実行しても科学的受入・採用判定には使用できない。
履歴調査に限る明示実行(GPU不要、Drive読み取りとGONGへのHTTP取得):
    python scratch/v68_magnetogram_probe.py --legacy-exploratory
"""


from __future__ import annotations


import json
import pickle
import time
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy.ndimage import (
    distance_transform_edt,
)
from sklearn.ensemble import HistGradientBoostingClassifier


try:
    from .v68_magnetogram_probe_impl import (
        CONF_GRID,
        DRIVE,
        FEATURE_KEYS,
        IMG,
        MAG_DIR,
        MAG_KEYS,
        MANIFEST,
        PKEEP_GRID,
        PREPROC,
        SPLIT_SEED,
        decode_rle_of,
        ensure_magnetogram,
        fit_disk,
        load_mag_array,
        load_rows,
        log,
        mag_features,
        mag_to_image_grid,
        pil_map,
    )
except ImportError:  # Direct `python scratch/...` or spec_from_file_location loading.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v68_magnetogram_probe_impl import (
        CONF_GRID,
        DRIVE,
        FEATURE_KEYS,
        IMG,
        MAG_DIR,
        MAG_KEYS,
        MANIFEST,
        PKEEP_GRID,
        PREPROC,
        SPLIT_SEED,
        decode_rle_of,
        ensure_magnetogram,
        fit_disk,
        load_mag_array,
        load_rows,
        log,
        mag_features,
        mag_to_image_grid,
        pil_map,
    )



__all__ = [
    "CONF_GRID",
    "DRIVE",
    "FEATURE_KEYS",
    "IMG",
    "MAG_DIR",
    "MAG_KEYS",
    "MANIFEST",
    "PKEEP_GRID",
    "PREPROC",
    "SPLIT_SEED",
    "decode_rle_of",
    "ensure_magnetogram",
    "fit_disk",
    "load_mag_array",
    "load_rows",
    "log",
    "mag_features",
    "mag_to_image_grid",
    "pil_map",
]


def legacy_exploratory_status(delta):
    return {
        "kind": "exploratory_v68_legacy_cache",
        "scientific_acceptance": False,
        "verdict": "blocked-unverified-provenance",
        "provenance_limit": "Legacy proposal/features caches lack frozen input, model and split identity; current v64 does not publish these caches.",
        "diagnostic_signal": ("positive" if delta > 0.005
                              else "noise" if abs(delta) < 0.002 else "negative"),
    }


def main(legacy_exploratory=False) -> int:
    if not legacy_exploratory:
        print("Blocked: legacy v64 cache provenance is unverified. "
              "Use --legacy-exploratory only for historical diagnostics; scientific acceptance remains blocked.")
        return 2
    MAG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    FEATS_CACHE = Path("output/v68_rows_with_feats.pkl")
    skip_to_selector = FEATS_CACHE.exists()

    if skip_to_selector:
        log("[cache] 特徴済み行をロード(1-3スキップ)")
        with open(FEATS_CACHE, "rb") as h:
            splits, mag_path, sanity_fil, sanity_rand = pickle.load(h)
        all_names = sorted({r["name"] for rows in splits.values() for r in rows})
        log({"cached_files": len(all_names)})
    else:
        log("[1/5] v64 ckpt 読み込み")
        splits = {s: load_rows(s) for s in ("val", "trainA", "trainB")}
        all_names = sorted({r["name"] for rows in splits.values() for r in rows})
        log({"files": len(all_names),
             "props": sum(len(r["props"]) for rows in splits.values() for r in rows)})

        log("[2/5] 磁気図解決 + DL")
        mag_path = {}
        for i, name in enumerate(all_names):
            pp = ensure_magnetogram(name)
            if pp:
                mag_path[name] = pp
            if i % 50 == 0:
                log(f"  {i}/{len(all_names)} 解決済み {len(mag_path)}")
        log(f"  磁気図ヒット: {len(mag_path)}/{len(all_names)}")

        log("[3/5] diskフィット + PIL写像 + 特徴計算")
        mag_cache = {}
        miss = {"no_mag": 0, "no_image": 0}

        def grid_for(name):
            if name in mag_cache:
                return mag_cache[name]
            if name not in mag_path:
                miss["no_mag"] += 1
                mag_cache[name] = None
                return None
            # cv2.imread はWindowsで非ASCIIパス(マイドライブ)を読めないため
            # imdecode 経由にする
            gray = cv2.imdecode(
                np.fromfile(str(PREPROC / name), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE)
            if gray is None:
                miss["no_image"] += 1
                mag_cache[name] = None
                return None
            cx, cy, radius = fit_disk(gray)
            mag = load_mag_array(mag_path[name])
            mag2048 = mag_to_image_grid(mag, cx, cy, radius)
            edge = pil_map(mag2048)
            dist = distance_transform_edt(~edge)
            gy, gx = np.gradient(mag2048)
            grad = np.hypot(gx, gy)
            mag_cache[name] = (mag2048, edge, dist, grad)
            return mag_cache[name]

        # partial checkpoint: {name: [feats|None per prop]} + sanity
        # 中断しても特徴計算・grid構築をやり直さない
        PARTIAL = Path("output/v68_feats_partial.pkl")
        partial = {"done": {}, "sanity": [[], []]}
        if PARTIAL.exists():
            with open(PARTIAL, "rb") as h:
                partial = pickle.load(h)
            log(f"[cache] partial resume: 済 {len(partial['done'])} ファイル")
        resumed = bool(partial["done"])

        # sanity check: hit=1提案(≈GTに一致)の重心→PIL距離 vs ランダム画素
        # (partial resume時はスキップ — grid再構築を避けるため)
        if not resumed:
            rng = np.random.default_rng(0)
            for rows in splits.values():
                for row in rows:
                    g = grid_for(row["name"])
                    if g is None:
                        continue
                    dist = g[2]
                    for pp in row["props"]:
                        if not pp.get("label_hit"):
                            continue
                        mask = decode_rle_of(pp)
                        ys, xs = np.where(mask)
                        if not len(xs):
                            continue
                        partial["sanity"][0].append(
                            float(dist[int(ys.mean()), int(xs.mean())]))
                        rx, ry = rng.integers(0, IMG, 2)
                        partial["sanity"][1].append(float(dist[ry, rx]))
                        break
        sanity_fil, sanity_rand = partial["sanity"]
        if sanity_fil:
            log({"hit提案->PIL距離(中央値)": float(np.median(sanity_fil)),
                 "ランダム->PIL距離(中央値)": float(np.median(sanity_rand)),
                 "n": len(sanity_fil)})
        else:
            log("警告: sanity check用のhit提案が取れなかった")

        done = partial["done"]
        for split, rows in splits.items():
            n_feat = 0
            for row in rows:
                name = row["name"]
                feats_list = done.get(name)
                if feats_list is None:
                    g = grid_for(name)
                    feats_list = []
                    for pp in row["props"]:
                        if g is None:
                            feats_list.append(None)
                            continue
                        mag2048, _, dist, grad = g
                        mask = decode_rle_of(pp)
                        feats_list.append(
                            mag_features(mag2048, grad, dist, mask))
                    done[name] = feats_list
                    if len(done) % 25 == 0:
                        with open(PARTIAL, "wb") as h:
                            pickle.dump(partial, h)
                for pp, feats in zip(row["props"], feats_list):
                    if feats:
                        pp["feat"].update(feats)
                        n_feat += 1
                    else:
                        for k in MAG_KEYS:
                            pp["feat"][k] = 0.0
            log(f"  {split}: {n_feat} proposals に磁気図特徴")
        log(f"  grid_forミス内訳: {miss} (有効grid {sum(1 for v in mag_cache.values() if v is not None)})")

        with open(FEATS_CACHE, "wb") as h:
            pickle.dump((splits, mag_path, sanity_fil, sanity_rand), h)
        PARTIAL.unlink(missing_ok=True)
        log(f"  特徴キャッシュ保存: {FEATS_CACHE}")

    log("[4/5] selector-v3 学習(train-A)")
    keys = FEATURE_KEYS + MAG_KEYS

    def to_xy(rows):
        X = np.asarray([[pp["feat"].get(k, 0.0) for k in keys]
                        for r in rows for pp in r["props"]])
        y = np.asarray([pp["label_hit"] for r in rows for pp in r["props"]])
        return X, y

    XA, yA = to_xy(splits["trainA"])
    sel = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                          max_depth=6, min_samples_leaf=20,
                                          random_state=SPLIT_SEED)
    sel.fit(XA, yA)
    log(f"  train-A: {len(yA)} (hit {int(yA.sum())} / fp {int((~yA.astype(bool)).sum())})")

    for rows in splits.values():
        for row in rows:
            if row["props"]:
                X = np.asarray([[pp["feat"].get(k, 0.0) for k in keys] for pp in row["props"]])
                p_keep = sel.predict_proba(X)[:, 1]
                for j, pp in enumerate(row["props"]):
                    pp["p"] = float(p_keep[j])

    def iou_matrix(truth, pred):
        if not truth or not pred:
            return np.zeros((len(truth), len(pred)))
        return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth))).T

    class PQ:
        def __init__(self):
            self.iou, self.fp, self.fn = [], 0, 0
        def add(self, m, n_gt, n_pred):
            if n_gt == 0:
                self.fp += n_pred; return
            if n_pred == 0:
                self.fn += n_gt; return
            hits = m > 0.5
            self.iou.extend(m[hits].tolist())
            self.fp += int((hits.sum(0) == 0).sum())
            self.fn += int((hits.sum(1) == 0).sum())
        def summary(self):
            tp = len(self.iou)
            den = tp + 0.5 * self.fp + 0.5 * self.fn
            return {"pq": float(sum(self.iou) / den) if den else 0.0,
                    "tp": tp, "fp": self.fp, "fn": self.fn}

    def apply_ownership(masks):
        if not masks:
            return []
        areas = np.array([m.sum() for m in masks], np.float64)
        if len(masks) >= 2:
            rles = mask_utils.encode(np.asfortranarray(np.stack(masks, -1).astype(np.uint8)))
            ar = np.asarray(mask_utils.area(rles), np.float64)
            si = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), np.float64)
            dropped = set()
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    if i in dropped or j in dropped:
                        continue
                    v = si[i, j]
                    if v <= 0:
                        continue
                    inter = v * (ar[i] + ar[j]) / (1.0 + v)
                    sm = i if ar[i] <= ar[j] else j
                    if inter / max(ar[sm], 1.0) >= 0.90:
                        dropped.add(sm)
            subset = [i for i in range(len(masks)) if i not in dropped]
        else:
            subset = list(range(len(masks)))
        occupied = np.zeros((IMG, IMG), bool)
        kept = []
        for idx in sorted(subset, key=lambda i: areas[i]):
            ex = masks[idx] & ~occupied
            if ex.sum() < 20:
                continue
            kept.append(ex); occupied |= ex
        return kept

    import glob
    tjson = glob.glob(str(DRIVE / "competition_data" / "**" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), recursive=True)[0]
    coco = json.loads(Path(tjson).read_text(encoding="utf-8"))
    ann_by_img = {}
    info_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    file_to_ids = {}
    for img in coco["images"]:
        info_by_img[img["id"]] = img
        file_to_ids.setdefault(img["file_name"], []).append(img["id"])

    def to_rle(ann, h, w):
        seg = ann["segmentation"]
        if isinstance(seg, list):
            return mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
        if isinstance(seg.get("counts"), list):
            return mask_utils.frPyObjects(seg, h, w)
        r = dict(seg)
        if isinstance(r["counts"], str):
            r["counts"] = r["counts"].encode()
        return r

    gt_rles = {iid: [to_rle(a, int(info_by_img[iid]["height"]), int(info_by_img[iid]["width"]))
                     for a in ann_by_img[iid] if not a.get("iscrowd", 0)]
               for iid in ann_by_img}

    def score(rows, conf_floor, pk):
        meter = PQ()
        for row in rows:
            kept = [pp for pp in row["props"] if pp["conf"] >= conf_floor and pp.get("p", 0) >= pk]
            preds = apply_ownership([decode_rle_of(pp) for pp in kept])
            prles = []
            if preds:
                enc = mask_utils.encode(np.asfortranarray(np.stack(preds, -1).astype(np.uint8)))
                prles = enc if isinstance(enc, list) else [enc]
            for iid in file_to_ids[row["name"]]:
                truth = gt_rles[iid]
                meter.add(iou_matrix(truth, prles), len(truth), len(prles))
        return meter.summary()

    log("[5/5] trainB grid -> 凍結点 -> val適用")
    grid = []
    best = None
    for cf in CONF_GRID:
        for pk in PKEEP_GRID:
            sc = score(splits["trainB"], cf, pk)
            grid.append({"conf": cf, "pk": pk, **sc})
            if best is None or sc["pq"] > best["pq"]:
                best = {"conf": cf, "pk": pk, **sc}
            log(f"  trainB conf={cf} pk={pk}: PQ {sc['pq']:.4f}")
    log(f"trainB 最良: conf={best['conf']} pk={best['pk']} (PQ {best['pq']:.4f})")

    base_val = score(splits["val"], 0.26, 0.0)
    frozen_val = score(splits["val"], best["conf"], best["pk"])
    delta = frozen_val["pq"] - base_val["pq"]
    status = legacy_exploratory_status(delta)
    manifest = {"baseline_val": base_val, "frozen_point": {"conf": best["conf"], "pk": best["pk"]},
                "frozen_val": frozen_val, "delta_pq": delta, **status,
                "magnetogram_coverage": len(mag_path) / len(all_names),
                "sanity": {"hit_pil_median": float(np.median(sanity_fil)) if sanity_fil else None,
                           "rand_pil_median": float(np.median(sanity_rand)) if sanity_rand else None},
                "trainB_grid": grid, "elapsed_min": round((time.time() - t0) / 60, 1)}
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    log(f"val baseline(0.26/pk0): {base_val}")
    log(f"val selector-v3(凍結): {frozen_val}")
    log(f"ΔPQ {delta:+.4f} -> {status['verdict']} (historical diagnostic only)")
    log(f"manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-exploratory", action="store_true",
                        help="Allow unverified historical cache diagnostics; never grants scientific acceptance")
    raise SystemExit(main(parser.parse_args().legacy_exploratory))
