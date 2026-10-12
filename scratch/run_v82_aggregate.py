"""v82 soft/hard_union refiner の manifest を集計し、採用ゲートを判定する。

ノートブックが書く `v82_{arm}_fold{f}_s{seed}_manifest.json` には fold 別の
`oof_pq_by_threshold`(PQ + delta_vs_baseline) が入る。per-image counts は入って
いないため、厳密な pooled bootstrap CI が必要な場合は
`run_refiner_pq_allfold.py --ckpt-pattern ...` で checkpoint を再評価する。

ゲート (PLAN §7.2.5):
  採用 = soft − hard_record が 3fold・両seed で正、かつプールした効果が一貫。
  中間解釈 = hard_union が soft と同等なら「クラスタ化」が効いた、soft のみ正なら
  「soft であること」が効いた。両方負なら教師定義レーンを閉じる。

使い方:
    python scratch/run_v82_aggregate.py [--drive "G:/マイドライブ"] [--threshold 0.45]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = (0, 1, 2)
SEEDS = (3407, 8407)
ARMS = ("soft", "hard_union")
MAIN_THRESHOLD = 0.45  # §7.2.4: 主結果は 0.45、他は補助掃引


def load_manifests(artifacts: Path) -> list[dict]:
    manifests = []
    for path in sorted(artifacts.glob("v82_*_fold*_s*_manifest.json")):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            print(f"警告: {path.name} を読めません ({error})")
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=Path, default=Path("G:/マイドライブ"))
    parser.add_argument("--threshold", type=float, default=MAIN_THRESHOLD)
    parser.add_argument("--out", type=Path, default=Path("output/v82"))
    args = parser.parse_args()

    artifacts = args.drive / "solar_filament_v30" / "artifacts"
    manifests = load_manifests(artifacts)
    expected = len(ARMS) * len(FOLDS) * len(SEEDS)
    print({"manifest": len(manifests), "expected": expected})

    rows = []
    for m in manifests:
        by_t = m.get("oof_pq_by_threshold", {})
        for t_str, entry in by_t.items():
            rows.append({
                "arm": m["arm"], "fold": m["fold"], "seed": m["seed"],
                "threshold": float(t_str), "pq": entry["pq"],
                "delta_vs_baseline": entry["delta_vs_baseline"],
                "baseline_pq": m.get("baseline_pq"),
                "best_crop_dice": m.get("best_crop_dice"),
            })
    if not rows:
        print("manifest が0件 — ノートブック未完了。判定不能。")
        return 1
    frame = pd.DataFrame(rows)

    # 主結果: 主閾値での (arm, seed, fold) 全セル
    main_rows = frame[np.isclose(frame["threshold"], args.threshold)]
    pivot = main_rows.pivot_table(index=["arm", "fold"], columns="seed",
                                  values="delta_vs_baseline")
    pivot["min_delta"] = pivot.min(axis=1)
    pivot["mean_delta"] = pivot.mean(axis=1)
    print(f"\n=== delta_vs_baseline @ τ={args.threshold} ===")
    print(pivot.round(4).to_string())

    # ゲート判定: soft が 3fold × 両seed で delta > 0
    verdict = {}
    for arm in ARMS:
        cells = main_rows[main_rows["arm"] == arm]
        complete = len(cells) == len(FOLDS) * len(SEEDS)
        all_positive = bool((cells["delta_vs_baseline"] > 0).all()) if len(cells) else False
        mean_delta = float(cells["delta_vs_baseline"].mean()) if len(cells) else float("nan")
        # fold ごとの両seed平均が正か(3fold一貫性)
        per_fold = cells.groupby("fold")["delta_vs_baseline"].mean()
        fold_consistent = bool((per_fold > 0).all()) if len(per_fold) == len(FOLDS) else False
        verdict[arm] = {
            "cells": int(len(cells)), "complete": complete,
            "all_cells_positive": all_positive,
            "fold_consistent": fold_consistent,
            "mean_delta": mean_delta,
        }
        if complete:
            print(f"{arm}: {'全セル正' if all_positive else '負セルあり'} "
                  f"meanΔ={mean_delta:+.4f} fold一貫={fold_consistent}")
        else:
            print(f"{arm}: {len(cells)}/{len(FOLDS) * len(SEEDS)} セルのみ - 判定保留")

    # 補助掃引: 全閾値での mean delta 表
    sweep = (frame.groupby(["arm", "threshold"])["delta_vs_baseline"]
             .agg(["mean", "min", "count"]).round(4))
    print("\n=== 補助: 閾値別 delta (mean/min/count) ===")
    print(sweep.to_string())

    # 中間解釈: hard_union vs soft の直接比較 (paired fold×seed)
    merged = None
    if {"soft", "hard_union"} <= set(main_rows["arm"]):
        s = main_rows[main_rows["arm"] == "soft"].set_index(["fold", "seed"])
        h = main_rows[main_rows["arm"] == "hard_union"].set_index(["fold", "seed"])
        common = s.index.intersection(h.index)
        if len(common):
            merged = pd.DataFrame({
                "soft_pq": s.loc[common, "pq"], "hard_union_pq": h.loc[common, "pq"],
            })
            merged["soft_minus_union"] = merged["soft_pq"] - merged["hard_union_pq"]
            print("\n=== soft − hard_union (paired) ===")
            print(merged.round(4).to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "main_threshold": args.threshold,
        "manifests_seen": len(manifests), "expected": expected,
        "verdict": verdict,
        "per_cell": json.loads(main_rows.round(6).to_json(orient="records")),
        "soft_minus_hard_union": (json.loads(merged.round(6).to_json(orient="index"))
                                  if merged is not None else None),
        "gate": "採用 = soft が 3fold×両seed で delta>0 (pooled CI は再評価で確認)",
    }
    (args.out / "v82_aggregate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out / 'v82_aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
