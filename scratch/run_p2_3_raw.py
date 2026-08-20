"""P2-3 の推論キャッシュを refiner なしで採点する。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scratch.run_p2_imgsz_raw_pq import (  # noqa: E402
    BASELINE_FOLD0_RAW,
    CONF,
    OUT,
    RULES,
    load_gt,
    score_cache,
)

CACHE = Path(os.environ.get("TEMP", "C:/Temp")) / "solar_p2_3" / "oof_npz"


def main() -> int:
    if not CACHE.exists():
        print(f"P2-3 cache が無い: {CACHE}")
        return 1
    file_to_ids, gt_rles, fold_of_file = load_gt()
    rows = []
    for rule in RULES:
        print(f"scoring p2_3 {rule}", flush=True)
        row = score_cache(CACHE, file_to_ids, gt_rles, fold_of_file, rule)
        row["detector"] = "v30w_infer1536"
        row["baseline_fold0_raw_conf_first"] = BASELINE_FOLD0_RAW
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "scope": "fold0 236観測、V30 重み・推論 imgsz 1536、refiner なし",
        "table": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p2_3_raw_yolo_fold0.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
