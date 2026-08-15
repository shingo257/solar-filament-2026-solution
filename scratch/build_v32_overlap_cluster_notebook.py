import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "solar_filament_v32_overlap_cluster_repair.ipynb"
CORE = (ROOT / "src" / "v32_overlap_cluster.py").read_text(encoding="utf-8")


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = [
    markdown(
        """# 太陽フィラメント V32 — 重複maskの同一個体統合修復

V30b（Public 0.35）は、重なった個体maskから先着maskの画素を差し引くことで提出形式を修復しました。
しかしV30の重複319組の中央値では、小さいmaskの約99%が大きいmaskに内包されていました。
これは異なる個体の単純な接触より、同一フィラメントを複数候補が予測した可能性が高い状態です。

V32では、重複を削る前に高類似maskをグラフクラスタリングし、同一候補をunionしてから個体間を排他化します。

- `V32a`: Dice 0.90以上だけを統合（第一候補・1453行）
- `V32b`: 包含率0.99以上かつDice 0.50以上を統合（1406行）
- `V32c`: Dice 0.70以上を統合（1398行）

学習・GPU・Internetは不要です。標準RAMのColab CPUで実行できます。
元の重複ありV30 CSVをDriveから読み、全RLEを再復号して重複0を確認します。
"""
    ),
    code(
        """# [1/5] Drive、入力CSV、出力先
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from google.colab import drive
from tqdm.auto import tqdm

drive.mount("/content/drive")

ARTIFACT_ROOT = Path("/content/drive/MyDrive/solar_filament_v30/artifacts")
SOURCE_CANDIDATES = [
    Path("/content/submission_v30_prize_instance_ensemble.csv"),
    ARTIFACT_ROOT / "submission_v30_prize_instance_ensemble.csv",
]
SOURCE_CSV = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_CSV is None:
    raise FileNotFoundError(f"重複修復前のV30 CSVが見つかりません: {SOURCE_CANDIDATES}")

ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
print({"入力": str(SOURCE_CSV), "出力先": str(ARTIFACT_ROOT)})
"""
    ),
    code("# [2/5] COCO圧縮RLE、重複クラスタリング、検証関数\n" + CORE),
    code(
        """# [3/5] 入力形式と純Python RLE codecを確認
source = load_submission(SOURCE_CSV)
assert len(source) == 1565, f"想定外のV30行数: {len(source)}"
assert_codec_roundtrip(source["segmentation_rle"].iloc[[0, 1, 10, 100, 1000, -1]].tolist())
print({"入力行数": len(source), "RLE codec": "round-trip OK"})
"""
    ),
    code(
        """# [4/5] 3段階の同一個体統合候補を生成
CONFIGS = [
    MergeConfig("v32a_ultraconservative_union_dice090", min_dice=0.90, min_containment=0.0),
    MergeConfig("v32b_containment099_dice050", min_dice=0.50, min_containment=0.99),
    MergeConfig("v32c_conservative_union_dice070", min_dice=0.70, min_containment=0.0),
]

frames = {}
audits = {}
output_paths = {}
for config in CONFIGS:
    frame, audit = build_candidate(source, config, progress=tqdm)
    output_path = Path("/content") / f"submission_{config.name}.csv"
    audit_path = ARTIFACT_ROOT / f"audit_{config.name}.csv"
    frame.to_csv(output_path, index=False, lineterminator="\\n")
    audit.to_csv(audit_path, index=False, lineterminator="\\n")
    shutil.copy2(output_path, ARTIFACT_ROOT / output_path.name)
    frames[config.name] = frame
    audits[config.name] = audit
    output_paths[config.name] = output_path
    print({
        "候補": config.name,
        "行数": len(frame),
        "統合グループ": int(audit["merged_groups"].sum()),
        "排他化後に除外": int(audit["dropped_masks"].sum()),
        "CSV": str(output_path),
    })
"""
    ),
    code(
        """# [5/5] 全候補を再復号し、null・空RLE・個体間重複を厳密検証
reports = []
for config in CONFIGS:
    name = config.name
    report = validate_submission(frames[name])
    report.update({
        "name": name,
        "file": str(output_paths[name]),
        "min_dice": config.min_dice,
        "min_containment": config.min_containment,
        "merged_groups": int(audits[name]["merged_groups"].sum()),
        "removed_cross_cluster_pixels": int(audits[name]["removed_cross_cluster_pixels"].sum()),
        "dropped_masks": int(audits[name]["dropped_masks"].sum()),
    })
    reports.append(report)

manifest = {
    "source": str(SOURCE_CSV),
    "source_rows": len(source),
    "v30b_public_score": 0.35,
    "recommended_first": str(output_paths["v32a_ultraconservative_union_dice090"]),
    "candidates": reports,
}
(ARTIFACT_ROOT / "v32_overlap_cluster_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"v32-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
