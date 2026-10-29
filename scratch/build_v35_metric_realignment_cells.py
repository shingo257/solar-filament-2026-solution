"""Embedded cell payloads for the build_v35_metric_realignment notebook builder.

Verbatim extraction of the cell-source literals from
build_v35_metric_realignment.py so the builder stays readable.
No logic; loaded by the builder via sibling import.
"""


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


METRICS = (ROOT / "src" / "metrics_official.py").read_text(encoding="utf-8")


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
        """# 太陽フィラメント V35 — 公式指標 (Panoptic Quality) への再整合

V19〜V34 で最適化してきた **Hungarian matched Dice は公式指標ではありません**。

リーダーボード指標は **Panoptic Quality (PQ)** です。2026-08-07 に評価方法が
変更され、2026-08-12 にリーダーボードが再採点されました (それ以前は Dice)。

$$PQ = \\\\frac{\\\\sum_{(y,\\\\hat y)\\\\in TP} IoU(y,\\\\hat y)}{|TP| + 0.5|FP| + 0.5|FN|},
\\\\quad TP \\\\iff IoU > 0.5$$

実装は主催者が公開した self-evaluation notebook (Apache-2.0) に準拠します。
<https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook>

効いてくる仕様が 3 つあります。

1. **IoU > 0.5 は硬い崖。** IoU 0.5 は Dice 0.667 相当。これを下回る予測は
   得点ゼロなだけでなく、FP として分母を 0.5 増やし、対応する GT も FN として
   さらに 0.5 増やします。惜しい予測は無得点ではなく**二重の減点**です。
2. **評価単位は (annotator, image)。** 同じ予測が各アノテータの GT と別々に
   突き合わされ、TP/FP/FN が全体へ足し込まれます。variant ごとにスコアを出して
   平均する V30/V31 の扱いとは別物です。
3. **追加候補の損益分岐は低い。** TP 確率 p・想定 IoU u の候補は
   `p*u > 0.5*PQ` なら出した方が得。PQ 0.35・IoU 0.7 なら **p > 0.25** で足ります。
   Dice 基準で絞ってきた conf 0.26 は保守的すぎる可能性が高い。

このノートブックは **学習も test 推論もしません**。既存の OOF cache を読み直して
次を確定させます。

1. PQ で既存 OOF を再採点し、conf 動作点を引き直す
2. ゲート基準 `0.552553` を PQ の全 fold 値で置き換える
3. duplicate の発生率をアノテータ数で層別し、後処理の問題か学習ラベルの問題かを決める
4. アノテータ不一致が課す PQ の天井を測る

GPU 不要。T4 も A100 も割り当てないで実行してください。
"""
    ),
    code(
        """# [1/9] Drive接続と前提artifactの存在確認
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from google.colab import drive
from tqdm.auto import tqdm

drive.mount("/content/drive")

class CFG:
    COMPETITION = "filament-segmentation-2026"
    V30_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")
    V34_ROOT = Path("/content/drive/MyDrive/solar_filament_v34")
    OUT_ROOT = Path("/content/drive/MyDrive/solar_filament_v35")
    DATA_CACHE = V30_ROOT / "competition_data"
    V30_ARTIFACTS = V30_ROOT / "artifacts"
    YOLO_OOF_CACHE = V34_ROOT / "prediction_cache" / "oof"
    N_FOLDS = 3
    EVAL_SIZE = 2048          # 提出は 2048x2048 固定。512 へ落として IoU を測らない
    IOU_THRESHOLD = 0.5
    # 過去の値 (fold 0 のみ・Dice 基準)。比較用に持つが、そのままゲートにしない。
    V31_FOLD0_DICE = 0.5525530052850139
    V31_FOLD0_COUNT_ERROR = 1.8531073446327686

CFG.OUT_ROOT.mkdir(parents=True, exist_ok=True)
(CFG.OUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
(CFG.OUT_ROOT / "oof").mkdir(parents=True, exist_ok=True)

required = {
    "V30 competition data": CFG.DATA_CACHE,
    "V30 grouped_folds.csv": CFG.V30_ARTIFACTS / "grouped_folds.csv",
    "V34 YOLO OOF cache": CFG.YOLO_OOF_CACHE,
}
missing = {name: str(path) for name, path in required.items() if not path.exists()}
if missing:
    raise FileNotFoundError(f"前提artifactが見つかりません: {missing}")
print({name: "OK" for name in required})
print({"出力先": str(CFG.OUT_ROOT)})
"""
    ),
    code("# [2/9] 公式指標の実装 (src/metrics_official.py と同一)\n" + METRICS),
    code(
        """# [3/9] COCO annotationとfoldを読み、アノテータ数テーブルを作る
try:
    from pycocotools import mask as coco_mask
except ImportError:
    !pip -q install pycocotools
    from pycocotools import mask as coco_mask

def find_dataset_root(base):
    matches = list(Path(base).rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if not matches:
        raise FileNotFoundError(f"学習annotation JSONが見つかりません: {base}")
    return matches[0].parents[2]

DATA_ROOT = find_dataset_root(CFG.DATA_CACHE)
TRAIN_JSON = next(DATA_ROOT.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

# image id は文字列のことがある。int へ変換しない。
images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

train_files = sorted(file_to_ids)
fold_frame = pd.read_csv(CFG.V30_ARTIFACTS / "grouped_folds.csv")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))

# grouped split のリーク検査 (§36 の即時停止条件)
leaked = [name for name in train_files if name not in fold_of_file]
assert not leaked, f"fold未割当の観測が{len(leaked)}件ある: {leaked[:5]}"

def annotation_mask(image_id, ann):
    info = images[image_id]
    height, width = int(info["height"]), int(info["width"])
    segmentation = ann["segmentation"]
    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation.get("counts"), list):
        rle = coco_mask.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return coco_mask.decode(rle).astype(bool)

def gt_variants(file_name):
    return [
        [annotation_mask(image_id, ann) for ann in annotations[image_id]]
        for image_id in file_to_ids[file_name]
    ]

annotator_table = pd.DataFrame({
    "file_name": train_files,
    "fold": [fold_of_file[name] for name in train_files],
    "n_annotators": [len(file_to_ids[name]) for name in train_files],
    "counts_per_variant": [
        [len(annotations[i]) for i in file_to_ids[name]] for name in train_files
    ],
})
annotator_table["gt_count_min"] = annotator_table["counts_per_variant"].map(min)
annotator_table["gt_count_max"] = annotator_table["counts_per_variant"].map(max)
annotator_table["gt_count_mean"] = annotator_table["counts_per_variant"].map(np.mean)
annotator_table["gt_count_spread"] = annotator_table["gt_count_max"] - annotator_table["gt_count_min"]

print({
    "train観測": len(train_files),
    "複数アノテータ観測": int((annotator_table["n_annotators"] > 1).sum()),
    "annotation総数": len(coco["annotations"]),
})
print("--- アノテータ間の個体数ばらつき (複数アノテータ観測のみ) ---")
multi = annotator_table[annotator_table["n_annotators"] > 1]
print(multi[["gt_count_min", "gt_count_max", "gt_count_spread"]].describe().round(3))
"""
    ),
    code(
        """# [4/9] V30 YOLOの全fold OOF cacheを読み込み、解像度と網羅性を検査
def yolo_oof_path(fold, file_name):
    return CFG.YOLO_OOF_CACHE / f"yolo_fold_{fold}" / f"{Path(file_name).stem}.npz"

def load_yolo_cache(path):
    data = np.load(path)
    height, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    masks = [np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed]
    return masks, data["confidences"].astype(np.float32)

available, absent = [], []
for name in train_files:
    (available if yolo_oof_path(fold_of_file[name], name).exists() else absent).append(name)

# 解像度確認: cache が縮んでいると IoU が提出時と食い違い、0.5 の崖の判定がずれる
probe_masks, probe_conf = load_yolo_cache(yolo_oof_path(fold_of_file[available[0]], available[0]))
probe_shape = probe_masks[0].shape if probe_masks else None
print({
    "OOF cacheあり": len(available),
    "OOF cacheなし": len(absent),
    "mask解像度": probe_shape,
    "先頭の候補数": len(probe_masks),
})
if probe_shape != (CFG.EVAL_SIZE, CFG.EVAL_SIZE):
    print(f"警告: cacheが{probe_shape}で保存されている。MIoUは評価解像度に依存するため、"
          f"{CFG.EVAL_SIZE}前提の値として扱わないこと。")
if absent:
    print(f"警告: {len(absent)}件のOOF cacheが欠落。これらは集計から除外される。")
"""
    ),
    code(
        """# [5/9] 【優先度1+2】全fold OOFを公式PQで採点し、conf動作点を引き直す
# PQは (annotator, image) 単位で TP/FP/FN を全体へ足し込む pooled 指標なので、
# 画像ごとにスコアを出して平均してはいけない。fold別も別々の累積器で集計する。
#
# conf を大きく下に振ってあるのは、PQ の限界条件が
#   「TP確率 p の候補は p > 0.5*PQ/IoU で追加が得」= PQ0.35, IoU0.7 なら p>0.25
# であり、Dice基準で絞ってきた 0.26 は保守的すぎる可能性が高いため。
CONF_GRID = [0.08, 0.12, 0.16, 0.20, 0.26, 0.35, 0.45]
MIN_AREA = 20

def make_disjoint(masks, confidences, conf, min_area):
    \"\"\"confidence降順の先着排他化。V30b と同じ方式。\"\"\"
    if len(confidences) == 0:
        return []
    order = np.argsort(-np.asarray(confidences))
    occupied = np.zeros((CFG.EVAL_SIZE, CFG.EVAL_SIZE), dtype=bool)
    kept = []
    for index in order:
        if confidences[index] < conf:
            continue
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept

# GT variant は繰り返し使うのでメモリではなくディスクI/O回数を減らす方針で都度生成する
detail_rows = []
pq_by_conf = {}
for conf in CONF_GRID:
    partial_path = CFG.OUT_ROOT / "oof" / f"pq_rows_conf{conf:.2f}.csv"
    overall = PanopticQuality()
    per_fold = {fold: PanopticQuality() for fold in range(CFG.N_FOLDS)}
    if partial_path.exists():
        print(f"conf={conf:.2f} は計算済み。読み込みます: {partial_path.name}")
        cached = pd.read_csv(partial_path)
        for row in cached.itertuples(index=False):
            for accumulator in (overall, per_fold[row.fold]):
                accumulator.tp_ious.extend([row.tp_iou_sum / row.tp] * row.tp if row.tp else [])
                accumulator.fp_count += int(row.fp)
                accumulator.fn_count += int(row.fn)
        detail_rows.append(cached)
    else:
        rows = []
        progress = tqdm(available, desc=f"PQ採点 conf={conf:.2f}")
        for name in progress:
            fold = fold_of_file[name]
            masks, confidences = load_yolo_cache(yolo_oof_path(fold, name))
            predictions = make_disjoint(masks, confidences, conf, MIN_AREA)
            for annotator_index, variant in enumerate(gt_variants(name)):
                matrix = iou_matrix_of(variant, predictions)
                row = overall.add_matrix(matrix, len(variant), len(predictions))
                per_fold[fold].add_matrix(matrix, len(variant), len(predictions))
                near_miss = int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum()) if matrix.size else 0
                row.update({"file_name": name, "fold": fold, "conf": conf,
                            "annotator_index": annotator_index,
                            "n_annotators": len(file_to_ids[name]),
                            "near_miss_predictions": near_miss})
                rows.append(row)
            progress.set_postfix(pq=f"{overall.score:.4f}")
        frame = pd.DataFrame(rows)
        frame.to_csv(partial_path, index=False)
        detail_rows.append(frame)
        print(f"保存: {partial_path}")
    pq_by_conf[conf] = {
        "overall": overall.summary(),
        "per_fold": {fold: accumulator.summary() for fold, accumulator in per_fold.items()},
    }

pq_detail = pd.concat(detail_rows, ignore_index=True)
pq_detail.to_csv(CFG.OUT_ROOT / "oof" / "pq_rows_all.csv", index=False)

summary = pd.DataFrame({
    conf: {
        "PQ": values["overall"]["pq"],
        "SQ(TPの平均IoU)": values["overall"]["sq"],
        "RQ(検出のF1相当)": values["overall"]["rq"],
        "TP": values["overall"]["tp"],
        "FP": values["overall"]["fp"],
        "FN": values["overall"]["fn"],
    } for conf, values in pq_by_conf.items()
}).T.round(4)
summary.index.name = "conf"
print("\\n=== 全fold OOF Panoptic Quality ===")
print(summary)

print("\\n=== fold別 PQ (平均に隠さない, §13.3) ===")
print(pd.DataFrame({
    conf: {f"fold{fold}": values["per_fold"][fold]["pq"] for fold in range(CFG.N_FOLDS)}
    for conf, values in pq_by_conf.items()
}).T.round(4))

best_conf = max(pq_by_conf, key=lambda c: pq_by_conf[c]["overall"]["pq"])
best = pq_by_conf[best_conf]["overall"]
print(f"\\n最良 conf={best_conf:.2f}  PQ={best['pq']:.4f} (SQ={best['sq']:.4f} x RQ={best['rq']:.4f})")
print(f"この PQ における追加候補の損益分岐TP確率: "
      f"{marginal_prediction_threshold(best['pq'], best['sq'] or 0.7):.3f}")
print("\\nnear-miss (IoU 0.3〜0.5 で崖を越えられなかった予測) の総数:")
print(pq_detail.groupby("conf")["near_miss_predictions"].sum())
"""
    ),
    code(
        """# [6/9] 上限診断: アノテータ間の不一致が課す PQ の天井を測る
# PQ は同じ予測を各アノテータの GT と別々に突き合わせて全体へ足し込む。
# したがって「あるアノテータの GT をそのまま提出した」場合ですら満点にならない。
# その値が、このデータで到達しうる PQ の実質的な上限になる。
# 現在の 0.35 と LB 1位 0.48 を、この天井との比で解釈するために測る。
ceiling = PanopticQuality()
single_only = PanopticQuality()
for name in tqdm(available, desc="上限診断 (アノテータ0番をそのまま予測とみなす)"):
    variants = gt_variants(name)
    prediction = variants[0]
    for variant in variants:
        matrix = iou_matrix_of(variant, prediction)
        ceiling.add_matrix(matrix, len(variant), len(prediction))
        if len(variants) == 1:
            single_only.add_matrix(matrix, len(variant), len(prediction))

ceiling_summary = ceiling.summary()
print("\\n=== PQ の天井 (アノテータ0番の GT を提出した場合) ===")
print(json.dumps(ceiling_summary, ensure_ascii=False, indent=2))
print(f"単一アノテータ画像のみに限れば PQ={single_only.score:.4f} (定義上 1.0 になるはず)")
print(f"\\n観測されている自己ベスト Public 0.35 は、この天井の "
      f"{0.35 / max(ceiling_summary['pq'], 1e-9):.1%} に相当する")
print(f"LB 1位 0.48 は天井の {0.48 / max(ceiling_summary['pq'], 1e-9):.1%}")
"""
    ),
    code(
        """# [7/9] 【優先度3】duplicate発生率をアノテータ数で層別する
# 仮説: 660,119の重複画素は後処理の詰めではなく、複数アノテータを独立教師として
# 学習したことで検出器が「分解の多峰分布」を全モード同時出力していることの帰結。
# 真なら、複数アノテータ観測でduplicate率が有意に高いはずである。
DUP_CONF = 0.26

def duplicate_stats(masks, confidences, conf, min_area):
    kept = [m for m, s in zip(masks, confidences) if s >= conf and m.sum() >= min_area]
    if len(kept) < 2:
        return {"proposals": len(kept), "overlap_pixels": 0, "foreground_pixels":
                int(kept[0].sum()) if kept else 0, "contained_pairs": 0, "overlapping_pairs": 0}
    occupancy = np.zeros((CFG.EVAL_SIZE, CFG.EVAL_SIZE), dtype=np.int16)
    for mask in kept:
        occupancy += mask
    union = occupancy > 0
    contained = overlapping = 0
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            intersection = int(np.logical_and(kept[i], kept[j]).sum())
            if intersection == 0:
                continue
            overlapping += 1
            smaller = min(int(kept[i].sum()), int(kept[j].sum()))
            if smaller and intersection / smaller >= 0.90:
                contained += 1
    return {
        "proposals": len(kept),
        "overlap_pixels": int((occupancy > 1).sum()),
        "foreground_pixels": int(union.sum()),
        "contained_pairs": contained,
        "overlapping_pairs": overlapping,
    }

dup_rows = []
for name in tqdm(available, desc=f"duplicate層別 conf={DUP_CONF:.2f}"):
    masks, confidences = load_yolo_cache(yolo_oof_path(fold_of_file[name], name))
    stats = duplicate_stats(masks, confidences, DUP_CONF, MIN_AREA)
    stats.update({
        "file_name": name,
        "fold": fold_of_file[name],
        "n_annotators": len(file_to_ids[name]),
    })
    dup_rows.append(stats)

duplicates = pd.DataFrame(dup_rows)
duplicates["overlap_ratio"] = duplicates["overlap_pixels"] / duplicates["foreground_pixels"].clip(lower=1)
duplicates["contained_ratio"] = duplicates["contained_pairs"] / duplicates["overlapping_pairs"].clip(lower=1)
duplicates.to_csv(CFG.OUT_ROOT / "oof" / "duplicate_stratification.csv", index=False)

grouped = duplicates.assign(multi=duplicates["n_annotators"] > 1).groupby("multi")
print("=== duplicate発生率: 単一アノテータ (False) vs 複数アノテータ (True) ===")
print(grouped[["proposals", "overlap_pixels", "overlap_ratio", "contained_ratio",
               "overlapping_pairs"]].agg(["mean", "median", "count"]).round(4))

single = duplicates[duplicates["n_annotators"] == 1]["overlap_ratio"]
multi_ann = duplicates[duplicates["n_annotators"] > 1]["overlap_ratio"]
try:
    from scipy.stats import mannwhitneyu
    statistic, p_value = mannwhitneyu(multi_ann, single, alternative="greater")
    verdict = "支持" if p_value < 0.01 else "不支持"
    print(f"\\nMann-Whitney U (複数 > 単一): U={statistic:.0f}, p={p_value:.3e} → 仮説を{verdict}")
except Exception as error:
    p_value, verdict = None, f"検定失敗: {error}"
    print(verdict)
"""
    ),
    code(
        """# [8/9] 【優先度2】ゲート基準を PQ で引き直す
# 旧ゲート 0.552553 は (a) 公式でない Dice 系指標で、(b) fold 0 のみの探索 best
# という二重の意味で使えない。PQ の全fold OOF で置き換える。
baseline = pq_by_conf[best_conf]["overall"]
fold_scores = {fold: pq_by_conf[best_conf]["per_fold"][fold]["pq"] for fold in range(CFG.N_FOLDS)}
worst_fold = min(fold_scores.values())

gate = {
    "metric_version": METRIC_VERSION,
    "leaderboard_metric": "Panoptic Quality (公式self-evaluation notebookに準拠)",
    "baseline_source": "V30 YOLO 3-fold OOF (V34 cache再利用, 学習なし)",
    "baseline_conf": best_conf,
    "eval_size": CFG.EVAL_SIZE,
    "measured_all_fold_pq": round(baseline["pq"], 6),
    "measured_sq": round(baseline["sq"], 6),
    "measured_rq": round(baseline["rq"], 6),
    "counts": {"tp": baseline["tp"], "fp": baseline["fp"], "fn": baseline["fn"]},
    "per_fold_pq": {int(f): round(float(v), 6) for f, v in fold_scores.items()},
    "annotator_disagreement_ceiling_pq": round(ceiling_summary["pq"], 6),
    "retired_gate": {
        "value": CFG.V31_FOLD0_DICE,
        "reason": "公式でないHungarian Dice、かつfold0のみの探索bestで選択バイアスあり",
    },
    "new_gate_pq": round(baseline["pq"] + 0.010, 6),
    "new_gate_worst_fold_pq": round(worst_fold, 6),
}
print(json.dumps(gate, ensure_ascii=False, indent=2))

print("\\n--- 新しい受入条件 ---")
print(f"1. all-fold PQ > {gate['new_gate_pq']:.6f}")
print(f"2. 最悪foldのPQが {gate['new_gate_worst_fold_pq']:.6f} を下回らない")
print("3. overlap画素 0 / null 0 / duplicate filament_id 0")
print("4. Public結果を見た後にtest側パラメータを再調整していない")
print(f"\\n参考: SQ={baseline['sq']:.4f} / RQ={baseline['rq']:.4f}。"
      f"{'RQ' if baseline['rq'] < baseline['sq'] else 'SQ'} 側が支配的な制約になっている。")
if baseline["rq"] < baseline["sq"]:
    print("→ 検出の取りこぼし・過剰(FN/FP)が主因。境界を磨くより候補を増やす方が効く。")
else:
    print("→ TPの形状品質が主因。IoU 0.5の崖を越えられていない候補の境界を改善する。")
"""
    ),
    code(
        """# [9/9] manifest保存 (§34の必須項目)
import hashlib
import platform
import subprocess

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "experiment_id": "v35_diagnostic_metric_realignment",
    "purpose": ["優先度1: 公式指標で再採点", "優先度2: ゲート再導出", "優先度3: duplicate層別"],
    "trained_anything": False,
    "ran_test_inference": False,
    "metric_definition": {
        "leaderboard": "Panoptic Quality, IoU>0.5, PQ=sum(IoU_TP)/(TP+0.5FP+0.5FN)",
        "source": "https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook",
        "pooling": "(annotator, image)単位でTP/FP/FNを全体へ集計するmicro平均",
        "version": METRIC_VERSION,
        "changed_on": "2026-08-07 (LB再採点 2026-08-12)。それ以前はDice。",
    },
    "python": platform.python_version(),
    "numpy": np.__version__,
    "observations_scored": len(available),
    "observations_missing_cache": len(absent),
    "mask_resolution": list(probe_shape) if probe_shape else None,
    "gate": gate,
    "annotator_ceiling": ceiling_summary,
    "duplicate_hypothesis": {
        "statement": "複数アノテータを独立教師にしたことが系統的duplicateの主因",
        "p_value_multi_greater_than_single": None if p_value is None else float(p_value),
        "verdict": verdict,
    },
    "outputs": {},
}
for path in sorted((CFG.OUT_ROOT / "oof").glob("*.csv")):
    manifest["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_of(path)}

report_path = CFG.OUT_ROOT / "reports" / "v35_diagnostic_manifest.json"
report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"\\n保存: {report_path}")
print("\\n次にやること: このmanifestの gate をV35学習の受入条件に固定し、"
      "duplicate仮説の verdict に応じて consensus GT 学習へ進むか判断する。")
"""
    ),
]
