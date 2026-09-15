# [2/8] COCO・fold・前処理画像(実データ)をローカルSSDへ
TRAIN_JSON = next(CFG.DATA_CACHE.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
with open(TRAIN_JSON, encoding="utf-8") as handle:
    coco = json.load(handle)

images = {item["id"]: item for item in coco["images"]}
annotations = defaultdict(list)
for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)
file_to_ids = defaultdict(list)
for image_id, info in images.items():
    file_to_ids[info["file_name"]].append(image_id)

def annotation_rank(image_id):
    anns = annotations[image_id]
    return len(anns), sum(float(a.get("area", 0)) for a in anns)

file_to_reference_id = {name: max(ids, key=annotation_rank) for name, ids in file_to_ids.items()}
train_files = sorted(file_to_ids)
fold_frame = pd.read_csv(CFG.ARTIFACT_ROOT / "grouped_folds.csv")
fold_of_file = dict(zip(fold_frame["file_name"], fold_frame["fold"].astype(int)))
assert all(name in fold_of_file for name in train_files)

for name in tqdm(train_files, desc="実データ前処理画像をSSDへ"):
    local = CFG.PREPROC_DIR / name
    if not local.exists():
        shutil.copy2(CFG.PREPROC_CACHE / name, local)

print({"実観測": len(train_files), "fold0": sum(v == 0 for v in fold_of_file.values()),
       "annotation": len(coco["annotations"])})