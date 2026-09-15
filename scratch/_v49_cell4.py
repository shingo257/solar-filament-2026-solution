# [4/8] fold0のYOLO dataset構築(実データ + pseudo-label、val=fold0実観測のみ)
def polygon_line(ann, width, height):
    polygons = [p for p in ann.get("segmentation", []) if len(p) >= 6]
    if not polygons:
        return None
    polygon = max(polygons, key=len)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.reshape(-1))

def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)

def write_variant(fold_root, split, file_name, image_id, suffix, source_dir):
    info = images[image_id]
    stem = Path(file_name).stem + suffix
    image_out = fold_root / "images" / split / f"{stem}.jpeg"
    label_out = fold_root / "labels" / split / f"{stem}.txt"
    link_or_copy((source_dir / file_name).resolve(), image_out)
    lines = []
    for ann in annotations[image_id]:
        if ann.get("iscrowd", 0):
            continue
        line = polygon_line(ann, info["width"], info["height"])
        if line:
            lines.append(line)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\n".join(lines), encoding="utf-8")

def write_pseudo_variant(fold_root, file_name, lines):
    stem = Path(file_name).stem + "__pseudo"
    image_out = fold_root / "images" / "train" / f"{stem}.jpeg"
    label_out = fold_root / "labels" / "train" / f"{stem}.txt"
    link_or_copy((CFG.PSEUDO_IMAGE_DIR / file_name).resolve(), image_out)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\n".join(lines), encoding="utf-8")

def build_fold0_dataset_with_pseudo():
    root = CFG.YOLO_ROOT / "fold_0_pseudo"
    if root.exists():
        shutil.rmtree(root)
    for file_name in tqdm(train_files, desc="build fold0(実データ)"):
        if fold_of_file[file_name] == CFG.FOLD:
            write_variant(root, "val", file_name, file_to_reference_id[file_name], "", CFG.PREPROC_DIR)
        else:
            for image_id in file_to_ids[file_name]:
                write_variant(root, "train", file_name, image_id, f"__labeler_{image_id}", CFG.PREPROC_DIR)
    for file_name, lines in tqdm(pseudo_lines_by_file.items(), desc="build fold0(pseudo-label追加)"):
        write_pseudo_variant(root, file_name, lines)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n",
        encoding="utf-8")
    n_train = len(list((root / "images" / "train").glob("*.jpeg")))
    n_val = len(list((root / "images" / "val").glob("*.jpeg")))
    print({"train_images": n_train, "val_images": n_val, "yaml": str(yaml_path)})
    return yaml_path

yaml_path = build_fold0_dataset_with_pseudo()