# [6/8] fold0実観測(val split=リークなし)へOOF推論
def result_masks(result):
    if result.masks is None:
        return [], np.zeros(0, np.float32)
    height, width = result.orig_shape
    output = []
    for mask in result.masks.data.detach().cpu().numpy():
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        output.append(mask >= 0.5)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)
    return output, confidences

def save_yolo_cache(path, masks, confidences):
    path.parent.mkdir(parents=True, exist_ok=True)
    if masks:
        packed = np.packbits(np.stack(masks).astype(np.uint8), axis=-1)
        shape = np.asarray(masks[0].shape, np.int32)
    else:
        packed = np.empty((0, 0, 0), np.uint8)
        shape = np.asarray([2048, 2048], np.int32)
    np.savez_compressed(path, packed=packed, shape=shape, confidences=confidences)

model = YOLO(str(best_path))
oof_files = [name for name in train_files if fold_of_file[name] == CFG.FOLD]
cache_dir = CFG.CACHE_ROOT / "yolo_pseudolabel_fold_0"
for name in tqdm(oof_files, desc="OOF推論(実観測のみ)"):
    path = cache_dir / f"{Path(name).stem}.npz"
    if path.exists():
        continue
    masks, confidences = result_masks(model.predict(
        str(CFG.PREPROC_DIR / name), imgsz=CFG.YOLO_IMGSZ, conf=0.08,
        iou=0.55, agnostic_nms=True, retina_masks=True, verbose=False,
    )[0])
    save_yolo_cache(path, masks, confidences)
print({"oof": len(oof_files), "cache": str(cache_dir), "npz": len(list(cache_dir.glob('*.npz')))})