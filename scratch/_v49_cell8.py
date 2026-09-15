# [8/8] fold0 OOF(実観測のみ)をV35確定後処理で採点し、0.4393と比較
def load_yolo_cache(path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
            data["confidences"].astype(np.float32))

def to_rle(annotation, height, width):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if isinstance(segmentation.get("counts"), list):
        return mask_utils.frPyObjects(segmentation, height, width)
    rle = dict(segmentation)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle

def mask_square_box(mask, context=1.65, minimum=96):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, width, height
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > width: x0 -= x1 - width; x1 = width
    if y1 > height: y0 -= y1 - height; y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)

@torch.no_grad()
def refine_proposal(image, proposal):
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= CFG.REFINER_THRESHOLD
    return refined & proposal  # shrink(確定構成)

def greedy_disjoint(masks, order, min_area=CFG.MIN_AREA):
    occupied = np.zeros((2048, 2048), dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept

def drop_contained_indices(masks, threshold=CFG.DROP_THRESHOLD):
    if len(masks) < 2:
        return list(range(len(masks)))
    rles = mask_utils.encode(np.asfortranarray(np.stack(masks, axis=-1).astype(np.uint8)))
    areas = np.asarray(mask_utils.area(rles), dtype=np.float64)
    self_iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float64)
    dropped = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if i in dropped or j in dropped:
                continue
            value = self_iou[i, j]
            if value <= 0:
                continue
            intersection = value * (areas[i] + areas[j]) / (1.0 + value)
            smaller = i if areas[i] <= areas[j] else j
            if intersection / max(areas[smaller], 1.0) >= threshold:
                dropped.add(smaller)
    return [i for i in range(len(masks)) if i not in dropped]

def apply_ownership(masks, scores):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    subset = drop_contained_indices(masks)
    return greedy_disjoint(masks, sorted(subset, key=lambda i: areas[i]))

def iou_matrix(truth, pred):
    if not truth or not pred:
        return np.zeros((len(truth), len(pred)), np.float64)
    return np.asarray(mask_utils.iou(pred, truth, [0] * len(truth)), dtype=np.float64).T

refiner = MediSegUNet(3, 1, False).cuda().eval()
ckpt = torch.load(CFG.ARTIFACT_ROOT / "mediseg_refiner_fold0_resume.pt", map_location="cuda", weights_only=False)
refiner.load_state_dict(ckpt.get("best_model") or ckpt["model"], strict=True)

gt_rles = {}
for image_id, info in images.items():
    height, width = int(info["height"]), int(info["width"])
    gt_rles[image_id] = [to_rle(ann, height, width) for ann in annotations[image_id] if not ann.get("iscrowd", 0)]

meter = PanopticQuality()
near_miss = 0
counts = []
cache_dir = CFG.CACHE_ROOT / "yolo_pseudolabel_fold_0"
for name in tqdm(oof_files, desc="PQ採点(実観測のみ)"):
    masks, confidences = load_yolo_cache(cache_dir / f"{Path(name).stem}.npz")
    keep = [i for i, (m, s) in enumerate(zip(masks, confidences)) if s >= CFG.CONF and m.sum() >= CFG.MIN_AREA]
    proposals = [masks[i] for i in keep]
    scores = confidences[keep] if len(keep) else np.zeros(0, np.float32)
    image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
    refined = [refine_proposal(image, m) for m in proposals]
    predictions = apply_ownership(refined, scores)
    counts.append(len(predictions))
    pred_rles = []
    if predictions:
        encoded = mask_utils.encode(np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8)))
        pred_rles = encoded if isinstance(encoded, list) else [encoded]
    for image_id in file_to_ids[name]:
        truth = gt_rles[image_id]
        matrix = iou_matrix(truth, pred_rles)
        meter.add_matrix(matrix, len(truth), len(pred_rles))
        if matrix.size:
            near_miss += int(((matrix > 0.3) & (matrix <= 0.5)).any(axis=0).sum())

summary = meter.summary()
delta = summary["pq"] - CFG.BASELINE_FOLD0_PQ
continue_folds = summary["pq"] > CFG.BASELINE_FOLD0_PQ
report = {
    "scope": "fold0のみ、実観測OOFで採点(pseudo-labelは学習train splitのみ、評価に混入なし)",
    "fold": CFG.FOLD,
    "observations": len(oof_files),
    "pseudo_label_images_used_in_train": len(pseudo_lines_by_file),
    "pseudo_label_instances_used_in_train": sum(len(v) for v in pseudo_lines_by_file.values()),
    "pq": round(summary["pq"], 6),
    "sq": round(summary["sq"], 6),
    "rq": round(summary["rq"], 6),
    "near_miss": int(near_miss),
    "inst/img": float(np.mean(counts)) if counts else 0.0,
    "baseline_fold0_pq": CFG.BASELINE_FOLD0_PQ,
    "delta": round(delta, 6),
    "continue_fold12": bool(continue_folds),
    "decision": ("fold1/2でも同様にpseudo-labelを混ぜて学習してよい"
                 if continue_folds else
                 "fold0がベースラインを超えていない。全foldへ進まない"),
}
(CFG.V35_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
(CFG.V35_ROOT / "artifacts" / "v49_pseudolabel_fold0.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))