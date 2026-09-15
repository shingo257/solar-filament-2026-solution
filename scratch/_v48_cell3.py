# [3/5] YOLO 3-fold・refinerのロード
yolo_models = {f: YOLO(str(path)) for f, path in CFG.YOLO_WEIGHTS.items()}

refiner_ckpt = torch.load(CFG.REFINER_CKPT, map_location="cpu", weights_only=False)
refiner = MediSegUNet(3, 1, False).cuda().eval()
refiner.load_state_dict(refiner_ckpt.get("best_model") or refiner_ckpt["model"], strict=True)
print("loaded: 3x YOLO fold models + all_data refiner")

def square_box(mask, context=1.65, minimum=96):
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

def refine_batch(refiner_model, image, proposals, batch=32):
    boxes, crops = [], []
    for proposal in proposals:
        x0, y0, x1, y1 = square_box(proposal)
        crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        boxes.append((x0, y0, x1, y1))
        crops.append(crop.transpose(2, 0, 1))
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            chunk = np.stack(crops[start:start + batch])
            tensor = torch.from_numpy(chunk.copy()).float().cuda() / 255.0
            output = refiner_model(tensor).sigmoid()[:, 0].cpu().numpy()
            for offset in range(output.shape[0]):
                x0, y0, x1, y1 = boxes[start + offset]
                probabilities.append(cv2.resize(output[offset], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR))
    return boxes, probabilities

def shrink_variant(proposal, box, probability, threshold):
    x0, y0, x1, y1 = box
    refined = np.zeros_like(proposal, dtype=bool)
    refined[y0:y1, x0:x1] = probability >= threshold
    return refined & proposal  # shrink = refined AND proposal(確定構成)

def greedy_disjoint_local(masks, order, min_area=CFG.MIN_AREA, canvas=2048):
    occupied = np.zeros((canvas, canvas), dtype=bool)
    kept = []
    for index in order:
        exclusive = masks[index] & ~occupied
        if exclusive.sum() < min_area:
            continue
        kept.append(exclusive)
        occupied |= exclusive
    return kept

def drop_contained_indices_local(masks, threshold):
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

def apply_ownership_local(masks, threshold):
    if not masks:
        return []
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    subset = drop_contained_indices_local(masks, threshold)
    return greedy_disjoint_local(masks, sorted(subset, key=lambda i: areas[i]))
