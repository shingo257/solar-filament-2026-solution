# [3/8] pseudo-labelをYOLO polygon形式へ変換(train splitにのみ追加)
def load_pseudo_cache(path):
    data = np.load(path)
    _, width = map(int, data["shape"])
    packed = data["packed"]
    if packed.size == 0:
        return [], np.zeros(0, np.float32)
    return ([np.unpackbits(item, axis=-1)[:, :width].astype(bool) for item in packed],
            data["confidences"].astype(np.float32))

def mask_polygon_lines(mask, width, height):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        points = approx.astype(np.float32)
        points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
        points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in points.reshape(-1)))
    return lines

pseudo_manifest = json.loads(CFG.PSEUDO_MANIFEST.read_text(encoding="utf-8"))
pseudo_files = [m["file_name"] for m in pseudo_manifest["images"] if m["n_kept"] > 0]
print(f"pseudo-label採用画像: {len(pseudo_files)}枚, 合計{pseudo_manifest['total_kept']}個体")

pseudo_lines_by_file = {}
for name in pseudo_files:
    masks, confidences = load_pseudo_cache(CFG.PSEUDO_CACHE_DIR / f"{Path(name).stem}.npz")
    lines = []
    for mask in masks:
        lines.extend(mask_polygon_lines(mask, 2048, 2048))
    if lines:
        pseudo_lines_by_file[name] = lines

print(f"YOLO形式に変換できた画像: {len(pseudo_lines_by_file)}枚, "
      f"合計{sum(len(v) for v in pseudo_lines_by_file.values())}行")