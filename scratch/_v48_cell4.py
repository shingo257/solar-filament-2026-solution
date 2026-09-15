# [4/5] 画像ごとに: 3-fold YOLO推論 -> WBF融合(2/3合意フィルタ) -> refiner(shrink@0.45) -> 排他化
started = time.perf_counter()
manifest = []
for position, name in enumerate(targets, 1):
    out_path = CFG.OUT_DIR / f"{Path(name).stem}.npz"
    image_path = CFG.PSEUDO_DIR / name
    image = imread_unicode(image_path)

    all_masks, all_scores, all_sources = [], [], []
    for fold, model in yolo_models.items():
        result = model.predict(str(image_path), conf=CFG.CONF_YOLO, verbose=False)[0]
        if result.masks is None:
            continue
        for mask_xy, box_conf in zip(result.masks.data.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            mask_full = cv2.resize(mask_xy, (2048, 2048), interpolation=cv2.INTER_LINEAR) >= 0.5
            if mask_full.sum() < CFG.MIN_AREA:
                continue
            all_masks.append(mask_full)
            all_scores.append(float(box_conf))
            all_sources.append(fold)

    if not all_masks:
        np.savez_compressed(out_path, packed=np.empty((0, 0, 0), np.uint8),
                            shape=np.asarray([2048, 2048], np.int32), confidences=np.zeros(0, np.float32),
                            agreement=np.zeros(0, np.int32))
        manifest.append({"file_name": name, "n_proposals": 0, "n_kept": 0})
        continue

    fused_masks, fused_scores, fused_agreement = weighted_mask_fusion(
        all_masks, np.asarray(all_scores, dtype=np.float32), all_sources,
        iou_threshold=CFG.WBF_IOU_THRESHOLD, vote_threshold=CFG.WBF_VOTE_THRESHOLD)

    # 2/3以上のfoldが合意 かつ confidence >= 0.5 のみ採用
    keep_indices = [i for i in range(len(fused_masks))
                    if fused_agreement[i] >= CFG.MIN_SOURCE_AGREEMENT and fused_scores[i] >= CFG.CONF_FINAL]
    proposals = [fused_masks[i] for i in keep_indices]
    kept_agreement = [fused_agreement[i] for i in keep_indices]
    kept_scores = [float(fused_scores[i]) for i in keep_indices]

    if not proposals:
        np.savez_compressed(out_path, packed=np.empty((0, 0, 0), np.uint8),
                            shape=np.asarray([2048, 2048], np.int32), confidences=np.zeros(0, np.float32),
                            agreement=np.zeros(0, np.int32))
        manifest.append({"file_name": name, "n_proposals": len(fused_masks), "n_kept": 0})
        continue

    boxes, probabilities = refine_batch(refiner, image, proposals)
    refined = [shrink_variant(proposals[i], boxes[i], probabilities[i], CFG.SHRINK_THRESHOLD)
               for i in range(len(proposals))]
    final_order = sorted(range(len(refined)), key=lambda i: -kept_scores[i])
    final_masks_ordered = [refined[i] for i in final_order]
    final = apply_ownership_local(final_masks_ordered, CFG.DROP_THRESHOLD)
    # apply_ownership_localはarea_asc昇順に並べ替えて処理するため、final要素順はsize順。
    # confidence/agreementは元のmaskとサイズで対応付け直すのではなく、
    # final各要素についてどのproposalに由来するか面積一致で逆引きする簡易対応。
    final_scores, final_agreement = [], []
    for f_mask in final:
        best_i, best_overlap = 0, -1.0
        for i, r_mask in enumerate(final_masks_ordered):
            overlap = np.logical_and(f_mask, r_mask).sum()
            if overlap > best_overlap:
                best_overlap, best_i = overlap, i
        final_scores.append(kept_scores[final_order[best_i]])
        final_agreement.append(kept_agreement[final_order[best_i]])

    if final:
        packed = np.packbits(np.stack(final).astype(np.uint8), axis=-1)
        shape = np.asarray(final[0].shape, np.int32)
    else:
        packed = np.empty((0, 0, 0), np.uint8)
        shape = np.asarray([2048, 2048], np.int32)
    np.savez_compressed(out_path, packed=packed, shape=shape,
                        confidences=np.asarray(final_scores, np.float32),
                        agreement=np.asarray(final_agreement, np.int32))
    manifest.append({"file_name": name, "n_proposals": len(fused_masks), "n_kept": len(final)})

    if position % 25 == 0 or position == len(targets):
        total_kept = sum(m["n_kept"] for m in manifest)
        print(f"  {position}/{len(targets)}  {time.perf_counter()-started:.0f}s  累計採用{total_kept}件", flush=True)

manifest_path = CFG.OUT_DIR.parent / "pseudo_label_manifest.json"
manifest_path.write_text(json.dumps({
    "scope": f"{len(targets)}枚のGONG追加画像。3-fold YOLO WBF(2/3合意)+refiner(shrink@0.45確定構成)",
    "conf_final": CFG.CONF_FINAL, "min_source_agreement": CFG.MIN_SOURCE_AGREEMENT,
    "total_kept": sum(m["n_kept"] for m in manifest),
    "images": manifest,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved manifest:", manifest_path)
print("total_kept:", sum(m["n_kept"] for m in manifest), "/ ", len(targets), "images")