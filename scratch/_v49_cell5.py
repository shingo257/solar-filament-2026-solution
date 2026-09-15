# [5/8] fold0をpseudo-label混合で学習(V30のruns/fold0は上書きしない)
def train_yolo_fold0_pseudo(yaml_path):
    run_dir = CFG.RUNS_ROOT / "yolo_pseudolabel_fold_0"
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    done_path = run_dir / "TRAINING_COMPLETE"
    if done_path.exists() and best_path.exists():
        print(f"完了済み: {best_path}")
        return best_path
    if last_path.exists():
        print(f"resume: {last_path}")
        YOLO(str(last_path)).train(resume=True)
        if not best_path.exists():
            raise FileNotFoundError(best_path)
        done_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        return best_path

    model = YOLO(CFG.YOLO_MODEL)
    model.train(
        data=str(yaml_path),
        epochs=CFG.YOLO_EPOCHS,
        imgsz=CFG.YOLO_IMGSZ,
        batch=CFG.YOLO_BATCH,
        device=0,
        workers=4,
        patience=CFG.YOLO_PATIENCE,
        seed=CFG.SEED,
        deterministic=True,
        single_cls=True,
        mask_ratio=CFG.YOLO_MASK_RATIO,
        overlap_mask=False,
        amp=True,
        cache="disk",
        optimizer="AdamW",
        lr0=7e-4,
        lrf=0.05,
        weight_decay=5e-4,
        warmup_epochs=3,
        degrees=180,
        translate=0.04,
        scale=0.25,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.18,
        mosaic=0.20,
        close_mosaic=10,
        project=str(CFG.RUNS_ROOT),
        name="yolo_pseudolabel_fold_0",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=5,
        verbose=True,
    )
    if not best_path.exists():
        raise FileNotFoundError(f"学習が{best_path}を作らずに終了")
    done_path.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return best_path

best_path = train_yolo_fold0_pseudo(yaml_path)
print({"best": str(best_path), "size": best_path.stat().st_size})