# v82 manifest 修復セル — クラッシュした [7/7] の代替。
# Colab カーネルが生きていれば、この内容を新しいセルに貼って実行するだけで
# manifest が Drive に書かれる (report/best_crop_dice/checkpoint_path/CFG は
# メモリに残っている前提)。カーネルが死んでいればローカル再構築経路を使う。

import hashlib
import json

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "experiment_id": "v82_soft_target_refiner",
    "arm": CFG.ARM, "fold": CFG.FOLD, "seed": CFG.SEED,
    "cluster_iou": CFG.CLUSTER_IOU, "repeat_factor": CFG.REPEAT_FACTOR,
    "model": "MediSegUNet(3,1) — 現行 refiner と同一",
    "epochs": CFG.REFINER_EPOCHS, "lr": CFG.REFINER_LR,
    "best_crop_dice": best_crop_dice,
    "baseline_pq": CFG.BASELINE_PQ,
    "oof_pq_by_threshold": report,
    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_of(checkpoint_path),
    "gate": "soft - hard_record が 3fold・両seedで正、pooled CI がゼロを除外",
}
report_path = CFG.ARTIFACT_ROOT / f"v82_{CFG.ARM}_fold{CFG.FOLD}_s{CFG.SEED}_manifest.json"
report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"\n保存: {report_path}")
