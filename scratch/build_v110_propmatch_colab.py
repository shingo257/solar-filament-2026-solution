"""V110 Colab ノートのビルダー: proposal-matched refiner 再学習 (RESEARCH_LOG §2.88)。

v96 third_f0 Colab ノートの凍結セル (依存/Drive・COCO/fold・refiner 定義・
評価関数) を無改変で再利用し、cell3 を「proposal 生成 + ProposalCropDataset
+ refiner 学習」に全面差替え、実行セルは refiner ckpt 差替 + rescue wrap
だけを行う。

**機構**: prod refiner は GT-positive crop でしか学習していないため、
v106 で「mean prob が existence を識別できない」と確定した。
検出器 proposal 自体を学習入力にして、正例=matched GT mask・負例=空マスク
を教えることで、(a) 実際の提案分布に合った精製、(b) mean prob の識別力獲得
を狙う — 残る唯一の構造的機構。

| arm | 内容 |
|---|---|
| `prop_matched` | prod detector + prop-matched refiner (refiner ckpt のみ差替) |
| `prop_rescue050` | 同 refiner + conf∈[0.08,0.26) で refiner 平均確率>=0.50 → 0.2601 ブースト |

置換は「ちょうど 1 回」一致を assert するので元ノートが変わると失敗する。

    python scratch/build_v110_propmatch_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_filament_v96_data_scaling_third_f0_colab.ipynb"
OUT = ROOT / "solar_filament_v110_propmatch_colab.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one match, found {count}: {old[:60]!r}")
    return text.replace(old, new)


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cell(cell_type: str, source: str) -> dict:
    base = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


TITLE = """# v110 — proposal-matched refiner 再学習 (Colab 版)

RESEARCH_LOG §2.88 の事前登録実験。prod refiner は **GT-positive crop のみ**
で学習されているため、v106 で「平均確率が existence を識別できない」と
確定した。本実験は**検出器 proposal 自体を学習入力**にして GT crop ↔
proposal crop の分布ギャップを塞ぐ、残る唯一の構造的機構。

| arm | 内容 |
|---|---|
| `prop_matched` | prod detector + prop-matched refiner (ckpt のみ差替) |
| `prop_rescue050` | 同 refiner + conf∈[0.08,0.26) で refiner 平均確率>=0.50 → 0.2601 ブースト |

**前提 (Drive `MyDrive/solar_filament_v30/`)**: `competition_data/`・
`artifacts/grouped_folds.csv`・`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`・
`preprocessed_v30_q96/`・`runs/yolo_fold_2/weights/best.pt`。

**ゲート (§2.88、結果取得前に固定)**: Δ@0.26 >= **+0.010** →
fold0 honest 複製へ。fold2 単独では提出を変更しない。

GPU 必須 (~2h: 提案生成 ~15分 + refiner 学習 ~40分 + 評価3回)。
上から順に全セル実行。結果は
`artifacts/v110_propmatch_colab_manifest.json` に逐次保存。
"""

CELL3 = '''# [3/6] v110: proposal-matched refiner 学習 (§2.88) — GT crop ↔ proposal crop の分布ギャップを塞ぐ
import math
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

PROP_CACHE = CFG.ARTIFACT_ROOT / "v110_proposals_train01.json"
PROPMATCH_CKPT = CFG.ARTIFACT_ROOT / "mediseg_edgeattnet_refiner_fold2_propmatch.pt"

def _prop_square_box(mask, context=CFG.CROP_CONTEXT, minimum=96):
    # cell5 の mask_square_box と同一手順 (提案 mask → context 正方形)
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

def generate_proposals():
    """folds(0,1) で prod 検出器推論 → 各提案を pooled 全 GT (全 record) と
    IoU>=0.5 で照合。結果は PROP_CACHE に JSON 保存 (resume 用)。"""
    if PROP_CACHE.exists():
        return json.loads(PROP_CACHE.read_text(encoding="utf-8"))
    model = YOLO(str(CFG.PROD_YOLO))
    samples = []
    train01 = [n for n in train_files if fold_of_file[n] != CFG.FOLD]
    for name in tqdm(train01, desc="proposal 生成 (folds 0,1)"):
        result = model.predict(str(CFG.PREPROC_DIR / name), imgsz=CFG.YOLO_IMGSZ,
                               conf=CFG.PREDICT_CONF_FLOOR, iou=CFG.PREDICT_IOU,
                               agnostic_nms=True, retina_masks=True, verbose=False)[0]
        masks, conf = result_masks(result)
        if not len(masks):
            continue
        prop_rles = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
        gt_pool, gt_key = [], []
        for gid in file_to_ids[name]:
            for k, r in enumerate(gt_rles[gid]):
                gt_pool.append(r); gt_key.append((gid, k))
        ious = (mask_utils.iou(prop_rles, gt_pool, [0] * len(gt_pool))
                if gt_pool else np.zeros((len(prop_rles), 0)))
        for i, rle in enumerate(prop_rles):
            j = int(ious[i].argmax()) if ious.shape[1] else -1
            matched = j >= 0 and ious[i, j] >= CFG.PROP_IOU_T
            samples.append({"file": name, "conf": float(conf[i]),
                            "rle": {"size": list(rle["size"]),
                                    "counts": rle["counts"].decode("ascii")},
                            "gt": list(gt_key[j]) if matched else None})
    del model; torch.cuda.empty_cache()
    stats = {"n_props": len(samples),
             "n_pos": sum(s["gt"] is not None for s in samples),
             "n_neg": sum(s["gt"] is None for s in samples)}
    payload = {"stats": stats, "samples": samples}
    PROP_CACHE.write_text(json.dumps(payload), encoding="utf-8")
    return payload

class ProposalCropDataset(Dataset):
    """提案 mask → square box crop (refine_proposal と同一) の学習セット。
    target = matched GT mask crop / 空マスク (負例)。"""

    def __init__(self, samples, augment=False):
        self.samples, self.augment = samples, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        image = cv2.imread(str(CFG.PREPROC_DIR / s["file"]), cv2.IMREAD_COLOR)
        prop = mask_utils.decode({"size": s["rle"]["size"],
                                  "counts": s["rle"]["counts"].encode("ascii")}) != 0
        x0, y0, x1, y1 = _prop_square_box(prop)
        image = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        if s["gt"] is None:
            target = np.zeros((CFG.CROP_SIZE, CFG.CROP_SIZE), np.float32)
        else:
            gid, k = s["gt"]
            gt = mask_utils.decode(gt_rles[gid][k]) != 0
            target = cv2.resize(gt[y0:y1, x0:x1].astype(np.float32),
                                (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        target = torch.from_numpy(target[None].copy()).float()
        if self.augment:
            k = random.randrange(4)
            image, target = torch.rot90(image, k, (1, 2)), torch.rot90(target, k, (1, 2))
            if random.random() < 0.5:
                image, target = torch.flip(image, (2,)), torch.flip(target, (2,))
            if random.random() < 0.5:
                image, target = torch.flip(image, (1,)), torch.flip(target, (1,))
            if random.random() < 0.5:
                image[:2] = torch.clamp(image[:2] ** random.uniform(0.85, 1.20), 0, 1)
        return image, target

class Fold2GTCropDataset(Dataset):
    """fold-2 GT crop valid (prod refiner と同じ選択基準の近似 — bbox ではなく
    mask-derived box を使う点だけ開示差分)。"""

    def __init__(self):
        self.samples = [(n, gid, k) for n in train_files if fold_of_file[n] == CFG.FOLD
                        for gid in file_to_ids[n] for k in range(len(gt_rles[gid]))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, gid, k = self.samples[index]
        image = cv2.imread(str(CFG.PREPROC_DIR / name), cv2.IMREAD_COLOR)
        gt = mask_utils.decode(gt_rles[gid][k]) != 0
        x0, y0, x1, y1 = _prop_square_box(gt)
        image = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(gt[y0:y1, x0:x1].astype(np.float32),
                            (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_NEAREST)
        return (torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0,
                torch.from_numpy(target[None].copy()).float())

def _soft_dice_loss(logits, targets):  # train_refiner_fold.py と同一
    probs = logits.sigmoid()
    intersection = (probs * targets).sum((1, 2, 3))
    denominator = probs.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()

def _boundary_map(tensor):  # train_refiner_fold.py と同一
    dilation = F.max_pool2d(tensor, 5, stride=1, padding=2)
    erosion = -F.max_pool2d(-tensor, 5, stride=1, padding=2)
    return torch.clamp(dilation - erosion, 0, 1)

def _refiner_loss(logits, targets):  # 0.35 BCE + 0.65 soft-dice + 0.15 edge-L1
    region = (0.35 * F.binary_cross_entropy_with_logits(logits, targets)
              + 0.65 * _soft_dice_loss(logits, targets))
    return region + 0.15 * F.l1_loss(_boundary_map(logits.sigmoid()),
                                     _boundary_map(targets))

def _crop_dice(model, loader, threshold=0.5):
    model.eval(); total = count = 0
    with torch.no_grad():
        for image, target in loader:
            pred = (model(image.cuda(non_blocking=True)).sigmoid() >= threshold).float()
            t = target.cuda(non_blocking=True)
            inter = (pred * t).sum(); total += (2 * inter / (pred.sum() + t.sum() + 1e-6)).item(); count += 1
    model.train()
    return total / max(count, 1)

def train_propmatched():
    """PROPMATCH_CKPT を返す (存在すれば resume)。"""
    if PROPMATCH_CKPT.exists():
        print("prop-matched ckpt 再利用:", PROPMATCH_CKPT)
        return PROPMATCH_CKPT
    payload = generate_proposals()
    stats = payload["stats"]
    print("proposal stats:", stats)
    train_ds = ProposalCropDataset(payload["samples"], augment=True)
    valid_ds = Fold2GTCropDataset()
    model = MediSegEdgeAttUNet(3, 1, False).cuda()
    batch = None
    for candidate in (8, 6, 4, 2):
        try:
            probe = DataLoader(train_ds, batch_size=candidate, shuffle=True, num_workers=0)
            image, target = next(iter(probe))
            with torch.amp.autocast("cuda"):
                loss = _refiner_loss(model(image.cuda()), target.cuda())
            loss.backward(); model.zero_grad(set_to_none=True)
            del probe, image, target, loss
            torch.cuda.empty_cache(); batch = candidate; break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    print("batch_size =", batch, flush=True)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.REFINER_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    best_dice, best_state = -1.0, None
    for epoch in range(CFG.REFINER_EPOCHS):
        model.train(); running = 0.0
        progress = tqdm(train_loader, desc=f"propmatch {epoch + 1}/{CFG.REFINER_EPOCHS}")
        for image, target in progress:
            image = image.cuda(non_blocking=True); target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = _refiner_loss(model(image), target)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item()
            progress.set_postfix(loss=f"{running / (progress.n + 1):.4f}")
        scheduler.step()
        dice = _crop_dice(model, valid_loader)
        if dice > best_dice:
            best_dice = dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"propmatch epoch {epoch + 1}: crop Dice {dice:.6f} (best {best_dice:.6f})", flush=True)
    torch.save({"model": model.state_dict(), "best_model": best_state,
                "fold": CFG.FOLD, "model_class": "edgeatt", "seed": CFG.REFINER_SEED,
                "init": "proposals01", "best_dice": best_dice,
                "proposals_sha256": file_sha256(PROP_CACHE),
                "training_config": {"crop_size": CFG.CROP_SIZE, "crop_context": CFG.CROP_CONTEXT,
                                    "epochs": CFG.REFINER_EPOCHS, "lr": 2e-4,
                                    "prop_conf_floor": CFG.PREDICT_CONF_FLOOR,
                                    "prop_iou_t": CFG.PROP_IOU_T, "batch_size": batch,
                                    **stats}}, PROPMATCH_CKPT)
    return PROPMATCH_CKPT
'''

V110_WRAPPERS = '''

# --- v110: rescue スコア挿入を CFG から読む同等 override
# (prod 既定 RESCUE_TAU=None では元関数と同一経路)。
_orig_result_masks = result_masks
def _refiner_mean(image, proposal):
    """v106 と同一式: 提案画素上の refiner sigmoid 平均確率。"""
    x0, y0, x1, y1 = mask_square_box(proposal)
    crop = cv2.resize(image[y0:y1, x0:x1], (CFG.CROP_SIZE, CFG.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
    with torch.no_grad():
        p = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
    p = cv2.resize(p, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    full = np.zeros_like(proposal, dtype=np.float32); full[y0:y1, x0:x1] = p
    return float((full * proposal).sum() / max(proposal.sum(), 1))

def result_masks(result):  # noqa: F811
    masks, conf = _orig_result_masks(result)
    if CFG.RESCUE_TAU is None:
        return masks, conf
    for i, m in enumerate(masks):
        if CFG.PREDICT_CONF_FLOOR <= conf[i] < CFG.PRIMARY_CONF \\
                and _refiner_mean(result.orig_img, m) >= CFG.RESCUE_TAU:
            conf[i] = 0.2601
    return masks, conf

'''


def patch_config(src: str) -> str:
    src = replace_once(src, "    ARMS = ('third_f0',)\n    TRAIN_FOLDS = (0,)",
                       "    ARMS = ('prop_matched', 'prop_rescue050')\n"
                       "    TRAIN_FOLDS = (0, 1)  # proposal 生成側 (fold-2 を除く)\n"
                       "    PROP_IOU_T = 0.5     # 採点と同じ一致閾値\n"
                       "    REFINER_EPOCHS = 14  # prod refiner と同一\n"
                       "    REFINER_SEED = 3407  # prod refiner と同一\n"
                       "    RESCUE_TAU = None    # prod 既定 (rescue なし)\n"
                       "    PARAMS_OF_ARM = {\n"
                       "        'prop_matched': {'refiner': 'propmatch'},\n"
                       "        'prop_rescue050': {'refiner': 'propmatch', 'rescue_tau': 0.50}}")
    src = replace_once(src, 'RUNS_ROOT = DRIVE_ROOT / "runs" / "v96_data_scaling"',
                       'RUNS_ROOT = DRIVE_ROOT / "runs" / "v110_propmatch"')
    src = replace_once(src, 'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v96_data_scaling"',
                       'YOLO_ROOT = DRIVE_ROOT / "yolo_datasets" / "v110_propmatch"')
    src = replace_once(src, 'LOCAL_ROOT = Path("/content/solar_v96")',
                       'LOCAL_ROOT = Path("/content/solar_v110")')
    src = replace_once(src, 'MANIFEST = ARTIFACT_ROOT / "v96_third_f0_colab_manifest.json"',
                       'MANIFEST = ARTIFACT_ROOT / "v110_propmatch_colab_manifest.json"')
    src = replace_once(src, "    GATE_DELTA = 0.005",
                       "    GATE_DELTA = 0.010  # §2.88: fold0 複製ゲート")
    return src


def patch_run(src: str) -> str:
    src = replace_once(src, "# [6/6] 実行: prod baseline → 1/3 データ arm 学習 → best/last 評価、manifest を逐次保存 (判定は §2.67 の arm 横断ゲートで行う)",
                       "# [6/6] 実行: prod baseline → prop-matched refiner 学習 → refiner 差替 arm 評価 (manifest 逐次保存、判定は §2.88 の per-arm ゲート)")
    src = replace_once(src, '"experiment_id": "v96_data_scaling"',
                       '"experiment_id": "v110_propmatch_refiner"')
    src = replace_once(src, '"protocol": "RESEARCH_LOG §2.67 (detector data-scaling learning curve)"',
                       '"protocol": "RESEARCH_LOG §2.88 (proposal-matched refiner retrain + rescue re-test)"')
    src = replace_once(src, '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,',
                       '"prod_yolo_sha256": file_sha256(CFG.PROD_YOLO), "yolo_cls": CFG.YOLO_CLS,\n'
                       '            "prop_iou_t": CFG.PROP_IOU_T, "prop_conf_floor": CFG.PREDICT_CONF_FLOOR,\n'
                       '            "refiner_epochs": CFG.REFINER_EPOCHS, "refiner_seed": CFG.REFINER_SEED,\n'
                       '            "params_of_arm": {k: dict(v) for k, v in CFG.PARAMS_OF_ARM.items()},')
    src = replace_once(src,
                       '"gate": "0.585*(PQ(prod,0.26) - mean PQ(third_f0, third_f1 @0.26)) >= +0.005 (§2.67)"',
                       '"gate": "PQ(arm,0.26)-PQ(prod,0.26) >= +0.010 → fold0 honest replication (§2.88)"')
    src = replace_once(src, 'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}',
                       'manifest = manifest or {"contract": CONTRACT, "runtime": RUNTIME_INFO, "results": {}, "datasets": {}}\n'
                       + V110_WRAPPERS)
    src = replace_once(
        src,
        """    weights, stats = train_arm(arm)
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""",
        """    p = {"refiner": None, "rescue_tau": None}
    p.update(CFG.PARAMS_OF_ARM.get(arm, {}))
    stats = {"infer_only": True, "params": p}
    if p["refiner"] == "propmatch":
        ckpt = train_propmatched()
        _st = torch.load(ckpt, map_location="cuda", weights_only=False)
        refiner.load_state_dict(_st.get("best_model") or _st["model"], strict=True)
        stats |= {"refiner_ckpt": str(ckpt), "refiner_sha256": file_sha256(ckpt)}
    CFG.RESCUE_TAU = p["rescue_tau"]
    weights = CFG.PROD_YOLO
    manifest["datasets"][arm] = stats
    by_conf = evaluate_yolo(weights, arm)""")
    src = replace_once(src, '    verdict = "cross_arm_gate_§2.67"',
                       '    verdict = "promote_fold0_replication" if delta >= CFG.GATE_DELTA else "reject_v110_fold2"')
    src = replace_once(src, "    if last.exists():  # 診断 (判定不使用): 全データ学習の best/last 選択方針の根拠",
                       "    if last.exists() and arm not in CFG.ARMS:  # 診断 (v110 は全て prod 重み・該当なし)")
    return src


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = _cell("markdown", TITLE)
    cells[1]["source"] = patch_config(_source(cells[1])).splitlines(keepends=True)
    cells[3]["source"] = CELL3.splitlines(keepends=True)  # YOLO 学習セル → proposal-matched refiner 学習に全面差替え
    # cell2 (COCO/fold/前処理)・cell4 (refiner 定義)・cell5 (評価関数) は無改変
    cells[6]["source"] = patch_run(_source(cells[6])).splitlines(keepends=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {OUT.name}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
