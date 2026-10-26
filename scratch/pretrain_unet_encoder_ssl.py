"""GONG gray コーパスで MediSeg U-Net エンコーダを SimCLR 事前学習する (v73)。

v72 (RESEARCH_LOG §2.25) は GONG 事前学習を **verifier** にだけ適用し、
AUC は改善したが OOF PQ は全閾値で悪化 → 棄却。v73 は残るレーンとして
prod refiner (`MediSegEdgeAttUNet`) の **エンコーダ部** (inc/down1-4) を
SSL init して同一レシピで再学習したとき fold2 OOF PQ が改善するかを検証する。

設計:
  - 入力: build_gong_ssl_corpus.py が出力した gray 512x512 npz
  - 下流との一致のため `solar_three_channel` (σ80 background除去 + CLAHE +
    blackhat、build_v30_prize_colab.py と同一) を全フレームへ先に適用し、
    3ch uint8 (N,512,512,3) を保持 (≈1.5GB)
  - 2-view: 正方形 RandomResizedCrop (scale 0.15-1.0) → 320px (refiner の
    CROP_SIZE) → h/v flip → ch0/1 gamma jitter (refiner aug の
    `image[:2] ** U(0.85,1.20)` と同型)
  - encoder: MediSegUNet の inc→down1..down4 → global avg pool (1024-d)。
    `MediSegEdgeAttUNet` と encoder 重み名が完全に一致する
  - head: Linear(1024→1024)→ReLU→Linear(1024→128), NT-Xent tau=0.2
  - 出力: encoder のみの state_dict (`inc.*`, `down1.`〜`down4.` キー)。
    `MediSegUNet`/`MediSegEdgeAttUNet` どちらにも strict=False で載る

使い方:
    python scratch/pretrain_unet_encoder_ssl.py --corpus output/gong_ssl_corpus \
        --train-json train/annotations.json --test-dir test/test_images \
        --epochs 60 --out output/v73/unet_encoder_ssl.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from datetime import timedelta
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def robust_uint8(array, mask=None, low=0.5, high=99.5):
    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)


def solar_three_channel(gray):
    """build_v30_prize_colab.py と同一の前処理 (train/serve 一致)。"""
    gray = gray.astype(np.float32)
    disk = gray > max(4.0, np.percentile(gray, 8))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=80, sigmaY=80)
    flattened = gray / np.maximum(background, 1.0)
    flat_u8 = robust_uint8(flattened, disk, 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(flat_u8)
    raw_u8 = robust_uint8(gray, disk, 0.5, 99.5)
    dark_detail = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    dark_detail = robust_uint8(dark_detail, disk, 0.0, 99.5)
    out = np.dstack([raw_u8, clahe, dark_detail])
    out[~disk] = 0
    return out


def load_corpus(corpus_dir: Path, train_json: Path, test_dir: Path) -> tuple:
    """Validate actual observations and decoded frames before consuming SSL data."""
    try:
        from scratch.build_gong_ssl_corpus import (
            BASE, GUARD_POLICY_VERSION, IMG_SIZE, corpus_config_fingerprint, parse_stamp)
    except ImportError:
        from build_gong_ssl_corpus import (
            BASE, GUARD_POLICY_VERSION, IMG_SIZE, corpus_config_fingerprint, parse_stamp)

    config_path = corpus_dir / "corpus_config.json"
    summary = json.loads((corpus_dir / "corpus_summary.json").read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError("corpus metadata must be objects")
    if any(value.get("guard_policy") != GUARD_POLICY_VERSION for value in (config, summary)):
        raise ValueError("corpus requires train-test-temporal-v2 provenance")
    for key in ("year", "days_per_month", "frames_per_day", "exclusion_minutes", "batch_size", "img_size"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"invalid corpus config: {key}")
    if (config["exclusion_minutes"] < 360 or summary.get("dry_run") is not False
            or summary.get("exclusion_minutes") != config["exclusion_minutes"]
            or config["img_size"] != IMG_SIZE or summary.get("img_size") != IMG_SIZE):
        raise ValueError("invalid corpus guard window, image size, or dry-run status")
    coco = json.loads(train_json.read_text(encoding="utf-8"))
    images = coco.get("images") if isinstance(coco, dict) else None
    if not isinstance(images, list) or not images:
        raise ValueError("actual train observations are missing")
    try:
        train_stamps = {parse_stamp(i["file_name"].replace(".jpeg", ".fits.fz")) for i in images}
        test_stamps = {parse_stamp(p.name.replace(".jpeg", ".fits.fz"))
                       for p in test_dir.iterdir() if p.is_file() and p.suffix == ".jpeg"}
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("invalid observation names") from error
    if not test_stamps:
        raise ValueError("actual test observations are missing")
    for key, stamps in (("train_stamps", train_stamps), ("test_stamps", test_stamps)):
        if config.get(key) != sorted(s.isoformat() for s in stamps):
            raise ValueError(f"corpus differs from actual {key}")
    fingerprint = corpus_config_fingerprint(test_stamps, train_stamps, SimpleNamespace(**config))
    if config.get("fingerprint") != fingerprint:
        raise ValueError("corpus config fingerprint mismatch")
    manifest_path = corpus_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    if (not rows or type(summary.get("kept")) is not int or summary["kept"] != len(rows)):
        raise ValueError("corpus manifest/summary count mismatch")
    seen, seen_stamps = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid manifest row")
        stamp = parse_stamp(row.get("file", ""))
        if stamp.year != config["year"]:
            raise ValueError("corpus observation differs from configured year")
        if row["file"] in seen or stamp in seen_stamps or row.get("utc") != stamp.isoformat():
            raise ValueError("duplicate or inconsistent manifest observation")
        seen.add(row["file"])
        seen_stamps.add(stamp)
        if row.get("url") != f"{BASE}/{stamp:%Y%m}/{stamp:%Y%m%d}/{row['file']}":
            raise ValueError("invalid manifest source URL")
        if any(not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key])
               for key in ("sha256", "frame_sha256")):
            raise ValueError("invalid manifest source/frame SHA256")
        if type(row.get("bytes")) is not int or row["bytes"] <= 0:
            raise ValueError("invalid manifest source byte count")
        if any(abs(stamp - anchor) <= timedelta(minutes=config["exclusion_minutes"])
               for anchor in train_stamps | test_stamps):
            raise ValueError("corpus observation overlaps train/test temporal window")
    arrays = []
    for path in sorted(corpus_dir.glob("batch_*.npz")):
        with np.load(path) as data:
            if ("batch_size" not in data.files or data["batch_size"].ndim != 0
                    or int(data["batch_size"]) != config["batch_size"]):
                raise ValueError(f"corpus batch size metadata mismatch: {path}")
            frames = next((data[k] for k in data.files if data[k].ndim >= 1),
                          None)
            if frames is None:
                raise ValueError(f"frame array not found in {path}")
            if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (IMG_SIZE, IMG_SIZE):
                raise ValueError(f"invalid corpus frame shape/type: {path}")
            if not 0 < len(frames) <= config["batch_size"]:
                raise ValueError(f"invalid corpus batch frame count: {path}")
            arrays.append(frames)
    if not arrays:
        raise ValueError(f"no batch_*.npz under {corpus_dir}")
    frames = np.concatenate(arrays)
    if len(frames) != len(rows):
        raise ValueError("corpus batch/manifest count mismatch")
    for frame, row in zip(frames, rows):
        if hashlib.sha256(frame.tobytes()).hexdigest() != row["frame_sha256"]:
            raise ValueError("corpus decoded frame SHA256 mismatch")
    return frames, {"guard_policy": GUARD_POLICY_VERSION,
                    "config_fingerprint": fingerprint,
                    "train_observations_sha256": hashlib.sha256(json.dumps(
                        sorted(s.isoformat() for s in train_stamps),
                        separators=(",", ":")).encode()).hexdigest(),
                    "test_observations_sha256": hashlib.sha256(json.dumps(
                        sorted(s.isoformat() for s in test_stamps),
                        separators=(",", ":")).encode()).hexdigest(),
                    "corpus_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "frames": len(frames), "exclusion_minutes": config["exclusion_minutes"]}


def to_three_channel_corpus(frames: np.ndarray) -> np.ndarray:
    """(N,512,512) uint8 gray → (N,512,512,3) uint8 solar3 表現。"""
    out = np.empty((len(frames), *frames.shape[1:], 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = solar_three_channel(frame)
        if (i + 1) % 200 == 0 or i + 1 == len(frames):
            print(f"  solar3 transform {i + 1}/{len(frames)}", flush=True)
    return out


def make_view(rng: np.random.Generator, frames3: np.ndarray,
              crop: int = 320) -> np.ndarray:
    """(N,512,512,3) uint8 → (N,3,crop,crop) float 0-1 の独立ビュー。

    正方形 RRC (scale 0.15-1.0) → h/v flip → ch0/1 gamma jitter
    (refiner 学習 aug `image[:2] ** U(0.85,1.20)` と同型)。
    """
    n, h = len(frames3), frames3.shape[1]
    out = np.empty((n, 3, crop, crop), dtype=np.float32)
    for i, img in enumerate(frames3):
        side = int(round(h * np.sqrt(rng.uniform(0.15, 1.0))))
        side = min(side, h)
        y, x = rng.integers(0, h - side + 1, 2)
        patch = img[y:y + side, x:x + side]
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        if rng.random() < 0.5:
            patch = patch[::-1]
        patch = cv2.resize(patch, (crop, crop), interpolation=cv2.INTER_AREA)
        view = np.ascontiguousarray(patch.transpose(2, 0, 1)).astype(np.float32) / 255.0
        view[:2] = np.clip(view[:2] ** rng.uniform(0.85, 1.20), 0, 1)
        out[i] = view
    return out


def encoder_feature(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """MediSegUNet/EdgeAttUNet の encoder 経路 (inc→down1..4) → 1024-d。"""
    x = model.inc(x)
    x = model.down1(x)
    x = model.down2(x)
    x = model.down3(x)
    x = model.down4(x)
    return x.mean(dim=(2, 3))


def encoder_state_dict(model: nn.Module) -> dict:
    prefix = ("inc.", "down1.", "down2.", "down3.", "down4.")
    return {k: v for k, v in model.state_dict().items()
            if k.startswith(prefix)}


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)
    return F.cross_entropy(sim, targets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate corpus provenance and return before model/GPU work")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--crop", type=int, default=320)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames, corpus_provenance = load_corpus(args.corpus, args.train_json, args.test_dir)
    if args.validate_only:
        print(json.dumps(corpus_provenance, sort_keys=True), flush=True)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from solar_filament.models.mediseg_models import MediSegUNet
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mediseg_models import MediSegUNet

    print(f"corpus frames: {frames.shape}", flush=True)
    frames3 = to_three_channel_corpus(frames)
    print(f"solar3 corpus: {frames3.shape} {frames3.dtype}", flush=True)

    encoder = MediSegUNet(3, 1, False)
    with torch.no_grad():
        feat_dim = int(encoder_feature(encoder, torch.zeros(1, 3, 32, 32)).shape[1])
    head = nn.Sequential(
        nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, 128))
    params = list(encoder.inc.parameters())
    for mod in (encoder.down1, encoder.down2, encoder.down3, encoder.down4):
        params += list(mod.parameters())
    params += list(head.parameters())
    model = nn.ModuleDict({"encoder": encoder, "head": head}).to(device)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    n = len(frames3)
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, args.batch):
            batch = frames3[order[start:start + args.batch]]
            v1 = torch.from_numpy(make_view(rng, batch, args.crop)).to(device)
            v2 = torch.from_numpy(make_view(rng, batch, args.crop)).to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                z1 = model["head"](encoder_feature(model["encoder"], v1))
                z2 = model["head"](encoder_feature(model["encoder"], v2))
            # NT-Xent must run outside autocast: fill_diagonal_(-1e9) overflows fp16.
            loss = nt_xent(z1.float(), z2.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append({"epoch": epoch,
                        "loss": round(float(np.mean(losses)), 4),
                        "sec": round(time.time() - t0, 1)})
        print(history[-1], flush=True)
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save({"encoder": encoder_state_dict(model["encoder"]),
                        "corpus_provenance": corpus_provenance}, args.out)
    (args.out.parent / "ssl_encoder_history.json").write_text(
        json.dumps({"config": {k: str(v) for k, v in vars(args).items()},
                    "frames": int(n), "corpus_provenance": corpus_provenance,
                    "history": history}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
