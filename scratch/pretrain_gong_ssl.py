"""GONG gray コーパスで ResNet18 を SimCLR 事前学習する (v72 / RESEARCH_LOG §2.22 留保)。

v70 深層 verifier の棄却理由の一つは「2.6k crops で視覚特徴が学べない」。
唯一残る検証は、無ラベル GONG 全日面画像でドメイン事前学習した
backbone を verifier の ImageNet init と置き換えること。

設計:
  - 入力: build_gong_ssl_corpus.py が出力した gray 512x512 npz
  - 2-view: RandomResizedCrop(128, scale 0.05-1.0) + h/v flip + 輝度 jitter
    (verifier crop と同じ 128px に揃える)
  - encoder: torchvision resnet18, weights=None, fc→projection head (128-d)
  - loss: NT-Xent (tau=0.2)
  - 出力: backbone state_dict (fc/proj head は捨てる)

転移: train_verifier.py --init ssl:<ckpt> で conv1 の RGB 3ch だけ移植
(mask ch は従来どおり RGB 平均で初期化)。

使い方:
    python scratch/pretrain_gong_ssl.py --corpus output/gong_ssl_corpus \
        --epochs 40 --out output/gong_ssl/resnet18_ssl.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_corpus(corpus_dir: Path) -> np.ndarray:
    arrays = []
    for path in sorted(corpus_dir.glob("batch_*.npz")):
        with np.load(path) as data:
            # key 順は保証されない (batch_size 等の補助キーが先頭になりうる)。
            # フレーム配列は最初の 1 次元以上のもの。
            frames = next((data[k] for k in data.files if data[k].ndim >= 1),
                          None)
            if frames is None:
                raise ValueError(f"frame array not found in {path}")
            arrays.append(frames)
    if not arrays:
        raise ValueError(f"no batch_*.npz under {corpus_dir}")
    return np.concatenate(arrays)


def make_view(rng: np.random.Generator, frames: np.ndarray,
              crop: int = 128) -> np.ndarray:
    """(N,512,512) uint8 gray → (N,crop,crop) float 0-1 の独立ビュー。

    RandomResizedCrop 相当: 正方形切り抜き (scale 0.05-1.0) → crop にリサイズ。
    gray は 3ch 化せずここでは 1ch のまま返す (正規化は tensor 化で)。
    """
    out = np.empty((len(frames), crop, crop), dtype=np.float32)
    import cv2
    h = frames.shape[1]
    for i, img in enumerate(frames):
        side = int(round(h * np.sqrt(rng.uniform(0.05, 1.0))))
        side = min(side, h)
        y, x = rng.integers(0, h - side + 1, 2)
        patch = img[y:y + side, x:x + side]
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        if rng.random() < 0.5:
            patch = patch[::-1]
        gain = rng.uniform(0.8, 1.25)
        patch = np.clip(patch.astype(np.float32) * gain / 255.0, 0, 1)
        out[i] = cv2.resize(patch, (crop, crop), interpolation=cv2.INTER_AREA)
    return out


def to_tensor(views: np.ndarray) -> torch.Tensor:
    """(N,crop,crop) 0-1 gray → (N,3,crop,crop) ImageNet 正規化 tensor。"""
    x = np.repeat(views[:, None], 3, axis=1)
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(np.ascontiguousarray(x))


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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = load_corpus(args.corpus)
    print(f"corpus frames: {frames.shape}", flush=True)

    from torchvision.models import resnet18
    backbone = resnet18(weights=None)
    dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 128))
    model = nn.ModuleDict({"backbone": backbone, "head": head}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    n = len(frames)
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, args.batch):
            batch = frames[order[start:start + args.batch]]
            v1 = to_tensor(make_view(rng, batch, args.crop)).to(device)
            v2 = to_tensor(make_view(rng, batch, args.crop)).to(device)
            z1 = model["head"](model["backbone"](v1))
            z2 = model["head"](model["backbone"](v2))
            loss = nt_xent(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append({"epoch": epoch,
                        "loss": round(float(np.mean(losses)), 4),
                        "sec": round(time.time() - t0, 1)})
        print(history[-1], flush=True)
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(model["backbone"].state_dict(), args.out)
    (args.out.parent / "ssl_history.json").write_text(
        json.dumps({"config": {k: str(v) for k, v in vars(args).items()},
                    "frames": int(n), "history": history}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
