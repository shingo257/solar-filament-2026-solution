"""深層 verifier (crop CNN) を fold0 で学習し fold1/2 の分離度を測る。

RESEARCH_LOG §2.21 / v64 後続: perfect verifier oracle +0.046 を静的特徴
(HistGBM x 10特徴) では解放できなかった。本実験は画像を直接見る CNN
verifier で分離可能な情報が存在するかを判定する:

  - train  = fold0 提案 (yolo_fold_0 = fold0を学習除外した検出器の OOF 提案)
  - select = fold1 (epoch 選択・後続の operating point 選択)
  - eval   = fold2 (凍結評価、prod と同じ holdout)

反証条件: fold2 AUC が conf 単独 (~0.71) を大きく超えられなければ
「分離情報は画像にも無い」と確定し verifier 系は完全に閉じる。

使い方:
    python scratch/train_verifier.py [--data output/verifier_v1] [--epochs 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

CROP = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(init: str = "imagenet"):
    """resnet18 を4ch入力化 (mask ch は RGB 重みの平均で初期化)。

    init:
      - "imagenet": ImageNet 事前学習 (従来どおり)
      - "ssl:<path>": pretrain_gong_ssl.py の backbone state_dict で初期化
        (GONG ドメイン SimCLR。fc=Identity 保存のため strict=False)。
    """
    from torchvision.models import resnet18, ResNet18_Weights
    if init.startswith("ssl:"):
        model = resnet18(weights=None)
        state = torch.load(init[4:], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    conv = model.conv1
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True)
    model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def to_tensor(crops_u8, flip=None):
    """uint8 (N,C,C,4) → 正規化 float tensor (N,4,C,C)。flip=(h,v) で拡張。

    RGB のみ /255 + ImageNet 正規化。mask ch は 0/1 の unit scale を保つ
    (全 ch を /255 すると mask が 0.0039 に潰れて学習信号にならない)。
    """
    x = crops_u8.astype(np.float32)
    if flip is not None:
        h, v = flip
        if h:
            x = x[:, :, ::-1]
        if v:
            x = x[:, ::-1]
    rgb = (x[..., :3] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    mask = x[..., 3:4]
    x = np.concatenate([rgb, mask], axis=-1)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def blend_report(labels_sel, conf_sel, scores_sel, labels_ev, conf_ev, scores_ev):
    """select fold で conf+verifier のロジスティック係数を学び、eval fold の
    blend AUC・相関・係数を返す (RESEARCH_LOG §2.22 の blend 行の正本計算)。"""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000).fit(
        np.c_[conf_sel, scores_sel], labels_sel)
    blend = model.predict_proba(np.c_[conf_ev, scores_ev])[:, 1]
    return {
        "corr_conf_verifier": round(float(np.corrcoef(conf_ev, scores_ev)[0, 1]), 4),
        "auc_fold2_blend": round(float(roc_auc_score(labels_ev, blend)), 4),
        "blend_coef": [round(float(c), 3) for c in model.coef_[0]],
    }


@torch.no_grad()
def predict(model, crops, device, batch=64):
    model.eval()
    scores = []
    for start in range(0, len(crops), batch):
        scores.append(model(to_tensor(crops[start:start + batch]).to(device))
                      .sigmoid()[:, 0].cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("output/verifier_v1"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", type=str, default="imagenet",
                        help="imagenet | ssl:<backbone ckpt path>")
    parser.add_argument("--out", type=Path, default=Path("output/verifier_v1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    crops = {f: np.load(args.data / f"crops_fold{f}.npy") for f in (0, 1, 2)}
    labels = {f: pd.read_csv(args.data / f"labels_fold{f}.csv")["label"].values
              for f in (0, 1, 2)}
    confs = {f: pd.read_csv(args.data / f"labels_fold{f}.csv")["conf"].values
             for f in (0, 1, 2)}
    n = {f: len(labels[f]) for f in (0, 1, 2)}
    print({"train(fold0)": n[0], "select(fold1)": n[1], "eval(fold2)": n[2],
           "pos_rate": {f: round(float(labels[f].mean()), 3) for f in (0, 1, 2)}},
          flush=True)

    model = build_model(args.init).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    history = []
    best = {"auc1": -1.0, "epoch": -1, "state": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n[0])
        t0 = time.perf_counter()
        losses = []
        for start in range(0, n[0], args.batch):
            idx = order[start:start + args.batch]
            # サンプル単位でランダム反転 (左右・上下)
            augmented = crops[0][idx].copy()
            for i in range(len(idx)):
                if rng.random() < 0.5:
                    augmented[i] = np.ascontiguousarray(augmented[i][:, ::-1])
                if rng.random() < 0.5:
                    augmented[i] = np.ascontiguousarray(augmented[i][::-1])
            x = to_tensor(augmented).to(args.device)
            y = torch.from_numpy(labels[0][idx].astype(np.float32)).to(args.device)
            loss = criterion(model(x)[:, 0], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scores1 = predict(model, crops[1], args.device)
        auc1 = roc_auc_score(labels[1], scores1)
        row = {"epoch": epoch, "loss": round(float(np.mean(losses)), 4),
               "auc_fold1": round(float(auc1), 4),
               "sec": round(time.perf_counter() - t0, 1)}
        history.append(row)
        print(row, flush=True)
        if auc1 > best["auc1"]:
            best = {"auc1": float(auc1), "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in
                              model.state_dict().items()}}

    # fold1 で最良 AUC の epoch を選択 (操作点選択の第一歩: epoch 選択は fold1)
    model.load_state_dict(best["state"])
    scores = {f: predict(model, crops[f], args.device) for f in (1, 2)}
    result = {
        "best_epoch": best["epoch"], "auc_fold1_best": round(best["auc1"], 4),
        "auc_fold2_verifier": round(float(roc_auc_score(labels[2], scores[2])), 4),
        "auc_fold2_conf": round(float(roc_auc_score(labels[2], confs[2])), 4),
        "auc_fold1_conf": round(float(roc_auc_score(labels[1], confs[1])), 4),
        "history": history,
    }
    result.update(blend_report(labels[1], confs[1], scores[1],
                               labels[2], confs[2], scores[2]))
    torch.save({"model": model.state_dict(), "epoch": best["epoch"],
                "auc1": best["auc1"]}, args.out / "verifier_fold0.pt")
    np.save(args.out / "scores_fold1.npy", scores[1])
    np.save(args.out / "scores_fold2.npy", scores[2])
    (args.out / "train_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
