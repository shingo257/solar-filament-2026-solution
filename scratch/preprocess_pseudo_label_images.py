"""pseudo-label候補画像(FITS→階調変換済みグレースケール)に、既存の学習
パイプラインと全く同一の前処理(`solar_three_channel`、
`build_v30_prize_colab.py`より一字一句同一に複製)を適用する。

背景勾配抑制(radial gradient除去)+CLAHE+暗部強調の3チャンネル画像を作り、
JPEG q96でキャッシュする。YOLO推論の直接の入力になる。

使い方:
    python scratch/preprocess_pseudo_label_images.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def robust_uint8(array, mask=None, low=0.5, high=99.5):
    values = array[mask] if mask is not None and mask.any() else array.reshape(-1)
    lo, hi = np.percentile(values, [low, high])
    return np.clip((array - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)


def solar_three_channel(gray):
    # build_v30_prize_colab.py の同名関数と同一(学習画像の前処理と厳密に揃える)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=Path("output/pseudo_label_raw/images"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/pseudo_label_raw/preprocessed"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.in_dir.glob("*.jpeg"))
    print(f"対象: {len(files)}件")
    converted = 0
    for path in tqdm(files, desc="preprocess"):
        destination = args.out_dir / path.name
        if destination.exists():
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  警告: {path.name} 読み込み失敗")
            continue
        cv2.imwrite(str(destination), solar_three_channel(gray), [cv2.IMWRITE_JPEG_QUALITY, 96])
        converted += 1

    print(f"完了: {converted}件 -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
