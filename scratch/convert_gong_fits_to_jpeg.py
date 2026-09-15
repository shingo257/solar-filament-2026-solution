"""GONGアーカイブから取得したFITS(.fits.fz)を、既存学習データと同じ階調の
グレースケールJPEGへ変換する(pseudo-labelingパイロット用)。

## 背景

`hag`フォルダのJPGサムネイルは文字ラベル・装飾枠が焼き込まれており使え
ないことを確認済み(README.md §2.1)。競技公式Dataタブに「H-Alpha観測は
FITSから8-bit JPEGへ変換したもの」と明記されているため、科学グレードの
FITS(`haf`フォルダ)を同じ考え方で変換する。正確な変換定数(スケーリング
方式)は非公開なので、`skimage.exposure.match_histograms`で実際の学習
画像の階調分布に合わせることで近似する(視覚的にほぼ同一のスタイルに
なることを検証済み)。

使い方:
    python scratch/convert_gong_fits_to_jpeg.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image
from skimage.exposure import match_histograms

# ヒストグラムマッチングの目標分布を作るための参照画像(実際の学習データから複数枚)
REFERENCE_IMAGES = [
    r"G:\マイドライブ\solar_filament_v30\competition_data\MAGFiLO_1.0_Kaggle_2026\train\train_images\20110109104734Ch.jpeg",
    r"G:\マイドライブ\solar_filament_v30\competition_data\MAGFiLO_1.0_Kaggle_2026\train\train_images\20110215083814Th.jpeg",
    r"G:\マイドライブ\solar_filament_v30\competition_data\MAGFiLO_1.0_Kaggle_2026\train\train_images\20110119082634Lh.jpeg",
    r"G:\マイドライブ\solar_filament_v30\competition_data\MAGFiLO_1.0_Kaggle_2026\train\train_images\20110123174914Mh.jpeg",
    r"G:\マイドライブ\solar_filament_v30\competition_data\MAGFiLO_1.0_Kaggle_2026\train\train_images\20110301082654Uh.jpeg",
]


def build_reference_histogram() -> np.ndarray:
    """複数の実学習画像を連結して、単一画像より安定した参照分布を作る。

    match_histogramsはimage/referenceのndimが一致している必要がある
    (channel数の判定に使われる)ため、1次元にravelせず2次元のまま
    縦に連結する(各画像の横幅は同じ2048pxなので単純にvstackできる)。
    """
    arrays = [np.array(Image.open(p).convert("L")) for p in REFERENCE_IMAGES]
    return np.vstack(arrays)


def convert_one(fits_path: Path, reference: np.ndarray) -> np.ndarray:
    with fits.open(fits_path) as hdul:
        data = hdul[1].data.astype(np.float32) if len(hdul) > 1 else hdul[0].data.astype(np.float32)
    matched = match_histograms(data, reference)
    return np.clip(matched, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=Path("output/pseudo_label_raw/fits"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/pseudo_label_raw/images"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference = build_reference_histogram()
    print(f"参照分布: {len(REFERENCE_IMAGES)}枚から構築")

    fits_files = sorted(args.in_dir.glob("*.fits.fz"))
    print(f"変換対象: {len(fits_files)}件")

    converted = 0
    failed = []
    for position, fits_path in enumerate(fits_files, 1):
        out_name = fits_path.name.replace(".fits.fz", ".jpeg")
        out_path = args.out_dir / out_name
        if out_path.exists():
            continue
        try:
            image = convert_one(fits_path, reference)
        except Exception as exc:  # noqa: BLE001
            print(f"  警告: {fits_path.name} 変換失敗 ({exc})")
            failed.append(fits_path.name)
            continue
        Image.fromarray(image, mode="L").save(out_path, quality=96)
        converted += 1
        if position % 20 == 0 or position == len(fits_files):
            print(f"  {position}/{len(fits_files)}  変換済み{converted}件", flush=True)

    print(f"完了: {converted}件変換、失敗{len(failed)}件 -> {args.out_dir}")
    if failed:
        print("失敗ファイル:", failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
