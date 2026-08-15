# Solar Filament Segmentation Challenge 2026 ☀️

Kaggle サブコンペ **「Solar Filament Segmentation Challenge 2026」** 用の 2D 画像セグメンテーションパイプラインです。

---

## 📁 ディレクトリ構造

```text
C:\work\code\Solar_Filament_Segmentation\
├── README.md                          # プロジェクト総合ガイド
├── requirements.txt                   # パッケージ依存一覧
├── src/
│   ├── config.py                      # 画像サイズ・モデル設定
│   ├── utils.py                       # RLE エンコード/デコード, Dice/IoU メトリクス
│   ├── dataset.py                     # Albumentations 2D Dataset
│   ├── model.py                       # smp (UnetPlusPlus, DeepLabV3+) モデル定義
│   ├── train.py                       # 5-Fold CV / AMP 高速学習
│   └── infer.py                       # RLE 提出用 submission.csv 生成
└── notebooks/
    └── solar_colab_starter.ipynb      # Colab Pro+ 用統合ノートブック
```

---

## ⚡ 使い方ガイド

```bash
# 1. パイプライン疎通テスト (ローカル)
python -m src.train

# 2. RLE 提出ファイル生成テスト
python -m src.infer
```
