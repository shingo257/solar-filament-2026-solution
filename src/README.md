# src/ — 後方互換シム層 + 撤去予定の旧実装

**実装の正本はすべて `solar_filament/`。** 新規コードは
`solar_filament.xxx` を直接 import すること。

## シムファイル（恒久的に残置）

`src/` の大半の `.py` は `from solar_filament.* import *` するだけの
6行シムであり、旧 `scratch/` スクリプトや歴史 notebook が書く
`from src.xxx import ...` を動かし続けるための compat 層。
凍結済み実験スクリプト約32ファイルが参照するため移行・撤去は行わない
（provenance 維持）。

## 旧実装（シムではない・撤去対象として識別済み）

以下6ファイルは V30 以降未使用の旧パイプライン実装（512×512/5-fold）で、
**シムではなく `solar_filament/` 側に対応物もない**:

`config.py`・`dataset.py`・`model.py`・`train.py`・`infer.py`・`utils.py`

> ⚠️ 特に `utils.py` の `rle_encode`/`rle_decode` は Kaggle形式
> （10進ペア文字列）であり、`solar_filament/pipeline/rle.py`
> （pycocotools圧縮カウント形式）とは**別のコーデック** — 誤用すると
> 提出CSVが壊れる。これらのファイルが checkout に存在する場合でも
> 新規コードで使ってはいけない。

参照ゼロを立証のうえ PR #596 で撤去。撤去後に必要になった場合は
git 履歴から復元すること。
