# Solar Filament Segmentation Challenge 2026 ☀️

Kaggle / IEEE Big Data Cup 2026 **Challenge 02: Pixel-Precise Segmentation of Solar
Filaments** への取り組み。GONG の H-Alpha 全面画像（2048×2048）から、太陽フィラメントを
**個体単位（instance）** で切り出して提出する。

- コンペ: <https://www.kaggle.com/competitions/filament-segmentation-2026>
- データ: MAGFiLO（Manually Annotated GONG Filaments from H-Alpha Observations）
- 締切: 2026-11-15

---

## 1. 評価指標 — 先に読むこと

**リーダーボード指標は Panoptic Quality (PQ) である。**

```text
PQ = Σ IoU(TP pairs) / (|TP| + 0.5|FP| + 0.5|FN|),   TP ⟺ IoU > 0.5
```

2026-08-07 に評価方法が変更され、08-12 にリーダーボードが再採点された（それ以前は Dice）。
実装は主催者が公開した [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
に準拠する。本リポジトリの `src/metrics_official.py` は、その実装との差分テスト付きで一致を保証している。

判断に効く仕様が3つある。

1. **IoU > 0.5 は硬い崖。** IoU 0.5 は Dice 0.667 相当。下回る予測は得点ゼロなだけでなく、
   FP として分母を +0.5、対応する GT も FN として +0.5 する。**惜しい予測は無得点ではなく二重減点。**
2. **評価単位は (annotator, image)。** GT の `filament_id` は `<annotator_id>-<image_id>_<n>` 形式。
   同一の予測集合が各アノテータの GT と個別に突き合わされ、TP/FP/FN が全体へ足し込まれる。
   **variant ごとにスコアを出して平均するのではない。**
3. **追加候補の損益分岐は `p·u > 0.5·PQ`。** TP 確率 p・想定 IoU u。PQ 0.35・IoU 0.7 なら p > 0.25。

> ⚠️ **Dice を性能指標として使わないこと。** V19〜V34 は Hungarian matched Dice を
> ローカル基準にしていたが、それは公式指標ではない。過去の Dice 値と PQ を直接比較してはいけない。

### 審査は定量70% / 定性30%

リーダーボード順位は考慮の一つに過ぎない。定性30%はパイプライン記述、予測マスクの見た目の
妥当性、**コード品質（モジュール性・文書化）**。2026-11-15 までに Google Form で
4ページの技術報告書と公開 Git リポジトリを提出しないと最終評価の対象外になる。

---

## 2. 現在地

| 項目 | 値 |
|---|---:|
| 確認できた自己ベスト Public | **0.37**（V35、2026-08-16） |
| 全fold OOF ベースライン（生 YOLO, conf 0.26） | PQ 0.3937 |
| + refiner `shrink@0.45` + 排他化（全fold, リークなし） | **PQ 0.4426** |
| アノテータ不一致が課す天井 | 0.7587（争いのある画像のみなら約 0.61） |
| LB 1位スナップショット | 0.48 |

**主因は境界品質。** 確定構成適用後も near-miss（IoU 0.3〜0.5）が 920 件残り、
それが全部 TP 化すれば PQ 約 0.50。PQ の分母は near-miss → TP の変換に対して不変
（TP +1、FP −0.5、FN −0.5 で相殺）なので、変換できた分がそのまま加点になる。

P2-2（YOLO を imgsz 1536 で fold0 再学習）は PQ 0.4295 < 0.4393 で不合格。
P2-3（V30 重みのまま推論 imgsz 1536）も PQ 0.4320 < 0.4393 で不合格。
解像度レバーは停止。提出正本は V35。

詳細は以下を参照。

- [`V35_PQ_REALIGNMENT_REPORT_2026-08-15.md`](V35_PQ_REALIGNMENT_REPORT_2026-08-15.md) — 指標再整合と実測（最新の作業報告）
- [`SOLAR_FILAMENT_EXPERIMENT_HANDOVER_2026-08-15.md`](SOLAR_FILAMENT_EXPERIMENT_HANDOVER_2026-08-15.md) — V19〜V35 の全経緯、既知の落とし穴、提出仕様
- [`SCIENTIFIC_UTILITY_ROADMAP.md`](SCIENTIFIC_UTILITY_ROADMAP.md) — 科学・実運用価値を検証する今後の正本計画

### 2.1 科学・実運用上の位置づけ

V35は、公式PQに整合した**競技用基準モデル**である。全fold OOFの改善は確認済みだが、
PQ改善だけではspine、barb、chirality、時間発展、観測条件変化への有用性を証明しない。
現時点の用途主張は「フィラメント候補抽出とinstanceカタログ作成の候補」までに限定し、
space-weather運用やchirality推定への適合は`not evaluated`と扱う。

今後はCompetitionとScientific / Operationalを別トラックで管理する。次期モデルを科学・実運用候補へ
昇格するには、PQに加えてspine・接続性・barb・chirality・時系列・time/site holdoutを評価する。
数値ゲートはV35、raw YOLO、V34を同一条件で再評価してから事前登録する。詳細と実行順序は
[`SCIENTIFIC_UTILITY_ROADMAP.md`](SCIENTIFIC_UTILITY_ROADMAP.md)を正本とする。

---

## 3. パイプライン

現行の提出系（V30 系）は **instance-first**。semantic 前景から連結成分を取る方式は
union Dice 0.63 でも Public 0.19 に留まり棄却済み。

```text
H-Alpha 2048x2048
  └─ 前処理（背景勾配補正・CLAHE 相当、JPEG q96 でキャッシュ）
      └─ YOLO11m-seg 3-fold（imgsz 1280, 観測画像単位の grouped split）
          │   ※ 複数アノテータは同一画像を注釈違いで独立サンプルとして投入
          └─ 個体候補（conf 0.26, min area 20）
              └─ MediSeg U-Net refiner（crop 320, context 1.65）
                  │   shrink = refined & proposal   ← 全fold OOF で最良
                  │   guided = refined & dilate(proposal, k)  ← 一貫して劣る
                  └─ 排他化（同一画像内で1画素も重複させない）
                      └─ compressed COCO RLE で submission.csv
```

**refiner は「削る」のであって「伸ばす」のではない。** `guided`（外側へ k 画素足せる）が
`shrink`（内側しか削れない）に全閾値で劣るという実測は、YOLO の候補が既に過剰に覆っている
ことを意味する。細構造を拾うために膨らませる直感は、IoU と 0.5 の崖の下では裏目に出る。
これはPQ上の結論であり、`shrink`がspineやbarbを最もよく保存することは未確認である。

---

## 4. リポジトリ構成

```text
Solar_Filament_Segmentation/
├── src/
│   ├── metrics_official.py     # ★公式PQ実装。新しい評価はすべてこれを使う
│   ├── mediseg_models.py       # MediSegUNet (31M params) ほか
│   ├── v32_overlap_cluster.py  # 重複mask統合とCOCO RLE codec（オフライン動作）
│   ├── config.py               # ⚠️ 旧構成(512x512/5-fold)。V30以降は未使用
│   ├── dataset.py / model.py / train.py / infer.py / utils.py   # ⚠️ 同上、旧構成
├── tests/
│   └── test_metrics_official.py   # 19件。公式実装との差分テストを含む
├── scratch/                    # ノートブック生成器と診断スクリプト（§5参照）
├── notebooks/                  # 初期のColabスターター
├── output/
│   ├── v32/                    # 提出候補CSVとaudit、manifest（SHA-256付き）
│   └── v35/                    # PQ再採点・天井診断・アブレーションの結果
└── solar_filament_v3*.ipynb    # 各版のColabノートブック（scratch/build_*.py から生成）
```

> ⚠️ **`src/config.py` を読んで V30 以降の実験条件を判断してはいけない。** 汎用の
> 512×512 / 5-fold / UnetPlusPlus 設定であり、V30 以降は各ビルダー内に独自 CFG を持つ。
> 設定の二重管理は既知の技術的負債（HANDOVER §17.4）。

**ノートブックは直接編集せず、`scratch/build_*.py` を直してから再生成すること。**
ipynb の JSON を手編集するより再現性が高い。

---

## 5. 再現手順

### 5.1 指標のテスト（依存: numpy, scipy）

```bash
python tests/test_metrics_official.py
```

19件すべて通ること。うち1件は主催者コードの逐語移植との差分テスト（乱数200ケース）、
1件は IoU 0.50 ちょうどが TP にならず FP+FN 両方を生むことの固定。

### 5.2 診断（GPU不要。ローカルPCで完結）

Google Drive 上の V30/V34 成果物にアクセスできる前提。スクリプト冒頭の `DRIVE` を環境に合わせる。

```bash
python scratch/run_annotator_ceiling.py      # アノテータ不一致によるPQ天井（数分）
python scratch/run_pq_oof_rescore.py         # 全fold OOFのPQ再採点＋conf掃引（約37分）
python scratch/run_ownership_ablation.py     # 排他化の割当方式8種＋オラクル上限（約60分）
python scratch/run_refiner_pq_allfold.py     # refiner方式×閾値、全fold（約87分）
python scratch/run_pipeline_stack.py         # refiner×排他化×confの重畳検証
python scratch/run_p2_imgsz_raw_pq.py        # P2-2: 1536 vs 1280 の生YOLO PQ（GPU不要）
```

### 5.3 学習（要GPU / Colab）

```bash
python scratch/build_v30_prize_colab.py         # 3-fold YOLO + refiner の本体
python scratch/build_v35_refiner_folds12.py     # fold1/2 refiner（T4で約3.5時間）
python scratch/build_v35_final_submission.py    # 確定構成の test 推論
python scratch/build_v36_yolo_imgsz_probe.py    # P2-2: imgsz 1536 再学習（不合格済み）
python scratch/build_v36_infer_imgsz_probe.py   # P2-3: 不合格（推論 imgsz 1536）
```

生成された `.ipynb` を Colab で開いて実行する。重みは Google Drive に保存される
（本リポジトリには含めない）。

---

## 6. 提出CSVの仕様

過去のエラーから確定した必須条件。詳細は HANDOVER §4。

- 列名は正確に `filament_id`, `segmentation_rle`（`rle_mask` は誤り）
- `segmentation_rle` は公式の compressed COCO RLE 文字列。size は 2048×2048 固定で保存不要
- **空 mask の行を出力しない**（空文字や NaN で残さない）
- **同一画像内の instance mask は1画素も重複してはいけない**（Kaggle が拒否する）
- COCO の image id は文字列のことがある。`int()` へ変換しない
- 提出前に全 RLE を decode し、round-trip・面積正・ID一意・occupancy>1 が0画素であることを検証する

---

## 7. 環境

`requirements.txt` を参照。V30 以降は加えて `ultralytics`, `pycocotools`, `scipy`,
`scikit-image` を使う。

**正式 run では `pip freeze` を成果物として保存すること。** Colab の Latest Container
更新で API や既定値が変わり得るため、`>=` 指定のままでは再現できない（HANDOVER §17.6）。

### ローカル実行時の注意

- **`cv2.imread` は Windows で非ASCIIパスを開けない。** `cv2.imdecode` を使う。
  この罠で一度、全画像の読み込みに失敗しながら全ゼロの表を「結果」として出力しかけた
- GPU を使わない診断は、Colab より手元の CPU の方が速い。Colab は Drive マウント経由の
  小ファイル大量読み出しが遅く、Compute Unit も消費する

---

## 8. やらない方がよいこと

HANDOVER §14 の要約。いずれも実測で裏付けられた失敗経路。

- semantic union Dice や crop Dice だけを根拠に提出する
- PQだけを根拠に、chirality・CME解析・実運用への有用性を主張する
- 単一 fold の後処理 best をそのまま test へ適用する（V31/V34 で一般化しなかった）
- overlap のある CSV をそのまま提出する
- 公開ノートブックの申告スコアを、同一条件の再現なしに性能根拠として扱う
  （申告 0.69 は LB 1位 0.48 を上回るため、この LB の値ではありえない。V33 の実測は 0.33）
