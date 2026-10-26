# Solar Filament Segmentation Challenge 2026 ☀️

Kaggle / IEEE Big Data Cup 2026 **Challenge 02: Pixel-Precise Segmentation of Solar
Filaments** への取り組み。GONG の H-Alpha 全面画像（2048×2048）から、太陽フィラメントを
**個体単位（instance）** で切り出して提出する。

- コンペ: <https://www.kaggle.com/competitions/filament-segmentation-2026>
- データ: MAGFiLO（Manually Annotated GONG Filaments from H-Alpha Observations）
- Historical mixed-source fingerprints: [inactive 25-path index](docs/history/mixed-source-20260828/README.md) (no research acceptance)
- 締切: 2026-11-15
- 優勝を狙うための実験計画: [`WINNING_STRATEGY_2026-08-29.md`](WINNING_STRATEGY_2026-08-29.md)
  （トップ層手法・外部文献・公式ルールの調査に基づく優先順位付き計画）
- 4ページ技術報告書（Overleaf/ACM形式）: [`report/main.tex`](report/main.tex)
  （公式Overleafテンプレート準拠、著者情報記入済み、**4回の外部査読を反映**、
  tectonicでローカルコンパイル済み・成功、[`report/main.pdf`](report/main.pdf)、
  **本文が実際に4ページへ収まり**page5はAcknowledgment/Referencesのみ。
  `report/`・`output/v51-v53`・関連notebookはcommit済み。提出前の残作業:
  requirements.txtは`>=`指定のまま（正式runの`pip freeze`が必要）。査読対応の詳細は
  `WINNING_STRATEGY_2026-08-29.md` §6.6-§6.7 参照）
- パイプライン全体を示す単一notebook: [`solar_filament_master_pipeline_colab.ipynb`](solar_filament_master_pipeline_colab.ipynb)
  （前処理→YOLO 3-fold学習→EdgeAttNet refiner学習→V52提出物生成→RLE検証、
  ビルダーは`scratch/build_master_pipeline_notebook.py`）
- 定性比較図+fold0詳細監査notebook: [`solar_filament_qualitative_analysis_colab.ipynb`](solar_filament_qualitative_analysis_colab.ipynb)
  （TP/FP/FN/SQ/RQ・IoU/Dice分布・one-to-many/many-to-one実測+改善例/失敗例の
  4パネル比較図生成、ビルダーは`scratch/build_qualitative_analysis.py`）
- **matched-parameter ablation notebook(2026-08-31、3fold全て実行完了・結果は§2.1.10参照)**:
  [`solar_filament_matched_param_ablation_colab.ipynb`](solar_filament_matched_param_ablation_colab.ipynb)
  ・[`_fold1_colab.ipynb`](solar_filament_matched_param_ablation_fold1_colab.ipynb)
  ・[`_fold2_colab.ipynb`](solar_filament_matched_param_ablation_fold2_colab.ipynb)
  （report Future Work item 1。base U-Net(31.04M) / 標準MHSA・edge priorなし
  (35.24M) / edge-attention(35.24M、V51と同一)の3構成×seed2本×3fold=18本を
  学習・OOF評価し、edge prior固有の効果とattention容量増加の効果を分離する。
  **最終結果: attention容量は3fold中2foldで効くが、edge priorがplain
  attentionを上回るfoldは無い(3fold全体PQ: base 0.4418 < edge_att 0.4429 <
  plain_mhsa 0.4432)。「偽陽性」ではなく「効果はattention容量一般に起因、
  edge prior固有の主張は支持されない」という精密化された結論。
  report/main.texを正直に書き換え済み(§2.1.10)。** 新規モデルクラス
  `MediSegPlainMHSAUNet`を`solar_filament/models/mediseg_models.py`に追加済み。
  全3fold分の生JSON・実行済みnotebookをGドライブと機械照合し`output/ablation/`
  に保存済み。ビルダーは`scratch/build_matched_param_ablation.py --fold {0,1,2}`）

- v58 提案 confidence 掃引（学習なし）:
  [`fold0`](solar_filament_v58_conf_sweep_fold0_colab.ipynb)・
  [`fold1`](solar_filament_v58_conf_sweep_fold1_colab.ipynb)・
  [`fold2`](solar_filament_v58_conf_sweep_fold2_colab.ipynb)。
  `scratch/build_v58_conf_sweep_probe.py --fold {0,1,2}` で再生成する。
  CPU 契約のみ検証済み。入力・重み・評価処理の SHA256 が変わった manifest は
  再利用せず保持する。欠損 checkpoint は部分実行として記録し、探索結果の採用には
  別途 GPU 検証と未使用 fold での確認が必要。

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
| 確認できた自己ベスト Public | 0.39（V52、2026-08-30、EdgeAttNet refiner採用） |
| V35（YOLOアンサンブル + 標準refiner） | 0.37（2026-08-16） |
| 全fold OOF ベースライン（生 YOLO, conf 0.26） | PQ 0.3937 |
| + refiner `shrink@0.45` + 排他化（全fold, リークなし） | PQ 0.4426（標準refiner） |
| + EdgeAttNet refiner（3-fold OOF、旧報告） | Δ+0.0028（95%CI [0.000915, 0.004756]、n=707）——**§2.1.10で精密化: attention容量一般は効くがedge prior固有の効果は3fold全て未確認** |
| 参照アノテーション一致ベースライン（訂正済み、§2.1.7） | PQ 0.3398（複数アノテータ観測296枚限定） |
| LB 上位クラスタ（0.55〜0.56、38チーム） | **テストセットリーク由来と判断・追走しない（2026-09-15、§2.1.11）** |
| Public LB の採点対象 | **testの約50%（≈90枚）**（LBページ明記、2026-10-09確認）。2桁表示では0.01未満の差を識別できないため提出間比較はOOFのみ |
| 作者調査の参照例（2026-10-09） | Public 0.40 / OOF 0.4597（`tushhaaarrrr` の公開記述。分割・来歴は独立未確認）。調査と次の優先順位は `WINNING_STRATEGY_2026-08-29.md` §6.23 |
| **最終提出構成（2026-10-11 確定）** | **prod = V52 レプリカ（fold-2 YOLO + all-data EdgeAttNet refiner）、Public 0.39**。定量レバー全滅（v54-v68 全15機構棄却 + A1 scorer LB反転 + v82/v67棄却 + v69タイル推論 + v70深層verifier、作者記録）のため確定 |

### 最終提出の再現（最短経路）

以下の外部実行・スコア・一致率は作者の既存記録であり、今回の統合で独立再実行した受入結果ではない。
`HANDOVER.md` 冒頭の通り、過去実行の入力・コード・モデル・分割・出力を結ぶ manifest は未確認で、
別の分割コピーには489/707件の割当相違がある。これらを新たな分割の正本や独立科学評価の証拠にしない。

学習済み重みからの**推論のみ**の再現。前処理・学習を含む全段の公式再現は
§5.0 の `solar_filament_submission_reproduction.ipynb` を参照。
**選択候補2提出（prod + plain_mhsa_f2）の再現コード経路を整備済み**。2026-10-15 にクリーン環境（Kaggle kernel v3/v4、`solar_filament_submission_reproduction.ipynb` + 重み dataset 参照のみ）で両経路の提出 CSV が byte-identical に再現されることを受入済み（prod sha16 `84f1b89466f988d9`、plain_mhsa_f2 sha16 `7b3de4e78a543cf2`、ともに提出済み参照と一致）。
2026-10-13の作者記録にはCLI両経路の180枚CSVバイト一致（PR #351）と、v69タイル推論のfold2棄却（PR #352）が追加された。
最終選択はprod＋plain_mhsa_f2（作者報告はいずれもPublic 0.39）。これらは過去実行の履歴として保持し、独立受入とは区別する。

```bash
# test 180枚 → submission_prod.csv（ローカルCPU ~50分 / Colab GPU ~4分）
# --drive で solar_filament_v30 を含む親ディレクトリを指定可（例: G:/マイドライブ）
python scratch/run_test_submission.py --mode prod --device cuda  # --device省略時はCPU

# 第2提出（plain_mhsa_f2）の再現
python scratch/run_test_submission.py --mode prod --refiner plain_mhsa_f2 --device cuda

# 重みを Drive 以外から読む: Kaggle dataset をダウンロードして --weights-dir に渡す
# （test画像には引き続き Drive 相当の配置が必要。両バリアント疎通は作者報告）
python scratch/run_test_submission.py --mode prod \
    --refiner plain_mhsa_f2 --weights-dir <dataset_dir> --device cuda
```

- **重み**: Kaggle dataset `shingo257/solar-filament-2026-weights`
  （`yolo_fold{0,1,2}_best.pt` + `refiner_all_data.pt` +
  `ablation_plain_mhsa_seed8407_fold2.pt`（第2提出用）+ `a1_scorer_*`、
  sha256 manifest 同梱）。ローカルでは Drive `solar_filament_v30/artifacts/` の
  実績パスを既定参照
- **fold 分割の provenance（2026-10-18 記録）**: 同 dataset に
  `grouped_folds_train.csv`（sha256 `febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e`）を
  追加同梱した。これが全 v30+ 実験・OOF キャッシュ・fold ckpt が実際に使った
  実験分割。一方、従来同梱の `grouped_folds.csv`
  （sha256 `24367120279820997396f7063b1ae5b235a442fd1c92617e1e8941d472504c3b`、
  提出再現パッケージの `GROUPED_FOLDS_SHA256` ピン対象）はパッケージング時に
  再生成された別分割であり、489/707件で割当が異なる。**推論再現は影響なし**
  （凍結重みを読むだけで fold を参照しない）が、**fold ckpt の再学習には
  `grouped_folds_train.csv` を使うこと**（検証経緯は RESEARCH_LOG §2.26 に記録）。
- **Colab 再現 notebook**: [`solar_filament_prod_repro_colab.ipynb`](solar_filament_prod_repro_colab.ipynb)
  （3セル直列・Drive経由・A100で223秒という作者報告）。2026-10-11 の記録では、ローカル生成CSVと
  **1172/1172 filament が IoU>0.5 で一致（99.91% IoU=1.0）**
- **提出履歴**: prod 0.39 → A1 scorer 0.37（OOF→LB反転）→ plain_mhsa_f2 **0.39**、
  edgeatt_f2 **0.39**（ablation 重み流用の refiner バリアント、学習コスト0）。
  **最終選択の推奨**: prod + plain_mhsa_f2（アーキテクチャ差で private ヘッジ最大）。
  `output/submission_log/log.json`
- **推論環境 freeze**: `pip-freeze-prod-2026-10-11.txt`（torch 2.11.0、ultralytics 8.4.120）

**主因は境界品質。** 確定構成適用後も near-miss（IoU 0.3〜0.5）が 920 件残り、
それが全部 TP 化すれば PQ 約 0.50。PQ の分母は near-miss → TP の変換に対して不変
（TP +1、FP −0.5、FN −0.5 で相殺）なので、変換できた分がそのまま加点になる。

P2-2（YOLO を imgsz 1536 で fold0 再学習）は PQ 0.4295 < 0.4393 で不合格。
P2-3（V30 重みのまま推論 imgsz 1536）も PQ 0.4320 < 0.4393 で不合格。
解像度レバーは停止。提出正本は V35。

**V44/V45（RF-DETR-Seg へのアーキテクチャ差し替え、2026-08-22〜28、fold0 raw 比較）:**
YOLO11-seg のハイパラ調整と後処理チューニング（V36〜V42）が全滅した後、NMS 不要な
DETR 系（Hungarian matching）への差し替えを試した。V44（既定解像度 432px）は
raw PQ 0.2069 で生 YOLO 基準 0.3977 を大幅に下回ったが、432px は YOLO 確定値
1280px よりはるかに低く、細長い filament 構造が潰れている疑いがあった。V45 で
解像度だけを 768px（RF-DETR-Seg 系の公式最大値）へ引き上げたところ、raw PQ が
0.3410 まで改善（epoch40、conf 閾値も 0.26→0.40 へ再校正）したが、依然として基準
未達（Δ-0.057）。epoch20→40 の追加学習効果はほぼゼロ（+0.003）で収束しており、
これ以上の epoch 延長では基準を超えないと判断できる。**解像度交絡は部分的に
支持されたが、それだけでは基準を超えられず、RF-DETR-Seg 方向は打ち切り。**

**YOLO×RF-DETR WBF アンサンブル（2026-08-28）も不採用:** アーキテクチャの
異なるモデル同士なら誤りの相関が低く、弱い RF-DETR でも足せば伸びるのでは
という仮説を試した。YOLO(conf0.26)+RF-DETR(conf0.40) を WBF（重み 1.0 固定
の単純平均）で統合し V35 確定後処理を通したところ、fold0 OOF PQ **0.3925**
で YOLO 単独基準 0.4393 はおろか生 YOLO 単独 0.3977 より悪化した。RF-DETR の
FP がそのまま混入し後段でも除去しきれなかったとみられる。confidence の
スケールを揃えた加重統合は未検証だが、優先度を下げた。

**near-miss境界refinerの閾値チューニング（2026-08-29、【訂正】監査でfold0
選択バイアスと判明、不採用）:** `shrink`（候補との交差制約）をやめ、確率
マップをそのまま閾値化する `free` variant を導入。閾値 0.86 で fold0 OOF PQ
**0.4361**（基準 0.4393 に Δ-0.0032）まで肉薄し、当初は「最良の非ベース
ライン構成」として扱っていた。

**しかし2026-08-29の監査でfold1・fold2の独立検証を行ったところ、
free variantはどの閾値でも両foldの基準を超えられないことが判明した。**
fold0で選んだ0.86はfold1でΔ-0.0157（fold内最良の0.65でもΔ-0.0075）、
fold2でΔ-0.0106（fold内最良の0.70でもΔ-0.0076）と、いずれも大きく
基準を下回った。これは2026-08-22のWBF `drop_contained@0.80`と全く同じ
「fold0選択バイアス」であり、**free variantは有望な近接結果ではなく、
不採用対象の負の結果として扱う。** `hybrid`・`drop_contained`再調整の
既存結果もすべてfold0限定であり、同様に一般化しないと判断する。

**refinerのfree-variant向け再学習（V46, 2026-08-29）も不採用:** free variant
の伸び悩みは、refinerがGTのbboxを中心に学習される一方、推論時はYOLO候補の
bboxを中心にクロップするという訓練/推論の不一致が原因という仮説を立てた。
リークなしの再学習manifest（fold1+fold2のOOFキャッシュ、YOLO候補bbox中心+
最良IoU一致GT、誤検出15%を全ゼロターゲットとして混入、3144サンプル）を作り、
既存重みからwarm startして14epoch再学習（Colab L4、crop_dice 0.7040）。
fold0 OOFで新refiner向けにfree閾値を再スイープしたが、最良でも PQ **0.4353**
（閾値0.85、基準比Δ-0.0040、既存refinerでのfree@0.86比でも Δ-0.0008 で
むしろ僅かに悪化）に留まり、既存refinerの閾値チューニングのみ（0.4361）を
超えられなかった。訓練/推論不一致の解消それ自体は、少なくともこの実装
（warm start・負例15%混入）では有効な改善レバーではなかった。

提出正本は引き続き V35。以下のV46/V47は、当時「fold0で基準にΔ-0.0032まで
肉薄していたfree@0.86」を上回れるかという観点で行った実験であり、その
比較対象自体が2026-08-29の監査でfold0限定の誤った理解だったと判明した
（後述）。ただしV46/V47はfold0の基準0.4393自体にも届いておらず、
その意味での不採用という結論は変わらない。

**V46失敗の原因候補（再挑戦する場合の優先順位）:**

1. **warm start → ゼロからの再学習。** 既存重みは「GTの位置を渡された前提」に
   最適化済みで、14epochでは新しい入力分布（YOLO候補中心クロップ）向けの
   解へ十分動けなかった可能性が高い（train_loss 0.249→0.234と下降が緩やか）。
   ただしゼロから学習すると3144サンプルは多くなく、epoch数・学習率も
   合わせて見直す必要がある。
2. **正例マッチングのIoU閾値が緩すぎる。** 現状 `NEGATIVE_MATCH_IOU=0.10`
   以上ならどんなに悪いproposal-GTペアでも正例として教師信号にしている。
   閾値を引き上げて教師信号の質を絞る余地がある。
3. **負例（誤検出の全ゼロターゲット）混入率15%の見直し。** ノイズとして
   働き既存の強みを薄めた可能性。0%/5%で比較する価値がある。
4. **drop_containedとの同時グリッドサーチ。** 今回はdrop_contained閾値を
   0.90に固定したままfree閾値だけ再スイープした。新モデルでは相互作用点が
   変わっている可能性がある。
5. **epoch数・学習率スケジュールの見直し。** 現行の14epoch/lr2e-4は
   warm start前提の流用値。ゼロから学習するなら再検討が必要。
6. **GT中心クロップの一部混合（マルチタスク的アプローチ）。** 候補中心
   クロップのみで学習すると破滅的忘却が起きている可能性があり、GT中心
   データを少量混ぜる案。

**V47(候補#1+#2+#3+#4を同時適用)も不採用、むしろV46より悪化:** ゼロから
再学習(warm startなし)+正例マッチングIoU閾値0.10→0.30+負例混入率0.15→0.05
+epoch14→30+free×drop_contained同時グリッドサーチ(32組合せ)を実施。
Colab L4で30epoch完走(学習時間37.8分、best_crop_dice **0.7622**でV46の
0.7040を大きく上回る好調な学習)。しかしfold0 OOFでの実測は
**PQ 0.4296(free0.8/drop0.9、基準比Δ-0.0097)**で、V46(0.4353)より
さらに悪化し、free@0.86単独(0.4361)からは最も遠ざかった。

**重要な発見: crop_dice(manifest内の検証crop指標)と実運用PQは乖離する。**
crop_diceが大きく改善したにもかかわらずPQが悪化したことから、正例IoU閾値の
厳格化(0.30以上のみ採用)がモデルを「綺麗にマッチした候補」に過適合させ、
実際のYOLO OOF候補集合(玉石混交、悪い候補も含む)に対しては予測が
甘くなった(過剰検出が増えた)可能性が高い。

**結論: refinerの再学習アプローチ(V46→V47)は試すたびに悪化しており、
これ以上の再挑戦(候補#2・#3を個別に切り分けるなど)は費用対効果が
見合わないと判断し打ち切る。**

**【2026-08-29、監査による最終訂正】near-miss境界refiner路線(閾値
チューニング・再学習の両方)は全体として不採用と結論する。** free variant
はfold0選択バイアスの産物であり全fold平均でΔ-0.010前後の悪化、V46/V47は
その誤った比較対象に対してすら勝てなかった。**現時点で確定している最良
構成は、V35のshrink@0.45（全fold OOF PQ 0.4426、Public 0.37）のままで
あり、近接ミス救済の後処理・再学習アプローチから全fold OOFで基準を
上回るものは一つも生まれなかった。**

refinerの再学習・後処理チューニングアプローチはここで打ち切り、次は
pseudo-labelingへ進む（下記2.2で詳述）。RF-DETR 2XLargeは優先度を下げた:
V39(yolo11l-seg、
モデルサイズ拡大)が生YOLOの時点で基準を下回り、RF-DETR自体もepoch20→40の
追加学習で+0.003しか伸びなかった実績があり、「モデルを大きくする」系統の
レバーはこのデータ規模(707観測)では一貫して効いていない。

#### 2.0.1 監査: free variantのfold1/fold2独立検証(2026-08-29、重大な訂正)

過去の結論が実データを正しく評価できているか監査した結果、
`run_refiner_free_variant_probe.py`等の関連スクリプトが**すべて
`FOLD = 0`にハードコードされており、free@0.86はfold1・fold2で一度も
検証されていなかった**ことが判明した。Colab A100で
`scratch/run_refiner_free_variant_allfold_probe.py`相当の検証を
fold1・fold2それぞれ独立に実行(fold0の0.86に引きずられないよう
0.30〜0.95の広い範囲を再スイープ)した結果:

| fold | 基準PQ(shrink@0.45) | free@0.86のPQ | Δ | fold内最良閾値 | 最良閾値でのΔ |
|---|---:|---:|---:|---:|---:|
| fold0 | 0.4393 | 0.4361 | -0.0032 | 0.86 | -0.0032 |
| fold1 | 0.4415 | 0.4258 | **-0.0157** | 0.65 | -0.0075 |
| fold2 | 0.4471 | 0.4365 | **-0.0106** | 0.70 | -0.0076 |

**free variantはfold1・fold2ではどの閾値を選んでも基準を超えられず、
fold0で選んだ0.86はfold1で最悪クラスの結果(全12閾値中で最下位に近い)
だった。** fold別の最適閾値も0.86・0.65・0.70とバラバラで、単一の
「良い閾値」は存在しない。これは2026-08-22のWBF `drop_contained@0.80`
(fold0でΔ+0.0007、fold1Δ-0.0027、fold2Δ-0.0062)と全く同じ
「fold0選択バイアス」パターンの再現であり、このプロジェクトで繰り返し
警告してきた「単一foldのbestを一般化しない」という教訓を、今度は
free variantという主力候補で自ら踏み抜いていたことになる。

**訂正: near-miss境界refinerの改善ライン(free variant・hybrid・
drop_contained再調整・V46・V47のすべて)は、fold0限定の探索に留まって
おり、全fold検証に基づく確定した改善は一つもない。** 現時点で確定して
いる最良構成はV35のshrink@0.45(全fold OOF PQ 0.4426)のみである。
生データは`output/refiner_free/refiner_free_variant_fold{1,2}_audit.json`
相当(Drive: `solar_filament_v35/refiner_retrain/free_variant_fold{1,2}_audit.json`)
に保存済み。

詳細は以下を参照。

- [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — 実験カタログを論文構成に整理したノート(独立投稿を見据えた草稿)
- [`TECHNICAL_REPORT_DRAFT.md`](TECHNICAL_REPORT_DRAFT.md) — コンペ提出用4ページ技術報告書の草稿(生きている文書、随時更新)
- [`V35_PQ_REALIGNMENT_REPORT_2026-08-15.md`](V35_PQ_REALIGNMENT_REPORT_2026-08-15.md) — 【凍結】指標再整合と実測(2026-08-15時点)
- [`SOLAR_FILAMENT_EXPERIMENT_HANDOVER_2026-08-15.md`](SOLAR_FILAMENT_EXPERIMENT_HANDOVER_2026-08-15.md) — 【凍結・改訂10で更新停止】V19〜V35 の全経緯、既知の落とし穴、提出仕様
- [`SCIENTIFIC_UTILITY_ROADMAP.md`](SCIENTIFIC_UTILITY_ROADMAP.md) — 科学・実運用価値を検証する今後の正本計画(別トラック、並行セッション管理)

### 2.1 pseudo-labeling計画(2026-08-29策定、次の優先施策)

**仮説: 707観測は少ないだけでなく、年代分布が偏っている。** 2011〜2017年
618件(87%)に対し2018〜2022年は89件(13%)。太陽周期(cycle24後半→cycle25
立ち上がり)でフィラメント出現傾向が変わりうる中、この偏りが検出器の
汎化性能(特に診断済みの主要失敗モード=密集シーンでの検出漏れ)を
損なっている可能性がある。密集シーン検出漏れ(hard FN 2138件がhard FP 1561件
を37%上回る、gt_count_mean 10-21の密集シーンに強く紐づく)という
2026-08-16の診断結果と直接つながる仮説であり、アーキテクチャ・
アンサンブル・refiner再学習という「モデル側」のレバーが軒並み不採用に
終わった今、唯一まだ試していない「データ側」のレバーになる。

**データ出典確認済み:** 現有707観測のファイル名(`20110109104734Ch.jpeg`)は
NSO GONGアーカイブ(`https://gong2.nso.edu/ftp/HA/`)の命名規則
(`YYYYMMDDHHMMSS[site]h`)と完全一致。フル解像度JPG(2048×2048)が同アーカイブ
から追加取得可能。サイトコード6種(B/C/L/M/T/U、Big Bear/Cerro Tololo/
Learmonth/Mauna Loa/Teide/Udaipur)がほぼ均等(100〜127件)に含まれる。

**ルール適合性(2026-08-29、公式ページ全タブ精査で再確認):** Evaluationタブ
「推論はtestディレクトリの画像のみ使用」「学習に他のground-truthメタデータを
使用禁止」、Rulesタブ§2.6「外部データは全参加者が無償・平等にアクセス可能
であれば使用可」の両方を満たす。GONGアーカイブは無料・公開なので適合。
pseudo-labelは自分のモデルが生成するものであり「他者のground-truth」では
ないため問題ない。AIツール(Claude Code含む)の使用もRules§2.8で明示的に
許可されている(受賞者は手法を説明・再現できる必要があるのみ)。

**競合状況(2026-08-29時点のPublic LB):** 上位7チームがPQ **0.55**で並び、
8位0.53、9〜49位は0.37〜0.40のクラスタ(参加1,982名・444チーム)。自チーム
実測Public 0.37はこのクラスタ下端。0.55が7チーム完全一致は技術的に不自然で、
運営も2026-08-20に「スコアの比較より本質的な取り組みを」と注意喚起して
おり、上位スコアの一部は素直な汎化性能を反映していない可能性がある。

**先行事例による効果較正(2026-08-29、文献調査):** 同一課題でpseudo-labeling
を実施した論文("A Modern ConvNet for Solar Filament Detection", arXiv
2607.24525)を発見。手動注釈120枚で訓練したモデルで異なる観測機器
(CHASE)の695枚を自動注釈化し815枚に拡張したところ、実GTでのIoU改善は
**わずか+0.003〜+0.006**(DenseCRF後処理の有無で変動)に留まった。著者ら
自身も「モデルバイアスの拡大」への懸念を明記している。**この結果から、
pseudo-labelingは効いても小幅な改善に留まる可能性が高いと事前に見込む
べき。** ただし当論文は異なる観測機器間の転移で、本計画は同一機器(GONG)の
別日付なのでドメインギャップはより小さく、より良い転移が期待できる。
なお同論文の指標はsemantic IoU/F1でありPQとは異なるため、数値の直接比較は
できない(方向性の参考情報)。関連文献: Flat U-Net(arXiv 2502.07259、
DSC0.82)、AA-UNet(IEEE 10046547、Jaccard0.63)など、いずれもsemantic
指標でPQとは接続不可。IEEE BigData Cup「Solar Filament Segmentation」
自体は2026年が初開催で過去大会・優勝手法は存在しない。

**フェーズ計画:**

1. **パイロットデータ取得**: 2018〜2022年に偏重した追加GONG画像を
   NSOアーカイブから取得(件数・容量を明示してユーザーに確認してから
   ダウンロード実行、既存707件と重複しない日付を選ぶ)。
2. **pseudo-label生成**: 既存の前処理(背景勾配補正+CLAHE相当+JPEG q96)を
   適用後、V35確定パイプライン(YOLO 3-fold+refiner+後処理)で推論。
   本番conf 0.26より厳しい閾値(例: 0.5)でフィルタし、可能なら複数fold
   モデルの合意(2/3以上)も要求してラベルノイズを抑える。
3. **人間によるQAビューア**: near-miss inspectorと同様の軽量アーティファクトで
   pseudo-label済み画像をサンプル表示し、目視で妥当性を確認してから
   GPU時間を投じる。
4. **リークなし拡張学習(fold0パイロット)**: fold0の学習データ=実データの
   fold1+fold2+pseudo-label画像(pseudo-labelはどのfoldの評価にも
   使われないためリークの心配はない)。V35と同一ハイパラ(yolo11m-seg,
   imgsz1280, batch4, epoch70, mosaic0.20)でfold0のみ再学習。
5. **判定**: fold0 OOF(実GTのみ)でraw PQ(基準0.3977)・確定後処理込みPQ
   (基準0.4393)を測定。ノイズレベルを超える改善(目安+0.005以上)が
   あればfold1/2へ展開し全foldゲート(>0.4526)で検証。改善が無ければ
   不採用として記録し、RF-DETR 2XLargeまたは競技trackのV35固定・
   報告書作成へ移る。

#### 2.1.1 V49結果(2026-08-29、fold0パイロット・不採用)

フェーズ1〜5を実行した(Colab A100、GONGパイロット150枚→2/3fold合意+
conf0.5フィルタで72個体/35枚採用→fold0のtrain splitにのみ追加、
val splitは実観測236枚のみでリークなし、70epoch、1.17時間)。結果は
`solar_filament_v35/artifacts/v49_pseudolabel_fold0.json`。

| 指標 | 値 |
|---|---:|
| fold0 OOF PQ(pseudo-label込み学習) | 0.4332 |
| fold0 OOF PQ(基準、V35 shrink@0.45) | 0.4393 |
| Δ | **−0.0061** |
| SQ / RQ | 0.6728 / 0.6439 |
| near-miss(IoU 0.3〜0.5) | 323件(基準相当から悪化) |

**当初判定: 不採用。** ただし2026-08-29のユーザー指摘(「軒並み不採用は
評価手法側の問題では」)を受けて監査した結果、**この判定は時期尚早
だったと判明した。以下の理由でV49は「不採用確定」ではなく「fold0単独
では結論不能」に訂正する。**

#### 2.1.2 監査: V49否定判定の再検証(2026-08-29)

過去にfold0選択バイアスが2度(WBF・free variant)発生した経緯を踏まえ、
「軒並み不採用という結果自体が評価手法の欠陥によるものではないか」を
4方向から監査した。

1. **fold分割の構成監査:** 3fold間で画像数(236/236/235)・個体密度
   (平均7.91/7.81/8.03)・GT面積・サイト分布・年代分布のいずれにも
   有意な偏りなし。分割ロジック自体に欠陥の証拠はない。
2. **fold別refiner品質監査:** fold0のrefinerはcrop-level best_dice
   0.7933で3fold中最良(fold1: 0.7864、fold2: 0.7887)。「fold0の
   refinerが学習不足」という仮説は否定された。
3. **raw→refined天井効果:** 生YOLO(refiner前)PQはfold0が3fold中最良
   (0.3977 > 0.3950 > 0.3882)だが、refiner+後処理の追加寄与は
   fold0が最小(+0.0416 vs fold1+0.0533、fold2+0.0521)。raw PQが
   高いfoldほど伸び代が小さいという負の相関で説明でき、「fold0が
   壊れている」という仮説は不要になる、自然な天井効果。
4. **【本命】V49のペア化観測クラスタ・ブートストラップ(2000回):**
   同一refiner・同一後処理で、baseline(V34キャッシュ)とV49
   (pseudo-label込み)の両方を236観測で再スコアし、同一リサンプル
   添字で対にしてΔの95%信頼区間を直接測定した。

   | | 値 |
   |---|---:|
   | baseline PQ(単独95%CI) | 0.4393 [0.4213, 0.4565] |
   | V49 PQ(単独95%CI) | 0.4332 [0.4186, 0.4471] |
   | **Δ(V49−baseline)のペア化95%CI** | **[−0.0162, +0.0041]** |
   | CIが0を跨ぐか | **跨ぐ→統計的に有意でない** |
   | リサンプルのうちΔ<0だった割合 | 88.7%(弱い悪化寄りの傾向はある) |

   さらに、fold0単独(236観測)のCI幅(0.0352)は、全fold707観測での
   既知CI幅(0.0204、並行Scientificトラックの実測)から√(707/236)
   スケーリング則で予測される値(0.0353)とほぼ完全に一致した。
   これはfold0が「特別ノイズが多い」のではなく、単純にサンプルサイズ
   の違いで説明できることの裏付けでもある。

**訂正した結論: V49のΔ-0.0061は統計的に0と区別できず、fold0単独の
パイロットでは「悪化した」と断定するだけの検出力がない。** 88.7%が
Δ<0という弱い悪化寄りの傾向はあるため、pseudo-labelingが有効という
証拠にもならないが、無効と確定した証拠にもならない、というのが
正確な現状である。

**free variant/WBFの不採用判定との違い:** これらはfold1・fold2という
**2つの独立foldが同じ符号(悪化)で一致**しており、単独foldでは有意
でなくても2fold一致という形の証拠はV49の1fold単独結果より強い。
V49はまだfold1で確認していないため、free variant/WBFと同じ基準に
照らすと「不採用確定」と呼ぶのは時期尚早である。

**次に取るべき行動: fold1でも同様のpseudo-label拡張学習を行い、
符号が一致するか確認してから最終判断する。** 監査結果一式は
`solar_filament_v35/artifacts/v49_evaluation_methodology_audit.json`
に保存済み。

#### 2.1.3 fold1確認結果(2026-08-29、pseudo-labelingは不採用に確定)

fold1でも同様にpseudo-label込みの拡張学習を実施した(fold1のtrain split=実
データのfold0+fold2+pseudo-label35画像、val split=fold1実観測236枚のみ、
リークなし、70epoch・1.213時間・A100)。

| | fold0 | fold1 | fold0+fold1統合(n=472) |
|---|---:|---:|---:|
| baseline PQ | 0.4393 | 0.4415 | 0.4404 |
| pseudo-label込みPQ | 0.4332 | 0.4357 | 0.4345 |
| Δ | -0.0061 | -0.0058 | -0.0059 |
| ペア化95%CI | [-0.0162, +0.0041] | [-0.0162, +0.0036] | **[-0.0133, +0.0011]** |
| リサンプルのうちΔ<0の割合 | 88.7% | 87.5% | **94.95%** |

**fold0・fold1が独立に同じ符号(悪化)・ほぼ同じ大きさ(-0.006前後)で一致した。**
これはfree variant/WBFの不採用判定と同じ「独立2fold一致」という強い証拠の
形であり、V49単独では「fold0だけでは結論不能」だったが、fold1確認により
**pseudo-labelingは不採用と結論する。** 2fold統合(n=472)のペア化95%CIは
[-0.0133, +0.0011]で、上限が0にわずかに届かず統計的な有意水準には届いて
いないが、リサンプルの94.95%がΔ<0(片側検定に近い解釈ではp≈0.05相当)で
有意性の境界に迫っており、実務的には「悪化する」という判断で十分な確度が
ある。CHASE論文が指摘した「モデルバイアス増幅」がこのプロジェクトでも
実際に再現されたと解釈できる。

以上により、検出器強化・アンサンブル・refiner再学習・データ拡張
(pseudo-labeling)の4系統すべてが不採用となり、**確定している最良構成は
依然としてV35(shrink@0.45、全fold OOF PQ 0.4426、Public 0.37)のまま**
である。fold2まで確認すれば2fold統合の検出力がさらに上がり結論がより
確定的になるが、既に十分な確度に達しているため、fold2実施はユーザー判断
とする。

### 2.1.4 診断実験: OOF(0.4426)とPublic LB(0.37)のギャップの原因(2026-08-29)

ユーザーから「ホームページに載せた全fold OOF PQ 0.4426はPublic LB 0.37と
どの程度比較可能か」と問われ、`build_v35_final_submission.py`を精査した
結果、両者は同じ指標(公式PQ)だが条件が異なることが判明した。①〜③(OOF)
は各foldの単一モデルをそのモデルの学習から除外した観測だけで評価するが、
④(Kaggle提出)は3つのfoldモデル全部をIoUクラスタリング+多数決投票で
統合(`ensemble_proposals`、V30本体から一度も検証されずに継承)してから
実際のテストデータで採点する。この統合ロジックが約0.07のギャップの一因
かを直接確認するため、**アンサンブル統合を無効化し、OOF PQ最良の
fold2単体モデルのみ(refiner shrink@0.45・drop_contained_then_area_asc
はV35と同一)を使うV50を実際にKaggleへ提出した**
(`scratch/build_v50_no_ensemble_submission.py`)。

**結果: V50のPublic LBは表示上0.37でV35(アンサンブルあり)と同じだった
が、リーダーボード順位で追加確認したところ V50 ≥ V35 であることが
判明した。** Kaggleは公開スコアを小数点2桁に丸めて表示するため、まず
表示だけでは区別が付かなかった。そこで以下の同日A/B比較を行った:

1. V50提出直後、順位が(提出前の水準から)**上昇**した。進行中のコンペ
   では時間経過とともに新規参入者が増え、同じスコアを維持していても
   順位は下がる方向に働くのが通常であり、この上昇は時間経過とは逆方向
   の変化である
2. 確認のため、**V35を無変更のまま同日に再提出**したところ、順位は
   一切動かなかった(Kaggleは通常チームごとに公開リーダーボード上の
   最良スコアのみを表示するため、順位が動かなかったことはV50が
   引き続き「より良い方」として採用され続けていることを意味する)

これは以下を示唆する:

- **アンサンブル統合は性能を落とさない(むしろ小数点以下でわずかに
  劣る可能性がある)。** 少なくとも明確に有効というわけではない。
  これは§2.1で既に見ていたOOFレベルの傍証(V42のmulti-seed単純連結は
  不合格、WBFは基準にほぼ並ぶが上回れない)とも整合する
- 約0.07のギャップの主因はアンサンブル統合ではなく、**学習(707観測)と
  テスト(180観測)の間の分布差(年代・サイトなど)である可能性が高まった**。
  これはpseudo-labelingの当初の仮説(年代分布の偏りが汎化性能を損なって
  いる)を裏付ける方向の結果だが、pseudo-labeling自体はfold0・fold1の
  独立検証で不採用と確定しているため(§2.1.3)、分布差そのものへの対処は
  依然として未解決の課題として残る
- 実務上の副産物として、**3-foldアンサンブルより計算コストが1/3で済む
  単一モデル(fold2)推論で、V35と同等以上のPublic LBスコアを達成できる**
  ことが分かった。計算効率も評価対象に含まれる本コンペにおいて、
  単純化の余地がある。今後の提出はV50方式への切り替えを検討してよい

### 2.1.5 V51/V52: EdgeAttNet refinerの採用(2026-08-29〜30、確定・Public LB反映済み)

優勝を狙う実験計画([`WINNING_STRATEGY_2026-08-29.md`](WINNING_STRATEGY_2026-08-29.md))
の一環として、EdgeAttNet論文(arXiv:2509.02964)のEdge-Guided Multi-Head
Self-Attention(EG-MHSA)を標準`MediSegUNet`のボトルネックへ移植した
`MediSegEdgeAttUNet`(`solar_filament/models/mediseg_models.py`、
+4.2Mパラメータ)を実装し、標準refinerとの3-fold独立比較を行った。

**方法:** 各foldでEdgeAttNet refinerを学習(該当foldを除く2foldで学習、
crop320/context1.65/14epoch/lr2e-4/CosineAnnealing、標準refinerと完全に
同一レシピ)し、既存V34キャッシュの同一YOLO候補集合上で標準refinerと
ペア比較。観測クラスタ・ブートストラップ(n=2000)でΔのCIを算出した
(pseudo-labeling監査と同一手法)。

| | Δ(EdgeAttNet − 標準) | frac_positive |
|---|---:|---:|
| fold0単独 | +0.006205 | 99.50% |
| fold1単独 | +0.000927 | 73.85%(不確か) |
| fold2単独 | +0.001242 | 77.25%(不確か) |
| **3-fold全結合(n=707)** | **+0.002814** | **99.80%** |

3-fold全結合の95%CIは**[0.000915, 0.004756]でゼロを完全に除外**しており、
pseudo-labelingを否定した基準(94.95% negative)と対をなす、本セッション
最強の肯定的シグナルとなった。fold1・fold2は単独では不確かだったが、
全fold同符号(正)であり、プールにより有意水準に達した。

**実提出での確認(V52):** all-data EdgeAttNet refiner(fold0 checkpointを
ウォームスタートし、低LR(0.35倍)・5epochで全707画像に再学習、V30本体の
all-data refinerと同じレシピ)を構築し、V50と同一のfold2単体YOLO・
後処理(shrink@0.45, drop_contained_then_area_asc)に組み込んで
`scratch/build_v52_edgeattnet_submission.py`で提出。

**結果: Public LB 0.39(V50/V35の0.37から+0.02)。** OOFで確認した
3-fold結合の改善(+0.0028)と同方向で、実際の公開testセットでも改善が
再現した。順位は444チーム中おおむね13〜16位圏(2026-08-30時点、V50提出前
35位から前進)。詳細は`v51_edgeattnet_refiner_3fold_manifest.json`と
`v52_edgeattnet_refiner_fold2_only_manifest.json`。

**結論: EdgeAttNet refinerは以後の主軸refinerとして正式採用。** 提出正本を
V35からV52へ更新する。

**追試(V53、2026-08-30): 3-foldアンサンブルはEdgeAttNet refinerでも
効かない。** V52を土台にYOLO推論をfold2単体から`ensemble_proposals`
(3-fold、IoUクラスタリング+多数決投票、V30本体から変更なし)へ差し替えた
V53を提出したところ、**Public LB 0.38(V52の0.39よりΔ-0.01で悪化)**。
標準refiner基準で確認済みだった「アンサンブル統合は効かない、むしろ
悪化しうる」(V50、§2.1.4)という結論が、refinerアーキテクチャを変えても
再現した。**V52(fold2単体+EdgeAttNet refiner)を引き続き提出正本とする。**
計算コストも1/3で済み、V53を採用する理由はない。詳細:
`v53_edgeattnet_ensemble_manifest.json`。

**追試(閾値再チューニング、2026-08-30): EdgeAttNet refinerでも
shrink閾値0.45は最適点付近のまま、有意な改善なし。** 確率マップを
一度だけキャッシュし閾値だけを振り直す方式(GPU再学習不要)でfold0を
0.25〜0.70の範囲でスイープした結果、0.45〜0.55はほぼ平坦(PQ≈0.4455)、
最良0.50でもΔ+0.0004と現行比で無視できる差。標準refinerで既知だった
「0.45〜0.65はほぼフラット」という性質がEdgeAttNet refinerでも再現した
だけであり、fold1/fold2への追試・ブートストラップ検証は行わない
(効果量が選択バイアスのリスクに見合わない)。**現行閾値0.45を維持。**

### 2.1.6 分布差調査(フェーズ3): 年代・サイト分布に有意差なし(2026-08-30)

V50診断(§2.1.4)で「OOF-Public LBギャップの主因は学習/テスト間の分布差
(年代・サイトなど)である可能性が高い」と示唆されていたため、実際に
定量化した。NSO GONGファイル名規則(`YYYYMMDDHHMMSS[site]h`)から年・
サイトコードをtrain(707)・test(180)双方で抽出し、カイ二乗検定で
分布差を検定した。

| | train | test | カイ二乗 p値 |
|---|---|---|---:|
| 年代(2011-2022、12区分) | 2011-2017が87%・2018-2022が13% | ほぼ同じ比率 | **0.423** |
| サイト(6区分: B/C/L/M/T/U) | 各14-18% | 各14-19% | **0.938** |

**結論: 年代・サイトいずれもtrain/test間で統計的に有意な分布差は検出
されない(両方ともp>0.05)。** 年代・サイト分布は同一母集団からの単純な
無作為分割と矛盾しない。**pseudo-labelingの前提だった「2018-2022偏重が
汎化性能を損なっている」という仮説は、少なくとも年代・サイトという
切り口では支持されない。**

これにより、OOF-Public LBギャップ(fold2 OOF PQ 0.4483 vs V52 Public LB
0.39、Δ約0.058)の真因は依然として未特定と分かった。残る仮説候補
(いずれも未検証): (1) test180枚という小サンプルでの評価ノイズ
(§2.5の検出力議論と同型)、(2) フィラメント密度・複雑度など年代/サイト
以外の画像特性の違い、(3) train/testでアノテーション基準・アノテータが
異なる可能性。詳細: `distribution_shift_audit_manifest.json`。

### 2.1.7 【重大な訂正】アノテータ一致ベースラインのバグ発見・修正(2026-08-30)

技術報告書(`report/main.tex`)を別のAIに査読してもらったところ、
`scratch/run_annotator_ceiling.py`に自己比較混入バグが見つかった。
`for image_id in ids:`が参照アノテータ自身のimage_idも含んでおり、
(1) 単一アノテータ観測(707件中411件)は参照アノテータ=自分自身の
比較しかできず常にPQ=1.0相当としてカウントされ、(2) 複数アノテータ
観測(296件)でも自己比較が他者比較に混入していた。**これにより
これまで使ってきた「アノテータ不一致が課すPQ天井 0.7587(争いのある
画像のみ約0.61)」は大幅に水増しされた誤った値だった。**

自己比較を除外し、単一アノテータ観測を集計対象から除く(独立した
第二の意見が存在しないため)よう修正して再計算した結果、
**正しい値はPQ 0.3398(複数アノテータ観測296枚限定)。** これは
これまでの値の半分以下であり、**我々のOOF PQ(0.4426〜0.445)は
この訂正後の値を上回っている。** これは「モデルがアノテータより
正確」という意味ではなく、モデルが複数アノテータのラベルをプールして
学習した結果、特定の1人のアノテータが別の1人と一致する頻度よりも、
モデルの予測がどの1人とも一致しやすい"コンセンサス的"な境界に
収束している可能性を示唆する——ただしこれは推測であり、今後の
検証課題として扱う。「天井」という呼称も、モデル性能の数学的上限を
意味しないため誤解を招く。以後は「参照アノテーション一致ベースライン」
と呼ぶ。詳細: `scratch/run_annotator_ceiling.py`のバグ修正コメント、
`output/v35/annotator_ceiling_corrected.json`。

**この訂正はプロジェクト全体に影響する。** primer.html・
learn-architecture.html等の公開Artifact/サイトも旧い0.7587を掲載して
おり、更新・再公開が必要(未着手)。report/main.texは訂正済み。

### 2.1.8 3回目の外部査読への対応(2026-08-31)

技術報告書を3回目に査読してもらい、9件の指摘全てをソースコード・
データファイル・外部文献(Crossref API等)で個別に検証した上で反映した。
主な成果:

- **`build_master_pipeline_notebook.py`の再現性バグを発見・修正**:
  YOLO学習呼び出しに`build_v30_prize_colab.py`で実際に使っている
  非デフォルトハイパラ(AdamW/lr0=7e-4/degrees=180等)が漏れており、
  ノートブックをそのまま実行すると論文と異なる設定で再学習してしまう
  状態だった。両方修正し、notebookを再生成・`ast.parse`で検証済み。
- Table 1のスコープ混在(707枚OOF vs fold0限定236枚)を分離し、
  fold0詳細statsを本文の別段落(`\label{sec:fold0-detail}`)に切り出し。
- near-miss「支配的」という過大主張を`near_miss_modes.json`の実測値
  (near-miss 955件 < hard-miss 2,138+1,561件)で訂正。
- MAGFiLO引用の書誌情報誤り(ページ数→article number、月）を
  Crossref APIで確認し修正。
- 査読者の「5ページは制限違反」という指摘は、Overleaf/ACMテンプレート
  自体が明記する「本文4ページ(謝辞・参考文献を除く)」という制限文言と
  照合した結果、字義通りの違反ではないと判断(本文はFuture work末尾で
  page5冒頭の1カラム途中まで、page5の残りは謝辞+参考文献)。
  詳細・全9項目の検証結果は`WINNING_STRATEGY_2026-08-29.md` §6.6。

### 2.1.9 【重大な撤回】matched-parameter ablationでEdgeAttNet refinerの改善を再現できず(2026-08-31)

§2.1.5で「確定・Public LB反映済み」としたEdgeAttNet refinerの改善効果は、
matched-parameter・multi-seedの追試で**再現しなかった**。base(プレーンU-Net、
31.04M params)/plain_mhsa(標準MHSA・edge priorなし、35.24M params)/
edge_att(edge-guided MHSA、35.24M params、V51と同一)の3構成×seed(3407,8407)
=6本をfold0で学習・OOF評価した結果:

| variant | 2seed平均 fold0 OOF PQ | std |
|---|---:|---:|
| base | **0.441412** | 0.000593 |
| plain_mhsa | 0.441259 | 0.000795 |
| edge_att | **0.440474**(3構成中最低) | 0.0000315 |

edge-attentionはbaselineを上回らず、むしろ最低だった。§2.1.5の3-fold delta
(fold0 +0.0062、fold1 +0.0009、fold2 +0.0012)のうち後者2つは、今回実測した
seed間ノイズ(std 6-8e-4)より小さく、「3fold全て符号一致」という採否基準自体が
学習ノイズで説明できる水準だった可能性が高い。

新規モデルクラス`MediSegPlainMHSAUNet`を`solar_filament/models/mediseg_models.py`
に追加(ビルダー: `scratch/build_matched_param_ablation.py`)。生データ:
`output/ablation/matched_param_ablation_fold0_manifest.json`(ユーザーがColabで
実行→貼り付けた出力の転記、機械照合ではない旨をmanifestの`provenance`に明記)。

**report/main.texは正直に書き換え済み**(Abstract/Table 1/§3.2/§3.3を全面改訂、
「唯一検証を生き残った改善」という主張を撤回しこの反転自体を主要な知見として
報告)。詳細は`WINNING_STRATEGY_2026-08-29.md` §6.9。

**未確定のまま残ること: このablation自体n=2・fold0限定なので断定はできない
(more seeds・fold1/2への拡張が次点)。提出モデル(現在Public 0.39のedge-attention
refiner)をplain refinerへ差し戻すかも未決定。**

### 2.1.10 【完了・確定】matched-parameter ablation 3fold全て完走(2026-08-31)

§2.1.9のfold0限定ablationを、Gドライブ直接検証(生JSON・実行済み
notebook、execution_count 1-8・cell7に181出力、いずれもチャット転記値と
完全一致)を伴ってfold1・fold2にも拡張し、3fold全て完走した。

| fold | base | plain_mhsa | edge_att |
|---|---:|---:|---:|
| fold0 | 0.4414 | 0.4413 | 0.4405 |
| fold1 | 0.4405 | 0.4426 | 0.4417 |
| fold2 | 0.4434 | 0.4458 | 0.4464 |

report自身が使うfold-sign-agreement基準で見ると: **attention容量
(edge priorの有無を問わず)を足すこと自体は3fold中2fold(fold1・fold2)
で両seed符号一致の正、fold0はほぼゼロ——概ね効いている。しかし
edge_att − plain_mhsaは3fold中2fold(fold0・fold1)でゼロ以下、
fold2はseed間で符号不一致——どのfoldもedge priorがplain attentionを
明確に上回らない。**

**最終結論: 「偽陽性だった」ではなく「効果はattention容量一般に
起因し、edge prior固有の主張は支持されない」という、より精密な
着地点。** 3fold全体のmicro-averaged PQ(TP/FP/FN summingで算出):
base 0.4418 < edge_att 0.4429 < plain_mhsa 0.4432(**plain_mhsaが
最高**)。

**report/main.texを全面反映済み**(Abstract/Table 1(3fold全体行)/
§3.2(fold-sign-agreement分析)/§3.3(「plain-attention refinerを
再提出候補として検討」という新しい実務提言を追加)。詳細は
`WINNING_STRATEGY_2026-08-29.md` §6.11。生データ・実行済みnotebookは
全fold分`output/ablation/`に保存済み(3fold全て機械照合済み、
`matched_param_ablation_all_folds_summary.json`に統合解析あり)。

### 2.1.11 【重要】リーダーボード上位クラスタはテストセットリーク由来と判明、追走を明確に中止(2026-09-15)

Public LB 0.55〜0.56に多数のチームが並ぶ異常なクラスタ(2026-09-15時点で
上位56チーム中、0.56が3チーム・0.55が35チームと完全一致)について、Kaggle
公式ディスカッションフォーラムを直接調査した。

**検証済みの事実(JAB100氏の投稿、`discussion/728062`。Raffi Ardhi Naufal氏が
メタデータレベルで独立に再現・確認済み):** 競技のテストセット180枚中139枚
(77.2%)が、公開データセットMAGFiLO 1.0(正解アノテーション付き)と**バイト
単位で完全一致**している。JAB100氏は実際に自モデル予測をMAGFiLO 1.0の正解へ
差し替えて提出し、スコアが0.73→0.82に跳ね上がることを確認した。フォーラムの
別の分析(未検証の推測、`discussion/737177`)では、この重複が2つのモードで
悪用されうるとされる: (1)重複77%分は公開正解をそのまま提出・残り23%は自モデル
予測という直接ブレンドで0.5〜0.66に着地、(2)リーク正解を学習データへ混入して
再学習する"洗浄"で0.45〜0.55帯に着地(この場合リーダーボード再採点では検出
不可能)。35チームの0.55完全一致クラスタと3チームの0.56クラスタは、この数値
予測とよく整合する。

**運営(Azim Ahmadzadeh氏)は複数スレッドで一貫して以下を明言している
(直近は6日前):**
> "our evaluation is not based on the scores shown in the leaderboard...
> Submissions without a decent strategy will NOT be rewarded in any ways."
> "Any PQ score of greater than 0.35 is of great value to us."

評価は§1に記載の**定量70%+定性30%(パイプラインの妥当性・再現性・コード
品質・個々の予測の判定)の合議制**であり、リーダーボードは一次フィルタに
過ぎないと明言されている。またMAGFiLO本家データセットの学習利用は競技規約
(Rules §2.6、既確認済み)で明確に禁止されているため、このリークを利用する
ことは**規約違反**であり、追走してはならない。

**結論・今後の方針への反映:**
1. **0.55の壁はモデルの実力差ではなくほぼ確実にテストリーク由来と判断し、
   これ以上追走しない。** 自チームのPublic 0.39(V52)は、リークを一切使わない
   誠実なモデルとしては運営の物差し(0.35超で価値あり)の上で妥当な位置にある
   (0.39〜0.40帯に他チーム十数件が集中しており、これがおそらく"正直な"ティア
   の実態)
2. 自チームはMAGFiLO本家の正解アノテーションを一度も学習に使っていない
   (pseudo-labeling実験は別出典のGONG生データ+自社モデル生成ラベルのみ、
   §2.1参照)。この点をreport/main.texに明記し、リーク問題が公になった
   以上の信頼シグナルとして活用する
3. 定性30%(報告書・再現性・コード品質)への投資を最優先級として継続する
4. 競合の公開コード(GitHub、誠実なティアの技術動向)を調査し、改善レバーの
   候補を洗い出した。**ただし着手前にコードを再確認したところ、boundary
   loss(境界マップのL1損失)は実はV30から一貫して`refiner_loss`に含まれて
   おり、未試行という前回の分析は誤りだったと判明(instrument-before-guessing、
   §2.1.12参照)。** 実際に未試行なのは`EMA/SWA`・refiner入力のマルチ
   チャンネル化・スケルトンベースのギャップ閉合の3つ(優先順位は
   `WINNING_STRATEGY_2026-08-29.md` §6.12に記載、訂正済み)

一次情報: Kaggle ディスカッション
[728062](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062)・
[737177](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/737177)。

### 2.1.12 【訂正】boundary lossは既に実装済み、実装着手前の再確認で発覚(2026-09-15)

§2.1.11で「未試行の改善レバー」として挙げた4項目のうち、boundary loss
(境界重視損失)の追加に実際に着手しようとして`scratch/build_v51_edgeattnet_
refiner.py`等の実際の学習コードを読み直したところ、`refiner_loss`関数が
既に以下を実装していることが判明した:

```python
def boundary_map(tensor):
    dilation = F.max_pool2d(tensor, 5, stride=1, padding=2)
    erosion = -F.max_pool2d(-tensor, 5, stride=1, padding=2)
    return torch.clamp(dilation - erosion, 0, 1)

def refiner_loss(logits, targets):
    region = 0.35 * F.binary_cross_entropy_with_logits(logits, targets) + 0.65 * soft_dice_loss(logits, targets)
    edge = F.l1_loss(boundary_map(logits.sigmoid()), boundary_map(targets))
    return region + 0.15 * edge
```

`grep`で確認したところ、この`refiner_loss`は**V30(最初の入賞狙いバージョン)
からV51・matched-parameter ablationまで一言一句変わらず使われ続けている。**
つまりbase/plain_mhsa/edge_attのmatched-parameter ablation(§2.1.10)は
すべてこの境界損失込みで学習されており、「boundary lossを足せば伸びるか」
という前回の分析の前提そのものが誤りだった。

**教訓([[feedback-instrument-before-guessing]]の実践例、今回は実装コードの
事前確認という形で):** 前回の分析は`solar_filament/models/mediseg_
models.py`の`CombinedLoss`(BCE+Diceのみ、実際には現行パイプラインでは
未使用の旧クラス)だけを見て「境界損失なし」と判断してしまい、実際に学習
ループで使われている`refiner_loss`(notebook builderスクリプト内に
埋め込まれた文字列コード)を確認していなかった。**実装に着手する前に
「本当に未試行か」を実コードで確認する**という、このプロジェクトが繰り返し
実践してきた規律を、今回は分析側で一度踏み外し、実装直前の確認で自己修正
できた。

**再整理した優先順位(WINNING_STRATEGY_2026-08-29.md §6.12を訂正済み):**
1. EMA/SWA(重み平均) — 確認済みで未試行、低コスト
2. refiner入力のマルチチャンネル化(距離変換・binary seed) — 確認済みで
   未試行(現状は3chの色画像crop部分ですきりの入力)
3. スケルトンベースのギャップ閉合 — 確認済みで未試行
4. (副次)既存boundary lossの強化 — 「初導入」ではなく「現行の粗い
   morphological-gradient定式化(重み0.15固定)を、距離変換ベースの
   より精密なboundary loss(例: Kervadecらのdistance-transform boundary
   loss)へ置き換える」という、新規軸ではなく改良軸として位置づけ直す

**(2026-10-05追記) 上記レバーの実験ステータス**: 優先度1(EMA、v54)は
fold0でΔ+0.0005の微小正だったがfold1でseed間符号不一致(−0.00004/
+0.00101)、fold2も両seed負(−0.0008/−0.0002)となり3fold通して不採用確定。優先度2(マルチチャンネル
化、v55)はΔ−0.0206で不採用確定、優先度3(スケルトンギャップ閉合、v56)
はgap結合の寄与が最大+0.0004程度で不採用。さらに優先度6(密集シーンの
学習サンプルoversampling、v57)はfold0で2seedともrefined PQ悪化
(−0.0014/−0.0085。dense FNは−49/−19と改善したがFPが+229/+146と爆増し
相殺)で不採用。残る未着手レバーは優先度4(現行boundary lossの精密化)・
5(IoU>0.5断崖向け閾値較正)・7(SAM2系refiner)。各実験の詳細は
`WINNING_STRATEGY_2026-08-29.md` §6.14〜6.16。

**(同日追記2) v57の陰性結果を踏まえた3本の後続パイロットを準備済み**
(`WINNING_STRATEGY_2026-08-29.md` §6.17、Colab実行待ち):
v58=提案confスイープ(学習なし・v57の既存ckpt使い回しでconf曲線を掃引、
fold0上の探索なので採用点はfold1/2独立確認が条件)、v59=距離変換
boundary loss(優先度4、refiner側の単変数置換)、v60=copy_paste
augmentation(優先度6の代替機構、シーン複製でなくインスタンス貼付)。

**(同日追記3) v59・v54確定 + v58確定**(§6.18): v59 dtbは3fold完結 —
fold0 [+0.0015/+0.0003]・fold1 [−0.0001/+0.0007]・fold2 [−0.0018/−0.0003]、
全6観測で平均≈±0・正負3対3・最大|Δ|0.0018の完全なノイズで
**不採用確定**(優先度4完了)。境界精度(SQ)を上げる効果自体は
fold2で検出できたがrecallとのトレードオフで正味ゼロ以下。v54 EMAはfold2も両seed負(−0.0008/−0.0002)で3fold通して
不採用確定。v58確定: prodのconfピークはfold0/fold2で0.26一致
(現行操作点は適切)。dense2xは「両seed・同confでbaselineを上回る点」が
存在せず、各config最良点同士の比較でも負けるためconf救済は棄却確定 —
oversampling系(優先度6)は完全に閉じた。

**(同日追記4) v60確定**(§6.18): copy_paste=0.2 は fold2 で両seedとも
仮説と逆方向 — dense FN は減らず増え(+23/+81)、FPは一貫して減少
(−47/−48、過剰提案抑制の正則化として作動)。ΔPQは−0.0007/−0.0200で
不採用確定(s8407はearly-stopで学習不安定化)。これで検出器側の
密集リコール系レバー(優先度6系: シーン複製・conf救済・インスタンス
貼付)は機構の方向を問わずPQ正味で改善しないことが確定。

**(同日追記5) v61確定 — 優先度7終了、全レバー実験完結**(§6.19):
SAM2.1 hiera_large の box-prompt zero-shot refiner 置換は、同一proposal
paired で sam2_shrink −0.0348・sam2_free −0.1354 と大幅負で**不採用確定**。
shrink揃えでもSQはraw並(U-Net refinerの+0.027 SQに届かず)、freeでは
box外はみ出しでFP+469・RQ崩壊 — ドメインギャップ敗北が確定的。これで
§6.12の優先度1〜7(EMA・マルチチャンネル・スケルトン・密集
oversampling・conf救済・距離変換境界損失・copy_paste・SAM2)は
**すべて実験済み・全件不採用**が確定し、現行V30+refiner shrink構成が
試した全代替案に対して最良のままであることが確認された。

**(2026-10-05、V57 記録回収)** 密集シーン oversampling の既存 fold0 pilot は
2seed とも paired refined PQ が悪化し、不採用。既存 Drive manifest の4runを
読取りで照合した（詳細・SHA256 は WINNING_STRATEGY §6.16）。回収版は入力・重みの
指紋による再開契約と CPU テストを補修し、GPU 実行・独立受入は未実施。
fold1/2 の追加実験・採用・提出構成の変更は行わず、S1 blocked gate は維持する。

### 2.2 科学・実運用上の位置づけ

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

**`solar_filament/` が実装の正本パッケージ。** `src/` のうち
`blob_split.py`・`candidate_verifier.py`・`io_images.py`・`mask_wbf.py`・
`mediseg_models.py`・`metrics_official.py`・`oof_cache.py`・`ownership.py`・
`paths.py`・`refine.py`・`rle.py`・`science_metrics.py`・`timing.py`・
`v32_overlap_cluster.py` は後方互換のためのre-exportシム層で、実体はすべて
`solar_filament/` 側にある(例: `src/metrics_official.py` は
`from solar_filament.metrics.metrics_official import *` するだけの1行
モジュール)。旧`scratch/`スクリプトの `from src.xxx import ...` はそのまま
動くが、**新規コードは `solar_filament.xxx` を直接importすること。**

**⚠️ `config.py`・`dataset.py`・`model.py`・`train.py`・`infer.py`・
`utils.py` の6つはシムではない。** V30以降は使われなくなった512×512/5-fold
の旧パイプラインの実装がそのまま残っているだけで、`solar_filament/`
側には対応物が無い(移行されていない)。特に `utils.py` の `rle_encode`/
`rle_decode` は Kaggle形式(10進ペア文字列)であり、`solar_filament/pipeline/
rle.py`(pycocotools圧縮カウント形式)とは**別のコーデック**なので、
新規コードで誤って `src.utils` 側を使うと提出CSVの形式が壊れる。

```text
Solar_Filament_Segmentation/
├── solar_filament/              # ★実装の正本パッケージ
│   ├── metrics/metrics_official.py   # 公式PQ実装。新しい評価はすべてこれを使う
│   ├── models/mediseg_models.py      # MediSegUNet (31M params) ほか
│   ├── pipeline/                     # v32_overlap_cluster・mask_wbf・ownership・refine等
│   ├── s1/                           # Scientific/Operational S1トラック(別セッション管理)
│   └── candidate_verifier.py / timing.py
├── src/                          # ほぼ後方互換シム層。実装はすべてsolar_filament/側
│   └── config.py / dataset.py / model.py / train.py / infer.py / utils.py
│                                 # ⚠️ 旧構成(512x512/5-fold)。V30以降は未使用、シムでもない生きた旧実装
├── tests/
│   └── test_metrics_official.py   # 19件。公式実装との差分テストを含む
├── scratch/                    # ノートブック生成器と診断スクリプト（§5参照）
├── notebooks/                  # 初期のColabスターター
├── output/
│   ├── v32/                    # 提出候補CSVとaudit、manifest（SHA-256付き）
│   ├── v35/                    # PQ再採点・天井診断・アブレーションの結果
│   ├── refiner_retrain/        # V46/V47のmanifest・チェックポイント・評価レポート
│   └── pseudo_label_raw/       # pseudo-labelingパイロット用のGONG生データ・変換画像
├── README.md                   # ★現在地の正本。まずここを読む
├── RESEARCH_LOG.md             # 実験カタログを論文構成に整理したノート
├── TECHNICAL_REPORT_DRAFT.md   # コンペ提出用4ページ報告書の草稿(生きている文書)
└── solar_filament_v*.ipynb     # 各版のColabノートブック（scratch/build_*.py から生成）
```

> ⚠️ **`src/config.py`・`dataset.py`・`model.py`・`train.py`・`infer.py`・
> `utils.py` を読んで V30 以降の実験条件を判断してはいけない。** 汎用の
> 512×512 / 5-fold / UnetPlusPlus 設定であり、V30 以降は各ビルダー内に独自 CFG を持つ。
> 設定の二重管理は既知の技術的負債（HANDOVER §17.4）。これら6ファイルは`src/`の
> 他のモジュールと異なりシムではなく、本当に使われなくなった旧実装がそのまま
> 残っているだけ。

**ノートブックは直接編集せず、`scratch/build_*.py` を直してから再生成すること。**
ipynb の JSON を手編集するより再現性が高い。

---

## 5. 再現手順

### 5.0 提出物の再現(審査用・ここだけ見ればよい)

**`solar_filament_submission_reproduction.ipynb`** が公式の再現 notebook。
「すべて実行」で前処理 → 検出器学習 → refiner 学習 → test 推論 →
`submission.csv` 生成まで通る。Kaggle(`/kaggle/input` 自動検出)と
Colab(Google Drive)の両方で動作し、学習済み重みを添付 dataset /
Drive artifacts から解決する(無ければフル再学習)。末尾セルで
`pip freeze` と重み SHA256 を記録する。
推論のみの短い経路は上記「最終提出の再現（最短経路）」を参照。

第2提出(`plain_mhsa_f2`)も同じ notebook で再現できる:
setup セルの `CFG.SUBMISSION_VARIANT` を `"plain_mhsa_f2"` に変更して
再実行すれば、添付 dataset の `ablation_plain_mhsa_seed8407_fold2.pt` を
解決して `submission_plain_mhsa_f2.csv` を生成する(主催者の
「選択した両方の提出を再現できること」要件への対応、2026-10-12)。

生成元は `scratch/build_submission_reproduction.py`。
以下の §5.1-5.4 は開発履歴・負の結果の再現用で、提出の再現には不要。

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
python scratch/build_v44_kaggle_rfdetr_probe.py           # RF-DETR-Seg fold0検証（不合格済み）
python scratch/build_v45_colab_rfdetr_hires_probe.py      # RF-DETR-Seg 768px版（不合格済み）
python scratch/build_v46_colab_refiner_free_retrain.py    # refiner warm start再学習（不合格済み）
python scratch/build_v47_colab_refiner_scratch_retrain.py # refinerゼロから再学習（不合格済み）
```

生成された `.ipynb` を Colab で開いて実行する。重みは Google Drive に保存される
（本リポジトリには含めない）。

### 5.4 pseudo-labelingパイロット（完了・不採用、§2.1.1参照）

```bash
python scratch/fetch_gong_pseudo_label_pilot.py --count 150     # NSO GONGアーカイブからFITS取得
python scratch/convert_gong_fits_to_jpeg.py                     # 実データと同階調へヒストグラムマッチング変換
```

取得したFITSは`output/pseudo_label_raw/fits/`、変換後JPEGは
`output/pseudo_label_raw/images/`に保存される（いずれもGit管理外）。
以降のpseudo-label生成・QA・学習手順は§2.1のフェーズ計画を参照。

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
- **LB 0.55〜0.56 クラスタを追走する。** テストセット77.2%がMAGFiLO 1.0と重複する
  リーク由来と判断済み（§2.1.11）。MAGFiLO本家データセットの学習利用は規約違反


### v59 boundary-DT pilot の CPU 回収版

`scratch/build_v59_boundary_dt_pilot.py --fold {0,1,2}` で各 fold の notebook を再生成する。
境界項のみを距離変換の離散近似に置き換える試験で、CPU の数値・入力契約・notebook 同期を検証した。
入力・検出器・実際の初期状態と完了 checkpoint の指紋を記録し、旧 v54 baseline は既定で再利用しない。
過去の GPU 結果を新しい受入証拠として採用せず、学習・独立 fold 評価は別途必要。


### v60 copy-paste pilot の CPU 回収版

`scratch/build_v60_copypaste_pilot.py --fold {0,1,2}` で notebook を生成する。
同じ dataset・seed・訓練レシピで `copy_paste=0` と `0.2` だけを変える試験。
入力と実 checkpoint の契約を確認し、旧 v57 baseline は再利用しない。この v60 契約で実行済みの baseline 記録だけを明示 opt-in で取り込める。
CPU 契約検証は GPU 訓練・独立 OOF 受入を代替しない。
