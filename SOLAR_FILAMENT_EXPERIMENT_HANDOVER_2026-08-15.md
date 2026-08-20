# Solar Filament Segmentation Challenge 2026 実験経緯・技術引継ぎ

- 更新日: 2026-08-18（改訂9: 検出器強化6本・後処理再設計(watershed)・multi-seedアンサンブルの**全経路が不合格**。V35確定パイプラインはこの規模のデータに対してかなり頑健な局所最適であることが複数の角度から裏付けられた）
- 対象プロジェクト: `C:\work\code\ER-FlowScan\Solar_Filament_Segmentation`
- 対象範囲: Kaggle提出形式の修正、V19～V35のモデル・後処理実験、Colab/Kaggle運用、既知の結果、未解決事項、次の推奨方針
- 現在の結論: **確認できたPublic最高スコアは0.37（V35、2026-08-16提出）。** V34は全OOFゲート不合格のため提出していない。
- **改訂2の要点: リーダーボード指標は Panoptic Quality であり、V19〜V34が最適化してきた Hungarian matched Dice ではなかった。§2.1 を先に読むこと。§7・§8.1・§13.2・§26 の結論は §2.1 で差し替えている。**
- **改訂3の要点: fold1/2のrefinerを学習し、refiner×排他化×confを全fold OOFで同時に確定した（リークなし、707観測）。最良構成は `conf 0.26 + shrink@0.45 + drop_contained_then_area_asc` で **PQ 0.4426**（生YOLOベースライン比 +0.0489）。新ゲートは **PQ > 0.4526**。§2.1.8.1・§2.1.8.2 を参照。**
- **改訂4の要点: 確定構成で test 180枚を推論し提出した。Public **0.37**（自己ベスト、V30b/V32の0.35から +0.02）。OOF 0.4426 との差は残る。次は検出器解像度（P2-2、`build_v36_yolo_imgsz_probe.py`）。§2.1.8.3 を参照。**
- **改訂5の要点: P2-2（YOLO を imgsz 1536 で fold0 再学習）は OOF PQ **0.4295 < 0.4393** で不合格。fold1/2 には進まない。提出正本は V35 のまま。次は学習せず V30 重みで推論 imgsz だけ 1536（P2-3）。§2.1.8.4 を参照。**
- **改訂6の要点: P2-3（V30 重みのまま推論 imgsz 1536）は fold0 PQ **0.4320 < 0.4393** で不合格。学習解像度も推論解像度も上げても確定後処理の下では勝てない。解像度レバーはここで止める。提出正本は V35。§2.1.8.5 を参照。**
- **改訂7の要点: 全707観測でcase diagnostics（`output/v35/case_diagnostics.csv`）とnear-miss被覆モード分類（`output/v35/near_miss_modes.json`）を完走した。near-miss 955件中 undercoverage 433件(45%)・overcoverage 400件(42%)とほぼ拮抗し、単一モードが支配的ではない。ただしIoU≤0.3の完全な取りこぼし(hard FN 2138件)がIoU≤0.3の余剰検出(hard FP 1561件)を大きく上回り、loss_unitsは近接ミス数(r=0.853)とGT個体数(r=0.515)・proposal数(r=0.533)に強く相関する。損失最大の画像はGT 10〜21個体の密集シーンに集中する。§27.1 を参照。**
- **改訂8の要点: Kaggle無料GPU枠でV38(augmentation強化: mosaic0.20→0.75+copy_paste0.3)とV39(モデルサイズ拡大: yolo11m-seg→yolo11l-seg、batch4→2)をfold0のみ・70epochで実行。fold0 OOF PQは V38が0.4290・V39が0.4328で、いずれもV35基準0.4393を下回り不合格。生YOLO(refinerなし)で切り分けたところ、P2-2/P2-3(後処理相性の問題で検出器自体は改善していた)と異なり、V38・V39は生YOLOの時点で既にV30基準(0.3977)を下回っており、検出器そのものが悪化している。ただしV38はV35と同じ70epochのままaugmentationだけ強めており収束不足の疑いが強く、V39はbatch4→2でlr0を据え置いたままなので実効バッチサイズ減少による学習不安定化が疑われる。単純な「強くする」だけでは効かないことが分かったが、学習スケジュール自体を合わせていない単軸変更なので、この2実験だけで「augmentation強化」「モデル大型化」という方向性自体を完全に否定はできない。§27.2 を参照。**
- **改訂9の要点: 後処理再設計(watershed癒着分離)は実データで閾値に極端に敏感で不採用(§27.3)。V38/V39の交絡を補正した再測定(V40: epoch70→100+warmup3→5、V41: lr0 7e-4→3.5e-4)は、いずれも改善せず**むしろV38よりV40がわずかに悪化(0.4290→0.4272)**。V41はKaggleのGPUバッチセッション上限(他プロジェクトと共有)とユーザーのGPU無料枠残量懸念によりColab(Drive経由、resume機構つき)へ切り替えて完走(fold0 PQ 0.4316)。imgsz・augmentation・モデルサイズの検出器強化系が**6本連続で不合格**(§27.4)。さらにmulti-seedアンサンブル(V42: seed3407+8407を単純連結)も不合格(fold0 PQ 0.4315、Δ-0.0078、FP+190がTP+53を上回る、§27.5)。**後処理再設計・検出器強化・アンサンブルの全経路が不合格**となり、V35確定パイプラインはこのデータ規模に対してかなり頑健な局所最適であることが複数の角度から裏付けられた。**

## 1. この文書の読み方と証拠の区分

この文書は、ローカルのノートブック／ビルダースクリプト、Kaggle実行ログ、Colab出力、ユーザーが共有したKaggle画面を統合した引継ぎ資料である。

数値の意味を混同しないことが重要である。

- **Public Score**: Kaggle公開リーダーボードの実スコア。最も直接的だが、公開test部分への適合を含む。
- **strict instance Dice / held-out instance Dice**: 個体maskをHungarian matchingして算出したローカルOOF指標。Public Scoreと同一とは限らない。
- **semantic/union Dice**: 画像内の全フィラメントを結合した前景maskのDice。個体分離性能を評価しない。
- **crop Dice**: 検出済み個体crop内部のrefiner精度。検出漏れや個体結合を評価しない。

本文では以下の区分を使う。

- **確認済み**: ログ、manifest、Kaggle画面で直接確認した事実。
- **推定**: 複数の結果から合理的に推定した原因。
- **未確認**: 追加調査が必要な事項。

## 2. エグゼクティブサマリー

1. 最初の提出失敗はモデル以前に提出形式が原因だった。`null`、誤った列名、空mask、非公式RLE、個体maskの重複を順番に修正した。
2. 形式修正後のV24は提出には成功したがPublic `0.00`。添付checkpointが実質的に合わず、1画像1semantic mask中心だったことが主因と判断した。
3. 実データ学習のMediSeg semanticモデルはunion Dice約`0.63`まで学習できたが、連結成分による個体化ではPublic `0.19`止まりだった。
4. V30でYOLO11m-seg 3-foldを個体検出に使い、MediSegを個体crop refinerとして組み込んだ。重複を修復したV30bがPublic `0.35`。
5. V31/V32で後処理を厳密化したがPublicは`0.34～0.35`で頭打ち。後処理だけでは大幅改善しなかった。
6. 公開ノートブックの申告Public `0.69`を再現するV33を作ったが、実提出は`0.33`。0.69との差の原因は未解明。
7. V34はsemantic-firstへ処理順を反転したが、全OOF strict instance Dice `0.441074`、個体数誤差`2.846535`でV31より大幅悪化。提出ゲートが正しく停止した。
8. 根本ボトルネックは、refinerのcrop境界精度よりも、**個体の検出漏れ、同一個体の重複、近接個体の結合／分断、検証と最終推論の不一致**にある。
9. V35で後処理を全fold OOFのPQに整合させて提出し、Public **0.37** を得た（§2.1.8.3）。P2-2（imgsz 1536 再学習）と P2-3（推論 imgsz 1536）はいずれも fold0 で不合格（§2.1.8.4・§2.1.8.5）。解像度レバーは停止。提出正本は V35。

## 2.1 【2026-08-15 改訂2】公式指標の確定と、それに伴う結論の差し替え

### 2.1.1 確定した事実

Kaggleの Evaluation / Leaderboard Ranking タブ、および主催者が公開した自己評価ノートブックの実装を直接確認した。

**リーダーボード指標は Panoptic Quality (PQ) である。**

```text
PQ = Σ IoU(TP pairs) / (|TP| + 0.5|FP| + 0.5|FN|),   TP ⟺ IoU > 0.5
```

- 出典（実装）: <https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook> (Apache-2.0)
- 出典（原典）: Kirillov et al., CVPR 2019 (10.1109/CVPR.2019.00963)。ただし**公式実装は原典そのままではない**ので、原典ではなく公式実装に合わせること。
- **2026-08-07に評価方法が変更され、2026-08-12にリーダーボードが再採点された。** それ以前の指標は Dice (`torchmetrics.segmentation.DiceScore`)。再採点は過去の提出にも遡及適用されたので、§6のPublic Score履歴は現時点ではすべてPQ値として読んでよい。
- ただし**§6末尾の「1位0.48／2位0.40」のスクリーンショットが8/12以前のものなら、その数値は既に無効**。取得時刻を確認できないため `unknown` として扱う。

**MIoU (Multiscale IoU, arXiv:2105.14572) はリーダーボード指標ではない。** IEEE Big Data Cupの紹介ページには挙がっているが、KaggleのEvaluationには含まれない。細構造(barb)の再現性を測る補助指標、および技術報告書の裏付けとしてのみ使う。

### 2.1.2 公式実装から読み取った、判断に効く仕様

1. **IoU > 0.5 は硬い崖。** IoU 0.5 は Dice 0.667 に相当する。これを下回る予測は得点ゼロなだけでなく、FPとして分母を+0.5し、対応するGTもFNとしてさらに+0.5する。**惜しい予測は「無得点」ではなく「二重の減点」**。refinerのcrop Dice 0.799は平均値なので、分布のうち0.667を下回る部分は純粋な負債である。
2. **評価単位は (annotator, image) ペア。** GTの`filament_id`は`<annotator_id>-<image_id>_<n>`形式で、annotator-imageごとに1行を作り、**同一の予測集合を各アノテータのGTと個別に突き合わせて、TP/FP/FNを全体へ足し込む**。§18.4に書いた「variantごとのDiceを平均する」扱いとは別物。複数アノテータ画像は事実上その人数ぶん重み付けされる。
3. **分母は全画像を通したmicro集計。** 画像単位のマクロ平均ではないので、個体数の多い画像が支配的になる。
4. **FPは「どのGTともIoU>0.5にならない予測」**で、アノテータごとに数える。
5. **追加候補の損益分岐が低い。** TP確率p・想定IoU uの候補を1つ足すときの期待PQ変化から、条件は `p*u > 0.5*PQ`。**PQ 0.35・IoU 0.7 なら p > 0.25** で足りる。過剰検出が`max(|GT|,|pred|)`の分母で直接罰せられるHungarian Diceとは最適動作点が大きく違う。

### 2.1.3 差し替える結論

| 元の記述 | 改訂後 |
|---|---|
| §7「最も信頼できるローカル比較軸は全fold OOFのstrict instance Dice」 | **誤り。** strict instance Diceは公式指標ではない。比較軸はPQ（およびSQ/RQ分解）。 |
| §8.1-3「個体数誤差が重要」 | 方向は正しいが、指標はcount errorではなく**FN/FPが分母へ与える0.5ずつの寄与**で測る。 |
| §8.1-7「排他化は必須だが先着画素削除は形状を壊す」 | **格上げ。** V30bの660,119画素の先着減算はIoUを直接下げるので、**TPを0.5の崖の下へ突き落とした可能性がある**。形状の美観ではなく得点の一次要因。 |
| §9.1-1「公開0.69を再現できない理由が不明」 | **調査打ち切り。** §6のLB 1位が0.48である以上、0.69はこのLBの値ではありえない。V33の0.33が正しい実測。 |
| §9.1-3「公式metricとローカル指標の一致性が未確認」 | **解決。** 一致していなかった。§2.1.1の通り。 |
| §9.1-6「複数アノテータの最適な扱いが未解決」 | **方向が決まった。** PQは各アノテータへ個別採点して合算するので、**両アノテータでIoU 0.5を越えられる予測は TP が2個入る**。consensus的な予測が有利になる。 |
| §13.2 acceptance gate `0.552553` | **廃止。** 公式でない指標であり、かつfold 0のみの探索bestで選択バイアスがある。§2.1.4の手順でPQ基準に置き換える。 |
| §26 P0〜P3 | PQ下では**まずSQ/RQ分解で主因を確定**してから仮説を選ぶ。§2.1.4参照。 |

### 2.1.4 ゲートの再設定手順（PQ基準）

`solar_filament_v35_metric_realignment_colab.ipynb` が以下を一度に行う。学習もtest推論もせず、V34 DriveにあるV30 YOLOの全fold OOF cache（2048×2048のまま保存済み）を読み直すだけなのでGPU不要。

1. conf 0.08〜0.45 を掃引して全fold OOFのPQを測り、動作点を引き直す
2. **PQ = SQ × RQ に分解**する。SQ=TPの平均IoU（形状の質）、RQ=TP/(TP+0.5FP+0.5FN)（検出の質）。どちらが支配的かで次の一手が変わる
   - RQが低い → 取りこぼし・過剰検出が主因。境界を磨くより候補を増やす
   - SQが低い → IoU 0.5を越えられていない。境界品質の改善
3. near-miss（IoU 0.3〜0.5）の予測数を数える。ここが厚ければ「検出は当たっているが崖を越えられていない」
4. fold別PQを併記し、平均に隠さない
5. 新ゲート: all-fold PQ > 実測ベースライン+0.010、かつ最悪foldが実測値を下回らない

### 2.1.5 アノテータ不一致が課すPQの天井【測定済み】

PQは全アノテータに対して採点するため、**あるアノテータのGTをそのまま提出しても満点にはならない**。その値がこのデータで到達しうるPQの実質的な上限であり、GTのみで計算できる。

`scratch/run_annotator_ceiling.py` をローカルPCで実行して測定した（GPU・モデル不要、COCO annotationのみ）。結果は `output/v35/annotator_ceiling.json`。

| 参照アノテータ | 対象観測 | PQ | SQ | RQ |
|---|---:|---:|---:|---:|
| 0番（全観測） | 707 | **0.7587** | 0.9133 | 0.8307 |
| 1番（複数アノテータのみ） | 296 | **0.6051** | 0.8365 | 0.7234 |
| 2番（3名以上のみ） | 151 | 0.5570 | 0.8105 | 0.6872 |

- 総annotation数は`8199`、annotator-image数は`1154`（707 + 296 + 151）
- **0.7587は411件の単一アノテータ観測（自己一致で自動的に満点）を含むため上振れしている。** 争いのある画像だけに限れば天井は約`0.61`。

**アノテータ間の一致度（1番以降のGTを0番のGTと突き合わせた実測）:**

- 比較したGT個体数: `2990`
- **IoU > 0.5 で一致した割合: `53.9%`**（1611/2990）
- best IoUの平均 `0.438` / 中央値 `0.523`

つまり**別々の人間が同じ画像を注釈したとき、片方が引いた個体のうち公式のTP閾値を越えて一致するのは半分強しかない**。中央値0.523は0.5の崖のほぼ真上にあり、この課題の難しさの本質を示している。

**位置づけ:**

| | PQ | 天井0.7587比 |
|---|---:|---:|
| 自己ベスト（V35） | 0.37 | 49% |
| 自己ベスト（V30b/V32系、更新前） | 0.35 | 46% |
| LB 1位スナップショット | 0.48 | 63% |
| 天井（アノテータ0番のGTを提出） | 0.7587 | 100% |

**最重要の含意:** 天井側ですら**SQ 0.913に対してRQ 0.831と、RQが制約になっている**。人間同士でも「形」ではなく「どの個体が存在するか」で食い違う。したがって0.35の段階ではなおさら、境界を磨くよりも**個体の取りこぼしと過剰検出（RQ）を直す方が効く**可能性が高い。実測での確認は§2.1.4のSQ/RQ分解で行う。

### 2.1.6 全fold OOFのPQ実測【2026-08-15、ローカルPCで37分】

`scratch/run_pq_oof_rescore.py`。V30 YOLOの全fold OOF cache（707観測すべて）を再採点。学習・推論なし、GPU不要。

| conf | PQ | SQ | RQ | TP | FP | FN | near-miss | pred/枚 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.08 | 0.3525 | 0.6500 | 0.5423 | 5420 | 6370 | 2779 | 1595 | 10.24 |
| 0.12 | 0.3705 | 0.6501 | 0.5699 | 5296 | 5091 | 2903 | 1507 | 9.01 |
| 0.16 | 0.3818 | 0.6506 | 0.5868 | 5161 | 4231 | 3038 | 1426 | 8.17 |
| 0.20 | 0.3885 | 0.6511 | 0.5967 | 5017 | 3600 | 3182 | 1340 | 7.50 |
| **0.26** | **0.3937** | 0.6522 | 0.6036 | 4765 | 2824 | 3434 | 1209 | 6.60 |
| 0.35 | 0.3867 | 0.6546 | 0.5908 | 4270 | 1987 | 3929 | 991 | 5.43 |
| 0.45 | 0.3674 | 0.6580 | 0.5584 | 3655 | 1236 | 4544 | 722 | 4.24 |

fold別PQ（conf 0.26）: fold0 `0.3978` / fold1 `0.3882` / fold2 `0.3950`。**極めて安定している。**

#### 実測で否定された仮説

1. **「conf 0.26 は保守的すぎる」→ 否定。** PQは conf `0.26` で最大。上下どちらへ動かしても下がる。§2.1.2-5の損益分岐式（PQ 0.394 で `p > 0.302`）は正しく機能しており、実測の限界TP確率は 0.26→0.16 で `22%`（棄却が正しい）、0.35→0.26 で `37%`（採用が正しい）。**式が最適点を正確に当てている。**
2. **「duplicateの主因は複数アノテータを独立教師にしたこと」→ 否定。** 単一アノテータ観測と複数アノテータ観測で重複ペア数に差がない（0.49 vs 0.52、Mann-Whitney p=`0.736`）。しかも単一fold OOFの重複は1枚あたり約0.5ペアしかない。**V30の660,119重複画素は3-fold ensembleの合流で生じたものであり、検出器の学習ラベルに由来しない。** 重複問題はensemble統合段の課題として切り分ける。
3. **「fold間が不安定（P3）」→ 否定。** fold差は0.010以内。

#### 判明した主因

**SQ = 0.652 が天井 0.913 に対して大きく劣っている。** TPと判定された個体ですら、平均IoUが閾値0.5のすぐ上にとどまる。RQ 0.604 / 天井 0.831 と比率では同程度（71% vs 73%）なので、**RQ単独が律速ではない**。

決定的なのは **near-miss（IoU 0.3〜0.5）が conf 0.26 で `1209` 件**あること。PQの分母は「near-missをTPへ変える」操作に対して**不変**である（TP +1、FP −0.5、FN −0.5 で相殺）。したがって分子だけが増える。

```text
現状:   PQ = 3107.8 / 7894 = 0.3937
1209件すべてがIoU 0.55のTPになった場合:
        PQ = (3107.8 + 1209*0.55) / 7894 = 0.4779
```

**境界品質の改善だけで理論上 +0.084（0.394 → 0.478）** で、LB 1位 0.48 と同水準に届く。ただしnear-missのうち相手GTが既に他の予測とマッチしている分は変換不能なので、これは**上限**である。それでも最大の単一レバーであることは動かない。

### 2.1.8 refinerの方式・閾値をPQで選び直す【測定済み・当たり】

`scratch/run_refiner_pq_ablation.py`、conf 0.26、fold0 held-out 236観測、ローカルCPUで55分。
リーク管理: fold0を学習から除いた `mediseg_refiner_fold0_resume.pt` のみ使用（§18.2）。

| variant | PQ | SQ | RQ | near-miss | Δ vs YOLO native |
|---|---:|---:|---:|---:|---:|
| **shrink@0.65** | **0.4363** | 0.6757 | 0.6457 | 294 | **+0.0386** |
| shrink@0.75 | 0.4362 | 0.6768 | 0.6446 | 296 | +0.0386 |
| shrink@0.55 | 0.4355 | 0.6746 | 0.6456 | 296 | +0.0378 |
| shrink@0.45 | 0.4348 | 0.6731 | 0.6459 | 295 | +0.0371 |
| guided3@0.75 | 0.4326 | 0.6764 | 0.6396 | 306 | +0.0349 |
| guided7@0.65 | 0.4299 | 0.6736 | 0.6383 | 311 | +0.0322 |
| **guided7@0.45（V31が選んだ設定）** | 0.4242 | 0.6703 | 0.6329 | 324 | +0.0265 |
| YOLO native（refinerなし） | 0.3977 | 0.6568 | 0.6055 | 394 | — |

**判明したこと:**

1. **`shrink` が `guided` を全閾値で上回る。** 閾値の違い（shrinkで0.4348〜0.4363）より方式の違いの方が3倍効く。
2. **V31が選んだ `guided7@0.45` は精製版12種のうち最下位。** `shrink@0.65` へ変えるだけで **+0.0121**。**V31aのPublicが0.35→0.34と下がった事実と符合する。** V31は「guided7 0.552553 vs YOLO native 0.552290、差はほぼない」とDiceで結論したが、PQでは同じ比較が +0.0322 の差になる。指標の取り違えが実スコアの低下として現れていた。
3. **機序:** `shrink = refined & proposal`（YOLOマスクの内側しか削れない）に対し `guided = refined & dilate(proposal, k)`（外側へk画素まで足せる）。**足す方が悪い**ということは、YOLOの候補は既に過剰に覆っており、refinerの仕事は「削ること」であって「伸ばすこと」ではない。細構造を拾うために膨らませる直感は、IoUと0.5の崖の下では裏目に出る。
4. near-missは 394 → 294 へ100件変換された。**まだ294件残っている。** 同じ算術で、残り全部が IoU 0.55 のTPになれば PQ は 0.4363 → 約 `0.497`。これは refiner では届かず、検出器側の課題である。**ただし P2-2 の「学習解像度を 1536 に上げる」は fold0 で不合格だった（§2.1.8.4）。**

**留保（改訂3で解消）:** fold0のみ。fold1/2のrefiner checkpointが存在しないため全foldで確認できていなかった。→ §2.1.8.1 で解消。

### 2.1.8.1 fold1/2 refiner学習と全fold refinerアブレーション【確定・2026-08-16】

`solar_filament_v35_refiner_folds12_colab.ipynb`（Colab T4）でfold1/2のrefinerを学習した。V30のレシピを一切変更していない（crop 320 / context 1.65 / 14 epoch / LR 2e-4 / crop Diceでbest選択）。

| fold | best crop Dice | 学習時間 |
|---|---:|---:|
| fold0（既存） | 0.7933 | — |
| fold1（新規） | 0.7864 | 104.8分 |
| fold2（新規） | 0.7887 | 105.4分 |

**実測は「T4で30分・1CU未満」という当初見積もりから7倍外れた。** 文書§28の「refiner 1 epoch約1分前後」を根拠にしたが、実測は1 epoch約5分だった。今後の時間見積もりでは§28の数値を鵜呑みにしないこと。

3つのcheckpointが揃ったので、`scratch/run_refiner_pq_allfold.py` で**全fold OOF・リークなし**（各観測へ、その foldを学習から除いたcheckpointのみ適用）の方式比較を実施した。conf 0.26、707観測、ローカルCPUで87分。

| variant | PQ | fold0 | fold1 | fold2 | near-miss | Δ vs YOLO native |
|---|---:|---:|---:|---:|---:|---:|
| **shrink@0.45** | **0.4379** | 0.4348 | 0.4379 | 0.4410 | 934 | **+0.0442** |
| shrink@0.55 | 0.4375 | 0.4355 | 0.4371 | 0.4400 | 947 | +0.0438 |
| shrink@0.65 | 0.4371 | 0.4363 | 0.4351 | 0.4400 | 952 | +0.0434 |
| guided3@0.65 | 0.4357 | 0.4318 | 0.4350 | 0.4401 | 959 | +0.0420 |
| guided7@0.45（V31の選択） | 0.4326 | 0.4242 | 0.4355 | 0.4382 | 963 | +0.0389 |
| YOLO native | 0.3937 | 0.3978 | 0.3882 | 0.3950 | 1209 | — |

**fold0限定の結論は方式の順位としては維持された**（shrinkがguidedに全閾値で優位、閾値は実質フラット）。ただし絶対値は訂正が必要。

**訂正:** §2.1.8で「V31の選択ミスがPublic 0.35→0.34の低下を説明する」と書いたが、これは**過剰な帰属だった**。fold0限定では `guided7@0.45` との差が+0.0121だったが、全foldでは**+0.0053に縮小**する。`guided7@0.45`が悪く見えたのはfold0に限った現象（fold0での順位はworst=0.4242）で、他foldでは大きな問題ではなかった。**単一fold のbestを一般化してはいけない**という、この文書が§13.2で繰り返し警告してきたことを、改訂2執筆時の自分自身が踏み抜いていた。

### 2.1.8.2 パイプライン重畳検証【確定・2026-08-16】

`scratch/run_pipeline_stack.py`。refiner・排他化・confを単独最適化した値をそのまま組み合わせてよい保証はない。

- ownership の `+0.0069`（§2.1.7）は**生YOLOマスク**で測った値。refine後はマスクが縮んで重なりが減るため、同じ利得が残るとは限らない
- conf `0.26`（§2.1.6）も**refinerなし**で測った値。refinerはFPを減らすため、最適な動作点が動く可能性がある

そこでrefiner `shrink@0.45`（§2.1.8.1で確定）を固定し、conf `{0.16, 0.20, 0.26, 0.32}` × 排他化規則 `{conf_first, area_asc, drop_contained_then_area_asc}` の全12通りを、全fold OOF・リークなし・707観測で同時に測った（ローカルCPU、約150分。Google Driveの瞬断で2回中断したため、25観測ごとのチェックポイント機構を追加して再開した）。

| config | PQ | fold0 | fold1 | fold2 | Δ vs baseline |
|---|---:|---:|---:|---:|---:|
| **conf0.26 \| drop_contained_then_area_asc** | **0.4426** | 0.4393 | 0.4415 | 0.4471 | **+0.0489** |
| conf0.26 \| area_asc | 0.4402 | 0.4355 | 0.4382 | 0.4468 | +0.0465 |
| conf0.20 \| drop_contained_then_area_asc | 0.4383 | 0.4355 | 0.4383 | 0.4412 | +0.0446 |
| conf0.26 \| conf_first | 0.4379 | 0.4348 | 0.4379 | 0.4410 | +0.0442 |
| conf0.32 \| drop_contained_then_area_asc | 0.4371 | 0.4338 | 0.4362 | 0.4413 | +0.0434 |
| conf0.16 \| conf_first | 0.4261 | 0.4230 | 0.4275 | 0.4279 | +0.0324 |
| BASELINE（生YOLO + conf0.26 + conf_first、V30b相当） | 0.3937 | 0.3977 | 0.3882 | 0.3950 | — |

（全12構成は `output/v35/pipeline_stack.csv` に保存。上位・両端のみ抜粋）

**確定したこと:**

1. **`conf 0.26` は refiner適用後も最適点のまま変わらなかった。** 「refinerがFPを減らすので最適confが下がるはず」という予想は**外れた**。conf 0.20/0.16 は refine 後も一貫して conf 0.26 に劣る
2. **`drop_contained_then_area_asc` が全conf水準で `conf_first`／`area_asc` を上回った。** ownershipの改善は refine 後も**消えずに残る**（10観測の試走で見えた「refine後は逆効果」という兆候は、707観測全件では**再現しなかった**）。refine後の上乗せは `+0.0047`（0.4379→0.4426）で、生YOLOでの上乗せ`+0.0069`よりやや小さいが、方向は変わらない
3. **単独最適化の合計値と、組み合わせ最適化の値はほぼ一致した。** refiner単独 `+0.0442` + ownership単独（生YOLO） `+0.0069` ≈ `+0.0511`。実測の組み合わせ利得は `+0.0489`で、差は`0.0022`と小さい。この課題ではP2-1とP2-3はほぼ独立に効いている

**新ゲート:** all-fold PQ > `0.4526`（best `0.4426` + 0.010）。旧ゲート（§13.2の`0.4037`）を置き換える。

### 2.1.8.3 V35最終提出【Public 0.37、2026-08-16】

確定構成（`conf 0.26 + shrink@0.45 + drop_contained_then_area_asc`）で test 180枚を推論し、Kaggleへ提出した。

| 項目 | 値 |
|---|---|
| 提出ファイル | `submission_v35_final_pq_aligned.csv` |
| SHA-256 | `10cd5f79000861b0c2fcdc103ad085efeb6f44589590bbc2c353b24c01402162` |
| 行数 | 1301（178画像。2枚は予測ゼロ） |
| overlap | 0 |
| 面積 min / median / max | 186 / 1452 / 22242 |
| 全fold OOF PQ | 0.4426 |
| **Public Score** | **0.37** |

予測ゼロの2枚（`20200310113230Ch`、`20211214174010Mh`）は V30 / V32 でも同じで、今回の後処理で新たに落としたものではない。行数が V30b 1468・V32b 1406 より少ないのは `shrink` と包含除去の想定どおり。

**読み方:** OOF 0.4426 と Public 0.37 の差は残る。3-fold 統合は OOF で検証できない層であり、test 分布とのずれもあり得る。ただし方向は一致し、Public も 0.35 → 0.37 と上がった。後処理だけでは頭打ち、という §5.8〜5.9 の教訓は維持される。

### 2.1.8.4 P2-2 YOLO imgsz 1536 再学習【不合格、2026-08-16】

仮説: mask prototype を 320²→384² にすると near-miss が TP になる。
変えたもの: 学習 `imgsz` のみ（VRAM のため batch 4→2）。後処理は V35 確定構成のまま。
V30 の `runs/yolo_fold_*` は上書きしていない。

| | V35 fold0（imgsz 1280 学習） | P2-2 fold0（imgsz 1536 学習） |
|---|---:|---:|
| 確定後処理つき PQ | **0.4393** | 0.4295 |
| SQ | 0.673〜0.678 | 0.6747 |
| RQ | 0.646〜0.653 | **0.6365** |
| near-miss | 295（shrink のみ） | **329** |
| inst/img | 6.42〜6.72 | 6.60 |

ゲート（fold0 PQ > 0.4393）を **−0.0098** で外したので fold1/2 には進まない。
Ultralytics の Mask mAP50-95 は V30 fold0 より良かった（0.238 vs 0.227）が、公式 PQ は悪化した。mAP は採用判定に使わない。

**切り分け（refiner なし、同一 conf 0.26、`output/v36/p2_2_raw_yolo_fold0.json`）:**

| 検出器 | 排他化 | PQ | SQ | RQ | TP | FP | near-miss |
|---|---|---:|---:|---:|---:|---:|---:|
| 1280 V30 | conf_first | 0.3977 | 0.6568 | 0.6055 | 1614 | 968 | 394 |
| 1536 再学習 | conf_first | **0.3996** | **0.6667** | 0.5993 | **1673** | 1161 | 403 |
| 1280 V30 | drop_contained | 0.4049 | 0.6570 | 0.6164 | 1613 | 872 | 368 |
| 1536 再学習 | drop_contained | **0.4076** | **0.6661** | 0.6119 | 1620 | 926 | 388 |

生YOLOでは 1536 の方がわずかに良い（+0.002）。SQ は上がり、TP も増えるが FP も増える。
確定後処理を掛けると **refiner の利得が 1280 の +0.034 から 1536 の +0.022 へ縮小**し、総合で負ける。
失敗の主座は「検出器が壊れた」ではなく **1280 向け shrink@0.45 が 1536 提案と噛み合わない**こと。
後処理を 1536 の OOF で再チューニングして採用判定には使わない（単一fold の選択バイアス）。

**提出正本は V35 のまま。** 1536 再学習重みは提出に使わない。

成果物: `output/v36/p2_2_imgsz1536_fold0.json`、`output/v36/p2_2_raw_yolo_fold0.json`、Drive `solar_filament_v35/runs/yolo_imgsz1536_fold_0`、OOF npz 236枚。

### 2.1.8.5 P2-3 推論 imgsz 1536（V30 重みのまま）【不合格、2026-08-16】

P2-2 の交絡を外すため、V30 fold0 重み（imgsz 1280 学習）を学習せず、predict の `imgsz` だけ 1536 にした。ローカル CPU、236 観測。後処理は V35 確定構成。

| | 1280 学習・1280 推論（V35） | 1280 学習・1536 推論（P2-3） | 1536 学習・1536 推論（P2-2） |
|---|---:|---:|---:|
| 後処理つき PQ | **0.4393** | 0.4320 | 0.4295 |
| SQ | 0.673〜0.678 | 0.6742 | 0.6747 |
| RQ | 0.646〜0.653 | 0.6409 | 0.6365 |
| near-miss | 295 | 306 | 329 |
| inst/img | 6.42〜6.72 | 6.92 | 6.60 |

ゲート（fold0 PQ > 0.4393）を **−0.0073** で外した。fold1/2 の 1536 推論には進まない。

学習解像度を上げても、推論解像度だけ上げても、確定後処理の下では V35 を超えない。**解像度レバー（P2 の prototype 仮説）はここで止める。**

**切り分け（refiner なし、`output/v36/p2_3_raw_yolo_fold0.json`）:**

| 検出器 | 排他化 | PQ |
|---|---|---:|
| V30 1280 推論 | conf_first | 0.3977 |
| P2-3 1536 推論 | conf_first | **0.4026** |
| V30 1280 推論 | drop_contained | 0.4049 |
| P2-3 1536 推論 | drop_contained | **0.4121** |

生YOLOでは推論 1536 が勝つ（+0.005〜0.007）。確定後処理つきでは 0.4320 < 0.4393 で負ける。P2-2 と同じく **検出器破壊ではなく、1280 向け shrink との相性**が主因。

成果物: `output/v36/p2_3_infer_imgsz1536_fold0.json`、`output/v36/p2_3_raw_yolo_fold0.json`、`scratch/run_p2_3_infer_local.py`。

### 2.1.9 ローカル実装

- `src/metrics_official.py`: `PanopticQuality`累積器（公式実装に準拠）、`summary()`でPQ=SQ×RQ分解、`marginal_prediction_threshold()`で損益分岐、副次指標としてMIoU
- `tests/test_metrics_official.py`: 19件。うち1件は**公式コードの逐語移植との差分テスト**（乱数200ケース）、1件はIoU 0.50ちょうどがTPにならずFP+FN両方を生むことの固定、1件はアノテータのプーリングがvariant平均でないことの固定
- `scratch/build_v35_metric_realignment.py`: 上記ノートブックのビルダー
- `scratch/build_v35_refiner_folds12.py`: fold1/2 refiner学習ノートブックのビルダー（Colab T4、V30と完全同一レシピ）
- `scratch/run_refiner_pq_allfold.py`: refiner方式×閾値の全fold OOF比較（リークなし、fold別checkpointを適用）
- `scratch/run_pipeline_stack.py`: refiner×排他化×confの重畳検証。Google Driveのnpz/画像読み込みにリトライ（最大4回）、25観測ごとのアトミックチェックポイント保存・自動再開機構を持つ（`--fresh`で強制やり直し）
- `scratch/build_v35_final_submission.py`: 最終提出ノートブックのビルダー。V30本体のfold間統合（IoUクラスタリング＋多数決投票、OOF検証不能なため変更なし）はそのまま流用し、refiner適用と排他化のみOOF確定実装に差し替えている。`src/mediseg_models.py` をノートへ埋め込む（Colabへ別ファイルを上げる必要はない）
- `scratch/validate_v35_submission.py`: 提出CSVの全件検証と過去提出との件数比較
- `scratch/build_v36_yolo_imgsz_probe.py`: P2-2 探針。YOLO `imgsz 1536` を **fold0のみ** 再学習。**不合格**（fold0 PQ 0.4295 < 0.4393）。V30 の `runs/yolo_fold_*` は上書きしていない
- `scratch/run_p2_imgsz_raw_pq.py`: P2-2 の 1536 OOF と V30 の 1280 OOF を refiner なしで同じ規則で採点し、検出器本体の差を切り分ける
- `scratch/build_v36_infer_imgsz_probe.py`: P2-3 探針。V30 fold0 重みを学習せず、推論 imgsz だけ 1536。**不合格**（fold0 PQ 0.4320 < 0.4393）
- `scratch/run_p2_3_infer_local.py`: P2-3 をローカル CPU で実行。日本語 Drive パスを YOLO に渡さない

## 3. データと課題の理解

### 3.1 確認できたデータ規模

V28のKaggleログおよびV30前処理ログから以下を確認した。

- 学習・検証観測画像: `707`（566 train + 141 validとして一度分割）
- 3-foldの観測画像数: `236 / 236 / 235`
- test画像: `180`
- 複数アノテータを持つ画像: `296`
- V30で利用した全アノテータ個体数: foldごとに`1866 / 1844 / 1887`、合計`5597`
- 前処理対象: train 707 + test 180 = `887`
- 画像上の前景率は非常に小さい。ログ例では`0.0023384`。

同一観測画像に複数アノテータ／annotation variantがあるため、単純なannotation行ランダム分割はリークを起こす。V30以降は観測画像単位のgrouped foldを基本とした。

### 3.2 コンペが要求する対象

この課題は単一の前景領域ではなく、**個々の太陽フィラメントinstance**を提出する。したがって以下が重要になる。

- 見逃した個体
- 1個体を複数maskに分断するsplit
- 複数個体を1maskに結合するmerge
- 同一個体を複数候補として重複提出するduplicate
- 細い枝やspine/barbの欠落

semantic union Diceが高くても、instanceの個数・対応・分離が悪ければPublic Scoreは低くなる。

## 4. 提出CSVの確定仕様

過去のエラーと成功提出から、少なくとも以下を満たす必要がある。

### 4.1 列とID

```text
filament_id,segmentation_rle
20110120105534Ch_1,...
20110120105534Ch_2,...
```

- 列名は正確に`filament_id`と`segmentation_rle`。
- 旧案の`rle_mask`は誤り。
- `filament_id`は画像stemに`_1`, `_2`, ...を付けたinstance ID。
- COCO JSON内のimage IDには`040301-20140609195854Bh`のような文字列がある。`int(item["id"])`へ変換してはいけない。

### 4.2 RLE

- `segmentation_rle`は公式のcompressed COCO RLE文字列。
- 空maskを空文字や`NaN`として行に残さない。空maskの行自体を出力しない。
- `keep_default_na=False`で再読込し、nullがないことを検証する。
- 全RLEをdecodeし、元の`2048 x 2048`へround-tripできることを確認する。

### 4.3 instance間重複

Kaggleから次のエラーを確認した。

> Invalid Submission! Submissions may not contain overlapping masks.

同一画像内の提出instance maskは1画素も重複してはいけない。提出前に各画像のoccupancyを作り、`occupancy > 1`が0画素であることを全件検証する。

### 4.4 提出前の必須検査

- 列順が正しい
- 行数が1以上
- nullがない
- 空RLEがない
- `filament_id`が一意
- 各RLEがdecode可能
- 面積が正
- 同一画像内のinstance重複が0画素

## 5. バージョン別の経緯

### 5.1 V18以前: 提出形式エラー

確認された代表エラー:

```text
Submission contains null values
```

初期CSVには`filament_id,rle_mask`形式や空mask由来のnullがあり、Kaggleが受理しなかった。ここではモデル精度を評価できていない。

### 5.2 V19～V21: オフライン提出形式の修正

- V19: 空maskを除外し、公式列名とcompressed COCO RLEを採用。2048角でround-trip検証。
- V20: Internetなし、`pycocotools`なしでもcompressed RLEを書けるようにした。
- V21: `best_solar_unet.pth`で使われた`FallbackUNet2D`を再現する方向へ修正。
- ノートブックがKaggleへアップロードされていることと、実行／提出が成功することは別である。ブラウザ画面上のDraft savedやVersion作成だけでは、モデル・入力・提出受理を保証しない。

### 5.3 V22～V25: Kaggle入力ZIPとcheckpoint復元

- Kaggle Inputのネストした配置を探索。
- `best_solar_unet.pth`またはZIPが未添付のときは`FileNotFoundError`になった。
- ZIPをKaggle Datasetとして添付後、EfficientNet-B3 + SMP 0.3.3 Unet++相当をオフライン再構築。
- V25ログでは180画像を推論し、179行のCSVを生成、RLE検証に成功。
- 形式上は成功したが、Kaggle PublicはV24で`0.00`。

**推定原因:** checkpoint／学習内容が実コンペのinstance segmentationに合っておらず、ほぼ1画像1maskのsemantic出力だった。形式が正しくても、instance表現が根本的に不足していた。

### 5.4 V26: 実データYOLO + U-Net refinerへの転換

- synthetic checkpointを捨て、competitionの実COCO annotationからinstance学習する設計。
- 公開Apache-2.0ノートブック「YOLO + U-Net」を参照。
- Kaggle側でInternet OFFのまま`ultralytics`をpip installしようとしてDNS失敗し、`ModuleNotFoundError`になったログがある。

**運用上の結論:** Ultralyticsや事前学習weightを初回取得する実行ではInternet ONが必要。Internet OFFがコンペ規則上必須かどうかは、ルール本文を別途確認する必要がある。CSVファイル提出方式であることと、NotebookのInternet制限は分けて考える。

### 5.5 V27～V28: 実データMediSeg semanticモデル

V27/V28はYOLOを外し、リポジトリの31,036,481パラメータ`MediSegUNet`を実MAGFiLO maskで学習した。

V28ログ:

- best validation Dice@0.5: `0.630328`
- threshold探索のbest: `0.35`、union Dice `0.630781`
- 提出行数: `1988`
- 予測あり画像: `178 / 180`
- 1画像当たりinstance中央値: `11`
- RLE面積: min `20`, median `987`, max `20585`
- 公式RLE検証: passed

一方、Kaggleで確認したMediSeg系のPublicはV27で`0.19`。

**確定した教訓:** semantic union Dice約0.63は、instance scoreの高さを意味しない。連結成分による個体化では分断・結合・個数誤差を解決できない。

### 5.6 V29: YOLO検出 + MediSeg crop refiner

- YOLOを個体候補検出に戻す。
- 各YOLO候補cropを31MパラメータMediSegUNetで精密化。
- Internet ONが必要。
- V28のsemantic-only問題を修正する設計だった。

**未確認:** V29単独のPublic Scoreを示す確実な画面／ログは残っていない。

### 5.7 V30: 3-fold YOLO11m-seg + MediSeg refiner

主要設計:

- 観測画像単位の3-fold grouped split
- YOLO11m-seg、`1280px`、最大`70 epoch`、patience `14`
- 複数アノテータを独立教師として利用
- MediSeg crop refiner、14 epoch + 全データ5 epoch
- アノテータvariant平均のHungarian instance Diceで後処理を選択
- 3-fold ensemble

学習結果の一部:

- refiner crop Diceは最終的に`0.799007`
- 初期held-out探索best: conf `0.26`、refiner threshold `0.60`、min area `28`
- 当時のinstance Dice: `0.513442`
- test推論: 180画像、元CSV`1565`行、平均約`8.7`個体／画像

最初の`submission_v30_prize_instance_ensemble.csv`はinstance mask同士が重なり、Kaggleに拒否された。

V30b修復:

- 修復前: `1565`行
- 修復後: `1468`行
- 除去した重複画素: `660119`
- 空または極小化して除外したmask: `97`
- 修復後重複: `0`
- Kaggle Public Score: **`0.35`**

**重要:** V30bは現在確認できる最高Publicだが、重複画素を先着mask優先で削ったため、境界形状を大きく破壊した可能性がある。

### 5.8 V31: リークなしheld-out後処理アブレーション

V30の重みを固定し、fold 0 held-outで以下を比較した。

- YOLO native mask
- MediSegで内側だけ削る`shrink`
- YOLO境界から3px/7pxのみ拡張する`guided`
- confidence、refiner threshold、minimum area

fold 0のbest:

- method: `guided7`
- threshold: `0.45`
- conf: `0.26`
- min area: `20`
- held-out strict instance Dice: `0.552553`
- mean absolute count error: `1.853107`

YOLO nativeとの差は非常に小さかった。

- guided7 best: `0.552553`
- YOLO native best: `0.552290`

生成物:

- V31a best ensemble: 1569行、178画像、Public `0.34`
- V31b fold0 only: 1258行（Public未確認）
- V31c YOLO native ensemble: 1769行（Public未確認）
- 全候補で重複0を検証

**教訓:** held-outで小さく改善してもPublicはV30bの0.35を超えなかった。後処理の局所改善はtest分布で安定しない。

### 5.9 V32: 重複候補のクラスタ統合

V30の重複319組を調べると、小さいmaskの約99%が大きいmaskに内包されるケースが多かった。同一個体の重複候補と考え、削る前にunionする方式を試した。

ローカルmanifestで確認済み:

| 候補 | 条件 | 行数 | merged groups | 除去cross-cluster画素 | dropped masks | overlap |
|---|---|---:|---:|---:|---:|---:|
| V32a | Dice >= 0.90 | 1453 | 28 | 558430 | 83 | 0 |
| V32b | containment >= 0.99 かつ Dice >= 0.50 | 1406 | 109 | 434497 | 46 | 0 |
| V32c | Dice >= 0.70 | 1398 | 100 | 329352 | 63 | 0 |

Kaggle画面で確認:

- V32a Public `0.35`
- V32b Public `0.35`
- V32c Publicは未確認

**教訓:** overlap修復の方法を改善してもPublicは0.35で頭打ち。重複処理は必要条件だが、主要な精度ボトルネックではない。

### 5.10 V33: 「公開0.69」ノートブックの再現実験

参照元が申告した構成へ戻した。

- raw grayscale
- YOLO11s-seg
- `1024px`
- `30 epoch`
- confidence `0.25`
- ImageNet pretrained ResNet18 U-Net refiner
- refiner 8 epoch、threshold `0.65`
- 文字列image IDを保持
- overlapがある画像だけconfidence順に排他化

実行結果:

- refiner best crop Dice@0.65: `0.812370`
- held-out instance Dice@512: `0.443369`
- held-out GT平均個体数: `7.831`
- held-out prediction平均個体数: `6.338`
- 重複発生画像: `63 / 142`
- test YOLO平均個体数: `7.05`
- test提出平均個体数: `6.90`
- 提出行数: `1242`
- 予測あり画像: `176 / 180`
- overlap: `0`
- Kaggle Public Score: **`0.33`**

**確定したこと:** この環境・データ・実装で「0.69」を再現できなかった。

**未解決:** 参照元0.69との差が、コード差、データ版、fold／学習seed、提出後処理、旧評価仕様、記載ミスのどれによるものかは特定できていない。単にモデル名や閾値を合わせるだけでは再現にならない。

### 5.11 V34: 高解像度semantic-first + YOLO seed

V34は、YOLO maskを最終maskにせず、semantic foregroundを先に推定してYOLOをwatershed seedとして使う逆転設計。

主要条件:

- 3-fold
- EfficientNet-B4 U-Net（失敗時ResNet34）
- train patch `768`
- inference tile `1024`
- 最大`32 epoch`、patience `7`
- BCE + Dice + clDice + boundary loss
- 背景勾配補正、CLAHE、dark-detail入力
- V30 YOLO 3-foldをseedとして再利用
- V31 strict Dice `0.552553`を`0.007`以上改善し、count errorも改善した場合のみtest推論

#### 学習結果

- fold 0 best validation union Dice@768: `0.366947`（epoch 14）、epoch 21でearly stop
- fold 1 best: `0.352138`（epoch 5）、epoch 12でearly stop
- fold 2 best: `0.412355`（epoch 1）、epoch 8でearly stop
- 3-foldのbest weight、完了marker、semantic/YOLO OOF cacheはGoogle Driveへ保存

#### 実行上の問題

- `tqdm(..., leave=False)`とColab実行履歴が古い`0/707`表示を残し、実行中セルの判別が難しかった。
- OOF全候補評価が`524/707`で長時間停止。
- 原因候補は、semantic foregroundから非常に多い小領域／markerが生じ、watershed後のmask生成と全mask間比較が極端に重くなったこと。
- 通常のinterruptがネイティブ処理中に効かず、session restartが必要になった。
- 救済評価ではinstance候補を最大32に制限し、3候補を各707画像で再評価した。
- 元の`[10/13]`を飛ばしたため、可視化セルで`fold0_names`が未定義になったが、ゲート結果には影響しなかった。

#### 最終OOF結果

| threshold | seed conf | min area | strict instance Dice | count error | splits | merges |
|---:|---:|---:|---:|---:|---:|---:|
| 0.52 | 0.22 | 60 | **0.441074** | **2.846535** | 0.361622 | 0.270156 |
| 0.46 | 0.22 | 60 | 0.433612 | 3.037011 | 0.373173 | 0.274635 |
| 0.46 | 0.22 | 20 | 0.432438 | 3.059642 | 0.373173 | 0.274635 |

ゲート比較:

- V31 strict Dice: `0.552553`
- 必要Dice: `0.559553`以上
- V34: `0.441074` → 不合格
- V31 count error: `1.853107`
- V34: `2.846535` → 不合格
- test推論許可: `false`
- V34提出CSV: 生成していない

**結論:** semantic-first方式は、今回の学習条件では個体検出数とinstance対応を悪化させた。微調整で埋まる差ではなく、この方向をそのまま提出へ進めるべきではない。

## 6. Kaggle Public Score履歴

画面で確認できたものだけを記載する。

| 提出 | 状態 | Public Score | 備考 |
|---|---|---:|---|
| V18 | Error | - | null values |
| V24 | Complete | 0.00 | 形式修正後だがモデル不適合 |
| V27 | Complete | 0.19 | semantic connected-components系 |
| V30 original | Error | - | instance mask overlap |
| V30b disjoint | Complete | 0.35 | 当時の自己最高 |
| V31a | Complete | 0.34 | held-out best ensemble |
| V32a | Complete | 0.35 | ultra-conservative union |
| V32b | Complete | 0.35 | containment-based union |
| V33 | Complete | 0.33 | 申告0.69構成の再現実験 |
| V34 | Not submitted | - | OOFゲート不合格 |
| V35 PQ-aligned | Complete | **0.37** | shrink@0.45 + drop_contained_then_area_asc。自己ベスト |

共有されたリーダーボード画面では当時の1位が`0.48`、2位`0.40`だった。これは画面取得時点のスナップショットであり、現在値としては未検証。

## 7. 指標の比較と注意

| バージョン | 指標 | 値 | 評価しているもの |
|---|---|---:|---|
| V28 | semantic union Dice | 0.630781 | 全前景の重なり。個体分離を含まない |
| V30 | refiner crop Dice | 0.799007 | 検出済みcropの境界。見逃しを含まない |
| V30 | 初期held-out instance Dice | 0.513442 | 個体対応を含むが単一fold中心 |
| V31 | held-out strict instance Dice | 0.552553 | 排他化後のfold 0個体評価 |
| V33 | held-out instance Dice@512 | 0.443369 | 公開構成再現の個体評価 |
| V34 | all-fold strict instance Dice@512 | 0.441074 | 全3-fold OOF個体評価 |

0.63や0.80をPublic Scoreと比較してはいけない。

> **【改訂2】この節の元の結論「最も信頼できるローカル比較軸は、全fold OOFのstrict instance Diceとcount errorの組である」は誤りだった。** strict instance Diceは公式指標ではない。正しい比較軸は Panoptic Quality と、その SQ / RQ 分解である。§2.1 を参照。
>
> なお「V31はfold 0中心、V34は全foldであり、完全に同条件ではない」という注記自体は正しく、かつ重要だった。にもかかわらず §13.2 ではそのfold 0の値をそのまま全fold結果への合格線に使っており、文書内で矛盾していた。実測ではfold 0 best → 全foldで約 -0.03 の系統差がある（§8.1-6）。**選択バイアスのかかった単一foldのbestを、全fold値の基準にしてはいけない。**

## 8. 分かったこと

### 8.1 モデル／評価

> **【改訂2】この節は「Hungarian matched Diceが評価軸である」という前提で書かれている。** 項目3・7の扱いが変わり、項目2の意味が強まった。差分は§2.1.3の表を参照。方向性としての1・4・5・6・8は改訂後も有効。

1. **instance-firstが必要。** semantic unionから連結成分を作るだけでは不足。
2. **crop refiner単体の高Diceは全体性能を保証しない。** detectorが見逃した個体はrefinerで復元できない。
3. **個体数誤差が重要。** V34はDiceだけでなくcount errorも悪化した。
4. **split/mergeを明示的に測る必要がある。** pixel Diceだけでは同一前景量でも個体対応を誤る。
5. **後処理だけの改善幅は小さい。** V31/V32は局所検証を改善してもPublic 0.35を超えなかった。
6. **全3-fold OOFが必要。** fold 0のbestが他foldへ一般化しない。V34ではfold 0約0.47の候補が全foldで約0.44へ低下した。
7. **排他化は提出の必須条件。** ただし単純な先着画素削除は形状を壊す。
8. **Publicノートブックの申告値は再現検証が必要。** V33は0.69を再現せず0.33だった。

### 8.2 実装／運用

1. COCO image IDを数値と仮定しない。
2. ノートブックの途中セルだけを再実行すると、`BATCH_CANDIDATES`、`CFG`、`fold0_names`などの`NameError`が起こる。
3. checkpoint、完了marker、OOF cache、途中評価CSVをDriveへ保存すべき。
4. 長い探索は候補ごとにpartial CSVを保存し、進捗に候補番号とfile名を表示する。
5. connected-components、watershed、mask間総当たりには個体数上限と異常画像診断が必要。
6. Colabの右側「実行数」は履歴であり、現在の実行セルを示すとは限らない。セル左が停止四角なら実行中、三角なら非実行。
7. PCを閉じてもColab Pro+が必ず動き続ける保証はない。ブラウザ切断、アイドル、backend recycleがあるためDrive checkpointが前提。

## 9. 分からなかったこと／未解決事項

### 9.1 精度・コンペ

1. ~~参照公開ノートブックのPublic `0.69`を再現できなかった正確な理由。~~ **【改訂2で解決・調査打ち切り】** §6のLB 1位が`0.48`である以上、`0.69`はこのリーダーボードの値ではありえない。V33の`0.33`が正しい実測。**これ以上時間を使わない。**
2. 当時1位`0.48`の手法。公開情報からは推定できない。**なお2026-08-12にLBが再採点されているので、スクリーンショットがそれ以前なら`0.48`という数値自体が無効。取得時刻が`unknown`。**
3. ~~Kaggle公式metricとローカルHungarian strict instance Diceの完全一致性。~~ **【改訂2で解決】一致していなかった。** 公式はPanoptic Quality。§2.1参照。
4. Public testとprivate testの分布差、およびPublic 0.35がprivateでも維持されるか。**Public LBはtestの約50%と明示されている。**
5. V29、V31b、V31c、V32cの未確認Public Score。
6. ~~複数アノテータを独立sample、soft consensus、variant平均のどれで扱うのが最適か。~~ **【改訂2で方向確定】** PQは各アノテータへ個別採点して合算するので、両アノテータでIoU 0.5を越えられる予測はTPが2個入る。consensus的な予測が有利。ただし最適な作り方（pixel voting / STAPLE / 中央値アノテータ採用）は未確認。
7. 細い枝を保つこととinstance分離を両立する最適なloss／解像度。**PQへは間接的（IoU経由）だが、定性評価30%の「予測マスクの見た目の妥当性」には直接効く（§19.0）。**
8. 外部MAGFiLOデータを追加利用できるか。**Evaluationタブに「推論には`test`ディレクトリのH-alpha画像のみを使用」「学習に他のground-truthメタデータを使ってはならない」と明記されている。外部データの利用自体は許容されるが、この2条件を満たすこと。**
9. **【改訂2で追加】アノテータ間の不一致が課すPQの天井。** あるアノテータのGTをそのまま提出しても満点にならない。GTのみで計算可能。未測定（§2.1.5）。

### 9.2 運用・規則

1. このコンペでKaggle NotebookのInternet OFFが必須かどうか。CSV提出画面は確認したが、ルール本文を独立検証していない。
2. failed submissionが日次提出枠を消費するか。提出ダイアログで残数とリセット時刻は確認できるが、消費規則は未確認。
3. ColabのA100/T4割当とCU消費率は時期・契約・需給で変動する。

## 10. Colab運用で得た実務知識

### 10.1 GPUとHigh-RAM

観測例:

- T4: 約`1.07 CU/hour`、GPU RAM約15GB、通常RAM約12.7GB
- A100: 画面上約`12.07 CU/hour`、GPU RAM40GB

V34のT4ではbatch 2 + gradient accumulation 2で1 epoch約7分半。batch 4時の約9分より速い場面があり、T4はコスト効率が高かった。

High-RAMについて:

- V31～V34は通常RAMで動作可能。
- V30の高速キャッシュ設定ではRAM使用が17GB前後まで増えた例があり、High-RAMは高速化に有効だが必須ではない。コードは低RAM時にworker/cacheを落とす設計。
- GPU RAMとシステムRAMを混同しない。

### 10.2 Internet

- Ultralytics、Hugging Face encoder、ImageNet weightの初回取得はInternet ONが必要。
- DriveやKaggle Datasetへ依存物を保存すれば、再実行はオフライン化できる。
- KaggleでInternet OFFのまま`pip install ultralytics`を行うと失敗した。
- HF Hubのunauthenticated warningはダウンロード制限の警告であり、モデル取得自体が完了すれば学習エラーではない。

### 10.3 再起動と再開

- runtime restartでローカル変数は消える。
- Drive上のweight、checkpoint、marker、cacheは残る。
- 「すべてのセルを実行」で安全に再開できるには、各セルが前段の変数だけでなく保存済みartifactから復元できる必要がある。
- 長いネイティブ処理は通常interruptが効かないことがある。その場合はsession restart。
- restart後に元の重い探索セルをそのまま実行せず、定義セルと救済評価セルを分ける。

## 11. 既知のエラーと対処

| エラー／症状 | 原因 | 対処 |
|---|---|---|
| Submission contains null values | 空mask、誤列、NaN | 空行を出さず、公式列、`keep_default_na=False`検査 |
| overlap masksでInvalid Submission | instance間画素重複 | confidence順排他化、全RLE再decode検証 |
| checkpoint not found | Kaggle Input未添付／ネスト | Dataset添付、再帰探索、ZIP復元 |
| Ultralytics not found | Internet OFFでpip失敗 | Internet ONまたはwheel/weightをDataset化 |
| invalid literal for int() | COCO IDが文字列 | IDを文字列のまま辞書keyにする |
| `BATCH_CANDIDATES` NameError | 途中セルだけ再実行 | 設定セルから順に実行、セル自己完結化 |
| `CFG` NameError | restart後の独立修復セル | Drive pathを独立定義する |
| `fold0_names` NameError | V34の元探索セルをskip | group定義から再構築。ゲート結果には影響なし |
| Colabが止まったように見える | stale tqdm、実行履歴表示 | セル左アイコン、タイマー、実数カウンタで判断 |
| `524/707`で実停止 | 異常な小領域と総当たりmask処理 | instance上限32、partial保存、session restart |

## 12. ローカル成果物一覧

### 12.1 主要ノートブック

- `solar_kaggle_ensemble_mediseg_v19_fixed_complete.ipynb`
- `solar_kaggle_ensemble_mediseg_v20_offline_fixed.ipynb`
- `solar_kaggle_ensemble_mediseg_v21_offline_fixed.ipynb`
- `solar_kaggle_ensemble_mediseg_v22_nested_input_fixed.ipynb`
- `solar_kaggle_ensemble_mediseg_v23_uploaded_zip_fixed.ipynb`
- `solar_kaggle_ensemble_mediseg_v24_uploaded_zip_validated.ipynb`
- `solar_kaggle_ensemble_mediseg_v25_uploaded_zip_validated.ipynb`
- `solar_kaggle_ensemble_mediseg_v26_realdata_yolo_refiner.ipynb`
- `solar_kaggle_ensemble_mediseg_v27_offline_realdata.ipynb`
- `solar_kaggle_ensemble_mediseg_v28_offline_realdata_stable.ipynb`
- `solar_kaggle_ensemble_mediseg_v29_yolo_mediseg_fusion.ipynb`
- `solar_filament_v30_prize_colab_instance_ensemble.ipynb`
- `solar_filament_v31_heldout_postprocess_ablation.ipynb`
- `solar_filament_v32_overlap_cluster_repair.ipynb`
- `solar_filament_v33_public069_reproduction_colab.ipynb`
- `solar_filament_v34_semantic_instance_hybrid_colab.ipynb`

### 12.2 ビルダー

`scratch/build_v*.py`に各ノートブックの生成元がある。直接ipynb JSONを手編集するより、ビルダーを修正して再生成する方が再現性が高い。

### 12.2.1 【改訂2で追加】公式指標まわりの成果物

| ファイル | 役割 |
|---|---|
| `src/metrics_official.py` | `PanopticQuality`累積器（公式実装準拠）、PQ=SQ×RQ分解、損益分岐計算、副次指標のMIoU |
| `tests/test_metrics_official.py` | 19件。公式コードの逐語移植との差分テスト（乱数200ケース）を含む |
| `scratch/run_annotator_ceiling.py` | アノテータ不一致が課すPQの天井。COCO annotationのみで完結、ローカルPCで数分 |
| `scratch/run_pq_oof_rescore.py` | 全fold OOFのPQ再採点＋conf掃引＋duplicate層別。**ローカルPCで約20分、GPU不要** |
| `scratch/build_v35_metric_realignment.py` | 同内容をColabで回す版のビルダー |
| `output/v35/` | 上記の出力（`annotator_ceiling.json`、`pq_conf_sweep.csv`、`pq_by_fold.csv`、`pq_gate.json`、`duplicate_stratification.csv`） |

**ローカル実行が可能な理由:** Google Driveが`G:\マイドライブ`にマウントされており、`solar_filament_v30`と`solar_filament_v34`へ直接アクセスできる。V30 YOLOのOOF cacheは707件すべて2048×2048のまま存在し、読み出しは1件0.016秒。Colabで同じ処理をするとDriveマウント経由の小ファイル大量読み出しが遅く、かつCUを消費する。**GPUを使わない診断はローカルPCで回す方が速くて安い。**

注意: システムドライブ`C:`の空きは約7GBしかない。大きな中間生成物を`C:`へ置かない。

### 12.3 V32のローカル提出候補

`output/v32/`にCSV、audit CSV、manifestがある。manifestにはSHA-256、行数、面積、重複0が記録されている。

### 12.4 Google Drive上の主要ディレクトリ

- V30: `/content/drive/MyDrive/solar_filament_v30`
- V33: `/content/drive/MyDrive/solar_filament_v33_public069`
- V34: `/content/drive/MyDrive/solar_filament_v34`

V34のDriveには少なくとも次が保存済み。

- `runs/semantic_fold_0/best.pt`
- `runs/semantic_fold_1/best.pt`
- `runs/semantic_fold_2/best.pt`
- fold完了marker
- semantic OOF cache
- YOLO OOF cache
- `v34_oof_calibration.csv`
- `v34_gate_report.json`

## 13. 次に進む場合の推奨V35

V34のsemantic-firstを微調整するより、V30のinstance-firstを再設計する。

### 13.1 推奨構成

1. YOLO instance segmentationを主モデルにする。
2. 3-foldすべてのOOF predictionを先に保存する。
3. refinerは新規個体を作らず、YOLO proposal内部または狭い境界帯だけを修正する。
4. 複数fold候補はmask voting／clusterで統合し、同一個体duplicateを先に統合する。
5. 最終排他化は、confidence先着ではなくpixel ownership scoreで割り当てる。
6. hard negative、低contrast、太陽縁、細長い個体を明示的にsampleする。
7. 検証指標をstrict instance Dice、count error、split、mergeの4つに固定する。
8. 後処理パラメータはfold 0で絞っても、採用判定は必ず全3-fold OOFで行う。
9. test推論は全OOFゲート通過後だけ許可する。

### 13.2 最低acceptance gate

> **【改訂2】以下のDiceベースのゲートは廃止した。** `0.552553` は (a) 公式でない Hungarian matched Dice の値であり、(b) fold 0 のみで探索したbestで選択バイアスがかかっている、という二重の理由で基準に使えない。差し替え後の基準は §2.1.4 を参照。
>
> **【改訂3】生YOLOベースのゲート（`0.403711`）も置き換えた。** refiner (`shrink@0.45`) と排他化 (`drop_contained_then_area_asc`) を組み込んだ構成で全fold OOFを測り直したところ（§2.1.8.2）、より高いベースラインが確定した。

**現行ゲート（PQ基準・実測済み、2026-08-16確定）:**

ベースラインは `output/v35/pipeline_stack.json`。V30 YOLO 3-fold OOF + refiner `shrink@0.45` + 排他化 `drop_contained_then_area_asc`、conf `0.26`、全707観測で **PQ `0.4426`**（fold別: fold0 `0.4393` / fold1 `0.4415` / fold2 `0.4471`）。

- all-fold **Panoptic Quality** > `0.4526`
- 最悪foldのPQが `0.4393` を下回らない
- overlap pixels = `0`
- null = `0`
- duplicate filament ID = `0`
- Public結果を見た後にtest側パラメータを再調整していない

**廃止した旧ゲート（記録として保持、判断には使わない）:**

- ~~all-fold strict instance Dice > `0.552553`、目標`0.56`以上~~
- ~~mean absolute count error < `1.853107`~~
- ~~all-fold Panoptic Quality > `0.403711`（生YOLO + conf先着排他化のみ、refiner未適用）~~

count errorは指標として捨てるのではなく、PQの分母に入るFN/FPの内訳として見る（§2.1.2-5）。

### 13.3 実験設計上の改善

- foldごとのOOF metricを別々に表示し、平均だけで隠さない。
- image単位のworst casesを保存する。
- GT count、prediction count、split、merge、false negative候補を画像別CSVへ出す。
- 1候補終了ごとにpartial CSVをDriveへ保存。
- progress barに候補番号、fold、file名を表示。
- pathological image向けにmax instances、max components、処理時間警告を入れる。
- 公開0.69再現を続けるなら、元ノートブックのcommit／出力CSV／データfingerprintまで揃えて差分検証する。

## 14. やらない方がよいこと

- semantic union Diceだけを根拠に提出する。
- crop Diceだけを根拠にモデル全体を評価する。
- overlapのあるCSVをそのまま提出する。
- 単一foldの後処理bestをそのままtestへ適用する。
- Public Scoreが同じ後処理候補を大量に提出して日次枠を消費する。
- restart後に途中セルだけを実行し、未定義変数を場当たり的に追加する。
- ColabローカルSSDだけにcheckpointを置く。
- 申告Public Scoreを、同一条件の再現なしに性能根拠として扱う。

## 15. 現在地点

- 形式問題は解決済み。
- overlap 0の提出生成・検証も解決済み。
- 実データ学習、3-fold、MediSeg refiner、YOLO ensemble、semantic-firstまで試行済み。
- 確認できるPublic最高はV35の`0.37`（V30b/V32は0.35）。
- V34はゲートで棄却され、提出していない。
- 次の改善余地は検出器側。P2-2（1536 再学習）も P2-3（推論 1536）も不合格。解像度レバーは停止。後処理の単独最適化は V35 で一通り使い切った。

## 16. 参照したローカル証拠

- `README.md`
- `scratch/build_v26_realdata_yolo_refiner.py`
- `scratch/build_v27_offline_mediseg.py`
- `scratch/build_v30_prize_colab.py`
- `scratch/build_v31_heldout_postprocess_ablation.py`
- `scratch/build_v32_overlap_cluster_notebook.py`
- `scratch/build_v33_public069_reproduction.py`
- `scratch/build_v34_semantic_instance_hybrid.py`
- `output/v32/v32_manifest.json`
- `C:\Users\hara\Downloads\solar-kaggle-ensemble-mediseg*.log`
- `C:\Users\hara\Downloads\ダウンロード (15).txt`～`ダウンロード (24).txt`のうちSolar Filament関連ログ
- この作業中に共有されたKaggle／Colab画面と実行出力

---

この文書の数値を更新するときは、Public Score、OOF instance Dice、semantic Dice、crop Diceを必ず別欄に記録し、未提出・未確認の値をPublic Scoreとして扱わないこと。

## 17. 追加引継ぎ事項

以下は再現性、依存関係、運用継続に関する追記事項である。

### 17.1 どの成果物を正本として扱うか

| 用途 | 正本候補 | 状態 | 注意 |
|---|---|---|---|
| 最高Publicを再現する学習基盤 | `solar_filament_v30_prize_colab_instance_ensemble.ipynb` | 基準 | 元CSVは重複あり。提出にはV30b/V32修復が必要 |
| V30重みの評価・後処理比較 | `solar_filament_v31_heldout_postprocess_ablation.ipynb` | 診断用 | fold 0中心で全fold OOFではない |
| overlap修復 | `solar_filament_v32_overlap_cluster_repair.ipynb` | 実用 | 学習不要。入力V30 CSVのprovenance確認が必要 |
| 公開0.69構成の再現記録 | `solar_filament_v33_public069_reproduction_colab.ipynb` | negative result | 実Public 0.33。性能基盤にはしない |
| semantic-first研究 | `solar_filament_v34_semantic_instance_hybrid_colab.ipynb` | negative prototype | ゲート不合格。ローカル版に救済修正未反映 |
| V19～V29 | 各version notebook | archive | 問題の変遷確認用。新規提出の起点にしない |

V35はV30のinstance-first構成を土台にし、V31の評価関数とV32の重複cluster処理を取り込む。V34を直接改良する形にはしない。

### 17.2 ローカルV34とColab実行結果の差

Colabで最終OOF評価を完了させた救済セルでは、`semantic_to_instances`の最大instance数を32へ制限した。しかし現時点の以下のローカルファイルには救済変更が入っていない。

- `solar_filament_v34_semantic_instance_hybrid_colab.ipynb`
- `scratch/build_v34_semantic_instance_hybrid.py`

ローカルV34をそのまま再実行すると、全OOF評価が再び特定画像で停止する可能性がある。保存目的で確定するならビルダーへ以下を反映して再生成する。

- `MAX_INSTANCES = 32`
- connected component候補の面積順制限
- 候補ごとのpartial CSV保存
- progress barへの候補番号・設定値表示
- `fold0_names`の可視化セル内再構築

### 17.3 ノートブック依存関係

```text
competition data
  └─ V30
      ├─ grouped_folds.csv
      ├─ YOLO fold 0/1/2 best.pt
      ├─ held-out MediSeg refiner
      ├─ all-data MediSeg refiner
      └─ overlapping submission CSV
          ├─ V30b disjoint repair
          └─ V32 overlap-cluster repair

V30 artifacts
  ├─ V31 held-out postprocess ablation
  ├─ V33 data-cache reuse
  └─ V34
      ├─ V30 grouped folds
      └─ V30 YOLO fold weights as watershed seeds
```

V30から後続版へ必要な主要成果物:

```text
/content/drive/MyDrive/solar_filament_v30/
├── competition_data/
├── preprocessed_v30_q96/
├── runs/
│   ├── yolo_fold_0/weights/best.pt
│   ├── yolo_fold_1/weights/best.pt
│   └── yolo_fold_2/weights/best.pt
└── artifacts/
    ├── grouped_folds.csv
    ├── mediseg_refiner_fold0_resume.pt
    ├── mediseg_refiner_all_data_resume.pt
    ├── calibrated_params.json
    ├── calibration_grid.csv
    ├── test_instance_audit.csv
    └── submission_v30_prize_instance_ensemble.csv
```

存在確認だけでなく、ファイルサイズ、hash、生成runをmanifestへ記録する。別runのweightを同名で上書きすると再現不能になる。

### 17.4 ローカルコードの役割

`src/config.py`は汎用の512x512、5-fold、UnetPlusPlus/EfficientNet-B3設定である。一方、V30以降は各ビルダー内に独自CFGを持つ。`src/config.py`を読んでV30/V31/V33/V34の実験条件を判断してはいけない。将来は設定の二重管理を解消する。

- `FallbackUNet2D`: 一般的な4段U-Net。
- `MediSegUNet`: 64→1024 channelのU-Net。V28～V31のcrop/semantic系で利用。
- `MediSegDeepLabV3Plus`: 独自stem + ASPP。主要Public提出での採用実績なし。
- `CombinedLoss`: BCE 0.5 + Dice 0.5。

V30 notebook内にもMediSeg実装が埋め込まれている。ローカルmoduleとnotebook埋込版の差分を防ぐため、将来は単一sourceから生成する。

### 17.5 既知のドキュメントずれ

- `README.md`の例示pathは`C:\work\code\Solar_Filament_Segmentation`で、現在の実pathと異なる。
- READMEは512x512・5-foldの初期構成を説明し、V30以降の3-fold instance pipelineを反映していない。
- V21とV22はローカルhashが同一で、V22先頭タイトルもV21のまま。V22は独立確定版ではなく配置修正履歴として扱う。

### 17.6 再現環境と依存パッケージ

リポジトリrequirements:

```text
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.7.0
pillow>=9.5.0
scikit-learn>=1.2.0
segmentation-models-pytorch>=0.3.3
timm>=0.9.0
albumentations>=1.3.0
einops>=0.6.0
kaggle>=1.5.13
tqdm>=4.65.0
matplotlib>=3.7.0
```

V30以降では、これに加えて`ultralytics`、`pycocotools`、`scipy`、`scikit-image`を使う。

requirementsが`>=`指定のため、Colab Latest Container更新でAPIや既定値が変わり得る。正式runではPython、PyTorch、CUDA、GPU、`pip freeze`、Ultralytics model YAML、pretrained weight、dataset JSONのSHA-256を保存する。

### 17.7 主要ローカル成果物のfingerprint

2026-08-15時点。内容変更後はhashを更新する。

| ファイル | bytes | SHA-256 |
|---|---:|---|
| `best_solar_unet.zip` | 55,121,275 | `5cbed2fc2e798b1c7a33f727015bcd081785ddc8932ecf06848c35c90f9132c8` |
| `solar_filament_v30_prize_colab_instance_ensemble.ipynb` | 52,686 | `8d50ef2ee8c0a1680e6c06b8aa515d3d4d1c78b122d8b245fa630ec6a534477f` |
| `solar_filament_v31_heldout_postprocess_ablation.ipynb` | 33,977 | `ad98d52ae071e712a21b65c46c52ea77e84352e1df64bbb30fea448d4b81b4d2` |
| `solar_filament_v32_overlap_cluster_repair.ipynb` | 14,860 | `5aaf81b6665ffc3af0d7c955872738aafbddb42a7423479fee6ff6937a59ef73` |
| `solar_filament_v33_public069_reproduction_colab.ipynb` | 33,320 | `d6df5a2f2f042d5f096f0bcf0ac8aaa9d698213d634fbe34013db55ac0553777` |
| `solar_filament_v34_semantic_instance_hybrid_colab.ipynb` | 46,046 | `aca3514c73e3f27d584e92a953b9eefc91f328d58deab0bf78660339293b72a8` |

`best_solar_unet.zip`は形式復旧履歴として保持するが、実データinstance提出のモデル正本にはしない。

## 18. 評価プロトコルとリーク管理

### 18.1 split単位

同じ観測画像に複数labeler annotationがあるため、split keyはannotation IDではなく観測画像file名／stemにする。全label variantを同じfoldへ固定する。

### 18.2 refinerのリーク

- パラメータ選択用refinerは対象held-out foldを学習に含めない。
- test用all-data refinerは全trainを使ってよいが、その出力でOOFパラメータを再調整しない。
- warm start元checkpointをmanifestへ記録する。

### 18.3 後処理探索

同じfoldで候補を大量探索してbestを報告するとselection biasが生じる。fold 0で候補空間を粗く絞り、上位候補を全OOFで再評価し、採用条件を事前固定する。Public Scoreを見た反復的閾値調整は最小限にする。

### 18.4 複数labelerの評価

V30/V31では各annotation variantに対するinstance Diceを平均した。公式metricとの一致は未確認。次回はvariant平均、variant最大、soft consensus、annotator間agreementを併記し、disagreementが大きい画像を別層で報告する。

## 19. 提出管理と最終選択

### 19.0 【改訂2で追加】Kaggle外の最終提出が必須（この文書に欠落していた）

**このコンペは純粋なリーダーボード競争ではない。** Kaggle Overviewで確認した。

- 審査は **定量70% / 定性30%** の委員会評価。「リーダーボード順位は評価の考慮の一つ」と明記されている
- 定量70%の内訳: Panoptic Quality、Dice分布、IoU分布、一対多／多対一関係の分布、および**エンドツーエンドの計算効率**
- 定性30%の内訳: パイプライン全体の記述（前処理から最終予測、アーキテクチャまで）、予測マスクの見た目の妥当性、**コード品質（モジュール性・文書化）**
- **2026-11-15までに、Google Form経由で以下を提出しないと最終評価の対象外になる**
  - 4ページの技術報告書（Overleafテンプレート指定、単一PDF）
  - 公開Gitリポジトリ（追加の許可申請なしにアクセスできること）。`requirements.txt`にバージョン付きで全依存を記載し、パイプライン全体を示すJupyter notebookを含めること。委員会が再現に追加ファイルを要さないこと
- リポジトリはコンペ終了と同時に公開し、受賞発表まで公開を維持する義務がある
- 賞金 $3,000 を上位3チームへ `score^2` で比例配分（`compute_prizes(scores, total_prize=3000, temperature=2.0)`）
- Public LBはtestの約50%のみ。残りは主催者のみが見る

**この文書への含意:**

- §17.4の設定二重管理（`src/config.py` と各ビルダー内CFG）、§17.5のREADMEずれ、§29の自動テスト不在は、**そのまま定性評価30%の失点になる**。技術的負債ではなく採点対象。
- 報告書とリポジトリ整備は締切直前に片付く量ではない。実験と並行して進める。
- 「計算効率」が定量評価に入っているので、推論時間も記録対象に加える（§34のmanifestへ`inference_seconds_per_image`を追加すること）。

### 19.1 日次提出枠

残り提出回数は`Submit Prediction`ダイアログ上部に表示される。共有画面では「残り3、13時間後リセット」の例を確認した。値は日ごとに変わるため固定値として扱わない。

### 19.2 最終リーダーボード選択

画面上では最大2提出をfinal leaderboard用に選択できる。未選択の場合はKaggleがbest scoring submissionsから自動選択すると表示されていた。

現時点の手動選択候補は Public `0.37` の V35 を第一候補とし、多様性用に V30b disjoint または V32b（いずれも 0.35）を残す。private diversity の証拠はないため、締切前に規則と提出一覧を再確認する。

### 19.3 提出説明欄

各提出にはmodel family、fold数、postprocess名、OOF Dice、count error、overlap 0、source manifestを記載し、Public結果と条件を後から対応可能にする。

## 20. Drive成果物の保全

Google DriveはV30～V34のweightとcacheの実質的な正本であり、単一障害点になっている。

バックアップ対象:

- competition data fingerprint
- grouped folds
- 全fold best/last weight
- optimizer/scheduler state
- refiner checkpoint
- OOF/test raw predictions
- OOF metrics
- submission CSVとvalidation report
- environment manifest

同じ`best.pt`を上書きせず、versionとrun IDをpathに含める。

推奨構造:

```text
solar_filament_archive/
└── vXX_<experiment_slug>/
    ├── README.md
    ├── environment.json
    ├── data_manifest.json
    ├── config.json
    ├── folds.csv
    ├── weights/
    ├── oof/
    ├── test/
    ├── submissions/
    └── reports/
```

## 21. 実験decision log

| 判断 | 根拠 | 現在の扱い |
|---|---|---|
| synthetic `best_solar_unet`を主モデルにしない | 形式成功後Public 0.00 | archiveのみ |
| semantic connected-componentsを主方式にしない | union Dice 0.63でもPublic 0.19 | 棄却 |
| YOLOをinstance detectorとして使う | V30bでPublic 0.35 | 継続 |
| MediSegを無条件に最終maskへ使わない | V31でYOLO nativeとの差がほぼない | 境界補正に限定 |
| overlap 0を提出ゲートにする | Kaggleが重複を拒否 | 必須 |
| 単一fold bestで提出しない | V31/V34で一般化差 | 全fold OOF必須 |
| 申告0.69を性能根拠にしない | V33実提出0.33 | negative resultとして保存 |
| semantic-first V34を継続しない | Dice/countともV31より悪化 | 棄却 |
| test推論前ゲートを置く | V34で無駄な提出を防止 | 継続・強化 |
| **【改訂2】ローカル指標をPanoptic Qualityへ全面移行** | **公式実装を確認。Hungarian Diceは公式と別物だった** | **確定。`src/metrics_official.py`** |
| **【改訂2】Diceゲート`0.552553`を廃止** | **公式でない指標、かつfold0のみの選択バイアス** | **廃止。PQ基準へ差し替え（§13.2）** |
| **【改訂2】公開0.69の再現追跡を打ち切り** | **LB1位0.48を上回るため論理的にありえない** | **終了。V33の0.33が正しい実測** |
| **【改訂2】GPU不要の診断はローカルPCで実行** | **DriveがG:\にマウント済み。Colabより速くCU消費ゼロ** | **継続（§12.2.1）** |
| **【改訂2】MIoUを主指標にしない** | **KaggleのEvaluationに含まれない。IEEE紹介ページのみ** | **副次指標として保持** |

## 22. リスク登録簿

| リスク | 影響 | 兆候 | 対策 |
|---|---|---|---|
| labeler leakage | OOF過大評価 | 同一fileが複数fold | grouped split assert |
| Public overfitting | private低下 | 小変更を多数提出 | OOF gate、提出回数制限 |
| dependency drift | 再現不能 | Latest Container更新 | exact freeze、hash保存 |
| Drive消失／上書き | weight喪失 | 同名best.pt | versioned archive |
| mask重複 | 提出拒否 | occupancy>1 | 全件decode検査 |
| instance explosion | 評価停止 | 1画像で長時間停止 | max instances/components |
| count underprediction | score低下 | pred count < GT count | detector recall診断 |
| merge/split | instance Dice低下 | pixel Diceは高い | split/merge metric |
| test distribution shift | Public低下 | fold差大 | fold別report、robust augmentation |
| secret漏洩 | account risk | notebookにkey | Colab Secretsのみ |

## 23. セキュリティと認証情報

- `KAGGLE_USERNAME`と`KAGGLE_KEY`はColab Secretsへ登録し、本文・出力・manifestへ書かない。
- `HF_TOKEN`はrate limit緩和用で必須ではない。使う場合もSecrets経由。
- Kaggle APIエラー時に環境変数全体をprintしない。
- 公開ノートブック由来コードはライセンスとURLを残す。参照したYOLO + U-Net notebookはApache-2.0として扱った。

## 24. 実行前・提出前チェックリスト

### 24.1 学習開始前

- [ ] competition rulesを再確認
- [ ] Internet要否を決定
- [ ] GPU、RAM、package versionsを記録
- [ ] train JSONと画像一覧のhashを保存
- [ ] grouped foldsの重複0をassert
- [ ] resume先run IDを確認
- [ ] DEBUGがOFFであることを確認

### 24.2 OOF評価前

- [ ] 各fold weightが別々である
- [ ] 対象foldを学習に含めていない
- [ ] raw OOF predictionsを保存
- [ ] instance数上限と異常画像logがある
- [ ] metric定義を固定
- [ ] fold別と全fold平均を両方出す

### 24.3 test推論・提出前

- [ ] 事前定義OOF gateに合格
- [ ] Public結果を見てtest parameterを再調整していない
- [ ] checkpoint hashをmanifestへ記録
- [ ] null 0、空RLE 0、duplicate ID 0
- [ ] RLE round-trip、overlap pixel 0
- [ ] row/image/count/area分布を確認
- [ ] CSV SHA-256を保存
- [ ] 日次提出残数を確認

## 25. 新しい担当者が最初に行うこと

1. この文書を読む。**§2.1（改訂2）を最優先で読むこと。指標が変わっている。**
2. `output/v32/v32_manifest.json`と主要notebookのhashを確認する。
3. Driveの`solar_filament_v30`、`v33`、`v34`を別保存する。
4. `grouped_folds.csv`と3個のV30 YOLO weightを確認する。
5. ~~V31のmetric関数を独立moduleへ抽出し、固定fixtureでtestを書く。~~ **【完了】** ただし抽出対象はV31のDiceではなく公式のPanoptic Quality。`src/metrics_official.py` + `tests/test_metrics_official.py`（19件）。
6. **`python scratch/run_pq_oof_rescore.py` をローカルPCで実行し、PQのベースラインとゲートを確定させる（約20分、GPU不要）。** 結果は`output/v35/pq_gate.json`。
7. V30bとV32候補を同じvalidatorで再検査する。
8. V35をV30 builderからforkする。ただし**組み込む評価はV31のDiceではなくPQ**。§2.1.4のSQ/RQ分解の結果に応じて、recall重視（RQ律速）か境界品質重視（SQ律速）かを決めてから設計する。
9. **§19.0を読み、技術報告書とGitリポジトリの準備を実験と並行で開始する。** 11月15日の締切はKaggleの提出ではなくGoogle Formの提出であり、これを逃すと最終評価の対象外になる。

## 26. V35で最初に検証する仮説

> **【改訂2】仮説を選ぶ前に、まずPQのSQ/RQ分解を測ること。** どれが主因かはP0〜P3を個別に検証しなくても、分解でほぼ決まる。
>
> - **RQ（検出の質）が支配的に低い** → P0（recall不足）とP1（duplicate＝FP）。候補を増やす方向へ動く。PQ 0.35・IoU 0.7 なら TP確率25%の候補でも出した方が得なので、conf 0.26 は保守的すぎる可能性が高い（§2.1.2-5）。
> - **SQ（TPの平均IoU）が支配的に低い** → P2（refinerが縮めすぎ）と排他化の副作用。IoU 0.3〜0.5 の near-miss が厚いかどうかで確認できる。
> - **fold間のPQ差が大きい** → P3。
>
> P2については、PQ下では **V30bの660,119画素の先着減算そのものが第一容疑**になった。重なり画素を confidence 先着で削るとマスクのIoUが直接下がり、0.5の崖を割った予測は得点ゼロかつFP+FNの二重減点になる。§13.1-5の「pixel ownership scoreで割り当てる」は、**「各マスクがIoU 0.5を越える数を最大化する」という明確な目的関数**を持つ最適化問題として定式化し直せる。
>
> **【2026-08-15 実測後の結論】§2.1.6を参照。P1（duplicate）とP3（fold不安定）は実測で否定された。P0（recall不足）はconf操作では解決しない（下げるとFPの方が増える）。**
>
> **残ったのはP2、すなわち境界品質。** near-miss（IoU 0.3〜0.5）が1209件あり、PQの分母はこれをTPへ変える操作に対して不変なので、変換できた分がそのまま加点になる。上限で +0.084（0.394→0.478）。V35はここに集中する。
>
> P2の具体的な下位仮説:
> 1. ~~先着排他化による画素削り~~ **【測定済み・ほぼ否定】** §2.1.7参照。オラクル上限でも +0.0095 しかない
> 2. YOLOのmask prototype解像度（1280px入力 → 実効320²相当）。細長い個体でIoUが頭打ちになる **【測定済み・外れ】** P2-2: 1536 再学習＋V35後処理は fold0 PQ 0.4295。P2-3: 1280 学習重みのまま推論だけ 1536 でも 0.4320 < 0.4393。解像度レバーは停止
> 3. MediSeg refinerの方式と閾値がDice最大化で選ばれている **【測定済み・当たり】** §2.1.8参照

### 2.1.7 排他化の画素割当方式【測定済み・ほぼ否定】

`scratch/run_ownership_ablation.py`、conf 0.26、全707観測、ローカルPCで60分。

| 方式 | PQ | Δ vs 現行 |
|---|---:|---:|
| oracle（GTを見た上限。提出不可） | 0.4033 | +0.0095 |
| raw（重複を許したまま。提出不可） | 0.4021 | +0.0083 |
| **drop_contained → area_asc** | **0.4007** | **+0.0069** |
| drop_any_overlap | 0.3984 | +0.0047 |
| drop_contained → conf | 0.3974 | +0.0037 |
| area_asc | 0.3964 | +0.0026 |
| **conf_first（V30bの現行方式）** | 0.3937 | — |
| area_desc | 0.3918 | −0.0019 |

**「V30bの660,119画素の先着減算が第一容疑」という改訂2時点の見立ては外れた。** オラクル上限ですら +0.0095 しかない。`raw`（排他化しない）が +0.0083 に留まることから、**Kaggleの重複禁止制約そのものはほぼ無害**である。失われる画素は平均2.86%（中央値0）。

ただし **+0.0069 は学習不要で即得られる改善**であり、最良規則が上限の73%を回収している。採用してよい。

**重要な留保:** これは単一fold OOFの値である。単一foldには奪い合う重複がほとんどない（1枚あたり0.5ペア）。V30の660,119重複画素は**3-fold ensembleの合流で生じた**もので、OOFでは再現できない（各foldのモデルが他foldを学習済みのためensembleをOOFで作れない）。**ensemble時の利得はこの値より大きい可能性があり、それはOOFでは検証できない。** これは方法論上の限界として明記しておく。

### P0: detector recall不足が主因

V33でGT平均7.83に対し予測6.34、refiner crop Dice 0.80以上でもPublicが改善しないことが根拠。proposal recallとfinal precisionを分離し、FN/FP/duplicate/merge/splitを画像別に分類する。

### P1: fold ensembleのduplicate統合が不十分

V30の66万重複画素とV32の高包含候補が根拠。fold間mask matching、cluster内probability voting、cluster間pixel ownershipを検証する。

### P2: refinerがmaskを縮めすぎる

V31でguidedとYOLO nativeがほぼ同値、V33でYOLO平均7.05から提出6.90へ減少。YOLO native、inside-only、narrow-band correctionを同一全fold OOFで比較する。

### P3: 画像前処理がfold間で不安定

V34のfold差が根拠。観測時刻、instrument suffix、contrast、foreground countで層別評価し、前処理有無を同一OOFで比較する。

## 27. 追加で収集すべき診断表

次回runでは画像ごとにfile名、fold、GT variant数、GT/pred count、proposal count、matched/FN/FP、split、merge、duplicate cluster、strict Dice、foreground fraction、contrast、runtimeをCSV化する。

worst 30画像について、原画像、GT variants、proposal、refined mask、final disjoint maskを並べたcontact sheetを保存する。

### 27.1 実行結果（改訂7、2026-08-16夜）

`scratch/run_case_diagnostics.py` と `scratch/run_near_miss_modes.py` を全707観測（V35確定パイプライン: conf 0.26 + shrink@0.45 + drop_contained_then_area_asc、リークなしOOF）で完走させた。

**発生した障害と対処（再現性のため記録）:**

1. `mediseg_refiner_fold*_resume.pt` を Google Drive の日本語パスから直接 `torch.load` すると `OSError: [Errno 22] Invalid argument`。→ `%TEMP%\solar_diag_ckpts\` にコピーしてから読む（既知、改訂6以前から）。
2. OOF npz キャッシュ（`prediction_cache/oof/yolo_fold_*`、1414ファイル・0.59GB）も同じ理由で460枚目付近で無言OSError。→ `%TEMP%\solar_diag_oof\` にrobocopyでコピーしてから読む（新規）。
3. 前処理画像（887枚・1.55GB）も同じ理由で同期不良のリスクがある。→ `%TEMP%\solar_diag_images\` にrobocopyでコピーしてから読む（新規）。
4. `run_case_diagnostics.py` は707画像すべての `(image, variants, proposals, final, matched_flags)` を `keep_for_sheet` に保持していたため、490枚目付近で `numpy._core._exceptions._ArrayMemoryError`（2048×2048 boolマスクの累積）。→ 2パス設計に変更: 1パス目はCSV行だけを計算して画像データは捨て、worst30 のcontact sheetは2パス目で該当ファイルだけ再計算する。

上記3点を修正後、707観測とも完走（診断: 所要2975秒、near-miss分類: 所要2444秒）。両スクリプトともTEMPコピーが無ければDriveへフォールバックするので、TEMPを消しても壊れない。

**`case_diagnostics.csv`（707行）の要点:**

- 全体: TP 5106 / FP 2494 / FN 3093、局所PQ平均 0.4325（公式のmicro集計PQ 0.4426とは定義が異なる順位付け用の値）
- `loss_units`（0.5×(FP+FN)、全体スコアへの押し下げ量）との相関: near_miss数 **0.853**（最強）、proposal_count 0.533、gt_count_mean 0.515、n_annotators 0.505、foreground_fraction 0.450、local_pq -0.316
- worst15画像はいずれも `gt_count_mean` 10〜21個体の密集シーン（フィラメント数が多い観測ほど損失が集中する）

**`near_miss_modes.json`（IoU 0.3〜0.5、955ペア）の要点:**

| モード | 件数 | 割合 |
|---|---:|---:|
| undercoverage | 433 | 45.3% |
| overcoverage | 400 | 41.9% |
| shape_mismatch | 76 | 8.0% |
| ambiguous | 46 | 4.8% |

参考: `hard_fn`（IoU≤0.3の完全な取りこぼし）2138件 vs `hard_fp`（IoU≤0.3の余剰検出）1561件。

**読み方（次の一手への示唆）:**

near-missのundercoverage/overcoverageはほぼ拮抗しており、どちらか一方に倒れる明確な支配モードではない。「削る系」だけ強化するとovercoverageは減らせてもundercoverageが相対的に悪化するリスクがあり、その逆も同様。一方で、IoU≤0.3の完全な取りこぼし（hard FN）が余剰検出（hard FP）を37%上回り、loss_unitsは検出個体数の多い密集シーンに強く紐づく。これらを総合すると、最有力なのは「後処理の閾値調整」ではなく「密集シーンでの検出漏れそのものを減らす検出器側の強化」（imgsz以外：モデルサイズ、augmentation、multi-seedアンサンブルなど）だが、モード拮抗という事実自体もこの文書に残す。次の本実験を設計する前に、この読みをユーザーと確認すること。

### 27.2 Kaggle無料GPU枠でのV38/V39実行結果（改訂8、2026-08-17）

§27.1の示唆（密集シーンの検出漏れを減らす検出器強化）を受けて、imgsz以外の2方向をKaggle無料GPU（T4、週30時間枠）でfold0のみ・V35と同じ70epochで実際に試した。ローカルDrive不要、Kaggle Dataset `shingo257/solar-filament-v35-artifacts`（前処理画像887枚+refiner checkpoint3個+fold分割csv、2.9GB、非公開）とcompetition `filament-segmentation-2026` を直接アタッチして学習した。

| 実験 | 変更点 | fold0 OOF PQ（確定後処理つき） | V35基準0.4393 | 判定 |
|---|---|---:|---:|---|
| V38 augmentation強化 | `mosaic 0.20→0.75` + `copy_paste 0.0→0.3`新規、他はV35と同一 | 0.4290 | Δ −0.0103 | 不合格 |
| V39 モデルサイズ拡大 | `yolo11m-seg→yolo11l-seg`、VRAM都合で`batch 4→2`、他はV35と同一 | 0.4328 | Δ −0.0065 | 不合格 |

**生YOLO（refinerなし、conf 0.26、同一排他化）での切り分け:**

| 検出器 | conf_first | drop_contained |
|---|---:|---:|
| V30 1280学習（参照値） | 0.3977 | — |
| V38 augmentation強化 | 0.3889 | 0.3950 |
| V39 モデルサイズ拡大 | 0.3859 | 0.3941 |

P2-2/P2-3（imgsz変更）は生YOLOでは検出器が僅かに改善していたのに確定後処理では負けていた（後処理相性の問題）。**V38/V39はこれと異なり、生YOLOの時点で既にV30基準を下回っている**。つまり検出器そのものが悪化しており、後処理の相性問題では説明できない。

**ただし単純な「効かなかった」で終わらせられない交絡がある:**

- V38はV35と同じ70epochのまま`mosaic`と`copy_paste`だけ大幅に強めた。YOLOの標準運用では`close_mosaic`で終盤にmosaicを切って収束させる設計になっており、augmentationを強めるほど収束に必要なepoch数も増えるのが一般的。70epoch据え置きでは収束不足だった疑いが強い。
- V39はモデルを大きくした一方、VRAM制約で`batch`を4→2に落としたが、`lr0=7e-4`（batch4向けにチューニングされた値）はそのまま使った。実効バッチサイズが半減すると勾配が不安定になりやすく、学習率かgradient accumulationで補正しないと大型モデルの効果を正しく測れない。

このためこの2つの実験結果は「augmentation強化」「モデル大型化」という方向性そのものを完全に否定するものではなく、「V35のスケジュールをそのまま流用した単軸変更ではダメだった」という結果として記録する。imgsz(P2-2/P2-3)・augmentation(V38)・モデルサイズ(V39)の4本連続不合格により、**検出器側を1箇所だけいじって確定後処理はそのまま、というアプローチの限界**が見えてきた。次にこの方向を続けるなら、変更した軸に応じてepoch数・学習率・close_mosaic時期なども一緒に調整する必要がある。

成果物: `output/v38_v39/v38_v39_raw_yolo_fold0.json`、Kaggleカーネル `shingo257/v37-kaggle-timing-probe`（パイプライン疎通確認、fold0のみ3epochタイミング計測: yolo11m-seg/imgsz1280/batch4で1epoch約224.5秒、70epoch換算で約4.37時間）、`shingo257/v38-kaggle-augmentation-probe`、`shingo257/v39-kaggle-modelsize-probe`（いずれも非公開）。

**つまずいた点（再現性のため記録）:**

- Kaggleのinput マウントレイアウトは `/kaggle/input/datasets/<owner>/<slug>/` 形式（`/kaggle/input/<slug>/` 決め打ちだと`FileNotFoundError`になる）。バージョンで変わりうるので `Path("/kaggle/input").rglob("grouped_folds.csv")` のようなglob探索で解決すること
- `kaggle kernels output` はWindowsのコンソールコードページ(cp932)によっては絵文字入りログで`UnicodeEncodeError`になる。標準出力をファイルへリダイレクト（`> log.txt 2>&1`）すれば回避できる
- YOLOの`cache="disk"`は`/kaggle/working`配下（=kernel出力）に画像ごとの`.npy`キャッシュを作るため、kernel outputが数GB〜10GB超に膨れる。ローカルのダウンロード先ディスクを圧迫するので、必要なファイルだけ`kaggle kernels output --file-pattern`で絞って取得すること
- Kaggleの同一アカウントはプロジェクトを跨いでGPUバッチセッション上限が2本に共有される。他プロジェクトのカーネルが動いていると`Kernel push error: Maximum batch GPU session count of 2 reached`になる。他プロジェクトを止めずに空き待ちでリトライするのが安全

### 27.3 後処理再設計: 癒着分離(watershed split)は不採用（改訂8、2026-08-17）

§27.1の示唆のもう一方（検出器強化ではなく後処理側）として、「近接する2個体をYOLOが1つの塊として検出し、一方がovercoverage・もう一方がundercoverage(検出漏れ)になる」という癒着仮説を検証した。GTを使わず、refiner shrink後のmask形状だけから、距離変換の大域閾値でcore領域を作り、core領域が2つ以上の連結成分に分かれる場合だけwatershedで分割する(`src/blob_split.py`)。挿入位置はrefiner shrink後・排他化(`drop_contained_then_area_asc`)前。

合成図形（ダンベル・単一楕円・湾曲したフィラメント）では正しく動作したが、実データでは閾値に非常に敏感だった。

| core_fraction | 40観測でのsplit発生回数 | PQ(baseline 0.4388 比) |
|---|---:|---:|
| 0.55 | 131回 | 0.2369（Δ **-0.2020**、壊滅的） |
| 0.80 | 2回 | 0.4361（Δ -0.0027、ほぼ横ばい） |

0.55では細長い実際のフィラメント形状に対して過分割が暴発し、PQが半減した。0.80まで厳しくすると過分割は止まるが、40観測中2回しか発動せず、near-miss被覆モード診断で見つかったunder/overcoverage(707観測で計833件、1画像あたり約1.2件)の規模に対してあまりに希少で、効果はノイズレベルに埋もれた。**「物理的に癒着した2個体」という現象自体が、under/overcoverageの主因ではない**と判断し、この実装での採用は見送る。全707観測での確認は行わずローカル40観測の段階で判断した（明確な負の結果のため）。

成果物: `src/blob_split.py`、`scratch/run_blob_split_probe.py`、`output/v35/blob_split_probe.json`。

### 27.4 交絡を補正したV40/V41の再測定（改訂9、2026-08-18）

§27.2で「V38/V39はV35のスケジュールを流用した単軸変更なので交絡がある」と書いた。
交絡を補正して再測定した結果、**epoch/warmup補正（V40）もlr補正（V41）も改善しなかった**。

| 実験 | 変更点 | fold0 OOF PQ | 基準0.4393との差 | 判定 |
|---|---|---:|---:|---|
| V38（元） | mosaic0.75+copy_paste0.3、epoch70 | 0.4290 | -0.0103 | 不合格 |
| **V40（epoch/warmup補正）** | 同上 + epoch70→**100**、warmup3→**5** | **0.4272** | **-0.0121** | 不合格（V38よりさらに悪化） |
| V39（元） | yolo11l-seg、batch4→2、lr0=7e-4据え置き | 0.4328 | -0.0065 | 不合格 |
| **V41（lr補正）** | 同上 + lr0 7e-4→**3.5e-4**（線形則） | **0.4316** | **-0.0077** | 不合格（V39よりわずかに悪化） |

V40はColab/Kaggle双方でなくKaggle実行（実測27372秒≈7.6時間）。V41はKaggleで
`Maximum batch GPU session count of 2 reached`（他プロジェクトとGPU枠を共有）に
遭遇後、ユーザーがGPU無料枠残量（週30時間中残り約12時間）を懸念して手動停止し、
Colab版（`scratch/build_v41_colab_modelsize_v2.py` → Drive経由、resume機構つき）
へ切り替えて完走させた（fold0 PQ 0.4316）。

**読み方:** 交絡の補正は効果を反転させなかった。むしろV40はV38よりわずかに悪化しており、
「epoch不足」「lr不整合」という交絡仮説そのものが的外れだった可能性が高い。
これでimgsz（P2-2/P2-3）・augmentation（V38/V40）・モデルサイズ（V39/V41）の
**検出器強化系6本すべてが不合格**になった。V35確定パイプライン（yolo11m-seg、
imgsz1280、batch4、epoch70、mosaic0.20、shrink@0.45+drop_contained_then_area_asc）
は、この課題・このデータ規模（707観測、fold0は約470枚学習）に対してかなり
局所最適に近い設定であり、検出器側の単純な強化では動かしにくいことが示唆される。
次に検出器を強化するなら、単一ハイパーパラメータの変更ではなく、データ拡張
（アノテーション不足を補う合成データ生成など）や複数軸の同時最適化が必要になる
可能性が高いが、コストが跳ね上がるため慎重な判断を要する。

成果物: `scratch/build_v40_kaggle_augmentation_v2.py`、
`scratch/build_v41_kaggle_modelsize_v2.py`（Kaggle、途中停止）、
`scratch/build_v41_colab_modelsize_v2.py`（Colab、完走）。

### 27.5 multi-seedアンサンブルも不合格（改訂9、2026-08-18）

検出器のハイパーパラメータには一切触れず、V35確定と完全同一設定（yolo11m-seg,
imgsz1280, batch4, epoch70, mosaic0.20）で`seed`だけ`3407→8407`に変えて
fold0モデルをもう1本学習した（`scratch/build_v42_kaggle_multiseed_probe.py`、
Kaggle T4、GPU予算節約のためseed3407本体は既存OOFキャッシュを再利用）。
画像ごとに両seedのproposalを単純に連結し、V35確定後処理（refiner shrink@0.45
+ drop_contained_then_area_asc）へそのまま通してアンサンブルPQを測った
（`scratch/run_multiseed_ensemble_probe.py`、ローカルCPU、236観測=fold0全件）。

| | PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| single seed3407（ローカル再計算、サニティチェック） | 0.4393 | 1713 | 789 | 1036 |
| ensemble seed3407+8407 | **0.4315** | 1766 | **979** | 983 |

single seed3407の再計算値がV35確定fold0実測(0.4393)と完全一致しており、
採点手法の整合性は確認できた。アンサンブルはTPを+53増やしたが、FPを+190
増やしてしまい正味でΔ**-0.0078**の悪化。単純な足し合わせでは、2モデル
それぞれ固有の誤検出が両方とも残ってしまい、`drop_contained`（IoU≥0.90の
重複除去）では吸収しきれなかった。より洗練された統合（WBF等の重み付き融合、
2モデルの一致度でconfを再スコアリングする等）なら結果が変わる可能性は
残るが、単純連結という最も安価なアンサンブル手法は不採用。

**これで、後処理再設計（watershed癒着分離、§27.3）・検出器強化6本
（imgsz/augmentation/モデルサイズ、§27.2・§27.4）・multi-seedアンサンブル
（本節）の**すべてが不合格**となった。V35確定パイプラインはこのデータ規模
（707観測）に対してかなり頑健な局所最適であることが、複数の独立な角度から
裏付けられた。

成果物: `scratch/build_v42_kaggle_multiseed_probe.py`、
`scratch/run_multiseed_ensemble_probe.py`、`output/v42/multiseed_ensemble_probe.json`。

### 27.6 WBF(Weighted Mask Fusion)で単純連結より改善、ただし基準未満（2026-08-18〜19）

単純連結アンサンブルの敗因（FP+190がTP+53を上回る）を受け、IoUクラスタ化+
confidence加重平均でmaskを統合するWBF（`src/mask_wbf.py`、box用WBFのmask版、
合成テストで往復性・クラスタリング挙動を確認済み）を実装し、
refinerの**前**（生YOLO proposal同士）に挿入して比較した
（`scratch/run_multiseed_wbf_probe.py`、fold0全236観測、ローカルCPU）。

| 統合方式 | fold0 PQ | 基準0.4393との差 | TP | FP |
|---|---:|---:|---:|---:|
| 単純連結（§27.5） | 0.4315 | -0.0078 | 1766 | 979 |
| **WBF(iou≥0.5、全クラスタ保持)** | **0.4366** | **-0.0027(最良)** | 1787 | 972 |
| WBF(iou≥0.5、コンセンサス限定=2seed一致のみ) | 0.4355 | -0.0038 | 1635 | 690 |

WBF全保持が最良。単純連結よりFP抑制とTP確保のバランスが改善したが、まだ
基準を下回る。コンセンサス限定（単独seedのクラスタ378件を全除外）は
FPを大きく削れた（789→690）が、その378件の中に真陽性も相当数含まれていた
ため、TPも大きく失って（1713→1635）正味では中間の結果に留まった。
**「片方のseedだけの検出＝誤検出」という単純な仮定は成り立たない。**

次の一手として、追加学習なしで試せる**TTA(test-time augmentation)+WBF**を
Colabで実行中（`scratch/build_v43_colab_tta_wbf_probe.py`、V30 fold0確定
モデルに対し元画像/左右反転/上下反転/180度回転の4視点で推論しWBF統合、
GPU予算をほぼ使わない）。結果は追記予定。

**IoU閾値スイープ(60観測サンプル、ローカルCPU、2026-08-19):** `iou_threshold`を
0.3/0.4/0.6/0.7で振ったところ、いずれも0.5(Δ-0.0027、全236観測)より劣った
(0.3: Δ-0.0053、0.4: Δ-0.0039、0.6: Δ-0.0106、0.7: Δ-0.0094)。0.5付近が
局所最適で、閾値だけのチューニングによるさらなる改善余地は乏しい。

成果物: `src/mask_wbf.py`、`scratch/run_multiseed_wbf_probe.py`、
`output/v42/multiseed_wbf_probe_iou0.50.json`、
`output/v42/multiseed_wbf_probe_iou0.50_consensus2.json`、
`output/v42/multiseed_wbf_probe_iou{0.30,0.40,0.60,0.70}.json`(60観測サンプル)。

## 28. 時間とコストの概算

実測例であり保証値ではない。

- V34 T4 semantic学習: batch 2 + accumulation 2で約7.5分／epoch
- V34 OOF semantic推論: foldあたり約6～7分
- V34 YOLO OOF: foldあたり約4分
- 救済all-fold候補評価: 707画像あたり約3分40秒、3候補約11分
- V30前処理: 887画像で約20分
- V30 refiner: 1 epoch約1分前後

T4はA100より遅いがCU消費が大幅に低かった。前処理、後処理探索、OOF cache生成はT4またはCPU、重いYOLO/semantic学習だけA100という分業が合理的。

## 29. 静的検証とテストの現状

- builderからipynbを生成し、JSON／Python構文を確認した履歴はある。
- credential付きKaggle/Colab実行はローカル静的検証とは別。
- 現在、このフォルダにはvalidator、metric、mask clusterに対する体系的な自動test suiteがない。

最低限、compressed RLE round-trip、string ID、empty mask、overlap 0、deterministic disjoint、Hungarian Dice、split/merge、grouped leakage、pathological componentsのtestを追加する。

## 30. 文書更新ルール

新しい実験ごとに、日時、notebook/builder hash、data/environment fingerprint、fold、checkpoint hash、OOF指標、test統計、CSV hash、Public Score、採否理由を記録する。

未確認値は`unknown`と書き、空欄や推定値で埋めない。Public Score、**OOF PQ（およびSQ/RQ）**、OOF instance Dice、semantic Dice、crop Diceを必ず別欄に記録する。

**指標を変更したときは、この文書の該当節を黙って書き換えず、改訂として明示し、旧記述を取り消し線か引用ブロックで残すこと。** 過去の判断がどの前提の上で行われたかを追えなくなると、negative resultの価値が失われる。

### 30.1 改訂履歴

| 版 | 日付 | 内容 |
|---|---|---|
| 初版 | 2026-08-15 | V19〜V34の経緯、Public最高0.35、V34のゲート不合格までを記録 |
| 改訂2 | 2026-08-15 | **リーダーボード指標がPanoptic Qualityであることを確認し、§2.1として追加。** §7・§8.1・§9.1・§13.2・§19・§25・§26・§31・§21を改訂。アノテータ不一致によるPQ天井を実測（§2.1.5）。Kaggle外の最終提出要件を追記（§19.0）。GPU不要の診断をローカルPCで回す方針を追加（§12.2.1） |
| 改訂3 | 2026-08-16 | fold1/2 refiner学習、全fold重畳検証。確定構成 PQ 0.4426、ゲート 0.4526 |
| 改訂4 | 2026-08-16 | V35最終提出 Public **0.37**。P2-2（imgsz 1536 fold0 探針）を次の実験として追加 |
| 改訂5 | 2026-08-16 | P2-2 不合格（fold0 PQ 0.4295）。提出正本は V35。次は P2-3（推論 imgsz のみ） |
| 改訂6 | 2026-08-16 | P2-3 不合格（fold0 PQ 0.4320）。解像度レバー停止。提出正本は V35 |
| 改訂7 | 2026-08-16 | 全707観測でcase diagnostics・near-miss被覆モード分類が完走。undercoverage 45% / overcoverage 42%でほぼ拮抗、hard FNがhard FPを37%上回る。§27.1 |
| 改訂8 | 2026-08-17 | Kaggle無料GPUでV38(augmentation強化)・V39(モデルサイズ拡大)をfold0実行、共に不合格(PQ 0.4290/0.4328 < 0.4393)。生YOLOも悪化しており後処理相性では説明不可。ただしV35と同じ70epochスケジュール流用が交絡。§27.2 |
| 改訂9 | 2026-08-18 | watershed癒着分離(§27.3)・交絡補正版V40/V41(§27.4、検出器強化系6本連続不合格)・multi-seedアンサンブルV42(§27.5、FP+190>TP+53で不合格)、全経路不合格。V35は頑健な局所最適と判断 |

## 31. 用語と指標の最小定義

| 用語 | この文書での意味 |
|---|---|
| observation | 1枚の太陽観測画像。複数labelerのannotationを持つ場合がある |
| label variant | 同一observationに対する個別annotatorのinstance集合 |
| proposal／seed | YOLOなどが出した個体候補。最終提出maskとは限らない |
| semantic mask | 個体IDを持たない前景／背景mask |
| instance mask | 1行1個体として扱う排他的なmask |
| **Panoptic Quality (PQ)** | **リーダーボード指標。`Σ IoU(TP) / (TP + 0.5FP + 0.5FN)`、TPはIoU>0.5。§2.1参照** |
| **SQ** | **TPの平均IoU。形状の質。PQ = SQ × RQ** |
| **RQ** | **`TP/(TP+0.5FP+0.5FN)`。検出の質（F1相当）** |
| **near-miss** | **IoU 0.3〜0.5 の予測。検出は当たっているが崖を越えられていない状態** |
| **annotator-image** | **PQの評価単位。`<annotator_id>-<image_id>`。同一予測が各アノテータのGTと個別に突き合わされる** |
| strict instance Dice | 予測個体とGT個体を対応付けて算出するローカル指標。**公式指標ではない（改訂2で確定）。過去数値との接続用にのみ使う** |
| crop Dice | 検出済みcrop内の形状精度。未検出個体を評価しない |
| union Dice | 全個体を1枚の前景へ統合したDice。個体分離を評価しない |
| split | 1個のGTを複数予測へ分割した誤り |
| merge | 複数GTを1個の予測へ結合した誤り |
| disjoint／排他化 | 同一画像内でどの画素も複数instanceに所属しない状態 |
| OOF | 対象画像を学習に使っていないfoldモデルによる予測 |

## 32. 再現性の受入レベル

「動いた」と「再現できた」を分ける。

1. **L0 構造確認:** notebook JSON、Python構文、path組立が正常。
2. **L1 smoke:** 少数画像・短epochで学習、推論、CSV生成まで完走。
3. **L2 run再現:** 同一data、fold、weight、環境でOOF指標とCSV統計が許容差内。
4. **L3 提出再現:** CSV validator合格後、Kaggleが受理しPublic Scoreが一致または説明可能な範囲。
5. **L4 汎化確認:** private leaderboardまたは独立held-outで改善が維持。

現時点でV30b/V32はL3、V34はL2相当のnegative result、V33は申告結果に対する再現失敗をL3で確認した状態である。L4は未確認。

## 33. 実験IDとファイル命名規則

今後は同じ`best.pt`、`submission.csv`、`run`を使い回さない。

推奨実験ID:

```text
v35_YYYYMMDD_<model>_<folds>_<main-change>_<seed>
```

推奨成果物名:

```text
weights/fold{fold}_best_<experiment-id>.pt
oof/oof_raw_<experiment-id>.npz
oof/oof_metrics_<experiment-id>.csv
submissions/submission_<experiment-id>_<postprocess>.csv
reports/manifest_<experiment-id>.json
```

ファイル名へ入らない設定はmanifestに全て記録する。`latest`や`final2`のように意味が変わる名前は使わない。

## 34. Manifestに保存する必須項目

最低限、次をJSONへ保存する。

```json
{
  "experiment_id": "v35_YYYYMMDD_model_3fold_change_seed",
  "created_at_jst": "YYYY-MM-DDTHH:MM:SS+09:00",
  "git_commit": "unknown-or-sha",
  "notebook_sha256": "...",
  "builder_sha256": "...",
  "dataset_json_sha256": "...",
  "folds_sha256": "...",
  "python": "...",
  "torch": "...",
  "cuda": "...",
  "gpu": "...",
  "packages_file": "pip-freeze.txt",
  "random_seeds": {},
  "training_config": {},
  "checkpoint_sha256": {},
  "metric_definition_version": "...",
  "oof_metrics": {},
  "test_statistics": {},
  "submission_sha256": "...",
  "kaggle_public_score": "unknown",
  "decision": "promote-or-reject",
  "decision_reason": "..."
}
```

OOF metricだけでなく、そのmetricを計算したコードのversion/hashを残す。同じ名前の指標でも実装差があると比較不能になる。

## 35. Colab障害復旧runbook

### 35.1 表示だけが止まって見える場合

1. セル左端が停止四角か再生三角か確認する。
2. 下部の経過時間、GPU/RAM使用量、Driveの更新時刻を確認する。
3. 右側「実行数」の古い`0%`表示だけを根拠に停止と判断しない。
4. 同じセルを重ねて実行しない。

### 35.2 interruptが効かない場合

1. 5分以上進捗、GPU使用、Drive更新がなく、同じカウンタなら停止候補とする。
2. まず実行中断を1回だけ押す。
3. 反応しない場合はruntimeを再起動する。再起動前にDrive成果物の存在と更新時刻を確認する。
4. 再起動後は`Run all`を無条件に押さず、設定、data、model定義、artifact復元、対象の軽量救済セルの順で実行する。

### 35.3 再開判定

- `last.pt`だけでなくoptimizer、scheduler、epoch、best scoreが保存されているか確認する。
- fold完了markerがあり、best weightがdecode/loadできるfoldは再学習しない。
- cacheは件数、画像名集合、fold、shape、hashを検査してから再利用する。
- 途中生成CSVは候補設定と処理済み画像数を持たせ、単純追記でduplicateを作らない。

## 36. V35の停止条件と昇格条件

実験を長引かせないため、事前に判断を固定する。

### 即時停止

- grouped split違反が1件でもある。
- OOF predictionが学習foldのweightで生成されている。
- mask重複、null、空RLE、duplicate IDを自動修復だけで黙って通している。
- pathological imageが設定上限を超え、結果が切り捨てられたまま評価される。

### 継続検討

- all-fold strict DiceがV31基準を超える。
- count error、split、mergeのうち少なくとも2項目が改善し、各foldで破綻しない。
- worst-case画像で改善理由を視覚的に説明できる。

### Kaggle提出へ昇格

- 13.2のacceptance gateを満たす。
- 完全なmanifest、validator report、CSV hashがある。
- 同一日の提出候補間に明確な仮説差があり、同値後処理の連投ではない。

## 37. 引継ぎ時に確認する質問

次の担当者は作業開始前に以下を`yes / no / unknown`で更新する。

1. Kaggle competition rulesとdata licenseを最新状態で確認したか。**→ 一部yes。Evaluation/Leaderboard/Submission/Prizesは2026-08-15に確認済み（§2.1、§19.0）。Rulesタブ本文は未確認。**
2. final submission選択数と締切を確認したか。**→ yes。Kaggle締切とは別に、2026-11-15までにGoogle Formで技術報告書＋公開Gitリポジトリの提出が必須（§19.0）。**
3. V30の3-fold weight、fold CSV、refiner weightを二重保存したか。**→ 未確認。`G:\マイドライブ`にあるが単一障害点のまま（§20）。**
4. 使用するmetricが公式metricと一致するか、差を説明できるか。**→ yes。公式はPanoptic Quality。`src/metrics_official.py`が公式実装との差分テスト付きで一致を保証している（§2.1）。過去のHungarian Diceは不一致だった。**
5. V35の主仮説を1つに絞ったか。**→ SQ/RQ分解の実測待ち（§2.1.4）。**
6. OOF改善前にtest推論やPublic閾値調整をしていないか。
7. Kaggle/Colab/HFのsecretが成果物へ混入していないか。
8. 失敗runもnegative resultとして条件と理由を保存したか。
9. **【改訂2で追加】技術報告書とリポジトリ整備に着手したか。定性評価が全体の30%を占める（§19.0）。**

