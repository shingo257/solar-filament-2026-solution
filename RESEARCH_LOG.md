# Research Log — Solar Filament Instance Segmentation

コンペの結果に関わらず、独立した技術ノート/プレプリント(arXiv等)として
成立させることを見据えた実験ログ。README.mdの「現在地」節が正本(最新の
数値・結論)であり、本ドキュメントはそれを論文の構成(仮説→方法→結果→
結論)に整理し直したもの。**数値の重複は避け、README.mdの該当節番号を
参照する形式にする。**

想定する核となる主張(2026-08-29改訂、監査結果を反映):

> 707観測という小規模データセットでの太陽フィラメントinstance
> segmentationにおいて、検出器アーキテクチャの差し替え・アンサンブル・
> refinerの再学習・refinerの後処理設計変更という4系統のレバーを試みたが、
> **全fold OOFで確定した改善は一つも得られなかった。** 特に後処理設計
> 変更(shrink→free variant)は当初fold0限定でΔ-0.0032まで肉薄する
> 「最も有望な非ベースライン結果」に見えたが、fold1・fold2への独立検証で
> Δ-0.0157・Δ-0.0106と大きく悪化することが判明し、単一fold探索への
> 選択バイアスだったと訂正した。**同種の選択バイアス(WBFのdrop_contained
> 閾値探索)が本研究で2度独立に発生・検出されたことは、小規模データセット
> での後処理閾値チューニング一般に対する方法論的警告として、それ自体が
> 独立した知見になりうる。** モデル容量を増やす系統のレバーが一貫して
> 効かなかったという結論は監査後も変わらないが、「唯一有効なレバーが
> あった」という当初の主張は撤回し、**707観測というデータ規模そのものが
> モデル側・後処理側どちらのレバーに対しても局所最適に近い可能性が高い**
> という、より保守的で防御可能な主張に改める。データ側のレバー
> (pseudo-labeling)はfold0パイロットで一度不採用と判定したが、
> ペア化ブートストラップ監査でΔが統計的に0と区別できないと判明し、
> 「fold1確認待ちの保留」に訂正した。その後fold1でも独立に同じ符号
> (Δ-0.0058、fold0のΔ-0.0061とほぼ同じ大きさ)が確認され、2fold統合
> (n=472)でリサンプルの94.95%がΔ<0となったことから、pseudo-labelingは
> 不採用と最終確定した(§2.4・§2.5)。**この「全fold OOFで確定した改善は
> 一つも得られなかった」という結論は、2026-08-30に初めて反例が生まれ更新
> された。** refinerのボトルネックへEdge-Guided Multi-Head Self-Attention
> (EdgeAttNet、arXiv:2509.02964の再実装)を追加したところ、3-fold OOF
> 全結合(n=707)でΔ+0.0028(95%CI[0.000915,0.004756]、ゼロを完全に除外)、
> 実際のPublic LBでも0.37→0.39と改善が再現した(§2.6)。**モデル容量を
> 増やすレバー一般が効かないのではなく、境界情報を明示的に注入する
> attentionという特定の帰納バイアスが707観測という小規模データでも
> 有効だった**、という、より精緻化された主張に改める。

## 1. 関連研究との位置づけ

| 文献 | 手法 | 指標 | 我々との関係 |
|---|---|---|---|
| Flat U-Net (arXiv 2502.07259, 2025) | SCA+CSA軽量U-Net | DSC 0.82 (semantic) | アーキテクチャ提案系。我々はrefinerとしてplain U-Net(MediSeg)を使用していたが、2026-08-30にEdge-Guided Multi-Head Self-Attention(EdgeAttNet系)をボトルネックへ追加し3-fold OOF+Public LBの両方で改善を確認した(§2.6)。attention機構は「未検証」から「検証済み・有効」に更新 |
| AA-UNet (IEEE, 10046547) | Axial attention U-Net | Jaccard 0.63 (semantic) | 同上 |
| MORDEN / AHAS (arXiv 2607.24525, 2026) | DenseCRF+DBSCAN後処理、pseudo-labeling | IoU 0.732 (semantic) | **pseudo-labelingの直接的先行事例。** 異なる観測機器(CHASE)695枚を自動注釈化、実GTでのIoU改善+0.003〜+0.006。我々は同一機器(GONG)の別日付を使いドメインギャップがより小さいという仮説で検証したが、fold0 OOF PQはΔ-0.0061と悪化し不採用(README §2.1.1)。同一機器でも「効いても小幅」ではなく悪化しうることを示した点で、この先行事例の楽観的な読み方に対する反証になる |
| EdgeAttNet (arXiv 2509.02964, 2025) | U-Net + Edge-Guided Multi-Head Self-Attention(EG-MHSA、境界priorをQ/Kへ加算) | MAGFiLO mIoU 0.57→0.65〜0.70 (semantic) | **本研究で唯一「実装→3-fold OOF確認→Public LB確認」まで完走した外部手法の移植。** 公開GitHub実装(`dasjar/EdgeAttNet`)は該当クラス定義が`main.py`の複製になっており実質存在しないため、論文本文の記述のみからEG-MHSAを独立再実装した(README §2.1.5)。3-fold OOF全結合(n=707)でΔ+0.0028(95%CI[0.000915,0.004756]、frac_positive 99.8%)、Public LBでも0.37→0.39で改善が再現 |

**注意: 上記はすべてsemantic (pixel-level) IoU/Dice指標であり、我々が
使うPQ(instance-level、IoU>0.5の崖、annotator×image単位のmicro集計)
とは定義が異なる。論文化する際は指標の違いを明示し、直接比較しないこと。**
(プロジェクト内の一貫した注意点。README.md §1参照)

BigDataCup「Solar Filament Segmentation」自体は2026年が初開催で、過去大会・
優勝手法は存在しない(2026-08-29確認)。

## 2. 実験カタログ(README.md §2の要約、詳細はそちらを参照)

### 2.1 不採用となったモデル側レバー(負の結果、それ自体に価値がある)

| # | 施策 | 結果 | README参照 |
|---|---|---|---|
| 1 | YOLO検出器強化(imgsz/augmentation/モデルサイズ、V36-V41) | 6本連続不合格 | §2 |
| 2 | multi-seedアンサンブル単純連結(V42) | 不合格(Δ-0.0078) | §2 |
| 3 | RF-DETR-Seg(アーキテクチャ差し替え、V44/V45) | 解像度交絡で部分的に説明できるが基準未達 | §2 |
| 4 | YOLO×RF-DETR WBFアンサンブル | 不合格(Δ-0.047) | §2 |
| 5 | refiner再学習(V46 warm start / V47 scratch) | 試すたびに悪化、crop_diceとPQの乖離を発見 | §2 |
| 6 | refinerのfree variant(後処理設計変更) | 【2026-08-29監査で訂正】fold0限定でΔ-0.0032、fold1/2独立検証でΔ-0.0157/-0.0106と判明、不採用 | §2.0.1 |

### 2.2 単一fold選択バイアスが2度独立に発生した事例(方法論的知見)

本研究で「fold0(または一部fold)限定では基準を上回るように見えたが、
全fold検証で覆った」事例が2度発生した:

1. **WBF `drop_contained@0.80`**(2026-08-22): fold0でΔ+0.0007。
   全fold検証でfold1Δ-0.0027・fold2Δ-0.0062と判明、不採用
2. **refinerのfree variant閾値0.86**(2026-08-28確立、2026-08-29監査):
   fold0でΔ-0.0032(基準未達だが最も近かった)。fold1・fold2への独立
   検証でΔ-0.0157・Δ-0.0106と判明、near-miss境界refiner路線全体を
   不採用へ訂正

いずれも「0.01刻みの精密探索」「複数閾値の系統的スイープ」という、
一見厳密に見える手続きの中で発生している点が重要である。**閾値探索の
精密さは選択バイアスへの防御にならない。唯一の防御は独立foldでの
検証である。** この2事例は、小規模データセット(707観測、3-fold)での
後処理閾値チューニング一般に対する再現可能な警告として提示できる。

### 2.3 確定している唯一の改善

V35の`shrink@0.45` + `drop_contained_then_area_asc`（全fold OOF PQ
0.4426、生YOLOベースライン比+0.0489）が、本研究で全fold検証を通過した
唯一の確定改善である(README §2)。

### 2.4 保留・追加検証待ち: pseudo-labeling(データ側レバー、2026-08-29)

README §2.1/§2.1.1/§2.1.2参照。2018-2022年の年代偏り是正を仮説に、GONG
追加150枚から3-fold合意+conf0.5フィルタで72個体(35画像)を採用し、
fold0のtrain splitにのみ追加して再学習(val=リークなしの実観測236枚)。
結果はfold0 OOF PQ 0.4393→0.4332(Δ-0.0061)で、当初は不採用と判定した。

**その後の監査で判定を訂正。** ユーザーから「軒並み不採用は評価手法
自体の問題では」と指摘を受け、(1)fold構成の偏り、(2)fold別refiner
品質、(3)raw→refined天井効果、(4)ペア化観測クラスタ・ブートストラップ
の4方向で監査した。(1)(2)(3)はいずれも「fold0が特別壊れている」という
仮説を支持しなかったが、(4)でΔ-0.0061のペア化95%CIが**[-0.0162,
+0.0041]と0を跨ぐ**ことが判明し、**fold0単独では統計的に有意な悪化
とは言えない**ことが確認された(2000回リサンプル中88.7%はΔ<0で
弱い悪化寄りの傾向はあるが、確定には至らない)。

free variant/WBFの不採用判定はfold1・fold2という**独立した2foldが
同じ符号で一致**していたのに対し、V49はまだfold0の1点のみであり、
同じ基準に照らすと「不採用確定」と呼ぶのは時期尚早だったと訂正する。
**fold1でも同様の拡張学習を行った結果(2026-08-29):** Δ-0.0058
(fold0のΔ-0.0061とほぼ同じ大きさ、同じ符号)。2fold統合(n=472)のペア化
ブートストラップは95%CI [-0.0133, +0.0011]でリサンプルの94.95%がΔ<0
(有意水準の境界に近い)。free variant/WBFと同じ「独立2fold一致」という
証拠の形が揃ったため、**pseudo-labelingは不採用と最終確定する。**
監査結果一式は
`solar_filament_v35/artifacts/v49_evaluation_methodology_audit.json`。

### 2.5 評価ノイズ床の定量化(方法論的知見、2026-08-29)

「軒並み不採用という結果パターン自体が評価手法の欠陥を示していないか」
というユーザーからの指摘を受け、単一fold OOF評価の統計的検出力を
初めて定量化した。並行するScientificトラックが実装していた観測
クラスタ・ブートストラップ手法(画像単位でリサンプルし比率を再計算、
2000回)を competition track の公式PQへ適用した。

**主要な結果:**
- 707観測(全fold)でのPQ95%CI幅は約0.020(並行トラックの既存測定、
  0.4403 [0.4296, 0.4500])
- fold0単独(236観測)でのCI幅は実測0.0352。√(707/236)スケーリング則
  による予測値0.0353とほぼ完全に一致し、fold0が特別ノイズが大きい
  わけではなく、単純なサンプルサイズの効果であることを裏付けた
- V49のΔ-0.0061をペア化ブートストラップで検定したところ、95%CI
  [-0.0162, +0.0041]は0を跨ぎ、統計的に有意ではないと判明

**意味すること:** 本研究がこれまで下してきた「不採用」判定の多くは、
単一fold(236観測)の点推定同士の単純比較に基づいており、Δが
±0.01程度の施策はこの検出力では原理的に有意差を検出できない。
これは「施策が効かない」ことの証明にはならず、「この検出力では
判別できない」という限界を示すに過ぎない。ただしfree variant/WBFの
不採用は独立2fold一致という別種の証拠に支えられており、この限界の
対象外である(§2.2)。**今後の閾値・施策選定は、単一fold点推定の
大小比較ではなく、(a)独立複数foldでの符号一致、または(b)ブートストラップ
CIが0を跨がないこと、のいずれかを採用基準とすべき。** この教訓自体が、
小規模データセットでの反復的な後処理・データ拡張実験に対する再利用
可能な方法論的知見である。

### 2.6 EdgeAttNet refinerの採用(初の確定的正の結果、2026-08-29〜30)

§2.5で定式化した採用基準「(a)独立複数foldでの符号一致、または(b)
ブートストラップCIが0を跨がないこと」を、初めて**肯定的な**結果に対して
適用し、実際に基準を満たす施策が見つかった事例。詳細はREADME §2.1.5。

**要点:**
- 標準`MediSegUNet`のボトルネックへEdgeAttNet論文(arXiv:2509.02964)の
  Edge-Guided Multi-Head Self-Attentionを移植した`MediSegEdgeAttUNet`
  (+4.2Mパラメータ)を実装
- fold0単独ではΔ+0.0062(frac_positive 99.5%)と非常に強いが、fold1
  (Δ+0.0009、73.9%)・fold2(Δ+0.0012、77.3%)は単独では§2.5の基準(b)
  を満たさない「不確か」な結果だった
- しかし3fold全て**同符号**であり、基準(a)は満たしていた。3-fold全結合
  (n=707)のプールされたブートストラップでは95%CI[0.000915,0.004756]と
  基準(b)も満たす有意水準に達した
- 実際にKaggleへ提出(V52)したところPublic LB 0.37→0.39と、OOFで予測
  した改善が符号・方向とも再現した

**方法論的意義:** これはfree variant/WBF/pseudo-labelingで確立してきた
「fold0だけの見せかけの改善を警戒する」という規律が、**否定判定だけで
なく肯定判定にも同じ厳密さで適用できる**ことを示した初の事例である。
fold0単独の結果(Δ+0.0062)だけを見て即座に採用していたら、事後的には
「たまたま正しかった」ことになるが、その判断プロセス自体はこれまで
繰り返し反証されてきたfold0選択バイアスと区別がつかなかった。3fold
全ての独立確認を経てから採用を決めたことで、判断の再現性・防御可能性
が担保されている。

### 2.7 OOF-Public LBギャップの分布差仮説は反証(2026-08-30)

README §2.1.6参照。V50診断が示唆した「学習/テスト間の年代・サイト分布差」
仮説を、ファイル名メタデータのカイ二乗検定で直接検証したところ、年代
(p=0.423)・サイト(p=0.938)いずれも有意差は検出されなかった。
pseudo-labelingの前提だった年代偏り仮説も同じ切り口で支持されない。
**約0.06のOOF-Public LBギャップの真因は依然として未特定**であり、
論文化する際はこの反証も含め「試した仮説のうち何が棄却されたか」を
明示するとよい(test小サンプルの評価ノイズ、画像特性差、アノテーション
基準差が残る候補)。

### 2.8 v80 ハイブリッド候補+期待IoUスコアラー(2026-10-10)

**2026-10-12 の証拠範囲:** 以下の v80/A1・v82・提出・外部再現結果は作者の既存記録。
`HANDOVER.md` 冒頭に記した過去実行 manifest の欠落と489/707件の分割相違は未解決であり、
今回追加する検証ガードは過去実行を遡及的に検証しない。
v80/A1 の scorer cross-fit は上流モデル・特徴キャッシュを outer fold ごとに作り直していない。
scorer 学習側の別foldキャッシュの生成モデルが評価outer foldを学習済みのため、
報告 OOF 値・CI・「ゲート通過」は探索結果に限定し、不偏な独立OOF評価や採用許可には使わない。
採用判断には、凍結分割の同一性・全キャッシュの網羅性と、outer fold を除外した上流学習を含む nested 検証が必要。

PLAN_2026-10-09_FINAL_SPRINT.md §7.1.8b参照。YOLO候補に V34 OOF 密確率
マップの component-tree 候補を加え、cross-fit HistGBM の期待IoU û>τ で
選択する E1 実験(707観測・149,693候補、全fold 公式PQ)。

**結果: 不採用**(hybrid−prod = −0.0142、CI95 [−0.0194,−0.0089])。
ただし負の結果ではなく分解可能な失敗:

- `hybrid_oracle` = +0.1067(TP +1454、FP −1063)→ **候補集合に大きな
  伸びしろが実在**。too.foo氏の selection oracle +0.083 と同符号・同桁で
  独立に再現
- hybrid は TP +106・FN −106(tree で新規検出できている)が FP +537 で
  正味負 — 追加検出は成功、FP 抑制が失敗
- スコアラー診断: ROC-AUC(h>0) tree 0.900、Spearman ρ(û,u) 0.676 —
  **û の順位付け自体は良好**。失敗は絶対校正(中位候補の û 過大評価が
  antichain DP の累積選択で断片化を誘発)

**方法論的意義:** 「候補の質(oracle余力)」「スコアラーの順位付け(AUC/ρ)」
「選択機構(antichain DP + 排他化)」の3層を分離評価する設計により、
v64/v59 系の「GBM selector が効かない」結論が「候補源・決定層の問題で
あって scorer の識別力の問題ではない」と精緻化できた。v62/v66 の
「提案不在 FN」洞察と oracle 余力 +0.107 が整合する。

## 3. 方法論として再利用可能な要素

- 公式PQのローカル再実装(`solar_filament/metrics/metrics_official.py`)、
  主催者ノートブックとの差分テスト付き(19件)
- リークなしOOF評価の設計(fold境界を跨がないrefiner学習データ構築、
  `build_refiner_free_retrain_manifest.py`)
- 「単一foldのbestを一般化しない」という繰り返し実証された落とし穴とその対策
  (0.01刻み精密探索→フル観測での再検証、drop_contained閾値・free variant
  閾値という異なる対象で2度独立に発生したfold0選択バイアス事例、§2.2)
- 「過去の結論が実際の実験結果を正しく評価できているか」を定期的に監査する
  習慣そのものが方法論的価値を持つ(free variantの訂正は監査で発見された)

### 2.9 v80-follow-up: A1 の test 配置可否(2026-10-10、prob_* ablation)

v80 でゲート通過した `yolo_scorer`(+0.0039)の提出配置には、scorer 33特徴のうち
`prob_*` 4個(V34 semantic OOF マップ由来)が test 画像に存在しない問題があった。
`--drop-features` で4列を除いて全 arm を再評価(`output/v80_noprobdense/`):

- **`yolo_scorer`(29特徴): PQ 0.4460、Δ +0.0034、fold別 0.4416/0.4464/0.4501
  (全fold prod 上回り)、bootstrap CI95 [+0.00045, +0.00642]** —
  **ゲート両条件を prob_* なしで維持 → 案(a) 採用**
- hybrid −0.0158 / hybrid_split 最良 0.4449 / oracle +0.1067(不変) —
  tree 追加は特徴を減らしても不利益という構造的結論が再確認された
- 提出物は YOLO+refiner+sklearn の3モデル構成のまま(V34 不要)。
  `train_a1_scorer.py` が全 OOF 行で h/m HistGBM を学習済み
  (`artifacts_scorer/a1_scorer_{h,m}.pkl` + meta.json、Drive artifacts 配置済)

### 2.10 公式採点 ↔ ローカル評価の対応表 (ずれ検知、2026-10-10)

ファイル提出制のため、提出は計算枠を消費しない。過去提出と今後の予定を
OOF↔LB の対応で追跡する (Public LB は test の約50%・2桁表示のため
<0.01 の差は識別不能、提出間比較は OOF が正本):

| 提出 | 構成 | OOF 参照値 | Public LB | 備考 |
|---|---|---|---|---|
| V32a/b, V33 | 初期 union/disjoint 系 | — | 0.33–0.35 | 前期 |
| V35 | YOLO 3fold ensemble + 標準 refiner + shrink@0.45 | 0.4426 (全fold) | 0.37 | 2026-08-16 |
| V50 | fold2 単一 + 標準 refiner | 0.4471 (fold2) | 0.37 | ensemble 効果なしを確認 |
| V35 再提出 | V35 同一 | 0.4426 | 0.37 | 同日比較用 |
| V52 | fold2 + EdgeAtt all-data refiner | 0.4483 (fold2) | **0.39** | 現行ベスト |
| V53 | 3fold ensemble + EdgeAtt | ~0.44 | 0.38 | ensemble は LB で悪化 |
| prod 再現 (57039986) | V52 と同一構成のローカル再生成 | — | **0.39** | 2026-10-10。V52 CSV との照合: IoU>0.5 一致 1171/1171 (100%)、mean best-IoU 0.9986。公式採点でも V52=0.39 を厳密再現 → ローカル推論経路が公式レベルで検証済み |
| A1 (57047601) | prod の conf0.26 を scorer 選択 û>0.22 に置換 | 0.4460 (全fold, +0.0034) | **0.37** | 2026-10-10。1174行・prod との IoU>0.5 一致率 91.4%。OOF +0.0034 が LB では −0.02 に反転 |

ずれの既知の大きさ: fold2 OOF 0.4483 → LB 0.39 (Δ≈0.058、真因未特定 —
README §2.1.4)。LB が ~90枚の半分サンプルなので、OOF ±0.004 規模の差は
LB では表示分解能以下。

**A1 の OOF→LB 反転 (2026-10-10 確定)**: OOF で +0.0034・CI ゼロ除外の
scorer 選択が Public LB で −0.02。仮説: (a) ~90枚サンプルでは数個の
filament 差が 0.02 を生む (統計的に識別不能なノイズ内); (b) scorer が
OOF 上では FP 削減、test 分布では TP も間引いた (a1 は prod 採用分の
101本を欠落・103本を新規採用); (c) test 画像の難易度分布差。**結論:
0.003 規模の OOF 改善は LB で確認できない — 以後の採否は「OOF で大きな
効果」(>0.01) または機構的に堅い変更に限定すべき**。prod=0.39 が
ローカル再現の公式ベースラインとして確定。

### 2.11 v82 soft annotator-mean 目標 — 棄却確定 (2026-10-11)

**結果 (soft arm 6本完了、manifest セルの `{{` バグはローカル再構築で回収)**:

| fold | s3407 | s8407 | mean Δ @τ0.45 |
|---|---|---|---|
| 0 | −0.0003 | −0.0039 | −0.0027 |
| 1 | −0.0058 | +0.0015 | −0.0034 |
| 2 | −0.0054 | −0.0062 | −0.0060 |

- 全閾値掃引でも mean Δ は全て負 (−0.0032〜−0.0040)。唯一の正セル
  fold1 s8407 +0.0015 は LB 識別能以下 (§2.10 の実測参照)
- **採用ゲート (3fold・両seed で正 + CI ゼロ除外): 明確に不合格**
- hard_union 対照 arm は未実行 — ゲートが soft で既に決定的に失敗するため
  判定に不要 (~9h GPU を節約)。制御としての価値はあるが優先度外
- 解釈: 「アノテータ平均 = PQ が報酬する中心」という仮説は、少なくとも
  この crop/損失構成では検出品質を改善しない。hard record 目標の
  鋭さが重要だった可能性

v82 は16番目の棄却レバーとして記録。全定量レバーの状況:
OOF 採用ゲート通過は A1 scorer (+0.0034) のみだったが LB で不採用
→ **定量改善レバーは全て枯渇確定、提出物作業に専念**。

### 2.12 Colab GPU 再現検証 — 提出パイプライン確定 (2026-10-11)

`solar_filament_prod_repro_colab.ipynb` (3セル構成、Drive `repro_code/` +
`repro_code.zip` フォールバック) を Colab A100 で実行:

| 指標 | 結果 |
|---|---|
| submission_prod.csv vs ローカル prod | **1172/1172 filament が IoU>0.5 で一致・99.91% は IoU=1.0** |
| 実行時間 | 223s 総計 (1.24 s/img, device=cuda) — ローカル CPU (19s/img) の 15 倍 |
| 出力 | `artifacts/repro_prod_colab/` に submission_prod.csv + audit + efficiency_prod.json |

**結論**: 別環境(GPU/Colab)でも事実上同一の提出 CSV を生成 — 最終提出
パイプラインの出力一致が作者記録で報告された。過去実行 manifest 未確認のため、
今回の統合による決定性・重み整合性の独立受入とは区別する。
`efficiency_prod.json` は最終提出物の efficiency 記録として使用。

### 2.13 Kaggle kernel 再現 push (2026-10-11)

`output/kaggle_kernel/` を `kaggle kernels push` で実行投入:
`shingo257/solar-filament-2026-reproduction` (private, GPU, 競技データ +
`shingo257/solar-filament-2026-weights` dataset 添付)。

- 添付 all-data refiner 重みが解決できる場合、fold別 refiner 学習を
  スキップするよう notebook を修正(提出経路では fold別 refiner は未使用、
  GPU 枠 ~3-6h を節約)
- 競技提出要件の「公開可能な再現 kernel」資産として位置づけ。

**実行結果 (COMPLETE、同日)**: `kaggle kernels output` で成果物を検証 —

| 検証 | 結果 |
|---|---|
| `submission_master_pipeline.csv` vs ローカル prod | **1172/1172 filament・IoU=1.0 で完全一致**(Colab GPU の 99.91% を上回る完全一致) |
| 推論速度 | Tesla T4・1.075 s/img・総計 193.6s |
| 重み sha256 | yolo_fold2 / refiner_all_data ともに dataset・Drive 実物と一致 |
| 学習スキップ | 添付重み解決で YOLO3fold・all-data refiner とも再学習なし |
| 環境記録 | `submission_reproduction_pip_freeze.txt` 出力済み |

3環境(Windows CPU / Colab A100 / Kaggle T4)全てで同一提出CSVを
再現できたという作者報告。これは当該3環境の出力比較記録であり、
環境非依存性一般や、今回のコード世代の独立再現を実証するものではない。

### 2.14 plain-MHSA refiner 提出 — 0.39 (2026-10-11)

matched-parameter ablation の知見(edge prior 固有効果は未確認)を
LB 上で直接検証するため `ablation_plain_mhsa_seed8407_fold2.pt`
(fold2 OOF 0.44630、ablation 最良 plain cell)を prod 構成の refiner
として差し替えた variant を生成・提出(`--refiner plain_mhsa_f2`)。

| 指標 | 結果 |
|---|---|
| public LB | **0.39**(prod と同点) |
| prod との差分 | 1170/1173 IoU>0.5 一致・mean best-IoU 0.9382 — 境界は有意に異なる |
| 解釈 | edge prior なしでも public 同等 = 「attention 容量一般が効く」
説と整合。private 側(~90枚)では境界構成差により異なる得点を
取り得るため、prod と並ぶ第2の 0.39 構成として価値あり |

学習コスト 0(既存 ablation 重み流用)・推論 ~33分・日次枠は残4。

### 2.15 3refiner 同一 0.39 — LB 分解能と最終選択 (2026-10-11)

`edgeatt_f2` (ablation 単一セル最良・fold2専用 refiner) も **0.39**。
3構成が public 同点に:

| variant | refiner | fold2 OOF | public | prod差分 mean-IoU |
|---|---|---|---|---|
| prod | edge-att **all-data** | (提出系) | **0.39** | — |
| edgeatt_f2 | edge-att fold2 のみ | 0.44749 | **0.39** | 0.9274 |
| plain_mhsa_f2 | MHSA・edge prior なし | 0.44630 | **0.39** | 0.9382 |

**分析**:
- 境界構成は 3 者とも有意に異なるのに public 同点 → ±0.003 規模の
  OOF 差は ~90枚・2桁表示の public では**原理的に識別不能**(A1 と同じ教訓)
- all-data fine-tune も edge prior も public 得点には必要十分でない
- **最終選択(2枠)の推奨**: `prod` + `plain_mhsa_f2`。
  edgeatt_f2 は prod と refiner 系統が同じ(差は学習データのみ)のに対し、
  plain_mhsa は**アーキテクチャ差**があるため private 側での
  分散ヘッジ効果が最大。最終提出選択時はこの2つを選ぶこと。
- 残り定量候補なし。今後の枠は「新候補が出た時」用に温存が妥当

### 2.16 外部手法・discussion 調査 (2026-10-11)

上位陣手法・公開 kernel・competition discussion (Kaggle API で全20スレッド
本文取得) を調査したという作者記録。**0.55–0.63 帯の漏洩説は参加者発言からの仮説であり、
各提出のデータ・分割・来歴は独立未確認。スコア帯だけで個別チームの違反や誠実性を判定しない**。
~0.36–0.45 は作者が比較対象に選んだ公開報告の帯域で、検証済みの実力上限ではない。

#### LB の見取り図 (複数独立ソースで一致)

| 帯域 | 解釈 | 根拠 |
|---|---|---|
| 0.55–0.63 | 漏洩説あり・個別提出は未確認 | 作者が記録した参加者発言: discussion 737177 は公開 test GT と想定スコア帯を解析、743958 は特定 kernel の使用データを疑問視、741892 は高スコアへの疑念を表明。いずれも今回の独立検証・主催者の個別裁定とは区別する |
| ~0.36–0.45 | 誠実系フロンティア | too.foo スタック LB 0.36 / Fususu pipeline 再採点 ~0.40 / hetspookjee 誠実 0.39 / Spaghetto99「train_images のみで 0.40」/ noisyislands 0.49 主張(未検証) |
| >0.35 | 主催者の審査ライン | Azim 主催者「0.35 超の提出はすべて精査」「単一指標に依存しない(定量70%+定性30%)」 |

補足: 評価関数は 8/7–12 に旧 misconfigured Dice から正式 PQ へ切替・
全 LB 再採点済み (733912・Overview 告知)。byte 同一ファイルが 0.66→0.31
に変化 — **8/12 以前の kernel 記載スコア (0.59/0.70/0.9 等) は旧値で比較不可**。

#### 手法サーベイ (公開 kernel/dossier ベース・未検証は明記)

| 手法 | 構成 | 実値 |
|---|---|---|
| **Fususu (phuongncn) + tushhaaarrrr fork** | Mask R-CNN @1024² (score≥0.70/mask≥0.50) → 正重複 mask を1つの refiner task にmerge → ResNet18 U-Net 256² crop refiner + **flip-4 TTA (トポロジーガード付)** thr 0.70/anchor 0.85 → crowding 条件で detector-seed split (maxIoU 0.10) → 正重複 merge → RLE。MD5 固定・外部ラベル不使用を明示 | 旧値 0.70 → 再採点 ~0.40 (fork 表記 0.40) |
| magfilo-v9 (muhammadshoaibaltaf) | 軸平行 YOLO + **YOLO OBB (回転検出)** + crop U-Net・unique-photo 推論・IoU0.5 明示 | 未記載 |
| juanmirod final report | 5fold Mask R-CNN @1024 + 4-TTA + crop U-Net + fold 間 agreement | v42 0.34 → v45 |
| sivapnlr80 improved pipeline | **YOLO26s-OBB**・回転保持 crop・Unet++ EffB4 @1280 | 未記載 |
| rokaiyasomapti "LB 1st" | ResNet34 U-Net baseline | 旧メトリクス 1.00 (artifact) |
| **too.foo (Shivam Bhardwaj) 公開 dossier** | U-Net++ EffB3 patch + 公式 scorer 移植ハーネス + **学習済み component selector** (12特徴 GBM・明暗コントラスト首位) +0.018/+0.083 oracle | LB 0.36・val 0.423 |

#### too.foo dossier の収束知見 (我々と独立に同結論 → 信頼度高)

- **soft ラベルは逆効果** (彼らの pure soft = recall 崩壊 0.323; 我々の v82 棄却と一致)。
  有効だったのは union 目標 + annotator 一致度重み付け
- **crop refiner は PQ に寄与しない** (crop IoU 0.63→0.67 改善しても PQ +0.001。
  我々の refiner 系上限議論と一致)
- **selector が最大の oracle lane (+0.083)** — 彼らは +0.018 捕捉・LB 0.32→0.36
  に改善 (baseline が低く FP 余地が大きいため有効)。我々の A1 は OOF +0.0034
  → LB −0.02 — **baseline が既にタイトだと逆効果**。selector の首位特徴は
  「局所 chromosphere に対する暗さコントラスト」— 我々の A1 特徴(29個)には
  photometric 特徴がなく、再挑戦するならこの特徴群が第一候補
- **検出漏れ(微小・淡い filament)が真のボトルネック** — 全 variant で ~210/576
  val filament を miss。我々の detector-blind FN 分析と一致
- **val ゲインの転移限界 ~+0.02** — 我々の LB 分解能議論と一致
- per-annotator-record 採点 (test は ~2.44 record/画像・予測は各 record ごとに
  採点) → 実効天井 ~0.45–0.55

#### 規約の重要確認 (discussion・主催者回答)

| 項目 | 裁定 |
|---|---|
| SWEFil 等の外部 GT データ | **禁止** (746318) |
| 無ラベル GONG 観測の学習利用 | 可・ただし test ±1h 除外必須・推論時使用は不可 (744047/748045) |
| 近接 GONG フレームへの訓練ラベル転写 | **許可** (748045) — temporal-neighbor 拡張(我々は定量棄却済み)の規約根拠 |
| 磁気図の補助学習信号 | 可・推論時依存は不可 (743521) — 我々は棄却済み |
| grayscale→3ch 複製 / FITS 原データ | 可 (744544/738951) |
| 最終提出 | 選択 2 提出が異なる戦略なら repo は**両方を再現**できること・可能なら統合推奨 (747960)。Google Form 提出が別途必要 |

#### 取り込み判断

- **refiner-level flip-4 TTA (トポロジーガード)**: Fususu が旧値で +0.01 報告・
  我々は detector TTA のみ試行済みで refiner TTA は未検証 → 唯一の残存候補だが
  効果規模が小さく OOF パイロットでも +0.005 未満が見込まれる。**低優先**
- **selector の photometric 特徴**: A1 が LB で反転済み・baseline がタイトなので
  **見送り** (記録のみ)
- **detector-seed split (Fususu)**: 我々の FN は detector-blind が支配的で
  merge 由来ではない → **対象外**
- **OBB 回転 crop**: 新規学習が必要・提案形状が変わるだけで recall 機構として
  未証明 → **再現は見送り** (文書記録に留める)
- 結論: **現行 prod + plain_mhsa_f2 の 2 選択方針を維持**。外部手法で
  「採用に値する新規定量レバー」はなし — むしろ我々の負の結果が
  独立系で再現され、棄却判断の妥当性が裏強化された

## 4. 執筆時の注意点

- README.mdが数値の正本。本ログは論文構成への整理のみ、数値はコピーせず参照する
- 図表化する際は既存のnear-miss inspectorアーティファクト
  (`scratch/build_near_miss_viewer_data.py`)の症例を素材にできる
- コード品質・再現性(定性30%評価軸と重複)は既にrequirements.txt/pip freeze
  の運用ルールあり(README §7)

### 2.17 両提出の再現コード経路を整備・クリーン受入は未完了 (2026-10-12)

主催者回答 (discussion 747960) の「選択した両方の提出を repo が再現
できること」要件について、第2提出 `plain_mhsa_f2` の再現経路を整備:

| 資産 | 状態 |
|---|---|
| Kaggle dataset `solar-filament-2026-weights` | **v2 公開という作者報告** — `ablation_plain_mhsa_seed8407_fold2.pt` (282,089,497B, sha256 `b5f331e1…`) を追加。manifest 更新済みという記録。外部公開状態・実物同一性は今回独立未確認 |
| `run_test_submission.py --weights-dir` | 追加 — dataset ダウンロード先から yolo/refiner/scorer 重みを解決 (重みの Drive 依存を除く。test画像には Drive 相当の配置が必要)。両バリアントのローカル疎通は作者報告 (CPU) |
| `solar_filament_submission_reproduction.ipynb` | `CFG.SUBMISSION_VARIANT` を追加 — `"plain_mhsa_f2"` で `MediSegPlainMHSAUNet` + dataset 重みに差し替え、`submission_plain_mhsa_f2.csv` を出力 (builder 再生成・AST 検証済み、本実行は F3 クリーンランで検証予定) |
| `output/submission_weights_manifest.json` | Drive 側 manifest にも plain_mhsa エントリを追記 |

残検証: 再現 kernel の再 push + 実行 (prod/variant 両経路) — F3
(11/08 予定) のクリーン環境通し実行で実施する。コード経路は提出実績の
ある `run_test_submission.py` と同一クラス・同一 ckpt ロード式だが、
これはモデル・入力・コードの同一性や現コード世代の実行成功を保証しない。

### 2.18 レポート確定版への統合 + レビュー対応 (2026-10-12)

**2026-10-13 統合注記:** 以下§2.18–2.21はmainのPR #349/#351/#352から保持した作者の履歴であり、今回の独立再実行ではない。
新たなhash・CSV・metadataの同梱は過去実行manifest未確認・489/707件の分割相違を解決しない。
`a1_scorer_meta.json`の従来の「gate passed」や固定hashも履歴として保持するが、上流モデル非nestedのため独立OOF・採用許可の証拠として受け入れない。
今回の修正済みコード世代について両バリアントのクリーン受入は未完了。

`report/main.tex` を最終提出版に更新し、MiKTeX/latexmk のローカルビルドで
**本文4p・謝辞+refs p5** を確認(従来はビルド未検証だった):

- §3.2 末尾に最終選択段落を追加 — prod + plain_mhsa_f2(両者 Public 0.39、
  best-match IoU 0.94/0.93)の2提出と選択根拠(refiner アーキテクチャ差による
  private 分散ヘッジ)、T4 ~1.1 s/img end-to-end、honest-tier 文脈
  (スコア帯に関する当時の仮説は独立未確認で、今回の本文はその限界を明示)
- §3.3 に too.foo dossier との独立合致を1文追加(soft label 逆効果・
  crop refiner は PQ に寄与せず・unproposed instances がボトルネック)
- Future work 更新: plain-MHSA 再提出は実施済みのため削除、
  component-tree + learned selection(oracle +0.107 / scored −0.014)を残す
- 4p 収束のため strengths/weaknesses/sweep 段落を事実保持で圧縮、
  Fig.2 を 0.74\linewidth に縮小。編集由来の overfull hbox は解消
- `main.pdf` を再ビルドして同梱、LaTeX 副産物を .gitignore へ、
  REPORT_NOTES の残タスクを更新

**Codex review (PR #346 ready 化で発火) への対応 — 4件すべて実在を確認・修正:**

| 指摘 | 判定 | 対応 |
|---|---|---|
| `build_v82` の `if y0 > height` typo (train+eval 2箇所) | 実在 — v82 限定(prod は正しい) | 修正 + 12本の生成 ipynb 再生成(diff=typo行のみ) |
| 再現 nb の variant 実行が `submission_master_pipeline.csv` を上書き | 実在 | artifact 名を variant 別に分離(builder+nb+kernel コピー) |
| `run_test_submission` の `N_ANNOTATORS_IMPUTE=1.0` が meta 宣言(≈2)と不一致 | 実在 | 2.0 に修正(A1 は棄却済みだが整合性として) |
| `compare_submissions` の `images_full_overlap` が過少計数 | 実在 | 両方向 per-instance 一致 + 件数一致に修正 |

さらに `output/test_submission/`(4 variant の audit+efficiency+submission CSV)、
`output/kernel_run/`(再現 kernel の pip freeze・sha256・生成CSV)、
`output/v80*/v82*/`(棄却実験の eval 結果)、Kaggle dataset metadata/manifest、
v80 候補収集 nb 3本 — 計46件の実験証跡をコミット(モジュール規約:
結果 JSON/CSV/txt は証跡としてコミット対象)。

残検証(F3 クリーンラン予定と同じ): 再現 nb の両 variant 完走。
`run_test_submission.py --weights-dir` は両 variant で疎通済み。

### 2.19 discussion/LB 再調査 — リフレッシュ差分 (2026-10-12)

§2.16 から1日後の再取得 (kagglesdk `DiscussionApi.ListTopics` を
category=COMPETITIONS + 複数検索語 union で全網羅、LB は
`competitions leaderboard --download` の全983チーム CSV を解析)。

#### Discussion: 新着ゼロ

- 全トピックの `last_comment_date` ≤ 2026-10-10 22:40 — **調査以降の新規
  スレッド・コメントはなし**。情報は §2.16 時点で確定済み
- 追加取得した本文の注目点:
  - 745880 "realistic scores": Spaghetto99 (LB 0.41) が Fususu の公開
    kernel「0.70」を実際に再現すると **0.35** にしかならないと報告 —
    §2.16 の「再採点 ~0.40」推定を下回る第三者複製データ点。
    同氏は誠実レンジを 0.35–0.50・noisyislands 0.49 を誠実系トップと推定
  - Trishanth Mellimi (team "son", LB 0.40): local PQ 0.46 vs LB 0.40、
    Spaghetto99 も local 0.4576 vs LB 0.41 — **val→LB 転移差 ~0.04-0.06 が
    複数チームで一致** (我々の OOF 0.44 → LB 0.39 と同じパターン)

#### LB スナップショット (983 チーム・自チーム ERS rank 136 / 0.39 / 16 subs)

| 帯域 | チーム数 | 解釈 |
|---|---|---|
| ≥0.55 | **78** (うち **ちょうど 0.55 が65チーム**) | 作者が記録した同点プラトー。使用データ・分割・来歴は独立未確認で、違反や同一手法の証拠ではない |
| 0.45–0.54 | 3 | Cac Cat 0.54・Thariq J 0.53・noisyislands 0.49 (未検証) — 想定した「laundering 帯」はほぼ空 |
| 0.40–0.44 | 31 | 作者が参照した0.41の6チームと0.40帯。各提出の来歴は独立未確認で、誠実性の分類や性能上限を示さない |
| 0.35–0.39 | 332 | 我々の 0.39 は 53チーム同点の帯域先頭群 |

→ **0.55に65チームという作者のスナップショット**は保持する。
  小数2桁の同点分布だけでは使用データや手法を識別できず、漏洩・違反や
  誠実系の性能上限を結論できない。

#### 新着公開 kernel (10-11 以降 + 境界の10-09/04 分)

| kernel | 手法 | 評価 |
|---|---|---|
| jamesmills1975 cage-filament-unet (10-11) | smp UNet resnet34-imagenet @1024 + hflip TTA + PQ調整postproc | 自己申告 PQ 0.33 — 誠実 baseline |
| hodohabalotchikp notebookc66249a467 (10-12) | TF/Keras attention U-Net・multi-seed・OOF で postproc tuning・1024/2048 解像度比較 | 誠実系・我々と同型の OOF 手法 |
| pandadev0069 solar-filament-yolo-refine (10-04) | **YOLO26 検出 + ResNet18 CropUNet refiner** | 我々の prod パイプラインと同構成の独立実装 (「test GT 非アクセス」明記) — 3例目の収束証拠 |
| topabaem filament-v02-* (10-09) | base64 bundle + 入力 digest 検証の決定的 replay ハーネス | 手法でなく再現性ハーネス — 我々の証跡運用と同思想 |
| b'bang1850 solar-filament-sovereign-sota (10-04) | 403 private — 内容不明 | — |

#### 結論

**prod + plain_mhsa_f2 の最終選択は不変**。新規採用レバーなし。
外部報告には同点分布、Fususu複製0.35、val→LB差0.04–0.06という記述がある。
各報告の条件・来歴は今回独立未確認で、他者の違反や誠実性・性能上限の
論証には用いない。main.texにもこの限界を保持する。

### 2.20 両提出のCLI再現に関する作者記録 + notebook 修正 (2026-10-13)

#### ローカル CPU 経路 — 両 variant byte-identical

`scratch/run_test_submission.py` で test 180枚全件を実行し、
コミット済み証跡 (`output/test_submission/`) と SHA256 比較:

| variant | 出力 | sha256 | 一致 |
|---|---|---|---|
| prod | `submission_prod.csv` (1172行, ~31min, 10.5s/img) | `84f1b89466f988d92b6a7d14e6ff26c665a78e98b350cf441109cd05faee241e` | byte-identical |
| plain_mhsa_f2 | `submission_prod_plain_mhsa_f2.csv` (1173行, 1525.5s, 8.475s/img) | `7b3de4e78a543cf29bba3e66af8ce9b2d9d0d8fd2ea085aef3120aad97e0bbba` | byte-identical |

重み3点 (yolo_fold2, refiner_all_data, ablation_plain_mhsa_fold2) は
dataset から DL し manifest SHA256 と全件一致を確認済み。

#### notebook 修正 (PR #350, squash `102ca5ba`)

Kaggle kernel v2 が `GROUPED_FOLDS_SHA256=""` で起動時エラー。
`build_submission_reproduction.py` を修正し正本+kernel コピーを再生成:

- 凍結 SHA256 `24367120…c3b` を `SETUP_CELL` に同梱
- `FOLD_CSV_STAGING`: `grouped_folds.csv` を添付 dataset (Kaggle) /
  Drive artifacts (Colab) から自動解決 — **SHA256 一致候補のみ採用**、
  不一致なら書き込みせず FileNotFoundError (Codex 指摘で
  不一致フォールバックを除去、永続領域への誤 CSV 残留を防止)
- `test_submission_reproduction_split.py` に staging 経路テスト2件追加
  (hash一致優先・不一致拒否) — 9/9 pass

#### 重み dataset 版履歴

- v3 (10-13): `grouped_folds.csv` 同梱 (16,983B, SHA256 検証済)。
  fold0/1 重みが欠落し Kaggle 側で再学習が走る状態だった
- v4 (10-13): fold0/1 復帰で完全同梱に復帰 (v2 + fold CSV)

#### 残存

- notebook 完全版の Kaggle/Colab 通し実行は未実施 (F3 任意検証)。
  CLI経路の過去の一致報告は、今回修正するコードのクリーン受入を代替しない
- `dataset_upload/` staging: `C:/work/solar_run/dataset_upload/`

### 2.21 タイル推論 (native-resolution) — fold2 OOF で棄却 (2026-10-13)

v62解剖で FN1071 の68%が提案不在 (blind+weak) と分かっており、検出
リコール系で唯一未検証だったのが「全画面2048→1280の0.625圧縮で細い
フィラメントが消えている」仮説。2x2タイル (1152px, overlap256, native
解像度、SAHI相当) で検出→既存 refiner/ownership/公式PQまで通して評価。

#### 検出レベルプローブ (`scratch/run_tile_infer_probe.py`, 全235観測)

| 指標 | 値 |
|---|---|
| blind GT (best IoU<0.1) | base 143 → tile 60 |
| 救出 IoU≥0.5 | **60件** (blind の42%) |
| 救出 IoU≥0.5 かつ conf≥0.26 | **10件**のみ |
| 提案数/img (conf≥0.05) | 16.0 → 30.9 |

救出機構は明確に実在。ただし83%の救出が conf<0.26 の低信頼帯に分布。

(prod-conf 救出の当初値 8 は「best-IoU 提案自身が conf≥0.26」のみを
数える定義で、別提案が IoU≥0.5∧conf≥0.26 を同時に満たす2件を数え
落としていた — Codex P2 指摘で修正。`recount_rescue_prod_conf.py` が
同一条件で再計算し blind=143/rescued=60/旧値8 を完全一致で再現した上で
修正値 10 を確定。証跡: `output/tile_infer_probe/prod_rescue_recount_fold2.json`)

#### 全パイプライン評価 (`scratch/run_tile_pipeline_oof.py`, 3.3h)

同一提案の refined mask は決定的なので、tile は nms 生存者∪union-NMS
生存者に一度だけ refiner を掛け、keep 閾値×NMS グリッドを後段で使い回し。
Codex P1 指摘を受け tile-only に加え **base∪tile の union arm** (合併候補を
cross-source NMS 後に同一パイプライン) も評価:

| arm | PQ | Δ vs base | TP | FP | FN | final/img |
|---|---|---|---|---|---|---|
| **base@0.26 (prod)** | **0.4442** | — | 1715 | 750 | 1070 | 6.38 |
| base@0.195 | 0.4355 | −0.009 | 1780 | 968 | 1005 | 7.16 |
| tile@0.26 | 0.4047 | −0.040 | 1829 | 1530 | 956 | 8.84 |
| tile@0.10 | 0.3359 | −0.108 | 1870 | 2871 | 915 | 12.40 |
| tile@0.05 | 0.2948 | −0.149 | 1845 | 3796 | 940 | 14.79 |
| tile@0.10\|nms0.3 | 0.3540 | −0.090 | 1965 | 2788 | 820 | 12.46 |
| union@0.26 | 0.4116 | **−0.033** | 1862 | 1506 | 923 | 8.87 |
| union@0.10 | 0.3378 | −0.106 | 1861 | 2814 | 924 | 12.23 |

(base PQ 0.4442 は v62 基底 0.4432 と整合 — 再現性確認済み)

#### 判定: 棄却

- 本命の union@0.26 (base 検出を全保持+tile 救出を追加) でも
  TP+147・FN−147 に対し **FP+756 (750→1506)** で −0.033
- tile-only でも TP+114・FN−114 の救出は実在するが FP+780 で同様に崩壊
- keep 閾値を下げるほど単調悪化 (−0.033→−0.106) — 低 conf 救出は
  refiner(0.45) を通るだけの形状を持ち TP に届かず FP 化
- nms0.3 は TP 最大 (1965) だが FP 2788 で不届
- 統制 base@0.195 も −0.009 → prod の 0.26 keep は適切だった裏付け
- このfold2探索で全armが負のため、作者は追加foldを実行せず棄却した。
  fold0/1への一般化や独立検証を済ませたという意味ではない。

**構造的含意**: blind-FN 救出を収益化するには rescued-TP と tile-FP を
分けるverifierを要するという作者の解釈。v64や今回のfold2探索で試した
範囲では有効策を得られず、最終選択を変更しなかった。
未試行手法の不存在や一般的な性能上限を証明する結果ではない。

証跡: `output/tile_infer_probe/` (prod_rescue_recount_fold2.json 含む)、
`output/tile_pipeline_oof/`、
`scratch/run_tile_{infer_probe,pipeline_oof}.py`、
`scratch/recount_rescue_prod_conf.py`
(cache npz は gitignore — 結果の CSV/JSON のみコミット)。

### 2.22 v70 深層 verifier (crop CNN) — fold2 で棄却 (2026-10-13)

§2.21 の構造的含意「rescued-TP と FP を分ける verifier が必要」に対し、
v64 (静的特徴10個・HistGBM) とは別の仮説クラスとして、**画像そのものを
見る CNN verifier** を検証。入力は提案クロップ RGB+mask の4ch
(128x128, square_box context=1.65)、ラベルは「その観測の全GTとの
best IoU>0.5」(= 公式PQのTP条件、境界値は陽性にしない)。GT は
annotator 間で union (公式PQは annotator×image group 毎に採点;
「提案がTPになり得るか」の対象として採用。別定義—例: 全 annotator
一致—は未検証の変種として留保)。

#### データセット (`scratch/build_verifier_dataset.py`)

prediction_cache/oof の fold 別 OOF 提案 (conf≥~0.08) から構築。
train=fold0 / select=fold1 / eval=fold2 (v64 と同型):

| fold | crops | 陽性率 | conf 単独 AUC |
|---|---|---|---|
| 0 (train) | 2,594 | 55.6% | 0.798 |
| 1 (select) | 2,555 | 54.1% | 0.802 |
| 2 (eval) | 2,614 | 54.9% | 0.797 |

#### 結果 (`scratch/train_verifier.py`, ImageNet resnet18・8epoch)

初版レビューで前処理バグ2件 (BGRをRGBとして正規化・mask ch を /255 で
0.0039に潰す) と境界ラベル (`>=0.5` で公式の `>0.5` と不整合、境界4行)
が指摘され、修正パイプラインで再生成・再学習した。下記は修正版の数値
(初版と同方向にさらに悪化したため、棄却は前処理バグ由来でない):

| 指標 | fold1 | fold2 |
|---|---|---|
| conf 単独 AUC | 0.802 | **0.797** |
| CNN verifier AUC | 0.664 | **0.638** |
| logistic blend (conf+ver, fold1 fit) | — | 0.7934 |

- fold0 の train loss 0.71→0.35 に対し fold1 AUC ~0.66 で平坦 =
  fold0 への過学習で一般化しないパターン
- blend は conf 単独 0.797 を改善せず (0.7934)
  → verifier スコアは conf と相関0.51の劣化版で独立情報を持たない

#### 判定: 棄却 (試した範囲では)

分離性能が conf 単独を大きく下回るため、verifier として採用できる
レベルにない。v64 (静的特徴)・v68 (磁気図特徴)・v70 (画像CNN) と
3つの異なる特徴クラスで「+0.046 oracle を解放する分離情報」は
**試した範囲では見つからなかった**。

#### 制限 (正直な留保)

- 2.6k crops は CNN 学習には小規模で、過学習パターンはデータ量に
  由来し得る。より大きな提案コーパス・IoU回帰ラベル・自教師事前学習
  (GONG無ラベル) は別の賭けだが、現時点の証拠では conf 0.80 を
  大きく超える見通しは薄い
- resnet18+flip aug のみ。別アーキ・大入力解像度は未検証だが、
  AUC差 (0.64 vs 0.80) をひっくり返す期待は持ちにくい
- 入力バグ修正後の値を報告。初版(BGR・mask潰れ・境界`>=`)は
  fold2 AUC 0.659/blend 0.7975 で同じく棄却だったが、数値は修正版を
  正とする

証跡: `output/verifier_v1/` (gitignore — 再生成スクリプトと本表の数値が
記録)、`scratch/build_verifier_dataset.py`、`scratch/train_verifier.py`、
`tests/test_{build_verifier_dataset,train_verifier}.py`。

### 2.23 v71 min_area sweep — 完全不感帯・dead parameter と確定 (2026-10-15)

§6.27 の dariushafshar カーネルが「人間マスクでは <300px 除去は wash」と
示したため、prod 唯一の未検証パラメータ `min_area` (ownership 内の
排他面積フロア=20) を fold2 OOF で掃引 (`scratch/run_min_area_sweep.py`)。
検出+refiner は prod 同一 operating point (YOLO@0.26・EdgeAtt fold2・
thr 0.45) で1回走り、min_area だけ変えて ownership+公式PQ を再評価。
sanity gate: min_area=20 arm が base PQ 0.444228 を厳密再現。

| min_area | PQ | Δ vs prod | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0 / 5 / 10 / **20** / 50 / 100 | 0.4442 | ±0.0000 | 1715 | 750 | 1070 |
| 300 | 0.4452 | +0.0010 | 1709 | 724 | 1076 |

- **0〜100 で結果が bitwise 同一** — prod 最終マスクに排他面積 <100px の
  インスタンスが fold2 上ゼロ。refiner+drop_contained が小破片を事前に
  消すため、フロアは operating range で dead parameter
- 300 の +0.0010 は TP−6/FP−26 のトレードで noise band (±0.002) 内の
  wash — 採用根拠なし。fold0/1 検証を行う価値もない
- 含意: dariushafshar の人間レベル予測がモデルでも再現。§6.27 留保の
  「別途要 OOF」を消化 — prod の全チューニングパラメータがこれで
  検証済み (conf/閾値/drop/min_area/重み選択、残存パラメータなし)

証跡: `output/min_area_sweep/` (gitignore — summary_fold2.json・
per_obs_fold2.csv・cache/)。スクリプト `scratch/run_min_area_sweep.py`。

### 2.24 F3 クリーンラン受入 — 両提出が byte-identical で再現 (2026-10-15)

`solar_filament_submission_reproduction.ipynb` を Kaggle kernel
(`shingo257/solar-filament-2026-reproduction`) へ push し、
`shingo257/solar-filament-2026-weights` dataset + 競技データのみを参照する
クリーン環境で両 variant を実行。

| variant | kernel ver | wall | CSV sha16 | 提出済み参照 | 判定 |
|---|---|---|---|---|---|
| prod | v3 | ~4h | `84f1b89466f988d9` | `84f1b89466f988d9` | **一致** |
| plain_mhsa_f2 | v4 | ~14.5h | `7b3de4e78a543cf2` | `7b3de4e78a543cf2` | **一致** |

- 重み解決も確認: kernel 出力の `yolo_fold_2/best.pt`・`refiner_all_data.pt` が
  dataset の `yolo_fold2_best.pt`・`refiner_all_data.pt` と SHA256 一致
  (`40ec31fd2c6aa8e3` / `e9f0e8dd7ff104d9`) — 再学習フォールバックではなく
  凍結重みをそのまま使用
- plain_mhsa_f2 の `submission_efficiency.json`: T4 で 0.97 s/img
  (report の ~1.1 s/img と整合)
- v4 の wall ~14.5h は Kaggle 側の依存 install・データ DL・前処理が律速
  (v3 の prod も ~4h)。推論そのものは ~3分
- 含意: 主催者要件「選択した両提出を再現できること」を現コード世代で充足。
  README/main.tex の「clean-run acceptance pending」記述を更新

### 2.25 v72 GONG SimCLR 事前学習 → verifier init 交換 (2026-10-16)

§2.22 の留保「2.6k crops では視覚特徴が学べない → 大規模ドメイン事前学習のみ残る」
を検証する唯一残ったレバー。無ラベル GONG Hα 全日面アーカイブ
(`gong2.nso.edu/ftp/HA/haf/`) で ResNet18 を SimCLR 事前学習し、v70 verifier の
ImageNet init と差し替えて fold2 AUC を比較する。

**データ契約** (`scratch/build_gong_ssl_corpus.py`):
- 2011年の12月×8日×20枚を均等サンプル → **1,889 フレーム取得**（6サイト全て）
- **漏洩ガード**: test 観測 (180枚) の timestamp ±360分のフレームを
  site 不問で除外 → **30 フレーム除外**、train 観測の完全一致も除外
- 各フレームの URL・UTC・sha256 を `output/gong_ssl_corpus/manifest.jsonl` に記録
  （npz 本体は再生成可能なので gitignore）
- 1 フレームのみ decode 失敗（npz から除外・manifest に不記載）

**手法** (`scratch/pretrain_gong_ssl.py`): ResNet18 (weights=None) +
projection head、NT-Xent (τ=0.2)、2-view = RandomResizedCrop(128, scale
0.05–1.0) + flip + gain jitter。`train_verifier.py --init ssl:<ckpt>` で
backbone を差し替え（conv1 RGB 3ch のみ移植、mask ch は従来どおり RGB 平均）。

**評価**: Kaggle T4 kernel `solar-filament-v72-gong-ssl` で
(i) ssl init と (ii) imagenet init の同一 seeds/epochs (8) を同一環境で比較。
fold0 学習 → fold1 で epoch 選択 → fold2 で凍結評価 (v70 と同一プロトコル)。

**棄却ゲート（事前固定）**: ssl init の fold2 AUC が imagenet init (0.638) を
有意に上回らない、または blend が conf 単独 (0.797) を超えない → 棄却。

**コーパス配布**: private dataset `shingo257/solar-gong-ssl` (corpus 231MB +
コード) と `shingo257/solar-verifier-crops` (crops+labels 340MB) +
`shingo257/solar-ssl-code` (コードのみ)。kernel v5 COMPLETE。

**結果** (fold2 凍結評価、8 epoch、seed 固定・同一環境):

| arm | verifier AUC (fold2) | blend AUC (fold2) |
|---|---:|---:|
| ImageNet init (v70 再現) | 0.6435 | 0.7939 |
| **GONG SSL init** | **0.6993** | **0.8020** |
| conf 単独 (参照) | — | 0.7973 |

SSL 事前学習 40 epoch / 1,889 フレーム / T4 で約5分。verifier fold1 AUC
0.7036 (best epoch 8)。**事前固定ゲートは通過**: fold2 AUC が ImageNet init を
+0.056 上回り、blend (0.8020) が conf 単独 (0.7973) を +0.0047 上回る —
3つの特徴クラス全滅 (v64 静的・v68 磁気図・v70 ImageNet CNN) 後、初めて
conf を超える分離情報を取得。

ただし **AUC 改善 ≠ PQ 改善**。fold2 相関 ≈0.48 で verifier は conf と
相補的だが、絶対ゲインは小さい。最終判断は prod operating point での
直接測定に委ねる (`scratch/run_verifier_filter_eval.py`):
conf≥0.26 かつ verifier prob ≥ t で提案を pre-filter し fold2 OOF PQ を
t ∈ {0,0.2,...,0.7} で sweep。

**PQ 評価結果 → 棄却** (235 obs、提案照合 1589/1589 = 100%):

| verifier t | PQ | Δ vs フィルタ無し | TP/FP/FN |
|---:|---:|---:|---|
| 0 (gate) | 0.4483 | — | 1731/752/1054 |
| 0.2 | 0.4467 | −0.0016 | 1687/686/1098 |
| 0.3 | 0.4366 | −0.0117 | 1587/598/1198 |
| 0.4 | 0.4088 | −0.0395 | 1368/449/1417 |
| 0.5 | 0.3668 | −0.0815 | 1112/298/1673 |
| 0.6 | 0.2942 | −0.1541 | 789/176/1996 |
| 0.7 | 0.1893 | −0.2590 | 441/74/2344 |

**全閾値で単調悪化**。最緩の t=0.2 (提案の ~4% 除去) でも −0.0016。
verifier が低スコアと判定する提案には TP が FP より多く含まれ、FN の増加が
FP 減少を常に上回る — conf との相関 0.48 で「conf が拾う本物を verifier は
見逃す」構造。blend AUC +0.0047 の信号は PQ の TP/FP 分離には変換されない。

gate PQ は 0.4483 で min_area sweep 再現値 0.4442 から +0.004 偏移
(OOF npz 提案プール由来の微小差; 採否の結論は不変)。

**結論**: SSL 事前学習は verifier の AUC を実質改善した (+0.056) が、
**prod 統合の根拠となる PQ 改善には到達せず棄却**。verifier 構想は
AUC レベル (v70) と PQ レベル (v72) の両方で終了。prod / plain_mhsa_f2 の
最終提出構成は不変。
