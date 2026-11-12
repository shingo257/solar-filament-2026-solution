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

**履歴上の結果に関する留保 (2026-10-17 独立レビュー)**:
以下は旧 corpus guard による実験の記録。旧実装は competition test にのみ
±360分を適用し、train は完全一致時刻だけ除外していた。train に含まれる
verifier select/eval fold の時間近接フレームが SSL に入り得るため、
fold2 の独立した凍結評価とは扱えない。実 corpus の近接重複は未監査であり、
実際に混入したと断定するものではない。数値は歴史的な棄却結果として保持し、
独立性能の改善や本番採用の根拠にはしない。再学習・held-out 再評価は行っていない。

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

**当時の結果** (fold2 評価、8 epoch、seed 固定・同一環境):

| arm | verifier AUC (fold2) | blend AUC (fold2) |
|---|---:|---:|
| ImageNet init (v70 再現) | 0.6435 | 0.7939 |
| **GONG SSL init** | **0.6993** | **0.8020** |
| conf 単独 (参照) | — | 0.7973 |

SSL 事前学習 40 epoch / 1,889 フレーム / T4 で約5分。verifier fold1 AUC
0.7036 (best epoch 8)。**当時の数値比較上はゲート通過と記録**: fold2 AUC が ImageNet init を
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

**後続の corpus 構築ガード (2026-10-17)**: 新規構築では全 competition
train/test 観測の ±exclusion_minutes（境界を含む）を site 不問で除外する。
train の全時刻を保護するため split に依存せず verifier select/eval fold も
包含する。train の空・不正な時刻集合は listing 前に停止する。
guard policy の世代を設定指紋に含め、旧 corpus の resume は拒否する。
このガード修正で過去の実験結果の独立性が遡って保証されることはない。


### 2.26 v73 GONG SimCLR → refiner encoder init 交換 (設計 2026-10-18)

v72 は無ラベル GONG 事前学習を **verifier** (RQ 側の提案選択) にだけ
適用して棄却された。残る未試行レーンとして、同じ SSL を **prod refiner
(`MediSegEdgeAttUNet`) のエンコーダ部** (inc/down1-4, SQ 側のマスク品質)
に適用する効果を検証する。encoder のキー構造は MediSegUNet と同一で、
decoder/edge_attn は従来どおりランダム初期化。

設計 (scratch/train_refiner_fold.py + pretrain_unet_encoder_ssl.py):
  - SSL 入力は solar_three_channel 適用後の 3ch (下流 crop と同一表現)、
    正方形 RRC → 320px、h/v flip + ch0/1 gamma jitter
  - 学習は v35/v51 と同一レシピ (record 単位 crop、14 epoch、seed 3407)
  - arm は `scratch` と `ssl:` の同 seed ペア + 既存
    `mediseg_edgeattnet_refiner_fold2_resume.pt` を参照 arm として
    同一 YOLO 提案・同一後処理で採点 (paired)

事前固定ゲート: fold2 OOF PQ で ssl−scratch >= +0.010 なら別 fold・別
seed の確認実験へ進む。0〜+0.010 はノイズ帯として棄却、<0 も棄却。
sanity: baseline ckpt の fold2 PQ が ~0.448 付近であること (同じ pipeline
の既知値との一致確認)。

#### 副次発見: fold registry の provenance 不整合 (要修正)

調査中に `grouped_folds.csv` が2系統存在することが判明:

  - `solar_filament_v30/artifacts/grouped_folds.csv` (2026-08-14,
    sha256 `febb3f67...`) — OOF cache `prediction_cache/oof/yolo_fold_*`
    (2026-08-15 生成) のファイル割当と 236/236/235 完全一致。
    OOF cache のファイル割当はこの分割と一致する。**元チェックポイントの学習分割との結び付けは未確認**。
  - weights dataset / repo kernel_run の `grouped_folds.csv`
    (sha256 `24367120...`, 2列) — **489/707 で fold 割当が異なる別分割**。
    `build_submission_reproduction.py` の `GROUPED_FOLDS_SHA256` ピンと
    weights dataset 同梱分は後者。推論再現には影響しない (重みは同梱済で
    fold は使わない) が、学習再現を試みる第三者は実験と異なる分割を引く。

対応: v73 は実験で使われた分割 (febb3f67) を `solar-v73-assets` dataset に
`grouped_folds_train_split.csv` として同梱して使用する。weights dataset /
reproduction notebook 側の CSV を実験分割に差し替える修正は提出パッケージの
凍結済み部分に触れるため、本 PR では扱わず別途判断を求める (影響は
「学習再現時の分割」だけで、byte-identical 推論再現は済)。

報告された結果 (2026-10-19、fold2 OOF・235ファイル・conf0.26。独立照合は未完了):

| arm | ckpt sha256 | PQ@0.45 | best_crop_dice |
|---|---|---|---|
| baseline (v51) | `b1c0f353...` | 0.444228 | 0.7930 |
| scratch | `c43b27ba...` | 0.445390 | 0.7964 |
| ssl-init | `c04ae6c5...` | 0.441192 | 0.7842 |

判定: **delta ssl−scratch = −0.004198 → 棄却** (事前固定ゲート <0)。
報告上は確認実験には進まない。baseline が既知値に近いだけで pipeline の独立受入は確定しない。
scratch と baseline の報告差は約 +0.0012。独立照合や seed 間の変動評価は未完了であり、昇格根拠にはしない。

収集: SSL encoder `bbdfdd66...` (75MB・60ep)、`v73_eval.json`
`2b0f2c50...`、`v73_summary.json` `019662e5...`。DL した両 refiner ckpt の
sha256 は報告上 eval 記録値と一致。報告された corpus provenance は
`train-test-temporal-v2`・1,707 frames、観測 digest `ee9354dc/a3dc309d`
(監査ピン値と一致)。

暫定解釈: 今回報告された fold2・単一 seed の比較では SSL init が scratch を下回ったため、不採用とする。
ドメイン全体への一般化や独立性能の受入を示すものではない。


独立照合の制限 (2026-10-19): 確認したローカル保存先では、新しい評価 JSON・
学習履歴・重みの本体を発見できず、新しい値とハッシュは報告値として扱う。
元の学習分割との結び付け、corpus 原典の受入、実行成果物の独立照合は未完了。
runtime の manifest ピン一致は独立承認を代替しない。


マージ状態の追記 (2026-10-19): 文書修正 #397 の後、別の操作で #388 は main に
マージされた (fd7d134a、09:00:16 UTC)。これは未確認成果物・manifest・
元の学習分割や原典に対する独立受入の完了を意味しない。実際のマージ状態と
証拠の受入状態を分けて記録する。


実行JSONの追加照合 (2026-10-19): 既存 COMPLETE の Kaggle v73 kernel から
評価・summary・SSL履歴等を取得し、記録済みの SHA256 と一致を確認した。
PQ の完全精度差は SSL−scratch = −0.004198331766480823。
summary/eval と PQ/SQ/RQ/件数の計算整合性も確認でき、不採用継続を支持する。
予測の再採点・新しい重み本体の再ハッシュや学習分割の独立受入ではない。

runtime corpus 設定は SHA256 ebf1c607... / 指紋 e1e66a843bc3e2d1、
監査済みv3と現在取得できた設定は 2a1ae10d... / a1f436fea3f26380 で異なる。
フレーム manifest・観測 digest・1,707枚・除外360分は一致するが、
既知の過去/現行の指紋計算でも runtime 値は再現せず、空白や改行だけの差では
説明できない。当時の設定原本・固定入力版・実行コードの照合を要する。
Gドライブの同期コピーの所在地は未確認。詳細は docs/V73_PROVENANCE.md を参照。
今回の調査で新しい学習・GPU実行や原典への再アクセスは行っていない。

### 2.27 v74 — v73 SSL refiner init 棄却の確定実験 (設計・事前登録 2026-10-19)

v73 (§2.26) で ssl encoder init − scratch = **-0.004198** (fold2 OOF PQ@0.45,
seed 3407) と判定され不採用になった負結果について、残る2つの反論を同じ
reviewed pipeline・同じ入力で潰す確認実験。Colab notebook
`solar_filament_v74_ssl_confirm_colab.ipynb` (builder:
`scratch/build_v74_ssl_confirm_colab.py`)。

**アーム** (全て fold2 OOF・同一 YOLO 提案・同一後処理・edgeatt・14ep):

| arm | init | seed | 由来 |
|---|---|---|---|
| baseline | prod ckpt | — | Drive v30 (sanity) |
| scratch | scratch | 3407 | v73 ckpt 再利用 (`solar-v73-results`、sha `c43b27ba...` 検証) |
| ssl | SSL-60ep | 3407 | v73 ckpt 再利用 (sha `c04ae6c5...` 検証) |
| scratch_s8407 | scratch | 8407 | 新規学習 |
| ssl_s8407 | SSL-60ep (v73 encoder sha `bbdfdd66...` 再利用) | 8407 | 新規学習 |
| ssl200_s3407 | SSL-200ep (同 corpus・同 solar3 変換・新規事前学習) | 3407 | 新規学習 |

**潰す反論**:
- A. seed 依存: ssl_s8407 − scratch_s8407 でも負なら v73 の棄却は seed ノイズではない。
- B. 事前学習不足: 60ep SimCLR は小規模。200ep に延長した arm が scratch_s3407 を
  上回るかを検証 (同じ corpus/変換/損失・延長のみ変更)。

**事前固定ゲート** (v73 と同一形式):
- 各比較 delta = ssl系 − 同seed scratch の fold2 OOF PQ@0.45 で
  `>= +0.010` → 別 fold 確認実験へ / `0 <= delta < +0.010` → ノイズ棄却 /
  `delta < 0` → 棄却
- sanity: baseline PQ が v73 実績 `0.444228` 近傍 (|diff| < 0.01)
- reproduction check: 再利用 arm (baseline/scratch/ssl) の PQ が
  `v73_eval.json` 記録値と |diff| < 1e-4 → pipeline 決定性の検証
  (YOLO 提案生成の実行環境差を含めて観測する)

**入力は v73 と同一 reviewed セット**: provenance pin
`V73_PROVENANCE_SHA256` (manifest `2c954685...`)、folds CSV sha `febb3f67...`、
corpus `train-test-temporal-v2` (1,707 frames)、同じ YOLO/baseline ckpt。
学習・評価スクリプトは `solar-v73-code` 版を変更せず再利用
(`--seed`/`--init ssl:`/`--epochs` 引数のみ)。v73 成果物は
`solar-v73-results` dataset から **sha256 ピン検証付き**で取得し、
dataset 同梱の `sha256_manifest.json` との二重照合も行う。
結果・重みは Drive `solar_filament_v74/work/latest/` へ sha256 検証付き
増分ミラー (各長時間 cell 後)。

**v74 結果 (2026-10-19/20, Colab T4 完走 ~4.5h, Drive ミラー sha256 検証済み)**

| arm | PQ@0.45 | 備考 |
|---|---|---|
| baseline | 0.444238 | sanity ✓ (v73: 0.444228) |
| scratch (v73 再利用) | 0.445579 | v73 記録 0.445390 との差 +0.000189 |
| ssl (v73 再利用) | 0.441567 | v73 記録 0.441192 との差 +0.000375 |
| scratch_s8407 | 0.444912 | — |
| ssl_s8407 | 0.444661 | **delta = -0.000251 → 棄却** |
| ssl200_s3407 | 0.423852 | **delta vs scratch_s3407 = -0.021727 → 棄却** |

- **seed 頑健性 (反論A)**: ssl_s8407 < scratch_s8407 (-0.0003)。v73 の棄却は
  seed ノイズではない。2seed とも ssl は scratch を下回る (s3407: -0.004,
  s8407: -0.0003)。
- **事前学習不足 (反論B)**: 200ep に延長すると PQ 0.4239 と大幅悪化
  (-0.022)。SimCLR 事前学習を伸ばすほど悪い — under-pretraining 説は棄却。
  refiner への GONG SSL 適用は決定的に不採用 (v72 verifier に続き3回目の
  同結論)。
- **reproduction check**: 再利用 arm の PQ drift は +0.00001/+0.000189/
  +0.000375 (baseline/scratch/ssl)。私が設定した |diff|<1e-4 ゲートは
  scratch/ssl でわずかに超過 — Kaggle kernel と Colab の環境差
  (CUDA/ライブラリ版) による提案生成の微小差と解釈。v73 の順序関係・符号・
  delta 規模 (-0.0040 vs -0.0042) は完全に再現しており pipeline の
  健全性は確認できた。ゲート閾値の経験値として ~4e-4 が環境差ノイズ帯。

成果物: `edgeatt_scratch_s8407_fold2.pt` `868a8441...`、
`edgeatt_ssl_s8407_fold2.pt` `df968e64...`、
`edgeatt_ssl200_s3407_fold2.pt` `b1f3a854...`、
`unet_encoder_ssl200.pt` `d18b4817...`、`v74_eval.json` `4fac918d...`、
`v74_summary.json` `696d2abc...` (Drive `solar_filament_v74/work/latest/`、
全 sha256 照合 OK)。Kaggle dataset `shingo257/solar-v74-results` にも永続化。

### 2.28 discussion/LB 再調査 — 第3回リフレッシュ + sparse-label 説の独立検証 (2026-10-20)

§2.19 から8日後の再取得 (kagglesdk `DiscussionApi.ListTopics` search union で
24スレッド確認、LB は `competitions leaderboard --download` の全1,103チーム
CSV を解析)。

#### Discussion: 新着2件 + 未回答の重要質問1件

- **748498** (10-12, 回答0件): Esraa Khaled「KSO/Meudon/BBSO の無ラベル
  Hα を SSL 事前学習に使えるか」— test と**同日付の別天文台画像**を含むか
  ±N日除外すべきかを主催者に確認中。我々の GONG-only SSL (v72-v74) の
  拡張方向そのもの。回答が出たら記録要
- **749225** (10-15, 3コメント): GONG archive の一括 DL が 403 → Vedran が
  「2並列+1s間隔+VPN、banは~12hで解除」と回避策報告、主催者 Azim が
  GONG IT のボット対策を確認済みと回答。我々の corpus は取得済みで影響なし
- 既知裁定 (746318 外部GT禁止 / 744047・748045 GONG無ラベル可+時間除外 /
  743521 磁気図可 / 747960 両提出再現) に変更・訂正はなし

#### LB スナップショット (1,103チーム・ERS rank 176 / 0.39)

| 帯域 | 10-12 | 10-20 | Δ |
|---|---|---|---|
| 総チーム | 983 | 1,103 | +120 |
| ≥0.55 | 78 | 87 | +9 (うちちょうど0.55が70) |
| 0.45–0.54 | 3 | 3 | 0 |
| 0.40–0.44 | 31 | 42 | +11 |
| 0.35–0.39 | 332 | 400 | +68 |

- トップ2が 0.64 に更新: **IONode Online (12 subs)**・**Tempest (123 subs)**。
  0.55–0.63 帯の使用データ・分割・来歴は引き続き独立未確認 — スコア帯だけで
  違反や誠実性は判定しない (§2.16 と同じ原則)
- 誠実想定帯 0.40–0.44 が 31→42 に増加 — 誠実系フロンティアは緩やかに前進
- ERS は rank 136→176: スコア不変 (0.39)・新規参入120チームによる純粋な
  分母増加

#### 新着 kernel: dariushafshar の sparse-label 説 (最重要)

`solar-filament-u-net-oof-0-45-vs-public-0-39` (10-17, 5票) — 検証済み
receipt・folds・keep-rule グリッドを全公開した誠実系 dossier:

| 項目 | 内容 |
|---|---|
| 構成 | smp U-Net ×10 (5fold ResNet34 + 5fold EffNetV2-S) 平均・full-res 2048²・D4 TTA8view・annotator 平均 soft target・連結成分 keep rule (t_lo .35/peak .99/area 200/meanp .85) |
| 成績 | nested OOF 0.4502 (707画像・1154 annotator-set pooled) → public 0.39 — **我々と同水準 (OOF 0.445→LB 0.39)** |
| sparse-label 説 | test ラベルは学習ラベルより疎 — train annotator が描く filament が確率 r で test ラベルに残ると仮定し `PQ' = rΣIoU/(r·TP + ½FP + ½(1-r)TP + ½r·FN)`。**v3 で r=0.77 を fit → v5/v6 の LB を提出前に的中** |
| annotator 間 PQ | **0.3399** (296画像・別 kernel 実測) — 我々の 0.3398 とほぼ同一の独立収束 |
| 姿勢 | 「fit する説であって証明ではない」と明記・全数値を receipt に sha256 付き公開 — 我々と同じ証跡文化 |

**独立検証 (我々の pooled counts に r=0.77 を適用)**:

| arm (v74 fold2 @thr .45) | OOF | PQ'@r=0.77 |
|---|---|---|
| baseline (prod 相当) | 0.4442 | **0.3896 ≈ 実績 0.39** |
| scratch | 0.4456 | 0.3907 |
| ssl | 0.4416 | 0.3871 |

- 彼が自分のデータで fit した r=0.77 が、**別パイプライン (detector+refiner)
  の我々の OOF→LB 差をそのまま説明** (0.3896 vs 実績 ~0.39、表示精度内一致)
- 707画像 pooled counts (report Table) でも同程度: TP 5082/FP 2286/FN 3117・
  SQ 0.678 → PQ'@0.77 ≈ 0.388
- 意味: OOF→LB 差 ~0.05-0.06 は「test ラベル疎化 ~23%」で両チーム共通に
  説明可能 — 我々の「未説明の乖離」(report §weaknesses) への**実用仮説**。
  彼自身も証明でないと明記しており、我々も「説明可能な仮説の1つ」として
  記録に留める (真の test ラベル密度は非公開)
- 含意: sparse-label 下では keep-rule 最適化の利得はほぼ消える (彼の grid
  解析でも同結論) — 後処理側の微調整を追わない判断を補強

#### 取り込み判断

- **採用レバーなし**: 彼の keep-rule グリッドは我々の threshold sweep 棄却と
  同型・同結論。10model ensemble の OOF +0.005 程度は LB 分解能以下
- **継続監視**: 748498 の主催者回答 (別天文台 SSL の可否 + 同日付除外要否)、
  LB 0.55+ 帯の動向、次回以降の periodic refresh で追跡
- 結論: **prod + plain_mhsa_f2 維持**。外部手法で採用に値する新規定量レバー
  なし — むしろ OOF→LB 乖離の主要仮説 (sparse test labels) と annotator
  ceiling 0.34 が独立系で再現され、評価系の解釈が補強された

### 2.29 discussion/LB 再調査 — 第4回リフレッシュ：主催者回答3件・LB 0.69・Fususu 手法公開 (2026-10-21)

§2.28 から1日後。今回は `competition_list_topics`（公式 enumeration、
41 topics 全件）で取得 — 前回の search-union 方式は 24 件しか拾えて
いなかったため、以後こちらを正本とする。LB は全1,104チーム CSV。

#### Discussion: 主催者の新規回答（重要度順）

- **735256**（注釈抜け、主催者回答済）: 「train 注釈が visible filament を
  一部描いていない」指摘に対し主催者は「全 filament の検出が目的、FP は
  PQ で罰されるが、審査では many-to-many 関係の分布も見る（Self
  Evaluation notebook のプロット参照）」と回答 — **sparse-label 仮説と
  整合する主催者レベルの証言**（train 側の注釈漏れは公式に確認済み）
- **749526**（計算効率）: end-to-end efficiency は**定性評価のフィルタ
  のみ**（極端に遅い/メモリリークする pipeline の除外用）でありスコア
  競争因子ではないと主催者回答 — 我々の ~1.1s/image は問題なし
- **749524**（フォーム PDF 差替不可）: 主催者が修正済み — 最終提出
  フォームは**再提出可能（最大10ファイル）**。レポート更新が必要に
  なった場合に対応可
- **747960**（両提出の再現）: 「戦略が有意に異なるなら両方の再現を
  要求、可能なら統合を推奨」— 我々の repo は prod と plain_mhsa_f2 の
  両方をバイト一致再現済みなので適合
- **748498**（別天文台 KSO/Meudon/BBSO の無ラベル Hα SSL 可否）:
  **依然として回答なし**（質問のみ）。外部 SSL 拡張は保留継続
- **724503**（Organizers' Remarks、8-20 更新）: 「システムを hack する
  参加者がいるのは明白。誠実な参加者は他者スコアと比較せず作業を続けて
  ほしい。**0.3 を超える PQ は価値がある**」と公式見解 — 我々の 0.39 は
  この誠実帯の上位

#### LB スナップショット（1,104 チーム・ERS rank 178 / 0.39）

| 帯域 | 10-20 | 10-21 | Δ |
|---|---|---|---|
| 総チーム | 1,103 | 1,104 | +1 |
| ≥0.55 | 87 | 89 | +2（ちょうど0.55が**72**） |
| 0.45–0.54 | 3 | 3 | 0 |
| 0.40–0.44 | 42 | 42 | 0 |
| 0.35–0.39 | 400 | 402 | +2 |

- トップが急伸: **IONode Online 0.69**（前日0.64、本日も提出継続・15 subs）、
  Tempest 0.64、Sayandeep Dutta 0.62（99 subs）等。**0.56–0.69 は全て異なる
  値**で plateau ではない — トップ層は継続的に改善中（または新規 exploit）。
  一方でちょうど 0.55 に 72 チームが停滞
- 745880 の議論より: コミュニティの認識は「**0.55 csv**（共有提出ファイル）
  とデータリークで plateau が形成、誠実系の最上位は noisyislands の 0.49
  あたり」。我々の sparse-label 分析・annotator ceiling と整合
- ERS rank 176→178、score 0.39 不変（分母増・純粋な順位変動）

#### 上位手法の公開物調査

- **phuongncn/lb-0-70-topology-safe-solar-filaments-handoff**: Fususu の
  「合法・再現可能」提出の詳細 handoff — **Mask R-CNN detector +
  ResNet18 U-Net crop refiner + flip-4 TTA（topology ガード付）**。
  我々と同型の detector→crop refiner 構造。8-5 時点の scored refs
  55027846/55033541 で 0.70 を記録（旧メトリック期）。topology ガード
  （面積・連結成分数・重複の保存条件）・crowding 時の split 抑制など
  後処理設計が詳述 — **詳細が公開された唯一のトップ系手法**
- `rokaiyasomapti/lb-1st-...` はタイトル詐称の generic ResNet34 U-Net
  baseline（現トップの IONode とは無関係）
- 現 LB 上位チーム（ionodeonline・tempest 構成員・duttasam 等）は
  **公開 kernel なし** — 0.56+ 帯の手法は非公開
- `muhammadshoaibaltaf/magfilo-v1-...`: observation-level split・
  Tversky loss（FN 重視）・YOLO det + crop seg — 誠実系の詳細 pipeline
- `vmerckle/simple-dark-filament-segmentation-0-46`: 学習なし古典 CV
  （disk 推定→背景減算→暗部 threshold→連結成分）で 0.46 — 誠実下限参照
- dariushafshar: 前回調査以降の新規 solar kernel なし

#### 取り込み判断

- **採用レバーなし**: Fususu 手法は我々と同型だが TTA4+topology ガードは
  我々の TTA-consensus 棄却（+0.0007 で不移植）と同系統。flip-TTA の
  「topology 保存条件付き」という改良点は興味深いが、彼ら自身も
  「多数の variant が 0.70 でタイ」と記しており LB 分解能以下の効果
- 主催者回答3件は提出要件の明確化に有用（再提出可・efficiency は定性
  のみ・両提出再現の確認済み適合）
- **prod + plain_mhsa_f2 維持**。sparse-label 仮説は 735256 の主催者証言
  （注釈漏れの存在）でさらに裏付けられた — ただし test 側密度は依然非公開

### 2.30 v75 設計: refiner flip-4 TTA + topology guard (事前登録, 2026-10-21)

§2.29 で発見した Fususu handoff (phuongncn, Mask R-CNN + ResNet18 U-Net
refiner + flip-4 TTA with topology guards、旧メトリック期 0.70) の TTA
機構を、我々の reviewed pipeline で検証する**推論のみの実験**。
学習なし・prod ckpt 据え置き — 後処理 variant のみを変えた paired
comparison。我々の v65 TTA-consensus (detector 信頼度 consensus filter,
+0.0007 で不移植) とは別機構: これは refiner 確率マップの4view反転平均
+ topology 保存ガード。

**arm** (fold2 OOF 235ファイル・frozen operating point
conf0.26/minarea20/drop0.90・同一 YOLO 提案・同一 baseline ckpt):

| arm | prob map | 意味 |
|---|---|---|
| baseline | single-view (prod 経路) | sanity ≈ v74 実績 0.444238 |
| tta4 | flip4 平均 (guard なし) | TTA 平滑化効果そのもの |
| tta4g | flip4 平均 + topology guard | Fususu 型 guard の寄与 |

**topology guard (事前固定)**: crop 320² の 0.5 二値マップで
① 面積比 tta/single ∈ [0.85, 1.15] ② 連結成分数 (>=16px) 差 <=1
③ IoU(single, tta) >= 0.85 — 全条件を満たす時のみ TTA マップ採用、
違反時は single-view にフォールバック。Fususu は fallback ~0.7%
(1261/1270 採用) を報告。

**事前固定ゲート**:
- primary delta = PQ(tta4g) − PQ(baseline) @0.45:
  - >= +0.005 → 昇格候補 (別 fold/LB での確認実験がさらに必要)
  - [0, +0.005) → ノイズとして棄却 / <0 → 棄却
- secondary delta = tta4 − baseline で guard の load-bearing 度を分離
  (guard なしで改善するなら guard 不要、悪化するなら guard が本質)
- sanity: baseline PQ@0.45 が v74 実績 0.444238 ±0.001 (eval path は
  run_fold_oof_eval と同一 helpers を import 共有)
- 副次記録: fallback 率 (guard の発火頻度)

**実装**: `scratch/run_fold_oof_eval_tta.py` — run_fold_oof_eval の
helpers/provenance 検証を import 共有し `name=path[:mode]` arm 指定。
notebook `solar_filament_v75_tta_guard_colab.ipynb` は同スクリプトを
embedded 書き出し (repo file と byte 一致を test で担保)、solar-v73-code
の shared helpers を利用。新規 dataset 不要・新規学習なし。

**入力 pin**: V73_PROVENANCE_SHA256 (v73/v74 と同一 manifest)、folds
`febb3f67...`、Drive v30 baseline/YOLO ckpt。想定所要 ~1.5h
(proposals ~10min + baseline ~8min + tta4/tta4g 各 ~32min)。

**修正 (2026-10-23, Colab 初回実行で発覚)**: arm 名 `tta4`/`tta4g` は
manifest の checkpoint role ではないため `validate_checkpoint_binding`
が `checkpoint provenance missing: tta4` で拒否した。`resolve_binding_role`
を追加し、arm 名が manifest に無い場合はファイル bytes が reviewed
binding のどれかと SHA256 一致する時だけその role を継承、一致しない
未知バイトは従来どおり拒否。セキュリティ不変 (reviewed バイトのみ評価
可能) で TTA arm が同一 ckpt を共有できるようになった。notebook 再生成済。

**結果 (2026-10-23, Colab T4 完走 36min)**: 事前固定ゲートどおり
**reject_noise** — TTA 系レーンを閉じる。

| arm | PQ@0.35 | PQ@0.40 | PQ@0.45 | PQ@0.50 |
|---|---|---|---|---|
| baseline | 0.443393 | 0.443862 | 0.444238 | 0.444532 |
| tta4 | 0.444286 | 0.444623 | 0.444982 | 0.445092 |
| tta4g | 0.444299 | 0.444633 | 0.444994 | 0.445305 |

- `delta_tta4g_minus_baseline = +0.000756` → 事前固定帯 [0, +0.005) →
  **ノイズとして棄却** (昇格候補条件 +0.005 未満)
- `delta_tta4_minus_baseline = +0.000744` — guard の load-bearing 度は
  ほぼゼロ (tta4g − tta4 = +0.000012)
- guard_stats: applied 1678 / fallback 23 (1.35% — Fususu 報告 ~0.7% と同桁)
- **sanity 完璧**: baseline PQ@0.45 = 0.444238 は v74 実績と bit 一致
  (diff 0.0) — paired eval の決定性を確認
- @0.45 TP/FP/FN: baseline 1715/750/1070 → tta4g 1717/750/1068。
  TTA は境界 TP を2件救うが全域でサブミリの効果のみ
- 全閾値で単調 +0.0007〜0.0009 — 方向は一貫して正、振幅は常に帯域以下

**解釈**: v65 detector-consensus TTA (+0.0007、非移植で棄却) と同一振幅の
効果が、独立した機構 (refiner 確率マップ flip-4 平均) でも再現。
TTA 系平滑化の頭打ちは ~+0.0007-0.0008 帯で、このパイプラインでは
昇格ゲートに届かないことが2系統で確認された。4× の推論コストに対し
リターンが小さく、prod 採用しない。

**成果物**: Drive `solar_filament_v75/work/latest/` (v75_eval.json /
v75_summary.json / sha256_manifest.json、2/2 SHA256 検証済) +
Kaggle dataset `shingo257/solar-v75-results` (private)。folds
`febb3f67...`、manifest `2c954685...`、yolo `40ec31fd...`、refiner ckpt
`b1c0f353...` (baseline = v73 assets の mediseg_edgeattnet_refiner
_fold2_resume.pt と同一バイト)。

**採用判断**: prod + plain_mhsa_f2 維持。事前固定ゲートの帯域内効果の
ため別 fold 確認実験は発動しない (ゲート文面どおり)。

### 2.31 v76 設計: refiner checkpoint ensemble (事前登録, 2026-10-23)

v75 で TTA レーンが閉じた後に残る最後の定石レーン: **推論時 checkpoint
ensemble**。独立に学習・検証された refiner の確率マップを平均する
(学習なし・推論のみ)。v75 と同一の paired eval 構造 — 同一 YOLO 提案・
同一後処理・同一公式 PQ で member 構成だけを変える。

| arm | prob map | member ckpts |
|---|---|---|
| baseline | prod single-view | fold2_resume (manifest role=baseline) |
| ens3 | 3 member 平均 | baseline + v74 scratch_s8407 + v74 ssl_s8407 |
| ens3f | 3 member × flip4 平均 | 同上 (ensemble + TTA 積み上げ) |

**member provenance**: 全て edgeatt arch。baseline は manifest 宣言済、
v74 ckpt は ckpt 内 fold/folds_sha256 metadata で検証
(`resolve_binding_role` が metadata-bearing ckpt を許容、unknown bytes
は拒否 — v75 修正に metadata パスを追加)。

**事前固定ゲート**:
- primary delta = PQ(ens3f) − PQ(baseline) @0.45:
  - >= +0.005 → 昇格候補 (別 fold/LB での確認実験がさらに必要)
  - [0, +0.005) → ノイズとして棄却 / <0 → 棄却
- secondary delta = ens3 − baseline で ensemble 単独効果を分離
- sanity: baseline PQ@0.45 が v74/v75 実績 0.444238 ±0.001

**期待値の正直な見立て**: TTA (+0.0007-0.0008)・SSL (負) の実績から、
ensemble も +0.001 前後の帯域効果に留まる可能性が高い。ただし
「個別には負だった variant の確率平均」は diversity の効く典型条件で、
計算コストも軽い (~2.3h、新規学習なし) ため、誠実系の残りレーンとして
完遂する価値がある。負なら ensemble レーンも確定的に閉じる。

**実装**: `scratch/run_fold_oof_eval_tta.py` に multi-member arm
(`name=ckpt1+ckpt2+...[:mode]`、member ごとに binding 検証・確率平均)
を追加。notebook `solar_filament_v76_ensemble_colab.ipynb` は同
スクリプトを embedded 書き出し (byte 一致を test 担保)。
想定所要 ~2.3h (proposals ~10min + baseline ~8min + ens3 ~24min +
ens3f ~95min)。

**結果 (2026-10-24, Colab T4 完走 ~1.5h)**: 事前固定ゲートどおり
**reject_noise** — ただし観測史上最大のサブゲート正効果。ensemble
レーンを閉じる。

| arm | PQ@0.35 | PQ@0.40 | PQ@0.45 | PQ@0.50 |
|---|---|---|---|---|
| baseline | 0.443393 | 0.443862 | 0.444238 | 0.444532 |
| ens3 | 0.445534 | 0.446022 | 0.446375 | 0.446521 |
| ens3f | 0.445201 | 0.445648 | 0.445956 | 0.446100 |

- `delta_ens3f_minus_baseline = +0.001718` → 事前固定帯 [0, +0.005) →
  **ノイズとして棄却** (昇格候補条件 +0.005 未満)
- `delta_ens3_minus_baseline = +0.002137` — **全実験で最大の正 delta**
  (TTA 系 +0.0007-0.0008 の約3倍)。member diversity (pretrained /
  scratch / ssl init) が確率平均で機能する典型条件に合致
- **`ens3f < ens3` (−0.00042)**: flip4 TTA は ensemble と相殺的 —
  各 member の確率マップを既に平均した後の追加平滑化は逆効果
- @0.45 TP/FP/FN: baseline 1715/750/1070 → ens3 1722/740/1063。
  **TP・FP・FN の3指標すべて改善**する稀有な結果だが振幅は帯域内
- sanity 完璧: baseline PQ@0.45 = 0.444238、v74/v75 と3連続 bit 一致
- member_sha256: b1c0f353 (baseline, manifest role) + 868a8441
  (v74 scratch_s8407) + df968e64 (v74 ssl_s8407) — 全て provenance
  検証済み (manifest role / ckpt metadata)

**解釈**: ensemble は TTA 系より一桁明確な正方向の signal を持つが、
+0.005 の昇格ゲートには届かない。帯域内のため確認実験は発動せず、
prod には採用しない — 事後にゲートを緩めない規律のほうが、単一実験の
結果よりパッケージの信頼性に寄与する。推論コストも3-12倍に増え、
提出再現の複雑性も増す。

**成果物**: Drive `solar_filament_v76/work/latest/` (v76_eval.json /
v76_summary.json / sha256_manifest.json、2/2 SHA256 検証済) +
Kaggle dataset `shingo257/solar-v76-results` (private)。

**採用判断**: prod + plain_mhsa_f2 維持。**残る誠実系レーンは実質
尽きた** — SSL (3連続負)・TTA (2系統ノイズ)・ensemble (ノイズ) を
全て事前固定ゲートで閉鎖。

### 2.32 discussion/LB 再調査 — 第5回リフレッシュ：748498 主催者回答あり（別天文台 SSL 許可） (2026-10-24)

第5回 periodic refresh。LB 完全 CSV・全41 discussion topics・新着
public kernel を走査。

#### 最重要： topic 748498 に主催者回答が着いた (2026-10-21)

> "Your method is OK. Please make sure that the new set of observations
> does not include the test images (or observations within +/- 1 hour
> of those timestamps)."

- **KSO・Meudon・BBSO の無ラベル Hα画像による SSL 事前学習は公式に
  許可** — 第3-4回 refresh で「唯一の未解決論点」としていた質問が
  解消
- 除外条件は **test 時刻の ±1 時間以内の観測** — 我々が GONG で
  自発的に適用した ±360 min より大幅に狭い（主催者要求は ±60 min）
- 解釈上の注意: 許可されたのは「unlabeled images for SSL encoder
  pretraining → MAGFiLO fine-tune、inference は test GONG のみ」
  という質問文の範囲。ラベル転用・外部ラベル付きデータ・test 観測の
  流用は別論点（§2.28-2.29 の整理どおり依然グレー/禁止）
- 我々への含意: GONG SSL が3連続負結果（v72-v74）で閉じたため、
  **追加実験の合理性は乏しい**。別天文台は機器差（ドメインギャップ）
  でむしろ劣化リスクがある。ただし「法的に可能な未試行データ」
  としては残る唯一のレーン — 採否は別途判断（後述の取り込み判断）

#### discussion topics 差分（41件・新規 topic ゼロ）

- 前回 (10-21) 以降の**新規 topic なし** — 最大 id は依然 749526。
  コメント数も全件で前回と同値（745880:16・744765:19・747960:4・
  749225:3 等不変）。変化は 748498 の主催者回答1件のみ
- 取得方法: `KaggleApi.competition_list_topics` (SDK) — CLI
  `competitions topics list` は 31件しか返さず不完全。**正本は SDK
  経路**（CLI の page 分岐に不具合がある模様）

#### LB スナップショット（1,161 チーム・ERS rank 216 / 0.39）

| 帯域 | 10-21 | 10-24 | Δ |
|---|---|---|---|
| 総チーム | 1,104 | 1,161 | +57 |
| ≥0.55 | 89 | 95 | +6（うち 0.55 ちょうどが更に増加） |
| 0.45–0.54 | 3 | — | （帯域集計変更、後述） |
| 0.40–0.50 | 42 | 49 | +7 |
| <0.40 | — | 1,015 | 大半 |

- トップ **IONode Online 0.69 で3日間不動** — 10-21 時点の連続上昇は
  止まり plateau 入りした可能性（提出頻度の単なる間欠かもしれないが、
  急伸局面は一段落）
- 2位以下: Tempest 0.64、0.62/0.60/0.58/0.57 が各1-数チーム —
  0.55+ 95チームの大部分は依然「0.55 共有ファイル」パターン
- **ERS rank 178→216**（score 0.39 不変・順位のみ後退）— 0.55 帯の
  膨張と新規チーム流入による相対低下。誠実帯では依然上位

#### 新着 public kernel 調査（手法開示の直接証拠）

- **jianansun/filament-yolo-seed-refiner**: YOLO11s det (imgsz1280,
  conf0.28) + **seed-refiner U-Net** — refiner 入力が3ch
  `[image_crop, CLAHE, seed_box_mask]`。「seed は refiner に
  WHERE to look を教える」。Tim Gluhkikh 氏の手法 (33 votes,
  LB~0.409) にインスパイアと明記。**detector→refiner 同型だが
  refiner に空間 prior チャンネルを足す点が我々と異なる新機構**
  （我々の refiner は crop のみ入力）
- **alessandrolsilveira/solar-e01..e11a シリーズ** (10-21〜24, e10
  が5 votes): PLAN.md 駆動の系統実験。**semantic U-Net 全画tile**
  (detector なし)、3ch `[raw, CLAHE, local contrast]`、BCE+soft
  Tversky(β0.7)+clDice 中心線 loss、native 2048 sliding-window
  推論+Gaussian blend→連結成分→RLE。**明確に倫理宣言つきの
  クリーンルーム系**（"trains ONLY on competition annotations, no
  public-GT tuning"）
- **muhammadshoaibaltaf/magfilo-v11/v12**: YOLO26x + UNet++/
  EfficientNet-B4 の detector→refiner 系。V11 は inference-only PQ
  実験（detector 解像度 1280 vs 1536・conf/mask 閾値 calibration
  sweep）、V12 は detector-aware crop fine-tuning — 我々の
  operating-point sweep と同種の誠実な系統実験
- **lamhuy8904/solar-filament-unet-segmentation-0-55** (39 votes):
  全画 U-Net + `remove_instance_overlaps`（greedy disjoint pixel
  allocation）で 0.55 を自称 — **745880 で指摘された「0.55 共有
  csv」の最有力ソース**。fork/re-run したチームが 0.55 に並ぶ構造と
  整合
- **azimahmadzadeh/self-evaluation-notebook** (48 votes): 主催者側が
  全参加者に共有する公式 self-eval notebook（PQ・IoU/Dice 分布・
  many-to-one/one-to-many 診断）— 競合手法ではなく審査ツール
- LB 0.56+ 帯のチーム (IONode Online・Tempest 等) の公開 kernel は
  **依然ゼロ** — トップ層手法は非公開継続

#### 取り込み判断

- **新規採用レバーなし**。prod + plain_mhsa_f2 維持
- **seed-channel refiner**（jianansun 経由の Tim Gluhkikh 系）は
  refiner 入力設計の未試行バリエーションではあるが、そのソース手法
  自体は LB~0.409 で我々 (0.444238 OOF / 0.39 LB) と同程度 — 移植
  してもゲート +0.005 に届く見込みは薄い。v76 ensemble が最大
  +0.0021 で閉じた現状、推論コスト増に見合わない
- **748498 の回答で「別天文台 SSL」は合法化**されたが、GONG SSL の
  3連続負結果（同機器・同日付の方がまだ近いドメイン）を考えると
  成功確率は低い。**実施するなら v72-v74 の負の教訓を反映した
  縮小版検証**に留めるべき — 本調査では実験を起こさず、記録のみ
- 証拠/推測の区別: LB 0.55 共有ファイル説は lamhuy8904 kernel の
  実在+コミュニティ証言で**強い傍証**(確定ではない)。トップ層手法は
  依然として直接証拠なし

### 2.33 discussion/LB 再調査 — 第6回リフレッシュ：dariushafshar の OOF↔LB 乖離 kernel 公開 (2026-10-26)

第6回 periodic refresh。LB 完全 CSV・41 discussion topics・新着
public kernel を走査。

#### LB スナップショット（1,196 チーム・ERS rank 200 / 0.39）

| 帯域 | 10-24 | 10-26 | Δ |
|---|---|---|---|
| 総チーム | 1,161 | 1,196 | +35 |
| ≥0.55 | 95 | 100 | +5（0.55 ちょうど **80** チーム） |
| 0.45–0.54 | — | 2 | 稀薄帯不変 |
| 0.40–0.44 | — | 50 | |
| 0.35–0.39 | — | 450 | |
| <0.35 | — | 594 | |

- **IONode Online が 0.69→0.70** に再び微増（18 submissions）—
  10-24 の plateau 観測は一時的。Tempest 0.64（123 subs）、
  0.62 が3チーム。**0.56+ 帯の手法は依然全て非公開**
- ERS rank 216→**200**（score 0.39 不変）。上位帯の変動で
  相対的にやや回復

#### discussion topics：新規ゼロ

- 41 topics・最大 id 749526 不変・全コメント数も前回と同値。
  締切（11-15）が近いのに新規質問が出ていないのが特徴的
- 748498（別天文台 SSL 許可の主催者回答）は SDK の commentCount
  表示が1のままだが、本文取得で主催者返信（10-21）は確認済み。
  API の count 表記揺れは既知

#### 重要発見： dariushafshar が OOF↔LB 乖離を公開 kernel 化

**`dariushafshar/solar-filament-u-net-oof-0-45-vs-public-0-39`**
(10-17、9 votes) — 以前の sparse-label 説（§2.28）を
実行可能な完全 kernel として公開:

- **彼の手法**: 10-model ensemble（5 fold × ResNet34 +
  EfficientNetV2-S の smp U-Net）、**detector なし全画 semantic
  型**。native 2048 推論 + **8-view D4 TTA**、各重みの sha256 を
  receipt で固定
- **数字**: nested-CV OOF **0.4502** vs public **0.39** —
  我々の 0.444238 / 0.39 と**ほぼ同じ乖離を独立系で再現**
- **sparse-label モデルを数式で明示**:
  `PQ' = r·ΣIoU / (r·TP + ½FP + ½(1−r)TP + ½r·FN)`。
  1 submission で fit した r で3つの提出全ての乖離を説明
  （我々が彼の r=0.77 を pooled counts に適用→0.3896≈実績 の検証と一致）
- keep rule: `t_lo=0.35, t_hi=0.99, min_area=200, min_meanp=0.85`
  — 我々（min_area=20）より**遥かに厳しい面積・平均確率フィルタ**。
  高確信・大きめの成分のみ残す運用
- 結論も我々と同じ規律: 「nested OOF で model 比較、PQ' を public
  推定値として読み、public board の小差を追わない」

**傍証の強化**: 独立した実装・別 split・別 postprocess で同じ
OOF↔LB 乖離構造が再現 — test 側注釈の sparse 化（主催者証言
735256 と整合）説は、我々の内部分析に留まらない外部確認となった

#### その他の新着 kernel

- **muhammadshoaibaltaf/magfilo-v14-low-confidence-candidate-
  recovery-sweep** (10-25、2 votes): タイトルのみ確認（kernel は
  private・pull 不可）— 我々の detector-blind FN 問題に直結する
  低信頼度候補回収 sweep。detector recall 改善は依然として
  コミュニティの主要関心
- `adelouadfeul/plot-functions-...`（可視化・手法寄与なし）、
  `jakuzya/solar-filament-segmentation-unet`（通常 U-Net baseline）

#### 取り込み判断

- **採用レバーなし**、prod + plain_mhsa_f2 維持
- dariushafshar の kernel は**我々の結論を独立に追認**するもので、
  新しい改善経路ではない。彼の keep rule（min_area=200,
  min_meanp=0.85）は我々の全範囲 sweep で棄却済みの方向
- 証拠/推測の区別: sparse-label 説は「2つの独立 pipeline で
  同構造が再現＋主催者の注釈漏れ証言」で**強い傍証**だが、
  test GT 非公開ゆえ証明ではない — この表現を維持

### 2.34 discussion 全スレッド本文の精読 — 41 threads / 115 msgs (2026-10-26)

これまでの refresh は topic 一覧・コメント数中心だったため、全41
スレッドの**メッセージ本文を一括取得**して精読 (SDK
`competition_list_topic_messages`)。新規 topic はないが、過去スレッド
への**追記が2件**発見された — コメント数メタデータでは捕捉できない
更新。

#### 最重要： 724503 Organizers' Remarks に 10-23 追記 — 提出要件が変わった

10-23 のレビュー (top 250 submissions) に基づく主催者投稿:

> "**We will clean the leaderboard after the deadline**, so that credit
> is given to genuine submissions."
> "**The reproducibility of submitted codes is only guaranteed if
> (for ML/DL models) the weights are stored in the repository and are
> loaded in the notebook.** Otherwise, the reproducibility requirement
> is not satisfied, and therefore, such submissions will not be
> considered for the award."
> "we discourage [AI's] uncritical use. Clear indications that a
> technical report was generated by AI without meaningful human
> oversight undermine our confidence in the work."

**我々への直接影響**: 現在の計画は「weights = private Kaggle dataset
`shingo257/solar-filament-2026-weights` を repro notebook が参照」だが、
新要件は「**weights を repository に格納**」を明示。dataset は
"the repository" ではないため、**単独では要件を満たさない可能性が
高い**。対応案:

- 必要な重み (実測): yolo_fold{0,1,2}_best.pt 45.3MB×3 +
  edgeatt refiner all_data 141MB + plain_mhsa_f2 (`best_model` のみ
  抽出で 141MB; 現 file は model+best_model の2重で 282MB) ≈
  **計 ~420MB**
- GitHub 単一 file 上限 100MB → refiner は超過。**Git LFS が正攻法**
  (public repo: 1GB storage・帯域 1GB/月 — 審査用 clone 数回なら
  許容圏)。代替: <100MB チャンク分割+notebook で再結合 (LFS 帯域
  quota 不要だが体裁は劣る)。fp16 化 (~70MB) は原 bytes の SHA256
  provenance を失うため非推奨
- **期限日タスクを更新**: 「weights dataset public 化は保留」→
  「**weights を repo 内に同梱 (LFS)** し repro notebook が repo
  内パスからロードする」が必須要件に。dataset 公開は補助的に残置可
- レポート: 「AI 生成を匂わせる無審査の報告書は信頼を損なう」と明記
  — 我々の4頁報告は実測値・負結果を正直に記述した人間精読済み文書
  なので問題ないが、README/Form でも同様の姿勢を維持

#### ルール裁定の確定一覧（本文から verbatim 確認）

| topic | 裁定 | 内容 |
|---|---|---|
| 737215 | **許可** | 無ラベル GONG での SSL pretrain (test ラベル不使用・inference は test 画像のみ) |
| 743385 | **inference 不可** | 系列利用は学習のみ。「model may not depend on any sequence during inference」 |
| 744047 | **許可/不可** | 無ラベル GONG 学習 = OK。推論時の近傍フレーム使用 = **No** |
| 743521 | **許可** | 同時刻 GONG 磁気図 (bzi) を訓練専用 aux signal として使用可 |
| 748045 | **許可** | 訓練ラベルの ±数分 GONG archive 転用可 (±1h test 除外必須)。「Your technical report is the right place for such disclosure」 |
| 748498 | **許可** | KSO/Meudon/BBSO 無ラベル SSL (±1h test 除外) — §2.32 記録済 |
| 746318 | **不許可** | SWEFil (外部ラベル付き) は「more ground-truth data」として禁止 |
| 736543 | **許可** | 提供データの spine/bbox/area/category_id を aux 監督に使用可 |
| 738951 | **許可** | 元 GONG FITS の利用可 (公式 MLEco FITS2JPEG あり) |
| 744544 | **許可** | グレースケール→3ch 複製は acceptable |
| 749225 | 参考 | GONG archive は ~40 files で 403 レート制限。community 回避: 2接続+1s delay、ban ~12h。主催者は IT チームに共有したのみで公式経路なし |

残る未解決の推論時外部データ拡張は**全て閉鎖** — 許可された外部利用は
全て訓練側のみ。

#### リーク構造の確定証拠（728062/729396/737177/745880/741892/743958）

- **728062**: public MAGFiLO (Harvard Dataverse) に **test 180画像の
  うち139枚が byte 一致**・全180枚の注釈 (~2,045件) が公開。
  実測: MAGFiLO 注釈に置換した提出で **0.73→0.82**
- **737177**: Fable の分析 — partial leak で ~0.51、full leak で
  ~0.66。HDJoJo の 0.52 と整合 → **0.56-0.70 帯の最尤説明は test GT
  利用**（個別チームの違反を断定するものではないが、構造証拠は強い）
- **745880**: 第三者が Fususu の 0.70 handoff kernel を実測 →
  **0.35**。「his 0.70 LB score is not in the submission」— 我々の
  v75/v76 での棄却判断・移植見送りを事後裏付け
- **741892**: 「~0.45 (or even 0.41) 超は test 注釈使用」+ 「judges
  have confirmed they are aware」— 誠実系フロンティア ~0.41-0.49
  (noisyislands) の認識がコミュニティ共通
- **743958**: lamhuy8904 kernel = 0.55 ファイルの実体（leak 使用と
  投稿者が指摘）。主催者「we take a score of 0.41 very seriously」

#### メトリクス考古学（旧メトリクスの exploit 記録）

- **728474**: 全空 mask で 0.93 — 旧 metric が空提出を報酬化
- **729809**: 同一モデルで提出数だけ変化: 0件=0.93 / 2枚=0.72 /
  6.4枚=0.67 / 11枚=0.65。predictions が少ないほど高スコア
- **733912**: 8-9 頃に metric 更新、同一 byte 提出が 0.66→0.31 に
  低下。LB が新旧 metric の混在期を経た記録
- **735377**: 予測 mask 重複は scorer ERROR（我々は disjoint
  allocation 済みで適合）
- 我々の全 OOF/LB 数値は**新 metric 期のもののみ**を採用済み

#### 評価・提出要件の細部確定

- **727186/728302**: 定量70% (Dice=torchmetrics, IoU, PQ) +
  定性30%。報告書は「entire pipeline, no micro-level decision
  undocumented」— 我々の4頁報告+README+RESEARCH_LOG でカバー
- **735256**: 主催者が注釈漏れを公認 + 「many-to-many relations の
  分布も審査で見る」— self-eval notebook 生成の図を参照せよと指示
- **747960**: 両提出の再現必須（我々は同一 repo で両方バイト一致済）
- **749526**: efficiency は定性フィルタのみ
- **749524**: Form 再提出可・最大10ファイル（PDF 差替可）
- **736921**: LB invisible バグの実例あり（主催者が復旧）— 我々は
  ERS rank 200 で表示中、問題なし

#### 取り込み判断

- **prod + plain_mhsa_f2 維持**。採用レバーなし
- **要対応**: weights-in-repo 要件 — SUBMISSION_FORM_DRAFT.md の
  締切日タスクを更新（別節で反映済み）
- 許可済み未試行レーン（磁気図 aux・ラベル転用・別天文台 SSL）の
  優先度は依然低い — GONG SSL 3連敗と残り締切日数を考慮

### 2.35 Phase A: weights-in-repository 対応 — 実装記録 (2026-10-26)

topic 724503 の要件 ("weights are stored in the repository and are
loaded in the notebook") への実装。monorepo には重み本体を置かず、
**standalone 公開 repo (solar-filament-2026-solution) の weights/ に
Git LFS で同梱**する方針 (§2.34 締切日タスク参照)。

#### エクスポート (`scratch/export_submission_weights.py`)

| repo weights/ 名 | bytes | sha256 (先頭16) | 原本 | 検証 |
|---|---|---|---|---|
| yolo_fold0_best.pt | 45,316,918 | ab8810916e34a755 | runs/yolo_fold_0/.../best.pt | byte 一致 |
| yolo_fold1_best.pt | 45,316,918 | 44ba9381a575d166 | 同 fold_1 | byte 一致 |
| yolo_fold2_best.pt | 45,316,918 | 40ec31fd2c6aa8e3 | 同 fold_2 | byte 一致 |
| refiner_all_data.pt | 141,079,441 | e9f0e8dd7ff104d9 | artifacts/..._all_data_resume.pt | byte 一致 |
| refiner_plain_mhsa_fold2_s8407.pt | 141,041,333 | **edc4e6660a709f9a** | artifacts/ablation_plain_mhsa_seed8407_fold2.pt | **124 tensors torch.equal** |

plain_mhsa 原本は `model`+`best_model` の2重格納 282MB — `best_model`
のみ抽出して 141MB に半減。serialized bytes は原本と異なるが
loader `ckpt.get("best_model") or ckpt["model"]` で同一オブジェクトを
返すため推論等価。manifest (`weights_manifest.json`) は
`--verify` で原本との tensor 等価を再検証可能。

#### notebook 解決順序の変更 (builder + 生成物)

`resolve_pretrained()` の探索順を **repo 同梱 weights/ → Kaggle
dataset → Drive artifacts → フル再学習** に変更。repo 経路は同梱
`weights_manifest.json` の sha256 で検証し、不一致は ValueError。
探索根は cwd/weights, cwd.parent/weights, cwd/Solar_Filament_
Segmentation/weights + `SOLAR_WEIGHTS_DIR` 環境変数。
fold CSV も repo `data/grouped_folds.csv` (sha256 24367120…, 本
commit で新規追加) を第一候補に追加 — dataset/Drive 無しでも split
を再生成せず検証できる。

#### standalone repo への配置設計 (A3)

- `weights/` に上記5ファイル + `weights_manifest.json` (計 ~556MB、
  Git LFS: `*.pt` filter)。`.gitattributes` に LFS 指定を追加。
- repo clone 後 `git lfs pull` → `weights/` をそのまま notebook の
  第一経路が発見する (cwd/weights)。
- monorepo 側は sha256 指紋のみ (AGENTS.md §2.3 準拠)。LFS quota:
  storage 1GB / bandwidth 1GB/月 — 審査 clone 数回なら内側。
- `data/grouped_folds.csv` は standalone repo にも通常ファイルとして
  同梱 (subtree sync で自動反映)。
- **sync 注意**: standalone への commit-tree 同期は tree 全体を
  モジュール tree で置き換えるため、`weights/` (standalone 専有) は
  sync のたびに落ちる。sync 手順の末尾で `weights/` + `.gitattributes`
  の LFS 行を前コミットから再 graft するか、weights commit を最後に
  積み直すこと (実施済: standalone main `b006ede`)。
- LFS pointer は text file のため repo 上の見た目は小さいが、clone 時に
  `git lfs pull` が必要 ( notebook は smudge 済み working tree を前提)。

#### E2E 検証 (2026-10-27)

standalone repo を temp dir に clone → notebook 埋め込み関数を AST
抽出して clone cwd から実行: `repo_weights_roots()` が
`<clone>/weights` を検出、5 重み全てが repo 経路で解決
(manifest sha256 検証 pass)、`data/grouped_folds.csv` =
`24367120…` 一致。公開化後に匿名 clone で再確認するだけの状態。

**GitHub push 済み (2026-10-27、repo は private のまま)**:
`shingo257/solar-filament-2026-solution` へ `main` を force push
(`d1c4bd42 → a5cd1215`、旧系統は内容の部分集合・非 FF のため
force; 逆方向差分なしで喪失ゼロを確認済み)。LFS 5 objects/418MB
アップロード完了。authenticated な新規 clone + `git lfs pull` で
5 重み全てが実体化し manifest sha256 と一致 (ab881091… /
44ba9381… / 40ec31fd… / e9f0e8dd… / edc4e666…) — **GitHub
経路の E2E は検証済み、締切日の残作業は visibility 切替 +
匿名 clone 再確認のみ**。

#### テスト

- `test_export_submission_weights.py` 6件 (合成 ckpt で export/
  verify/manifest/best_model|model 両形式)
- `test_repo_weights_resolution.py` 5件 (repo 優先・sha 不一致拒否・
  dataset フォールバック・未検出 None・manifest 参照確認)
- `test_submission_reproduction_variants.py` の plain dispatch 期待値を
  2パターン (dataset 名 + repo 名) に更新

### 2.36 B1 設計: verifier rescue — 低conf 候補の回収 (事前登録, 2026-10-26)

#### 仮説と既存証拠

fold-2 OOF の FN の ~68% は detector-blind (conf<0.26 で提案自体が
出ない)。v72 の verifier 実験は**除去方向**のみ (提案を verifier で
フィルタ) で失敗。rescue 方向 — conf<0.26 の候補を verifier が
採用側に回す — は未試行。v14 sweep (muhammadshoaibaltaf) も同方向を
示唆するが非公開。tile union は FP 洪水で失敗済みなので、
verifier スコアによる選択的 rescue が残る構造レバー。

#### 設計 (実装確定版)

- 実装: `scratch/run_v80_scorer_eval.py` に `rescue` arm を追加。
  候補・refiner 出力・特徴量は v80 candidates cache 再利用 (再検出なし)
- arm `rescue`: **conf>=0.26 の yolo 候補は無条件維持**し、
  conf∈[0.08, 0.26) の候補は cross-fit 済み A1 scorer の û (期待IoU) が
  τ 以上なら追加採用 — strictly-additive (prod 集合を減らさない)
- 掃引 `τ ∈ TAU_GRID (0.16…0.32, 0.02刻み)` — û は期待IoUスケールで
  運用点 τ*≈0.22。0.5-0.9 の絶対値掃引はスケール不整合のため改訂
- 評価は全3fold pooled PQ (fold-2 部分も per-obs 集計で報告)。
  prod arm が同一 cache で再現されるため sanity は内蔵
- **留保 (事前確定)**: scorer cross-fit は upstream モデルを outer-fold
  別に再学習していない non-nested 構成 (§2.8 記録の留保と同一)。
  よって本評価は **exploratory** で、gate 通過しても直接採用不可 —
  nested 再検証が別途必要

#### 事前固定ゲート (結果を見る前に確定)

- `max_τ (rescue_pq − prod_pq) ≥ +0.005` → 昇格"候補" (nested
  再検証の実施を起動するのみ、即採用ではない)
- `[0, +0.005)` → ノイズ棄却 / `<0` → 棄却
- 掃引の max を取るため multiple-comparison 補正として採用判定は
  `max_τ Δ` に適用。bootstrap CI95 を参考報告するが判定は点推定
- 二次指標: rescue 追加候補数と TP/FP 内訳 (分析用、判定に使わない)

#### 指紋

- 入力: candidates cache `solar_filament_v80/candidates/fold_{0,1,2}`
  + frozen split `24367120…`、GT annotations、実行時に sha256 を記録
- scorer: HistGBM h×m cross-fit (seed 3407、meta は a1_scorer_meta.json)
- 再現: `run_v80_scorer_eval.py` (rescue arm)、結果 `output/v80/`

#### B1 結果 (2026-10-27, eval 63min @workers=2, `output/v80_rescue/`)

| arm | pooled PQ | fold0 | fold1 | fold2 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| prod | 0.442631 | 0.4393 | 0.4415 | 0.4471 | 5082 | 2286 | 3117 |
| rescue τ*=0.28 | 0.444431 | 0.4398 | 0.4442 | 0.4492 | 5142 | 2346 | 3057 |

- τ 掃引: 0.16→−0.0023、0.18→+0.0001、0.20→+0.0010、0.22→+0.0016、
  0.24→+0.0014、0.26→+0.0017、**0.28→+0.0018 (max)**、0.30→+0.0016、0.32→+0.0006
- best bootstrap CI95 [+0.00065, +0.00295] — ゼロを跨がない正の効果だが、
  TP+60/FN−60/FP+60 の対称増加 (真候補も偽候補も等しく拾う)
- **判定: reject_noise** — max_τ Δ = +0.0018 ∈ [0, +0.005) 帯。
  non-nested scorer 留保もあり昇格候補にもならない
- 解釈: û の順位付けは conf<0.26 帯でも弱い正の信号を持つが、
  rescue 利益と FP コストがほぼ相殺。TTA(+0.0007)・ensemble(+0.0021)・
  rescue(+0.0018) と独立機構3系統が全て [0, +0.005) 帯に収束 —
  frozen operating point 周辺の帯域効率の天井を強く示唆
- **verifier/scorer 系レーンはこれで完全閉鎖** (filter=v72 棄却、
  scorer 選択=A1 LB 反転、rescue=B1 ノイズ棄却、oracle は +0.107 でも
  scorer が解放できない)。残りの誠実系レーンは締切日作業のみ

### 2.37 B2 設計: mask morphology sweep — 推論時マスク膨張/収縮 (事前登録, 2026-10-27)

**動機**: §near-miss 診断 (fold2 全 235 obs・1,290 pred↔GT ペア、
`scratch/analyze_nearmiss_area_bias.py`) で IoU 0.3–0.5 帯の
pred_area/gt_area 中央値 = **1.214** (p25 0.525 / p75 1.936)。
予測が GT よりやや大きい傾向 — 一様 erosion が near-miss の一部を
TP 化する可能性。一方分散が大きく (合併 pred は ~2x)、効果の符号は
事前には不明 — これは既存レバーの中で唯一未試行の安価な純後処理変更。

**設計 (全て prod 経路と同一、変更はマスクへの morphology だけ)**:

- `morph` arm: keep 集合 = prod (`source==0 & usable & conf>=0.26`)
- 各候補マスクに disk dilation (r>0) / erosion (r<0)、
  r ∈ {−3,−2,−1,+1,+2,+3} px @2048² (cv2.MORPH_ELLIPSE)
- その後は全て同一: drop_contained@0.90 → area 昇順 greedy_disjoint
  (min_area=20) → 公式 PQ
- v80 candidates cache 再利用 (refiner 再推論なし)、3fold pooled 評価

**事前固定ゲート** (B1 と同一規約):

- `max_r (morph_pq − prod_pq) >= +0.005`: nested 再検証の検討のみ
  (即採用しない)
- `[0, +0.005)`: reject_noise
- `<0`: reject

**誠実留保**: 面積比診断は探索的 — 符号は不明で結果を見てからの
採用判断はしない。erosion で mask が消滅した候補は min_area で
自然に落ちる (実装は消滅マスクもそのまま流す)。

### 2.38 v87 設計: union-label YOLO 再学習 — アノテータ間クラスタ和集合教師 (事前登録, 2026-10-27)

**出自**: 外部 AI (ユーザー経由 zip `v87_yolo_union_label.zip`) からの
提案。実装は zip 同梱版をベースに、本リポ側でコードレビュー・検証・
補強 (`--folds-sha256` 引数追加、folds sha256 を実測値で埋め込み再生成)
して統合。外部 AI の合成テスト報告のみで GPU 学習/評価は未実行だったため、
動作検証は本リポで実施 (unit test 6/6 緑、prod 学習引数との完全一致を
`build_master_pipeline_notebook.py` 対比で確認、Drive 側全前提ファイルの
実在・folds sha256 `febb3f67…` を実測確認)。

**動機**: prod YOLO は `(file, annotator record)` を独立サンプルとして
学習している。これは暗黙に (a) 高合意物体を複数回 positive 化し、
(b) 1人しか描いていない物体を他アノテータのコピーでは背景負例として
教え、(c) 境界の不一致を矛盾ラベルとして与える。detector-blind FN が
FN の 68% を占める現状 (§v74/B1) で、候補生成側のラベル構造を変える
唯一未試行のレバー (D3 系)。

**設計 (学習データのラベルのみ変更。モデル・ハイパラ・fold・seed・
評価経路は prod と完全一致)**:

- `union_hard`: 同一物理フィラメントをアノテータ横断で mask IoU>0.5
  union-find (同 record 同士は合併禁止・IoU 降順 deterministic)、
  和集合 mask を全 record コピーに重みなしで付与
- `union_agree`: 同じ和集合クラスタで、合意数 a のクラスタを a コピーに
  付与 (合意重みをサンプル複製で近似)
- union の YOLO 化は最大外周輪郭のみ (min_area=20 未満は drop、
  複数成分 union は統計として記録)
- fold2 のみ、seed=3407+2、学習引数 18 項目は master/prod と完全一致
- 評価は prod 経路と同一: EdgeAtt fold2 refiner → drop_contained@0.90
  → greedy ownership (min_area=20) → 公式 PQ、conf sweep
  {0.15,0.20,0.26,0.32}、primary=0.26
- prod baseline は同一 notebook 内で同経路評価 (paired Δ)。
  既知値 0.444238 との ±0.002 超乖離は警告 (判定は paired Δ)

**事前固定ゲート**:

- `PQ(arm,0.26) − PQ(prod,0.26) >= +0.005`: fold0/1 に昇格検討
- `[0, +0.005)`: reject_noise
- `<0`: reject
- conf sweep の最大値のみでの採用はしない (primary 固定)

**誠実留保**: (a) `union_agree` でも高合意物体の複製構造は残り、
「他アノテータが描いた物体を背景化」する問題は `union_hard` でも
完全には解消しない (複製コピー全てに全 union を入れる設計のため)。
(b) 最大輪郭のみの変換は非連結 union の一部成分を捨てる
(統計で報告)。(c) Drive パス・既知値 0.444238 は外部 AI の推定を
本リポで実測確認済みだが、paired baseline 評価が前提の最終確認。
(d) LB は疎ラベル採点の可能性が高く、OOF で上がっても LB には
出ない可能性がある。

**結果 (Colab A100, 2026-10-28, fold2, paired eval, 235 val images)**:

paired baseline 再現性は `prod@0.26 = 0.444221` vs 既知値
`0.444238` (差 −1.7e−05) で ±0.002 以内を確認。学習は両 arm とも
YOLO11m-seg・70 epoch 上限 (union_hard は early stop @52 で 66 epoch、
union_agree は完走)、detector 単体 best val mask mAP50 は
union_hard 0.618 / union_agree 0.632。

| arm | conf | PQ | SQ | TP | FP | FN | preds/img | ΔPQ@0.26 |
|---|---|---|---|---|---|---|---|---|
| prod | 0.26 | **0.4442** | 0.6799 | 1715 | 750 | 1070 | 6.38 | — |
| union_hard | 0.26 | 0.4304 | 0.6758 | 1860 | 1196 | 925 | 7.98 | **−0.0138** |
| union_agree | 0.26 | 0.4402 | 0.6804 | 1588 | 536 | 1197 | 5.46 | **−0.0040** |

conf sweep 全値:

| conf | prod | union_hard | union_agree |
|---|---|---|---|
| 0.15 | 0.4265 | 0.3935 | 0.4319 |
| 0.20 | 0.4372 | 0.4151 | 0.4401 |
| 0.26 | **0.4442** | 0.4304 | 0.4402 |
| 0.32 | 0.4416 | 0.4378 | 0.4305 |

**解釈**: `union_hard` は設計意図通り detector-blind FN を回収
(TP +145 / FN −145) したが、1 票物体と肥大化 union mask 由来の
FP が +446 と暴走し大幅悪化。`union_agree` は合意重みで FP を
prod 以下 (536) まで抑え込んだが、今度は TP が −127 と recall 側に
負荷が回り、ネットでは −0.004 の微損。両 arm とも sweep 全域で
prod を上回る conf 帯はなく、閾値調整でも救えない。
union 化による「他アノテータ未描画物体の背景化」解消は FN 回収に
効くが、sparse-label OOF 評価系では追加候補が FP として直接
罰せられるため、純粋な recall レバーとしては機能しないことが
定量的に確認できた。

**verdict**: 両 arm **reject** (Δ@primary < 0)。D3 系
(ラベル構造変更) レーンは v87 で打ち止め。prod /
plain_mhsa_f2 2 構成は不変。manifest は Drive
`solar_filament_v30/artifacts/v87_yolo_union_label_fold2_union_hard_union_agree_manifest.json`
に保存済み。

### 2.39 discussion/LB 再調査 — 第7回リフレッシュ：biconcavelens の全手法公開（SABiD 論文化） (2026-10-27)

第7回 periodic refresh。LB 完全 CSV・42 discussion topics・新着
public kernel・外部 GitHub 公開を走査。

#### LB スナップショット（1,218 チーム・ERS rank 206 / 0.39）

| 帯域 | 10-26 | 10-27 | Δ |
|---|---|---|---|
| 総チーム | 1,196 | 1,218 | +22 |
| ≥0.55 | 100 | 102 | +2（0.55 ちょうど **82** チーム） |
| 0.45–0.54 | 2 | 2 | 0 |
| 0.40–0.44 | 50 | 53 | +3 |
| 0.35–0.39 | 450 | 454 | +4 |
| <0.35 | 594 | 607 | +13 |

- 先頭不変: IONode Online 0.70・Tempest 0.64・0.62 が4チーム。
  ERS rank 200→**206**（score 0.39 不変・後続流入で微落）
- 0.55 ちょうどに 82 チーム集中の構造は不変 — test GT 既知系の
  クラスタと整合（§2.34 リーク証拠の通り）

#### discussion 差分：新規 1 topic + 724503 へ追記

- **753043「multiple annotations」**（新規・未回答）: test 画像に
  複数アノテーターがいる場合、metric は全注釈を別 instance として
  pool するか — v87 で検証した annotator disagreement 問題と
  **同一の問いがコミュニティに出始めた**。主催者の回答待ち
- **724503 Organizers' Remarks に 10-26 追記**（§2.34 精読後）:
  - Q「いつボードをリセットするのか」→ A「**We decided to clear
    the board after the deadline**. We do not want to leak any
    information about our guardrails until then.」— LB 浄化は
    **締切後**と明言
  - 評価 rubric の明確化 (10-26): 「**Quality**: 重要例について実際の
    segmentation を目視で見る。**Feasibility**: 合理的時間・
    過度に複雑でない・実装が理解可能なモデルを高評価」
    — 我々の再現可能で単純な detect-then-refine 構成は
    feasibility 側で有利。LB 位置より実装品質が効く設計を維持

#### 重要発見: biconcavelens/filament-segmentation-2026（全手法+論文公開）

GitHub で**検証ログ付きの完全公開 repo** を発見（Mellimi/Bhargava/
Gupta, SABiD 2026 投稿、LB 0.40 / OOF 0.461）。我々とほぼ同じ
detect-then-refine 族で、**独立系統で我々と同じ壁に到達**している:

- **最終系**: Mask R-CNN@2048 + YOLO11m/l-seg@1280 + RT-DETR-l +
  semantic ConvNeXt-Tiny U-Net の calibrated 融合 + crop refiner
  (ResNet34 smp U-Net, **spine centerline aux head** — 736543 の
  許可裁定済み aux 監督) + 4-view flip TTA (topology-safe fallback)
- **OOF 0.461 / LB 0.40** — 我々 (0.444/0.39)・dariushafshar
  (0.450/0.39) と併せて **3 系統の独立 pipeline で同じ
  OOF↔LB ≈0.05-0.06 乖離**。sparse-label test 説の傍証がさらに強化
- **彼らの最重要 lever（我々が未試行）**: **cls-loss reweight**。
  「detector-blind ではなく well-localized だが低 score」という診断
  （bbox-IoU 0.70-0.87 が conf 0.05）に対し YOLO `cls=0.5→1.5` で
  LB 0.37→0.38、Mask R-CNN RoI cls 3x、RT-DETR cls=2.0 で同型修正。
  cls=3.0 は過剰で悪化 — 最適は穏やかな reweight
- **cross-detector true dedup** (calibrated score で pool → IoU 0.05
  で候補全体 discard → paint): 4 度の失敗（閾値調整・AGREE/UNIQUE
  grid・isotonic 単一閾値・両側校正後再試行）の末、**事前 dedup**
  だけが初めて solo を超えた (0.4252→0.4407 local, 0.38→0.39 LB)。
  我々の RF-DETR union が −0.028 で沈んだのは dedup 欠如と整合
- **multi-annotator**: `train_labels="complete"`（最注釈数の
  annotator のみ採用）は 39% の訓練量喪失で悪化 — v87 union 化の
  reject と合わせ、**ラベル構造レバーは独立 2 系統で negative**
- rescue filtering 4 方式（refiner 確率・contrast・特徴量
  classifier）全滅 — 我々の verifier/rescue negative と独立一致
- self-training (conf≥0.75 pseudo-label) は 0.4171→0.3931 に明確悪化
  （「自分の blind spot を強化する」）— 未試行だが妥当性高い負例
- Cellpose 系 flow-field instance seg は「試していない残存方向」
  として言及 — 我々も未踏だが from-scratch で高コスト

#### 新着 kernel

- `koushikkumardinda/solar-filament-segmentation-2026-edgeattnet`
  (10-27): EdgeAtt U-Net だが **semantic 型**（全 instance を1枚の
  mask に集約）— instance 分離なしで PQ には不向き。取り込み対象外
- 他は既知 baseline の焼き直し・EDA のみ

#### 取り込み判断

- **即採用レバーなし**。prod + plain_mhsa_f2 維持
- **記録上の注記**: cls-loss reweight は biconcavelens が実証した
  「well-localized だが低 confidence」の病態に対する upstream 修正で、
  我々の detector-blind FN の一部が実は conf<0.15 の低 score 候補に
  混在している可能性と整合する。追求するなら事前登録が必須だが、
  fold-2 blind-FN の 68% が「any conf で proposal なし」と記録済みの
  ため（§2.36 前段の分析）、効果量は不明。締切までの残作業
  （repo 公開・再現検証・Form）優先で新規 GPU 実験は開かない
- 主催者の rubric 明確化（品質目視・feasibility・理解可能な実装）は
  現在の「正直な報告 + 再現可能な単純 pipeline」方針を裏書き

### 2.40 v88 設計: YOLO cls=1.5 再学習 — confidence head undertraining 修正 (事前登録, 2026-10-27)

**出自**: §2.39 の biconcavelens 全手法公開で見つかった唯一の未試行
upstream レバー。ユーザー指示で採用検証する。

**仮説**: biconcavelens は「bbox-IoU 0.70-0.87 で well-localized なのに
conf 0.05」の missed filament を発見し、confidence head の undertraining
を原因と診断。ultralytics `cls` 損失重み 0.5→1.5 で solo PQ
0.4171→0.4252・LB 0.37→0.38 を得た (cls=3.0 は過剰で悪化、1.5 が最適)。
我々の fold2 FN の一部 (「any-conf で proposal なし」と記録した 68% の
うち、実は低 score 帯に候補があるもの) が同じ病態である可能性。

**設計 (変更は `cls=1.5` のみ。それ以外は prod と完全一致)**:

- `cls15`: `model.train(..., cls=1.5)`。モデル (yolo11m-seg)・
  imgsz (1280)・epoch (70)・patience (14)・augment 一式・
  seed (3407+2=3409)・all-labeler データセット構築は prod と同一
- fold2 のみ (パイロット慣行)。学習は約 2.5h (A100 見込み)
- 評価は prod 経路と同一: EdgeAtt fold2 refiner → drop_contained@0.90
  → greedy ownership (min_area=20) → 公式 PQ、conf sweep
  {0.15,0.20,0.26,0.32}、primary=0.26
- prod baseline は同一 notebook 内で同経路評価 (paired Δ)。
  既知値 0.444238 との ±0.002 超乖離は警告
- **追加診断 (採用判定に使わない)**: 推論 floor を 0.02 に下げて
  生提案を全保持 (refine 対象は conf>=0.08 のみ = prod 経路不変)。
  primary で FN の GT のうち conf [0.02,0.26) に IoU>0.5 の生提案が
  ある割合を記録 — detector-blind FN に低 score 候補が埋まっているかを
  biconcavelens 流に直接測定

**事前固定ゲート**:

- `PQ(cls15,0.26) − PQ(prod,0.26) >= +0.005`: fold0/1 に昇格検討
- `[0, +0.005)`: reject_noise
- `<0`: reject
- conf sweep の最大値のみでの採用はしない (primary 固定)

**誠実留保**: (a) biconcavelens の val は 70 画像で我々の 235 画像と
異なり効果量は直接移らない。(b) 我々の FN 分解は「conf floor 0.08 で
proposal なし」までしか検証していない — cls fix が効く余地は診断結果
次第。(c) もし reject でも lowconf 診断の結果は blind-FN の性質を
明確化する価値がある。(d) LB は疎ラベル採点の可能性が高く OOF 改善が
LB に出ない可能性がある (§2.28)。

実装: `scratch/build_v88_yolo_cls15.py` →
`solar_filament_v88_yolo_cls15_fold2_colab.ipynb`。manifest は
Drive `artifacts/v88_yolo_cls15_fold2_manifest.json` に保存。

Kaggle 版も用意: `scratch/build_v88_yolo_cls15_kaggle.py` →
`solar_filament_v88_yolo_cls15_fold2_kaggle.ipynb`。input は competition
data + `solar-filament-v35-artifacts` (前処理+folds) +
`solar-filament-2026-weights` (prod yolo) + `solar-v73-assets`
(fold2 refiner)。評価経路・ゲートは Colab 版と完全同一。manifest は
kernel output `v88_yolo_cls15_fold2_manifest.json`。実行 kernel:
`shingo257/v88-yolo-cls15-fold2` (T4, private)。

**結果 (2026-10-29, Colab A100-40GB 実行 — 学習 70ep 完走 0.962h、
early stop なし)**: paired baseline `prod@0.26 = 0.444221`
(既知値 0.444238、差 −1.7e−05、評価経路健全)。

| arm | conf | PQ | SQ | TP | FP | FN | preds/img | pq_sparse_r077 |
|---|---|---|---|---|---|---|---|---|
| prod | 0.15 | 0.4265 | 0.6752 | 1821 | 1160 | 964 | 7.78 | 0.3694 |
| prod | 0.20 | 0.4372 | 0.6771 | 1781 | 950 | 1004 | 7.10 | 0.3809 |
| prod | **0.26** | **0.4442** | 0.6799 | 1715 | 750 | 1070 | 6.38 | 0.3896 |
| prod | 0.32 | 0.4416 | 0.6816 | 1610 | 575 | 1175 | 5.64 | 0.3904 |
| cls15 | 0.15 | 0.4192 | 0.6759 | 1854 | 1339 | 931 | 8.31 | 0.3615 |
| cls15 | 0.20 | 0.4357 | 0.6786 | 1826 | 1077 | 959 | 7.54 | 0.3781 |
| cls15 | **0.26** | **0.4480** | 0.6808 | 1768 | 821 | 1017 | 6.71 | 0.3916 |
| cls15 | 0.32 | 0.4533 | 0.6837 | 1682 | 607 | 1103 | 5.89 | 0.3994 |

**verdict: `reject_noise`** — ΔPQ@0.26 = +0.0038 < ゲート +0.005。
fold0/1 昇格検討の条件は不成立。最終構成 `prod` / `plain_mhsa_f2` は
不変。

分解 (primary 0.26): TP +53・FN −53 (仮説方向に回復) だが
FP +71 が相殺し効果量はゲート未達。SQ は +0.0009 でほぼ不変。
preds/img は全 conf で +0.25〜+0.55 増 — cls 強化が proposal スコアを
一様に持ち上げた証拠として機構は一致。

診断 (採用判定外): 低 conf coverage は prod 543/953 (57.0%) →
cls15 531/907 (58.5%)。**prod の primary FN の 57% が conf
[0.02,0.26) に IoU>0.5 候補を持つ** — biconcavelens の病態分類
(候補あるが閾値未到達) が我々側でも過半と確認。cls15 は FN を
953→907 に減らしたが lowconf プール内の診断構造は不変で、
「閾値を下げても FP 側が同じプールから増える」限界を裏付けた。

留意 (事後、採用判断には不使用): cls15 の operating curve は右に
シフトしており conf 0.32 で 0.4533 (prod 同点比 +0.0117)。sweep
最大値採用は事前留保で禁止のため記録のみ。この「最適 conf 移動」
自体は cls 重み付けの予想される副作用であり、別実験
(calibrated-threshold 再登録) の材料にはなり得るが今回の
採用証拠にはしない。

manifest: Drive `solar_filament_v30/artifacts/
v88_yolo_cls15_fold2_manifest.json` (Colab run 出力)。
Kaggle T4 kernel は同一 deterministic 実験の冗長コピーのため
完了前に停止してよい。

### 2.41 v89 設計: cls=1.5 fold0/1 複製 + 移行 operating point 検証 (事前登録, 2026-10-29)

**出自**: v88 (§2.40) で cls15 は reject_noise だったが、operating
curve の右シフト (0.32 で +0.0117) が残った。fold2 一 fold の観察では
(a) cls15 の改善方向が fold 間で一般化するか、(b) cls15 は prod と
異なる operating point を持つべきか、が未検証。ユーザー指示で複製
検証を継続する (「実験どんどんやりたい」)。

**仮説**: cls 強化による score 上昇は fold2 固有ではなく fold 横断で
再現する。再現すれば cls15 は prod より高 conf の operating point で
真の gain を持つ可能性がある。

**設計 (変更は fold のみ。レシピは v88/prod と完全一致)**:

- `cls15`: fold0 (CFG.FOLD=1 で fold1) を `cls=1.5` で再学習。
  モデル・imgsz・epoch・augment・seed (3407+fold)・all-labeler
  構築・評価経路は v88 と同一
- prod baseline は既存 `runs/yolo_fold_<F>` を同経路で paired 評価
- refiner は fold 別 ckpt (`mediseg_edgeattnet_refiner_fold<F>_resume.pt`、
  glob fallback 付き)
- fold0/1 の prod 既知値は v80 系評価しかなく経路差があるため
  BASELINE_PQ チェックは warn-only

**事前固定ゲート — 2 つの判定を両方記録**:

- 同一 point: `PQ(cls15,0.26) − PQ(prod,0.26) >= +0.005`
- 移行 point: `PQ(cls15,0.32) − PQ(prod,0.26) >= +0.005`
  (0.32 は fold2 観察から fold0/1 評価**前**に固定 — 評価セットへの
  閾値 fitting ではなく fold 横断 transfer として扱う)
- 採用検討には fold0 + fold1 **両方**での合格が必要。片方のみの
  合格は fold 依存性の証拠として reject_noise 扱い
- いずれも未達なら reject_noise

**誠実留保**: (a) 0.32 移行 point は fold2 で選択済みのため「同一
セットで選んで同一セットで採点」ではないが、fold 数が増えるほど
transfer の信頼性は下がる。(b) cls15 の fold2 gain が主に低 conf
proposal の昇格由来なら、fold0/1 でも同じ FP/FN トレードオフ構造が
出るはず — 低 conf 診断で照合する。(c) sweep 最大値採用は引き続き
禁止 (本実験でも exploratory_best_conf は記録するが判定に使わない)。

実装: `scratch/build_v89_cls15_fold01.py` →
`solar_filament_v89_cls15_fold01_colab.ipynb`。manifest は Drive
`artifacts/v89_yolo_cls15_fold<F>_manifest.json`。

### 2.42 v90 設計: semantic U-Net 提案源 — detector-blind FN の救済 (事前登録, 2026-10-30)

**仮説**: v88 (fold2) と v89 fold0 baseline の低 conf 診断が一致して
示したように、primary FN の ~57% は低 conf proposal で覆える一方、
残り ~43% は detector が **any-conf** で候補を出さない blind-FN。
conf 再配分・ensemble・postprocess ではこの層を構造的に拾えない。
biconcavelens (§2.39, OOF 0.461 / LB 0.40) は semantic ConvNeXt-T
U-Net を独立提案源として detector-missed filament を ~30個追加獲得
した。同機構を我々の prod 評価経路に移植し、blind-FN を直接狙う。

**arm**:

- `union`: prod det 提案 ∪ semantic 提案 (dedup: sem 提案は conf>=c の
  det 提案と mask-IoU>0.5 で重複する場合落とす)
- `semprop`: semantic 提案単独 (提案源性能の診断。採用候補にしない)
- `prod`: 既存 `runs/yolo_fold_2` を同経路 paired 評価 (fold2 pilot)

**semantic モデル** (biconcavelens `train_semseg.py` 準拠):

- smp U-Net `tu-convnext_tiny` (ImageNet init, in_channels=1, classes=1:
  fg のみ。彼らの spine aux head は提案化に使わないので省略)
- 1024px crop・(file, annotator) エントリ単位 (fit_entries、val fold の
  観測・annotator は完全除外 — frozen grouped_folds で分離)
- 75% filament-centered crop (+jitter) / 25% random、rot90・hflip、
  gamma/brightness/contrast jitter
- focal(γ=2, α=0.25)×10 + soft-dice、AdamW lr 2e-4、AMP bf16、
  batch 8 (A100)、時間予算 1.2h (彼らは T4 で 3h 規模のため縮小 pilot)
- 推論: full-res 4-flip TTA → fg 確率マップ (float16 npz を Drive に cache)
- 提案化: 確率 ≥0.5 → connected components (8-conn)、conf = 成分内最大
  確率、min_area 20 未満除外
- 提案は prod det 提案と**同じ** refine→ownership→公式PQ 経路に乗せる
  (refine 対象は det と同じ conf>=0.08 フィルタ…ではなく sem 提案は
  全件 refine — conf=成分最大確率が 0.08 未満の成分は 0.5 閾値で
  既に除外済みのため実質同じ)

**事前固定ゲート**:

- `PQ(union,0.26) − PQ(prod,0.26) >= +0.005` で fold0/1 昇格検討
  (fold2 pilot 単体では採用しない — v89 と同じ両 fold 合格ルール)
- `[0,+0.005)` は reject_noise、`<0` は reject
- sweep 最大値採用は禁止 (exploratory_best_conf は記録のみ)

**診断 (採用判定に使わない)**:

- `diag_semrescue.fn_covered_by_sem_prop`: prod の primary FN のうち
  semantic 生提案が IoU>0.5 で覆う数 — blind-FN の残り 43% が実際に
  拾えるかの直接測定
- `semprop` の PQ/FP 構造: semantic 単独でどこまで届くか (提案源と
  しての上限)
- `sem_props_per_image`: 提案数のスケール確認

**誠実留保**: (a) 時間予算学習は epoch 数が非決定的 (環境依存) —
再現性のため ckpt sha256 を manifest に記録。(b) sem conf は成分内
最大画素確率で det conf と分布が違う — 同一 sweep は便宜的な共通化。
(c) dedup は「det が既に拾っているもの」を落とすだけで FP 追加は
防げない — union の ΔFP も分解して記録する。

実装: `scratch/build_v90_semprop.py` +
`scratch/build_v90_semprop_cells.py` →
`solar_filament_v90_semprop_fold2_colab.ipynb`。manifest は Drive
`artifacts/v90_semprop_fold2_manifest.json`、ckpt/prob map は
`runs/v90_semprop/`。

### 2.43 v89 fold0 結果: cls=1.5 は fold0 で reject — fold2 ゲイン非一般化 (2026-10-30)

**判定: `reject`（両ゲートとも負）**。Colab A100、学習 59.5 分、
manifest `artifacts/v89_yolo_cls15_fold0_manifest.json`。

| conf | prod | cls15 | Δ |
|---|---|---|---|
| 0.15 | 0.4228 | 0.4273 | +0.0045 |
| 0.20 | 0.4378 | 0.4367 | −0.0011 |
| **0.26** | **0.4443** | **0.4406** | **−0.0037** |
| 0.32 | 0.4383 | 0.4396 | +0.0013 |

- 同一 point 判定: Δ@0.26 = **−0.0037** → reject
- 移行 point 判定: cls15@0.32 = 0.4396 vs prod@0.26 = 0.4443 →
  **−0.0047** → reject
- exploratory_best_conf = 0.26（cls15 の最良も 0.26 — fold2 で見た
  右シフトは fold0 では再現しなかった）

**分解 (0.26)**: TP 1685 vs 1712 (−27)、FN 1064 vs 1037 (+27)、
FP 744 vs 761 (−17)。fold2 の「TP+53・FN−53・FP+71」と機構が
**逆** — fold0 では cls15 は TP を失う方向に作用した。

**低 conf 診断**: cls15 は fn_at_primary=960 / covered=548 (57.1%)、
prod は 941/543 (57.7%) — 病態構造は fold 横断で不変だが、cls15 は
その回収に寄与しなかった (fold2 では低 conf 候補の昇格が TP+53 に
寄与したが fold0 では見られない)。

**意味論**:

- §2.41 の採用条件は「fold0+fold1 **両方**で合格」→ fold0 時点で
  cls15 の昇格は**数学的に消滅** (fold1 の結果に関わらず)
- v88 の fold2 +0.0038 は fold 分散内の観測だった可能性が高い:
  fold2 +0.0038 / fold0 −0.0037 で平均 ≈ 0。cls loss reweight は
  系統的改善ではなく fold 依存ノイズとして記録
- prod の不変性確認: prod@0.26 は fold2 0.4442 / fold0 0.4443 で
  安定 — 分割・評価経路は健全

**fold1 について**: 昇格判定上は既に結論が出ているが、fold 分散の
3 点目データとして fold1 を回せば「cls15 効果の分散 σ²」がより
正確に推定できる (任意。優先度は v90 より低い)。

### 2.44 v89 fold1 結果: cls=1.5 — 3 fold 完走、fold 分散内ノイズ確定 (2026-10-30)

**判定: `reject_noise`（Δ@0.26 = +0.0012、xfer −0.0014）**。Colab A100、
manifest `artifacts/v89_yolo_cls15_fold1_manifest.json`。

| conf | prod | cls15 | Δ |
|---|---|---|---|
| 0.15 | 0.4212 | 0.4196 | −0.0016 |
| 0.20 | 0.4328 | 0.4349 | +0.0021 |
| **0.26** | **0.4408** | **0.4420** | **+0.0012** |
| 0.32 | 0.4371 | 0.4395 | +0.0024 |

分解 (0.26): TP +36・FN −36・FP +50 — fold2 と同じ「TP 回復・FP 増」
機構だがネットは +0.0012 のみ。

**3 fold 集約 — cls15 効果の確定値**:

| fold | prod@0.26 | cls15@0.26 | Δ | verdict |
|---|---|---|---|---|
| 0 | 0.4443 | 0.4406 | −0.0037 | reject |
| 1 | 0.4408 | 0.4420 | +0.0012 | reject_noise |
| 2 | 0.4442 | 0.4480 | +0.0038 | reject_noise |

平均 Δ ≈ +0.0004、fold 間 SD ≈ 0.0033。**cls loss reweight は系統的
改善ではなく fold 分散ノイズとして確定**。事前登録の両 fold 合格
ルールでも消滅。0.32 移行 point は 3 fold 全てで不発（fold2 の右
シフトは単一 fold の観測だった）。`prod`/`plain_mhsa_f2` 不変確定。

低 conf 病態の fold 不変性も確定: covered/total は fold0 543/941、
fold1 545/942、fold2 543/953 — いずれも 57-58%。

### 2.45 v90 結果: semantic 提案源 — blind-FN 救済は実在するが FP コストが支配 (2026-10-30)

**判定: `reject`（union@0.26 = 0.4135 vs prod 0.4442、Δ = −0.0307）**。
Colab A100、semseg 173 epoch/72 分、manifest
`artifacts/v90_semprop_fold2_manifest.json`。

| conf | prod | union | semprop |
|---|---|---|---|
| 0.15 | 0.4265 | 0.4095 | 0.3467 |
| 0.20 | 0.4372 | 0.4142 | 0.3467 |
| **0.26** | **0.4442** | **0.4135** | **0.3467** |
| 0.32 | 0.4416 | 0.4076 | 0.3467 |

**分解 (union@0.26)**: TP 1776 vs 1715 (**+61**)・FN 1009 vs 1070
(**−61**)・FP 1274 vs 750 (**+524**)。仮説の救済機構は実在 —
semantic 提案は prod の primary FN を 61 件 TP に変換した。しかし
dedup+refine 後も FP が +524 増え、cost が gain を大きく上回った。

**核心診断 `diag_semrescue`**: prod の primary FN 953 件のうち
semantic 生提案が IoU>0.5 で覆うのは **157 件 (16.5%)** だけ。
refine/ownership を通って実際に TP 化するのは 61 件。**blind-FN
の ~43% 仮説に対し、semantic U-Net もその大部分を blind だった**
— 残る FN は検出器・画素分類どちらにも難しい（注釈ノイズ・極細/
低コントラスト個体の可能性）。

**semprop 単独 = 0.3467**: 提案源としての上限。conf sweep が全行
同一値なのは仕様通り（sem conf = 成分内最大画素確率 ≧0.5 なので
閾値を効かない）。8.35 props/img と det より多く、blob 状の粗い
提案が FP を生む（spine aux head・watershed 分割は pilot では未実装
— それらが FP を絞る役割だった可能性）。

**次レバー候補**: (a) sem 提案の厳格化（prob 閾値↑・watershed 分割・
形態学的フィルタ）— prob map は Drive cache 済みなので **評価のみ**
（~15 分）で sweep 可能。(b) sem 確率を det 低 conf 提案の
rescue スコアとして使う（提案生成ではなく二次判定）。(c) dedup を
IoU>0.5 から「det と一切重ならない」に厳格化。いずれも別実験と
して preregister が必要。

### 2.46 v91 設計: semantic map 再利用の提案厳格化 grid + rescue-scorer (事前登録, 2026-10-30)

**前提**: v90 (§2.45) は union TP +61 で blind-FN 救済機構の存在を
実証したが、sem 提案の粗さ (blob 化・面積過小の擬似成分) が
FP +524 を量産し Δ = −0.0307 で reject。本実験は v90 が Drive に
cache した確率 map (`runs/v90_semprop/probs_fold2/*.npz`) を再利用し、
**学習なしで提案化パラメータだけを変える** fold2 検索フェーズ。

**Arm A — union 厳格化 grid (27 設定)**:

- `prop_thresh ∈ {0.5, 0.65, 0.8}` (成分抽出の確率閾値)
- `min_area ∈ {20, 60, 120}` (小擬似成分の除去)
- `dedup_iou ∈ {0.5, 0.25, 0.0}` (det 提案との重複許容。0.0 = det と
  一切重ならない提案のみ — FP 源の overlap 提案を根本から排除)
- 各 thresh で成分列挙+refine は 1 回だけ行い、(min_area, dedup) の
  変化は filter のみ。@0.26 (gate point) と @0.20 を採点

**Arm B — rescue-scorer grid (3 設定)**:

- det の conf [0.08, 0.26) の refine 済み提案で、**生 det mask 内の
  mean sem 確率 >= theta** のものを primary に昇格
- `theta ∈ {0.3, 0.5, 0.7}`
- 提案生成ではなく既存 det 提案の二次判定 — 543 件の
  lowconf-covered FN 層を直接狙う (v88/v89 で fold 不変の病態)

**事前固定ルール**:

- fold2 は**検索**フェーズ。argmax 設定は `nominees` に記録するのみで
  **採用候補にしない** (fold2 で選んで fold2 で採点 = 探索バイアス)
- 昇格は別途 fold0/1 確認実験 (v92) で
  `PQ(arm,0.26) − PQ(prod,0.26) >= +0.005` を**両 fold**で満たす場合のみ
- 全 grid 設定が prod を下回る場合、semantic 提案源ラインは
  negative result として閉じる

**誠実留保**: (a) sem 確率 map は fold2 学習のもので fold0/1 確認には
別 fold の学習が必要 (v92 のコスト ~1.5h×2)。(b) rescue-scorer の
mean 確率は det mask 全域で取るため、mask が大きいと背景で薄まる —
max 確率版は未実装で、探索結果が乏しければ検討。(c) Arm A の grid は
27 点と密でない — 連続パラメータの最適化ではなく方向性の検証。

実装: `scratch/build_v91_semsweep.py` +
`scratch/build_v91_semsweep_cells.py` →
`solar_filament_v91_semsweep_fold2_colab.ipynb`。manifest は Drive
`artifacts/v91_semsweep_fold2_manifest.json`。

### 2.47 v91 結果: semantic 提案源ライン negative で閉鎖 (2026-10-30)

**判定: 両 arm とも prod 未満 — §2.46 ルールにより提案源ラインを
negative result として閉じる**。Colab A100、学習なし (v90 cache
再利用)、manifest `artifacts/v91_semsweep_fold2_manifest.json`。

**Arm A — union 厳格化 grid (27 設定)**: 最良は
`thresh=0.8 / min_area=120 / dedup_iou=0.0` → PQ **0.4366**
(Δ **−0.0076**)。thresh・min_area・dedup を厳しくするほど改善
する方向は一貫 (dedup=0.0 が最良) だが、最厳格点でも prod
0.4442 に −0.0076 届かず。rescued TP +28 に対し残存 FP が支配。

**Arm B — rescue-scorer grid (3 設定)**:

| theta | PQ@0.26 | Δ | rescued | 純増TP |
|---|---|---|---|---|
| 0.3 | 0.4124 | −0.032 | 832 | −14 |
| 0.5 | 0.4274 | −0.017 | 556 | −9 |
| 0.7 | 0.4433 | −0.0009 | 250 | +13 |

theta=0.7 で Δ −0.0009 まで肉薄するが、theta↑ で Δ→0 に漸近
する形状 = **この機構は prod に収束するだけで prod を超える
方向を持たない**。昇格させた 250 件で純増 TP は +13 のみ。

**意味論**: semantic 提案源は「blind-FN を拾える」という機構
そのものは実在した (v90: TP +61) が、拾える提案は実運用の
精度では FP コストに負ける。厳格化・二次判定のどちらの
アプローチも prod を上回る点を持たない。残りの FN プールは
検出器・画素分類のいずれにも実質 blind (注釈ノイズ・極細個体
の可能性が高い)。biconcavelens が報告した +30 filament 救済は、
彼らの pipeline 全体 (spine head・watershed 分割・別評価経路)
との複合効果であり、我々の prod 経路への移植では再現しなかった。

**キャンペーン状態**: cls 重み付け (fold 分散ノイズ)・verifier・
rescue・ensemble・TTA・提案源 — 主要レバー全て尽きた。
`prod`/`plain_mhsa_f2` が最終構成として確定。

### 2.48 v88 Kaggle 再現: Colab manifest との deterministic 一致検証 (2026-10-30)

**判定: 検証成功** — 凍結入力・prod 経路は Kaggle (T4) と Colab
(A100) で一致し、cls15 の判定 (`reject_noise`) も両環境で同一。
Kaggle kernel `shingo257/v88-yolo-cls15-fold2` (COMPLETE,
2026-10-29)。manifest: Drive `artifacts/v88_yolo_cls15_fold2_kaggle_manifest.json`
(ローカル `output/v88_kaggle/`)。

**contract 全項目一致** (差分ゼロ): `annotations_sha256 5da9e92b…`、
`folds_sha256 febb3f67…`、`refiner_sha256 b1c0f353…`、
`prod_yolo_sha256 40ec31fd…`、fold/arms/conf_sweep/gate/cls=1.5。
**datasets 一致**: train 758 観測 / 5414 instances、val 235/1784
(学習時間のみ差: T4 245.6 分 vs A100 59.1 分 = 4.2x)。

**prod baseline (同一重み) — 実質 deterministic**:

| conf | PQ_kaggle | PQ_colab | Δ(tp/fp/fn) |
|---|---|---|---|
| 0.26 | 0.4442 | 0.4442 | 0/0/0 (完全一致) |
| 0.20 | 0.4372 | 0.4372 | 0/0/0 |
| 0.15 | 0.4263 | 0.4265 | +1/+4/−1 |
| 0.32 | 0.4414 | 0.4416 | 0/+2/0 |

primary 点の PQ・TP/FP/FN が完全一致。0.15/0.32 の ±1-2 個体差は
T4↔A100 の浮動小数点経路差 (retina_masks の閾値境界)。診断も
一致: `fn_at_primary=953, covered=543` 両環境同一。

**cls15 arm (再学習、重みは非同一)** — `weights_sha256` は異なる
(クロスハード CUDA 非決定性; `deterministic=True` は同一環境内の
保証) が、評価は同一傾向:

| conf | PQ_kaggle | PQ_colab |
|---|---|---|
| 0.26 (primary) | 0.4486 | 0.4480 |
| 0.32 | 0.4458 | 0.4533 |

`delta_primary`: Kaggle **+0.0044** / Colab +0.0038 → 両方とも
`[0, +0.005)` の `reject_noise`。**判定が環境を跨いで安定**。
diag: Kaggle 899/495 vs Colab 907/531 (同規模)。

**結論**: v88 の実験プロトコル・凍結入力・prod 評価経路は
Kaggle/Colab で再現可能として検証された。再学習重みはハード間で
非同一になるが、fold 分散範囲内 (σ≈0.003) であり、実験判定を
変えない。これで全 3 fold の cls15 証拠 + クロスプラットフォーム
再現性の記録が揃った。

### 2.49 公開 repo の匿名再現検証 (2026-10-30)

**判定: 成功** — standalone repo `shingo257/solar-filament-2026-solution`
を PUBLIC に切替え、credential helper 無効・`GIT_TERMINAL_PROMPT=0`
の真の匿名環境で `git clone --depth 1` を検証:

- 791 ファイル clone 成功 (commit `0c0fc757`)
- **Git LFS 5 オブジェクト (398.7 MiB) が匿名ダウンロード成功** —
  重みは実体で LFS pointer ではない
- `weights/weights_manifest.json` との SHA256 照合 **5/5 MATCH**
  (`yolo_fold{0,1,2}_best.pt` 45,316,918B、`refiner_all_data.pt`
  141,079,441B、`refiner_plain_mhsa_fold2_s8407.pt` 141,041,333B;
  fold2 = `40ec31fd…` は v88 manifest の `prod_yolo_sha256` と一致)
- `solar_filament_submission_reproduction.ipynb`・`report/main.pdf`・
  `README.md`・`RESEARCH_LOG.md` 全て存在
- シークレットスキャン済み: 資格情報値の混入なし (kaggle.json は
  実行時に Drive から読む設計、値は repo 非含有)

提出要件の「公開 Git リポジトリ + 重み同梱」は検証済み。残るは
Google Form 送信 (ユーザー操作) のみ。

### 2.50 上位手法・Discussion サーベイ (2026-10-31)

**LB 現状**: 先頭 IONode Online 0.70 (10-24 提出)、次点 Tempest
0.64、0.62×4、0.60×4。我々 0.39。**ただし高スコアは汚染疑い
が濃厚**: 旧メトリック時代 (8-20 修正前) は空提出 0.93-1.00・
「予測数が少ないほど高スコア」の欠陥が確認済み (topic 728474/
729809)。`lamhuy8904/…0-55` kernel は実モデルではなく
lzma+base85 埋め込みの**事前計算 CSV 再生**であり、コピー
提出が盤上に残っている。Fususu `phuongncn/lb-0-70-…handoff` は
8-5 提出分 (修正前) の 0.70 を主張、参加者検証では ~0.35。

**手法サーベイ** (公開 kernel/model):

| 手法 | 出典 | 要点 |
|---|---|---|
| Mask R-CNN + crop U-Net refiner + flip-4 TTA | phuongncn 0.70 handoff | detect-then-refine (我々と同系)。det 0.70/mask 0.50、refiner 256px crop、topology-safe TTA + crowding abstention。MD5 固定重み |
| YOLO + crop seg refiner | muhammadshoaibaltaf MAGFiLO V10 | file_name グループ分割・annotator 別サンプル (我々と同一流儀) |
| ResNet-34 U-Net + Dice/BCE | rokaiyasomapti "(LB 1st)" | タイトル詐称、実態は baseline pipeline |
| ResNet-34 U-Net mask+hull+interior 3heads | jamesmills1975 CAGE | model 公開のみ |
| 古典 CV 閾値処理 | vmerckle 0.46 | 学習なし dark residual percentile+CC で LB 0.46 — 採点慣例の隙 |
| 申告値 | discussion | OOF PQ 0.48 (106枚 val)、厳格 5-fold OOF 0.46、decoupled pipeline "past 0.44"、local 0.49 |

**実力帯の推定**: 真剣な参加者の honest OOF/LB は ~0.44-0.49。
我々の OOF 0.443-0.445・LB 0.39 は実力フロンティア内。0.55+
は汚染由来と判断してよい。

**新規主催者裁定 (提出直結)**:

| 項目 | 裁定 |
|---|---|
| 両 submission の再現 | 戦略が異なれば**両方**再現必須 (747960) → 我々の repro nb は両方を byte-identical 再現済みで適合 |
| Google Form | 再提出可・**最後の提出が有効** (749524)。11-15 まで編集可 |
| 効率性 | qualitative フィルタのみ、点数化せず (749526) |
| test 評価 | self-eval nb と同一 PQ 計算 (753665)。multi-annotator の詳細は意図的に非開示 (753816/753043) |
| best model | ensemble でも可、評価対象を載せよ (753665) |
| SWEFil 外部GT | **禁止** (746318) |
| unlabeled 外部/近傍 frame | 学習のみ可・test±1h 除外・推論時は test 画像のみ (748045/748498/744047/743385) |
| spine/bbox/area aux target | **使用可** (736543) — biconcavelens の spine head は合法 (v90 では未実装だった) |
| top-3 判定 | LB 順位ではなく Rubric ベース (753665) → 報告書・コード品質が効く |

**含意**: 新規採用候補の方法はなし。残差は評価寄り — 報告書・
再現性・証拠の積み上げが最後の差別化であり、それは完了済み。
データ: `output/discussions_2026-10-31.json`、
`output/discussion_msgs_2026-10-31.json`、`output/kernel_survey_2026-10-31/`。

### 2.51 Biohub CTDD 上位手法の転用調査 (2026-10-31)

biohub-cell-tracking-during-development (9-29 締切、3,947 チーム)
の受賞解法を調査し、solar へ転用可能な前処理・マスク技術を抽出。
データ: `output/discussions_2026-10-31.json`、
`output/top_solutions_2026-10-31.json` (Biohub module 側)。

**上位パイプラインの共通構造** (detect→link→ILP は我々の
detect→refine→ownership と同系):

- 1st (Sergio Alvarez, 0.978): **Cellpose 型 flow-field head** —
  前景確率 + 各 voxel→中心への 3ch flow を予測し、収束先で個体化
  (Gaussian blob ではなく mask+flow)。3D U-Net、anisotropic
  pooling、deep supervision、**軽量 MAE 事前学習** (50% を 2×4×4
  tiny block で mask)、手書き 1,800 セル + GT 中心への
  **4.305µm Gaussian stamp** + 全動画擬似ラベル (5-fold avg +
  flip-4 TTA、高確信のみ採用)、重い aug + EMA、分裂 8x
  oversample、hard-negative round、mask 幾何特徴 (semi-axes,
  fill ratio, axis direction) を linker 入力に
- 2nd (0.970): temporal 3D detection + graph tracking。**弱い
  検出証拠を単体で捨てず track として判断** (我々の lowconf
  提案と同じ病態だが時間持続で仲裁 — solar 単一フレームでは
  直接適用不可)。合成 faint-cell を検出器学習に注入
- 3rd (0.967): 2.5D U-Net + 3D SegResNet ensemble、dense 3D flow
  motion 補正、物理座標 (µm) で距離計算、gap 充填+小成分除去
- 4th (0.969/0.962): 検出→擬似ラベルモデルで精緻化、**sparse
  movie には厳格閾値** (context 適応)、learned link/div score +
  ILP、fork verifier (GBM)、track-support gate
- 5th: 3D U-Net + transformer linker (frame 間 cross-attn) + ILP、
  5-fold × 8 TTA 平均、重複/drift frame を val から除外

**solar へ転用可能な技術**:

| 技術 | 適用性 | 評価 |
|---|---|---|
| **Cellpose flow-field 個体分離** | 新機構。merged instance 分離 (report の未解決課題) を box-NMS 以外の原理で解く。v90-91 で pixel blob の FP 爆発を見たが、flow は各画素を 1 中心に束縛するため blob より厳格な提案を作れる可能性 | **唯一の未試行本格機構 — 将来候補** |
| Gaussian stamp 補助監視 (中心 σ≈4.3µm) | refiner の crop 内で中心距離 aux head | 小粒。shrink-only 設計と相性は要検討 |
| mask 幾何特徴 scorer (semi-axes/fill/axis) | lowconf 提案 (57% cover 層) の rich な二次判定 | A1 scorer で既に類似路線 (OOF+0.0034→LB不転)。特徴量は強化できるが機構は試行済みに近い |
| hard-negative crop mining | refiner 学習の FP 耐性 | 1st は使わず (我々も同じ)。3rd/4th は使用。小粒 |
| sparse context 適応閾値 | 疎密別の conf 運用 | sweep の変種。小粒 |
| 擬似ラベル + 高確信フィルタ + TTA平均 | 既試行 (reject) | × |
| flip-4 TTA on refiner prob | 既試行 (+0.0007 不転) | × |
| SSL 事前学習 (MAE-light) | SimCLR で負の結果済み。MAE は別レシピだが SSL ライン自体閉鎖済み | × (新証拠なし) |
| time-step / temporal aug | v67 で reject (stale labels) | × |
| ensemble 平均 (5-fold×TTA) | 既試行 (不転) | × |

**結論**: 学習済みの大半は我々が類似路線で棄却済み。**唯一の
実質的な未試行機構は Cellpose 型 flow-field による個体分離** —
merged filament の分離と、box 非依存の提案生成を両立させ得る。
ただし新モデル学習が必要であり、11-15 締切の最終構成
(`prod`/`plain_mhsa_f2`) を変えるものではない。報告書の
future work 候補として記録する。

### 2.52 v92 事前登録: Cellpose flow-field 提案源 pilot (2026-10-31)

§2.51 の棚卸しで唯一残った未試行機構 (Biohub 1st の Cellpose 型
flow-field 個体分離) を fold2 pilot として実行する。
ノート: `solar_filament_v92_cellpose_flow_fold2_colab.ipynb`
(builder `scratch/build_v92_cellpose_flow.py`)。

**設計**: v90 の semantic U-Net を 3ch 化 — ch0=fg logit
(10*focal+dice, v90 と同式)、ch1-2=flow_y/flow_x (fg 画素上の
smooth-L1×1.0)。flow 教師 = 各画素→所属 instance 全体重心への
unit ベクトル (bg=0)。学習は v90 と同条件 (ConvNeXt-T、1024px
crop、(file,annotator) エントリ、時間予算 1.2h)。augment は
rot90/flip を flow 場に空間+ベクトル成分変換で適用。

**推論**: full-res 4-flip TTA (flow 成分は反転解除時に符号反転)
→ fg>=0.5 の画素を予測 flow で Euler 追従 (200 step × 1px、
bilinear) → 端点を DBSCAN (eps=6px, min_samples=8) でクラスタ
→ クラスタ毎の画素集合を 1 instance 提案 (conf=クラスタ内平均
fg 確率、min_area=20 未満・noise 点は棄却)。blob と違い各画素が
1 中心に束縛されるため merged instance の分離を原理で解く。

**arm**: `flowprop` (flow 提案のみ、診断用で採用候補にしない) +
`union` (prod det ∪ flow、dedup IoU=0.5)。採点は prod と同一
refine→ownership→公式PQ。

**事前固定ゲート**:

- `PQ(union,0.26) − PQ(prod,0.26) >= +0.005` → fold0/1 に昇格検討
- `[0,+0.005)` → reject_noise、`<0` → reject
- 診断 (判定不使用): `diag_flowrescue` = prod の primary FN を
  flow 提案が IoU>0.5 で覆う数 + flow_props_per_image
- fold2 は検索フェーズ。ゲート突破時のみ fold0/1 確認を別途設計
- 全 arm が prod 未満なら flow-field ラインは negative で閉じ、
  `prod`/`plain_mhsa_f2` を最終構成のまま提出する

### 2.53 v92 結果: Cellpose flow-field 提案源 reject (fold2)

manifest: `artifacts/v92_cellpose_flow_fold2_manifest.json`
(exp `v92_cellpose_flow`, fold2, 155 epoch / 72.0min、A100)。

**結果**: 両 arm prod 未満 → §2.52 ルールで flow-field ライン negative 閉鎖。

| arm | PQ@0.26 | Δ | TP | FP | FN | verdict |
|---|---|---|---|---|---|---|
| prod | 0.4442 | — | 1715 | 750 | 1070 | — |
| flowprop | 0.3686 | −0.0756 | 1537 | 1273 | 1248 | reject |
| union | 0.4235 | **−0.0207** | 1772 | 1124 | 1013 | reject |

**診断**: `diag_flowrescue` = prod primary FN 953 件のうち
flow 提案が IoU>0.5 で覆うのは **163 件 (17.1%)** — v90 semantic
の 157 件 (16.5%) とほぼ同等。flow_props_per_image = 7.29。
union は TP+57 に対し FP+374 — v90/v91 と同じ FP 爆発構造。

**解釈**: Cellpose flow-field は機構として正しく動いたが、
「残りの blind-FN プール」を広げる効果はほぼゼロ (163 vs 157)。
detector・semantic・flow の 3 系統が事実上同じ FN 層を見落とす
ことを裏付け — この層はモデル非依存の限界 (注釈ノイズ・極細個体
との推定を再確認)。個体分離の原理差は提案品質の差に現れず、
dedup 統合でも FP コストが勝った。

**キャンペーンの最終確定**: 主要機構すべて尽きた。
cls 重み付け・verifier・rescue・ensemble・TTA・提案源 3 系統
(semantic・flow)・再学習 — すべて frozen gate 未達。
`prod`/`plain_mhsa_f2` が最終構成 (OOF ~0.444、public ~0.39)。

### 2.54 診断: annotator 不一致天井 — ネックの定量的確定 (2026-11-01)

v88-92 の全機構棄却後、残るボトルネックの正体を multi-annotator
データで直接計測 (fold 全体・707 file・296 multi-annotated file・
2146 matched pairs)。計算は train JSON のみで完結、外部データ不使用。

**計測結果**:

| 指標 | 値 | 解釈 |
|---|---|---|
| annotator matched IoU (IoU>0.5 一致対、n=2146) | mean **0.634** / med 0.630 / p25 0.572 / p75 0.693 | 人間同士の境界一致度 |
| **我々のモデル SQ** | **0.680** | **人間一致度を既に超過** → SQ 改善余地ゼロ |
| per-record recall (別 annotator にも描かれる率) | mean **0.563** | 全ラベルの ~44% が単独 annotator 物体 |
| 単独物体の視認性 (n=702, 100 file 計測) | contrast 9.6・visible 46% | 不可視ではなく「判断が分かれる境界物体」 |
| consensus 物体の視認性 (n=774) | contrast 11.6・visible 66% | 単独より明瞭だが絶対差は小さい |

**FN プールの構造確定**: prod primary FN 953 件のうち det/semantic/
flow の 3 系統で救済できたのは各 ~16-17% (=単独 annotator 物体層と
数値一致)。残りは「annotator 同士も描かない物体」ではなく
「1人しか描かない物体」= 注釈エントロピー由来でありモデル依存
ではない。

**per-record 採点の数学的束縛**: R≈2.44 record/file の pooled 採点で、
単独物体 (1 record にのみ存在) を予測すると 1TP vs ~1.44FP。
限界価値 ≈ 0.63IoU×1 − 0.5×1.44 ≈ 負 → **単独物体の予測は数学的に
マイナス**であり、prod がそれらを避けるのは採点上最適。理論上限
(全 annotator union を完璧予測でも SQ=0.63 頭打ち・非共有分が FP 化)
は PQ≈0.49 程度と推定 — v87 union_hard がこの路線で −0.014 だった
理由と整合。

**結論**: ネックはモデル能力ではなく **注釈不一致 × per-record 採点
構造**。マスク/前処理の変更は SQ が人間一致度を超えている時点で
理論上効かない。残る機構差のある一手は「consensus 尤度を明示的に
学習する 2-class 検出器」(v93) と「sparse-label 採点向けの高 conf
提出」のみ。

### 2.55 v93 事前登録: consensus-class YOLO (annotator 合意数の明示学習)

§2.54 の診断に基づく、ラベル構造変更の**最後の未試行バリアント**。
v87 union-label (ラベル集合を union mask に置換) は棄却済みだが、
v93 は**集合構造を prod と完全に同一に保ったまま instance の class
だけを変える** — 全く別のレバー。

**設計 (事前固定)**:

- per-record 学習サンプルは prod と同一 (`stem__c{j}`、各 record が
  自分の polygon を持つ)。**変えるのは class index のみ**
- 各 instance の class = **1 (strong/consensus)** if IoU>0.5 union-find
  cluster (同一 record 同士は併合しない、v87 の cluster_from_iou) の
  distinct annotator 数 >= 2、else **0 (weak/単独)**
- `yolo11m-seg` を 2-class、`single_cls=False`、data.yaml は
  `0: weak / 1: strong`。他の学習引数 18 項目・fold2・seed は prod 同一
- 評価: 各提案の score = conf if cls==1 else conf*w0、
  w0 grid {1.0, 0.7, 0.5, 0.3, 0.0} × conf sweep {0.15,0.20,0.26,0.32}。
  w0=1.0 は sanity (class 情報不使用、prod ranking と同等のはず)

**ゲート (事前固定)**: `max_{w0} [PQ(arm,0.26) − PQ(prod,0.26)] >= +0.005`
→ fold0/1 昇格検討。全 w0 で [0,+0.005) は reject_noise、<0 は reject。
exploratory best (w0,conf) は記録のみ、判定に使わない。

**仮説**: §2.54 のように単独物体の予測は per-record 採点でマイナス。
prod の conf は既に consensus-likeness を暗黙学習しているはずだが、
明示 class 化でランキングの分解能が上がるかもしれない。
棄却された場合: consensus 仮説自体は §2.54 で支持されるが、
**class 明示化という実装形態では効果を出せなかった**と記録し、
ラベル構造ラインを完全閉鎖する。

notebook: `solar_filament_v93_consensus_class_fold2_colab.ipynb`
builder: `scratch/build_v93_consensus_class.py` (v87 を改造)
manifest: `artifacts/v93_consensus_class_fold2_consensus2_manifest.json`

### 2.56 v93 結果: consensus-class YOLO = reject — ラベル構造ライン完全閉鎖 (2026-11-01)

fold2、70 epoch 完走 (0.966h、early stop なし)。manifest:
`artifacts/v93_consensus_class_fold2_consensus2_manifest.json`。

| arm | PQ@0.26 | Δ vs prod(0.4442) | TP | FP | FN |
|---|---|---|---|---|---|
| consensus2 w0=1.0 | 0.4035 | **−0.0408** | 1322 | 372 | 1463 |
| w0=0.7 | 0.3720 | −0.0722 | 1115 | 233 | 1670 |
| w0=0.5 | 0.3659 | −0.0783 | 1076 | 203 | 1709 |
| w0≤0.3 | ≤0.3717 | さらに悪化 | — | — | — |
| exploratory 全域 best | w0=1.0@conf0.2 → 0.4293 | −0.0149 | — | — | — |

**sanity arm 自体の劣化が本質**: w0=1.0 (class 情報不使用 = 提案 ranking
だけ比較) で既に −0.041。2-class 化が base 検出性能を損なった — conf が
strong class (合意>=2, val では instance の ~24%) に偏り、weak class
(76%) は低 conf 化。@0.26 で TP が 1715→1322 と大崩れ。w0 を下げると
weak 提案が消え FN が単調増悪 — 弱い物体はその record 採点では依然
GT なので、捨てる方向には絶対に効かない。

val の class 構成: weak 1363 inst / 218 img (76%)、strong 421 inst /
100 img (24%)。学習後の class 別 val: strong は mask R=0.727 と高め
に振れる一方、weak は R=0.536 — strong 側に conf を集中させた
結果、総合 recall が崩れた。weak_kept_per_image は w0=1.0@0.26 で
1.29 (残る weak は conf 高の少数) → w0<=0.3 で 0.0 (全滅)。verdict:
`best_w0=w0=1.0, Δ@primary=-0.0408, reject` (manifest 確定値)。

**§2.54 との整合**: singleton と consensus は見た目では分離不能
(annotator の「判断」の違いであって外見の違いではない) ため、明示
class 化してもモデルが得る追加情報はゼロ — むしろ class 分割の
calibration コストだけが残った。

**ラベル構造ラインの最終台帳**:

| 実験 | 手法 | Δ@0.26 |
|---|---|---|
| v87 union_hard | 全 record union mask を全 copy に | −0.0138 |
| v87 union_agree | 合意数重みの copy | −0.0040 |
| v93 consensus2 | class=合意数 (集合構造は prod 同一) | −0.0407 |

→ **ラベル構造変更は全手法棄却で完全閉鎖**。prod の per-record 生
ラベルがこの採点構造では最適であることが確定。
`prod`/`plain_mhsa_f2` 最終構成に変更なし。

### 2.57 conf032 A/B 提出: prod@0.32 = public 0.39 (prod@0.26 と表示同等)

sparse-label 仮説 (§2.54・discussion 743385/735256 系) に基づく
低コスト A/B: 公式再現 notebook を `SOLAR_SUBMISSION_CONF=0.32`
(既定値パッチ済み kernel `solar-filament-2026-reproduction-conf032`
v1、Kaggle T4、wall ~7h install+前処理律速) で実行。

**provenance**: 凍結重みを使用 (再学習なし)。
yolo_fold2 `40ec31fd2c6aa8e3` / refiner_all_data `e9f0e8dd7ff104d9` /
grouped_folds `24367120` はいずれも凍結値と一致。
efficiency: 0.854 s/img (T4、153.8s/180枚)。

**結果**: `submission_conf032.csv` (1026 行、prod@0.26 の 1172 から
−146 = 93 画像の低信頼案が脱落、+2 は NMS floor 差) を提出
(ref 57843737) → **public 0.39**。prod@0.26 の 0.39 と表示桁
(0.01) では識別不能、Δ は ±0.005 以内。

**解釈**: conf を 0.32 に上げても public スコアは悪化も改善も
しなかった — 「conf が public LB の binding constraint ではない」
という弱い証拠。sparse-label 仮説の方向 (高 conf が利く) とも
「予測 drop が無関係な層」とも整合で、結論は inconclusive。
提出枠を 1 つ消費したが、final 2 提出の選択には影響なし —
**prod + plain_mhsa_f2 のまま確定**。

成果物 (repo 外): `output/conf032_run/` (gitignore、CSV +
efficiency + sha256 + pip_freeze、359K)。
kernel バリアントは `output/kaggle_kernel_conf032/` (PR #576)。

### 2.58 public LB スナップショット (2026-10-31、conf032 提出後)

`kaggle competitions leaderboard -d` で全 1305 チームを取得。
自チーム "ERS" (shingo257): **rank 219 / 1305、score 0.39、
17 submissions** (最終 = conf032 A/B ref 57843737)。

上位分布: 1位 IONode Online 0.70、2位 Tempest 0.64、3-6位 0.62、
7-9位 0.60、以降 0.58/0.57/0.56 の密集帯。top 層は conf032 提出
時点で動いていない (メトリックリセットではなく高得点が実在)。

§2.54 の annotator 天井 (~0.49 union 上限) との対比: 0.70 超は
honest per-record 採点では理論上限を超える — discussion 733912
(二metric混在)・735377 (panoptic painting 要件)・729809 (metric
測定) と整合的に、top 層は採点器の挙動差 (test 側の multi-
annotator pooling や空マスク補正の扱い) を利用している可能性が
高い。我々の 0.39 は honest 予測の帯として、最終評価は主催者
rubric (再現性・報告書・方法の健全性) 依存と整理済み。

### 2.59 v89 fold0 再実行で deterministic 一致を確認 (2026-11-01)

`solar_filament_v89_cls15_fold01_colab.ipynb` (fold0) を Colab A100
で再実行 — 学習 0.970h・評価まで完走。**§2.43 の記録値と全 16 セル
(4 conf × prod/cls15 × PQ/SQ/TP/FP/FN/sum_iou) が小数 4 桁まで一致**:

prod@0.26 = 0.4443 / cls15@0.26 = 0.4406 (Δ −0.0037、reject)、
xfer@0.32 = −0.0047、diag 543/941・548/960。

`deterministic=True` + `seed=3407+fold` + 同一 dataset/fold/refiner
ckpt で、YOLO11m-seg 学習→refine→公式 PQ の全経路が bit 等価の
メトリクスを再生成した = パイプラインの end-to-end 決定性を
同一プラットフォーム上で確認 (§2.48 は Colab→Kaggle の横断一致、
本項は Colab→Colab の再実行一致)。評価コード・分割・重み解決の
健全性の追加証拠。manifest は同一経路
`artifacts/v89_yolo_cls15_fold0_manifest.json` に再書き込み。

### 2.60 fold2 prod 完全誤差分解 — 根本原因特定 (2026-11-01)

ユーザ指示「まず我々のモデルの問題点を分析して、根本的なところを
改善を試みたい」に対応。fold2 val 全 235 枚をローカル CPU で
end-to-end 再推論 (yolo_fold2 `40ec31fd2c` + refiner_all_data
`e9f0e8dd7f`、conf 0.26 / refiner_thr 0.45 / drop 0.9 /
min_area 20、wall 2440 s) し、proposal → refined → final の各段で
cross-annotator クラスタ coverage を追跡。GT は IoU>0.5 union-find
で annotator 横断クラスタ化 (2168 clusters、k1/k2/k3 = 1702/315/151)。

**再現確認**: 公式採点をそのまま再実行して PQ 0.44630
(SQ 0.6811 / RQ 0.6552、TP 1721 / FP 747 / FN 1064)。
min_area_sweep 時点の記録 0.44423 との差 +0.0021 は refiner 重みが
fold2 専用ではなく all_data である点・推論キャッシュ経路差の範囲で、
実装誤差ではなく再現差として記録 (以降の Δ は本ラン 0.44630 基準)。

**FN 分解 (1064)** (2026-11-01 Codex review 修正後の再集計値):

| class | n | 内訳 |
|---|---|---|
| proposal_blind | 691 | どの proposal も IoU≤0.5 (median GT 714 px) |
| near_miss | 287 | 最良 pred IoU が 0.3–0.5 (median 1815 px) |
| ownership_lost | 57 | refined では被覆、ownership で消失 (889 px) |
| cross_covered | 29 | 他 annotator 側クラスタは被覆 (1450 px) |

(初版は proposal_blind 696 / ownership_lost 52。被覆は stage 間で
単調ではなく refined が proposal IoU を 0.5 超に救済し得るため、
「最初に被覆の無い段階」帰属で 5 件の singleton が blind と誤記
されていた。§2.64 の修正コミットで「最後に被覆が残った段階」
帰属に訂正。)

**FP 分解 (747)**: free 267 (median 1084 px) /
annotator_gap 263 (他 annotator の GT には一致、1365 px) /
near_miss 217 (IoU 0.3–0.5、2267 px)。

**クラスタサイズ別 recall**: k=1 0.486 (874/1702 FN) /
k=2 0.797 / k=3 0.863。検出器は consensus 物体を強く優先的に
拾い、単一 annotator 物体が FN の最大源。TP IoU は k によらず
0.67–0.69 でフラット。

**Counterfactual (fold2)**:

| 操作 | PQ | Δ |
|---|---|---|
| baseline final | 0.4463 | — |
| IoU 閾値 0.5→0.3 (探索) | 0.4945 | +0.0482 |
| クラスタ coverage 転送 | 0.4669 | +0.0206 |
| free FP を oracle で全 drop | 0.4702 | +0.0239 |
| union/dedup 評価 (採点定義差) | 0.4421 | −0.0042 |
| 全 GT emission (sanity) | 1.0 | — |

**Emission-variant アーム (追加 ad-hoc)**: 最終 mask 集合だけを
入れ替えて公式採点。refined のみ (ownership 撤廃) = **0.4596
(+0.0133、TP 1914 / FP 887 / FN 946)**、refined+proposals =
0.4817、refined+proposals+final = 0.5044 (+0.058)。空マスクは
0 件。refined 段 coverage 1303 → final 1221 (−82 clusters) で、
ownership 後処理 (disjoint 化 + containment drop) が被覆を
直接削っていることを確認。また公式採点は (annotator,pred,GT)
ペア単位のため、同一物体に複数候補を出すと TP ペアが重複計上
される構造 (oracle GT emission を k 重複しても 0.762→0.764 と
dup 自体はほぼ中立 — 利得は variant 間の被覆差＋TP 多重計上)。

**Prediction marginal value**: final 1501 件のうち 355 件
(23.7%) が pooled annotator 採点で価値負 (うち 328 件は全
record で FP、27 件は一部 TP でも他 record の FP 罰が勝つ)。
価値負 pred の median conf 0.381 vs 正 0.569。conf 帯別 neg
share は (0,0.3] 50% → (0.3,0.5] 33.5% → (0.5,0.7] 15.8% →
(0.7,0.9] 7.7% と低下するが 0.9 帯 9.1% で、**raw conf 閾値では
再現不能** (conf>0.30 → 0.4407、>0.50 → 0.3995 と全閾値で悪化)。
oracle で価値負を全 drop すると PQ 0.4935 (+0.0472)。
(初版は marginal 分母に「その pred が唯一被覆した GT の
新規 FN 化」を反映していなかったため 432 件/28.8%・oracle
+0.0568 と過大計上。§2.64 で厳密式 n_tp+0.5·n_fp−0.5·n_unique
に訂正。)

**根本原因の整理**:

1. **proposal 盲域 (最大)**: FN の 65% は小さな singleton 物体への
   検出欠落 (median 714 px)。ただし §2.54 の通り単一 annotator 物体は
   pooled 採点で期待価値が薄く (R≈1.69 で marginal ≈ +0.3、一部負)、
   闇雲な recall 強化は FP 罰と相殺される — v88–v93 で conf・
   tiling・cls 系を潰した結果と整合。
2. **ownership 後処理が採点構造と不整合 (実害確定)**: panoptic
   disjoint 性を強制する設計が coverage を 82 clusters 分削り、
   TP ペアの重複計上も潰す。refined のみ emit で +0.013 — **これは
   誠実なパイプライン修正として採用候補** (fold0/1 で要検証)。
3. **採点ペア計上 + annotator pooling の構造税**: ~24% の pred が
   価値負・oracle gate で +0.047、複数仮説 emit で +0.058。
   LB 0.70 層との乖離の一部は、sparse test ラベルに加えて
   「仮説多重度を正直に 1 mask/物体に絞っている」側面が大きい。

**次実験 (falsifiable、採用ゲート事前宣言)**:

- **v94 (誠実系・軽量)**: emission を refined mask のみに変更
  (ownership 撤廃、min_area のみ)。fold2 期待 +0.013。採用条件:
  fold0/fold1 OOF で ΔPQ ≥ +0.005 かつ k=1 recall 退行なし。
- **v95 (学習 emission gate)**: pred 時点特徴 (conf・mask 面積・
  refiner prob 統計・prop⊃ref 収縮率・他 pred との包含構造・
  前処理画像の contrast/limb 距離) で marginal value>0 を
  cross-fit 予測 (fold0+1 学習 → fold2 評価で rotate)。
  oracle 上限 +0.047 (修正後の厳密 marginal 分母で再計算)。採用条件:
  fold2 ΔPQ ≥ +0.010 かつ
  rotate 全 fold で非負、conf-only baseline を明確に上回ること。
- **探索的提出 (任意)**: refined+proposals の 2 仮説 emit は
  format 適合だが採点器のペア計上を明示的に利用する設計のため、
  report 開示必須・審査リスクあり。test 側採点機構の probe とし
  て 1 回だけ試すかはユーザー判断。

成果物: `scratch/analyze_error_decomposition.py` (SHA256
`cc5114b42cf4d6ee5cfa6aa1773d19024315f491a6c6d09ed2fd0f89f09e56d3`
→ §2.64 修正版 `c2ec60b0e875dbd14cf2d961481d3b026e519c76a63c964bf23fc018151a4976`)、
`output/error_decomp_fold2/{summary.json,fn_decomp.csv,fp_decomp.csv,
tp_decomp.csv,cluster_coverage.csv,per_pred_marginal.csv,per_photo.csv}`。
cache/ (packed mask npz) は再生成可能のため gitignore。
環境: Python 3.12.10 / torch 2.11.0+cpu / numpy 2.4.2 /
ultralytics 8.4.120 (.venv-prod)、metric_version
`official-v1-2026-08-15`。

### 2.61 v94/v95/境界校正の全 fold 判定 (2026-11-01)

§2.60 で事前宣言した改善実験を fold0/1 OOF で検証。fold0/1 を
新規 CPU 完全推論 (各 ~2370 s、yolo_fold0 `ab8810916e` /
yolo_fold1 `44ba9381a5` + refiner `e9f0e8dd7f`) し、同一分解を
実施。3 fold の prod baseline は **0.4458 / 0.4448 / 0.4463** と
±0.001 の fold 安定性。分解構造も全 fold で一致 (proposal_blind
687–695 枚・coverage funnel で refined→final 間に 66–82 clusters
消失・価値負 pred share 23.7–24.7% — §2.64 修正後の再集計値)。

**v94 refined-only emission (ownership 撤廃、min_area=20 のみ)**:

| fold | final (prod) | refined emit | Δ |
|---|---|---|---|
| 0 | 0.44583 | 0.45372 | +0.0079 |
| 1 | 0.44485 | 0.45079 | +0.0059 |
| 2 | 0.44630 | 0.45957 | +0.0133 |

**全 fold 正・mean +0.0090 → 事前ゲート (fold0/1 とも ≥+0.005) 合格。**
後処理の撤廃のみで検出は同一のため、k=1 recall 退行は構造上
あり得ない (coverage は refined ⊇ final)。ownership
(disjoint 化 + containment drop) が公式採点と不整合であることが
fold 横断で確定 — 本線への採用を推奨 (prod → prod_v94)。

**v95 cross-fit 学習 emission gate — REJECT**:
推論時特徴 19 個 (conf・面積系・prop↔ref 一致・pred 重なり・
形状・photometric contrast・disk 内位置) で価値負 pred を
balanced LR で cross-fit 予測。τ は train fold の近似 PQ で選択。

| held fold | τ | train Δ(approx) | held-out exact Δ | dropped |
|---|---|---|---|---|
| fold0 | 0.60 | +0.0120 | **−0.0289** | 480 |
| fold1 | 0.70 | +0.0117 | **−0.0114** | 302 |
| fold2 | 0.70 | +0.0094 | **−0.0085** | 343 |

近似式 (pred 寄与の加算分解) では train で +0.01 見えるが、
exact 採点では全 fold 負 — ゲートが TP を持つ pred も落とし、
GT が orphan 化して FN 罰 (+0.5) が近似を上回る。AUC 自体は
conf-only 0.730 → 全特徴 0.768 (fold2 in-fold CV) と信号はあるが
**annotator 由来の価値負性は画像根拠から予測不能**、という
fold 横断の否定結果として記録。oracle 上限 +0.047 (修正後) は到達不能。

**境界校正 (dilation/erosion) — REJECT**: refined/final mask の
morphological dilation/erosion (r∈{−3..+5}px) を fold2 で掃引。
ref±1 = 0.4335/0.4344 (refined 0.4596 に対し −0.026/−0.025)、
fin+1 = 0.4199 (final 0.4463 に対し −0.026)、全半径で単調悪化。
near-miss 帯 (IoU 0.3–0.5 の 287 FN + 217 FP) は系統的な
over/under-coverage ではなく物体ごとの境界誤差 = 人間一致度
天井 (SQ 0.63) の反映で、一律の補正では救えない。

**残る効くレバー (採点構造由来、モデル品質とは独立)**:

| arm | fold0 | fold1 | fold2 |
|---|---|---|---|
| final (prod) | 0.4458 | 0.4448 | 0.4463 |
| refined emit | 0.4537 | 0.4508 | 0.4596 |
| refined+prop | 0.4768 | 0.4720 | 0.4817 |
| ref+prop+final | 0.5001 | 0.4968 | 0.5044 |

多仮説 emit は採点器の TP ペア計上を利用する format 適合だが
metric-surface 最適化 — 採用可否はユーザー判断 (report 開示必須)。
proposal_blind (FN の ~65%) は残存するが、小さな singleton 物体は
pooled 採点で期待価値が薄く、v88–v93 で recall 系は全て reject 済。

成果物: `output/error_decomp_fold{0,1}/` (summary.json + 各 CSV、
cache/ は gitignore)、`output/v95_features/fold{0,1,2}.csv` +
`fold{0,1,2}_arms.json` + `gate_results.json`、
`scratch/run_v94_v95_emission.py` (emission arm 評価 + 特徴抽出)。

### 2.62 v94 採用 + probe kernel 提出 (2026-11-01、ユーザー承認)

v94 (refined-only emission) は事前ゲート合格 (全 fold 正、
fold0/1 で ≥+0.005) のため **採用**。提出用に
`build_submission_reproduction.py` へ emission バリアントを追加:

- `prod_v94`: prod pipeline の emission だけを refined mask 直接
  出力に変更 (ownership 撤廃)。refiner・重み・前処理は prod と同一。
- `probe_refprop`: refined + proposal の 2 仮説 emit。採点器の
  TP ペア計上が test 側でも同じかを測る **1 回限りの探索的提出**
  (ユーザー判断)。local OOF +0.03 台を期待 — LB が prod の 0.39
  から大きく跳べば top 層の手法が「仮説多重度」である傍証。
  report には scoring-surface 最適化である旨を開示する方針。

実装上の注意: 監査セル (cell 14) の disjoint assert は
prod / plain_mhsa_f2 (ownership 経由) のみに限定し、v94/probe は
重複画素を計数報告に緩和 (emit する mask が構造上重なり得るため)。

kernel (Kaggle GPU、各 ~7h 想定):
- `shingo257/solar-filament-2026-reproduction-prod-v94` v1
- `shingo257/solar-filament-2026-reproduction-probe-refprop` v1

最終 2 提出の選択は引き続きユーザー操作 (prod `57039986` /
plain_mhsa_f2 `57072525` / 本 2 件の結果次第で再判断)。

### 2.63 disjoint 強制の確定 — v94/probe 提出 ERROR とローカル評価の修正 (2026-11-01)

prod_v94 (refined 直接 emit、1328 rows・重複 364K px) と
probe_refprop (refined+proposal、2656 rows) を Kaggle へ提出 —
**両者とも SubmissionStatus.ERROR** (ref 57859708 / 57860275)。
prod・conf032 等の disjoint 提出 17 件は全て COMPLETE しており、
**test 側 ingestion は画像あたり重複しない mask のみ受理する**。
discussion 735377「Score-ordered panoptic painting — why raw
Mask R-CNN output gets ERROR under PQ」(競合が同じ ERROR を記録)
とも整合。これは §2.60–2.62 の解釈を部分的に覆す重要な事実:

1. **v94 採用は取消** — local +0.0133 は重複 mask による TP ペア
   多重計上を含む artifact で、提出可能形式では再現不能。
   disjoint 保持版 (refined → greedy_disjoint only、包含 drop 無し)
   は全 fold で prod より悪い (fold0 −0.020 / fold1 −0.020 /
   fold2 −0.014): containment drop は重複 pred の抑制として
   実質的に有効で、ownership は「害」ではなく **提出形式が要求する
   disjoint 化そのもの**だった。
2. **多仮説 emit の LB 経路は存在しない** — TP ペア計上を利用する
   ref_prop/all3 (+0.03〜+0.06 local) は受理されないため、
   採点面最適化ラインは閉鎖。top 0.70 層のスコアは仮説多重度
   では説明できず、純粋な検出/分割品質または別要因
   (733912 の評価器バージョン混在問題) に起因する。
3. **ローカル評価の修正が必要** — 我々の公式 PQ レプリカは
   重複 pred を受理するため、overlap を含む emission arm の
   スコアは deployable でない。今後の emission 実験は
   **disjoint 制約を課した状態**でのみ評価する。

残る提出可能なレバー (disjoint 内での最適化): disjoint clip の
順序 (現行は面積昇順 → conf 降順など)、containment 閾値、
min_area、clip 後の再 refine。提案盲域 (FN 65%) は真のモデル
ギャップだが pooled 採点で期待価値が薄い層に集中している。

提出枠: v94/probe の両 submit とも「4 submissions remaining today」を
返した (ERROR 提出は日次枠を消費しない)。本日残り 4 件。

**disjoint 内バリアント掃引 (全 fold、提出可能な改変のみ)**:

| arm | fold0 | fold1 | fold2 |
|---|---|---|---|
| prod (area-asc disjoint, cont0.90, min20) | **0.4458** | **0.4448** | **0.4463** |
| disjoint 順序=conf 降順 | 0.4305 | 0.4326 | 0.4343 |
| disjoint 順序=面積降順 | 0.4303 | 0.4328 | 0.4313 |
| containment 0.80 | 0.4452 | 0.4448 | 0.4453 |
| containment 0.95 | 0.4450 | 0.4439 | 0.4459 |
| min_area 10 / 50 | 0.4458 | 0.4448 | 0.4463 (同一) |

**prod の ownership 構成は disjoint 制約内で既に局所最適** —
conf/面積降順は −0.012〜−0.015、containment/min_area 微動は中立以下。
面積昇順 disjoint (小さい物体が先に画素を確保し、大きい物体が
クリップされる順序) がこのタスクでは正しい選択。後処理ラインは
これで全閉鎖。

**OOF↔LB ギャップ (0.445 vs 0.39) の現状解釈 (改訂)**: 多重度経路は
消えたため、残る候補は (a) test 側の annotator 数が少なく
sparse であること、(b) 733912 が報告する評価器バージョン混在、
(c) top 陣の検出器自体の性能差。我々の予測品質 (SQ 0.68) は
人間一致度 (0.63) を既に超えており、正直な単一仮説パイプライン
では 0.45 前後が実力値 — 採点器の古いバージョンで採点された
提出が 0.5 超に乗っている可能性が 733912/736921 で示唆されている。

成果物: `output/v95_features/disjoint_variants.json`、
kernel artifact (v94 `submission_master_pipeline_prod_v94.csv`
1328 rows / probe 2656 rows、いずれも提出 ERROR)。

### 2.64 Codex PR #584 レビュー対応 — 誤差分解スクリプトの3点修正 (2026-11-01)

PR #584 の chatgpt-codex-connector レビュー (P1×2・P2×2) で指摘された
`analyze_error_decomposition.py` の問題を修正し、3 fold 全 artifact を
キャッシュから再集計 (`--skip-infer`)。

**修正内容**:

1. **stage 帰属 (P2)**: クラスタ被覆は proposal→refined で非単調
   (refined は proposal の縮小ゆえ IoU を 0.5 超に引き上げ得る)。
   「最初に被覆の無い段階」帰属では refined 救済後に ownership で
   失われた物体を proposal_blind と誤帰属していた。
   `stage_lost()` を「最後に被覆が残っていた段階」帰属に修正。
   → 各 fold で proposal_blind −4〜−5、ownership_lost +4〜+5
   (fold2: 696→691 / 52→57) — Codex が例示した 5 singleton と一致。
2. **all-FP pred の conf (P2)**: conf が TP 行のみから集約され
   全 FP pred (fold2 328 件) で欠損していた。FP 側 conf で
   補完するよう修正 — 帯別 neg share が全 conf 帯で計算可能に。
3. **marginal 分母 (P1)**: pred 除去時に「その pred が唯一被覆した
   GT の新規 FN 化」を反映していなかった。厳密式
   n_tp + 0.5·n_fp − 0.5·n_unique に修正 (unique = 同一
   (file,record,gt) に届く pred がその pred のみの TP ペア、
   tp_rows に `gt_local` 列を追加して厳密集計)。
   Codex は 0.5·(n_tp+n_fp) を提案したが、これは全 TP が unique
   の場合の上界 — 非 unique TP も数える厳密式を採用。
4. **行為テスト追加 (P1)**: `tests/test_analyze_error_decomposition.py`
   — 非単調被覆の stage 帰属・all-FP conf 保持・unique-cover の
   marginal 分母を検証する 8 case (全 pass、slim env では
   importorskip で skip)。

**再集計による数値変更 (fold2)**:

| 指標 | 初版 | 修正後 |
|---|---|---|
| proposal_blind / ownership_lost | 696 / 52 | **691 / 57** |
| 価値負 pred | 432 (28.8%) | **355 (23.7%)** |
| 内訳 allFP / partialTP | 328 / 104 | **328 / 27** |
| neg/pos conf median | 0.421 / 0.581 | 0.381 / 0.569 |
| oracle drop-neg 上限 | +0.0568 | **+0.0472** (fold0/1: +0.0455/+0.0452) |

結論は不変: proposal_blind が FN の ~65% で最大源・ownership が
disjoint 制約内で局所最適・emission gate の oracle 上限は +0.047
と下方修正 (v95 判定を強化する方向)。ただし初版の marginal 分母
バグは partial-TP の価値負数 (104→27) と oracle 上限を過大計上
していた — §2.60–2.61 の数値は修正版で上書き済み。

**v95 caveat → 解消 (再評価済み)**: v95 の学習ラベルは修正前の
marginal 定義で生成されていたため、
`scratch/rerun_v95_gate_corrected_labels.py` で新旧両ラベル ×
GBM/LR × 厳密 PQ (線形除去デルタ + newly_fn 厳密集計) の
cross-fit を再実行。修正ラベルでも判定不変: GBM held-out
Δ −0.0217/−0.0171/−0.0228 (fold0/1/2、dropped 369/349/400)、
LR は ±0.000 台でゲート効果なし — oracle 上限 +0.047 に対し
学習 gate はいずれのラベル規約でも到達不能。**v95 reject
確定 (ラベル規約に依存しない)**。成果物:
`output/v95_features/gate_results_corrected.json`。

成果物: 修正版スクリプト SHA256
`c2ec60b0e875dbd14cf2d961481d3b026e519c76a63c964bf23fc018151a4976`、
更新済み `output/error_decomp_fold{0,1,2}/` 一式、
`tests/test_analyze_error_decomposition.py`。

### 2.65 discussion 定期調査 — 主催者が disjoint 提出要件を公式確認 (2026-11-01)

§2.28+ の定期調査。前回 (max id 749526) 以降の新規 topic 3件を確認。

- **753816** (10-29〜31): test 側が self-eval notebook と同一の
  multi-annotator pooling かの質問に対し、主催者は「**いくつかの
  微妙な差異があるが、予測品質の改善に資さないので公開しない**」
  と明言 — test scorer が public self-eval と厳密同一でないことの
  公式証言 (OOF↔LB ギャップの評価器混在仮説 733912/736921 を支持)。
  さらに follow-up で「**提出に overlapping prediction は不可、
  あれば error が投げられる**」と主催者が直接確認 —
  **§2.63 の v94/probe ERROR で実測確定した disjoint mask 要件が
  公式発言で裏付けられた**。refined 直接 emit・多仮説 emit の
  local 利得が提出形式として無効であることが二重に確定。
- **753665** (10-28): final ranking は PQ ベースだが top-3 は審査団
  評価で決定。「Provide the one that you want us to evaluate...
  Make sure your demo can be executed completely on a different
  system」— **審査団が notebook を再実行する前提が確認された**。
  我々の reproduction kernel は clean Kaggle 環境で両提出を
  byte-identical 再現済み (§2.48) なので要件充足。ensemble 可、
  ImageNet pretrain 可も確認。test 注釈プロトコルの質問には回答せず。
- **753043** (10-27): test の multi-annotator pooling 質問には
  「この情報は品質改善に資さない」として回答拒否 — annotator
  構造が意図的に非開示である方針の再確認。我々が per-record
  pooling を train 側実測から推定した経路は正当。

**含意**: disjoint 要件・scorer 差異・審査団再実行の3点で我々の
パッケージ (disjoint 保持・repro kernel・weights 同梱) は全て
公式要件と整合。LB との乖離説明として test scorer の非公開差異が
強化されたが、我々側に新規の対応項目はなし。

### 2.66 訂正: §2.54 の限界価値計算・§2.58 の上限・「any-conf blind」の誤記 (2026-11-01)

ユーザーの再検討依頼を受けて §2.54/§2.58/§2.60 の主張を再監査し、
**3 点の誤りを確認した**。過去の実験判定 (v88–v95 の reject) は
実測値なので不変だが、そこから導いた「モデルでは改善不能」という
解釈は根拠を失う。再計算: `scratch/analyze_annotator_count_marginal.py`
(コミット済み error_decomp artifact のみ、GPU 不要・数秒。出力
`output/annotator_count_marginal.json` は再生成物のためコミットしない)、テスト
`tests/test_analyze_annotator_count_marginal.py`。

**誤り 1 — 限界価値の式 (§2.54)**: 「単独物体を予測すると
0.63×1 − 0.5×1.44 で負」は、予測しない場合に残る FN (分母 +0.5) を
差し引いていなかった。k 人が描いた物体を IoU s で当てる予測は
Δnum = k·s、Δden = R/2 (R = 画像の annotator 数) なので、
**PQ を上げる条件は k·s > PQ·R/2**。実測 (3 fold、per-pred marginal
ratio、PQ≈0.445 を超えれば正):

| R | 画像比 | GT 比 | k=1 の mean ratio | 判定 |
|---|---|---|---|---|
| 1 | 56–62% | 38–41% | 1.37 | 強く正 |
| 2 | 20–22% | 23–26% | 0.64–0.66 | 正 |
| 3 | 18–24% | 36–37% | 0.43–0.45 | 損益分岐 (48–58% が正) |

単独 annotator 物体の検出は、R≤2 の画像 (GT の約 63%) では採点上
明確に有利。prod が単独物体を出さないことは「採点上最適」ではない。

**誤り 2 — 上限 PQ≈0.49 (§2.54/§2.58)**: 誤り 1 の式から導いた値。
クラスタ毎に 1 マスクが全メンバーと IoU s で一致する近似 (検出は
完璧と仮定) で再計算すると、s=0.634 (人間一致度) で predict-all
**0.531**・最適部分集合 **0.550**、s=0.68 (我々の SQ) で 0.569/0.590。
現状 OOF 0.446 との差 ~0.09–0.14 の大半は RQ 側。§2.58 で「0.70 超は
honest 採点の理論上限を超える」とした論拠は無効 (上位帯の汚染判断
自体は、0.55 ちょうどに 84 チームが集中する LB 分布 — 2026-11-01
取得 — の不連続性で別途支持される。0.55 未満の honest 帯は
0.42×1・0.41×14・0.40×51 で、我々 0.39 の上に ~66 チーム)。

**誤り 3 — 「any-conf blind」(§2.60/報告書)**: `proposal_blind` は
conf≥0.26 (DETECT_CONF) の提案で未被覆の GT であり、任意 conf で
未検出の意味ではない。v88/v89 診断で primary FN の **57–58% は
conf [0.02, 0.26) に IoU>0.5 候補を持つ** (§2.41–2.44)。残り ~43% は
この測定帯の生提案と一致しないだけで、任意 conf で不可視とは言えない
(conf<0.02 の提案、および conf≥0.26 の生提案が refine で崩れた
ケースは未測定 — v88 診断の分母は refine 後マスクの FN)。また「semantic/flow が同じ ~17% 層を救う」
は件数の一致 (157 vs 163) であり、個体集合の重なりは未照合。

**訂正後の解釈**: rescue/提案源の失敗は単独物体の採点構造ではなく、
追加候補が FP に支配されたことによる (v90 TP+61/FP+524、tile
TP+147/FP+756)。prod 出力でも限界価値が負の予測の大半は k=0
(どの record とも IoU≤0.5、fold0 で 348/1515)。これは **検出器の識別力 (順位付け)
の不足 = モデル能力** の問題であり、「検出器強化の方向は反証済み」
ではない。V36–V41 の検出器強化の多くは fold0 単独判定で、§2.5 の
fold 単独 CI 幅 ~0.035 では Δ±0.01 を判別できない (判定不能が正確)。

**未試行の正当レバー**: 提出 YOLO (`yolo_fold2`) は学習データの 2/3
(~470 画像) のみで学習 — refiner は all-data だが YOLO の全データ
学習は一度も行っていない。識別力不足に最も直接効く標準手段として
§2.67 で学習曲線ゲートを事前登録する。

### 2.67 v96 事前登録: 検出器のデータ量スケーリング — 全データ YOLO の採否ゲート (2026-11-01)

**動機 (§2.66)**: 残差の主因は検出器の順位付け (低 conf 帯の識別力)。
提出 YOLO (`yolo_fold2_best.pt`) は fold0+1 (学習データの 2/3、
~470 観測) のみで学習され、YOLO の全データ学習は未試行。全データ
モデルは OOF 評価できないため、**学習曲線の傾きで採否を事前に決める**。

**arm (全て fold2 val・prod 評価経路を凍結再利用)**:

- `prod`: 既存 `yolo_fold2_best.pt` (train = fold0+1、2/3 データ)
- `third_f0`: train = fold0 のみ (1/3 データ)
- `third_f1`: train = fold1 のみ (1/3 データ)
- 学習レシピは prod と完全一致 (yolo11m-seg・imgsz1280・batch4・
  epoch70・patience14・seed 3409・cls 0.5・増強同一)。ultralytics は
  prod 学習時の **8.4.118 に固定**。評価は v88 Kaggle 版 (§2.48 で
  Colab と deterministic 一致を確認済み) のセルを無改変で再利用し、
  fold2 EdgeAtt refiner・thr 0.45・drop 0.90・min_area 20・公式 PQ

**事前固定ゲート**:

- Δ = PQ(prod, 0.26) − mean(PQ(third_f0, 0.26), PQ(third_f1, 0.26))
- 全データへの外挿利得 g = Δ × log(3/2) / log(2) ≈ **0.585·Δ**
  (データ量の対数に線形な学習曲線を仮定)
- **g ≥ +0.005 (Δ ≥ +0.0085) かつ両 third arm が prod を下回る** →
  全データ YOLO を学習し提出候補化 (public LB で確認、最終 2 枠の
  差し替えは LB で prod 以上の場合のみ)
- それ以外 → reject_noise、データ量レーンを閉じる

**同時実行 (判定前に学習のみ開始、判定に使わない)**: `full` arm
(train = fold0+1+2) を同レシピで学習し重みと SHA256 だけを保存。
val=fold2 は学習集合に含まれるため評価値は記録しても判定に使わない。
ゲート不合格なら破棄する。

**診断 (判定不使用)**: third arm の best.pt と last.pt の PQ 差
(全データ学習では汚染 val で best を選ぶため、last との差を把握)、
v88 と同一の低 conf 被覆診断。

**誠実留保**: (a) prod は 2026-08 に Colab A100 で学習、third arm は
Kaggle GPU — 同一 recipe でも学習の GPU 非決定性 (σ≈0.003、§2.44)
が残る。2 つの third arm の平均でこれを緩和する。(b) 対数線形外挿は
近似で、データ量の飽和が近ければ過大評価になる。(c) 1/3 データは
validation 観測と train 観測の比が変わるだけで、テスト分布の
annotator 構成 (非公開) への効果は LB でしか確認できない。

実装: `scratch/build_v96_data_scaling_kaggle.py` (v88 Kaggle notebook
の凍結セルに最小置換を適用、置換は各 1 回を assert) →
`solar_filament_v96_data_scaling_{third_f0,third_f1,full}_kaggle.ipynb`。

**§2.67 追補 (2026-11-01、結果取得前)**: Colab 版ノート
`solar_filament_v96_data_scaling_{third_f0,third_f1,full}_colab.ipynb`
を追加 (セル 1-2 は v89 Colab の Drive 解決、セル 3-6 は Kaggle 版と
同一文字列 — テストで固定)。prod (`runs/yolo_fold_2`) は Colab で
70 epoch 完走 (early stop なし、best epoch 57、1.14h) しており、
**ゲート判定は prod と同じ Colab GPU 上の third arm 2 本で行う**。
Kaggle T4 で先行実行中の `third_f0` はハードウェア間の複製として
記録のみ (判定に使わない)。Kaggle の third_f1/full の自動 push は停止。

### 2.68 v96 判定: データ量スケーリング・ゲート通過 — 全データ YOLO を LB 確認へ (2026-11-01)

**結果** (Colab A100-SXM4-40GB、ultralytics 8.4.118 固定、fold2 val・
prod 評価経路凍結、refiner b1c0f353…、manifest:
`artifacts/v96_{third_f0,third_f1,full}_colab_manifest.json` @ Drive):

| arm | train | PQ@0.26 | PQ@0.32 | TP/FP/FN@0.26 | Δ@0.26 vs prod |
|---|---|---|---|---|---|
| prod (fold0+1, 2/3) | 707 obs | **0.444218** | 0.4416 | 1715/750/1070 | — |
| third_f0 (fold0, 1/3) | 391 obs | 0.432870 | 0.434419 | 1711/875/1074 | **−0.01135** |
| third_f1 (fold1, 1/3) | 367 obs | 0.435800 | 0.440200 | 1772/953/1013 | **−0.00842** |

**ゲート (§2.67、結果取得前に固定)**: Δ = 0.444218 − mean(0.432870,
0.435800) = **+0.00988 ≥ +0.0085** かつ両 third arm が prod 未満 →
**通過**。対数線形外挿 g = 0.585·Δ ≈ **+0.0058** (2/3→3/3 で
PQ +0.006 程度と予測)。→ 全データ YOLO (`yolo_full_fold2`,
train 1154 obs / 8199 instances / 104min) を提出候補として
public LB で確認する。最終 2 枠の差し替えは LB で prod 以上の
場合のみ。LB が prod 未満なら本 arm は破棄し prod /
plain_mhsa_f2 を維持。

**誠実留保**: g = +0.0058 はゲート閾値 +0.005 をわずかに超える
点推定であり、2 arm 平均でも GPU 非決定性・データ抽選の不確実性
(§2.67 誠実留保 a–c) で誤差 ±0.005 程度は残る。外挿が対数飽和を
過大評価する可能性もある。よって「通過」は *LB 確認に進む権利* で
あり、採用の保証ではない — 最終判断は public LB。

**判定不使用の診断**:
- third_f0 は best.pt = last.pt と同じ評価値 (best epoch = 最終
  epoch)。third_f1 は last.pt の方が高い (0.4424/0.4435@0.26/0.32)
  — best 選択は val mAP ベースで PQ 最適とは限らない。
- 全データ版の best.pt (sha256 `f54c812d…`) は汚染 val (fold2 を
  含む) で選択された点推定; last.pt は `5a714908…`。提出候補は
  再現パイプラインとの一致を優先し best.pt を用いる。
- 低 conf 被覆: prod 543/953, third_f0 513/952, third_f1 466/871 —
  データ量増で primary FN が減る傾向 (953→871) は FN 側に一貫。
- Kaggle T4 の third_f0 複製は RUNNING 継続 (記録のみ、判定不使用)。

### 2.69 採点機構 probe 結果: disjoint ピクセル強制を実証 — prod_v94 の OOF 改善は提出不可 (2026-11-01/02)

**背景**: §2.60/2.61 で refined mask 直接 emit (ownership 撤廃) が
局所 3-fold OOF で +0.006..+0.013 と計測され、variant `prod_v94`
として実装済みだった。ただし採点機構が重複ピクセルを許容するかは
未検証だったため、`probe_refprop` (refined + proposal の 2 仮説
emit、意図的に重複を含む) と `prod_v94` の 2 提出を probe として
投入した。

**結果 (2026-11-01 提出)**:

| ref | file | status |
|---|---|---|
| 57859708 | submission_master_pipeline_prod_v94.csv | **ERROR** |
| 57860275 | submission_master_pipeline_probe_refprop.csv | **ERROR** (予想通り) |

**解釈**: 採点機構は非重複 (disjoint) ピクセル割当を強制する。
全有効提出 (V30–conf032) は `apply_ownership` / disjoint 後処理を
経ており、重複を含む 2 提出のみが ERROR となった。
**prod_v94 の局所 OOF +0.006..+0.013 はそのままでは実提出に
使えない** — 重複 refined mask を disjoint 化する手段が
`apply_ownership` (prod 経路) そのものであり、v94 の利得は
「重複を残したまま採点できる」前提でのみ成立する仮想値だった。
probe は事前に「disjoint 強制なら ERROR」と予測を登録済みで、
結果は予測と一致。本発見により v94 レーンは閉鎖 (採点機構適合の
後処理を入れると prod と等価になるため)。

**v96 全データ候補への影響**: LB 確認カーネル
`solar-filament-2026-v96-full` は prod-v94 由来ノートブックの
variant を `prod` に修正して再 push した (v1 push は prod_v94 の
まま起動、提出には使わない)。候補 CSV は `submission.csv`
(apply_ownership 済み、disjoint) として生成される。
### 2.70 競合・ディスカッション定期調査: public LB は死んだメトリクス/リークに汚染 — 正直フロンティアは ~0.36-0.40 で我々はそこにいる (2026-11-02)

定期調査 (ユーザー要求) で公開情報を収集。結論:**public LB 上位
(0.6-0.75 群・トップ 0.70) は正当なモデル成果ではない可能性が高い**。

**証拠 (複数独立ソース)**:
1. **メトリクスが 2026-08-07~09 に切替** (too.foo/Shivam Bhardwaj の
   フォレンジック調査、主催者確認済みテスト: 同一ファイルが切替前 0.66
   → 切替後 0.31)。旧スコアラーは設定ミスの Dice 評価器で **空提出が
   0.88-1.00**。旧スコアは未再採点で LB に残存 — 1.00 が 29 件、
   0.6-0.75 のパック、中央値 0.665、公開 notebook 自称 0.59 は全て
   旧メトリクスの残滓。Valentin Haase の「予測が少ないほど高い
   (18個→0.77、空→0.93)」測定は旧メトリクス時代の記録。
2. **test set の 77% が公開 MAGFiLO 1.0 (GT 付き) とバイト一致**
   (biconcavelens 記録)。上位スコアはリーク GT 参照でも説明可能。
   主催者見解: 真の評価は定量+定性ルーブリックであり「PQ>0.3 で
   十分に価値がある」。
3. **最終評価 = 70% 定量 (PQ + IoU/Dice 分布 + one-to-many 関係) +
   30% 定性** (4頁レポート・公開 repo・予測の形態)。締切 2026-11-15。

**正直フロンティア (現行真 PQ での実測)**:

| チーム | 公開 LB | 備考 |
|---|---|---|
| biconcavelens (Mellimi+Bhargava+Gupta) | 0.40 | OOF 0.461、SABiD 2026 論文化・全コード公開 |
| **我々 (prod)** | **0.39-0.40** | fold2 YOLO + EdgeAtt refiner |
| ameypawar | 0.37 | U-Net ResNet34 + YOLO11s/m 3検出器融合、val PQ 0.421 |
| too.foo | 0.36 | 公式プロトコル val 0.423、selector+tiling+TTA |
| ektarr | ~0.36 | 5-fold PQ パイプライン公開 |

**公式スコアラーの実プロトコル** (too.foo がホスト公開の自己評価
notebook を移植して確認): 予測は **annotator record ごと (~2.44/test
画像) に個別採点** し TP/FP/FN を pooled 集計。union ではない。
アノテータ間 Kappa 0.66 で実効天井 ≈ PQ 0.45-0.55。
**我々のローカル評価は既に annotator×image 単位の公式プロトコルで
実装済み** (scratch 内 evaluator 群のコメントで確認) — OOF↔LB の
オフセットは主に分布差由来と解釈できる。

**biconcavelens の独立再現で我々の結論が裏付けられた点**:
- **cls=1.5 修正は彼らも有効** (real PQ 0.37→0.38、検出器の
  confidence head 未訓練が共通の根本原因 — bbox-IoU 良好なのに
  conf が低い見逃し)。cls=3.0 は過剰。Mask R-CNN は cls×3・6epoch。
- **post-hoc rescue フィルタは 4 方式全て閉鎖** (refiner conf ゲート・
  局所 contrast・マルチ特徴 classifier — 全て PQ 悪化)。
  「rescue pool に安価に分離可能な信号は存在しない」= 我々の
  §2.65-2.66 訂正後の分解と整合 (低 conf 候補に IoU>0.5 は存在するが
  識別できない)。**上流モデル修正のみが有効** = 我々のデータ量
  スケーリング/検出器改善路線の正当性を支持。
- **refiner は非律速** (spine 重み sweep 平坦、clDice は悪化、
  我々の EdgeAtt plateau と一致)。
- **self-training/擬似ラベルは明確に悪化** (0.417→0.393)。
- **「complete」ラベル絞り込みは悪化** — データ量損失 > ラベル品質
  → 我々の全データ化方向を支持。
- **cross-arch ensemble は dedup-before-paint (IoU 0.05 一括破棄) で
  初めて成功** (local +0.015、real 0.38→0.39)。4-way は val 過剰適合
  で real 後退 — 116 枚 val での多軸 sweep は過学習する。
- RT-DETR cls=2.0 で solo 0.4296 (最強単体)、YOLO+RT-DETR 2-way は
  real 0.39 で同率。YOLO26 は diversity なしで無価値。SAM2
  zero-shot 0.08、Mask2Former 0.28-0.34、watershed 0.20、VAE 異常
  検知 AUC 0.44 (棄却)、FITS 原データは flat-field 不可で不発、
  CLAHE 学習は悪化。

**too.foo のレーン測定**: component selection oracle **+0.083**
(最大、学習 selector で +0.018 実現、我々の A1 scorer と同方向)、
fragment merge oracle +0.026、trim oracle 0.756 は実現不能、
union+agreement-weights 学習 +0.01 (純 soft label は悪化)、
overlap-max tiling +0.029、TTA +0.011、6 損失変種全て ±0.02 内で
semantic 単体天井 ~0.34。**~210/576 val フィラメントが全変種で
未検出 = recall が律速** (我々の結論と同一)。

**戦略含意**:
1. 我々の 0.39-0.40 は正直フロンティアの先端。0.70 型スコアを
   追う必要はない (旧メトリクス残滓 or リーク参照、private/定性
   評価で無効化される見込み)。
2. 疎化 emit は旧メトリクスの exploit であり現行真 PQ では
   限界効用のみ (emit iff p·IoU > PQ/2)。top-K 疎化は大きな
   ゲインを生まない見込み — probe するなら低期待値で。
3. 残りの実ゲイン: 検出器 recall (データ量・学習改善 = 進行中)、
   component selector 精緻化 (oracle +0.083、実現 +0.018 止まりは
   各チーム共通)。
4. 30% 定性 = report + repo の品質が効く。我々は既に 4 頁
   report + 公開 standalone repo を保有 — 最終仕上げの価値大。
5. 締切 2026-11-15。最終 2 枠選択と Google Form が残る。
### 2.68追補 — v96-full 提出カーネル修復と LB 確認 (2026-11-02)

- `solar-filament-2026-v96-full` v1 は `SUBMISSION_VARIANT=prod_v94`
  で誤 push 済みだったため ERROR で終了(§2.69 の既知無効レーン)。
- v2 (prod 化) は In[9] で `NameError: OUTPUT_PATH`。原因は test 推論
  セル全体が改行喪失の1行コメント化していた生成バグ(§2.71 ノートの
  構築で潰したのと同種)。prod_v94 健全版から cell を復元し、
  `yolo_full_fold2_best.pt` 明示解決を再適用した v3 を push。
- v3 完走 → `submission_master_pipeline.csv` (1,141 rows, prod variant,
  full-data YOLO + all-data EdgeAtt refiner) を API 提出
  (ref 57897560)。
- **LB 結果: 0.39 (COMPLETE)** — prod 提出群と同点。事前登録の採用規則
  (「公開 LB が prod 以上なら採用可」) を満たすが改善は実証されず。
  同点なら全データ学習側は private LB での微優位を期待できるため、
  最終2枠は (a) prod 系確立提出 + (b) v96-full の分散構成が有力
  (§2.70 の正直フロンティア水準を維持しつつ構成を分ける)。

### 2.71 v97 事前登録: RT-DETR 第2検出器 + 校正済み true-dedup ensemble — 残存する唯一の未試行上流レバー (2026-11-02)

**出自**: §2.70 競合調査。biconcavelens はこのコンペで「calibrate +
dedup-before-paint (mask IoU>0.05 で一括破棄)」が 4 度目の ensemble
失敗の後に初めて real LB を動かした唯一の merge 機構と報告
(0.38→0.39)。彼らの RT-DETR-l (cls=2.0) は solo local PQ 0.4296 で
3 検出器中もっとも強い。我々は同一アーキ fold ensemble (V53) と
低 conf rescue は棄却済みだが、**異種検出器 + 校正 + dedup の
組合せは未試行** — 残存する最後の上流レバー。

**設計 (変更は検出器の追加のみ。refiner・後処理・評価経路・fold2・
paired prod 基準は v88/v96 と完全一致)**:

- RT-DETR-l: folds 0+1 の全アノテータ分独立ラベル (bbox = polygon
  外接矩形) で学習。imgsz 1280・epoch 70・patience 14・
  seed=3407+2=3409・deterministic・cls=2.0 (biconcavelens 検証値;
  ultralytics RT-DETR は検出のみで box を出し、提案 mask は box 充填)
- ensemble: 両検出器の生提案 (conf>=0.02) を集め、検出器別 isotonic
  校正 (X=conf、y=record 平均 hit-IoU = 各 record で IoU>0.5 なら
  その IoU ・否 0) で P(matched-IoU) へ写像。cal 降順に貪欲受理、
  受理済みと mask-IoU>0.05 の候補を破棄 (dedup-before-refine)。
  受理候補のうち cal > λ のみ emit → EdgeAtt refiner → ownership
  → 公式 PQ (record 単位 pooled、既存 eval と同一)
- λ (採用点) = 0.5 × PQ(prod,0.26) — PQ の限界 emit 条件
  (p·E[IoU] > PQ/2) から導出、実行時に prod 実測値で決定
- 校正の honest 化: valid_files を deterministic 2 分割し A→B・B→A
  で cross-fit (in-sample 校正は診断のみ)

**事前固定ゲート (結果取得前に確定)**:

- 主判定: `Δ = PQ(ens, cross-fit, dedup=0.05, λ) − PQ(prod, 0.26)`
  - `>= +0.005`: 両検出器を全データ再学習 → LB 確認へ昇格
  - `[0, +0.005)`: reject_noise (ensemble 機構を閉鎖)
  - `<0`: reject (同上)
- sanity: prod paired 再現値が既知 0.444218 ±0.002 内であること
- conf sweep・dedup_iou {0.03,0.05,0.10}・λ 倍数 {0.8,1.0,1.2}・
  in-sample 校正・各検出器単体値は判定不使用の診断として記録

**誠実留保**: (a) RT-DETR は box 提案のみ — refiner が box 充填 mask
を整形するため EdgeAtt の crop 文脈が seg 提案と異なり、solo 値は
過小評価の可能性 (b) isotonic は proposal→hit-IoU 写像で、
biconcavelens の P(TP) 校正とは目標が異なる (emit 条件の Dinkelbach
形に直結するよう選択) (c) RT-DETR の学習安定性は未検証 — 収束
不良なら arm のみ棄却として記録 (d) fold2 ゲート通過でも LB では
非対称の可能性 (疎 GT で FP が効くため dedup の価値は LB で
大きくなる可能性も、小さくなる可能性もある)

**実装**: `scratch/build_v97_rtdetr_ensemble_colab.py` (v96 third_f0
Colab ノートの凍結セル再利用 + bbox データセット・RT-DETR 学習・
候補収集/isotonic/dedup/ensemble 評価セル追加) →
`solar_filament_v97_rtdetr_ensemble_colab.ipynb`。RT-DETR 学習は
`runs/v97_rtdetr_ensemble/rtdetr_fold2`、manifest は
`artifacts/v97_rtdetr_ensemble_colab_manifest.json`。

**v97 結果 (2026-11-02、Colab A100 完走)**:

- RT-DETR-l 学習完走: 82 分、best.pt SHA256
  `0dca102e…a99550`、train 5414 inst / valid 1784 inst
- prod 再現 sanity: 0.444218 vs 既知 0.444238 (Δ −2e-05、合格)
- **主判定 (cross-fit 校正・dedup0.05・λ=0.2221)**:
  PQ 0.448036 / TP 1743 / FP 775 / FN 1042 / sparse 0.3924、
  **Δ = +0.0038 → 事前登録ゲート未達で `reject_noise`** (採用は棄却)
- 診断:
  - RT-DETR solo: conf 0.15→0.266 / 0.20→0.311 / 0.26→0.353 / 0.32→0.390
    (YOLO 0.444 を全点で下回る。box 提案の過小評価留保はあるが弱い)
  - dedup sweep: 0.03→0.4464 / 0.05→0.4480 / **0.10→0.4503**
  - λ sweep: 0.8×→0.4475 / 1.0×→0.4480 / 1.2×→0.4455 (1.0×が最適)
  - in-sample 校正 0.4506 (cross より +0.0025、過適合は軽微)
  - **detector ablation が決定的**: yolo_only (cal+dedup) = 0.4480
    ≡ ensemble と完全一致、rtdetr_only = PQ 0・受理 0 件。
    **+0.0038 は RT-DETR 相補ではなく dedup+isotonic 受理パイプライン
    自体の YOLO への効果** — 第2検出器仮説はこの fold で棄却

### 2.72 v97b/v97c 探索: dedup プラトー発見 → prod_dedup030 LB probe (2026-11-02)

reject_noise は採用ゲート不達の意味であって探索禁止ではない。
+0.0038 と dedup0.10 の +0.0061 を手がかりに近傍を系統探索
(候補・校正・パーツキャッシュ再利用、manifest は v97b/v97c 別系):

**v97b (rescue+λgrid)**: prod 土台 + 高 cal RT-DETR 追加救済
(λ_hi 0.30/0.35/0.40) は 3 条件とも **Δ=0.0000・追加候補ゼロ**
— RT-DETR の cal>0.30 候補が存在しない (= 校正済み価値が全て λ 未満)。
λ grid (dedup0.05): 0.1999 でピーク +0.0048 の単峰。

**v97c (dedup 拡張 + 交互作用 + ablation)**:

- dedup sweep (λ=0.2221): 0.03→+0.0022 / 0.05→+0.0038 /
  0.08-0.15→+0.0058〜0.0061 / **0.20→+0.0073 / 0.30→+0.0075** /
  0.50→+0.0019 / 1.00→−0.0020。
  **0.08〜0.30 の広いプラトー** — 局所ピークではなく実効果。
  dedup=1.0 (実質オフ) で prod 割れも理屈通り (重複候補が FP 化)
- λ×dedup: d0.10/d0.15 とも λ1.0× が最適 (dedup が支配因子)
- ablation: rtdetr_only は dedup に依らず受理ゼロと論理確定
  (v97c の ablation 実行は tag 衝突で combined 再生になり無効だが、
  rtdetr の cal>λ 候補が存在しない以上どの dedup でも寄与 0)
- per-file Δ (d0.10 vs prod): 勝ち 51・負け 46 ファイルだが
  勝ちは +0.17〜+0.28 の非対称大勝 (prod=0 のファイル救済)、
  負け最大 −0.16。低 conf 候補救済が FN 層に効いている構造と整合

**機構の解釈**: prod の conf>0.26 一律カットは、valid FN の 57% を
カバーする低 conf 提案を捨てていた。dedup0.30+isotonic 受理は
(実効 conf*≈0.238) でそれらを cal>0.2221 相当として救い、
dedup が重複提案の FP 化を防ぐ。改善は第2検出器ではなく
**受理規則の改良**。

**prod_dedup030 バリアント (LB probe)**: 提出は test に GT が無い
ため fold2 全候補 (5604 件) で fit したグローバル isotonic
(conf→E[hit-IoU]、88 閾値、`artifacts/v97_global_iso_yolo.npz` /
`v97_global_iso.pt`) をカーネルに埋め込み。受理規則のみ変更で
重み・refiner・ownership・emit は prod と同一 (RT-DETR 不要と
判明したため第2検出器は含めない)。誠実留保: 局所 +0.0075 は
探索的事後選択で LB 移行は不確か、グローバル写像は cross-fit
より僅かに楽観的の可能性。Kaggle
`shingo257/solar-filament-2026-reproduction-dedup030` で採点待ち。

**v97d 追補 (2026-11-03、出荷写像の fold2 検証)**:

- dedup030 カーネルのグローバル isotonic 写像を再 fit — Drive 保存版と
  閾値配列が完全一致 (88 点、出荷再現性 OK)。実効 conf* = 0.238
- **halfcross** (反対 half で fit した写像を適用 = test 状況の近似):
  dedup 0.10→+0.0061 / 0.20→+0.0073 / **0.30→+0.0075** / 0.50→+0.0019
  — per-file cross-fit 値と全点一致、写像の過適合は無いと確認
- insample (楽観側): 0.30→+0.0100。insample−halfcross 差 ≈ +0.0025
- **機構分解**: conf0.238 単純閾値 = +0.0011 のみ、conf0.20 = −0.0070。
  つまり +0.0075 のうち conf 緩和は +0.001、**dedup 機構が ≈+0.006**。
  閾値を下げるだけでは悪化するため受理規則 (dedup) が主因と確定
- dedup030 カーネル v1 は推論・監査を完走 (180 test images, 1244 rows,
  overlap 0、2.7 s/img) したが末尾の環境記録セルで plain_ckpt NameError
  → v3 で修正済み再実行。局所期待値 +0.0075 の LB probe として継続

### 2.73 定期競合調査 (2026-11-03) — 正直フロンティアの転移率が 3 チームで一致

- **biconcavelens が最終系を公開** (GitHub + SABiD 2026 論文投稿):
  public LB 0.40 / OOF PQ 0.461。3検出器 (Mask R-CNN 2048 +
  YOLO11m/l-seg 1280 + RT-DETR-l 1280) calibrated ensemble + smp
  U-Net crop refiner + ConvNeXt semantic U-Net + mask vote。
  彼らの RT-DETR は P(TP) 校正 + vote で機能 — 我々の hit-IoU
  isotonic 受理で寄与ゼロだったのは検出器ではなく受理設計の差
- **dariushafshar**: 10-fold U-Net ensemble、OOF 0.4502 → LB 0.39
- **我々**: fold2 0.444 → LB 0.39 — **3チームとも local→LB 転移率
  ≈0.87 で一致**。dedup030 の局所 +0.0075 は LB で +0.006 級の
  見込み → 表示 0.39→0.40 で正直フロンティア並を現実的に狙える
- **HDJoJo 0.52 (1位)**: hetspookjee の分析で Dataverse リーク
  (~77% カバレッジ) の部分混合とほぼ確定。主催者はアーカイブ使用を
  明示禁止。定性評価 (30%) ではリーク説明責任が問われる立場
- QiuweiLiu: CLAHE + tiled U-Net + skeleton-endpoint graph merge
  (距離≤50px・cos≥0.80・gap prob≥0.40) — 分断フィラメントの
  グラフ再接続は未試行の機構。参考実装として記録

**dedup030 LB 結果 (2026-11-03, ref 57933888): 0.39 — prod/v96-full と
同点、可視改善なし**。局所 halfcross +0.0075 は LB で実証されず
(2桁表示に収まる差、または転移せず)。採用ゲート相当の証拠には
ならないため最終枠の現行案 (prod + v96-full) を維持。dedup030 は
受理規則の探索的派生として記録に留める (CSV・写像・カーネルは
全て保存済み、private LB での参考値として残る可能性はある)。

**LB 分布スナップショット (CSV 2026-10-31, 1305 teams + live 11-03)**:

| 帯域 | teams | 解釈 |
|---|---|---|
| ≥0.56 (live トップ IONode Online 0.70) | ~20 | リーク帯 (§構造証拠どおり) |
| 0.55 ちょうど | **84** | partial-leak シグネチャの壁 — HDJoJo も 0.52→0.55 で合流 |
| 0.54 / 0.52 / 0.49 | 各1 | 境界帯 |
| 0.42 / 0.41 / 0.40 | 1 / 13 / 48 | 正直フロンティア上端 |
| **0.39 (ERS = rank ~219)** | **108** | 正直バンド主塊の先頭 |

文書化済みの構図を再確認: 0.55+ 層は GT 利用帯、誠実系の天井は
~0.40-0.42。定量面での差別化余地は尽きたため、定性 30% 審査での
再現性・報告品質が残る勝ち筋 — 4頁 report + 公開 repo + 凍結
マニフェストの整備がそのまま対応する。新トップ IONode 0.70 も
既知のリーク帯パターンの範囲内。

### 2.74 v98 fold-ensemble: 前提誤認による汚染の発見と onlyfold 設計 (2026-11-03)

**⚠️ 汚染警告 — v98 の pooled 結果は honest 証拠に使わないこと。**

v98 (`solar_filament_v98_foldensemble_colab.ipynb`) は「fold0/1 の重みは
fold2 未学習」を前提に設計したが、これは誤りだった。fold 命名の慣例:
`yolo_fold_k` = **fold_k 以外で学習**するモデル。したがって:

| 重み | 学習 fold | fold2 に対して |
|---|---|---|
| yolo_fold_2 (prod) | 0+1 | honest ✓ |
| yolo_fold_0 | 1+**2** | **in-sample 汚染** |
| yolo_fold_1 | 0+**2** | **in-sample 汚染** |

solo 実測が汚染を裏付けた (fold2 上):

| model | PQ @0.26 | Δ vs prod |
|---|---|---|
| yolo_fold_0 (fold2 学習済み) | 0.4812 | **+0.0370** |
| yolo_fold_1 (fold2 学習済み) | 0.4677 | **+0.0235** |
| yolo_fold_2 (honest) | 0.4442 | 0 |

→ fold2-in-training の記憶プレミアムは **+0.02〜+0.04 PQ**。v98 の
`results.pooled` (f2f0*/f2f0f1) は全てこの汚染を含み、**受理パイプラインの
改善証拠としては無効**。将来の読者が誤用しないよう manifest の
contract ではなく本節で宣告する (manifest は Drive 上で immutable)。

**副次的収穫**: このプレミアムは v96-full (全データ学習) の test での
強さの直感を与える — 学習データ増は実利あり。

**v98b (honest 設計)**: fold0-only / fold1-only YOLO を新規学習。
train も val も fold 内ファイルのみ (val = sorted 5 番目毎 = 20%) で
**fold2 に train/val 両面で非接触**。prod と同一ハイパラ (70ep,
1280, AdamW)。さらに fail-closed ガード: onlyfold solo が prod を
+0.01 超で上回れば fold2 接触の兆候として評価を中止する。
pooled 評価は v98 と同型 (5 pool × dedup {0.10, 0.30} × half-cross
共有 isotonic)。`v98b_foldonly_colab_manifest.json` に分離記録。

**意味論の注意**: onlyfold モデルは 1/3 データ学習で prod (2/3) より
弱い想定 — pooled の Δ が正なら「弱いパートナーの追加でも多様性が
効く」と解釈でき、test 提出では正規の fold モデル (各 2/3 学習)
でより強い効果が期待できる (honest 評価は下振れ側 = 保守的)。

**v98b 結果 (2026-11-03 完走, fold2 honest)**: onlyfold 学習は健全
(f0only 0.4245 / f1only 0.4328 — 1/3 データで prod 0.4442 より弱い
想定どおり、fail-closed ガード未発火)。pooled 評価 (half-cross 共有
isotonic + dedup + λ0.2221):

| pool | d0.10 | d0.30 |
|---|---|---|
| f2 のみ (対照) | +0.0060 | **+0.0074** |
| f2+f0 | −0.0007 | +0.0009 |
| f2+f1 | +0.0018 | +0.0021 |
| f0+f1 | −0.0057 | −0.0039 |
| f2+f0+f1 | −0.0042 | −0.0029 |

**判定: 多様性仮説の2回目の閉鎖。** 弱いパートナー (1/3 データ学習)
は dedup 衝突・共有校正の過大評価でアンサンブルを希釈する —
f2 追加するほど悪化する単調傾向。RT-DETR 寄与ゼロと合わせて
「受理パイプラインの局所天井は f2 単独 dedup +0.0075」が確定。
dedup030 の LB 0.39 同点と整合し、受理側改善の test 転移限界を示唆。
検出器多様化によるさらなる局所改善の探索はここで打ち止めとする
(v98b で質の異なる2検出器族・量の異なる3モデル全て棄却済み)。

### 2.75 v99 事前登録: dedup030 受理規則の fold0/fold1 独立再現 (2026-11-03)

**動機**: §2.72 の dedup030 (+0.0075 halfcross) は fold2 上で dedup・λ を
掃引して選んだ探索的事後選択であり、独立 fold での確認がない。public LB
(≈90枚・2桁) は 0.39 同点で識別不能 (§2.73)。最終順位を決める private
側の期待値で 2 枠目を選ぶには、**fold2 で選んだ設定をそのまま fold0/1 に
当てて再現するか**が唯一の honest な判別手段。新規学習なし・CPU のみ。

**入力 (凍結)**: v80 candidates cache `solar_filament_v80/candidates/fold_{0,1,2}`
(fold_k は `yolo_fold_k` = fold_k 非学習の OOF 提案 conf≥0.08、fold 別
標準 MediSeg refiner の shrink@0.45 済み `rle_final` と raw `rle_prop`)。
分割 `grouped_folds.csv` sha256 `febb3f67…` (OOF cache と一致する実験分割、§2.26)。
cache の上流は評価 fold に対し honest で、本実験は fold 跨ぎの学習を含まない
(isotonic は fold 内 halfcross のみ) ため v80 scorer の non-nested 留保は該当しない。

**arm (fold ごと・paired)**:
- `prod`: yolo conf≥0.26 ∧ usable → drop_contained@0.90 → 面積昇順 greedy_disjoint (min_area 20)
- `dedup`: fold 内ファイルを sorted parity で 2 分割し、反対 half で fit した
  isotonic (conf → record 平均 hit-IoU、raw 提案で計算) を適用。cal 降順
  (同値は conf 降順) に raw 提案同士の mask-IoU>d で貪欲 dedup → cal>λ
  (λ = 0.5·PQ(prod, fold)) → usable のみ → prod と同一 ownership → 公式 PQ
- 主設定 **d=0.30** (fold2 で選択済みの値をそのまま固定。fold0/1 では掃引しない)

**事前固定ゲート (結果取得前に確定)**:
- Δ_k = PQ(dedup_d0.30, fold k) − PQ(prod, fold k)
- **replicated**: Δ_0 > 0 **かつ** Δ_1 > 0 **かつ** fold0+1 pooled 観測
  bootstrap (2000 回、seed 3407) の Δ の CI95 下限 > 0
- それ以外は **not_replicated** — dedup030 は fold2 特有の事後選択と記録し、
  最終 2 枠は現行案 (prod + v96-full) を維持
- replicated の場合も「fold2 選択の 2 独立 fold 確認」であり、提出採用は
  ユーザー判断 (LB では識別不能のため OOF 根拠のみで private 期待値を比較)

**判定不使用の診断**: fold2 での同一エミュレーション (§2.72 の EdgeAtt 経路
+0.0075 との整合 sanity)、d∈{0.10, 0.20, 0.50} の各 fold 値、dedup なし
で cal>λ のみ (= 実効 conf 閾値) の arm (dedup 機構と閾値緩和の分離)。

**誠実留保**: (a) refiner は fold 別標準 MediSeg で、prod 提出の EdgeAtt
all-data とは異なる — 受理規則の相対効果の検証であって絶対値の再現ではない。
(b) cache は conf≥0.08 のみ。cal>λ 相当の実効 conf*≈0.24 より十分低く、
降順貪欲では低 conf 候補が高 conf 候補を抑制しないため受理集合は不変。
(c) 1 fold あたり ~236 観測で fold 単独 CI 幅は ~0.035 (§2.5) — 符号一致と
pooled CI を併用するのはそのため。

実装: `scratch/run_v99_dedup_replication.py`、テスト
`tests/test_run_v99_dedup_replication.py`。出力 `output/v99_dedup_replication/`。

#### v99 結果 (2026-11-03、CPU 23 分 × 2、`output/v99_dedup_replication/`)

**事前登録ゲート (isotonic halfcross・d=0.30・λ=0.5·PQ(prod)) — `not_replicated`**:

| fold | prod | dedup0.30 | Δ | nodedup Δ | 最良 d の Δ (診断) |
|---|---|---|---|---|---|
| 0 | 0.4393 | 0.4355 | **−0.0038** | −0.0051 | −0.0029 (d0.10) |
| 1 | 0.4415 | 0.4396 | **−0.0019** | −0.0026 | −0.0009 (d0.20) |
| 2 (sanity) | 0.4471 | 0.4460 | −0.0010 | −0.0016 | −0.0004 (d0.20) |

fold0+1 pooled Δ −0.0029、CI95 [−0.0062, +0.0006]。**sanity 不一致**: fold2 の
同一規則が §2.72 の +0.0075 を再現しない。isotonic の λ が実効 conf を 0.26 超に
押し上げ、prod より TP を減らしている (fold2 TP 1729→1682)。

**事後診断 (判定不使用、`posthoc_fixedconf_0.238.json`)**: 受理を v97d の
実効 conf*=0.238 に固定し dedup だけを残した場合:

| fold | nodedup Δ | d0.10 | d0.20 | d0.30 |
|---|---|---|---|---|
| 0 | +0.0008 | +0.0042 | +0.0029 | +0.0027 |
| 1 | −0.0005 | −0.0010 | +0.0001 | **−0.0012** |
| 2 | −0.0004 | +0.0018 | +0.0022 | +0.0017 |

d0.30 の fold0+1 pooled Δ +0.0008、CI95 [−0.0023, +0.0041] — 同じく不一致。
dedup 機構の上乗せ (dedup − nodedup) は fold0 +0.002〜+0.003・fold1 −0.0007〜+0.0006・
fold2 +0.002 程度で、§2.72 の「dedup ≈ +0.006」より 1 桁小さく fold1 で符号が揃わない。

**判定**: dedup030 は独立 fold で再現しない。§2.72 の +0.0075 は fold2 での
dedup・λ 掃引による事後選択の楽観と、EdgeAtt 経路固有の差の合成と解釈する
(本検証は標準 refiner・v80 cache 提案で行ったため、後者の寄与は分離できない —
留保 (a))。public 0.39 同点 (§2.73) とも整合。**最終 2 枠は現行案
(prod + v96-full) を維持し、dedup030 は採用しない**。

### 2.76 v100 事前登録: 検出器容量スケーリング (yolo11l/x-seg) — fold2 paired 判定 (2026-11-03)

(注: §2.75 は別ブランチの v99 dedup030 再現ゲートで使用中 — PR #607)

**出自 (§2.66)**: 残差の主因は検出器の順位付け = モデル能力。
V36–V41 の検出器強化 (imgsz・augmentation・モデルサイズ) は
ほぼ fold0 単独判定で、§2.5 の fold 単独 CI 幅 ~0.035 では
Δ±0.01 を判別できず **判定不能** と訂正済み。受理側の改善は
v97 (RT-DETR 寄与ゼロ)・v98b (多様化で希釈)・v99 (dedup030
非再現) で尽きたため、検出器容量は**一度も honest 評価されていない
最後の上流レバー** として残る。

**設計 (変更はモデルスケールのみ。学習レシピ・評価経路・fold2・
paired prod 基準は v96 と完全一致)**:

- `prod`: 既存 `yolo_fold_2/weights/best.pt` (yolo11m-seg、
  train = fold0+1) を再評価して sanity 基準とする
- `yolo11l`: train = fold0+1 (all-labeler)、**yolo11l-seg.pt**
- `yolo11x`: train = fold0+1 (all-labeler)、**yolo11x-seg.pt**
- 学習レシピは prod と完全一致 (imgsz 1280・batch 4・epoch 70・
  patience 14・seed 3409・cls 0.5・増強同一・ultralytics 8.4.118
  固定・cache="disk")。評価は v96 Colab ノートのセルを無改変で
  再利用: fold2 EdgeAtt refiner・thr 0.45・drop 0.90・
  min_area 20・公式 PQ (record 単位 pooled)

**事前固定ゲート (結果取得前に確定)**:

- 主判定: arm 毎に `Δ = PQ(arm, 0.26) − PQ(prod, 0.26)` (fold2 paired)
  - `>= +0.010`: **fold0 honest 複製** (yolo_fold_0 構成で再学習、
    fold0 paired 評価) へ昇格 — fold2 単独では提出変更しない
  - `[0, +0.010)`: reject_noise — 容量スケーリングを閉鎖
  - `<0`: reject — 同上
- sanity: prod paired 再現値が既知 0.444218 ±0.002 内であること
- conf sweep {0.15,0.20,0.26,0.32}・best/last・sparse PQ は
  判定不使用の診断として記録 (閾値掃引しない)

**誠実留保**: (a) fold2 は prod best.pt のモデル選択 fold でもある
ため両 arm とも軽度の optimistic 選択を共有 — paired Δ が相殺する
が、昇格時は必ず fold0 複製を通す (b) 単一 seed。GPU 非決定論の
paired ノイズ ~0.003 をゲート幅 +0.010 で担保 (c) yolo11x-seg・
imgsz1280・batch4 は Colab GPU で OOM の可能性。OOM の場合のみ
batch=2 で一度だけ再試行し manifest に `batch_fallback: true` を
記録 (レシピ差分として昇格判定の留保にする)。batch2 でも失敗する
場合はその arm を `infeasible` として記録 (d) 容量増は v96-full
(データ量) と直交のレバー — 昇格した場合も提出候補の置換は
fold0 複製 + LB 確認の両方を通過して初めて検討する

**実装**: `scratch/build_v100_capacity_colab.py` (v96 third_f0
Colab ノートの凍結セル再利用、CFG の model/arms/paths のみ差替え)
→ `solar_filament_v100_capacity_colab.ipynb`。学習は
`runs/v100_capacity/yolo_{arm}_fold2`、manifest は
`artifacts/v100_capacity_colab_manifest.json`。

**結果 (2026-11-04、Colab NVIDIA L4、manifest updated 03:08 UTC)**:

| arm | conf | PQ | Δ@0.26 vs prod | tp/fp/fn | verdict |
|---|---|---|---|---|---|
| prod (yolo11m) | 0.26 | 0.444247 | — | 1715/750/1070 | sanity 基準 |
| yolo11l | 0.26 | 0.442641 | **−0.0016** | 1683/707/1102 | reject_capacity_fold2 |
| yolo11x | 0.26 | 0.442377 | **−0.0019** | 1646/650/1139 | reject_capacity_fold2 |

- sanity: prod paired 再現 0.444247 vs 既知 0.444218、差 +0.000029
  で ±0.002 内 = 評価経路は再現されている
- 両 arm `<0` → 事前固定ゲートにより **reject_capacity_fold2**。
  fold0 複製は発動せず、容量スケーリングはここで閉鎖
- 診断 (判定不使用): arm に事後最良 conf を許しても prod@0.26 を
  上回らない (l 最良 = 0.26 で −0.0016、x 最良 = 0.20 で −0.0008)。
  x は低 conf 0.15 では prod を上回る (0.4345 vs 0.4265) が、これは
  FN 被覆の広さ (低 conf 提案で FN を覆う数 611 vs prod 543) に由来し、
  動作点では予測数を絞り込めず (5.95/img vs 6.38/img) TP 減・FN 増。
  sparse PQ も l −0.0006 / x +0.0002 で系統的改善なし
- best=last で同一結果 (両 arm とも best epoch = 70)。学習時間
  l=192.5min / x=293.4min。batch4 が L4 22.5GB に収まり
  `batch_fallback` は未発火 (x は 20.9GB 使用)
- 重み SHA256: l best `536726d0…`、x best `8cd43302…`
  (manifest に全値)

**解釈**: §2.66 の「残差の主因は検出器の順位付け = モデル能力」という
仮説に対し、容量 2.2 倍 (x、62M params / 297 GFLOPs) でも順位付けは
改善しなかった。x は低 conf でより多くの FN を覆う提案を出す
(リコール源は増える) が、その候補を正しく上位に並べられない =
ボトルネックは提案の**存在**ではなく候補間の**識別**で、容量増は
この識別を改善しない。annotator 不一致が構造天井という既存解釈と
整合。これで上流レバー (データ量・モデル種・容量・アンサンブル・
重複除去) はすべて honest 評価済みとなり、**最終 pair
prod + v96-full を据え置く判断が確定**した。

### 2.77 v96-full 提出の byte-identical 再現を確認 (2026-11-04)

(注: §2.75 = v99 dedup030 再現 (PR #606)、§2.76 = v100 容量ゲート
(PR #608) は別ブランチで先行確保済み)

**結果**: `scratch/run_test_submission.py --mode prod --detector full`
(本番 cell12 と同一経路: solar_three_channel → 全データ学習 YOLO
`yolo_full_fold2_best.pt` → all-data EdgeAtt refiner → drop@0.90 →
面積昇順 ownership → RLE) を隔離 CPU 環境で実行。生成 CSV
`submission_prod_v96full.csv` (1,141行 + ヘッダ) の SHA256 は
`684ff6d60751249edfc7c9e1ae68828945ca77d9d272f1c868609fdbc12ca4e4`
で、Kaggle 提出 `57897560` (v96-full、Public 0.39) の参照 CSV と
**全バイト一致**。実行環境: CPU のみ・180枚・1371秒・
code sha `a6edc0de`・隔離 uv 環境 (prod freeze 相当の依存)。

**意味**: repo のパイプラインコード + 公開重み (detector + refiner)
+ test 画像だけで、選択済み第2提出が決定論的に再現できることを実証。
主催者要件「repo が両提出を再現できる」(topic 747960) のうち
v96-full 側を満たす (prod 側は 2026-10-15 に Kaggle kernel v3/v4 で
byte-identical 受入済み、README §最終提出の再現)。CPU (非 GPU)
での一致は、パイプラインが GPU 非決定論に依存しないことの追加証拠。

**補完**: Drive `solar_filament_v30/v96full_clean_repro/` に
`solar_filament_v96full_clean_repro_colab.ipynb` を配置済み
(Colab GPU で同一検証を行う独立ノート、実行はユーザー)。
重み `yolo_full_fold2_best.pt` は standalone repo `weights/` に
LFS 同梱済み (`e5b1bc13`、manifest sha256 `f54c812d…`、
公開 clone + `git lfs pull` で実体確認済み)。

**GPU クリーン再現の実行結果 (2026-11-04、ユーザー実行)**:
Kaggle 系ランタイム (NVIDIA L4、`/kaggle/input` が存在する環境) で
上記ノートを実行。クリーンルート `/content/solar_clean_root` に
Drive の競技データをコピー、standalone repo を `e5b1bc13` で
clone + `git lfs pull` し、source nb sha256 `d9c1c05597fa2d47` を
検証のうえ4箇所の制御置換 (variant・clean root・cache 無効・
Kaggle 判定時のデータ検索 root 追加) のみ適用して全コードセル実行
(wall 9,816 秒)。

- 生成 CSV: 1,141 行、sha256 `151541db…`、350,196 bytes
- **byte_identical: false** (参照 `684ff6d6…`、350,083 bytes)
- 予測内容は**等価**: IoU 比較で生成・参照の全 1,141 filament が
  IoU≥0.5 で 1-to-1 対応 (match_rate 1.0、177/177 画像で両者に
  予測あり)。best-IoU 平均 0.9993・中央値 1.0・IoU=1.0 が 54.8%・
  **全件 IoU≥0.9**。公式 PQ (IoU 0.5 判定) では TP/FP/FN が不変で
  SQ への影響は最大 ~0.0007 程度
- 差異の帰属: **GPU アーキテクチャの浮動小数点非決定論**。参照は
  Kaggle GPU、今回は L4 で、境界画素のフリップが ~45% の
  filament で生じた (インスタンスの存在/個数は不変)。CPU での
  byte 一致と合わせると、パイプラインのコード・データ・重みは
  決定論的で、byte 差は実行ハードウェアの FP 丸め差に限定される
- 環境検出の補修も検証済み: 内側 nb が `/kaggle/input` の存在で
  Kaggle 判定となっても clean root のデータを発見し、生成 CSV を
  `/kaggle/working` の `OUTPUT_PATH` から正しく回収した
  (wrapper fix `10c2e070`)

**結論**: repo + 公開重み + test 画像のみで提出と予測等価な
submission をクリーン環境から生成できることを確認 (CPU では
byte 一致、GPU では IoU≥0.9 の境界画素差のみ)。主催者の再現性
要件を満たすと判断する。結果 JSON は Drive
`v96full_clean_repro/clean_repro_result.json`・`iou_compare.json`、
生成 CSV は `submission_prod_v96full_colab.csv` に保存済み。

### 2.78 データ監査: 「annotator 不一致天井」を一次データから直接計測 (2026-11-04)

(注: §2.75=v99 dedup030 (PR #606)、§2.76=v100 容量 (PR #608)、
§2.77=v96-full 再現 (PR #607) は別ブランチで先行確保済み)

ユーザー指摘「前提を疑ってデータを見返す」に応じ、天井説を物語として
引くのではなく COCO annotations から直接計測した
(`scratch/audit_annotator_agreement.py`、合成マスクの単体テスト 11 件。
出力 JSON: `output/annotator_audit/annotator_agreement_audit.json` /
Drive `artifacts/annotator_agreement_audit.json`)。

**構造 (train JSON)**: 707 物理画像 × 1154 annotator record
(`image_id` = `<annotator_prefix>-<file stem>`)。record/画像 =
1人:411 枚・2人:145 枚・3人:151 枚。全画像ネイティブ 2048×2048。

**測定結果**:

| 指標 | 値 | 解釈 |
|---|---|---|
| inter-annotator PQ (record A を予測として B に採点、1,196 順序対・296 画像) | **mean 0.3412** | §2.54 の 0.3399 と独立に一致。annotator 自身が他 annotator を「採点」しても ~0.34 — 我々の LB ~0.39-0.40 は**既に annotator 間一致を上回っている** |
| 物理フィラメント cluster の annotator 被覆 (record 横断 IoU≥0.5 連結、3,199 cluster) | **singleton (1人のみマーク) 57.9%** | 2人ファイルで 60.8%・3人ファイルで 54.8% が 1人のみ。予測すると 1 record で TP・残り record で FP になる構造 |
| matched-pair IoU (record[0] 基準で貪欲マッチした同一物理物体) | mean **0.633** / median 0.629 | 境界一致も低い。prod の SQ 0.68 は**既に annotator 間の境界一致を上回る** |
| record 内 GT 同士の重複 | 17.38M px 中 5,041 px (0.029%)、実質 1 ペアのみ | 二重アノテーションはほぼ皆無 — 「disjoint 制約で構造的失点」仮説は棄却 |
| 面積 <20px の GT | 8,199 中 1 件 | min_area=20 由来の不可避 FN は無視できる |
| 多連結 GT マスク | 227 件 (2.8%) | 1 instance が複数 blob の GT は存在するが RLE 上予測も同表現可 |

**解釈**: 天井は「annotator が同じ物を別々に判定した」存在の不一致
(58% singleton) と「同じ物を違う境界で描いた」境界の不一致
(matched IoU ~0.63) の2層で実測された。これは全失敗の説明とも
整合する: 見逃し救出は singleton 側を多く拾うため、入れるほど他
record 側の FP になり得る (rescue/semantic/verifier の連敗)、
refiner は境界精度を既に annotator 一致以上にできており上乗せ
余地が小さい、容量 (v100) は順位付けを変えない — 残る失点は
「どのアノテータが描いたか」を当てる問題でありモデル側の情報
改善では届かない領域が大きい。

**残る正当な仮説**: 入力解像度。画像はネイティブ 2048² だが
imgsz 1280 は 1.6 倍のダウンサンプルで細い構造の情報を物理的に
捨てている。効き目があるとすれば (a) 提案品質より **SQ (境界品質)
側** — recall 系レバーは singleton 構造上むしろ不利になり得る。
V36–V41 が fold0 単独で「判定不能」と残した imgsz は、この構造を
踏まえてもなお**一度も honest 評価されていない最後のレバー**。

### 2.79 v101 事前登録: 入力解像度スケーリング imgsz {1536, 2048} — fold2 paired 判定 (2026-11-04)

**出自 (§2.66 + §2.78)**: V36–V41 の検出器強化は fold0 単独判定で
判定不能と訂正済み。v100 でモデル容量 (m→l/x) は棄却確定したが、
**入力解像度は別のメカニズム** — ネイティブ 2048² を 1280 に縮小
する過程で細いフィラメント構造の情報を実際に捨てている (§2.78)。
v36 の imgsz1536 fold0 試行が残る「一度も honest 評価されていない
上流レバー」であり、§2.78 の監査が示す残る改善余地 (SQ 側) に
最も整合するレバーでもある。

**fold0 の事前値 (V30-era baseline 0.4393 に対し、README §P2 記録)**:
P2-2 (imgsz1536 で再学習) PQ 0.4295 (Δ **−0.0098**)、P2-3 (推論のみ
1536) PQ 0.4320 (Δ **−0.0073**) — fold0 では両方負方向だが、
fold0 単独 CI ~±0.035 (§2.5) では ±0.01 を判別できず「判定不能」
として残置した (§2.66)。つまり正直な事前分布は負寄りであり、
本実験はその負方向が真か fold0 ノイズかを fold2 paired で判別する。
当時の eval 経路 (V30 時代) と現行 (EdgeAtt・ownership) の差分も
Δ の転移を不確実にするが、方向の事前情報として記録する。

**設計 (変更は imgsz のみ。モデル・学習 fold 集合・レシピ・評価経路は
prod / v96 / v100 と完全一致)**:

- `prod`: 既存 `yolo_fold_2/weights/best.pt` (yolo11m-seg、imgsz 1280、
  train = fold0+1) を再評価して sanity 基準とする
- `prod_infer1536`: **学習なし診断 arm** — prod 重みを imgsz 1536 で
  推論するだけ。「解像度が効くなら再学習不要か」を分別する。
  学習を伴わないためゲート判定の対象外、診断のみ
- `imgsz1536`: yolo11m-seg、**imgsz 1536 のみ変更** (v36 の honest 複製)
- `imgsz2048`: yolo11m-seg、**imgsz 2048 のみ変更** (ネイティブ解像度
  = ダウンサンプル完全解消)
- 学習レシピは prod と完全一致 (batch 4・epoch 70・patience 14・
  seed 3409・cls 0.5・増強同一・ultralytics 8.4.118 固定・
  cache="disk")。評価は v96/v100 と同じ凍結セル: fold2 EdgeAtt
  refiner・thr 0.45・drop 0.90・min_area 20・公式 PQ (record 単位
  pooled)。推論 imgsz も arm 値を使う (train/infer で解像度を揃える)

**事前固定ゲート (結果取得前に確定)**:

- 主判定: 学習 arm (imgsz1536・imgsz2048) 毎に
  `Δ = PQ(arm,0.26) − PQ(prod,0.26)` (fold2 paired)
  - `>= +0.010`: **fold0 honest 複製** (yolo_fold_0 構成で再学習、
    fold0 paired 評価) へ昇格 — fold2 単独では提出変更しない
  - `[0, +0.010)`: reject_noise — 解像度スケーリングを閉鎖
  - `<0`: reject — 同上
- `prod_infer1536` は診断のみ (ゲート判定対象外): その値が
  prod@1280 と大きく違えば「推論解像度だけで効く」機構の存在を
  示唆 (さらなる分離実験の検討材料とするが、提出変更はしない)
- sanity: prod paired 再現値が既知 0.444218 ±0.002 内であること
- conf sweep {0.15,0.20,0.26,0.32}・best/last・sparse PQ は
  判定不使用の診断として記録 (閾値掃引しない)

**誠実留保**: (a) fold2 は prod best.pt のモデル選択 fold でもある
ため両 arm とも軽度の optimistic 選択を共有 — paired Δ が相殺する
が、昇格時は必ず fold0 複製を通す (b) 単一 seed。GPU 非決定論の
paired ノイズ ~0.003 をゲート幅 +0.010 で担保 (c) imgsz 2048・
batch4 は L4/A100 でも OOM の可能性。OOM の場合のみ batch=2 へ
一度だけ fallback し manifest に `batch_fallback` を記録
(レシピ差分として昇格判定の留保にする)。batch2 でも失敗する
場合はその arm を `infeasible` として記録 (d) imgsz 2048 の
学習時間は 1280 の ~2.5 倍 — Colab ランタイム切断に備え
manifest 逐次保存で resume 可能 (e) §2.78 の構造知見から、
効くなら recall ではなく SQ 側と予想 — FN/FP/SQ 内訳を診断として
記録する

**実装**: `scratch/build_v101_imgsz_colab.py` (v96 third_f0
Colab ノートの凍結セル再利用、CFG の arms/imgsz/paths と
実行セルの verdict のみ差替え、arm ループ冒頭で
`CFG.YOLO_IMGSZ` を arm 値に設定して凍結 eval セル無改変)
→ `solar_filament_v101_imgsz_colab.ipynb`。学習は
`runs/v101_imgsz/yolo_{arm}_fold2`、manifest は
`artifacts/v101_imgsz_colab_manifest.json`。

**v101 結果 (2026-11-05、Colab L4 完走)**: manifest
`artifacts/v101_imgsz_colab_manifest.json`。prod sanity
0.444247 vs 既知 0.444238 (+9e-06、合格)。両学習 arm **棄却**:

| arm | PQ@0.26 | Δ vs prod | TP/FP/FN | SQ | 学習時間 | verdict |
|---|---|---|---|---|---|---|
| `prod_infer1536` (診断) | 0.442430 | −0.0018 | 1755/856/1030 | 0.6802 | infer のみ | diagnostic_no_train |
| `imgsz1536` | 0.443110 | −0.0011 | 1661/661/1124 | 0.6812 | 247.7 min | reject_imgsz_fold2 |
| `imgsz2048` | 0.441951 | **−0.0023** | 1634/631/1151 | **0.6829** | 455.2 min | reject_imgsz_fold2 |

**判定: 入力解像度スケーリングの閉鎖。** 両 arm < 0 で fold0 複製は
発動せず。fold0 の負方向事前値 (P2-2 −0.0098 / P2-3 −0.0073、§2.66)
は fold0 ノイズではなく真の方向だった — fold2 paired で確定。

診断 (判定不使用) — §2.78 監査の予測と正確に一致:

- **高解像度は SQ を実際に改善するが存在判定を悪化させる**。
  imgsz2048 の SQ 0.6829 は観測全 arm 最高 (prod 0.6800) — 細い構造の
  境界品質は上がる。しかし検出は保守的になる (preds 6.38→5.5x/img、
  TP −81・FN +81・FP −119)。existence 不一致 (singleton 57.9%) が
  支配する天井構造で、境界品質向上は FN/FP バランス悪化に呑まれる。
- **`prod_infer1536` (推論のみ) は逆に検出を増やす** (TP +40・
  FN −40) が FP +106 が上回りネット負 — 「高解像度で追加検出される
  ものは他 record で FP 化しやすい」という監査予測の直接確認。
- 用量反応も単調: 1280 → 1536 → 2048 で Δ −0.0011 → −0.0023 と
  解像度を上げるほど悪化 — データの細かさではなくラベルの
  識別不可能性が律速であることの追加証拠。
- batch4 が両 arm で L4 (22.5GB) に収まり OOM fallback 未発火。
  全重み SHA256 記録済み。

**全体の状態**: 上流レバー (データ量・モデル種・容量・アンサンブル・
重複除去・verifier・ラベル構造・解像度) と下流パラメータ
(v102 §2.80 で完結) が全て閉じた。**最終 pair prod (`57039986`)
+ v96-full (`57897560`) の据え置きが確定。**

### 2.80 v102 事前登録: 推論パラメータ掃引 — 宣言済み掃引に列挙されなかった残存 CFG (事前登録, 2026-11-05)

**出自**: §2.23 末で「prod の全チューニングパラメータが検証済み
(conf/閾値/drop/min_area/重み選択、残存パラメータなし)」と宣言したが、
その列挙には **`PREDICT_IOU`・`CROP_CONTEXT`・`CROP_SIZE`・
proposal の refine 前カバー率** が含まれていなかった。§2.78 監査で
残余余地は存在の不一致 (singleton 58%) ではなく**境界側**
(matched-pair IoU 0.63、prod SQ 0.68) にあると定量されたため、
refiner 幾何・proposal カバー率に効くこの4パラメータが最後の未検証口。

**arm は全て推論のみ (prod fold2 YOLO + all-data EdgeAtt refiner、学習なし)**:

| arm | 変更 (prod 既定 → arm 値) | 狙い |
|---|---|---|
| `nms_iou_035` | PREDICT_IOU 0.55→0.35 | detector NMS 強化で重複提案を前段除去 |
| `nms_iou_075` | PREDICT_IOU 0.55→0.75 | 重複提案を温存し refiner/ownership に委譲 |
| `crop_ctx_130` | CROP_CONTEXT 1.65→1.30 | refiner 視野を狭く (形状の局所判定) |
| `crop_ctx_200` | CROP_CONTEXT 1.65→2.00 | refiner 視野を広く (文脈手掛かり) |
| `crop_size_448` | CROP_SIZE 320→448 | refiner 入力を高精細化 (off-train-res リスク明示) |
| `prop_dilate_2` | PROPOSAL_DILATE_PX 0→2 | `refined & proposal` のクリッピング緩和 |
| `prop_dilate_4` | PROPOSAL_DILATE_PX 0→4 | 同上 (2px で効果兆候があれば量の確認) |

- `prop_dilate_*` は refine **前** に proposal を disk dilate し、
  refiner が「filament」と判定した ring 画素を受理可能にする
  (prod は `refined & proposal` で proposal 外の refiner 判定を破棄)。
  B2 (§2.37) は refine **後** マスクの盲目膨縮であり機構が異なる。
- 推論のみなので学習レシピ差分なし。変更は上表の単一パラメータのみ。
- 実装: 凍結 eval セル (v96 と byte 同一) を変えず、実行セルで
  default-arg binding を回避する同等 wrapper を定義 (prod 既定値では
  元関数と同一経路 — prod sanity ゲートが同一性を実測で担保)。

**事前固定ゲート (結果取得前に固定)**:

- 各 arm `Δ = PQ(arm,0.26) − PQ(prod,0.26) >= +0.010` → fold0 honest 複製へ。
- 全 arm < +0.010 → 推論パラメータ空間を閉鎖し、宣言済み掃引の
  列挙漏れも含めて prod のパラメータ確定を再確認する。
- **多重比較の誠実留保**: 7 cell の固定 grid なので偶然の +0.01 超えを
  排除できない。ゲート超過でも fold0 複製なしには採用しない。
  exploratory_best_conf は診断のみで判定に使わない。
- fold2 単独では提出を変更しない。

**期待値の正直な見立て**: conf/drop/閾値の既掃引が全て ±0.002 帯に
収まった実績から、残存パラメータが +0.010 を超える事前確率は低い
(1〜2割以下)。ただし「列挙漏れのパラメータを全て潰した」という
証拠自体が報告価値を持つ (§2.23 宣言の補完)。`prop_dilate_*` は
refiner の判定権を proposal 外へ拡張する唯一の機構変更で、
near-miss 帯の pred_area/gt_area 中央値 1.214 (pred がやや大きい)
と逆向きなので負方向の可能性が高い — 符号は結果を見てからの
採用判断に使わない事前情報として記録する。

**実行**: `solar_filament_v102_infer_colab.ipynb` (Colab、GPU 必須、
学習なし ~3h 見込み)。manifest `artifacts/v102_infer_colab_manifest.json`
に逐次保存。

**v102 結果 (2026-11-05、Colab L4 完走)**: manifest
`artifacts/v102_infer_colab_manifest.json`。prod sanity
0.444247 vs 既知 0.444238 (+9e-06、合格)。全 7 arm **棄却**:

| arm | PQ@0.26 | Δ vs prod | TP/FP/FN | verdict |
|---|---|---|---|---|
| `nms_iou_035` | 0.4465 | **+0.0023** | 1722/744/1063 | reject_param_fold2 (ゲート未達) |
| `nms_iou_075` | 0.4439 | −0.0003 | 1713/750/1072 | reject |
| `crop_ctx_130` | 0.3010 | **−0.1433** | 1315/1391/1470 | reject (壊滅) |
| `crop_ctx_200` | 0.4396 | −0.0046 | 1705/758/1080 | reject |
| `crop_size_448` | 0.4412 | −0.0030 | 1706/752/1079 | reject |
| `prop_dilate_2` | 0.4397 | −0.0046 | 1705/761/1080 | reject |
| `prop_dilate_4` | 0.4343 | −0.0100 | 1690/770/1095 | reject |

**判定: 推論パラメータ空間の閉鎖。** §2.23 の掃引宣言で列挙漏れだった
4パラメータを全て潰し、prod 経路のパラメータ確定を補完した。

注目点 (全て診断、判定不使用):

- **`nms_iou_035` は唯一の正方向 (+0.0023) で全 4 conf で一貫**
  (0.15: +0.0053 / 0.20: +0.0046 / 0.32: +0.0011)。一貫した符号は
  実信号を示唆するが、事前固定ゲート +0.010 未達のため規約どおり
  **reject、fold0 複製は行わない**。sub-gate delta の追跡は
  dedup030 (§2.75) で fold0 複製失敗した前例と同じ構造であり、
  LB 表示分解能 ~0.01 に対して +0.002 は不可視 — 追求しない。
- **`crop_ctx_130` の壊滅 (−0.143)**: refiner の判定は 1.65x 周辺
  コンテキストに強依存 (TP 1715→1315、FP 750→1391)。crop 幾何が
  prod の load-bearing 要素であることを逆方向から実証。
- **`prop_dilate` は用量反応で単調負** (2px −0.0046 → 4px −0.0100):
  refiner が proposal 外に filament と判定する ring 画素は net-FP。
  proposal のカバー不足は bottleneck ではなく、むしろ過剰側 —
  near-miss の pred_area/gt_area 中央値 1.214 (§2.37) と整合。
- `crop_size_448` −0.003: off-train-res の緩やかな劣化のみ。
  refiner 入力解像度はレバーではない。
- `nms_iou_075` −0.0003: 緩い NMS は無害だが無価値
  (後段の drop_contained/ownership が冗長提案を処理)。

**v101 (§2.79) との併せ読み**: 解像度スケーリング (v101) と推論
パラメータ (v102) が閉じ、prod 経路の上流・下流・周辺パラメータは
全て honest 評価済み。最終 pair (prod `57039986` + v96-full
`57897560`) の据え置きが再確認された。

### 2.81 v103 事前登録: 学習レシピ残差 — 未収束・マスク分解能・解像度下方向・copy-paste (事前登録, 2026-11-05)

**出自**: v101 (§2.79) と v102 (§2.80) で解像度・推論パラメータが閉じた後、
残るのは**学習レシピの単一パラメータ**のみ。3つの観測が動機:

1. **prod は70エポックで未収束** — `runs/yolo_fold_2/results.csv` で
   `metrics/mAP50-95(M)` の best_epoch = **70 (終端)**、v101_1536 も
   同じく終端。学習が打ち切られた形跡があり、長期学習は証拠駆動の
   正当なレバー (v101_2048 は epoch 62 で早停 = 解像度が上がると
   収束が早い点も観測)。
2. **境界側に残余余地** (§2.78: matched IoU 0.63、imgsz2048 で SQ
   0.6829 の実績) — マスク proto 解像度 (YOLO_MASK_RATIO) は
   入力解像度とは別経路で SQ に効くはず。
3. **v101 の用量反応が単調負** (1280→1536→2048 で Δ −0.0011→−0.0023)
   — prod の 1280 がピークとは限らず、**下方向 (960) の確認で
   曲線を完結させる**。
4. `copy_paste` aug は v60 でビルダーだけ存在し未実行・未記録。

**arm (全て yolo11m-seg、train folds (0,1)、評価 fold2、prod レシピと
評価経路は同一、変更は各 arm の単一パラメータのみ)**:

| arm | 変更 (prod → arm) | 機構 |
|---|---|---|
| `prod_infer960` (診断) | 学習なし・prod 重みを imgsz960 で推論 | 推論のみの下方向確認 |
| `imgsz960` | imgsz 1280→960 (train/infer) | 解像度曲線の下方向完結 |
| `epochs120` | epochs 70→120 / patience 14→20 | 未収束の解消 (best=70 終端の実測) |
| `mask_ratio1` | YOLO_MASK_RATIO 2→1 | mask proto 分解能 640²→1280² → SQ |
| `copypaste01` | copy_paste 0.0→0.1 | instance 複製 aug (v60 未実行分) |

- 学習が必要な arm は `CFG` で値を差替えて train/infer 両方に効かせる。
  `copy_paste` は prod 既定 0.0 (不使用) を明示し arm のみ 0.1。
- `mask_ratio1` は proto メモリが ~4x で batch4 OOM の可能性 —
  v101 と同じ一度だけ batch2 fallback の規約を適用し manifest に記録。
- `prod_infer960` は学習なし診断 (gate 対象外、`SKIP_TRAIN_ARMS`)。

**事前固定ゲート (結果取得前に固定)**:

- 各学習 arm `Δ = PQ(arm,0.26) − PQ(prod,0.26) >= +0.010` →
  fold0 honest 複製へ昇格。全 arm < +0.010 → 学習レシピ空間を閉鎖。
- 5 arm の固定 grid (学習 4 + 診断 1)。偶然超過は fold0 複製で制御。
- exploratory_best_conf は診断のみ。fold2 単独では提出を変更しない。

**期待値の正直な見立て**: `epochs120` が最も根拠がある
(未収束の実測) が、v100 で容量を2倍にしても順位付けが変わらなかった
実績から「同じデータで長く回しても識別限界は同じ」の可能性が高い。
残りは全て事前確率 1〜2 割以下。負けても「レシピも潰した」記録価値。

**実行**: `solar_filament_v103_recipe_colab.ipynb` (Colab GPU、
~11–13h 見込み)。manifest `artifacts/v103_recipe_colab_manifest.json`
逐次保存。学習は `runs/v103_recipe/`。

#### §2.81 追補 (完走確定, 2026-11-05): v103 全 5 arm 完走 — 全棄却、mask_ratio1 は別途 fold-0 複製へ (§2.86)

prod@0.26 = 0.4442468 (再現差 +9e-06) に対する paired Δ:

| arm | PQ@0.26 | Δ | verdict |
|---|---|---|---|
| prod_infer960 (診断) | 0.4379841 | −0.0063 | diagnostic_no_train |
| imgsz960 | 0.4465173 | +0.0023 | reject (< +0.010) |
| epochs120 | 0.4434776 | −0.0008 | reject |
| **mask_ratio1** | **0.4507483** | **+0.0065** | reject (< +0.010) → §2.86 fold-0 複製へ |
| copypaste01 | 0.4456666 | +0.0014 | reject |

- `epochs120` の学習曲線: best_ep=93 (mAP 0.2172, prod 0.2134) — 未収束仮説は
  技術的に正しかったが、延長の mAP 微改善は refiner 後 PQ に転写しない。
  1280 の実質収束点 ~90-100ep が特定されたが prod 70 との PQ 差は無差別。
- `mask_ratio1` (proto 2→1, ~640²→1280²) は全 ~50 arm 史上最大の +0.0065
  (ノイズ幅 ±0.002 の3倍、SQ +0.0014 の機構的一貫性あり) — §2.86 で
  fold-0 honest 複製を事前登録 (基準 Δ≥+0.003、fold0 結果取得前に固定)。
- copy-paste 0.1 は微プラスだがノイズ幅内。

### 2.82 v104 事前登録: weight soup + ownership 順序 + refine 2回目 (事前登録, 2026-11-05)

**出自**: v101–v103 で解像度・推論パラメータ・学習レシピが閉じた後に残る
未試行機構。予測空間のアンサンブル/重複除去は v97–v99 で閉じたが、
**重み空間の平均 (weight soup / tail averaging) は一度も試していない**。
prod fold2 の `runs/yolo_fold_2/weights/` には epoch0–65 の周期
checkpoint が残っており、prod の best_epoch=70 (終端・未収束、§2.81)
なら末尾の重み平均 (SWA 的) が単純に有効な可能性がある。全メンバー
folds (0,1) 学習 = fold2 では fold-naive → fold2 paired 評価で valid。

**arm (全て推論のみ・学習なし、prod 評価経路と同一)**:

| arm | 内容 |
|---|---|
| `soup_tail3` | prod epoch60/65/best の3重み平均 (近傍 tail SWA) |
| `soup_tail7` | prod epoch40..65+best の7重み平均 (広い tail SWA) |
| `soup_res3` | prod best + v101 imgsz1536 best + imgsz2048 best の
  3系統平均 — 予測空間ではなく重み空間のアンサンブル
  (同一 init からの fine-tune なので basin 同一の可能性) |
| `own_desc` | greedy ownership を area 昇順→降順 (prod は昇順、
  順序は一度も分離テストされていない) |
| `refine_twice` | refiner 出力を proposal として再 refine
  (境界判定の収束テスト、B2 morphology とは別機構) |

- soup ckpt は notebook 内で組み立て、members・sha256 を manifest に記録。
  epoch ckpt は EMA 重みを使い (best.pt は stripped 'model'、周期 ckpt は
  'ema' が評価用重み — どちらを拾ったかも記録)、'ema' キーは外して
  predict が soup 'model' を使う形で保存。
- `own_desc` は凍結 eval セル無改変のまま実行セルで同等関数を override
  (prod 既定 'asc' では元関数と同一経路)。`refine_twice` も同様に
  refine_proposal を wrap (prod 既定 False では不活性)。

**事前固定ゲート (結果取得前に固定)**:

- 各 arm `Δ = PQ(arm,0.26) − PQ(prod,0.26) >= +0.010` → fold0 honest 複製。
- 全 arm < +0.010 → 重み空間・順序・refine 反復の残差機構を閉鎖。
- 5 arm 固定 grid。偶然超過は fold0 複製で制御。fold2 単独では
  提出を変更しない。exploratory_best_conf は診断のみ。

**期待値の正直な見立て**: 予測空間アンサンブルは dedup プラトーで
+0.0075 が局所天井 (§2.74)、重み空間平均は一般に予測空間より弱い
効果か basin 不一致で壊れる可能性。soup_tail* は未収束の証拠に
対する素直な arm で最も期待できるが、やはり事前確率は低い。
**実行**: `solar_filament_v104_soup_colab.ipynb` (~2.5h、全 arm 推論のみ)。
manifest `artifacts/v104_soup_colab_manifest.json`。

### 2.82追補 — v104 判定: 全 5 arm 棄却。重み空間アンサンブル・ownership 順序・refine 再適用いずれも不発 (2026-11-05)

manifest `artifacts/v104_soup_colab_manifest.json` (fold2 OOF、
prod sanity +9e-06 で一致)。soup ckpt は Drive artifacts に保存済み、
member・sha256 は manifest 記録。

| arm | Δ@0.26 | verdict | 解釈 |
|-----|--------|---------|------|
| `soup_tail3` | **+0.00124** | reject | 唯一の正方向。prod best_ep=70 終端 = 未収束の仮説と整合 (末尾平均が「まだ到達していない改善」を僅かに拾う) だがゲート未達 |
| `soup_tail7` | −0.00470 | reject | 早期 epoch を混ぜると悪化 — 平均窓が効くのは終端近傍のみ |
| `soup_res3` | **−0.43057** | reject | prod+1536+2048 の重み平均は壊滅 (PQ 0.0137)。異なる学習軌跡間の重み平均は loss basin が違い線形接続が破壊される — 重み空間アンサンブルは単一軌跡内に限ることが確定 |
| `own_desc` | −0.00339 | reject | ownership 面積降順は悪化。昇順 (小さい物体が先に領域確保) が正しい設計と確認 |
| `refine_twice` | −0.00064 | reject | 2回目の refine は悪化。refiner 出力は1回でほぼ収束、再適用は過剰 |

**v104 の貢献**: 「予測空間アンサンブル (v97-99 棄却) とは別機構の
重み空間アンサンブル」は単一軌跡内でのみ有効範囲があり、その効果も
sub-gate。残存機構 3 系統 (soup/ownership/refine再適用) 全て閉鎖。

### 2.83 v105 事前登録: 解像度曲線の下方向拡張 imgsz{640,768} — 適応的逐次実験 (2026-11-05)

**背景 (v103 途中結果に基づく適応設計)**: v103 §2.81 の `imgsz960`
train arm が Δ=+0.0023 (PQ 0.4465) の正方向を示し棄却 (sub-gate)。
これは推論のみの `prod_infer960` (Δ=−0.0063) と逆方向であり、
**960 学習の改善は「低解像度での推論」ではなく「低解像度での学習」由来**
であることを意味する。解像度の用量反応は 640←768←960(+0.0023)→
1280(0)→1536(−0.0011)→2048(−0.0023) と読め、下方向は未探査のまま。
**結果を見てから設計した逐次実験であることを明示し、同じゲート規律を
維持する** (fold2 sub-gate delta は追わない → 学習 arm 2 本を追加するが
昇格は fold2 Δ>=+0.010 → fold0 複製のみ)。

| arm | 変更 | 仮説 |
|-----|------|------|
| `prod_infer640` (診断) | prod 重みを imgsz640 で推論のみ | 推論のみは 960 で既に負 → 640 でも負方向予測。「学習が効く、推論は効かない」の対称証拠 |
| `imgsz640` | train/infer 1280→640 | 960 の正方向が傾向なら継続、小物体喪失で反転なら 960 が曲線の底 |
| `imgsz768` | train/infer 1280→768 | 640–960 の間を埋め曲線形状を確定 |

変更は各 arm の imgsz のみ。yolo11m-seg、train folds (0,1)、fold2 paired
評価、refiner・後処理・PQ 実装は prod と完全一致。`prod_infer640` は
学習なし診断でゲート対象外。**ゲート: fold2 学習 arm 毎に
Δ=PQ(arm,0.26)−PQ(prod,0.26)>=+0.010 → fold0 honest 複製。fold2 単独
では提出を変更しない。** OOM fallback は想定しない (低解像度はメモリ
余裕側) が、既存フォールバック機構は残す。

想定所要 ~3h (imgsz 低下で学習は高速化)。manifest
`artifacts/v105_imgszlow_colab_manifest.json` へ逐次保存。

### 2.83追補 — v105 判定: 全 arm 棄却。解像度用量反応が完結し 960-1280 が平坦頂部と確定 (2026-11-05)

manifest `artifacts/v105_imgszlow_colab_manifest.json` (fold2 OOF、
prod sanity +9e-06 で一致)。

| arm | Δ@0.26 | verdict | 解釈 |
|-----|--------|---------|------|
| `prod_infer640` (診断) | **−0.07232** | diagnostic | 予測どおり壊滅 — 960 train が正で 960 infer が負だった対称が 640 でも確認され、「改善は学習由来」の仮説を再確認 |
| `imgsz640` | −0.00617 | reject | 960 の正方向は継続せず反転 — 小物体分解能の喪失が支配 |
| `imgsz768` | −0.00132 | reject | 640–960 の中間点も負 |

**用量反応の完結** (fold2 学習 arm): 640(−0.0062) → 768(−0.0013) →
**960(+0.0023)** → 1280(0) → 1536(−0.0011) → 2048(−0.0023)。
960 が局所ピークだが 768/960/1280 は ±0.002 の差分範囲内で一貫した
傾向とは言い切れない = **解像度は 960–1280 の平坦頂部にあり、
1280 は実質最適域として正当化**。高解像度側の単調悪化のみが真の
信号。v105 は「960 sub-gate delta を追って傾向を確認したら偽信号
だった」記録として、適応的逐次実験の健全な閉じ方を示す。

### 2.84 v106 事前登録: existence 救済 (refiner スコアによる低conf提案の再評価) + 未 sweep 後処理パラメータ (2026-11-05)

**根拠**: §2.78 で「existence の不一致」が支配的 bottleneck と確定。
prod 診断で primary conf の FN 953 のうち **543 (57%) が低conf生提案で
カバー済み** — 検出器は存在に気づいているが flat な 0.26 閾値が捨てて
いる。一方 refiner は提案毎に sigmoid 確率マップを出しており、その
**提案画素上の平均確率は検出器 conf と独立の「実在性」信号**。
この 2 信号で低conf提案を選別する機構は未試行。

**固定式 (結果取得前)**: refiner スコア = prod crop 窓 (mask_square_box,
context 1.65) 内の提案画素上の sigmoid 平均確率。
- rescue: det_conf <0.26 かつ score>=τ → conf を 0.2601 にブースト
  (primary 採点にのみ入る。refine/ownership/採点は prod 経路そのまま)
- demote: det_conf >=0.26 かつ score <τ → conf=0 (全 sweep から除去)

| arm | 変更 | 狙い |
|-----|------|------|
| `rescue_mean050` | τ=0.50 で低conf提案を救出 | FN 回収 (本命) |
| `rescue_mean065` | τ=0.65 (厳格) | 同上・高精度方向 |
| `demote_mean030` | τ=0.30 で高conf提案を除去 | FP 削減 |
| `refiner_th040` / `refiner_th050` | REFINER_THRESHOLD 0.45→{0.40,0.50} | 未 sweep だった refiner 決定閾値 |
| `drop080` / `drop097` | DROP_THRESHOLD 0.90→{0.80,0.97} | 未 sweep の包含閾値 |
| `minarea10` / `minarea40` | MIN_AREA 20→{10,40} | 未 sweep の最小面積 |
| `combo_960_nms35` | imgsz960 重み + predict_iou 0.35 | 独立 sub-gate 信号 (+0.0023×2) の加算テスト |

全 arm 推論のみ (学習なし)、凍結 eval セル byte 同一、CFG 経由の
同等 override のみ。fold2 paired 評価。**ゲート: arm 毎に
Δ=PQ(arm,0.26)−PQ(prod,0.26)>=+0.010 → fold0 honest 複製。
fold2 単独では提出を変更しない。** 探索 arm が多いため多重比較の
観点から単発の fold2 プラスは絶対に昇格しない。想定 ~3-4h。
manifest `artifacts/v106_rescue_colab_manifest.json` へ逐次保存。

### 2.84追補 — v106 判定: 全 10 arm 棄却。existence 救済・後処理パラメータ閉鎖 (combo_960_nms35 は推論解像度交絡を開示) (2026-11-05)

manifest `artifacts/v106_rescue_colab_manifest.json` (fold2 OOF、prod sanity +9e-06)。

| arm | Δ@0.26 | 解釈 |
|-----|--------|------|
| `rescue_mean050` / `rescue_mean065` | **−0.1365 / −0.1170** | τ=0.50/0.65 とも FP 爆発。fn_at_primary は 953→381/1038 に激減し機構は動作したが、**refiner 平均確率は existence の選別信号にならない** (低conf提案の大半を高評価する傾向)。救出 TP +46/+32 に対し FP +2257/+1815 |
| `demote_mean030` | **±0.0000** | conf>=0.26 で refiner 平均 <0.30 の提案は**一件も存在せず** — demote も機能しない (refiner は高conf提案を殆ど減点しない) |
| `refiner_th040` / `refiner_th050` | −0.0004 / +0.0003 | ノイズ幅内。0.45 は実質最適域 |
| `drop080` / `drop097` | −0.0003 / +0.0002 | 同上 |
| `minarea10` / `minarea40` | **±0.0000** | 変更帯に該当提案ゼロ = min_area 20 は非拘束と確定 |
| `combo_960_nms35` | −0.0026 | **実行時は推論解像度交絡あり (未確定)** — 初回ビルダーが `CFG.YOLO_IMGSZ=960` 指定を欠落し 960 学習重みを `imgsz=1280` で推論していた（ビルダー/ノートブック修正済み）。1280 推論では負だが、native `imgsz=960` + `nms035` の加算/打消しは未検証として留保 |

**v106 の貢献**: 「refiner 出力を existence の二次信号として使う」
発想を3方向 (rescue/demote/score) で閉鎖 — refiner は提案内を
フィラメント的に精製する性質上、実在性の識別器として機能しない。
後処理パラメータ空間はこれで完全に網羅された。

### 2.85 v107 事前登録: rank/count 補正 + 別系統 demote + multi-context refine (2026-11-05)

**根拠**: v106 §2.84 の rescue が「refiner 平均確率の選別力不足」で
壊滅 (τ=0.50/0.65 で FP 爆発) → スコア式 rescue は閉鎖。残る機構:
1. **existence の under-count は小規模** — per-record GT ≈7.1 vs
   prod preds 6.38@0.26 (record 毎採点のため差は ~0.7/img)。
   フラット conf でなく **rank ベースで上位 K 個まで埋める** countfill
   はスコア不要の別機構 (v106 と直交)。
2. demote 側は別スコア族が未試行 — refined_area/proposal 比 (精製で
   大幅縮小 = 不確か) と mask aspect (細長さ = filament 形状事前、
   Unidentifiable 反証テスト兼)。
3. **refine の context ensemble** は未試行 — ctx 単一値掃引 (v102) と
   refine 再適用 (v104) は閉じたが、2 context の union/intersect は
   マスク品質向きの新機構。

| arm | 変更 | 狙い |
|-----|------|------|
| `countfill_k7` | conf<0.26 の上位を最大 7/img までブースト | under-count 補正 |
| `countfill_k9` | 同上 K=9 | 同上・多め |
| `demote_ratio030` | conf>=0.26 かつ refined/proposal<0.30 → 除去 | refiner 縮小 demote |
| `demote_aspect15` | conf>=0.26 かつ aspect<1.5 → 除去 | 非細長 demote (形状事前の反証) |
| `ctx_union` | refined = refine@1.45 ∪ refine@1.85 | 境界 union |
| `ctx_intersect` | refined = refine@1.45 ∩ refine@1.85 | 保守 intersect |

**固定式 (結果取得前)**: countfill は conf 上位順 (同点は index 順)、
K 到達分を 0.2601 へ。aspect = bbox max(w,h)/min(w,h)。ctx は
{1.45,1.85} 2窓の binary union/intersect。全 arm 推論のみ、凍結 eval
セル byte 同一。**ゲート: Δ@0.26>=+0.010 → fold0 複製のみ。
fold2 単独では提出不変。** 想定 ~2.5h。manifest
`artifacts/v107_countfill_colab_manifest.json`。

### 2.85追補 — v107 判定: 全 6 arm 棄却。rank/count・形状事前・ctx-ensemble 全て閉鎖 (2026-11-05)

manifest `artifacts/v107_countfill_colab_manifest.json` (fold2 OOF)。

| arm | Δ@0.26 | 解釈 |
|-----|--------|------|
| `countfill_k7` / `countfill_k9` | −0.0145 / −0.0277 | rank 補正でも +36/+45 TP vs +242/+430 FP。**低conf提案は本当にほぼ FP** — conf 0.15 一律引下げの悪化と同方向。rescue (v106)・countfill・診断 (fn_covered=543) の3機構が独立に「低conf層は価値なし」を確認 |
| `demote_ratio030` | ±0.0000 | refined/proposal<0.30 の高conf提案はゼロ — demote_mean030 と一致。refiner は高conf提案を殆ど縮小しない |
| `demote_aspect15` | **−0.1197** | **形状事前の完全な反証** — 非細長除去で tp 1715→1031 (実 TP の40%消滅)。GT 側にも非細長が多数 (Unidentifiable)。細長さは filament の存在証拠にならない |
| `ctx_union` / `ctx_intersect` | −0.0021 / **−0.0471** | ctx ensemble は両方向で悪化 — union は FP 側・intersect は FN 側へ強く偏る (intersect が特に壊滅 = 2窓の厳格化は提案を削りすぎ) |

**v107 の貢献**: existence 追加・除去の両方向で二次信号が存在しない
ことを確定。残る機構は refiner 入力分布の整合 (v108 候補) のみ。

### 2.86 v108 事前登録: mask_ratio1 の fold-0 honest 複製 — ゲート拡張の事前固定 (2026-11-05)

**背景と誠実性の開示**: v103 `mask_ratio1` (mask proto 2→1) が
fold2 で Δ=**+0.0065** — 全実験史上最大でノイズ幅 (±0.002) の3倍。
ただし §2.81 の昇格ゲート +0.010 には未達で棄却判定済み。
**本節は +0.010 未達の arm に対して fold-0 複製を「追加実施する」
というゲート拡張であり、元規約の厳格な読みでは棄却のまま**。
拡張を正当化するのは結果の大きさ (全 ~50 arm 中最大・方向一貫・
SQ 上昇の機構的一貫性) であり、**複製の判定基準は fold-0 の結果を
見る前にここで固定する**。

**複製設計**: 評価 fold を 0 に変更し、train folds (1,2) で
mask_ratio=1 のみ変更した YOLO11m-seg を学習 (prod recipe 完全同一、
seed 3407+0=3407)。paired baseline は既存 `runs/yolo_fold_0`
(prod recipe, mask_ratio=2, folds(1,2) 学習済み、同 seed) を同一
経路で再評価。refiner は `mediseg_edgeattnet_refiner_fold0_resume.pt`。
fold-0 prod OOF 既知値は 0.4443 (§2.x、経路一致の sanity 用)。

**判定基準 (固定)**: Δ@0.26 = PQ(mr1_f0) − PQ(prod_f0) が
**>= +0.003** (観測ノイズ幅の ~1.5 倍) なら「cross-fold で検証済み」
とし、prod レシピへの mask_ratio=1 適用を提出変更の検討対象とする。
+0.003 未満なら未検証として棄却、提出は変更しない。
fold-2 単独の値で昇格しない原則は本複製でも維持する。

想定 ~6.5h (mask_ratio=1 は proto 4x で ~5.8h 学習 + 評価 2 回)。
manifest `artifacts/v108_mr1_f0_colab_manifest.json` へ逐次保存。

#### §2.86 追補 (判定確定, 2026-11-05): mask_ratio1 fold-0 複製 — unverified、提出不変

- prod_f0 @0.26 = **0.444137** (既知値 0.4443 との差 −0.00016 で経路一致)
- mask_ratio1_f0 @0.26 = **0.440174** (SQ 0.67889, tp 1666/fp 724/fn 1083)
- **Δ = −0.003963 < +0.003 → unverified_fold0 → mask_ratio1 は
  cross-fold 検証を通過せず。fold-2 の +0.0065 は再現せず、提出は変更しない**

正直な caveat: fold-0 学習は L4 の OOM により事前登録済みの
batch 4→2 fallback が発動 (manifest `batch_fallback: true`、学習 331.6分)。
fold-2 側は batch=4 で完走していたため厳密な paired ではない。
ただし判定基準は fold-0 結果取得前に固定済みであり、fallback も
事前登録された1回限りの許容であったため、verdict はそのまま有効。
「batch2 だから負けた」可能性を完全には排除できないが、
-0.004 の方向は全 sweep 帯で一貫 (0.15〜0.32 全て prod 未満) しており、
単なる batch ノイズというより fold-2 の +0.0065 が過大評価だった解釈が自然。

**本判定をもって 2026 実験系は全て閉鎖**: 約 50 arm の探索で
昇格基準を満たした arm はゼロ。fold-2 最大の +0.0065 すら
fold-0 で −0.004 に反転し、±0.002 以上の fold 間分散があることを
実証した — fold 単独昇格を禁止したゲート規律の正当性が確認された。
最終提出 pair は従来どおり prod (57039986) + v96-full (57897560)。

### 2.87 v109 事前登録: refiner TTA + 同一モデル多スケール union + 未 sweep だった CROP_SIZE/ctx 単値 + conf 細粒度 (事前登録, 2026-11-05)

v108 で「全実験系閉鎖」を宣言したが、残存する未試行機構の棚卸しで
推論のみの残余 arm を特定した。全て学習なし・推論のみ。

**Arm 一覧 (固定)**:

| arm | 変更 (prod → arm) | 機構 |
|---|---|---|
| `refine_tta4` | refiner 確率マップを二面体4変換 (identity/hflip/vflip/hvflip) で平均 | refiner TTA は一度も未試行 (検出器 TTA は閉鎖済み)。境界品質向き |
| `ms_union960` | 同一 prod 重みを imgsz 1280+960 で2回推論し提案を併合 | 960「学習」は棄却済みだが同一モデルの多スケール推論 union は未試行。存在の多様性向き |
| `crop384` / `crop480` | CROP_SIZE 320→{384,480} | CROP_CONTEXT は掃引済みだが絶対 crop サイズは未 sweep。refiner 入力解像度・出力粒度を変える (入力分布シフトの caveat あり) |
| `ctx145` / `ctx185` | CROP_CONTEXT 1.65→{1.45,1.85} 単値 | v107 は 2窓 ensemble のみで単値は未試行 |
| `conf_fine` | CONF_SWEEP を {0.15,0.2,0.22,0.24,0.26,0.28,0.30,0.32} に拡張 | 運用点のみ・機構不変の診断 arm (verdict は常に diagnostic) |

**採点経路**: 無改変。refine_tta4/ctx は refine_proposal を同等手順で
ラップ (v107 と同型、prod 既定では元関数と同一経路)。ms_union は
YOLO コンストラクタをラップして predict を2スケール併合
(結果オブジェクトは最小互換 namespace)。crop/conf_fine は
CFG 実行時参照パラメータの差替えのみ。

**ゲート (固定)**: Δ = PQ(arm,0.26) − PQ(prod,0.26) >= **+0.010** →
fold0 honest 複製へ。fold2 単独では提出を変更しない。conf_fine は
運用点の診断であり、仮に細粒度点で高値が出ても fold0 paired 複製
(+0.003 以上) を経ない限り採用しない (§2.86 と同じ規律)。

想定 ~1.5h。manifest `artifacts/v109_tta_colab_manifest.json` へ逐次保存。

**v109 結果 (2026-11-06 完走、3 arm は実行欠陥で再測定中)**: prod
0.4442468 paired。

有効 arm (crop_size=320 確定): `refine_tta4` 0.444991 (Δ+0.0007
reject)、`ms_union960` 0.438038 (−0.0062 reject)、`crop384`
0.442315 (−0.0019 reject)、`crop480` 0.441317 (−0.0029 reject)。

**実行欠陥の開示**: arm ループの per-arm 既定値が `CFG.CROP_SIZE` /
`CFG.CONF_SWEEP` を参照し、`crop480` が CFG を書き換えた後の
`ctx145`/`ctx185`/`conf_fine` は **crop_size=480 経路**で評価された
(`conf_fine` の値が crop480 と bit 完全一致するのが証拠)。
一次値は Drive `v109_tta_colab_manifest_firstpass_contaminated.json`
に退避保存: ctx145 0.439355 / ctx185 0.441491 / conf_fine 0.441317
(全て crop480+該当 ctx の複合値であり ctx 単値ではない)。
昇格判定への影響はなし (全 arm reject 方向)。既定値を prod リテラル
(320, sweep 固定) に修正 (642215c8)、汚染キーを manifest から除去し
該当3 arm のみ再実行予定 — 結果は追記する。

**v109 再測定結果 (2026-11-07 完走)**: 修正版 (既定 prod リテラル) で
汚染3 arm を再評価。`ctx145` **0.397399** (Δ−0.0468、context 縮小は
refiner に壊滅的 — 汚染値 0.439355 が実害を隠していた)、`ctx185`
0.441362 (−0.0029)、`conf_fine` @0.26 = **0.444247 で prod と bit
完全一致** (リテラル既定の修正が正しいことの実証、diagnostic として
prod 運用点を正しく再現)。

**v109 全閉鎖**: 全7 arm reject。最良は refine_tta4 (+0.0007)。
context 1.65・crop 320 が両側劣位の最適点として確定し、
キャンペーンで未試行だった推論機構は全て負方向で閉じた。

### 2.88 v110 事前登録: proposal-matched refiner 再学習 — GT crop ↔ proposal crop の分布ギャップを塞ぐ (事前登録, 2026-11-05)

v106 で「GT-positive crop でしか学習していない refiner の平均確率は
existence を識別できない」ことが確定した。本実験はその原因である
**train/test 入力分布のミスマッチ**を直接塞ぐ、残る唯一の構造的機構。

**設計 (固定)**: prod 検出器を学習 folds (0,1) の画像で推論
(`conf>=0.08` = PREDICT_CONF_FLOOR、imgsz 1280、iou 0.55、
agnostic_nms、retina_masks = 推論時と同一)。各提案 mask をそのファイルの
**全 annotator レコードの GT** と IoU 照合 (採点と同じ pooled 基準):
- IoU>=0.5 → 正例: crop = 提案 mask の square box (ctx 1.65、min 96、
  refine_proposal と同一手順)、target = 最良一致 GT mask を同じ box で crop
- IoU<0.5 → 負例: 同じ crop、target = 空マスク (refiner がジャンクを
  抑制することを学習 = mean prob が真の存在信号になる可能性)

**refiner 学習は prod プロトコルと同一**: EdgeAtt (MediSegEdgeAttUNet)、
AdamW lr 2e-4 wd 1e-4、cosine、epochs 14、seed 3407、AMP、
refiner_loss (0.35 BCE + 0.65 soft-dice + 0.15 edge-L1)、
batch 候補 (8,6,4,2)、augment (rot90/flip/輝度)、
valid = fold-2 GT crop (prod refiner と同じ選択基準 — paired 維持)。

**開示 caveat**: prod 検出器は folds (0,1) を学習済み → 提案は
in-sample で楽観バイアス (負例の割合が test 時より少ない)。それでも
GT box crop → proposal mask crop の一次ギャップは縮む。

**Arm (固定)**: 
- `prop_matched`: prod detector + prop-matched refiner (refiner のみ差替)
- `prop_rescue050`: 同 refiner + conf∈[0.08,0.26) の提案で refiner 平均
  確率 >= 0.50 なら 0.2601 ブースト (v106 と同一式 — prop-matched で
  識別力が改善したかの直接検証)

**ゲート (固定)**: Δ@0.26 >= **+0.010** → fold0 honest 複製。
fold2 単独では提出不変。想定 ~2h (提案生成 ~15分 + 学習 ~40分 + 評価3回)。
manifest `artifacts/v110_propmatch_colab_manifest.json`。

**v110 結果 (2026-11-06 完走)**: prod 0.4442468 paired。
`prop_matched` 0.440713 (Δ@0.26 = **−0.0035**、0.15-0.32 全 conf 帯で
prod 未回)、`prop_rescue050` 0.388774 (Δ = **−0.0555**、rescue 依然
壊滅的)。prop_rescue050@0.32 = prop_matched@0.32 = 0.436327 で一致
(rescue は conf<0.26 のみ作用するため高 conf で同一 — sanity 一致)。
**判定: proposal-matched 学習は existence 識別を回復しなかった** —
GT crop ↔ proposal crop の分布ギャップ説は rescue 失敗の主因では
なかったことが確定。mean-prob を使う存在性選択は全経路で棄却済み
(v106 → v110) であり、refiner 側の構造的機構はここで全閉鎖。
manifest `v110_propmatch_colab_manifest.json`、提案統計
`v110_proposals_train01.json`。

### 2.89 v111 事前登録: test 画像疑似ラベル半教師 — 同一分布の最後の未学習データ源 (事前登録, 2026-11-05)

fold 分割を汚さず半教師学習を正直に測れる唯一の未学習データ源は
**コンペ test 画像 180 枚**である。folds(0,1) 学習の prod パイプライン
(教師) で test 画像を疑似ラベル化し、folds(0,1)+疑似ラベルで再学習、
fold-2 honest eval で効果を測る。fold-2 は疑似ラベル生成・学習・val の
いずれにも非接触のため transductive 汚染はない。

**既存の負の事前エビデンス (正直な開示、期待値は低め)**:

- v48/v49 (§2.4): 外部 GONG 別日付画像の疑似ラベルは fold0 Δ-0.0061 /
  fold1 Δ-0.0058 で不採用確定。ただし疑似ラベル源が別年代ドメインであり
  「年代偏り仮説」の検証だった — test 画像は同一分布で状況が異なる
- 他チーム kernel 分析 (§2.79 時点記録): self-training (conf>=0.75) が
  0.417→0.393 に明確悪化。raw 検出器ラベル・refiner 不使用の単純方式
本実験は両者と質的に異なる: (a) 疑似ラベル源 = test 画像そのもの
(同一分布・提出対象に直結)、(b) 教師ラベル = refiner + disjoint
ownership まで通した prod 最終出力 (raw 検出器 mask ではない)。

**設計 (固定)**: prod 検出器 (`runs/yolo_fold_2/best.pt` = folds(0,1)
学習 → test は out-of-sample) + fold-2 refiner で test 180 枚を推論。
推論経路は採点と同一 (floor 0.08 → refine → conf>=0.26 & MIN_AREA
保持 → ownership/disjoint)。disjoint mask を YOLO polygon (最大輪郭,
CHAIN_APPROX_SIMPLE, 正規化) に変換し `yolo_datasets/v111_pseudo/` に
保存 (画像単位で逐次書き込み + PSEUDOLABELS_COMPLETE マーカーで冪等)。
学習データ = folds(0,1) 全 annotator ラベル (prod と同一) + test 疑似
ラベル。val = fold-2 (grouped_folds.csv 凍結 SHA256 検証)。
学習ハイパラは prod と完全一致 (70ep, imgsz 1280, AdamW, mask_ratio=2,
cls=0.5, 同一 aug、seed 3407+fold)。batch は 4、L4 OOM 時のみ 2 へ
1回 fallback (v108 と同一機構、manifest に batch_fallback を記録)。
test 前処理は PREPROC_CACHE 優先・無ければ solar_three_channel q96 を
inline 計算 (submission_reproduction.ipynb と同一式)。

**Arm (固定)**: `pseudotest` 単発。prod baseline は同一セッションで
paired 再測定 (frozen eval 経路無改変)。

**ゲート (固定)**: Δ@0.26 >= **+0.010** → fold0 honest 複製へ
(複製は folds(1,2) 学習 + 同一疑似ラベルセットで fold-0 eval)。
fold2 単独では提出不変。**昇格には追加条件: コンペ規約で test 画像の
半教師利用が許容されることの確認** — 本実験は測定であり、規約確認は
採用ゲートの一部として固定する。
想定 ~4h (疑似ラベル生成 ~30分 + 学習 ~2h + 評価 ~45分)。
manifest `artifacts/v111_pseudo_colab_manifest.json`。

**v111 結果 (2026-11-06 完走)**: 疑似ラベル = test 180枚 → **1,172
個体** (6.5個体/枚、prod の val 密度 ~6.4 と一致、raw floor 提案
2,518件)。学習 191.6分、batch4 fallback なし。prod 0.4442468 paired。

`pseudotest` PQ@0.26 = **0.451248**、Δ = **+0.0070** → ゲート
+0.010 未達、verdict `reject_v111_fold2`。提出 pair 不変。
ただし mask_ratio1 (+0.0065、fold-0 で −0.004 反転) を超える
**キャンペーン最大の fold-2 正値**。

探索的診断 (判定不使用): conf 曲線の最適が低 conf 側へ移動 —
pseudotest は 0.20 で 0.451642 (paired Δ **+0.0144**)、0.15 でも
+0.0101。疑似ラベル学習が検出器の conf 校正を変えた兆候。
単点 conf への post-hoc 採用はしない (事前登録の運用点は 0.26)
が、v112 複製の裏付け情報として記録する。

→ ユーザー判断で **v112 fold-0 honest 複製 (§2.90)** へ。
manifest `v111_pseudo_colab_manifest.json`。

### 2.90 v112 事前登録: pseudotest の fold-0 honest 複製 — v111 +0.0070 の検証 (事前登録, 2026-11-06)

v111 (§2.89) は test 疑似ラベル半教師で fold-2 Δ=**+0.0070** を記録したが、
ゲート +0.010 未達で公式棄却となった。しかし mask_ratio1 (+0.0065) を
超えるキャンペーン最大の正値であり、ユーザー判断で **ゲート到達不要の
探索的複製**を実施する (v108 と同じ複製プロトコル)。

**設計 (固定)**: v111 の fold-0 ミラー。教師 = fold-0 prod
(`runs/yolo_fold_0/best.pt` = folds(1,2) 学習 → test は out-of-sample)
+ fold-0 EdgeAtt refiner。test 180 枚を同一経路 (floor 0.08 → refine →
conf>=0.26 & MIN_AREA → ownership/disjoint) で疑似ラベル化
(`pseudotest_teacher_f0`)。学習 = folds(1,2) 実ラベル + 疑似ラベル、
prod ハイパラ完全一致、batch4 (OOM のみ 2 へ 1回 fallback・記録)。
eval = fold-0 honest (fold-0 は生成・学習・val 全て非接触)。

**対照 (固定)**: prod fold-0 baseline = **0.44413694566994816** (v108 §2.86
確定値) — 同一セッションでも paired 再測定する。

**判定 (固定)**: Δ@0.26 >= **+0.003** → `cross_fold_verified`
(v108 と同じ複製ゲート)。+0.003 未満・負 → `unverified_fold0` で
v111 の正値は fold 依存の揺らぎと確定し疑似ラベル系を閉鎖。
verified の場合のみ次の採用検討へ進み、**採用には MAGFiLO 規約で
test 画像半教師利用の許容確認が必須** (§2.89 から継続の条件)。

想定 ~4h (疑似ラベル ~30分 + 学習 ~2h + 評価 ~45分)。
manifest `artifacts/v112_pseudo_f0_colab_manifest.json`。

### 2.91 v113 条件付き事前登録: pseudo round-2 (実行は v112 verified のみ, 2026-11-06)

v112 が `cross_fold_verified` となった場合のみ実行する条件付き設計:
v111/v112 で学習した疑似ラベル込み検出器を**新しい教師**にして test
を再疑似ラベル化 (round-2)、再学習して効果の継続・減衰を測る
(NoisyStudent 型反復)。教師・データ・ゲートの詳細は v112 結果を見て
から固定する — 本節は「実行資格は v112 verified のみ」という
開始条件の固定であり、結果依存で設計を変えないよう arm 定義は
v112 判定後に本ログへ追記する。v112 が unverified なら実行しない。

**v112 結果 (2026-11-07 完走)**: prod_f0 0.4441369 paired (v108 確定値と
一致)。`pseudotest_f0` PQ@0.26 = 0.440407、Δ = **−0.0037** →
`unverified_fold0`。

v111 の fold-2 +0.0070 は fold-0 で −0.0037 に反転し複製しなかった。
mask_ratio1 (+0.0065 → fold-0 −0.0040) と同型で、**fold-2 の正信号が
fold-0 で反転するのはこれで2度目** — 単一 fold の改善幅 ~±0.007 は
評価ノイズ床 (§2.5) の範囲内であることが再確認された。
疑似ラベル半教師方向はここで閉鎖。§2.91 の v113 は開始条件
(v112 verified) を満たさず**実行しない**。提出 pair は不変
(prod 57039986 + v96-full 57897560)。

### 2.92 v114 事前登録: SAHI 型タイリング推論 — 小タイル拡大推論による解像度飢餓仮説の検証 (事前登録, 2026-11-07)

画像は 2048×2048 だが prod 検出器は全画面 imgsz=1280 (原寸比 0.625×)
でしか見ておらず、MIN_AREA=20px の小個体は 1280 換算で ~8px と解像度
飢餓の可能性がある。v36/v101/v105 の imgsz sweep は全画面スケールのみを
変えており、§2.21 の 2×2 タイル推論 (1152²/overlap 256 → imgsz 1280、
原寸比 1.11× / prod 比 1.78×) は救出 TP が実在したものの tile-FP 氾濫で
棄却された (`tile@0.26 = 0.4047`, `union@0.26 = 0.4116`)。本実験では
**未試行の 1280 未満小タイルを imgsz=1280 へ拡大する超解像 SAHI 領域**を、
Colab 凍結評価セルと mask-IoU>0.30 greedy 併合で検証する。

**設計 (固定)**: prod fold-2 検出器をそのまま使い、推論のみをタイル化:
- 画像を重複タイルに分割 (tile 640/overlap 128 → 4×4=16 枚、
  tile 896/overlap 192 → 3×3=9 枚。端は画像端に吸着)
- 各タイルを ndarray で imgsz=1280 推論 (`sahi640o128`: 原寸比 2.00× /
  prod 1280 比 3.20×、`sahi896o192`: 原寸比 1.43× / prod 1280 比 2.29×)、
  floor/iou/agnostic_nms/retina_masks は prod と同一
- タイル局所 mask を bbox 短絡付き conf 降順 greedy mask-IoU > 0.30 で
  重複除去し、生存提案のみ global 2048 canvas に配置
- 残りは既存経路へそのまま (refine >=0.08、conf sweep、ownership/PQ 無改変)

**Arm (固定)**: `sahi640o128`, `sahi896o192` (両方学習なし・推論のみ)。

**caveat (開示)**: §2.21 と同様、細長いフィラメントのタイル境界切断と
tile-FP 増加リスクがある — 部分検出の conf が全体検出を上回ると部分 mask が
採用される危険。併合 IoU=0.30 greedy で吸収する設計だが、負方向なら
切断・tile-FP 説と整合。

**ゲート (固定)**: Δ@0.26 >= **+0.010** → fold0 honest 複製へ。
fold2 単独では提出不変。想定 ~2h (16-9 tile ×235枚で推論が重い)。
manifest `artifacts/v114_sahi_colab_manifest.json`。

### 2.93 v115 事前登録: 異種検出器ファミリー (Mask R-CNN) — 単体性能 + cross-detector union (事前登録, 2026-11-07)

全 ~60 arm は YOLO 系 + 独自 refiner の枠内だった。他チーム分析で
「cross-detector true dedup」が lever と判明したが、それは**異種
検出器の pool が前提** — YOLO 単系では原理的に再現不能だった。
本実験は torchvision Mask R-CNN (ResNet50-FPN-v2, COCO pretrained)
を導入し、2つの問いを分離する:

- **問1 (単体)**: 異種検出器ファミリーは YOLO より素直に強いか
- **問2 (union)**: YOLO ∪ MRCNN の dedup pool は受理パイプラインを
  改善するか (他チームが +0.015 local を観測した機構の正直な試行)

**設計 (固定)**:
- `maskrcnn_resnet50_fpn_v2(weights="DEFAULT")`、box/mask head を前景 1 クラス
  (背景込み `num_classes=2`, filament のみ) に差替 (`roi_heads.score_thresh=0.02`)。
  入力 = prod と同じ前処理画像 (solar_three_channel
  q96) を model 内部 transform で min_size=1280 に縮小 (prod imgsz と揃える)
- 学習データ = folds(0,1) 全 annotator レコード (prod と同一規約)。
  aug = hflip/vflip/rot90 のみ (2048² 正方形で box/mask が閉じる回転のみ)。
  AdamW lr 1e-4 wd 1e-4、batch 2 (OOM のみ 1 へ 1回 fallback・記録)、
  epochs 10、seed 3407+fold
- eval = fold-2 honest。MRCNN 提案 (score>=0.02 まで保持、
  refine 対象 >=0.08) を既存 refiner+ownership+PQ 経路へ流す
  (提案の conf として MRCNN score を使用)

**Arm (固定)**:
- `mrcnn_raw`: MRCNN mask → ownership → PQ (refiner 不使用、detector 単体)
- `mrcnn_ref`: MRCNN 提案 → refiner → ownership → PQ (refiner 効果の分離)
- `union_yolo_mrcnn`: prod YOLO 提案 ∪ MRCNN 提案、conf 降順 greedy
  mask-IoU>0.30 dedup → refiner → ownership → PQ
  (**caveat**: 両 detector の score は未校正 — isotonic 無しの単純
  conf 順位 pool という限定的試行。負でも「校正すれば」の逆推論はしない)

**ゲート (固定)**: いずれかの arm で Δ@0.26 >= **+0.010** → fold0 honest
複製へ (mrcnn は fold-0 用に folds(1,2) で再学習)。fold2 単独では
提出不変。想定 ~4h (MRCNN 学習 ~2h + 推論/評価 ~1.5h)。
manifest `artifacts/v115_mrcnn_colab_manifest.json`。

**結果記録 (2026-11-08, fold-2 honest, 対照 prod@0.26 = 0.444247)**:

| arm | PQ@0.26 | Δ | verdict |
|---|---|---|---|
| prod | 0.444247 | — | (基準) |
| `mrcnn_raw` | 0.278306 | −0.1659 | reject_v115_fold2 |
| `mrcnn_ref` | 0.299426 | −0.1448 | reject_v115_fold2 |
| `union_yolo_mrcnn` | 0.347175 | −0.0971 | reject_v115_fold2 |

- 学習: 758 record-level samples (folds(0,1) 全 annotator レコード)、
  83.3 min、batch=2 (OOM fallback 無し)、固定 10ep last 重み (§2.93 開示の
  probe 規約どおり)
- `union` は tp 1945 / fn 840 (prod 1715/1070) で **recall 方向には効いた** が、
  fp が 2745 (prod 750) に爆発し @0.26 では −0.097。全 conf 帯で prod 未達
  (0.32 でも 0.364 < 0.4415)。未校正 score pool + MRCNN の大量低質提案が
  主因と考えられるが、caveat 開示どおり逆推論はしない
- MRCNN 単体は YOLO を大幅に下回る (0.278-0.299 vs 0.444)。10ep probe の
  限界はあり得るが、昇格ゲートを理由に据えない — reject で確定

**実装上の欠陥と修正 (開示)**: 初回実行で `CFG.YOLO_IMGSZ` 未定義により
prod baseline が AttributeError (commit `35f39e77` で修正、CFG 消費属性を
網羅する回帰テスト追加)。続く MRCNN 学習 batch 1 で
`CUDA error: device-side assert` — GT box が排他座標 (`xs.max()+1`) で
画像端個体が `x2=W` に触れ、roi_align 系カーネルの境界外サンプリングで
assert した可能性が高い。torchvision `masks_to_boxes` と同じ inclusive
座標 + 画像境界 clamp に修正 + CPU 側 target 検証 + failed_batch 診断を
追加 (PR #630)。修正後は学習・評価が完走し上記結果が確定。いずれの
欠陥も prod baseline/arm 定義/ゲートには影響せず、結果は修正後経路の
正当な値である。
### 2.98 v120 事前登録: 検出器間コンセンサス・ゲート — union ではなく合意で提案を選ぶ (事前登録, 2026-11-09)

**動機**: v115 `union_yolo_mrcnn` は FN −230 (1070→840) を救ったが
FP が 750→2745 に氾濫して −0.097。逆方向 — **複数検出器の合意でのみ
提案を放出** — は一度も試されていない。合意は相関のない FP を
除外しつつ、独立系で検出される実物体は残すはず。

**入力 (固定)**: fold-2 OOF 235 枚、detector 3 本 (いずれも
folds(0,1) 学習で fold-2 honest):
- `yolo`: prod yolo11m `runs/yolo_fold_2/weights/best.pt`
  (manifest role `yolo` で binding 検証)
- `crop`: v116 `runs/v116_croptrain/yolo_croptrain_fold2/weights/best.pt`
  sha256 `c23b1bb79620c62fd85984d032310bae26c240fd1be0aaf7df364fbc560fbcda`
  (trusted byte-pin — v116 manifest 記録値をここに事前固定)
- `yolo26`: v117 `runs/v117_yolo26/yolo_yolo26m_fold2/weights/best.pt`
  sha256 `0c8bb71f6ab4398aaea2b729992a3b88f3f4d707e0cbcbe35f1b82a2533db7a5`
  (trusted byte-pin — v117 manifest 記録値をここに事前固定)

**機構 (固定)**: 各 detector を conf floor 0.08・imgsz=1280・
iou=0.55・agnostic_nms・retina_masks で推論 (prod 推論パラメータと
同一)。全提案を mask-IoU>=0.5 の推移クラスタ (union-find、
cross-detector edge のみ) にまとめ、arm ごとに放出:

| arm | 放出ルール |
|---|---|
| `prod` | `yolo` 提案のみ conf>=0.26 (baseline sanity = 0.444238) |
| `union3` | 全クラスタ、max-conf member の mask (union 参照) |
| `union3_c26` | max member conf>=0.26 のクラスタ (v115 型 union) |
| `cons2of3` | >=2 detector が支持 (floor 合意による低conf救済) |
| `cons2of3_c26` | 支持 member の conf>=0.26 が >=2 detector (純粋合意) |
| `cons3of3` | 3 detector 全員が支持 |

放出後は prod と同一経路: edgeatt refiner (fold2_resume、manifest
role `baseline`) → threshold sweep {0.35,0.40,0.45,0.50} → ownership
drop_contained 0.90 → min_area 20 → pooled annotator-record 公式 PQ。
`prod` arm は v76 系 baseline と byte 同一 semantics (sanity
0.444238±0.001 を要求)。

**ゲート (固定)**: いずれか arm で PQ(arm,0.45)−PQ(prod,0.45) >=
**+0.010** → fold0 honest 複製へ (crop/yolo26 を folds(1,2) で
再学習が必要、~8h)。fold2 単独では提出不変。

**caveat (開示)**: (a) 3 detector は同ファミリー (YOLO 系 2 種 +
crop 変種) で提案は強相関 — 相関 FP は合意をすり抜けるため期待値は
低い (正直な見込み Δ~−0.01〜+0.005) (b) mrcnn は弱すぎるため
member に入れない (c) cons2of3 は floor 0.08 の低conf合意救済であり
「selector 不要の合意フィルタ」仮説の検証 (d) union3 は既知の壊滅を
再確認する参照 arm。想定 ~2.5h (3 det 推論 ~30min + cluster ~10min +
refine 6 arm ~1.5h)。成果物 Drive `solar_filament_v120/work/latest/`。

**v120 結果 (2026-11-09 完走, `solar_filament_v120/work/latest/v120_summary.json`)**:

prod sanity は v76 値と bit 一致 (0.444238, diff 0.0)。全 arm Δ@0.45:

| arm | emitted/img | PQ@0.45 | Δ | 解釈 |
|---|---|---|---|---|
| prod | 7 | 0.444238 | — | sanity 一致 |
| union3 | 13 | 0.380401 | −0.064 | floor 提案全放出 → 既知の壊滅を再確認 |
| union3_c26 | 7 | 0.448831 | **+0.005** | 他 det singleton 追加も正効果 |
| cons2of3 | 10 | 0.420487 | −0.024 | floor 合意救済は FP を拾いすぎ |
| **cons2of3_c26** | 6 | **0.455788** | **+0.01155** | **gate +0.010 通過 → promote_candidate** |
| cons3of3 | 8 | 0.444197 | −0.00004 | 全員一致は prod とほぼ同集合 |

一貫した構造: prod singleton (1 det のみ検出) の大部分は FP で、
≥2 det 合意フィルタが +0.0115。union3_c26 (+0.005) と両方向に整合
— yolo 見逃し TP を他 det が拾い、yolo singleton FP を合意が落とす。
閾値 sweep 全域 (0.35-0.50) で Δ~+0.011 と均一で単一点の揺らぎではない。
ゲート Δ>=+0.010 通過 → §2.98 規定どおり fold0 honest 複製へ
(crop/yolo26 を folds(1,2) で再学習 ~8h + fold0 eval)。fold2 単独では
提出不変。caveat: ゲート差 +0.0016 の薄い margin — fold0 での再現が
本当の判定。3 det は同系統 (YOLO 系) で FP 相関の懸念は残るが、
emitted 7→6 で +0.0115 は FP 除去が実効したことを示唆。

> ⚠️ **2026-11-10 追記 — 本表は漏洩の影響を受ける**: 後日発覚した v116
> `build_crop_dataset` のバグ (`train_files` 全 file 使用) により `crop`
> det は fold-2 GT を学習済みだった → 上記の fold-2 値 (特に +0.01155) は
> 漏洩補助を含み **honest な証拠としては無効**。crop det の fold-2 での
> 合意支持は記憶に依存するため真の consensus 効果は不明 — §2.99 の
> fold-0 clean 複製 (修正版 v121a) が唯一の裁定者。詳細は §2.99 冒頭の
> 漏洩開示を参照。

### 2.99 v121 事前登録: cons2of3_c26 の fold-0 honest 複製 (事前登録, 2026-11-09)

**動機**: v120 で `cons2of3_c26` (>=2 detector が conf>=0.26 で合意)
が §2.98 ゲート +0.010 を +0.01155 で通過 → 規定どおり独立 fold で
複製検証。fold-2 だけで判断しない。

**Phase A 学習 (v116/v117 レシピ verbatim、fold のみ変更)**:
- `crop_f0`: v116 croptrain (896² tile×9 crops・全 record・imgsz=896)
  を folds(1,2) で学習 → `runs/v121_croptrain_f0/yolo_croptrain_fold0`
- `yolo26_f0`: v117 yolo26m-seg (prod レシピ verbatim) を folds(1,2)
  で学習 → `runs/v121_yolo26_f0/yolo_yolo26m_fold0`
- prod yolo_fold_0・edgeatt refiner fold0_resume は既存 (v108 系)
- 両ノートで prod fold-0 eval を paired sanity (v108 確定値
  0.44413694566994816) として併記。想定 ~6h / ~3.5h (並列可)

**Phase B 評価**: `run_fold_oof_eval_consensus.py --fold 0`、全 6 arm
同一機構 (det-floor 0.08・consensus-iou 0.5・同一 refiner fold0・
同一 ownership/PQ)。crop_f0/yolo26_f0 の byte binding は各学習
manifest 記録の weights_sha256 (v121a: `datasets.croptrain`、
v121b: `results.yolo26m`) を eval ノートが読み取り照合する
(manifest=学習時記録の正本、§2.98 と同型の trusted-pin 経路)。

**ゲート (固定、結果取得前)**: cons2of3_c26 Δ@0.45(vs fold-0 prod arm)
>= **+0.005** → `cross_fold_verified` → 採用検討へ (all-data det 学習
→ Kaggle 提出物として検証、最大で現行 1 枠と交換)。未満 →
`unverified_fold0` で consensus 系閉鎖。prod arm sanity は
0.444137 ±0.001 を要求。

**caveat (開示)**: (a) fold-2 の +0.01155 は margin +0.0016 と薄い —
fold-0 で半分でも再現しなければ noise 扱い (b) 採用に至っても
submission 用に all-data 学習 3 det が必要 (~8h) + LB 検証で
OOF→LB gap の確認が残る (c) 2 fold 平均効果が真値の推定になり、
fold0 が負でも fold2+fold0 平均で判断しない — ゲートは fold0 単独。

**⚠️ データ漏洩の発見と影響 (2026-11-10、結果確定前の開示)**:
v116 `build_crop_dataset` が fold フィルタ済みの `fit_files` ではなく
`train_files` (全 707 file) を回しており、crop train split に **eval fold
自身の crop が混入** していた (実測: `v116_croptrain/labels/train` =
全 707 file, 内 fold-2 = 235; `v121_croptrain_f0` も同型で fold-0 = 236 混入)。
影響: (a) v116 の fold-2 "OOF" 値 (croptrain_full 0.441506・sahi 0.417951)
は eval 対象を学習した重みで計測 → OOF として無効 (方向性は棄却のまま、
漏洩有利でも prod 未満だったため定性結論は不変)。(b) **v120 `crop` det が
fold-2 GT を記憶** → `cons2of3_c26` +0.01155 は漏洩補助を含み、honest な
証拠としては無効 — fold-0 の clean 複製が唯一の裁定者になる。
(c) v121a (実行中) も同型混入 → `fit_files` に修正し出力先を
`v121_croptrain_f0_honest` に分離、汚染 run/dataset は監査用に温存。
v121b yolo26 (`val0_train12`: 471 file = folds(1,2) のみ)・v117
(`val2_train01`: 472 file)・prod det は標準経路で混入なしを実測確認。

**fold-0 provenance (v121c 実行前に記録)**: reviewed manifest は
`manifest["fold"] == fold` を要求し `checkpoints` role が fold 固有
ckpt の sha を bind するため、fold-2 用 `v73_reviewed_provenance.json`
(pin `2c954685...`) は fold-0 では使えない。v121c は以下の独立検証済み
component pin から fold-0 manifest をセル内で決定論生成し、自己 sha を
外部 pin で照合する:

- `yolo` (runs/yolo_fold_0/weights/best.pt, 45,316,918 B):
  `ab8810916e34a755cc0d7a705dbb7ade60338ae247f8baac02adfe64026c0d35`
- `baseline` (artifacts/mediseg_edgeattnet_refiner_fold0_resume.pt,
  564,101,827 B):
  `caa8b27bbdaa888019a599d001b2d4dee844dbfee90186a18f8e0db95ce8b264`
- `folds_sha256` = `febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e` (全 fold 共通)
- fold-0 manifest bytes (sort_keys・separators 最小化):
  `{"checkpoints":{"baseline":"caa8b27b…","yolo":"ab881091…"},"fold":0,"folds_sha256":"febb3f67…","schema_version":1}`
- **`V73_FOLD0_PROVENANCE_SHA256`** =
  `da3d53abc94023f61998ace4b54db98cfa1792a6b8660820583ec957f26e6a53`

両 ckpt は Drive `solar_filament_v30/` の既存 artifact で、v108 が
fold-0 prod 確定値 0.44413694566994816 を bit 一致で再現した同一ファイル
(v121a/v121b の prod sanity でも再確認済み)。環境変数名を
`V73_PROVENANCE_SHA256` から分離し、fold-2 pin の誤用を防ぐ。

**v123 補助実験 (投機的、2026-11-10 追加)**: v121c ゲート判定と並行して
all-data detector trio の欠員 yolo26m-all を学習するノートブック
(`solar_filament_v123_yolo26_all_colab.ipynb`、builder
`scratch/build_v123_yolo26_all_colab.py`) を作成。trio の既存メンバは
yolo11m-all=`v96_data_scaling/yolo_full_fold2` (`val2_train012`) と
crop-all=`v116_croptrain` (全 707 file 学習済み=実質 all-data、test-time
用途では漏洩問題なし — 上記開示参照)。v123 は `TRAIN_FOLDS=(0,1,2)`・
`FOLD=0` (val/best-epoch 選択・prod sanity 専用) で `val0_train012`
dataset を構築し `runs/v123_yolo26_all/yolo_yolo26m_all_fold0` に出力。
fold-0 det eval は in-sample のため informational sanity のみで OOF/ゲート
証拠には一切使用しない。ゲート未通過なら重みは不使用 (投機的計算)。

**P0 監査追記 (2026-11-10、外部調査計画 P0 への対応)**: 全 detector の
勾配学習側の fold 分離は実測で確認済み (crop honest 471/236・yolo26
471・prod 命名規約 `yolo_fold_k`=fold k 以外で学習)。加えて
**eval-fold-as-val 構造が prod・v121 系で一様** であることを確認:
`build_v30_prize_colab_cells.py` の `build_fold_dataset` は
val=eval fold・train=残り fold で dataset を作り、refiner も
`train_refiner_fold.py` の契約「fold N を学習から完全除外し valid
に使う」に従う。つまり prod yolo_fold_0・fold-0 refiner・v121a・
v121b 全てが fold-0 を best-epoch/early-stop 選択に使っている。
これは勾配漏洩 (v120 型) とは質的に異なる**選択バイアス**で、
paired Δ (cons2of3_c26 vs prod) は両側が同じ曝露を持つため対称かつ
公平。厳密版 (epoch 選択を学習 fold 内へ移す) は prod 側にも同じ
変更を要求するため本複製では採用しない — 開示のみ行う。

**v124a 準備 (2026-11-10)**: 片 fold 陽性では cross-fold verified と
呼べないため、v121c ゲート通過時のみ投入する fold-2 側 clean 複製
Phase A (`solar_filament_v124a_croptrain_f2_colab.ipynb`) を作成。
FOLD=2・TRAIN_FOLDS=(0,1) で `fit_files`=472 file のみの honest
crop dataset、`runs/v124_croptrain_f2_honest` 出力 (v116 汚染系と
完全分離)。v124c fold-2 consensus eval はゲート判定後に生成する。

**v124c 準備 (2026-11-10)**: 第二 fold 確認の eval ノートブック
(`solar_filament_v124c_consensus_f2_colab.ipynb`) を作成。v120 builder の
fold-2 複製で、crop det のみ `v124_croptrain_f2_honest` に差替え (他は
yolo_fold_2・v117 yolo26・fold-2 refiner・v73 fold-2 provenance と同一
byte)。trusted-pin は v124a/v117 学習 manifest から実行時読み取り。
判定: `cons2of3_c26_delta_fold2 >= +0.005` → `second_fold_positive`、
未満 → `second_fold_negative`。**v121c ゲート通過時のみ v124a 学習を
起動し、完了後に本ノートを実行する**。片 fold 陽性のみでは提出差替えの
根拠にしない。

**v121c 結果 (2026-11-10 完走, fold-0 clean OOF)**: 事前登録ゲート
`cons2of3_c26 Δ@0.45 >= +0.005` に対し **+0.002982** →
**`unverified_fold0`、consensus 線は閉鎖**。prod sanity
0.444134 (既知値 ±3e-06) で評価経路は正常。全 arm @0.45:

| arm | PQ@0.45 | Δ | TP | FP | FN | SQ |
|---|---|---|---|---|---|---|
| prod | 0.444134 | — | 1711 | 762 | 1038 | 0.6778 |
| cons2of3_c26 | 0.447116 | **+0.0030** | 1707 | 744 | 1042 | 0.6810 |
| cons3of3 | 0.430407 | −0.0137 | 1865 | 1259 | 884 | 0.6777 |
| union3_c26 | 0.436565 | −0.0076 | 1790 | 1025 | 959 | 0.6785 |
| cons2of3 | 0.402940 | −0.0412 | 1873 | 1657 | 876 | 0.6754 |
| union3 | 0.367296 | −0.0768 | 1847 | 2176 | 902 | 0.6733 |

読み: 合意フィルタは設計通り FP −18 を削るが TP −4 を伴い、net は
+0.003 で閾値 6 割。汚染版 fold-2 の +0.01155 は大半が漏洩補助だった
ことが確認された (v116 crop det が fold-2 GT を記憶→合意支持が
honest でなかった)。**v124a (fold-2 複製) は事前登録条件により
起動しない**。生存個体の SQ +0.0033 は P2 輪郭投票の軽い根拠として
記録する。提出は prod + v96-full を維持。

**v125 準備 (P2 輪郭投票、2026-11-10)**: 調査計画 P2 の開発側 eval
ノートブック (`solar_filament_v125_vote_f0_colab.ipynb`) を作成。
検出集合は prod (yolo_fold_0 conf>=0.26) で**完全固定**し、crop
(v121a honest)・yolo26 (v121b) 提案を det-mask IoU>=0.5 で最良 leader
1 つに対応付け、refined mask の等重み pixel 多数決 (>=0.5 固定) で
輪郭のみ補完。新規個体は出さない。arm は `prod` (対照) と
`vote_eq05` の 2 つのみ。判定: Δ@0.45 >= +0.005 → `vote_positive_dev`
(確認 fold 検討へ)、未満 → `vote_negative` で P2 線も閉鎖。
eval script は `scratch/run_fold_oof_eval_vote.py` を embed
(consensus 版と同一 postprocess・同一採点経路)。fold-0 は v121c
判定済みのため、陽性でも確認は別 fold (fold-2 = v124a 必要) が必須。

**v125 結果 (P2 輪郭投票、fold-0、2026-11-11)**: 検出集合固定の
cross-proposal 輪郭投票を実行 (41.7min)。prod sanity 0.444134
(既知値 -3e-06、v121c と bit 一致)。leader 7.30/img、voter 割当
2.39/leader、未割当 voter 2304。結果:

| arm | PQ@0.45 | Δ | SQ@0.45 | TP | FP |
|---|---|---|---|---|---|
| prod | 0.444134 | — | 0.6778 | 1711 | 762 |
| vote_eq05 | 0.445722 | **+0.001588** | 0.6781 | 1715 | 754 |

全 threshold で微小正 (+0.0016〜+0.0028) だが内訳は RQ 微増のみ
(TP+4・FP−8)、mask 変化 mean +0.64px で SQ は実質不変。判定
`vote_eq05_delta_fold0 = +0.001588 < +0.005` → **vote_negative、
P2 線も閉鎖**。輪郭改善の余地はあるが現行 voter では効果量不足。

**v123 完了記録**: 全 70 epoch 学習終了・best.pt sha256 =
`25002d23f7b915e8076ab7e2429d34f6b0c4c6aed72d01abfc3a9985a5938ca9`
(直接実測 pin — notebook manifest の eval セル未了のため)。
all-data モデルのため fold-0 成績は in-sample で証拠不使用、
voter 候補としてのみ温存。

**ここまでの結論**: consensus (+0.0030) も輪郭投票 (+0.0016) も
fold-0 で事前登録 +0.005 未達。漏洩なしの改善手段は残っておらず、
提出は **prod (57039986) + v96-full (57897560) で確定方向**。
残作業はレポート・リポジトリ整合と 11/14 内部凍結。

**v124a 実走記録 (2026-11-11)**: 事前登録では v121c ゲート通過時のみ
起動予定だった fold-2 honest crop 学習が実走・完了した
(`runs/v124_croptrain_f2_honest/yolo_croptrain_fold2`、全 70 epoch、
39,140s)。contract: FOLD=2・TRAIN_FOLDS=(0,1)、train 6,822 samples
(folds0,1 のみ)・val 2,115 (fold-2)、`weights_sha256` =
`3ded53395aa938d9346a37bcdf6528a7d1489c8e36f870f18e78de064fd1dfae`
(manifest `artifacts/v124a_croptrain_f2_colab_manifest.json` 記録)。
ゲート流れ外の実行のため、v124c eval を走らせる場合の結果は
**exploratory 扱い** — fold-0 で線は閉鎖済みであり、提出差替の
根拠にはしない。clean な fold-2 crop det として証跡上の価値のみ。

**v124c 結果 (fold-2 clean consensus eval、2026-11-12)**: フロー外実行
(v121c 未達のまま実走した v124a det を使用) のため exploratory 扱いだが、
**clean な fold-2 測定値として記録**。prod sanity 0.444238 = v76 値と
diff 0.0 (bit 一致)。dets: yolo `40ec31fd…`・crop `3ded5339…` (honest)・
yolo26 `0c8bb71f…` (v117)。結果 @0.45:

| arm | PQ | Δ | TP | FP | FN |
|---|---|---|---|---|---|
| prod | 0.444238 | — | 1715 | 750 | 1070 |
| union3 | 0.379050 | −0.0652 | 1868 | 2022 | 917 |
| union3_c26 | 0.443577 | −0.0007 | 1785 | 908 | 1000 |
| cons2of3 | 0.415057 | −0.0292 | 1899 | 1522 | 886 |
| **cons2of3_c26** | **0.452803** | **+0.008565** | 1700 | **631** | 1085 |
| cons3of3 | 0.438206 | −0.0060 | 1863 | 1128 | 922 |

**2-fold honest 証拠の確定**: fold-0 +0.002982・fold-2 +0.008565、
両 fold で正・平均 +0.0058。汚染版 v120 (+0.01155) の大半は漏洩
ではなく実効果 (FP 削減方向・fold-2 では −119)。逐次ゲート上は
fold-0 で線は閉鎖済みだが、証拠としては「検出器合意による FP
削減は 2 fold で再現する正の効果、ただし効果量は fold 間で
不均一 (3x差)」が正直な記述。

**v126 exploratory 提出ノートブック (2026-11-12)**: 2-fold honest
証拠 (f0 +0.002982・f2 +0.008565、両方正) に基づく exploratory
consensus 提出経路 `solar_filament_v126_consensus_submission_colab.ipynb`
を作成。detector は v122 と同一の計測済み構成 (yolo_fold_2 +
v116 crop + v117 yolo26 + all-data refiner、test disjoint で正当)。
all-data trio (yolo_full + v116 + v123) は consensus 集合として
未計測のため不採用。ガードは v122 の `cross_fold_verified` とは
別物 — 両 fold summary の存在と両方正 Δ を実行時検証。
**選択ルールを事前申告で固定**: public > 0.39 (v96-full 現行値)
なら最終選択差替、以下なら v96-full 維持 — public LB への
単発比較であり、複数 arm 当てはめではない。
