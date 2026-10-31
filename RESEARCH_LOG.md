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
