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
パイプラインの決定性と重み整合性が公式レベルで検証済み。
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
再現 — 提出パイプラインの環境非依存性が完全に実証された。

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
本文取得) を調査。結論: **0.55–0.63 の上位パックは漏洩由来である根拠が
強く、誠実系フロンティアは ~0.36–0.45 で我々の 0.39 はその先端**。

#### LB の見取り図 (複数独立ソースで一致)

| 帯域 | 解釈 | 根拠 |
|---|---|---|
| 0.55–0.63 | **漏洩由来の可能性が高い** | discussion 737177: 参加者解析で Harvard Dataverse 上の全 test GT が公開ダウンロード可能と特定 — 直接ブレンドで ~0.51–0.66・学習混入(laundering)で ~0.45–0.55 に着地する計算と上位分布が一致。743958: lamhuy8904 の 0.55 kernel はコミュニティが漏洩と特定。741892: 「~0.45(または 0.41)超は公開 test アノテーション使用」(Alisher 氏) |
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

### 2.17 「両提出を再現可能」の規約対応を完了 (2026-10-12)

主催者回答 (discussion 747960) の「選択した両方の提出を repo が再現
できること」要件について、第2提出 `plain_mhsa_f2` の再現経路を整備:

| 資産 | 状態 |
|---|---|
| Kaggle dataset `solar-filament-2026-weights` | **v2 公開** — `ablation_plain_mhsa_seed8407_fold2.pt` (282,089,497B, sha256 `b5f331e1…`) を追加。manifest 更新済み |
| `run_test_submission.py --weights-dir` | 追加 — dataset ダウンロード先から yolo/refiner/scorer 重みを解決 (Drive 不要)。両バリアントでローカル疎通確認済み (CPU) |
| `solar_filament_submission_reproduction.ipynb` | `CFG.SUBMISSION_VARIANT` を追加 — `"plain_mhsa_f2"` で `MediSegPlainMHSAUNet` + dataset 重みに差し替え、`submission_plain_mhsa_f2.csv` を出力 (builder 再生成・AST 検証済み、本実行は F3 クリーンランで検証予定) |
| `output/submission_weights_manifest.json` | Drive 側 manifest にも plain_mhsa エントリを追記 |

残検証: 再現 kernel の再 push + 実行 (prod/variant 両経路) — F3
(11/08 予定) のクリーン環境通し実行で実施する。コード経路は提出実績の
ある `run_test_submission.py` と同一クラス・同一 ckpt ロード式のため
リスクは低い。

### 2.18 レポート確定版への統合 + レビュー対応 (2026-10-12)

`report/main.tex` を最終提出版に更新し、MiKTeX/latexmk のローカルビルドで
**本文4p・謝辞+refs p5** を確認(従来はビルド未検証だった):

- §3.2 末尾に最終選択段落を追加 — prod + plain_mhsa_f2(両者 Public 0.39、
  best-match IoU 0.94/0.93)の2提出と選択根拠(refiner アーキテクチャ差による
  private 分散ヘッジ)、T4 ~1.1 s/img end-to-end、honest-tier 文脈
  (0.55–0.63 帯は test アノテーション公開漏洩と整合、誠実系 0.36–0.45、
  >0.35 は主催者精査対象 — §2.16 の所見を反映)
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
| ≥0.55 | **78** (うち **ちょうど 0.55 が65チーム**) | 単一値プラトー = 同一リーク blend の署名。前日比 +~8 でまだ増殖中 (Sayandeep 0.62・Tarunkush019 0.62 等 10-11/12 投稿) |
| 0.45–0.54 | 3 | Cac Cat 0.54・Thariq J 0.53・noisyislands 0.49 (未検証) — 想定した「laundering 帯」はほぼ空 |
| 0.40–0.44 | 31 | **誠実系トップは 0.41** (Spag99・Salty Taro・trycat・supersede・Manurupan M・Swarup S.M. の6チーム)。tushhaaarrrr (Fususu fork)・JAB100・Ishan Sharma・Varad 等は 0.40 |
| 0.35–0.39 | 332 | 我々の 0.39 は 53チーム同点の帯域先頭群 |

→ 「0.55–0.63 クラスタ」の描写を精緻化: **正確に 0.55 に65チームが
  収束するプラトー** は、同一の公開 test アノテーション CSV を同様に
  ブレンドした結果としか説明がつかない定量的指紋。誠実系実効天井は
  ~0.45 でなく **0.41** と判明。

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
一方で honest-tier 論証の外部証拠が強化された:
(1) 0.55 プラトー ×65チーム = リークの定量指紋、
(2) 誠実系フロンティアは 0.41、
(3) 第三者による Fususu 複製 0.35・val→LB ギャップ 0.04-0.06 の複数一致。
レポートの「0.55–0.63 帯は漏洩と整合・誠実系は 0.36–0.45」の記述は
「誠実系実効天井 ~0.41・0.55 同一値プラトー65チーム」と
より強い形に更新可能 (main.tex 更新は別途)。

### 2.20 両提出の決定的再現検証 + notebook 修正 (2026-10-13)

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
  CLI 経路が決定的再現の第一正本
- `dataset_upload/` staging: `C:/work/solar_run/dataset_upload/`
