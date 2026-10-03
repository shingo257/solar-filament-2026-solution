# 優勝を狙うための実験計画(2026-08-29策定)

本文書は、トップ層の手法分析・外部文献調査・公式ルール精読を踏まえて策定した、
残り約2.5ヶ月(最終締切2026-11-15)の優先順位付き実験計画である。数値の正本は
`README.md`のまま、本文書は戦略・優先順位の整理に専念する。

## 1. 前提となる調査結果(2026-08-29実施)

### 1.1 コンペティションの実像(公式Rules/Evaluationタブ精読)

- **賞金総額$3,000**(1〜3位、金額は変動制)。加えて上位2〜3チームはIEEE
  BigData 2026(2026-12-14〜17、Phoenix AZ)への招待講演・論文掲載機会あり
- **評価は「定量70% + 定性30%」の合議制。** リーダーボードのPQ値がすべて
  ではない
  - 定量70%: mean Dice、PQ、Dice/IoU分布、one-to-many/many-to-one分布
  - 定性30%: **パイプライン全体の説明の詳細さ**(前処理から最終予測まで、
    使用アルゴリズムのアーキテクチャ)、**予測の見た目の妥当性**(H-Alpha
    画像上での形態学的な自然さ)、**コード品質**(モジュール性・文書化)
  - これらの評価は**ソースコードの公開が前提**(Open-Access Policy)
- **最終提出物**: 4ページ技術報告書(Overleafテンプレート指定)+ 公開
  Gitリポジトリ(requirements.txt必須、パイプライン全体を示すnotebook必須、
  追加ファイルなしで再現可能なこと)。Googleフォームで申請
- **運営からの重大な警告(2026-08-20付アナウンス)**:
  > 「一部の参加者はシステムをハックすることに時間とエネルギーを費やして
  > いるのが明らかだ。正直な参加者にはスコアを他人と比較せず作業を続けて
  > ほしい。PQ 0.3を超えるスコアは我々にとって大きな価値がある」
  
  現在の公開リーダーボードで上位7チームがPQ **完全同率0.55**で並んでいる
  異常な挙動(既にREADMEで指摘済み)は、運営もこの警告で暗に問題視している
  可能性が高い。**リーダーボードの見かけの順位を追いかけること自体が
  トラップになりうる**——定性30%の合議制評価では、正直で再現可能な
  0.37〜0.4台のスコアの方が、説明不能な0.55より高く評価される可能性がある

### 1.2 外部データルールの確認(重要な制約)

Evaluationタブの正確な文言(2026-08-07更新):

> 「Although participants may use other external data sources, please note
> that: (1) for inference of ML models, only the H-alpha images provided
> in the test directory will be used; and (2) **the models may not use
> any other ground-truth metadata for training.**」

MAGFiLOはNature Scientific Data誌に公開された独立データセット
(DOI 10.1038/s41597-024-03876-y)であり、コンペ提供分(707観測)より
大きい可能性がある。しかし上記(2)により、**MAGFiLO本家の追加正解ラベルを
学習に使うことは明確に禁止**と判断する。この道は完全に閉じる。
(本プロジェクトが既に実施したpseudo-labelingは自社モデル生成の予測であり
「他者のground-truth」に該当しないため、この制約に抵触しない。従来の
判断は正しかった)

### 1.3 公開ノートブックの調査: semantic segmentation路線の実力天井

`realtalhacelik`(リーダーボード上のTalhaBacak、2026-08-27時点PQ 0.38)の
公開ノートブック「Solar Filaments: U-Net++ Baseline」(kaggle kernels pull
で取得、61セル)を精読した。この人物は非常に丁寧に以下を試している:

- U-Net++ (EfficientNet-B3) + patch-based学習(512×512、native解像度で
  barbを学習)
- Overlapping sliding window推論(50%オーバーラップ)
- 形態学的クロージングによる「filament stitching」(9×9カーネル)
- Test-Time Augmentation(水平/垂直flip)
- Global-Local(512リサイズ全体像+512パッチ)アンサンブル
- DeepLabV3+(ASPP)との比較 → 収束せず撃墜(5epochでは学習不足)
- 確率閾値・面積閾値のチューニング

この人物はDice指標時代(コンペ変更前)には**Public 0.59**まで到達していたが、
現在のPQ指標では**0.38**——我々の0.37とほぼ同じ帯にいる。**つまり、
semantic segmentation路線でどれだけ後処理を凝らしても、PQ指標下では
0.37〜0.40クラスタに漸近的に張り付く**ことが、独立した実験者によって
既に実証されている。これは重要な負の情報で、「もっと凝った後処理を
積む」路線への投資が費用対効果に乏しいことを裏付ける(我々自身の
V38-V49の負の結果とも整合)。

### 1.4 EdgeAttNet論文(arXiv:2509.02964、2025年9月) — 最有力の技術的手がかり

MAGFiLOデータセット上で直接ベンチマークされた、最も関連性の高い論文。
GitHub(`dasjar/EdgeAttNet`、MITライセンス、コード公開)。

**アーキテクチャ:**
- U-Netバックボーンに「Edge-Guided Multi-Head Self-Attention (EG-MHSA)」を
  ボトルネックに追加
- エッジ事前分布: `E = σ(Conv3x3(入力画像))` という単純な3x3畳み込み+
  sigmoidだけでエッジマップを生成
- ボトルネックの解像度(H/16×W/16)に合わせてbilinear補間し、1x1畳み込みで
  チャンネル射影: `E' = Conv1x1(Interpolate(E))`
- Self-Attentionに**加算**で注入: `Q = x_flat + E'`, `K = x_flat + E'`,
  `V = x_flat`(位置エンコーディングは意図的に省略)

**MAGFiLO上の実測(semantic mIoU、PQではない):**

| モデル | mIoU_pairwise | mIoU_multiscale | パラメータ数 |
|---|---:|---:|---:|
| U-Net | 0.5724 | 0.5848 | 31M |
| U-Net + MHSA(PE無) | 0.5856 | 0.6000 | ~31-35M |
| U-Net + MHSA(PE有) | 0.6200 | 0.6601 | ~31-35M |
| **EdgeAttNet** | **0.6451** | **0.7032** | **22.6M** |

学習設定: 1,295枚学習/45枚検証/99枚テスト、50epoch、Adam(lr 1e-4)、
BCE+Dice損失、**transfer learningもaugmentationも無し**でこの数値。

**なぜこれが重要か:** 我々のrefiner(`MediSegUNet`、31,036,481パラメータの
プレーンU-Net)は、まさにEdgeAttNetが比較対象にしている「U-Net」そのもの
であり、**アーキテクチャの置き換え先として直接的に対応する**。しかも
EdgeAttNetが明示的に狙っているのは「barbの認識精度向上」——我々が
一貫して診断してきた主要な失敗モード(near-miss境界品質、920件の
IoU 0.3-0.5)と正確に一致する。パラメータ数はむしろ減っている(22.6M <
31M)ため、計算効率(評価対象に明記されている)の面でも有利。

**留意点:** 論文の指標はsemantic mIoUであり、我々の指標(instance-level
PQ)とは定義が異なるため、上記の改善幅をそのままPQ改善に読み替えては
いけない(RESEARCH_LOG.md §1の既存の注意点と同じ)。

**【2026-08-29追記・重要な訂正】GitHubリポジトリの実態を直接確認したところ、
`model/edgeattnet_model.py`(本来`UNetEdgeTransformer`クラスが定義されて
いるべきファイル)の中身が誤って`main.py`と同一内容になっており、
**実際のEG-MHSAアーキテクチャのコードはリポジトリ上に存在しない**
(`gh api`でファイル一覧を全走査して確認、他にそれらしいファイルもない)。
学習済み重みの配布も確認できなかった(Releases無し、`has_downloads: false`)。
**「コード・学習済みモデルがGitHubで公開されている」という論文の記述は
現状のリポジトリでは裏付けが取れない。** よってPhase 2は論文本文の
アーキテクチャ記述(§1.4の数式)のみを根拠に、自前で再実装する前提に
変更する。幸い記述は十分に具体的(3x3conv+sigmoidのエッジ抽出、
bilinear補間、1x1conv射影、Q/Kへの加算、位置エンコーディング省略)で
再実装は難しくないが、著者オリジナル実装との一致は保証できない点に注意。

## 2. 見送る施策(根拠つき)

| 施策 | 見送り理由 |
|---|---|
| MAGFiLO本家データセットの追加取得・学習利用 | Evaluationタブで明確に禁止(§1.2) |
| semantic segmentation路線への後処理追加投資(sliding window改良、TTA強化等) | 独立した実験者が同じ路線でPQ 0.38止まりと実証済み(§1.3)、費用対効果に乏しい |
| 上位7チーム(PQ 0.55完全同率)のスコアを追いかける | 運営自身が「システムのハック」を問題視する警告を出している(§1.1)。定性30%評価では説明可能性・再現性の方が価値を持つ可能性が高い |
| pseudo-labeling再挑戦 | fold0・fold1の独立検証で不採用が確定済み(README §2.1.3) |

## 3. 優先実験計画

### フェーズ1: 定性30%への投資(即着手、GPU不要、費用対効果が最も高い) — 報告書ドラフト完了(2026-08-30)

リーダーボードのPQ値と独立に評価されるため、**残りの実験と並行して今すぐ
着手すべき最優先事項**。

1. **4ページ技術報告書をOverleafテンプレートに合わせて仕上げる — ✅ドラフト完了。**
   Kaggle公式Overview「Final Submission」節から実際のOverleafテンプレート
   (ACM sigconf、`main.tex`/`preamble.tex`/`main.bib`、4ページ制限は
   Acknowledgment/References除く、必須構成: Introduction/Methodology
   /Evaluation/Acknowledgment)を取得し、要求事項(パイプライン全体図・
   セグメンテーション例図・評価戦略・誤差付き結果・長所短所・改善案)を
   すべて満たす形で執筆した。ローカルに`report/main.tex`・`preamble.tex`
   ・`main.bib`・`figures/segmentation_example.png`として保存済み。
   Overleafは匿名閲覧リンクのみでCopy Project権限が無い(アカウント作成は
   代行できない操作)ため、代わりにtectonic 0.17.0(公式GitHub Releaseから
   取得、約80MB)をローカルへ導入し実際にコンパイル検証した:
   **成功、3ページ(本文2.5ページ+参考文献、4ページ制限に余裕あり)、
   TikZパイプライン図・セグメンテーション例図とも正しく描画、warningのみで
   fatalエラーなし。** `report/main.pdf`として保存済み。メールアドレスは
   記入済み(shingo257@gmail.com)。**残タスク: 氏名・所属・タイトルの記入
   のみ**(ユーザー確認: 2026-08-30「メールアドレスのみ記入、氏名/所属は
   プレースホルダのまま」)。
   `TECHNICAL_REPORT_DRAFT.md`は詳細な研究ログ版として引き続き参照用に残す。
2. **公開コードリポジトリの整備 — ✅ 単一notebook完成(2026-08-30)。**
   Kaggleルール「a jupyter notebook should illustrate the entire
   pipeline」に対応し、`scratch/build_master_pipeline_notebook.py`から
   `solar_filament_master_pipeline_colab.ipynb`(15セル)を生成した:
   (A)前処理・3-fold層化分割 → (B)YOLO11-seg検出器3-fold学習(既存
   checkpointがあれば自動スキップ) → (C)EdgeAttNet refiner 3-fold学習+
   all-data fine-tune → (D)V52構成でのtest推論・提出物生成 →
   (E)公式RLE厳密検証、の全ステージを1本に統合。全コードセルの構文
   (ast.parse)・セル間依存順序を検証済み(Colab実行はこれから)。
   これに伴い、これまでfold1/fold2/all-dataのEdgeAttNet refiner学習が
   Colabの一時セルとしてしか存在せず**リポジトリのコードだけではV52を
   再現できない**という欠落があったことも発覚・修正した
   (`scratch/build_v51_edgeattnet_refiner.py`を`--fold {0,1,2,all}`
   引数対応に一般化、旧fold0専用版は削除)。
   `requirements.txt`は既存のもので必要パッケージ(scikit-learn含む)を
   カバー済みと確認。
3. **予測結果の形態学的妥当性を示す図の作成** — 報告書用に1点は作成済み
   (`report/figures/segmentation_example.png`、既存showcaseの予測ポリゴン
   から生成)。`run_case_diagnostics.py`のcontact sheet活用は未着手
4. これらは定性30%の評価基準(パイプライン説明・コード品質・視覚的妥当性)
   に直接対応しており、GPU予算を一切使わずに勝率を上げられる

### フェーズ2: EdgeAttNetアーキテクチャによるrefiner置き換え実験 — ✅完了・採用確定(2026-08-30)

公開GitHub実装が実質存在しなかったため(§1.4)、論文本文の記述のみから
`EdgeGuidedMHSA`を独立再実装(`solar_filament/models/mediseg_models.py`の
`MediSegEdgeAttUNet`、+4.2Mパラメータ)。fold0のみで即断せず、必ず
fold1・fold2まで独立検証してから採用可否を判断するという当初方針通りに
3-fold全て実行した。

- fold0単独 Δ+0.0062(frac_positive 99.5%)、fold1単独 Δ+0.0009(73.9%、
  不確か)、fold2単独 Δ+0.0012(77.3%、不確か)、**3-fold全結合(n=707)
  Δ+0.0028、95%CI[0.000915,0.004756]でゼロを完全に除外**
- all-data refiner(fold0からウォームスタート、707枚で5epoch再学習)を
  V50と同一のfold2単体YOLO・後処理へ組み込み、実際にKaggleへ提出(V52)
- **結果: Public LB 0.37→0.39。OOFで予測した改善が符号・方向とも再現。**
  順位は444チーム中35位圏→13〜16位圏まで前進(2026-08-30)
- 詳細: README §2.1.5、`v51_edgeattnet_refiner_3fold_manifest.json`、
  `v52_edgeattnet_refiner_fold2_only_manifest.json`

**EdgeAttNet refinerを提出正本として正式採用。次の優先候補は、この
attentionベースrefinerをV50方式(フェーズ4)の恒久構成として固定した上で、
フェーズ1(定性30%)とフェーズ3(分布差調査)を進めること。**

### フェーズ3: OOF-Public LBギャップ(分布差仮説)の深掘り — ✅完了・有意差は検出されず(2026-08-30、【表現訂正】「反証」ではなく「未検出」)

V50実験(§README 2.1.4)で、ギャップの主因はアンサンブル統合ではなく学習/
テスト間の分布差である可能性が高まったため、実際に定量化した。

NSO GONGファイル名規則(`YYYYMMDDHHMMSS[site]h`)から年・サイトコードを
train(707)・test(180)双方で抽出し、カイ二乗検定で分布差を検定した結果、
**年代分布(p=0.423)・サイト分布(p=0.938)ともに統計的に有意な差は
検出されなかった。** train/testは年代・サイトという切り口では同一母集団
からの単純な無作為分割と矛盾しない。

**結論: 「学習/テスト間の年代・サイト分布差」仮説は、この2軸では有意差が
検出されなかった(「反証された」ではない — p値は「差がない証拠」ではなく
「差を検出できなかった」ことしか示さず、180枚という小サンプルでは
検出力も限られる)。**
pseudo-labelingの前提だった「2018-2022偏重が汎化性能を損なっている」
という仮説も、この切り口では積極的な支持は得られなかった。OOF-Public LBギャップ
(fold2 OOF PQ 0.4483 vs V52 Public LB 0.39、Δ約0.058)の真因は未特定の
まま残る。次に調べる価値がある候補: (1) test180枚の小サンプル評価ノイズ、
(2) フィラメント密度・複雑度など年代/サイト以外の画像特性差、
(3) train/testのアノテーション基準・アノテータの違い。詳細: README §2.1.6、
`distribution_shift_audit_manifest.json`。

### フェーズ4: V50方式(アンサンブル簡略化)への提出切り替え — ✅完了(V52として実現)

既に確認済みの通り、fold2単体モデルはV35の3-foldアンサンブルと同等以上の
Public LBスコアを、計算コスト1/3で達成する。V52はこの簡略化構成に
EdgeAttNet refinerを組み合わせたものであり、フェーズ2・4は同時に完了した。
**提出正本はV35→V52へ更新済み(Public LB 0.39)。**

## 4. 実行順序の推奨

計算資源を気にしない前提だが、**フェーズ1(定性30%対応)はGPU不要で
即着手でき、かつ評価への影響が確実**なため、フェーズ2のGPU実験と完全に
並行して進めるべき。フェーズ2(EdgeAttNet)はfold0パイロット→独立
2fold確認という、本プロジェクトが繰り返し確立してきた検証規律を維持する。

## 5. まだ調査できていないこと(次に調べる価値があるもの)

- 上位7チーム個別の手法は非公開(Kaggle Code/Discussionタブに彼ら自身の
  投稿は確認できなかった)。Discussionタブへの再訪・新規投稿の監視は
  継続する価値がある
- ~~EdgeAttNetの事前学習済み重みが実際に配布されているか~~ →
  2026-08-29確認済み: 配布されていない(§1.4追記参照)。自前で再学習する
- 自己評価notebook(`azimahmadzadeh/self-evaluation-notebook`)の詳細な
  中身は未精読。公式PQ実装との差分が無いか再確認する価値がある
- OOF-Public LBギャップ(Δ約0.058)の真因: 年代・サイト分布差は有意差
  未検出(フェーズ3、反証ではない)。test180枚の評価ノイズ、フィラメント
  密度/複雑度差、アノテーション基準差、いずれも未検証

## 6. 報告書査読(他AI、2026-08-30)を受けた再評価と計画への影響

技術報告書(`report/main.tex`)を別のAIに査読してもらい、7点の指摘を
全て実コード・原論文で裏付けを取った上で反映した(詳細: README §2.1.7、
memory)。**過去の実験の採否判定(WBF・free variant・pseudo-labeling・
V46/V47・検出器強化6本・EdgeAttNet)はいずれもbaseline PQとの差分だけで
判断しており、アノテータ一致ベースラインとは無関係だったため、
この訂正によって覆るものはない。** 影響があるのは主に「文脈づけの
語り口」と、新たに浮上した2つの検証課題。

### 6.1 アノテータ一致ベースラインの訂正がもたらした新課題

`run_annotator_ceiling.py`の自己比較混入バグを修正した結果、
「アノテータ不一致が課す天井」はPQ 0.7587ではなく**0.3398**だった
(README §2.1.7)。この訂正により、**我々のOOF PQ(0.4426〜0.445)が
訂正後ベースラインを上回っている**という、これまで想定していなかった
状況が判明した。「モデルが人間より正確」ではなく、複数アノテータの
学習データを見たモデルがコンセンサス的境界に収束している可能性、
という仮説を提示したが未検証(`report/main.tex` §3.2)。**これ自体が
定性30%評価・論文化の両方にとって独立した見どころになりうる**——
安易に「我々のモデルは人間を超えた」と喧伝せず、慎重に検証してから
扱うこと。

### 6.2 EdgeAttNet refinerの効果を分離する追加実験(未着手、優先度: 中)

査読で指摘された通り、現行のEG-MHSA実装(1ブロック・8head・+4.2M
パラメータ)は原論文(2ブロック・4head・-8.4Mパラメータ)とは別物であり、
+0.0028の改善がedge prior由来かattention容量増加由来か未分離。
`plain U-Net`(既存)・`U-Net+MHSA(edgeなし)`(未学習)・`現行EdgeAttNet風`
(既存)の3点比較で切り分け可能。fold0のみでのパイロットも、本プロジェクトの
規律(fold0単独では確定しない)に従えば3-fold実行が必要——GPU予算的には
EdgeAttNet導入時と同等(fold毎14epoch×3)。**優勝可能性への直接的寄与は
限定的(既にEdgeAttNet refinerは採用・提出済みで、この実験は「なぜ効くか」
の説明を精緻化するだけで新たなスコア向上は生まないため、定性30%
(パイプライン説明の厳密さ)への投資として位置づける。**

### 6.3 閾値再チューニング実験(2026-08-30実施、確認的null結果)

EdgeAttNet refinerのshrink閾値をfold0で0.25〜0.70の範囲でスイープした
(確率マップをキャッシュし閾値だけ振り直す方式、GPU再学習不要)。
結果は0.45(現行)〜0.55でほぼ平坦(0.4455前後)、最良0.50でもΔ+0.0004と
現行比で無視できる差。標準refinerで既知だった「0.45〜0.65はほぼフラット」
という性質がEdgeAttNet refinerでも再現しただけであり、**新たな改善は
見つからなかった。現行閾値0.45を維持する。** fold1/fold2への追試・
ブートストラップ検証は行わない(効果量が小さすぎて選択バイアスの
リスクに見合わない)。

### 6.4 2回目の査読(2026-08-30、氏名確定後)への対応

ユーザーがShingo Hara(ER Systems, Ltd.、神奈川県、日本)の情報を提供し、
別AIによる2回目の査読(6.8/10、Major Revision→提出直前水準)を受けた。
指摘のうち以下はその場で修正・検証済み:

- **著者placeholder・running header衝突(最優先ブロッカー)**: 氏名・所属を
  記入、`\title[短縮形]{正式タイトル}`+`\shortauthors`でrunning header
  衝突を解消。tectonic再コンパイルで視覚確認済み(査読が指摘した3ページ目の
  文字重なりは完全に消えた)
- **学習条件の欠落**: loss内訳・optimizer・LR・scheduler・epoch数・
  batch size・crop設定・正例生成法・augmentation・seed・checkpoint選択則
  ・detector側の全パラメータを本文に追記
  ("Boundary Refinement"/"Detection"節を大幅拡充)
- **0.4454の監査可能性**: `output/v51/`・`v52/`・`v53/`・
  `distribution_shift/`に、このセッションの記録から書き起こした
  manifest JSONを新規保存(Drive原本との機械照合は未実施と明記)。
  査読が求めた「per-image TP/FP/FN」までは未到達で、Future Workとして
  明記
- **公式ルーブリック(SQ/RQ/TP/FP/FN)不足**: `output/v35/pipeline_stack.json`
  に実在した生検出器・plain refiner+resolutionの正確なSQ/RQ/TP/FP/FNを
  Table 1へ追加。edge-attention行は「保存していない」と正直に開示
- **Abstract の因果過剰主張**: "same-parameter-budget design"という
  誤解を招く表現を削除
- **PQの"greedily matched"という誤記**: 公式実装(閾値越えを全部カウント、
  一対一割当てではない)に合わせて訂正、self-evaluation notebookを
  Kirillovと並べて引用
- **RF-DETR比較の"matched settings"表現**: "matched data split,
  preprocessing, and evaluator"に訂正(ハイパラは異なる旨を明記)
- **"one third of inference cost"の過剰一般化**: detector forward pass
  のみに限定する注記を追加
- **統計的検出力の甘さ**: 「3fold全部同符号」だけでは対称帰無仮説下で
  片側12.5%の偶然が起こりうる旨を明記、本セッションで試したアイデアが
  約12件あり多重比較補正をしていないことも開示
- **best-OOF-fold選択・OOF反復利用による楽観バイアス**: Weaknessesに
  明示的な限界として追加
- **軽微な事実誤り**: README「複数アノテータ観測236枚」→296枚に訂正
  (296が正)、MAGFiLO引用を@INPROCEEDINGS→@ARTICLEに、EdgeAttNetの
  arXiv primary categoryをastro-ph.SR→cs.CVに訂正(いずれも実際に
  Web上で確認して修正)、Ultralytics引用のversion表記を実態(8.3.x
  パッケージ、"11.0.0"という別リリースは存在しない)に合わせて訂正

**2026-08-31、上記の残タスクも解消:** Colab MCPブリッジがセッション中に
完全切断されたため、対話実行から「1本のnotebookを組んでユーザーが
Colabで実行し、結果を貼り戻してもらう」方式に切り替えた
(`scratch/build_qualitative_analysis.py`
→`solar_filament_qualitative_analysis_colab.ipynb`)。実行結果を反映:

- **fold0 OOF全236画像で両refinerの詳細監査を実測**(標準refiner:
  PQ=0.439288/SQ=0.673/RQ=0.652/TP=1713/FP=789/FN=1036、EdgeAttNet
  refiner: PQ=0.445492/SQ=0.680/RQ=0.655/TP=1717/FP=775/FN=1032、
  いずれも既知値と完全一致) — `output/v51/v51_fold0_detailed_audit_manifest.json`
- **IoU/Dice分布(中央値・四分位)を実測**: IoU中央値0.678→0.688、
  Dice中央値0.808→0.816(標準→EdgeAttNet)
- **one-to-many/many-to-oneは両方とも0**(排他化ステップが予測を
  強制的に非重複にしているため、という構造的理由込みで報告書に記載)
- **改善例+失敗例の4パネル(GT/raw/plain/edge)比較図を生成**
  (`report/figures/qualitative_comparison.png`)。失敗例は当初「完全な
  検出漏れ(IoU=0)」を選んでいたが、査読の意図(split/merge等の
  境界誤差を見せる)により合う「GT3個体中2個体がマージされ1個体は
  完全に見逃されている」ケースに差し替えた

report/main.texへ全て反映: Figure 2を新しい4パネル比較図に差し替え、
Table 1にedge-attention行の実測TP/FP/FN/SQ/RQを追加(fold0限定と明記)、
Future Workの記述を更新。tectonicで再コンパイル・視覚確認済み(5ページ、
本文4ページ+Ack/References)。

**Google Drive接続のトラブルシューティング(記録):** ローカルの`G:`
ドライブ(Google Driveデスクトップアプリのマウント)が今回のセッション
中に切断され、Colabが書き込んだ新しいファイルを取得できなくなった。
Googleドライブアプリを再起動してもらったら即座に復旧した。今後同様の
症状が出たら、まずアプリの再起動を試すこと。

**残るのはrequirements.txtの完全なpip freezeのみ**(既存ファイル自身が
この欠落を明記済み、実行環境を確保してから対応)。

### 6.6 3回目の査読(2026-08-31)への対応

2回目査読後のPDFをさらに査読してもらい、9点の指摘を受けた。全て実
コード・原論文で裏付けを取った上で対応(1点を除き査読は正確だった):

- **ページ数(5→4への圧縮要求): 部分的に反証。** Overleaf公式テンプレート
  自体に"Please limit your report to four (4) pages, **excluding the
  Acknowledgment and References sections**"と明記されており(このセッション
  冒頭でテンプレート本文を直接確認済み)、本文はAcknowledgment開始前
  (4ページ目途中)で終わっているため規定に抵触していない。査読へその旨を
  返信済み。ただしEdgeAttNetリポジトリの説明・Future Workは他の理由
  (冗長性)でも軽量化した
- **YOLO学習条件「defaults」という記述: 事実誤認と確認、修正。**
  `build_v30_prize_colab.py`を確認したところ、AdamW/lr0=7e-4/lrf=0.05/
  weight_decay=5e-4/warmup_epochs=3/degrees=180/translate=0.04/scale=0.25/
  fliplr・flipud=0.5/hsv_v=0.18/close_mosaic=10等を明示的に設定しており、
  Ultralytics既定値ではなかった。report/main.texを修正し、
  **`scratch/build_master_pipeline_notebook.py`(統合notebook)自体にも
  同じ欠落があった実バグを発見・修正**(このnotebookをそのまま実行すると
  別ハイパラで再学習してしまうところだった)
- **near-miss「dominant error mode」という表現: 過大主張と確認、修正。**
  `output/v35/near_miss_modes.json`を確認したところ、near-miss(955件)より
  hard FN(2138件)・hard FP(1561件)の方が多く、「dominant」ではなく
  「a major, actionable error mode」に訂正
- **Table 1でPQ(707画像)とSQ/RQ(fold0の236画像)を同じ行に混在: 構造上の
  問題と認め、修正。** edge-attention行のSQ/RQをダッシュに戻し、fold0限定の
  詳細内訳を本文の別段落("Fold-0 detailed breakdown")へ分離
- **one-to-many/many-to-one=0の解釈: 妥当な批判、限界を追記。**
  IoU>0.5閾値後のhitsを数えているため、排他化された予測集合では構造的に
  ほぼ0になる。Figure 2(b)が示すmerge型エラーは各GTとのIoUが0.5未満に
  なるため純粋なFN×2として数えられ、「many-to-one」としては捕捉されない。
  raw-overlap(閾値なし)ベースの二部グラフ集計が必要だが未実施、と明記
- **IoU/Dice分布がTP限定: 妥当、明記追加。** unmatched予測・見逃しGTを
  除外しているため「正しく検出できた場合の境界品質」であり、instance
  レベル全体の分布ではないと明示
- **「already published」という記述とGit未追跡の不一致: 妥当、表現softening。**
  「公開済み」から「今後コミットする」(Future Work)という表現に変更
- **V51 manifestの「reconstructed from transcript」という自己開示との
  不整合(0.4454を"directly measured"と呼んでいた): 妥当、修正。**
  「reconstructed from the recorded per-fold bootstrap outputs」に変更
- **MAGFiLO書誌情報(1031=ページ数、Dec. 2024): 事実誤認と確認、修正。**
  Crossref APIで確認したところ1031はarticle number(ページ数ではない)、
  公開日は2024年9月27日(12月ではない)。main.bibを修正
  (@ARTICLE、journal={Scientific Data}、month=sep)

tectonicで再コンパイル・視覚確認済み。

### 6.7 4回目の査読(2026-08-31)への対応 — ページ数を実際に4ページへ圧縮

4回目査読(総合8.3/10、判定「4ページ化とGit反映が終われば提出可能」)を
受けた。今回は指摘の核心(ページ数)を裏付けから取り直した:

- **公式競技ページ自体を実ブラウザで確認**(`kaggle.com/competitions/
  filament-segmentation-2026/overview` の「Final Submission」節):
  "A 4-page technical report is required... More details about the
  content of this report are provided in the template." — ページ数の
  詳細規定はテンプレート委任であり、§6.6で確認済みの「Acknowledgment・
  References除く4ページ」という同テンプレートの文言と矛盾しない。
- **しかし実際に測ってみると、本文(Future Work末尾まで)がpage5の
  1カラム目55%程度まで溢れており、テンプレートの寛容な解釈でも
  4ページに収まっていなかった。** これは前回査読時の「5ページ全体は
  違反ではない」という反論だけでは不十分だったということ。
  Abstract/Strengths重複、3.1と3.3の多重比較キャベット重複、
  distribution-shift段落の冗長性、EdgeAttNet著者コード説明、
  Future Work、Table 1 caption、CV節等、20箇所以上を圧縮
  (内容は一切削らず、重複除去と言い回しの短縮のみ)した結果、
  **本文が正味4ページに完全に収まり、page5はAcknowledgment+
  Referencesのみになった(tectonicで確認、overfull hbox警告ゼロ)。**
- **CI記述の過大主張: 妥当、修正。** §3.1で「every accepted or rejected
  changeにCIを付与」と書いていたが、実際に個別CIを示しているのは
  採用されたedge-attention refinerのみ(却下案のCIは記載していない)。
  「採用された変更のみpooled intervalを報告し、却下案はSection 3.3で
  個別区間なしに要約する」という正確な表現に修正
- **Abstract「Every design decision was evaluated」: 妥当、修正。**
  「Major design decisions」へ弱め、Strengthsとの重複列挙も削除し
  Section参照に置き換え
- **Git未反映(`report/`、`output/v51-v53`、6本のV51-53系notebook):
  査読の指摘通り、実際に全て`git status`で `??`(未追跡)と確認。**
  本文Future Workは既に「今後コミットする」という正確な表現になって
  いる(§6.6で対応済み)。**このセッションのルールにより、ユーザーの
  明示的な指示なしにgit commitは行わない** — 提出前にユーザー自身が
  commit判断をする必要がある、と明確に伝える
- **requirements.txtの`>=`指定: 査読の指摘は妥当、かつ公式ルール文言でも
  再確認。** 実ブラウザで確認した公式ページの文言は
  "a file named requirements.txt must include all utilized packages
  and their corresponding versions" — "corresponding versions"は
  `>=`範囲ではなく固定バージョンを要求していると読める。ただし
  正式runの実環境からの`pip freeze`がなければ正確な値を書けないため、
  **捏造せず、既存の自己開示コメント(このファイルの§17.6/§34相当)を
  維持し、ユーザーに実環境からの`pip freeze`取得を促す**にとどめた

tectonicで再コンパイル(10ラウンド以上の圧縮反復)、overfull hbox警告
ゼロ、page1-5全て視覚確認済み。**本文が公式テンプレート規定の4ページに
文字通り収まったのは今回が初めて。**

### 6.8 5回目の査読(2026-08-31)への対応

5回目査読(「条件付きMinor Revision」、ページ問題は解消と確認)を受け、
残り2点を修正:

- **採否基準のロジック矛盾: 妥当、修正。** §3.1が「(a)全fold符号一致
  OR (b)pooled CIがゼロ除外」と書いていたのに、直後で「(a)は
  screening heuristicでありsignificance testではない」と弱めていた
  ——弱い条件だけで採用できると読める論理矛盾。**(a) AND (b)**に
  変更(採用済みのedge-attention refinerは実際に両方満たしているため、
  結果への影響なし)
- **"five plausible improvements"なのに4項目しか列挙されていない:
  妥当、修正。** Strengthsの列挙にweighted mask fusion/threshold
  retuning/refiner retraining/pseudo-labelingの4つしかなかった。
  実際に不採用となった5つ目の施策(3-fold detector ensembling、
  §3.2で「public scoreが0.39→0.38に悪化」と既述)を追加し、
  「5件」の内訳を正確にした
- **査読が触れていない実バグを追加発見・修正: `\pm180°`・`$90°$`の
  リテラル度記号(Unicode U+00B0)がOT1エンコード数式フォントで
  `Missing character`警告(3passとも再現、最終passのPDFにも影響)を
  出していた——実際にPDF上で文字が欠落しうる状態。`^\circ`
  (数式モードの標準的な度記号)に置換して解消。** これは全4回の
  査読で誰も指摘していなかった問題で、ログを`grep overfull`だけで
  なく`grep "Missing character"`でも確認する習慣がなかったことが
  発見の遅れた原因。以後、tectonicコンパイル後は両方を必ず確認する
  こと
- Further limitations段落の多重比較キャベットをSection 3.2への
  相互参照のみに整理(査読のoptional提案を反映)

tectonicで再コンパイル、`Missing character`警告ゼロ・overfull hbox
(15pt超)ゼロ・page1-4に本文収納・page5はAck+Referencesのみを確認。

### 6.9 【重大な撤回】matched-parameter ablationでEdgeAttNet効果を再現できず(2026-08-31)

5回目査読が指摘したFuture Work item 1(edge priorの効果とattention容量増加の効果を
分離するablation)を、ユーザーの「積極的にcolabを用いて実施」という指示を受けて実施。
colab-mcpのライブ接続は切断されたままのため、notebook-builder-script(`scratch/
build_matched_param_ablation.py`) + ユーザーがColabで実行 + 結果貼り戻しの方式で対応。

新規モデルクラス`MediSegPlainMHSAUNet`(標準MHSA・edge priorなし、35.238M params、
`MediSegEdgeAttUNet`との差はわずか2,076個=0.006%)を追加し、base(31.04M)/plain_mhsa
(35.24M)/edge_att(35.24M)の3構成×seed(3407,8407)=6本をfold0で学習・OOF評価した。

**結果(fold0 OOF PQ、2seed平均): base 0.441412 > plain_mhsa 0.441259 > edge_att 0.440474。**
edge-attentionは3構成中最低で、旧報告の+0.0062という改善を再現できなかった。旧報告の
3-fold delta(fold0 +0.0062、fold1 +0.0009、fold2 +0.0012)のうち後者2つは、今回実測した
seed間ノイズ(std 6-8e-4)より小さく、「3fold全て符号一致」という採否基準自体が学習ノイズ
で説明できる水準だった可能性が高い。

**ユーザーは「report/main.texを正直に書き換える」を選択(AskUserQuestionで4択提示、
追加seed確認や差し戻し検討ではなくこの選択肢)。** Abstract・Table 1・§3.2 Results・
§3.3 Strengths/Weaknesses/Future workを全面改訂し、「唯一検証を生き残った改善」という
主張を撤回して、この反転自体を主要な知見として報告する構成に変更。tectonicで再コンパイル・
overfull hbox解消・page1-4に本文収納を再確認済み。詳細は
README.md §2.1.9参照。

**未確定のまま残ること:** ablation自体n=2・fold0限定なので断定はできない。提出モデル
(現在Public 0.39のedge-attention refiner)をplain refinerへ差し戻すかは未決定。

### 6.10 6回目の査読(「条件付きMajor Revision」)への対応(2026-08-31)

撤回の方向自体は妥当と認めつつ、書き方が「確定」しすぎていると指摘。
主な修正: (1) 「fold1/2のdeltaはseed間stdより小さい」という算術誤りを
発見・修正(実際は大きい、「same order as」に訂正、std表記も生の2値
表記へ変更)、(2) paired差分の再計算でedge_att-plain_mhsaの符号が
seed間で一致しないと判明、「attention hurts」ではなく「証拠がない」
という弱い主張に精密化、(3) 「two fresh seeds」「protocol caught its
own false positive」という不正確な表現を修正、(4) notebook実行証跡
(execution count 0)・manifestのチャット転記という監査可能性の欠如を
本文に明記、(5) Figure 2キャプションに「旧単一seed runのもの、平均では
再現せず」の注記を追加。詳細はREADME.md §2.1.9
参照。20箇所以上の再トリムを経てtectonicで本文4ページに再収納。

### 6.11 【完了・確定】Gドライブ直接検証+3fold matched-parameter ablation完走(2026-08-31)

6回目査読の懸念点4(監査証跡の欠如)への対応中、ユーザーから「Gドライブは
実際にマウントされている」との指摘を受け、`G:\マイドライブ`へ直接アクセス
できることが判明。以下を直接検証:

- fold0の生JSON(`matched_param_ablation_fold0_manifest.json`)と実行済み
  notebook(`Colab Notebooks/solar_filament_matched_param_ablation_colab
  (1).ipynb`、execution_count 1-8・cell7に181出力)を読み、**チャット転記値と
  完全一致**を確認。査読の監査証跡懸念を解消
- 同じ手順でfold1・fold2の生JSON・実行済みnotebookも取得・検証。全3fold分を
  `output/ablation/`に保存(生JSON・実行済みnotebook・統合解析
  `matched_param_ablation_all_folds_summary.json`)

**fold1・fold2の実測(ユーザーがColabで完走):**

| fold | base | plain_mhsa | edge_att |
|---|---:|---:|---:|
| fold0 | 0.4414 | 0.4413 | 0.4405 |
| fold1 | 0.4405 | 0.4426 | 0.4417 |
| fold2 | 0.4434 | 0.4458 | 0.4464 |

report自身のfold-sign-agreement基準を適用: attention容量(edge priorの
有無を問わず)は3fold中2fold(fold1・fold2)で両seed符号一致の正、fold0は
ほぼゼロ——概ね効いている。**しかしedge_att − plain_mhsaは3fold中2fold
(fold0・fold1)でゼロ以下、fold2はseed間で符号不一致——どのfoldもedge
priorがplain attentionを明確に上回らない。**

**最終結論: 「偽陽性だった」から「効果はattention容量一般に起因し、
edge prior固有の主張は支持されない」という、より精密な着地点に到達。**
3fold全体のmicro-averaged PQ: base 0.4418 < edge_att 0.4429 <
plain_mhsa 0.4432(plain_mhsaが最高)。

**report/main.texを3度目の全面改訂:** Abstract(反転ではなく精密化と
再定義)・Table 1(fold0限定行を3fold全体行に置換)・§3.2(fold-sign-
agreement分析を含む新パラグラフ)・§3.3 Strengths/Weaknesses(「protocol
自体が捕まえたのではなく、後付けの多fold検証が正しいmechanism特定を
可能にした」という枠組みへ)・Future work(「plain-attention refinerを
再提出候補として検討」という新しい実務提言を追加)を改訂。tectonicで
再コンパイル・overfull hbox解消・page1-4本文収納を再確認。

詳細はREADME.md §2.1.10参照。

### 6.12 【重要】リーダーボードリーク判明・追走中止、競合コード調査に基づく新優先順位(2026-09-15)

Kaggle公式ディスカッションフォーラムを実ブラウザで調査した結果、Public LB
0.55〜0.56に38チームが並ぶ異常なクラスタが、テストセット180枚中77.2%が
公開MAGFiLO 1.0(正解アノテーション付き)とバイト単位で完全一致するという
検証済みのデータリーク(`discussion/728062`、JAB100氏の投稿、独立に再現
確認済み)に由来すると判明した。運営も複数スレッドで「リーダーボードでは
なく定量70%+定性30%の合議制で評価する」「0.35超で十分価値がある」と
明言しており、MAGFiLO本家の学習利用は規約違反でもある。**結論: 0.55の壁は
追走しない。詳細はREADME.md §2.1.11参照。**

この調査と並行して、GitHub上の公開ソリューション(誠実なティアの技術動向)
を調査し、自チームが一度も試していない改善レバーを特定した:

| リポジトリ | 手法 |
|---|---|
| `gopi470/VisionX` | YOLO11+CLAHE連結成分(recall強化)→4chクロップ(raw+CLAHE+binary seed+距離変換)→smp.Unet refiner(BCE+Dice+**boundary loss**、**EMA/SWA**、hard-negative mining)→6-passTTA+**スケルトンベースのギャップ閉合** |
| `srivatsav-kannan/solar-seg` | コンパクトU-Net単体、Public LB 0.31(「誠実な」下限の参考点) |

**【2026-09-15、実装着手直前に訂正】以下の優先順位1(boundary loss)は
誤りだった。** 実装に入る前に`scratch/build_v51_edgeattnet_refiner.py`等の
実コードを再確認したところ、`refiner_loss`関数が既にregion loss(BCE+Dice)
とboundary loss(境界マップのL1損失、重み0.15)の合成になっており、
**V30から一言一句変わらず全refiner学習(matched-parameter ablation含む)で
使われ続けていた。** 前回の分析は`mediseg_models.py`の`CombinedLoss`
(実際には未使用の旧クラス)だけを見て「境界損失なし」と誤判定していた。
詳細はREADME.md §2.1.12。優先順位を以下の通り訂正する。

**新しい優先順位(既に複数fold検証で否定済みのレバー——検出器解像度/
augmentation/モデルサイズ強化、マスクレベルWBF/multi-seedアンサンブル、
RF-DETRとのクロスアーキテクチャアンサンブル、refiner再学習V46/V47、
pseudo-labeling、edge prior固有効果、TTA、watershed、~~boundary loss~~
(既に実装済み)は再挑戦しない):**

1. **EMA/SWA(重み平均)の追加(優先度: 最高、低コスト)** — 実コード確認済みで
   genuinely未試行。refiner学習に安価に追加でき、経験的に無コストに近い
   改善が出ることが多い。
2. **refiner入力のマルチチャンネル化(優先度: 高)** — 実コード確認済みで
   genuinely未試行(現状`InstanceCropDataset`は3ch色画像cropのみ)。距離変換
   マップ・binary seedチャンネルの追加。VisionXの設計そのもの。
3. **スケルトンベースのギャップ閉合(優先度: 中)** — 実コード確認済みで
   genuinely未試行。watershed(過剰結合の分離)とは逆方向。fragmentationした
   細い構造の再接続に特化した後処理。
4. **(副次・優先度低)既存boundary lossの定式化を強化** — 「初導入」では
   なく「現行の粗いmorphological-gradient定式化(重み0.15固定)を、距離変換
   ベースのより精密なboundary loss(例: Kervadecらのdistance-transform
   boundary loss)へ置き換える」という改良軸として位置づけ直す。
5. **PQのIoU>0.5断崖を意識した閾値較正(優先度: 中)** — 現行の
   drop_contained/refiner閾値探索はすべて経験的グリッドサーチ。
6. **密集シーンへの学習サンプリング重み付け(優先度: 中)** — 診断済みの
   主要失敗モード(密集シーンでの検出漏れ)への、解像度/augmentation/
   モデルサイズ強化とは異なる軸。
7. **SAM2をpromptable refinerとして使う(優先度: 低・大きな賭け)** —
   YOLO検出をbox/pointプロンプト化。過去のアーキテクチャ差し替え
   (RF-DETR・EdgeAttNet)はいずれも失敗しているため期待値は不確実。

2026-09-15、ユーザーの指示によりドキュメント更新後、上記1〜3(EMA/SWA→
refinerマルチチャンネル化→スケルトンギャップ閉合)を優先度順に着手する。

### 6.5 優先順位の再整理(2026-08-30時点)

1. **primer.html・learn-architecture.html等の公開Artifactの訂正**
   (0.7587→0.3398、"ceiling"→"agreement baseline"表現含む) — 誤った
   数値を公開したままにしない、最優先の残タスク
2. フェーズ1残り(定性30%、形態学的妥当性の追加図) — 未着手のまま
3. §6.2のmatched-parameter ablation — 定性30%(説明の厳密さ)への投資
4. §5に挙げた未調査項目(OOF-Publicギャップの真因、自己評価notebook精読) — 
   低優先度のまま
5. 閾値再チューニング(§6.3) — 完了・現状維持で決着、これ以上追わない

### 6.13 §6.12優先度1〜3のパイロット準備完了、Colab実行待ち(2026-09-15)

fold0パイロット3本のnotebook・実装をローカルで検証済み(構文チェック・
ユニットテスト・CPUスモークテスト)、Colab実行のみ残っている状態。

| 優先度 | 内容 | builder script | notebook | ローカル検証 |
|---|---|---|---|---|
| 1 | EMA(重み平均) | `build_v54_colab_ema_pilot.py` | `solar_filament_v54_ema_pilot_fold0_colab.ipynb` | ast全セル・EMA+refiner_lossのCPUスモークテスト |
| 2 | refinerマルチチャンネル化 | `build_v55_colab_multichannel_refiner_pilot.py` | `solar_filament_v55_multichannel_refiner_pilot_fold0_colab.ipynb` | ast全セル・3ch/5ch両方のCPUスモークテスト(学習・推論) |
| 3 | スケルトンギャップ閉合 | `build_v56_colab_gap_closure_probe.py`(--fold 0/1/2対応) | `solar_filament_v56_gap_closure_probe_fold{0,1,2}_colab.ipynb` | `tests/test_gap_close.py`7件全パス(202件中) |

優先度3は学習不要(既存の標準refinerチェックポイントを再利用した推論のみ)
のため、fold0〜2を1回のColabセッションで通しても10〜20分×3程度で済む見込み。
優先度1・2はfold0のみ先に実行し(各1〜1.5時間)、有望な結果が出た場合のみ
fold1・fold2へ展開する(単一foldのbestを一般化しない、という本プロジェクト
一貫の規律に従う)。

### 6.14 優先度3(スケルトンギャップ閉合)fold0・fold1実行結果: 不採用(2026-09-27)

Colab実行完了(fold0: 236画像を18.5分、fold1: 236画像を19.5分、いずれも推論のみ)。

**fragmentation診断**: proposalの13〜15%が複数連結成分に分断していた
(fold0: 1722 proposal中13.3%、最大9成分。fold1: 1673 proposal中15.1%、最大8成分)。
分断は確かに存在する。

**max_gapスイープ結果(OOF PQ)**:

| fold | baseline(§6.11実測) | 0.0(現行再現) | 4.0 | 8.0 | 16.0 | sweep内最大Δ |
|---|---|---|---|---|---|---|
| 0 | 0.4393 | 0.4357 | 0.4357 | 0.4357 | 0.4361 | +0.0004 |
| 1 | 0.4415 | 0.4401 | 0.4401 | 0.4401 | 0.4401 | ≈0.0000 |

- **ギャップ閉合の効果は実質ゼロ。** 分断が13〜15%あるにもかかわらず、
  `max_gap` を16pxまで広げてもPQは±0.0004以内。分断proposalの多くは
  FP側の断片か、IoU≥0.5のTP判定を左右するほどの画素変化を生じないと解釈。
  fn≈1000が依然として最大のロス源であり、fragmentationは主失敗ではなかった。
- 独立2foldで効果ゼロが一致したため、**優先度3は不採用で確定**。
  fold2は結果追記の余地を残すが、採否を変える必要性はない。
- **再現ギャップの記録**: `max_gap=0.0`(現行のまま)でもbaselineより
  fold0 −0.0036 / fold1 −0.0014と系統的に低い。probe側の再実装
  (推論〜採点パス)とmatched-parameter ablation本筋の間に小さな差分
  (非決定性を含む)があると考えられる。同一probe内のsweep比較には影響しないが、
  baselineとの絶対値比較には注意。

**notebook保守の教訓**: Colab実行中に2件のバグを修正済み
(1: cell2の`shutil`未import。2: `from __future__ import annotations`を含む
gap_close.pyをセルにverbatim埋め込むと、対話実行では`annotations`名が
`_Feature`オブジェクトに上書きされる — 今後.pyをセル埋め込みする際は
同名のモジュールレベル変数との衝突に注意)。修正は`annotations`→
`annotations_by_image`のリネームで対応、モジュール本体は不変。

残る改善レバーは優先度1(EMA, v54)と優先度2(マルチチャンネル化, v55)。
fold0パイロット(各1〜1.5時間)の実行結果待ち。

### 6.15 優先度1・2のfold0パイロット実行結果(2026-09-27)

Colab実行完了(各4本学習、fold0のみ)。

| pilot | 構成 | seed別PQ | 2seed平均 | vs参照値(0.4414) | 判定 |
|---|---|---|---|---|---|
| v55 multichannel | baseline(3ch) | 0.4420 / 0.4408 | 0.4414 | ±0.0000 | — |
| | multichannel(5ch=色+seed mask+距離変換) | 0.4218 / 0.4198 | 0.4208 | **−0.0206** | **不採用確定** |
| v54 EMA | baseline | 0.4413 / 0.4418 | 0.4416 | +0.0002 | — |
| | ema(decay=0.999) | 0.4421 / 0.4420 | 0.4421 | **+0.0005** | 要fold1/2確認 |

- **v55(マルチチャンネル化)は明確な悪化で不採用。** Δ−0.0206はseed間変動
  (σ≈0.001)を桁で上回り、両seedで符号一致。seedマスク+距離変換の追加chは
  refiner学習を阻害した(train/inferenceのseed生成差異が原因の可能性、
  V46の設計判断参照)。fold1/2への展開は行わない。
  **【手法上の注記(2026-10-02、PR #181 の Codex review 指摘)】** v55 builderは
  検証Datasetでも`augment=False`のまま共有RNG(`_seed_rng`)から`perturb_seed`を
  呼ぶため、5ch構成ではepochごとに検証入力(seedマスク・距離変換)が変わり、
  `crop_dice`によるbest epoch選択にノイズが入っていた。影響は5ch構成の
  checkpoint選択のみ(3ch baselineは先頭3chのみ使用で影響なし、最終の
  OOF PQ評価はYOLO proposalを入力とする決定的な経路)。Δ−0.0206はこの選択
  ノイズで説明できる幅を大きく超えるため不採用の判断は維持するが、将来
  多チャンネル入力を再検討する場合は検証proposalを固定(事前計算または
  index由来の決定的RNG)すること。記録との整合のため builder 自体は変更しない。
- **v54(EMA)は微小ながら正。** Δ+0.0005、2/2seedで正の一致。ただしプロジェクト
  一貫の「意味ある改善」の目安+0.005を大きく下回る。EMA本来の期待効果である
  seed間分散の縮小は観測された(pstdev 0.00023→0.00002)。fold1・fold2の
  notebookを生成済み(`--fold 1/2`対応をbuilderに追加)——符号一致が独立
  2foldで確認できれば採用検討、ただし効果量が実務閾値に遠く及ばないため
  優先度は低い。
  **(2026-10-02追記)** fold1・fold2を1回のColabセッションで直列に回す統合版
  `solar_filament_v54_ema_pilot_fold1_2_colab.ipynb`を追加(`--folds 1 2`)。
  中身は単独fold版と同一で、途中再開(Driveのpartial JSONから完了済みrunを
  飛ばす)・ピークVRAM/所要時間のmanifest記録・fold0を含む全foldの採否判定
  セルだけを足した。見込み3〜3.5時間。
