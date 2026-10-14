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

### 6.16 優先度6(密集シーンへの学習サンプリング重み付け)fold0パイロット実行・不採用(2026-10-05)

§6.12の残レバーのうち優先度6を実装。診断済みの最大ロス源は密集シーン
(gt_count_mean 10-21)での検出漏れ(hard FN 2138件、hard FP 1561件を37%
上回る、2026-08-16診断)であり、これはproposal自体が存在しないGTへの
失敗なのでrefinerではなく**検出器(YOLO)側のレバー**として実装した。

| 項目 | 内容 |
|---|---|
| builder | `scratch/build_v57_colab_dense_oversample_pilot.py`(--fold 0/1/2) |
| notebook | `solar_filament_v57_dense_oversample_fold{N}_colab.ipynb` |
| ヘルパー | `solar_filament/pipeline/dense_oversample.py`(`dense_cut_value`/`dense_replication_counts`)、`tests/test_dense_oversample.py` 8件パス |
| 設計 | V30のYOLO学習と完全同一のレシピで、`baseline` vs `dense2x`(trainサンプルの非crowd GT数がfold内75分位点以上なら複製して出現頻度2倍) × seed(3407,8407) = 4本 |
| 評価 | raw PQ(生YOLO proposals→ownership)と refined PQ(既存標準refinerを使い回した確定構成)の2段 + dense/sparseスライス別FN |
| 工数 | YOLO 70epoch×imgsz1280を4本直列 — A100で数時間〜半日規模。manifestをDriveへ逐次追記するので中断再開可 |
| ローカル検証 | builder全コードセルのastパース通過(fold0生成確認)、ヘルパーの単体テスト8件 |

採否はプロジェクト規律通り: 2seedで符号一致すればfold1・2へ独立確認を広げる。
denseスライスのFNが減らない場合はメカニズム不成立として不採用。

**fold0実行結果(2026-10-05、Colabで全4本完走・manifest
`artifacts/v57_dense_oversample_fold0_manifest.json`):**

| run | raw PQ | refined PQ | TP | FP | FN | FN(dense) | FN(sparse) |
|---|---|---|---|---|---|---|---|
| baseline s3407 | 0.3865 | 0.4259 | 1664 | 841 | 1085 | 532 | 553 |
| baseline s8407 | 0.3889 | 0.4281 | 1646 | 782 | 1103 | 553 | 550 |
| dense2x s3407 | 0.3803 | 0.4245 | 1760 | 1070 | 989 | 483 | 506 |
| dense2x s8407 | 0.3787 | 0.4197 | 1673 | 928 | 1076 | 534 | 542 |

paired Δ(dense2x − baseline): seed3407 Δraw **−0.0062** / Δrefined **−0.0014** /
ΔFN(dense) **−49**; seed8407 Δraw **−0.0102** / Δrefined **−0.0085** /
ΔFN(dense) **−19**。

**判定: 不採用(fold0パイロットで終了、fold1・2には展開しない)。**

- **メカニズムは作動したがネットで負**: dense FNは2seedとも減少(−49/−19、
  TP +96/+27)と仮説通り密集シーンの検出リコールは上がった。しかし同時に
  FPが +229/+146 と爆増し、PQの分母増(TP+0.5(FP+FN))が回収分を完全に
  上回った。「密集画像を2倍見せる→密集シーンで過剰提案する」という
  副作用がFN改善を食い潰す形。メカニズム不成立(FNが減らない)ではなく、
  FN改善のコストが効果を超えることが確定した。
- **方法論上の注意**: 今回のrun内baselineのrefined PQ(0.426-0.428)は
  歴史的fold0参照値(0.4393)より0.011-0.015低い。検出器を新規に学習
  し直すとproposal分布が変わり、同じpretrained refinerを通しても絶対値が
  ずれる(検出器seed差はrefiner学習のseed差 σ≈0.001 よりはるかに大きい
  別ノイズ源)。評価はnotebook設計通りrun内paired比較のみを信用する。
- **残る含意**: dense FNを減らす手段自体は存在するが、増えた提案の品質を
  制御しないとPQでは損になる。検出器リコールを上げるだけではこの
  パイプラインでは報われない。次に試すなら「提案品質側」のレバー
  (優先度5: IoU>0.5断崖に対応した閾値・conf較正、または低品質提案の
  除外機構)が筋。oversampling系の亜種(extra回数・閾値変更)も同じ
  FP副作用を持つため推奨しない。

### 6.17 v58・v59・v60 パイロット準備完了、Colab実行待ち(2026-10-05)

Google AI Ultra 移行で Colab 計算資源に余裕ができたため、v57 の陰性結果
(§6.16)を踏まえて系統立てて設計した3本のパイロット notebook を用意した。
いずれも `devin/solar-next-pilots` ブランチ(PRはv57ブランチにスタック)の
`Solar_Filament_Segmentation/` 直下に生成済み。

#### v58 — 提案 conf スイープ(学習なし・既存ckpt使い回し)

- builder: `scratch/build_v58_conf_sweep_probe.py`
- notebook: `solar_filament_v58_conf_sweep_fold0_colab.ipynb`
- 仮説: v57 で dense2x が残した dense FN 改善は、conf 閾値を上げれば FP 爆増を
  抑えて残せる可能性がある。また現行操作点 CONF=0.26 自体がこの評価経路で
  未掃引だった。
- 設計: 既存検出器 ckpt 5本(`yolo_fold_0` prod / v57 baseline×2seed /
  v57 dense2x×2seed)に対し、predict floor 0.03 で一度だけ推論+全提案を
  refine して RLE 保持、conf ∈ {0.05,0.08,0.12,0.16,0.20,0.26,0.32,0.40} の
  スイープはメモリ上で採点。raw/refined PQ + dense/sparse FN を各点で記録。
- **バイアス対策**: これは評価fold上の探索であり、ここで選んだ点をそのまま
  採用しない(free variant 0.86 の前科)。両config・両seedで同方向に改善する
  点のみ fold1/2 で独立確認(v57のfold1/2 ckptは未学習なので必要なら別途用意)。
- コスト: 学習なし、推論+refineのみで約1〜1.5時間。

#### v59 — 距離変換 boundary loss(§6.12優先度4)

- builder: `scratch/build_v59_boundary_dt_pilot.py`
- notebook: `solar_filament_v59_boundary_dt_fold0_colab.ipynb`
- helper: `solar_filament/pipeline/boundary_dt.py` + `tests/test_boundary_dt.py`
  (9テスト、ローカルで全パス)
- 仮説: IoU>0.5 崖の近傍で TP 化するためには境界精度が効く。現行の
  morphological-gradient L1(V30から不変の粗い定式化)を Kervadec 系の
  距離変換境界損失 `mean(sigmoid·φ_G)` に置き換える。φ_G は GT の正規化済み
  符号付き距離写像(内部負・外部正・境界0)、augment適用後のtargetから
  バッチ内で計算する。重み0.15は既存edge項からそのまま引き継ぐ単変数置換。
- 設計: v54テンプレートと同一構造。`v54_ema_pilot_fold0_manifest.json` の
  baseline行を再利用するため新規学習は dtb×2seed のみ(見込み40分)。
  refiner側の変更なので検出器seed差ノイズを持ち込まない。
- 判定: 2seedで符号一致した場合のみfold1/2へ展開。境界項の効果はSQに
  現れるはずなので、PQが動かなくてもSQの系統的改善があれば次の手を検討。

#### v60 — copy_paste augmentation(v57の代替機構)

- builder: `scratch/build_v60_copypaste_pilot.py`
- notebook: `solar_filament_v60_copypaste_fold0_colab.ipynb`
- 仮説: v57のシーン複製が持った「そのシーンで過剰提案するpriorを学ぶ」副作用を、
  インスタンス単位の貼付(ultralytics組込み `copy_paste=0.2`)なら避けられる
  可能性がある。密集の「インスタンスが多い」信号だけを増やす。
- 設計: V30学習レシピ完全同一で `copy_paste` のみ変数。baseline は v57
  manifest の行を再利用(同一レシピ・同一seed・同一評価経路)するため
  新規学習は copypaste×2seed のみ(見込み2〜3時間)。dense/sparse FN
  スライスもv57と同じcut定義で記録し、機構比較を可能にする。
- 判定: v57と同規約。FN改善があってもFP爆増でPQが下がれば不採用
  (confスイープはv58で網羅済み)。copypasteでもdense FNが減らなければ
  検出器側の密集リコール系レバーは総括して不採用方向。

**推奨実行順**: v58(最安・学習なし)→ v59(40分)→ v60(2〜3時間)。
v58 の結果は v60 の解釈(copypasteでFPが増えた場合の対処法の有無)にも効く。

### 6.18 v59 fold2先行実行結果 + v54 fold2補完 + v58中間報告(2026-10-05)

#### v59(距離変換boundary loss)fold2 結果

fold2を先行実行。baseline行はv54 manifestから再利用(同一評価経路・同一seed
のpaired)。結果:

| run | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| baseline s3407 | 0.445722 | 0.678268 | 0.657148 | 1726 | 742 | 1059 |
| dtb s3407 | 0.443924 | 0.678805 | 0.653978 | 1718 | 751 | 1067 |
| baseline s8407 | 0.444964 | 0.678945 | 0.655376 | 1722 | 748 | 1063 |
| dtb s8407 | 0.444704 | 0.679345 | 0.654608 | 1719 | 748 | 1066 |

paired Δ(dtb − baseline): s3407 PQ **−0.0018** / SQ +0.0005 / RQ −0.0032;
s8407 PQ **−0.0003** / SQ +0.0004 / RQ −0.0008。

**読み取り**: 仮説の半分が作動した — SQ(マッチしたTPの平均IoU)は2seedとも
微増し、距離変換境界損失が「マッチした予測の境界精度」を上げる効果は確認
できる。しかし RQ が2seedとも低下(TP −8/−3、FN +8/+3)し、ネットでは
両seedで負。効果量は ±0.001 程度のノイズ級だが符号は一致して負。
「境界を精密に学ぶ → わずかに保守的な予測 → recallが微減」という挙動。

**判定**: fold0・fold1 への展開は低優先(効果量が採用閾値 +0.005 を大きく
下回り、方向も負)。アーカイブ目的で fold0 を回す余地はあるが、重みや
正規化の調整(§6.12優先度4の探索余地)を試すにしても、この1fold観測で
確実に言えるのは「0.15 単変数置換では改善しない」ことまで。

#### v54(EMA)fold2 補完 — 3foldで不採用確定

v59 の baseline 再利用で v54 fold2 manifest の存在が判明。ema行:
s3407 = 0.444965(Δ−0.0008)、s8407 = 0.444769(Δ−0.0002)で**両seed負**。
fold0(微小正)・fold1(符号不一致)・fold2(両負)を通してEMAは一貫して
無効〜微負であり、**不採用確定**。重み平均系レバーの探索を正式に閉じる。

#### v58(confスイープ)中間報告

- **prod は fold0・fold2 で conf=0.26 がピーク**(fold0: 0.4357、fold2:
  0.4433)と2fold独立で一致 — 現行操作点は適切で、prodのconfを動かす
  余地はほぼ無いことが確定
- v57baseline(再学習検出器)はピーク位置がseed不一致(s3407: 0.32、
  s8407: 0.26)で、「一律にconfを上げる」改善は成立しない
- dense2x の評価は fold0 manifest が conf=0.08 で途中停止
  (26/40行)。残りは notebook 再実行で resume される
- 低conf域の2点では dense2x は baseline に PQ で劣後(−0.015/−0.017)、
  FP 過剰(+422〜+444)が残り、dense FN優位も縮小(−11〜−17)

#### v59 fold0 補完 — fold間符号不一致で不採用確定

fold0 も実行完了(baseline行はv54 manifestから再利用):

| run | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| baseline s3407 | 0.441331 | 0.676897 | 0.651991 | 1703 | 772 | 1046 |
| dtb s3407 | 0.442857 | 0.676991 | 0.654155 | 1708 | 765 | 1041 |
| baseline s8407 | 0.441795 | 0.677506 | 0.652091 | 1700 | 765 | 1049 |
| dtb s8407 | 0.442102 | 0.676362 | 0.653647 | 1707 | 767 | 1042 |

paired Δ(dtb − baseline): fold0 s3407 **+0.0015** / s8407 **+0.0003**
(RQ経由: TP+5/+7、FN−5/−7); fold2 s3407 **−0.0018** / s8407 **−0.0003**
(RQ経由: TP−8/−3、FN+8/+3)。

**判定: 不採用(0.15単変数置換)。** fold0で両seed正・fold2で両seed負と
**fold間で符号が反転**し、効果量も全観測で |Δ|≤0.002 とノイズ級。
fold1をtiebreakとして回しても採用閾値+0.005に届く見込みがなく、
どちらに転んでも実効がない。距離変換境界損失が「マッチした予測の
境界精度」を上げる効果自体(fold2でSQ+0.0005/+0.0004)は検出できたが、
recallとのトレードオフがfold依存で正負に振れるため、この定式化・重みでは
正味ゼロ以下。優先度4を実験済みとして閉じる。残る未着手レバーは
優先度7(SAM2系refiner)のみ。

#### v58 fold0 完走 — dense2x の conf 救済は棄却確定

fold0 の dense2x 全16点が揃った(計40行)。paired Δ(dense2x − baseline、
同seed・同conf)の結果:

- **「両seed・同じconfで dense2x が baseline を上回る点」は存在しない**
  (s3407 で唯一の正は conf0.40 の +0.0038、s8407 では conf0.32 の +0.0007
  と、正になるconfがseed間で一致しない = 事前宣言の採用条件を満たさず)
- **各configの最良点同士でも dense2x が負ける**: s3407 baseline best
  0.4322@0.32 vs dense2x best 0.4279@0.32;s8407 baseline best 0.4281@0.26
  vs dense2x best 0.4248@0.32
- dense2x の FN(dense)優位は conf を上げると消える(s8407 conf0.40 では
  +1と逆転)。「提案量が多い」だけのリコール改善で、conf で絞ると
  実質が残らないことが確定
- FP 過剰は全conf域で残留(+60〜+444)

**確定知見**: oversampling 系レバー(優先度6)は機構の作動確認済みだが
「FP救済策としてのconf掃引」でも救えないことが確定し、完全に閉じた。
prod CONF=0.26 の最適性は fold0/fold2 でピーク一致として独立確認済み。
再学習検出器はピークconfがseed不一致になるだけのばらつきを持つことも
実測された(検出器実験はpaired必須の裏付けが追加)。

#### v59 fold1 補完 — 3fold完結、不採用確定

fold1: dtb s3407 0.440288(Δ−0.0001) / s8407 0.442721(Δ+0.0007)
(baseline v54再利用: 0.440425 / 0.441976)。

**全3fold・6観測の確定集計**: ΔPQ は fold0 [+0.0015, +0.0003]、
fold1 [−0.0001, +0.0007]、fold2 [−0.0018, −0.0003]。平均 ≈ ±0、
正負3対3、最大|Δ| 0.0018 — 完全にノイズ。SQ差もfold0/1では符号不一致、
fold2のみ両seed微増。距離変換境界損失(0.15単変数置換)はPQに対して
検出可能な正味なし。**優先度4、3foldの独立確認を以て不採用確定・完了。**

#### v60(copy_paste=0.2)fold2 完走 — 不採用確定、密集リコール系レバー総括終了

fold2 を4run(baseline/copypaste × seed 3407/8407)実行
(`runs/v60_copypaste/`、notebookはfold2実行)。採点は manifest
`v60_copypaste_fold2_manifest.json`、dense/sparse FN スライスはv57と同じ
cut定義。結果(refined OOF):

| run | PQ | SQ | RQ | TP | FP | FN | FN(dense) | FN(sparse) |
|---|---|---|---|---|---|---|---|---|
| baseline s3407 | 0.446406 | 0.681190 | 0.655333 | 1754 | 814 | 1031 | 487 | 544 |
| copypaste s3407 | 0.445746 | 0.681001 | 0.654545 | 1728 | 767 | 1057 | 510 | 547 |
| baseline s8407 | 0.445104 | 0.679765 | 0.654791 | 1746 | 802 | 1039 | 500 | 539 |
| copypaste s8407 | 0.425128 | 0.677786 | 0.627230 | 1617 | 754 | 1168 | 581 | 587 |

paired Δ(copypaste − baseline):

| seed | ΔPQ | Δraw PQ | ΔFP | ΔFN(dense) | ΔFN(sparse) |
|---|---|---|---|---|---|
| 3407 | **−0.0007** | +0.0027 | −47 | **+23** | +3 |
| 8407 | **−0.0200** | −0.0177 | −48 | **+81** | +48 |

**読み取り**: 仮説(インスタンス貼付が密集リコールを上げる)と**逆方向**の
挙動が両seedで一致した — copy_paste は dense FN を減らさず**増やす**
(+23/+81)。代わりに FP を一貫して減らす(−47/−48)。v57 dense2xの
「FN減・FP爆増」と表裏の関係で、copy_paste はこの設定では「過剰提案を
抑制する正則化」として働いた。さらに s8407 では学習が不安定化し
early-stop(ep47、best=ep33)して生の検出品質から落ちた
(raw PQ −0.0177、refined −0.0200)。

**判定: 不採用確定(事前宣言の採用条件に非適合)。** dense FN削減という
主目的は2seedとも達成されず、PQ正味も負(s8407で顕著)。fold0/fold1への
展開は機構の方向性が確定したため不要。FP削減(−47/−48)は一貫した副次
効果として記録に残すが、単体の採用根拠にはならない(FP削減を活かすには
conf下げ等の別レバーとの組合せが要り、その組合せ実験の費用対効果は
優先度7未着手を考えると低い)。

**総括**: v57(シーン複製oversampling)・v58(conf救済の棄却)・
v60(インスタンス貼付copy_paste)を通じて、**検出器側の密集リコール系
レバー(優先度6系)は機構の方向を問わずPQ正味で改善しない**ことが
確定した。残る未着手レバーは優先度7(SAM2系refiner)のみ。

### 6.19 v61 — SAM2 promptable refiner パイロット準備完了(優先度7、学習なし)

最後の未着手レバー(優先度7)を検証する notebook を用意した。

- builder: `scratch/build_v61_sam2_refiner_pilot.py`
- notebook: `solar_filament_v61_sam2_refiner_fold2_colab.ipynb`
- 仮説: 現行 refiner(MediSeg U-Net、crop320/context1.65、shrink確定構成)は
  小データ学習の頭打ち感がある。基盤モデル SAM2.1 hiera_large に
  **YOLO提案の外接矩形を box prompt として与え**、学習なしで refiner を
  置き換えられないか。H-alphaフィラメントは SAM2 学習ドメイン外なので
  「大きな賭け」だが、refiner が変えられるのは主に SQ(FNは提案集合で固定)
  のため悪化リスクが限定的。
- 設計: 学習なし・prod検出器(`runs/yolo_fold_{N}`)1本の提案
  (conf>=0.26・area>=20)に対し、同一proposal集合で4armをpaired評価:
  `raw` / `refined`(現行製品再現arm) / `sam2_free`(SAM2出力そのまま) /
  `sam2_shrink`(SAM2∩proposal)。画像ごとに `set_image` は1回だけなので
  推論コストは小さい。`refined` arm が fold2基準 0.4471 をほぼ再現するか
  どうかを経路健全性のチェックに使う。
- 診断: SAM2 の predicted IoU・IoU(SAM2, proposal)・面積比・空mask率を
  manifest の `sam2_diagnostics` に集計し、ドメインギャップの大きさを
  数値で残す(IoU(SAM2,proposal)が低いままならドメインギャップ敗北と
  判断し、ファインチューン系の追加投資はしない)。
- **採用ゲート(事前宣言)**: sam2系armの refined PQ が `refined` を
  **+0.005 超**で上回った場合のみ「候補」とし、同構成を第2foldで独立確認
  してから採否を決める。|Δ|<0.002 はノイズ棄却。負なら優先度7を不採用
  として閉じ、全優先度の実験が完了する。
- コスト: 学習なし・推論+SAM2のみで約1時間見込み。初回のみckpt約2.4GBを
  Driveへキャッシュ。manifest: `v61_sam2_refiner_fold{N}_manifest.json`。

#### v61 fold2 実行結果 — 不採用確定、優先度7終了(全レバー実験完結)

fold2 で4arm完走(`v61_sam2_refiner_fold2_manifest.json`)。同一proposal
集合での paired 結果:

| arm | PQ | SQ | RQ | TP | FP | FN | FN(dense) |
|---|---|---|---|---|---|---|---|
| raw | 0.3985 | 0.6516 | 0.6115 | 1605 | 859 | 1180 | 578 |
| refined(現行) | 0.4432 | 0.6788 | 0.6530 | 1714 | 751 | 1071 | 517 |
| sam2_free | 0.3079 | 0.6413 | 0.4801 | 1265 | 1220 | 1520 | 717 |
| sam2_shrink | 0.4084 | 0.6513 | 0.6271 | 1652 | 832 | 1133 | 549 |

paired Δ(arm − refined): sam2_shrink **−0.0348**(ΔSQ −0.0275、FP +81)、
sam2_free **−0.1354**(ΔSQ −0.0376、FP +469、RQ崩壊0.480)。

**読み取り**: ドメインギャップ敗北が確定的。

- sam2_shrink の SQ(0.6513)は raw(0.6516)とほぼ同じ — **∩proposalを掛けても
  SAM2の zero-shot マスクは提案品質をほぼ改善しない**。U-Net refiner が
  提案に乗せる +0.027 SQ は小データ学習で得たドメイン固有の精密化であり、
  汎用基盤モデルのbox-prompt zero-shotでは代替できない
- sam2_free は box 外へ大きくはみ出し(FP +469)、延長したマスクが他instanceを
  侵食して FN も +449 増 — 現行 shrink 構成の「提案で切る」設計が本質的に
  有効であることの裏返しでもある
- `refined` arm は 0.4432 で fold2基準(0.4471)より −0.004。prod ckpt と
  基準算出時の評価経路差と考えられるが、同一proposal内 paired なので
  判定には影響しない(健全性確認の範囲)

**判定: 不採用確定(事前宣言のノイズ帯 |Δ|<0.002 を大幅に下回る負)。**
box prompt zero-shot での refiner 置換は機構として成立せず。
sam2_diagnostics の IoU(SAM2,proposal) が低い=ドメインギャップ直接の証拠
として manifest に記録済み。ファインチューン系の追加投資は行わない方針を
事前宣言通り確認。

**総括**: §6.12 の優先度1〜7はすべて実験完了。採用に至ったレバーは0件、
全て paired 評価で棄却が確定した(EMA v54・マルチチャンネル v55・
スケルトンギャップ閉合 v56・密集oversampling v57・conf救済 v58・
距離変換境界損失 v59・copy_paste v60・SAM2 refiner v61)。現行 V30+refiner
shrink構成が、試した全代替案に対して最良のままであることが確定。

### 6.20 ポスト・レバーリスト実験計画 — エラー解剖駆動の機構選択(2026-10-06)

§6.12レバーリストは全消化。8連敗のパターンから確定した構造的事実:
**FNかFPを改善する単変数変更はもう片方を悪化させ正味±0**(操作点近傍で
PQが局所平坦)。以後の探索は「思い込みの優先度」ではなく、**エラーを機構
別クラスタに分解し各機構のoracle上界を実測してから選ぶ**方針に切り替える。
計算資源(AI Ultra)に余裕があるため、診断・実験はfold並列で進める。

#### Phase A — エラー解剖(v62、学習なし・約30〜60分)

prod fold2 検出器の提案→refiner→ownership 後の予測を1回生成してキャッシュし、
GT×pred IoU 行列から以下を計測:

- **FN分解**: 各FN GTの best pred IoU をバケット化 — `blind`(<0.1、
  検出器が完全に見逃し、提案なしでは救済不可)/ `weak`(0.1〜0.3)/
  `near-miss`(0.3〜0.5、境界改善で TP 化し得る帯)
- **FP分解**: `orphan`(best GT IoU<0.1、背景誤検出 — verifier系で除去可能)/
  `fragment`(TP済みGTに重なる余剰予測 — merge系で除去可能)/
  `misassigned`(FN GTに重なるが IoU<0.5 — 境界不良で、boundary改善の対象)
- **over-merge検出**: 1予測が複数GTの best match になる件数(split系の対象)
- **oracle上界**: 予測集合を機械的に書き換えてPQを再採点 —
  `drop_orphan` / `drop_fragment` / `perfect_verifier`(全FP除去)/
  `merge_oracle`(GTに重なる断片をunion)の各ΔPQ
- **verifier分離可能性**: conf・refined/raw IoU・refiner prob-mass の
  TP-vs-FP 分離度(rank AUC) — 特徴でFPが識別できるかを学習前に判定

出力 manifest: `v62_error_anatomy_fold{N}_manifest.json`。
各oracleのΔPQ = その機構が理想実装された場合の天井。**天井が+0.01以下の
機構は実験せず棄却**、最大天井の機構のみ Phase B へ。

#### Phase B — 機構別パイロット(v62の結果で条件付き起動)

| ID | 機構 | 起動条件(v62のoracle) | コスト |
|---|---|---|---|
| v63 | 提案を跨ぐ断片merge後処理 | merge系天井が有意 | 低(後処理のみ) |
| v64 | zero-training FP verifier(conf/probmass/面積比の閾値)+ 必要なら小分類器 | orphan FPが大・特徴AUCが高い | 低〜中 |
| v65 | refiner/detector TTA(hflip union等) | near-miss FN帯が大 | 低 |
| v66 | SAM2 decoder部分fine-tune | v61拒否はzero-shot範囲の話で fine-tuneは別機構。ただし期待値低め | 中(学習あり) |

機構実験の採用規約は従来通り: fold2で候補選定→別foldで独立確認、
ノイズ帯|Δ|<0.002は棄却。

#### Phase C — 定性30%(並行トラック、PQと独立)

LBリークで定量70%の伸びしろが限られる一方、合議制の定性30%は未投資。
`SCIENTIFIC_UTILITY_ROADMAP.md` の評価図・形態学的妥当性の証拠積み増し。
PQ実験の合間の待ち時間で進める。

### 6.21 競合・ディスカッション再調査(2026-10-06)— 外部oracle値で Phase B を較正

#### タイムライン訂正・確定

- 旧評価器は「matched pairのDice平均のみを取る」壊れた実装で、**空提出が
  0.88〜1.00を取れた**(discussion/729809 Valentin Haase氏の制御実験で確定:
  同一モデルで0予測=0.93、1.9件=0.72、11件=0.65と予測数が増えるほど低下)
- 8月7〜9日に評価器を正式PQに差替え、**8月12日にLB全件再採点済み**(公式
  アナウンス)。再採点後も残る 0.55〜0.56 クラスタ38チームは§6.12の通り
  MAGFiLOリーク由来で、誠実ティアの実力は **LB ~0.36程度が最前線**
  (too.foo氏のselector構成がLB 0.36を計測)
- 評価は**annotator record単位**(テスト1画像あたり平均2.4レコード、
  各レコードの個別アノテーションに対してPQ採点)。我々のOOF評価経路
  (annotator×image)と同じ構造で整合。annotator不一致(Kappa 0.66)により
  実質天井は PQ 0.45〜0.55 と推定(too.foo)

#### 競合の実力と手法(2026-10-06時点)

| 勢力 | 構成 | 実測 |
|---|---|---|
| too.foo(最も詳細な公開) | U-Net++ EffNet-B3 512pxパッチ → 連結成分 → **12特徴GBMコンポーネントselector** → adaptive crop refiner | val PQ 0.423→0.427、LB 0.36 |
| VisionX(gopi470) | v2: YOLO11s@1280 + CLAHE連結成分(recall) → IoU-NMS統合 → 可変GSD 4ch crop(raw/CLAHE/seed/DT) → smp.Unet refiner(0.3BCE+0.4Dice+0.3Boundary、EMA/SWA/HNM) → 6pass TTA+Otsu+skelギャップ閉合 | スコア非公開 |
| solar-seg(srivatsav) | 幅24コンパクトU-Net + 4flip TTA、独自holdout/embargo | holdout 0.346、LB 0.31 |
| 我々 | YOLO11m-seg + MediSeg refiner shrink | OOF refined 0.443(fold2) |

**注目点: 我々のOOF 0.443は公表されている誠実ティアの最高水準(0.423〜0.427)
を上回っている可能性が高い**(val protocol差はあるが同一公式プロトコル基準)。

#### too.foo の oracle 実測値 — 我々のv62結果の外部キャリブレーション

彼らが公式プロトコルvalで測った機構別の天井:

- **selection oracle(TP相当のみ残す): +0.083(0.398→0.481)** — 提出成分の
  約半分がどのannotatorとも一致しない。「keep/drop判定」が最大の実現可能
  レーン。GBM selector v1で +0.018 実取(0.398→0.416)、v2(23特徴+
  record単位Dinkelbach閾値 p*=λ/2ī≈0.32)で 0.423 まで
- **trim-aware oracle(マスクを完全整形): 0.756** — ただしcrop refiner実装は
  crop-IoU 0.63→0.67 に改善しても **PQは不動〜微負(0.414<0.423)**。
  「平均crop-IoUが上がってもIoU 0.5断崖を跨ぐ数が増えない」と同じ理由で
  refiner系は機能しない — **我々のv59/v61棄却と独立に一致**
- **fragment merge oracle: +0.026** — 実在するが中程度
- **hysteresis再構成(弱い確率橋で断片を接合): −0.004 棄却** — 我々の
  v56スケルトン閉合(+0.0004)と同じファミリーの失敗
- 損失/target変種6系統(consensus M*、endpoint重み、skeleton-recall、
  768px context等)は全て ±0.02 内 — semantic単体モデルの天井実証

#### 我々の Phase B への示唆(計画較正)

1. **v64 verifier(FP selector)が最優先候補** — too.fooの最大レーン(+0.083)
   と一致。彼らのselector特徴(局所chromosphereに対する暗さコントラスト、
   細長さ、確率統計、サイズ、limb近接度)をv62の特徴セット
   (conf/probmass/IoU/面積)に追加検討価値あり。我々のFP=751は同規模の
   機会を示唆
2. **v63 mergeは天井+0.026程度と期待値を下方修正** — v62のmerge oracleが
   それを大きく上回る場合のみ実施
3. **refiner精密化レーンは二チーム独立に否定済み** — v59/v61/v62系の
   「境界を磨く」方向は閉じたまま維持
4. **recall specialistのinstance-level統合**(too.foo計画中、未結果)は
   我々のv57/v60密集recall失敗と同じリスクを持つが、**selector通過を
   挟む設計**が彼らの差別化点 — v64が先に成否を決める

#### 定性30%の現実的価値(再確認)

審査はNSO/GSU/NJIT/UMSL委員会の合議制。最終提出は4ページレポート+公開
再現repo+予測morphology(締切はtoo.foo記載で11月15日付近、要公式確認)。
誠実ティアでレポートを書いているのはtoo.fooのみ — **定性30%は最も
競争密度の低い得点源**。

#### v62 fold2 実行結果 — 解剖確定、v63棄却・v64採用(2026-10-06)

fold2 prod ckpt の全最終予測(1498個)を解剖(`v62_error_anatomy_fold2_manifest.json`)。
baseline再現: refined PQ 0.4432(v61と一致、経路健全)。

**エラー解剖**(annotator×image基準、n_gt=2785):

| クラスタ | 個数 | 機構の対応 |
|---|---|---|
| FN blind(IoU<0.1) | 561 | 提案なし — 検出リコール系(全滅済み) |
| FN weak(0.1-0.3) | 168 | 提案弱 — 同上 |
| FN near-miss(0.3-0.5) | 333 | 境界不良 — refiner系(2チーム否定) |
| FP orphan | 136 | 背景誤検出 — **verifierで除去可** |
| FP misassigned | 196 | FN GTに重なる不良品 — verifier候補 |
| FP fragment | 2 | **ほぼ存在しない**(ownershipが処理済み) |

**oracle上界**: perfect_verifier **+0.0462** / drop_orphan **+0.0169** /
merge +0.0035 / drop_fragment +0.0003。

**判定**: 事前宣言ゲートにより v63 merge **棄却**(天井+0.0035<+0.01)、
v64 verifier **採用**(最大天井、too.fooの+0.083独立実証と同構造)。
v65 TTAは保留(near-miss残るがrefiner系は否定済み)。

**verifier分離度(proposal単位rank AUC)**: conf 0.710 / probmass 0.537 /
ref_raw_iou 0.531 / 面積 ~0.48。**既存特徴だけでは弱いが、conf一本で切る
現行より学習selectorの改善余地が実測で存在**。too.fooの実証済み特徴
(暗さコントラスト・細長さ・limb近接度)を追加する設計へ。

**構造的洞察**: FNの68%(729/1071)は提案不在(blind+weak)。verifierが
強ければ conf floor を下げて提案を増やしてもFPを制御できる = 全滅だった
「FN↔FPトレードオフ」を壊せる唯一の機構。v64は conf floor 0.05 で
全提案を取り、selector×conf のフロンティアを測る設計とした。

#### v64 fold2 結果 — selector棄却確定(2026-10-07)

HistGBM selector(10特徴: conf/probmass/IoU/面積比/aspect/solidity/
暗さコントラスト/limb距離、train-Aのみ学習、train-Bで操作点選定):

- **trainB最良点は conf=0.26・p_keep=0.0 = selector不使用**(trainB PQ 0.4903)
- val側gridでも凍結点以外の全セルが +0.002未満
  (val oracle最良 conf=0.26・p_keep=0.3 → 0.4445、Δ+0.0013のみ)
- 凍結点 ΔPQ +0.0000 → **実験的棄却**。残りFPはconf≥0.26・refiner・
  ownershipを通過した難しい誤検出で、この特徴集合では分離不能と確定。
  oracle +0.0462は「この特徴/モデルでは届かない天井」だった
- 特徴重要度(train-B permutation AUC): **conf=0.2412**、probmass=0.0306、
  ref_raw_iou=0.011、他(solidity/darkness/limb/aspect/面積系)は全部<0.008
  → 追加9特徴がほぼ無情報で、selectorはconfの再表現に留まった

実装上の障害も克服して記録: collect 提案をRLE保持化(2048×2048 bool全保持で
OOM ~100GB → ~40分の1)、ファイル単位pickle ckptで中断再開、
`v64_status_fold2.json` で Drive 経由の外部進捗監視。

#### 残る機構 — v65/v66 を実装済み(評価待ち)

- **v65 TTA-consensus**(fold2): id/hflip/imgsz1536の3viewから
  `tta_hflip_iou/conf`・`tta_big_iou/conf`・`tta_support` 特徴を作り、
  (a)consensusフィルタ単体 (b)selector-v2(15特徴)を同じA/B凍結プロトコルで
  評価。`build_v65_tta_consensus.py` → `solar_filament_v65_tta_consensus_fold2_colab.ipynb`
- **v66 RF-DETR union**(fold0): v45のOOFキャッシュが残っているので
  再学習なしで「yolo@0.26 ∪ rfdetr@f」のrefiner+ownership評価。
  blind-FN rescue診断付き。`build_v66_rfdetr_union.py` →
  `solar_filament_v66_rfdetr_union_fold0_colab.ipynb`
- 両者棄却なら定量系は機構的に枯渇と宣言し、定性30%へ移行
  (→ **v65・v66 ともに棄却確定、枯渇宣言が発動**(下記))

#### v65 fold2 結果 — TTA-consensus ノイズ棄却(2026-10-07、manifest確定値)

3view(id/hflip/imgsz1536)の変換安定性特徴で2系統評価:

- trainB 最良: selector-v2(pk) conf=0.26・param=0.3(trainB 0.4932)→ 凍結適用で
  **val 0.4440(Δ+0.0007)** — |Δ|<0.002 のノイズ棄却域
- 挙動は再びゼロ和: TP −16 / FP −40 → 正味+0.0007
- **val側oracle最良(診断): pk conf=0.26・param=0.5 → 0.4484(Δ+0.0052)**
  — oracleは採用閾値を超えるが trainB が選んだ点と一致せず、
  操作点の汎化性が無い(v64と同じ「信号はあるが操作点が転移しない」構造)
- **特徴重要度でTTA特徴の合計(~0.126)が conf(0.083)を上回った**:
  tta_hflip_conf 0.059・tta_hflip_iou 0.026・tta_big_* 各0.021。
  変換安定性は実信号を持つが、残りFPは「3viewでも安定的に出る難しい誤検出」で
  分離しきれなかった
- support(consensusフィルタ単体)は低conf洪水を制御できず全滅

#### 定量レーン枯渇の確定宣言(2026-10-07)

この操作点(YOLO@0.26 → refiner → ownership)で試せる機構は全て出尽くした:

| 系列 | 機構 | ΔPQ |
|---|---|---:|
| v54-v61 | EMA・multi-ch・skeleton・oversample・conf-sweep・DT-loss・copy-paste・SAM2 | 全棄却 |
| v63 | merge後処理 | oracle +0.0035(事前棄却) |
| v64 | 静的特徴selector | +0.0000 |
| v65 | TTA-consensus | +0.0007 |
| v66 | RF-DETR union | -0.0276 |
| v67 | temporal近傍増強 | -0.0087 |
| v68 | 磁気図PIL補助selector-v3 | +0.0000 |

残存方向は「この枠組みに無い情報源」の持ち込みのみ:
時間近傍無ラベルフレーム増強・共時刻磁気図補助信号(共に主催者裁定済み合法、
コスト見積要)→ または定性30%トラックへ資源移行(11/15締切の提出物準備)。

#### v68 magnetogram PIL features — 棄却確定(2026-10-10、最後の定量レバー)

最後に残った合法新情報源 = 共時刻GONG `bqr` 磁気図(180×180・10分cadence)。
観測時刻+サイトから最近傍を解決し、H-α disk に写像してPIL(分極反転線)
距離等5特徴を提案へ付与、v64と同じselectorプロトコルで評価:

| 項目 | 値 |
|---|---|
| 磁気図カバレッジ | **671/707 (95%)** |
| sanity(信号確認) | hit→PIL中央値 **63.3px** vs ランダム **80.7px** (n=573) |
| 特徴付与 | 11,353/11,854 提案 |
| trainB 最良 | conf=0.26 pk=0.0(selector不使用) |
| 凍結 val ΔPQ | **+0.0000 → noise-reject** |

**結論**: PIL 近接性は物理的に実在する信号だが、conf≥0.26・refiner・
ownership を通過した残り FP751 は「磁気図でも区別できない難しい誤検出」
と確定。**全定量レバー全滅確定(v54-v68、15機構)**。残りは定性30%+
提出物トラックのみ。

- `scratch/v68_magnetogram_probe.py`(ローカル実行・partial checkpoint再開対応)
- manifest: `output/v68_magnetogram_probe_manifest.json`
- 教訓: `cv2.imread` は非ASCIIパス(マイドライブ)を読めない → `imdecode`経由必須。
  GONG HTTP は~50req/短時間で遮断 → DL済みはローカル索引から引く

#### v67 temporal-neighbor augmentation — 棄却確定(2026-10-09)

枯渇宣言後の唯一残る合法定量レバー。fold2 pilot(±20分・近傍最大2件・
同一annotation流用、prod同一ハイパラ、**同一セル内ペア採点**):

| arm | PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| prod 再採点 | 0.4432 | 1714 | 751 | 1071 |
| v67 近傍増強 | 0.4345 | 1658 | 735 | 1127 |

**ΔPQ −0.0087 → 棄却**。SQは同等(0.678)だがRQ低下(0.640 vs 0.653)、
TPが−56 — 増強で検出漏れが**増えた**。推定原因: ±20分のラベル陳腐化と
FITS→JPEG階調近似のドメイン差が、ラベルノイズ+分布ノイズとして働いた。
「同じfilamentの別実現」は augmentation として有効ではなく、
ノイズとして作用したと読む。

**これで全定量レバーが尽きた** — v54-v67 の14機構すべて棄却。
残る合法方向は共時刻磁気図補助信号のみ(コスト見積レベル)か、
定性30%トラックへの全振り。

- `build_v67_temporal_aug.py` → `solar_filament_v67_temporal_aug_fold2_colab.ipynb`
- manifest: `artifacts/v67_temporal_aug_fold2_manifest.json`

#### v66 fold0 結果 — naive union棄却、ただし相補リコールの存在を確認(2026-10-07)

RF-DETR OOF キャッシュ(v45由来、再学習なし)で yolo@0.26 ∪ rfdetr@f を評価:

| arm | PQ | ΔPQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| baseline yolo@0.26 | 0.4393 | — | — | — | — (fold0基準と完全一致) |
| union@0.4(最良) | 0.4117 | **-0.0276** | 1752 | 1202 | 997 |
| union@0.05 | 0.1737 | -0.2656 | 1560 | 7399 | 1189 |

→ naive union **棄却**(ゲート通り負)。RF-DETR は低conf帯で数千のジャンクを
出し、ownership経由で既存TPのマスクも破壊する。

**だが rescue 診断に重要な残滓**: blind GT 612のうち RF-DETR@0.10 が
IoU>0.3 で 493個、**IoU>0.5 で 317個をカバー** — 提案源の相補性は実在。
ただし raw box IoU>0.5 の 317 が refine+ownership+match 後に TP 化するのは
ごく一部(実際のFN減は -74)。「box段階の相補カバー → 最終TP化」の変換効率が
悪いことが次のボトルネックと判明。条件付きunion(YOLO非検出領域のみ追加)や
source-aware verifier は別機構として原理的には残るが、優先度は低い。

### 6.22 競合・規則偵察(2026-10-07)

Kaggle API(LB全895チーム)+公開カーネル10本 pull+ディスカッション全スレッド
走査の結果。詳細は `scratch/competitor_recon_2026-10-07.md`。

**評価構造(実態)**: 定量70% = PQ + Dice/IoU分布 + **one-to-many/many-to-one
分布**(断片化・過剰マージ — v62解剖のクラスタがそのまま審査項目)。
定性30% = pipeline記述 + 形態 + コード品質。public LBは test の ~50%のみ、
**private LB(残50%)も存在**。**締切 11/15**: 4p report + 公開repo +
requirements.txt + 全パイプラインnotebook が必須。

**リーク構造の確定**: 公開 MAGFiLO v1.0 がテスト観測をカバー → GT dump で
~0.55。LB 0.55+の72チームがリーククラスタ、0.40-0.50帯は26チームのみ。
主催者: "0.35超は十分価値あり"・"0.41は真剣に受け止める"・"fake onesは
容易に除去する"・LB最終クリーン化予定。**OOF 0.443 は誠実ティア最前線**。

**重要な較正値**(tushhaaarrrr): **human-vs-human PQ ≈ 0.34**(1人のannotatorを
他annotatorで採点)。0.443 は単一annotator一貫性を上回る。

**合法レバーの境界(主催者裁定済)**:
- ✅ train JSONの spine/bbox/area/category_id、mask由来 skeleton/距離変換
- ✅ **近傍・共時刻の無ラベルGONGフレームを学習時に使用**(temporal増強)
- ✅ **共時刻GONG磁気図(bzi)を学習時のみの補助信号に**(「good idea」)
- ✅ MAGFiLO期間外の無ラベルGONG半教師あり
- ❌ 推論時の外部入力(単一test画像のみ)
- ❌ 外部GT标注(SWEFil 明示的に禁止)

**競合で確認された我々と同型/近い手法**:
- timgluhkikh: seed-refine cascade(我々と同型!)+ **Box-Jitter TTA** → local ~0.409
- muhammadshoaibaltaf V9: yolo26l→crop UNet++/EdgeAttNet+OBB — 同型2段
- sivapnlr80: MaskRCNN×5fold+**4-TTA+cross-fold consensus**+refiner → LB 0.34
- tushhaaarrrr: 物理チャネル(flat-field/Lindeberg ridge)+annotator-mean soft
  mask+spine aux+PQ-aware後処理+FNバケット分析(我々のv62と同じ発想)

**含意**: (a)TTA/consensus系(v65)は外部でも検証されている方向で妥当。
(b)未開拓の合法レバー = 時間近傍フレーム増強と磁気図補助 — v64で失敗した
orphan FP判別に「別種の物理情報」を持ち込める可能性。コスト見積が先決。
(c)報告書は審査基準(分布統計+形態+コード品質)に直接マップする構成で
書く — v62解剖・負の結果の記録はそのまま加点材料。

---

### 6.23 競合・ディスカッション・Biohub上位解法の再調査(2026-10-09)

**2026-10-12 の統合時注記:** 本節は作者の当時の調査・計画を保持した記録。
外部スコア・実行結果は今回独立再検証していない。`HANDOVER.md` 冒頭の manifest 未確認・
489/707件の分割相違は未解決で、後続 v80/A1 の報告 OOF 値・CI・採用ゲート通過は探索扱いとする。
scorer の fold 分離だけでは、別foldキャッシュを作った上流モデルが評価outer foldを学習している影響を除けない。
不偏な独立評価・採用判断には、上流モデルとキャッシュも outer fold ごとに再構築する nested 検証と分割同一性確認が必要。
「リーク系」「誠実ティア」は作者の当時の仮説・比較対象の呼称で、個別提出の違反や誠実性を検証した分類ではない。
他者の公開スコア・分割・来歴は独立未確認で、スコア帯だけから違反や実力上限を結論しない。

**【番号について】並行レーン `devin/solar-next-pilots`(PR #244)が §6.17〜6.22 と v58〜v68 を使用済みのため、本節は §6.23、実験IDは v80 以降を使う。** §6.12(2026-09-15)以降のKaggleディスカッション・公開notebook・Biohub
(Cell Tracking During Development、2026-09-29終了)の公開writeupを調査し、
§6.16末尾で示した「次は提案品質側のレバー」という方向を外部証拠で裏付けた。
一次情報はKaggle discussion `747399`・`746318`・`744765`・`743958`・`743385`・
`743521`・`744047`・`741892`・`740400`、notebook `tushhaaarrrr/solar-filament-
segmentation-2026-0-40`・`juanmirod/solar-filament-2026-final-report`
(いずれも`kaggle kernels pull`で取得・精読)、Biohub writeup 1st/3rd/12th place。

#### 6.23.1 競技の実像の更新

- **Public LBはtestの約50%(≈90枚)で採点される**とリーダーボードページに明記
  されている。README §2.1.6のOOF-Public LBギャップ議論は「test 180枚」を前提に
  していたが、実際の採点枚数はその半分で、tushhaaarrrr氏の見積りでは
  リサンプリングだけでsd≈0.012。**Public LBの2桁表示は0.01未満の差を
  原理的に識別できない**。提出間比較はOOFのみで行うという既存方針を維持し、
  LBは「順位が崩れていないことの確認」にのみ使う。
- リーク系クラスタはさらに上積みされ、上位は0.63/0.62/0.61…(1位は108提出)。
  運営(Azim氏、16時間前)の最新発言: 「PQ 0.35超の全提出を精査する」
  「fakeは容易に弾ける」「0.41は非常に真剣に受け止める」「LBのクリーンアップは
  行う(詳細は未公表)」。追走しないという§6.12の判断を維持。
- **ルール裁定(新規)**: 外部GT(SWEFil等)は**禁止**(`746318`)。推論時の近傍
  フレーム利用も**禁止**(`743385`・`744047`)。一方、**訓練限定**なら、ラベル
  なしGONG時系列・**同時刻GONG磁場図の補助損失**・グレースケール3ch複製は
  いずれも明示的に許可された。本プロジェクトの現行構成はすべて適合。
- 誠実ティアのPublic最前線は**0.40**(tushhaaarrrr氏)。我々の0.39は並走圏。
  ただしOOFでは差がある(下表)。

| チーム | OOF PQ | Public | 構成 |
|---|---:|---:|---|
| 我々(V52 / plain_mhsa) | 0.4432(707枚) | 0.39 | YOLO11m → U-Net refiner shrink@0.45 → 排他化 |
| tushhaaarrrr (filseg v4) | **0.4597**(703枚) | 0.40 | 密な確率マップ → component tree → 学習スコアラー → 厳密DP → 回転正規化refiner |
| juanmirod | 0.44(106枚) | 0.37 | YOLO11m → crop refiner、我々とほぼ同型 |
| Spaghetto99 | 0.48(106枚、分割法不明・未検証) | 不明 | Focal+Dice、詳細非公開 |

#### 6.23.2 tushhaaarrrr氏の設計から学ぶべき点(最重要)

スコア差の源泉は**モデルではなく決定層**。要点:

1. 密な確率マップ p(x) = 「ランダムなアノテータが画素xを塗る確率」を
   soft annotator-mean目標で学習(BCE+0.5 Dice、EMA、D4 TTA、3モデル×5fold)。
   **校正を壊す損失(clDice −0.045、Lovász)は全部不採用**。
2. 15閾値のcomponent tree(max-tree)を2048px nativeで構築。forestの任意の
   antichainが有効な非重複提出になる。
3. 各候補ノードに**期待matched IoU û = ĥ·m̂**(ĥ=IoU>0.5で一致する確率、
   m̂=一致時のIoU)をLightGBM hurdleモデルで予測。56記述子+RF-DETRの
   per-node証拠4つ。GroupKFoldでcross-fit、行重み=アノテータ数R。
4. **Dinkelbach**: pooled PQは2つの加法量の比なので、`Σ(û − τ)`を
   τ=PQ*/2(≈0.26、**単一のグローバル定数**。画像別閾値は最適でない)で
   最大化するantichainを木のDPで厳密解。本人計測で**固定閾値比+0.06**。
5. Stage 2: 選択済み個体ごとに**長軸を水平に回転正規化した384×128クロップ**
   (5ch: flat field・p・spine・seed・**他の選択済み個体**)で再セグメント。
   出力閾値は期待IoU最適点 t*=J*/(1+J*)≈0.40 を理論導出(実測最適と一致)。
6. 不採用リストが我々の負の結果とよく整合: 時系列TTA −0.02、磁場図補助
   ヘッド n.s.、noisy student(外部1736枚) = seedノイズ、neural critic −0.009、
   morphological dilate/erode ±1px で −0.025、per-instance閾値 n.s.。
   **「2048 nativeで木を切らないと−0.006」**(1024マスクの劣化)は我々の
   refiner解像度にも関係しうる(未確認)。

**我々との対応関係**: README §1に`p·u > 0.5·PQ`の損益分岐が既に書かれて
いるが、実装は`conf 0.26固定 + drop_contained`のまま。V42の11特徴logistic
verifier(§27.13)は**pre-refiner代理目標**で最終PQへ移らず不採用となったが、
S1計画(§27.18)で予定していた「**post-refiner・最終PQ利得目標**」版は競技
トラックでは未実行。tush氏の設計はまさにその形(post-候補・期待IoU目標・
GBDT・グローバルτ)であり、§6.16が結論づけた「提案品質側のレバー(優先度5)」
と一致する。OOF cacheは`conf 0.08`まで候補を保持している
(`output/v35/pq_conf_sweep.csv`: raw TP 5420@0.08 vs 5017@0.20)ため、
**GPU不要・既存cacheのみ**で3-fold cross-fit検証が可能。

#### 6.23.3 Biohub上位解法からの移植可能性

| Biohub(順位) | 内容 | Solarへの対応 | 評価 |
|---|---|---|---|
| 1st | 校正された小モデル群(検出・Soon Net・linker)+ greedy decode | 校正スコアラー+厳密最適化(=6.17.2と同原理) | **最有力** |
| 1st | 2周目でhard-negative mining(OOFで誤分類した候補を1:1混入) | スコアラーが誤採択する候補をOOFで掘り再学習 | 上記に同梱 |
| 1st | EMA+強augmentation | v54 EMA pilot(fold1/2待ち、§6.15) | 継続 |
| 1st | MAE事前学習(ラベルなし、競技180動画+外部) | ラベルなしGONGで許可。ただし本人も効果不明、tush氏のnoisy studentもnoise水準 | 低 |
| 3rd | 検出heatmapをfold平均+flip TTAで**確率平均**(3backbone加重) | refinerの確率マップを3fold refiner+flipで平均。V43/V53の失敗は**maskレベル投票**であり確率平均は未試行 | 中・安価(推論のみ) |
| 3rd | 疎アノテーションで不確実な背景を損失から除外 | 単一アノテータ画像で「塗られていない≠負例」。soft annotator-mean目標が対応物 | 中 |
| 3rd | public/private分布差をprobingで把握、選抜はOOF固定 | Public≈90枚、選抜はOOFのみ(既存方針) | 確認済 |
| 1st/3rd | 高信頼pseudo-label | V49で否定済み(§README 2.1.3) | 不採用 |
| 12th | 各段を計測済み・fail-safeな修復ステージとして積む | 本プロジェクトの規律と同じ | 確認済 |

#### 6.23.4 再整理した優先順位(§6.12・§6.16を更新)

1. **post-refiner期待IoUスコアラー + τ=PQ/2選択(優先度: 最高、CPUのみ)** —
   YOLO confを0.10程度まで下げて候補を広げ、各候補にrefiner適用後の
   記述子(YOLO conf・面積・伸長度・refiner確率統計・shrink比・limb距離ρ・
   局所密度・他候補との重なり等)からû=ĥ·m̂をLightGBM hurdleで予測、
   `û > τ`で選択→既存排他化。教師はOOF cache上で各recordのGTとのIoUから
   `u = (1/R)Σ_r IoU·1[IoU>0.5]`として直接計算。3-fold cross-fit、
   採否は従来通り3fold符号一致+observation bootstrap CI。V42との差分
   (post-refiner目標・GBDT・グローバルτ・候補拡張)を明示して記録する。
   **tush氏のE6(hurdle > direct regression)・E9(neural critic −0.009)に
   従い、最初からGBDT hurdleで行き、NNは試さない。**
2. **refiner確率マップのfold平均+flip TTA(優先度: 高、推論のみ)** —
   3fold refinerの確率を平均してからshrink@0.45。V53の失敗はYOLO proposal
   のmask投票統合であり、refiner確率平均とは別物。
3. 回転正規化クロップ+隣接個体チャンネル(優先度: 中、refiner再学習) —
   v55多チャンネル化が−0.02で失敗した経緯(検証Datasetのseed RNG問題含む)が
   あるため、着手するなら検証proposalを固定した上で1と2の後。
4. 報告書・README反映(定性30%) — 「Public≈90枚」「運営の精査方針」
   「外部GT禁止裁定への適合(MAGFiLO本家・SWEFil不使用)」「訓練限定なら
   磁場図・時系列が許可(未使用と明記)」を`report/main.tex`・README §1に反映。

締切までの週次スケジュール・実験ID採番・採用ゲートは
[`PLAN_2026-10-09_FINAL_SPRINT.md`](PLAN_2026-10-09_FINAL_SPRINT.md)を正本とする。

優先度1の設計は`docs/s1/S1_PROPOSAL_MARGINAL_UTILITY_PILOT_REPORT_2026-08-24.md`
の教師生成手順(leave-one-out最終PQ利得、31.5秒/観測)より単純な
「候補単位の期待IoU」を目標にする。ownership相互作用は目標に含めず、
選択後に実際の排他化を通した3-fold OOF PQで直接判定する。

### 6.24 並行レーン(devin/solar-competitor-survey-2026-10)の v80/v81 結果 —
「全定量レバー全滅」宣言の訂正(2026-10-10)

§6.22 までの HANDOVER(070213aa)は「v54-v68 全15機構棄却・残りは提出物
のみ」と宣言したが、これは**調査レーン側の v80 結果確定前**に書かれた
もの。同レーン(PLAN_2026-10-09_FINAL_SPRINT.md §7.1.8b/8c、worktree
`devin-solar-competitor-survey`)の 707観測・全fold OOF 公式PQ評価で:

- **A1(yolo_scorer)= +0.0039 で採用ゲート通過**: YOLO 候補 conf≥0.08 に
  fold別 refiner(shrink@0.45)を適用し、post-refiner 特徴から期待IoU
  û=ĥ·m̂(cross-fit HistGBM hurdle)を予測、û>0.22 で選択。3fold 全て正
  (+0.0008/+0.0069/+0.0038)、pooled bootstrap CI95 [+0.00057,+0.00735]
  でゼロを除外。v64 静的 selector(+0.0000)との差分は「post-refiner 特徴
  + hurdle 目標 + conf≥0.08 候補拡張 + 3fold cross-fit」の設計差。
  **この節の記録時点では提出構成への組込み候補だった**。後続の
  `HANDOVER.md` §5 と `report/main.tex` では、A1 の Public LB が
  prod の 0.39 に対して 0.37 となり、不採用になったと記録されている。
- E1(hybrid、tree 候補併合)= −0.0142 で不採用。v81 tree 側 τ 独立掃引
  (0.20–0.68)は単調に yolo_scorer 漸近線へ収束し、**tree 候補追加は
  どんな τ でも不利益**と確定(û の絶対校正では解放不可)。
- **hybrid_oracle = +0.1067**: component-tree 候補(v34 OOF 密確率マップ)
  を含む候補集合に大きな余力が実在 — too.foo 氏 selection oracle +0.083、
  v62 blind-FN 68% と整合。解放には別機構(校正マップ再学習 v86、強い
  記述子、ranking 損失、FP ペナルティ選択)が必要。
- スコアラー診断: ROC-AUC(h>0) tree 0.900・Spearman ρ(û,u) 0.676 —
  û の順位付け自体は良好で、失敗は絶対校正(中位候補の過大評価が
  antichain DP の累積選択で断片化を誘発)。

**この節の記録時点の判断**: 「定量レバーは A1(+0.004)のみ残存・採用候補、
tree 候補路線は v80/v81 で閉じた」。これは後続の A1 不採用判断より前の
経緯であり、現行の提出構成への採用指示ではない。最新の提出方針は
`HANDOVER.md` §3・§5 を参照する。

**2026-10-05 回収・再開契約補修:** 元 PR #239 の既存 fold0 manifest を Drive から
読取りで照合した。SHA256 は
`84f254f835f706631b229fc3a059f17adc033bc2a99b5e29292bb13c6f46b7ef`。
上記4runの数値と一致し、不採用判断と提出正本は維持する。補修版での GPU 学習・
Colab 実行・独立性能受入は未実施。元 manifest は入力/重み指紋と GPU/VRAM 計測を
含まないため、補修版の再開入力には使わず、歴史的な実行記録として保持する。

dataset・checkpoint・manifest の再開時は、注釈・fold・前処理画像・refiner・初期 YOLO 重み・生成コード
の SHA256 と実験設定を照合し、不一致や指紋欠落で停止する。完了結果の再利用では
checkpoint 本体の SHA256 も照合する。初期 YOLO は実際に解決した checkpoint パスの
SHA256 を固定し、未完了 arm の load 前にも照合する。新規実行時の manifest は GPU 名・PyTorch の
ピーク割当 VRAM・所要時間・UTC 完了時刻を記録する。計測は今回の実行範囲に限定し、
新規学習＋評価・途中学習の再開＋評価・完了 checkpoint の評価のみを区別する。
過去の中断セッションの学習時間/GPU/VRAM は未取得と明示する。Colab compute units は runtime API
から取得できないため null と理由を残し、セッション UI で別途記録する。
これは競技用探索 pilot の保守であり、S1 登録契約と blocked gate は変更しない。

### 6.25 v69 タイル推論(native解像度SAHI相当)— fold2 OOF 棄却確定(2026-10-13)

作者のfold2探索記録をPR #352から保持する。独立再実行や他foldの確認は未実施で、
過去実行manifest未確認と分割相違は解消していない。以下の棄却は試した構成に限定する。

v62 解剖で FN の68%が提案不在(blind+weak)と判明後、検出リコール系に
唯一残っていた仮説: 「全画面2048→1280の0.625圧縮で細いフィラメントが
消えている」。これは v65 TTA(imgsz1536全画面=0.75スケール)では
カバーされない native 解像度の機構。2x2タイル(1152px・overlap256)で
検出→prod同一 refiner/ownership/公式PQまで通して fold2 OOF 評価。

**検出プローブ**(`run_tile_infer_probe.py`、全235観測):
blind GT 143→60、うち IoU≥0.5 に達する救出 **60件(42%)**。
prod conf≥0.26 に届くのは10件のみ(別提案の同時適合を数える修正版
カウント、旧値8を訂正) — 救出の83%は低conf帯。

**全パイプライン**(`run_tile_pipeline_oof.py`、keep閾値×NMS×unionの
グリッドを union-refine1回で使い回し、base PQ 0.4442 = v62基底0.4432と整合):

| arm | PQ | Δ | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| base@0.26 (prod) | 0.4442 | — | 1715 | 750 | 1070 |
| base@0.195 | 0.4355 | −0.009 | 1780 | 968 | 1005 |
| tile@0.26 | 0.4047 | −0.040 | 1829 | 1530 | 956 |
| tile@0.10 | 0.3359 | −0.108 | 1870 | 2871 | 915 |
| tile@0.05 | 0.2948 | −0.149 | 1845 | 3796 | 940 |
| tile@0.10\|nms0.3 | 0.3540 | −0.090 | 1965 | 2788 | 820 |
| union@0.26 | 0.4116 | −0.033 | 1862 | 1506 | 923 |
| union@0.10 | 0.3378 | −0.106 | 1861 | 2814 | 924 |

**判定: 作者はこのfold2探索で棄却し、fold0/1は未実施**。
本命の union@0.26(base全保持+tile追加、Codex P1対応)でも TP+147 に
対し FP+756。tile-only では TP+114/FP+780。keepを下げるほど悪化、
強NMSでも不届。

**含意**: v80 hybrid_oracle +0.1067 と v62 blind-FN 68% が示す容量は
検出側に実在するが、rescued-TPとtile-FPを分離するverifierなしでは
収益化できなかった。v64(+0.0000)・v68(+0.0000)も含む試行範囲では
追加採用策が得られず、最終選択を変更しなかった。
未試行手法を含む改善余地の不存在や、一般的な性能上限の証明ではない。

証跡: `output/tile_infer_probe/`、`output/tile_pipeline_oof/`、
`scratch/run_tile_{infer_probe,pipeline_oof}.py`、
`scratch/recount_rescue_prod_conf.py` (P2訂正の再計算)。

### 6.26 v70 深層verifier(crop CNN)— fold2 棄却確定(2026-10-13)

§6.25の「rescued-TPとtile-FPを分けるverifierが必要」という帰結に対し、
v64静的特徴とは別の仮説クラスとして画像を直接見るCNN verifierを検証。
提案クロップ(RGB+mask4ch,128px)を入力、「best IoU>0.5の提案か」を
2値分類。train=fold0/select=fold1/eval=fold2(v64と同型)。

データ: OOF提案conf≥0.08から各fold約2,600crops(陽性率~55%)。

| 指標 | fold1 | fold2 |
|---|---|---|
| conf単独 AUC | 0.802 | **0.797** |
| CNN verifier AUC | 0.664 | **0.638** |
| blend(conf+ver, fold1 fit) | — | 0.7934 |

**判定: 棄却**。レビューで前処理バグ2件(BGR→RGB・mask /255潰れ)と境界
ラベルを修正した上での再学習の値(初版0.659と同方向)。train loss半減に
対しfold1 AUC ~0.66平坦=fold0過学習。blendもconf単独未満(0.7934)で
独立情報なし(相関0.51の劣化版)。+0.046 oracleを解放する分離情報は、
静的特徴(v64)・磁気図(v68)・画像CNN(v70)の
**3クラス全ての試行範囲で見つからなかった**。

留保: 2.6k cropsはCNN学習に小規模で過学習由来の可能性は排除しない。
大規模コーパス・IoU回帰・GONG自教師事前学習は残るが、conf 0.80との差を
覆す見通しは現時点の証拠では薄い。未試行手法の不存在証明ではない。

証跡: `scratch/build_verifier_dataset.py`、`scratch/train_verifier.py`
(生成物 `output/verifier_v1/` は gitignore)。

### 6.27 競合・ディスカッション再調査(2026-10-13)

§6.23 以降の差分。kaggle CLI の leaderboard/kernels 一覧と公開カーネル
本文から取得(discussion 本文は SPA 描画のため検索スニペット+引用経由)。

#### LB 変動と主催者動向

- チーム数 1016、我々は rank 148 @ 0.39
- **トップが 0.55 プラトーから 0.63 へ上昇**(Tempest 0.63・他 0.62x3、
  全て 10/8-14 の投稿)。メトリック修正(§733912)完了後の新規スコアなので
  死んだ旧メトリックの残滓ではなく、test/train 重複 77% の GT コピー系が
  最有力。主催者は「LB クリーンアップ」(discussion/740400)と
  「高スコア提出の精査」(discussion/747399)を表明 — 上位失格の蓋然性は
  依然として高い
- too.foo の dossier が文脈を再解釈: **修正前の 0.55-0.75 帯の一部は
  「再採点されない旧メトリック(Dice 誤設定、空提出で0.88-1.00)の亡霊」**
  で、漏洩だけが説明ではなかった。ただし修正後の新規 0.6+ は同じ結論
  (重複領域のコピー)に帰着する

#### dariushafshar「Two Experts Agree at 0.34」(10/13、最重要カーネル)

主催者配布の Self_Evaluation_Notebook を完全解像度で再実装した校正標準:

| 基準点 | 値 |
|---|---|
| 人間 vs 人間の PQ (annotator 相互、296枚) | **0.340** (SQ 0.634・IoU>0.5 一致率 54%) |
| 90枚 public half の SD | 0.0144 (同一予測でも ±0.03 揺れる) |
| public↔private 半々の差 | SD 0.0205 (±0.04) |
| 同一画像・別予測の差 | SD 0.0103 (±0.02) |
| 10/12 Public LB 分位 (972チーム) | median 0.34 / p75 0.37 / p90 0.40 |

- 「OOF で選べ」という我々の方法論全体を独立に裏付け。A1 の
  OOF+0.0034→LB-0.02 反転は ±0.04 の public/private ノイズ帯内の
  正常現象と確定的に説明できる
- PQ/2 損益分岐 P(match)xE[IoU|match] > 0.5xPQ を再確認
  (tushhaaarrrr 由来の同一則)。人間レベルでは <300px 除去は±0 (wash)
- 清浄マスクの連結成分化損失は -0.0008 のみ (instance head の価値小)
- 留保: 0.34 は天井ではなく「annotator 同意の物差し」。多数 annotator が
  引くものを学んだモデルは 0.34 を超え得る(我々の 0.39/0.44 と整合)

#### 誠実ティアの収束 — 3者が同型アーキに到達

| カーネル | LB | 構成 |
|---|---|---|
| magfilo_v9 (Muhammad S. Altaf, 12票) | 未明記 | YOLO11s@1536 → UNet++/EffB4 crop refiner → conf 順 disjoint |
| juanmirod Final Report (3票) | 0.37 | YOLO11m@1280 → resnet34 crop refiner (box-seed ch) → greedy disjoint |
| pandadev0069 (0.36) | 0.36 | YOLO26s-seg full-res 2048 → CC分割 |
| 我々 | 0.39 | fold2 YOLO11m@1280 → EdgeAttNet/plain-MHSA crop refiner → ownership |

detector→crop refiner→disjoint ownership の収斂、及び棄却リストの一致
(union target・ensemble・小フィラメント改善の難しさ)は独立3者で再現。
juanmirod は「faint barbs の FN」「plage 縁の FP」という我々と同じ
失敗モードを特定。我々の OOF 0.4442 は too.foo val 0.416・
tushhaaarrrr 0.4597 とともに誠実系の先端帯に位置。

#### 含意

- 戦略変更の根拠になる新手法の公開は**なし**。我々の再現性・4p レポート
  ・棄却列挙の厚さは依然として誠実ティア最強級の差別化点
- 未検証の micro-variant として唯一残った示唆: <300px 面積フロアの
  sweep (人間レベルでは wash、モデル精度では別途要 OOF — 優先度低)
