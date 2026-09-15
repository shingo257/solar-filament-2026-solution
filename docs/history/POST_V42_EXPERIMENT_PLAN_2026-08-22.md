# Post-V42 実験計画 — CPU-first Scientific S0

- 作成日: 2026-08-22
- 状態: active（S0-4盲検package生成完了、独立人手レビュー待ち）
- Competition正本: V35（全fold OOF PQ 0.442633、Public 0.37）
- V42判定: 不採用（全fold OOF PQ 0.439932、V35比-0.002701）
- 主トラック: Scientific / Operational S0
- 計算方針: ローカルCPUを既定とし、新しい予測生成が必要な段階だけGPUを使う

## 1. 計画の結論

次に行うべきなのは、別のYOLO設定や後処理閾値の探索ではない。既存のOOF成果物と
MAGFiLOのspine/chirality注釈を使い、V35が科学利用で何を壊しているかを定量化する。
S0で支配的な失敗モードと単一主仮説を確定するまでは、新規GPU学習を開始しない。

CompetitionトラックはV35で凍結する。同じ707 OOFを見ながらfold0のbestを再探索する
ことは、V42で観測した選択バイアスを強めるため行わない。

## 2. 2026-08-22ローカル監査で確認した入力

| 対象 | 確認結果 | 計画への意味 |
|---|---|---|
| COCO image records | 1,154 records / 707 unique observations | annotator variantをobservation単位に再結合する |
| annotations | 8,199 instances | 個体・形態指標を計算可能 |
| spine | annotation fieldとして存在 | spine coverage、centerline距離、clDiceを実装可能 |
| chirality | Left / Right / Unidentifiable / Ambiguousの4カテゴリ | segmentation-conditioned下流評価を設計可能 |
| time | `date_captured`あり | time-block splitを作成可能 |
| site | file name末尾にsite codeあり | code単位の層別・leave-one-site-out候補 |
| seeing/cloud metadata | 独立ファイルはローカルで未確認 | 現段階で気象・seeing頑健性を主張しない |
| V30/V35 YOLO OOF | fold0/1/2 = 236/236/235 NPZ | GPUなしでraw/V35を再構成可能 |
| V34 semantic OOF | fold0/1/2 = 236/236/235 PNG | topology probeをGPUなしで再評価可能 |
| V35 refiner | 3 fold checkpointすべて存在 | 全foldリークなし再評価が可能 |
| V42 seed8407 OOF | fold0/1/2すべて存在 | negative controlとして利用可能 |
| case diagnostics | 707観測のCSVあり | 層別・worst-case抽出を再利用可能 |

site codeを物理観測所名へ対応付ける場合は、MAGFiLO/GONGの一次資料で確認してから
記録する。ファイル名から推測した観測所名を確定値として使わない。

## 3. 検証する仮説

S0では次の仮説を順位付けするが、採用モデルはまだ決めない。

1. **H1: `shrink`がspineや接続性を切断している。** PQ改善と科学形態保存が競合している
   可能性をV35対raw YOLOで測る。
2. **H2: V34 semantic-firstはPQでは不利でもtopologyを保存する。** V34 semantic OOFを
   spine、接続成分、端点、Boundary F1で再評価する。
3. **H3: V42のRecall改善は科学カタログには有益だがFP増加が大きい。** V42をnegative
   controlとして、密集・低contrast層でRecall利得が局在するか確認する。
4. **H4: time/site domain shiftが平均値に隠れている。** 年代blockとsite code別の性能差を
   測り、ランダムgrouped CVだけでは見えない劣化を確認する。

## 4. 実行フェーズ

| ID | 内容 | 計算資源 | 主成果物 | 完了条件 |
|---|---|---|---|---|
| S0-0 | provenance freeze **完了** | CPU | `output/s0/provenance_manifest.json` | 重要ファイル・全9 cacheをcontent SHA-256で固定、missing 0 |
| S0-1 | annotation/time/site audit **完了** | CPU | `output/s0/annotation_audit.json`、層別・time block CSV | 707観測・8,199個体を監査、locked 160観測を固定 |
| S0-2 | science evaluator実装 **完了** | CPU | `src/science_metrics.py`、unit tests | maximum-cardinality matching、科学指標、bootstrapを回帰testで固定 |
| S0-3 | baseline rescore **完了** | CPU | `output/s0/science_*_v3.*`、S0正式報告 | 707観測でV35/raw/V34/V42とnested verifier正式PQを比較 |
| S0-4 | blinded visual review **package生成完了・人手判定待ち** | CPU + 人手、半日 | seed固定HTML、判定ログ | model名を隠した層別レビュー完了 |
| S0-5 | endpoint registration **構造draft済み・margin未固定** | CPU不要、半日 | `S1_ENDPOINT_REGISTRATION_DRAFT_2026-08-24.md` | 主endpoint、非劣性条件、holdoutを固定 |
| S1 | 単一モデル仮説の固定probe | GPU | checkpoint、OOF cache、manifest | 事前登録ゲートを満たす場合だけ全foldへ |

### S0-0 provenance freeze

V35、raw YOLO、V34 semantic、V42について、入力パス、観測数、fold、checkpoint、
metric version、生成スクリプト、SHA-256を1つのmanifestへ保存する。Driveの成果物を
暗黙の正本にせず、欠損時は`missing`として停止する。

### S0-1 annotation/time/site audit

次を観測単位・annotator variant単位で集計する。

- spine座標の偶数長、点数、画像範囲内率、mask内部率
- chirality 4カテゴリの分布とannotator間不一致
- `date_captured`のparse率、年代分布、近接時刻重複
- site codeの分布、年代との交絡、fold間偏り
- 面積、太陽中心からの距離、個体数、labeler数の分布

監査の結果、最後の連続年代blockである2017〜2022は160観測で6 siteすべてを含むため、
`output/s0/time_block_assignment.csv`へ将来S1用の`locked_evaluation`として固定した。
既存モデルのOOF層別値はretrospective分析であり、独立holdout性能とは呼ばない。

### S0-2 science evaluator

公式PQのHungarian対応を再利用し、matched instanceごとに以下を計算する。

- GT spine pointsが予測mask内に残る割合（spine coverage）
- 予測skeletonとGT spineのprecision / recall、clDice、対称centerline距離
- Boundary F1、面積・周長・主軸長・重心誤差
- 接続成分、端点、分枝数、split / merge、個体数誤差
- unmatched GT/予測を含むfilament Recall / Precisionと1画像あたりFP

空mask、1点spine、画像外座標、split、merge、完全一致、細線切断を合成testで固定する。
複数annotatorは平均で潰さず、variant別結果とobservation cluster bootstrapを併記する。

### S0-3 baseline rescore

V35、raw YOLO、V34 semantic、V42を同じ707観測・同じmatchingで評価する。V35/V42の
refiner出力は一度だけ生成してmask cacheへ保存し、後続分析で再推論しない。

比較は全体平均だけでなく、面積、低contrast、密集度、太陽縁距離、site code、年代、
annotator disagreementで層別する。V42は採用候補ではなく、RecallとFPのtrade-offを
理解するnegative controlとして扱う。

### S0-4 blinded visual review

固定seed `20260824`と保存した層別規則により、24の一意観測を選定した。比較は
(1) raw対V35のmatched-spine trade-off、(2) V35対V42のsmall/dense/limb救済、
(3) V42対nested verifierの2017〜2022候補棄却を各8件とし、各比較内のA/B位置を4/4にした。

`output/s0/blinded_review_v1/review.html`にはmodel対応表を含めない。形態保存、カタログ用途の
有害error、spine欠落、切断、merge、split、偽陽性、太陽縁artifact、確信度、メモを記録し、
localStorage自動保存と回答JSONのexport/importを備える。選択scoreと対応表は`private/`へ分離した。
回答JSONを書き出すまでは`private/`を開かず、review回答を学習labelへ自動exportしない。

package生成と完全性検証は完了したが、独立reviewerの24件判定は未実施である。したがって
S0-4の完了条件はまだ満たしていない。

### S0-5 endpoint registration

S0 baselineを確認後、S1の主要endpointを1つだけ選ぶ。最低改善幅、bootstrap CI、
filament Recall/PQ/chiralityの非劣性条件、locked time/site evaluationを文書へ固定する。
S1候補の結果を見る前にコミットし、以後変更しない。

## 5. S0結果から選ぶS1分岐

| S0で支配的な問題 | S1候補 | GPUの必要性 |
|---|---|---|
| shrinkによるspine切断 | topology-aware refiner、限定的外側復元 | refiner学習のみ |
| V34がtopology優位 | cached semantic guidance + V35 instanceのhybrid | CPU prototype後、必要なら学習 |
| V42が特定層だけRecall優位 | proposal confirmation / calibrated rejection | 小型確認器から開始 |
| time/site shift | domain-balanced sampling、site/time holdout学習 | detector再学習 |
| segmentationは十分でchiralityだけ弱い | mask/spine入力のchirality classifier | 軽量分類器 |

## 6. GPU実験の開始・停止条件

- S0-5で単一主仮説と評価コードが固定されるまでGPUを使わない。
- 最初のGPU runは1 foldまたは固定development blockのみ。1回に変更する仮説は1つ。
- 事前登録した最低改善幅を満たさなければ残りfoldへ進まない。
- 単一fold通過は採用ではない。全foldまたはlocked time/site evaluationで再現すること。
- Competitionへ戻す場合は、全fold OOF PQ `> 0.452633`と提出仕様検証を必須とする。
- Scientific S1ではPQ単独を採用理由にせず、主要endpoint改善と副次endpoint非劣性を必須とする。

## 7. 直近の実行推奨

次は`output/s0/blinded_review_v1/review.html`だけを開き、独立reviewerが24件を回答する。
回答JSONを書き出すまでは`private/`を開かない。全件回答後にのみunblindして比較別の選好、
error tag、低確信例を集計する。

その後のS0-5で、S1主仮説をpost-refiner candidate value + time/site robustnessへ固定する。
S0-4/S0-5完了前に新規GPU学習や同じ707 OOF上の追加閾値探索へ戻らない。

## 8. 2026-08-22 S0-0 / S0-1 実測結果

- provenance manifestを生成し、COCO JSON、fold CSV、V35/V42基準JSON、refiner 3 checkpointをcontent SHA-256で固定した。
- V30/V34/V42 OOF cacheは全9 directoryの236/236/235件を確認した。Google Drive仮想PNGのfull hashは停止したため、cacheはpath・size・suffix inventoryのSHA-256、重要JSON/CSV/checkpointはcontent SHA-256で固定した。manifestは`complete-inventory-hash`、missingは0。
- 707観測、1,154 image records、8,199 instances。全annotationがpolygonで、ID重複・未知image参照・fold欠落はいずれも0。
- spine座標は偶数長・有限・画像内の監査を全件通過。点数中央値10、polyline長中央値111.27 px、GT polygon内点率の平均0.9525。1 px許容では平均0.9961、全点が許容範囲内のannotationは97.30%。
- chiralityのinstance数はLeft 2,535、Right 2,590、Unidentifiable 3,074、Ambiguous 0。annotator構成が一致しない観測は280/707（39.60%）。
- annotator数は1名411観測、2名145、3名151。instance数がannotator間で異なる観測は256/707（36.21%）。この不一致層を平均で潰さない。
- timestamp parse失敗、variant間timestamp不一致、異なる観測間の同一timestamp衝突はいずれも0。2017〜2022の160観測（22.63%）をlocked evaluationに固定した。
- site code Bh/Ch/Lh/Mh/Th/Uhは各102〜127観測。locked blockにも15〜36観測ずつ存在する。物理観測所名は一次資料で確認するまで未確定。
- bbox中心の画像中心からの正規化距離は中央値0.5120。これは太陽円盤を校正したheliographic距離ではなく、層別用のimage-center proxyである。

成果物: `output/s0/provenance_manifest.json`、`annotation_audit.json`、`observation_strata.csv`、
`time_block_assignment.csv`、`annotation_spine_audit.csv`。

## 9. 2026-08-24 S0-2 / S0-3 実測結果

- science evaluatorと707観測rescoreを完了した。詳細・数値・SHA-256は
  `S0_SCIENCE_RESCORE_REPORT_2026-08-24.md`を正本とする。
- V35はrawより全spine coverage +0.02749、filament Recall +0.03672、Precision +0.05453。
  一方rawはmatched spine coverage +0.01298、clDice +0.01074で、`shrink`のtrade-offを確認した。
- V42はV35より全spine coverage +0.01404、Recall +0.02053だが、matched spine coverage
  -0.00687、Precision -0.02927、merge +0.04939。利得はsmall/dense/limbへ局在した。
- V35のdense層はspine coverage 0.5152、Recall 0.5602、count error 3.725、merge 0.536。
  small層はcoverage 0.5468、Recall 0.5839で、S1の優先失敗モードとする。
- nested verifierは5,444 proposalから3,860を受理し、候補代理効用をaccept-all比18.2%改善した。
  consensus-only対比でも観測cluster bootstrapの95% CIは正だった。
- しかしrefiner・ownership後のofficial PQは0.440303。V42 accept-all 0.439932より+0.000371、
  V35 0.442633より-0.002330で、V35置換・formal gateとも不合格。FP-1,284と同時にTP-663、
  FN+663となり、候補単位代理効用がfull pipeline目的を表せていなかった。
- verifierのretrospective time層はdevelopment PQ 0.4427 / accepted 71.6%に対し、
  2017〜2022 PQ 0.4176 / accepted 65.1%。独立holdoutではないがdomain-shift兆候として残す。

### 次の主仮説

V42全体置換や追加threshold sweepではなく、small/dense/limbで有用な候補を
**post-refiner full-pipeline marginal value**で確認し、time/site shift時に過剰棄却しない仕組みを試す。
主要endpoint候補は全spine coverage。filament Precision / Recall、merge、matched spine coverage、
official PQを副次非劣性endpointとする。数値marginはblinded review後、S1結果を見る前に登録する。

## 10. 2026-08-24 S0-4 package生成結果

- 生成script: `scratch/run_s0_blinded_review.py`
- package: `output/s0/blinded_review_v1/`
- 24一意観測、3比較各8件、A/B位置は各比較4/4
- 6 panel PNG 24枚（full reference/A/B + disagreement zoom reference/A/B）
- 公開manifest 24行、PNG欠損0、SHA-256不一致0、公開HTMLのmodel identity漏えい0
- 入力、A/B、checkbox、確信度、メモ、localStorage、JSON export/importをunit testで固定
- status: `awaiting independent blinded human review`

独立reviewerの回答がないため、model優劣の定性結論、S0-4完了、S0-5 margin確定はまだ主張しない。

public-only readiness監査では24/24 PNGが同一寸法でdecodeし、near-blank・hash不一致・model漏えい0。
ただしlocked層14/24（corpus 160/707、representation ratio 2.58）とfold1 11/24の濃縮がある。
review結果は失敗モード判定に使い、母集団prevalenceや独立性能推定には使わない。

`scratch/analyze_s0_blinded_review.py`はReviewer ID・24 case・全必須回答を公開manifestだけで
検証し、不完全ならprivate keyを読む前に停止する。S1事前登録の構造はdraft化したが、主要改善幅、
非劣性、受理率、worst-site、latency marginはS0-4完了まで`TBD`である。

## 11. 2026-08-24 S1 margin calibration

S1結果を一切使わず、既存707観測だけでendpointの観測間SD、複数annotator再現性、locked 160件の
power/MDEを計算した。annotator variantはobservation内で先に平均し、観測を等重みとした。

主要all-spine coverageのlocked V35平均は0.5845（bootstrap 95% CI 0.5323〜0.6366）。V42−V35の
paired SD 0.1647を用いると、marginを越える80%/90% MDEは0.0324/0.0381。margin 0で改善0.01、
0.02、0.03のpowerは19.0%、45.7%、74.5%、80% power必要数は1,678、420、187観測である。

locked 160件ではV42平均利得+0.0133程度を確証できない。したがって小さい改善を「差なし」で棄却せず、
意味のある効果を狙う、事前固定stratumでpaired varianceを下げる、追加独立データを得る、または
探索的主張へ限定する。正式値は`../s1/S1_ENDPOINT_MARGIN_CALIBRATION_2026-08-24.md`を参照する。
marginは引き続き`TBD`で、S0-4完了と現場許容errorの入力後、S1結果を見る前にのみ固定する。

## 12. 2026-08-24 S1 conditional rescue headroom

V35/V42を画像単位で条件付き選択する上限と実装可能性を、既存707 OOF・ローカルCPUだけで診断した。
annotator variantはobservation/model内で先に平均し、development 547件の3-fold OOFだけでgate学習と
閾値選択を行った。locked 160件は過去に閲覧済みなのでretrospective転移診断に限定した。

- V42 allのlocked coverage差は+0.01326（95% CI -0.01248〜+0.03922）。
- GT oracle primaryはlocked +0.03409（MDEの1.053倍）。
- Precision/merge/count errorもcase単位で非悪化とするoracleはlocked +0.02831（MDEの0.875倍）。
- prediction count/componentsでcoverage改善を予測するdevelopment OOF AUCは0.765。
- 27個のGT-free mask geometryを足してもAUC 0.780で、改善は小さい。
- 選択率10%以上でPrecision/merge/count errorを同時に悪化させない閾値は全8構成で0。
- catalog-safe caseを直接targetにするとdevelopment OOF AUCは0.44〜0.50で識別不能。

したがって画像単位conditional rescueをS1候補から除外する。次のGPU実験を正当化するのは、
S0-4完了後に次をすべて満たすproposal-level設計だけである。

1. post-refiner/ownership後の最終pipeline marginal utilityをtargetとする。
2. coverage単独でなくPrecision、merge、count errorを同時に扱う。
3. detector confidence、seed support、refiner calibration、局所重複/形状を推論特徴に含める。
4. development nested OOFで閾値とmarginを固定し、lockedを再選択に使わない。
5. 新しい独立time/site blockを確保できなければ探索的結論に限定する。

正式報告は`../s1/S1_CONDITIONAL_RESCUE_HEADROOM_REPORT_2026-08-24.md`。S1 prediction、blind review response、
private keyは未使用で、marginは未登録のままである。

## 13. 2026-08-24 proposal-level marginal utility CPU pilot

S1候補を学習する前に、post-refiner/ownership後の候補別教師をローカルCPUで生成できるかを固定pilotで確認した。
development各foldから結果非依存hashで4観測ずつ選び、計12観測・137候補を評価した。fold別OOF refinerを
全候補へ一度適用してcacheし、1候補ずつ除いたfinal pipelineとの差を測った。locked行は0件で、S1 prediction、
blind review response、private key、閾値選択は未使用である。

- pre-refiner proxy対final PQ正利得: TP 80 / FP 7 / FN 8 / TN 42、precision 91.95%、recall 90.91%。
- 連続proxy効用対final PQ効用: Spearman ρ=0.766。
- 既存11特徴verifier対final PQ正利得: precision 77.42%、recall 81.82%。
- proxy対catalog-safe: precision 63.22%、recall 98.21%。
- 既存verifier対catalog-safe: precision 51.61%、recall 85.71%。
- proxy-positive 87件中、count error悪化22件、spine非改善8件、Precision悪化4件、merge悪化1件
  （失敗理由は重複あり）。

実測377.7秒から、development全547観測・4,899候補の教師生成は3.75〜4.78時間と見積もる。
したがってfull-pipeline教師生成はCPUで実行可能だが、pilotの小標本値は性能主張・S1登録・閾値固定に使わない。
またleave-one-out候補効果はownership相互作用があるため加算しない。

次の候補設計は、単一のbenefit scoreを閾値処理する既存verifierではなく、以下のharm-first multi-taskとする。

1. final PQ効用とspine回収をbenefit targetとして分ける。
2. count error、Precision、mergeの悪化を独立harm targetとして学ぶ。
3. development nested OOFで、最低選択率と全harm非劣性を同時に評価する。
4. S0-4独立reviewと現場許容errorからmarginを固定するまで、全development探索とGPU学習を保留する。
5. locked 160件を再選択へ使わず、新規time/site blockがなければ最終結論を探索的とする。

正式報告は`../s1/S1_PROPOSAL_MARGINAL_UTILITY_PILOT_REPORT_2026-08-24.md`。

## 14. 2026-08-24 harm-target label feasibility

固定12観測・137候補のfinal差分だけを使い、count / Precision / merge害が独立headを設計できる量かを
ローカルCPUで監査した。モデル学習、feature選択、threshold sweep、locked、review responseは未使用。

| harm | 候補events | event観測 | 上位3観測への集中 | 4,899候補へのcluster bootstrap計画範囲 |
|---|---:|---:|---:|---:|
| count | 39 | 6/12 | 76.9% | 358〜2,690 |
| Precision | 44 | 12/12 | 38.6% | 1,217〜1,991 |
| merge | 7 | 5/12 | 71.4% | 77〜459 |

count×mergeの候補Jaccardは0.045で、害targetは互換ではない。候補random splitと単一`any_harm`最適化を
禁止し、次の事前仕様を追加する。

1. benefit、count、Precision、mergeを別targetとして保持する。
2. proposal総数、ownership後output数、候補除外時output count差、重複競合、観測内rank・近傍密度を
   GT-free observation/set contextとして追加する。
3. 学習、calibration、outer evaluationをobservation grouped nested OOFで行う。
4. 各headは全label生成後に最低10 events / effective fitted degree of freedomを満たす容量に制限する。
5. merge labelが不足する場合はstandalone headの性能を主張せず、final merge非劣性gateを必須のまま残す。
6. candidate classifier指標だけで採用せず、選択集合をownershipへ戻した最終pipeline endpointで判断する。

害0件の独立採用候補だけで片側95%上限を<5% / <2% / <1%にする楽観的下限は59 / 149 / 299件。
実際は観測内相関があるため、独立観測数、time/site、GT密度を別途固定する。

正式報告は`../s1/S1_HARM_TARGET_FEASIBILITY_REPORT_2026-08-24.md`。この結果はlabel設計のfeasibilityであり、
S1性能・feature・threshold・marginの登録ではない。S0-4完了前に全development探索へ進まない。

## 15. 2026-08-24 GT-free set-context feature contract

harm-first model fitting前に、固定12観測cacheから推論時に計算できる観測・競合特徴を実装した。
candidate tableからはidentifierと既存11 inference featureだけをwhitelistで読み、target/IoU/marginal/verifier列、
GT、locked、review responseは読み込まない。

- 137候補、既存11＋context 29＝40 features。欠損・非有限値0。
- frozen ownershipのfull/without output countは既存pilotと137/137一致。
- outputへ残る125候補、containment drop 12、overlap/containment conflict 22。
- `output_count_delta`: +1が115、0が20、-1が2。
- 再実行feature CSV SHA-256一致。

`-1`の2候補は、自身がoutputへ残りながら小候補を包含除去し、総output数を1減らした。candidate-only scoreでは
この相互作用を扱えないため、以下をdownstream contractとする。

1. proposal/output数、候補除外時count差、観測内z/rank、refined area、IoU/包含、ownership traceを使う。
2. site/year/time-block/selection hashは監査metadataとし、予測featureへ入れない。
3. split、scaling、calibration、outer evaluationをobservation groupedで行う。
4. feature ablationは事前固定family単位とし、OOF結果を見ながら単一featureを逐次選ばない。
5. candidate score後に採用集合へfrozen ownershipを再適用し、final endpointで判定する。
6. 固定pilotで定数のmin-area dropをpilotだけで削除せず、全development生成後に事前規則で扱う。

正式契約は`../s1/S1_SET_CONTEXT_FEATURE_CONTRACT_2026-08-24.md`。feature-target関連、model、threshold、marginは
未評価で、S0-4完了前に全development探索へ進まない。

## 16. 2026-08-24 observation-grouped nested split registry

label/feature結合前に、547 development観測・4,899候補のouter/inner境界をtarget非依存で固定した。
candidate tableからはfile/fold/indexだけを読み、target、utility、feature値、lockedは未使用。

| outer test | test観測/候補 | train観測/候補 | inner validation観測 | inner validation候補 |
|---:|---:|---:|---:|---:|
| 0 | 183 / 1,672 | 364 / 3,227 | 117〜124 | 1,074〜1,077 |
| 1 | 184 / 1,535 | 363 / 3,364 | 119〜125 | 1,121〜1,122 |
| 2 | 180 / 1,692 | 367 / 3,207 | 119〜127 | 1,067〜1,071 |

全9 inner validationで36 site×yearセルを保持し、候補数rangeは1〜4件。locked混入、outer cycle内重複、
outer testのinner割当はいずれも0。registryは同一入力で決定的に再生成できた。

downstream手順を次に固定する。

1. outer testを隔離し、outer trainingの3 inner foldでscaler/model/calibration/policyを選ぶ。
2. inner validation以外の2 inner foldだけを各inner fitに使う。
3. 選択済み仕様をouter training全体で再fitし、outer testを1回だけ予測する。
4. candidate予測を観測集合へ戻し、ownership後endpointを評価する。
5. target imbalanceを見てもregistryを変更せず、weighting・CI・容量で扱う。
6. site/yearはbalance/report専用で、予測featureにしない。

registry SHA-256は`48b69680ac40a411b65cdc8cd9246a475ca88002f8483bae3acf2f31a9db2219`。
正式契約は`../s1/S1_GROUPED_NESTED_SPLIT_CONTRACT_2026-08-24.md`。S0-4/margin前のmodel fittingは始めない。

## 17. 2026-08-24 harm-first primary model spec and execution guard

full development targetを見る前に、一次解析の構造を次へ固定した。

- estimator: fold内`StandardScaler`＋L2 logistic regression、class weight `balanced`。
- heads: final PQ benefit、spine benefit、count harm、Precision harm、merge harmの5個。
- features: headごとに機構から選んだ6個へ固定。site/year/fold/target/verifier出力は禁止。
- capacity selection: C 0.01/0.1/1/10をinner OOF observation-equal weighted log-lossで選ぶ。
- event gate: 6 features＋interceptに対しouter trainで70 candidate eventsかつ30 event観測。
- final decision: candidate classifier指標ではなく、固定ownership後の科学・catalog endpoint。

event不足headは独立性能を主張せず、operational promotionをblockする。特にmerge headが不足してもfinal merge
非劣性gateは外さず、別threshold探索で救済しない。benefit/harm確率thresholdと最低選択率は、S0-4と現場許容
errorからmarginを登録するまで`TBD`のまま保持する。

CPU validatorはmodel spec、feature contract hash、split registry hashを検証し、構造error 0を確認した。一方、
statusは`structure_ready_execution_blocked`、`ready_to_fit=false`で、blockerは次の5件である。

1. 24件のS0-4 responseが未完成。
2. endpoint marginが未登録。
3. decision probability policyと登録provenanceが未固定。
4. full development feature 4,899行が未生成。
5. 5 binary head列を持つfull development marginal label 4,899行が未生成。

validatorはprivate key、locked row、target値を読まず、model fitting・threshold selectionを行わない。正本は
`S1_HARM_FIRST_MODEL_SPEC_2026-08-24.json`、実測報告は
`../s1/S1_HARM_FIRST_MODEL_SPEC_REPORT_2026-08-24.md`とする。

## 18. 2026-08-24 nested OOF execution plan and preflight

固定spec/registry/readinessだけから、target非読込で全compute unitを列挙した。

| phase | jobs |
|---|---:|
| inner selection: 3 outer×5 heads×4 C×3 inner | 180 |
| outer refit and score once | 15 |
| model fit total | 195 |
| frozen policy＋ownership replay | 3 |
| threshold fit | 0 |

各jobへtraining/validation observation・candidate countとobservation hashを保存し、inner training＋validationが
outer trainingと一致することを確認した。registry hash不一致、source fold改変、outer leakage、cycle内重複を
拒否する。

preflight設計中に、(1) endpoint margin登録後もdecision policy未登録でfitできる余地、(2) label CSVがidentity
3列だけでも4,899行なら完成扱いになる余地を確認した。spec/readiness schemaをversion 2へ更新し、policyの
target読込前provenanceと5 binary head列を開始gateへ追加した。

実測statusは`execution_plan_frozen_fit_blocked`、`fit_start_permitted=false`。通常plan生成は成功し、
`--require-fit-ready`は非ゼロ終了してfitを拒否した。feature値、target値、review response、private key、locked
rowは未読。正式報告は`../s1/S1_NESTED_OOF_EXECUTION_PLAN_REPORT_2026-08-24.md`。

## 19. 2026-08-24 decision-policy registration transaction

policyをmanual editで登録せず、次のpublic evidenceを検証するtransactionを追加した。

1. complete S0-4 response 24件と実file SHA-256。
2. 全margin登録済みendpoint documentと実file SHA-256。
3. public rationale document pathとSHA-256。
4. source model spec SHA-256、登録者、timezone付き登録時刻。
5. benefit/harm probability 5個とminimum selection rate。
6. target未読、review用途限定、endpoint margin優先の3 attestations。

full development target label fileが存在する場合は、readiness表示にかかわらず登録を拒否する。public response hash
不一致、private配下rationale、範囲外threshold、timezoneなし、attestation欠損も拒否する。

合格時だけ`registered_structure_frozen_ready_for_execution`のspec candidateを別pathへ書き、既存validatorで
構造再検証する。canonical specと同じ出力pathは禁止し、自動昇格しない。

synthetic complete caseは合格した。実状態ではpublic response不存在、endpoint `TBD` 13、templateの担当・時刻・
rationale・6 threshold・3 attestationが未記入なので`registration_blocked`。candidateは未生成で、
`--require-registered`も非ゼロ終了した。正式報告は
`../s1/S1_DECISION_POLICY_REGISTRATION_TRANSACTION_REPORT_2026-08-24.md`。
