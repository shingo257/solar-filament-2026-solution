# S1科学・実運用endpoint事前登録 draft

- 作成日: 2026-08-24
- status: `draft_blocked_on_completed_S0-4_review_and_full_development_proposal_registration`
- track: Scientific / Operational
- Competition正本: V35（変更しない）
- S1 baseline: V35-Competitionの固定OOF/full pipeline
- 禁止: このdraftの未確定marginを埋める前のGPU学習、locked層を用いた学習・閾値選択

## 1. 単一主仮説

small / dense / limb条件でV42由来の候補を一律採用・棄却せず、**post-refiner full-pipelineでの
科学的価値とtime/site頑健性を直接学ぶ条件付きproposal confirmation**は、V35より全spine coverageを
改善し、カタログ上有害なfalse positive / mergeと後期年代での過剰棄却を事前登録margin内に保つ。

これはV42全体置換、追加WBF/conf/ownership閾値探索、pre-refiner IoU代理効用の再最適化ではない。

## 2. 対象と分割

- observation単位でsplitし、同一observationの全annotator variantを同一splitへ固定する。
- development: 2011〜2016の547観測。学習、閾値固定、ablationはこの範囲だけで行う。
- locked evaluation: 2017〜2022の160観測。学習、early stopping、閾値・margin選択に使わない。
- site code Bh / Ch / Lh / Mh / Th / Uhをすべて報告し、物理観測所名は一次資料確認まで推測しない。
- 既存V35/V42のlocked層OOF値はretrospective診断であり、新S1の独立holdout結果とは区別する。

## 3. 主要endpoint

**observation-clustered all-spine coverage**を主要endpoint候補とする。未検出filamentのspine点を0として
含み、annotator variantを平均で消さずに評価する。比較はS1対固定V35のpaired differenceとし、
observation cluster bootstrap 95% CIを報告する。

- 最小改善幅 `delta_primary_min`: `TBD_AFTER_S0_4_BEFORE_S1`
- 合格条件: paired differenceの95% CI下限が`delta_primary_min`以上
- 主要解析対象: locked evaluation 160観測
- development結果は設計・sanity checkとして別表にし、正式合格判定と混ぜない

### 3.1 S1前margin calibration

`docs/s1/S1_ENDPOINT_MARGIN_CALIBRATION_2026-08-24.md`で、S1 predictionを見ずに既存707観測を再集計した。
locked 160件のall-spine coverageについて、V42−V35 paired SDは0.16470、登録境界を越える80% MDEは
0.03238、90% MDEは0.03810である。marginを0とした近似powerは、真の改善0.01で19.0%、0.02で
45.7%、0.03で74.5%。80% power必要数はそれぞれ1,678、420、187観測である。

この結果から次を登録契約へ追加する。

- `delta_primary_min`は「検出できるから」ではなく、現場で意味のあるspine回収量として決める。
- true effectは、登録marginに加えて約0.0324を越えないとlocked 160件で80% powerを得にくい。
- 小効果を対象にする場合、事前固定small/dense/limb解析、追加独立観測、探索的結論のどれかを選ぶ。
- review sampleの選好率、annotator pair差、normal近似MDEのいずれか単独でmarginを決めない。
- 最終判定はnormal近似ではなくobservation-cluster bootstrap CIを用いる。

calibrationはmargin候補を狭める資料であり、`delta_primary_min`は引き続き`TBD`とする。

### 3.2 conditional rescue headroomによる設計制約

既存V35/V42を画像単位で切り替えるCPU診断では、GT oracle primaryのlocked coverage上限は+0.03409、
catalog-safe oracleは+0.02831だった。一方、予測個数・連結成分・mask geometry・year/siteを使う
8構成は、development OOFで選択率10%以上かつPrecision/merge/count error同時非悪化を満たす閾値が
0だった。catalog-safe caseを直接targetにしたdevelopment OOF AUCも0.44〜0.50だった。

このnegative resultにより、**画像単位conditional rescueをS1実装候補から除外する。** S1は
post-refiner proposal単位のconfirmationだけを対象とし、coverage単独ではなくPrecision/merge/countを
含む最終pipeline marginal utilityをtargetとする。既存V42 candidate verifierのpre-refiner代理効用を
再利用して合格扱いにしない。詳細は`docs/s1/S1_CONDITIONAL_RESCUE_HEADROOM_REPORT_2026-08-24.md`。

### 3.3 proposal-level教師生成pilotによる登録制約

developmentだけの固定12観測・137候補で、fold別OOF refiner後のleave-one-proposal-out効用をCPU生成した。
pre-refiner proxy対final PQ正利得はprecision 91.95%、recall 90.91%、連続効用のSpearman ρ=0.766で、
最終pipeline教師の生成可能性を確認した。一方、既存11特徴verifierのcatalog-safe precisionは51.61%、
proxy-positive 87件中22件でcount errorが悪化した。

このpilotから登録契約へ次を追加する。

- 単一benefit targetを直接acceptanceへ変換しない。final PQ/spine利益とcount / Precision / merge害を
  個別targetとして、harm-firstに採否する。
- 全developmentの教師生成、feature/model比較、閾値選択はnested OOF内で完結させる。
- leave-one-out効用を足し上げた期待改善値を報告しない。候補間ownership相互作用をfinal pipelineで再評価する。
- 固定12観測pilotのprecision/recall/相関は実現可能性指標であり、S1性能、margin、閾値ではない。
- S0-4独立reviewと現場許容errorが得られるまで全development探索を開始せず、locked 160件は再選択に使わない。

正式値・sample contract・CPU時間は`docs/s1/S1_PROPOSAL_MARGINAL_UTILITY_PILOT_REPORT_2026-08-24.md`を参照する。
この証拠によりproposal-level evidenceの「生成可能性」は満たしたが、S1登録はS0-4完了、harm margin固定、
development nested OOFの事前仕様化までblockedのままとする。

### 3.4 harm-target feasibilityによる登録制約

固定137候補ではcount harm 39件・6観測、Precision harm 44件・12観測、merge harm 7件・5観測だった。
count/merge害は上位3観測に76.9%/71.4%が集中し、count×mergeのJaccardは0.045。候補を独立sampleと
みなすrandom splitや、全害を単一targetへ圧縮する設計は登録しない。

S1 feature/validation contractへ次を追加する。

- GT-freeのobservation/set contextとして、proposal総数、ownership後output数、候補除外時output count差、
  重複競合、観測内rank・近傍密度を候補特徴へ含める。
- benefit、count harm、Precision harm、merge harmを別targetとし、harm-firstに採否する。
- split、calibration、outer evaluationの全段でobservation groupを維持する。
- 各headのeffective fitted degree of freedomは、全developmentで少なくとも10 events / degree of freedomを
  満たす範囲に固定する。満たさないheadの独立性能を主張しない。
- 候補scoreのAUC/Precisionだけで採用せず、選択集合をownershipへ戻し、§4の全endpointを再評価する。
- merge headのlabelが不足してもmerge非劣性gateは解除せず、制御不能ならS1を不合格とする。

運用acceptanceでは、害0件でも独立採用候補59 / 149 / 299件がそれぞれharm rate <5% / <2% / <1%の
片側95%上限に必要な楽観的下限である。候補は観測内相関を持つので、登録時には独立観測数、time/site、
GT密度を別条件として加える。

cluster bootstrapによる全development harm event数は計画用であり、model性能・閾値の根拠にしない。
正式値は`docs/s1/S1_HARM_TARGET_FEASIBILITY_REPORT_2026-08-24.md`。

### 3.5 GT-free set-context feature contract

固定137候補で既存11 inference feature＋set-context 29特徴をtarget非依存に生成し、欠損/非有限値0、
ownership full/without countのarchived pilotとの不一致0を確認した。`output_count_delta=-1`の2候補では、
候補追加が包含競合を通じてoutput総数を減らし、count害に必要な集合相互作用をGTなしで表現できた。

S1 feature contractへ次を登録する。

- base featureは既存11 inference feature、contextはproposal/output数、候補除外時count差、観測内z/rank、
  refined面積、pairwise IoU/包含、競合数、ownership保持/排他面積/順序の固定familyとする。
- `site_code`、`year`、time-block role、selection hashはstratification metadataであり、予測featureにしない。
- target、IoU、marginal utility、既存verifier出力をfeature tableへ混入させない。
- scaler、imputer、model容量、calibrationは各inner-developmentだけでfitする。
- feature ablationはfamily単位で事前指定し、単一featureの逐次OOF選択を禁止する。
- candidate score後に選択集合へ固定ownershipを再適用し、§4 endpointで最終判定する。
- 固定pilot内の定数/低頻度だけを理由にfeatureを削除しない。全development生成後の処理規則をS1結果前に固定する。

feature contractの正本は`docs/s1/S1_SET_CONTEXT_FEATURE_CONTRACT_2026-08-24.md`。この実装完了はモデル開始条件を
満たすものではなく、S0-4独立review、harm margin、nested OOF仕様が未登録のためblockedを維持する。

### 3.6 observation-grouped nested split registry

label join前にdevelopment 547観測・4,899候補のregistryを固定した。frozen source foldをouter testとし、
各outer-trainingをsite×yearと予測候補数で均衡化した3 inner foldへ分割する。registry SHA-256は
`48b69680ac40a411b65cdc8cd9246a475ca88002f8483bae3acf2f31a9db2219`。

登録契約へ次を追加する。

- 同一observationの全candidateを同じouter role / inner foldへ固定する。
- 各observationはouter testへ正確に1回だけ入り、outer testをinner tuningへ使わない。
- scaler、imputer、model容量、class weight、calibration、harm-first policyは各outer-trainingのinner OOFだけで決める。
- inner modelはinner validation以外の2 foldだけでfitし、outer testは仕様固定後に1回だけ予測する。
- target prevalenceの不均衡を見てもregistryを再生成しない。weightingとreport CIはinner training内で決める。
- site/yearはsplit balanceとstratified reportだけに使い、予測featureへ入れない。
- candidate出力を観測集合へ戻し、固定ownership後の§4 endpointでouter採否する。

全9 inner validationは36/36 site×yearセルを保持し、候補数max−minはouter cycleごとに1〜4件。
locked 160観測のregistry混入は0。source foldは新しいtime/site holdoutではないため、新規独立blockがなければ
outer OOF結果も探索的と明記する。

split contractの正本は`docs/s1/S1_GROUPED_NESTED_SPLIT_CONTRACT_2026-08-24.md`。nested OOF境界は固定できたが、
S0-4 reviewとmarginが未完了なのでS1 model開始はblockedのままとする。

### 3.7 harm-first primary model structure and execution guard

full development labelを見る前に、一次モデルを次へ固定した。

- L2 logistic regression、fold内`StandardScaler`、class weight `balanced`、追加calibrationなし。
- C候補0.01/0.1/1/10をinner OOFのobservation-equal weighted binary log-lossで選ぶ。
- final PQ/spine benefit、count/Precision/merge harmの5 headを分離する。
- 各headは事前固定6特徴＋intercept。site/year/time-block/fold、target/utility、verifier出力を禁止する。
- 各outer trainingで最低70 candidate eventsかつ30 event観測を要求する。
- 不足headはoperational promotionをblockし、threshold探索や単一`any_harm`で代替しない。
- candidate metricだけで採用せず、固定ownership後の§4 endpointを再計算する。

確率thresholdと最低選択率は`TBD_AFTER_S0_4_AND_MARGIN_REGISTRATION`とし、S0-4結果・現場ownerの許容error・
§4 margin登録後、model結果を見る前に固定する。

decision policy登録には、登録者・時刻・rationale document、public review response SHA-256、本registration
SHA-256、target値を読む前に登録した証明を必須とする。

実行開始には次をすべて要求する。

1. public S0-4 response 24件の完全回答。
2. 本draftの全endpoint margin登録。
3. decision probability policyとprovenanceのtarget読込前登録。
4. full development feature 4,899行。
5. `final_pq_benefit`、`spine_benefit`、`count_harm`、`precision_harm`、`merge_harm`の5 binary列を持つ
   full development marginal label 4,899行。

CPU validatorではmodel/feature/split構造は合格したが、上記5条件が未完了のため
`structure_ready_execution_blocked`、`ready_to_fit=false`。private key、locked row、target値は未使用である。
正本は`S1_HARM_FIRST_MODEL_SPEC_2026-08-24.json`、実測報告は
`docs/s1/S1_HARM_FIRST_MODEL_SPEC_REPORT_2026-08-24.md`。

### 3.8 nested OOF execution plan

登録後の実行順序もmodel結果を見る前に固定する。

- inner selection: 3 outer×5 heads×4 C×3 inner＝180 fit。
- outer refit: 選択Cで3 outer×5 heads＝15 fit。
- model fit total: 195。threshold fitは0。
- outer testはheadごとに1回だけscoreする。
- 登録済みpolicy適用後、候補集合をfrozen ownershipへ戻して§4 endpointを評価する。

現在のexecution planは`fit_start_permitted=false`で、fit-ready必須preflightは非ゼロ終了する。正式値は
`docs/s1/S1_NESTED_OOF_EXECUTION_PLAN_REPORT_2026-08-24.md`。

### 3.9 decision-policy registration transaction

本draftのmargin固定後、`S1_DECISION_POLICY_REGISTRATION_TEMPLATE_2026-08-24.json`へ6 threshold、登録者、
timezone付き時刻、public rationale、3 attestationを記入する。transactionは次を検証する。

- complete public responseと本draftの実file hashがreadiness記録と一致する。
- full development target labelがまだ存在しない。
- rationaleがpublic pathにあり、SHA-256を保存できる。
- thresholdが確率範囲内、minimum selection rateが0より大きい。
- target未読、review用途限定、endpoint margin優先を明示している。

合格時もcanonical specを直接変えず、registered candidateを別artifactへ生成して構造再検証する。現在は
`registration_blocked`でcandidate未生成。正式値は
`docs/s1/S1_DECISION_POLICY_REGISTRATION_TRANSACTION_REPORT_2026-08-24.md`。

## 4. 副次非劣性endpoint

| endpoint | 意味 | margin | 判定方向 |
|---|---|---|---|
| filament Recall | 未検出を含む個体回収 | `TBD` | S1−V35がmargin以上 |
| filament Precision | 余剰候補抑制 | `TBD` | S1−V35がmargin以上 |
| merge rate | 複数個体の癒着 | `TBD` | S1−V35がmargin以下 |
| matched spine coverage | 対応済み個体の細構造保持 | `TBD` | S1−V35がmargin以上 |
| official PQ / SQ / RQ | Competition互換性 | `TBD` | PQ非劣性、SQ/RQ併記 |
| count absolute error | カタログ個体数 | `TBD` | S1−V35がmargin以下 |

数値marginはS0-4完了後、S1の予測・学習結果を見る前に固定する。S0-4の24件は仮説濃縮sampleであり、
その単純選好率だけからmarginを決めない。V35の707観測bootstrap分布、measurement repeatability、
実運用上許容できるerrorを併用する。

## 5. time/site・棄却の運用endpoint

- locked層のcoverage–rejection曲線とaccepted fraction
- development対lockedの受理率差
- site別主要endpointとworst-site値
- small / dense / limb proxy別のRecall、Precision、spine coverage
- 低信頼棄却で失われるtrue filament数と、除去されるfalse positive数
- 入力異常時のfail-safe率、平均/p95 latency、peak RAM/VRAM

次の値をS1前に固定する。

- locked受理率低下の最大許容幅: `TBD`
- worst-site主要endpointの非劣性margin: `TBD`
- 推論p95 latency上限: `TBD_WITH_TARGET_HARDWARE`
- fail-safe対象入力と期待動作: `TBD_WITH_OPERATION_OWNER`

## 6. S0-4からS1仮説へのdecision rule

1. V42側が形態保存で選ばれ、カタログ品質でV35側が選ばれるcaseが主なら、条件付きrescueを継続する。
2. V42側が形態・カタログの両方で選ばれる特定stratumがある場合、そのstratumだけを事前固定する。
3. verifier側が後期年代で繰り返し選ばれない場合、全体棄却器を中止し、calibration/domain-aware gatingを検討する。
4. tie / neither / 低確信が多い場合、S1学習へ進まずreference adjudicationまたは評価UIを改善する。
5. review回答をcandidate labelやtraining targetへ直接変換しない。

## 7. 統計・報告契約

- observationをcluster単位としたpaired bootstrapを使う。
- annotator variant数の多いobservationが過大加重されない集計を主解析とする。
- overallだけでなくarea / density / radial / time / site / annotator disagreementを併記する。
- missing / decode failure / abstention / low confidenceを除外せず件数と理由を報告する。
- 主要endpoint、margin、対象split、code hashをS1結果生成前にGitへ固定する。
- multiple secondary endpointsは探索的と明記し、主要endpoint不合格を副次指標で覆さない。

## 8. 昇格・停止条件

S1を科学・実運用候補へ昇格するには、主要endpoint合格、副次非劣性、locked time blockでの再現、
site worst-case監査、目視worst-case監査、latency/fail-safe記録をすべて満たす必要がある。

次の場合は停止する。

- 主要endpointの95% CI下限が事前登録値未満
- Precision / merge / official PQのいずれかが非劣性違反
- proposal-level post-refiner marginal utilityをdevelopment nested OOFで示せない
- 画像単位gate、coverage単独target、またはlocked層の再探索へ戻る
- locked層または特定siteで過剰棄却が再発
- 改善がannotation disagreementやreference漏れだけで説明される
- provenance、response、checkpoint、prediction cacheが再現不能

## 9. 未確定項目チェックリスト

- [ ] S0-4の24件response JSONをexportし、変更前hashを保存
- [ ] strict analyzerを完走し、低確信・unclearを監査
- [ ] margin calibrationと現場の許容errorを照合
- [ ] `delta_primary_min`を固定
- [ ] 6副次endpointの非劣性marginを固定
- [ ] locked受理率、worst-site、latency、fail-safe条件を固定
- [ ] code / data / split / metric versionのSHA-256を記録
- [ ] `draft`を外し、S1結果を見る前にcommit

このチェックリスト完了前はS1を開始しない。
