# S1 harm-first primary model specification and execution guard

- 実施日: 2026-08-24
- status: `pre_registration_candidate_structure_frozen_execution_blocked`
- track: Scientific / Operational
- model fitting / threshold selection / target analysis: 未実施
- locked / private key: 未使用

## 1. 目的

full development labelを生成する前に、結果を見てmodel/head/featureを後付けしないため、harm-first一次モデルの
構造を機械可読JSONへ固定した。同時に、S0-4、margin、全development artifactが揃わない限りmodel fitを
開始できないfail-closed validatorを実装した。

これは事前登録候補の構造であり、S1性能や採用を意味しない。確率thresholdは意図的に`TBD`のまま残す。

## 2. 一次model

- estimator: L2 logistic regression
- preprocessing: fold内`StandardScaler`、missing/non-finiteはfail closed
- class weight: `balanced`
- solver: `lbfgs`、max iterations 2,000
- inverse regularization `C`: 0.01 / 0.1 / 1 / 10
- C選択: inner OOFのobservation-equal weighted binary log-loss最小
- candidate weight: 同一observation内で`1 / candidate_count`
- additional calibration: primaryでは行わない
- single-feature stepwise search: 禁止

複雑なtree/boosting/neural modelを一次候補にせず、固定小feature、CPU再現性、係数監査、event数制約を優先する。

## 3. 固定5 heads

各headは物理・pipeline機構から事前選択した6特徴だけを使う。

| head | role | target | 固定featureの主な意味 |
|---|---|---|---|
| final PQ benefit | benefit | final PQ一次効用 > 0 | detector/source/semantic/contrast/area/ownership保持 |
| spine benefit | benefit | spine差 > 0 | detector/source/elongation/perimeter/semantic/area |
| count harm | harm | count absolute error差 > 0 | proposal/output/count差/包含競合/source |
| Precision harm | harm | Precision差 < 0 | detector/semantic/contrast/area/center/count差 |
| merge harm | harm | merge差 > 0 | IoU/相互包含/競合/area/count差 |

site/year/time-block/fold/selection hash、target/utility、既存verifier出力は予測featureとして禁止する。

## 4. event sufficiency gate

各一次headは6 feature＋interceptの7 fitted degrees of freedomとし、最低10 events / degree of freedomを要求する。

- minimum candidate events per outer training: **70**
- minimum independent event observations per outer training: **30**
- 不足時: operational promotionを遮断し、別threshold探索で救済しない

特にmerge headが不足する可能性がある。headを学べない場合もfinal merge非劣性gateを解除しない。

## 5. nested selection

registry SHA-256
`48b69680ac40a411b65cdc8cd9246a475ca88002f8483bae3acf2f31a9db2219`を固定する。

各outer cycleで、3 inner OOFのlog-lossだけからCを選ぶ。feature setはheadごとに固定し、threshold選択は行わない。
outer testは仕様固定後に1回だけ予測する。Brier、AUPRC、AUROC、calibrationは記述的に報告するが、
candidate指標だけで採用しない。選択集合を固定ownershipへ戻して科学・catalog endpointを評価する。

## 6. thresholdとfinal policy

次はすべて`TBD_AFTER_S0_4_AND_MARGIN_REGISTRATION`のまま固定した。

- PQ/spine benefit minimum probability
- count/Precision/merge harm maximum probability
- minimum proposal selection rate

S0-4と現場許容errorからendpoint marginを登録した後、model結果を見る前に埋める。probability thresholdを
classifier accuracy最大化だけで決めない。最終acceptanceにはspine superiority、Precision/merge/count/PQ
noninferiority、time/site report、operational zero-harm validationをすべて要求する。

## 7. readiness guard実測

validator結果は`structure_ready_execution_blocked`。

| gate | status |
|---|---|
| model structure validation | pass |
| frozen feature contract hash | pass |
| frozen split registry hash | pass |
| complete S0-4 response | **blocked** |
| all endpoint margins registered | **blocked** |
| decision policy registered before target/model results | **blocked** |
| full development features 4,899 rows | **blocked** |
| full development labels 4,899 rows with five binary heads | **blocked** |

したがって`ready_to_fit=false`。validatorはprivate keyを読まず、default public response pathが存在しても
24完全回答だけを確認する。unblindは別のstrict analyzerの責務である。

## 8. 次の開始条件

1. 独立reviewerが24件を完全回答しpublic response JSONを保存する。
2. S0-4をstrict analyzerで完了し、現場ownerの許容errorと合わせてmarginを登録する。
3. decision probability policyとhash/担当/時刻/rationale provenanceをtarget読込前に登録する。
4. frozen builderで4,899 featureと5 binary headを持つ4,899 labelを生成する。
5. readiness validatorが全gateをpassした場合だけnested OOF fittingを開始する。
6. 新規time/site blockがなければ結論は探索的とする。

## 9. 成果物

- `S1_HARM_FIRST_MODEL_SPEC_2026-08-24.json`
- `scratch/validate_s1_harm_first_model_spec.py`
- `tests/test_validate_s1_harm_first_model_spec.py`
- `output/s0/s1_harm_first_model_spec_readiness.json`

readiness JSONには構造error、全gate、blocker、public response状態、margin TBD数、full table状態、scope、SHA-256を
保存した。
