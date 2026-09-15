# S1 nested OOF execution plan and fail-closed preflight

- 実施日: 2026-08-24
- status: `execution_plan_frozen_fit_blocked`
- track: Scientific / Operational
- model fitting / threshold selection / target analysis: 未実施
- locked / private key / review response: 未使用

## 1. 目的

S0-4と登録完了後に、結果を見ながらfit順序やsplitを変えないよう、固定model specとgrouped nested split
registryから全compute unitを機械可読JSONへ列挙した。同時に、現在のblocker下でfit開始要求を非ゼロ終了にする
preflightを追加した。

execution planは実行グラフであり、model fitting implementationや性能結果ではない。

## 2. preflight前に修正した開始条件

実行グラフを組む過程で、従来readinessに次の2つのfail-open余地を確認し、先に閉じた。

1. endpoint margin登録後も、5 headのprobability policyが未登録のままfit開始できる余地があった。
2. full label CSVはidentity 3列しか検査しておらず、5 target列が欠けても4,899行なら完成扱いになり得た。

model spec schemaをversion 2へ更新し、正式label列を次へ固定した。

- `final_pq_benefit`
- `spine_benefit`
- `count_harm`
- `precision_harm`
- `merge_harm`

全列は0/1 binary、欠損なし、`file_name/fold/candidate_index`一意、4,899行を必須とする。

decision policyは、数値thresholdだけでなく、target値を読む前の登録証明、登録時刻・担当、public review response
SHA-256、endpoint registration SHA-256、rationale documentを必須とした。blocked specでは数値thresholdを拒否し、
registered specでは`TBD`やprovenance欠損を拒否する。

## 3. 固定compute graph

| phase | formula | jobs |
|---|---:|---:|
| inner model selection | 3 outer × 5 heads × 4 C × 3 inner validation | 180 |
| outer refit and score once | 3 outer × 5 heads | 15 |
| **model fit total** | inner + outer | **195** |
| frozen policy + ownership replay | 3 outer | 3 |
| threshold fit | 禁止 | 0 |

各inner jobはvalidation以外の2 inner foldだけで`StandardScaler`とlogistic regressionをfitし、
observation-equal weighted log-lossを保存してmodel artifactを破棄する。headごとにCを決めた後だけouter training全体で
再fitし、outer testを1回だけscoreする。最後に登録済みpolicyとfrozen ownershipを適用してpipeline endpointを
再計算する。

## 4. split accounting

| outer | training observations / candidates | test observations / candidates |
|---:|---:|---:|
| 0 | 364 / 3,227 | 183 / 1,672 |
| 1 | 363 / 3,364 | 184 / 1,535 |
| 2 | 367 / 3,207 | 180 / 1,692 |

全inner jobでtraining＋validationが対応outer trainingと一致する。job IDは180件すべて一意で、outer testは
headごとに1回だけ予測する。registry SHA-256不一致、source fold改変、outer leakage、cycle内重複はpreflightで
拒否する。

## 5. readiness実測

model structureとexecution plan integrityは合格したが、`fit_start_permitted=false`。blockerは5件である。

1. S0-4 public response 24件の完全回答。
2. 全endpoint marginの登録。
3. decision probability policyのtarget読込前登録とprovenance固定。
4. full development feature 4,899行。
5. 5 binary head列を持つfull development label 4,899行。

通常plan生成はstatusを保存して終了する。`--require-fit-ready`を付けた開始preflightは、現在の状態で非ゼロ終了し、
fitを開始しなかった。

## 6. scope boundary

plan生成で読んだのはregistryのouter/inner identityとcandidate count、model spec、readinessだけである。

- feature values loaded: false
- target values loaded: false
- review response loaded: false
- private key read: false
- locked rows read: 0
- model fitting performed: false
- threshold selection performed: false

event sufficiencyは登録済みfull labelsが存在してからouter training単位に監査する。70 candidate eventsかつ30 event
observations未満のheadはoperational promotionをblockし、別threshold探索で救済しない。

## 7. 成果物

- `scratch/build_s1_nested_oof_execution_plan.py`
- `tests/test_build_s1_nested_oof_execution_plan.py`
- `output/s0/s1_nested_oof_execution_plan.json`
- `S1_HARM_FIRST_MODEL_SPEC_2026-08-24.json` schema version 2
- `scratch/validate_s1_harm_first_model_spec.py` readiness schema version 2

source foldは歴史的grouped foldであり、新規time/site holdoutではない。execution planが再現可能でも、独立blockなしの
S1結果は探索的主張に限定する。
