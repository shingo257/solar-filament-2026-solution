# S1 decision-policy registration transaction

- 実施日: 2026-08-24
- status: `registration_blocked`
- track: Scientific / Operational
- canonical model spec overwrite: false
- target values / feature values / private key / locked rows: 未使用

## 1. 目的

S0-4とendpoint margin登録後、full development targetを見る前に、5 headのdecision probability policyを
再現可能なtransactionとして固定する。canonical model specを直接編集せず、公開evidenceと登録payloadを検証して
registered spec candidateを別artifactとして生成する。

これは将来の登録手順であり、現時点のthreshold提案やmodel性能主張ではない。

## 2. 登録payload

`S1_DECISION_POLICY_REGISTRATION_TEMPLATE_2026-08-24.json`へ次をすべて記入する。

- source model spec SHA-256
- timezone付き登録時刻と登録担当
- public rationale document path
- PQ/spine benefit minimum probability
- count/Precision/merge harm maximum probability
- minimum selection rate
- 3 attestations

attestationsは、(1) development target未読、(2) review responseを定性的error priorityだけに使用、
(3) endpoint marginを主要safety contractにすることを明示する。6数値は0〜1、minimum selection rateは0より大きい
必要がある。

## 3. transaction prerequisites

registered candidate生成前に次をすべて検証する。

1. model structure、feature hash、registry hashが合格。
2. public S0-4 response 24件が完全で、readiness記録SHA-256と実fileが一致。
3. endpointの全marginが登録され、readiness記録SHA-256と実fileが一致。
4. decision policyがまだ未登録。
5. full development target label fileが存在しない。
6. readiness scopeがprivate key未読、locked 0、target未解析、fit/threshold selection未実施。
7. rationale documentが存在し、`private/`配下ではない。

transactionはpublic responseの内容を読み込まず、SHA-256だけを検証する。private blind keyは読まない。

## 4. registered candidate provenance

合格時だけ別pathへcandidate specを書き、次をprovenanceとして埋める。

- `registered_before_target_values_loaded=true`
- registered at / by
- public review response SHA-256
- endpoint registration SHA-256
- rationale document path / SHA-256

candidateは既存のmodel spec validatorで再検証する。canonical specと同じpathは明示的に拒否し、自動昇格しない。

## 5. CPU test

syntheticな全prerequisite合格caseではregistered candidateが構造合格した。次はすべてfail closedとなった。

- thresholdが範囲外、bool、文字列
- minimum selection rateが0
- timezoneなしregistered_at
- target label fileが既に存在
- public response hash不一致
- rationale documentが`private/`配下
- provenance hash欠損

## 6. 実状態audit

現在の実auditは`registration_blocked`、`registration_permitted=false`、candidate未生成。主な理由は次のとおり。

- S0-4 public responseが存在しない。
- endpoint draftに13個の`TBD`が残る。
- registration templateの担当・時刻・rationale・6 threshold・3 attestationが未記入。

`--require-registered`実行も非ゼロ終了し、candidateを書かなかった。これは意図した停止である。

## 7. 成果物

- `S1_DECISION_POLICY_REGISTRATION_TEMPLATE_2026-08-24.json`
- `scratch/register_s1_decision_policy.py`
- `tests/test_register_s1_decision_policy.py`
- `output/s0/s1_decision_policy_registration_audit.json`

次の外部入力は独立reviewerの24件回答と、現場ownerを含むendpoint margin/policy rationaleの決定である。それまでは
templateを埋めず、full target label生成とmodel fittingへ進まない。
