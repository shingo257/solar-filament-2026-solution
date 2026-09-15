# S1 proposal-level post-refiner marginal utility CPU pilot

- 実施日: 2026-08-24
- status: `fixed_development_cpu_pilot_not_s1_not_registered`
- track: Scientific / Operational
- 計算資源: ローカルCPUのみ
- sample: 2011〜2016 development 12観測、137 fused proposals
- locked / S1 prediction / blinded response / private key: 未使用

## 1. 目的

画像単位conditional rescueはGT-free gateでcatalog安全性を識別できず停止した。残した唯一のS1候補である
**post-refiner proposal単位confirmation**について、既存cacheとfold別refinerから最終pipeline marginal
utility教師を実際に生成できるか、また既存pre-refiner proxyとの不一致がどこにあるかをCPUで確認した。

これはモデル学習、閾値選択、S1開始判定ではない。outcome非依存の固定sampleで、ラベル生成可能性、
概算コスト、既存proxy/verifierの誤差構造だけを調べるpreflightである。

## 2. 固定sampleとリーク境界

`output/s0/observation_strata.csv`と既存V42 candidate tableを使い、candidateを1件以上持つdevelopment観測を
各fold内で`SHA-256(20260824:file_name)`順に並べ、先頭4件ずつを選んだ。GT endpoint、verifier score、
candidate utilityは選定に使用していない。

- fold 0/1/2: 各4観測
- year: 2011〜2016
- site: Bh / Ch / Lh / Mh / Th / Uhを含む
- candidate count: 1観測6〜17、合計137
- time block: 全件`development`
- locked rows read: 0

各観測には対象foldを学習から除いた既存refiner checkpointだけを使った。

## 3. post-refiner marginal utility

観測ごとにV42 fused proposalを全件一度だけ`shrink@0.45` refinerへ通し、refined proposalをcacheした。
その後、accept-all集合と各候補を1件だけ除いた集合へ固定ownership ruleを適用し、差を測った。

### 3.1 official PQ局所効用

候補追加による公式PQの一次効用を次で定義した。

```text
utility = Δ TP-IoU numerator - archived V35 PQ × Δ denominator
archived V35 PQ = 0.4426332944
```

正なら、他のcorpus集計を固定した局所近似でV35 PQ比を改善する方向である。

### 3.2 科学・catalog endpoint

2048 px final maskを512 pxへresizeし、annotator variant内でall-spine coverage、filament Recall/Precision、
merge、count absolute error、PQ@512を再評価した。catalog-safe targetは次を同時に要求した。

- spine coverage差 `> 0`
- filament Precision差 `>= 0`
- merge差 `<= 0`
- count absolute error差 `<= 0`

差はすべて「候補ありのaccept-all − その候補なし」。leave-one-outなので候補間interactionは非加法である。

## 4. 実測結果

137候補中、final PQ一次効用が正は88件、0近傍は14件、catalog-safe陽性は56件だった。

| predictor | final target | TP | FP | FN | TN | agreement | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pre-refiner proxy target | final PQ positive | 80 | 7 | 8 | 42 | **89.1%** | **92.0%** | **90.9%** |
| existing verifier accept | final PQ positive | 72 | 21 | 16 | 28 | 73.0% | 77.4% | 81.8% |
| pre-refiner proxy target | final catalog-safe | 55 | 32 | 1 | 49 | 75.9% | 63.2% | **98.2%** |
| existing verifier accept | final catalog-safe | 48 | 45 | 8 | 36 | 61.3% | 51.6% | 85.7% |

pre-refiner continuous utilityとfinal PQ一次効用のSpearman相関は`rho=0.7665`、final spine差とは
`rho=0.6414`だった。固定sampleでは、pre-refiner教師そのものは最終効用と無関係ではない。
一方、11特徴へ圧縮した既存logistic verifierと閾値選択でalignmentが大きく失われている。

pre-refiner proxy陽性87件のcatalog-safe不合格理由は、重複可で次の通り。

| failure | candidates |
|---|---:|
| count absolute error増加 | **22** |
| spine差が非正 | 8 |
| Precision低下 | 4 |
| merge増加 | 1 |

このsampleでは、proposal単位の安全性を壊す最大要因はcount errorである。画像単位gateで見えた
Precision/merge trade-offだけをharm targetにしても不足する。

## 5. CPUコスト

- 初回12観測/137候補: 377.7秒
- 1観測あたり: 31.48秒
- 1候補あたり: 2.76秒
- development: 547観測、4,899候補
- 全development概算: candidate単価で約3.75時間、observation単価で約4.78時間

refined proposalは観測単位cacheへ保存する。再実行時にrefiner forwardは省略できるが、ownershipと
leave-one-out endpoint再計算は残る。全development label生成はCPUで現実的だが、S0-4完了前に
model fittingやthreshold選択を行う理由にはしない。

## 6. 判断

### 維持する判断

- 画像単位conditional rescueの停止は変更しない。
- 既存verifierをそのままS1へ昇格しない。
- locked結果、review回答、同じ707 OOF上の追加threshold sweepを使わない。

### proposal-level S1について新たに分かったこと

- post-refiner/ownership後のcandidate marginal labelは既存artifactからCPUで生成可能。
- pre-refiner proxyはfinal PQ/spineと一定のalignmentを持ち、仮説自体を全否定する根拠ではない。
- 主な損失は11特徴logistic verifierへの圧縮と、PQ proxyがcount/catalog harmを直接扱わない点にある。
- 次のtargetは「benefit verifier」単独ではなく、benefitとcount/Precision/merge harmを分けるmulti-taskまたは
  harm-first gateであるべき。

### 次の開始条件

S0-4人手review完了後、S1 marginを固定してからのみ、全development 4,899候補のlabel生成とnested OOFを
行う。feature/target/thresholdを固定し、新しい独立time/site blockを確保できない場合は探索的結果に
限定する。この12観測pilotから性能、margin、採用閾値を決めない。

## 7. 成果物

- `scratch/run_s1_proposal_marginal_utility_pilot.py`
- `tests/test_s1_proposal_marginal_utility_pilot.py`
- `output/s0/s1_proposal_marginal_utility_selection.csv`
- `output/s0/s1_proposal_marginal_utility_pilot.csv`
- `output/s0/s1_proposal_marginal_utility_pilot.json`
- `output/s0/s1_proposal_marginal_refined_cache/`

JSONにはscript/input/selection/row CSVのSHA-256、scope、runtime、limitationsを記録した。
