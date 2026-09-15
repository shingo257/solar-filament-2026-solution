# S1 harm-target label feasibility CPU report

- 実施日: 2026-08-24
- status: `fixed_pilot_harm_label_feasibility_not_model_not_threshold_not_registered`
- track: Scientific / Operational
- 計算資源: ローカルCPUのみ
- sample: 2011〜2016 development 12観測、137 fused proposals
- locked / S1 prediction / blinded response / private key: 未使用

## 1. 目的

proposal marginal utility pilotで、既存verifierの最大の不足はcatalog harmを直接扱わない点と分かった。
次のharm-first設計へ進む前に、count error、Precision、mergeの害が何候補・何観測に存在し、独立headを
学べる量か、候補間で同じ害を表しているかを診断した。

モデル学習、feature選択、threshold sweepは行っていない。固定137候補のlabel prevalence、観測cluster、
共起、全development label生成時の計画範囲だけを測る。

## 2. 固定target contract

候補ありのaccept-all pipelineと、その候補だけを除いたpipelineとの差を使う。

| target | 定義 |
|---|---|
| final PQ benefit | archived `final_pq_positive_target == 1` |
| spine benefit | `delta_spine_coverage > 0` |
| count harm | `delta_count_abs_error > 0` |
| Precision harm | `delta_filament_precision < 0` |
| merge harm | `delta_merge_count > 0` |
| any catalog harm | count / Precision / merge harmのOR |
| catalog safe | spine benefitかつcatalog harmなし |

再計算したcatalog-safeとpilot CSVのarchived targetは137/137で一致した。

## 3. 害の量と観測cluster

| target | 候補events / 137 | event観測 / 12 | 最大1観測のevent比 | 上位3観測のevent比 | 3 foldに存在 |
|---|---:|---:|---:|---:|---:|
| final PQ benefit | 88 (64.2%) | 12 | 15.9% | 38.6% | yes |
| spine benefit | 92 (67.2%) | 12 | 14.1% | 37.0% | yes |
| **count harm** | **39 (28.5%)** | **6** | **33.3%** | **76.9%** | yes |
| **Precision harm** | **44 (32.1%)** | **12** | **18.2%** | **38.6%** | yes |
| **merge harm** | **7 (5.1%)** | **5** | **28.6%** | **71.4%** | yes |
| any catalog harm | 68 (49.6%) | 12 | 19.1% | 45.6% | yes |
| catalog safe | 56 (40.9%) | 9 | 21.4% | 57.1% | yes |

Precision harmは全観測に分布するが、count harmとmerge harmは強く観測cluster化している。候補を独立sampleと
みなしたrandom splitはperformanceを過大評価する。学習・validationは必ずobservation group単位にする。

## 4. 害targetは互換ではない

候補単位の共起は次の通り。

| pair | 共起events | Jaccard |
|---|---:|---:|
| count harm × Precision harm | 17 | 0.258 |
| count harm × merge harm | 2 | 0.045 |
| Precision harm × merge harm | 5 | 0.109 |

`any_harm`だけへ圧縮すると、頻度の高いPrecision/countに最適化され、低頻度mergeを見落とすおそれがある。
countとmergeは特に別事象であり、独立endpointとして保持する。

## 5. 全development label生成時の計画範囲

12観測をcluster単位で10,000回bootstrapし、4,899 development候補へ単純外挿した。これは計算・label量の
計画値であり、S1性能の推定ではない。

| harm | point events | cluster bootstrap 95% planning range | pilot event観測 |
|---|---:|---:|---:|
| count | 1,395 | 358〜2,690 | 6/12 |
| Precision | 1,573 | 1,217〜1,991 | 12/12 |
| merge | 250 | 77〜459 | 5/12 |

count/Precisionは独立head用label数を確保できる可能性が高い。mergeは候補eventsより独立観測数が制約になり、
全development生成後もevent数とevent観測数を確認するまでstandalone headを確定しない。

## 6. 実運用validationの最低規模

独立な採用候補で害が0件だった場合、片側95%上限を所定率未満にするためのexact binomial下限は次の通り。

| 証明したいharm rate上限 | 害0件で必要な独立採用候補 |
|---|---:|
| < 5% | 59 |
| < 2% | 149 |
| < 1% | 299 |

実際の候補は観測内で相関するため、これは楽観的な下限にすぎない。最終validationでは候補数だけでなく、
独立観測数、time/site coverage、GT個体密度を固定する。0件でない場合はexact intervalをそのまま報告する。

## 7. S1設計への判断

### 採用する設計

1. benefitとcount / Precision / merge harmを別targetとして保持する。
2. candidate-only 11特徴へ戻さず、GT-freeの観測状態を追加する。少なくとも全proposal数、ownership後output数、
   候補除外時のoutput count差、重複競合数、同一観測内rank・近傍密度を候補ごとに持たせる。
3. 学習・calibration・outer evaluationの全段をobservation grouped nested OOFにする。
4. candidate probabilityだけで性能を主張せず、選択集合をownershipへ戻した最終pipeline endpointで判定する。
5. full label生成後、各headは最低10 events / effective fitted degree of freedomを満たすようmodel容量を制限する。
   mergeが満たさなければ独立headの性能主張を行わず、最終merge非劣性だけを必須gateとして残す。

### 変更しない境界

- 画像単位conditional rescueは停止したまま。
- このpilotからfeature、model、threshold、marginを選ばない。
- S0-4独立人手reviewと現場許容errorによるmargin固定前に全development 4,899候補を探索しない。
- locked 160観測は再選択へ使わず、新しいtime/site blockがなければ結果を探索的とする。

## 8. 成果物

- `scratch/analyze_s1_harm_target_feasibility.py`
- `tests/test_analyze_s1_harm_target_feasibility.py`
- `output/s0/s1_harm_target_feasibility_observations.csv`
- `output/s0/s1_harm_target_feasibility.json`

JSONにはtarget contract、scope、input/artifact SHA-256、cluster bootstrap、zero-event validation下限、
limitationsを保存した。
