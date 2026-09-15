# S1 endpoint margin calibration

- 実施日: 2026-08-24
- status: `calibration_only_margin_not_registered`
- 入力: `output/s0/science_case_metrics_v3.csv`
- 対象: 707観測、locked 2017〜2022は160観測
- S1 prediction参照: なし
- bootstrap: observation cluster、5,000回、seed `20260824`

## 1. 結論

locked 160観測は、all-spine coverageの小さな改善を確証するにはpowerが不足する。V42−V35の
locked paired SD `0.16470`をS1の差の分散proxyとすると、登録marginを越えて80% powerを得る
minimum detectable effect（MDE）は`0.03238`、90%では`0.03810`である。

marginを0とした近似powerと80% power必要sample数は次のとおり。

| 真のall-spine coverage改善 | locked 160でのpower | 80% power必要観測数 |
|---:|---:|---:|
| 0.010 | 19.0% | 1,678 |
| 0.015 | 31.1% | 746 |
| 0.020 | 45.7% | 420 |
| 0.030 | 74.5% | 187 |

したがって、`delta_primary_min`を統計的に検出できる最小値だけから決めない。locked 160件のまま
進めるなら、S1はV42の平均利得`+0.01326`より大きい、実運用上意味のある改善を狙う必要がある。
小さい効果を狙う場合は、事前固定したsmall/dense/limb stratumでpaired varianceを下げる、追加の
独立観測を得る、または確証的主張ではなく探索的結果に限定する。

## 2. 集計契約

同一observationのannotator variantをmodel内で先に平均し、その後707観測を等重みにした。
paired differenceは`comparator − V35`。higher-is-better endpointでは正、lower-is-better endpointでは
負が改善方向である。MDEは、事前登録した境界を改善方向へさらに越える差の**絶対値**として示す。

実際のS1判定はobservation-cluster bootstrapで行う。ここでのpower/MDEはone-sided normal
approximation、alpha 0.05による設計用近似であり、最終p値やCIではない。

## 3. locked baseline、annotator再現性、MDE

| endpoint | 改善方向 | V35 locked平均 | V35 mean 95% CI | annotator pair差 平均 / p90 | V42−V35 locked平均 | V42差SD | MDE 80% / 90% |
|---|---|---:|---:|---:|---:|---:|---:|
| all-spine coverage | + | 0.58453 | 0.53228–0.63661 | 0.18781 / 0.44309 | +0.01326 | 0.16470 | 0.03238 / 0.03810 |
| filament Recall | + | 0.62801 | 0.57426–0.68117 | 0.19039 / 0.50000 | +0.02142 | 0.17915 | 0.03522 / 0.04145 |
| filament Precision | + | 0.60939 | 0.55471–0.66354 | 0.19414 / 0.50000 | -0.01801 | 0.20203 | 0.03971 / 0.04674 |
| merge count | − | 0.08646 | 0.04479–0.13438 | 0.23746 / 1.00000 | +0.02500 | 0.18523 | 0.03641 / 0.04285 |
| matched spine coverage | + | 0.92127 | 0.90039–0.93805 | 0.07164 / 0.16060 | -0.00769 | 0.03967 | 0.00846 / 0.00995 |
| count absolute error | − | 1.05208 | 0.90313–1.21250 | 1.51672 / 4.00000 | +0.01667 | 0.48571 | 0.09548 / 0.11237 |
| PQ@512 | + | 0.40446 | 0.36823–0.43969 | 0.12134 / 0.26706 | -0.00274 | 0.12157 | 0.02390 / 0.02813 |

annotator pair差は同一observation内のV35評価差で、296 multi-annotator観測を基本とする
（matched spineは有効値284観測）。これはcase単位のreference不確実性を表し、model間paired差の
SEとは同じではない。paired設計により共通annotationの変動は相殺されるため、annotator pair差を
そのままmarginにしない。

## 4. 実運用上の読み方

1. V42のlocked all-spine利得`+0.01326`は方向としては有望だが、現sampleでは確証的に小さい。
2. Recallは`+0.02142`でも80% MDE `0.03522`未満。一方Precisionは`-0.01801`、mergeは`+0.02500`
   と悪化方向であり、Recallだけを主張して採用しない。
3. matched spine coverageは比較的低分散でMDE `0.00846`だが、V42は`-0.00769`。細構造保持の
   非劣性marginを明示しないと、全spine改善と対応済み個体の形態悪化を混同する。
4. `pq_512`は設計用compatibility proxyで、公式2048 px PQのmarginではない。公式PQ/SQ/RQは別に
   登録し、既存official pipelineで評価する。
5. count errorやmergeは離散・zero-heavyであり、normal MDEは目安。最終解析ではbootstrap分布と
   error countの実数を併記する。

## 5. margin固定時の必須入力

S0-4完了後、次を同時に確認してから`TBD`を埋める。

- 盲検reviewで形態保存とcatalog品質のtrade-offが人手でも再現するか
- 707観測baselineのbootstrap分布と本calibrationのMDE
- false positive、merge、false rejectが現場で許容される絶対件数
- target hardwareでのlatency/fail-safe要件
- locked 160件で確証可能な効果か、追加データまたは探索的扱いが必要か

S0-4の24件はlocked層を意図的に濃縮しているため、その選好率だけでmarginを決めない。

## 6. 成果物とSHA-256

- `output/s0/s1_endpoint_margin_calibration.json`:
  `0a2170f286d1c1d7090dd66b252783a4b3db6a2284ded25fcb7d7a59cfad039e`
- `scratch/calibrate_s1_endpoint_margins.py`:
  `7e84daed458add6f379129718213b8886e8c4eec7596436604317cad5cb4b6b5`
