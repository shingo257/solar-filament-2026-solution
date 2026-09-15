# S1 observation-grouped nested split contract

- 実施日: 2026-08-24
- status: `target_independent_grouped_nested_split_registry_frozen_not_model_not_s1`
- track: Scientific / Operational
- scope: 2011〜2016 development 547観測、4,899 proposals
- locked evaluation: 160観測を除外、registry混入0
- GT / target / feature値 / S1 prediction / review response / private key: 未使用

## 1. 目的

S1 featureとharm labelを結合する前に、モデル選択と最終outer評価の観測境界をtarget非依存で固定した。
同一観測内の候補をtrain/validationへ分割せず、site/yearを予測featureにせずに各inner foldへ均衡配置する。

既存candidate verifierは、outer foldを除いた残り2 source foldをinner validationに使うため、各inner fitが
1 source foldだけになる。S1 registryではouter 3 foldをそのまま最終評価境界に維持し、各outer-trainingを
新しい3 inner foldへ分割する。各inner fitはouter-trainingの約2/3を使える。

## 2. 入力境界

registry builderが読むのは次だけである。

1. `observation_strata.csv`のfile/fold/site/year/time-block列
2. candidate tableのfile/fold/candidate index列

candidate target、marginal utility、verifier score、feature値、annotation数などは`usecols`で読み込まない。
site/yearと予測候補数はsplit balanceと監査にだけ使い、予測featureへ入れない。

## 3. split contract

outer cycle 0/1/2について次を行う。

1. frozen source foldとouter testを一致させる。各観測はouter testに正確に1回だけ入る。
2. 残り2 source foldをouter trainingとする。
3. outer training内でsite×yearの36セルを先に均衡化し、その後に予測候補数と観測数をgreedyに均衡化する。
4. assignment tieは`SHA-256(seed, outer fold, stratum, file)`で決め、targetを使わない。
5. 同一fileの全候補は同じouter role / inner foldへ固定する。
6. registry SHA-256をlabel join前に固定し、以後変更しない。

## 4. outer cycle

| outer test fold | outer test観測 | outer test候補 | outer train観測 | outer train候補 |
|---:|---:|---:|---:|---:|
| 0 | 183 | 1,672 | 364 | 3,227 |
| 1 | 184 | 1,535 | 363 | 3,364 |
| 2 | 180 | 1,692 | 367 | 3,207 |

registryは547観測×3 cycle = 1,641行。outer test行547、outer train行1,094、outer cycle内重複0、
locked/non-development行0である。

## 5. inner validation balance

| outer | inner validation | 観測 | 候補 | site×year cells |
|---:|---:|---:|---:|---:|
| 0 | 0 | 117 | 1,077 | 36/36 |
| 0 | 1 | 124 | 1,076 | 36/36 |
| 0 | 2 | 123 | 1,074 | 36/36 |
| 1 | 0 | 119 | 1,122 | 36/36 |
| 1 | 1 | 119 | 1,121 | 36/36 |
| 1 | 2 | 125 | 1,121 | 36/36 |
| 2 | 0 | 121 | 1,067 | 36/36 |
| 2 | 1 | 119 | 1,069 | 36/36 |
| 2 | 2 | 127 | 1,071 | 36/36 |

inner間の候補数max−minはouter 0/1/2で3/1/4件、平均に対するrangeは0.279%/0.089%/0.374%。
観測数rangeは7/6/8件。全9 inner validationで36 site×yearセルすべてを保持した。

これはtarget prevalenceの均衡を保証しない。harm label生成後もregistryを変えず、target imbalanceはweighting、
報告CI、head容量で扱う。

## 6. downstream nested OOF手順

各outer cycleで次の順序を固定する。

1. outer testを完全に隔離する。
2. outer trainingの3 inner foldを順にvalidationとし、残り2 inner foldだけでscaler/imputer/modelをfitする。
3. 3 inner OOF predictionを結合し、feature family、容量、calibration、harm-first policyをinner OOFだけで決める。
4. 選択済み仕様をouter training全体で再fitし、outer testを1回だけ予測する。
5. candidate predictionを観測ごとの選択集合へ戻し、frozen ownership後の科学・catalog endpointを評価する。
6. 3 outer test結果を結合してdevelopment nested OOFとし、lockedを再選択に使わない。

scaler、class weight、calibration、thresholdはouter testを見ずにinner OOFで完結させる。site/yearは全工程で
監査stratumに限定する。

## 7. 再現性と固定値

- seed: `20260824`
- registry SHA-256: `48b69680ac40a411b65cdc8cd9246a475ca88002f8483bae3acf2f31a9db2219`
- 同一入力で再実行したregistry CSV SHA-256: 完全一致
- script、strata、candidate table、registryの記録SHA-256: 全一致

## 8. 境界

- registry作成はS1 model開始や性能達成を意味しない。
- source outer foldは歴史的grouped foldで、新しいtime/site holdoutではない。
- S0-4独立review、現場許容error、margin固定前にlabel/model/threshold探索を始めない。
- 新規独立time/site blockを確保できなければ、outer OOF結果も探索的と明記する。

## 9. 成果物

- `scratch/build_s1_grouped_nested_split_registry.py`
- `tests/test_build_s1_grouped_nested_split_registry.py`
- `output/s0/s1_grouped_nested_split_registry.csv`
- `output/s0/s1_grouped_nested_split_registry.json`

JSONにはcolumn whitelist、scope、split contract、全outer/inner balance、入力・出力SHA-256、limitationsを保存した。
