# S1 conditional rescue headroom CPU診断

- 実施日: 2026-08-24
- status: `retrospective_feasibility_only_not_registered`
- track: Scientific / Operational
- baseline: V35
- comparator: V42
- 計算資源: ローカルCPUのみ
- S1 prediction: 未使用
- blinded review response / private key: 未使用

## 1. 問い

V42はV35よりall-spine coverageとRecallを改善する一方、Precisionとmergeを悪化させる。このtrade-offに
対し、V35/V42を画像単位で条件付き選択すれば、現場のカタログ品質を壊さずS1へ進むだけの改善余地が
あるかを検証した。Leaderboard/PQや単一coverageへの最適化ではなく、次を同時に問う。

1. GTを見られる非実装oracleなら、どこまで改善できるか。
2. 推論時に観測できるV35/V42出力だけで、安全なcaseを識別できるか。
3. その改善量はlocked 160件の80% MDE 0.03238と比べてどの程度か。

これはS1そのものではない。既存OOFを再利用した画像単位のretrospective headroom診断であり、
2017〜2022層は過去のS0解析で既に閲覧済みなので独立holdoutではない。

## 2. 評価契約

- 同一observation/modelのannotator variantを先に平均し、707観測を等重み化した。
- developmentは2011〜2016の547観測。3 fold OOFでgate確率を生成し、閾値選択もここだけで行った。
- locked evaluationは2017〜2022の160観測。学習・閾値選択に使わず、retrospective転移診断だけ行った。
- V42を選ばない観測はV35をそのまま使う。
- gate閾値はV42選択率10%以上を要求し、development OOFで次をすべて満たす候補から
  all-spine coverage差を最大化するよう固定した。
  - filament Precision差 `>= 0`
  - merge count差 `<= 0`
  - count absolute error差 `<= 0`
- endpoint差はすべて「選択policy − V35」。lower-is-better endpointは負が改善である。
- `pq_512`は観測単位proxyであり、公式2048 px micro PQではない。

## 3. 非実装oracle上限

| policy | block | V42選択 | spine coverage差 | 95% CI | Precision差 | merge差 | count error差 | PQ@512差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V42 all | development | 547/547 | +0.01718 | +0.01117〜+0.02309 | -0.03085 | +0.05484 | -0.09689 | -0.00090 |
| V42 all | locked | 160/160 | +0.01326 | -0.01248〜+0.03922 | -0.01801 | +0.02500 | +0.01667 | -0.00274 |
| oracle primary | development | 190/547 | +0.02956 | +0.02477〜+0.03443 | -0.00037 | +0.02102 | -0.16514 | +0.01210 |
| oracle primary | locked | 20/160 | **+0.03409** | +0.01608〜+0.05620 | +0.02631 | -0.00625 | -0.03125 | +0.01914 |
| oracle catalog-safe | development | 80/547 | +0.01311 | +0.00969〜+0.01695 | +0.00728 | -0.00366 | -0.10177 | +0.00767 |
| oracle catalog-safe | locked | 14/160 | **+0.02831** | +0.01126〜+0.04944 | +0.02916 | -0.00625 | -0.05000 | +0.01818 |

`oracle primary`はGTを見てcoverageが上がる観測だけV42を選ぶ。locked差+0.03409はMDEの1.053倍で、
理論上の改善余地は存在する。ただし`oracle catalog-safe`はさらにPrecision非低下、merge/count error
非増加をcase単位で要求し、locked差は+0.02831、MDEの0.875倍へ下がる。oracleはいずれも非実装で、
特にlockedは陽性20件/14件しかないため、この上限を独立汎化性能とは解釈しない。

## 4. 推論時observable gate

### 4.1 特徴

最初に予測個数・連結成分数のV35値、V42値、差の6特徴を使った。次に既存1414 cacheを全件decodeし、
GTを使わず次を追加した。

- 各モデルのmask面積合計、union面積、重複面積、平均/中央値/最大面積、area CV
- bbox fill、elongation、thinness
- V35/V42 union IoU、V42追加/消失面積
- V35↔V42 best mask IoUとunmatched fraction
- sensitivityとしてyear/site operational contextあり/なし

1414/1414 cacheを読め、cache内mask数とscience metricsのprediction count不一致は0。安全な特徴CSVには
予測形状とoperational metadataだけを保存し、GT endpoint、annotation、review回答を含めていない。

### 4.2 coverage改善target

| 特徴 | development OOF AUC | locked retrospective AUC | 安全閾値 |
|---|---:|---:|---|
| count/components | 0.765 | 0.833 | なし |
| count/components + year/site | 0.752 | 0.847 | なし |
| prediction geometry | **0.780** | 0.799 | なし |
| prediction geometry + year/site | 0.771 | 0.836 | なし |

coverageが上がるcase自体はある程度順位付けできる。しかし選択率10%以上の全閾値で、Precision非劣性を
満たす候補は0、merge非劣性を満たす候補も0だった。geometry-onlyの制約なし最良点はdevelopmentで
選択率58.1%、coverage +0.02075だが、Precision -0.02919、merge +0.04906、PQ@512 -0.00015。
coverage最適化はカタログ効用へ移らなかった。

### 4.3 catalog-safe効用target

学習targetを、coverage改善だけでなくPrecision非低下・merge非増加・count error非増加を同時に満たす
caseへ変更した。これは評価手法への最適化を避け、development GTから実用効用を直接学ぶ診断である。

| 特徴 | development陽性 | development OOF AUC | locked陽性 | locked retrospective AUC | 安全閾値 |
|---|---:|---:|---:|---:|---|
| count/components | 80/547 | 0.502 | 14/160 | 0.656 | なし |
| count/components + year/site | 80/547 | 0.441 | 14/160 | 0.597 | なし |
| prediction geometry | 80/547 | 0.493 | 14/160 | 0.803 | なし |
| prediction geometry + year/site | 80/547 | 0.477 | 14/160 | 0.820 | なし |

最も重要なのはdevelopment OOFでAUCが0.44〜0.50、すなわち安全な改善caseをほぼ識別できないこと。
locked AUCだけが高い構成は14陽性の既閲覧層に対するretrospective値であり、モデル選択に使わない。
実用効用targetへ変えても、選択率10%以上で3つのcatalog制約を満たす閾値は全8構成で0だった。

## 5. 判断

### 停止するもの

- V35/V42を画像単位で切り替えるconditional rescueは停止する。
- prediction count/components、全体mask geometry、year/siteだけを使う画像単位gateへGPUを使わない。
- coverage AUCやoracle上限だけを根拠にS1を開始しない。
- locked retrospective AUCを新規holdout性能として報告しない。

### S1を残す条件

S1を続けるなら、画像全体ではなく**post-refiner proposal単位の局所confirmation**に限定する。入力候補は
detector confidence、seed間support、refiner probability/calibration、局所重複、proposal geometry、
近傍密度とし、targetはcoverage単独ではなくPrecision/merge/countまで含むmulti-endpoint utilityとする。
development内nested OOFで設計し、S0-4完了後にmarginを事前固定し、新しい独立time/site blockを確保する。

ただし既存V42 candidate verifierは候補代理効用を改善してもfull pipeline PQへ移らなかった。したがって
proposal-level S1も、post-refiner/ownership後の最終出力に対するmarginal utilityを学べない場合は開始しない。

## 6. 成果物

- `scratch/evaluate_s1_conditional_rescue_headroom.py`
- `scratch/evaluate_s1_prediction_geometry_gate.py`
- `tests/test_evaluate_s1_conditional_rescue_headroom.py`
- `tests/test_evaluate_s1_prediction_geometry_gate.py`
- `output/s0/s1_conditional_rescue_headroom.json`
- `output/s0/s1_prediction_geometry_features.csv`
- `output/s0/s1_prediction_geometry_gate.json`

全成果物は`status=...not_registered`で、S1 prediction・盲検review回答・private keyを読んでいない。
