# S1 GT-free set-context feature contract

- 実施日: 2026-08-24
- status: `fixed_gt_free_set_context_features_not_model_not_threshold_not_registered`
- track: Scientific / Operational
- sample: 2011〜2016 development 12観測、137 fused proposals
- 計算資源: ローカルCPUのみ
- GT / target / locked / S1 prediction / blinded response / private key: 未使用

## 1. 目的

harm-target feasibilityで、count/merge害は候補独立ではなく観測単位にcluster化し、count×mergeも
ほぼ別事象と分かった。既存11 candidate特徴だけへ戻らず、refiner後maskと固定ownershipから推論時に
計算できる観測・集合contextを、model fitting前にfeature contractとして固定した。

この成果物は特徴生成の再現性確認であり、feature-target関連、モデル、閾値、margin、S1性能を評価しない。

## 2. label非依存の入力境界

feature builderが読むのは次だけである。

1. development固定12観測の公開selection列
2. V42 candidate tableのidentifierと既存11 inference featureだけ
3. fold別OOF refiner済みmask cache

candidate tableに同居する`target`、IoU、marginal utility、verifier出力は`read_csv(usecols=...)`で
読み込まない。GT JSON、science endpoint、pilot harm labelも読まない。`site_code`と`year`はstratified report用
metadataであり、学習feature contractには含めない。

## 3. 固定した特徴

### 3.1 既存inference feature（11）

`detector_score`、`source_count`、`log_area`、`bbox_extent`、`log_aspect`、
`perimeter_over_sqrt_area`、`solidity`、`center_radius`、`semantic_mean`、`semantic_q10`、
`local_contrast`。

### 3.2 GT-free set-context feature（29）

- 観測規模: proposal数、ownership後output数、output/proposal比
- leave-one-out set効果: 候補あり−候補なしのoutput count差、候補なしoutput数
- 観測内相対値: detector/semanticのmean・SD・z-score・percentile、source count percentile
- refined geometry: 面積率・観測内percentile
- proposal競合: overlap neighbor数、pairwise IoU max/mean/sum、相互包含率、containment conflict数
- ownership trace: containment/min-area drop、output保持、exclusive面積率、保持率、出力順percentile

固定値は2048×2048、minimum exclusive area 20、containment threshold 0.80で、V42最終pipelineと同じ。

## 4. 固定pilotでの構造監査

- 137候補、12観測、40 feature。欠損0、非有限値0。
- ownership outputへ残る候補125、containment drop 12、min-area drop 0。
- overlap neighborを持つ候補22、containment conflictを持つ候補22。
- `output_count_delta`: `+1`が115件、`0`が20件、`-1`が2件。
- 既存pilotのfull/without output countと137/137で一致。
- 同一入力で再実行したfeature CSV SHA-256は完全一致。
- 40 feature中、固定pilotで定数だったのは`dropped_by_min_area_after_ownership`だけ。小標本だけを理由に
  contractから削除せず、全development生成後に事前規則で扱う。

## 5. `output_count_delta=-1`の意味

2候補は自分自身がownership outputへ残る一方、別の小候補を包含除去し、候補追加後のoutput総数を1減らした。

| observation | candidate | proposal count | full output | candidateなし | containment conflicts |
|---|---:|---:|---:|---:|---:|
| `20150423120034Ch.jpeg` | 5 | 16 | 14 | 15 | 2 |
| `20150706083014Th.jpeg` | 14 | 17 | 15 | 16 | 2 |

これはGTを見なくても計算でき、count harmの候補集合相互作用を直接表す。candidate probabilityを独立に閾値処理する
だけでは失われるため、各候補の採否後にownershipを再実行する必要がある。

## 6. downstream contract

1. 全splitをobservation group単位にする。同一観測の候補をtrain/validationへ分けない。
2. `site_code`、`year`、selection hash、time-block roleは監査metadataとし、予測featureへ入れない。
3. benefit/count/Precision/merge headは同じGT-free feature tableを使うが、targetは別々に保持する。
4. 各outer foldのfeature scaling、欠損処理、容量選択はinner developmentだけでfitする。
5. candidate scoreだけで採用せず、選択集合に固定ownershipを再適用してfinal endpointを評価する。
6. feature ablationは事前固定family単位で行い、単一featureをOOF結果に合わせて逐次追加しない。

## 7. 境界と次の開始条件

- このfeature tableとharm labelを結合して関連を探索しない。
- S0-4独立人手review、現場許容error、margin固定後にのみ、全development label/feature生成とnested OOFを行う。
- locked 160観測はfeature/model/threshold選択に使わない。
- 新しいtime/site blockを確保できなければ、最終結論を探索的に限定する。

## 8. 成果物

- `scratch/build_s1_set_context_features.py`
- `tests/test_build_s1_set_context_features.py`
- `output/s0/s1_set_context_features.csv`
- `output/s0/s1_set_context_features.json`

JSONにはfeature whitelist/forbidden tokens、scope、入力・cache manifest・出力SHA-256、ownership/競合summary、
limitationsを保存した。
