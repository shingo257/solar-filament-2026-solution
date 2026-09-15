# S0 科学・実運用再評価報告

- 実施日: 2026-08-24
- track: scientific / operational
- 対象: MAGFiLO 707観測、1,154 annotator-image、8,199 instance
- 状態: S0-2/S0-3完了、S0-4盲検package生成完了・独立人手レビュー待ち
- 独立性: grouped OOFによるretrospective比較。2017–2022層も既存OOFの層別であり、独立holdoutではない

## 1. 結論

V35は「PQだけに合わせて科学情報を一律に失ったモデル」ではない。raw YOLOより、未検出を含む
全spine coverage、中心線位置、面積誤差、filament Recall / Precisionを改善した。一方で、IoU > 0.5で
対応できた個体だけを見るとraw YOLOの方がspine coverageとclDiceが高い。したがってV35の
`shrink`は、候補の過剰領域を削って検出・位置精度を上げる代わりに、一部の真の細構造も削る
**測定可能なtrade-off**を持つ。

V42は小個体・高密集・太陽縁proxyでRecallと全spine coverageを改善したが、Precisionとmatched
spine coverageを悪化させた。利得の所在は科学カタログ上有望だが、全観測をV42へ置換する根拠には
ならない。

V42 proposalをnested-OOF logistic verifierで選別すると、候補単位の代理効用は18.2%改善した。
しかしrefiner・ownership後の公式PQは0.440303で、accept-all V42より+0.000371に留まり、V35
0.442633より-0.002330だった。**代理効用の大幅改善が最終パイプライン価値へ移らなかった**ため、
このverifierはV35置換として不採用とする。

## 2. 評価契約

- 各観測には、そのfoldを学習から除いたdetector/refiner checkpointだけを適用した。
- 同一観測のannotator variantは同じfoldへ固定し、集計時はvariantを保持した。
- 科学指標は512 pxで統一し、公式2048 px PQと区別した。
- instance対応はIoU > 0.5のmaximum-cardinality Hungarian matchingを使った。
- 平均差の95% CIはobservation-cluster bootstrapで算出した。
- 2017–2022の160観測は将来S1用のlocked time blockだが、既存OOFを用いる本報告の値は
  retrospective stratumであり、独立評価性能とは呼ばない。

## 3. 707観測baseline rescore

| model | PQ@512 | spine coverage | matched spine coverage | spine centerline F1 | clDice | filament Recall | Precision | F1 | count誤差 | split | merge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.392124 | 0.571795 | **0.935322** | 0.920526 | **0.861013** | 0.610913 | 0.621487 | 0.595999 | **1.7877** | 0.2114 | **0.1993** |
| V34 | 0.276172 | 0.513612 | 0.934818 | 0.912630 | 0.851852 | 0.548277 | 0.412391 | 0.451686 | 2.8969 | 0.3657 | 0.2426 |
| V35 | **0.440711** | 0.599281 | 0.921913 | **0.926501** | 0.849163 | 0.647635 | **0.676021** | **0.639444** | 1.8406 | **0.1101** | 0.2166 |
| V42 | 0.438302 | **0.613317** | 0.914930 | 0.925769 | 0.847641 | **0.668167** | 0.646748 | 0.635436 | 1.8206 | 0.1109 | 0.2660 |

`PQ@512`は科学指標と同じ縮尺での診断値であり、Competitionの公式2048 px PQではない。

### 3.1 raw YOLO対V35

rawからV35へ変えると、全spine coverageは+0.02749、spine centerline F1は+0.00620、
filament Recallは+0.03672、Precisionは+0.05453改善した。中心線距離は0.06267 px相当、
area relative errorは0.10317低下した。これらの95% CIは0を跨がない。

一方、rawはmatched spine coverageで+0.01298、strict matched coverageで+0.03961、clDiceで
+0.01074高い。V35のBoundary F1差は+0.00040相当でCIが0を跨ぎ、境界全体の改善では説明できない。
V35は「見つけて中心へ寄せる」面では良いが、対応済み個体の細いspine支持領域を一部削っている。

### 3.2 V42対V35

V42は全spine coverage +0.01404（95% CI 0.00756–0.02066）、filament Recall +0.02053
（0.01329–0.02777）を得た。一方、matched spine coverage -0.00687、Precision -0.02927、
merge率 +0.04939と悪化した。spine centerline F1、Boundary F1、clDice、filament F1、PQ@512の
paired差はCIが0を跨ぐ。

V42の利得は次の層へ局在した。

| stratum | ΔPQ@512 | Δspine coverage | ΔRecall | ΔPrecision | Δmerge |
|---|---:|---:|---:|---:|---:|
| small ≤ 1,000 px | +0.00548 | +0.02572 | +0.03217 | -0.02441 | +0.05450 |
| dense > 10 GT | +0.00650 | +0.02506 | +0.03051 | -0.02302 | +0.15022 |
| limb proxy > 0.66 | +0.00071 | +0.01946 | +0.02611 | -0.02628 | +0.06098 |
| 2017–2022 retrospective | -0.01445 | +0.00292 | +0.00968 | -0.02341 | +0.02317 |

後期年代ではRecall利得が縮みPQが悪化する。small/dense/limbへの条件付き救済仮説は残るが、
年代shiftを無視した全体置換は行わない。

## 4. V35の現場弱点

V35はdense層でspine coverage 0.5152、Recall 0.5602、count absolute error 3.725、merge 0.536。
small層ではcoverage 0.5468、Recall 0.5839である。limb proxy coverage 0.5761もmid層0.6462を
下回る。site code別PQ@512は0.4296–0.4549の幅がある。

2017–2022 retrospective層はdevelopment層に比べ、V35のPQ@512が0.4320対0.4417、Precisionが
0.6133対0.6942だった。count errorやmergeは後期層の方が低いため、単純な「年代が新しいほど全て
悪い」ではなく、観測条件・個体密度・site構成を分離する必要がある。

## 5. V42 candidate verifier

### 5.1 leakage-safe screening

V42の5,444 fused proposalへ、score、2-seed合意数、形状、semantic確率、局所contrastなど11特徴を
与えた。各outer foldは他2 foldだけで学習・内側選択し、全foldでthreshold 0.5が選ばれた。

| policy | accepted | 候補代理効用 |
|---|---:|---:|
| accept all | 5,444 | 1,099.221 |
| consensus-only | 4,311 | 1,256.788 |
| nested verifier | 3,860 | **1,299.199** |

verifier対accept-allは+18.2%、+0.2829/観測（95% CI 0.2239–0.3429）。consensus-only対比でも
+0.0600/観測（0.0229–0.0963）だった。全3 outer foldで代理効用差は正だった。

### 5.2 refiner・ownership後の公式PQ

| variant | official PQ | SQ | RQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| V35 | **0.442633** | 0.677929 | 0.652920 | 5,082 | 2,286 | 3,117 |
| V42 accept-all | 0.439932 | 0.677494 | 0.649352 | 5,263 | 2,748 | 2,936 |
| V42 nested verifier | 0.440303 | **0.682614** | 0.645026 | 4,600 | **1,464** | 3,599 |

verifierはaccept-allからFPを1,284減らしたが、TPも663減りFNが663増えた。SQは上がったがRQは
下がり、PQ利得は+0.000371だけだった。V35比は-0.002330、formal Competition gate 0.452633にも
届かない。fold別PQ 0.435006 / 0.440693 / 0.445112は、対応するV35 0.439288 / 0.441499 /
0.447052を全foldで下回る。

候補単位labelはrefiner前maskの独立なbest IoUから作ったため、候補間ownership、refiner後の境界、
corpus PQ分母、ある候補を消したときのsplit/merge変化を表せていない。代理効用に対する最適化が
最終目的へ移らなかった具体例であり、今後は**post-refiner full-pipeline marginal value**を学習対象に
するか、最終endpointを直接nested評価する。

### 5.3 time/site診断

verifierのofficial PQはdevelopment 0.442668に対し、2017–2022 retrospective層で0.417613。
候補受理率も71.6%から65.1%へ下がった。site別official PQはCh 0.4275、Mh 0.4282からUh 0.4577
まで開いた。独立holdoutではないが、後期年代やsiteで過剰棄却するdomain-shift兆候として扱う。

## 6. 判定と次の実行順序

1. V35をCompetition正本として維持する。V42/verifierを提出・運用置換しない。
2. S0-4 packageは生成済みである。(a) rawがV35よりmatched spineを保持する例、(b) V42がsmall /
   dense / limbで救済する例、(c) verifierが2017–2022層で落とす例を各8件、model名を隠して監査する。
   独立reviewerの判定は未完了なので、定性結論はまだ出さない。
3. S0-5では単一主仮説を「small/dense/limb候補の確認を、post-refiner価値とtime/site頑健性で学ぶ」
   とする。追加のWBF/conf/ownership閾値掃引へ戻らない。
4. S1の主要endpoint候補は全spine coverage。副次非劣性はfilament Precision / Recall、merge、
   matched spine coverage、official PQとする。数値marginはS0-4完了後、S1結果を見る前に固定する。
5. 2017–2022を学習・閾値選択から完全に外したtime-block評価を必須とし、site別worst-caseと
   coverage-rejection曲線を併記する。
6. 新規GPU学習前に、S0-4独立reviewとS0-5 endpoint registrationを完了する。

### 6.1 S0-4盲検package

`scratch/run_s0_blinded_review.py`をローカルCPUで実行し、`output/s0/blinded_review_v1/`へ
24一意観測を生成した。3比較は各8件、各比較の第1モデルがA/Bになる件数は4/4である。
各PNGはfull reference/A/Bと最大不一致領域のzoom reference/A/Bからなる6 panelで、annotation
variantの輪郭とspineを表示する。公開`review.html`にはmodel対応表を含めず、対応表と選定scoreは
`private/`へ分離した。公開manifest 24行、PNG 24枚、欠損0、SHA-256不一致0、公開HTMLの
model identity漏えい0を確認した。6件のunit testで選定一意性、A/B balance、focus box、
独立radio group、localStorage/JSON export、およびmodel名非包含を固定した。

statusは`awaiting independent blinded human review`。回答JSONを書き出す前に`private/`を開かず、
review結果を学習labelへ自動転用しない。人手判定完了前はS0-4完了やmodelの定性優位を主張しない。

### 6.2 public-only readiness監査と集計契約

`scratch/audit_s0_blinded_review_readiness.py`は`private/`を読まず、実24 PNGを全件decodeした。
全件`1650×1100×3`、pixel standard deviation 35.52〜48.24、file size 244,094〜411,423 byteで、
decode失敗、near-blank、hash不一致、公開HTMLのmodel identity漏えいはいずれも0。statusは
`ready_for_independent_review_with_sampling_caveats`である。

sampleは仮説濃縮型で、locked層は14/24（58.3%）に対しcorpusは160/707（22.6%）、representation
ratio 2.58。fold0/1/2も4/11/9である。6 siteすべてと2011〜2022の9年を含むが、24件の選好率を
母集団改善率・独立holdout性能として扱わない。

`scratch/analyze_s0_blinded_review.py`は公開manifestでReviewer ID、24 case、必須回答、tag集合を
検証した後だけprivate keyを読む。不完全responseはunblind前に拒否することをunit testで固定した。

### 6.3 S1 endpoint margin calibration

707観測をobservation等重みへ再集計し、7 endpointでV35 baseline、annotator pair差、raw/V42対V35の
paired差、locked 160件のpowerをCPU計算した。S1 predictionは読んでいない。主要all-spine coverageの
locked V35平均は0.58453（95% CI 0.53228〜0.63661）、V42−V35は平均+0.01326、SD 0.16470。

one-sided alpha 0.05の設計近似では、登録境界を改善方向へ越える80% MDE 0.03238、90% MDE
0.03810。margin 0で真の改善0.01/0.02/0.03のpowerは19.0%/45.7%/74.5%、80% power必要数は
1,678/420/187。locked 160件は小改善の確証には不足する。

V35の同一observation annotator pair差はall-spine coverageで平均0.18781、p90 0.44309だったが、
model間paired設計では共通reference変動が相殺されるため、この値をそのままmarginにしない。
Recallの80% MDEは0.03522、Precision 0.03971、merge count 0.03641、matched spine coverage
0.00846、count absolute error 0.09548、PQ@512 0.02390。PQ@512を公式2048 px PQと混同しない。

marginは統計的detectabilityだけで決めず、S0-4人手reviewと現場のFP/merge/false reject許容値を
併用する。詳細は`S1_ENDPOINT_MARGIN_CALIBRATION_2026-08-24.md`を正本とする。

## 7. 成果物とSHA-256

主要成果物は`output/s0/`配下にある。

- `science_case_metrics_v3.csv`: `5524d693fca869fd89cb693bd3c8563c74a3d99b35647cefa9755077a451108a`
- `science_model_summary_v3.csv`: `25ae9b3982100ea3fe4f89d355e0f6cf86091ff211afded06b1ab86f727fe941`
- `science_stratum_summary_v3.csv`: `0f48dfc3adcb8bf59e206b17453b45189afb60bd381968fc43d546fc6dff4f01`
- `science_paired_deltas_v3.csv`: `90ee5dbf3528fab21d254069c3dfcc31b3264f9394ed1a45c7c1629d1b975f43`
- `science_rescore_manifest_v3.json`: `8ae4d2bd3c983fa51d224d9b97db276365899204bb45333685cf6af386ad17ed`
- `science_rescore_provenance_reconciliation_2026-08-24.json`: `881108ab3f370b02bc3a3c785c924965478461063c349820a278d825eff6b730`

科学CSVの再生成時に、面積・密度・太陽中心距離proxyの6層別列が追加されたため、生成時点の
historical source hashと現在のhashは異なる。共通47列・4,616行はkey整列後に文字列不一致0、
最大数値差8.9×10^-16である。S0-4/S1成果物が保持するhistorical source hashは生成時証跡として
書き換えず、対応関係を`science_rescore_provenance_reconciliation_2026-08-24.json`に固定した。
- `v42_nested_candidate_screen.json`: `3cda5c860a049f0ef385b4789f5d141b784f72b620c35392929a172a72d0ecbb`
- `v42_verifier_official_eval_0-1-2.json`: `87a7ff44a815cf787cb88683eb2f8a3318d9a1da05235b3b19425edaeb66fba6`
- `v42_verifier_official_case_metrics_0-1-2.csv`: `586fe1b51572bf964ea878990ec5223df1da0ae704e2c61b0b7fed7cf0bf6b9e`
- `v42_verifier_official_stratum_metrics_0-1-2.csv`: `42d96ad2c7ead229feb8d2ad541e3276bc71a9881d6a4946d3238e11242ec2fb`
- `v42_verifier_official_prediction_manifest_0-1-2.csv`: `c2172f606ae30fedf401abf15a1a088e37d5fd0ab3ed3658494a7b69cfcb47aa`
- `blinded_review_v1/package_manifest.json`: `b19dc87a261bedef8cbdd4d3467f84f5bcde0e0efb8be8164cc5d565f24cd340`
- `blinded_review_v1/review.html`: `119ad756fb101681dc07c60fb18ab2fbf0733a4b045af07669b5f56d2d69e5dc`
- `blinded_review_v1/REVIEW_PROTOCOL.md`: `c8db8d156ab4578e1cdcdb20b58c10b4618f927e4fb31296ae75a34f10dfdac5`
- `s0_blinded_review_readiness_audit_public.json`: `be203a7fb22e479161fe59dc48ea1c172d79063b31e0dc75ac9a6f6a16ac3ab2`
- `scratch/analyze_s0_blinded_review.py`: `bf1956730581e649ae36f35ba542e530f4762ec59b95aaf37884adec2d082d2d`
- `scratch/audit_s0_blinded_review_readiness.py`: `a87351501ca3acad5fcaf9ee63f3453ccff74e821c730ba21fa15948a63cacf2`
- `s1_endpoint_margin_calibration.json`: `0a2170f286d1c1d7090dd66b252783a4b3db6a2284ded25fcb7d7a59cfad039e`
- `scratch/calibrate_s1_endpoint_margins.py`: `7e84daed458add6f379129718213b8886e8c4eec7596436604317cad5cb4b6b5`

## 8. 制約

- annotationのspineは独立した専門家再評価ではなく、polygon/chiralityと同じMAGFiLO注釈に由来する。
- barbの独立label、seeing/cloud、FITS parity、連続track、CME/flare endpointは未評価である。
- 512 px science metricsと2048 px official PQを直接同一値として比較しない。
- nested verifierはouter-fold leakageを避けたが、同じ707観測で仮説選択している。最終運用性能には
  independent time/site evaluationが必要である。
- V35もspace-weather運用適合が確認されたわけではない。現状は候補抽出・カタログ作成のbaselineである。
