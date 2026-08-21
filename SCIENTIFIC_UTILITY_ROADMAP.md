# Solar Filament 科学・実運用価値ロードマップ

- 制定日: 2026-08-20
- 対象: Solar Filament Segmentation Challenge 2026 と、その後の科学利用・運用利用
- 位置づけ: 今後のモデル設計、実験採否、技術報告、論文化で参照する正本

## 1. 方針決定

V35 は **Kaggle の公式 Panoptic Quality (PQ) に整合した競技用基準モデル**として固定する。
全fold OOFでの改善は有効な実験結果だが、PQ改善だけでは、太陽物理で重要な spine、barb、
chirality、時間発展、観測条件変化への有用性を証明しない。

今後は次の2トラックを分離して管理する。

| トラック | 目的 | 主な合格条件 |
|---|---|---|
| Competition | コンペ提出と4ページ技術報告 | 公式PQ、SQ/RQ、提出仕様、全fold OOF、再現性 |
| Scientific / Operational | カタログ化、形態解析、chirality、時系列監視 | 形態・下流タスク・外部妥当性・運用頑健性 |

同じモデルを両トラックで比較してよいが、**片方の合格をもう片方の合格として扱わない**。
特に「Public/OOF PQが上がった」ことから「CME・フレア解析に役立つ」と推論してはならない。

## 2. V35で確認済みのこと／未確認のこと

### 確認済み

- H-Alpha全面画像から個体候補を検出し、instance maskとして分離できる。
- `shrink@0.45`と排他化により、全fold OOF PQは生YOLOの`0.3937`から`0.4426`へ改善した。
- grouped split、fold別refiner、公式PQ実装を用いたリークなし比較がある。
- アノテータ不一致がPQ上限へ与える影響を定量化している。

### 未確認

- `shrink = refined & proposal`がspine・barbの欠落を増やさないこと。
- 面積、長さ、中心線、接続性、分枝方向が科学利用に十分な精度であること。
- MAGFiLOのspine・chirality情報を利用した下流タスク性能。
- 観測時刻、GONG観測所、seeing、雲、彩度、太陽縁距離が変わったときの頑健性。
- JPEG q96経路と原観測/FITS経路の同等性。
- 連続観測でのtrack継続、出現・消失、面積変化の信頼性。
- 実運用に必要なレイテンシ、失敗検知、低信頼時の棄却動作。

したがって、現時点のV35の用途主張は「フィラメント候補抽出とinstanceカタログ作成の候補」
までに限定する。chirality推定やspace-weather運用への適合は未検証と明記する。

## 3. 科学・実運用評価契約

### 3.1 個体検出とカタログ品質

- filament単位のRecall / Precisionと、太陽面1枚あたりのFP数
- 大きさ、コントラスト、太陽縁距離、混雑度で層別したRecall
- split / merge率、個体数誤差
- 面積、周長、主軸長、重心位置の誤差

PQ、SQ、RQは継続して併記するが、科学トラックの単独採否指標にはしない。

### 3.2 形態保存

- GT spineに対するcenterline recall / precision、clDice、中心線距離
- 接続成分数、端点数、分枝数、長さの誤差
- Boundary F1と、細い構造に限定したRecall
- barb保持率と方向誤差

MAGFiLOにはspineとchiralityがあるが、barbの独立ラベルが利用可能とは限らない。
barb評価は、既存フィールドから再現可能なproxyを定義するか、盲検の小規模専門家評価セットを
別途作成してから行う。ラベルがない状態でbarb性能を主張しない。

### 3.3 下流の太陽物理タスク

- left / right / unidentifiableを含むchirality macro-F1、クラス別Recall、校正誤差
- 低信頼予測を棄却したときのcoverage-accuracy曲線
- 予測maskまたはspineを入力した場合とGTを入力した場合の性能差
- 半球則など既知の集団統計との整合は補助的sanity checkとし、個体精度の代替にしない

CME・フレアとの関連を扱う場合は、画像単位のsegmentation評価とは別の時系列データ分割、
イベント定義、リードタイム、偽警報率が必要である。相関だけで予報性能を主張しない。

### 3.4 時系列と運用頑健性

- 時系列trackの継続率、ID switch、寿命、面積・位置変化の誤差
- 観測年代を分離したtime-block holdout
- metadataで識別可能な場合のGONG観測所leave-one-site-out評価
- seeing、雲、blur、彩度異常、太陽縁距離別の性能と失敗率
- 1画像あたり平均/p95推論時間、VRAM/RAM、入力異常時のfail-safe
- JPEG入力と原観測/FITS入力のparity、および前処理依存性

## 4. データ分割とリーク防止

1. 同一observationの全annotator variantは、従来どおり同一splitへ固定する。
2. ランダムgrouped CVはコンペ比較用として維持する。
3. 科学・運用評価では、別にtime-block holdoutを作り、モデル選択へ使わない。
4. 観測所metadataが十分ならleave-one-site-outを追加する。metadataの欠損率を先に監査する。
5. chiralityや時系列の閾値調整に、最終holdoutやKaggle testを使わない。
6. 同一または近接時刻の観測がtrain/holdoutへまたがる場合は、時間窓単位でgroup化する。

## 5. モデル開発方針

### 5.1 基準モデル

- `V35-Competition`: 現在の`YOLO11m-seg + shrink refiner + 排他化`を凍結する。
- `V34-Topology-Probe`: semantic-first、clDice、boundary lossの負の結果を、PQだけでなく
  spine・接続性指標で再評価する。競技ゲート不合格を科学的価値の否定とみなさない。

### 5.2 次期候補 `V35-Sci`

最初から新規学習へ進まず、S0評価でV35の科学的な失敗モードを確定してから設計する。
候補となる出力は次のとおり。

- instance mask
- spine / centerline
- chirality（left / right / unidentifiable）と信頼度
- 必要性が実測された場合のみbarb補助出力または形態保持loss

`shrink`のhard intersectionは競技上有効だったが、proposal外の真の細構造を復元できない。
V35-Sciでは、中心線整合や境界確信度を根拠にした限定的な外側復元を候補にする。ただし、
PQ低下と形態改善を同時に測り、見た目だけで採用しない。

## 6. 二段階の昇格ゲート

### S0: 評価基盤の確立

1. MAGFiLOのspine、chirality、観測時刻、観測所metadataを監査する。
2. V35、raw YOLO、V34のOOF predictionを同一対象で評価する。
3. §3の各指標についてbaseline、欠損、算出不能条件を記録する。
4. 大小、太陽縁、低contrast、密集、annotator disagreementで層別する。
5. baselineを見た後に、S1の主要/副次endpointと最小改善幅を事前登録する。

S0完了前は、科学トラックの数値合格線を推定で置かない。

### S1: 科学・実運用モデルへの昇格

次をすべて満たす場合だけ、V35-Sciを科学・実運用候補へ昇格する。

- 事前登録した主要科学endpointを改善し、bootstrap信頼区間を報告する。
- filament Recall、split/merge、chiralityなど重要な副次endpointを破壊しない。
- time-block holdoutで改善方向が再現する。
- site holdoutが可能な場合、単一観測所への依存を評価する。
- worst-caseを目視監査し、改善がアーティファクトやラベル漏れ利用でないと説明できる。
- latency、失敗検知、モデル・データprovenanceを保存する。

PQは互換性指標として必ず報告するが、S1ではPQ単独の上昇を昇格理由にしない。

## 7. 直近の実行順序

1. **S0-1 annotation audit**: spine/chirality/metadataの利用可能性と分布を確定する。
2. **S0-2 evaluator**: spine、接続性、boundary、split/merge、層別集計を実装してテストする。
3. **S0-3 baseline rescore**: V35、raw YOLO、V34を再採点する。
4. **S0-4 visual review**: 条件を隠した比較画像で、細い枝、spine、個体分離を監査する。
5. **S0-5 endpoint registration**: 結果を見てV35-Sciの単一主仮説とS1ゲートを固定する。
6. その後にだけV35-Sciを学習する。Kaggle提出はCompetitionトラックのゲートで別に判断する。

## 8. 論文化と文書更新のルール

- Challenge reportでは、V35を「PQ整合モデル」と記述し、科学・運用有用性は未検証と明記する。
- 拡張論文では、PQだけでなく形態、下流タスク、time/site holdoutを主要結果に含める。
- 各実験ログに`track: competition | scientific | operational`、主要endpoint、採否ゲートを記載する。
- 「高PQ」「見た目が良い」「高速」のいずれか単独から、実運用適合を主張しない。
- 指標、split、入力形式、アノテーション範囲が違う結果を同じ順位表で比較しない。
- 未実施の評価は`unknown`または`not evaluated`と記録し、成功扱いしない。

## 9. 参照

- [V35 PQ再整合報告](V35_PQ_REALIGNMENT_REPORT_2026-08-15.md)
- [実験経緯・技術引継ぎ](SOLAR_FILAMENT_EXPERIMENT_HANDOVER_2026-08-15.md)
- [技術報告書草稿](TECHNICAL_REPORT_DRAFT.md)
- [MAGFiLO dataset paper](https://www.nature.com/articles/s41597-024-03876-y)
- [clDice](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf)
- [EdgeAttNet](https://arxiv.org/abs/2509.02964)
