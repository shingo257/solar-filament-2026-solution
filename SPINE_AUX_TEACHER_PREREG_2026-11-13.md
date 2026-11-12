# Spine 補助教師 A/B — 事前登録 (2026-11-13)

新規実験計画 (2026-11-11 ユーザー提示) の分割・条件・ゲートを、**いかなる
実験結果の閲覧より前に**ここへ固定する。本書は計画の実行仕様であり、
解釈が分かれた場合はこの文面を正本とする。学習比較は締切 (11/15) 後の
研究枠。締切前に行うのは監査・再現確認のみ。

## 1. 分割の固定 (実測済み・台帳 pin)

台帳: `output/s0/time_block_assignment.csv` (707 obs、`file_name,fold,
site_code,year,time_block_role`)。実消費 fold 分割は Drive
`artifacts/grouped_folds.csv` sha256
`febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e`。

| 用途 | 定義 | 件数 |
|---|---|---|
| パイロット fit | `time_block_role=development` ∩ fold {0,1} | 367 obs |
| パイロット eval | `development` ∩ fold 2 | 180 obs |
| 時系列確認 fit | `development` 全体 547 − 除外リスト | ≤547 |
| 時系列確認 eval | `locked_evaluation` 全体 160 − 除外リスト | ≤160 |

- `locked_evaluation` (2017-2022、160 obs) は**パイロットの fit・eval の
  両方から完全除外**する (febb3f67 fold 2 は dev 180 + locked 55 の混合
  なので、パイロット eval は dev 制限の 180 obs とする)。
- 除外リスト: 確認観測 (locked) とその前後 ±5 日に該当する観測を、dev 側
  fit・eval および時系列確認側 fit から除く。**算出済み・pin 済み**:
  `output/s0/spine_aux_prereg_exclusions_2026-11-13.csv` sha256
  `a56372ea6493a6d3571910e60f291606b864a4f84a61baf1df475a6627270b0a`
  (4 件 — dev 側 2016-12-29/12-31 の 2 件、locked 側 2017-01-02/01-03
  の 2 件)。実効数: パイロット fit 365・eval 180、時系列確認 fit 545・
  eval 160 (locked 側 2 件は計画文言上 fit のみ除外、eval には残す)。
- bootstrap: 観測単位の対応あり再標本化 (アノテータを一緒に)。
  時系列確認では 5 日ブロック単位の区間ブートストラップ — locked_eval は
  5 日ギャップ基準で **58 独立クラスタ** (148 観測日) を持つことを確認済み。
- 既知の制限: dev 3-fold 内に同一時刻 (±0-1min)・異サイト (GONG B/C/L/M/T/U)
  の準同一画像ペアが 8 組存在 (RESEARCH_LOG 2026-11-13 記録)。画像外観の
  記憶経路は残るが注釈セットは独立 (別 labeler・個体数差大)。全 arm 同一
  条件のため A/B 比較は公平。dev/lock 境界を跨ぐ同時ペアは 0 件を確認済み。

## 2. arm 定義

| arm | 学習 | 推論 |
|---|---|---|
| A 対照 | YOLO11m-seg、prod 相当の画像・前処理・損失・学習レシピ | 単一 Hα 画像 → 検出 → 共通 fold 内 refiner → ownership |
| B spine 補助 | A と同一 + 共有 P3 特徴から spine ヒートマップを予測する小ヘッド 1 個。spine を同一座標変換でラスタライズし、正負別平均 BCE を aux 損失としてバックボーンに伝播 | aux ヘッド除去、A と同じ経路 |

固定条件: 両 arm 同一の初期検出器重み・seed・fit データ・入力 1280・
70 epoch・拡張・バッチ。**外側評価で checkpoint を選ばず 70 epoch 最終
重みを使う**。refiner は各 split の fit 内のみで学習した共通モデル。
conf floor 0.08、採用 conf 0.26、refiner 閾値 0.45、ownership/drop
0.90、min area 20 px。rasterize は P3 座標で 1 セル半径の正例帯、
欠損 spine・padding は教師対象外。mosaic/crop/flip で mask と spine の
幾何変換一致を学習前検証。aux 損失係数は fit-only スモークで 1 値に固定。

spine 被覆率 (事前監査実測): `annotation_spine_audit.csv` 8,199 注釈で
spine 有効率 100% (even・finite・完全 in-bounds)、GT ≤1024px² の
3,476 件も全件 spine あり。

## 3. 指標とゲート (事前固定)

主指標: native 2048 pooled ΔPQ (conf 0.45 採用、既存採点器と同一)。
機構指標: GT 面積 ≤1024px² の提案 recall (floor 0.08、最大 IoU>0.5) と
conf ≥0.26 recall。併記: TP/FP/FN、SQ/RQ、all-spine coverage、merge/split、
秒/画像、アノテータ合意数・limb・密集層別、対応あり bootstrap、
時系列確認では 5 日ブロック CI。

- **パイロット継続条件**: dev ΔPQ ≥ +0.010 かつ対象層提案 recall ≥ +2pt
  かつ最終 FP/画像 相対増 ≤5%。未達なら結果と誤差内訳を記録して終了。
- **時系列確認の採用条件**: 別分割で 2 arm 新規学習、ΔPQ ≥ +0.005 かつ
  ブロック bootstrap 95% CI 下限 >0 かつ recall 改善と FP ガード再現。
  十分な独立ブロックが取れない場合は探索結果扱い。
- 現行候補 (v96-full `57897560`) の置換判断は時系列確認通過後のみ。

## 4. 記録要件

各実行にコード・依存ロック・設定・画像/注釈・実消費 split・重み・出力の
sha256 と件数を保存。キャッシュキーは画像+split+重み+前処理+推論設定。
親観測の split 交差・予測重複・欠落・注釈 ID 不整合で停止。
旧採点との一対一マッチ差は対照 arm でも計測し、差があれば両方記録。
