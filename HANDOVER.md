# HANDOVER — Solar Filament Segmentation (2026-10-11 時点)

## 2026-10-12 の統合・検証範囲

以下の実験結果は作者の既存記録であり、今回のコード統合で再実行した結果ではない。
CPU 契約テストとローカル独立レビューは、GPU 実行・独立 fold 評価・公式スコアの受入を代替しない。
Drive の prod 再現 CSV は1172件のIDがローカルと一致し、RLE文字列は1171件一致したが、
過去実行の入力・コード・モデル・分割・出力を結ぶ manifest は確認できていない。
別の `kernel_run` 分割コピーには489/707件の割当相違があるため、どちらも新たな正本として採用しない。
master と提出用再現 notebook、および v67 では、モデルに対応する外部凍結分割の SHA256 を明示して検証する。
v68 の旧 v64 キャッシュ経路は現行の検証契約と接続できず、実行・採用判断を保留する。
本文の「完走」「最終」等は既存記録の状態を表し、今回の独立再現を意味しない。

別AIへの引き継ぎ資料。まずこのファイルを読めば現在地・次にやるべきこと・
環境の罠がすべて分かる。実験の全履歴は `WINNING_STRATEGY_2026-08-29.md`、
レポート草稿は `report/main.tex` + `TECHNICAL_REPORT_DRAFT.md`、
提出物チェックリストは `report/REPORT_NOTES.md` が正本。

## 0. TL;DR

- 目標: Solar Filament Segmentation Challenge 2026 (Kaggle) 優勝追求。
  **締切 11/15** — 4p report + 公開repo + requirements.txt + 全パイプライン
  再現notebook が必須。
- 現状: **OOF PQ 0.4432**(fold2 val、YOLO11m-seg → refiner → ownership)。
  LBの0.55帯は MAGFiLO リーク由来(運営が除去予告済み)で、誠実ティアの
  最前線と位置づけ。
- **定量レバー全滅確定 (2026-10-11 最終確定)**: v54-v68 全15機構 +
  A1 scorer(OOF 通過 → **LB 0.37 で反転・不採用**) + v82 soft(5/6負) +
  v67(−0.0087)。新たな定量実験を安易に始めないこと — 残る得点源は
  定性30%(レポート・再現性・コード品質)と提出物整備のみ。
- **本番構成 = prod(V52 レプリカ)、LB 0.39 確定**。ローカル推論
  `scratch/run_test_submission.py --mode prod` → CSV提出(計算枠消費0、
  日次5枠)。Colab GPU 再現済み(1172/1172一致)。
- **次の最優先タスクは提出パッケージ完成**(§5)。残りは Overleaf 4p 確認・
  requirements 凍結・repo 公開化。

## 1. コンペの要点

- タスク: 2048×2048 GONG H-α 画像から filament をインスタンス分割。
  評価は instance-level **PQ**(IoU>0.5 マッチング)。
- データ: train 707観測(MAGFiLO v1.0 COCO JSON、image_id単位で
  同一物理観測の複数annotationあり → file_nameでまとめて扱うこと)。
- **データ契約**: file_name = `YYYYMMDDHHMMSS + サイト1文字 + h.(jpeg|fits)`。
  サイトコード: H-α1文字→磁気図2文字 `B→bb C→ct L→le M→ml T→td U→ud`。
- リーク禁止: MAGFiLO 非train版・テスト GT・外部GT(SWEFil等)は禁止。
  LBの高得点クラスタはリーク由来で踏まないこと。
- ルール確認済みの合法レバー: 学習時の無ラベルGONGフレーム・共時刻磁気図
  (両方試して棄却済み)。推論時の外部入力は禁止 = 推論は単一画像のみ。

## 2. リポジトリ・パス

- worktree: `C:\work\code\worktrees\devin-solar-next-pilots\`
  ブランチ `devin/solar-next-pilots`(push済み、main直push禁止・PR経由)
- **並行調査レーン**: `C:\work\code\worktrees\devin-solar-competitor-survey\`
  ブランチ `devin/solar-competitor-survey-2026-10`(v80/v81/v82 実験の正本、
  PLAN_2026-10-09_FINAL_SPRINT.md §7 が設計書)
- モジュールルート: `Solar_Filament_Segmentation/`
- Drive 成果物(Windows): `G:/マイドライブ/solar_filament_v30/`
  = Colab側 `/content/drive/MyDrive/solar_filament_v30/`
- 主要Drive artifacts:
  - `competition_data/` (train JSON + 画像)
  - `artifacts/grouped_folds.csv` (fold定義、file_name基準)
  - `artifacts/v64_ckpt_fold2/{val,trainA,trainB}/*.pkl`
    (提案キャッシュ: row={name,props}, prop={conf,label_hit,feat,raw,refined})
  - `runs/yolo_fold_{0,1,2}/weights/best.pt`
  - `artifacts/mediseg_edgeattnet_refiner_fold{0,1,2}_resume.pt` +
    `mediseg_edgeattnet_refiner_all_data_resume.pt`
  - `preprocessed_v30_q96/*.jpeg` (前処理済2048²)
  - `solar_filament_v80/candidates/fold_{0,1,2}/<stem>.npz` — v80 全候補
    キャッシュ(YOLO+component-tree、refiner適用済、707/707 完走)
  - `solar_filament_v80/repo/` — v80 候補生成用の Colab コードスナップショット
  - `solar_filament_v34/prediction_cache/oof/semantic_fold_{0,1,2}/<stem>.png`
    — V34 密確率マップ(uint16、v80 tree 候補のソース)

## 3. 本番パイプライン(確定構成)

YOLO11m-seg @1280 (3fold、seed3407+fold offset、AdamW lr7e-4 wd5e-4、
mosaic0.20/最終10ep off、±180°回転) → conf=0.26, NMS IoU0.55 →
MediSegEdgeAttUNet refiner (crop320, ctx1.65, thr0.45) →
ownership解決(mask非重複化、min_area20)。fold2を採用。

再現notebook(提出物): **`solar_filament_submission_reproduction.ipynb`**
(builder: `scratch/build_submission_reproduction.py` — 生成物は必ず
builder経由で再生成、直接編集しない)。Kaggle `/kaggle/input` ⇔ Colab Drive
両対応・添付重み→Drive→再学習の3段解決・inline前処理fallback・
pip freeze + SHA256記録・`submission_efficiency.json`出力。

## 4. 実験全結果(すべて棄却)

採否ゲート(事前宣言): 凍結点 ΔPQ>+0.005→別fold独立確認、
|Δ|<0.002→ノイズ棄却、負→棄却。操作点は trainB で選び val に凍結適用
(valでグリッド探索しない — oracle は診断のみ)。

| 系列 | 機構 | 凍結ΔPQ |
|---|---|---:|
| v54-v61 | EMA/multi-ch/skeleton/oversample/conf-sweep/DT-loss/copy-paste/SAM2 | 全棄却 |
| v63 | merge後処理 | oracle+0.0035(事前棄却) |
| v64 | 静的特徴selector | +0.0000 |
| v65 | TTA-consensus/selector-v2 | +0.0007 |
| v66 | RF-DETR union | −0.0276 |
| v67 | temporal近傍増強(±20分同ラベル流用、データ3倍) | −0.0087 |
| v68 | 磁気図PIL特徴selector-v3(最後の新情報源) | **+0.0000** |

重要な負の知見:
- 残り FP751 は conf≥0.26・refiner・ownership・3view-TTA・磁気図の
  **すべてを通過する難しい誤検出** — selector系では構造的に届かない。
- 残り FN1071 の68%は detector-blind(proposal自体が無い)。
  RF-DETR は box 相補カバー 317/612 を持つが union は FP 洪水で敗北。
- v68 sanity: filament重心→PIL中央値63px vs ランダム81px —
  物理信号は実在するが分離力不足。
- val oracle 天井 ≈ +0.046(proposal scoring の限界)。

## 5. 残タスク(優先順) — 2026-10-11 時点の状態更新

1. ~~重みの公開 Kaggle dataset 化~~ → **完了(2026-10-11)**:
   `shingo257/solar-filament-2026-weights` (private→公開化は提出時に切替)。
   yolo_fold{0,1,2}_best.pt + refiner_all_data.pt + a1 scorer 3点 + manifest。
   fold 別 refiner checkpoint は容量の関係で未同梱(Kaggle/Colab 側で再学習可)。
   **v2 (2026-10-12)**: `ablation_plain_mhsa_seed8407_fold2.pt` を追加 —
   主催者要件「選択した両方の提出を再現できること」で第2提出
   (plain_mhsa_f2)の再現に必須のため。
2. ~~再現実行~~ → **完了(2026-10-11, Colab GPU 版)**:
   `solar_filament_prod_repro_colab.ipynb`(3セル直列構成)が A100 で完走、
   `artifacts/repro_prod_colab/` に submission_prod.csv + efficiency。
   ローカル prod CSV と filament 一致率100%(IoU>0.5)・99.91% IoU=1.0。
   **本番 notebook の完全版再現は未実施 — 提出に使うなら Kaggle 側での
   実行が残る**(kernel パッケージ `output/kaggle_kernel/` 準備済み)。
   notebook は `CFG.SUBMISSION_VARIANT="plain_mhsa_f2"` で第2提出も
   再現できるよう拡張済み(2026-10-12、未実行 — F3 のクリーン環境
   通し実行で prod/variant 両経路を検証すること)。
3. **Overleaf で `report/main.tex` ビルド** → 4ページに収まるか確認。
   A1 LB反転+v82棄却を追記済み(2文追加、ページ影響は微小の見込み)。
4. `requirements.txt` 凍結(最終推論環境の pip freeze)。
5. repo 公開化(重み外部参照は済、READMEの提出vs実験の区別は追加済み)。
   **両 worktree の未pushコミットが多数**(solar-competitor-survey 側 ~23、
   solar-next-pilots 側 ~3)。push してから公開化。
6. (低優先) Dice/IoU分布・断片化/マージ統計の図 — 審査項目に対応する
   資産として `v62` 解剖結果が使える。
7. ~~plain-attention refiner 再提出~~ → v82 で汎用改善が出なかった(5/6負)ため
   優先度さらに低下。提出スロットの価値としては乏しい。

## 6. 環境の罠(実測済み)

- **`cv2.imread` は非ASCIIパス(`G:/マイドライブ`)を読めない** →
  必ず `cv2.imdecode(np.fromfile(path), ...)` 経由。
- **GONG HTTP は ~50req/短時間で遮断** → DL済みファイルはローカル索引
  (`local_mag_files`)から引く。`fetch` にはリトライ実装済み。
- **Devin CLIのバックグラウンドシェルはセッション再起動で死ぬ** →
  長時間ジョブはチェックポイント保存必須(v68は
  `output/v68_feats_partial.pkl` に25ファイル毎保存、
  `output/v68_rows_with_feats.pkl` があれば1-3スキップ)。
- **DriveFS は再起動直後が遅い**(pkl読み込みに10分かかった実績)—
  ハングと誤認しない。
- `pycocotools` の正しいimport: `from pycocotools import mask as mask_utils`
  (`pycocotools.mask_utils` はこの環境で欠落)。
- **ノートブックへのモジュールソース埋め込みで `from __future__ import
  annotations` を除去すること** — これは名前 `annotations` を `_Feature`
  オブジェクトとしてセル名前空間に束縛し、coco `annotations` dict を
  上書きする(`'_Feature' object is not subscriptable` 事故)。
  `scratch/build_v82_soft_target_refiner.py` の `_embed_source` が正本。
- `defaultdict.get(k)` は default_factory を呼ばず None を返す — 埋め込み
  データセットで TypeError 事故。`defaultdict[k]` を使うこと。
- DriveFS の一時的 `OSError errno 22` は `imread_unicode_with_retry` /
  `load_yolo_cache_with_retry` / `_read_bytes_retry` で吸収する。
- 並列書き込みが絡む成果物は tmp+rename の atomic write(`--skip-existing`
  と組み合わせでローカル/Colab 並走可)。
- `*.pkl`/`*.pt`/`*.log`/`output/` は gitignore — コミットしない。

## 7. 実験プロトコル(新規実験をするなら)

- v64 ckpt の提案キャッシュを再利用して selector 系はローカル実行可能
  (GPU不要)。`scratch/v68_magnetogram_probe.py` が完全な参照実装
  (解決→写像→特徴→selector→trainBグリッド→凍結val)。
- 新機構も同じ採否ゲート・同じ trainA学習/trainB選定/val凍結に従うこと。
- 採用判定は manifest に記録 → `WINNING_STRATEGY_2026-08-29.md` と
  `TECHNICAL_REPORT_DRAFT.md` に結果(棄却含む)を必ず記録。
- commit: Conventional Commits、`feat(solar):`/`fix(solar):`/`docs(solar):`。
  モジュール外に触れない・main直push禁止・`--no-verify`禁止。

## 8. キーファイル地図

| ファイル | 役割 |
|---|---|
| `solar_filament_submission_reproduction.ipynb` | 提出用再現notebook(審査対象) |
| `scratch/build_submission_reproduction.py` | 上記のbuilder(正本) |
| `scratch/v68_magnetogram_probe.py` | selector系ローカル実験の参照実装 |
| `scratch/build_v67_temporal_aug.py` | v67 builder(棄却済み・再現用) |
| `scratch/fetch_temporal_neighbors.py` | GONG近傍フレーム解決プローブ |
| `scratch/competitor_recon_2026-10-07.md` | LB構造・競合手法・ルール偵察 |
| `WINNING_STRATEGY_2026-08-29.md` | 全実験履歴・戦略の正本 |
| `TECHNICAL_REPORT_DRAFT.md` | レポート草稿(md版) |
| `report/main.tex` | 4p公式レポート(Overleafテンプレ) |
| `report/REPORT_NOTES.md` | 提出物チェックリスト |
| `solar_filament/metrics/metrics_official.py` | 公式PQ実装 |
| `solar_filament/pipeline/ownership.py` | マスク非重複化 |
| `tests/` | metrics 19件・submission 14件等 |
