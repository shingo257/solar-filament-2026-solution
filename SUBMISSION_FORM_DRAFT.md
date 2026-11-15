# Google Form 提出ドラフト (2026-11-15 締切)

> ⚠️ **2026-11-15 最終確定 (`0.40` 突破・差替ゲート成立)**:
> - **Kaggle Submissions 実画面**: **「最終リーダーボードに最大 1 件選択 (select up to 1 submission for final leaderboard)」** と表示 (2026-11-11 実画面確認)。
> - **締切**: **Kaggle 最終提出・チェック選択締切 = 本日 `2026-11-15 15:00:00 JST` (`06:00 UTC`)** / **Google Form 締切 = `2026-11-15 23:59 CDT` (`11/16 14:59 JST`)** — 事故防止のため **両方とも本日 `15:00 JST` 前に完了** させる。
> - **最終選択 (最大 1 件枠)**: **`58363293` (`cons2of3_sptal_maj3` / `submission_v137_cons2of3_sptal_prod_z_c26_maj3.csv`, Public LB `0.40`)** で確定！事前申告ルール (`public > 0.39 → v96-full から差替`) を Fold-2 honest OOF `0.45803 (+0.01380)`・2-Fold Small-TAL OOF `0.45238 (+0.00810, P>0=1.000)`・Public LB **`0.40`** で正式通過。(※万一 UI 側で 2 件選択可能な場合のみ、第 2 枠に単一検出器アンカー **`57897560` (`prod_v96full`, Public LB `0.39`)** を追加選択する。)

競技 overview の必須項目は「4ページ技術報告書 (Overleaf テンプレ・単一PDF)」と
「公開 Git リポジトリ URL」の2点。Form 本体は Kaggle overview の
"submit through this google form" リンクから。記入ミスで委員会が
資料にアクセスできない場合は最終評価の対象外になる旨明記されている。

> このファイルは記入用ドラフト。実送信はユーザーが Kaggle アカウントで行う。
> Form の実際の設問一覧は送信前に画面で確認すること (下記は想定項目)。

## 想定設問と回答案

| 設問(想定) | 回答 |
|---|---|
| Name | Shingo Hara |
| Affiliation / Organization | `ER SYSTEMS,Ltd.` (report と完全一致) |
| Email | shingo257@gmail.com (report と一致) |
| Kaggle team name | **`ERS`** (Leaderboard 表示名と完全一致) |
| Kaggle username / account | `shingo257` |
| Team members | Shingo Hara (単独) |
| Technical report (PDF) | `report/main.pdf` — **厳密 4ページ全体** (本文 + Section 4 Acknowledgment 原文 + References を含む; Overleaf テンプレ準拠) |
| Code repository URL | `https://github.com/shingo257/solar-filament-2026-solution` — **public 済み・匿名 clone + `git lfs pull` 全 8 重み SHA256 一致検証済み** |
| (あれば) Final submissions | **最終選択 (1枠)**: Kaggle ref **`58363293`** (`cons2of3_sptal_maj3`: 全データ `YOLO11m-seg` + `YOLO26m-seg` + `YOLO11m-seg [Spine-Aux 0.5 + Small-TAL 4.0]` の 1対1 二部合意 `2-of-3 @ 0.26` + 3票ピクセル多数決 `majority_3plus`、**Public LB `0.40`**、事前申告 `> 0.39` 差替ゲート通過、CSV SHA256 `9e229401…`、1,136 rows、overlap=0)。<br>**バックアップ単一検出器アンカー (2枠目がある場合の選択)**: Kaggle ref **`57897560`** (`prod_v96full`: 全データ `YOLO11m-seg` 単体 + 同一 EdgeAttNet refiner・ownership、Public LB `0.39`、CSV SHA256 `684ff6d6…`)。予備: `57039986` (`prod`, LB `0.39`) |
| (あれば) Public score | **`0.40`** (`58363293` `cons2of3_sptal_maj3`) / バックアップ単体アンカー `0.39` (`57897560` `prod_v96full`) |
| (あれば) Method/efficiency | 3 チャネル物理前処理 (`solar_three_channel`: raw percentile + Gaussian 周辺減光平坦化/CLAHE + black-hat) → 3 本の直交全データ検出器 (`YOLO11m-seg`, `YOLO26m-seg`, `YOLO11m-seg + Spine-Aux P3 + Small-TAL`) の 1対1 二部マッチング (`IoU >= 0.45`, `2-of-3 @ 0.26`) → EdgeAttNet U-Net 境界 refiner (`320px` crop) + 3 票ピクセル多数決 (`majority_3plus`) → 面積昇順 disjoint ownership (`overlap = 0`)。A100 で ~1.0s/image (3 検出器合計)、単体アンカーは T4 で ~1.1s/image |

## リポジトリ方針 — A 決定・実行済み (2026-10-19 → 2026-11-15 最終同期)

- **A. スタンドアロン repo (採用・public 済み)**: `Solar_Filament_Segmentation/` を
  `shingo257/solar-filament-2026-solution` として公開 (2026-10-24 public 化済み)。
  ルートに `requirements.txt`・`solar_filament_submission_reproduction.ipynb`・
  `solar_filament_v139_sptal_clean_repro_colab.ipynb`・
  `solar_filament_v96full_clean_repro_colab.ipynb`・`report/main.pdf`・`weights/` (Git LFS 全 8 重み) が揃う。
  `RESEARCH_LOG.md` には v99–v126 全実験 (v116/v120 漏洩発見と撤回・v121 honest 再実行含む) および
  **v127–v137 (`Spine-Aux` + `Small-TAL` + 1対1 二部マッチング + `majority_3plus` による Public LB `0.40` 突破)** まで完全記録済み。
- **重み同梱要件 (2026-10-23 主催者追記、topic 724503)**:
  「weights are stored in the repository and are loaded in the notebook」に対応し、
  全 8 チェックポイント (~563MB / 537MiB) を standalone repo `weights/` (Git LFS) + `weights_manifest.json` に同梱済み。

要件チェック (2026-11-15 最終確認):

- [x] `requirements.txt` あり (バージョン付き、`ultralytics>=8.3.0,<8.5.0` で YOLO11m-seg / YOLO26m-seg 両対応)
- [x] 全体パイプラインを示す notebook あり
  (`solar_filament_submission_reproduction.ipynb`: `cons2of3_sptal_maj3` [`58363293`, LB `0.40`]・`prod_v96full` [`57897560`, LB `0.39`]・`prod` [`57039986`]・`plain_mhsa_f2` [`57072525`] の全バリアントを `weights/` 第一優先で再現)
- [x] 4ページ報告書 `report/main.pdf` (本文 + Section 4 Acknowledgment 原文 + References で **厳密 4 ページ**、Overleaf テンプレ準拠)
- [x] リポジトリ public 化済み (`https://github.com/shingo257/solar-filament-2026-solution`)
- [x] 全 8 重みを `weights/` (Git LFS) + `weights_manifest.json` に同梱・SHA256 検証済み:
  - `yolo_fold{0,1,2}_best.pt`
  - `refiner_all_data.pt`
  - `refiner_plain_mhsa_fold2_s8407.pt`
  - `yolo_full_fold2_best.pt` (`f54c812d…`)
  - `yolo26m_all_fold0_best.pt` (`25002d23…`)
  - `yolo_spine_small_tal_all_fold2_best.pt` (`f6623992…`)
- [ ] **Kaggle UI (`Submissions` タブ) で `58363293` (`cons2of3_sptal_maj3`, Public LB `0.40`) に最終選択チェックを入れる** (最大 1 件枠のためこれ 1 件を選択。万一 2 件枠なら `58363293` + `57897560` を選択)
- [x] 再現 notebook (`solar_filament_submission_reproduction.ipynb`) に `cons2of3_sptal_maj3` / `prod_v96full` バリアント追加済み (欠落時はフォールバックせず停止、`refiner_all_data.pt` 同梱時は fold 0–2 refiner 再学習を自動スキップして約 5 分で完走)
- [x] GPU クリーン再現ラッパ `solar_filament_v139_sptal_clean_repro_colab.ipynb` 整備・単体テスト通過済み
- [x] `report/main.tex` → `report/main.pdf` を 3 図 (`Figure 1–3`) + `0.40` 突破結果に更新して再ビルド (**厳密 4 ページ目視確認済み**)

## 提出前の最終確認リスト

1. **Kaggle `Submissions` 画面の最終選択チェック (`15:00 JST` 厳守)**:
   - **`58363293` (`submission_v137_cons2of3_sptal_prod_z_c26_maj3.csv`, Public Score `0.40`)** にチェックが入っていることを目視確認・スクリーンショット保存 (最大 1 件選択 UI のためこれ 1 件。2 件選択可能なら第 2 枠に `57897560` [`0.39`])。
2. **公開 repo ログアウト閲覧 + 匿名 LFS 確認**:
   - シークレットウィンドウで `https://github.com/shingo257/solar-filament-2026-solution` が開け、`report/main.pdf`・`solar_filament_submission_reproduction.ipynb`・`weights/weights_manifest.json` が閲覧できること。
3. **Google Form の PDF アップロードは一発で確定させる** (topic 749524):
   - 再送信はファイル差替ではなく**追加**になり旧ファイルは消せない (委員会は「最後の提出」を見ると明言)。
   - アップロード前に `C:\work\code\solar-filament-2026-solution\report\main.pdf` が **4 ページ・`0.40` 記載の最新版** であることを開いて確認してから送信する。
4. Form 送信後に確認画面 / 確認メールのスクリーンショットを保管する。
