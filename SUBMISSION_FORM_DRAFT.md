# Google Form 提出ドラフト (2026-11-15 締切)

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
| Email | shingo257@gmail.com (report と一致) |
| Kaggle account / team name | `shingo257` (要 UI 確認: leaderboard の表示名) |
| Team members | Shingo Hara (単独) |
| Technical report (PDF) | `report/main.pdf` — 4ページ本文 + acknowledgments/references |
| Code repository URL | `https://github.com/shingo257/solar-filament-2026-solution` — **public 済み・匿名 clone/LFS 検証済み** |
| (あれば) Final submissions | Kaggle ref `57039986` (prod) + `57072525` (plain_mhsa_f2) |
| (あれば) Public score | 0.39 (両提出とも; conf032 A/B `57843737` も 0.39、§2.57) |
| (あれば) Method/efficiency | detect-then-refine 2 構成。主催者 rubric は品質目視 + feasibility (理解可能な実装を高評価、topic 724503 追記) — 単純で再現可能な構成をそのまま記述 |

## リポジトリ方針 — A 決定・実行済み (2026-10-19)

- **A. スタンドアロン repo (採用・public 済み)**: `Solar_Filament_Segmentation/` を
  subtree 相当で履歴ごと切り出し、
  `shingo257/solar-filament-2026-solution` へ push。**2026-10-24 に public 化済み**
  (PR #563、匿名 clone + `git lfs pull` 5/5 重み sha256 一致を実測確認)。
  ルートに `requirements.txt`・`solar_filament_prod_repro_colab.ipynb`・
  `solar_filament_submission_reproduction.ipynb`・`report/main.pdf` が揃う。
  抽出側の追従ブランチは ER-FlowScan の `solar-solution-v4`
  (sync commit = main@837eccc3)。以降の更新は standalone main へ直接同期済み
  (最新 `415d788c` = conf032 成果物含む)。
- ~~B. モノレポを公開~~: 不採用 (他モジュールを晒すため)。
- ~~weights dataset `shingo257/solar-filament-2026-weights` の public 化は
  要件再確認まで保留 (2026-10-19 ユーザー決定)。~~ **要件確定済み
  (2026-10-23 主催者追記、topic 724503)**: 「weights are stored in the
  repository and are loaded in the notebook」— repo 内格納が必須。
  Kaggle dataset 参照だけでは reproducibility 要件を満たさない可能性。
  → **weights を standalone repo 内に同梱する計画に変更**（下記参照）。

要件チェック (2026-10-15 時点で確認済み):

- [x] `requirements.txt` あり (バージョン付き)
- [x] 全体パイプラインを示す notebook あり
  (`solar_filament_submission_reproduction.ipynb`: prod と
  plain_mhsa_f2 の両提出を再現)
- [x] 4ページ報告書 `report/main.pdf` (本文4p + refs、テンプレ準拠)
- [x] ~~リポジトリを public にする操作~~ → **public 化済み**
  (2026-10-24、LFS 418MB 含む; 匿名 clone + `git lfs pull` で
  5 重みの sha256 一致を実測確認済み、§2.49)
- [x] **重みを repo 内に同梱 (2026-10-23 主催者要件: topic 724503)**
  — **実施済み (PR #478, standalone `weights/` + Git LFS)**:
  `yolo_fold{0,1,2}_best.pt` + `refiner_all_data.pt` +
  `refiner_plain_mhsa_fold2_s8407.pt` (best_model 抽出、124 tensors
  verified equal) + `weights_manifest.json` ≈ **計 ~418MB (399MiB、
  公開 clone の `git lfs pull` で実測確認済み)**。
  notebook は `cwd/weights` を第一経路で解決 (manifest sha256 検証、
  不一致は ValueError) → dataset → Drive → 再学習の順。
  2026-10-26 にローカル clone で E2E 検証済み (5 重み全て repo 解決・
  fold CSV sha256 `24367120…` 一致)。残るのは public 化後の
  匿名 clone + `git lfs pull` 再確認のみ
- [ ] prod + plain_mhsa_f2 を Kaggle UI で最終選択

## 提出前の最終確認リスト

1. repo public + 上記ファイル群がログアウト状態で閲覧できる
2. **repo 内の weights (LFS) が公開 clone で取得できる** +
   notebook が repo 内パスから重みをロードできる
3. report PDF が最新 — **v92 flow-field・v93 consensus-class の
   負結果と annotator 不一致天井まで反映済み** (本文5p + refs、
   PR #575)。conf032 A/B (public 0.39 で prod と表示同等) は
   §2.57 に記録 — report 本文は確定済み
4. Kaggle 最終選択が prod + plain_mhsa_f2 になっている
5. **Form の PDF アップロードは一発で確定させる** — topic 749524 の
   主催者回答: 再送信はファイル差替ではなく**追加**になり、
   旧ファイルは消せない。委員会は「最後の提出」を見ると明言。
   誤った PDF を上げた場合は訂正版を**末尾に追加**する形で対処
6. form 送信後に確認メール/スクリーンショットを保管
