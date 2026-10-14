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
| Code repository URL | **要決定**: 下記「リポジトリ方針」参照 |
| (あれば) Final submissions | Kaggle ref `57039986` (prod) + `57072525` (plain_mhsa_f2) |
| (あれば) Public score | 0.39 (両提出とも) |

## リポジトリ方針 — 2案

- **A. スタンドアロン repo (推奨)**: `Solar_Filament_Segmentation/` を
  `git subtree split` で履歴ごと切り出し、新規 public repo (例:
  `shingo257/solar-filament-2026-solution`) へ push。
  ルートに `requirements.txt`・`solar_filament_prod_repro_colab.ipynb`・
  `solar_filament_submission_reproduction.ipynb`・`report/main.pdf` が
  そのまま揃う。他モジュールを晒さない。
- **B. モノレポを公開**: ER-FlowScan 全体を public 化。
  手順は最小だが他モジュールのコード・未整理ファイルも公開される。
  コンペ要件は「コンペ終了と同時に公開・受賞発表まで維持」なので
  締切日に public 切替すればよい。

要件チェック (2026-10-15 時点で確認済み):

- [x] `requirements.txt` あり (バージョン付き)
- [x] 全体パイプラインを示す notebook あり
  (`solar_filament_submission_reproduction.ipynb`: prod と
  plain_mhsa_f2 の両提出を再現)
- [x] 4ページ報告書 `report/main.pdf` (本文4p + refs、テンプレ準拠)
- [ ] **リポジトリを public にする操作** (締切日に実行)
- [ ] **重み Kaggle dataset `shingo257/solar-filament-2026-weights` を
  public 化** — repro notebook はこの dataset を参照するため、
  「追加ファイル・許可なしで再現」の要件に必要
- [ ] prod + plain_mhsa_f2 を Kaggle UI で最終選択 (B1)

## 提出前の最終確認リスト

1. repo public + 上記ファイル群がログアウト状態で閲覧できる
2. weights dataset public + notebook が dataset を解決できる
3. report PDF が最新 (v70 ・再調査 §6.27 まで反映済み)
4. Kaggle 最終選択が prod + plain_mhsa_f2 になっている
5. form 送信後に確認メール/スクリーンショットを保管
