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
| Code repository URL | `https://github.com/shingo257/solar-filament-2026-solution` — **作成済み(private)。締切日に public 切替** |
| (あれば) Final submissions | Kaggle ref `57039986` (prod) + `57072525` (plain_mhsa_f2) |
| (あれば) Public score | 0.39 (両提出とも) |

## リポジトリ方針 — A 決定・実行済み (2026-10-19)

- **A. スタンドアロン repo (採用・作成済み)**: `Solar_Filament_Segmentation/` を
  subtree 相当で履歴ごと切り出し、private repo
  `shingo257/solar-filament-2026-solution` へ push 済み (main、711 files)。
  ルートに `requirements.txt`・`solar_filament_prod_repro_colab.ipynb`・
  `solar_filament_submission_reproduction.ipynb`・`report/main.pdf` が揃う。
  抽出側の追従ブランチは ER-FlowScan の `solar-solution-v4`
  (sync commit = main@837eccc3)。**締切日に public 切替するだけでよい。**
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
- [x] ~~リポジトリを public にする操作~~ → **private repo 作成済み**;
  締切日に `gh repo edit --visibility public` で切替のみ
- [ ] **重みを repo 内に同梱 (2026-10-23 主催者要件: topic 724503)**
  — 「stored in the repository AND loaded in the notebook」。
  必要重みの実測 (Drive 上の reviewed artifacts):
  `yolo_fold{0,1,2}_best.pt` 45.3MB×3 + `mediseg_edgeattnet_refiner_
  all_data_resume.pt` 141MB + `ablation_plain_mhsa_seed8407_fold2.pt`
  の `best_model` state_dict 141MB ≈ **計 ~420MB**
  - GitHub 1file 上限 100MB 超のため **Git LFS 採用が第一案**
    (public repo: storage 1GB・帯域 1GB/月 — 審査 clone 数回なら許容)
  - 代替案: <100MB チャンク分割 + notebook 内で再結合 (LFS 帯域
    quota 不要・体裁は劣る)。fp16 化は原 bytes の SHA256 provenance
    を失うため非推奨
  - repro notebook は repo 内相対パスから重みをロードする形に更新
    が必要。Kaggle dataset 公開は補助として残置可
- [ ] prod + plain_mhsa_f2 を Kaggle UI で最終選択 (B1)

## 提出前の最終確認リスト

1. repo public + 上記ファイル群がログアウト状態で閲覧できる
2. **repo 内の weights (LFS) が公開 clone で取得できる** +
   notebook が repo 内パスから重みをロードできる
3. report PDF が最新 (v72/v73 SSL 負結果まで反映済み・本文4p再検証済 2026-10-19 #400)
4. Kaggle 最終選択が prod + plain_mhsa_f2 になっている
5. form 送信後に確認メール/スクリーンショットを保管
