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
| Technical report (PDF) | `report/main.pdf` — **4ページ全体** (本文 + acknowledgments + references を含む; Overleaf テンプレ準拠) |
| Code repository URL | `https://github.com/shingo257/solar-filament-2026-solution` — **public 済み・匿名 clone/LFS 検証済み** |
| (あれば) Final submissions | Kaggle ref `57039986` (prod) + `57897560` (v96-full 全データ学習、ゲート +0.0099 通過)。代替候補: `57072525` (plain_mhsa_f2 アーキテクチャ hedge)。`57933888` (prod_dedup030) は v99 で fold0/1 再現せず (Δ −0.0038/−0.0019、§2.75) のため候補から除外 |
| (あれば) Public score | 0.39 (上記4提出とも; conf032 A/B `57843737` も 0.39、§2.57/§2.72) |
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
  (最新 = v99-v125 全実験キャンペーン §2.75-§2.99 全記録・
  v109 再測定修正値・v112 fold-0 複製判定・v96-full byte-identical
  再現証跡・conf032 成果物・v115 MRCNN 結果記録・
  **v116/v120 漏洩発見と撤回・v121 honest 再実行 (cons2of3_c26
  +0.002982 < +0.005 → unverified_fold0)・v125 輪郭投票
  (+0.001588 < +0.005 → vote_negative) まで含む**)。
  ER-FlowScan 側最終 sync = `devin/solar-v120-consensus` @ 9f86af71。
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
- [ ] prod + v96-full を Kaggle UI で最終選択 (代替: plain_mhsa_f2、LB 0.39)
- [x] 再現 notebook に `prod_v96full` バリアントを追加 (検出器のみ
  `yolo_full_fold2_best.pt`、欠落時は fold 重みへフォールバックせず停止)
- [x] `yolo_full_fold2_best.pt` (sha256 `f54c812d…`、45MB) を standalone repo
  `weights/` (LFS) + `weights_manifest.json` に追加して push
  (standalone `e5b1bc13`、公開 clone + `git lfs pull` で実体確認済み)
- [x] `prod_v96full` を実行し、提出済み CSV (`57897560`) と再現性を確認
  — ローカル隔離 CPU 環境の `run_test_submission.py --detector full` で
  SHA256 `684ff6d6…` が全バイト一致 (RESEARCH_LOG §2.77)。GPU クリーン
  環境 (L4) では byte 差は GPU FP 非決定論の範囲で予測等価
  (全 1,141 filament が 1-to-1 対応・平均 IoU 0.9993・全件 IoU≥0.9、
  §2.77 GPU 補完結果)
- [ ] report §Submission evidence を最終2枠 (prod + full-data detector) に更新して再ビルド

## 提出前の最終確認リスト

1. repo public + 上記ファイル群がログアウト状態で閲覧できる
2. **repo 内の weights (LFS) が公開 clone で取得できる** +
   notebook が repo 内パスから重みをロードできる
3. report PDF が最新 — **v92 flow-field・v93 consensus-class・
   §2.60–2.63 の3-fold誤差分解・v94/v95 reject・disjoint提出制約
   まで反映済み**。本文+ack+refs でちょうど4ページに圧縮済み
   (PR #584・standalone `d3bcc2ed`)。conf032 A/B (public 0.39) は
   §2.57 に記録 — report 本文は確定済み
4. Kaggle 最終選択が prod + v96-full になっている (§2.72: dedup030 は
   局所 +0.0075 だが LB 0.39 で可視改善なし、§2.75: 独立 fold でも
   再現せず採用見送り、探索結果は RESEARCH_LOG と v97b/c/d manifest に記録済み。
   **2026-11-11 確定: v120 consensus 系は fold-0 clean 複製で事前登録
   ゲート未達 (+0.0030 < +0.005)、v125 輪郭投票も未達 (+0.0016) のため
   最終ペアは prod `57039986` + v96-full `57897560` で確定**)
4b. **repo が選択した両提出を再現できる** (747960): prod と prod_v96full の
   両バリアントが repo `weights/` だけで解決できること
5. **Form の PDF アップロードは一発で確定させる** — topic 749524 の
   主催者回答: 再送信はファイル差替ではなく**追加**になり、
   旧ファイルは消せない。委員会は「最後の提出」を見ると明言。
   誤った PDF を上げた場合は訂正版を**末尾に追加**する形で対処
6. form 送信後に確認メール/スクリーンショットを保管
