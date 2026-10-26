# Final Phase Plan — Solar Filament Segmentation 2026

作成: 2026-10-26 ・ 締切: **2026-11-15** ・ 根拠: RESEARCH_LOG §2.28-2.34
（全41スレッド本文精読・LB 完全CSV・kernel 調査済み）

提出要件の変更点の詳細は同梱の `SUBMISSION_REQ_CHANGE_2026-10-23.html`
（または RESEARCH_LOG §2.34）を参照。本書は残り全作業の実行計画。

---

## Phase A — 必須作業: weights-in-repo 対応（審査要件、先行作業可）

要件 (10-23 topic 724503): 「weights are stored in the repository and are
loaded in the notebook」。現行の Kaggle dataset 参照だけでは不適合リスク。

| # | タスク | 内容 | 検証 |
|---|---|---|---|
| A1 | weights-only エクスポート | `ablation_plain_mhsa_seed8407_fold2.pt` から `best_model` state_dict のみ抽出 (282→141MB)。`scratch/export_submission_weights.py` として残す | 全 tensor を `torch.equal` で原本と一致確認 (CPU・GPU 不要) |
| A2 | SHA256 provenance 登録 | 新 bytes の sha256 を計算し manifest/RESEARCH_LOG に登録。既存 reviewed SHA は正本維持 | manifest 整合テスト |
| A3 | LFS 設定 (standalone repo) | `solar-filament-2026-solution` に `.gitattributes` (`*.pt filter=lfs`) + `weights/` 配置。計 ~420MB (yolo 45.3MB×3 + edgeatt 141MB + mhsa 141MB + scorer 小物) | `git lfs ls-files`、ポインタ/実体サイズ確認 |
| A4 | repro notebook 更新 | 重み解決順序: ① repo 内 `weights/` → ② Kaggle dataset (fallback)。builder + review-gate テスト更新 | 既存 reproduction テスト + notebook builder 一致性 |
| A5 | OOF sanity（任意だが推奨） | LFS 経由の重みで fold2 OOF の baseline arm を再実行、0.444238 の bit 一致を確認 | Colab ~15min (proposals cache 再利用可なら更に短縮) |
| A6 | 締切日の公開化 + 匿名 clone 検証 | `gh repo edit --visibility public` → ログアウト環境で clone + LFS 取得 + notebook 実行 | ユーザー締切日作業に合流 |
| A7 | Kaggle dataset 公開 (補助) | `solar-filament-2026-weights` を public — LFS 帯域枯渇時の冗長経路。README に「repo が正本」と明記 | dataset public 後に notebook fallback 経路確認 |

リスク: LFS 帯域 1GB/月は審査 clone 数回で使い切りうる → A7 の冗長
経路と README のフォールバック注記で緩和。fp16 化は provenance 喪失で
非推奨 (§2.34)。

## Phase B — 残存実験レーン (計算資源投入枠)

全レーンの合法性は §2.34 の主催者裁定表で確定済み。いずれも**事前登録
ゲート (+0.005 @ fold2 OOF PQ0.45) なしに実施しない**。採否は実測のみ。

| # | レーン | 合法性 | 過去の試行 | 事前見込み | コスト | 優先度 |
|---|---|---|---|---|---|---|
| B1 | **verifier rescue** — 候補拡大 (tile/low-conf) + verifier が conf<0.26 候補を**採用** | — | v72 は filter 方向のみ失敗 (除去すると単調悪化)。rescue 方向は未検証 | 低〜中 (verifier 識別力 corr 0.48 が上限) | ~1-2h Colab (verifier 推論+PQ) | **中** — detector-blind FN (FN の68%) への残る唯一の構造レバー |
| B2 | label transfer ±5min (GONG archive) | 許可 (748045, ±1h test 除外) | 広幅 temporal-neighbor は stale label で負済み。±5min 同 site は未試行 | 低 | ~1400 files DL が GONG 403 で困難 (749225) | 低 (取得リスク大) |
| B3 | magnetogram aux (GONG bzi) | 許可 (743521, 訓練専用) | 未試行 | 低〜中 (極性反転線 prior は filament 物理に整合) | 大 (aux head 設計+再学習+bzi 取得) | 低〜中 |
| B4 | 別天文台 SSL (KSO/Meudon/BBSO) | 許可 (748498) | GONG SSL 3連敗 (v72-74) — より遠いドメイン | 最低 | 大 (各天文台 archive の取得系新規構築) | 最低 |
| B5 | seed-channel refiner | — | 未試行 (jianansun 経由の Tim Gluhkikh 系) | 低 (ソース LB~0.409 ≈ 我々) | 中 (refiner 再学習) | 見送り推奨 |

**推奨シーケンス**: B1 をまず閉じる (安価・既存 npz/verifier 資産で即実施可)。
B2-B4 は B1 の結果と残日数で判断 — 全て実施しても情報価値は限定的だが、
「やれることを全てやった」記録としての価値はある。計算資源無制限の
方針なら B1 → B3 → B2 → B4 の順。

## Phase C — 締切日 (11-15) チェックリスト

1. standalone repo `shingo257/solar-filament-2026-solution` を public 化
   (`gh repo edit --visibility public`)
2. 匿名環境で clone → LFS 重み実体の取得 → repro notebook 完走を確認
3. Kaggle dataset `solar-filament-2026-weights` public 化 (補助経路)
4. Kaggle UI で最終提出選択: prod `57039986` + plain_mhsa_f2 `57072525`
5. Google Form 送信 (ユーザー操作) — repo URL + `report/main.pdf`
   (再提出可・最大10ファイル、749524)
6. 送信確認メール/スクリーンショットを保管

## Phase D — 継続監視 (締切まで)

- discussion/LB の periodic refresh: 2-3 日毎。特に:
  - 主催者の最終提出アナウンス (724503 への追記が要件通知の主経路)
  - 「LB クリーニング」の詳細発表
  - トップ層の手法公開 (現時点で 0.56+ は全て非公開)
- 新規 topic は 10-21 以降ゼロ — 沈静化しているが、締切直前に
  最終要件の追加通知が出る可能性があるため継続監視は必須

## 採用ステータス (2026-10-26 時点)

- **prod + plain_mhsa_f2 維持** (LB 0.39、ERS rank 200/1196)
- 全15機構 + TTA 2系統 + ensemble が事前固定ゲートで閉鎖済み
- 誠実系フロンティア認識 (~0.41-0.49) はコミュニティ・主催者
  (「0.41 very seriously」「0.35+ has merit」) 双方と整合
