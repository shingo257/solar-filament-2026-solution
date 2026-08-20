# 作業引継ぎ — Solar Filament（2026-08-16 夜）

## 1. いまの結論（30秒）

| 項目 | 値 |
|---|---|
| 提出正本 | **V35**（Public **0.37**） |
| 全fold OOF | PQ **0.4426**（`conf 0.26 + shrink@0.45 + drop_contained_then_area_asc`） |
| 採用ゲート | all-fold PQ > **0.4526**、最悪fold ≥ V35 実測 |
| 解像度レバー | **停止**（P2-2・P2-3 とも不合格） |
| 次にやること | **near-miss 診断**（学習ではない）→ 分類結果で refiner / 検出器を分岐 |

正本リポジトリ: `C:\work\code\ER-FlowScan\Solar_Filament_Segmentation`
ブランチ: `feat/solar-filament-pq-realignment`（コミットはユーザー指示があるまでしない）
詳細経緯: `SOLAR_FILAMENT_EXPERIMENT_HANDOVER_2026-08-15.md`（改訂6まで反映済み）

使わない: `C:\work\code\Solar_Filament_Segmentation`（git なしの旧スタブ）

---

## 2. P2 解像度実験（完了・不合格）

仮説: mask prototype 解像度不足 → near-miss が TP になる。
ゲート（fold0）: PQ > **0.4393**（V35 fold0）。超えなければ fold1/2 に進まない。

| 探針 | 内容 | 後処理つき fold0 PQ | 判定 |
|---|---|---:|---|
| P2-2 | YOLO を imgsz **1536** で fold0 再学習 | **0.4295**（Δ −0.0098） | 不合格 |
| P2-3 | V30 重みのまま推論だけ **1536**（学習なし・ローカル CPU） | **0.4320**（Δ −0.0073） | 不合格 |

### 生YOLO切り分け（refiner なし・同一 conf 0.26）

| 検出器 | conf_first | drop_contained |
|---|---:|---:|
| V30 1280 推論 | 0.3977 | 0.4049 |
| P2-2 1536 再学習 | 0.3996 | 0.4076 |
| P2-3 1280学習・1536推論 | **0.4026** | **0.4121** |

**読み方:** 生YOLOでは 1536 がわずかに勝つ。確定後処理を掛けると負ける。
主因は「検出器破壊」ではなく **1280 向け shrink@0.45 と 1536 提案の相性**。
**1536 OOF で後処理を再チューニングして採用判定してはいけない**（単一fold 選択バイアス）。

成果物:

- `output/v36/p2_2_imgsz1536_fold0.json`
- `output/v36/p2_2_raw_yolo_fold0.json`
- `output/v36/p2_3_infer_imgsz1536_fold0.json`
- `output/v36/p2_3_raw_yolo_fold0.json`
- Drive: `solar_filament_v35/runs/yolo_imgsz1536_fold_0`、OOF `prediction_cache/oof/yolo_imgsz1536_fold_0`（236 npz）
- TEMP（P2-3 推論 cache）: `%TEMP%\solar_p2_3\oof_npz`

---

## 3. 中断した作業（ここから再開）

ユーザー指示「進めて」→ **case diagnostics** を開始したが **未完了**。

### 3.1 起きたこと

1. `python scratch/run_case_diagnostics.py --sheets 30 --threads 14` を実行
2. Drive 上の `mediseg_refiner_fold*_resume.pt` を `torch.load` したところで
   `OSError: [Errno 22] Invalid argument`（日本語 Drive パス／同期の問題）
3. 対策として `%TEMP%\solar_diag_ckpts\` へ checkpoint をコピーする方針に変更
4. **fold0–2 すべて** `%TEMP%\solar_diag_ckpts\` へコピー済み（各 496840065 bytes）。3つとも `torch.load(..., weights_only=False)` 成功確認済み
5. `run_case_diagnostics.py` は TEMP 優先読み込みにパッチ済み（`refiner_checkpoint_path` + `weights_only=False`）
6. 診断 CSV / contact sheet は **まだ無い**（`output/v35/case_diagnostics.csv` 未生成）

再開時はコピー済みなので、そのまま診断を走らせてよい。

### 3.2 再開手順（この順）

```powershell
cd C:\work\code\ER-FlowScan\Solar_Filament_Segmentation

# 1) fold0–2 の refiner を TEMP に揃える（未コピーなら）
$dst = Join-Path $env:TEMP "solar_diag_ckpts"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
foreach ($f in 0,1,2) {
  $src = "G:\マイドライブ\solar_filament_v30\artifacts\mediseg_refiner_fold${f}_resume.pt"
  $out = Join-Path $dst "mediseg_refiner_fold${f}_resume.pt"
  if (-not (Test-Path -LiteralPath $out)) {
    Copy-Item -LiteralPath $src -Destination $out -Force
  }
}
Get-ChildItem $dst | Format-Table Name, Length

# 2) 画像別診断（707観測・CPU・数十分〜1時間超の見込み）
python scratch/run_case_diagnostics.py --sheets 30 --threads 14

# 3) near-miss 被覆モード分類（同じく全fold・時間がかかる。診断後でも可）
python scratch/run_near_miss_modes.py --threads 14
```

期待出力:

- `output/v35/case_diagnostics.csv`
- `output/v35/contact_sheets/worst*.png`（損失上位30）
- `output/v35/near_miss_modes.json` / `.csv`

### 3.3 分類の意味（次実験の分岐）

| near-miss の主モード | 次の本実験 | やらないこと |
|---|---|---|
| **overcoverage**（pred が太すぎ） | より強く「削る」系（形態学・PQ向け refiner 再設計） | guided / 膨張ブレンドの安易な復活 |
| **undercoverage**（細い枝欠け） | 検出器を **imgsz 以外**で強化（モデル大・aug・seed） | imgsz 1536 再探針 |
| **shape_mismatch** | 検出器の形状／位置 | 後処理閾値の fold0 再スイープだけ |
| アノテータ不一致が主 | 合意寄り予測の探針 | Public 連打 |

ゲート（本実験）: all-fold PQ > **0.4526**、最悪fold ≥ V35。

---

## 4. 確定パイプライン（触らない）

- YOLO: V30 `yolo_fold_{0,1,2}/weights/best.pt`（imgsz **1280** 学習）
- OOF cache: `G:\マイドライブ\solar_filament_v34\prediction_cache\oof\yolo_fold_*`（707 npz）
- Refiner: `mediseg_refiner_fold{0,1,2}_resume.pt`（リークなし・各fold除外学習）
- 後処理: `shrink@0.45` + `drop_contained_then_area_asc`、conf **0.26**
- 提出ノート: `solar_filament_v35_final_submission_colab.ipynb`（builder: `scratch/build_v35_final_submission.py`）
- 公式指標: **Panoptic Quality**（`src/metrics_official.py`）。Ultralytics mAP は採用判定に使わない

### パス注意

- YOLO / torch に **日本語 Drive パスを直接渡さない**（P2-3 と同様 TEMP コピー）
- `from __future__ import annotations` を Colab セルに埋め込むと、変数名 `annotations` が壊れる → `coco_anns` を使う
- `iou_matrix` の `iscrowd` は `len(truth)`（`len(pred)` は誤り）
- このフォルダで `uv run` は numpy 無しになりがち → `python tests/test_*.py` / `python scratch/...`

---

## 5. やらないこと（再確認）

- imgsz 1536 の再学習・推論探針
- 1536 OOF だけで後処理を再チューニングして採用
- conf だけの再スイープ
- OOF 未通過での test 提出連打
- 明示指示なしの git commit / push
- V30 の `runs/yolo_fold_*` を上書き

---

## 6. 関連スクリプト一覧

| ファイル | 役割 | 状態 |
|---|---|---|
| `scratch/run_case_diagnostics.py` | 画像別 CSV + contact sheet | TEMP 読み込みパッチ済み・**未実行完了** |
| `scratch/run_near_miss_modes.py` | near-miss を over/under/shape 分類 | 新規・**未実行** |
| `scratch/run_p2_3_infer_local.py` | P2-3 ローカル推論 | 完了 |
| `scratch/run_p2_3_raw.py` | P2-3 生YOLO採点 | 完了 |
| `scratch/run_p2_imgsz_raw_pq.py` | P2-2 生YOLO切り分け | 完了 |
| `scratch/build_v36_yolo_imgsz_probe.py` | P2-2 Colab ビルダー | 不合格で終了 |
| `scratch/build_v36_infer_imgsz_probe.py` | P2-3 Colab ビルダー | 不合格で終了 |

---

## 7. 次エージェントへの最短指示

1. `%TEMP%\solar_diag_ckpts` に fold0–2 の `.pt` が揃っているか確認（各 ~497MB）
2. `run_case_diagnostics.py` を完走させる
3. `run_near_miss_modes.py` を完走させる
4. `mode_counts` を見て §3.3 の分岐で **次の1本の本実験**だけ設計する
5. HANDOVER 本編（`SOLAR_FILAMENT_EXPERIMENT_HANDOVER_2026-08-15.md`）に改訂7として診断結果を追記する

診断が終わるまで新しい学習ノートは作らない。
