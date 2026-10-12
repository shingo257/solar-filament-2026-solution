# 最終スプリント実験計画(2026-10-09 → 2026-11-15)

`WINNING_STRATEGY_2026-08-29.md` §6.23(競合・Biohub調査)を受けて策定した、
締切までの約5週間の詳細計画。**Colab計算資源には余裕がある**という前提で、
期待値の低い施策も「反証可能な実験」として積極的に回し、負の結果も
報告書の材料として残す。数値の正本は `README.md`、戦略判断の履歴は
`WINNING_STRATEGY_2026-08-29.md`、本文書は**実行計画と進捗台帳**。

---

## 0. 目標と採用ゲート(Iteration Compact)

| 項目 | 内容 |
|---|---|
| 目的 | 誠実ティア最前線(Public 0.40 / OOF 0.4597)を OOF で上回る構成を、締切の1週間前(11/08)までに確定し、報告書・公開リポジトリとともに提出する |
| 成功指標(主) | 全fold OOF 公式PQ(707枚、fold別refiner・リークなし) |
| 成功指標(副) | Public LB(≈90枚・2桁)は「順位が崩れていない」確認のみ。提出間比較には使わない |
| ガードレール | 推論時間(定性評価に「計算効率」が含まれる)、報告書4ページ、再現notebook1本で全段再現できること |
| 許容できないミス | 外部GT・推論時の近傍フレーム利用(規約違反)、fold跨ぎのリーク、fold0単独結果の一般化 |
| 許容できるミス | 期待値の低い実験が negative で終わること(報告書の §Rejected に載せる) |
| ベースライン | OOF 0.4432(plain_mhsa refiner、3fold micro PQ)/ 提出正本 V52 Public 0.39 |
| 採用ゲート | (a) 3fold すべてで Δ の符号が正 **かつ** (b) observation-cluster bootstrap 95%CI がゼロを除外。seed 2本で学習する施策は seed 間でも符号一致 |
| 提出切替ゲート | 採用ゲートを満たした構成を all-data で再構築し、公式RLE検証を通したうえで提出。Public が V52 より下がっても OOF を信じるが、−0.02 以上の下落は原因調査 |
| ロールバック | V52 の提出物・重み・manifest は凍結保持。最終提出は V52 と新構成の2本を Google Form で説明 |

---

## 0.5 【2026-10-09 判明】並行レーン(PR #244 `devin/solar-next-pilots`)との整合

本計画の起草後、Drive の `artifacts/` に v58〜v68 の manifest があることから、
別レーン(worktree `devin-solar-next-pilots`、PR #244、main 未マージ、最終コミット
2026-10-09 11:39)が 10/05〜10/09 に以下を**既に実験・棄却済み**と判明した
(同レーンの WINNING_STRATEGY §6.17〜6.22)。本計画はそれを前提に下記の通り修正する。

| 同レーンの結果 | 本計画への影響 |
|---|---|
| v54 EMA: fold2 両seed負 → **3fold で不採用確定** | **C1 を削除** |
| v59 距離変換 boundary loss: 3fold・6観測で ±0、符号 3対3 → 不採用 | **C5 を削除** |
| v58 conf 掃引: prod の conf=0.26 が fold0/fold2 でピーク一致 | 「conf を一律に動かす」余地なし。A1 は conf ではなく候補別 û で選ぶ設計なので矛盾しないが、期待値は下方修正 |
| v60 copy-paste、v57 oversampling: 密集リコール系は FP 増で正味負 | D 系の recall 施策(D2 SAHI、D3)は**選別器が先に成立しない限り着手しない** |
| v61 SAM2 box-prompt refiner: −0.035 | 本計画に無し(変更なし) |
| v62 エラー解剖(fold2): FN blind 561 / weak 168 / near-miss 333、FP orphan 136 / misassigned 196。oracle: perfect_verifier **+0.046**、drop_orphan +0.017、merge +0.0035 | FN の 68% は**提案不在**。「YOLO 候補の中から選ぶ」だけでは blind FN に届かない → **候補集合の拡張(E1)が本質** |
| v64 静的特徴 HistGBM selector(10特徴、fold2 内 A/B 分割): **Δ+0.0000**、重要度は conf のみ | **A1 を「YOLO 候補のフィルタ」として単独で走らせる期待値は低い**(ほぼ同型の負の結果)。A1 の設計差(post-refiner 期待IoU の hurdle 目標、τ=PQ/2、3fold cross-fit で学習データ約3倍、conf≥0.08 の候補拡張)は残るが、**E1 の候補拡張とセットでのみ実施**し、単独 arm は E1 内の対照として1本だけ置く |
| v65 TTA-consensus 特徴 selector: 凍結点 +0.0007、val oracle +0.0052 だが操作点が転移せず | 同上。B1(refiner 確率の TTA 平均)は検出器 TTA 特徴とは別機構なので残すが、too.foo 氏の「crop-IoU 改善は PQ に移らない」を負の事前分布として期待値は中→低 |
| v66 RF-DETR union: −0.028、ただし blind GT 612 のうち RF-DETR@0.10 が IoU>0.5 で 317 をカバー | 候補源の相補性は実在。E1 で木候補を足す際、**source-aware な記述子**を入れる根拠 |
| v67 時間近傍フレーム増強: −0.0087 | 本計画に無し(変更なし) |
| v68 磁場図補助信号: 実現性確認済み・実験中 | **D4 は同レーンに委譲**、本計画から削除 |
| §6.21/6.22 競合調査(10/06・10/07): too.foo 氏の selection oracle +0.083、GBM selector v2 で 0.398→0.423 | §6.23 と相補。too.foo 氏の selector が効いたのは**連結成分候補(閾値1本の semantic 候補)に対して**であり、YOLO 候補に対する v64 では効かなかった — 候補源の性質で selector の価値が変わることを示す |

**修正後の重心**: (1) **D1 密な確率マップ → E1 ハイブリッド候補 + 期待IoU スコアラー**
(同レーン未着手、tush 氏の +0.06 の構造そのもの、blind FN に届く唯一の機構)、
(2) **C2 soft annotator-mean 目標**(未着手)、(3) C3 回転正規化・C4 アーキ差替
(未着手、refiner 系の負の事前分布あり)、(4) B1。
A1 単独・C1・C5・D4 は実施しない。

**番号の衝突回避**: 同レーンが §6.17〜6.22 と v58〜v68 を使用済みのため、本調査は
WINNING_STRATEGY **§6.23**、実験 ID は **v80〜** を使う。PR #244 のマージ後に
本ブランチを rebase して §番号の重複が無いことを確認する。

---

## 1. 施策カタログ(トラック別)

記法: **期待値** = 事前の主観確率×効果量の目安。**コスト** = Colab 実行時間の見込み。
「実験 ID」は notebook/builder/manifest の命名に使う(`v80_...` から採番(v58〜v68 は並行レーン PR #244 が使用済み))。

### トラック A — 決定層(CPU のみ、既存 OOF cache)

最有力。tushhaaarrrr 氏の +0.06(固定閾値比)の源泉であり、§6.16 の結論
「次は提案品質側」とも一致。GPU 不要なので最初に着手し、他トラックと並行する。

| ID | 施策 | 仮説 | 設計の要点 | コスト | 期待値 |
|---|---|---|---|---|---|
| **A1 (v80)** | post-refiner 期待IoU スコアラー + τ=PQ/2 選択 | conf 0.26 の固定足切りを、候補ごとの期待IoU û=ĥ·m̂ で置き換えると、低conf の真陽性を拾いつつ FP を抑えられる | 候補 = OOF cache の conf≥0.08 全件に fold別 refiner(shrink@0.45)を適用。教師 u = (1/R)Σ_r IoU(候補, r の最良GT)·1[IoU>0.5]。記述子: YOLO conf・面積・伸長度・refiner 確率の mean/max/>0.5比率・shrink 比(refined/proposal 面積)・中心からの距離 ρ・近傍候補数・他候補との最大IoU・画像内候補数。LightGBM hurdle(ĥ: 二値、m̂: TP のみ回帰)、**GroupKFold を fold 境界と一致させて cross-fit**、行重み R。選択 û>τ → 既存排他化 → 公式PQ。τ は 0.18〜0.30 を掃引し、PQ/2 の自己無撞着点と比較 | CPU 数時間(refiner 適用は Colab T4 で 1〜2h、以降 CPU) | **高** |
| A2 (v80b) | A1 の hard-negative 2周目 | OOF で û を過大評価した FP を 1:1 で混ぜて再学習すると FP がさらに減る(Biohub 1位 Soon Net の手法) | A1 の OOF 予測から û>τ かつ u=0 の候補を抽出し、重み2で再学習。cross-fit 維持 | CPU 1h | 中 |
| A3 | τ のグローバル性検証 | 画像別・密度別の τ は効かない(tush 氏 E38、Dinkelbach の帰結) | 画像あたり候補数・R で層別した τ を試し、グローバル τ に劣ることを確認 | CPU 30min | 低(確認的) |
| A4 | 排他化後の再スコア | 排他化で削られた候補の û が τ を割る場合に除外すると FP が減る | 排他化後マスクで記述子を再計算し û を再評価 | CPU 1h | 低〜中 |
| A5 | 候補ソースの拡張(E1 への橋渡し) | YOLO 候補に密な確率マップ由来の候補(トラック E)を加えても同じスコアラーで選別できる | E1 の後 | — | — |

A1 の成否判定: 3fold OOF PQ が 0.4432 比で採用ゲートを満たすこと。
V42 verifier(pre-refiner 代理目標、11特徴 logistic、conf0.26 候補のみ、−0.0023)との差分を manifest に明記。

### トラック B — refiner 推論側(Colab 推論のみ、再学習なし)

| ID | 施策 | 仮説 | 設計の要点 | コスト | 期待値 |
|---|---|---|---|---|---|
| **B1 (v81)** | 3fold refiner の確率平均 + flip/rot TTA | V53 の失敗は mask 多数決。確率を平均してから shrink@0.45 すれば境界が安定し ΣIoU が伸びる(Biohub 3位の heatmap 平均、tush 氏 40パス/クロップ) | OOF では各 fold の候補に「その fold を見ていない refiner」しか使えないため、OOF 検証は **TTA(D4、8視点)のみ**。fold 平均は all-data 提出時のみ適用し、OOF で TTA が効くことを根拠に採用する | T4 1〜2h | 中 |
| B2 | マルチスケールクロップ平均 | 320 固定クロップは細い barb を潰す。320+480 の確率平均で境界が改善 | 既存 refiner を 480 入力でも推論(FCN なので可)、平均 | T4 1h | 低〜中 |
| B3 | 出力閾値の理論値確認 | t*=J*/(1+J*)≈0.40(J*≈0.65)が shrink の最適点と一致するか | 既存の閾値掃引(0.45〜0.55 平坦)と照合するだけ | CPU 10min | 確認的 |
| B4 | クロップ→原解像度の貼り戻し精度 | リサイズ往復で 1〜2px の境界損失(tush 氏: 1024→2048 で −0.006) | 貼り戻しを bilinear 確率→閾値 の順に変更(現状が nearest なら) | CPU 30min | 低〜中 |

### トラック C — refiner 学習側(Colab GPU、各 1〜3h/fold)

| ID | 施策 | 仮説 | 設計の要点 | コスト | 期待値 |
|---|---|---|---|---|---|
| **C1 (v54)** | EMA fold1/2 | fold0 で +0.0005・分散縮小。fold1/2 で符号一致なら低コストで採用 | 準備済み notebook `solar_filament_v54_ema_pilot_fold1_2_colab.ipynb` | L4 3.5h | 低(効果量小) |
| **C2 (v82)** | soft annotator-mean 目標 | 現行 refiner は「そのアノテータの硬いマスク」を教師にしている(`InstanceCropDataset` は record 単位)。同一フィラメントを描いた全アノテータのマスク平均(soft)を教師にすると、PQ が報酬する「分布の中心」に収束する(§6.23.2、IoU 0.61→0.71 の算術) | GT 同士を IoU>0.3 でクラスタ化し、クラスタ内マスクの平均を soft target。BCE(soft)+Dice、損失は現行 `refiner_loss` 構造を維持。plain_mhsa を base に 3fold×2seed | L4 1.5h×6 | **中〜高** |
| C3 (v83) | 回転正規化クロップ + 隣接個体チャンネル | 長軸を水平に揃えた 384×128 クロップは細長い構造に有利(tush 氏 stage 2)。v55 の失敗は検証 proposal の RNG 問題を含むため再設計 | minAreaRect で回転、長軸方向 50% パディング、4ch(画像3ch + 他候補マスク)。**検証 proposal は事前計算して固定**。2seed×3fold | L4 2h×6 | 中 |
| C4 (v84) | エンコーダ/デコーダ差し替え | UNet++ R34 / U-Net ConvNeXt-T は tush 氏の stage 2 で plain U-Net より +0.0036。smp で実装可能 | 現行レシピそのまま、アーキのみ差替。ImageNet 事前学習あり | L4 1.5h×6 | 中 |
| C5 (v85) | 距離変換ベース boundary loss | 現行は morphological gradient L1(重み0.15)。Kervadec 型の signed-distance 重み付けで境界精度が上がる | 重み 0.15 を保ち定式化のみ差替。校正を壊す損失(clDice 等)は tush 氏の −0.045 を踏まえ**やらない** | L4 1.5h×6 | 低〜中 |
| C6 (v86) | 学習長延長 + cosine 再始動 | juanmirod 氏は 12 epoch で Dice 上昇中。現行 14 epoch も同様の可能性 | 14→30 epoch、他は同一 | L4 3h×6 | 低〜中 |
| C7 | 上記の有望2件の組合せ | C2+C4、C2+C3 など、単独で採用ゲートを満たしたものだけ組み合わせる | — | — | — |

### トラック D — 検出器・データ側(Colab GPU、長時間。期待値は低いが資源があるので回す)

| ID | 施策 | 仮説 | 設計の要点 | コスト | 期待値 |
|---|---|---|---|---|---|
| **D1 (v87)** | 密な確率マップモデル(soft annotator-mean、U-Net R34、2048 native タイル) | スコアラーの記述子(候補内の平均 p・spread)および E1 の候補源として使える。tush 氏の設計の中核部品 | 512 タイル、BCE+0.5Dice、EMA、D4 aug、3fold。出力は 2048 の確率マップを fold 別に cache | A100 2h×3 | **高(間接)** |
| D2 (v88) | SAHI タイル推論 | 1536 全体リサイズは失敗したが、1280 学習モデルを 1024 タイル(overlap 25%)で推論すると小さいフィラメントの recall が上がる(juanmirod 氏の到達率分析) | 既存 fold 重みで推論のみ。NMS 後に A1 スコアラーへ | T4 1h | 低 |
| D3 (v89) | YOLO 学習の soft 多アノテータ処理 | 現行は record を独立サンプルとして複製。アノテータ数 R で損失重みを割る/合意数で重み付けすると FP が減る | Ultralytics のサンプル重み付けはデータ複製比で近似 | A100 4h×3 | 低 |
| D4 (v90) | 磁場図補助損失(訓練限定、規約許可) | PIL(極性反転線)近傍にフィラメントが乗るという物理事前分布で検出器の FP が減る。tush 氏は n.s. だが報告書の「科学的妥当性」の材料になる | GONG bzi を日時一致で取得、refiner または D1 に補助ヘッド(PIL 距離回帰) | 取得 1日 + A100 2h×3 | 低 |
| D5 (v91) | ラベルなし GONG で MAE 事前学習 | 707 枚の小データで encoder の初期化が効く(Biohub 1位)。tush 氏の noisy student は seed ノイズ水準 | 既存 pseudo-label 用 150 枚 + 追加 1,000 枚、D1 encoder を MAE で 30 epoch 事前学習 | A100 3h + D1 再学習 | 低 |
| D6 (v92) | YOLO26 / yolo11x 1280 | 公開 notebook が YOLO26 full-res で 0.36。V39(11l)は不合格だが世代が変わった | 現行レシピそのまま | A100 4h×3 | 低 |

### トラック E — ハイブリッド候補集合(D1 完了後。大きな賭け)

| ID | 施策 | 仮説 | 設計の要点 | コスト | 期待値 |
|---|---|---|---|---|---|
| **E1 (v93)** | YOLO 候補 ∪ component-tree 候補 を A1 スコアラーで一括選別 | YOLO が出せない小さい/淡いフィラメント(FN≈1000 の主因)を確率マップの木が候補化し、スコアラーが選別する | D1 の確率マップから 15 閾値の max-tree を 2048 native で構築。各ノードを候補とし、YOLO 候補と同じ記述子 + 「どのソース由来か」を付与。選択は û>τ の後、重なりは antichain 制約 + 既存排他化 | CPU 数時間 + メモリ | 中(効けば大) |
| E2 | E1 で選ばれた木ノードを refiner で再整形 | tush 氏 stage 2 と同じ | 既存 refiner を流用 | T4 1h | — |

### トラック F — 定性30%・提出物(GPU 不要、毎週少しずつ)

| ID | 内容 | 期限 |
|---|---|---|
| F1 | `report/main.tex` に §6.23 の事実を反映(Public≈90枚、リーク精査方針、外部GT禁止適合、訓練限定許可項目の不使用明記) | 10/18 |
| F2 | 決定層(A1)の理論説明(Dinkelbach、τ=PQ/2)と README §1 の損益分岐の統合記述 | A1 結果後 |
| F3 | `solar_filament_master_pipeline_colab.ipynb` を最終構成へ更新し、**クリーン環境で通し実行**して `pip freeze` を保存 | 11/08 |
| F4 | Rejected 実験表(本計画の negative 結果)を報告書 §3.3 と README に反映 | 11/10 |
| F5 | 定性図: 改善例/失敗例の 4 パネル比較を最終構成で再生成 | 11/10 |
| F6 | Google Form 提出(報告書 PDF + 公開リポ URL)。提出後に LB 順位・提出 ID を記録 | 11/12(締切 11/15 の3日前) |

---

## 2. 週次スケジュール

Colab は L4/A100 を**2〜3 セッション並列**で回す前提(Pro+)。各 notebook は Drive の
partial manifest から再開できる形(v54/v57 と同じ)で作る。

| 週 | 期間 | CPU(ローカル) | Colab セッション1 | Colab セッション2 | Colab セッション3 | 判断ポイント(週末) |
|---|---|---|---|---|---|---|
| W1 | 10/09–10/15 | A1 教師・記述子生成、LightGBM cross-fit、τ 掃引 | A1 用 refiner 一括適用(conf≥0.08 候補、3fold) → B1 TTA | C1 EMA fold1/2 | D1 密マップ fold0→1→2 | **A1 の3fold 結果**で W2 以降の重心を決める。D1 完了で記述子に p 統計を追加 |
| W2 | 10/16–10/22 | A2 hard-negative、A3/A4、E1 の木構築コード | C2 soft target 3fold×2seed | C4 UNet++/ConvNeXt 3fold×2seed | D2 SAHI、B2 マルチスケール | C2/C4 の fold0 2seed 結果で fold1/2 継続可否 |
| W3 | 10/23–10/29 | E1 fold0 パイロット → 3fold | C3 回転クロップ 3fold×2seed | C5 boundary loss、C6 長期学習 | D4 磁場図取得+補助損失 fold0、D5 MAE 事前学習 | **E1 の採否**。採用ゲート通過施策の一覧を凍結 |
| W4 | 10/30–11/05 | C7 組合せの OOF 評価、提出候補の all-data 再構築計画 | all-data 学習(採用施策分) | D3/D6 探索(余力があれば) | F3 クリーン環境通し実行 | **提出構成の凍結(11/05)**。以降は新実験を開始しない |
| W5 | 11/06–11/15 | 報告書・README・RESEARCH_LOG 更新 | 最終推論・RLE 検証・提出 | 再現 notebook の再実行確認 | — | 11/08 提出物ドラフト完成 → 外部査読1回 → 11/12 Form 提出 |

**原則**: W4 水曜(11/05)以降に始めた実験は提出構成に入れない(fold0 だけで
採用する誘惑を断つ)。間に合わなかった施策は「将来課題」として報告書に残す。

---

## 3. 実験の進め方(全トラック共通)

1. **builder script → notebook 生成**(`scratch/build_vNN_*.py --fold {0,1,2}`)。
   notebook は直接編集しない。全コードセルを `ast.parse`、ヘルパーは `tests/` に単体テスト。
2. **manifest を Drive に逐次追記**(入力 SHA256、fold、seed、GPU 名、ピーク VRAM、所要時間、UTC 完了時刻)。
   中断再開で完了 run を飛ばす。
3. **評価は run 内 paired 比較**(§6.16 の教訓: 検出器を再学習すると絶対値がずれる)。
   同一 notebook 内に baseline arm を必ず置く。
4. **fold0 2seed → 符号一致なら fold1/2** の順。fold0 で |Δ| < seed σ(≈0.001)なら
   その時点で「効果なし」と記録して打ち切り(Colab は余るが、分析時間は余らない)。
5. 採用ゲート通過後、`output/vNN/` に manifest・結果 JSON を保存し、README §2 と
   WINNING_STRATEGY に 1 段落で結果を記す(採否にかかわらず)。
6. 提出は **A1 のスコアラーを含む構成**と**含まない構成**の両方を all-data で作り、
   V52 と並べて LB を見る(順位崩れの監視のみ)。

---

## 4. 判断ツリー

```text
A1(スコアラー)が採用ゲート通過?
├─ Yes → 以後すべての施策は「A1 込み」の OOF で評価する(選別器がある前提で
│         FP 増を伴う recall 系施策 D2/D3/E1 の価値が変わるため)
│         └─ D1 完了 → 記述子に p 統計追加(A1')→ E1 へ
└─ No  → 原因を切り分ける(教師 u の定義 / 候補拡張の有無 / cross-fit の漏れ)。
          それでも No なら「候補の選別は conf でほぼ飽和」と結論し、
          重心を C2(soft target)・C4(アーキ)・B1(TTA)の境界品質側へ移す

C2(soft target)が通過? ─ Yes → C3/C4/C5 は C2 を base に積む
                        └ No  → 現行 hard target を維持、C4 のみ継続

E1(ハイブリッド候補)が通過? ─ Yes → 提出構成の中核にし、報告書の主張を
                                     「YOLO + 木候補 + 学習決定層」に書き換える
                            └ No  → YOLO 単独候補 + A1 を提出構成とする
```

---

## 5. リスクと対策

| リスク | 対策 |
|---|---|
| A1 が V42 と同じく最終 PQ に移らない | 教師を post-refiner・最終候補の u に固定し、cross-fit を fold 境界に一致させる。候補拡張(conf 0.08)の有無を別 arm にして切り分ける |
| OOF と Public の乖離(≈0.05)が新構成で拡大 | 乖離は label 側(tush 氏 E4: 画像特徴では train/test 判別不能)。OOF を信じる方針を維持し、Public −0.02 超のみ調査 |
| Colab セッション切断・quota | 全 notebook を再開可能に。A100 を要する D1/D3/D6 は夜間に投入 |
| fold0 選択バイアスの再発 | 採用ゲート(3fold 符号一致 + CI)を機械的に適用。例外は作らない |
| 定性30%の準備が後回しになる | F1〜F6 を週次表に固定。W5 は実験禁止 |
| LB クリーンアップで順位が動く | 我々は影響を受けない(外部 GT 不使用)。報告書に不使用を明記済み |
| 複数エージェントの並行作業で worktree が衝突 | `tools/agent_worktree.py claim/check` を遵守。実験 ID(vNN)を本文書で採番して重複を避ける |

---

## 6. 進捗台帳(随時更新)

| ID | 状態 | fold0 | fold1 | fold2 | 判定 | 記録先 |
|---|---|---|---|---|---|---|
| A1 v80 (yolo_scorer arm) | 3fold 完了 | +0.0008 | +0.0069 | +0.0038 | **採用ゲート通過**(CI [+0.00057,+0.00735] ゼロ除外) | PLAN §7.1.8c |
| B1 v81 | 完了(tree τ 掃引) | −0.0005 | +0.0067 | +0.0032 | 不採用 — tree τ で単調増加し yolo_scorer 漸近線に収束、tree 追加は不利益と確定 | PLAN §7.1.8c |
| C1 v54 | fold0 +0.0005、fold1/2 notebook 準備済み | +0.0005 | | | 保留 | WINNING_STRATEGY §6.15 |
| C2 v82 | soft 6本完了(manifest セルバグ→ローカル再構築) | −0.0003 | −0.0058 | −0.0054 | **棄却** — mean Δ −0.0034、6セル中5負、seed 間で符号不一致。hard_union は判定不能に不要のため未実行(GPU 節約) | PLAN §7.2、RESEARCH_LOG §2.11 |
| D1 v87 | 未着手 | | | | | |
| E1 v80 (hybrid arm) | 3fold 完了 | −0.0161 | −0.0016 | −0.0246 | **不採用**(CI [−0.0194,−0.0089])。oracle +0.107 で候補集合の価値は確認 | PLAN §7.1.8b |

---

## 7. 設計書(PR #244 マージ待ちの間に確定、マージ後すぐ実装に入る)

### 7.1 v80 — ハイブリッド候補集合 + 期待IoU スコアラー(E1、CPU のみで着手可能)

#### 7.1.1 発見: 密な OOF 確率マップは既に存在する

V34(`scratch/build_v34_semantic_instance_hybrid.py`)が 3-fold で学習した
semantic モデル(U-Net / timm-efficientnet-b4、soft consensus 教師 = 全アノテータ
union mask の平均、BCE+Dice+clDice+boundary、768 patch 学習・1024 tile +
hflip TTA 推論)の **held-out 確率マップが `solar_filament_v34/prediction_cache/oof/
semantic_fold_{0,1,2}/<stem>.png`(2048×2048 uint16、236/236/235 枚)として
残っている**(2026-10-09 実読して確認)。したがって **D1(新規学習)を待たずに
E1 の機構検証を CPU だけで始められる**。V34 当時はこのマップを watershed で
個体化して Public 0.19 に終わったが、それは「閾値1本 + 分水嶺」の決定層の
失敗であり、マップ自体の候補源としての価値は未評価。

留意点: V34 の損失には clDice が入っており校正は保証されない(tush 氏は clDice
で −0.045)。v80 ではマップを「候補を生む装置」としてだけ使い、確率値そのものは
記述子の一部に留める。v80 で機構が成立したら、校正を壊さない損失で D1 を再学習
し(v86)、差し替える。

#### 7.1.2 候補集合の定義

画像 i(fold f)に対して、候補 c は次の2ソースの和集合:

| ソース | 生成法 | 想定件数/画像 |
|---|---|---|
| `yolo` | 既存 OOF cache `yolo_fold_f/<stem>.npz` の全提案(conf≥0.08、area≥20) | ≈10 |
| `tree` | V34 OOF マップ p を閾値 T={0.10,0.15,…,0.80}(15段階)で二値化 → 各段で連結成分(8連結)→ 入れ子で親子を張った max-tree。各ノードを候補とする。area<20 と、親と IoU>0.98 のノード(ほぼ同一)は除去 | ≈30〜80 |

実装は `solar_filament/pipeline/component_tree.py`(新規)に `build_tree(prob: np.ndarray, levels: Sequence[float], min_area: int) -> list[Node]`、
`Node = (mask_rle, level, parent_index, children, area)` を置く。
2048 native で構築し、メモリ節約のため mask は RLE(pycocotools)で保持する
(v64 の OOM 教訓: bool 全保持は ~100GB)。

#### 7.1.3 候補への refiner 適用

両ソースの候補に既存 fold 別標準 refiner(`mediseg_refiner_fold{f}_resume.pt`、
MediSegUNet 3ch、crop320/context1.65)を `run_refiner_pq_allfold.py` と同じ経路で
適用し、`shrink@0.45` 後のマスクを「最終候補マスク」とする。tree 候補も YOLO
候補と同じく「提案で切る」(shrink)— V61/V66 で確認済みの「提案外へ伸ばすと
他個体を侵食する」失敗を避ける。

コスト: 候補 ≈60/画像 × 707 画像 ≈ 42k クロップ。Colab T4 で約 1.5〜2h、
ローカル CPU 14 スレッドなら約 6〜8h(`run_refiner_pq_allfold.py` の 87min@conf0.26
から外挿)。結果は `solar_filament_v80/candidates/fold_{f}/<stem>.npz`
(RLE・ソース・level・親・conf)に 1 回だけキャッシュし、以降は CPU のみ。

#### 7.1.4 教師(目標変数)

候補 c、画像 i のアノテータ record 集合 R_i(|R_i| = R):

```
u(c) = (1/R) Σ_{r∈R_i}  max_g IoU(c, g_r) · 1[max_g IoU(c, g_r) > 0.5]
h(c) = (1/R) Σ_r 1[max_g IoU(c, g_r) > 0.5]          # 当たる確率
m(c) = u(c) / h(c)  (h>0 のとき)                       # 当たったときの IoU
```

公式 PQ の実装(`metrics_official.PanopticQuality`)と同じ IoU 行列
(`mask_util.iou`)で計算する。注意: 公式実装は一対一割当てではなく閾値越えを
全て数えるため、「max_g」で近似する(同一候補が同一 record の2 GT と IoU>0.5
になるケースは細長い構造では稀。件数を manifest に記録する)。

#### 7.1.5 記述子(特徴量)

全候補に共通(ソースに依らず計算できるもの)+ ソース固有(欠損は −1)。

| 群 | 記述子 |
|---|---|
| 幾何 | area(log)、bbox 長辺/短辺、minAreaRect の伸長度、solidity、周長/面積、重心の円盤中心からの距離 ρ(半径 899.7px で正規化)、limb 近接フラグ(ρ>0.9) |
| 画像 | 候補内の前処理画像 平均/5%分位 輝度、候補境界 5px 外側との輝度差(暗さコントラスト) |
| 密マップ p | 候補内 p の mean/median/max、p>0.5 画素比率、候補を 5px 膨張した環状帯の p mean(境界の鋭さ) |
| refiner | refiner 確率の mean/max/>0.45 比率、shrink 比(refined 面積/候補面積)、refined mask の連結成分数 |
| YOLO 固有 | conf(tree 候補は −1)、同一画像内で IoU>0.5 の YOLO 候補の最大 conf(tree 候補用、「YOLO も見ているか」) |
| tree 固有 | level、木の深さ、子の数、親との IoU、兄弟の数(YOLO 候補は −1) |
| 文脈 | 画像内候補総数、画像内 YOLO 候補数、他候補との最大 IoU、重なる候補数 |
| ソース | one-hot(yolo/tree) |

v64 の教訓(静的特徴 9 個が無情報)を踏まえ、**p 統計・refiner 統計・ソース間の
相互参照**を主役にする。too.foo 氏が有効と報告した暗さコントラスト・細長さ・
limb 近接度も含める。

#### 7.1.6 モデルと cross-fit

- LightGBM(`lightgbm` を requirements に追加、無ければ sklearn HistGradientBoosting)
  hurdle: 分類器 ĥ(目的 `binary`、重み R)、回帰器 m̂(TP 候補のみ、目的 `regression`、重み R)。
  û = ĥ·m̂。
- **cross-fit は既存 3-fold と一致させる**: fold f の候補を採点するモデルは fold ≠ f の
  候補だけで学習。ハイパラは固定(num_leaves 31、lr 0.05、500 木、early stop なし)で
  探索しない(選択バイアス回避)。
- 対照 arm として「YOLO 候補のみ + 同じスコアラー」(= A1 単独、v64 相当)を必ず置く。

#### 7.1.7 選択(決定層)

1. û > τ の候補を残す。τ は 0.16〜0.32 を 0.02 刻みで掃引し、**PQ/2 との自己無撞着点**
   (τ ≈ 達成 PQ/2)を主結果とする。掃引で最良の τ を選ぶ場合は fold 跨ぎで
   同一 τ を使い、fold 別最良は報告のみ。
2. tree 候補は **Σ(û−τ) を最大化する antichain** を木 DP で厳密に選ぶ
   (`select_antichain_forest`。根 0.6 vs 子 0.5×2 のように「各パス最大を貪欲に
   取る」方式は最適でないことに注意)。YOLO 候補と tree 候補の重なりは既存
   `drop_contained`(0.90)に委ねる。
3. 既存 `apply_ownership(rule="drop_contained_then_area_asc", containment_threshold=0.90)`。
4. 公式 PQ で採点。

**実装状況 (2026-10-09)**: `component_tree.py`(max-tree 構築 + Σ(û−τ) 厳密
antichain DP)+ テスト 15 件、`run_v80_hybrid_candidates.py`(候補生成・refiner
適用・RLE キャッシュ、CPU/Colab)、`run_v80_scorer_eval.py`(教師・cross-fit
HistGBM hurdle・5 arm 掃引・観測 bootstrap)を実装済み。残りは Drive/Colab での
候補生成実行と eval のみ。

arm 一覧(すべて同一画像集合・同一 refiner・paired):

| arm | 候補 | 選択 |
|---|---|---|
| `prod` | yolo conf≥0.26 | なし(現行) |
| `yolo_scorer` | yolo conf≥0.08 | û>τ |
| `tree_only` | tree | û>τ + antichain |
| `hybrid` | yolo∪tree | û>τ + antichain |
| `hybrid_oracle` | yolo∪tree | u>τ(真値、上界の参考) |

#### 7.1.8 判定・成果物・工数

- 採用ゲート: `hybrid` − `prod` が 3fold すべて正 かつ pooled bootstrap 95%CI がゼロを除外。
  `hybrid_oracle` が +0.02 未満なら候補集合の上界が低いと判断し、D1(校正マップ)
  へ進まず E1 を閉じる。
- 成果物: `scratch/run_v80_hybrid_candidates.py`(候補生成・refiner 適用・キャッシュ、
  `--fold`)、`scratch/run_v80_scorer_eval.py`(教師・記述子・cross-fit・選択・採点)、
  `solar_filament/pipeline/component_tree.py` + `tests/test_component_tree.py`
  (入れ子保証・antichain 選択の総当たり検証 1,000 木)、`output/v80/` に manifest。
- 工数: 実装 1.5 日、refiner 適用 2h(Colab)/8h(CPU)、評価 1h。**W1 内に 3fold 結果**。

#### 7.1.8b 【2026-10-10 実測】全fold 結果 — 不採用、ただし oracle 余力 +0.107 を確認

候補生成 707/707 完走(Colab fold0 GPU 84分、残りも Colab/ローカルで完了)。
評価 89分、707観測・149,693候補・u>0率 36.7%、τ*(PQ/2 固定点)=0.22:

| arm | PQ | Δ vs prod | fold0 | fold1 | fold2 |
|---|---|---|---|---|---|
| prod | 0.4426 | — | 0.4393 | 0.4415 | 0.4471 |
| yolo_scorer | 0.4465 | **+0.0039(3fold すべて正)** | 0.4401 | 0.4484 | 0.4509 |
| tree_only | 0.4243 | −0.0184 | | | |
| hybrid | 0.4284 | **−0.0142** | 0.4232 | 0.4399 | 0.4225 |
| hybrid_oracle | 0.5493 | **+0.1067** | 0.5422 | 0.5538 | 0.5521 |

- **採用ゲート: 不合格** — hybrid−prod CI95 [−0.0194, −0.0089](有意に負)。
  prod の fold 別値は README ベースラインと完全一致(評価配線の正しさ確認)。
- **oracle 余力 +0.107 は +0.02 閾値を大きく上回る → 候補集合の価値は実証済み**
  (oracle: TP +1454 かつ FP −1063 = 良い tree ノードが悪い yolo 予測を置き換え)。
- 分解: hybrid は TP +106・FN −106(tree 候補で新規検出)だが FP +537、SQ/RQ 低下。
  つまり追加検出は成功、FP 抑制が失敗。
- スコアラー診断(180観測、cross-fit 同一条件): ROC-AUC(h>0) tree **0.900** /
  yolo 0.805、Spearman ρ(û,u) tree **0.676** / yolo 0.591、u≥0.3 の平均
  percentile rank tree 0.775 — **û の順位付けは良好**。失敗は「ランク」では
  なく **û の絶対校正**: 中位候補の û が過大評価され、antichain DP が
  Σ(û−τ) で子ノード群を累積選択 → 断片化・FP 爆発。
- 次の一手(同じ候補キャッシュ再利用、eval 再実行のみ):
  **v81 = ソース別 τ**(tree 側 τ を yolo より高く)。tree_util 側に独立
  τ 掃引する `hybrid_split` arm を eval に追加して実施中。

#### 7.1.8c 【2026-10-10 実測・追記】v81 hybrid_split 掃引 — 2つの確定

`hybrid_split`(yolo τ=0.22 固定、tree τ 0.20–0.68 独立掃引)を追加して
全fold再評価(評価 110分)。2つの重要な結果:

1. **`yolo_scorer`(=A1)が採用ゲートを通過**: Δ+0.0039、3fold 全て正
   (+0.0008/+0.0069/+0.0038)、pooled bootstrap CI95 **[+0.00057, +0.00735]
   でゼロを除外**。v64(+0.0000)を post-refiner 期待IoU 特徴で上回る
   小幅だが統計的に確かな改善。**提出構成に yolo_scorer(û>0.22)を採用候補**
   として登録(提出版では conf≥0.08 候補にも fold別 refiner + scorer を
   適用する経路)。
2. **tree 候補はどんな τ でもネット正にならない**: hybrid_split の PQ は
   tree τ に対し単調増加(0.4270→0.4458)で端点に収束 — 最適は「tree を
   ほぼ選ばない」yolo_scorer 漸近線。τ=0.68 でも fold0 は −0.0005 で
   符号一致せず。**û の絶対校正調整では tree 解放は不可能**と判定。
   oracle +0.107 の解放には別機構(v86 校正マップ再学習 / 強い記述子 /
   ranking 損失 / FP ペナルティ付き選択)が必要。

#### 7.1.9 v86 — 校正済み密マップの再学習(v80 が成立した場合のみ)

V34 レシピから clDice を外し BCE+0.5Dice のみ、EMA、D4 TTA、教師は soft consensus
のまま。3fold、A100 で各 1.5h。v80 の `tree` ソースを差し替えて再評価。

### 7.2 v82 — refiner の soft annotator-mean 目標(C2)

#### 7.2.1 現状と仮説

`InstanceCropDataset`(`build_v51_edgeattnet_refiner.py` 等)は **(file, image_id=record, ann)
の組を 1 サンプル**とし、教師はその record の硬いマスク。同じフィラメントを 3 人が
描いていれば 3 サンプル(中心・輪郭が少しずつ違う)として独立に学習され、モデルは
暗黙に平均へ寄る。仮説: **明示的に soft target(同一個体を描いた record 群の平均)を
与えると、各アノテータとの IoU が d/√2 の算術(0.61→0.71)どおり上がり、IoU 0.5
断崖を跨ぐ near-miss(fold2 で 333 件)の一部が TP 化する。**

負の事前分布: too.foo 氏は crop-IoU 改善が PQ に移らなかったと報告(ただし
target の変更ではなく refiner 実装の改善)。v59/v61 も refiner 系は正味ゼロ。
効果が無ければ「refiner の教師定義は PQ に無関係」という負の結果として記録する。

#### 7.2.2 soft target の構築

1. 画像 file ごとに全 record の GT インスタンスを集め、IoU>0.3 の連結で
   クラスタ化(union-find)。1 クラスタ = 1 物理フィラメント候補。
2. クラスタ k の soft mask: `S_k = (1/R) Σ_{r∈R_i} M_{k,r}`(record r がそのクラスタに
   インスタンスを持たなければ 0)。値域 [0,1]。**R は画像のアノテータ数**(描いて
   いない人も分母に入れる = 「ランダムなアノテータが塗る確率」)。
3. 学習サンプルは**クラスタ単位**(record 単位ではない): クロップ中心は S_k>0 の
   bbox、教師は S_k のクロップ。サンプル数は現行(record 単位 ≈8,199)より減る
   (≈5,500 見込み)ため、epoch 数は現行 14 のまま、1 epoch 内の反復回数を
   合わせるために `repeats=1.5` を入れる。
4. 対照 arm `hard_record`(現行)と `hard_union`(クラスタの union を 1/0 教師に)も
   同時に学習し、「soft であること」と「クラスタ化すること」の効果を分離する。

#### 7.2.3 損失

`refiner_loss` の構造は維持し、BCE は soft label 対応(`binary_cross_entropy_with_logits`
はそのまま soft target を受け付ける)、Dice は soft Dice(予測確率と soft target の積)。
boundary 項は `boundary_map(target)` を soft target に対して計算(勾配 = 分布の縁)。

#### 7.2.4 推論・評価

推論経路は変更なし(shrink@0.45)。ただし soft target では確率の最適閾値が
`t*=J*/(1+J*)` に寄る可能性があるため、閾値 0.35/0.40/0.45/0.50 を**事前宣言した
補助掃引**として記録する(主結果は 0.45 固定)。
評価は v54/v59 と同じ paired 経路(fold 別 OOF、同一 YOLO 候補集合、bootstrap CI)。
arm: `hard_record`(現行)/`hard_union`/`soft` × seed(3407, 8407) × fold(0,1,2) = 18 run、
L4 で各 1.5h → 3 セッション並列で 1.5 日。

#### 7.2.5 判定

- 採用: `soft` − `hard_record` が 3fold・両 seed で正、pooled CI がゼロを除外。
- 中間解釈: `hard_union` が `soft` と同等なら「クラスタ化が効いた」、`soft` だけが
  正なら「soft であること」が効いた。どちらも負なら refiner 教師定義レーンを閉じる。
- 成果物: `scratch/build_v82_soft_target_refiner.py --fold {0,1,2} --arm {soft,hard_union}
  --seed {3407,8407}`(Colab ipynb ビルダー)、
  `solar_filament/pipeline/soft_target.py`(クラスタ化・soft mask 生成)+
  `tests/test_soft_target.py`(3 record の合成例で S_k の値を固定)、`output/v82/`。

**実装状況 (2026-10-09)**: `soft_target.py`(union-find クラスタ化・同一 record 非併合・
`soft_target_crop` で遅延クロップ生成)+ テスト 6 件、`build_v82_soft_target_refiner.py`
(soft/hard_union 両 arm・閾値掃引 {0.35,…,0.50}・seed 引数つき、全セル構文検証済み)
を実装済み。`hard_record` arm は既存 `mediseg_refiner_fold{f}_resume.pt` が対照。
残りは Colab での 18 run 実行と採点のみ。

### 7.3 実装着手順(PR #244 マージ後)

1. `git fetch && git rebase origin/main`、§番号・README 表の重複解消、文書 PR を先に出す
2. v80: ~~`component_tree.py` + テスト → 候補生成スクリプト~~(実装済み)→ Colab で refiner 適用(fold 並列)→ CPU 評価
3. v82: ~~`soft_target.py` + テスト → builder~~(実装済み)→ Colab 3 セッション × 2 arm × 2 seed = 12 run(hard_record は既存 refiner が対照)
4. v80 の `hybrid_oracle` と `hybrid` の結果で、v86(校正マップ再学習)の起動可否を決める

### 7.4 A1(yolo_scorer)の test 提出パス設計【判定確定: 案(a)】

**判定結果 (2026-10-10、`output/v80_noprobdense/`)**: prob_* 4特徴を除いた
29特徴の `yolo_scorer` が **PQ 0.4460 (Δ +0.0034)、fold別 0.4416/0.4464/0.4501
(全fold prod 上回り)、bootstrap CI95 [+0.00045, +0.00642] でゼロ除外** —
採用ゲート両条件を維持。→ **案(a) で決定**。案(b)は不要になった。

#### 案(a) — prob_* なし scorer(29特徴)【採用】

- test 推論に新規モデルを追加しない: YOLO+refiner+sklearn のみ
- scorer 学習: OOF cross-fit は「評価の正直性」のためのもの。test 配置用は
  全707観測分(yolo+tree 全候補、評価時と同じ学習集合)で h_model/m_model を
  単一 HistGBM として再学習して保存(標準的な全データ再学習)
  → `scratch/train_a1_scorer.py` 実装済み、`artifacts_scorer/a1_scorer_{h,m}.pkl`
  生成済み(149,693行)+ Drive artifacts に配置済み
- 成果物: `a1_scorer_h.pkl` + `a1_scorer_m.pkl`(各 ~1.8MB)+
  `a1_scorer_meta.json`(特徴名・τ・選択ルール・sha256)を重み dataset に追加。
  sklearn は notebook が既に import 済み

#### 案(b) — V34 semantic を test 推論に追加【不採用 — 判定で不要と確定】

- 特徴一致だが推論モデルが3→4個に増え、`submission_efficiency.json` の
  s/image が悪化。案(a) がゲートを通過したため検討不要

#### test 推論時の既知の論点(案(a) で確定したもの)

| 論点 | 対処 |
|---|---|
| `n_annotators` は test で未知 | 学習時の中央値(≈2)で補完 — HistGBM にとって定数列は無害だが、OOF と同じ補完を再現用に明記 |
| `yolo_support_conf` は yolo 候補で常に NaN | OOF と同じく NaN のまま(HistGBM が欠損として処理) |
| conf 0.26→0.08 で refiner 負荷が増える | OOF では kept≈11.15/観測(conf≥0.08)。prod より増える分 efficiency に影響 — 増分を測定して記録 |
| 選択順序の完全一致 | `û>0.22` → `drop_contained@0.90` → 面積昇順 `greedy_disjoint`(min_area 20) — `_evaluate_image` と同順を cell 12 に移植 |
| τ=0.22 固定 | OOF で固定した値をそのまま使う。test での再調整は禁止(リーク) |

#### 組み込み先

`build_submission_reproduction.py` が master pipeline から派生生成する構成のため、
scorer 選択は **master pipeline notebook の推論セル側の変更**として扱い、
builder 側は `SUBMISSION_CONF=0.08` + scorer 重み解決パターン
(`*scorer*.pkl`)の追加に留める。生成 ipynb の直接編集はしない。

#### 公式採点の結果 (2026-10-10) — A1 は LB で不採用

`scratch/run_test_submission.py` で prod/a1 両 CSV をローカル生成・提出:

| 提出 | 構成 | Public LB | 備考 |
|---|---|---|---|
| prod (57039986) | V52 と同一 (fold2 YOLO + EdgeAtt all-data) | **0.39** | V52 実CSVとの照合: IoU>0.5 一致 1171/1171。公式レベルで再現確定 → 以後のローカルベースライン |
| A1 (57047601) | conf0.08 候補→scorer û>0.22→refiner/ownership | **0.37** | 1174行・prod と 91.4% 一致。OOF +0.0034 が −0.02 に反転 |

**教訓**: 0.003 規模の OOF 改善は ~90枚・2桁表示の Public LB では識別不能
(サンプリングノイズで容易に反転)。scorer の FP 削減は test では TP も削った
可能性。以後の採否は **OOF で明確に大きい効果 (>0.01) または機構的に堅い
変更**に限定する。日次枠 = 5提出/日 (API 応答確認)、ファイル提出制で計算枠
消費ゼロ。RESEARCH_LOG §2.10 に全提出の OOF↔LB 対応表を記録。

### 2026-10-11 追記: refiner バリアント提出 — 3構成とも 0.39

| 提出 | 構成 | Public LB | 備考 |
|---|---|---|---|
| plain_mhsa_f2 (57072525) | fold2 YOLO + plain-MHSA refiner (edge prior なし・fold2 s8407) | **0.39** | prod と境界構成が異なる (mean IoU 0.938)。ablation の「edge prior 固有効果なし」と整合 |
| edgeatt_f2 (57073220) | fold2 YOLO + edge-att refiner fold2 のみ (s8407) | **0.39** | all-data fine-tune なしでも同点 |

**最終選択の推奨 (2枠)**: `prod` + `plain_mhsa_f2` — prod と edgeatt_f2 は
refiner 系統が同じ(学習データ差のみ)だが、plain_mhsa はアーキテクチャ差があり
private 側分散ヘッジとして最大。残る提出枠は新候補登場時に温存。

### 2026-10-11 追記2: 外部手法・discussion 調査の結論

全20 discussion スレッド(Kaggle API 本文取得)・上位 kernel 6本・
too.foo 公開 dossier を調査(RESEARCH_LOG §2.16 に詳細)。

- **0.55–0.63 上位パックは漏洩由来の可能性が高い** — 公開 Harvard Dataverse
  上の test GT を直接ブレンド/学習混入した場合の着地帯と一致すると
  複数参加者が解析。主催者も「>0.35 は全件精査・単一指標非依存」を明言
- **誠実系フロンティア ≈ 0.36–0.45** — too.foo LB 0.36 / Fususu ~0.40 /
  hetspookjee 0.39 と我々の 0.39 は先端帯。評価関数は 8/12 に全再採点
  済みで旧値(0.59/0.70 等)は比較不可
- **規約確認**: SWEFil 等外部 GT 禁止・無ラベル GONG 学習利用は可
  (test±1h 除外)・ラベル転写・磁気図補助信号は可・最終 repo は
  選択した両方の提出を再現できること(747960)
- **収束知見**: soft ラベル逆効果・crop refiner の PQ 非寄与・検出漏れ
  が真のボトルネック・val 転移限界 ~+0.02 — 全て我々の結論と独立に一致
- **取り込み**: 新規定量レバーなし(refiner-TTA のみ未検証だが効果微小
  見込み・低優先)。`prod` + `plain_mhsa_f2` の最終選択方針を維持
