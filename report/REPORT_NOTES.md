# 最終提出パッケージ — 作業メモ(2026-10-07)

## 2026-11-15 最終更新 (Public LB 0.40 到達・外部査読反映・最終確定)

- **3 検出器 1対1 二部合意 + 3票ピクセル多数決 (`cons2of3_sptal_maj3`, ref `58363293`) で Public LB `0.40` 突破**:
  `YOLO11m-seg` + `YOLO26m-seg` + `YOLO11m-seg (Spine-Aux 0.5 + Small-TAL 4.0)` の全データ 70ep 3 検出器
  および `refiner_all_data.pt` (`MediSegEdgeAttUNet` 全データ微調整・固定) により Fold-2 held-out `0.4580 (+0.0138, FP -15.7%)`・Public LB **`0.40`** を達成。
  第2選択アンカーは `prod_v96full` (ref `57897560`, Public LB `0.39`)。
- **独立査読指摘 5 件 + 統計表現すべて修正・厳密 4p (`Overfull \hbox` ゼロ) 確認済み**:
  1. Fig. 3(a) (`conf 0.02`) と §3.3 本文 (`conf 0.26` で 65% = 691/1,064、`[0.02, 0.26)` で 57% = 543/953 回復・43% 未回復) の閾値整合を明記。Hard misses (`2,138 FN / 1,561 FP`) の母集団 (`707` 枚 plain-refiner OOF, `IoU<=0.3`) も明記。
  2. EdgeAttNet / bottleneck attention の主張を matched-param 2-seed ablation (`0.4418` vs `0.4432` / `0.4429`, fold0 微減) に合わせ「edge prior の効果は検出されず、attention 自体の差 (~0.001) も決定的でない」に修正。タイトルも 3 段階 + 多検出器合意を表す題名へ更新。
  3. LB `0.37 → 0.39` の帰属交絡を解消 (`0.37` は初期 fold0 検出器、fold2 単一検出器は plain-MHSA / edge-MHSA いずれでも `0.39`)。
  4. Table 1 を `707` 枚 3-fold CV セクションと後半の subset held-out (`n=471` / `n=235`) セクションに明確分離。
  5. 最終提出の両方が `refiner_all_data.pt` (`MediSegEdgeAttUNet`) を固定使用している旨、Spine-Aux が fold2 単一 fold の探索的検証である旨、LB 2 桁丸め、77% MAGFiLO 重複 (`Sp-77%` 導入理由)、GPU 計測条件差 (A100 vs T4) を明記。
- **v140 (2-Fold 471観測 H-Flip TTA 検証)**: 70ep 完走モデル同士の Fold-2 で `58363293` (`0.45803`) が全反転 TTA 候補を上回ったため `HOLD` (`58363293` + `57897560` で最終ロック)。

## 2026-11-09 更新

- **Fig.3 (error_anatomy.png) 追加**: fold-2 OOF prod 実測の残存誤差2例
  ((a) proposal-blind miss — conf0.02 でも候補無し、(b) near-miss IoU0.404)。
  生成スクリプトは ER-FlowScan `scratch/make_qualitative_figures.py` (PR #645)、
  原図は Drive `artifacts/qualitative/`。本文圧縮(weaknesses 列挙の圧縮・
  fig:example 0.42lw)で **4p 維持を再確認**(latexmk ローカルビルド)。

締切: **2026-11-15**(Googleフォーム経由で提出)。
入賞発表 11-30 / IEEE BigData 12/14-17。

## 必須提出物(公式要件)

| 物 | 状態 | 作業 |
|---|---|---|
| 4ページ技術レポート(Overleafテンプレ必須) | **完了 (厳密4p)** `report/main.tex` / `report/main.pdf` (著者 `Shingo Hara` / `ER SYSTEMS,Ltd.`・図1–4・Table 1–2・bib・謝辞定型文すべて完備、全フォント埋込・Overfull 0 検証済み) | ローカル `pdflatex` + `bibtex` (`acmart` 2.20) で厳密 4 ページ確認済み |
| 公開Git repo | **公開済み (PUBLIC)** `https://github.com/shingo257/solar-filament-2026-solution` | Git LFS で全 8 重み (~537 MiB) 同梱・`weights_manifest.json` SHA256 検証済み |
| requirements.txt(全パッケージ+バージョン) | **完了 (`==` 固定済み)** | `requirements.txt` を最終再現環境の確定バージョン (`==`) で固定済み (完全な `pip freeze` もリポジトリ内同梱) |
| 全パイプラインを示すJupyterノートブック | **完了・クリーン再現検証済み** `solar_filament_submission_reproduction.ipynb` | `SOLAR_SUBMISSION_VARIANT=cons2of3_sptal_maj3` (第1選択 LB 0.40, ref `58363293`; A100 クリーン再現 1,135/1,136 = 99.91%, mean IoU 0.9998) および `prod_v96full` (第2選択 LB 0.39, ref `57897560`) の両最終提出に対応 |

## main.tex に既に含まれるもの(2026-10-07 時点)

- 著者情報: Shingo Hara / ER Systems, Ltd. / shingo257@gmail.com(記入済)
- tikz パイプライン図 + `figures/`(qualitative_comparison.png 等)
- 公式PQの差異記述(全IoU>0.5ペア=TP、Hungarian ではないことの明示)
- **inter-annotator PQ 実測値 0.3398**(n=296 multi-annotator)済み
- OOF 結果表(plain 0.4426 / edge-attn 0.4454 single-seed + matched-param
  ablation 3fold×2seed)+ public LB(0.37/0.39/ensemble)
- EdgeAttNet の正直な撤回記述(single-seed受理→ablationでmechanism帰属誤り)
- 5件の棄却記録 + **Phase B 追加**: v54-v66 の11機構棄却のまとめ段落、
  fold2 エラー解剖(68% blind FN等)、oracle天井+0.046、残存合法レバー
  (時間近傍フレーム・磁気図)を Future work へ
- OOF↔LB乖離の考察(年/サイト分布のχ²検定で差なし)

## report 残タスク(2026-10-12 更新)

外部スコア・実行時間・重み公開状態は作者の既存記録であり、今回の独立受入ではない。
`HANDOVER.md` 冒頭の manifest 未確認・489/707件の分割相違は未解決。
A1 の OOF/CI は上流モデルまで nested 化していない探索結果で、採用ゲートの証拠には使わない。
両バリアントの現コード世代のクリーン実行と来歴固定は残タスク。

1. ~~**ページ数確認**~~ — **済**: MiKTeX/latexmk でローカルビルド。**2026-11-01
   最終状態: PDF全体でちょうど4p** (本文・謝辞・refsすべて込み、
   devin/solar-error-decomp `1d79b4c5`)。§2.60–2.63 追記で一時6pまで
   増えたが、機構列挙のfamily化・ablation記述・図キャプション・
   bib note の圧縮で4p全体に収斂。Codex レビュー修正値
   (neg-value 24%・oracle上限+0.047) も反映済み。
   今後本文を追加する場合は毎回ビルドして総頁数を確認する。
2. **Dice/IoU分布図** — ルーブリック定量項目だが4p予算いっぱいのため不採用
   (定性比較図Fig.2を優先)
3. ~~**計算効率の数値**~~ — **済**: T4 ~1.1 s/img end-to-end を §3.2 に記載
4. ~~main.pdf の再ビルド確認~~ — **済**: 最新ソースで同梱
5. 外部survey文脈・最終選択の反映 — prod+plain_mhsa_f2 の選択根拠を記録。
   他者スコア帯だけから漏洩・誠実性を分類しない。公開報告との類似は独立再現とは区別する。

## 新レバー見積結果(A案、2026-10-07 実測)

- **時間近傍フレーム**: **取得パイプライン実装・実測済み**。
  `scratch/fetch_temporal_neighbors.py` — file_name(YYYYMMDDHHMMSS+サイト)
  から観測時刻を復元し `haf`(FITS)の日次dirを列挙、±N分の同サイトフレームを
  時間差昇順で選定。実測: 20150421140454Bh に ±1-4分の近傍4件ヒット、
  1枚 2.9MB/2.6秒でDL成功(FITSヘッダ正常)。全707観測×2枚なら ~4GB・~1時間。
  ※ hag(JPEG)は装飾焼き込みで不可のため haf(FITS)+`convert_gong_fits_to_jpeg.py`
    で階調変換する既存パイプラインを使う。ラベルは同一 annotation を流用
    (±数分では filament 形状はほぼ不変)
- **磁気図**: **実現可能・実測済み(2026-10-08)**。
  `https://gong.nso.edu/data/magmap/QR/bqr/{YYYYMM}/{site2}bqr{YYMMDD}/`
  にサイト別 LOS 磁気図 FITS(**180×180 float32**, ~10分cadence, ~150KB)
  が存在。サイトコード対応は H-α1文字→2文字(B→bb, C→ct, L→le, M→ml,
  T→td, U→ud)。実測: 観測 20150421140454Bh に `bbbqr150421t1404.fits`
  (同じ分)がヒット。ただし日次カバレッジは疎(その日19枚/理論144枚) —
  全707観測のカバレッジ率計測が次ステップ。
  用途: PIL(分極反転線)への距離特徴→「3viewでも安定に出るFP」の物理的判別子
- **v68 磁気図PIL補助selector**: **実施済み・棄却**(2026-10-10、
  `scratch/v68_magnetogram_probe.py` ローカル実行)。カバレッジ671/707(95%)、
  実信号確認(hit→PIL中央値63px vs rand 81px)だが trainB最良=selector不使用で
  ΔPQ +0.0000 → noise-reject。**これで全定量レバー15機構全滅確定**
- **v67 時間近傍増強**: **実施済み・棄却**(fold2ペア採点 ΔPQ −0.0087)。
  ±20分・同annotation流用でデータ~3倍にしたが RQ 低下(TP−56)。
  ラベル陳腐化+FITS→JPEG階調近似のドメイン差がノイズとして作用

## repo公開化チェックリスト

- [ ] Solar_Filament_Segmentation/ 以下を standalone repo 化 or
      モノレポの該当dirを公開に向けて整理
- [ ] *.pt/*.pkl/生成物の外部参照化 + SHA256 manifest(既存方針通り)
- [ ] scratch/ の実験ビルダー群は残してよい(負の結果の再現性の証拠)が、
      README で「提出パイプライン vs 実験」を区別する説明を追加
- [x] 再現notebook: `solar_filament_submission_reproduction.ipynb`
      (確定構成: YOLO11m-seg@1280 conf0.26 → refiner crop320/ctx1.65 → ownership。
      Kaggle/Colab両対応・添付重み解決・環境固定記録つき)
- [x] 重みの公開Kaggle dataset化: `shingo257/solar-filament-2026-weights`(v2)
      — yolo_fold{0,1,2}_best.pt + refiner_all_data.pt +
      ablation_plain_mhsa_seed8407_fold2.pt + A1 scorer一式、
      `weights_manifest.json` に SHA256 記録済み
