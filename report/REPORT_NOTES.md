# 最終提出パッケージ — 作業メモ(2026-10-07)

締切: **2026-11-15**(Googleフォーム経由で提出)。
入賞発表 11-30 / IEEE BigData 12/14-17。

## 必須提出物(公式要件)

| 物 | 状態 | 作業 |
|---|---|---|
| 4ページ技術レポート(Overleafテンプレ必須) | **成熟原稿あり** `report/main.tex`(commit 140d8d62 由来、著者・所属・図・bib・謝辞定型文すべて完備)。2026-10-07 に Phase B の結果を統合済み | Overleaf の公式プロジェクト(read-only: `vjztpvrnbrdh`)を**自分のアカウントにコピー**してビルド確認。4p オーバーしていないか要計測(Phase B 追記で増量したため削る必要があるかも) |
| 公開Git repo | **作成済み(private)** `shingo257/solar-filament-2026-solution` (2026-10-19、抽出追従ブランチ `solar-solution-v4` = main@837eccc3、711 files) | 締切日に public 切替のみ。重み・データ・成果物は外部(Drive/Kaggle dataset)参照+SHA256 記録 |
| requirements.txt(全パッケージ+バージョン) | 土台あり | `Solar_Filament_Segmentation/requirements.txt` は `>=` 指定。提出前に**最終inference runの `pip freeze` で固定値に置き換え**必須 |
| 全パイプラインを示すJupyterノートブック | **作成済み** `solar_filament_submission_reproduction.ipynb`(2026-10-07) | master pipeline を派生。Kaggle/Colab 両対応(環境自動検出)、重みは添付dataset→Drive→再学習の順で解決、前処理キャッシュ無し環境ではinline計算、末尾に pip freeze+SHA256 記録。重みは公開 dataset `shingo257/solar-filament-2026-weights`(v2)にup済み — `VARIANT` で prod / plain_mhsa_f2 の**両最終提出**を再現可能。builder: `scratch/build_submission_reproduction.py` |

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

1. ~~**ページ数確認**~~ — **済**: MiKTeX/latexmk でローカルビルド。本文4p・
   謝辞+refsがp5に収まることを確認。Phase B 列挙は圧縮proseで収束
   (表化は不要だった)。**再検証 2026-10-19 (#400)**: v72/v73 SSL 負結果を
   追加しつつ本文が p5 へ流出していたため同節を圧縮し本文4pに復帰。
   今後本文を追加する場合は毎回ビルドして本文末尾の頁位置を確認する。
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
