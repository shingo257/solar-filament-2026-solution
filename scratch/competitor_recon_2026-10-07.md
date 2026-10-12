# 競合・ディスカッション偵察レポート(2026-10-07)

Kaggle API + 公開ノートブック pull + ディスカッション全スレッド走査による調査。
目的: 上位陣の手法・ルール裁定・評価構造の実態把握。

## 1. リーダーボード地形(2026-10-07 時点、895チーム)

| 帯域 | チーム数 | 解釈 |
|---|---:|---|
| 0.60–0.63 | 7 | リーク+改造クラスタ(最上位) |
| 0.55–0.59 | ~65 | **リーククラスタ**(GT dump ≈0.55) |
| 0.50–0.55 | ~2 | ほぼ真空 |
| 0.40–0.50 | ~26 | 誠実優秀層 or 部分リーク(検証不能) |
| ≤0.40 | ~795 | 誠実ティア本体(中央値 0.34) |

- リークの実態: **公開 MAGFiLO v1.0 アノテーションがテスト画像をカバー**。
  test画像(2015–2017のファイル名はMAGFiLO収録観測と一致)のGTをそのまま
  submission に dump すると ~0.55(マルチアノテータGTとの完全一致ではないため
  1.0 にはならない)。0.55 超は GT+摂動/合成。
- 主催者(Azim Ahmadzadeh)の明言:
  - "**Any PQ score of greater than 0.35 is of great value to us**"(Aug 20 告知)
  - "**we take a score of 0.41 very seriously**"(743958スレッド)
  - "we can easily weed out the fake ones" + LBは最終的にクリーン化予定(740400)
  - "leaderboard is not important in the final score calculation"(参加者間で共有)
- 評価器崩壊の経緯(artkomissar "Please fix the Leaderboard" の実証):
  旧評価器は**180枚中5枚だけ提出し全て空マスクで満点1.00**。未提出画像への
  ペナルティなし+空マスクに正当性を与える設計ミス。全画像に1空マスク→0.88、
  7個ずつ→0.70。→ 8/7に公式PQへ置換、8/12に全LB再採点。

## 2. 評価構造の実態(最重要)

- **定量70%**: PQ(LB) + **Dice分布** + **IoU分布** +
  **one-to-many / many-to-one 対応の分布**(断片化・過剰マージの統計 —
  我々の v62 エラー解剖がこのまま審査対象の構造と一致)
- **定性30%**: パイプライン全体の詳細記述 + H-α上の予測形態 +
  **コード品質(モジュール性・ドキュメント)**
- LBは public(~50%のテスト) + **private(残り50%、非公開)** 両方あり
- 「the how you arrived at a solution is more important than the public
  leaderboard」(742143、運営側コメント)
- **締切: 2026-11-15**(最終提出) → 11-30 入賞発表 → 12/14-17 IEEE BigData
- 最終提出物: **4ページ技術レポート(Overleafテンプレ) + 公開Git repo +
  requirements.txt + 全パイプラインを示すJupyterノートブック**
  「The committee should not require any additional file to reproduce」
- 賞金総額$3,000を入賞3チームにスコア比例配分
- SABID workshop(IEEE BigData内)がレポート投稿を歓迎(9/20告知)

## 3. 主催者裁定 — 合法レバーの境界(新規情報)

| 要求 | 裁定 |
|---|---|
| train JSONの spine/bbox/area/category_id を補助監督に | **許可**(736543) |
| 学習mask由来の skeleton/距離変換/boundary aux | **許可**(同上) |
| 共時刻・近傍の**無ラベル** GONGフレームを**学習時**に使用 | **許可・推奨**(743385) |
| MAGFiLO期間外(2023–2026)の無ラベルGONGで半教師あり学習 | **許可**(744047) |
| 共時刻 GONG **磁気図(bzi)** を学習時のみの補助信号に | **許可・「good idea」**(743521) |
| 推論時の外部入力(近傍フレーム等) | **禁止** — 単一test画像のみ(743385) |
| 外部GT标注データ(SWEFil等) | **禁止** — 「more ground-truth dataは不可」(746318) |

→ **未開拓の合法レバーが2つ残っている**:
  1. 時間近傍フレーム増強(同一filamentが数日持続 = 天然の augmentation /
     temporal consistency semi-supervision)
  2. 磁気図補助信号(filament は分極反転線上に存在 → orphan FP の排除に
     直結する物理的判別情報)

## 4. 競合手法カタログ(Kaggle 公開カーネル実測)

| 手法 | 作者 | 構成 | スコア |
|---|---|---|---|
| **Seed-Refine Cascade** | timgluhkikh | YOLO11s detect → crop refiner `[image, CLAHE, seed]` 3ch + **Box-Jitter TTA** | local PQ ~0.409 |
| **MAGFiLO V9** | muhammadshoaibaltaf | yolo26l detect → crop UNet++ EffB4 + EdgeAttNet(edge注意) + OBB姉妹版、file_name分割、conf降順所有権 | 「evaluation-aligned」 |
| **物理チャネル型** | tushhaaarrrr | フラット場補正+局所コントラスト+**Lindeberg dark-ridge scale space** → Residual UNet on **annotator-mean soft mask** + spine aux head → CC + PQ-aware後処理(OOF調整) + FNバケット分析 | 0.4600(全annotator対象,全703枚ベース) |
| MaskRCNN ensemble | sivapnlr80 | MaskRCNN-R50FPN×5fold + **4-TTA(hflip/vflip/rot180)** + cross-fold consensus(min_agree=3) + crop refiner | LB 0.34(v42系) |
| MaskRCNN+UNet | nomannic19 | MaskRCNN detect → UNet refiner(2段、我々と同型) | — |
| YOLO+UNet | ektarr(Maxim) | YOLO detect → UNet | — (リークに嫌気で離脱気味) |
| UNet baseline | rokaiyasomapti他 | smp.Unet ResNet34、random 80/20 split | 旧metric時代の「LB 1st」 |
| Spaghetto99 | — | backbone不明、Focal+Dice、val 0.4843 自称だが**split非grouped疑惑**(106/707、tushhaaarrrrが分割法を質問) | LB ~0.40 |
| too.foo | — | UNet++ EffB3 + filament-biased patch + leakage-free split + component selector(+0.018実取) | val 0.416(8/9時点) |
| HDJoJo | — | yolov8l-seg imgsz2048(公開nbは推論のみ) | LB 0.55(87回提出、リーク疑惑帯) |
| lamhuy8904 | — | UNet系+disjoint割当「0.55+」— **リーク実例として特定済**(Hermengardo指摘) | LB 0.55 |

**外部較正の最重要数値**(tushhaaarrrr):
- **human-vs-human PQ ≈ 0.34** — 1人のannotatorを他annotatorで評価した場合。
  これが「このデータでの実質的性能の物差し」。我々の OOF 0.443 は
  単一annotator一貫性を上回る = annotator間の共通構造を捉えている。

## 5. 我々の位置づけ

- OOF 0.4432(grouped 3-fold,全annotator基準)は主催者が「真剣に受け止める」
  0.41 超 + human-vs-human 0.34 超 → **誠実ティアでは最前線、おそらく単独首位級**
- 公開誠実値の確認範囲: too.foo 0.416、Spaghetto99 ~0.40(自称0.48は分割疑惑)、
  tushhaaarrrr 0.46(全annotator×全枚数ベースで比較基準がやや異なる)
- LB上位72チーム(0.55+)はリーク由来で、主催者が手動で除去予定

## 6. 行動への含意

1. **定量はこのまま死守** — v65/v66 の結果待ち。両者棄却でも 0.443 は十分に強い
2. **新レバー検討の価値あり**(残り~5週間):
   - 近傍無ラベルフレームの学習時活用(temporal augmentation / 半教師あり)
   - 磁気図補助信号(orphan FP への物理的判別 — v64 で失敗した静的特徴の
     代替になり得る「別種の情報源」)
   - ただし両方とも外部データ取得パイプラインが必要で、コスト見積が先決
3. **審査基準に合わせた報告書設計**:
   - 定量70% = PQ + Dice/IoU分布 + 断片/マージ統計 → v62 anatomy の
     クラスタ分解(blind/near/orphan/misassigned/overmerge)はそのまま審査項目
   - 定性30% = pipeline記述 + 形態 + コード品質 → 既存の設計判断記録・
     失敗実験の記録は直接の加点材料
4. **提出物の準備(11/15締切)**: 4p report + 公開repo + requirements.txt +
   全パイプライン再現notebook — repo の公開可能化(重み外部参照化)が必要
5. **TTA方向の外部裏付け**: timgluhkikh の Box-Jitter TTA、sivapnlr80 の
   4-TTA+cross-fold consensus が「変換安定性」系を支持 → v65 の方向は正しい
