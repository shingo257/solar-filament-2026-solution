# S0-4盲検レビュー運用ガイド

- package: `output/s0/blinded_review_v1/`
- status: `ready_for_independent_review_with_sampling_caveats`
- 対象: 24一意観測、3比較各8件
- 禁止事項: 回答JSON export前の`private/`閲覧、回答の学習labelへの自動転用

## 1. 回答

1. `output/s0/blinded_review_v1/review.html`だけを開く。
2. Reviewer IDを入力する。
3. 24件すべてで形態保存、カタログ品質、A/B error tag、確信度を回答する。
4. 「判定JSONを書き出す」でresponse JSONを保存する。
5. export完了までは`private/blind_key.json`を開かない。

回答はブラウザlocalStorageにも保存されるが、正本はexportしたJSONとする。response JSONには
Reviewer IDと自由記述が含まれるため、公開・commit前に取扱範囲を確認する。

## 2. 完全性検証とunblind集計

export後、次を実行する。

```powershell
py -3.12 scratch/analyze_s0_blinded_review.py `
  --responses <exportしたresponse JSONの絶対path>
```

このtoolは公開`selection_manifest.csv`だけで次を先に検証する。

- package / protocol version一致
- Reviewer IDが空でない
- 公開24 caseとresponse caseが完全一致し、重複・未知caseがない
- morphology / catalog / confidenceが全件回答済み
- error tagが事前定義集合内

1項目でも不完全なら停止し、秘密鍵を読み込まない。全条件通過後にのみ
`private/blind_key.json`を読み、次を`output/s0/blinded_review_v1/review_results/`へ出力する。

- `review_summary.json`: 比較別の形態・カタログ選好、error tag、確信度分布
- `unblinded_case_review.csv`: case単位の対応と回答

これらはreview evidenceであり、training labelではない。

## 3. 解釈上の制約

公開readiness監査では24 PNGすべてが`1650×1100×3`でデコードでき、hash不一致、near-blank、
公開HTMLのmodel identity漏えいはいずれも0だった。一方、sampleは仮説検証用に意図的に濃縮されている。

- 2017–2022 locked層: review `14/24 = 58.3%`、707観測全体 `160/707 = 22.6%`
- fold: fold0 / 1 / 2 = `4 / 11 / 9`
- site: 6 siteすべてを含むが、各`2〜6`件
- year: 2011〜2022の9年を含む

したがって、24件の選好率を母集団での改善率、独立holdout性能、site/time頑健性の推定値として
扱わない。比較別・fold別・time role別にcaseを示し、低確信例とreference不一致を併記する。

## 4. S0-4完了条件

- response JSONが24件完全で、Reviewer IDとexport時刻がある
- analyzerが拒否なしで完走する
- 低確信例と`unclear`例を別表にする
- model対応を見た後の回答変更を行わない
- review結果を`S1_ENDPOINT_REGISTRATION`の仮説選択には使ってよいが、marginの唯一の根拠にしない

完了後にだけS0-5の数値marginを固定する。新規GPU学習はその後とする。

margin固定後は`S1_DECISION_POLICY_REGISTRATION_TEMPLATE_2026-08-24.json`を埋め、
`scratch/register_s1_decision_policy.py --require-registered`でpublic response・endpoint・rationaleのhashと
target未生成を検証する。transactionがregistered spec candidateを生成するまではfull target label生成とmodel fitへ
進まない。canonical specを手編集してpolicyを埋めない。
