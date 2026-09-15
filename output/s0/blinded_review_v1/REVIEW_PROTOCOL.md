# S0-4 blinded visual review protocol

1. `review.html`だけを開く。回答JSONを書き出すまで`private/`を開かない。
2. 各caseでfull diskとzoomを確認する。cyan / magenta / yellowはannotator variantの輪郭とspine。
3. 形態保存、カタログ上の有害error、A/Bそれぞれのerror tag、確信度を独立に回答する。
4. modelを推測して回答しない。reference間不一致が大きければ「判定困難」を付ける。
5. 24件すべて完了後、Reviewer IDを入れて「判定JSONを書き出す」を押す。
6. JSONを保存してからのみ`private/blind_key.json`でunblindする。

ブラウザ内回答はlocalStorageへ自動保存される。学習labelへ自動転用しない。
