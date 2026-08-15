import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_BUILDER = Path(__file__).with_name("build_v30_prize_colab_final.py")
OUTPUT = ROOT / "solar_filament_v30_prize_colab_instance_ensemble.ipynb"

runpy.run_path(str(PREVIOUS_BUILDER), run_name="__main__")
notebook = json.loads(OUTPUT.read_text(encoding="utf-8"))


def replace(cell_index, old, new, count=1):
    source = notebook["cells"][cell_index]["source"]
    if source.count(old) < count:
        raise RuntimeError(f"セル{cell_index}: 置換対象が不足しています: {old!r}")
    notebook["cells"][cell_index]["source"] = source.replace(old, new, count)


# システムRAM容量を自動判定する。通常RAMでは速度より安定性を優先する。
replace(
    2,
    '''    DEFAULT_MIN_AREA = 35''',
    '''    DEFAULT_MIN_AREA = 35

    # 30GB未満なら通常RAMと判定し、worker数とファイルキャッシュを抑える。
    SYSTEM_RAM_GB = int(Path("/proc/meminfo").read_text().splitlines()[0].split()[1]) / 1024**2
    HIGH_RAM = SYSTEM_RAM_GB >= 30
    DATA_WORKERS = 4 if HIGH_RAM else 2
    YOLO_CACHE = "disk" if HIGH_RAM else False''',
)
replace(
    2,
    '"drive_root": str(CFG.DRIVE_ROOT),',
    '"drive_root": str(CFG.DRIVE_ROOT),\n    "system_ram_gb": round(CFG.SYSTEM_RAM_GB, 1),\n    "high_ram_mode": CFG.HIGH_RAM,',
)
replace(6, "workers=4,", "workers=CFG.DATA_WORKERS,")
replace(6, 'cache="disk",', "cache=CFG.YOLO_CACHE,")
replace(8, "num_workers=2, pin_memory=True)", "num_workers=CFG.DATA_WORKERS, pin_memory=True)")
replace(
    9,
    "num_workers=4, pin_memory=True, persistent_workers=True)",
    "num_workers=CFG.DATA_WORKERS, pin_memory=True, persistent_workers=(CFG.DATA_WORKERS > 0))",
    count=2,
)

notebook["cells"][0]["source"] = """# 太陽フィラメントセグメンテーション V30 — 入賞志向の個体アンサンブル（Colab A100）

V27のスコアが **0.19** に留まった主因は、全フィラメントを結合したsemantic maskを学習し、
連結成分で分割していたことです。コンペは**個々のフィラメント**を評価し、分断や結合を
減点します。このノートブックでは、単なる閾値変更ではなく学習対象と検証方法を作り直します。

処理構成：

1. 同じ観測画像が学習・検証を跨がないgroup分割
2. 高解像度YOLO11m-segを3-foldで学習
3. MAGFiLOの複数アノテータ注釈を捨てず、独立した教師データとして利用
4. リポジトリの31Mパラメータ`MediSegUNet`で各検出個体を精密化
5. アノテータ平均のHungarian matching個体Diceで信頼度・mask閾値・最小面積を調整
6. 3-foldをアンサンブルし、公式compressed COCO RLEを出力

**必要なColab設定:** A100 GPUを推奨。High-RAMは現行高速設定では推奨ですが必須ではありません。
システムRAMが30GB未満の場合は、worker数とキャッシュを自動的に抑える省メモリモードへ切り替わります。
Internetを有効にし、Colab Secretsへ`KAGGLE_USERNAME`と`KAGGLE_KEY`を登録してください。
完了したfoldとrefiner epochはGoogle Driveから再開できます。

参考：[コンペ概要](https://www.kaggle.com/competitions/filament-segmentation-2026/overview)、
[MAGFiLO](https://www.mlecofi.net/magfilo)、[EdgeAttNet](https://arxiv.org/abs/2509.02964)
"""

translations = {
    "# [1/12] Install runtime dependencies": "# [1/12] 実行環境の依存パッケージをインストール",
    "# [2/12] Imports, Drive, and configuration": "# [2/12] ライブラリ、Google Drive、設定",
    "# [3/12] Download once to Drive, then stage data on the local SSD": "# [3/12] データをDriveへ一度だけ保存し、ColabのSSDへ配置",
    "# [4/12] Solar preprocessing and leakage-safe folds": "# [4/12] 太陽画像の前処理と、データ漏洩を防ぐfold分割",
    "    # Inspired by EdgeAttNet: suppress the radial/background gradient and expose\n    # faint dark filaments without throwing away the original observation.": "    # EdgeAttNetを参考に、元画像を保持しながら半径方向の背景勾配を抑え、\n    # 淡く暗いフィラメントを強調する。",
    "# [5/12] Build all-labeler YOLO instance datasets": "# [5/12] 全アノテータを利用したYOLOインスタンスデータセットを作成",
    "    # MAGFiLO specifies one polygon per filament. Largest is a safe fallback for\n    # malformed/multipart records without turning one filament into many labels.": "    # MAGFiLOでは1フィラメントにつき1ポリゴンが仕様。異常な複数ポリゴン記録では、\n    # 1個体を複数ラベルへ分断しないよう最大ポリゴンを使う。",
    "# Every independent annotation is useful training supervision.": "# 各アノテータの独立した注釈を、すべて学習教師として利用する。",
    "# [6/12] Train/resume three high-resolution instance models": "# [6/12] 3個の高解像度インスタンスモデルを学習または再開",
    "# [7/12] MediSeg-AI architecture (repository implementation)": "# [7/12] MediSeg-AIアーキテクチャ（リポジトリ実装）",
    "# 1. Combined Loss (BCE + Dice Loss)": "# 1. 複合損失（BCE + Dice損失）",
    "# 2. MediSeg UNet Architecture": "# 2. MediSeg U-Netアーキテクチャ",
    "# 3. MediSeg DeepLabV3Plus with ASPP Module": "# 3. ASPP搭載MediSeg DeepLabV3Plus",
    "# Feature Extractor": "# 特徴抽出部",
    "# [8/12] Individual-instance crop dataset and edge-aware objective": "# [8/12] 個体別cropデータセットとエッジ考慮損失",
    "# [9/12] Train/resume the held-out MediSeg refiner": "# [9/12] 検証用MediSeg refinerを学習または再開",
    "# [10/12] Competition-aligned instance metric and calibration": "# [10/12] コンペ評価に合わせた個体指標とパラメータ調整",
    "# Sequential search avoids repeating expensive mask matching for all 100\n# combinations while retaining the parameters that materially affect count.": "# 高コストなマスク照合を100通りすべてで繰り返さず、個体数に影響する\n# パラメータを保ったまま段階的に探索する。",
    "# [11/12] Final all-data refiner and three-fold ensemble inference": "# [11/12] 全データrefinerと3-foldアンサンブルで最終推論",
    "# The full-data stage starts from the leakage-safe best model and uses a short,\n# lower-LR fit. A held-out loader is retained only for progress diagnostics.": "# 全データ学習はデータ漏洩のない最良モデルから始め、低い学習率で短時間調整する。\n# 検証loaderは進捗確認だけに使う。",
    "# [12/12] Strict official-RLE validation and artifact manifest": "# [12/12] 公式RLEの厳密検証と成果物一覧の保存",
    "# Quick sanity plot: submitted instance count distribution.": "# 簡易確認として、提出する画像別インスタンス数を描画する。",
}
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    for old, new in translations.items():
        cell["source"] = cell["source"].replace(old, new)

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
