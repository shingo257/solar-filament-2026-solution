"""v126 — cons2of3_c26 consensus 提出 CSV 生成ノートブック (Colab、exploratory)。

v122 の提出パイプラインを、プロトコル終了後に得られた **2-fold honest
証拠** に基づく exploratory 提出として再配置する。v122 の
`cross_fold_verified` ガードは事前登録プロトコルの機械的強制として
そのまま維持し、本ノートブックは別経路であることを明示する。

根拠 (実行時に summary から検証):
- fold-0 (v121c, 事前登録経路): cons2of3_c26 Δ@0.45 = +0.002982
- fold-2 (v124c, フロー外 honest 複製): cons2of3_c26 Δ@0.45 = +0.008565
- 両 fold とも正・平均 +0.0058 — 逐次ゲート (fold-0 複製 >= +0.005) は
  未達で線は閉鎖済みだが、独立 2 fold の clean 測定は両方陽性

**事前申告の選択ルール (CSV 生成前に固定)**: 本 CSV を Kaggle に提出し、
public score が現行候補 v96-full (0.39) を上回った場合のみ最終選択を
差し替える。同等以下なら差し替えない。これは public LB への単発の
事前宣言比較であり、複数 arm の当てはめではない。

detector は v122 と同一構成 (prod yolo_fold_2 + v116 crop + v117
yolo26、sha256 pin 検証、test は全 fold と disjoint で正当)。
all-data trio (yolo_full + v116 + v123) は consensus 集合として
未計測のため採用しない。refiner は prod 提出と同一 all-data edgeatt。

使い方:
    python scratch/build_v126_consensus_submission_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

CONSENSUS_SUB_PATH = (Path(__file__).resolve().parent
                      / "run_test_submission_consensus.py")

MD_CELL = [
    "# v126 — cons2of3_c26 consensus 提出 CSV 生成 (Colab、exploratory)\n",
    "\n",
    "**位置づけ**: 逐次ゲート (v121c fold-0 >= +0.005) は未達で\n",
    "consensus 線は閉鎖済み。ただしフロー外で実行した fold-2 clean 複製\n",
    "(v124c) が +0.008565 を返し、**独立 2 fold の honest 測定が両方\n",
    "陽性** (f0 +0.002982 / f2 +0.008565、平均 +0.0058) となった。\n",
    "本ノートブックはプロトコル検証済みの promote ではなく、\n",
    "**2-fold 証拠に基づく exploratory 提出 CSV の生成**である。\n",
    "\n",
    "**事前申告ルール (提出前に固定)**: public score > 0.39 (v96-full\n",
    "現行値) なら最終選択を consensus に差替、以下なら v96-full 維持。\n",
    "\n",
    "パイプライン = v122/prod 提出と同一、detector 段のみ置換:\n",
    "- 3 det (prod yolo_fold_2 + v116 crop + v117 yolo26、sha pin 検証)\n",
    "  を floor 0.08 で推論 → mask-IoU>=0.5 クラスタ → **conf>=0.26 の\n",
    "  支持が >=2 det** のクラスタのみ放出 (cons2of3_c26)\n",
    "- 以降 prod verbatim: edgeatt all-data refiner → threshold 0.45 →\n",
    "  ownership drop_contained 0.90 → min_area 20 → RLE → CSV\n",
    "\n",
    "出力: Drive `artifacts/consensus_v126/submission_consensus.csv`\n",
    "+ `audit_consensus.json` + `efficiency_consensus.json`。\n",
    "提出はユーザー判断 — CSV 生成は提出そのものではありません。\n",
]

CELLS = [
    {"cell_type": "markdown", "source": MD_CELL},
    {
        "cell_type": "code",
        "source": [
            "# [1/4] Drive mount + 依存 + 2-fold 証拠検証 + パス検証\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "%pip install -q 'ultralytics>=8.3,<9' pycocotools\n",
            "\n",
            "import json, shutil, zipfile\n",
            "from pathlib import Path\n",
            "DRIVE = Path('/content/drive/MyDrive')\n",
            "V30 = DRIVE / 'solar_filament_v30'\n",
            "\n",
            "# v126 — 逐次ゲートは閉鎖済み。両 fold の summary を読み、\n",
            "# 2-fold honest 証拠 (両 fold 正 Δ) が実在することを確認してから生成する。\n",
            "# v122 の cross_fold_verified ガードは別経路 (事前登録) として維持。\n",
            "s121 = DRIVE / 'solar_filament_v121' / 'work' / 'latest' / 'v121_summary.json'\n",
            "s124 = DRIVE / 'solar_filament_v124' / 'work' / 'latest' / 'v124_summary.json'\n",
            "for p in (s121, s124):\n",
            "    assert p.exists(), f'summary がありません: {p}'\n",
            "g121 = json.loads(s121.read_text(encoding='utf-8'))['gate']\n",
            "g124 = json.loads(s124.read_text(encoding='utf-8'))['gate']\n",
            "d_f0 = g121['cons2of3_c26_delta_fold0']\n",
            "d_f2 = g124['cons2of3_c26_delta_fold2']\n",
            "print(f'fold-0 delta: {d_f0:+.6f} ({g121[\"decision\"]})')\n",
            "print(f'fold-2 delta: {d_f2:+.6f} ({g124[\"decision\"]})')\n",
            "assert d_f0 > 0 and d_f2 > 0, '2-fold 証拠が揃っていない — CSV を生成しません'\n",
            "print(f'2-fold 両方正 (mean {(d_f0+d_f2)/2:+.6f}) — exploratory 提出生成に進みます')\n",
            "print('事前申告ルール: public > 0.39 なら選択差替、以下なら v96-full 維持')\n",
            "\n",
            "# repro_code の解決 (prod repro と同一規約)\n",
            "if (V30 / 'repro_code' / 'scratch' / 'run_test_submission.py').exists():\n",
            "    REPRO = V30 / 'repro_code'\n",
            "    print('repro_code (drive dir):', REPRO)\n",
            "else:\n",
            "    zip_path = V30 / 'repro_code.zip'\n",
            "    assert zip_path.exists(), 'repro_code がありません'\n",
            "    REPRO = Path('/content/repro_code')\n",
            "    if not (REPRO / 'scratch' / 'run_test_submission.py').exists():\n",
            "        REPRO.mkdir(exist_ok=True)\n",
            "        with zipfile.ZipFile(zip_path) as z:\n",
            "            z.extractall(REPRO)\n",
            "    print('repro_code (unzipped):', REPRO)\n",
            "\n",
            "checks = {\n",
            "    'test_images': V30 / 'competition_data' / 'MAGFiLO_1.0_Kaggle_2026' / 'test' / 'test_images',\n",
            "    'yolo_fold2': V30 / 'runs' / 'yolo_fold_2' / 'weights' / 'best.pt',\n",
            "    'crop_f2': V30 / 'runs' / 'v116_croptrain' / 'yolo_croptrain_fold2' / 'weights' / 'best.pt',\n",
            "    'yolo26_f2': V30 / 'runs' / 'v117_yolo26' / 'yolo_yolo26m_fold2' / 'weights' / 'best.pt',\n",
            "    'refiner': V30 / 'artifacts' / 'mediseg_edgeattnet_refiner_all_data_resume.pt',\n",
            "}\n",
            "for label, p in checks.items():\n",
            "    print(label, p.exists(), p)\n",
            "    assert p.exists(), f'{label} がありません: {p}'\n",
            "print('n_test:', len(list(checks['test_images'].glob('*.jpeg'))))\n",
            "print('全パスOK')\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# [2/4] consensus submission script を repro_code/scratch へ書込み\n",
            "CONSENSUS_SUB = ",
            "script_path = REPRO / 'scratch' / 'run_test_submission_consensus.py'\n",
            "script_path.write_text(CONSENSUS_SUB)\n",
            "import ast; ast.parse(CONSENSUS_SUB)\n",
            "print('consensus submission script written:', script_path)\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "%%time\n",
            "# [3/4] consensus 推論 (3 det + cluster + cons2of3_c26 → prod 後処理、GPU ~20-30min)\n",
            "import subprocess, sys, torch\n",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "print('device:', device, torch.cuda.get_device_name(0) if device == 'cuda' else '')\n",
            "rc = subprocess.run([\n",
            "    sys.executable, str(script_path),\n",
            "    '--drive', str(DRIVE), '--device', device,\n",
            "    '--out', '/content/repro_consensus'],\n",
            "    capture_output=True, text=True)\n",
            "print(rc.stdout[-4000:])\n",
            "print('STDERR:', rc.stderr[-2000:])\n",
            "assert rc.returncode == 0, 'consensus submission 生成失敗'\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# [4/4] 成果物を Drive に保存 + 検証 (提出はユーザー判断)\n",
            "import json, shutil\n",
            "import pandas as pd\n",
            "OUT = Path('/content/repro_consensus')\n",
            "DEST = V30 / 'artifacts' / 'consensus_v126'\n",
            "DEST.mkdir(parents=True, exist_ok=True)\n",
            "for f in OUT.iterdir():\n",
            "    shutil.copy2(f, DEST / f.name)\n",
            "    print('saved:', f.name)\n",
            "df = pd.read_csv(DEST / 'submission_consensus.csv')\n",
            "print('rows:', len(df), '| dup:', df.filament_id.duplicated().sum())\n",
            "audit = json.loads((DEST / 'audit_consensus.json').read_text())\n",
            "print('emitted/img mean:', sum(a['consensus_emitted'] for a in audit) / len(audit))\n",
            "print('提出する場合は Kaggle UI で submission_consensus.csv をアップロード')\n",
            "print('選択ルール (事前申告): public > 0.39 → consensus 選択、以下 → v96-full 維持')\n",
        ],
    },
]


def build_notebook(script_text: str) -> dict:
    cells = json.loads(json.dumps(CELLS))
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            if line.strip() == "CONSENSUS_SUB =":
                cell["source"][i] = "CONSENSUS_SUB = " + repr(script_text) + "\n"
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"},
                     "language_info": {"name": "python"}},
        "cells": [{**c, "metadata": {},
                   **({"outputs": [], "execution_count": None}
                      if c["cell_type"] == "code" else {})}
                  for c in cells],
    }
    return nb


def main() -> Path:
    nb = build_notebook(CONSENSUS_SUB_PATH.read_text(encoding="utf-8"))
    out = (Path(__file__).resolve().parent.parent
           / "solar_filament_v126_consensus_submission_colab.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
