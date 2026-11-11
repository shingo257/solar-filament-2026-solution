"""v122 — cons2of3_c26 consensus 提出 CSV 生成ノートブック (Colab、contingent)。

v120 で promote_candidate → v121c fold-0 eval が `cross_fold_verified`
(Δ@0.45 >= +0.005, §2.99) を出した場合のみ実行する提出パイプライン。
`run_test_submission_consensus.py` (run_test_submission.py prod の派生、
detector 段のみ 3-det + consensus に置換、後処理一切同一) で
`submission_consensus.csv` を生成する。

実行前ガード: `solar_filament_v121/work/latest/v121_summary.json` の
`gate.decision == 'cross_fold_verified'` を必須とする — 未検証構成の
提出 CSV は生成しない (事前登録プロトコルの機械的強制)。

detector は計測済み fold-2 構成 (prod yolo_fold_2 + v116 crop + v117
yolo26、sha256 pin で検証、test は完全 disjoint)。all-data 版は未計測
構成なので採用しない。refiner は prod 提出と同一 all-data edgeatt。

注: v116 crop det は v116 dataset バグにより全 707 file (fold-2 含む) で
学習済み — fold-2 OOF 計測値は無効だが、提出用途 (test は全 fold と
disjoint) では全データ学習 det として正当に使える。判定は fold-0 の
clean 複製 eval (v121a honest + v121c) が担う。

使い方:
    python scratch/build_v122_consensus_submission_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

CONSENSUS_SUB_PATH = (Path(__file__).resolve().parent
                      / "run_test_submission_consensus.py")

MD_CELL = [
    "# v122 — cons2of3_c26 consensus 提出 CSV 生成 (Colab、verified のみ)\n",
    "\n",
    "**前提**: `solar_filament_v121/work/latest/v121_summary.json` の\n",
    "`gate.decision == 'cross_fold_verified'` を実行前に確認 —\n",
    "fold-0 で Δ@0.45 >= +0.005 が再現された場合のみ提出物を生成する\n",
    "(§2.99 事前登録プロトコルの機械的強制)。未検証なら中断します。\n",
    "\n",
    "パイプライン = prod 提出と同一、detector 段のみ置換:\n",
    "- 3 det (prod yolo_fold_2 + v116 crop + v117 yolo26、sha pin 検証)\n",
    "  を floor 0.08 で推論 → mask-IoU>=0.5 クラスタ → **conf>=0.26 の\n",
    "  支持が >=2 det** のクラスタのみ放出 (cons2of3_c26)\n",
    "- 以降 prod verbatim: edgeatt all-data refiner → threshold 0.45 →\n",
    "  ownership drop_contained 0.90 → min_area 20 → RLE → CSV\n",
    "\n",
    "出力: Drive `artifacts/consensus_v122/submission_consensus.csv`\n",
    "+ `audit_consensus.json` + `efficiency_consensus.json`。\n",
    "提出はユーザー判断 — CSV 生成は提出そのものではありません。\n",
]

CELLS = [
    {"cell_type": "markdown", "source": MD_CELL},
    {
        "cell_type": "code",
        "source": [
            "# [1/4] Drive mount + 依存 + v121c verified ガード + パス検証\n",
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
            "# §2.99 — fold-0 複製が cross_fold_verified でなければ提出 CSV を生成しない\n",
            "summary_path = DRIVE / 'solar_filament_v121' / 'work' / 'latest' / 'v121_summary.json'\n",
            "if not summary_path.exists():\n",
            "    raise RuntimeError(\n",
            "        f'v121c summary がありません: {summary_path} — fold-0 eval を先に実行してください')\n",
            "gate = json.loads(summary_path.read_text(encoding='utf-8'))['gate']\n",
            "print('v121c gate:', json.dumps(gate, indent=1)[:600])\n",
            "if gate.get('decision') != 'cross_fold_verified':\n",
            "    raise RuntimeError(\n",
            "        f\"fold-0 複製が未検証 ({gate.get('decision')}) — consensus 提出は生成しません (§2.99)\")\n",
            "print('cross_fold_verified — consensus 提出生成に進みます')\n",
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
            "DEST = V30 / 'artifacts' / 'consensus_v122'\n",
            "DEST.mkdir(parents=True, exist_ok=True)\n",
            "for f in OUT.iterdir():\n",
            "    shutil.copy2(f, DEST / f.name)\n",
            "    print('saved:', f.name)\n",
            "df = pd.read_csv(DEST / 'submission_consensus.csv')\n",
            "print('rows:', len(df), '| dup:', df.filament_id.duplicated().sum())\n",
            "audit = json.loads((DEST / 'audit_consensus.json').read_text())\n",
            "print('emitted/img mean:', sum(a['consensus_emitted'] for a in audit) / len(audit))\n",
            "print('提出する場合は Kaggle UI で submission_consensus.csv をアップロード')\n",
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
           / "solar_filament_v122_consensus_submission_colab.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
