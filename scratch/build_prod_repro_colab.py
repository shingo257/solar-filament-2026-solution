"""Colab 用 prod 再現ノートブック (solar_filament_prod_repro_colab.ipynb) を生成する。

Drive の `repro_code/` (scratch + solar_filament + src のコピー) 上の
`run_test_submission.py --mode prod` を実行するだけの3セル構成。
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "solar_filament_prod_repro_colab.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


CELLS = [
    md("""# Solar Filament 2026 — prod 提出物の再現 (Colab GPU)

prod (=V52, Public LB 0.39) と同じパイプラインを Colab GPU で実行し、
`submission_prod.csv` + `efficiency_prod.json` を Drive に保存します。
コードは Drive の `repro_code/` (ローカル検証済み `run_test_submission.py` と同一) を使います。

**実行順は上から順に** (途中セルが失敗すると以降の変数が未定義になります)。"""),

    code('''# [1/3] Drive mount + 依存インストール + パス検証
from google.colab import drive
drive.mount('/content/drive')

%pip install -q "ultralytics>=8.3,<9" pycocotools

import shutil, zipfile
from pathlib import Path
DRIVE = Path('/content/drive/MyDrive')
V30 = DRIVE / 'solar_filament_v30'

# repro_code の解決: Drive 上の展開済み dir を優先、無ければ zip をローカル解凍
if (V30 / 'repro_code' / 'scratch' / 'run_test_submission.py').exists():
    REPRO = V30 / 'repro_code'
    print('repro_code (drive dir):', REPRO)
else:
    zip_path = V30 / 'repro_code.zip'
    assert zip_path.exists(), (
        'repro_code も repro_code.zip も見つかりません — Drive 同期待ちか'
        ' solar_filament_v30 直下への zip 配置を確認してください')
    REPRO = Path('/content/repro_code')
    if not (REPRO / 'scratch' / 'run_test_submission.py').exists():
        REPRO.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(REPRO)
    print('repro_code (unzipped):', REPRO,
          '| files:', len(list(REPRO.rglob('*.py'))))

checks = {
    'test_images': V30 / 'competition_data' / 'MAGFiLO_1.0_Kaggle_2026' / 'test' / 'test_images',
    'yolo_fold2': V30 / 'runs' / 'yolo_fold_2' / 'weights' / 'best.pt',
    'refiner': V30 / 'artifacts' / 'mediseg_edgeattnet_refiner_all_data_resume.pt',
}
for label, p in checks.items():
    print(label, p.exists(), p)
    assert p.exists(), f'{label} がありません: {p}'
print('n_test:', len(list(checks['test_images'].glob('*.jpeg'))))
print('全パスOK — このまま下のセルへ')
'''),

    code('''# [2/3] prod 推論 (GPU で ~10-20分)
import subprocess, sys, torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, torch.cuda.get_device_name(0) if device == 'cuda' else '')
rc = subprocess.run([
    sys.executable,
    str(REPRO / 'scratch' / 'run_test_submission.py'),
    '--mode', 'prod',
    '--drive', str(DRIVE),
    '--device', device,
    '--out', '/content/repro_prod',
], check=False).returncode
print('returncode:', rc)
assert rc == 0, '推論が失敗しました — 上のログを確認してください'
'''),

    code('''# [3/3] 成果物を Drive に保存 + 検証
import json, shutil
import pandas as pd
OUT = Path('/content/repro_prod')
DEST = V30 / 'artifacts' / 'repro_prod_colab'
DEST.mkdir(parents=True, exist_ok=True)
for f in OUT.iterdir():
    shutil.copy2(f, DEST / f.name)
    print('saved:', f.name)
df = pd.read_csv(DEST / 'submission_prod.csv')
print('rows:', len(df), '| dup:', df.filament_id.duplicated().sum())
eff = json.loads((DEST / 'efficiency_prod.json').read_text())
print('efficiency:', eff)
print('DONE — submission_prod.csv が Drive にあります')
'''),
]


def main() -> int:
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
        "cells": CELLS,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
