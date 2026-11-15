"""cons2of3_sptal_maj3 第1提出 (Public LB 0.40, ref 58363293) のクリーン環境再現 Colab notebook 生成器。

審査経路と完全一致させるため、重みは公開 standalone repo を新規 clone + `git lfs pull`
した `weights/` から、fold CSV は repo `data/` から解決する。Drive の既存成果物ルート
(`solar_filament_v30`) は使わず、`/content/solar_clean_root` を新しい成果物ルートにする。
all-data refiner (`refiner_all_data.pt`) が `weights/` に同梱されているため、
fold 0-2 refiner の再学習は自動スキップされ、A100 で約 5-6 分で完走する。

生成した `submission_cons2of3_sptal_maj3.csv` を Kaggle 提出済み CSV
(`ref 58363293`, sha256 `9e22940183ebf88f…`) と byte 比較し、GPU 差分がある場合は
`compare_submissions.py` で 1対1 Hungarian IoU 照合も記録する。

使い方:
    python scratch/build_v139_sptal_clean_repro_colab.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
SOURCE = MODULE / "solar_filament_submission_reproduction.ipynb"
OUTPUT = MODULE / "solar_filament_v139_sptal_clean_repro_colab.ipynb"

REPO_URL = "https://github.com/shingo257/solar-filament-2026-solution.git"
DRIVE_DIR = "/content/drive/MyDrive/solar_filament_v30/v139_sptal_clean_repro"
REFERENCE_CSV = "reference_submission_58363293.csv"
REFERENCE_SHA256 = "9e22940183ebf88f7bbcde577593b0c6bc95c73e793f789b576092b1cc492184"
FULL_YOLO_SHA256 = "f54c812d7406dafad82f02b6d320b634ccf0efb21f2f78c9d1be44fa41895afa"
YOLO26_SHA256 = "25002d23f7b915e8076ab7e2429d34f6b0c4c6aed72d01abfc3a9985a5938ca9"
SPTAL_SHA256 = "f6623992b998ded0a460d5e61b5d60e7a7723bad9814b10c02cd99e4b958d00d"
REFINER_ALL_SHA256 = "e9f0e8dd7ff104d922177e82b63755d9b025c22b8f45bf9aff3d7ea9aa406fe2"

SETUP_PATCHES = (
    ('    SUBMISSION_VARIANT = "prod"\n', '    SUBMISSION_VARIANT = "cons2of3_sptal_maj3"\n'),
    ('        DRIVE_ROOT = Path("/content/drive/MyDrive/solar_filament_v30")\n',
     '        DRIVE_ROOT = Path("/content/solar_clean_root")\n'),
    ('        PREPROC_CACHE = DRIVE_ROOT / "preprocessed_v30_q96"\n',
     '        PREPROC_CACHE = None\n'),
    ('    DATA_SEARCH_ROOTS = [Path("/kaggle/input")]\n',
     ('    DATA_SEARCH_ROOTS = [Path("/kaggle/input"), '
      'Path("/content/solar_clean_root/competition_data")]\n')),
)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def patch_setup(source: str) -> str:
    for old, new in SETUP_PATCHES:
        if source.count(old) != 1:
            raise ValueError(f"setup patch anchor not unique: {old.strip()}")
        source = source.replace(old, new)
    return source


HEADER = f"""\
# v139 — 第1提出 `cons2of3_sptal_maj3` (Public LB 0.40, ref 58363293) クリーン環境再現 (Colab)

- 重み: 公開 repo `{REPO_URL}` を新規 clone + `git lfs pull` した `weights/` (8 重みすべて SHA256 検証)
- fold CSV: repo `data/grouped_folds.csv` (再現 notebook の pin `24367120…`)
- 競技データ: Drive `solar_filament_v30/competition_data` をクリーン領域へコピー
- 成果物ルート: `/content/solar_clean_root` (Drive の既存 artifacts・checkpoint は一切参照しない)
- 判定: 生成 CSV (`submission_cons2of3_sptal_maj3.csv`) と提出済み CSV (Kaggle ref `58363293`, Public LB `0.40`,
  sha256 `{REFERENCE_SHA256[:16]}…`) の byte 一致および 1対1 Hungarian IoU 照合 (A100 で約 5〜6 分で完走)。
"""

CELL_SETUP = f"""\
# [1/5] Drive・GPU・固定値検証
from google.colab import drive
drive.mount("/content/drive")
import hashlib, json, os, shutil, subprocess, time
from pathlib import Path
import torch
assert torch.cuda.is_available(), "GPU ランタイムを選択してください"
DRIVE_DIR = Path("{DRIVE_DIR}")
SOURCE_NB = DRIVE_DIR / "{SOURCE.name}"
SOURCE_SHA256 = "{{source_sha256}}"
REFERENCE = DRIVE_DIR / "{REFERENCE_CSV}"
REFERENCE_SHA256 = "{REFERENCE_SHA256}"
FULL_YOLO_SHA256 = "{FULL_YOLO_SHA256}"
YOLO26_SHA256 = "{YOLO26_SHA256}"
SPTAL_SHA256 = "{SPTAL_SHA256}"
REFINER_ALL_SHA256 = "{REFINER_ALL_SHA256}"
REPO = Path("/content/solar-filament-2026-solution")
CLEAN_ROOT = Path("/content/solar_clean_root")
STARTED = time.time()

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

for path, expected in ((SOURCE_NB, SOURCE_SHA256), (REFERENCE, REFERENCE_SHA256)):
    actual = sha256(path)
    assert actual == expected, f"{{path.name}} sha256 不一致: {{actual}}"
print({{"gpu": torch.cuda.get_device_name(0), "source_nb": SOURCE_SHA256[:16], "reference": REFERENCE_SHA256[:16]}})
"""

CELL_REPO = f"""\
# [2/5] 公開 repo を新規 clone + LFS 実体化し、全 8 重みを manifest で検証
if REPO.exists():
    shutil.rmtree(REPO)
if subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y", "git-lfs"], check=True)
subprocess.run(["git", "lfs", "install"], check=True)
subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}", str(REPO)], check=True)
subprocess.run(["git", "lfs", "pull"], cwd=REPO, check=True)
REPO_HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout.strip()
manifest = json.loads((REPO / "weights" / "weights_manifest.json").read_text(encoding="utf-8"))
weight_check = {{w["file_name"]: sha256(REPO / "weights" / w["file_name"]) == w["sha256"]
                for w in manifest["weights"]}}
assert all(weight_check.values()), weight_check
assert sha256(REPO / "weights" / "yolo_full_fold2_best.pt") == FULL_YOLO_SHA256
assert sha256(REPO / "weights" / "yolo26m_all_fold0_best.pt") == YOLO26_SHA256
assert sha256(REPO / "weights" / "yolo_spine_small_tal_all_fold2_best.pt") == SPTAL_SHA256
assert sha256(REPO / "weights" / "refiner_all_data.pt") == REFINER_ALL_SHA256
print({{"repo_head": REPO_HEAD, "weights_count": len(weight_check), "weights_ok": all(weight_check.values())}})
"""

CELL_DATA = """\
# [3/5] 競技データをクリーンな成果物ルートへコピー (Drive の artifacts/runs は使わない)
if CLEAN_ROOT.exists():
    shutil.rmtree(CLEAN_ROOT)
(CLEAN_ROOT / "competition_data").mkdir(parents=True)
shutil.copytree(Path("/content/drive/MyDrive/solar_filament_v30/competition_data/MAGFiLO_1.0_Kaggle_2026"),
                CLEAN_ROOT / "competition_data" / "MAGFiLO_1.0_Kaggle_2026")
print({"copied": sum(1 for _ in (CLEAN_ROOT / "competition_data").rglob("*.jpeg"))})
"""

CELL_RUN = f"""\
# [4/5] 再現 notebook を setup 置換 4 箇所のみで全コードセル実行 (cwd = clone した repo)
SETUP_PATCHES = {SETUP_PATCHES!r}
nb = json.loads(SOURCE_NB.read_text(encoding="utf-8"))
sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
for old, new in SETUP_PATCHES:
    assert sources[0].count(old) == 1, old
    sources[0] = sources[0].replace(old, new)
os.chdir(REPO)
ip = get_ipython()
for index, src in enumerate(sources):
    print(f"===== source code cell {{index + 1}}/{{len(sources)}} =====", flush=True)
    result = ip.run_cell(src)
    result.raise_error()
"""

CELL_VERIFY = """\
# [5/5] 判定: 提出済み CSV (58363293, LB 0.40) との byte 一致 + Hungarian IoU 照合 → Drive に保存
import sys
generated = Path(globals().get("OUTPUT_PATH", "/content/submission_cons2of3_sptal_maj3.csv"))
gen_sha = sha256(generated)
report = {"repo_head": REPO_HEAD, "source_nb_sha256": SOURCE_SHA256,
          "generated_sha256": gen_sha, "reference_sha256": REFERENCE_SHA256,
          "byte_identical": gen_sha == REFERENCE_SHA256,
          "generated_rows": sum(1 for _ in open(generated, encoding="utf-8")) - 1,
          "reference_rows": sum(1 for _ in open(REFERENCE, encoding="utf-8")) - 1,
          "gpu": torch.cuda.get_device_name(0),
          "wall_s": round(time.time() - STARTED, 1)}
if not report["byte_identical"]:
    sys.path.insert(0, str(REPO / "scratch"))
    out = DRIVE_DIR / "iou_compare.json"
    subprocess.run([sys.executable, str(REPO / "scratch" / "compare_submissions.py"),
                    str(generated), str(REFERENCE), "--out", str(out)], check=True)
    report["iou_compare"] = json.loads(out.read_text(encoding="utf-8"))
shutil.copy2(generated, DRIVE_DIR / "submission_cons2of3_sptal_maj3_clean_repro.csv")
(DRIVE_DIR / "clean_repro_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
"""


def main() -> None:
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    src_nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    setup = "".join(next(c for c in src_nb["cells"] if c["cell_type"] == "code")["source"])
    patch_setup(setup)
    cells = [md(HEADER), code(CELL_SETUP.replace("{source_sha256}", source_sha)),
             code(CELL_REPO), code(CELL_DATA), code(CELL_RUN), code(CELL_VERIFY)]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    import ast
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    print(f"wrote {OUTPUT} (source sha256 {source_sha[:16]})")


if __name__ == "__main__":
    main()
