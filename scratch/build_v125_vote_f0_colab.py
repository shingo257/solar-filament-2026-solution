"""v125 — 検出集合固定の cross-proposal 輪郭投票 eval (Colab、調査計画 P2)。

v121c (fold-0) で consensus 線は閉鎖 (+0.002982 < +0.005)。別機構として
P2: prod 検出集合 (yolo conf>=0.26) を leader として**固定**し、
crop (v121a honest)・yolo26 (v121b) の提案を mask-IoU>=0.5 で対応付け、
refined mask の等重み pixel 多数決 (>=0.5) で輪郭だけを改善する
探索実験。新規個体は一切出さない (存在フィルタとは別機構)。

detector 3 本 (v121c と同じ honest セット、全て folds(1,2) 学習):
- `yolo`: prod yolo_fold_0 (manifest binding) — leader のみに使用
- `crop`: v121a honest (trusted-pin = v121a manifest)
- `yolo26`: v121b (trusted-pin = v121b manifest)

判定 (事前固定・開発側 eval):
  - vote_eq05 Δ@0.45 (vs prod arm) >= +0.005 → `vote_positive_dev`
    → 確認 fold (fold-2、v124a crop 要) への昇格検討
    未満 → `vote_negative` で P2 線も閉鎖
  - sanity: prod arm PQ@0.45 が fold-0 prod 確定値 0.444137 ±0.001
  - fold-0 は既に v121c 判定に使用済みのため、陽性でも「確認」は
    別 fold でしか取れない (適応的バイアスは開示)

入力: v121c と同一 (fold-0 provenance pin・各 ckpt・学習 manifest)。
eval script は embedded copy of repo scratch/run_fold_oof_eval_vote.py。
"""

from __future__ import annotations

import json
from pathlib import Path

VOTE_SCRIPT_PATH = (Path(__file__).resolve().parent
                    / "run_fold_oof_eval_vote.py")

MD_CELL = [
    "# v125 — 検出集合固定の cross-proposal 輪郭投票 eval (fold-0, Colab)\n",
    "\n",
    "consensus 線は fold-0 で閉鎖 (+0.0030 < +0.005)。別機構 P2:\n",
    "prod 検出集合を leader として**固定**し、crop/yolo26 提案を\n",
    "mask-IoU>=0.5 で対応付け、refined mask の等重み pixel 多数決\n",
    "(>=0.5) で輪郭のみ改善。新規個体は出さない。\n",
    "\n",
    "| arm | mask 合成 |\n",
    "|---|---|\n",
    "| `prod` | leader refined のみ (対照、sanity ≈0.444137) |\n",
    "| `vote_eq05` | leader+voter refined の多数決 (>=0.5 固定) |\n",
    "\n",
    "**判定** (事前固定・開発側):\n",
    "- vote_eq05 Δ@0.45 ≥ +0.005 → `vote_positive_dev` (確認 fold へ検討)\n",
    "- 未満 → `vote_negative` で P2 線も閉鎖\n",
    "- sanity: prod arm ≈ fold-0 prod 確定値 0.444137 ±0.001\n",
    "- fold-0 は v121c 判定済み: 陽性でも確認は別 fold が必要\n",
    "\n",
    "**Provenance pin:** `V73_FOLD0_PROVENANCE_SHA256` (v121c と同一)。\n",
    "yolo det は manifest binding、crop/yolo26 det は v121a/v121b 学習\n",
    "manifest 記録の weights_sha256 を trusted-pin として受理。\n",
    "Drive `solar_filament_v35/kaggle.json` で kaggle 認証 (shingo257)。\n",
    "eval 終了時に Drive `solar_filament_v125/work/latest/` へ sha256 検証付きミラー。\n",
]

CELLS = [
    {"cell_type": "markdown", "source": MD_CELL},
    {
        "cell_type": "code",
        "source": [
            "import os, re, json, subprocess, sys, hashlib, shutil, time\n",
            "from pathlib import Path\n",
            "# Reviewed provenance pin is supplied externally — env var, Colab secret,\n",
            "# Kaggle secret — or declared interactively. Never derived from artifacts.\n",
            "EXPECTED_PROVENANCE_SHA = os.environ.get('V73_FOLD0_PROVENANCE_SHA256', '')\n",
            "if not EXPECTED_PROVENANCE_SHA:\n",
            "    try:\n",
            "        from google.colab import userdata\n",
            "        EXPECTED_PROVENANCE_SHA = userdata.get('V73_FOLD0_PROVENANCE_SHA256') or ''\n",
            "    except Exception:\n",
            "        pass\n",
            "if not EXPECTED_PROVENANCE_SHA:\n",
            "    try:\n",
            "        from kaggle_secrets import UserSecretsClient\n",
            "        EXPECTED_PROVENANCE_SHA = UserSecretsClient().get_secret('V73_FOLD0_PROVENANCE_SHA256')\n",
            "    except Exception:\n",
            "        pass\n",
            "if not re.fullmatch(r'[0-9a-f]{64}', EXPECTED_PROVENANCE_SHA):\n",
            "    try:\n",
            "        EXPECTED_PROVENANCE_SHA = input('V73_FOLD0_PROVENANCE_SHA256 (fold-0 reviewed manifest hash): ').strip()\n",
            "    except (EOFError, OSError):\n",
            "        pass\n",
            "if not re.fullmatch(r'[0-9a-f]{64}', EXPECTED_PROVENANCE_SHA):\n",
            "    raise RuntimeError('Set V73_FOLD0_PROVENANCE_SHA256 after independent provenance review (§2.99: da3d53ab...)')\n",
            "\n",
            "# [0] Drive mount — primary data source (same pattern as all past colab runs)\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "V30 = Path('/content/drive/MyDrive/solar_filament_v30')\n",
            "COMP = V30 / 'competition_data/MAGFiLO_1.0_Kaggle_2026'\n",
            "inputs = {\n",
            "    'train_json': COMP / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',\n",
            "    'test_dir': COMP / 'test/test_images',\n",
            "    'preproc_dir': V30 / 'preprocessed_v30_q96',\n",
            "    'folds_csv': V30 / 'artifacts/grouped_folds.csv',\n",
            "    'baseline_ckpt': V30 / 'artifacts/mediseg_edgeattnet_refiner_fold0_resume.pt',\n",
            "    'yolo_ckpt': V30 / 'runs/yolo_fold_0/weights/best.pt',\n",
            "    'crop_ckpt': V30 / 'runs/v121_croptrain_f0_honest/yolo_croptrain_fold0/weights/best.pt',\n",
            "    'yolo26_ckpt': V30 / 'runs/v121_yolo26_f0/yolo_yolo26m_fold0/weights/best.pt',\n",
            "    'v121a_manifest': V30 / 'artifacts/v121a_croptrain_f0_colab_manifest.json',\n",
            "    'v121b_manifest': V30 / 'artifacts/v121b_yolo26_f0_colab_manifest.json',\n",
            "    'kaggle.json': Path('/content/drive/MyDrive/solar_filament_v35/kaggle.json'),\n",
            "}\n",
            "missing = {k: str(v) for k, v in inputs.items() if not v.exists()}\n",
            "if missing:\n",
            "    raise FileNotFoundError(f'Drive inputs missing: {missing}')\n",
            "train_json, test_dir, preproc_dir = (inputs['train_json'], inputs['test_dir'],\n",
            "                                     inputs['preproc_dir'])\n",
            "folds_csv, baseline_ckpt, yolo_ckpt = (inputs['folds_csv'], inputs['baseline_ckpt'],\n",
            "                                      inputs['yolo_ckpt'])\n",
            "crop_ckpt, yolo26_ckpt = inputs['crop_ckpt'], inputs['yolo26_ckpt']\n",
            "# §2.99 — crop/yolo26 fold-0 重みの trusted-pin は各学習 manifest が正本\n",
            "# (学習時に記録された weights_sha256 を読み取り、eval cell で byte 照合)\n",
            "_v121a_man = json.loads(inputs['v121a_manifest'].read_text(encoding='utf-8'))\n",
            "_v121b_man = json.loads(inputs['v121b_manifest'].read_text(encoding='utf-8'))\n",
            "TRUSTED_SHA256 = {\n",
            "    str(crop_ckpt): _v121a_man['datasets']['croptrain']['weights_sha256'],\n",
            "    str(yolo26_ckpt): _v121b_man['results']['yolo26m']['weights_sha256'],\n",
            "}\n",
            "print('trusted pins from training manifests:',\n",
            "      {Path(k).parent.parent.parent.name: v[:12] for k, v in TRUSTED_SHA256.items()})\n",
            "if not any(test_dir.iterdir()):\n",
            "    raise RuntimeError('test_images dir is empty')\n",
            "EXPECTED_FOLDS_SHA = 'febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e'\n",
            "actual = hashlib.sha256(folds_csv.read_bytes()).hexdigest()\n",
            "if actual != EXPECTED_FOLDS_SHA:\n",
            "    raise RuntimeError(f'fold CSV sha mismatch: {actual} (Drive artifacts/grouped_folds.csv)')\n",
            "print('folds_csv sha256 OK (true experimental split)')\n",
            "\n",
            "# [1] Kaggle auth + code/manifest download\n",
            "os.makedirs('/root/.kaggle', exist_ok=True)\n",
            "cred = inputs['kaggle.json'].read_bytes()\n",
            "print('auth: Drive kaggle.json')\n",
            "try:\n",
            "    _u = json.loads(cred)['username']\n",
            "except Exception:\n",
            "    _u = '?'\n",
            "print('kaggle user:', _u)\n",
            "Path('/root/.kaggle/kaggle.json').write_bytes(cred)\n",
            "os.chmod('/root/.kaggle/kaggle.json', 0o600)\n",
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',\n",
            "                'kaggle', 'ultralytics', 'pycocotools'], check=True)\n",
            "\n",
            "WORK = Path('/content/work'); IN = WORK / 'input'; IN.mkdir(parents=True, exist_ok=True)\n",
            "def _dl(cmd):\n",
            "    r = subprocess.run(cmd, capture_output=True, text=True)\n",
            "    if r.returncode != 0:\n",
            "        raise RuntimeError('download failed rc=%s: %s\\n%s\\n%s' % (\n",
            "            r.returncode, cmd, r.stdout[-800:], r.stderr[-800:]))\n",
            "def _sha(p):\n",
            "    d = hashlib.sha256()\n",
            "    with open(p, 'rb') as h:\n",
            "        for b in iter(lambda: h.read(1 << 20), b''):\n",
            "            d.update(b)\n",
            "    return d.hexdigest()\n",
            "\n",
            "_dl(['kaggle', 'datasets', 'download', 'shingo257/solar-v73-code',\n",
            "     '-p', str(IN), '--unzip', '-q'])\n",
            "\n",
            "code_dir = next(p.parent for p in IN.rglob('train_refiner_fold.py'))\n",
            "sys.path.insert(0, str(code_dir))\n",
            "from v73_split_provenance import (load_reviewed_manifest, load_fold_mapping,\n",
            "                                  validate_checkpoint_binding)\n",
            "# §2.99 — fold-0 reviewed manifest は fold-0 ckpt の独立 pin から\n",
            "# 決定論的に生成し、自己 sha を外部 pin (V73_FOLD0_PROVENANCE_SHA256)\n",
            "# で照合する。fold-2 manifest は fold/binding が fold-0 と不一致のため\n",
            "# 使用しない (checkpoints.yolo/baseline は fold-2 重みの sha)。\n",
            "FOLD0_YOLO_SHA = 'ab8810916e34a755cc0d7a705dbb7ade60338ae247f8baac02adfe64026c0d35'\n",
            "FOLD0_REFINER_SHA = 'caa8b27bbdaa888019a599d001b2d4dee844dbfee90186a18f8e0db95ce8b264'\n",
            "fold0_manifest = {'schema_version': 1, 'fold': 0,\n",
            "                  'folds_sha256': EXPECTED_FOLDS_SHA,\n",
            "                  'checkpoints': {'yolo': FOLD0_YOLO_SHA,\n",
            "                                  'baseline': FOLD0_REFINER_SHA}}\n",
            "manifest_path = WORK / 'v73_reviewed_provenance_fold0.json'\n",
            "manifest_path.write_text(json.dumps(fold0_manifest, sort_keys=True,\n",
            "                                    separators=(',', ':')), encoding='utf-8')\n",
            "manifest = load_reviewed_manifest(manifest_path, EXPECTED_PROVENANCE_SHA, 0)\n",
            "if manifest['folds_sha256'] != EXPECTED_FOLDS_SHA:\n",
            "    raise RuntimeError('reviewed split differs')\n",
            "coco = json.loads(train_json.read_text())\n",
            "load_fold_mapping(folds_csv, {i['file_name'] for i in coco['images']}, 0,\n",
            "                  manifest['folds_sha256'])\n",
            "validate_checkpoint_binding(yolo_ckpt, 'yolo', manifest)\n",
            "validate_checkpoint_binding(baseline_ckpt, 'baseline', manifest)\n",
            "\n",
            "# [2] vote eval variant — embedded copy of repo scratch/run_fold_oof_eval_vote.py (P2・fold-0 実行)\n",
            "VOTE_SCRIPT = ",
            "Path(code_dir / 'run_fold_oof_eval_vote.py').write_text(VOTE_SCRIPT)\n",
            "import ast; ast.parse(VOTE_SCRIPT)\n",
            "print('vote eval script written:', code_dir / 'run_fold_oof_eval_vote.py')\n",
            "for name, p in [('preproc', preproc_dir), ('folds', folds_csv),\n",
            "                ('baseline', baseline_ckpt), ('yolo', yolo_ckpt),\n",
            "                ('crop', crop_ckpt), ('yolo26', yolo26_ckpt),\n",
            "                ('train_json', train_json), ('code', code_dir)]:\n",
            "    print(name, '->', p)\n",
            "import torch\n",
            "print('cuda:', torch.cuda.is_available(),\n",
            "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n",
            "gpu_env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')\n",
            "\n",
            "MIRROR_DIR = Path('/content/drive/MyDrive/solar_filament_v125/work/latest')\n",
            "def mirror():\n",
            "    if not os.path.ismount('/content/drive'):\n",
            "        raise RuntimeError('Drive not mounted')\n",
            "    MIRROR_DIR.mkdir(parents=True, exist_ok=True)\n",
            "    digests = {}\n",
            "    for p in sorted(WORK.rglob('*')):\n",
            "        if p.is_file() and 'input' not in p.relative_to(WORK).parts:\n",
            "            dst = MIRROR_DIR / p.relative_to(WORK)\n",
            "            expected = _sha(p)\n",
            "            if dst.exists() and _sha(dst) == expected:\n",
            "                digests[str(p.relative_to(WORK))] = expected; continue\n",
            "            dst.parent.mkdir(parents=True, exist_ok=True)\n",
            "            shutil.copy2(p, dst)\n",
            "            if _sha(dst) != expected:\n",
            "                raise RuntimeError(f'Drive mirror sha mismatch: {p.name}')\n",
            "            digests[str(p.relative_to(WORK))] = expected\n",
            "            print('mirrored', p.name, p.stat().st_size)\n",
            "    (MIRROR_DIR / 'sha256_manifest.json').write_text(json.dumps(digests, indent=1))\n",
            "    print('mirror DONE ->', MIRROR_DIR)\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "%%time\n",
            "# [eval] 2 arm — fold-0 OOF、prod leader 固定・cross-proposal 輪郭投票 (P2)\n",
            "# crop/yolo26 det の byte-pin を実行時に照合 (trusted-sha256 受理の前提)\n",
            "for path_str, expected in TRUSTED_SHA256.items():\n",
            "    p = Path(path_str)\n",
            "    actual = _sha(p)\n",
            "    if actual != expected:\n",
            "        raise RuntimeError(f'trusted sha256 mismatch: {p} {actual[:16]} != {expected[:16]}')\n",
            "    print('trusted sha256 OK:', p, actual[:16])\n",
            "trusted_args = [x for sha in TRUSTED_SHA256.values() for x in ('--trusted-sha256', sha)]\n",
            "r = subprocess.run([sys.executable, str(code_dir / 'run_fold_oof_eval_vote.py'),\n",
            "                    '--train-json', str(train_json),\n",
            "                    '--test-dir', str(test_dir),\n",
            "                    '--preproc-dir', str(preproc_dir),\n",
            "                    '--folds-csv', str(folds_csv), '--fold', '0',\n",
            "                    '--provenance-manifest', str(manifest_path),\n",
            "                    '--provenance-sha256', EXPECTED_PROVENANCE_SHA,\n",
            "                    '--detector', f'yolo={yolo_ckpt}',\n",
            "                    '--detector', f'crop={crop_ckpt}',\n",
            "                    '--detector', f'yolo26={yolo26_ckpt}',\n",
            "                    '--model', 'edgeatt',\n",
            "                    *trusted_args,\n",
            "                    '--refiner', f'baseline={baseline_ckpt}',\n",
            "                    '--out', str(WORK / 'v125_eval.json')],\n",
            "                   capture_output=True, text=True, env=gpu_env)\n",
            "print(r.stdout[-4000:]); print('STDERR:', r.stderr[-2000:])\n",
            "if r.returncode != 0 or not (WORK / 'v125_eval.json').exists():\n",
            "    raise RuntimeError(f'eval failed rc={r.returncode}')\n",
            "mirror()\n",
        ],
    },
    {
        "cell_type": "code",
        "source": [
            "# 結果サマリ — 開発側判定 (P2: vote_eq05 の fold-0 dev eval)\n",
            "rep = json.load(open(WORK / 'v125_eval.json'))\n",
            "arms = rep['arms']\n",
            "pq = lambda a: arms[a]['pq_by_threshold']['0.45']\n",
            "print(json.dumps({a: r['pq_by_threshold'] for a, r in arms.items()}, indent=2))\n",
            "print('leaders/img:', rep.get('leaders_per_image'),\n",
            "      ' voters/leader:', rep.get('voters_per_leader'),\n",
            "      ' unassigned:', rep.get('unassigned_voters'))\n",
            "base = pq('baseline::prod')\n",
            "deltas = {a: round(pq(a) - base, 6) for a in arms if a != 'baseline::prod'}\n",
            "rep_delta = deltas['baseline::vote_eq05']  # P2 の複製対象 arm\n",
            "gate = {\n",
            "  'prod_pq_0.45': base,\n",
            "  'sanity_prod_near_v108': abs(base - 0.444137) < 0.001,\n",
            "  'prod_diff_vs_v108': round(base - 0.444137, 6),\n",
            "  'delta_by_arm': deltas,\n",
            "  'vote_eq05_delta_fold0': rep_delta,\n",
            "  'v121c_reference_delta': 0.002982,  # consensus 線 (閉鎖済み) 参照\n",
            "  'note': 'fold-0 は v121c 判定済み — 陽性でも確認は別 fold が必要',\n",
            "  'decision': ('vote_positive_dev' if rep_delta >= 0.005\n",
            "               else 'vote_negative'),\n",
            "}\n",
            "print(json.dumps(gate, indent=2))\n",
            "Path(WORK / 'v125_summary.json').write_text(json.dumps(\n",
            "    {'eval': rep, 'gate': gate}, indent=2))\n",
            "mirror()\n",
        ],
    },
]


def build_notebook(script_text: str) -> dict:
    cells = json.loads(json.dumps(CELLS))
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            if line.strip() == "VOTE_SCRIPT =":
                cell["source"][i] = "VOTE_SCRIPT = " + repr(script_text) + "\n"
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
    nb = build_notebook(VOTE_SCRIPT_PATH.read_text(encoding="utf-8"))
    out = (Path(__file__).resolve().parent.parent
           / "solar_filament_v125_vote_f0_colab.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
