"""A full set of positive historical/repaired cells is not adoption evidence."""

import importlib.util
import json
import sys
from pathlib import Path


def test_complete_positive_repaired_manifests_remain_exploratory(tmp_path, monkeypatch):
    builder = Path(__file__).resolve().parents[1] / "scratch" / "run_v82_aggregate.py"
    spec = importlib.util.spec_from_file_location("v82_aggregate", builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifacts = tmp_path / "solar_filament_v30" / "artifacts"
    artifacts.mkdir(parents=True)
    for arm in module.ARMS:
        for fold in module.FOLDS:
            for seed in module.SEEDS:
                payload = {"arm": arm, "fold": fold, "seed": seed,
                    "reconstructed_from_printed": True,
                    "oof_pq_by_threshold": {"0.45": {"pq": 0.5, "delta_vs_baseline": 0.1}}}
                (artifacts / f"v82_{arm}_fold{fold}_s{seed}_manifest.json").write_text(
                    json.dumps(payload), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [str(builder), "--drive", str(tmp_path), "--out", str(out)])
    assert module.main() == 0
    report = json.loads((out / "v82_aggregate.json").read_text(encoding="utf-8"))
    assert report["adoption_eligible"] is False
    assert report["result_kind"] == "exploratory_manifest_summary"
    for arm in module.ARMS:
        assert report["verdict"][arm]["complete"] is True
        assert report["verdict"][arm]["all_cells_positive"] is True
        assert report["verdict"][arm]["adoption_eligible"] is False
