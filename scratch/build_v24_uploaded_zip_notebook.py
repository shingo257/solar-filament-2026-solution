"""Build the validated V24 Kaggle notebook from the V23 source notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_kaggle_ensemble_mediseg_v23_uploaded_zip_fixed.ipynb"
TARGET = ROOT / "solar_kaggle_ensemble_mediseg_v24_uploaded_zip_validated.ipynb"


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    replacements = {
        "Version 23 uploaded ZIP fixed": "Version 24 uploaded ZIP validated",
        "submission_v23_uploaded_zip_fixed.csv": "submission_v24_uploaded_zip_validated.csv",
        "source_encoder = [(k, v) for k, v in state.items() if k.startswith('encoder.')]": (
            "source_encoder = [(k, v) for k, v in state.items() "
            "if k.startswith('encoder.') and not k.startswith("
            "('encoder._conv_head', 'encoder._bn1'))]"
        ),
    }

    changed = {key: 0 for key in replacements}
    for cell in notebook["cells"]:
        updated_source = []
        for line in cell.get("source", []):
            for old, new in replacements.items():
                if old in line:
                    line = line.replace(old, new)
                    changed[old] += 1
            updated_source.append(line)
        cell["source"] = updated_source

    missing = [key for key, count in changed.items() if count != 1]
    if missing:
        raise RuntimeError(f"Expected exactly one replacement for each marker: {changed}")

    TARGET.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(TARGET)


if __name__ == "__main__":
    main()
