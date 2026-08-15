"""Build V25 by tightening validation around torchvision's unused final head."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solar_kaggle_ensemble_mediseg_v24_uploaded_zip_validated.ipynb"
TARGET = ROOT / "solar_kaggle_ensemble_mediseg_v25_uploaded_zip_validated.ipynb"


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    replacements = {
        "Version 24 uploaded ZIP validated": "Version 25 uploaded ZIP validated",
        "submission_v24_uploaded_zip_validated.csv": "submission_v25_uploaded_zip_validated.csv",
        "allowed_missing = {k for k in model.features.state_dict() if k.startswith('8.')}": (
            "allowed_missing = {k for k in missing if k.startswith('8.')}"
        ),
    }
    changed = {key: 0 for key in replacements}
    for cell in notebook["cells"]:
        new_source = []
        for line in cell.get("source", []):
            for old, new in replacements.items():
                if old in line:
                    line = line.replace(old, new)
                    changed[old] += 1
            new_source.append(line)
        cell["source"] = new_source
    if any(count != 1 for count in changed.values()):
        raise RuntimeError(f"Unexpected replacement counts: {changed}")
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
