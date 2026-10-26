# Inactive mixed-source Solar fingerprint index

This index records 25 Solar paths in retained mixed source commit
a49c1adc5c6ef2e3a02ce76d050399056fee1243, compared with the fixed main snapshot
c2c4098d972beda92805a52e714a4df08144ccdd.
[MANIFEST.json](MANIFEST.json) records each path, both commits, Git blob IDs,
raw-byte SHA256 and byte lengths. Its current snapshot is fixed; it does not
track future main.

The suffix 20260828 preserves a historical branch label. It does not establish
a collection or experiment date. The start of the machine-learning work is not
established by this index. Commit timestamps, this index's creation date, and
dates inside the original records must not be used to infer that start.

## Disposition of all 25 paths

| Classification | Paths | What this index establishes |
|---|---:|---|
| evolved_readme | 1 | The current README has later documentation and distinguishes legacy implementations from compatibility shims. Keep the current version |
| hash_reference_text_difference | 6 | The contract/history Markdown texts match only after masking 64-hex references. Their original bytes and references differ |
| fingerprint_leaf_difference | 16 | One model-spec JSON and 15 output/provenance JSON files differ in fingerprint leaves; no old payload is restored |
| previously_ported_code_with_guard_corrections | 2 | PR #35 ported the code with bounded guard corrections; no runtime code changes are made here |

The complete path inventory and each classification are in the manifest.
The six Markdown paths include endpoint registration, grouped nested split and
harm-first model contracts, plus historical experiment/rescore/handover reports.
The 16 JSON paths include the harm-first model specification and the existing
S0/S1 manifests, readiness/geometry/feature reports and execution-plan records.
They are indexed as historical fingerprints, not imported as new results.

## Prior code recovery

PR #35, merge commit 140d8d62c6076d573e47c97cde5f56f908fc3d96,
already recovered scratch/evaluate_s1_prediction_geometry_gate.py and
solar_filament/s1/set_context_features.py. The current snapshot retains exactly
those ported file bytes. Relative to the mixed source, the former avoids calling
max on an empty eligible set; the latter clamps overlap-neighbor counts to
zero. This is evidence of a bounded source port, not proof that all old runtime
behavior, model predictions or scientific results are equivalent.

The current README also replaces obsolete status and broad shim descriptions
with later information. This index does not roll it back or certify its research
claims. Later documentation and unrelated work remain outside this comparison.

## Provenance and acceptance limits

Masking hashes in text, or observing that JSON differences occur only at
fingerprint leaves, does **not** establish equality of the original raw inputs,
models, output payloads or scientific provenance. A reference inside an old
manifest is an original statement, not a newly verified input identity.
The index's SHA256 values identify the Git payload bytes at the two commits;
they do not authenticate external artifacts, collected data, run environments,
performance numbers, split independence or evaluation histories. No hash is
rewritten to make the old and current records appear interchangeable.

No original report, specification, contract, generated output, dataset, image,
weight or log payload is copied into this directory. This is a fingerprint-only
record, not raw source preservation. Hashes alone cannot recover uncopied bytes:
keep the retained source objects/references or a separately verified recoverable
bundle. The shared mixed source reference, the 23 pending RSNA data paths and
all stashes remain unresolved by this index; it authorizes no retirement.

This index grants no registration, fit, training, model promotion, scoring,
independent scientific acceptance, submission or external-data access.
The original gates remain in force. It authorizes no GONG retry, GPU/cloud job,
paid review, API call or Actions execution.

See the [current module README](../../../README.md) for current navigation.
This historical index supplies no replay instructions.
