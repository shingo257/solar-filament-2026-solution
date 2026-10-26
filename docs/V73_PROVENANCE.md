# v73 provenance gate

The legacy 1,889-frame corpus and legacy weight filenames do not establish acceptance.
Rebuild the corpus with the current builder and the actual competition train JSON and
test_images directory. A matching policy name alone is insufficient: complete config,
observation anchors, manifest source identities and decoded frame hashes must validate.

## Inputs required before execution

- A rebuilt corpus with full corpus_config.json, corpus_summary.json, manifest.jsonl
  and batch files. Dry runs, incomplete metadata, changed anchors and inclusive
  train/test +/-360-minute overlap across sites are rejected.
- Actual train annotations and test_images. The Colab downloader currently fetches
  only train annotations; provide test_images separately in the prepared input assets.
  Missing test images intentionally block preflight.
- Current matching code, including v73_split_provenance.py and build_gong_ssl_corpus.py.
  Previously uploaded private code/data assets are not upgraded by this Git change.
- An independently reviewed JSON manifest and its full SHA256, pinned as
  V73_PROVENANCE_SHA256 in the notebooks. No approved manifest or replacement dataset
  is supplied by this source change.

## Reviewed manifest format

schema_version is 1; fold is the held-out integer (0, 1 or 2); folds_sha256 is the
full SHA256 of the frozen split CSV; checkpoints maps role names to full SHA256
of checkpoint bytes. The yolo role is mandatory. Legacy refiners without metadata
also need their explicit arm role (for example baseline). These byte bindings must
come from a review of actual training provenance; a filename is not evidence.

Do not compute the expected manifest hash from the manifest during execution.
Resolve the documented competing split histories before approving legacy bindings.
A reviewed legacy binding does not override contradictory checkpoint metadata.
New refiner checkpoints must retain their fold and folds_sha256.

## Checkpoint format and checks

SSL encoder checkpoints now contain encoder and corpus_provenance. Bare legacy
encoder dictionaries are rejected. Corpus provenance carries config/manifest hashes
and canonical train/test observation hashes. Training, repeated resume and evaluation
compare protected observations with the current actual inputs before CUDA work.

The trainer requires --expected-folds-sha256 and --test-dir. The evaluator requires
--provenance-manifest, --provenance-sha256 and --test-dir. The notebooks pass these
inputs and validate corpus and legacy weight bindings before training.

Use pretrain_unet_encoder_ssl.py --validate-only with --corpus, --train-json,
--test-dir and --out to inspect provenance without writing a checkpoint or starting
model work. Validation still reads and hashes corpus frames.

Source/CPU contract checks are not a new OOF score, GPU acceptance, independent
official evaluation or model-promotion decision. Keep PR #388 Draft until its
remaining experimental acceptance is resolved.


## Recovered source evidence (2026-10-18)

The original Drive artifacts were read through the Drive connector after local
sync reads returned OSError22. These are historical records, not new experiment results.

| Artifact | Raw bytes | SHA256 |
| --- | ---: | --- |
| Drive artifacts/grouped_folds.csv | 20089 | `febb3f67237a058fa8a71f188fac262035048b55d860c0e4509514877f6af67e` |
| Tracked output/kernel_run/solar_filament_v30/artifacts/grouped_folds.csv | 16983 | `24367120279820997396f7063b1ae5b235a442fd1c92617e1e8941d472504c3b` |
| Drive v51_edgeattnet_refiner_3fold_manifest.json | 2525 | `de9e6cbc20423cb2334b892ef668d2f64da7090bceac2dee9255f248a8d86985` |

The source split hash equals the v73 notebook's expected experimental split.
Both tables have707 unique file names and fold counts236/236/235, but489 assignments
differ. This is not simply a permutation of fold labels. The Drive table also
contains reference_instances and labeler_count; the tracked table has only file_name/fold.
Neither table was replaced or promoted by this audit.

The original v51 record corroborates the transcribed metric values in
[output/v51/v51_edgeattnet_refiner_3fold_manifest.json](../output/v51/v51_edgeattnet_refiner_3fold_manifest.json):
PQ/delta agree after rounding to6 decimals; CI, frac_positive, crop Dice (a renamed
key), and combined summaries agree exactly. Some prose and schema differ; the
transcript is not a byte-for-byte copy. Its original transcription history is retained.

This closes the missing-original/transcription comparison gap only. The v51 record
contains checkpoint paths, but no checkpoint hashes or frozen training-split binding.
The v35 record hashes standard MediSegUNet refiners, which cannot attest the target
EdgeAttNet weights. The v30 run_manifest refers to a standard all-data refiner.
No approved v73 weight manifest is established, and corpus provenance/actual
experimental acceptance remain pending. Keep PR #388 Draft.

## Actual protected-input audit (2026-10-18)

The original competition training JSON was retrieved from Drive file
`1SBzVIhqhYbSB4n1oaliz3pCeB-ICHzDj`: 48,616,700 bytes, SHA256
`5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`.
Its 1,154 image rows contain 707 unique file names. The recovered frozen split
covers exactly those names; fold 2 holds out 235 names.

Using the current `v73_split_provenance.observation_digest`, the protected
observation hashes are:

- Training: `ee9354dc938f1257e3ce9cce72d8a101f24d891a9c2e085b6bc4cbfd9d41dfd6`.
- Test: `a3dc309d018a90a09140824f8eda5969c3224efbd3cb2afb30b8698b71a2ac84`.

The test hash uses the 180 actual file names listed in Drive folder
`1i3tDAtqu296jjixvKuL8482knnbQl9rm`. This is a metadata-only observation
identity audit: test image bytes were not downloaded or validated. The helper
canonicalizes unique UTC timestamps across sites; these hashes are not image
content hashes. They identify the protected observations for a future corpus
validation and do not prove that any existing corpus excluded them.

The original v51 fold-0 manifest (Drive file
`1m_d0rbKtiJUk9-D69DV9163FlJS5Sx1e`, 942 bytes, SHA256
`a28a84d38c42d27d1c302418b9adee7bbcb6d8934e93f5a60587d0203858d20a`)
records checkpoint SHA256
`caa8b27bbdaa888019a599d001b2d4dee844dbfee90186a18f8e0db95ce8b264`.
The checkpoint bytes were not checked; a fold-0 record cannot bind the target
fold-2 baseline. No fold-2 manifest was found in the 135-entry artifacts listing
or the scoped filename search. This does not establish global absence.

The remaining legacy roles are the EdgeAttNet fold-2 baseline
(`artifacts/mediseg_edgeattnet_refiner_fold2_resume.pt`) and YOLO fold-2
(`runs/yolo_fold_2/weights/best.pt`, prepared alias `yolo_fold2_best.pt`).
Historical source reads `artifacts/grouped_folds.csv` but does not freeze its
hash in the checkpoint. Actual target bytes and their original training-split
binding, a reviewed manifest, rebuilt-v2 decoded-frame checks and experimental
acceptance are still required. No approved binding is created by this audit.
## Input claims and independent acceptance (2026-10-18)

A later source update reports uploaded `solar-v73-assets` and `solar-gong-ssl`
inputs, including a manifest with SHA256
`2c9546855df1b601a06e367b37ade4db2aa72086452bdb9fb416e12ca708a2eb`.
These are reported inputs, not an independently accepted weight-to-split binding.
Matching a checkpoint's best_dice to the v51 score does not establish the split
used for its training. Both notebooks require V73_PROVENANCE_SHA256 to be supplied
from independent review; they do not default to this unverified manifest hash.

The actual Drive YOLO fold-2 best.pt (file
`1yDH5XcIUcc1TlqgNB3xSQ8j0Vc7GtSt6`) was downloaded and hashed without
unpickling: 45,316,918 bytes, SHA256
`40ec31fd2c6aa8e32c952e64be453f4351ace339d31ef747b4f28624eafe7e8c`.
This confirms its recorded byte identity, not its original training-split binding.
The target EdgeAttNet fold-2 resume file is 564,101,827 bytes (Drive file
`1t_RtU0FPQOWVm1fiqslug1bNexuluyrb`); its download was rejected by the
connector's 268,435,456-byte limit. Its bytes remain unverified by this audit.

Legacy corpus migration is optional and requires a preexisting, independently
pinned source record with the original manifest hash, sampling configuration and
per-frame decoded hashes. It cannot create those expected hashes from the current
unverified corpus. Use `upgrade_gong_corpus_v2.py --corpus OLD --out NEW`
with actual `--train-json`, `--test-dir`, `--source-record` and
`--expected-source-record-sha256`. Migration preserves the source, validates
staged output with the full corpus loader and publishes only to a new directory.
Without that source record, rebuild from actual source data instead.

No actual migrated corpus, checkpoint-to-split binding, reviewed v73 manifest or
GPU experiment is accepted by these source changes. Keep PR #388 Draft.

## Trusted source record and operator pin (2026-10-18, second pass)

The immutable `solar-gong-ssl` dataset version 1 (241,812,666-byte zip,
`?datasetVersionNumber=1`) anchors the legacy corpus. From those bytes the
trusted source record was derived:

- `v1_source_record.json` SHA256:
  `b359767ec7ee918691c46d29170b30d31493448344480bddd624ff139889d616`
- binds original `manifest.jsonl` SHA256
  `d9fd444078b1fb35449e0d4ced69d14fc43af4b58dd4d37b16a4187b93025acc`,
  actual sampling config `{year: 2011, days_per_month: 8, frames_per_day: 20}`
  (manifest evidence: 96 distinct days, 8/month, ~20 frames/day — the builder
  defaults of 4×12 were NOT used) and 1,889 ordered decoded-frame SHA256s.

The source-record pin is a reproducibility claim to check against the immutable
version-1 archive. Committing the pin does not itself complete independent review
or prove that decoded legacy frames correspond to the recorded original FITS bytes.

Both notebooks resolve V73_PROVENANCE_SHA256 from an external environment value,
a Colab/Kaggle secret or an operator prompt. The candidate manifest hash
2c9546855df1b601a06e367b37ade4db2aa72086452bdb9fb416e12ca708a2eb
was verified against the downloaded 844-byte Kaggle file. Its review_notes bind
the baseline by matching best_dice; they do not establish the original training
split. Do not use this candidate as an approved pin until independent review
resolves that binding. The approximately 0.4483 OOF sanity result is a functional
comparison only and cannot prove which observations the checkpoint trained on.

A subsequent download of the target baseline copy from Kaggle solar-v73-assets
verified 564,101,827 bytes and SHA256
`b1c0f3536056e72b9e600439fda6bfb174dd78409ddde112a0eae083567de303`.
This agrees with the candidate manifest byte identity; it was not unpickled and
does not establish the historical training split. The immutable solar-gong-ssl
version-1 ZIP was also retrieved: 241,812,666 bytes, SHA256
`ec43bf03791ec64bc447874e2c5959873f7cc5808d9919ffd32414c2f790b13d`.
Its original manifest hash matches the recorded d9fd4440...25acc and has 1,889
rows. Ordered decoded-frame/source-record comparison remains separate.

## Version-3 corpus content and persistence checks (2026-10-18)

All eleven files of Kaggle solar-gong-ssl version 3 were downloaded with the
explicit version identifier. The 1,707 kept manifest rows exactly preserve the
version-1 ordering after the 182 inclusive train/test +/-360-minute overlaps
are removed. All 1,707 decoded frame hashes match the independently inspected
version-1 frames. Batch counts are 250/250/250/250/250/250/207.

- Version-3 manifest SHA256: `f869e7678981587c07cc3c47583d798ca4aae4f1e01c115af5b27c36532c2f04`.
- Version-3 config SHA256: `2a1ae10d52883432a5341d77a19345f1683374a1348736357b51819e9468d9de`.
- The dropped-file list exactly matches the actual protected-observation audit.

This establishes the version-1-to-version-3 content relationship and temporal
filtering, not the original sampling settings or raw FITS-to-decoded-image
correspondence. The source record named by the migration summary has not been
independently accepted. The remaining checkpoint training-split/experimental gates remain separate.

The optional Colab mirror now stops on Drive mount failure or a missing mount,
checks copied destination hashes, and checks the mount again before reporting
completion. Mock tests cover mount failure, absent mount, corrupt copy and a
valid copy. No real Drive persistence/session-disconnection acceptance was run.

Actual-input CPU preflight subsequently passed with the downloaded version-3
corpus, original 48,616,700-byte train JSON and 180 real downloaded JPEGs
from solar-v73-assets version 3 (142,985,911 bytes total). The command used
`pretrain_unet_encoder_ssl.py --validate-only`; it created no checkpoint and
started no GPU/training work. Reported config fingerprint: `a1f436fea3f26380`.
The loader verified both protected-observation hashes, metadata, the exclusion
window and all 1,707 decoded frame hashes. This is CPU contract acceptance,
not original sampling/FITS provenance, checkpoint training-split binding or
experimental performance acceptance.

## Original-source and checkpoint-record audit (2026-10-18)

A diagnostic sample (first, middle and last retained row) was downloaded directly
from the official GONG URLs. All three raw source byte counts/SHA256 values and
reconstructed image SHA256 values matched. This sample does not establish full
source acceptance.

The subsequent full-source batch audit strictly verified 357 distinct rows out
of 1,707 against the current version-3 manifest: original FITS byte count/hash
and decoded-frame hash all matched. The archive then returned HTTP 403; the job
was stopped. The remaining 1,350 rows in that batch are unverified. There were
984 HTTP-403 failure records, which are not image/hash failures and are not
counted as successful checks. The final batch log SHA256 is
`8ff9aa9875d1433b9de3dabe2746da1529bbc321acbf25be065ced0ef0801b55`.
An independent offline join checked every successful row against the manifest.
The local audit has a persistent access-refusal guard; no further archive reads
or automatic retry are scheduled. Full FITS/source acceptance remains open.

Safe checkpoint ZIP/pickle-opcode inspection did not execute pickle code or load
tensors. The actual baseline checkpoint stores only epoch, model, best_model,
optimizer, scheduler and best_dice. The actual YOLO checkpoint preserves fold-2
training arguments (including its data.yaml path, seed and run name), but neither
checkpoint stores the frozen CSV hash or training file inventory. Tracked weight
manifests corroborate the actual checkpoint hashes only. The historical v51
fold-2 notebook has no saved execution outputs. None of these records establishes
the original checkpoint-to-split binding.

Useful missing evidence is an original run-bound train/validation inventory or
cache, plus an executed v51 fold-2 record tying the split and checkpoint bytes.
Do not infer that binding from the training directory name or matching Dice/PQ.
Original sampling settings, full FITS correspondence, historical split binding
and GPU experimental acceptance remain unaccepted.

## Recovered historical settings and OOF inventory (2026-10-19)

The preexisting [research record](../RESEARCH_LOG.md) documents the original
GONG observation sampling: observation year 2011, 8 days per month and 20 frames
per day, with 1,889
kept frames, 30 test-window exclusions and one decode failure. This text was
introduced by commit `15f7b639a3834fe97401f035d5d0f564e795f0c1` on
2026-10-17, before the later migration/source-record claims. The immutable
version-1 summary independently contains the same kept/excluded/failed counts.
This corroborates the recorded sampling history; it is not the original command
invocation or full raw-source attestation. The source gate remains open.

The [S0 provenance inventory](../output/s0/provenance_manifest.json), dated
2026-08-22, records the recovered split SHA256 `febb3f67...f67e` and the
YOLO fold-2 OOF cache as 235 files. The actual known cache directory's 235 NPZ
file names exactly match the frozen fold-2 membership (no missing/extra names).
Its recorded cache hash mode is path-size-inventory, not a content hash. Current
sample byte reading failed with OSError22; prediction bytes were not verified.

These records strengthen historical evaluation-cohort identification. They do
not establish the original checkpoint's training inventory or exclusion of held-out
observations. No original run-bound train/validation inventory was found in the
checked module paths; this is a bounded search, not proof of global absence.
The candidate weight-to-training-split manifest remains unapproved.

The year 2011 refers to the GONG image observation timestamps, not the start
of machine-learning work. The research record and experiments discussed here
are from 2026. The builder year parameter selects the observation year.

## Reported experiment outcome (2026-10-19; acceptance pending)

The experiment report states that the kernel completed with runtime input checks (pin resolved
via Kaggle secret/env, folds SHA febb3f67, manifest 2c954685, corpus
train-test-temporal-v2 with 1,707 frames and observation digests matching the
pinned audit values).

Reported fold-2 OOF PQ at conf 0.26 (235 files, official-v1 metric):

- baseline `b1c0f353...`: 0.444228 (reported sanity comparison: approximately 0.448)
- scratch `c43b27ba...`: 0.445390
- ssl-init `c04ae6c5...`: 0.441192

Reported decision: delta ssl-minus-scratch = -0.004198 < 0 → REJECT. No
confirmation runs. GONG SimCLR encoder init is not promoted to the refiner.

Artifacts listed by the report (hash checks not independently repeated):

- `unet_encoder_ssl.pt` `bbdfdd66374c3f68...` (75 MB, 60 epochs)
- `edgeatt_scratch_fold2.pt` `c43b27ba52696c9e0c7b14b2da42e35d2f369eaa3cf35a7d7a11f3f68dfbd40a`
- `edgeatt_ssl_fold2.pt` `c04ae6c58ae37367b410c254d0d80e05b5e86145b309d012d85d51eb2d47cce8`
- `v73_eval.json` `2b0f2c50...`, `v73_summary.json` `019662e5...`,
  `ssl_encoder_history.json` `efbedcee...`

The reported negative result supports a provisional decision against promotion;
production refiner remains the existing EdgeAttNet checkpoints. This single
reported fold/seed comparison does not establish a domain-wide conclusion.


The runtime-supplied manifest pin and byte identity do not independently
approve the historical checkpoint-to-training-split binding. The existing
844-byte candidate manifest still hashes to 2c954685..., but remains unapproved.

A subsequent independent bounded local audit did not locate these new result
JSONs or checkpoints in the checked worktree/output, known local artifact,
Downloads or synchronized Drive locations. This is not proof of global absence;
their bytes and run records remain unverified by that audit.

PR #388 was subsequently merged. Original source/config acceptance, historical split binding
and independent reconciliation of the new execution records remain open.
The reported negative result does not close those provenance gates.


## Merge state and acceptance state (2026-10-19)

PR #397 merged the reported-result wording corrections into the #388 source
branch. A separate operation then merged PR #388 to main at
`fd7d134aa61ac372f5bcdc6feff6a5f4262206bc` (2026-10-19 09:00:16 UTC).
The earlier review requested Draft status because its provenance gates were
open. The recorded merge does not constitute independent approval of those
gates, the candidate manifest, new result artifacts or model performance.
Track the actual merge state separately from the still-open evidence checks.


## Recovered execution JSONs and remaining config mismatch (2026-10-19)

A later read-only retrieval from the existing COMPLETE Kaggle kernel
`shingo257/solar-filament-v73-ssl-refiner` recovered the evaluation, summary,
SSL history, both training-result JSONs and log. No new run was started.
The immutable kernel/input version IDs and executed source hashes were not
captured; the retrieval used the existing kernel slug.

The three captured JSON hashes match the previously reported prefixes:

| File | SHA256 |
|---|---|
| v73_eval.json | `2b0f2c50329afd32e65bb2174c6479495d0d9e685b3274c15a9cf8459c1b649d` |
| v73_summary.json | `019662e554d47d808e2de179d66f742212029f074171bc3bd8622a9f79c9326f` |
| ssl_encoder_history.json | `efbedceefbebc0d3e59508b6cd7fa0e54ec6c1a1d28f241fa986c14370f8ce9e` |

Summary/eval content and all twelve PQ/SQ/RQ/count relationships are consistent.
Full-precision baseline/scratch/SSL PQ values are 0.44422825344606803,
0.44539006814079485 and 0.441191736374314: SSL minus scratch is
-0.004198331766480823. This supports retaining the no-promotion decision.
It is record consistency, not an independent scorer rerun, prediction-byte
verification or rehashing of the new checkpoint bytes.

A provenance difference remains:

| Record | corpus_config SHA256 | Config fingerprint |
|---|---|---|
| Audited v3 and currently retrieved dataset config | `2a1ae10d52883432a5341d77a19345f1683374a1348736357b51819e9468d9de` | `a1f436fea3f26380` |
| Runtime SSL history/evaluation | `ebf1c6070cc19c4ba00a5b560d2dab5663bcc0fc56fa2c8359a1779682094488` | `e1e66a843bc3e2d1` |

The frame manifest, observation digests, 1,707 frames and 360-minute exclusion
agree. The runtime config original is not among the eight output files.
Known historical/current fingerprint implementations applied to audited v3
produce either c596d313f675bed4 or a1f436fea3f26380, never the runtime value.
The fingerprint covers sampling parameters and sorted train/test timestamps;
JSON spacing or line endings cannot explain this semantic difference.
The changed field or executed-code difference is not established.

Resolve the runtime config original, immutable input versions and executed code
before accepting that provenance. Historical checkpoint training-split binding,
candidate-manifest approval and full FITS/source acceptance also remain open.
The earlier bounded local/Drive searches missed these result files; retrieval
from Kaggle does not establish a synchronized Drive copy.

The retrieved log identifies Tesla T4 and 60 SSL epochs. SSL history sums to
3,745.9 seconds; reported scratch/refiner-SSL/evaluation times are
80.3/79.7/25.2 minutes. Peak VRAM and compute units are unavailable in the
retrieved records. These describe an existing run, not compute initiated by
this audit.
