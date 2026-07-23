# ML_Cancer_Detection_Smoke — MultiSmokeCancerNet

Novel cell-level lung cancer risk prediction by smoke type, from single-cell
gene expression to a subject-level cancer probability.

Given a subject's pool of scRNA-seq cells, the model:

1. Classifies which smoke type damaged each cell (cigarette, vape/e-cig,
   cigar, cannabis, dual-use, unexposed — 6 classes)
2. Scores each cell's malignancy risk (continuous, 0–1)
3. Has a head that regresses a continuous exposure dose and enforces
   monotonic dose→malignancy ordering (see `DoseResponseHead` below) — no
   existing smoke-cell model treats exposure as anything but categorical.
   **This head is architecturally complete but currently untrained on real
   data** — no public dataset with per-cell/per-sample exposure dose has
   been identified yet (see the novel-contributions table below)
4. Aggregates across all of a subject's cells via attention-based MIL to
   produce one subject-level `P(cancer)`

Full design rationale, layer-by-layer specs, data source justification, and
the novelty case vs. existing literature are in [ARCHITECTURE.md](ARCHITECTURE.md#9-novel-contributions-vs-literature).

## Where the data comes from

Every dataset below is public and free. `python3 src/data/downloaders.py --all`
fetches all of them (except NLST and TCGA, which need extra steps — see below).

| Dataset | What it is | Smoke type | Access |
|---|---|---|---|
| [GSE994](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE994) | Bronchial epithelial microarray, 75 subjects | Cigarette (active/former/never) | Free, no login |
| [GSE123352](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123352) | Lung tissue RNA-seq, 176 subjects (118 ever-smokers, 58 never-smokers) | Cigarette (ever/never) | Free, no login — Illumina probe IDs mapped to real gene symbols via GEO's own GPL10558 platform annotation file, merged into `microarray_sources` |
| [GSE136831](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE136831) | Lung scRNA-seq atlas, 312,928 real single cells | Unknown by default; COPD status available as an explicit, opt-in weak proxy — see caveat below | Free, no login — largest source (~2GB), converted via the streaming mtx parser (`src/data/converters.py::_read_mtx_streaming`), with real per-cell donor IDs from GEO's own metadata table |
| [GSE288003](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288003) | Mouse lung scRNA-seq, e-cig aerosol exposure — 23,595 real cells (10,467 unexposed control + 13,128 e-cig exposed) | Vape/e-cig (per-sample, real condition) | Free, no login — its real count matrix ships inside `RAW.tar`, which the downloader now extracts; each of the two GSM samples keeps its real exposure condition instead of a blanket label |
| [GSE307690](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307690) (CANUCK study) | Real human airway epithelial brushings, 61 samples (139 cannabis smokers + 57 never-smokers in the full published cohort) | Cannabis, dual-use, cigarette, vape, unexposed | Free, no login |
| TCGA-LUAD / TCGA-LUSC | Tumor + adjacent-normal tissue, real per-sample malignancy labels | Unknown — TCGA doesn't record verified smoking history; never defaulted | Free, but needs a personal [GDC token](https://portal.gdc.cancer.gov/) (register → profile menu → "Download Token"). Bulk RNA-seq, loaded only through the dedicated bulk_tcga path — never merged into the single-cell pipeline |
| NLST | ~26,722 subjects, 10-year cancer outcome + smoking history (cigar/dual-use labels) | — (label source, not expression data) | Requires a Data Use Agreement via [cdas.cancer.gov/nlst](https://cdas.cancer.gov/nlst/) (manual, 1–3 business days) — cannot be automated |

```bash
python3 src/data/downloaders.py --all              # fetches every free GEO source above
python3 src/data/downloaders.py --tcga --token /path/to/gdc_token.txt   # TCGA-LUAD/LUSC, needs your own token
python3 src/data/downloaders.py --nlst-instructions # prints the manual NLST steps
python3 src/data/converters.py --all               # raw downloads → clean CSV/h5ad in data/processed/converted/
python3 -c "from preprocess import run_pipeline; run_pipeline('configs/default.yaml')"
```

**A note on data honesty**: an earlier version of this project cited a
"Loiselle 2018" cannabis/tobacco dataset at accession `GSE130148`. That
dataset does not exist — `GSE130148` is a real GEO accession, but for an
unrelated human lung scRNA-seq study with no cannabis or smoke-exposure data
at all, and no dataset matching that description could be found anywhere in
GEO. It's been replaced with GSE307690 (CANUCK), a real, verified, published
cannabis-smoking cohort. If you see any dataset name in this repo you can't
verify on GEO yourself, treat it as unverified until you check.

**A note on GSE136831's smoke_type label**: this accession is the
Vanderbilt/Habermann interstitial lung disease atlas — its real per-cell
metadata (`Disease_Identity`) is COPD, IPF, or Control, not a direct
smoking-status field. Every cell's smoke label defaults to unknown
(`smoke_type_known=False`); COPD status is preserved as a separate,
explicitly opt-in weak proxy rather than being written into the primary
smoke label — see "Dataset status, label integrity, and leakage-safe
preprocessing" below for the full policy. Donor IDs and the disease label
itself are real, joined per-cell from GEO's own
`*_AllCells.Samples.CellType.MetadataTable.txt.gz` (exact barcode match,
not a prefix guess — see [converters.py](src/data/converters.py)
`_load_gse136831_cell_metadata`).

## Dataset status, label integrity, and leakage-safe preprocessing

This section tracks each data source's actual implementation status, and
the policies that keep label handling and preprocessing honest.

### Dataset status

| Dataset | Status | Notes |
|---|---|---|
| GSE994 | Implemented, verified public download | Bulk microarray, loaded as one pseudo-bulk row per sample (`is_pseudo_bulk=True`). Kept out of the default single-cell training pipeline — see "Assay separation" below. Per-sample smoking status parsed from series metadata where it matches a documented pattern, otherwise `smoke_type_known=False` rather than the accession-level default. |
| GSE123352 | Implemented, verified public download | Bulk RNA-seq, same pseudo-bulk/assay-separation treatment as GSE994. Ever/never-smoker status from series metadata under the same known/unknown parsing (not independently re-verified against GEO in this change — no network access in this environment; see `configs/datasets.yaml`). |
| GSE136831 | Implemented, verified-label default with an explicit opt-in weak proxy | Real per-cell donor IDs and disease status (COPD/IPF/Control). Every cell's smoke label defaults to unknown (`smoke_type_known=False`); COPD status is recorded as a separate, documented weak proxy (`weak_smoke_proxy_*` fields) that only feeds smoke-classification supervision when `data.weak_labels.enabled=true` — see the caveat below. |
| GSE288003 (mouse) | Implemented, species-separated | Real per-sample e-cig/control condition; excluded from the pipeline entirely unless `data.experiment_mode` is set away from the default `human_only` (see `src/data/species_policy.py`). Ortholog mapping is now a versioned, cacheable artifact (`src/data/ortholog.py`) instead of an uncached live BioMart query. |
| GSE307690 (CANUCK) | Adapter implemented; sample completeness not independently re-verified in this environment | Bulk RNA-seq pseudo-bulk rows, same assay-separation treatment as GSE994/GSE123352. See `configs/datasets.yaml`'s `known_limitations` for this entry. |
| TCGA-LUAD / TCGA-LUSC | Adapter implemented; downloading requires a personal GDC token (not present in this environment) | Bulk RNA-seq, kept out of the single-cell pipeline by construction — see "Assay separation" below. |
| NLST | Controlled-access, blocked in this environment | No participant-level file has been obtained or committed. `src/data/nlst_adapter.py` resolves a local path from `NLST_DATA_ROOT` (or `data.nlst.local_root_env`) and validates required columns; with no approved DUA in this environment, real ingestion is unavailable and the adapter says so explicitly rather than substituting a fixture. |

### Assay separation (single-cell vs. bulk)

A bulk microarray/RNA-seq sample loaded as one AnnData row ("pseudo-bulk",
`is_pseudo_bulk=True` — see `data/loaders.py::load_microarray`) is not a
cell: it has no real cell type, no per-cell malignancy signal, and no
biological meaning as one element of a subject's MIL cell bag. Earlier
versions of this pipeline only rejected rows explicitly tagged
`assay_mode="bulk_tcga"` (TCGA's own tag), which meant GSE994, GSE123352,
and GSE307690/CANUCK — pseudo-bulk but never stamped `bulk_tcga` — could
still enter the same matrix as real single-cell data (GitHub issue #13).

`src/data/assay_policy.py` is now the single, versioned gate for this,
keyed on the row-level `is_pseudo_bulk` fact rather than a source/accession
name or the `assay_mode` tag — renaming a converted file cannot bypass it.
Three experiment-level policies (`data.assay_policy` in
`configs/default.yaml`):

* **`single_cell_only`** (the default) — accepts real single-cell rows
  only; any `is_pseudo_bulk=True` row anywhere in the input is rejected.
* **`bulk_only`** — accepts pseudo-bulk/bulk rows only; rejects real
  cells. Loading and validating a bulk manifest is supported (TCGA via
  `preprocess.py::load_tcga_bulk_dataset`); there is no bulk model or
  training loop, so requesting a trainable bulk dataset raises a typed
  `BulkTrainingNotImplementedError` rather than silently reusing the
  single-cell model/loss on bulk expression.
* **`multimodal`** — not implemented. Requesting it anywhere raises a
  typed `MultimodalTrainingNotImplementedError`; there is no separate-
  encoder, modality-aware fusion architecture in this codebase, and this
  project deliberately does not approximate one by just accepting mixed
  rows in one run.

Enforcement happens at every boundary that can see a mix of rows: source
loading (`preprocess.py::_load_all_sources`), `merge_sources`,
`fit_preprocessing`/`apply_preprocessing`, `CellLevelDataset` construction,
`assemble_subject_bags`, per-fold preprocessing refits, and model-bundle
loading — each one independently rejects a policy violation, so a hand-
built or corrupted AnnData is checked the same way real pipeline output is.
`PreprocessingArtifact` records `assay_policy`, `assay_policy_version`,
`observed_assay_modes`, `pseudo_bulk_rows_present_at_fit`,
`training_data_modality`, and `allowed_inference_modality`, and all of
these are part of the artifact's scientific fingerprint — a mismatched
artifact/checkpoint/bundle assay policy is rejected, and an artifact that
predates this enforcement (`assay_policy=None`) is refused for any real
(non-synthetic) `ExperimentContext` or bundle rather than assumed safe.

`configs/default.yaml`'s `data.microarray_sources` no longer lists GSE994,
GSE123352, or GSE307690/CANUCK — they moved to a disabled-by-default
`data.bulk_sources` block, kept only as a dataset-manifest/download/
conversion record (download and conversion support is unchanged; only the
default single-cell training pipeline's source list changed). TCGA remains
under `data.tcga` (`enabled: false`), unchanged. Even if a config is
hand-edited to route a pseudo-bulk source back through
`scrna_sources`/`microarray_sources`, the row-level check still rejects it
— the config change alone was not the only enforcement mechanism.

No real bulk training exists, and no multimodal architecture exists. This
change does not add either; it only makes sure bulk data cannot silently
substitute for single-cell data in the pipeline that does exist. Model
performance figures produced before this change may have been computed
against a pipeline that accepted pseudo-bulk rows into the single-cell
matrix under a permissive config — this document does not restate those
figures as scientifically comparable to a genuinely single-cell-only run,
since that has not been separately verified.

**Missing provenance fails closed, not open.** A real (non-diagnostic) run
that reaches `fit_preprocessing`, `apply_preprocessing`, `CellLevelDataset`
construction, or `CellLevelDataset.from_dir()` without row-level
`is_pseudo_bulk` provenance raises `MissingAssayProvenanceError` rather
than defaulting the missing column to "every row is a real cell". That
default-to-safe behavior existed for a period during this module's
development and has been removed — it is not the current behavior of any
of the functions above. The only sanctioned exception is an explicit,
narrowly-scoped `diagnostic_mode=True` argument, reserved for deliberately
synthetic fixtures (unit tests, `--synthetic` CLI runs); it is never
inferred from a source name, a file path, or the absence of real data, and
a diagnostic-mode dataset is rejected by `ExperimentContext` validation and
by every real bundle/report path. `CellLevelDataset.from_dir()` additionally
refuses to load a legacy exported directory (missing
`cell_metadata.csv` or its `subject_id`/`source`/`is_pseudo_bulk`
columns) with a typed `LegacyCellDatasetDirectoryError` — such a directory
must be regenerated with the current pipeline, not loaded with relaxed
assumptions.

**Boolean provenance is parsed strictly.** `data/assay_policy.py::
parse_strict_bool_array` is the one parser used for persisted/user-provided
`is_pseudo_bulk` values throughout the codebase (loaders, `from_dir()`,
bundle-input validation). It accepts real booleans and the literal strings
`"true"/"True"/"1"` / `"false"/"False"/"0"`; it rejects `NaN`, `None`, empty
strings, and any other value outright — `bool("False")` evaluating to
`True` is exactly the kind of silent misparse this project does not rely
on `astype(bool)`/`bool(x)` to avoid.

**Unsupported training modes are blocked before fitting, not just at the
policy helper.** `data.assay_policy.require_trainable()` is called at
every production path that can reach a trainable result or a constructed
model: `preprocess.py::run_pipeline`/`run_pipeline_split_aware` (before any
source is loaded), `ExperimentContext.from_pipeline_result` (the shared
validation every CV/OOF/final-fit/source-held-out path builds on),
`Trainer.from_config`, `Trainer.from_experiment_context`, and the entry
points of `run_smoke_cv`/`run_cancer_cv`,
`generate_subject_oof_predictions`, `fit_final_candidate_on_dev_pool`, and
both source-held-out functions. `assay_policy='bulk_only'` raises
`BulkTrainingNotImplementedError` and `'multimodal'` raises
`MultimodalTrainingNotImplementedError` before any model, optimizer, or
preprocessing fit is constructed — loading/validating a bulk manifest
(`load_tcga_bulk_dataset`) remains available separately and is unaffected.

**Legacy datasets/artifacts/bundles must be regenerated, not patched
around.** An exported cell-dataset directory, a `PreprocessingArtifact`, or
a model bundle produced before this enforcement existed has no reliable
way to retroactively prove its row-level provenance was correct, so each
of those loaders refuses to guess — see `LegacyCellDatasetDirectoryError`
above and `assert_real_assay_provenance()` for the artifact/bundle case.

### Label integrity

- A missing label is never converted into a negative label. `malignancy_known`
  and `cancer_label_known` (already existed pre-this-change) gate which cells/
  subjects contribute to their respective supervised losses and metrics —
  see `data/labellers.py::add_malignancy_labels` and
  `data/assembly.py::assemble_subject_bags`. `src/data/label_state.py` adds a
  shared five-state vocabulary (known positive / known negative / unknown /
  not applicable / excluded by policy) for any new label-producing code to
  use instead of inventing another ad hoc pair of columns.
- **COPD proxy policy (GSE136831)**: `Disease_Identity=COPD` is a weak proxy
  for cigarette exposure, not a verified smoking record — this dataset is an
  interstitial lung disease atlas, not a smoking cohort. Every GSE136831
  cell's converter output carries `smoke_type_name="unknown"` and
  `smoke_type_known=False` by default (see
  `data/converters.py::_load_gse136831_cell_metadata`); COPD-diagnosed
  donors additionally carry `weak_smoke_proxy_known=True`,
  `weak_smoke_proxy_type="COPD_diagnosis"`,
  `weak_smoke_proxy_value="cigarette"`, and a `weak_smoke_proxy_limitation`
  string. `smoke_type_known=False` cells are excluded from smoke-
  classification loss, class-weighting, subject-balanced sampling
  distributions, stratification, and evaluation metrics (see
  `train.CellLevelDataset.smoke_class_weights`,
  `model.MultiTaskLoss._ls`, `data.sampling.SubjectClassIndex`,
  `evaluate.Evaluator._known_smoke_metrics`) — a corrupted or arbitrary
  placeholder value for these cells cannot change any of those outputs,
  since they're masked out before the placeholder is ever read.
  `data/labellers.py::apply_weak_smoke_proxies` is the only way COPD's
  weak proxy is ever promoted into the primary smoke label, and it only
  runs when `data.weak_labels.enabled=true` (default `false`) — this is a
  disclosed, non-default experiment, never silent.
- **TCGA smoking/bulk policy**: TCGA smoke type is never defaulted to
  cigarette — `data/converters.py::convert_tcga` writes
  `smoke_type="unknown"`/`smoke_type_known=False` for every sample. TCGA is
  primarily bulk RNA-seq: `configs/default.yaml`'s `microarray_sources` no
  longer lists TCGA-LUAD/TCGA-LUSC at all, and every TCGA sample carries
  `is_pseudo_bulk=True` and `assay_mode="bulk_tcga"`, both of which
  `preprocess.py::_load_all_sources` refuses to load into the human
  single-cell pipeline even if a config is misconfigured to reference it —
  see "Assay separation" above for the general row-level mechanism and
  `preprocess.py::load_tcga_bulk_dataset` for the dedicated bulk
  loading/validation path, which stays disabled by default
  (`data.tcga.enabled=false`) and raises `BulkTrainingNotImplementedError`
  if asked to produce a trainable bulk dataset (no bulk model exists in
  this project). TCGA's tumor/solid-tissue-normal `sample_type` remains a
  bulk-sample-level malignancy label, reachable only through that same
  bulk-only path — it is never merged into a single-cell dataset's
  per-cell malignancy field.

### Human/mouse separation

`data.experiment_mode` (default `human_only`) gates whether any mouse data
is loaded at all — see `src/data/species_policy.py`. Mouse subject/animal
IDs are namespaced (`mouse::<id>`) so they can never collide with a human
subject_id. `data/assembly.py::merge_sources` refuses to concatenate sources
spanning more than one species unless `experiment_mode` is explicitly
`cross_species_pretraining` or `cross_species_domain_adaptation` — both
exist only as safe hooks in this change (disabled by default); a real
domain-adaptation training loop is not implemented.

### Ortholog mapping

`src/data/ortholog.py` resolves mouse→human gene pairs through an explicit,
versioned `OrthologMappingArtifact` — ambiguous cases (one mouse gene with
several human candidates, several mouse genes mapping to the same human
gene) are dropped under the default `one_to_one_only` policy rather than
resolved by picking the first match. The artifact can be cached to disk and
reloaded (`artifact_path=`) so tests and CI never need a live BioMart query.

### Split-before-preprocessing, train-only HVG/scaling

Unchanged from the existing pipeline (already implemented before this
change; see `src/preprocess.py::run_pipeline_split_aware` and
`src/data/preprocessing.py`): the subject-level split is computed before
`fit_preprocessing` ever runs, and gene selection/scaling are fit on the
training partition's cells only. `tests/test_leakage_regression.py`
consolidates the isolation regression tests for this and the additions
above into one auditable file.

### Preprocessing artifacts and reproducibility

`fit_preprocessing`/`apply_preprocessing` (`src/data/preprocessing.py`)
already separated fitting (train subjects only) from applying (val/test/
inference, unchanged). On top of that, `PreprocessingArtifact` now carries:

* **A scientific fingerprint** (`artifact.scientific_fingerprint()`): a
  SHA-256 over the artifact's gene list, scaling statistics, HVG/label/
  cell-type provenance, and gene-contract policy — deliberately excluding
  `created_at` and free-text notes. Two artifacts fit from the same
  training subjects, config, and code produce the same fingerprint
  regardless of output directory or wall-clock time; changing any
  scientifically meaningful input changes it.
* **An explicit gene contract**: `missing_gene_policy` (`error` by
  default), `duplicate_gene_policy` (`error`, unconditionally — no
  aggregation policy is implemented), `unexpected_gene_policy` (`ignore`
  by default — recorded in `apply_preprocessing`'s output
  `uns["preprocessing_compatibility_diagnostics"]`, never fatal on its
  own), and `minimum_gene_coverage` (`1.0` by default). These match
  `configs/default.yaml`'s `preprocessing.*` keys
  (`tests/test_preprocessing_artifact_contract.py::test_config_defaults_match_code_defaults`
  fails the build if they drift). `zero_fill` for missing genes is
  supported but never the default — it must be requested explicitly, is
  recorded in the artifact and in the per-call diagnostics, and is never
  presented as observed expression.
* **A checkpoint-artifact binding**: `Trainer._save()` embeds the
  fingerprint of whichever `PreprocessingArtifact` the run was actually
  wired with (`Trainer.set_preprocessing_artifact`) into the checkpoint.
  `Predictor.from_config()` recomputes the fingerprint of whatever
  `preprocessing_artifact.json` sits next to that checkpoint and refuses to
  pair them (`ArtifactCompatibilityError`) if they disagree — this closes
  the gap where a stale or swapped artifact file could silently be treated
  as compatible with an unrelated checkpoint.
* **Safe, atomic serialization**: `PreprocessingArtifact.save()` writes
  through the same temp-file-plus-`os.replace()` atomic path the rest of
  this project's reporting/guard files use
  (`src/benchmarks/atomic_io.py`), so a reader never observes a partially
  written artifact. `.load()` rejects unparsable JSON, an unrecognized
  schema version, and a payload that doesn't match the current
  `PreprocessingArtifact` fields, all with a specific exception type
  (`PreprocessingArtifactError`, `GeneContractError`,
  `ArtifactCompatibilityError`, `LegacyArtifactError` — see
  `src/data/preprocessing.py`) rather than an opaque failure.
* **A read-only inspection report**: `artifact.inspect()` returns schema
  version, fingerprint, selected gene count, gene-contract policy, cell-type
  annotation provenance, and creation metadata as a plain dict — never the
  raw scaling arrays or participant-level data — suitable for logging or
  attaching to a run's report.

This closes gaps that existed even though the underlying leakage-safe
fit/apply split was already correct: previously, `missing_gene_policy` was
declared in config but never actually read by `apply_preprocessing`, and a
checkpoint recorded only a (never-populated) artifact *path*, not the
artifact's actual fitted content — so a checkpoint could silently be paired
with any file that happened to sit at that path. The frozen-test guard
(`src/benchmarks/test_guard.py`, already a durable, atomically-acquired
one-time lock) was audited and is unchanged — it already met the bar this
section describes.

### Per-fold preprocessing artifacts

Every CV fold already refit its own `PreprocessingArtifact` from only that
fold's training subjects (`src/benchmarks/fold_preprocessing.py`,
`refit_artifact_for_fold`) — that part predates this change. What was
missing was persistence: a fold's artifact lived only in memory for the
duration of the run. `save_fold_artifact()`/`load_fold_artifact()` now
persist each fold's artifact to
`<output_root>/preprocessing/fold_XX/{artifact.json,manifest.json}`
(`run_smoke_cv`/`run_cancer_cv`'s optional `artifact_output_root` parameter
wires this in; `runner.py`'s benchmark CLI always passes the run's own
output directory). `manifest.json` is written last, after `artifact.json`
is fully and atomically on disk, and records the fold's train/val-subject
fingerprints and its own artifact fingerprint — `load_fold_artifact()`
verifies all of this before returning anything, rejects an
incomplete/interrupted persist (`IncompleteFoldArtifactError`), and rejects
a resume attempt whose expected subjects don't match what was actually
persisted (`FoldArtifactMismatchError`) — so one fold's persisted artifact
can never be silently reused for a different fold.

### Batch correction

`preprocessing.batch_correction.mode` is `none` by default (no Harmony run
at all in the leakage-free path). `transductive_diagnostic_only` is the
named opt-in for running Harmony across the full train+val+test dataset —
this is explicitly disclosed as non-leakage-free and must not be used for
frozen-test evaluation. `train_fitted_inductive` is accepted as a config
value but currently always raises: Harmony has no train-only-fit /
apply-to-new-data transform, so there is no inductive implementation behind
that name yet.

A run that opted into `transductive_diagnostic_only` now carries that fact
on `ExperimentContext.transductive_batch_correction` all the way through:
`assert_batch_correction_safe()` (`src/data/transforms.py`) is called at
the entry point of every leakage-free protocol — grouped CV/OOF fold
refitting, the final development-pool fit, and immediately before the
frozen-test guard is acquired — and raises `UnsafeBatchCorrectionError`
if that flag is set, rather than letting a transductively-corrected run
quietly reach any of them. `PreprocessingArtifact.batch_correction_status`
(`"disabled"` or `"transductive_diagnostic_only"`) records which state a
given artifact's run was in and is part of its scientific fingerprint.
There remains no inductive (train-only-fit, apply-to-new-data) batch
correction implementation in this project — Harmony is transductive by
construction, and this change does not claim otherwise; it only makes the
transductive state impossible to smuggle into a leakage-free protocol.

### Model bundles

A deployable bundle (`src/benchmarks/bundle.py`) is a directory containing
a model checkpoint, a copy of the `PreprocessingArtifact` it was trained
with, and one `bundle_manifest.json` recording: model configuration, class
vocabulary, label policy, species policy, assay mode, dataset-manifest and
split fingerprints, calibration state and decision threshold (when
applicable), an environment snapshot, and a SHA-256 for every referenced
file plus one `bundle_fingerprint` covering the whole manifest.
`Trainer.write_bundle()` builds one from an already-saved checkpoint and
whatever `PreprocessingArtifact` was wired in via
`set_preprocessing_artifact()`. `load_and_validate_bundle()` re-hashes and
re-derives every one of those fields on load and raises a specific error
(`BundleCorruptionError` for a missing/unparsable/wrong-schema manifest,
`BundleValidationError` for any hash/fingerprint/gene-count mismatch) the
instant anything doesn't match — including another fold's artifact being
substituted in, or the gene list being altered after the fact.
`validate_bundle_for_model()` additionally checks a constructed model's
`input_dim`/`num_smoke` against the bundle's own artifact/class vocabulary.
A directory with a checkpoint but no `bundle_manifest.json` at all (a
legacy, pre-Phase-4 checkpoint) raises `LegacyBundleError` unless the
caller passes `allow_legacy=True` explicitly — and even then, no bundle
validation runs at all, so a legacy checkpoint loaded this way is never
described as bundle-verified.

### Frozen-test access sentinels

`src/benchmarks/sentinel.py`'s `FrozenAccessSentinel` is a stand-in object
that raises `FrozenDataAccessError` on essentially any attempt to read it —
attribute access, iteration, indexing, `len()`, array/DataFrame conversion,
even `repr()`. It complements (does not replace) the durable
`FrozenTestGuard`: the guard controls *when* the one sanctioned
test-touching function may run at all; a sentinel lets a test wrap real
test data and run the *entire* development pipeline (CV, OOF generation,
the final development-pool fit) against it, turning "the dev pipeline never
touches test data" from a code-review claim into something that fails
loudly, with a traceback pointing at the exact call site, if it's ever
violated.

### Environment snapshot

`benchmarks/reporting.py::write_environment_artifact` records Python
version, platform, installed versions of every scientific dependency this
project's behavior depends on (numpy, pandas, scikit-learn, torch, scanpy,
anndata, scipy, ...), CUDA availability/version (when torch is installed),
whether `torch.use_deterministic_algorithms` is enabled, the run's random
seed, its config fingerprint, and the git commit SHA — written atomically
as `environment.json`, referenced from `Trainer.write_bundle()`'s model
bundles. It never records a username, hostname, environment-variable dump,
or absolute local path, and recording a seed is not a claim of exact
cross-machine numerical determinism (see Limitations).

### Artifact and bundle CLI

`python3 src/inference.py --inspect-artifact path/to/artifact.json` prints
a read-only JSON summary of a `PreprocessingArtifact` (schema version,
fingerprint, gene count, gene-contract policy, provenance) — no checkpoint
or model needed. `--validate-artifact path/to/artifact.json` loads it,
runs its own integrity checks, and prints `OK`/exits 0 on success or prints
the error/exits 1 on failure. `--allow-legacy-checkpoint` is the explicit,
never-default opt-in to load a checkpoint with no
`preprocessing_artifact.json` next to it (maps to
`Predictor.from_config`'s `unsafe_legacy_mode`).

### Dataset manifest

`configs/datasets.yaml` is the checked-in provenance seed (accession, URLs,
species, identifier fields, documented limitations) for every dataset above.
`src/data/manifest.py` builds the full manifest, computing real SHA-256
checksums only for files actually present locally — an entry for a dataset
with no local files yet still validates, with `files_present=false` and
every checksum explicitly `null`, never a fabricated placeholder.

### NLST smoking-history and cancer-outcome handling

`data/nlst_smoking.py` parses NLST's `CIGSMOK`/`CIGAR` fields against only
the codes this repository's own reviewed access instructions document
(`CIGSMOK` 1=current/2=former smoker, `CIGAR` 1=yes) — any other value
(missing, blank, null, an undocumented code such as `0`, or a malformed
entry) stays an unknown smoking history rather than being coerced into
cigarette, cigar, or "unexposed". `data/labellers.py::transfer_nlst_labels`
uses this parser for every matched subject, so NLST participation alone is
never treated as cigarette exposure — the subject's own CIGSMOK/CIGAR
values have to actually parse to a documented code. `smoke_type_known`
gates loss/class-weighting/sampling/stratification/metrics the same way it
does everywhere else in this pipeline. `data/converters.py::
convert_nlst_outcomes` similarly excludes a subject with a missing `candx`
value from the outcomes CSV rather than writing `cancer_label=0` for them.

The same missing-value-must-not-become-a-verified-label rule applies to
GSE994/GSE123352 (`data/converters.py::_infer_smoke_column`) and GSE307690/
CANUCK (`convert_canuck`): a sample whose GEO characteristics don't parse
to a documented smoking-status pattern is written as
`smoke_type_known=False`, never silently defaulted to the accession-level
label declared in `configs/default.yaml`.

## Current results (real data)

As of the last real run (not synthetic), with GSE994 + GSE307690 merged and
harmonized to a common gene-symbol space (136 real samples total):

| Metric | Value |
|---|---|
| Smoke-type accuracy | 77.2% |
| Smoke-type **macro-F1** | **0.27** |
| Per-class F1 | cigarette 0.95, dual_use 0.66, vape/cigar/cannabis/unexposed **0.0** |
| Subject-level cancer `P(cancer)` | **not evaluable yet** — see below |

**Read this honestly, not optimistically.** 77% accuracy sounds good; it
isn't. The model has collapsed onto the two majority classes (cigarette=83
samples, dual_use=30 samples) and has learned nothing for vape (7), cannabis
(6), cigar (1), or unexposed (10) samples — each too small to learn from.
Macro-F1 (0.27) is the metric that reflects this. **This table is also a
training-set evaluation, not held-out performance** — it predates the
subject-level splitting work below, so the model was scored on data it was
trained on. It is left in place, unaltered, as an honest record of what was
actually measured at the time; it is not being re-labelled as validation or
test performance retroactively, and no new real-data experiment has been run
against a held-out split as of this update (see
[Scientific rigor and known limitations](#scientific-rigor-and-known-limitations)).
**There is no subject-level cancer accuracy at all yet**: GSE994/GSE307690
are bulk RNA-seq (one expression vector per subject, not per cell), so the
MIL attention aggregator — which needs many cells per subject to attend
over — has zero usable bags.

**What would change this**: real per-cell, per-subject data with hundreds of
cells per subject. That means GSE136831 (312,928 real cells) or TCGA
(blocked until a GDC token is supplied). Both are prerequisites for any
meaningful subject-level cancer prediction number; nothing before that point
is a real model-quality result.

## Scientific rigor and known limitations

This section documents a set of correctness/leakage fixes made on top of the
original pipeline, on branch `improve/valid-evaluation-and-training`, and is
intentionally blunt about what is and isn't resolved.

**Why macro-F1, not accuracy, is the primary smoke-type metric.** The 77.2%
accuracy number above is a textbook example of why: a model that always
predicts "cigarette" on this data would score close to that accuracy while
having zero ability to distinguish any other class. Macro-F1 weights every
class equally regardless of its size, so it can't be inflated by collapsing
onto the majority class — see `train.py`'s Phase 1 checkpoint selection and
`evaluate.py::_smoke_metrics`.

**Subject-level splitting is now real and unavoidable end-to-end**
(`src/data/splitting.py`, `src/train.py`, `src/preprocess.py`). Every prior
result in this repo — including the 77.2%/0.27 table above — was computed by
training and evaluating on the same cells, with no subject held out at all.
A subject's cells share genetic background and batch/technical variation, so
even a naive train/test split *by cell* would leak: a model that's seen 80%
of a subject's cells trivially recognizes the other 20%. `splitting.py`
provides `subject_train_val_test_split()` and `grouped_kfold()`, both
grouped by `subject_id`, with reproducible JSON manifests
(`data/processed/split_manifest.json` by default, see `configs/default.yaml`'s
`split:` block) and a SHA-256 dataset fingerprint (subjects + effective
labels + split config) — `load_or_create_split()` now **raises** rather than
silently reusing or regenerating a manifest if the current data/config no
longer matches it (a subject added/removed, a label changed, a different
seed/fraction/rare-class policy); pass `force_regenerate=True` to
deliberately discard it.

Earlier, `Trainer` computed the split itself: `phase1`/`phase2`/`phase3`
called `random_split()` internally on whatever dataset was handed to them,
so a subject's cells could still land in both the internal "train" and
"validation" partition even though `splitting.py` existed. This is fixed:
`Trainer.phase1/phase2/phase3` now take **explicit, pre-split**
`train_*_dataset`/`val_*_dataset` arguments and never split anything
internally; `CellLevelDataset` carries a required `subject_ids` array (real
training data must supply real, non-"unknown" subject IDs — a
`diagnostic_mode=True` escape hatch exists only for synthetic smoke tests)
plus a `subset_by_subjects()` method and a module-level
`assert_disjoint_subjects()` guard that every phase calls before training
anything. `preprocess.py::run_pipeline_split_aware()` returns explicit
`train_cell_dataset`/`val_cell_dataset`/`test_cell_dataset` and
`train_bags`/`val_bags`/`test_bags` built directly from the split manifest,
so a caller never has to manually filter one combined array (the
old `cell_data`/`bags` keys are kept for backward compatibility only). A
dedicated `Trainer.final_test_evaluation()` is the one sanctioned place test
data is used — after checkpoint selection is done, evaluated once, and
labelled `is_held_out=True` in its output so it can't be confused with a
validation or training-set number. **No model in this repo has yet been
retrained on real data against one of these splits** — the plumbing is in
place and tested (leakage-guard tests included); running it against the
real merged dataset is a real training run, out of scope for this pass (see
[Next steps](#next-steps)).

**Unknown cancer outcomes are no longer treated as negative.**
`assemble_subject_bags()` used to do `outcome_map.get(str(sid), 0)` — any
subject never matched to an NLST/TCGA outcome silently became a fabricated
cancer-negative. It now sets `cancer_label=None` /
`cancer_label_known=False` for those subjects, logs known-positive/known-
negative/unknown counts, and `SubjectLevelDataset` excludes unknown-outcome
subjects from supervised training/evaluation by default. `train.py`'s new
`check_mil_eligibility()` also refuses to run Phase 2/3 MIL training/eval
when there aren't enough independent subjects or both outcome classes
represented, instead of silently producing a meaningless AUC.

**Unknown malignancy labels are no longer treated as verified-normal.**
`add_malignancy_labels()` stamps cells with no real label with a 0.0
placeholder so the array always has a value — but that placeholder was being
used directly as a BCE training target, teaching the model "benign unless
proven otherwise" from data nobody actually labelled. A `malignancy_known`
provenance column (real for `tumor_barcodes` matches and loader-set per-
sample labels like TCGA tumor/NAT, false otherwise) now masks the
malignancy loss to known cells only, and `evaluate.py` restricts malignancy
metrics to known cells and reports known-positive/known-negative/unknown
counts instead of an AUC partly computed against fabricated labels.

**Preprocessing leakage fix, and label order corrected.**
`merge_sources()` z-scored, and `smoke_aware_hvg()` selected highly-variable
genes across, the *entire* merged dataset — before any split existed, so
validation/test cells influenced which genes became features and how they
were scaled. `src/data/preprocessing.py` adds a `PreprocessingArtifact`
(versioned, JSON-serialisable) fit on train-split cells only
(`fit_preprocessing`/`apply_preprocessing`), automatically persisted next to
the split manifest AND the training checkpoint directory so inference can
find it. `preprocess.py::run_pipeline_split_aware()`'s step order is now:
load sources → attach final smoke labels (including NLST transfer) and
malignancy provenance → apply the rare-class policy to that final label →
compute the subject-level split **on the final effective label** → fit
scaling/HVG on train only → apply to everyone. Previously the split was
computed *before* NLST label transfer, so the split (and its report) could
reflect a label that was about to change. The original label is preserved
unmutated as `obs["smoke_type_raw"]` (exported in `cell_metadata.csv`); the
NLST-join count (or its absence) is logged in `label_provenance_report`
rather than only printed. The original `run_pipeline()` is unchanged and
still has this leakage — it's kept only for backward compatibility and
synthetic smoke-testing, with an explicit warning in its docstring.
**Batch correction (Harmony) is a documented, now-configurable exception
that is *not* leakage-free**: Harmony has no train-only-fit /
apply-to-new-data mode, so running it uses held-out expression values to
compute the correction embedding — a transductive step. `run_pipeline_split_aware()`
now **skips Harmony by default** (`preprocessing.batch_correction.allow_transductive_harmony: false`
in `configs/default.yaml`); enabling it requires an explicit opt-in, prints a
prominent warning, and the returned `transductive_batch_correction` flag
records the fact so it can be carried into evaluation/checkpoint metadata.

**Rare smoke-type classes are now actually wired in**
(`src/data/rare_class.py`). The cigar class has ~1 independent subject in
the real merged data — not enough to learn or evaluate as its own class by
any reasonable statistical standard. `apply_rare_class_policy()` existed as
a tested utility but nothing called it; `run_pipeline_split_aware()` now
applies the configured policy (`keep_with_warning` /
`merge_into_dual_use_or_other` / `exclude_from_training_and_evaluation`, set
via `configs/default.yaml`'s `rare_class:` block) to the final label *before*
the subject-level split, so the effective label the policy produces is what
splitting, training, evaluation, and bag assembly all consistently see. Raw
labels are never mutated — `smoke_type_raw` always preserves the original —
and every action is recorded in the returned `rare_class_report`.

**Structured, reproducible checkpoints.** `Trainer._save()` used to write a
bare `model.state_dict()`. It now saves a structured checkpoint containing
the state dict plus model/training config, split-manifest path, preprocessing-
artifact path, effective label mapping, rare-class policy, random seed,
metric name/value, epoch/phase, input dimension, and git commit SHA (when
available) — the metadata needed to know what a checkpoint actually is
without re-deriving it. A shared `train.load_checkpoint_into()` loads both
this format and legacy bare-state-dict checkpoints (with a printed warning
for the latter); `Evaluator.from_checkpoint()`, `Predictor.from_config()`,
and `Trainer._load_best()` all use it.

**Safe H5AD inference.** `Predictor.predict_h5ad()` used to call
`verify_compatible()` (checks required genes exist) and then forward the
*original, unreordered, unscaled* matrix — silently wrong for any input
whose gene order didn't already exactly match the artifact. It now applies
`data/preprocessing.py::apply_preprocessing()` (reorder → subset → train-fit
scale) for input explicitly declared `input_stage="normalized_expression"`,
verifies exact gene-order equality via `verify_input_matrix()` for
`input_stage="model_ready"`, verifies the final width against
`model.input_dim`, and refuses to run input with no `preprocessing_artifact`
unless `unsafe_legacy_mode=True` is explicitly set.

**Honest input-stage contract — `already_preprocessed: bool` replaced.**
The old boolean's `False` branch was documented as accepting "RAW/unprocessed
expression" but only ever reordered/subset genes and applied train-fit
scaling — it never reproduced QC, library-size normalization, or
log-transformation, so genuinely raw counts silently produced invalid
predictions. `PreprocessingArtifact` does not store QC thresholds, a
library-size target, or log-transform parameters, so this repository cannot
honestly claim to support raw counts at inference time. `predict_h5ad()` now
takes an explicit `input_stage` argument instead:
`"model_ready"` (exact gene order/scaling match required, no transform
applied, artifact-version and finite-value checked — was `already_preprocessed=True`),
`"normalized_expression"` (already QC'd/normalized/log-transformed
upstream, same as training's `data/transforms.py::normalize` — reorder/
subset/scale via the artifact only, duplicate genes and non-finite values
rejected — was `already_preprocessed=False`), or `"raw_counts"`, which is
now **always rejected** with an explicit "not supported" error rather than
silently mishandled. `already_preprocessed` is kept only as a deprecated
alias (`DeprecationWarning`, unambiguous mapping to the two supported
stages — never to `"raw_counts"`). `PreprocessingArtifact` gained an
`expected_input_stage` field, validated against the caller's declared stage.

**Full cross-task leakage validation.** The per-phase disjointness checks
above only compared same-modality datasets (train cells vs. val cells, train
bags vs. val bags) — a subject whose *cells* landed in train but whose *bag*
landed in val (or vice versa) went undetected, and Phase 3 uses all four
datasets together. `train.py::validate_experiment_partitions()` unions each
split's cell-subject-ids and bag-subject-ids and requires the per-split
unions to be pairwise disjoint (a subject appearing in both its own split's
cell dataset and that same split's bag dataset is fine; only cross-split
overlap raises), and `Trainer.phase3` now calls it in addition to the
same-modality checks.

**Held-out test evaluation is now enforced, not just documented.**
`Trainer.final_test_evaluation()` previously relied on its docstring saying
"call this once." It now tracks every subject seen by `_validate_train_val`
across all phases on that `Trainer` instance and raises `ValueError` if any
test subject was already used for train or validation; a second call raises
`RuntimeError` by default (`allow_repeat=True` is required to deliberately
re-run, and the resulting report is marked `is_pristine: False`). The report
is written to its own `heldout_test_report.json` /
`heldout_test_predictions.json` (separate from `evaluation_report.json`,
which can also be produced from train/val data elsewhere) with explicit
provenance: split name, held-out flag, checkpoint id, split-manifest path,
threshold source, UTC timestamp, and run count.

**Consistent macro-F1 between checkpoint selection and reporting.**
`Trainer.phase1`'s checkpoint-selection metric used to call
`f1_score(..., average="macro")` with no explicit label list — sklearn
restricts averaging to classes *observed in that call* when `labels=None`,
so a validation batch missing one smoke class silently computed macro-F1
over 5 classes instead of 6, while `evaluate.py::_smoke_metrics` (which
already passed an explicit label list) would report a different number for
the same checkpoint. `src/metrics.py::multiclass_f1_report()` is now the one
place "smoke-type macro-F1" is defined — explicit full class list, classes
absent from the target set recorded and the result flagged `is_partial` —
and both `Trainer.phase1` and `evaluate.py::_smoke_metrics` call it.

**Cell-type IDs are now validated before every forward pass.**
`inference._validate_h5ad()` used to check only that the cell-type column
existed. `src/metrics.py::validate_cell_type_ids()` now checks length,
rejects NaN/fractional values, and enforces `0 <= id < num_cell_types`,
naming the actual bad values and the expected range; it's called by
`Predictor.predict_subject()`, `Predictor.predict_h5ad()`, and
`Trainer.predict()`.

**Grouped K-fold no longer crashes on small class counts.**
`grouped_kfold()`'s per-class fold assignment reset its index to 0 for every
class bucket, so e.g. two subjects in two different classes could both be
assigned fold 0 — leaving one fold with an empty training set and another
with an empty validation set, hitting an `assert`. Fold assignment now uses
one cursor shared across all class buckets, and the internal `assert`s were
replaced with explicit `ValueError`/`RuntimeError` (data-dependent failures
should never be silenced by `python -O`).

**Effective contiguous smoke-label space.** A rare-class policy that merged
cigar into dual_use, or excluded it entirely, changed which raw labels were
*used* — but `model.num_smoke`, macro-F1's class count, the confusion
matrix, and every displayed class name stayed fixed at 6, leaving a dead
output neuron and a permanent zero-support row. `src/data/label_mapping.py`
adds `EffectiveLabelMapping`: built once, deterministically, directly from
`apply_rare_class_policy()`'s report (never inferred from which classes
happen to appear in one evaluation split) — a contiguous `0..K-1` space that
drops merged-away/excluded raw ids. `run_pipeline_split_aware()` transforms
`obs["smoke_type"]` into this contiguous space before it reaches the split,
the exported cell dataset, or the bags, and persists the mapping in
`PreprocessingArtifact.label_mapping`. `MultiSmokeCancerNet.from_config()`
accepts a `num_smoke_types` override so the model's actual output width
equals `K`, not a config default. `Trainer.set_label_mapping()` validates
`mapping.k == model.num_smoke` before wiring it in (raises on mismatch);
`Trainer.phase1`'s checkpoint-selection macro-F1 and `train_cell_dataset.
smoke_class_weights()` now use `self.model.num_smoke`, never the fixed
6-class constant. `Evaluator.from_checkpoint()` and `Predictor.from_config()`
both peek at a checkpoint's metadata (`train.read_checkpoint_metadata()`)
*before* constructing the model, so they build it with the checkpoint's
actual `K` up front instead of failing with an opaque shape-mismatch error
inside `load_state_dict()` — and if a loaded `PreprocessingArtifact` also
carries a `label_mapping`, the two are cross-validated
(`EffectiveLabelMapping.validate_compatible()`), raising a clear error if a
checkpoint and an artifact came from different rare-class-policy runs.
Raw labels are preserved unmutated (`smoke_type_raw`) throughout.

**What this pass does *not* include** (explicitly out of scope, not
silently skipped): a real training run against a held-out split (so there
is no new "real held-out macro-F1" number to report — see the table above);
the full raw-count preprocessing chain reproduced at inference time (species
validation, gene-ID harmonization, library-size normalization, and log
transform from genuine raw counts) — `predict_h5ad(input_stage="raw_counts")`
is explicitly rejected rather than silently mishandled, since
`PreprocessingArtifact` doesn't store the QC/normalization parameters needed
to reproduce that chain; only `"model_ready"` and `"normalized_expression"`
are supported (see above); an `ExperimentContext`/`Trainer.from_experiment_data()`
that wires pipeline output into a Trainer automatically (metadata fields like
`split_manifest_path` are still set manually after construction); checkpoint
checksum verification and full resume support (optimizer/scheduler/
early-stopper state is not yet saved or restorable); a baseline-model
comparison runner (logistic regression / random forest / XGBoost / small MLP
vs. the neural model); a grouped-cross-validation experiment *runner* (the
primitive `splitting.py::grouped_kfold` is leakage-tested, including the
small-class-count fix above, but no training loop consumes it yet); a
hyperparameter-search runner; MIL pooling-baseline comparisons (mean/max
pooling vs. gated attention) or attention-stability analysis (subject-aware
class -> subject -> cell sampling, with a per-batch max-cells-per-subject
cap, is now implemented — see "Phase 2 — subject-aware class-imbalance
correction" above; this list predates that work and is retained for
historical context on the other still-open items); explicit bulk-vs-single-cell-vs-MIL
experiment-mode separation; species-provenance / cross-species-merge guards
(GSE288003's mouse→human ortholog mapping still runs unconditionally, with
no recorded mapped/unmapped gene counts); probability calibration /
validation-selected threshold tooling (the model
still reports risk at a fixed 0.70 cutoff in `train.py::predict` and
`inference.py`, which is **not** a clinically validated threshold — treat it
as an arbitrary placeholder; `final_test_evaluation`'s provenance honestly
records `threshold_source: "default_0.50"` rather than claiming a
validation-selected threshold that doesn't exist); dose-response head
supervision gating (the head still runs unconditionally regardless of how
many real dose labels are present). Each of these is a legitimate,
separately-scoped follow-up, not an oversight.

**No clinical claim.** Nothing in this repository has been clinically
validated. `P(cancer)` is a research-model output on unlabelled or
approximately-labelled data; it is not a diagnostic and should not be
treated as one.

## Novel contributions vs. literature

Every claim below is backed by working code in this repo, not just design intent:

| Claim | Closest existing paper | Gap | Implemented at |
|---|---|---|---|
| Multi-smoke-type cell classifier (6 types) | Ma et al. 2024 (cigarette only, 3 states) | No vape/cigar/cannabis/dual-use | [constants.py:15](src/constants.py#L15) `SMOKE_TYPES`, [model.py:74](src/model.py#L74) `SmokeTypeHead` |
| Per-cell malignancy risk score | Long et al. 2024 (susceptibility genes only) | Not a predictive model | [model.py:96](src/model.py#L96) `MalignancyHead` |
| MIL aggregation from scRNA-seq to subject | Used in WSI histopathology (ABMIL 2018) | Never applied to scRNA-seq bags | [model.py:145](src/model.py#L145) `GatedAttentionMIL` |
| Cannabis lung cell cancer model | CDC acknowledges gap officially (2024) | Does not exist anywhere as a per-cell/per-sample ML model | [converters.py:146](src/data/converters.py#L146) `convert_canuck` — real data: GSE307690 (CANUCK study, 61 human airway epithelium samples, 139-cannabis-smoker cohort), wired via [default.yaml](configs/default.yaml) `microarray_sources` |
| Dual-use cellular signature | Bittoni et al. 2024 (epidemiology only) | No cell-level ML model | [labellers.py:15](src/data/labellers.py#L15) `transfer_nlst_labels` (NLST), plus [converters.py:146](src/data/converters.py#L146) `convert_canuck` (GSE307690 samples with both cannabis + cigarette/vape) |
| Continuous dose-response modeling (exposure duration → malignancy trajectory) | All existing smoke-cell models are categorical only | No monotonic dose→malignancy ordering at single-cell resolution anywhere | [model.py:112](src/model.py#L112) `DoseResponseHead`, `MultiTaskLoss.dose_response_loss` (pairwise ranking hinge) — architecture + loss are implemented and tested, but **no wired source currently supplies real dose data** (see docstring); this is an honest open gap, not a trained claim |
| End-to-end smoke→malignancy→cancer pipeline | Not in any paper, preprint, or conference | Confirmed gap across all source types | [preprocess.py](src/preprocess.py) → [train.py](src/train.py) → [evaluate.py](src/evaluate.py) → [inference.py](src/inference.py) |

Full table with citations: [ARCHITECTURE.md §9](ARCHITECTURE.md#9-novel-contributions-vs-literature).
This table is also reproduced automatically in the generated demo report — see
[Running the full demo](#running-the-full-demo).

## Benchmarking framework (Phase 1: does the neural model beat simple baselines?)

`src/benchmarks/` answers one narrow, honest question: under the exact same
subject-level splits, does `MultiSmokeCancerNet` actually beat simple
baselines? It does **not** do causal modelling, counterfactual generation,
pathway-constrained learning, or foundation-model integration — those are
explicitly out of scope for this phase.

**Two tasks, defined once and never mixed:**

| | Task A — smoke-type classification | Task B — subject-level cancer prediction |
|---|---|---|
| Unit of independence | subject | subject bag |
| Labels | effective smoke classes 0..K-1 | known cancer outcome only (never fabricated) |
| Primary metric | **subject-weighted** macro-F1 (one vote per subject — a subject with 10,000 cells cannot outvote one with 100) | AUROC (undefined, not 0.5, if a split has one class) |
| Secondary metrics | cell-weighted macro-F1, balanced accuracy, weighted F1, per-class P/R/F1, confusion matrix | AUPRC, balanced accuracy, sensitivity, specificity, F1, Brier score, ECE |
| Eligibility check | [eligibility.py](src/benchmarks/eligibility.py)`::check_task_a_eligibility` | `check_task_b_eligibility` — subjects without a real NLST/TCGA-linked outcome are excluded, never scored as a fabricated negative |

**Baselines** ([baselines.py](src/benchmarks/baselines.py)): majority/prevalence
predictor, logistic regression, random forest, HistGradientBoosting, and a
small MLP (a few thousand parameters vs. MultiSmokeCancerNet's ~2.9M) — all
fit only on the data they're given, with deterministic seeds and recorded
hyperparameters/library versions. Task A also runs the real neural model
(`neural`); Task B runs three MIL pooling variants sharing one encoder
(`mean_mil` / `max_mil` / `attention_mil` — [model.py](src/model.py)'s
`MIL_POOLINGS`) so gated attention has to earn its extra parameters over
plain mean/max pooling, not just be assumed better.

**Grouped CV** ([cross_validation.py](src/benchmarks/cross_validation.py)):
wraps `data/splitting.py::grouped_kfold` over the **train+val subject pool
only** — test is never touched during CV or hyperparameter search. Folds
are subject-disjoint; a fold too small for `MILEligibilityError`
(train.py's `check_mil_eligibility`) is recorded as an undefined result with
its reason, not silently dropped or coerced to AUROC=0.5. Results are
aggregated (mean/std/median/95% bootstrap CI, n valid vs. n undefined) across
folds and seeds `[42, 43, 44]` by default.

**Calibration and frozen threshold** ([calibration.py](src/benchmarks/calibration.py),
Task B only): Platt/isotonic calibration and threshold selection (fixed 0.5,
Youden's J, balanced-accuracy, sensitivity-constrained) are fit on
validation predictions only, then frozen into one `FrozenThresholdPolicy`
applied to test **exactly once** (`apply_to_test` raises on a second call) —
replacing the previous non-clinical, unvalidated fixed 0.70 cutoff.

**Statistical comparison** ([reporting.py](src/benchmarks/reporting.py)):
paired fold-level differences, bootstrap CI, Cohen's d, win/tie/loss count.
A model is only reported as "meaningfully better" if it wins >=70% of paired
folds AND the CI on the paired difference excludes zero — a numerically
higher mean alone is never sufficient (`summarize_comparison`).

**Leave-one-source-out** ([ood.py](src/benchmarks/ood.py)): trains on every
GEO source except one, evaluates only on the held-out source's subjects.
Species/label-semantics compatibility across sources is config-driven
(`incompatible_sources`, `species_by_source`), not auto-inferred — a source
without that information declared explicitly compatible is marked
`NOT_COMPARABLE`, not silently included.

```bash
python -m benchmarks.runner --synthetic --fast     # CI: software-only check, no real data
PYTHONPATH=src python3 -m benchmarks.runner --synthetic --fast --task smoke

PYTHONPATH=src python3 -m benchmarks.runner --config configs/default.yaml --task smoke \
    --models majority logistic random_forest gradient_boosting small_mlp neural \
    --cv-folds 5 --seeds 42 43 44 --output artifacts/benchmarks

PYTHONPATH=src python3 -m benchmarks.runner --config configs/default.yaml --task cancer \
    --models prevalence logistic random_forest gradient_boosting small_mlp \
    mean_mil max_mil attention_mil --calibration auto --output artifacts/benchmarks
```

Each run writes an **immutable** `artifacts/benchmarks/<run_id>/` (a
colliding `--run-id` raises rather than overwriting) containing
`run_manifest.json` (git SHA, seeds, split fingerprint, label mapping),
`eligibility.json`, `metrics/*_folds.{json,csv}`, `calibration/frozen_policy.json`,
`comparisons.json`, `summary.json`, and a human-readable `report.md` — every
synthetic run is stamped `synthetic: true` and the report opens with a
"validates software only" warning so it can never be mistaken for a
real-data result.

**Fixed since the first Phase 1 pass** (a real, previously-undiscovered
leakage bug, not a style change): grouped CV used to reuse the
`ExperimentContext`'s outer `PreprocessingArtifact` — fit once on ALL
original-train subjects — across every fold. Any original-train subject that
became an *inner* CV validation subject had therefore already influenced the
gene-scaling means/stds and HVG selection it was then "held out" against.
CV now refits a fresh `PreprocessingArtifact` per fold, per seed, from
`context.normalized_adata_for_refit` (the pre-HVG, pre-scaling normalized
expression `run_pipeline_split_aware()` now captures), using only that
fold's own training subjects — see
[fold_preprocessing.py](src/benchmarks/fold_preprocessing.py). The same
per-fold cell/bag datasets fixed a second leak: cancer-task MIL encoder
pretraining used to always use the OUTER train/val cell split regardless of
which subjects were in a given cancer-CV fold, so a fold's own validation
subjects could appear in Phase 1 encoder training. Leave-one-source-out
similarly now refits preprocessing using only the remaining-source training
subjects, and defaults an unlabeled source's species/semantics compatibility
to `NOT_COMPARABLE` rather than assuming it's fine. Ground-truth malignancy
labels are no longer a Task B feature (they can proxy the cancer outcome
directly for outcome-linked sources) — the only sanctioned path back in is a
validated out-of-fold prediction (`features.py::validate_oof_predictions`).
Logistic regression and the small MLP now fit a `StandardScaler` inside a
`Pipeline`, scoped to whatever data each fold passes in. A single-class
training fold (an expected outcome of small grouped-CV folds, not a bug) no
longer crashes baselines outright or silently mis-indexes `predict_proba`'s
positive-class column. Statistical comparison now reports two confidence
intervals: a fold-level one (folds from repeated seeds over the same subject
pool overlap and aren't independent — descriptive only) and a seed-level one
(bootstraps independent per-seed means — the one `summarize_comparison`
actually requires to call a model "meaningfully better").

**Fixed in the second pass** (see git history for the exact commits): the
snapshot CV/OOD refit from (`normalized_adata_for_refit`) is now captured
AFTER cell-type annotation runs, not before — every fold and OOD evaluation
used to silently see `cell_type_id=0` for every cell regardless of its real
CellTypist annotation (see [preprocess.py](src/preprocess.py) and
[fold_preprocessing.py](src/benchmarks/fold_preprocessing.py)). The declared
`SMOKE_SEARCH_SPACE`/`CANCER_SEARCH_SPACE` are now consulted by real,
leakage-free nested grouped-CV selection
(`hyperparameter_search.py::select_hyperparameters_nested`) — wired into Task
B's final frozen-test path, computed strictly within the outer-train
partition; a model with no declared search space is recorded as such
explicitly, never silently skipped. `features.py::cap_cells_per_subject` is
now wired into the neural adapter's per-fold cell-level training
(deterministic, computed independently per split — see
`cross_validation.py::run_smoke_cv`'s `feature_mode`/`max_cells_per_subject`
fold fields). Leave-one-source-out no longer infers its reference
species/cohort from whichever source sorts first lexicographically — an
explicit `reference_species` is now required whenever `species_by_source` is
declared — and a subject assigned to more than one `dataset_source` is
rejected outright rather than silently resolved. `ExperimentContext` now
rejects a manifest subject *missing* from its cell dataset (previously only
the reverse direction — an unexpected subject — was checked), rejects
blank/placeholder bag subject IDs, and exposes fingerprinted `run_identity()`
(manifest + preprocessing-artifact + label-mapping + config fingerprints)
that a checkpoint/result reload can verify against
(`validate_run_identity()`). A durable, restart-and-concurrency-safe one-time
frozen-test guard (`test_guard.py::FrozenTestGuard`, atomic
`O_CREAT|O_EXCL` file creation) is wired into Task B's final path.

**Fixed in the third pass** (see git history for the exact commits — this
supersedes the "opt-in guard" / "baseline-only final evaluation" / "search
space declared but not consulted for neural/MIL" limitations listed above):

- **Cell-type annotation is now inductive.** `annotate_cell_types()`
  defaults to CellTypist's `majority_voting=False` mode: a cell's predicted
  label is a pure function of that cell's own expression, independent of
  which other cells (in particular, held-out validation/test cells) are
  present in the same call — previously `majority_voting=True` let an
  over-clustering pass computed across train+val+test cells influence a
  training cell's own label. The fixed `CELL_TYPE_MAP` label-to-ID table is
  unchanged (a static dict, never data-derived) but is now fingerprinted and
  persisted on `PreprocessingArtifact` (`cell_type_map_fingerprint`,
  `cell_type_annotation_mode`) for audit — see
  [data/transforms.py](src/data/transforms.py).
- **Neural/MIL candidates can win the final frozen-test evaluation.**
  `final_evaluation.py::select_final_candidate` ranks every requested
  model — classical baseline or MIL (`mean_mil`/`max_mil`/`attention_mil`/
  `neural`) — by CV/development evidence alone; the winner is never silently
  swapped for a baseline. A model with an undefined CV metric across every
  fold is recorded ineligible with a reason, not dropped without a trace.
- **The frozen-test guard is now mandatory for every non-synthetic run.** A
  safe default location (`<output>/.frozen_test_guards/`) is derived from
  the run's own output root when `benchmarks.frozen_test_guard_dir` isn't
  configured, and the guard is keyed by
  `ExperimentContext.guard_identity_fingerprint()` (manifest + preprocessing
  + label-mapping + config + selected-model fingerprints combined) rather
  than by `run_id`, so a fresh `--run-id` cannot bypass it. Disabling the
  guard is possible only via the explicit, synthetic-only
  `benchmarks.disable_frozen_test_guard` config flag (or CLI
  `--disable-frozen-test-guard`, which `main()` rejects outside
  `--synthetic`); attempting to disable it for a real run raises
  `FrozenTestGuardDisabledInRealModeError`.
- **Hyperparameter selection is integrated into every outer CV fold, for
  both tasks.** `hyperparameter_search.py::select_nested_hyperparameters_with_refit`
  runs a real inner grouped-CV — refitting its own `PreprocessingArtifact`
  from only each inner fold's own training subjects — inside every outer CV
  fold, for classical baselines (Task A and Task B) and, via a small bounded
  fixed candidate set (`NEURAL_SEARCH_SPACE`/`MIL_SEARCH_SPACE` in
  [cross_validation.py](src/benchmarks/cross_validation.py)), for the
  neural/MIL models too. Every outer-fold record now carries its
  `hyperparameter_search` result (candidates tried, inner scores, selected
  params, inner fold partitions); a model with no tunable parameters records
  `no_search_space: true` explicitly.
- **The final development/fit/calibration protocol now uses the whole
  train+val pool, correctly.** `final_evaluation.py` selects one final
  configuration via nested CV over the entire development pool, generates
  subject-grouped out-of-fold (OOF) predictions across that same pool (each
  OOF subject predicted by a fold-refit model/artifact that never saw that
  subject), fits calibration and selects the threshold from those OOF
  predictions exclusively (never a plain validation split, never test), then
  refits ONE final preprocessing artifact and model on all (and only)
  development subjects before the single, guarded test evaluation. MIL
  candidates' final refit carves a small internal validation slice out of
  the development pool for Trainer's own checkpoint-selection bookkeeping
  (see limitations below) — classical baselines have no such requirement and
  use every development subject directly.

**Fixed in the fourth pass** (this supersedes several "third pass" claims
above and in `ARCHITECTURE.md` that were, on closer review, still
inaccurate — see git history for the exact commits):

- **The frozen-test guard is now acquired strictly before ANY test access.**
  `run_cancer_task` (`src/benchmarks/runner.py`) is split into a
  development-only stage and a guarded stage. The development-only stage —
  candidate selection, OOF generation, calibration/threshold fitting, and
  the final dev-pool fit (`final_evaluation.py::fit_final_candidate_on_dev_pool`)
  — has function signatures that structurally cannot accept test subject
  IDs, test bags, or test labels; `evaluate_frozen_test` is the only
  function that reads `context.test_bags`/`context.subjects_for("test")`,
  and it is called for the first time only inside the `try` block that
  follows `FrozenTestGuard.acquire()`. Previously the guard was acquired
  only immediately before computing the final metric, after test labels had
  already been read and test predictions already generated.
- **OOF predictions are now selection-clean.** Previously
  `generate_subject_oof_predictions` accepted ONE hyperparameter
  configuration selected once from the whole development pool and reused it
  for every OOF subject — meaning a held-out subject's own label had
  already influenced the configuration used to predict it, even though the
  fold's model weights excluded that subject. Each OOF fold now runs its
  OWN inner grouped-CV hyperparameter/config selection
  (`final_evaluation.py::_oof_fold_hyperparameters`) using only that fold's
  OOF-training subjects, then fits on the full OOF-training set with the
  fold-selected configuration before predicting the OOF-held-out subjects —
  the same nested-CV pattern already used by the outer CV loop, applied one
  level deeper. The dev-pool-wide selection is still computed once, but is
  now used only for the final refit's configuration, never reused as every
  OOF fold's configuration.
- **The final MIL fit now trains on every eligible development subject.**
  `train.py::Trainer.phase1_final_fit`/`phase2_final_fit` are new,
  fixed-epoch training methods with no internal validation split and no
  validation-based checkpoint selection — unlike the normal `phase1`/`phase2`
  (still used for CV/OOF fold fits, which correctly hold out a real
  validation split). `NeuralCancerAdapter.fit_final` uses these for the
  final dev-pool refit, so every development subject now contributes to
  gradient updates; the "MIL final refit carves out an internal validation
  slice" limitation from the previous pass is resolved, not merely
  documented.
- **CellTypist annotation failure now fails loudly by default.**
  `annotate_cell_types()` previously printed a warning and returned `adata`
  unchanged on any CellTypist failure — leaving `obs["cell_type_id"]`
  either absent or stale, not actually defaulted to anything. It now raises
  `CellTypeAnnotationError` by default; an explicit
  `allow_diagnostic_fallback=True` (never passed by the real pipeline
  entry points in `src/preprocess.py`) is required to fall back to a fixed
  placeholder label for every cell, which stamps
  `cell_type_annotation_degraded=True` on the resulting `PreprocessingArtifact`.
  `ExperimentContext.from_pipeline_result` rejects a degraded artifact
  outright — a real run can never silently proceed on fabricated cell types.
- **Real OOF predictions are now persisted, not just fold-membership
  counts.** `predictions/cancer_<candidate>_oof.csv` — one row per
  development subject with its actual OOF probability, fold, and
  fold-local selected-hyperparameters/fingerprint columns — is now written
  alongside `calibration/frozen_policy.json`'s `oof_summary` (which still
  carries fold membership only, for the markdown report).

**Fixed in the fifth pass** (see git history for the exact commits):

- **Task B eligibility no longer reads test labels or test class counts.**
  `check_task_b_eligibility` previously computed known-outcome/class counts
  from `context.test_bags` and rejected the whole run if the test split had
  only one class — test composition could change whether an experiment
  proceeded at all. It is now split into
  `eligibility.py::check_task_b_development_eligibility(train_bags, val_bags)`
  (the only gate run before the guard; its signature structurally cannot
  accept `test_bags`) and `check_test_evaluability(test_bags)` (run only
  inside the guarded stage, after `FrozenTestGuard.acquire()`): a one-class
  test split now produces `auroc_auprc_defined=False` with a recorded
  reason, and the run continues rather than being rejected pre-guard.
- **The frozen-test guard identity is now deterministic across runs of the
  same scientific configuration.** It previously included
  `fitted.model_metadata`, which for neural/MIL candidates carries
  `fit_seconds` — real wall-clock timing that differs on every run, making
  the guard identity itself non-reproducible. The identity now uses
  `context.guard_identity_fingerprint`'s `extra` payload with
  `final_model_state_fingerprint` — a SHA-256 of the ACTUAL fitted model
  weights (`benchmarks/model_fingerprint.py`: canonicalized `state_dict`
  bytes for neural/MIL models via `torch_state_dict_fingerprint`, canonicalized
  fitted sklearn attributes via `sklearn_model_state_fingerprint` for
  baselines) — plus a deterministic `calibration_fingerprint` and a new
  `test_membership_fingerprint` (`ExperimentContext.test_membership_fingerprint`,
  derived only from `split_manifest.test_subjects`, never from `test_bags`).
  No timestamp, run ID, or output directory name enters the identity.
- **The guarded frozen-test transaction is now complete, not just guarded
  metric computation.** `run_cancer_task` now builds an immutable
  `calibration/frozen_test_result.json` (identity/model-state/preprocessing/
  calibration fingerprints, the frozen threshold, aggregate metrics, test
  membership fingerprint, evaluated-subject count, synthetic flag, and its
  own `artifact_fingerprint` — no raw test labels or probabilities),
  persists it atomically, reloads and verifies it, and only then calls
  `guard.mark_completed(...)`, which now references that exact artifact
  fingerprint. Any failure in this sequence (evaluation, calibration,
  serialization, or verification) marks the guard failed instead — `mark_completed`
  is structurally unreachable unless persistence and reload verification
  both succeeded.
- **Every JSON/CSV artifact this framework writes is now genuinely atomic.**
  `benchmarks/atomic_io.py` writes a uniquely named temp file in the
  destination directory, fsyncs it, and calls `os.replace()` onto the final
  path — `reporting.py::write_json`/`write_csv_table` and
  `test_guard.py::FrozenTestGuard.mark_completed`/`mark_failed` all use it
  now, replacing the previous direct `open(path, "w")` writes that offered
  no atomicity guarantee. `FrozenTestGuard.acquire()` still uses
  `O_CREAT | O_EXCL` (a different, exclusive-creation primitive, not
  replace-based) since that is what makes first-acquisition itself race-free.
  The guard also now records a per-acquisition `owner_token`; a
  `FrozenTestGuard` instance that never itself called `acquire()` (or whose
  token doesn't match what's on disk) raises `FrozenTestGuardOwnershipError`
  from `mark_completed`/`mark_failed` rather than being able to finalize a
  guard file it does not own.
- **The OOF CSV now carries real fingerprints instead of raw JSON under a
  "fingerprint" column name.** `training_subjects_fingerprint` and
  `validation_subjects_fingerprint` are now SHA-256 hashes of the canonical
  sorted subject list (previously `training_subjects_fingerprint` was a raw
  JSON-serialized list). Every successfully predicted row now also carries
  `selected_params_fingerprint`, `inner_selection_fingerprint` (already
  present), and `model_state_fingerprint` for the exact fold model that
  produced that subject's OOF probability — writing refuses to proceed if a
  "predicted" row is missing its model-state fingerprint. The CSV is
  written atomically, reloaded, and its row count verified; the file's own
  SHA-256 is recorded as `oof_summary.oof_artifact_fingerprint` in
  `calibration/frozen_policy.json`, so calibration output references the
  exact OOF artifact it was fit from.

**Fixed in the sixth pass** (see git history for the exact commits):

- **`atomic_write_bytes` now guarantees complete writes.** `os.write()` is
  only guaranteed to write *up to* the requested number of bytes — a short
  write is normal OS behavior, not an error. `benchmarks/atomic_io.py` now
  loops (`_write_all`) until every byte is written, retries
  `InterruptedError` explicitly, and treats zero-byte progress as a hard
  error rather than looping forever. A failure anywhere before the
  temp-to-destination `os.replace()` (write, fsync, or close) now always
  removes the temp file and leaves the previous destination content
  untouched.
- **`model_fingerprint.py` no longer falls back to `repr()` for unsupported
  fitted state.** The fallback was unsafe for a scientific identity: fitted
  `sklearn.tree._tree.Tree` objects (used by every random forest, including
  `random_forest` in both `SMOKE_BASELINES`/`CANCER_BASELINES`) are a Cython
  extension type with no `__dict__`, so they previously fell all the way
  through to `repr(obj)`, which embeds the object's memory address —
  different every process, meaning two identically fitted forests could
  fingerprint differently. Canonicalization now explicitly handles
  `sklearn.tree._tree.Tree` (node/threshold/feature/value arrays),
  `sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor` (whose
  fitted fields don't follow sklearn's trailing-underscore convention, so a
  generic reflection walk silently collected nothing for it),
  `HistGradientBoostingClassifier` itself (whose actual learned trees live
  in the private `_predictors`/`_baseline_prediction`/`_bin_mapper`
  attributes, not any public trailing-underscore attribute), and
  `DummyClassifier` (whose `strategy="constant"` predicted value lives in
  the constructor parameter `constant`, not a fitted attribute — the model
  used for this framework's single-training-class fallback path). Any
  fitted value with no defined canonicalization now raises
  `UnsupportedModelStateError` instead of silently using `repr()`.
- **CellTypist/scikit-learn pretrained-model version compatibility is now
  detected and surfaced explicitly, not silently absorbed.** CellTypist's
  published `Immune_All_Low.pkl` model was serialized with scikit-learn
  0.24.1; loading it under this project's scikit-learn (>=1.4, 1.9.0 in CI)
  always emits scikit-learn's own `InconsistentVersionWarning` — a known,
  disclosed, *upstream* compatibility gap this project cannot fix directly
  (it does not control CellTypist's published model artifact, and no
  scikit-learn-1.x-compatible replacement has been identified). As of the
  ninth pass below, real (non-diagnostic) preprocessing fails closed on
  this mismatch by default — this bullet is retained only as the
  historical record of when the mismatch was first detected and persisted
  as provenance; see the ninth pass for the current, fail-closed behavior.

**Fixed in the eighth pass** (see git history for the exact commits):

- **`FrozenTestGuard.acquire()` no longer risks a truncated guard file.**
  The previous implementation wrote directly into an `O_CREAT | O_EXCL`
  destination fd, which left a window — between the exclusive create and the
  write completing — during which a concurrent reader could observe a
  guard file that exists but holds truncated/incomplete JSON. It now writes
  the complete payload to a temp file, fsyncs it, and only then atomically
  `os.link()`s it into the guard path (`atomic_io.py::exclusive_create_bytes`)
  — a hard link is a single atomic operation that fails with
  `FileExistsError` if the destination already exists (the same race-free
  create-if-absent guarantee `O_CREAT | O_EXCL` gives), and the instant it
  succeeds the guard path refers to content that was already complete and
  durable. Malformed/truncated guard JSON now raises a dedicated
  `FrozenTestGuardCorruptedError` from `_read()` instead of ever being
  silently treated as "no guard" (which could have re-opened one-time
  frozen-test access). A real multiprocess acquisition race test (six
  separate OS processes, `spawn` context) proves exactly one process wins
  and every loser gets `FrozenTestInProgressError`, never a corrupted-state
  false negative.
- **`model_fingerprint.py`'s remaining generic `hasattr(obj, "__dict__")`
  fallback is gone.** It was still unrestricted — any fitted object with a
  `__dict__` (even one holding no genuinely fitted state, or fitted state
  under a non-trailing-underscore name) was silently accepted. Writing
  tests against the now-explicit whitelist (`LogisticRegression`,
  `StandardScaler`, `RandomForestClassifier`, `MLPClassifier`,
  `DecisionTreeClassifier`) surfaced a second real, previously-undetected
  bug: `HistGradientBoostingClassifier._bin_mapper` (a
  `sklearn.ensemble._hist_gradient_boosting.binning._BinMapper` holding the
  actual learned bin thresholds) was also falling through the old generic
  fallback — now explicitly canonicalized. Any object with a `__dict__` but
  no whitelist entry, or a `__slots__`-only/empty-`__dict__` object, now
  raises `UnsupportedModelStateError`; determinism across independent OS
  processes (not just repeated calls in one process) is tested for every
  registered baseline.
- **Real cell-type annotation provenance validation now fails closed.** The
  previous check (`getattr(artifact, "cell_type_annotation_degraded",
  False)`) treated a *missing* provenance field as "not degraded" — safe by
  accident, not by design. `data/preprocessing.py::validate_cell_type_provenance`
  now requires all three provenance fields to be explicitly present and
  valid: `cell_type_annotation_degraded is False` exactly (missing/None/True
  all refused), `cell_type_annotation_mode` in an explicit allow-list
  (`inductive_per_cell` or the new `pseudo_bulk_no_cell_type_identity`;
  `majority_voting`/`diagnostic_fallback`/anything else refused), and for
  `inductive_per_cell`, a well-formed 64-hex-char `cell_type_map_fingerprint`
  that matches the *current* `CELL_TYPE_MAP` fingerprint exactly. Pseudo-bulk
  sources (no per-cell expression for CellTypist to classify) now get an
  explicit, documented provenance mode instead of silently leaving every
  field unset. `fit_preprocessing()` now propagates these fields from
  `adata.uns` onto every artifact it produces — including per-fold/per-OOD
  refits (`benchmarks/fold_preprocessing.py`), which previously produced
  artifacts with unset (`None`) cell-type provenance regardless of the outer
  artifact's real annotation state.
- **CellTypist/scikit-learn compatibility provenance is now persisted, not
  just warned about.** `annotate_cell_types` now records
  `cell_type_annotation_compatibility` (CellTypist version, model name,
  runtime scikit-learn version, the serialized scikit-learn version(s)
  sklearn's own `InconsistentVersionWarning` reports, and a `compatible`
  boolean) onto the AnnData and, via `fit_preprocessing`, onto every
  `PreprocessingArtifact`. The warning itself is still never suppressed
  (see the sixth pass's decision on why hard-failing by default would be a
  functional regression, not a fix, given this exact warning has been
  present in every prior passing CI run) — this pass adds the durable
  record, not a behavior change to the default warn-vs-fail policy. Making
  a real (non-synthetic) run fail closed on this mismatch **by default**
  remains open — see limitations below.
- **CI dependency installation is now reproducible.** `requirements.txt`'s
  entries are lower-bound ranges (`scikit-learn>=1.4.0`, etc.) — installing
  from it alone does not reproduce a specific environment. A new
  `constraints-ci.txt` pins the exact Linux/Python-3.11 versions this
  project's tests are validated against (captured from an environment whose
  independent resolution of the same unconstrained `requirements.txt`
  matched every version the last completed CI job reported); `.github/workflows/tests.yml`
  now installs via `pip install -r requirements.txt -c constraints-ci.txt`,
  pins Python to `3.11.15`, records core dependency versions in the CI log,
  and busts its pip cache when either file changes.
- **CI workflow triggers fixed.** The `push` trigger referenced a specific,
  temporary feature branch name that had already gone stale (the repository
  had since moved to a different branch) — it now triggers on push/PR
  against `main` plus an on-demand `workflow_dispatch`, so it keeps working
  regardless of which branch is checked out.
- **New reproducibility artifacts**: every run now writes `environment.json`
  (Python version, platform, git SHA, working-tree dirty flag, synthetic
  flag, and the installed version of every scientific dependency that could
  affect results) and `preprocessing/final_artifact.json` (the full outer
  `PreprocessingArtifact` this run used, cross-referenced with its own
  fingerprint).

**Fixed in the ninth pass** (see git history for the exact commits):

- **CI's dependency installation was actually broken, not just imprecise.**
  `constraints-ci.txt` only narrows a version pip is already trying to
  install — it does not add `pytest` as a dependency, and `requirements.txt`
  never listed it, so the CI job's test step failed with `No module named
  pytest`. Fixed with a new `requirements-test.txt` (pytest only) installed
  alongside `requirements.txt` via `pip install -r requirements.txt -r
  requirements-test.txt -c constraints-ci.txt`, plus a `python -m pytest
  --version` verification step so a future break in this chain fails at the
  install step, not silently inside the test step.
- **CI's "Record environment" step was silently producing fake output.**
  It called `importlib.metadata.version(...)` without importing
  `importlib.metadata` (only the parent `importlib` package), so every
  lookup raised `AttributeError`, caught by a bare `except Exception` that
  printed `UNAVAILABLE` for every package — and the whole step was wrapped
  in `|| true`, so this never failed the build. It also looked up the
  import name `sklearn` instead of the installable distribution name
  `scikit-learn`, which would have returned `None`/not-found even with the
  import fixed. Replaced with `src/benchmarks/env_versions.py`
  (`collect_core_package_versions`), a small tested utility shared by CI
  (`python -m benchmarks.env_versions`, which now exits non-zero if any
  required package is missing — no `|| true`) and by
  `reporting.py::_environment_snapshot`'s reproducibility artifact, so both
  callers use the same correct distribution-name mapping.
- **Real (non-diagnostic) preprocessing now fails closed on a
  CellTypist/scikit-learn version mismatch by default.** The previous
  passes recorded the mismatch as provenance but left the default
  behavior permissive (`strict_sklearn_compatibility=False`, an opt-in
  flag real pipeline entry points never set). `annotate_cell_types` no
  longer has a `strict_sklearn_compatibility` parameter or a
  `CELLTYPIST_STRICT_SKLEARN_COMPAT` environment-variable escape hatch —
  strictness is now tied unconditionally to `allow_diagnostic_fallback`
  (the same flag that already gated CellTypist *failure* fallback): a real
  call (the default) raises `CellTypistCompatibilityError` — propagated
  undisturbed, never wrapped in the generic `CellTypeAnnotationError` —
  and `allow_diagnostic_fallback=True` is the only way to tolerate the
  mismatch, which now always stamps `cell_type_annotation_degraded=True`
  (previously it stayed `False`), so real `ExperimentContext` construction
  automatically rejects a diagnostic-tolerated mismatch through the
  existing degraded-provenance guard. `run_pipeline`/`run_pipeline_split_aware`
  expose this only via an explicit `cell_type_allow_diagnostic_fallback`
  config key that no real production config sets (absent = fail-closed by
  default, not dependent on a caller remembering to pass anything).
- **`fold_preprocessing.py::artifact_fingerprint` now incorporates
  cell-type/compatibility provenance.** Previously scoped to gene
  scaling/HVG selection only, so two artifacts with identical gene
  statistics but different (or stale) cell-type annotation/compatibility
  status collided on identity. Now includes
  `cell_type_map_fingerprint`/`cell_type_annotation_mode`/
  `cell_type_annotation_degraded`/`cell_type_annotation_compatibility` —
  closing the gap disclosed in the eighth pass's limitations.
- **CI action versions upgraded.** `actions/checkout@v4` and
  `actions/setup-python@v5` both still ran on the Node 20 runtime GitHub
  was warning about; upgraded to `actions/checkout@v7` and
  `actions/setup-python@v6` (verified via each action's own `action.yml`
  on its GitHub release tag: both declare `runs.using: node24`).

**Known Phase 1 limitations remaining** (see `report.md`'s own limitations
section for the same list, generated fresh per run): the neural/MIL bounded
search spaces compared inside nested CV (both the outer-CV-fold and the
OOF-fold-local selections) are deliberately small (one or two fixed
candidates, e.g. `phase1_epochs`/`pretrain_epochs`), not a real
hyperparameter grid — a full grid would mean many more full training runs
per fold; Task A baselines still run in subject-summary mode only (one
feature vector per subject) for their own training/prediction — the neural
adapter is the only Task A model exercising true cell-level training, so
"cell-weighted" metrics for baselines remain explicitly marked
not-applicable rather than presented as real per-cell scores; the immutable
artifact directory now has `environment.json` and
`preprocessing/final_artifact.json` but still does not contain every file
the ideal schema calls for — a sanitized `configuration.json` snapshot and a
`models/final_model_manifest.json` recording the final candidate's
architecture/hyperparameters/model-state fingerprint are not yet written
(per-fold identity is already covered by the fold-local fingerprints in
`metrics/*_folds_partitions.json`, now provenance-complete per the ninth
pass above); this project has never had access to a CellTypist model
reserialized against a matching scikit-learn version, so real (non-
synthetic) preprocessing genuinely cannot complete in this environment
without the disclosed `cell_type_allow_diagnostic_fallback` diagnostic
override — this is the intended, honest fail-closed behavior, not a bug,
and is not something a future pass can silently "fix" without an actually
compatible upstream model; attention weights remain an interpretability
aid, not a causal explanation, in every pooling variant.

## Phase 2 — subject-aware class-imbalance correction

**Why cell-level weighted sampling is insufficient.** The "Current results"
table above shows the collapse a class-imbalanced smoke-type head produces:
77% accuracy but 0.27 macro-F1, with zero F1 on the smallest classes. The
obvious fix — a per-cell `WeightedRandomSampler` giving every cell an
inverse-frequency weight — does not actually solve the underlying problem:
a subject with 50,000 cells and a subject with 500 cells in the *same*
class still receive wildly different total sampling probability under that
scheme, even though they are the same number of independent observations
(one subject each). A cell-heavy subject can still dominate optimization
purely by cell count, regardless of any inverse-frequency correction
applied at the cell level.

**Class → subject → cell sampling.** `src/data/sampling.py` implements a
`SubjectBalancedBatchSampler` that draws every training index in three
explicit stages: (1) pick an effective smoke class, (2) pick a unique
subject belonging to that class (uniformly, by default — independent of
that subject's cell count), (3) pick a cell belonging to that subject. A
subject's cell count therefore only controls the diversity of cells *drawn
from it*, never its own selection probability — verified directly by
`tests/test_subject_balanced_sampling.py`, which constructs a class with a
5,000-cell subject and a 50-cell subject and confirms both receive
approximately equal draws. `class_selection` supports `uniform` (default —
equal probability per observed effective class), `natural` (proportional to
unique-subject count), and `inverse_subject_frequency`; `subject_selection`
currently supports `uniform`. Only the training split may use this sampler
— it is wired into `Trainer.phase1`/`phase1_final_fit`/`phase3`'s cell-level
component and the corresponding `NeuralSmokeAdapter`/`NeuralCancerAdapter`
benchmark entry points; validation and test `DataLoader`s remain
`shuffle=False` with no sampler attached, unchanged from Phase 1, and are
never oversampled, duplicated, capped, or reweighted — see
`tests/test_phase2_imbalance_integration.py`'s validation-loader tests and
`tests/test_benchmarks_imbalance_ablation.py`'s validation-fingerprint tests.

**`cells_per_subject_per_batch` is a hard, per-batch MAXIMUM — never a
required minimum.** A subject with fewer cells than the cap is still a
fully valid participant; it just has a smaller effective capacity. Each
subject's effective per-batch capacity is `cells_per_subject_per_batch`
when `replacement: true` (the same cell can be redrawn), or
`min(cells_per_subject_per_batch, that subject's own unique cell count)`
when `replacement: false` (a subject cannot yield more distinct cells than
it has). No subject may contribute more than its effective capacity to any
single batch, with no fallback to an already-exhausted subject. Once every
subject in a class has reached its effective capacity *within the batch
being built*, that class simply stops being drawn for the rest of that
batch (its probability mass is redistributed over classes that still have
capacity) — a feasible degradation of the requested class balance, never a
cap violation. Feasibility is judged against the *sum of effective
capacities* across every subject, not `cap * n_unique_subjects` — a
configuration is infeasible only when that sum cannot fill the largest
requested batch. This is checked once at sampler construction; an
infeasible combination raises `SamplingImpossibleError` immediately, never
mid-epoch and never by silently exceeding the cap. Without replacement, a
physical cell cannot repeat within one batch, but that uniqueness is scoped
to the batch, not the epoch — the same cell may reappear in a later batch;
capacity itself always resets fully between batches. See
`tests/test_subject_balanced_sampling.py`'s cap-enforcement tests, including
mixed-capacity, exact-boundary, and single-subject-class cases.

**`samples_per_epoch` is an exact sample count, not a rounding target.**
`samples_per_epoch=95` with `batch_size=10` yields batch lengths `[10] * 9 +
[5]` — 95 indices total, never rounded up to 100 by a stray ceil-division.
`len(sampler)` always equals the number of batches actually yielded
(`floor(samples_per_epoch / batch_size)`, plus one more for a nonzero
remainder). The final partial batch still obeys every class/subject-cap
constraint. `batches_per_epoch` (an alternative, mutually exclusive way to
set epoch length) always yields `batch_size`-sized batches, unchanged.

**Class weighting vs. sampling — two different corrections.** Inverse-
frequency `CrossEntropyLoss` class weights (`CellLevelDataset.
smoke_class_weights`, computed from the training partition only — never
validation/test, never a fold's held-out subjects) already existed in
Phase 1 and are unchanged in what they compute; Phase 2 makes *when* they
apply configurable via `training.smoke_imbalance.class_weighting`
(`none` | `inverse_frequency`, default `inverse_frequency` — preserves
pre-Phase-2 behavior when the whole `smoke_imbalance` config section is
absent). Sampling and loss weighting correct the same imbalance through
different mechanisms (which subjects/cells appear in a batch, vs. how much
each class's error counts), and combining both is a real double-correction
risk: `focal_alpha_mode: none` (only meaningful when
`loss: focal`) is the explicit way to run subject-balanced sampling without
also reweighting the loss. `training.smoke_imbalance.seed_offset` keeps a
multi-seed benchmark loop's sampling sequences independent of each other
while remaining reproducible from the base seed.

**Focal loss (`model.py`'s `FocalLoss`) is a configurable ablation, not a
replacement.** `training.smoke_imbalance.loss` is `cross_entropy` (default)
or `focal`. For each example: `ce = cross_entropy(logits, target,
reduction="none")` — always **unweighted**; `pt = exp(-ce)`; `focal_factor =
(1 - pt) ** gamma`; `loss = focal_factor * ce`, then, if class alpha is
enabled, `loss = alpha[target] * focal_factor * ce` — alpha is applied
**exactly once**, after the focal modulation, never folded into the
cross-entropy term used to compute `pt` (computing `pt` from an
already-weighted cross-entropy would let alpha distort "how easy is this
example" as well as the final scale — a double-counted correction).
`reduction="mean"` is the plain arithmetic mean of the per-example values —
**not** `nn.CrossEntropyLoss(weight=...)`'s weight-normalized mean — a
deliberate, documented, and tested choice; `reduction="sum"` is
convention-independent and is what the test suite uses to verify alpha is
applied exactly once. `focal_gamma=0` without alpha is mathematically
identical to plain (unweighted) cross-entropy under the same reduction —
verified directly in `tests/test_smoke_imbalance_loss.py`, including a
manually hand-derived tensor example combining `gamma>0` and alpha. The
malignancy, cancer, and dose-response loss terms are unaffected by this
setting. `class_weighting` (sampling-adjacent, computed once per training
run) and `focal_alpha_mode` (loss-adjacent) are two independently
configurable axes — see the double-correction discussion below.

**Rare-class handling is unchanged and still honest.** Phase 2 does not
touch `data/rare_class.py` or `data/label_mapping.py`: the sampler and
class-weighting machinery both operate on the *effective* (post-merge,
contiguous 0..K-1) smoke label, read from whatever `EffectiveLabelMapping`
the run already has wired in, and reject (fail loudly, not silently) any
subject whose per-cell effective labels disagree or fall outside `[0, K)`.
A merged-away or excluded raw class (e.g. `cigar`, ~1 subject in the real
data — see the rare-class discussion above) can never be resurrected by the
sampler, since it never appears in the effective label space the sampler is
built from. This project still does not claim six-class classification
success — the effective, reported class count is whatever
`EffectiveLabelMapping.k` resolves to for the configured rare-class policy.

**Only the training half of each fold is ever capped or rebalanced.**
`run_smoke_imbalance_ablation` calls `cap_cell_dataset` on the fold's
training cells only; every strategy compared in one fold is scored against
that fold's *complete, uncapped, natural* validation population — never
resampled, duplicated, or filtered. Every strategy's fold record carries a
`validation_fingerprint` (subject IDs, per-subject cell counts, total cell
count, a label checksum) so this can be verified directly rather than taken
on faith — `tests/test_benchmarks_imbalance_ablation.py` asserts this
fingerprint is byte-identical across every strategy in the same fold.

**Primary metrics are subject-level, not cell-level.** Every fold record's
`subject_level` block (from
`metrics.py::subject_weighted_full_smoke_metrics_report`) — macro-F1,
balanced accuracy, per-class precision/recall/F1/support, confusion matrix
— is computed after majority-voting each subject's cell-level predictions
into one prediction per subject first, so a subject with many cells cannot
dominate any of these numbers (per-class `support` counts SUBJECTS, not
cells). `subject_weighted_macro_f1`, the ablation's primary comparison
metric, is exactly `subject_level["macro_f1"]`. The equivalent cell-level
numbers are still reported, but only under the explicitly-labeled
`cell_level_diagnostic` key — a secondary diagnostic, never described as
"subject-weighted." Class ordering is always `range(num_classes)`
(stable, independent of which classes happen to be present in a given
fold); absent classes are recorded in `classes_absent_from_val` and get an
honest zero support, never a manufactured value.

**Development-only ablation protocol.**
`src/benchmarks/imbalance_ablation.py::run_smoke_imbalance_ablation` compares
five named strategies (`natural_no_weight`, `natural_inverse_frequency`,
`subject_balanced_no_weight`, `subject_balanced_inverse_frequency`,
`subject_balanced_focal`) using the exact same grouped-subject-CV machinery
`cross_validation.py::run_smoke_cv` already uses — identical outer folds,
identical per-fold preprocessing refit, identical candidate architecture,
only `train.smoke_imbalance` differs between strategies. It runs entirely
over the experiment context's train+val subject pool; the frozen test split
is never accessed and this comparison never invokes the frozen-test guard —
`tests/test_benchmarks_imbalance_ablation.py` proves this directly by
installing a sentinel in place of the context's test data that raises on
any access whatsoever (attribute lookup, indexing, iteration, `len()`) and
running the full ablation against it.

**Honest paired strategy comparisons — never a winner from the mean
alone.** The report's `paired_fold_differences` gives per-(seed, fold)
differences against the declared baseline (`natural_no_weight` by default);
`comparisons` reuses `reporting.py`'s existing `compare_models`/
`summarize_comparison` (the same machinery Phase 1's own CV report uses,
never a duplicated weaker implementation) to add win/tie/loss counts (a
fixed, deterministic tie tolerance), mean and median paired differences, a
count of missing/invalid pairs, and — only when >=2 seeds were run — a
seed-level bootstrap confidence interval (`ci_diff_by_seed`, explicitly
labeled with its resampling unit). `comparisons[name].summary.
meaningfully_better` is `False` whenever the evidence (win fraction,
seed-level CI) does not support a confident selection — including whenever
only one seed was run, regardless of how consistent that one seed's folds
look. Every comparison carries an `independence_note`: folds drawn from
repeated seeds over the same overlapping development subject pool are
explicitly **not** independent biological replications, and per-fold
win/loss counts are descriptive, not a formal significance test.

**Reproducibility, identity, and artifacts.** `training.smoke_imbalance`
lives inside `ExperimentContext.config`, so it is already covered by
`config_fingerprint`/`guard_identity_fingerprint` (see `benchmarks/
context.py`) with no additional plumbing — two runs with different
imbalance strategies never share scientific or frozen-test-guard identity.
Every checkpoint (`Trainer._save`) persists the resolved
`smoke_imbalance_config` and, when the subject-balanced sampler was used,
its realized per-epoch sampling diagnostics (`SamplingDiagnostics`,
`data/sampling.py`) — populated only after a full epoch has actually been
iterated, never for a shuffle-based run (which stays `null`, not a
fabricated value). Alongside the expected/configured fields
(observed/absent effective classes, unique subjects per class, exact
`samples_per_epoch`), the realized block records exactly what the sampler
did: `realized_total_samples` (must equal `samples_per_epoch`),
`realized_batch_sizes` (the exact yielded size of every batch, including
the final partial one), `realized_cells_per_subject` (total cells drawn per
subject across the epoch), `realized_subject_counts_per_batch` (a
per-subject count for every batch — the direct way to verify the hard cap
independently), `realized_max_subject_cells_per_batch` (the largest single
subject's contribution in each batch), and
`realized_repeated_cell_draws_per_batch`/`_total` (repeated physical-cell
draws per batch — always zero when `replacement: false`, informative when
`replacement: true` and a subject's cell pool is smaller than its share of
the batch). An interrupted or failed epoch leaves the previous
`last_realized_diagnostics` unchanged rather than exposing a partial,
mislabeled result. `NeuralSmokeAdapter.metadata()`/
`NeuralCancerAdapter.metadata()` expose the same fields for benchmark
reports. `run_smoke_imbalance_ablation`'s own output is not only an
in-memory dict: `write_imbalance_ablation_artifact` persists it atomically
under `<run_dir>/metrics/smoke_imbalance_ablation.json` (plus a flattened
per-fold CSV), reusing the same atomic-write and environment-snapshot
machinery every other benchmark artifact uses (`schema_version`, git SHA,
package versions, `development_only: true`, `frozen_test_data_accessed:
false`) — never a parallel, incompatible persistence system.
`python -m benchmarks.runner --task smoke --imbalance-ablation ...` wires it
into the existing CLI entry point.

**Default sampler status: `shuffle`, deliberately unchanged.** No real
(non-synthetic) development-only comparison has yet established that
`subject_balanced` — or any specific `class_weighting`/`loss` combination —
actually outperforms the pre-Phase-2 default on real data; the ablation
framework above exists to eventually produce that evidence, not to
presuppose its outcome. `configs/default.yaml`'s `train.smoke_imbalance`
block therefore matches `data/sampling.py::DEFAULT_SMOKE_IMBALANCE_CONFIG`
exactly (`sampler: shuffle`, `class_weighting: inverse_frequency`,
`loss: cross_entropy`) — verified by
`tests/test_subject_balanced_sampling.py`'s YAML/Python consistency tests —
so an absent or default `smoke_imbalance` config reproduces pre-Phase-2
behavior exactly. `subject_balanced` sampling and focal loss are fully
implemented, tested, and available as opt-in/experimental configuration;
switching the *default* requires real development-only ablation evidence
selected before any frozen-test use, never a frozen-test result and never
the mere existence of the feature.

**No real held-out imbalance-strategy comparison has been run.** Everything
above has been exercised on synthetic data (`tests/`, and the CLI's
`--synthetic --fast` workflows) and is unit/integration tested, but no real
frozen-test evaluation of any imbalance strategy has occurred — running the
frozen-test guard is a deliberate, one-time, separately-authorized action
(see the guard discussion above), not something this pass performs. Any
future macro-F1/accuracy numbers for a specific strategy must come from an
actual run of `run_smoke_imbalance_ablation` (development-only) or the
frozen-test protocol (at most once), never from this description — and a
development-only ablation result never by itself authorizes a claim of
scientific superiority.

## Pipeline

```
raw scRNA-seq (AnnData)
  → src/preprocess.py   (QC, normalize, HVG, batch correction, cell typing)
  → src/model.py         (shared encoder → smoke head + malignancy head → gated attention MIL)
  → src/train.py         (3-phase curriculum: cell-level → aggregator → end-to-end)
  → src/evaluate.py       (cell + subject metrics, attention interpretability)
  → src/inference.py      (predict on new unlabelled subjects — single, batch, or H5AD)
```

## Project layout

```
src/
  constants.py       label maps, smoke marker genes, dimension constants
  data/
    loaders.py        I/O — one loader per data source type
    transforms.py      QC filtering, normalization, HVG selection, batch correction, cell typing
    labellers.py        smoke-type label transfer, malignancy labels, class weights
    assembly.py          merge sources, build MIL bags, export arrays
    splitting.py          subject-level train/val/test split + grouped K-fold CV
    preprocessing.py       leakage-free fit/transform preprocessing artifact
    rare_class.py           configurable policy for statistically-too-small classes
    sampling.py               subject-aware class -> subject -> cell balanced batch sampler (Phase 2)
  preprocess.py        orchestrates data/ into run_pipeline(config) / run_pipeline_split_aware(config)
  model.py              MultiSmokeCancerNet (encoder, both heads, gated attention MIL)
  train.py                three-phase Trainer (cell-level, aggregator, end-to-end)
  evaluate.py             cell-level + subject-level metrics, interpretability report
  inference.py            Predictor — predict_subject / predict_batch / predict_h5ad, plus CLI
  benchmarks/            Phase 1 leakage-free benchmarking (see "Benchmarking framework" above)
    context.py             ExperimentContext — the one object every benchmark is built from
    eligibility.py          Task A/B eligibility reports, computed before any training starts
    features.py              subject-summary feature construction, out-of-fold helper
    baselines.py              majority/prevalence, logistic, random forest, gradient boosting, small MLP
    neural.py                  adapters wrapping Trainer/MultiSmokeCancerNet for CV comparison
    metrics.py                  subject/cell-weighted F1, AUROC/AUPRC/Brier/ECE, bootstrap CI
    cross_validation.py          grouped-CV runner around data/splitting.py::grouped_kfold
    calibration.py                 OOF-based calibration + guarded one-time frozen-test evaluation
    ood.py                          leave-one-dataset-source-out validation
    reporting.py                    statistical comparison, immutable artifacts, Markdown/CSV report
    imbalance_ablation.py             development-only smoke-imbalance-strategy comparison (Phase 2)
    runner.py                        CLI entry point (`python -m benchmarks.runner`)
configs/
  default.yaml           data / model / train config used by model.py and train.py
tests/                  one file per src/data module + model/pipeline integration tests
notebooks/              data download, preprocessing, training, evaluation walkthroughs
requirements.txt
```

## Status

| Component | State |
|---|---|
| `src/data/*` (loaders, transforms, labellers, assembly) | Implemented |
| `src/preprocess.py` | Implemented, passes synthetic smoke test |
| `src/model.py` (MultiSmokeCancerNet, MultiTaskLoss) | Implemented, passes synthetic smoke test |
| `src/train.py` (3-phase Trainer) | Implemented, passes synthetic smoke test |
| `src/evaluate.py` | Implemented, passes synthetic smoke test |
| `src/inference.py` | Implemented, passes synthetic smoke test |
| `tests/*` | All modules covered (1162 tests, `python3 -m pytest tests/ -q`): `test_model.py`, `test_pipeline.py`, `test_loaders.py`, `test_transforms.py`, `test_transforms_inductive_annotation.py`, `test_labellers.py`, `test_assembly.py`, `test_converters.py`, `test_train.py`, `test_splitting.py`, `test_preprocessing.py`, `test_preprocess_split_aware.py`, `test_rare_class.py`, `test_label_mapping.py`, `test_evaluate.py`, `test_inference.py`, `test_subject_balanced_sampling.py`, `test_smoke_imbalance_loss.py`, `test_phase2_imbalance_integration.py`, plus 25 `test_benchmarks_*.py` files (including `test_benchmarks_atomic_io.py`, `test_benchmarks_model_fingerprint.py`, `test_benchmarks_cell_type_provenance.py`, `test_benchmarks_env_versions.py`, `test_benchmarks_imbalance_ablation.py`, and `test_benchmarks_final_evaluation.py`), and Phase 6's `test_domain_losses.py`, `test_source_balanced_sampling_domain.py`, `test_source_eligibility.py`, `test_source_held_out.py`, `test_source_held_out_diagnostics.py`, `test_domain_shift_diagnostics.py`, `test_biological_stability.py`, `test_uncertainty_diagnostics.py`, `test_robustness_report.py`, `test_pathway_bundle.py`, `test_domain_robustness_ablation.py`, `test_ablation_report.py` |
| `src/benchmarks/*` (Phase 1 rigorous benchmarking) | Implemented — see [Benchmarking framework](#benchmarking-framework-phase-1-does-the-neural-model-beat-simple-baselines) — passes a fast synthetic end-to-end CLI run; **not yet run against real merged data**, so no real baseline-vs-neural comparison number exists yet |
| CI | `.github/workflows/tests.yml` runs the full pytest suite (synthetic fixtures only, no dataset downloads) on push to this branch and on PRs into `main` |
| `notebooks/*` | `01_data_download`, `02_preprocessing`, `03_training`, `04_evaluation` all implemented |
| Real data — GSE994, GSE307690 | Downloaded, converted, harmonized, and actually trained on — see [Current results](#current-results-real-data) |
| Real data — GSE136831 (312,928 real cells) | Downloaded and converted with real per-cell donor IDs — not yet re-trained on, see [Next steps](#next-steps) |
| Real data — GSE123352 (176 subjects) | Downloaded, probe IDs mapped to real gene symbols, merged into `microarray_sources` |
| Real data — GSE288003 (23,595 real mouse cells) | Downloaded, `RAW.tar` extracted, both conditions (Con/E-cigs) correctly labelled |
| Real data — TCGA-LUAD/LUSC | Wired in code; blocked on a personal GDC token (not obtained yet) |
| Real data — NLST | Wired in code; blocked on a Data Use Agreement (not obtained yet) |

Every implemented `src/*.py` module (`preprocess.py`, `model.py`, `train.py`,
`evaluate.py`, `inference.py`) has a `__main__` smoke test that runs it
end-to-end on synthetic data — run any of them directly (no arguments) to
sanity-check the pipeline without needing real datasets or trained
checkpoints.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Requires Python 3.10+. Key dependencies: `torch`, `scanpy`, `anndata`,
`harmonypy`, `celltypist`, `pybiomart`, `scikit-learn`.

## Running the smoke tests

```bash
python3 src/preprocess.py   # synthetic AnnData → full preprocessing pipeline → MIL bags
python3 src/model.py        # forward passes + loss computation, shape/gradient assertions
python3 src/train.py        # short synthetic run of all 3 training phases + inference
python3 src/evaluate.py     # cell + subject metrics, interpretability report → checkpoints/evaluation_report.json
python3 src/inference.py    # untrained-model predict_subject/predict_batch/save_results smoke test
```

## Running the full demo

```bash
./scripts/run_demo.sh
```

Installs dependencies, runs the pytest suite, runs every module's smoke test
in order (`preprocess.py` → `model.py` → `train.py` → `evaluate.py` →
`inference.py`), renders every plot in `src/visualize.py` from the artifacts
those steps just wrote (training curves, ROC/PR/calibration curves, smoke-type
confusion matrix, attention-by-cell-type/smoke-type, a sample patient risk
profile), and bundles logs, plots, and metrics into a single `report.docx` and
`report.pdf`. Everything lands in a timestamped `demo_run_<timestamp>/`
directory (gitignored) — nothing here needs real data or a trained checkpoint.

## Running the test suite

```bash
python3 -m pytest tests/ -q
```

Unit tests for every `src/data/*` module (loaders, transforms, labellers,
assembly, converters), plus integration tests running `run_pipeline()` and
`MultiSmokeCancerNet` together on synthetic data.

## How the pieces fit together

See [Where the data comes from](#where-the-data-comes-from) for what to
download and why. Mechanically:

Raw GEO/TCGA/NLST files don't arrive in the shape `src/data/loaders.py`
expects (GEO series matrices carry metadata headers, GSE136831/GSE288003 are
10x-style sparse matrices, TCGA ships one HTSeq count file per case plus a
GDC file manifest, NLST outcomes use different column names). Different
sources also use different gene-ID namespaces — Affymetrix probes (GSE994),
Illumina probes (GSE123352), Ensembl IDs (GSE307690) — none of which overlap
directly, so merging sources with zero shared genes is a real failure mode,
not an edge case. `src/data/converters.py` bridges the format gap;
`src/data/transforms.py::harmonize_gene_ids` bridges the gene-ID gap for
Affymetrix + Ensembl IDs via BioMart. Illumina isn't BioMart-queryable, so
that mapping happens earlier instead — `convert_microarray()` resolves
GSE123352's probe IDs to real gene symbols at conversion time via GEO's own
GPL10558 platform annotation file (`converters.py::_load_probe_to_symbol_map`),
so GSE123352.csv already carries gene symbols by the time it reaches
`harmonize_gene_ids`. `configs/default.yaml` already points at converters'
output paths.

TCGA-LUAD/LUSC supply per-cell malignancy labels (tumor vs. solid-tissue-normal)
and subject-level cancer-positive outcomes — `convert_tcga()` reads the
`sample_type`/`case_id` fields `downloaders.py` saves to `file_meta.csv`
alongside the GDC manifest, and writes both a `_samples_meta.csv` (malignancy +
subject_id, consumed by `load_microarray`) and an `_outcomes.csv` (merged with
NLST outcomes in `run_pipeline`). TCGA cohorts don't carry per-patient smoking
history in this pipeline, so smoke_type defaults to `cigarette` — documented
approximation, consistent with the cigar/dual-use label-transfer caveats in
[ARCHITECTURE.md](ARCHITECTURE.md#3a-smoke-type-classification-head-head-a).

Or walk through the same steps interactively in
[notebooks/01_data_download.ipynb](notebooks/01_data_download.ipynb) and
[notebooks/02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb).

(`python3 src/preprocess.py` with no arguments only runs its synthetic-data
smoke test — it does not read `configs/default.yaml`. Call `run_pipeline()`
directly, as shown in [Where the data comes from](#where-the-data-comes-from),
to process real data.)

Any source missing at conversion time is skipped with a message rather than
failing the whole pipeline — run with whatever subset you already have.

## Running inference on real data

```bash
python3 src/inference.py --h5ad path/to/subject.h5ad --out results.json
python3 src/inference.py --h5ad path/to/subject.h5ad --phase 2 --out results.json   # load Phase 2 checkpoint
python3 src/inference.py --h5ad path/to/subject.h5ad --device cuda --out results.json
```

The H5AD must already be preprocessed by `run_pipeline()` (scaled expression
matrix, `subject_id` and `cell_type_id` in `.obs`), and a matching
`checkpoints/phase{1,2,3}_best.pt` must exist from a prior `train.py` run.

## Configuration

All model/train/data hyperparameters live in [configs/default.yaml](configs/default.yaml),
consumed via `MultiSmokeCancerNet.from_config()` and `Trainer.from_config()`.

## Pathway-aware hierarchical MIL (research candidate)

`src/pathway_hierarchical_mil.py` implements an optional second
architecture, code identifier `pathway_hierarchical_mil`: a pathway/gene-
module-aware cell encoder feeding a two-level (cell -> cell-type -> subject)
gated-attention multi-instance model with smoke-type and cancer-risk heads.
It is a research candidate — not described here or elsewhere in this
repository as clinically validated, superior to `MultiSmokeCancerNet`, or
scientifically novel. It is disabled by default
(`model.pathway_hierarchical_mil.enabled: false`) and does not change
`MultiSmokeCancerNet`'s behavior.

The pathway/gene-module layer requires an explicit, versioned module
definition. Supply a GMT-style file (`module<TAB>description<TAB>GENE1<TAB>GENE2...`)
via `model.pathway_hierarchical_mil.gene_modules.path`; module membership is
aligned to the same ordered gene list the preprocessing artifact was fit
with, genes not in that list are dropped from membership (never leaked back
into preprocessing), and a module falling below
`minimum_genes_per_module` is dropped or rejected per
`empty_module_policy`. No pathway resource is downloaded automatically and
none is committed to this repository. With no file supplied, the model
refuses to construct outside test/synthetic mode — a deterministic,
clearly-labelled synthetic module scheme
(`GeneModuleCollection.synthetic`) is available only when
`gene_modules.allow_synthetic_modules` is explicitly set for a synthetic
workflow.

Cell-type-aware hierarchical pooling produces two levels of gated-attention
diagnostics (per-cell, within cell type; per-cell-type, within subject),
both masked so padded cells and unobserved cell types contribute exactly
zero weight. Smoke and cancer labels each carry an explicit known/unknown
mask; an unknown label never contributes to that task's loss and is never
treated as a negative outcome. The model integrates with the existing
Phase 4 bundle system (`benchmarks/bundle.py`): its checkpoint identity is
bound to both the preprocessing artifact's fingerprint and a separate gene-
module fingerprint, and loading a bundle built for a different module set
or preprocessing artifact fails loudly rather than silently.

Run its test suite (unit coverage for the gene-module contract, masked
pathway encoder, hierarchical attention masking/normalization, multitask
masking, optional source/species conditioning, and checkpoint/bundle
identity, plus a synthetic end-to-end training-loop integration test) with:

```
pytest tests/test_pathway_hierarchical_mil.py tests/test_pathway_hierarchical_mil_integration.py \
       tests/test_pathway_hierarchical_cv_integration.py
```

The model is registered as `pathway_hierarchical_mil` wherever
`benchmarks/runner.py` selects a model, and participates in the same
grouped-subject nested cross-validation, out-of-fold prediction, and
final-development-fit protocol every other candidate uses (see
ARCHITECTURE.md §15.8):

```
python -m benchmarks.runner --synthetic --fast --task smoke \
    --models majority pathway_hierarchical_mil
python -m benchmarks.runner --synthetic --fast --task cancer \
    --models prevalence pathway_hierarchical_mil
python -m benchmarks.runner --synthetic --fast --task smoke \
    --models majority pathway_hierarchical_mil --pathway-hierarchical-ablation
```

A real (non-synthetic) run requires a supplied gene-module file
(`model.pathway_hierarchical_mil.gene_modules.path`) — without one, the
model refuses to construct with an actionable configuration error rather
than silently substituting the synthetic diagnostic scheme.

**Scope note.** A model-specific calibration fit (the existing generic
post-hoc calibrator is reused unchanged instead) remains unimplemented. The
domain-adversarial training head mentioned here in earlier phases is now
implemented — see "Phase 6 — Domain robustness and biological validation"
below. All results produced against synthetic data anywhere in this
repository (including this model's tests) are software-correctness checks,
not scientific evidence, and no result from this model has been produced
against real data or the frozen test set.

## Phase 6 — Domain robustness and biological validation

Phase 5 established that `pathway_hierarchical_mil` can be trained and
evaluated through the same protocol as every other candidate. Phase 6 asks a
different question: does whatever it learns on the development sources
carry over to a source it never saw during development? This section
distinguishes that question sharply from "did the model fit the data it was
given."

### Source-held-out evaluation vs. the frozen final test

These are two different, non-interchangeable things:

- **The frozen final test** (`benchmarks/test_guard.py`) is this
  repository's one-shot, durably-guarded evaluation against subjects held
  out before any development work began. It may be evaluated exactly once
  per scientific identity, ever.
- **Source-held-out evaluation** (`benchmarks/source_held_out.py`) is a
  repeatable, development-only DIAGNOSTIC: for each dataset source present
  in the train+val pool, train on every other source and evaluate on the
  held-out one. It never reads `context.test_bags`, never resolves
  `split_manifest.test_subjects`, and never acquires or references
  `FrozenTestGuard` — running it any number of times, for any number of
  sources, has no effect on the frozen test's one-shot guarantee, and it
  does not need or use `--disable-frozen-test-guard`.

Held-out-source results are development evidence about robustness to
acquisition-source shift. They are not a substitute for, and must not be
described as equivalent to, the frozen test result.

### Why source shift matters here

Single-cell/microarray transcriptomics datasets differ by processing
batch, platform, donor population, and library preparation — a model that
distinguishes smoke-exposure or cancer status well within one source can be
partly (or entirely) relying on structure specific to that source's
acquisition pipeline rather than the underlying biology. Source-held-out
evaluation is the standard way to get development-only evidence about
whether that has happened, though a small number of sources (this
repository currently has at most a handful of distinct GEO/TCGA sources per
task) limits how strong that evidence can ever be — see "Honest limitations"
below.

### Running it (synthetic, software-only)

```
python -m benchmarks.runner --synthetic --fast --task smoke \
    --models majority pathway_hierarchical_mil --leave-one-source-out
python -m benchmarks.runner --synthetic --fast --task cancer \
    --models prevalence pathway_hierarchical_mil --domain-strategy coral --coral-weight 0.1
python -m benchmarks.runner --synthetic --fast --task cancer \
    --models prevalence pathway_hierarchical_mil --domain-robustness-ablation
```

`--leave-one-source-out` is the pre-existing (Phase 1) Task-A/classical-
baseline-only diagnostic (`benchmarks/ood.py`) and is unchanged.
`--domain-strategy {erm,source_balanced,coral,mmd,domain_adversarial}` (with
`--source-balanced`, `--coral-weight`, `--mmd-weight`,
`--domain-loss-weight`, `--gradient-reversal-lambda`) and
`--domain-robustness-ablation` run the richer, both-task, all-model-kind
protocol in `benchmarks/source_held_out.py` and
`benchmarks/domain_robustness_ablation.py`, writing
`<run_dir>/metrics/domain_robustness_{smoke,cancer}.json` (and, if
`--robustness-report <path>` is given, a copy at that path). Every one of
these commands above has actually been run against the synthetic context
during this phase's development — none of the numbers they print are
fabricated — but a synthetic run proves the software runs correctly, not
that the model generalizes across real acquisition sources.

### Source eligibility

Not every dataset source is automatically eligible to serve as a held-out
external-domain evaluation set. `benchmarks/source_eligibility.py` assigns
each source one of: `eligible`, `not_evaluable`, `insufficient_classes`,
`insufficient_outcomes` (still eligible — only AUROC/AUPRC are undefined),
`species_mismatch`, `controlled_access_unavailable`, or
`excluded_by_policy`. An ineligible source is recorded with its reason and
skipped — it never crashes the sweep over the other sources, and an
undefined metric is reported as undefined, never as `0.0`. Unknown-species
metadata always resolves to `species_mismatch`, never to
assumed-compatible.

### Model/strategy support matrix, and requested vs. applied strategy

| Candidate kind | Supported strategies |
|---|---|
| Classical baseline (`prevalence`, `logistic`, `random_forest`, ...) | `erm` only |
| Pooling-based (non-module) MIL (`neural`, `mean_mil`, `max_mil`, `attention_mil`) | `erm` only |
| `pathway_hierarchical_mil` (module-based) | `erm`, `source_balanced`, `coral`, `mmd`, `domain_adversarial` |

Only `pathway_hierarchical_mil` has a training-time attachment point for
anything other than plain ERM — `benchmarks/candidate_registry.py`'s
`SUPPORTED_STRATEGIES_BY_KIND`/`resolve_strategy_application` is the single
canonical place this matrix is expressed. This matters because the general
source-held-out path (`run_cancer_source_held_out`) does MODEL SELECTION:
for each held-out source, it picks whichever candidate (classical, pooling
MIL, or pathway) had the best development-only OOF AUROC — and that winner
is not always `pathway_hierarchical_mil`, even when a non-ERM strategy was
requested for the run. A classical or pooling-MIL winner cannot receive
CORAL/MMD/source-balanced/domain-adversarial training at all, regardless of
what was requested.

To keep this honest, every robustness report distinguishes:

- **`requested_strategy`** — what the caller asked for.
- **`strategy`** — what the winning candidate ACTUALLY applied. Equals
  `requested_strategy` only when the winner's registry-derived candidate
  kind supports it; otherwise it is always `"erm"`, because that is
  genuinely what happened.
- **`strategy_applicable`** / **`strategy_applicability_reason`** — whether
  the requested strategy could be applied, and why/why not.

A report can therefore legitimately read `requested_strategy: "coral"`,
`strategy: "erm"`, `strategy_applicable: false` — meaning: a CORAL run was
requested, a classical baseline won that source's model-selection sweep,
and the reported metric reflects an ordinary ERM fit, not CORAL. Such a
result is not, and must never be treated as, evidence for CORAL.
`validate_robustness_report` enforces the whole consistency: a non-module
winner's `strategy` must be `"erm"`; a `pathway_hierarchical_mil` winner's
`strategy` must equal `requested_strategy` (its training path always
receives and genuinely applies whatever was requested); and only a report
whose `strategy` (applied) is actually `"domain_adversarial"` may carry a
real domain-head/vocabulary identity.

### Fixed-model domain-robustness ablation vs. general model selection

`--domain-robustness-ablation` (`benchmarks/domain_robustness_ablation.py`)
compares training STRATEGIES, not candidate MODELS — for the cancer task it
therefore fixes every variant (ERM included) to the single candidate
`pathway_hierarchical_mil`, the only one capable of running every declared
strategy, rather than re-running the classical-vs-MIL-vs-pathway
model-selection sweep per variant. A caller-supplied `model_names`/
`--models` list has no effect on which candidate this ablation evaluates
for the cancer task (it is still recorded, for provenance, in the report's
`requested_model_names` field) — mixing "which model is best" with "which
strategy is best" would let a classical baseline that merely won one
variant's OOF sweep silently stand in for every strategy variant, which is
exactly the attribution bug this fixed-candidate design prevents
structurally. General classical-vs-MIL-vs-pathway model comparisons belong
to the plain `--domain-strategy` path instead. The paired comparison against
ERM additionally requires, per (source, seed) pair, that both sides'
winning candidate match the ablation's declared `candidate_name`, that the
variant side's `strategy_applicable` is true, and that preprocessing/
module/split identity fingerprints agree between the two sides — a
mismatch on any of these excludes that pair from the comparison (counted
and reasoned in `excluded_pairs`) rather than silently averaging
incompatible results. Task A's ablation has no such fixed-candidate
concern: only ERM is ever evaluated for smoke, so `candidate_name` is a
structured `not_applicable` value there and the ERM variant still sweeps
`model_names` as a model-SELECTION comparison, not a strategy comparison.

### Domain-robust training strategies

`benchmarks/domain_losses.py` implements, for `pathway_hierarchical_mil`
only (the one architecture with a subject-embedding attachment point —
`HierarchicalMILOutput.subject_embeddings`):

- **`erm`** — unchanged empirical risk minimization; the default.
- **`source_balanced`** — source -> subject -> cell sampling
  (`data/source_sampling.py`), so a source with more subjects does not
  dominate training purely because of subject count.
- **`coral`** — development-only CORAL covariance-alignment penalty across
  development sources' subject embeddings (Sun & Saenko 2016's formula).
- **`mmd`** — development-only RBF-kernel maximum-mean-discrepancy penalty
  across development sources' subject embeddings (Gretton et al.'s
  two-sample test statistic), median-bandwidth heuristic.
- **`domain_adversarial`** — a gradient-reversal layer
  (Ganin & Lempitsky 2015) feeding a development-source classifier head;
  verified by an explicit test that the reversed gradient is the exact
  negation of the un-reversed one.

Every regularizer is weighted, defaults to weight `0.0`/disabled, and is
computed only from development-source subject embeddings — a held-out
source's representation never enters any of these loss terms (structurally
true: the loss functions only ever see whatever `sources` list the caller
passes, and the caller — `source_held_out.py` — never includes the held-out
source's subjects in a development fit). None of these are described as
making the model "domain invariant" — they are training-time penalties
whose actual effect is reported empirically per source in the robustness
report, never assumed.

### Nested source-aware hyperparameter selection

The final development-pool fit's hyperparameters (including, when
applicable, the domain-robustness strategy's own weights) are selected via
`benchmarks/hyperparameter_search.py`'s existing nested, per-fold-refit
selection restricted to development-source subjects only — the held-out
source's subjects are excluded from the subject pool passed into selection
at the call site, not merely down-weighted afterward.

### Calibration and uncertainty under source shift

Calibration and the decision threshold are fit exclusively from
development out-of-fold predictions (`benchmarks/calibration.py`, reused
unchanged) and applied to the held-out source's raw probabilities exactly
once per source (a fresh `FrozenThresholdPolicy` per held-out source — this
one-shot-per-policy discipline is unrelated to, and does not touch, the
real frozen-test guard). `benchmarks/uncertainty.py` adds predictive
entropy, MC-dropout dispersion (explicitly not a calibrated confidence
interval), and an optional abstention-coverage diagnostic whose threshold
is selected from development data only; abstention is reported alongside,
never instead of, the full-population held-out-source metric.

### Domain-shift diagnostics

`benchmarks/domain_shift.py` reports label-free diagnostics computed from
development-fitted features only: gene/module coverage, subject/cell
composition, and distribution-shift distance (centroid, energy, CORAL, MMD)
between development and held-out-source feature distributions, plus an
optional source-predictability classifier (how much source identity remains
encoded in a representation, compared against a label-permutation
baseline). A high source-predictability score is reported as a fact about
the representation, not automatically as model failure; a low one is not
proof of invariance.

### Biological stability — synthetic-module-scoped

**No real gene-set (GMT) resource ships with this repository.** Every
concrete module-stability or attention-stability number this phase has
actually produced (in tests, in CI, in the synthetic CLI runs above) comes
from `GeneModuleCollection.synthetic()` — the same deterministic,
non-biological placeholder scheme Phase 5 already used — and is therefore a
**software sensitivity diagnostic, not a biological-plausibility finding**.
`benchmarks/biological_stability.py`'s real-mode entry point
(`require_real_modules`) refuses to run against a synthetic module source
at all, so a real run cannot silently produce a "biological stability"
result that is actually synthetic. What it measures once a real module
resource is supplied: module-ablation "model-weighted contribution" scores
and their rank stability/top-k overlap across folds/seeds, cell-type
attention distribution and its stability, an attention-vs-cell-type-
abundance correlation against a permutation null (is attention just
tracking which cell type is most numerous?), and a cell-order permutation-
invariance check (verified in this repository: shuffling cell order within
a bag changes the predicted logit by less than `1e-3`, consistent with the
pooling architecture's masked-attention construction). Attention weights
are reported as "pooling weight" / "model-weighted contribution," never as
biological importance, and a performance drop after removing a module is
reported as a model-sensitivity finding, never as evidence that module is
biologically causal.

All three diagnostic families above (domain-shift, uncertainty/abstention,
and biological stability) are wired directly into every applicable
per-source robustness report, not just implemented as standalone functions
— `report.domain_shift`/`report.uncertainty`/`report.biological_stability`
are populated for every held-out source a candidate was actually fit and
evaluated for, and stamped `not_applicable`/`not_evaluable` with an explicit
reason when a diagnostic genuinely does not apply (e.g. biological-stability
diagnostics for a non-pathway candidate, MC-dropout for a non-neural one).
The biological-stability section additionally runs a genuinely
separately-fitted label-permutation null (a second full development-pool
fit on label-permuted outcomes, never a relabeled copy of the real ranking)
and a cell-type-label permutation check alongside the existing matched-
size-random-module control and attention-vs-abundance permutation null.

### Reading the robustness report

`benchmarks/robustness_report.py` defines a versioned JSON schema
(`schema_version`, always stamped `development_only: true` and
`frozen_test_accessed: false`), atomic writes with reload verification, and
cross-source aggregation (`benchmarks/robustness_report.py::
aggregate_source_reports`) that reports the **worst-source** result
explicitly rather than only a pooled average, alongside the macro-source
average, a separate (secondary) subject-weighted average, median, and IQR.
An ineligible source is listed with its reason, never silently dropped from
the aggregate's source count.

### Honest limitations

- No real GMT gene-set resource is committed to or referenced by this
  repository — every biological-stability number produced so far is a
  synthetic-module software diagnostic (see above).
- Task A (smoke-type) source-held-out evaluation covers classical
  baselines and `pathway_hierarchical_mil`'s plain-ERM fit; it does not
  extend to the pooling-based MIL models or to `MultiSmokeCancerNet`'s
  cell-level `Trainer` curriculum — that would require a materially larger
  rework of the per-source refit flow. `--domain-strategy` values other than
  `erm` are therefore not evaluated for Task A and are recorded as
  `not_evaluable` by the ablation runner rather than silently skipped.
  Task B (cancer) source-held-out evaluation covers classical baselines,
  the pooling-based MIL models, and `pathway_hierarchical_mil` as candidates
  in the same model-selection sweep, but only `pathway_hierarchical_mil` can
  ever actually apply a non-ERM domain-robustness strategy — see "Model/
  strategy support matrix" above. A classical or pooling-MIL winner under a
  requested non-ERM strategy is reported honestly (`strategy: "erm"`,
  `strategy_applicable: false`), never mislabeled as having run that
  strategy.
- With only two dataset sources in the synthetic CI context (and a small
  number of real GEO/TCGA sources in the real dataset manifest), the
  aggregate cross-source statistics (median, IQR, worst/best-source) have
  very little statistical power — they are reported honestly rather than
  suppressed, but should not be read as strong evidence either way.
- Domain-adversarial training's gradient-reversal direction is unit-tested
  directly; its practical effect on real-source generalization has not
  been evaluated against real data in this phase.
- No claim of domain invariance, clinical validity, biomarker discovery, or
  state-of-the-art performance is made anywhere for this phase's work — see
  ARCHITECTURE.md §16 for the full scientific-claims policy this repository
  follows.
- Source eligibility prefers the canonical dataset manifest
  (`data/manifest.py`) for species/controlled-access metadata whenever a
  source is declared there, and raises `SourcePolicyDriftError`
  (`benchmarks/source_eligibility.py::resolve_source_policy`) if a
  caller-supplied `species_by_source`/`controlled_access_sources` override
  contradicts it; a source the manifest does not declare (e.g. a synthetic
  test source) still resolves from the caller-supplied dicts unchanged.
- `PathwayHierarchicalAdapter.save_bundle`/`load_bundle` persist and
  restore a fitted candidate's complete state through a Phase-4-style
  bundle (`benchmarks/bundle.py`) — model weights, gene modules, the
  domain-robustness configuration, and (when built) the domain-adversarial
  head's own weights and fixed development-source vocabulary all round-trip
  and are cross-checked on load; an altered checkpoint, a missing domain
  head where the recorded strategy requires one, or a vocabulary mismatch
  all raise rather than silently loading a partial or inconsistent state.
- The domain-robustness ablation runner accepts multiple seeds
  (`run_domain_robustness_ablation(..., seeds=[...])`) and reports a
  paired comparison against ERM per (source, seed) pair — mean/median
  difference, win/tie/loss counts, and an explicit `insufficient_evidence`
  status when fewer than two common (source, seed) pairs were evaluated,
  never a point estimate presented without its sample size.
- Task A's smoke-type ground truth for source-held-out evaluation and
  candidate selection is restricted to `smoke_type_known=True` cells only
  (unknown and, unless `data.weak_labels.enabled=true`, weak-proxy cells
  never contribute); a subject whose own verified cells disagree on
  `smoke_type` raises rather than being resolved by majority vote. Each
  source-held-out report's `label_state` field records verified/unknown/
  weak-proxy-only/conflicting/excluded-by-policy subjects, the verified
  class distribution, and a weak-label policy fingerprint. Missing
  `smoke_type_known` metadata is a typed `LabelSchemaError`, never silently
  treated as "every label is verified".
- The robustness report schema is versioned `3.0`: every report — including
  ineligible/not-evaluated branches — carries all twelve identity fields
  (`dataset_manifest_fingerprint`, `source_policy_fingerprint`,
  `source_split_manifest_fingerprint`, `preprocessing_fingerprint`,
  `gene_list_fingerprint`, `module_fingerprint`, `model_fingerprint`,
  `domain_head_fingerprint`, `domain_vocabulary_fingerprint`,
  `calibration_fingerprint`, `threshold_policy_fingerprint`,
  `environment_fingerprint`) plus `seed`, plus (schema v3, new) the
  requested-vs-applied strategy fields `requested_strategy`, `strategy`
  (applied), `strategy_applicable`, `strategy_applicability_reason` — see
  "Model/strategy support matrix" above. A field that is genuinely
  inapplicable (e.g. a classical baseline's module fingerprint) is the
  structured `{"status": "not_applicable", "reason": ...}` value — never a
  bare `None` — and `validate_robustness_report` rejects a bare `None`, a
  malformed (non-SHA-256) hash, an `evaluated=True` report missing a
  mandatory model/preprocessing/gene identity, a non-module candidate whose
  applied `strategy` is not `"erm"`, a module-based candidate whose applied
  `strategy` disagrees with `requested_strategy`, a report whose applied
  `strategy` is `"domain_adversarial"` but is missing its domain-head/
  vocabulary identities, or a calibration-bearing report missing its
  calibration/threshold identities. `validate_report_fingerprint_unchanged`
  detects any post-write tampering with a report's own fields.
  `_environment_fingerprint()` raises a typed `EnvironmentSnapshotError`
  rather than silently recording `None` when a core package's version
  cannot be collected. The domain-robustness ablation report schema is
  separately versioned `2.0` (bumped for the `candidate_name`/
  `requested_model_names` fields and the per-variant applied-strategy/
  candidate consistency checks described above).
- Source-aware strategies (CORAL, MMD, source-balanced sampling,
  domain-adversarial training, source-predictability diagnostics) reject a
  blank, placeholder (`""`, `"unknown"`, `"none"`, `"nan"`, ...), or
  cross-subject-conflicting `dataset_source` value with a typed
  `MissingSourceProvenanceError`/`CrossSourceSubjectConflictError` rather
  than silently mapping it to a literal "unknown" bucket.
- `resolve_source_policy` folds manifest-declared assay mode, label-policy
  version, weak-label-field status, cohort role (real vs. synthetic
  fixture), and exclusion policy into the SAME dict every caller already
  fingerprints in full as `source_policy_fingerprint`. `reference_assay_mode`
  rejects a source whose declared assay disagrees (e.g. bulk TCGA data can
  never enter this single-cell protocol) via the (previously unused)
  `ASSAY_MISMATCH` eligibility status.
- Two additional null/perturbation controls are wired into
  `biological_stability`: `gene_module_permutation_null` (same random
  column permutation applied to every module's membership mask — preserves
  module sizes and the ordered gene universe exactly, destroys the real
  gene<->module association) and `within_gene_expression_permutation_null`
  (each gene's expression independently reshuffled across every valid cell
  in the diagnostic bag set — preserves that gene's own marginal
  distribution exactly, reads no subject_id or label). `cross_run_stability`
  reports `insufficient_evidence` by default and only becomes `evaluated`
  when a caller supplies `stability_extra_seeds`/`extra_seeds`, which
  trigger GENUINE independent refits (never repeated calls to one already-
  fitted model) whose module-ablation rankings are compared via
  `module_ranking_stability`.
- Task A's multi-class uncertainty/abstention diagnostic
  (`smoke_uncertainty_report`) selects its development abstention threshold
  from GENUINE out-of-fold probabilities — `_smoke_candidate_dev_score`'s
  grouped-CV sweep now returns `oof_by_subject` (each verified development
  subject predicted only by the one fold it was held out of, columns
  re-aligned to the full class vocabulary even when a fold's training data
  missed a class) for whichever candidate wins selection, and that is what
  drives threshold selection — never the final dev-pool-fitted model's
  in-sample predictions on its own training pool. Full-population macro-F1
  remains primary; retained-subset macro-F1 is secondary. Held-out-source
  labels and expression never influence the threshold.
- The robustness report schema's `module_fingerprint` requirement is now
  candidate-kind-aware (`is_module_based_candidate`, stamped by the caller):
  an evaluated classical baseline or non-module MIL model must record a
  structured `not_applicable` module identity; only `pathway_hierarchical_
  mil` must record a real module hash. `domain_head_fingerprint`/
  `domain_vocabulary_fingerprint` remain required only for
  `strategy=domain_adversarial`; `calibration_fingerprint`/
  `threshold_policy_fingerprint` remain required only when a report's
  `calibration` block is non-empty.
- Every per-source report is validated (`validate_robustness_report` +
  `validate_report_fingerprint_unchanged`) before it is allowed to enter an
  aggregate (`build_aggregate_report`, `run_domain_robustness_ablation`) or
  reach disk — this is enforced by the production functions themselves,
  not left to tests to call manually. `validate_aggregate_report` checks
  the aggregate's own schema/stamps and recursively validates every nested
  per-source report; `write_aggregate_report` mirrors
  `write_robustness_report`'s atomic-write-plus-reload-and-validate
  contract for the aggregate shape.
- `gene_space_compatibility` no longer reports a self-vs-self "100%
  compatible" placeholder. This pipeline unifies every source onto one
  shared gene space at ingestion time, before any per-source raw gene
  panel would even be distinguishable, so no current caller has a real
  held-out raw gene panel to compare against — the function now returns a
  structured `not_evaluable` status (`"reason": "raw source-specific gene
  contract unavailable"`) by default, and performs a real missing/
  unexpected/duplicate-mapping/coverage computation whenever a caller does
  supply one (tested directly against synthetic partial gene panels).
- `GeneModuleCollection` (`pathway_hierarchical_mil.py`) records
  `source_name`/`source_version` provenance and a content fingerprint, but
  does not yet carry a full real-module provenance schema (organism,
  namespace, mapping-policy, license, checksum). Rather than emit
  `scope: real_module_sensitivity_analysis` without that contract,
  `cancer_biological_stability_report` now explicitly REJECTS a non-
  synthetic module source (`status: "unsupported"`,
  `scope: "unsupported_real_module_analysis"`) — real-module biological-
  stability analysis is disabled, not silently under-documented. Synthetic-
  module results remain `scope: "software_diagnostic_only"`.
- `dataset_manifest_fingerprint` is now required to be a real hash for
  every EVALUATED report (`_REQUIRED_WHEN_EVALUATED`) — it can no longer
  default to `not_applicable` just because no caller threaded a manifest
  through. `runner.py`'s `_dataset_manifest_entries_for_run` builds an
  explicit synthetic manifest for `--synthetic` runs (never exempted from
  the requirement) or loads `configs/datasets.yaml` for real runs, raising
  `MissingDatasetManifestError` if that seed is absent — never silently
  falling back to `not_applicable` for a real run. Ineligible/non-evaluated
  reports get the same real fingerprint whenever a manifest was supplied
  (computed once, independent of any one source's eligibility);
  `not_applicable` remains acceptable there only when no manifest was ever
  supplied.
- `is_module_based_candidate` is no longer trusted verbatim from the
  caller. `benchmarks/candidate_registry.py` derives candidate kind
  (classical baseline / non-module MIL / pathway module-based MIL) from
  the canonical `baselines.py`/`mil_registry.py` registries and raises a
  typed `UnknownCandidateNameError` for any unrecognized model name;
  `validate_robustness_report` recomputes this from the report's own
  `model` field and rejects any report whose declared
  `is_module_based_candidate` disagrees with the registry-derived kind.
- Task A's OOF coverage is now enforced, not just collected.
  `_smoke_candidate_dev_score` validates every OOF probability row
  (correct dimension, finite, sums to 1) and rejects a fold that predicts
  a subject it also trained on; after the sweep, `_oof_coverage_summary`
  checks whether every verified-label development subject received
  exactly one OOF prediction and forces the candidate's score to `None`
  (never selectable) if not. The winning candidate's coverage counts and
  fingerprints (subject-set/fold-assignment/class-order — never a raw
  subject-ID list) are persisted in `comparisons` and in
  `smoke_uncertainty_report`'s output.
- Domain-robustness ablation output now has its own versioned schema
  (`benchmarks/ablation_report.py`, `ABLATION_REPORT_SCHEMA_VERSION`):
  `build_ablation_report` validates every nested per-source report
  recursively (the same schema-v2 choke point `build_aggregate_report`
  uses) plus the per-variant/per-seed structure and paired-comparison
  shape before stamping a whole-report `aggregate_fingerprint`;
  `write_ablation_report` atomically writes, reloads, and re-validates.
  `runner.py` persists ablation output through this writer instead of the
  generic `write_json`.

## Phase 7 — real-world evidence and clinical-readiness framework

Phases 1-6 established leakage-safe splitting, imbalance handling, label
semantics, reproducible preprocessing artifacts, pathway-hierarchical MIL,
and domain-robustness evaluation — all validated against synthetic
fixtures and, where noted above, one training-set real-data run. Phase 7
adds a separate framework (`src/evidence/`) for stating, precisely, what
would count as real-world clinical evidence, and for refusing to report
evidence that does not exist.

**What this phase adds:**
- `src/evidence/evidence_contract.py` — a versioned evidence-report schema
  with seven evidence levels (`synthetic_software_validation` through
  `regulatory_evidence`), 29 required identity fields per report, and a
  structured `not_evaluable(...)` object that is the only sanctioned way
  to represent missing evidence (never `0`, `False`, an empty metric dict,
  or a "successful" result with a caveat attached).
- `configs/cohorts.yaml` / `src/evidence/cohort_registry.py` — a canonical
  registry of every cohort this repository knows about (GSE136831,
  GSE288003, GSE123352, GSE307690/CANUCK, TCGA-LUAD/LUSC, NLST), recording
  per-cohort what it can and cannot support: GSE136831's COPD field is a
  weak exposure proxy, not a verified smoke label; GSE288003 is mouse and
  stays species-separated; the bulk/pseudo-bulk cohorts (GSE123352,
  GSE307690, TCGA-LUAD/LUSC) cannot enter single-cell MIL without a bulk
  pipeline this repository does not have; NLST is controlled-access and
  unavailable without an authorized local dataset. The registry is
  cross-checked against `configs/datasets.yaml` for contradictions.
- `src/evidence/eligibility.py` — deterministic eligibility gates (Step 6)
  that classify every (task, cohort) pair as impossible, exploratory-only,
  or eligible for internal/external evaluation, from real on-disk counts
  only.
- `src/evidence/audit.py` — a **read-only** audit CLI:
  ```
  PYTHONPATH=src python -m evidence.audit \
      --config configs/default.yaml --cohort-config configs/cohorts.yaml \
      --output artifacts/evidence/data_audit.json
  ```
  It never trains, fits preprocessing, or downloads anything; it reports
  which datasets have local files, which cohorts are controlled-access and
  unauthorized, and a per-task eligibility table.
- `src/evidence/clinical_readiness.py` + `src/evidence/clinical_manifest.py`
  + `configs/clinical_readiness.yaml` + `docs/CLINICAL_READINESS.md` — a
  22-dimension staged clinical-readiness assessment. `assess_clinical_readiness()`
  can only report `clinically_not_ready` unless every mandatory dimension
  is independently `complete` with a real evidence reference, and
  `guard_clinical_claim()` requires a real, Ed25519-signed
  `ClinicalEvidenceManifest` (`evidence/clinical_manifest.py`) whose
  signature verifies against a caller-configured approved-public-key
  allow-list and whose referenced evidence reports independently validate
  — not a caller-controlled boolean. This repository's default
  configuration ships no approved public key, so by default no manifest
  can unlock a clinical-readiness claim regardless of its content.
- `src/evidence/tracks.py` — Track A (smoke classification), Track B
  (malignancy classification), Track C (subject-level cancer prediction),
  each with a real registry-checked path (honestly `not_evaluable` for
  every task in this environment) and a synthetic-fixture path that runs
  the real metric-computation code end to end.
- `src/evidence/external_validation.py` — a poison-object sentinel for
  external-cohort data plus a gate that structurally cannot release a
  cohort for external validation before development is explicitly frozen,
  and that tells apart "no eligible external cohort exists" from "this is
  an internal split masquerading as external validation."
- `tests/test_evidence_leakage_isolation.py` — corruption-isolation tests
  proving the evidence framework's development-only code paths cannot
  read a held-out/test partition, even adversarially.
- `src/evidence/candidate_comparison.py` — identical-partition comparison
  across classical baselines and MIL-kind candidates for a task, reusing
  the existing cross-validation/final-evaluation machinery so every
  candidate shares one fold assignment, preprocessing artifact, and label
  mapping by construction.
- `src/evidence/uncertainty.py` — repeated grouped-resampling uncertainty
  reporting, structurally confined to development data (its one input type
  requires the literal role `"development"`), with subject-level bootstrap
  CIs and a paired candidate comparison that is explicitly labeled
  descriptive, never a formal equivalence test.
- `src/evidence/subgroups.py` — subgroup/fairness diagnostics over the
  subgroup dimensions this repository genuinely has a data source for
  (cohort source, exposure type, disease status, assay platform, species);
  it refuses to fabricate a dimension (e.g. sex, age, race/ethnicity) that
  has no field anywhere in this repository's data model, and it refuses to
  render any summary claiming fairness has been established.
- `src/evidence/calibration.py` — frozen calibration (Platt/isotonic) and
  thresholding fit only on development out-of-fold predictions; applying
  the frozen artifact to new probabilities takes no fitting parameters at
  all, so it cannot be refit on test data.
- `src/evidence/artifact_bundle.py` — an atomic, checksummed, immutable
  writer/reader for `artifacts/evidence/<run_id>/` run directories; a run
  directory already marked complete can never be written into again.
- `src/evidence/run_identity.py` — real, versioned identity primitives for
  non-fixture evidence reports: a genuine UTC timestamp
  (`utc_now_iso`/`validate_real_timestamp` — placeholders like the Unix
  epoch are rejected), a complete split manifest fingerprinted from the
  actual train/validation/development-holdout partitions, seed,
  stratification and label-mapping policy (`build_split_manifest` /
  `split_manifest_fingerprint`, with `sanitize_split_manifest` producing a
  publication-safe, count-only view), a real environment snapshot
  (`build_environment_snapshot`), and real fitted-model/preprocessing
  fingerprints for the bulk pipeline (`bulk_model_fingerprint` /
  `bulk_preprocessing_fingerprint`, reusing
  `benchmarks/model_fingerprint.py`'s whitelist-based canonicalization of
  actual fitted state — never predictions, never a descriptive-label
  fallback).
- `src/evidence/development.py` — the canonical implementation behind
  `evidence.runner development`/`internal-test`/`external-test`:
  `run_development()` runs the real GSE123352 bulk-pipeline evaluation (or
  a single-repeat estimate for any other cohort/task) plus
  `run_gse123352_repeated_development()` for the repeated, seed-averaged
  development estimate; `run_internal_test()`/`run_external_test()` are
  honest gates that always return a specifically-reasoned `not_evaluable`
  (no cohort has ever had a frozen internal-test partition created and
  guarded; no cohort carries `role_eligibility=[external_validation]`) —
  never a generic stub and never a fabricated result.
- `src/evidence/publication.py` — derives a sanitized, non-identifying
  summary (aggregate metrics, confusion matrix, fingerprints,
  configuration, checksums — never subject/sample IDs, row-level
  predictions, or local paths) from a validated private run bundle, writes
  it to `evidence/published/<run_id>/summary.json`, and reloads +
  re-validates it before returning.
- `src/evidence/runner.py` — the `python -m evidence.runner` umbrella CLI
  (`audit`, `inspect`, `validate`, `clinical-readiness`, `development` are
  fully implemented; `internal-test`/`external-test` are honest,
  specifically-reasoned gates — see `evidence/development.py` above).

**Current audited data availability in this environment:** GSE136831 and
GSE123352 are downloaded locally (`data/raw/`, gitignored); the other five
registered cohorts (GSE288003, GSE307690/CANUCK, TCGA-LUAD/LUSC, NLST) are
not, and NLST's controlled-access authorization check is `false` (no
`NLST_DATA_ROOT`). `python -m evidence.audit` reports
`partial_local_data_present` accordingly, never a fabricated eligible
count for the five cohorts that are still absent.

**Evidence status, stated plainly (current, canonical — supersedes PR #15's
stale description and any number typed directly into a document rather than
loaded from a validated evidence artifact):**
- **GSE123352 verified binary bulk smoke-history result: exploratory,
  development-only, real.** `src/data/bulk_pipeline.py` parses a real,
  independently-verified per-subject **lifetime ever-versus-never
  cigarette-smoking history** (not "at time of sampling" — a former smoker
  is correctly included in the "ever" class) from the real downloaded
  GSE123352 series matrix. Subject identity is independently verified per
  sample (GEO `Sample_title`'s `patient_<N>` field, parsed and validated
  one-to-one against GSM accessions — see `data/converters.py::
  _infer_subject_id_column`); a sample whose subject identity cannot be
  verified is excluded, never trusted. Strict boolean parsing
  (`data.bulk_pipeline.parse_strict_bool`) means a malformed or ambiguous
  `smoke_type_known` value is rejected outright, never silently coerced. A
  single 70/30 split (124 train / 50 test) achieved **macro-F1 0.653,
  balanced accuracy 0.700** on 50 held-out subjects (validated
  2026-07-23, commit `c28eeee166ae98b3198ede46bcfc426bdd62a4f7`) — this
  remains a labeled, single-split **exploratory** finding, not the primary
  estimate; the split changed from an earlier pre-remediation run because
  it is now correctly keyed on verified donor subject IDs rather than raw
  GSM accessions (see blocker 8 below).
  `evidence.development.run_gse123352_repeated_development()` adds a
  repeated grouped-development-holdout protocol across 8 predeclared seeds
  (`configs/evidence.yaml`'s `development_repeat_seeds`) with fold-local C
  selection (using only each seed's own outer-train partition, never the
  outer held-out subjects) and subject-level bootstrap confidence intervals
  per seed. Same validated run, all 8 seeds completed (0 excluded):
  **macro-F1 mean 0.726 (std 0.037), balanced accuracy mean 0.728 (std
  0.044)** — this is the current primary development estimate. Paired
  against three required baselines fit on the identical per-seed
  train/test partition as the candidate: the candidate beats a majority-class
  baseline and a constant-prevalence-probability baseline on all 8/8 seeds
  (mean macro-F1 diff +0.321), and beats an untuned all-gene bulk logistic
  baseline on 4/8 seeds with 4 losses (mean diff +0.021 — a modest,
  honestly-reported margin, not a large one). The full per-seed metric
  bundle (balanced accuracy, weighted F1, per-class precision/recall/F1/
  support, confusion matrix, AUROC, AUPRC, Brier score, log loss, ECE) and a
  development-only calibration+threshold pathway (fit on an inner
  train/validation split of each seed's own outer-train partition, evaluated
  once against outer test, method selected automatically between
  uncalibrated/sigmoid/isotonic per `configs/evidence.yaml`'s
  `calibration_policy`) are both reported alongside the primary macro-F1/
  balanced-accuracy summary — see `evidence.development._full_metric_bundle`
  and the `calibration_by_seed` field. A deterministic, versioned-policy
  frozen-internal-test eligibility assessment
  (`evidence.development.assess_frozen_internal_test_eligibility`,
  `configs/evidence.yaml`'s `frozen_internal_test_policy`) reports GSE123352
  as **ineligible** for a frozen internal-test partition today: 176 verified
  subjects (58 minority-class) versus a configured minimum of 300 total /
  50 per class — carving a third partition out of a cohort this small would
  leave every partition too small to trust, so no frozen partition is
  created. Sanitized artifacts: `evidence/published/gse123352_verified_label_smoke_v2/summary.json`
  (single-split) and `evidence/published/gse123352_repeated_development_v3/summary.json`
  (repeated, baselines, full metrics, calibration, and the frozen-test
  eligibility decision); private bundles under `artifacts/evidence/`
  (gitignored — regenerate via `evidence.development.run_gse123352_repeated_development`
  to reproduce). Six-class single-cell smoke classification remains **not
  established** on real data.
- **GSE136831 COPD-vs-Control disease-status proxy analysis: real,
  explicitly NOT smoke-classification evidence.** GSE136831 carries no
  verified per-subject cigarette-exposure field. COPD is a clinical
  diagnosis with strong smoking association but also documented
  non-smoking causes; Control does not prove never-smoking.
  `evidence.tracks.run_copd_control_proxy_analysis` (task=
  `exploratory_disease_proxy_analysis`, never `smoke_classification`;
  `verified_label_count` always `0`) scores this explicitly as a disease-
  status proxy sensitivity analysis — a perfect score reflects, at most,
  COPD-vs-Control disease-state separability, not smoke-exposure
  classification performance. Its historical macro-F1 1.0 on 6
  development-holdout subjects remains recorded only as a labeled,
  low-support exploratory proxy result — it can never enter a verified
  smoke-evidence table, cannot be selected by a caller flag, and cannot
  satisfy any clinical-readiness dimension (see
  `scripts/run_gse136831_copd_control_proxy_analysis.py`).
- **Real cancer-prediction performance: not established.** No cohort in
  this repository has both compatible expression input and a genuinely
  linked subject-level cancer outcome — downloading more data does not
  change this; see `configs/cohorts.yaml`'s cohort-by-cohort
  `expression_outcome_linkable_at_subject_level: false`. TCGA bulk tumour
  labels are never treated as per-cell malignancy labels; NLST outcomes
  are never linked to unrelated expression cohorts.
- **Malignancy classification: not established.** No single-cell cohort
  carries a per-cell/per-sample malignancy label.
- **External validation: not performed.** No cohort in `configs/cohorts.yaml`
  carries `role_eligibility=[external_validation]`;
  `evidence.development.run_external_test()` always returns a structured
  `NO_ELIGIBLE_EXTERNAL_COHORT`.
- **Clinical readiness: not established (`clinically_not_ready`).** See
  `docs/CLINICAL_READINESS.md`. `guard_clinical_claim()` requires a real,
  signed `ClinicalEvidenceManifest` verified against a configured
  approved-public-key allow-list; the default repository configuration
  ships no such key, so no manifest — however constructed — can unlock a
  clinical-readiness claim by default.
- **Historical 77.2% accuracy / 0.27 macro-F1** (referenced elsewhere in
  this document) remains a training-set-only, pre-subject-split number,
  incompatible with the current assay-separated, subject-level-split
  evidence framework — not a current performance estimate.

**What Phase 7 does and does not do:** Track A has real, honestly-scoped
results for GSE123352 (exploratory single-split + repeated development) and
a real, explicitly-non-smoke GSE136831 disease-status proxy analysis, all
`development`-role only. Track B (malignancy classification) and Track C
(subject-level cancer prediction), the candidate comparison, subgroup
diagnostics, and calibration/thresholding remain implemented and tested
only against synthetic fixtures — none has been run against real data,
because no cohort satisfies their eligibility gates. `src/evidence/` is
infrastructure plus genuine development-only real-data results, not a
claim that clinical, external, or cancer-prediction evidence exists — see
`docs/EVIDENCE_PROTOCOL.md` for exactly which steps remain and why.

## Next steps

- Run `python -m benchmarks.runner` against the real single-cell data
  (GSE136831, once re-converted, plus any additional real single-cell
  source under `data.assay_policy=single_cell_only`) for both tasks — the
  framework exists and is tested against synthetic data, but has not been
  run against real data yet, so no real baseline-vs-neural comparison number
  can be reported honestly today. GSE994/GSE123352/GSE307690/CANUCK are
  bulk (see "Assay separation" above) and cannot be added to this run
  without a genuinely implemented bulk training pipeline.
- Wire the neural/MIL adapter into the same one-shot frozen-threshold final
  test evaluation path baselines already use (`benchmarks/runner.py::run_cancer_task`
  currently only does this for baseline models — a documented, not silent, gap)
- Wire the declared hyperparameter search spaces (`baselines.py`'s
  `SMOKE_SEARCH_SPACE`/`CANCER_SEARCH_SPACE`) into an actual nested-grouped-CV
  selection loop — every baseline currently trains with defaults only, the
  search spaces are unused constants today
- Implement true cell-mode Task A evaluation (fit baselines directly on
  capped per-subject cells, not just subject-summary features) so
  cell-weighted metrics stop being marked not-applicable for baselines
- Get a GDC token and NLST DUA to unlock TCGA-LUAD/LUSC (real malignancy
  labels) and NLST (real cancer outcomes + cigar/dual-use history) — this
  also unblocks Task B eligibility on real data, not just the current
  synthetic CI check
- Run the Phase 2 development-only imbalance-strategy ablation
  (`benchmarks/imbalance_ablation.py::run_smoke_imbalance_ablation`) against
  real data once it's available, and — separately, at most once — the
  frozen-test protocol for whichever strategy that ablation selects; no real
  imbalance-strategy comparison number exists yet, only the synthetic/
  unit-tested implementation (see "Phase 2 — subject-aware class-imbalance
  correction" above)
- Phase 3+ of the wider improvement plan (causal modelling, counterfactual
  generation, pathway-constrained learning, foundation-model integration) is
  explicitly out of scope for this benchmarking framework and not started
- Bulk-vs-single-cell assay separation is now enforced by policy
  (`data.assay_policy`, `src/data/assay_policy.py`) and species-provenance
  is tracked (`src/data/species_policy.py`, mouse subjects namespaced) — a
  real BULK training pipeline (`assay_policy=bulk_only`) and a genuine
  multimodal architecture (`assay_policy=multimodal`) remain unimplemented;
  requesting either currently raises a typed not-implemented error rather
  than silently reusing the single-cell model/loss
- Add MIL pooling baselines (mean/max pooling vs. the current gated
  attention) and attention-stability analysis under repeated cell subsampling
