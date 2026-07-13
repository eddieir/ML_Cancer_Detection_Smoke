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
| [GSE136831](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE136831) | Lung scRNA-seq atlas, 312,928 real single cells | Cigarette (documented approximation — see caveat below) | Free, no login — largest source (~2GB), converted via the streaming mtx parser (`src/data/converters.py::_read_mtx_streaming`), with real per-cell donor IDs from GEO's own metadata table |
| [GSE288003](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288003) | Mouse lung scRNA-seq, e-cig aerosol exposure — 23,595 real cells (10,467 unexposed control + 13,128 e-cig exposed) | Vape/e-cig (per-sample, real condition) | Free, no login — its real count matrix ships inside `RAW.tar`, which the downloader now extracts; each of the two GSM samples keeps its real exposure condition instead of a blanket label |
| [GSE307690](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307690) (CANUCK study) | Real human airway epithelial brushings, 61 samples (139 cannabis smokers + 57 never-smokers in the full published cohort) | Cannabis, dual-use, cigarette, vape, unexposed | Free, no login |
| TCGA-LUAD / TCGA-LUSC | Tumor + adjacent-normal tissue, real per-sample malignancy labels | Cigarette (default; TCGA doesn't record smoke type) | Free, but needs a personal [GDC token](https://portal.gdc.cancer.gov/) (register → profile menu → "Download Token") |
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
smoking-status field. `cigarette` is applied as a documented approximation
(COPD is strongly smoking-associated, and no better per-subject label
exists for this accession), the same pattern already used for TCGA-LUAD/
LUSC below. Donor IDs and the disease label itself are real, joined
per-cell from GEO's own `*_AllCells.Samples.CellType.MetadataTable.txt.gz`
(exact barcode match, not a prefix guess — see
[converters.py](src/data/converters.py) `_load_gse136831_cell_metadata`).

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
pooling vs. gated attention) or attention-stability analysis;
subject-aware/subject-capped sampling (class weights are now correctly
computed from the train split only, but there is no per-epoch
max-cells-per-subject sampler yet); explicit bulk-vs-single-cell-vs-MIL
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

**Known Phase 1 limitations** (see `report.md`'s own limitations section for
the same list, generated fresh per run): grouped CV reuses the
`ExperimentContext`'s already train-fit `PreprocessingArtifact` rather than
refitting HVG/scaling independently inside each fold (safe — the artifact
was fit on the *original* train split, a subset of every fold's train
partition — but not the fully independent per-fold refit the ideal protocol
calls for); the final frozen-threshold test evaluation is wired for baseline
models only (the neural/MIL adapter isn't yet plugged into that same
one-shot path); attention weights remain an interpretability aid, not a
causal explanation, in every pooling variant.

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
    calibration.py                 validation-only calibration + frozen threshold
    ood.py                          leave-one-dataset-source-out validation
    reporting.py                    statistical comparison, immutable artifacts, Markdown/CSV report
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
| `tests/*` | All modules covered (267 tests): `test_model.py`, `test_pipeline.py`, `test_loaders.py`, `test_transforms.py`, `test_labellers.py`, `test_assembly.py`, `test_converters.py`, `test_train.py`, `test_splitting.py`, `test_preprocessing.py`, `test_preprocess_split_aware.py`, `test_rare_class.py`, `test_label_mapping.py`, `test_evaluate.py`, `test_inference.py`, plus 9 `test_benchmarks_*.py` files |
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

## Next steps

- Run `python -m benchmarks.runner` against the real merged data (GSE994 +
  GSE307690 + GSE123352 + GSE136831 once re-converted) for both tasks — the
  framework exists and is tested against synthetic data, but has not been
  run against real data yet, so no real baseline-vs-neural comparison number
  can be reported honestly today
- Wire the neural/MIL adapter into the same one-shot frozen-threshold final
  test evaluation path baselines already use (`benchmarks/runner.py::run_cancer_task`
  currently only does this for baseline models — a documented, not silent, gap)
- Refit HVG/scaling independently inside each grouped-CV fold instead of
  reusing the context's already train-fit `PreprocessingArtifact` — safe as
  currently implemented (see the Benchmarking framework section's
  limitations), but not the fully independent per-fold refit the ideal
  protocol calls for
- Get a GDC token and NLST DUA to unlock TCGA-LUAD/LUSC (real malignancy
  labels) and NLST (real cancer outcomes + cigar/dual-use history) — this
  also unblocks Task B eligibility on real data, not just the current
  synthetic CI check
- Add a subject-aware/subject-capped sampler (bound max cells sampled per
  subject per epoch) so a handful of subjects with very large cell counts
  can't dominate a training epoch — class weights are now train-split-only,
  but no per-epoch subject-balancing sampler exists yet (`benchmarks/features.py::cap_cells_per_subject`
  exists for benchmark baselines but is not yet wired into `train.py`'s own curriculum)
- Phase 2+ of the wider improvement plan (causal modelling, counterfactual
  generation, pathway-constrained learning, foundation-model integration) is
  explicitly out of scope for this benchmarking framework and not started
- Add explicit bulk-vs-single-cell-vs-MIL experiment-mode separation and
  species-provenance tracking (GSE288003's mouse→human ortholog mapping
  still runs unconditionally with no recorded mapped/unmapped gene counts)
- Add MIL pooling baselines (mean/max pooling vs. the current gated
  attention) and attention-stability analysis under repeated cell subsampling
