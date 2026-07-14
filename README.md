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
    calibration.py                 OOF-based calibration + guarded one-time frozen-test evaluation
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
| `tests/*` | All modules covered (495 tests, `python3 -m pytest tests/ -q`): `test_model.py`, `test_pipeline.py`, `test_loaders.py`, `test_transforms.py`, `test_transforms_inductive_annotation.py`, `test_labellers.py`, `test_assembly.py`, `test_converters.py`, `test_train.py`, `test_splitting.py`, `test_preprocessing.py`, `test_preprocess_split_aware.py`, `test_rare_class.py`, `test_label_mapping.py`, `test_evaluate.py`, `test_inference.py`, plus 24 `test_benchmarks_*.py` files (including `test_benchmarks_atomic_io.py`, `test_benchmarks_model_fingerprint.py`, `test_benchmarks_cell_type_provenance.py`, `test_benchmarks_env_versions.py`, and `test_benchmarks_final_evaluation.py`) |
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
