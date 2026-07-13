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

**Subject-level splitting** (`src/data/splitting.py`, new). Every prior
result in this repo — including the 77.2%/0.27 table above — was computed by
training and evaluating on the same cells, with no subject held out at all.
A subject's cells share genetic background and batch/technical variation, so
even a naive train/test split *by cell* would leak: a model that's seen 80%
of a subject's cells trivially recognizes the other 20%. `splitting.py` now
provides `subject_train_val_test_split()` and `grouped_kfold()`, both
grouped by `subject_id` with reproducible JSON manifests
(`data/processed/split_manifest.json` by default, see `configs/default.yaml`'s
`split:` block) and leakage assertions covered by 16 tests. **No model in
this repo has yet been retrained/re-evaluated against one of these splits on
real data** — the tooling is in place and tested; the experiment hasn't been
run (would need a real training run, out of scope for this pass — see
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

**Preprocessing leakage fix.** `merge_sources()` z-scored, and
`smoke_aware_hvg()` selected highly-variable genes across, the *entire*
merged dataset — before any split existed, so validation/test cells
influenced which genes became features and how they were scaled.
`src/data/preprocessing.py` adds a `PreprocessingArtifact` (versioned,
JSON-serialisable) fit on train-split cells only
(`fit_preprocessing`/`apply_preprocessing`), and
`preprocess.py::run_pipeline_split_aware()` wires it end-to-end: split
first, fit on train, apply unchanged to val/test. The original
`run_pipeline()` is unchanged and still has this leakage — it's kept only
for backward compatibility and synthetic smoke-testing, with an explicit
warning in its docstring. **Batch correction (Harmony) is a documented
exception that is *not* leakage-free**: Harmony has no train-only-fit /
apply-to-new-data mode, so it is still fit across the full merged dataset
even in the split-aware path — see `PreprocessingArtifact.notes`.

**Rare smoke-type classes** (`src/data/rare_class.py`, new). The cigar class
has ~1 independent subject in the real merged data — not enough to learn or
evaluate as its own class by any reasonable statistical standard.
`apply_rare_class_policy()` makes this an explicit, configurable, auditable
decision (`keep_with_warning` / `merge_into_dual_use_or_other` /
`exclude_from_training_and_evaluation`, set via `configs/default.yaml`'s
`rare_class:` block) instead of a class that's silently never predicted.
Raw labels are never mutated; every action is recorded in a report dict.

**What this pass does *not* include** (explicitly out of scope, not
silently skipped): a real training run against a held-out split (so there
is no new "real held-out macro-F1" number to report — see the table above);
a baseline-model comparison runner (logistic regression / random forest /
XGBoost / small MLP vs. the neural model); a grouped-cross-validation
experiment runner (the primitive exists in `splitting.py::grouped_kfold`,
tested for leakage, but no training loop consumes it yet); a
hyperparameter-search runner; MIL pooling-baseline comparisons or attention-
stability analysis; probability calibration / validation-selected threshold
tooling (the model still reports risk at a fixed 0.70 cutoff in
`train.py::predict` and `inference.py`, which is **not** a clinically
validated threshold — treat it as an arbitrary placeholder). Each of these
is a legitimate, separately-scoped follow-up, not an oversight.

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
| `tests/*` | All modules covered (117 tests): `test_model.py`, `test_pipeline.py`, `test_loaders.py`, `test_transforms.py`, `test_labellers.py`, `test_assembly.py`, `test_converters.py`, `test_train.py`, `test_splitting.py`, `test_preprocessing.py`, `test_preprocess_split_aware.py`, `test_rare_class.py`, `test_evaluate.py`, `test_inference.py` |
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

- Run `preprocess.py::run_pipeline_split_aware()` on the real merged data
  (GSE994 + GSE307690 + GSE123352 + GSE136831 once re-converted), then
  actually train against the resulting split and report real held-out
  macro-F1/balanced-accuracy/per-class-F1 — the tooling exists and is
  tested; this experiment has not been run yet, so no held-out number can
  be reported honestly today
- Re-run Phase 1 training with the inverse-frequency class weighting
  (`CellLevelDataset.smoke_class_weights`, `train.py`) against a real
  held-out split, not the training-set numbers in the table above
- Build the baseline-model runner from the improvement plan (majority-class,
  logistic regression, random forest, small MLP) on the same subject-level
  splits, so the neural model's real held-out macro-F1 can be judged against
  something simpler rather than assumed better
- Wire `grouped_kfold()` into an actual training loop for a real
  cross-validation experiment (aggregate mean/std/median across folds)
- Add validation-selected threshold + calibration reporting for the cancer
  head instead of the current fixed, non-clinical 0.70 cutoff
- Get a GDC token and NLST DUA to unlock TCGA-LUAD/LUSC (real malignancy
  labels) and NLST (real cancer outcomes + cigar/dual-use history)
