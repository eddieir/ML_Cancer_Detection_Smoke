# MultiSmokeCancerNet — Full Architecture Specification

**Novel cell-level lung cancer risk prediction by smoke type**  
Combines Gap 1 (multi-smoke-type cell classifier) + Gap 3 (MIL aggregation from cells to subject)

---

## 1. Problem Statement

Given a set of N lung cells from a single subject, each described by its scRNA-seq gene expression profile:

1. Classify which smoke type damaged each cell (6 classes)
2. Score each cell's malignancy risk (continuous [0, 1])
3. Aggregate across all N cells to produce one subject-level cancer probability

No existing paper does all three together.

---

## 2. Pipeline Overview

```
[Raw scRNA-seq: N cells x ~28,000 genes]
            |
     STAGE 1: PREPROCESSING
            |
     STAGE 2: SHARED CELL ENCODER
            |
        /-------\
  STAGE 3A     STAGE 3B
  Smoke Type   Malignancy
  Classifier   Scorer
        \-------/
            |
     STAGE 4: CELL FEATURE ASSEMBLY
            |
     STAGE 5: GATED ATTENTION MIL
            |
     STAGE 6: SUBJECT CLASSIFIER
            |
       P(cancer) in [0,1]
```

Total trainable parameters: **~2.86M**

---

## 3. Stage-by-Stage Specification

---

### Stage 1: Preprocessing (Scanpy pipeline)

**Input:** Raw count matrix from scRNA-seq (AnnData format)

| Step | Tool / Function | Purpose |
|------|----------------|---------|
| QC filtering | `sc.pp.filter_cells(min_genes=200)` | Remove damaged cells |
| MT% filter | `adata.obs.pct_counts_mt <= 20` | Remove dying cells |
| Gene filter | `sc.pp.filter_genes(min_cells=3)` | Remove rare genes |
| Normalization | `sc.pp.normalize_total(target_sum=1e4)` | CPM normalization |
| Log transform | `sc.pp.log1p()` | Stabilize variance |
| HVG selection | `sc.pp.highly_variable_genes(n_top_genes=2000)` | Reduce dimensionality |
| Scaling | `sc.pp.scale(max_value=10)` | Z-score per gene |
| Batch correction | `harmonypy.run_harmony()` | Remove dataset-level batch effects |
| Cell type annotation | `celltypist.annotate(model='Immune_All_Low.pkl')` | Label each cell type |

**Output:** Gene matrix of shape `[N_cells x 2000]` with metadata columns:
- `smoke_type` (int 0-5)
- `malignancy` (float 0.0 or 1.0)
- `cell_type_id` (int 0-3)

---

### Stage 2: Shared Cell Encoder (MLP)

**Why shared?** Forces a single representation to be useful for both smoke-type classification and malignancy scoring simultaneously. Shared weights create cross-task regularization — the encoder cannot overfit to one head's signal alone.

```
Input: x in R^2000  (2,000 HVGs, z-scored)
  |
  FC(2000 -> 1024) + BatchNorm1d(1024) + GELU + Dropout(0.3)
  |
  FC(1024 -> 512)  + BatchNorm1d(512)  + GELU + Dropout(0.3)
  |
  FC(512 -> 256)   + BatchNorm1d(256)  + GELU
  |
Output: z in R^256  (cell embedding, shared by both heads)
```

**Parameter count:** 2000*1024 + 1024*512 + 512*256 = ~2.6M

**Design decisions:**
- GELU over ReLU: smoother gradients, better for gene expression data
- BatchNorm before activation: stabilizes training across heterogeneous cell populations
- No dropout on final layer: preserve information for both downstream heads

---

### Stage 3A: Smoke Type Classification Head (Head A)

**Purpose:** Given cell embedding z, predict which smoke type caused the molecular changes in this cell.

```
Input: z in R^256
  |
  FC(256 -> 128) + GELU + Dropout(0.2)
  |
  FC(128 -> 6)
  |
  Softmax -> smoke_probs in R^6
```

**Output classes:**

| Index | Class | Training data source | Data caveat |
|-------|-------|---------------------|-------------|
| 0 | cigarette | GEO GSE994, GSE123352, GSE136831 | True scRNA-seq / microarray |
| 1 | vape / e-cig | GEO GSE288003 + human bronchial cell datasets | GSE288003 is mouse — requires human ortholog mapping via biomaRt |
| 2 | cigar | NLST cigar-reported subjects, cells from GSE136831 re-labelled by smoking history metadata | No dedicated cigar scRNA-seq exists; label transferred from clinical metadata |
| 3 | cannabis | GSE307690 (CANUCK study) | Real human airway epithelial brushings, 139 cannabis smokers vs. 57 never-smokers — bulk RNA-seq, used as pseudo-bulk |
| 4 | dual-use | NLST dual-reported subjects (cigarette + vape); GSE307690 samples with both `cannabis group: Cannabis` and `cigarette: current/former` or `vape: Yes` | Approximated from clinical/GEO annotation |
| 5 | unexposed | Never-smoker controls from all above datasets | — |

**Loss:** `CrossEntropyLoss(weight=class_weights)`  
Class weights computed from inverse frequency of each smoke type in training set to correct imbalance.

**Why this is novel:** No existing paper builds an ML model that can distinguish all 6 smoke exposure types from single-cell gene expression.

---

### Stage 3B: Malignancy Risk Scoring Head (Head B)

**Purpose:** Given cell embedding z, score the probability that this cell is undergoing malignant transformation.

```
Input: z in R^256
  |
  FC(256 -> 128) + GELU + Dropout(0.2)
  |
  FC(128 -> 1)
  |
  Sigmoid -> malignancy_score in [0, 1]
```

**Training labels:**

| Label | Source | Rationale |
|-------|--------|-----------|
| 1.0 (malignant) | TCGA-LUAD / TCGA-LUSC tumor tissue cells | Confirmed cancer |
| 1.0 (malignant) | BEAS-2B cells after 30 CS passages (Ma et al. 2019) | In vitro transformation |
| 0.0 (benign) | TCGA normal adjacent tissue (NAT) | Same patients, healthy cells |
| 0.0 (benign) | Never-smoker controls | Negative class |

**Loss:** `BCELoss()`

---

### Stage 3C: Dose-Response Head (Head C) — novel

**Purpose:** Given cell embedding z, regress a normalised smoke exposure
dose (0-1) for cells whose source records one. Every existing smoke-cell
classifier (Ma et al. 2024 included) treats exposure as categorical only —
smoker vs. never-smoker, or smoke type without intensity. Nothing published
models exposure as a continuous variable at single-cell resolution, or ties
it to malignancy trajectory.

```
Input: z in R^256
  |
  FC(256 -> 64) + GELU + Dropout(0.2)
  |
  FC(64 -> 1)
  |
  Sigmoid -> dose_score in [0, 1]
```

**Training labels:** honest status — **no wired source currently supplies a
real per-cell exposure dose.** The "Loiselle 2018 / GSE130148" dataset this
head originally cited does not exist (verified directly against GEO;
GSE130148 is an unrelated human lung scRNA-seq study with no cannabis or
dose-response data). Every cell is stamped `DOSE_UNKNOWN` (-1) by
`_attach_standard_obs()` (`src/data/loaders.py`) and excluded from this
head's loss — the architecture and loss function are real, tested, and
wired end-to-end (see `tests/test_model.py`), but they currently train on
zero real signal. Wiring a genuine continuous exposure-dose source (none
identified yet — GSE307690/CANUCK only exposes categorical joint-year bins
in its public GEO metadata, not per-sample) would activate this head with
no further code changes.

**Loss — two terms, masked to cells with a known dose:**

```
L_dose = MSE( dose_pred, dose_target )
       + mean_over_pairs( ReLU( margin - (malignancy_i - malignancy_j) ) )
         for every pair (i, j) with dose_i > dose_j + margin
```

The second term is a pairwise monotonic-ranking hinge: a cell exposed longer
must not be scored *less* malignant than a cell exposed for less time (same
smoke type, by construction of the pairing within a batch). This is the
piece that makes the model dose-response-aware rather than dose-blind — it
directly operationalises the "no cannabis/dual-use dose-response model
exists" gap in §9 as a differentiable training signal, not just a labeled
class.

Implemented in `model.py::DoseResponseHead` and
`MultiTaskLoss.dose_response_loss`; wired into Phase 1 training in
`train.py::Trainer.phase1` with weight `lambda_dose` (default 0.10,
`configs/default.yaml`).

---

### Stage 4: Cell Feature Assembly

**Purpose:** Concatenate all per-cell information before subject-level aggregation. This lets the aggregator use smoke type and malignancy predictions — not just the raw embedding — as signals for attention weighting.

```
For each cell i in subject's cell pool:

  h_i = concat[
    z_i          (256 dims)  -- cell embedding from Stage 2
    smoke_probs_i  (6 dims)  -- Head A softmax output
    malignancy_i   (1 dim)   -- Head B sigmoid output
    cell_type_i    (4 dims)  -- one-hot: epithelial/endothelial/immune/stromal
  ]

  h_i in R^267
```

**Cell type encoding:**

| ID | Type | Cancer relevance |
|----|------|-----------------|
| 0 | epithelial | Primary site of lung carcinogenesis |
| 1 | endothelial | Angiogenesis signal in tumor microenvironment |
| 2 | immune | Immune evasion signature |
| 3 | stromal | Desmoplastic reaction in cancer |

**Why include cell type?** Epithelial cells have fundamentally different malignancy kinetics than immune cells. The aggregator needs to know what kind of cell it is weighting, not just how risky it looks.

---

### Stage 5: Gated Attention MIL Aggregator

**Reference:** Ilse et al. ICML 2018 — Attention-based Deep Multiple Instance Learning

**Why MIL over mean pooling?**  
Mean pooling weights all cells equally. A subject with 500 cells may have only 20 early-malignant cells — those 20 get diluted to noise. Gated attention learns to up-weight those 20 cells and effectively ignore the 480 healthy bystanders.

**Gated attention mechanism:**

```
For each cell i with feature vector h_i in R^267:

  value_gate    = tanh( V * h_i )       -- V: R^(267 x 128)
  selector_gate = sigmoid( U * h_i )    -- U: R^(267 x 128)
  
  gates_i = value_gate * selector_gate  -- element-wise product in R^128
  
  raw_score_i = w^T * gates_i           -- w: R^128, scalar score per cell
  
  a_i = softmax( [raw_score_1, ..., raw_score_N] )   -- normalized across N cells

Subject representation:
  Z_subject = sum_i( a_i * z_i )        -- attention-weighted bag in R^256
```

**Two gates vs one:**  
The tanh gate controls the direction (sign) of the contribution. The sigmoid gate controls the magnitude. Together they prevent all-zero attention collapse that can happen with single-gate attention.

**Interpretability:**  
After inference, `a_i` weights are inspectable per cell. The cells with the highest attention weights are the ones most responsible for the cancer prediction — enabling biological investigation of which specific cells drove the model's output.

---

### Stage 6: Subject-Level Classifier

```
Input: Z_subject in R^256
  |
  FC(256 -> 64) + GELU + Dropout(0.2)
  |
  FC(64 -> 1)
  |
  Sigmoid -> P(cancer) in [0, 1]
```

**Output thresholds:**

| P(cancer) | Clinical flag | Action |
|-----------|--------------|--------|
| >= 0.70 | HIGH RISK | Immediate clinical follow-up |
| 0.40 - 0.69 | MODERATE RISK | Increased surveillance |
| < 0.40 | LOW RISK | Routine screening |

**Loss:** `BCELoss()`

---

## 4. Multi-Task Loss Function

```
L_total = lambda_smoke     * L_CE( smoke_type_logits, smoke_labels )
        + lambda_malignancy * L_BCE( malignancy_scores, malignancy_labels )
        + lambda_subject    * L_BCE( cancer_prob, cancer_label )
        + lambda_dose       * L_dose( dose_scores, dose_labels, malignancy_scores )   -- Phase 1 only
```

**Lambda values:**

| Phase | lambda_smoke | lambda_malignancy | lambda_subject | lambda_dose |
|-------|-------------|------------------|---------------|------------|
| Phase 1 (cell pre-train) | 0.50 | 0.50 | 0.00 | 0.10 |
| Phase 2 (aggregator train) | 0.00 | 0.00 | 1.00 | 0.00 |
| Phase 3 (end-to-end) | 0.30 | 0.30 | 0.40 | 0.00 |

Subject prediction carries 0.40 weight in Phase 3 because it's the primary clinical objective.
`L_dose` is Phase-1-only: it needs per-cell exposure duration, which only
exists at cell-level pretraining time (see Stage 3C).

---

## 5. Three-Phase Training Curriculum

Every phase below takes **explicit, pre-split** train/val datasets built
from a `SplitManifest` (`data/splitting.py`) — `Trainer.phase1/2/3` do not
split anything internally, and reject overlapping train/val subject IDs
before training starts. See §11 for the full rationale; the per-phase specs
below (optimizer, schedule, batch size) are otherwise unchanged.

### Phase 1 — Cell-Level Pre-training

**Goal:** Teach the encoder to produce discriminative cell embeddings for smoke type and malignancy.

```
Layers trained:  CellEncoder + SmokeTypeHead + MalignancyHead
Layers frozen:   GatedAttentionMIL (aggregator not touched)

Optimizer:   Adam(lr=1e-3, weight_decay=1e-4)
Scheduler:   CosineAnnealingLR(T_max=15)
Epochs:      15
Batch size:  512 cells
Grad clip:   max_norm=1.0
Val metric:  smoke_accuracy + malignancy_AUC
Checkpoint:  save on best smoke_accuracy
```

**Training data:** Cell-level GEO datasets with smoke_type labels + TCGA malignancy labels

---

### Phase 2 — Subject-Level Aggregator Training

**Goal:** Teach the aggregator which cell patterns signal cancer at the subject level. Encoder is frozen so its representations don't drift.

```
Layers trained:  GatedAttentionMIL (V, U, w, subject_classifier)
Layers frozen:   CellEncoder + SmokeTypeHead + MalignancyHead

Optimizer:   Adam(lr=5e-4, weight_decay=1e-4)
Scheduler:   ReduceLROnPlateau(mode=max, patience=3, factor=0.5)
Epochs:      12
Batch size:  1 subject (variable N cells per subject)
Grad clip:   max_norm=1.0
Val metric:  subject_AUC (ROC-AUC on cancer/no-cancer)
Checkpoint:  save on best subject_AUC
```

**Training data:** NLST subjects (cancer outcome known) + TCGA subjects as positives

---

### Phase 3 — End-to-End Fine-Tuning

**Goal:** Jointly optimize all layers with full multi-task loss. Lower LR prevents catastrophic forgetting of Phase 1 and 2 knowledge.

```
Layers trained:  ALL (encoder + both heads + aggregator)

Optimizer:   Adam(lr=1e-4, weight_decay=1e-5)
Epochs:      8 (with early stopping, patience=5)
Batch:       Alternate between cell batches and subject batches per step
Grad clip:   max_norm=1.0
Val metric:  subject_AUC (primary), smoke_acc + malig_AUC (secondary)
Checkpoint:  save on best subject_AUC
Stop:        Early stop if no improvement for 5 consecutive epochs
```

---

## 6. Data Sources

### Cell-level (smoke type + malignancy labels)

| Dataset | Type | Cells | Smoke type | Access |
|---------|------|-------|-----------|--------|
| GEO GSE994 | Bronchial epithelial microarray | ~75 subjects | Cigarette (active/former/never) | Free |
| GEO GSE123352 | Lung tissue RNA-seq | 176 subjects | Cigarette (ever/never) | Free |
| GEO GSE136831 | scRNA-seq, lung atlas | 312,928 cells | Cigarette | Free |
| GEO GSE288003 | Lung cells, e-cig aerosol | Mouse model | Vape / e-cig | Free — map mouse genes to human orthologs via biomaRt before use |
| GSE307690 (CANUCK study) | Real human airway epithelial brushings | 61 samples (139 cannabis smokers + 57 never-smokers in the full cohort; public GEO release covers a subset) | Cannabis, dual-use (cannabis+cigarette/vape), cigarette, vape, unexposed | Free — bulk RNA-seq, pseudo-bulk; one vector per sample not per cell |
| TCGA-LUAD | Tumor + NAT cells | ~541 patients | Malignancy labels | Free (TCIA) |
| TCGA-LUSC | Tumor + NAT cells | ~512 patients | Malignancy labels | Free (TCIA) |

### Subject-level (cancer outcome labels)

| Dataset | Subjects | Outcome | Access |
|---------|----------|---------|--------|
| NLST | ~26,722 | 10-year cancer outcome + smoking history | Free (cdas.cancer.gov/nlst) |
| TCGA-LUAD/LUSC combined | ~1,053 | Cancer positive | Free (TCIA) |

---

## 7. Model Parameters Summary

| Component | Parameters |
|-----------|-----------|
| CellEncoder (FC 2000->1024->512->256) | ~2,631,680 |
| SmokeTypeHead (FC 256->128->6) | ~33,414 |
| MalignancyHead (FC 256->128->1) | ~32,897 |
| GatedAttentionMIL (V, U, w, classifier) | ~137,601 |
| **Total** | **~2,835,592** |

---

## 8. Evaluation Metrics

### Cell-level (Phase 1 validation)
- Smoke type: **macro-F1 is the primary model-selection metric** (not
  accuracy — see §11), plus balanced accuracy, weighted-F1, per-class
  precision/recall/F1/support, and raw + row-normalized confusion matrices
  (`evaluate.py::_smoke_metrics`)
- Malignancy: ROC-AUC, precision-recall AUC — computed only on cells with a
  real (`malignancy_known=True`) label; see §11

### Subject-level (Phase 2 + 3 validation, primary)
- ROC-AUC on cancer vs. no-cancer (main metric)
- Sensitivity / specificity at threshold 0.70 (HIGH RISK cutoff) — **this
  0.70 cutoff is an arbitrary placeholder, not a clinically validated
  threshold.** No calibration or validation-selected-threshold study has
  been performed. Treat any HIGH/MODERATE/LOW risk_flag output the same way.
- Calibration curve (predicted probability vs. observed frequency)

### Interpretability
- Attention weight distribution across cell types per subject
- Which cell types receive highest attention in cancer vs. non-cancer subjects
- Smoke type profile of the top-attended cells

---

## 9. Novel Contributions vs. Literature

| Claim | Closest existing paper | Gap |
|-------|----------------------|-----|
| Multi-smoke-type cell classifier (6 types) | Ma et al. 2024 (cigarette only, 3 states) | No vape/cigar/cannabis/dual-use |
| Per-cell malignancy risk score | Long et al. 2024 (susceptibility genes only) | Not a predictive model |
| MIL aggregation from scRNA-seq to subject | Used in WSI histopathology (ABMIL 2018) | Never applied to scRNA-seq bags |
| Cannabis lung cell cancer model | CDC acknowledges gap officially (2024) | Does not exist anywhere |
| Dual-use cellular signature | Bittoni et al. 2024 (epidemiology only) | No cell-level ML model |
| Continuous dose-response modeling (exposure duration -> malignancy trajectory) | All existing smoke-cell models are categorical only (smoker/never-smoker) | No model regresses exposure dose or enforces a monotonic dose->malignancy ordering at single-cell resolution |
| End-to-end smoke->malignancy->cancer pipeline | Not in any paper, preprint, or conference | Confirmed gap across all source types |

---

## 10. Build Order (Next Steps)

```
Step 1 (this file):  Architecture specification         [DONE]
Step 2:              src/preprocess.py                  [DONE]
Step 3:              src/model.py                       [DONE]
Step 4:              src/train.py                       [DONE]
Step 5:              src/evaluate.py                    [DONE]
Step 6:              src/inference.py                   [DONE]
Step 7:              notebooks/01_data_download.ipynb   [DONE]
Step 8:              notebooks/02_preprocessing.ipynb   [DONE]
Step 9:              notebooks/03_training.ipynb        [DONE]
Step 10:             notebooks/04_evaluation.ipynb      [DONE]
```

TCGA-LUAD/LUSC (section 6) is wired end-to-end: `data/downloaders.py --tcga`
→ `data/converters.py` (`convert_tcga`) → `data/loaders.py::load_microarray`
(malignancy + subject_id overrides) → `configs/default.yaml`. Tumor/NAT
samples supply per-cell malignancy labels (Stage 3B) and TCGA cases with a
Primary Tumor sample count as subject-level cancer positives (Stage 6),
merged with NLST outcomes in `preprocess.py::run_pipeline`.

---

## 11. Scientific Validity: Splitting, Leakage, and Label Provenance

Everything in sections 1–10 above describes the model architecture and is
unchanged. Separately, on branch `improve/valid-evaluation-and-training`, a
set of correctness/leakage issues in the pipeline that feeds that
architecture were fixed. Full detail and rationale live in README.md's
[Scientific rigor and known limitations](README.md#scientific-rigor-and-known-limitations)
section; summarized here for architectural completeness:

1. **Subject-level splitting is now real and unavoidable end-to-end**, not
   just an available utility. `data/splitting.py` provides subject-grouped
   train/val/test splitting and grouped K-fold CV with reproducible,
   fingerprinted manifests (`load_or_create_split()` raises rather than
   silently reusing a stale split if subjects/labels/config changed).
   Critically, `Trainer.phase1/2/3` (§5) previously called `random_split()`
   internally regardless of any manifest, so a subject's cells could still
   land in both the "train" and "validation" partition passed to a phase.
   This is fixed: every phase now takes explicit, pre-split
   `train_*_dataset`/`val_*_dataset` arguments, calls
   `assert_disjoint_subjects()` before training anything, and never accepts
   a test dataset — `Trainer.final_test_evaluation()` is the one sanctioned
   place test data is used, after checkpoint selection, evaluated once, and
   labelled `is_held_out=True`. `preprocess.py::run_pipeline_split_aware()`
   builds the per-split `CellLevelDataset`/bag lists directly from the
   manifest via `CellLevelDataset.subset_by_subjects()`, rather than
   returning one combined array for the caller to filter. No model has yet
   been retrained against one of these splits on real data.
2. **Preprocessing leakage + label order**: gene scaling and HVG selection
   (Stage 1, §3.1) were fit across the entire merged dataset, including
   cells that should have been held out. `data/preprocessing.py` fits both
   on the train split only via a versioned `PreprocessingArtifact`,
   automatically persisted next to the split manifest and the checkpoint
   directory. Separately, `run_pipeline_split_aware()` used to compute the
   subject-level split *before* NLST label transfer, so the split could be
   based on a label that was about to change — label assignment (including
   NLST transfer) now happens first, the rare-class policy (point 5) is
   applied to that final label, and only then is the split computed.
   Harmony batch correction remains a transductive exception (no
   train-only-fit mode exists for it): `run_pipeline_split_aware()` skips it
   by default and requires an explicit `allow_transductive_harmony: true`
   opt-in, which sets a `transductive_batch_correction` flag on the result.
3. **Unknown cancer outcomes were defaulted to 0** (cancer-negative)
   instead of being excluded from Stage 6 supervision — fixed via
   `cancer_label_known` and `train.py::check_mil_eligibility`, now checked
   separately on train and val before Phase 2/3 train.
4. **Unknown malignancy labels were defaulted to 0.0** and trained against
   directly in Stage 3B — fixed via a `malignancy_known` provenance mask
   applied to `MultiTaskLoss`'s malignancy BCE term.
5. **Rare smoke-type classes (e.g. cigar, ~1 independent subject) now
   actually go through the configured policy** (`data/rare_class.py`)
   instead of the utility existing but nothing calling it.
   `run_pipeline_split_aware()` applies `keep_with_warning` /
   `merge_into_dual_use_or_other` / `exclude_from_training_and_evaluation`
   to the final label before splitting; the raw label survives unmutated as
   `smoke_type_raw`.
6. **Checkpoints are now structured**, not a bare `state_dict` — they carry
   split-manifest path, preprocessing-artifact path, effective label
   mapping, rare-class policy, seed, metric, epoch, input dim, and git SHA.
   `Predictor.predict_h5ad()` was validating gene *presence* but not
   actually reordering/scaling incoming data to match training; it now
   applies the fitted `PreprocessingArtifact` (or requires an explicit
   `already_preprocessed=True` declaration with exact gene-order
   verification) before every forward pass.
7. **Cross-task leakage validation was incomplete**: the per-phase checks in
   point 1 only compare same-modality datasets (train cells vs. val cells,
   train bags vs. val bags), missing a subject whose cells are in train but
   whose bag is in val (or vice versa) — a real path Phase 3 exercises since
   it uses all four datasets jointly. `train.py::validate_experiment_partitions()`
   now checks this directly and `Trainer.phase3` calls it.
8. **Held-out test evaluation was enforced only by docstring.**
   `Trainer.final_test_evaluation()` now tracks every subject seen during
   training/validation on that `Trainer` and raises if a test subject
   overlaps them, blocks a second call by default (`allow_repeat=True`
   required, and the result is marked non-pristine), and writes a separate
   `heldout_test_report.json`/`heldout_test_predictions.json` with explicit
   provenance (manifest path, checkpoint id, threshold source, timestamp,
   run count) rather than relying on `evaluation_report.json`'s shared path.
9. **Checkpoint-selection macro-F1 and reported macro-F1 could silently
   disagree.** `Trainer.phase1` computed its selection metric with
   `f1_score(..., average="macro")` and no explicit label list, which
   restricts averaging to classes observed in that validation batch;
   `evaluate.py` already passed an explicit label list. `src/metrics.py`
   now defines this once (`multiclass_f1_report`, flags `is_partial` when a
   class had zero true examples) and both call sites use it.
10. **Cell-type IDs were never validated**, only checked for column
    presence. `metrics.py::validate_cell_type_ids()` enforces
    `0 <= id < num_cell_types`, integer-valued, no NaN, correct length —
    used by `Predictor.predict_subject/predict_h5ad` and `Trainer.predict`.
11. **`grouped_kfold` could crash on small class counts**: independently
    resetting a fold-index counter per class bucket meant two subjects in
    two different classes could collide on fold 0, leaving another fold
    with an empty train or validation set. Fold assignment now uses one
    cursor shared across all class buckets.
12. **The effective smoke-label space wasn't actually contiguous or
    model-visible.** A rare-class policy could merge or exclude a raw
    class, but `model.num_smoke`, macro-F1's class count, confusion
    matrices, and inference's displayed class names all stayed fixed at 6
    — a dead output neuron and a permanent zero-support row.
    `data/label_mapping.py::EffectiveLabelMapping` now builds a
    deterministic `0..K-1` space directly from the rare-class-policy
    report; `run_pipeline_split_aware()` transforms labels into it before
    the split; `MultiSmokeCancerNet.from_config(num_smoke_types=...)`,
    `Trainer.set_label_mapping()`, `Evaluator.from_checkpoint()`, and
    `Predictor.from_config()` all size the model to `K` and validate the
    mapping matches (checkpoint metadata is peeked via
    `train.read_checkpoint_metadata()` *before* the model is constructed,
    so a config/checkpoint conflict fails clearly instead of as an opaque
    `load_state_dict` shape error). Raw labels remain preserved unmutated.
13. **`predict_h5ad(already_preprocessed=False)` claimed to accept "raw"
    input** but only ever reordered/subset genes and applied train-fit
    scaling — never QC, library-size normalization, or log-transform, so
    genuinely raw counts silently produced invalid predictions. The
    boolean is replaced with an explicit `input_stage` argument:
    `"model_ready"` (exact match, no transform — was `True`),
    `"normalized_expression"` (reorder/subset/scale via the artifact only,
    duplicate-gene and finite-value checks added — was `False`), or
    `"raw_counts"`, which is now always rejected with a clear "not
    supported" error, since `PreprocessingArtifact` doesn't store the
    QC/normalization parameters needed to reproduce that chain.
    `already_preprocessed` remains as a deprecated, warned alias mapped
    unambiguously to the two supported stages — never to `"raw_counts"`.

Not yet done, tracked in README's "What this pass does not include": the
full raw-count preprocessing chain reproduced inside `predict_h5ad` (species/
gene-ID/normalization steps — `input_stage="raw_counts"` is explicitly
rejected rather than silently mishandled, but not implemented); checkpoint
checksum verification and optimizer/scheduler resume; subject-aware sampling
wired into `train.py`'s own curriculum (it exists for benchmark baselines,
§12); bulk/single-cell/MIL mode separation; species/ortholog-mapping safety
beyond the config-driven check in §12's leave-one-source-out; dose-head
supervision gating.

The `ExperimentContext`/`Trainer.from_experiment_context()` auto-wiring,
baseline/grouped-CV/MIL-comparison experiment runners, and
validation-selected threshold/calibration tooling previously listed here as
not-yet-done are now implemented — see §12.

None of these are architecture changes — Stages 1–6 as specified above are
unchanged. They are pipeline-around-the-architecture fixes that any real
reported metric must now go through for the metric to be scientifically
valid.

## 12. Phase 1 Benchmarking Framework

`src/benchmarks/` (branch `improve/phase1-rigorous-benchmarking`) answers
whether `MultiSmokeCancerNet` actually beats simple baselines under
identical subject-level splits — not causal modelling, counterfactual
generation, pathway-constrained learning, or foundation models, which are
out of scope for this phase. Full task definitions, metrics, baseline list,
CV/calibration protocol, CLI usage, and output layout are in README's
[Benchmarking framework](README.md#benchmarking-framework-phase-1-does-the-neural-model-beat-simple-baselines)
section; architectural summary here:

- **`ExperimentContext`** (`benchmarks/context.py`) is built once from
  `run_pipeline_split_aware()`'s result and is the only object every
  baseline, the neural adapter, and the CV runner read train/val/test data
  from — never the raw pipeline dict, which still exposes the whole,
  unsplit `bags`/`cell_data` keys a benchmark must not accidentally read.
- **`Trainer.from_experiment_context()`** (`train.py`) derives
  `MultiSmokeCancerNet`'s `input_dim` and `num_smoke_types` from the
  context's `PreprocessingArtifact`/`EffectiveLabelMapping` rather than a
  static config, and raises on any mismatch — the K-class enforcement §11.12
  already gave `Trainer.set_label_mapping()` is now the *only* path a
  benchmark can construct a model through, so it's structurally impossible
  for a benchmark to build the wrong-width model.
- **MIL pooling ablation**: `model.py`'s `GatedAttentionMIL` now has two
  siblings, `MeanPoolingMIL`/`MaxPoolingMIL`, behind the same
  `forward(z_bag, h_bag) -> (prob, attn_or_None)` interface and a
  `MultiSmokeCancerNet(pooling=...)` switch (`MIL_POOLINGS` dict) — the
  encoder, heads, and loss are all unchanged; only the aggregator swaps.
- **Grouped CV runs over train+val only** (`benchmarks/cross_validation.py`),
  never test; a fold that fails `train.py::check_mil_eligibility` is caught
  and recorded as an undefined result with its reason rather than crashing
  the run or being coerced to a filler AUROC.
- **Calibration/threshold freezing** (`benchmarks/calibration.py`) is a new,
  separate mechanism from the model's own fixed 0.70 cutoff mentioned
  elsewhere in this document — `FrozenThresholdPolicy.apply_to_test` is
  built to raise on a second call, structurally preventing the
  fit-calibration-then-peek-then-refit pattern that would invalidate a test
  evaluation.

A second pass on this same branch fixed a fold-level leakage bug of the same
character as §11's whole-dataset-scaling fix, one level deeper: grouped CV
was reusing the context's OUTER `PreprocessingArtifact` (fit on ALL
original-train subjects) across every fold, so an inner-CV-validation
subject's expression had already influenced the scaling/HVG selection it was
then evaluated against. `benchmarks/fold_preprocessing.py` now refits a
fresh artifact per fold from `context.normalized_adata_for_refit` (the
pre-HVG, pre-scaling normalized expression `run_pipeline_split_aware()`
captures before its own `fit_preprocessing` call), using only that fold's
training subjects — and the same fold-specific cell/bag datasets fixed a
matching cross-task leak where cancer-CV's MIL encoder pretraining used to
always see the OUTER train/val split regardless of the actual CV fold.
Ground-truth malignancy labels were removed as a Task B feature (a real
outcome-proxy risk for sources like TCGA); leave-one-source-out now refits
per held-out source and defaults undeclared source metadata to
`NOT_COMPARABLE`; a single-class training fold (expected in small CV folds)
no longer crashes a baseline or silently mis-indexes `predict_proba`'s
positive-class column (`baselines.py::positive_class_proba` maps via
`classes_`); logistic regression and the small MLP fit a `StandardScaler`
scoped to each call's own data; and statistical comparison now reports a
seed-level bootstrap CI (independent samples) alongside the descriptive
fold-level one (overlapping, not independent) that `summarize_comparison`
actually requires before calling a model "meaningfully better".

A third pass fixed a related second-order instance of the same fold-level
leakage class: the refit snapshot (`normalized_adata_for_refit`) used to be
captured BEFORE `annotate_cell_types()` ran, so every fold/OOD
reconstruction silently saw `cell_type_id=0` for every cell regardless of
its real CellTypist annotation — fixed by moving that call to run exactly
once, before the snapshot (`preprocess.py`). (The fourth pass below replaced
the majority-voting mode this call used, which the rest of this sentence
used to justify running only once, with an inductive per-cell mode — see
below.) This pass also added: real nested grouped-CV hyperparameter
selection for the classical baselines (`benchmarks/hyperparameter_search.py`,
computed strictly within the outer-train partition, wired into Task B's
final frozen-test path); `cap_cells_per_subject` wired into the neural
adapter's per-fold cell-level training, applied independently per split;
leave-one-source-out now requires an explicit `reference_species` (never
inferred from lexicographic source-name order) and rejects a subject
assigned to more than one `dataset_source`; `ExperimentContext` now rejects
a manifest subject missing from its cell dataset (previously checked only
the reverse direction), rejects blank/placeholder bag subject IDs, and
exposes fingerprinted `run_identity()` that a checkpoint/result reload can
verify against; and a durable, restart/concurrency-safe one-time frozen-test
guard (`benchmarks/test_guard.py`, atomic `O_CREAT|O_EXCL` file creation)
was added, wired into Task B's final path as an opt-in
(`benchmarks.frozen_test_guard_dir`) feature — made mandatory in the fourth
pass below.

A fourth pass closed the five remaining blockers to treating this framework
as scientifically load-bearing (see README's Benchmarking framework section
for the full list, generated fresh per run in `report.md`'s limitations
section):

1. **Inductive cell-type annotation.** `annotate_cell_types()`
   (`data/transforms.py`) now calls CellTypist with `majority_voting=False`
   (CellTypist's own default) instead of `True` — a cell's predicted label
   becomes a pure function of that cell's own expression, independent of
   which other cells (in particular held-out validation/test cells) are
   present in the same call. The previous `majority_voting=True` mode's
   over-clustering pass, run across train+val+test together, let held-out
   cells influence a training cell's own annotation — the actual bug the old
   "must run once, before the snapshot" comment was unknowingly working
   around rather than fixing. The fixed `CELL_TYPE_MAP` table is now
   fingerprinted (`cell_type_map_fingerprint()`) and persisted on
   `PreprocessingArtifact` for audit.
2. **Neural/MIL candidates can win the final frozen-test evaluation.**
   `benchmarks/final_evaluation.py::select_final_candidate` ranks every
   requested model (baseline or MIL) by CV/development evidence alone; the
   previous restriction to `CANCER_BASELINES` for the final path is gone.
3. **The frozen-test guard is now mandatory for every non-synthetic run**
   (`test_guard.py::default_guard_dir`, `ExperimentContext.
   guard_identity_fingerprint()`, `runner.py`'s
   `FrozenTestGuardDisabledInRealModeError`) — a safe default location is
   derived from the run's own output root, keyed by scientific identity
   (manifest + preprocessing + label-mapping + config + selected model), not
   by `run_id`. Disabling it is possible only through an explicit,
   synthetic-only config/CLI flag.
4. **Hyperparameter selection is integrated into every outer CV fold.**
   `hyperparameter_search.py::select_nested_hyperparameters_with_refit` runs
   a real inner grouped-CV — refitting preprocessing from only each inner
   fold's own training subjects — inside every outer fold of
   `cross_validation.py::run_smoke_cv`/`run_cancer_cv`, for classical
   baselines on both tasks and, via small bounded fixed candidate sets, for
   the neural/MIL models too.
5. **The final development/fit/calibration protocol now uses the whole
   train+val pool correctly**, replacing the previous "final model fit on
   train only, calibrated on val only" split:
   `final_evaluation.generate_subject_oof_predictions` produces
   subject-grouped out-of-fold predictions across the WHOLE development
   pool (each OOF subject predicted by a fold-refit model that never saw
   it); calibration/threshold are fit exclusively from those OOF
   predictions; `final_evaluation.refit_final_candidate_on_dev_pool` then
   fits ONE final preprocessing artifact and model on all development
   subjects (using the already-selected configuration, never reselected)
   before the single guarded test evaluation.

A fifth pass closed five further problems the fourth pass's own claims did
not actually hold up to (see README's Benchmarking framework section for
the full list):

1. **The frozen-test guard is now acquired before ANY test access**, not
   just before the final metric computation. `run_cancer_task`
   (`benchmarks/runner.py`) is split into a development-only stage —
   `select_final_candidate`, `generate_subject_oof_predictions`,
   `fit_final_candidate_on_dev_pool` (`final_evaluation.py`), none of which
   accept test subject IDs/bags/labels as arguments — and a guarded stage
   whose only test-touching call, `evaluate_frozen_test`, runs strictly
   inside the `try` block that follows `FrozenTestGuard.acquire()`.
   Previously test labels/predictions were already computed by
   `refit_final_candidate_on_dev_pool` before the guard was acquired.
2. **OOF predictions are now selection-clean.** The old
   `generate_subject_oof_predictions` accepted one globally-selected
   hyperparameter dict and reused it for every OOF fold — an OOF-held-out
   subject's own label had already influenced the configuration used to
   predict it. `final_evaluation._oof_fold_hyperparameters` now runs a
   fresh inner grouped-CV selection per OOF fold, using only that fold's
   OOF-training subjects, mirroring the same nested pattern the outer CV
   loop already used.
3. **The final MIL fit trains on every eligible development subject.**
   `Trainer.phase1_final_fit`/`phase2_final_fit` (`train.py`) are new
   fixed-epoch training methods with no internal validation carve-out or
   validation-based checkpoint selection; `NeuralCancerAdapter.fit_final`
   (`benchmarks/neural.py`) uses them for the final dev-pool refit. The
   fourth pass's claim that only classical baselines used every development
   subject (with MIL "documented" as carving out a validation slice) is
   resolved, not merely disclosed.
4. **CellTypist failure fails loudly by default.** `annotate_cell_types()`
   (`data/transforms.py`) previously printed a warning on any CellTypist
   exception and returned `adata` unchanged — `obs["cell_type_id"]` ended
   up missing or stale, not actually defaulted to anything despite the
   printed message claiming "defaulting to epithelial". It now raises
   `CellTypeAnnotationError` unless `allow_diagnostic_fallback=True` is
   explicitly passed (never by `preprocess.py`'s real pipeline entry
   points); the fallback path stamps
   `PreprocessingArtifact.cell_type_annotation_degraded=True`, which
   `ExperimentContext.from_pipeline_result` rejects outright.
5. **Real OOF predictions are persisted.** `predictions/cancer_<candidate>_oof.csv`
   now carries one row per development subject with its actual OOF
   probability and fold-local selected-hyperparameters/fingerprint columns
   (`runner.py::_write_oof_predictions_csv`) — previously only fold
   membership counts were written under `calibration_report["oof_summary"]`.

Full list of what's fixed vs. still open: README's Benchmarking framework
section.
