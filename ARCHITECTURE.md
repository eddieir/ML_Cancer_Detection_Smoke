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
checksum verification and optimizer/scheduler resume; bulk/single-cell/MIL
mode separation; species/ortholog-mapping safety beyond the config-driven
check in §12's leave-one-source-out; dose-head supervision gating.
Subject-aware class -> subject -> cell sampling wired into `train.py`'s own
curriculum (`data/sampling.py`) is now implemented — see §13.

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

A sixth pass closed five further problems (see README's Benchmarking
framework section for the full list):

1. **Task B eligibility no longer reads test labels/class counts.**
   `eligibility.check_task_b_eligibility` is replaced by
   `check_task_b_development_eligibility(train_bags, val_bags)` — a
   signature that structurally cannot accept `test_bags` — as the sole gate
   run before the guard, plus `check_test_evaluability(test_bags)`, run only
   inside the guarded stage, which reports (never rejects on) undefined
   AUROC/AUPRC for a one-class test split.
2. **Guard identity is now deterministic across runs of the same
   configuration.** It previously embedded `fitted.model_metadata`
   (`fit_seconds` for neural/MIL candidates — real wall-clock timing).
   `benchmarks/model_fingerprint.py` adds `torch_state_dict_fingerprint`
   (canonicalized `state_dict` tensor bytes) and
   `sklearn_model_state_fingerprint` (canonicalized fitted attributes);
   `run_cancer_task` now passes `final_model_state_fingerprint` plus a
   `calibration_fingerprint` and the new `ExperimentContext.
   test_membership_fingerprint` (derived only from `split_manifest.
   test_subjects`) into `guard_identity_fingerprint`'s `extra` payload,
   never the raw metadata dict.
3. **The guarded transaction now persists a complete, verified frozen-test
   result before `mark_completed`.** `calibration/frozen_test_result.json`
   (fingerprints, threshold, aggregate metrics, membership fingerprint, its
   own `artifact_fingerprint` — no raw labels/probabilities) is written
   atomically, reloaded, and verified; the guard's completed record
   references that exact fingerprint. Any failure in evaluation,
   calibration, serialization, or verification marks the guard failed.
4. **Every JSON/CSV write is now genuinely atomic.** `benchmarks/atomic_io.py`
   (temp file in the destination directory, fsync, `os.replace()`) backs
   `reporting.py::write_json`/`write_csv_table` and
   `test_guard.py::FrozenTestGuard.mark_completed`/`mark_failed` (acquisition
   itself still uses `O_CREAT | O_EXCL`, a separate exclusive-creation
   primitive). The guard also records a per-acquisition `owner_token`;
   `FrozenTestGuardOwnershipError` is raised if a `FrozenTestGuard` instance
   that never itself acquired the guard tries to finalize it.
5. **OOF CSV fingerprints are now real hashes, not raw JSON.**
   `training_subjects_fingerprint`/`validation_subjects_fingerprint` are
   SHA-256 of the canonical sorted subject list; every predicted row also
   carries `selected_params_fingerprint` and the fold's
   `model_state_fingerprint`. The file is written atomically, reloaded and
   row-count-verified, and its own SHA-256 is recorded as
   `oof_summary.oof_artifact_fingerprint` in `calibration/frozen_policy.json`.

A seventh pass closed three further problems (see README's Benchmarking
framework section for the full list):

1. **`atomic_write_bytes` now guarantees complete writes.** `os.write()` is
   only guaranteed to write up to the requested byte count — a short write
   is expected OS behavior, not an error. `_write_all` now loops until every
   byte is written, retries `InterruptedError`, and raises on zero-byte
   progress; any failure before `os.replace()` (write/fsync/close) now
   always removes the temp file and leaves the previous destination intact.
2. **`model_fingerprint.py` no longer falls back to `repr()`.** The
   fallback was unsafe: `sklearn.tree._tree.Tree` (every random forest's
   actual tree structure) is a Cython extension type with no `__dict__`, so
   it previously reached `repr(obj)`, which embeds a memory address —
   non-deterministic across processes. Canonicalization now explicitly
   handles `Tree`, `HistGradientBoostingClassifier`'s
   `TreePredictor`/private `_predictors`/`_bin_mapper` state (none of which
   follow the trailing-underscore convention a generic reflection walk
   relies on), and `DummyClassifier`'s `constant` (a constructor parameter,
   not a fitted attribute, that nonetheless determines
   `strategy="constant"`'s predictions — the single-training-class fallback
   model). Anything unsupported now raises `UnsupportedModelStateError`.
3. **CellTypist/scikit-learn pretrained-model compatibility is now detected
   and surfaced, not silently absorbed.** CellTypist's `Immune_All_Low.pkl`
   was serialized with scikit-learn 0.24.1; loading it under 1.9.0 always
   emits `InconsistentVersionWarning` — an unresolved upstream gap this
   project cannot fix directly. `annotate_cell_types` does not fail on it by
   default (that warning was already present in every prior passing CI run;
   defaulting to a hard failure would be a regression, not a fix);
   `strict_sklearn_compatibility=True` / `CELLTYPIST_STRICT_SKLEARN_COMPAT=1`
   turned it into `CellTypistCompatibilityError` for callers who wanted to
   enforce matching versions (a ninth pass below later replaced this with an
   unconditional fail-closed default for real runs). The warning itself is
   never suppressed.

An eighth pass closed six further problems (see README's Benchmarking
framework section for the full list):

1. **`FrozenTestGuard.acquire()` no longer risks a truncated guard.** Writing
   directly into an `O_CREAT | O_EXCL`-opened destination left a window where
   a concurrent reader could see an existing-but-incomplete guard file. It
   now writes the full payload to a temp file, fsyncs it, then atomically
   `os.link()`s it into place — the destination only ever appears fully
   formed. A real multiprocess race test (six `spawn`-context processes)
   proves exactly one winner. Corrupted guard JSON now raises
   `FrozenTestGuardCorruptedError` rather than being treated as absent.
2. **`model_fingerprint.py`'s last generic `__dict__` fallback removed**,
   replaced with an explicit whitelist (`LogisticRegression`,
   `StandardScaler`, `RandomForestClassifier`, `MLPClassifier`,
   `DecisionTreeClassifier`, plus the previously-added `Tree`/
   `TreePredictor`/`HistGradientBoostingClassifier`/`DummyClassifier`
   branches). Tightening it surfaced a second real bug:
   `HistGradientBoostingClassifier._bin_mapper` was silently falling through
   the old fallback. Anything else now raises `UnsupportedModelStateError`;
   determinism is tested across independent OS processes, not just repeated
   in-process calls.
3. **Strict cell-type provenance validation.** The historical
   `getattr(artifact, "cell_type_annotation_degraded", False)` check treated
   a *missing* provenance field as safe. `data/preprocessing.py::
   validate_cell_type_provenance` now requires `degraded is False` exactly,
   `mode` in an explicit allow-list (`inductive_per_cell` or the new
   `pseudo_bulk_no_cell_type_identity`), and — for `inductive_per_cell` — a
   well-formed fingerprint matching the current `CELL_TYPE_MAP` exactly.
   `fit_preprocessing()` now propagates these fields from `adata.uns` onto
   every artifact it produces, closing a gap where per-fold/per-OOD refits
   (`fold_preprocessing.py`) previously produced artifacts with unset
   provenance regardless of the outer artifact's real state.
4. **CellTypist/scikit-learn compatibility provenance persisted.**
   `cell_type_annotation_compatibility` (celltypist version, model name,
   runtime/serialized sklearn versions, a `compatible` boolean) is now
   recorded on the AnnData and propagated onto every `PreprocessingArtifact`.
   The default warn-vs-fail policy from the sixth pass is unchanged — making
   a real run fail closed on this mismatch by default remains open (see
   README's limitations).
5. **CI dependency installation reproducibility.** `requirements.txt` was
   documented as "exact versions" while its entries are lower bounds. A new
   `constraints-ci.txt` pins the exact Linux/Python-3.11 combination CI is
   validated against; the workflow installs via
   `pip install -r requirements.txt -c constraints-ci.txt`, pins
   Python 3.11.15, and records dependency versions in the CI log.
6. **CI workflow triggers fixed** (was hardcoded to a since-abandoned
   feature-branch name) and **new reproducibility artifacts**
   (`environment.json`, `preprocessing/final_artifact.json`) added to every
   run directory.

A ninth pass fixed a broken CI job and closed the CellTypist-compatibility
default-behavior gap the eighth pass had left open (see README's
Benchmarking framework section for the full list):

1. **CI's test step was actually failing** (`No module named pytest`):
   `constraints-ci.txt` only narrows an already-requested install, it does
   not add pytest as a dependency, and `requirements.txt` never listed it.
   Fixed with `requirements-test.txt` installed alongside `requirements.txt`
   plus a `python -m pytest --version` verification step.
2. **CI's "Record environment" step was silently faking its output**: it
   called `importlib.metadata.version(...)` without importing
   `importlib.metadata`, looked up the import name `sklearn` instead of the
   distribution name `scikit-learn`, and was wrapped in `|| true` — so every
   lookup failed, printed `UNAVAILABLE`, and never failed the build. Replaced
   with `src/benchmarks/env_versions.py`, a small tested utility shared by
   CI and by `reporting.py`'s environment-artifact writer, that fails loudly
   (non-zero exit / raised `RuntimeError`) if a required package is missing.
3. **Real preprocessing now fails closed on a CellTypist/scikit-learn
   version mismatch by default.** `strict_sklearn_compatibility` and
   `CELLTYPIST_STRICT_SKLEARN_COMPAT` are gone; strictness is now tied
   unconditionally to the existing `allow_diagnostic_fallback` flag (real
   pipeline entry points never set it). A real call raises
   `CellTypistCompatibilityError` undisturbed (never wrapped in the generic
   `CellTypeAnnotationError`); tolerating the mismatch via
   `allow_diagnostic_fallback=True` now always stamps
   `cell_type_annotation_degraded=True`, so real `ExperimentContext`
   construction rejects it through the existing degraded-provenance guard.
4. **`fold_preprocessing.py::artifact_fingerprint` now includes cell-type
   and compatibility provenance**, closing the fingerprint-scope gap the
   eighth pass had disclosed as open.
5. **CI action versions upgraded** to `actions/checkout@v7` and
   `actions/setup-python@v6` (both verified on Node 24, resolving the
   Node 20 deprecation warning).

Full list of what's fixed vs. still open: README's Benchmarking framework
section.

## 13. Subject-Aware Class-Imbalance Correction ("Phase 2")

Referred to as "Phase 2" in README/PR history — distinct from §5's
per-run training-curriculum "Phase 1/2/3" (cell-level pretraining /
aggregator training / end-to-end fine-tuning), which is unchanged by this
work. Full rationale, configuration reference, and ablation protocol: see
README's "Phase 2 — subject-aware class-imbalance correction" section.
Architectural summary:

- **`data/sampling.py`** is a new, standalone module: `SubjectClassIndex`
  (validated class -> subject -> cell-index mapping, built once from a
  training `CellLevelDataset`'s `subject_ids`/`smoke` arrays) and
  `SubjectBalancedBatchSampler` (a `torch.utils.data.Sampler` yielding
  batches of dataset indices by drawing class, then subject within that
  class, then cell within that subject, on every single index — never a
  per-cell inverse-frequency weight applied directly, which does not
  prevent a cell-heavy subject from dominating a class). Consumed via
  `DataLoader(dataset, batch_sampler=...)`, never combined with
  `shuffle=True`.
  - `cells_per_subject_cap` is a **hard per-batch maximum**, not a required
    minimum — a subject with fewer cells than the cap is still a valid
    participant. Each subject's *effective* capacity, computed once at
    construction (`_effective_capacity`), is `cap` when `replacement=True`
    (redraws allowed) or `min(cap, that subject's own unique cell count)`
    when `replacement=False` (cannot yield more distinct cells than it
    has). Per-batch remaining-capacity bookkeeping
    (`class_remaining_subjects`) is tracked against this effective value:
    once every subject of a class is at ITS effective capacity within the
    batch being built, that class is excluded from the remaining draws
    (probability renormalized over the classes still eligible) — there is
    no fallback to an already-exhausted subject. Feasibility is judged
    against the sum of effective capacities across all subjects (not
    `cap * n_unique_subjects`) versus the largest batch the sampler will
    ever need to produce; this is checked once at construction, and an
    infeasible configuration raises `SamplingImpossibleError` immediately,
    not mid-iteration. Without-replacement uniqueness is scoped to one
    batch — the same physical cell may reappear in a later batch — and
    capacity always resets fully between batches.
  - `samples_per_epoch` is resolved into an exact, explicit list of
    per-batch sizes (`_batch_sizes`) at construction — full `batch_size`
    batches followed by exactly one partial batch of the exact remainder —
    rather than a batch count derived by ceiling division and then filled
    with full-size batches (which would silently over-sample). `__len__`
    returns `len(_batch_sizes)`.
  - After a batch_sampler-driven `DataLoader` completes one full epoch,
    `last_realized_diagnostics` is populated with per-batch realized
    provenance: `realized_total_samples`, `realized_batch_sizes`,
    `realized_cells_per_subject`, `realized_subject_counts_per_batch`,
    `realized_max_subject_cells_per_batch`, and
    `realized_repeated_cell_draws_per_batch`/`_total` — computed
    incrementally during `__iter__` and committed to
    `last_realized_diagnostics` only after every batch has yielded, so an
    interrupted or failed epoch leaves the prior value (or `None`)
    unchanged rather than exposing a mislabeled partial result. `epoch_index`
    and `complete` identify which epoch a realized block describes.
- **`Trainer._train_cell_loader()`** (`train.py`) is the single place every
  training cell `DataLoader` is now built (`phase1`, `phase1_final_fit`,
  `phase3`'s cell-level component) — it reads
  `self.smoke_imbalance_config["sampler"]` (`"shuffle"` |
  `"subject_balanced"`) and dispatches accordingly. Validation/test
  `DataLoader`s are built inline elsewhere with `shuffle=False`, exactly as
  before, and never call this method.
- **`Trainer._smoke_loss_weights_and_type()`** resolves
  (reported class weights, loss alpha, loss type, focal gamma) from
  `self.smoke_imbalance_config` — `class_weighting` (`"none"` |
  `"inverse_frequency"`) controls whether weights are computed at all
  (train-partition-only, via the pre-existing
  `CellLevelDataset.smoke_class_weights`, unchanged); `loss` (`"cross_entropy"`
  | `"focal"`) selects `model.py`'s new `FocalLoss` vs. the existing
  `nn.CrossEntropyLoss`; `focal_alpha_mode` (`"class_weights"` | `"none"`)
  is the explicit double-correction guard for when both subject-balanced
  sampling AND inverse-frequency weighting are active simultaneously.
- **`model.py`'s `FocalLoss`** — standard per-example
  `(1-pt)**gamma * CE` formulation with `pt` derived from **unweighted**
  cross-entropy; class alpha (if given) is applied exactly once, multiplied
  onto the already gamma-modulated per-example loss, never folded into the
  cross-entropy term `pt` is computed from (which would let alpha distort
  the focal modulation itself, not just the final scale). `gamma=0` without
  alpha is mathematically identical to plain unweighted `CrossEntropyLoss`
  under the same reduction; `reduction="mean"` is a plain arithmetic mean
  (not `CrossEntropyLoss(weight=...)`'s weight-normalized mean) — a
  deliberate, documented, tested choice. Constructor/forward validate
  `gamma >= 0`, `class_weight` shape/finiteness/non-negativity, class-count
  match, and target validity. `MultiTaskLoss` gained `loss_type`/
  `focal_gamma` constructor parameters; the malignancy/cancer/
  dose-response loss terms are untouched.
- **`data/sampling.py::resolve_smoke_imbalance_config`** merges a
  (possibly absent) `train.smoke_imbalance` config block with an explicit
  default dict that reproduces exact pre-Phase-2 behavior
  (`sampler: shuffle`, `class_weighting: inverse_frequency`,
  `loss: cross_entropy`) — absent configuration changes nothing for
  existing configs/checkpoints. Every value is validated at resolution
  time (`SamplingConfigurationError` on anything invalid), never silently
  coerced.
- **Provenance**: `train.smoke_imbalance` lives inside
  `ExperimentContext.config`, so §12's `config_fingerprint`/
  `guard_identity_fingerprint` already change with it — no separate
  fingerprint plumbing was needed. `Trainer._save()` persists the resolved
  `smoke_imbalance_config` and (when the subject-balanced sampler ran) its
  realized per-epoch `SamplingDiagnostics` in every checkpoint;
  `NeuralSmokeAdapter`/`NeuralCancerAdapter.metadata()` expose the same
  fields for benchmark reports.
- **`benchmarks/imbalance_ablation.py::run_smoke_imbalance_ablation`** — a
  focused, standalone experiment function (not folded into §12's nested-
  hyperparameter-search machinery, since imbalance strategy is a
  qualitatively different axis than a hyperparameter grid) comparing five
  named strategies over identical outer folds/seeds/preprocessing/
  architecture, development data (context's train+val pool) only.
  - Only `cap_cell_dataset` on the fold's TRAINING split; every strategy in
    a fold shares the fold's exact, uncapped, natural validation set —
    each fold record carries a `validation_fingerprint` proving this.
  - Primary metrics (`subject_level`, via `metrics.py::
    subject_weighted_full_smoke_metrics_report`) are majority-voted to one
    prediction per subject before computing macro-F1/balanced accuracy/
    per-class precision-recall-F1-support/confusion matrix; the equivalent
    cell-level numbers are kept only under `cell_level_diagnostic`.
  - `comparisons` reuses `reporting.py::compare_models`/
    `summarize_comparison` (§12's own statistical-comparison machinery) to
    report win/tie/loss counts, mean/median paired differences, and a
    seed-level bootstrap CI when >=2 seeds ran, with `summary.
    meaningfully_better` staying `False` whenever the evidence doesn't
    support a confident selection.
  - `write_imbalance_ablation_artifact`/`read_imbalance_ablation_artifact`
    persist/reload the report as a schema-versioned JSON (+ CSV) artifact
    under `<run_dir>/metrics/`, using the same `atomic_io.py` writes and
    `_environment_snapshot` every other benchmark artifact uses. Wired into
    `benchmarks/runner.py`'s CLI via `--imbalance-ablation` (smoke task only).
  - The default `configs/default.yaml` `sampler: shuffle` is unchanged by
    this module's existence — no real (non-synthetic) ablation evidence has
    yet been produced, so the pre-Phase-2 default remains authoritative
    until such evidence exists.

## 14. Dataset Provenance, Label State, and Species/Assay Boundaries ("Phase 3/4")

This section covers the additions layered on top of §11's existing
split/leakage machinery: explicit dataset provenance, a shared label-state
vocabulary, and hard boundaries around cross-species and bulk/single-cell
mixing.

### 14.1 Dataset manifest and provenance flow

`configs/datasets.yaml` (checked in) records the facts about each dataset
that don't depend on the local filesystem: accession, source/official-record
URLs, species, assay type, identifier fields, and documented limitations.
`src/data/manifest.py::build_dataset_manifest` reads that seed and adds the
facts that DO depend on what's actually downloaded — real SHA-256 checksums
for files present under `data/raw/<raw_subdir>`, `files_present=False` and
`null` checksums for anything not found. `DatasetManifestEntry.validate()`
refuses an entry missing a required provenance field, and refuses a
checksum recorded without `files_present=True` (a checksum must never be
fabricated for a file that wasn't actually hashed).
`manifest_fingerprint()` hashes every entry's provenance fields (excluding
`download_date`, which changes without the underlying data changing) — this
is the value meant to be folded into `PreprocessingArtifact`/experiment
identity so a changed accession or label-policy version changes what an
experiment "is."

### 14.2 Label-state model

`src/data/label_state.py` defines five states — `known_positive`,
`known_negative`, `unknown`, `not_applicable`, `excluded_by_policy` — and a
`LabelProvenance` record (status/source/method/confidence/limitation) for
any label-producing code to attach to a value. `malignancy_known`/
`cancer_label_known` (`data/assembly.py`, `data/labellers.py`, `model.py`'s
`MultiTaskLoss`) were the first two enforcement points; `smoke_type_known`
is the same pattern applied to the smoke label, gating:

- `train.CellLevelDataset.smoke_class_weights` (class-weight computation)
- `data.sampling.SubjectClassIndex` (subject-balanced sampling's class
  index — a cell with `smoke_known=False` keeps its true position but is
  never added to any subject's sampleable index)
- `model.MultiTaskLoss._ls` (the smoke-classification loss term)
- `evaluate.Evaluator._known_smoke_metrics` (accuracy/F1/confusion matrix)
- `preprocess.py::run_pipeline_split_aware`'s subject-level stratification
  (a subject with no known smoke label is stratified as `None` — pooled,
  unstratified placement — not counted toward any class's proportion)

`data/converters.py::_load_gse136831_cell_metadata` is the first real
producer of `smoke_type_known=False` rows: GSE136831 has no verified
per-subject smoking record, so every cell defaults to unknown, and
`weak_smoke_proxy_*` fields carry COPD status as a separate, clearly
labeled proxy. `data/labellers.py::apply_weak_smoke_proxies` is the only
path that ever promotes a weak proxy into the primary smoke label, gated
by `data.weak_labels.enabled` (default `false`).

`data/nlst_smoking.py::parse_nlst_smoking_row` is the explicit parser for
NLST's `CIGSMOK`/`CIGAR` fields: it only recognizes the codes documented in
`src/data/downloaders.py::print_nlst_instructions` (`CIGSMOK` 1/2,
`CIGAR` 1) as verified evidence, and treats every other value — missing,
blank, null, an undocumented code, or malformed input — as unknown, never
raising and never guessing at an unconfirmed codebook meaning.
`data/labellers.py::transfer_nlst_labels` calls it per matched subject and
writes `smoke_type_known` plus `smoke_type_source`/`smoke_type_method`/
`smoke_type_limitation` provenance columns (default `None` for every cell
this function doesn't touch); it can overwrite an upstream
`smoke_type_known=True` back to `False` when NLST linkage itself finds no
usable evidence, since NLST is this project's intended source of truth for
a scRNA-seq subject's smoking status. The same pattern (a per-sample value
that doesn't parse to a documented code stays unknown rather than
inheriting the accession-level default) applies to GSE994/GSE123352
(`data/converters.py::_infer_smoke_column`) and GSE307690/CANUCK
(`convert_canuck`'s empty-metadata case).

### 14.3 Subject-split boundary and fit/transform preprocessing interface

Unchanged from §11: `run_pipeline_split_aware` computes the subject-level
split before `fit_preprocessing` runs, and `fit_preprocessing`/
`apply_preprocessing` (`data/preprocessing.py`) enforce the fit/transform
split for HVG selection and scaling. This section's additions don't modify
that interface; `tests/test_leakage_regression.py` re-exercises the
guarantee alongside the newer boundaries below in one file.

### 14.4 Cross-species boundary

`src/constants.py` defines four experiment modes (`human_only` — the
default, `mouse_only`, `cross_species_pretraining`,
`cross_species_domain_adaptation`). `src/data/species_policy.py` is the
single enforcement point:

- `preprocess.py::_load_all_sources` never loads the mouse source
  (GSE288003) at all unless `data.experiment_mode` excludes `human_only`.
- Every loader stamps `obs["species"]`; `data/loaders.py::load_mouse_scrna`
  namespaces subject/animal IDs (`mouse::<id>`) so they cannot collide with
  a human `subject_id`.
- `data/assembly.py::merge_sources` checks every source's species against
  `experiment_mode` before concatenating anything, and raises
  `SpeciesPolicyError` if more than one species is present without an
  explicit cross-species mode.
- `data/ortholog.py` replaces the old uncached, "pick the first match"
  live BioMart call with a versioned `OrthologMappingArtifact`
  (source/release/policy/mapping/counts), resolvable from an injected
  fixture, a cached file, or (last resort) a live query.

`cross_species_pretraining`/`cross_species_domain_adaptation` are
implemented only as safe hooks: they allow `merge_sources` to combine
species when explicitly requested, but there is no pretraining loop or
domain-adaptation training procedure built on top of them in this change.
Both remain disabled by default.

### 14.5 Bulk/single-cell boundary

`src/constants.py` defines `human_single_cell`/`bulk_tcga` assay modes.
Every loader stamps `obs["assay_mode"]`, defaulting to `human_single_cell`;
`data/converters.py::convert_tcga` is the only producer of
`assay_mode="bulk_tcga"` (written into its `samples_meta.csv` output,
picked up by `data/loaders.py::load_microarray`). Enforcement:

- `configs/default.yaml`'s `microarray_sources` no longer lists TCGA-LUAD/
  TCGA-LUSC — they moved to `data.tcga.bulk_sources`, a separate config key
  the default single-cell loading path never reads.
- `preprocess.py::_load_all_sources` additionally checks every loaded
  source's `assay_mode` column directly and raises `AssayModeError`
  (`data/assay_mode.py`) if any `bulk_tcga` row is present — a defensive
  check against a config that still manually lists a bulk source under
  `scrna_sources`/`microarray_sources`, not just reliance on the config
  default being correct.
- `preprocess.py::load_tcga_bulk_dataset` is the only sanctioned way to
  load TCGA's bulk matrices: it requires `data.tcga.enabled=true`,
  validates every loaded source actually carries `assay_mode="bulk_tcga"`,
  and raises `BulkTrainingNotImplementedError` if asked for a trainable
  dataset (`require_trainable=True`) — this project has no bulk RNA-seq
  model or training loop, and that gap is a raised error, not a silent
  fallback onto the single-cell model.
- TCGA's `sample_type`-derived malignancy (tumor vs. solid-tissue-normal)
  is written only into this bulk path's `samples_meta.csv`; it is never
  reachable from a single-cell `CellLevelDataset`/MIL bag because the
  bulk CSV itself never enters `_load_all_sources`.

`data/assay_mode.py::assert_no_pseudo_bulk_rows`/`assert_no_single_cell_rows`
remain available as generic guards for any additional single-cell-only or
bulk-only code path that needs the same check.

### 14.6 Controlled-access boundary (NLST)

`src/data/nlst_adapter.py` resolves a local data root from an environment
variable (`NLST_DATA_ROOT` by default, configurable via
`data.nlst.local_root_env`) — never a committed path, never a credential in
config. `check_nlst_availability`/`require_nlst_available` validate that
`screen.csv`/`prsn.csv` are actually present with the required columns
before anything downstream trusts them, and report unavailability as an
explicit status rather than substituting a fixture. No participant-level
row content is included in the availability report. This project has no
approved NCI Data Use Agreement in this environment; `src/data/
downloaders.py::print_nlst_instructions` documents the real, manual,
authorized-access steps.

### 14.7 Frozen-test access boundary

Unchanged: `src/benchmarks/test_guard.py` remains the sole point that may
touch held-out test data, and nothing in this change modifies it. The new
dataset-manifest and label-quality-report machinery in this section
operates entirely on development-partition data and provenance metadata; it
never inspects frozen-test labels or outcomes.

### 14.8 Label-quality report

`src/data/label_quality_report.py::build_label_quality_report` derives a
per-split/known-vs-unknown summary directly from
`run_pipeline_split_aware`'s existing return value (`split_manifest.report`,
`label_provenance_report`) rather than recomputing those counts
independently, so the report can't drift from what the pipeline itself
recorded. It flags at minimum: zero subjects with a known cancer outcome,
and classes that couldn't be stratified across splits. Persisted as JSON
(full detail) plus a compact per-split CSV summary.

### 14.9 Preprocessing artifact contract ("Phase 4")

The fit/apply boundary itself (§14.3) was already correct: `fit_preprocessing`
only ever sees training-subject cells, and `apply_preprocessing` only ever
subsets/reorders/scales using parameters already fixed at fit time. What
this section adds is a stricter, checkable contract around that boundary.

**Fingerprint hierarchy.** `PreprocessingArtifact.scientific_fingerprint()`
hashes a fixed, explicit field list (`_FINGERPRINT_FIELDS` in
`data/preprocessing.py`): gene list/order, scaling statistics, HVG count,
forced-marker list, fit cell/subject counts, label-mapping, cell-type
annotation provenance, and gene-contract policy. `created_at` and free-text
`notes` are excluded — they're informational, not scientific state. This is
a single flat fingerprint over the artifact itself, not a tree of
sub-fingerprints; `fold_preprocessing.py::artifact_fingerprint` (already
present) computes a related but distinct identity used specifically for
per-fold/OOF bookkeeping and is unchanged by this section.

**Checkpoint/artifact binding.** `Trainer._save()` embeds
`preprocessing_artifact_fingerprint` (and the artifact's gene count) in
every checkpoint, sourced from whichever artifact `Trainer.
set_preprocessing_artifact()` was given (`Trainer.from_experiment_context`
wires this automatically from `ExperimentContext.preprocessing_artifact`).
`Predictor.from_config()` recomputes the fingerprint of the
`preprocessing_artifact.json` it finds next to the checkpoint and raises
`ArtifactCompatibilityError` if the two disagree, or if the artifact's gene
count doesn't match what the checkpoint was trained with. A checkpoint
saved without a wired-in artifact (legacy runs, or a Trainer that never
called `set_preprocessing_artifact`) records `None` and is treated by
`Predictor.from_config` exactly as before this change: it proceeds only
under the existing `unsafe_legacy_mode` opt-in.

**Gene contract.** `verify_compatible()` now performs four checks, each
independently configurable on the artifact and defaulting to the
conservative option: duplicate identifiers in the input (always fatal —
no aggregation policy exists), missing required genes
(`missing_gene_policy`, default `error`; `zero_fill` is available only as
an explicit, recorded, non-default opt-in applied inside
`apply_preprocessing`, never silently), genes present but outside the
artifact's selected panel (`unexpected_gene_policy`, default `ignore` —
recorded in `uns["preprocessing_compatibility_diagnostics"]`, never fatal
by itself), and overall coverage against `minimum_gene_coverage` (default
`1.0`). `configs/default.yaml`'s `preprocessing.*` keys mirror these
defaults exactly, checked by a dedicated consistency test.

**Serialization.** `PreprocessingArtifact.save()`/`.load()` route through
`benchmarks/atomic_io.py`'s existing temp-file-plus-`os.replace()` atomic
write path (the same primitive `test_guard.py` and `reporting.py` already
use) and raise typed errors (`PreprocessingArtifactError`,
`GeneContractError`, `ArtifactCompatibilityError`, `LegacyArtifactError`)
on corruption, an unrecognized schema version, or a field-set mismatch,
rather than a generic exception or a silent partial load.

### 14.10 Per-fold artifact ownership

`fold_preprocessing.py::refit_artifact_for_fold` already fit an
independent `PreprocessingArtifact` per CV fold, from only that fold's
training subjects — a property that predates this section. What this
section adds is persistence: `save_fold_artifact(artifact, output_root,
fold_idx, train_subjects, val_subjects)` writes
`<output_root>/preprocessing/fold_XX/artifact.json` (via
`PreprocessingArtifact.save`, atomic) and, LAST, `manifest.json`
(deterministic train/val-subject fingerprints via SHA-256 over a
sorted-and-JSON-encoded subject-ID list, plus the artifact's own
`scientific_fingerprint()`, plus `status: "complete"`). Writing the
manifest last, after the artifact file is durably on disk, is what makes
an interrupted persist detectable: `load_fold_artifact()` raises
`IncompleteFoldArtifactError` if `manifest.json` is missing or doesn't
record `status="complete"`. `load_fold_artifact(expected_train_subjects=,
expected_val_subjects=, expected_artifact_fingerprint=)` verifies every
supplied expectation against what was actually persisted before returning
anything, raising `FoldArtifactMismatchError` on any mismatch — this is
the mechanism that stops one fold's persisted artifact from being reused
under another fold's identity. `cross_validation.py`'s
`run_smoke_cv`/`run_cancer_cv` gained an optional `artifact_output_root`
parameter (default `None`, preserving prior in-memory-only behavior
exactly) that wires this persistence in, keyed by `f"s{seed}_f{fold_idx}"`
so multiple seeds never collide on the same fold directory; `runner.py`'s
benchmark CLI always passes the run's own output directory.

### 14.11 Batch-correction safety gate

`data/transforms.py::assert_batch_correction_safe(transductive_batch_correction,
context_name)` raises `UnsafeBatchCorrectionError` when its first argument
is `True`. It is called from three places: `fold_preprocessing.py::
require_normalized_adata` (the single chokepoint every CV/OOF fold refit
and the final development-pool fit already funneled through), and
`runner.py`, immediately before frozen-test guard acquisition. Previously,
`ExperimentContext.transductive_batch_correction` was recorded and
reported but never used to gate anything — a run that opted into
disclosed, non-leakage-free Harmony at the outer-split stage could still
reach CV, OOF generation, the final fit, and frozen-test evaluation
without any of them refusing it. `PreprocessingArtifact.
batch_correction_status` (`"disabled"` default, or
`"transductive_diagnostic_only"`) is set from the same resolved
`bc_mode`/`allow_transductive_harmony` config values `preprocess.py`
already computed, moved earlier in that function so the artifact's own
field reflects the actual resolved mode rather than being patched onto an
already-constructed (and, by this section's own fit/apply contract,
treated-as-immutable) artifact after the fact. `fold_preprocessing.
artifact_fingerprint()` was changed from an independently-computed SHA-256
over a hand-picked field subset to a thin wrapper around
`PreprocessingArtifact.scientific_fingerprint()` — every caller (per-fold
records, OOF prediction records, the final development artifact, the
frozen-test guard identity, `ExperimentContext.
preprocessing_artifact_fingerprint`) now collides on exactly one
fingerprint definition instead of two that happened to agree by
construction.

### 14.12 Model bundle manifest

`benchmarks/bundle.py` defines a bundle: a directory holding a model
checkpoint (referenced by a bundle-relative path plus SHA-256, never
copied — checkpoints are large binaries this repository does not commit),
a copy of the `PreprocessingArtifact` it was trained with, and
`bundle_manifest.json` — model configuration, class vocabulary, label
policy, species policy, assay mode, dataset-manifest fingerprint, split
fingerprint, calibration state, decision threshold, an environment
snapshot, per-component SHA-256 hashes, and one `bundle_fingerprint`
covering the whole manifest (order-independent — computed over a
canonical sorted-key JSON encoding, so unrelated key reordering never
changes it, but any content change does). `write_model_bundle()` is the
only writer; `load_and_validate_bundle()` re-derives every hash and
fingerprint and raises `BundleCorruptionError` (missing/unparsable/wrong-
schema manifest, missing referenced file) or `BundleValidationError`
(hash/fingerprint/gene-count mismatch) before returning anything usable.
`validate_bundle_for_model(manifest, model)` additionally checks a
constructed model's `input_dim`/`num_smoke` against the bundle's own
gene count/class vocabulary. `validate_bundle_matches_identity(manifest,
expected)` is the resume-safety check: it compares
`dataset_manifest_fingerprint`/`split_fingerprint` against caller-supplied
expectations before a bundle is treated as reusable. `Trainer.
write_bundle()` (`train.py`) is the integration point: it requires
`self.preprocessing_artifact` to already be wired in (via
`set_preprocessing_artifact`, itself now called automatically by
`Trainer.from_experiment_context`) and refuses to build a bundle without
one. A directory with a checkpoint but no `bundle_manifest.json` — the
shape every pre-Phase-4 checkpoint directory has — raises
`LegacyBundleError` unless the caller passes `allow_legacy=True`
explicitly, mirroring `inference.py`'s existing `unsafe_legacy_mode` policy
rather than introducing a second, differently-shaped legacy contract.

### 14.13 Frozen-test access sentinels

`benchmarks/sentinel.py::FrozenAccessSentinel` is deliberately narrow in
scope: it does not gate WHEN test data may be evaluated (that remains
`test_guard.py`'s job, unchanged) — it makes "was this specific object
ever read" a directly testable property. Every dunder a real piece of test
data (an AnnData, a numpy array, a pandas object, a plain dict/list of bag
metadata) could plausibly be read through — `__getattr__`, `__iter__`,
`__getitem__`, `__len__`, `__array__`, `__bool__`, `__repr__`, `__eq__`,
`__contains__`, and others — raises `FrozenDataAccessError` immediately.
`tests/test_frozen_test_sentinel.py` wraps a real synthetic
`ExperimentContext`'s `test_bags`/`test_cell_dataset` in sentinels and runs
the actual development call sequence `runner.py` uses before guard
acquisition (`run_smoke_cv`, `run_cancer_cv`, `generate_subject_oof_predictions`,
`fit_final_candidate_on_dev_pool`) end to end — if any of those functions
ever touched test data, the test would fail with a traceback pointing at
the exact line that did.

**Scope note.** This section (§14.9–§14.13) hardens the artifact's own
identity/contract, its binding to a checkpoint and to a full bundle
manifest, per-fold artifact persistence, batch-correction gating at every
leakage-free protocol's entry point, and frozen-test access verifiability.
It does not introduce a new inductive batch-correction method — Harmony
remains transductive-only, and `UnsafeBatchCorrectionError` makes that
limitation enforced rather than merely documented, not fixed. It does not
change `test_guard.py` itself. These remain the section's honest
boundaries, not claims made beyond them.

## 15. Pathway-Aware Hierarchical Multi-Instance Network ("Phase 5")

`src/pathway_hierarchical_mil.py` implements a second, optional model —
code identifier `pathway_hierarchical_mil` — alongside `model.py`'s
`MultiSmokeCancerNet`. It is a research-candidate architecture: nothing in
this section, the module, or its tests claims clinical validity,
superiority over the existing baselines, or scientific novelty beyond
"this is a different architecture we implemented and unit-tested."
`MultiSmokeCancerNet` is unchanged and remains the default; the new model
is selected explicitly and never runs without a compatible preprocessing
artifact and gene-module artifact (§15.6).

### 15.1 Data flow

```
preprocessed expression, fixed artifact gene order      [G]
    -> masked gene-to-module projection (GeneModuleCollection-aligned)
    -> module activations                                [P]
    -> optional residual gene projection                 [R]
    -> fused, LayerNorm'd cell embedding                  [D]
    -> optional cell-type / source / species conditioning
    -> cell-type-local gated attention (Level 1)
    -> per-cell-type representation                     [C, D]
    -> cell-type gated attention (Level 2)
    -> subject representation                              [D]
        -> smoke-type head (multiclass logits)
        -> cancer-risk head (single logit)
        -> optional domain/source head (diagnostic only)
```

Every stage operates on already-preprocessed, fixed-gene-order expression.
Nothing in this file calls a preprocessing *fit* method; it only consumes
an already-fit `PreprocessingArtifact`'s gene order.

### 15.2 Gene-module contract

`GeneModuleCollection` binds a set of named modules to one exact, ordered
gene list (`gene_names`) via a boolean `membership_mask` of shape
`[n_modules, n_genes]`. Two construction paths:

- `from_gmt(path, gene_order, ...)` — parses a
  `module<TAB>description<TAB>GENE1<TAB>GENE2...` file, keeps only genes
  present in `gene_order` (unavailable genes are dropped from membership,
  never fed back into preprocessing), deduplicates genes within a module
  deterministically, and either drops or raises on a module whose aligned
  gene count falls below `minimum_genes_per_module`, per
  `empty_module_policy`.
- `synthetic(gene_order, n_modules, genes_per_module, seed)` — a
  deterministic, seeded, clearly-labelled (`source_name =
  "synthetic_diagnostic_v1"`) scheme with no participant data, used only by
  tests and by synthetic workflows that explicitly opt in via
  `gene_modules.allow_synthetic_modules: true`. The default configuration
  (`gene_modules.path: null`, `allow_synthetic_modules: false`) makes a
  real training run fail with an actionable configuration error rather
  than silently substituting synthetic modules.

`fingerprint()` hashes gene order, module order, and full membership;
changing any of them changes the model's `module_fingerprint`, which is
recorded alongside the checkpoint and (optionally) the bundle manifest's
`extra` fields, so a checkpoint built against one module set cannot be
silently paired with another. Module membership is decided once, from the
module source and the artifact's gene order alone — never from labels and
never from a validation or test split (see
`test_module_membership_decided_before_any_validation_split`).

### 15.3 Masked pathway encoder

`MaskedModuleProjection` holds a `nn.Parameter` weight the same shape as
`membership_mask`, but every forward pass recomputes
`effective_weight = weight * membership_mask` before the linear map, and a
backward hook on `weight` multiplies its incoming gradient by the same
mask — disallowed (gene, module) connections are exactly zero on every
forward pass and receive exactly zero gradient, not merely a small one.
`PathwayCellEncoder` combines the resulting module activations (GELU +
LayerNorm + dropout) with an optional compact residual gene projection,
fuses them, and LayerNorm's the result into a `D`-dimensional cell
embedding. LayerNorm is used deliberately instead of BatchNorm1d: a batch
of cells routinely mixes multiple subjects (and, when conditioning is
enabled, multiple sources), and BatchNorm1d's cross-example statistics
would leak information across subject boundaries in a way LayerNorm's
per-sample normalization does not.

### 15.4 Hierarchical attention pooling

Level 1 (`HierarchicalAttentionPooling`, cells -> cell-type
representation) computes one gated-attention score per cell
(`tanh(Vh) * sigmoid(Uh)`, scored by `w`), then, independently for each
cell-type bucket, takes a masked softmax over the cell dimension restricted
to real (non-padded) cells of that bucket for that subject. Buckets with no
matching cells produce an all-zero weight vector (`cell_type_present=False`
for that bucket) rather than a fabricated representation. Level 2 pools the
resulting (up to `num_cell_type_buckets`) cell-type representations with a
second gated attention, masked to only the buckets actually observed for
that subject. Both levels' weights sum to exactly one within any non-empty
group and exactly zero for an empty one (`masked_softmax`); attention is
computed independently per row of the batch, so no attention ever crosses
a subject boundary (verified directly by
`test_no_attention_computed_across_subjects`). A subject whose bag is
entirely padding fails immediately at the top-level `forward()` call rather
than silently producing a degenerate zero-vector prediction.

`num_cell_type_buckets` reserves one bucket (the highest index) for cells
whose type is unknown — an explicit policy, not a silent drop.

### 15.5 Multitask heads, masking, and loss

The smoke-type head and cancer-risk head both read the same subject
embedding. `MultitaskMaskedLoss` masks each task's inputs to only its
`*_known` examples before computing `F.cross_entropy` (smoke, with an
optional class-weight tensor applied exactly once) or
`F.binary_cross_entropy_with_logits` (cancer) — an unknown label never
becomes a fabricated negative and never contributes to either loss term. A
batch with zero known labels for one task contributes a differentiable
zero for that task only; a batch with zero known labels for *both* tasks
follows an explicit, tested `empty_batch_policy` ("skip" returns a
differentiable zero total loss; "error" raises `EmptyBatchLossError`).

### 15.6 Configuration, identity, and bundle integration

`configs/default.yaml`'s `model.pathway_hierarchical_mil` section
(`enabled: false` by default) mirrors `PathwayHierarchicalMILConfig`
field-for-field; `test_pathway_hierarchical_mil_yaml_defaults_match_python_dataclass`
keeps the two from drifting apart. `PathwayHierarchicalMILConfig.validate()`
rejects non-positive dimensions, out-of-range dropout, negative loss
weights, and an unrecognized `empty_batch_policy` before a model is ever
constructed. The model exposes `input_dim` and `num_smoke` attributes with
the same names `MultiSmokeCancerNet` uses, so the existing generic
`benchmarks/bundle.py::validate_bundle_for_model` check works unchanged for
either architecture; `write_model_bundle`'s `extra` argument is used to
record `model_type="pathway_hierarchical_mil"` and the model's
`module_fingerprint` in the bundle manifest, so a bundle built for this
architecture cannot be silently loaded against a mismatched gene-module
set (`test_bundle_round_trip_and_module_fingerprint_binding`,
`test_bundle_rejects_swapped_preprocessing_artifact`,
`test_bundle_rejects_corrupted_checkpoint`). Calibration reuses
`benchmarks/calibration.py::fit_calibration`/`FrozenCalibrator` unchanged —
both already operate on post-hoc numpy probability arrays and have no
model-specific logic to duplicate.

### 15.7 Honest scope boundaries

This implementation covers the model itself (gene-module contract, masked
pathway encoder, two-level hierarchical attention, optional source/species
conditioning, multitask heads and masked loss, uncertainty diagnostics
via predictive entropy and an explicitly-opt-in MC-dropout utility,
bundle-manifest identity binding) with unit and integration test coverage
across gene modules, the pathway encoder, hierarchical attention, multitask
masking, domain conditioning, and checkpoint/bundle identity, plus a
synthetic end-to-end training loop and a frozen-test-sentinel
non-access check. A second pass (§15.8) additionally wires the model into
`benchmarks/runner.py`'s CLI, the nested cross-validation loop, the
hyperparameter search, the OOF/final-development-fit protocol, and a
dedicated ablation runner.

A model-specific calibration fit (as opposed to reusing the existing
generic post-hoc calibrator) remains unimplemented. An adversarial domain-
training head was implemented in Phase 6 — see §16.4 — together with
CORAL/MMD regularizers and the source-held-out evaluation protocol those
strategies are compared under.

### 15.8 CLI, nested cross-validation, hyperparameter search, and ablation integration

`benchmarks/mil_registry.py` is the one place that maps a Task A/B
MIL-kind candidate name to its adapter class:
`NeuralCancerAdapter` for `"neural"`/`"mean_mil"`/`"max_mil"`/
`"attention_mil"` (unchanged), and the new
`PathwayHierarchicalAdapter` (`benchmarks/pathway_hierarchical_adapter.py`)
for `"pathway_hierarchical_mil"`. `PathwayHierarchicalAdapter.fit`/
`fit_final`/`predict_proba` intentionally mirror `NeuralCancerAdapter`'s
signatures exactly (including the unused cell-dataset positional arguments
and the `pretrain_epochs` keyword, which this adapter treats as its single
AdamW training loop's epoch count — the architecture has no separate
cell-level pretraining phase, so nothing needs a Phase 1 step), so
`cross_validation.py` and `final_evaluation.py` dispatch through
`build_mil_adapter()` at construction time and otherwise call every
MIL-kind candidate identically. `run_cancer_cv`'s existing `mil_names`
list, `final_evaluation.py`'s `is_mil_candidate`/`_fit_mil`/
`fit_final_candidate_on_dev_pool`, and `runner.py`'s
`_final_dev_pool_hyperparameters` were extended with one branch each
(`name == "pathway_hierarchical_mil"`) selecting this candidate's own
declared search space and fit-score function instead of the pooling-based
`MIL_SEARCH_SPACE`/`_mil_fit_score_fn` — every other candidate's code path
is untouched.

`run_smoke_cv` previously had no subject-level MIL branch at all (only a
cell-level `"neural"` special case and a subject-summary baseline path);
a third branch was added that builds one-cell-minimum MIL bags via
`fold_preprocessing.bags_from_fold_cell_dataset(..., {}, min_cells_per_subject=1)`
(no cancer outcomes required — every bag's `cancer_label_known=False` for
this task) and scores subject-level Macro-F1 restricted to subjects with a
known majority smoke label.

`check_mil_eligibility` (>=10 known-outcome subjects, >=2 per class) is
applied explicitly wherever this adapter's cancer-task fit happens — the
pooling-based adapters get this check for free inside `Trainer.phase2`
(both train AND val subject sets); since `PathwayHierarchicalAdapter` has
no `Trainer`, the same two-sided check is called explicitly at every
matching call site (`_pathway_cancer_fit_score_fn`, the `run_cancer_cv` mil
loop, `_fit_mil`, and the ablation runner), so a too-small/degenerate fold
is rejected identically regardless of which MIL-kind candidate is running.
The one final development-pool fit (`fit_final_candidate_on_dev_pool`) has
no validation split to check for ANY MIL-kind candidate, matching
`Trainer.phase2_final_fit`'s own no-internal-validation contract exactly.

The declared search space (`mil_registry.PATHWAY_SEARCH_SPACE`) covers
`embedding_dim` ([64, 128]), `attention_dim` ([32, 64]), `dropout`
([0.1, 0.3]), `use_gene_residual` ([True, False]), `smoke_loss_weight`
([0.5, 1.0]), and `cancer_loss_weight` ([0.5, 1.0]) — selected via the
same `select_nested_hyperparameters_with_refit` every other candidate
uses (fold-local inner-CV selection, a fresh `PreprocessingArtifact` refit
per inner fold, no reuse of one fold's selection inside another fold's
OOF prediction). A reduced single-candidate grid
(`pathway_search_space(fast=True)`) exists for synthetic/CI use but is not
yet wired into a `--fast`-conditional call site — the full grid runs even
under `--synthetic --fast` today, which is correct but slower than
necessary; see the honest-limitations list.

OOF prediction records (`generate_subject_oof_predictions`) and per-fold
CV records now carry `module_fingerprint` alongside the
`preprocessing_fingerprint`/`model_state_fingerprint` every candidate
already recorded; `runner.py`'s OOF CSV gained a `module_fingerprint`
column (empty for every non-pathway candidate).

`benchmarks/pathway_hierarchical_adapter.py::validate_pathway_bundle_identity`
cross-checks an already-loaded bundle manifest's `model_type` and
`module_fingerprint` fields against a constructed model instance — the
same shape of check `bundle.py::validate_bundle_for_model` already performs
generically for `input_dim`/`num_smoke` — so a bundle built for a different
architecture, or against a different gene-module set, is rejected before
its checkpoint is trusted.

`benchmarks/pathway_hierarchical_ablation.py` (CLI flag
`--pathway-hierarchical-ablation`, following the existing
`--imbalance-ablation` convention) compares six variants over the SAME
cancer-task grouped-subject CV folds `run_cancer_cv` would use:
`existing_attention_mil` (the pre-existing gated-attention MIL baseline,
via `NeuralCancerAdapter`), `full_multitask`, `no_gene_residual`,
`no_cell_type_embedding`, `single_task_cancer`
(`smoke_loss_weight=0.0`), and `single_task_smoke`
(`cancer_loss_weight=0.0`). Because this architecture produces both a
cancer prediction and a smoke prediction from one fit, every pathway
variant's fold record carries BOTH task's metrics (cancer AUROC/AUPRC as
the primary comparison metric, smoke Macro-F1 as a secondary diagnostic
restricted to subjects with a known majority smoke label) from the same
fitted model — `existing_attention_mil` has no subject-level smoke
prediction surface, so its smoke metric is recorded as not evaluated,
never fabricated. This ablation predates the domain-adversarial training
head (§16.4), which is now implemented as a separate, source-held-out-
specific ablation — see `benchmarks/domain_robustness_ablation.py`. Every
result is explicitly marked `development_only`/`software_only`; the frozen
test split is never touched.

## 16. Source-Held-Out Domain Robustness and Biological Stability ("Phase 6")

Phase 6 adds a source-held-out evaluation protocol, optional development-
only domain-robust training strategies for `pathway_hierarchical_mil`,
label-free domain-shift and source-predictability diagnostics, and
synthetic-module-scoped biological-stability diagnostics. It changes
nothing about `MultiSmokeCancerNet`'s default behavior, the frozen-test
protocol (§14.7), or any existing candidate's ordinary CV/OOF path unless a
new Phase 6 flag is explicitly passed.

### 16.1 Source-held-out data flow

`benchmarks/source_held_out.py` implements, for each dataset source present
in the train+val pool:

```
All sources in train+val pool
    -> per source: assess eligibility (source_eligibility.py)
    -> eligible?
         no  -> record status + reason, continue to the next source
         yes -> development_subjects = pool - held_out_source's subjects
                -> generate_subject_oof_predictions(..., dev_subjects=development_subjects, ...)
                   (final_evaluation.py, UNMODIFIED — restricting its
                   dev_subjects argument to development-only subjects is
                   what makes this a source-safe selection: the function's
                   own per-OOF-fold nested hyperparameter search never sees
                   the held-out source at all)
                -> rank candidates by development-only OOF AUROC/macro-F1
                -> one nested hyperparameter selection over the FULL
                   development pool (hyperparameter_search.py, unmodified)
                -> fit_final_candidate_on_dev_pool(..., dev_subjects=development_subjects, ...)
                   (final_evaluation.py, UNMODIFIED)
                -> build_frozen_policy from development OOF predictions
                   (calibration.py, unmodified)
                -> evaluate_frozen_test(fitted, context, held_out_subjects, ...)
                   (final_evaluation.py, UNMODIFIED — despite the name, this
                   function only transforms+predicts against whatever
                   subject list it is given; source_held_out.py is the only
                   caller that gives it a train/val-pool subset instead of
                   the real frozen test subjects)
                -> policy.apply_to_test(...) exactly once for this source
                -> record RobustnessReport (robustness_report.py)
```

This reuses `final_evaluation.py`'s functions completely unmodified in
their leakage-relevant behavior (only two new, backward-compatible
`domain_robustness_config=None` keyword parameters were added, threaded
through to `build_mil_adapter` — see §16.4) precisely because every one of
those functions already takes explicit subject-ID-list arguments rather
than reading `context.train_bags`/`val_bags`/`test_bags` directly. That
structural property — not a new mechanism — is what makes "development
sources' subjects" and "held-out source's subjects" safe to substitute for
"train+val" and "test" here.

### 16.2 Structural separation from the frozen-test guard

`source_held_out.py` never imports `test_guard.py`, never constructs a
`FrozenTestGuard`, and its public functions take no `run_dir`/
`output_root`/guard-path argument at all — there is no code path by which
calling `run_smoke_source_held_out`/`run_cancer_source_held_out` could
create, acquire, or check a guard file. `evaluate_frozen_test` itself
(§16.1) carries no guard of its own by design (see its docstring in
`final_evaluation.py`); `runner.py` is the only call site that pairs it with
guard acquisition, and it only does so for the real frozen-test stage, not
for the source-held-out stage. `tests/test_source_held_out.py` verifies
both the import-graph property (via AST inspection, not just a docstring
claim) and that no guard directory appears after running either protocol.

### 16.3 Source eligibility and the split manifest

`benchmarks/source_eligibility.py` assigns one of nine statuses per
(source, task) pair (`eligible`, `not_evaluable`, `diagnostic_only`,
`excluded_by_policy`, `controlled_access_unavailable`,
`insufficient_classes`, `insufficient_outcomes`, `species_mismatch`,
`assay_mismatch`, `gene_contract_mismatch`) — unknown species/label-
semantics metadata always resolves to `species_mismatch`/
`excluded_by_policy`, never to assumed-compatible, mirroring §12's
pre-existing `run_leave_one_source_out` policy. `build_source_held_out_manifest`
produces a versioned (`schema_version`), fingerprinted manifest
(`SHA-256` over development/held-out subject-ID lists, never the raw IDs
themselves) and raises outright if development and held-out subject sets
are found to overlap — this is a hard precondition, not a soft warning.

### 16.4 Domain-loss placement and the sampler hierarchy

`benchmarks/domain_losses.py`'s CORAL/MMD/domain-adversarial terms attach
to exactly one point in the architecture:
`HierarchicalMILOutput.subject_embeddings` (the pooled per-subject
representation `pathway_hierarchical_mil.py`'s `HierarchicalAttentionPooling`
already produced before Phase 6). `PathwayHierarchicalAdapter._train_loop`
(the adapter's single AdamW loop — see §15.8) computes the selected
strategy's regularizer term after the forward pass and adds it to the
existing masked multitask loss:
`total = task_loss + domain_term` (§16.4's `_domain_regularizer`), logged
separately from `task_loss` via `last_loss_components`. The
`domain_adversarial` strategy additionally builds a `DomainClassifierHead`
lazily, once, from whichever development sources appear in the FIRST
training call (`_maybe_build_domain_head`) — its vocabulary is fixed from
that point on, and a source name outside it at any later call raises
`DomainVocabularyError` rather than being silently mapped to an arbitrary
class. `data/source_sampling.py`'s `SourceBalancedBatchSampler` (the
`source_balanced` strategy) is a structurally independent sampler from
`data/sampling.py`'s pre-existing smoke-class-balanced sampler — the two
solve different imbalance problems and are not composed in this phase.

### 16.5 Fingerprint hierarchy

`RobustnessReport` (`robustness_report.py`) binds together, per held-out-
source evaluation: `dataset_manifest_fingerprint`,
`source_split_manifest_fingerprint` (§16.3's manifest), the development
preprocessing artifact's `preprocessing_fingerprint`
(`fold_preprocessing.artifact_fingerprint` — the same function every other
fold/OOF/final-fit record in this repository uses), `module_fingerprint`
(for `pathway_hierarchical_mil`), `model_fingerprint`
(`model_state_fingerprint()`, the same helper every adapter/baseline
already exposes), and `calibration_fingerprint` — the same identity
hierarchy §7/§14 established, extended with one new manifest type rather
than a parallel one.

### 16.6 Calibration/uncertainty boundary

Calibration and threshold selection (`calibration.py`, unmodified) see
development out-of-fold predictions only, exactly as in the real frozen-
test protocol (§12); `uncertainty.py`'s abstention-threshold selection is
likewise restricted, by argument signature, to development
uncertainty/correctness arrays — it has no parameter through which a
held-out-source label could reach it.

### 16.7 Reporting schema and failure modes

`RobustnessReport`/`build_aggregate_report` (`robustness_report.py`) always
stamp `development_only: true` and `frozen_test_accessed: false`;
`validate_robustness_report` rejects a report missing either stamp or
carrying the wrong value, and `aggregate_source_reports` reports the
worst-source result as a first-class field (`worst_source`) rather than
folding it into a single pooled average, with ineligible sources listed
(with their reason) but excluded from the evaluated-source count.

### 16.8 Honest scope boundaries

- **No real gene-set (GMT) resource ships with this repository.** Every
  concrete `biological_stability.py` result produced in this phase's tests,
  CI, and synthetic CLI runs is computed against
  `GeneModuleCollection.synthetic()` and is a software sensitivity
  diagnostic, not biological-plausibility evidence —
  `require_real_modules` refuses to run the real-mode entry point against a
  synthetic source, so this cannot be silently misreported.
- Task A (smoke) source-held-out evaluation does not extend to the
  pooling-based MIL models or to `MultiSmokeCancerNet`'s cell-level
  `Trainer` curriculum — only classical baselines and
  `pathway_hierarchical_mil`'s plain-ERM fit. `domain_robustness_ablation.py`
  records this honestly (`not_evaluable`) rather than silently omitting
  those rows.
- With few dataset sources available (2 in the synthetic CI context; a
  handful in the real dataset manifest), cross-source aggregate statistics
  have limited statistical power, reported as such rather than suppressed.
- No cross-species (human/mouse) source-held-out mode was added in this
  phase — mouse sources remain excluded from the ordinary human-only
  protocol via the same `species_mismatch` eligibility status §11/§14
  already established for human/mouse separation elsewhere.
