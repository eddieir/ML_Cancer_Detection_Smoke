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
| 3 | cannabis | Loiselle 2018 (BioProject) | Bulk RNA-seq on BEAS-2B cell lines, not true scRNA-seq; used as pseudo-bulk |
| 4 | dual-use | NLST dual-reported subjects (cigarette + vape), cells re-labelled from GSE136831 | Approximated from clinical annotation |
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
```

**Lambda values:**

| Phase | lambda_smoke | lambda_malignancy | lambda_subject |
|-------|-------------|------------------|---------------|
| Phase 1 (cell pre-train) | 0.50 | 0.50 | 0.00 |
| Phase 2 (aggregator train) | 0.00 | 0.00 | 1.00 |
| Phase 3 (end-to-end) | 0.30 | 0.30 | 0.40 |

Subject prediction carries 0.40 weight in Phase 3 because it's the primary clinical objective.

---

## 5. Three-Phase Training Curriculum

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
| Loiselle 2018 (BioProject) | BEAS-2B + NCI-H1975 | Cell lines (bulk RNA-seq, not scRNA-seq) | Cigarette + Cannabis | Free — pseudo-bulk; one vector per condition not per cell |
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
- Smoke type: macro-averaged F1 across 6 classes, per-class accuracy
- Malignancy: ROC-AUC, precision-recall AUC

### Subject-level (Phase 2 + 3 validation, primary)
- ROC-AUC on cancer vs. no-cancer (main metric)
- Sensitivity / specificity at threshold 0.70 (HIGH RISK cutoff)
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
| End-to-end smoke->malignancy->cancer pipeline | Not in any paper, preprint, or conference | Confirmed gap across all source types |

---

## 10. Build Order (Next Steps)

```
Step 1 (this file):  Architecture specification         [DONE]
Step 2:              src/preprocess.py                  [ ]
Step 3:              src/model.py                       [ ]
Step 4:              src/train.py                       [ ]
Step 5:              src/evaluate.py                    [ ]
Step 6:              src/inference.py                   [ ]
Step 7:              notebooks/01_data_download.ipynb   [ ]
Step 8:              notebooks/02_preprocessing.ipynb   [ ]
Step 9:              notebooks/03_training.ipynb        [ ]
Step 10:             notebooks/04_evaluation.ipynb      [ ]
```
