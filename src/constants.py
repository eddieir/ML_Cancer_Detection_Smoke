"""
constants.py
Shared across ALL modules. Single source of truth — never duplicate these.
"""

SMOKE_TYPE_MAP: dict[str, int] = {
    "cigarette": 0,
    "vape": 1, "ecig": 1, "e-cig": 1,
    "cigar": 2,
    "cannabis": 3, "weed": 3,
    "dual_use": 4, "dual": 4,
    "unexposed": 5, "never": 5, "control": 5,
}

SMOKE_TYPES: dict[int, str] = {
    0: "cigarette", 1: "vape_ecig", 2: "cigar",
    3: "cannabis", 4: "dual_use", 5: "unexposed",
}

CELL_TYPE_MAP: dict[str, int] = {
    # coarse
    "epithelial": 0, "endothelial": 1, "immune": 2, "stromal": 3,
    # CellTypist granular → coarse
    "Basal cell": 0, "Club cell": 0, "Ciliated cell": 0,
    "AT1 cell": 0, "AT2 cell": 0, "Bronchial epithelial cell": 0,
    "Endothelial cell": 1, "Capillary EC": 1, "Arterial EC": 1, "Venous EC": 1,
    "Macrophage": 2, "T cell": 2, "NK cell": 2, "B cell": 2,
    "Monocyte": 2, "Dendritic cell": 2, "Neutrophil": 2,
    "Fibroblast": 3, "Smooth muscle cell": 3, "Pericyte": 3,
}

CELL_TYPES: dict[int, str] = {
    0: "epithelial", 1: "endothelial", 2: "immune", 3: "stromal",
}

# Ma et al. 2024 — smoking-discriminative markers per cell type.
# Force-included in HVG selection regardless of variance rank.
SMOKE_MARKER_GENES: dict[str, list[str]] = {
    "endothelial": ["B2M", "EEF1A1", "TPT1"],
    "epithelial":  ["FTL", "MT-ATP8", "SCGB1A1", "MUC5AC"],
    "immune":      ["HLA-B", "HLA-C", "S100A8", "S100A9"],
    "stromal":     ["HSP90B1", "LCN2", "COL1A1"],
}

ALL_SMOKE_MARKERS: list[str] = [
    g for genes in SMOKE_MARKER_GENES.values() for g in genes
]

N_SMOKE_CLASSES = 6
N_CELL_TYPES    = 4
N_HVGS_DEFAULT  = 2000

# Dose-response modeling (exposure duration -> malignancy trajectory).
# No wired source currently supplies a real per-cell exposure duration, so
# every cell is stamped DOSE_UNKNOWN today — see model.py::DoseResponseHead.
DOSE_UNKNOWN      = -1.0

# ─── Species / cross-domain policy ─────────────────────────────────────────
# Every loader stamps obs["species"] with one of these (data/loaders.py).
# merge_sources() (data/assembly.py) refuses to silently concatenate cells
# whose species differs unless the caller explicitly acknowledges it via
# allow_mixed_species=True — see that function's docstring.
SPECIES_HUMAN = "human"
SPECIES_MOUSE = "mouse"
VALID_SPECIES = frozenset({SPECIES_HUMAN, SPECIES_MOUSE})

# Explicit experiment modes governing whether/how mouse (cross-species) data
# may enter a run — see data/species_policy.py. Default is human_only:
# mouse sources are never loaded at all unless a caller opts into one of the
# other modes.
EXPERIMENT_MODE_HUMAN_ONLY                  = "human_only"
EXPERIMENT_MODE_MOUSE_ONLY                  = "mouse_only"
EXPERIMENT_MODE_CROSS_SPECIES_PRETRAINING   = "cross_species_pretraining"
EXPERIMENT_MODE_CROSS_SPECIES_DOMAIN_ADAPT  = "cross_species_domain_adaptation"
VALID_EXPERIMENT_MODES = frozenset({
    EXPERIMENT_MODE_HUMAN_ONLY,
    EXPERIMENT_MODE_MOUSE_ONLY,
    EXPERIMENT_MODE_CROSS_SPECIES_PRETRAINING,
    EXPERIMENT_MODE_CROSS_SPECIES_DOMAIN_ADAPT,
})
DEFAULT_EXPERIMENT_MODE = EXPERIMENT_MODE_HUMAN_ONLY

# ─── Assay mode (single-cell vs. TCGA-style bulk) ──────────────────────────
# TCGA is primarily bulk expression; it must never be silently combined with
# true single-cell sources into one training loader — see data/tcga_mode.py.
ASSAY_MODE_SINGLE_CELL = "human_single_cell"
ASSAY_MODE_BULK_TCGA   = "bulk_tcga"
VALID_ASSAY_MODES = frozenset({ASSAY_MODE_SINGLE_CELL, ASSAY_MODE_BULK_TCGA})
DEFAULT_ASSAY_MODE = ASSAY_MODE_SINGLE_CELL

# ─── Assay policy (experiment-level single-cell/bulk/multimodal gate) ──────
# Row-level assay provenance (is_pseudo_bulk, assay_mode above) says what a
# given observation IS; assay_policy says what an EXPERIMENT is allowed to
# accept. See data/assay_policy.py, the single enforcement module every
# source-loading, preprocessing, dataset-construction, training, evaluation,
# bundle, and inference boundary in this project must consult before
# accepting a row. Never compare dataset/source NAMES to decide whether a
# row is safe — always consult is_pseudo_bulk on the row itself.
ASSAY_POLICY_SINGLE_CELL_ONLY = "single_cell_only"
ASSAY_POLICY_BULK_ONLY        = "bulk_only"
ASSAY_POLICY_MULTIMODAL       = "multimodal"
VALID_ASSAY_POLICIES = frozenset({
    ASSAY_POLICY_SINGLE_CELL_ONLY, ASSAY_POLICY_BULK_ONLY, ASSAY_POLICY_MULTIMODAL,
})
# single_cell_only is the only policy with a real, trainable pipeline today
# — see data/assay_policy.py::BulkTrainingNotImplementedError/
# MultimodalTrainingNotImplementedError for the other two.
DEFAULT_ASSAY_POLICY = ASSAY_POLICY_SINGLE_CELL_ONLY
# Bumped whenever the *meaning* of assay_policy enforcement changes (not on
# every code edit) — recorded on every PreprocessingArtifact/bundle/report so
# a reload can tell an artifact fit under an older enforcement contract from
# one fit under the current one. See data/assay_policy.py.
#
# "2": missing row-level is_pseudo_bulk provenance now fails closed
# (MissingAssayProvenanceError) in fit_preprocessing/apply_preprocessing/
# CellLevelDataset/CellLevelDataset.from_dir, instead of defaulting to
# all-real-cells outside an explicit diagnostic_mode; require_trainable()
# is now enforced at every production training entry point rather than
# only at the loading/manifest boundary. An artifact/bundle recorded with
# version "1" was fit under the older, more permissive contract.
ASSAY_POLICY_VERSION = "2"
