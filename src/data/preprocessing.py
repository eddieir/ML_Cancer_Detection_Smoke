"""
data/preprocessing.py — leakage-free fit/transform preprocessing artifact.

merge_sources(scale=True) (the legacy default) z-scores the FULL merged
dataset and smoke_aware_hvg() picks highly-variable genes across the FULL
merged dataset — both BEFORE any train/val/test split exists. That means
validation/test cells influence the scaling statistics and which genes
become model features: classic preprocessing leakage, forbidden by this
project's scientific rules (see README.md).

This module separates FIT (train cells only) from TRANSFORM (applied
identically, unchanged, to val/test/inference), and packages the fitted
state into a versioned, JSON-serialisable artifact that can be saved
alongside a checkpoint and reloaded for inference-time compatibility
checks (required genes, gene order, input dimension, artifact version).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import anndata as ad
import numpy as np
import scanpy as sc

ARTIFACT_VERSION = "1"

# The only input stages apply_preprocessing()/predict_h5ad() know how to
# handle — see inference.py. "normalized_expression" is what this artifact's
# apply_preprocessing() actually expects: input already QC'd, library-size
# normalized, and log-transformed (data/transforms.py::normalize) upstream
# of this artifact; the artifact itself only stores gene subsetting/ordering
# and train-fit mean/std scaling, NOT QC thresholds, a library-size target,
# or log-transform parameters — so it cannot reproduce a full raw-count
# pipeline, only pick up from already-normalized expression.
EXPECTED_INPUT_STAGE = "normalized_expression"


@dataclass
class PreprocessingArtifact:
    version:                   str
    gene_list:                 List[str]    # ordered HVG gene names, fit on train cells only
    gene_means:                List[float]  # per-gene mean, fit on train cells only
    gene_stds:                 List[float]  # per-gene std,  fit on train cells only (0 -> 1.0)
    n_hvgs:                    int
    smoke_marker_genes_forced: List[str]
    fit_n_cells:                int
    fit_n_subjects:              int
    notes:                     List[str] = field(default_factory=list)
    # Smoke-label effective mapping (data/label_mapping.py), set after fitting
    # once the rare-class policy has run — None only for artifacts fit before
    # this field existed (legacy) or that never had label-mapping context.
    label_mapping:              Optional[Dict] = None
    # What input stage apply_preprocessing() expects to receive — see
    # EXPECTED_INPUT_STAGE above. Stored explicitly (not just documented) so
    # inference can validate a caller's claimed input_stage against what
    # this artifact can actually consume.
    expected_input_stage:       str = EXPECTED_INPUT_STAGE
    # SHA-256 of the fixed cell-type label-name -> ID table (constants.py's
    # CELL_TYPE_MAP) in effect when obs["cell_type_id"] was assigned — see
    # data/transforms.py::cell_type_map_fingerprint(). None for artifacts fit
    # before this field existed, or when cell-type annotation never ran
    # (e.g. pseudo-bulk-only sources). A mismatch on reload means the code's
    # mapping table changed since this artifact was fit, not that different
    # cells were annotated.
    cell_type_map_fingerprint:  Optional[str] = None
    # "inductive_per_cell" (CellTypist majority_voting=False — a pure
    # function of each cell's own expression, independent of which other
    # cells were annotated alongside it) or "majority_voting" (legacy,
    # dataset-dependent) — see annotate_cell_types(). None if annotation
    # never ran.
    cell_type_annotation_mode:  Optional[str] = None
    # True only when annotate_cell_types() fell back to a fixed placeholder
    # label for every cell after CellTypist itself failed (see
    # allow_diagnostic_fallback in data/transforms.py) — never a real,
    # scientifically meaningful per-cell annotation. None/False for a normal
    # artifact. ExperimentContext.from_pipeline_result rejects an artifact
    # with this set to True (see benchmarks/context.py) — a real run must
    # never silently proceed on degraded cell-type labels.
    cell_type_annotation_degraded: Optional[bool] = None
    # CellTypist/scikit-learn version-compatibility provenance for this
    # artifact's cell-type annotation — see data/transforms.py::
    # _load_celltypist_model_checking_sklearn_compatibility. Keys:
    # celltypist_version, model_name, runtime_sklearn_version,
    # serialized_sklearn_versions (list), compatible (bool). None when
    # annotation never ran (pseudo-bulk) or failed to a diagnostic fallback
    # before a model could be loaded, or for artifacts fit before this
    # field existed.
    cell_type_annotation_compatibility: Optional[Dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[preprocessing] artifact saved → {path}  ({len(self.gene_list)} genes)")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PreprocessingArtifact":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


def fit_preprocessing(
    adata:              ad.AnnData,
    train_subject_ids:  set,
    n_hvgs:             int = 2000,
    batch_key:          Optional[str] = "batch",
    subject_col:        str = "subject_id",
) -> PreprocessingArtifact:
    """
    Fit gene mean/std scaling + smoke-aware HVG selection using ONLY cells
    whose subject_id is in train_subject_ids. Mirrors merge_sources()'s
    z-scoring and transforms.smoke_aware_hvg()'s variance-based selection
    with forced smoke-marker inclusion, but restricted to the train
    partition so val/test statistics never influence what genes are kept
    or how they're scaled.
    """
    from constants import ALL_SMOKE_MARKERS

    train_mask = adata.obs[subject_col].astype(str).isin(
        {str(s) for s in train_subject_ids}
    ).values
    if train_mask.sum() == 0:
        raise ValueError("fit_preprocessing: no cells matched train_subject_ids")

    train_adata = adata[train_mask].copy()

    X = train_adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    stds[stds == 0] = 1.0  # constant genes: avoid divide-by-zero, contribute nothing either way

    bk = batch_key if (batch_key and batch_key in train_adata.obs.columns) else None
    sc.pp.highly_variable_genes(
        train_adata, n_top_genes=min(n_hvgs, train_adata.n_vars), batch_key=bk
    )
    hvg_mask = train_adata.var["highly_variable"].values.copy()

    present_markers = [g for g in ALL_SMOKE_MARKERS if g in train_adata.var_names]
    forced = [
        g for g in present_markers
        if not hvg_mask[train_adata.var_names.get_loc(g)]
    ]
    if forced:
        non_marker_idx = [
            i for i, hv in enumerate(hvg_mask)
            if hv and train_adata.var_names[i] not in present_markers
        ]
        for i in non_marker_idx[-len(forced):]:
            hvg_mask[i] = False
        for g in forced:
            hvg_mask[train_adata.var_names.get_loc(g)] = True

    gene_list = list(train_adata.var_names[hvg_mask])
    idx = [train_adata.var_names.get_loc(g) for g in gene_list]

    print(f"[preprocessing] fit  {len(gene_list)} genes  "
          f"(forced {len(forced)} smoke markers)  on {train_mask.sum():,} train cells")

    return PreprocessingArtifact(
        version=ARTIFACT_VERSION,
        gene_list=gene_list,
        gene_means=[float(means[i]) for i in idx],
        gene_stds=[float(stds[i]) for i in idx],
        n_hvgs=n_hvgs,
        smoke_marker_genes_forced=forced,
        fit_n_cells=int(train_mask.sum()),
        fit_n_subjects=len(set(train_adata.obs[subject_col].astype(str))),
        # Cell-type annotation provenance is a property of `adata.obs`
        # (annotate_cell_types() ran once, upstream, on the full merged
        # dataset before any fold/OOD/outer-split refit ever calls this
        # function) — never re-derived or defaulted here. Every caller of
        # fit_preprocessing (the outer split in preprocess.py, and every
        # per-fold/per-OOD-reconstruction refit in
        # benchmarks/fold_preprocessing.py) passes the SAME `adata` these
        # fields came from, so propagating them here means a fold-refit
        # artifact carries the identical, correct provenance the outer
        # artifact does — closing the gap where a fold refit silently
        # produced an artifact with unset (None) provenance fields.
        cell_type_map_fingerprint=adata.uns.get("cell_type_map_fingerprint"),
        cell_type_annotation_mode=adata.uns.get("cell_type_annotation_mode"),
        cell_type_annotation_degraded=adata.uns.get("cell_type_annotation_degraded"),
        cell_type_annotation_compatibility=adata.uns.get("cell_type_annotation_compatibility"),
        notes=[
            "Batch correction (Harmony) is NOT part of this artifact: Harmony has "
            "no native train-only-fit / apply-to-new-data transform, so it cannot "
            "be included in a leakage-free fit/transform artifact the way scaling "
            "and HVG selection can. batch_correct() must be re-fit independently "
            "per split if used — a documented limitation, not full leakage-free "
            "batch correction. See README.md's preprocessing-leakage section.",
        ],
    )


def apply_preprocessing(adata: ad.AnnData, artifact: PreprocessingArtifact) -> ad.AnnData:
    """
    Transform-only: subset/reorder genes to artifact.gene_list (in that
    exact order) and apply the train-fit mean/std scaling, clipped to
    [-10, 10] to match sc.pp.scale(max_value=10)'s legacy behaviour.
    Raises if the input is missing genes the artifact requires.
    """
    verify_compatible(artifact, adata.var_names)
    out = adata[:, artifact.gene_list].copy()
    X = out.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    means = np.array(artifact.gene_means)
    stds  = np.array(artifact.gene_stds)
    X = np.clip((X - means) / stds, -10, 10)
    out.X = X.astype(np.float32)
    return out


def verify_compatible(artifact: PreprocessingArtifact, gene_names) -> None:
    """
    Raise a clear error if `gene_names` cannot be safely transformed with
    this artifact — missing required genes, or (for direct array input,
    where reordering isn't possible after the fact) wrong gene order.
    Called by apply_preprocessing() and should also be called by inference
    code paths that receive an already-HVG-selected array instead of an
    AnnData (see inference.py).
    """
    gene_names = list(gene_names)
    missing = [g for g in artifact.gene_list if g not in gene_names]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} gene(s) required by "
            f"PreprocessingArtifact version {artifact.version}: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "") +
            ". Re-run the same conversion/harmonization pipeline used to fit "
            "this artifact, or refit a new artifact for this gene panel."
        )


class CellTypeProvenanceError(ValueError):
    """Raised when a PreprocessingArtifact's cell-type annotation provenance
    is missing, malformed, or records a mode a real (non-synthetic)
    ExperimentContext must not silently accept. Lives here rather than in
    benchmarks/context.py to avoid a circular import (data/transforms.py,
    which defines cell_type_map_fingerprint(), already sits below
    benchmarks/ in the dependency graph, and this module already owns
    PreprocessingArtifact itself)."""


# The only cell-type annotation modes a real (non-synthetic) pipeline result
# may carry. "inductive_per_cell" (data/transforms.py::annotate_cell_types,
# majority_voting=False) requires a matching cell_type_map_fingerprint.
# "pseudo_bulk_no_cell_type_identity" is the explicit, documented policy for
# pseudo-bulk sources: CellTypist annotation never runs on pseudo-bulk data
# (there is no per-cell expression to classify), so no per-cell mapping
# fingerprint applies — this is a deliberate absence of cell-type identity,
# not a degraded/failed annotation, and callers requiring real per-cell
# identity (e.g. a cell-type-aware MIL pooling path) must reject this mode
# themselves rather than have it silently pass as "the same as annotated".
# "majority_voting" (dataset-dependent, non-inductive) and
# "diagnostic_fallback" (every cell given the same placeholder label after
# CellTypist itself failed) are both real annotation modes that exist in
# this codebase but are NEVER accepted for a real ExperimentContext.
_VALID_REAL_CELL_TYPE_ANNOTATION_MODES = frozenset({
    "inductive_per_cell", "pseudo_bulk_no_cell_type_identity",
})


def validate_cell_type_provenance(artifact: "PreprocessingArtifact") -> None:
    """
    Strict, fail-closed validation of a PreprocessingArtifact's cell-type
    annotation provenance for a REAL (non-synthetic) ExperimentContext.
    Unlike the historical `getattr(artifact, "cell_type_annotation_degraded",
    False)` check this replaces, a MISSING provenance field is never treated
    as "not degraded" / safe — every field below must be explicitly present
    and valid, or this raises CellTypeProvenanceError.
    """
    degraded = artifact.cell_type_annotation_degraded
    mode = artifact.cell_type_annotation_mode
    fp = artifact.cell_type_map_fingerprint

    if degraded is not False:
        raise CellTypeProvenanceError(
            f"PreprocessingArtifact.cell_type_annotation_degraded must be exactly False "
            f"for a real pipeline result, got {degraded!r} (missing/None/True are all "
            "refused — a missing field is never treated as 'not degraded')."
        )
    if mode not in _VALID_REAL_CELL_TYPE_ANNOTATION_MODES:
        raise CellTypeProvenanceError(
            f"PreprocessingArtifact.cell_type_annotation_mode={mode!r} is not one of the "
            f"accepted real-run modes {sorted(_VALID_REAL_CELL_TYPE_ANNOTATION_MODES)} — "
            "missing/None/'majority_voting'/'diagnostic_fallback'/any other mode is refused."
        )
    if mode == "inductive_per_cell":
        if not fp or not isinstance(fp, str) or len(fp) != 64 or any(
            c not in "0123456789abcdef" for c in fp
        ):
            raise CellTypeProvenanceError(
                f"PreprocessingArtifact.cell_type_map_fingerprint is missing or malformed "
                f"({fp!r}) for cell_type_annotation_mode='inductive_per_cell' — a real "
                "per-cell annotation must record a valid 64-character hex SHA-256 "
                "fingerprint of the mapping table used."
            )
        from data.transforms import cell_type_map_fingerprint as _current_cell_type_map_fingerprint
        expected = _current_cell_type_map_fingerprint()
        if fp != expected:
            raise CellTypeProvenanceError(
                f"PreprocessingArtifact.cell_type_map_fingerprint {fp!r} does not match the "
                f"current canonical CELL_TYPE_MAP fingerprint {expected!r} — this artifact's "
                "cell-type annotation was produced under a different mapping table than is "
                "currently in effect. Re-run preprocessing rather than silently trusting a "
                "stale mapping; if this is expected (e.g. an intentional constants.py "
                "change), reprocess and refit a new artifact rather than reusing the old one."
            )
    elif mode == "pseudo_bulk_no_cell_type_identity" and fp is not None:
        raise CellTypeProvenanceError(
            f"PreprocessingArtifact.cell_type_map_fingerprint must be None for "
            f"cell_type_annotation_mode='pseudo_bulk_no_cell_type_identity' (no per-cell "
            f"mapping was ever applied), got {fp!r}."
        )


def verify_input_matrix(artifact: PreprocessingArtifact, gene_order: List[str]) -> None:
    """
    Strict check for raw-array inference inputs (no AnnData to reorder):
    gene_order must match artifact.gene_list exactly, in the same order,
    since a raw float32 matrix has no column labels of its own once it
    reaches the model.
    """
    if list(gene_order) != artifact.gene_list:
        raise ValueError(
            f"Input gene order does not match PreprocessingArtifact version "
            f"{artifact.version} (expected {len(artifact.gene_list)} genes in a "
            "fixed order). Feeding a differently-ordered or differently-sized "
            "gene panel to the model silently produces meaningless predictions "
            "— use apply_preprocessing() on an AnnData instead of hand-building "
            "the input array."
        )
