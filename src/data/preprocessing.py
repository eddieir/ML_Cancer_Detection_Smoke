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
