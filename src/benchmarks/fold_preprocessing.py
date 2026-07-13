"""
benchmarks/fold_preprocessing.py — per-fold PreprocessingArtifact refit.

CV must never reuse the ExperimentContext's OUTER PreprocessingArtifact
(fit_preprocessing() called once, on ALL original-train subjects) across
folds: when an original-train subject becomes an INNER CV validation
subject, it has already influenced the gene scaling means/stds and HVG
selection that artifact encodes, which then get applied to that same
subject's "held-out" fold evaluation. That's preprocessing leakage,
identical in kind to the whole-dataset-scaling bug run_pipeline_split_aware
itself was built to fix (see preprocessing.py's module docstring) — it's
just one level deeper, at the CV-fold level instead of the outer split level.

The fix: refit a fresh PreprocessingArtifact per fold, using ONLY that
fold's training subjects, from context.normalized_adata_for_refit (the
full-gene, normalized-but-not-yet-HVG-selected-or-scaled AnnData captured in
run_pipeline_split_aware() before its own fit_preprocessing call). Requires
context.normalized_adata_for_refit to be present — see
require_normalized_adata's error if it's not, per the "fail clearly, never
silently reuse the outer artifact" rule.
"""

import hashlib
import json
from typing import Optional, Sequence

import numpy as np

from data.preprocessing import PreprocessingArtifact, apply_preprocessing, fit_preprocessing
from train import CellLevelDataset


def require_normalized_adata(context) -> "object":
    if context.normalized_adata_for_refit is None:
        raise ValueError(
            "This ExperimentContext has no normalized_adata_for_refit (the pre-HVG, "
            "pre-scaling normalized expression run_pipeline_split_aware() captures) — "
            "grouped CV cannot refit preprocessing per fold without it, and silently "
            "falling back to the outer PreprocessingArtifact would reintroduce the exact "
            "fold-level leakage this module exists to prevent. Rebuild the context from "
            "a real run_pipeline_split_aware() result, or pass normalized_adata_for_refit "
            "explicitly for synthetic/test contexts."
        )
    return context.normalized_adata_for_refit


def artifact_fingerprint(artifact: PreprocessingArtifact) -> str:
    payload = {
        "gene_list": artifact.gene_list, "gene_means": artifact.gene_means,
        "gene_stds": artifact.gene_stds, "fit_n_cells": artifact.fit_n_cells,
        "fit_n_subjects": artifact.fit_n_subjects,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def refit_artifact_for_fold(
    normalized_adata, fold_train_subjects: Sequence[str], n_hvgs: int,
) -> PreprocessingArtifact:
    return fit_preprocessing(normalized_adata, set(str(s) for s in fold_train_subjects), n_hvgs=n_hvgs)


def build_fold_cell_dataset(
    normalized_adata, artifact: PreprocessingArtifact, subject_list: Sequence[str],
) -> CellLevelDataset:
    """
    Apply a fold-specific artifact to exactly the cells belonging to
    subject_list, and wrap the result as a CellLevelDataset — same
    downstream shape (`.X`, `.smoke`, `.subject_ids`, ...) the rest of
    benchmarks/ already expects, so build_smoke_subject_summary_features,
    baselines, and the neural adapter all work unchanged on fold data.

    cell_type_id in normalized_adata_for_refit is always the pre-annotation
    placeholder (0 for every cell) — annotate_cell_types() (celltypist) runs
    AFTER the snapshot point in run_pipeline_split_aware(). This is a
    documented limitation (loses cell-type-proportion feature richness in
    CV), not a leakage risk: celltypist is deterministic pretrained-model
    inference, not fit to this dataset, so re-running it per fold would add
    no leakage protection, only runtime.
    """
    subject_ids_col = normalized_adata.obs["subject_id"].astype(str)
    wanted = {str(s) for s in subject_list}
    mask = subject_ids_col.isin(wanted).values
    if mask.sum() == 0:
        raise ValueError(f"build_fold_cell_dataset: none of {len(wanted)} requested subjects found")

    subset = apply_preprocessing(normalized_adata[mask], artifact)
    obs = subset.obs
    n = subset.n_obs
    X = np.asarray(subset.X, dtype=np.float32)

    dose = obs["exposure_dose"].values.astype(np.float32) if "exposure_dose" in obs.columns else None
    malig_known = obs["malignancy_known"].values.astype(bool) if "malignancy_known" in obs.columns else None
    source = obs["source"].astype(str).values if "source" in obs.columns else None

    return CellLevelDataset(
        gene_matrix=X,
        smoke_labels=obs["smoke_type"].values.astype(np.int64),
        malignancy_labels=obs["malignancy"].values.astype(np.float32) if "malignancy" in obs.columns
                           else np.zeros(n, dtype=np.float32),
        cell_type_ids=obs["cell_type_id"].values.astype(np.int64) if "cell_type_id" in obs.columns
                      else np.zeros(n, dtype=np.int64),
        exposure_dose=dose,
        malignancy_known=malig_known,
        subject_ids=obs["subject_id"].astype(str).values,
        dataset_source=source,
    )


def fold_train_val_datasets(
    context, fold_train_subjects: Sequence[str], fold_val_subjects: Sequence[str], n_hvgs: Optional[int] = None,
):
    """One-call convenience: refit + apply for both sides of a fold. Returns
    (artifact, train_cell_dataset, val_cell_dataset)."""
    normalized_adata = require_normalized_adata(context)
    n_hvgs = n_hvgs if n_hvgs is not None else context.preprocessing_artifact.n_hvgs
    artifact = refit_artifact_for_fold(normalized_adata, fold_train_subjects, n_hvgs)
    train_ds = build_fold_cell_dataset(normalized_adata, artifact, fold_train_subjects)
    val_ds = build_fold_cell_dataset(normalized_adata, artifact, fold_val_subjects)
    return artifact, train_ds, val_ds


def bags_from_fold_cell_dataset(
    fold_cell_dataset: CellLevelDataset, outcomes_by_subject: dict, min_cells_per_subject: int = 50,
) -> list:
    """
    Rebuild MIL bags directly from a fold-refit CellLevelDataset (not
    assembly.assemble_subject_bags's AnnData signature, since the fold
    dataset is already a CellLevelDataset) — used by the cancer-task CV so
    Task B bags are built from the SAME fold-specific, leakage-free gene
    space as the fold's cell-level data, not the outer artifact's bags.
    outcomes_by_subject: {subject_id: 0/1} for subjects with a KNOWN cancer
    outcome only — a subject absent from this dict gets cancer_label_known=False,
    never a fabricated negative.
    """
    bags = []
    subj = fold_cell_dataset.subject_ids
    for sid in sorted(set(subj.tolist())):
        mask = subj == sid
        if mask.sum() < min_cells_per_subject:
            continue
        outcome = outcomes_by_subject.get(str(sid))
        bags.append({
            "subject_id": sid,
            "gene_matrix": fold_cell_dataset.X[mask].numpy(),
            "cell_type_ids": fold_cell_dataset.ctype[mask].numpy(),
            "smoke_labels": fold_cell_dataset.smoke[mask].numpy(),
            "malig_labels": fold_cell_dataset.malig[mask].numpy(),
            "malig_known": fold_cell_dataset.malig_known[mask].numpy(),
            "cancer_label": outcome,
            "cancer_label_known": outcome is not None,
        })
    return bags
