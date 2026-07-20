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

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

from data.preprocessing import PreprocessingArtifact, apply_preprocessing, fit_preprocessing
from data.transforms import assert_batch_correction_safe
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
    # Every caller of this function refits preprocessing per fold/OOF split
    # under the assumption that isolation between folds is real — a context
    # whose outer artifact was produced under transductive batch correction
    # (Harmony run across the full train+val+test dataset) never had that
    # isolation to begin with, so refitting per fold on top of it would
    # only hide, not fix, the leakage. See data/transforms.py::
    # assert_batch_correction_safe / UnsafeBatchCorrectionError.
    assert_batch_correction_safe(
        getattr(context, "transductive_batch_correction", False),
        context_name="fold_preprocessing.require_normalized_adata",
    )
    return context.normalized_adata_for_refit


def artifact_fingerprint(artifact: PreprocessingArtifact) -> str:
    """This fold module's own name for PreprocessingArtifact.
    scientific_fingerprint() (data/preprocessing.py) — kept as a thin
    wrapper (not a second, independently-computed fingerprint) so every
    caller in this codebase (per-fold records, OOF prediction records, the
    final development artifact, the frozen-test guard identity) collides on
    exactly the same identity for the same scientific content, instead of
    two subtly different hashes that could disagree for reasons that don't
    matter (or, worse, agree by coincidence for reasons that do). Includes
    gene list/order, scaling statistics, cell-type annotation provenance,
    CellTypist/scikit-learn compatibility, and gene-contract/batch-
    correction policy — see scientific_fingerprint()'s own field list."""
    return artifact.scientific_fingerprint()


def refit_artifact_for_fold(
    normalized_adata, fold_train_subjects: Sequence[str], n_hvgs: int,
) -> PreprocessingArtifact:
    return fit_preprocessing(normalized_adata, set(str(s) for s in fold_train_subjects), n_hvgs=n_hvgs)


# ─── Per-fold artifact persistence ──────────────────────────────────────────
#
# Every fold artifact returned by refit_artifact_for_fold() lives only in
# memory unless a caller explicitly persists it here. Directory layout
# mirrors the run-directory convention reporting.py already uses for the
# outer artifact (run_dir/preprocessing/final_artifact.json):
#
#   <output_root>/preprocessing/fold_00/artifact.json
#   <output_root>/preprocessing/fold_00/manifest.json
#   <output_root>/preprocessing/fold_01/artifact.json
#   <output_root>/preprocessing/fold_01/manifest.json
#   ...
#
# manifest.json is written LAST (after artifact.json is fully, atomically
# on disk) and is what marks a fold's persisted artifact "complete" — see
# load_fold_artifact()'s incomplete-fold rejection below. Both files are
# themselves written atomically (PreprocessingArtifact.save /
# benchmarks.atomic_io.atomic_write_json), so a reader never observes a
# partially written file; manifest.json's write ordering is what protects
# against observing a fold directory that has an artifact.json but was
# never actually finished (e.g. a crash between the two writes).

def fold_artifact_dir(output_root: Union[str, Path], fold_idx: Union[int, str]) -> Path:
    """fold_idx is normally a plain fold number (zero-padded to 2 digits:
    fold_00, fold_01, ...). A caller running multiple seeds x folds (see
    cross_validation.py's run_smoke_cv/run_cancer_cv) that needs a unique
    directory per (seed, fold) pair may pass any other string instead —
    used as-is, not reformatted."""
    name = f"fold_{fold_idx:02d}" if isinstance(fold_idx, int) else f"fold_{fold_idx}"
    return Path(output_root) / "preprocessing" / name


def save_fold_artifact(
    artifact: PreprocessingArtifact,
    output_root: Union[str, Path],
    fold_idx: Union[int, str],
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    extra: Optional[Dict] = None,
) -> Path:
    """
    Persist `artifact` (already fit on ONLY this fold's training subjects)
    to <output_root>/preprocessing/fold_{fold_idx:02d}/. Records the fold's
    train/val subject-list fingerprints (data/manifest.py-style: a
    deterministic hash, never the raw subject IDs, mirroring how
    final_evaluation.py fingerprints OOF fold membership) so
    load_fold_artifact() can verify a resumed fold's expected identity
    before reuse. Returns the fold directory.
    """
    import hashlib
    import json as _json

    fold_dir = fold_artifact_dir(output_root, fold_idx)
    artifact.save(fold_dir / "artifact.json")

    def _subject_fp(subjects: Sequence[str]) -> str:
        ids = sorted(str(s) for s in subjects)
        return hashlib.sha256(_json.dumps(ids, sort_keys=True).encode("utf-8")).hexdigest()

    manifest = {
        "fold_idx": fold_idx,
        "artifact_fingerprint": artifact.scientific_fingerprint(),
        "train_subjects_fingerprint": _subject_fp(train_subjects),
        "val_subjects_fingerprint": _subject_fp(val_subjects),
        "n_train_subjects": len(set(str(s) for s in train_subjects)),
        "n_val_subjects": len(set(str(s) for s in val_subjects)),
        "status": "complete",
    }
    if extra:
        manifest.update(extra)

    from .atomic_io import atomic_write_json
    atomic_write_json(fold_dir / "manifest.json", manifest)
    return fold_dir


class IncompleteFoldArtifactError(RuntimeError):
    """Raised when a fold artifact directory exists but its manifest.json
    (written last, after artifact.json — see save_fold_artifact) is
    missing or does not record status='complete'. Never silently treated
    as "no artifact" or "safe to reuse" — an interrupted fold-artifact
    write must be regenerated, not resumed from a partial state."""


class FoldArtifactMismatchError(RuntimeError):
    """Raised when a resumed fold's expected train/val-subject fingerprint
    (or an explicitly expected artifact fingerprint) does not match what a
    persisted fold artifact actually recorded — e.g. a caller accidentally
    pointed fold_01's subjects at fold_00's persisted artifact directory."""


def load_fold_artifact(
    output_root: Union[str, Path],
    fold_idx: Union[int, str],
    expected_train_subjects: Optional[Sequence[str]] = None,
    expected_val_subjects: Optional[Sequence[str]] = None,
    expected_artifact_fingerprint: Optional[str] = None,
) -> PreprocessingArtifact:
    """
    Load a persisted fold artifact, verifying compatibility BEFORE
    returning it (never after) whenever the caller supplies an expectation
    to check against:

      * expected_train_subjects / expected_val_subjects, if given, must
        fingerprint-match manifest.json's recorded values — this is what
        stops fold_01's subjects from silently being paired with
        fold_00's persisted artifact on resume.
      * expected_artifact_fingerprint, if given, must match exactly.

    Raises IncompleteFoldArtifactError if manifest.json is missing or
    doesn't record status='complete' (an interrupted/partial persist),
    and FoldArtifactMismatchError on any fingerprint mismatch.
    """
    import hashlib
    import json as _json

    fold_dir = fold_artifact_dir(output_root, fold_idx)
    manifest_path = fold_dir / "manifest.json"
    if not manifest_path.exists():
        raise IncompleteFoldArtifactError(
            f"load_fold_artifact: no manifest.json at {manifest_path} — either fold {fold_idx} "
            "was never persisted, or a previous persist attempt was interrupted before "
            "completion. Regenerate this fold's artifact rather than assuming it is reusable."
        )
    with open(manifest_path) as f:
        manifest = _json.load(f)
    if manifest.get("status") != "complete":
        raise IncompleteFoldArtifactError(
            f"load_fold_artifact: manifest.json at {manifest_path} does not record "
            f"status='complete' (got {manifest.get('status')!r}) — refusing to reuse an "
            "incomplete fold artifact."
        )

    artifact = PreprocessingArtifact.load(fold_dir / "artifact.json")

    if artifact.scientific_fingerprint() != manifest.get("artifact_fingerprint"):
        raise FoldArtifactMismatchError(
            f"load_fold_artifact: fold {fold_idx}'s artifact.json fingerprint "
            f"({artifact.scientific_fingerprint()[:16]}...) does not match what manifest.json "
            f"recorded ({str(manifest.get('artifact_fingerprint'))[:16]}...) — the artifact file "
            "was modified after this fold was persisted, or the two files came from different "
            "runs. Refusing to reuse it."
        )

    def _subject_fp(subjects: Sequence[str]) -> str:
        ids = sorted(str(s) for s in subjects)
        return hashlib.sha256(_json.dumps(ids, sort_keys=True).encode("utf-8")).hexdigest()

    if expected_train_subjects is not None:
        fp = _subject_fp(expected_train_subjects)
        if fp != manifest.get("train_subjects_fingerprint"):
            raise FoldArtifactMismatchError(
                f"load_fold_artifact: fold {fold_idx}'s persisted train-subject fingerprint "
                f"does not match the expected training subjects for this resume attempt — this "
                "looks like an attempt to reuse a DIFFERENT fold's (or a stale) persisted "
                "artifact. Refusing to reuse it; refit this fold instead."
            )
    if expected_val_subjects is not None:
        fp = _subject_fp(expected_val_subjects)
        if fp != manifest.get("val_subjects_fingerprint"):
            raise FoldArtifactMismatchError(
                f"load_fold_artifact: fold {fold_idx}'s persisted val-subject fingerprint does "
                "not match the expected validation subjects for this resume attempt. Refusing "
                "to reuse it; refit this fold instead."
            )
    if expected_artifact_fingerprint is not None and expected_artifact_fingerprint != manifest.get("artifact_fingerprint"):
        raise FoldArtifactMismatchError(
            f"load_fold_artifact: fold {fold_idx}'s persisted artifact fingerprint does not "
            "match the caller's expected fingerprint. Refusing to reuse it."
        )
    return artifact


def build_fold_cell_dataset(
    normalized_adata, artifact: PreprocessingArtifact, subject_list: Sequence[str],
) -> CellLevelDataset:
    """
    Apply a fold-specific artifact to exactly the cells belonging to
    subject_list, and wrap the result as a CellLevelDataset — same
    downstream shape (`.X`, `.smoke`, `.subject_ids`, ...) the rest of
    benchmarks/ already expects, so build_smoke_subject_summary_features,
    baselines, and the neural adapter all work unchanged on fold data.

    cell_type_id in normalized_adata_for_refit carries the REAL CellTypist
    annotation (run_pipeline_split_aware() runs annotate_cell_types() once,
    BEFORE capturing this snapshot — see preprocess.py) — every fold and the
    outer split see the same deterministic cell-type labels.
    annotate_cell_types() defaults to majority_voting=False (INDUCTIVE):
    each cell's predicted label is a pure function of that cell's own
    expression vector, independent of which other cells are present in the
    same call — so even though annotation runs once here (a performance
    convenience, not a leakage requirement), reannotating any subset of
    these same cells alone would reproduce the identical per-cell labels.
    fit_preprocessing/apply_preprocessing only ever touch gene expression
    (.X), never obs["cell_type_id"], so this label survives fold
    reconstruction unchanged.
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
    is_pseudo_bulk = obs["is_pseudo_bulk"].values.astype(bool) if "is_pseudo_bulk" in obs.columns else None

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
        is_pseudo_bulk=is_pseudo_bulk,
        assay_policy=getattr(artifact, "assay_policy", None) or "single_cell_only",
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
    never a fabricated negative. Pass {} for a caller that only needs the
    smoke-related fields (e.g. a smoke-task MIL model) — every bag then
    carries cancer_label_known=False, never a fabricated cancer outcome.

    Each bag also carries "smoke_known" (per-cell bool array, Phase 5) so a
    subject-level consumer can restrict smoke supervision to cells with a
    verified (non-weak-proxy, non-placeholder) label — see
    CellLevelDataset's own smoke_known docstring for what the underlying
    per-cell value means when smoke_known is False.
    """
    bags = []
    subj = fold_cell_dataset.subject_ids
    for sid in sorted(set(subj.tolist())):
        mask = subj == sid
        if mask.sum() < min_cells_per_subject:
            continue
        outcome = outcomes_by_subject.get(str(sid))
        bag_source = None
        if fold_cell_dataset.dataset_source is not None:
            srcs = np.asarray(fold_cell_dataset.dataset_source)[mask]
            if len(srcs) > 0:
                # A subject is expected to have a single dataset_source across
                # all its cells (see ood.py::_find_cross_source_subjects,
                # which rejects the opposite upstream) — take the majority
                # value defensively rather than asserting here, since this
                # constructor has no access to raise a data-assembly error.
                vals, counts = np.unique(srcs, return_counts=True)
                bag_source = str(vals[np.argmax(counts)])
        bags.append({
            "subject_id": sid,
            "gene_matrix": fold_cell_dataset.X[mask].numpy(),
            "cell_type_ids": fold_cell_dataset.ctype[mask].numpy(),
            "smoke_labels": fold_cell_dataset.smoke[mask].numpy(),
            "smoke_known": fold_cell_dataset.smoke_known[mask].numpy(),
            "malig_labels": fold_cell_dataset.malig[mask].numpy(),
            "malig_known": fold_cell_dataset.malig_known[mask].numpy(),
            "cancer_label": outcome,
            "cancer_label_known": outcome is not None,
            "source": bag_source,
        })
    return bags
