"""
Phase 4 — batch-correction safety (Step 7 of the preprocessing-artifact
hardening effort).

Harmony (data/transforms.py::batch_correct), this project's only
implemented batch-correction method, is TRANSDUCTIVE: harmonypy has no
train-only-fit / apply-to-new-data API, so running it at all requires joint
access to every cell it corrects. preprocess.py already gates it behind an
explicit opt-in (preprocessing.batch_correction.mode) and records whether a
run used it on ExperimentContext.transductive_batch_correction. This file
covers what was previously missing: an explicit UnsafeBatchCorrectionError
that STOPS a transductively-batch-corrected run from reaching CV/OOF
fold refitting, the final development-pool fit, or frozen-test evaluation,
plus the artifact's own explicit batch_correction_status field (items
45-48 of the original adversarial checklist).
"""
import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.preprocessing import GeneContractError, PreprocessingArtifact, fit_preprocessing
from data.transforms import UnsafeBatchCorrectionError, assert_batch_correction_safe


def _adata(n_subjects=10, cells_per_subject=20, n_genes=30, seed=0):
    rng = np.random.default_rng(seed)
    subject_ids, batches = [], []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * cells_per_subject
        batches += [f"batch_{i % 3}"] * cells_per_subject
    n = len(subject_ids)
    X = rng.random((n, n_genes)).astype("float32")
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame({"subject_id": subject_ids, "batch": batches, "is_pseudo_bulk": [False] * n}, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))


# ─── assert_batch_correction_safe / UnsafeBatchCorrectionError ────────────

def test_assert_batch_correction_safe_passes_when_disabled():
    assert_batch_correction_safe(False, context_name="unit_test")  # must not raise


def test_assert_batch_correction_safe_rejects_transductive():
    with pytest.raises(UnsafeBatchCorrectionError):
        assert_batch_correction_safe(True, context_name="unit_test")


def test_fold_preprocessing_require_normalized_adata_rejects_transductive_context():
    from benchmarks.fold_preprocessing import require_normalized_adata

    class _Ctx:
        normalized_adata_for_refit = object()
        transductive_batch_correction = True

    with pytest.raises(UnsafeBatchCorrectionError):
        require_normalized_adata(_Ctx())


def test_fold_preprocessing_require_normalized_adata_allows_non_transductive_context():
    from benchmarks.fold_preprocessing import require_normalized_adata

    sentinel = object()

    class _Ctx:
        normalized_adata_for_refit = sentinel
        transductive_batch_correction = False

    assert require_normalized_adata(_Ctx()) is sentinel


def test_runner_frozen_test_stage_rejects_transductive_context():
    """A run whose outer artifact used transductive Harmony must never even
    reach guard acquisition for the frozen-test stage."""
    from benchmarks.runner import build_synthetic_context, new_run_dir, run_cancer_task

    class _Args:
        models = ["prevalence"]
        cv_folds = 2
        seeds = [42]
        device = "cpu"
        pooling = None
        calibration = "auto"
        threshold_strategy = "youden"

    ctx = build_synthetic_context(seed=1, fast=True)
    ctx = dataclasses.replace(ctx, transductive_batch_correction=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
            with pytest.raises(UnsafeBatchCorrectionError):
                run_cancer_task(ctx, _Args(), run_dir)
    finally:
        shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


# ─── Item 45: unsafe transductive correction is rejected ──────────────────

def test_transductive_status_artifact_is_rejected_from_leakage_free_paths():
    adata = _adata(seed=1)
    artifact = fit_preprocessing(
        adata, {f"sub_{i}" for i in range(10)}, n_hvgs=30,
        batch_correction_status="transductive_diagnostic_only",
    )
    assert artifact.batch_correction_status == "transductive_diagnostic_only"
    # The artifact itself doesn't refuse to exist (preprocess.py's outer,
    # disclosed diagnostic path is allowed to produce one) — what must be
    # refused is USING that status to justify skipping the CV/OOF/frozen-
    # test guard, which is exactly what assert_batch_correction_safe does
    # at every one of those call sites (see tests above).
    with pytest.raises(UnsafeBatchCorrectionError):
        assert_batch_correction_safe(
            artifact.batch_correction_status == "transductive_diagnostic_only",
            context_name="unit_test",
        )


# ─── Item 46: validation/test corruption cannot change fitted correction
#     state (no inductive state exists to corrupt — the strongest available
#     honest proof is that heavily corrupting held-out expression AND batch
#     composition never changes the training artifact under the default,
#     batch-correction-disabled path) ────────────────────────────────────

def test_heavy_val_test_expression_and_batch_corruption_leaves_training_artifact_byte_identical():
    adata = _adata(n_subjects=12, seed=2)
    train_subjects = {f"sub_{i}" for i in range(8)}  # 8/12 train

    artifact1 = fit_preprocessing(adata, train_subjects, n_hvgs=30)

    corrupted = adata.copy()
    non_train_mask = ~corrupted.obs["subject_id"].isin(train_subjects).values
    X = corrupted.X.copy()
    X[non_train_mask] = X[non_train_mask] * 10_000 + 50_000
    corrupted.X = X
    # Heavily scramble batch composition for non-train cells too — every
    # non-train cell gets a brand-new, never-before-seen batch label.
    new_batches = corrupted.obs["batch"].astype(str).to_numpy(copy=True)
    new_batches[non_train_mask] = [f"corrupted_batch_{i}" for i in range(non_train_mask.sum())]
    corrupted.obs["batch"] = new_batches

    artifact2 = fit_preprocessing(corrupted, train_subjects, n_hvgs=30)

    assert artifact1.scientific_fingerprint() == artifact2.scientific_fingerprint()
    assert artifact1.gene_list == artifact2.gene_list
    assert artifact1.gene_means == artifact2.gene_means
    assert artifact1.gene_stds == artifact2.gene_stds


# ─── Item 47: unseen/unsupported batches fail clearly ──────────────────────

def test_unseen_batch_at_apply_time_fails_clearly_when_artifact_recorded_transductive_use():
    """There is no supported inductive "apply Harmony to a new batch" step
    at all — the only way an artifact could claim batch correction ran is
    batch_correction_status='transductive_diagnostic_only', and
    assert_batch_correction_safe refuses to let that state reach any
    leakage-free application regardless of what batch values later appear.
    This is the fail-clearly behavior item 47 asks for: it fails at the
    point the artifact's provenance is checked, not silently at inference
    time with a plausible-looking but meaningless output."""
    with pytest.raises(UnsafeBatchCorrectionError):
        assert_batch_correction_safe(True, context_name="apply_to_new_batch")


# ─── Item 48: disabled correction is represented explicitly in the artifact

def test_default_artifact_explicitly_records_batch_correction_disabled():
    adata = _adata(seed=3)
    artifact = fit_preprocessing(adata, {f"sub_{i}" for i in range(10)}, n_hvgs=30)
    assert artifact.batch_correction_status == "disabled"


def test_invalid_batch_correction_status_rejected():
    adata = _adata(seed=4)
    artifact = fit_preprocessing(adata, {f"sub_{i}" for i in range(10)}, n_hvgs=30)
    d = artifact.to_dict()
    d["batch_correction_status"] = "silently_fine"
    with pytest.raises(GeneContractError):
        PreprocessingArtifact(**d)


def test_batch_correction_status_included_in_scientific_fingerprint():
    adata = _adata(seed=5)
    train = {f"sub_{i}" for i in range(10)}
    a_disabled = fit_preprocessing(adata, train, n_hvgs=30, batch_correction_status="disabled")
    a_transductive = fit_preprocessing(adata, train, n_hvgs=30, batch_correction_status="transductive_diagnostic_only")
    assert a_disabled.scientific_fingerprint() != a_transductive.scientific_fingerprint()
