"""
Regression tests for PR7 requirement 3: cap_cells_per_subject must be wired
deterministically into cell-level CV (the neural adapter's Phase 1 training),
applied independently per split, so a subject with many more cells than
others cannot dominate a fold's training/evaluation — and every model's CV
report must explicitly label whether it ran in "subject_summary" or
"cell_capped" feature mode rather than leaving the distinction implicit.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.features import cap_cell_dataset, cap_cells_per_subject


def _make_cell_dataset(subjects_and_counts, g=6, seed=0):
    from train import CellLevelDataset
    rng = np.random.RandomState(seed)
    subj_ids, X, smoke = [], [], []
    for sid, n in subjects_and_counts:
        subj_ids += [sid] * n
        X.append(rng.randn(n, g).astype("float32"))
        smoke += [0] * n
    X = np.vstack(X)
    return CellLevelDataset(
        gene_matrix=X, smoke_labels=np.array(smoke), malignancy_labels=np.zeros(len(smoke), dtype="float32"),
        cell_type_ids=np.zeros(len(smoke), dtype=int), subject_ids=np.array(subj_ids, dtype=object),
        is_pseudo_bulk=np.zeros(len(smoke), dtype=bool),
    )


def test_cap_cells_per_subject_is_deterministic_given_seed():
    ids = np.array(["a"] * 50 + ["b"] * 3)
    keep1 = cap_cells_per_subject(ids, max_cells_per_subject=10, seed=7)
    keep2 = cap_cells_per_subject(ids, max_cells_per_subject=10, seed=7)
    assert np.array_equal(keep1, keep2)


def test_cap_cells_per_subject_caps_large_subjects_not_small_ones():
    ids = np.array(["a"] * 50 + ["b"] * 3)
    keep = cap_cells_per_subject(ids, max_cells_per_subject=10, seed=1)
    assert keep[:50].sum() == 10   # subject "a" capped down to the limit
    assert keep[50:].sum() == 3    # subject "b" (fewer than the cap) untouched


def test_cap_cell_dataset_group_isolation_train_cap_does_not_see_val_data():
    """Capping the train split must depend only on the train split's own
    per-subject cell counts, never on the val split's — proven by capping
    each split separately and confirming the KEPT subject-id set for each
    split's own capped output only ever contains that split's subjects."""
    train_ds = _make_cell_dataset([("s1", 40), ("s2", 5)], seed=1)
    val_ds = _make_cell_dataset([("s3", 40), ("s4", 5)], seed=2)

    train_capped = cap_cell_dataset(train_ds, max_cells_per_subject=10, seed=42)
    val_capped = cap_cell_dataset(val_ds, max_cells_per_subject=10, seed=42)

    assert set(train_capped.subject_ids.tolist()) == {"s1", "s2"}
    assert set(val_capped.subject_ids.tolist()) == {"s3", "s4"}
    assert (train_capped.subject_ids == "s1").sum() == 10
    assert (train_capped.subject_ids == "s2").sum() == 5
    assert (val_capped.subject_ids == "s3").sum() == 10
    assert (val_capped.subject_ids == "s4").sum() == 5


def test_cap_cell_dataset_preserves_gene_matrix_alignment():
    ds = _make_cell_dataset([("s1", 20)], g=4, seed=3)
    capped = cap_cell_dataset(ds, max_cells_per_subject=5, seed=1)
    assert capped.X.shape == (5, 4)
    assert len(capped.smoke) == 5
    assert len(capped.ctype) == 5


def test_run_smoke_cv_neural_reports_cell_capped_feature_mode():
    from benchmarks.cross_validation import run_smoke_cv
    from benchmarks.runner import build_synthetic_context
    import shutil

    ctx = build_synthetic_context(seed=1, fast=True)
    result = run_smoke_cv(ctx, ["majority", "neural"], n_folds=2, seeds=[42], max_cells_per_subject=5)
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)

    neural_folds = result["results"]["neural"]["folds"]
    assert all(f["feature_mode"] == "cell_capped" for f in neural_folds)
    assert all(f["max_cells_per_subject"] == 5 for f in neural_folds)
    assert all(f["n_cells_after_cap"]["train"] <= f["n_cells_before_cap"]["train"] for f in neural_folds)

    baseline_folds = result["results"]["majority"]["folds"]
    assert all(f["feature_mode"] == "subject_summary" for f in baseline_folds)
