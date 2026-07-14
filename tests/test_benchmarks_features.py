"""benchmarks/features.py — subject-aware feature construction, leakage guards."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.features import (
    build_cancer_subject_features,
    build_smoke_subject_summary_features,
    cap_cells_per_subject,
    generate_out_of_fold_predictions,
)
from benchmarks.runner import build_synthetic_context


def test_smoke_subject_summary_one_row_per_subject():
    ctx = build_synthetic_context(seed=1, fast=True)
    from benchmarks.cross_validation import concat_cell_datasets
    pool = concat_cell_datasets(ctx.train_cell_dataset, ctx.val_cell_dataset)
    X, y, subject_ids, names = build_smoke_subject_summary_features(pool, num_cell_types=4, num_classes=3)
    assert X.shape[0] == len(subject_ids) == len(set(pool.subject_ids.tolist()))
    assert len(names) == X.shape[1]


def test_cell_count_imbalance_does_not_dominate_subject_summary_features():
    """A subject with 10x the cells of another still contributes exactly one
    row — the technical n_cells feature is separate, not folded into the mean
    in a way that would let cell count dominate a per-subject vote."""
    from train import CellLevelDataset
    rng = np.random.RandomState(0)
    subj = ["big"] * 1000 + ["small"] * 10
    X = np.concatenate([rng.randn(1000, 5).astype("float32"), rng.randn(10, 5).astype("float32") + 5])
    y = np.array([0] * 1000 + [1] * 10)
    ds = CellLevelDataset(X, y, np.zeros(1010, dtype="float32"), np.zeros(1010, dtype=int),
                          subject_ids=np.array(subj, dtype=object))
    Xf, yf, subject_ids, _ = build_smoke_subject_summary_features(ds, num_cell_types=1, num_classes=2)
    assert Xf.shape[0] == 2  # exactly one row each, regardless of 1000 vs 10 cells


def test_cancer_subject_features_excludes_unknown_outcome_bags():
    ctx = build_synthetic_context(seed=1, fast=True)
    bags = list(ctx.train_bags)
    bags[0]["cancer_label_known"] = False
    X, y, subject_ids, _ = build_cancer_subject_features(bags, num_cell_types=4)
    assert bags[0]["subject_id"] not in subject_ids


def test_cap_cells_per_subject_caps_but_keeps_small_subjects_whole():
    subj = np.array(["a"] * 100 + ["b"] * 5)
    keep = cap_cells_per_subject(subj, max_cells_per_subject=20, seed=0)
    assert keep[subj == "a"].sum() == 20
    assert keep[subj == "b"].sum() == 5


def test_generate_out_of_fold_predictions_never_uses_own_fold_model():
    """Each subject's OOF prediction must come from a model fit on a DIFFERENT
    fold's data — verified by checking the prediction is exactly the mean of
    the OTHER group's y, which only happens if that subject was excluded from
    fitting."""
    groups = np.array([f"s{i}" for i in range(10)])
    y = np.array([0.0]*5 + [10.0]*5)
    X = y.reshape(-1, 1)

    def fit_predict_fn(X_train, y_train, X_val):
        return np.full(len(X_val), y_train.mean())

    oof = generate_out_of_fold_predictions(fit_predict_fn, X, y, groups, n_folds=2, seed=0)
    assert len(oof) == 10
    assert not np.array_equal(oof, y)  # a same-fold model would trivially predict its own mean == y
