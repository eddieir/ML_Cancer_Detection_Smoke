"""
Regression tests for PR7 blocker 4: hyperparameter selection must be
integrated into every OUTER CV fold via a proper INNER grouped-CV that
refits preprocessing independently per inner fold, using only that inner
fold's own training subjects — never the outer fold's validation subjects,
and never any subject outside the outer fold's training set.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import anndata as ad


def _normalized_adata(n_subjects=12, cells_per_subject=12, g=8, n_ct=4, seed=0):
    rng = np.random.RandomState(seed)
    subject_ids, smoke_types, cancer, cell_types, sources = [], [], [], [], []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * cells_per_subject
        smoke_types += [i % 2] * cells_per_subject
        cell_types += [i % n_ct] * cells_per_subject
        sources += ["sourceA"] * cells_per_subject
    n = len(subject_ids)
    X = rng.randn(n, g).astype("float32")
    obs = pd.DataFrame({
        "subject_id": subject_ids, "smoke_type": smoke_types, "cell_type_id": cell_types,
        "malignancy": 0.0, "malignancy_known": False, "exposure_dose": -1.0, "source": sources,
        "is_pseudo_bulk": False,
    })
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))


class _Ctx:
    def __init__(self, normalized_adata, n_hvgs=8):
        self.normalized_adata_for_refit = normalized_adata

        class _Artifact:
            pass
        _Artifact.n_hvgs = n_hvgs
        self.preprocessing_artifact = _Artifact()


def _smoke_score_fn():
    from benchmarks.baselines import SmokeLogisticRegression
    from benchmarks.features import build_smoke_subject_summary_features
    from benchmarks.metrics import full_smoke_metrics_report

    def fn(params, artifact, train_ds, val_ds, seed):
        Xtr, ytr, _, _ = build_smoke_subject_summary_features(train_ds, num_cell_types=4, num_classes=2)
        Xva, yva, _, _ = build_smoke_subject_summary_features(val_ds, num_cell_types=4, num_classes=2)
        model = SmokeLogisticRegression(**params).fit(Xtr, ytr, seed=seed)
        preds = model.predict(Xva)
        return full_smoke_metrics_report(yva, preds, 2)["macro_f1"]
    return fn


def test_inner_folds_are_subject_disjoint():
    from benchmarks.hyperparameter_search import select_nested_hyperparameters_with_refit, build_param_grid
    from benchmarks.baselines import SMOKE_SEARCH_SPACE

    na = _normalized_adata()
    ctx = _Ctx(na)
    outer_train = [f"sub_{i}" for i in range(10)]
    label_by_subject = {f"sub_{i}": i % 2 for i in range(12)}

    result = select_nested_hyperparameters_with_refit(
        ctx, outer_train, label_by_subject, build_param_grid(SMOKE_SEARCH_SPACE["logistic"]),
        fit_score_fn=_smoke_score_fn(), seed=42, n_inner_folds=2, n_hvgs=8,
    )
    assert result["no_search_space"] is False
    assert result["inner_folds"], "no inner folds recorded"
    for fold in result["inner_folds"]:
        assert set(fold["train"]).isdisjoint(set(fold["val"]))
        # inner folds only ever subdivide the OUTER-TRAIN subjects
        assert set(fold["train"]) | set(fold["val"]) <= set(outer_train)


def test_corrupting_subjects_outside_outer_train_does_not_change_selection():
    """Subjects outside the outer fold's own training set (standing in for
    outer-validation/test subjects) must never influence which
    hyperparameters get selected."""
    from benchmarks.hyperparameter_search import select_nested_hyperparameters_with_refit, build_param_grid
    from benchmarks.baselines import SMOKE_SEARCH_SPACE

    na = _normalized_adata()
    outer_train = [f"sub_{i}" for i in range(10)]
    label_by_subject = {f"sub_{i}": i % 2 for i in range(12)}
    candidates = build_param_grid(SMOKE_SEARCH_SPACE["logistic"])

    result_before = select_nested_hyperparameters_with_refit(
        _Ctx(na), outer_train, label_by_subject, candidates,
        fit_score_fn=_smoke_score_fn(), seed=42, n_inner_folds=2, n_hvgs=8,
    )

    na_corrupted = na.copy()
    outside_mask = ~na_corrupted.obs["subject_id"].astype(str).isin(set(outer_train)).values
    na_corrupted.X[outside_mask] = 999999.0

    result_after = select_nested_hyperparameters_with_refit(
        _Ctx(na_corrupted), outer_train, label_by_subject, candidates,
        fit_score_fn=_smoke_score_fn(), seed=42, n_inner_folds=2, n_hvgs=8,
    )

    assert result_before["selected_params"] == result_after["selected_params"]
    assert result_before["inner_folds"] == result_after["inner_folds"]


def test_deterministic_seed_reproduces_identical_selection():
    from benchmarks.hyperparameter_search import select_nested_hyperparameters_with_refit, build_param_grid
    from benchmarks.baselines import SMOKE_SEARCH_SPACE

    na = _normalized_adata()
    outer_train = [f"sub_{i}" for i in range(10)]
    label_by_subject = {f"sub_{i}": i % 2 for i in range(12)}
    candidates = build_param_grid(SMOKE_SEARCH_SPACE["logistic"])

    r1 = select_nested_hyperparameters_with_refit(
        _Ctx(na), outer_train, label_by_subject, candidates,
        fit_score_fn=_smoke_score_fn(), seed=7, n_inner_folds=2, n_hvgs=8,
    )
    r2 = select_nested_hyperparameters_with_refit(
        _Ctx(na), outer_train, label_by_subject, candidates,
        fit_score_fn=_smoke_score_fn(), seed=7, n_inner_folds=2, n_hvgs=8,
    )
    assert r1["selected_params"] == r2["selected_params"]
    assert r1["candidates"] == r2["candidates"]


def test_no_search_space_recorded_true_for_empty_grid():
    from benchmarks.hyperparameter_search import select_nested_hyperparameters_with_refit

    na = _normalized_adata()
    result = select_nested_hyperparameters_with_refit(
        _Ctx(na), [f"sub_{i}" for i in range(10)], {f"sub_{i}": i % 2 for i in range(12)},
        candidates=[], fit_score_fn=lambda *a: 1.0, seed=42,
    )
    assert result["no_search_space"] is True
    assert result["selected_params"] == {}


def test_undefined_inner_folds_never_coerced_to_a_filler_score():
    from benchmarks.hyperparameter_search import select_nested_hyperparameters_with_refit

    na = _normalized_adata()
    result = select_nested_hyperparameters_with_refit(
        _Ctx(na), [f"sub_{i}" for i in range(10)], {f"sub_{i}": i % 2 for i in range(12)},
        candidates=[{"C": 1.0}, {"C": 10.0}], fit_score_fn=lambda *a: None,
        seed=42, n_inner_folds=2, n_hvgs=8,
    )
    assert result["no_search_space"] is False
    assert "note" in result
    for c in result["candidates"]:
        assert c["inner_score_mean"] is None
        assert c["inner_scores"] == []


def test_task_a_outer_fold_records_selected_hyperparameters():
    from benchmarks.runner import build_synthetic_context
    from benchmarks.cross_validation import run_smoke_cv

    ctx = build_synthetic_context(seed=1, fast=True)
    report = run_smoke_cv(ctx, ["logistic"], n_folds=2, seeds=[42], device="cpu")
    fold = report["results"]["logistic"]["folds"][0]
    assert "hyperparameter_search" in fold
    assert fold["hyperparameter_search"]["no_search_space"] is False
    assert "selected_params" in fold["hyperparameter_search"]


def test_task_b_outer_fold_records_selected_hyperparameters():
    from benchmarks.runner import build_synthetic_context
    from benchmarks.cross_validation import run_cancer_cv

    ctx = build_synthetic_context(seed=1, fast=True)
    report = run_cancer_cv(ctx, ["logistic"], n_folds=2, seeds=[42], device="cpu")
    fold = report["results"]["logistic"]["folds"][0]
    assert "hyperparameter_search" in fold
    assert fold["hyperparameter_search"]["no_search_space"] is False
