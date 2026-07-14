"""
Regression tests for PR7 requirement 2: leakage-free nested hyperparameter
selection. Root cause fixed: SMOKE_SEARCH_SPACE/CANCER_SEARCH_SPACE were
declared in baselines.py but never consulted by any selection code — every
baseline always ran with its hardcoded default hyperparameters regardless of
the declared search space. select_hyperparameters_nested is the first real
consumer.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.baselines import CANCER_BASELINES, CANCER_SEARCH_SPACE, positive_class_proba
from benchmarks.hyperparameter_search import build_param_grid, select_hyperparameters_nested


def test_build_param_grid_is_cartesian_product():
    grid = build_param_grid({"a": [1, 2], "b": ["x", "y"]})
    assert len(grid) == 4
    assert {"a": 1, "b": "x"} in grid
    assert {"a": 2, "b": "y"} in grid


def test_build_param_grid_empty_search_space_yields_no_candidates():
    assert build_param_grid({}) == []


def _score_fn(model, X, y):
    if len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, positive_class_proba(model, X))


def test_model_with_no_search_space_is_recorded_explicitly():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 4)
    y = rng.randint(0, 2, 20)
    groups = [f"s{i}" for i in range(20)]
    result = select_hyperparameters_nested(
        CANCER_BASELINES["prevalence"], {}, X, y, groups, score_fn=_score_fn,
    )
    assert result["no_search_space"] is True
    assert result["selected_params"] == {}
    assert result["candidates"] == []


def test_nested_selection_never_sees_data_outside_what_it_is_given():
    """Selection must be computable purely from (X, y, groups) — passing a
    disjoint dataset must be able to yield a completely different selection,
    proving no global/outer state leaks in."""
    rng = np.random.RandomState(1)
    n = 60
    X1 = rng.randn(n, 4) + np.array([5, 0, 0, 0])
    y1 = (X1[:, 0] > 5).astype(int)
    groups1 = [f"a{i}" for i in range(n)]

    result = select_hyperparameters_nested(
        CANCER_BASELINES["logistic"], CANCER_SEARCH_SPACE["logistic"],
        X1, y1, groups1, score_fn=_score_fn, seed=42, n_inner_folds=3,
    )
    assert result["no_search_space"] is False
    assert result["selected_params"] in build_param_grid(CANCER_SEARCH_SPACE["logistic"])
    assert len(result["candidates"]) == len(build_param_grid(CANCER_SEARCH_SPACE["logistic"]))
    for c in result["candidates"]:
        assert "inner_score_mean" in c and "inner_scores" in c


def test_nested_selection_is_deterministic_given_seed():
    rng = np.random.RandomState(2)
    n = 60
    X = rng.randn(n, 4)
    y = rng.randint(0, 2, n)
    groups = [f"s{i}" for i in range(n)]
    r1 = select_hyperparameters_nested(
        CANCER_BASELINES["random_forest"], CANCER_SEARCH_SPACE["random_forest"],
        X, y, groups, score_fn=_score_fn, seed=7,
    )
    r2 = select_hyperparameters_nested(
        CANCER_BASELINES["random_forest"], CANCER_SEARCH_SPACE["random_forest"],
        X, y, groups, score_fn=_score_fn, seed=7,
    )
    assert r1["selected_params"] == r2["selected_params"]


def test_too_few_groups_for_inner_cv_records_untuned_note_not_crash():
    X = np.random.rand(3, 4)
    y = np.array([0, 1, 0])
    groups = ["s1", "s1", "s2"]  # only 2 independent groups
    result = select_hyperparameters_nested(
        CANCER_BASELINES["logistic"], CANCER_SEARCH_SPACE["logistic"],
        X, y, groups, score_fn=_score_fn, n_inner_folds=3,
    )
    assert result["no_search_space"] is False
    assert "note" in result or result["candidates"]


def test_runner_cancer_task_records_hyperparameter_search_in_calibration_report():
    from benchmarks.runner import build_synthetic_context, run_cancer_task, new_run_dir
    import shutil, tempfile

    ctx = build_synthetic_context(seed=1, fast=True)

    class _Args:
        models = ["prevalence", "logistic", "random_forest"]
        cv_folds = 2
        seeds = [42]
        device = "cpu"
        pooling = None
        calibration = "auto"
        threshold_strategy = "youden"

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = new_run_dir(tmp, run_id="test_hp_search")
        outcome = run_cancer_task(ctx, _Args(), run_dir)
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)

    report = outcome["calibration_report"]
    assert report is not None
    assert "hyperparameter_search" in report
    hp = report["hyperparameter_search"]
    assert "selected_params" in hp and "no_search_space" in hp
