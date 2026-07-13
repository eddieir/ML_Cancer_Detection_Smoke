"""benchmarks/baselines.py — common interface, determinism."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.baselines import CANCER_BASELINES, SMOKE_BASELINES


def _toy(seed=0, n=60, g=8, k=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, g).astype("float32")
    y = rng.randint(0, k, n)
    return X, y


def test_every_smoke_baseline_implements_common_interface():
    X, y = _toy()
    for name, cls in SMOKE_BASELINES.items():
        model = cls().fit(X, y, seed=42)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        proba = model.predict_proba(X)
        assert proba.shape[0] == len(y)
        meta = model.metadata()
        assert meta["name"] == name
        assert "sklearn_version" in meta


def test_every_cancer_baseline_implements_common_interface():
    X, y = _toy(k=2)
    for name, cls in CANCER_BASELINES.items():
        model = cls().fit(X, y, seed=42)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)


def test_baseline_fit_is_deterministic_given_seed():
    X, y = _toy(k=2)
    m1 = CANCER_BASELINES["random_forest"]().fit(X, y, seed=42)
    m2 = CANCER_BASELINES["random_forest"]().fit(X, y, seed=42)
    assert np.array_equal(m1.predict_proba(X), m2.predict_proba(X))


def test_baseline_never_sees_data_it_is_not_given():
    """fit() must not silently read from any global state — verified by
    confirming two disjoint datasets produce different fitted models."""
    X1, y1 = _toy(seed=1, k=2)
    X2, y2 = _toy(seed=2, k=2)
    m1 = CANCER_BASELINES["logistic"]().fit(X1, y1, seed=42)
    m2 = CANCER_BASELINES["logistic"]().fit(X2, y2, seed=42)
    # logistic regression is now wrapped in Pipeline(StandardScaler, model)
    # to give it fold-only-fitted feature scaling — see baselines.py.
    assert not np.array_equal(m1.model.named_steps["model"].coef_, m2.model.named_steps["model"].coef_)


def test_scaled_baselines_fit_scaler_only_on_given_training_data():
    """Changing 'validation' feature magnitude after fitting must not alter
    the fitted scaler's mean/variance — proves the scaler is fit once, on
    fit()'s X only, not refit or updated when predict() sees different data."""
    X, y = _toy(seed=3, k=2, n=80)
    model = CANCER_BASELINES["logistic"]().fit(X, y, seed=42)
    scaler = model.model.named_steps["scaler"]
    mean_before, var_before = scaler.mean_.copy(), scaler.var_.copy()

    X_huge = X * 1000 + 500  # wildly different magnitude "validation" data
    model.predict_proba(X_huge)

    assert np.array_equal(scaler.mean_, mean_before)
    assert np.array_equal(scaler.var_, var_before)


def test_scaled_small_mlp_also_uses_fold_only_scaler():
    X, y = _toy(seed=4, k=2, n=80)
    model = CANCER_BASELINES["small_mlp"]().fit(X, y, seed=42)
    assert "scaler" in model.model.named_steps
