"""
Regression tests for benchmarks/model_fingerprint.py — deterministic
fingerprints of ACTUAL fitted model state (blocker 2), not fit_seconds or
parameter counts.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.model_fingerprint import (
    UnsupportedModelStateError,
    _canonicalize,
    sklearn_model_state_fingerprint,
    torch_state_dict_fingerprint,
)

_MEMORY_ADDRESS_PATTERN = re.compile(r"0x[0-9a-fA-F]{4,}")


def _synthetic_Xy(seed: int, n: int = 60, n_features: int = 5):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] + rng.randn(n) * 0.1 > 0).astype(int)
    return X, y


def _fit_logreg(C=1.0, seed=42):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(seed)
    X = rng.randn(40, 5)
    y = (X[:, 0] + rng.randn(40) * 0.1 > 0).astype(int)
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X, y)
    return model


def test_sklearn_fingerprint_reproducible_for_identical_fits():
    m1 = _fit_logreg()
    m2 = _fit_logreg()
    assert sklearn_model_state_fingerprint(m1) == sklearn_model_state_fingerprint(m2)


def test_sklearn_fingerprint_differs_for_different_hyperparameters():
    m1 = _fit_logreg(C=0.01)
    m2 = _fit_logreg(C=100.0)
    assert sklearn_model_state_fingerprint(m1) != sklearn_model_state_fingerprint(m2)


def test_sklearn_fingerprint_is_a_64_char_hex_digest():
    fp = sklearn_model_state_fingerprint(_fit_logreg())
    assert len(fp) == 64
    int(fp, 16)  # raises ValueError if not valid hex


def test_sklearn_fingerprint_handles_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(1)
    X = rng.randn(30, 4)
    y = (X[:, 0] > 0).astype(int)
    pipe1 = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    pipe1.fit(X, y)
    pipe2 = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    pipe2.fit(X, y)
    assert sklearn_model_state_fingerprint(pipe1) == sklearn_model_state_fingerprint(pipe2)


def test_torch_fingerprint_reproducible_and_device_independent():
    torch = pytest.importorskip("torch")

    def make_model():
        torch.manual_seed(0)
        return torch.nn.Linear(4, 2)

    m1 = make_model()
    m2 = make_model()
    fp1 = torch_state_dict_fingerprint(m1)
    fp2 = torch_state_dict_fingerprint(m2)
    assert fp1 == fp2
    assert len(fp1) == 64

    # An explicit .to("cpu") round trip (device placement) must not change it.
    m1.to("cpu")
    assert torch_state_dict_fingerprint(m1) == fp1


def test_torch_fingerprint_differs_for_different_weights():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    m1 = torch.nn.Linear(4, 2)
    torch.manual_seed(1)
    m2 = torch.nn.Linear(4, 2)
    assert torch_state_dict_fingerprint(m1) != torch_state_dict_fingerprint(m2)


# ─── Every estimator type SMOKE_BASELINES / CANCER_BASELINES can produce ─────
# (registered in benchmarks/baselines.py) — each must fingerprint
# deterministically and without a repr() fallback.

def _fit_dummy_prior(seed):
    from sklearn.dummy import DummyClassifier
    X, y = _synthetic_Xy(seed)
    m = DummyClassifier(strategy="prior")
    m.fit(X, y)
    return m


def _fit_dummy_constant(seed, constant=1):
    from sklearn.dummy import DummyClassifier
    X, y = _synthetic_Xy(seed)
    m = DummyClassifier(strategy="constant", constant=constant)
    m.fit(X, y)
    return m


def _fit_random_forest(seed, n_estimators=5):
    from sklearn.ensemble import RandomForestClassifier
    X, y = _synthetic_Xy(seed)
    m = RandomForestClassifier(n_estimators=n_estimators, random_state=0, max_depth=3)
    m.fit(X, y)
    return m


def _fit_hist_gradient_boosting(seed, max_iter=5):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y = _synthetic_Xy(seed)
    m = HistGradientBoostingClassifier(max_iter=max_iter, random_state=0)
    m.fit(X, y)
    return m


def _fit_small_mlp(seed, hidden_layer_sizes=(8,)):
    from sklearn.neural_network import MLPClassifier
    X, y = _synthetic_Xy(seed)
    m = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=50, random_state=0)
    m.fit(X, y)
    return m


def _fit_scaled_logistic_pipeline(seed, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = _synthetic_Xy(seed)
    m = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=C, max_iter=1000))])
    m.fit(X, y)
    return m


_BASELINE_FITTERS = {
    "dummy_prior": _fit_dummy_prior,
    "dummy_constant": _fit_dummy_constant,
    "logistic_pipeline": _fit_scaled_logistic_pipeline,
    "random_forest": _fit_random_forest,
    "hist_gradient_boosting": _fit_hist_gradient_boosting,
    "small_mlp": _fit_small_mlp,
}


@pytest.mark.parametrize("name", sorted(_BASELINE_FITTERS))
def test_registered_baseline_fingerprint_reproducible_for_identical_seeded_fits(name):
    fitter = _BASELINE_FITTERS[name]
    m1 = fitter(seed=7)
    m2 = fitter(seed=7)
    fp1 = sklearn_model_state_fingerprint(m1)
    fp2 = sklearn_model_state_fingerprint(m2)
    assert fp1 == fp2
    assert len(fp1) == 64
    int(fp1, 16)


@pytest.mark.parametrize("name", sorted(_BASELINE_FITTERS))
def test_registered_baseline_fingerprint_differs_for_different_data(name):
    fitter = _BASELINE_FITTERS[name]
    m1 = fitter(seed=7)
    m2 = fitter(seed=99)
    assert sklearn_model_state_fingerprint(m1) != sklearn_model_state_fingerprint(m2)


@pytest.mark.parametrize("name", sorted(_BASELINE_FITTERS))
def test_registered_baseline_fingerprint_stable_across_repeated_calls(name):
    m = _BASELINE_FITTERS[name](seed=3)
    fp1 = sklearn_model_state_fingerprint(m)
    fp2 = sklearn_model_state_fingerprint(m)
    assert fp1 == fp2


@pytest.mark.parametrize("name", sorted(_BASELINE_FITTERS))
def test_registered_baseline_fingerprinting_does_not_change_predictions(name):
    m = _BASELINE_FITTERS[name](seed=5)
    X, _ = _synthetic_Xy(5)
    before = m.predict(X).tolist()
    sklearn_model_state_fingerprint(m)
    after = m.predict(X).tolist()
    assert before == after


@pytest.mark.parametrize("name", sorted(_BASELINE_FITTERS))
def test_registered_baseline_canonical_state_contains_no_memory_address(name):
    m = _BASELINE_FITTERS[name](seed=11)
    canonical = _canonicalize(m)
    blob = json.dumps(canonical, sort_keys=True)
    assert not _MEMORY_ADDRESS_PATTERN.search(blob), (
        f"{name}: canonical state contains what looks like a memory address"
    )


def test_dummy_constant_fingerprint_differs_by_constant_value():
    fp0 = sklearn_model_state_fingerprint(_fit_dummy_constant(seed=1, constant=0))
    fp1 = sklearn_model_state_fingerprint(_fit_dummy_constant(seed=1, constant=1))
    assert fp0 != fp1


def test_random_forest_fingerprint_reflects_actual_tree_structure_not_just_repr():
    """Two forests fit on different data but with identical shape/config
    must not collide — this is the case the previous repr()-fallback could
    not distinguish reliably because sklearn.tree._tree.Tree has no
    __dict__ and its repr() carries only a memory address."""
    m1 = _fit_random_forest(seed=1)
    m2 = _fit_random_forest(seed=2)
    assert sklearn_model_state_fingerprint(m1) != sklearn_model_state_fingerprint(m2)


def test_hist_gradient_boosting_fingerprint_reflects_learned_trees():
    """The learned trees live in the private _predictors attribute, not in
    any trailing-underscore public attribute — this test fails if that
    private state is ever dropped from canonicalization."""
    m1 = _fit_hist_gradient_boosting(seed=1)
    m2 = _fit_hist_gradient_boosting(seed=2)
    assert sklearn_model_state_fingerprint(m1) != sklearn_model_state_fingerprint(m2)


def test_unsupported_fitted_object_type_raises_instead_of_using_repr():
    class OpaqueFittedThing:
        __slots__ = ()  # no __dict__, mimics a Cython extension type

    with pytest.raises(UnsupportedModelStateError):
        sklearn_model_state_fingerprint(OpaqueFittedThing())


def test_sklearn_tree_extension_type_is_explicitly_canonicalized_not_repr_fallback():
    """sklearn.tree._tree.Tree has no __dict__; confirms it is handled by
    the explicit Tree branch rather than reaching the repr() fallback (which
    no longer exists) or raising UnsupportedModelStateError."""
    m = _fit_random_forest(seed=4, n_estimators=1)
    tree_obj = m.estimators_[0].tree_
    canonical = _canonicalize(tree_obj)
    assert canonical["__class__"] == "sklearn.tree._tree.Tree"
    assert "children_left" in canonical
    assert "value" in canonical
