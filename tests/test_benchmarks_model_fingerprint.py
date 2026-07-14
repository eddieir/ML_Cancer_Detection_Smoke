"""
Regression tests for benchmarks/model_fingerprint.py — deterministic
fingerprints of ACTUAL fitted model state (blocker 2), not fit_seconds or
parameter counts.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.model_fingerprint import (
    sklearn_model_state_fingerprint,
    torch_state_dict_fingerprint,
)


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
