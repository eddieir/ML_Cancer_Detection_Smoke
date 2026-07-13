"""
benchmarks/baselines.py — simple, deterministic baseline models sharing one
common fit/predict interface, for both Task A (smoke-type) and Task B
(cancer prediction).

Every baseline:
  * fits only on the X/y it's given (callers are responsible for passing
    train-only data — see cross_validation.py)
  * uses a fixed random_state derived from the seed passed to fit()
  * records its hyperparameters and the sklearn version used
  * exposes predict() and (for classifiers) predict_proba()

No baseline here tunes itself against test data — hyperparameter selection
lives in cross_validation.py and only ever sees train/validation folds.
"""

from typing import Dict, Optional

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class Baseline:
    """Common interface every baseline implements."""

    name = "baseline"

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> "Baseline":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def metadata(self) -> Dict:
        return {
            "name": self.name,
            "hyperparams": dict(self.hyperparams),
            "sklearn_version": sklearn.__version__,
        }


def _sklearn_fit(baseline: Baseline, model_cls, X, y, seed, **extra_params):
    params = dict(baseline.hyperparams)
    params.update(extra_params)
    if "random_state" in model_cls().get_params():
        params["random_state"] = seed
    baseline.model = model_cls(**params)
    baseline.model.fit(X, y)
    baseline.classes_ = baseline.model.classes_
    return baseline


# ─── Task A: smoke-type classification ────────────────────────────────────────

class SmokeMajorityBaseline(Baseline):
    name = "majority"

    def fit(self, X, y, seed=42):
        self.model = DummyClassifier(strategy="most_frequent")
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self


class SmokeLogisticRegression(Baseline):
    name = "logistic"

    def __init__(self, C: float = 1.0, class_weight: Optional[str] = "balanced"):
        super().__init__(C=C, class_weight=class_weight)

    def fit(self, X, y, seed=42):
        return _sklearn_fit(
            self, lambda **p: LogisticRegression(max_iter=2000, **p), X, y, seed,
        )


class SmokeRandomForest(Baseline):
    name = "random_forest"

    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1, class_weight: Optional[str] = "balanced"):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                          min_samples_leaf=min_samples_leaf, class_weight=class_weight)

    def fit(self, X, y, seed=42):
        return _sklearn_fit(self, RandomForestClassifier, X, y, seed)


class SmokeGradientBoosting(Baseline):
    name = "gradient_boosting"

    def __init__(self, learning_rate: float = 0.1, max_depth: Optional[int] = None,
                 max_iter: int = 200):
        super().__init__(learning_rate=learning_rate, max_depth=max_depth, max_iter=max_iter)

    def fit(self, X, y, seed=42):
        return _sklearn_fit(self, HistGradientBoostingClassifier, X, y, seed)


class SmokeSmallMLP(Baseline):
    """Deliberately tiny relative to MultiSmokeCancerNet's ~2.9M parameters:
    a single 32-unit hidden layer is a few thousand parameters."""
    name = "small_mlp"

    def __init__(self, hidden_layer_sizes=(32,), alpha: float = 1e-3, max_iter: int = 500):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=max_iter)

    def fit(self, X, y, seed=42):
        return _sklearn_fit(
            self, MLPClassifier, X, y, seed, early_stopping=True, n_iter_no_change=10,
        )


SMOKE_BASELINES = {
    "majority":          SmokeMajorityBaseline,
    "logistic":          SmokeLogisticRegression,
    "random_forest":     SmokeRandomForest,
    "gradient_boosting": SmokeGradientBoosting,
    "small_mlp":         SmokeSmallMLP,
}


# ─── Task B: subject-level cancer prediction ──────────────────────────────────

class CancerPrevalenceBaseline(Baseline):
    """Predicts the train-fold positive rate for every subject — the floor
    any real model must beat."""
    name = "prevalence"

    def fit(self, X, y, seed=42):
        self.model = DummyClassifier(strategy="prior")
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self


class CancerLogisticRegression(Baseline):
    name = "logistic"

    def __init__(self, C: float = 1.0, class_weight: Optional[str] = "balanced"):
        super().__init__(C=C, class_weight=class_weight)

    def fit(self, X, y, seed=42):
        return _sklearn_fit(self, lambda **p: LogisticRegression(max_iter=2000, **p), X, y, seed)


class CancerRandomForest(Baseline):
    name = "random_forest"

    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1, class_weight: Optional[str] = "balanced"):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                          min_samples_leaf=min_samples_leaf, class_weight=class_weight)

    def fit(self, X, y, seed=42):
        return _sklearn_fit(self, RandomForestClassifier, X, y, seed)


class CancerGradientBoosting(Baseline):
    name = "gradient_boosting"

    def __init__(self, learning_rate: float = 0.1, max_depth: Optional[int] = None,
                 max_iter: int = 200):
        super().__init__(learning_rate=learning_rate, max_depth=max_depth, max_iter=max_iter)

    def fit(self, X, y, seed=42):
        return _sklearn_fit(self, HistGradientBoostingClassifier, X, y, seed)


class CancerSmallMLP(Baseline):
    name = "small_mlp"

    def __init__(self, hidden_layer_sizes=(32,), alpha: float = 1e-3, max_iter: int = 500):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=max_iter)

    def fit(self, X, y, seed=42):
        return _sklearn_fit(
            self, MLPClassifier, X, y, seed, early_stopping=True, n_iter_no_change=10,
        )


CANCER_BASELINES = {
    "prevalence":        CancerPrevalenceBaseline,
    "logistic":          CancerLogisticRegression,
    "random_forest":     CancerRandomForest,
    "gradient_boosting": CancerGradientBoosting,
    "small_mlp":         CancerSmallMLP,
}


# ─── Hyperparameter search spaces (validation-only, see cross_validation.py) ──

SMOKE_SEARCH_SPACE = {
    "logistic":          {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]},
    "random_forest":      {"n_estimators": [200, 500], "max_depth": [None, 10, 30],
                            "min_samples_leaf": [1, 5, 10], "class_weight": [None, "balanced"]},
    "gradient_boosting":  {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [None, 6, 12]},
    "small_mlp":          {"hidden_layer_sizes": [(16,), (32,), (32, 16)], "alpha": [1e-4, 1e-3, 1e-2]},
}

CANCER_SEARCH_SPACE = SMOKE_SEARCH_SPACE
