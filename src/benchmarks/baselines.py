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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def positive_class_proba(model: Baseline, X: np.ndarray) -> np.ndarray:
    """
    P(class 1), robust to a training fold that only ever saw one class — a
    real, expected outcome of small grouped-CV folds (see
    check_task_b_eligibility / MILEligibilityError), not a bug to crash on.
    sklearn's predict_proba() only has as many columns as classes_ has
    entries, in classes_'s (sorted) order — blindly indexing column 1
    assumes both classes were seen AND that class 1 sorts second, which is
    usually true for {0, 1} labels but not guaranteed and outright wrong
    with one class. Returns all-zeros if the model never saw class 1,
    all-ones if it never saw class 0.
    """
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    if 0 not in classes:
        return np.ones(len(X), dtype=float)
    return proba[:, classes.index(1)]


def _sklearn_fit(baseline: Baseline, model_cls, X, y, seed, **extra_params):
    if len(set(y.tolist())) < 2:
        # A fold/subset with only one training class isn't a bug — it's an
        # expected outcome of small grouped-CV folds — but LogisticRegression/
        # HistGradientBoostingClassifier/MLPClassifier all raise ValueError
        # outright on single-class y. Fall back to a constant predictor for
        # the one class actually seen, recorded explicitly so this is never
        # confused with the real model having been fit.
        baseline.model = DummyClassifier(strategy="constant", constant=y[0])
        baseline.model.fit(X, y)
        baseline.classes_ = baseline.model.classes_
        baseline.hyperparams = dict(baseline.hyperparams, fallback="single_class_constant")
        return baseline
    params = dict(baseline.hyperparams)
    params.update(extra_params)
    if "random_state" in model_cls().get_params():
        params["random_state"] = seed
    baseline.model = model_cls(**params)
    baseline.model.fit(X, y)
    baseline.classes_ = baseline.model.classes_
    return baseline


def _scaled_sklearn_fit(baseline: Baseline, model_cls, X, y, seed, **extra_params):
    """
    Like _sklearn_fit, but wraps the estimator in a Pipeline(StandardScaler,
    model) fit as one unit — the scaler's mean/variance are computed only
    from whatever X this call receives (the caller's fold-train rows, never
    validation/test; see cross_validation.py), so per-fold scaling is
    automatic here, not something callers must remember to do separately.
    Used for models sensitive to feature magnitude (logistic regression,
    small MLP) where an unscaled `n_cells` feature could otherwise dominate
    purely by scale. Tree-based models (random forest, gradient boosting)
    don't need this and aren't wrapped.
    """
    if len(set(y.tolist())) < 2:
        baseline.model = DummyClassifier(strategy="constant", constant=y[0])
        baseline.model.fit(X, y)
        baseline.classes_ = baseline.model.classes_
        baseline.hyperparams = dict(baseline.hyperparams, fallback="single_class_constant")
        return baseline
    params = dict(baseline.hyperparams)
    params.update(extra_params)
    if "random_state" in model_cls().get_params():
        params["random_state"] = seed
    # MLPClassifier's early_stopping=True carves an internal validation split
    # out of whatever X/y it's given — with very few training rows (a small
    # grouped-CV fold or OOD held-out-source split, a real and expected
    # occurrence, not a bug) that internal split can end up empty and
    # MLPClassifier raises outright. Disabling early_stopping for small
    # inputs is a training-stability fallback, not a leakage change: it
    # doesn't touch what data the model is fit or evaluated on, only
    # whether it also carves an early-stopping split out of its own
    # training rows.
    if params.get("early_stopping") and len(y) < 20:
        params = dict(params, early_stopping=False)
        baseline.hyperparams = dict(baseline.hyperparams, early_stopping_disabled_small_n=True)
    baseline.model = Pipeline([("scaler", StandardScaler()), ("model", model_cls(**params))])
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
        return _scaled_sklearn_fit(
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
        return _scaled_sklearn_fit(
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
        return _scaled_sklearn_fit(self, lambda **p: LogisticRegression(max_iter=2000, **p), X, y, seed)


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
        return _scaled_sklearn_fit(
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
