"""
benchmarks/calibration.py — validation-only calibration and threshold
selection for Task B cancer probabilities, frozen and applied to test data
exactly once.

Rules enforced here (see runner.py for where these are called):
  * fit_calibration / select_threshold see ONLY validation (or nested
    out-of-fold) predictions — never test.
  * The resulting FrozenThresholdPolicy can be applied to test data exactly
    once (apply_to_test raises on a second call) — mirrors the
    once-only-test-evaluation discipline train.py's final_test_evaluation
    already enforces for the neural model.
  * Too few validation samples -> calibration is skipped (method="none",
    NOT_EVALUABLE) rather than fit unstably.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .metrics import cancer_metrics_at_threshold, cancer_prediction_metrics

MIN_SIGMOID_N = 20
MIN_ISOTONIC_N = 200


class FrozenCalibrator:
    def __init__(self, method: str, model: Optional[object] = None, reason: Optional[str] = None):
        self.method = method
        self.model = model
        self.reason = reason

    def apply(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if self.method == "none":
            return probs
        if self.method == "sigmoid":
            return self.model.predict_proba(probs.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.model.predict(probs)
        raise ValueError(f"Unknown calibration method {self.method!r}")

    def to_dict(self) -> Dict:
        d = {"method": self.method, "reason": self.reason}
        if self.method == "sigmoid" and self.model is not None:
            d["coef"] = float(self.model.coef_.ravel()[0])
            d["intercept"] = float(self.model.intercept_.ravel()[0])
        if self.method == "isotonic" and self.model is not None:
            d["n_thresholds"] = int(len(self.model.X_thresholds_))
        return d


def fit_calibration(y_val: np.ndarray, prob_val: np.ndarray, method: str = "auto") -> FrozenCalibrator:
    y_val = np.asarray(y_val)
    prob_val = np.asarray(prob_val, dtype=float)
    n = len(y_val)

    if len(set(y_val.tolist())) < 2:
        return FrozenCalibrator("none", reason=f"validation set has only one class (n={n}) — calibration NOT_EVALUABLE")

    if method == "auto":
        if n >= MIN_ISOTONIC_N:
            method = "isotonic"
        elif n >= MIN_SIGMOID_N:
            method = "sigmoid"
        else:
            return FrozenCalibrator("none", reason=f"only {n} validation samples (< {MIN_SIGMOID_N}) — calibration NOT_EVALUABLE")

    if method == "sigmoid":
        lr = LogisticRegression()
        lr.fit(prob_val.reshape(-1, 1), y_val)
        return FrozenCalibrator("sigmoid", model=lr)
    if method == "isotonic":
        if n < MIN_ISOTONIC_N:
            return FrozenCalibrator(
                "none", reason=f"only {n} validation samples (< {MIN_ISOTONIC_N} required for isotonic) — falling back skipped, calibration NOT_EVALUABLE",
            )
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(prob_val, y_val)
        return FrozenCalibrator("isotonic", model=iso)
    if method == "none":
        return FrozenCalibrator("none")
    raise ValueError(f"Unknown calibration method {method!r}")


def _threshold_grid(prob: np.ndarray) -> np.ndarray:
    return np.unique(np.clip(prob, 1e-6, 1 - 1e-6))


def select_threshold(
    y_val: np.ndarray, prob_val: np.ndarray, strategy: str = "youden",
    min_sensitivity: Optional[float] = None,
) -> Dict:
    """
    All strategies search validation-only. Returns
    {"threshold", "strategy", "criterion_value", "reason"} — reason is set
    (and threshold falls back to 0.5) only when the strategy is undefined
    for this validation set (e.g. a single class present).
    """
    y_val = np.asarray(y_val)
    prob_val = np.asarray(prob_val, dtype=float)

    if strategy == "fixed":
        return {"threshold": 0.5, "strategy": "fixed", "criterion_value": None, "reason": None}

    if len(set(y_val.tolist())) < 2:
        return {"threshold": 0.5, "strategy": strategy, "criterion_value": None,
                "reason": "validation set has only one class — threshold selection NOT_EVALUABLE, using fixed 0.5"}

    grid = _threshold_grid(prob_val)
    best_t, best_score = 0.5, -np.inf

    for t in grid:
        m = cancer_metrics_at_threshold(y_val, prob_val, t)
        if m["sensitivity"] is None:
            continue
        if strategy == "youden":
            score = m["sensitivity"] + m["specificity"] - 1
        elif strategy == "balanced_accuracy":
            score = m["balanced_accuracy"]
        elif strategy == "sensitivity_constrained":
            if min_sensitivity is None:
                raise ValueError("sensitivity_constrained requires min_sensitivity")
            if m["sensitivity"] < min_sensitivity:
                continue
            score = m["specificity"]
        else:
            raise ValueError(f"Unknown threshold strategy {strategy!r}")
        if score > best_score:
            best_score, best_t = score, float(t)

    if best_score == -np.inf:
        return {"threshold": 0.5, "strategy": strategy, "criterion_value": None,
                "reason": "no threshold in the validation grid satisfied the strategy's constraint"}
    return {"threshold": best_t, "strategy": strategy, "criterion_value": float(best_score), "reason": None}


@dataclass
class FrozenThresholdPolicy:
    calibrator: FrozenCalibrator
    threshold: float
    threshold_strategy: str
    threshold_reason: Optional[str] = None
    fitted_on: str = "validation"
    _used: bool = field(default=False, repr=False)

    def to_dict(self) -> Dict:
        return {
            "calibration": self.calibrator.to_dict(), "threshold": self.threshold,
            "threshold_strategy": self.threshold_strategy, "threshold_reason": self.threshold_reason,
            "fitted_on": self.fitted_on,
        }

    def apply_to_test(self, y_test: np.ndarray, prob_test_raw: np.ndarray) -> Dict:
        """
        The one sanctioned place a frozen calibration+threshold touches test
        data. Raises if called twice on the same policy — a second "final"
        test evaluation is not final, and re-running it after seeing the
        first result is exactly the kind of test-data peeking this framework
        forbids.
        """
        if self._used:
            raise RuntimeError(
                "FrozenThresholdPolicy.apply_to_test already ran once for this policy — "
                "refitting/reselecting to try again would use test data for model/threshold "
                "selection. Build a new policy from validation data for a new experiment."
            )
        self._used = True
        calibrated = self.calibrator.apply(prob_test_raw)
        result = cancer_prediction_metrics(y_test, calibrated, threshold=self.threshold)
        result["policy"] = self.to_dict()
        return result


def build_frozen_policy(
    y_val: np.ndarray, prob_val_raw: np.ndarray,
    calibration_method: str = "auto", threshold_strategy: str = "youden",
    min_sensitivity: Optional[float] = None,
) -> FrozenThresholdPolicy:
    calibrator = fit_calibration(y_val, prob_val_raw, method=calibration_method)
    prob_val_calibrated = calibrator.apply(prob_val_raw)
    thr = select_threshold(y_val, prob_val_calibrated, strategy=threshold_strategy, min_sensitivity=min_sensitivity)
    return FrozenThresholdPolicy(
        calibrator=calibrator, threshold=thr["threshold"], threshold_strategy=threshold_strategy,
        threshold_reason=thr["reason"],
    )
