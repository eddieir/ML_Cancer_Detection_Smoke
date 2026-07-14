"""benchmarks/calibration.py — validation-only fitting, frozen once-only test application."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.calibration import build_frozen_policy, fit_calibration, select_threshold


def _binary(seed=0, n=40):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    prob = np.clip(y * 0.6 + rng.rand(n) * 0.4, 0, 1)
    return y, prob


def test_calibration_only_touches_the_arrays_it_is_given():
    y_val, prob_val = _binary(seed=1)
    calibrator = fit_calibration(y_val, prob_val, method="sigmoid")
    assert calibrator.method == "sigmoid"
    calibrated = calibrator.apply(prob_val)
    assert calibrated.shape == prob_val.shape


def test_calibration_not_evaluable_with_too_few_samples():
    y_val, prob_val = _binary(seed=1, n=10)
    calibrator = fit_calibration(y_val, prob_val, method="auto")
    assert calibrator.method == "none"
    assert "NOT_EVALUABLE" in calibrator.reason


def test_calibration_not_evaluable_with_single_class():
    prob_val = np.random.rand(30)
    y_val = np.zeros(30, dtype=int)
    calibrator = fit_calibration(y_val, prob_val)
    assert calibrator.method == "none"


def test_threshold_selected_on_validation_only_then_frozen_and_applied_once():
    y_val, prob_val = _binary(seed=1)
    y_test, prob_test = _binary(seed=2)
    policy = build_frozen_policy(y_val, prob_val, threshold_strategy="youden")
    assert policy.fitted_on == "validation"

    result = policy.apply_to_test(y_test, prob_test)
    assert result["threshold"] == policy.threshold
    with pytest.raises(RuntimeError, match="already ran once"):
        policy.apply_to_test(y_test, prob_test)


def test_undefined_auroc_stays_undefined_never_becomes_half():
    from benchmarks.metrics import cancer_prediction_metrics
    y = np.zeros(10, dtype=int)  # single class
    prob = np.random.rand(10)
    result = cancer_prediction_metrics(y, prob)
    assert result["auroc"] is None
    assert result["auroc_auprc_undefined_reason"] is not None


def test_select_threshold_undefined_with_single_class_falls_back_to_fixed():
    y_val = np.zeros(20, dtype=int)
    prob_val = np.random.rand(20)
    result = select_threshold(y_val, prob_val, strategy="youden")
    assert result["threshold"] == 0.5
    assert result["reason"] is not None
