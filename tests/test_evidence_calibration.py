"""
tests/test_evidence_calibration.py — Phase 7 Step 14 tests: frozen
calibration/thresholding artifacts and decision-curve analysis.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.calibration import (
    CalibrationPolicyError,
    DevelopmentOOFPredictions,
    FrozenCalibrationThresholdArtifact,
    apply_frozen_calibration_and_threshold,
    build_development_oof_predictions,
    decision_curve_analysis,
    fit_frozen_calibration_and_threshold,
    load_calibration_policy,
)


def _synthetic_oof(n=300, seed=0, prevalence=0.4):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < prevalence).astype(int)
    prob = np.clip(y * 0.6 + rng.rand(n) * 0.4, 0, 1)
    subj = [f"s{i}" for i in range(n)]
    return build_development_oof_predictions(subj, y, prob), y, prob


# ─── DevelopmentOOFPredictions validation ──────────────────────────────────

def test_development_oof_predictions_rejects_empty():
    with pytest.raises(CalibrationPolicyError):
        build_development_oof_predictions([], [], [])


def test_development_oof_predictions_rejects_length_mismatch():
    with pytest.raises(CalibrationPolicyError):
        DevelopmentOOFPredictions(subject_ids=("a", "b"), y_true=(0,), y_prob=(0.1, 0.2))


def test_development_oof_predictions_rejects_duplicate_subjects():
    with pytest.raises(CalibrationPolicyError):
        build_development_oof_predictions(["a", "a"], [0, 1], [0.1, 0.9])


def test_development_oof_predictions_rejects_non_binary_labels():
    with pytest.raises(CalibrationPolicyError):
        build_development_oof_predictions(["a", "b"], [0, 2], [0.1, 0.9])


def test_development_oof_predictions_rejects_out_of_range_probability():
    with pytest.raises(CalibrationPolicyError):
        build_development_oof_predictions(["a", "b"], [0, 1], [0.1, 1.5])


def test_development_oof_predictions_is_frozen():
    oof, _, _ = _synthetic_oof(n=5)
    with pytest.raises(Exception):
        oof.subject_ids = ("x",)


# ─── load_calibration_policy ───────────────────────────────────────────────

def test_load_calibration_policy_from_real_config():
    policy = load_calibration_policy("configs/evidence.yaml")
    assert policy["min_isotonic_n"] >= 200
    assert policy["min_platt_n"] >= 1


def test_load_calibration_policy_falls_back_when_config_missing():
    policy = load_calibration_policy("configs/does_not_exist_evidence.yaml")
    assert policy["min_isotonic_n"] == 200
    assert policy["min_platt_n"] == 20


# ─── fit_frozen_calibration_and_threshold — calibration methods ───────────

def test_fit_requires_development_oof_predictions_instance():
    with pytest.raises(CalibrationPolicyError):
        fit_frozen_calibration_and_threshold({"not": "an instance"})


def test_fit_rejects_unknown_calibration_method():
    oof, _, _ = _synthetic_oof()
    with pytest.raises(CalibrationPolicyError):
        fit_frozen_calibration_and_threshold(oof, calibration_method="bogus")


def test_fit_rejects_unknown_threshold_method():
    oof, _, _ = _synthetic_oof()
    with pytest.raises(CalibrationPolicyError):
        fit_frozen_calibration_and_threshold(oof, threshold_method="bogus")


def test_fit_uncalibrated_is_identity():
    oof, y, prob = _synthetic_oof()
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="uncalibrated", threshold_method="youden")
    assert art.calibration_method == "uncalibrated"
    out = apply_frozen_calibration_and_threshold(art, prob)
    np.testing.assert_allclose(out["y_prob_calibrated"], prob, atol=1e-9)


def test_fit_platt_above_minimum_produces_platt_params():
    oof, _, _ = _synthetic_oof(n=100)
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="platt", threshold_method="youden")
    assert art.calibration_method == "platt"
    assert "coef" in art.calibration_parameters
    assert art.calibration_parameters["coef"] is not None


def test_fit_platt_below_minimum_falls_back_to_uncalibrated():
    oof, _, _ = _synthetic_oof(n=10, prevalence=0.5)
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="platt", threshold_method="youden")
    assert art.calibration_method == "uncalibrated"
    assert "development OOF subjects" in art.calibration_reason


def test_fit_isotonic_above_minimum_produces_isotonic_params():
    oof, _, _ = _synthetic_oof(n=250)
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="isotonic", threshold_method="youden")
    assert art.calibration_method == "isotonic"
    assert "n_thresholds" in art.calibration_parameters


def test_fit_isotonic_below_configured_minimum_falls_back_to_uncalibrated():
    oof, _, _ = _synthetic_oof(n=50)
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="isotonic", threshold_method="youden")
    assert art.calibration_method == "uncalibrated"
    assert "isotonic regression is not offered" in art.calibration_reason


def test_fit_auto_selects_isotonic_when_large_enough():
    oof, _, _ = _synthetic_oof(n=250)
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="auto", threshold_method="youden")
    assert art.calibration_method == "isotonic"


def test_fit_auto_selects_platt_for_medium_n():
    oof, _, _ = _synthetic_oof(n=50)
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="auto", threshold_method="youden")
    assert art.calibration_method == "platt"


def test_fit_auto_selects_uncalibrated_for_small_n():
    oof, _, _ = _synthetic_oof(n=10, prevalence=0.5)
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="auto", threshold_method="youden")
    assert art.calibration_method == "uncalibrated"


# ─── threshold methods ─────────────────────────────────────────────────────

def test_threshold_youden():
    oof, _, _ = _synthetic_oof()
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="uncalibrated", threshold_method="youden")
    assert 0.0 <= art.threshold_value <= 1.0


def test_threshold_balanced_accuracy():
    oof, _, _ = _synthetic_oof()
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="uncalibrated", threshold_method="balanced_accuracy")
    assert 0.0 <= art.threshold_value <= 1.0


def test_threshold_sensitivity_constrained_requires_param():
    oof, _, _ = _synthetic_oof()
    with pytest.raises(CalibrationPolicyError):
        fit_frozen_calibration_and_threshold(oof, calibration_method="uncalibrated", threshold_method="sensitivity_constrained")


def test_threshold_sensitivity_constrained_meets_constraint():
    oof, y, prob = _synthetic_oof(n=400)
    art = fit_frozen_calibration_and_threshold(
        oof, calibration_method="uncalibrated", threshold_method="sensitivity_constrained", min_sensitivity=0.7,
    )
    from benchmarks.metrics import cancer_metrics_at_threshold
    m = cancer_metrics_at_threshold(y, prob, art.threshold_value)
    if m["sensitivity"] is not None and art.threshold_constraints:
        assert m["sensitivity"] >= 0.7 - 1e-9 or "reason" in art.threshold_constraints


def test_threshold_specificity_constrained_requires_param():
    oof, _, _ = _synthetic_oof()
    with pytest.raises(CalibrationPolicyError):
        fit_frozen_calibration_and_threshold(oof, calibration_method="uncalibrated", threshold_method="specificity_constrained")


def test_threshold_specificity_constrained_meets_constraint():
    oof, y, prob = _synthetic_oof(n=400)
    art = fit_frozen_calibration_and_threshold(
        oof, calibration_method="uncalibrated", threshold_method="specificity_constrained", min_specificity=0.7,
    )
    from benchmarks.metrics import cancer_metrics_at_threshold
    m = cancer_metrics_at_threshold(y, prob, art.threshold_value)
    assert m["specificity"] is None or m["specificity"] >= 0.7 - 1e-9


def test_threshold_predeclared_fixed_requires_param():
    oof, _, _ = _synthetic_oof()
    with pytest.raises(CalibrationPolicyError):
        fit_frozen_calibration_and_threshold(oof, calibration_method="uncalibrated", threshold_method="predeclared_fixed")


def test_threshold_predeclared_fixed_uses_exact_value():
    oof, _, _ = _synthetic_oof()
    art = fit_frozen_calibration_and_threshold(
        oof, calibration_method="uncalibrated", threshold_method="predeclared_fixed", fixed_threshold=0.37,
    )
    assert art.threshold_value == 0.37
    assert art.threshold_constraints["fixed_threshold"] == 0.37


# ─── artifact identity / fingerprint / immutability ────────────────────────

def test_artifact_is_frozen_dataclass():
    oof, _, _ = _synthetic_oof(n=10, prevalence=0.5)
    art = fit_frozen_calibration_and_threshold(oof)
    with pytest.raises(Exception):
        art.threshold_value = 0.9


def test_artifact_fingerprint_deterministic_for_same_input():
    oof, _, _ = _synthetic_oof(n=250, seed=7)
    art1 = fit_frozen_calibration_and_threshold(oof, calibration_method="isotonic", threshold_method="youden")
    art2 = fit_frozen_calibration_and_threshold(oof, calibration_method="isotonic", threshold_method="youden")
    assert art1.fingerprint == art2.fingerprint


def test_artifact_fingerprint_differs_for_different_threshold_method():
    oof, _, _ = _synthetic_oof(n=250, seed=7)
    art1 = fit_frozen_calibration_and_threshold(oof, calibration_method="uncalibrated", threshold_method="youden")
    art2 = fit_frozen_calibration_and_threshold(oof, calibration_method="uncalibrated", threshold_method="balanced_accuracy")
    assert art1.fingerprint != art2.fingerprint


def test_artifact_records_selection_subjects_and_fingerprint():
    oof, _, _ = _synthetic_oof(n=20, prevalence=0.5)
    art = fit_frozen_calibration_and_threshold(oof, calibration_method="uncalibrated")
    assert set(art.selection_subject_ids) == set(oof.subject_ids)
    assert art.selection_subject_ids_fingerprint == oof.subject_ids_fingerprint()


# ─── apply_frozen_calibration_and_threshold — no fitting surface ──────────

def test_apply_has_no_fitting_parameters():
    import inspect
    sig = inspect.signature(apply_frozen_calibration_and_threshold)
    params = list(sig.parameters)
    assert params == ["artifact", "y_prob"]
    for forbidden in ("y_true", "refit", "labels"):
        assert forbidden not in sig.parameters


def test_apply_requires_real_artifact():
    with pytest.raises(CalibrationPolicyError):
        apply_frozen_calibration_and_threshold({"not": "an artifact"}, [0.1, 0.9])


def test_apply_produces_decision_at_frozen_threshold():
    oof, _, _ = _synthetic_oof(n=250, seed=3)
    art = fit_frozen_calibration_and_threshold(
        oof, calibration_method="uncalibrated", threshold_method="predeclared_fixed", fixed_threshold=0.5,
    )
    out = apply_frozen_calibration_and_threshold(art, [0.1, 0.6, 0.4, 0.9])
    assert out["y_pred"] == [0, 1, 0, 1]
    assert out["threshold"] == 0.5
    assert out["artifact_fingerprint"] == art.fingerprint


# ─── decision_curve_analysis — exploratory only ────────────────────────────

def test_decision_curve_analysis_labels_itself_exploratory():
    _, y, prob = _synthetic_oof(n=200)
    result = decision_curve_analysis(y, prob)
    assert result["status"] == "exploratory_analysis_only"
    assert "does not demonstrate clinical utility" in result["disclaimer"]


def test_decision_curve_analysis_rejects_empty_input():
    with pytest.raises(CalibrationPolicyError):
        decision_curve_analysis([], [])


def test_decision_curve_analysis_treat_none_is_always_zero():
    _, y, prob = _synthetic_oof(n=100)
    result = decision_curve_analysis(y, prob)
    assert all(row["net_benefit_treat_none"] == 0.0 for row in result["curve"])


def test_decision_curve_analysis_never_appears_in_evidence_contract_fields():
    from evidence.evidence_contract import REQUIRED_IDENTITY_FIELDS, OPTIONAL_IDENTITY_FIELDS
    for field_name in REQUIRED_IDENTITY_FIELDS + OPTIONAL_IDENTITY_FIELDS:
        assert "decision_curve" not in field_name
        assert "net_benefit" not in field_name
