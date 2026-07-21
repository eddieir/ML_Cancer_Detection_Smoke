"""
tests/test_evidence_tracks.py — Phase 7 evaluation-track (Step 5) tests.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.cohort_registry import load_cohort_registry
from evidence.evidence_contract import is_not_evaluable, validate_report, report_from_dict
from evidence import tracks

COHORTS_YAML = Path(__file__).parents[1] / "configs" / "cohorts.yaml"


@pytest.fixture(scope="module")
def cohorts():
    return load_cohort_registry(str(COHORTS_YAML))


# ─── against-registry paths: must all be honest not_evaluable in this repo ─

def test_track_a_against_registry_is_not_evaluable(cohorts):
    result = tracks.run_track_a_against_registry(cohorts)
    assert is_not_evaluable(result)
    assert result["reason_code"] == "NO_ELIGIBLE_COHORT"
    assert result["task"] == tracks.TASK_SMOKE


def test_track_b_against_registry_is_not_evaluable(cohorts):
    result = tracks.run_track_b_against_registry(cohorts)
    assert is_not_evaluable(result)
    assert result["reason_code"] == "NO_ELIGIBLE_COHORT"
    assert result["task"] == tracks.TASK_MALIGNANCY


def test_track_c_against_registry_is_not_evaluable(cohorts):
    result = tracks.run_track_c_against_registry(cohorts)
    assert is_not_evaluable(result)
    assert result["reason_code"] == "NO_ELIGIBLE_EXPRESSION_OUTCOME_LINKAGE"
    assert result["task"] == tracks.TASK_CANCER_PREDICTION


def test_track_b_against_registry_never_accepts_bulk_tcga_as_single_cell(cohorts):
    # TCGA-LUAD/LUSC carry a sample_type field but must never satisfy Track B,
    # which requires a single-cell-compatible malignancy label.
    result = tracks.run_track_b_against_registry(cohorts)
    assert is_not_evaluable(result)
    assert "TCGA" in result["reason"] or "bulk" in result["reason"].lower()


def test_eligible_cohorts_for_track_returns_empty_for_unsupported_task(cohorts):
    assert tracks.eligible_cohorts_for_track(cohorts, "not_a_real_task") == []


# ─── fixture paths: real scoring code, explicitly synthetic ────────────────

def test_track_a_on_fixture_produces_valid_synthetic_report():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 0, 2, 2]
    subject_ids = [f"s{i}" for i in range(6)]
    out = tracks.run_track_a_on_fixture(y_true, y_pred, subject_ids, num_classes=3)
    validate_report(report_from_dict(out))
    assert out["identity"]["synthetic_flag"] is True
    assert out["identity"]["evidence_level"] == "synthetic_software_validation"
    assert out["identity"]["unique_subject_count"] == 6
    assert out["metrics"]["primary_metric"] == "macro_f1"
    assert "macro_f1" in out["metrics"]


def test_track_a_on_fixture_weak_label_experiment_is_stamped_and_zero_verified():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    subject_ids = [f"s{i}" for i in range(4)]
    out = tracks.run_track_a_on_fixture(
        y_true, y_pred, subject_ids, num_classes=2, weak_label_experiment=True,
    )
    assert out["identity"]["verified_label_count"] == 0
    assert out["metrics"]["weak_label_experiment"] is True
    assert any("weak_label_experiment" in lim for lim in out["identity"]["limitations"])
    assert out["identity"]["endpoint"].endswith("weak_label_sensitivity")


def test_track_a_on_fixture_flags_classes_below_support_threshold():
    # Class "1" has a single subject, well under MIN_SUBJECT_SUPPORT_FOR_LEARNABILITY_CLAIM.
    y_true = [0, 0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 1, 1]
    subject_ids = [f"s{i}" for i in range(6)]
    out = tracks.run_track_a_on_fixture(y_true, y_pred, subject_ids, num_classes=2)
    assert out["metrics"]["classes_below_support_threshold"]
    assert any("support threshold" in lim for lim in out["identity"]["limitations"])


def test_track_b_on_fixture_produces_valid_synthetic_report():
    y_true = [0, 1, 1, 0, 1, 0]
    y_prob = [0.1, 0.8, 0.6, 0.3, 0.9, 0.2]
    sample_ids = [f"sample{i}" for i in range(6)]
    out = tracks.run_track_b_on_fixture(y_true, y_prob, sample_ids)
    validate_report(report_from_dict(out))
    assert out["identity"]["synthetic_flag"] is True
    assert out["identity"]["prediction_unit"] == "sample"
    assert "calibration_curve" in out["metrics"]


def test_track_c_on_fixture_binary_fixed_horizon_only():
    y_true = [0, 1, 1, 0, 1]
    y_prob = [0.2, 0.7, 0.9, 0.1, 0.6]
    subject_ids = [f"subj{i}" for i in range(5)]
    out = tracks.run_track_c_on_fixture(y_true, y_prob, subject_ids)
    validate_report(report_from_dict(out))
    assert out["identity"]["prediction_unit"] == "subject"
    assert out["metrics"]["endpoint_fields"]["censoring_status"] == "not_applicable_binary_fixed_horizon"
    # No survival/time-to-event metric is fabricated for a fixture with no
    # genuine follow-up/censoring semantics.
    assert "c_index" not in out["metrics"]
    assert "integrated_brier_score" not in out["metrics"]


def test_track_c_on_fixture_endpoint_definition_states_horizon_and_field():
    out = tracks.run_track_c_on_fixture(
        [0, 1], [0.3, 0.7], ["a", "b"],
        event_indicator_field="malignancy_dx", prediction_horizon_years=2.0,
    )
    assert "2.0" in out["identity"]["endpoint_definition"]
    assert "malignancy_dx" in out["identity"]["endpoint_definition"]
