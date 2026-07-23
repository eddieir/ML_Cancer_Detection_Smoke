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
FAKE_FP = "b" * 64
FAKE_FP2 = "c" * 64


def _real_identity_kwargs():
    return dict(
        split_manifest_fingerprint=FAKE_FP,
        model_fingerprint=FAKE_FP,
        environment_fingerprint=FAKE_FP,
        preprocessing_artifact_fingerprint=FAKE_FP,
    )


@pytest.fixture(scope="module")
def cohorts():
    return load_cohort_registry(str(COHORTS_YAML))


# ─── against-registry paths: must all be honest not_evaluable in this repo ─

def test_track_a_against_registry_is_not_evaluable(cohorts):
    # gse123352 is now task_support['smoke_classification']='yes' with
    # role_eligibility=[development] (a real bulk pipeline exists — see
    # data/bulk_pipeline.py) — so this generic path finds it eligible, but
    # its own contract (never open a dataset file) still means it returns
    # not_evaluable(REAL_FITTING_NOT_IMPLEMENTED) rather than a real result.
    result = tracks.run_track_a_against_registry(cohorts)
    assert is_not_evaluable(result)
    assert result["reason_code"] == "REAL_FITTING_NOT_IMPLEMENTED"
    assert result["task"] == tracks.TASK_SMOKE
    assert "gse123352" in result["candidate_cohorts"]


def test_track_a_against_registry_never_opens_a_file_even_when_eligible(cohorts):
    # Regression guard for the documented contract in
    # run_track_a_against_registry's docstring: eligibility alone must never
    # produce a real metric — only the dedicated real-data functions do that.
    result = tracks.run_track_a_against_registry(cohorts)
    assert "macro_f1" not in result
    assert "content_fingerprint" not in result


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


# ─── real-data Track A paths (Phase 7 round 2) ────────────────────────────

def _tiny_real_file(tmp_path, name="raw_fixture.txt", content=b"real-bytes-not-a-placeholder"):
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_track_a_real_weak_label_data_is_stamped_correctly(tmp_path):
    raw_file = _tiny_real_file(tmp_path)
    y_true = [0, 1, 1, 0, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]
    subject_ids = [f"donor{i}" for i in range(6)]
    out = tracks.run_copd_control_proxy_analysis(
        y_true, y_pred, subject_ids, num_classes=2, raw_file_paths=[raw_file],
        **_real_identity_kwargs(),
    )
    validate_report(report_from_dict(out))
    assert out["identity"]["synthetic_flag"] is False
    assert out["identity"]["development_only_flag"] is True
    assert out["identity"]["evidence_level"] == "development_only_real_data"
    assert out["identity"]["dataset_accession"] == "GSE136831"
    assert out["identity"]["verified_label_count"] == 0
    assert out["identity"]["task"] == tracks.TASK_PROXY_ANALYSIS
    assert out["identity"]["task"] != tracks.TASK_SMOKE
    assert out["identity"]["split_role"] == "development_holdout"
    assert out["metrics"]["weak_label_experiment"] is True
    assert any("proxy analysis" in lim for lim in out["identity"]["limitations"])
    # real sha256 fingerprint of the actual file content, not a placeholder
    assert len(out["identity"]["dataset_manifest_fingerprint"]) == 64


def test_copd_control_proxy_analysis_missing_raw_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        tracks.run_copd_control_proxy_analysis(
            [0, 1], [0, 1], ["a", "b"], num_classes=2, raw_file_paths=[missing],
            **_real_identity_kwargs(),
        )


def test_copd_control_proxy_analysis_cannot_be_selected_as_smoke_evidence(cohorts):
    # the proxy analysis's task is not TASK_SMOKE, so it can never appear
    # among cohorts eligible for the verified smoke_classification task —
    # there is no caller flag that promotes it.
    eligible = tracks.eligible_cohorts_for_track(cohorts, tracks.TASK_PROXY_ANALYSIS, "development")
    assert eligible == []


def test_track_a_real_verified_label_data_is_stamped_correctly(tmp_path):
    raw_file = _tiny_real_file(tmp_path, name="series_matrix_fixture.txt")
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 1, 0, 0, 0]
    subject_ids = [f"GSM{i}" for i in range(6)]
    out = tracks.run_track_a_on_real_verified_label_data(
        y_true, y_pred, subject_ids, num_classes=2, raw_file_paths=[raw_file],
        **_real_identity_kwargs(),
    )
    validate_report(report_from_dict(out))
    assert out["identity"]["synthetic_flag"] is False
    assert out["identity"]["development_only_flag"] is True
    assert out["identity"]["evidence_level"] == "development_only_real_data"
    assert out["identity"]["dataset_accession"] == "GSE123352"
    assert out["identity"]["verified_label_count"] == len(y_true)
    assert out["metrics"]["weak_label_experiment"] is False
    assert out["identity"]["assay_modality"] == "bulk_microarray"
    assert out["identity"]["split_role"] == "development_holdout"
    assert "lifetime ever-versus-never" in out["identity"]["endpoint_definition"]


def test_track_a_real_data_fingerprint_changes_with_file_content(tmp_path):
    file_a = _tiny_real_file(tmp_path, name="a.txt", content=b"content-one")
    file_b = _tiny_real_file(tmp_path, name="b.txt", content=b"content-two")
    y_true, y_pred, subject_ids = [0, 1], [0, 1], ["s0", "s1"]
    out_a = tracks.run_track_a_on_real_verified_label_data(
        y_true, y_pred, subject_ids, num_classes=2, raw_file_paths=[file_a],
        **_real_identity_kwargs(),
    )
    out_b = tracks.run_track_a_on_real_verified_label_data(
        y_true, y_pred, subject_ids, num_classes=2, raw_file_paths=[file_b],
        **_real_identity_kwargs(),
    )
    assert out_a["identity"]["dataset_manifest_fingerprint"] != out_b["identity"]["dataset_manifest_fingerprint"]


def test_track_a_real_verified_label_data_requires_real_fingerprints(tmp_path):
    raw_file = _tiny_real_file(tmp_path, name="series_matrix_fixture.txt")
    with pytest.raises(TypeError):
        tracks.run_track_a_on_real_verified_label_data(
            [0, 1], [0, 1], ["s0", "s1"], num_classes=2, raw_file_paths=[raw_file],
        )
