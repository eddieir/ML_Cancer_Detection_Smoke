"""
tests/test_evidence_contract.py — Phase 7 evidence-contract adversarial tests.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.errors import EvidenceContractError, UnsupportedEvidenceClaimError
from evidence.evidence_contract import (
    EVIDENCE_LEVELS,
    REQUIRED_IDENTITY_FIELDS,
    EvidenceReport,
    ReportValidationError,
    build_report,
    compute_content_fingerprint,
    is_not_evaluable,
    not_evaluable,
    report_from_dict,
    validate_identity,
    validate_report,
)

FAKE_FP = "a" * 64


def _base_identity(**overrides) -> dict:
    identity = {
        "task": "smoke_classification",
        "endpoint": "subject_level_smoke_class",
        "endpoint_definition": "verified cigarette/vape/never exposure at time of sampling",
        "prediction_unit": "subject",
        "biological_specimen_type": "lung_tissue",
        "assay_modality": "single_cell_rna_seq",
        "dataset_accession": "GSE288003",
        "cohort_role": "development",
        "species": "mouse",
        "sample_count": 100,
        "unique_subject_count": 10,
        "class_counts": {"cigarette": 5, "never": 5},
        "verified_label_count": 10,
        "unknown_label_count": 0,
        "excluded_subject_count": 0,
        "split_role": "development_train",
        "split_manifest_fingerprint": FAKE_FP,
        "dataset_manifest_fingerprint": FAKE_FP,
        "preprocessing_artifact_fingerprint": FAKE_FP,
        "model_fingerprint": FAKE_FP,
        "random_seed": 0,
        "code_commit_sha": "0" * 40,
        "environment_fingerprint": FAKE_FP,
        "synthetic_flag": True,
        "development_only_flag": True,
        "frozen_test_access_status": "not_applicable",
        "evidence_level": "synthetic_software_validation",
        "limitations": ["synthetic fixture only"],
        "evaluation_timestamp": "2026-07-20T00:00:00Z",
        "report_schema_version": "1",
    }
    identity.update(overrides)
    return identity


def test_valid_synthetic_identity_passes():
    validate_identity(_base_identity())


def test_round_trip_valid_report():
    report = build_report(_base_identity(), {"macro_f1": 0.5})
    validate_report(report)
    reloaded = report_from_dict(report.to_dict())
    validate_report(reloaded)


@pytest.mark.parametrize("field_name", REQUIRED_IDENTITY_FIELDS)
def test_missing_identity_field_rejected(field_name):
    identity = _base_identity()
    del identity[field_name]
    with pytest.raises(EvidenceContractError):
        validate_identity(identity)


@pytest.mark.parametrize("placeholder", ["TBD", "", None, "unknown", "TODO"])
def test_placeholder_identity_value_rejected(placeholder):
    identity = _base_identity(dataset_accession=placeholder)
    with pytest.raises(ReportValidationError):
        validate_identity(identity)


def test_malformed_fingerprint_rejected():
    identity = _base_identity(model_fingerprint="not-a-real-hash")
    with pytest.raises(ReportValidationError):
        validate_identity(identity)


def test_synthetic_report_cannot_claim_real_evidence():
    identity = _base_identity(evidence_level="internal_held_out_real_data")
    with pytest.raises(UnsupportedEvidenceClaimError):
        validate_identity(identity)


def test_nonsynthetic_report_cannot_claim_synthetic_level():
    identity = _base_identity(
        synthetic_flag=False, development_only_flag=False,
        evidence_level="synthetic_software_validation",
        frozen_test_access_status="not_applicable",
    )
    with pytest.raises(UnsupportedEvidenceClaimError):
        validate_identity(identity)


def test_development_only_report_cannot_claim_frozen_test_evidence():
    identity = _base_identity(
        synthetic_flag=False, development_only_flag=True,
        evidence_level="internal_held_out_real_data",
        frozen_test_access_status="acquired_once",
    )
    with pytest.raises(UnsupportedEvidenceClaimError):
        validate_identity(identity)


def test_internal_split_cannot_be_called_external_validation():
    identity = _base_identity(
        synthetic_flag=False, development_only_flag=False,
        evidence_level="external_retrospective_validation",
        frozen_test_access_status="acquired_once",
        cohort_role="internal_test",
    )
    with pytest.raises(UnsupportedEvidenceClaimError):
        validate_identity(identity)


def test_internal_held_out_requires_frozen_guard_acquired():
    identity = _base_identity(
        synthetic_flag=False, development_only_flag=False,
        evidence_level="internal_held_out_real_data",
        frozen_test_access_status="not_accessed",
        cohort_role="internal_test",
    )
    with pytest.raises(UnsupportedEvidenceClaimError):
        validate_identity(identity)


def test_post_write_tampering_detected():
    report = build_report(_base_identity(), {"macro_f1": 0.5})
    report.metrics["macro_f1"] = 0.99  # tamper after construction
    with pytest.raises(EvidenceContractError):
        validate_report(report)


def test_unique_subject_count_cannot_exceed_sample_count():
    identity = _base_identity(unique_subject_count=1000, sample_count=100)
    with pytest.raises(ReportValidationError):
        validate_identity(identity)


def test_empty_limitations_rejected():
    identity = _base_identity(limitations=[])
    with pytest.raises(ReportValidationError):
        validate_identity(identity)


def test_not_evaluable_requires_all_keys():
    with pytest.raises(Exception):
        not_evaluable("", "", "")
    obj = not_evaluable("NO_DATA", "no data present", "download data")
    assert is_not_evaluable(obj)
    assert not is_not_evaluable({"status": "not_evaluable"})  # missing keys
    assert not is_not_evaluable({"macro_f1": 0.0})


def test_evidence_yaml_stays_in_sync_with_code_constants():
    with open(Path(__file__).parents[1] / "configs" / "evidence.yaml") as f:
        policy = yaml.safe_load(f)
    yaml_levels = {e["name"] for e in policy["evidence_levels"]}
    assert yaml_levels == set(EVIDENCE_LEVELS)
    assert set(policy["required_identity_fields"]) == set(REQUIRED_IDENTITY_FIELDS)
