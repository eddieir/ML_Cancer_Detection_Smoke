"""
tests/test_clinical_readiness.py — Phase 7 clinical-readiness framework.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.clinical_manifest import (
    ClinicalEvidenceManifest,
    ClinicalManifestValidationError,
    sign_manifest,
)
from evidence.clinical_readiness import (
    REQUIRED_DISCLAIMERS,
    ClinicalReadinessNotEstablishedError,
    DimensionAssessment,
    assess_clinical_readiness,
    default_not_started_assessment,
    guard_clinical_claim,
    load_dimension_policy,
)
from evidence.evidence_contract import build_report

POLICY_PATH = Path(__file__).parents[1] / "configs" / "clinical_readiness.yaml"

FAKE_FP = "d" * 64


def test_policy_defines_22_dimensions():
    policy = load_dimension_policy(POLICY_PATH)
    assert len(policy) == 22


def test_current_repository_state_is_not_clinically_ready():
    policy = load_dimension_policy(POLICY_PATH)
    dims = [default_not_started_assessment(d, mandatory, "deadbeef") for d, mandatory in policy.items()]
    report = assess_clinical_readiness(dims, "deadbeef")
    assert report.overall_status == "clinically_not_ready"


def test_all_required_disclaimers_present_in_every_report():
    policy = load_dimension_policy(POLICY_PATH)
    dims = [default_not_started_assessment(d, mandatory, "deadbeef") for d, mandatory in policy.items()]
    report = assess_clinical_readiness(dims, "deadbeef")
    for statement in REQUIRED_DISCLAIMERS:
        assert statement in report.disclaimers


def _mandatory_complete_report():
    policy = load_dimension_policy(POLICY_PATH)
    dims = [
        DimensionAssessment(dimension_id=d, mandatory=m, status="complete",
                             evidence_references=["fabricated_reference_for_test"])
        if m else default_not_started_assessment(d, m, "deadbeef")
        for d, m in policy.items()
    ]
    return assess_clinical_readiness(dims, "deadbeef")


def _report(evidence_level, cohort_role, split_role, frozen_status, development_only, synthetic=False,
            model_fp=FAKE_FP, preprocessing_fp=FAKE_FP):
    identity = {
        "task": "smoke_classification", "endpoint": "e", "endpoint_definition": "e",
        "prediction_unit": "subject", "biological_specimen_type": "lung_tissue",
        "assay_modality": "bulk_microarray", "dataset_accession": "TEST",
        "cohort_role": cohort_role, "species": "human", "sample_count": 10,
        "unique_subject_count": 10, "class_counts": {"a": 5, "b": 5},
        "verified_label_count": 10, "unknown_label_count": 0, "excluded_subject_count": 0,
        "split_role": split_role, "split_manifest_fingerprint": FAKE_FP,
        "dataset_manifest_fingerprint": FAKE_FP, "preprocessing_artifact_fingerprint": preprocessing_fp,
        "model_fingerprint": model_fp, "random_seed": 0, "code_commit_sha": "0" * 40,
        "environment_fingerprint": FAKE_FP, "synthetic_flag": synthetic,
        "development_only_flag": development_only, "frozen_test_access_status": frozen_status,
        "evidence_level": evidence_level, "limitations": ["test fixture"],
        "evaluation_timestamp": "2026-01-01T00:00:00Z", "report_schema_version": "1",
    }
    return build_report(identity, {"macro_f1": 0.5}).to_dict()


def _valid_evidence_reports():
    return {
        "dev_ref": _report("development_only_real_data", "development", "development_holdout",
                            "not_applicable", True),
        "internal_ref": _report("internal_held_out_real_data", "internal_test", "internal_test",
                                 "acquired_once", False),
        "external_ref": _report("external_retrospective_validation", "external_validation",
                                 "external_validation", "not_applicable", False),
        "prospective_ref": _report("prospective_validation", "external_validation",
                                    "external_validation", "not_applicable", False),
        "utility_ref": _report("clinical_utility_validation", "external_validation",
                                "external_validation", "not_applicable", False),
    }


def _valid_manifest(**overrides):
    now = datetime.now(timezone.utc)
    fields = dict(
        schema_version="1", manifest_id="manifest-1",
        created_at=now.isoformat(), expires_at=(now + timedelta(days=1)).isoformat(),
        intended_use="research evaluation only", target_population="test",
        clinical_setting="test", endpoint="e", prediction_horizon="n/a_binary",
        model_fingerprint=FAKE_FP, preprocessing_fingerprint=FAKE_FP,
        label_mapping_fingerprint=FAKE_FP, calibration_fingerprint=FAKE_FP,
        threshold_fingerprint=FAKE_FP,
        development_evidence_refs=["dev_ref"], internal_test_evidence_refs=["internal_ref"],
        external_validation_evidence_refs=["external_ref"], prospective_evidence_refs=["prospective_ref"],
        clinical_utility_evidence_refs=["utility_ref"],
        privacy_security_assessment_ref="psa-1", human_factors_assessment_ref="hfa-1",
        regulatory_review_ref="rr-1", reviewer_name="Test Reviewer", reviewer_role="test_role",
        approved_public_key_id="test-key-1", code_commit_sha="0" * 40,
    )
    fields.update(overrides)
    return ClinicalEvidenceManifest(**fields)


def _signed_manifest_and_keys(**overrides):
    private_key = Ed25519PrivateKey.generate()
    manifest = _valid_manifest(**overrides)
    signed = sign_manifest(manifest, private_key)
    public_bytes = private_key.public_key().public_bytes_raw()
    return signed, {"test-key-1": public_bytes}


def test_cannot_claim_ready_from_synthetic_or_internal_only_evidence():
    policy = load_dimension_policy(POLICY_PATH)
    dims = []
    for dim_id, mandatory in policy.items():
        if mandatory:
            dims.append(DimensionAssessment(
                dimension_id=dim_id, mandatory=True, status="partial",
                blocking_requirements=["only synthetic/internal evidence available"],
            ))
        else:
            dims.append(default_not_started_assessment(dim_id, mandatory, "deadbeef"))
    report = assess_clinical_readiness(dims, "deadbeef")
    assert report.overall_status == "clinically_not_ready"
    signed, keys = _signed_manifest_and_keys()
    with pytest.raises(ClinicalReadinessNotEstablishedError):
        guard_clinical_claim(report, evidence_manifest=signed, approved_public_keys=keys,
                              evidence_reports=_valid_evidence_reports())


def test_guard_no_longer_accepts_a_boolean_argument():
    report = _mandatory_complete_report()
    with pytest.raises(TypeError):
        guard_clinical_claim(report, signed_external_evidence_manifest=True)


def test_guard_rejects_unsigned_manifest():
    report = _mandatory_complete_report()
    manifest = _valid_manifest()  # signature_hex left None
    with pytest.raises(ClinicalManifestValidationError):
        guard_clinical_claim(report, evidence_manifest=manifest, approved_public_keys={},
                              evidence_reports=_valid_evidence_reports())


def test_guard_rejects_unknown_public_key():
    signed, _ = _signed_manifest_and_keys()
    report = _mandatory_complete_report()
    with pytest.raises(ClinicalManifestValidationError):
        guard_clinical_claim(report, evidence_manifest=signed, approved_public_keys={},
                              evidence_reports=_valid_evidence_reports())


def test_guard_rejects_invalid_signature():
    signed, keys = _signed_manifest_and_keys()
    tampered = ClinicalEvidenceManifest.from_dict(signed.to_dict())
    tampered.intended_use = "tampered after signing"
    report = _mandatory_complete_report()
    with pytest.raises(ClinicalManifestValidationError):
        guard_clinical_claim(report, evidence_manifest=tampered, approved_public_keys=keys,
                              evidence_reports=_valid_evidence_reports())


def test_guard_rejects_expired_manifest():
    now = datetime.now(timezone.utc)
    signed, keys = _signed_manifest_and_keys(
        created_at=(now - timedelta(days=10)).isoformat(),
        expires_at=(now - timedelta(days=1)).isoformat(),
    )
    report = _mandatory_complete_report()
    with pytest.raises(ClinicalManifestValidationError):
        guard_clinical_claim(report, evidence_manifest=signed, approved_public_keys=keys,
                              evidence_reports=_valid_evidence_reports())


def test_guard_rejects_synthetic_evidence_reference():
    signed, keys = _signed_manifest_and_keys()
    reports = _valid_evidence_reports()
    reports["dev_ref"] = _report("synthetic_software_validation", "development", "development_train",
                                  "not_applicable", True, synthetic=True)
    report = _mandatory_complete_report()
    with pytest.raises(ClinicalManifestValidationError):
        guard_clinical_claim(report, evidence_manifest=signed, approved_public_keys=keys,
                              evidence_reports=reports)


def test_guard_rejects_internal_split_relabeled_as_external():
    signed, keys = _signed_manifest_and_keys()
    reports = _valid_evidence_reports()
    reports["external_ref"] = _report("development_only_real_data", "development", "development_holdout",
                                       "not_applicable", True)
    report = _mandatory_complete_report()
    with pytest.raises(ClinicalManifestValidationError):
        guard_clinical_claim(report, evidence_manifest=signed, approved_public_keys=keys,
                              evidence_reports=reports)


def test_guard_rejects_mismatched_model_fingerprint():
    signed, keys = _signed_manifest_and_keys()
    reports = _valid_evidence_reports()
    reports["dev_ref"] = _report("development_only_real_data", "development", "development_holdout",
                                  "not_applicable", True, model_fp="e" * 64)
    report = _mandatory_complete_report()
    with pytest.raises(ClinicalManifestValidationError):
        guard_clinical_claim(report, evidence_manifest=signed, approved_public_keys=keys,
                              evidence_reports=reports)


def test_guard_rejects_missing_prospective_evidence():
    signed, keys = _signed_manifest_and_keys(prospective_evidence_refs=[])
    report = _mandatory_complete_report()
    with pytest.raises(ClinicalManifestValidationError):
        guard_clinical_claim(report, evidence_manifest=signed, approved_public_keys=keys,
                              evidence_reports=_valid_evidence_reports())


def test_guard_rejects_missing_utility_evidence():
    signed, keys = _signed_manifest_and_keys(clinical_utility_evidence_refs=[])
    report = _mandatory_complete_report()
    with pytest.raises(ClinicalManifestValidationError):
        guard_clinical_claim(report, evidence_manifest=signed, approved_public_keys=keys,
                              evidence_reports=_valid_evidence_reports())


def test_guard_accepts_valid_signed_manifest_with_valid_evidence():
    signed, keys = _signed_manifest_and_keys()
    report = _mandatory_complete_report()
    guard_clinical_claim(report, evidence_manifest=signed, approved_public_keys=keys,
                          evidence_reports=_valid_evidence_reports())  # does not raise


def test_default_repository_configuration_has_no_approved_public_keys():
    # configs/clinical_readiness.yaml ships no approved-reviewer public-key
    # allow-list — the default repository configuration cannot emit an
    # approval claim regardless of what manifest is supplied.
    with open(POLICY_PATH) as f:
        raw = yaml.safe_load(f) or {}
    assert "approved_reviewer_public_keys" not in raw


def test_dimension_complete_requires_evidence_reference():
    with pytest.raises(ValueError):
        DimensionAssessment(dimension_id="internal_validation", mandatory=True, status="complete")


def test_dimension_incomplete_requires_blocker_or_limitation():
    with pytest.raises(ValueError):
        DimensionAssessment(dimension_id="internal_validation", mandatory=True, status="blocked")


def test_dimension_invalid_status_rejected():
    with pytest.raises(ValueError):
        DimensionAssessment(dimension_id="internal_validation", mandatory=True, status="ready",
                             evidence_references=["x"])


def test_no_external_validation_dimension_means_not_ready():
    policy = load_dimension_policy(POLICY_PATH)
    dims = []
    for dim_id, mandatory in policy.items():
        if dim_id == "external_validation":
            dims.append(default_not_started_assessment(dim_id, mandatory, "deadbeef"))
        elif mandatory:
            dims.append(DimensionAssessment(dimension_id=dim_id, mandatory=True, status="complete",
                                             evidence_references=["ref"]))
        else:
            dims.append(default_not_started_assessment(dim_id, mandatory, "deadbeef"))
    report = assess_clinical_readiness(dims, "deadbeef")
    assert report.overall_status == "clinically_not_ready"
