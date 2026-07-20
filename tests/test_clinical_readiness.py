"""
tests/test_clinical_readiness.py — Phase 7 clinical-readiness framework.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.clinical_readiness import (
    REQUIRED_DISCLAIMERS,
    ClinicalReadinessNotEstablishedError,
    DimensionAssessment,
    assess_clinical_readiness,
    default_not_started_assessment,
    guard_clinical_claim,
    load_dimension_policy,
)

POLICY_PATH = Path(__file__).parents[1] / "configs" / "clinical_readiness.yaml"


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
    with pytest.raises(ClinicalReadinessNotEstablishedError):
        guard_clinical_claim(report, signed_external_evidence_manifest=True)


def test_guard_rejects_claim_even_when_mandatory_complete_but_no_signed_manifest():
    policy = load_dimension_policy(POLICY_PATH)
    dims = [
        DimensionAssessment(dimension_id=d, mandatory=m, status="complete",
                             evidence_references=["fabricated_reference_for_test"])
        if m else default_not_started_assessment(d, m, "deadbeef")
        for d, m in policy.items()
    ]
    report = assess_clinical_readiness(dims, "deadbeef")
    assert report.overall_status == "mandatory_dimensions_complete_expert_review_required"
    with pytest.raises(ClinicalReadinessNotEstablishedError):
        guard_clinical_claim(report, signed_external_evidence_manifest=False)


def test_guard_accepts_only_when_both_conditions_hold():
    policy = load_dimension_policy(POLICY_PATH)
    dims = [
        DimensionAssessment(dimension_id=d, mandatory=m, status="complete",
                             evidence_references=["fabricated_reference_for_test"])
        if m else default_not_started_assessment(d, m, "deadbeef")
        for d, m in policy.items()
    ]
    report = assess_clinical_readiness(dims, "deadbeef")
    guard_clinical_claim(report, signed_external_evidence_manifest=True)  # does not raise


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
