"""
evidence/clinical_readiness.py — Step 12: staged clinical-readiness
assessment across 22 dimensions.

This module never produces a single pass/fail boolean. It produces a
per-dimension status (not_started/blocked/partial/complete) plus an
overall result that can only be "not clinically ready" unless every
mandatory dimension (configs/clinical_readiness.yaml) is independently
"complete" with real evidence_references — there is no code path in this
module that weakens that requirement to manufacture a passing status.

Emitting any clinical-readiness claim (a --clinical-ready CLI flag,
clinical_validated=true, or similar) without an explicit, signed external
evidence manifest satisfying the configured mandatory dimensions raises
ClinicalReadinessNotEstablishedError. There is no "fake certificate" path
here: assess_clinical_readiness() always computes the real status from
the dimension records it is given, never from a caller-supplied override.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from .errors import ClinicalReadinessNotEstablishedError

VALID_STATUSES = ("not_started", "blocked", "partial", "complete")

REQUIRED_DISCLAIMERS = (
    "This is research software.",
    "This is not a medical device.",
    "Not validated for diagnosis, screening, prognosis, or treatment.",
    "A model probability is not an individual clinical risk estimate unless calibration "
    "and target-population validation establish that.",
    "Retrospective public-dataset performance cannot establish clinical utility.",
    "Prospective and independent external validation remain necessary.",
    "Regulatory readiness requires expert legal/regulatory review outside this repository.",
)


@dataclass
class DimensionAssessment:
    dimension_id: str
    mandatory: bool
    status: str = "not_started"
    evidence_references: List[str] = field(default_factory=list)
    blocking_requirements: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    responsible_evidence_level: Optional[str] = None
    last_assessed_commit: Optional[str] = None

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"dimension {self.dimension_id!r}: invalid status {self.status!r}")
        if self.status == "complete" and not self.evidence_references:
            raise ValueError(
                f"dimension {self.dimension_id!r}: status='complete' requires at least one "
                "evidence_references entry — a completion claim without a cited reference is rejected"
            )
        if self.status != "complete" and not self.blocking_requirements and not self.limitations:
            raise ValueError(
                f"dimension {self.dimension_id!r}: status={self.status!r} must record at least "
                "one blocking_requirements or limitations entry explaining why it is not complete"
            )

    def to_dict(self) -> dict:
        return {
            "dimension_id": self.dimension_id,
            "mandatory": self.mandatory,
            "status": self.status,
            "evidence_references": self.evidence_references,
            "blocking_requirements": self.blocking_requirements,
            "limitations": self.limitations,
            "responsible_evidence_level": self.responsible_evidence_level,
            "last_assessed_commit": self.last_assessed_commit,
        }


def load_dimension_policy(path: str) -> Dict[str, bool]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {d["id"]: bool(d.get("mandatory", False)) for d in raw.get("dimensions", [])}


def default_not_started_assessment(dimension_id: str, mandatory: bool, commit_sha: str) -> DimensionAssessment:
    return DimensionAssessment(
        dimension_id=dimension_id,
        mandatory=mandatory,
        status="not_started",
        blocking_requirements=[
            "No evidence has been assembled for this dimension in this repository."
        ],
        last_assessed_commit=commit_sha,
    )


@dataclass
class ClinicalReadinessReport:
    dimensions: List[DimensionAssessment]
    overall_status: str
    disclaimers: List[str]
    commit_sha: str

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "disclaimers": self.disclaimers,
            "commit_sha": self.commit_sha,
        }


def assess_clinical_readiness(dimensions: List[DimensionAssessment], commit_sha: str) -> ClinicalReadinessReport:
    """Computes overall_status purely from the supplied dimension records
    — there is no parameter that lets a caller assert readiness directly.
    overall_status is 'clinically_not_ready' unless every mandatory
    dimension is status='complete', in which case it is still only
    'mandatory_dimensions_complete_expert_review_required' — this module
    never emits an unqualified "ready" status, because regulatory/clinical
    sign-off is explicitly out of scope for this repository."""
    mandatory = [d for d in dimensions if d.mandatory]
    incomplete_mandatory = [d.dimension_id for d in mandatory if d.status != "complete"]

    if incomplete_mandatory:
        overall = "clinically_not_ready"
    else:
        overall = "mandatory_dimensions_complete_expert_review_required"

    return ClinicalReadinessReport(
        dimensions=dimensions,
        overall_status=overall,
        disclaimers=list(REQUIRED_DISCLAIMERS),
        commit_sha=commit_sha,
    )


def guard_clinical_claim(
    report: ClinicalReadinessReport, *,
    evidence_manifest: "ClinicalEvidenceManifest",
    approved_public_keys: Dict[str, bytes],
    evidence_reports: Dict[str, dict],
) -> None:
    """The single choke point every CLI/report code path MUST call before
    emitting any clinical-readiness claim (a --clinical-ready flag,
    clinical_validated=true, a model-card "ready for clinical use"
    statement, etc). Raises unless BOTH:
      (a) the assessed overall_status has actually reached the
          (still-qualified) complete state, and
      (b) `evidence_manifest` is a real, unexpired
          evidence/clinical_manifest.ClinicalEvidenceManifest whose
          signature verifies against `approved_public_keys` and whose
          referenced evidence is independently validated against
          `evidence_reports` (see clinical_manifest.py).

    There is no boolean parameter here — a caller cannot satisfy this
    function by passing True. `approved_public_keys` must come from the
    caller's own configuration; this repository's default configuration
    ships none (see configs/clinical_readiness.yaml), so by default this
    function is unsatisfiable regardless of what manifest is supplied."""
    from .clinical_manifest import validate_clinical_evidence_manifest

    if report.overall_status != "mandatory_dimensions_complete_expert_review_required":
        raise ClinicalReadinessNotEstablishedError(
            f"cannot emit a clinical-readiness claim: overall_status={report.overall_status!r} — "
            "mandatory dimensions are not all complete"
        )
    validate_clinical_evidence_manifest(
        evidence_manifest, approved_public_keys=approved_public_keys, evidence_reports=evidence_reports,
    )
