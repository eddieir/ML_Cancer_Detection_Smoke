"""
evidence/errors.py — typed exceptions for the Phase 7 evidence contract.

Every failure mode below is a distinct, catchable class rather than a bare
ValueError/RuntimeError, so calling code (and tests) can assert on exactly
which contract was violated instead of pattern-matching a message string.
"""


class EvidenceContractError(Exception):
    """Base class for every Phase 7 evidence-contract violation."""


class UnsupportedEvidenceClaimError(EvidenceContractError):
    """Raised when a report claims an evidence level its own identity
    fields (synthetic flag, development-only flag, frozen-test-access
    status, cohort role) do not support — e.g. a synthetic run claiming
    internal_held_out_real_data, or a development-only run claiming
    external_retrospective_validation."""


class MissingOutcomeError(EvidenceContractError):
    """Raised when code attempts to treat a subject with no recorded
    outcome as though a negative/absent outcome had been observed."""


class MissingVerifiedLabelError(EvidenceContractError):
    """Raised when code attempts to score a subject whose label is not a
    verified label (unknown, or weak/proxy without an explicit weak-label
    experiment flag) as part of a primary metric."""


class InsufficientClassSupportError(EvidenceContractError):
    """Raised when a class/outcome has too few subjects to support the
    metric being requested (e.g. AUROC with zero positives)."""


class InsufficientSubjectCountError(EvidenceContractError):
    """Raised when a cohort/split has too few subjects to meet the
    minimum eligibility gate for the task being evaluated."""


class InvalidCohortRoleError(EvidenceContractError):
    """Raised when a cohort is used in a role it is not registered as
    eligible for (e.g. a development cohort used as external validation)."""


class EvidenceIdentityMismatchError(EvidenceContractError):
    """Raised when two artifacts that must share identity (e.g. a model
    fingerprint and the report that describes it) disagree."""


class ExternalValidationRequiredError(EvidenceContractError):
    """Raised when a claim requires external validation evidence that is
    not present in the report being validated."""


class ClinicalReadinessNotEstablishedError(EvidenceContractError):
    """Raised when any code path attempts to emit a clinical-readiness
    claim (e.g. clinical_validated=true, a --clinical-ready flag) without
    an explicit, signed external-evidence manifest satisfying the
    configured requirements. See evidence/clinical_readiness.py."""


class ControlledAccessUnavailableError(EvidenceContractError):
    """Raised when code attempts to evaluate against a controlled-access
    cohort (e.g. NLST) without an authorized local dataset present."""


class OutcomeExpressionLinkageError(EvidenceContractError):
    """Raised when a subject-level cancer-prediction claim requires a
    genuine linkage between expression input and a clinical outcome that
    the cohort registry does not attest exists."""


class EndpointDefinitionError(EvidenceContractError):
    """Raised when a report's endpoint definition is missing required
    fields (time horizon, censoring semantics, event indicator, etc.)."""
