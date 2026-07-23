"""
evidence/evidence_contract.py — the versioned Phase 7 evidence report schema.

Every metric or report this framework produces is either:
  (a) a fully-identified EvidenceReport at one of the seven EVIDENCE_LEVELS
      below, with every REQUIRED_IDENTITY_FIELD populated with a real,
      non-placeholder value, or
  (b) a structured `not_evaluable()` object.

There is no third option. A metric must never be represented as 0, False,
an empty dict, NaN without explanation, or "success with a warning" when
the underlying evidence does not actually exist — see NOT_EVALUABLE_KEYS
and the various ...Error classes in evidence/errors.py that make the
alternative (silently downgrading missing evidence into a fake result)
impossible rather than merely discouraged.

configs/evidence.yaml is the versioned policy this module enforces —
EVIDENCE_LEVELS, REQUIRED_IDENTITY_FIELDS and PLACEHOLDER_VALUES here must
stay in sync with it (test_evidence_contract.py checks that).
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .errors import (
    EvidenceContractError,
    EvidenceIdentityMismatchError,
    UnsupportedEvidenceClaimError,
)
from .run_identity import validate_real_timestamp

REPORT_SCHEMA_VERSION = "1"

# Evidence levels, ordered from weakest to strongest. A report's
# `evidence_level` must be exactly one of these — nothing else is a valid
# claim this framework will accept.
EVIDENCE_LEVELS = (
    "synthetic_software_validation",
    "development_only_real_data",
    "internal_held_out_real_data",
    "external_retrospective_validation",
    "prospective_validation",
    "clinical_utility_validation",
    "regulatory_evidence",
)

# Levels that require synthetic_flag=True (software-validation only, no
# claim about real evidence is permitted).
SYNTHETIC_ONLY_LEVELS = ("synthetic_software_validation",)

# Levels that must never be reachable from a synthetic or development-only
# run — reaching them requires a frozen-test guard having actually been
# acquired (internal) or an external cohort sentinel having actually been
# released after freeze (external+).
REAL_HELD_OUT_LEVELS = (
    "internal_held_out_real_data",
    "external_retrospective_validation",
    "prospective_validation",
    "clinical_utility_validation",
    "regulatory_evidence",
)

EXTERNAL_LEVELS = (
    "external_retrospective_validation",
    "prospective_validation",
    "clinical_utility_validation",
    "regulatory_evidence",
)

VALID_COHORT_ROLES = ("development", "internal_test", "external_validation", "excluded")

# A report's split_role must name what the scored subjects actually were —
# never the generic "train" for a result that is in fact a held-out
# development/internal/external partition (Issue #16 blocker 1). Kept
# deliberately narrow: any code path that cannot honestly name one of
# these is not allowed to build a real evidence report at all.
VALID_SPLIT_ROLES = (
    "development_train",
    "development_validation",
    "development_oof",
    "development_holdout",
    "internal_test",
    "external_validation",
)

REQUIRED_IDENTITY_FIELDS = (
    "task",
    "endpoint",
    "endpoint_definition",
    "prediction_unit",
    "biological_specimen_type",
    "assay_modality",
    "dataset_accession",
    "cohort_role",
    "species",
    "sample_count",
    "unique_subject_count",
    "class_counts",
    "verified_label_count",
    "unknown_label_count",
    "excluded_subject_count",
    "split_role",
    "split_manifest_fingerprint",
    "dataset_manifest_fingerprint",
    "preprocessing_artifact_fingerprint",
    "model_fingerprint",
    "random_seed",
    "code_commit_sha",
    "environment_fingerprint",
    "synthetic_flag",
    "development_only_flag",
    "frozen_test_access_status",
    "evidence_level",
    "limitations",
    "evaluation_timestamp",
    "report_schema_version",
)

# Fields that are legitimately optional depending on task (e.g.
# gene_list_fingerprint only applies to gene-list-based candidates;
# calibration/threshold fingerprints only apply once calibration has run).
OPTIONAL_IDENTITY_FIELDS = (
    "gene_list_fingerprint",
    "calibration_fingerprint",
    "threshold_fingerprint",
)

# Values that look "filled in" but carry no real information — a report
# with any of these in a REQUIRED_IDENTITY_FIELD is rejected exactly as if
# the field were missing.
PLACEHOLDER_VALUES = {
    None, "", "TBD", "TODO", "N/A", "n/a", "unknown", "UNKNOWN",
    "placeholder", "xxx", "TBD_", "FIXME", "...",
}

FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT_FIELDS = (
    "split_manifest_fingerprint",
    "dataset_manifest_fingerprint",
    "preprocessing_artifact_fingerprint",
    "model_fingerprint",
    "environment_fingerprint",
    "gene_list_fingerprint",
    "calibration_fingerprint",
    "threshold_fingerprint",
)

NOT_EVALUABLE_STATUS = "not_evaluable"
REQUIRED_NOT_EVALUABLE_KEYS = ("status", "reason_code", "reason", "required_next_action")


class ReportValidationError(EvidenceContractError):
    """Raised by validate_report() for any structural violation of the
    evidence contract — missing/placeholder identity field, malformed
    fingerprint, or an evidence-level claim unsupported by the report's
    own synthetic/development/frozen-test-access flags."""


def not_evaluable(reason_code: str, reason: str, required_next_action: str,
                   **extra: Any) -> Dict[str, Any]:
    """The one sanctioned way to represent evidence that does not exist.
    Never substitute None, 0, an empty dict, or a metric value here."""
    if not reason_code or not reason or not required_next_action:
        raise ReportValidationError(
            "not_evaluable() requires non-empty reason_code, reason, and "
            "required_next_action — a structured blocked result must explain "
            "itself, not just assert that it is blocked."
        )
    out = {
        "status": NOT_EVALUABLE_STATUS,
        "reason_code": reason_code,
        "reason": reason,
        "required_next_action": required_next_action,
    }
    out.update(extra)
    return out


def is_not_evaluable(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and obj.get("status") == NOT_EVALUABLE_STATUS
        and all(k in obj for k in REQUIRED_NOT_EVALUABLE_KEYS)
    )


def _is_placeholder(value: Any) -> bool:
    if isinstance(value, str) and value.strip() in PLACEHOLDER_VALUES:
        return True
    return value in PLACEHOLDER_VALUES if not isinstance(value, (list, dict)) else False


@dataclass
class EvidenceReport:
    """A fully-identified Phase 7 evidence report. Construct via
    `build_report(...)` (which stamps content_fingerprint) rather than the
    dataclass constructor directly in application code, so the fingerprint
    can never drift from the fields it describes."""

    identity: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    content_fingerprint: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "identity": dict(self.identity),
            "metrics": self.metrics,
            "content_fingerprint": self.content_fingerprint,
        }


def compute_content_fingerprint(identity: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    payload = json.dumps({"identity": identity, "metrics": metrics}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_report(identity: Dict[str, Any], metrics: Dict[str, Any]) -> EvidenceReport:
    validate_identity(identity)
    fp = compute_content_fingerprint(identity, metrics)
    return EvidenceReport(identity=dict(identity), metrics=metrics, content_fingerprint=fp)


def validate_identity(identity: Dict[str, Any]) -> None:
    """Raises ReportValidationError (or a more specific EvidenceContractError
    subclass) if `identity` does not satisfy the Phase 7 identity contract.
    Called by build_report() and independently by validate_report()."""
    missing = [
        f for f in REQUIRED_IDENTITY_FIELDS
        if f not in identity or _is_placeholder(identity.get(f))
    ]
    if missing:
        raise ReportValidationError(
            f"evidence report identity is missing required field(s) or has "
            f"placeholder values for: {missing}"
        )

    level = identity["evidence_level"]
    if level not in EVIDENCE_LEVELS:
        raise ReportValidationError(f"unknown evidence_level {level!r} — must be one of {EVIDENCE_LEVELS}")

    synthetic = bool(identity["synthetic_flag"])
    development_only = bool(identity["development_only_flag"])
    frozen_status = identity["frozen_test_access_status"]
    if frozen_status not in ("not_accessed", "acquired_once", "not_applicable"):
        raise ReportValidationError(
            f"frozen_test_access_status must be one of "
            f"('not_accessed', 'acquired_once', 'not_applicable'), got {frozen_status!r}"
        )

    if synthetic and level not in SYNTHETIC_ONLY_LEVELS:
        raise UnsupportedEvidenceClaimError(
            f"a synthetic_flag=True report cannot claim evidence_level={level!r} — "
            f"synthetic runs are restricted to {SYNTHETIC_ONLY_LEVELS}"
        )
    if not synthetic and level in SYNTHETIC_ONLY_LEVELS:
        raise UnsupportedEvidenceClaimError(
            "a non-synthetic report cannot claim evidence_level="
            "'synthetic_software_validation' — that level requires synthetic_flag=True"
        )

    if level in REAL_HELD_OUT_LEVELS and frozen_status != "acquired_once" and level == "internal_held_out_real_data":
        raise UnsupportedEvidenceClaimError(
            "evidence_level='internal_held_out_real_data' requires "
            "frozen_test_access_status='acquired_once' — a report cannot claim frozen-test "
            "evidence without the frozen-test guard having actually been acquired"
        )
    if level in EXTERNAL_LEVELS and identity.get("cohort_role") != "external_validation":
        raise UnsupportedEvidenceClaimError(
            f"evidence_level={level!r} requires cohort_role='external_validation' — "
            f"got cohort_role={identity.get('cohort_role')!r}. An internal held-out split "
            "from the same cohort as development is never external validation."
        )
    if development_only and level in REAL_HELD_OUT_LEVELS:
        raise UnsupportedEvidenceClaimError(
            f"a development_only_flag=True report cannot claim evidence_level={level!r} — "
            "development-only evidence is exploratory by definition"
        )

    if identity["cohort_role"] not in VALID_COHORT_ROLES:
        raise ReportValidationError(
            f"cohort_role {identity['cohort_role']!r} is not one of {VALID_COHORT_ROLES}"
        )

    if identity["split_role"] not in VALID_SPLIT_ROLES:
        raise ReportValidationError(
            f"split_role {identity['split_role']!r} is not one of {VALID_SPLIT_ROLES} — a report "
            "must name what the scored subjects actually were (e.g. a held-out development "
            "partition must never be recorded as 'train')"
        )

    try:
        validate_real_timestamp(identity["evaluation_timestamp"])
    except ValueError as exc:
        raise ReportValidationError(f"evaluation_timestamp is invalid: {exc}") from exc

    for fp_field in FINGERPRINT_FIELDS:
        if fp_field in identity and identity[fp_field] is not None:
            if not FINGERPRINT_RE.match(str(identity[fp_field])):
                raise ReportValidationError(
                    f"{fp_field} must be a 64-character lowercase hex sha256 digest, "
                    f"got {identity[fp_field]!r}"
                )

    for count_field in ("sample_count", "unique_subject_count", "verified_label_count",
                         "unknown_label_count", "excluded_subject_count"):
        v = identity[count_field]
        if not isinstance(v, int) or v < 0:
            raise ReportValidationError(f"{count_field} must be a non-negative integer, got {v!r}")
    if identity["unique_subject_count"] > identity["sample_count"] and identity["sample_count"] > 0:
        # sample_count counts rows/cells, subject_count counts unique subjects;
        # subjects can never exceed samples/cells in a well-formed report.
        raise ReportValidationError(
            "unique_subject_count cannot exceed sample_count — "
            f"got unique_subject_count={identity['unique_subject_count']}, "
            f"sample_count={identity['sample_count']}"
        )

    if not isinstance(identity["class_counts"], dict):
        raise ReportValidationError("class_counts must be a dict of class -> count")

    if not isinstance(identity["limitations"], list) or not identity["limitations"]:
        raise ReportValidationError(
            "limitations must be a non-empty list — every evidence report must state at "
            "least one limitation; there is no such thing as a limitation-free result here"
        )


def validate_report(report: EvidenceReport) -> None:
    """Full round-trip validation: re-validates identity AND confirms the
    stored content_fingerprint still matches the identity/metrics content
    (catches post-write tampering — see test_evidence_contract.py)."""
    validate_identity(report.identity)
    expected_fp = compute_content_fingerprint(report.identity, report.metrics)
    if report.content_fingerprint != expected_fp:
        raise EvidenceIdentityMismatchError(
            "report.content_fingerprint does not match a fresh hash of its own "
            "identity+metrics — the report was tampered with after being written, or "
            "was constructed without build_report()"
        )


def report_from_dict(data: Dict[str, Any]) -> EvidenceReport:
    return EvidenceReport(
        identity=dict(data.get("identity", {})),
        metrics=data.get("metrics", {}),
        content_fingerprint=data.get("content_fingerprint"),
    )
