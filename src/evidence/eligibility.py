"""
evidence/eligibility.py — dataset/task eligibility gates for the three
Phase 7 evaluation tracks (Step 6).

These are deterministic, config-driven gates over cohort-level facts
(subject counts, label/event counts, linkage, assay/species compatibility)
— they never open a raw dataset file. Thresholds live in
configs/clinical_readiness.yaml under `eligibility_thresholds` and are
documented as engineering/scientific guardrails, not regulatory
requirements.

Each gate returns one of five statuses:
  IMPOSSIBLE                    — cohort/task combination can never be evaluable (e.g. species/assay mismatch)
  EXPLORATORY_DEVELOPMENT_ONLY  — evaluable, but only as development-only exploratory evidence
  ELIGIBLE_INTERNAL_HELD_OUT    — meets the bar for an internal held-out evaluation
  ELIGIBLE_EXTERNAL_VALIDATION  — meets the bar for external retrospective validation
  INSUFFICIENT_FOR_CLINICAL_CLAIMS — never independently returned; clinical-readiness
                                      claims additionally require evidence this module cannot produce
                                      (see evidence/clinical_readiness.py)
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .cohort_registry import Cohort

STATUS_IMPOSSIBLE = "IMPOSSIBLE"
STATUS_EXPLORATORY = "EXPLORATORY_DEVELOPMENT_ONLY"
STATUS_INTERNAL = "ELIGIBLE_INTERNAL_HELD_OUT"
STATUS_EXTERNAL = "ELIGIBLE_EXTERNAL_VALIDATION"
STATUS_INSUFFICIENT_CLINICAL = "INSUFFICIENT_FOR_CLINICAL_CLAIMS"

VALID_STATUSES = (
    STATUS_IMPOSSIBLE, STATUS_EXPLORATORY, STATUS_INTERNAL,
    STATUS_EXTERNAL, STATUS_INSUFFICIENT_CLINICAL,
)

# Minimum subject counts (engineering guardrails, not regulatory
# thresholds — see configs/clinical_readiness.yaml).
MIN_SUBJECTS_EXPLORATORY = 6
MIN_SUBJECTS_INTERNAL = 20
MIN_SUBJECTS_EXTERNAL = 20
MIN_PER_CLASS_EXPLORATORY = 2
MIN_PER_CLASS_INTERNAL = 5
MIN_INDEPENDENT_SOURCES_EXTERNAL = 1


@dataclass
class EligibilityDecision:
    task: str
    cohort_id: str
    status: str
    reasons: List[str] = field(default_factory=list)
    counts: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task, "cohort_id": self.cohort_id, "status": self.status,
            "reasons": self.reasons, "counts": self.counts,
        }


def _base_compatibility_reasons(cohort: Cohort, task: str) -> List[str]:
    """Structural (never-evaluable) incompatibilities — species/assay/unit
    mismatches that no amount of additional data could fix."""
    reasons = []
    if task == "smoke_classification" and cohort.single_cell_or_bulk != "single_cell":
        reasons.append(
            f"cohort {cohort.cohort_id!r} is {cohort.single_cell_or_bulk!r}, not single_cell — "
            "the smoke-classification track requires single-cell/MIL-compatible input"
        )
    if task == "malignancy_classification" and cohort.single_cell_or_bulk != "single_cell":
        reasons.append(
            f"cohort {cohort.cohort_id!r} is {cohort.single_cell_or_bulk!r} — bulk tumour/normal "
            "labels cannot be translated into per-cell malignancy labels"
        )
    if task == "subject_level_cancer_prediction" and not cohort.expression_outcome_linkable_at_subject_level:
        reasons.append(
            f"cohort {cohort.cohort_id!r} has no attested subject-level linkage between "
            "expression input and a cancer outcome"
        )
    if cohort.task_support.get(task) == "no":
        reasons.append(f"cohort registry marks task_support[{task!r}]='no' for {cohort.cohort_id!r}")
    return reasons


def assess_eligibility(
    task: str,
    cohort: Cohort,
    *,
    subject_count: int,
    per_class_counts: Dict[str, int],
    independent_source_count: int = 1,
    role: str = "development",
) -> EligibilityDecision:
    """Deterministic eligibility assessment for one (task, cohort) pair.
    Never opens data; every count must be supplied by the caller from
    already-computed, real provenance (typically the audit CLI)."""
    counts = {
        "subject_count": subject_count,
        "per_class_counts": dict(per_class_counts),
        "independent_source_count": independent_source_count,
        "role": role,
    }

    structural_reasons = _base_compatibility_reasons(cohort, task)
    if cohort.access_level == "controlled":
        structural_reasons.append(
            f"cohort {cohort.cohort_id!r} is controlled-access and no authorized local dataset "
            "was verified — eligibility cannot be assessed against unverified/absent data"
        )
    if cohort.task_support.get(task) == "not_currently":
        structural_reasons.append(
            f"cohort registry marks task_support[{task!r}]='not_currently' for {cohort.cohort_id!r} "
            "— no supported pipeline exists yet for this cohort/task combination"
        )
    if structural_reasons:
        return EligibilityDecision(task=task, cohort_id=cohort.cohort_id,
                                    status=STATUS_IMPOSSIBLE, reasons=structural_reasons, counts=counts)

    if not cohort.eligible_for_role(role):
        return EligibilityDecision(
            task=task, cohort_id=cohort.cohort_id, status=STATUS_IMPOSSIBLE,
            reasons=[f"cohort {cohort.cohort_id!r} is not registered eligible for role={role!r} "
                     f"(role_eligibility={cohort.role_eligibility})"],
            counts=counts,
        )

    reasons = []
    min_class_count = min(per_class_counts.values()) if per_class_counts else 0

    if role == "external_validation":
        if subject_count < MIN_SUBJECTS_EXTERNAL:
            reasons.append(f"only {subject_count} subjects (< {MIN_SUBJECTS_EXTERNAL} required for external validation)")
        if independent_source_count < MIN_INDEPENDENT_SOURCES_EXTERNAL:
            reasons.append("no independent source verified for external validation")
        if reasons:
            return EligibilityDecision(task=task, cohort_id=cohort.cohort_id,
                                        status=STATUS_EXPLORATORY, reasons=reasons, counts=counts)
        return EligibilityDecision(task=task, cohort_id=cohort.cohort_id,
                                    status=STATUS_EXTERNAL, reasons=[], counts=counts)

    if subject_count < MIN_SUBJECTS_EXPLORATORY:
        reasons.append(f"only {subject_count} subjects (< {MIN_SUBJECTS_EXPLORATORY} minimum for exploratory evidence)")
        return EligibilityDecision(task=task, cohort_id=cohort.cohort_id,
                                    status=STATUS_IMPOSSIBLE, reasons=reasons, counts=counts)

    if subject_count < MIN_SUBJECTS_INTERNAL or min_class_count < MIN_PER_CLASS_INTERNAL:
        reasons.append(
            f"subject_count={subject_count} / min_per_class={min_class_count} below internal held-out "
            f"thresholds (subjects>={MIN_SUBJECTS_INTERNAL}, per_class>={MIN_PER_CLASS_INTERNAL}) — "
            "eligible only as exploratory development evidence"
        )
        return EligibilityDecision(task=task, cohort_id=cohort.cohort_id,
                                    status=STATUS_EXPLORATORY, reasons=reasons, counts=counts)

    return EligibilityDecision(task=task, cohort_id=cohort.cohort_id,
                                status=STATUS_INTERNAL, reasons=[], counts=counts)
