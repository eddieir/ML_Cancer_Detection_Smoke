"""
evidence/external_validation.py — Step 11: a real, testable separation
between development and external-cohort evidence.

Two independent protections live here:

  1. `ExternalCohortSentinel` / `ExternalValidationGate` — a poison-object
     mechanism (following the pattern in benchmarks/sentinel.py) that makes
     it structurally impossible to read external-cohort data before a
     caller has explicitly frozen development (model, preprocessing,
     label-mapping, calibration, and threshold) via
     `ExternalValidationGate.freeze_development(...)`. Any attempt to
     obtain external data from the gate before that call raises
     ExternalValidationRequiredError; a handle obtained before release
     is a sentinel that raises on any real access at all (attribute,
     iteration, indexing, len(), conversion, repr, ...).

  2. `evaluate_external_cohort_eligibility()` / `select_external_validation_cohort()`
     — deterministic legitimacy checks that tell a genuinely independent
     external cohort (no subject overlap with development, registry-
     compatible assay/species/task, registered role_eligibility including
     'external_validation') apart from an internal held-out split of the
     *same* cohort as development, which must never be silently accepted
     as external evidence. These reuse cohort_registry.py's own
     Cohort.supports()/eligible_for_role() rather than re-implementing
     eligibility logic.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

from benchmarks.sentinel import FrozenAccessSentinel

from .cohort_registry import Cohort
from .errors import ExternalValidationRequiredError, InternalSplitMislabeledAsExternalError
from .evidence_contract import not_evaluable

EXTERNAL_ROLE = "external_validation"

REQUIRED_FREEZE_FIELDS = (
    "model_fingerprint",
    "preprocessing_fingerprint",
    "label_mapping_fingerprint",
    "calibration_fingerprint",
    "threshold_fingerprint",
)


class ExternalCohortSentinel(FrozenAccessSentinel):
    """A FrozenAccessSentinel specialization standing in for an external
    cohort's data before ExternalValidationGate.release_external_cohort()
    has actually released it. Every access raises
    ExternalValidationRequiredError (not the generic FrozenDataAccessError
    benchmarks/sentinel.py raises) so callers of this module can catch
    exactly the exception evidence/errors.py documents for this purpose."""

    def _raise(self, action: str):
        raise ExternalValidationRequiredError(
            f"Attempted to {action} on an ExternalCohortSentinel standing in for "
            f"{object.__getattribute__(self, '_sentinel_label')} — external cohort data must "
            "not be read before ExternalValidationGate.freeze_development() has been called and "
            "release_external_cohort() has explicitly released this cohort's data."
        )


@dataclass
class ExternalCohortEligibilityDecision:
    """Result of evaluate_external_cohort_eligibility() for ONE candidate
    cohort. `status` is one of:
        legitimate_external                    — passed every check
        not_registry_eligible                  — cohort registry itself
                                                    does not support this
                                                    task/role combination
        internal_split_mislabeled_as_external   — same cohort_id as
                                                    development, or shares
                                                    subject IDs with it
    """

    cohort_id: str
    status: str
    is_legitimate_external: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "cohort_id": self.cohort_id,
            "status": self.status,
            "is_legitimate_external": self.is_legitimate_external,
            "reason": self.reason,
        }


def evaluate_external_cohort_eligibility(
    candidate: Cohort,
    *,
    task: str,
    development_cohort_ids: Iterable[str],
    development_subject_ids: Iterable[str],
    external_subject_ids: Iterable[str],
) -> ExternalCohortEligibilityDecision:
    """Deterministic legitimacy assessment for ONE candidate external
    cohort. Reuses Cohort.supports()/eligible_for_role() from
    cohort_registry.py rather than re-deriving task/role compatibility.
    Never raises — returns a decision the caller (typically
    ExternalValidationGate.release_external_cohort) decides how to act on."""
    dev_cohort_ids = {str(c) for c in development_cohort_ids}
    dev_subjects = {str(s) for s in development_subject_ids}
    ext_subjects = {str(s) for s in external_subject_ids}

    if not candidate.supports(task) or not candidate.eligible_for_role(EXTERNAL_ROLE):
        return ExternalCohortEligibilityDecision(
            cohort_id=candidate.cohort_id,
            status="not_registry_eligible",
            is_legitimate_external=False,
            reason=(
                f"cohort {candidate.cohort_id!r} is not registered eligible for "
                f"task={task!r} at role={EXTERNAL_ROLE!r} (task_support={candidate.task_support.get(task)!r}, "
                f"role_eligibility={candidate.role_eligibility})"
            ),
        )

    if candidate.cohort_id in dev_cohort_ids:
        return ExternalCohortEligibilityDecision(
            cohort_id=candidate.cohort_id,
            status="internal_split_mislabeled_as_external",
            is_legitimate_external=False,
            reason=(
                f"candidate cohort {candidate.cohort_id!r} is the SAME cohort used for "
                "development — an internal held-out split of the development cohort is never "
                "external validation, regardless of how the split was drawn."
            ),
        )

    overlap = dev_subjects & ext_subjects
    if overlap:
        sample = sorted(overlap)[:5]
        return ExternalCohortEligibilityDecision(
            cohort_id=candidate.cohort_id,
            status="internal_split_mislabeled_as_external",
            is_legitimate_external=False,
            reason=(
                f"{len(overlap)} subject id(s) (e.g. {sample}) appear in both the development "
                f"subject set and cohort {candidate.cohort_id!r}'s candidate external subject "
                "set — a cohort that shares subjects with development is an internal split, "
                "not an independent external cohort."
            ),
        )

    return ExternalCohortEligibilityDecision(
        cohort_id=candidate.cohort_id,
        status="legitimate_external",
        is_legitimate_external=True,
        reason=(
            f"cohort {candidate.cohort_id!r} supports task={task!r} at role={EXTERNAL_ROLE!r}, "
            "is a distinct cohort from development, and shares no subject IDs with the "
            "development subject set."
        ),
    )


def select_external_validation_cohort(
    cohorts: Sequence[Cohort],
    *,
    task: str,
    development_cohort_ids: Iterable[str],
    development_subject_ids: Iterable[str],
    external_subject_ids_by_cohort: Optional[Dict[str, Sequence[str]]] = None,
):
    """Registry-wide scan for a legitimately external cohort for `task`.
    Never opens a dataset. Returns the first
    ExternalCohortEligibilityDecision with is_legitimate_external=True, or
    a structured not_evaluable(reason_code='NO_ELIGIBLE_EXTERNAL_COHORT')
    dict if none exists — this is the expected, honest answer in this
    repository today, since no cohort in configs/cohorts.yaml currently has
    role_eligibility including 'external_validation'."""
    external_subject_ids_by_cohort = external_subject_ids_by_cohort or {}
    dev_cohort_ids = list(development_cohort_ids)
    dev_subjects = list(development_subject_ids)

    registry_eligible = [
        c for c in cohorts if c.supports(task) and c.eligible_for_role(EXTERNAL_ROLE)
    ]
    if not registry_eligible:
        return not_evaluable(
            reason_code="NO_ELIGIBLE_EXTERNAL_COHORT",
            reason=(
                f"No cohort in configs/cohorts.yaml has task_support[{task!r}]='yes' with "
                f"role_eligibility including {EXTERNAL_ROLE!r} — no candidate external cohort "
                "exists at all for this task in this environment."
            ),
            required_next_action=(
                "Register a genuinely independent external cohort in configs/cohorts.yaml with "
                f"role_eligibility including {EXTERNAL_ROLE!r} and task_support[{task!r}]='yes' "
                "once one is actually available, then re-run this selection."
            ),
            task=task,
        )

    decisions = []
    for candidate in registry_eligible:
        decision = evaluate_external_cohort_eligibility(
            candidate,
            task=task,
            development_cohort_ids=dev_cohort_ids,
            development_subject_ids=dev_subjects,
            external_subject_ids=external_subject_ids_by_cohort.get(candidate.cohort_id, []),
        )
        if decision.is_legitimate_external:
            return decision
        decisions.append(decision)

    return not_evaluable(
        reason_code="NO_ELIGIBLE_EXTERNAL_COHORT",
        reason=(
            f"{len(registry_eligible)} cohort(s) are registry-eligible for task={task!r}/"
            f"role={EXTERNAL_ROLE!r}, but every candidate failed the legitimacy check: "
            f"{[d.to_dict() for d in decisions]}"
        ),
        required_next_action=(
            "Provide a candidate cohort that is genuinely distinct from development and shares "
            "no subject IDs with it, or register a new independent external cohort."
        ),
        task=task,
    )


@dataclass
class _PendingExternalCohort:
    cohort: Cohort
    data: object
    subject_ids: frozenset


@dataclass
class ExternalValidationGate:
    """Structural gate between a frozen development state and any external
    cohort's real data.

    Usage:
        gate = ExternalValidationGate()
        handle = gate.register_external_cohort(cohort, real_data, subject_ids)
        # handle is an ExternalCohortSentinel — any real access raises.
        gate.freeze_development(
            model_fingerprint=..., preprocessing_fingerprint=...,
            label_mapping_fingerprint=..., calibration_fingerprint=...,
            threshold_fingerprint=..., development_cohort_ids=[...],
            development_subject_ids=[...],
        )
        real_data = gate.release_external_cohort(cohort.cohort_id, task="...")
    """

    _frozen: bool = field(default=False, init=False, repr=False)
    _freeze_manifest: Optional[Dict] = field(default=None, init=False, repr=False)
    _development_cohort_ids: frozenset = field(default_factory=frozenset, init=False, repr=False)
    _development_subject_ids: frozenset = field(default_factory=frozenset, init=False, repr=False)
    _pending: Dict[str, _PendingExternalCohort] = field(default_factory=dict, init=False, repr=False)
    _released: set = field(default_factory=set, init=False, repr=False)

    def register_external_cohort(self, cohort: Cohort, data: object, subject_ids: Iterable[str]) -> ExternalCohortSentinel:
        """Registers a candidate external cohort's real data internally
        (never returned directly) and hands back an ExternalCohortSentinel
        — the only handle available until release_external_cohort()
        succeeds. Safe to call before or after freeze_development()."""
        self._pending[cohort.cohort_id] = _PendingExternalCohort(
            cohort=cohort, data=data, subject_ids=frozenset(str(s) for s in subject_ids),
        )
        return ExternalCohortSentinel(label=f"external cohort {cohort.cohort_id!r} (not yet released)")

    def freeze_development(
        self,
        *,
        model_fingerprint: str,
        preprocessing_fingerprint: str,
        label_mapping_fingerprint: str,
        calibration_fingerprint: str,
        threshold_fingerprint: str,
        development_cohort_ids: Iterable[str],
        development_subject_ids: Iterable[str],
    ) -> Dict:
        """Marks development (model + preprocessing + label-mapping +
        calibration + threshold) as frozen. Every fingerprint must be a
        non-empty, non-placeholder value — this is the one and only
        precondition release_external_cohort() checks for. Can only be
        called once; a second call raises rather than silently letting
        development be re-frozen (and therefore re-opened) after external
        data may already have been released."""
        if self._frozen:
            raise ExternalValidationRequiredError(
                "freeze_development() was already called for this ExternalValidationGate — "
                "development can only be frozen once. Build a new gate for a new experiment."
            )
        values = {
            "model_fingerprint": model_fingerprint,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "label_mapping_fingerprint": label_mapping_fingerprint,
            "calibration_fingerprint": calibration_fingerprint,
            "threshold_fingerprint": threshold_fingerprint,
        }
        missing = [k for k, v in values.items() if not v]
        if missing:
            raise ExternalValidationRequiredError(
                f"freeze_development() requires every one of {REQUIRED_FREEZE_FIELDS} to be a "
                f"real, non-empty value — missing/empty: {missing}"
            )
        self._development_cohort_ids = frozenset(str(c) for c in development_cohort_ids)
        self._development_subject_ids = frozenset(str(s) for s in development_subject_ids)
        self._freeze_manifest = dict(values)
        self._frozen = True
        return dict(self._freeze_manifest)

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def release_external_cohort(self, cohort_id: str, *, task: str):
        """The one sanctioned way to obtain a registered external cohort's
        real data. Raises ExternalValidationRequiredError if
        freeze_development() has not been called yet, or if no cohort was
        registered under `cohort_id`; raises
        InternalSplitMislabeledAsExternalError if the candidate is not
        legitimately external (same cohort as development, or subject
        overlap) — a distinct exception so callers can tell that failure
        mode apart from "development was never frozen"."""
        if not self._frozen:
            raise ExternalValidationRequiredError(
                "release_external_cohort() was called before freeze_development() — "
                "development (model/preprocessing/label-mapping/calibration/threshold) must be "
                "frozen before any external cohort data may be released for evaluation."
            )
        pending = self._pending.get(cohort_id)
        if pending is None:
            raise ExternalValidationRequiredError(
                f"no external cohort was registered under cohort_id={cohort_id!r} via "
                "register_external_cohort() — nothing to release."
            )
        decision = evaluate_external_cohort_eligibility(
            pending.cohort, task=task,
            development_cohort_ids=self._development_cohort_ids,
            development_subject_ids=self._development_subject_ids,
            external_subject_ids=pending.subject_ids,
        )
        if decision.status == "internal_split_mislabeled_as_external":
            raise InternalSplitMislabeledAsExternalError(decision.reason)
        if not decision.is_legitimate_external:
            raise ExternalValidationRequiredError(decision.reason)
        self._released.add(cohort_id)
        return pending.data

    def is_released(self, cohort_id: str) -> bool:
        return cohort_id in self._released
