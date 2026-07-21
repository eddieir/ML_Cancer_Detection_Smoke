"""
tests/test_external_validation.py — Phase 7 Step 11 tests: the
ExternalValidationGate sentinel mechanism and the internal-split-vs-
external legitimacy checks in evidence/external_validation.py.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.cohort_registry import Cohort, load_cohort_registry
from evidence.errors import (
    ExternalValidationRequiredError,
    InternalSplitMislabeledAsExternalError,
)
from evidence.evidence_contract import is_not_evaluable
from evidence.external_validation import (
    ExternalCohortSentinel,
    ExternalValidationGate,
    evaluate_external_cohort_eligibility,
    select_external_validation_cohort,
)

COHORTS_YAML = Path(__file__).parents[1] / "configs" / "cohorts.yaml"


def _cohort(cohort_id="ext_cohort", role_eligibility=("external_validation",), task_yes=True):
    return Cohort(
        cohort_id=cohort_id, accession="ACC", dataset_id=cohort_id, access_level="public",
        species="human", assay_type="single_cell_rna_seq", single_cell_or_bulk="single_cell",
        subject_identifier_field="subject_id",
        expression_outcome_linkable_at_subject_level=True,
        task_support={
            "smoke_classification": "yes" if task_yes else "no",
            "malignancy_classification": "no",
            "subject_level_cancer_prediction": "no",
            "external_validation": "no",
        },
        role_eligibility=list(role_eligibility),
    )


@pytest.fixture(scope="module")
def real_cohorts():
    return load_cohort_registry(str(COHORTS_YAML))


# ─── ExternalCohortSentinel — poison-object behavior ──────────────────────

def test_sentinel_raises_on_attribute_access():
    s = ExternalCohortSentinel(label="x")
    with pytest.raises(ExternalValidationRequiredError):
        s.some_attribute


def test_sentinel_raises_on_indexing():
    s = ExternalCohortSentinel(label="x")
    with pytest.raises(ExternalValidationRequiredError):
        s[0]


def test_sentinel_raises_on_iteration():
    s = ExternalCohortSentinel(label="x")
    with pytest.raises(ExternalValidationRequiredError):
        list(s)


def test_sentinel_raises_on_len():
    s = ExternalCohortSentinel(label="x")
    with pytest.raises(ExternalValidationRequiredError):
        len(s)


def test_sentinel_raises_on_bool_conversion():
    s = ExternalCohortSentinel(label="x")
    with pytest.raises(ExternalValidationRequiredError):
        bool(s)


def test_sentinel_raises_on_repr():
    s = ExternalCohortSentinel(label="x")
    with pytest.raises(ExternalValidationRequiredError):
        repr(s)


def test_sentinel_raises_on_numpy_array_conversion():
    np = pytest.importorskip("numpy")
    s = ExternalCohortSentinel(label="x")
    with pytest.raises(ExternalValidationRequiredError):
        np.asarray(s)


def test_sentinel_never_leaks_wrapped_value_via_dict():
    s = ExternalCohortSentinel(label="secret label only")
    # __dict__ access itself goes through __getattribute__ (not overridden)
    # but real test-data values are never stored on the sentinel at all —
    # only the label string is, and only for the error message.
    assert "value" not in vars(s)


# ─── ExternalValidationGate — structural freeze-before-release ────────────

def test_release_before_freeze_raises():
    gate = ExternalValidationGate()
    cohort = _cohort()
    gate.register_external_cohort(cohort, data={"real": "data"}, subject_ids=["s1", "s2"])
    with pytest.raises(ExternalValidationRequiredError):
        gate.release_external_cohort(cohort.cohort_id, task="smoke_classification")


def test_register_returns_sentinel_not_real_data():
    gate = ExternalValidationGate()
    cohort = _cohort()
    handle = gate.register_external_cohort(cohort, data={"real": "data"}, subject_ids=["s1"])
    assert isinstance(handle, ExternalCohortSentinel)
    with pytest.raises(ExternalValidationRequiredError):
        handle["real"]


def _frozen_gate():
    gate = ExternalValidationGate()
    gate.freeze_development(
        model_fingerprint="m" * 64, preprocessing_fingerprint="p" * 64,
        label_mapping_fingerprint="l" * 64, calibration_fingerprint="c" * 64,
        threshold_fingerprint="t" * 64,
        development_cohort_ids=["dev_cohort"], development_subject_ids=["d1", "d2"],
    )
    return gate


def test_freeze_development_requires_every_fingerprint():
    gate = ExternalValidationGate()
    with pytest.raises(ExternalValidationRequiredError):
        gate.freeze_development(
            model_fingerprint="", preprocessing_fingerprint="p" * 64,
            label_mapping_fingerprint="l" * 64, calibration_fingerprint="c" * 64,
            threshold_fingerprint="t" * 64,
            development_cohort_ids=["dev"], development_subject_ids=["d1"],
        )


def test_freeze_development_cannot_be_called_twice():
    gate = _frozen_gate()
    with pytest.raises(ExternalValidationRequiredError):
        gate.freeze_development(
            model_fingerprint="m" * 64, preprocessing_fingerprint="p" * 64,
            label_mapping_fingerprint="l" * 64, calibration_fingerprint="c" * 64,
            threshold_fingerprint="t" * 64,
            development_cohort_ids=["dev_cohort"], development_subject_ids=["d1"],
        )


def test_release_after_freeze_with_legitimate_cohort_returns_real_data():
    gate = _frozen_gate()
    cohort = _cohort(cohort_id="ext_cohort")
    real_data = {"real": "external data"}
    gate.register_external_cohort(cohort, data=real_data, subject_ids=["e1", "e2"])
    released = gate.release_external_cohort(cohort.cohort_id, task="smoke_classification")
    assert released is real_data
    assert gate.is_released(cohort.cohort_id)


def test_release_same_cohort_id_as_development_raises_mislabeled_error():
    gate = _frozen_gate()
    same_as_dev = _cohort(cohort_id="dev_cohort")
    gate.register_external_cohort(same_as_dev, data={"x": 1}, subject_ids=["e1"])
    with pytest.raises(InternalSplitMislabeledAsExternalError):
        gate.release_external_cohort("dev_cohort", task="smoke_classification")


def test_release_with_subject_overlap_raises_mislabeled_error():
    gate = _frozen_gate()
    cohort = _cohort(cohort_id="ext_cohort")
    # d1 is in the development subject set frozen above.
    gate.register_external_cohort(cohort, data={"x": 1}, subject_ids=["d1", "e2"])
    with pytest.raises(InternalSplitMislabeledAsExternalError):
        gate.release_external_cohort(cohort.cohort_id, task="smoke_classification")


def test_release_unregistered_cohort_raises():
    gate = _frozen_gate()
    with pytest.raises(ExternalValidationRequiredError):
        gate.release_external_cohort("never_registered", task="smoke_classification")


def test_release_not_registry_eligible_cohort_raises():
    gate = _frozen_gate()
    ineligible = _cohort(cohort_id="ineligible", role_eligibility=("development",))
    gate.register_external_cohort(ineligible, data={"x": 1}, subject_ids=["e1"])
    with pytest.raises(ExternalValidationRequiredError):
        gate.release_external_cohort("ineligible", task="smoke_classification")


# ─── evaluate_external_cohort_eligibility ──────────────────────────────────

def test_evaluate_legitimate_external_cohort():
    candidate = _cohort(cohort_id="ext")
    decision = evaluate_external_cohort_eligibility(
        candidate, task="smoke_classification",
        development_cohort_ids=["dev"], development_subject_ids=["d1", "d2"],
        external_subject_ids=["e1", "e2"],
    )
    assert decision.is_legitimate_external
    assert decision.status == "legitimate_external"


def test_evaluate_same_cohort_id_is_mislabeled():
    candidate = _cohort(cohort_id="dev")
    decision = evaluate_external_cohort_eligibility(
        candidate, task="smoke_classification",
        development_cohort_ids=["dev"], development_subject_ids=["d1"],
        external_subject_ids=["d1"],
    )
    assert not decision.is_legitimate_external
    assert decision.status == "internal_split_mislabeled_as_external"


def test_evaluate_subject_overlap_is_mislabeled():
    candidate = _cohort(cohort_id="ext")
    decision = evaluate_external_cohort_eligibility(
        candidate, task="smoke_classification",
        development_cohort_ids=["dev"], development_subject_ids=["d1", "d2"],
        external_subject_ids=["d2", "e9"],
    )
    assert not decision.is_legitimate_external
    assert decision.status == "internal_split_mislabeled_as_external"


def test_evaluate_registry_ineligible_cohort():
    candidate = _cohort(cohort_id="ext", role_eligibility=("development",))
    decision = evaluate_external_cohort_eligibility(
        candidate, task="smoke_classification",
        development_cohort_ids=["dev"], development_subject_ids=["d1"],
        external_subject_ids=["e1"],
    )
    assert not decision.is_legitimate_external
    assert decision.status == "not_registry_eligible"


def test_evaluate_task_not_supported():
    candidate = _cohort(cohort_id="ext", task_yes=False)
    decision = evaluate_external_cohort_eligibility(
        candidate, task="smoke_classification",
        development_cohort_ids=["dev"], development_subject_ids=["d1"],
        external_subject_ids=["e1"],
    )
    assert not decision.is_legitimate_external
    assert decision.status == "not_registry_eligible"


# ─── select_external_validation_cohort — registry-wide scan ───────────────

def test_select_returns_not_evaluable_when_registry_has_no_external_cohort(real_cohorts):
    # As of this policy version, no cohort in configs/cohorts.yaml has
    # role_eligibility including 'external_validation' — this is the
    # expected, honest state, not a bug.
    result = select_external_validation_cohort(
        real_cohorts, task="smoke_classification",
        development_cohort_ids=["gse136831"], development_subject_ids=["d1"],
    )
    assert is_not_evaluable(result)
    assert result["reason_code"] == "NO_ELIGIBLE_EXTERNAL_COHORT"


def test_select_returns_legitimate_decision_when_one_exists():
    cohorts = [_cohort(cohort_id="ext1")]
    decision = select_external_validation_cohort(
        cohorts, task="smoke_classification",
        development_cohort_ids=["dev"], development_subject_ids=["d1"],
        external_subject_ids_by_cohort={"ext1": ["e1", "e2"]},
    )
    assert decision.is_legitimate_external
    assert decision.cohort_id == "ext1"


def test_select_rejects_when_only_candidate_overlaps_development():
    cohorts = [_cohort(cohort_id="ext1")]
    result = select_external_validation_cohort(
        cohorts, task="smoke_classification",
        development_cohort_ids=["dev"], development_subject_ids=["d1"],
        external_subject_ids_by_cohort={"ext1": ["d1"]},
    )
    assert is_not_evaluable(result)
    assert result["reason_code"] == "NO_ELIGIBLE_EXTERNAL_COHORT"
    assert "internal_split_mislabeled_as_external" in result["reason"]


def test_no_real_cohort_in_registry_currently_eligible_for_external_role(real_cohorts):
    eligible = [c for c in real_cohorts if c.eligible_for_role("external_validation")]
    assert eligible == []
