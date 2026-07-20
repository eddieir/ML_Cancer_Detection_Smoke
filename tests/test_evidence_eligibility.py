"""
tests/test_evidence_eligibility.py — Phase 7 dataset/task eligibility gates.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.cohort_registry import Cohort
from evidence.eligibility import (
    STATUS_EXPLORATORY,
    STATUS_EXTERNAL,
    STATUS_IMPOSSIBLE,
    STATUS_INTERNAL,
    assess_eligibility,
)


def _cohort(**overrides) -> Cohort:
    base = dict(
        cohort_id="test_cohort", accession="TEST", dataset_id="test_cohort",
        access_level="public", species="human", assay_type="single_cell_rna_seq",
        single_cell_or_bulk="single_cell", subject_identifier_field="subject_id",
        expression_outcome_linkable_at_subject_level=True,
        task_support={
            "smoke_classification": "yes", "malignancy_classification": "no",
            "subject_level_cancer_prediction": "yes", "external_validation": "yes",
        },
        role_eligibility=["development", "internal_test", "external_validation"],
    )
    base.update(overrides)
    return Cohort(**base)


def test_bulk_cohort_impossible_for_smoke_classification():
    cohort = _cohort(single_cell_or_bulk="bulk")
    decision = assess_eligibility(
        task="smoke_classification", cohort=cohort,
        subject_count=1000, per_class_counts={"a": 500, "b": 500},
    )
    assert decision.status == STATUS_IMPOSSIBLE


def test_no_linkage_impossible_for_cancer_prediction():
    cohort = _cohort(expression_outcome_linkable_at_subject_level=False)
    decision = assess_eligibility(
        task="subject_level_cancer_prediction", cohort=cohort,
        subject_count=1000, per_class_counts={"pos": 500, "neg": 500},
    )
    assert decision.status == STATUS_IMPOSSIBLE


def test_controlled_access_cohort_impossible():
    cohort = _cohort(access_level="controlled")
    decision = assess_eligibility(
        task="smoke_classification", cohort=cohort,
        subject_count=1000, per_class_counts={"a": 500, "b": 500},
    )
    assert decision.status == STATUS_IMPOSSIBLE


def test_too_few_subjects_impossible():
    cohort = _cohort()
    decision = assess_eligibility(
        task="smoke_classification", cohort=cohort,
        subject_count=2, per_class_counts={"a": 1, "b": 1},
    )
    assert decision.status == STATUS_IMPOSSIBLE


def test_moderate_subjects_exploratory_only():
    cohort = _cohort()
    decision = assess_eligibility(
        task="smoke_classification", cohort=cohort,
        subject_count=10, per_class_counts={"a": 5, "b": 5},
    )
    assert decision.status == STATUS_EXPLORATORY


def test_large_well_balanced_cohort_eligible_internal():
    cohort = _cohort()
    decision = assess_eligibility(
        task="smoke_classification", cohort=cohort,
        subject_count=50, per_class_counts={"a": 25, "b": 25},
    )
    assert decision.status == STATUS_INTERNAL


def test_external_validation_role_requires_role_eligibility():
    cohort = _cohort(role_eligibility=["development"])  # not eligible for external_validation role
    decision = assess_eligibility(
        task="smoke_classification", cohort=cohort,
        subject_count=50, per_class_counts={"a": 25, "b": 25}, role="external_validation",
    )
    assert decision.status == STATUS_IMPOSSIBLE


def test_external_validation_role_with_sufficient_subjects():
    cohort = _cohort()
    decision = assess_eligibility(
        task="smoke_classification", cohort=cohort,
        subject_count=50, per_class_counts={"a": 25, "b": 25},
        independent_source_count=1, role="external_validation",
    )
    assert decision.status == STATUS_EXTERNAL


def test_task_support_no_is_impossible_regardless_of_counts():
    cohort = _cohort()
    decision = assess_eligibility(
        task="malignancy_classification", cohort=cohort,
        subject_count=10000, per_class_counts={"a": 5000, "b": 5000},
    )
    assert decision.status == STATUS_IMPOSSIBLE


def test_task_support_not_currently_is_impossible():
    cohort = _cohort(task_support={
        "smoke_classification": "not_currently", "malignancy_classification": "no",
        "subject_level_cancer_prediction": "no", "external_validation": "no",
    })
    decision = assess_eligibility(
        task="smoke_classification", cohort=cohort,
        subject_count=10000, per_class_counts={"a": 5000, "b": 5000},
    )
    assert decision.status == STATUS_IMPOSSIBLE
