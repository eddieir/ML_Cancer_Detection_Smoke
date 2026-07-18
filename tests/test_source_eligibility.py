"""Tests for benchmarks/source_eligibility.py — per-source eligibility
assessment and the source-held-out split manifest."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.source_eligibility import (
    CONTROLLED_ACCESS_UNAVAILABLE,
    ELIGIBLE,
    INSUFFICIENT_CLASSES,
    INSUFFICIENT_OUTCOMES,
    NOT_EVALUABLE,
    SPECIES_MISMATCH,
    assess_cancer_source_eligibility,
    assess_smoke_source_eligibility,
    build_source_held_out_manifest,
)


def test_smoke_source_eligible_with_enough_subjects_and_classes():
    r = assess_smoke_source_eligibility(
        "sourceA", [0, 1, 2, 0, 1], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    assert r.eligible and r.status == ELIGIBLE


def test_smoke_source_not_evaluable_too_few_subjects():
    r = assess_smoke_source_eligibility(
        "sourceA", [0, 1], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    assert not r.eligible and r.status == NOT_EVALUABLE


def test_smoke_source_insufficient_classes():
    r = assess_smoke_source_eligibility(
        "sourceA", [0, 0, 0, 0], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    assert not r.eligible and r.status == INSUFFICIENT_CLASSES


def test_smoke_source_species_mismatch_never_defaults_to_compatible():
    r = assess_smoke_source_eligibility("mouseSrc", [0, 1, 2], species_by_source={}, reference_species="human")
    assert r.status == SPECIES_MISMATCH
    r2 = assess_smoke_source_eligibility(
        "mouseSrc", [0, 1, 2], species_by_source={"mouseSrc": "mouse"}, reference_species="human",
    )
    assert r2.status == SPECIES_MISMATCH


def test_smoke_source_controlled_access_unavailable():
    r = assess_smoke_source_eligibility(
        "restricted", [0, 1, 2], species_by_source={"restricted": "human"}, reference_species="human",
        controlled_access_sources=["restricted"],
    )
    assert r.status == CONTROLLED_ACCESS_UNAVAILABLE
    assert not r.eligible


def test_cancer_source_eligible_with_both_classes():
    r = assess_cancer_source_eligibility(
        "sourceA", [1, 0, 1, 0, 1], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    assert r.eligible and r.status == ELIGIBLE


def test_cancer_source_unknown_outcomes_never_coerced_to_negative():
    r = assess_cancer_source_eligibility(
        "sourceA", [1, None, None, 0, 1, None], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    assert r.counts["n_known_outcome"] == 3
    assert r.counts["n_subjects_total"] == 6


def test_cancer_source_one_class_marked_insufficient_outcomes_but_still_eligible():
    r = assess_cancer_source_eligibility(
        "sourceA", [1, 1, 1, 1, 1], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    assert r.status == INSUFFICIENT_OUTCOMES
    assert r.eligible is True  # not fully excluded — only AUROC/AUPRC undefined


def test_cancer_source_not_evaluable_too_few_known_outcomes():
    r = assess_cancer_source_eligibility(
        "sourceA", [1, 0], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    assert r.status == NOT_EVALUABLE and not r.eligible


def test_manifest_fingerprint_deterministic():
    elig = assess_cancer_source_eligibility(
        "sourceA", [1, 0, 1, 0], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    kwargs = dict(
        task="cancer_prediction", held_out_source="sourceA", development_sources=["sourceB", "sourceC"],
        development_subjects=["b1", "b2", "c1"], held_out_subjects=["a1", "a2"],
        known_label_counts={"n": 2}, class_distribution={}, eligibility=elig, seed=42,
    )
    m1 = build_source_held_out_manifest(**kwargs)
    m2 = build_source_held_out_manifest(**kwargs)
    assert m1.fingerprint() == m2.fingerprint()


def test_manifest_fingerprint_changes_with_held_out_source():
    elig = assess_cancer_source_eligibility(
        "sourceA", [1, 0], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    m1 = build_source_held_out_manifest(
        task="cancer_prediction", held_out_source="sourceA", development_sources=["sourceB"],
        development_subjects=["b1", "b2"], held_out_subjects=["a1", "a2"],
        known_label_counts={}, class_distribution={}, eligibility=elig, seed=1,
    )
    m2 = build_source_held_out_manifest(
        task="cancer_prediction", held_out_source="sourceB", development_sources=["sourceA"],
        development_subjects=["a1", "a2"], held_out_subjects=["b1", "b2"],
        known_label_counts={}, class_distribution={}, eligibility=elig, seed=1,
    )
    assert m1.fingerprint() != m2.fingerprint()


def test_manifest_fingerprint_changes_with_subject_membership():
    elig = assess_cancer_source_eligibility(
        "sourceA", [1, 0], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    base = dict(
        task="cancer_prediction", held_out_source="sourceA", development_sources=["sourceB"],
        held_out_subjects=["a1", "a2"], known_label_counts={}, class_distribution={}, eligibility=elig, seed=1,
    )
    m1 = build_source_held_out_manifest(development_subjects=["b1", "b2"], **base)
    m2 = build_source_held_out_manifest(development_subjects=["b1", "b3"], **base)
    assert m1.fingerprint() != m2.fingerprint()


def test_manifest_rejects_development_held_out_overlap():
    elig = assess_cancer_source_eligibility(
        "sourceA", [1, 0], species_by_source={"sourceA": "human"}, reference_species="human",
    )
    with pytest.raises(ValueError):
        build_source_held_out_manifest(
            task="cancer_prediction", held_out_source="sourceA", development_sources=["sourceB"],
            development_subjects=["a1", "b1"], held_out_subjects=["a1", "a2"],
            known_label_counts={}, class_distribution={}, eligibility=elig, seed=1,
        )
