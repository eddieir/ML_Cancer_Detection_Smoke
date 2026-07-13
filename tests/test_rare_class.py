"""data/rare_class.py — configurable rare smoke-type-class policies."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import SMOKE_TYPE_MAP
from data.rare_class import apply_rare_class_policy

CIGAR = SMOKE_TYPE_MAP["cigar"]
CIGARETTE = SMOKE_TYPE_MAP["cigarette"]
DUAL_USE = SMOKE_TYPE_MAP["dual_use"]


def _one_rare_subject_dataset():
    # 1 subject (5 cells) with cigar, 20 subjects (100 cells) with cigarette.
    ids = np.array([CIGAR] * 5 + [CIGARETTE] * 100)
    subjects = ["cigar_subject"] * 5 + [f"cig_{i}" for i in range(20) for _ in range(5)]
    return ids, subjects


def test_keep_with_warning_does_not_change_labels():
    ids, subjects = _one_rare_subject_dataset()
    new_ids, keep_mask, report = apply_rare_class_policy(
        ids, subjects, policy="keep_with_warning", target_classes=["cigar"], min_subjects_required=3,
    )
    assert (new_ids == ids).all()
    assert keep_mask.all()
    assert report["affected_classes"]["cigar"]["action"] == "kept_with_warning"
    assert report["affected_classes"]["cigar"]["too_rare"] is True


def test_merge_policy_reassigns_rare_class_to_dual_use():
    ids, subjects = _one_rare_subject_dataset()
    new_ids, keep_mask, report = apply_rare_class_policy(
        ids, subjects, policy="merge_into_dual_use_or_other",
        target_classes=["cigar"], min_subjects_required=3,
    )
    assert (new_ids[:5] == DUAL_USE).all()      # cigar cells reassigned
    assert (new_ids[5:] == CIGARETTE).all()     # untouched otherwise
    assert keep_mask.all()                       # merge never drops cells
    assert report["affected_classes"]["cigar"]["action"] == "merged_into_dual_use"


def test_exclude_policy_drops_rare_class_cells():
    ids, subjects = _one_rare_subject_dataset()
    new_ids, keep_mask, report = apply_rare_class_policy(
        ids, subjects, policy="exclude_from_training_and_evaluation",
        target_classes=["cigar"], min_subjects_required=3,
    )
    assert keep_mask[:5].sum() == 0             # all cigar cells excluded
    assert keep_mask[5:].all()                   # cigarette cells kept
    assert report["affected_classes"]["cigar"]["action"] == "excluded"


def test_original_array_never_mutated():
    ids, subjects = _one_rare_subject_dataset()
    ids_copy = ids.copy()
    apply_rare_class_policy(ids, subjects, policy="merge_into_dual_use_or_other",
                             target_classes=["cigar"], min_subjects_required=3)
    assert (ids == ids_copy).all()  # caller's original array is untouched


def test_class_with_enough_subjects_is_not_affected():
    """A class with >= min_subjects_required subjects is not "too rare" —
    no policy action should be taken even if the policy is exclude/merge."""
    ids = np.array([CIGAR] * 15)
    subjects = [f"s{i}" for i in range(5) for _ in range(3)]  # 5 independent subjects
    new_ids, keep_mask, report = apply_rare_class_policy(
        ids, subjects, policy="exclude_from_training_and_evaluation",
        target_classes=["cigar"], min_subjects_required=3,
    )
    assert keep_mask.all()
    assert report["affected_classes"]["cigar"]["too_rare"] is False
    assert report["affected_classes"]["cigar"]["action"] == "kept"


def test_absent_class_reports_zero_subjects_and_is_not_too_rare():
    ids = np.array([CIGARETTE] * 10)
    subjects = [f"s{i}" for i in range(10)]
    _, keep_mask, report = apply_rare_class_policy(
        ids, subjects, policy="exclude_from_training_and_evaluation",
        target_classes=["cigar"], min_subjects_required=3,
    )
    assert report["affected_classes"]["cigar"]["n_subjects"] == 0
    assert report["affected_classes"]["cigar"]["too_rare"] is False  # 0 present, nothing to exclude
    assert keep_mask.all()


def test_invalid_policy_raises():
    ids, subjects = _one_rare_subject_dataset()
    with pytest.raises(ValueError):
        apply_rare_class_policy(ids, subjects, policy="not_a_real_policy")
