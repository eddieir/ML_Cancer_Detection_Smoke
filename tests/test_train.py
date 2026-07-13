"""train.py — CellLevelDataset class-weighting and Trainer checkpoint selection."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from constants import N_SMOKE_CLASSES
from train import CellLevelDataset, SubjectLevelDataset, MILEligibilityError, check_mil_eligibility

GENES = 20


def _dataset(smoke_labels: np.ndarray) -> CellLevelDataset:
    n = len(smoke_labels)
    return CellLevelDataset(
        gene_matrix       = np.random.randn(n, GENES).astype("float32"),
        smoke_labels      = smoke_labels,
        malignancy_labels = np.random.randint(0, 2, n).astype("float32"),
        cell_type_ids     = np.zeros(n, dtype="int64"),
    )


def test_smoke_class_weights_balanced_dataset_all_equal():
    labels = np.tile(np.arange(N_SMOKE_CLASSES), 5)   # equal counts per class
    weights = _dataset(labels).smoke_class_weights()
    assert torch.allclose(weights, torch.full((N_SMOKE_CLASSES,), 1.0), atol=1e-5)


def test_smoke_class_weights_favors_minority_classes():
    """
    README's real-data run: cigarette=83, dual_use=30, vape=7, cannabis=6,
    cigar=1, unexposed=10 out of a 6-class label space. The rare classes
    must get strictly larger weight than the majority class, otherwise the
    weighting doesn't do what it's for.
    """
    counts = [83, 7, 1, 6, 30, 10]
    labels = np.concatenate([np.full(c, cls) for cls, c in enumerate(counts)])
    weights = _dataset(labels).smoke_class_weights()

    majority_cls, minority_cls = 0, 2  # cigarette (83) vs cigar (1)
    assert weights[minority_cls] > weights[majority_cls]
    # sanity check against the closed-form balanced-weight formula
    n = sum(counts)
    expected = n / (N_SMOKE_CLASSES * counts[majority_cls])
    assert abs(weights[majority_cls].item() - expected) < 1e-4


def test_smoke_class_weights_zero_for_absent_classes():
    """A class with zero samples in this merge gets weight 0, not inf/nan."""
    labels = np.array([0, 0, 1, 1])  # classes 2..5 entirely absent
    weights = _dataset(labels).smoke_class_weights()
    assert torch.isfinite(weights).all()
    assert (weights[2:] == 0).all()
    assert (weights[:2] > 0).all()


def _bag(subject_id, cancer_label=None, cancer_label_known=False, n=10):
    return {
        "subject_id":         subject_id,
        "gene_matrix":        np.random.randn(n, GENES).astype("float32"),
        "cell_type_ids":      np.zeros(n, dtype="int64"),
        "smoke_labels":       np.zeros(n, dtype="int64"),
        "malig_labels":       np.zeros(n, dtype="float32"),
        "cancer_label":       cancer_label,
        "cancer_label_known": cancer_label_known,
    }


def test_subject_level_dataset_excludes_unknown_outcome_by_default():
    bags = [
        _bag("known_pos", cancer_label=1, cancer_label_known=True),
        _bag("known_neg", cancer_label=0, cancer_label_known=True),
        _bag("unknown",   cancer_label=None, cancer_label_known=False),
    ]
    ds = SubjectLevelDataset(bags)
    assert len(ds) == 2
    subject_ids = {ds[i]["subject_id"] for i in range(len(ds))}
    assert subject_ids == {"known_pos", "known_neg"}
    assert ds.n_excluded_unknown_outcome == 1


def test_subject_level_dataset_can_keep_unknown_outcome_when_requested():
    bags = [
        _bag("known", cancer_label=1, cancer_label_known=True),
        _bag("unknown", cancer_label=None, cancer_label_known=False),
    ]
    ds = SubjectLevelDataset(bags, require_known_outcome=False)
    assert len(ds) == 2


def test_mil_eligibility_rejects_too_few_subjects():
    bags = [_bag(f"s{i}", cancer_label=i % 2, cancer_label_known=True) for i in range(4)]
    ds = SubjectLevelDataset(bags)
    with pytest.raises(MILEligibilityError):
        check_mil_eligibility(ds, min_subjects=10)


def test_mil_eligibility_rejects_single_class():
    bags = [_bag(f"s{i}", cancer_label=1, cancer_label_known=True) for i in range(15)]
    ds = SubjectLevelDataset(bags)
    with pytest.raises(MILEligibilityError):
        check_mil_eligibility(ds, min_subjects=10, min_positive=2, min_negative=2)


def test_mil_eligibility_passes_balanced_sufficient_data():
    bags = [_bag(f"s{i}", cancer_label=i % 2, cancer_label_known=True) for i in range(20)]
    ds = SubjectLevelDataset(bags)
    report = check_mil_eligibility(ds, min_subjects=10, min_positive=2, min_negative=2)
    assert report["n_total"] == 20
    assert report["problems"] == []
