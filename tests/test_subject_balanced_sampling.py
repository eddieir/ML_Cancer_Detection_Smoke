"""
Tests for data/sampling.py — the class -> subject -> cell subject-balanced
batch sampler (Phase 2 subject-aware class-imbalance correction).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.sampling import (
    SamplingConfigurationError,
    SamplingImpossibleError,
    SubjectBalancedBatchSampler,
    SubjectClassIndex,
    class_selection_probabilities,
    resolve_smoke_imbalance_config,
    subject_selection_probabilities,
)


def _dataset(spec, num_classes=2):
    """spec: list of (subject_id, label, n_cells) -> (subject_ids, labels) arrays."""
    subject_ids, labels = [], []
    for sid, label, n in spec:
        subject_ids += [sid] * n
        labels += [label] * n
    return np.array(subject_ids, dtype=object), np.array(labels, dtype=np.int64)


IMBALANCED_SPEC = [
    ("A", 0, 5000),   # class 0: huge subject
    ("B", 0, 50),     # class 0: tiny subject
    *[(f"C{i}", 1, 50) for i in range(20)],  # class 1: 20 subjects
]


# ─── 1. Subject mapping built from unique training subjects ───────────────────

def test_subject_mapping_built_from_unique_subjects():
    subj, labels = _dataset([("A", 0, 10), ("A", 0, 5), ("B", 1, 3)], num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    assert idx.unique_subjects == ["A", "B"]
    assert len(idx.subject_to_indices["A"]) == 15
    assert len(idx.subject_to_indices["B"]) == 3
    assert idx.class_to_subjects[0] == ["A"]
    assert idx.class_to_subjects[1] == ["B"]


# ─── 2. Cell-count disparity does not change subject-selection probability ────

def test_subject_selection_probability_independent_of_cell_count():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    probs = subject_selection_probabilities(idx, cls=0, strategy="uniform")
    assert probs["A"] == pytest.approx(0.5)
    assert probs["B"] == pytest.approx(0.5)


def test_subject_selection_realized_probability_independent_of_cell_count():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(
        idx, batch_size=64, seed=0, batches_per_epoch=300,
        class_selection="uniform", subject_selection="uniform",
        cells_per_subject_cap=8, replacement=True,
    )
    counts = {"A": 0, "B": 0}
    for batch in sampler:
        for i in batch:
            s = subj[i]
            if s in counts:
                counts[s] += 1
    total = counts["A"] + counts["B"]
    # A has 100x the cells of B but must receive roughly equal draws.
    assert abs(counts["A"] / total - 0.5) < 0.08


# ─── 3. Uniform class sampling approaches equal class frequency ───────────────

def test_uniform_class_sampling_approaches_equal_frequency():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(
        idx, batch_size=64, seed=1, batches_per_epoch=300, class_selection="uniform",
    )
    class_counts = {0: 0, 1: 0}
    for batch in sampler:
        for i in batch:
            class_counts[int(labels[i])] += 1
    total = sum(class_counts.values())
    assert abs(class_counts[0] / total - 0.5) < 0.03
    assert abs(class_counts[1] / total - 0.5) < 0.03


def test_natural_class_sampling_is_proportional_to_subject_count():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    probs = class_selection_probabilities(idx, "natural")
    # class 0 has 2 subjects, class 1 has 20 subjects -> natural ~ 2/22, 20/22
    assert probs[0] == pytest.approx(2 / 22)
    assert probs[1] == pytest.approx(20 / 22)


def test_inverse_subject_frequency_favors_the_smaller_class():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    probs = class_selection_probabilities(idx, "inverse_subject_frequency")
    assert probs[0] > probs[1]  # class 0 has fewer subjects -> higher inverse weight


# ─── 4. Within-class subject selection approaches equal probability ───────────

def test_within_class_subject_selection_approaches_uniform():
    subj, labels = _dataset([(f"C{i}", 1, 50) for i in range(5)], num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=50, seed=2, batches_per_epoch=200,
                                           class_selection="uniform")
    subj_counts = {f"C{i}": 0 for i in range(5)}
    for batch in sampler:
        for i in batch:
            subj_counts[subj[i]] += 1
    total = sum(subj_counts.values())
    for c, n in subj_counts.items():
        assert abs(n / total - 0.2) < 0.03


# ─── 5/6. Every yielded index belongs to the provided dataset; no val/test leak ─

def test_every_yielded_index_belongs_to_the_dataset():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=32, seed=3, batches_per_epoch=20)
    n = len(subj)
    for batch in sampler:
        for i in batch:
            assert 0 <= i < n


def test_sampler_only_sees_the_dataset_it_was_built_from():
    """A separate 'validation' array is never passed to the sampler at all —
    the sampler has no mechanism to reach outside index.subject_to_indices,
    so no validation/test index can ever be yielded by construction."""
    train_subj, train_labels = _dataset([("A", 0, 20), ("B", 1, 20)], num_classes=2)
    val_subj, val_labels = _dataset([("V1", 0, 20)], num_classes=2)
    idx = SubjectClassIndex(subject_ids=train_subj, labels=train_labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=8, seed=4, batches_per_epoch=10)
    all_indices = {i for batch in sampler for i in batch}
    assert all_indices <= set(range(len(train_subj)))
    assert "V1" not in idx.unique_subjects


# ─── 7/8. Reproducibility ──────────────────────────────────────────────────────

def test_same_seed_produces_identical_sequence():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    s1 = SubjectBalancedBatchSampler(idx, batch_size=16, seed=7, batches_per_epoch=5)
    s2 = SubjectBalancedBatchSampler(idx, batch_size=16, seed=7, batches_per_epoch=5)
    assert list(s1) == list(s2)


def test_different_seeds_produce_different_sequences():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    s1 = SubjectBalancedBatchSampler(idx, batch_size=16, seed=7, batches_per_epoch=5)
    s2 = SubjectBalancedBatchSampler(idx, batch_size=16, seed=8, batches_per_epoch=5)
    assert list(s1) != list(s2)


# ─── 9. Batch size and epoch length are exact ──────────────────────────────────

def test_batch_size_and_epoch_length_are_exact():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=17, seed=0, batches_per_epoch=9)
    assert len(sampler) == 9
    batches = list(sampler)
    assert len(batches) == 9
    assert all(len(b) == 17 for b in batches)


def test_samples_per_epoch_derives_batches_per_epoch_by_ceil_division():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=10, seed=0, samples_per_epoch=95)
    assert len(sampler) == 10  # ceil(95/10)


def test_batches_per_epoch_and_samples_per_epoch_both_set_raises():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    with pytest.raises(SamplingConfigurationError):
        SubjectBalancedBatchSampler(idx, batch_size=10, seed=0, batches_per_epoch=5, samples_per_epoch=50)


# ─── 10. Cells-per-subject batch cap is enforced ───────────────────────────────

def test_cells_per_subject_cap_is_enforced():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(
        idx, batch_size=64, seed=5, batches_per_epoch=50, cells_per_subject_cap=4,
    )
    for batch in sampler:
        per_subject = {}
        for i in batch:
            per_subject[subj[i]] = per_subject.get(subj[i], 0) + 1
        # class 0 only has subjects A and B (2 candidates); a batch of 64
        # cells drawing from only 2 class-0-eligible subjects at a time
        # (interleaved with class 1 draws) can still exceed the cap once
        # both candidates are already at the cap — allow that documented
        # fallback, but the OVERWHELMING majority of draws must respect it.
        over_cap = sum(1 for v in per_subject.values() if v > 4)
        assert over_cap <= 2


def test_cells_per_subject_cap_must_be_positive():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    with pytest.raises(SamplingConfigurationError):
        SubjectBalancedBatchSampler(idx, batch_size=10, seed=0, batches_per_epoch=5, cells_per_subject_cap=0)


# ─── 11/12. Replacement behaviour ──────────────────────────────────────────────

def test_replacement_true_allows_repeated_cells_within_a_batch():
    subj, labels = _dataset([("A", 0, 3)], num_classes=1)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=1)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=10, seed=0, batches_per_epoch=1, replacement=True)
    batch = next(iter(sampler))
    assert len(batch) == 10  # more draws than the 3 available cells -> must repeat


def test_replacement_false_without_cap_raises():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    with pytest.raises(SamplingConfigurationError):
        SubjectBalancedBatchSampler(idx, batch_size=10, seed=0, batches_per_epoch=5, replacement=False)


def test_replacement_false_impossible_cap_raises_at_construction():
    subj, labels = _dataset([("A", 0, 3), ("B", 1, 3)], num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    with pytest.raises(SamplingImpossibleError):
        SubjectBalancedBatchSampler(
            idx, batch_size=4, seed=0, batches_per_epoch=1,
            replacement=False, cells_per_subject_cap=5,  # subjects only have 3 cells each
        )


def test_replacement_false_feasible_cap_succeeds_without_repeats():
    subj, labels = _dataset([("A", 0, 10), ("B", 1, 10)], num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(
        idx, batch_size=4, seed=0, batches_per_epoch=1, replacement=False, cells_per_subject_cap=5,
    )
    batch = next(iter(sampler))
    assert len(batch) == len(set(batch))  # no repeated cell index within the batch


# ─── 13. Placeholder subject IDs are rejected ──────────────────────────────────

@pytest.mark.parametrize("placeholder", ["unknown", "", "none", "None", "nan"])
def test_placeholder_subject_ids_are_rejected(placeholder):
    subj = np.array(["A", "A", placeholder], dtype=object)
    labels = np.array([0, 0, 0])
    with pytest.raises(SamplingConfigurationError, match="placeholder"):
        SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=1)


# ─── 14. Conflicting labels for one subject are rejected ──────────────────────

def test_conflicting_labels_for_one_subject_are_rejected():
    subj = np.array(["A", "A", "A"], dtype=object)
    labels = np.array([0, 0, 1])
    with pytest.raises(SamplingConfigurationError, match="conflicting"):
        SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)


# ─── 15. Out-of-range labels are rejected ──────────────────────────────────────

def test_out_of_range_labels_are_rejected():
    subj = np.array(["A", "A"], dtype=object)
    labels = np.array([0, 5])
    with pytest.raises(SamplingConfigurationError, match="out of range"):
        SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)


# ─── 16. Empty dataset is rejected ─────────────────────────────────────────────

def test_empty_dataset_is_rejected():
    with pytest.raises(SamplingConfigurationError):
        SubjectClassIndex(subject_ids=np.array([], dtype=object), labels=np.array([], dtype=np.int64), num_classes=2)


# ─── 17. Absent effective classes are reported honestly ───────────────────────

def test_absent_classes_are_reported():
    subj, labels = _dataset([("A", 0, 10), ("B", 1, 10)], num_classes=4)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=4)
    assert idx.observed_classes == [0, 1]
    assert idx.absent_classes == [2, 3]


# ─── 18. Excluded/absent classes are never sampled ─────────────────────────────

def test_absent_classes_are_never_sampled():
    subj, labels = _dataset([("A", 0, 10), ("B", 1, 10)], num_classes=4)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=4)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=16, seed=0, batches_per_epoch=20)
    seen_classes = set()
    for batch in sampler:
        for i in batch:
            seen_classes.add(int(labels[i]))
    assert seen_classes <= {0, 1}


def test_class_selection_probabilities_zero_for_absent_classes():
    subj, labels = _dataset([("A", 0, 10), ("B", 1, 10)], num_classes=4)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=4)
    probs = class_selection_probabilities(idx, "uniform")
    assert probs[2] == 0.0
    assert probs[3] == 0.0


# ─── 19. Underlying arrays are not mutated ─────────────────────────────────────

def test_sampling_does_not_mutate_input_arrays():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    subj_copy, labels_copy = subj.copy(), labels.copy()
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=32, seed=0, batches_per_epoch=10)
    for _ in sampler:
        pass
    assert np.array_equal(subj, subj_copy)
    assert np.array_equal(labels, labels_copy)


# ─── Diagnostics ────────────────────────────────────────────────────────────────

def test_diagnostics_expected_fields_populated():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=16, seed=0, batches_per_epoch=10, cells_per_subject_cap=4)
    diag = sampler.diagnostics()
    assert diag.observed_effective_classes == [0, 1]
    assert diag.unique_subjects_per_class == {0: 2, 1: 20}
    assert diag.batches_per_epoch == 10
    assert diag.batch_size == 16
    assert diag.cells_per_subject_cap == 4
    assert diag.realized_cells_per_class is None  # not iterated yet


def test_realized_diagnostics_populated_after_one_epoch():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=16, seed=0, batches_per_epoch=10)
    for _ in sampler:
        pass
    diag = sampler.last_realized_diagnostics
    assert diag is not None
    assert sum(diag.realized_cells_per_class.values()) == 16 * 10
    assert diag.realized_unique_subjects <= 22


# ─── Config resolution ──────────────────────────────────────────────────────────

def test_resolve_smoke_imbalance_config_defaults_are_backward_compatible():
    resolved = resolve_smoke_imbalance_config(None)
    assert resolved["sampler"] == "shuffle"
    assert resolved["class_weighting"] == "inverse_frequency"
    assert resolved["loss"] == "cross_entropy"


def test_resolve_smoke_imbalance_config_rejects_unknown_key():
    with pytest.raises(SamplingConfigurationError):
        resolve_smoke_imbalance_config({"not_a_real_key": 1})


def test_resolve_smoke_imbalance_config_rejects_invalid_sampler():
    with pytest.raises(SamplingConfigurationError):
        resolve_smoke_imbalance_config({"sampler": "bogus"})


def test_resolve_smoke_imbalance_config_merges_partial_overrides():
    resolved = resolve_smoke_imbalance_config({"sampler": "subject_balanced"})
    assert resolved["sampler"] == "subject_balanced"
    assert resolved["class_weighting"] == "inverse_frequency"  # untouched default preserved
