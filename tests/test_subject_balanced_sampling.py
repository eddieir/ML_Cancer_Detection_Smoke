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


def test_samples_per_epoch_produces_exact_total_with_one_partial_final_batch():
    """samples_per_epoch=95, batch_size=10 must yield batch lengths
    [10]*9 + [5] — 95 indices total, NEVER rounded up to 100 (10 full
    batches) by a stray ceil-division."""
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=10, seed=0, samples_per_epoch=95)
    assert len(sampler) == 10  # 9 full + 1 partial
    batches = list(sampler)
    assert len(batches) == 10
    assert [len(b) for b in batches] == [10] * 9 + [5]
    assert sum(len(b) for b in batches) == 95


def test_samples_per_epoch_evenly_divisible_yields_only_full_batches():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=10, seed=0, samples_per_epoch=100)
    batches = list(sampler)
    assert len(batches) == 10
    assert all(len(b) == 10 for b in batches)
    assert sum(len(b) for b in batches) == 100


def test_samples_per_epoch_smaller_than_batch_size_yields_one_partial_batch():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=64, seed=0, samples_per_epoch=7)
    assert len(sampler) == 1
    batches = list(sampler)
    assert len(batches) == 1
    assert len(batches[0]) == 7


def test_samples_per_epoch_final_partial_batch_still_respects_subject_cap():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(
        idx, batch_size=10, seed=0, samples_per_epoch=25, cells_per_subject_cap=3,
    )
    batches = list(sampler)
    assert [len(b) for b in batches] == [10, 10, 5]
    for batch in batches:
        per_subject = {}
        for i in batch:
            per_subject[subj[i]] = per_subject.get(subj[i], 0) + 1
        assert all(v <= 3 for v in per_subject.values())


def test_len_matches_number_of_yielded_batches_for_samples_per_epoch():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=10, seed=0, samples_per_epoch=95)
    assert len(sampler) == len(list(sampler))


def test_samples_per_epoch_realized_diagnostics_report_exact_total():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(idx, batch_size=10, seed=0, samples_per_epoch=95)
    for _ in sampler:
        pass
    diag = sampler.last_realized_diagnostics
    assert diag.samples_per_epoch == 95
    assert sum(diag.realized_cells_per_class.values()) == 95


def test_batches_per_epoch_and_samples_per_epoch_both_set_raises():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    with pytest.raises(SamplingConfigurationError):
        SubjectBalancedBatchSampler(idx, batch_size=10, seed=0, batches_per_epoch=5, samples_per_epoch=50)


# ─── 10. Cells-per-subject batch cap is enforced (HARD, no exceptions) ─────────

def test_cells_per_subject_cap_is_enforced():
    """cells_per_subject_cap is a hard per-batch ceiling — every subject's
    contribution to every batch must be <= cap, with zero exceptions (no
    permitted violation count, unlike a prior looser version of this test)."""
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(
        idx, batch_size=64, seed=5, batches_per_epoch=50, cells_per_subject_cap=4,
    )
    for batch in sampler:
        per_subject = {}
        for i in batch:
            per_subject[subj[i]] = per_subject.get(subj[i], 0) + 1
        assert all(v <= 4 for v in per_subject.values()), per_subject


def test_cells_per_subject_cap_enforced_across_multiple_seeds_and_epochs():
    subj, labels = _dataset(IMBALANCED_SPEC, num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    for seed in (0, 1, 2, 17):
        sampler = SubjectBalancedBatchSampler(
            idx, batch_size=32, seed=seed, batches_per_epoch=20, cells_per_subject_cap=3,
        )
        for _epoch in range(3):  # multiple epochs from the SAME sampler instance
            for batch in sampler:
                per_subject = {}
                for i in batch:
                    per_subject[subj[i]] = per_subject.get(subj[i], 0) + 1
                assert all(v <= 3 for v in per_subject.values())


def test_cells_per_subject_cap_at_exact_capacity_boundary_succeeds():
    """batch_size exactly equals cap * n_unique_subjects — the tightest
    feasible configuration; must succeed and use every subject to the cap."""
    subj, labels = _dataset([("A", 0, 20), ("B", 0, 20), ("C", 1, 20)], num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    cap = 4
    n_subjects = 3
    sampler = SubjectBalancedBatchSampler(
        idx, batch_size=cap * n_subjects, seed=0, batches_per_epoch=5, cells_per_subject_cap=cap,
    )
    for batch in sampler:
        assert len(batch) == cap * n_subjects
        per_subject = {}
        for i in batch:
            per_subject[subj[i]] = per_subject.get(subj[i], 0) + 1
        assert all(v <= cap for v in per_subject.values())
        # at the exact boundary every subject must be used to its full cap
        assert all(per_subject.get(s, 0) == cap for s in ("A", "B", "C"))


def test_cells_per_subject_cap_infeasible_batch_size_raises_at_construction():
    """cap * n_unique_subjects < batch_size can never be filled without a
    cap violation — must fail clearly at construction, never mid-iteration."""
    subj, labels = _dataset([("A", 0, 100), ("B", 1, 100)], num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    with pytest.raises(SamplingImpossibleError):
        SubjectBalancedBatchSampler(
            idx, batch_size=64, seed=0, batches_per_epoch=5, cells_per_subject_cap=4,  # 4*2=8 << 64
        )


def test_cells_per_subject_cap_single_subject_class_redistributes_not_violates():
    """A minority class with exactly one subject exhausts its cap quickly;
    the sampler must redistribute remaining draws to other classes rather
    than exceed the cap on that one subject."""
    subj, labels = _dataset([("A", 0, 200)] + [(f"C{i}", 1, 200) for i in range(10)], num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(
        idx, batch_size=24, seed=0, batches_per_epoch=10, cells_per_subject_cap=4,
    )
    for batch in sampler:
        per_subject = {}
        for i in batch:
            per_subject[subj[i]] = per_subject.get(subj[i], 0) + 1
        assert per_subject.get("A", 0) <= 4
        assert all(v <= 4 for v in per_subject.values())
        assert len(batch) == 24  # still fully filled, using class-1 subjects for the rest


def test_cells_per_subject_cap_no_hang_across_many_epochs():
    """A tight-but-feasible cap configuration must complete promptly across
    many epochs — no infinite loop / unbounded retry."""
    subj, labels = _dataset([("A", 0, 50), ("B", 1, 50)], num_classes=2)
    idx = SubjectClassIndex(subject_ids=subj, labels=labels, num_classes=2)
    sampler = SubjectBalancedBatchSampler(
        idx, batch_size=8, seed=0, batches_per_epoch=30, cells_per_subject_cap=4,  # 4*2=8 == batch_size
    )
    for _ in range(5):
        batches = list(sampler)
        assert len(batches) == 30


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


# ─── configs/default.yaml agrees exactly with the Python default ──────────────

def _load_default_yaml_smoke_imbalance():
    import yaml
    path = Path(__file__).parents[1] / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["train"]["smoke_imbalance"]


def test_default_yaml_smoke_imbalance_matches_python_default_exactly():
    """configs/default.yaml's train.smoke_imbalance block must resolve to
    EXACTLY DEFAULT_SMOKE_IMBALANCE_CONFIG — any drift here would mean a
    real config file silently activates different behaviour than the
    documented backward-compatible default (see resolve_smoke_imbalance_config)."""
    from data.sampling import DEFAULT_SMOKE_IMBALANCE_CONFIG
    yaml_cfg = _load_default_yaml_smoke_imbalance()
    assert yaml_cfg == DEFAULT_SMOKE_IMBALANCE_CONFIG


def test_default_yaml_smoke_imbalance_resolves_without_modification():
    resolved = resolve_smoke_imbalance_config(_load_default_yaml_smoke_imbalance())
    from data.sampling import DEFAULT_SMOKE_IMBALANCE_CONFIG
    assert resolved == DEFAULT_SMOKE_IMBALANCE_CONFIG


def test_default_yaml_smoke_imbalance_sampler_is_shuffle_by_default():
    """The default sampler stays 'shuffle' (subject_balanced remains
    opt-in/experimental) until real development-only ablation evidence
    justifies changing it — see configs/default.yaml's own comment."""
    assert _load_default_yaml_smoke_imbalance()["sampler"] == "shuffle"
