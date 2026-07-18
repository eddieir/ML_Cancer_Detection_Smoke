"""
Tests for data/source_sampling.py — the source-first (source -> subject ->
cell) sampler for Phase 6 domain-robust training. Named distinctly from
tests/test_subject_balanced_sampling.py, which covers the pre-existing
smoke-class-balanced sampler (data/sampling.py) — a different, independent
mechanism.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.source_sampling import (
    SamplingConfigurationError,
    SamplingImpossibleError,
    SourceBalancedBatchSampler,
    SourceSubjectIndex,
    build_source_balanced_sampler,
)


def _imbalanced_index():
    subj_ids, sources = [], []
    for i in range(8):
        subj_ids += [f"a{i}"] * 5
        sources += ["sourceA"] * 5
    for i in range(2):
        subj_ids += [f"b{i}"] * 50
        sources += ["sourceB"] * 50
    return SourceSubjectIndex(subject_ids=np.array(subj_ids), sources=np.array(sources))


def test_source_selection_approximately_balanced_despite_subject_count_imbalance():
    idx = _imbalanced_index()
    sampler = SourceBalancedBatchSampler(idx, batch_size=20, seed=42, samples_per_epoch=400, source_selection="uniform")
    for _ in sampler:
        pass
    realized = sampler.last_realized_diagnostics.realized_samples_per_source
    total = sum(realized.values())
    # sourceA has 8 subjects, sourceB has 2 — a naive per-subject scheme
    # would draw ~4x more from sourceA; uniform source selection instead
    # keeps both sources within a reasonable band of 50/50.
    assert abs(realized["sourceA"] / total - 0.5) < 0.1
    assert abs(realized["sourceB"] / total - 0.5) < 0.1


def test_subject_selection_balanced_within_source():
    idx = _imbalanced_index()
    sampler = SourceBalancedBatchSampler(idx, batch_size=20, seed=1, samples_per_epoch=400)
    for _ in sampler:
        pass
    per_subject = sampler.last_realized_diagnostics.realized_samples_per_subject
    a_subject_counts = [v for s, v in per_subject.items() if s.startswith("a")]
    # every sourceA subject should receive roughly comparable draws (no
    # single subject dominating because of its own or its source's cell count)
    assert max(a_subject_counts) / max(min(a_subject_counts), 1) < 5


def test_cells_per_subject_cap_enforced_per_batch():
    idx = _imbalanced_index()
    sampler = SourceBalancedBatchSampler(idx, batch_size=5, seed=1, samples_per_epoch=50, cells_per_subject_cap=2)
    for batch in sampler:
        counts = {}
        for cell_idx in batch:
            subj = idx.subject_ids[cell_idx]
            counts[subj] = counts.get(subj, 0) + 1
        assert max(counts.values()) <= 2


def test_exact_samples_per_epoch_including_partial_final_batch():
    idx = _imbalanced_index()
    sampler = SourceBalancedBatchSampler(idx, batch_size=10, seed=1, samples_per_epoch=95)
    batches = list(sampler)
    assert sum(len(b) for b in batches) == 95
    assert [len(b) for b in batches] == [10] * 9 + [5]


def test_deterministic_given_seed():
    idx = _imbalanced_index()
    s1 = SourceBalancedBatchSampler(idx, batch_size=10, seed=7, samples_per_epoch=50)
    s2 = SourceBalancedBatchSampler(idx, batch_size=10, seed=7, samples_per_epoch=50)
    assert list(s1) == list(s2)


def test_different_seeds_differ():
    idx = _imbalanced_index()
    s1 = SourceBalancedBatchSampler(idx, batch_size=10, seed=1, samples_per_epoch=50)
    s2 = SourceBalancedBatchSampler(idx, batch_size=10, seed=2, samples_per_epoch=50)
    assert list(s1) != list(s2)


def test_infeasible_configuration_fails_at_construction_not_mid_iteration():
    idx = _imbalanced_index()
    with pytest.raises(SamplingImpossibleError):
        SourceBalancedBatchSampler(idx, batch_size=1000, seed=1, samples_per_epoch=1000, cells_per_subject_cap=1)


def test_replacement_false_requires_explicit_cap():
    idx = _imbalanced_index()
    with pytest.raises(SamplingConfigurationError):
        SourceBalancedBatchSampler(idx, batch_size=5, seed=1, samples_per_epoch=20, replacement=False)


def test_subject_assigned_to_two_sources_is_rejected():
    subj = np.array(["s1", "s1", "s2"])
    src = np.array(["A", "B", "A"])
    with pytest.raises(SamplingConfigurationError):
        SourceSubjectIndex(subject_ids=subj, sources=src)


def test_placeholder_subject_id_rejected():
    subj = np.array(["unknown", "s2", "s3"])
    src = np.array(["A", "A", "B"])
    with pytest.raises(SamplingConfigurationError):
        SourceSubjectIndex(subject_ids=subj, sources=src)


def test_build_source_balanced_sampler_excludes_held_out_subjects_entirely():
    """excluded_subjects (e.g. a held-out source's subjects) must never be
    drawn — not merely down-weighted."""

    class _FakeDataset:
        subject_ids = np.array(["a0"] * 5 + ["a1"] * 5 + ["held_out_0"] * 5)
        dataset_source = np.array(["sourceA"] * 10 + ["sourceB"] * 5)
        diagnostic_mode = False

    ds = _FakeDataset()
    sampler = build_source_balanced_sampler(
        ds, batch_size=5, seed=1, samples_per_epoch=50, excluded_subjects=["held_out_0"],
    )
    assert "held_out_0" not in sampler.index.unique_subjects
    for _ in sampler:
        pass
    realized_subjects = set()
    for subs in sampler.last_realized_diagnostics.realized_subjects_per_source.values():
        realized_subjects.update(subs)
    assert "held_out_0" not in realized_subjects


def test_diagnostics_match_emitted_samples():
    idx = _imbalanced_index()
    sampler = SourceBalancedBatchSampler(idx, batch_size=10, seed=3, samples_per_epoch=60)
    batches = list(sampler)
    diag = sampler.last_realized_diagnostics
    assert sum(diag.realized_samples_per_source.values()) == sum(len(b) for b in batches)
    assert diag.complete is True
