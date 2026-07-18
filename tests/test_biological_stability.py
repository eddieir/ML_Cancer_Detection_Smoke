"""
Tests for benchmarks/biological_stability.py — module ranking stability,
attention stability/attention-vs-abundance, cell-order permutation
invariance, and the real-vs-synthetic module policy guard.

These are all SOFTWARE-DIAGNOSTIC tests against the synthetic gene-module
scheme (no real gene-set resource ships with this repository — see
README.md's Phase 6 section) and must not be read as biological validation.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.biological_stability import (
    RealModuleRequiredError,
    attention_vs_abundance,
    cell_order_permutation_invariance_check,
    cell_type_attention_by_subject,
    module_ablation_scores,
    module_ranking_stability,
    require_real_modules,
    spearman_rank_correlation,
    top_k_overlap,
)
from benchmarks.pathway_hierarchical_adapter import PathwayHierarchicalAdapter
from benchmarks.runner import build_synthetic_context
from train import SubjectLevelDataset


@pytest.fixture(scope="module")
def fitted_adapter():
    ctx = build_synthetic_context(seed=1, fast=True)
    train_sd = SubjectLevelDataset(ctx.train_bags, require_known_outcome=False)
    val_sd = SubjectLevelDataset(ctx.val_bags, require_known_outcome=False)
    adapter = PathwayHierarchicalAdapter(device="cpu")
    adapter.fit(ctx, None, None, train_sd, val_sd, seed=1, pretrain_epochs=2)
    return adapter, ctx


def test_spearman_rank_correlation_known_values():
    # identical ranking -> correlation 1.0
    a = {"m1": 3.0, "m2": 2.0, "m3": 1.0}
    b = {"m1": 30.0, "m2": 20.0, "m3": 10.0}
    assert spearman_rank_correlation(a, b) == pytest.approx(1.0)
    # fully reversed ranking -> correlation -1.0
    c = {"m1": 1.0, "m2": 2.0, "m3": 3.0}
    assert spearman_rank_correlation(a, c) == pytest.approx(-1.0)


def test_top_k_overlap_correct():
    a = {"m1": 5, "m2": 4, "m3": 1}
    b = {"m1": 5, "m2": 1, "m3": 4}
    assert top_k_overlap(a, b, k=1) == 1.0  # both agree m1 is top-1
    assert top_k_overlap(a, b, k=2) < 1.0   # disagree on 2nd place


def test_module_ranking_stability_deterministic_for_fixed_inputs(fitted_adapter):
    adapter, ctx = fitted_adapter
    scores1 = module_ablation_scores(adapter, ctx.val_bags)
    scores2 = module_ablation_scores(adapter, ctx.val_bags)
    assert scores1 == scores2  # same fitted model, same bags -> identical scores


def test_module_ranking_stability_requires_at_least_two_runs():
    result = module_ranking_stability([{"m1": 1.0}])
    assert result["status"] == "insufficient_evidence"


def test_module_ranking_stability_aggregates_across_runs(fitted_adapter):
    adapter, ctx = fitted_adapter
    s1 = module_ablation_scores(adapter, ctx.val_bags)
    s2 = {k: v * 1.1 for k, v in s1.items()}  # same ranking, different scale
    result = module_ranking_stability([s1, s2], top_k=2)
    assert result["status"] == "evaluated"
    assert result["mean_rank_correlation"] == pytest.approx(1.0)


def test_cell_type_attention_normalized_per_subject(fitted_adapter):
    adapter, ctx = fitted_adapter
    att = cell_type_attention_by_subject(adapter, ctx.val_bags)
    for sid, weights in att.items():
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)


def test_attention_vs_abundance_diagnostic_returns_null_comparison(fitted_adapter):
    adapter, ctx = fitted_adapter
    att = cell_type_attention_by_subject(adapter, ctx.val_bags)
    abundance = {}
    for b in ctx.val_bags:
        sid = str(b["subject_id"])
        vals, counts = np.unique(b["cell_type_ids"], return_counts=True)
        abundance[sid] = {int(v): int(c) for v, c in zip(vals, counts)}
    result = attention_vs_abundance(att, abundance, seed=0)
    assert result["status"] == "evaluated"
    assert "permutation_null_mean" in result


def test_cell_order_permutation_invariance_holds(fitted_adapter):
    adapter, ctx = fitted_adapter
    result = cell_order_permutation_invariance_check(adapter, ctx.val_bags, seed=0)
    assert result["max_abs_cancer_logit_difference"] < 1e-3


def test_real_module_required_error_rejects_synthetic_modules(fitted_adapter):
    adapter, _ = fitted_adapter
    with pytest.raises(RealModuleRequiredError):
        require_real_modules(adapter.modules)
