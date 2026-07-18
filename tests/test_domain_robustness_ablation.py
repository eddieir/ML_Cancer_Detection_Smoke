"""Tests for benchmarks/domain_robustness_ablation.py — multi-seed paired
strategy comparison against the ERM baseline."""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.domain_robustness_ablation import run_domain_robustness_ablation
from benchmarks.runner import build_synthetic_context

_BENCH_CFG = {"species_by_source": {"sourceA": "human", "sourceB": "human"}, "reference_species": "human"}


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_multi_seed_ablation_reports_seeds_and_paired_comparison():
    ctx = build_synthetic_context(seed=1, fast=True)
    report = run_domain_robustness_ablation(
        ctx, "cancer", ["prevalence"], device="cpu", seeds=[1, 2], include_adversarial=False, **_BENCH_CFG,
    )
    assert report["seeds"] == [1, 2]
    assert set(report["results"]["erm"]["per_seed"]) == {1, 2}
    for variant in ("source_balanced", "coral", "mmd"):
        comparison = report["paired_comparison_vs_erm"][variant]
        assert comparison["status"] in ("evaluated", "insufficient_evidence")
        if comparison["status"] == "evaluated":
            assert "wins" in comparison and "losses" in comparison and "ties" in comparison
            assert comparison["wins"] + comparison["losses"] + comparison["ties"] == comparison["n_common_source_seed_pairs"]


def test_single_seed_backward_compatible_default():
    ctx = build_synthetic_context(seed=1, fast=True)
    report = run_domain_robustness_ablation(
        ctx, "cancer", ["prevalence"], device="cpu", seed=3, include_adversarial=False, **_BENCH_CFG,
    )
    assert report["seeds"] == [3]


def test_smoke_task_non_erm_variants_marked_not_evaluable_per_seed():
    ctx = build_synthetic_context(seed=1, fast=True)
    report = run_domain_robustness_ablation(
        ctx, "smoke", ["majority"], device="cpu", seeds=[1, 2], include_adversarial=False, **_BENCH_CFG,
    )
    for variant in ("source_balanced", "coral", "mmd"):
        for s in (1, 2):
            assert report["results"][variant]["per_seed"][s]["status"] == "not_evaluable"
