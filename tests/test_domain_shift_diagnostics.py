"""Tests for benchmarks/domain_shift.py — label-free gene-space/composition/
distribution-shift diagnostics and the source-predictability classifier."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.domain_shift import (
    composition_summary,
    distribution_shift_report,
    gene_space_compatibility,
    module_coverage_compatibility,
    source_predictability_diagnostic,
)


def test_gene_space_compatibility_identical_is_full_coverage():
    r = gene_space_compatibility(["g1", "g2", "g3"], ["g1", "g2", "g3"])
    assert r["gene_coverage"] == 1.0
    assert r["gene_order_compatible"] is True
    assert r["missing_genes"] == []


def test_gene_space_compatibility_reports_missing_genes():
    r = gene_space_compatibility(["g1", "g2", "g3"], ["g1", "g3"])
    assert r["missing_genes"] == ["g2"]
    assert r["gene_coverage"] < 1.0
    assert r["status"] == "evaluated"


def test_gene_space_compatibility_not_evaluable_without_raw_gene_provenance():
    """No fabricated 100% compatibility when the caller has no raw
    held-out gene panel to compare against — a structured not_evaluable
    status instead."""
    r = gene_space_compatibility(["g1", "g2", "g3"], None)
    assert r["status"] == "not_evaluable"
    assert r["reason"] == "raw source-specific gene contract unavailable"
    assert "gene_coverage" not in r


def test_gene_space_compatibility_deliberately_missing_held_out_gene_panel_reduces_coverage():
    """A held-out source's raw gene panel missing several required
    development genes must show REDUCED coverage — never the self-vs-self
    100% a caller could get by (mis)supplying the required list twice."""
    required = [f"g{i}" for i in range(10)]
    held_out_raw_panel = required[:6]  # source is missing the last 4 required genes entirely
    r = gene_space_compatibility(required, held_out_raw_panel)
    assert r["status"] == "evaluated"
    assert r["n_missing_genes"] == 4
    assert r["missing_genes"] == ["g6", "g7", "g8", "g9"]
    assert r["gene_coverage"] == pytest.approx(0.6)
    # the required (development) gene list itself is a plain input here —
    # this function never mutates it, and the caller's own frozen artifact
    # is never touched by anything this diagnostic computes.
    assert required == [f"g{i}" for i in range(10)]


def test_gene_space_compatibility_detects_duplicate_mappings():
    r = gene_space_compatibility(["g1", "g2"], ["g1", "g1", "g2"])
    assert r["n_duplicate_mappings"] == 1
    assert r["duplicate_mappings"] == ["g1"]


def test_module_coverage_compatibility_flags_empty_and_below_minimum():
    r = module_coverage_compatibility({"m1": 0, "m2": 2, "m3": 10}, min_genes_per_module=3)
    assert r["empty_modules"] == ["m1"]
    assert r["below_minimum_modules"] == ["m2"]


def test_composition_summary_reports_min_median_max_and_missing_types():
    r = composition_summary(
        cells_per_subject={"s1": 10, "s2": 100},
        cell_type_counts_per_subject={"s1": {0: 10}, "s2": {0: 50, 1: 50}},
    )
    assert r["cells_per_subject_summary"]["min"] == 10.0
    assert r["cells_per_subject_summary"]["max"] == 100.0
    assert "1" in r["missing_cell_types_per_subject"]["s1"]


def test_distribution_shift_higher_for_genuinely_shifted_features():
    rng = np.random.RandomState(0)
    dev = rng.randn(30, 6)
    similar = rng.randn(10, 6)
    shifted = rng.randn(10, 6) * 5 + 8
    r_similar = distribution_shift_report(dev, similar)
    r_shifted = distribution_shift_report(dev, shifted)
    for key in ("centroid_distance", "energy_distance", "coral_distance", "mmd_distance"):
        assert r_shifted[key] > r_similar[key], key


def test_distribution_shift_is_label_free_takes_only_feature_matrices():
    import inspect
    sig = inspect.signature(distribution_shift_report)
    assert set(sig.parameters) == {"dev_features", "held_out_features"}


def test_source_predictability_separable_sources_score_above_permutation_baseline():
    rng = np.random.RandomState(1)
    feat = np.concatenate([rng.randn(20, 5), rng.randn(20, 5) + 10])
    sources = ["a"] * 20 + ["b"] * 20
    groups = [f"s{i}" for i in range(40)]
    diag = source_predictability_diagnostic(feat, sources, groups, seed=1, n_folds=3)
    assert diag["status"] == "evaluated"
    assert diag["balanced_accuracy"] > diag["permutation_baseline_balanced_accuracy"]


def test_source_predictability_insufficient_evidence_with_one_source():
    diag = source_predictability_diagnostic(
        np.random.randn(5, 3), ["only_source"] * 5, [f"s{i}" for i in range(5)],
    )
    assert diag["status"] == "insufficient_evidence"


def test_source_predictability_never_reads_held_out_labels():
    import inspect
    sig = inspect.signature(source_predictability_diagnostic)
    # structurally cannot accept a held-out-source argument at all
    assert "held_out_features" not in sig.parameters
    assert "held_out_sources" not in sig.parameters
