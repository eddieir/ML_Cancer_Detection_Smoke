"""Tests for benchmarks/uncertainty.py — development-only uncertainty and
abstention diagnostics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.uncertainty import (
    apply_abstention_threshold,
    binary_predictive_uncertainty,
    mc_dropout_uncertainty_report,
    select_abstention_threshold_from_development,
)


def test_entropy_highest_at_p_half():
    u = binary_predictive_uncertainty(np.array([0.01, 0.5, 0.99]))
    assert u["entropy"][1] > u["entropy"][0]
    assert u["entropy"][1] > u["entropy"][2]


def test_max_class_probability_correct():
    u = binary_predictive_uncertainty(np.array([0.2, 0.8]))
    assert u["max_class_probability"][0] == pytest.approx(0.8)
    assert u["max_class_probability"][1] == pytest.approx(0.8)


def test_abstention_threshold_selected_from_development_only():
    import inspect
    sig = inspect.signature(select_abstention_threshold_from_development)
    assert set(sig.parameters) == {"dev_uncertainty", "dev_correct", "target_coverage"}


def test_abstention_threshold_selection_realizes_target_coverage():
    dev_unc = np.linspace(0.0, 1.0, 20)
    dev_correct = np.ones(20, dtype=bool)
    sel = select_abstention_threshold_from_development(dev_unc, dev_correct, target_coverage=0.5)
    assert sel["development_realized_coverage"] == pytest.approx(0.5, abs=0.1)


def test_apply_abstention_threshold_reports_coverage_and_accuracy():
    held_out_unc = np.array([0.1, 0.2, 0.9, 0.95])
    held_out_correct = np.array([True, True, False, True])
    result = apply_abstention_threshold(held_out_unc, held_out_correct, threshold=0.5)
    assert result["n_retained"] == 2
    assert result["accuracy_at_coverage"] == 1.0
    assert result["accuracy_full_population"] == 0.75


def test_apply_abstention_never_drops_subjects_from_full_population_metric():
    held_out_unc = np.array([0.9, 0.9, 0.9])
    held_out_correct = np.array([True, False, True])
    result = apply_abstention_threshold(held_out_unc, held_out_correct, threshold=0.1)
    assert result["n_total"] == 3
    assert result["accuracy_full_population"] == pytest.approx(2 / 3)


def test_mc_dropout_report_is_not_labeled_a_confidence_interval():
    from benchmarks.pathway_hierarchical_adapter import PathwayHierarchicalAdapter
    from benchmarks.runner import build_synthetic_context
    from train import SubjectLevelDataset

    ctx = build_synthetic_context(seed=1, fast=True)
    train_sd = SubjectLevelDataset(ctx.train_bags, require_known_outcome=False)
    val_sd = SubjectLevelDataset(ctx.val_bags, require_known_outcome=False)
    adapter = PathwayHierarchicalAdapter(device="cpu")
    adapter.fit(ctx, None, None, train_sd, val_sd, seed=1, pretrain_epochs=2)
    report = mc_dropout_uncertainty_report(adapter, ctx.val_bags, n_passes=5)
    assert "not a calibrated confidence interval" in report["note"].lower()
    assert len(report["cancer_proba_mean"]) == len(ctx.val_bags)
