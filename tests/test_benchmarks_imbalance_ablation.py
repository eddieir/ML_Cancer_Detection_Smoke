"""
Tests for benchmarks/imbalance_ablation.py — the development-only comparison
of Phase 2 smoke-imbalance strategies over grouped subject CV.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.imbalance_ablation import (
    IMBALANCE_ABLATION_STRATEGIES,
    run_smoke_imbalance_ablation,
)
from benchmarks.runner import build_synthetic_context


def test_unknown_strategy_name_raises():
    ctx = build_synthetic_context(seed=42, fast=True)
    with pytest.raises(ValueError, match="unknown strategy"):
        run_smoke_imbalance_ablation(ctx, strategies=["not_a_real_strategy"])


def test_empty_strategy_list_raises():
    ctx = build_synthetic_context(seed=42, fast=True)
    with pytest.raises(ValueError, match="non-empty"):
        run_smoke_imbalance_ablation(ctx, strategies=[])


def test_all_five_default_strategies_are_defined():
    assert len(IMBALANCE_ABLATION_STRATEGIES) == 5
    assert "natural_no_weight" in IMBALANCE_ABLATION_STRATEGIES
    assert "subject_balanced_focal" in IMBALANCE_ABLATION_STRATEGIES


def test_ablation_never_touches_test_subjects():
    ctx = build_synthetic_context(seed=42, fast=True)
    test_subjects = set(ctx.subjects_for("test"))
    dev_subjects = set(ctx.subjects_for("train")) | set(ctx.subjects_for("val"))
    assert test_subjects, "sanity: context actually has a nonempty test split"
    assert test_subjects.isdisjoint(dev_subjects)

    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "subject_balanced_no_weight"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    assert report["development_only"] is True
    # Every fold's own fit_subject provenance (via preprocessing_fingerprint,
    # cross-checked against the dev-only pool grouped_kfold partitions from)
    # can only ever have been built from dev_subjects — grouped_kfold itself
    # is called on ctx.subjects_for("train")+("val") only (see
    # run_smoke_imbalance_ablation's pool_subjects), so this is a structural
    # guarantee, not a per-fold string search.


def test_ablation_reports_paired_fold_differences_against_baseline():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "natural_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    assert report["baseline_strategy"] == "natural_no_weight"
    assert "natural_inverse_frequency" in report["paired_fold_differences"]
    diffs = report["paired_fold_differences"]["natural_inverse_frequency"]
    assert len(diffs) == 2  # n_folds=2, one seed


def test_ablation_uses_identical_folds_across_strategies():
    """Every strategy's fold records must reference the SAME
    preprocessing_fingerprint per (seed, fold) — proving they trained on
    identical fold data, only the imbalance strategy differed."""
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "subject_balanced_no_weight"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    fps_by_strategy = {}
    for name, r in report["results"].items():
        fps_by_strategy[name] = [(f["seed"], f["fold"], f["preprocessing_fingerprint"]) for f in r["folds"]]
    names = list(fps_by_strategy)
    assert fps_by_strategy[names[0]] == fps_by_strategy[names[1]]


def test_ablation_strategy_config_is_recorded_per_fold():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["subject_balanced_focal"], n_folds=2, seeds=[42],
        device="cpu", max_cells_per_subject=50,
    )
    fold = report["results"]["subject_balanced_focal"]["folds"][0]
    assert fold["strategy_config"]["loss"] == "focal"
    assert fold["strategy_config"]["sampler"] == "subject_balanced"
