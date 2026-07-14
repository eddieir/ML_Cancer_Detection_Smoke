"""benchmarks/eligibility.py — Task A/B eligibility reports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.eligibility import (
    check_task_a_eligibility,
    check_task_b_development_eligibility,
    check_test_evaluability,
)
from benchmarks.runner import build_synthetic_context


def test_task_a_eligible_on_synthetic_context():
    ctx = build_synthetic_context(seed=1, fast=True)
    report = check_task_a_eligibility(ctx)
    assert report.eligible
    assert report.status == "ELIGIBLE"


def test_task_b_development_eligible_on_synthetic_context():
    ctx = build_synthetic_context(seed=1, fast=True)
    report = check_task_b_development_eligibility(ctx.train_bags, ctx.val_bags)
    assert report.eligible
    assert report.status == "ELIGIBLE"
    assert report.counts["train"]["n_positive"] > 0
    assert report.counts["train"]["n_negative"] > 0


def test_task_b_development_not_evaluable_when_no_known_outcomes():
    ctx = build_synthetic_context(seed=1, fast=True)
    for b in ctx.train_bags + ctx.val_bags:
        b["cancer_label_known"] = False
        b["cancer_label"] = None
    report = check_task_b_development_eligibility(ctx.train_bags, ctx.val_bags)
    assert not report.eligible
    assert report.status == "NOT_EVALUABLE"
    assert any("known" in r for r in report.reasons)


def test_task_b_development_eligibility_never_reads_test_bags():
    """A one-class (or fully corrupted) test split must not change
    development eligibility at all — check_task_b_development_eligibility's
    signature doesn't even accept test_bags, so this is enforced
    structurally, not just by convention."""
    import inspect
    sig = inspect.signature(check_task_b_development_eligibility)
    assert "test_bags" not in sig.parameters
    assert set(sig.parameters) == {"train_bags", "val_bags"}


def test_test_evaluability_reports_undefined_auroc_without_rejecting():
    ctx = build_synthetic_context(seed=1, fast=True)
    for b in ctx.test_bags:
        b["cancer_label"] = 1
        b["cancer_label_known"] = True
    result = check_test_evaluability(ctx.test_bags)
    assert result["auroc_auprc_defined"] is False
    assert any("only one class" in r for r in result["reasons"])
    # never silently rejects/raises — always returns a structured report
    assert result["n_known"] > 0


def test_test_evaluability_defined_when_both_classes_present():
    ctx = build_synthetic_context(seed=1, fast=True)
    result = check_test_evaluability(ctx.test_bags)
    assert result["auroc_auprc_defined"] is True
    assert result["reasons"] == []


def test_task_a_not_evaluable_with_zero_subjects_in_a_split():
    ctx = build_synthetic_context(seed=1, fast=True)
    ctx.test_cell_dataset = ctx.test_cell_dataset.subset_by_subjects([])
    report = check_task_a_eligibility(ctx)
    assert not report.eligible
    assert any("test split" in r for r in report.reasons)
