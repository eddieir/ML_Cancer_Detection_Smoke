"""benchmarks/eligibility.py — Task A/B eligibility reports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.eligibility import check_task_a_eligibility, check_task_b_eligibility
from benchmarks.runner import build_synthetic_context


def test_task_a_eligible_on_synthetic_context():
    ctx = build_synthetic_context(seed=1, fast=True)
    report = check_task_a_eligibility(ctx)
    assert report.eligible
    assert report.status == "ELIGIBLE"


def test_task_b_eligible_on_synthetic_context():
    ctx = build_synthetic_context(seed=1, fast=True)
    report = check_task_b_eligibility(ctx)
    assert report.eligible
    assert report.status == "ELIGIBLE"
    assert report.counts["train"]["n_positive"] > 0
    assert report.counts["train"]["n_negative"] > 0


def test_task_b_not_evaluable_when_no_known_outcomes():
    ctx = build_synthetic_context(seed=1, fast=True)
    for b in ctx.train_bags + ctx.val_bags + ctx.test_bags:
        b["cancer_label_known"] = False
        b["cancer_label"] = None
    report = check_task_b_eligibility(ctx)
    assert not report.eligible
    assert report.status == "NOT_EVALUABLE"
    assert any("known" in r for r in report.reasons)


def test_task_b_not_evaluable_when_test_split_single_class():
    ctx = build_synthetic_context(seed=1, fast=True)
    for b in ctx.test_bags:
        b["cancer_label"] = 1
    report = check_task_b_eligibility(ctx)
    assert not report.eligible
    assert any("test split" in r for r in report.reasons)


def test_task_a_not_evaluable_with_zero_subjects_in_a_split():
    ctx = build_synthetic_context(seed=1, fast=True)
    ctx.test_cell_dataset = ctx.test_cell_dataset.subset_by_subjects([])
    report = check_task_a_eligibility(ctx)
    assert not report.eligible
    assert any("test split" in r for r in report.reasons)
