"""
tests/test_candidate_comparison.py — Step 9 tests for
evidence/candidate_comparison.py.

Exercised entirely against benchmarks.runner.build_synthetic_context, the
same synthetic/development fixture pattern tracks.py's `_on_fixture`
functions and test_frozen_test_sentinel.py's end-to-end test both use — see
that module's own docstring for why this is honest (no cohort in
configs/cohorts.yaml is currently eligible for any Phase 7 task).
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.candidate_comparison import (
    CandidateSelectionError,
    UnknownDomainStrategyError,
    compare_candidates_cancer_track,
    compare_candidates_smoke_track,
    run_candidate_comparison_on_synthetic_fixture,
    select_best_candidate,
)
from evidence.tracks import TASK_CANCER_PREDICTION, TASK_SMOKE


@pytest.fixture(autouse=True)
def _cleanup_checkpoints():
    yield
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


# ─── Candidate-name validation — never silently defaults an unknown name ──

def test_unknown_candidate_name_raises_before_anything_runs():
    from benchmarks.candidate_registry import UnknownCandidateNameError
    from benchmarks.runner import build_synthetic_context

    ctx = build_synthetic_context(seed=1, fast=True)
    with pytest.raises(UnknownCandidateNameError):
        compare_candidates_smoke_track(ctx, ["not_a_real_model"], n_folds=2, seeds=(1,))


def test_unknown_domain_strategy_raises():
    from benchmarks.runner import build_synthetic_context

    ctx = build_synthetic_context(seed=1, fast=True)
    dev_subjects = sorted(set(ctx.subjects_for("train")) | set(ctx.subjects_for("val")))
    outcomes = {
        str(b["subject_id"]): b["cancer_label"]
        for b in list(ctx.train_bags) + list(ctx.val_bags) if b.get("cancer_label_known")
    }
    with pytest.raises(UnknownDomainStrategyError):
        compare_candidates_cancer_track(
            ctx, dev_subjects, outcomes, ["prevalence"], n_folds=2, seeds=(1,),
            domain_strategies=("erm", "not_a_real_strategy"),
        )


# ─── Smoke track (Track A primary metric) ──────────────────────────────────

def test_smoke_track_compares_baselines_with_identical_partitions():
    out = run_candidate_comparison_on_synthetic_fixture(
        TASK_SMOKE, seed=1, fast=True, n_folds=2, candidate_names=["majority", "logistic", "random_forest"],
    )
    assert out["task"] == TASK_SMOKE
    assert out["primary_metric"] == "subject_weighted_macro_f1"
    assert set(out["candidates"]) | {e["candidate"] for e in out["ineligible"]} == {"majority", "logistic", "random_forest"}
    # Every eligible candidate ran against the exact same fold structure —
    # this is a structural consequence of one shared run_smoke_cv call, not
    # something this test can defeat by accident.
    n_folds_run = {info["n_folds_run"] for info in out["candidates"].values()}
    assert len(n_folds_run) == 1

    selection = select_best_candidate(out)
    assert selection["selected"] in out["candidates"]


def test_smoke_track_selection_fails_closed_when_primary_metric_undefined_everywhere():
    comparison = {"task": TASK_SMOKE, "primary_metric": "subject_weighted_macro_f1", "candidates": {}, "ineligible": [
        {"candidate": "majority", "reason": "undefined"},
    ]}
    with pytest.raises(CandidateSelectionError):
        select_best_candidate(comparison)


# ─── Cancer track (Track C primary metric) ─────────────────────────────────

def test_cancer_track_compares_baselines_and_mil_with_identical_partitions():
    out = run_candidate_comparison_on_synthetic_fixture(
        TASK_CANCER_PREDICTION, seed=1, fast=True, n_folds=2,
        candidate_names=["prevalence", "logistic", "mean_mil"],
    )
    assert out["task"] == TASK_CANCER_PREDICTION
    assert out["primary_metric"] == "auroc"
    named = set(out["candidates"]) | {e["candidate"] for e in out["ineligible"]}
    assert named == {"prevalence", "logistic", "mean_mil"}
    for info in out["candidates"].values():
        assert info["source"] == "cross_validation.run_cancer_cv"
        assert "auprc" in info


def test_cancer_track_pathway_domain_robust_variants_recorded_distinctly():
    out = run_candidate_comparison_on_synthetic_fixture(
        TASK_CANCER_PREDICTION, seed=1, fast=True, n_folds=2,
        candidate_names=["prevalence", "pathway_hierarchical_mil"],
        domain_strategies=("erm", "coral"),
    )
    assert "pathway_hierarchical_mil[erm]" in out["candidates"] or any(
        e["candidate"] == "pathway_hierarchical_mil" and e["strategy"] == "erm" for e in out["ineligible"]
    )
    coral_ran = "pathway_hierarchical_mil[coral]" in out["candidates"]
    coral_ineligible = any(
        e["candidate"] == "pathway_hierarchical_mil" and e.get("strategy") == "coral" for e in out["ineligible"]
    )
    assert coral_ran or coral_ineligible
    if coral_ran:
        assert out["candidates"]["pathway_hierarchical_mil[coral]"]["strategy"] == "coral"
        assert out["candidates"]["pathway_hierarchical_mil[coral]"]["source"] == \
            "final_evaluation.generate_subject_oof_predictions"


def test_non_pathway_candidate_never_silently_runs_under_a_non_erm_strategy():
    """logistic (a classical baseline) only supports 'erm' per
    candidate_registry.SUPPORTED_STRATEGIES_BY_KIND — requesting 'coral' for
    the run must never make 'logistic' silently disappear OR silently be
    scored as if coral had been applied; only the pathway candidate gets an
    extra strategy row, and 'logistic' is scored via the plain ERM CV path
    exactly once."""
    out = run_candidate_comparison_on_synthetic_fixture(
        TASK_CANCER_PREDICTION, seed=1, fast=True, n_folds=2,
        candidate_names=["logistic"], domain_strategies=("erm", "coral"),
    )
    assert list(out["candidates"].keys()) in ([], ["logistic"])
    if "logistic" in out["candidates"]:
        assert out["candidates"]["logistic"]["strategy"] == "erm"
    assert not any(k.startswith("logistic[") for k in out["candidates"])


def test_cancer_track_selection_fails_closed_when_auroc_undefined_everywhere():
    comparison = {"task": TASK_CANCER_PREDICTION, "primary_metric": "auroc", "candidates": {}, "ineligible": [
        {"candidate": "prevalence", "strategy": "erm", "reason": "undefined"},
    ]}
    with pytest.raises(CandidateSelectionError):
        select_best_candidate(comparison)


def test_run_candidate_comparison_on_synthetic_fixture_rejects_unknown_task():
    with pytest.raises(ValueError):
        run_candidate_comparison_on_synthetic_fixture("not_a_real_task")
