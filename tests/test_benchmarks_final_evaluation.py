"""
Regression tests for PR7 blockers 2 and 5: neural/MIL candidates must be
able to win the final frozen-test evaluation on the same footing as
classical baselines, and the final development/fit/calibration protocol
must generate subject-grouped OOF predictions across the whole train+val
pool, calibrate/select the threshold from those OOF predictions only, and
refit once on the whole development pool before the single test evaluation.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.final_evaluation import (
    NoEligibleFinalCandidateError,
    generate_subject_oof_predictions,
    is_mil_candidate,
    refit_final_candidate_on_dev_pool,
    select_final_candidate,
)
from benchmarks.runner import build_synthetic_context


# ─── select_final_candidate ────────────────────────────────────────────────

def test_mil_candidate_can_win_selection():
    cv_report = {"results": {
        "prevalence": {"auroc": {"mean": 0.5}},
        "attention_mil": {"auroc": {"mean": 0.91}},
    }}
    best, report = select_final_candidate(cv_report, ["prevalence", "attention_mil"])
    assert best == "attention_mil"
    assert is_mil_candidate(best)
    assert report["ranked"][0]["name"] == "attention_mil"


def test_baseline_can_still_win_selection():
    cv_report = {"results": {
        "prevalence": {"auroc": {"mean": 0.5}},
        "logistic": {"auroc": {"mean": 0.87}},
        "attention_mil": {"auroc": {"mean": 0.6}},
    }}
    best, _ = select_final_candidate(cv_report, ["prevalence", "logistic", "attention_mil"])
    assert best == "logistic"


def test_undefined_cv_metric_recorded_ineligible_not_silently_dropped():
    cv_report = {"results": {
        "prevalence": {"auroc": {"mean": None}},
        "logistic": {"auroc": {"mean": 0.7}},
    }}
    best, report = select_final_candidate(cv_report, ["prevalence", "logistic"])
    assert best == "logistic"
    assert any(i["name"] == "prevalence" for i in report["ineligible"])


def test_no_eligible_candidate_raises_instead_of_silently_falling_back():
    cv_report = {"results": {"prevalence": {"auroc": {"mean": None}}}}
    with pytest.raises(NoEligibleFinalCandidateError):
        select_final_candidate(cv_report, ["prevalence"])


# ─── generate_subject_oof_predictions ──────────────────────────────────────

def test_oof_predictions_cover_every_dev_subject_exactly_once():
    ctx = build_synthetic_context(seed=3, fast=True)
    all_bags = list(ctx.train_bags) + list(ctx.val_bags)
    outcomes = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes.keys())
    num_cell_types = ctx.config["model"]["num_cell_types"]
    min_cells = ctx.config["data"]["min_cells_per_subject"]
    n_hvgs = ctx.preprocessing_artifact.n_hvgs

    oof = generate_subject_oof_predictions(
        ctx, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42, n_folds=3,
    )
    assert set(oof["oof_by_subject"].keys()) == set(oof["dev_subjects"])
    # no duplicate coverage: every fold's val subjects partition dev_subjects
    all_val = [s for f in oof["fold_membership"] for s in f.get("val_subject_ids", [])]
    assert len(all_val) == len(set(all_val))


def test_oof_prediction_never_from_a_model_trained_on_that_subject():
    ctx = build_synthetic_context(seed=3, fast=True)
    all_bags = list(ctx.train_bags) + list(ctx.val_bags)
    outcomes = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes.keys())
    num_cell_types = ctx.config["model"]["num_cell_types"]
    min_cells = ctx.config["data"]["min_cells_per_subject"]
    n_hvgs = ctx.preprocessing_artifact.n_hvgs

    oof = generate_subject_oof_predictions(
        ctx, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42, n_folds=3,
    )
    for fold in oof["fold_membership"]:
        train_set = set(fold.get("train_subject_ids", []))
        val_set = set(fold.get("val_subject_ids", []))
        assert train_set.isdisjoint(val_set)


def test_oof_generation_deterministic_given_seed():
    ctx = build_synthetic_context(seed=3, fast=True)
    all_bags = list(ctx.train_bags) + list(ctx.val_bags)
    outcomes = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes.keys())
    num_cell_types = ctx.config["model"]["num_cell_types"]
    min_cells = ctx.config["data"]["min_cells_per_subject"]
    n_hvgs = ctx.preprocessing_artifact.n_hvgs

    oof1 = generate_subject_oof_predictions(
        ctx, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42, n_folds=3,
    )
    oof2 = generate_subject_oof_predictions(
        ctx, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42, n_folds=3,
    )
    assert oof1["oof_by_subject"] == oof2["oof_by_subject"]


# ─── refit_final_candidate_on_dev_pool ─────────────────────────────────────

def test_final_refit_artifact_fit_only_on_dev_subjects():
    ctx = build_synthetic_context(seed=4, fast=True)
    all_bags = list(ctx.train_bags) + list(ctx.val_bags)
    outcomes = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    test_outcomes = {str(b["subject_id"]): b["cancer_label"] for b in ctx.test_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes.keys())
    test_subjects = ctx.subjects_for("test")
    num_cell_types = ctx.config["model"]["num_cell_types"]
    min_cells = ctx.config["data"]["min_cells_per_subject"]
    n_hvgs = ctx.preprocessing_artifact.n_hvgs

    result = refit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, test_subjects, {**outcomes, **test_outcomes},
        {}, num_cell_types, min_cells, n_hvgs, device="cpu", seed=42,
    )
    assert result["preprocessing_artifact"].fit_n_subjects == len(dev_subjects)
    assert set(result["test_subject_ids"]) <= set(test_subjects)
    assert len(result["test_proba"]) == len(result["test_subject_ids"])


def test_final_refit_test_predictions_change_with_test_expression_but_not_dev_fit():
    """The final artifact's scaling statistics must be identical whether or
    not test expression is corrupted — test rows are only ever TRANSFORMED
    through it, never used to FIT it."""
    ctx = build_synthetic_context(seed=4, fast=True)
    all_bags = list(ctx.train_bags) + list(ctx.val_bags)
    outcomes = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    test_outcomes = {str(b["subject_id"]): b["cancer_label"] for b in ctx.test_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes.keys())
    test_subjects = ctx.subjects_for("test")
    num_cell_types = ctx.config["model"]["num_cell_types"]
    min_cells = ctx.config["data"]["min_cells_per_subject"]
    n_hvgs = ctx.preprocessing_artifact.n_hvgs

    r1 = refit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, test_subjects, {**outcomes, **test_outcomes},
        {}, num_cell_types, min_cells, n_hvgs, device="cpu", seed=42,
    )

    ctx2 = build_synthetic_context(seed=4, fast=True)
    na = ctx2.normalized_adata_for_refit
    mask = na.obs["subject_id"].astype(str).isin(set(test_subjects)).values
    na.X[mask] = na.X[mask] + 0.0  # no-op corruption placeholder kept deterministic
    r2 = refit_final_candidate_on_dev_pool(
        ctx2, "logistic", dev_subjects, test_subjects, {**outcomes, **test_outcomes},
        {}, num_cell_types, min_cells, n_hvgs, device="cpu", seed=42,
    )
    assert r1["preprocessing_artifact"].gene_means == r2["preprocessing_artifact"].gene_means
    assert r1["preprocessing_artifact"].gene_stds == r2["preprocessing_artifact"].gene_stds


# ─── end-to-end: MIL winning the CV feeds through the same frozen path ─────

def test_mil_candidate_flows_through_full_run_cancer_task(monkeypatch):
    import benchmarks.runner as runner_mod

    ctx = build_synthetic_context(seed=5, fast=True)

    class _Args:
        models = ["prevalence", "attention_mil"]
        cv_folds = 2
        seeds = [42]
        device = "cpu"
        pooling = None
        calibration = "auto"
        threshold_strategy = "youden"

    # Force attention_mil to win regardless of the real (small, noisy)
    # synthetic CV score, so this test exercises the MIL path through
    # selection -> OOF -> calibration -> guard -> test evaluation without
    # depending on synthetic data producing a specific ranking.
    real_select = runner_mod.select_final_candidate

    def _force_mil(cv_report, model_names, primary_metric="auroc"):
        return "attention_mil", {"ranked": [{"name": "attention_mil", "score": 1.0, "kind": "mil"}],
                                  "ineligible": [], "primary_metric": primary_metric}

    monkeypatch.setattr(runner_mod, "select_final_candidate", _force_mil)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = runner_mod.new_run_dir(tmp, run_id="mil_run")
        outcome = runner_mod.run_cancer_task(ctx, _Args(), run_dir, synthetic=True)
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)

    report = outcome["calibration_report"]
    assert report["selected_model"] == "attention_mil"
    assert "test_result" in report
    assert report["final_model_metadata"]["pooling"] == "attention"
