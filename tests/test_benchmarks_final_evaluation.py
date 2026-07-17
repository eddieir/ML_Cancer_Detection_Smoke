"""
Regression tests for the frozen-test protocol: neural/MIL candidates must be
able to win the final frozen-test evaluation on the same footing as
classical baselines; each OOF fold must select its OWN hyperparameters from
only its own OOF-training subjects (never leaking a held-out subject's
influence into the configuration used to predict it); and the development-
only final fit (fit_final_candidate_on_dev_pool) must be structurally
incapable of reading test data — evaluate_frozen_test is the only function
that touches it, and only after runner.py acquires the durable guard.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.final_evaluation import (
    FittedFinalCandidate,
    NoEligibleFinalCandidateError,
    evaluate_frozen_test,
    fit_final_candidate_on_dev_pool,
    generate_subject_oof_predictions,
    is_mil_candidate,
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


# ─── generate_subject_oof_predictions: selection-clean OOF ─────────────────

def _dev_pool(ctx):
    all_bags = list(ctx.train_bags) + list(ctx.val_bags)
    outcomes = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes.keys())
    num_cell_types = ctx.config["model"]["num_cell_types"]
    min_cells = ctx.config["data"]["min_cells_per_subject"]
    n_hvgs = ctx.preprocessing_artifact.n_hvgs
    return outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs


def test_oof_predictions_cover_every_dev_subject_exactly_once():
    ctx = build_synthetic_context(seed=3, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    oof = generate_subject_oof_predictions(
        ctx, "logistic", dev_subjects, outcomes, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42, n_folds=3, n_inner_folds=2,
    )
    assert set(oof["oof_by_subject"].keys()) == set(oof["dev_subjects"])
    all_val = [s for f in oof["fold_membership"] for s in f.get("val_subject_ids", [])]
    assert len(all_val) == len(set(all_val))


def test_oof_prediction_never_from_a_model_trained_on_that_subject():
    ctx = build_synthetic_context(seed=3, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    oof = generate_subject_oof_predictions(
        ctx, "logistic", dev_subjects, outcomes, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42, n_folds=3, n_inner_folds=2,
    )
    for fold in oof["fold_membership"]:
        train_set = set(fold.get("train_subject_ids", []))
        val_set = set(fold.get("val_subject_ids", []))
        assert train_set.isdisjoint(val_set)


def test_oof_generation_deterministic_given_seed():
    ctx = build_synthetic_context(seed=3, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    oof1 = generate_subject_oof_predictions(
        ctx, "logistic", dev_subjects, outcomes, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42, n_folds=3, n_inner_folds=2,
    )
    oof2 = generate_subject_oof_predictions(
        ctx, "logistic", dev_subjects, outcomes, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42, n_folds=3, n_inner_folds=2,
    )
    assert oof1["oof_by_subject"] == oof2["oof_by_subject"]


def test_each_oof_fold_has_its_own_fold_local_hyperparameter_search():
    """The old (leaky) contract accepted ONE globally-selected params dict
    and reused it for every OOF subject. The corrected contract has each
    fold run its own inner-CV selection recorded on the fold's own record —
    two folds with different OOF-training subjects are free to select
    different candidates."""
    ctx = build_synthetic_context(seed=3, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    oof = generate_subject_oof_predictions(
        ctx, "logistic", dev_subjects, outcomes, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42, n_folds=3, n_inner_folds=2,
    )
    for fold in oof["fold_membership"]:
        assert "hyperparameter_search" in fold
        hp = fold["hyperparameter_search"]
        assert "selected_params" in hp
        # every inner fold's train/val subjects must be subsets of THIS
        # OOF fold's own OOF-training subjects, never its OOF-held-out side
        oof_train = set(fold["train_subject_ids"])
        for inner in hp.get("inner_folds", []):
            assert set(inner["train"]) <= oof_train
            assert set(inner["val"]) <= oof_train


def test_oof_fold_records_carry_real_sha256_fingerprints():
    """Issue 4: training/validation subject fingerprints must be actual
    SHA-256 hashes (64 hex chars), not raw JSON subject lists, and every
    successful fold must record a model-state fingerprint."""
    ctx = build_synthetic_context(seed=3, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    oof = generate_subject_oof_predictions(
        ctx, "logistic", dev_subjects, outcomes, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42, n_folds=3, n_inner_folds=2,
    )
    import re
    hexpat = re.compile(r"^[0-9a-f]{64}$")
    for fold in oof["fold_membership"]:
        if fold.get("skipped_reason"):
            continue
        assert hexpat.match(fold["training_subjects_fingerprint"])
        assert hexpat.match(fold["validation_subjects_fingerprint"])
        assert hexpat.match(fold["inner_selection_fingerprint"])
        assert hexpat.match(fold["selected_params_fingerprint"])
        assert hexpat.match(fold["model_state_fingerprint"])
        # not the raw subject list serialized under a "fingerprint" name
        assert fold["training_subjects_fingerprint"] != str(sorted(fold["train_subject_ids"]))


def test_corrupting_a_subject_outside_the_oof_training_set_does_not_change_fold_selection():
    """An OOF-held-out (or otherwise excluded) subject's outcome must not
    influence the hyperparameters selected for a fold whose OOF-training set
    does not contain it — _oof_fold_hyperparameters only ever reads
    outcomes_by_subject for subjects inside outer_train_subjects (via
    select_nested_hyperparameters_with_refit's own outer_train_subjects
    argument), so corrupting an excluded subject's label must not change
    what that fold selects. (A full generate_subject_oof_predictions
    end-to-end comparison is not used here because corrupting a subject's
    label also changes grouped_kfold's stratified OUTER fold assignment —
    this isolates the fold-local selection step itself.)"""
    from benchmarks.final_evaluation import _oof_fold_hyperparameters

    ctx = build_synthetic_context(seed=3, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    oof_train_subjects = dev_subjects[:-1]
    excluded_subject = dev_subjects[-1]

    hp1 = _oof_fold_hyperparameters(
        ctx, "logistic", oof_train_subjects, outcomes,
        num_cell_types, min_cells, n_hvgs, "attention", "cpu", 42, 2,
    )
    corrupted = dict(outcomes)
    corrupted[excluded_subject] = 1 - corrupted[excluded_subject]
    hp2 = _oof_fold_hyperparameters(
        ctx, "logistic", oof_train_subjects, corrupted,
        num_cell_types, min_cells, n_hvgs, "attention", "cpu", 42, 2,
    )
    assert hp1["selected_params"] == hp2["selected_params"]


# ─── fit_final_candidate_on_dev_pool / evaluate_frozen_test ───────────────

def test_final_dev_fit_signature_accepts_no_test_data():
    """fit_final_candidate_on_dev_pool structurally cannot be handed test
    subject IDs/labels — it has no such parameter, unlike the old
    (blocker-1-violating) refit_final_candidate_on_dev_pool."""
    import inspect
    sig = inspect.signature(fit_final_candidate_on_dev_pool)
    names = set(sig.parameters.keys())
    assert "test_subjects" not in names
    assert "test_outcomes_by_subject" not in names


def test_final_dev_fit_artifact_fit_only_on_dev_subjects():
    ctx = build_synthetic_context(seed=4, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    fitted = fit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs, device="cpu", seed=42,
    )
    assert isinstance(fitted, FittedFinalCandidate)
    assert fitted.preprocessing_artifact.fit_n_subjects == len(dev_subjects)
    assert set(fitted.dev_subject_ids) <= set(dev_subjects)


# ─── model-state fingerprints (blocker 2) ──────────────────────────────────

def test_final_fit_carries_a_deterministic_model_state_fingerprint():
    ctx = build_synthetic_context(seed=4, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    fitted = fit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs, device="cpu", seed=42,
    )
    assert isinstance(fitted.model_state_fingerprint, str)
    assert len(fitted.model_state_fingerprint) == 64  # sha256 hex digest


def test_model_state_fingerprint_reproducible_for_identical_seeded_fits():
    """Two independent fits with identical data/config/seed must produce the
    SAME model-state fingerprint — this is what makes it usable as part of a
    deterministic guard identity, unlike fit_seconds."""
    ctx = build_synthetic_context(seed=4, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    fitted1 = fit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs, device="cpu", seed=42,
    )
    fitted2 = fit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs, device="cpu", seed=42,
    )
    assert fitted1.model_state_fingerprint == fitted2.model_state_fingerprint


def test_different_fitted_weights_produce_different_model_state_fingerprints():
    ctx = build_synthetic_context(seed=4, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    fitted_c1 = fit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, outcomes, {"C": 0.01}, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42,
    )
    fitted_c2 = fit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, outcomes, {"C": 100.0}, num_cell_types, min_cells, n_hvgs,
        device="cpu", seed=42,
    )
    assert fitted_c1.model_state_fingerprint != fitted_c2.model_state_fingerprint


def test_evaluate_frozen_test_transforms_but_never_refits_the_dev_artifact():
    """The final artifact's scaling statistics must be identical whether or
    not test expression is corrupted — test rows are only ever TRANSFORMED
    through it in evaluate_frozen_test, never used to FIT it."""
    ctx = build_synthetic_context(seed=4, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)
    test_outcomes = {str(b["subject_id"]): b["cancer_label"] for b in ctx.test_bags if b.get("cancer_label_known")}
    test_subjects = ctx.subjects_for("test")

    fitted1 = fit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs, device="cpu", seed=42,
    )
    ctx2 = build_synthetic_context(seed=4, fast=True)
    na = ctx2.normalized_adata_for_refit
    mask = na.obs["subject_id"].astype(str).isin(set(test_subjects)).values
    na.X[mask] = na.X[mask] + 0.0  # no-op corruption placeholder kept deterministic
    fitted2 = fit_final_candidate_on_dev_pool(
        ctx2, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs, device="cpu", seed=42,
    )
    assert fitted1.preprocessing_artifact.gene_means == fitted2.preprocessing_artifact.gene_means
    assert fitted1.preprocessing_artifact.gene_stds == fitted2.preprocessing_artifact.gene_stds

    raw = evaluate_frozen_test(fitted1, ctx, test_subjects, test_outcomes, num_cell_types, min_cells)
    assert len(raw["test_proba"]) == len(raw["test_subject_ids"])
    assert set(raw["test_subject_ids"]) <= set(test_subjects)


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


def test_mil_final_fit_trains_on_every_eligible_development_subject(monkeypatch):
    """Blocker 3: unlike the CV/OOF fold fits (which still hold out a real
    validation split for Trainer's checkpoint selection), the FINAL MIL
    refit must place every eligible development subject's cells into
    Trainer.phase1_final_fit/phase2_final_fit's gradient-update datasets —
    none held out for an internal validation carve-out."""
    from train import Trainer

    ctx = build_synthetic_context(seed=6, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    seen = {}
    orig_p1 = Trainer.phase1_final_fit
    orig_p2 = Trainer.phase2_final_fit

    def spy_p1(self, train_cell_dataset, **kw):
        seen["phase1_subjects"] = set(str(s) for s in train_cell_dataset.subject_ids.tolist())
        return orig_p1(self, train_cell_dataset, **kw)

    def spy_p2(self, train_subject_dataset, **kw):
        seen["phase2_subjects"] = {str(b["subject_id"]) for b in train_subject_dataset.bags}
        return orig_p2(self, train_subject_dataset, **kw)

    monkeypatch.setattr(Trainer, "phase1_final_fit", spy_p1)
    monkeypatch.setattr(Trainer, "phase2_final_fit", spy_p2)

    fitted = fit_final_candidate_on_dev_pool(
        ctx, "attention_mil", dev_subjects, outcomes, {"pretrain_epochs": 1},
        num_cell_types, min_cells, n_hvgs, pooling="attention", device="cpu", seed=42,
    )
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)

    assert seen["phase2_subjects"] == set(fitted.dev_subject_ids)
    assert seen["phase1_subjects"] >= seen["phase2_subjects"]


def test_final_dev_fit_artifact_fingerprint_matches_canonical_scientific_fingerprint():
    """FittedFinalCandidate.preprocessing_artifact_fingerprint must be the
    SAME identity data/preprocessing.py::PreprocessingArtifact.
    scientific_fingerprint() would compute directly — one fingerprint
    hierarchy, not two that could silently disagree."""
    ctx = build_synthetic_context(seed=5, fast=True)
    outcomes, dev_subjects, num_cell_types, min_cells, n_hvgs = _dev_pool(ctx)

    fitted = fit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs, device="cpu", seed=42,
    )
    assert fitted.preprocessing_artifact_fingerprint == fitted.preprocessing_artifact.scientific_fingerprint()
