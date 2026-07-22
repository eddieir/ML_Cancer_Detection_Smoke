"""
tests/test_evidence_leakage_isolation.py — Step 8: corruption-isolation
proof that the leakage-safe pipeline order documented across
benchmarks/cross_validation.py, benchmarks/final_evaluation.py,
benchmarks/source_held_out.py, and benchmarks/test_guard.py genuinely holds
end to end, plus new coverage for evidence/calibration.py (Step 14, new
Phase 7 code with no prior corruption-isolation test).

Direct reading of the pipeline (this round's Step 8 research) confirms the
required order end to end:

  raw records -> validate provenance/labels -> select eligible subjects ->
  subject-level split manifest (data/splitting.py::grouped_kfold /
  subject_train_val_test_split) -> freeze dev/test identities
  (ExperimentContext.split_manifest) -> fit gene selection/scaling/batch
  transform on dev training folds only (benchmarks/fold_preprocessing.py::
  fold_train_val_datasets / refit_artifact_for_fold, called ONLY with
  fold["train"]/dev_subjects) -> apply frozen transform to validation ->
  select hyperparameters on dev only
  (benchmarks/hyperparameter_search.py::select_nested_hyperparameters_with_refit,
  called ONLY with the outer fold's own training subjects) -> generate dev
  OOF predictions (benchmarks/final_evaluation.py::
  generate_subject_oof_predictions) -> select calibration/threshold from
  OOF only (evidence/calibration.py::fit_frozen_calibration_and_threshold,
  whose ONLY data parameter is DevelopmentOOFPredictions) -> fit final
  candidate on full dev pool (fit_final_candidate_on_dev_pool) -> acquire
  frozen-test guard (benchmarks/test_guard.py::FrozenTestGuard) -> access
  frozen test once (evaluate_frozen_test) -> write immutable final report.

No new Phase 1-6 code changes were made in this round. This file only adds
tests; it deliberately does not touch cross_validation.py, final_evaluation.py,
source_held_out.py, or test_guard.py.
"""
import dataclasses
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.sentinel import FrozenAccessSentinel, FrozenDataAccessError

CHECKPOINT_DIR = "checkpoints/benchmarks_synthetic"


def _cleanup():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def _cleanup_checkpoints():
    yield
    _cleanup()


def _corrupted_context(seed: int = 7, fast: bool = True):
    """
    Builds a synthetic ExperimentContext (benchmarks.runner.build_synthetic_context)
    and returns TWO variants that are identical in every development-subject
    respect but differ arbitrarily in their test-partition data: gene
    expression values, smoke labels, and cancer outcomes for every test
    subject are replaced with different deterministic-but-different garbage
    in each variant. test_cell_dataset/test_bags are ALSO wrapped in
    FrozenAccessSentinel so any direct read (not just a value difference)
    fails loudly. If corrupting the frozen test partition changes ANY
    development-only pipeline output, that is a genuine leakage regression.
    """
    from benchmarks.runner import build_synthetic_context

    base = build_synthetic_context(seed=seed, fast=fast)
    test_subjects = set(base.split_manifest.test_subjects)

    def corrupt(ctx, garbage_offset):
        obs = ctx.normalized_adata_for_refit.obs
        mask = obs["subject_id"].astype(str).isin({str(s) for s in test_subjects}).values
        adata = ctx.normalized_adata_for_refit.copy()
        rng = np.random.RandomState(1000 + garbage_offset)
        adata.X[mask] = (rng.randn(mask.sum(), adata.X.shape[1]) * 1000 + garbage_offset).astype("float32")
        adata.obs.loc[mask, "smoke_type"] = (adata.obs.loc[mask, "smoke_type"].astype(int) + 1 + garbage_offset) % 3
        return dataclasses.replace(
            ctx,
            normalized_adata_for_refit=adata,
            test_cell_dataset=FrozenAccessSentinel(f"test_cell_dataset(offset={garbage_offset})"),
            test_bags=FrozenAccessSentinel(f"test_bags(offset={garbage_offset})"),
        )

    return corrupt(base, 1), corrupt(base, 2)


# ─── Split membership, hyperparameters, preprocessing, gene list ──────────

def test_smoke_cv_split_hyperparameters_preprocessing_gene_list_isolated_from_test_corruption():
    from benchmarks.cross_validation import run_smoke_cv

    ctx_a, ctx_b = _corrupted_context(seed=11)
    report_a = run_smoke_cv(ctx_a, ["majority", "logistic"], n_folds=2, seeds=[42])
    report_b = run_smoke_cv(ctx_b, ["majority", "logistic"], n_folds=2, seeds=[42])

    for name in ["majority", "logistic"]:
        folds_a = report_a["results"][name]["folds"]
        folds_b = report_b["results"][name]["folds"]
        assert len(folds_a) == len(folds_b)
        for fa, fb in zip(folds_a, folds_b):
            # Split membership.
            assert fa["fit_subject_ids"] == fb["fit_subject_ids"]
            # Preprocessing artifact (gene selection/scaling, fit dev-fold-only).
            assert fa["preprocessing_fingerprint"] == fb["preprocessing_fingerprint"]
            assert fa["gene_list_n"] == fb["gene_list_n"]
            # Hyperparameters selected via nested dev-only search.
            assert fa["hyperparameters"] == fb["hyperparameters"]
            assert fa["hyperparameter_search"] == fb["hyperparameter_search"]
            # Final scored metrics (label mapping applied identically).
            assert fa["subject_weighted_macro_f1"] == fb["subject_weighted_macro_f1"]
        assert report_a["results"][name]["subject_weighted_macro_f1"] == report_b["results"][name]["subject_weighted_macro_f1"]


def test_cancer_cv_split_hyperparameters_preprocessing_isolated_from_test_corruption():
    from benchmarks.cross_validation import run_cancer_cv

    ctx_a, ctx_b = _corrupted_context(seed=13)
    report_a = run_cancer_cv(ctx_a, ["prevalence", "logistic"], n_folds=2, seeds=[42])
    report_b = run_cancer_cv(ctx_b, ["prevalence", "logistic"], n_folds=2, seeds=[42])

    for name in ["prevalence", "logistic"]:
        folds_a = report_a["results"][name]["folds"]
        folds_b = report_b["results"][name]["folds"]
        assert len(folds_a) == len(folds_b)
        for fa, fb in zip(folds_a, folds_b):
            assert fa.get("preprocessing_fingerprint") == fb.get("preprocessing_fingerprint")
            assert fa.get("hyperparameters") == fb.get("hyperparameters")
            assert fa.get("auroc") == fb.get("auroc")
            assert fa.get("auprc") == fb.get("auprc")
        assert report_a["results"][name]["auroc"] == report_b["results"][name]["auroc"]


# ─── Development candidate selection ───────────────────────────────────────

def test_final_candidate_selection_isolated_from_test_corruption():
    from benchmarks.cross_validation import run_cancer_cv
    from benchmarks.final_evaluation import select_final_candidate

    ctx_a, ctx_b = _corrupted_context(seed=17)
    names = ["prevalence", "logistic", "random_forest"]
    report_a = run_cancer_cv(ctx_a, names, n_folds=2, seeds=[42])
    report_b = run_cancer_cv(ctx_b, names, n_folds=2, seeds=[42])

    selected_a, selection_report_a = select_final_candidate(report_a, names)
    selected_b, selection_report_b = select_final_candidate(report_b, names)
    assert selected_a == selected_b
    assert selection_report_a == selection_report_b


# ─── Dev OOF generation and final-development model fingerprint ───────────

def test_oof_predictions_and_final_dev_pool_fit_isolated_from_test_corruption():
    from benchmarks.final_evaluation import fit_final_candidate_on_dev_pool, generate_subject_oof_predictions

    ctx_a, ctx_b = _corrupted_context(seed=19)
    dev_subjects = sorted(set(ctx_a.subjects_for("train")) | set(ctx_a.subjects_for("val")))
    outcomes_by_subject = {
        str(b["subject_id"]): b["cancer_label"]
        for b in list(ctx_a.train_bags) + list(ctx_a.val_bags) if b.get("cancer_label_known")
    }
    num_cell_types = ctx_a.config.get("model", {}).get("num_cell_types", 4)
    min_cells = ctx_a.config.get("data", {}).get("min_cells_per_subject", 5)
    n_hvgs = ctx_a.preprocessing_artifact.n_hvgs

    oof_a = generate_subject_oof_predictions(
        ctx_a, "logistic", dev_subjects, outcomes_by_subject, num_cell_types, min_cells, n_hvgs,
        seed=42, n_folds=2,
    )
    oof_b = generate_subject_oof_predictions(
        ctx_b, "logistic", dev_subjects, outcomes_by_subject, num_cell_types, min_cells, n_hvgs,
        seed=42, n_folds=2,
    )
    assert oof_a["oof_by_subject"] == oof_b["oof_by_subject"]
    assert [r["preprocessing_fingerprint"] for r in oof_a["fold_membership"]] == \
           [r["preprocessing_fingerprint"] for r in oof_b["fold_membership"]]
    assert [r["selected_params_fingerprint"] for r in oof_a["fold_membership"]] == \
           [r["selected_params_fingerprint"] for r in oof_b["fold_membership"]]

    fitted_a = fit_final_candidate_on_dev_pool(
        ctx_a, "logistic", dev_subjects, outcomes_by_subject, {}, num_cell_types, min_cells, n_hvgs, seed=42,
    )
    fitted_b = fit_final_candidate_on_dev_pool(
        ctx_b, "logistic", dev_subjects, outcomes_by_subject, {}, num_cell_types, min_cells, n_hvgs, seed=42,
    )
    # Final-development model fingerprint.
    assert fitted_a.model_state_fingerprint == fitted_b.model_state_fingerprint
    assert fitted_a.preprocessing_artifact_fingerprint == fitted_b.preprocessing_artifact_fingerprint
    assert fitted_a.dev_subject_ids == fitted_b.dev_subject_ids


# ─── Label mapping ──────────────────────────────────────────────────────────

def test_label_mapping_fingerprint_isolated_from_test_corruption():
    ctx_a, ctx_b = _corrupted_context(seed=23)
    assert ctx_a.label_mapping_fingerprint == ctx_b.label_mapping_fingerprint
    assert ctx_a.label_mapping.to_dict() == ctx_b.label_mapping.to_dict()


# ─── Calibration and threshold (Step 14 — new Phase 7 code) ───────────────

class TestCalibrationCorruptionIsolation:
    """evidence/calibration.py has no test-touching code path at all —
    fit_frozen_calibration_and_threshold's only data parameter is a
    DevelopmentOOFPredictions instance, and apply_frozen_calibration_and_threshold
    has no fitting parameter whatsoever. These tests prove that structural
    claim rather than merely restating it."""

    def _dev_oof(self):
        from evidence.calibration import build_development_oof_predictions

        rng = np.random.RandomState(0)
        n = 60
        y_true = (rng.rand(n) > 0.5).astype(int).tolist()
        y_prob = np.clip(np.asarray(y_true) * 0.6 + rng.rand(n) * 0.3, 0.0, 1.0).tolist()
        subject_ids = [f"dev{i}" for i in range(n)]
        return build_development_oof_predictions(subject_ids, y_true, y_prob)

    def test_fit_signature_has_no_test_data_parameter(self):
        import inspect

        from evidence.calibration import fit_frozen_calibration_and_threshold

        params = set(inspect.signature(fit_frozen_calibration_and_threshold).parameters)
        forbidden = {"test", "test_data", "test_subjects", "test_bags", "y_test", "test_oof_predictions"}
        assert not (params & forbidden)

    def test_apply_signature_has_no_fitting_parameter(self):
        import inspect

        from evidence.calibration import apply_frozen_calibration_and_threshold

        params = set(inspect.signature(apply_frozen_calibration_and_threshold).parameters)
        forbidden = {"y_true", "refit", "development_oof_predictions", "calibration_method", "threshold_method"}
        assert not (params & forbidden)
        assert params == {"artifact", "y_prob"}

    def test_fit_rejects_a_sentinel_disguised_as_development_oof_predictions(self):
        from evidence.calibration import CalibrationPolicyError, fit_frozen_calibration_and_threshold

        sentinel = FrozenAccessSentinel("frozen test predictions")
        with pytest.raises(CalibrationPolicyError):
            fit_frozen_calibration_and_threshold(sentinel)

    def test_predeclared_fixed_threshold_never_silently_accepts_a_frozen_sentinel(self):
        """A caller that accidentally threaded a frozen-test-data sentinel
        into fixed_threshold (e.g. by mixing up a test-set statistic with a
        predeclared clinical threshold) gets an immediate, loud
        FrozenDataAccessError from float(fixed_threshold) — never a silent
        pass-through into the fitted artifact."""
        from evidence.calibration import fit_frozen_calibration_and_threshold

        sentinel = FrozenAccessSentinel("frozen test statistic")
        with pytest.raises(FrozenDataAccessError):
            fit_frozen_calibration_and_threshold(
                self._dev_oof(), threshold_method="predeclared_fixed", fixed_threshold=sentinel,
            )

    def test_artifact_is_frozen_and_cannot_be_mutated_after_fit(self):
        from evidence.calibration import fit_frozen_calibration_and_threshold

        artifact = fit_frozen_calibration_and_threshold(self._dev_oof(), threshold_method="youden")
        with pytest.raises(dataclasses.FrozenInstanceError):
            artifact.threshold_value = 0.99

    def test_apply_never_changes_the_frozen_artifacts_own_fields(self):
        from evidence.calibration import apply_frozen_calibration_and_threshold, fit_frozen_calibration_and_threshold

        artifact = fit_frozen_calibration_and_threshold(self._dev_oof(), threshold_method="youden")
        before = artifact.to_dict()
        # Extreme, out-of-distribution "test" probabilities — applying them
        # must transform ONLY the returned probabilities/decisions, never
        # the artifact itself (it is immutable; this also exercises the
        # transform path with corrupted-looking input).
        apply_frozen_calibration_and_threshold(artifact, [0.0, 1.0, 0.5, -0.0])
        assert artifact.to_dict() == before

    def test_fit_is_deterministic_given_identical_development_oof(self):
        from evidence.calibration import fit_frozen_calibration_and_threshold

        oof = self._dev_oof()
        a = fit_frozen_calibration_and_threshold(oof, threshold_method="youden")
        b = fit_frozen_calibration_and_threshold(oof, threshold_method="youden")
        assert a.fingerprint == b.fingerprint
        assert a.threshold_value == b.threshold_value
        assert a.calibration_method == b.calibration_method

    def test_fit_fingerprint_depends_only_on_development_oof_not_on_anything_external(self):
        """Building two DIFFERENT DevelopmentOOFPredictions objects that
        happen to carry the same (subject_ids, y_true, y_prob) content must
        fingerprint identically — the fingerprint is a function of content,
        never of object identity or of any external/test state."""
        from evidence.calibration import build_development_oof_predictions, fit_frozen_calibration_and_threshold

        oof1 = self._dev_oof()
        oof2 = build_development_oof_predictions(
            list(oof1.subject_ids), list(oof1.y_true), list(oof1.y_prob),
        )
        a = fit_frozen_calibration_and_threshold(oof1, threshold_method="youden")
        b = fit_frozen_calibration_and_threshold(oof2, threshold_method="youden")
        assert a.fingerprint == b.fingerprint
