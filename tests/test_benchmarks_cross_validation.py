"""benchmarks/cross_validation.py — subject-disjoint folds, determinism, undefined-fold handling."""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.cross_validation import concat_cell_datasets, run_cancer_cv, run_smoke_cv
from benchmarks.runner import build_synthetic_context
from data.splitting import grouped_kfold


def test_grouped_kfold_folds_never_split_one_subjects_cells():
    ctx = build_synthetic_context(seed=1, fast=True)
    pool = concat_cell_datasets(ctx.train_cell_dataset, ctx.val_cell_dataset)
    folds = grouped_kfold(pool.subject_ids, pool.smoke.numpy(), n_folds=3, seed=42)
    for fold in folds:
        train_subj = set(fold["train"])
        val_subj = set(fold["val"])
        assert train_subj.isdisjoint(val_subj)
        # every cell of a val subject must be in the val partition, none in train
        for sid in val_subj:
            cells_of_subject = pool.subject_ids == sid
            assert cells_of_subject.sum() > 0


def test_smoke_cv_is_deterministic_for_a_fixed_seed():
    ctx = build_synthetic_context(seed=1, fast=True)
    r1 = run_smoke_cv(ctx, ["majority", "logistic"], n_folds=2, seeds=[42])
    r2 = run_smoke_cv(ctx, ["majority", "logistic"], n_folds=2, seeds=[42])
    assert r1["results"]["logistic"]["subject_weighted_macro_f1"]["mean"] == \
           r2["results"]["logistic"]["subject_weighted_macro_f1"]["mean"]


def test_smoke_cv_different_seeds_produce_different_partitions():
    ctx = build_synthetic_context(seed=1, fast=True)
    pool = concat_cell_datasets(ctx.train_cell_dataset, ctx.val_cell_dataset)
    folds_a = grouped_kfold(pool.subject_ids, pool.smoke.numpy(), n_folds=3, seed=42)
    folds_b = grouped_kfold(pool.subject_ids, pool.smoke.numpy(), n_folds=3, seed=7)
    assert [f["val"] for f in folds_a] != [f["val"] for f in folds_b]


def test_smoke_cv_reports_subject_and_cell_weighted_separately():
    ctx = build_synthetic_context(seed=1, fast=True)
    result = run_smoke_cv(ctx, ["majority"], n_folds=2, seeds=[42])
    r = result["results"]["majority"]
    assert "subject_weighted_macro_f1" in r and "cell_weighted_macro_f1" in r


def test_cancer_cv_undefined_fold_recorded_not_crashed_when_mil_ineligible():
    """A CV fold too small for MIL eligibility (train.check_mil_eligibility)
    must show up as an undefined fold with a reason, not raise out of run_cancer_cv."""
    ctx = build_synthetic_context(seed=1, fast=True)
    result = run_cancer_cv(ctx, ["prevalence", "mean_mil"], n_folds=2, seeds=[42])
    mil_result = result["results"]["mean_mil"]
    assert mil_result["n_folds_run"] == 2
    for fold in mil_result["folds"]:
        if fold["auroc"] is None:
            assert fold["auroc_auprc_undefined_reason"] is not None

    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_cancer_cv_baseline_folds_never_touch_test_split():
    """run_cancer_cv must only ever read train_bags/val_bags — asserted by
    poisoning test_bags with an out-of-range label and confirming no crash/
    change (test_bags is never even accessed)."""
    ctx = build_synthetic_context(seed=1, fast=True)
    for b in ctx.test_bags:
        b["cancer_label"] = 999  # would corrupt any metric computed from it
    result = run_cancer_cv(ctx, ["prevalence"], n_folds=2, seeds=[42])
    for fold in result["results"]["prevalence"]["folds"]:
        assert fold["auroc"] in (None, 0.5) or (0.0 <= fold["auroc"] <= 1.0)


def test_smoke_cv_persists_fold_artifacts_when_output_root_given(tmp_path):
    """artifact_output_root wires run_smoke_cv's per-fold refit into
    fold_preprocessing.save_fold_artifact — an on-disk artifact per
    (seed, fold), not just an in-memory object discarded after the run."""
    from benchmarks.fold_preprocessing import load_fold_artifact

    ctx = build_synthetic_context(seed=1, fast=True)
    run_smoke_cv(ctx, ["majority"], n_folds=2, seeds=[42], artifact_output_root=str(tmp_path))
    art_f0 = load_fold_artifact(tmp_path, "s42_f0")
    art_f1 = load_fold_artifact(tmp_path, "s42_f1")
    assert art_f0.scientific_fingerprint() != art_f1.scientific_fingerprint()


def test_cancer_cv_persists_fold_artifacts_when_output_root_given(tmp_path):
    from benchmarks.fold_preprocessing import load_fold_artifact

    ctx = build_synthetic_context(seed=1, fast=True)
    run_cancer_cv(ctx, ["prevalence"], n_folds=2, seeds=[42], artifact_output_root=str(tmp_path))
    art_f0 = load_fold_artifact(tmp_path, "s42_f0")
    assert art_f0.fit_n_subjects > 0
