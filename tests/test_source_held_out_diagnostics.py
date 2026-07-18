"""
Tests for benchmarks/source_held_out_diagnostics.py — the domain-shift/
uncertainty/biological-stability wiring used by run_cancer_source_held_out.
Exercises the diagnostic functions directly against a genuinely-fitted
pathway_hierarchical_mil candidate (built with fit_final_candidate_on_dev_pool,
exactly like source_held_out.py does), rather than going through the full
LOSO sweep, since the tiny synthetic fixture is too small for
pathway_hierarchical_mil to win development-only OOF selection there.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.final_evaluation import fit_final_candidate_on_dev_pool
from benchmarks.fold_preprocessing import bags_from_fold_cell_dataset, build_fold_cell_dataset, require_normalized_adata
from benchmarks.runner import build_synthetic_context
from benchmarks.source_held_out_diagnostics import (
    cancer_biological_stability_report,
    cancer_domain_shift_report,
    cancer_uncertainty_report,
)


@pytest.fixture(scope="module")
def fixture():
    ctx = build_synthetic_context(seed=1, fast=True)
    all_bags = list(ctx.train_bags) + list(ctx.val_bags)
    outcomes = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    subj_to_source = {str(b["subject_id"]): str(b.get("source") or "unknown") for b in all_bags}
    all_subjects = sorted(outcomes)
    dev_subjects, held_out_subjects = all_subjects[:-4], all_subjects[-4:]
    num_cell_types = ctx.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = ctx.preprocessing_artifact.n_hvgs
    min_cells = 1

    fitted = fit_final_candidate_on_dev_pool(
        ctx, "pathway_hierarchical_mil", dev_subjects, outcomes, {}, num_cell_types, min_cells, n_hvgs,
        pooling=None, device="cpu", seed=1,
    )
    normalized_adata = require_normalized_adata(ctx)
    dev_ds = build_fold_cell_dataset(normalized_adata, fitted.preprocessing_artifact, dev_subjects)
    held_out_ds = build_fold_cell_dataset(normalized_adata, fitted.preprocessing_artifact, held_out_subjects)
    dev_bags = bags_from_fold_cell_dataset(dev_ds, outcomes, min_cells)
    held_out_bags = bags_from_fold_cell_dataset(held_out_ds, outcomes, min_cells)
    return {
        "context": ctx, "fitted": fitted, "dev_bags": dev_bags, "held_out_bags": held_out_bags,
        "outcomes": outcomes, "subj_to_source": subj_to_source, "num_cell_types": num_cell_types,
        "min_cells": min_cells,
    }


def test_domain_shift_report_populated_and_label_free(fixture):
    report = cancer_domain_shift_report(
        fixture["dev_bags"], fixture["held_out_bags"], fixture["num_cell_types"], fixture["subj_to_source"], seed=1,
    )
    for key in ("centroid_distance", "energy_distance", "coral_distance", "mmd_distance", "source_predictability",
                "held_out_composition"):
        assert key in report
    assert np.isfinite(report["centroid_distance"])


def test_uncertainty_report_selects_threshold_from_development_only(fixture):
    fitted = fixture["fitted"]
    held_out_sd_bags = fixture["held_out_bags"]
    from train import SubjectLevelDataset
    dev_sd = SubjectLevelDataset(fixture["dev_bags"])
    held_out_sd = SubjectLevelDataset(held_out_sd_bags)
    prob_oof = fitted.predictor.predict_proba(dev_sd)
    y_oof = np.array([b["cancer_label"] for b in dev_sd.bags])
    held_out_proba = fitted.predictor.predict_proba(held_out_sd)
    y_held_out = np.array([b["cancer_label"] for b in held_out_sd.bags])

    report = cancer_uncertainty_report(
        y_oof, prob_oof, y_held_out, held_out_proba, fitted=fitted, held_out_bags=held_out_sd_bags,
    )
    assert report["development_threshold_selection"]["status"] == "selected"
    assert report["abstention_at_held_out"]["n_total"] == len(y_held_out)
    assert report["mc_dropout"]["n_passes"] > 0


def test_biological_stability_report_labels_synthetic_modules(fixture):
    report = cancer_biological_stability_report(
        fixture["context"], fixture["fitted"], fixture["dev_bags"], fixture["held_out_bags"],
        fixture["outcomes"], fixture["num_cell_types"], fixture["min_cells"], seed=1,
    )
    assert report["is_synthetic_modules"] is True
    assert report["scope"] == "software_diagnostic_only"
    assert "module_ablation_scores" in report
    assert "cell_type_label_permutation_null" in report
    assert "matched_size_random_module_null" in report
    assert "attention_vs_abundance" in report
    assert "label_permutation_null" in report


def test_biological_stability_not_applicable_for_non_pathway_candidate(fixture):
    from benchmarks.final_evaluation import FittedFinalCandidate

    baseline_fitted = FittedFinalCandidate(
        candidate_name="prevalence", kind="baseline", pooling=None, selected_params={},
        preprocessing_artifact=fixture["fitted"].preprocessing_artifact,
        preprocessing_artifact_fingerprint=fixture["fitted"].preprocessing_artifact_fingerprint,
        dev_subject_ids=fixture["fitted"].dev_subject_ids, model_metadata={}, model_state_fingerprint="x",
        predictor=None,
    )
    report = cancer_biological_stability_report(
        fixture["context"], baseline_fitted, fixture["dev_bags"], fixture["held_out_bags"],
        fixture["outcomes"], fixture["num_cell_types"], fixture["min_cells"], seed=1,
    )
    assert report["status"] == "not_applicable"
