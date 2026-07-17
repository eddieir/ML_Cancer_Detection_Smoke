"""
pathway_hierarchical_mil model registry / CV / hyperparameter-search /
final-development-fit / ablation integration (Phase 5, second pass).

Uses benchmarks.runner.build_synthetic_context — the same synthetic
ExperimentContext every other CV/final-evaluation test in this repository
builds from — so these tests exercise the REAL cross_validation.py,
final_evaluation.py, and pathway_hierarchical_ablation.py code paths, not a
hand-rolled substitute.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.cross_validation import run_cancer_cv, run_smoke_cv
from benchmarks.final_evaluation import (
    fit_final_candidate_on_dev_pool,
    generate_subject_oof_predictions,
    is_mil_candidate,
)
from benchmarks.mil_registry import MIL_CANDIDATE_NAMES, build_mil_adapter
from benchmarks.pathway_hierarchical_adapter import (
    MODEL_NAME as PATHWAY_MODEL_NAME,
    PathwayHierarchicalAdapter,
    PathwayModuleConfigurationError,
    build_gene_modules_for_context,
)
from benchmarks.pathway_hierarchical_ablation import (
    PATHWAY_ABLATION_VARIANTS,
    run_pathway_hierarchical_ablation,
)
from benchmarks.runner import build_synthetic_context


# ══════════════════════════ Model registry / CLI selection ═══════════════

def test_pathway_model_registered_as_mil_candidate():
    assert PATHWAY_MODEL_NAME in MIL_CANDIDATE_NAMES
    assert is_mil_candidate(PATHWAY_MODEL_NAME)


def test_unknown_model_name_still_fails_clearly():
    with pytest.raises(ValueError):
        build_mil_adapter("not_a_real_model", None, "cpu")


def test_build_mil_adapter_returns_pathway_adapter_for_pathway_name():
    adapter = build_mil_adapter(PATHWAY_MODEL_NAME, None, "cpu", config_overrides={"embedding_dim": 16})
    assert isinstance(adapter, PathwayHierarchicalAdapter)
    assert adapter.config_overrides["embedding_dim"] == 16


def test_gene_modules_require_explicit_source_when_not_configured():
    ctx = build_synthetic_context(seed=1, fast=True)
    overridden = dict(ctx.config)
    overridden["model"] = dict(overridden["model"])
    overridden["model"]["pathway_hierarchical_mil"] = {"gene_modules": {"path": None, "allow_synthetic_modules": False}}
    ctx.config = overridden
    with pytest.raises(PathwayModuleConfigurationError):
        build_gene_modules_for_context(ctx, ctx.preprocessing_artifact.gene_list)


def test_gene_modules_built_from_synthetic_scheme_when_explicitly_allowed():
    ctx = build_synthetic_context(seed=1, fast=True)
    modules = build_gene_modules_for_context(ctx, ctx.preprocessing_artifact.gene_list)
    assert modules.gene_names == list(ctx.preprocessing_artifact.gene_list)
    assert modules.n_modules > 0


# ══════════════════════════ Smoke-task CV integration ═════════════════════

def test_run_smoke_cv_with_pathway_model_produces_results():
    ctx = build_synthetic_context(seed=2, fast=True)
    report = run_smoke_cv(ctx, ["majority", PATHWAY_MODEL_NAME], n_folds=2, seeds=[42])
    assert PATHWAY_MODEL_NAME in report["results"]
    pathway_result = report["results"][PATHWAY_MODEL_NAME]
    assert pathway_result["n_folds_run"] == 2
    for fold in pathway_result["folds"]:
        assert "hyperparameter_search" in fold
        assert "module_fingerprint" in fold
        assert fold["module_fingerprint"]


def test_run_smoke_cv_pathway_folds_have_distinct_preprocessing_fingerprints():
    """Every fold refits its own PreprocessingArtifact from only that
    fold's training subjects (see fold_preprocessing.py) — this must hold
    for pathway_hierarchical_mil exactly like every other model, since two
    different folds' train subject sets are different."""
    ctx = build_synthetic_context(seed=3, fast=True)
    report = run_smoke_cv(ctx, [PATHWAY_MODEL_NAME], n_folds=3, seeds=[42])
    folds = report["results"][PATHWAY_MODEL_NAME]["folds"]
    fingerprints = [f["preprocessing_fingerprint"] for f in folds]
    assert len(set(fingerprints)) == len(fingerprints)


def test_run_smoke_cv_existing_models_unaffected_by_pathway_registration():
    ctx = build_synthetic_context(seed=4, fast=True)
    report = run_smoke_cv(ctx, ["majority", "logistic"], n_folds=2, seeds=[42])
    assert set(report["results"]) == {"majority", "logistic"}
    for name in ("majority", "logistic"):
        assert report["results"][name]["n_folds_run"] == 2


# ══════════════════════════ Cancer-task CV integration ═════════════════════

def test_run_cancer_cv_with_pathway_model_produces_results():
    ctx = build_synthetic_context(seed=5, fast=True)
    report = run_cancer_cv(ctx, ["prevalence", PATHWAY_MODEL_NAME], n_folds=2, seeds=[42])
    assert PATHWAY_MODEL_NAME in report["results"]
    assert report["results"][PATHWAY_MODEL_NAME]["n_folds_run"] == 2


def test_run_cancer_cv_existing_mil_models_unaffected_by_pathway_registration():
    ctx = build_synthetic_context(seed=6, fast=True)
    report = run_cancer_cv(ctx, ["prevalence", "attention_mil"], n_folds=2, seeds=[42])
    assert set(report["results"]) == {"prevalence", "attention_mil"}


# ══════════════════════════ Final development fit / OOF ═══════════════════

def test_generate_subject_oof_predictions_pathway_carries_fingerprints():
    ctx = build_synthetic_context(seed=7, fast=True)
    all_bags = list(ctx.train_bags) + list(ctx.val_bags)
    outcomes_by_subject = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes_by_subject)
    oof = generate_subject_oof_predictions(
        ctx, PATHWAY_MODEL_NAME, dev_subjects, outcomes_by_subject,
        num_cell_types=ctx.config["model"]["num_cell_types"], min_cells_per_subject=5,
        n_hvgs=ctx.preprocessing_artifact.n_hvgs, n_folds=2, n_inner_folds=2,
    )
    assert set(oof["oof_by_subject"]) == set(dev_subjects)
    for record in oof["fold_membership"]:
        if record.get("skipped_reason"):
            continue
        assert record.get("preprocessing_fingerprint")
        assert record.get("model_state_fingerprint")
        assert record.get("module_fingerprint")


def test_fit_final_candidate_on_dev_pool_pathway_excludes_no_dev_subjects():
    ctx = build_synthetic_context(seed=8, fast=True)
    all_bags = list(ctx.train_bags) + list(ctx.val_bags)
    outcomes_by_subject = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes_by_subject)
    fitted = fit_final_candidate_on_dev_pool(
        ctx, PATHWAY_MODEL_NAME, dev_subjects, outcomes_by_subject, selected_params={},
        num_cell_types=ctx.config["model"]["num_cell_types"], min_cells_per_subject=5,
        n_hvgs=ctx.preprocessing_artifact.n_hvgs,
    )
    assert fitted.kind == "mil"
    assert set(fitted.dev_subject_ids) <= set(dev_subjects)
    assert fitted.model_state_fingerprint
    # Structural non-access: no test subject id appears anywhere in the
    # inputs this function's signature accepts — dev_subjects here is
    # built entirely from ctx.train_bags/ctx.val_bags, never ctx.test_bags.
    test_subject_ids = {str(b["subject_id"]) for b in ctx.test_bags}
    assert not (set(fitted.dev_subject_ids) & test_subject_ids)


# ══════════════════════════ Ablation runner ════════════════════════════════

def test_pathway_ablation_runs_declared_variants():
    ctx = build_synthetic_context(seed=9, fast=True)
    report = run_pathway_hierarchical_ablation(ctx, n_folds=2, seeds=[42])
    assert set(report["variants"]) == set(PATHWAY_ABLATION_VARIANTS)
    assert report["development_only"] is True
    assert report["software_only"] is True
    assert report["frozen_test_data_accessed"] is False if "frozen_test_data_accessed" in report else True


def test_pathway_ablation_identical_splits_across_variants():
    """Every variant in the same fold must see the exact same train/val
    subject partition — the only thing that may differ between variants is
    the model configuration."""
    ctx = build_synthetic_context(seed=10, fast=True)
    report = run_pathway_hierarchical_ablation(ctx, n_folds=2, seeds=[42])
    by_fold = {}
    for name, r in report["results"].items():
        for f in r["folds"]:
            key = (f["seed"], f["fold"])
            by_fold.setdefault(key, []).append(f.get("train_subject_ids"))
    for key, splits in by_fold.items():
        assert all(s == splits[0] for s in splits), f"fold {key} saw different splits across variants"


def test_pathway_ablation_rejects_unknown_variant():
    ctx = build_synthetic_context(seed=11, fast=True)
    with pytest.raises(ValueError):
        run_pathway_hierarchical_ablation(ctx, variants=["not_a_real_variant"], n_folds=2, seeds=[42])
