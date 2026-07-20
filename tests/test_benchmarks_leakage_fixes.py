"""
Regression tests for the PR7 scientific-validity fixes: per-fold
preprocessing refit, cross-task MIL fold isolation, malignancy
ground-truth-proxy removal, OOD source isolation, one-class-fold safety,
seed-level statistical uncertainty, and ExperimentContext validation.

These are written to fail under the PREVIOUS implementation, not just pass
under the current one — see each test's docstring for what it would have
caught.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.runner import build_synthetic_context


# ─── 0. Artifact fingerprint includes cell-type/compatibility provenance ──────

def test_artifact_fingerprint_changes_when_cell_type_compatibility_differs():
    """Two artifacts with byte-identical gene scaling but different
    CellTypist/scikit-learn compatibility status must NOT collide on
    identity — a stale/incompatible cell-type annotation is a materially
    different scientific run even if the gene-level statistics match."""
    from benchmarks.fold_preprocessing import artifact_fingerprint
    from data.preprocessing import PreprocessingArtifact

    base_kwargs = dict(
        version="1", gene_list=["g0", "g1"], gene_means=[0.0, 0.0], gene_stds=[1.0, 1.0],
        n_hvgs=2, smoke_marker_genes_forced=[], fit_n_cells=10, fit_n_subjects=2,
        cell_type_map_fingerprint="a" * 64, cell_type_annotation_mode="inductive_per_cell",
        cell_type_annotation_degraded=False,
    )
    art_compatible = PreprocessingArtifact(
        **base_kwargs, cell_type_annotation_compatibility={"compatible": True, "diagnostic_override_used": False},
    )
    art_incompatible = PreprocessingArtifact(
        **base_kwargs, cell_type_annotation_compatibility={"compatible": False, "diagnostic_override_used": True},
    )
    assert artifact_fingerprint(art_compatible) != artifact_fingerprint(art_incompatible)


def test_artifact_fingerprint_changes_when_cell_type_map_fingerprint_differs():
    from benchmarks.fold_preprocessing import artifact_fingerprint
    from data.preprocessing import PreprocessingArtifact

    base_kwargs = dict(
        version="1", gene_list=["g0", "g1"], gene_means=[0.0, 0.0], gene_stds=[1.0, 1.0],
        n_hvgs=2, smoke_marker_genes_forced=[], fit_n_cells=10, fit_n_subjects=2,
        cell_type_annotation_mode="inductive_per_cell", cell_type_annotation_degraded=False,
    )
    art1 = PreprocessingArtifact(**base_kwargs, cell_type_map_fingerprint="a" * 64)
    art2 = PreprocessingArtifact(**base_kwargs, cell_type_map_fingerprint="b" * 64)
    assert artifact_fingerprint(art1) != artifact_fingerprint(art2)


# ─── 1. Per-fold preprocessing leakage ─────────────────────────────────────────

def test_fold_artifact_unaffected_by_corrupting_its_own_validation_subjects():
    """Would have failed before the fix: CV used to reuse the OUTER artifact
    (fit on ALL original-train subjects) across every fold, so corrupting an
    inner-CV-validation subject's expression couldn't possibly change
    anything (nothing was ever refit per fold) — this test only means
    something once refit_artifact_for_fold is real per-fold fitting."""
    from benchmarks.fold_preprocessing import artifact_fingerprint, refit_artifact_for_fold
    from data.splitting import grouped_kfold

    ctx = build_synthetic_context(seed=1, fast=True)
    na = ctx.normalized_adata_for_refit
    pool_subjects = sorted(set(ctx.subjects_for("train")) | set(ctx.subjects_for("val")))
    fold = grouped_kfold(np.array(pool_subjects), None, n_folds=3, seed=42)[0]

    art1 = refit_artifact_for_fold(na, fold["train"], n_hvgs=15)
    fp1 = artifact_fingerprint(art1)

    na2 = na.copy()
    val_mask = na2.obs["subject_id"].astype(str).isin(set(fold["val"])).values
    na2.X[val_mask] = 999999.0
    art2 = refit_artifact_for_fold(na2, fold["train"], n_hvgs=15)
    assert artifact_fingerprint(art2) == fp1


def test_fold_artifact_unaffected_by_corrupting_outer_test_split():
    from benchmarks.fold_preprocessing import artifact_fingerprint, refit_artifact_for_fold
    from data.splitting import grouped_kfold

    ctx = build_synthetic_context(seed=1, fast=True)
    na = ctx.normalized_adata_for_refit
    pool_subjects = sorted(set(ctx.subjects_for("train")) | set(ctx.subjects_for("val")))
    fold = grouped_kfold(np.array(pool_subjects), None, n_folds=3, seed=42)[0]
    art1 = refit_artifact_for_fold(na, fold["train"], n_hvgs=15)
    fp1 = artifact_fingerprint(art1)

    na2 = na.copy()
    test_mask = na2.obs["subject_id"].astype(str).isin(set(ctx.subjects_for("test"))).values
    na2.X[test_mask] = -999999.0
    art2 = refit_artifact_for_fold(na2, fold["train"], n_hvgs=15)
    assert artifact_fingerprint(art2) == fp1


def test_fold_artifact_fit_subjects_exactly_equal_fold_train_subjects():
    from benchmarks.fold_preprocessing import refit_artifact_for_fold
    from data.splitting import grouped_kfold

    ctx = build_synthetic_context(seed=1, fast=True)
    na = ctx.normalized_adata_for_refit
    pool_subjects = sorted(set(ctx.subjects_for("train")) | set(ctx.subjects_for("val")))
    fold = grouped_kfold(np.array(pool_subjects), None, n_folds=3, seed=42)[0]
    art = refit_artifact_for_fold(na, fold["train"], n_hvgs=15)
    assert art.fit_n_subjects == len(set(fold["train"]))


def test_cv_requires_normalized_adata_fails_clearly_when_absent():
    """Section 1: never silently fall back to the outer artifact."""
    import dataclasses
    from benchmarks.cross_validation import run_smoke_cv

    ctx = build_synthetic_context(seed=1, fast=True)
    ctx = dataclasses.replace(ctx, normalized_adata_for_refit=None)
    with pytest.raises(ValueError, match="normalized_adata_for_refit"):
        run_smoke_cv(ctx, ["majority"], n_folds=2, seeds=[42])


# ─── 2. Cross-task MIL leakage in cancer CV ────────────────────────────────────

def test_cancer_fold_phase1_pretrains_only_on_that_folds_bag_subjects():
    """Would have failed before the fix: Phase 1 pretraining always used
    context.train_cell_dataset/val_cell_dataset (the OUTER split), which
    could include cancer-CV-validation subjects in Phase 1 training."""
    from benchmarks.cross_validation import run_cancer_cv

    ctx = build_synthetic_context(seed=1, fast=True)
    result = run_cancer_cv(ctx, ["prevalence", "mean_mil"], n_folds=2, seeds=[42])
    # If it ran without raising validate_experiment_partitions' leakage error
    # (called inside run_cancer_cv for every mil model), fold isolation held.
    assert result["results"]["mean_mil"]["n_folds_run"] == 2
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


# ─── 3. Malignancy ground-truth proxy removal ──────────────────────────────────

def test_default_cancer_features_contain_no_malignancy_ground_truth():
    from benchmarks.features import build_cancer_subject_features
    ctx = build_synthetic_context(seed=1, fast=True)
    _, _, _, feature_names = build_cancer_subject_features(ctx.train_bags, num_cell_types=4)
    assert not any("malignancy" in n for n in feature_names)


def test_requesting_malignancy_known_mean_is_rejected():
    from benchmarks.features import build_cancer_subject_features
    ctx = build_synthetic_context(seed=1, fast=True)
    with pytest.raises(ValueError, match="ground-truth"):
        build_cancer_subject_features(
            ctx.train_bags, num_cell_types=4,
            feature_groups=("gene_mean", "malignancy_known_mean"),
        )


def test_oof_malignancy_missing_subject_prediction_rejected():
    from benchmarks.features import build_cancer_subject_features
    ctx = build_synthetic_context(seed=1, fast=True)
    with pytest.raises(ValueError, match="no out-of-fold"):
        build_cancer_subject_features(ctx.train_bags, num_cell_types=4, oof_malignancy={})


def test_validate_oof_predictions_rejects_in_fold_and_duplicate_and_missing():
    from benchmarks.features import OOFMalignancyPrediction, validate_oof_predictions

    good = [
        OOFMalignancyPrediction("s1", 0.2, "rf", ["s2", "s3"], "fp1", 42),
        OOFMalignancyPrediction("s2", 0.5, "rf", ["s1", "s3"], "fp1", 42),
    ]
    out = validate_oof_predictions(good, ["s1", "s2"])
    assert out == {"s1": 0.2, "s2": 0.5}

    with pytest.raises(ValueError, match="missing"):
        validate_oof_predictions(good, ["s1", "s2", "s3"])

    dup = good + [OOFMalignancyPrediction("s1", 0.9, "rf", ["s2"], "fp1", 42)]
    with pytest.raises(ValueError, match="duplicate"):
        validate_oof_predictions(dup, ["s1", "s2"])

    in_fold = [OOFMalignancyPrediction("s1", 0.2, "rf", ["s1", "s2"], "fp1", 42)]
    with pytest.raises(ValueError, match="in-fold"):
        validate_oof_predictions(in_fold, ["s1"])


# ─── 4. OOD source isolation ────────────────────────────────────────────────────

def test_ood_missing_source_metadata_defaults_not_comparable():
    from benchmarks.ood import run_leave_one_source_out
    ctx = build_synthetic_context(seed=1, fast=True)
    result = run_leave_one_source_out(ctx, ["majority"])  # no species_by_source given
    assert all(v["status"] == "NOT_COMPARABLE" for v in result.values())


def test_ood_species_mismatch_rejected():
    from benchmarks.ood import run_leave_one_source_out
    ctx = build_synthetic_context(seed=1, fast=True)
    result = run_leave_one_source_out(
        ctx, ["majority"], species_by_source={"sourceA": "human", "sourceB": "mouse"},
        reference_species="human",
    )
    assert result["sourceB"]["status"] == "NOT_COMPARABLE"
    assert "species" in result["sourceB"]["reason"]


def test_ood_evaluated_source_never_saw_held_out_cells_during_refit():
    from benchmarks.fold_preprocessing import artifact_fingerprint, refit_artifact_for_fold

    ctx = build_synthetic_context(seed=1, fast=True)
    na = ctx.normalized_adata_for_refit
    pool_subjects = sorted(set(ctx.subjects_for("train")) | set(ctx.subjects_for("val")))
    obs = na.obs
    subj = obs["subject_id"].astype(str)
    pool_mask = subj.isin(pool_subjects).values
    source = obs["source"].astype(str).values[pool_mask]
    subj_pool = subj.values[pool_mask]

    held_out = sorted(set(subj_pool[source == "sourceA"].tolist()))
    train_subj = sorted(set(subj_pool.tolist()) - set(held_out))
    art1 = refit_artifact_for_fold(na, train_subj, n_hvgs=15)
    fp1 = artifact_fingerprint(art1)

    na2 = na.copy()
    held_mask = na2.obs["subject_id"].astype(str).isin(set(held_out)).values
    na2.X[held_mask] = 12345.0
    art2 = refit_artifact_for_fold(na2, train_subj, n_hvgs=15)
    assert artifact_fingerprint(art2) == fp1


# ─── 5. One-class folds / probability-column mapping ───────────────────────────

def test_single_class_training_fold_does_not_crash_any_cancer_baseline():
    from benchmarks.baselines import CANCER_BASELINES, positive_class_proba
    X = np.random.rand(12, 4)
    for name, cls in CANCER_BASELINES.items():
        for const in (0, 1):
            y = np.full(12, const)
            model = cls().fit(X, y, seed=42)
            proba = positive_class_proba(model, X)
            assert (proba == const).all(), name


def test_positive_class_proba_maps_via_classes_not_column_index():
    from sklearn.linear_model import LogisticRegression
    from benchmarks.baselines import Baseline, positive_class_proba

    class _Wrapped(Baseline):
        def fit(self, X, y, seed=42):
            self.model = LogisticRegression().fit(X, y)
            self.classes_ = self.model.classes_
            return self

    X = np.random.rand(30, 3)
    y = np.random.randint(0, 2, 30)
    model = _Wrapped().fit(X, y)
    expected = model.model.predict_proba(X)[:, list(model.model.classes_).index(1)]
    assert np.allclose(positive_class_proba(model, X), expected)


# ─── 6. Statistically correct uncertainty unit ─────────────────────────────────

def test_aggregate_metric_by_seed_uses_seed_as_resampling_unit():
    from benchmarks.metrics import aggregate_metric_by_seed
    values = [0.5, 0.6, 0.7, 0.8, 0.4, 0.9]
    seeds  = [42, 42, 43, 43, 44, 44]
    result = aggregate_metric_by_seed(values, seeds)
    assert result["resampling_unit"] == "seed"
    assert result["n_valid"] == 3  # one mean per seed, not one per fold


def test_aggregate_metric_by_seed_undefined_with_fewer_than_two_seeds():
    from benchmarks.metrics import aggregate_metric_by_seed
    result = aggregate_metric_by_seed([0.5, 0.6], [42, 42])
    assert result["ci"] is None
    assert "note" in result


def test_cv_reports_both_fold_level_and_seed_level_aggregation():
    from benchmarks.cross_validation import run_smoke_cv
    ctx = build_synthetic_context(seed=1, fast=True)
    result = run_smoke_cv(ctx, ["majority"], n_folds=2, seeds=[42, 43])
    r = result["results"]["majority"]
    assert r["subject_weighted_macro_f1"]["resampling_unit"] == "fold"
    assert r["subject_weighted_macro_f1_by_seed"]["resampling_unit"] == "seed"


# ─── 7. ExperimentContext validation ────────────────────────────────────────────

def test_from_pipeline_result_rejects_gene_count_mismatch():
    from benchmarks.context import ExperimentContext
    from data.label_mapping import identity_label_mapping
    from data.preprocessing import PreprocessingArtifact
    from data.splitting import SplitManifest
    from train import CellLevelDataset

    n = 20
    all_ids = np.array([f"s{i}" for i in range(n)], dtype=object)
    ds = CellLevelDataset(
        np.random.rand(n, 5).astype("float32"), np.zeros(n, dtype=int),
        np.zeros(n, dtype="float32"), np.zeros(n, dtype=int),
        subject_ids=all_ids, is_pseudo_bulk=np.zeros(n, dtype=bool),
    )
    train_ids, val_ids, test_ids = all_ids[:14].tolist(), all_ids[14:17].tolist(), all_ids[17:].tolist()
    artifact = PreprocessingArtifact(
        version="1", gene_list=[f"g{i}" for i in range(9)],  # deliberately wrong count
        gene_means=[0.0] * 9, gene_stds=[1.0] * 9, n_hvgs=9,
        smoke_marker_genes_forced=[], fit_n_cells=14, fit_n_subjects=14,
    )
    manifest = SplitManifest(seed=1, train_subjects=train_ids, val_subjects=val_ids, test_subjects=test_ids)
    result = {
        "train_cell_dataset": ds.subset_by_subjects(train_ids),
        "val_cell_dataset": ds.subset_by_subjects(val_ids),
        "test_cell_dataset": ds.subset_by_subjects(test_ids),
        "train_bags": [], "val_bags": [], "test_bags": [],
        "split_manifest": manifest, "preprocessing_artifact": artifact,
        "label_mapping": identity_label_mapping(), "rare_class_report": {},
        "label_provenance_report": {}, "transductive_batch_correction": False,
    }
    with pytest.raises(ValueError, match="genes"):
        ExperimentContext.from_pipeline_result(result, config={})


def test_from_pipeline_result_deep_copies_config_snapshot(monkeypatch):
    """Mutating the caller's config dict after building a context must not
    retroactively change the context's own config snapshot."""
    import sys as _sys
    import tempfile
    import types

    from preprocess import run_pipeline_split_aware
    from benchmarks.context import ExperimentContext
    sys.path.insert(0, str(Path(__file__).parents[0]))
    from test_preprocess_split_aware import _synthetic_h5ad_consistent_labels

    # This test exercises config-snapshot deep-copy semantics, not CellTypist
    # compatibility — install a fake celltypist with no version mismatch so
    # a real (non-degraded) ExperimentContext can actually be constructed in
    # this environment, where the real installed model is version-
    # incompatible (see data/transforms.py::CellTypistCompatibilityError and
    # tests/test_transforms_inductive_annotation.py's own fake-celltypist
    # tests for the exhaustively-covered compatibility behavior itself).
    def fake_annotate(adata, model, majority_voting):
        n = adata.n_obs
        import pandas as _pd
        return types.SimpleNamespace(predicted_labels=_pd.DataFrame({
            "predicted_labels": ["Basal cell"] * n, "majority_voting": ["Basal cell"] * n,
        }))
    fake_models = types.SimpleNamespace(Model=types.SimpleNamespace(load=lambda model: object()))
    fake_celltypist = types.SimpleNamespace(annotate=fake_annotate, models=fake_models)
    monkeypatch.setitem(_sys.modules, "celltypist", fake_celltypist)
    monkeypatch.setitem(_sys.modules, "celltypist.models", fake_models)

    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad_consistent_labels(h5ad)
        config = {
            "data": {"scrna_sources": [(h5ad, "cigarette", "donor_id")], "n_hvgs": 50,
                      "min_cells_per_subject": 5, "out_dir": str(Path(tmp) / "processed")},
            "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
        }
        result = run_pipeline_split_aware(config)
        ctx = ExperimentContext.from_pipeline_result(result, config)
        config["split"]["seed"] = 999999  # mutate the caller's own dict after the fact
        assert ctx.config["split"]["seed"] == 1
