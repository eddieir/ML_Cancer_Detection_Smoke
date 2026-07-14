"""
Regression tests for PR7 requirement 1: cell-type annotations must survive
CV fold reconstruction instead of being silently reset to the cell_type_id=0
placeholder.

Root cause: run_pipeline_split_aware() used to capture
normalized_adata_for_refit (the AnnData every benchmark CV fold and OOD
evaluation is rebuilt from) BEFORE calling annotate_cell_types(), so every
fold-reconstructed cell got the pre-annotation placeholder (0) regardless of
its real CellTypist-predicted type. Fixed by moving the annotate_cell_types()
call to run exactly once, before the snapshot is taken (see preprocess.py).

These tests would fail under the previous ordering — see each docstring.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[0]))
from test_preprocess_split_aware import _synthetic_h5ad_consistent_labels as _synthetic_h5ad


# ─── 1. annotate_cell_types runs exactly once, before the refit snapshot ───────

def test_annotate_cell_types_runs_once_before_refit_snapshot(monkeypatch):
    """Would fail under the previous ordering: the fake annotator's
    subject-derived, multi-valued cell_type_id would be overwritten back to
    0 by whatever ran the export step, or the snapshot would still show all
    zeros because it was captured before this function ran."""
    import preprocess as preprocess_mod

    call_count = {"n": 0}
    real_annotate = preprocess_mod.annotate_cell_types

    def fake_annotate(adata, allow_diagnostic_fallback=False):
        call_count["n"] += 1
        # Deterministic, subject-derived, multi-class fake annotation —
        # stands in for a real CellTypist prediction without depending on
        # celltypist actually classifying synthetic noise meaningfully.
        subj = adata.obs["subject_id"].astype(str)
        adata.obs["cell_type_id"] = subj.map(lambda s: hash(s) % 4).astype(int)
        return adata

    monkeypatch.setattr(preprocess_mod, "annotate_cell_types", fake_annotate)

    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad(h5ad)
        result = preprocess_mod.run_pipeline_split_aware({
            "data": {"scrna_sources": [(h5ad, "cigarette", "donor_id")], "n_hvgs": 50,
                      "min_cells_per_subject": 5, "out_dir": str(Path(tmp) / "processed")},
            "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
        })

    assert call_count["n"] == 1, "annotate_cell_types must run exactly once, not per fold/split"
    snapshot_types = set(result["normalized_adata_for_refit"].obs["cell_type_id"].unique().tolist())
    assert len(snapshot_types) > 1, (
        "normalized_adata_for_refit still shows a single cell_type_id value — "
        "the snapshot was taken before annotation ran"
    )
    # The exported/split cell datasets must show the SAME annotation, not a
    # second, independently-computed one.
    exported_types = set(np.unique(result["cell_data"]["cell_type_ids"]).tolist())
    assert snapshot_types == exported_types


# ─── 2. Multiple real cell types survive fold reconstruction ──────────────────

def _hand_built_normalized_adata(n_subjects=8, cells_per_subject=15, g=10, n_ct=4, seed=0):
    """Small AnnData with obs["cell_type_id"] varied deterministically per
    subject — stands in for a post-annotate_cell_types snapshot without
    depending on celltypist producing diverse output on random noise."""
    rng = np.random.RandomState(seed)
    subject_ids, smoke_types, cell_types, sources = [], [], [], []
    for i in range(n_subjects):
        ct = i % n_ct
        subject_ids += [f"sub_{i}"] * cells_per_subject
        smoke_types += [i % 2] * cells_per_subject
        cell_types += [ct] * cells_per_subject
        sources += ["sourceA"] * cells_per_subject
    n = len(subject_ids)
    X = rng.randn(n, g).astype("float32")
    obs = pd.DataFrame({
        "subject_id": subject_ids, "smoke_type": smoke_types, "cell_type_id": cell_types,
        "malignancy": 0.0, "malignancy_known": False, "exposure_dose": -1.0, "source": sources,
    })
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)])), subject_ids


def test_multiple_cell_types_survive_fold_reconstruction():
    from benchmarks.fold_preprocessing import fold_train_val_datasets
    from benchmarks.context import ExperimentContext
    import dataclasses

    na, subject_ids = _hand_built_normalized_adata()
    all_subjects = sorted(set(subject_ids))
    fold_train, fold_val = all_subjects[:6], all_subjects[6:]

    # Minimal stand-in context carrying only what fold_train_val_datasets needs.
    class _Ctx:
        normalized_adata_for_refit = na
        class preprocessing_artifact:
            n_hvgs = 10

    artifact, train_ds, val_ds = fold_train_val_datasets(_Ctx(), fold_train, fold_val, n_hvgs=10)
    train_types = set(train_ds.ctype.numpy().tolist())
    val_types = set(val_ds.ctype.numpy().tolist())
    assert len(train_types) > 1, "fold reconstruction collapsed cell types to a single value"
    assert len(val_types) > 1


def test_cell_type_ids_stable_across_different_folds():
    """A given subject's cells must carry the SAME cell_type_id whether that
    subject lands in fold A's train set or fold B's train set — the mapping
    is a deterministic property of the cell, not something refit per fold."""
    from benchmarks.fold_preprocessing import build_fold_cell_dataset, refit_artifact_for_fold

    na, subject_ids = _hand_built_normalized_adata()
    all_subjects = sorted(set(subject_ids))

    artifact_a = refit_artifact_for_fold(na, all_subjects[:6], n_hvgs=10)
    ds_a = build_fold_cell_dataset(na, artifact_a, ["sub_0"])
    artifact_b = refit_artifact_for_fold(na, all_subjects[2:], n_hvgs=10)
    ds_b = build_fold_cell_dataset(na, artifact_b, ["sub_0"])

    assert set(ds_a.ctype.numpy().tolist()) == set(ds_b.ctype.numpy().tolist()) == {0}


def test_val_expression_corruption_does_not_change_fold_cell_types():
    from benchmarks.fold_preprocessing import build_fold_cell_dataset, refit_artifact_for_fold

    na, subject_ids = _hand_built_normalized_adata()
    all_subjects = sorted(set(subject_ids))
    fold_train, fold_val = all_subjects[:6], all_subjects[6:]

    artifact = refit_artifact_for_fold(na, fold_train, n_hvgs=10)
    ds_before = build_fold_cell_dataset(na, artifact, fold_train)
    types_before = ds_before.ctype.numpy().copy()

    na2 = na.copy()
    val_mask = na2.obs["subject_id"].astype(str).isin(set(fold_val)).values
    na2.X[val_mask] = 999999.0
    artifact2 = refit_artifact_for_fold(na2, fold_train, n_hvgs=10)
    ds_after = build_fold_cell_dataset(na2, artifact2, fold_train)
    assert np.array_equal(types_before, ds_after.ctype.numpy())


# ─── 3. Subject features / MIL bags receive preserved annotations ─────────────

def test_subject_features_reflect_preserved_cell_type_diversity():
    from benchmarks.fold_preprocessing import bags_from_fold_cell_dataset, fold_train_val_datasets
    from benchmarks.features import build_cancer_subject_features

    na, subject_ids = _hand_built_normalized_adata(n_subjects=8, cells_per_subject=15)
    all_subjects = sorted(set(subject_ids))

    class _Ctx:
        normalized_adata_for_refit = na
        class preprocessing_artifact:
            n_hvgs = 10

    artifact, train_ds, val_ds = fold_train_val_datasets(_Ctx(), all_subjects[:6], all_subjects[6:], n_hvgs=10)
    outcomes = {s: i % 2 for i, s in enumerate(all_subjects)}
    bags = bags_from_fold_cell_dataset(train_ds, outcomes, min_cells_per_subject=5)
    assert len(bags) > 0
    # Bags must carry the real, varied cell_type_ids, not all-zero placeholders.
    all_bag_ct = np.concatenate([b["cell_type_ids"] for b in bags])
    assert len(set(all_bag_ct.tolist())) > 1

    X, y, subj, feature_names = build_cancer_subject_features(bags, num_cell_types=4)
    ct_prop_idx = [i for i, n in enumerate(feature_names) if n.startswith("cell_type_prop")]
    assert ct_prop_idx, "cell_type_prop features missing from feature_names"
    ct_props = X[:, ct_prop_idx]
    # With real per-subject cell-type diversity, proportions must not all be
    # identical across subjects (which is what a constant/placeholder
    # cell_type_id=0 for everyone would produce).
    assert not np.allclose(ct_props, ct_props[0])
