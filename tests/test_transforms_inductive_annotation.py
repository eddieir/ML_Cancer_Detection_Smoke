"""
Regression tests for PR7 blocker 1: CellTypist cell-type annotation must be
inductive — a cell's annotation must not depend on which other cells
(especially held-out validation/test cells) are present in the same
annotate_cell_types() call.

annotate_cell_types() now defaults to majority_voting=False, CellTypist's own
per-cell prediction mode: each cell's label is a pure function of that cell's
own expression row, independent of any other row supplied alongside it. These
tests install a fake celltypist module whose annotate() mimics that contract
(label computed from each row's own values only) and prove the pipeline
actually gets an inductive result end-to-end — under the old
majority_voting=True call, a real over-clustering pass would let held-out
rows change a training cell's label; this fake would not by itself catch
that, so the key assertion is that annotate_cell_types() invokes CellTypist
with majority_voting=False and reads the corresponding non-transductive
column, not majority_voting's.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import anndata as ad


def _obs(n, subject=None):
    return pd.DataFrame({
        "is_pseudo_bulk": [False] * n,
        "subject_id": subject if subject is not None else [f"s{i}" for i in range(n)],
    })


def _install_fake_celltypist(monkeypatch, calls):
    """Fake celltypist whose per-cell label depends ONLY on that row's own
    first-gene value — never on any other row in the same call — modeling
    CellTypist's real per-cell-independent prediction contract."""

    def _label_for_row(x0: float) -> str:
        return "Basal cell" if x0 >= 0 else "Macrophage"

    class _FakeResult:
        def __init__(self, adata):
            labels = [_label_for_row(float(row[0])) for row in np.asarray(adata.X)]
            self.predicted_labels = pd.DataFrame({
                "predicted_labels": labels,
                # majority_voting column intentionally different (simulates a
                # transductive smoothing step) so a test can catch the code
                # accidentally reading this column instead.
                "majority_voting": ["Fibroblast"] * len(labels),
            })

    def fake_annotate(adata, model, majority_voting):
        calls.append({"n_cells": adata.n_obs, "majority_voting": majority_voting})
        return _FakeResult(adata)

    fake_models = types.SimpleNamespace(Model=types.SimpleNamespace(load=lambda model: object()))
    fake_celltypist = types.SimpleNamespace(annotate=fake_annotate, models=fake_models)
    monkeypatch.setitem(sys.modules, "celltypist", fake_celltypist)
    monkeypatch.setitem(sys.modules, "celltypist.models", fake_models)
    return calls


def test_annotate_cell_types_calls_celltypist_with_majority_voting_disabled(monkeypatch):
    from data.transforms import annotate_cell_types

    calls = []
    _install_fake_celltypist(monkeypatch, calls)

    n, g = 6, 4
    X = np.array([[1.0, 0, 0, 0]] * 3 + [[-1.0, 0, 0, 0]] * 3, dtype="float32")
    adata = ad.AnnData(X=X, obs=_obs(n), var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))

    out = annotate_cell_types(adata)

    assert calls[0]["majority_voting"] is False
    # Must read the per-cell "predicted_labels" column, not "majority_voting".
    assert set(out.obs["cell_type_name"].unique()) == {"Basal cell", "Macrophage"}


def test_training_cell_annotation_identical_with_or_without_held_out_cells(monkeypatch):
    """The core inductive-annotation guarantee: annotating the training
    cells alone must produce the exact same per-cell labels as annotating
    them together with additional held-out (validation/test) cells."""
    from data.transforms import annotate_cell_types

    rng = np.random.RandomState(0)
    g = 4
    train_X = rng.randn(10, g).astype("float32")
    holdout_X = rng.randn(10, g).astype("float32") * 100  # deliberately different scale

    calls_alone = []
    _install_fake_celltypist(monkeypatch, calls_alone)
    train_only = ad.AnnData(X=train_X.copy(), obs=_obs(10, subject=[f"train_{i}" for i in range(10)]),
                             var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out_alone = annotate_cell_types(train_only)
    labels_alone = out_alone.obs["cell_type_name"].tolist()

    calls_with_holdout = []
    _install_fake_celltypist(monkeypatch, calls_with_holdout)
    combined_X = np.concatenate([train_X.copy(), holdout_X])
    combined_subj = [f"train_{i}" for i in range(10)] + [f"holdout_{i}" for i in range(10)]
    combined = ad.AnnData(X=combined_X, obs=_obs(20, subject=combined_subj),
                           var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out_combined = annotate_cell_types(combined)
    labels_with_holdout = out_combined.obs["cell_type_name"].tolist()[:10]

    assert labels_alone == labels_with_holdout


def test_corrupting_held_out_expression_does_not_change_training_annotations(monkeypatch):
    from data.transforms import annotate_cell_types

    rng = np.random.RandomState(1)
    g = 4
    train_X = rng.randn(8, g).astype("float32")
    holdout_X = rng.randn(8, g).astype("float32")
    corrupted_holdout_X = holdout_X * 0 + 999999.0

    def _run(holdout):
        calls = []
        _install_fake_celltypist(monkeypatch, calls)
        combined_X = np.concatenate([train_X.copy(), holdout])
        subj = [f"train_{i}" for i in range(8)] + [f"holdout_{i}" for i in range(8)]
        combined = ad.AnnData(X=combined_X, obs=_obs(16, subject=subj),
                               var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
        out = annotate_cell_types(combined)
        return out.obs["cell_type_name"].tolist()[:8]

    assert _run(holdout_X) == _run(corrupted_holdout_X)


def test_cell_type_map_fingerprint_is_deterministic_and_not_data_derived():
    from data.transforms import cell_type_map_fingerprint

    fp1 = cell_type_map_fingerprint()
    fp2 = cell_type_map_fingerprint()
    assert fp1 == fp2
    assert isinstance(fp1, str) and len(fp1) == 64


def test_annotate_cell_types_persists_fingerprint_and_mode(monkeypatch):
    from data.transforms import annotate_cell_types, cell_type_map_fingerprint

    calls = []
    _install_fake_celltypist(monkeypatch, calls)
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = annotate_cell_types(adata)
    assert out.uns["cell_type_map_fingerprint"] == cell_type_map_fingerprint()
    assert out.uns["cell_type_annotation_mode"] == "inductive_per_cell"


def test_preprocessing_artifact_persists_cell_type_map_fingerprint():
    from data.preprocessing import PreprocessingArtifact
    from data.transforms import cell_type_map_fingerprint

    artifact = PreprocessingArtifact(
        version="1", gene_list=["g0", "g1"], gene_means=[0.0, 0.0], gene_stds=[1.0, 1.0],
        n_hvgs=2, smoke_marker_genes_forced=[], fit_n_cells=10, fit_n_subjects=2,
        cell_type_map_fingerprint=cell_type_map_fingerprint(), cell_type_annotation_mode="inductive_per_cell",
    )
    d = artifact.to_dict()
    assert d["cell_type_map_fingerprint"] == cell_type_map_fingerprint()
    reloaded = PreprocessingArtifact(**d)
    assert reloaded.cell_type_annotation_mode == "inductive_per_cell"
