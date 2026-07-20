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


def _install_broken_celltypist(monkeypatch):
    def fake_annotate(adata, model, majority_voting):
        raise RuntimeError("simulated CellTypist model download failure")
    fake_models = types.SimpleNamespace(Model=types.SimpleNamespace(load=lambda model: object()))
    fake_celltypist = types.SimpleNamespace(annotate=fake_annotate, models=fake_models)
    monkeypatch.setitem(sys.modules, "celltypist", fake_celltypist)
    monkeypatch.setitem(sys.modules, "celltypist.models", fake_models)


def test_celltypist_failure_raises_by_default(monkeypatch):
    from data.transforms import CellTypeAnnotationError, annotate_cell_types

    _install_broken_celltypist(monkeypatch)
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    with pytest.raises(CellTypeAnnotationError, match="simulated CellTypist model download failure"):
        annotate_cell_types(adata)


def test_celltypist_failure_with_explicit_fallback_stamps_degraded(monkeypatch):
    from data.transforms import annotate_cell_types

    _install_broken_celltypist(monkeypatch)
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = annotate_cell_types(adata, allow_diagnostic_fallback=True)

    assert out.uns["cell_type_annotation_degraded"] is True
    assert out.uns["cell_type_annotation_mode"] == "diagnostic_fallback"
    assert "cell_type_fallback_reason" in out.uns
    assert set(out.obs["cell_type_name"].unique()) == {"epithelial"}
    assert (out.obs["cell_type_id"] == out.obs["cell_type_id"].iloc[0]).all()


def test_successful_annotation_is_not_marked_degraded(monkeypatch):
    from data.transforms import annotate_cell_types

    calls = []
    _install_fake_celltypist(monkeypatch, calls)
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = annotate_cell_types(adata)
    assert out.uns["cell_type_annotation_degraded"] is False


def test_context_validation_rejects_degraded_annotation_in_real_mode():
    from benchmarks.context import ExperimentContext
    from data.label_mapping import identity_label_mapping
    from data.preprocessing import PreprocessingArtifact
    from data.splitting import SplitManifest
    from train import CellLevelDataset

    n_genes = 3
    train_subj = ["s0", "s1"]
    val_subj = ["s2"]
    test_subj = ["s3"]
    ds = CellLevelDataset(
        gene_matrix=np.zeros((2, n_genes), dtype="float32"), smoke_labels=np.array([0, 1]),
        malignancy_labels=np.zeros(2, dtype="float32"), cell_type_ids=np.zeros(2, dtype=np.int64),
        subject_ids=np.array(train_subj, dtype=object), dataset_source=np.array(["a", "a"], dtype=object),
        is_pseudo_bulk=np.zeros(2, dtype=bool),
    )
    val_ds = CellLevelDataset(
        gene_matrix=np.zeros((1, n_genes), dtype="float32"), smoke_labels=np.array([0]),
        malignancy_labels=np.zeros(1, dtype="float32"), cell_type_ids=np.zeros(1, dtype=np.int64),
        subject_ids=np.array(val_subj, dtype=object), dataset_source=np.array(["a"], dtype=object),
        is_pseudo_bulk=np.zeros(1, dtype=bool),
    )
    test_ds = CellLevelDataset(
        gene_matrix=np.zeros((1, n_genes), dtype="float32"), smoke_labels=np.array([1]),
        malignancy_labels=np.zeros(1, dtype="float32"), cell_type_ids=np.zeros(1, dtype=np.int64),
        subject_ids=np.array(test_subj, dtype=object), dataset_source=np.array(["a"], dtype=object),
        is_pseudo_bulk=np.zeros(1, dtype=bool),
    )
    artifact = PreprocessingArtifact(
        version="1", gene_list=[f"g{i}" for i in range(n_genes)], gene_means=[0.0] * n_genes,
        gene_stds=[1.0] * n_genes, n_hvgs=n_genes, smoke_marker_genes_forced=[],
        fit_n_cells=2, fit_n_subjects=2, cell_type_annotation_degraded=True,
    )
    manifest = SplitManifest(seed=1, train_subjects=train_subj, val_subjects=val_subj, test_subjects=test_subj)
    mapping = identity_label_mapping({0: "cigarette", 1: "vape"})
    result = {
        "train_cell_dataset": ds, "val_cell_dataset": val_ds, "test_cell_dataset": test_ds,
        "train_bags": [], "val_bags": [], "test_bags": [], "split_manifest": manifest,
        "preprocessing_artifact": artifact, "label_mapping": mapping, "rare_class_report": {},
        "label_provenance_report": {}, "transductive_batch_correction": False,
    }
    with pytest.raises(ValueError, match="cell_type_annotation_degraded"):
        ExperimentContext.from_pipeline_result(result, config={})


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


# ─── scikit-learn / CellTypist pretrained-model version compatibility ────────

def _install_version_mismatched_celltypist(monkeypatch, calls):
    """Fake celltypist whose Model.load() emits the exact
    InconsistentVersionWarning scikit-learn raises when unpickling an
    estimator serialized under a different scikit-learn version — mirrors
    the real Immune_All_Low.pkl compatibility situation without depending on
    a downloaded model file."""
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    def fake_load(model):
        warnings.warn(
            InconsistentVersionWarning(
                estimator_name="LogisticRegression",
                current_sklearn_version="1.9.0",
                original_sklearn_version="0.24.1",
            )
        )
        return object()

    labels_fn = lambda x0: "Basal cell" if x0 >= 0 else "Macrophage"

    class _FakeResult:
        def __init__(self, adata):
            labels = [labels_fn(float(row[0])) for row in np.asarray(adata.X)]
            self.predicted_labels = pd.DataFrame({
                "predicted_labels": labels,
                "majority_voting": ["Fibroblast"] * len(labels),
            })

    def fake_annotate(adata, model, majority_voting):
        calls.append({"n_cells": adata.n_obs, "majority_voting": majority_voting})
        return _FakeResult(adata)

    fake_models = types.SimpleNamespace(Model=types.SimpleNamespace(load=fake_load))
    fake_celltypist = types.SimpleNamespace(annotate=fake_annotate, models=fake_models)
    monkeypatch.setitem(sys.modules, "celltypist", fake_celltypist)
    monkeypatch.setitem(sys.modules, "celltypist.models", fake_models)


def test_sklearn_version_mismatch_raises_by_default_real_mode(monkeypatch):
    """Real (allow_diagnostic_fallback=False, the default) preprocessing
    must fail closed on a scikit-learn/CellTypist version mismatch — it must
    never silently continue with a version-inconsistent prediction."""
    from data.transforms import CellTypistCompatibilityError, annotate_cell_types

    calls = []
    _install_version_mismatched_celltypist(monkeypatch, calls)
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    with pytest.raises(CellTypistCompatibilityError, match="0.24.1"):
        annotate_cell_types(adata)


def test_sklearn_version_mismatch_error_is_not_wrapped_in_generic_annotation_error(monkeypatch):
    """The specific, remediation-bearing CellTypistCompatibilityError must
    reach the caller undisturbed — never hidden inside the generic
    CellTypeAnnotationError the broad except-Exception fallback raises."""
    from data.transforms import CellTypeAnnotationError, annotate_cell_types

    calls = []
    _install_version_mismatched_celltypist(monkeypatch, calls)
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    with pytest.raises(Exception) as exc_info:
        annotate_cell_types(adata)
    assert not isinstance(exc_info.value, CellTypeAnnotationError)


def test_sklearn_version_mismatch_cannot_be_bypassed_by_a_generic_env_var(monkeypatch):
    """There must be no generic environment variable that weakens real-mode
    enforcement — CELLTYPIST_STRICT_SKLEARN_COMPAT no longer exists at all,
    real mode fails closed unconditionally."""
    from data.transforms import CellTypistCompatibilityError, annotate_cell_types

    calls = []
    _install_version_mismatched_celltypist(monkeypatch, calls)
    monkeypatch.setenv("CELLTYPIST_STRICT_SKLEARN_COMPAT", "0")
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    with pytest.raises(CellTypistCompatibilityError):
        annotate_cell_types(adata)


def test_sklearn_version_mismatch_still_emits_the_warning_before_raising(monkeypatch):
    """The InconsistentVersionWarning must never be silently swallowed, even
    though it now also causes a hard failure in real mode."""
    from data.transforms import CellTypistCompatibilityError, annotate_cell_types
    from sklearn.exceptions import InconsistentVersionWarning

    calls = []
    _install_version_mismatched_celltypist(monkeypatch, calls)
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    with pytest.warns(InconsistentVersionWarning):
        with pytest.raises(CellTypistCompatibilityError):
            annotate_cell_types(adata)


def test_matching_versions_do_not_raise_in_real_mode(monkeypatch):
    from data.transforms import annotate_cell_types

    calls = []
    _install_fake_celltypist(monkeypatch, calls)  # no version-mismatch warning
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = annotate_cell_types(adata)
    assert out.uns["cell_type_annotation_degraded"] is False
    assert out.uns["cell_type_annotation_compatibility"]["compatible"] is True


def test_diagnostic_fallback_tolerates_mismatch_but_stamps_degraded(monkeypatch):
    """allow_diagnostic_fallback=True may proceed past a version mismatch,
    but the result must be unambiguously marked degraded/non-scientific —
    never indistinguishable from a genuinely compatible annotation."""
    from data.transforms import annotate_cell_types

    calls = []
    _install_version_mismatched_celltypist(monkeypatch, calls)
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = annotate_cell_types(adata, allow_diagnostic_fallback=True)
    assert out.uns["cell_type_annotation_degraded"] is True
    assert out.uns["cell_type_annotation_compatibility"]["compatible"] is False
    assert out.uns["cell_type_annotation_compatibility"]["diagnostic_override_used"] is True
    assert "cell_type_fallback_reason" in out.uns


def test_diagnostic_fallback_with_matching_versions_is_not_marked_degraded(monkeypatch):
    """allow_diagnostic_fallback=True must not itself force a degraded
    stamp when there was no actual mismatch to tolerate."""
    from data.transforms import annotate_cell_types

    calls = []
    _install_fake_celltypist(monkeypatch, calls)
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = annotate_cell_types(adata, allow_diagnostic_fallback=True)
    assert out.uns["cell_type_annotation_degraded"] is False
    assert out.uns["cell_type_annotation_compatibility"]["diagnostic_override_used"] is False


def test_degraded_compatibility_override_is_rejected_by_real_context():
    """A diagnostic-mode annotation that tolerated a real version mismatch
    must be rejected by real (non-synthetic) ExperimentContext construction —
    the existing degraded-provenance guard already covers this, since the
    compatibility override always sets cell_type_annotation_degraded=True."""
    from benchmarks.context import ExperimentContext
    from data.label_mapping import identity_label_mapping
    from data.preprocessing import PreprocessingArtifact
    from data.splitting import SplitManifest
    from train import CellLevelDataset

    n_genes = 3
    ds = CellLevelDataset(
        gene_matrix=np.zeros((2, n_genes), dtype="float32"), smoke_labels=np.array([0, 1]),
        malignancy_labels=np.zeros(2, dtype="float32"), cell_type_ids=np.zeros(2, dtype=np.int64),
        subject_ids=np.array(["s0", "s1"], dtype=object), dataset_source=np.array(["a", "a"], dtype=object),
        is_pseudo_bulk=np.zeros(2, dtype=bool),
    )
    val_ds = CellLevelDataset(
        gene_matrix=np.zeros((1, n_genes), dtype="float32"), smoke_labels=np.array([0]),
        malignancy_labels=np.zeros(1, dtype="float32"), cell_type_ids=np.zeros(1, dtype=np.int64),
        subject_ids=np.array(["s2"], dtype=object), dataset_source=np.array(["a"], dtype=object),
        is_pseudo_bulk=np.zeros(1, dtype=bool),
    )
    test_ds = CellLevelDataset(
        gene_matrix=np.zeros((1, n_genes), dtype="float32"), smoke_labels=np.array([1]),
        malignancy_labels=np.zeros(1, dtype="float32"), cell_type_ids=np.zeros(1, dtype=np.int64),
        subject_ids=np.array(["s3"], dtype=object), dataset_source=np.array(["a"], dtype=object),
        is_pseudo_bulk=np.zeros(1, dtype=bool),
    )
    artifact = PreprocessingArtifact(
        version="1", gene_list=[f"g{i}" for i in range(n_genes)], gene_means=[0.0] * n_genes,
        gene_stds=[1.0] * n_genes, n_hvgs=n_genes, smoke_marker_genes_forced=[],
        fit_n_cells=2, fit_n_subjects=2, cell_type_annotation_degraded=True,
        cell_type_annotation_compatibility={"compatible": False, "diagnostic_override_used": True},
    )
    manifest = SplitManifest(seed=1, train_subjects=["s0", "s1"], val_subjects=["s2"], test_subjects=["s3"])
    mapping = identity_label_mapping({0: "cigarette", 1: "vape"})
    result = {
        "train_cell_dataset": ds, "val_cell_dataset": val_ds, "test_cell_dataset": test_ds,
        "train_bags": [], "val_bags": [], "test_bags": [], "split_manifest": manifest,
        "preprocessing_artifact": artifact, "label_mapping": mapping, "rare_class_report": {},
        "label_provenance_report": {}, "transductive_batch_correction": False,
    }
    with pytest.raises(ValueError, match="cell_type_annotation_degraded"):
        ExperimentContext.from_pipeline_result(result, config={})


def test_compatibility_provenance_persisted_for_a_mismatched_model(monkeypatch):
    """cell_type_annotation_compatibility must record the actual mismatch
    facts (not a fabricated/omitted value) so a reproducibility artifact can
    show exactly which CellTypist/scikit-learn combination produced this
    run's cell-type labels."""
    import sklearn as _sklearn_pkg
    from data.transforms import annotate_cell_types

    calls = []
    _install_version_mismatched_celltypist(monkeypatch, calls)
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    # Real mode now fails closed on the mismatch (see
    # test_sklearn_version_mismatch_raises_by_default_real_mode); use the
    # explicit diagnostic override to still inspect the persisted
    # compatibility provenance for a tolerated mismatch.
    out = annotate_cell_types(adata, allow_diagnostic_fallback=True)
    compat = out.uns["cell_type_annotation_compatibility"]
    assert compat["compatible"] is False
    assert compat["serialized_sklearn_versions"] == ["0.24.1"]
    assert compat["runtime_sklearn_version"] == _sklearn_pkg.__version__
    assert compat["model_name"] == "Immune_All_Low.pkl"


def test_compatibility_provenance_persisted_for_a_matching_model(monkeypatch):
    from data.transforms import annotate_cell_types

    calls = []
    _install_fake_celltypist(monkeypatch, calls)  # no version-mismatch warning
    n, g = 4, 3
    adata = ad.AnnData(X=np.ones((n, g), dtype="float32"), obs=_obs(n),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = annotate_cell_types(adata)
    compat = out.uns["cell_type_annotation_compatibility"]
    assert compat["compatible"] is True
    assert compat["serialized_sklearn_versions"] == []


def test_compatibility_provenance_round_trips_through_preprocessing_artifact():
    """fit_preprocessing must copy cell_type_annotation_compatibility from
    adata.uns onto the PreprocessingArtifact, exactly like the other
    cell-type provenance fields (see data/preprocessing.py::fit_preprocessing)."""
    from data.preprocessing import fit_preprocessing

    n, g = 6, 3
    obs = _obs(n)
    obs["subject_id"] = [f"s{i}" for i in range(n)]
    adata = ad.AnnData(X=np.random.rand(n, g).astype("float32"), obs=obs,
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    fake_compat = {
        "celltypist_version": "1.7.1", "model_name": "Immune_All_Low.pkl",
        "runtime_sklearn_version": "1.9.0", "serialized_sklearn_versions": ["0.24.1"],
        "compatible": False,
    }
    adata.uns["cell_type_annotation_compatibility"] = fake_compat
    adata.uns["cell_type_map_fingerprint"] = "a" * 64
    adata.uns["cell_type_annotation_mode"] = "inductive_per_cell"
    adata.uns["cell_type_annotation_degraded"] = False
    artifact = fit_preprocessing(adata, {f"s{i}" for i in range(n)}, n_hvgs=g)
    assert artifact.cell_type_annotation_compatibility == fake_compat
