"""data/preprocessing.py — leakage-free fit/transform preprocessing artifact."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.preprocessing import (
    PreprocessingArtifact,
    apply_preprocessing,
    fit_preprocessing,
    verify_compatible,
    verify_input_matrix,
)


def _adata(n_subjects=10, cells_per_subject=20, n_genes=50, seed=0, offset=0.0):
    rng = np.random.default_rng(seed)
    subject_ids = []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * cells_per_subject
    n = len(subject_ids)
    X = (rng.random((n, n_genes)) + offset).astype("float32")
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame({
        "subject_id": subject_ids,
        "batch": ["source_0"] * n,
    }, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))


def test_fit_uses_only_train_cells_statistics():
    """Perturbing non-train (val/test) cells' values must not change the
    fitted artifact — this is the core no-leakage guarantee."""
    adata = _adata(n_subjects=10)
    train_subjects = {f"sub_{i}" for i in range(6)}  # 6/10 subjects = train

    artifact1 = fit_preprocessing(adata, train_subjects, n_hvgs=20, batch_key=None)

    # Wildly perturb the non-train subjects' expression values.
    perturbed = adata.copy()
    non_train_mask = ~perturbed.obs["subject_id"].isin(train_subjects).values
    perturbed.X[non_train_mask] = perturbed.X[non_train_mask] * 1000 + 500

    artifact2 = fit_preprocessing(perturbed, train_subjects, n_hvgs=20, batch_key=None)

    assert artifact1.gene_means == artifact2.gene_means
    assert artifact1.gene_stds == artifact2.gene_stds
    assert artifact1.gene_list == artifact2.gene_list


def test_fit_raises_when_no_cells_match_train_subjects():
    adata = _adata(n_subjects=5)
    with pytest.raises(ValueError):
        fit_preprocessing(adata, {"nonexistent_subject"}, n_hvgs=10, batch_key=None)


def test_gene_list_is_stable_and_ordered():
    adata = _adata(n_subjects=10)
    train_subjects = {f"sub_{i}" for i in range(6)}
    a1 = fit_preprocessing(adata, train_subjects, n_hvgs=15, batch_key=None)
    a2 = fit_preprocessing(adata, train_subjects, n_hvgs=15, batch_key=None)
    assert a1.gene_list == a2.gene_list
    assert len(a1.gene_list) == len(set(a1.gene_list))  # no duplicates


def test_apply_preprocessing_matches_manual_zscore():
    adata = _adata(n_subjects=10, n_genes=10)
    train_subjects = {f"sub_{i}" for i in range(6)}
    artifact = fit_preprocessing(adata, train_subjects, n_hvgs=5, batch_key=None)

    transformed = apply_preprocessing(adata, artifact)
    assert list(transformed.var_names) == artifact.gene_list
    assert transformed.n_vars == len(artifact.gene_list)

    # Manual check on one gene
    gene = artifact.gene_list[0]
    raw_col = adata[:, gene].X.flatten()
    mean = artifact.gene_means[artifact.gene_list.index(gene)]
    std  = artifact.gene_stds[artifact.gene_list.index(gene)]
    expected = np.clip((raw_col - mean) / std, -10, 10)
    got = transformed[:, gene].X.flatten()
    assert np.allclose(expected, got, atol=1e-4)


def test_apply_preprocessing_raises_on_missing_genes():
    adata = _adata(n_subjects=10, n_genes=10)
    train_subjects = {f"sub_{i}" for i in range(6)}
    artifact = fit_preprocessing(adata, train_subjects, n_hvgs=5, batch_key=None)

    stripped = adata[:, [g for g in adata.var_names if g != artifact.gene_list[0]]].copy()
    with pytest.raises(ValueError):
        apply_preprocessing(stripped, artifact)


def test_verify_compatible_passes_for_matching_genes():
    adata = _adata(n_subjects=10, n_genes=10)
    train_subjects = {f"sub_{i}" for i in range(6)}
    artifact = fit_preprocessing(adata, train_subjects, n_hvgs=5, batch_key=None)
    verify_compatible(artifact, adata.var_names)  # should not raise


def test_verify_input_matrix_rejects_wrong_order():
    artifact = PreprocessingArtifact(
        version="1", gene_list=["G0", "G1", "G2"],
        gene_means=[0.0, 0.0, 0.0], gene_stds=[1.0, 1.0, 1.0],
        n_hvgs=3, smoke_marker_genes_forced=[], fit_n_cells=10, fit_n_subjects=2,
    )
    verify_input_matrix(artifact, ["G0", "G1", "G2"])  # exact match — ok
    with pytest.raises(ValueError):
        verify_input_matrix(artifact, ["G1", "G0", "G2"])  # reordered
    with pytest.raises(ValueError):
        verify_input_matrix(artifact, ["G0", "G1"])  # missing a gene


def test_artifact_save_and_load_roundtrip(tmp_path):
    adata = _adata(n_subjects=10, n_genes=10)
    train_subjects = {f"sub_{i}" for i in range(6)}
    artifact = fit_preprocessing(adata, train_subjects, n_hvgs=5, batch_key=None)

    path = tmp_path / "artifact.json"
    artifact.save(path)
    loaded = PreprocessingArtifact.load(path)
    assert loaded.gene_list == artifact.gene_list
    assert loaded.gene_means == artifact.gene_means
    assert loaded.version == artifact.version
