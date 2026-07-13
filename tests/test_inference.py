"""inference.py — Predictor preprocessing-compatibility validation."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import N_CELL_TYPES
from data.preprocessing import PreprocessingArtifact
from inference import Predictor
from model import MultiSmokeCancerNet

GENES = 10


def _artifact(genes):
    return PreprocessingArtifact(
        version="1", gene_list=list(genes),
        gene_means=[0.0] * len(genes), gene_stds=[1.0] * len(genes),
        n_hvgs=len(genes), smoke_marker_genes_forced=[],
        fit_n_cells=100, fit_n_subjects=10,
    )


def _h5ad(path, genes, n=20):
    obs = pd.DataFrame({
        "subject_id":   ["s1"] * n,
        "cell_type_id": np.zeros(n, dtype=int),
    }, index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=np.random.randn(n, len(genes)).astype("float32"), obs=obs,
                    var=pd.DataFrame(index=genes))
    a.write_h5ad(path)


def test_predict_h5ad_rejects_incompatible_gene_panel():
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact([f"G{i}" for i in range(GENES)])
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mismatched.h5ad"
        _h5ad(path, [f"OTHER{i}" for i in range(GENES)])  # completely different gene panel
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path))


def test_predict_h5ad_accepts_compatible_gene_panel():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "matched.h5ad"
        _h5ad(path, genes)
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1


def test_predict_h5ad_rejects_raw_input_without_artifact_by_default():
    """No artifact loaded and no explicit unsafe_legacy_mode — refuse rather
    than silently run unreordered/unscaled genes through the model."""
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model)  # no artifact, unsafe_legacy_mode=False

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "any.h5ad"
        _h5ad(path, [f"WHATEVER{i}" for i in range(GENES)])
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path))


def test_predict_h5ad_unsafe_legacy_mode_bypasses_missing_artifact():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, unsafe_legacy_mode=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "any.h5ad"
        _h5ad(path, genes)
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1


def test_predict_h5ad_reorders_genes_that_are_present_but_shuffled():
    """A raw file with the right genes in the WRONG order must still be
    correctly reordered by apply_preprocessing before scoring."""
    genes = [f"G{i}" for i in range(GENES)]
    shuffled = list(reversed(genes))
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shuffled.h5ad"
        _h5ad(path, shuffled)
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1


def test_predict_h5ad_rejects_extra_genes_not_in_artifact():
    """Extra genes beyond the artifact's panel must be dropped, not error —
    only MISSING required genes are fatal (see verify_compatible)."""
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "extra.h5ad"
        _h5ad(path, genes + ["EXTRA_GENE_1", "EXTRA_GENE_2"])
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1


def test_predict_h5ad_already_preprocessed_requires_exact_gene_order():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shuffled.h5ad"
        _h5ad(path, list(reversed(genes)))
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path), already_preprocessed=True)


def test_predict_h5ad_already_preprocessed_accepts_exact_match():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "exact.h5ad"
        _h5ad(path, genes)
        results = predictor.predict_h5ad(str(path), already_preprocessed=True)
        assert len(results) == 1


def test_predict_h5ad_rejects_wrong_model_input_width():
    """Model expects GENES+5 features but the artifact/H5AD only has GENES —
    must fail with a clear error, not a confusing shape mismatch deep in the model."""
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES + 5, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "matched.h5ad"
        _h5ad(path, genes)
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path))


def test_predict_h5ad_handles_sparse_input():
    import scipy.sparse as sp
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sparse.h5ad"
        n = 15
        obs = pd.DataFrame({
            "subject_id":   ["s1"] * n,
            "cell_type_id": np.zeros(n, dtype=int),
        }, index=[f"c{i}" for i in range(n)])
        a = ad.AnnData(
            X=sp.csr_matrix(np.random.randn(n, len(genes)).astype("float32")),
            obs=obs, var=pd.DataFrame(index=genes),
        )
        a.write_h5ad(path)
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1
