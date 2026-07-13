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


def test_predict_h5ad_skips_validation_without_artifact():
    """No artifact loaded (e.g. legacy checkpoint) — falls back to no gene-panel check."""
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model)  # no artifact

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "any.h5ad"
        _h5ad(path, [f"WHATEVER{i}" for i in range(GENES)])
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1
