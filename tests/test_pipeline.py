"""End-to-end pipeline integration test."""
import pytest
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import ALL_SMOKE_MARKERS, N_SMOKE_CLASSES, N_CELL_TYPES

def _synthetic_h5ad(path, n=300, g=500):
    genes = [f"G{i}" for i in range(g)]
    genes[:5] = [f"MT-{i}" for i in range(5)]
    for i, mk in enumerate(ALL_SMOKE_MARKERS[:4]):
        genes[50+i] = mk
    raw = np.random.negative_binomial(5, 0.7, (n, g)).astype("float32")
    obs = pd.DataFrame({
        "donor_id":       [f"sub_{i//30}" for i in range(n)],
        "subject_id":     [f"sub_{i//30}" for i in range(n)],
        "smoke_type":     np.random.randint(0, N_SMOKE_CLASSES, n),
        "smoke_type_name":"mixed",
        "data_modality":  "scrna",
        "is_pseudo_bulk": False,
        "malignancy":     0.0,
        "cell_type_id":   0,
    }, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=sp.csr_matrix(raw), obs=obs,
                       var=pd.DataFrame(index=genes))
    adata.write_h5ad(path)

def test_run_pipeline_produces_cell_data():
    from preprocess import run_pipeline
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        out  = str(Path(tmp) / "processed")
        _synthetic_h5ad(h5ad)
        cell_data, bags = run_pipeline({
            "scrna_sources": [(h5ad, "cigarette", "donor_id")],
            "n_hvgs": 50,
            "min_cells_per_subject": 5,
            "out_dir": out,
            # This environment's installed CellTypist model was serialized
            # under scikit-learn 0.24.1, incompatible with the installed
            # 1.9.0 (see data/transforms.py::CellTypistCompatibilityError) —
            # a real run must fail closed on that by default. This test only
            # exercises pipeline shape/mechanics on synthetic random data, so
            # it explicitly opts into the disclosed, always-degraded
            # diagnostic override rather than testing CellTypist compatibility.
            "cell_type_allow_diagnostic_fallback": True,
        })
        assert "gene_matrix"  in cell_data
        assert "smoke_labels" in cell_data
        assert cell_data["gene_matrix"].shape[1] == 50

def test_run_pipeline_produces_bags():
    from preprocess import run_pipeline
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad(h5ad)
        _, bags = run_pipeline({
            "scrna_sources": [(h5ad, "cigarette", "donor_id")],
            "n_hvgs": 50,
            "min_cells_per_subject": 5,
            "out_dir": str(Path(tmp) / "p"),
            # See test_run_pipeline_produces_cell_data for why this
            # disclosed diagnostic override is used here.
            "cell_type_allow_diagnostic_fallback": True,
        })
        assert len(bags) > 0
        assert "gene_matrix" in bags[0]
        assert "cancer_label" in bags[0]

def test_model_forward_after_pipeline():
    from preprocess import run_pipeline
    from model import MultiSmokeCancerNet
    import torch
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad(h5ad, n=60, g=500)  # qc_filter's default min_genes=200 needs headroom above it
        cell_data, bags = run_pipeline({
            "scrna_sources": [(h5ad, "cigarette", "donor_id")],
            "n_hvgs": 50,
            "min_cells_per_subject": 5,
            "out_dir": str(Path(tmp) / "p"),
            # See test_run_pipeline_produces_cell_data for why this
            # disclosed diagnostic override is used here.
            "cell_type_allow_diagnostic_fallback": True,
        })
        model = MultiSmokeCancerNet(input_dim=50, embedding_dim=32, attention_dim=16)
        b = bags[0]
        out = model.forward_subject(
            torch.FloatTensor(b["gene_matrix"]),
            torch.LongTensor(b["cell_type_ids"]),
        )
        assert 0.0 <= out["cancer_probability"].item() <= 1.0