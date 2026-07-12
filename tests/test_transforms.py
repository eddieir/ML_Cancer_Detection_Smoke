"""data/transforms.py — pure AnnData -> AnnData transforms, no I/O, no labels."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import ALL_SMOKE_MARKERS


def _base_obs(n, is_pseudo_bulk=False, batch=None):
    obs = pd.DataFrame({
        "data_modality":  "scrna",
        "is_pseudo_bulk": is_pseudo_bulk,
    }, index=[f"c{i}" for i in range(n)])
    if batch is not None:
        obs["batch"] = batch
    return obs


def test_qc_filter_removes_low_gene_and_high_mito_cells():
    from data.transforms import qc_filter
    n, g = 50, 30
    genes = [f"G{i}" for i in range(g)]
    genes[:3] = ["MT-1", "MT-2", "MT-3"]
    X = np.random.negative_binomial(20, 0.5, (n, g)).astype("float32")
    # Cell 0: almost no expression -> fails min_genes.
    X[0, :] = 0
    X[0, 0] = 1
    # Cell 1: dominated by MT genes -> fails pct_mito.
    X[1, :] = 1
    X[1, :3] = 1000

    adata = ad.AnnData(X=sp.csr_matrix(X), obs=_base_obs(n),
                        var=pd.DataFrame(index=genes))
    out = qc_filter(adata, min_genes=5, min_cells=1, max_pct_mito=20.0)
    assert "c0" not in out.obs_names
    assert "c1" not in out.obs_names
    assert out.n_obs < n


def test_qc_filter_skips_thresholds_for_pseudo_bulk():
    from data.transforms import qc_filter
    n, g = 10, 15
    X = np.random.rand(n, g).astype("float32")
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=_base_obs(n, is_pseudo_bulk=True),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = qc_filter(adata)
    assert out.n_obs == n  # no per-cell thresholds applied


def test_normalize_skips_scaling_for_microarray():
    from data.transforms import normalize
    n, g = 10, 5
    X = np.random.rand(n, g).astype("float32") + 1.0
    obs = _base_obs(n)
    obs["data_modality"] = "microarray"
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = normalize(adata)
    assert np.allclose(out.layers["lognorm"], out.layers["counts"])


def test_normalize_applies_cpm_log1p_for_scrna():
    from data.transforms import normalize
    n, g = 10, 5
    X = (np.random.rand(n, g).astype("float32") + 1.0) * 100
    X_orig = X.copy()  # normalize() mutates adata.X in-place; AnnData doesn't copy on construction
    adata = ad.AnnData(X=X, obs=_base_obs(n), var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = normalize(adata)
    assert not np.allclose(out.X, X_orig)  # log1p(CPM) must differ from raw counts
    assert "lognorm" in out.layers


def test_smoke_aware_hvg_forces_markers_in():
    from data.transforms import smoke_aware_hvg
    n, g = 60, 100
    genes = [f"G{i}" for i in range(g)]
    marker = ALL_SMOKE_MARKERS[0]
    genes[0] = marker

    X = np.random.negative_binomial(5, 0.7, (n, g)).astype("float32")
    # Zero out the marker's variance so standard HVG selection would never pick it.
    X[:, 0] = 5.0

    adata = ad.AnnData(X=X, obs=_base_obs(n), var=pd.DataFrame(index=genes))
    out = smoke_aware_hvg(adata, n_hvgs=10)
    assert marker in out.var_names


def test_batch_correct_skips_single_batch():
    from data.transforms import batch_correct
    n, g = 20, 10
    X = np.random.rand(n, g).astype("float32")
    adata = ad.AnnData(X=X, obs=_base_obs(n, batch="source_0"),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = batch_correct(adata)
    assert "X_pca_harmony" not in out.obsm


def test_annotate_cell_types_skips_pseudo_bulk():
    from data.transforms import annotate_cell_types
    n, g = 10, 5
    X = np.random.rand(n, g).astype("float32")
    adata = ad.AnnData(X=X, obs=_base_obs(n, is_pseudo_bulk=True),
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    out = annotate_cell_types(adata)
    # Default cell_type_id (0) is untouched since celltypist never runs.
    assert "cell_type_name" not in out.obs.columns
