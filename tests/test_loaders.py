"""data/loaders.py — I/O only, one loader per data source type."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import SMOKE_TYPE_MAP


def _synthetic_scrna(n=40, g=20, donor_col="donor_id"):
    obs = pd.DataFrame({donor_col: [f"sub_{i // 10}" for i in range(n)]},
                        index=[f"c{i}" for i in range(n)])
    X = np.random.rand(n, g).astype("float32")
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs,
                       var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))


def test_load_scrna_attaches_standard_obs():
    from data.loaders import load_scrna
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.h5ad")
        _synthetic_scrna().write_h5ad(path)
        adata = load_scrna(path, "cigarette", "donor_id")
        assert (adata.obs["smoke_type"] == SMOKE_TYPE_MAP["cigarette"]).all()
        assert (adata.obs["data_modality"] == "scrna").all()
        assert set(adata.obs["subject_id"]) == {"sub_0", "sub_1", "sub_2", "sub_3"}


def test_load_scrna_falls_back_to_unknown_subject():
    from data.loaders import load_scrna
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.h5ad")
        _synthetic_scrna(donor_col="something_else").write_h5ad(path)
        adata = load_scrna(path, "cigarette", "donor_id")
        assert (adata.obs["subject_id"] == "unknown").all()


def test_load_microarray_blanket_smoke_type():
    from data.loaders import load_microarray
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "GSEXXX.csv"
        pd.DataFrame({"s1": [1, 2], "s2": [3, 4]}, index=["G1", "G2"]).to_csv(csv_path)
        adata = load_microarray(str(csv_path), "cigarette")
        assert adata.n_obs == 2
        assert (adata.obs["smoke_type"] == SMOKE_TYPE_MAP["cigarette"]).all()
        assert adata.obs["is_pseudo_bulk"].all()


def test_load_microarray_per_sample_smoke_type_override():
    """GSE994-style: one series matrix contains both smokers and never-smokers."""
    from data.loaders import load_microarray
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "GSE994.csv"
        pd.DataFrame({"s1": [1, 2], "s2": [3, 4]}, index=["G1", "G2"]).to_csv(csv_path)
        pd.DataFrame({
            "sample_id": ["s1", "s2"], "smoke_type": ["cigarette", "unexposed"],
        }).to_csv(Path(tmp) / "GSE994_samples_meta.csv", index=False)

        adata = load_microarray(str(csv_path), "cigarette")
        assert adata.obs.loc["s1", "smoke_type"] == SMOKE_TYPE_MAP["cigarette"]
        assert adata.obs.loc["s2", "smoke_type"] == SMOKE_TYPE_MAP["unexposed"]


def test_load_microarray_malignancy_and_subject_id_override():
    """TCGA-style: samples_meta carries real malignancy + subject_id, not just smoke_type."""
    from data.loaders import load_microarray
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "TCGA-LUAD.csv"
        pd.DataFrame({"f1": [10, 20], "f2": [1, 2]}, index=["ENSG001", "ENSG002"]).to_csv(csv_path)
        pd.DataFrame({
            "sample_id": ["f1", "f2"], "smoke_type": ["cigarette", "cigarette"],
            "malignancy": [1.0, 0.0], "subject_id": ["case1", "case2"],
        }).to_csv(Path(tmp) / "TCGA-LUAD_samples_meta.csv", index=False)

        adata = load_microarray(str(csv_path), "cigarette")
        assert adata.obs.loc["f1", "malignancy"] == 1.0
        assert adata.obs.loc["f2", "malignancy"] == 0.0
        assert adata.obs.loc["f1", "subject_id"] == "case1"
        assert adata.obs.loc["f2", "subject_id"] == "case2"


def test_other_loaders_default_exposure_dose_to_unknown():
    """Sources without real exposure duration must not silently look like dose=0."""
    from data.loaders import load_microarray
    from constants import DOSE_UNKNOWN
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "GSE1.csv"
        pd.DataFrame({"s1": [1, 2], "s2": [3, 4]}, index=["G1", "G2"]).to_csv(csv_path)
        adata = load_microarray(str(csv_path), "cigarette")
        assert (adata.obs["exposure_dose"] == DOSE_UNKNOWN).all()


def test_load_mouse_scrna_sets_vape():
    from data.loaders import load_mouse_scrna
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "mouse.h5ad")
        _synthetic_scrna().write_h5ad(path)
        adata = load_mouse_scrna(path)
        assert (adata.obs["smoke_type"] == SMOKE_TYPE_MAP["vape"]).all()
        assert (adata.obs["data_modality"] == "mouse_scrna").all()
