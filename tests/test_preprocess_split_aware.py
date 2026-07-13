"""preprocess.py::run_pipeline_split_aware — leakage-free end-to-end integration."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import ALL_SMOKE_MARKERS, N_SMOKE_CLASSES


def _synthetic_h5ad_consistent_labels(path, n_subjects=12, cells_per_subject=30, g=500, n_classes=3):
    """Unlike test_pipeline.py's generator, every cell of a given subject
    shares the SAME smoke_type — required for subject-level stratified
    splitting, and realistic (a donor has one smoking status, not a random
    one per cell)."""
    genes = [f"G{i}" for i in range(g)]
    genes[:5] = [f"MT-{i}" for i in range(5)]
    for i, mk in enumerate(ALL_SMOKE_MARKERS[:4]):
        genes[50 + i] = mk

    subject_ids, smoke_types = [], []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * cells_per_subject
        smoke_types  += [i % n_classes] * cells_per_subject
    n = len(subject_ids)

    raw = np.random.negative_binomial(5, 0.7, (n, g)).astype("float32")
    obs = pd.DataFrame({
        "donor_id":        subject_ids,
        "subject_id":      subject_ids,
        "smoke_type":      smoke_types,
        "smoke_type_name": "mixed",
        "data_modality":   "scrna",
        "is_pseudo_bulk":  False,
        "malignancy":      0.0,
        "cell_type_id":    0,
    }, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=sp.csr_matrix(raw), obs=obs, var=pd.DataFrame(index=genes))
    adata.write_h5ad(path)


def test_run_pipeline_split_aware_produces_split_and_artifact():
    from preprocess import run_pipeline_split_aware
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad_consistent_labels(h5ad)
        result = run_pipeline_split_aware({
            "data": {
                "scrna_sources": [(h5ad, "cigarette", "donor_id")],
                "n_hvgs": 50,
                "min_cells_per_subject": 5,
                "out_dir": str(Path(tmp) / "processed"),
            },
            "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
        })
        assert "cell_data" in result
        assert "bags" in result
        assert "split_manifest" in result
        assert "preprocessing_artifact" in result

        manifest = result["split_manifest"]
        all_subjects = manifest.train_subjects + manifest.val_subjects + manifest.test_subjects
        assert len(all_subjects) == len(set(all_subjects))  # no leakage

        artifact = result["preprocessing_artifact"]
        assert artifact.fit_n_subjects == len(manifest.train_subjects)
        assert len(artifact.gene_list) <= 50


def test_run_pipeline_split_aware_test_cells_do_not_affect_gene_scaling():
    """The core leakage guarantee, exercised end-to-end: refitting after
    corrupting the val/test cells' raw values must not change which genes
    were selected or their scaling statistics."""
    from preprocess import run_pipeline_split_aware
    import anndata as ad_module

    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad_consistent_labels(h5ad)

        cfg = {
            "data": {
                "scrna_sources": [(h5ad, "cigarette", "donor_id")],
                "n_hvgs": 50,
                "min_cells_per_subject": 5,
                "out_dir": str(Path(tmp) / "processed"),
            },
            "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
        }
        result1 = run_pipeline_split_aware(cfg)
        artifact1 = result1["preprocessing_artifact"]

        # Corrupt the source file's expression matrix, but only for
        # subjects NOT in the training split, then re-run.
        adata = ad_module.read_h5ad(h5ad)
        non_train = ~adata.obs["donor_id"].isin(result1["split_manifest"].train_subjects).values
        adata.X = adata.X.toarray()
        adata.X[non_train] = adata.X[non_train] * 1000 + 5000
        h5ad2 = str(Path(tmp) / "test_corrupted.h5ad")
        adata.write_h5ad(h5ad2)

        cfg2 = dict(cfg)
        cfg2["data"] = dict(cfg["data"])
        cfg2["data"]["scrna_sources"] = [(h5ad2, "cigarette", "donor_id")]
        cfg2["data"]["out_dir"] = str(Path(tmp) / "processed2")
        cfg2["split"] = dict(cfg["split"])
        cfg2["split"]["manifest_path"] = None
        result2 = run_pipeline_split_aware(cfg2)
        artifact2 = result2["preprocessing_artifact"]

        assert artifact1.gene_means == artifact2.gene_means
        assert artifact1.gene_stds == artifact2.gene_stds
