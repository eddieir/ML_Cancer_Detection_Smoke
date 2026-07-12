"""data/assembly.py — merging sources, building MIL bags, exporting arrays."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


def _source(n, genes, seed):
    rng = np.random.default_rng(seed)
    X = rng.random((n, len(genes))).astype("float32")
    return ad.AnnData(X=X, obs=pd.DataFrame(index=[f"s{seed}_c{i}" for i in range(n)]),
                       var=pd.DataFrame(index=genes))


def test_merge_sources_keeps_only_common_genes_and_tags_batch():
    from data.assembly import merge_sources
    a = _source(10, ["G1", "G2", "G3", "ONLY_A"], seed=1)
    b = _source(8,  ["G1", "G2", "G3", "ONLY_B"], seed=2)

    merged = merge_sources(a, b)
    assert set(merged.var_names) == {"G1", "G2", "G3"}
    assert merged.n_obs == 18
    assert set(merged.obs["batch"]) == {"source_0", "source_1"}


def _bagged_adata(subject_ids, cells_per_subject):
    rows = []
    for sid, n in zip(subject_ids, cells_per_subject):
        rows += [sid] * n
    n_total = len(rows)
    obs = pd.DataFrame({
        "subject_id":   rows,
        "cell_type_id": np.zeros(n_total, dtype=int),
        "smoke_type":   np.zeros(n_total, dtype=int),
        "malignancy":   np.zeros(n_total, dtype="float32"),
    }, index=[f"c{i}" for i in range(n_total)])
    X = np.random.rand(n_total, 5).astype("float32")
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(5)]))


def test_assemble_subject_bags_drops_small_subjects():
    from data.assembly import assemble_subject_bags
    adata = _bagged_adata(["big", "small"], [20, 3])
    bags = assemble_subject_bags(adata, min_cells_per_subject=10)
    assert len(bags) == 1
    assert bags[0]["subject_id"] == "big"
    assert bags[0]["gene_matrix"].shape == (20, 5)


def test_assemble_subject_bags_attaches_cancer_outcome():
    from data.assembly import assemble_subject_bags
    adata = _bagged_adata(["p1", "p2"], [15, 15])
    outcomes = pd.DataFrame({"subject_id": ["p1", "p2"], "cancer_label": [1, 0]})
    bags = assemble_subject_bags(adata, cancer_outcomes=outcomes, min_cells_per_subject=10)
    by_subject = {b["subject_id"]: b["cancer_label"] for b in bags}
    assert by_subject["p1"] == 1
    assert by_subject["p2"] == 0


def test_assemble_subject_bags_defaults_to_zero_without_outcomes():
    from data.assembly import assemble_subject_bags
    adata = _bagged_adata(["p1"], [15])
    bags = assemble_subject_bags(adata, cancer_outcomes=None, min_cells_per_subject=10)
    assert bags[0]["cancer_label"] == 0


def test_export_cell_dataset_writes_arrays_and_returns_matching_dict():
    from data.assembly import export_cell_dataset
    n, g = 12, 5
    obs = pd.DataFrame({
        "smoke_type":   np.random.randint(0, 6, n),
        "malignancy":   np.random.randint(0, 2, n).astype("float32"),
        "cell_type_id": np.random.randint(0, 4, n),
    }, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=np.random.rand(n, g).astype("float32"), obs=obs,
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))

    with tempfile.TemporaryDirectory() as tmp:
        result = export_cell_dataset(adata, out_dir=tmp)
        for name in ("gene_matrix.npy", "smoke_labels.npy",
                     "malignancy_labels.npy", "cell_type_ids.npy",
                     "cell_metadata.csv", "gene_list.csv"):
            assert (Path(tmp) / name).exists()

        assert result["gene_matrix"].shape == (n, g)
        assert np.array_equal(np.load(Path(tmp) / "gene_matrix.npy"), result["gene_matrix"])
