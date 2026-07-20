"""data/assembly.py — merging sources, building MIL bags, exporting arrays."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest

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

    merged = merge_sources(a, b, diagnostic_mode=True)
    assert set(merged.var_names) == {"G1", "G2", "G3"}
    assert merged.n_obs == 18
    assert set(merged.obs["batch"]) == {"source_0", "source_1"}


def test_merge_sources_rejects_missing_provenance_in_real_mode():
    from data.assembly import merge_sources
    from data.assay_policy import MissingAssayProvenanceError
    a = _source(10, ["G1", "G2", "G3"], seed=1)
    b = _source(8,  ["G1", "G2", "G3"], seed=2)
    with pytest.raises(MissingAssayProvenanceError):
        merge_sources(a, b)


def test_merge_sources_diagnostic_mode_stamps_synthetic_provenance():
    from data.assembly import merge_sources
    a = _source(10, ["G1", "G2", "G3"], seed=1)
    b = _source(8,  ["G1", "G2", "G3"], seed=2)
    merged = merge_sources(a, b, diagnostic_mode=True)
    assert merged.uns["diagnostic_mode"] is True
    assert not merged.obs["is_pseudo_bulk"].any()


def test_merge_sources_rejects_invalid_provenance_string():
    from data.assembly import merge_sources
    from data.assay_policy import InvalidAssayProvenanceError
    a = _source(4, ["G1", "G2"], seed=1)
    a.obs["is_pseudo_bulk"] = ["not_a_bool"] * 4
    b = _source(4, ["G1", "G2"], seed=2)
    b.obs["is_pseudo_bulk"] = [False] * 4
    with pytest.raises(InvalidAssayProvenanceError):
        merge_sources(a, b)


def test_merge_sources_rejects_pseudo_bulk_under_single_cell_only():
    from data.assembly import merge_sources
    from data.assay_policy import AssayPolicyError
    a = _source(4, ["G1", "G2"], seed=1)
    a.obs["is_pseudo_bulk"] = [False] * 4
    b = _source(4, ["G1", "G2"], seed=2)
    b.obs["is_pseudo_bulk"] = [True] * 4  # bulk row hiding behind a renamed source
    with pytest.raises(AssayPolicyError):
        merge_sources(a, b)


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
    bags = assemble_subject_bags(adata, min_cells_per_subject=10, diagnostic_mode=True)
    assert len(bags) == 1
    assert bags[0]["subject_id"] == "big"
    assert bags[0]["gene_matrix"].shape == (20, 5)


def test_assemble_subject_bags_attaches_cancer_outcome():
    from data.assembly import assemble_subject_bags
    adata = _bagged_adata(["p1", "p2"], [15, 15])
    outcomes = pd.DataFrame({"subject_id": ["p1", "p2"], "cancer_label": [1, 0]})
    bags = assemble_subject_bags(adata, cancer_outcomes=outcomes, min_cells_per_subject=10,
                                  diagnostic_mode=True)
    by_subject = {b["subject_id"]: b["cancer_label"] for b in bags}
    assert by_subject["p1"] == 1
    assert by_subject["p2"] == 0
    assert all(b["cancer_label_known"] for b in bags)


def test_assemble_subject_bags_marks_unknown_outcome_not_negative():
    """A subject never matched to a cancer-outcome source has an UNKNOWN
    outcome, not a verified negative — silently defaulting to 0 would
    fabricate a negative label (see data/assembly.py::assemble_subject_bags)."""
    from data.assembly import assemble_subject_bags
    adata = _bagged_adata(["p1"], [15])
    bags = assemble_subject_bags(adata, cancer_outcomes=None, min_cells_per_subject=10,
                                  diagnostic_mode=True)
    assert bags[0]["cancer_label"] is None
    assert bags[0]["cancer_label_known"] is False


def test_assemble_subject_bags_partial_outcome_coverage_marks_missing_unknown():
    """When cancer_outcomes only covers some subjects, the uncovered ones must
    stay unknown rather than silently becoming cancer_label=0."""
    from data.assembly import assemble_subject_bags
    adata = _bagged_adata(["p1", "p2"], [15, 15])
    outcomes = pd.DataFrame({"subject_id": ["p1"], "cancer_label": [1]})
    bags = assemble_subject_bags(adata, cancer_outcomes=outcomes, min_cells_per_subject=10,
                                  diagnostic_mode=True)
    by_subject = {b["subject_id"]: b for b in bags}
    assert by_subject["p1"]["cancer_label"] == 1
    assert by_subject["p1"]["cancer_label_known"] is True
    assert by_subject["p2"]["cancer_label"] is None
    assert by_subject["p2"]["cancer_label_known"] is False


def test_assemble_subject_bags_rejects_missing_provenance_in_real_mode():
    from data.assembly import assemble_subject_bags
    from data.assay_policy import MissingAssayProvenanceError
    adata = _bagged_adata(["p1"], [15])
    with pytest.raises(MissingAssayProvenanceError):
        assemble_subject_bags(adata, min_cells_per_subject=10)


def test_assemble_subject_bags_rejects_bulk_only_policy_even_in_diagnostic_mode():
    from data.assembly import assemble_subject_bags
    from data.assay_policy import BulkTrainingNotImplementedError
    adata = _bagged_adata(["p1"], [15])
    with pytest.raises(BulkTrainingNotImplementedError):
        assemble_subject_bags(adata, min_cells_per_subject=10, diagnostic_mode=True,
                               assay_policy="bulk_only")


def test_assemble_subject_bags_carries_provenance_fields():
    from data.assembly import assemble_subject_bags
    adata = _bagged_adata(["p1"], [15])
    bags = assemble_subject_bags(adata, min_cells_per_subject=10, diagnostic_mode=True)
    assert bags[0]["assay_policy"] == "single_cell_only"
    assert bags[0]["diagnostic_mode"] is True
    assert len(bags[0]["is_pseudo_bulk"]) == 15
    assert not np.asarray(bags[0]["is_pseudo_bulk"]).any()


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
        result = export_cell_dataset(adata, out_dir=tmp, diagnostic_mode=True)
        for name in ("gene_matrix.npy", "smoke_labels.npy",
                     "malignancy_labels.npy", "cell_type_ids.npy",
                     "exposure_dose.npy", "cell_metadata.csv", "gene_list.csv",
                     "dataset_metadata.json"):
            assert (Path(tmp) / name).exists()

        assert result["gene_matrix"].shape == (n, g)
        assert np.array_equal(np.load(Path(tmp) / "gene_matrix.npy"), result["gene_matrix"])

        import json
        with open(Path(tmp) / "dataset_metadata.json") as f:
            meta = json.load(f)
        assert meta["diagnostic_mode"] is True
        assert meta["assay_policy"] == "single_cell_only"


def test_export_cell_dataset_rejects_missing_provenance_in_real_mode():
    from data.assembly import export_cell_dataset
    from data.assay_policy import MissingAssayProvenanceError
    n, g = 6, 3
    obs = pd.DataFrame({
        "smoke_type":   np.zeros(n, dtype=int),
        "malignancy":   np.zeros(n, dtype="float32"),
        "cell_type_id": np.zeros(n, dtype=int),
    }, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=np.random.rand(n, g).astype("float32"), obs=obs,
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(MissingAssayProvenanceError):
            export_cell_dataset(adata, out_dir=str(Path(tmp) / "out"))
        # No partial output left behind after a validation failure.
        assert not (Path(tmp) / "out").exists()


def test_export_cell_dataset_rejects_pseudo_bulk_row_under_single_cell_only():
    from data.assembly import export_cell_dataset
    from data.assay_policy import AssayPolicyError
    n, g = 6, 3
    obs = pd.DataFrame({
        "smoke_type":     np.zeros(n, dtype=int),
        "malignancy":     np.zeros(n, dtype="float32"),
        "cell_type_id":   np.zeros(n, dtype=int),
        "is_pseudo_bulk": [False] * (n - 1) + [True],
    }, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=np.random.rand(n, g).astype("float32"), obs=obs,
                        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        with pytest.raises(AssayPolicyError):
            export_cell_dataset(adata, out_dir=str(out))
        assert not out.exists()
