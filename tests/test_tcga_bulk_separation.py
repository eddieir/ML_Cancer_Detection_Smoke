"""
tests/test_tcga_bulk_separation.py — TCGA bulk expression must never enter
the single-cell pipeline. Uses small synthetic TCGA-shaped fixtures only.
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import data.converters as converters
from data.assay_mode import AssayModeError
from data.loaders import load_microarray
from constants import ASSAY_MODE_BULK_TCGA, ASSAY_MODE_SINGLE_CELL


@pytest.fixture()
def tmp_paths(monkeypatch):
    """Same pattern as tests/test_converters.py's own fixture: redirect
    converters.CONVERTED to a scratch dir so no test ever writes into the
    real repository's data/processed/converted/."""
    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / "converted"
        monkeypatch.setattr(converters, "CONVERTED", converted)
        yield Path(tmp), converted


def _convert_fixture_tcga(tmp_dir, project="TCGA-LUAD"):
    src = tmp_dir / project
    src.mkdir()
    pd.DataFrame([
        {"file_id": "f1", "file_name": "f1.htseq.counts", "case_id": "case1", "sample_type": "Primary Tumor"},
        {"file_id": "f2", "file_name": "f2.htseq.counts", "case_id": "case2", "sample_type": "Solid Tissue Normal"},
    ]).to_csv(src / "file_meta.csv", index=False)
    for fid, counts in [("f1", {"ENSG001.1": 10, "ENSG002.3": 20}),
                         ("f2", {"ENSG001.1": 1, "ENSG002.3": 2})]:
        d = src / fid
        d.mkdir()
        with open(d / f"{fid}.htseq.counts", "w") as f:
            for gene, count in counts.items():
                f.write(f"{gene}\t{count}\n")
    csv_path = converters.convert_tcga(project, src)
    return csv_path


# ─── No cigarette-by-default, assay_mode=bulk_tcga written ──────────────────

def test_tcga_samples_meta_never_defaults_smoke_type_to_cigarette(tmp_paths):
    tmp, converted = tmp_paths
    _convert_fixture_tcga(tmp)
    meta = pd.read_csv(converted / "TCGA-LUAD_samples_meta.csv")
    assert (meta["smoke_type"] == "unknown").all()
    assert (meta["smoke_type_known"] == False).all()  # noqa: E712
    assert (meta["assay_mode"] == "bulk_tcga").all()
    assert not (meta["smoke_type"] == "cigarette").any()


def test_tcga_malignancy_is_bulk_sample_label_only(tmp_paths):
    """sample_type-derived malignancy exists (tumor=1.0/NAT=0.0) but is
    documented as a BULK sample label — the separation from per-cell
    malignancy is architectural (only reachable via load_microarray ->
    assay_mode=bulk_tcga, never merged into a single-cell AnnData)."""
    tmp, converted = tmp_paths
    _convert_fixture_tcga(tmp)
    meta = pd.read_csv(converted / "TCGA-LUAD_samples_meta.csv").set_index("sample_id")
    assert meta.loc["f1", "malignancy"] == 1.0
    assert meta.loc["f2", "malignancy"] == 0.0


def test_load_microarray_tcga_csv_carries_bulk_assay_mode(tmp_paths):
    tmp, converted = tmp_paths
    _convert_fixture_tcga(tmp)
    csv_path = converted / "TCGA-LUAD.csv"
    adata = load_microarray(str(csv_path), "unknown")
    assert (adata.obs["assay_mode"] == ASSAY_MODE_BULK_TCGA).all()
    assert not adata.obs["smoke_type_known"].any()
    assert adata.obs["is_pseudo_bulk"].all()


# ─── Rejected from the default single-cell pipeline ─────────────────────────

def test_preprocess_load_all_sources_rejects_tcga_in_microarray_sources(tmp_paths):
    from preprocess import _load_all_sources
    tmp, converted = tmp_paths
    _convert_fixture_tcga(tmp)
    csv_path = str(converted / "TCGA-LUAD.csv")
    cfg = {"microarray_sources": [(csv_path, "unknown")]}
    with pytest.raises(AssayModeError):
        _load_all_sources(cfg)


def test_preprocess_load_all_sources_rejects_bulk_tagged_scrna_source(tmp_path):
    """Even if someone mistakenly wires a bulk_tcga h5ad through
    scrna_sources, the same guard must catch it — build a tiny h5ad with
    assay_mode=bulk_tcga stamped directly to simulate that misconfiguration."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    from preprocess import _load_all_sources

    genes = [f"G{i}" for i in range(10)]
    obs = pd.DataFrame({
        "donor_id": ["d1", "d2"],
        "assay_mode": [ASSAY_MODE_BULK_TCGA, ASSAY_MODE_BULK_TCGA],
    }, index=["c0", "c1"])
    adata = ad.AnnData(X=sp.csr_matrix(np.random.rand(2, 10).astype("float32")),
                        obs=obs, var=pd.DataFrame(index=genes))
    h5ad_path = tmp_path / "fake_bulk.h5ad"
    adata.write_h5ad(h5ad_path)

    cfg = {"scrna_sources": [(str(h5ad_path), "unknown", "donor_id")]}
    with pytest.raises(AssayModeError):
        _load_all_sources(cfg)


# ─── Dedicated bulk path: disabled by default, actionable errors ───────────

def test_load_tcga_bulk_dataset_disabled_by_default_raises_actionable_error():
    from preprocess import load_tcga_bulk_dataset
    with pytest.raises(ValueError, match="disabled"):
        load_tcga_bulk_dataset({"data": {"tcga": {"enabled": False}}})


def test_load_tcga_bulk_dataset_validates_and_loads_when_enabled(tmp_paths):
    from preprocess import load_tcga_bulk_dataset
    tmp, converted = tmp_paths
    _convert_fixture_tcga(tmp)
    csv_path = str(converted / "TCGA-LUAD.csv")

    cfg = {"data": {"tcga": {"enabled": True, "bulk_sources": [(csv_path, "TCGA-LUAD")]}}}
    datasets = load_tcga_bulk_dataset(cfg)
    assert "TCGA-LUAD" in datasets
    assert (datasets["TCGA-LUAD"].obs["assay_mode"] == ASSAY_MODE_BULK_TCGA).all()


def test_load_tcga_bulk_dataset_require_trainable_raises_not_implemented(tmp_paths):
    from preprocess import load_tcga_bulk_dataset, BulkTrainingNotImplementedError
    tmp, converted = tmp_paths
    _convert_fixture_tcga(tmp)
    csv_path = str(converted / "TCGA-LUAD.csv")

    cfg = {"data": {"tcga": {"enabled": True, "bulk_sources": [(csv_path, "TCGA-LUAD")]}}}
    with pytest.raises(BulkTrainingNotImplementedError):
        load_tcga_bulk_dataset(cfg, require_trainable=True)


def test_load_tcga_bulk_dataset_missing_files_raises_actionable_error():
    from preprocess import load_tcga_bulk_dataset
    cfg = {"data": {"tcga": {"enabled": True, "bulk_sources": [("/nonexistent/TCGA-LUAD.csv", "TCGA-LUAD")]}}}
    with pytest.raises(ValueError):
        load_tcga_bulk_dataset(cfg)


# ─── Real default config never routes TCGA through the single-cell path ────

def test_default_config_microarray_sources_excludes_tcga():
    import yaml
    repo_root = Path(__file__).parents[1]
    with open(repo_root / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    sources = cfg["data"]["microarray_sources"]
    assert not any("TCGA" in str(row) for row in sources)
    assert "bulk_sources" in cfg["data"]["tcga"]
    assert cfg["data"]["tcga"]["enabled"] is False


def test_assay_mode_guard_rejects_bulk_rows_from_cell_level_dataset_construction():
    """A single-cell-only construction path that calls the assay_mode
    guard before building a CellLevelDataset must reject bulk rows
    outright — the same guard subject-balanced sampling and MIL bag
    construction should call before accepting external rows."""
    from data.assay_mode import assert_no_pseudo_bulk_rows
    is_pseudo_bulk = [False, False, True]  # simulates one TCGA sample slipping in
    with pytest.raises(AssayModeError):
        assert_no_pseudo_bulk_rows(is_pseudo_bulk, assay_mode=ASSAY_MODE_SINGLE_CELL)


# ─── Mixed assay modes fail before preprocessing/training ──────────────────

def test_mixed_assay_mode_in_one_source_list_fails_before_merge(tmp_paths):
    """A config mixing a real human scrna source with a bulk_tcga
    microarray source must fail at _load_all_sources — before
    merge_sources, fit_preprocessing, or any training loop ever runs."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    from preprocess import _load_all_sources

    tmp, converted = tmp_paths
    _convert_fixture_tcga(tmp)
    tcga_csv = str(converted / "TCGA-LUAD.csv")

    n, g = 30, 250
    genes = [f"G{i}" for i in range(g)]
    obs = pd.DataFrame({"donor_id": ["d1"] * n}, index=[f"c{i}" for i in range(n)])
    human_adata = ad.AnnData(
        X=sp.csr_matrix(np.random.negative_binomial(5, 0.5, (n, g)).astype("float32")),
        obs=obs, var=pd.DataFrame(index=genes),
    )
    human_path = tmp / "human.h5ad"
    human_adata.write_h5ad(human_path)

    cfg = {
        "scrna_sources": [(str(human_path), "unknown", "donor_id")],
        "microarray_sources": [(tcga_csv, "unknown")],
    }
    with pytest.raises(AssayModeError):
        _load_all_sources(cfg)


# ─── Bulk expression cannot influence single-cell HVG/scaling ──────────────

def test_bulk_tcga_excluded_means_fit_preprocessing_never_sees_it(tmp_path):
    """End-to-end: run _load_all_sources with ONLY a human scrna source
    configured (no TCGA) — the resulting adatas list must contain no
    bulk_tcga rows at all, so fit_preprocessing (which operates on
    whatever _load_all_sources returns) can never see TCGA bulk expression
    values under the default config."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    from preprocess import _load_all_sources

    n, g = 30, 250
    genes = [f"G{i}" for i in range(g)]
    obs = pd.DataFrame({"donor_id": ["d1"] * n}, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(
        X=sp.csr_matrix(np.random.negative_binomial(5, 0.5, (n, g)).astype("float32")),
        obs=obs, var=pd.DataFrame(index=genes),
    )
    h5ad_path = tmp_path / "human.h5ad"
    adata.write_h5ad(h5ad_path)

    cfg = {"scrna_sources": [(str(h5ad_path), "unknown", "donor_id")]}
    adatas = _load_all_sources(cfg)
    for a in adatas:
        assert (a.obs["assay_mode"] != ASSAY_MODE_BULK_TCGA).all()
