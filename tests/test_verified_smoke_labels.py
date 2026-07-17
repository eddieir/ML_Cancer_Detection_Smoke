"""
tests/test_verified_smoke_labels.py — GSE136831 weak-proxy handling and
the smoke_known gate through loading, assembly, sampling, loss, and
metrics. Proves the default verified_only label policy never fabricates a
verified cigarette label from COPD status or dataset membership alone.
"""
import gzip
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import data.converters as converters
from constants import SMOKE_TYPE_MAP
from data.labellers import apply_weak_smoke_proxies, compute_smoke_class_weights
from data.loaders import load_scrna
from data.assembly import assemble_subject_bags, export_cell_dataset


# ─── Converter-level: no COPD -> cigarette fabrication ──────────────────────

def _gse136831_fixture(tmp_path):
    """Real-shape GSE136831 fixture: two donors, one COPD one Control."""
    src = tmp_path / "GSE136831"
    src.mkdir()
    meta_path = src / "GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz"
    with gzip.open(meta_path, "wt") as f:
        f.write('"CellBarcode_Identity"\t"CellType_Category"\t"Disease_Identity"\t"Subject_Identity"\n')
        f.write('"BC1"\t"Lymphoid"\t"COPD"\t"donor_copd"\n')
        f.write('"BC2"\t"Lymphoid"\t"Control"\t"donor_control"\n')
    return converters._load_gse136831_cell_metadata(src)


def test_gse136831_copd_metadata_does_not_produce_verified_cigarette_label(tmp_path):
    meta = _gse136831_fixture(tmp_path)
    assert meta.loc["BC1", "smoke_type_name"] == "unknown"
    assert meta.loc["BC1", "smoke_type_known"] == False  # noqa: E712
    assert meta.loc["BC2", "smoke_type_name"] == "unknown"
    assert meta.loc["BC2", "smoke_type_known"] == False  # noqa: E712


def test_gse136831_weak_proxy_only_set_for_copd_not_control(tmp_path):
    meta = _gse136831_fixture(tmp_path)
    assert meta.loc["BC1", "weak_smoke_proxy_known"] == True   # noqa: E712
    assert meta.loc["BC1", "weak_smoke_proxy_type"] == "COPD_diagnosis"
    assert meta.loc["BC1", "weak_smoke_proxy_value"] == "cigarette"
    assert meta.loc["BC1", "weak_smoke_proxy_source"] == "GSE136831 Disease_Identity"
    assert isinstance(meta.loc["BC1", "weak_smoke_proxy_limitation"], str)
    assert meta.loc["BC2", "weak_smoke_proxy_known"] == False  # noqa: E712


def test_missing_smoking_metadata_stays_unknown_not_cigarette(tmp_path):
    """A donor with no Disease_Identity match at all (e.g. IPF, or a
    barcode absent from the metadata table) must never default to
    cigarette."""
    src = tmp_path / "GSE_IPF"
    src.mkdir()
    meta_path = src / "GSE_IPF_AllCells.Samples.CellType.MetadataTable.txt.gz"
    with gzip.open(meta_path, "wt") as f:
        f.write('"CellBarcode_Identity"\t"CellType_Category"\t"Disease_Identity"\t"Subject_Identity"\n')
        f.write('"BC1"\t"Lymphoid"\t"IPF"\t"donor_ipf"\n')
    meta = converters._load_gse136831_cell_metadata(src)
    assert meta.loc["BC1", "smoke_type_name"] == "unknown"
    assert meta.loc["BC1", "smoke_type_known"] == False  # noqa: E712
    assert meta.loc["BC1", "weak_smoke_proxy_known"] == False  # noqa: E712


# ─── Loader-level: end-to-end h5ad -> obs columns ───────────────────────────

def _write_gse136831_like_h5ad(path, n_copd=6, n_control=6, g=20):
    genes = [f"G{i}" for i in range(g)]
    n = n_copd + n_control
    X = np.random.negative_binomial(5, 0.7, (n, g)).astype("float32")
    obs = pd.DataFrame({
        "donor_id": ["copd_donor"] * n_copd + ["control_donor"] * n_control,
        "disease_identity": ["COPD"] * n_copd + ["Control"] * n_control,
        "smoke_type_name": ["unknown"] * n,
        "smoke_type_known": [False] * n,
        "weak_smoke_proxy_known": [True] * n_copd + [False] * n_control,
        "weak_smoke_proxy_type": ["COPD_diagnosis"] * n_copd + [None] * n_control,
        "weak_smoke_proxy_value": ["cigarette"] * n_copd + [None] * n_control,
        "weak_smoke_proxy_source": ["GSE136831 Disease_Identity"] * n_copd + [None] * n_control,
        "weak_smoke_proxy_limitation": ["COPD is a proxy, not verified exposure"] * n_copd + [None] * n_control,
    }, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes))
    adata.write_h5ad(path)


def test_loader_preserves_unknown_smoke_status_through_load_scrna(tmp_path):
    h5ad = tmp_path / "gse136831_like.h5ad"
    _write_gse136831_like_h5ad(h5ad)
    loaded = load_scrna(str(h5ad), "unknown", "donor_id")
    assert not loaded.obs["smoke_type_known"].any()
    assert (loaded.obs["smoke_type_name"] == "unknown").all()
    # smoke_type numeric placeholder exists (SMOKE_TYPE_MAP has no "unknown"
    # entry, so .get(..., 5) applies) but is never claimed as verified.
    assert (loaded.obs["smoke_type"] == 5).all()


def test_loader_never_maps_missing_smoke_metadata_to_cigarette(tmp_path):
    h5ad = tmp_path / "gse136831_like.h5ad"
    _write_gse136831_like_h5ad(h5ad)
    loaded = load_scrna(str(h5ad), "unknown", "donor_id")
    cigarette_id = SMOKE_TYPE_MAP["cigarette"]
    assert not (loaded.obs["smoke_type"] == cigarette_id).any()


# ─── Weak-proxy opt-in gate ──────────────────────────────────────────────────

def test_weak_proxy_disabled_by_default_leaves_smoke_type_unknown(tmp_path):
    h5ad = tmp_path / "gse136831_like.h5ad"
    _write_gse136831_like_h5ad(h5ad)
    loaded = load_scrna(str(h5ad), "unknown", "donor_id")
    out = apply_weak_smoke_proxies(loaded, enabled=False)
    assert not out.obs["smoke_type_known"].any()
    assert (out.obs["smoke_type_name"] == "unknown").all()


def test_weak_proxy_opt_in_promotes_only_copd_cells(tmp_path):
    h5ad = tmp_path / "gse136831_like.h5ad"
    _write_gse136831_like_h5ad(h5ad, n_copd=6, n_control=6)
    loaded = load_scrna(str(h5ad), "unknown", "donor_id")
    out = apply_weak_smoke_proxies(loaded, enabled=True)

    copd_mask = out.obs["weak_smoke_proxy_known"].astype(bool)
    assert out.obs.loc[copd_mask, "smoke_type_known"].all()
    assert (out.obs.loc[copd_mask, "smoke_type_name"] == "cigarette").all()
    assert (out.obs.loc[copd_mask, "smoke_type"] == SMOKE_TYPE_MAP["cigarette"]).all()
    # Control cells (no weak proxy) are unaffected by the opt-in.
    assert not out.obs.loc[~copd_mask, "smoke_type_known"].any()


# ─── Assembly / export propagate smoke_known ────────────────────────────────

def test_export_cell_dataset_writes_smoke_labels_known(tmp_path):
    h5ad = tmp_path / "gse136831_like.h5ad"
    _write_gse136831_like_h5ad(h5ad)
    loaded = load_scrna(str(h5ad), "unknown", "donor_id")
    loaded.obs["subject_id"] = loaded.obs["donor_id"]
    loaded.obs["cell_type_id"] = 0
    out_dir = tmp_path / "processed"
    cell_data = export_cell_dataset(loaded, str(out_dir))
    assert "smoke_labels_known" in cell_data
    assert not cell_data["smoke_labels_known"].any()
    assert (out_dir / "smoke_labels_known.npy").exists()


def test_assemble_subject_bags_carries_smoke_known_per_bag(tmp_path):
    h5ad = tmp_path / "gse136831_like.h5ad"
    _write_gse136831_like_h5ad(h5ad, n_copd=60, n_control=60)
    loaded = load_scrna(str(h5ad), "unknown", "donor_id")
    loaded.obs["subject_id"] = loaded.obs["donor_id"]
    loaded.obs["cell_type_id"] = 0
    bags = assemble_subject_bags(loaded, cancer_outcomes=None, min_cells_per_subject=5)
    assert len(bags) > 0
    for b in bags:
        assert "smoke_known" in b
        assert not np.asarray(b["smoke_known"]).any()


# ─── Class weighting / sampling / loss must ignore unknown smoke cells ─────

def test_compute_smoke_class_weights_ignores_unknown_cells():
    labels = np.array([0, 0, 0, 1, 1, 2])
    known  = np.array([True, True, False, True, False, True])
    # Corrupt the unknown cells' placeholder values arbitrarily — must not
    # change the resulting weights at all.
    corrupted_labels = labels.copy()
    corrupted_labels[~known] = 4

    w1 = compute_smoke_class_weights(labels, n_classes=6, smoke_known=known)
    w2 = compute_smoke_class_weights(corrupted_labels, n_classes=6, smoke_known=known)
    assert np.allclose(w1, w2)


def test_cell_level_dataset_smoke_class_weights_masks_unknown():
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from train import CellLevelDataset

    n = 30
    smoke = np.array([0] * 10 + [1] * 10 + [2] * 10)
    known = np.array([True] * 20 + [False] * 10)
    ds = CellLevelDataset(
        gene_matrix=np.random.rand(n, 5).astype("float32"),
        smoke_labels=smoke,
        malignancy_labels=np.zeros(n, dtype="float32"),
        cell_type_ids=np.zeros(n, dtype="int64"),
        smoke_known=known,
        diagnostic_mode=True,
    )
    weights = ds.smoke_class_weights(num_classes=6)
    # Class 2 (all unknown) must get weight 0 — no known cell contributes to it.
    assert weights[2].item() == 0.0

    # Corrupting the unknown cells' placeholder label must not change weights.
    ds2 = CellLevelDataset(
        gene_matrix=ds.X.numpy(),
        smoke_labels=np.where(known, smoke, 5),  # arbitrary corruption
        malignancy_labels=np.zeros(n, dtype="float32"),
        cell_type_ids=np.zeros(n, dtype="int64"),
        smoke_known=known,
        diagnostic_mode=True,
    )
    weights2 = ds2.smoke_class_weights(num_classes=6)
    assert torch_allclose(weights, weights2)


def torch_allclose(a, b):
    import torch
    return torch.allclose(a, b)


def test_multi_task_loss_smoke_component_ignores_unknown_cells():
    import torch
    from model import MultiTaskLoss

    torch.manual_seed(0)
    loss_fn = MultiTaskLoss()
    logits = torch.randn(8, 6, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])
    known = torch.tensor([True, True, True, True, False, False, False, False])

    l1 = loss_fn._ls(logits, targets, known)
    corrupted_targets = targets.clone()
    corrupted_targets[~known] = 5  # arbitrary corruption of unknown placeholders
    l2 = loss_fn._ls(logits, corrupted_targets, known)
    assert torch.isclose(l1, l2)


def test_subject_class_index_excludes_unknown_smoke_cells_but_keeps_indices_valid():
    from data.sampling import SubjectClassIndex

    subject_ids = np.array(["s1"] * 5 + ["s2"] * 5)
    labels = np.array([0] * 5 + [1] * 5)
    known = np.array([True, True, True, False, False, True, True, True, True, True])

    index = SubjectClassIndex(subject_ids=subject_ids, labels=labels, num_classes=6, known_mask=known)
    # s1's known-cell indices must be a subset of the TRUE global positions (0..4).
    assert set(index.subject_to_indices["s1"].tolist()) == {0, 1, 2}
    assert set(index.subject_to_indices["s2"].tolist()) == {5, 6, 7, 8, 9}


# ─── Evaluation must exclude weak-proxy / unknown cells under verified_only ─

def test_evaluate_known_smoke_metrics_excludes_unknown_cells():
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from evaluate import Evaluator

    p = {
        "smoke_true": [0, 1, 2, 3],
        "smoke_pred": [0, 1, 5, 5],   # cells 2/3 wildly wrong but UNKNOWN
        "smoke_known": [True, True, False, False],
    }

    class _FakeEvaluator(Evaluator):
        def __init__(self):
            self.label_mapping = None

    ev = _FakeEvaluator()
    metrics = ev._known_smoke_metrics(p)
    assert metrics["n_known"] == 2
    assert metrics["n_unknown"] == 2
    assert metrics["accuracy"] == 1.0  # only the two KNOWN, correct cells count


# ─── configs/default.yaml agrees with the documented opt-in-only default ───

def test_default_config_weak_labels_disabled_by_default():
    """A regression guard against config drift: if someone accidentally
    flips configs/default.yaml's data.weak_labels.enabled to true,
    GSE136831's COPD proxy would silently start feeding smoke-
    classification supervision on every default run. This test fails
    loudly the moment that default changes."""
    import yaml

    repo_root = Path(__file__).parents[1]
    with open(repo_root / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["data"]["weak_labels"]["enabled"] is False
