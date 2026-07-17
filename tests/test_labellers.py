"""data/labellers.py — smoke type label transfer, malignancy labels, class weights."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import N_SMOKE_CLASSES


def _adata_with_subjects(subject_ids):
    n = len(subject_ids)
    obs = pd.DataFrame({
        "subject_id": subject_ids,
        "smoke_type": [0] * n,   # all start as cigarette; NLST should relabel some
    }, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=np.zeros((n, 3), dtype="float32"), obs=obs,
                       var=pd.DataFrame(index=["G1", "G2", "G3"]))


def test_transfer_nlst_labels_assigns_cigar_and_dual_use():
    """p4 has CIGSMOK=0/CIGAR=0 — neither is a documented positive code
    (see data/nlst_smoking.py), so p4 must remain UNKNOWN, never a
    verified 'unexposed' class. See
    test_transfer_nlst_labels_unsupported_codes_stay_unknown below for the
    dedicated known/unknown assertions this predates."""
    from data.labellers import transfer_nlst_labels
    adata = _adata_with_subjects(["p1", "p2", "p3", "p4"])
    with tempfile.TemporaryDirectory() as tmp:
        nlst_csv = Path(tmp) / "prsn.csv"
        pd.DataFrame({
            "pid":     ["p1", "p2", "p3", "p4"],
            "CIGSMOK": [1, 0, 1, 0],   # p1, p3 currently smoke cigarettes
            "CIGAR":   [0, 1, 1, 0],   # p2, p3 smoke cigars
        }).to_csv(nlst_csv, index=False)

        out = transfer_nlst_labels(adata, str(nlst_csv))
        labels = out.obs.set_index("subject_id")["smoke_type"].groupby(level=0).first()
        known = out.obs.set_index("subject_id")["smoke_type_known"].groupby(level=0).first()
        assert labels["p1"] == 0 and known["p1"]   # cigarette only, verified
        assert labels["p2"] == 2 and known["p2"]   # cigar only, verified
        assert labels["p3"] == 4 and known["p3"]   # dual-use, verified
        assert not known["p4"]                     # CIGSMOK=0/CIGAR=0 — undocumented codes, unknown


def test_transfer_nlst_labels_skips_when_file_missing():
    from data.labellers import transfer_nlst_labels
    adata = _adata_with_subjects(["p1", "p2"])
    out = transfer_nlst_labels(adata, "/nonexistent/path/prsn.csv")
    assert (out.obs["smoke_type"] == 0).all()  # unchanged


def test_add_malignancy_labels_with_tumor_barcodes():
    from data.labellers import add_malignancy_labels
    n = 5
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=np.zeros((n, 2), dtype="float32"), obs=obs,
                        var=pd.DataFrame(index=["G1", "G2"]))
    out = add_malignancy_labels(adata, tumor_barcodes=["c0", "c2"])
    assert out.obs["malignancy"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert out.obs["malignancy"].dtype == np.float32


def test_add_malignancy_labels_defaults_to_zero_without_barcodes():
    from data.labellers import add_malignancy_labels
    n = 3
    adata = ad.AnnData(X=np.zeros((n, 2), dtype="float32"),
                        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                        var=pd.DataFrame(index=["G1", "G2"]))
    out = add_malignancy_labels(adata, tumor_barcodes=None)
    assert (out.obs["malignancy"] == 0.0).all()


def test_add_malignancy_labels_marks_known_and_unknown():
    from data.labellers import add_malignancy_labels
    n = 5
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=np.zeros((n, 2), dtype="float32"), obs=obs,
                        var=pd.DataFrame(index=["G1", "G2"]))
    out = add_malignancy_labels(adata, tumor_barcodes=["c0", "c2"])
    assert out.obs["malignancy_known"].tolist() == [True, False, True, False, False]


def test_add_malignancy_labels_without_barcodes_is_entirely_unknown():
    from data.labellers import add_malignancy_labels
    n = 3
    adata = ad.AnnData(X=np.zeros((n, 2), dtype="float32"),
                        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                        var=pd.DataFrame(index=["G1", "G2"]))
    out = add_malignancy_labels(adata, tumor_barcodes=None)
    assert (out.obs["malignancy_known"] == False).all()  # noqa: E712 — pandas needs elementwise ==


def test_add_malignancy_labels_preserves_existing_labels():
    """TCGA-style: malignancy already set per-sample by the loader — don't overwrite it."""
    from data.labellers import add_malignancy_labels
    n = 3
    obs = pd.DataFrame({"malignancy": [1.0, 0.0, 1.0]}, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=np.zeros((n, 2), dtype="float32"), obs=obs,
                        var=pd.DataFrame(index=["G1", "G2"]))
    out = add_malignancy_labels(adata, tumor_barcodes=None)
    assert out.obs["malignancy"].tolist() == [1.0, 0.0, 1.0]


def test_compute_smoke_class_weights_upweights_rare_classes():
    from data.labellers import compute_smoke_class_weights
    # Class 0 dominant, class 3 (cannabis) rare.
    labels = np.array([0] * 90 + [3] * 2 + [1, 2, 4, 5])
    weights = compute_smoke_class_weights(labels, n_classes=N_SMOKE_CLASSES)
    assert weights.shape == (N_SMOKE_CLASSES,)
    assert weights[3] > weights[0]   # rarer class gets a larger weight
    assert np.isclose(weights.sum(), N_SMOKE_CLASSES, atol=1e-3)
