"""
data/labellers.py — assigns smoke type, malignancy, and class weights.
No I/O, no transformation logic.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import anndata as ad

from constants import SMOKE_TYPE_MAP, SMOKE_TYPES, N_SMOKE_CLASSES


def transfer_nlst_labels(
    adata: ad.AnnData,
    nlst_csv: str,
    subject_col: str = "subject_id",
) -> ad.AnnData:
    """
    Transfer cigar and dual-use labels from NLST clinical metadata
    to cells matched by subject_id.

    Novel: first linkage of NLST clinical smoking categories to
    single-cell gene expression data.

    NLST fields: CIGSMOK (cigarette), CIGAR (cigar use flag).
    """
    if not Path(nlst_csv).exists():
        print("[label] NLST CSV not found — skipping label transfer")
        return adata

    nlst = pd.read_csv(nlst_csv, low_memory=False)
    nlst["subject_id"] = nlst["pid"].astype(str)

    def _assign(row) -> int:
        cigar = row.get("CIGAR",   0) == 1
        cig   = row.get("CIGSMOK", 0) in [1, 2]
        if cigar and cig: return 4   # dual-use
        if cigar:         return 2   # cigar only
        if cig:           return 0   # cigarette only
        return 5                     # unexposed

    nlst_map = dict(zip(nlst["subject_id"], nlst.apply(_assign, axis=1)))
    matched  = adata.obs[subject_col].map(nlst_map)
    n        = matched.notna().sum()

    if n > 0:
        adata.obs.loc[matched.notna(), "smoke_type"] = matched.dropna().astype(int)
        print(f"[label] NLST  {n:,} cells relabelled ({n/adata.n_obs:.1%})")
    else:
        print("[label] NLST  no subject ID overlap — labels unchanged")
    return adata


def add_malignancy_labels(
    adata: ad.AnnData,
    tumor_barcodes: Optional[list] = None,
) -> ad.AnnData:
    """
    Assign per-cell malignancy labels.
    Priority: tumor_barcodes list > values a loader already set in obs
    (e.g. TCGA tumor/NAT via convert_tcga's samples_meta.csv) > 0.0 default.
    """
    if "malignancy" not in adata.obs.columns:
        adata.obs["malignancy"] = 0.0

    if tumor_barcodes:
        mask = adata.obs_names.isin(set(tumor_barcodes))
        adata.obs.loc[mask, "malignancy"] = 1.0
        print(f"[label] malignancy  {mask.sum():,} tumor cells set to 1.0")

    adata.obs["malignancy"] = adata.obs["malignancy"].astype(np.float32)
    return adata


def compute_smoke_class_weights(
    smoke_labels: np.ndarray,
    n_classes: int = N_SMOKE_CLASSES,
) -> np.ndarray:
    """
    Inverse-frequency weights for CrossEntropyLoss.
    Cannabis and dual-use are severely under-represented vs cigarette.
    Returns array of shape [n_classes] for direct use in nn.CrossEntropyLoss.
    """
    counts  = np.bincount(smoke_labels, minlength=n_classes).astype(np.float32)
    counts  = np.maximum(counts, 1)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.sum() * n_classes   # sum to n_classes

    for i, (w, c) in enumerate(zip(weights, counts)):
        print(f"[label] weight  {SMOKE_TYPES[i]:<12}  count={int(c):>6,}  weight={w:.3f}")
    return weights
