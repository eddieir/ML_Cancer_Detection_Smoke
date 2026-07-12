"""
data/assembly.py — merging sources, building MIL bags, and exporting.
No label logic, no transformation logic.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from constants import DOSE_UNKNOWN


def merge_sources(*adatas: ad.AnnData) -> ad.AnnData:
    """
    Concatenate heterogeneous sources on common gene intersection.
    Assigns batch column for downstream Harmony correction.
    Scales the merged matrix (z-score) once across the full distribution.
    """
    genes = adatas[0].var_names
    for a in adatas[1:]:
        genes = genes.intersection(a.var_names)
    genes = list(genes)
    print(f"[assembly] merge  {len(genes):,} common genes across {len(adatas)} sources")

    clipped = []
    for i, a in enumerate(adatas):
        c = a[:, genes].copy()
        c.obs["batch"] = f"source_{i}"
        clipped.append(c)

    merged = ad.concat(clipped, axis=0, join="inner", label="source")
    merged.var_names_make_unique()
    import scipy.sparse as sp
    if sp.issparse(merged.X):
        merged.X = merged.X.toarray()              # explicit: avoids UserWarning from scale()
    sc.pp.scale(merged, max_value=10)
    print(f"[assembly] merge  final {merged.n_obs:,} cells x {merged.n_vars:,} genes")
    return merged


def assemble_subject_bags(
    adata: ad.AnnData,
    cancer_outcomes: Optional[pd.DataFrame] = None,
    min_cells_per_subject: int = 50,
) -> list:
    """
    Group cells by subject_id into MIL bags for Phase 2/3 training.
    Novel: first MIL bag construction from heterogeneous multi-source
    scRNA-seq smoke data linked to NLST cancer outcomes.

    Parameters
    ----------
    cancer_outcomes : DataFrame[subject_id, cancer_label]
    """
    outcome_map: dict = {}
    if cancer_outcomes is not None:
        outcome_map = dict(zip(
            cancer_outcomes["subject_id"].astype(str),
            cancer_outcomes["cancer_label"].astype(int),
        ))

    X = np.array(
        adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray(),
        dtype=np.float32,
    )
    bags, skipped = [], 0

    for sid in adata.obs["subject_id"].unique():
        mask = (adata.obs["subject_id"] == sid).values
        if mask.sum() < min_cells_per_subject:
            skipped += 1
            continue
        bags.append({
            "subject_id":    sid,
            "gene_matrix":   X[mask],
            "cell_type_ids": adata.obs["cell_type_id"].values[mask].astype(int),
            "smoke_labels":  adata.obs["smoke_type"].values[mask].astype(int),
            "malig_labels":  adata.obs["malignancy"].values[mask].astype(np.float32),
            "cancer_label":  outcome_map.get(str(sid), 0),
        })

    print(f"[assembly] bags  {len(bags)} subjects  |  {skipped} dropped (<{min_cells_per_subject} cells)")
    return bags


def export_cell_dataset(
    adata: ad.AnnData,
    out_dir: str = "data/processed",
) -> dict:
    """Save numpy arrays for CellLevelDataset (Phase 1 training)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X     = np.array(adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray(), dtype=np.float32)
    smoke = adata.obs["smoke_type"].values.astype(np.int64)
    malig = adata.obs["malignancy"].values.astype(np.float32)
    ctype = adata.obs["cell_type_id"].values.astype(np.int64)
    dose  = (adata.obs["exposure_dose"].values.astype(np.float32)
             if "exposure_dose" in adata.obs.columns
             else np.full(X.shape[0], DOSE_UNKNOWN, dtype=np.float32))

    np.save(out / "gene_matrix.npy",       X)
    np.save(out / "smoke_labels.npy",      smoke)
    np.save(out / "malignancy_labels.npy", malig)
    np.save(out / "cell_type_ids.npy",     ctype)
    np.save(out / "exposure_dose.npy",     dose)
    adata.obs.to_csv(out / "cell_metadata.csv")
    adata.var.to_csv(out / "gene_list.csv")

    n_known = int((dose >= 0).sum())
    print(f"[assembly] export  {X.shape[0]:,} x {X.shape[1]} → {out}/")
    print(f"           smoke   {dict((i, int((smoke==i).sum())) for i in range(6))}")
    print(f"           malig   {malig.mean():.2%} positive")
    print(f"           dose    {n_known:,} cells with known exposure duration")
    return {"gene_matrix": X, "smoke_labels": smoke,
            "malignancy_labels": malig, "cell_type_ids": ctype, "exposure_dose": dose}
