"""
data/loaders.py — I/O only.
Each function loads one data source type and attaches standard obs columns.
No transformation logic here.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from constants import SMOKE_TYPE_MAP


def _try_auto_download(path: Path) -> None:
    """
    If a raw file is missing, check if its parent directory name matches
    a known GEO accession and trigger the downloader automatically.
    """
    from data.downloaders import GEO_DATASETS, download_geo
    for accession, (subdir, *_) in GEO_DATASETS.items():
        if accession in str(path):
            print(f"[loader] {path.name} not found — auto-downloading {accession}")
            download_geo(accession)
            return
    raise FileNotFoundError(
        f"{path} not found.\n"
        "Run: python3 src/data/downloaders.py --check\n"
        "Then: python3 src/data/downloaders.py --geo"
    )


def _attach_standard_obs(adata: ad.AnnData, smoke_type: str,
                          subject_id_series: pd.Series,
                          modality: str, is_pseudo_bulk: bool) -> ad.AnnData:
    """DRY helper: stamps required obs columns onto any AnnData."""
    adata.obs["smoke_type"]      = SMOKE_TYPE_MAP.get(smoke_type.lower(), 5)
    adata.obs["smoke_type_name"] = smoke_type.lower()
    adata.obs["data_modality"]   = modality
    adata.obs["is_pseudo_bulk"]  = is_pseudo_bulk
    adata.obs["subject_id"]      = subject_id_series.values
    adata.obs["malignancy"]      = 0.0          # overwritten by labellers.py
    adata.obs["cell_type_id"]    = 0            # overwritten by transforms.py
    return adata


def load_scrna(h5ad_path: str, smoke_type: str,
               subject_col: str = "donor_id") -> ad.AnnData:
    """Load a true scRNA-seq h5ad. Auto-downloads if missing and accession known."""
    path = Path(h5ad_path)
    if not path.exists():
        _try_auto_download(path)
    adata = sc.read_h5ad(h5ad_path)
    sid = (adata.obs[subject_col].astype(str)
           if subject_col in adata.obs.columns
           else pd.Series(["unknown"] * adata.n_obs, index=adata.obs_names))
    adata = _attach_standard_obs(adata, smoke_type, sid, "scrna", False)
    print(f"[loader] scrna      {adata.n_obs:>7,} cells    {smoke_type}  {Path(h5ad_path).name}")
    return adata


def load_microarray(csv_path: str, smoke_type: str) -> ad.AnnData:
    """
    Load a genes-x-samples bulk microarray CSV (GSE994, GSE123352).
    Each sample becomes one row (pseudo-bulk cell).
    """
    df    = pd.read_csv(csv_path, index_col=0)       # genes x samples
    X     = df.T.values.astype(np.float32)
    obs   = pd.DataFrame(index=df.columns)
    sid   = pd.Series(df.columns.astype(str), index=df.columns)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=df.index))
    adata = _attach_standard_obs(adata, smoke_type, sid, "microarray", True)
    print(f"[loader] microarray {adata.n_obs:>7,} samples  {smoke_type}  {Path(csv_path).name}")
    return adata


def load_pseudo_bulk_loiselle(csv_path: str) -> ad.AnnData:
    """
    Loiselle 2018: bulk RNA-seq on BEAS-2B / NCI-H1975 exposed to
    tobacco or cannabis smoke across 1-17 weeks.
    Each row = one condition; treated as a synthetic cell.

    Expected CSV columns: [gene_1..gene_N, smoke_type, cell_line, week, malignancy?]
    """
    df       = pd.read_csv(csv_path)
    meta     = ["smoke_type", "cell_line", "week", "malignancy"]
    gene_cols = [c for c in df.columns if c not in meta]

    X   = df[gene_cols].values.astype(np.float32)
    obs = pd.DataFrame(index=pd.RangeIndex(len(df)))
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=gene_cols))

    sid = df["cell_line"].astype(str) + "_w" + df["week"].astype(str)
    adata = _attach_standard_obs(adata, "mixed", sid, "pseudo_bulk", True)

    # Override smoke_type per-row since the CSV has per-row labels
    adata.obs["smoke_type"] = df["smoke_type"].map(
        lambda s: SMOKE_TYPE_MAP.get(str(s).lower(), 5)
    ).values
    adata.obs["smoke_type_name"] = df["smoke_type"].values

    # Malignancy: BEAS-2B under continuous CS for >= 10 weeks = malignant
    if "malignancy" in df.columns:
        adata.obs["malignancy"] = df["malignancy"].values.astype(np.float32)
    else:
        adata.obs["malignancy"] = (
            (df["smoke_type"].str.lower().isin(["cigarette", "tobacco"])) &
            (df["week"] >= 10)
        ).astype(np.float32)

    print(f"[loader] loiselle   {adata.n_obs:>7,} conditions  cannabis+tobacco")
    return adata


def load_mouse_scrna(h5ad_path: str) -> ad.AnnData:
    """
    Load GSE288003 (mouse lung cells, e-cig aerosol).
    Ortholog mapping happens in transforms.py, not here.
    """
    adata = sc.read_h5ad(h5ad_path)
    sid   = (adata.obs["donor_id"].astype(str)
             if "donor_id" in adata.obs.columns
             else pd.Series(["mouse_unknown"] * adata.n_obs, index=adata.obs_names))
    adata = _attach_standard_obs(adata, "vape", sid, "mouse_scrna", False)
    print(f"[loader] mouse scrna {adata.n_obs:>6,} cells    vape  {Path(h5ad_path).name}")
    return adata