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

from constants import SMOKE_TYPE_MAP, DOSE_UNKNOWN


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
    adata.obs["exposure_dose"]   = DOSE_UNKNOWN # no wired source currently supplies a real dose (see DoseResponseHead)
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

    If a sibling `<name>_samples_meta.csv` exists (written by
    data/converters.py), per-sample columns override the blanket defaults:
      smoke_type  — e.g. GSE994 contains both smokers and never-smokers
                    in one series matrix.
      malignancy  — e.g. TCGA tumor/NAT samples (convert_tcga), which have
                    a real per-sample malignancy label instead of the 0.0
                    default.
      subject_id  — e.g. TCGA case_id, needed so tumor/NAT samples from the
                    same patient share one MIL bag instead of each sample
                    becoming its own "subject".
    """
    df    = pd.read_csv(csv_path, index_col=0)       # genes x samples
    X     = df.T.values.astype(np.float32)
    obs   = pd.DataFrame(index=df.columns)
    sid   = pd.Series(df.columns.astype(str), index=df.columns)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=df.index))
    adata = _attach_standard_obs(adata, smoke_type, sid, "microarray", True)

    meta_path = Path(csv_path).with_name(Path(csv_path).stem + "_samples_meta.csv")
    if meta_path.exists():
        meta = pd.read_csv(meta_path).set_index("sample_id")

        if "smoke_type" in meta.columns:
            per_sample = adata.obs_names.map(meta["smoke_type"]).fillna(smoke_type)
            adata.obs["smoke_type_name"] = per_sample.values
            adata.obs["smoke_type"] = per_sample.map(
                lambda s: SMOKE_TYPE_MAP.get(str(s).lower(), 5)
            ).values
            n_over = (per_sample != smoke_type).sum()
            print(f"[loader] microarray  {n_over} samples relabelled from {meta_path.name}")

        if "malignancy" in meta.columns:
            adata.obs["malignancy"] = (
                adata.obs_names.map(meta["malignancy"]).fillna(0.0).astype(np.float32).values
            )
            print(f"[loader] microarray  malignancy labels loaded from {meta_path.name}")

        if "subject_id" in meta.columns:
            override = adata.obs_names.to_series().map(meta["subject_id"])
            adata.obs["subject_id"] = override.fillna(sid).values

    print(f"[loader] microarray {adata.n_obs:>7,} samples  {smoke_type}  {Path(csv_path).name}")
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