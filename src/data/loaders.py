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

from constants import SMOKE_TYPE_MAP, DOSE_UNKNOWN, SPECIES_HUMAN, SPECIES_MOUSE, ASSAY_MODE_SINGLE_CELL


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
                          modality: str, is_pseudo_bulk: bool,
                          species: str = SPECIES_HUMAN) -> ad.AnnData:
    """
    DRY helper: stamps required obs columns onto any AnnData.

    smoke_type is a blanket default for every cell in this source — correct
    for single-condition sources (e.g. GSE994 is real smokers) but wrong
    for multi-condition ones (e.g. GSE288003 has both e-cig-exposed AND
    unexposed control mice under one accession). If the converter already
    wrote a real per-cell 'smoke_type_name' (converters.py does this for
    GSE288003's Con/E-cigs split), keep it instead of overwriting every
    cell with the single blanket label — same per-sample-override pattern
    load_microarray() already uses via its _samples_meta.csv sidecar.
    """
    # smoke_type_known: True unless the converter already stamped a real
    # per-cell value (e.g. converters.py's GSE136831 handling sets this
    # False for every cell — no verified smoking-status measurement exists
    # for that accession; see _load_gse136831_cell_metadata). A source with
    # no such column at all is treated as carrying its existing, already-
    # reviewed accession-level/per-sample label at face value — unchanged
    # legacy behaviour for every source except GSE136831.
    had_known_col = "smoke_type_known" in adata.obs.columns
    smoke_type_known = (
        adata.obs["smoke_type_known"].astype(bool)
        if had_known_col else pd.Series(True, index=adata.obs_names)
    )

    if "smoke_type_name" in adata.obs.columns:
        per_cell = adata.obs["smoke_type_name"].astype(str)
        n_over = (per_cell.str.lower() != smoke_type.lower()).sum()
        if n_over:
            print(f"[loader] {modality}  {n_over:,}/{adata.n_obs:,} cells kept "
                  f"real per-cell smoke_type (differs from default '{smoke_type}')")
        adata.obs["smoke_type_name"] = per_cell.str.lower().values
        adata.obs["smoke_type"] = per_cell.str.lower().map(
            lambda s: SMOKE_TYPE_MAP.get(s, 5)
        ).values
    else:
        adata.obs["smoke_type"]      = SMOKE_TYPE_MAP.get(smoke_type.lower(), 5)
        adata.obs["smoke_type_name"] = smoke_type.lower()
    adata.obs["smoke_type_known"] = smoke_type_known.values
    n_unknown_smoke = int((~smoke_type_known).sum())
    if n_unknown_smoke:
        print(f"[loader] {modality}  {n_unknown_smoke:,}/{adata.n_obs:,} cells have "
              "smoke_type_known=False (no verified smoking-status label) — excluded "
              "from smoke-classification supervision under the default verified_only "
              "label policy; see weak_smoke_proxy_* columns if present.")
    # weak_smoke_proxy_* columns are passthrough-only here: a source that
    # already carries them (GSE136831) keeps its real per-cell values;
    # every other source gets the safe "no proxy" defaults so downstream
    # code can rely on these columns always being present.
    for col, default in (
        ("weak_smoke_proxy_known", False),
        ("weak_smoke_proxy_type", None),
        ("weak_smoke_proxy_value", None),
        ("weak_smoke_proxy_source", None),
        ("weak_smoke_proxy_limitation", None),
    ):
        if col not in adata.obs.columns:
            adata.obs[col] = default
    adata.obs["data_modality"]   = modality
    adata.obs["is_pseudo_bulk"]  = is_pseudo_bulk
    adata.obs["species"]         = species
    # assay_mode defaults to human_single_cell for every source — only
    # TCGA (load_microarray's samples_meta.csv "assay_mode" column, written
    # by data/converters.py::convert_tcga) overrides this to "bulk_tcga".
    # See data/assay_mode.py for the enforcement point that keeps bulk_tcga
    # rows out of the single-cell pipeline.
    if "assay_mode" not in adata.obs.columns:
        adata.obs["assay_mode"] = ASSAY_MODE_SINGLE_CELL
    adata.obs["subject_id"]      = subject_id_series.values
    adata.obs["malignancy"]      = 0.0          # overwritten by labellers.py
    adata.obs["malignancy_known"] = False       # True only where a real label exists (see labellers.py)
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

        if "smoke_type_known" in meta.columns:
            known_mapped = adata.obs_names.to_series().map(meta["smoke_type_known"])
            # A sample absent from meta's smoke_type_known column keeps
            # whatever _attach_standard_obs already set (True, the legacy
            # default) — .fillna(True) here, NOT False, so this column's
            # mere presence for SOME samples (e.g. TCGA's explicit
            # smoke_type_known=False) can't silently downgrade OTHER
            # samples that meta simply didn't annotate.
            adata.obs["smoke_type_known"] = known_mapped.fillna(True).astype(bool).values
            n_unknown = int((~adata.obs["smoke_type_known"]).sum())
            if n_unknown:
                print(f"[loader] microarray  {n_unknown}/{adata.n_obs} samples have "
                      f"smoke_type_known=False from {meta_path.name} — no verified smoking "
                      "history for this cohort; excluded from smoke-classification supervision.")

        if "assay_mode" in meta.columns:
            mapped_mode = adata.obs_names.to_series().map(meta["assay_mode"])
            adata.obs["assay_mode"] = mapped_mode.fillna(ASSAY_MODE_SINGLE_CELL).values

        if "malignancy" in meta.columns:
            mapped = adata.obs_names.to_series().map(meta["malignancy"])
            adata.obs["malignancy"] = mapped.fillna(0.0).astype(np.float32).values
            adata.obs["malignancy_known"] = mapped.notna().values
            print(f"[loader] microarray  malignancy labels loaded from {meta_path.name}  "
                  f"({int(mapped.notna().sum())}/{adata.n_obs} samples have a real label)")

        if "subject_id" in meta.columns:
            override = adata.obs_names.to_series().map(meta["subject_id"])
            adata.obs["subject_id"] = override.fillna(sid).values

    print(f"[loader] microarray {adata.n_obs:>7,} samples  {smoke_type}  {Path(csv_path).name}")
    return adata


def load_mouse_scrna(h5ad_path: str) -> ad.AnnData:
    """
    Load GSE288003 (mouse lung cells, e-cig aerosol).
    Ortholog mapping happens in transforms.py, not here.

    obs["species"] is stamped "mouse" and every subject/animal id is
    namespaced ("mouse::<id>") so it can never collide with a human
    subject_id sharing the same raw string once this source is merged
    with anything else — see data/species_policy.py. This is the ONLY
    loader that produces non-human cells; every other loader in this
    module defaults to species="human" via _attach_standard_obs.
    """
    from data.species_policy import namespace_subject_id

    adata = sc.read_h5ad(h5ad_path)
    sid   = (adata.obs["donor_id"].astype(str)
             if "donor_id" in adata.obs.columns
             else pd.Series(["mouse_unknown"] * adata.n_obs, index=adata.obs_names))
    sid   = sid.map(lambda s: namespace_subject_id(s, SPECIES_MOUSE))
    adata = _attach_standard_obs(adata, "vape", sid, "mouse_scrna", False, species=SPECIES_MOUSE)
    print(f"[loader] mouse scrna {adata.n_obs:>6,} cells    vape  {Path(h5ad_path).name}  "
          f"(species=mouse, subjects namespaced)")
    return adata