"""
data/transforms.py — pure data transformation. No I/O, no label logic.
Each function takes AnnData, returns AnnData.
"""

import re
from typing import Optional
import numpy as np
import anndata as ad
import scanpy as sc

from constants import ALL_SMOKE_MARKERS


# ─── Gene ID harmonization ────────────────────────────────────────────────────

# Platform-specific probe/gene ID patterns -> BioMart attribute name that maps
# them to human gene symbols. Detected by sniffing var_names, so callers don't
# need to know each source's platform ahead of time.
_ID_PATTERNS: dict[str, "re.Pattern"] = {
    "affy_hg_u133a_2": re.compile(r"^\d+(_[a-z]+)?_at$", re.I),
    "ensembl_gene_id": re.compile(r"^ENSG\d+(\.\d+)?$"),
}


def _detect_platform(var_names) -> Optional[str]:
    sample = list(var_names[:20])
    if not sample:
        return None
    for platform, pattern in _ID_PATTERNS.items():
        if all(pattern.match(str(v)) for v in sample):
            return platform
    return None


def harmonize_gene_ids(adata: ad.AnnData, mapping: Optional[dict] = None) -> ad.AnnData:
    """
    Maps platform-specific probe/gene IDs (Affymetrix HG-U133A probes,
    Ensembl gene IDs) to human gene symbols via BioMart, so merge_sources()
    can find real overlap across heterogeneous sources instead of the zero
    overlap you get when one source uses "1007_s_at" and another uses
    "ENSG00000000003" for the same gene.

    Auto-detects platform from var_names; sources already using gene symbols
    (no pattern match) pass through unchanged. Illumina HumanHT-12 probes
    (e.g. GSE123352's "ILMN_...") are NOT covered — Ensembl's BioMart doesn't
    expose that array as a queryable attribute, so those sources need GEO's
    own GPL platform annotation file instead and are deliberately left out of
    auto-detection here rather than silently mismapped.

    `mapping` lets tests (and repeat calls across sources on the same
    platform) skip the live BioMart query.
    """
    platform = _detect_platform(adata.var_names)
    if platform is None:
        return adata  # already gene symbols, or an unrecognised/unsupported platform

    if mapping is None:
        try:
            from pybiomart import Dataset
            ds = Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")
            df = ds.query(attributes=[platform, "external_gene_name"]).dropna()
            mapping = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        except Exception as e:
            print(f"[transform] gene ID mapping unavailable ({e}) — skipping harmonization")
            return adata

    new_names = [mapping.get(g, "") for g in adata.var_names]
    valid = [i for i, n in enumerate(new_names) if n]
    adata = adata[:, valid].copy()
    adata.var_names = [new_names[i] for i in valid]
    adata.var_names_make_unique()
    print(f"[transform] {platform}  {len(valid):,}/{len(new_names):,} probes → gene symbols")
    return adata


def map_mouse_to_human(adata: ad.AnnData) -> ad.AnnData:
    """
    Convert mouse gene symbols → human 1:1 orthologs via PyBiomart.
    Novel: enables GSE288003 (only vape scRNA-seq) to train in human gene space.
    """
    try:
        from pybiomart import Dataset
        ds    = Dataset(name="mmusculus_gene_ensembl", host="http://www.ensembl.org")
        ortho = ds.query(
            attributes=["external_gene_name", "hsapiens_homolog_associated_gene_name"],
            only_unique=False,
        )
        ortho.columns = ["mouse_gene", "human_gene"]
        ortho = ortho.dropna().drop_duplicates("mouse_gene")
        omap  = dict(zip(ortho["mouse_gene"], ortho["human_gene"]))
    except Exception as e:
        print(f"[transform] ortholog mapping unavailable ({e}) — skipping")
        return adata

    new_names = [omap.get(g, "") for g in adata.var_names]
    valid     = [i for i, n in enumerate(new_names) if n]
    adata     = adata[:, valid].copy()
    adata.var_names = [new_names[i] for i in valid]
    adata.var_names_make_unique()
    print(f"[transform] {len(valid):,} mouse genes → human orthologs retained")
    return adata


def qc_filter(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
) -> ad.AnnData:
    """Standard QC. Skips per-cell thresholds for pseudo-bulk."""
    if adata.obs["is_pseudo_bulk"].all():
        sc.pp.filter_genes(adata, min_cells=1)
        return adata
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # percent_top defaults to [50,100,200,500] and raises IndexError on any
    # dataset with fewer than 500 genes (e.g. HVG-reduced or small test data);
    # we don't use that metric, so disable it rather than requiring >=500 genes.
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    n_before = adata.n_obs
    adata = adata[adata.obs.n_genes_by_counts >= min_genes].copy()
    adata = adata[adata.obs.pct_counts_mt    <= max_pct_mito].copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"[transform] qc  {n_before:,} → {adata.n_obs:,} cells")
    return adata


def normalize(adata: ad.AnnData) -> ad.AnnData:
    """CPM + log1p. Skips if microarray (already log-transformed)."""
    adata.layers["counts"] = adata.X.copy()
    if adata.obs["data_modality"].iloc[0] == "microarray":
        adata.layers["lognorm"] = adata.X.copy()   # already log-scaled
        return adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()        # preserve before any scaling
    return adata


def smoke_aware_hvg(
    adata: ad.AnnData,
    n_hvgs: int = 2000,
    batch_key: Optional[str] = "batch",
) -> ad.AnnData:
    """
    Variance-based HVG selection with forced inclusion of smoke markers.
    Novel: prevents standard HVG from dropping smoke-discriminative genes
    when one smoke type dominates the dataset distribution.
    """
    bk = batch_key if (batch_key and batch_key in adata.obs.columns) else None
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, batch_key=bk)

    present = [g for g in ALL_SMOKE_MARKERS if g in adata.var_names]
    forced  = [g for g in present if not adata.var.loc[g, "highly_variable"]]

    if forced:
        non_marker_hvgs = [
            g for g in adata.var_names[adata.var["highly_variable"]]
            if g not in present
        ]
        drop = non_marker_hvgs[-len(forced):]
        adata.var.loc[drop,   "highly_variable"] = False
        adata.var.loc[forced, "highly_variable"] = True
        print(f"[transform] hvg  forced {len(forced)} smoke markers in")

    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"[transform] hvg  {adata.n_vars} genes selected")
    return adata


def batch_correct(adata: ad.AnnData, batch_key: str = "batch") -> ad.AnnData:
    """Harmony batch correction on PCA embeddings."""
    if batch_key not in adata.obs.columns or adata.obs[batch_key].nunique() < 2:
        print("[transform] harmony skipped — fewer than 2 batches")
        return adata
    import harmonypy as hm
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)
    ho = hm.run_harmony(
        adata.obsm["X_pca"], adata.obs, batch_key, max_iter_harmony=20
    )
    # harmonypy's Z_corr orientation has flipped across versions (some
    # return n_pcs x n_cells, others n_cells x n_pcs) — orient against the
    # known n_obs rather than assuming a fixed convention.
    z_corr = ho.Z_corr if ho.Z_corr.shape[0] == adata.n_obs else ho.Z_corr.T
    adata.obsm["X_pca_harmony"] = z_corr
    print(f"[transform] harmony  {adata.obs[batch_key].nunique()} batches corrected")
    return adata


def annotate_cell_types(adata: ad.AnnData) -> ad.AnnData:
    """CellTypist majority-vote → coarse 4-class cell_type_id."""
    from constants import CELL_TYPE_MAP
    if adata.obs["is_pseudo_bulk"].all():
        print("[transform] celltypist skipped for pseudo-bulk")
        return adata
    try:
        import celltypist
        from celltypist import models

        # CellTypist requires log1p normalized X (CPM → log1p).
        # After merge_sources() X holds z-scores, so we restore lognorm layer.
        ct_input = adata.copy()
        if "lognorm" in adata.layers:
            ct_input.X = adata.layers["lognorm"]

        model = models.Model.load(model="Immune_All_Low.pkl")
        pred  = celltypist.annotate(ct_input, model=model, majority_voting=True)
        adata.obs["cell_type_name"] = pred.predicted_labels.majority_voting.values
        adata.obs["cell_type_id"]   = (
            adata.obs["cell_type_name"]
            .map(lambda c: CELL_TYPE_MAP.get(c, 0))
            .astype(int)
        )
        print(f"[transform] celltypist  {adata.n_obs:,} cells annotated")
    except Exception as e:
        print(f"[transform] celltypist failed ({e}) — defaulting to epithelial")
    return adata
