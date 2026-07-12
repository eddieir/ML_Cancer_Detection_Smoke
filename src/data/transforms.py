"""
data/transforms.py — pure data transformation. No I/O, no label logic.
Each function takes AnnData, returns AnnData.
"""

from typing import Optional
import numpy as np
import anndata as ad
import scanpy as sc

from constants import ALL_SMOKE_MARKERS


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
    adata.obsm["X_pca_harmony"] = ho.Z_corr.T
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
