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
    (e.g. GSE123352's "ILMN_...") are NOT covered here — Ensembl's BioMart
    doesn't expose that array as a queryable attribute, so that mapping
    happens earlier, at conversion time, via GEO's own GPL platform
    annotation file (converters.py::_load_probe_to_symbol_map,
    GSE123352.csv already ships real gene symbols by the time it reaches
    this pipeline) rather than being silently mismapped here.

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


def cell_type_map_fingerprint() -> str:
    """
    SHA-256 of the fixed CELL_TYPE_MAP label-name -> ID table (constants.py),
    so a checkpoint/artifact can record and later verify exactly which
    mapping produced its cell_type_id column. The map itself is a static
    dict, never built from data (held-out frequency/order never enters it),
    so this fingerprint is the same for any run of this codebase — a
    mismatch on reload means the code's mapping table itself changed, not
    that different cells were annotated.
    """
    import hashlib
    import json
    from constants import CELL_TYPE_MAP
    blob = json.dumps(CELL_TYPE_MAP, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class CellTypeAnnotationError(RuntimeError):
    """Raised when CellTypist annotation fails and no diagnostic fallback
    was explicitly requested. Real benchmark runs must never silently
    continue with missing, stale, or fabricated cell-type labels — a prior
    version of this function printed a warning and returned `adata`
    unchanged on failure, leaving obs["cell_type_id"] either absent or
    stale from a previous call, which then surfaced only much later as a
    confusing downstream KeyError or a silently wrong MIL cell-type input."""


_DIAGNOSTIC_FALLBACK_LABEL = "epithelial"


def annotate_cell_types(
    adata: ad.AnnData, majority_voting: bool = False, allow_diagnostic_fallback: bool = False,
) -> ad.AnnData:
    """
    CellTypist per-cell prediction -> coarse 4-class cell_type_id.

    majority_voting defaults to False (INDUCTIVE): CellTypist's
    majority_voting=True mode over-clusters the supplied cells and
    replaces each cell's raw prediction with its cluster's majority label —
    a step whose output for a given cell depends on which OTHER cells are
    present in the same call. Running it once over train+val+test cells (as
    this pipeline used to) therefore lets held-out validation/test cells
    change a training cell's annotation, and reannotating a different
    subject subset per CV fold would silently reassign cell_type_id for
    cells that didn't change at all.

    majority_voting=False instead returns CellTypist's raw per-cell
    prediction from the fixed pretrained model: a pure function of that
    cell's own expression vector, independent of any other cell supplied in
    the same call. A cell's annotation is therefore identical whether it is
    annotated alone, with its own split, or with the full merged dataset —
    the property every fold/OOD reconstruction in benchmarks/ depends on.
    This is CellTypist's own default (majority_voting=False); this pipeline
    previously opted OUT of that default by passing majority_voting=True.

    The label-name -> ID table (CELL_TYPE_MAP) is a fixed dict in
    constants.py, never built from this dataset's label frequency/order —
    see cell_type_map_fingerprint() for the persisted proof of that.

    Failure behavior: if CellTypist itself fails (missing dependency,
    unreachable model download, malformed input, ...), this function raises
    CellTypeAnnotationError by default — a real scientific run must never
    silently proceed with no/fabricated cell types. allow_diagnostic_fallback
    =True is the ONLY way to continue past that failure; it is meant for a
    deliberate small-scale diagnostic run only (never for a real benchmark:
    src/preprocess.py's real pipeline entry points never pass it). The
    fallback assigns EVERY cell the same fixed placeholder label/ID and
    stamps adata.uns["cell_type_annotation_degraded"] = True plus the
    failure reason, so any downstream consumer (ExperimentContext
    validation in particular — see benchmarks/context.py) can detect and
    reject a degraded annotation rather than silently treating it as real.
    """
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
        pred  = celltypist.annotate(ct_input, model=model, majority_voting=majority_voting)
        labels = (
            pred.predicted_labels.majority_voting
            if majority_voting else
            pred.predicted_labels.predicted_labels
        )
        adata.obs["cell_type_name"] = labels.values
        adata.obs["cell_type_id"]   = (
            adata.obs["cell_type_name"]
            .map(lambda c: CELL_TYPE_MAP.get(c, 0))
            .astype(int)
        )
        adata.uns["cell_type_map_fingerprint"] = cell_type_map_fingerprint()
        adata.uns["cell_type_annotation_mode"] = "majority_voting" if majority_voting else "inductive_per_cell"
        adata.uns["cell_type_annotation_degraded"] = False
        print(f"[transform] celltypist  {adata.n_obs:,} cells annotated "
              f"({'majority_voting' if majority_voting else 'inductive per-cell'})")
    except Exception as e:
        if not allow_diagnostic_fallback:
            raise CellTypeAnnotationError(
                f"CellTypist cell-type annotation failed: {e!r}. Refusing to continue with "
                "missing or fabricated cell-type labels for a real run. Pass "
                "allow_diagnostic_fallback=True only for a deliberate small-scale diagnostic "
                "run — never for a real scientific benchmark."
            ) from e
        adata.obs["cell_type_name"] = _DIAGNOSTIC_FALLBACK_LABEL
        adata.obs["cell_type_id"] = int(CELL_TYPE_MAP.get(_DIAGNOSTIC_FALLBACK_LABEL, 0))
        adata.uns["cell_type_map_fingerprint"] = cell_type_map_fingerprint()
        adata.uns["cell_type_annotation_mode"] = "diagnostic_fallback"
        adata.uns["cell_type_annotation_degraded"] = True
        adata.uns["cell_type_fallback_reason"] = repr(e)
        print(f"[transform] celltypist FAILED ({e!r}) — DIAGNOSTIC FALLBACK: every cell labeled "
              f"{_DIAGNOSTIC_FALLBACK_LABEL!r} (degraded, non-scientific annotation; "
              "allow_diagnostic_fallback=True was explicitly set)")
    return adata
