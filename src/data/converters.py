"""
data/converters.py — raw GEO/NLST downloads → clean formats loaders.py expects.

downloaders.py fetches raw files as GEO/GDC publish them (series_matrix.txt.gz,
10x-style mtx triples, NLST screen/prsn CSVs). loaders.py only knows how to read
already-clean shapes (genes x samples CSV, h5ad). This module is the one-time
bridge between the two, so neither has to know about the other's format quirks.

Usage:
    python3 src/data/converters.py --all
    python3 src/data/converters.py --accession GSE994
    python3 src/data/converters.py --nlst
"""

import argparse
import gzip
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RAW       = Path(__file__).parents[2] / "data" / "raw"
CONVERTED = Path(__file__).parents[2] / "data" / "processed" / "converted"


def _mkout() -> Path:
    CONVERTED.mkdir(parents=True, exist_ok=True)
    return CONVERTED


# ─── GEO series matrix parsing ────────────────────────────────────────────────

def _parse_sample_metadata(gz_path: Path) -> tuple[list[str], pd.DataFrame, list[str]]:
    """
    Shared low-level parse of a GEO `*_series_matrix.txt.gz` file's header
    section: GSM sample IDs (in column order), their `!Sample_characteristics_ch1`
    fields as a metadata DataFrame, and the raw table lines (may be empty when
    the series ships its expression matrix as a separate supplementary file
    instead of embedding it, e.g. GSE307690/CANUCK).
    """
    opener = gzip.open if gz_path.suffix == ".gz" else open
    sample_ids: list[str] = []
    char_rows: dict[str, list[str]] = {}
    table_lines: list[str] = []
    in_table = False

    with opener(gz_path, "rt", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [s.strip('"') for s in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [s.strip('"') for s in line.split("\t")[1:]]
                for v in vals:
                    if ":" in v:
                        key, val = v.split(":", 1)
                        char_rows.setdefault(key.strip().lower(), []).append(val.strip())
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                table_lines.append(line)

    meta = pd.DataFrame(index=sample_ids)
    for key, vals in char_rows.items():
        if len(vals) == len(sample_ids):
            meta[key] = vals

    return sample_ids, meta, table_lines


def _parse_series_matrix(gz_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a GEO `*_series_matrix.txt.gz` file with an embedded expression table.

    Returns
    -------
    expr  : DataFrame, genes/probes (rows) x GSM sample IDs (cols)
    meta  : DataFrame, GSM sample IDs (rows) x characteristic fields (cols),
            parsed from `!Sample_characteristics_ch1` lines of the form
            "key: value".
    """
    sample_ids, meta, table_lines = _parse_sample_metadata(gz_path)
    if not table_lines:
        raise ValueError(f"No expression table found in {gz_path}")

    from io import StringIO
    expr = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t", index_col=0)
    expr.columns = [c.strip('"') for c in expr.columns]

    return expr, meta


_SMOKE_STATUS_PATTERNS = {
    "cigarette": re.compile(r"\bcurrent\b|\bsmoker\b|\bever\b|\bcigarette\b", re.I),
    "unexposed": re.compile(r"\bnever\b|\bnon-?smoker\b|\bcontrol\b", re.I),
}


def _infer_smoke_column(meta: pd.DataFrame, default: str) -> pd.Series:
    """Best-effort per-sample smoke type from GEO characteristic fields."""
    status_col = next(
        (c for c in meta.columns if "smok" in c or "status" in c), None
    )
    if status_col is None:
        return pd.Series(default, index=meta.index)

    def _classify(v: str) -> str:
        if _SMOKE_STATUS_PATTERNS["unexposed"].search(str(v)):
            return "unexposed"
        if _SMOKE_STATUS_PATTERNS["cigarette"].search(str(v)):
            return "cigarette"
        return default

    return meta[status_col].map(_classify)


def _load_probe_to_symbol_map(annot_path: Path) -> dict[str, str]:
    """
    Parses a GEO platform annotation file (`GPL*.annot.gz`) into a
    {probe_id: gene_symbol} dict. Format is a `!platform_table_begin` /
    `!platform_table_end`-delimited TSV table with "ID" and "Gene symbol"
    columns — used for platforms like GSE123352's Illumina HumanHT-12
    (ILMN_ probe IDs) that BioMart doesn't expose as a queryable attribute,
    so harmonize_gene_ids() (transforms.py) can't map them; this is the
    documented alternative referenced in that function's docstring.
    Probes with no annotated symbol are dropped (empty string maps to
    nothing meaningful, and it's a single value that would collapse many
    genes into one row).
    """
    lines = []
    in_table = False
    opener = gzip.open if annot_path.suffix == ".gz" else open
    with opener(annot_path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                lines.append(line.rstrip("\n"))

    from io import StringIO
    table = pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str)
    table = table.dropna(subset=["ID", "Gene symbol"])
    table = table[table["Gene symbol"].str.strip() != ""]
    return dict(zip(table["ID"], table["Gene symbol"]))


def convert_microarray(accession: str, gz_path: Path, default_smoke_type: str,
                        platform_annot_path: Optional[Path] = None) -> Path:
    """
    GEO series matrix → (genes x samples CSV for load_microarray) +
    (sibling `_samples_meta.csv` with per-sample smoke_type, used by
    load_microarray to override the blanket default where GEO metadata
    lets us tell smokers from never-smokers within one series).

    platform_annot_path: optional GPL*.annot.gz to map probe IDs (e.g.
    GSE123352's ILMN_... Illumina probes) to gene symbols before writing
    the CSV. Probes with no mapped symbol are dropped; probes that share a
    symbol (multiple probes per gene is normal on array platforms) are
    collapsed via mean expression, matching how harmonize_gene_ids()
    (transforms.py) already resolves BioMart's many-probes-to-one-gene case.
    """
    out_dir = _mkout()
    expr, meta = _parse_series_matrix(gz_path)

    if platform_annot_path is not None:
        n_probes = len(expr)
        probe_to_symbol = _load_probe_to_symbol_map(platform_annot_path)
        expr = expr.loc[expr.index.isin(probe_to_symbol)]
        expr.index = expr.index.map(probe_to_symbol)
        expr = expr.groupby(expr.index).mean()
        print(f"[convert] {accession}  platform annotation  {n_probes} probes → "
              f"{len(expr)} gene symbols ({Path(platform_annot_path).name})")

    csv_path  = out_dir / f"{accession}.csv"
    meta_path = out_dir / f"{accession}_samples_meta.csv"

    expr.to_csv(csv_path)

    smoke = _infer_smoke_column(meta, default_smoke_type)
    pd.DataFrame({"sample_id": expr.columns, "smoke_type": smoke.reindex(expr.columns).values}
                 ).to_csv(meta_path, index=False)

    n_over = (smoke.reindex(expr.columns) != default_smoke_type).sum()
    print(f"[convert] {accession}  {expr.shape[1]} samples x {expr.shape[0]} genes "
          f"→ {csv_path.name}  ({n_over} samples relabelled from GEO metadata)")
    return csv_path


def convert_canuck(accession: str, gz_path: Path, processed_data_path: Path) -> Path:
    """
    GSE307690 (CANUCK study — Halayko/Tam et al., real human airway epithelial
    brushings from 139 cannabis smokers + 57 never-smokers) → genes x samples
    CSV + `_samples_meta.csv` for load_microarray.

    The series matrix's embedded expression table is empty for this
    accession — GEO stores the real matrix as a separate supplementary file
    (`processed_data_path`, space-delimited, columns "sample1".."sampleN").
    Sample order in that file matches the series matrix's GSM column order
    (both follow GEO submission order), so samples are joined positionally,
    not by name.

    Per-sample smoke_type is derived from three independent characteristics
    fields — "cannabis group", "cigarette", "vape" — so a subject using both
    cannabis and tobacco/vape is correctly labelled dual_use rather than
    just cannabis.
    """
    out_dir = _mkout()
    sample_ids, meta, _ = _parse_sample_metadata(gz_path)

    expr = pd.read_csv(processed_data_path, sep=r"\s+", index_col=0)
    expr.index = expr.index.str.split(".").str[0]   # strip Ensembl version/dedup suffix
    if expr.shape[1] != len(sample_ids):
        raise ValueError(
            f"{accession}: processed data has {expr.shape[1]} samples, "
            f"series matrix lists {len(sample_ids)} — cannot align positionally"
        )
    expr.columns = sample_ids  # positional join: column order matches GSM order

    def _classify(row) -> str:
        cannabis = str(row.get("cannabis group", "")).lower().startswith("cannabis")
        cig      = str(row.get("cigarette", "")).lower() in ("current", "former")
        vape     = str(row.get("vape", "")).lower() == "yes"
        if cannabis and (cig or vape):
            return "dual_use"
        if cannabis:
            return "cannabis"
        if cig:
            return "cigarette"
        if vape:
            return "vape"
        return "unexposed"

    smoke = meta.apply(_classify, axis=1) if not meta.empty else pd.Series("unexposed", index=sample_ids)

    csv_path  = out_dir / f"{accession}.csv"
    meta_path = out_dir / f"{accession}_samples_meta.csv"
    expr.to_csv(csv_path)
    pd.DataFrame({"sample_id": expr.columns, "smoke_type": smoke.reindex(expr.columns).values}
                 ).to_csv(meta_path, index=False)

    counts = smoke.value_counts().to_dict()
    print(f"[convert] {accession}  {expr.shape[1]} samples x {expr.shape[0]} genes "
          f"→ {csv_path.name}  {counts}")
    return csv_path


# ─── scRNA (10x-style) conversion ─────────────────────────────────────────────

def _read_mtx_streaming(path: Path, chunk_rows: int = 20_000_000):
    """
    Memory-disciplined MatrixMarket reader for very large sparse matrices.

    scipy.io.mmread's pure-Python line parser ballooned to >17GB RSS on a
    real 692M-nonzero file (GSE136831, ~8GB of actual triplet data),
    exhausting swap on a 16GB machine. This reads the coordinate list in
    fixed-size chunks via pandas' C parser, straight into arrays
    pre-allocated from the MatrixMarket header's known nnz — so peak memory
    stays close to the matrix's real footprint instead of several multiples
    of it.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        line = f.readline()
        if not line.startswith("%%MatrixMarket"):
            raise ValueError(f"{path} is not a MatrixMarket file")
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        nrows, ncols, nnz = (int(x) for x in line.split())

        row = np.empty(nnz, dtype=np.int32)
        col = np.empty(nnz, dtype=np.int32)
        val = np.empty(nnz, dtype=np.float32)
        filled = 0
        for chunk in pd.read_csv(
            f, sep=r"\s+", header=None, chunksize=chunk_rows,
            names=["row", "col", "val"],
            dtype={"row": np.int32, "col": np.int32, "val": np.float32},
        ):
            n = len(chunk)
            row[filled:filled + n] = chunk["row"].values
            col[filled:filled + n] = chunk["col"].values
            val[filled:filled + n] = chunk["val"].values
            filled += n

    row -= 1  # MatrixMarket indices are 1-based
    col -= 1
    import scipy.sparse as sp
    return sp.coo_matrix((val, (row, col)), shape=(nrows, ncols)).tocsr()


def convert_scrna_10x(accession: str, src_dir: Path, donor_map: Optional[dict] = None,
                       cell_metadata: Optional[pd.DataFrame] = None) -> Path:
    """
    10x-style raw counts (matrix.mtx[.gz] + barcodes.tsv[.gz] + features/genes.tsv[.gz],
    or a single combined `*_RawCounts_Sparse.mtx.gz` with sibling barcode/feature
    files) → h5ad for load_scrna / load_mouse_scrna.

    donor_map: optional {barcode_prefix: donor_id} to populate obs['donor_id']
    when the raw files don't already encode it (GEO supplementary files rarely do —
    check the accession's associated paper/metadata for the barcode→donor mapping).
    Only used as a fallback when cell_metadata isn't supplied — prefix-splitting
    is a heuristic and can misparse barcodes whose subject ID doesn't literally
    prefix the barcode (see GSE136831 below).

    cell_metadata: optional DataFrame indexed by *full* barcode (exact match
    against the source barcode file, not a prefix) with a 'donor_id' column
    and any other per-cell columns to carry through to obs. Takes priority
    over donor_map — an exact per-cell join is more reliable than a regex
    prefix guess. Needed for GSE136831: its Subject_Identity values don't
    always literally prefix the barcode (e.g. subject "1372C" has barcodes
    prefixed "137C-a_..." — a real quirk in the deposited metadata, not a
    parsing bug), so only a full-barcode join against the published
    per-cell metadata table gives 100% correct donor assignment.
    """
    import scanpy as sc

    out_dir = _mkout()
    out_path = out_dir / f"{accession}.h5ad"

    triples_present = any(src_dir.glob("matrix.mtx*"))
    if triples_present:
        adata = sc.read_10x_mtx(src_dir, var_names="gene_symbols", cache=False)
    else:
        mtx_files = list(src_dir.glob("*RawCounts*.mtx.gz")) + list(src_dir.glob("*.mtx.gz"))
        if not mtx_files:
            raise FileNotFoundError(
                f"No .mtx.gz found in {src_dir} for {accession}. "
                "Download the supplementary files first: "
                "python3 src/data/downloaders.py --accession " + accession
            )
        import scipy.sparse as sp

        mtx = _read_mtx_streaming(mtx_files[0])
        # Case-insensitive substring match: GEO supplementary files use all
        # sorts of casing/naming (barcodes.tsv, cellBarcodes.txt, GeneIDs.txt,
        # features.tsv...) that a fixed-case glob silently misses.
        def _find_sibling(*keywords: str) -> list[Path]:
            return [p for p in src_dir.iterdir()
                    if p != mtx_files[0] and any(k in p.name.lower() for k in keywords)]

        barcode_files = _find_sibling("barcode")
        feature_files = _find_sibling("feature", "gene")

        barcodes = (_read_id_list(barcode_files[0], expected_len=mtx.shape[1]) if barcode_files else None)
        genes    = (_read_id_list(feature_files[0], expected_len=mtx.shape[0], prefer_symbol_col=True)
                    if feature_files else None)

        # 10x convention is genes x cells (rows x cols); AnnData needs the
        # opposite (obs=cells x var=genes). Orient by matching mtx dims
        # against the known barcode/gene counts rather than assuming —
        # GEO supplementary matrices are not consistently oriented.
        if barcodes is not None and mtx.shape[1] == len(barcodes) and mtx.shape[0] != len(barcodes):
            mtx = mtx.T
        mtx = mtx.tocsr()

        if barcodes is None:
            barcodes = [f"cell_{i}" for i in range(mtx.shape[0])]
        if genes is None:
            genes = [f"gene_{i}" for i in range(mtx.shape[1])]

        import anndata as ad
        adata = ad.AnnData(X=sp.csr_matrix(mtx),
                            obs=pd.DataFrame(index=barcodes),
                            var=pd.DataFrame(index=genes))

    if cell_metadata is not None:
        joined = cell_metadata.reindex(adata.obs_names)
        n_missing = joined["donor_id"].isna().sum()
        if n_missing:
            print(f"[convert] {accession}  WARNING: {n_missing:,}/{adata.n_obs:,} "
                  "barcodes had no match in cell_metadata — labelled 'unknown'")
        for col in joined.columns:
            adata.obs[col] = joined[col].fillna("unknown").values
    elif donor_map:
        prefixes = adata.obs_names.str.extract(r"^([^-_]+)")[0]
        adata.obs["donor_id"] = prefixes.map(donor_map).fillna("unknown").values
    elif "donor_id" not in adata.obs.columns:
        adata.obs["donor_id"] = "unknown"
        print(f"[convert] {accession}  WARNING: no donor_map supplied and no "
              f"donor_id in source — all {adata.n_obs:,} cells will collapse "
              "into a single MIL bag under subject 'unknown'. Pass donor_map= "
              "with the accession's real barcode→donor mapping before training.")

    adata.write_h5ad(out_path)
    print(f"[convert] {accession}  {adata.n_obs:,} cells x {adata.n_vars:,} genes → {out_path.name}")
    return out_path


_GSE288003_CONDITION_SMOKE_TYPE = {"con": "unexposed", "control": "unexposed",
                                    "e-cigs": "vape", "ecig": "vape", "ecigs": "vape"}


def convert_gse288003(accession: str, src_dir: Path) -> Optional[Path]:
    """
    GSE288003 (mouse lung, e-cig aerosol) ships its real count matrix only
    inside `*_RAW.tar` — one standard 10x triple per GSM sample, e.g.
    GSM8757329_Con_{barcodes,genes,matrix}.{tsv,mtx}.gz (unexposed control)
    and GSM8757330_E-cigs_{...} (e-cig exposed). downloaders.py's
    _extract_tar() now extracts that tar into src_dir before this runs.

    Both samples are converted and concatenated here, each cell tagged with
    the REAL condition from its own filename ("Con" -> unexposed, "E-cigs"
    -> vape) via obs['smoke_type_name'] — not the accession's blanket
    "vape" default from GEO_DATASETS, which would mislabel the Con
    (unexposed) mouse's cells. loaders.py's _attach_standard_obs() already
    knows to keep a pre-set smoke_type_name instead of overwriting it.

    convert_scrna_10x's generic combined-matrix path can't be reused as-is:
    its `*.mtx.gz` glob would match both GSM samples' mtx files and
    silently pick just one (losing the other condition entirely), and its
    sibling barcode/feature lookup has the same ambiguity.
    """
    matrix_files = sorted(src_dir.glob("GSM*_matrix.mtx.gz"))
    if not matrix_files:
        return _missing(accession, src_dir)

    import anndata as ad
    import scipy.sparse as sp

    per_sample = []
    for mtx_path in matrix_files:
        prefix = mtx_path.name[:-len("_matrix.mtx.gz")]   # e.g. "GSM8757329_Con"
        gsm_id, condition = prefix.split("_", 1)
        barcode_path = src_dir / f"{prefix}_barcodes.tsv.gz"
        gene_path    = src_dir / f"{prefix}_genes.tsv.gz"
        if not (barcode_path.exists() and gene_path.exists()):
            print(f"[convert] {accession}  WARNING: missing barcodes/genes for {prefix} — skipped")
            continue

        mtx = _read_mtx_streaming(mtx_path)   # genes x cells, 10x convention
        barcodes = _read_id_list(barcode_path)
        genes    = _read_id_list(gene_path, prefer_symbol_col=True)
        if mtx.shape[1] == len(barcodes) and mtx.shape[0] != len(barcodes):
            mtx = mtx.T
        mtx = mtx.tocsr()

        smoke_type_name = _GSE288003_CONDITION_SMOKE_TYPE.get(condition.lower())
        if smoke_type_name is None:
            print(f"[convert] {accession}  WARNING: unrecognised condition '{condition}' "
                  f"in {prefix} — leaving smoke_type unset (falls back to accession default)")

        obs = pd.DataFrame({
            "donor_id": gsm_id,
            **({"smoke_type_name": smoke_type_name} if smoke_type_name else {}),
        }, index=[f"{prefix}_{bc}" for bc in barcodes])
        sample_adata = ad.AnnData(X=sp.csr_matrix(mtx), obs=obs, var=pd.DataFrame(index=genes))
        # Mouse gene symbol column has real duplicates (unannotated genes
        # share "", multiple Ensembl IDs share one symbol) — ad.concat
        # requires a unique var index, same reason harmonize_gene_ids'
        # BioMart-mapped symbols would need this too.
        sample_adata.var_names_make_unique()
        per_sample.append(sample_adata)
        print(f"[convert] {accession}  {prefix}  {mtx.shape[0]:,} cells x {mtx.shape[1]:,} genes "
              f"→ smoke_type={smoke_type_name}")

    if not per_sample:
        return _missing(accession, src_dir)

    adata = ad.concat(per_sample, join="outer", fill_value=0) if len(per_sample) > 1 else per_sample[0]
    out_path = _mkout() / f"{accession}.h5ad"
    adata.write_h5ad(out_path)
    print(f"[convert] {accession}  {adata.n_obs:,} cells x {adata.n_vars:,} genes "
          f"({len(per_sample)} samples) → {out_path.name}")
    return out_path


def _load_gse136831_cell_metadata(src_dir: Path) -> Optional[pd.DataFrame]:
    """
    GSE136831's `*_AllCells.Samples.CellType.MetadataTable.txt.gz` carries
    real per-cell Subject_Identity, Disease_Identity (COPD/IPF/Control) and
    CellType_Category, keyed by the exact same barcode strings used in
    `*_AllCells.cellBarcodes.txt.gz` / the mtx column order. Returns a
    DataFrame indexed by that barcode with donor_id/disease_identity/
    cell_type columns, or None if the file isn't present yet.

    Disease_Identity is COPD/IPF/Control, not a direct smoking-status field
    — this dataset is the Vanderbilt/Habermann interstitial lung disease
    atlas, not a dedicated smoking cohort. smoke_type still defaults to
    "cigarette" for these samples (COPD is strongly smoking-associated,
    and this pipeline has no better per-subject label for this accession),
    the same documented-approximation pattern already used for TCGA in
    convert_tcga above — not a claim that GSE136831 records smoking status.
    """
    matches = list(src_dir.glob("*Samples.CellType.MetadataTable*"))
    if not matches:
        return None
    meta = pd.read_csv(matches[0], sep="\t", quotechar='"')
    meta = meta.set_index("CellBarcode_Identity")
    return pd.DataFrame({
        "donor_id":         meta["Subject_Identity"],
        "disease_identity": meta["Disease_Identity"],
        "cell_type":        meta["CellType_Category"],
    })


def _read_id_list(path: Path, expected_len: Optional[int] = None,
                   prefer_symbol_col: bool = False) -> list[str]:
    """
    prefer_symbol_col: real 10x features.tsv convention is
    [ensembl_id, gene_symbol, feature_type] — sc.read_10x_mtx's
    var_names="gene_symbols" already prefers column 2 on the standard
    triples path, so the manual combined-matrix path (used for GEO
    supplementary `*_RawCounts_Sparse.mtx.gz` files without an accompanying
    matrix.mtx triple, e.g. GSE136831) does the same here for consistency:
    when the file has >=2 tab-separated columns, use column 2 (the symbol)
    instead of column 1 (the Ensembl ID) — this also means
    harmonize_gene_ids() (transforms.py) doesn't need a live BioMart
    round-trip for genes that already ship a symbol.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    col = 0
    if prefer_symbol_col:
        with opener(path, "rt") as f:
            first = f.readline()
        if len(first.rstrip("\n").split("\t")) >= 2:
            col = 1
    with opener(path, "rt") as f:
        ids = [line.rstrip("\n").split("\t")[col].strip().strip('"') for line in f if line.strip()]
    # Some GEO supplementary files ship a header row (e.g. GSE136831's
    # GeneIDs.txt has `"Ensembl_GeneID"	"HGNC_EnsemblAlt_GeneID"` on line 1);
    # detect it by comparing against the matrix's known dimension rather than
    # assuming every file either always or never has one.
    if expected_len is not None and len(ids) == expected_len + 1:
        print(f"[convert] {path.name}  dropping header row ({len(ids)} -> {expected_len})")
        ids = ids[1:]
    return ids


# ─── TCGA conversion ──────────────────────────────────────────────────────────

def convert_tcga(project: str, src_dir: Path) -> Optional[Path]:
    """
    TCGA HTSeq/STAR gene-count files (one per case, downloaded via gdc-client
    into `src_dir/<file_id>/<filename>`) → genes x samples CSV for
    load_microarray, + sibling `_samples_meta.csv` carrying the malignancy
    and subject_id fields load_microarray needs (extended alongside
    smoke_type — see loaders.py).

    Malignancy label: 1.0 for "Primary Tumor" samples, 0.0 for solid tissue
    normal (NAT), read from downloaders.py's file_meta.csv (case_id +
    sample_type — the GDC fields the manifest itself doesn't carry).

    Smoke type: TCGA-LUAD/LUSC don't carry per-patient smoking history in
    this pipeline, and >85% of these cohorts are smokers (per TCGA clinical
    characteristics) — defaulted to "cigarette" as a documented approximation,
    consistent with the cigar/dual-use label-transfer caveats already in
    ARCHITECTURE.md section 3A.
    """
    out_dir  = _mkout()
    meta_csv = src_dir / "file_meta.csv"
    if not meta_csv.exists():
        print(f"[convert] {project}  no file_meta.csv in {src_dir} — "
              f"re-download with: python3 src/data/downloaders.py --tcga")
        return None

    file_meta = pd.read_csv(meta_csv, dtype=str).fillna("")
    columns, malignancy, subject_id = {}, {}, {}

    for _, row in file_meta.iterrows():
        matches = list(src_dir.glob(f"{row['file_id']}/*"))
        matches = [p for p in matches if p.is_file() and p.name != "annotations.txt"]
        if not matches:
            continue
        counts = pd.read_csv(matches[0], sep="\t", header=None,
                              names=["gene_id", "count"], dtype={"gene_id": str})
        counts = counts[~counts["gene_id"].str.startswith("__")]
        counts["gene_id"] = counts["gene_id"].str.split(".").str[0]  # drop Ensembl version
        sample_id = row["file_id"]
        columns[sample_id]   = counts.set_index("gene_id")["count"]
        malignancy[sample_id]= 1.0 if "Tumor" in row["sample_type"] else 0.0
        subject_id[sample_id]= row["case_id"] or sample_id

    if not columns:
        print(f"[convert] {project}  no downloaded count files found under {src_dir} — "
              f"run: python3 src/data/downloaders.py --tcga --token /path/to/token.txt")
        return None

    expr = pd.DataFrame(columns).fillna(0)
    csv_path  = out_dir / f"{project}.csv"
    meta_path = out_dir / f"{project}_samples_meta.csv"
    expr.to_csv(csv_path)
    pd.DataFrame({
        "sample_id":  expr.columns,
        "smoke_type": "cigarette",
        "malignancy": [malignancy[s] for s in expr.columns],
        "subject_id": [subject_id[s] for s in expr.columns],
    }).to_csv(meta_path, index=False)

    n_tumor = sum(v == 1.0 for v in malignancy.values())
    print(f"[convert] {project}  {expr.shape[1]} samples x {expr.shape[0]} genes → {csv_path.name}"
          f"  ({n_tumor} tumor / {expr.shape[1] - n_tumor} normal)")

    outcomes_path = out_dir / f"{project}_outcomes.csv"
    outcomes = (
        pd.DataFrame({"subject_id": list(subject_id.values()), "cancer_label": list(malignancy.values())})
        .groupby("subject_id", as_index=False)["cancer_label"].max()
    )
    outcomes["cancer_label"] = outcomes["cancer_label"].astype(int)
    outcomes.to_csv(outcomes_path, index=False)
    print(f"[convert] {project}  {len(outcomes):,} subjects → {outcomes_path.name}")
    return csv_path


# ─── NLST outcomes ─────────────────────────────────────────────────────────────

def convert_nlst_outcomes(prsn_csv: Path) -> Path:
    """
    NLST prsn.csv (pid, candx, ...) → clean subject_id/cancer_label CSV for
    assemble_subject_bags()'s `cancer_outcomes` argument.
    """
    out_path = _mkout() / "nlst_outcomes.csv"
    df = pd.read_csv(prsn_csv, low_memory=False)
    out = pd.DataFrame({
        "subject_id":   df["pid"].astype(str),
        "cancer_label": (df.get("candx", 0) == 1).astype(int),
    })
    out.to_csv(out_path, index=False)
    print(f"[convert] NLST outcomes  {len(out):,} subjects → {out_path.name}")
    return out_path


# ─── Dispatch ─────────────────────────────────────────────────────────────────

def convert_accession(accession: str) -> Optional[Path]:
    from data.downloaders import GEO_DATASETS
    if accession not in GEO_DATASETS:
        print(f"[convert] unknown accession {accession}")
        return None

    subdir, desc, smoke_type, expected_file = GEO_DATASETS[accession]
    src = RAW / subdir

    if accession == "GSE307690":
        gz = src / expected_file
        processed = src / f"{accession}_processed_data.txt.gz"
        if not (gz.exists() and processed.exists()):
            return _missing(accession, processed)
        return convert_canuck(accession, gz, processed)

    if accession == "GSE136831":
        if not src.exists():
            return _missing(accession, src)
        cell_metadata = _load_gse136831_cell_metadata(src)
        if cell_metadata is None:
            print(f"[convert] {accession}  WARNING: no *Samples.CellType.MetadataTable* "
                  f"file found in {src} — falling back to prefix-guessed donor_id. "
                  f"Re-run downloaders.py --accession {accession} to fetch it.")
        return convert_scrna_10x(accession, src, cell_metadata=cell_metadata)

    if accession == "GSE288003":
        return convert_gse288003(accession, src) if src.exists() else _missing(accession, src)

    gz = src / expected_file
    if not gz.exists():
        return _missing(accession, gz)

    from data.downloaders import GEO_PLATFORM_ANNOTATIONS
    gpl = GEO_PLATFORM_ANNOTATIONS.get(accession)
    platform_annot_path = None
    if gpl:
        candidate = src / f"{gpl}.annot.gz"
        if candidate.exists():
            platform_annot_path = candidate
        else:
            print(f"[convert] {accession}  WARNING: no {gpl}.annot.gz in {src} — "
                  f"probe IDs will stay unmapped. Re-run: "
                  f"python3 src/data/downloaders.py --accession {accession}")

    return convert_microarray(accession, gz, smoke_type, platform_annot_path=platform_annot_path)


def _missing(accession: str, path: Path) -> None:
    print(f"[convert] {accession}  source not found at {path} — download it first:\n"
          f"  python3 src/data/downloaders.py --accession {accession}")
    return None


def convert_all() -> None:
    from data.downloaders import GEO_DATASETS, TCGA_DATASETS
    for accession in GEO_DATASETS:
        convert_accession(accession)

    for project, cfg in TCGA_DATASETS.items():
        src = RAW / cfg["subdir"]
        if src.exists():
            convert_tcga(project, src)
        else:
            print(f"[convert] {project}  source not found at {src} — "
                  "run: python3 src/data/downloaders.py --tcga --token /path/to/token.txt")

    nlst_prsn = RAW / "subjects" / "NLST" / "prsn.csv"
    if nlst_prsn.exists():
        convert_nlst_outcomes(nlst_prsn)
    else:
        print(f"[convert] NLST outcomes  source not found at {nlst_prsn} — "
              "see: python3 src/data/downloaders.py --nlst-instructions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw downloads to clean loader formats")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--accession", type=str)
    parser.add_argument("--tcga", type=str, help="Convert one TCGA project, e.g. TCGA-LUAD")
    parser.add_argument("--nlst", action="store_true", help="Convert NLST prsn.csv outcomes only")
    args = parser.parse_args()

    if args.all:
        convert_all()
    elif args.accession:
        convert_accession(args.accession)
    elif args.tcga:
        from data.downloaders import TCGA_DATASETS
        if args.tcga not in TCGA_DATASETS:
            print(f"[convert] unknown TCGA project {args.tcga}")
        else:
            convert_tcga(args.tcga, RAW / TCGA_DATASETS[args.tcga]["subdir"])
    elif args.nlst:
        nlst_prsn = RAW / "subjects" / "NLST" / "prsn.csv"
        if nlst_prsn.exists():
            convert_nlst_outcomes(nlst_prsn)
        else:
            print(f"missing {nlst_prsn}")
    else:
        parser.print_help()
        print("\nQuick start:\n  python3 src/data/converters.py --all")
