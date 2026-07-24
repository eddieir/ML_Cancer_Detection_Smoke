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
    sample_ids, meta, table_lines, _titles = _parse_sample_metadata_with_titles(gz_path)
    return sample_ids, meta, table_lines


def _parse_sample_metadata_with_titles(gz_path: Path) -> tuple[list[str], pd.DataFrame, list[str], list[str]]:
    """Same as `_parse_sample_metadata`, plus the raw `!Sample_title` values
    (in the same GSM column order) — used by `_infer_subject_id_column` to
    recover a genuine donor identifier for cohorts whose series matrix
    encodes one in the sample title (e.g. GSE123352's
    "non_involved_lung_tissue_patient_N")."""
    opener = gzip.open if gz_path.suffix == ".gz" else open
    sample_ids: list[str] = []
    titles: list[str] = []
    char_rows: dict[str, list[str]] = {}
    table_lines: list[str] = []
    in_table = False

    with opener(gz_path, "rt", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [s.strip('"') for s in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [s.strip('"') for s in line.split("\t")[1:]]
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

    if len(titles) != len(sample_ids):
        titles = []

    return sample_ids, meta, table_lines, titles


_PATIENT_TITLE_RE = re.compile(r"patient[_\s]?(\d+)", re.I)


def _infer_subject_id_column(
    accession: str, sample_ids: list[str], titles: list[str],
) -> "tuple[pd.Series, pd.Series, str]":
    """Cohort-specific, versioned subject-identity policy for series-matrix
    accessions that encode a donor number in `!Sample_title` (documented
    today for GSE123352: titles of the form
    "non_involved_lung_tissue_patient_<N> <GEO array position>", one
    distinct N per GSM with no repeats — i.e. one bulk sample per donor;
    see docs/DATA_CARD.md).

    Returns (subject_id, subject_id_verified, policy_note). subject_id is
    only trusted (subject_id_verified=True for every row) when every
    sample's title matches the documented "patient_<N>" pattern AND the
    resulting donor numbers are unique across the whole series — i.e. a
    verified one-sample-per-donor mapping. If titles are absent, any
    sample's title fails to match, or two samples resolve to the same
    donor number, this fails closed: subject_id falls back to sample_id
    (GSM accession) but subject_id_verified=False for every row, so a
    caller that requires verified subject independence (see
    data/bulk_pipeline.py) must refuse to make a subject-level claim
    rather than silently trust sample_id as if it were a genuine donor id.
    """
    if not titles:
        return (
            pd.Series(sample_ids, index=sample_ids),
            pd.Series(False, index=sample_ids),
            "no !Sample_title field was present to parse a donor identifier from — "
            "subject independence could not be verified; sample_id was used as a "
            "fallback and must not be treated as a verified subject id.",
        )

    matches = [_PATIENT_TITLE_RE.search(t) for t in titles]
    if not all(matches):
        return (
            pd.Series(sample_ids, index=sample_ids),
            pd.Series(False, index=sample_ids),
            "one or more !Sample_title values did not match the documented "
            "'patient_<N>' pattern — subject independence could not be verified; "
            "sample_id was used as a fallback and must not be treated as a "
            "verified subject id.",
        )

    donor_nums = [m.group(1) for m in matches]
    if len(set(donor_nums)) != len(donor_nums):
        return (
            pd.Series(sample_ids, index=sample_ids),
            pd.Series(False, index=sample_ids),
            "two or more samples resolved to the same donor number parsed from "
            "!Sample_title — this contradicts the documented one-sample-per-donor "
            "assumption for this accession; subject independence could not be "
            "verified and sample_id was used as a fallback only.",
        )

    subject_ids = pd.Series(
        [f"{accession}_patient_{n}" for n in donor_nums], index=sample_ids,
    )
    return (
        subject_ids,
        pd.Series(True, index=sample_ids),
        "subject_id parsed and verified from !Sample_title's documented "
        "'patient_<N>' pattern: every sample resolved to a distinct donor number, "
        "confirming one bulk sample per donor for this accession (see "
        "docs/DATA_CARD.md's GSE123352 subject-identity policy).",
    )


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
    expr, meta, _titles = _parse_series_matrix_with_titles(gz_path)
    return expr, meta


def _parse_series_matrix_with_titles(gz_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Same as `_parse_series_matrix`, plus the raw `!Sample_title` values in
    GSM column order (see `_parse_sample_metadata_with_titles`)."""
    sample_ids, meta, table_lines, titles = _parse_sample_metadata_with_titles(gz_path)
    if not table_lines:
        raise ValueError(f"No expression table found in {gz_path}")

    from io import StringIO
    expr = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t", index_col=0)
    expr.columns = [c.strip('"') for c in expr.columns]

    return expr, meta, titles


_SMOKE_STATUS_PATTERNS = {
    "cigarette": re.compile(r"\bcurrent\b|\bsmoker\b|\bever\b|\bcigarette\b", re.I),
    "unexposed": re.compile(r"\bnever\b|\bnon-?smoker\b|\bcontrol\b", re.I),
}


def _infer_smoke_column(meta: pd.DataFrame, default: str) -> "tuple[pd.Series, pd.Series]":
    """
    Per-sample smoke type from GEO characteristic fields, where available.

    Returns (smoke_type_name, smoke_type_known) — a sample's smoke type is
    "known" ONLY when its own characteristics text actually matched one of
    the documented patterns (current/former/ever smoker -> cigarette;
    never/non-smoker/control -> unexposed). This repository's `default`
    parameter (the accession-level smoke_type declared in
    configs/default.yaml's microarray_sources) is a fallback LABEL for
    display purposes only now — it is never used to fill a genuinely
    missing or unmatched per-sample value, since accessions like GSE994
    are documented to contain a mix of smokers and never-smokers (see
    README.md), not a single uniform condition. A sample with no
    "smok"/"status"-named characteristics column at all, or whose value
    doesn't match either documented pattern, gets smoke_type="unknown" and
    smoke_type_known=False instead of silently inheriting the accession
    blanket.
    """
    status_col = next(
        (c for c in meta.columns if "smok" in c or "status" in c), None
    )
    if status_col is None:
        return (pd.Series("unknown", index=meta.index),
                pd.Series(False, index=meta.index))

    def _classify(v: str) -> str:
        if _SMOKE_STATUS_PATTERNS["unexposed"].search(str(v)):
            return "unexposed"
        if _SMOKE_STATUS_PATTERNS["cigarette"].search(str(v)):
            return "cigarette"
        return "unknown"

    smoke_type = meta[status_col].map(_classify)
    known = smoke_type != "unknown"
    return smoke_type, known


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
    expr, meta, titles = _parse_series_matrix_with_titles(gz_path)

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

    smoke, known = _infer_smoke_column(meta, default_smoke_type)
    smoke = smoke.reindex(expr.columns)
    known = known.reindex(expr.columns).fillna(False)
    limitation = pd.Series(
        np.where(
            known,
            "Per-sample smoking status parsed from this accession's own GEO characteristics field.",
            "No 'smok'/'status'-named GEO characteristics field matched a documented current/former/"
            "ever-smoker or never/non-smoker/control pattern for this sample — left unknown rather "
            "than defaulted to the accession-level label.",
        ),
        index=expr.columns,
    )

    subject_id, subject_id_verified, subject_id_policy_note = _infer_subject_id_column(
        accession, list(expr.columns), titles,
    )
    subject_id = subject_id.reindex(expr.columns)
    subject_id_verified = subject_id_verified.reindex(expr.columns).fillna(False)

    pd.DataFrame({
        "sample_id": expr.columns,
        "smoke_type": smoke.values,
        "smoke_type_known": known.values,
        "smoke_type_source": f"GEO {accession} series matrix characteristics",
        "smoke_type_method": "regex_pattern_match",
        "smoke_type_limitation": limitation.values,
        "subject_id": subject_id.values,
        "subject_id_verified": subject_id_verified.values,
        "subject_id_policy_note": subject_id_policy_note,
    }).to_csv(meta_path, index=False)

    n_unknown = int((~known).sum())
    print(f"[convert] {accession}  {expr.shape[1]} samples x {expr.shape[0]} genes "
          f"→ {csv_path.name}  ({int(known.sum())} known from GEO metadata, "
          f"{n_unknown} unknown — never defaulted to '{default_smoke_type}')")
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

    # meta.empty means the series matrix carried NO characteristics rows at
    # all (a total metadata-parsing failure, not "every sample is
    # documented never-smoker/control") — that must stay unknown, not
    # silently become a blanket "unexposed" for the whole accession. When
    # meta IS populated, _classify's three-field logic runs per real,
    # documented sample characteristics (this accession's own published
    # design accounts for every sample: 139 cannabis smokers + 57
    # never-smokers — see this function's docstring), so "unexposed" from
    # _classify itself is a genuine parsed negative, not a missing-value
    # default.
    if meta.empty:
        smoke = pd.Series("unknown", index=sample_ids)
        known = pd.Series(False, index=sample_ids)
    else:
        smoke = meta.apply(_classify, axis=1)
        known = pd.Series(True, index=meta.index)

    csv_path  = out_dir / f"{accession}.csv"
    meta_path = out_dir / f"{accession}_samples_meta.csv"
    expr.to_csv(csv_path)
    smoke = smoke.reindex(expr.columns)
    known = known.reindex(expr.columns).fillna(False)
    pd.DataFrame({
        "sample_id": expr.columns,
        "smoke_type": smoke.values,
        "smoke_type_known": known.values,
        "smoke_type_source": f"GEO {accession} series matrix characteristics "
                              "(cannabis group / cigarette / vape fields)",
        "smoke_type_method": "documented_three_field_classification",
        "smoke_type_limitation": np.where(
            known.values,
            "Derived from this accession's own published cannabis group/cigarette/vape "
            "characteristics fields.",
            "Series matrix carried no characteristics rows for this accession — "
            "smoking status could not be determined and was not defaulted.",
        ),
    }).to_csv(meta_path, index=False)

    counts = smoke.value_counts().to_dict()
    print(f"[convert] {accession}  {expr.shape[1]} samples x {expr.shape[0]} genes "
          f"→ {csv_path.name}  {counts}  (known={int(known.sum())}, unknown={int((~known).sum())})")
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
        # Boolean-known columns (e.g. smoke_type_known, weak_smoke_proxy_known)
        # must fill missing rows with False, never the string "unknown" —
        # that would silently corrupt a bool column into mixed
        # True/False/"unknown" object dtype, and "unknown" is truthy in
        # Python, which would make a barcode with NO metadata match look
        # like it has a KNOWN label. Every other column keeps the previous
        # "unknown" string fill for a missing per-cell value.
        _bool_known_cols = {c for c in joined.columns if c.endswith("_known")}
        for col in joined.columns:
            if col in _bool_known_cols:
                adata.obs[col] = joined[col].fillna(False).astype(bool).values
            else:
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
    atlas, not a dedicated smoking cohort. This function does NOT assign a
    verified smoke_type: it never writes "cigarette" (or any other verified
    label) for these cells. Instead every cell gets an explicit, honest
    provenance record:

      smoke_type_name    "unknown" for every cell (loaders.py's
                          _attach_standard_obs keeps this per-cell value
                          instead of falling back to the accession-level
                          blanket default that used to apply "cigarette"
                          here regardless of Disease_Identity).
      smoke_type_known    False for every cell — no verified smoking-status
                          measurement exists for this accession.
      weak_smoke_proxy_known      True only where Disease_Identity == "COPD"
                                  (COPD is strongly smoking-associated, but
                                  is not itself a verified cigarette-exposure
                                  measurement).
      weak_smoke_proxy_type       "COPD_diagnosis" where applicable, else None.
      weak_smoke_proxy_value      "cigarette" where applicable, else None —
                                  a candidate label, never written into
                                  smoke_type/smoke_type_name directly. See
                                  data/labellers.py::apply_weak_smoke_proxies
                                  for the explicit, opt-in-only promotion
                                  path (data.weak_labels.enabled).
      weak_smoke_proxy_source     "GSE136831 Disease_Identity" where applicable.
      weak_smoke_proxy_limitation human-readable caveat where applicable —
                                  COPD status is correlational, not a
                                  verified individual exposure record.

    IPF and Control rows get weak_smoke_proxy_known=False — IPF is not a
    smoking proxy, and "Control" here means "no interstitial lung disease",
    not "confirmed never-smoker" (this pipeline has no verified never-smoker
    field for this accession either, so Control rows stay smoke_type_known
    =False rather than being labeled "unexposed").
    """
    matches = list(src_dir.glob("*Samples.CellType.MetadataTable*"))
    if not matches:
        return None
    meta = pd.read_csv(matches[0], sep="\t", quotechar='"')
    meta = meta.set_index("CellBarcode_Identity")

    disease = meta["Disease_Identity"]
    is_copd = disease.astype(str).str.strip().str.upper() == "COPD"

    limitation = (
        "GSE136831 Disease_Identity=COPD is a documented weak proxy for cigarette "
        "exposure, not a verified individual smoking record — this accession is the "
        "Vanderbilt/Habermann interstitial lung disease atlas (COPD/IPF/Control), not "
        "a dedicated smoking cohort. COPD is strongly smoking-associated but not proof "
        "of exposure for any single donor."
    )

    return pd.DataFrame({
        "donor_id":         meta["Subject_Identity"],
        "disease_identity": disease,
        "cell_type":        meta["CellType_Category"],
        "smoke_type_name":  "unknown",
        "smoke_type_known": False,
        "weak_smoke_proxy_known":      is_copd,
        "weak_smoke_proxy_type":       np.where(is_copd, "COPD_diagnosis", None),
        "weak_smoke_proxy_value":      np.where(is_copd, "cigarette", None),
        "weak_smoke_proxy_source":     np.where(is_copd, "GSE136831 Disease_Identity", None),
        "weak_smoke_proxy_limitation": np.where(is_copd, limitation, None),
    }, index=meta.index)


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

    Malignancy label: sample_type ("Primary Tumor" vs. solid tissue normal
    (NAT), read from downloaders.py's file_meta.csv) is a BULK SAMPLE-LEVEL
    tumor/normal label, not a per-cell malignancy call — it describes which
    bulk specimen a sample was dissected from, and is only ever attached
    through load_microarray's samples_meta.csv pathway, which
    data/assay_mode.py keeps confined to the dedicated bulk_tcga path (see
    load_tcga_bulk_dataset in this module / preprocess.py). It must never
    be interpreted as, or merged into, a per-cell malignancy label for an
    unrelated single-cell dataset.

    Smoke type: TCGA-LUAD/LUSC do not carry verified per-patient smoking
    history in this pipeline. ">85% of these cohorts are smokers" (a
    cohort-level statistic sometimes cited for these projects) describes
    the COHORT, not any individual patient, and is never written here as a
    per-sample label — smoke_type_name is "unknown" and smoke_type_known
    is False for every TCGA sample by default. This replaces an earlier
    version of this function that defaulted every sample to "cigarette" as
    a "documented approximation" — that was a fabricated per-patient label
    with no verified basis and has been removed.
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
        # No verified per-patient smoking history — never "cigarette" by
        # default. See this function's docstring.
        "smoke_type": "unknown",
        "smoke_type_known": False,
        # sample_type-derived tumor/normal is a bulk-sample label, not a
        # per-cell malignancy call — see this function's docstring.
        "malignancy": [malignancy[s] for s in expr.columns],
        "subject_id": [subject_id[s] for s in expr.columns],
        "assay_mode": "bulk_tcga",
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


_TCGA_STAR_STAT_ROWS = {"N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"}


def _read_tcga_star_gene_counts(path: Path, value_col: str = "tpm_unstranded") -> pd.Series:
    """One GDC 'STAR - Counts' augmented gene-count file (open-access tier —
    downloaded directly via the GDC /data/{file_id} API, no gdc-client/token
    needed) -> a gene_id (Ensembl, version-stripped) -> value Series. Drops
    the four alignment-statistics pseudo-rows GDC includes in the same file
    (N_unmapped/N_multimapping/N_noFeature/N_ambiguous) — these are QC
    counters, not genes, and would otherwise silently enter the expression
    matrix as four bogus "genes" with mostly-empty values."""
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str, compression=None)
    df = df[~df["gene_id"].isin(_TCGA_STAR_STAT_ROWS)]
    df["gene_id"] = df["gene_id"].str.split(".").str[0]
    return df.set_index("gene_id")[value_col].astype(float)


def convert_tcga_vital_status(project: str, src_dir: Path, value_col: str = "tpm_unstranded") -> Optional[Path]:
    """
    TCGA-LUAD/TCGA-LUSC primary-tumor gene expression -> genuine subject-level
    vital-status (deceased vs. alive at last follow-up) outcome CSV, for the
    real subject-level cancer-outcome-prediction task Issue #16 could not
    previously produce (no cohort had both compatible expression AND a
    genuinely linked subject-level outcome). Unlike `convert_tcga`'s
    tumor/normal `sample_type` malignancy label (a property of which
    specimen was dissected, known at collection time), vital_status is a
    real, independently-recorded clinical outcome from GDC's demographic
    record for the same `case_id` the expression file is filed under — a
    genuine expression -> outcome subject-level link, not an inferred or
    fabricated one.

    Reads `src_dir/outcome_meta.csv` (file_id, case_id, submitter_id,
    vital_status, days_to_death — written by the real GDC case-query results
    used to build the download manifest) and `src_dir/<file_id>.tsv.gz`
    (GDC's open-access "STAR - Counts" augmented gene-count file for that
    case's primary-tumor sample; despite the extension these are plain
    tab-separated text, not gzip-compressed — GDC serves them that way).

    A case is excluded (never coerced) if: its vital_status is neither
    'Alive' nor 'Dead'; its expression file is missing; or more than one
    downloaded file resolves to the same case_id (this simple pipeline, like
    data/bulk_pipeline.py, requires exactly one sample per subject and does
    not implement multi-sample aggregation).
    """
    out_dir = _mkout()
    meta_csv = src_dir / "outcome_meta.csv"
    if not meta_csv.exists():
        print(f"[convert] {project}  no outcome_meta.csv in {src_dir} — re-download real TCGA vital-status data first.")
        return None

    meta = pd.read_csv(meta_csv, dtype=str)
    duplicate_cases = meta["case_id"].value_counts()
    duplicate_cases = set(duplicate_cases[duplicate_cases > 1].index)

    columns, vital_status, subject_id, excluded, reasons = {}, {}, {}, [], {}
    for _, row in meta.iterrows():
        file_id = row["file_id"]
        case_id = row["case_id"]
        vs = row["vital_status"]
        fpath = src_dir / f"{file_id}.tsv.gz"
        if case_id in duplicate_cases:
            excluded.append(file_id)
            reasons["duplicate_case_id"] = reasons.get("duplicate_case_id", 0) + 1
            continue
        if vs not in ("Alive", "Dead"):
            excluded.append(file_id)
            reasons["vital_status_unknown"] = reasons.get("vital_status_unknown", 0) + 1
            continue
        if not fpath.exists():
            excluded.append(file_id)
            reasons["file_missing"] = reasons.get("file_missing", 0) + 1
            continue
        try:
            columns[file_id] = _read_tcga_star_gene_counts(fpath, value_col=value_col)
        except Exception as exc:
            excluded.append(file_id)
            reasons["parse_error"] = reasons.get("parse_error", 0) + 1
            print(f"[convert] {project}  failed to parse {fpath.name}: {exc}")
            continue
        vital_status[file_id] = 1.0 if vs == "Dead" else 0.0
        subject_id[file_id] = case_id

    if not columns:
        print(f"[convert] {project}  no usable vital-status expression samples found under {src_dir}.")
        return None

    expr = pd.DataFrame(columns).fillna(0.0)
    csv_path = out_dir / f"{project}_vital_status.csv"
    meta_path = out_dir / f"{project}_vital_status_samples_meta.csv"
    expr.to_csv(csv_path)
    pd.DataFrame({
        "sample_id": expr.columns,
        "subject_id": [subject_id[s] for s in expr.columns],
        # case_id comes directly from GDC's own case record for this
        # file — an authoritative, GDC-assigned subject identifier, not an
        # inference from a title/metadata field the way GSE123352's
        # subject_id is — so it is always verified when present.
        "subject_id_verified": True,
        "vital_status_known": True,
        "vital_status": [vital_status[s] for s in expr.columns],
    }).to_csv(meta_path, index=False)

    n_dead = sum(v == 1.0 for v in vital_status.values())
    print(f"[convert] {project}  {expr.shape[1]} samples x {expr.shape[0]} genes → {csv_path.name}"
          f"  ({n_dead} dead / {expr.shape[1] - n_dead} alive; excluded {len(excluded)}: {reasons})")
    return csv_path


# ─── NLST outcomes ─────────────────────────────────────────────────────────────

def convert_nlst_outcomes(prsn_csv: Path) -> Path:
    """
    NLST prsn.csv (pid, candx, ...) → clean subject_id/cancer_label CSV for
    assemble_subject_bags()'s `cancer_outcomes` argument.

    A subject with a missing/NaN `candx` value is dropped from this CSV
    entirely, NOT written as cancer_label=0 — assemble_subject_bags()
    already treats a subject absent from this CSV as an unknown outcome
    (cancer_label_known=False), and a NaN candx must be treated identically
    to "no diagnosis field recorded", never silently promoted to a
    verified cancer-negative. If the `candx` column is missing from the
    source file entirely, every subject is dropped the same way (an empty
    outcomes CSV, not a blanket cancer_label=0 for everyone).
    """
    out_path = _mkout() / "nlst_outcomes.csv"
    df = pd.read_csv(prsn_csv, low_memory=False)
    if "candx" not in df.columns:
        print("[convert] NLST outcomes  WARNING: no 'candx' column in "
              f"{prsn_csv} — writing an empty outcomes CSV (every subject unknown), "
              "not a fabricated cancer_label.")
        out = pd.DataFrame({"subject_id": pd.Series(dtype=str), "cancer_label": pd.Series(dtype=int)})
    else:
        known = df[df["candx"].notna()]
        n_dropped = len(df) - len(known)
        if n_dropped:
            print(f"[convert] NLST outcomes  {n_dropped:,}/{len(df):,} subjects have a missing "
                  "candx value — excluded from this CSV (unknown outcome), not defaulted to 0.")
        out = pd.DataFrame({
            "subject_id":   known["pid"].astype(str),
            "cancer_label": (known["candx"] == 1).astype(int),
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
