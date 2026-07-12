"""
data/converters.py — raw GEO/NLST downloads → clean formats loaders.py expects.

downloaders.py fetches raw files as GEO/GDC publish them (series_matrix.txt.gz,
10x-style mtx triples, NLST screen/prsn CSVs). loaders.py only knows how to read
already-clean shapes (genes x samples CSV, h5ad, Loiselle wide CSV). This module
is the one-time bridge between the two, so neither has to know about the other's
format quirks.

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

def _parse_series_matrix(gz_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a GEO `*_series_matrix.txt.gz` file.

    Returns
    -------
    expr  : DataFrame, genes/probes (rows) x GSM sample IDs (cols)
    meta  : DataFrame, GSM sample IDs (rows) x characteristic fields (cols),
            parsed from `!Sample_characteristics_ch1` lines of the form
            "key: value".
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

    if not table_lines:
        raise ValueError(f"No expression table found in {gz_path}")

    from io import StringIO
    expr = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t", index_col=0)
    expr.columns = [c.strip('"') for c in expr.columns]

    meta = pd.DataFrame(index=sample_ids)
    for key, vals in char_rows.items():
        if len(vals) == len(sample_ids):
            meta[key] = vals

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


def convert_microarray(accession: str, gz_path: Path, default_smoke_type: str) -> Path:
    """
    GEO series matrix → (genes x samples CSV for load_microarray) +
    (sibling `_samples_meta.csv` with per-sample smoke_type, used by
    load_microarray to override the blanket default where GEO metadata
    lets us tell smokers from never-smokers within one series).
    """
    out_dir = _mkout()
    expr, meta = _parse_series_matrix(gz_path)

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


def convert_loiselle(accession: str, gz_path: Path) -> Path:
    """
    GSE130148 (Loiselle 2018) series matrix → wide CSV expected by
    load_pseudo_bulk_loiselle: samples (rows) x [gene columns..., smoke_type,
    cell_line, week, malignancy].

    Cell line / week / treatment are recovered from `!Sample_characteristics_ch1`
    where GEO exposes them; unmatched fields fall back to safe defaults with a
    printed warning so a reviewer can fix the source CSV by hand if needed.
    """
    out_dir = _mkout()
    expr, meta = _parse_series_matrix(gz_path)
    wide = expr.T  # samples x genes
    wide.index.name = "sample_id"

    def _find(colnames_contains: str) -> Optional[pd.Series]:
        col = next((c for c in meta.columns if colnames_contains in c), None)
        return meta[col] if col is not None else None

    def _find_first(*candidates: str) -> Optional[pd.Series]:
        # NB: can't chain with `or` — truthiness of a multi-row Series raises.
        for c in candidates:
            found = _find(c)
            if found is not None:
                return found
        return None

    cell_line = _find("cell line")
    week      = _find("week")
    treatment = _find_first("treatment", "agent", "exposure")

    wide["cell_line"] = (cell_line.reindex(wide.index).fillna("BEAS-2B").values
                          if cell_line is not None else "BEAS-2B")
    wide["week"] = (
        pd.to_numeric(week.reindex(wide.index).str.extract(r"(\d+)")[0], errors="coerce")
        .fillna(0).astype(int).values
        if week is not None else 0
    )

    def _smoke_from_treatment(v) -> str:
        v = str(v).lower()
        if "cannabis" in v or "weed" in v or "marijuana" in v:
            return "cannabis"
        if "tobacco" in v or "cigarette" in v:
            return "cigarette"
        return "unexposed"

    wide["smoke_type"] = (
        treatment.reindex(wide.index).map(_smoke_from_treatment).values
        if treatment is not None else "unexposed"
    )
    wide["malignancy"] = (
        (wide["smoke_type"] == "cigarette") & (wide["week"] >= 10)
    ).astype(float)

    missing = [name for name, s in
               [("cell line", cell_line), ("week", week), ("treatment", treatment)]
               if s is None]
    if missing:
        print(f"[convert] {accession}  WARNING: GEO metadata missing {missing} — "
              f"used defaults, verify {out_dir / (accession + '_loiselle.csv')} by hand")

    out_path = out_dir / f"{accession}_loiselle.csv"
    wide.to_csv(out_path)
    print(f"[convert] {accession}  {wide.shape[0]} conditions → {out_path.name}")
    return out_path


# ─── scRNA (10x-style) conversion ─────────────────────────────────────────────

def convert_scrna_10x(accession: str, src_dir: Path, donor_map: Optional[dict] = None) -> Path:
    """
    10x-style raw counts (matrix.mtx[.gz] + barcodes.tsv[.gz] + features/genes.tsv[.gz],
    or a single combined `*_RawCounts_Sparse.mtx.gz` with sibling barcode/feature
    files) → h5ad for load_scrna / load_mouse_scrna.

    donor_map: optional {barcode_prefix: donor_id} to populate obs['donor_id']
    when the raw files don't already encode it (GEO supplementary files rarely do —
    check the accession's associated paper/metadata for the barcode→donor mapping).
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
        import scipy.io as sio
        import scipy.sparse as sp

        mtx = sio.mmread(str(mtx_files[0])).tocsr()
        barcode_files = list(src_dir.glob("*barcodes*")) or list(src_dir.glob("*Barcodes*"))
        feature_files = (list(src_dir.glob("*features*")) or list(src_dir.glob("*genes*"))
                          or list(src_dir.glob("*Genes*")))

        n_obs_from_mtx_rows = mtx.shape[0]
        barcodes = (_read_id_list(barcode_files[0]) if barcode_files
                    else [f"cell_{i}" for i in range(n_obs_from_mtx_rows)])
        genes = (_read_id_list(feature_files[0]) if feature_files
                 else [f"gene_{i}" for i in range(mtx.shape[1])])

        import anndata as ad
        adata = ad.AnnData(X=sp.csr_matrix(mtx),
                            obs=pd.DataFrame(index=barcodes),
                            var=pd.DataFrame(index=genes))

    if donor_map:
        prefixes = adata.obs_names.str.extract(r"^([^-_]+)")[0]
        adata.obs["donor_id"] = prefixes.map(donor_map).fillna("unknown").values
    elif "donor_id" not in adata.obs.columns:
        adata.obs["donor_id"] = "unknown"

    adata.write_h5ad(out_path)
    print(f"[convert] {accession}  {adata.n_obs:,} cells x {adata.n_vars:,} genes → {out_path.name}")
    return out_path


def _read_id_list(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [line.split("\t")[0].strip() for line in f if line.strip()]


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

    if accession == "GSE130148":
        gz = src / expected_file
        return convert_loiselle(accession, gz) if gz.exists() else _missing(accession, gz)

    if accession in ("GSE136831", "GSE288003"):
        return convert_scrna_10x(accession, src) if src.exists() else _missing(accession, src)

    gz = src / expected_file
    return convert_microarray(accession, gz, smoke_type) if gz.exists() else _missing(accession, gz)


def _missing(accession: str, path: Path) -> None:
    print(f"[convert] {accession}  source not found at {path} — download it first:\n"
          f"  python3 src/data/downloaders.py --accession {accession}")
    return None


def convert_all() -> None:
    from data.downloaders import GEO_DATASETS
    for accession in GEO_DATASETS:
        convert_accession(accession)

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
    parser.add_argument("--nlst", action="store_true", help="Convert NLST prsn.csv outcomes only")
    args = parser.parse_args()

    if args.all:
        convert_all()
    elif args.accession:
        convert_accession(args.accession)
    elif args.nlst:
        nlst_prsn = RAW / "subjects" / "NLST" / "prsn.csv"
        if nlst_prsn.exists():
            convert_nlst_outcomes(nlst_prsn)
        else:
            print(f"missing {nlst_prsn}")
    else:
        parser.print_help()
        print("\nQuick start:\n  python3 src/data/converters.py --all")
