"""
data/downloaders.py
Downloads all raw data sources to data/raw/.
Called before preprocess.py — not part of the preprocessing pipeline itself.

Usage:
    python3 src/data/downloaders.py --all
    python3 src/data/downloaders.py --geo
    python3 src/data/downloaders.py --tcga   # requires GDC token
    python3 src/data/downloaders.py --check  # show what is / isn't downloaded
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

import requests

# ─── Raw data root ────────────────────────────────────────────────────────────
RAW = Path(__file__).parents[2] / "data" / "raw"

# ─── GEO dataset registry ─────────────────────────────────────────────────────
# Each entry: accession → (subdir, description, smoke_type, expected_filename)
GEO_DATASETS = {
    "GSE994": (
        "cigarette/GSE994",
        "Bronchial epithelial microarray — 75 subjects, cigarette",
        "cigarette",
        "GSE994_series_matrix.txt.gz",
    ),
    "GSE123352": (
        "cigarette/GSE123352",
        "Lung tissue RNA-seq — 176 subjects, ever/never smoker",
        "cigarette",
        "GSE123352_series_matrix.txt.gz",
    ),
    "GSE136831": (
        "cigarette/GSE136831",
        "Lung scRNA-seq atlas — 312,928 cells, cigarette smokers",
        "cigarette",
        "GSE136831_RawCounts_Sparse.mtx.gz",
    ),
    "GSE288003": (
        "vape/GSE288003",
        "Mouse lung scRNA-seq — e-cig aerosol exposure",
        "vape",
        "GSE288003_series_matrix.txt.gz",
    ),
    "GSE130148": (
        "cannabis/GSE130148",
        "Loiselle 2018 — BEAS-2B exposed to tobacco and cannabis smoke",
        "cannabis",
        "GSE130148_series_matrix.txt.gz",
    ),
}

# ─── TCGA dataset registry ────────────────────────────────────────────────────
TCGA_DATASETS = {
    "TCGA-LUAD": {
        "subdir": "malignancy/TCGA-LUAD",
        "description": "Lung adenocarcinoma — tumor + NAT cells, malignancy labels",
        "gdc_filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id",
                                         "value": ["TCGA-LUAD"]}},
                {"op": "in", "content": {"field": "data_type",
                                         "value": ["Gene Expression Quantification"]}},
                {"op": "in", "content": {"field": "experimental_strategy",
                                         "value": ["RNA-Seq"]}},
            ],
        },
    },
    "TCGA-LUSC": {
        "subdir": "malignancy/TCGA-LUSC",
        "description": "Lung squamous cell carcinoma — tumor + NAT cells",
        "gdc_filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id",
                                         "value": ["TCGA-LUSC"]}},
                {"op": "in", "content": {"field": "data_type",
                                         "value": ["Gene Expression Quantification"]}},
                {"op": "in", "content": {"field": "experimental_strategy",
                                         "value": ["RNA-Seq"]}},
            ],
        },
    },
}

GDC_FILES_API = "https://api.gdc.cancer.gov/files"
GDC_DATA_API  = "https://api.gdc.cancer.gov/data"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _already_downloaded(dest: Path, filename: str) -> bool:
    return (dest / filename).exists()


def _download_file(url: str, dest: Path, filename: str, chunk_size: int = 8192) -> Path:
    """Stream-download a file with a simple progress indicator."""
    out = dest / filename
    if out.exists():
        print(f"  [skip] already exists: {out.name}")
        return out

    print(f"  [download] {filename} ...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0

    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {pct:5.1f}%  {downloaded//1024//1024} MB", end="", flush=True)
    print()
    return out


# ─── GEO download ─────────────────────────────────────────────────────────────

def download_geo(accession: str, dest_override: Optional[Path] = None) -> Path:
    """
    Download a GEO series matrix and supplementary files via NCBI FTP.
    Handles both matrix files and raw count files.
    """
    try:
        import GEOparse
    except ImportError:
        raise ImportError("pip install GEOparse")

    subdir, desc, smoke_type, expected_file = GEO_DATASETS[accession]
    dest = _mkdir(dest_override or RAW / subdir)

    print(f"\n[GEO] {accession} — {desc}")

    # GEOparse downloads the series matrix and caches it
    gse = GEOparse.get_GEO(geo=accession, destdir=str(dest), silent=True)

    # Also grab supplementary files (raw counts for scRNA-seq datasets)
    ncbi_ftp = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{accession[:-3]}nnn/{accession}/suppl/"
    )
    try:
        r = requests.get(ncbi_ftp, timeout=30)
        if r.ok:
            from html.parser import HTMLParser

            class _LinkParser(HTMLParser):
                links = []
                def handle_starttag(self, tag, attrs):
                    if tag == "a":
                        for k, v in attrs:
                            if k == "href" and v.endswith((".gz", ".h5", ".mtx.gz", ".h5ad")):
                                self.links.append(v)

            parser = _LinkParser()
            parser.feed(r.text)
            for link in parser.links:
                _download_file(ncbi_ftp + link, dest, link)
    except Exception as e:
        print(f"  [warn] supplementary FTP fetch failed: {e}")

    print(f"  [done] {accession} → {dest}/")
    return dest


def download_all_geo() -> None:
    """Download all GEO datasets in the registry."""
    for accession in GEO_DATASETS:
        download_geo(accession)


# ─── TCGA download ────────────────────────────────────────────────────────────

def download_tcga(project: str, token_file: Optional[str] = None) -> Path:
    """
    Download TCGA gene expression files via GDC API.
    Requires a GDC user token for controlled-access data.
    Token: https://portal.gdc.cancer.gov/ → Login → Download Token

    Files downloaded: HTSeq counts (RNA-Seq), one per case.
    """
    if token_file is None:
        token_file = os.environ.get("GDC_TOKEN_FILE")
    if not token_file or not Path(token_file).exists():
        print(
            f"\n[TCGA] {project} — GDC token required.\n"
            "  1. Register at https://portal.gdc.cancer.gov/\n"
            "  2. Log in → click your name → Download Token\n"
            "  3. Re-run: python3 src/data/downloaders.py --tcga "
            "--token /path/to/token.txt\n"
            "  OR set env var: export GDC_TOKEN_FILE=/path/to/token.txt"
        )
        return RAW / TCGA_DATASETS[project]["subdir"]

    cfg  = TCGA_DATASETS[project]
    dest = _mkdir(RAW / cfg["subdir"])
    print(f"\n[TCGA] {project} — {cfg['description']}")

    # Query GDC for file UUIDs
    payload = {
        "filters": cfg["gdc_filters"],
        "fields": "file_id,file_name,cases.case_id,cases.samples.sample_type",
        "format": "JSON",
        "size": "2000",
    }
    r = requests.post(GDC_FILES_API, json=payload, timeout=60)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    print(f"  {len(hits)} files found")

    # Write manifest for gdc-client
    manifest_path = dest / "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for hit in hits:
            f.write(f"{hit['file_id']}\t{hit['file_name']}\t\t\t\n")

    # Download via gdc-client (fastest method for large TCGA files)
    if shutil.which("gdc-client"):
        token = Path(token_file).read_text().strip()
        cmd = [
            "gdc-client", "download",
            "-t", token_file,
            "-d", str(dest),
            "-m", str(manifest_path),
            "--retry-amount", "3",
        ]
        print(f"  [gdc-client] downloading {len(hits)} files ...")
        subprocess.run(cmd, check=True)
    else:
        print(
            "  [warn] gdc-client not found. Install: "
            "https://gdc.cancer.gov/access-data/gdc-data-transfer-tool\n"
            f"  Manifest saved: {manifest_path}\n"
            "  Run manually: gdc-client download -t token.txt "
            f"-d {dest} -m {manifest_path}"
        )

    print(f"  [done] {project} → {dest}/")
    return dest


def download_all_tcga(token_file: Optional[str] = None) -> None:
    for project in TCGA_DATASETS:
        download_tcga(project, token_file)


# ─── NLST instructions ────────────────────────────────────────────────────────

def print_nlst_instructions() -> None:
    """
    NLST requires manual registration — cannot be automated.
    Prints the exact steps needed.
    """
    dest = RAW / "subjects/NLST"
    print(f"""
[NLST] Manual download required — automated access not permitted.

Steps:
  1. Register at https://cdas.cancer.gov/nlst/
  2. Complete the Data Use Agreement (takes 1-3 business days)
  3. After approval, download:
       - NLST_Study_Data_Description_2011.xlsx
       - screen.csv   (participant screening data, includes CIGSMOK + CIGAR fields)
       - prsn.csv     (person-level data with cancer outcome)
  4. Place files in:
       {dest}/

Fields used by labellers.py:
  screen.pid       → subject_id
  screen.CIGSMOK   → cigarette smoker flag (1=current, 2=former)
  screen.CIGAR     → cigar use (1=yes)
  prsn.candx       → cancer diagnosis (1=yes) → cancer_label in subject bags
""")


# ─── Status check ─────────────────────────────────────────────────────────────

def check_status() -> None:
    """Show what has / hasn't been downloaded."""
    print("\n=== Data Download Status ===\n")

    print("GEO datasets:")
    for acc, (subdir, desc, smoke_type, expected_file) in GEO_DATASETS.items():
        path = RAW / subdir / expected_file
        status = "✓" if path.exists() else "✗ missing"
        print(f"  [{status}] {acc:<12} {smoke_type:<12} {desc[:55]}")

    print("\nTCGA datasets:")
    for proj, cfg in TCGA_DATASETS.items():
        path  = RAW / cfg["subdir"]
        files = list(path.glob("**/*.gz")) if path.exists() else []
        status = f"✓ {len(files)} files" if files else "✗ missing"
        print(f"  [{status}] {proj:<14} {cfg['description'][:50]}")

    print("\nNLST:")
    nlst_screen = RAW / "subjects/NLST/screen.csv"
    nlst_prsn   = RAW / "subjects/NLST/prsn.csv"
    print(f"  [{'✓' if nlst_screen.exists() else '✗ missing'}] screen.csv")
    print(f"  [{'✓' if nlst_prsn.exists()   else '✗ missing'}] prsn.csv")
    if not nlst_screen.exists():
        print("  → Run: python3 src/data/downloaders.py --nlst-instructions")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download raw data sources")
    parser.add_argument("--all",               action="store_true", help="Download GEO + TCGA")
    parser.add_argument("--geo",               action="store_true", help="Download all GEO datasets")
    parser.add_argument("--tcga",              action="store_true", help="Download TCGA-LUAD + LUSC")
    parser.add_argument("--accession",         type=str,            help="Download one GEO accession")
    parser.add_argument("--token",             type=str,            help="GDC token file path")
    parser.add_argument("--nlst-instructions", action="store_true", help="Print NLST download steps")
    parser.add_argument("--check",             action="store_true", help="Show download status")
    args = parser.parse_args()

    if args.check:
        check_status()
    elif args.nlst_instructions:
        print_nlst_instructions()
    elif args.accession:
        download_geo(args.accession)
    elif args.geo or args.all:
        download_all_geo()
        if args.all:
            download_all_tcga(args.token)
    elif args.tcga:
        download_all_tcga(args.token)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python3 src/data/downloaders.py --check")
        print("  python3 src/data/downloaders.py --geo")
        print("  python3 src/data/downloaders.py --tcga --token /path/to/gdc_token.txt")
        print("  python3 src/data/downloaders.py --nlst-instructions")