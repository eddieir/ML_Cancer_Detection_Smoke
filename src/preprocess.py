"""
preprocess.py — Pipeline orchestrator. No logic lives here.
Accepts a config dict or a path to configs/default.yaml.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import tempfile

import numpy as np
import pandas as pd
import yaml

from data.loaders    import load_scrna, load_microarray, load_pseudo_bulk_loiselle, load_mouse_scrna
from data.transforms import map_mouse_to_human, qc_filter, normalize, smoke_aware_hvg, batch_correct, annotate_cell_types
from data.labellers  import transfer_nlst_labels, add_malignancy_labels, compute_smoke_class_weights
from data.assembly   import merge_sources, assemble_subject_bags, export_cell_dataset
from constants       import N_HVGS_DEFAULT


def load_config(config: Union[dict, str, Path]) -> dict:
    """Accept a plain dict or a path to a YAML config file."""
    if isinstance(config, (str, Path)):
        with open(config) as f:
            return yaml.safe_load(f)
    return config


def run_pipeline(config: Union[dict, str, Path]) -> Tuple[dict, list]:
    """
    Full preprocessing pipeline from raw files to training-ready arrays.

    Parameters
    ----------
    config : dict | str | Path
        Plain dict or path to configs/default.yaml.

    config keys
    -----------
    scrna_sources      list[(h5ad_path, smoke_type, subject_col)]
    microarray_sources list[(csv_path, smoke_type)]
    loiselle_path      str | None
    gse288003_path     str | None   mouse e-cig → triggers ortholog mapping
    nlst_csv           str | None   cigar/dual-use label transfer
    nlst_outcomes_csv  str | None   subject_id + cancer_label for Phase 2/3
    extra_outcomes_csvs list | None additional subject_id + cancer_label sources
                                    (e.g. TCGA tumor/NAT outcomes from converters.py)
    tumor_barcodes     list | None
    n_hvgs             int          default 2000
    out_dir            str

    Returns
    -------
    cell_data   dict  →  CellLevelDataset  (Phase 1)
    bags        list  →  SubjectLevelDataset (Phase 2/3)
    """
    cfg    = load_config(config)
    adatas = []

    def _exists(path: str) -> bool:
        ok = Path(path).exists()
        if not ok:
            print(f"[preprocess] skip  {path}  (not found — run downloaders.py / converters.py)")
        return ok

    for path, stype, scol in cfg.get("scrna_sources", []):
        if _exists(path):
            adatas.append(normalize(qc_filter(load_scrna(path, stype, scol))))

    for path, stype in cfg.get("microarray_sources", []):
        if _exists(path):
            adatas.append(normalize(load_microarray(path, stype)))

    if cfg.get("loiselle_path") and _exists(cfg["loiselle_path"]):
        adatas.append(load_pseudo_bulk_loiselle(cfg["loiselle_path"]))

    if cfg.get("gse288003_path") and _exists(cfg["gse288003_path"]):
        a = map_mouse_to_human(load_mouse_scrna(cfg["gse288003_path"]))
        adatas.append(normalize(qc_filter(a)))

    if not adatas:
        raise ValueError(
            "No data sources found. Run:\n"
            "  python3 src/data/downloaders.py --all\n"
            "  python3 src/data/converters.py --all\n"
            "or pass a config with paths to already-converted files."
        )

    merged = merge_sources(*adatas)
    merged = smoke_aware_hvg(merged, n_hvgs=cfg.get("n_hvgs", N_HVGS_DEFAULT))
    merged = batch_correct(merged)
    merged = annotate_cell_types(merged)

    if cfg.get("nlst_csv"):
        merged = transfer_nlst_labels(merged, cfg["nlst_csv"])

    merged = add_malignancy_labels(merged, cfg.get("tumor_barcodes"))

    cell_data = export_cell_dataset(merged, cfg.get("out_dir", "data/processed"))

    outcome_sources = []
    if cfg.get("nlst_outcomes_csv") and Path(cfg["nlst_outcomes_csv"]).exists():
        outcome_sources.append(pd.read_csv(cfg["nlst_outcomes_csv"]))
    for path in cfg.get("extra_outcomes_csvs", []):
        if _exists(path):
            outcome_sources.append(pd.read_csv(path))

    outcomes: Optional[pd.DataFrame] = None
    if outcome_sources:
        # A subject appearing in more than one source (e.g. NLST + TCGA) is
        # a cancer positive if any source says so.
        outcomes = (
            pd.concat(outcome_sources, ignore_index=True)
            .astype({"subject_id": str})
            .groupby("subject_id", as_index=False)["cancer_label"].max()
        )

    bags = assemble_subject_bags(
        merged, outcomes,
        min_cells_per_subject=cfg.get("min_cells_per_subject", 50),
    )
    return cell_data, bags


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import anndata as ad
    import scipy.sparse as sp
    from constants import ALL_SMOKE_MARKERS

    print("=== preprocess.py smoke test ===\n")
    np.random.seed(42)
    N, G = 600, 3000

    genes = [f"GENE_{i}" for i in range(G)]
    genes[:10] = [f"MT-{i}" for i in range(10)]
    for i, mk in enumerate(ALL_SMOKE_MARKERS[:8]):
        genes[100 + i] = mk

    raw = np.random.negative_binomial(5, 0.7, (N, G)).astype("float32")
    obs = pd.DataFrame({
        "donor_id":       [f"sub_{i // 20}" for i in range(N)],
        "subject_id":     [f"sub_{i // 20}" for i in range(N)],
        "smoke_type":     np.random.randint(0, 6, N),
        "smoke_type_name": "mixed",
        "data_modality":  "scrna",
        "is_pseudo_bulk": False,
        "malignancy":     0.0,
        "cell_type_id":   0,
    }, index=[f"c{i}" for i in range(N)])

    adata = ad.AnnData(X=sp.csr_matrix(raw), obs=obs, var=pd.DataFrame(index=genes))

    # Save temp h5ad so run_pipeline() is tested end-to-end
    with tempfile.TemporaryDirectory() as tmp:
        h5ad_path = str(Path(tmp) / "test.h5ad")
        out_dir   = str(Path(tmp) / "processed")
        adata.write_h5ad(h5ad_path)

        config = {
            "scrna_sources":          [(h5ad_path, "cigarette", "donor_id")],
            "microarray_sources":     [],
            "n_hvgs":                 200,
            "min_cells_per_subject":  10,
            "out_dir":                out_dir,
        }

        cell_data, bags = run_pipeline(config)

    print(f"\nbags : {len(bags)}")
    print(f"keys : {list(bags[0].keys())}")
    print(f"shape: {bags[0]['gene_matrix'].shape}")
    print("\n=== PASSED ===")