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

from data.loaders    import load_scrna, load_microarray, load_mouse_scrna
from data.transforms import map_mouse_to_human, harmonize_gene_ids, qc_filter, normalize, smoke_aware_hvg, batch_correct, annotate_cell_types
from data.labellers  import transfer_nlst_labels, add_malignancy_labels, compute_smoke_class_weights
from data.assembly   import merge_sources, assemble_subject_bags, export_cell_dataset
from constants       import N_HVGS_DEFAULT


def load_config(config: Union[dict, str, Path]) -> dict:
    """Accept a plain dict or a path to a YAML config file."""
    if isinstance(config, (str, Path)):
        with open(config) as f:
            return yaml.safe_load(f)
    return config


def _load_all_sources(cfg: dict) -> list:
    """Shared source-loading step for run_pipeline() and run_pipeline_split_aware()."""
    adatas = []

    def _exists(path: str) -> bool:
        ok = Path(path).exists()
        if not ok:
            print(f"[preprocess] skip  {path}  (not found — run downloaders.py / converters.py)")
        return ok

    for path, stype, scol in cfg.get("scrna_sources", []):
        if _exists(path):
            adatas.append(normalize(qc_filter(harmonize_gene_ids(load_scrna(path, stype, scol)))))

    for path, stype in cfg.get("microarray_sources", []):
        if _exists(path):
            adatas.append(normalize(harmonize_gene_ids(load_microarray(path, stype))))

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
    return adatas


def _load_outcomes(cfg: dict) -> Optional[pd.DataFrame]:
    """Shared cancer-outcome loading step for run_pipeline() and run_pipeline_split_aware()."""
    outcome_sources = []
    if cfg.get("nlst_outcomes_csv") and Path(cfg["nlst_outcomes_csv"]).exists():
        outcome_sources.append(pd.read_csv(cfg["nlst_outcomes_csv"]))
    for path in cfg.get("extra_outcomes_csvs", []):
        if Path(path).exists():
            outcome_sources.append(pd.read_csv(path))

    if not outcome_sources:
        return None
    # A subject appearing in more than one source (e.g. NLST + TCGA) is a
    # cancer positive if any source says so.
    return (
        pd.concat(outcome_sources, ignore_index=True)
        .astype({"subject_id": str})
        .groupby("subject_id", as_index=False)["cancer_label"].max()
    )


def run_pipeline(config: Union[dict, str, Path]) -> Tuple[dict, list]:
    """
    Full preprocessing pipeline from raw files to training-ready arrays.

    WARNING — preprocessing leakage: this function scales (merge_sources)
    and selects highly-variable genes (smoke_aware_hvg) across the ENTIRE
    merged dataset before any train/val/test split exists, so validation/
    test cells influence those statistics. It has no notion of a split at
    all. Kept only for backward compatibility with existing callers/tests
    and for quick synthetic smoke-testing. For any real train/val/test
    experiment, use run_pipeline_split_aware() instead, which determines a
    subject-level split first and fits scaling/HVG selection on the train
    split only (see data/preprocessing.py, data/splitting.py).

    Parameters
    ----------
    config : dict | str | Path
        Plain dict or path to configs/default.yaml.

    config keys
    -----------
    scrna_sources      list[(h5ad_path, smoke_type, subject_col)]
    microarray_sources list[(csv_path, smoke_type)]
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
    cfg    = cfg.get("data", cfg)  # configs/default.yaml nests these under "data:"
    adatas = []

    def _exists(path: str) -> bool:
        ok = Path(path).exists()
        if not ok:
            print(f"[preprocess] skip  {path}  (not found — run downloaders.py / converters.py)")
        return ok

    for path, stype, scol in cfg.get("scrna_sources", []):
        if _exists(path):
            adatas.append(normalize(qc_filter(harmonize_gene_ids(load_scrna(path, stype, scol)))))

    for path, stype in cfg.get("microarray_sources", []):
        if _exists(path):
            adatas.append(normalize(harmonize_gene_ids(load_microarray(path, stype))))

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


def run_pipeline_split_aware(config: Union[dict, str, Path]) -> dict:
    """
    Leakage-free preprocessing pipeline: determines a subject-level
    train/val/test split BEFORE fitting any scaling or HVG-selection
    statistics, fits those statistics on the train split only
    (data/preprocessing.py::fit_preprocessing), and applies the same
    fitted transform unchanged to every split.

    Batch correction (Harmony) is the one remaining documented exception:
    Harmony has no train-only-fit / apply-to-new-data mode, so it is still
    fit across the full merged dataset after scaling — see
    PreprocessingArtifact.notes for why, and README.md's preprocessing-
    leakage section for the scientific implication.

    config keys — same `data:` keys as run_pipeline(), plus a top-level
    `split:` block (train_frac/val_frac/test_frac/seed/manifest_path/n_folds,
    see configs/default.yaml).

    Returns a dict:
      cell_data        — dict, CellLevelDataset arrays for ALL cells (every
                          split); filter by cell_metadata.csv's subject_id
                          + the split manifest to get train/val/test subsets
      bags              — list of subject-level MIL bags for ALL subjects
      split_manifest     — data.splitting.SplitManifest
      preprocessing_artifact — data.preprocessing.PreprocessingArtifact
    """
    full_cfg  = load_config(config)
    cfg       = full_cfg.get("data", full_cfg)
    split_cfg = full_cfg.get("split", {})

    from data.preprocessing import fit_preprocessing, apply_preprocessing
    from data.splitting import subject_train_val_test_split

    adatas = _load_all_sources(cfg)
    merged = merge_sources(*adatas, scale=False)   # gene intersection + concat only, NOT scaled

    manifest = subject_train_val_test_split(
        merged.obs["subject_id"].values,
        merged.obs["smoke_type"].values,
        train_frac=split_cfg.get("train_frac", 0.70),
        val_frac=split_cfg.get("val_frac", 0.15),
        test_frac=split_cfg.get("test_frac", 0.15),
        seed=split_cfg.get("seed", 42),
        manifest_path=split_cfg.get("manifest_path"),
    )

    artifact = fit_preprocessing(
        merged, set(manifest.train_subjects),
        n_hvgs=cfg.get("n_hvgs", N_HVGS_DEFAULT),
    )
    merged = apply_preprocessing(merged, artifact)

    merged = batch_correct(merged)     # see docstring: not leakage-free, documented limitation
    merged = annotate_cell_types(merged)

    if cfg.get("nlst_csv"):
        merged = transfer_nlst_labels(merged, cfg["nlst_csv"])
    merged = add_malignancy_labels(merged, cfg.get("tumor_barcodes"))

    cell_data = export_cell_dataset(merged, cfg.get("out_dir", "data/processed"))
    outcomes  = _load_outcomes(cfg)
    bags = assemble_subject_bags(
        merged, outcomes,
        min_cells_per_subject=cfg.get("min_cells_per_subject", 50),
    )

    return {
        "cell_data": cell_data,
        "bags": bags,
        "split_manifest": manifest,
        "preprocessing_artifact": artifact,
    }


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