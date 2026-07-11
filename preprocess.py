"""
preprocess.py — Pipeline orchestrator. No logic lives here.
Imports from data/ submodules and composes the full pipeline.
"""

from typing import Optional, Tuple
import pandas as pd

from data.loaders    import load_scrna, load_microarray, load_pseudo_bulk_loiselle, load_mouse_scrna
from data.transforms import map_mouse_to_human, qc_filter, normalize, smoke_aware_hvg, batch_correct, annotate_cell_types
from data.labellers  import transfer_nlst_labels, add_malignancy_labels, compute_smoke_class_weights
from data.assembly   import merge_sources, assemble_subject_bags, export_cell_dataset
from constants       import N_HVGS_DEFAULT


def run_pipeline(config: dict) -> Tuple[dict, list]:
    """
    Full preprocessing pipeline from raw files to training-ready arrays.

    config keys
    -----------
    scrna_sources      list[(h5ad_path, smoke_type, subject_col)]
    microarray_sources list[(csv_path, smoke_type)]
    loiselle_path      str | None
    gse288003_path     str | None   mouse e-cig → triggers ortholog mapping
    nlst_csv           str | None   cigar/dual-use label transfer
    nlst_outcomes_csv  str | None   subject_id + cancer_label for Phase 2/3
    tumor_barcodes     list | None
    n_hvgs             int          default 2000
    out_dir            str

    Returns
    -------
    cell_data   dict  →  CellLevelDataset  (Phase 1)
    bags        list  →  SubjectLevelDataset (Phase 2/3)
    """
    adatas = []

    for path, stype, scol in config.get("scrna_sources", []):
        a = load_scrna(path, stype, scol)
        adatas.append(normalize(qc_filter(a)))

    for path, stype in config.get("microarray_sources", []):
        adatas.append(normalize(load_microarray(path, stype)))

    if config.get("loiselle_path"):
        adatas.append(load_pseudo_bulk_loiselle(config["loiselle_path"]))

    if config.get("gse288003_path"):
        a = map_mouse_to_human(load_mouse_scrna(config["gse288003_path"]))
        adatas.append(normalize(qc_filter(a)))

    if not adatas:
        raise ValueError("No data sources provided in config.")

    merged = merge_sources(*adatas)
    merged = smoke_aware_hvg(merged, n_hvgs=config.get("n_hvgs", N_HVGS_DEFAULT))
    merged = batch_correct(merged)
    merged = annotate_cell_types(merged)

    if config.get("nlst_csv"):
        merged = transfer_nlst_labels(merged, config["nlst_csv"])

    merged = add_malignancy_labels(merged, config.get("tumor_barcodes"))

    cell_data = export_cell_dataset(merged, config.get("out_dir", "data/processed"))

    outcomes: Optional[pd.DataFrame] = None
    if config.get("nlst_outcomes_csv"):
        outcomes = pd.read_csv(config["nlst_outcomes_csv"])

    bags = assemble_subject_bags(merged, outcomes)
    return cell_data, bags
