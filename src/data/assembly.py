"""
data/assembly.py — merging sources, building MIL bags, and exporting.
No label logic, no transformation logic.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from constants import DOSE_UNKNOWN, SPECIES_HUMAN, DEFAULT_EXPERIMENT_MODE, DEFAULT_ASSAY_POLICY


def merge_sources(*adatas: ad.AnnData, scale: bool = True,
                   experiment_mode: str = DEFAULT_EXPERIMENT_MODE,
                   assay_policy: str = DEFAULT_ASSAY_POLICY) -> ad.AnnData:
    """
    Concatenate heterogeneous sources on common gene intersection.
    Assigns batch column for downstream Harmony correction.

    scale=True (default, legacy behaviour) z-scores the merged matrix across
    every cell from every source BEFORE any train/val/test split exists —
    this is preprocessing leakage: validation/test cells influence the
    mean/std used to scale the training cells. Kept as the default only for
    backward compatibility with existing callers/tests; the leakage-free
    path (preprocess.py::run_pipeline_split_aware, data/preprocessing.py)
    calls merge_sources(*adatas, scale=False) and fits scaling on the
    train split only via PreprocessingArtifact.

    Species safety (see data/species_policy.py): every source's
    obs["species"] (defaulting to "human" for sources that predate this
    field) is checked against `experiment_mode` before concatenating.
    human_only (the default) refuses to merge anything but human cells;
    mixing species requires experiment_mode='cross_species_pretraining' or
    'cross_species_domain_adaptation'. This is the single enforcement point
    for the rule that ortholog-mapped mouse expression must never be
    silently treated as the same domain as measured human expression.

    Assay safety (see data/assay_policy.py): every source's
    obs["is_pseudo_bulk"] is checked against `assay_policy` before
    concatenating — a defensive second gate independent of preprocess.py::
    _load_all_sources's own check, so a caller that builds/loads AnnData
    objects directly (bypassing _load_all_sources entirely, e.g. a hand-
    constructed or corrupted object in a test or a notebook) still cannot
    merge pseudo-bulk rows into a single_cell_only run. A source with no
    obs["is_pseudo_bulk"] column at all (predates this field, e.g. an older
    synthetic fixture) is treated as all-real-cells for backward
    compatibility — the same "missing column -> legacy default, not a
    validated safety claim" pattern this module already uses for
    obs["species"] above and obs["smoke_type_known"]/["malignancy_known"]
    elsewhere in this codebase.
    """
    from data.species_policy import assert_single_species_or_explicit
    from data.assay_policy import assert_rows_match_policy

    species_values = [
        (a.obs["species"].iloc[0] if "species" in a.obs.columns and a.n_obs else SPECIES_HUMAN)
        for a in adatas
    ]
    assert_single_species_or_explicit(species_values, experiment_mode)

    for i, a in enumerate(adatas):
        is_bulk = (
            a.obs["is_pseudo_bulk"].values if "is_pseudo_bulk" in a.obs.columns
            else np.zeros(a.n_obs, dtype=bool)
        )
        assert_rows_match_policy(is_bulk, assay_policy, context=f"merge_sources source #{i}")

    genes = adatas[0].var_names
    for a in adatas[1:]:
        genes = genes.intersection(a.var_names)
    genes = list(genes)
    print(f"[assembly] merge  {len(genes):,} common genes across {len(adatas)} sources")

    clipped = []
    for i, a in enumerate(adatas):
        c = a[:, genes].copy()
        c.obs["batch"] = f"source_{i}"
        clipped.append(c)

    merged = ad.concat(clipped, axis=0, join="inner", label="source")
    merged.var_names_make_unique()
    if scale:
        import scipy.sparse as sp
        if sp.issparse(merged.X):
            merged.X = merged.X.toarray()          # explicit: avoids UserWarning from scale()
        sc.pp.scale(merged, max_value=10)
    print(f"[assembly] merge  final {merged.n_obs:,} cells x {merged.n_vars:,} genes")
    return merged


def assemble_subject_bags(
    adata: ad.AnnData,
    cancer_outcomes: Optional[pd.DataFrame] = None,
    min_cells_per_subject: int = 50,
    assay_policy: str = DEFAULT_ASSAY_POLICY,
) -> list:
    """
    Group cells by subject_id into MIL bags for Phase 2/3 training.
    Novel: first MIL bag construction from heterogeneous multi-source
    scRNA-seq smoke data linked to NLST cancer outcomes.

    Assay safety (see data/assay_policy.py): a pseudo-bulk row has no
    biological meaning as one element of a subject's MIL cell bag (no real
    cell type, no per-cell malignancy signal) — assay_policy is enforced
    here as well as at merge_sources()/export_cell_dataset(), so a bag can
    never silently be built from a mix of real cells and pseudo-bulk
    "cells" even if this function is called directly on a hand-built or
    corrupted AnnData. Missing obs["is_pseudo_bulk"] defaults to
    all-real-cells (legacy fixtures) rather than raising — see
    merge_sources()'s docstring for the same pattern.

    Parameters
    ----------
    cancer_outcomes : DataFrame[subject_id, cancer_label]
    """
    from data.assay_policy import assert_rows_match_policy

    is_bulk = (
        adata.obs["is_pseudo_bulk"].values if "is_pseudo_bulk" in adata.obs.columns
        else np.zeros(adata.n_obs, dtype=bool)
    )
    assert_rows_match_policy(is_bulk, assay_policy, context="assemble_subject_bags")

    outcome_map: dict = {}
    if cancer_outcomes is not None:
        outcome_map = dict(zip(
            cancer_outcomes["subject_id"].astype(str),
            cancer_outcomes["cancer_label"].astype(int),
        ))

    X = np.array(
        adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray(),
        dtype=np.float32,
    )
    bags, skipped = [], 0
    n_known_pos = n_known_neg = n_unknown = 0

    for sid in adata.obs["subject_id"].unique():
        mask = (adata.obs["subject_id"] == sid).values
        if mask.sum() < min_cells_per_subject:
            skipped += 1
            continue

        # A subject absent from cancer_outcomes has an UNKNOWN outcome, not
        # a verified negative. Silently defaulting missing outcomes to 0
        # would train/evaluate the cancer head against fabricated negative
        # labels for every subject we simply have no outcome data for
        # (e.g. any scRNA-seq donor never linked to NLST/TCGA). cancer_label
        # is None and cancer_label_known=False for those; consumers
        # (SubjectLevelDataset, Trainer.phase2/3, Evaluator) must exclude
        # unknown-outcome bags from cancer-outcome supervision/evaluation,
        # though the bag remains usable for cell-level smoke/malignancy tasks.
        outcome = outcome_map.get(str(sid))
        known = outcome is not None
        if known:
            n_known_pos += int(outcome == 1)
            n_known_neg += int(outcome == 0)
        else:
            n_unknown += 1

        bags.append({
            "subject_id":         sid,
            "gene_matrix":        X[mask],
            "cell_type_ids":      adata.obs["cell_type_id"].values[mask].astype(int),
            "smoke_labels":       adata.obs["smoke_type"].values[mask].astype(int),
            "smoke_known":        (
                adata.obs["smoke_type_known"].values[mask].astype(bool)
                if "smoke_type_known" in adata.obs.columns
                else np.ones(mask.sum(), dtype=bool)  # legacy sources: treated as known, unchanged
            ),
            "malig_labels":       adata.obs["malignancy"].values[mask].astype(np.float32),
            "malig_known":        (
                adata.obs["malignancy_known"].values[mask].astype(bool)
                if "malignancy_known" in adata.obs.columns
                else np.zeros(mask.sum(), dtype=bool)
            ),
            "cancer_label":       outcome if known else None,
            "cancer_label_known": known,
        })

    print(f"[assembly] bags  {len(bags)} subjects  |  {skipped} dropped (<{min_cells_per_subject} cells)")
    print(f"[assembly] cancer outcomes  known_positive={n_known_pos}  "
          f"known_negative={n_known_neg}  unknown={n_unknown}")
    if cancer_outcomes is None:
        print("[assembly] WARNING: no cancer_outcomes source given — "
              "ALL subjects have unknown cancer outcome")
    return bags


def export_cell_dataset(
    adata: ad.AnnData,
    out_dir: str = "data/processed",
    assay_policy: str = DEFAULT_ASSAY_POLICY,
) -> dict:
    """
    Save numpy arrays for CellLevelDataset (Phase 1 training).

    Assay safety (see data/assay_policy.py): the final gate before rows
    become the numpy files CellLevelDataset.from_dir() reads — a pseudo-
    bulk row must never reach a "cell-level" training array. Raises before
    writing anything if any row's obs["is_pseudo_bulk"] is incompatible
    with assay_policy. Missing obs["is_pseudo_bulk"] defaults to
    all-real-cells (legacy fixtures) — see merge_sources()'s docstring.
    """
    from data.assay_policy import assert_rows_match_policy

    is_bulk = (
        adata.obs["is_pseudo_bulk"].values if "is_pseudo_bulk" in adata.obs.columns
        else np.zeros(adata.n_obs, dtype=bool)
    )
    assert_rows_match_policy(is_bulk, assay_policy, context="export_cell_dataset")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X     = np.array(adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray(), dtype=np.float32)
    smoke = adata.obs["smoke_type"].values.astype(np.int64)
    smoke_known = (adata.obs["smoke_type_known"].values.astype(bool)
                   if "smoke_type_known" in adata.obs.columns
                   else np.ones(X.shape[0], dtype=bool))  # legacy sources: treated as known, unchanged
    malig = adata.obs["malignancy"].values.astype(np.float32)
    malig_known = (adata.obs["malignancy_known"].values.astype(bool)
                   if "malignancy_known" in adata.obs.columns
                   else np.zeros(X.shape[0], dtype=bool))
    ctype = adata.obs["cell_type_id"].values.astype(np.int64)
    dose  = (adata.obs["exposure_dose"].values.astype(np.float32)
             if "exposure_dose" in adata.obs.columns
             else np.full(X.shape[0], DOSE_UNKNOWN, dtype=np.float32))

    np.save(out / "gene_matrix.npy",       X)
    np.save(out / "smoke_labels.npy",      smoke)
    np.save(out / "smoke_labels_known.npy", smoke_known)
    np.save(out / "malignancy_labels.npy", malig)
    np.save(out / "malignancy_known.npy",  malig_known)
    np.save(out / "cell_type_ids.npy",     ctype)
    np.save(out / "exposure_dose.npy",     dose)
    adata.obs.to_csv(out / "cell_metadata.csv")
    adata.var.to_csv(out / "gene_list.csv")

    n_known = int((dose >= 0).sum())
    n_malig_known_pos = int((malig_known & (malig == 1.0)).sum())
    n_malig_known_neg = int((malig_known & (malig == 0.0)).sum())
    n_malig_unknown   = int((~malig_known).sum())
    n_smoke_known   = int(smoke_known.sum())
    n_smoke_unknown = int((~smoke_known).sum())
    print(f"[assembly] export  {X.shape[0]:,} x {X.shape[1]} → {out}/")
    print(f"           smoke   known={n_smoke_known:,}  unknown={n_smoke_unknown:,}  "
          f"{dict((i, int((smoke[smoke_known]==i).sum())) for i in range(6))}")
    print(f"           malig   known_positive={n_malig_known_pos:,}  "
          f"known_negative={n_malig_known_neg:,}  unknown={n_malig_unknown:,}")
    print(f"           dose    {n_known:,} cells with known exposure duration")
    return {"gene_matrix": X, "smoke_labels": smoke, "smoke_labels_known": smoke_known,
            "malignancy_labels": malig, "malignancy_known": malig_known,
            "cell_type_ids": ctype, "exposure_dose": dose}
