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

from constants import (
    DOSE_UNKNOWN, SPECIES_HUMAN, DEFAULT_EXPERIMENT_MODE, DEFAULT_ASSAY_POLICY,
    ASSAY_POLICY_VERSION,
)


def _resolve_source_provenance(a: ad.AnnData, diagnostic_mode: bool, context: str) -> np.ndarray:
    """
    The single place merge_sources/assemble_subject_bags/export_cell_dataset
    resolve a source's row-level is_pseudo_bulk provenance. A missing
    obs["is_pseudo_bulk"] column is NEVER treated as "every row is a real
    cell" for a real (diagnostic_mode=False, the default) call — that would
    let a hand-built, corrupted, or legacy AnnData built outside
    preprocess.py::_load_all_sources bypass this module's fail-closed
    contract entirely. Only an explicit diagnostic_mode=True caller (a
    deliberately synthetic fixture) may fall back to an explicit all-False
    synthetic array. Present columns are always run through the strict
    boolean parser — a stray NaN/None/unrecognized-string value is rejected
    even in diagnostic mode.
    """
    from data.assay_policy import parse_strict_bool_array
    from data.assay_policy import MissingAssayProvenanceError

    if "is_pseudo_bulk" not in a.obs.columns:
        if diagnostic_mode:
            return np.zeros(a.n_obs, dtype=bool)
        raise MissingAssayProvenanceError(
            f"{context}: input is missing obs['is_pseudo_bulk'] — real (non-diagnostic) "
            "assembly requires explicit row-level assay provenance on every source. A "
            "missing column is never treated as 'every row is a real cell'. Pass "
            "diagnostic_mode=True only for a deliberately synthetic fixture with no "
            "provenance, or stamp obs['is_pseudo_bulk'] explicitly before calling."
        )
    return parse_strict_bool_array(a.obs["is_pseudo_bulk"].values, n_hint=a.n_obs)


def merge_sources(*adatas: ad.AnnData, scale: bool = True,
                   experiment_mode: str = DEFAULT_EXPERIMENT_MODE,
                   assay_policy: str = DEFAULT_ASSAY_POLICY,
                   diagnostic_mode: bool = False) -> ad.AnnData:
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
    obs["is_pseudo_bulk"] column at all raises MissingAssayProvenanceError
    in real (diagnostic_mode=False, the default) mode — a missing column is
    NEVER treated as "every row is a real cell". Every present value is run
    through the strict boolean parser (parse_strict_bool_array); an
    invalid/NaN/unrecognized-string value raises InvalidAssayProvenanceError
    before any gene intersection or concatenation happens. Renaming a
    source or its file cannot bypass this — only the row-level
    obs["is_pseudo_bulk"] values are ever consulted. Sources with
    incompatible assay policies (e.g. one single-cell, one pseudo-bulk) are
    never silently concatenated: each source is checked against the same
    requested `assay_policy` before any merge happens.

    diagnostic_mode=True is the ONLY way a source may omit
    obs["is_pseudo_bulk"] — it is then treated as an explicit, synthetic
    all-False array for that source (never inferred from source name, file
    path, or any other metadata). The merged output is stamped
    obs["is_pseudo_bulk"]/uns["diagnostic_mode"]=True so downstream
    consumers can tell a diagnostic merge from a real one; diagnostic
    output must never be described as a scientific/production result.
    """
    from data.species_policy import assert_single_species_or_explicit
    from data.assay_policy import assert_rows_match_policy

    species_values = [
        (a.obs["species"].iloc[0] if "species" in a.obs.columns and a.n_obs else SPECIES_HUMAN)
        for a in adatas
    ]
    assert_single_species_or_explicit(species_values, experiment_mode)

    per_source_bulk = []
    for i, a in enumerate(adatas):
        is_bulk = _resolve_source_provenance(a, diagnostic_mode, context=f"merge_sources source #{i}")
        assert_rows_match_policy(is_bulk, assay_policy, context=f"merge_sources source #{i}")
        per_source_bulk.append(is_bulk)

    genes = adatas[0].var_names
    for a in adatas[1:]:
        genes = genes.intersection(a.var_names)
    genes = list(genes)
    print(f"[assembly] merge  {len(genes):,} common genes across {len(adatas)} sources")

    clipped = []
    for i, a in enumerate(adatas):
        c = a[:, genes].copy()
        c.obs["batch"] = f"source_{i}"
        # Stamp an explicit, strictly-parsed is_pseudo_bulk column on every
        # clipped source before concat — this guarantees the merged output
        # always carries genuine per-row provenance (never re-derived from
        # a post-concat column that could silently disagree with what was
        # actually validated above).
        c.obs["is_pseudo_bulk"] = per_source_bulk[i]
        clipped.append(c)

    merged = ad.concat(clipped, axis=0, join="inner", label="source")
    merged.var_names_make_unique()
    merged.uns["diagnostic_mode"] = bool(diagnostic_mode)
    merged.uns["assay_policy"] = assay_policy
    merged.uns["assay_policy_version"] = ASSAY_POLICY_VERSION
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
    diagnostic_mode: bool = False,
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
    corrupted AnnData. Missing obs["is_pseudo_bulk"] raises
    MissingAssayProvenanceError in real (diagnostic_mode=False, the
    default) mode; only an explicit diagnostic_mode=True caller may fall
    back to a synthetic all-False array. `assay_policy` must be trainable
    (require_trainable) — only single_cell_only can back the current MIL
    architecture; bulk_only/multimodal raise their typed *NotImplemented
    errors before any bag is built, since pseudo-bulk/mixed rows have no
    per-cell malignancy/cell-type/attention semantics to bag. Every bag
    carries its own "is_pseudo_bulk", "assay_policy", "assay_policy_version",
    and "diagnostic_mode" fields so downstream consumers (folds, subsets,
    ablations) can independently verify provenance without re-deriving it.

    Parameters
    ----------
    cancer_outcomes : DataFrame[subject_id, cancer_label]
    """
    from data.assay_policy import assert_rows_match_policy, require_trainable

    is_bulk = _resolve_source_provenance(adata, diagnostic_mode, context="assemble_subject_bags")
    # A bag is a per-cell MIL structure with no meaning under bulk_only/
    # multimodal, regardless of diagnostic_mode — a diagnostic caller still
    # cannot build bags for a policy this architecture cannot represent.
    require_trainable(assay_policy)
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
            "is_pseudo_bulk":       is_bulk[mask],
            "assay_policy":         assay_policy,
            "assay_policy_version": ASSAY_POLICY_VERSION,
            "diagnostic_mode":      bool(diagnostic_mode),
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
    diagnostic_mode: bool = False,
) -> dict:
    """
    Save numpy arrays for CellLevelDataset (Phase 1 training).

    Assay safety (see data/assay_policy.py): the final gate before rows
    become the numpy files CellLevelDataset.from_dir() reads — a pseudo-
    bulk row must never reach a "cell-level" training array. Validation
    (missing/invalid provenance, policy mismatch, trainability) happens
    BEFORE the output directory is created or any file is written — a
    failed call leaves no partial output on disk. Missing
    obs["is_pseudo_bulk"] raises MissingAssayProvenanceError in real
    (diagnostic_mode=False, the default) mode; only an explicit
    diagnostic_mode=True caller may fall back to a synthetic all-False
    array (written through the same is_pseudo_bulk column so
    CellLevelDataset.from_dir() sees it explicitly either way). A
    dataset_metadata.json file records assay_policy, assay_policy_version,
    data_modality, and diagnostic_status alongside the per-row
    cell_metadata.csv column, so CellLevelDataset.from_dir() can cross-
    check dataset-level and row-level provenance agree.
    """
    from data.assay_policy import assert_rows_match_policy, require_trainable

    is_bulk = _resolve_source_provenance(adata, diagnostic_mode, context="export_cell_dataset")
    require_trainable(assay_policy)
    assert_rows_match_policy(is_bulk, assay_policy, context="export_cell_dataset")

    out = Path(out_dir)

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

    # All validation above passes — only now do we touch the filesystem.
    out.mkdir(parents=True, exist_ok=True)

    obs_out = adata.obs.copy()
    obs_out["is_pseudo_bulk"] = is_bulk

    np.save(out / "gene_matrix.npy",       X)
    np.save(out / "smoke_labels.npy",      smoke)
    np.save(out / "smoke_labels_known.npy", smoke_known)
    np.save(out / "malignancy_labels.npy", malig)
    np.save(out / "malignancy_known.npy",  malig_known)
    np.save(out / "cell_type_ids.npy",     ctype)
    np.save(out / "exposure_dose.npy",     dose)
    obs_out.to_csv(out / "cell_metadata.csv")
    adata.var.to_csv(out / "gene_list.csv")

    import json as _json
    import hashlib as _hashlib
    dataset_fingerprint = _hashlib.sha256(
        _json.dumps({
            "n_obs": int(X.shape[0]), "n_genes": int(X.shape[1]),
            "assay_policy": assay_policy, "diagnostic_mode": bool(diagnostic_mode),
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dataset_metadata = {
        "assay_policy": assay_policy,
        "assay_policy_version": ASSAY_POLICY_VERSION,
        "data_modality": "bulk" if assay_policy == "bulk_only" else "single_cell",
        "diagnostic_mode": bool(diagnostic_mode),
        "dataset_fingerprint": dataset_fingerprint,
    }
    with open(out / "dataset_metadata.json", "w") as f:
        _json.dump(dataset_metadata, f, indent=2)

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
