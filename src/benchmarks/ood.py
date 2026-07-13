"""
benchmarks/ood.py — leave-one-dataset-source-out validation for Task A.

Trains on every eligible source except one, evaluates only on the held-out
source's subjects. Preprocessing (gene scaling + HVG selection) is refit
using ONLY the remaining-source training subjects, from
context.normalized_adata_for_refit — reusing the outer/CV-fold artifact
here would mean the held-out source's own cells influenced the scaling/HVG
selection it's then evaluated against, the same leakage class
fold_preprocessing.py fixes for grouped CV.

A source is skipped (NOT_COMPARABLE) rather than silently included when its
label-semantics/species metadata is missing or mismatched — unknown
metadata defaults to NOT_COMPARABLE, never to "assumed compatible" (section
4's explicit requirement). Species/semantics metadata is config-driven here
(dataset_source is just a GEO accession string with no built-in annotation),
not auto-inferred.
"""

from typing import Dict, Iterable, List, Optional

import numpy as np

from .baselines import SMOKE_BASELINES
from .features import build_smoke_subject_summary_features
from .fold_preprocessing import (
    artifact_fingerprint,
    build_fold_cell_dataset,
    refit_artifact_for_fold,
    require_normalized_adata,
)
from .metrics import full_smoke_metrics_report

MIN_SUBJECTS_PER_SOURCE = 3


def run_leave_one_source_out(
    context, model_names: Iterable[str], device: str = "cpu",
    incompatible_sources: Optional[List[str]] = None,
    species_by_source: Optional[Dict[str, str]] = None,
) -> Dict:
    normalized_adata = require_normalized_adata(context)
    incompatible_sources = set(incompatible_sources or [])
    species_by_source = species_by_source or {}
    num_classes = context.num_smoke_classes
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = context.preprocessing_artifact.n_hvgs

    pool_subjects = sorted(set(context.subjects_for("train")) | set(context.subjects_for("val")))
    obs = normalized_adata.obs
    subj_series = obs["subject_id"].astype(str)
    pool_mask = subj_series.isin(pool_subjects).values
    pool_source = obs["source"].astype(str).values[pool_mask]
    pool_subject_arr = subj_series.values[pool_mask]

    sources = sorted(set(pool_source.tolist()))
    default_species = species_by_source.get(sources[0]) if sources else None
    results: Dict[str, Dict] = {}

    for src in sources:
        if src == "unknown":
            results[src] = {"status": "NOT_COMPARABLE", "reason": "dataset_source not recorded for these cells"}
            continue
        if src in incompatible_sources:
            results[src] = {"status": "NOT_COMPARABLE", "reason": "explicitly listed as label-semantics incompatible"}
            continue
        if src not in species_by_source:
            results[src] = {
                "status": "NOT_COMPARABLE",
                "reason": f"no species/label-semantics metadata declared for source {src!r} — "
                          "unknown metadata defaults to NOT_COMPARABLE, never to assumed-compatible",
            }
            continue
        src_species = species_by_source[src]
        if default_species is not None and src_species != default_species:
            results[src] = {"status": "NOT_COMPARABLE",
                             "reason": f"species mismatch ({src_species} vs pool default {default_species}) — "
                                       "species are kept separate by default"}
            continue

        held_out_subjects = sorted(set(pool_subject_arr[pool_source == src].tolist()))
        if len(held_out_subjects) < MIN_SUBJECTS_PER_SOURCE:
            results[src] = {"status": "NOT_EVALUABLE",
                             "reason": f"only {len(held_out_subjects)} subjects (< {MIN_SUBJECTS_PER_SOURCE})"}
            continue

        train_subjects = sorted(set(pool_subject_arr.tolist()) - set(held_out_subjects))
        if not train_subjects:
            results[src] = {"status": "NOT_EVALUABLE", "reason": "no subjects left outside the held-out source"}
            continue

        try:
            artifact = refit_artifact_for_fold(normalized_adata, train_subjects, n_hvgs)
            train_ds = build_fold_cell_dataset(normalized_adata, artifact, train_subjects)
            held_out_ds = build_fold_cell_dataset(normalized_adata, artifact, held_out_subjects)
        except ValueError as e:
            results[src] = {"status": "NOT_EVALUABLE",
                             "reason": f"held-out source's genes could not be transformed with the "
                                       f"training-source-only artifact: {e}"}
            continue

        fp = artifact_fingerprint(artifact)
        Xtr, ytr, _, _ = build_smoke_subject_summary_features(train_ds, num_cell_types, num_classes)
        Xte, yte, _, _ = build_smoke_subject_summary_features(held_out_ds, num_cell_types, num_classes)

        per_model = {}
        for name in model_names:
            model = SMOKE_BASELINES[name]()
            model.fit(Xtr, ytr, seed=42)
            preds = model.predict(Xte)
            report = full_smoke_metrics_report(yte, preds, num_classes)
            per_model[name] = {
                "subject_weighted_macro_f1": report["macro_f1"],
                "classes_absent_from_held_out_source": report["classes_absent_from_targets"],
            }

        results[src] = {
            "status": "EVALUATED", "n_held_out_subjects": len(held_out_subjects),
            "n_train_subjects": len(train_subjects), "preprocessing_fingerprint": fp,
            "per_model": per_model,
        }

    return results
