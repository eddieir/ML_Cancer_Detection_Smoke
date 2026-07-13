"""
benchmarks/ood.py — leave-one-dataset-source-out validation for Task A.

Trains on every eligible source except one, evaluates only on the held-out
source's subjects. A source is skipped (NOT_COMPARABLE) rather than silently
included when its label semantics don't match the rest of the pool (e.g. a
mouse source mixed into a human-only comparison) — species/semantics
compatibility is config-driven here (dataset_source is just a GEO accession
string with no built-in species/semantics annotation), not auto-inferred, so
callers must list incompatible sources explicitly via `incompatible_sources`
or `species_by_source` in the benchmarks config. This is a documented
simplification, not a silent gap.
"""

from typing import Dict, Iterable, List, Optional

import numpy as np

from .cross_validation import concat_cell_datasets
from .features import build_smoke_subject_summary_features
from .baselines import SMOKE_BASELINES
from .metrics import full_smoke_metrics_report

MIN_SUBJECTS_PER_SOURCE = 3


def run_leave_one_source_out(
    context, model_names: Iterable[str], device: str = "cpu",
    incompatible_sources: Optional[List[str]] = None,
    species_by_source: Optional[Dict[str, str]] = None,
) -> Dict:
    incompatible_sources = set(incompatible_sources or [])
    species_by_source = species_by_source or {}
    pool = concat_cell_datasets(context.train_cell_dataset, context.val_cell_dataset)
    num_classes = context.num_smoke_classes
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)

    sources = sorted(set(pool.dataset_source.tolist()))
    default_species = species_by_source.get(sources[0]) if sources else None
    results: Dict[str, Dict] = {}

    for src in sources:
        if src == "unknown":
            results[src] = {"status": "NOT_COMPARABLE", "reason": "dataset_source not recorded for these cells"}
            continue
        if src in incompatible_sources:
            results[src] = {"status": "NOT_COMPARABLE", "reason": "explicitly listed as label-semantics incompatible"}
            continue
        src_species = species_by_source.get(src)
        if default_species is not None and src_species is not None and src_species != default_species:
            results[src] = {"status": "NOT_COMPARABLE",
                             "reason": f"species mismatch ({src_species} vs pool default {default_species}) — "
                                       "species are kept separate by default"}
            continue

        held_out_mask = pool.dataset_source == src
        held_out_subjects = set(pool.subject_ids[held_out_mask].tolist())
        if len(held_out_subjects) < MIN_SUBJECTS_PER_SOURCE:
            results[src] = {"status": "NOT_EVALUABLE",
                             "reason": f"only {len(held_out_subjects)} subjects (< {MIN_SUBJECTS_PER_SOURCE})"}
            continue

        train_mask = ~np.isin(pool.subject_ids, list(held_out_subjects))
        train_subjects = sorted(set(pool.subject_ids[train_mask].tolist()))
        if not train_subjects:
            results[src] = {"status": "NOT_EVALUABLE", "reason": "no subjects left outside the held-out source"}
            continue

        train_pool = pool.subset_by_subjects(train_subjects)
        held_out_pool = pool.subset_by_subjects(sorted(held_out_subjects))

        Xtr, ytr, _, _ = build_smoke_subject_summary_features(train_pool, num_cell_types, num_classes)
        Xte, yte, _, _ = build_smoke_subject_summary_features(held_out_pool, num_cell_types, num_classes)

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
            "n_train_subjects": len(train_subjects), "per_model": per_model,
        }

    return results
