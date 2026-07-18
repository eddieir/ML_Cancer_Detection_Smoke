"""
benchmarks/source_held_out_diagnostics.py — post-fit diagnostic wiring for
the source-held-out protocol's domain_shift/uncertainty/biological_stability
report sections.

Every function here operates on ALREADY-FITTED state (a frozen candidate
plus development/held-out bags built from that frozen preprocessing
artifact) and never refits, reselects a hyperparameter, or lets a held-out
LABEL influence a decision — held-out labels are read only by the
after-the-fact reporting calls (apply_abstention_threshold,
distribution_shift_report's feature-only inputs), never by any selection
function. See domain_shift.py/uncertainty.py/biological_stability.py's own
module docstrings for the underlying guarantees this module composes.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from .biological_stability import (
    attention_vs_abundance,
    cell_order_permutation_invariance_check,
    cell_type_attention_by_subject,
    cell_type_label_permutation_check,
    label_permutation_null_record,
    matched_size_random_module_scores,
    module_ablation_scores,
)
from .domain_shift import composition_summary, distribution_shift_report, source_predictability_diagnostic
from .features import build_cancer_subject_features, build_smoke_subject_summary_features
from .pathway_hierarchical_adapter import MODEL_NAME as PATHWAY_MODEL_NAME
from .uncertainty import (
    apply_abstention_threshold,
    binary_predictive_uncertainty,
    mc_dropout_uncertainty_report,
    select_abstention_threshold_from_development,
)

# Bounds the cost of the biological-stability forward-pass diagnostics (each
# one runs several extra forward passes over the sampled bags) — a
# development-only DIAGNOSTIC never needs the full held-out cohort to
# characterize model sensitivity, and this keeps the per-source cost
# bounded regardless of how many held-out subjects a real source has.
MAX_BIOLOGICAL_STABILITY_BAGS = 25


def _cell_type_counts(bag: dict) -> Dict[int, int]:
    types, counts = np.unique(np.asarray(bag["cell_type_ids"]), return_counts=True)
    return {int(t): int(c) for t, c in zip(types, counts)}


def cancer_domain_shift_report(
    dev_bags: Sequence[dict], held_out_bags: Sequence[dict], num_cell_types: int,
    subject_to_source: Dict[str, str], seed: int = 0,
) -> Dict:
    """Label-free: built entirely from development-fitted subject-summary
    features (dev_bags/held_out_bags were both produced by TRANSFORMING,
    never refitting, the frozen development preprocessing artifact — the
    caller guarantees this, see source_held_out.py)."""
    if not dev_bags or not held_out_bags:
        return {"status": "not_evaluable", "reason": "empty development or held-out bag set"}
    Xdev, _, dev_ids, _ = build_cancer_subject_features(dev_bags, num_cell_types)
    Xho, _, ho_ids, _ = build_cancer_subject_features(held_out_bags, num_cell_types)
    report = distribution_shift_report(Xdev, Xho)
    dev_sources = [subject_to_source.get(str(s), "unknown") for s in dev_ids]
    report["source_predictability"] = source_predictability_diagnostic(Xdev, dev_sources, dev_ids, seed=seed)
    cells_per_subject = {str(b["subject_id"]): len(b["gene_matrix"]) for b in held_out_bags}
    ctype_counts = {str(b["subject_id"]): _cell_type_counts(b) for b in held_out_bags}
    report["held_out_composition"] = composition_summary(cells_per_subject, ctype_counts)
    return report


def smoke_domain_shift_report(
    dev_cell_dataset, held_out_cell_dataset, num_cell_types: int, num_classes: int,
    subject_to_source: Dict[str, str], seed: int = 0,
) -> Dict:
    """Task A analogue of cancer_domain_shift_report — label-free, built
    from subject-summary features (build_smoke_subject_summary_features)
    computed from cell datasets that were TRANSFORMED (never refit) through
    the frozen development-only preprocessing artifact."""
    if len(dev_cell_dataset) == 0 or len(held_out_cell_dataset) == 0:
        return {"status": "not_evaluable", "reason": "empty development or held-out cell dataset"}
    Xdev, _, dev_ids, _ = build_smoke_subject_summary_features(dev_cell_dataset, num_cell_types, num_classes)
    Xho, _, ho_ids, _ = build_smoke_subject_summary_features(held_out_cell_dataset, num_cell_types, num_classes)
    report = distribution_shift_report(Xdev, Xho)
    dev_sources = [subject_to_source.get(str(s), "unknown") for s in dev_ids]
    report["source_predictability"] = source_predictability_diagnostic(Xdev, dev_sources, dev_ids, seed=seed)
    return report


def cancer_uncertainty_report(
    y_oof: np.ndarray, prob_oof: np.ndarray, y_held_out: np.ndarray, held_out_proba_raw: np.ndarray,
    fitted=None, held_out_bags: Optional[Sequence[dict]] = None, target_coverage: float = 0.8,
) -> Dict:
    """dev_* arguments are OOF (out-of-fold) development predictions — the
    abstention threshold is selected from these only; held_out_* arguments
    are read only to report the ALREADY-SELECTED threshold's effect."""
    dev_unc = binary_predictive_uncertainty(prob_oof)
    dev_correct = (np.round(prob_oof) == y_oof)
    selection = select_abstention_threshold_from_development(dev_unc["entropy"], dev_correct, target_coverage)
    result: Dict = {"development_threshold_selection": selection}
    if selection.get("status") == "selected":
        held_out_unc = binary_predictive_uncertainty(held_out_proba_raw)
        held_out_correct = (np.round(held_out_proba_raw) == y_held_out)
        result["abstention_at_held_out"] = apply_abstention_threshold(
            held_out_unc["entropy"], held_out_correct, selection["uncertainty_threshold"],
        )
    else:
        result["abstention_at_held_out"] = {"status": "not_applicable", "reason": selection.get("reason")}
    if fitted is not None and fitted.kind == "mil" and fitted.candidate_name == PATHWAY_MODEL_NAME and held_out_bags:
        sample = held_out_bags[:MAX_BIOLOGICAL_STABILITY_BAGS]
        result["mc_dropout"] = mc_dropout_uncertainty_report(fitted.predictor, sample)
    else:
        result["mc_dropout"] = {"status": "not_applicable",
                                 "reason": "MC-dropout dispersion is defined only for pathway_hierarchical_mil"}
    return result


def cancer_biological_stability_report(
    context, fitted, dev_bags: Sequence[dict], held_out_bags: Sequence[dict],
    outcomes_by_subject: Dict[str, int], num_cell_types: int, min_cells_per_subject: int, seed: int = 0,
) -> Dict:
    """
    Synthetic-versus-real module scope is stamped explicitly
    (is_synthetic_modules) — see biological_stability.py's module docstring.
    Every result here characterizes the FITTED model's sensitivity to
    module/cell-type structure, never a biological or causal claim, and a
    synthetic-module result is always labeled software_diagnostic_only.
    """
    if fitted.kind != "mil" or fitted.candidate_name != PATHWAY_MODEL_NAME:
        return {"status": "not_applicable",
                "reason": "biological-stability diagnostics apply only to pathway_hierarchical_mil"}
    if not held_out_bags:
        return {"status": "not_evaluable", "reason": "no held-out bags available"}
    from pathway_hierarchical_mil import is_synthetic_module_source

    adapter = fitted.predictor
    modules = adapter.modules
    is_synthetic = is_synthetic_module_source(modules.source_name)
    sample = list(held_out_bags[:MAX_BIOLOGICAL_STABILITY_BAGS])

    invariance = cell_order_permutation_invariance_check(adapter, sample, seed=seed)
    cell_type_null = cell_type_label_permutation_check(adapter, sample, seed=seed)
    ablation = module_ablation_scores(adapter, sample, target="cancer")
    random_module_null = matched_size_random_module_scores(adapter, sample, modules, target="cancer", seed=seed)
    attention_by_subject = cell_type_attention_by_subject(adapter, sample)
    abundance_by_subject = {str(b["subject_id"]): _cell_type_counts(b) for b in sample}
    attn_vs_abund = attention_vs_abundance(attention_by_subject, abundance_by_subject, seed=seed)

    label_permutation_result: Dict = {"status": "not_evaluable",
                                       "reason": "fewer than 2 classes among development subjects to permute"}
    dev_subject_ids = sorted({str(b["subject_id"]) for b in dev_bags if str(b["subject_id"]) in outcomes_by_subject})
    y_dev = [outcomes_by_subject[s] for s in dev_subject_ids]
    if len(set(y_dev)) >= 2 and len(dev_subject_ids) >= 2:
        from .final_evaluation import fit_final_candidate_on_dev_pool

        rng = np.random.RandomState(seed)
        permuted_y = rng.permutation(y_dev)
        permuted_outcomes = {s: int(v) for s, v in zip(dev_subject_ids, permuted_y)}
        # A genuinely SEPARATELY fitted model on label-permuted development
        # data (never a relabeled copy of the real ranking) — same
        # candidate, same hyperparameters, same preprocessing artifact
        # fingerprint, only the development labels are permuted.
        permuted_fitted = fit_final_candidate_on_dev_pool(
            context, fitted.candidate_name, dev_subject_ids, permuted_outcomes, fitted.selected_params,
            num_cell_types, min_cells_per_subject, len(fitted.preprocessing_artifact.gene_list),
            pooling=fitted.pooling, device=adapter.device, seed=seed,
        )
        permuted_ranking = module_ablation_scores(permuted_fitted.predictor, sample, target="cancer")
        label_permutation_result = label_permutation_null_record(ablation, permuted_ranking)

    return {
        "is_synthetic_modules": is_synthetic,
        "module_source_name": modules.source_name,
        "scope": "software_diagnostic_only" if is_synthetic else "real_module_sensitivity_analysis",
        "cell_order_permutation_invariance": invariance,
        "cell_type_label_permutation_null": cell_type_null,
        "module_ablation_scores": ablation,
        "matched_size_random_module_null": random_module_null,
        "attention_vs_abundance": attn_vs_abund,
        "label_permutation_null": label_permutation_result,
        "note": (
            "every result above characterizes this fitted model's sensitivity to gene-module and "
            "cell-type structure — a synthetic-module result is a software diagnostic, not "
            "biological-plausibility evidence; a real-module result is a sensitivity finding, never "
            "a causal or mechanistic claim."
        ),
    }
