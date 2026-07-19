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
    gene_module_permutation_null,
    label_permutation_null_record,
    matched_size_random_module_scores,
    module_ablation_scores,
    module_ranking_stability,
    within_gene_expression_permutation_null,
)
from .domain_shift import (
    composition_summary,
    distribution_shift_report,
    gene_space_compatibility,
    module_coverage_compatibility,
    source_predictability_diagnostic,
)
from .features import build_cancer_subject_features, build_smoke_subject_summary_features
from .pathway_hierarchical_adapter import MODEL_NAME as PATHWAY_MODEL_NAME
from .uncertainty import (
    apply_abstention_threshold,
    binary_predictive_uncertainty,
    mc_dropout_uncertainty_report,
    multiclass_predictive_uncertainty,
    select_abstention_threshold_from_development,
)

MIN_GENES_PER_MODULE = 2

# Bounds the cost of the biological-stability forward-pass diagnostics (each
# one runs several extra forward passes over the sampled bags) — a
# development-only DIAGNOSTIC never needs the full held-out cohort to
# characterize model sensitivity, and this keeps the per-source cost
# bounded regardless of how many held-out subjects a real source has.
MAX_BIOLOGICAL_STABILITY_BAGS = 25


def _cell_type_counts(bag: dict) -> Dict[int, int]:
    types, counts = np.unique(np.asarray(bag["cell_type_ids"]), return_counts=True)
    return {int(t): int(c) for t, c in zip(types, counts)}


def _module_gene_counts(modules) -> Dict[str, int]:
    return {name: int(modules.membership_mask[i].sum().item()) for i, name in enumerate(modules.module_names)}


def _gene_and_module_coverage(
    required_gene_list: Optional[Sequence[str]], modules, held_out_raw_gene_list: Optional[Sequence[str]] = None,
) -> Dict:
    """
    required_gene_list: the frozen development artifact's own ordered gene
    list — never affected by anything computed here.
    held_out_raw_gene_list: the held-out source's RAW, pre-alignment gene
    identities captured before the frozen transform, if the caller has
    them. This pipeline currently unifies every source onto one shared
    HVG-selected gene space at ingestion time, before any per-source raw
    panel would even be distinguishable (see ARCHITECTURE.md §16.10) — no
    current caller has raw per-source gene identities to supply, so this
    defaults to None and gene_space_compatibility reports a structured
    not_evaluable status rather than a fabricated 100%-compatible result.
    """
    coverage: Dict = {}
    if required_gene_list is not None:
        coverage["gene_space_compatibility"] = gene_space_compatibility(required_gene_list, held_out_raw_gene_list)
    if modules is not None:
        coverage["module_coverage"] = module_coverage_compatibility(_module_gene_counts(modules), MIN_GENES_PER_MODULE)
    return coverage


def cancer_domain_shift_report(
    dev_bags: Sequence[dict], held_out_bags: Sequence[dict], num_cell_types: int,
    subject_to_source: Dict[str, str], seed: int = 0,
    required_gene_list: Optional[Sequence[str]] = None, modules=None,
    held_out_raw_gene_list: Optional[Sequence[str]] = None,
) -> Dict:
    """Label-free: built entirely from development-fitted subject-summary
    features (dev_bags/held_out_bags were both produced by TRANSFORMING,
    never refitting, the frozen development preprocessing artifact — the
    caller guarantees this, see source_held_out.py). required_gene_list/
    modules, when supplied, add gene-space and module-coverage compatibility
    fields — both computed only from the FROZEN development artifact's own
    gene list/module structure, never from held-out expression values.
    held_out_raw_gene_list, when the caller actually has the held-out
    source's raw pre-alignment gene identities, enables a REAL coverage
    measurement instead of the default not_evaluable status (see
    domain_shift.gene_space_compatibility)."""
    if not dev_bags or not held_out_bags:
        return {"status": "not_evaluable", "reason": "empty development or held-out bag set"}
    Xdev, _, dev_ids, _ = build_cancer_subject_features(dev_bags, num_cell_types)
    Xho, _, ho_ids, _ = build_cancer_subject_features(held_out_bags, num_cell_types)
    report = distribution_shift_report(Xdev, Xho)
    report.update(_gene_and_module_coverage(required_gene_list, modules, held_out_raw_gene_list))
    missing = [str(s) for s in dev_ids if str(s) not in subject_to_source]
    if missing:
        from .domain_losses import MissingSourceProvenanceError
        raise MissingSourceProvenanceError(
            f"{len(missing)} development subject(s) have no resolved dataset_source (e.g. "
            f"{missing[:5]}) — source_predictability_diagnostic requires a real source identity "
            "for every subject, never a placeholder default."
        )
    dev_sources = [subject_to_source[str(s)] for s in dev_ids]
    report["source_predictability"] = source_predictability_diagnostic(Xdev, dev_sources, dev_ids, seed=seed)
    cells_per_subject = {str(b["subject_id"]): len(b["gene_matrix"]) for b in held_out_bags}
    ctype_counts = {str(b["subject_id"]): _cell_type_counts(b) for b in held_out_bags}
    report["held_out_composition"] = composition_summary(cells_per_subject, ctype_counts)
    return report


def smoke_domain_shift_report(
    dev_cell_dataset, held_out_cell_dataset, num_cell_types: int, num_classes: int,
    subject_to_source: Dict[str, str], seed: int = 0,
    required_gene_list: Optional[Sequence[str]] = None, modules=None,
    held_out_raw_gene_list: Optional[Sequence[str]] = None,
) -> Dict:
    """Task A analogue of cancer_domain_shift_report — label-free, built
    from subject-summary features (build_smoke_subject_summary_features)
    computed from cell datasets that were TRANSFORMED (never refit) through
    the frozen development-only preprocessing artifact. See
    cancer_domain_shift_report's docstring for held_out_raw_gene_list."""
    if len(dev_cell_dataset) == 0 or len(held_out_cell_dataset) == 0:
        return {"status": "not_evaluable", "reason": "empty development or held-out cell dataset"}
    Xdev, _, dev_ids, _ = build_smoke_subject_summary_features(dev_cell_dataset, num_cell_types, num_classes)
    Xho, _, ho_ids, _ = build_smoke_subject_summary_features(held_out_cell_dataset, num_cell_types, num_classes)
    report = distribution_shift_report(Xdev, Xho)
    report.update(_gene_and_module_coverage(required_gene_list, modules, held_out_raw_gene_list))
    missing = [str(s) for s in dev_ids if str(s) not in subject_to_source]
    if missing:
        from .domain_losses import MissingSourceProvenanceError
        raise MissingSourceProvenanceError(
            f"{len(missing)} development subject(s) have no resolved dataset_source (e.g. "
            f"{missing[:5]}) — source_predictability_diagnostic requires a real source identity "
            "for every subject, never a placeholder default."
        )
    dev_sources = [subject_to_source[str(s)] for s in dev_ids]
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


def smoke_uncertainty_report(
    dev_proba: Optional[np.ndarray], dev_labels: Optional[np.ndarray],
    held_out_proba: Optional[np.ndarray], held_out_labels: Optional[np.ndarray],
    num_classes: int, target_coverage: float = 0.8, oof_coverage: Optional[Dict] = None,
) -> Dict:
    """
    Multi-class (Task A) analogue of cancer_uncertainty_report: predictive
    entropy + maximum class probability over the softmax output, a
    development-only abstention threshold, and retained-subset macro-F1 as a
    SECONDARY diagnostic — the full-population macro-F1 already reported in
    the main metrics block remains primary. dev_proba/dev_labels here MUST
    be genuine out-of-fold predictions (see
    source_held_out.py::_smoke_candidate_dev_score's oof_by_subject — each
    dev subject predicted only by the one grouped-CV fold it was held out
    of, never a fold it also trained on) — never the final dev-pool-fitted
    model's in-sample predictions on its own training pool, which would
    make the selected threshold optimistic about development calibration.
    held-out-source labels never influence the selected threshold
    regardless of which probabilities are supplied.
    """
    if dev_proba is None or held_out_proba is None or len(dev_proba) == 0 or len(held_out_proba) == 0:
        return {"status": "not_applicable", "reason": "candidate does not produce class probabilities, or "
                                                        "no development/held-out subject had a verified label"}
    from .metrics import full_smoke_metrics_report

    dev_unc = multiclass_predictive_uncertainty(dev_proba)
    dev_pred = np.asarray(dev_proba).argmax(axis=1)
    dev_correct = dev_pred == np.asarray(dev_labels)
    selection = select_abstention_threshold_from_development(dev_unc["entropy"], dev_correct, target_coverage)

    held_out_unc = multiclass_predictive_uncertainty(held_out_proba)
    held_out_pred = np.asarray(held_out_proba).argmax(axis=1)
    held_out_labels = np.asarray(held_out_labels)

    result: Dict = {
        "development_threshold_selection": selection,
        "note": "development probabilities are genuine out-of-fold predictions from the dev-only "
                "nested grouped-CV sweep that selected this candidate — never the final dev-pool-"
                "fitted model's in-sample predictions; held-out-source labels are never read by the "
                "threshold-selection step regardless.",
        "oof_coverage": oof_coverage or {},
    }
    if selection.get("status") == "selected":
        threshold = selection["uncertainty_threshold"]
        retained = held_out_unc["entropy"] <= threshold
        retained_macro_f1 = None
        if retained.any():
            retained_report = full_smoke_metrics_report(held_out_labels[retained], held_out_pred[retained], num_classes)
            retained_macro_f1 = retained_report["macro_f1"]
        result["abstention_at_held_out"] = {
            "threshold": threshold, "n_total": int(len(held_out_labels)), "n_retained": int(retained.sum()),
            "coverage": float(retained.mean()) if len(retained) else None,
            "macro_f1_at_coverage": retained_macro_f1,
        }
    else:
        result["abstention_at_held_out"] = {"status": "not_applicable", "reason": selection.get("reason")}
    return result


def cancer_biological_stability_report(
    context, fitted, dev_bags: Sequence[dict], held_out_bags: Sequence[dict],
    outcomes_by_subject: Dict[str, int], num_cell_types: int, min_cells_per_subject: int, seed: int = 0,
    extra_seeds: Sequence[int] = (),
) -> Dict:
    """
    Synthetic-versus-real module scope is stamped explicitly
    (is_synthetic_modules) — see biological_stability.py's module docstring.
    Every result here characterizes the FITTED model's sensitivity to
    module/cell-type structure, never a biological or causal claim, and a
    synthetic-module result is always labeled software_diagnostic_only.

    extra_seeds, when non-empty, triggers GENUINE independent refits of the
    same candidate/hyperparameters on the SAME development pool at each
    extra seed (never a repeated deterministic call against the one
    already-fitted model) — module_ablation_scores from each independent
    fit are combined via module_ranking_stability into a cross_run_stability
    field. With no extra seeds (the default — refitting is expensive and
    opt-in), cross_run_stability is reported as insufficient_evidence with
    an explicit reason, never fabricated from repeated calls to the single
    fitted model.
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
    if not is_synthetic:
        # Real-module analysis requires a complete provenance contract
        # (resource name/version, organism, gene-identifier namespace,
        # mapping policy, source URL/citation, license, local artifact
        # checksum, ordered module fingerprint, mapped/unmapped gene
        # counts, minimum-coverage policy) bound into the model/report
        # identities — GeneModuleCollection does not yet carry that
        # contract (only source_name/source_version/a content fingerprint;
        # see ARCHITECTURE.md §16.10), so real-module analysis is disabled
        # here rather than emitting scope="real_module_sensitivity_analysis"
        # without it. This module source is real (not the synthetic
        # diagnostic scheme), but no provenance contract exists yet to
        # validate it against.
        return {
            "status": "unsupported",
            "reason": "real-module biological-stability analysis requires a complete provenance "
                      "contract (organism, namespace, mapping policy, license, checksum, ordered "
                      "module fingerprint, coverage policy) that GeneModuleCollection does not yet "
                      "implement — disabled rather than reported without it.",
            "is_synthetic_modules": False,
            "module_source_name": modules.source_name,
            "scope": "unsupported_real_module_analysis",
        }
    sample = list(held_out_bags[:MAX_BIOLOGICAL_STABILITY_BAGS])

    invariance = cell_order_permutation_invariance_check(adapter, sample, seed=seed)
    cell_type_null = cell_type_label_permutation_check(adapter, sample, seed=seed)
    ablation = module_ablation_scores(adapter, sample, target="cancer")
    random_module_null = matched_size_random_module_scores(adapter, sample, modules, target="cancer", seed=seed)
    gene_module_permutation = gene_module_permutation_null(adapter, sample, target="cancer", seed=seed)
    gene_module_permutation["ablation_rank_correlation_vs_real"] = label_permutation_null_record(
        ablation, gene_module_permutation["ablation_scores_under_permuted_module_assignment"],
    )["rank_correlation_vs_label_permuted_model"]
    within_gene_permutation = within_gene_expression_permutation_null(adapter, sample, target="cancer", seed=seed)
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

    cross_run_stability: Dict = {
        "status": "insufficient_evidence",
        "reason": "extra_seeds was empty — cross-run stability requires >= 2 GENUINE independent "
                  "refits (never repeated deterministic calls to the one already-fitted model); pass "
                  "extra_seeds to opt into the extra refit cost.",
    }
    if extra_seeds:
        from .final_evaluation import fit_final_candidate_on_dev_pool

        run_scores = [ablation]
        run_seeds = [seed]
        for extra_seed in extra_seeds:
            if extra_seed == seed:
                continue
            extra_fitted = fit_final_candidate_on_dev_pool(
                context, fitted.candidate_name, dev_subject_ids, outcomes_by_subject, fitted.selected_params,
                num_cell_types, min_cells_per_subject, len(fitted.preprocessing_artifact.gene_list),
                pooling=fitted.pooling, device=adapter.device, seed=extra_seed,
            )
            run_scores.append(module_ablation_scores(extra_fitted.predictor, sample, target="cancer"))
            run_seeds.append(extra_seed)
        if len(run_scores) >= 2:
            cross_run_stability = module_ranking_stability(run_scores)
            cross_run_stability["seeds"] = run_seeds
        else:
            cross_run_stability = {
                "status": "insufficient_evidence",
                "reason": "extra_seeds contained no seed distinct from the primary fit's own seed.",
            }

    return {
        # is_synthetic is always True here — the not-synthetic case
        # returns early above with scope="unsupported_real_module_analysis"
        # rather than ever reaching this point.
        "is_synthetic_modules": is_synthetic,
        "module_source_name": modules.source_name,
        "scope": "software_diagnostic_only",
        "cell_order_permutation_invariance": invariance,
        "cell_type_label_permutation_null": cell_type_null,
        "module_ablation_scores": ablation,
        "matched_size_random_module_null": random_module_null,
        "gene_module_permutation_null": gene_module_permutation,
        "within_gene_expression_permutation_null": within_gene_permutation,
        "attention_vs_abundance": attn_vs_abund,
        "label_permutation_null": label_permutation_result,
        "cross_run_stability": cross_run_stability,
        "note": (
            "every result above characterizes this fitted model's sensitivity to gene-module and "
            "cell-type structure — a synthetic-module result is a software diagnostic, not "
            "biological-plausibility evidence; a real-module result is a sensitivity finding, never "
            "a causal or mechanistic claim."
        ),
    }
