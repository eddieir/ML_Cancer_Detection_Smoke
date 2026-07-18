"""
benchmarks/domain_robustness_ablation.py — development-only comparison of
domain-robustness training strategies for pathway_hierarchical_mil, run
under the identical source-held-out protocol (source_held_out.py) so every
variant is compared on the same held-out sources, the same development
subjects, the same preprocessing/module artifacts, and the same seeds.

Never touches the frozen test split or its guard — see source_held_out.py's
module docstring for why that is true structurally, not just by convention.
"""

from typing import Dict, List, Optional, Sequence

from .robustness_report import aggregate_source_reports
from .source_held_out import run_cancer_source_held_out, run_smoke_source_held_out

# Required comparison strategies (Step 27). "domain_adversarial" is included
# only when the caller explicitly opts in — see run_domain_robustness_ablation's
# include_adversarial parameter — since it is the strategy this Phase's spec
# permits leaving out if it cannot be completed safely; here it CAN run (the
# gradient-reversal head is fully implemented and tested), so it is included
# by default, but remains easy to exclude for a faster/CI-constrained pass.
ABLATION_VARIANTS = {
    "erm": {"strategy": "erm"},
    "source_balanced": {"strategy": "source_balanced", "source_balancing": {"enabled": True}},
    "coral": {"strategy": "coral", "coral": {"enabled": True, "weight": 0.1}},
    "mmd": {"strategy": "mmd", "mmd": {"enabled": True, "weight": 0.1}},
    "domain_adversarial": {
        "strategy": "domain_adversarial",
        "adversarial": {"enabled": True, "weight": 0.1, "gradient_reversal_lambda": 1.0, "warmup_epochs": 0},
    },
}


def run_domain_robustness_ablation(
    context, task: str, model_names: Sequence[str], device: str = "cpu",
    seed: int = 42, include_adversarial: bool = True,
    incompatible_sources: Optional[Sequence[str]] = None,
    species_by_source: Optional[Dict[str, str]] = None,
    reference_species: Optional[str] = None,
) -> Dict:
    """
    Runs the SAME source-held-out protocol once per domain-robustness
    strategy variant (ERM, source_balanced, CORAL, MMD, and optionally
    domain_adversarial) and aggregates each variant's per-source results —
    a paired comparison, since every variant sees identical held-out
    sources, development subjects, and seeds. task must be "smoke" or
    "cancer".
    """
    if task not in ("smoke", "cancer"):
        raise ValueError(f"run_domain_robustness_ablation: task must be 'smoke' or 'cancer', got {task!r}")

    variants = dict(ABLATION_VARIANTS)
    if not include_adversarial:
        variants.pop("domain_adversarial", None)

    primary_metric = "macro_f1" if task == "smoke" else "auroc"
    results = {}
    for variant_name, cfg in variants.items():
        if task == "cancer":
            per_source = run_cancer_source_held_out(
                context, model_names, device=device, domain_robustness_config=cfg, seed=seed,
                incompatible_sources=incompatible_sources, species_by_source=species_by_source,
                reference_species=reference_species,
            )
        else:
            # Task A's source-held-out protocol (source_held_out.py) does
            # not yet accept a domain_robustness_config (see that module's
            # docstring — Task A LOSO covers classical baselines and the
            # pathway model's plain-ERM fit only). ERM is the only variant
            # actually exercised for task="smoke"; other variants are
            # recorded as not_evaluable so the ablation report is honest
            # about what ran rather than silently omitting rows.
            if variant_name != "erm":
                results[variant_name] = {
                    "status": "not_evaluable",
                    "reason": "Task A source-held-out protocol does not yet support non-ERM domain "
                              "strategies (no cell-level Trainer curriculum integration) — see README.md",
                }
                continue
            per_source = run_smoke_source_held_out(
                context, model_names, device=device, seed=seed,
                incompatible_sources=incompatible_sources, species_by_source=species_by_source,
                reference_species=reference_species,
            )

        reports_list = list(per_source.values())
        agg = aggregate_source_reports(reports_list, primary_metric)
        results[variant_name] = {"per_source": per_source, "aggregate": agg}

    paired = _paired_comparison(results, primary_metric, baseline="erm")
    return {
        "task": task, "primary_metric": primary_metric, "variants": list(variants.keys()),
        "results": results, "paired_comparison_vs_erm": paired,
        "development_only": True, "frozen_test_accessed": False,
    }


def _paired_comparison(results: Dict, metric: str, baseline: str = "erm") -> Dict:
    base = results.get(baseline, {}).get("aggregate", {}).get("per_source", {})
    if not base:
        return {"status": "insufficient_evidence", "reason": f"baseline {baseline!r} produced no evaluated source"}
    comparison = {}
    for name, res in results.items():
        if name == baseline or "aggregate" not in res:
            continue
        other = res["aggregate"].get("per_source", {})
        common_sources = sorted(set(base) & set(other))
        if not common_sources:
            comparison[name] = {"status": "insufficient_evidence", "reason": "no source common to both variants"}
            continue
        diffs = [other[s] - base[s] for s in common_sources]
        comparison[name] = {
            "status": "evaluated", "n_common_sources": len(common_sources),
            "mean_paired_difference": sum(diffs) / len(diffs),
            "per_source_difference": {s: other[s] - base[s] for s in common_sources},
        }
    return comparison
