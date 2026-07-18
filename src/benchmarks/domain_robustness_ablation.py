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

import numpy as np

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


MIN_SOURCE_SEED_PAIRS_FOR_EVIDENCE = 2


def run_domain_robustness_ablation(
    context, task: str, model_names: Sequence[str], device: str = "cpu",
    seed: int = 42, seeds: Optional[Sequence[int]] = None, include_adversarial: bool = True,
    incompatible_sources: Optional[Sequence[str]] = None,
    species_by_source: Optional[Dict[str, str]] = None,
    reference_species: Optional[str] = None,
) -> Dict:
    """
    Runs the SAME source-held-out protocol once per domain-robustness
    strategy variant (ERM, source_balanced, CORAL, MMD, and optionally
    domain_adversarial) x once per declared seed, and aggregates each
    variant-seed's per-source results — a paired comparison, since every
    variant sees identical held-out sources, development subjects, and
    seeds. task must be "smoke" or "cancer". seeds, if given, overrides the
    single `seed` (backward-compatible default: seeds=(seed,)).
    """
    if task not in ("smoke", "cancer"):
        raise ValueError(f"run_domain_robustness_ablation: task must be 'smoke' or 'cancer', got {task!r}")
    seeds = list(seeds) if seeds is not None else [seed]

    variants = dict(ABLATION_VARIANTS)
    if not include_adversarial:
        variants.pop("domain_adversarial", None)

    primary_metric = "macro_f1" if task == "smoke" else "auroc"
    results: Dict[str, Dict] = {}
    for variant_name, cfg in variants.items():
        per_seed: Dict[int, Dict] = {}
        for s in seeds:
            if task == "cancer":
                per_source = run_cancer_source_held_out(
                    context, model_names, device=device, domain_robustness_config=cfg, seed=s,
                    incompatible_sources=incompatible_sources, species_by_source=species_by_source,
                    reference_species=reference_species,
                )
            else:
                # Task A's source-held-out protocol (source_held_out.py)
                # supports ERM only (see UnsupportedSmokeDomainStrategyError)
                # — other variants are recorded as not_evaluable so the
                # ablation report is honest about what ran rather than
                # silently omitting rows.
                if variant_name != "erm":
                    per_seed[s] = {
                        "status": "not_evaluable",
                        "reason": "Task A source-held-out protocol supports ERM only — domain-robust "
                                  "training strategies are cancer-only in this repository — see README.md",
                    }
                    continue
                per_source = run_smoke_source_held_out(
                    context, model_names, device=device, seed=s,
                    incompatible_sources=incompatible_sources, species_by_source=species_by_source,
                    reference_species=reference_species,
                )
            reports_list = list(per_source.values())
            agg = aggregate_source_reports(reports_list, primary_metric)
            per_seed[s] = {"per_source": per_source, "aggregate": agg}
        results[variant_name] = {"per_seed": per_seed}

    paired = _paired_comparison_multi_seed(results, primary_metric, seeds, baseline="erm")
    return {
        "task": task, "primary_metric": primary_metric, "variants": list(variants.keys()), "seeds": seeds,
        "results": results, "paired_comparison_vs_erm": paired,
        "development_only": True, "frozen_test_accessed": False,
    }


def _source_seed_values(variant_result: Dict, seeds: Sequence[int]) -> Dict[tuple, float]:
    """{(source, seed): metric_value} for every seed where this variant was
    actually evaluated (not_evaluable/insufficient_evidence seeds contribute
    nothing, never a fabricated 0.0)."""
    out = {}
    for s in seeds:
        seed_result = variant_result.get("per_seed", {}).get(s, {})
        per_source = seed_result.get("aggregate", {}).get("per_source", {})
        for src, val in per_source.items():
            out[(src, s)] = val
    return out


def _paired_comparison_multi_seed(results: Dict, metric: str, seeds: Sequence[int], baseline: str = "erm") -> Dict:
    """
    Per-(source, seed) paired comparison against the ERM baseline — never
    treats cells as independent replicates; the independent units here are
    (source, seed) pairs. Reports mean/median paired difference, dispersion
    (std across pairs), win/tie/loss counts, and an explicit
    insufficient_evidence status when fewer than
    MIN_SOURCE_SEED_PAIRS_FOR_EVIDENCE common (source, seed) pairs exist —
    never a point estimate presented without its sample size.
    """
    base_values = _source_seed_values(results.get(baseline, {}), seeds)
    if not base_values:
        return {"status": "insufficient_evidence", "reason": f"baseline {baseline!r} produced no evaluated source"}

    comparison = {}
    for name, res in results.items():
        if name == baseline:
            continue
        other_values = _source_seed_values(res, seeds)
        common = sorted(set(base_values) & set(other_values))
        if len(common) < MIN_SOURCE_SEED_PAIRS_FOR_EVIDENCE:
            comparison[name] = {
                "status": "insufficient_evidence",
                "reason": f"only {len(common)} common (source, seed) pair(s) evaluated for both variants "
                          f"(< {MIN_SOURCE_SEED_PAIRS_FOR_EVIDENCE})",
                "n_common_pairs": len(common),
            }
            continue
        diffs = np.array([other_values[k] - base_values[k] for k in common])
        wins = int((diffs > 0).sum())
        losses = int((diffs < 0).sum())
        ties = int((diffs == 0).sum())
        comparison[name] = {
            "status": "evaluated", "n_common_source_seed_pairs": len(common),
            "mean_paired_difference": float(diffs.mean()),
            "median_paired_difference": float(np.median(diffs)),
            "std_paired_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else None,
            "wins": wins, "losses": losses, "ties": ties,
            "per_source_seed_difference": {f"{src}|seed={s}": float(d) for (src, s), d in zip(common, diffs)},
        }
    return comparison
