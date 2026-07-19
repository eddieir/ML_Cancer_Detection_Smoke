"""
benchmarks/domain_robustness_ablation.py — development-only comparison of
domain-robustness training strategies for pathway_hierarchical_mil, run
under the identical source-held-out protocol (source_held_out.py) so every
variant is compared on the same held-out sources, the same development
subjects, the same preprocessing/module artifacts, and the same seeds.

This is a FIXED-MODEL strategy ablation, not a model-selection benchmark:
for the cancer task, every variant (including the ERM reference) is run
against the SAME single candidate — pathway_hierarchical_mil, the only
registered candidate with a training-time attachment point for source-
balanced sampling / CORAL / MMD / domain-adversarial training at all (see
candidate_registry.py). A caller-supplied model_names list is therefore
intentionally NOT used to run candidate selection across heterogeneous
candidate kinds inside this ablation — mixing "which model is best" with
"which strategy is best" would let a classical baseline that merely won
that source's OOF sweep silently stand in for every strategy variant,
which is exactly the scientific-attribution bug this module's fixed-
candidate design prevents structurally. General model-selection
comparisons (classical vs. MIL vs. pathway) belong to the plain
--domain-strategy path (run_cancer_source_held_out called directly), not
to this ablation.

Never touches the frozen test split or its guard — see source_held_out.py's
module docstring for why that is true structurally, not just by convention.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from .ablation_report import build_ablation_report
from .pathway_hierarchical_adapter import MODEL_NAME as PATHWAY_MODEL_NAME
from .robustness_report import aggregate_source_reports, is_not_applicable, not_applicable, validate_per_source_reports
from .source_held_out import run_cancer_source_held_out, run_smoke_source_held_out

# The one and only candidate this ablation ever fixes the cancer-task
# strategy comparison to — see this module's docstring.
FIXED_CANCER_ABLATION_CANDIDATE = PATHWAY_MODEL_NAME

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
    reference_assay_mode: Optional[str] = None,
    dataset_manifest_entries=None,
) -> Dict:
    """
    Runs the SAME source-held-out protocol once per domain-robustness
    strategy variant (ERM, source_balanced, CORAL, MMD, and optionally
    domain_adversarial) x once per declared seed, and aggregates each
    variant-seed's per-source results — a paired comparison, since every
    variant sees identical held-out sources, development subjects, and
    seeds. task must be "smoke" or "cancer". seeds, if given, overrides the
    single `seed` (backward-compatible default: seeds=(seed,)).

    For the cancer task, `model_names` is NOT used to select a candidate
    per source/variant — every variant (including ERM) is run against the
    single FIXED_CANCER_ABLATION_CANDIDATE (pathway_hierarchical_mil), the
    only registered candidate capable of running every declared strategy —
    see this module's docstring. `model_names` is accepted for smoke-task
    backward compatibility (Task A's ERM-only sweep) and is otherwise
    ignored for cancer, but the caller-supplied value is still recorded in
    the returned report's `requested_model_names` field for provenance.
    """
    if task not in ("smoke", "cancer"):
        raise ValueError(f"run_domain_robustness_ablation: task must be 'smoke' or 'cancer', got {task!r}")
    seeds = list(seeds) if seeds is not None else [seed]

    variants = dict(ABLATION_VARIANTS)
    if not include_adversarial:
        variants.pop("domain_adversarial", None)

    primary_metric = "macro_f1" if task == "smoke" else "auroc"
    fixed_candidate = FIXED_CANCER_ABLATION_CANDIDATE if task == "cancer" else None
    candidate_names = [FIXED_CANCER_ABLATION_CANDIDATE] if task == "cancer" else model_names

    results: Dict[str, Dict] = {}
    for variant_name, cfg in variants.items():
        per_seed: Dict[int, Dict] = {}
        for s in seeds:
            if task == "cancer":
                per_source = run_cancer_source_held_out(
                    context, candidate_names, device=device, domain_robustness_config=cfg, seed=s,
                    incompatible_sources=incompatible_sources, species_by_source=species_by_source,
                    reference_species=reference_species, reference_assay_mode=reference_assay_mode,
                    dataset_manifest_entries=dataset_manifest_entries,
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
                    context, candidate_names, device=device, seed=s,
                    incompatible_sources=incompatible_sources, species_by_source=species_by_source,
                    reference_species=reference_species, reference_assay_mode=reference_assay_mode,
                    dataset_manifest_entries=dataset_manifest_entries,
                )
            reports_list = list(per_source.values())
            # Validate every per-source report BEFORE it enters this
            # ablation's own results structure — this is the production
            # enforcement point, since run_domain_robustness_ablation's
            # report shape is not a build_aggregate_report() aggregate and
            # so is not covered by that function's internal validation.
            validate_per_source_reports(reports_list)
            if fixed_candidate is not None:
                _reject_non_fixed_candidate_winners(reports_list, variant_name, s, fixed_candidate)
            agg = aggregate_source_reports(reports_list, primary_metric)
            per_seed[s] = {"per_source": per_source, "aggregate": agg}
        results[variant_name] = {"per_seed": per_seed}

    paired = _paired_comparison_multi_seed(
        results, primary_metric, seeds, baseline="erm", fixed_candidate=fixed_candidate,
    )
    candidate_name_field = fixed_candidate if fixed_candidate is not None else not_applicable(
        "Task A ablation does not fix a single candidate — the ERM variant sweeps model_names "
        "and non-ERM variants are not_evaluable (cancer-only strategies)."
    )
    # build_ablation_report validates the WHOLE structure — including
    # recursively validating every nested per-source robustness report —
    # and stamps a fingerprint over the final assembled content before
    # returning; this is the production enforcement point, not something a
    # caller/test must remember to invoke separately.
    return build_ablation_report(
        task, primary_metric, list(variants.keys()), seeds, results, paired,
        candidate_name=candidate_name_field, requested_model_names=list(model_names),
    )


def _reject_non_fixed_candidate_winners(reports_list, variant_name: str, seed: int, fixed_candidate: str) -> None:
    """
    Defensive structural check for the fixed-model ablation design (Step 5):
    every EVALUATED per-source report for this variant/seed must have won
    with `fixed_candidate` — since candidate_names was itself restricted to
    [fixed_candidate] above, this can only fail if run_cancer_source_held_out
    itself has a bug, but this ablation never silently trusts that instead
    of checking.
    """
    bad = [
        r["held_out_source"] for r in reports_list
        if r.get("evaluated") and r.get("model") != fixed_candidate
    ]
    if bad:
        raise RuntimeError(
            f"run_domain_robustness_ablation: variant={variant_name!r} seed={seed!r} produced evaluated "
            f"report(s) whose winning candidate is not the fixed ablation candidate {fixed_candidate!r} "
            f"for held-out source(s) {bad} — the fixed-model ablation design requires every variant to "
            "evaluate the identical candidate; this indicates a genuine bug, never an expected outcome."
        )


# Identity fields that must agree between the ERM baseline's and a
# variant's report for the SAME (source, seed) pair before that pair may
# enter a paired comparison — a mismatch means the two runs did not
# actually hold everything but the strategy fixed (e.g. a different
# preprocessing refit, a different module snapshot, a different held-out/
# development split), so averaging them would compare apples to oranges.
_PAIRING_IDENTITY_FIELDS = (
    "preprocessing_fingerprint", "module_fingerprint", "source_split_manifest_fingerprint",
)


def _evaluated_source_seed_reports(variant_result: Dict, seeds: Sequence[int], metric: str) -> Dict[tuple, Dict]:
    """{(source, seed): per-source robustness report dict} for every
    (source, seed) where this variant was actually evaluated (produced a
    defined `metric` value) — not_evaluable/insufficient_evidence/
    ineligible entries contribute nothing, never a fabricated value.
    Returns the FULL report dict (not just the metric) so the caller can
    check candidate/candidate-kind/applied-strategy/preprocessing identity
    before pairing, not merely compare bare numbers."""
    out = {}
    for s in seeds:
        seed_result = variant_result.get("per_seed", {}).get(s, {})
        per_source = seed_result.get("per_source")
        if not isinstance(per_source, dict):
            continue
        for src, report in per_source.items():
            if report.get("metrics", {}).get(metric) is not None:
                out[(src, s)] = report
    return out


def _paired_comparison_multi_seed(
    results: Dict, metric: str, seeds: Sequence[int], baseline: str = "erm",
    fixed_candidate: Optional[str] = None,
) -> Dict:
    """
    Per-(source, seed) paired comparison against the ERM baseline — never
    treats cells as independent replicates; the independent units here are
    (source, seed) pairs. Reports mean/median paired difference, dispersion
    (std across pairs), win/tie/loss counts, and an explicit
    insufficient_evidence status when fewer than
    MIN_SOURCE_SEED_PAIRS_FOR_EVIDENCE USABLE common (source, seed) pairs
    exist — never a point estimate presented without its sample size.

    A common (source, seed) pair is USABLE only if, in addition to both
    sides producing a defined metric value: (a) both sides' winning
    candidate is `fixed_candidate` when one was declared for this ablation
    (Step 5/9); (b) the variant side's `strategy_applicable` is True — a
    result whose winning candidate could not actually apply the requested
    strategy (strategy_applicable=False) must never be compared as if it
    were evidence for that strategy; (c) every field in
    _PAIRING_IDENTITY_FIELDS agrees between the baseline and variant
    reports for that pair. Excluded pairs are counted and reasoned, never
    silently dropped without a trace.
    """
    base_reports = _evaluated_source_seed_reports(results.get(baseline, {}), seeds, metric)
    if not base_reports:
        return {"status": "insufficient_evidence", "reason": f"baseline {baseline!r} produced no evaluated source"}

    comparison = {}
    for name, res in results.items():
        if name == baseline:
            continue
        other_reports = _evaluated_source_seed_reports(res, seeds, metric)
        common_keys = sorted(set(base_reports) & set(other_reports))
        excluded = {"candidate_mismatch": 0, "strategy_not_applicable": 0, "identity_mismatch": 0}
        usable = []
        for key in common_keys:
            b, o = base_reports[key], other_reports[key]
            if fixed_candidate is not None and (b.get("model") != fixed_candidate or o.get("model") != fixed_candidate):
                excluded["candidate_mismatch"] += 1
                continue
            if b.get("model") != o.get("model"):
                excluded["candidate_mismatch"] += 1
                continue
            if not o.get("strategy_applicable", True):
                excluded["strategy_not_applicable"] += 1
                continue
            if any(b.get(f) != o.get(f) for f in _PAIRING_IDENTITY_FIELDS):
                excluded["identity_mismatch"] += 1
                continue
            usable.append(key)
        if len(usable) < MIN_SOURCE_SEED_PAIRS_FOR_EVIDENCE:
            comparison[name] = {
                "status": "insufficient_evidence",
                "reason": f"only {len(usable)} usable common (source, seed) pair(s) for both variants "
                          f"(< {MIN_SOURCE_SEED_PAIRS_FOR_EVIDENCE}); {len(common_keys) - len(usable)} "
                          f"common pair(s) excluded: {excluded}",
                "n_common_pairs": len(usable), "excluded_pairs": excluded,
            }
            continue
        diffs = np.array([other_reports[k]["metrics"][metric] - base_reports[k]["metrics"][metric] for k in usable])
        wins = int((diffs > 0).sum())
        losses = int((diffs < 0).sum())
        ties = int((diffs == 0).sum())
        comparison[name] = {
            "status": "evaluated", "n_common_source_seed_pairs": len(usable),
            "mean_paired_difference": float(diffs.mean()),
            "median_paired_difference": float(np.median(diffs)),
            "std_paired_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else None,
            "wins": wins, "losses": losses, "ties": ties,
            "excluded_pairs": excluded,
            "per_source_seed_difference": {f"{src}|seed={s}": float(d) for (src, s), d in zip(usable, diffs)},
        }
    return comparison
