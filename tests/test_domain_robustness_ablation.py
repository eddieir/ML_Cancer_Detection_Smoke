"""Tests for benchmarks/domain_robustness_ablation.py — multi-seed paired
strategy comparison against the ERM baseline, held to a FIXED candidate
(pathway_hierarchical_mil) for the cancer task so no variant's result can
be attributed to a different winning model than the others (see the
module's docstring)."""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.domain_robustness_ablation import (
    FIXED_CANCER_ABLATION_CANDIDATE,
    _paired_comparison_multi_seed,
    _reject_non_fixed_candidate_winners,
    run_domain_robustness_ablation,
)
from benchmarks.robustness_report import build_robustness_report
from benchmarks.runner import _synthetic_dataset_manifest_entries, build_synthetic_context

_DATASET_MANIFEST_ENTRIES = _synthetic_dataset_manifest_entries(build_synthetic_context(seed=0, fast=True))

_BENCH_CFG = {
    "species_by_source": {"sourceA": "human", "sourceB": "human"}, "reference_species": "human",
    "dataset_manifest_entries": _DATASET_MANIFEST_ENTRIES,
}

_HASH_A = "a" * 64
_HASH_B = "b" * 64


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def _assert_paired_comparison_well_formed(paired, variants):
    """Accepts either shape validate_ablation_report itself accepts: a
    whole-dict insufficient_evidence status (e.g. the ERM baseline itself
    produced no evaluated source at all — a real possibility on this tiny
    synthetic fixture, since pathway_hierarchical_mil's nested-CV OOF
    selection needs more development subjects per fold than it provides —
    see test_source_held_out.py's identically-documented limitation), or a
    per-variant dict."""
    if paired.get("status") == "insufficient_evidence":
        assert paired.get("reason")
        return
    for variant in variants:
        if variant == "erm":
            continue
        comparison = paired[variant]
        assert comparison["status"] in ("evaluated", "insufficient_evidence")
        if comparison["status"] == "evaluated":
            assert "wins" in comparison and "losses" in comparison and "ties" in comparison
            assert comparison["wins"] + comparison["losses"] + comparison["ties"] == comparison["n_common_source_seed_pairs"]


def test_multi_seed_ablation_reports_seeds_and_paired_comparison():
    ctx = build_synthetic_context(seed=1, fast=True)
    report = run_domain_robustness_ablation(
        ctx, "cancer", ["prevalence"], device="cpu", seeds=[1, 2], include_adversarial=False, **_BENCH_CFG,
    )
    assert report["seeds"] == [1, 2]
    assert set(report["results"]["erm"]["per_seed"]) == {1, 2}
    _assert_paired_comparison_well_formed(report["paired_comparison_vs_erm"], report["variants"])


def test_cancer_ablation_fixes_candidate_regardless_of_requested_model_names():
    """The cancer strategy ablation must ALWAYS evaluate
    pathway_hierarchical_mil for every variant — a caller-supplied
    model_names list (here deliberately a classical-only list that would,
    under the old design, have won every variant's candidate-selection
    sweep) must never change which candidate this ablation actually runs;
    it is only recorded for provenance in requested_model_names."""
    ctx = build_synthetic_context(seed=1, fast=True)
    report = run_domain_robustness_ablation(
        ctx, "cancer", ["prevalence", "logistic"], device="cpu", seed=5, include_adversarial=False, **_BENCH_CFG,
    )
    assert report["candidate_name"] == FIXED_CANCER_ABLATION_CANDIDATE
    assert report["requested_model_names"] == ["prevalence", "logistic"]
    for variant, variant_result in report["results"].items():
        for seed_result in variant_result["per_seed"].values():
            if seed_result.get("status") == "not_evaluable":
                continue
            for source_report in seed_result["per_source"].values():
                if source_report.get("evaluated"):
                    assert source_report["model"] == FIXED_CANCER_ABLATION_CANDIDATE
                    assert source_report["strategy"] == variant


def test_single_seed_backward_compatible_default():
    ctx = build_synthetic_context(seed=1, fast=True)
    report = run_domain_robustness_ablation(
        ctx, "cancer", ["prevalence"], device="cpu", seed=3, include_adversarial=False, **_BENCH_CFG,
    )
    assert report["seeds"] == [3]


def test_smoke_task_non_erm_variants_marked_not_evaluable_per_seed():
    ctx = build_synthetic_context(seed=1, fast=True)
    report = run_domain_robustness_ablation(
        ctx, "smoke", ["majority"], device="cpu", seeds=[1, 2], include_adversarial=False, **_BENCH_CFG,
    )
    for variant in ("source_balanced", "coral", "mmd"):
        for s in (1, 2):
            assert report["results"][variant]["per_seed"][s]["status"] == "not_evaluable"


def test_smoke_task_candidate_name_is_not_applicable():
    """Task A never fixes a single candidate — it sweeps model_names for
    its ERM-only variant instead — so candidate_name must be a structured
    not_applicable value, never a bare candidate name or None."""
    ctx = build_synthetic_context(seed=1, fast=True)
    report = run_domain_robustness_ablation(
        ctx, "smoke", ["majority"], device="cpu", seed=1, include_adversarial=False, **_BENCH_CFG,
    )
    assert report["candidate_name"]["status"] == "not_applicable"


# ─── Direct unit tests for the paired-comparison identity/attribution
# checks (Step 9 items 9-21) — constructed from synthetic report dicts so
# they run instantly and don't depend on pathway_hierarchical_mil actually
# being MIL-eligible under a tiny fixture (see test_source_held_out.py's
# test_cancer_source_held_out_no_candidate_branch_validates for why the
# unit-test-scale synthetic fixture cannot reliably produce an evaluated
# pathway result at all). ──────────────────────────────────────────────────

def _evaluated_report(source, seed, model, applied_strategy, requested_strategy, auroc,
                       preprocessing_fingerprint=_HASH_A, module_fingerprint=_HASH_A,
                       source_split_manifest_fingerprint=_HASH_A):
    return build_robustness_report(
        task="cancer_prediction", model=model, strategy=applied_strategy, requested_strategy=requested_strategy,
        strategy_applicable=(applied_strategy == requested_strategy), held_out_source=source,
        eligibility={"status": "eligible"}, development_sources=["x"], metrics={"auroc": auroc}, seed=seed,
        evaluated=True, is_module_based_candidate=(model == FIXED_CANCER_ABLATION_CANDIDATE),
        preprocessing_fingerprint=preprocessing_fingerprint, gene_list_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, module_fingerprint=module_fingerprint if model == FIXED_CANCER_ABLATION_CANDIDATE else None,
        source_policy_fingerprint=_HASH_A, source_split_manifest_fingerprint=source_split_manifest_fingerprint,
        environment_fingerprint=_HASH_A, dataset_manifest_fingerprint=_HASH_A,
    ).to_dict()


def _results_with(erm_reports, other_variant, other_reports, seeds=(1, 2)):
    return {
        "erm": {"per_seed": {s: {"per_source": {r["held_out_source"]: r for r in erm_reports if r["seed"] == s},
                                  "aggregate": {"metric": "auroc"}} for s in seeds}},
        other_variant: {"per_seed": {s: {"per_source": {r["held_out_source"]: r for r in other_reports if r["seed"] == s},
                                          "aggregate": {"metric": "auroc"}} for s in seeds}},
    }


def test_paired_comparison_excludes_candidate_mismatched_pairs():
    """A (source, seed) pair where the two variants' winning candidate
    differs must never be paired — Step 9 item 15."""
    erm = [_evaluated_report("a", s, FIXED_CANCER_ABLATION_CANDIDATE, "erm", "erm", 0.6) for s in (1, 2)]
    coral = [_evaluated_report("a", s, "logistic", "erm", "coral", 0.9) for s in (1, 2)]
    results = _results_with(erm, "coral", coral)
    paired = _paired_comparison_multi_seed(results, "auroc", [1, 2], fixed_candidate=FIXED_CANCER_ABLATION_CANDIDATE)
    assert paired["coral"]["status"] == "insufficient_evidence"
    assert paired["coral"]["excluded_pairs"]["candidate_mismatch"] == 2


def test_paired_comparison_excludes_strategy_not_applicable_pairs():
    """A pair whose variant-side winner could NOT actually apply the
    requested strategy (strategy_applicable=False) must never be compared
    as if it were evidence for that strategy — Step 9 item 1/9."""
    erm = [_evaluated_report("a", s, FIXED_CANCER_ABLATION_CANDIDATE, "erm", "erm", 0.6) for s in (1, 2)]
    coral = [_evaluated_report("a", s, FIXED_CANCER_ABLATION_CANDIDATE, "erm", "coral", 0.9) for s in (1, 2)]
    results = _results_with(erm, "coral", coral)
    paired = _paired_comparison_multi_seed(results, "auroc", [1, 2], fixed_candidate=FIXED_CANCER_ABLATION_CANDIDATE)
    assert paired["coral"]["status"] == "insufficient_evidence"
    assert paired["coral"]["excluded_pairs"]["strategy_not_applicable"] == 2


def test_paired_comparison_excludes_identity_mismatched_pairs():
    """A pair whose preprocessing/module/split identity differs between the
    baseline and variant reports must never be paired — Step 9 item 16-18."""
    erm = [_evaluated_report("a", s, FIXED_CANCER_ABLATION_CANDIDATE, "erm", "erm", 0.6) for s in (1, 2)]
    coral = [
        _evaluated_report("a", s, FIXED_CANCER_ABLATION_CANDIDATE, "coral", "coral", 0.9,
                           preprocessing_fingerprint=_HASH_B)
        for s in (1, 2)
    ]
    results = _results_with(erm, "coral", coral)
    paired = _paired_comparison_multi_seed(results, "auroc", [1, 2], fixed_candidate=FIXED_CANCER_ABLATION_CANDIDATE)
    assert paired["coral"]["status"] == "insufficient_evidence"
    assert paired["coral"]["excluded_pairs"]["identity_mismatch"] == 2


def test_paired_comparison_accepts_matched_pairs():
    erm = [_evaluated_report("a", s, FIXED_CANCER_ABLATION_CANDIDATE, "erm", "erm", 0.5) for s in (1, 2)]
    coral = [_evaluated_report("a", s, FIXED_CANCER_ABLATION_CANDIDATE, "coral", "coral", 0.7) for s in (1, 2)]
    results = _results_with(erm, "coral", coral)
    paired = _paired_comparison_multi_seed(results, "auroc", [1, 2], fixed_candidate=FIXED_CANCER_ABLATION_CANDIDATE)
    assert paired["coral"]["status"] == "evaluated"
    assert paired["coral"]["n_common_source_seed_pairs"] == 2
    assert paired["coral"]["wins"] == 2


def test_reject_non_fixed_candidate_winners_raises_on_mismatch():
    bad = _evaluated_report("a", 1, "logistic", "erm", "erm", 0.7)
    with pytest.raises(RuntimeError):
        _reject_non_fixed_candidate_winners([bad], "erm", 1, FIXED_CANCER_ABLATION_CANDIDATE)


def test_reject_non_fixed_candidate_winners_accepts_fixed_candidate():
    good = _evaluated_report("a", 1, FIXED_CANCER_ABLATION_CANDIDATE, "erm", "erm", 0.7)
    _reject_non_fixed_candidate_winners([good], "erm", 1, FIXED_CANCER_ABLATION_CANDIDATE)
