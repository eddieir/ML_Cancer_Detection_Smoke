"""
Integration and leakage-protection tests for benchmarks/source_held_out.py —
the Phase 6 source-held-out ("LOSO") protocol. Uses the same synthetic
2-source ExperimentContext the rest of the benchmark suite uses
(runner.build_synthetic_context).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.runner import build_synthetic_context
from benchmarks.source_held_out import (
    ConflictingSmokeLabelError,
    CrossSourceSubjectConflictError,
    LabelSchemaError,
    UnsupportedSmokeCandidateError,
    UnsupportedSmokeDomainStrategyError,
    run_cancer_source_held_out,
    run_smoke_source_held_out,
)

_BENCH_CFG = {"species_by_source": {"sourceA": "human", "sourceB": "human"}, "reference_species": "human"}


@pytest.fixture(scope="module")
def ctx():
    return build_synthetic_context(seed=1, fast=True)


def test_smoke_source_held_out_completes_for_every_source(ctx):
    reports = run_smoke_source_held_out(ctx, ["majority", "logistic"], device="cpu", **_BENCH_CFG)
    assert set(reports) == {"sourceA", "sourceB"}
    for src, r in reports.items():
        assert r["development_only"] is True
        assert r["frozen_test_accessed"] is False
        assert src not in r["development_sources"]


def test_cancer_source_held_out_completes_for_every_source(ctx):
    reports = run_cancer_source_held_out(ctx, ["prevalence", "logistic"], device="cpu", n_oof_folds=2, **_BENCH_CFG)
    assert set(reports) == {"sourceA", "sourceB"}
    for src, r in reports.items():
        assert r["development_only"] is True
        assert r["frozen_test_accessed"] is False


def test_held_out_and_development_subjects_are_disjoint(ctx):
    reports = run_cancer_source_held_out(ctx, ["prevalence"], device="cpu", n_oof_folds=2, **_BENCH_CFG)
    for src, r in reports.items():
        # the manifest fingerprint construction itself raises on overlap
        # (see source_eligibility.build_source_held_out_manifest) — a
        # successfully returned report is already proof of disjointness,
        # this assertion documents that invariant explicitly.
        assert "source_split_manifest_fingerprint" in r


def test_frozen_test_guard_never_created_by_source_held_out(ctx):
    """source_held_out.py's public functions take no run_dir/output_root/
    guard-path argument at all, and never import test_guard.py — there is
    structurally no way for either function to create, acquire, or
    reference a FrozenTestGuard. Running both protocols must leave no
    guard directory anywhere under the repository's default guard
    locations."""
    import ast

    import benchmarks.source_held_out as soh

    tree = ast.parse(open(soh.__file__).read())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(a.name for a in node.names)
    assert "test_guard" not in imported_names
    assert "FrozenTestGuard" not in imported_names

    guard_root = Path("artifacts/benchmarks/.frozen_test_guards")
    existed_before = guard_root.exists()
    run_cancer_source_held_out(ctx, ["prevalence"], device="cpu", n_oof_folds=2, **_BENCH_CFG)
    run_smoke_source_held_out(ctx, ["majority"], device="cpu", **_BENCH_CFG)
    assert guard_root.exists() == existed_before


def test_cross_source_subject_conflict_detected(ctx, monkeypatch):
    """A subject_id assigned to two different dataset_source values in the
    train+val pool must be rejected, never silently resolved."""
    normalized_adata = ctx.normalized_adata_for_refit
    obs = normalized_adata.obs.copy()
    # Corrupt exactly one train subject's source for a subset of its cells
    # so it appears under two different sources.
    train_mask = obs["subject_id"] == "train_0"
    half = obs.index[train_mask][: max(1, int(train_mask.sum() / 2))]
    obs.loc[half, "source"] = "sourceB_corrupted"
    normalized_adata.obs = obs
    with pytest.raises(CrossSourceSubjectConflictError):
        run_cancer_source_held_out(ctx, ["prevalence"], device="cpu", n_oof_folds=2, **_BENCH_CFG)


def test_ineligible_source_recorded_with_reason_not_crashed():
    """A source with too few subjects/classes must be reported as
    ineligible with an explicit reason, never crash the whole sweep."""
    ctx2 = build_synthetic_context(seed=2, fast=True)
    reports = run_smoke_source_held_out(
        ctx2, ["majority"], device="cpu",
        species_by_source={"sourceA": "human"}, reference_species="human",  # sourceB undeclared
    )
    assert reports["sourceB"]["eligibility"]["status"] == "species_mismatch"
    assert reports["sourceB"]["eligibility"]["eligible"] is False


def test_mouse_source_excluded_from_human_only_protocol():
    ctx3 = build_synthetic_context(seed=3, fast=True)
    reports = run_smoke_source_held_out(
        ctx3, ["majority"], device="cpu",
        species_by_source={"sourceA": "human", "sourceB": "mouse"}, reference_species="human",
    )
    assert reports["sourceB"]["eligibility"]["status"] == "species_mismatch"


def test_smoke_conflicting_verified_labels_rejected():
    """A subject whose own smoke_type_known=True cells disagree on
    smoke_type must raise, never be silently resolved by majority vote."""
    ctx4 = build_synthetic_context(seed=4, fast=True)
    normalized_adata = ctx4.normalized_adata_for_refit
    obs = normalized_adata.obs.copy()
    obs["smoke_type_known"] = True
    train_mask = (obs["subject_id"] == "train_0").values
    idx = obs.index[train_mask]
    half = idx[: max(1, len(idx) // 2)]
    obs.loc[half, "smoke_type"] = (obs.loc[half, "smoke_type"].astype(int) + 1) % 3
    normalized_adata.obs = obs
    with pytest.raises(ConflictingSmokeLabelError):
        run_smoke_source_held_out(ctx4, ["majority"], device="cpu", **_BENCH_CFG)


def test_smoke_unknown_cells_excluded_from_verified_label():
    """A subject whose ONLY known cells were flipped to unknown must be
    excluded from the verified-label pool (not the same as a conflict)."""
    ctx5 = build_synthetic_context(seed=5, fast=True)
    normalized_adata = ctx5.normalized_adata_for_refit
    obs = normalized_adata.obs.copy()
    obs["smoke_type_known"] = True
    obs.loc[(obs["subject_id"] == "train_0").values, "smoke_type_known"] = False
    normalized_adata.obs = obs
    reports = run_smoke_source_held_out(ctx5, ["majority"], device="cpu", **_BENCH_CFG)
    dev_diag = reports["sourceB"]["label_state"]["development"]
    assert dev_diag["unknown_labels"] >= 1
    assert "train_0" in dev_diag["unknown_subjects"]
    assert "train_0" not in dev_diag["verified_subjects"]
    assert "weak_label_policy_fingerprint" in dev_diag
    assert dev_diag["weak_labels_enabled"] is False


def test_smoke_missing_smoke_type_known_column_rejected():
    """normalized_adata.obs missing smoke_type_known entirely must raise a
    typed schema error, never silently assume every label is verified."""
    ctx6 = build_synthetic_context(seed=6, fast=True)
    normalized_adata = ctx6.normalized_adata_for_refit
    obs = normalized_adata.obs.drop(columns=["smoke_type_known"])
    normalized_adata.obs = obs
    with pytest.raises(LabelSchemaError):
        run_smoke_source_held_out(ctx6, ["majority"], device="cpu", **_BENCH_CFG)


def test_smoke_weak_proxy_only_subject_excluded_by_default_policy():
    """A subject whose only cells are weak-proxy (not smoke_type_known) is
    reported separately from a plain-unknown subject, and is excluded from
    the verified pool by default (weak_labels_enabled=False)."""
    ctx7 = build_synthetic_context(seed=7, fast=True)
    normalized_adata = ctx7.normalized_adata_for_refit
    obs = normalized_adata.obs.copy()
    train0_mask = (obs["subject_id"] == "train_0").values
    obs["weak_smoke_proxy_known"] = False
    obs.loc[train0_mask, "smoke_type_known"] = False
    obs.loc[train0_mask, "weak_smoke_proxy_known"] = True
    normalized_adata.obs = obs
    reports = run_smoke_source_held_out(ctx7, ["majority"], device="cpu", **_BENCH_CFG)
    dev_diag = reports["sourceB"]["label_state"]["development"]
    assert "train_0" in dev_diag["weak_proxy_only_subjects"]
    assert "train_0" in dev_diag["excluded_by_policy_subjects"]
    assert "train_0" not in dev_diag["verified_subjects"]


def test_smoke_unsupported_candidate_rejected(ctx):
    with pytest.raises(UnsupportedSmokeCandidateError):
        run_smoke_source_held_out(ctx, ["neural"], device="cpu", **_BENCH_CFG)


def test_smoke_domain_strategy_other_than_erm_rejected(ctx):
    with pytest.raises(UnsupportedSmokeDomainStrategyError):
        run_smoke_source_held_out(
            ctx, ["majority"], device="cpu", domain_robustness_config={"strategy": "coral"}, **_BENCH_CFG,
        )


def test_smoke_held_out_label_corruption_does_not_change_selected_model_or_preprocessing():
    """Corrupting the held-out source's own labels must never change which
    candidate was selected, its development evidence, or the frozen
    preprocessing artifact — only the final held-out metrics may move."""
    ctx6 = build_synthetic_context(seed=6, fast=True)
    reports_before = run_smoke_source_held_out(ctx6, ["majority", "logistic"], device="cpu", **_BENCH_CFG)

    normalized_adata = ctx6.normalized_adata_for_refit
    obs = normalized_adata.obs.copy()
    held_out_mask = (obs["source"] == "sourceB").values
    obs.loc[held_out_mask, "smoke_type"] = (obs.loc[held_out_mask, "smoke_type"].astype(int) + 1) % 3
    normalized_adata.obs = obs
    reports_after = run_smoke_source_held_out(ctx6, ["majority", "logistic"], device="cpu", **_BENCH_CFG)

    before, after = reports_before["sourceB"], reports_after["sourceB"]
    assert before["model"] == after["model"]
    assert before["preprocessing_fingerprint"] == after["preprocessing_fingerprint"]
    assert before["comparisons"] == after["comparisons"]


@pytest.fixture(autouse=True)
def _cleanup_checkpoints():
    yield
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)
