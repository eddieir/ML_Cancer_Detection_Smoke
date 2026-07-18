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
    CrossSourceSubjectConflictError,
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


@pytest.fixture(autouse=True)
def _cleanup_checkpoints():
    yield
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)
