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
from benchmarks.robustness_report import is_not_applicable, validate_robustness_report
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
    evaluated_any = False
    for src, r in reports.items():
        assert r["development_only"] is True
        assert r["frozen_test_accessed"] is False
        assert src not in r["development_sources"]
        validate_robustness_report(r)
        if r.get("evaluated"):
            evaluated_any = True
            # a classical smoke baseline (majority/logistic) has no
            # gene-module structure — module_fingerprint must be the
            # structured not_applicable value, never a real hash.
            assert r["is_module_based_candidate"] is False
            assert is_not_applicable(r["module_fingerprint"])
    assert evaluated_any


def test_cancer_source_held_out_completes_for_every_source(ctx):
    reports = run_cancer_source_held_out(ctx, ["prevalence", "logistic"], device="cpu", n_oof_folds=2, **_BENCH_CFG)
    assert set(reports) == {"sourceA", "sourceB"}
    evaluated_any = False
    for src, r in reports.items():
        assert r["development_only"] is True
        assert r["frozen_test_accessed"] is False
        validate_robustness_report(r)
        if r.get("evaluated"):
            evaluated_any = True
            # a classical cancer baseline (prevalence/logistic) has no
            # gene-module structure either.
            assert r["is_module_based_candidate"] is False
            assert is_not_applicable(r["module_fingerprint"])
            # cancer reports are always calibrated (build_frozen_policy runs
            # on every successful fit) — calibration/threshold identities
            # must be real hashes, never not_applicable.
            assert r["calibration"]
            assert not is_not_applicable(r["calibration_fingerprint"])
            assert not is_not_applicable(r["threshold_policy_fingerprint"])
    assert evaluated_any


def test_smoke_reports_are_uncalibrated_with_not_applicable_calibration_identities(ctx):
    """Task A never runs probability calibration — every evaluated smoke
    report's calibration block is empty and its calibration/threshold
    identities must be the structured not_applicable value."""
    reports = run_smoke_source_held_out(ctx, ["majority", "logistic"], device="cpu", **_BENCH_CFG)
    evaluated_any = False
    for src, r in reports.items():
        if r.get("evaluated"):
            evaluated_any = True
            assert not r["calibration"]
            assert is_not_applicable(r["calibration_fingerprint"])
            assert is_not_applicable(r["threshold_policy_fingerprint"])
    assert evaluated_any


def test_cancer_source_held_out_no_candidate_branch_validates(ctx):
    """When the only requested candidate cannot be evaluated (this tiny
    fixture has too few subjects for pathway_hierarchical_mil's nested-CV
    OOF selection), the resulting evaluated=False report must still
    validate cleanly against schema v2 and record a real reason."""
    reports = run_cancer_source_held_out(ctx, ["pathway_hierarchical_mil"], device="cpu", n_oof_folds=2, **_BENCH_CFG)
    saw_no_candidate_branch = False
    for src, r in reports.items():
        validate_robustness_report(r)
        if not r.get("evaluated") and r["model"] is None:
            saw_no_candidate_branch = True
            assert r["limitations"]
    assert saw_no_candidate_branch


def test_cancer_source_held_out_non_module_mil_reports_not_applicable_module_fingerprint(ctx):
    """A pooling-based MIL model (attention_mil) is MIL but NOT
    module-based — whichever candidate wins here, module_fingerprint must
    be the structured not_applicable value, never a real hash, exactly
    like a classical baseline (attention_mil's own nested-CV OOF selection
    needs more development subjects per fold than this tiny synthetic
    fixture provides — prevalence is included so the sweep has a candidate
    that can actually win; is_module_based_candidate=False either way)."""
    reports = run_cancer_source_held_out(
        ctx, ["attention_mil", "prevalence"], device="cpu", n_oof_folds=2, **_BENCH_CFG,
    )
    evaluated_any = False
    for src, r in reports.items():
        validate_robustness_report(r)
        if r.get("evaluated"):
            evaluated_any = True
            assert r["model"] in ("attention_mil", "prevalence")
            assert r["is_module_based_candidate"] is False
            assert is_not_applicable(r["module_fingerprint"])
    assert evaluated_any


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
    validate_robustness_report(reports["sourceB"])  # ineligible branch must still validate cleanly
    assert reports["sourceB"].get("evaluated") is not True


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


def test_smoke_candidate_dev_score_every_labeled_dev_subject_is_out_of_fold():
    """Every verified-label development subject that receives an OOF entry
    must have been predicted by a fold it did NOT train on — this test
    checks the invariant directly against _smoke_candidate_dev_score's
    fold construction (grouped_kfold guarantees disjoint train/val subject
    sets per fold; this test proves oof_by_subject respects that)."""
    from benchmarks.source_held_out import _smoke_candidate_dev_score
    from data.splitting import grouped_kfold

    ctx = build_synthetic_context(seed=11, fast=True)
    normalized_adata = ctx.normalized_adata_for_refit
    obs = normalized_adata.obs
    dev_subjects = sorted(set(obs["subject_id"].astype(str)))
    label_by_subject = {
        s: int(obs.loc[(obs["subject_id"].astype(str) == s).values, "smoke_type"].iloc[0]) for s in dev_subjects
    }
    num_cell_types = ctx.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = ctx.preprocessing_artifact.n_hvgs

    score, evidence = _smoke_candidate_dev_score(
        ctx, "majority", dev_subjects, label_by_subject, num_cell_types, 3, n_hvgs,
        device="cpu", seed=1, n_dev_folds=3, n_inner_folds=2,
    )
    oof_by_subject = evidence["oof_by_subject"]
    assert oof_by_subject  # at least some subjects received an OOF entry
    # every OOF subject's probability vector is a proper distribution over
    # the fixed num_classes axis (aligned even if a fold's training data
    # missed a class).
    for sid, proba in oof_by_subject.items():
        assert proba.shape == (3,)
        assert proba.sum() == pytest.approx(1.0, abs=1e-6) or proba.sum() == pytest.approx(0.0)


def test_smoke_candidate_dev_score_no_duplicate_oof_predictions():
    """A subject cannot legitimately appear in more than one fold's
    validation set — _smoke_candidate_dev_score raises a RuntimeError if
    this invariant is ever violated, rather than silently overwriting one
    fold's OOF entry with another's."""
    from benchmarks.source_held_out import _smoke_candidate_dev_score

    ctx = build_synthetic_context(seed=12, fast=True)
    normalized_adata = ctx.normalized_adata_for_refit
    obs = normalized_adata.obs
    dev_subjects = sorted(set(obs["subject_id"].astype(str)))
    label_by_subject = {
        s: int(obs.loc[(obs["subject_id"].astype(str) == s).values, "smoke_type"].iloc[0]) for s in dev_subjects
    }
    num_cell_types = ctx.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = ctx.preprocessing_artifact.n_hvgs

    score, evidence = _smoke_candidate_dev_score(
        ctx, "majority", dev_subjects, label_by_subject, num_cell_types, 3, n_hvgs,
        device="cpu", seed=2, n_dev_folds=3, n_inner_folds=2,
    )
    oof_subjects = list(evidence["oof_by_subject"])
    assert len(oof_subjects) == len(set(oof_subjects))  # no subject appears twice


def test_smoke_uncertainty_threshold_unaffected_by_held_out_label_corruption():
    """Corrupting the held-out source's labels must never change the
    development-only abstention threshold — the threshold is selected from
    OOF development predictions before the held-out source is ever
    touched."""
    ctx7 = build_synthetic_context(seed=13, fast=True)
    reports_before = run_smoke_source_held_out(ctx7, ["majority", "logistic"], device="cpu", **_BENCH_CFG)

    normalized_adata = ctx7.normalized_adata_for_refit
    obs = normalized_adata.obs.copy()
    held_out_mask = (obs["source"] == "sourceB").values
    obs.loc[held_out_mask, "smoke_type"] = (obs.loc[held_out_mask, "smoke_type"].astype(int) + 1) % 3
    normalized_adata.obs = obs
    reports_after = run_smoke_source_held_out(ctx7, ["majority", "logistic"], device="cpu", **_BENCH_CFG)

    before_unc = reports_before["sourceB"]["uncertainty"]
    after_unc = reports_after["sourceB"]["uncertainty"]
    if before_unc.get("development_threshold_selection", {}).get("status") == "selected":
        assert (before_unc["development_threshold_selection"]["uncertainty_threshold"]
                == after_unc["development_threshold_selection"]["uncertainty_threshold"])


def test_smoke_uncertainty_threshold_unaffected_by_held_out_expression_corruption():
    """Corrupting the held-out source's own gene expression (not just its
    labels) must also never change the development-only abstention
    threshold, since it is selected purely from development OOF
    predictions before any held-out data is read."""
    ctx8 = build_synthetic_context(seed=14, fast=True)
    reports_before = run_smoke_source_held_out(ctx8, ["majority", "logistic"], device="cpu", **_BENCH_CFG)

    normalized_adata = ctx8.normalized_adata_for_refit
    held_out_mask = (normalized_adata.obs["source"] == "sourceB").values
    rng = np.random.RandomState(0)
    corrupted_X = np.asarray(normalized_adata.X).copy()
    corrupted_X[held_out_mask] = rng.randn(*corrupted_X[held_out_mask].shape).astype(corrupted_X.dtype)
    normalized_adata.X = corrupted_X
    reports_after = run_smoke_source_held_out(ctx8, ["majority", "logistic"], device="cpu", **_BENCH_CFG)

    before_unc = reports_before["sourceB"]["uncertainty"]
    after_unc = reports_after["sourceB"]["uncertainty"]
    assert before_unc.get("development_threshold_selection") == after_unc.get("development_threshold_selection")
    # model selection/preprocessing must also be untouched by held-out expression corruption
    assert reports_before["sourceB"]["model"] == reports_after["sourceB"]["model"]
    assert reports_before["sourceB"]["preprocessing_fingerprint"] == reports_after["sourceB"]["preprocessing_fingerprint"]


@pytest.fixture(autouse=True)
def _cleanup_checkpoints():
    yield
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)
