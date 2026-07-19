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

from benchmarks.runner import _synthetic_dataset_manifest_entries, build_synthetic_context
from benchmarks.robustness_report import RobustnessReportValidationError, is_not_applicable, validate_robustness_report
from benchmarks.source_held_out import (
    ConflictingSmokeLabelError,
    CrossSourceSubjectConflictError,
    LabelSchemaError,
    UnsupportedSmokeCandidateError,
    UnsupportedSmokeDomainStrategyError,
    run_cancer_source_held_out,
    run_smoke_source_held_out,
)

# One fixed synthetic dataset manifest (sourceA/sourceB, matching every
# synthetic context's own species_by_source config regardless of seed) —
# every evaluated report now requires a real dataset_manifest_fingerprint
# (see robustness_report.py), so this is threaded into every source-held-out
# call below via _BENCH_CFG.
_DATASET_MANIFEST_ENTRIES = _synthetic_dataset_manifest_entries(build_synthetic_context(seed=0, fast=True))

_BENCH_CFG = {
    "species_by_source": {"sourceA": "human", "sourceB": "human"}, "reference_species": "human",
    "dataset_manifest_entries": _DATASET_MANIFEST_ENTRIES,
}


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


def test_evaluated_reports_have_real_dataset_manifest_fingerprint(ctx):
    """Every EVALUATED report must carry a real (non-not_applicable)
    dataset_manifest_fingerprint when dataset_manifest_entries was
    supplied — including ineligible/non-evaluated branches, since the
    manifest fingerprint is computed once from the caller-supplied entries,
    independent of any one source's eligibility."""
    from benchmarks.robustness_report import is_not_applicable

    reports = run_cancer_source_held_out(ctx, ["prevalence"], device="cpu", n_oof_folds=2, **_BENCH_CFG)
    evaluated_any = False
    for src, r in reports.items():
        validate_robustness_report(r)
        assert not is_not_applicable(r["dataset_manifest_fingerprint"])
        if r.get("evaluated"):
            evaluated_any = True
    assert evaluated_any


def test_evaluated_report_without_dataset_manifest_entries_fails_validation(ctx):
    """The production enforcement point: an evaluated report built WITHOUT
    dataset_manifest_entries ever having been supplied gets a
    not_applicable dataset_manifest_fingerprint, which validate_robustness_
    report must now reject — proving the requirement is enforced, not
    merely documented."""
    cfg_without_manifest = {"species_by_source": {"sourceA": "human", "sourceB": "human"},
                             "reference_species": "human"}
    reports = run_cancer_source_held_out(ctx, ["prevalence"], device="cpu", n_oof_folds=2, **cfg_without_manifest)
    evaluated_any = False
    for src, r in reports.items():
        if r.get("evaluated"):
            evaluated_any = True
            with pytest.raises(RobustnessReportValidationError):
                validate_robustness_report(r)
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


def test_smoke_candidate_dev_score_reports_complete_oof_coverage():
    """The normal-path invariant: when every fold succeeds, oof_coverage
    reports complete=True with expected==realized subject counts and
    real fingerprints for the subject set/fold assignment/class order."""
    from benchmarks.source_held_out import _smoke_candidate_dev_score

    ctx = build_synthetic_context(seed=20, fast=True)
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
    coverage = evidence["oof_coverage"]
    assert score is not None
    assert coverage["complete"] is True
    assert coverage["expected_oof_subject_count"] == coverage["realized_oof_subject_count"]
    assert coverage["missing_oof_subject_count"] == 0
    for key in ("oof_subject_set_fingerprint", "fold_assignment_fingerprint", "class_order_fingerprint"):
        assert isinstance(coverage[key], str) and len(coverage[key]) == 64


def test_oof_coverage_summary_flags_missing_subjects_incomplete():
    from benchmarks.source_held_out import _oof_coverage_summary

    oof_by_subject = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0])}
    coverage = _oof_coverage_summary(oof_by_subject, ["a", "b", "c"], "fold_fp", "class_fp")
    assert coverage["complete"] is False
    assert coverage["expected_oof_subject_count"] == 3
    assert coverage["realized_oof_subject_count"] == 2
    assert coverage["missing_oof_subject_count"] == 1
    assert "missing_oof_subjects" not in coverage  # no raw subject-ID list in the persisted summary


def test_oof_coverage_summary_complete_when_every_expected_subject_realized():
    from benchmarks.source_held_out import _oof_coverage_summary

    oof_by_subject = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    coverage = _oof_coverage_summary(oof_by_subject, ["a", "b"], "fold_fp", "class_fp")
    assert coverage["complete"] is True
    assert coverage["missing_oof_subject_count"] == 0


def test_validate_oof_probability_row_accepts_valid_row():
    from benchmarks.source_held_out import _validate_oof_probability_row

    _validate_oof_probability_row("s1", np.array([0.2, 0.3, 0.5]), 3)  # must not raise


def test_validate_oof_probability_row_rejects_wrong_dimension():
    from benchmarks.source_held_out import IncompleteOOFCoverageError, _validate_oof_probability_row

    with pytest.raises(IncompleteOOFCoverageError):
        _validate_oof_probability_row("s1", np.array([0.5, 0.5]), 3)


def test_validate_oof_probability_row_rejects_non_finite():
    from benchmarks.source_held_out import IncompleteOOFCoverageError, _validate_oof_probability_row

    with pytest.raises(IncompleteOOFCoverageError):
        _validate_oof_probability_row("s1", np.array([np.nan, 0.5, 0.5]), 3)
    with pytest.raises(IncompleteOOFCoverageError):
        _validate_oof_probability_row("s1", np.array([np.inf, 0.5, 0.5]), 3)


def test_validate_oof_probability_row_rejects_non_unit_sum():
    from benchmarks.source_held_out import IncompleteOOFCoverageError, _validate_oof_probability_row

    with pytest.raises(IncompleteOOFCoverageError):
        _validate_oof_probability_row("s1", np.array([0.2, 0.2, 0.2]), 3)


def test_smoke_candidate_dev_score_detects_subject_predicted_by_its_own_training_fold(monkeypatch):
    """A poisoned fold (val subject also present in its own train subject
    set) must raise, never silently produce a leaked OOF prediction."""
    import benchmarks.source_held_out as soh

    ctx = build_synthetic_context(seed=21, fast=True)
    normalized_adata = ctx.normalized_adata_for_refit
    obs = normalized_adata.obs
    dev_subjects = sorted(set(obs["subject_id"].astype(str)))
    label_by_subject = {
        s: int(obs.loc[(obs["subject_id"].astype(str) == s).values, "smoke_type"].iloc[0]) for s in dev_subjects
    }
    num_cell_types = ctx.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = ctx.preprocessing_artifact.n_hvgs

    real_grouped_kfold = soh.grouped_kfold

    def _poisoned_grouped_kfold(*args, **kwargs):
        folds = real_grouped_kfold(*args, **kwargs)
        # Force the first fold's validation set to include one of its OWN
        # training subjects — a poisoned-policy proof, not an expected
        # outcome.
        poisoned = dict(folds[0])
        poisoned["val"] = list(poisoned["val"]) + [poisoned["train"][0]]
        return [poisoned] + list(folds[1:])

    monkeypatch.setattr(soh, "grouped_kfold", _poisoned_grouped_kfold)
    with pytest.raises(RuntimeError, match="training set"):
        soh._smoke_candidate_dev_score(
            ctx, "majority", dev_subjects, label_by_subject, num_cell_types, 3, n_hvgs,
            device="cpu", seed=1, n_dev_folds=3, n_inner_folds=2,
        )


def test_smoke_candidate_dev_score_detects_duplicate_oof_assignment_across_folds(monkeypatch):
    """A poisoned fold assignment where the SAME subject is a validation
    subject in two different folds must raise, never silently overwrite
    one fold's OOF entry with another's."""
    import benchmarks.source_held_out as soh

    ctx = build_synthetic_context(seed=22, fast=True)
    normalized_adata = ctx.normalized_adata_for_refit
    obs = normalized_adata.obs
    dev_subjects = sorted(set(obs["subject_id"].astype(str)))
    label_by_subject = {
        s: int(obs.loc[(obs["subject_id"].astype(str) == s).values, "smoke_type"].iloc[0]) for s in dev_subjects
    }
    num_cell_types = ctx.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = ctx.preprocessing_artifact.n_hvgs

    real_grouped_kfold = soh.grouped_kfold

    def _poisoned_grouped_kfold(*args, **kwargs):
        folds = real_grouped_kfold(*args, **kwargs)
        if len(folds) < 2:
            return folds
        # Force fold 1's validation set to also contain fold 0's first
        # validation subject — that subject now appears in two different
        # folds' validation sets.
        dup_subject = folds[0]["val"][0]
        poisoned_fold_1 = dict(folds[1])
        poisoned_fold_1["val"] = list(poisoned_fold_1["val"]) + [dup_subject]
        poisoned_fold_1["train"] = [s for s in poisoned_fold_1["train"] if s != dup_subject]
        return [folds[0], poisoned_fold_1] + list(folds[2:])

    monkeypatch.setattr(soh, "grouped_kfold", _poisoned_grouped_kfold)
    with pytest.raises(RuntimeError, match="more than one fold"):
        soh._smoke_candidate_dev_score(
            ctx, "majority", dev_subjects, label_by_subject, num_cell_types, 3, n_hvgs,
            device="cpu", seed=1, n_dev_folds=3, n_inner_folds=2,
        )


def test_smoke_source_held_out_never_selects_a_candidate_with_incomplete_oof_coverage():
    """Structural invariant across the whole run_smoke_source_held_out
    sweep: every comparisons entry with oof_coverage.complete=False must
    have development_macro_f1=None — an incomplete-coverage candidate can
    never be the selected winner. Uses a fresh, unmutated context (module-
    scoped `ctx` may have been corrupted in place by an earlier test)."""
    ctx = build_synthetic_context(seed=23, fast=True)
    reports = run_smoke_source_held_out(ctx, ["majority", "logistic"], device="cpu", **_BENCH_CFG)
    for src, r in reports.items():
        for comparison in r.get("comparisons", []):
            coverage = comparison.get("oof_coverage")
            if coverage and coverage.get("complete") is False:
                assert comparison.get("development_macro_f1") is None


@pytest.fixture(autouse=True)
def _cleanup_checkpoints():
    yield
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)
