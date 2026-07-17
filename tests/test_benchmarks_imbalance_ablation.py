"""
Tests for benchmarks/imbalance_ablation.py — the development-only comparison
of Phase 2 smoke-imbalance strategies over grouped subject CV.
"""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.imbalance_ablation import (
    IMBALANCE_ABLATION_STRATEGIES,
    read_imbalance_ablation_artifact,
    run_smoke_imbalance_ablation,
    write_imbalance_ablation_artifact,
)
from benchmarks.runner import build_synthetic_context


def test_unknown_strategy_name_raises():
    ctx = build_synthetic_context(seed=42, fast=True)
    with pytest.raises(ValueError, match="unknown strategy"):
        run_smoke_imbalance_ablation(ctx, strategies=["not_a_real_strategy"])


def test_empty_strategy_list_raises():
    ctx = build_synthetic_context(seed=42, fast=True)
    with pytest.raises(ValueError, match="non-empty"):
        run_smoke_imbalance_ablation(ctx, strategies=[])


def test_all_five_default_strategies_are_defined():
    assert len(IMBALANCE_ABLATION_STRATEGIES) == 5
    assert "natural_no_weight" in IMBALANCE_ABLATION_STRATEGIES
    assert "subject_balanced_focal" in IMBALANCE_ABLATION_STRATEGIES


def test_ablation_never_touches_test_subjects():
    ctx = build_synthetic_context(seed=42, fast=True)
    test_subjects = set(ctx.subjects_for("test"))
    dev_subjects = set(ctx.subjects_for("train")) | set(ctx.subjects_for("val"))
    assert test_subjects, "sanity: context actually has a nonempty test split"
    assert test_subjects.isdisjoint(dev_subjects)

    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "subject_balanced_no_weight"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    assert report["development_only"] is True
    # Every fold's own fit_subject provenance (via preprocessing_fingerprint,
    # cross-checked against the dev-only pool grouped_kfold partitions from)
    # can only ever have been built from dev_subjects — grouped_kfold itself
    # is called on ctx.subjects_for("train")+("val") only (see
    # run_smoke_imbalance_ablation's pool_subjects), so this is a structural
    # guarantee, not a per-fold string search.
    for r in report["results"].values():
        for fold in r["folds"]:
            assert set(fold["validation_fingerprint"]["subject_ids"]) <= dev_subjects


class _PoisonedTestAccessError(RuntimeError):
    """Raised the instant anything touches the poisoned test-data sentinel —
    see test_ablation_never_accesses_a_poisoned_test_dataset below."""


class _Poison:
    """A stand-in for real test data that raises immediately on ANY access:
    attribute lookup, indexing, iteration, or len(). Used to prove the
    ablation genuinely never touches test data, not merely that its subject
    IDs happen to be disjoint from train/val."""

    def __getattr__(self, name):
        raise _PoisonedTestAccessError(f"poisoned test data: attribute {name!r} was accessed")

    def __getitem__(self, item):
        raise _PoisonedTestAccessError(f"poisoned test data: indexed with {item!r}")

    def __iter__(self):
        raise _PoisonedTestAccessError("poisoned test data: iterated")

    def __len__(self):
        raise _PoisonedTestAccessError("poisoned test data: len() called")

    def __eq__(self, other):
        raise _PoisonedTestAccessError("poisoned test data: compared for equality")

    def __repr__(self):
        # repr() is used by test-runner failure messages / debuggers, not by
        # production code paths — must not itself count as "access".
        return "<_Poison test-data sentinel>"


def test_ablation_never_accesses_a_poisoned_test_dataset():
    """Replace test_cell_dataset/test_bags with sentinels that raise on any
    access whatsoever, then run the full development-only ablation and
    prove it completes without ever triggering the sentinel — a much
    stronger guarantee than checking that subject-ID sets are disjoint."""
    ctx = build_synthetic_context(seed=42, fast=True)
    poisoned = dataclasses.replace(ctx, test_cell_dataset=_Poison(), test_bags=_Poison())

    report = run_smoke_imbalance_ablation(
        poisoned, strategies=["natural_no_weight", "subject_balanced_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    assert report["development_only"] is True
    assert len(report["results"]["natural_no_weight"]["folds"]) == 2


def test_ablation_reports_paired_fold_differences_against_baseline():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "natural_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    assert report["baseline_strategy"] == "natural_no_weight"
    assert "natural_inverse_frequency" in report["paired_fold_differences"]
    diffs = report["paired_fold_differences"]["natural_inverse_frequency"]
    assert len(diffs) == 2  # n_folds=2, one seed


def test_ablation_uses_identical_folds_across_strategies():
    """Every strategy's fold records must reference the SAME
    preprocessing_fingerprint per (seed, fold) — proving they trained on
    identical fold data, only the imbalance strategy differed."""
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "subject_balanced_no_weight"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    fps_by_strategy = {}
    for name, r in report["results"].items():
        fps_by_strategy[name] = [(f["seed"], f["fold"], f["preprocessing_fingerprint"]) for f in r["folds"]]
    names = list(fps_by_strategy)
    assert fps_by_strategy[names[0]] == fps_by_strategy[names[1]]


def test_ablation_strategy_config_is_recorded_per_fold():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["subject_balanced_focal"], n_folds=2, seeds=[42],
        device="cpu", max_cells_per_subject=50,
    )
    fold = report["results"]["subject_balanced_focal"]["folds"][0]
    assert fold["strategy_config"]["loss"] == "focal"
    assert fold["strategy_config"]["sampler"] == "subject_balanced"


# ─── Blocker 2: validation is never capped/mutated ─────────────────────────────

def test_validation_fingerprint_identical_across_every_strategy_in_a_fold():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=list(IMBALANCE_ABLATION_STRATEGIES), n_folds=2, seeds=[42],
        device="cpu", max_cells_per_subject=5,  # smaller than synthetic's 10 cells/subject -> cap bites train
    )
    by_key = {}
    for name, r in report["results"].items():
        for f in r["folds"]:
            key = (f["seed"], f["fold"])
            by_key.setdefault(key, []).append(f["validation_fingerprint"])
    for key, fingerprints in by_key.items():
        first = fingerprints[0]
        assert all(fp == first for fp in fingerprints), f"validation fingerprint diverged across strategies for {key}"


def test_only_training_data_is_capped_not_validation():
    ctx = build_synthetic_context(seed=42, fast=True)
    max_cells = 5  # synthetic context has 10 cells/subject when fast=True
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight"], n_folds=2, seeds=[42],
        device="cpu", max_cells_per_subject=max_cells,
    )
    for f in report["results"]["natural_no_weight"]["folds"]:
        # train WAS capped (n_train_cells_capped <= n_train_cells_before_cap,
        # and strictly less whenever the fold's subjects exceed max_cells)
        assert f["n_train_cells_capped"] <= f["n_train_cells_before_cap"]
        # validation cell count equals exactly the natural fold population —
        # never reduced to at-most max_cells-per-subject.
        assert f["n_val_cells"] == sum(f["validation_fingerprint"]["subject_counts"].values())
        per_subject_val_counts = f["validation_fingerprint"]["subject_counts"].values()
        # at least one validation subject exceeds max_cells (proving no cap
        # was silently applied to validation) — synthetic fast context uses
        # 10 cells/subject uniformly, well above max_cells=5.
        assert any(c > max_cells for c in per_subject_val_counts)


def test_train_val_test_subjects_pairwise_disjoint():
    ctx = build_synthetic_context(seed=42, fast=True)
    train = set(ctx.subjects_for("train"))
    val = set(ctx.subjects_for("val"))
    test = set(ctx.subjects_for("test"))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_no_validation_duplication_from_sampler():
    """Validation loaders never use the subject-balanced sampler — a
    strategy using it in training must still report the exact same (not
    inflated) n_val_cells as a strategy using plain shuffling."""
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "subject_balanced_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    counts = {
        name: [f["n_val_cells"] for f in r["folds"]]
        for name, r in report["results"].items()
    }
    names = list(counts)
    assert counts[names[0]] == counts[names[1]]


# ─── Issue 5: subject-level primary metrics ────────────────────────────────────

def test_fold_record_has_subject_level_and_cell_level_diagnostic_keys():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight"], n_folds=2, seeds=[42],
        device="cpu", max_cells_per_subject=50,
    )
    fold = report["results"]["natural_no_weight"]["folds"][0]
    assert "subject_level" in fold
    assert "cell_level_diagnostic" in fold
    for key in ("macro_f1", "balanced_accuracy", "per_class", "confusion_matrix"):
        assert key in fold["subject_level"]
    assert fold["subject_weighted_macro_f1"] == fold["subject_level"]["macro_f1"]


def test_subject_level_support_counts_subjects_not_cells():
    """Directly exercise the metrics function: a subject with far more
    cells than another must not inflate that class's support beyond the
    number of SUBJECTS in that class."""
    import numpy as np
    from benchmarks.metrics import subject_weighted_full_smoke_metrics_report

    subject_ids = ["big"] * 500 + ["small"] * 5
    y_true = np.array([0] * 500 + [0] * 5)
    y_pred = np.array([0] * 500 + [0] * 5)
    report = subject_weighted_full_smoke_metrics_report(y_true, y_pred, subject_ids, num_classes=2)
    assert report["per_class"]["0"]["support"] == 2  # two SUBJECTS, not 505 cells


def test_subject_level_confusion_matrix_class_ordering_is_stable():
    import numpy as np
    from benchmarks.metrics import subject_weighted_full_smoke_metrics_report

    subject_ids = ["a", "b", "c"]
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    r1 = subject_weighted_full_smoke_metrics_report(y_true, y_pred, subject_ids, num_classes=3)
    r2 = subject_weighted_full_smoke_metrics_report(y_true, y_pred, list(reversed(subject_ids)),
                                                      num_classes=3)
    # confusion_matrix is always indexed by range(num_classes), independent
    # of the order subjects happen to appear in the input arrays.
    assert r1["confusion_matrix"] == r2["confusion_matrix"]


def test_subject_level_absent_classes_recorded_honestly():
    import numpy as np
    from benchmarks.metrics import subject_weighted_full_smoke_metrics_report

    subject_ids = ["a", "b"]
    y_true = np.array([0, 0])
    y_pred = np.array([0, 0])
    report = subject_weighted_full_smoke_metrics_report(y_true, y_pred, subject_ids, num_classes=3)
    assert report["classes_absent_from_targets"] == [1, 2]
    assert report["per_class"]["1"]["support"] == 0
    assert report["per_class"]["2"]["support"] == 0


def test_cell_level_diagnostic_can_differ_from_subject_level():
    """A subject with many more cells than another can shift the CELL-level
    numbers while the SUBJECT-level numbers stay one-vote-per-subject."""
    import numpy as np
    from benchmarks.metrics import full_smoke_metrics_report, subject_weighted_full_smoke_metrics_report

    subject_ids = ["big"] * 100 + ["small"] * 2
    y_true = np.array([0] * 100 + [1] * 2)
    y_pred = np.array([0] * 100 + [0] * 2)  # "small" subject's cells are all misclassified
    subj = subject_weighted_full_smoke_metrics_report(y_true, y_pred, subject_ids, num_classes=2)
    cell = full_smoke_metrics_report(y_true, y_pred, num_classes=2)
    assert subj["per_class"]["1"]["support"] == 1     # 1 subject
    assert cell["per_class"]["1"]["support"] == 2       # 2 cells
    assert subj != cell


# ─── Issue 6: honest paired strategy comparisons ───────────────────────────────

def test_comparisons_present_for_every_non_baseline_strategy():
    ctx = build_synthetic_context(seed=42, fast=True)
    strategies = ["natural_no_weight", "natural_inverse_frequency", "subject_balanced_no_weight"]
    report = run_smoke_imbalance_ablation(
        ctx, strategies=strategies, n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    assert set(report["comparisons"]) == {"natural_inverse_frequency", "subject_balanced_no_weight"}
    for cmp in report["comparisons"].values():
        for key in ("wins_a", "ties", "losses_a", "mean_diff", "median_diff",
                     "n_pairs", "n_missing_pairs", "independence_note", "summary"):
            assert key in cmp


def test_comparison_win_tie_loss_counts_are_deterministic_and_sum_to_pairs():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "natural_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    cmp = report["comparisons"]["natural_inverse_frequency"]
    assert cmp["wins_a"] + cmp["ties"] + cmp["losses_a"] == cmp["n_pairs"]


def test_comparison_reruns_deterministically():
    ctx1 = build_synthetic_context(seed=42, fast=True)
    ctx2 = build_synthetic_context(seed=42, fast=True)
    r1 = run_smoke_imbalance_ablation(
        ctx1, strategies=["natural_no_weight", "natural_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    r2 = run_smoke_imbalance_ablation(
        ctx2, strategies=["natural_no_weight", "natural_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    c1 = r1["comparisons"]["natural_inverse_frequency"]
    c2 = r2["comparisons"]["natural_inverse_frequency"]
    assert c1["wins_a"] == c2["wins_a"]
    assert c1["ties"] == c2["ties"]
    assert c1["losses_a"] == c2["losses_a"]
    assert c1["mean_diff"] == pytest.approx(c2["mean_diff"])


def test_single_seed_comparison_reports_insufficient_evidence():
    """With only 1 seed, summarize_comparison's own honesty rule applies:
    no independent-seed evidence yet, so meaningfully_better must be False
    regardless of the raw win fraction."""
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "natural_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    summary = report["comparisons"]["natural_inverse_frequency"]["summary"]
    assert summary["meaningfully_better"] is False


def test_comparison_independence_note_present_for_overlapping_folds():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "natural_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    note = report["comparisons"]["natural_inverse_frequency"]["independence_note"]
    assert "not" in note.lower() and "independent" in note.lower()


def test_comparison_schema_is_json_serializable():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "natural_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    json.dumps(report["comparisons"], default=str)  # must not raise


# ─── Issue 9: reproducible artifact persistence ────────────────────────────────

def test_write_and_read_imbalance_ablation_artifact_round_trip():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "subject_balanced_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run_001"
        run_dir.mkdir()
        json_path = write_imbalance_ablation_artifact(run_dir, ctx, report, synthetic=True)
        assert json_path.exists()
        loaded = read_imbalance_ablation_artifact(json_path)

    assert loaded["schema_version"] >= 1
    assert loaded["synthetic"] is True
    assert loaded["development_only"] is True
    assert loaded["frozen_test_data_accessed"] is False
    assert loaded["baseline_strategy"] == report["baseline_strategy"]
    assert loaded["config_fingerprint"] == ctx.config_fingerprint
    assert "results" in loaded and "comparisons" in loaded


def test_artifact_round_trip_preserves_realized_sampler_diagnostics_exactly():
    ctx = build_synthetic_context(seed=42, fast=True)
    report = run_smoke_imbalance_ablation(
        ctx, strategies=["natural_no_weight", "subject_balanced_inverse_frequency"],
        n_folds=2, seeds=[42], device="cpu", max_cells_per_subject=50,
    )
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run_001"
        run_dir.mkdir()
        json_path = write_imbalance_ablation_artifact(run_dir, ctx, report, synthetic=True)
        loaded = read_imbalance_ablation_artifact(json_path)

    shuffle_fold = loaded["results"]["natural_no_weight"]["folds"][0]
    balanced_fold = loaded["results"]["subject_balanced_inverse_frequency"]["folds"][0]
    # shuffle-based strategy must never fabricate subject-balanced realized
    # sampler diagnostics — hyperparameters.smoke_sampling_diagnostics stays
    # null (see NeuralSmokeAdapter.metadata / Trainer._train_cell_loader).
    assert shuffle_fold["hyperparameters"]["smoke_sampling_diagnostics"] is None
    balanced_diag = balanced_fold["hyperparameters"]["smoke_sampling_diagnostics"]
    assert balanced_diag is not None
    assert balanced_diag["complete"] is True
    assert balanced_diag["realized_total_samples"] == sum(balanced_diag["realized_batch_sizes"])
    assert sum(balanced_diag["realized_cells_per_subject"].values()) == balanced_diag["realized_total_samples"]
    for subject_counts in balanced_diag["realized_subject_counts_per_batch"]:
        assert max(subject_counts.values()) <= balanced_diag["cells_per_subject_cap"]


def test_artifact_identity_changes_with_imbalance_strategy_config():
    """A config change that changes trained-model behavior (here: which
    strategies were compared/what the resolved config was) must be
    reflected by a different config_fingerprint, per the existing
    context.config_fingerprint mechanism (no new plumbing needed)."""
    ctx_a = build_synthetic_context(seed=42, fast=True)
    ctx_b = build_synthetic_context(seed=42, fast=True)
    ctx_b.config = dict(ctx_b.config)
    ctx_b.config["train"] = dict(ctx_b.config.get("train", {}),
                                  smoke_imbalance={"sampler": "subject_balanced"})
    assert ctx_a.config_fingerprint != ctx_b.config_fingerprint
