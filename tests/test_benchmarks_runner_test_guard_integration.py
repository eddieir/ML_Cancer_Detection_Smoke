"""
Integration tests for PR7 blocker 3: the durable frozen-test guard is
MANDATORY for every non-synthetic run_cancer_task call — it is no longer
opt-in via frozen_test_guard_dir. A safe default guard location is derived
from the run's own output root when that config key is absent. Disabling it
is permitted ONLY for --synthetic runs via an explicit config flag; a
non-synthetic run that tries to disable it is refused outright.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.runner import build_synthetic_context, new_run_dir, run_cancer_task
from benchmarks.test_guard import FrozenTestAlreadyEvaluatedError, FrozenTestGuardDisabledInRealModeError


class _Args:
    models = ["prevalence", "logistic"]
    cv_folds = 2
    seeds = [42]
    device = "cpu"
    pooling = None
    calibration = "auto"
    threshold_strategy = "youden"


def test_guard_dir_blocks_second_evaluation_of_same_context():
    ctx = build_synthetic_context(seed=1, fast=True)
    with tempfile.TemporaryDirectory() as tmp:
        guard_dir = str(Path(tmp) / "guards")
        ctx.config.setdefault("benchmarks", {})["frozen_test_guard_dir"] = guard_dir

        run_dir1 = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
        outcome1 = run_cancer_task(ctx, _Args(), run_dir1)
        assert outcome1["calibration_report"] is not None

        run_dir2 = new_run_dir(str(Path(tmp) / "runs"), run_id="run2")
        with pytest.raises(FrozenTestAlreadyEvaluatedError):
            run_cancer_task(ctx, _Args(), run_dir2)
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_guard_mandatory_by_default_blocks_second_evaluation_without_explicit_dir():
    """No frozen_test_guard_dir configured -> the guard is still created (a
    safe default location is derived from the run's own output root) and
    still blocks a second evaluation of the same underlying context."""
    ctx = build_synthetic_context(seed=1, fast=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir1 = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
        outcome1 = run_cancer_task(ctx, _Args(), run_dir1)
        assert outcome1["calibration_report"]["guard"]["enabled"] is True

        run_dir2 = new_run_dir(str(Path(tmp) / "runs"), run_id="run2")
        with pytest.raises(FrozenTestAlreadyEvaluatedError):
            run_cancer_task(ctx, _Args(), run_dir2)
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_different_run_dir_names_cannot_bypass_the_same_guard_identity():
    """A fresh --run-id (and therefore a different run_dir.name) must not
    let a second evaluation through if the underlying scientific identity
    (manifest/preprocessing/label-mapping/config/selected model) is
    unchanged — the guard is keyed by that identity, not by run_id."""
    ctx = build_synthetic_context(seed=2, fast=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir1 = new_run_dir(str(Path(tmp) / "runs"), run_id="completely_different_name_one")
        run_cancer_task(ctx, _Args(), run_dir1)
        run_dir2 = new_run_dir(str(Path(tmp) / "runs"), run_id="totally_unrelated_name_two")
        with pytest.raises(FrozenTestAlreadyEvaluatedError):
            run_cancer_task(ctx, _Args(), run_dir2)
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_real_mode_cannot_disable_the_guard():
    ctx = build_synthetic_context(seed=1, fast=True)
    ctx.config.setdefault("benchmarks", {})["disable_frozen_test_guard"] = True
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
        with pytest.raises(FrozenTestGuardDisabledInRealModeError):
            run_cancer_task(ctx, _Args(), run_dir, synthetic=False)
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_synthetic_mode_remains_repeatable_when_explicitly_disabled():
    ctx = build_synthetic_context(seed=1, fast=True)
    ctx.config.setdefault("benchmarks", {})["disable_frozen_test_guard"] = True
    with tempfile.TemporaryDirectory() as tmp:
        run_dir1 = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
        outcome1 = run_cancer_task(ctx, _Args(), run_dir1, synthetic=True)
        assert outcome1["calibration_report"]["guard"]["enabled"] is False

        run_dir2 = new_run_dir(str(Path(tmp) / "runs"), run_id="run2")
        outcome2 = run_cancer_task(ctx, _Args(), run_dir2, synthetic=True)  # must not raise
        assert outcome2["calibration_report"]["guard"]["enabled"] is False
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


# ─── Guard-before-test-access ordering (blocker 1) ─────────────────────────
#
# run_cancer_task is split into a development-only stage (candidate
# selection, hyperparameter search, OOF generation, calibration fitting, the
# final dev-pool fit) and a guarded stage that resolves test subjects/labels
# and evaluates the frozen test split. These tests assert that ordering
# directly against the real run_cancer_task call, not against a hand-built
# stand-in.

def test_test_subject_resolution_never_happens_before_guard_acquired():
    from benchmarks.context import ExperimentContext
    from benchmarks.test_guard import FrozenTestGuard

    ctx = build_synthetic_context(seed=7, fast=True)
    guard_acquired = {"v": False}

    orig_acquire = FrozenTestGuard.acquire
    def spy_acquire(self, *a, **kw):
        orig_acquire(self, *a, **kw)
        guard_acquired["v"] = True
    FrozenTestGuard.acquire = spy_acquire

    orig_subjects_for = ExperimentContext.subjects_for
    def spy_subjects_for(self, split):
        if split == "test" and not guard_acquired["v"]:
            raise AssertionError("context.subjects_for('test') was called before the frozen-test guard was acquired")
        return orig_subjects_for(self, split)
    ExperimentContext.subjects_for = spy_subjects_for

    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
            outcome = run_cancer_task(ctx, _Args(), run_dir, synthetic=True)
        assert outcome["calibration_report"]["test_result"] is not None
        assert guard_acquired["v"] is True
    finally:
        FrozenTestGuard.acquire = orig_acquire
        ExperimentContext.subjects_for = orig_subjects_for
        shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_development_fit_result_carries_no_test_subject_ids():
    """fit_final_candidate_on_dev_pool's return value (FittedFinalCandidate)
    has no field that could hold test-derived data — dev_subject_ids must be
    a subset of the development pool only."""
    from benchmarks.final_evaluation import fit_final_candidate_on_dev_pool

    ctx = build_synthetic_context(seed=7, fast=True)
    all_bags = list(ctx.train_bags) + list(ctx.val_bags)
    outcomes = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes.keys())
    test_subjects = set(ctx.subjects_for("test"))

    fitted = fit_final_candidate_on_dev_pool(
        ctx, "logistic", dev_subjects, outcomes, {},
        ctx.config["model"]["num_cell_types"], ctx.config["data"]["min_cells_per_subject"],
        ctx.preprocessing_artifact.n_hvgs, device="cpu", seed=42,
    )
    assert set(fitted.dev_subject_ids).isdisjoint(test_subjects)


def test_exception_during_test_evaluation_marks_guard_failed_not_completed():
    import benchmarks.runner as runner_mod

    ctx = build_synthetic_context(seed=8, fast=True)

    def _boom(*a, **kw):
        raise RuntimeError("simulated failure during frozen-test evaluation")
    orig = runner_mod.evaluate_frozen_test
    runner_mod.evaluate_frozen_test = _boom
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
            with pytest.raises(RuntimeError, match="simulated failure"):
                run_cancer_task(ctx, _Args(), run_dir, synthetic=False)
            # default_guard_dir is derived from run_dir.parent
            guard_dir = run_dir.parent / ".frozen_test_guards"
            guard_files = list(guard_dir.glob("*.json"))
            assert len(guard_files) == 1
            import json
            payload = json.loads(guard_files[0].read_text())
            assert payload["status"] == "failed"
            assert "test_result" not in payload
    finally:
        runner_mod.evaluate_frozen_test = orig
        shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_guard_identity_changes_when_selected_hyperparameters_differ():
    ctx = build_synthetic_context(seed=1, fast=True)
    fp_a = ctx.guard_identity_fingerprint("logistic", extra={"selected_hyperparameters": {"C": 1.0}})
    fp_b = ctx.guard_identity_fingerprint("logistic", extra={"selected_hyperparameters": {"C": 10.0}})
    assert fp_a != fp_b


def test_guard_file_never_contains_raw_test_labels_or_probabilities():
    ctx = build_synthetic_context(seed=1, fast=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
        run_cancer_task(ctx, _Args(), run_dir)
        guard_dir = run_dir.parent / ".frozen_test_guards"
        guard_files = list(guard_dir.glob("*.json"))
        assert len(guard_files) == 1
        import json
        payload = json.loads(guard_files[0].read_text())
        assert "test_labels" not in payload
        assert "test_proba" not in payload
        assert "test_result" not in payload
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)
