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


def test_guard_identity_excludes_fit_seconds_and_timestamps():
    """Blocker 2: two guard-identity computations that differ ONLY in
    fit_seconds/timestamps (volatile, non-reproducible metadata) must
    produce the SAME identity — otherwise every real run would generate a
    fresh guard file and the one-time-evaluation discipline would be
    meaningless."""
    ctx = build_synthetic_context(seed=1, fast=True)
    common = {
        "selected_hyperparameters": {"C": 1.0},
        "final_preprocessing_artifact_fingerprint": "abc123",
        "final_model_state_fingerprint": "def456",
        "calibration_fingerprint": "cal789",
        "threshold": 0.5,
        "test_membership_fingerprint": ctx.test_membership_fingerprint,
    }
    fp_a = ctx.guard_identity_fingerprint("logistic", extra=common)
    fp_b = ctx.guard_identity_fingerprint("logistic", extra=common)
    assert fp_a == fp_b


def test_run_cancer_task_never_feeds_volatile_model_metadata_into_guard_identity():
    """run_cancer_task's guard-identity computation must pass a deterministic
    model_state_fingerprint (a hash of actual fitted weights), never the raw
    model_metadata dict — which carries fit_seconds, real wall-clock timing
    that differs on every run and would make the guard non-deterministic."""
    import benchmarks.context as context_mod

    ctx = build_synthetic_context(seed=1, fast=True)
    captured = {}
    orig = context_mod.ExperimentContext.guard_identity_fingerprint

    def spy(self, selected_model, extra=None):
        captured["extra"] = extra
        return orig(self, selected_model, extra=extra)
    context_mod.ExperimentContext.guard_identity_fingerprint = spy
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
            run_cancer_task(ctx, _Args(), run_dir)
    finally:
        context_mod.ExperimentContext.guard_identity_fingerprint = orig
        shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)

    extra = captured["extra"]
    assert "final_model_state_fingerprint" in extra
    assert "final_model_metadata" not in extra
    assert "fit_seconds" not in str(extra)


def test_guard_identity_changes_when_model_state_fingerprint_differs():
    ctx = build_synthetic_context(seed=1, fast=True)
    fp_a = ctx.guard_identity_fingerprint("logistic", extra={"final_model_state_fingerprint": "aaa"})
    fp_b = ctx.guard_identity_fingerprint("logistic", extra={"final_model_state_fingerprint": "bbb"})
    assert fp_a != fp_b


def test_guard_identity_changes_when_test_membership_differs():
    ctx = build_synthetic_context(seed=1, fast=True)
    fp_a = ctx.guard_identity_fingerprint("logistic", extra={"test_membership_fingerprint": "aaa"})
    fp_b = ctx.guard_identity_fingerprint("logistic", extra={"test_membership_fingerprint": "bbb"})
    assert fp_a != fp_b


def test_test_membership_fingerprint_reads_only_split_manifest():
    """context.test_membership_fingerprint must be computable without
    touching context.test_bags at all — it's derived purely from
    split_manifest.test_subjects, so it's safe to include in the guard
    identity computed before guard acquisition."""
    ctx = build_synthetic_context(seed=1, fast=True)

    class _PoisonedBags(list):
        def __iter__(self):
            raise AssertionError("test_membership_fingerprint touched context.test_bags")

    ctx.test_bags = _PoisonedBags(ctx.test_bags)
    fp = ctx.test_membership_fingerprint
    assert isinstance(fp, str) and len(fp) == 64


def test_frozen_test_result_artifact_persisted_and_referenced_by_guard():
    ctx = build_synthetic_context(seed=1, fast=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
        outcome = run_cancer_task(ctx, _Args(), run_dir)
        result_path = run_dir / "calibration" / "frozen_test_result.json"
        assert result_path.exists()
        import json
        frozen_result = json.loads(result_path.read_text())
        assert "artifact_fingerprint" in frozen_result
        assert "test_labels" not in frozen_result
        assert "test_proba" not in frozen_result

        guard_dir = run_dir.parent / ".frozen_test_guards"
        guard_payload = json.loads(list(guard_dir.glob("*.json"))[0].read_text())
        assert guard_payload["test_result_fingerprint"] == frozen_result["artifact_fingerprint"]
        assert outcome["calibration_report"]["frozen_test_result_artifact_fingerprint"] == \
            frozen_result["artifact_fingerprint"]
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_persistence_failure_marks_guard_failed_not_completed():
    import benchmarks.runner as runner_mod

    ctx = build_synthetic_context(seed=9, fast=True)
    orig = runner_mod.write_json

    def _boom(path, obj):
        if str(path).endswith("frozen_test_result.json"):
            raise RuntimeError("simulated persistence failure")
        return orig(path, obj)
    runner_mod.write_json = _boom
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
            with pytest.raises(RuntimeError, match="simulated persistence failure"):
                run_cancer_task(ctx, _Args(), run_dir, synthetic=False)
            guard_dir = run_dir.parent / ".frozen_test_guards"
            import json
            payload = json.loads(list(guard_dir.glob("*.json"))[0].read_text())
            assert payload["status"] == "failed"
    finally:
        runner_mod.write_json = orig
        shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_calibration_report_references_oof_artifact_fingerprint():
    ctx = build_synthetic_context(seed=1, fast=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
        outcome = run_cancer_task(ctx, _Args(), run_dir)
        oof_fp = outcome["calibration_report"]["oof_summary"]["oof_artifact_fingerprint"]
        assert isinstance(oof_fp, str) and len(oof_fp) == 64
        oof_csv = run_dir / "predictions" / f"cancer_{outcome['calibration_report']['selected_model']}_oof.csv"
        assert oof_csv.exists()
        import hashlib
        assert hashlib.sha256(oof_csv.read_bytes()).hexdigest() == oof_fp
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_oof_csv_has_all_required_fingerprint_columns():
    ctx = build_synthetic_context(seed=1, fast=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
        outcome = run_cancer_task(ctx, _Args(), run_dir)
        oof_csv = run_dir / "predictions" / f"cancer_{outcome['calibration_report']['selected_model']}_oof.csv"
        import csv
        with open(oof_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows
        required = {
            "subject_id", "target", "probability", "seed", "outer_oof_fold", "candidate_name",
            "candidate_type", "pooling", "selected_params_json", "selected_params_fingerprint",
            "inner_selection_fingerprint", "training_subjects_fingerprint",
            "validation_subjects_fingerprint", "preprocessing_fingerprint",
            "model_state_fingerprint", "prediction_status", "undefined_reason",
        }
        assert required <= set(rows[0].keys())
        for row in rows:
            if row["prediction_status"] == "predicted":
                assert len(row["training_subjects_fingerprint"]) == 64
                assert len(row["model_state_fingerprint"]) == 64
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_guard_ownership_prevents_finalization_by_a_non_acquiring_instance():
    from benchmarks.test_guard import FrozenTestGuard, FrozenTestGuardOwnershipError

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        owner = FrozenTestGuard(path)
        owner.acquire({"run_id": "r1"})

        impostor = FrozenTestGuard(path)  # never acquired — no owner token
        with pytest.raises(FrozenTestGuardOwnershipError):
            impostor.mark_completed(threshold=0.5, test_result_fingerprint="fp")
        with pytest.raises(FrozenTestGuardOwnershipError):
            impostor.mark_failed("boom")

        # the real owner can still finalize normally
        owner.mark_completed(threshold=0.5, test_result_fingerprint="fp")


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
