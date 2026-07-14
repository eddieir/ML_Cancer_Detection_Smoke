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
