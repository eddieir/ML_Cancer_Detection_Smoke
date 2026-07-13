"""
Integration test: run_cancer_task's opt-in frozen_test_guard_dir wiring
actually creates a durable guard and refuses a second frozen-test evaluation
against the same context/config.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.runner import build_synthetic_context, new_run_dir, run_cancer_task
from benchmarks.test_guard import FrozenTestAlreadyEvaluatedError


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


def test_guard_disabled_by_default_allows_repeated_evaluation():
    """No frozen_test_guard_dir configured -> repeated invocations (the
    normal test/CI pattern) are NOT blocked."""
    ctx = build_synthetic_context(seed=1, fast=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir1 = new_run_dir(str(Path(tmp) / "runs"), run_id="run1")
        run_cancer_task(ctx, _Args(), run_dir1)
        run_dir2 = new_run_dir(str(Path(tmp) / "runs"), run_id="run2")
        run_cancer_task(ctx, _Args(), run_dir2)  # must not raise
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)
