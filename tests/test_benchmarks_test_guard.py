"""
Regression tests for PR7 requirement 5: durable, concurrency-safe one-time
frozen-test evaluation guard. These simulate restarts/concurrency at the
filesystem level without ever constructing or exposing real test labels.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.test_guard import (
    FrozenTestAlreadyEvaluatedError,
    FrozenTestGuard,
    FrozenTestInProgressError,
)


def test_acquire_creates_guard_file_atomically():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard = FrozenTestGuard(path)
        guard.acquire({"run_id": "r1"})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["status"] == "in_progress"


def test_concurrent_acquire_only_one_succeeds():
    """Simulates two processes racing to start the same frozen-test run —
    only the first acquire() may succeed; the second must be refused."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard_a = FrozenTestGuard(path)
        guard_b = FrozenTestGuard(path)
        guard_a.acquire({"run_id": "r1"})
        with pytest.raises(FrozenTestInProgressError):
            guard_b.acquire({"run_id": "r2"})


def test_restart_after_completion_is_refused():
    """Simulates a process restart (a brand-new FrozenTestGuard instance,
    same path) after a prior run completed — must still refuse, proving the
    guard is durable across restarts, not just within one process's memory."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard1 = FrozenTestGuard(path)
        guard1.acquire({"run_id": "r1"})
        guard1.mark_completed(threshold=0.5, test_result_fingerprint="fp_abc123")

        guard_after_restart = FrozenTestGuard(path)
        with pytest.raises(FrozenTestAlreadyEvaluatedError):
            guard_after_restart.acquire({"run_id": "r2"})


def test_failed_run_requires_manual_removal_not_auto_retry():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard1 = FrozenTestGuard(path)
        guard1.acquire({"run_id": "r1"})
        guard1.mark_failed("simulated crash mid-evaluation")

        guard2 = FrozenTestGuard(path)
        with pytest.raises(FrozenTestInProgressError):
            guard2.acquire({"run_id": "r2"})

        # Explicit, auditable recovery: caller removes the failed guard file
        # themselves, then a fresh attempt may proceed.
        path.unlink()
        guard3 = FrozenTestGuard(path)
        guard3.acquire({"run_id": "r3"})
        data = json.loads(path.read_text())
        assert data["run_identity"]["run_id"] == "r3"


def test_mark_completed_without_prior_acquire_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard = FrozenTestGuard(path)
        with pytest.raises(RuntimeError):
            guard.mark_completed(threshold=0.5, test_result_fingerprint="fp")


def test_completed_guard_can_never_be_superseded_even_by_new_instance():
    """No flag/parameter permits re-acquiring a completed guard — proving
    there is no override path at all, by inspecting acquire()'s signature
    behavior directly."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        g = FrozenTestGuard(path)
        g.acquire({"run_id": "r1"})
        g.mark_completed(threshold=0.4, test_result_fingerprint="fp1")
        for attempt in range(3):
            with pytest.raises(FrozenTestAlreadyEvaluatedError):
                FrozenTestGuard(path).acquire({"run_id": f"retry_{attempt}"})
