"""
Regression tests for PR7 requirement 5: durable, concurrency-safe one-time
frozen-test evaluation guard. These simulate restarts/concurrency at the
filesystem level without ever constructing or exposing real test labels.
"""
import json
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import benchmarks.atomic_io as atomic_io
from benchmarks.test_guard import (
    FrozenTestAlreadyEvaluatedError,
    FrozenTestGuard,
    FrozenTestGuardCorruptedError,
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


# --- Issue 1 (PR7-5): acquire() must never leave a truncated/corrupted
# guard file behind, regardless of where in the write/fsync/close sequence
# a failure occurs, and corruption must fail closed rather than be treated
# as "no guard". -------------------------------------------------------


def test_acquire_survives_short_writes_across_multiple_os_write_calls(monkeypatch):
    real_write = os.write
    call_count = {"n": 0}

    def flaky_write(fd, data):
        call_count["n"] += 1
        if call_count["n"] <= 3 and len(data) > 1:
            return real_write(fd, data[: max(1, len(data) // 3)])
        return real_write(fd, data)

    monkeypatch.setattr(atomic_io.os, "write", flaky_write)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard = FrozenTestGuard(path)
        guard.acquire({"run_id": "short-write-run"})
        assert call_count["n"] > 1
        data = json.loads(path.read_text())
        assert data["run_identity"]["run_id"] == "short-write-run"
        assert guard._owner_token is not None


def test_acquire_retries_interrupted_error(monkeypatch):
    real_write = os.write
    state = {"raised": False}

    def interrupt_once(fd, data):
        if not state["raised"]:
            state["raised"] = True
            raise InterruptedError()
        return real_write(fd, data)

    monkeypatch.setattr(atomic_io.os, "write", interrupt_once)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard = FrozenTestGuard(path)
        guard.acquire({"run_id": "interrupted-run"})
        assert state["raised"] is True
        assert json.loads(path.read_text())["status"] == "in_progress"


def test_acquire_raises_and_leaves_no_file_on_zero_byte_progress(monkeypatch):
    monkeypatch.setattr(atomic_io.os, "write", lambda fd, data: 0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard = FrozenTestGuard(path)
        with pytest.raises(OSError):
            guard.acquire({"run_id": "zero-progress-run"})
        assert not path.exists()
        assert guard._owner_token is None


def test_acquire_cleans_up_incomplete_guard_on_write_failure(monkeypatch):
    def failing_write(fd, data):
        raise OSError("simulated write failure")

    monkeypatch.setattr(atomic_io.os, "write", failing_write)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard = FrozenTestGuard(path)
        with pytest.raises(OSError):
            guard.acquire({"run_id": "write-fail-run"})
        assert not path.exists()
        assert guard._owner_token is None


def test_acquire_cleans_up_incomplete_guard_on_fsync_failure(monkeypatch):
    def failing_fsync(fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(atomic_io.os, "fsync", failing_fsync)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard = FrozenTestGuard(path)
        with pytest.raises(OSError):
            guard.acquire({"run_id": "fsync-fail-run"})
        assert not path.exists()
        assert guard._owner_token is None


def test_losing_process_never_deletes_the_winning_guard_file():
    """A process that loses the acquisition race (because O_CREAT|O_EXCL
    fails with FileExistsError) must never touch the winner's file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        winner = FrozenTestGuard(path)
        winner.acquire({"run_id": "winner"})
        original_bytes = path.read_bytes()

        loser = FrozenTestGuard(path)
        with pytest.raises(FrozenTestInProgressError):
            loser.acquire({"run_id": "loser"})

        assert path.read_bytes() == original_bytes
        assert loser._owner_token is None


def test_malformed_guard_json_raises_corruption_error_not_treated_as_absent():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        path.write_text("{not valid json")
        guard = FrozenTestGuard(path)
        with pytest.raises(FrozenTestGuardCorruptedError):
            guard.acquire({"run_id": "should-never-run"})


def test_successful_acquisition_writes_complete_valid_json_with_owner_token():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "guard.json"
        guard = FrozenTestGuard(path)
        guard.acquire({"run_id": "complete-run"})
        data = json.loads(path.read_text())
        assert data["status"] == "in_progress"
        assert data["owner_token"] == guard._owner_token
        assert len(data["owner_token"]) == 32


def _mp_acquire_attempt(path_str, run_id, result_queue):
    """Top-level (picklable) worker for the real multiprocessing race test."""
    import benchmarks.test_guard as tg

    guard = tg.FrozenTestGuard(path_str)
    try:
        guard.acquire({"run_id": run_id})
        result_queue.put(("ok", run_id, guard._owner_token, ""))
    except Exception as exc:  # noqa: BLE001 — reporting outcome to parent
        result_queue.put(("error", run_id, type(exc).__name__, str(exc)))


def test_real_multiprocess_acquisition_race_has_exactly_one_winner():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "guard.json")
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        n_workers = 6
        procs = [
            ctx.Process(target=_mp_acquire_attempt, args=(path, f"run_{i}", result_queue))
            for i in range(n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert not p.is_alive(), "worker process did not terminate cleanly"

        results = [result_queue.get(timeout=5) for _ in range(n_workers)]
        winners = [r for r in results if r[0] == "ok"]
        losers = [r for r in results if r[0] == "error"]

        assert len(winners) == 1, f"expected exactly one winner, got {winners}"
        assert len(losers) == n_workers - 1
        assert all(
            exc_type == "FrozenTestInProgressError" for _, _, exc_type, _ in losers
        ), f"unexpected loser error types: {losers}"

        on_disk = json.loads(Path(path).read_text())
        assert on_disk["status"] == "in_progress"
        assert on_disk["owner_token"] == winners[0][2]
        assert on_disk["run_identity"]["run_id"] == winners[0][1]
