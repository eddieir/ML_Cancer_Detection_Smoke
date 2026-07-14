"""
benchmarks/test_guard.py — durable, concurrency-safe one-time frozen-test
evaluation guard.

Replaces a process-local-only `_used` boolean (which offers no protection
across process restarts, and none at all against two concurrent processes
racing to evaluate the same frozen test split) with a persistent guard file
created via an atomic filesystem operation. `os.open(path, O_CREAT | O_EXCL)`
either creates the file or raises FileExistsError — there is no window where
two processes can both believe they created it, on any POSIX filesystem
(this is the same primitive `flock`-free atomic lock files rely on).

Guard file contents record: run identity (see ExperimentContext.run_identity
— manifest/artifact/label-mapping/config fingerprints), the selected model,
the frozen threshold, a timestamp, and a status: "in_progress", "completed",
or "failed".

Recovery policy for a "failed" run (section 5's explicit requirement — must
not be able to expose test labels repeatedly): a failed guard may be
superseded by a NEW attempt only via `retry_after_failure=True`, which
requires the caller to have already deleted or renamed the failed guard file
out of the way — this module never auto-deletes a guard file itself, so a
failed run always leaves an auditable trace rather than quietly vanishing.
A "completed" guard can NEVER be superseded, regardless of any flag.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union


def default_guard_dir(output_root: Union[str, Path]) -> Path:
    """
    Safe default guard location derived purely from the immutable output
    root every run already has to supply — used whenever a real (non-
    synthetic) run does not explicitly configure
    benchmarks.frozen_test_guard_dir. Guard files here are keyed by
    ExperimentContext.guard_identity_fingerprint(), not by run_id, so this
    directory can safely be shared by every run against the same output
    root: two runs with different --run-id but identical scientific
    identity (manifest/preprocessing/label-mapping/config/selected model)
    still collide on the same guard file — see runner.py.
    """
    return Path(output_root) / ".frozen_test_guards"


class FrozenTestGuardDisabledInRealModeError(RuntimeError):
    """Raised when a real (non-synthetic) run's configuration attempts to
    disable the durable frozen-test guard — never permitted, regardless of
    how the request is phrased in config."""


class FrozenTestAlreadyEvaluatedError(RuntimeError):
    """Raised when a guard file already records a completed frozen-test run."""


class FrozenTestInProgressError(RuntimeError):
    """Raised when a guard file shows another process's run is (or was, at
    last unclean exit) in progress — never assumed safe to re-run without
    an explicit, auditable decision."""


class FrozenTestGuard:
    def __init__(self, guard_path: Union[str, Path]):
        self.guard_path = Path(guard_path)

    def _read(self) -> Optional[Dict]:
        if not self.guard_path.exists():
            return None
        with open(self.guard_path) as f:
            return json.load(f)

    def acquire(self, run_identity: Dict, selected_model: Optional[str] = None) -> None:
        """
        Atomically create the guard file in "in_progress" state. Raises:
          FrozenTestAlreadyEvaluatedError — a completed run already exists;
            refuses unconditionally, there is no override.
          FrozenTestInProgressError — an in_progress guard already exists
            (another process is running, or a previous process crashed
            mid-run without marking completed/failed); resolving this
            requires a human to inspect and explicitly remove the stale
            guard file, never an automatic override, since this module
            cannot distinguish "still running" from "crashed" by itself.
        """
        existing = self._read()
        if existing is not None:
            status = existing.get("status")
            if status == "completed":
                raise FrozenTestAlreadyEvaluatedError(
                    f"Frozen test guard at {self.guard_path} already records a COMPLETED "
                    f"evaluation (run_id={existing.get('run_identity', {}).get('run_id')}) — "
                    "refusing to evaluate the frozen test split again. This is by design: "
                    "test labels may be examined exactly once per frozen run."
                )
            if status == "in_progress":
                raise FrozenTestInProgressError(
                    f"Frozen test guard at {self.guard_path} shows status=in_progress "
                    f"(started {existing.get('started_at')}) — either another process is "
                    "currently evaluating this frozen test, or a previous process crashed "
                    "mid-run. Refusing to proceed automatically; inspect and remove the guard "
                    "file manually once you've confirmed no other process is running."
                )
            if status == "failed":
                raise FrozenTestInProgressError(
                    f"Frozen test guard at {self.guard_path} records a FAILED prior attempt — "
                    "a new attempt requires deleting/renaming this guard file first (an "
                    "explicit, auditable action), not an automatic retry that could silently "
                    "re-open test-label access. See FrozenTestGuard docstring for the recovery "
                    "policy."
                )

        payload = {
            "status": "in_progress", "run_identity": run_identity,
            "selected_model": selected_model, "started_at": time.time(),
        }
        self.guard_path.parent.mkdir(parents=True, exist_ok=True)
        # O_CREAT | O_EXCL: atomically fails with FileExistsError if the
        # file already exists — closes the race window between "check if
        # it exists" and "create it" that a plain open()/exists() pair has.
        fd = os.open(str(self.guard_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, json.dumps(payload, indent=2).encode("utf-8"))
        finally:
            os.close(fd)

    def mark_completed(self, threshold: float, test_result_fingerprint: str) -> None:
        """Overwrite the (already-acquired, in_progress) guard file with a
        terminal "completed" record. Once written, acquire() on this path
        can never succeed again."""
        existing = self._read()
        if existing is None or existing.get("status") != "in_progress":
            raise RuntimeError(
                f"FrozenTestGuard.mark_completed called without a matching in_progress "
                f"guard at {self.guard_path} — acquire() must be called first."
            )
        existing.update({
            "status": "completed", "threshold": threshold,
            "test_result_fingerprint": test_result_fingerprint, "completed_at": time.time(),
        })
        with open(self.guard_path, "w") as f:
            json.dump(existing, f, indent=2)

    def mark_failed(self, error: str) -> None:
        existing = self._read()
        if existing is None:
            return
        existing.update({"status": "failed", "error": str(error), "failed_at": time.time()})
        with open(self.guard_path, "w") as f:
            json.dump(existing, f, indent=2)
