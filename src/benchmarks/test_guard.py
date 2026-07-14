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
import uuid
from pathlib import Path
from typing import Dict, Optional, Union

from .atomic_io import atomic_write_json, exclusive_create_bytes


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


class FrozenTestGuardOwnershipError(RuntimeError):
    """Raised when a FrozenTestGuard instance that did not itself acquire
    the guard (or whose acquisition token doesn't match what's on disk)
    tries to mark it completed/failed — prevents one process/instance from
    finalizing a guard it never legitimately owned."""


class FrozenTestGuardCorruptedError(RuntimeError):
    """Raised when a guard file exists but does not contain valid, complete
    JSON. Corruption must never be silently treated as "no guard exists" —
    that would let a truncated/damaged guard be bypassed, re-opening
    supposedly one-time frozen-test access. Recovery requires a human to
    inspect the file at `guard_path`, confirm no frozen-test evaluation is
    genuinely in flight, and explicitly move or delete it before any new
    attempt can proceed."""


class FrozenTestGuard:
    def __init__(self, guard_path: Union[str, Path]):
        self.guard_path = Path(guard_path)
        self._owner_token: Optional[str] = None

    def _read(self) -> Optional[Dict]:
        if not self.guard_path.exists():
            return None
        try:
            with open(self.guard_path) as f:
                text = f.read()
            return json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise FrozenTestGuardCorruptedError(
                f"Frozen test guard at {self.guard_path} exists but could not be parsed "
                f"as valid JSON ({exc!r}). This must not be treated as 'no guard' — doing "
                "so could silently re-open one-time frozen-test access. Recovery: a human "
                "must confirm no evaluation is genuinely in flight, then explicitly move "
                "or delete this file before any new attempt is permitted."
            ) from exc

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

        token = uuid.uuid4().hex
        payload = {
            "status": "in_progress", "run_identity": run_identity,
            "selected_model": selected_model, "started_at": time.time(),
            "owner_token": token,
        }
        self.guard_path.parent.mkdir(parents=True, exist_ok=True)
        # O_CREAT | O_EXCL: atomically fails with FileExistsError if the
        # file already exists — closes the race window between "check if
        # it exists" and "create it" that a plain open()/exists() pair has.
        # This exclusive-creation primitive is what makes acquisition
        # itself race-free; it is deliberately NOT replaced by the
        # temp-file+os.replace() pattern used for terminal-state updates
        # below, since that pattern is atomic for REPLACING a file, not for
        # exclusively creating one. exclusive_create_bytes writes the full
        # payload to a temp file first (retrying short writes / EINTR,
        # rejecting zero-progress) and fsyncs it, THEN atomically links it
        # into self.guard_path — so self.guard_path never exists in a
        # partially-written state a concurrent reader could observe, and a
        # crash mid-write only ever leaves behind an orphaned temp file
        # (cleaned up), never a truncated guard reserving this identity. If
        # the final link fails (FileExistsError — another process won the
        # acquisition race), the guard path is untouched.
        payload_bytes = json.dumps(payload, indent=2).encode("utf-8")
        try:
            exclusive_create_bytes(self.guard_path, payload_bytes)
        except FileExistsError:
            # Another process won the acquisition race between our _read()
            # check above and this write — re-read to report the real
            # reason, exactly as if we had observed it in the initial check.
            existing = self._read()
            status = existing.get("status") if existing else None
            if status == "completed":
                raise FrozenTestAlreadyEvaluatedError(
                    f"Frozen test guard at {self.guard_path} already records a COMPLETED "
                    "evaluation (lost acquisition race to another process) — refusing to "
                    "evaluate the frozen test split again."
                ) from None
            raise FrozenTestInProgressError(
                f"Frozen test guard at {self.guard_path} was created by another process "
                "during acquisition (lost the acquisition race) — refusing to proceed."
            ) from None
        # _owner_token is only set once the payload above is confirmed
        # durably and completely written — never before, so a failed
        # acquisition can never be mistaken for a legitimately owned guard.
        self._owner_token = token

    def _require_ownership(self, existing: Dict) -> None:
        if self._owner_token is None or existing.get("owner_token") != self._owner_token:
            raise FrozenTestGuardOwnershipError(
                f"FrozenTestGuard at {self.guard_path}: this instance did not acquire the "
                "guard currently on disk (owner_token mismatch or never acquired) — refusing "
                "to update a guard it does not own. Only the FrozenTestGuard instance whose "
                "acquire() call created this file may mark it completed/failed."
            )

    def mark_completed(self, threshold: float, test_result_fingerprint: str) -> None:
        """Atomically replace the (already-acquired, in_progress) guard
        file with a terminal "completed" record, and only if this exact
        instance is the one that acquired it. Once written, acquire() on
        this path can never succeed again."""
        existing = self._read()
        if existing is None or existing.get("status") != "in_progress":
            raise RuntimeError(
                f"FrozenTestGuard.mark_completed called without a matching in_progress "
                f"guard at {self.guard_path} — acquire() must be called first."
            )
        self._require_ownership(existing)
        existing.update({
            "status": "completed", "threshold": threshold,
            "test_result_fingerprint": test_result_fingerprint, "completed_at": time.time(),
        })
        atomic_write_json(self.guard_path, existing)

    def mark_failed(self, error: str) -> None:
        existing = self._read()
        if existing is None:
            return
        self._require_ownership(existing)
        existing.update({"status": "failed", "error": str(error), "failed_at": time.time()})
        atomic_write_json(self.guard_path, existing)
