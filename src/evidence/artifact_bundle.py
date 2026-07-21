"""
evidence/artifact_bundle.py — Step 15: the immutable Phase 7 evidence-run
artifact directory.

A run directory under artifacts/evidence/<run_id>/ is built by ONE call to
write_evidence_run(), which:
  - accepts a mapping of {relative_path: content} for whichever files this
    particular run actually produced (a development-only, not_evaluable
    run has no internal_test.csv, and that is correct — this writer never
    fabricates a file that wasn't produced),
  - writes each file atomically (reusing benchmarks/atomic_io.py — never a
    bare open()/write()),
  - computes a sha256 checksum for every file actually written into
    checksums.json,
  - writes manifest.json LAST, binding every other file's relative path +
    checksum + a `run_status` field ("complete" or "incomplete").

Once a run directory's manifest.json records run_status="complete", NO
further write_evidence_run() call may target the same run_id —
RunAlreadyCompleteError is raised instead. This makes a completed evidence
run directory immutable in practice, not just by convention.

validate_evidence_run() is the read-only counterpart: it re-checksums
every file manifest.json lists and raises ArtifactValidationError the
instant anything is missing or does not match — this is what
evidence.runner's `validate` subcommand calls.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from benchmarks.atomic_io import atomic_write_bytes, atomic_write_csv_rows, atomic_write_json
from data.manifest import sha256_of_file

from .errors import ArtifactBundleError, ArtifactValidationError, RunAlreadyCompleteError

BUNDLE_SCHEMA_VERSION = "1"

VALID_RUN_STATUSES = ("complete", "incomplete")

# Names the writer itself owns — a caller passing one of these in `files`
# is a usage error (they are computed and written by this module only).
RESERVED_RELATIVE_PATHS = ("manifest.json", "checksums.json")

# Documented run-directory layout (docs/EVIDENCE_PROTOCOL.md +
# task Step 15) — not every run populates every one of these; this tuple
# is used only to sanity-check that a genuinely unexpected relative path
# (a typo, or a path escaping the run directory) is caught early rather
# than silently written somewhere unintended. A relative_path outside this
# set is still permitted (the framework may grow new artifact kinds), as
# long as it is not a path-traversal attempt.
KNOWN_RELATIVE_PATH_PREFIXES = (
    "manifest.json", "configuration.json", "environment.json", "data_audit.json",
    "cohort_flow.json", "split_manifest.json",
    "preprocessing/", "models/", "predictions/", "metrics/", "reports/",
    "checksums.json",
)


def _validate_relative_path(rel_path: str) -> Path:
    if not rel_path or rel_path.startswith("/") or rel_path.startswith("\\"):
        raise ArtifactBundleError(f"invalid relative path {rel_path!r} — must be a non-empty, non-absolute path")
    parts = Path(rel_path).parts
    if ".." in parts:
        raise ArtifactBundleError(f"invalid relative path {rel_path!r} — path traversal ('..') is not permitted")
    if rel_path in RESERVED_RELATIVE_PATHS:
        raise ArtifactBundleError(
            f"relative path {rel_path!r} is reserved — manifest.json and checksums.json are "
            "written by write_evidence_run() itself and must not be supplied by the caller."
        )
    return Path(rel_path)


def _write_one(run_dir: Path, rel_path: str, content: Any) -> Path:
    path = run_dir / _validate_relative_path(rel_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        if not isinstance(content, (dict, list)):
            raise ArtifactBundleError(f"{rel_path}: .json content must be a dict or list, got {type(content).__name__}")
        atomic_write_json(path, content)
    elif suffix == ".csv":
        if not isinstance(content, list):
            raise ArtifactBundleError(f"{rel_path}: .csv content must be a list of row dicts, got {type(content).__name__}")
        atomic_write_csv_rows(path, content)
    elif suffix == ".md":
        if not isinstance(content, str):
            raise ArtifactBundleError(f"{rel_path}: .md content must be a str, got {type(content).__name__}")
        atomic_write_bytes(path, content.encode("utf-8"))
    else:
        raise ArtifactBundleError(f"{rel_path}: unsupported file extension {suffix!r} (only .json/.csv/.md are written by this module)")
    return path


def _manifest_path(artifacts_root: Union[str, Path], run_id: str) -> Path:
    return Path(artifacts_root) / run_id / "manifest.json"


def _load_manifest_if_present(artifacts_root: Union[str, Path], run_id: str) -> Optional[Dict]:
    path = _manifest_path(artifacts_root, run_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ArtifactBundleError(f"existing manifest.json at {path} could not be parsed: {exc!r}") from exc


def write_evidence_run(
    artifacts_root: Union[str, Path],
    run_id: str,
    files: Dict[str, Any],
    *,
    run_status: str = "complete",
    extra_manifest_fields: Optional[Dict[str, Any]] = None,
) -> Path:
    """Writes every file in `files` (relative_path -> content) atomically
    into artifacts/<artifacts_root>/<run_id>/, then checksums.json, then
    manifest.json last. Raises RunAlreadyCompleteError if `run_id` already
    has a manifest.json recording run_status='complete' — such a run
    directory can never be written into again, by this function or any
    other caller that respects it."""
    if run_status not in VALID_RUN_STATUSES:
        raise ArtifactBundleError(f"run_status must be one of {VALID_RUN_STATUSES}, got {run_status!r}")

    existing = _load_manifest_if_present(artifacts_root, run_id)
    if existing is not None and existing.get("run_status") == "complete":
        raise RunAlreadyCompleteError(
            f"run_id={run_id!r} under {artifacts_root} already has a manifest.json recording "
            "run_status='complete' — a completed evidence run directory is immutable and can "
            "never be written into again. Use a new run_id for a new run."
        )

    run_dir = Path(artifacts_root) / run_id
    checksums: Dict[str, str] = {}
    file_records: Dict[str, Dict[str, Any]] = {}

    for rel_path, content in files.items():
        written_path = _write_one(run_dir, rel_path, content)
        digest = sha256_of_file(written_path)
        checksums[rel_path] = digest
        file_records[rel_path] = {"sha256": digest, "size": written_path.stat().st_size}

    checksums_path = run_dir / "checksums.json"
    atomic_write_json(checksums_path, checksums)

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run_id": run_id,
        "run_status": run_status,
        "files": file_records,
        "checksums_file": "checksums.json",
    }
    if extra_manifest_fields:
        overlap = set(extra_manifest_fields) & set(manifest)
        if overlap:
            raise ArtifactBundleError(f"extra_manifest_fields cannot override reserved manifest keys {overlap}")
        manifest.update(extra_manifest_fields)

    atomic_write_json(run_dir / "manifest.json", manifest)
    return run_dir


def validate_evidence_run(artifacts_root: Union[str, Path], run_id: str) -> Dict[str, Any]:
    """Read-only: re-checksums every file manifest.json lists for `run_id`
    and raises ArtifactValidationError the instant anything is missing or
    mismatched. Never writes anything. Returns the manifest dict on
    success."""
    run_dir = Path(artifacts_root) / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise ArtifactValidationError(f"no manifest.json found for run_id={run_id!r} at {manifest_path}")
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ArtifactValidationError(f"manifest.json at {manifest_path} could not be parsed: {exc!r}") from exc

    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"manifest.json at {manifest_path} has schema_version={manifest.get('schema_version')!r}, "
            f"this code understands {BUNDLE_SCHEMA_VERSION!r} only."
        )
    if manifest.get("run_status") not in VALID_RUN_STATUSES:
        raise ArtifactValidationError(
            f"manifest.json at {manifest_path} has invalid run_status={manifest.get('run_status')!r}"
        )

    files = manifest.get("files") or {}
    if not isinstance(files, dict):
        raise ArtifactValidationError(f"manifest.json at {manifest_path} has a malformed 'files' section")

    for rel_path, info in files.items():
        full_path = run_dir / rel_path
        if not full_path.exists():
            raise ArtifactValidationError(
                f"run_id={run_id!r}: manifest.json lists {rel_path!r}, but the file is missing at {full_path}"
            )
        actual = sha256_of_file(full_path)
        expected = info.get("sha256")
        if actual != expected:
            raise ArtifactValidationError(
                f"run_id={run_id!r}: {rel_path!r} checksum mismatch — manifest.json recorded "
                f"{expected!r}, file on disk hashes to {actual!r}."
            )

    checksums_path = run_dir / str(manifest.get("checksums_file", "checksums.json"))
    if not checksums_path.exists():
        raise ArtifactValidationError(f"run_id={run_id!r}: checksums.json referenced by manifest.json is missing")
    try:
        with open(checksums_path) as f:
            checksums = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ArtifactValidationError(f"checksums.json at {checksums_path} could not be parsed: {exc!r}") from exc
    for rel_path, info in files.items():
        if checksums.get(rel_path) != info.get("sha256"):
            raise ArtifactValidationError(
                f"run_id={run_id!r}: checksums.json and manifest.json disagree on the checksum for {rel_path!r}"
            )

    return manifest


def read_manifest(artifacts_root: Union[str, Path], run_id: str) -> Dict[str, Any]:
    """Read-only, no checksum re-verification — a lightweight load used by
    inspect-style tooling that only needs to know what a run directory
    claims to contain, not to re-validate it (use validate_evidence_run
    for that)."""
    manifest_path = _manifest_path(artifacts_root, run_id)
    if not manifest_path.exists():
        raise ArtifactValidationError(f"no manifest.json found for run_id={run_id!r} at {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)


def list_run_ids(artifacts_root: Union[str, Path]) -> List[str]:
    """Read-only directory listing of run_ids with a manifest.json under
    `artifacts_root` — never opens or checksums any file."""
    root = Path(artifacts_root)
    if not root.exists():
        return []
    return sorted(
        p.parent.name for p in root.glob("*/manifest.json") if p.is_file()
    )
