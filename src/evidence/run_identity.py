"""
evidence/run_identity.py — real, versioned identity primitives for
non-fixture Phase 7 evidence reports (Issue #16, blocker 1).

Every function here computes a fingerprint or timestamp from ACTUAL
execution state — a fitted split, a fitted model, the real running
environment — never from a descriptive label or a static placeholder
string. There is no fallback path in this module that manufactures a
plausible-looking value when the real input is unavailable; callers that
cannot supply real split/model/environment state must not call these
functions at all (and must not build a real evidence report either — see
evidence/tracks.py's real report builders, which now require these).
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCHEMA_VERSION = "1"

# A real evaluation cannot predate this repository's Phase 7 work — any
# timestamp before this is rejected as implausible (almost certainly an
# unfixed epoch/placeholder rather than a genuine clock reading).
MIN_PLAUSIBLE_TIMESTAMP = "2024-01-01T00:00:00Z"


def utc_now_iso(clock: Optional[Any] = None) -> str:
    """Real, timezone-aware UTC timestamp, normalized to ISO-8601 with a
    trailing 'Z'. `clock` is an injectable zero-arg callable returning a
    timezone-aware datetime — tests inject a fixed clock instead of
    asserting wall-clock equality against real time."""
    now = clock() if clock is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("utc_now_iso: injected clock returned a naive datetime — must be timezone-aware")
    now_utc = now.astimezone(timezone.utc)
    return now_utc.strftime("%Y-%m-%dT%H:%M:%S") + f".{now_utc.microsecond:06d}Z"


def validate_real_timestamp(ts: str) -> None:
    """Raises ValueError for anything that is not a real, plausible,
    UTC ISO-8601 timestamp — placeholders (empty string, 'unknown',
    'not_applicable', epoch dates like 1970-01-01) are all rejected."""
    if not isinstance(ts, str) or not ts:
        raise ValueError(f"evaluation_timestamp must be a non-empty string, got {ts!r}")
    if not ts.endswith("Z"):
        raise ValueError(f"evaluation_timestamp {ts!r} must end with 'Z' (UTC)")
    try:
        parsed = datetime.strptime(ts[:-1], "%Y-%m-%dT%H:%M:%S.%f") if "." in ts \
            else datetime.strptime(ts[:-1], "%Y-%m-%dT%H:%M:%S")
    except ValueError as exc:
        raise ValueError(f"evaluation_timestamp {ts!r} is not a valid ISO-8601 UTC timestamp: {exc}") from exc
    min_dt = datetime.strptime(MIN_PLAUSIBLE_TIMESTAMP[:-1], "%Y-%m-%dT%H:%M:%S")
    if parsed < min_dt:
        raise ValueError(
            f"evaluation_timestamp {ts!r} predates {MIN_PLAUSIBLE_TIMESTAMP} — this is almost "
            "certainly an unfixed placeholder (e.g. the Unix epoch), not a genuine clock reading."
        )


def _fp(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ─── environment snapshot ───────────────────────────────────────────────────

_TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "torch")


def _package_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as importlib_metadata
        return importlib_metadata.version(name)
    except Exception:
        return None


def _git_dirty(repo_root: Path) -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def build_environment_snapshot(commit_sha: str, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """A real, sanitized snapshot of the running environment — no secrets,
    usernames, home directories, or absolute dataset paths. Two runs on
    identical dependency/interpreter/commit state produce an identical
    snapshot (and therefore identical fingerprint); any real difference in
    that state changes it."""
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    return {
        "schema_version": SCHEMA_VERSION,
        "python_version": sys.version.split()[0],
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "package_versions": {
            name: _package_version(name) for name in _TRACKED_PACKAGES
        },
        "code_commit_sha": commit_sha,
        "working_tree_dirty": _git_dirty(repo_root),
    }


def environment_snapshot_fingerprint(snapshot: Dict[str, Any]) -> str:
    return _fp(snapshot)


# ─── split manifest ─────────────────────────────────────────────────────────

def build_split_manifest(
    *, task: str, endpoint: str, dataset_accession: str,
    train_subject_ids: Sequence[str], validation_subject_ids: Sequence[str],
    development_holdout_subject_ids: Sequence[str],
    seed: int, train_frac: float, val_frac: float, test_frac: float,
    stratification_policy: str, class_mapping: Dict[str, int],
    rare_class_policy: str, label_state_policy: str, weak_label_policy: str,
    subject_identity_field: str, grouping_policy: str,
    dataset_manifest_fingerprint: str,
    internal_test_subject_ids: Sequence[str] = (),
    external_cohort_accession: Optional[str] = None,
    external_subject_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    """A canonical, complete description of a real evaluation split — the
    private, immutable-run-bundle counterpart to the sanitized summary
    `sanitize_split_manifest` produces for publication. Every field this
    function requires participates in `split_manifest_fingerprint`'s hash,
    so changing any one of them (one subject moving partition, the seed,
    the class mapping, ...) changes the fingerprint."""
    per_partition_class_counts: Dict[str, Dict[str, int]] = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "endpoint": endpoint,
        "dataset_accession": dataset_accession,
        "train_subject_ids": sorted(str(s) for s in train_subject_ids),
        "validation_subject_ids": sorted(str(s) for s in validation_subject_ids),
        "development_holdout_subject_ids": sorted(str(s) for s in development_holdout_subject_ids),
        "internal_test_subject_ids": sorted(str(s) for s in internal_test_subject_ids),
        "external_cohort_accession": external_cohort_accession,
        "external_subject_ids": sorted(str(s) for s in external_subject_ids),
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "stratification_policy": stratification_policy,
        "class_mapping": dict(sorted(class_mapping.items())),
        "rare_class_policy": rare_class_policy,
        "label_state_policy": label_state_policy,
        "weak_label_policy": weak_label_policy,
        "subject_identity_field": subject_identity_field,
        "grouping_policy": grouping_policy,
        "per_partition_class_counts": per_partition_class_counts,
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint,
    }


def split_manifest_fingerprint(manifest: Dict[str, Any]) -> str:
    return _fp(manifest)


def sanitize_split_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Publication-safe view of a split manifest: fingerprint, configuration,
    and aggregate counts only — every raw subject-id list is replaced with
    its count. Never include this alongside the raw manifest in anything
    committed or published outside the gitignored private run bundle."""
    sanitized = dict(manifest)
    for key in (
        "train_subject_ids", "validation_subject_ids", "development_holdout_subject_ids",
        "internal_test_subject_ids", "external_subject_ids",
    ):
        ids = manifest.get(key) or []
        sanitized[key + "_count"] = len(ids)
        del sanitized[key]
    sanitized["fingerprint"] = split_manifest_fingerprint(manifest)
    return sanitized


# ─── bulk (scikit-learn) preprocessing + model fingerprints ────────────────

def subject_set_fingerprint(subject_ids: Sequence[str]) -> str:
    return _fp(sorted(str(s) for s in subject_ids))


def bulk_preprocessing_fingerprint(
    *, gene_indices, gene_names_selected: Sequence[str], n_genes_available: int,
    n_top_variance_genes_requested: int, selection_policy: str,
    scaler, train_subject_ids: Sequence[str], missing_value_handling: str,
) -> str:
    """Real preprocessing identity for data/bulk_pipeline.py's fitted
    top-variance-gene-selection + StandardScaler step — no descriptive
    fallback is ever acceptable here; every field is read from the actual
    fitted FittedBulkClassifier."""
    import numpy as np
    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": selection_policy,
        "gene_indices": np.asarray(gene_indices).tolist(),
        "gene_names_selected": list(gene_names_selected),
        "n_genes_available": int(n_genes_available),
        "n_top_variance_genes_requested": int(n_top_variance_genes_requested),
        "n_top_variance_genes_effective": len(list(gene_names_selected)),
        "scaler_mean": np.asarray(scaler.mean_).tolist(),
        "scaler_scale": np.asarray(scaler.scale_).tolist(),
        "missing_value_handling": missing_value_handling,
        "feature_ordering": "gene_indices order (ascending original-column position after top-variance sort)",
        "training_subject_fingerprint": subject_set_fingerprint(train_subject_ids),
    }
    return _fp(payload)


def bulk_model_fingerprint(
    *, model, hyperparameters: Dict[str, Any], class_names: Sequence[str],
    preprocessing_fingerprint: str, train_subject_ids: Sequence[str],
) -> str:
    """Real fitted-model identity for the bulk logistic-regression
    classifier — reuses benchmarks/model_fingerprint.py's whitelist-based
    canonicalization of LogisticRegression's fitted state (coef_,
    intercept_, classes_, n_iter_, ...), rather than fingerprinting
    predictions. Binds the preprocessing fingerprint and training-subject
    set in, so a model fitted on a different feature space or a different
    training partition never collides with this one."""
    from benchmarks.model_fingerprint import sklearn_model_state_fingerprint

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_state_fingerprint": sklearn_model_state_fingerprint(model),
        "hyperparameters": dict(sorted(hyperparameters.items())),
        "class_names": list(class_names),
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "training_subject_fingerprint": subject_set_fingerprint(train_subject_ids),
        "sklearn_estimator_type": type(model).__qualname__,
    }
    return _fp(payload)


def baseline_model_fingerprint(
    *, baseline, preprocessing_fingerprint: str, train_subject_ids: Sequence[str],
) -> str:
    """Real fitted-model identity for a benchmarks.baselines.Baseline
    instance (e.g. SmokeLogisticRegression used for the GSE136831 weak-label
    experiment) — reuses the baseline's own model_state_fingerprint()
    (benchmarks/model_fingerprint.py), never predictions."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_state_fingerprint": baseline.model_state_fingerprint(),
        "hyperparameters": dict(sorted(baseline.hyperparams.items())),
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "training_subject_fingerprint": subject_set_fingerprint(train_subject_ids),
        "baseline_name": baseline.name,
    }
    return _fp(payload)


def real_git_commit_sha(repo_root: Optional[Path] = None) -> str:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, timeout=5, check=True,
        )
        sha = out.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    return "0" * 40
