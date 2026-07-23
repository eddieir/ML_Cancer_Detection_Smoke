"""
evidence/publication.py — Issue #16 blocker 5: sanitized, independently
auditable publication of a real evidence run.

Private run bundles under artifacts/evidence/ are gitignored — they can
carry subject-level information (predictions/*.csv, split_manifest.json's
raw subject-id lists) that must never be committed. This module derives a
sanitized, non-identifying summary from a VALIDATED private bundle and
writes it to a tracked location (evidence/published/<run_id>/summary.json)
so results are independently reviewable without exposing subject-level
data.

Order of operations (never reordered):
  1. Validate the private bundle (evidence.artifact_bundle.validate_evidence_run
     — re-checksums every file; raises on any mismatch).
  2. Load and validate the named evidence report
     (evidence.evidence_contract.validate_report).
  3. Derive the sanitized summary from an explicit field allowlist — never
     a blacklist walk over the raw report, so a field this module doesn't
     know about cannot leak through by omission from a blocklist.
  4. Scan the derived summary for prohibited content (subject/sample IDs,
     local filesystem paths, anything key-shaped like a credential) as a
     defense-in-depth check, even though the allowlist construction should
     already exclude all of it.
  5. Compute a checksum over the sanitized content and bind it to a
     fingerprint of the source private bundle.
  6. Write the summary, then immediately reload and re-validate it
     (round-trip check) before returning.

Never writes participant/sample IDs, row-level predictions, local paths,
usernames, tokens, credentials, raw expression, model weights/checkpoints,
or signing private keys.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .artifact_bundle import validate_evidence_run
from .errors import ArtifactValidationError
from .evidence_contract import report_from_dict, validate_report
from .run_identity import sanitize_split_manifest

PUBLICATION_SCHEMA_VERSION = "1"

_ALLOWED_IDENTITY_FIELDS = (
    "task", "endpoint", "endpoint_definition", "prediction_unit",
    "biological_specimen_type", "assay_modality", "dataset_accession",
    "cohort_role", "species", "sample_count", "unique_subject_count",
    "class_counts", "verified_label_count", "unknown_label_count",
    "excluded_subject_count", "split_role", "split_manifest_fingerprint",
    "dataset_manifest_fingerprint", "preprocessing_artifact_fingerprint",
    "model_fingerprint", "random_seed", "code_commit_sha",
    "environment_fingerprint", "synthetic_flag", "development_only_flag",
    "frozen_test_access_status", "evidence_level", "limitations",
    "evaluation_timestamp", "report_schema_version",
    "calibration_fingerprint", "threshold_fingerprint",
)

_ALLOWED_METRIC_FIELDS = (
    "primary_metric", "macro_f1", "weighted_f1", "balanced_accuracy",
    "per_class", "confusion_matrix", "accuracy", "effective_label_mapping",
    "weak_label_experiment", "classes_below_support_threshold",
    "threshold_method", "calibration_curve", "roc_auc", "average_precision",
    "brier_score", "expected_calibration_error",
)

_PROHIBITED_KEY_RE = re.compile(
    r"(^|_)(subject_ids?|sample_ids?|gsm_ids?|raw_file_paths?|local_paths?|"
    r"password|token|secret|private_key|username)($|_)"
)
_GSM_RE = re.compile(r"\bGSM\d{4,}\b")
_ABS_PATH_RE = re.compile(r"(^|[\s\"'])/(Users|home)/")


class PublicationError(ArtifactValidationError):
    """Raised for any failure deriving, scanning, writing, or round-trip
    validating a sanitized publication package."""


def _scan_for_prohibited(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_lower = str(k).lower()
            # a "*_count" key (e.g. sanitize_split_manifest's
            # train_subject_ids_count) is an aggregate integer, not a raw
            # identifier list — explicitly exempted from the prohibition.
            if not key_lower.endswith("_count") and _PROHIBITED_KEY_RE.search(key_lower):
                raise PublicationError(
                    f"sanitized publication content has a prohibited key {k!r} at {path} — "
                    "publication must not carry subject/sample identifiers, local paths, or "
                    "credential-shaped fields"
                )
            _scan_for_prohibited(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _scan_for_prohibited(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        if _GSM_RE.search(obj):
            raise PublicationError(f"sanitized publication content at {path} contains a GSM accession: {obj!r}")
        if _ABS_PATH_RE.search(obj):
            raise PublicationError(f"sanitized publication content at {path} contains a local filesystem path: {obj!r}")


def _content_checksum(obj: Dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _private_bundle_fingerprint(manifest: Dict[str, Any]) -> str:
    blob = json.dumps(manifest.get("files", {}), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def derive_sanitized_summary(
    manifest: Dict[str, Any], report: Dict[str, Any],
    *, split_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pure function: given an already-validated manifest + report (+
    optional raw split_manifest, which is itself sanitized here via
    run_identity.sanitize_split_manifest before inclusion), returns the
    sanitized summary dict. Does no I/O and performs no validation of its
    own inputs — callers (publish_evidence_run) are responsible for that."""
    identity = report["identity"]
    metrics = report["metrics"]

    summary: Dict[str, Any] = {"schema_version": PUBLICATION_SCHEMA_VERSION}
    for field in _ALLOWED_IDENTITY_FIELDS:
        if field in identity:
            summary[field] = identity[field]
    summary["metrics"] = {k: v for k, v in metrics.items() if k in _ALLOWED_METRIC_FIELDS}
    summary["report_content_fingerprint"] = report.get("content_fingerprint")
    summary["source_private_bundle_fingerprint"] = _private_bundle_fingerprint(manifest)
    summary["run_status"] = manifest.get("run_status")

    if split_manifest is not None:
        summary["split_configuration"] = sanitize_split_manifest(split_manifest)

    summary["checksum_sha256"] = _content_checksum(summary)
    return summary


def publish_evidence_run(
    artifacts_root: Union[str, Path], run_id: str, *,
    report_relative_path: str,
    published_root: Union[str, Path],
    split_manifest_relative_path: Optional[str] = "split_manifest.json",
) -> Dict[str, Any]:
    """End-to-end: validate the private bundle, load+validate the named
    report, derive the sanitized summary, scan it, write it to
    `published_root`/`run_id`/summary.json, then reload and re-validate
    the written file before returning it. Raises PublicationError (or a
    more specific ArtifactValidationError/ReportValidationError subclass)
    on any failure — never writes a partial or unvalidated summary."""
    manifest = validate_evidence_run(artifacts_root, run_id)

    run_dir = Path(artifacts_root) / run_id
    report_path = run_dir / report_relative_path
    if report_relative_path not in manifest.get("files", {}):
        raise PublicationError(
            f"{report_relative_path!r} is not listed in the validated manifest for run_id={run_id!r} "
            "— only files the manifest itself binds may be published"
        )
    with open(report_path) as f:
        report = json.load(f)
    validate_report(report_from_dict(report))

    split_manifest = None
    if split_manifest_relative_path and split_manifest_relative_path in manifest.get("files", {}):
        with open(run_dir / split_manifest_relative_path) as f:
            split_manifest = json.load(f)

    summary = derive_sanitized_summary(manifest, report, split_manifest=split_manifest)
    _scan_for_prohibited(summary)

    from benchmarks.atomic_io import atomic_write_json

    out_path = Path(published_root) / run_id / "summary.json"
    atomic_write_json(out_path, summary)

    with open(out_path) as f:
        reloaded = json.load(f)
    _scan_for_prohibited(reloaded)
    if reloaded.get("checksum_sha256") != summary["checksum_sha256"]:
        raise PublicationError(
            f"published summary at {out_path} does not match the freshly derived checksum — "
            "the file was corrupted or tampered with between write and reload"
        )
    if reloaded != summary:
        raise PublicationError(
            f"published summary at {out_path} does not byte-for-byte match the derived summary "
            "after reload — the write or reload path is not faithful"
        )

    return reloaded


def publish_repeated_development_summary(
    result: Dict[str, Any], *, run_id: str, published_root: Union[str, Path],
    code_commit_sha: str,
) -> Dict[str, Any]:
    """Sanitizes and publishes a evidence.development.run_gse123352_repeated_development()
    result. Unlike publish_evidence_run, this does not validate against the
    single-report EvidenceReport schema (a repeated-development summary is
    an aggregate across many seeds, not one report) — it instead applies
    the SAME defense-in-depth scan (_scan_for_prohibited) used for every
    other published artifact, since this result already carries no
    subject/sample identifiers by construction (see
    evidence/development.py::run_gse123352_repeated_development — only
    aggregate counts, per-seed metric values, and fingerprints are
    returned). Raises PublicationError if the scan finds anything
    prohibited anyway, and re-validates the written file by reload before
    returning."""
    if result.get("status") != "complete":
        raise PublicationError(
            f"refusing to publish a repeated-development result with status={result.get('status')!r} "
            "— only a complete result may be published."
        )

    summary = dict(result)
    summary["schema_version"] = PUBLICATION_SCHEMA_VERSION
    summary["run_id"] = run_id
    summary["code_commit_sha"] = code_commit_sha
    from .run_identity import utc_now_iso
    summary["publication_timestamp"] = utc_now_iso()
    summary["checksum_sha256"] = _content_checksum(summary)

    _scan_for_prohibited(summary)

    from benchmarks.atomic_io import atomic_write_json

    out_path = Path(published_root) / run_id / "summary.json"
    atomic_write_json(out_path, summary)

    with open(out_path) as f:
        reloaded = json.load(f)
    _scan_for_prohibited(reloaded)
    if reloaded != summary:
        raise PublicationError(
            f"published repeated-development summary at {out_path} does not byte-for-byte match "
            "the derived summary after reload — the write or reload path is not faithful"
        )
    return reloaded
