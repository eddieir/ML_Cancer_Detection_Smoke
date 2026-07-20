"""
evidence/audit.py — read-only real-data audit CLI (Step 7).

    python -m evidence.audit --config configs/default.yaml \
        --cohort-config configs/cohorts.yaml \
        --output artifacts/evidence/data_audit.json

This module never fits preprocessing, never trains a model, and never
opens a raw expression matrix. It only:
  - builds the dataset provenance manifest (src/data/manifest.py, which
    itself only computes checksums of files that are actually present),
  - loads and cross-checks the cohort registry (evidence/cohort_registry.py),
  - checks controlled-access availability via the existing adapter
    (src/data/nlst_adapter.check_nlst_availability), and
  - reports, per cohort, whether each Phase 7 track is
    IMPOSSIBLE / EXPLORATORY_DEVELOPMENT_ONLY / ELIGIBLE_* given the real
    (usually zero, since files_present is almost always False without an
    authorized local download) subject counts observed on disk.

A dry run against an environment with no local datasets must finish
successfully with a structured "blocked" status for every track — it must
never silently report a track as evaluated, and never fabricate a
subject/class count for data that was not actually found on disk.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.manifest import build_dataset_manifest, load_manifest_seed, manifest_fingerprint  # noqa: E402
from data.nlst_adapter import check_nlst_availability  # noqa: E402

from .cohort_registry import (  # noqa: E402
    cross_check_against_dataset_manifest,
    load_cohort_registry,
    registry_fingerprint,
)
from .eligibility import assess_eligibility  # noqa: E402
from .evidence_contract import not_evaluable  # noqa: E402

AUDIT_SCHEMA_VERSION = "1"
TASKS = ("smoke_classification", "malignancy_classification", "subject_level_cancer_prediction")


def _sanitize(value: Any) -> Any:
    """Never emit an absolute local filesystem path or NLST root env value
    in the audit report — booleans/counts only."""
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return "<path redacted>"
    return value


def _dataset_summary(entries) -> List[dict]:
    out = []
    for e in entries:
        n_present = sum(1 for v in e.raw_file_checksums.values() if v is not None)
        n_expected = len(e.raw_file_names)
        out.append({
            "dataset_id": e.dataset_id,
            "accession": e.accession,
            "species": e.species,
            "assay_type": e.assay_type,
            "controlled_access": e.controlled_access,
            "files_present": e.files_present,
            "raw_files_found": n_present,
            "raw_files_expected": n_expected,
            "checksum_status": "computed_from_local_files" if e.files_present else "no_local_files_no_checksum",
        })
    return out


def _access_status(cohort, entries_by_id) -> dict:
    entry = entries_by_id.get(cohort.dataset_id)
    files_present = bool(entry.files_present) if entry is not None else False
    status: Dict[str, Any] = {
        "cohort_id": cohort.cohort_id,
        "accession": cohort.accession,
        "access_level": cohort.access_level,
        "local_files_present": files_present,
    }
    if cohort.access_level == "controlled":
        avail = check_nlst_availability()
        status["controlled_access_authorized_locally"] = bool(avail.available)
        if not avail.available:
            status["blocker"] = not_evaluable(
                reason_code="CONTROLLED_ACCESS_UNAVAILABLE",
                reason=f"{cohort.accession} is controlled-access and no authorized local dataset "
                       "was found in this environment.",
                required_next_action=(
                    "Obtain an approved NCI Data Use Agreement, point the NLST_DATA_ROOT "
                    "environment variable at the authorized local extract, and re-run the audit."
                ),
            )
    else:
        if not files_present:
            status["blocker"] = not_evaluable(
                reason_code="LOCAL_DATA_NOT_PRESENT",
                reason=f"{cohort.accession} is public but no raw files were found on disk in "
                       "this environment (no download was attempted by this audit).",
                required_next_action=(
                    f"Download {cohort.accession} raw files into the directory documented in "
                    "configs/datasets.yaml and re-run the audit; this audit does not download data itself."
                ),
            )
    return status


def _task_eligibility_table(cohorts, entries_by_id) -> List[dict]:
    """Per (cohort, task) eligibility using ONLY real, on-disk subject
    counts. Since files_present is false for every cohort in an
    environment with no downloaded data, every row here is honestly
    IMPOSSIBLE with reason 'no local data' rather than a fabricated
    count."""
    rows = []
    for cohort in cohorts:
        entry = entries_by_id.get(cohort.dataset_id)
        files_present = bool(entry.files_present) if entry is not None else False
        # subject_count/per_class_counts are ONLY ever non-zero here if a
        # real, checksummed local file was found — this audit never
        # estimates or assumes a count for absent data.
        subject_count = 0
        per_class_counts: Dict[str, int] = {}
        for task in TASKS:
            decision = assess_eligibility(
                task=task, cohort=cohort,
                subject_count=subject_count, per_class_counts=per_class_counts,
                independent_source_count=1, role="development",
            )
            row = decision.to_dict()
            row["local_data_present"] = files_present
            rows.append(row)
    return rows


def run_audit(config_path: str, cohort_config_path: str) -> dict:
    dataset_entries = build_dataset_manifest(seed_path="configs/datasets.yaml")
    entries_by_id = {e.dataset_id: e for e in dataset_entries}
    dataset_fp = manifest_fingerprint(dataset_entries)

    cohorts = load_cohort_registry(cohort_config_path)
    cohort_fp = registry_fingerprint(cohorts)

    dataset_seed = load_manifest_seed("configs/datasets.yaml")
    contradictions = cross_check_against_dataset_manifest(cohorts, dataset_seed)

    dataset_summary = _dataset_summary(dataset_entries)
    access_status = [_access_status(c, entries_by_id) for c in cohorts]
    eligibility_table = _task_eligibility_table(cohorts, entries_by_id)

    any_local_data = any(d["files_present"] for d in dataset_summary)
    any_controlled_authorized = any(
        s.get("controlled_access_authorized_locally") for s in access_status
    )

    blockers = []
    for s in access_status:
        if "blocker" in s:
            blockers.append({"cohort_id": s["cohort_id"], **s["blocker"]})

    assay_species_summary = {
        c.cohort_id: {"species": c.species, "assay_type": c.assay_type,
                      "single_cell_or_bulk": c.single_cell_or_bulk}
        for c in cohorts
    }

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_by": "evidence.audit",
        "config_path": _sanitize(config_path),
        "dataset_manifest_fingerprint": dataset_fp,
        "cohort_registry_fingerprint": cohort_fp,
        "config_cohort_contradictions": contradictions,
        "discovered_datasets": dataset_summary,
        "access_status_by_cohort": access_status,
        "assay_species_summary": assay_species_summary,
        # No real subject/cell/label/outcome data was found on disk in
        # this environment, so every count-based section below is a
        # structured not_evaluable object rather than a fabricated number.
        "subject_counts": not_evaluable(
            reason_code="NO_LOCAL_DATA",
            reason="No cohort had local raw files present; subject counts cannot be computed.",
            required_next_action="Download and place raw files per configs/datasets.yaml, then re-run the audit.",
        ) if not any_local_data else "computed_subject_counts_not_implemented_pending_local_data",
        "row_cell_counts": not_evaluable(
            reason_code="NO_LOCAL_DATA",
            reason="No cohort had local raw files present; row/cell counts cannot be computed.",
            required_next_action="Download and place raw files per configs/datasets.yaml, then re-run the audit.",
        ) if not any_local_data else "computed_row_cell_counts_not_implemented_pending_local_data",
        "verified_unknown_weak_label_counts": not_evaluable(
            reason_code="NO_LOCAL_DATA",
            reason="No cohort had local raw files present; label counts cannot be computed.",
            required_next_action="Download and place raw files per configs/datasets.yaml, then re-run the audit.",
        ) if not any_local_data else "computed_label_counts_not_implemented_pending_local_data",
        "cancer_outcome_counts": not_evaluable(
            reason_code="NO_LOCAL_DATA",
            reason="No cohort had local raw files present; outcome counts cannot be computed.",
            required_next_action="Download and place raw files per configs/datasets.yaml, then re-run the audit.",
        ) if not any_local_data else "computed_outcome_counts_not_implemented_pending_local_data",
        "missingness_summary": not_evaluable(
            reason_code="NO_LOCAL_DATA",
            reason="No local data to compute missingness against.",
            required_next_action="Download and place raw files per configs/datasets.yaml, then re-run the audit.",
        ) if not any_local_data else "computed_missingness_not_implemented_pending_local_data",
        "duplicate_id_checks": not_evaluable(
            reason_code="NO_LOCAL_DATA",
            reason="No local data to check for duplicate subject IDs.",
            required_next_action="Download and place raw files per configs/datasets.yaml, then re-run the audit.",
        ) if not any_local_data else "computed_duplicate_checks_not_implemented_pending_local_data",
        "cross_source_subject_collision_checks": not_evaluable(
            reason_code="NO_LOCAL_DATA",
            reason="No local data across sources to check for subject-ID collisions.",
            required_next_action="Download and place raw files for at least two cohorts, then re-run the audit.",
        ) if not any_local_data else "computed_collision_checks_not_implemented_pending_local_data",
        "class_by_source_table": not_evaluable(
            reason_code="NO_LOCAL_DATA",
            reason="No local label data to tabulate by source.",
            required_next_action="Download and place raw files per configs/datasets.yaml, then re-run the audit.",
        ) if not any_local_data else "computed_class_by_source_not_implemented_pending_local_data",
        "outcome_by_source_table": not_evaluable(
            reason_code="NO_LOCAL_DATA",
            reason="No local outcome data to tabulate by source.",
            required_next_action="Download and place raw files per configs/datasets.yaml, then re-run the audit.",
        ) if not any_local_data else "computed_outcome_by_source_not_implemented_pending_local_data",
        "task_eligibility": eligibility_table,
        "blockers": blockers,
        "controlled_access_cohorts_authorized_in_this_environment": any_controlled_authorized,
        "any_local_real_data_present": any_local_data,
        "overall_status": "blocked_no_real_data" if not any_local_data else "partial_local_data_present",
    }

    payload_for_fp = json.dumps(report, sort_keys=True, default=str)
    import hashlib
    report["content_fingerprint"] = hashlib.sha256(payload_for_fp.encode("utf-8")).hexdigest()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7 read-only real-data audit (never trains or fits anything)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cohort-config", default="configs/cohorts.yaml")
    parser.add_argument("--output", default="artifacts/evidence/data_audit.json")
    parser.add_argument("--validate-only", action="store_true",
                         help="Run the audit but do not write --output; exit nonzero on any config/cohort contradiction.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON to stdout.")
    args = parser.parse_args(argv)

    report = run_audit(args.config, args.cohort_config)

    if args.json or args.validate_only:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))

    if not args.validate_only:
        from benchmarks.atomic_io import atomic_write_json
        atomic_write_json(args.output, report)
        print(f"[evidence.audit] wrote {args.output}")

    if report["config_cohort_contradictions"]:
        print("[evidence.audit] CONTRADICTIONS between configs/datasets.yaml and configs/cohorts.yaml:",
              file=sys.stderr)
        for c in report["config_cohort_contradictions"]:
            print(f"  - {c}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
