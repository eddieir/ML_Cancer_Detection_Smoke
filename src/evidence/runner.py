"""
evidence/runner.py — Step 16: the umbrella Phase 7 evidence CLI.

    python -m evidence.runner audit [--config ...] [--cohort-config ...] [--output ...]
    python -m evidence.runner inspect --run-dir artifacts/evidence/<run_id>
    python -m evidence.runner validate --run-dir artifacts/evidence/<run_id>
    python -m evidence.runner clinical-readiness --run-dir artifacts/evidence/<run_id>
    python -m evidence.runner development | internal-test | external-test

Design rules enforced by this module (never bypassed by any flag):
  - `audit` is a thin wrapper around evidence.audit.run_audit() — the
    logic is not duplicated here.
  - `inspect` and `validate` never write anything; `inspect` loads
    manifest.json read-only (evidence/artifact_bundle.py::read_manifest),
    `validate` re-checksums every file it lists
    (evidence/artifact_bundle.py::validate_evidence_run).
  - `clinical-readiness` never trains anything. It re-validates the given
    run directory (read-only) and then calls
    evidence/clinical_readiness.py::assess_clinical_readiness() against
    dimension records — defaulting every dimension to "not_started" when,
    as in this repository today, no real dimension evidence has been
    assembled. It never fabricates a "complete" dimension.
  - `development` / `internal-test` / `external-test` are honest stubs:
    the leakage-safe candidate-comparison pipeline integration (Steps
    8-10) does not exist as a callable pipeline in this repository yet, so
    these subcommands return a structured not_evaluable(reason_code=
    'TRACK_RUNNER_NOT_YET_INTEGRATED') rather than pretending to run
    something that isn't there.
  - `--diagnostic-mode` is accepted only by `audit` (where it is a no-op —
    audit.py has no diagnostic path at all) and is REJECTED with a typed
    EvidenceRunnerUsageError on every other subcommand. This mirrors how
    the rest of this repository treats diagnostic_mode — an explicit,
    narrow opt-out reserved for synthetic/diagnostic fixtures, never
    permitted on a real execution path (see src/data/assay_policy.py,
    src/data/preprocessing.py).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).parents[1]))

from .artifact_bundle import read_manifest, validate_evidence_run  # noqa: E402
from .clinical_readiness import (  # noqa: E402
    assess_clinical_readiness,
    default_not_started_assessment,
    load_dimension_policy,
)
from .errors import ArtifactValidationError, EvidenceRunnerUsageError  # noqa: E402
from .evidence_contract import not_evaluable  # noqa: E402

DEFAULT_DIMENSION_CONFIG = "configs/clinical_readiness.yaml"
REAL_SUBCOMMANDS = (
    "clinical-readiness", "inspect", "validate", "development", "internal-test", "external-test",
)
TRACK_STUB_SUBCOMMANDS = ("development", "internal-test", "external-test")


def _git_commit_sha() -> str:
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=5, check=True,
        )
        sha = out.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    return "0" * 40


def _reject_diagnostic_mode(args: argparse.Namespace, subcommand: str) -> None:
    if getattr(args, "diagnostic_mode", False) and subcommand != "audit":
        raise EvidenceRunnerUsageError(
            f"--diagnostic-mode was passed to the {subcommand!r} subcommand — real evidence "
            "execution rejects diagnostic_mode unconditionally. diagnostic_mode is reserved for "
            "narrow, explicit synthetic/diagnostic fixture paths elsewhere in this repository "
            "(see src/data/assay_policy.py, src/data/preprocessing.py) and is never accepted by "
            "a real (non-audit) evidence.runner subcommand."
        )


def _split_run_dir(run_dir: Union[str, Path]) -> Tuple[Path, str]:
    run_dir = Path(run_dir)
    return run_dir.parent, run_dir.name


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


# ─── audit — thin wrapper, no duplicated logic ─────────────────────────────

def cmd_audit(args: argparse.Namespace) -> int:
    from . import audit as audit_module

    report = audit_module.run_audit(args.config, args.cohort_config)
    if args.output:
        from benchmarks.atomic_io import atomic_write_json
        atomic_write_json(args.output, report)
    _print(report)
    if report["config_cohort_contradictions"]:
        return 1
    return 0


# ─── inspect — read-only summary ───────────────────────────────────────────

def cmd_inspect(args: argparse.Namespace) -> int:
    _reject_diagnostic_mode(args, "inspect")
    artifacts_root, run_id = _split_run_dir(args.run_dir)
    try:
        manifest = read_manifest(artifacts_root, run_id)
    except ArtifactValidationError as exc:
        _print({"status": "not_found", "run_dir": str(args.run_dir), "error": str(exc)})
        return 1
    summary = {
        "run_dir": str(args.run_dir),
        "run_id": run_id,
        "run_status": manifest.get("run_status"),
        "schema_version": manifest.get("schema_version"),
        "file_count": len(manifest.get("files", {})),
        "files": sorted(manifest.get("files", {}).keys()),
    }
    _print(summary)
    return 0


# ─── validate — read-only manifest/checksum verification ──────────────────

def cmd_validate(args: argparse.Namespace) -> int:
    _reject_diagnostic_mode(args, "validate")
    artifacts_root, run_id = _split_run_dir(args.run_dir)
    try:
        manifest = validate_evidence_run(artifacts_root, run_id)
    except ArtifactValidationError as exc:
        _print({"status": "fail", "run_dir": str(args.run_dir), "error": str(exc)})
        return 1
    _print({"status": "pass", "run_dir": str(args.run_dir), "run_status": manifest.get("run_status")})
    return 0


# ─── clinical-readiness — read-only, never trains ──────────────────────────

def cmd_clinical_readiness(args: argparse.Namespace) -> int:
    _reject_diagnostic_mode(args, "clinical-readiness")
    artifacts_root, run_id = _split_run_dir(args.run_dir)
    try:
        manifest = validate_evidence_run(artifacts_root, run_id)
    except ArtifactValidationError as exc:
        _print({"status": "fail", "run_dir": str(args.run_dir), "error": str(exc)})
        return 1

    policy = load_dimension_policy(args.dimension_config)
    commit_sha = manifest.get("code_commit_sha") or _git_commit_sha()
    # This CLI never trains and never synthesizes evidence — every
    # dimension defaults to "not_started" unless a genuine, previously
    # assembled reports/clinical_readiness.json (itself produced by a real
    # assessment, outside this stub) is present in the run directory to
    # read instead. No such file format is produced by any pipeline that
    # exists in this repository yet, so this honestly reports
    # "not_started" for every dimension rather than fabricating "complete".
    dimensions = [
        default_not_started_assessment(dim_id, mandatory, commit_sha)
        for dim_id, mandatory in sorted(policy.items())
    ]
    report = assess_clinical_readiness(dimensions, commit_sha)
    _print(report.to_dict())
    return 0


# ─── development / internal-test / external-test — honest stubs ───────────

def cmd_track_stub(args: argparse.Namespace, subcommand: str) -> int:
    _reject_diagnostic_mode(args, subcommand)
    if getattr(args, "run_dir", None) is not None:
        raise EvidenceRunnerUsageError(
            f"--run-dir is not a supported flag for the {subcommand!r} subcommand — it does not "
            "read or write any run directory; it only reports that the underlying pipeline "
            "integration does not exist yet."
        )
    result = not_evaluable(
        reason_code="TRACK_RUNNER_NOT_YET_INTEGRATED",
        reason=(
            f"evidence.runner {subcommand!r} has no callable leakage-safe candidate-comparison "
            "pipeline to invoke in this repository yet — the Steps 8-10 pipeline integration is "
            "being built in a separate round."
        ),
        required_next_action=(
            "Re-run this subcommand once the development/internal-test/external-test pipeline "
            "integration (Steps 8-10) lands in this repository."
        ),
        subcommand=subcommand,
    )
    _print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evidence.runner",
        description="Umbrella CLI for the Phase 7 real-world-evidence framework.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_audit = sub.add_parser("audit", help="Read-only real-data audit (thin wrapper around evidence.audit).")
    p_audit.add_argument("--config", default="configs/default.yaml")
    p_audit.add_argument("--cohort-config", default="configs/cohorts.yaml")
    p_audit.add_argument("--output", default=None)
    p_audit.add_argument("--diagnostic-mode", action="store_true", help="No-op for audit — audit.py has no diagnostic path.")
    p_audit.set_defaults(func=cmd_audit)

    p_inspect = sub.add_parser("inspect", help="Read-only summary of a run directory.")
    p_inspect.add_argument("--run-dir", required=True)
    p_inspect.add_argument("--diagnostic-mode", action="store_true")
    p_inspect.set_defaults(func=cmd_inspect)

    p_validate = sub.add_parser("validate", help="Read-only manifest/checksum validation of a run directory.")
    p_validate.add_argument("--run-dir", required=True)
    p_validate.add_argument("--diagnostic-mode", action="store_true")
    p_validate.set_defaults(func=cmd_validate)

    p_cr = sub.add_parser("clinical-readiness", help="Read-only clinical-readiness assessment of a run directory. Never trains.")
    p_cr.add_argument("--run-dir", required=True)
    p_cr.add_argument("--dimension-config", default=DEFAULT_DIMENSION_CONFIG)
    p_cr.add_argument("--diagnostic-mode", action="store_true")
    p_cr.set_defaults(func=cmd_clinical_readiness)

    for name in TRACK_STUB_SUBCOMMANDS:
        p = sub.add_parser(name, help=f"Stub — {name} pipeline integration is not wired up yet.")
        p.add_argument("--run-dir", default=None)
        p.add_argument("--diagnostic-mode", action="store_true")
        p.set_defaults(func=lambda args, _name=name: cmd_track_stub(args, _name))

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except EvidenceRunnerUsageError as exc:
        print(f"[evidence.runner] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
