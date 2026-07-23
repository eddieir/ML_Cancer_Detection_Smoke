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
  - `development` is real orchestration (evidence/development.py::
    run_development) for cohort/task combinations that have a genuine
    dataset-specific pipeline wired up (today: gse123352/
    smoke_classification, via data/bulk_pipeline.py). Any other cohort/task
    combination returns a structured not_evaluable naming the specific
    reason (no eligible cohort, or no pipeline wired up yet).
  - `internal-test` / `external-test` are honest gates
    (evidence/development.py::run_internal_test / run_external_test): no
    cohort in configs/cohorts.yaml has ever had a frozen internal-test
    partition created and guarded, and none carries
    role_eligibility=[external_validation], so both subcommands always
    return a structured, specifically-reasoned not_evaluable rather than a
    generic stub or a fabricated result.
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
DEFAULT_COHORT_CONFIG = "configs/cohorts.yaml"
REAL_SUBCOMMANDS = (
    "clinical-readiness", "inspect", "validate", "development", "internal-test", "external-test",
)


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


# ─── development — real orchestration for wired-up cohort/task pairs ──────

def cmd_development(args: argparse.Namespace) -> int:
    _reject_diagnostic_mode(args, "development")
    from .cohort_registry import load_cohort_registry
    from .development import run_development
    from .evidence_contract import is_not_evaluable

    cohorts = load_cohort_registry(args.cohort_config)
    result = run_development(
        args.cohort, args.task, cohorts=cohorts, output_root=args.output_root,
        run_id=args.run_id, seed=args.seed, train_frac=args.train_frac,
        test_frac=args.test_frac, n_top_variance_genes=args.n_top_variance_genes,
    )
    _print(result)
    return 1 if is_not_evaluable(result) else 0


# ─── internal-test / external-test — honest gates, not stubs ──────────────

def cmd_internal_test(args: argparse.Namespace) -> int:
    _reject_diagnostic_mode(args, "internal-test")
    from .development import run_internal_test
    from .evidence_contract import is_not_evaluable
    result = run_internal_test(
        args.run_dir, args.authorization_file, cohort_id=args.cohort, task=args.task,
    )
    _print(result)
    return 1 if is_not_evaluable(result) else 0


def cmd_external_test(args: argparse.Namespace) -> int:
    _reject_diagnostic_mode(args, "external-test")
    from .development import run_external_test
    result = run_external_test(args.run_dir, args.external_cohort)
    _print(result)
    return 1


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

    p_dev = sub.add_parser("development", help="Real development orchestration for wired-up cohort/task pairs.")
    p_dev.add_argument("--cohort", required=True)
    p_dev.add_argument("--task", required=True)
    p_dev.add_argument("--config", default="configs/default.yaml")
    p_dev.add_argument("--evidence-config", default="configs/evidence.yaml")
    p_dev.add_argument("--cohort-config", default=DEFAULT_COHORT_CONFIG)
    p_dev.add_argument("--output-root", default="artifacts/evidence")
    p_dev.add_argument("--run-id", default=None)
    p_dev.add_argument("--seed", type=int, default=42)
    p_dev.add_argument("--train-frac", type=float, default=0.70)
    p_dev.add_argument("--test-frac", type=float, default=0.30)
    p_dev.add_argument("--n-top-variance-genes", type=int, default=2000)
    p_dev.add_argument("--diagnostic-mode", action="store_true")
    p_dev.set_defaults(func=cmd_development)

    p_internal = sub.add_parser("internal-test", help="Frozen internal-test gate (currently always blocked — see module docstring).")
    p_internal.add_argument("--run-dir", required=True)
    p_internal.add_argument("--authorization-file", default=None)
    p_internal.add_argument("--cohort", default=None, help="If given with --task, reports the real computed frozen-test eligibility decision for this cohort/task instead of the generic gate.")
    p_internal.add_argument("--task", default=None)
    p_internal.add_argument("--diagnostic-mode", action="store_true")
    p_internal.set_defaults(func=cmd_internal_test)

    p_external = sub.add_parser("external-test", help="External-validation gate (currently always blocked — see module docstring).")
    p_external.add_argument("--run-dir", required=True)
    p_external.add_argument("--external-cohort", default=None)
    p_external.add_argument("--diagnostic-mode", action="store_true")
    p_external.set_defaults(func=cmd_external_test)

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
