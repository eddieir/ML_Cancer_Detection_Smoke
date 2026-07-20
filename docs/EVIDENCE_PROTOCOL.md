# Evidence protocol

This document describes the Phase 7 evidence framework — what it enforces,
what it currently produces in this repository, and what remains
unimplemented. It is meant to be read alongside
`src/evidence/evidence_contract.py`, `configs/evidence.yaml`, and the
Phase 7 section of `README.md`.

## The contract

Every metric or report this framework can produce is either:

1. A fully-identified `EvidenceReport` — 29 required identity fields
   (task, endpoint, endpoint definition, prediction unit, specimen type,
   assay modality, dataset accession, cohort role, species, sample/subject
   counts, class/verified/unknown/excluded counts, split role, five
   provenance fingerprints, random seed, commit SHA, environment
   fingerprint, synthetic/development-only flags, frozen-test-access
   status, evidence level, limitations, timestamp, schema version), plus a
   content fingerprint over the whole report — validated by
   `evidence_contract.validate_report()`, or
2. A structured `not_evaluable({"status", "reason_code", "reason",
   "required_next_action"})` object.

There is no third representation. `not_evaluable` is never encoded as
`0`, `False`, an empty metric dict, `NaN` without explanation, or a
"successful" result carrying a warning.

## Evidence levels

`synthetic_software_validation` → `development_only_real_data` →
`internal_held_out_real_data` → `external_retrospective_validation` →
`prospective_validation` → `clinical_utility_validation` →
`regulatory_evidence`. A report's own flags gate which levels it may
claim — see `validate_identity()` for the exact rules (synthetic reports
are confined to level 1; development-only reports can never claim level 3
or above; external+ levels require `cohort_role=external_validation`).

## What is implemented in this change

- `src/evidence/errors.py` — 13 typed exceptions for the specific
  contract violations enumerated in the Phase 7 specification.
- `src/evidence/evidence_contract.py` + `configs/evidence.yaml` — the
  report schema, identity validation, fingerprint round-trip/tamper
  detection, and the `not_evaluable` helper.
- `src/evidence/cohort_registry.py` + `configs/cohorts.yaml` — the
  canonical cohort/endpoint registry for all seven cohorts this repository
  knows about, cross-checked against `configs/datasets.yaml`.
- `src/evidence/eligibility.py` — deterministic per-(task, cohort)
  eligibility gates (impossible / exploratory-only / internal-eligible /
  external-eligible), driven only by real counts supplied by the caller.
- `src/evidence/audit.py` — the read-only `python -m evidence.audit` CLI.
- `src/evidence/clinical_readiness.py` + `configs/clinical_readiness.yaml`
  + `docs/CLINICAL_READINESS.md` — the 22-dimension staged assessment and
  the `guard_clinical_claim()` choke point.

Each of the above has a dedicated adversarial test file under `tests/`
(`test_evidence_contract.py`, `test_cohort_registry.py`,
`test_evidence_eligibility.py`, `test_evidence_audit.py`,
`test_clinical_readiness.py`).

## What is not implemented

The following pieces of the full Phase 7 specification are **not**
implemented in this change, in every case because they depend on real
evaluable data that is not present in this environment, or because they
require substantial additional engineering beyond what was completed
here:

- **Tracks A/B/C evaluation runners.** No code path in this change trains
  a model, runs cross-validation, or produces an actual smoke/malignancy/
  cancer-prediction metric against real data. The audit CLI establishes
  that no cohort currently has real local files present, so every
  eligibility row is `IMPOSSIBLE` — there is nothing for a Track A/B/C
  runner to evaluate yet.
- **Leakage-safe pipeline integration (Step 8 order) for real data.** The
  existing Phase 1-6 safeguards (`src/data/splitting.py`,
  `src/benchmarks/test_guard.py`, `src/benchmarks/sentinel.py`) are
  unmodified and still enforced for the existing synthetic/benchmark path;
  this change does not wire a new real-data pipeline through them.
- **Complete candidate comparison (Step 9) for real data**, since it
  requires Track A/B/C actually running.
- **Repeated development estimates and uncertainty (Step 10)** for real
  data, for the same reason.
- **External-validation sentinel (Step 11)** — no external cohort is
  currently eligible (see `configs/cohorts.yaml`), so there is nothing to
  guard against premature access yet; the typed
  `ExternalValidationRequiredError` exists in `errors.py` but is not wired
  into a runnable external-test code path.
- **Subgroup/fairness diagnostics (Step 13), calibration/thresholding
  (Step 14) against real predictions, and the full
  `artifacts/evidence/<run_id>/` bundle structure (Step 15)** — all
  require real development/test predictions that do not exist.
- **`evidence.runner` CLI (Step 16)** beyond `evidence.audit` and the
  clinical-readiness module — `development`, `internal-test`,
  `external-test`, `inspect`, and `validate` subcommands are not
  implemented.
- **Step 18's full adversarial test matrix** — the subset covering the
  evidence contract, cohort registry, eligibility gates, audit CLI, and
  clinical-readiness framework is implemented and passing; the subset
  covering splitting/leakage corruption-isolation against a real pipeline,
  metric-manual-calculation tests against real predictions, and
  artifact-checksum tests against a real `artifacts/evidence/<run_id>/`
  bundle is not, because the underlying pipeline (above) is not
  implemented.

## Why this is the honest outcome

Running `PYTHONPATH=src python -m evidence.audit` against this repository,
with no datasets downloaded and no NLST DUA obtained, returns
`overall_status: "blocked_no_real_data"`. Per the specification: "If no
suitable real dataset is locally available (this is the expected case —
check first) ... produce a complete data audit and not_evaluable evidence
report ... state clearly that real evidence remains blocked by data
availability." That is exactly the state this change leaves the
repository in — infrastructure and audited-blocker reporting, not
manufactured evidence.
