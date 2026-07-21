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
- `src/evidence/tracks.py` — the three Track A/B/C evaluation entry
  points, each with a real registry-checked `..._against_registry()` path
  (currently `not_evaluable` for every task in this repository, honestly)
  and a `..._on_fixture()` path that runs the real metric-computation code
  against a synthetic fixture (`synthetic_flag=True`).
- `src/evidence/external_validation.py` — Step 11: `ExternalCohortSentinel`
  (a poison object standing in for external-cohort data, following the
  pattern in `benchmarks/sentinel.py`), `ExternalValidationGate` (structural
  freeze-before-release: `release_external_cohort()` raises
  `ExternalValidationRequiredError` before `freeze_development()` has run),
  and `evaluate_external_cohort_eligibility()` /
  `select_external_validation_cohort()`, which distinguish a genuinely
  independent external cohort from an internal held-out split of the same
  cohort as development (`InternalSplitMislabeledAsExternalError`) versus
  no eligible external cohort existing at all
  (`not_evaluable(reason_code="NO_ELIGIBLE_EXTERNAL_COHORT")` — the
  expected state today, since no cohort in `configs/cohorts.yaml` has
  `role_eligibility` including `external_validation`).
- `src/evidence/calibration.py` — Step 14: `DevelopmentOOFPredictions` (the
  only data input `fit_frozen_calibration_and_threshold()` accepts),
  wrapping `benchmarks/calibration.py`'s real Platt/isotonic fitting code
  with an immutable, sha256-fingerprinted
  `FrozenCalibrationThresholdArtifact` (method, parameters, the exact
  selection-subject list/fingerprint, threshold objective/value/
  constraints). Isotonic regression is only offered once the development
  OOF set clears `configs/evidence.yaml`'s `calibration_policy.min_isotonic_n`
  (200 by default). Threshold methods: Youden's J, balanced-accuracy
  optimum, sensitivity-constrained, specificity-constrained, and a
  predeclared fixed threshold. `apply_frozen_calibration_and_threshold(artifact,
  y_prob)` has no fitting parameter at all (no `y_true`, no refit flag) —
  it can only transform probabilities into a decision at the frozen
  threshold. `decision_curve_analysis()` is a separately labeled
  exploratory helper whose output never flows into `evidence_level` or
  `clinical_readiness` claims.
- `src/evidence/artifact_bundle.py` — Step 15: `write_evidence_run()`
  atomically writes whichever files a run actually produced under
  `artifacts/evidence/<run_id>/`, computes a sha256 checksum for each into
  `checksums.json`, and writes `manifest.json` last with a `run_status`
  ("complete"/"incomplete") field. A run directory whose manifest already
  records `run_status="complete"` can never be written into again
  (`RunAlreadyCompleteError`). `validate_evidence_run()` is the read-only
  counterpart: it re-checksums every file `manifest.json` lists and raises
  `ArtifactValidationError` on anything missing or mismatched.
- `src/evidence/runner.py` — the `python -m evidence.runner` umbrella CLI:
  `audit` (thin wrapper around `evidence.audit.run_audit()`), `inspect` and
  `validate` (read-only, never write), `clinical-readiness` (re-validates
  the given run directory, then calls `assess_clinical_readiness()` —
  never trains anything), and honest `development`/`internal-test`/
  `external-test` stubs returning
  `not_evaluable(reason_code="TRACK_RUNNER_NOT_YET_INTEGRATED")` since the
  Steps 8-10 pipeline integration does not exist as a callable pipeline
  yet. `--diagnostic-mode` is accepted only by `audit` (a no-op there) and
  raises a typed `EvidenceRunnerUsageError` on every other subcommand.

Each of the above has a dedicated adversarial test file under `tests/`
(`test_evidence_contract.py`, `test_cohort_registry.py`,
`test_evidence_eligibility.py`, `test_evidence_audit.py`,
`test_clinical_readiness.py`, `test_evidence_tracks.py`,
`test_external_validation.py`, `test_evidence_calibration.py`,
`test_artifact_bundle.py`, `test_evidence_runner.py`).

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
- **Subgroup/fairness diagnostics (Step 13)** against real predictions —
  requires real development/test predictions that do not exist.
- **Calibration/thresholding (Step 14) and the artifact bundle (Step 15)
  against REAL predictions.** The framework (`evidence/calibration.py`,
  `evidence/artifact_bundle.py`) is implemented and tested against
  synthetic fixtures; no real development/internal-test/external-test
  predictions exist in this repository to run it against yet, because the
  Steps 8-10 pipeline integration does not exist as a callable pipeline.
- **`evidence.runner development`/`internal-test`/`external-test`** are
  honest stubs (`not_evaluable(reason_code="TRACK_RUNNER_NOT_YET_INTEGRATED")`)
  for the same reason — `audit`, `inspect`, `validate`, and
  `clinical-readiness` are fully implemented.
- **External-validation release against a REAL external cohort (Step 11).**
  `ExternalValidationGate`/`evaluate_external_cohort_eligibility()` are
  implemented and tested (with synthetic `Cohort` fixtures and against the
  real registry's current, expected-empty external-eligible set); no
  cohort in `configs/cohorts.yaml` currently has `role_eligibility`
  including `external_validation`, so there is no real external cohort to
  release in this environment yet.
- **Step 18's full adversarial test matrix** — the subset covering the
  evidence contract, cohort registry, eligibility gates, audit CLI,
  clinical-readiness framework, evaluation tracks, external-validation
  gate, calibration/thresholding, artifact bundle, and `evidence.runner`
  CLI is implemented and passing; the subset covering
  splitting/leakage corruption-isolation against a real pipeline and
  metric-manual-calculation tests against real (non-fixture) predictions
  is not, because the underlying real-data pipeline integration (Steps
  8-10) is not implemented.

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
