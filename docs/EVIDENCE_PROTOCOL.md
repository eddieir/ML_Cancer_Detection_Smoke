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
- `tests/test_evidence_leakage_isolation.py` — Step 8: corruption-isolation
  tests reusing `benchmarks/sentinel.py`'s poison-object pattern, proving
  that development-only code paths in this evidence framework (tracks,
  calibration fitting, uncertainty resampling) cannot read a held-out/test
  partition even adversarially.
- `src/evidence/candidate_comparison.py` — Step 9: identical-partition
  candidate comparison across the classical baselines and MIL-kind models
  already registered in `benchmarks/candidate_registry.py`. Every candidate
  for a task shares one `run_smoke_cv`/`run_cancer_cv` call, so the fold
  assignment, per-fold-refit preprocessing artifact, and label mapping are
  identical across candidates by construction. `select_best_candidate()`
  raises `CandidateSelectionError` rather than falling back to a secondary
  metric when no candidate has a defined primary metric.
- `src/evidence/uncertainty.py` — Step 10: repeated grouped-resampling
  uncertainty reporting, structurally confined to development data
  (`DevelopmentRepeatedOOF.role` must be the literal `"development"`).
  Subject-level bootstrap CIs, seed-level aggregation via
  `benchmarks.metrics.aggregate_metric_by_seed`, and a paired candidate
  comparison. `confidence_intervals_overlap()` is explicitly labeled
  `interpretation='descriptive_only'` — never a formal equivalence test.
- `src/evidence/subgroups.py` — Step 13: subgroup/fairness diagnostics over
  the five subgroup dimensions this repository actually has a genuine data
  source for (`cohort_source`, `exposure_type`, `disease_status`,
  `assay_platform`, `species`); requesting any other dimension (e.g. sex,
  age band, race/ethnicity, site — none of which exist anywhere in this
  repository's data model) raises `UnsupportedSubgroupDimensionError`
  rather than fabricating one. `assert_no_banned_claims()` rejects any
  rendered summary asserting fairness has been established.

Each of the above has a dedicated adversarial test file under `tests/`
(`test_evidence_contract.py`, `test_cohort_registry.py`,
`test_evidence_eligibility.py`, `test_evidence_audit.py`,
`test_clinical_readiness.py`, `test_evidence_tracks.py`,
`test_external_validation.py`, `test_evidence_calibration.py`,
`test_artifact_bundle.py`, `test_evidence_runner.py`,
`test_evidence_leakage_isolation.py`, `test_candidate_comparison.py`,
`test_evidence_uncertainty.py`, `test_evidence_subgroups.py`).

## What is not implemented

Every framework module for Steps 1-16 of the Phase 7 specification now
exists and is tested against synthetic fixtures: the evidence contract,
cohort registry, eligibility gates, audit CLI, evaluation tracks (A/B/C),
clinical-readiness assessment, the external-validation sentinel/gate,
leakage corruption-isolation tests, candidate comparison, repeated-
resampling uncertainty reporting, subgroup/fairness diagnostics, frozen
calibration/thresholding, the artifact bundle writer/reader, and the
`evidence.runner` umbrella CLI.

GSE136831 and GSE123352 are downloaded locally in this environment
(`data/raw/`, gitignored — never committed). Two real Track A runs have
been executed against them:

- **GSE136831 weak-label smoke-exposure experiment (real data, weak
  proxy).** `scripts/run_gse136831_weak_label_evidence.py` runs the real
  single-cell pipeline end to end; 46 real COPD/Control subjects (32 IPF
  subjects excluded), subject-level split 33/7/6, cells subsampled to
  300/subject for this environment's memory. Result:
  `run_track_a_on_real_weak_label_data` — macro-F1 1.0 on 6 held-out test
  subjects, but both classes are below the 5-subject learnability
  threshold, so no learnability claim is made. `weak_label_experiment=True`
  always; never merged with a verified-label result. Report artifact:
  `artifacts/evidence/gse136831_weak_label_smoke_v1/`.
- **GSE123352 verified-label smoke classification (real data, real
  labels).** GSE123352 is bulk microarray — `src/data/bulk_pipeline.py` is
  a new, minimal, separate path (never touches the single-cell MIL
  pipeline; see `data/assay_policy.py`) that parses real per-sample
  `ever_never_smoker` labels from the real downloaded series matrix. All
  176 real subjects carried a verified label; 126/50 subject-level
  train/test split; a train-only-fit logistic regression scored
  **macro-F1 0.592, balanced accuracy 0.605** on the 50 held-out real test
  subjects. `run_track_a_on_real_verified_label_data` stamps this
  `evidence_level=development_only_real_data`,
  `cohort_role=development` — `gse123352`'s `role_eligibility` is
  `[development]` only; this is not an internal-held-out or external
  claim. Report artifact:
  `artifacts/evidence/gse123352_verified_label_smoke_v1/`.

What remains unimplemented past these two runs is, in every case, **not a
missing piece of code — it is the absence of a suitable real dataset file,
or a real cross-assay/expression-outcome linkage, in this environment**:

- **Five of seven registered cohorts still have no real local files
  present** (GSE288003, GSE307690/CANUCK, TCGA-LUAD, TCGA-LUSC, NLST — the
  last controlled-access). `python -m evidence.audit` reports
  `partial_local_data_present` given GSE136831/GSE123352 are present.
- **Track B (malignancy) and Track C (subject-level cancer prediction)
  still have nothing eligible to evaluate**, independent of which cohorts
  are downloaded: no single-cell cohort carries a malignancy label, and no
  cohort has a genuine expression<->outcome linkage
  (`expression_outcome_linkable_at_subject_level: false` everywhere in
  `configs/cohorts.yaml`). `run_track_b/c_against_registry()` remain
  honest `not_evaluable`. Their `..._on_fixture()` paths still run the
  real metric-computation code end-to-end only against synthetic data,
  stamped `synthetic_flag=True`.
- **Candidate comparison (Step 9), repeated-resampling uncertainty
  (Step 10), and subgroup diagnostics (Step 13) are consequently framework-
  only.** `run_candidate_comparison_on_synthetic_fixture()`,
  `repeated_development_oof_for_cancer_track()`, and `subgroup_report()`
  all run the real comparison/statistics code against real Phase 1-6
  training machinery, but only against a synthetic development context
  (`benchmarks.runner.build_synthetic_context`) — no real candidate
  comparison, uncertainty estimate, or subgroup breakdown has been
  produced anywhere in this repository.
- **Calibration/thresholding (Step 14) and the artifact bundle (Step 15)
  have never been run against real predictions**, for the same reason —
  there are no real development/internal-test/external-test predictions
  to fit a frozen calibrator on or bundle into an artifact run.
- **`evidence.runner development`/`internal-test`/`external-test`** are
  honest stubs (`not_evaluable(reason_code="TRACK_RUNNER_NOT_YET_INTEGRATED")`)
  — `audit`, `inspect`, `validate`, and `clinical-readiness` are fully
  implemented and read-only.
- **External-validation release against a REAL external cohort (Step 11)**
  has not happened — no cohort in `configs/cohorts.yaml` currently has
  `role_eligibility` including `external_validation`.
- **Step 18's adversarial test matrix** is complete for every framework
  module above (1535 tests passing, including the leakage-isolation suite
  in `tests/test_evidence_leakage_isolation.py`). The two real Track A
  runs above are exercised by their own scripts/module tests against small
  synthetic fixtures for speed (CI does not depend on the multi-gigabyte
  real downloads); the real runs themselves were executed once, by hand,
  against the real downloaded data, and their artifact bundles validate
  cleanly via `evidence.runner validate`.
- **External-validation release against a REAL external cohort (Step 11)**
  has not happened — no cohort in `configs/cohorts.yaml` currently has
  `role_eligibility` including `external_validation`, independent of the
  two development-only real results above.
- **Candidate comparison (Step 9), repeated-resampling uncertainty
  (Step 10), subgroup diagnostics (Step 13), calibration/thresholding
  (Step 14), and `evidence.runner development`/`internal-test`/
  `external-test`** remain framework-only, run only against synthetic
  development contexts — none of them has been run against the two real
  Track A results above yet; that is genuine remaining work, not a data
  blocker.

No further engineering round changes real cancer-prediction, malignancy,
external-validation, or clinical-readiness status: those blockers are
genuinely about missing linkage/authorization/cohorts, not missing
framework code or missing local files for the two cohorts already
downloaded.

## Why this is the honest outcome

Two of seven registered cohorts (GSE136831, GSE123352) have real local
files in this environment, and both now have a genuine, once-executed,
checksummed real-data Track A result — one weak-label, one verified-label,
both `development`-role only. The other five cohorts remain undownloaded
or (for NLST) unauthorized, and no cohort anywhere has a malignancy label
or an expression<->outcome linkage, so Track B, Track C, external
validation, and clinical readiness remain genuinely `not_evaluable`/
`not_established` regardless of further downloads. Per the specification:
state plainly what is real, what is weak-proxy, and what is still blocked
— never merge the three. That is exactly the state this change leaves the
repository in.
