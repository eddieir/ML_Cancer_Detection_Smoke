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
  never trains anything), `development` (real orchestration via
  `evidence/development.py::run_development` — today wired up for
  `gse123352`/`smoke_classification`; any other cohort/task combination
  returns a structured, specifically-reasoned `not_evaluable`), and
  `internal-test`/`external-test` (honest gates via
  `run_internal_test`/`run_external_test` — no cohort has ever had a
  frozen internal-test partition created and guarded, and none carries
  `role_eligibility=[external_validation]`, so both always return a
  specifically-reasoned `not_evaluable`, never a fabricated result;
  `internal-test --cohort gse123352 --task smoke_classification` reports
  the real computed `assess_frozen_internal_test_eligibility()` decision —
  real subject/class counts against a versioned support policy — instead
  of the generic gate).
  `--diagnostic-mode` is accepted only by `audit` (a no-op there) and
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
(`data/raw/`, gitignored — never committed). Real Track A results have
been executed against them:

- **GSE123352 verified binary bulk smoke-history classification (real
  data, real labels).** GSE123352 is bulk microarray —
  `src/data/bulk_pipeline.py` is a separate path (never touches the
  single-cell MIL pipeline; see `data/assay_policy.py`) that parses a real
  per-sample, independently-verified **lifetime ever-versus-never
  cigarette-smoking history** from the real downloaded series matrix.
  Subject identity is independently verified from GEO `Sample_title`'s
  `patient_<N>` field (`data/converters.py::_infer_subject_id_column`) — a
  sample whose subject identity cannot be verified is excluded, never
  trusted; `parse_strict_bool` rejects any malformed/ambiguous
  `smoke_type_known` value outright. All 176 real subjects carried a
  verified label and verified subject identity in the most recently
  validated run (2026-07-23, commit
  `c28eeee166ae98b3198ede46bcfc426bdd62a4f7`). A single 70/30
  subject-level split (124/50) with a train-only-fit logistic regression
  scored **macro-F1 0.653, balanced accuracy 0.700, accuracy 0.66**
  (confusion matrix `[[13,3],[14,20]]`, class counts unexposed=16/
  cigarette=34) on the 50 held-out real test subjects — labeled a
  single-split **exploratory** result, not the primary estimate.
  `evidence.development.run_gse123352_repeated_development()` adds a
  repeated grouped-development-holdout protocol (8 predeclared seeds in
  `configs/evidence.yaml`'s `development_repeat_seeds`, fold-local C
  selection using only each seed's own outer-train partition, subject-level
  bootstrap confidence intervals per seed via `evidence/uncertainty.py`,
  reused not reimplemented). Same validated run, all 8 seeds completed:
  **macro-F1 mean 0.726 (std 0.037), balanced accuracy mean 0.728 (std
  0.044)** — this is the current primary development estimate for
  GSE123352. Alongside the primary metric, the same run reports:
  - **required baselines**, fit on the identical per-seed subject partition
    as the candidate (`evidence.development._baseline_predictions`) —
    majority-class, constant-prevalence-probability, and an untuned
    all-gene bulk logistic regression — compared via
    `evidence.uncertainty.paired_candidate_comparison`. The candidate beats
    majority/prevalence on 8/8 seeds (mean macro-F1 diff +0.321); it beats
    the untuned all-gene linear baseline on 4/8 seeds and loses on 4/8
    (mean diff +0.021) — reported honestly as a small, mixed margin, not
    rounded into a clean win. No clinical/metadata baseline is reported
    because no legitimate non-leaking clinical covariate exists for this
    cohort beyond expression itself.
  - **the full per-seed metric bundle**
    (`evidence.development._full_metric_bundle`): balanced accuracy,
    weighted F1, per-class precision/recall/F1/support, confusion matrix,
    AUROC, AUPRC, Brier score, log loss, and expected calibration error —
    every metric undefined for a seed's class support stays `None` with an
    explicit reason, never silently 0.
  - **a development-only calibration+threshold pathway**
    (`benchmarks/calibration.py::build_frozen_policy`, reused not
    reimplemented): fit on the SAME inner train/validation split each
    seed's fold-local C-selection already builds, applied exactly once to
    outer test via `FrozenThresholdPolicy.apply_to_test`. Because no
    outer-train-disjoint data remains to calibrate the primary
    full-outer-train candidate without touching outer test, this pathway
    reports calibration for a model fit on the inner-train partition only
    (~75% of each seed's outer-train subjects) — explicitly labeled as
    such, never conflated with the primary candidate's own probabilities.
  - a deterministic **frozen internal-test eligibility decision**
    (`evidence.development.assess_frozen_internal_test_eligibility`,
    policy in `configs/evidence.yaml`'s `frozen_internal_test_policy`):
    GSE123352's 176 verified subjects (58 minority-class) fall below the
    configured minimum (300 total / 50 per class), so the result is
    `eligible=False` with the real computed counts — not a bare "no
    partition exists" statement.

  `run_track_a_on_real_verified_label_data` stamps the single-split result
  `evidence_level=development_only_real_data`, `cohort_role=development`,
  `split_role=development_holdout` — `gse123352`'s `role_eligibility` is
  `[development]` only; this is not an internal-held-out or external claim.
  Report artifacts: `artifacts/evidence/gse123352_verified_label_smoke_v2/`
  (single-split), `artifacts/evidence/gse123352_repeated_development_v3/`
  (repeated, baselines, full metrics, calibration, frozen-test eligibility —
  both gitignored); sanitized publication summaries:
  `evidence/published/gse123352_verified_label_smoke_v2/summary.json`
  (single-split) and
  `evidence/published/gse123352_repeated_development_v3/summary.json`
  (repeated).
- **GSE136831 COPD-vs-Control disease-status proxy sensitivity analysis
  (real data, explicitly NOT smoke-classification evidence).** GSE136831
  carries no verified per-subject cigarette-exposure field.
  `scripts/run_gse136831_copd_control_proxy_analysis.py` runs the real
  single-cell pipeline end to end; 46 real COPD/Control subjects (32 IPF
  subjects excluded — IPF is not a smoking-relevant comparator),
  subject-level split 33/7/6, cells subsampled to 300/subject for this
  environment's memory. Result:
  `evidence.tracks.run_copd_control_proxy_analysis` —
  `task=exploratory_disease_proxy_analysis` (never `smoke_classification`),
  `verified_label_count=0` always, macro-F1 1.0 on 6 development-holdout
  subjects, but both classes are below the 5-subject learnability
  threshold, so no learnability claim is made either way. COPD is a
  clinical diagnosis with strong smoking association but also documented
  non-smoking causes; Control does not prove never-smoking — this result
  can never be merged with, aggregate into, or be selected in place of a
  verified-label smoke-classification result, and cannot satisfy any
  clinical-readiness dimension. Report artifact:
  `artifacts/evidence/gse136831_copd_control_proxy_v1/`.
- **TCGA-LUAD+TCGA-LUSC vital-status subject-level cancer-outcome
  prediction (real data, real linked outcome, Track C).** The first cohort
  in this repository with genuinely linked, subject-level expression<->
  cancer-outcome data: real primary-tumor bulk RNA-seq (GDC "STAR - Counts",
  open-access) and real `vital_status` (Dead/Alive at last GDC follow-up)
  from the same case's demographic record — downloaded directly from
  `api.gdc.cancer.gov`, no DUA required
  (`data/converters.py::convert_tcga_vital_status`,
  `data/tcga_outcome_pipeline.py`, cohort `tcga_lung_vital_status` in
  `configs/cohorts.yaml`). Explicitly NOT a survival/time-to-event label —
  `evidence.tracks.run_track_c_on_real_tcga_vital_status_data` stamps
  `endpoint_fields.censoring_status='not_modeled_binary_vital_status_only'`
  on every report. 996 real subjects (602 alive / 394 dead) after excluding
  unknown-vital-status and cross-project-duplicate-case_id samples. A
  single 70/30 split scored AUROC ≈0.51 (near chance); the repeated
  grouped-development protocol (8 seeds, same fold-local C selection,
  baselines, full metric bundle, and calibration pathway as GSE123352)
  scored **macro-F1 mean 0.493 (std 0.015)** — a real but weak signal.
  Paired against the required baselines on the identical per-seed
  partition: beats majority/prevalence (mean diff +0.116, 8/8 seeds) but
  **loses to the untuned all-gene linear baseline on 6/8 seeds** (mean diff
  −0.026) — reported as found. Unlike GSE123352, this cohort's real counts
  (996 total, 394 minority-class) clear the configured frozen-internal-test
  support policy (`eligible=True`) — no frozen partition has been created
  yet, but the computed decision genuinely differs from GSE123352's.
  Report artifacts: `artifacts/evidence/tcga_lung_vital_status_v1/`
  (single-split), `artifacts/evidence/tcga_lung_vital_status_repeated_development_v1/`
  (repeated — both gitignored); sanitized publication summaries:
  `evidence/published/tcga_lung_vital_status_v1/summary.json` and
  `evidence/published/tcga_lung_vital_status_repeated_development_v1/summary.json`.

What remains unimplemented past these results is, in every case, **not a
missing piece of code — it is the absence of a suitable real dataset file,
or a real cross-assay/expression-outcome linkage, in this environment**:

- **Four of eight registered cohorts still have no real local files
  present, or remain controlled-access** (GSE288003, GSE307690/CANUCK,
  NLST — controlled-access, no DUA obtained). `tcga_luad`/`tcga_lusc`'s raw
  files ARE now present (used by `tcga_lung_vital_status`); their
  originally-scoped tumor/normal `malignancy_classification` task remains
  `not_currently` supported (no bulk malignancy-classification pipeline
  exists). `python -m evidence.audit` reports `partial_local_data_present`.
- **Track B (malignancy) still has nothing eligible to evaluate**: no
  single-cell cohort carries a malignancy label. `run_track_b_against_registry()`
  remains honest `not_evaluable`. Its `..._on_fixture()` path still runs the
  real metric-computation code end-to-end only against synthetic data,
  stamped `synthetic_flag=True`.
- **Track C (subject-level cancer prediction) now has one real, wired
  cohort** (`tcga_lung_vital_status`, above) — `run_track_c_against_registry()`
  now finds it structurally eligible and returns `REAL_FITTING_NOT_IMPLEMENTED`
  only in the sense that the registry-consulting path itself never opens a
  dataset file (by design — see this module's docstring); the real
  orchestration lives in `evidence.development.run_development`/
  `run_tcga_lung_vital_status_repeated_development`, both of which DO run
  against real data. True time-to-event survival prediction (with
  censoring/follow-up duration) remains unimplemented — no cohort in this
  repository carries genuine censoring data.
- **Candidate comparison (Step 9) and subgroup diagnostics (Step 13) remain
  framework-only.** `run_candidate_comparison_on_synthetic_fixture()` and
  `subgroup_report()` run the real comparison/statistics code against real
  Phase 1-6 training machinery, but only against a synthetic development
  context (`benchmarks.runner.build_synthetic_context`) — no real
  candidate comparison or subgroup breakdown has been produced anywhere in
  this repository. Repeated-resampling uncertainty (Step 10) IS now
  exercised against real data for GSE123352 —
  `evidence.development.run_gse123352_repeated_development()` reuses
  `evidence/uncertainty.py`'s `RepeatRecord`/`repeated_metric_summary`/
  `subject_level_bootstrap_ci` against real repeated held-out predictions.
- **Calibration/thresholding (Step 14) has been run against real,
  development-only GSE123352 predictions** (see above — a per-seed
  calibration+threshold pathway fit on an inner train/validation split of
  each seed's own outer-train partition), but there are still no real
  internal-test/external-test predictions to fit or evaluate a FROZEN
  calibrator on — that remains blocked on the frozen-test-eligibility gap
  above, not on missing code.
- **`evidence.runner internal-test`/`external-test`** are honest,
  specifically-reasoned gates (`NO_FROZEN_TEST_PARTITION_EXISTS`,
  `NO_ELIGIBLE_EXTERNAL_COHORT`) — no cohort has ever had a frozen
  internal-test partition created and guarded, and none carries
  `role_eligibility=[external_validation]`.
- **Step 18's adversarial test matrix** is complete for every framework
  module above. The real Track A/development results above are exercised
  by their own scripts/module tests against small synthetic fixtures for
  speed (CI does not depend on the multi-gigabyte real downloads); the
  real runs themselves were executed once, by hand, against the real
  downloaded data, and their artifact bundles validate cleanly via
  `evidence.runner validate`.

No further engineering round changes real cancer-prediction, malignancy,
external-validation, or clinical-readiness status: those blockers are
genuinely about missing linkage/authorization/cohorts, not missing
framework code or missing local files for the two cohorts already
downloaded.

## Why this is the honest outcome

Two of seven registered cohorts (GSE136831, GSE123352) have real local
files in this environment. GSE123352 has a genuine, checksummed,
development-only verified-label smoke-history result (single-split
exploratory plus a repeated grouped-development-holdout estimate).
GSE136831 has a genuine, checksummed, EXPLICITLY NON-SMOKE COPD-vs-Control
disease-status proxy analysis — it does not and cannot contribute to
verified smoke evidence. The other five cohorts remain undownloaded or
(for NLST) unauthorized, and no cohort anywhere has a malignancy label or
an expression<->outcome linkage, so Track B, Track C, external validation,
and clinical readiness remain genuinely `not_evaluable`/`not_established`
regardless of further downloads. Per the specification: state plainly what
is real, what is a non-smoke proxy, and what is still blocked
— never merge the three. That is exactly the state this change leaves the
repository in.
