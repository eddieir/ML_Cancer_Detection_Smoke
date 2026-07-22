# Clinical readiness

This document is generated from, and must stay consistent with,
`src/evidence/clinical_readiness.py` and `configs/clinical_readiness.yaml`.
It is not a certificate and does not authorize any clinical use.

## Statements that hold regardless of any number in this repository

- This is research software.
- This is not a medical device.
- It is not validated for diagnosis, screening, prognosis, or treatment.
- A model probability is not an individual clinical risk estimate unless
  calibration and target-population validation establish that it is.
- Retrospective public-dataset performance cannot establish clinical
  utility.
- Prospective and independent external validation remain necessary before
  any clinical claim would be appropriate.
- Regulatory readiness requires expert legal/regulatory review outside
  this repository; nothing here performs or substitutes for that review.

## Overall status

**`clinically_not_ready`.**

`assess_clinical_readiness()` computes this purely from the 22 dimension
records below — there is no parameter or flag anywhere in this codebase
that can set overall status directly. As of the last commit reviewed
against this document, no dimension below has real, cited evidence
supporting a `complete` status, so every mandatory dimension is
`not_started` or `blocked`.

## The 22 dimensions

| Dimension | Mandatory | Status | Why |
|---|---|---|---|
| Intended use | yes | not_started | No intended-use statement has been reviewed and signed off. |
| Target population | yes | not_started | No target population has been defined or characterized against real cohort demographics. |
| Clinical setting | yes | not_started | No clinical setting has been specified. |
| Prediction target and time horizon | yes | not_started | Track C (subject-level cancer prediction) is not_evaluable — see `docs/EVIDENCE_PROTOCOL.md`. |
| Input specimen and assay | yes | partial | Specimen/assay are well-documented per cohort (`configs/cohorts.yaml`), but no cohort currently supports the full pipeline end to end on real data. |
| Data provenance | yes | partial | `configs/datasets.yaml` / `src/data/manifest.py` provide real provenance and checksums for whichever files are actually present; none are present in this environment. |
| Analytical validity | yes | blocked | Requires a real held-out evaluation, which does not exist. |
| Internal validation | yes | blocked | No frozen-test evaluation against real data has been run. |
| External validation | yes | blocked | No eligible external cohort has been identified (see `configs/cohorts.yaml` role_eligibility). |
| Calibration | yes | blocked | Requires real development OOF predictions, which do not exist. |
| Clinical utility | yes | not_started | Out of scope without prospective data. |
| Fairness / subgroup performance | yes | blocked | Requires real, verified subgroup metadata, which is not available. |
| Robustness to site/platform/domain shift | yes | partial | Phase 6 domain-robustness machinery exists and is tested against synthetic data; not run against multiple real sources yet. |
| Missing-data handling | no | partial | Unknown-label exclusion is enforced in code (Phase 3) and tested; not evaluated against a real missingness pattern. |
| Failure detection and abstention | no | not_started | Not implemented. |
| Reproducibility | yes | partial | Fingerprinting/versioned artifacts exist (Phases 4-6, extended here); no real end-to-end run to reproduce yet. |
| Model/version governance | no | partial | `src/benchmarks/model_fingerprint.py` and `src/benchmarks/bundle.py` exist; no deployed-version registry. |
| Privacy/security | yes | partial | No participant-level data is committed; no formal privacy/security review has been performed. |
| Human-factors/usability validation | no | not_started | Not applicable without a deployed interface. |
| Prospective validation | yes | not_started | Not implemented; explicitly out of scope for this repository. |
| Regulatory pathway | yes | not_started | No regulatory strategy has been defined; requires expert review outside this repository. |
| Post-deployment monitoring | no | not_started | Not applicable — nothing is deployed. |

## What would change this

Real held-out and external evidence, produced through `src/evidence/`,
recorded with real `evidence_references` in each dimension, reviewed by
someone qualified to assess clinical and regulatory readiness, and backed
by a signed external-evidence manifest before `guard_clinical_claim()`
would allow any downstream code to state readiness at all. No single
metric improvement changes this status by itself.
