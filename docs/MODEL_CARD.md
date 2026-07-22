# Model card

## Overview

This repository implements single-cell RNA-seq classifiers for two
research tasks: subject-level smoke-exposure classification, and
subject-level cancer prediction, using multiple-instance-learning (MIL)
architectures (mean/max/gated-attention pooling, and a pathway-hierarchical
MIL variant — see `src/pathway_hierarchical_mil.py`). Several classical
baselines (logistic regression, random forest, small MLP) are compared
under the same subject-level splits via `src/benchmarks/`.

## Intended use

Research and methods development only. **This is not a medical device and
is not intended for diagnostic, screening, prognostic, or treatment use.**
See `docs/CLINICAL_READINESS.md` for the full staged assessment; overall
status is `clinically_not_ready`.

## Training data

Public single-cell and bulk GEO/TCGA accessions and, where authorized, the
controlled-access NLST cohort — see `configs/datasets.yaml` and
`configs/cohorts.yaml` for the complete, per-cohort provenance and known
limitations, and `docs/DATA_CARD.md` for a narrative summary. No dataset
is representative of a clinical deployment population by default; that
would require evidence this repository does not currently have.

## Evaluation status

- Synthetic fixtures: exercised continuously via
  `PYTHONPATH=src python -m benchmarks.runner --synthetic --fast --task smoke|cancer`
  — proves the pipeline runs correctly end to end. No claim about
  real-world performance.
- Real data, training-set only (historical, Phase 1): 77.2% accuracy /
  0.27 macro-F1 on a merged GSE994+GSE307690 sample, computed before
  subject-level splitting existed — not held-out evidence, not compatible
  with the current assay-separated protocol, not a current performance
  estimate. Left in the README, clearly labeled, as a historical record.
- Real held-out evidence: **not established** — see `docs/EVIDENCE_PROTOCOL.md`.
- External validation: **not performed.**

## Limitations

- No cohort in this repository currently provides both compatible
  single-cell expression input and a genuinely linked subject-level cancer
  outcome (Track C is `not_evaluable`).
- Attention weights from MIL models are not causal biomarkers; pathway
  modules used by the pathway-hierarchical variant are not mechanistic
  proof of biological involvement — they are model-internal weightings,
  nothing more, absent independent experimental validation.
- Public retrospective datasets are not evidence of representativeness for
  any specific clinical deployment population.
- Small per-class subject counts in several cohorts make many per-class
  metrics statistically unreliable; see `src/benchmarks/eligibility.py`
  and `src/evidence/eligibility.py` for the exact gates applied.

## How to reproduce what exists today

```
source .venv/bin/activate
PYTHONPATH=src python -m benchmarks.runner --synthetic --fast --task smoke
PYTHONPATH=src python -m benchmarks.runner --synthetic --fast --task cancer
PYTHONPATH=src python -m evidence.audit --config configs/default.yaml \
    --cohort-config configs/cohorts.yaml --output artifacts/evidence/data_audit.json
```

The first two commands run the existing synthetic benchmark pipeline. The
third runs the Phase 7 read-only data audit and, in an environment with no
datasets downloaded, reports a structured blocked status rather than a
fabricated result.
