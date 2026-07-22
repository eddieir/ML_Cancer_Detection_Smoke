"""
evidence — Phase 7 real-world evidence and clinical-readiness framework.

This package is deliberately separate from `benchmarks` (the synthetic/
development benchmark-runner infrastructure from Phases 1-6). It defines:

  - errors.py             typed exceptions for evidence-contract violations
  - evidence_contract.py  the versioned evidence report schema + validation
  - cohort_registry.py    the canonical cohort/endpoint registry (configs/cohorts.yaml)
  - eligibility.py        dataset/task eligibility gates (Step 6)
  - audit.py              read-only real-data audit CLI (`python -m evidence.audit`)
  - clinical_readiness.py the 22-dimension clinical-readiness assessment

None of this package trains models, fits preprocessing, or touches raw
subject records. It reports on what evidence exists (or does not) and
refuses, structurally, to let an absence of evidence be reported as a
result.
"""
