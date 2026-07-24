# Data card

Full machine-readable provenance lives in `configs/datasets.yaml` (raw
provenance/checksum facts) and `configs/cohorts.yaml` (Phase 7
task-eligibility facts). This document is a narrative summary; the YAML
files are authoritative.

## Cohorts

| Cohort | Species | Assay | Single-cell/bulk | Access | Verified smoke label | Cancer outcome | Notes |
|---|---|---|---|---|---|---|---|
| GSE136831 | human | scRNA-seq | single-cell | public | none (COPD is a weak proxy only) | none | Vanderbilt/Habermann ILD atlas; real per-donor cells, no verified cigarette field. |
| GSE288003 | mouse | scRNA-seq | single-cell | public | condition (Con/E-cigs), per-GSM | none | Species-separated by default; ortholog-mapped only under explicit cross-species mode. |
| GSE123352 | human | bulk microarray | bulk | public | ever/never smoker | none | Bulk — cannot enter single-cell MIL without a bulk pipeline this repository does not have. |
| GSE307690 (CANUCK) | human | bulk microarray/RNA-seq | bulk | public | cannabis/never smoker | none | Same bulk limitation as GSE123352. |
| TCGA-LUAD | human | bulk RNA-seq | bulk | public (open GDC) | none | tumor/normal sample_type only | Bulk tumour/normal status only; never per-cell malignancy; no verified smoking history. |
| TCGA-LUSC | human | bulk RNA-seq | bulk | public (open GDC) | none | tumor/normal sample_type only | Same as TCGA-LUAD. |
| TCGA-LUAD+TCGA-LUSC (tcga_lung_vital_status) | human | bulk RNA-seq | bulk | public (open GDC, no DUA) | none | vital_status (Dead/Alive at last GDC follow-up), real subject-level linkage | The first cohort here with genuine expression<->outcome linkage; NOT a survival/time-to-event label (no censoring/follow-up duration modeled); combines LUAD+LUSC on shared genes. |
| NLST | human | clinical/tabular | not applicable | controlled (DUA required) | CIGSMOK/CIGAR | candx | No expression data at all; controlled access; not accessed in this environment. |

## What "verified" vs "weak/proxy" means here

A **verified** label comes from a field the source explicitly records as
the exposure/outcome in question (e.g. NLST's `CIGSMOK`, GSE288003's
per-GSM `condition`). A **weak/proxy** label is inferred from a
correlated-but-different field (e.g. GSE136831's `Disease_Identity=COPD`
as a proxy for cigarette exposure). Weak/proxy labels are excluded from
primary supervised metrics by default; using them requires an explicit,
separately-reported weak-label experiment flag (see
`configs/cohorts.yaml`'s `task_support` and `notes` per cohort).

## Missingness

A subject with no recorded outcome is `*_label_known=False`, never
silently converted to a negative outcome — see `src/data/label_state.py`
and `src/data/converters.py`. The Phase 7 audit CLI
(`src/evidence/audit.py`) reports missingness explicitly as a structured
`not_evaluable` object when no local data is present to compute it from,
rather than reporting a zero or empty table.

## What is not committed to this repository

Raw or processed genomics data, model checkpoints, and any
participant-level NLST record — see `.gitignore` (`data/raw/`,
`data/processed/`, `checkpoints/`, `artifacts/benchmarks/`,
`artifacts/evidence/`).
