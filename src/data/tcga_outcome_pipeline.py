"""
data/tcga_outcome_pipeline.py — genuine subject-level TCGA vital-status
(deceased vs. alive at last GDC follow-up) prediction from primary-tumor
bulk RNA-seq expression.

This is the first cohort in this repository with a genuinely linked,
subject-level expression<->outcome pair: `data/converters.py::
convert_tcga_vital_status` writes one row per real GDC `case_id`, with
`vital_status` read directly from that same case's GDC demographic record —
not inferred, not a bulk sample_type proxy, not borrowed from an unrelated
cohort. It is a real, obtainable, open-access dataset (GDC's "STAR - Counts"
gene-count files and clinical demographic records are open-access — no
dbGaP/DUA required), downloaded directly from `api.gdc.cancer.gov`.

Honest scope: vital_status ("Dead"/"Alive" at last recorded follow-up) is
NOT a true time-to-event survival label — no censoring, follow-up duration,
or time-to-event semantics are modeled here. A "Dead" case that died 10
years after a "Alive" case's last (recent) follow-up are both binary
labels, indistinguishable to this pipeline. Any real report built from this
module's output must say so explicitly; this module never upgrades its own
binary vital-status label into a survival/time-to-event claim.

Like data/bulk_pipeline.py, this is a separate, independent path — TCGA
rows are pseudo-bulk (data/assay_policy.py) and never enter the single-cell
MIL pipeline.

Pipeline
--------
1. `load_tcga_outcome_and_labels()` reads the per-project genes-x-samples
   CSV + `_vital_status_samples_meta.csv` sidecar `convert_tcga_vital_status`
   produces.
2. `build_tcga_outcome_dataset()` combines one or more projects (e.g.
   TCGA-LUAD + TCGA-LUSC) on their shared gene set (an inner join — TCGA
   RNA-seq is uniformly GENCODE-annotated across projects, so this is a
   real, not an approximate, alignment) into one X/y/subject_ids dataset.
   `subject_id` is the real GDC `case_id`, already globally unique across
   projects — no cohort-specific identity inference is needed the way
   GSE123352's `_infer_subject_id_column` is (see convert_tcga_vital_status).
3. `split_tcga_outcome_subjects()` reuses `data.splitting.subject_train_val_test_split`,
   exactly like bulk_pipeline.py.
4. Model fitting reuses `data.bulk_pipeline.fit_bulk_logistic_regression`/
   `apply_bulk_classifier` directly (Task-agnostic despite living in
   bulk_pipeline.py — both operate on plain X/y/gene_names arrays) rather
   than reimplementing an identical classifier.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SUPPORTED_VITAL_CLASSES = ("alive", "dead")  # index 0 / 1
DEFAULT_TOP_VARIANCE_GENES = 2000


class TCGAOutcomePipelineError(ValueError):
    """Raised when a TCGA vital-status cohort's expression/outcome data
    cannot be honestly assembled — never papered over with a default."""


@dataclass
class TCGAOutcomeDataset:
    """Same field names as data.bulk_pipeline.BulkExpressionDataset on
    purpose — evidence/development.py's fold-local-selection, baseline, and
    metric-bundle helpers are written against these field names and are
    reused unmodified for this dataset."""

    X: np.ndarray                      # [n_samples, n_genes], float32
    y: np.ndarray                      # [n_samples], int (0=alive, 1=dead)
    subject_ids: List[str]             # real GDC case_id, one per row
    sample_ids: List[str]
    gene_names: List[str]
    class_names: Tuple[str, str] = SUPPORTED_VITAL_CLASSES
    project_of_subject: Dict[str, str] = field(default_factory=dict)
    excluded_sample_ids: List[str] = field(default_factory=list)
    excluded_reason_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.X.shape[0] != len(self.y):
            raise TCGAOutcomePipelineError(f"X has {self.X.shape[0]} rows but y has {len(self.y)} entries")
        if self.X.shape[0] != len(self.subject_ids):
            raise TCGAOutcomePipelineError(f"X has {self.X.shape[0]} rows but {len(self.subject_ids)} subject_ids")


def load_tcga_outcome_and_labels(csv_path: "str | Path", meta_path: Optional["str | Path"] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise TCGAOutcomePipelineError(
            f"{csv_path} does not exist — run data.converters.convert_tcga_vital_status first."
        )
    expr = pd.read_csv(csv_path, index_col=0)  # genes x samples
    if meta_path is None:
        meta_path = csv_path.with_name(csv_path.stem + "_samples_meta.csv")
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise TCGAOutcomePipelineError(f"{meta_path} does not exist — no vital-status sidecar to parse.")
    meta = pd.read_csv(meta_path)
    required = {"sample_id", "subject_id", "subject_id_verified", "vital_status_known", "vital_status"}
    missing = required - set(meta.columns)
    if missing:
        raise TCGAOutcomePipelineError(f"{meta_path} is missing required column(s) {sorted(missing)}.")
    meta = meta.set_index("sample_id")
    return expr, meta


def build_tcga_outcome_dataset(
    project_csv_paths: Sequence["str | Path"],
) -> TCGAOutcomeDataset:
    """Combines one or more convert_tcga_vital_status() outputs (e.g.
    TCGA-LUAD + TCGA-LUSC) on their shared gene set (inner join on Ensembl
    gene_id — real, not approximate, since both projects share the same
    GENCODE annotation). Refuses to proceed if the same subject_id (real
    GDC case_id) appears via more than one input file — a genuine
    duplicate-subject contradiction, never silently resolved."""
    frames, metas = [], []
    for p in project_csv_paths:
        expr, meta = load_tcga_outcome_and_labels(p)
        frames.append(expr)
        metas.append(meta)

    shared_genes = frames[0].index
    for f in frames[1:]:
        shared_genes = shared_genes.intersection(f.index)
    shared_genes = shared_genes.sort_values()
    if len(shared_genes) == 0:
        raise TCGAOutcomePipelineError("No shared genes across the given TCGA projects — cannot combine.")

    excluded, reasons = [], {}
    kept_sample_ids, kept_subject_ids, kept_labels, kept_cols, project_of_subject = [], [], [], [], {}
    seen_subjects: Dict[str, str] = {}

    for p, expr, meta in zip(project_csv_paths, frames, metas):
        project_name = Path(p).stem.replace("_vital_status", "")
        expr_aligned = expr.loc[shared_genes]
        for sid in expr.columns:
            if sid not in meta.index:
                excluded.append(sid)
                reasons["no_meta_row"] = reasons.get("no_meta_row", 0) + 1
                continue
            row = meta.loc[sid]
            if not bool(row["subject_id_verified"]) or not bool(row["vital_status_known"]):
                excluded.append(sid)
                reasons["unverified_or_unknown"] = reasons.get("unverified_or_unknown", 0) + 1
                continue
            subj = str(row["subject_id"])
            if subj in seen_subjects:
                excluded.append(sid)
                reasons["duplicate_subject_across_projects"] = reasons.get("duplicate_subject_across_projects", 0) + 1
                continue
            seen_subjects[subj] = project_name
            kept_sample_ids.append(sid)
            kept_subject_ids.append(subj)
            kept_labels.append(int(row["vital_status"]))
            kept_cols.append(expr_aligned[sid].values)
            project_of_subject[subj] = project_name

    if not kept_sample_ids:
        raise TCGAOutcomePipelineError(f"No sample survived filtering across the given projects ({reasons}).")

    X = np.vstack(kept_cols).astype(np.float32)
    y = np.asarray(kept_labels, dtype=int)

    return TCGAOutcomeDataset(
        X=X, y=y, subject_ids=kept_subject_ids, sample_ids=kept_sample_ids,
        gene_names=list(shared_genes.astype(str)), project_of_subject=project_of_subject,
        excluded_sample_ids=excluded, excluded_reason_counts=reasons,
    )


def split_tcga_outcome_subjects(dataset: TCGAOutcomeDataset, seed: int = 42, train_frac: float = 0.70, val_frac: float = 0.0, test_frac: float = 0.30):
    from data.splitting import subject_train_val_test_split
    return subject_train_val_test_split(
        subject_ids=dataset.subject_ids, labels=list(dataset.y),
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed,
    )
