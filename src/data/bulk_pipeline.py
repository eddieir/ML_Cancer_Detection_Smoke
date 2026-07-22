"""
data/bulk_pipeline.py — minimal, honest bulk-microarray smoke-classification
path.

Every other training path in this repository (train.py's CellLevelDataset /
SubjectLevelDataset / Trainer, data/preprocessing.py's PreprocessingArtifact)
is single-cell only by design: data/assay_policy.py exists specifically to
keep pseudo-bulk rows (is_pseudo_bulk=True — GSE123352, GSE994, GSE307690,
TCGA) OUT of that pipeline, because a bulk/pseudo-bulk sample is not a cell
and mixing the two would silently corrupt the MIL bag structure. This module
does not touch any of that machinery. It is a separate, independent, much
simpler path for cohorts that carry a verified per-sample label and a real
bulk expression matrix but no single-cell compatibility — currently only
GSE123352 (human bulk microarray, Illumina HumanHT-12 V4.0, verified
`ever_never_smoker` GEO characteristic).

Pipeline
--------
1. `load_bulk_expression_and_labels()` reads the already-converted
   genes-x-samples CSV + `_samples_meta.csv` sidecar that
   `data/converters.py::convert_microarray` produces (real probe->gene-symbol
   mapping via the platform annotation file, real per-sample smoking-status
   parsed from the GEO series-matrix characteristics — see that module's
   docstrings). Samples whose smoke_type_known is False are excluded, never
   guessed.
2. `build_bulk_smoke_dataset()` assembles a samples x genes matrix `X`,
   integer labels `y`, and subject ids (one sample = one subject for this
   cohort) restricted to the two verified classes this module supports.
3. `split_bulk_subjects()` reuses `data.splitting.subject_train_val_test_split`
   for a deterministic, leakage-free subject-level train/test split (every
   sample here is already one subject, so this is one row per group, but
   still goes through the same infrastructure the rest of the repository
   uses so the split is auditable/reproducible the same way).
4. `fit_bulk_logistic_regression()` / `evaluate_bulk_logistic_regression()`
   — a deliberately simple classifier: train-only top-variance gene
   selection, train-only standardization, L2-regularized logistic
   regression with balanced class weights. This is meant to be a first
   honest bulk pipeline, not a publication-grade one.
5. `run_bulk_smoke_classification()` wires the above into one function that
   returns real y_true/y_pred/y_prob/subject_ids for both splits plus the
   fitted preprocessing choices, for a caller (e.g.
   evidence.tracks.run_track_a_on_real_gse123352_data) to turn into metrics.

Nothing in this module fabricates a label. If the phenotype field cannot
actually be parsed from the real downloaded file, `load_bulk_expression_and_labels`
raises `BulkPipelineError` rather than defaulting every sample to some
label — the caller is expected to turn that into a `not_evaluable` result.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SUPPORTED_BULK_CLASSES = ("unexposed", "cigarette")  # index 0 / 1 — binary ever/never
DEFAULT_TOP_VARIANCE_GENES = 2000


class BulkPipelineError(ValueError):
    """Raised when a bulk cohort's real expression matrix or verified label
    field cannot be honestly parsed/assembled — never papered over with a
    guessed or defaulted label."""


@dataclass
class BulkExpressionDataset:
    """Real samples x genes matrix plus verified binary smoke labels, ready
    for a subject-level split. `excluded_sample_ids` records any sample
    dropped for not carrying a verified label or not belonging to one of
    SUPPORTED_BULK_CLASSES — dropped, never coerced."""

    X: np.ndarray                      # [n_samples, n_genes], float32
    y: np.ndarray                      # [n_samples], int (0=unexposed, 1=cigarette)
    subject_ids: List[str]
    sample_ids: List[str]
    gene_names: List[str]
    class_names: Tuple[str, str] = SUPPORTED_BULK_CLASSES
    excluded_sample_ids: List[str] = field(default_factory=list)
    excluded_reason_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.X.shape[0] != len(self.y):
            raise BulkPipelineError(
                f"X has {self.X.shape[0]} rows but y has {len(self.y)} entries"
            )
        if self.X.shape[0] != len(self.subject_ids):
            raise BulkPipelineError(
                f"X has {self.X.shape[0]} rows but {len(self.subject_ids)} subject_ids"
            )


def load_bulk_expression_and_labels(
    csv_path: "str | Path", meta_path: Optional["str | Path"] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the genes-x-samples CSV (`converters.convert_microarray`'s
    output) and its `_samples_meta.csv` sidecar. Raises BulkPipelineError —
    never guesses a label — if either file is missing, if the meta file
    carries no `smoke_type`/`smoke_type_known` columns, or if not a single
    sample has a verified label."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise BulkPipelineError(
            f"{csv_path} does not exist — run "
            f"`python3 src/data/converters.py --accession <ACCESSION>` first."
        )
    expr = pd.read_csv(csv_path, index_col=0)  # genes x samples

    if meta_path is None:
        meta_path = csv_path.with_name(csv_path.stem + "_samples_meta.csv")
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise BulkPipelineError(
            f"{meta_path} does not exist — the converted expression matrix has no "
            "per-sample phenotype sidecar to parse a verified smoke label from."
        )
    meta = pd.read_csv(meta_path)
    required_cols = {"sample_id", "smoke_type", "smoke_type_known"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise BulkPipelineError(
            f"{meta_path} is missing required column(s) {sorted(missing)} — cannot "
            "honestly determine which samples carry a verified smoke label."
        )
    meta = meta.set_index("sample_id")
    n_known = int(meta["smoke_type_known"].astype(bool).sum())
    if n_known == 0:
        raise BulkPipelineError(
            f"{meta_path} carries zero samples with smoke_type_known=True — no "
            "verified smoke label could be parsed from the real downloaded phenotype "
            "field for any sample in this cohort."
        )
    return expr, meta


def build_bulk_smoke_dataset(
    csv_path: "str | Path", meta_path: Optional["str | Path"] = None,
) -> BulkExpressionDataset:
    """Assembles a BulkExpressionDataset restricted to verified
    unexposed/cigarette samples. Any sample whose smoke_type_known is False,
    or whose smoke_type is not one of SUPPORTED_BULK_CLASSES (e.g. a cohort
    that also carries vape/cannabis/dual_use samples), is excluded and
    recorded in excluded_sample_ids/excluded_reason_counts — never coerced
    into one of the two supported classes."""
    expr, meta = load_bulk_expression_and_labels(csv_path, meta_path)

    sample_ids_ordered = [str(c) for c in expr.columns]
    excluded: List[str] = []
    reasons: Dict[str, int] = {}
    kept_ids: List[str] = []
    kept_labels: List[int] = []

    for sid in sample_ids_ordered:
        if sid not in meta.index:
            excluded.append(sid)
            reasons["no_phenotype_row"] = reasons.get("no_phenotype_row", 0) + 1
            continue
        row = meta.loc[sid]
        if not bool(row["smoke_type_known"]):
            excluded.append(sid)
            reasons["smoke_type_unknown"] = reasons.get("smoke_type_unknown", 0) + 1
            continue
        smoke_type = str(row["smoke_type"]).strip().lower()
        if smoke_type not in SUPPORTED_BULK_CLASSES:
            excluded.append(sid)
            reasons["unsupported_class"] = reasons.get("unsupported_class", 0) + 1
            continue
        kept_ids.append(sid)
        kept_labels.append(SUPPORTED_BULK_CLASSES.index(smoke_type))

    if not kept_ids:
        raise BulkPipelineError(
            "No sample survived verified-label filtering — every sample was excluded "
            f"({reasons})."
        )

    X = expr[kept_ids].T.values.astype(np.float32)  # samples x genes
    y = np.asarray(kept_labels, dtype=int)

    return BulkExpressionDataset(
        X=X, y=y, subject_ids=list(kept_ids), sample_ids=list(kept_ids),
        gene_names=list(expr.index.astype(str)),
        excluded_sample_ids=excluded, excluded_reason_counts=reasons,
    )


def split_bulk_subjects(
    dataset: BulkExpressionDataset, seed: int = 42,
    train_frac: float = 0.70, val_frac: float = 0.0, test_frac: float = 0.30,
):
    """Deterministic, leakage-free subject-level split via the same
    `data.splitting.subject_train_val_test_split` infrastructure the rest
    of this repository uses. Each sample in this cohort is already exactly
    one subject, so this mainly buys reproducibility/auditability rather
    than group-leakage protection per se — but it is the real split
    machinery, not a hand-rolled shuffle."""
    from data.splitting import subject_train_val_test_split

    return subject_train_val_test_split(
        subject_ids=dataset.subject_ids,
        labels=list(dataset.y),
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        seed=seed,
    )


def _select_top_variance_genes(X_train: np.ndarray, n_genes: int) -> np.ndarray:
    """Train-only gene selection by variance — fit exclusively on the
    training rows passed in, never on validation/test rows, so no split
    information leaks into which genes the classifier gets to see."""
    n_genes = min(n_genes, X_train.shape[1])
    variances = X_train.var(axis=0)
    return np.argsort(variances)[::-1][:n_genes]


@dataclass
class FittedBulkClassifier:
    model: "object"
    scaler: "object"
    gene_indices: np.ndarray
    gene_names_selected: List[str]
    n_genes_available: int
    n_top_variance_genes: int


def fit_bulk_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, gene_names: Sequence[str],
    seed: int = 42, n_top_variance_genes: int = DEFAULT_TOP_VARIANCE_GENES,
    C: float = 1.0,
) -> FittedBulkClassifier:
    """Deliberately simple: train-only top-variance gene selection,
    train-only standardization, L2 logistic regression with balanced class
    weights. This is a first honest bulk baseline, not meant to be tuned."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(set(y_train.tolist())) < 2:
        raise BulkPipelineError(
            "Training split contains only one class — cannot fit a binary classifier "
            "without both classes represented in the training partition."
        )

    gene_idx = _select_top_variance_genes(X_train, n_top_variance_genes)
    X_sel = X_train[:, gene_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    model = LogisticRegression(
        C=C, max_iter=2000, class_weight="balanced", random_state=seed,
    )
    model.fit(X_scaled, y_train)

    return FittedBulkClassifier(
        model=model, scaler=scaler, gene_indices=gene_idx,
        gene_names_selected=[gene_names[i] for i in gene_idx],
        n_genes_available=X_train.shape[1],
        n_top_variance_genes=len(gene_idx),
    )


def apply_bulk_classifier(fitted: FittedBulkClassifier, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform-then-predict only — no refitting. Returns (y_pred, y_prob_class1)."""
    X_sel = X[:, fitted.gene_indices]
    X_scaled = fitted.scaler.transform(X_sel)
    y_pred = fitted.model.predict(X_scaled)
    proba = fitted.model.predict_proba(X_scaled)
    classes = list(fitted.model.classes_)
    col1 = classes.index(1) if 1 in classes else (1 if proba.shape[1] > 1 else 0)
    y_prob = proba[:, col1]
    return y_pred, y_prob


def run_bulk_smoke_classification(
    csv_path: "str | Path", meta_path: Optional["str | Path"] = None,
    seed: int = 42, train_frac: float = 0.70, test_frac: float = 0.30,
    n_top_variance_genes: int = DEFAULT_TOP_VARIANCE_GENES, C: float = 1.0,
) -> Dict:
    """End-to-end: parse -> assemble -> split -> fit -> evaluate. Returns a
    plain dict (not an EvidenceReport — evidence.tracks wraps this) with
    real y_true/y_pred/y_prob/subject_ids for the train and test splits, the
    fitted classifier's gene selection, and dataset-level counts including
    excluded samples — nothing here is synthetic or guessed."""
    dataset = build_bulk_smoke_dataset(csv_path, meta_path)
    split = split_bulk_subjects(
        dataset, seed=seed, train_frac=train_frac, val_frac=0.0, test_frac=test_frac,
    )

    id_to_row = {sid: i for i, sid in enumerate(dataset.subject_ids)}
    train_idx = [id_to_row[s] for s in split.train_subjects if s in id_to_row]
    test_idx = [id_to_row[s] for s in split.test_subjects if s in id_to_row]

    X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]

    fitted = fit_bulk_logistic_regression(
        X_train, y_train, dataset.gene_names, seed=seed,
        n_top_variance_genes=n_top_variance_genes, C=C,
    )
    y_pred_train, y_prob_train = apply_bulk_classifier(fitted, X_train)
    y_pred_test, y_prob_test = apply_bulk_classifier(fitted, X_test)

    return {
        "dataset": dataset,
        "split": split,
        "train": {
            "subject_ids": [dataset.subject_ids[i] for i in train_idx],
            "y_true": y_train.tolist(), "y_pred": y_pred_train.tolist(),
            "y_prob": y_prob_train.tolist(),
        },
        "test": {
            "subject_ids": [dataset.subject_ids[i] for i in test_idx],
            "y_true": y_test.tolist(), "y_pred": y_pred_test.tolist(),
            "y_prob": y_prob_test.tolist(),
        },
        "fitted_classifier": fitted,
        "seed": seed,
        "n_top_variance_genes": n_top_variance_genes,
        "excluded_sample_ids": dataset.excluded_sample_ids,
        "excluded_reason_counts": dataset.excluded_reason_counts,
    }
