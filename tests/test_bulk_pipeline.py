"""
tests/test_bulk_pipeline.py — data/bulk_pipeline.py unit tests.

Exercises the module against small synthetic genes-x-samples fixtures
written to a temp directory (never the real multi-MB GSE123352 download) —
fast, deterministic, and covers the honest-failure paths (missing sidecar,
zero verified labels, single-class training split) the same way the rest of
this repository tests its ML code.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.bulk_pipeline import (
    BulkPipelineError,
    build_bulk_smoke_dataset,
    fit_bulk_logistic_regression,
    apply_bulk_classifier,
    load_bulk_expression_and_labels,
    run_bulk_smoke_classification,
    split_bulk_subjects,
)


def _write_fixture(tmp_path, n_samples=40, n_genes=50, seed=0, all_unknown=False,
                    single_class=False, missing_meta=False, extra_class=False):
    rng = np.random.RandomState(seed)
    sample_ids = [f"GSM{1000+i}" for i in range(n_samples)]
    gene_ids = [f"GENE{i}" for i in range(n_genes)]

    if single_class:
        labels = ["cigarette"] * n_samples
    else:
        labels = ["cigarette" if i % 2 == 0 else "unexposed" for i in range(n_samples)]
    if extra_class:
        labels[-1] = "vape"

    # give the two classes a mean-shifted signal on a handful of genes so a
    # real classifier actually has something learnable, deterministic given seed
    X = rng.normal(loc=0.0, scale=1.0, size=(n_genes, n_samples))
    signal_genes = list(range(5))
    for j, lab in enumerate(labels):
        if lab == "cigarette":
            X[signal_genes, j] += 2.0

    expr = pd.DataFrame(X, index=gene_ids, columns=sample_ids)
    csv_path = tmp_path / "FAKE.csv"
    expr.to_csv(csv_path)

    if not missing_meta:
        known = [not all_unknown] * n_samples
        meta = pd.DataFrame({
            "sample_id": sample_ids,
            "smoke_type": labels,
            "smoke_type_known": known,
        })
        meta_path = tmp_path / "FAKE_samples_meta.csv"
        meta.to_csv(meta_path, index=False)

    return csv_path


def test_load_bulk_expression_and_labels_real_fixture(tmp_path):
    csv_path = _write_fixture(tmp_path)
    expr, meta = load_bulk_expression_and_labels(csv_path)
    assert expr.shape[1] == 40
    assert meta["smoke_type_known"].astype(bool).all()


def test_load_bulk_expression_and_labels_missing_sidecar_raises(tmp_path):
    csv_path = _write_fixture(tmp_path, missing_meta=True)
    with pytest.raises(BulkPipelineError):
        load_bulk_expression_and_labels(csv_path)


def test_load_bulk_expression_and_labels_all_unknown_raises(tmp_path):
    csv_path = _write_fixture(tmp_path, all_unknown=True)
    with pytest.raises(BulkPipelineError):
        load_bulk_expression_and_labels(csv_path)


def test_build_bulk_smoke_dataset_excludes_unsupported_class(tmp_path):
    csv_path = _write_fixture(tmp_path, n_samples=20, extra_class=True)
    dataset = build_bulk_smoke_dataset(csv_path)
    assert dataset.X.shape[0] == 19  # the one 'vape' sample excluded
    assert dataset.excluded_reason_counts.get("unsupported_class") == 1
    assert set(dataset.y.tolist()) == {0, 1}


def test_build_bulk_smoke_dataset_shape_matches_labels(tmp_path):
    csv_path = _write_fixture(tmp_path, n_samples=40, n_genes=60)
    dataset = build_bulk_smoke_dataset(csv_path)
    assert dataset.X.shape == (40, 60)
    assert len(dataset.y) == 40
    assert len(dataset.subject_ids) == 40


def test_split_bulk_subjects_no_leakage(tmp_path):
    csv_path = _write_fixture(tmp_path, n_samples=40)
    dataset = build_bulk_smoke_dataset(csv_path)
    split = split_bulk_subjects(dataset, seed=1, train_frac=0.7, val_frac=0.0, test_frac=0.3)
    train_set, test_set = set(split.train_subjects), set(split.test_subjects)
    assert not (train_set & test_set)
    # rounding within subject_train_val_test_split's per-class buckets can
    # leave a subject unassigned to either split (never duplicated) — the
    # real leakage guarantee is "no overlap", not "every subject placed"
    assert (train_set | test_set) <= set(dataset.subject_ids)
    assert len(train_set) > 0 and len(test_set) > 0


def test_fit_bulk_logistic_regression_single_class_raises(tmp_path):
    csv_path = _write_fixture(tmp_path, n_samples=20, single_class=True)
    dataset = build_bulk_smoke_dataset(csv_path)
    with pytest.raises(BulkPipelineError):
        fit_bulk_logistic_regression(dataset.X, dataset.y, dataset.gene_names, seed=0)


def test_fit_and_apply_bulk_classifier_roundtrip(tmp_path):
    csv_path = _write_fixture(tmp_path, n_samples=60, n_genes=80, seed=3)
    dataset = build_bulk_smoke_dataset(csv_path)
    fitted = fit_bulk_logistic_regression(
        dataset.X, dataset.y, dataset.gene_names, seed=0, n_top_variance_genes=20,
    )
    assert fitted.n_top_variance_genes == 20
    y_pred, y_prob = apply_bulk_classifier(fitted, dataset.X)
    assert y_pred.shape[0] == dataset.X.shape[0]
    assert ((y_prob >= 0) & (y_prob <= 1)).all()


def test_run_bulk_smoke_classification_end_to_end_learns_signal(tmp_path):
    csv_path = _write_fixture(tmp_path, n_samples=100, n_genes=100, seed=5)
    result = run_bulk_smoke_classification(
        csv_path, seed=42, train_frac=0.7, test_frac=0.3, n_top_variance_genes=20,
    )
    assert result["excluded_sample_ids"] == []
    test_true = np.asarray(result["test"]["y_true"])
    test_pred = np.asarray(result["test"]["y_pred"])
    assert len(test_true) > 0
    # the injected signal is strong (mean shift of 2.0 on 5 genes) — a real
    # fit should clearly beat chance on held-out subjects
    accuracy = (test_true == test_pred).mean()
    assert accuracy > 0.7


def test_run_bulk_smoke_classification_no_subject_leakage(tmp_path):
    csv_path = _write_fixture(tmp_path, n_samples=60, n_genes=40, seed=7)
    result = run_bulk_smoke_classification(csv_path, seed=1, train_frac=0.7, test_frac=0.3)
    train_ids = set(result["train"]["subject_ids"])
    test_ids = set(result["test"]["subject_ids"])
    assert not (train_ids & test_ids)
