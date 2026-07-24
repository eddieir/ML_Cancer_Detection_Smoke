"""
tests/test_run_identity.py — Issue #16 blocker 1: real evidence identity
primitives (timestamp, split manifest, environment snapshot, bulk
model/preprocessing fingerprints).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.run_identity import (
    build_environment_snapshot,
    build_split_manifest,
    bulk_model_fingerprint,
    bulk_preprocessing_fingerprint,
    environment_snapshot_fingerprint,
    split_manifest_fingerprint,
    subject_set_fingerprint,
    utc_now_iso,
    validate_real_timestamp,
)


# ─── timestamps ─────────────────────────────────────────────────────────────

def test_utc_now_iso_ends_with_z():
    ts = utc_now_iso()
    assert ts.endswith("Z")
    validate_real_timestamp(ts)  # does not raise


def test_utc_now_iso_uses_injected_clock():
    fixed = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)
    ts = utc_now_iso(clock=lambda: fixed)
    assert ts.startswith("2026-03-05T12:00:00")


def test_utc_now_iso_rejects_naive_clock():
    with pytest.raises(ValueError):
        utc_now_iso(clock=lambda: datetime(2026, 1, 1))


@pytest.mark.parametrize("bad_ts", [
    "1970-01-01T00:00:00Z", "", "unknown", "not_applicable",
    "2026-01-01", "2026-01-01T00:00:00", "not-a-timestamp",
])
def test_validate_real_timestamp_rejects_placeholders(bad_ts):
    with pytest.raises(ValueError):
        validate_real_timestamp(bad_ts)


def test_validate_real_timestamp_accepts_plausible_recent_timestamp():
    validate_real_timestamp("2026-03-05T12:00:00Z")  # does not raise


# ─── split manifest ─────────────────────────────────────────────────────────

def _base_split_kwargs(**overrides):
    kwargs = dict(
        task="smoke_classification", endpoint="e", dataset_accession="GSE123352",
        train_subject_ids=["s1", "s2", "s3"], validation_subject_ids=[],
        development_holdout_subject_ids=["s4", "s5"], seed=42, train_frac=0.7,
        val_frac=0.0, test_frac=0.3, stratification_policy="p", class_mapping={"a": 0, "b": 1},
        rare_class_policy="p", label_state_policy="p", weak_label_policy="p",
        subject_identity_field="subject_id", grouping_policy="p",
        dataset_manifest_fingerprint="a" * 64,
    )
    kwargs.update(overrides)
    return kwargs


def test_split_manifest_fingerprint_changes_when_a_subject_moves_partition():
    m1 = build_split_manifest(**_base_split_kwargs())
    m2 = build_split_manifest(**_base_split_kwargs(
        train_subject_ids=["s1", "s2"], development_holdout_subject_ids=["s3", "s4", "s5"],
    ))
    assert split_manifest_fingerprint(m1) != split_manifest_fingerprint(m2)


def test_split_manifest_fingerprint_changes_with_seed():
    m1 = build_split_manifest(**_base_split_kwargs(seed=42))
    m2 = build_split_manifest(**_base_split_kwargs(seed=43))
    assert split_manifest_fingerprint(m1) != split_manifest_fingerprint(m2)


def test_split_manifest_fingerprint_changes_with_class_mapping():
    m1 = build_split_manifest(**_base_split_kwargs(class_mapping={"a": 0, "b": 1}))
    m2 = build_split_manifest(**_base_split_kwargs(class_mapping={"a": 1, "b": 0}))
    assert split_manifest_fingerprint(m1) != split_manifest_fingerprint(m2)


def test_split_manifest_fingerprint_changes_with_dataset_fingerprint():
    m1 = build_split_manifest(**_base_split_kwargs(dataset_manifest_fingerprint="a" * 64))
    m2 = build_split_manifest(**_base_split_kwargs(dataset_manifest_fingerprint="b" * 64))
    assert split_manifest_fingerprint(m1) != split_manifest_fingerprint(m2)


def test_split_manifest_fingerprint_stable_for_identical_inputs():
    m1 = build_split_manifest(**_base_split_kwargs())
    m2 = build_split_manifest(**_base_split_kwargs())
    assert split_manifest_fingerprint(m1) == split_manifest_fingerprint(m2)


def test_sanitize_split_manifest_drops_raw_subject_ids():
    from evidence.run_identity import sanitize_split_manifest
    m = build_split_manifest(**_base_split_kwargs())
    sanitized = sanitize_split_manifest(m)
    assert "train_subject_ids" not in sanitized
    assert sanitized["train_subject_ids_count"] == 3
    assert sanitized["development_holdout_subject_ids_count"] == 2
    assert sanitized["fingerprint"] == split_manifest_fingerprint(m)


# ─── environment snapshot ───────────────────────────────────────────────────

def test_environment_snapshot_contains_no_home_directory():
    import os
    snapshot = build_environment_snapshot("0" * 40)
    blob = str(snapshot)
    assert os.path.expanduser("~") not in blob


def test_environment_snapshot_fingerprint_changes_with_commit_sha():
    s1 = build_environment_snapshot("a" * 40)
    s2 = build_environment_snapshot("b" * 40)
    assert environment_snapshot_fingerprint(s1) != environment_snapshot_fingerprint(s2)


def test_environment_snapshot_fingerprint_deterministic_for_same_commit():
    s1 = build_environment_snapshot("a" * 40)
    s2 = build_environment_snapshot("a" * 40)
    assert environment_snapshot_fingerprint(s1) == environment_snapshot_fingerprint(s2)


# ─── bulk model / preprocessing fingerprints ───────────────────────────────

def _fit_logreg(seed=0, n=40, n_genes=10, c=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, n_genes))
    y = (rng.random(n) > 0.5).astype(int)
    if len(set(y.tolist())) < 2:
        y[0] = 1 - y[0]
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=c, max_iter=200, random_state=seed).fit(scaler.transform(X), y)
    return model, scaler


def test_bulk_preprocessing_fingerprint_changes_with_selected_genes():
    model, scaler = _fit_logreg()
    fp1 = bulk_preprocessing_fingerprint(
        gene_indices=np.array([0, 1, 2]), gene_names_selected=["g0", "g1", "g2"],
        n_genes_available=10, n_top_variance_genes_requested=3, selection_policy="p",
        scaler=scaler, train_subject_ids=["s1", "s2"], missing_value_handling="none",
    )
    fp2 = bulk_preprocessing_fingerprint(
        gene_indices=np.array([0, 1, 3]), gene_names_selected=["g0", "g1", "g3"],
        n_genes_available=10, n_top_variance_genes_requested=3, selection_policy="p",
        scaler=scaler, train_subject_ids=["s1", "s2"], missing_value_handling="none",
    )
    assert fp1 != fp2


def test_bulk_model_fingerprint_changes_with_coefficients():
    model_a, scaler = _fit_logreg(seed=1)
    model_b, _ = _fit_logreg(seed=2)
    preprocessing_fp = "a" * 64
    fp_a = bulk_model_fingerprint(
        model=model_a, hyperparameters={"C": 1.0}, class_names=["x", "y"],
        preprocessing_fingerprint=preprocessing_fp, train_subject_ids=["s1", "s2"],
    )
    fp_b = bulk_model_fingerprint(
        model=model_b, hyperparameters={"C": 1.0}, class_names=["x", "y"],
        preprocessing_fingerprint=preprocessing_fp, train_subject_ids=["s1", "s2"],
    )
    assert fp_a != fp_b


def test_bulk_model_fingerprint_changes_with_hyperparameters():
    model, scaler = _fit_logreg(c=1.0)
    preprocessing_fp = "a" * 64
    fp1 = bulk_model_fingerprint(
        model=model, hyperparameters={"C": 1.0}, class_names=["x", "y"],
        preprocessing_fingerprint=preprocessing_fp, train_subject_ids=["s1"],
    )
    fp2 = bulk_model_fingerprint(
        model=model, hyperparameters={"C": 10.0}, class_names=["x", "y"],
        preprocessing_fingerprint=preprocessing_fp, train_subject_ids=["s1"],
    )
    assert fp1 != fp2


def test_bulk_model_fingerprint_is_not_a_prediction_fingerprint():
    # two identically-fit models on the same data produce the same model
    # fingerprint regardless of what they're later asked to predict on —
    # this fingerprint is fitted-state-only, never derived from predictions.
    model_a, scaler = _fit_logreg(seed=5)
    model_b, _ = _fit_logreg(seed=5)
    preprocessing_fp = "a" * 64
    fp_a = bulk_model_fingerprint(
        model=model_a, hyperparameters={"C": 1.0}, class_names=["x", "y"],
        preprocessing_fingerprint=preprocessing_fp, train_subject_ids=["s1"],
    )
    fp_b = bulk_model_fingerprint(
        model=model_b, hyperparameters={"C": 1.0}, class_names=["x", "y"],
        preprocessing_fingerprint=preprocessing_fp, train_subject_ids=["s1"],
    )
    assert fp_a == fp_b


def test_bulk_model_fingerprint_reproducible_for_identical_fitted_state():
    model, scaler = _fit_logreg(seed=7)
    preprocessing_fp = "c" * 64
    fp1 = bulk_model_fingerprint(
        model=model, hyperparameters={"C": 1.0}, class_names=["x", "y"],
        preprocessing_fingerprint=preprocessing_fp, train_subject_ids=["s1", "s2"],
    )
    fp2 = bulk_model_fingerprint(
        model=model, hyperparameters={"C": 1.0}, class_names=["x", "y"],
        preprocessing_fingerprint=preprocessing_fp, train_subject_ids=["s1", "s2"],
    )
    assert fp1 == fp2


def test_subject_set_fingerprint_is_order_independent():
    assert subject_set_fingerprint(["a", "b", "c"]) == subject_set_fingerprint(["c", "a", "b"])


def test_subject_set_fingerprint_changes_with_membership():
    assert subject_set_fingerprint(["a", "b"]) != subject_set_fingerprint(["a", "c"])
