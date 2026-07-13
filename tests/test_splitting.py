"""Tests for data/splitting.py — subject-level splitting and grouped K-fold."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.splitting import (
    SplitManifest,
    build_split_report,
    grouped_kfold,
    load_or_create_split,
    subject_train_val_test_split,
)


def _synthetic_cells(n_subjects=30, cells_per_subject=20, n_classes=6, seed=0):
    rng = np.random.RandomState(seed)
    subject_ids, labels = [], []
    class_of_subject = {f"sub_{i}": i % n_classes for i in range(n_subjects)}
    for i in range(n_subjects):
        sid = f"sub_{i}"
        n_cells = cells_per_subject + rng.randint(-5, 5)
        subject_ids += [sid] * n_cells
        labels += [class_of_subject[sid]] * n_cells
    return subject_ids, labels


# ─── No leakage ───────────────────────────────────────────────────────────

def test_no_subject_appears_in_multiple_splits():
    subject_ids, labels = _synthetic_cells()
    manifest = subject_train_val_test_split(subject_ids, labels, seed=1)
    tr, va, te = set(manifest.train_subjects), set(manifest.val_subjects), set(manifest.test_subjects)
    assert not (tr & va)
    assert not (tr & te)
    assert not (va & te)


def test_every_subject_assigned_exactly_once():
    subject_ids, labels = _synthetic_cells()
    manifest = subject_train_val_test_split(subject_ids, labels, seed=1)
    all_assigned = manifest.train_subjects + manifest.val_subjects + manifest.test_subjects
    assert sorted(all_assigned) == sorted(set(str(s) for s in subject_ids))
    assert len(all_assigned) == len(set(all_assigned))


def test_manifest_constructor_rejects_overlap():
    with pytest.raises(ValueError):
        SplitManifest(seed=0, train_subjects=["a", "b"], val_subjects=["b", "c"], test_subjects=["d"])


# ─── Reproducibility ────────────────────────────────────────────────────────

def test_same_seed_gives_identical_split():
    subject_ids, labels = _synthetic_cells()
    m1 = subject_train_val_test_split(subject_ids, labels, seed=7)
    m2 = subject_train_val_test_split(subject_ids, labels, seed=7)
    assert m1.train_subjects == m2.train_subjects
    assert m1.val_subjects   == m2.val_subjects
    assert m1.test_subjects  == m2.test_subjects


def test_different_seed_gives_different_split():
    subject_ids, labels = _synthetic_cells(n_subjects=40)
    m1 = subject_train_val_test_split(subject_ids, labels, seed=1)
    m2 = subject_train_val_test_split(subject_ids, labels, seed=2)
    assert m1.train_subjects != m2.train_subjects


# ─── Correct grouping ───────────────────────────────────────────────────────

def test_split_uses_subject_as_grouping_variable():
    """A cell-level random split would put different cells of the same subject in
    different splits; this must never happen regardless of class imbalance."""
    subject_ids, labels = _synthetic_cells(n_subjects=50, cells_per_subject=100)
    manifest = subject_train_val_test_split(subject_ids, labels, seed=3)
    cell_split = {}
    for sid in subject_ids:
        s = manifest.split_of(sid)
        cell_split.setdefault(str(sid), set()).add(s)
    for sid, splitset in cell_split.items():
        assert len(splitset) == 1, f"subject {sid} split across {splitset}"


def test_approximate_split_proportions_on_large_balanced_data():
    subject_ids, labels = _synthetic_cells(n_subjects=300, cells_per_subject=10, n_classes=3)
    manifest = subject_train_val_test_split(subject_ids, labels, seed=5)
    n_total = len(manifest.train_subjects) + len(manifest.val_subjects) + len(manifest.test_subjects)
    train_frac = len(manifest.train_subjects) / n_total
    assert 0.60 < train_frac < 0.80


# ─── Rare class handling ─────────────────────────────────────────────────────

def test_rare_class_with_one_subject_does_not_crash_or_leak():
    subject_ids = [f"s{i}" for i in range(20) for _ in range(5)]
    labels = [0] * 95 + [1] * 5  # only one subject worth of a rare label in expectation
    # Force exactly one subject to have the rare label
    subject_ids = []
    labels = []
    for i in range(19):
        subject_ids += [f"common_{i}"] * 5
        labels += [0] * 5
    subject_ids += ["rare_0"] * 5
    labels += [1] * 5

    manifest = subject_train_val_test_split(subject_ids, labels, seed=9)
    assert "rare_0" in (manifest.train_subjects + manifest.val_subjects + manifest.test_subjects)
    # rare_0 must land in exactly one split
    assert sum(["rare_0" in getattr(manifest, f"{s}_subjects") for s in ("train", "val", "test")]) == 1
    assert "1" in manifest.report["unstratified_classes"] or 1 in [
        int(c) if str(c).lstrip("-").isdigit() else c for c in manifest.report["unstratified_classes"]
    ]


def test_grouped_kfold_reduces_folds_for_rare_class():
    subject_ids, labels = _synthetic_cells(n_subjects=20, n_classes=10)  # 2 subjects/class
    folds = grouped_kfold(subject_ids, labels, n_folds=5, seed=1)
    assert len(folds) <= 5
    assert len(folds) >= 2


# ─── Grouped K-fold leakage + coverage ───────────────────────────────────────

def test_grouped_kfold_no_leakage_per_fold():
    subject_ids, labels = _synthetic_cells(n_subjects=30, n_classes=3)
    folds = grouped_kfold(subject_ids, labels, n_folds=5, seed=2)
    for fold in folds:
        assert not (set(fold["train"]) & set(fold["val"]))


def test_grouped_kfold_val_sets_cover_all_subjects_once():
    subject_ids, labels = _synthetic_cells(n_subjects=30, n_classes=3)
    folds = grouped_kfold(subject_ids, labels, n_folds=5, seed=2)
    all_val = [s for fold in folds for s in fold["val"]]
    assert sorted(all_val) == sorted(set(str(s) for s in subject_ids))
    assert len(all_val) == len(set(all_val))


# ─── Manifest save/load ──────────────────────────────────────────────────────

def test_manifest_save_and_load_roundtrip(tmp_path):
    subject_ids, labels = _synthetic_cells()
    manifest = subject_train_val_test_split(subject_ids, labels, seed=4)
    path = tmp_path / "split.json"
    manifest.save(path)
    loaded = SplitManifest.load(path)
    assert loaded.train_subjects == manifest.train_subjects
    assert loaded.val_subjects   == manifest.val_subjects
    assert loaded.test_subjects  == manifest.test_subjects
    assert loaded.report == manifest.report


def test_load_or_create_split_reuses_existing_manifest(tmp_path):
    subject_ids, labels = _synthetic_cells()
    path = tmp_path / "manifest.json"
    m1 = load_or_create_split(path, subject_ids, labels, seed=11)
    # Call again with a DIFFERENT seed — should still load the existing file, not re-split
    m2 = load_or_create_split(path, subject_ids, labels, seed=999)
    assert m1.train_subjects == m2.train_subjects
    assert m2.seed == 11


# ─── Split report ────────────────────────────────────────────────────────────

def test_split_report_has_required_fields():
    subject_ids, labels = _synthetic_cells(n_subjects=30, n_classes=3)
    manifest = subject_train_val_test_split(subject_ids, labels, seed=6)
    for split_name in ("train", "val", "test"):
        s = manifest.report["splits"][split_name]
        assert "n_subjects" in s
        assert "n_cells" in s
        assert "class_distribution" in s
        assert "n_unknown_label" in s
        assert s["n_cells"] > 0
        assert s["n_subjects"] > 0


def test_split_report_counts_unknown_labels():
    subject_ids = [f"s{i}" for i in range(10) for _ in range(3)]
    labels = ([0] * 15) + ([np.nan] * 15)
    report = build_split_report(
        subject_ids, labels,
        {"train": [f"s{i}" for i in range(7)], "val": [], "test": [f"s{i}" for i in range(7, 10)]},
    )
    total_unknown = sum(report["splits"][sp]["n_unknown_label"] for sp in report["splits"])
    assert total_unknown == 5  # s5..s9 are unknown-label subjects (5 of the 10)


def test_conflicting_per_subject_labels_raise():
    subject_ids = ["s0", "s0", "s1"]
    labels = [0, 1, 0]  # s0 has two different labels across its cells
    with pytest.raises(ValueError):
        subject_train_val_test_split(subject_ids, labels)
