"""
Phase 4 — frozen-test-data access sentinels (Step 11).

Unit tests for FrozenAccessSentinel's individual access-pattern rejections,
plus one end-to-end proof: the full development pipeline (CV, OOF
generation, final development-pool fit) runs to completion with
ExperimentContext.test_bags/test_cell_dataset replaced by sentinels,
without ever triggering one — a real, testable property, not just a
code-review claim.
"""
import dataclasses
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.sentinel import FrozenAccessSentinel, FrozenDataAccessError, wrap_frozen


# ─── Unit tests: every access pattern must raise ───────────────────────────

def test_attribute_access_raises():
    s = FrozenAccessSentinel("test labels")
    with pytest.raises(FrozenDataAccessError):
        s.some_attribute


def test_gene_name_style_attribute_access_raises():
    s = FrozenAccessSentinel("test gene names")
    with pytest.raises(FrozenDataAccessError):
        s.var_names


def test_batch_access_raises():
    s = FrozenAccessSentinel("test batch metadata")
    with pytest.raises(FrozenDataAccessError):
        s.batch


def test_label_access_raises():
    s = FrozenAccessSentinel("test labels")
    with pytest.raises(FrozenDataAccessError):
        s.cancer_label


def test_iteration_raises():
    s = FrozenAccessSentinel()
    with pytest.raises(FrozenDataAccessError):
        for _ in s:
            pass


def test_indexing_raises():
    s = FrozenAccessSentinel()
    with pytest.raises(FrozenDataAccessError):
        s[0]


def test_len_raises():
    s = FrozenAccessSentinel()
    with pytest.raises(FrozenDataAccessError):
        len(s)


def test_array_conversion_raises():
    s = FrozenAccessSentinel()
    with pytest.raises(FrozenDataAccessError):
        np.asarray(s)


def test_dataframe_conversion_raises():
    s = FrozenAccessSentinel()
    with pytest.raises(FrozenDataAccessError):
        pd.DataFrame(s)


def test_bool_conversion_raises():
    s = FrozenAccessSentinel()
    with pytest.raises(FrozenDataAccessError):
        bool(s)


def test_repr_raises():
    s = FrozenAccessSentinel()
    with pytest.raises(FrozenDataAccessError):
        repr(s)


def test_contains_raises():
    s = FrozenAccessSentinel()
    with pytest.raises(FrozenDataAccessError):
        "x" in s


def test_wrap_frozen_never_stores_the_real_value():
    real_secret = {"cancer_label": 1, "subject_id": "real_subject"}
    s = wrap_frozen(real_secret, label="test bag")
    assert "_sentinel_label" in object.__getattribute__(s, "__dict__")
    assert real_secret not in object.__getattribute__(s, "__dict__").values()
    with pytest.raises(FrozenDataAccessError):
        s.get("cancer_label")


# ─── End-to-end: the dev pipeline never triggers a sentinel ───────────────

def test_development_pipeline_never_touches_sentinel_wrapped_test_data():
    """Replace test_bags/test_cell_dataset with sentinels on a real
    synthetic ExperimentContext, then run CV (Task A and Task B) and OOF
    generation + the final development-pool fit — the same development-only
    call sequence runner.py uses before ever acquiring the frozen-test
    guard. If any of these functions touches test_bags/test_cell_dataset,
    the sentinel raises immediately and this test fails with a traceback
    pointing at the exact call site."""
    from benchmarks.cross_validation import run_cancer_cv, run_smoke_cv
    from benchmarks.final_evaluation import fit_final_candidate_on_dev_pool, generate_subject_oof_predictions
    from benchmarks.runner import build_synthetic_context

    ctx = build_synthetic_context(seed=1, fast=True)
    ctx = dataclasses.replace(
        ctx,
        test_bags=FrozenAccessSentinel("test_bags"),
        test_cell_dataset=FrozenAccessSentinel("test_cell_dataset"),
    )

    try:
        run_smoke_cv(ctx, ["majority"], n_folds=2, seeds=[42])
        run_cancer_cv(ctx, ["prevalence"], n_folds=2, seeds=[42])

        dev_subjects = sorted(set(ctx.subjects_for("train")) | set(ctx.subjects_for("val")))
        outcomes_by_subject = {
            str(b["subject_id"]): b["cancer_label"]
            for b in list(ctx.train_bags) + list(ctx.val_bags) if b.get("cancer_label_known")
        }
        num_cell_types = ctx.config.get("model", {}).get("num_cell_types", 4)
        min_cells = ctx.config.get("data", {}).get("min_cells_per_subject", 5)
        n_hvgs = ctx.preprocessing_artifact.n_hvgs

        oof = generate_subject_oof_predictions(
            ctx, "logistic", dev_subjects, outcomes_by_subject, num_cell_types, min_cells, n_hvgs,
            seed=42, n_folds=2,
        )
        assert "oof_by_subject" in oof

        fitted = fit_final_candidate_on_dev_pool(
            ctx, "logistic", dev_subjects, outcomes_by_subject, {}, num_cell_types, min_cells, n_hvgs,
            device="cpu", seed=42,
        )
        assert fitted.preprocessing_artifact_fingerprint
    finally:
        shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)
