"""
tests/test_nlst_smoking.py — NLST CIGSMOK/CIGAR parsing must never coerce
missing, blank, null, malformed, or undocumented values into a verified
smoke class. Fixture data below uses synthetic pid values (p1, p2, ...)
only — never real NLST participant identifiers.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import SMOKE_TYPE_MAP
from data.nlst_smoking import (
    EVIDENCE_POSITIVE,
    EVIDENCE_UNKNOWN,
    parse_nlst_smoking_row,
)


# ── 1/2/3/4/5. parse_nlst_smoking_row: missing / blank / null / malformed ───

def test_missing_both_fields_stay_unknown():
    rec = parse_nlst_smoking_row(None, None)
    assert rec.smoke_type_known is False
    assert rec.effective_smoke_type is None
    assert rec.cigarette_evidence == EVIDENCE_UNKNOWN
    assert rec.cigar_evidence == EVIDENCE_UNKNOWN


def test_one_missing_field_does_not_default_to_zero_and_still_finds_the_other():
    """CIGAR missing entirely must not become 0 (a documented 'no' would
    require an actual 0 code — here it's just absent) — but CIGSMOK=1 is
    independently sufficient for a verified cigarette label regardless."""
    rec = parse_nlst_smoking_row(1, None)
    assert rec.cigarette_evidence == EVIDENCE_POSITIVE
    assert rec.cigar_evidence == EVIDENCE_UNKNOWN
    assert rec.effective_smoke_type == "cigarette"
    assert rec.smoke_type_known is True


def test_blank_string_values_stay_unknown():
    rec = parse_nlst_smoking_row("", "   ")
    assert rec.smoke_type_known is False
    assert rec.cigarette_evidence == EVIDENCE_UNKNOWN
    assert rec.cigar_evidence == EVIDENCE_UNKNOWN


def test_nan_values_stay_unknown():
    rec = parse_nlst_smoking_row(float("nan"), np.nan)
    assert rec.smoke_type_known is False


def test_none_and_pandas_na_stay_unknown():
    rec = parse_nlst_smoking_row(None, pd.NA)
    assert rec.smoke_type_known is False


def test_malformed_string_values_stay_unknown_not_raise():
    rec = parse_nlst_smoking_row("not_a_number", "???")
    assert rec.smoke_type_known is False
    assert rec.cigarette_evidence == EVIDENCE_UNKNOWN
    assert rec.cigar_evidence == EVIDENCE_UNKNOWN


def test_non_integer_float_values_stay_unknown():
    """1.5 is not a documented whole-number code — must not round to 1 or 2."""
    rec = parse_nlst_smoking_row(1.5, 2.5)
    assert rec.smoke_type_known is False


def test_undocumented_numeric_codes_stay_unknown():
    """CIGSMOK=0/CIGAR=0 (and any code outside {1,2}/{1}) are not
    documented in this repository's reviewed NLST access instructions —
    must not be treated as a verified 'never smoked'/'no cigar' negative."""
    rec = parse_nlst_smoking_row(0, 0)
    assert rec.smoke_type_known is False
    assert rec.effective_smoke_type is None

    rec2 = parse_nlst_smoking_row(9, 9)
    assert rec2.smoke_type_known is False


# ── 6. Known supported values map correctly ─────────────────────────────────

def test_cigsmok_current_and_former_both_count_as_cigarette_evidence():
    for code in (1, 2):
        rec = parse_nlst_smoking_row(code, None)
        assert rec.cigarette_evidence == EVIDENCE_POSITIVE
        assert rec.effective_smoke_type == "cigarette"
        assert rec.smoke_type_known is True


def test_float_coded_integer_values_parse_correctly():
    """pandas often reads an int column with any NaN present as float64
    (1.0 instead of 1) — this must still parse as a valid code."""
    rec = parse_nlst_smoking_row(1.0, None)
    assert rec.cigarette_evidence == EVIDENCE_POSITIVE
    assert rec.smoke_type_known is True


def test_cigar_only_maps_to_cigar():
    rec = parse_nlst_smoking_row(None, 1)
    assert rec.effective_smoke_type == "cigar"
    assert rec.smoke_type_known is True


def test_both_positive_maps_to_dual_use():
    rec = parse_nlst_smoking_row(1, 1)
    assert rec.effective_smoke_type == "dual_use"
    assert rec.smoke_type_known is True


# ── 7. Conflicting values do not silently create a verified label ──────────

def test_one_valid_one_invalid_code_only_uses_the_valid_one():
    """CIGSMOK=1 (valid) with CIGAR=7 (undocumented) must produce
    'cigarette', not silently promote the undocumented CIGAR code or
    refuse to use the valid CIGSMOK evidence."""
    rec = parse_nlst_smoking_row(1, 7)
    assert rec.effective_smoke_type == "cigarette"
    assert rec.cigar_evidence == EVIDENCE_UNKNOWN


# ── transfer_nlst_labels integration ────────────────────────────────────────

def _adata_with_subjects(subject_ids):
    n = len(subject_ids)
    obs = pd.DataFrame({
        "subject_id": subject_ids,
        "smoke_type": [0] * n,
        "smoke_type_known": [True] * n,  # simulate an upstream (possibly fabricated) known default
    }, index=[f"c{i}" for i in range(n)])
    return __import__("anndata").AnnData(
        X=np.zeros((n, 3), dtype="float32"), obs=obs,
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )


def test_transfer_nlst_labels_missing_fields_do_not_become_verified_labels():
    from data.labellers import transfer_nlst_labels
    adata = _adata_with_subjects(["p1", "p2"])
    with tempfile.TemporaryDirectory() as tmp:
        nlst_csv = Path(tmp) / "prsn.csv"
        pd.DataFrame({"pid": ["p1", "p2"], "CIGSMOK": [None, ""], "CIGAR": [None, None]}).to_csv(
            nlst_csv, index=False
        )
        out = transfer_nlst_labels(adata, str(nlst_csv))
        assert not out.obs["smoke_type_known"].any()
        assert (out.obs["smoke_type_name"] == "unknown").all()


def test_transfer_nlst_labels_overwrites_upstream_known_true_with_false_when_unsupported():
    """A cell that entered this function with smoke_type_known=True (a
    stale/upstream default) must be corrected to False once NLST linkage
    finds no interpretable evidence — NLST is this project's intended
    smoking-status source of truth for scRNA-seq subjects."""
    from data.labellers import transfer_nlst_labels
    adata = _adata_with_subjects(["p1"])
    assert adata.obs["smoke_type_known"].iloc[0] == True  # noqa: E712 — precondition
    with tempfile.TemporaryDirectory() as tmp:
        nlst_csv = Path(tmp) / "prsn.csv"
        pd.DataFrame({"pid": ["p1"], "CIGSMOK": [0], "CIGAR": [0]}).to_csv(nlst_csv, index=False)
        out = transfer_nlst_labels(adata, str(nlst_csv))
        assert out.obs["smoke_type_known"].iloc[0] == False  # noqa: E712


def test_transfer_nlst_labels_stamps_source_method_limitation_for_unknown():
    from data.labellers import transfer_nlst_labels
    adata = _adata_with_subjects(["p1"])
    with tempfile.TemporaryDirectory() as tmp:
        nlst_csv = Path(tmp) / "prsn.csv"
        pd.DataFrame({"pid": ["p1"], "CIGSMOK": [None], "CIGAR": [None]}).to_csv(nlst_csv, index=False)
        out = transfer_nlst_labels(adata, str(nlst_csv))
        row = out.obs.iloc[0]
        assert row["smoke_type_source"] == "NLST screen.csv CIGSMOK/CIGAR"
        assert row["smoke_type_method"] == "documented_code_lookup"
        assert isinstance(row["smoke_type_limitation"], str) and len(row["smoke_type_limitation"]) > 0


def test_transfer_nlst_labels_never_infers_cigarette_from_participation_alone():
    """A subject present in the NLST CSV with no parseable CIGSMOK/CIGAR
    value must not become cigarette merely by being in the file."""
    from data.labellers import transfer_nlst_labels
    adata = _adata_with_subjects(["p1"])
    with tempfile.TemporaryDirectory() as tmp:
        nlst_csv = Path(tmp) / "prsn.csv"
        pd.DataFrame({"pid": ["p1"]}).to_csv(nlst_csv, index=False)  # no CIGSMOK/CIGAR columns at all
        out = transfer_nlst_labels(adata, str(nlst_csv))
        assert not out.obs["smoke_type_known"].any()
        cigarette_id = SMOKE_TYPE_MAP["cigarette"]
        assert not (out.obs["smoke_type"] == cigarette_id).any()


# ── 8. Corrupting the unknown placeholder cannot affect downstream outputs ──

def test_corrupting_unknown_placeholder_does_not_change_class_weights_or_loss():
    from data.labellers import compute_smoke_class_weights
    from model import MultiTaskLoss
    import torch

    smoke = np.array([0, 0, 1, 5])       # last cell is the unknown placeholder (5)
    known = np.array([True, True, True, False])
    w1 = compute_smoke_class_weights(smoke, n_classes=6, smoke_known=known)

    corrupted = smoke.copy()
    corrupted[-1] = 3  # arbitrary corruption of the unknown cell's placeholder
    w2 = compute_smoke_class_weights(corrupted, n_classes=6, smoke_known=known)
    assert np.allclose(w1, w2)

    loss_fn = MultiTaskLoss()
    logits = torch.randn(4, 6, requires_grad=True)
    targets = torch.tensor(smoke.tolist())
    known_t = torch.tensor(known.tolist())
    l1 = loss_fn._ls(logits, targets, known_t)
    targets2 = torch.tensor(corrupted.tolist())
    l2 = loss_fn._ls(logits, targets2, known_t)
    assert torch.isclose(l1, l2)


def test_corrupting_unknown_placeholder_does_not_change_sampling_index():
    from data.sampling import SubjectClassIndex

    subject_ids = np.array(["s1", "s1", "s2", "s2"])
    labels = np.array([0, 0, 1, 1])
    known = np.array([True, False, True, True])

    idx1 = SubjectClassIndex(subject_ids=subject_ids, labels=labels, num_classes=6, known_mask=known)
    corrupted_labels = labels.copy()
    corrupted_labels[1] = 4  # corrupt the unknown cell's placeholder label
    idx2 = SubjectClassIndex(subject_ids=subject_ids, labels=corrupted_labels, num_classes=6, known_mask=known)

    assert idx1.class_to_subjects == idx2.class_to_subjects
    assert set(idx1.subject_to_indices["s1"].tolist()) == set(idx2.subject_to_indices["s1"].tolist())


# ── 9. Unknown NLST smoke labels appear in the label-quality report ────────

def test_nlst_join_report_counts_verified_and_unknown_cells():
    import scipy.sparse as sp
    import anndata as ad
    from preprocess import _nlst_join_report

    n = 4
    genes = [f"G{i}" for i in range(5)]
    obs = pd.DataFrame({
        "subject_id": ["p1", "p2", "p3", "p4"],
    }, index=[f"c{i}" for i in range(n)])
    merged = ad.AnnData(X=sp.csr_matrix(np.random.rand(n, 5).astype("float32")), obs=obs,
                         var=pd.DataFrame(index=genes))

    with tempfile.TemporaryDirectory() as tmp:
        nlst_csv = Path(tmp) / "screen.csv"
        pd.DataFrame({
            "pid": ["p1", "p2", "p3", "p4"],
            "CIGSMOK": [1, 0, 2, None],
            "CIGAR": [0, 0, 1, None],
        }).to_csv(nlst_csv, index=False)

        from data.labellers import transfer_nlst_labels
        merged = transfer_nlst_labels(merged, str(nlst_csv))

        report = _nlst_join_report({"nlst_csv": str(nlst_csv)}, merged)
        assert report["nlst_cells_verified_smoke_label"] == 2   # p1 (cigarette), p3 (dual_use)
        assert report["nlst_cells_unknown_smoke_label"] == 2    # p2 (0/0), p4 (missing)


def test_label_quality_report_flags_nlst_unknown_cells():
    from data.label_quality_report import build_label_quality_report

    class _FakeManifest:
        report = {"splits": {"train": {"n_subjects": 1, "n_cells": 1}}}
        fingerprint = "fp"

    pipeline_result = {
        "split_manifest": _FakeManifest(),
        "label_provenance_report": {
            "cancer_outcome_known_subjects": 1,
            "cancer_outcome_unknown_subjects": 0,
            "nlst_cells_verified_smoke_label": 2,
            "nlst_cells_unknown_smoke_label": 3,
        },
    }
    report = build_label_quality_report(pipeline_result)
    assert any("nlst_matched_but_unknown_smoke_label" in f for f in report.flags)
    assert report.per_split["all"]["nlst_cells_unknown_smoke_label"] == 3


# ── 10. Unknown cancer outcomes and unknown smoke histories are independent ─

def test_unknown_cancer_outcome_and_unknown_smoke_history_are_independent():
    """A subject can have a known cancer outcome with unknown smoking
    history, or vice versa — the two must be tracked/excluded
    independently, never conflated."""
    import scipy.sparse as sp
    import anndata as ad
    from data.labellers import transfer_nlst_labels
    from data.assembly import assemble_subject_bags

    n = 2
    genes = [f"G{i}" for i in range(5)]
    obs = pd.DataFrame({
        "subject_id": ["p1", "p2"],
        "cell_type_id": [0, 0],
        "malignancy": [0.0, 0.0],
        "malignancy_known": [False, False],
    }, index=["c0", "c1"])
    merged = ad.AnnData(X=sp.csr_matrix(np.random.rand(n, 5).astype("float32")), obs=obs,
                         var=pd.DataFrame(index=genes))

    with tempfile.TemporaryDirectory() as tmp:
        nlst_csv = Path(tmp) / "screen.csv"
        # p1: known cigarette history. p2: unknown smoking history.
        pd.DataFrame({"pid": ["p1", "p2"], "CIGSMOK": [1, None], "CIGAR": [None, None]}).to_csv(
            nlst_csv, index=False
        )
        merged = transfer_nlst_labels(merged, str(nlst_csv))

    # p1 has a KNOWN cancer outcome, p2 does not — independent of smoke history.
    outcomes = pd.DataFrame({"subject_id": ["p1"], "cancer_label": [1]})
    bags = {b["subject_id"]: b for b in assemble_subject_bags(merged, outcomes, min_cells_per_subject=1)}

    assert bags["p1"]["cancer_label_known"] is True
    assert bool(bags["p1"]["smoke_known"][0]) is True
    assert bags["p2"]["cancer_label_known"] is False
    assert bool(bags["p2"]["smoke_known"][0]) is False


# ── 11. Fixture data contains no real participant identifiers ──────────────

def test_fixture_pids_are_synthetic_placeholders_only():
    """A guard against accidentally using a real-looking NLST participant
    ID pattern in this test file — every pid used above is a short
    synthetic placeholder ("p1", "p2", ...), never a bare large numeric
    study ID (the shape a real NLST pid would take)."""
    import re
    this_file = Path(__file__).read_text()
    pid_lists = re.findall(r'"pid":\s*\[([^\]]*)\]', this_file)
    assert pid_lists, "expected at least one pid fixture in this file"
    for pid_list in pid_lists:
        for token in re.findall(r'"([^"]+)"', pid_list):
            assert re.fullmatch(r"p\d+", token), (
                f"fixture pid {token!r} does not match the synthetic p<n> placeholder "
                "pattern — only synthetic pids are permitted in this test file."
            )
