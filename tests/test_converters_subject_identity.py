"""
tests/test_converters_subject_identity.py — Issue #16 step 4: adversarial
coverage for data/converters.py::_infer_subject_id_column, the function
that parses a verified per-subject donor identifier out of a GEO series
matrix's !Sample_title field (documented today for GSE123352).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.converters import _infer_subject_id_column


def test_verified_subject_ids_parsed_from_patient_titles():
    sample_ids = ["GSM1", "GSM2", "GSM3"]
    titles = [
        "non_involved_lung_tissue_patient_1 A01",
        "non_involved_lung_tissue_patient_2 A02",
        "non_involved_lung_tissue_patient_3 A03",
    ]
    subject_id, verified, note = _infer_subject_id_column("GSE123352", sample_ids, titles)
    assert list(subject_id) == ["GSE123352_patient_1", "GSE123352_patient_2", "GSE123352_patient_3"]
    assert bool(verified.all())


def test_renaming_sample_ids_does_not_change_donor_grouping():
    """The verified subject_id is derived from !Sample_title, never from
    the sample_id (GSM accession) string — renaming every sample_id while
    keeping the same titles must produce the identical subject_id
    assignment."""
    titles = ["patient_5 x", "patient_7 y"]
    original_ids = ["GSM100", "GSM200"]
    renamed_ids = ["SAMPLE_A", "SAMPLE_B"]

    original_subject, original_verified, _ = _infer_subject_id_column("GSE123352", original_ids, titles)
    renamed_subject, renamed_verified, _ = _infer_subject_id_column("GSE123352", renamed_ids, titles)

    assert list(original_subject.values) == list(renamed_subject.values)
    assert bool(original_verified.all()) and bool(renamed_verified.all())


def test_missing_titles_fail_closed_not_fallback_to_verified():
    sample_ids = ["GSM1", "GSM2"]
    subject_id, verified, note = _infer_subject_id_column("GSE123352", sample_ids, [])
    assert list(subject_id) == sample_ids  # falls back to sample_id...
    assert not verified.any()              # ...but is NEVER marked verified
    assert "title" in note.lower()


def test_malformed_title_fails_closed():
    sample_ids = ["GSM1", "GSM2"]
    titles = ["patient_1 x", "unrelated_description_no_patient_number"]
    subject_id, verified, note = _infer_subject_id_column("GSE123352", sample_ids, titles)
    assert not verified.any()
    assert list(subject_id) == sample_ids


def test_duplicate_donor_number_across_samples_fails_closed():
    """Two different samples resolving to the SAME donor number contradicts
    the documented one-sample-per-donor assumption and must fail closed —
    never silently verified as two independent subjects, and never
    silently collapsed into one without flagging the contradiction."""
    sample_ids = ["GSM1", "GSM2"]
    titles = ["patient_9 x", "patient_9 y"]
    subject_id, verified, note = _infer_subject_id_column("GSE123352", sample_ids, titles)
    assert not verified.any()
    assert "same donor" in note.lower()


def test_altered_title_mapping_changes_subject_ids():
    """A one-character change to a title (a different donor number) must
    change the resulting subject_id — the mapping is content-derived, not
    a stable arbitrary label."""
    sample_ids = ["GSM1", "GSM2"]
    titles_a = ["patient_1 x", "patient_2 y"]
    titles_b = ["patient_1 x", "patient_3 y"]
    subject_a, _, _ = _infer_subject_id_column("GSE123352", sample_ids, titles_a)
    subject_b, _, _ = _infer_subject_id_column("GSE123352", sample_ids, titles_b)
    assert list(subject_a.values) != list(subject_b.values)


def test_ambiguous_title_case_insensitive_and_whitespace_variants_still_verified():
    sample_ids = ["GSM1", "GSM2"]
    titles = ["Patient_10 tissue", "PATIENT 11 tissue"]
    subject_id, verified, _ = _infer_subject_id_column("GSE123352", sample_ids, titles)
    assert bool(verified.all())
    assert subject_id.iloc[0] != subject_id.iloc[1]
