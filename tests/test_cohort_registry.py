"""
tests/test_cohort_registry.py — Phase 7 cohort/endpoint registry.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.manifest import load_manifest_seed
from evidence.cohort_registry import (
    CohortRegistryError,
    cohorts_supporting,
    cross_check_against_dataset_manifest,
    find_cohort,
    load_cohort_registry,
    registry_fingerprint,
)

COHORTS_YAML = Path(__file__).parents[1] / "configs" / "cohorts.yaml"
DATASETS_YAML = Path(__file__).parents[1] / "configs" / "datasets.yaml"


def test_real_cohort_registry_loads_and_validates():
    cohorts = load_cohort_registry(COHORTS_YAML)
    ids = {c.cohort_id for c in cohorts}
    assert {"gse136831", "gse288003", "gse123352", "gse307690_canuck",
            "tcga_luad", "tcga_lusc", "nlst"} <= ids


def test_real_registry_cross_checks_clean_against_datasets_yaml():
    cohorts = load_cohort_registry(COHORTS_YAML)
    seed = load_manifest_seed(DATASETS_YAML)
    problems = cross_check_against_dataset_manifest(cohorts, seed)
    assert problems == []


def test_gse136831_weak_proxy_never_supports_smoke_classification_directly():
    cohorts = load_cohort_registry(COHORTS_YAML)
    gse136831 = find_cohort(cohorts, "gse136831")
    assert gse136831.task_support["smoke_classification"] != "yes"
    assert gse136831.verified_smoke_label_fields == []
    assert "Disease_Identity" in gse136831.weak_smoke_label_fields


def test_gse288003_mouse_cohort_excluded_from_human_roles():
    cohorts = load_cohort_registry(COHORTS_YAML)
    mouse = find_cohort(cohorts, "gse288003")
    assert mouse.species == "mouse"
    assert mouse.role_eligibility == ["excluded"]
    assert mouse.task_support["external_validation"] != "yes"


def test_bulk_cohorts_never_support_single_cell_tasks():
    cohorts = load_cohort_registry(COHORTS_YAML)
    for cohort_id in ("gse123352", "gse307690_canuck", "tcga_luad", "tcga_lusc"):
        cohort = find_cohort(cohorts, cohort_id)
        assert cohort.single_cell_or_bulk == "bulk"
        assert cohort.task_support["smoke_classification"] != "yes"
        assert cohort.task_support["malignancy_classification"] != "yes"


def test_tcga_cannot_supply_verified_smoke_labels():
    cohorts = load_cohort_registry(COHORTS_YAML)
    for cohort_id in ("tcga_luad", "tcga_lusc"):
        cohort = find_cohort(cohorts, cohort_id)
        assert cohort.verified_smoke_label_fields == []


def test_nlst_is_controlled_access_and_conservative():
    cohorts = load_cohort_registry(COHORTS_YAML)
    nlst = find_cohort(cohorts, "nlst")
    assert nlst.access_level == "controlled"
    assert nlst.expression_outcome_linkable_at_subject_level is False
    assert all(v != "yes" for v in nlst.task_support.values())


def test_no_cohort_claims_subject_level_cancer_prediction_without_linkage():
    cohorts = load_cohort_registry(COHORTS_YAML)
    for c in cohorts:
        if c.task_support["subject_level_cancer_prediction"] == "yes":
            assert c.expression_outcome_linkable_at_subject_level is True


def test_controlled_cohort_entry_rejects_task_support_yes():
    entries = {"cohorts": [{
        "cohort_id": "bad", "accession": "X", "dataset_id": "nlst",
        "access_level": "controlled", "species": "human", "assay_type": "clinical_tabular",
        "single_cell_or_bulk": "not_applicable", "subject_identifier_field": "pid",
        "expression_outcome_linkable_at_subject_level": True,
        "task_support": {
            "smoke_classification": "yes", "malignancy_classification": "no",
            "subject_level_cancer_prediction": "no", "external_validation": "no",
        },
        "role_eligibility": ["development"],
    }]}
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(entries, f)
        path = f.name
    with pytest.raises(CohortRegistryError):
        load_cohort_registry(path)


def test_duplicate_cohort_id_rejected():
    entry = {
        "cohort_id": "dup", "accession": "X", "dataset_id": "gse136831",
        "access_level": "public", "species": "human", "assay_type": "single_cell_rna_seq",
        "single_cell_or_bulk": "single_cell", "subject_identifier_field": "s",
        "expression_outcome_linkable_at_subject_level": False,
        "task_support": {
            "smoke_classification": "no", "malignancy_classification": "no",
            "subject_level_cancer_prediction": "no", "external_validation": "no",
        },
        "role_eligibility": ["development"],
    }
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump({"cohorts": [entry, dict(entry)]}, f)
        path = f.name
    with pytest.raises(CohortRegistryError):
        load_cohort_registry(path)


def test_linkage_contradiction_rejected():
    entry = {
        "cohort_id": "bad_linkage", "accession": "X", "dataset_id": "nlst",
        "access_level": "public", "species": "human", "assay_type": "clinical_tabular",
        "single_cell_or_bulk": "not_applicable", "subject_identifier_field": "pid",
        "expression_outcome_linkable_at_subject_level": False,
        "task_support": {
            "smoke_classification": "no", "malignancy_classification": "no",
            "subject_level_cancer_prediction": "yes",  # contradicts linkage=False
            "external_validation": "no",
        },
        "role_eligibility": ["development"],
    }
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump({"cohorts": [entry]}, f)
        path = f.name
    with pytest.raises(CohortRegistryError):
        load_cohort_registry(path)


def test_cross_check_flags_species_mismatch():
    cohorts = load_cohort_registry(COHORTS_YAML)
    seed = load_manifest_seed(DATASETS_YAML)
    tampered_seed = [dict(d) for d in seed]
    for d in tampered_seed:
        if d["dataset_id"] == "gse288003":
            d["species"] = "human"  # actually mouse — must be flagged
    problems = cross_check_against_dataset_manifest(cohorts, tampered_seed)
    assert any("gse288003" in p and "species" in p for p in problems)


def test_registry_fingerprint_changes_on_content_change():
    cohorts = load_cohort_registry(COHORTS_YAML)
    fp1 = registry_fingerprint(cohorts)
    cohorts[0].species = "changed"
    fp2 = registry_fingerprint(cohorts)
    assert fp1 != fp2


def test_cohorts_supporting_smoke_classification_is_empty_today():
    cohorts = load_cohort_registry(COHORTS_YAML)
    supporting = cohorts_supporting(cohorts, "smoke_classification")
    assert supporting == []
