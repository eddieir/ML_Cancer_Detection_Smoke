"""
evidence/cohort_registry.py — loads and validates configs/cohorts.yaml
(Step 4: canonical cohort and endpoint registry).

This module does not download, open, or process any dataset. It only
reasons about the declarative facts in configs/cohorts.yaml, and
cross-checks them against configs/datasets.yaml (the existing Phase 1-6
provenance seed) so the two files cannot silently drift apart — e.g. a
cohort registry entry claiming a species that its datasets.yaml
counterpart does not.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .errors import EvidenceContractError

REQUIRED_COHORT_FIELDS = (
    "cohort_id", "accession", "dataset_id", "access_level", "species",
    "assay_type", "single_cell_or_bulk", "subject_identifier_field",
    "expression_outcome_linkable_at_subject_level", "task_support",
    "role_eligibility",
)

VALID_ACCESS_LEVELS = ("public", "controlled")
VALID_ROLES = ("development", "internal_test", "external_validation", "excluded")
VALID_TASK_SUPPORT_VALUES = ("yes", "no", "not_currently")
TASKS = (
    "smoke_classification", "malignancy_classification",
    "subject_level_cancer_prediction", "external_validation",
)


class CohortRegistryError(EvidenceContractError):
    """Raised for malformed cohort-registry entries or contradictions
    between configs/cohorts.yaml and configs/datasets.yaml."""


@dataclass
class Cohort:
    cohort_id: str
    accession: str
    dataset_id: str
    access_level: str
    species: str
    assay_type: str
    single_cell_or_bulk: str
    subject_identifier_field: str
    expression_outcome_linkable_at_subject_level: bool
    task_support: Dict[str, str]
    role_eligibility: List[str]
    biological_tissue: Optional[str] = None
    sample_identifier_field: Optional[str] = None
    cell_identifier_field: Optional[str] = None
    verified_smoke_label_fields: List[str] = field(default_factory=list)
    weak_smoke_label_fields: List[str] = field(default_factory=list)
    malignancy_label_fields: List[str] = field(default_factory=list)
    cancer_outcome_fields: List[str] = field(default_factory=list)
    outcome_time_horizon: Optional[str] = None
    censoring_fields: List[str] = field(default_factory=list)
    source_limitations: List[str] = field(default_factory=list)
    licensing_access_limitations: Optional[str] = None
    checksum_rule: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    version_or_retrieval_date: Optional[str] = None

    def supports(self, task: str) -> bool:
        return self.task_support.get(task) == "yes"

    def eligible_for_role(self, role: str) -> bool:
        return role in self.role_eligibility


def _validate_entry(raw: dict) -> None:
    missing = [f for f in REQUIRED_COHORT_FIELDS if f not in raw]
    if missing:
        raise CohortRegistryError(
            f"cohort entry {raw.get('cohort_id', '<unknown>')!r} missing required field(s) {missing}"
        )
    if raw["access_level"] not in VALID_ACCESS_LEVELS:
        raise CohortRegistryError(
            f"cohort {raw['cohort_id']!r}: access_level {raw['access_level']!r} not in {VALID_ACCESS_LEVELS}"
        )
    for role in raw["role_eligibility"]:
        if role not in VALID_ROLES:
            raise CohortRegistryError(f"cohort {raw['cohort_id']!r}: invalid role {role!r} in role_eligibility")
    task_support = raw["task_support"]
    missing_tasks = [t for t in TASKS if t not in task_support]
    if missing_tasks:
        raise CohortRegistryError(f"cohort {raw['cohort_id']!r}: task_support missing {missing_tasks}")
    for t, v in task_support.items():
        if v not in VALID_TASK_SUPPORT_VALUES:
            raise CohortRegistryError(
                f"cohort {raw['cohort_id']!r}: task_support[{t!r}]={v!r} not in {VALID_TASK_SUPPORT_VALUES}"
            )
    if not isinstance(raw["expression_outcome_linkable_at_subject_level"], bool):
        raise CohortRegistryError(
            f"cohort {raw['cohort_id']!r}: expression_outcome_linkable_at_subject_level must be a bool"
        )
    # A cohort with no verified subject-level expression<->outcome linkage
    # must not simultaneously claim it supports subject-level cancer
    # prediction — that combination is a contradiction the registry itself
    # must reject rather than allow a downstream report to fabricate it.
    if (not raw["expression_outcome_linkable_at_subject_level"]
            and task_support.get("subject_level_cancer_prediction") == "yes"):
        raise CohortRegistryError(
            f"cohort {raw['cohort_id']!r}: claims subject_level_cancer_prediction='yes' but "
            "expression_outcome_linkable_at_subject_level=False — a subject-level cancer "
            "prediction claim requires genuine linkage, which this entry says does not exist"
        )
    if raw["access_level"] == "controlled" and any(
            v == "yes" for v in task_support.values()):
        raise CohortRegistryError(
            f"cohort {raw['cohort_id']!r}: access_level='controlled' cohort cannot claim "
            "task_support='yes' for any task without an authorized local dataset being "
            "verified at evaluation time — the registry entry itself must stay conservative"
        )


def load_cohort_registry(path) -> List[Cohort]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cohorts = []
    seen_ids = set()
    for entry in raw.get("cohorts", []):
        _validate_entry(entry)
        if entry["cohort_id"] in seen_ids:
            raise CohortRegistryError(f"duplicate cohort_id {entry['cohort_id']!r} in {path}")
        seen_ids.add(entry["cohort_id"])
        cohorts.append(Cohort(
            cohort_id=entry["cohort_id"],
            accession=entry["accession"],
            dataset_id=entry["dataset_id"],
            access_level=entry["access_level"],
            species=entry["species"],
            assay_type=entry["assay_type"],
            single_cell_or_bulk=entry["single_cell_or_bulk"],
            subject_identifier_field=entry["subject_identifier_field"],
            expression_outcome_linkable_at_subject_level=entry["expression_outcome_linkable_at_subject_level"],
            task_support=entry["task_support"],
            role_eligibility=entry["role_eligibility"],
            biological_tissue=entry.get("biological_tissue"),
            sample_identifier_field=entry.get("sample_identifier_field"),
            cell_identifier_field=entry.get("cell_identifier_field"),
            verified_smoke_label_fields=entry.get("verified_smoke_label_fields", []),
            weak_smoke_label_fields=entry.get("weak_smoke_label_fields", []),
            malignancy_label_fields=entry.get("malignancy_label_fields", []),
            cancer_outcome_fields=entry.get("cancer_outcome_fields", []),
            outcome_time_horizon=entry.get("outcome_time_horizon"),
            censoring_fields=entry.get("censoring_fields", []),
            source_limitations=entry.get("source_limitations", []),
            licensing_access_limitations=entry.get("licensing_access_limitations"),
            checksum_rule=entry.get("checksum_rule"),
            notes=entry.get("notes", []),
            version_or_retrieval_date=entry.get("version_or_retrieval_date"),
        ))
    return cohorts


def cross_check_against_dataset_manifest(cohorts: List[Cohort], dataset_seed_entries: List[dict]) -> List[str]:
    """Returns a list of contradiction strings (empty if none) between
    configs/cohorts.yaml entries and the corresponding configs/datasets.yaml
    seed entries (dataset_id, accession, species must agree)."""
    by_id = {d["dataset_id"]: d for d in dataset_seed_entries}
    problems = []
    for c in cohorts:
        d = by_id.get(c.dataset_id)
        if d is None:
            problems.append(f"cohort {c.cohort_id!r} references dataset_id {c.dataset_id!r} not present in configs/datasets.yaml")
            continue
        if d["accession"] != c.accession:
            problems.append(
                f"cohort {c.cohort_id!r}: accession {c.accession!r} disagrees with "
                f"configs/datasets.yaml accession {d['accession']!r}"
            )
        if d["species"] != c.species:
            problems.append(
                f"cohort {c.cohort_id!r}: species {c.species!r} disagrees with "
                f"configs/datasets.yaml species {d['species']!r}"
            )
        d_controlled = d.get("controlled_access", False)
        c_controlled = c.access_level == "controlled"
        if d_controlled != c_controlled:
            problems.append(
                f"cohort {c.cohort_id!r}: access_level={c.access_level!r} disagrees with "
                f"configs/datasets.yaml controlled_access={d_controlled!r}"
            )
    return problems


def registry_fingerprint(cohorts: List[Cohort]) -> str:
    payload = []
    for c in sorted(cohorts, key=lambda c: c.cohort_id):
        d = dict(c.__dict__)
        payload.append(d)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def find_cohort(cohorts: List[Cohort], cohort_id: str) -> Cohort:
    for c in cohorts:
        if c.cohort_id == cohort_id:
            return c
    raise CohortRegistryError(f"no cohort registered with cohort_id={cohort_id!r}")


def cohorts_supporting(cohorts: List[Cohort], task: str, role: Optional[str] = None) -> List[Cohort]:
    out = [c for c in cohorts if c.supports(task)]
    if role is not None:
        out = [c for c in out if c.eligible_for_role(role)]
    return out
