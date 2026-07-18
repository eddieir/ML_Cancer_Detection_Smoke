"""
benchmarks/source_eligibility.py — per-source eligibility assessment and the
versioned, fingerprinted source-held-out split manifest for the domain-
robustness (source-held-out / "LOSO") protocol.

A source is not automatically eligible to serve as a held-out external-
domain evaluation set just because it is present in the dataset. This module
decides, PER TASK, whether a given source can be held out at all, and
records an explicit machine-readable reason whenever it cannot — never a
silent skip and never a fabricated result for an ineligible source. See
benchmarks/ood.py for the pre-existing (Task A classical-baseline-only, no
manifest) leave-one-source-out implementation this module generalizes.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

ELIGIBLE = "eligible"
NOT_EVALUABLE = "not_evaluable"
DIAGNOSTIC_ONLY = "diagnostic_only"
EXCLUDED_BY_POLICY = "excluded_by_policy"
CONTROLLED_ACCESS_UNAVAILABLE = "controlled_access_unavailable"
INSUFFICIENT_CLASSES = "insufficient_classes"
INSUFFICIENT_OUTCOMES = "insufficient_outcomes"
SPECIES_MISMATCH = "species_mismatch"
ASSAY_MISMATCH = "assay_mismatch"
GENE_CONTRACT_MISMATCH = "gene_contract_mismatch"

SOURCE_ELIGIBILITY_STATUSES = (
    ELIGIBLE, NOT_EVALUABLE, DIAGNOSTIC_ONLY, EXCLUDED_BY_POLICY,
    CONTROLLED_ACCESS_UNAVAILABLE, INSUFFICIENT_CLASSES, INSUFFICIENT_OUTCOMES,
    SPECIES_MISMATCH, ASSAY_MISMATCH, GENE_CONTRACT_MISMATCH,
)

MIN_SUBJECTS_PER_SOURCE_SMOKE = 3
MIN_CLASSES_PER_SOURCE_SMOKE = 2
MIN_SUBJECTS_PER_SOURCE_CANCER = 4
MIN_POSITIVE_PER_SOURCE_CANCER = 1
MIN_NEGATIVE_PER_SOURCE_CANCER = 1


@dataclass
class SourceEligibilityReport:
    source: str
    task: str
    status: str
    eligible: bool
    reasons: List[str] = field(default_factory=list)
    counts: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source, "task": self.task, "status": self.status,
            "eligible": self.eligible, "reasons": self.reasons, "counts": self.counts,
        }


def _not_eligible(source: str, task: str, status: str, reason: str, counts: Optional[Dict] = None) -> SourceEligibilityReport:
    return SourceEligibilityReport(
        source=source, task=task, status=status, eligible=False, reasons=[reason], counts=counts or {},
    )


def assess_smoke_source_eligibility(
    source: str,
    held_out_subject_labels: Sequence[int],
    incompatible_sources: Optional[Sequence[str]] = None,
    species_by_source: Optional[Dict[str, str]] = None,
    reference_species: Optional[str] = None,
    controlled_access_sources: Optional[Sequence[str]] = None,
) -> SourceEligibilityReport:
    """
    held_out_subject_labels: effective smoke-type label per subject in this
    source (already restricted to subjects with a VERIFIED, non-weak-proxy
    label by the caller — see data/sampling.py's own known_mask exclusion
    for the same rule applied to training).
    """
    incompatible_sources = set(incompatible_sources or [])
    species_by_source = species_by_source or {}
    controlled_access_sources = set(controlled_access_sources or [])
    counts = {"n_subjects": len(held_out_subject_labels),
              "classes_present": sorted(set(int(l) for l in held_out_subject_labels))}

    if source in controlled_access_sources:
        return _not_eligible(source, "smoke_classification", CONTROLLED_ACCESS_UNAVAILABLE,
                              "source is controlled-access and not available in this environment", counts)
    if source in incompatible_sources:
        return _not_eligible(source, "smoke_classification", EXCLUDED_BY_POLICY,
                              "explicitly listed as label-semantics incompatible", counts)
    if source not in species_by_source:
        return _not_eligible(source, "smoke_classification", SPECIES_MISMATCH,
                              "no species metadata declared for this source — unknown metadata "
                              "defaults to not-comparable, never assumed-compatible", counts)
    if reference_species is not None and species_by_source[source] != reference_species:
        return _not_eligible(source, "smoke_classification", SPECIES_MISMATCH,
                              f"species {species_by_source[source]!r} != reference {reference_species!r}", counts)
    if counts["n_subjects"] < MIN_SUBJECTS_PER_SOURCE_SMOKE:
        return _not_eligible(source, "smoke_classification", NOT_EVALUABLE,
                              f"only {counts['n_subjects']} evaluable subject(s) "
                              f"(< {MIN_SUBJECTS_PER_SOURCE_SMOKE})", counts)
    if len(counts["classes_present"]) < MIN_CLASSES_PER_SOURCE_SMOKE:
        return _not_eligible(source, "smoke_classification", INSUFFICIENT_CLASSES,
                              f"only {len(counts['classes_present'])} class(es) present in held-out "
                              f"source (< {MIN_CLASSES_PER_SOURCE_SMOKE}) — macro-F1 would not reflect "
                              "genuine multi-class discrimination", counts)
    return SourceEligibilityReport(source=source, task="smoke_classification", status=ELIGIBLE,
                                     eligible=True, reasons=[], counts=counts)


def assess_cancer_source_eligibility(
    source: str,
    held_out_outcomes: Sequence[Optional[int]],
    incompatible_sources: Optional[Sequence[str]] = None,
    species_by_source: Optional[Dict[str, str]] = None,
    reference_species: Optional[str] = None,
    controlled_access_sources: Optional[Sequence[str]] = None,
) -> SourceEligibilityReport:
    """held_out_outcomes: one entry per subject in this source; None means
    unknown outcome (never coerced to 0/negative — excluded from counts, not
    from the subject list)."""
    incompatible_sources = set(incompatible_sources or [])
    species_by_source = species_by_source or {}
    controlled_access_sources = set(controlled_access_sources or [])
    known = [o for o in held_out_outcomes if o is not None]
    n_pos = sum(1 for o in known if int(o) == 1)
    n_neg = sum(1 for o in known if int(o) == 0)
    counts = {"n_subjects_total": len(held_out_outcomes), "n_known_outcome": len(known),
              "n_positive": n_pos, "n_negative": n_neg}

    if source in controlled_access_sources:
        return _not_eligible(source, "cancer_prediction", CONTROLLED_ACCESS_UNAVAILABLE,
                              "source is controlled-access and not available in this environment", counts)
    if source in incompatible_sources:
        return _not_eligible(source, "cancer_prediction", EXCLUDED_BY_POLICY,
                              "explicitly listed as label-semantics incompatible", counts)
    if source not in species_by_source:
        return _not_eligible(source, "cancer_prediction", SPECIES_MISMATCH,
                              "no species metadata declared for this source", counts)
    if reference_species is not None and species_by_source[source] != reference_species:
        return _not_eligible(source, "cancer_prediction", SPECIES_MISMATCH,
                              f"species {species_by_source[source]!r} != reference {reference_species!r}", counts)
    if len(known) < MIN_SUBJECTS_PER_SOURCE_CANCER:
        return _not_eligible(source, "cancer_prediction", NOT_EVALUABLE,
                              f"only {len(known)} known-outcome subject(s) "
                              f"(< {MIN_SUBJECTS_PER_SOURCE_CANCER})", counts)
    if n_pos < MIN_POSITIVE_PER_SOURCE_CANCER or n_neg < MIN_NEGATIVE_PER_SOURCE_CANCER:
        # Not fully excluded — a source with only one class can still report
        # threshold-based metrics (sensitivity OR specificity, whichever is
        # defined), it just can't report AUROC/AUPRC. Callers must mark
        # those specific metrics undefined rather than substitute 0.5 — see
        # eligibility.py::check_test_evaluability for the identical pattern
        # applied to the frozen test split.
        return SourceEligibilityReport(
            source=source, task="cancer_prediction", status=INSUFFICIENT_OUTCOMES, eligible=True,
            reasons=[f"only one outcome class present (pos={n_pos}, neg={n_neg}) — AUROC/AUPRC "
                     "undefined for this source, other metrics remain defined"],
            counts=counts,
        )
    return SourceEligibilityReport(source=source, task="cancer_prediction", status=ELIGIBLE,
                                     eligible=True, reasons=[], counts=counts)


# ─── Source-held-out split manifest ────────────────────────────────────────

def _sha256_json(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _subject_list_fingerprint(subject_ids: Sequence[str]) -> str:
    """Deterministic, privacy-safe fingerprint of a subject-ID set — never
    the raw IDs themselves in any shareable report (only this hash is)."""
    return _sha256_json(sorted(str(s) for s in subject_ids))


SPLIT_MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass
class SourceHeldOutSplitManifest:
    schema_version: str
    task: str
    held_out_source: str
    development_sources: List[str]
    development_subject_fingerprint: str
    held_out_subject_fingerprint: str
    n_development_subjects: int
    n_held_out_subjects: int
    known_label_counts: Dict
    class_distribution: Dict
    species: Optional[str]
    assay_mode: Optional[str]
    dataset_manifest_fingerprint: Optional[str]
    label_policy_fingerprint: Optional[str]
    preprocessing_policy_fingerprint: Optional[str]
    module_fingerprint: Optional[str]
    seed: int
    eligibility: Dict
    exclusion_reasons: List[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        payload = {k: v for k, v in self.__dict__.items() if k != "manifest_fingerprint"}
        return _sha256_json(payload)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["manifest_fingerprint"] = self.fingerprint()
        return d


def build_source_held_out_manifest(
    task: str,
    held_out_source: str,
    development_sources: Sequence[str],
    development_subjects: Sequence[str],
    held_out_subjects: Sequence[str],
    known_label_counts: Dict,
    class_distribution: Dict,
    eligibility: SourceEligibilityReport,
    seed: int,
    species: Optional[str] = None,
    assay_mode: Optional[str] = None,
    dataset_manifest_fingerprint: Optional[str] = None,
    label_policy_fingerprint: Optional[str] = None,
    preprocessing_policy_fingerprint: Optional[str] = None,
    module_fingerprint: Optional[str] = None,
) -> SourceHeldOutSplitManifest:
    """
    Build the versioned manifest. Changing held_out_source, or the exact
    membership of development_subjects/held_out_subjects, changes
    fingerprint() — this is the property the required tests check (see
    tests/test_source_held_out_manifest.py).
    """
    dev_set = set(str(s) for s in development_subjects)
    held_set = set(str(s) for s in held_out_subjects)
    overlap = dev_set & held_set
    if overlap:
        raise ValueError(
            f"build_source_held_out_manifest: {len(overlap)} subject(s) appear in both development "
            f"and held-out sets — e.g. {sorted(overlap)[:5]} — source-held-out isolation requires "
            "disjoint subject sets."
        )
    return SourceHeldOutSplitManifest(
        schema_version=SPLIT_MANIFEST_SCHEMA_VERSION, task=task, held_out_source=str(held_out_source),
        development_sources=sorted(str(s) for s in development_sources),
        development_subject_fingerprint=_subject_list_fingerprint(development_subjects),
        held_out_subject_fingerprint=_subject_list_fingerprint(held_out_subjects),
        n_development_subjects=len(dev_set), n_held_out_subjects=len(held_set),
        known_label_counts=known_label_counts, class_distribution=class_distribution,
        species=species, assay_mode=assay_mode,
        dataset_manifest_fingerprint=dataset_manifest_fingerprint,
        label_policy_fingerprint=label_policy_fingerprint,
        preprocessing_policy_fingerprint=preprocessing_policy_fingerprint,
        module_fingerprint=module_fingerprint, seed=seed,
        eligibility=eligibility.to_dict(), exclusion_reasons=list(eligibility.reasons),
    )
