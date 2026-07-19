"""
benchmarks/candidate_registry.py — the single source of truth for what KIND
a candidate model name is: classical baseline, non-module (pooling-based)
MIL, or pathway module-based MIL. Derived entirely from the canonical
registries baselines.py (CANCER_BASELINES/SMOKE_BASELINES) and
mil_registry.py (POOLING_BASED_NAMES/PATHWAY_MODEL_NAME) already declare —
never inferred from a caller-supplied boolean flag and never defaulted for
an unrecognized name.

robustness_report.py's validator uses resolve_candidate_kind() to
RECOMPUTE and check a report's declared is_module_based_candidate against
the model name itself, so a report can never falsely declare its own kind.
"""

from .baselines import CANCER_BASELINES, SMOKE_BASELINES
from .mil_registry import PATHWAY_MODEL_NAME, POOLING_BASED_NAMES

CLASSICAL_BASELINE_NAMES = frozenset(set(CANCER_BASELINES) | set(SMOKE_BASELINES))
NON_MODULE_MIL_NAMES = frozenset(POOLING_BASED_NAMES)
PATHWAY_MODULE_MIL_NAMES = frozenset({PATHWAY_MODEL_NAME})

CANDIDATE_KIND_CLASSICAL_BASELINE = "classical_baseline"
CANDIDATE_KIND_NON_MODULE_MIL = "non_module_mil"
CANDIDATE_KIND_PATHWAY_MODULE_MIL = "pathway_module_mil"


class UnknownCandidateNameError(ValueError):
    """Raised when a model name is not present in ANY canonical registry —
    an unrecognized name must never silently default to a candidate kind
    (in particular never "non-module"), since that would let a typo'd or
    newly-added-but-unregistered pathway/module variant slip past the
    module-fingerprint requirement completely undetected."""


def resolve_candidate_kind(name) -> str:
    if name in PATHWAY_MODULE_MIL_NAMES:
        return CANDIDATE_KIND_PATHWAY_MODULE_MIL
    if name in NON_MODULE_MIL_NAMES:
        return CANDIDATE_KIND_NON_MODULE_MIL
    if name in CLASSICAL_BASELINE_NAMES:
        return CANDIDATE_KIND_CLASSICAL_BASELINE
    raise UnknownCandidateNameError(
        f"resolve_candidate_kind: {name!r} is not present in any canonical model registry "
        f"(classical baselines: {sorted(CLASSICAL_BASELINE_NAMES)}; non-module MIL: "
        f"{sorted(NON_MODULE_MIL_NAMES)}; pathway module-based MIL: "
        f"{sorted(PATHWAY_MODULE_MIL_NAMES)}) — refusing to default to a candidate kind for an "
        "unrecognized name."
    )


def is_module_based(name) -> bool:
    return resolve_candidate_kind(name) == CANDIDATE_KIND_PATHWAY_MODULE_MIL
