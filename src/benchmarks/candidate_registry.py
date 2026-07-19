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

This module is ALSO the single source of truth for which domain-robustness
STRATEGIES a candidate kind actually supports (resolve_strategy_application
below) — only pathway_hierarchical_mil (the only registered pathway
module-based MIL candidate) has a domain-adversarial head, a source-aware
batch sampler attachment point, and CORAL/MMD regularizers wired into its
training loop (see pathway_hierarchical_adapter.py); every classical
baseline and every non-module (pooling-based) MIL candidate is trained with
plain ERM regardless of what domain-robustness strategy was requested for
the run. Callers must never assume a requested strategy was actually
applied just because it was passed to a training-time config object —
resolve_strategy_application is the one place that turns "requested" into
"applied" honestly.
"""

from .baselines import CANCER_BASELINES, SMOKE_BASELINES
from .domain_losses import DOMAIN_STRATEGIES
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


# Only pathway_hierarchical_mil supports the full domain-robustness strategy
# set — see this module's docstring. A classical baseline or non-module MIL
# candidate is ALWAYS trained with plain ERM, no matter what strategy was
# requested for the run it happened to win.
SUPPORTED_STRATEGIES_BY_KIND = {
    CANDIDATE_KIND_CLASSICAL_BASELINE: frozenset({"erm"}),
    CANDIDATE_KIND_NON_MODULE_MIL: frozenset({"erm"}),
    CANDIDATE_KIND_PATHWAY_MODULE_MIL: frozenset(DOMAIN_STRATEGIES),
}


class UnknownStrategyNameError(ValueError):
    """Raised when a requested strategy name is not one of
    domain_losses.DOMAIN_STRATEGIES — never silently treated as ERM."""


def resolve_strategy_application(name, requested_strategy: str):
    """
    Returns (applied_strategy, strategy_applicable, reason) for the
    candidate `name` that actually won selection and was fit under
    `requested_strategy`.

    This is the ONE place that turns a caller's requested strategy into
    what was scientifically actually applied: if `name`'s registry-derived
    candidate kind supports `requested_strategy`, it was genuinely applied
    (pathway_hierarchical_mil's training path is only ever invoked WITH the
    requested domain-robustness config when it wins — see
    source_held_out.py) and strategy_applicable=True. Otherwise the
    candidate was trained with plain ERM regardless of what was requested,
    and strategy_applicable=False with a reason explaining the mismatch —
    such a result must never be attributed to the requested strategy.
    """
    if requested_strategy not in DOMAIN_STRATEGIES:
        raise UnknownStrategyNameError(
            f"resolve_strategy_application: requested_strategy={requested_strategy!r} is not one of "
            f"the canonical domain_losses.DOMAIN_STRATEGIES {DOMAIN_STRATEGIES} — refusing to guess."
        )
    kind = resolve_candidate_kind(name)
    supported = SUPPORTED_STRATEGIES_BY_KIND[kind]
    if requested_strategy in supported:
        return (
            requested_strategy, True,
            f"{name!r} (candidate kind={kind!r}) supports strategy={requested_strategy!r}; it was "
            "actually trained with this strategy.",
        )
    return (
        "erm", False,
        f"{name!r} (candidate kind={kind!r}) does not support strategy={requested_strategy!r} — only "
        f"{sorted(supported)} — so it was trained with plain ERM instead; this result is not evidence "
        f"for the requested strategy.",
    )
