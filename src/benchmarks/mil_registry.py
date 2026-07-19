"""
benchmarks/mil_registry.py — the one place that maps a Task A/B MIL-kind
candidate name to the adapter class and search space that trains it, so
cross_validation.py, final_evaluation.py, and runner.py's ablation runner
share one dispatch instead of each growing its own copy of "if name is one
of these, do X."

Adding a new MIL-kind model means adding it here once, not touching every
call site that currently special-cases "neural"/"mean_mil"/"max_mil"/
"attention_mil" inline.
"""

from typing import Dict, Optional

from .neural import NeuralCancerAdapter
from .pathway_hierarchical_adapter import MODEL_NAME as PATHWAY_MODEL_NAME
from .pathway_hierarchical_adapter import PathwayHierarchicalAdapter

POOLING_BASED_NAMES = ("neural", "mean_mil", "max_mil", "attention_mil")
MIL_CANDIDATE_NAMES = POOLING_BASED_NAMES + (PATHWAY_MODEL_NAME,)

# Small, declared search space for pathway_hierarchical_mil — architecture
# and loss-weight candidates only (epoch count is not searched, exactly
# like MIL_SEARCH_SPACE's pretrain_epochs is the only thing searched for
# the pooling-based MIL models; the two search spaces intentionally cover
# different knobs of their respective models).
PATHWAY_SEARCH_SPACE = {
    "embedding_dim": [64, 128],
    "attention_dim": [32, 64],
    "dropout": [0.1, 0.3],
    "use_gene_residual": [True, False],
    "smoke_loss_weight": [0.5, 1.0],
    "cancer_loss_weight": [0.5, 1.0],
}
# Reduced grid for --synthetic --fast runs (CI) — same knobs, fewer values,
# so the nested search still exercises real candidate comparison without
# the full 2^4 * 2 = 32-candidate grid's cost.
PATHWAY_SEARCH_SPACE_FAST = {
    "embedding_dim": [16],
    "attention_dim": [8],
    "dropout": [0.1],
    "use_gene_residual": [True],
    "smoke_loss_weight": [1.0],
    "cancer_loss_weight": [1.0],
}


def pathway_search_space(fast: bool = False) -> Dict:
    return dict(PATHWAY_SEARCH_SPACE_FAST if fast else PATHWAY_SEARCH_SPACE)


def build_mil_adapter(
    candidate_name: str, pooling: Optional[str], device: str, config_overrides: Optional[Dict] = None,
    domain_robustness_config: Optional[Dict] = None,
):
    """Construct the adapter for `candidate_name`. config_overrides is only
    meaningful for pathway_hierarchical_mil (its selected hyperparameter
    candidate); it is ignored for the pooling-based names, which take their
    selected candidate (pretrain_epochs) at fit() time instead, exactly as
    they always have.

    domain_robustness_config (Phase 6) is likewise only meaningful for
    pathway_hierarchical_mil — the only architecture with a subject-
    embedding attachment point for CORAL/MMD/domain-adversarial
    regularizers (see pathway_hierarchical_adapter.py). It is silently
    ignored (never applied, never an error) for the pooling-based names,
    which have no such attachment point today; a caller that actually
    requires domain-robust training for those architectures must be told so
    explicitly by its own caller, not by this dispatch function."""
    if candidate_name == PATHWAY_MODEL_NAME:
        return PathwayHierarchicalAdapter(
            device=device, config_overrides=config_overrides or {},
            domain_robustness_config=domain_robustness_config,
        )
    if candidate_name in POOLING_BASED_NAMES:
        return NeuralCancerAdapter(pooling=pooling or "attention", device=device)
    raise ValueError(f"build_mil_adapter: unknown MIL-kind candidate {candidate_name!r}")
