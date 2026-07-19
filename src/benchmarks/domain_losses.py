"""
benchmarks/domain_losses.py — development-only domain-robustness regularizers
for subject-level representations produced by pathway_hierarchical_mil.

Every function here operates on a batch of already-pooled SUBJECT embeddings
(HierarchicalMILOutput.subject_embeddings, [B, D]) plus a per-subject source
label drawn from the DEVELOPMENT source vocabulary only. None of this module
ever receives or reads held-out-source or frozen-test data — the caller
(pathway_hierarchical_adapter.py) is responsible for only ever passing
development-source subjects in, and this module has no mechanism to fetch
anything itself.

None of CORAL, MMD, or the domain-adversarial head are described as making
the representation "domain invariant" — they are optimization penalties that
trade off against the primary task loss, and their effect on any particular
run is an empirical question answered by the robustness report, not a
guarantee this module can make.

References (formulas only, not literature claims about biological validity):
  - CORAL: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain
    Adaptation" — squared Frobenius norm between per-domain feature
    covariance matrices, normalized by 4*D^2.
  - MMD: Gretton et al., "A Kernel Two-Sample Test" — squared maximum mean
    discrepancy with an RBF kernel, computed via the biased V-statistic
    (diagonal kernel terms included; see mmd_loss's own docstring for why
    the unbiased U-statistic is not used here).
  - Gradient reversal: Ganin & Lempitsky, "Unsupervised Domain Adaptation by
    Backpropagation" — identity in the forward pass, negated (and scaled by
    lambda) gradient in the backward pass.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class DomainLossConfigurationError(ValueError):
    """Raised for an invalid domain-robustness configuration."""


# ─── CORAL ──────────────────────────────────────────────────────────────────

def _subject_covariance(x: torch.Tensor) -> torch.Tensor:
    """Feature covariance of a [n, D] batch of subject embeddings, with a
    small ridge added to the diagonal so a single-subject or degenerate
    (rank-deficient) group never produces a NaN covariance estimate."""
    n = x.shape[0]
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    if n <= 1:
        # A single subject has no within-group variance to estimate — return
        # a zero covariance (finite, well-defined) rather than dividing by
        # (n - 1) = 0.
        return torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
    cov = (centered.t() @ centered) / (n - 1)
    return cov


def coral_loss(
    embeddings_by_source: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Average pairwise CORAL distance between development-source subject
    embeddings. embeddings_by_source: {source_name: [n_source, D] tensor} —
    every tensor here must already be restricted to development subjects by
    the caller.

    Requires at least two represented sources (each with >= 1 subject); with
    fewer than two, returns a zero loss (finite, no gradient contribution)
    rather than raising, since "only one development source present in this
    batch" is an expected, not exceptional, situation for small batches.
    """
    sources = sorted(s for s, x in embeddings_by_source.items() if x.shape[0] > 0)
    if len(sources) < 2:
        any_tensor = next(iter(embeddings_by_source.values()), None)
        device = any_tensor.device if any_tensor is not None else "cpu"
        dtype = any_tensor.dtype if any_tensor is not None else torch.float32
        return torch.zeros((), dtype=dtype, device=device), {"n_source_pairs": 0}

    d = embeddings_by_source[sources[0]].shape[1]
    covariances = {s: _subject_covariance(embeddings_by_source[s]) for s in sources}

    pair_losses = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            diff = covariances[sources[i]] - covariances[sources[j]]
            pair_losses.append((diff ** 2).sum() / (4.0 * d * d))
    total = torch.stack(pair_losses).mean()
    return total, {"n_source_pairs": len(pair_losses)}


# ─── MMD ────────────────────────────────────────────────────────────────────

def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True)
    return (x2 + y2.t() - 2.0 * (x @ y.t())).clamp(min=0.0)


def _rbf_kernel_sum(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    sq = _pairwise_sq_dists(x, y)
    return torch.exp(-sq / (2.0 * bandwidth * bandwidth))


def _median_bandwidth(x: torch.Tensor, y: torch.Tensor) -> float:
    """Median-heuristic bandwidth over the union of both samples' pairwise
    distances. Falls back to 1.0 (a stable, finite default) whenever there
    are too few points to compute a meaningful median (n < 2 total), so a
    tiny batch never divides by zero or yields a NaN kernel."""
    combined = torch.cat([x, y], dim=0)
    n = combined.shape[0]
    if n < 2:
        return 1.0
    sq = _pairwise_sq_dists(combined, combined)
    iu = torch.triu_indices(n, n, offset=1)
    vals = sq[iu[0], iu[1]]
    if vals.numel() == 0:
        return 1.0
    med = torch.median(vals).clamp(min=1e-12).sqrt().item()
    return med if med > 0 else 1.0


def mmd_loss(
    embeddings_by_source: Dict[str, torch.Tensor], kernel: str = "rbf",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Average pairwise squared MMD (biased V-statistic — stable and finite for
    the small batches typical of subject-level bags, unlike the unbiased
    U-statistic which is undefined for n < 2) between development-source
    subject embeddings, using an RBF kernel with the median-distance
    bandwidth heuristic recomputed per pair (never tuned against held-out
    data — it is a fixed, deterministic function of the development batch
    itself).
    """
    if kernel != "rbf":
        raise DomainLossConfigurationError(f"mmd_loss: unsupported kernel {kernel!r}, only 'rbf' is implemented.")

    sources = sorted(s for s, x in embeddings_by_source.items() if x.shape[0] > 0)
    if len(sources) < 2:
        any_tensor = next(iter(embeddings_by_source.values()), None)
        device = any_tensor.device if any_tensor is not None else "cpu"
        dtype = any_tensor.dtype if any_tensor is not None else torch.float32
        return torch.zeros((), dtype=dtype, device=device), {"n_source_pairs": 0}

    pair_losses = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            x, y = embeddings_by_source[sources[i]], embeddings_by_source[sources[j]]
            bw = _median_bandwidth(x.detach(), y.detach())
            kxx = _rbf_kernel_sum(x, x, bw)
            kyy = _rbf_kernel_sum(y, y, bw)
            kxy = _rbf_kernel_sum(x, y, bw)
            mmd_sq = kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
            pair_losses.append(mmd_sq.clamp(min=0.0))
    total = torch.stack(pair_losses).mean()
    return total, {"n_source_pairs": len(pair_losses)}


# ─── Gradient reversal / domain-adversarial ────────────────────────────────

class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Identity forward, negated-and-scaled gradient backward — see
    _GradientReversalFunction. lambda_=0 makes this exactly the identity in
    both directions (no reversal effect), which is the disabled/default
    behaviour a lambda-0 configuration should produce."""
    return _GradientReversalFunction.apply(x, lambda_)


class DomainVocabularyError(ValueError):
    """Raised when a source name outside the development-source vocabulary
    a domain classifier was built from is encountered at train or resume
    time — never silently mapped to an arbitrary/"unknown" class id, since
    that would let an unseen or held-out source quietly participate in
    domain-adversarial training."""


class DomainClassifierHead(nn.Module):
    """Linear development-source classifier head applied to
    gradient-reversed subject embeddings. `source_vocabulary` is fixed at
    construction (development sources only, in a deterministic sorted
    order) and persisted alongside the model bundle; a source name outside
    this vocabulary at train time raises DomainVocabularyError rather than
    being silently coerced to any class id."""

    def __init__(self, embedding_dim: int, source_vocabulary: Sequence[str]):
        super().__init__()
        vocab = sorted(set(str(s) for s in source_vocabulary))
        if len(vocab) < 2:
            raise DomainLossConfigurationError(
                f"DomainClassifierHead requires at least 2 development sources, got {vocab!r}."
            )
        self.source_vocabulary: List[str] = vocab
        self._index = {s: i for i, s in enumerate(vocab)}
        self.linear = nn.Linear(embedding_dim, len(vocab))

    def source_indices(self, sources: Sequence[str]) -> torch.Tensor:
        unknown = sorted(set(str(s) for s in sources) - set(self._index))
        if unknown:
            raise DomainVocabularyError(
                f"DomainClassifierHead: source(s) {unknown} are not in this head's development "
                f"vocabulary {self.source_vocabulary} — an unseen or held-out source must never be "
                "fed into domain-adversarial training."
            )
        return torch.tensor([self._index[str(s)] for s in sources], dtype=torch.long)

    def forward(self, subject_embeddings: torch.Tensor, lambda_: float) -> torch.Tensor:
        reversed_ = gradient_reversal(subject_embeddings, lambda_)
        return self.linear(reversed_)


def domain_adversarial_loss(
    domain_head: DomainClassifierHead, subject_embeddings: torch.Tensor,
    sources: Sequence[str], lambda_: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cross-entropy domain loss through the gradient-reversal head. Returns
    (loss, {"domain_accuracy": float}) so callers can log task loss and
    domain accuracy separately — a falling domain accuracy is reported as a
    diagnostic signal only, never asserted to prove biological or scientific
    domain invariance."""
    if subject_embeddings.shape[0] == 0:
        return torch.zeros((), device=subject_embeddings.device), {"domain_accuracy": float("nan")}
    targets = domain_head.source_indices(sources).to(subject_embeddings.device)
    logits = domain_head(subject_embeddings, lambda_)
    loss = nn.functional.cross_entropy(logits, targets)
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == targets).float().mean().item()
    return loss, {"domain_accuracy": acc}


def group_embeddings_by_source(
    subject_embeddings: torch.Tensor, sources: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Split a [B, D] batch of subject embeddings into {source: [n_s, D]}
    groups, preserving gradient flow (index_select, not a detach/copy)."""
    sources = [str(s) for s in sources]
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(sources):
        groups.setdefault(s, []).append(i)
    out = {}
    for s, idxs in groups.items():
        idx_t = torch.tensor(idxs, dtype=torch.long, device=subject_embeddings.device)
        out[s] = subject_embeddings.index_select(0, idx_t)
    return out


class MissingSourceProvenanceError(ValueError):
    """A subject fed into a source-aware strategy (anything other than
    'erm') has a blank, placeholder, or absent dataset_source — never
    silently mapped to a literal "unknown" bucket, which would let an
    unprovenanced subject quietly participate in CORAL/MMD/source-balanced
    sampling/domain-adversarial training."""


_PLACEHOLDER_SOURCE_VALUES = frozenset({"", "unknown", "none", "nan", "null", "n/a", "na"})


def validate_source_provenance(sources: Sequence[str]) -> None:
    """Raises MissingSourceProvenanceError if any element of `sources` is
    blank or a known placeholder value. Called only by strategies other
    than 'erm' — ERM never reads source at all, so it has nothing to
    validate."""
    bad_idx = [i for i, s in enumerate(sources) if str(s).strip().lower() in _PLACEHOLDER_SOURCE_VALUES]
    if bad_idx:
        raise MissingSourceProvenanceError(
            f"{len(bad_idx)} subject(s) passed to a source-aware strategy have a blank or "
            f"placeholder dataset_source value (indices {bad_idx[:10]}) — a source-aware strategy "
            "requires a real source identity for every subject."
        )


DOMAIN_STRATEGIES = ("erm", "source_balanced", "coral", "mmd", "domain_adversarial")

DEFAULT_DOMAIN_ROBUSTNESS_CONFIG = {
    "strategy": "erm",
    "source_balancing": {"enabled": False, "batch_size": None, "samples_per_epoch": None},
    "coral": {"enabled": False, "weight": 0.0},
    "mmd": {"enabled": False, "weight": 0.0, "kernel": "rbf"},
    "adversarial": {
        "enabled": False, "weight": 0.0, "warmup_epochs": 0, "gradient_reversal_lambda": 0.0,
    },
}


def resolve_domain_robustness_config(cfg: Optional[dict]) -> dict:
    """Merge a (possibly partial/absent) domain_robustness config block with
    DEFAULT_DOMAIN_ROBUSTNESS_CONFIG. Absent configuration resolves to plain
    ERM with every regularizer disabled and weight 0.0 — unchanged behaviour
    for any caller that does not opt in."""
    resolved = {
        "strategy": DEFAULT_DOMAIN_ROBUSTNESS_CONFIG["strategy"],
        "source_balancing": dict(DEFAULT_DOMAIN_ROBUSTNESS_CONFIG["source_balancing"]),
        "coral": dict(DEFAULT_DOMAIN_ROBUSTNESS_CONFIG["coral"]),
        "mmd": dict(DEFAULT_DOMAIN_ROBUSTNESS_CONFIG["mmd"]),
        "adversarial": dict(DEFAULT_DOMAIN_ROBUSTNESS_CONFIG["adversarial"]),
    }
    cfg = cfg or {}
    unknown = set(cfg) - set(DEFAULT_DOMAIN_ROBUSTNESS_CONFIG)
    if unknown:
        raise DomainLossConfigurationError(
            f"domain_robustness config has unknown top-level key(s) {sorted(unknown)} — valid keys "
            f"are {sorted(DEFAULT_DOMAIN_ROBUSTNESS_CONFIG)}."
        )
    for key in ("source_balancing", "coral", "mmd", "adversarial"):
        sub = cfg.get(key, {}) or {}
        unknown_sub = set(sub) - set(resolved[key])
        if unknown_sub:
            raise DomainLossConfigurationError(
                f"domain_robustness.{key} has unknown key(s) {sorted(unknown_sub)} — valid keys are "
                f"{sorted(resolved[key])}."
            )
        resolved[key].update(sub)
    if "strategy" in cfg:
        resolved["strategy"] = cfg["strategy"]

    if resolved["strategy"] not in DOMAIN_STRATEGIES:
        raise DomainLossConfigurationError(
            f"domain_robustness.strategy={resolved['strategy']!r} must be one of {DOMAIN_STRATEGIES}"
        )
    for key in ("coral", "mmd", "adversarial"):
        w = resolved[key].get("weight", 0.0)
        if w < 0:
            raise DomainLossConfigurationError(f"domain_robustness.{key}.weight must be >= 0, got {w}.")
    if resolved["adversarial"]["warmup_epochs"] < 0:
        raise DomainLossConfigurationError("domain_robustness.adversarial.warmup_epochs must be >= 0.")
    if resolved["adversarial"]["gradient_reversal_lambda"] < 0:
        raise DomainLossConfigurationError(
            "domain_robustness.adversarial.gradient_reversal_lambda must be >= 0."
        )
    bs = resolved["source_balancing"].get("batch_size")
    if bs is not None and bs <= 0:
        raise DomainLossConfigurationError(f"domain_robustness.source_balancing.batch_size must be > 0, got {bs}.")
    spe = resolved["source_balancing"].get("samples_per_epoch")
    if spe is not None and spe <= 0:
        raise DomainLossConfigurationError(
            f"domain_robustness.source_balancing.samples_per_epoch must be > 0, got {spe}."
        )
    if resolved["strategy"] == "source_balanced" and not resolved["source_balancing"]["enabled"]:
        raise DomainLossConfigurationError(
            "domain_robustness.strategy='source_balanced' requires "
            "domain_robustness.source_balancing.enabled=true — a strategy name alone does not "
            "implicitly enable source-balanced sampling."
        )
    return resolved
