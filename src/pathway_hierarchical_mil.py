"""
pathway_hierarchical_mil.py — Pathway-Aware Hierarchical Multi-Instance
Network (development-name code identifier: ``pathway_hierarchical_mil``).

Research-candidate architecture (Phase 5). This module is additive: it does
not modify ``model.py``'s ``MultiSmokeCancerNet`` and existing baselines
continue to work unchanged whether or not this model is enabled.

Data flow (see ARCHITECTURE.md for full detail):

    preprocessed expression [G]  (fixed, artifact-ordered gene list)
        -> masked gene-to-module projection + residual gene projection
        -> per-cell embedding                              [D]
        -> cell-type-aware gated attention (subject/cell-type local)
        -> per-cell-type representation                      [C, D]
        -> cell-type gated attention (subject local)
        -> subject representation                             [D]
            -> smoke-type head       (multiclass logits)
            -> cancer-risk head      (single logit)
            -> domain/source head    (optional, diagnostic only)

Everything in this file operates on already-preprocessed, fixed-gene-order
expression tensors. It never fits, refits, or otherwise touches
preprocessing statistics — see ``GeneModuleCollection`` and
``PathwayCellEncoder`` docstrings for the gene-order/module-alignment
contract this depends on.

Interpretability note: attention weights produced anywhere in this module
are model diagnostics reflecting learned pooling behavior. They are not
causal explanations and do not, by themselves, establish a biomarker or a
biological mechanism.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

SYNTHETIC_MODULE_SOURCE = "synthetic_diagnostic_v1"
UNKNOWN_CELL_TYPE_BUCKET_NAME = "unknown"


# ─────────────────────────────────────────────────────────────────────────
# Gene module / pathway collection
# ─────────────────────────────────────────────────────────────────────────

class GeneModuleError(ValueError):
    """Raised for any gene-module configuration or alignment failure."""


@dataclass
class GeneModuleCollection:
    """
    Pathway/gene-module membership aligned to a fixed, ordered gene list.

    Attributes:
        module_names: unique module identifiers, in deterministic order.
        gene_names: the gene order this collection is aligned to — MUST be
            identical (same order) to the owning ``PreprocessingArtifact``'s
            ``gene_list``. This class does not read artifacts directly; the
            caller is responsible for passing the artifact's gene order in.
        membership_mask: bool tensor [len(module_names), len(gene_names)].
            membership_mask[m, g] is True iff module m includes gene g.
        source_name: human-readable provenance label (e.g. a GMT filename,
            or ``SYNTHETIC_MODULE_SOURCE`` for the deterministic test/
            diagnostic scheme).
        source_version: caller-supplied version string for the source file.
    """

    module_names: List[str]
    gene_names: List[str]
    membership_mask: torch.Tensor
    source_name: str
    source_version: str = "unspecified"

    def __post_init__(self) -> None:
        if len(set(self.module_names)) != len(self.module_names):
            raise GeneModuleError("GeneModuleCollection: module_names must be unique.")
        if self.membership_mask.shape != (len(self.module_names), len(self.gene_names)):
            raise GeneModuleError(
                "GeneModuleCollection: membership_mask shape "
                f"{tuple(self.membership_mask.shape)} does not match "
                f"(n_modules={len(self.module_names)}, n_genes={len(self.gene_names)})."
            )
        if self.membership_mask.dtype != torch.bool:
            self.membership_mask = self.membership_mask.bool()

    @property
    def n_modules(self) -> int:
        return len(self.module_names)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    def module_gene_counts(self) -> Dict[str, int]:
        counts = self.membership_mask.sum(dim=1).tolist()
        return dict(zip(self.module_names, (int(c) for c in counts)))

    def coverage_report(self) -> Dict[str, object]:
        """Fraction of genes claimed by at least one module, plus per-module
        gene counts — a development-time diagnostic, never used to alter
        module membership itself."""
        gene_covered = self.membership_mask.any(dim=0)
        return {
            "n_modules": self.n_modules,
            "n_genes": self.n_genes,
            "genes_covered": int(gene_covered.sum()),
            "gene_coverage_fraction": float(gene_covered.float().mean()) if self.n_genes else 0.0,
            "module_gene_counts": self.module_gene_counts(),
        }

    def fingerprint(self) -> str:
        """Deterministic content fingerprint covering gene order, module
        order, and full membership — changing any of these changes the
        fingerprint, which downstream code binds into checkpoint/bundle
        identity (see PathwayCellEncoder / bundle integration)."""
        payload = {
            "source_name": self.source_name,
            "source_version": self.source_version,
            "gene_names": list(self.gene_names),
            "module_names": list(self.module_names),
            "membership": self.membership_mask.to(torch.uint8).tolist(),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ── construction ───────────────────────────────────────────────────

    @classmethod
    def from_gmt(
        cls,
        path: Union[str, Path],
        gene_order: Sequence[str],
        source_version: str = "unspecified",
        minimum_genes_per_module: int = 3,
        minimum_module_coverage: float = 0.5,
        empty_module_policy: str = "error",
    ) -> "GeneModuleCollection":
        """
        Parse a GMT-like file: ``module_name<TAB>description<TAB>GENE1<TAB>GENE2...``

        Alignment rules:
          - Only genes present in ``gene_order`` are kept (unavailable genes
            in the file are silently ignored for membership, never leaked
            into preprocessing).
          - Duplicate genes within one module line are deduplicated
            deterministically (first occurrence order preserved).
          - A module whose *aligned* gene count falls below
            ``minimum_genes_per_module`` is either dropped (and reported) or
            raises, according to ``empty_module_policy`` ("error" | "drop").
          - Coverage (fraction of ``gene_order`` claimed by at least one
            retained module) is reported by the caller via
            ``coverage_report()``; ``minimum_module_coverage`` is advisory
            and enforced by the caller/config validation, not here, since
            what to do about low coverage is a policy decision made once
            per training run, not per file parse.
        """
        if empty_module_policy not in ("error", "drop"):
            raise GeneModuleError(
                f"GeneModuleCollection.from_gmt: empty_module_policy must be 'error' or "
                f"'drop', got {empty_module_policy!r}."
            )
        gene_order = list(gene_order)
        gene_index = {g: i for i, g in enumerate(gene_order)}
        path = Path(path)
        if not path.exists():
            raise GeneModuleError(f"GeneModuleCollection.from_gmt: file not found at {path}.")

        module_names: List[str] = []
        rows: List[torch.Tensor] = []
        dropped: List[str] = []
        with open(path) as f:
            for lineno, line in enumerate(f, start=1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                name, _description, *genes = parts
                if not name:
                    raise GeneModuleError(f"GeneModuleCollection.from_gmt: empty module name at line {lineno}.")
                if name in module_names:
                    raise GeneModuleError(
                        f"GeneModuleCollection.from_gmt: duplicate module name {name!r} at line {lineno}."
                    )
                seen = set()
                indices = []
                for g in genes:
                    if g in seen or g not in gene_index:
                        continue
                    seen.add(g)
                    indices.append(gene_index[g])
                if len(indices) < minimum_genes_per_module:
                    if empty_module_policy == "error":
                        raise GeneModuleError(
                            f"GeneModuleCollection.from_gmt: module {name!r} has only "
                            f"{len(indices)} genes present in the artifact gene list "
                            f"(minimum_genes_per_module={minimum_genes_per_module}). Supply a "
                            "richer module file, lower the threshold, or set "
                            "empty_module_policy='drop'."
                        )
                    dropped.append(name)
                    continue
                row = torch.zeros(len(gene_order), dtype=torch.bool)
                row[indices] = True
                module_names.append(name)
                rows.append(row)

        if not module_names:
            raise GeneModuleError(
                f"GeneModuleCollection.from_gmt: no usable modules remained after alignment to "
                f"{len(gene_order)} artifact genes (source={path}). dropped={dropped!r}"
            )
        mask = torch.stack(rows, dim=0)
        return cls(
            module_names=module_names,
            gene_names=gene_order,
            membership_mask=mask,
            source_name=str(path),
            source_version=source_version,
        )

    @classmethod
    def synthetic(
        cls,
        gene_order: Sequence[str],
        n_modules: int = 4,
        genes_per_module: int = 8,
        seed: int = 0,
    ) -> "GeneModuleCollection":
        """
        Deterministic, clearly-labelled diagnostic module scheme for tests
        and synthetic workflows ONLY — never used unless the caller
        explicitly requests it (config path=null + synthetic/test mode).
        Contains no participant data: membership is derived purely from
        gene position via a fixed seeded permutation.
        """
        gene_order = list(gene_order)
        n_genes = len(gene_order)
        if n_modules < 1 or genes_per_module < 1:
            raise GeneModuleError("GeneModuleCollection.synthetic: n_modules and genes_per_module must be >= 1.")
        if n_genes == 0:
            raise GeneModuleError("GeneModuleCollection.synthetic: gene_order must be non-empty.")
        generator = torch.Generator().manual_seed(seed)
        module_names = [f"synthetic_module_{i:03d}" for i in range(n_modules)]
        mask = torch.zeros(n_modules, n_genes, dtype=torch.bool)
        for m in range(n_modules):
            k = min(genes_per_module, n_genes)
            idx = torch.randperm(n_genes, generator=generator)[:k]
            mask[m, idx] = True
        return cls(
            module_names=module_names,
            gene_names=gene_order,
            membership_mask=mask,
            source_name=SYNTHETIC_MODULE_SOURCE,
            source_version=f"seed={seed}",
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        payload = {
            "module_names": self.module_names,
            "gene_names": self.gene_names,
            "membership": self.membership_mask.to(torch.uint8).tolist(),
            "source_name": self.source_name,
            "source_version": self.source_version,
            "fingerprint": self.fingerprint(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GeneModuleCollection":
        path = Path(path)
        payload = json.loads(path.read_text())
        mask = torch.tensor(payload["membership"], dtype=torch.bool)
        obj = cls(
            module_names=payload["module_names"],
            gene_names=payload["gene_names"],
            membership_mask=mask,
            source_name=payload["source_name"],
            source_version=payload["source_version"],
        )
        if obj.fingerprint() != payload.get("fingerprint"):
            raise GeneModuleError(f"GeneModuleCollection.load: fingerprint mismatch — {path} appears corrupted.")
        return obj


def is_synthetic_module_source(source_name: str) -> bool:
    return source_name == SYNTHETIC_MODULE_SOURCE


# ─────────────────────────────────────────────────────────────────────────
# Pathway-aware cell encoder
# ─────────────────────────────────────────────────────────────────────────

class MaskedModuleProjection(nn.Module):
    """
    Linear gene -> module projection whose connectivity is restricted to
    ``membership_mask``. Disallowed (gene, module) connections are exactly
    zero on every forward pass, and never receive gradient (a backward hook
    zeroes their gradient before any optimizer step sees it) — see
    ``effective_weight`` below.
    """

    def __init__(self, membership_mask: torch.Tensor):
        super().__init__()
        n_modules, n_genes = membership_mask.shape
        self.register_buffer("membership_mask", membership_mask.bool())
        weight = torch.empty(n_modules, n_genes)
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
        weight = weight * self.membership_mask  # zero disallowed entries at init
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(n_modules))
        self.weight.register_hook(lambda grad: grad * self.membership_mask.to(grad.dtype))

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.membership_mask.to(self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


class PathwayCellEncoder(nn.Module):
    """
    x in R^G  ->  masked gene-to-module projection  ->  module activations
    p in R^P  ->  optional residual gene projection  ->  fused cell
    embedding h in R^D.

    Uses LayerNorm (sample-local) rather than BatchNorm1d, deliberately —
    BatchNorm1d would mix statistics across cells that may belong to
    different subjects/sources within one batch, which is unsafe for this
    architecture's subject-local pooling guarantees.
    """

    def __init__(
        self,
        modules: GeneModuleCollection,
        embedding_dim: int = 128,
        residual_gene_dim: int = 64,
        use_gene_residual: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        if modules.n_modules == 0:
            raise GeneModuleError("PathwayCellEncoder: module collection has zero modules.")
        self.gene_names = list(modules.gene_names)
        self.module_names = list(modules.module_names)
        self.module_fingerprint = modules.fingerprint()
        self.input_dim = modules.n_genes
        self.pathway_dim = modules.n_modules
        self.use_gene_residual = use_gene_residual
        self.embedding_dim = embedding_dim

        self.module_projection = MaskedModuleProjection(modules.membership_mask)
        self.module_norm = nn.LayerNorm(self.pathway_dim)
        self.module_dropout = nn.Dropout(dropout)

        fusion_in = self.pathway_dim
        if use_gene_residual:
            self.residual_projection = nn.Sequential(
                nn.Linear(self.input_dim, residual_gene_dim),
                nn.LayerNorm(residual_gene_dim),
                nn.GELU(),
            )
            fusion_in += residual_gene_dim
        else:
            self.residual_projection = None

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(x).all():
            raise ValueError("PathwayCellEncoder: input expression contains non-finite values.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"PathwayCellEncoder: input last dim {x.shape[-1]} does not match the module "
                f"collection's gene count {self.input_dim}."
            )
        module_features = F.gelu(self.module_projection(x))
        module_features = self.module_dropout(self.module_norm(module_features))
        parts = [module_features]
        if self.use_gene_residual:
            parts.append(self.residual_projection(x))
        fused = torch.cat(parts, dim=-1)
        return self.fusion(fused)


# ─────────────────────────────────────────────────────────────────────────
# Masked attention utilities
# ─────────────────────────────────────────────────────────────────────────

class GatedAttentionScorer(nn.Module):
    """a_i = w^T( tanh(V h_i) * sigmoid(U h_i) ) — raw (pre-softmax) score."""

    def __init__(self, dim: int, attention_dim: int):
        super().__init__()
        self.V = nn.Linear(dim, attention_dim)
        self.U = nn.Linear(dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        gates = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        return self.w(gates).squeeze(-1)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Softmax restricted to ``mask`` (True = included). Positions/groups that
    are entirely masked out produce exactly zero weight everywhere (instead
    of NaN from softmax-of-all-(-inf)).
    """
    neg_inf = torch.finfo(scores.dtype).min
    filled = scores.masked_fill(~mask, neg_inf)
    weights = torch.softmax(filled, dim=dim)
    all_masked = (~mask).all(dim=dim, keepdim=True)
    weights = torch.where(all_masked, torch.zeros_like(weights), weights)
    return weights


# ─────────────────────────────────────────────────────────────────────────
# Cell-type-aware hierarchical pooling
# ─────────────────────────────────────────────────────────────────────────

class HierarchicalAttentionPooling(nn.Module):
    """
    Level 1 (cells -> cell-type representation): gated attention, masked
    softmax computed independently per (subject, cell-type) group over the
    cell dimension. Padded cells and cells of a different type receive
    exactly zero weight; weights sum to one within every non-empty
    subject/cell-type group.

    Level 2 (cell-types -> subject representation): gated attention over
    the (small, fixed) cell-type axis, masked softmax restricted to
    cell-types actually observed for that subject.

    ``num_cell_type_buckets`` includes one bucket for cells whose type is
    unknown (index ``num_cell_type_buckets - 1``, by convention) — an
    explicit policy rather than dropping unknown-type cells silently.
    """

    def __init__(self, embedding_dim: int, num_cell_type_buckets: int, attention_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_cell_type_buckets = num_cell_type_buckets
        self.cell_scorer = GatedAttentionScorer(embedding_dim, attention_dim)
        self.cell_type_scorer = GatedAttentionScorer(embedding_dim, attention_dim)

    def forward(
        self,
        cell_embeddings: torch.Tensor,   # [B, N, D]
        cell_type_ids: torch.Tensor,     # [B, N]  long, in [0, C)
        cell_mask: torch.Tensor,         # [B, N]  bool, True = real cell
    ):
        B, N, D = cell_embeddings.shape
        C = self.num_cell_type_buckets
        raw_scores = self.cell_scorer(cell_embeddings)  # [B, N]

        cell_attention = torch.zeros(B, C, N, dtype=cell_embeddings.dtype, device=cell_embeddings.device)
        cell_type_repr = torch.zeros(B, C, D, dtype=cell_embeddings.dtype, device=cell_embeddings.device)
        cell_type_present = torch.zeros(B, C, dtype=torch.bool, device=cell_embeddings.device)

        for c in range(C):
            type_mask = cell_mask & (cell_type_ids == c)          # [B, N]
            weights_c = masked_softmax(raw_scores, type_mask, dim=1)  # [B, N]
            cell_attention[:, c, :] = weights_c
            cell_type_repr[:, c, :] = torch.einsum("bn,bnd->bd", weights_c, cell_embeddings)
            cell_type_present[:, c] = type_mask.any(dim=1)

        ct_scores = self.cell_type_scorer(cell_type_repr)  # [B, C]
        cell_type_attention = masked_softmax(ct_scores, cell_type_present, dim=1)  # [B, C]
        subject_embeddings = torch.einsum("bc,bcd->bd", cell_type_attention, cell_type_repr)
        valid_subject_mask = cell_type_present.any(dim=1)

        return {
            "subject_embeddings": subject_embeddings,           # [B, D]
            "cell_attention": cell_attention,                   # [B, C, N]
            "cell_type_attention": cell_type_attention,         # [B, C]
            "cell_type_present": cell_type_present,              # [B, C]
            "valid_subject_mask": valid_subject_mask,             # [B]
        }


# ─────────────────────────────────────────────────────────────────────────
# Model output contract
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class HierarchicalMILOutput:
    smoke_logits: torch.Tensor            # [B, num_smoke]
    cancer_logits: torch.Tensor           # [B]
    subject_embeddings: torch.Tensor      # [B, D]
    valid_subject_mask: torch.Tensor      # [B] bool
    cell_type_attention: Optional[torch.Tensor] = None   # [B, C]
    cell_attention: Optional[torch.Tensor] = None         # [B, C, N]
    pathway_activations: Optional[torch.Tensor] = None    # [B, N, P] or None
    domain_logits: Optional[torch.Tensor] = None           # [B, n_sources] diagnostic only


# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PathwayHierarchicalMILConfig:
    enabled: bool = False
    embedding_dim: int = 128
    pathway_dim: int = 128  # informational; actual pathway_dim is len(modules)
    residual_gene_dim: int = 64
    attention_dim: int = 64
    dropout: float = 0.2
    use_gene_residual: bool = True
    use_cell_type_embedding: bool = True
    cell_type_embedding_dim: int = 16
    use_source_embedding: bool = False
    source_embedding_dim: int = 8
    use_species_embedding: bool = False
    species_embedding_dim: int = 4
    smoke_loss_weight: float = 1.0
    cancer_loss_weight: float = 1.0
    return_attention_during_training: bool = False
    num_cell_type_buckets: int = 5  # 4 known coarse types + 1 unknown, matches constants.N_CELL_TYPES + 1
    num_sources: int = 1
    num_species: int = 2
    empty_batch_policy: str = "skip"  # "skip" | "error" — see MultitaskMaskedLoss

    def validate(self) -> None:
        if self.embedding_dim <= 0 or self.attention_dim <= 0 or self.residual_gene_dim <= 0:
            raise ValueError("PathwayHierarchicalMILConfig: embedding/attention/residual dims must be positive.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("PathwayHierarchicalMILConfig: dropout must be in [0, 1).")
        if self.smoke_loss_weight < 0 or self.cancer_loss_weight < 0:
            raise ValueError("PathwayHierarchicalMILConfig: loss weights must be non-negative.")
        if self.num_cell_type_buckets < 1:
            raise ValueError("PathwayHierarchicalMILConfig: num_cell_type_buckets must be >= 1.")
        if self.empty_batch_policy not in ("skip", "error"):
            raise ValueError("PathwayHierarchicalMILConfig: empty_batch_policy must be 'skip' or 'error'.")

    @classmethod
    def from_dict(cls, d: Dict) -> "PathwayHierarchicalMILConfig":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        cfg = cls(**filtered)
        cfg.validate()
        return cfg


# ─────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────

MODEL_TYPE_NAME = "pathway_hierarchical_mil"


class PathwayHierarchicalMIL(nn.Module):
    """
    Pathway-aware, cell-type-aware, hierarchical multi-instance model for
    subject-level smoking-type classification and cancer-risk prediction.

    Research candidate architecture — see module docstring and
    ARCHITECTURE.md for the full data-flow and masking contract. Not
    described here or anywhere in this repository as clinically validated
    or superior to existing baselines.
    """

    def __init__(
        self,
        modules: GeneModuleCollection,
        config: PathwayHierarchicalMILConfig,
        num_smoke: int,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.model_type = MODEL_TYPE_NAME
        self.module_fingerprint = modules.fingerprint()
        self.gene_names = list(modules.gene_names)
        self.input_dim = modules.n_genes           # mirrors MultiSmokeCancerNet.input_dim for bundle checks
        self.num_smoke = num_smoke
        self.num_cell_type_buckets = config.num_cell_type_buckets

        self.encoder = PathwayCellEncoder(
            modules=modules,
            embedding_dim=config.embedding_dim,
            residual_gene_dim=config.residual_gene_dim,
            use_gene_residual=config.use_gene_residual,
            dropout=config.dropout,
        )

        cond_dim = config.embedding_dim
        if config.use_cell_type_embedding:
            self.cell_type_embedding = nn.Embedding(config.num_cell_type_buckets, config.cell_type_embedding_dim)
            cond_dim += config.cell_type_embedding_dim
        else:
            self.cell_type_embedding = None
        if config.use_source_embedding:
            self.source_embedding = nn.Embedding(config.num_sources, config.source_embedding_dim)
            cond_dim += config.source_embedding_dim
        else:
            self.source_embedding = None
        if config.use_species_embedding:
            self.species_embedding = nn.Embedding(config.num_species, config.species_embedding_dim)
            cond_dim += config.species_embedding_dim
        else:
            self.species_embedding = None

        if cond_dim != config.embedding_dim:
            self.conditioning_projection = nn.Sequential(
                nn.Linear(cond_dim, config.embedding_dim),
                nn.LayerNorm(config.embedding_dim),
                nn.GELU(),
            )
        else:
            self.conditioning_projection = None

        self.pooling = HierarchicalAttentionPooling(
            embedding_dim=config.embedding_dim,
            num_cell_type_buckets=config.num_cell_type_buckets,
            attention_dim=config.attention_dim,
        )

        self.smoke_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim // 2, num_smoke),
        )
        self.cancer_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim // 2, 1),
        )
        self.domain_head: Optional[nn.Module]
        if config.num_sources > 1:
            self.domain_head = nn.Linear(config.embedding_dim, config.num_sources)
        else:
            self.domain_head = None

    def _condition(
        self,
        cell_embeddings: torch.Tensor,
        cell_type_ids: torch.Tensor,
        source_ids: Optional[torch.Tensor],
        species_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts = [cell_embeddings]
        if self.cell_type_embedding is not None:
            parts.append(self.cell_type_embedding(cell_type_ids))
        if self.source_embedding is not None:
            if source_ids is None:
                raise ValueError("PathwayHierarchicalMIL: source_embedding enabled but source_ids not provided.")
            src = source_ids if source_ids.dim() == cell_type_ids.dim() else source_ids.unsqueeze(1).expand_as(cell_type_ids)
            parts.append(self.source_embedding(src))
        if self.species_embedding is not None:
            if species_ids is None:
                raise ValueError("PathwayHierarchicalMIL: species_embedding enabled but species_ids not provided.")
            spc = species_ids if species_ids.dim() == cell_type_ids.dim() else species_ids.unsqueeze(1).expand_as(cell_type_ids)
            parts.append(self.species_embedding(spc))
        fused = torch.cat(parts, dim=-1)
        if self.conditioning_projection is not None:
            return self.conditioning_projection(fused)
        return fused

    def forward(
        self,
        expression: torch.Tensor,          # [B, N, G]
        cell_type_ids: torch.Tensor,       # [B, N] long
        cell_mask: torch.Tensor,           # [B, N] bool
        source_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        return_attention: Optional[bool] = None,
    ) -> HierarchicalMILOutput:
        if expression.dim() != 3:
            raise ValueError(f"PathwayHierarchicalMIL: expression must be [B, N, G], got {tuple(expression.shape)}.")
        if cell_mask.sum(dim=1).eq(0).any():
            raise ValueError("PathwayHierarchicalMIL: at least one subject bag has zero real cells.")

        return_attention = self.config.return_attention_during_training if return_attention is None else return_attention

        B, N, G = expression.shape
        flat = expression.reshape(B * N, G)
        cell_embeddings = self.encoder(flat).reshape(B, N, -1)
        cell_embeddings = self._condition(cell_embeddings, cell_type_ids, source_ids, species_ids)

        # Padded cells must never influence pooling: zero them defensively in
        # addition to the mask consumed by masked_softmax, so that any
        # accidental non-attention aggregation downstream stays safe too.
        cell_embeddings = cell_embeddings * cell_mask.unsqueeze(-1).to(cell_embeddings.dtype)

        pooled = self.pooling(cell_embeddings, cell_type_ids, cell_mask)
        subject_embeddings = pooled["subject_embeddings"]

        smoke_logits = self.smoke_head(subject_embeddings)
        cancer_logits = self.cancer_head(subject_embeddings).squeeze(-1)
        domain_logits = self.domain_head(subject_embeddings) if self.domain_head is not None else None

        return HierarchicalMILOutput(
            smoke_logits=smoke_logits,
            cancer_logits=cancer_logits,
            subject_embeddings=subject_embeddings,
            valid_subject_mask=pooled["valid_subject_mask"],
            cell_type_attention=pooled["cell_type_attention"] if return_attention else None,
            cell_attention=pooled["cell_attention"] if return_attention else None,
            pathway_activations=None,
            domain_logits=domain_logits,
        )

    @classmethod
    def from_config(
        cls,
        modules: GeneModuleCollection,
        config: Union[Dict, PathwayHierarchicalMILConfig],
        num_smoke: int,
    ) -> "PathwayHierarchicalMIL":
        if isinstance(config, dict):
            config = PathwayHierarchicalMILConfig.from_dict(config)
        return cls(modules=modules, config=config, num_smoke=num_smoke)


# ─────────────────────────────────────────────────────────────────────────
# Multitask masked loss
# ─────────────────────────────────────────────────────────────────────────

class EmptyBatchLossError(RuntimeError):
    """Raised when empty_batch_policy='error' and a batch has no known
    label for either task."""


class MultitaskMaskedLoss(nn.Module):
    """
    total_loss = smoke_loss_weight * masked_smoke_loss
               + cancer_loss_weight * masked_cancer_loss

    Masking is applied before reduction for both tasks; unknown-label
    examples never contribute to either loss term. Class weights (if
    supplied) are applied exactly once, via ``nn.CrossEntropyLoss(weight=)``
    for the smoke task. A batch with zero known smoke labels contributes
    zero smoke loss (differentiable zero); same for cancer. If BOTH tasks
    have zero known labels in a batch, behavior is governed by
    ``empty_batch_policy``: "skip" (default) returns a differentiable zero
    total loss; "error" raises ``EmptyBatchLossError``.
    """

    def __init__(
        self,
        smoke_loss_weight: float = 1.0,
        cancer_loss_weight: float = 1.0,
        smoke_class_weights: Optional[torch.Tensor] = None,
        empty_batch_policy: str = "skip",
    ):
        super().__init__()
        if empty_batch_policy not in ("skip", "error"):
            raise ValueError("MultitaskMaskedLoss: empty_batch_policy must be 'skip' or 'error'.")
        self.smoke_loss_weight = smoke_loss_weight
        self.cancer_loss_weight = cancer_loss_weight
        self.smoke_class_weights = smoke_class_weights
        self.empty_batch_policy = empty_batch_policy

    def forward(
        self,
        smoke_logits: torch.Tensor,     # [B, num_smoke]
        smoke_targets: torch.Tensor,    # [B] long
        smoke_known: torch.Tensor,      # [B] bool
        cancer_logits: torch.Tensor,    # [B]
        cancer_targets: torch.Tensor,   # [B] float
        cancer_known: torch.Tensor,     # [B] bool
    ):
        smoke_known = smoke_known.bool()
        cancer_known = cancer_known.bool()
        n_smoke_known = int(smoke_known.sum())
        n_cancer_known = int(cancer_known.sum())

        if n_smoke_known == 0 and n_cancer_known == 0:
            if self.empty_batch_policy == "error":
                raise EmptyBatchLossError(
                    "MultitaskMaskedLoss: batch has zero known labels for both smoke and cancer "
                    "tasks and empty_batch_policy='error'."
                )
            zero = smoke_logits.sum() * 0.0 + cancer_logits.sum() * 0.0
            return zero, {
                "total": 0.0, "smoke": 0.0, "cancer": 0.0,
                "n_smoke_known": 0, "n_cancer_known": 0,
            }

        if n_smoke_known == 0:
            smoke_loss = smoke_logits.sum() * 0.0
        else:
            weight = self.smoke_class_weights
            if weight is not None:
                weight = weight.to(device=smoke_logits.device, dtype=smoke_logits.dtype)
            smoke_loss = F.cross_entropy(
                smoke_logits[smoke_known], smoke_targets[smoke_known], weight=weight
            )

        if n_cancer_known == 0:
            cancer_loss = cancer_logits.sum() * 0.0
        else:
            cancer_loss = F.binary_cross_entropy_with_logits(
                cancer_logits[cancer_known], cancer_targets[cancer_known].float()
            )

        total = self.smoke_loss_weight * smoke_loss + self.cancer_loss_weight * cancer_loss
        return total, {
            "total": total.item(),
            "smoke": smoke_loss.item(),
            "cancer": cancer_loss.item(),
            "n_smoke_known": n_smoke_known,
            "n_cancer_known": n_cancer_known,
        }


# ─────────────────────────────────────────────────────────────────────────
# Uncertainty diagnostics
# ─────────────────────────────────────────────────────────────────────────

def categorical_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Predictive entropy of a categorical distribution, per row. probs: [B, K]."""
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def binary_entropy(prob: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Binary entropy in nats. prob: [B], values in [0, 1]."""
    p = prob.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log())


@torch.no_grad()
def mc_dropout_predict(
    model: PathwayHierarchicalMIL,
    forward_kwargs: Dict,
    n_passes: int = 20,
) -> Dict[str, torch.Tensor]:
    """
    Development-only diagnostic. Runs ``n_passes`` stochastic forward passes
    with dropout layers active while every other module (LayerNorm, etc.)
    remains in eval mode, and returns mean/variance/entropy summaries. This
    is NOT a calibrated confidence interval — it is a diagnostic dispersion
    measure only, disabled unless explicitly requested by the caller.
    """
    was_training = model.training
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    cancer_probs = []
    smoke_probs = []
    try:
        for _ in range(n_passes):
            out = model(**forward_kwargs)
            cancer_probs.append(torch.sigmoid(out.cancer_logits))
            smoke_probs.append(F.softmax(out.smoke_logits, dim=-1))
    finally:
        model.train(was_training)

    cancer_stack = torch.stack(cancer_probs, dim=0)  # [passes, B]
    smoke_stack = torch.stack(smoke_probs, dim=0)     # [passes, B, K]
    cancer_mean = cancer_stack.mean(dim=0)
    return {
        "cancer_mean": cancer_mean,
        "cancer_variance": cancer_stack.var(dim=0, unbiased=False),
        "cancer_entropy": binary_entropy(cancer_mean),
        "smoke_mean": smoke_stack.mean(dim=0),
        "smoke_variance": smoke_stack.var(dim=0, unbiased=False),
        "smoke_entropy": categorical_entropy(smoke_stack.mean(dim=0)),
        "n_passes": n_passes,
    }


# ─────────────────────────────────────────────────────────────────────────
# Batch collation for variable-sized subject bags
# ─────────────────────────────────────────────────────────────────────────

def collate_subject_bags(bags: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pad a list of per-subject bags (each a dict with variable-length
    ``expression`` [n_i, G], ``cell_type_ids`` [n_i], and scalar
    ``smoke_label``/``smoke_known``/``cancer_label``/``cancer_known``) into
    one padded batch. Padded cells get ``cell_mask=False`` and contribute
    exactly zero attention weight (enforced in the pooling module, not
    merely by convention here). Raises on an empty bag — an explicit,
    early failure rather than a silently-degenerate subject.
    """
    if not bags:
        raise ValueError("collate_subject_bags: received an empty list of bags.")
    for i, bag in enumerate(bags):
        if bag["expression"].shape[0] == 0:
            raise ValueError(f"collate_subject_bags: bag[{i}] has zero cells.")

    n_max = max(bag["expression"].shape[0] for bag in bags)
    n_genes = bags[0]["expression"].shape[1]
    B = len(bags)

    expression = torch.zeros(B, n_max, n_genes, dtype=bags[0]["expression"].dtype)
    cell_type_ids = torch.zeros(B, n_max, dtype=torch.long)
    cell_mask = torch.zeros(B, n_max, dtype=torch.bool)

    for i, bag in enumerate(bags):
        n = bag["expression"].shape[0]
        expression[i, :n] = bag["expression"]
        cell_type_ids[i, :n] = bag["cell_type_ids"]
        cell_mask[i, :n] = True

    out = {
        "expression": expression,
        "cell_type_ids": cell_type_ids,
        "cell_mask": cell_mask,
        "smoke_label": torch.stack([torch.as_tensor(bag["smoke_label"]) for bag in bags]),
        "smoke_known": torch.stack([torch.as_tensor(bag["smoke_known"]) for bag in bags]).bool(),
        "cancer_label": torch.stack([torch.as_tensor(bag["cancer_label"]) for bag in bags]).float(),
        "cancer_known": torch.stack([torch.as_tensor(bag["cancer_known"]) for bag in bags]).bool(),
    }
    if "source_id" in bags[0]:
        out["source_ids"] = torch.stack([torch.as_tensor(bag["source_id"]) for bag in bags])
    if "species_id" in bags[0]:
        out["species_ids"] = torch.stack([torch.as_tensor(bag["species_id"]) for bag in bags])
    return out
