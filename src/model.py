"""
model.py — MultiSmokeCancerNet
Shared encoder → dual heads → gated attention MIL → subject cancer probability.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from constants import N_CELL_TYPES, N_HVGS_DEFAULT, N_SMOKE_CLASSES, SMOKE_TYPES


# ─── DRY layer builder ────────────────────────────────────────────────────────

def _mlp(
    dims:            list,
    dropout:         float = 0.0,
    batch_norm:      bool  = False,
    last_activation: bool  = False,
) -> nn.Sequential:
    """
    Single source of truth for all FC stacks in this file.

    Non-final layers : Linear → (BN?) → GELU → (Drop?)
    Final layer      : Linear only — unless last_activation=True,
                       which adds BN+GELU without dropout (encoder pattern).
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if not is_last or last_activation:
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
            if not is_last and dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ─── Stage 2: Shared Cell Encoder ────────────────────────────────────────────

class CellEncoder(nn.Module):
    """
    Shared MLP — produces embedding z ∈ R^embedding_dim for both heads.
    All layers get BN+GELU; dropout only on hidden layers (not the final).
    """

    def __init__(
        self,
        input_dim:     int   = N_HVGS_DEFAULT,
        hidden_dims:   list  = [1024, 512],
        embedding_dim: int   = 256,
        dropout:       float = 0.3,
    ):
        super().__init__()
        self.net = _mlp(
            [input_dim] + hidden_dims + [embedding_dim],
            dropout        = dropout,
            batch_norm     = True,
            last_activation= True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                          # [B, embedding_dim]


# ─── Stage 3A: Smoke Type Head ────────────────────────────────────────────────

class SmokeTypeHead(nn.Module):
    """
    Classifies which smoke type damaged each cell.
    Returns logits — softmax applied downstream.
    6 classes: cigarette · vape · cigar · cannabis · dual-use · unexposed
    """

    def __init__(
        self,
        embedding_dim: int   = 256,
        num_classes:   int   = N_SMOKE_CLASSES,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.net = _mlp([embedding_dim, 128, num_classes], dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)                          # [B, num_classes]


# ─── Stage 3B: Malignancy Head ────────────────────────────────────────────────

class MalignancyHead(nn.Module):
    """
    Scores per-cell malignancy probability ∈ [0, 1].
    Sigmoid applied in forward — not buried in Sequential.
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = _mlp([embedding_dim, 128, 1], dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))           # [B, 1]


# ─── Stage 5+6: Gated Attention MIL ──────────────────────────────────────────

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL — Ilse et al., ICML 2018.

    Aggregates N cell embeddings into one subject vector, weighted by
    attention that focuses on the most cancer-relevant cells.

        gates_i = tanh(V·h_i)  ⊙  sigmoid(U·h_i)
        a_i     = softmax_N( w^T · gates_i )
        Z_sub   = Σ  a_i · z_i
        P(cancer) = σ( classifier(Z_sub) )

    Two gates prevent attention collapse vs single-gate attention.
    a_i is interpretable: shows which cells drove the cancer prediction.
    """

    def __init__(
        self,
        feat_dim:     int   = 267,     # z(256) + smoke(6) + risk(1) + type(4)
        embed_dim:    int   = 256,
        attention_dim:int   = 128,
        dropout:      float = 0.2,
    ):
        super().__init__()
        self.V          = nn.Linear(feat_dim, attention_dim)
        self.U          = nn.Linear(feat_dim, attention_dim)
        self.w          = nn.Linear(attention_dim, 1)
        self.classifier = _mlp([embed_dim, 64, 1], dropout=dropout)

    def forward(
        self,
        z_bag: torch.Tensor,    # [N, embed_dim]
        h_bag: torch.Tensor,    # [N, feat_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = torch.tanh(self.V(h_bag)) * torch.sigmoid(self.U(h_bag))  # [N, attn]
        a     = torch.softmax(self.w(gates), dim=0)                        # [N, 1]
        z_sub = (a * z_bag).sum(dim=0, keepdim=True)                      # [1, embed]
        return torch.sigmoid(self.classifier(z_sub)), a.squeeze(1)
        # → cancer_prob [1,1],  attention_weights [N]


# ─── Full Model ───────────────────────────────────────────────────────────────

class MultiSmokeCancerNet(nn.Module):
    """
    Novel unified pipeline — first model to simultaneously:
      1. Classify smoke type per cell (6 classes)
      2. Score malignancy risk per cell ∈ [0, 1]
      3. Aggregate N cells → subject cancer probability via gated attention MIL

    Two forward modes:
      forward_cell()    → Phase 1 (cell-level pre-training)
      forward_subject() → Phase 2 / 3 (MIL aggregation)
    """

    def __init__(
        self,
        input_dim:      int   = N_HVGS_DEFAULT,
        embedding_dim:  int   = 256,
        num_smoke:      int   = N_SMOKE_CLASSES,
        num_cell_types: int   = N_CELL_TYPES,
        encoder_dropout:float = 0.3,
        head_dropout:   float = 0.2,
        attention_dim:  int   = 128,
    ):
        super().__init__()
        self.embedding_dim  = embedding_dim
        self.num_smoke      = num_smoke
        self.num_cell_types = num_cell_types

        self.encoder         = CellEncoder(input_dim, [1024, 512], embedding_dim, encoder_dropout)
        self.smoke_head      = SmokeTypeHead(embedding_dim, num_smoke, head_dropout)
        self.malignancy_head = MalignancyHead(embedding_dim, head_dropout)
        self.aggregator      = GatedAttentionMIL(
            feat_dim      = embedding_dim + num_smoke + 1 + num_cell_types,
            embed_dim     = embedding_dim,
            attention_dim = attention_dim,
            dropout       = head_dropout,
        )

    @classmethod
    def from_config(cls, config: Union[dict, str, Path]) -> "MultiSmokeCancerNet":
        """Instantiate from a config dict or path to configs/default.yaml."""
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        c = config.get("model", config)
        return cls(
            input_dim      = c.get("input_dim",       N_HVGS_DEFAULT),
            embedding_dim  = c.get("embedding_dim",   256),
            num_smoke      = c.get("num_smoke_types", N_SMOKE_CLASSES),
            num_cell_types = c.get("num_cell_types",  N_CELL_TYPES),
            encoder_dropout= c.get("encoder_dropout", 0.3),
            head_dropout   = c.get("head_dropout",    0.2),
            attention_dim  = c.get("attention_dim",   128),
        )

    def forward_cell(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase 1 — batch of cells. Returns z, smoke_logits, malignancy."""
        z = self.encoder(x)
        return z, self.smoke_head(z), self.malignancy_head(z)

    def forward_subject(
        self,
        x_bag:         torch.Tensor,   # [N, input_dim]
        cell_type_ids: torch.Tensor,   # [N]  long
    ) -> Dict[str, torch.Tensor]:
        """Phase 2/3 — one subject (N cells). Returns dict with all outputs."""
        z            = self.encoder(x_bag)
        smoke_logits = self.smoke_head(z)
        smoke_probs  = F.softmax(smoke_logits, dim=1)
        malignancy   = self.malignancy_head(z)
        ct_onehot    = F.one_hot(cell_type_ids, self.num_cell_types).float()
        h_bag        = torch.cat([z, smoke_probs, malignancy, ct_onehot], dim=1)

        cancer_prob, attn = self.aggregator(z, h_bag)
        return {
            "cancer_probability": cancer_prob,    # [1, 1]
            "attention_weights":  attn,            # [N]
            "cell_smoke_probs":   smoke_probs,     # [N, 6]
            "cell_malignancy":    malignancy,       # [N, 1]
            "cell_embeddings":    z,               # [N, 256]
        }

    def forward(self, x_bag, cell_type_ids):
        return self.forward_subject(x_bag, cell_type_ids)


# ─── Multi-Task Loss ──────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Phase 1 : λ_smoke·L_CE(smoke)  +  λ_malig·L_BCE(malignancy)
    Phase 2 : L_BCE(subject)
    Phase 3 : λ_smoke·L_CE  +  λ_malig·L_BCE(cell)  +  λ_sub·L_BCE(subject)

    Private helpers _ls / _lm / _lsb keep the three public methods DRY.
    """

    def __init__(
        self,
        lambda_smoke:     float                   = 0.30,
        lambda_malignancy:float                   = 0.30,
        lambda_subject:   float                   = 0.40,
        smoke_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.λs  = lambda_smoke
        self.λm  = lambda_malignancy
        self.λsb = lambda_subject
        self.ce  = nn.CrossEntropyLoss(weight=smoke_class_weights)
        self.bce = nn.BCELoss()

    def _ls (self, logits, targets): return self.ce (logits, targets)
    def _lm (self, preds,  targets): return self.bce(preds.view(-1),  targets.float())
    def _lsb(self, prob,   target) : return self.bce(prob.view(-1),   target.view(-1).float())

    def cell_level_loss(
        self,
        smoke_logits: torch.Tensor, smoke_targets: torch.Tensor,
        malig_preds:  torch.Tensor, malig_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ls, lm = self._ls(smoke_logits, smoke_targets), self._lm(malig_preds, malig_targets)
        total  = self.λs * ls + self.λm * lm
        return total, {"total": total.item(), "smoke": ls.item(), "malignancy": lm.item()}

    def subject_level_loss(
        self,
        cancer_prob: torch.Tensor, cancer_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l = self._lsb(cancer_prob, cancer_target)
        return l, {"total": l.item(), "subject": l.item()}

    def end_to_end_loss(
        self,
        smoke_logits: torch.Tensor, smoke_targets: torch.Tensor,
        malig_preds:  torch.Tensor, malig_targets: torch.Tensor,
        cancer_prob:  torch.Tensor, cancer_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ls, lm, lsb = (
            self._ls (smoke_logits, smoke_targets),
            self._lm (malig_preds,  malig_targets),
            self._lsb(cancer_prob,  cancer_target),
        )
        total = self.λs * ls + self.λm * lm + self.λsb * lsb
        return total, {
            "total": total.item(), "smoke": ls.item(),
            "malignancy": lm.item(), "subject": lsb.item(),
        }


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    CFG  = Path(__file__).parents[1] / "configs" / "default.yaml"
    B, N = 32, 256                       # batch cells, cells per subject

    model = MultiSmokeCancerNet.from_config(CFG)
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1 — cell-level
    x            = torch.randn(B, N_HVGS_DEFAULT)
    z, logits, m = model.forward_cell(x)
    assert z.shape      == (B, 256),             f"z: {z.shape}"
    assert logits.shape == (B, N_SMOKE_CLASSES), f"logits: {logits.shape}"
    assert m.shape      == (B, 1),               f"malignancy: {m.shape}"
    print(f"forward_cell    ✓  z{tuple(z.shape)}  logits{tuple(logits.shape)}  malig{tuple(m.shape)}")

    # Phase 2/3 — subject-level
    x_bag  = torch.randn(N, N_HVGS_DEFAULT)
    ct_ids = torch.randint(0, N_CELL_TYPES, (N,))
    out    = model.forward_subject(x_bag, ct_ids)
    assert out["cancer_probability"].shape == (1, 1)
    assert out["attention_weights"].shape  == (N,)
    assert abs(out["attention_weights"].sum().item() - 1.0) < 1e-5, "attn must sum to 1"
    print(f"forward_subject ✓  P(cancer)={out['cancer_probability'].item():.4f}"
          f"  attn_sum={out['attention_weights'].sum().item():.6f}")

    # All three loss modes
    loss_fn  = MultiTaskLoss()
    smoke_t  = torch.randint(0, N_SMOKE_CLASSES, (B,))
    malig_t  = torch.randint(0, 2, (B,)).float()
    cancer_t = torch.tensor([1.0])

    l1, d1 = loss_fn.cell_level_loss(logits, smoke_t, m, malig_t)
    l2, d2 = loss_fn.subject_level_loss(out["cancer_probability"], cancer_t)
    l3, d3 = loss_fn.end_to_end_loss(logits, smoke_t, m, malig_t, out["cancer_probability"], cancer_t)

    assert l1.requires_grad and l2.requires_grad and l3.requires_grad, "losses must be differentiable"
    print(f"cell_level_loss ✓  {d1}")
    print(f"subject_loss    ✓  {d2}")
    print(f"end_to_end_loss ✓  {d3}")
    print("\n=== PASSED ===")