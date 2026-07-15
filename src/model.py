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

from constants import N_CELL_TYPES, N_HVGS_DEFAULT, N_SMOKE_CLASSES, SMOKE_TYPES, DOSE_UNKNOWN


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


# ─── Stage 3C: Dose-Response Head ─────────────────────────────────────────────

class DoseResponseHead(nn.Module):
    """
    Regresses normalised smoke exposure duration/dose (0-1) per cell.

    Novel: every published smoke-cell classifier treats exposure as
    categorical (smoker/never-smoker or smoke-type only). This head is the
    first to model exposure as a continuous dose, so MultiTaskLoss can
    enforce a monotonic dose -> malignancy relationship (see
    `MultiTaskLoss.dose_response_loss`) — i.e. cells exposed longer must
    score at least as malignant as cells exposed for less time, within the
    same smoke type.

    Status: no wired data source currently supplies a real per-cell exposure
    duration — the previously-cited "Loiselle 2018 / GSE130148" dataset does
    not exist (verified against GEO directly; GSE130148 is an unrelated lung
    scRNA-seq study). Every cell is therefore stamped DOSE_UNKNOWN today and
    this head trains on zero real signal (see dose_response_loss's masking).
    The architecture and loss are real and tested; only the labelled data is
    missing. Wiring a real source (e.g. joint-year exposure categories from
    GSE307690/CANUCK, if a future release exposes them per-sample) would
    activate it with no further code changes.
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = _mlp([embedding_dim, 64, 1], dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))           # [B, 1]  in [0, 1]


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


# ─── MIL pooling ablations ────────────────────────────────────────────────────
# Mean/max pooling share GatedAttentionMIL's forward(z_bag, h_bag) -> (prob, attn)
# interface (attn is None where there is no meaningful per-cell weight) so
# MultiSmokeCancerNet can swap pooling strategy without touching forward_subject.
# Used by src/benchmarks to ablate whether gated attention actually earns its
# extra parameters over simple pooling on identical encoder features.

class MeanPoolingMIL(nn.Module):
    def __init__(self, feat_dim: int = 267, embed_dim: int = 256,
                 attention_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.classifier = _mlp([embed_dim, 64, 1], dropout=dropout)

    def forward(self, z_bag: torch.Tensor, h_bag: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z_sub = z_bag.mean(dim=0, keepdim=True)
        return torch.sigmoid(self.classifier(z_sub)), None


class MaxPoolingMIL(nn.Module):
    def __init__(self, feat_dim: int = 267, embed_dim: int = 256,
                 attention_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.classifier = _mlp([embed_dim, 64, 1], dropout=dropout)

    def forward(self, z_bag: torch.Tensor, h_bag: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z_sub, _ = z_bag.max(dim=0, keepdim=True)
        return torch.sigmoid(self.classifier(z_sub)), None


MIL_POOLINGS = {
    "attention": GatedAttentionMIL,
    "mean":      MeanPoolingMIL,
    "max":       MaxPoolingMIL,
}


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
        pooling:        str   = "attention",
    ):
        super().__init__()
        if pooling not in MIL_POOLINGS:
            raise ValueError(f"pooling={pooling!r} must be one of {sorted(MIL_POOLINGS)}")
        self.input_dim      = input_dim
        self.embedding_dim  = embedding_dim
        self.num_smoke      = num_smoke
        self.num_cell_types = num_cell_types
        self.pooling        = pooling

        self.encoder         = CellEncoder(input_dim, [1024, 512], embedding_dim, encoder_dropout)
        self.smoke_head      = SmokeTypeHead(embedding_dim, num_smoke, head_dropout)
        self.malignancy_head = MalignancyHead(embedding_dim, head_dropout)
        self.dose_head       = DoseResponseHead(embedding_dim, head_dropout)
        self.aggregator      = MIL_POOLINGS[pooling](
            feat_dim      = embedding_dim + num_smoke + 1 + num_cell_types,
            embed_dim     = embedding_dim,
            attention_dim = attention_dim,
            dropout       = head_dropout,
        )

    @classmethod
    def from_config(
        cls,
        config: Union[dict, str, Path],
        num_smoke_types: Optional[int] = None,
        pooling: Optional[str] = None,
    ) -> "MultiSmokeCancerNet":
        """
        Instantiate from a config dict or path to configs/default.yaml.

        num_smoke_types, if given, OVERRIDES config['model']['num_smoke_types']
        — used when the actual effective smoke-label space (K) is only known
        at load time (data/label_mapping.py::EffectiveLabelMapping.k, read
        from a checkpoint's or artifact's persisted mapping), since a config
        file has no way to know a rare-class policy already shrank the
        output space. Without an override, the config value (or the fixed
        6-class default) is used — the no-merge / legacy path.
        """
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        c = config.get("model", config)
        return cls(
            input_dim      = c.get("input_dim",       N_HVGS_DEFAULT),
            embedding_dim  = c.get("embedding_dim",   256),
            num_smoke      = num_smoke_types if num_smoke_types is not None else c.get("num_smoke_types", N_SMOKE_CLASSES),
            num_cell_types = c.get("num_cell_types",  N_CELL_TYPES),
            encoder_dropout= c.get("encoder_dropout", 0.3),
            head_dropout   = c.get("head_dropout",    0.2),
            attention_dim  = c.get("attention_dim",   128),
            pooling        = pooling if pooling is not None else c.get("pooling", "attention"),
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


# ─── Focal loss (smoke-head imbalance ablation) ──────────────────────────────

VALID_SMOKE_LOSS_TYPES = ("cross_entropy", "focal")


class FocalLoss(nn.Module):
    """
    Multiclass focal loss (Lin et al., ICCV 2017) for the smoke-type head —
    a configurable ABLATION against plain CrossEntropyLoss (see
    MultiTaskLoss's loss_type parameter), never an unconditional
    replacement.

    Computes per-example cross-entropy first, derives pt = exp(-ce), then
    scales by (1 - pt) ** gamma so confidently-correct ("easy") examples
    contribute progressively less to the loss as gamma increases; gamma=0
    makes the scaling factor exactly 1 for every example, i.e. identical to
    plain (optionally class-weighted) cross-entropy. class_weight (alpha),
    if given, is passed once into the underlying per-example cross-entropy
    term — it is never applied a second time to the reduced/modulated loss,
    which would double-count the class correction.
    """

    def __init__(
        self,
        gamma:        float                   = 2.0,
        class_weight: Optional[torch.Tensor] = None,
        reduction:    str                     = "mean",
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"FocalLoss: gamma must be >= 0, got {gamma}.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"FocalLoss: reduction must be 'mean', 'sum' or 'none', got {reduction!r}.")
        self.gamma = gamma
        self.class_weight = class_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.class_weight, reduction="none")
        # clamp(max=1.0) guards against a pt slightly > 1.0 from floating-point
        # error when ce is extremely close to 0, which would make (1-pt)
        # negative and (1-pt)**gamma NaN for non-integer gamma.
        pt = torch.exp(-ce).clamp(max=1.0)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# ─── Multi-Task Loss ──────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Phase 1 : λ_smoke·L_smoke  +  λ_malig·L_BCE(malignancy)
    Phase 2 : L_BCE(subject)
    Phase 3 : λ_smoke·L_smoke  +  λ_malig·L_BCE(cell)  +  λ_sub·L_BCE(subject)

    L_smoke is plain CrossEntropyLoss by default (loss_type="cross_entropy")
    or FocalLoss (loss_type="focal") — a controlled, explicitly configured
    ablation for the class-imbalance work in Phase 2 (see
    data/sampling.py's module docstring for the sampling side of that same
    problem). smoke_class_weights is used as the alpha term for whichever
    loss is selected; it is applied exactly once either way.

    Private helpers _ls / _lm / _lsb keep the three public methods DRY.
    """

    def __init__(
        self,
        lambda_smoke:     float                   = 0.30,
        lambda_malignancy:float                   = 0.30,
        lambda_subject:   float                   = 0.40,
        lambda_dose:      float                   = 0.10,
        dose_margin:      float                   = 0.05,
        smoke_class_weights: Optional[torch.Tensor] = None,
        loss_type:         str                    = "cross_entropy",
        focal_gamma:        float                  = 2.0,
    ):
        super().__init__()
        if loss_type not in VALID_SMOKE_LOSS_TYPES:
            raise ValueError(f"MultiTaskLoss: loss_type={loss_type!r} must be one of {VALID_SMOKE_LOSS_TYPES}")
        self.λs   = lambda_smoke
        self.λm   = lambda_malignancy
        self.λsb  = lambda_subject
        self.λd   = lambda_dose
        self.margin = dose_margin
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        if loss_type == "focal":
            self.ce = FocalLoss(gamma=focal_gamma, class_weight=smoke_class_weights)
        else:
            self.ce = nn.CrossEntropyLoss(weight=smoke_class_weights)
        self.bce  = nn.BCELoss()
        self.mse  = nn.MSELoss()

    def _ls (self, logits, targets): return self.ce (logits, targets)

    def _lm(self, preds, targets, known_mask=None):
        """
        BCE for malignancy, optionally restricted to cells with a REAL label.
        Cells with no verified malignancy call are stamped 0.0 as a numeric
        placeholder (see labellers.py::add_malignancy_labels) — training
        against that placeholder as if it were a confirmed negative would
        teach the model "everything is benign unless proven otherwise",
        which is not a label anyone actually assigned. known_mask=None
        preserves the old unmasked behaviour for callers (tests, the
        model.py smoke test) that pass fully-synthetic, fully-known labels.
        """
        preds, targets = preds.view(-1), targets.float().view(-1)
        if known_mask is not None:
            known_mask = known_mask.view(-1).bool()
            if known_mask.sum() == 0:
                return preds.sum() * 0.0
            preds, targets = preds[known_mask], targets[known_mask]
        return self.bce(preds, targets)

    def _lsb(self, prob,   target) : return self.bce(prob.view(-1),   target.view(-1).float())

    def dose_response_loss(
        self,
        dose_preds:   torch.Tensor,   # [N, 1] or [N]  predicted dose fraction
        dose_targets: torch.Tensor,   # [N]  normalised dose, DOSE_UNKNOWN where unlabelled
        malig_preds:  torch.Tensor,   # [N, 1] or [N]  predicted malignancy, same cells
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Two terms, both restricted to cells with a known exposure dose:
          1. MSE(dose_pred, dose_target)              — regress the dose itself
          2. Pairwise monotonic ranking hinge          — cells exposed longer
             must not score LESS malignant than cells exposed for less time:
                 for dose_i > dose_j + margin:  hinge(malig_j - malig_i + margin)
        This is what makes the model dose-response-aware rather than merely
        dose-blind-categorical, per DoseResponseHead's docstring.
        Returns zero loss (still a tensor, safe to add into a total) when no
        cell in the batch has a known dose.
        """
        dose_preds   = dose_preds.view(-1)
        malig_preds  = malig_preds.view(-1)
        dose_targets = dose_targets.view(-1)
        known = dose_targets >= 0
        if known.sum() < 2:
            zero = dose_preds.sum() * 0.0
            return zero, {"total": 0.0, "regression": 0.0, "ranking": 0.0, "n_known": int(known.sum())}

        dp, dt, mp = dose_preds[known], dose_targets[known], malig_preds[known]
        regression = self.mse(dp, dt)

        # Pairwise dose ordering -> malignancy ordering, vectorised.
        dose_diff  = dt.unsqueeze(0) - dt.unsqueeze(1)         # [n, n]  dose_i - dose_j
        malig_diff = mp.unsqueeze(0) - mp.unsqueeze(1)         # [n, n]  malig_i - malig_j
        pair_mask  = dose_diff > self.margin                    # only strictly-ordered pairs
        ranking = (
            F.relu(self.margin - malig_diff)[pair_mask].mean()
            if pair_mask.any() else dp.sum() * 0.0
        )

        total = regression + ranking
        return total, {
            "total": total.item(), "regression": regression.item(),
            "ranking": ranking.item() if pair_mask.any() else 0.0,
            "n_known": int(known.sum()),
        }

    def cell_level_loss(
        self,
        smoke_logits: torch.Tensor, smoke_targets: torch.Tensor,
        malig_preds:  torch.Tensor, malig_targets: torch.Tensor,
        malig_known:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ls = self._ls(smoke_logits, smoke_targets)
        lm = self._lm(malig_preds, malig_targets, malig_known)
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
        malig_known:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ls, lm, lsb = (
            self._ls (smoke_logits, smoke_targets),
            self._lm (malig_preds,  malig_targets, malig_known),
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

    # Dose-response head + loss — novel continuous exposure modeling
    dose_pred = model.dose_head(z)
    assert dose_pred.shape == (B, 1)
    dose_t_known   = torch.rand(B)                       # all known
    dose_t_unknown = torch.full((B,), DOSE_UNKNOWN)       # none known
    l4, d4 = loss_fn.dose_response_loss(dose_pred, dose_t_known,   m)
    l5, d5 = loss_fn.dose_response_loss(dose_pred, dose_t_unknown, m)
    assert l4.requires_grad, "dose loss must be differentiable when doses are known"
    assert d5["n_known"] == 0 and d5["total"] == 0.0, "unknown doses must contribute zero loss"
    print(f"dose_response   ✓  known={d4}  unknown={d5}")

    assert l1.requires_grad and l2.requires_grad and l3.requires_grad, "losses must be differentiable"
    print(f"cell_level_loss ✓  {d1}")
    print(f"subject_loss    ✓  {d2}")
    print(f"end_to_end_loss ✓  {d3}")
    print("\n=== PASSED ===")