"""model.py — MultiSmokeCancerNet architecture and MultiTaskLoss."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import N_CELL_TYPES, N_SMOKE_CLASSES, DOSE_UNKNOWN
from model import (
    MultiSmokeCancerNet, MultiTaskLoss, SmokeTypeHead, MalignancyHead,
    DoseResponseHead, GatedAttentionMIL,
)

GENES = 200


def _model():
    return MultiSmokeCancerNet(input_dim=GENES, embedding_dim=32, attention_dim=16)


def test_smoke_type_head_output_shape():
    z = torch.randn(10, 32)
    logits = SmokeTypeHead(embedding_dim=32, num_classes=N_SMOKE_CLASSES)(z)
    assert logits.shape == (10, N_SMOKE_CLASSES)


# ─── Effective label-space K wiring (MultiSmokeCancerNet.from_config) ───────

def test_from_config_defaults_to_six_classes():
    cfg = {"model": {"input_dim": GENES, "embedding_dim": 32, "attention_dim": 16}}
    model = MultiSmokeCancerNet.from_config(cfg)
    assert model.num_smoke == N_SMOKE_CLASSES
    z = torch.randn(4, 32)
    assert model.smoke_head(z).shape == (4, N_SMOKE_CLASSES)


def test_from_config_num_smoke_types_override_shrinks_output_dimension():
    """A rare-class policy that merges/excludes a class must actually shrink
    the model's smoke-head output — not leave a dead, unreachable neuron."""
    cfg = {"model": {"input_dim": GENES, "embedding_dim": 32, "attention_dim": 16,
                      "num_smoke_types": 6}}
    K = 5
    model = MultiSmokeCancerNet.from_config(cfg, num_smoke_types=K)
    assert model.num_smoke == K
    x = torch.randn(8, GENES)
    _, logits, _ = model.forward_cell(x)
    assert logits.shape == (8, K)


def test_from_config_override_none_falls_back_to_config_value():
    cfg = {"model": {"input_dim": GENES, "embedding_dim": 32, "attention_dim": 16,
                      "num_smoke_types": 4}}
    model = MultiSmokeCancerNet.from_config(cfg, num_smoke_types=None)
    assert model.num_smoke == 4


def test_malignancy_head_outputs_probability():
    z = torch.randn(10, 32)
    out = MalignancyHead(embedding_dim=32)(z)
    assert out.shape == (10, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_dose_response_head_outputs_bounded_dose():
    z = torch.randn(10, 32)
    out = DoseResponseHead(embedding_dim=32)(z)
    assert out.shape == (10, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_gated_attention_mil_attention_sums_to_one():
    n, embed_dim, feat_dim = 40, 32, 32 + N_SMOKE_CLASSES + 1 + N_CELL_TYPES
    agg = GatedAttentionMIL(feat_dim=feat_dim, embed_dim=embed_dim, attention_dim=16)
    z_bag = torch.randn(n, embed_dim)
    h_bag = torch.randn(n, feat_dim)
    prob, attn = agg(z_bag, h_bag)
    assert prob.shape == (1, 1)
    assert attn.shape == (n,)
    assert abs(attn.sum().item() - 1.0) < 1e-5


def test_forward_cell_shapes():
    model = _model()
    x = torch.randn(8, GENES)
    z, logits, malig = model.forward_cell(x)
    assert z.shape == (8, 32)
    assert logits.shape == (8, N_SMOKE_CLASSES)
    assert malig.shape == (8, 1)


def test_forward_subject_attention_sums_to_one_and_prob_in_range():
    model = _model()
    n = 50
    x_bag = torch.randn(n, GENES)
    ct_ids = torch.randint(0, N_CELL_TYPES, (n,))
    out = model.forward_subject(x_bag, ct_ids)
    assert out["cancer_probability"].shape == (1, 1)
    assert 0.0 <= out["cancer_probability"].item() <= 1.0
    assert abs(out["attention_weights"].sum().item() - 1.0) < 1e-5
    assert out["cell_smoke_probs"].shape == (n, N_SMOKE_CLASSES)
    assert out["cell_malignancy"].shape == (n, 1)


def test_from_config_matches_yaml(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "model:\n"
        "  input_dim: 64\n"
        "  embedding_dim: 16\n"
        "  num_smoke_types: 6\n"
        "  num_cell_types: 4\n"
        "  attention_dim: 8\n"
    )
    model = MultiSmokeCancerNet.from_config(cfg)
    z, logits, malig = model.forward_cell(torch.randn(4, 64))
    assert z.shape == (4, 16)


def test_multi_task_loss_all_three_modes_are_differentiable():
    model = _model()
    x = torch.randn(6, GENES)
    z, logits, malig = model.forward_cell(x)
    n = 20
    out = model.forward_subject(torch.randn(n, GENES), torch.randint(0, N_CELL_TYPES, (n,)))

    loss_fn = MultiTaskLoss()
    smoke_t = torch.randint(0, N_SMOKE_CLASSES, (6,))
    malig_t = torch.randint(0, 2, (6,)).float()
    cancer_t = torch.tensor([1.0])

    l1, d1 = loss_fn.cell_level_loss(logits, smoke_t, malig, malig_t)
    l2, d2 = loss_fn.subject_level_loss(out["cancer_probability"], cancer_t)
    l3, d3 = loss_fn.end_to_end_loss(logits, smoke_t, malig, malig_t,
                                      out["cancer_probability"], cancer_t)
    assert l1.requires_grad and l2.requires_grad and l3.requires_grad
    assert set(d1) == {"total", "smoke", "malignancy"}
    assert set(d2) == {"total", "subject"}
    assert set(d3) == {"total", "smoke", "malignancy", "subject"}


def test_malignancy_loss_ignores_unknown_cells():
    """Cells with malig_known=False must contribute zero gradient signal to
    the malignancy loss — they carry a 0.0 placeholder, not a real label."""
    loss_fn = MultiTaskLoss()
    malig_preds = torch.rand(10, 1)
    malig_targets = torch.zeros(10)          # all placeholder negatives
    known_mask = torch.zeros(10, dtype=torch.bool)   # none actually known
    loss = loss_fn._lm(malig_preds, malig_targets, known_mask)
    assert loss.item() == 0.0


def test_malignancy_loss_uses_only_known_cells():
    loss_fn = MultiTaskLoss()
    torch.manual_seed(0)
    malig_preds = torch.rand(6, 1, requires_grad=True)
    malig_targets = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    known_mask = torch.tensor([True, True, False, False, False, False])

    full_loss = loss_fn.bce(malig_preds.view(-1)[:2], malig_targets[:2])
    masked_loss = loss_fn._lm(malig_preds, malig_targets, known_mask)
    assert torch.allclose(full_loss, masked_loss)
    masked_loss.backward()
    assert malig_preds.grad is not None


def test_cell_level_loss_with_no_known_malignancy_is_differentiable_zero_malig_term():
    model = _model()
    x = torch.randn(6, GENES)
    _, logits, malig = model.forward_cell(x)
    smoke_t = torch.randint(0, N_SMOKE_CLASSES, (6,))
    malig_t = torch.zeros(6)
    known = torch.zeros(6, dtype=torch.bool)

    loss_fn = MultiTaskLoss()
    total, stats = loss_fn.cell_level_loss(logits, smoke_t, malig, malig_t, malig_known=known)
    assert stats["malignancy"] == 0.0
    assert total.requires_grad


def test_dose_response_loss_ignores_unknown_doses():
    loss_fn = MultiTaskLoss()
    dose_pred = torch.rand(10, 1)
    malig = torch.rand(10, 1)
    dose_t = torch.full((10,), DOSE_UNKNOWN)
    loss, stats = loss_fn.dose_response_loss(dose_pred, dose_t, malig)
    assert stats["n_known"] == 0
    assert stats["total"] == 0.0
    assert loss.item() == 0.0


def test_dose_response_loss_is_differentiable_with_known_doses():
    loss_fn = MultiTaskLoss()
    dose_pred = torch.rand(10, 1, requires_grad=True)
    malig = torch.rand(10, 1, requires_grad=True)
    dose_t = torch.rand(10)
    loss, stats = loss_fn.dose_response_loss(dose_pred, dose_t, malig)
    assert stats["n_known"] == 10
    assert loss.requires_grad
    loss.backward()
    assert dose_pred.grad is not None


def test_dose_response_loss_penalises_non_monotonic_malignancy():
    """
    Two cells, same smoke type: cell A has a much higher dose than cell B.
    If the model (wrongly) scores B as more malignant than A, the ranking
    term must be strictly positive — this is the monotonicity constraint
    that makes the head genuinely dose-response-aware, not just a regressor.
    """
    loss_fn = MultiTaskLoss(dose_margin=0.05)
    dose_pred = torch.zeros(2, 1)  # unused by ranking term, only regression
    dose_t = torch.tensor([0.9, 0.1])
    malig_violating = torch.tensor([[0.1], [0.9]])   # A (high dose) scored LESS malignant
    malig_consistent = torch.tensor([[0.9], [0.1]])   # A (high dose) scored MORE malignant

    _, stats_violating  = loss_fn.dose_response_loss(dose_pred, dose_t, malig_violating)
    _, stats_consistent = loss_fn.dose_response_loss(dose_pred, dose_t, malig_consistent)
    assert stats_violating["ranking"] > stats_consistent["ranking"]
    assert stats_consistent["ranking"] == 0.0
