"""
Tests for the Phase 2 smoke-imbalance loss machinery: model.py's FocalLoss/
MultiTaskLoss(loss_type=...) and CellLevelDataset.smoke_class_weights /
Trainer._smoke_loss_weights_and_type's train-only class-weighting contract.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from model import FocalLoss, MultiSmokeCancerNet, MultiTaskLoss
from train import CellLevelDataset, Trainer


# ─── Focal loss ─────────────────────────────────────────────────────────────────

def _random_logits_targets(seed=0, n=32, k=4):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, k, generator=g, requires_grad=True)
    targets = torch.randint(0, k, (n,), generator=g)
    return logits, targets


def test_focal_gamma_zero_matches_cross_entropy():
    logits, targets = _random_logits_targets()
    focal = FocalLoss(gamma=0.0)
    ce = torch.nn.CrossEntropyLoss()
    assert torch.allclose(focal(logits, targets), ce(logits.detach(), targets), atol=1e-5)


def test_focal_easy_examples_are_downweighted():
    """A confidently-correct example (small but nonzero CE) should have its
    loss shrink relative to plain CE much more than a hard example's does."""
    easy_logits = torch.tensor([[6.0, -6.0, -6.0]])
    easy_target = torch.tensor([0])
    hard_logits = torch.tensor([[0.1, 0.0, -0.1]])
    hard_target = torch.tensor([0])

    ce = torch.nn.CrossEntropyLoss(reduction="none")
    focal = FocalLoss(gamma=2.0, reduction="none")

    easy_ce, easy_focal = ce(easy_logits, easy_target).item(), focal(easy_logits, easy_target).item()
    hard_ce, hard_focal = ce(hard_logits, hard_target).item(), focal(hard_logits, hard_target).item()

    easy_ratio = easy_focal / easy_ce
    hard_ratio = hard_focal / hard_ce
    assert easy_ratio < hard_ratio
    assert easy_ratio < 0.01   # easy example's loss is almost entirely suppressed


def test_focal_hard_examples_retain_more_loss():
    hard_logits = torch.tensor([[0.0, 0.0, 5.0]])   # confidently WRONG (target=0)
    hard_target = torch.tensor([0])
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    focal = FocalLoss(gamma=2.0, reduction="none")
    ratio = focal(hard_logits, hard_target).item() / ce(hard_logits, hard_target).item()
    assert ratio > 0.8   # confidently wrong -> pt near 0 -> (1-pt)^gamma near 1


def test_focal_class_alpha_applied_exactly_once():
    logits, targets = _random_logits_targets(n=64, k=3)
    alpha = torch.tensor([0.2, 0.5, 2.0])
    # Compare SUM reductions (not "mean") so the comparison isn't confounded
    # by CrossEntropyLoss(reduction="mean")'s weight-normalized averaging
    # convention, which differs from focal loss's plain per-example mean —
    # summed per-example values are identical either way IF and ONLY IF
    # alpha was applied exactly once to each example.
    focal = FocalLoss(gamma=0.0, class_weight=alpha, reduction="sum")  # gamma=0 -> pure weighted CE
    weighted_ce = torch.nn.CrossEntropyLoss(weight=alpha, reduction="sum")
    # If alpha were applied twice, this would NOT match a single-application
    # weighted cross-entropy at gamma=0.
    assert torch.allclose(focal(logits, targets), weighted_ce(logits.detach(), targets), atol=1e-4)


def test_focal_alpha_does_not_influence_pt_when_gamma_positive():
    """The bug this guards against: computing pt from ALREADY class-weighted
    cross-entropy would make alpha influence the focal modulation itself
    (not just the final scale), so a large alpha shrinks pt, which changes
    (1-pt)**gamma, which is wrong. With alpha applied only after
    modulation, scaling alpha by a constant factor must scale the loss by
    that exact same factor — never touching pt/the modulation term."""
    logits, targets = _random_logits_targets(n=32, k=3, seed=5)
    alpha = torch.tensor([1.0, 3.0, 5.0])
    focal_a = FocalLoss(gamma=2.0, class_weight=alpha, reduction="none")
    focal_2a = FocalLoss(gamma=2.0, class_weight=alpha * 10.0, reduction="none")
    ratio = focal_2a(logits, targets) / focal_a(logits, targets)
    assert torch.allclose(ratio, torch.full_like(ratio, 10.0), atol=1e-4)


def test_focal_matches_manually_computed_tensor_with_gamma_and_alpha():
    """Blocker 3 requirement 1: compare against a small, fully hand-derived
    example combining gamma>0 AND alpha, computed independently of
    FocalLoss's own implementation."""
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 3.0]])
    targets = torch.tensor([0, 2])
    alpha = torch.tensor([2.0, 1.0, 0.5])
    gamma = 2.0

    # Manual per-example computation using plain softmax cross-entropy,
    # entirely independent of F.cross_entropy/FocalLoss internals.
    log_probs = torch.log_softmax(logits, dim=1)
    manual_ce = torch.tensor([
        -log_probs[0, 0].item(),
        -log_probs[1, 2].item(),
    ])
    manual_pt = torch.exp(-manual_ce)
    manual_focal = alpha[targets] * ((1.0 - manual_pt) ** gamma) * manual_ce

    focal = FocalLoss(gamma=gamma, class_weight=alpha, reduction="none")
    assert torch.allclose(focal(logits, targets), manual_focal, atol=1e-5)


def test_focal_reduction_none_returns_per_example_tensor():
    logits, targets = _random_logits_targets(n=10, k=3)
    focal = FocalLoss(gamma=1.5, reduction="none")
    out = focal(logits, targets)
    assert out.shape == (10,)


def test_focal_reduction_sum_equals_sum_of_none():
    logits, targets = _random_logits_targets(n=10, k=3)
    none_out = FocalLoss(gamma=1.5, reduction="none")(logits, targets)
    sum_out = FocalLoss(gamma=1.5, reduction="sum")(logits, targets)
    assert torch.allclose(sum_out, none_out.sum())


def test_focal_reduction_mean_is_arithmetic_mean_of_none():
    """Documented choice: 'mean' is the plain arithmetic mean (sum/N), not
    nn.CrossEntropyLoss(weight=...)'s weight-normalized mean."""
    logits, targets = _random_logits_targets(n=10, k=3)
    alpha = torch.tensor([1.0, 2.0, 5.0])
    none_out = FocalLoss(gamma=1.5, class_weight=alpha, reduction="none")(logits, targets)
    mean_out = FocalLoss(gamma=1.5, class_weight=alpha, reduction="mean")(logits, targets)
    assert torch.allclose(mean_out, none_out.mean())


def test_focal_invalid_class_weight_dimensionality_raises():
    with pytest.raises(ValueError):
        FocalLoss(gamma=1.0, class_weight=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_focal_non_finite_class_weight_raises():
    with pytest.raises(ValueError):
        FocalLoss(gamma=1.0, class_weight=torch.tensor([1.0, float("inf"), 2.0]))


def test_focal_negative_class_weight_raises():
    with pytest.raises(ValueError):
        FocalLoss(gamma=1.0, class_weight=torch.tensor([1.0, -0.5, 2.0]))


def test_focal_class_weight_count_mismatch_raises():
    logits, targets = _random_logits_targets(n=8, k=3)
    focal = FocalLoss(gamma=1.0, class_weight=torch.tensor([1.0, 2.0]))  # only 2, logits have 3 classes
    with pytest.raises(ValueError):
        focal(logits, targets)


def test_focal_invalid_target_index_raises():
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 5, 2])  # 5 is out of range for 3 classes
    focal = FocalLoss(gamma=1.0)
    with pytest.raises(ValueError):
        focal(logits, targets)


def test_focal_preserves_device_and_dtype():
    logits = torch.randn(6, 3, dtype=torch.float64)
    targets = torch.randint(0, 3, (6,))
    alpha = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)  # deliberately different dtype
    focal = FocalLoss(gamma=1.0, class_weight=alpha)
    out = focal(logits, targets)
    assert out.dtype == torch.float64
    assert out.device == logits.device


def test_focal_extreme_logits_remain_finite():
    logits = torch.tensor([[80.0, -80.0, -80.0], [-80.0, -80.0, 80.0]])
    targets = torch.tensor([0, 0])  # second row is confidently WRONG
    focal = FocalLoss(gamma=2.0)
    loss = focal(logits, targets)
    assert torch.isfinite(loss).all()


def test_focal_backprop_produces_finite_gradients():
    logits, targets = _random_logits_targets(n=16, k=5)
    focal = FocalLoss(gamma=3.0)
    loss = focal(logits, targets)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_focal_invalid_gamma_raises():
    with pytest.raises(ValueError):
        FocalLoss(gamma=-0.5)


def test_focal_invalid_reduction_raises():
    with pytest.raises(ValueError):
        FocalLoss(gamma=1.0, reduction="bogus")


def test_multi_task_loss_default_is_cross_entropy():
    loss_fn = MultiTaskLoss()
    assert loss_fn.loss_type == "cross_entropy"
    assert isinstance(loss_fn.ce, torch.nn.CrossEntropyLoss)


def test_multi_task_loss_focal_mode_uses_focal_loss():
    loss_fn = MultiTaskLoss(loss_type="focal", focal_gamma=1.5)
    assert isinstance(loss_fn.ce, FocalLoss)
    assert loss_fn.ce.gamma == 1.5


def test_multi_task_loss_rejects_invalid_loss_type():
    with pytest.raises(ValueError):
        MultiTaskLoss(loss_type="not_a_real_loss")


def test_multi_task_loss_focal_class_weights_applied_exactly_once():
    """smoke_class_weights is passed into FocalLoss's class_weight exactly
    once — verify MultiTaskLoss._ls (the smoke-loss call site) matches a
    direct FocalLoss(gamma=..., class_weight=...) call bit-for-bit, i.e.
    MultiTaskLoss does not re-apply the weights anywhere else."""
    torch.manual_seed(1)
    weights = torch.tensor([1.0, 4.0, 0.5])
    logits = torch.randn(12, 3, requires_grad=False)
    targets = torch.randint(0, 3, (12,))
    loss_fn = MultiTaskLoss(loss_type="focal", focal_gamma=2.0, smoke_class_weights=weights)
    direct = FocalLoss(gamma=2.0, class_weight=weights)
    assert torch.allclose(loss_fn._ls(logits, targets), direct(logits, targets), atol=1e-6)


def test_malignancy_cancer_dose_losses_unchanged_by_loss_type():
    """Switching the smoke loss to focal must not alter the other loss terms."""
    torch.manual_seed(0)
    B = 16
    malig_preds = torch.rand(B, 1)
    malig_targets = torch.randint(0, 2, (B,)).float()
    cancer_prob = torch.rand(1, 1)
    cancer_target = torch.tensor([1.0])

    ce_loss = MultiTaskLoss(loss_type="cross_entropy")
    focal_loss = MultiTaskLoss(loss_type="focal", focal_gamma=2.0)

    lm_ce = ce_loss._lm(malig_preds, malig_targets)
    lm_focal = focal_loss._lm(malig_preds, malig_targets)
    assert torch.allclose(lm_ce, lm_focal)

    lsb_ce = ce_loss._lsb(cancer_prob, cancer_target)
    lsb_focal = focal_loss._lsb(cancer_prob, cancer_target)
    assert torch.allclose(lsb_ce, lsb_focal)


def test_cross_entropy_remains_the_default_smoke_loss():
    loss_fn = MultiTaskLoss(smoke_class_weights=None)
    assert loss_fn.loss_type == "cross_entropy"


# ─── Train-only class weights ──────────────────────────────────────────────────

def _cell_dataset(labels, subject_ids, num_classes=6):
    n = len(labels)
    return CellLevelDataset(
        gene_matrix=np.random.RandomState(0).randn(n, 8).astype("float32"),
        smoke_labels=np.array(labels, dtype=np.int64),
        malignancy_labels=np.zeros(n, dtype="float32"),
        cell_type_ids=np.zeros(n, dtype=np.int64),
        subject_ids=np.array(subject_ids, dtype=object),
    )


def test_smoke_class_weights_use_training_data_only():
    train_ds = _cell_dataset([0] * 80 + [1] * 20, [f"s{i}" for i in range(100)], num_classes=2)
    w_before = train_ds.smoke_class_weights(num_classes=2)

    # A DIFFERENT dataset standing in for "validation" must not affect this.
    val_ds = _cell_dataset([1] * 50, [f"v{i}" for i in range(50)], num_classes=2)
    w_after = train_ds.smoke_class_weights(num_classes=2)  # unchanged — val never touched
    assert torch.allclose(w_before, w_after)
    assert not torch.allclose(w_before, val_ds.smoke_class_weights(num_classes=2))


def test_changing_validation_labels_does_not_change_train_weights():
    train_ds = _cell_dataset([0] * 60 + [1] * 40, [f"s{i}" for i in range(100)], num_classes=2)
    w1 = train_ds.smoke_class_weights(num_classes=2)
    # Construct a "validation" dataset with wildly different label balance —
    # train_ds's own weights must be identical since only train_ds.smoke is read.
    _cell_dataset([1] * 100, [f"v{i}" for i in range(100)], num_classes=2)
    w2 = train_ds.smoke_class_weights(num_classes=2)
    assert torch.allclose(w1, w2)


def test_weights_are_finite():
    train_ds = _cell_dataset([0] * 5 + [1] * 500, [f"s{i}" for i in range(505)], num_classes=2)
    w = train_ds.smoke_class_weights(num_classes=2)
    assert torch.isfinite(w).all()


def test_missing_classes_get_zero_weight_not_inf():
    train_ds = _cell_dataset([0] * 10, ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"], num_classes=3)
    w = train_ds.smoke_class_weights(num_classes=3)
    assert w[1].item() == 0.0
    assert w[2].item() == 0.0
    assert torch.isfinite(w).all()


def _trainer(smoke_imbalance=None, num_smoke=2):
    cfg = {
        "model": {"input_dim": 8, "num_smoke_types": num_smoke, "num_cell_types": 2},
        "train": {} if smoke_imbalance is None else {"smoke_imbalance": smoke_imbalance},
    }
    model = MultiSmokeCancerNet.from_config(cfg)
    return Trainer.from_config(model, cfg, device="cpu")


def test_class_weighting_none_produces_unweighted_ce():
    trainer = _trainer(smoke_imbalance={"class_weighting": "none"})
    train_ds = _cell_dataset([0] * 80 + [1] * 20, [f"s{i}" for i in range(100)], num_classes=2)
    reported, loss_alpha, loss_type, gamma = trainer._smoke_loss_weights_and_type(train_ds, None)
    assert reported is None
    assert loss_alpha is None


def test_class_weighting_inverse_frequency_matches_dataset_weights():
    trainer = _trainer(smoke_imbalance={"class_weighting": "inverse_frequency"})
    train_ds = _cell_dataset([0] * 80 + [1] * 20, [f"s{i}" for i in range(100)], num_classes=2)
    reported, loss_alpha, loss_type, gamma = trainer._smoke_loss_weights_and_type(train_ds, None)
    assert torch.allclose(reported, train_ds.smoke_class_weights(num_classes=2))
    assert torch.allclose(loss_alpha, reported)


def test_explicit_weights_override_config():
    trainer = _trainer(smoke_imbalance={"class_weighting": "none"})
    train_ds = _cell_dataset([0] * 80 + [1] * 20, [f"s{i}" for i in range(100)], num_classes=2)
    explicit = torch.tensor([5.0, 1.0])
    reported, loss_alpha, loss_type, gamma = trainer._smoke_loss_weights_and_type(train_ds, explicit)
    assert torch.allclose(reported, explicit)


def test_focal_alpha_mode_none_drops_alpha_even_with_inverse_frequency_weighting():
    """The documented double-correction guard: class_weighting=inverse_frequency
    + loss=focal + focal_alpha_mode=none must report the weights (for
    provenance) but NOT pass them into the loss as alpha."""
    trainer = _trainer(smoke_imbalance={
        "class_weighting": "inverse_frequency", "loss": "focal", "focal_alpha_mode": "none",
    })
    train_ds = _cell_dataset([0] * 80 + [1] * 20, [f"s{i}" for i in range(100)], num_classes=2)
    reported, loss_alpha, loss_type, gamma = trainer._smoke_loss_weights_and_type(train_ds, None)
    assert reported is not None
    assert loss_alpha is None
    assert loss_type == "focal"


def test_smoke_imbalance_config_persisted_in_resolved_config():
    trainer = _trainer(smoke_imbalance={"loss": "focal", "focal_gamma": 3.0})
    assert trainer.smoke_imbalance_config["loss"] == "focal"
    assert trainer.smoke_imbalance_config["focal_gamma"] == 3.0
    # untouched keys still resolve to documented defaults
    assert trainer.smoke_imbalance_config["sampler"] == "shuffle"
