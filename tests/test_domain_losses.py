"""
Tests for benchmarks/domain_losses.py — CORAL, MMD, gradient reversal, the
domain-adversarial classifier head, and domain_robustness config resolution.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.domain_losses import (
    DomainClassifierHead,
    DomainLossConfigurationError,
    DomainVocabularyError,
    coral_loss,
    domain_adversarial_loss,
    gradient_reversal,
    group_embeddings_by_source,
    mmd_loss,
    resolve_domain_robustness_config,
)


def test_coral_identical_distributions_near_zero():
    torch.manual_seed(0)
    x = torch.randn(40, 6)
    y = x + torch.randn(40, 6) * 1e-6
    loss, meta = coral_loss({"a": x, "b": y})
    assert loss.item() < 1e-6
    assert meta["n_source_pairs"] == 1


def test_coral_shifted_covariance_positive():
    torch.manual_seed(1)
    x = torch.randn(40, 6)
    y = torch.randn(40, 6) * 4 + 3
    loss, _ = coral_loss({"a": x, "b": y})
    assert loss.item() > 0.1


def test_coral_single_source_is_finite_zero():
    x = torch.randn(10, 4)
    loss, meta = coral_loss({"a": x})
    assert loss.item() == 0.0
    assert meta["n_source_pairs"] == 0
    assert torch.isfinite(loss)


def test_coral_small_source_no_nan():
    """A source with a single subject has no within-group variance to
    estimate — must not produce NaN."""
    x = torch.randn(1, 5)
    y = torch.randn(10, 5)
    loss, _ = coral_loss({"single": x, "many": y})
    assert torch.isfinite(loss)


def test_mmd_identical_distributions_near_zero():
    torch.manual_seed(2)
    x = torch.randn(30, 5)
    y = x + torch.randn(30, 5) * 1e-6
    loss, _ = mmd_loss({"a": x, "b": y})
    assert loss.item() < 1e-4


def test_mmd_shifted_distributions_higher():
    torch.manual_seed(3)
    x = torch.randn(30, 5)
    y = torch.randn(30, 5) * 3 + 6
    loss_shifted, _ = mmd_loss({"a": x, "b": y})
    loss_same, _ = mmd_loss({"a": x, "b": x + torch.randn(30, 5) * 1e-6})
    assert loss_shifted.item() > loss_same.item()


def test_mmd_finite_for_small_samples():
    x = torch.randn(1, 4)
    y = torch.randn(1, 4)
    loss, _ = mmd_loss({"a": x, "b": y})
    assert torch.isfinite(loss)


def test_mmd_unsupported_kernel_raises():
    with pytest.raises(DomainLossConfigurationError):
        mmd_loss({"a": torch.randn(3, 3), "b": torch.randn(3, 3)}, kernel="linear")


def test_gradient_reversal_reverses_encoder_gradient():
    torch.manual_seed(4)
    lin = torch.nn.Linear(4, 4)
    inp = torch.randn(6, 4, requires_grad=True)

    out_rev = gradient_reversal(lin(inp), lambda_=1.0)
    out_rev.sum().backward()
    grad_reversed = inp.grad.clone()

    inp.grad = None
    out_plain = lin(inp)
    out_plain.sum().backward()
    grad_plain = inp.grad.clone()

    assert torch.allclose(grad_reversed, -grad_plain, atol=1e-6)


def test_gradient_reversal_lambda_zero_is_identity_in_gradient():
    torch.manual_seed(5)
    inp = torch.randn(5, 3, requires_grad=True)
    out = gradient_reversal(inp * 2.0, lambda_=0.0)
    out.sum().backward()
    assert torch.allclose(inp.grad, torch.zeros_like(inp.grad))


def test_domain_classifier_head_rejects_unseen_source():
    head = DomainClassifierHead(8, ["sourceA", "sourceB"])
    with pytest.raises(DomainVocabularyError):
        head.source_indices(["sourceC"])


def test_domain_classifier_head_requires_at_least_two_sources():
    with pytest.raises(DomainLossConfigurationError):
        DomainClassifierHead(8, ["only_one"])


def test_domain_adversarial_loss_gradient_flows_to_embeddings():
    head = DomainClassifierHead(6, ["a", "b", "c"])
    emb = torch.randn(9, 6, requires_grad=True)
    sources = ["a", "b", "c"] * 3
    loss, meta = domain_adversarial_loss(head, emb, sources, lambda_=1.0)
    assert "domain_accuracy" in meta
    loss.backward()
    assert emb.grad is not None


def test_group_embeddings_by_source_preserves_gradient():
    emb = torch.randn(4, 3, requires_grad=True)
    groups = group_embeddings_by_source(emb, ["a", "a", "b", "b"])
    assert set(groups) == {"a", "b"}
    assert groups["a"].shape == (2, 3)
    loss = groups["a"].sum() + groups["b"].sum()
    loss.backward()
    assert emb.grad is not None


def test_resolve_domain_robustness_config_defaults_to_disabled_erm():
    cfg = resolve_domain_robustness_config(None)
    assert cfg["strategy"] == "erm"
    assert cfg["coral"]["enabled"] is False and cfg["coral"]["weight"] == 0.0
    assert cfg["mmd"]["enabled"] is False
    assert cfg["adversarial"]["enabled"] is False


def test_resolve_domain_robustness_config_rejects_unknown_strategy():
    with pytest.raises(DomainLossConfigurationError):
        resolve_domain_robustness_config({"strategy": "not_a_real_strategy"})


def test_resolve_domain_robustness_config_rejects_unknown_keys():
    with pytest.raises(DomainLossConfigurationError):
        resolve_domain_robustness_config({"coral": {"not_a_real_key": 1}})


def test_resolve_domain_robustness_config_rejects_negative_weight():
    with pytest.raises(DomainLossConfigurationError):
        resolve_domain_robustness_config({"coral": {"weight": -0.1}})
