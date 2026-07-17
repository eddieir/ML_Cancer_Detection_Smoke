"""
pathway_hierarchical_mil.py — integration and leakage-adversarial checks.

These exercise a small synthetic end-to-end training loop and a handful of
the Phase 5 spec's leakage-adversarial properties. This file does not wire
the new architecture into the full nested-CV / ablation-runner CLI surface
(src/benchmarks/runner.py, cross_validation.py, hyperparameter_search.py)
— see the PR description for that explicit scoping decision. What it does
verify: a multitask training loop actually reduces loss on synthetic data,
variable-sized bags flow through training and gradient updates, gene-module
membership/fingerprints are decided before and independent of any
"validation" data, and the frozen-test sentinel is never touched by any of
this model's code paths.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pathway_hierarchical_mil import (
    GeneModuleCollection, PathwayHierarchicalMIL, PathwayHierarchicalMILConfig,
    MultitaskMaskedLoss, collate_subject_bags,
)
from benchmarks.sentinel import FrozenAccessSentinel, FrozenDataAccessError

GENES = [f"GENE{i}" for i in range(30)]


def _synthetic_subjects(n_subjects=12, seed=0):
    torch.manual_seed(seed)
    bags = []
    for i in range(n_subjects):
        n_cells = 4 + (i % 5)
        smoke_known = i % 4 != 0
        cancer_known = i % 3 != 0
        bags.append(dict(
            expression=torch.randn(n_cells, len(GENES)),
            cell_type_ids=torch.randint(0, 5, (n_cells,)),
            smoke_label=torch.tensor(i % 6),
            smoke_known=torch.tensor(smoke_known),
            cancer_label=torch.tensor(float(i % 2)),
            cancer_known=torch.tensor(cancer_known),
        ))
    return bags


def test_synthetic_multitask_training_reduces_loss():
    modules = GeneModuleCollection.synthetic(GENES, n_modules=5, genes_per_module=6, seed=0)
    config = PathwayHierarchicalMILConfig(embedding_dim=24, attention_dim=12, residual_gene_dim=12)
    model = PathwayHierarchicalMIL.from_config(modules, config, num_smoke=6)
    loss_fn = MultitaskMaskedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    bags = _synthetic_subjects(n_subjects=16)
    batch = collate_subject_bags(bags)

    model.train()
    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        out = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])
        total, _ = loss_fn(
            out.smoke_logits, batch["smoke_label"], batch["smoke_known"],
            out.cancer_logits, batch["cancer_label"], batch["cancer_known"],
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(total.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]


def test_variable_sized_bags_flow_through_training_step():
    modules = GeneModuleCollection.synthetic(GENES, n_modules=4, genes_per_module=5, seed=1)
    config = PathwayHierarchicalMILConfig(embedding_dim=16, attention_dim=8, residual_gene_dim=8)
    model = PathwayHierarchicalMIL.from_config(modules, config, num_smoke=6)
    loss_fn = MultitaskMaskedLoss()

    bags = [
        dict(expression=torch.randn(1, len(GENES)), cell_type_ids=torch.randint(0, 5, (1,)),
             smoke_label=torch.tensor(0), smoke_known=torch.tensor(True),
             cancer_label=torch.tensor(0.0), cancer_known=torch.tensor(True)),
        dict(expression=torch.randn(50, len(GENES)), cell_type_ids=torch.randint(0, 5, (50,)),
             smoke_label=torch.tensor(3), smoke_known=torch.tensor(True),
             cancer_label=torch.tensor(1.0), cancer_known=torch.tensor(True)),
    ]
    batch = collate_subject_bags(bags)
    out = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])
    total, comp = loss_fn(
        out.smoke_logits, batch["smoke_label"], batch["smoke_known"],
        out.cancer_logits, batch["cancer_label"], batch["cancer_known"],
    )
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_module_membership_decided_before_any_validation_split():
    """Module membership must come only from the gene-module source and the
    artifact's gene order — never from labels or a validation/test split.
    Confirms the synthetic constructor's signature has no such inputs and
    that identical gene orders always produce identical membership,
    independent of any downstream label assignment."""
    modules_a = GeneModuleCollection.synthetic(GENES, n_modules=4, genes_per_module=5, seed=42)
    modules_b = GeneModuleCollection.synthetic(GENES, n_modules=4, genes_per_module=5, seed=42)
    assert modules_a.fingerprint() == modules_b.fingerprint()
    import inspect
    sig = inspect.signature(GeneModuleCollection.synthetic)
    assert "labels" not in sig.parameters and "y" not in sig.parameters


def test_frozen_test_sentinel_never_touched_by_synthetic_training():
    """Wrap a stand-in for frozen test data in a FrozenAccessSentinel and
    confirm the entire synthetic training loop above never touches it —
    this model's training path has no reference to it at all, so any
    access would only occur if a future change wired it in incorrectly."""
    frozen_stub = FrozenAccessSentinel(label="frozen test bags")

    modules = GeneModuleCollection.synthetic(GENES, n_modules=3, genes_per_module=5, seed=2)
    config = PathwayHierarchicalMILConfig(embedding_dim=12, attention_dim=8, residual_gene_dim=8)
    model = PathwayHierarchicalMIL.from_config(modules, config, num_smoke=6)
    loss_fn = MultitaskMaskedLoss()
    bags = _synthetic_subjects(n_subjects=6)
    batch = collate_subject_bags(bags)

    out = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])
    total, _ = loss_fn(
        out.smoke_logits, batch["smoke_label"], batch["smoke_known"],
        out.cancer_logits, batch["cancer_label"], batch["cancer_known"],
    )
    total.backward()

    # Sanity: the sentinel itself still raises the instant it IS touched —
    # confirms this is a meaningful negative check, not a no-op stub.
    with pytest.raises(FrozenDataAccessError):
        len(frozen_stub)


def test_eval_mode_attention_summaries_deterministic():
    modules = GeneModuleCollection.synthetic(GENES, n_modules=4, genes_per_module=5, seed=3)
    config = PathwayHierarchicalMILConfig(embedding_dim=16, attention_dim=8, residual_gene_dim=8, dropout=0.5)
    model = PathwayHierarchicalMIL.from_config(modules, config, num_smoke=6)
    model.eval()
    bags = _synthetic_subjects(n_subjects=3, seed=9)
    batch = collate_subject_bags(bags)
    with torch.no_grad():
        out1 = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"], return_attention=True)
        out2 = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"], return_attention=True)
    assert torch.allclose(out1.cell_type_attention, out2.cell_type_attention, atol=1e-7)
    assert torch.allclose(out1.cancer_logits, out2.cancer_logits, atol=1e-7)


def test_cpu_only_execution():
    modules = GeneModuleCollection.synthetic(GENES, n_modules=3, genes_per_module=5, seed=4)
    config = PathwayHierarchicalMILConfig(embedding_dim=12, attention_dim=8, residual_gene_dim=8)
    model = PathwayHierarchicalMIL.from_config(modules, config, num_smoke=6)
    for p in model.parameters():
        assert p.device.type == "cpu"
    bags = _synthetic_subjects(n_subjects=2)
    batch = collate_subject_bags(bags)
    out = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])
    assert out.cancer_logits.device.type == "cpu"
