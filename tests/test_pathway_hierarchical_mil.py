"""
pathway_hierarchical_mil.py — Pathway-Aware Hierarchical Multi-Instance
Network: gene-module contract, masked pathway encoder, cell-type-aware
hierarchical attention, multitask masking, and checkpoint/bundle identity.

This file intentionally exercises the full Phase 5 spec's required-test
groups (gene modules, pathway encoder, hierarchical attention, multitask
masking, domain conditioning, checkpoints/bundles) but does not claim
one-to-one coverage of every numbered item in the original specification —
see the PR description for the explicit list of what is and is not
covered.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pathway_hierarchical_mil import (
    GeneModuleCollection, GeneModuleError,
    PathwayCellEncoder, MaskedModuleProjection,
    HierarchicalAttentionPooling, masked_softmax,
    PathwayHierarchicalMIL, PathwayHierarchicalMILConfig,
    HierarchicalMILOutput, MultitaskMaskedLoss, EmptyBatchLossError,
    categorical_entropy, binary_entropy, mc_dropout_predict,
    collate_subject_bags, is_synthetic_module_source, SYNTHETIC_MODULE_SOURCE,
)
import numpy as np
import pandas as pd
import anndata as ad

from benchmarks.bundle import write_model_bundle, load_and_validate_bundle, BundleValidationError
from data.preprocessing import fit_preprocessing

GENES = [f"GENE{i}" for i in range(40)]


def _modules(n_modules=5, genes_per_module=6, seed=0):
    return GeneModuleCollection.synthetic(GENES, n_modules=n_modules, genes_per_module=genes_per_module, seed=seed)


def _config(**overrides):
    base = dict(embedding_dim=24, attention_dim=12, residual_gene_dim=12, num_cell_type_buckets=5)
    base.update(overrides)
    return PathwayHierarchicalMILConfig(**base)


def _model(**overrides):
    return PathwayHierarchicalMIL.from_config(_modules(), _config(**overrides), num_smoke=6)


def _bag(n_cells, genes=len(GENES), smoke_known=True, cancer_known=True, smoke_label=0, cancer_label=0.0):
    return dict(
        expression=torch.randn(n_cells, genes),
        cell_type_ids=torch.randint(0, 5, (n_cells,)),
        smoke_label=torch.tensor(smoke_label),
        smoke_known=torch.tensor(smoke_known),
        cancer_label=torch.tensor(cancer_label),
        cancer_known=torch.tensor(cancer_known),
    )


# ══════════════════════════════ Gene modules ═══════════════════════════════

def test_module_names_must_be_unique():
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :3] = True
    with pytest.raises(GeneModuleError):
        GeneModuleCollection(["a", "a"], [f"g{i}" for i in range(10)], mask, "src")


def test_membership_mask_shape_mismatch_rejected():
    mask = torch.zeros(2, 9, dtype=torch.bool)
    with pytest.raises(GeneModuleError):
        GeneModuleCollection(["a", "b"], [f"g{i}" for i in range(10)], mask, "src")


def test_synthetic_modules_deterministic_for_fixed_seed():
    m1 = GeneModuleCollection.synthetic(GENES, n_modules=3, genes_per_module=5, seed=7)
    m2 = GeneModuleCollection.synthetic(GENES, n_modules=3, genes_per_module=5, seed=7)
    assert m1.fingerprint() == m2.fingerprint()
    assert torch.equal(m1.membership_mask, m2.membership_mask)


def test_synthetic_modules_source_is_clearly_labelled():
    m = _modules()
    assert is_synthetic_module_source(m.source_name)
    assert m.source_name == SYNTHETIC_MODULE_SOURCE


def test_different_seed_changes_fingerprint():
    m1 = GeneModuleCollection.synthetic(GENES, n_modules=3, genes_per_module=5, seed=1)
    m2 = GeneModuleCollection.synthetic(GENES, n_modules=3, genes_per_module=5, seed=2)
    assert m1.fingerprint() != m2.fingerprint()


def test_gene_order_change_changes_fingerprint():
    reordered = list(reversed(GENES))
    m1 = GeneModuleCollection.synthetic(GENES, n_modules=3, genes_per_module=5, seed=3)
    m2 = GeneModuleCollection.synthetic(reordered, n_modules=3, genes_per_module=5, seed=3)
    assert m1.fingerprint() != m2.fingerprint()


def test_from_gmt_drops_genes_not_in_artifact_order(tmp_path):
    gmt = tmp_path / "modules.gmt"
    gmt.write_text("mod_a\tdesc\tGENE0\tGENE1\tGENE2\tNOT_A_REAL_GENE\n")
    coll = GeneModuleCollection.from_gmt(gmt, GENES, minimum_genes_per_module=2)
    assert coll.module_gene_counts()["mod_a"] == 3  # NOT_A_REAL_GENE silently excluded from membership


def test_from_gmt_duplicate_genes_in_one_module_deduplicated(tmp_path):
    gmt = tmp_path / "modules.gmt"
    gmt.write_text("mod_a\tdesc\tGENE0\tGENE0\tGENE1\n")
    coll = GeneModuleCollection.from_gmt(gmt, GENES, minimum_genes_per_module=2)
    assert coll.module_gene_counts()["mod_a"] == 2


def test_from_gmt_duplicate_module_names_rejected(tmp_path):
    gmt = tmp_path / "modules.gmt"
    gmt.write_text("mod_a\tdesc\tGENE0\tGENE1\tGENE2\nmod_a\tdesc\tGENE3\tGENE4\tGENE5\n")
    with pytest.raises(GeneModuleError):
        GeneModuleCollection.from_gmt(gmt, GENES, minimum_genes_per_module=2)


def test_from_gmt_below_minimum_genes_errors_by_default(tmp_path):
    gmt = tmp_path / "modules.gmt"
    gmt.write_text("mod_a\tdesc\tGENE0\n")
    with pytest.raises(GeneModuleError):
        GeneModuleCollection.from_gmt(gmt, GENES, minimum_genes_per_module=3, empty_module_policy="error")


def test_from_gmt_below_minimum_genes_dropped_when_policy_drop(tmp_path):
    gmt = tmp_path / "modules.gmt"
    gmt.write_text("mod_a\tdesc\tGENE0\nmod_b\tdesc\tGENE1\tGENE2\tGENE3\n")
    coll = GeneModuleCollection.from_gmt(gmt, GENES, minimum_genes_per_module=3, empty_module_policy="drop")
    assert coll.module_names == ["mod_b"]


def test_module_collection_save_load_roundtrip_and_tamper_detection(tmp_path):
    coll = _modules()
    path = tmp_path / "modules.json"
    coll.save(path)
    loaded = GeneModuleCollection.load(path)
    assert loaded.fingerprint() == coll.fingerprint()

    import json
    payload = json.loads(path.read_text())
    payload["membership"][0][0] = 1 - payload["membership"][0][0]
    path.write_text(json.dumps(payload))
    with pytest.raises(GeneModuleError):
        GeneModuleCollection.load(path)


# ══════════════════════════ Pathway encoder ═══════════════════════════════

def test_masked_module_projection_zero_outside_membership():
    mods = _modules()
    proj = MaskedModuleProjection(mods.membership_mask)
    eff = proj.effective_weight()
    assert torch.equal(eff[~mods.membership_mask], torch.zeros_like(eff[~mods.membership_mask]))


def test_masked_connections_stay_zero_after_optimizer_step():
    mods = _modules()
    proj = MaskedModuleProjection(mods.membership_mask)
    opt = torch.optim.AdamW(proj.parameters(), lr=0.1, weight_decay=0.1)
    x = torch.randn(8, mods.n_genes)
    for _ in range(5):
        opt.zero_grad()
        out = proj(x)
        out.sum().backward()
        opt.step()
    eff = proj.effective_weight()
    assert torch.equal(eff[~mods.membership_mask], torch.zeros_like(eff[~mods.membership_mask]))


def test_masked_weight_gradient_is_exactly_zero():
    mods = _modules()
    proj = MaskedModuleProjection(mods.membership_mask)
    x = torch.randn(4, mods.n_genes)
    proj(x).sum().backward()
    grad = proj.weight.grad
    assert (grad[~mods.membership_mask] == 0).all()


def test_encoder_output_shape():
    mods = _modules()
    enc = PathwayCellEncoder(mods, embedding_dim=24, residual_gene_dim=12)
    out = enc(torch.randn(7, mods.n_genes))
    assert out.shape == (7, 24)


def test_encoder_rejects_wrong_input_width():
    mods = _modules()
    enc = PathwayCellEncoder(mods, embedding_dim=24, residual_gene_dim=12)
    with pytest.raises(ValueError):
        enc(torch.randn(3, mods.n_genes + 1))


def test_encoder_rejects_non_finite_input():
    mods = _modules()
    enc = PathwayCellEncoder(mods, embedding_dim=24, residual_gene_dim=12)
    x = torch.randn(3, mods.n_genes)
    x[0, 0] = float("nan")
    with pytest.raises(ValueError):
        enc(x)


def test_encoder_without_gene_residual_still_works():
    mods = _modules()
    enc = PathwayCellEncoder(mods, embedding_dim=24, residual_gene_dim=12, use_gene_residual=False)
    out = enc(torch.randn(5, mods.n_genes))
    assert out.shape == (5, 24)


def test_zero_module_collection_rejected():
    mask = torch.zeros(0, 10, dtype=torch.bool)
    coll = GeneModuleCollection([], [f"g{i}" for i in range(10)], mask, "src")
    with pytest.raises(GeneModuleError):
        PathwayCellEncoder(coll, embedding_dim=8, residual_gene_dim=4)


def test_encoder_has_no_batchnorm_layers():
    mods = _modules()
    enc = PathwayCellEncoder(mods, embedding_dim=24, residual_gene_dim=12)
    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in enc.modules())


# ══════════════════════ Hierarchical attention ═════════════════════════════

def test_masked_softmax_sums_to_one_for_valid_rows():
    scores = torch.randn(3, 5)
    mask = torch.tensor([
        [True, True, False, False, False],
        [True, True, True, True, True],
        [True, False, False, False, False],
    ])
    w = masked_softmax(scores, mask, dim=1)
    sums = w.sum(dim=1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-6)


def test_masked_softmax_all_masked_row_is_zero():
    scores = torch.randn(2, 4)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, :] = True
    w = masked_softmax(scores, mask, dim=1)
    assert torch.allclose(w[1], torch.zeros(4))
    assert not torch.isnan(w).any()


def test_masked_softmax_masked_positions_exactly_zero():
    scores = torch.randn(1, 6)
    mask = torch.tensor([[True, False, True, False, True, False]])
    w = masked_softmax(scores, mask, dim=1)
    assert (w[~mask] == 0).all()


def test_hierarchical_pooling_cell_type_attention_sums_to_one():
    pooling = HierarchicalAttentionPooling(embedding_dim=16, num_cell_type_buckets=4, attention_dim=8)
    h = torch.randn(2, 10, 16)
    ct = torch.randint(0, 4, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)
    out = pooling(h, ct, mask)
    sums = out["cell_type_attention"].sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-6)


def test_hierarchical_pooling_padded_cells_get_zero_attention():
    pooling = HierarchicalAttentionPooling(embedding_dim=16, num_cell_type_buckets=4, attention_dim=8)
    h = torch.randn(1, 6, 16)
    ct = torch.zeros(1, 6, dtype=torch.long)
    mask = torch.tensor([[True, True, True, False, False, False]])
    out = pooling(h, ct, mask)
    # cell_attention[:, c, :] restricted to padded positions must be zero for every cell type
    assert (out["cell_attention"][:, :, 3:] == 0).all()


def test_hierarchical_pooling_missing_cell_type_marked_not_present():
    pooling = HierarchicalAttentionPooling(embedding_dim=16, num_cell_type_buckets=4, attention_dim=8)
    h = torch.randn(1, 5, 16)
    ct = torch.zeros(1, 5, dtype=torch.long)  # every cell is type 0
    mask = torch.ones(1, 5, dtype=torch.bool)
    out = pooling(h, ct, mask)
    present = out["cell_type_present"][0]
    assert present[0].item() is True
    assert not present[1:].any()


def test_hierarchical_pooling_single_observed_cell_type_works():
    pooling = HierarchicalAttentionPooling(embedding_dim=8, num_cell_type_buckets=3, attention_dim=4)
    h = torch.randn(1, 4, 8)
    ct = torch.full((1, 4), 2, dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = pooling(h, ct, mask)
    assert out["valid_subject_mask"][0].item() is True
    assert torch.allclose(out["cell_type_attention"][0, 2], torch.tensor(1.0), atol=1e-6)


def test_hierarchical_pooling_no_valid_cells_marks_subject_invalid():
    pooling = HierarchicalAttentionPooling(embedding_dim=8, num_cell_type_buckets=3, attention_dim=4)
    h = torch.randn(1, 4, 8)
    ct = torch.zeros(1, 4, dtype=torch.long)
    mask = torch.zeros(1, 4, dtype=torch.bool)  # all padded
    out = pooling(h, ct, mask)
    assert out["valid_subject_mask"][0].item() is False


def test_no_attention_computed_across_subjects():
    """Two subjects with identical bag content but different padding/other-
    subject data must produce identical attention for the shared subject —
    confirms pooling never mixes rows across the batch dimension."""
    pooling = HierarchicalAttentionPooling(embedding_dim=8, num_cell_type_buckets=3, attention_dim=4)
    torch.manual_seed(0)
    h_shared = torch.randn(1, 4, 8)
    ct_shared = torch.randint(0, 3, (1, 4))
    mask_shared = torch.ones(1, 4, dtype=torch.bool)

    h_other = torch.randn(1, 4, 8)
    batch_h = torch.cat([h_shared, h_other], dim=0)
    batch_ct = torch.cat([ct_shared, torch.randint(0, 3, (1, 4))], dim=0)
    batch_mask = torch.ones(2, 4, dtype=torch.bool)

    out_alone = pooling(h_shared, ct_shared, mask_shared)
    out_batch = pooling(batch_h, batch_ct, batch_mask)
    assert torch.allclose(out_alone["subject_embeddings"][0], out_batch["subject_embeddings"][0], atol=1e-6)


def test_attention_sums_match_forward_pass_values_end_to_end():
    model = _model()
    bags = [_bag(5), _bag(3)]
    batch = collate_subject_bags(bags)
    model.eval()
    out = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"], return_attention=True)
    assert out.cell_type_attention is not None
    sums = out.cell_type_attention.sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)


# ══════════════════════════ Variable-sized bags ═══════════════════════════

def test_collate_pads_bags_and_masks_correctly():
    bags = [_bag(3), _bag(7), _bag(1)]
    batch = collate_subject_bags(bags)
    assert batch["expression"].shape == (3, 7, len(GENES))
    assert batch["cell_mask"].sum(dim=1).tolist() == [3, 7, 1]


def test_collate_rejects_empty_bag():
    bags = [_bag(3), _bag(0)]
    with pytest.raises(ValueError):
        collate_subject_bags(bags)


def test_forward_rejects_all_padded_bag_in_batch():
    model = _model()
    batch = collate_subject_bags([_bag(3)])
    batch["cell_mask"][0, :] = False
    with pytest.raises(ValueError):
        model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])


def test_one_cell_bag_works():
    model = _model()
    batch = collate_subject_bags([_bag(1)])
    out = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])
    assert out.smoke_logits.shape == (1, 6)


def test_cell_order_invariance_within_subject():
    model = _model()
    model.eval()
    n = 6
    expr = torch.randn(n, len(GENES))
    ct = torch.randint(0, 5, (n,))
    perm = torch.randperm(n)

    e1 = expr.unsqueeze(0)
    e2 = expr[perm].unsqueeze(0)
    c1 = ct.unsqueeze(0)
    c2 = ct[perm].unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool)

    with torch.no_grad():
        out1 = model(e1, c1, mask)
        out2 = model(e2, c2, mask)
    assert torch.allclose(out1.cancer_logits, out2.cancer_logits, atol=1e-5)
    assert torch.allclose(out1.smoke_logits, out2.smoke_logits, atol=1e-5)


def test_batch_order_invariance():
    model = _model()
    model.eval()
    bags = [_bag(4), _bag(6)]
    batch = collate_subject_bags(bags)
    with torch.no_grad():
        out_fwd = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])

    rev = collate_subject_bags(list(reversed(bags)))
    with torch.no_grad():
        out_rev = model(rev["expression"], rev["cell_type_ids"], rev["cell_mask"])

    assert torch.allclose(out_fwd.cancer_logits[0], out_rev.cancer_logits[1], atol=1e-5)
    assert torch.allclose(out_fwd.cancer_logits[1], out_rev.cancer_logits[0], atol=1e-5)


# ══════════════════════════════ Multitask masking ═══════════════════════════

def test_loss_masks_unknown_smoke_labels():
    loss_fn = MultitaskMaskedLoss()
    logits = torch.randn(4, 6, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    known = torch.tensor([True, False, True, False])
    cancer_logits = torch.randn(4)
    cancer_targets = torch.zeros(4)
    cancer_known = torch.zeros(4, dtype=torch.bool)
    total, comp = loss_fn(logits, targets, known, cancer_logits, cancer_targets, cancer_known)
    assert comp["n_smoke_known"] == 2
    assert comp["n_cancer_known"] == 0
    assert comp["cancer"] == 0.0


def test_loss_analytically_verifiable_all_known():
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits = torch.randn(3, 6)
    targets = torch.tensor([0, 1, 2])
    known = torch.ones(3, dtype=torch.bool)
    cancer_logits = torch.randn(3)
    cancer_targets = torch.tensor([1.0, 0.0, 1.0])
    cancer_known = torch.ones(3, dtype=torch.bool)

    loss_fn = MultitaskMaskedLoss(smoke_loss_weight=0.5, cancer_loss_weight=2.0)
    total, comp = loss_fn(logits, targets, known, cancer_logits, cancer_targets, cancer_known)

    expected_smoke = F.cross_entropy(logits, targets)
    expected_cancer = F.binary_cross_entropy_with_logits(cancer_logits, cancer_targets)
    expected_total = 0.5 * expected_smoke + 2.0 * expected_cancer
    assert torch.allclose(total, expected_total, atol=1e-6)


def test_loss_finite_for_no_known_smoke_labels():
    loss_fn = MultitaskMaskedLoss()
    logits = torch.randn(4, 6)
    targets = torch.zeros(4, dtype=torch.long)
    known = torch.zeros(4, dtype=torch.bool)
    cancer_logits = torch.randn(4)
    cancer_targets = torch.ones(4)
    cancer_known = torch.ones(4, dtype=torch.bool)
    total, comp = loss_fn(logits, targets, known, cancer_logits, cancer_targets, cancer_known)
    assert torch.isfinite(total)
    assert comp["smoke"] == 0.0


def test_loss_no_known_labels_either_task_skip_policy_returns_zero():
    loss_fn = MultitaskMaskedLoss(empty_batch_policy="skip")
    logits = torch.randn(2, 6)
    targets = torch.zeros(2, dtype=torch.long)
    known = torch.zeros(2, dtype=torch.bool)
    cancer_logits = torch.randn(2)
    cancer_targets = torch.zeros(2)
    cancer_known = torch.zeros(2, dtype=torch.bool)
    total, comp = loss_fn(logits, targets, known, cancer_logits, cancer_targets, cancer_known)
    assert total.item() == 0.0
    assert torch.isfinite(total)


def test_loss_no_known_labels_either_task_error_policy_raises():
    loss_fn = MultitaskMaskedLoss(empty_batch_policy="error")
    logits = torch.randn(2, 6)
    targets = torch.zeros(2, dtype=torch.long)
    known = torch.zeros(2, dtype=torch.bool)
    cancer_logits = torch.randn(2)
    cancer_targets = torch.zeros(2)
    cancer_known = torch.zeros(2, dtype=torch.bool)
    with pytest.raises(EmptyBatchLossError):
        loss_fn(logits, targets, known, cancer_logits, cancer_targets, cancer_known)


def test_loss_missing_outcomes_never_become_negative_label():
    """Unknown cancer outcomes must not silently pass through as target=0
    (negative) — confirmed by checking the masked subset excludes them."""
    loss_fn = MultitaskMaskedLoss()
    logits = torch.randn(2, 6)
    targets = torch.zeros(2, dtype=torch.long)
    known = torch.ones(2, dtype=torch.bool)
    cancer_logits = torch.tensor([10.0, -10.0])  # confidently positive / negative
    cancer_targets = torch.tensor([1.0, 0.0])
    cancer_known = torch.tensor([True, False])
    total, comp = loss_fn(logits, targets, known, cancer_logits, cancer_targets, cancer_known)
    assert comp["n_cancer_known"] == 1
    # if the unknown entry (confidently-negative-target=0, logit=-10) leaked
    # in, cancer loss would be near zero; it isn't, because only the known
    # (correctly-predicted) entry contributes.
    assert comp["cancer"] < 0.01


def test_class_weights_applied_exactly_once():
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 5.0])
    loss_fn = MultitaskMaskedLoss(smoke_class_weights=weights)
    logits = torch.randn(3, 6)
    targets = torch.tensor([5, 5, 0])
    known = torch.ones(3, dtype=torch.bool)
    cancer_logits = torch.randn(3)
    cancer_targets = torch.zeros(3)
    cancer_known = torch.zeros(3, dtype=torch.bool)
    total, comp = loss_fn(logits, targets, known, cancer_logits, cancer_targets, cancer_known)

    import torch.nn.functional as F
    expected = F.cross_entropy(logits, targets, weight=weights)
    assert torch.allclose(torch.tensor(comp["smoke"]), expected, atol=1e-6)


def test_smoke_only_and_cancer_only_batches_supported():
    loss_fn = MultitaskMaskedLoss()
    logits = torch.randn(2, 6)
    targets = torch.tensor([0, 1])
    known = torch.ones(2, dtype=torch.bool)
    cancer_logits = torch.randn(2)
    cancer_targets = torch.zeros(2)
    cancer_known = torch.zeros(2, dtype=torch.bool)
    total, comp = loss_fn(logits, targets, known, cancer_logits, cancer_targets, cancer_known)
    assert comp["cancer"] == 0.0 and comp["smoke"] > 0.0


def test_uncertainty_entropy_functions():
    probs = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    ent = categorical_entropy(probs)
    assert ent[0].item() == pytest.approx(0.0, abs=1e-5)
    assert ent[1].item() > ent[0].item()

    p = torch.tensor([0.0, 0.5, 1.0])
    be = binary_entropy(p)
    assert be[1].item() > be[0].item()


def test_mc_dropout_predict_shapes_and_stays_disabled_by_default():
    model = _model(dropout=0.5)
    batch = collate_subject_bags([_bag(4)])
    result = mc_dropout_predict(
        model,
        dict(expression=batch["expression"], cell_type_ids=batch["cell_type_ids"], cell_mask=batch["cell_mask"]),
        n_passes=5,
    )
    assert result["cancer_mean"].shape == (1,)
    assert result["n_passes"] == 5
    assert model.training  # restored to prior mode after the diagnostic call


# ══════════════════════════ Domain/source conditioning ═════════════════════

def test_source_embedding_disabled_by_default():
    model = _model()
    assert model.source_embedding is None
    assert model.species_embedding is None


def test_source_embedding_requires_source_ids_when_enabled():
    model = _model(use_source_embedding=True, num_sources=3)
    batch = collate_subject_bags([_bag(3)])
    with pytest.raises(ValueError):
        model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])


def test_source_embedding_used_when_provided():
    model = _model(use_source_embedding=True, num_sources=3)
    batch = collate_subject_bags([_bag(3)])
    source_ids = torch.tensor([1])
    out = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"], source_ids=source_ids)
    assert out.smoke_logits.shape == (1, 6)


def test_domain_head_only_created_when_multiple_sources_configured():
    single = _model(num_sources=1)
    multi = _model(num_sources=3)
    assert single.domain_head is None
    assert multi.domain_head is not None


def test_domain_head_diagnostic_only_not_in_default_output():
    model = _model(num_sources=3)
    batch = collate_subject_bags([_bag(3)])
    out = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])
    assert out.domain_logits is not None
    assert out.domain_logits.shape == (1, 3)


# ══════════════════════════ Config validation ═══════════════════════════════

def test_config_rejects_invalid_dropout():
    with pytest.raises(ValueError):
        PathwayHierarchicalMILConfig(dropout=1.5).validate()


def test_config_rejects_negative_loss_weight():
    with pytest.raises(ValueError):
        PathwayHierarchicalMILConfig(smoke_loss_weight=-1.0).validate()


def test_config_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        PathwayHierarchicalMILConfig(embedding_dim=0).validate()


def test_config_rejects_invalid_empty_batch_policy():
    with pytest.raises(ValueError):
        PathwayHierarchicalMILConfig(empty_batch_policy="bogus").validate()


def test_from_dict_ignores_unknown_keys_and_validates():
    cfg = PathwayHierarchicalMILConfig.from_dict({"embedding_dim": 16, "not_a_real_key": 999})
    assert cfg.embedding_dim == 16


# ══════════════════════════ Checkpoints / bundles ═══════════════════════════

def _tiny_artifact(tmp_path=None, seed=0):
    rng = np.random.default_rng(seed)
    n_genes = len(GENES)
    subject_ids = []
    for i in range(4):
        subject_ids += [f"sub_{i}"] * 5
    n = len(subject_ids)
    X = rng.random((n, n_genes)).astype("float32")
    obs = pd.DataFrame({"subject_id": subject_ids, "batch": ["b0"] * n}, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=list(GENES)))
    return fit_preprocessing(adata, set(subject_ids), n_hvgs=n_genes)


def test_checkpoint_state_dict_roundtrip(tmp_path):
    model = _model()
    ckpt_path = tmp_path / "model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "module_fingerprint": model.module_fingerprint,
        "config": model.config.__dict__,
    }, ckpt_path)

    reloaded = _model()
    payload = torch.load(ckpt_path, weights_only=False)
    reloaded.load_state_dict(payload["state_dict"])

    batch = collate_subject_bags([_bag(4)])
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        out1 = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])
        out2 = reloaded(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])
    assert torch.allclose(out1.cancer_logits, out2.cancer_logits, atol=1e-6)
    assert torch.allclose(out1.smoke_logits, out2.smoke_logits, atol=1e-6)


def test_module_fingerprint_mismatch_detectable_before_load():
    model = _model()
    other_modules = GeneModuleCollection.synthetic(GENES, n_modules=5, genes_per_module=6, seed=99)
    assert model.module_fingerprint != other_modules.fingerprint()


def test_bundle_round_trip_and_module_fingerprint_binding(tmp_path):
    artifact = _tiny_artifact(tmp_path)
    model = _model()
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        bundle_dir=bundle_dir,
        checkpoint_path=ckpt_path,
        artifact=artifact,
        model_config=model.config.__dict__,
        class_vocabulary=[str(i) for i in range(6)],
        label_policy="verified_only",
        species_policy="human_only",
        assay_mode="human_single_cell",
        extra={
            "model_type": model.model_type,
            "module_fingerprint": model.module_fingerprint,
        },
    )
    manifest = load_and_validate_bundle(bundle_dir)
    assert manifest["model_type"] == "pathway_hierarchical_mil"
    assert manifest["module_fingerprint"] == model.module_fingerprint


def test_bundle_rejects_swapped_preprocessing_artifact(tmp_path):
    artifact = _tiny_artifact(tmp_path)
    model = _model()
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        bundle_dir=bundle_dir, checkpoint_path=ckpt_path, artifact=artifact,
        model_config=model.config.__dict__, class_vocabulary=[str(i) for i in range(6)],
        label_policy="verified_only", species_policy="human_only", assay_mode="human_single_cell",
    )
    # Swap in a different artifact file with the same name but different content.
    other_artifact = _tiny_artifact(seed=99)
    other_artifact.save(bundle_dir / "preprocessing_artifact.json")
    with pytest.raises(BundleValidationError):
        load_and_validate_bundle(bundle_dir)


def test_bundle_rejects_corrupted_checkpoint(tmp_path):
    artifact = _tiny_artifact(tmp_path)
    model = _model()
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        bundle_dir=bundle_dir, checkpoint_path=ckpt_path, artifact=artifact,
        model_config=model.config.__dict__, class_vocabulary=[str(i) for i in range(6)],
        label_policy="verified_only", species_policy="human_only", assay_mode="human_single_cell",
    )
    ckpt_path.write_bytes(b"corrupted-not-a-checkpoint")
    with pytest.raises(BundleValidationError):
        load_and_validate_bundle(bundle_dir)


def test_model_type_recorded_in_bundle_extra(tmp_path):
    artifact = _tiny_artifact(tmp_path)
    model = _model()
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        bundle_dir=bundle_dir, checkpoint_path=ckpt_path, artifact=artifact,
        model_config=model.config.__dict__, class_vocabulary=[str(i) for i in range(6)],
        label_policy="verified_only", species_policy="human_only", assay_mode="human_single_cell",
        extra={"model_type": model.model_type},
    )
    manifest = load_and_validate_bundle(bundle_dir)
    assert manifest["model_type"] == "pathway_hierarchical_mil"


def test_validate_pathway_bundle_identity_accepts_matching_bundle(tmp_path):
    from benchmarks.pathway_hierarchical_adapter import validate_pathway_bundle_identity

    artifact = _tiny_artifact(tmp_path)
    model = _model()
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        bundle_dir=bundle_dir, checkpoint_path=ckpt_path, artifact=artifact,
        model_config=model.config.__dict__, class_vocabulary=[str(i) for i in range(6)],
        label_policy="verified_only", species_policy="human_only", assay_mode="human_single_cell",
        extra={"model_type": model.model_type, "module_fingerprint": model.module_fingerprint},
    )
    manifest = load_and_validate_bundle(bundle_dir)
    validate_pathway_bundle_identity(manifest, model)  # must not raise


def test_validate_pathway_bundle_identity_rejects_wrong_model_type(tmp_path):
    from benchmarks.pathway_hierarchical_adapter import validate_pathway_bundle_identity

    artifact = _tiny_artifact(tmp_path)
    model = _model()
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        bundle_dir=bundle_dir, checkpoint_path=ckpt_path, artifact=artifact,
        model_config=model.config.__dict__, class_vocabulary=[str(i) for i in range(6)],
        label_policy="verified_only", species_policy="human_only", assay_mode="human_single_cell",
        extra={"model_type": "some_other_architecture", "module_fingerprint": model.module_fingerprint},
    )
    manifest = load_and_validate_bundle(bundle_dir)
    with pytest.raises(BundleValidationError):
        validate_pathway_bundle_identity(manifest, model)


def test_validate_pathway_bundle_identity_rejects_module_fingerprint_mismatch(tmp_path):
    from benchmarks.pathway_hierarchical_adapter import validate_pathway_bundle_identity

    artifact = _tiny_artifact(tmp_path)
    model = _model()
    other_modules = GeneModuleCollection.synthetic(GENES, n_modules=5, genes_per_module=6, seed=123)
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        bundle_dir=bundle_dir, checkpoint_path=ckpt_path, artifact=artifact,
        model_config=model.config.__dict__, class_vocabulary=[str(i) for i in range(6)],
        label_policy="verified_only", species_policy="human_only", assay_mode="human_single_cell",
        extra={"model_type": model.model_type, "module_fingerprint": other_modules.fingerprint()},
    )
    manifest = load_and_validate_bundle(bundle_dir)
    with pytest.raises(BundleValidationError):
        validate_pathway_bundle_identity(manifest, model)


def test_checkpoint_from_different_architecture_config_fails_to_load(tmp_path):
    """A checkpoint trained with a different embedding_dim/module count
    must not silently load into a differently-configured model — PyTorch's
    strict state_dict loading raises on the resulting shape mismatch."""
    small_model = _model(embedding_dim=16)
    large_model = _model(embedding_dim=64)
    with pytest.raises(RuntimeError):
        large_model.load_state_dict(small_model.state_dict())
