"""benchmarks/neural.py + model.py MIL pooling ablation — output shapes, parameter counts."""
import shutil
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from model import MultiSmokeCancerNet
from benchmarks.neural import MIL_POOLINGS, NeuralCancerAdapter, NeuralSmokeAdapter, count_parameters
from benchmarks.runner import build_synthetic_context


def test_all_mil_poolings_share_output_shape_and_encoder_dims():
    x_bag = torch.randn(20, 15)
    ct_ids = torch.randint(0, 4, (20,))
    param_counts = {}
    for pooling in MIL_POOLINGS:
        model = MultiSmokeCancerNet(input_dim=15, embedding_dim=16, num_smoke=3,
                                      num_cell_types=4, attention_dim=8, pooling=pooling)
        out = model.forward_subject(x_bag, ct_ids)
        assert out["cancer_probability"].shape == (1, 1)
        assert model.encoder.net[0].in_features == 15  # same encoder dims across poolings
        param_counts[pooling] = count_parameters(model)
    # attention has strictly more aggregator parameters than mean/max (extra gates)
    assert param_counts["attention"] > param_counts["mean"]
    assert param_counts["attention"] > param_counts["max"]


def test_mean_and_max_pooling_report_no_attention_weights():
    x_bag = torch.randn(10, 15)
    ct_ids = torch.randint(0, 4, (10,))
    for pooling in ("mean", "max"):
        model = MultiSmokeCancerNet(input_dim=15, embedding_dim=16, num_smoke=3,
                                      num_cell_types=4, attention_dim=8, pooling=pooling)
        out = model.forward_subject(x_bag, ct_ids)
        assert out["attention_weights"] is None  # nothing to misinterpret as causal


def test_attention_pooling_weights_still_sum_to_one():
    x_bag = torch.randn(10, 15)
    ct_ids = torch.randint(0, 4, (10,))
    model = MultiSmokeCancerNet(input_dim=15, embedding_dim=16, num_smoke=3,
                                  num_cell_types=4, attention_dim=8, pooling="attention")
    out = model.forward_subject(x_bag, ct_ids)
    assert abs(out["attention_weights"].sum().item() - 1.0) < 1e-5


def test_invalid_pooling_name_rejected():
    with pytest.raises(ValueError, match="pooling"):
        MultiSmokeCancerNet(pooling="median")


def test_neural_smoke_adapter_predicts_correct_shape():
    ctx = build_synthetic_context(seed=1, fast=True)
    adapter = NeuralSmokeAdapter(ctx.config, device="cpu")
    adapter.fit(ctx, ctx.train_cell_dataset, ctx.val_cell_dataset, seed=42)
    preds = adapter.predict(ctx.val_cell_dataset)
    assert preds.shape == (len(ctx.val_cell_dataset),)
    meta = adapter.metadata()
    assert meta["n_parameters"] > 0
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_neural_cancer_adapter_reports_pooling_and_param_counts():
    ctx = build_synthetic_context(seed=1, fast=True)
    from train import SubjectLevelDataset
    import numpy as np
    rng = np.random.RandomState(2)

    def make_bag(sid, label, n_genes=15, n=10):
        return {
            "subject_id": sid, "gene_matrix": rng.randn(n, n_genes).astype("float32") + label,
            "cell_type_ids": rng.randint(0, 4, n), "smoke_labels": rng.randint(0, 3, n),
            "malig_labels": rng.rand(n).astype("float32"), "malig_known": np.zeros(n, dtype=bool),
            "cancer_label": label, "cancer_label_known": True,
        }
    train_bags = [make_bag(f"tr_{i}", i % 2) for i in range(12)]
    val_bags = [make_bag(f"va_{i}", i % 2) for i in range(10)]
    train_sd = SubjectLevelDataset(train_bags)
    val_sd = SubjectLevelDataset(val_bags)
    adapter = NeuralCancerAdapter(pooling="mean", device="cpu")
    adapter.fit(ctx, ctx.train_cell_dataset, ctx.val_cell_dataset, train_sd, val_sd, seed=42, pretrain_epochs=1)
    proba = adapter.predict_proba(val_sd)
    assert proba.shape == (len(val_sd),)
    meta = adapter.metadata()
    assert meta["pooling"] == "mean"
    assert meta["n_aggregator_parameters"] > 0
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)
