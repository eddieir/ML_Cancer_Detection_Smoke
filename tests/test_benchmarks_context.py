"""benchmarks/context.py + Trainer.from_experiment_context."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.runner import build_synthetic_context


def test_from_pipeline_result_rejects_non_split_aware_dict():
    from benchmarks.context import ExperimentContext
    with pytest.raises(ValueError, match="run_pipeline_split_aware"):
        ExperimentContext.from_pipeline_result({"cell_data": {}, "bags": []}, config={})


def test_context_basic_properties():
    ctx = build_synthetic_context(seed=1, fast=True)
    assert ctx.num_smoke_classes == 3
    assert ctx.input_dim == 15
    assert set(ctx.subjects_for("train")) & set(ctx.subjects_for("val")) == set()
    assert set(ctx.subjects_for("train")) & set(ctx.subjects_for("test")) == set()


def test_trainer_from_experiment_context_builds_correct_k_class_model():
    from train import Trainer
    ctx = build_synthetic_context(seed=1, fast=True)
    trainer = Trainer.from_experiment_context(ctx, device="cpu")
    assert trainer.model.num_smoke == ctx.num_smoke_classes == 3
    assert trainer.model.input_dim == ctx.input_dim
    assert trainer.label_mapping.k == 3


def test_trainer_from_experiment_context_rejects_input_dim_mismatch():
    from train import Trainer
    ctx = build_synthetic_context(seed=1, fast=True)
    ctx.config["model"]["input_dim"] = ctx.input_dim + 5
    with pytest.raises(ValueError, match="input_dim"):
        Trainer.from_experiment_context(ctx, device="cpu")


def test_merged_five_class_pipeline_cannot_train_six_output_model():
    """Direct end-to-end proof: a K=5 effective label mapping (one raw class
    merged away) must never silently produce a 6-class model head."""
    from train import Trainer
    from data.label_mapping import build_effective_label_mapping

    ctx = build_synthetic_context(seed=1, fast=True)
    five_class_report = {
        "policy": "merge_into_dual_use_or_other",
        "affected_classes": {"cannabis": {"action": "merged_into_dual_use"}},
    }
    ctx.label_mapping = build_effective_label_mapping(
        five_class_report, raw_id_to_name={0: "cigarette", 1: "vape", 2: "cannabis", 3: "dual_use", 4: "cigar", 5: "unexposed"},
    )
    assert ctx.label_mapping.k == 5
    trainer = Trainer.from_experiment_context(ctx, device="cpu")
    assert trainer.model.num_smoke == 5, "model must be built with the effective K, never the fixed 6-class default"
