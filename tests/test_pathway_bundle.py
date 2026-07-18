"""
Tests for PathwayHierarchicalAdapter.save_bundle/load_bundle — persisting
and restoring domain-adversarial state (domain head weights + fixed source
vocabulary + domain-robustness config) through a Phase-4-style bundle
(benchmarks/bundle.py), not just the primary task head.
"""
import shutil
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.bundle import BundleValidationError, LegacyBundleError
from benchmarks.pathway_hierarchical_adapter import PathwayHierarchicalAdapter
from benchmarks.runner import build_synthetic_context
from train import SubjectLevelDataset


def _fit_adapter(domain_robustness_config=None, seed=1):
    ctx = build_synthetic_context(seed=seed, fast=True)
    # ctx.train_bags/val_bags carry no per-bag "source" field (that field is
    # only populated by bags_from_fold_cell_dataset, the source-held-out
    # protocol's own bag builder) — stamp one on here so a domain_adversarial
    # config has >= 2 development sources to build its head from.
    train_bags = [dict(b, source="sourceA" if i % 2 == 0 else "sourceB") for i, b in enumerate(ctx.train_bags)]
    val_bags = [dict(b, source="sourceA" if i % 2 == 0 else "sourceB") for i, b in enumerate(ctx.val_bags)]
    train_sd = SubjectLevelDataset(train_bags, require_known_outcome=False)
    val_sd = SubjectLevelDataset(val_bags, require_known_outcome=False)
    adapter = PathwayHierarchicalAdapter(device="cpu", domain_robustness_config=domain_robustness_config)
    adapter.fit(ctx, None, None, train_sd, val_sd, seed=seed, pretrain_epochs=2)
    return ctx, adapter, val_sd


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_exact_round_trip_erm(tmp_path):
    ctx, adapter, val_sd = _fit_adapter()
    bundle_dir = tmp_path / "bundle_erm"
    adapter.save_bundle(bundle_dir, ctx.preprocessing_artifact)

    reloaded = PathwayHierarchicalAdapter.load_bundle(bundle_dir, device="cpu")
    assert reloaded.model_state_fingerprint() == adapter.model_state_fingerprint()
    assert reloaded.domain_head is None
    before = adapter.predict_proba(val_sd)
    after = reloaded.predict_proba(val_sd)
    assert before == pytest.approx(after, abs=1e-6)


def test_exact_round_trip_domain_adversarial(tmp_path):
    cfg = {"strategy": "domain_adversarial",
           "adversarial": {"enabled": True, "weight": 0.1, "gradient_reversal_lambda": 1.0, "warmup_epochs": 0}}
    ctx, adapter, val_sd = _fit_adapter(domain_robustness_config=cfg)
    assert adapter.domain_head is not None
    bundle_dir = tmp_path / "bundle_adv"
    adapter.save_bundle(bundle_dir, ctx.preprocessing_artifact)

    reloaded = PathwayHierarchicalAdapter.load_bundle(bundle_dir, device="cpu")
    assert reloaded.domain_head is not None
    assert reloaded.domain_source_vocabulary == adapter.domain_source_vocabulary
    assert reloaded.model_state_fingerprint() == adapter.model_state_fingerprint()
    for p_before, p_after in zip(adapter.domain_head.state_dict().values(), reloaded.domain_head.state_dict().values()):
        assert torch.equal(p_before, p_after)


def test_missing_domain_head_checkpoint_rejected_when_strategy_domain_adversarial(tmp_path):
    import json
    cfg = {"strategy": "domain_adversarial",
           "adversarial": {"enabled": True, "weight": 0.1, "gradient_reversal_lambda": 1.0, "warmup_epochs": 0}}
    ctx, adapter, _ = _fit_adapter(domain_robustness_config=cfg)
    bundle_dir = tmp_path / "bundle_missing_head"
    adapter.save_bundle(bundle_dir, ctx.preprocessing_artifact)

    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["domain_head_checkpoint"] = None
    manifest["bundle_fingerprint"] = _refingerprint(manifest)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(BundleValidationError):
        PathwayHierarchicalAdapter.load_bundle(bundle_dir, device="cpu")


def test_altered_domain_head_weight_rejected(tmp_path):
    cfg = {"strategy": "domain_adversarial",
           "adversarial": {"enabled": True, "weight": 0.1, "gradient_reversal_lambda": 1.0, "warmup_epochs": 0}}
    ctx, adapter, _ = _fit_adapter(domain_robustness_config=cfg)
    bundle_dir = tmp_path / "bundle_altered_head"
    adapter.save_bundle(bundle_dir, ctx.preprocessing_artifact)

    head_path = bundle_dir / "domain_head_checkpoint.pt"
    state = torch.load(head_path)
    for k in state:
        state[k] = state[k] + 1.0
    torch.save(state, head_path)

    with pytest.raises(BundleValidationError):
        PathwayHierarchicalAdapter.load_bundle(bundle_dir, device="cpu")


def test_altered_model_checkpoint_rejected(tmp_path):
    ctx, adapter, _ = _fit_adapter()
    bundle_dir = tmp_path / "bundle_altered_model"
    adapter.save_bundle(bundle_dir, ctx.preprocessing_artifact)

    ckpt_path = bundle_dir / "model_checkpoint.pt"
    state = torch.load(ckpt_path)
    for k in state:
        state[k] = state[k] + 1.0
    torch.save(state, ckpt_path)

    with pytest.raises(BundleValidationError):
        PathwayHierarchicalAdapter.load_bundle(bundle_dir, device="cpu")


def test_legacy_checkpoint_without_manifest_rejected(tmp_path):
    ctx, adapter, _ = _fit_adapter()
    bundle_dir = tmp_path / "legacy_dir"
    bundle_dir.mkdir()
    torch.save(adapter.model.state_dict(), bundle_dir / "model_checkpoint.pt")
    with pytest.raises(LegacyBundleError):
        PathwayHierarchicalAdapter.load_bundle(bundle_dir, device="cpu")


def _refingerprint(manifest: dict) -> str:
    import hashlib
    import json
    payload = {k: v for k, v in manifest.items() if k != "bundle_fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
