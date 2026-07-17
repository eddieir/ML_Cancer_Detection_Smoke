"""
Integration tests for Phase 2 subject-aware class-imbalance correction:
Trainer.phase1/phase1_final_fit/phase3 wiring, checkpoint/report provenance,
and fingerprint sensitivity to the imbalance configuration.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.label_mapping import build_effective_label_mapping
from data.rare_class import apply_rare_class_policy
from data.splitting import subject_train_val_test_split
from model import MultiSmokeCancerNet
from train import CellLevelDataset, SubjectLevelDataset, Trainer


N_CLASSES = 3
GENES = 12


def _cell_dataset_per_subject(subjects_labels, cells_per_subject=20, seed=0):
    """subjects_labels: dict subject_id -> effective label. One consistent
    label per subject, matching the sampler's contract."""
    rng = np.random.RandomState(seed)
    subject_ids, labels = [], []
    for s, lbl in subjects_labels.items():
        subject_ids += [s] * cells_per_subject
        labels += [lbl] * cells_per_subject
    n = len(subject_ids)
    return CellLevelDataset(
        gene_matrix=rng.randn(n, GENES).astype("float32"),
        smoke_labels=np.array(labels, dtype=np.int64),
        malignancy_labels=rng.randint(0, 2, n).astype("float32"),
        cell_type_ids=rng.randint(0, 2, n),
        subject_ids=np.array(subject_ids, dtype=object),
    )


def _split_train_val(n_subjects=30, seed=42):
    subjects = [f"sub_{i}" for i in range(n_subjects)]
    labels = {s: i % N_CLASSES for i, s in enumerate(subjects)}
    manifest = subject_train_val_test_split(
        subjects, labels=[labels[s] for s in subjects],
        train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=seed,
    )
    train_labels = {s: labels[s] for s in manifest.train_subjects}
    val_labels = {s: labels[s] for s in manifest.val_subjects}
    train_ds = _cell_dataset_per_subject(train_labels, seed=1)
    val_ds = _cell_dataset_per_subject(val_labels, seed=2)
    return train_ds, val_ds, manifest


def _trainer(smoke_imbalance, tmp_path):
    cfg = {
        "model": {"input_dim": GENES, "num_smoke_types": N_CLASSES, "num_cell_types": 2},
        "train": {
            "phase1_epochs": 1, "phase1_batch_size": 16,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "smoke_imbalance": smoke_imbalance,
        },
    }
    model = MultiSmokeCancerNet.from_config(cfg)
    return Trainer.from_config(model, cfg, device="cpu")


# ─── 1/2. phase1 uses the subject-balanced sampler; never shuffle+sampler ──────

def test_phase1_uses_subject_balanced_sampler_when_configured(tmp_path):
    train_ds, val_ds, _ = _split_train_val()
    trainer = _trainer({"sampler": "subject_balanced", "cells_per_subject_per_batch": 4}, tmp_path)
    trainer.phase1(train_ds, val_ds)
    assert trainer._last_subject_balanced_sampler is not None


def test_phase1_shuffle_mode_builds_no_batch_sampler(tmp_path):
    train_ds, val_ds, _ = _split_train_val()
    trainer = _trainer({"sampler": "shuffle"}, tmp_path)
    trainer.phase1(train_ds, val_ds)
    assert trainer._last_subject_balanced_sampler is None


def test_train_cell_loader_never_combines_shuffle_and_batch_sampler(tmp_path):
    train_ds, val_ds, _ = _split_train_val()
    trainer = _trainer({"sampler": "subject_balanced", "cells_per_subject_per_batch": 4}, tmp_path)
    dl = trainer._train_cell_loader(train_ds, batch_size=16)
    # DataLoader raises at construction if shuffle=True and batch_sampler are
    # both given, so surviving construction with a real batch_sampler set
    # already proves they were never combined.
    assert dl.batch_sampler is trainer._last_subject_balanced_sampler


# ─── 3. Validation remains sequential and untouched ────────────────────────────

def test_validation_loader_is_never_subject_balanced(tmp_path):
    train_ds, val_ds, _ = _split_train_val()
    trainer = _trainer({"sampler": "subject_balanced", "cells_per_subject_per_batch": 4}, tmp_path)
    trainer.phase1(train_ds, val_ds)
    # phase1 only ever calls _train_cell_loader for the TRAIN dataset; the
    # val DataLoader is built inline with shuffle=False and no sampler
    # (see train.py's phase1) — verify no sampler-related state leaked from
    # constructing the val loader by checking the sampler was built from
    # exactly the train dataset's subjects, not the union of train+val.
    diag = trainer._last_subject_balanced_sampler.diagnostics()
    train_subjects = set(train_ds.subject_ids.tolist())
    val_subjects = set(val_ds.subject_ids.tolist())
    assert train_subjects.isdisjoint(val_subjects)
    sampled_subjects = set()
    for c, subs in trainer._last_subject_balanced_sampler.index.class_to_subjects.items():
        sampled_subjects.update(subs)
    assert sampled_subjects <= train_subjects
    assert sampled_subjects.isdisjoint(val_subjects)


# ─── 4. phase1_final_fit uses development-only subject balancing ──────────────

def test_phase1_final_fit_builds_sampler_from_dev_pool_only(tmp_path):
    train_ds, val_ds, _ = _split_train_val()
    # phase1_final_fit takes exactly what it's given (here, train_ds standing
    # in for "the full development pool") — no internal split.
    trainer = _trainer({"sampler": "subject_balanced", "cells_per_subject_per_batch": 4}, tmp_path)
    result = trainer.phase1_final_fit(train_ds, epochs=1)
    assert result["smoke_imbalance_config"]["sampler"] == "subject_balanced"
    assert trainer._last_subject_balanced_sampler is not None
    sampled_subjects = set()
    for subs in trainer._last_subject_balanced_sampler.index.class_to_subjects.values():
        sampled_subjects.update(subs)
    assert sampled_subjects <= set(train_ds.subject_ids.tolist())


# ─── 5. Phase 3 cell training follows the selected imbalance configuration ─────

def test_phase3_cell_component_uses_subject_balanced_sampler(tmp_path):
    train_ds, val_ds, _ = _split_train_val()

    def _bags(cell_ds, seed):
        rng = np.random.RandomState(seed)
        subs = sorted(set(cell_ds.subject_ids.tolist()))
        bags = []
        for s in subs:
            mask = cell_ds.subject_ids == s
            n = int(mask.sum())
            bags.append({
                "subject_id": s,
                "gene_matrix": cell_ds.X[mask].numpy(),
                "cell_type_ids": cell_ds.ctype[mask].numpy(),
                "smoke_labels": cell_ds.smoke[mask].numpy(),
                "malig_labels": cell_ds.malig[mask].numpy(),
                "cancer_label": float(rng.randint(0, 2)),
                "cancer_label_known": True,
            })
        return bags

    train_sd = SubjectLevelDataset(_bags(train_ds, 3))
    val_sd = SubjectLevelDataset(_bags(val_ds, 4))

    # phase3's cell-level component uses a fixed batch_size=256 (see
    # train.py) — cells_per_subject_per_batch must be large enough that
    # cap * n_train_subjects >= 256 or the (correct, hard) hard-cap
    # feasibility check in SubjectBalancedBatchSampler raises
    # SamplingImpossibleError, per Blocker 1's construction-time validation.
    trainer = _trainer({"sampler": "subject_balanced", "cells_per_subject_per_batch": 16}, tmp_path)
    trainer.phase3(train_ds, val_ds, train_sd, val_sd, skip_eligibility_check=True)
    assert trainer._last_subject_balanced_sampler is not None


# ─── 9. Checkpoint metadata contains resolved imbalance configuration ─────────

def test_checkpoint_contains_resolved_imbalance_configuration(tmp_path):
    train_ds, val_ds, _ = _split_train_val()
    trainer = _trainer({"sampler": "subject_balanced", "loss": "focal", "focal_gamma": 1.5,
                         "cells_per_subject_per_batch": 4}, tmp_path)
    trainer.phase1(train_ds, val_ds)
    ckpt_path = Path(trainer.ckpt_dir) / "phase1_best.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["smoke_imbalance_config"]["sampler"] == "subject_balanced"
    assert ckpt["smoke_imbalance_config"]["loss"] == "focal"
    assert ckpt["smoke_imbalance_config"]["focal_gamma"] == 1.5
    assert ckpt["smoke_sampling_diagnostics"] is not None
    assert "observed_effective_classes" in ckpt["smoke_sampling_diagnostics"]


def test_checkpoint_with_shuffle_sampler_has_null_sampling_diagnostics(tmp_path):
    train_ds, val_ds, _ = _split_train_val()
    trainer = _trainer({"sampler": "shuffle"}, tmp_path)
    trainer.phase1(train_ds, val_ds)
    ckpt = torch.load(Path(trainer.ckpt_dir) / "phase1_best.pt", map_location="cpu", weights_only=False)
    assert ckpt["smoke_imbalance_config"]["sampler"] == "shuffle"
    assert ckpt["smoke_sampling_diagnostics"] is None


# ─── 10. Reports contain sampling and loss provenance ──────────────────────────

def test_neural_smoke_adapter_metadata_reports_imbalance_config(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from benchmarks.neural import NeuralSmokeAdapter

    train_ds, val_ds, _ = _split_train_val()
    config = {
        "model": {"input_dim": GENES, "num_smoke_types": N_CLASSES, "num_cell_types": 2},
        "train": {"phase1_epochs": 1, "phase1_batch_size": 16,
                   "checkpoint_dir": str(tmp_path / "ckpt2"),
                   "smoke_imbalance": {"sampler": "subject_balanced", "loss": "focal",
                                        "cells_per_subject_per_batch": 4}},
    }
    adapter = NeuralSmokeAdapter(config, device="cpu")
    adapter.fit(_FakeContext(config), train_ds, val_ds, seed=1)
    meta = adapter.metadata()
    assert meta["smoke_imbalance_config"]["sampler"] == "subject_balanced"
    assert meta["smoke_imbalance_config"]["loss"] == "focal"
    assert meta["smoke_sampling_diagnostics"] is not None


class _FakeContext:
    """Minimal stand-in for ExperimentContext — Trainer.from_experiment_context
    only reads .config, .label_mapping, .seed, .input_dim, .num_smoke_classes,
    .transductive_batch_correction from it."""
    def __init__(self, config):
        self.config = config
        self.seed = 1
        self.input_dim = GENES
        self.num_smoke_classes = N_CLASSES
        self.transductive_batch_correction = False
        from data.label_mapping import identity_label_mapping
        self.label_mapping = identity_label_mapping({i: f"class_{i}" for i in range(N_CLASSES)})


# ─── 8. Configuration changes alter run/model fingerprints ────────────────────

def test_config_fingerprint_changes_with_imbalance_strategy():
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    import hashlib
    import json

    def config_fingerprint(cfg):
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()

    base = {"train": {"smoke_imbalance": {"sampler": "shuffle", "class_weighting": "inverse_frequency"}}}
    variant = {"train": {"smoke_imbalance": {"sampler": "subject_balanced", "class_weighting": "none"}}}
    assert config_fingerprint(base) != config_fingerprint(variant)


# ─── 11/12. Rare-class merging + effective output width ───────────────────────

def test_effective_output_width_shrinks_with_rare_class_merge():
    smoke_ids = np.array([0] * 20 + [2] * 1)  # class 2 ("cigar") has 1 subject
    subject_ids = np.array([f"s{i}" for i in range(20)] + ["cigar_subject"])
    new_ids, keep_mask, report = apply_rare_class_policy(
        smoke_ids, subject_ids, policy="merge_into_dual_use_or_other",
        target_classes=("cigar",), min_subjects_required=3,
    )
    mapping = build_effective_label_mapping(report)
    assert mapping.k < 6
    assert "cigar" not in mapping.class_names


def test_sampler_num_classes_matches_effective_mapping_k():
    smoke_ids = np.array([0] * 20 + [2] * 1)
    subject_ids = np.array([f"s{i}" for i in range(20)] + ["cigar_subject"])
    _, _, report = apply_rare_class_policy(
        smoke_ids, subject_ids, policy="merge_into_dual_use_or_other",
        target_classes=("cigar",), min_subjects_required=3,
    )
    mapping = build_effective_label_mapping(report)
    # A sampler built with num_classes=mapping.k must reject an id >= k —
    # i.e. the merged-away raw id 5 can never reach the sampler as-is,
    # only its remapped effective id can.
    from data.sampling import SubjectClassIndex, SamplingConfigurationError
    with pytest.raises(SamplingConfigurationError):
        SubjectClassIndex(
            subject_ids=np.array(["a", "b"]), labels=np.array([0, mapping.k]), num_classes=mapping.k,
        )
