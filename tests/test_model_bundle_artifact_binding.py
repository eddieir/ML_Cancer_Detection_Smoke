"""
Phase 4 — checkpoint <-> preprocessing-artifact fingerprint binding.

Trainer._save() embeds the scientific_fingerprint() of the
PreprocessingArtifact it was wired with (see Trainer.set_preprocessing_artifact)
into the saved checkpoint. Predictor.from_config() must reject pairing that
checkpoint with a DIFFERENT preprocessing_artifact.json — pairing a
checkpoint with the wrong artifact silently reorders/rescales inference
input incorrectly and must never be allowed to pass quietly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.preprocessing import ArtifactCompatibilityError, fit_preprocessing
from inference import Predictor
from model import MultiSmokeCancerNet
from train import Trainer

GENES = 12


def _artifact(n_genes=GENES, seed=0, n_subjects=6):
    rng = np.random.default_rng(seed)
    subject_ids = []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * 10
    n = len(subject_ids)
    X = rng.random((n, n_genes)).astype("float32")
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame({"subject_id": subject_ids, "batch": ["b0"] * n}, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    return fit_preprocessing(adata, set(subject_ids), n_hvgs=n_genes)


def _train_and_save(tmp_path, artifact, num_smoke=6):
    model = MultiSmokeCancerNet.from_config(
        {"model": {"input_dim": len(artifact.gene_list), "num_smoke_types": num_smoke,
                    "embedding_dim": 16, "attention_dim": 8}},
    )
    trainer = Trainer(model, {"train": {"checkpoint_dir": str(tmp_path)}}, device="cpu", seed=0)
    trainer.set_preprocessing_artifact(artifact)
    trainer._save(phase=3, metric=0.5, metric_name="macro_f1")
    artifact.save(tmp_path / "preprocessing_artifact.json")
    return tmp_path


def test_matching_artifact_loads_successfully(tmp_path):
    artifact = _artifact(seed=1)
    ckpt_dir = _train_and_save(tmp_path, artifact)
    predictor = Predictor.from_config(
        {"model": {"input_dim": len(artifact.gene_list), "num_smoke_types": 6,
                    "embedding_dim": 16, "attention_dim": 8},
         "train": {"checkpoint_dir": str(ckpt_dir)}},
        phase=3,
    )
    assert predictor.preprocessing_artifact.scientific_fingerprint() == artifact.scientific_fingerprint()


def test_mismatched_artifact_is_rejected(tmp_path):
    """A checkpoint saved with artifact A must refuse to load if the
    preprocessing_artifact.json on disk is later swapped for a different,
    incompatible artifact B (same gene count so it does not fail the
    input_dim check first, different scientific content)."""
    artifact_a = _artifact(seed=2)
    ckpt_dir = _train_and_save(tmp_path, artifact_a)

    # Swap in a differently-fit artifact with the same gene count/order but
    # different scaling statistics (different seed -> different means/stds).
    artifact_b = _artifact(seed=3)
    assert artifact_b.gene_list == artifact_a.gene_list  # same synthetic panel
    assert artifact_b.scientific_fingerprint() != artifact_a.scientific_fingerprint()
    artifact_b.save(ckpt_dir / "preprocessing_artifact.json")

    with pytest.raises(ArtifactCompatibilityError):
        Predictor.from_config(
            {"model": {"input_dim": len(artifact_a.gene_list), "num_smoke_types": 6,
                        "embedding_dim": 16, "attention_dim": 8},
             "train": {"checkpoint_dir": str(ckpt_dir)}},
            phase=3,
        )


def test_trainer_rejects_artifact_with_wrong_gene_count():
    artifact = _artifact(seed=4, n_genes=GENES)
    model = MultiSmokeCancerNet.from_config(
        {"model": {"input_dim": GENES + 1, "num_smoke_types": 6, "embedding_dim": 16, "attention_dim": 8}},
    )
    trainer = Trainer(model, {"train": {}}, device="cpu", seed=0)
    with pytest.raises(ValueError):
        trainer.set_preprocessing_artifact(artifact)
