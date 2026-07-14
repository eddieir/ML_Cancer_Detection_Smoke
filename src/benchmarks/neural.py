"""
benchmarks/neural.py — adapters wrapping the existing Trainer/
MultiSmokeCancerNet so the real neural model can be compared against
baselines.py inside the same cross_validation.py runner.

Deliberately thin: all real training logic still lives in train.py's
Trainer.phase1/phase2 — these adapters only add a predict()/predict_proba()
surface comparable to baselines.Baseline, plus parameter-count/runtime
bookkeeping for the MIL pooling ablation (section 7).
"""

import time
from typing import Dict, List, Optional

import numpy as np
import torch

from train import CellLevelDataset, SubjectLevelDataset, Trainer


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


class NeuralSmokeAdapter:
    """Task A adapter: Trainer.phase1 (cell-level curriculum) + eval-mode predict."""

    name = "neural"

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.trainer: Optional[Trainer] = None
        self.fit_seconds: Optional[float] = None

    def fit(self, context, train_cell_dataset: CellLevelDataset,
            val_cell_dataset: CellLevelDataset, seed: int = 42) -> "NeuralSmokeAdapter":
        t0 = time.time()
        self.trainer = Trainer.from_experiment_context(context, device=self.device, seed=seed)
        self.trainer.phase1(train_cell_dataset, val_cell_dataset)
        self.fit_seconds = time.time() - t0
        return self

    def predict_proba(self, cell_dataset: CellLevelDataset) -> np.ndarray:
        self.trainer.model.eval()
        with torch.no_grad():
            x = cell_dataset.X.to(self.device)
            _, logits, _ = self.trainer.model.forward_cell(x)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, cell_dataset: CellLevelDataset) -> np.ndarray:
        return self.predict_proba(cell_dataset).argmax(axis=1)

    def metadata(self) -> Dict:
        return {
            "name": self.name,
            "n_parameters": count_parameters(self.trainer.model) if self.trainer else None,
            "fit_seconds": self.fit_seconds,
            "pooling": getattr(self.trainer.model, "pooling", None) if self.trainer else None,
        }


class NeuralCancerAdapter:
    """
    Task B adapter for the MIL pooling ablation (section 7): "attention"
    (current GatedAttentionMIL), "mean", or "max" — same encoder dimensions,
    same input features, same subject splits, selected via
    MultiSmokeCancerNet(pooling=...).

    fit() runs a short Phase 1 cell-level pretraining pass (so the encoder
    isn't producing random embeddings for the frozen-encoder aggregator
    training in Phase 2) followed by Phase 2 aggregator-only training — the
    same two curriculum stages train.py already validates, just with the
    pooling variant swapped in.
    """

    name = "mil"

    def __init__(self, pooling: str = "attention", device: str = "cpu"):
        self.pooling = pooling
        self.device = device
        self.trainer: Optional[Trainer] = None
        self.fit_seconds: Optional[float] = None

    def fit(
        self, context,
        train_cell_dataset: CellLevelDataset, val_cell_dataset: CellLevelDataset,
        train_subject_dataset: SubjectLevelDataset, val_subject_dataset: SubjectLevelDataset,
        seed: int = 42, pretrain_epochs: Optional[int] = None,
    ) -> "NeuralCancerAdapter":
        t0 = time.time()
        self.trainer = Trainer.from_experiment_context(
            context, device=self.device, seed=seed, pooling=self.pooling,
        )
        if pretrain_epochs is not None:
            self.trainer.cfg = dict(self.trainer.cfg)
            self.trainer.cfg["phase1_epochs"] = pretrain_epochs
        self.trainer.phase1(train_cell_dataset, val_cell_dataset)
        self.trainer.phase2(train_subject_dataset, val_subject_dataset)
        self.fit_seconds = time.time() - t0
        return self

    def fit_final(
        self, context,
        dev_cell_dataset: CellLevelDataset, dev_subject_dataset: SubjectLevelDataset,
        seed: int = 42, pretrain_epochs: Optional[int] = None, phase2_epochs: Optional[int] = None,
    ) -> "NeuralCancerAdapter":
        """
        The ONE final MIL fit for the frozen-test protocol (blocker 3): every
        subject in dev_cell_dataset/dev_subject_dataset contributes to
        gradient updates — unlike fit() above, there is no internal
        validation split held out for checkpoint selection. pretrain_epochs/
        phase2_epochs must already be decided from development-only nested-
        CV/OOF evidence before this runs (never selected here); None falls
        back to this context's configured defaults.
        """
        t0 = time.time()
        self.trainer = Trainer.from_experiment_context(
            context, device=self.device, seed=seed, pooling=self.pooling,
        )
        self.trainer.phase1_final_fit(dev_cell_dataset, epochs=pretrain_epochs)
        self.trainer.phase2_final_fit(dev_subject_dataset, epochs=phase2_epochs)
        self.fit_seconds = time.time() - t0
        return self

    def predict_proba(self, subject_dataset: SubjectLevelDataset) -> np.ndarray:
        self.trainer.model.eval()
        probs = []
        with torch.no_grad():
            for i in range(len(subject_dataset)):
                item = subject_dataset[i]
                out = self.trainer.model.forward_subject(
                    item["gene_matrix"].to(self.device), item["cell_type_ids"].to(self.device),
                )
                probs.append(out["cancer_probability"].item())
        return np.array(probs)

    def predict(self, subject_dataset: SubjectLevelDataset, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(subject_dataset) >= threshold).astype(int)

    def metadata(self) -> Dict:
        return {
            "name": self.name, "pooling": self.pooling,
            "n_parameters": count_parameters(self.trainer.model) if self.trainer else None,
            "n_aggregator_parameters": count_parameters(self.trainer.model.aggregator) if self.trainer else None,
            "fit_seconds": self.fit_seconds,
        }

    def model_state_fingerprint(self) -> str:
        """Deterministic SHA-256 of the actual fitted network weights — see
        model_fingerprint.py. Device placement and fit_seconds never affect
        this value; only the learned state_dict does."""
        from .model_fingerprint import torch_state_dict_fingerprint
        return torch_state_dict_fingerprint(self.trainer.model)


MIL_POOLINGS = ("attention", "mean", "max")
