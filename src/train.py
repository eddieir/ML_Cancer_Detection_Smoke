"""
train.py — Three-phase curriculum training for MultiSmokeCancerNet.

Phase 1 : cell-level   — encoder + both heads
Phase 2 : subject-level — aggregator only (encoder frozen)
Phase 3 : end-to-end  — all layers, joint loss, early stopping
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

from constants import N_CELL_TYPES, N_SMOKE_CLASSES, SMOKE_TYPES, DOSE_UNKNOWN
from model import MultiSmokeCancerNet, MultiTaskLoss


# ─── Datasets ─────────────────────────────────────────────────────────────────

class CellLevelDataset(Dataset):
    """
    Wraps numpy arrays produced by export_cell_dataset() (preprocess.py).
    Used in Phase 1 cell-level pre-training.
    """

    def __init__(
        self,
        gene_matrix:       np.ndarray,             # [N, genes]  float32
        smoke_labels:      np.ndarray,              # [N]         int64
        malignancy_labels: np.ndarray,              # [N]         float32
        cell_type_ids:     np.ndarray,              # [N]         int64
        exposure_dose:     Optional[np.ndarray] = None,  # [N]  float32, DOSE_UNKNOWN if absent
    ):
        self.X     = torch.FloatTensor(gene_matrix)
        self.smoke = torch.LongTensor(smoke_labels)
        self.malig = torch.FloatTensor(malignancy_labels)
        self.ctype = torch.LongTensor(cell_type_ids)
        self.dose  = torch.FloatTensor(
            exposure_dose if exposure_dose is not None
            else np.full(len(gene_matrix), DOSE_UNKNOWN, dtype=np.float32)
        )

    def __len__(self):  return len(self.X)

    def __getitem__(self, idx):
        return {
            "x":               self.X[idx],
            "smoke_label":     self.smoke[idx],
            "malignancy_label":self.malig[idx],
            "cell_type_id":    self.ctype[idx],
            "exposure_dose":   self.dose[idx],
        }

    @classmethod
    def from_dir(cls, processed_dir: Union[str, Path]) -> "CellLevelDataset":
        """Load directly from the directory written by export_cell_dataset()."""
        d = Path(processed_dir)
        dose_path = d / "exposure_dose.npy"
        return cls(
            gene_matrix       = np.load(d / "gene_matrix.npy"),
            smoke_labels      = np.load(d / "smoke_labels.npy"),
            malignancy_labels = np.load(d / "malignancy_labels.npy"),
            cell_type_ids     = np.load(d / "cell_type_ids.npy"),
            exposure_dose     = np.load(dose_path) if dose_path.exists() else None,
        )


class SubjectLevelDataset(Dataset):
    """
    Wraps the list of bag dicts produced by assemble_subject_bags() (preprocess.py).
    Used in Phase 2/3 subject-level training.
    """

    def __init__(self, bags: List[dict]):
        self.bags = bags

    def __len__(self):  return len(self.bags)

    def __getitem__(self, idx):
        b = self.bags[idx]
        return {
            "gene_matrix":   torch.FloatTensor(b["gene_matrix"]),
            "cell_type_ids": torch.LongTensor(b["cell_type_ids"]),
            "smoke_labels":  torch.LongTensor(b["smoke_labels"]),
            "malig_labels":  torch.FloatTensor(b["malig_labels"]),
            "cancer_label":  torch.FloatTensor([b.get("cancer_label", 0)]),
            "subject_id":    b["subject_id"],
        }


def subject_collate_fn(batch: list) -> list:
    """
    Identity collate — subjects have variable cell counts so they cannot
    be stacked. Each item in the DataLoader remains an individual dict.
    """
    return batch


# ─── Early stopping ───────────────────────────────────────────────────────────

class _EarlyStopper:
    """Stops training when a monitored metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience  = patience
        self.mode      = mode
        self.best      = -float("inf") if mode == "max" else float("inf")
        self.no_improve= 0

    def step(self, metric: float) -> bool:
        improved = (metric > self.best) if self.mode == "max" else (metric < self.best)
        if improved:
            self.best       = metric
            self.no_improve = 0
        else:
            self.no_improve += 1
        return self.no_improve >= self.patience   # True → stop


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """
    Orchestrates all three training phases.

    DRY utilities (_grad_step, _save, _load_best, _log) are shared
    across all phases — no repeated optimizer / checkpoint logic.
    """

    def __init__(
        self,
        model:     MultiSmokeCancerNet,
        config:    dict,
        device:    str = "cpu",
    ):
        self.model     = model.to(device)
        self.device    = device
        self.cfg       = config.get("train", config)
        ckpt_dir = self.cfg.get("checkpoint_dir", "checkpoints")
        self.ckpt_dir  = (
            Path(ckpt_dir) if Path(ckpt_dir).is_absolute()
            else Path(__file__).parents[1] / ckpt_dir
        )
        self.grad_clip = self.cfg.get("grad_clip", 1.0)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        model:  MultiSmokeCancerNet,
        config: Union[dict, str, Path],
        device: str = "cpu",
    ) -> "Trainer":
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        return cls(model, config, device)

    # ── Shared utilities ──────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        print(msg)

    def _grad_step(
        self,
        loss:      torch.Tensor,
        optimizer: torch.optim.Optimizer,
        params,
    ) -> None:
        """Backward + gradient clip + optimizer step."""
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, self.grad_clip)
        optimizer.step()

    def _save(self, phase: int, metric: float) -> None:
        path = self.ckpt_dir / f"phase{phase}_best.pt"
        torch.save(self.model.state_dict(), path)
        self._log(f"    ✓ checkpoint saved  (metric={metric:.4f})")

    def _load_best(self, phase: int) -> None:
        path = self.ckpt_dir / f"phase{phase}_best.pt"
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )

    def _save_history(self, history: Dict, name: str) -> None:
        """Persist training history to JSON for later plotting."""
        import json
        path = self.ckpt_dir / f"{name}_history.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def _make_optimizer(self, params, lr: float, wd: float = 1e-4):
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def _split(self, ds: Dataset, val_frac: float):
        n_val   = max(1, int(len(ds) * val_frac))
        n_train = len(ds) - n_val
        return random_split(ds, [n_train, n_val])

    # ── Phase 1: Cell-level pre-training ─────────────────────────────────────

    def phase1(
        self,
        cell_dataset: CellLevelDataset,
        smoke_class_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Train encoder + both heads on labeled single cells.
        Aggregator is NOT updated.
        """
        self._log("\n=== Phase 1 — Cell-Level Pre-training ===")

        epochs     = self.cfg.get("phase1_epochs",     15)
        lr         = self.cfg.get("phase1_lr",         1e-3)
        batch_size = self.cfg.get("phase1_batch_size", 512)

        train_ds, val_ds = self._split(cell_dataset, 0.15)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        loss_fn = MultiTaskLoss(
            lambda_smoke=0.50, lambda_malignancy=0.50,
            smoke_class_weights=smoke_class_weights,
        )
        params  = [
            *self.model.encoder.parameters(),
            *self.model.smoke_head.parameters(),
            *self.model.malignancy_head.parameters(),
            *self.model.dose_head.parameters(),
        ]
        opt     = self._make_optimizer(params, lr)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lambda_dose = self.cfg.get("lambda_dose", 0.10)
        history = []
        best_acc= -1.0

        for epoch in range(1, epochs + 1):
            # train
            self.model.train()
            for batch in train_dl:
                x      = batch["x"].to(self.device)
                smoke_t= batch["smoke_label"].to(self.device)
                malig_t= batch["malignancy_label"].to(self.device)
                dose_t = batch["exposure_dose"].to(self.device)
                z, logits, malig = self.model.forward_cell(x)
                loss, _ = loss_fn.cell_level_loss(logits, smoke_t, malig, malig_t)
                dose_loss, _ = loss_fn.dose_response_loss(self.model.dose_head(z), dose_t, malig)
                self._grad_step(loss + lambda_dose * dose_loss, opt, params)
            sched.step()

            # validate
            self.model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in val_dl:
                    _, logits, _ = self.model.forward_cell(batch["x"].to(self.device))
                    preds.extend(logits.argmax(1).cpu().tolist())
                    targets.extend(batch["smoke_label"].tolist())

            acc = sum(p == t for p, t in zip(preds, targets)) / len(targets)
            history.append({"epoch": epoch, "smoke_acc": acc})
            self._log(f"  epoch {epoch:02d}/{epochs}  smoke_acc={acc:.3f}")

            if acc > best_acc:
                best_acc = acc
                self._save(1, acc)

        self._load_best(1)
        self._save_history({"phase1": history}, "phase1")
        self._log(f"Phase 1 done.  best smoke_acc={best_acc:.3f}")
        return {"history": history, "best_smoke_acc": best_acc}

    # ── Phase 2: Aggregator training (encoder frozen) ─────────────────────────

    def phase2(self, subject_dataset: SubjectLevelDataset) -> Dict:
        """
        Freeze encoder + heads. Train only the MIL aggregator on
        subject-level cancer outcomes (NLST / TCGA labels).
        """
        self._log("\n=== Phase 2 — Aggregator Training ===")

        epochs = self.cfg.get("phase2_epochs", 12)
        lr     = self.cfg.get("phase2_lr",     5e-4)

        # freeze everything except the aggregator
        for p in [*self.model.encoder.parameters(),
                  *self.model.smoke_head.parameters(),
                  *self.model.malignancy_head.parameters()]:
            p.requires_grad = False

        train_ds, val_ds = self._split(subject_dataset, 0.20)
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=subject_collate_fn)
        val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=subject_collate_fn)

        loss_fn = MultiTaskLoss()
        params  = list(self.model.aggregator.parameters())
        opt     = self._make_optimizer(params, lr)
        sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5)
        stopper = _EarlyStopper(patience=self.cfg.get("patience", 5))
        history = []
        best_auc= -1.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            for [item] in train_dl:
                x_bag  = item["gene_matrix"].to(self.device)
                ct_ids = item["cell_type_ids"].to(self.device)
                cancer = item["cancer_label"].to(self.device)
                out    = self.model.forward_subject(x_bag, ct_ids)
                loss, _= loss_fn.subject_level_loss(out["cancer_probability"], cancer)
                self._grad_step(loss, opt, params)

            auc = self._subject_auc(val_dl)
            sched.step(auc)
            history.append({"epoch": epoch, "val_auc": auc})
            self._log(f"  epoch {epoch:02d}/{epochs}  val_AUC={auc:.3f}")

            if auc > best_auc:
                best_auc = auc
                self._save(2, auc)
            if stopper.step(auc):
                self._log("  early stop")
                break

        # unfreeze for Phase 3
        for p in self.model.parameters():
            p.requires_grad = True

        self._load_best(2)
        self._save_history({"phase2": history}, "phase2")
        self._log(f"Phase 2 done.  best_AUC={best_auc:.3f}")
        return {"history": history, "best_auc": best_auc}

    # ── Phase 3: End-to-end fine-tuning ───────────────────────────────────────

    def phase3(
        self,
        cell_dataset:    CellLevelDataset,
        subject_dataset: SubjectLevelDataset,
    ) -> Dict:
        """
        All layers unfrozen. Jointly optimises all three loss terms.
        Alternates cell-level and subject-level steps each iteration.
        Early stopping on subject-level validation AUC.
        """
        self._log("\n=== Phase 3 — End-to-End Fine-Tuning ===")

        epochs = self.cfg.get("phase3_epochs", 8)
        lr     = self.cfg.get("phase3_lr",     1e-4)

        train_sub, val_sub = self._split(subject_dataset, 0.20)
        cell_dl  = DataLoader(cell_dataset, batch_size=256, shuffle=True, drop_last=True)
        sub_dl   = DataLoader(train_sub,    batch_size=1,   shuffle=True, collate_fn=subject_collate_fn)
        val_dl   = DataLoader(val_sub,      batch_size=1,   shuffle=False, collate_fn=subject_collate_fn)

        loss_fn = MultiTaskLoss(lambda_smoke=0.30, lambda_malignancy=0.30, lambda_subject=0.40)
        params  = list(self.model.parameters())
        opt     = self._make_optimizer(params, lr, wd=1e-5)
        stopper = _EarlyStopper(patience=self.cfg.get("patience", 5))
        history = []
        best_auc= -1.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            cell_iter = iter(cell_dl)
            sub_iter  = iter(sub_dl)
            n_steps   = max(len(cell_dl), len(sub_dl))

            for _ in range(n_steps):
                loss = torch.tensor(0.0, device=self.device)

                try:
                    cb = next(cell_iter)
                    x      = cb["x"].to(self.device)
                    smoke_t= cb["smoke_label"].to(self.device)
                    malig_t= cb["malignancy_label"].to(self.device)
                    _, logits, malig = self.model.forward_cell(x)
                    cl, _  = loss_fn.cell_level_loss(logits, smoke_t, malig, malig_t)
                    loss   = loss + cl * 0.5
                except StopIteration:
                    pass

                try:
                    [item] = next(sub_iter)
                    out    = self.model.forward_subject(
                        item["gene_matrix"].to(self.device),
                        item["cell_type_ids"].to(self.device),
                    )
                    sl, _  = loss_fn.subject_level_loss(
                        out["cancer_probability"],
                        item["cancer_label"].to(self.device),
                    )
                    loss   = loss + sl * 0.5
                except StopIteration:
                    pass

                if loss.requires_grad:
                    self._grad_step(loss, opt, params)

            auc = self._subject_auc(val_dl)
            history.append({"epoch": epoch, "val_auc": auc})
            self._log(f"  epoch {epoch:02d}/{epochs}  val_AUC={auc:.3f}")

            if auc > best_auc:
                best_auc = auc
                self._save(3, auc)
            if stopper.step(auc):
                self._log("  early stop")
                break

        self._load_best(3)
        self._save_history({"phase3": history}, "phase3")
        self._log(f"Phase 3 done.  best_AUC={best_auc:.3f}")
        return {"history": history, "best_auc": best_auc}

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        gene_matrix:   np.ndarray,    # [N, genes]  float32
        cell_type_ids: np.ndarray,    # [N]         int
    ) -> Dict:
        """Run inference on one subject. Returns cancer risk + interpretability."""
        self.model.eval()
        with torch.no_grad():
            out = self.model.forward_subject(
                torch.FloatTensor(gene_matrix).to(self.device),
                torch.LongTensor(cell_type_ids).to(self.device),
            )
        prob  = out["cancer_probability"].item()
        attn  = out["attention_weights"].cpu().numpy()
        smoke = out["cell_smoke_probs"].cpu().numpy().argmax(axis=1)

        return {
            "cancer_probability":  prob,
            "risk_flag":           "HIGH" if prob >= 0.7 else "MODERATE" if prob >= 0.4 else "LOW",
            "top5_cells":          attn.argsort()[-5:][::-1].tolist(),
            "dominant_smoke_type": SMOKE_TYPES[int(np.bincount(smoke).argmax())],
            "attention_weights":   attn,
        }

    # ── Private helper ────────────────────────────────────────────────────────

    def _subject_auc(self, loader: DataLoader) -> float:
        """ROC-AUC on subject cancer predictions. Falls back to 0.5 if only one class."""
        from sklearn.metrics import roc_auc_score
        probs, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for [item] in loader:
                out = self.model.forward_subject(
                    item["gene_matrix"].to(self.device),
                    item["cell_type_ids"].to(self.device),
                )
                probs.append(out["cancer_probability"].item())
                labels.append(item["cancer_label"].item())
        if len(set(labels)) < 2:
            return 0.5                             # can't compute AUC with one class
        return roc_auc_score(labels, probs)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    torch.manual_seed(42)
    np.random.seed(42)

    GENES, N_CELLS, N_SUBJ = 2000, 1000, 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Synthetic datasets — half the cells carry a known exposure dose, so
    # Phase 1 actually exercises the dose-response head + loss end to end.
    dose = np.where(
        np.random.rand(N_CELLS) < 0.5,
        np.random.rand(N_CELLS).astype("float32"),
        DOSE_UNKNOWN,
    ).astype("float32")
    cell_ds = CellLevelDataset(
        gene_matrix       = np.random.randn(N_CELLS, GENES).astype("float32"),
        smoke_labels      = np.random.randint(0, N_SMOKE_CLASSES, N_CELLS),
        malignancy_labels = np.random.randint(0, 2, N_CELLS).astype("float32"),
        cell_type_ids     = np.random.randint(0, N_CELL_TYPES, N_CELLS),
        exposure_dose     = dose,
    )
    def _bag(n):
        return {
            "gene_matrix":   np.random.randn(n, GENES).astype("float32"),
            "cell_type_ids": np.random.randint(0, N_CELL_TYPES,    n),
            "smoke_labels":  np.random.randint(0, N_SMOKE_CLASSES, n),
            "malig_labels":  np.random.randint(0, 2, n).astype("float32"),
            "cancer_label":  random.randint(0, 1),
        }

    subject_ds = SubjectLevelDataset([
        {"subject_id": f"sub_{i}", **_bag(random.randint(30, 80))}
        for i in range(N_SUBJ)
    ])

    CFG = Path(__file__).parents[1] / "configs" / "default.yaml"
    model   = MultiSmokeCancerNet.from_config(CFG)
    trainer = Trainer.from_config(model, CFG, device)

    # Override epochs for speed
    trainer.cfg.update({"phase1_epochs": 2, "phase2_epochs": 2, "phase3_epochs": 2})

    r1 = trainer.phase1(cell_ds)
    r2 = trainer.phase2(subject_ds)
    r3 = trainer.phase3(cell_ds, subject_ds)

    # Inference
    result = trainer.predict(
        np.random.randn(50, GENES).astype("float32"),
        np.random.randint(0, N_CELL_TYPES, 50),
    )
    print(f"\nInference:  P(cancer)={result['cancer_probability']:.4f}"
          f"  flag={result['risk_flag']}"
          f"  smoke={result['dominant_smoke_type']}")
    print("\n=== PASSED ===")