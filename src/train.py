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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
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
        malignancy_known:  Optional[np.ndarray] = None,  # [N]  bool, False if absent (see labellers.py)
        subject_ids:       Optional[np.ndarray] = None,  # [N]  str — required unless diagnostic_mode
        split_name:        Optional[str] = None,          # "train" | "val" | "test" | None
        dataset_source:    Optional[np.ndarray] = None,  # [N]  str, e.g. GEO accession per cell
        diagnostic_mode:   bool = False,
    ):
        """
        subject_ids is required for any dataset used in real training/
        evaluation: without it, nothing can verify that a subject's cells
        weren't split across train/val/test (see subset_by_subjects,
        assert_disjoint_subjects). Pass diagnostic_mode=True only to build a
        purely synthetic dataset (e.g. a smoke test with random data and no
        real subjects) — in that mode missing/"unknown" subject_ids are
        allowed and no disjointness guarantee is implied.
        """
        n = len(gene_matrix)
        self.X     = torch.FloatTensor(gene_matrix)
        self.smoke = torch.LongTensor(smoke_labels)
        self.malig = torch.FloatTensor(malignancy_labels)
        self.ctype = torch.LongTensor(cell_type_ids)
        self.dose  = torch.FloatTensor(
            exposure_dose if exposure_dose is not None
            else np.full(n, DOSE_UNKNOWN, dtype=np.float32)
        )
        self.malig_known = torch.BoolTensor(
            malignancy_known if malignancy_known is not None
            else np.zeros(n, dtype=bool)
        )

        self.diagnostic_mode = diagnostic_mode
        if subject_ids is None:
            if not diagnostic_mode:
                raise ValueError(
                    "CellLevelDataset requires subject_ids for real training/evaluation "
                    "— without them, leakage across train/val/test splits cannot be "
                    "verified. Pass diagnostic_mode=True only for synthetic smoke tests."
                )
            subject_ids = np.full(n, "unknown", dtype=object)
        subject_ids = np.array([str(s) for s in subject_ids], dtype=object)
        if not diagnostic_mode:
            missing = np.isin(subject_ids, ["unknown", "", "none", "None", "nan"])
            if missing.any():
                raise ValueError(
                    f"{int(missing.sum())}/{n} cell(s) have a missing or 'unknown' "
                    "subject_id — real training/evaluation requires a real subject_id "
                    "per cell. Pass diagnostic_mode=True only for synthetic smoke tests."
                )
        self.subject_ids = subject_ids
        self.split_name = split_name
        self.dataset_source = (
            np.array([str(s) for s in dataset_source], dtype=object)
            if dataset_source is not None else np.full(n, "unknown", dtype=object)
        )

    def __len__(self):  return len(self.X)

    def __getitem__(self, idx):
        return {
            "x":               self.X[idx],
            "smoke_label":     self.smoke[idx],
            "malignancy_label":self.malig[idx],
            "malignancy_known": self.malig_known[idx],
            "cell_type_id":    self.ctype[idx],
            "exposure_dose":   self.dose[idx],
            "subject_id":      self.subject_ids[idx],
        }

    def subset_by_subjects(self, subject_list) -> "CellLevelDataset":
        """
        Build a new CellLevelDataset containing only cells whose subject_id
        is in subject_list, preserving cell order and label alignment. This
        is the ONLY sanctioned way to derive a train/val/test cell dataset
        from a manifest — never a random per-cell split.
        """
        wanted = {str(s) for s in subject_list}
        mask = np.isin(self.subject_ids, list(wanted))
        return CellLevelDataset(
            gene_matrix       = self.X[mask].numpy(),
            smoke_labels      = self.smoke[mask].numpy(),
            malignancy_labels = self.malig[mask].numpy(),
            cell_type_ids     = self.ctype[mask].numpy(),
            exposure_dose     = self.dose[mask].numpy(),
            malignancy_known  = self.malig_known[mask].numpy(),
            subject_ids       = self.subject_ids[mask],
            dataset_source    = self.dataset_source[mask],
            diagnostic_mode   = self.diagnostic_mode,
        )

    @classmethod
    def from_dir(
        cls, processed_dir: Union[str, Path], diagnostic_mode: bool = False,
    ) -> "CellLevelDataset":
        """Load directly from the directory written by export_cell_dataset()."""
        import pandas as pd
        d = Path(processed_dir)
        dose_path = d / "exposure_dose.npy"
        malig_known_path = d / "malignancy_known.npy"
        meta_path = d / "cell_metadata.csv"

        subject_ids = None
        dataset_source = None
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            if "subject_id" in meta.columns:
                subject_ids = meta["subject_id"].astype(str).values
            if "source" in meta.columns:
                dataset_source = meta["source"].astype(str).values

        return cls(
            gene_matrix       = np.load(d / "gene_matrix.npy"),
            smoke_labels      = np.load(d / "smoke_labels.npy"),
            malignancy_labels = np.load(d / "malignancy_labels.npy"),
            cell_type_ids     = np.load(d / "cell_type_ids.npy"),
            exposure_dose     = np.load(dose_path) if dose_path.exists() else None,
            malignancy_known  = np.load(malig_known_path) if malig_known_path.exists() else None,
            subject_ids       = subject_ids,
            dataset_source    = dataset_source,
            diagnostic_mode   = diagnostic_mode,
        )

    def smoke_class_weights(self, num_classes: int = N_SMOKE_CLASSES) -> torch.Tensor:
        """
        Inverse-frequency class weights for CrossEntropyLoss, so majority
        classes (e.g. cigarette=83, dual_use=30 samples in the current real
        merge) don't drown out minority ones (vape=7, cannabis=6, cigar=1,
        unexposed=10) — the collapse documented in README.md's "Current
        results" table (macro-F1 0.27 despite 77% accuracy).

        weight[c] = n_samples / (num_classes * count[c]), the standard
        sklearn/PyTorch balanced-weight formula. Classes absent from this
        dataset get weight 0 (nothing to learn, and 1/0 would be inf).
        """
        counts = np.bincount(self.smoke.numpy(), minlength=num_classes).astype(np.float64)
        n = counts.sum()
        weights = np.zeros(num_classes, dtype=np.float32)
        present = counts > 0
        weights[present] = n / (num_classes * counts[present])
        return torch.FloatTensor(weights)


def assert_disjoint_subjects(*datasets, names: Optional[List[str]] = None) -> None:
    """
    Raise if any two datasets share a subject_id. Accepts CellLevelDataset
    instances, SubjectLevelDataset instances, or plain iterables of subject
    ids. This is the leakage guard actually invoked before training so a
    manifest bug or a manual dataset-construction mistake fails loudly
    instead of silently letting a subject's cells appear in two splits.
    """
    def _ids(ds) -> set:
        if isinstance(ds, CellLevelDataset):
            return set(ds.subject_ids.tolist())
        if isinstance(ds, SubjectLevelDataset):
            return {str(b["subject_id"]) for b in ds.bags}
        return {str(s) for s in ds}

    names = names or [f"dataset_{i}" for i in range(len(datasets))]
    id_sets = [_ids(ds) for ds in datasets]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            overlap = id_sets[i] & id_sets[j]
            if overlap:
                raise ValueError(
                    f"Subject leakage: {names[i]!r} and {names[j]!r} share "
                    f"{len(overlap)} subject(s): {sorted(overlap)[:10]}"
                    + (" ..." if len(overlap) > 10 else "")
                )


class SubjectLevelDataset(Dataset):
    """
    Wraps the list of bag dicts produced by assemble_subject_bags() (preprocess.py).
    Used in Phase 2/3 subject-level training.

    A bag with no real cancer outcome (assemble_subject_bags sets
    cancer_label_known=False when the subject was never matched to an
    NLST/TCGA outcome) carries no valid supervision signal for the cancer
    head. Training or evaluating against a fabricated label for those
    subjects is exactly the "unknown treated as negative" failure mode this
    dataset must not reproduce, so by default only known-outcome bags are
    kept. Pass require_known_outcome=False to keep every bag (e.g. for
    cell-level-only uses of the bag's smoke/malignancy arrays).
    """

    def __init__(self, bags: List[dict], require_known_outcome: bool = True):
        if require_known_outcome:
            known = [b for b in bags if b.get("cancer_label_known", b.get("cancer_label") is not None)]
            n_excluded = len(bags) - len(known)
            if n_excluded:
                print(f"[train] SubjectLevelDataset  excluded {n_excluded}/{len(bags)} "
                      "subjects with unknown cancer outcome from supervised training/eval")
            self.bags = known
        else:
            self.bags = bags
        self.n_excluded_unknown_outcome = (
            len(bags) - len(self.bags) if require_known_outcome else None
        )

    def __len__(self):  return len(self.bags)

    def __getitem__(self, idx):
        b = self.bags[idx]
        cancer_label = b.get("cancer_label")
        if cancer_label is None:
            raise ValueError(
                f"Subject {b.get('subject_id')} has unknown cancer_label but was "
                "included in a SubjectLevelDataset — construct with "
                "require_known_outcome=True (the default) to exclude it."
            )
        n = len(b["malig_labels"])
        malig_known = b.get("malig_known")
        return {
            "gene_matrix":   torch.FloatTensor(b["gene_matrix"]),
            "cell_type_ids": torch.LongTensor(b["cell_type_ids"]),
            "smoke_labels":  torch.LongTensor(b["smoke_labels"]),
            "malig_labels":  torch.FloatTensor(b["malig_labels"]),
            "malig_known":   torch.BoolTensor(malig_known if malig_known is not None else np.zeros(n, dtype=bool)),
            "cancer_label":  torch.FloatTensor([cancer_label]),
            "subject_id":    b["subject_id"],
        }


def subject_collate_fn(batch: list) -> list:
    """
    Identity collate — subjects have variable cell counts so they cannot
    be stacked. Each item in the DataLoader remains an individual dict.
    """
    return batch


def load_checkpoint_into(model: nn.Module, ckpt_path: Union[str, Path], device: str = "cpu") -> Dict:
    """
    Shared checkpoint loader used by Evaluator/Predictor/Trainer so all
    three understand both the structured format written by
    Trainer._save (format_version>=2: {"model_state_dict": ..., metadata...})
    and legacy bare state_dict checkpoints from before this format existed.
    Legacy checkpoints load with a printed warning and no metadata.
    Returns the full checkpoint dict (empty dict for legacy checkpoints).
    """
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        return obj
    print(f"[checkpoint] WARNING: {ckpt_path} is a legacy raw state_dict checkpoint "
          "(no split/preprocessing/seed metadata). Loading weights only.")
    model.load_state_dict(obj)
    return {}


# ─── MIL eligibility ──────────────────────────────────────────────────────────

class MILEligibilityError(ValueError):
    """Raised when a SubjectLevelDataset cannot support valid MIL training/eval."""


def check_mil_eligibility(
    subject_dataset: "SubjectLevelDataset",
    min_subjects: int = 10,
    min_positive: int = 2,
    min_negative: int = 2,
) -> Dict:
    """
    Subject-level MIL training/evaluation is only meaningful when there are
    enough independent subjects with a known outcome, and both outcome
    classes are represented — otherwise ROC-AUC/sensitivity/specificity are
    undefined or trivially perfect/degenerate. Called at the start of
    Trainer.phase2/phase3 so a too-small merge fails loudly instead of
    silently training a cancer head with best_auc stuck at 0.5.
    """
    labels = [subject_dataset[i]["cancer_label"].item() for i in range(len(subject_dataset))]
    n_pos = sum(l == 1.0 for l in labels)
    n_neg = sum(l == 0.0 for l in labels)
    n_total = len(labels)

    problems = []
    if n_total < min_subjects:
        problems.append(f"only {n_total} subjects with known cancer outcome (need >= {min_subjects})")
    if n_pos < min_positive:
        problems.append(f"only {n_pos} known-positive subjects (need >= {min_positive})")
    if n_neg < min_negative:
        problems.append(f"only {n_neg} known-negative subjects (need >= {min_negative})")

    report = {"n_total": n_total, "n_positive": n_pos, "n_negative": n_neg, "problems": problems}
    if problems:
        raise MILEligibilityError(
            "Subject-level MIL training/evaluation is not valid on this dataset: "
            + "; ".join(problems)
            + ". Provide more subjects with real cancer outcomes (NLST/TCGA linkage), "
              "or lower min_subjects/min_positive/min_negative if this is a deliberate "
              "small-scale diagnostic run."
        )
    return report


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

    Every phase takes EXPLICIT train and validation datasets, already
    subsetted from a subject-level SplitManifest (data/splitting.py) via
    CellLevelDataset.subset_by_subjects / the bag-filtering done when
    building SubjectLevelDataset per split. No phase ever internally
    re-splits a dataset — that was the source of subject-level leakage this
    class used to have (a subject's cells could land in both the internal
    train and validation partition). Test datasets are never accepted by
    any phaseN method; see final_test_evaluation() for the one sanctioned
    place test data is used, after training/checkpoint-selection is done.
    """

    def __init__(
        self,
        model:     MultiSmokeCancerNet,
        config:    dict,
        device:    str = "cpu",
        seed:      int = 42,
    ):
        self.model     = model.to(device)
        self.device    = device
        self.full_cfg  = config
        self.cfg       = config.get("train", config)
        ckpt_dir = self.cfg.get("checkpoint_dir", "checkpoints")
        self.ckpt_dir  = (
            Path(ckpt_dir) if Path(ckpt_dir).is_absolute()
            else Path(__file__).parents[1] / ckpt_dir
        )
        self.grad_clip = self.cfg.get("grad_clip", 1.0)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Experiment metadata carried into every saved checkpoint (section 8/22).
        self.split_manifest_path:      Optional[str] = None
        self.preprocessing_artifact_path: Optional[str] = None
        self.effective_label_mapping:  Optional[Dict] = None
        self.rare_class_policy:        Optional[str] = None
        self.transductive_batch_correction: bool = False

    @classmethod
    def from_config(
        cls,
        model:  MultiSmokeCancerNet,
        config: Union[dict, str, Path],
        device: str = "cpu",
        seed:   int = 42,
    ) -> "Trainer":
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        return cls(model, config, device, seed=seed)

    # ── Shared utilities ──────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        print(msg)

    def _generator(self) -> torch.Generator:
        """Deterministic per-call generator so DataLoader shuffling is reproducible."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        return g

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

    def _validate_train_val(
        self,
        train_ds, val_ds,
        train_name: str = "train", val_name: str = "val",
    ) -> None:
        """
        Fail loudly before training anything if the split is unusable:
        empty train/val, or (for real, non-diagnostic datasets) a subject
        shared between train and val. This is the enforcement point for
        "test/val data is never used for the wrong purpose" at the Trainer
        boundary — see assert_disjoint_subjects.
        """
        if len(train_ds) == 0:
            raise ValueError(f"Trainer: {train_name} dataset is empty — nothing to train on.")
        if len(val_ds) == 0:
            raise ValueError(f"Trainer: {val_name} dataset is empty — cannot select checkpoints.")
        diagnostic = getattr(train_ds, "diagnostic_mode", False) or getattr(val_ds, "diagnostic_mode", False)
        if not diagnostic:
            assert_disjoint_subjects(train_ds, val_ds, names=[train_name, val_name])

    def _save(self, phase: int, metric: float, metric_name: str = "metric", extra: Optional[Dict] = None) -> None:
        """
        Save a structured checkpoint: model_state_dict plus the metadata
        needed to reproduce or safely re-load this experiment later
        (section 8/22) — not just a bare state_dict.
        """
        import json, subprocess

        git_sha = None
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            pass

        payload = {
            "format_version":     2,
            "model_state_dict":   self.model.state_dict(),
            "model_config":       self.full_cfg.get("model", {}),
            "training_config":    self.cfg,
            "split_manifest_path": self.split_manifest_path,
            "preprocessing_artifact_path": self.preprocessing_artifact_path,
            "effective_label_mapping": self.effective_label_mapping,
            "rare_class_policy":  self.rare_class_policy,
            "transductive_batch_correction": self.transductive_batch_correction,
            "random_seed":        self.seed,
            "metric_name":        metric_name,
            "metric_value":       metric,
            "phase":              phase,
            "input_dim":          getattr(self.model, "input_dim", None),
            "git_commit_sha":     git_sha,
        }
        if extra:
            payload.update(extra)

        path = self.ckpt_dir / f"phase{phase}_best.pt"
        torch.save(payload, path)
        self._log(f"    ✓ checkpoint saved  ({metric_name}={metric:.4f})")

    def _load_best(self, phase: int, unsafe_legacy_mode: bool = False) -> Dict:
        """
        Load the best checkpoint for `phase`. Supports both the structured
        format written by _save() (format_version>=2) and bare state_dict
        checkpoints from before this change — loading the latter prints a
        clear warning and returns an empty metadata dict, since none of the
        reproducibility metadata exists for them. unsafe_legacy_mode is
        accepted for symmetry with Predictor's safe-inference gate but has
        no additional effect here (Trainer always loads what's on disk).
        """
        path = self.ckpt_dir / f"phase{phase}_best.pt"
        return load_checkpoint_into(self.model, path, self.device)

    def _save_history(self, history: Dict, name: str) -> None:
        """Persist training history to JSON for later plotting."""
        import json
        path = self.ckpt_dir / f"{name}_history.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def _make_optimizer(self, params, lr: float, wd: float = 1e-4):
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    # ── Phase 1: Cell-level pre-training ─────────────────────────────────────

    def phase1(
        self,
        train_cell_dataset: CellLevelDataset,
        val_cell_dataset:   CellLevelDataset,
        smoke_class_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Train encoder + both heads on labeled single cells.
        Aggregator is NOT updated.

        train_cell_dataset / val_cell_dataset must already be disjoint
        subject-level subsets (CellLevelDataset.subset_by_subjects against a
        SplitManifest) — this method does not split anything itself.

        smoke_class_weights defaults to inverse-frequency weights computed
        from TRAIN data only (CellLevelDataset.smoke_class_weights on
        train_cell_dataset) — pass an explicit tensor only to override that.
        Unweighted CE lets the loss minimize by predicting only the
        majority class(es), which is exactly the collapse README.md
        documents (77% accuracy, 0.27 macro-F1, zero F1 on vape/cannabis/
        cigar/unexposed).
        """
        self._log("\n=== Phase 1 — Cell-Level Pre-training ===")
        self._validate_train_val(train_cell_dataset, val_cell_dataset)

        epochs     = self.cfg.get("phase1_epochs",     15)
        lr         = self.cfg.get("phase1_lr",         1e-3)
        batch_size = self.cfg.get("phase1_batch_size", 512)

        if smoke_class_weights is None:
            smoke_class_weights = train_cell_dataset.smoke_class_weights()
            self._log(f"  smoke class weights (auto, inverse-freq, train-only): "
                       f"{[round(w, 3) for w in smoke_class_weights.tolist()]}")

        train_dl = DataLoader(train_cell_dataset, batch_size=batch_size, shuffle=True,
                               drop_last=len(train_cell_dataset) >= batch_size,
                               generator=self._generator())
        val_dl   = DataLoader(val_cell_dataset,   batch_size=batch_size, shuffle=False)

        loss_fn = MultiTaskLoss(
            lambda_smoke=0.50, lambda_malignancy=0.50,
            smoke_class_weights=smoke_class_weights.to(self.device),
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
        best_f1 = -1.0

        for epoch in range(1, epochs + 1):
            # train
            self.model.train()
            for batch in train_dl:
                x      = batch["x"].to(self.device)
                smoke_t= batch["smoke_label"].to(self.device)
                malig_t= batch["malignancy_label"].to(self.device)
                malig_k= batch["malignancy_known"].to(self.device)
                dose_t = batch["exposure_dose"].to(self.device)
                z, logits, malig = self.model.forward_cell(x)
                loss, _ = loss_fn.cell_level_loss(logits, smoke_t, malig, malig_t, malig_known=malig_k)
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
            f1  = f1_score(targets, preds, average="macro", zero_division=0)
            history.append({"epoch": epoch, "smoke_acc": acc, "smoke_macro_f1": f1})
            self._log(f"  epoch {epoch:02d}/{epochs}  smoke_acc={acc:.3f}  smoke_macro_f1={f1:.3f}")

            # Select on macro-F1, not accuracy: accuracy rewards collapsing
            # onto majority classes (see README's "Current results" table),
            # macro-F1 doesn't.
            if f1 > best_f1:
                best_f1 = f1
                self._save(1, f1, metric_name="val_smoke_macro_f1")

        self._load_best(1)
        self._save_history({"phase1": history}, "phase1")
        self._log(f"Phase 1 done.  best smoke_macro_f1={best_f1:.3f}")
        return {"history": history, "best_smoke_macro_f1": best_f1}

    # ── Phase 2: Aggregator training (encoder frozen) ─────────────────────────

    def phase2(
        self,
        train_subject_dataset: SubjectLevelDataset,
        val_subject_dataset:   SubjectLevelDataset,
        skip_eligibility_check: bool = False,
    ) -> Dict:
        """
        Freeze encoder + heads. Train only the MIL aggregator on
        subject-level cancer outcomes (NLST / TCGA labels).

        train_subject_dataset / val_subject_dataset must already be
        disjoint subject-level subsets of a SplitManifest — this method
        does not split anything itself.

        MIL eligibility (check_mil_eligibility) is checked separately on
        BOTH train and val, since a phase can have enough total subjects
        while still having e.g. zero known-positive subjects in val, which
        would make val_AUC undefined all the same. Pass
        skip_eligibility_check=True only for a deliberate small-scale
        diagnostic run.
        """
        self._log("\n=== Phase 2 — Aggregator Training ===")
        self._validate_train_val(train_subject_dataset, val_subject_dataset)
        if not skip_eligibility_check:
            elig_tr = check_mil_eligibility(train_subject_dataset)
            elig_va = check_mil_eligibility(val_subject_dataset)
            self._log(f"  MIL eligibility ✓  train={elig_tr['n_total']} subjects "
                       f"({elig_tr['n_positive']} pos, {elig_tr['n_negative']} neg)  "
                       f"val={elig_va['n_total']} subjects "
                       f"({elig_va['n_positive']} pos, {elig_va['n_negative']} neg)")

        epochs = self.cfg.get("phase2_epochs", 12)
        lr     = self.cfg.get("phase2_lr",     5e-4)

        # freeze everything except the aggregator
        for p in [*self.model.encoder.parameters(),
                  *self.model.smoke_head.parameters(),
                  *self.model.malignancy_head.parameters()]:
            p.requires_grad = False

        train_dl = DataLoader(train_subject_dataset, batch_size=1, shuffle=True,
                               collate_fn=subject_collate_fn, generator=self._generator())
        val_dl   = DataLoader(val_subject_dataset,   batch_size=1, shuffle=False,
                               collate_fn=subject_collate_fn)

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
                self._save(2, auc, metric_name="val_cancer_auc")
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
        train_cell_dataset:    CellLevelDataset,
        val_cell_dataset:      CellLevelDataset,
        train_subject_dataset: SubjectLevelDataset,
        val_subject_dataset:   SubjectLevelDataset,
        skip_eligibility_check: bool = False,
    ) -> Dict:
        """
        All layers unfrozen. Jointly optimises all three loss terms.
        Alternates cell-level and subject-level steps each iteration.
        Early stopping on subject-level validation AUC.

        All four datasets must already be disjoint subject-level subsets of
        the SAME SplitManifest used by phase1/phase2 — this method does not
        split anything itself. See phase2's docstring for the MIL
        eligibility check semantics (checked on train and val separately).
        """
        self._log("\n=== Phase 3 — End-to-End Fine-Tuning ===")
        self._validate_train_val(train_cell_dataset, val_cell_dataset,
                                  "train_cell", "val_cell")
        self._validate_train_val(train_subject_dataset, val_subject_dataset,
                                  "train_subject", "val_subject")
        if not skip_eligibility_check:
            elig_tr = check_mil_eligibility(train_subject_dataset)
            elig_va = check_mil_eligibility(val_subject_dataset)
            self._log(f"  MIL eligibility ✓  train={elig_tr['n_total']} subjects "
                       f"({elig_tr['n_positive']} pos, {elig_tr['n_negative']} neg)  "
                       f"val={elig_va['n_total']} subjects "
                       f"({elig_va['n_positive']} pos, {elig_va['n_negative']} neg)")

        epochs = self.cfg.get("phase3_epochs", 8)
        lr     = self.cfg.get("phase3_lr",     1e-4)

        cell_dl  = DataLoader(train_cell_dataset, batch_size=256, shuffle=True,
                               drop_last=len(train_cell_dataset) >= 256,
                               generator=self._generator())
        sub_dl   = DataLoader(train_subject_dataset, batch_size=1, shuffle=True,
                               collate_fn=subject_collate_fn, generator=self._generator())
        val_dl   = DataLoader(val_subject_dataset,   batch_size=1, shuffle=False,
                               collate_fn=subject_collate_fn)

        loss_fn = MultiTaskLoss(
            lambda_smoke=0.30, lambda_malignancy=0.30, lambda_subject=0.40,
            smoke_class_weights=train_cell_dataset.smoke_class_weights().to(self.device),
        )
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
                    malig_k= cb["malignancy_known"].to(self.device)
                    _, logits, malig = self.model.forward_cell(x)
                    cl, _  = loss_fn.cell_level_loss(logits, smoke_t, malig, malig_t, malig_known=malig_k)
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
                self._save(3, auc, metric_name="val_cancer_auc")
            if stopper.step(auc):
                self._log("  early stop")
                break

        self._load_best(3)
        self._save_history({"phase3": history}, "phase3")
        self._log(f"Phase 3 done.  best_AUC={best_auc:.3f}")
        return {"history": history, "best_auc": best_auc}

    # ── Final held-out test evaluation ────────────────────────────────────────

    def final_test_evaluation(
        self,
        test_cell_dataset:    CellLevelDataset,
        test_subject_dataset: SubjectLevelDataset,
        phase: int = 3,
        out_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        The ONE sanctioned place test data is used: load the validation-
        selected checkpoint for `phase` and evaluate it ONCE on the
        untouched test split. Never call this more than once per experiment
        and never feed its output back into model/hyperparameter/threshold
        selection — doing so turns the test set into a second validation
        set and invalidates the "held out" label on the result.

        Returns evaluate.py's cell_level/subject_level metrics wrapped with
        describe_split()-style provenance so the result is unambiguously
        labeled held-out test performance and can't be confused with
        training-set or validation-set numbers.
        """
        from evaluate import Evaluator  # local import — evaluate.py imports from this module

        self._load_best(phase)
        ev = Evaluator(self.model, self.device)
        report, raw = ev.full_report(
            test_cell_dataset, test_subject_dataset,
            out_dir=out_dir or self.ckpt_dir,
        )
        report["provenance"] = {
            "split_name":  "test",
            "is_held_out": True,
            "checkpoint":  f"phase{phase}_best.pt",
            "note": "Final held-out test evaluation — run once, not used for model/"
                    "threshold/hyperparameter selection.",
        }
        self._log("[train] final_test_evaluation complete — this is a HELD-OUT TEST result, "
                   "not a validation or training-set number.")
        return report

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
    # ── SYNTHETIC DIAGNOSTIC RUN ────────────────────────────────────────────
    # Random data, random labels, no real subjects. This exercises the
    # Trainer plumbing (subject-disjoint train/val datasets, checkpointing,
    # inference) end-to-end quickly. It is NOT a real training run and its
    # metrics are meaningless — see evaluate.py's __main__ for the same
    # caveat spelled out for evaluation metrics. Real experiments must
    # build train/val/test datasets from preprocess.py::run_pipeline_split_aware
    # and a real SplitManifest, never from this block.
    import random
    from data.splitting import subject_train_val_test_split

    torch.manual_seed(42)
    np.random.seed(42)

    GENES, N_SUBJ, CELLS_PER_SUBJ = 2000, 40, 25
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    subject_ids_list = [f"sub_{i}" for i in range(N_SUBJ)]
    manifest = subject_train_val_test_split(
        subject_ids_list, labels=None, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42,
    )

    # Synthetic per-cell data — half the cells carry a known exposure dose, so
    # Phase 1 actually exercises the dose-response head + loss end to end.
    n_cells = N_SUBJ * CELLS_PER_SUBJ
    cell_subject_ids = np.repeat(subject_ids_list, CELLS_PER_SUBJ)
    dose = np.where(
        np.random.rand(n_cells) < 0.5,
        np.random.rand(n_cells).astype("float32"),
        DOSE_UNKNOWN,
    ).astype("float32")
    full_cell_ds = CellLevelDataset(
        gene_matrix       = np.random.randn(n_cells, GENES).astype("float32"),
        smoke_labels       = np.random.randint(0, N_SMOKE_CLASSES, n_cells),
        malignancy_labels  = np.random.randint(0, 2, n_cells).astype("float32"),
        cell_type_ids      = np.random.randint(0, N_CELL_TYPES, n_cells),
        exposure_dose      = dose,
        subject_ids        = cell_subject_ids,
    )
    train_cell_ds = full_cell_ds.subset_by_subjects(manifest.train_subjects)
    val_cell_ds   = full_cell_ds.subset_by_subjects(manifest.val_subjects)
    test_cell_ds  = full_cell_ds.subset_by_subjects(manifest.test_subjects)

    def _bag(n):
        return {
            "gene_matrix":   np.random.randn(n, GENES).astype("float32"),
            "cell_type_ids": np.random.randint(0, N_CELL_TYPES,    n),
            "smoke_labels":  np.random.randint(0, N_SMOKE_CLASSES, n),
            "malig_labels":  np.random.randint(0, 2, n).astype("float32"),
            "cancer_label":  random.randint(0, 1),
            "cancer_label_known": True,
        }

    all_bags = {sid: {"subject_id": sid, **_bag(random.randint(30, 80))} for sid in subject_ids_list}
    train_subject_ds = SubjectLevelDataset([all_bags[s] for s in manifest.train_subjects])
    val_subject_ds   = SubjectLevelDataset([all_bags[s] for s in manifest.val_subjects])
    test_subject_ds  = SubjectLevelDataset([all_bags[s] for s in manifest.test_subjects])

    CFG = Path(__file__).parents[1] / "configs" / "default.yaml"
    model   = MultiSmokeCancerNet.from_config(CFG)
    trainer = Trainer.from_config(model, CFG, device)
    trainer.split_manifest_path = "synthetic-diagnostic-run (no manifest file saved)"

    # Override epochs for speed
    trainer.cfg.update({"phase1_epochs": 2, "phase2_epochs": 2, "phase3_epochs": 2})

    r1 = trainer.phase1(train_cell_ds, val_cell_ds)
    r2 = trainer.phase2(train_subject_ds, val_subject_ds, skip_eligibility_check=True)
    r3 = trainer.phase3(train_cell_ds, val_cell_ds, train_subject_ds, val_subject_ds,
                         skip_eligibility_check=True)
    test_report = trainer.final_test_evaluation(test_cell_ds, test_subject_ds, phase=3)
    print(f"\nfinal_test_evaluation provenance: {test_report['provenance']}")

    # Inference
    result = trainer.predict(
        np.random.randn(50, GENES).astype("float32"),
        np.random.randint(0, N_CELL_TYPES, 50),
    )
    print(f"\nInference:  P(cancer)={result['cancer_probability']:.4f}"
          f"  flag={result['risk_flag']}"
          f"  smoke={result['dominant_smoke_type']}")
    print("\n=== PASSED ===")