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
from torch.utils.data import DataLoader, Dataset
import yaml

from constants import N_CELL_TYPES, N_SMOKE_CLASSES, SMOKE_TYPES, DOSE_UNKNOWN
from data.label_mapping import EffectiveLabelMapping
from data.sampling import (
    SubjectBalancedBatchSampler,
    build_subject_balanced_sampler,
    resolve_smoke_imbalance_config,
)
from metrics import multiclass_f1_report, validate_cell_type_ids
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
        smoke_known:       Optional[np.ndarray] = None,  # [N]  bool, True if absent (legacy sources — see labellers.py)
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

        smoke_known=None defaults to all-True — this is the legacy behaviour
        for every existing caller/source that has no verified/unknown
        distinction wired in yet, NOT a claim that every cell's smoke_type
        is actually verified. Sources that DO carry a real distinction (e.g.
        GSE136831's weak-proxy-only cells — see data/loaders.py::
        _attach_standard_obs, data/assembly.py::export_cell_dataset) pass a
        real per-cell array here. smoke_known=False cells still carry SOME
        integer value in self.smoke (there is no separate "unknown" slot in
        the fixed smoke-class space) — that value is a placeholder only,
        and every consumer of self.smoke (smoke_class_weights below,
        model.py's MultiTaskLoss._ls via the smoke_known mask threaded
        through train.py's loss calls, data.sampling's class index) must
        gate on smoke_known, never read the placeholder as if it were a
        real label.
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
        self.smoke_known = torch.BoolTensor(
            smoke_known if smoke_known is not None
            else np.ones(n, dtype=bool)
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
            "smoke_known":     self.smoke_known[idx],
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
            smoke_known       = self.smoke_known[mask].numpy(),
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
        smoke_known_path = d / "smoke_labels_known.npy"
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
            smoke_known       = np.load(smoke_known_path) if smoke_known_path.exists() else None,
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

        Only cells with smoke_known=True contribute to the counts — a cell
        with no verified (or explicitly opted-in weak-proxy) smoke label
        carries a meaningless placeholder in self.smoke (see __init__'s
        docstring) and must never influence class weighting, the same way
        malignancy_known already gates the malignancy loss.
        """
        known = self.smoke_known.numpy()
        labels = self.smoke.numpy()[known] if known.any() else np.array([], dtype=np.int64)
        counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
        n = counts.sum()
        weights = np.zeros(num_classes, dtype=np.float32)
        if n > 0:
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
    names = names or [f"dataset_{i}" for i in range(len(datasets))]
    id_sets = [_dataset_subject_ids(ds) for ds in datasets]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            overlap = id_sets[i] & id_sets[j]
            if overlap:
                raise ValueError(
                    f"Subject leakage: {names[i]!r} and {names[j]!r} share "
                    f"{len(overlap)} subject(s): {sorted(overlap)[:10]}"
                    + (" ..." if len(overlap) > 10 else "")
                )


def _dataset_subject_ids(ds) -> set:
    """Subject-id set for a CellLevelDataset, a SubjectLevelDataset, or a
    plain iterable of ids — shared by assert_disjoint_subjects and
    validate_experiment_partitions."""
    if ds is None:
        return set()
    if isinstance(ds, CellLevelDataset):
        return set(ds.subject_ids.tolist())
    if isinstance(ds, SubjectLevelDataset):
        return {str(b["subject_id"]) for b in ds.bags}
    return {str(s) for s in ds}


def validate_experiment_partitions(
    train_cell_dataset:    Optional["CellLevelDataset"] = None,
    val_cell_dataset:      Optional["CellLevelDataset"] = None,
    train_subject_dataset: Optional["SubjectLevelDataset"] = None,
    val_subject_dataset:   Optional["SubjectLevelDataset"] = None,
    test_cell_dataset:     Optional["CellLevelDataset"] = None,
    test_subject_dataset:  Optional["SubjectLevelDataset"] = None,
) -> Dict[str, set]:
    """
    Full cross-task leakage check. assert_disjoint_subjects, called
    separately on (train_cell, val_cell) and (train_subject, val_subject) by
    Trainer._validate_train_val, misses two overlap pairs: a subject whose
    CELLS are in train but whose BAG is in val, and vice versa. That is a
    real leakage path Phase 3 exercises (it uses all four datasets together),
    so this checks it directly by unioning each split's cell-subject-ids and
    bag-subject-ids into one set per split, then requiring the per-split sets
    to be pairwise disjoint. (A subject legitimately appearing in both its
    OWN split's cell dataset and that SAME split's bag dataset is fine and
    expected — only cross-split overlap is leakage.)

    Diagnostic-mode datasets (synthetic smoke tests with no real subject
    identity) are skipped entirely — this function only enforces real
    experiments. Returns the per-split subject-id sets actually checked, so
    callers (e.g. Trainer.final_test_evaluation) can compare a later test set
    against previously-validated train/val sets without recomputing them.
    """
    all_ds = [train_cell_dataset, val_cell_dataset, train_subject_dataset,
              val_subject_dataset, test_cell_dataset, test_subject_dataset]
    if any(getattr(ds, "diagnostic_mode", False) for ds in all_ds if ds is not None):
        return {}

    groups: Dict[str, set] = {
        "train": _dataset_subject_ids(train_cell_dataset) | _dataset_subject_ids(train_subject_dataset),
        "val":   _dataset_subject_ids(val_cell_dataset)   | _dataset_subject_ids(val_subject_dataset),
    }
    if test_cell_dataset is not None or test_subject_dataset is not None:
        groups["test"] = _dataset_subject_ids(test_cell_dataset) | _dataset_subject_ids(test_subject_dataset)

    for name, ids in groups.items():
        if not ids:
            raise ValueError(f"validate_experiment_partitions: {name!r} partition has no subjects.")

    names = list(groups.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = groups[names[i]] & groups[names[j]]
            if overlap:
                raise ValueError(
                    f"Cross-task subject leakage: {names[i]!r} and {names[j]!r} share "
                    f"{len(overlap)} subject(s) — checked across BOTH cell-level and "
                    f"subject-level (bag) datasets, not just same-modality pairs: "
                    f"{sorted(overlap)[:10]}" + (" ..." if len(overlap) > 10 else "")
                )
    return groups


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
        smoke_known = b.get("smoke_known")
        return {
            "gene_matrix":   torch.FloatTensor(b["gene_matrix"]),
            "cell_type_ids": torch.LongTensor(b["cell_type_ids"]),
            "smoke_labels":  torch.LongTensor(b["smoke_labels"]),
            "smoke_known":   torch.BoolTensor(smoke_known if smoke_known is not None else np.ones(n, dtype=bool)),
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


def read_checkpoint_metadata(ckpt_path: Union[str, Path], device: str = "cpu") -> Dict:
    """
    Peek at a checkpoint's metadata WITHOUT constructing or loading weights
    into any model. Used to discover the checkpoint's actual effective
    smoke-label space (effective_label_mapping) BEFORE building the model,
    so a model can be constructed with the correct num_smoke_types up front
    — otherwise a config/checkpoint class-count mismatch only surfaces as an
    opaque shape-mismatch error deep inside load_state_dict(). Returns {}
    for a legacy bare-state_dict checkpoint (no metadata to read).
    """
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj
    return {}


def _label_mapping_from_checkpoint_meta(ckpt_meta: Dict) -> Optional["EffectiveLabelMapping"]:
    """Shared helper: reconstruct EffectiveLabelMapping from a checkpoint's
    saved effective_label_mapping dict, or None if absent (legacy checkpoint
    / no rare-class policy was wired in for that run)."""
    raw = ckpt_meta.get("effective_label_mapping") if ckpt_meta else None
    return EffectiveLabelMapping.from_dict(raw) if raw else None


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

        # Phase 2 subject-aware class-imbalance configuration (data/sampling.py).
        # Absent config resolves to the pre-Phase-2 defaults (shuffle sampler,
        # train-only inverse-frequency class weights, plain cross-entropy) —
        # see resolve_smoke_imbalance_config's docstring.
        self.smoke_imbalance_config = resolve_smoke_imbalance_config(self.cfg.get("smoke_imbalance"))
        # The most recently constructed subject-balanced batch sampler, if
        # any — kept so _save() can persist its realized sampling
        # diagnostics (populated only after a full epoch has been iterated).
        self._last_subject_balanced_sampler: Optional[SubjectBalancedBatchSampler] = None

        # Experiment metadata carried into every saved checkpoint (section 8/22).
        self.split_manifest_path:      Optional[str] = None
        self.preprocessing_artifact_path: Optional[str] = None
        # The actual fitted PreprocessingArtifact this Trainer's data was
        # transformed with, if known — set via set_preprocessing_artifact()
        # (called automatically by from_experiment_context()). _save()
        # embeds its scientific_fingerprint() in every checkpoint so a
        # later loader (Predictor.from_config, evaluate.py) can reject a
        # checkpoint paired with a DIFFERENT preprocessing artifact instead
        # of silently trusting whatever preprocessing_artifact.json happens
        # to sit next to the checkpoint file. None means "no artifact was
        # wired in" (legacy/diagnostic/synthetic run) — checkpoints saved
        # this way carry no fingerprint to check against.
        self.preprocessing_artifact = None
        self.effective_label_mapping:  Optional[Dict] = None
        self.rare_class_policy:        Optional[str] = None
        self.transductive_batch_correction: bool = False
        # The typed EffectiveLabelMapping object (data/label_mapping.py) —
        # set via set_label_mapping(), which also keeps
        # effective_label_mapping/rare_class_policy above in sync and
        # validates it against self.model.num_smoke. None means "no
        # rare-class policy was wired in" (six-class default / legacy /
        # diagnostic runs) — class-name lookups fall back to constants.SMOKE_TYPES.
        self.label_mapping: Optional["EffectiveLabelMapping"] = None

        # Held-out-test enforcement state: subjects seen as train/val during
        # this Trainer's lifetime, and how many times final_test_evaluation
        # has already run — see _record_seen_subjects / final_test_evaluation.
        self._train_subjects_seen: set = set()
        self._val_subjects_seen:   set = set()
        self._test_eval_run_count: int = 0

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

    @classmethod
    def from_experiment_context(
        cls,
        context: "benchmarks.context.ExperimentContext",
        device:  str = "cpu",
        seed:    Optional[int] = None,
        pooling: Optional[str] = None,
    ) -> "Trainer":
        """
        Build a Trainer whose model, label mapping, and provenance are all
        derived from one ExperimentContext (src/benchmarks/context.py) — so a
        benchmark comparing baselines against MultiSmokeCancerNet can never
        accidentally construct the model with the wrong input_dim or the
        fixed 6-class head when the rare-class policy already shrank K.

        Rejects (raises ValueError) rather than silently truncating/padding
        when the context's preprocessing artifact's gene count doesn't match
        model_config['input_dim'] (if input_dim is explicitly set in config —
        otherwise it's derived from the context) — a mismatch there means the
        config was written for a different HVG count than this context
        actually produced.
        """
        model_cfg = dict(context.config.get("model", {}))
        configured_input_dim = model_cfg.get("input_dim")
        if configured_input_dim is not None and configured_input_dim != context.input_dim:
            raise ValueError(
                f"Trainer.from_experiment_context: config model.input_dim="
                f"{configured_input_dim} does not match context.input_dim="
                f"{context.input_dim} (len(preprocessing_artifact.gene_list)). "
                "Remove the explicit input_dim from config to derive it from "
                "the context, or fix the mismatch."
            )
        model_cfg["input_dim"] = context.input_dim

        model = MultiSmokeCancerNet.from_config(
            {"model": model_cfg}, num_smoke_types=context.num_smoke_classes, pooling=pooling,
        )
        trainer = cls(model, context.config, device=device, seed=seed if seed is not None else context.seed)
        trainer.set_label_mapping(context.label_mapping)
        trainer.transductive_batch_correction = context.transductive_batch_correction
        trainer.rare_class_policy = context.label_mapping.policy
        # getattr, not context.preprocessing_artifact directly: some
        # lightweight/fake ExperimentContext test doubles (and any future
        # minimal context) may not define this attribute at all — treated
        # the same as "no artifact wired in" (None), matching Trainer's own
        # default, rather than an AttributeError unrelated to preprocessing.
        trainer.set_preprocessing_artifact(getattr(context, "preprocessing_artifact", None))
        return trainer

    def set_preprocessing_artifact(self, artifact) -> None:
        """Wire this Trainer's PreprocessingArtifact in so _save() can embed
        its scientific_fingerprint() in every checkpoint — see the
        docstring on self.preprocessing_artifact. Also cross-checks the
        artifact's selected-gene count against self.model.input_dim
        (when set), since a checkpoint whose model input width doesn't
        match its own artifact's gene count is already unusable — better to
        fail here than after training."""
        if artifact is not None:
            model_input_dim = getattr(self.model, "input_dim", None)
            if model_input_dim is not None and model_input_dim != len(artifact.gene_list):
                raise ValueError(
                    f"Trainer.set_preprocessing_artifact: model.input_dim={model_input_dim} "
                    f"does not match artifact.gene_list length={len(artifact.gene_list)} — "
                    "this model cannot consume this artifact's output."
                )
        self.preprocessing_artifact = artifact

    def set_label_mapping(self, mapping: "EffectiveLabelMapping") -> None:
        """
        The sanctioned way to wire an EffectiveLabelMapping (data/label_mapping.py)
        into this Trainer — validates it against self.model.num_smoke (the
        model's actual output width) and keeps effective_label_mapping/
        rare_class_policy in sync so they're persisted correctly in every
        checkpoint. Raises ValueError immediately, before any training runs,
        if the model was NOT built with num_smoke == mapping.k (see
        MultiSmokeCancerNet.from_config's num_smoke_types override) —
        catching a config/mapping conflict here is far clearer than letting
        it surface as a silently-wrong macro-F1 or a dead output neuron.
        """
        if mapping.k != self.model.num_smoke:
            raise ValueError(
                f"Trainer.set_label_mapping: mapping has K={mapping.k} effective classes "
                f"but self.model.num_smoke={self.model.num_smoke} — the model must be "
                f"constructed with num_smoke_types={mapping.k} (see "
                "MultiSmokeCancerNet.from_config's num_smoke_types override) before wiring "
                "this mapping in."
            )
        self.label_mapping = mapping
        self.effective_label_mapping = mapping.to_dict()
        self.rare_class_policy = mapping.policy

    def _class_names(self) -> List[str]:
        """Ordered smoke-class display names for this Trainer's current
        output space — the wired EffectiveLabelMapping if set, else the
        fixed 6-class constants.SMOKE_TYPES (no-merge / legacy default)."""
        return self.label_mapping.class_names if self.label_mapping else list(SMOKE_TYPES.values())

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
            self._train_subjects_seen |= _dataset_subject_ids(train_ds)
            self._val_subjects_seen   |= _dataset_subject_ids(val_ds)

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
            # Scientific fingerprint (data/preprocessing.py::
            # PreprocessingArtifact.scientific_fingerprint) of the artifact
            # this checkpoint's training data was actually transformed
            # with, plus its selected-gene count — None when no artifact
            # was wired in (see set_preprocessing_artifact). A loader
            # (Predictor.from_config) that finds BOTH this fingerprint and
            # a preprocessing_artifact.json on disk must verify they match
            # before trusting the pair together.
            "preprocessing_artifact_fingerprint": (
                self.preprocessing_artifact.scientific_fingerprint()
                if self.preprocessing_artifact is not None else None
            ),
            "preprocessing_artifact_gene_count": (
                len(self.preprocessing_artifact.gene_list)
                if self.preprocessing_artifact is not None else None
            ),
            "effective_label_mapping": self.effective_label_mapping,
            "rare_class_policy":  self.rare_class_policy,
            "transductive_batch_correction": self.transductive_batch_correction,
            "smoke_imbalance_config": self.smoke_imbalance_config,
            "smoke_sampling_diagnostics": (
                self._last_subject_balanced_sampler.last_realized_diagnostics.to_dict()
                if self._last_subject_balanced_sampler is not None
                and self._last_subject_balanced_sampler.last_realized_diagnostics is not None
                else None
            ),
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

    def write_bundle(
        self,
        phase: int,
        bundle_dir: Optional[Union[str, Path]] = None,
        dataset_manifest_fingerprint: Optional[str] = None,
        split_fingerprint: Optional[str] = None,
        calibration_state: Optional[Dict] = None,
        decision_threshold: Optional[float] = None,
        environment_snapshot: Optional[Dict] = None,
    ) -> Path:
        """
        Write a full model-bundle manifest (benchmarks/bundle.py) for the
        checkpoint already saved at ckpt_dir/phase{phase}_best.pt, bound to
        self.preprocessing_artifact (see set_preprocessing_artifact). Raises
        ValueError if no artifact is wired in — a bundle with no
        preprocessing artifact reference is not a real Phase 4 bundle, and
        this method never fabricates a placeholder one.
        """
        from benchmarks.bundle import write_model_bundle
        from benchmarks.reporting import _environment_snapshot

        if self.preprocessing_artifact is None:
            raise ValueError(
                "Trainer.write_bundle: no preprocessing artifact is wired in (see "
                "set_preprocessing_artifact) — cannot build a model bundle without one."
            )
        if environment_snapshot is None:
            environment_snapshot = _environment_snapshot(synthetic=False, seed=self.seed)
        ckpt_path = self.ckpt_dir / f"phase{phase}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Trainer.write_bundle: no checkpoint at {ckpt_path} — call _save()/train this "
                "phase first."
            )
        bundle_dir = Path(bundle_dir) if bundle_dir is not None else self.ckpt_dir / f"bundle_phase{phase}"
        class_vocabulary = self._class_names()
        species_policy = self.full_cfg.get("data", {}).get("experiment_mode", "human_only")
        assay_mode = self.full_cfg.get("data", {}).get("assay_mode", "human_single_cell")
        label_policy = self.rare_class_policy or self.full_cfg.get("data", {}).get("label_policy", "verified_only")
        return write_model_bundle(
            bundle_dir, ckpt_path, self.preprocessing_artifact,
            model_config=self.full_cfg.get("model", {}), class_vocabulary=class_vocabulary,
            label_policy=label_policy, species_policy=species_policy, assay_mode=assay_mode,
            dataset_manifest_fingerprint=dataset_manifest_fingerprint, split_fingerprint=split_fingerprint,
            calibration_state=calibration_state, decision_threshold=decision_threshold,
            environment_snapshot=environment_snapshot,
        )

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

    def _train_cell_loader(self, train_cell_dataset: "CellLevelDataset", batch_size: int) -> DataLoader:
        """
        Build the Phase 1 / Phase 3 / final-fit training cell DataLoader
        per self.smoke_imbalance_config["sampler"]:

          "shuffle"           — the original DataLoader(shuffle=True) behaviour.
          "subject_balanced"  — data.sampling.SubjectBalancedBatchSampler
                                 (class -> subject -> cell). Never combined
                                 with shuffle=True — DataLoader forbids
                                 batch_sampler together with shuffle/sampler/
                                 batch_size/drop_last, which is correct here too.

        Only ever called with a TRAINING dataset — validation/test loaders
        are built directly with shuffle=False and never call this method.
        """
        mode = self.smoke_imbalance_config["sampler"]
        if mode == "subject_balanced":
            sampler = build_subject_balanced_sampler(
                train_cell_dataset, num_classes=self.model.num_smoke,
                batch_size=batch_size, seed=self.seed,
                resolved_cfg=self.smoke_imbalance_config,
            )
            self._last_subject_balanced_sampler = sampler
            return DataLoader(train_cell_dataset, batch_sampler=sampler)

        self._last_subject_balanced_sampler = None
        return DataLoader(
            train_cell_dataset, batch_size=batch_size, shuffle=True,
            drop_last=len(train_cell_dataset) >= batch_size,
            generator=self._generator(),
        )

    def _smoke_loss_weights_and_type(
        self, train_cell_dataset: "CellLevelDataset", explicit_weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], str, float]:
        """
        Resolve (reported_weights, loss_alpha, loss_type, focal_gamma) for
        the smoke-type loss, per self.smoke_imbalance_config.

        reported_weights — what class_weighting actually produced (or the
        caller's explicit override); this is what gets logged/persisted as
        "the class weights for this run", regardless of whether focal's
        alpha ends up using them.
        loss_alpha — what is actually passed into CrossEntropyLoss/FocalLoss
        as the class weight/alpha term. Equal to reported_weights UNLESS
        loss="focal" and focal_alpha_mode="none", in which case it is None
        — applying inverse-frequency correction via BOTH subject-balanced
        sampling and loss weighting is a real double-correction risk (see
        README.md), so focal_alpha_mode gives an explicit way to keep
        sampling-only correction even when class_weighting=inverse_frequency.
        """
        cfg = self.smoke_imbalance_config
        if explicit_weights is not None:
            reported = explicit_weights
        elif cfg["class_weighting"] == "inverse_frequency":
            reported = train_cell_dataset.smoke_class_weights(num_classes=self.model.num_smoke)
        else:
            reported = None

        loss_alpha = reported
        if cfg["loss"] == "focal" and cfg["focal_alpha_mode"] == "none":
            loss_alpha = None

        return reported, loss_alpha, cfg["loss"], cfg["focal_gamma"]

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

        # num_classes = the model's actual output width, not the fixed
        # 6-class constant — a rare-class policy may have shrunk it to K < 6
        # (see data/label_mapping.py); weighting/sampling a merged/excluded
        # class that no longer exists in the model's output would be
        # silently meaningless. class_weighting/loss/focal_gamma come from
        # self.smoke_imbalance_config (data/sampling.py) unless the caller
        # passed an explicit override.
        smoke_class_weights, loss_alpha, loss_type, focal_gamma = self._smoke_loss_weights_and_type(
            train_cell_dataset, smoke_class_weights,
        )
        self._log(f"  smoke imbalance config: sampler={self.smoke_imbalance_config['sampler']!r} "
                   f"class_weighting={self.smoke_imbalance_config['class_weighting']!r} "
                   f"loss={loss_type!r}")
        if smoke_class_weights is not None:
            self._log(f"  smoke class weights (train-only): "
                       f"{[round(w, 3) for w in smoke_class_weights.tolist()]}")

        train_dl = self._train_cell_loader(train_cell_dataset, batch_size)
        val_dl   = DataLoader(val_cell_dataset,   batch_size=batch_size, shuffle=False)

        loss_fn = MultiTaskLoss(
            lambda_smoke=0.50, lambda_malignancy=0.50,
            smoke_class_weights=loss_alpha.to(self.device) if loss_alpha is not None else None,
            loss_type=loss_type, focal_gamma=focal_gamma,
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
                smoke_k= batch["smoke_known"].to(self.device)
                malig_t= batch["malignancy_label"].to(self.device)
                malig_k= batch["malignancy_known"].to(self.device)
                dose_t = batch["exposure_dose"].to(self.device)
                z, logits, malig = self.model.forward_cell(x)
                loss, _ = loss_fn.cell_level_loss(logits, smoke_t, malig, malig_t, malig_known=malig_k, smoke_known=smoke_k)
                dose_loss, _ = loss_fn.dose_response_loss(self.model.dose_head(z), dose_t, malig)
                self._grad_step(loss + lambda_dose * dose_loss, opt, params)
            sched.step()

            # validate
            self.model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in val_dl:
                    _, logits, _ = self.model.forward_cell(batch["x"].to(self.device))
                    known = batch["smoke_known"].bool()
                    # Cells with no verified (or opted-in weak-proxy) smoke
                    # label carry a meaningless placeholder target — scoring
                    # against it would silently corrupt smoke_acc/macro_f1
                    # (and therefore Phase 1 checkpoint selection) with
                    # fabricated "ground truth". See CellLevelDataset's
                    # smoke_known docstring.
                    preds.extend(logits.argmax(1)[known].cpu().tolist())
                    targets.extend(batch["smoke_label"][known].tolist())

            if not targets:
                raise ValueError(
                    "Phase 1 validation set has zero cells with a known smoke label — "
                    "smoke_acc/smoke_macro_f1 are undefined. Check that val_cell_dataset "
                    "isn't entirely weak-proxy-only cells under the default verified_only "
                    "label policy."
                )
            acc = sum(p == t for p, t in zip(preds, targets)) / len(targets)
            # Same definition evaluate.py's _smoke_metrics uses (explicit
            # label list over ALL effective classes, not just classes
            # observed in this validation batch) — otherwise the metric used
            # to select this checkpoint could silently disagree with the
            # macro-F1 later reported for it. See metrics.py. num_classes is
            # self.model.num_smoke (the model's actual output width), NOT
            # the fixed 6-class constant — a rare-class policy may have
            # shrunk the effective label space to K < 6 (data/label_mapping.py).
            f1_report = multiclass_f1_report(targets, preds, self.model.num_smoke)
            f1 = f1_report["macro_f1"]
            history.append({"epoch": epoch, "smoke_acc": acc, "smoke_macro_f1": f1,
                             "classes_absent_from_val": f1_report["classes_absent_from_targets"]})
            self._log(f"  epoch {epoch:02d}/{epochs}  smoke_acc={acc:.3f}  smoke_macro_f1={f1:.3f}")
            if f1_report["is_partial"]:
                self._log(f"    WARNING: validation split has zero examples of class(es) "
                           f"{f1_report['classes_absent_from_targets']} — this macro_f1 is "
                           "partial, not a full-class-set score.")

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

    # ── Final-fit variants (no internal validation split) ─────────────────────
    #
    # phase1/phase2 above always require a subject-disjoint validation split
    # to select the best-on-val checkpoint. That is the right behaviour for
    # every normal training run, but it is architecturally incompatible with
    # "fit the ONE final frozen-test candidate on every eligible development
    # subject" (see benchmarks/final_evaluation.py): a normal phase1/phase2
    # call would always hold some development subjects out of gradient
    # updates for its own bookkeeping. These two methods exist only for that
    # final refit: they train for a FIXED, already-decided epoch count (from
    # development-only nested-CV/OOF evidence, selected before this ever
    # runs), take every subject supplied as a gradient-update subject, and
    # never consult a validation split for checkpoint selection or early
    # stopping. They must never be used for CV/model-selection training.

    def phase1_final_fit(
        self,
        train_cell_dataset: CellLevelDataset,
        epochs: Optional[int] = None,
        smoke_class_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        self._log("\n=== Phase 1 (final fit, no validation split) ===")
        if len(train_cell_dataset) == 0:
            raise ValueError("Trainer: train dataset is empty — nothing to train on.")
        if not getattr(train_cell_dataset, "diagnostic_mode", False):
            self._train_subjects_seen |= _dataset_subject_ids(train_cell_dataset)

        epochs = epochs if epochs is not None else self.cfg.get("phase1_epochs", 15)
        lr = self.cfg.get("phase1_lr", 1e-3)
        batch_size = self.cfg.get("phase1_batch_size", 512)

        # The imbalance strategy (sampler/class-weighting/loss) is decided
        # during development and FROZEN by the time this runs — this refit
        # uses whatever self.smoke_imbalance_config already holds (set at
        # Trainer construction / from_experiment_context), deriving class/
        # subject/class-weight statistics only from train_cell_dataset (the
        # full development pool passed in here), never from test data.
        smoke_class_weights, loss_alpha, loss_type, focal_gamma = self._smoke_loss_weights_and_type(
            train_cell_dataset, smoke_class_weights,
        )

        train_dl = self._train_cell_loader(train_cell_dataset, batch_size)
        loss_fn = MultiTaskLoss(
            lambda_smoke=0.50, lambda_malignancy=0.50,
            smoke_class_weights=loss_alpha.to(self.device) if loss_alpha is not None else None,
            loss_type=loss_type, focal_gamma=focal_gamma,
        )
        params = [
            *self.model.encoder.parameters(),
            *self.model.smoke_head.parameters(),
            *self.model.malignancy_head.parameters(),
            *self.model.dose_head.parameters(),
        ]
        opt = self._make_optimizer(params, lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        lambda_dose = self.cfg.get("lambda_dose", 0.10)

        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch in train_dl:
                x = batch["x"].to(self.device)
                smoke_t = batch["smoke_label"].to(self.device)
                smoke_k = batch["smoke_known"].to(self.device)
                malig_t = batch["malignancy_label"].to(self.device)
                malig_k = batch["malignancy_known"].to(self.device)
                dose_t = batch["exposure_dose"].to(self.device)
                z, logits, malig = self.model.forward_cell(x)
                loss, _ = loss_fn.cell_level_loss(logits, smoke_t, malig, malig_t, malig_known=malig_k, smoke_known=smoke_k)
                dose_loss, _ = loss_fn.dose_response_loss(self.model.dose_head(z), dose_t, malig)
                self._grad_step(loss + lambda_dose * dose_loss, opt, params)
            sched.step()
            self._log(f"  epoch {epoch:02d}/{epochs}  (final fit — every subject trained, no checkpoint selection)")

        self._log(f"Phase 1 (final fit) done — trained {epochs} epoch(s) on all "
                   f"{len(train_cell_dataset)} cells, no held-out checkpoint selection.")
        return {
            "epochs": epochs, "n_cells": len(train_cell_dataset),
            "smoke_imbalance_config": self.smoke_imbalance_config,
            "smoke_class_weights": smoke_class_weights.tolist() if smoke_class_weights is not None else None,
            "smoke_sampling_diagnostics": (
                self._last_subject_balanced_sampler.last_realized_diagnostics.to_dict()
                if self._last_subject_balanced_sampler is not None
                and self._last_subject_balanced_sampler.last_realized_diagnostics is not None
                else None
            ),
        }

    def phase2_final_fit(
        self,
        train_subject_dataset: SubjectLevelDataset,
        epochs: Optional[int] = None,
    ) -> Dict:
        self._log("\n=== Phase 2 (final fit, no validation split) ===")
        if len(train_subject_dataset) == 0:
            raise ValueError("Trainer: train dataset is empty — nothing to train on.")
        if not getattr(train_subject_dataset, "diagnostic_mode", False):
            self._train_subjects_seen |= _dataset_subject_ids(train_subject_dataset)

        epochs = epochs if epochs is not None else self.cfg.get("phase2_epochs", 12)
        lr = self.cfg.get("phase2_lr", 5e-4)

        for p in [*self.model.encoder.parameters(),
                  *self.model.smoke_head.parameters(),
                  *self.model.malignancy_head.parameters()]:
            p.requires_grad = False

        train_dl = DataLoader(train_subject_dataset, batch_size=1, shuffle=True,
                               collate_fn=subject_collate_fn, generator=self._generator())
        loss_fn = MultiTaskLoss()
        params = list(self.model.aggregator.parameters())
        opt = self._make_optimizer(params, lr)

        for epoch in range(1, epochs + 1):
            self.model.train()
            for [item] in train_dl:
                x_bag = item["gene_matrix"].to(self.device)
                ct_ids = item["cell_type_ids"].to(self.device)
                cancer = item["cancer_label"].to(self.device)
                out = self.model.forward_subject(x_bag, ct_ids)
                loss, _ = loss_fn.subject_level_loss(out["cancer_probability"], cancer)
                self._grad_step(loss, opt, params)
            self._log(f"  epoch {epoch:02d}/{epochs}  (final fit — every subject trained, no checkpoint selection)")

        for p in self.model.parameters():
            p.requires_grad = True

        self._log(f"Phase 2 (final fit) done — trained {epochs} epoch(s) on all "
                   f"{len(train_subject_dataset)} subjects, no held-out checkpoint selection.")
        return {"epochs": epochs, "n_subjects": len(train_subject_dataset)}

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
        # The two checks above only catch same-modality overlap (train_cell
        # vs val_cell, train_subject vs val_subject). Phase 3 uses all four
        # datasets jointly, so a subject whose CELLS are in train but whose
        # BAG is in val (or vice versa) would otherwise go undetected.
        validate_experiment_partitions(
            train_cell_dataset=train_cell_dataset, val_cell_dataset=val_cell_dataset,
            train_subject_dataset=train_subject_dataset, val_subject_dataset=val_subject_dataset,
        )
        if not skip_eligibility_check:
            elig_tr = check_mil_eligibility(train_subject_dataset)
            elig_va = check_mil_eligibility(val_subject_dataset)
            self._log(f"  MIL eligibility ✓  train={elig_tr['n_total']} subjects "
                       f"({elig_tr['n_positive']} pos, {elig_tr['n_negative']} neg)  "
                       f"val={elig_va['n_total']} subjects "
                       f"({elig_va['n_positive']} pos, {elig_va['n_negative']} neg)")

        epochs = self.cfg.get("phase3_epochs", 8)
        lr     = self.cfg.get("phase3_lr",     1e-4)

        cell_dl  = self._train_cell_loader(train_cell_dataset, 256)
        sub_dl   = DataLoader(train_subject_dataset, batch_size=1, shuffle=True,
                               collate_fn=subject_collate_fn, generator=self._generator())
        val_dl   = DataLoader(val_subject_dataset,   batch_size=1, shuffle=False,
                               collate_fn=subject_collate_fn)

        smoke_class_weights, loss_alpha, loss_type, focal_gamma = self._smoke_loss_weights_and_type(
            train_cell_dataset, None,
        )
        loss_fn = MultiTaskLoss(
            lambda_smoke=0.30, lambda_malignancy=0.30, lambda_subject=0.40,
            smoke_class_weights=loss_alpha.to(self.device) if loss_alpha is not None else None,
            loss_type=loss_type, focal_gamma=focal_gamma,
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
                    smoke_k= cb["smoke_known"].to(self.device)
                    malig_t= cb["malignancy_label"].to(self.device)
                    malig_k= cb["malignancy_known"].to(self.device)
                    _, logits, malig = self.model.forward_cell(x)
                    cl, _  = loss_fn.cell_level_loss(logits, smoke_t, malig, malig_t, malig_known=malig_k, smoke_known=smoke_k)
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
        allow_repeat: bool = False,
    ) -> Dict:
        """
        The ONE sanctioned place test data is used: load the validation-
        selected checkpoint for `phase` and evaluate it ONCE on the
        untouched test split.

        Enforcement, not just documentation:
          - test subjects must be disjoint from every subject seen by
            _validate_train_val on this Trainer instance so far (i.e. every
            subject that has appeared in ANY train/val call for ANY phase) —
            raises ValueError on overlap instead of silently scoring on data
            that leaked into training/validation.
          - a second call raises RuntimeError by default (allow_repeat=True
            is required to deliberately re-run, and the result is then
            marked non-pristine) — repeated test evaluation without that
            guard is how a test set quietly becomes a second validation set.
          - the report is written to its own heldout_test_report.json /
            heldout_test_predictions.json (separate from evaluation_report.json,
            which full_report() also writes and which may be called on
            train/val data elsewhere) with explicit provenance.

        Diagnostic-mode datasets (synthetic smoke tests) skip the
        subject-overlap check, matching _validate_train_val's behaviour.
        """
        import json as _json
        from datetime import datetime, timezone
        from evaluate import Evaluator  # local import — evaluate.py imports from this module

        diagnostic = (getattr(test_cell_dataset, "diagnostic_mode", False)
                      or getattr(test_subject_dataset, "diagnostic_mode", False))
        test_subjects = _dataset_subject_ids(test_cell_dataset) | _dataset_subject_ids(test_subject_dataset)

        if not diagnostic:
            overlap_train = test_subjects & self._train_subjects_seen
            overlap_val   = test_subjects & self._val_subjects_seen
            if overlap_train or overlap_val:
                raise ValueError(
                    "final_test_evaluation: test subjects overlap subjects already used for "
                    f"training ({sorted(overlap_train)[:10]}) or validation "
                    f"({sorted(overlap_val)[:10]}) on this Trainer — refusing to score a "
                    "'held-out' result on data the model or checkpoint selection has seen."
                )

        if self._test_eval_run_count > 0 and not allow_repeat:
            raise RuntimeError(
                f"final_test_evaluation has already run {self._test_eval_run_count} time(s) "
                "on this Trainer. Repeated test evaluation risks the test set being used, "
                "even inadvertently, to guide model/hyperparameter/threshold choices — pass "
                "allow_repeat=True only for a deliberate re-run, which will be marked "
                "non-pristine in the saved report."
            )

        self._load_best(phase)
        ev = Evaluator(self.model, self.device, label_mapping=self.label_mapping)
        report, raw = ev.full_report(
            test_cell_dataset, test_subject_dataset,
            out_dir=out_dir or self.ckpt_dir,
        )
        self._test_eval_run_count += 1
        report["provenance"] = {
            "split_name":     "test",
            "is_held_out":    True,
            "checkpoint":     f"phase{phase}_best.pt",
            "split_manifest_path": self.split_manifest_path,
            "threshold_source": "default_0.50" if not hasattr(self, "_selected_threshold")
                                 else "validation_selected",
            "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_evaluation_run_count": self._test_eval_run_count,
            "is_pristine": self._test_eval_run_count == 1,
            "note": "Final held-out test evaluation — run once, not used for model/"
                    "threshold/hyperparameter selection.",
        }

        out = Path(out_dir) if out_dir else self.ckpt_dir
        if not out.is_absolute():
            out = Path(__file__).parents[1] / out
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "heldout_test_report.json", "w") as f:
            _json.dump(report, f, indent=2)
        with open(out / "heldout_test_predictions.json", "w") as f:
            _json.dump({**raw, "provenance": report["provenance"]}, f)

        self._log("[train] final_test_evaluation complete — this is a HELD-OUT TEST result, "
                   "not a validation or training-set number. "
                   f"Saved → {out / 'heldout_test_report.json'}")
        return report

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        gene_matrix:   np.ndarray,    # [N, genes]  float32
        cell_type_ids: np.ndarray,    # [N]         int
    ) -> Dict:
        """Run inference on one subject. Returns cancer risk + interpretability."""
        cell_type_ids = validate_cell_type_ids(
            cell_type_ids, self.model.num_cell_types, n_expected=len(gene_matrix),
        )
        self.model.eval()
        with torch.no_grad():
            out = self.model.forward_subject(
                torch.FloatTensor(gene_matrix).to(self.device),
                torch.LongTensor(cell_type_ids).to(self.device),
            )
        prob  = out["cancer_probability"].item()
        attn  = out["attention_weights"].cpu().numpy()
        smoke = out["cell_smoke_probs"].cpu().numpy().argmax(axis=1)
        # class_names is indexed by the model's actual effective smoke id
        # (0..K-1) — using constants.SMOKE_TYPES directly here would
        # mislabel predictions whenever a rare-class policy has shrunk K
        # below 6 (see data/label_mapping.py, set_label_mapping()).
        class_names = self._class_names()

        return {
            "cancer_probability":  prob,
            "risk_flag":           "HIGH" if prob >= 0.7 else "MODERATE" if prob >= 0.4 else "LOW",
            "top5_cells":          attn.argsort()[-5:][::-1].tolist(),
            "dominant_smoke_type": class_names[int(np.bincount(smoke).argmax())],
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