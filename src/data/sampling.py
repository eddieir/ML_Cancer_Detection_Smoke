"""
data/sampling.py — subject-first, class-balanced training sampling.

Why a plain per-cell WeightedRandomSampler is not enough: assigning an
inverse-frequency weight directly to every cell still lets a subject with
many cells dominate the sampling population within its own class — a
subject with 50,000 cells and a subject with 500 cells in the same class
would receive wildly different total sampling probability even though they
are the same number of INDEPENDENT observations (one subject each). This
module instead samples in three explicit stages every time it produces one
cell index:

    1. pick an effective smoke class
    2. pick a subject that belongs to that class
    3. pick a cell belonging to that subject

so a subject's cell count only controls the diversity of cells drawn FROM
it, never its own selection probability. Only the training split may ever
use this — validation/test must remain the natural, untouched, unweighted
distribution (see train.py's DataLoader construction).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler

_PLACEHOLDER_SUBJECT_IDS = {"unknown", "", "none", "None", "nan"}

CLASS_SELECTION_STRATEGIES = ("uniform", "natural", "inverse_subject_frequency")
SUBJECT_SELECTION_STRATEGIES = ("uniform",)


class SamplingConfigurationError(ValueError):
    """Raised for an invalid/inconsistent sampler configuration."""


class SamplingImpossibleError(ValueError):
    """Raised when the configured sampling contract cannot be honored
    (e.g. no-replacement sampling would need more cells than a subject has)."""


# ─── Subject/class index ───────────────────────────────────────────────────────

@dataclass
class SubjectClassIndex:
    """
    Deterministic class -> subject -> cell-index mapping built once from a
    training dataset's subject ids and per-cell effective smoke labels.

    Every field here is derived from the arrays actually passed in — it is
    never inferred from anything else (e.g. an unrelated class list), so
    "classes absent from this index" is unambiguous: any effective class in
    [0, num_classes) with zero entries in class_to_subjects.
    """

    subject_ids:   np.ndarray            # [N] str, canonical dataset order
    labels:        np.ndarray            # [N] int, effective smoke label per cell
    num_classes:   int
    is_synthetic:  bool = False          # provenance only — never relaxes validation
    # [N] bool, True if absent (legacy/synthetic callers — unchanged
    # behaviour). A cell with known_mask[i]=False has no verified (or
    # opted-in weak-proxy) smoke label — its class distribution/subject
    # weighting must never be influenced by it, so it is excluded from
    # subject_to_indices/class_to_subjects entirely below, the same
    # guarantee CellLevelDataset.smoke_class_weights enforces for the loss
    # weighting side of this same problem. Its position (index `i`) is
    # still a valid, unchanged index into the ORIGINAL dataset — it is
    # simply never offered up by this index, never physically removed or
    # renumbered, so every index this class DOES hand out remains a correct
    # index into the caller's full, unfiltered dataset.
    known_mask:    Optional[np.ndarray] = None

    unique_subjects:     List[str]              = field(init=False)
    subject_to_indices:  Dict[str, np.ndarray]  = field(init=False)
    subject_to_label:    Dict[str, int]         = field(init=False)
    class_to_subjects:   Dict[int, List[str]]   = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.subject_ids)
        if n == 0:
            raise SamplingConfigurationError(
                "SubjectClassIndex: dataset is empty — nothing to sample from."
            )
        if len(self.labels) != n:
            raise SamplingConfigurationError(
                f"SubjectClassIndex: len(subject_ids)={n} != len(labels)={len(self.labels)}."
            )
        if self.num_classes <= 0:
            raise SamplingConfigurationError(
                f"SubjectClassIndex: num_classes must be positive, got {self.num_classes}."
            )

        subj = np.array([str(s) for s in self.subject_ids], dtype=object)
        placeholder_mask = np.isin(subj, list(_PLACEHOLDER_SUBJECT_IDS))
        if placeholder_mask.any():
            bad = sorted(set(subj[placeholder_mask].tolist()))
            raise SamplingConfigurationError(
                f"SubjectClassIndex: {int(placeholder_mask.sum())}/{n} cell(s) have a "
                f"placeholder subject_id {bad} — subject-balanced sampling requires a real "
                "subject_id per cell (see CellLevelDataset's own placeholder rejection)."
            )
        self.subject_ids = subj

        labels = np.asarray(self.labels)
        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(np.int64)
        out_of_range = (labels < 0) | (labels >= self.num_classes)
        if out_of_range.any():
            bad_vals = sorted(set(labels[out_of_range].tolist()))
            raise SamplingConfigurationError(
                f"SubjectClassIndex: label(s) {bad_vals} are out of range "
                f"[0, {self.num_classes}) — effective labels must already be contiguous "
                "(see data/label_mapping.py)."
            )
        self.labels = labels

        known = (
            np.asarray(self.known_mask, dtype=bool) if self.known_mask is not None
            else np.ones(n, dtype=bool)
        )
        if len(known) != n:
            raise SamplingConfigurationError(
                f"SubjectClassIndex: len(known_mask)={len(known)} != len(subject_ids)={n}."
            )
        self.known_mask = known
        n_excluded = int((~known).sum())
        if n_excluded:
            print(f"[sampling] SubjectClassIndex: excluding {n_excluded:,}/{n:,} cell(s) with "
                  "no verified smoke label from subject-balanced sampling's class index.")

        subject_to_indices: Dict[str, List[int]] = {}
        for i, s in enumerate(self.subject_ids):
            if not known[i]:
                continue
            subject_to_indices.setdefault(s, []).append(i)

        subject_to_label: Dict[str, int] = {}
        for s, idxs in subject_to_indices.items():
            subj_labels = set(int(labels[i]) for i in idxs)
            if len(subj_labels) > 1:
                raise SamplingConfigurationError(
                    f"SubjectClassIndex: subject {s!r} has conflicting effective smoke "
                    f"labels {sorted(subj_labels)} across its cells — a subject must map "
                    "to exactly one effective label."
                )
            subject_to_label[s] = subj_labels.pop()

        self.unique_subjects = sorted(subject_to_indices)
        self.subject_to_indices = {
            s: np.array(idxs, dtype=np.int64) for s, idxs in subject_to_indices.items()
        }
        self.subject_to_label = subject_to_label

        class_to_subjects: Dict[int, List[str]] = {c: [] for c in range(self.num_classes)}
        for s in self.unique_subjects:
            class_to_subjects[subject_to_label[s]].append(s)
        self.class_to_subjects = {c: sorted(subs) for c, subs in class_to_subjects.items()}

    @classmethod
    def from_cell_dataset(cls, dataset, num_classes: int) -> "SubjectClassIndex":
        """Build directly from a train.CellLevelDataset (or any object with
        the same .subject_ids / .smoke / .smoke_known / .diagnostic_mode
        contract). Cells with smoke_known=False (no verified or opted-in
        weak-proxy smoke label — see CellLevelDataset's docstring) are
        excluded from the class index; a dataset with no smoke_known
        attribute at all (a legacy caller predating this field) is treated
        as fully known, unchanged from previous behaviour."""
        labels = dataset.smoke.numpy() if hasattr(dataset.smoke, "numpy") else np.asarray(dataset.smoke)
        smoke_known = getattr(dataset, "smoke_known", None)
        known_mask = (
            smoke_known.numpy() if hasattr(smoke_known, "numpy")
            else np.asarray(smoke_known) if smoke_known is not None
            else None
        )
        return cls(
            subject_ids  = np.asarray(dataset.subject_ids),
            labels       = labels,
            num_classes  = num_classes,
            is_synthetic = bool(getattr(dataset, "diagnostic_mode", False)),
            known_mask   = known_mask,
        )

    @property
    def observed_classes(self) -> List[int]:
        return [c for c, subs in self.class_to_subjects.items() if subs]

    @property
    def absent_classes(self) -> List[int]:
        return [c for c, subs in self.class_to_subjects.items() if not subs]

    def cells_per_subject(self) -> Dict[str, int]:
        return {s: int(len(idx)) for s, idx in self.subject_to_indices.items()}


# ─── Class-selection probabilities ─────────────────────────────────────────────

def class_selection_probabilities(
    index: SubjectClassIndex, strategy: str,
) -> Dict[int, float]:
    """
    Probability of selecting each OBSERVED effective class for one draw.
    Absent classes always get probability 0 and are never sampled — a rare-
    class policy that already excluded/merged a class must not be
    resurrected by the sampler.

    natural                   : proportional to unique-subject count per class
    uniform                   : equal probability across observed classes
    inverse_subject_frequency : proportional to 1 / unique-subject count,
                                 renormalized over observed classes
    """
    if strategy not in CLASS_SELECTION_STRATEGIES:
        raise SamplingConfigurationError(
            f"class_selection={strategy!r} must be one of {CLASS_SELECTION_STRATEGIES}"
        )
    observed = index.observed_classes
    if not observed:
        raise SamplingConfigurationError(
            "class_selection_probabilities: no observed effective classes in this "
            "training partition — nothing to sample."
        )

    counts = {c: len(index.class_to_subjects[c]) for c in observed}
    if strategy == "uniform":
        p = {c: 1.0 for c in observed}
    elif strategy == "natural":
        p = {c: float(counts[c]) for c in observed}
    else:  # inverse_subject_frequency
        p = {c: 1.0 / counts[c] for c in observed}

    total = sum(p.values())
    probs = {c: v / total for c, v in p.items()}
    full = {c: 0.0 for c in range(index.num_classes)}
    full.update(probs)
    return full


def subject_selection_probabilities(
    index: SubjectClassIndex, cls: int, strategy: str,
) -> Dict[str, float]:
    """Probability of selecting each subject WITHIN an already-chosen class.
    Deliberately independent of that subject's cell count."""
    if strategy not in SUBJECT_SELECTION_STRATEGIES:
        raise SamplingConfigurationError(
            f"subject_selection={strategy!r} must be one of {SUBJECT_SELECTION_STRATEGIES}"
        )
    subjects = index.class_to_subjects.get(cls, [])
    if not subjects:
        return {}
    p = 1.0 / len(subjects)
    return {s: p for s in subjects}


# ─── Diagnostics ────────────────────────────────────────────────────────────────

@dataclass
class SamplingDiagnostics:
    is_synthetic:                bool
    seed:                        int
    class_selection:              str
    subject_selection:            str
    replacement:                  bool
    cells_per_subject_cap:        Optional[int]
    batch_size:                   int
    batches_per_epoch:             int
    samples_per_epoch:            int          # exact total cells yielded per epoch
    observed_effective_classes:   List[int]
    absent_effective_classes:     List[int]
    unique_subjects_per_class:    Dict[int, int]
    cells_per_subject_summary:    Dict[str, float]     # min/median/max over all subjects
    expected_class_probabilities: Dict[int, float]
    expected_subject_probabilities: Dict[int, Dict[str, float]]
    realized_cells_per_class:     Optional[Dict[int, int]]   = None
    realized_subjects_per_class:  Optional[Dict[int, List[str]]] = None
    realized_unique_subjects:     Optional[int]              = None

    # ── Per-batch realized provenance (populated only after a COMPLETE
    # epoch has been iterated — see SubjectBalancedBatchSampler.__iter__).
    # These are what let a caller independently verify the sampling
    # contract was actually honored, not just what was requested.
    epoch_index:                          Optional[int]              = None
    complete:                             Optional[bool]              = None
    realized_total_samples:               Optional[int]              = None
    realized_batch_sizes:                 Optional[List[int]]        = None
    realized_cells_per_subject:           Optional[Dict[str, int]]   = None
    realized_subject_counts_per_batch:    Optional[List[Dict[str, int]]] = None
    realized_max_subject_cells_per_batch: Optional[List[int]]        = None
    realized_repeated_cell_draws_per_batch: Optional[List[int]]      = None
    realized_repeated_cell_draws_total:   Optional[int]              = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["unique_subjects_per_class"] = {str(k): v for k, v in d["unique_subjects_per_class"].items()}
        d["expected_class_probabilities"] = {str(k): v for k, v in d["expected_class_probabilities"].items()}
        d["expected_subject_probabilities"] = {
            str(k): v for k, v in d["expected_subject_probabilities"].items()
        }
        if d.get("realized_cells_per_class") is not None:
            d["realized_cells_per_class"] = {str(k): v for k, v in d["realized_cells_per_class"].items()}
        if d.get("realized_subjects_per_class") is not None:
            d["realized_subjects_per_class"] = {
                str(k): v for k, v in d["realized_subjects_per_class"].items()
            }
        if d.get("realized_cells_per_subject") is not None:
            d["realized_cells_per_subject"] = {str(k): v for k, v in d["realized_cells_per_subject"].items()}
        if d.get("realized_subject_counts_per_batch") is not None:
            d["realized_subject_counts_per_batch"] = [
                {str(k): v for k, v in batch.items()} for batch in d["realized_subject_counts_per_batch"]
            ]
        return d


def _cells_per_subject_summary(index: SubjectClassIndex) -> Dict[str, float]:
    counts = list(index.cells_per_subject().values())
    if not counts:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    arr = np.asarray(counts, dtype=np.float64)
    return {"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max())}


# ─── Batch sampler ──────────────────────────────────────────────────────────────

class SubjectBalancedBatchSampler(Sampler):
    """
    Yields lists of dataset indices (batches) following the class -> subject
    -> cell contract. Consume via:

        DataLoader(train_cell_dataset, batch_sampler=SubjectBalancedBatchSampler(...))

    Never combine with shuffle=True or another sampler — DataLoader forbids
    batch_sampler + shuffle/sampler/batch_size/drop_last together, which is
    the correct behaviour here too.

    cells_per_subject_cap, when set, is a HARD per-batch MAXIMUM: no subject
    may contribute more than that many cells to any single batch, with no
    exception and no silent fallback to an already-exhausted subject. It is
    a ceiling, not a required minimum — a subject with fewer cells than the
    cap is still a fully valid participant. Each subject's EFFECTIVE
    per-batch capacity is:

        replacement=True  : cells_per_subject_cap (redraws are allowed)
        replacement=False : min(cells_per_subject_cap, that subject's own
                             available unique cell count)

    Feasibility is judged against the SUM of effective capacities across
    every subject, not cap * subject_count — a configuration is only
    infeasible if that sum cannot fill the largest requested batch. Once a
    class's every subject has reached ITS effective capacity within the
    batch being built, that class simply stops being drawn for the REST of
    that batch (its probability mass is redistributed over classes that
    still have capacity) — this is a feasible degradation of the requested
    class balance, never a cap violation. Capacity resets fully at the
    start of every batch; without-replacement uniqueness is scoped to one
    batch, not the whole epoch — the same physical cell may reappear in a
    later batch. If every class is simultaneously exhausted before a batch
    reaches its target size, construction-time validation below is designed
    to make that unreachable; a SamplingImpossibleError is raised
    defensively if it happens anyway, rather than silently yielding a short
    or malformed batch.

    After a batch_sampler-driven DataLoader completes one full epoch,
    last_realized_diagnostics holds per-batch realized provenance (exact
    batch sizes, per-subject cell counts per batch, per-batch max subject
    contribution, and repeated physical-cell draws per batch) that lets a
    caller independently verify the contract above was actually honored —
    see SamplingDiagnostics. It is only set after a batch_sampler-driven
    DataLoader (or equivalent full consumption of __iter__) completes every
    batch of an epoch; an interrupted or failed epoch leaves it unchanged.
    """

    def __init__(
        self,
        index:                  SubjectClassIndex,
        batch_size:              int,
        seed:                    int,
        batches_per_epoch:        Optional[int] = None,
        samples_per_epoch:        Optional[int] = None,
        class_selection:          str  = "uniform",
        subject_selection:        str  = "uniform",
        cells_per_subject_cap:    Optional[int] = None,
        replacement:              bool = True,
    ):
        if batch_size <= 0:
            raise SamplingConfigurationError(f"batch_size must be positive, got {batch_size}.")
        if batches_per_epoch is not None and samples_per_epoch is not None:
            raise SamplingConfigurationError(
                "SubjectBalancedBatchSampler: specify at most one of batches_per_epoch/"
                "samples_per_epoch, not both (ambiguous epoch length)."
            )
        if batches_per_epoch is not None and batches_per_epoch <= 0:
            raise SamplingConfigurationError(f"batches_per_epoch must be positive, got {batches_per_epoch}.")
        if samples_per_epoch is not None and samples_per_epoch <= 0:
            raise SamplingConfigurationError(f"samples_per_epoch must be positive, got {samples_per_epoch}.")
        if cells_per_subject_cap is not None and cells_per_subject_cap <= 0:
            raise SamplingConfigurationError(
                f"cells_per_subject_cap must be positive when supplied, got {cells_per_subject_cap}."
            )

        self.index = index
        self.batch_size = batch_size
        self.seed = seed

        # _batch_sizes[i] is the EXACT number of cells batch i must contain.
        # samples_per_epoch is an exact contract: full_batches batches of
        # batch_size, followed by exactly one partial batch of the remainder
        # (never rounded up to a full extra batch — see requirement that
        # samples_per_epoch=95/batch_size=10 yields [10]*9 + [5], 95 total,
        # not 100). batches_per_epoch (explicit) always yields batch_size-
        # sized batches, matching the pre-existing contract for that mode.
        if samples_per_epoch is not None:
            full_batches, remainder = divmod(samples_per_epoch, batch_size)
            self._batch_sizes = [batch_size] * full_batches + ([remainder] if remainder else [])
            if not self._batch_sizes:
                self._batch_sizes = [samples_per_epoch]  # samples_per_epoch < batch_size
            self.batches_per_epoch = len(self._batch_sizes)
        elif batches_per_epoch is not None:
            self.batches_per_epoch = batches_per_epoch
            self._batch_sizes = [batch_size] * batches_per_epoch
        else:
            self.batches_per_epoch = max(1, -(-len(index.subject_ids) // batch_size))
            self._batch_sizes = [batch_size] * self.batches_per_epoch
        self.samples_per_epoch = sum(self._batch_sizes)

        self.class_selection = class_selection
        self.subject_selection = subject_selection
        self.cells_per_subject_cap = cells_per_subject_cap
        self.replacement = replacement
        self._epoch = 0
        self.last_realized_diagnostics: Optional[SamplingDiagnostics] = None

        self.class_probs = class_selection_probabilities(index, class_selection)
        self.classes_present = [c for c in index.observed_classes]
        self._class_p = np.array([self.class_probs[c] for c in self.classes_present], dtype=np.float64)

        self.subject_probs = {
            c: subject_selection_probabilities(index, c, subject_selection)
            for c in self.classes_present
        }

        if not self.replacement and self.cells_per_subject_cap is None:
            raise SamplingConfigurationError(
                "SubjectBalancedBatchSampler: replacement=False requires an explicit "
                "cells_per_subject_cap so the maximum cells needed from any one subject "
                "in a batch is known in advance."
            )

        # cells_per_subject_cap is a MAXIMUM contribution, not a required
        # minimum cell count — a subject with fewer cells than the cap is
        # still a fully valid participant, just with a smaller effective
        # capacity. With replacement, a subject can be redrawn from, so its
        # effective per-batch capacity is the cap itself. Without
        # replacement, a subject cannot yield more DISTINCT cells than it
        # actually has, so its effective capacity is capped further by its
        # own available unique cell count.
        self._effective_capacity: Optional[Dict[str, int]] = None
        if self.cells_per_subject_cap is not None:
            if self.replacement:
                self._effective_capacity = {
                    s: self.cells_per_subject_cap for s in self.index.unique_subjects
                }
            else:
                self._effective_capacity = {
                    s: min(self.cells_per_subject_cap, len(self.index.subject_to_indices[s]))
                    for s in self.index.unique_subjects
                }

        # Hard feasibility check: the maximum cells any single batch needs
        # must not exceed the total EFFECTIVE capacity available across
        # every subject (capacity resets fully at the start of each batch,
        # so this bound applies independently to every batch). If it does,
        # no batch could ever be filled to its required size without
        # violating the cap — fail now, deterministically, rather than
        # raising deep inside a later __iter__() call or (worse) silently
        # yielding a short/malformed batch.
        if self.cells_per_subject_cap is not None:
            total_capacity = sum(self._effective_capacity.values())
            max_batch_needed = max(self._batch_sizes) if self._batch_sizes else 0
            if total_capacity < max_batch_needed:
                raise SamplingImpossibleError(
                    "SubjectBalancedBatchSampler: total effective per-batch capacity "
                    f"(sum of min(cells_per_subject_cap, available cells) per subject when "
                    f"replacement=False, else cells_per_subject_cap x subject count) = "
                    f"{total_capacity} across {len(self.index.unique_subjects)} unique subject(s), "
                    f"which is less than the {max_batch_needed} cells the largest requested batch "
                    "needs. Lower batch_size/samples_per_epoch's per-batch size, raise "
                    "cells_per_subject_cap, or provide more subjects/cells."
                )

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _expected_diagnostics(self) -> SamplingDiagnostics:
        return SamplingDiagnostics(
            is_synthetic=self.index.is_synthetic,
            seed=self.seed,
            class_selection=self.class_selection,
            subject_selection=self.subject_selection,
            replacement=self.replacement,
            cells_per_subject_cap=self.cells_per_subject_cap,
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            samples_per_epoch=self.samples_per_epoch,
            observed_effective_classes=list(self.classes_present),
            absent_effective_classes=list(self.index.absent_classes),
            unique_subjects_per_class={c: len(self.index.class_to_subjects[c])
                                        for c in range(self.index.num_classes)},
            cells_per_subject_summary=_cells_per_subject_summary(self.index),
            expected_class_probabilities=dict(self.class_probs),
            expected_subject_probabilities=dict(self.subject_probs),
        )

    def diagnostics(self) -> SamplingDiagnostics:
        """Expected (config-derived) diagnostics — does not consume randomness
        or require iteration. See last_realized_diagnostics for an actually-
        sampled epoch's realized counts, populated after DataLoader iterates
        this sampler once."""
        return self._expected_diagnostics()

    def _derive_epoch_seed(self) -> int:
        # Deterministic per (seed, epoch) — same seed + same epoch index always
        # reproduces the same batch sequence; different epochs within one run
        # draw different (but still seed-derived) sequences.
        return int((self.seed * 1_000_003 + self._epoch) % (2**31 - 1))

    def __iter__(self):
        this_epoch = self._epoch
        rng = np.random.RandomState(self._derive_epoch_seed())
        self._epoch += 1
        cap = self.cells_per_subject_cap
        effective_capacity = self._effective_capacity  # None, or {subject: capacity}

        realized_cells: Dict[int, int] = {c: 0 for c in self.classes_present}
        realized_subjects: Dict[int, set] = {c: set() for c in self.classes_present}

        # Epoch-level realized provenance — only committed to
        # last_realized_diagnostics after every batch below has yielded
        # successfully, so a failed/interrupted epoch never leaves stale
        # partial counts mislabeled as a completed one.
        realized_batch_sizes: List[int] = []
        realized_cells_per_subject: Dict[str, int] = {}
        realized_subject_counts_per_batch: List[Dict[str, int]] = []
        realized_max_subject_cells_per_batch: List[int] = []
        realized_repeated_cell_draws_per_batch: List[int] = []

        for batch_size in self._batch_sizes:
            batch: List[int] = []
            per_subject_used: Dict[str, int] = {}
            no_replace_used: Dict[str, set] = {}
            # Per-batch remaining-capacity bookkeeping (the cap is a HARD,
            # per-batch constraint — capacity resets fully at the start of
            # every batch). class_remaining_subjects[c] holds exactly the
            # subjects of class c that have not yet reached their EFFECTIVE
            # capacity WITHIN this batch; once it empties, class c is
            # skipped for the rest of this batch (never a fallback to an
            # exhausted subject).
            class_remaining_subjects: Dict[int, List[str]] = (
                {
                    c: [s for s in self.index.class_to_subjects[c] if effective_capacity[s] > 0]
                    for c in self.classes_present
                }
                if cap is not None else {}
            )

            for _ in range(batch_size):
                if cap is None:
                    eligible_classes = self.classes_present
                    eligible_p = self._class_p
                else:
                    eligible_classes = [c for c in self.classes_present if class_remaining_subjects[c]]
                    if not eligible_classes:
                        raise SamplingImpossibleError(
                            "SubjectBalancedBatchSampler: every observed class's subjects "
                            f"reached their effective cells_per_subject_cap={cap} before this "
                            f"batch reached its required size ({batch_size}) — this should have "
                            "been prevented by the construction-time capacity check; lower "
                            "batch_size/samples_per_epoch's per-batch size or raise "
                            "cells_per_subject_cap."
                        )
                    raw_p = np.array([self.class_probs[c] for c in eligible_classes], dtype=np.float64)
                    eligible_p = raw_p / raw_p.sum()

                cls = int(rng.choice(eligible_classes, p=eligible_p))
                candidates = class_remaining_subjects[cls] if cap is not None else self.index.class_to_subjects[cls]
                subj = str(rng.choice(candidates))

                cell_idx = self.index.subject_to_indices[subj]
                if self.replacement:
                    chosen = int(rng.choice(cell_idx))
                else:
                    used = no_replace_used.setdefault(subj, set())
                    avail = np.array([i for i in cell_idx if i not in used])
                    if len(avail) == 0:
                        raise SamplingImpossibleError(
                            f"SubjectBalancedBatchSampler: subject {subj!r} has no remaining "
                            "unused cells within this batch under replacement=False — this "
                            "should have been prevented by the construction-time capacity check."
                        )
                    chosen = int(rng.choice(avail))
                    used.add(chosen)

                per_subject_used[subj] = per_subject_used.get(subj, 0) + 1
                if cap is not None and per_subject_used[subj] >= effective_capacity[subj]:
                    class_remaining_subjects[cls].remove(subj)
                batch.append(chosen)
                realized_cells[cls] += 1
                realized_subjects[cls].add(subj)

            if cap is not None:
                for subj, count in per_subject_used.items():
                    if count > effective_capacity[subj]:
                        raise SamplingImpossibleError(
                            "SubjectBalancedBatchSampler: internal invariant violated — subject "
                            f"{subj!r} received {count} cells in one batch, exceeding its "
                            f"effective capacity of {effective_capacity[subj]}."
                        )

            realized_batch_sizes.append(len(batch))
            for subj, count in per_subject_used.items():
                realized_cells_per_subject[subj] = realized_cells_per_subject.get(subj, 0) + count
            realized_subject_counts_per_batch.append(dict(per_subject_used))
            realized_max_subject_cells_per_batch.append(
                max(per_subject_used.values()) if per_subject_used else 0
            )
            realized_repeated_cell_draws_per_batch.append(len(batch) - len(set(batch)))

            yield batch

        realized_all_subjects = set()
        for subs in realized_subjects.values():
            realized_all_subjects |= subs
        diag = self._expected_diagnostics()
        diag.realized_cells_per_class = dict(realized_cells)
        diag.realized_subjects_per_class = {c: sorted(s) for c, s in realized_subjects.items()}
        diag.realized_unique_subjects = len(realized_all_subjects)
        diag.epoch_index = this_epoch
        diag.complete = True
        diag.realized_total_samples = sum(realized_batch_sizes)
        diag.realized_batch_sizes = realized_batch_sizes
        diag.realized_cells_per_subject = realized_cells_per_subject
        diag.realized_subject_counts_per_batch = realized_subject_counts_per_batch
        diag.realized_max_subject_cells_per_batch = realized_max_subject_cells_per_batch
        diag.realized_repeated_cell_draws_per_batch = realized_repeated_cell_draws_per_batch
        diag.realized_repeated_cell_draws_total = sum(realized_repeated_cell_draws_per_batch)
        self.last_realized_diagnostics = diag


# ─── Configuration resolution ───────────────────────────────────────────────────

SAMPLER_MODES = ("shuffle", "subject_balanced")
CLASS_WEIGHTING_MODES = ("none", "inverse_frequency")
FOCAL_ALPHA_MODES = ("class_weights", "none")

DEFAULT_SMOKE_IMBALANCE_CONFIG = {
    # "shuffle" preserves the pre-Phase-2 DataLoader(shuffle=True) behaviour
    # exactly — absent configuration must not change existing runs.
    "sampler":                     "shuffle",
    "class_selection":             "uniform",
    "subject_selection":           "uniform",
    "batches_per_epoch":           None,
    "samples_per_epoch":           None,
    "cells_per_subject_per_batch": None,
    "replacement":                 True,
    # "inverse_frequency" preserves Trainer.phase1's pre-Phase-2 default of
    # auto-computing train-only inverse-frequency smoke class weights.
    "class_weighting":             "inverse_frequency",
    "loss":                        "cross_entropy",
    "focal_gamma":                 2.0,
    "focal_alpha_mode":            "class_weights",
    "seed_offset":                 0,
}


def resolve_smoke_imbalance_config(cfg: Optional[dict]) -> dict:
    """
    Merge a (possibly partial or absent) training.smoke_imbalance config
    block with DEFAULT_SMOKE_IMBALANCE_CONFIG and validate every value.
    Absent configuration resolves to the explicit backward-compatible
    defaults above — old configs/checkpoints without this section behave
    exactly as they did before Phase 2. Never depends on dict iteration
    order: every key is looked up by name, not iterated positionally.
    """
    resolved = dict(DEFAULT_SMOKE_IMBALANCE_CONFIG)
    resolved.update(cfg or {})

    unknown = set(resolved) - set(DEFAULT_SMOKE_IMBALANCE_CONFIG)
    if unknown:
        raise SamplingConfigurationError(
            f"smoke_imbalance config has unknown key(s) {sorted(unknown)} — "
            f"valid keys are {sorted(DEFAULT_SMOKE_IMBALANCE_CONFIG)}."
        )

    if resolved["sampler"] not in SAMPLER_MODES:
        raise SamplingConfigurationError(
            f"smoke_imbalance.sampler={resolved['sampler']!r} must be one of {SAMPLER_MODES}"
        )
    if resolved["class_selection"] not in CLASS_SELECTION_STRATEGIES:
        raise SamplingConfigurationError(
            f"smoke_imbalance.class_selection={resolved['class_selection']!r} must be one of "
            f"{CLASS_SELECTION_STRATEGIES}"
        )
    if resolved["subject_selection"] not in SUBJECT_SELECTION_STRATEGIES:
        raise SamplingConfigurationError(
            f"smoke_imbalance.subject_selection={resolved['subject_selection']!r} must be one of "
            f"{SUBJECT_SELECTION_STRATEGIES}"
        )
    if resolved["class_weighting"] not in CLASS_WEIGHTING_MODES:
        raise SamplingConfigurationError(
            f"smoke_imbalance.class_weighting={resolved['class_weighting']!r} must be one of "
            f"{CLASS_WEIGHTING_MODES}"
        )
    if resolved["loss"] not in ("cross_entropy", "focal"):
        raise SamplingConfigurationError(
            f"smoke_imbalance.loss={resolved['loss']!r} must be 'cross_entropy' or 'focal'"
        )
    if resolved["focal_alpha_mode"] not in FOCAL_ALPHA_MODES:
        raise SamplingConfigurationError(
            f"smoke_imbalance.focal_alpha_mode={resolved['focal_alpha_mode']!r} must be one of "
            f"{FOCAL_ALPHA_MODES}"
        )
    if resolved["focal_gamma"] < 0:
        raise SamplingConfigurationError(
            f"smoke_imbalance.focal_gamma must be >= 0, got {resolved['focal_gamma']}."
        )
    if resolved["batches_per_epoch"] is not None and resolved["samples_per_epoch"] is not None:
        raise SamplingConfigurationError(
            "smoke_imbalance: specify at most one of batches_per_epoch/samples_per_epoch."
        )
    for key in ("batches_per_epoch", "samples_per_epoch", "cells_per_subject_per_batch"):
        v = resolved[key]
        if v is not None and v <= 0:
            raise SamplingConfigurationError(f"smoke_imbalance.{key} must be positive when set, got {v}.")
    if not isinstance(resolved["replacement"], bool):
        raise SamplingConfigurationError(
            f"smoke_imbalance.replacement must be a bool, got {resolved['replacement']!r}."
        )
    return resolved


def build_subject_balanced_sampler(
    dataset, num_classes: int, batch_size: int, seed: int, resolved_cfg: dict,
) -> SubjectBalancedBatchSampler:
    """Construct a SubjectBalancedBatchSampler for `dataset` from an already
    -resolved smoke_imbalance config dict (see resolve_smoke_imbalance_config)."""
    index = SubjectClassIndex.from_cell_dataset(dataset, num_classes=num_classes)
    return SubjectBalancedBatchSampler(
        index,
        batch_size=batch_size,
        seed=seed + resolved_cfg.get("seed_offset", 0),
        batches_per_epoch=resolved_cfg["batches_per_epoch"],
        samples_per_epoch=resolved_cfg["samples_per_epoch"],
        class_selection=resolved_cfg["class_selection"],
        subject_selection=resolved_cfg["subject_selection"],
        cells_per_subject_cap=resolved_cfg["cells_per_subject_per_batch"],
        replacement=resolved_cfg["replacement"],
    )
