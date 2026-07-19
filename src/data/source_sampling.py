"""
data/source_sampling.py — source-first subject-balanced training sampling.

Extends the class -> subject -> cell hierarchy in data/sampling.py with an
outer source level: source -> subject -> cell. A subject's cell count never
controls its own selection probability (same guarantee sampling.py makes for
smoke class), and now a source's SUBJECT COUNT never controls its own
selection probability either — a source with many subjects does not
dominate sampling purely because it has more subjects than a smaller source.

This is a distinct, independent sampler from SubjectBalancedBatchSampler
(smoke-class balancing) — the two solve different imbalance problems (label
imbalance vs. domain/source imbalance) and are not composed together in this
module. Only the TRAINING split may ever use this; validation and held-out-
source evaluation must remain the natural, unweighted distribution, and a
held-out source must never appear in the index this sampler is built from —
enforced by the caller (the source-held-out protocol only ever builds this
sampler from development-source training subjects).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler

from data.sampling import SamplingConfigurationError, SamplingImpossibleError

_PLACEHOLDER_SUBJECT_IDS = {"unknown", "", "none", "None", "nan"}

SOURCE_SELECTION_STRATEGIES = ("uniform", "natural", "inverse_subject_frequency")
SUBJECT_SELECTION_STRATEGIES = ("uniform",)


@dataclass
class SourceSubjectIndex:
    """Deterministic source -> subject -> cell-index mapping, built once
    from per-cell subject ids and per-cell dataset_source labels. A subject
    assigned to more than one source is a data-assembly inconsistency (see
    benchmarks/ood.py::_find_cross_source_subjects, which performs the same
    check for the LOSO protocol) and is rejected here too, never silently
    resolved by picking one source."""

    subject_ids: np.ndarray   # [N] str
    sources:     np.ndarray   # [N] str
    is_synthetic: bool = False

    unique_sources:      List[str]              = field(init=False)
    unique_subjects:     List[str]              = field(init=False)
    subject_to_indices:  Dict[str, np.ndarray]  = field(init=False)
    subject_to_source:   Dict[str, str]         = field(init=False)
    source_to_subjects:  Dict[str, List[str]]   = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.subject_ids)
        if n == 0:
            raise SamplingConfigurationError("SourceSubjectIndex: dataset is empty — nothing to sample from.")
        if len(self.sources) != n:
            raise SamplingConfigurationError(
                f"SourceSubjectIndex: len(subject_ids)={n} != len(sources)={len(self.sources)}."
            )

        subj = np.array([str(s) for s in self.subject_ids], dtype=object)
        placeholder_mask = np.isin(subj, list(_PLACEHOLDER_SUBJECT_IDS))
        if placeholder_mask.any():
            bad = sorted(set(subj[placeholder_mask].tolist()))
            raise SamplingConfigurationError(
                f"SourceSubjectIndex: {int(placeholder_mask.sum())}/{n} cell(s) have a placeholder "
                f"subject_id {bad} — source-balanced sampling requires a real subject_id per cell."
            )
        self.subject_ids = subj
        self.sources = np.array([str(s) for s in self.sources], dtype=object)

        subject_to_indices: Dict[str, List[int]] = {}
        for i, s in enumerate(self.subject_ids):
            subject_to_indices.setdefault(s, []).append(i)

        subject_to_source: Dict[str, str] = {}
        for s, idxs in subject_to_indices.items():
            srcs = set(self.sources[i] for i in idxs)
            if len(srcs) > 1:
                raise SamplingConfigurationError(
                    f"SourceSubjectIndex: subject {s!r} is assigned to more than one dataset_source "
                    f"{sorted(srcs)} — a subject must map to exactly one source. This indicates a "
                    "data-assembly inconsistency, not something to resolve by picking one here."
                )
            subject_to_source[s] = srcs.pop()

        self.unique_subjects = sorted(subject_to_indices)
        self.subject_to_indices = {s: np.array(idxs, dtype=np.int64) for s, idxs in subject_to_indices.items()}
        self.subject_to_source = subject_to_source

        source_to_subjects: Dict[str, List[str]] = {}
        for s in self.unique_subjects:
            source_to_subjects.setdefault(subject_to_source[s], []).append(s)
        self.source_to_subjects = {src: sorted(subs) for src, subs in source_to_subjects.items()}
        self.unique_sources = sorted(self.source_to_subjects)

    @classmethod
    def from_cell_dataset(cls, dataset) -> "SourceSubjectIndex":
        return cls(
            subject_ids=np.asarray(dataset.subject_ids),
            sources=np.asarray(dataset.dataset_source),
            is_synthetic=bool(getattr(dataset, "diagnostic_mode", False)),
        )

    def cells_per_subject(self) -> Dict[str, int]:
        return {s: int(len(idx)) for s, idx in self.subject_to_indices.items()}


def source_selection_probabilities(index: SourceSubjectIndex, strategy: str) -> Dict[str, float]:
    """Probability of selecting each observed source for one draw.
    'natural' is proportional to unique-subject count per source (which is
    exactly the quantity a plain per-cell or per-subject scheme would let
    dominate); 'uniform' and 'inverse_subject_frequency' counteract that."""
    if strategy not in SOURCE_SELECTION_STRATEGIES:
        raise SamplingConfigurationError(
            f"source_selection={strategy!r} must be one of {SOURCE_SELECTION_STRATEGIES}"
        )
    sources = index.unique_sources
    if not sources:
        raise SamplingConfigurationError("source_selection_probabilities: no sources present.")
    counts = {s: len(index.source_to_subjects[s]) for s in sources}
    if strategy == "uniform":
        p = {s: 1.0 for s in sources}
    elif strategy == "natural":
        p = {s: float(counts[s]) for s in sources}
    else:
        p = {s: 1.0 / counts[s] for s in sources}
    total = sum(p.values())
    return {s: v / total for s, v in p.items()}


def subject_selection_probabilities_within_source(
    index: SourceSubjectIndex, source: str, strategy: str,
) -> Dict[str, float]:
    if strategy not in SUBJECT_SELECTION_STRATEGIES:
        raise SamplingConfigurationError(
            f"subject_selection={strategy!r} must be one of {SUBJECT_SELECTION_STRATEGIES}"
        )
    subjects = index.source_to_subjects.get(source, [])
    if not subjects:
        return {}
    p = 1.0 / len(subjects)
    return {s: p for s in subjects}


@dataclass
class SourceSamplingDiagnostics:
    is_synthetic: bool
    seed: int
    source_selection: str
    subject_selection: str
    batch_size: int
    batches_per_epoch: int
    samples_per_epoch: int
    sources_present: List[str]
    unique_subjects_per_source: Dict[str, int]
    expected_source_probabilities: Dict[str, float]
    realized_samples_per_source: Optional[Dict[str, int]] = None
    realized_subjects_per_source: Optional[Dict[str, List[str]]] = None
    realized_samples_per_subject: Optional[Dict[str, int]] = None
    complete: Optional[bool] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class SourceBalancedBatchSampler(Sampler):
    """
    Yields lists of dataset indices following source -> subject -> cell.
    cells_per_subject_cap is a hard per-batch maximum, mirroring
    SubjectBalancedBatchSampler's contract exactly (see that class's
    docstring for the full capacity/feasibility reasoning — reused here
    unchanged, just keyed by source->subject instead of class->subject).
    samples_per_epoch is an exact contract (full batches + one partial
    remainder, never rounded up).

    This sampler must never be constructed from a validation or held-out-
    source population — enforced by the caller, not by this class (which
    has no way to know which subjects are "development" ones).
    """

    def __init__(
        self,
        index: SourceSubjectIndex,
        batch_size: int,
        seed: int,
        samples_per_epoch: Optional[int] = None,
        source_selection: str = "uniform",
        subject_selection: str = "uniform",
        cells_per_subject_cap: Optional[int] = None,
        replacement: bool = True,
    ):
        if batch_size <= 0:
            raise SamplingConfigurationError(f"batch_size must be positive, got {batch_size}.")
        if samples_per_epoch is not None and samples_per_epoch <= 0:
            raise SamplingConfigurationError(f"samples_per_epoch must be positive, got {samples_per_epoch}.")
        if cells_per_subject_cap is not None and cells_per_subject_cap <= 0:
            raise SamplingConfigurationError(
                f"cells_per_subject_cap must be positive when supplied, got {cells_per_subject_cap}."
            )
        if not replacement and cells_per_subject_cap is None:
            raise SamplingConfigurationError(
                "SourceBalancedBatchSampler: replacement=False requires an explicit cells_per_subject_cap."
            )

        self.index = index
        self.batch_size = batch_size
        self.seed = seed
        self.replacement = replacement
        self.cells_per_subject_cap = cells_per_subject_cap

        if samples_per_epoch is not None:
            full_batches, remainder = divmod(samples_per_epoch, batch_size)
            self._batch_sizes = [batch_size] * full_batches + ([remainder] if remainder else [])
            if not self._batch_sizes:
                self._batch_sizes = [samples_per_epoch]
        else:
            n_cells = sum(len(idx) for idx in index.subject_to_indices.values())
            self.batches_per_epoch_requested = max(1, -(-n_cells // batch_size))
            self._batch_sizes = [batch_size] * self.batches_per_epoch_requested
        self.batches_per_epoch = len(self._batch_sizes)
        self.samples_per_epoch = sum(self._batch_sizes)

        self.source_selection = source_selection
        self.subject_selection = subject_selection
        self.sources_present = list(index.unique_sources)
        self.source_probs = source_selection_probabilities(index, source_selection)
        self._source_p = np.array([self.source_probs[s] for s in self.sources_present], dtype=np.float64)
        self.subject_probs = {
            s: subject_selection_probabilities_within_source(index, s, subject_selection)
            for s in self.sources_present
        }

        self._effective_capacity: Optional[Dict[str, int]] = None
        if cells_per_subject_cap is not None:
            if replacement:
                self._effective_capacity = {s: cells_per_subject_cap for s in index.unique_subjects}
            else:
                self._effective_capacity = {
                    s: min(cells_per_subject_cap, len(index.subject_to_indices[s]))
                    for s in index.unique_subjects
                }
            total_capacity = sum(self._effective_capacity.values())
            max_batch_needed = max(self._batch_sizes) if self._batch_sizes else 0
            if total_capacity < max_batch_needed:
                raise SamplingImpossibleError(
                    "SourceBalancedBatchSampler: total effective per-batch capacity "
                    f"({total_capacity}) across {len(index.unique_subjects)} unique subject(s) is less "
                    f"than the {max_batch_needed} cells the largest requested batch needs. Lower "
                    "batch_size/samples_per_epoch, raise cells_per_subject_cap, or provide more subjects."
                )

        self.last_realized_diagnostics: Optional[SourceSamplingDiagnostics] = None

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _expected_diagnostics(self) -> SourceSamplingDiagnostics:
        return SourceSamplingDiagnostics(
            is_synthetic=self.index.is_synthetic, seed=self.seed,
            source_selection=self.source_selection, subject_selection=self.subject_selection,
            batch_size=self.batch_size, batches_per_epoch=self.batches_per_epoch,
            samples_per_epoch=self.samples_per_epoch, sources_present=list(self.sources_present),
            unique_subjects_per_source={s: len(self.index.source_to_subjects[s]) for s in self.sources_present},
            expected_source_probabilities=dict(self.source_probs),
        )

    def diagnostics(self) -> SourceSamplingDiagnostics:
        return self._expected_diagnostics()

    def _derive_epoch_seed(self, epoch: int) -> int:
        return int((self.seed * 1_000_003 + epoch) % (2**31 - 1))

    def __iter__(self):
        rng = np.random.RandomState(self._derive_epoch_seed(getattr(self, "_epoch", 0)))
        self._epoch = getattr(self, "_epoch", 0) + 1
        cap = self.cells_per_subject_cap
        effective_capacity = self._effective_capacity

        realized_samples: Dict[str, int] = {s: 0 for s in self.sources_present}
        realized_subjects: Dict[str, set] = {s: set() for s in self.sources_present}
        realized_samples_per_subject: Dict[str, int] = {}

        for batch_size in self._batch_sizes:
            batch: List[int] = []
            per_subject_used: Dict[str, int] = {}
            no_replace_used: Dict[str, set] = {}
            source_remaining_subjects: Dict[str, List[str]] = (
                {s: [sub for sub in self.index.source_to_subjects[s] if effective_capacity[sub] > 0]
                 for s in self.sources_present} if cap is not None else {}
            )

            for _ in range(batch_size):
                if cap is None:
                    eligible_sources = self.sources_present
                    eligible_p = self._source_p
                else:
                    eligible_sources = [s for s in self.sources_present if source_remaining_subjects[s]]
                    if not eligible_sources:
                        raise SamplingImpossibleError(
                            "SourceBalancedBatchSampler: every source's subjects reached their "
                            f"effective cells_per_subject_cap={cap} before this batch reached its "
                            f"required size ({batch_size})."
                        )
                    raw_p = np.array([self.source_probs[s] for s in eligible_sources], dtype=np.float64)
                    eligible_p = raw_p / raw_p.sum()

                src = str(rng.choice(eligible_sources, p=eligible_p))
                candidates = source_remaining_subjects[src] if cap is not None else self.index.source_to_subjects[src]
                subj = str(rng.choice(candidates))

                cell_idx = self.index.subject_to_indices[subj]
                if self.replacement:
                    chosen = int(rng.choice(cell_idx))
                else:
                    used = no_replace_used.setdefault(subj, set())
                    avail = np.array([i for i in cell_idx if i not in used])
                    if len(avail) == 0:
                        raise SamplingImpossibleError(
                            f"SourceBalancedBatchSampler: subject {subj!r} has no remaining unused "
                            "cells within this batch under replacement=False."
                        )
                    chosen = int(rng.choice(avail))
                    used.add(chosen)

                per_subject_used[subj] = per_subject_used.get(subj, 0) + 1
                if cap is not None and per_subject_used[subj] >= effective_capacity[subj]:
                    source_remaining_subjects[src].remove(subj)
                batch.append(chosen)
                realized_samples[src] += 1
                realized_subjects[src].add(subj)

            for subj, count in per_subject_used.items():
                realized_samples_per_subject[subj] = realized_samples_per_subject.get(subj, 0) + count
            yield batch

        diag = self._expected_diagnostics()
        diag.realized_samples_per_source = dict(realized_samples)
        diag.realized_subjects_per_source = {s: sorted(v) for s, v in realized_subjects.items()}
        diag.realized_samples_per_subject = realized_samples_per_subject
        diag.complete = True
        self.last_realized_diagnostics = diag


def build_source_balanced_sampler(
    dataset, batch_size: int, seed: int, samples_per_epoch: Optional[int] = None,
    source_selection: str = "uniform", subject_selection: str = "uniform",
    cells_per_subject_cap: Optional[int] = None, replacement: bool = True,
    excluded_subjects: Optional[Sequence[str]] = None,
) -> SourceBalancedBatchSampler:
    """
    Construct a SourceBalancedBatchSampler from a CellLevelDataset (or
    fold-refit equivalent). excluded_subjects (e.g. the held-out source's
    subjects, or validation subjects) is removed BEFORE the index is built —
    a subject named there can never be drawn, not merely down-weighted.
    """
    subject_ids = np.asarray(dataset.subject_ids)
    sources = np.asarray(dataset.dataset_source)
    if excluded_subjects:
        excluded = set(str(s) for s in excluded_subjects)
        keep = np.array([str(s) not in excluded for s in subject_ids])
        subject_ids = subject_ids[keep]
        sources = sources[keep]
    index = SourceSubjectIndex(
        subject_ids=subject_ids, sources=sources,
        is_synthetic=bool(getattr(dataset, "diagnostic_mode", False)),
    )
    return SourceBalancedBatchSampler(
        index, batch_size=batch_size, seed=seed, samples_per_epoch=samples_per_epoch,
        source_selection=source_selection, subject_selection=subject_selection,
        cells_per_subject_cap=cells_per_subject_cap, replacement=replacement,
    )
