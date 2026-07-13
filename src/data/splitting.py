"""
data/splitting.py — subject-level train/val/test splitting and grouped k-fold CV.

Cell-level random splitting leaks subject identity across splits: a subject's
cells share genetic background, batch, and technical variation, so a model
that has seen 80% of a subject's cells in training will trivially recognise
the other 20% at test time. Every split produced here groups by subject_id,
so a subject's cells (or a bulk subject's single sample) always land in
exactly one split.

All splits are deterministic given (subject_ids, labels, seed) and are saved
as JSON manifests so an experiment can be exactly reproduced or audited later
without re-running the random split.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


MIN_SUBJECTS_FOR_STRATIFICATION = 3  # need >=1 subject per split to stratify a class


class StaleManifestError(ValueError):
    """Raised when an on-disk split manifest no longer matches the current data/config."""


def compute_dataset_fingerprint(
    subject_ids:   Sequence,
    labels:        Optional[Sequence] = None,
    train_frac:    float = 0.70,
    val_frac:      float = 0.15,
    test_frac:     float = 0.15,
    seed:          int = 42,
    rare_class_policy: Optional[str] = None,
    extra:         Optional[Dict] = None,
) -> str:
    """
    Deterministic hash of everything that determines a split's validity:
    the subject set, each subject's final effective label, split fractions,
    seed, and the rare-class policy in effect when the split was made.
    Any change to these invalidates a saved manifest — reusing it silently
    would either leak new subjects into no split, keep stale subjects that
    no longer exist, or apply a split computed under a different label
    mapping than the one now in use.
    """
    subject_labels = _unique_subject_labels(subject_ids, labels)
    payload = {
        "subjects":    sorted((sid, str(lab)) for sid, lab in subject_labels.items()),
        "train_frac":  round(train_frac, 6),
        "val_frac":    round(val_frac, 6),
        "test_frac":   round(test_frac, 6),
        "seed":        seed,
        "rare_class_policy": rare_class_policy,
        "extra":       extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ─── Manifest ──────────────────────────────────────────────────────────────

@dataclass
class SplitManifest:
    """Reproducible record of which subjects landed in which split."""

    seed:           int
    train_subjects: List[str]
    val_subjects:   List[str]
    test_subjects:  List[str]
    report:         Dict = field(default_factory=dict)
    fingerprint:    Optional[str] = None

    def __post_init__(self):
        self.train_subjects = [str(s) for s in self.train_subjects]
        self.val_subjects   = [str(s) for s in self.val_subjects]
        self.test_subjects  = [str(s) for s in self.test_subjects]
        self._assert_no_overlap()

    def _assert_no_overlap(self) -> None:
        tr, va, te = set(self.train_subjects), set(self.val_subjects), set(self.test_subjects)
        overlap = (tr & va) | (tr & te) | (va & te)
        if overlap:
            raise ValueError(f"Subject leakage across splits detected: {sorted(overlap)}")

    def split_of(self, subject_id) -> Optional[str]:
        sid = str(subject_id)
        if sid in self.train_subjects: return "train"
        if sid in self.val_subjects:   return "val"
        if sid in self.test_subjects:  return "test"
        return None

    def subjects_for(self, split: str) -> List[str]:
        return {"train": self.train_subjects, "val": self.val_subjects,
                "test": self.test_subjects}[split]

    def to_dict(self) -> Dict:
        return {
            "seed":           self.seed,
            "train_subjects": self.train_subjects,
            "val_subjects":   self.val_subjects,
            "test_subjects":  self.test_subjects,
            "report":         self.report,
            "fingerprint":    self.fingerprint,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[splitting] manifest saved → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SplitManifest":
        with open(path) as f:
            d = json.load(f)
        return cls(
            seed=d["seed"], train_subjects=d["train_subjects"],
            val_subjects=d["val_subjects"], test_subjects=d["test_subjects"],
            report=d.get("report", {}), fingerprint=d.get("fingerprint"),
        )


# ─── Helpers ───────────────────────────────────────────────────────────────

def _unique_subject_labels(
    subject_ids: Sequence, labels: Optional[Sequence]
) -> Dict[str, object]:
    """One label per unique subject. Raises if a subject has conflicting labels."""
    if labels is None:
        return {str(s): None for s in set(str(x) for x in subject_ids)}
    per_subject: Dict[str, set] = {}
    for sid, lab in zip(subject_ids, labels):
        per_subject.setdefault(str(sid), set()).add(lab)
    out = {}
    for sid, labset in per_subject.items():
        if len(labset) > 1:
            raise ValueError(
                f"Subject {sid} has conflicting labels {labset} — "
                "labels passed to splitting must be per-subject, not per-cell."
            )
        out[sid] = next(iter(labset))
    return out


def _class_buckets(subject_labels: Dict[str, object]) -> Dict[object, List[str]]:
    buckets: Dict[object, List[str]] = {}
    for sid, lab in subject_labels.items():
        buckets.setdefault(lab, []).append(sid)
    return buckets


# ─── Train/val/test split ──────────────────────────────────────────────────

def subject_train_val_test_split(
    subject_ids:    Sequence,
    labels:         Optional[Sequence] = None,
    train_frac:     float = 0.70,
    val_frac:       float = 0.15,
    test_frac:      float = 0.15,
    seed:           int = 42,
    cell_counts:    Optional[Dict[str, int]] = None,
    manifest_path:  Optional[Union[str, Path]] = None,
    rare_class_policy: Optional[str] = None,
) -> SplitManifest:
    """
    Group-aware train/val/test split by subject_id.

    Parameters
    ----------
    subject_ids : one entry per cell/sample (subjects repeat).
    labels      : optional, aligned 1:1 with subject_ids — the label used
                  for stratification (e.g. per-cell smoke_type; every cell
                  belonging to one subject must carry the same label here,
                  since subjects are the split unit).
    cell_counts : optional {subject_id: n_cells} for the report; computed
                  from subject_ids if omitted.
    manifest_path : if given, the resulting manifest is saved there.

    Classes with fewer than 3 subjects cannot be stratified across three
    splits (each split needs >=1 subject to be represented). Those subjects
    are pooled and assigned by a single unstratified deterministic shuffle
    instead — correctness (no leakage) is preserved, only stratification
    is relaxed, and this is recorded in the report's `unstratified_classes`.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    subject_labels = _unique_subject_labels(subject_ids, labels)
    all_subjects = sorted(subject_labels.keys())
    rng = np.random.RandomState(seed)

    buckets = _class_buckets(subject_labels)
    train, val, test = [], [], []
    unstratified_classes = []
    pooled: List[str] = []

    stratifiable = labels is not None
    for lab, subs in buckets.items():
        if stratifiable and len(subs) >= MIN_SUBJECTS_FOR_STRATIFICATION:
            subs = sorted(subs)
            rng.shuffle(subs)
            n = len(subs)
            n_train = max(1, round(n * train_frac))
            n_val   = max(1, round(n * val_frac))
            n_train = min(n_train, n - 2)  # leave >=1 for val and test
            n_val   = min(n_val, n - n_train - 1)
            train += subs[:n_train]
            val   += subs[n_train:n_train + n_val]
            test  += subs[n_train + n_val:]
        else:
            if stratifiable:
                unstratified_classes.append(lab)
            pooled += subs

    if pooled:
        pooled = sorted(pooled)
        rng.shuffle(pooled)
        n = len(pooled)
        n_train = round(n * train_frac)
        n_val   = round(n * val_frac)
        train += pooled[:n_train]
        val   += pooled[n_train:n_train + n_val]
        test  += pooled[n_train + n_val:]

    report = build_split_report(
        subject_ids, labels, {"train": train, "val": val, "test": test},
        cell_counts=cell_counts,
    )
    report["unstratified_classes"] = [str(c) for c in unstratified_classes]
    report["seed"] = seed

    fingerprint = compute_dataset_fingerprint(
        subject_ids, labels, train_frac, val_frac, test_frac, seed, rare_class_policy,
    )
    manifest = SplitManifest(seed=seed, train_subjects=train, val_subjects=val,
                              test_subjects=test, report=report, fingerprint=fingerprint)
    if manifest_path:
        manifest.save(manifest_path)
    return manifest


def load_or_create_split(
    manifest_path:     Union[str, Path],
    subject_ids:       Sequence,
    labels:            Optional[Sequence] = None,
    train_frac:        float = 0.70,
    val_frac:          float = 0.15,
    test_frac:         float = 0.15,
    seed:              int = 42,
    rare_class_policy: Optional[str] = None,
    force_regenerate:  bool = False,
    **kwargs,
) -> SplitManifest:
    """
    Load an existing manifest if present and still valid for the current
    data/config, else create and save a new one.

    Validity is checked via compute_dataset_fingerprint: the manifest's
    fingerprint must match a fingerprint recomputed from the CURRENT
    subject_ids/labels/split fractions/seed/rare_class_policy. A mismatch
    (a subject added/removed, an effective label changed, a different seed
    or split fraction, a different rare-class policy) means the on-disk
    split no longer describes this dataset — silently reusing it risks an
    unassigned or stale subject, silently regenerating it would quietly
    throw away a previously-audited split. Both are wrong by default, so
    this raises StaleManifestError; pass force_regenerate=True to
    deliberately discard the old manifest and write a fresh one.

    A manifest saved before fingerprinting existed (fingerprint=None) is
    treated as unverifiable and also raises, since there's no way to know
    whether it still matches — regenerate it once with force_regenerate=True
    to adopt fingerprinting going forward.
    """
    path = Path(manifest_path)
    expected_fp = compute_dataset_fingerprint(
        subject_ids, labels, train_frac, val_frac, test_frac, seed, rare_class_policy,
    )
    if path.exists() and not force_regenerate:
        existing = SplitManifest.load(path)
        if existing.fingerprint != expected_fp:
            raise StaleManifestError(
                f"Split manifest at {path} no longer matches the current dataset/config "
                "(subjects, effective labels, split fractions, seed, or rare-class policy "
                "changed since it was created). Refusing to silently reuse or regenerate it. "
                "Pass force_regenerate=True to deliberately create a fresh split, or "
                "investigate why the underlying data/config changed."
            )
        print(f"[splitting] loading existing manifest ← {path}  (fingerprint verified)")
        return existing
    if path.exists() and force_regenerate:
        print(f"[splitting] force_regenerate=True — discarding existing manifest at {path}")
    return subject_train_val_test_split(
        subject_ids, labels=labels, train_frac=train_frac, val_frac=val_frac,
        test_frac=test_frac, seed=seed, rare_class_policy=rare_class_policy,
        manifest_path=path, **kwargs
    )


# ─── Grouped K-fold CV ──────────────────────────────────────────────────────

def grouped_kfold(
    subject_ids: Sequence,
    labels:      Optional[Sequence] = None,
    n_folds:     int = 5,
    seed:        int = 42,
) -> List[Dict[str, List[str]]]:
    """
    Grouped (subject-level), approximately-stratified K-fold split.

    Returns a list of {"train": [...], "val": [...], "classes_absent_from_val": [...],
    "n_folds_used": int, "stratified": bool} dicts, one per fold. n_folds is
    reduced automatically (down to a floor of 2) if any label class has
    fewer subjects than requested folds, since a class can't be represented
    in every fold otherwise. Each fold also records which classes have zero
    subjects in its validation set, so callers don't silently compute a
    per-class metric on a fold that never saw that class.

    Raises ValueError if fewer than 2 independent subjects are present —
    K-fold CV is undefined with 0 or 1 subject.
    """
    subject_labels = _unique_subject_labels(subject_ids, labels)
    if len(subject_labels) < 2:
        raise ValueError(
            f"grouped_kfold requires >=2 independent subjects, got {len(subject_labels)}."
        )
    buckets = _class_buckets(subject_labels)

    smallest_class = min((len(v) for v in buckets.values()), default=0)
    effective_folds = max(2, min(n_folds, smallest_class)) if smallest_class else n_folds
    effective_folds = min(effective_folds, n_folds, len(subject_labels))
    stratified = smallest_class >= effective_folds
    if effective_folds < n_folds:
        print(f"[splitting] grouped_kfold: reducing n_folds {n_folds} → {effective_folds} "
              f"(smallest class has {smallest_class} subjects)")
    if not stratified:
        print("[splitting] grouped_kfold: WARNING — approximate/non-stratified folds "
              "(not every class fits >=1 subject per fold); see per-fold "
              "'classes_absent_from_val' for which classes are missing where.")

    # A per-class-bucket "i % effective_folds" assignment resets i=0 for every
    # class, so e.g. two subjects in two different classes both land in fold 0
    # (i=0 in each bucket) — leaving another fold with an empty validation set
    # and this one with an empty training set. A single cursor shared across
    # all buckets spreads subjects from different classes into different
    # folds instead of colliding on fold 0 every time.
    rng = np.random.RandomState(seed)
    fold_assignment: Dict[str, int] = {}
    fold_cursor = 0
    for lab, subs in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        subs = sorted(subs)
        rng.shuffle(subs)
        for sid in subs:
            fold_assignment[sid] = fold_cursor % effective_folds
            fold_cursor += 1

    folds = []
    all_subjects = sorted(subject_labels.keys())
    all_classes = {str(lab) for lab in buckets.keys()}
    for f in range(effective_folds):
        val_subs   = [s for s in all_subjects if fold_assignment[s] == f]
        train_subs = [s for s in all_subjects if fold_assignment[s] != f]
        if set(val_subs) & set(train_subs):
            raise RuntimeError(f"grouped_kfold: fold {f} has overlapping train/val subjects (internal bug)")
        if not train_subs:
            raise ValueError(
                f"grouped_kfold: fold {f} would have an empty training set with "
                f"{len(subject_labels)} subjects and n_folds={effective_folds}. "
                "Request fewer folds or provide more subjects."
            )
        if not val_subs:
            raise ValueError(
                f"grouped_kfold: fold {f} would have an empty validation set with "
                f"{len(subject_labels)} subjects and n_folds={effective_folds}. "
                "Request fewer folds or provide more subjects."
            )
        val_classes = {str(subject_labels[s]) for s in val_subs}
        folds.append({
            "train": train_subs,
            "val": val_subs,
            "classes_absent_from_val": sorted(all_classes - val_classes),
            "n_folds_used": effective_folds,
            "stratified": stratified,
        })
    return folds


# ─── Report ─────────────────────────────────────────────────────────────────

def build_split_report(
    subject_ids: Sequence,
    labels:      Optional[Sequence],
    assignment:  Dict[str, List[str]],
    cell_counts: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    subjects/cells per split, class distribution per split, and
    missing/unknown-label counts (label is None/NaN for a subject).
    """
    if cell_counts is None:
        cell_counts = {}
        for sid in subject_ids:
            sid = str(sid)
            cell_counts[sid] = cell_counts.get(sid, 0) + 1

    subject_labels = _unique_subject_labels(subject_ids, labels)

    def _is_unknown(lab) -> bool:
        if lab is None:
            return True
        try:
            return bool(np.isnan(lab))
        except (TypeError, ValueError):
            return False

    report: Dict = {"splits": {}}
    for split_name, subs in assignment.items():
        subs = [str(s) for s in subs]
        class_dist: Dict[str, int] = {}
        n_unknown = 0
        for sid in subs:
            lab = subject_labels.get(sid)
            if _is_unknown(lab):
                n_unknown += 1
                key = "unknown"
            else:
                key = str(lab)
            class_dist[key] = class_dist.get(key, 0) + 1

        report["splits"][split_name] = {
            "n_subjects":        len(subs),
            "n_cells":           int(sum(cell_counts.get(s, 0) for s in subs)),
            "class_distribution": class_dist,
            "n_unknown_label":    n_unknown,
        }
    return report
