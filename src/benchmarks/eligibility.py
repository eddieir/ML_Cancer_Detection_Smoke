"""
benchmarks/eligibility.py — explicit, upfront eligibility reports for the two
benchmark tasks, computed before any training starts.

Task A (smoke-type classification) and Task B (subject-level cancer
prediction) must never be silently attempted on data that can't support them
— see check_mil_eligibility (train.py) for the same principle applied inside
Trainer. These reports are the pre-training equivalent, run against an
ExperimentContext instead of a single SubjectLevelDataset.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .context import ExperimentContext

MIN_SUBJECTS_TASK_A = 6     # >=2 per split at minimum, else metrics are near-meaningless
MIN_SUBJECTS_TASK_B = 10
MIN_POSITIVE_TASK_B = 2
MIN_NEGATIVE_TASK_B = 2


@dataclass
class EligibilityReport:
    task: str
    eligible: bool
    status: str            # "ELIGIBLE" | "NOT_EVALUABLE" | "NOT_COMPARABLE"
    reasons: List[str] = field(default_factory=list)
    counts: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task, "eligible": self.eligible, "status": self.status,
            "reasons": self.reasons, "counts": self.counts,
        }


def _subject_count(cell_dataset) -> int:
    if cell_dataset is None or len(cell_dataset) == 0:
        return 0
    return len(set(cell_dataset.subject_ids.tolist()))


def check_task_a_eligibility(context: ExperimentContext) -> EligibilityReport:
    """Smoke-type classification eligibility: every split must have >=1
    subject and the label space must have >=2 effective classes overall."""
    counts = {
        "train_subjects": _subject_count(context.train_cell_dataset),
        "val_subjects":   _subject_count(context.val_cell_dataset),
        "test_subjects":  _subject_count(context.test_cell_dataset),
        "k_effective_classes": context.num_smoke_classes,
    }
    reasons = []
    if counts["train_subjects"] == 0:
        reasons.append("train split has zero subjects")
    if counts["val_subjects"] == 0:
        reasons.append("validation split has zero subjects")
    if counts["test_subjects"] == 0:
        reasons.append("test split has zero subjects")
    total_subjects = counts["train_subjects"] + counts["val_subjects"] + counts["test_subjects"]
    if total_subjects < MIN_SUBJECTS_TASK_A:
        reasons.append(
            f"only {total_subjects} total subjects (< {MIN_SUBJECTS_TASK_A}) — "
            "per-class metrics would not be statistically meaningful"
        )
    if context.num_smoke_classes < 2:
        reasons.append(f"only {context.num_smoke_classes} effective smoke class(es) — nothing to classify")

    eligible = not reasons
    return EligibilityReport(
        task="smoke_classification", eligible=eligible,
        status="ELIGIBLE" if eligible else "NOT_EVALUABLE",
        reasons=reasons, counts=counts,
    )


def _known_outcome_counts(bags: List[dict]) -> Dict[str, int]:
    known = [b for b in bags if b.get("cancer_label_known")]
    n_pos = sum(1 for b in known if b.get("cancer_label") == 1)
    n_neg = sum(1 for b in known if b.get("cancer_label") == 0)
    return {"n_known": len(known), "n_positive": n_pos, "n_negative": n_neg}


def check_task_b_eligibility(context: ExperimentContext) -> EligibilityReport:
    """
    Subject-level cancer prediction eligibility. Requires known-outcome
    subjects with BOTH classes present, split across train/val/test — never
    substitutes a fabricated negative for an unknown outcome (see
    SubjectLevelDataset's require_known_outcome default).
    """
    train_c = _known_outcome_counts(context.train_bags)
    val_c   = _known_outcome_counts(context.val_bags)
    test_c  = _known_outcome_counts(context.test_bags)
    counts = {"train": train_c, "val": val_c, "test": test_c}

    reasons = []
    total_known = train_c["n_known"] + val_c["n_known"] + test_c["n_known"]
    if total_known < MIN_SUBJECTS_TASK_B:
        reasons.append(f"only {total_known} subjects with known cancer outcome (< {MIN_SUBJECTS_TASK_B})")
    for name, c in (("train", train_c), ("val", val_c), ("test", test_c)):
        if c["n_known"] == 0:
            reasons.append(f"{name} split has zero known-outcome subjects")
    total_pos = train_c["n_positive"] + val_c["n_positive"] + test_c["n_positive"]
    total_neg = train_c["n_negative"] + val_c["n_negative"] + test_c["n_negative"]
    if total_pos < MIN_POSITIVE_TASK_B:
        reasons.append(f"only {total_pos} known-positive subjects (< {MIN_POSITIVE_TASK_B})")
    if total_neg < MIN_NEGATIVE_TASK_B:
        reasons.append(f"only {total_neg} known-negative subjects (< {MIN_NEGATIVE_TASK_B})")
    if test_c["n_positive"] == 0 or test_c["n_negative"] == 0:
        reasons.append(
            f"test split has only one class present (pos={test_c['n_positive']}, "
            f"neg={test_c['n_negative']}) — AUROC on test is undefined"
        )

    eligible = not reasons
    return EligibilityReport(
        task="cancer_prediction", eligible=eligible,
        status="ELIGIBLE" if eligible else "NOT_EVALUABLE",
        reasons=reasons, counts=counts,
    )
