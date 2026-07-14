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


def check_task_b_development_eligibility(train_bags: List[dict], val_bags: List[dict]) -> EligibilityReport:
    """
    Subject-level cancer prediction eligibility, computed from DEVELOPMENT
    data only (train+val bags) — never reads context.test_bags, test
    labels, or test class counts. This is the ONLY eligibility gate allowed
    to run before the frozen-test guard is acquired: test composition must
    never be able to influence whether an experiment proceeds (blocker 1).
    Whether test metrics end up mathematically defined is decided later, by
    check_test_evaluability, strictly inside the guarded stage.
    """
    train_c = _known_outcome_counts(train_bags)
    val_c   = _known_outcome_counts(val_bags)
    counts = {"train": train_c, "val": val_c}

    reasons = []
    total_known = train_c["n_known"] + val_c["n_known"]
    if total_known < MIN_SUBJECTS_TASK_B:
        reasons.append(f"only {total_known} development subjects with known cancer outcome (< {MIN_SUBJECTS_TASK_B})")
    for name, c in (("train", train_c), ("val", val_c)):
        if c["n_known"] == 0:
            reasons.append(f"{name} split has zero known-outcome subjects")
    total_pos = train_c["n_positive"] + val_c["n_positive"]
    total_neg = train_c["n_negative"] + val_c["n_negative"]
    if total_pos < MIN_POSITIVE_TASK_B:
        reasons.append(f"only {total_pos} known-positive development subjects (< {MIN_POSITIVE_TASK_B})")
    if total_neg < MIN_NEGATIVE_TASK_B:
        reasons.append(f"only {total_neg} known-negative development subjects (< {MIN_NEGATIVE_TASK_B})")

    eligible = not reasons
    return EligibilityReport(
        task="cancer_prediction_development", eligible=eligible,
        status="ELIGIBLE" if eligible else "NOT_EVALUABLE",
        reasons=reasons, counts=counts,
    )


def check_test_evaluability(test_bags: List[dict]) -> Dict:
    """
    Determines which frozen-test metrics are mathematically defined. Called
    ONLY from inside the guarded stage, strictly after FrozenTestGuard.
    acquire() has already succeeded (this function itself never acquires
    or checks a guard — the caller is responsible for ordering). Never
    raises to reject the run and never substitutes 0.5 for an undefined
    metric — a one-class test split simply means AUROC/AUPRC come back None
    with a recorded reason while every threshold-based metric still runs
    normally (see benchmarks.metrics.cancer_prediction_metrics).
    """
    test_c = _known_outcome_counts(test_bags)
    reasons = []
    if test_c["n_known"] == 0:
        reasons.append("test split has zero known-outcome subjects — no metrics can be computed")
    auroc_auprc_defined = test_c["n_positive"] > 0 and test_c["n_negative"] > 0
    if test_c["n_known"] > 0 and not auroc_auprc_defined:
        reasons.append(
            f"test split has only one class present (pos={test_c['n_positive']}, "
            f"neg={test_c['n_negative']}) — AUROC/AUPRC are undefined for this evaluation, "
            "not replaced with 0.5; other threshold-based metrics remain defined"
        )
    return {"counts": test_c, "auroc_auprc_defined": auroc_auprc_defined,
            "n_known": test_c["n_known"], "reasons": reasons}
