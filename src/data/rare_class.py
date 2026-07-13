"""
data/rare_class.py — configurable policy for smoke-type classes with too
few independent subjects to learn or evaluate reliably.

The real merged data currently has ~1 subject carrying the "cigar" label
(see README.md). A six-class classifier cannot legitimately claim to learn
or measure a class represented by a single independent subject — any
per-class metric for it is one data point, not a rate. This module makes
that limitation an explicit, auditable, testable policy decision instead
of a silent modeling artifact (e.g. the class simply never gets predicted
and nobody notices why).
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from constants import SMOKE_TYPE_MAP, SMOKE_TYPES

VALID_POLICIES = {
    "keep_with_warning",
    "merge_into_dual_use_or_other",
    "exclude_from_training_and_evaluation",
}

# Where merge_into_dual_use_or_other sends cells of a too-rare class. There is
# no generic "other" bucket in the fixed 6-class label space (constants.py
# SMOKE_TYPES) without changing every shape that depends on N_SMOKE_CLASSES,
# so this policy merges into the existing "dual_use" class — the closest
# available "mixed/other tobacco exposure" bucket — and records the merge
# in the report rather than pretending it's a clean semantic fit.
MERGE_TARGET = "dual_use"


def apply_rare_class_policy(
    smoke_type_ids: np.ndarray,
    subject_ids:    Sequence,
    policy:         str = "merge_into_dual_use_or_other",
    target_classes: Sequence[str] = ("cigar",),
    min_subjects_required: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Parameters
    ----------
    smoke_type_ids : [N] int class ids (constants.SMOKE_TYPES), one per cell/sample.
    subject_ids    : [N] aligned subject identifiers.
    policy         : one of VALID_POLICIES.
    target_classes : class names (constants.SMOKE_TYPE_MAP keys) this policy applies to.
    min_subjects_required : a class with fewer independent subjects than this
                             is considered "too rare" and the policy applies.

    Returns
    -------
    new_smoke_type_ids : [N] int, smoke_type_ids unless merge_into_dual_use_or_other
                          reassigned some entries. The input array is never
                          mutated in place — original raw labels stay intact
                          in whatever the caller passed in.
    keep_mask           : [N] bool, False only for cells dropped by
                           exclude_from_training_and_evaluation.
    report               : per-class subject counts and the action taken —
                           for confusion-matrix labeling and experiment metadata.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(f"Unknown rare_class policy {policy!r} — must be one of {VALID_POLICIES}")

    target_ids = {SMOKE_TYPE_MAP[t] for t in target_classes if t in SMOKE_TYPE_MAP}
    subj = np.asarray([str(s) for s in subject_ids])
    ids  = np.asarray(smoke_type_ids)

    new_ids   = ids.copy()
    keep_mask = np.ones(len(ids), dtype=bool)
    report: Dict = {"policy": policy, "min_subjects_required": min_subjects_required,
                     "affected_classes": {}}

    for cls_id in sorted(target_ids):
        cls_mask   = ids == cls_id
        n_subjects = len(set(subj[cls_mask]))
        cls_name   = SMOKE_TYPES.get(cls_id, str(cls_id))
        too_rare   = 0 < n_subjects < min_subjects_required
        entry = {"n_subjects": n_subjects, "too_rare": too_rare, "action": "kept"}

        if too_rare:
            if policy == "merge_into_dual_use_or_other":
                merge_id = SMOKE_TYPE_MAP[MERGE_TARGET]
                new_ids[cls_mask] = merge_id
                entry["action"] = f"merged_into_{MERGE_TARGET}"
            elif policy == "exclude_from_training_and_evaluation":
                keep_mask[cls_mask] = False
                entry["action"] = "excluded"
            else:  # keep_with_warning
                entry["action"] = "kept_with_warning"
                print(f"[rare_class] WARNING: '{cls_name}' has only {n_subjects} independent "
                      f"subject(s) (< {min_subjects_required}) — kept per policy="
                      "'keep_with_warning'. Per-class metrics for this class are not "
                      "statistically meaningful and should be reported as such.")
        report["affected_classes"][cls_name] = entry

    return new_ids, keep_mask, report
