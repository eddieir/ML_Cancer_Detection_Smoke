"""
data/label_mapping.py — deterministic, contiguous effective smoke-label space.

apply_rare_class_policy() (rare_class.py) can merge a rare raw class into
another raw class, or exclude a raw class from training/evaluation entirely.
Either way, at least one of the six raw smoke_type ids (constants.SMOKE_TYPES)
becomes dead: a merged-away or excluded raw id can never appear as a target
again once the policy runs, but until now the model still had a fixed
6-output head, `evaluate.py`/`train.py` still scored macro-F1 over 6 classes,
and inference still displayed all 6 names — one output neuron with no
possible target, and every reported metric silently averaging in a class
that could not be predicted correctly.

EffectiveLabelMapping is the single source of truth mapping the FIXED raw
label space (0..5, constants.SMOKE_TYPES) to a CONTIGUOUS effective space
(0..K-1) that excludes dead classes. It is built once, directly from the
rare-class policy's report — never inferred from which classes happen to
appear in one particular data split, since a validation split missing a
class is not the same thing as a policy excluding that class — and is meant
to be persisted (via to_dict/from_dict) in the preprocessing artifact, the
training checkpoint, and every evaluation/inference report so a model's
actual output space is always auditable against how it was produced.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from constants import SMOKE_TYPES


@dataclass
class EffectiveLabelMapping:
    policy:               str
    raw_id_to_name:       Dict[int, str]   # fixed source taxonomy (constants.SMOKE_TYPES)
    raw_to_effective:     Dict[int, int]   # only active (kept) raw ids -> contiguous effective ids
    effective_id_to_name: Dict[int, str]   # 0..K-1 -> name, contiguous by construction
    excluded_raw_ids:     List[int] = field(default_factory=list)
    merged_away_raw_ids:  List[int] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.effective_id_to_name)

    @property
    def class_names(self) -> List[str]:
        """Ordered 0..K-1 — safe to index directly with an effective id or a
        model argmax over the smoke head's output."""
        return [self.effective_id_to_name[i] for i in range(self.k)]

    def transform(self, raw_ids) -> np.ndarray:
        """
        Map raw (post rare-class-merge-reassignment) ids to contiguous
        effective ids. Raises on any id with no effective mapping — i.e. a
        raw id excluded by policy, which must be filtered out via
        apply_rare_class_policy's keep_mask BEFORE calling this. transform()
        never silently drops or reassigns an unmapped id.
        """
        raw_ids = np.asarray(raw_ids)
        bad = sorted(set(int(r) for r in raw_ids.tolist()) - set(self.raw_to_effective))
        if bad:
            raise ValueError(
                f"EffectiveLabelMapping.transform: raw id(s) {bad} have no effective "
                "mapping (excluded by the rare-class policy) — filter these rows out "
                "(see apply_rare_class_policy's keep_mask) before calling transform()."
            )
        return np.array([self.raw_to_effective[int(r)] for r in raw_ids], dtype=np.int64)

    def inverse_transform(self, effective_ids) -> np.ndarray:
        """Effective id -> raw id, kept for auditability (see module docstring)."""
        effective_ids = np.asarray(effective_ids)
        eff_to_raw = {v: k for k, v in self.raw_to_effective.items()}
        bad = sorted(set(int(e) for e in effective_ids.tolist()) - set(eff_to_raw))
        if bad:
            raise ValueError(f"inverse_transform: effective id(s) {bad} out of range [0, {self.k}).")
        return np.array([eff_to_raw[int(e)] for e in effective_ids], dtype=np.int64)

    def to_dict(self) -> dict:
        return {
            "policy":               self.policy,
            "raw_id_to_name":       {str(k): v for k, v in self.raw_id_to_name.items()},
            "raw_to_effective":     {str(k): v for k, v in self.raw_to_effective.items()},
            "effective_id_to_name": {str(k): v for k, v in self.effective_id_to_name.items()},
            "excluded_raw_ids":     list(self.excluded_raw_ids),
            "merged_away_raw_ids":  list(self.merged_away_raw_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EffectiveLabelMapping":
        return cls(
            policy               = d["policy"],
            raw_id_to_name       = {int(k): v for k, v in d["raw_id_to_name"].items()},
            raw_to_effective     = {int(k): int(v) for k, v in d["raw_to_effective"].items()},
            effective_id_to_name = {int(k): v for k, v in d["effective_id_to_name"].items()},
            excluded_raw_ids     = list(d.get("excluded_raw_ids", [])),
            merged_away_raw_ids  = list(d.get("merged_away_raw_ids", [])),
        )

    def validate_compatible(self, other: "EffectiveLabelMapping",
                             self_name: str = "this", other_name: str = "other") -> None:
        """
        Raise ValueError naming the exact mismatch if two mappings disagree
        on the effective label space they describe. Called wherever a
        checkpoint, a preprocessing artifact, and/or a config are loaded
        together — mixing artifacts/checkpoints from different rare-class
        policy runs would otherwise silently corrupt which name a
        prediction's effective class id actually refers to.
        """
        if (self.raw_to_effective != other.raw_to_effective
                or self.effective_id_to_name != other.effective_id_to_name):
            raise ValueError(
                f"Effective label mapping mismatch between {self_name} and {other_name}: "
                f"{self_name}={self.to_dict()}  vs  {other_name}={other.to_dict()}. "
                "These must come from the same rare-class-policy run against the same "
                "raw label space — loading mismatched checkpoint/artifact/config "
                "combinations would silently mislabel predictions."
            )


def identity_label_mapping(raw_id_to_name: Optional[Dict[int, str]] = None) -> EffectiveLabelMapping:
    """
    No merge/exclusion — effective space == raw space, unchanged. This is
    the mapping used whenever no rare-class-policy report is available: the
    six-class default, and legacy/diagnostic code paths.
    """
    raw_id_to_name = dict(raw_id_to_name or SMOKE_TYPES)
    ids = sorted(raw_id_to_name)
    return EffectiveLabelMapping(
        policy               = "keep_with_warning",
        raw_id_to_name       = raw_id_to_name,
        raw_to_effective     = {i: i for i in ids},
        effective_id_to_name = {i: raw_id_to_name[i] for i in ids},
    )


def build_effective_label_mapping(
    rare_class_report: dict,
    raw_id_to_name: Optional[Dict[int, str]] = None,
) -> EffectiveLabelMapping:
    """
    Build the effective mapping directly and deterministically from
    apply_rare_class_policy()'s report (data/rare_class.py) — never from
    scanning which classes happen to appear in one particular evaluation
    split. Supports both the no-merge six-class case (report has no
    "excluded"/"merged_into_*" actions -> identity mapping, K=6) and any
    merged/excluded-class case (K < 6, contiguous).
    """
    raw_id_to_name = dict(raw_id_to_name or SMOKE_TYPES)
    name_to_raw_id = {name: rid for rid, name in raw_id_to_name.items()}

    excluded, merged_away = set(), set()
    for cls_name, entry in rare_class_report.get("affected_classes", {}).items():
        if cls_name not in name_to_raw_id:
            continue
        raw_id = name_to_raw_id[cls_name]
        action = entry.get("action", "")
        if action == "excluded":
            excluded.add(raw_id)
        elif action.startswith("merged_into_"):
            merged_away.add(raw_id)

    active_ids = sorted(set(raw_id_to_name) - excluded - merged_away)
    raw_to_effective = {raw_id: eff for eff, raw_id in enumerate(active_ids)}
    effective_id_to_name = {eff: raw_id_to_name[raw_id] for raw_id, eff in raw_to_effective.items()}

    return EffectiveLabelMapping(
        policy               = rare_class_report.get("policy", "keep_with_warning"),
        raw_id_to_name       = raw_id_to_name,
        raw_to_effective      = raw_to_effective,
        effective_id_to_name = effective_id_to_name,
        excluded_raw_ids      = sorted(excluded),
        merged_away_raw_ids  = sorted(merged_away),
    )
