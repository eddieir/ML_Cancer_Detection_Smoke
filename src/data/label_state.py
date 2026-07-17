"""
data/label_state.py — explicit label-state vocabulary shared by every label
type (smoke, malignancy, cancer outcome).

A bare numeric label cannot distinguish "verified negative" from "we never
measured this" — and this codebase has already been bitten by that once
(see data/labellers.py::add_malignancy_labels' malignancy/malignancy_known
pair, data/assembly.py::assemble_subject_bags' cancer_label/
cancer_label_known pair). This module gives that existing known/unknown
split a named, typed vocabulary so every label field in the pipeline uses
the same five states instead of a scattered mix of None, NaN, 0, and
ad hoc boolean flags:

  KNOWN_POSITIVE      — a verified positive value exists.
  KNOWN_NEGATIVE      — a verified negative value exists (this is NOT the
                         same thing as "we don't know" — see below).
  UNKNOWN             — no verified value exists yet; must never be
                         silently treated as KNOWN_NEGATIVE.
  NOT_APPLICABLE      — the label doesn't apply to this row at all (e.g. a
                         malignancy label for a pseudo-bulk sample with no
                         per-cell identity).
  EXCLUDED_BY_POLICY  — a value may exist, but an explicit, documented
                         policy (e.g. rare-class merge/exclusion, a
                         controlled-access gate) removes it from
                         supervision/evaluation regardless.

Only KNOWN_POSITIVE / KNOWN_NEGATIVE rows may contribute to a supervised
loss or a supervised metric for that label. Every other state must be
excluded — never defaulted to a negative.

This module intentionally does NOT change the six fixed raw smoke_type ids
(constants.SMOKE_TYPES) or the malignancy_known/cancer_label_known columns
already wired through data/assembly.py, model.py's MultiTaskLoss, and
train.py — those existing known/unknown gates remain the enforcement
points. What this module adds is the vocabulary and a small typed
provenance record so *new* label-producing code (dataset manifest,
GSE136831 weak-proxy handling, TCGA smoking metadata, NLST fields) has one
place to describe status/source/method/confidence/limitation instead of
inventing another ad hoc pair of columns per label.
"""

from dataclasses import dataclass
from typing import Optional

KNOWN_POSITIVE = "known_positive"
KNOWN_NEGATIVE = "known_negative"
UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"
EXCLUDED_BY_POLICY = "excluded_by_policy"

VALID_LABEL_STATES = frozenset({
    KNOWN_POSITIVE, KNOWN_NEGATIVE, UNKNOWN, NOT_APPLICABLE, EXCLUDED_BY_POLICY,
})

# States that may contribute to a supervised loss / supervised metric for
# the label they describe. Every other state must be excluded from both.
SUPERVISABLE_STATES = frozenset({KNOWN_POSITIVE, KNOWN_NEGATIVE})


def is_supervisable(state: str) -> bool:
    """True only for KNOWN_POSITIVE / KNOWN_NEGATIVE. Raises on an
    unrecognized state rather than silently treating it as excluded, so a
    typo in a caller's state string fails loudly instead of quietly
    dropping real labels."""
    if state not in VALID_LABEL_STATES:
        raise ValueError(f"Unknown label state {state!r} — must be one of {sorted(VALID_LABEL_STATES)}")
    return state in SUPERVISABLE_STATES


@dataclass
class LabelProvenance:
    """
    One label's full provenance record: status plus source/method/
    confidence/limitation. Meant to be attached (as a set of obs/report
    columns, or serialized inline) wherever a label value is produced —
    see data/manifest.py's per-dataset smoke/cancer/malignancy metadata
    fields and data/converters.py's GSE136831 weak-proxy handling.

    confidence is a coarse, human-set qualitative bucket (not a
    statistical estimate) — "high" for a directly measured/verified field,
    "low" for a documented weak proxy (e.g. COPD diagnosis standing in for
    cigarette exposure), "unknown_confidence" only when status is UNKNOWN.
    """
    status:      str
    source:      str
    method:      str
    confidence:  str = "unknown_confidence"
    limitation:  Optional[str] = None

    def __post_init__(self):
        if self.status not in VALID_LABEL_STATES:
            raise ValueError(
                f"LabelProvenance.status={self.status!r} must be one of {sorted(VALID_LABEL_STATES)}"
            )
        if self.status == UNKNOWN and self.confidence not in ("unknown_confidence", "low"):
            raise ValueError(
                "LabelProvenance: status=UNKNOWN must not carry a confidence implying a real "
                f"measurement exists (got confidence={self.confidence!r})."
            )

    @property
    def known(self) -> bool:
        return is_supervisable(self.status)

    def to_dict(self) -> dict:
        return {
            "status": self.status, "source": self.source, "method": self.method,
            "confidence": self.confidence, "limitation": self.limitation,
        }


def unknown_provenance(source: str, method: str = "not_measured",
                        limitation: Optional[str] = None) -> LabelProvenance:
    """Shorthand for the common case: no value was ever measured for this
    row. Never use this to represent "measured and turned out negative" —
    use LabelProvenance(status=KNOWN_NEGATIVE, ...) for that."""
    return LabelProvenance(status=UNKNOWN, source=source, method=method,
                            confidence="unknown_confidence", limitation=limitation)


def weak_proxy_provenance(source: str, proxy_type: str, limitation: str) -> LabelProvenance:
    """
    Shorthand for a documented weak proxy (e.g. GSE136831 COPD diagnosis
    standing in for verified cigarette exposure). status is deliberately
    UNKNOWN, not KNOWN_POSITIVE/KNOWN_NEGATIVE — a weak proxy must never be
    promoted into a primary supervised label implicitly; see
    data/converters.py's GSE136831 handling and README.md's COPD-proxy
    policy section for how an explicit opt-in promotes it instead.
    """
    return LabelProvenance(
        status=UNKNOWN, source=source, method=f"weak_proxy:{proxy_type}",
        confidence="low", limitation=limitation,
    )
