"""
data/nlst_smoking.py — explicit parser for NLST CIGSMOK/CIGAR smoking-
history fields.

This project has no independently re-verified copy of NLST's full data
dictionary in this environment (controlled access, no network). The ONLY
codes this parser trusts as verified evidence are the ones already
documented in this repository's own reviewed access instructions (see
src/data/downloaders.py::print_nlst_instructions):

    CIGSMOK  1 = current smoker, 2 = former smoker
    CIGAR    1 = yes (cigar use)

Any other raw value — missing, blank, NaN/null, 0, an undocumented code,
or a malformed/non-numeric string — is treated as UNSUPPORTED/UNKNOWN,
never as a verified negative ("never smoked") and never coerced into a
verified positive. This is deliberately conservative: this repo's
documentation does not state what CIGSMOK=0 (or any code besides 1/2)
means, so this parser does not guess. A subject's absence of a positive
code is not evidence of non-exposure — it is simply the absence of
evidence, and non-exposure must never be inferred from that (see
NON-NEGOTIABLE rule: never treat missing data as a negative label).

parse_nlst_smoking_row() is deterministic and side-effect free — it never
touches an AnnData or a DataFrame, only a single row's raw field values —
so it can be unit-tested directly against small synthetic fixtures.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

# Documented codes only — see module docstring.
CIGSMOK_CIGARETTE_CODES = {1, 2}
CIGAR_YES_CODE = 1

EVIDENCE_POSITIVE = "positive"
EVIDENCE_UNKNOWN = "unknown"

SOURCE_NLST_CIGSMOK_CIGAR = "NLST screen.csv CIGSMOK/CIGAR"

LIMITATION_KNOWN = (
    "NLST CIGSMOK/CIGAR are self-reported clinical screening fields with a "
    "documented code (CIGSMOK=1 current smoker/2 former smoker, CIGAR=1 cigar "
    "use) — not a biological measurement, and not evidence about any other "
    "substance."
)
LIMITATION_UNKNOWN = (
    "NLST CIGSMOK/CIGAR value was missing, blank, null, an undocumented code, "
    "or unparseable for this subject — this repository trusts only the "
    "codes documented in src/data/downloaders.py::print_nlst_instructions "
    "(CIGSMOK in {1,2}, CIGAR==1) as verified evidence; no other code is "
    "assumed to mean 'never smoked' or anything else, so this remains an "
    "unknown smoking history rather than a verified negative."
)


@dataclass
class NLSTSmokingRecord:
    cigarette_evidence:   str             # EVIDENCE_POSITIVE | EVIDENCE_UNKNOWN
    cigar_evidence:        str             # EVIDENCE_POSITIVE | EVIDENCE_UNKNOWN
    effective_smoke_type:  Optional[str]   # "cigarette" | "cigar" | "dual_use" | None
    smoke_type_known:      bool
    source:                str
    method:                str
    limitation:             str


def _parse_code(raw, valid_codes) -> str:
    """
    Returns EVIDENCE_POSITIVE only when `raw` parses to an int in
    `valid_codes`; EVIDENCE_UNKNOWN for anything else — missing (None),
    NaN/NaT, blank/whitespace-only strings, non-numeric strings, floats
    with a fractional part that isn't a clean integer code, or any
    numeric value not in `valid_codes`. Never raises: a malformed value is
    a controlled "stays unknown" outcome, not a crash, so one bad row
    cannot halt an entire NLST CSV's ingestion.
    """
    if raw is None:
        return EVIDENCE_UNKNOWN
    if isinstance(raw, str) and raw.strip() == "":
        return EVIDENCE_UNKNOWN
    try:
        is_missing = bool(pd.isna(raw))
    except (TypeError, ValueError):
        is_missing = False  # pd.isna can't evaluate some exotic types — fall through to the numeric parse
    if is_missing:
        return EVIDENCE_UNKNOWN
    try:
        # pandas commonly reads an integer-coded column as float64 once any
        # row in it is NaN (e.g. 1.0 instead of 1) — parse as float first,
        # then require it to be an exact whole number before treating it as
        # one of the documented integer codes. "1.5" or any other
        # non-whole value is left unknown, not rounded.
        fval = float(str(raw).strip())
    except (TypeError, ValueError):
        return EVIDENCE_UNKNOWN
    if not fval.is_integer():
        return EVIDENCE_UNKNOWN
    return EVIDENCE_POSITIVE if int(fval) in valid_codes else EVIDENCE_UNKNOWN


def parse_nlst_smoking_row(cigsmok_raw, cigar_raw) -> NLSTSmokingRecord:
    """
    Deterministic policy (documented here, not inferred per-row):
      - CIGSMOK alone documents cigarette history on its own — it is a
        complete statement ("this person is a current/former cigarette
        smoker") independent of whether CIGAR is known.
      - CIGAR alone documents cigar history the same way.
      - Both positive together -> dual_use (not a conflict: cigarette and
        cigar are independent, non-exclusive habits).
      - Neither parses to a documented positive code -> unknown; no
        effective_smoke_type is assigned, ever, from absence of evidence.
    There is no code path in this function that returns a verified
    "unexposed"/negative smoke type — NLST's documented codes only ever
    supply POSITIVE evidence for cigarette/cigar use in this repository's
    current, reviewed understanding of the codebook (see module docstring).
    """
    cig_evidence = _parse_code(cigsmok_raw, CIGSMOK_CIGARETTE_CODES)
    cigar_evidence = _parse_code(cigar_raw, {CIGAR_YES_CODE})

    if cig_evidence == EVIDENCE_POSITIVE and cigar_evidence == EVIDENCE_POSITIVE:
        effective = "dual_use"
    elif cig_evidence == EVIDENCE_POSITIVE:
        effective = "cigarette"
    elif cigar_evidence == EVIDENCE_POSITIVE:
        effective = "cigar"
    else:
        effective = None

    known = effective is not None
    return NLSTSmokingRecord(
        cigarette_evidence=cig_evidence,
        cigar_evidence=cigar_evidence,
        effective_smoke_type=effective,
        smoke_type_known=known,
        source=SOURCE_NLST_CIGSMOK_CIGAR,
        method="documented_code_lookup",
        limitation=LIMITATION_KNOWN if known else LIMITATION_UNKNOWN,
    )
