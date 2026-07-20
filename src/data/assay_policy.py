"""
data/assay_policy.py — the single experiment-level gate governing whether a
row of loaded expression data is allowed to enter a given run.

Background (see README.md's "Bulk/single-cell separation" section and
GitHub issue #13): a bulk microarray/RNA-seq sample represented as one
AnnData row ("pseudo-bulk", data/loaders.py::load_microarray,
obs["is_pseudo_bulk"]=True) is not a cell. It has no real cell type, no
per-cell malignancy signal, and no biological meaning as one element of a
subject's MIL cell bag. Before this module existed, the default pipeline
(preprocess.py::_load_all_sources) only rejected rows explicitly stamped
obs["assay_mode"]=="bulk_tcga" — every other pseudo-bulk source (GSE994,
GSE123352, GSE307690/CANUCK) has obs["assay_mode"] still defaulted to
"human_single_cell" (data/loaders.py::_attach_standard_obs never overrides
it for non-TCGA sources), so that check silently let them through.

This module fixes that by keying enforcement on `is_pseudo_bulk` — the
row-level fact of whether a sample IS a real cell — never on a source name,
an accession string, or the (frequently unset) assay_mode label. Renaming a
config entry or a converted file cannot bypass it.

Three experiment-level policies (constants.ASSAY_POLICY_*):

  single_cell_only (default) — accepts real single-cell rows only; any
    is_pseudo_bulk=True row anywhere in the input is rejected.
  bulk_only — accepts pseudo-bulk/bulk rows only; any real single-cell row
    is rejected. Loading and validating a bulk manifest is supported (see
    preprocess.py::load_tcga_bulk_dataset); there is no bulk training loop,
    so requesting trainable bulk data raises BulkTrainingNotImplementedError
    rather than silently reusing the single-cell model on bulk expression.
  multimodal — not implemented. Requesting it anywhere (loading, fitting,
    training, resolving capabilities) raises MultimodalTrainingNotImplemented
    Error immediately; there is no separate-encoder, modality-aware fusion
    architecture in this codebase, and a "just allow mixed rows" mode would
    be exactly the defect this module exists to prevent.

Every boundary that can receive a heterogeneous mix of rows — source
loading, source merging, preprocessing fit/transform, CellLevelDataset/bag
construction, CV/OOF fold construction, final development fit, frozen-test
transform, source-held-out evaluation, inference/bundle loading — must call
assert_rows_match_policy() (or a wrapper documented alongside it) before
using the rows. A caller passing a hand-built or corrupted AnnData/array is
checked exactly the same way as one produced by the real loaders: this
module only ever looks at the row-level `is_pseudo_bulk` values it is
given, never at how they got there.
"""

from typing import NamedTuple, Optional, Sequence

import numpy as np

from constants import (
    ASSAY_MODE_BULK_TCGA,
    ASSAY_MODE_SINGLE_CELL,
    ASSAY_POLICY_BULK_ONLY,
    ASSAY_POLICY_MULTIMODAL,
    ASSAY_POLICY_SINGLE_CELL_ONLY,
    ASSAY_POLICY_VERSION,
    DEFAULT_ASSAY_POLICY,
    VALID_ASSAY_POLICIES,
)
from data.assay_mode import AssayModeError


class AssayPolicyError(AssayModeError):
    """Raised when rows would enter/leave a boundary in a way the current
    assay_policy does not explicitly allow, or when an assay_policy/row-
    provenance value itself is invalid or missing where it is required.
    Subclasses data.assay_mode.AssayModeError (itself a ValueError) so
    existing `pytest.raises(AssayModeError)` / `pytest.raises(ValueError)`
    call sites written against the pre-issue-13 TCGA-only guard keep
    working unchanged against this project-wide policy gate."""


class MissingAssayProvenanceError(AssayPolicyError):
    """Raised when a real (non-diagnostic) code path needs row-level
    is_pseudo_bulk provenance and none was supplied — a missing column, a
    None array, or an absent required field on a loaded artifact/dataset.
    A missing value is NEVER treated as "real cell" / safe by default;
    only an explicitly declared synthetic diagnostic fixture may omit
    provenance, and only through a narrow, explicit diagnostic_mode path
    that never reaches real training/evaluation/inference."""


class InvalidAssayProvenanceError(AssayPolicyError):
    """Raised when supplied is_pseudo_bulk provenance is present but not a
    genuine boolean per row: NaN, None mixed with real values, an empty or
    unrecognized string, a length mismatch against the number of rows, or
    any other value the strict boolean parser (parse_strict_bool_array)
    refuses to interpret unambiguously."""


class AssayPolicyMismatchError(AssayPolicyError):
    """Raised when two pieces of assay-policy-bearing state disagree where
    they must agree — e.g. an artifact fit under one assay_policy applied
    to input governed by another, a bundle's declared assay_policy
    disagreeing with its embedded artifact, or a dataset class restricted
    to a single trainable policy being constructed under a different one."""


class BulkTrainingNotImplementedError(RuntimeError):
    """Raised when a caller asks for a TRAINABLE bulk-expression dataset or
    tries to train under assay_policy='bulk_only'. Loading/validating a
    bulk manifest (preprocess.py::load_tcga_bulk_dataset) is supported;
    there is no bulk model or training loop in this project. This is an
    explicit, actionable failure — never a silent fallback to reusing the
    single-cell model or training loop on bulk expression."""


class MultimodalTrainingNotImplementedError(RuntimeError):
    """Raised whenever assay_policy='multimodal' is requested anywhere in
    this codebase. Multimodal support requires separate per-assay encoders
    and an explicit, modality-aware fusion/evaluation protocol, none of
    which exist here. There is deliberately no shallow "accept mixed rows"
    fallback — that is exactly the defect (bulk pseudo-cells silently
    entering a single-cell MIL bag) this module exists to prevent."""


class AssayCapabilities(NamedTuple):
    """What assay_policy actually permits right now, resolved in one place
    instead of scattered string comparisons across modules."""

    policy: str
    accepts_single_cell: bool
    accepts_bulk: bool
    trainable: bool  # False means: loading/validation may work, training must raise.


def validate_assay_policy(policy: str) -> str:
    if policy not in VALID_ASSAY_POLICIES:
        raise AssayPolicyError(
            f"Unknown data.assay_policy={policy!r} — must be one of {sorted(VALID_ASSAY_POLICIES)}."
        )
    return policy


def resolve_assay_capabilities(policy: str) -> AssayCapabilities:
    """
    Central capability resolver — every module that needs to know "can this
    policy accept single-cell rows / bulk rows / actually train" calls this
    instead of re-deriving the answer from policy name comparisons.
    Immediately raises MultimodalTrainingNotImplementedError for
    'multimodal': there is no capability to resolve because no multimodal
    architecture exists.
    """
    validate_assay_policy(policy)
    if policy == ASSAY_POLICY_MULTIMODAL:
        raise MultimodalTrainingNotImplementedError(
            "assay_policy='multimodal' was requested, but this project implements no "
            "multimodal architecture (separate per-assay encoders + modality-aware fusion "
            "+ modality-valid evaluation). Multimodal support stays disabled until that is "
            "genuinely implemented — it is never approximated by accepting mixed single-cell "
            "and pseudo-bulk rows in one run."
        )
    if policy == ASSAY_POLICY_SINGLE_CELL_ONLY:
        return AssayCapabilities(policy, accepts_single_cell=True, accepts_bulk=False, trainable=True)
    return AssayCapabilities(policy, accepts_single_cell=False, accepts_bulk=True, trainable=False)


def require_trainable(policy: str) -> None:
    """Raise the appropriate typed error if `policy` cannot back a real
    training run. single_cell_only is the only trainable policy today."""
    caps = resolve_assay_capabilities(policy)
    if not caps.trainable:
        raise BulkTrainingNotImplementedError(
            f"assay_policy={policy!r} was requested for training, but this project has no "
            "bulk RNA-seq/microarray model or training loop implemented — only manifest/"
            "loading/validation support exists for bulk data. Bulk training stays disabled "
            "with this explicit error rather than silently reusing the single-cell model "
            "or training loop on bulk expression."
        )


_TRUE_STRINGS = frozenset({"true", "1"})
_FALSE_STRINGS = frozenset({"false", "0"})


def parse_strict_bool_array(values: Sequence, n_hint: Optional[int] = None) -> np.ndarray:
    """
    The single canonical strict boolean parser for persisted/user-provided
    assay provenance (is_pseudo_bulk and anything with the same contract).
    Deliberately NOT `np.asarray(...).astype(bool)` or `bool(x)` — both of
    those treat any non-empty string (including "False", "false", "0") as
    truthy, which would silently turn an explicit "this row is NOT
    pseudo-bulk" string marker into is_pseudo_bulk=True.

    Accepted:
      True values  — python bool True, numpy bool_ True, "true", "True", "1"
      False values — python bool False, numpy bool_ False, "false", "False", "0"

    Rejected (raises InvalidAssayProvenanceError): None/NaN, empty string,
    any other string, any non-bool/non-string/non-NaN value, and a length
    that doesn't match n_hint (when given).
    """
    # Fast path: already a genuine numpy boolean array — nothing ambiguous
    # to parse, avoid an O(n) python-level loop over large real datasets.
    arr_fast = values if isinstance(values, np.ndarray) else None
    if arr_fast is not None and arr_fast.dtype == np.bool_:
        if n_hint is not None and len(arr_fast) != n_hint:
            raise InvalidAssayProvenanceError(
                f"assay_policy: is_pseudo_bulk has {len(arr_fast)} entries, expected {n_hint}."
            )
        return arr_fast.astype(bool, copy=True)

    raw = list(values)
    if n_hint is not None and len(raw) != n_hint:
        raise InvalidAssayProvenanceError(
            f"assay_policy: is_pseudo_bulk has {len(raw)} entries, expected {n_hint}."
        )

    out = np.empty(len(raw), dtype=bool)
    bad_indices = []
    for i, v in enumerate(raw):
        parsed = _parse_one_strict_bool(v)
        if parsed is None:
            bad_indices.append(i)
        else:
            out[i] = parsed
    if bad_indices:
        sample = bad_indices[:10]
        raise InvalidAssayProvenanceError(
            f"assay_policy: is_pseudo_bulk contains {len(bad_indices)} invalid/missing "
            f"value(s) at index(es) {sample}{' ...' if len(bad_indices) > 10 else ''} "
            f"(raw values: {[raw[i] for i in sample]!r}) — every row must carry an "
            "explicit, unambiguous boolean. NaN/None/empty-string/unrecognized-string "
            "values are never coerced to a default; 'False' as a string is never "
            "silently treated as truthy."
        )
    return out


def _parse_one_strict_bool(v) -> Optional[bool]:
    """Return True/False for an unambiguous value, or None if `v` cannot be
    strictly parsed (caller collects these as errors, never guesses)."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip()
        if s in _TRUE_STRINGS or s == "True":
            return True
        if s in _FALSE_STRINGS or s == "False":
            return False
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if v is None:
        return None
    # Reject everything else (ints other than 0/1 handled below, objects,
    # etc.) rather than guessing via a bare bool(v) truthiness check.
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        if v == 1:
            return True
        if v == 0:
            return False
        return None
    return None


def _as_bool_array(is_pseudo_bulk: Sequence, n_hint: Optional[int] = None) -> np.ndarray:
    """Back-compat wrapper around parse_strict_bool_array — kept as the
    name existing call sites in this module already use."""
    return parse_strict_bool_array(is_pseudo_bulk, n_hint=n_hint)


def assert_rows_match_policy(
    is_pseudo_bulk: Sequence, policy: str = DEFAULT_ASSAY_POLICY, context: str = "",
) -> None:
    """
    The core row-level enforcement call. Raise AssayPolicyError (or a
    typed *NotImplementedError for multimodal) unless every row in
    `is_pseudo_bulk` is compatible with `policy`:

      single_cell_only — every row must be is_pseudo_bulk=False.
      bulk_only        — every row must be is_pseudo_bulk=True.
      multimodal       — always raises MultimodalTrainingNotImplementedError.

    `is_pseudo_bulk` is read directly off the rows being checked — this
    function never looks at a dataset/source name, so renaming a config
    entry or a converted file cannot bypass it. Called at every boundary
    documented in this module's docstring; `context` is included in the
    error message only, to make it obvious which boundary rejected the data.
    """
    resolve_assay_capabilities(policy)  # raises MultimodalTrainingNotImplementedError early
    arr = _as_bool_array(is_pseudo_bulk)
    where = f" ({context})" if context else ""
    if policy == ASSAY_POLICY_SINGLE_CELL_ONLY:
        n_bulk = int(arr.sum())
        if n_bulk:
            raise AssayPolicyError(
                f"{n_bulk}/{len(arr)} pseudo-bulk row(s) present under "
                f"assay_policy='single_cell_only'{where} — bulk/pseudo-bulk expression "
                "(e.g. GSE994, GSE123352, GSE307690/CANUCK, TCGA-LUAD, TCGA-LUSC) must "
                "never enter a single-cell-only training/evaluation path, regardless of "
                "which source/config list it was loaded from. Configure it under "
                "data.bulk_sources / data.tcga instead, or set assay_policy='bulk_only' "
                "for a dedicated bulk experiment."
            )
    elif policy == ASSAY_POLICY_BULK_ONLY:
        n_single = int((~arr).sum())
        if n_single:
            raise AssayPolicyError(
                f"{n_single}/{len(arr)} real single-cell row(s) present under "
                f"assay_policy='bulk_only'{where} — a bulk-only experiment must not "
                "silently include real single-cell rows."
            )


def assert_row_provenance_present(obs, required=("assay_mode", "is_pseudo_bulk", "data_modality",
                                                   "species", "subject_id")) -> None:
    """
    Fail closed if any required provenance column is absent from an obs-like
    mapping/DataFrame. Never infer "safe" from a missing field — this is the
    counterpart of assert_rows_match_policy() for the columns themselves,
    not just their values.
    """
    missing = [c for c in required if c not in getattr(obs, "columns", obs)]
    if missing:
        raise AssayPolicyError(
            f"assay_policy: input is missing required row-provenance column(s) {missing} — "
            "missing assay provenance is never treated as safe; every observation must "
            "carry an explicit assay_mode/is_pseudo_bulk/data_modality/species/subject_id."
        )


def assert_row_provenance_consistent(assay_mode: Sequence, is_pseudo_bulk: Sequence) -> None:
    """
    Reject internally contradictory row-level provenance:
      is_pseudo_bulk=True  with assay_mode==ASSAY_MODE_SINGLE_CELL, or
      is_pseudo_bulk=False with assay_mode==ASSAY_MODE_BULK_TCGA.
    This is exactly the combination that let GSE994/GSE123352/GSE307690
    bypass the pre-issue-13 assay_mode=='bulk_tcga'-only check: they are
    pseudo-bulk (is_pseudo_bulk=True) but were never stamped
    assay_mode='bulk_tcga', so a name/assay_mode-only check missed them.
    """
    modes = np.asarray([str(m) for m in assay_mode])
    bulk = _as_bool_array(is_pseudo_bulk, n_hint=len(modes))
    bad_single = bulk & (modes == ASSAY_MODE_SINGLE_CELL)
    if bad_single.any():
        raise AssayPolicyError(
            f"{int(bad_single.sum())} row(s) have is_pseudo_bulk=True but "
            f"assay_mode={ASSAY_MODE_SINGLE_CELL!r} — contradictory row-level assay "
            "provenance. A pseudo-bulk row must never be stamped as single-cell assay_mode."
        )
    bad_bulk = (~bulk) & (modes == ASSAY_MODE_BULK_TCGA)
    if bad_bulk.any():
        raise AssayPolicyError(
            f"{int(bad_bulk.sum())} row(s) have is_pseudo_bulk=False but "
            f"assay_mode={ASSAY_MODE_BULK_TCGA!r} — contradictory row-level assay "
            "provenance. A real single-cell row must never be stamped bulk_tcga."
        )


__all__ = [
    "AssayPolicyError",
    "MissingAssayProvenanceError",
    "InvalidAssayProvenanceError",
    "AssayPolicyMismatchError",
    "BulkTrainingNotImplementedError",
    "MultimodalTrainingNotImplementedError",
    "AssayCapabilities",
    "validate_assay_policy",
    "resolve_assay_capabilities",
    "require_trainable",
    "assert_rows_match_policy",
    "assert_row_provenance_present",
    "assert_row_provenance_consistent",
    "parse_strict_bool_array",
    "ASSAY_POLICY_VERSION",
]
