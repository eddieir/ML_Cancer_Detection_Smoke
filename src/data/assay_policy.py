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


def _as_bool_array(is_pseudo_bulk: Sequence, n_hint: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(is_pseudo_bulk)
    if arr.dtype == object:
        # Never treat a missing/None marker as False (a real cell) by
        # default — fail closed on ambiguous provenance.
        if any(v is None for v in arr.tolist()):
            raise AssayPolicyError(
                "assay_policy: is_pseudo_bulk contains None/missing value(s) — row-level "
                "assay provenance must be an explicit boolean for every row. A missing "
                "value is never assumed to mean 'real cell'."
            )
    arr = arr.astype(bool)
    if n_hint is not None and len(arr) != n_hint:
        raise AssayPolicyError(
            f"assay_policy: is_pseudo_bulk has {len(arr)} entries, expected {n_hint}."
        )
    return arr


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
    "BulkTrainingNotImplementedError",
    "MultimodalTrainingNotImplementedError",
    "AssayCapabilities",
    "validate_assay_policy",
    "resolve_assay_capabilities",
    "require_trainable",
    "assert_rows_match_policy",
    "assert_row_provenance_present",
    "assert_row_provenance_consistent",
    "ASSAY_POLICY_VERSION",
]
