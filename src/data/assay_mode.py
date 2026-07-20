"""
data/assay_mode.py — explicit single_cell / bulk_tcga assay-mode vocabulary
and row-level guard, kept for the TCGA-specific "assay_mode" tag.

HISTORICAL NOTE (see GitHub issue #13 and README.md's "Bulk/single-cell
separation" section): before issue #13, the default pipeline
(preprocess.py::_load_all_sources) only rejected rows whose obs["assay_mode"]
was explicitly "bulk_tcga" — TCGA's own tag. GSE994, GSE123352, and
GSE307690/CANUCK are also pseudo-bulk (data/loaders.py::load_microarray,
is_pseudo_bulk=True) but were never stamped assay_mode="bulk_tcga" (only
TCGA's converter writes that tag), so that name-keyed check silently let
them into the same AnnData used to build CellLevelDataset. That gap is now
closed by data/assay_policy.py, which enforces the experiment-level assay
policy against obs["is_pseudo_bulk"] directly — the actual row-level fact
of whether a row is a real cell — rather than the assay_mode tag or a
source/accession name. See data/assay_policy.py's module docstring for the
full enforcement surface.

This module still provides the narrower assay_mode vocabulary
(constants.ASSAY_MODE_SINGLE_CELL / ASSAY_MODE_BULK_TCGA) and its own
is_pseudo_bulk guard functions, used as an additional, independent check
specifically for TCGA's assay_mode tag (see preprocess.py::
_load_all_sources's second check) and by data/assay_policy.py's
AssayPolicyError, which subclasses AssayModeError for backward
compatibility with call sites written against the pre-issue-13 guard.
"""

from typing import Sequence

import numpy as np

from constants import ASSAY_MODE_BULK_TCGA, ASSAY_MODE_SINGLE_CELL, VALID_ASSAY_MODES


class AssayModeError(ValueError):
    """Raised when bulk and single-cell rows would be combined under a
    mode that does not explicitly allow it."""


def validate_assay_mode(mode: str) -> str:
    if mode not in VALID_ASSAY_MODES:
        raise AssayModeError(f"Unknown assay_mode {mode!r} — must be one of {sorted(VALID_ASSAY_MODES)}.")
    return mode


def assert_no_pseudo_bulk_rows(is_pseudo_bulk: Sequence[bool], assay_mode: str = ASSAY_MODE_SINGLE_CELL) -> None:
    """
    Raise AssayModeError if any row is pseudo-bulk while assay_mode is
    ASSAY_MODE_SINGLE_CELL. A caller building a single-cell-only dataset
    (e.g. a new, stricter CellLevelDataset construction path) should call
    this before accepting rows sourced from data/loaders.py::load_microarray
    (TCGA, GSE994, GSE123352, CANUCK) rather than assuming is_pseudo_bulk
    is already filtered out.
    """
    validate_assay_mode(assay_mode)
    if assay_mode != ASSAY_MODE_SINGLE_CELL:
        return
    arr = np.asarray(is_pseudo_bulk, dtype=bool)
    n_bulk = int(arr.sum())
    if n_bulk:
        raise AssayModeError(
            f"{n_bulk} pseudo-bulk row(s) present under assay_mode='single_cell' — bulk "
            "expression (e.g. TCGA microarray/RNA-seq pseudo-bulk samples) must not enter "
            "a single-cell-only training/evaluation path. Use assay_mode='bulk_tcga' for a "
            "dedicated bulk experiment, or filter these rows out first."
        )


def assert_no_single_cell_rows(is_pseudo_bulk: Sequence[bool], assay_mode: str = ASSAY_MODE_BULK_TCGA) -> None:
    """Mirror check for a bulk-only path: raise if any row is NOT
    pseudo-bulk while assay_mode is ASSAY_MODE_BULK_TCGA."""
    validate_assay_mode(assay_mode)
    if assay_mode != ASSAY_MODE_BULK_TCGA:
        return
    arr = np.asarray(is_pseudo_bulk, dtype=bool)
    n_single = int((~arr).sum())
    if n_single:
        raise AssayModeError(
            f"{n_single} single-cell row(s) present under assay_mode='bulk_tcga' — a bulk-only "
            "experiment must not silently include real single-cell rows."
        )
