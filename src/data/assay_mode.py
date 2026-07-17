"""
data/assay_mode.py — explicit single_cell / bulk_tcga assay-mode guard.

KNOWN EXISTING LIMITATION (see README.md's "Bulk/single-cell separation"
section): this project's current default pipeline (preprocess.py::
run_pipeline / run_pipeline_split_aware, as configured by
configs/default.yaml's `microarray_sources`) already merges TCGA-LUAD/
TCGA-LUSC pseudo-bulk samples (data/loaders.py::load_microarray,
is_pseudo_bulk=True) into the SAME AnnData used to build CellLevelDataset
— i.e. bulk RNA-seq "samples" are currently treated as single MIL-bag
"cells" alongside real single-cell data. This is a pre-existing, documented
design choice (see qc_filter/normalize's explicit pseudo-bulk branches)
that this module does not retroactively change — undoing it touches the
default training data composition, model dimensions, and every benchmark
that currently runs against the merged data, which is out of scope for
this change.

What this module provides is the explicit assay-mode vocabulary
(constants.ASSAY_MODE_SINGLE_CELL / ASSAY_MODE_BULK_TCGA) and a guard
function any NEW single-cell-only code path can call to refuse pseudo-bulk
rows explicitly instead of silently accepting them — see
assert_no_pseudo_bulk_rows(). Adopting this guard in the existing default
pipeline is tracked as follow-up work, not done here.
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
