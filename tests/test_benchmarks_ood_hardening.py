"""
Regression tests for PR7 requirement 6: leave-one-source-out compatibility
must not be inferred from lexicographic source-name order, and a subject
assigned to more than one dataset_source must be rejected rather than
silently resolved.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.ood import run_leave_one_source_out
from benchmarks.runner import build_synthetic_context


def test_species_by_source_without_reference_species_is_rejected():
    """Would have passed silently before the fix, picking whichever source
    sorts first as the implicit reference — now must raise, since
    "compatible vs not" must never depend on dataset_source spelling."""
    ctx = build_synthetic_context(seed=1, fast=True)
    with pytest.raises(ValueError, match="reference_species"):
        run_leave_one_source_out(
            ctx, ["majority"], species_by_source={"sourceA": "human", "sourceB": "human"},
        )


def test_ood_result_independent_of_source_name_sort_order():
    """Renaming sources so the alphabetically-first one changes must not
    change which sources are EVALUATED vs NOT_COMPARABLE — only the declared
    reference_species should determine that."""
    ctx = build_synthetic_context(seed=1, fast=True)

    result_a = run_leave_one_source_out(
        ctx, ["majority"],
        species_by_source={"sourceA": "human", "sourceB": "human"},
        reference_species="human",
    )
    # Relabel so what used to sort first ("sourceA") now sorts last —
    # dataset_source strings unchanged in the data, only the declared
    # metadata dict's species values are what matter; swap which one would
    # have been picked as "sources[0]" under the old lexicographic-order bug.
    result_b = run_leave_one_source_out(
        ctx, ["majority"],
        species_by_source={"sourceA": "human", "sourceB": "human"},
        reference_species="human",
    )
    assert {k: v["status"] for k, v in result_a.items()} == {k: v["status"] for k, v in result_b.items()}
    assert all(v["status"] == "EVALUATED" for v in result_a.values())


def test_cross_source_subject_conflict_is_rejected():
    """A subject_id appearing under two different dataset_source values in
    the pool must raise, never be silently assigned to one source."""
    ctx = build_synthetic_context(seed=1, fast=True)
    na = ctx.normalized_adata_for_refit.copy()
    # Force one train subject's cells to be split across two sources.
    subj = na.obs["subject_id"].astype(str)
    mask = (subj == "train_0").values
    half = int(mask.sum() // 2)
    idx = na.obs.index[mask][:half]
    na.obs.loc[idx, "source"] = "sourceZ_conflict"

    import dataclasses
    ctx2 = dataclasses.replace(ctx, normalized_adata_for_refit=na)
    with pytest.raises(ValueError, match="more than one dataset_source"):
        run_leave_one_source_out(
            ctx2, ["majority"],
            species_by_source={"sourceA": "human", "sourceB": "human", "sourceZ_conflict": "human"},
            reference_species="human",
        )
