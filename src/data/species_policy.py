"""
data/species_policy.py — explicit experiment-mode gate for mixing human and
mouse expression data (NON-NEGOTIABLE rule: never merge human and mouse
expression matrices as if from the same domain).

Before this module existed, preprocess.py's `_load_all_sources` loaded
GSE288003 (mouse lung, e-cig aerosol), ran it through
data/transforms.py::map_mouse_to_human (ortholog mapping to human gene
symbols), and appended it to the SAME list of AnnData objects as every real
human source — data/assembly.py::merge_sources then concatenated all of
them together unconditionally. Ortholog-mapped mouse expression is not the
same domain as measured human expression: different organism, different
library prep/assay batch, an imperfect and lossy gene mapping. Training or
evaluating as if they were one dataset conflates species effects with
smoke/cancer effects with no way to separate them afterward.

This module is the single gate every mouse-including code path must pass
through. Default experiment_mode is human_only: a mouse source is never
loaded in the first place. Any other mode requires an explicit, named
config choice, and even then this module enforces:
  - every subject/animal id is namespaced by species so a mouse "sub_1"
    can never collide with a human "sub_1" in a split manifest or a bag;
  - mouse subjects are never included in a human validation/test split;
  - mixing species in one call to merge_sources requires the caller to
    pass allow_mixed_species=True (and only cross_species_pretraining /
    cross_species_domain_adaptation modes may do so).
"""

from typing import Sequence

from constants import (
    EXPERIMENT_MODE_CROSS_SPECIES_DOMAIN_ADAPT,
    EXPERIMENT_MODE_CROSS_SPECIES_PRETRAINING,
    EXPERIMENT_MODE_HUMAN_ONLY,
    EXPERIMENT_MODE_MOUSE_ONLY,
    SPECIES_HUMAN,
    SPECIES_MOUSE,
    VALID_EXPERIMENT_MODES,
    VALID_SPECIES,
)

# Species prefix used to namespace a non-human subject/animal id so it can
# never collide with a human subject_id sharing the same raw string (e.g.
# both a human donor and a mouse GSM sample end up called "1"). See
# namespace_subject_id().
MOUSE_SUBJECT_NAMESPACE = "mouse::"


class SpeciesPolicyError(ValueError):
    """Raised when a code path would mix species in a way the current
    experiment_mode does not explicitly allow."""


def validate_experiment_mode(mode: str) -> str:
    if mode not in VALID_EXPERIMENT_MODES:
        raise SpeciesPolicyError(
            f"Unknown data.experiment_mode={mode!r} — must be one of {sorted(VALID_EXPERIMENT_MODES)}."
        )
    return mode


def species_allowed(experiment_mode: str, species: str) -> bool:
    """Whether `species` may be loaded at all under `experiment_mode`."""
    validate_experiment_mode(experiment_mode)
    if species not in VALID_SPECIES:
        raise SpeciesPolicyError(f"Unknown species {species!r} — must be one of {sorted(VALID_SPECIES)}.")
    if experiment_mode == EXPERIMENT_MODE_HUMAN_ONLY:
        return species == SPECIES_HUMAN
    if experiment_mode == EXPERIMENT_MODE_MOUSE_ONLY:
        return species == SPECIES_MOUSE
    # cross_species_pretraining / cross_species_domain_adaptation: both
    # species may be loaded, but see mixed_species_allowed() below for
    # whether they may be concatenated into one merge_sources() call.
    return True


def mixed_species_allowed(experiment_mode: str) -> bool:
    """Whether a single merge_sources() call may legally contain more than
    one species under `experiment_mode`. human_only/mouse_only never allow
    this (there is only ever one species present by construction);
    cross_species_pretraining/cross_species_domain_adaptation opt in
    explicitly."""
    validate_experiment_mode(experiment_mode)
    return experiment_mode in (
        EXPERIMENT_MODE_CROSS_SPECIES_PRETRAINING,
        EXPERIMENT_MODE_CROSS_SPECIES_DOMAIN_ADAPT,
    )


def namespace_subject_id(subject_id: str, species: str) -> str:
    """
    Prefix a non-human subject/animal id so it can never collide with a
    human subject_id. Human ids are returned unchanged (namespacing every
    id would silently change every existing human split manifest's subject
    strings — an unrelated, unnecessary behavior change).
    """
    sid = str(subject_id)
    if species == SPECIES_MOUSE and not sid.startswith(MOUSE_SUBJECT_NAMESPACE):
        return f"{MOUSE_SUBJECT_NAMESPACE}{sid}"
    return sid


def assert_single_species_or_explicit(species_values: Sequence[str], experiment_mode: str) -> None:
    """
    Raise SpeciesPolicyError if `species_values` (one entry per source/cell
    being merged) contains more than one distinct species and
    experiment_mode does not explicitly allow mixing. Called by
    data/assembly.py::merge_sources before it concatenates anything.
    """
    validate_experiment_mode(experiment_mode)
    distinct = sorted(set(species_values))
    if len(distinct) <= 1:
        return
    if not mixed_species_allowed(experiment_mode):
        raise SpeciesPolicyError(
            f"Refusing to merge sources spanning species {distinct} under "
            f"experiment_mode={experiment_mode!r}. Ortholog-mapped mouse expression is "
            "not the same domain as measured human expression — merging them "
            "unconditionally conflates species effects with the effects this pipeline "
            "is trying to measure. Set data.experiment_mode to "
            "'cross_species_pretraining' or 'cross_species_domain_adaptation' to "
            "explicitly opt into a cross-species run, or keep species separate "
            "(human_only / mouse_only, the default is human_only)."
        )
