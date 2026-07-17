"""
tests/test_species_separation.py — human/mouse domain-separation guarantees
(NON-NEGOTIABLE rule: never merge human and mouse expression matrices as
if from the same domain).
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import (
    EXPERIMENT_MODE_CROSS_SPECIES_PRETRAINING,
    EXPERIMENT_MODE_HUMAN_ONLY,
    EXPERIMENT_MODE_MOUSE_ONLY,
    SPECIES_HUMAN,
    SPECIES_MOUSE,
)
from data.assembly import merge_sources
from data.species_policy import (
    SpeciesPolicyError,
    assert_single_species_or_explicit,
    mixed_species_allowed,
    namespace_subject_id,
    species_allowed,
    validate_experiment_mode,
)


def _mini_adata(species, subject_prefix, n=20, g=30, seed=0):
    rng = np.random.RandomState(seed)
    genes = [f"G{i}" for i in range(g)]
    X = rng.negative_binomial(5, 0.7, (n, g)).astype("float32")
    obs = pd.DataFrame({
        "subject_id": [f"{subject_prefix}_{i // 5}" for i in range(n)],
        "species": species,
    }, index=[f"c{species}_{i}" for i in range(n)])
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes))


def test_experiment_mode_validation_rejects_unknown_mode():
    with pytest.raises(SpeciesPolicyError):
        validate_experiment_mode("not_a_real_mode")


def test_human_only_mode_excludes_mouse_species():
    assert species_allowed(EXPERIMENT_MODE_HUMAN_ONLY, SPECIES_HUMAN) is True
    assert species_allowed(EXPERIMENT_MODE_HUMAN_ONLY, SPECIES_MOUSE) is False


def test_mouse_only_mode_excludes_human_species():
    assert species_allowed(EXPERIMENT_MODE_MOUSE_ONLY, SPECIES_MOUSE) is True
    assert species_allowed(EXPERIMENT_MODE_MOUSE_ONLY, SPECIES_HUMAN) is False


def test_mixed_species_never_allowed_under_default_mode():
    assert mixed_species_allowed(EXPERIMENT_MODE_HUMAN_ONLY) is False
    assert mixed_species_allowed(EXPERIMENT_MODE_MOUSE_ONLY) is False


def test_mixed_species_requires_explicit_cross_species_mode():
    assert mixed_species_allowed(EXPERIMENT_MODE_CROSS_SPECIES_PRETRAINING) is True


def test_merge_sources_refuses_to_mix_species_under_default_mode():
    human = _mini_adata(SPECIES_HUMAN, "h")
    mouse = _mini_adata(SPECIES_MOUSE, "mouse::m")
    with pytest.raises(SpeciesPolicyError):
        merge_sources(human, mouse, scale=False)  # default experiment_mode=human_only


def test_merge_sources_allows_pure_human_under_default_mode():
    human1 = _mini_adata(SPECIES_HUMAN, "h1")
    human2 = _mini_adata(SPECIES_HUMAN, "h2")
    merged = merge_sources(human1, human2, scale=False)
    assert merged.n_obs == human1.n_obs + human2.n_obs


def test_merge_sources_allows_mixed_species_when_explicitly_opted_in():
    human = _mini_adata(SPECIES_HUMAN, "h")
    mouse = _mini_adata(SPECIES_MOUSE, "mouse::m")
    merged = merge_sources(
        human, mouse, scale=False, experiment_mode=EXPERIMENT_MODE_CROSS_SPECIES_PRETRAINING
    )
    assert merged.n_obs == human.n_obs + mouse.n_obs


def test_assert_single_species_raises_with_named_species_list():
    with pytest.raises(SpeciesPolicyError, match="human"):
        assert_single_species_or_explicit(["human", "mouse"], EXPERIMENT_MODE_HUMAN_ONLY)


def test_namespace_subject_id_prefixes_mouse_only():
    assert namespace_subject_id("sub_1", SPECIES_MOUSE) == "mouse::sub_1"
    assert namespace_subject_id("sub_1", SPECIES_HUMAN) == "sub_1"
    # idempotent — namespacing an already-namespaced id doesn't double-prefix
    assert namespace_subject_id("mouse::sub_1", SPECIES_MOUSE) == "mouse::sub_1"


def test_mouse_subject_id_can_never_collide_with_human_subject_id():
    """A mouse and a human source sharing the same raw subject string (e.g.
    both use "1") must never land in the same bag once namespaced."""
    human_id = namespace_subject_id("1", SPECIES_HUMAN)
    mouse_id = namespace_subject_id("1", SPECIES_MOUSE)
    assert human_id != mouse_id


def test_loaders_stamp_species_and_load_mouse_scrna_namespaces_subjects(tmp_path):
    from data.loaders import load_mouse_scrna, load_scrna

    # human scrna
    genes = [f"G{i}" for i in range(20)]
    obs = pd.DataFrame({"donor_id": ["h1"] * 5 + ["h2"] * 5}, index=[f"c{i}" for i in range(10)])
    human_adata = ad.AnnData(
        X=sp.csr_matrix(np.random.negative_binomial(5, 0.7, (10, 20)).astype("float32")),
        obs=obs, var=pd.DataFrame(index=genes),
    )
    human_path = tmp_path / "human.h5ad"
    human_adata.write_h5ad(human_path)
    loaded_human = load_scrna(str(human_path), "cigarette", "donor_id")
    assert (loaded_human.obs["species"] == SPECIES_HUMAN).all()

    # mouse scrna
    mouse_obs = pd.DataFrame({"donor_id": ["1"] * 5 + ["2"] * 5}, index=[f"mc{i}" for i in range(10)])
    mouse_adata = ad.AnnData(
        X=sp.csr_matrix(np.random.negative_binomial(5, 0.7, (10, 20)).astype("float32")),
        obs=mouse_obs, var=pd.DataFrame(index=genes),
    )
    mouse_path = tmp_path / "mouse.h5ad"
    mouse_adata.write_h5ad(mouse_path)
    loaded_mouse = load_mouse_scrna(str(mouse_path))
    assert (loaded_mouse.obs["species"] == SPECIES_MOUSE).all()
    assert all(s.startswith("mouse::") for s in loaded_mouse.obs["subject_id"])


def test_preprocess_load_all_sources_never_loads_mouse_under_default_mode(tmp_path):
    """preprocess.py::_load_all_sources must not load gse288003_path at all
    under the default experiment_mode, even if the path is configured and
    exists — the mouse source must be excluded before merge_sources ever
    runs, not merely rejected afterward."""
    from preprocess import _load_all_sources

    genes = [f"G{i}" for i in range(20)]
    mouse_obs = pd.DataFrame({"donor_id": ["1"] * 5}, index=[f"mc{i}" for i in range(5)])
    mouse_adata = ad.AnnData(
        X=sp.csr_matrix(np.random.negative_binomial(5, 0.7, (5, 20)).astype("float32")),
        obs=mouse_obs, var=pd.DataFrame(index=genes),
    )
    mouse_path = tmp_path / "mouse.h5ad"
    mouse_adata.write_h5ad(mouse_path)

    cfg = {"gse288003_path": str(mouse_path)}
    with pytest.raises(ValueError):
        # no human sources configured either -> "no data sources found",
        # proving the mouse source was skipped rather than silently loaded
        _load_all_sources(cfg)


# ─── configs/default.yaml agrees with the documented human_only default ────

def test_default_config_experiment_mode_is_human_only():
    """A regression guard against config drift: if someone accidentally
    flips configs/default.yaml's data.experiment_mode away from
    human_only, GSE288003 (mouse) would silently re-enter every default
    run. This test fails loudly the moment that default changes, rather
    than relying on a human reviewer to notice a one-line YAML edit."""
    import yaml

    repo_root = Path(__file__).parents[1]
    with open(repo_root / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["data"]["experiment_mode"] == EXPERIMENT_MODE_HUMAN_ONLY
