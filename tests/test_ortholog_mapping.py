"""
tests/test_ortholog_mapping.py — versioned mouse->human ortholog mapping
artifact. Uses a small fixed fixture only; never queries live BioMart.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.ortholog import (
    OrthologMappingArtifact,
    build_ortholog_artifact,
    resolve_mapping_pairs,
)
from data.transforms import map_mouse_to_human

FIXTURE_PAIRS = [
    ("Mgene1", "HGENE1"),   # clean one-to-one
    ("Mgene2", "HGENE2"),   # clean one-to-one
    ("Mgene3", "HGENE3A"),  # one mouse gene -> two human candidates (ambiguous)
    ("Mgene3", "HGENE3B"),
    ("Mgene4", "HGENE4"),   # two mouse genes -> same human gene (ambiguous)
    ("Mgene5", "HGENE4"),
    ("Mgene6", "HGENE6"),   # clean one-to-one
]


def test_resolve_mapping_pairs_one_to_one_only_drops_ambiguous_cases():
    mapping, counts = resolve_mapping_pairs(FIXTURE_PAIRS, policy="one_to_one_only")
    assert mapping == {"Mgene1": "HGENE1", "Mgene2": "HGENE2", "Mgene6": "HGENE6"}
    assert counts["n_ambiguous_one_to_many"] == 1  # Mgene3
    assert counts["n_ambiguous_many_to_one"] == 1  # HGENE4 claimed twice
    assert counts["n_final_retained"] == 3


def test_resolve_mapping_pairs_rejects_unknown_policy():
    with pytest.raises(ValueError):
        resolve_mapping_pairs(FIXTURE_PAIRS, policy="not_a_real_policy")


def test_build_ortholog_artifact_records_counts_and_fingerprint():
    artifact = build_ortholog_artifact(FIXTURE_PAIRS, source="fixture", release="test-1")
    assert artifact.n_input_pairs == len(FIXTURE_PAIRS)
    assert artifact.n_final_retained == 3
    assert artifact.policy == "one_to_one_only"
    fp = artifact.fingerprint()
    assert isinstance(fp, str) and len(fp) == 64


def test_artifact_fingerprint_changes_when_mapping_changes():
    a1 = build_ortholog_artifact(FIXTURE_PAIRS, source="fixture")
    a2 = build_ortholog_artifact(FIXTURE_PAIRS[:-1], source="fixture")
    assert a1.fingerprint() != a2.fingerprint()


def test_artifact_round_trip_save_load(tmp_path):
    artifact = build_ortholog_artifact(FIXTURE_PAIRS, source="fixture", release="test-1")
    path = tmp_path / "ortholog_artifact.json"
    artifact.save(path)
    reloaded = OrthologMappingArtifact.load(path)
    assert reloaded.mapping == artifact.mapping
    assert reloaded.fingerprint() == artifact.fingerprint()


def test_map_mouse_to_human_uses_injected_fixture_artifact_no_network():
    artifact = build_ortholog_artifact(FIXTURE_PAIRS, source="fixture")
    genes = ["Mgene1", "Mgene2", "Mgene3", "Mgene4", "unrelated_gene"]
    X = np.random.negative_binomial(5, 0.7, (6, len(genes))).astype("float32")
    obs = pd.DataFrame({"donor_id": ["m1"] * 6}, index=[f"c{i}" for i in range(6)])
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes))

    mapped = map_mouse_to_human(adata, artifact=artifact)
    # only Mgene1/Mgene2 survive one_to_one_only; Mgene3 (ambiguous) and
    # Mgene4 (duplicate human target) and the unmapped gene are dropped.
    assert set(mapped.var_names) == {"HGENE1", "HGENE2"}
    assert mapped.uns["ortholog_mapping_fingerprint"] == artifact.fingerprint()
    assert mapped.uns["ortholog_mapping_policy"] == "one_to_one_only"


def test_map_mouse_to_human_load_from_cached_artifact_path(tmp_path):
    artifact = build_ortholog_artifact(FIXTURE_PAIRS, source="fixture")
    path = tmp_path / "cached.json"
    artifact.save(path)

    genes = ["Mgene1", "Mgene6"]
    X = np.random.negative_binomial(5, 0.7, (4, len(genes))).astype("float32")
    obs = pd.DataFrame({"donor_id": ["m1"] * 4}, index=[f"c{i}" for i in range(4)])
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes))

    mapped = map_mouse_to_human(adata, artifact_path=str(path))
    assert set(mapped.var_names) == {"HGENE1", "HGENE6"}
