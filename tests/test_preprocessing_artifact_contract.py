"""
Phase 4 — reproducible preprocessing artifact contract.

Covers: scientific-fingerprint determinism/sensitivity (volatile fields
excluded, scientific fields included), gene-contract policy enforcement
(duplicate/missing/unexpected/coverage), safe serialization (atomic
save/load, corruption/schema-version detection), and the checkpoint <->
artifact fingerprint binding used by train.py/inference.py.
"""
import copy
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.preprocessing import (
    ArtifactCompatibilityError,
    GeneContractError,
    LegacyArtifactError,
    PreprocessingArtifact,
    PreprocessingArtifactError,
    apply_preprocessing,
    fit_preprocessing,
    verify_compatible,
)


def _adata(n_subjects=10, cells_per_subject=20, n_genes=50, seed=0, offset=0.0):
    rng = np.random.default_rng(seed)
    subject_ids = []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * cells_per_subject
    n = len(subject_ids)
    X = (rng.random((n, n_genes)) + offset).astype("float32")
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame({
        "subject_id": subject_ids,
        "batch": ["source_0"] * n,
    }, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))


def _fit(n_subjects=10, n_genes=50, seed=0):
    adata = _adata(n_subjects=n_subjects, n_genes=n_genes, seed=seed)
    train_subjects = {f"sub_{i}" for i in range(n_subjects)}
    return fit_preprocessing(adata, train_subjects, n_hvgs=n_genes), adata


# ─── Scientific fingerprint ─────────────────────────────────────────────────

def test_fingerprint_deterministic_for_identical_content():
    artifact, _ = _fit(seed=1)
    assert artifact.scientific_fingerprint() == artifact.scientific_fingerprint()


def test_fingerprint_identical_across_two_independent_fits_of_same_data():
    a1, _ = _fit(seed=7)
    a2, _ = _fit(seed=7)
    assert a1.scientific_fingerprint() == a2.scientific_fingerprint()


def test_fingerprint_excludes_creation_timestamp():
    artifact, _ = _fit(seed=2)
    fp_before = artifact.scientific_fingerprint()
    artifact.created_at = artifact.created_at + 999999
    assert artifact.scientific_fingerprint() == fp_before


def test_fingerprint_changes_when_gene_list_changes():
    artifact, _ = _fit(seed=3)
    fp_before = artifact.scientific_fingerprint()
    mutated = copy.deepcopy(artifact)
    mutated.gene_list = list(reversed(mutated.gene_list))
    assert mutated.scientific_fingerprint() != fp_before


def test_fingerprint_changes_when_scaling_state_changes():
    artifact, _ = _fit(seed=4)
    fp_before = artifact.scientific_fingerprint()
    mutated = copy.deepcopy(artifact)
    mutated.gene_means[0] += 1.0
    assert mutated.scientific_fingerprint() != fp_before


def test_fingerprint_changes_when_gene_contract_policy_changes():
    artifact, _ = _fit(seed=5)
    fp_before = artifact.scientific_fingerprint()
    mutated = copy.deepcopy(artifact)
    mutated.unexpected_gene_policy = "error"
    assert mutated.scientific_fingerprint() != fp_before


def test_fingerprint_unaffected_by_output_directory(tmp_path):
    artifact, _ = _fit(seed=6)
    fp_before = artifact.scientific_fingerprint()
    d1 = tmp_path / "run_a"
    d2 = tmp_path / "some" / "other" / "run_b"
    artifact.save(d1 / "artifact.json")
    reloaded = PreprocessingArtifact.load(d1 / "artifact.json")
    artifact.save(d2 / "artifact.json")
    reloaded2 = PreprocessingArtifact.load(d2 / "artifact.json")
    assert reloaded.scientific_fingerprint() == fp_before
    assert reloaded2.scientific_fingerprint() == fp_before


# ─── Safe serialization / round-trip / corruption ──────────────────────────

def test_save_load_round_trip_preserves_fingerprint(tmp_path):
    artifact, _ = _fit(seed=8)
    path = tmp_path / "artifact.json"
    artifact.save(path)
    reloaded = PreprocessingArtifact.load(path)
    assert reloaded.scientific_fingerprint() == artifact.scientific_fingerprint()
    assert reloaded.gene_list == artifact.gene_list
    assert reloaded.gene_means == artifact.gene_means


def test_apply_preprocessing_identical_before_and_after_serialization(tmp_path):
    artifact, adata = _fit(seed=9)
    path = tmp_path / "artifact.json"
    artifact.save(path)
    reloaded = PreprocessingArtifact.load(path)
    out_before = apply_preprocessing(adata, artifact)
    out_after = apply_preprocessing(adata, reloaded)
    np.testing.assert_array_equal(np.asarray(out_before.X), np.asarray(out_after.X))


def test_corrupted_json_is_rejected(tmp_path):
    artifact, _ = _fit(seed=10)
    path = tmp_path / "artifact.json"
    artifact.save(path)
    with open(path, "w") as f:
        f.write("{not valid json")
    with pytest.raises(PreprocessingArtifactError):
        PreprocessingArtifact.load(path)


def test_unknown_schema_version_is_rejected(tmp_path):
    artifact, _ = _fit(seed=11)
    path = tmp_path / "artifact.json"
    artifact.save(path)
    d = json.loads(path.read_text())
    d["version"] = "999"
    path.write_text(json.dumps(d))
    with pytest.raises(LegacyArtifactError):
        PreprocessingArtifact.load(path)


def test_missing_component_is_rejected(tmp_path):
    artifact, _ = _fit(seed=12)
    path = tmp_path / "artifact.json"
    artifact.save(path)
    d = json.loads(path.read_text())
    del d["gene_stds"]
    path.write_text(json.dumps(d))
    with pytest.raises(PreprocessingArtifactError):
        PreprocessingArtifact.load(path)


def test_inconsistent_gene_and_statistic_lengths_rejected():
    artifact, _ = _fit(seed=13)
    d = artifact.to_dict()
    d["gene_stds"] = d["gene_stds"][:-1]
    with pytest.raises(GeneContractError):
        PreprocessingArtifact(**d)


def test_empty_gene_list_rejected():
    artifact, _ = _fit(seed=14)
    d = artifact.to_dict()
    d["gene_list"], d["gene_means"], d["gene_stds"] = [], [], []
    with pytest.raises(GeneContractError):
        PreprocessingArtifact(**d)


def test_duplicate_selected_genes_rejected():
    artifact, _ = _fit(seed=15)
    d = artifact.to_dict()
    d["gene_list"] = [d["gene_list"][0]] + d["gene_list"][1:]
    d["gene_list"][1] = d["gene_list"][0]
    with pytest.raises(GeneContractError):
        PreprocessingArtifact(**d)


# ─── Gene compatibility / inference contract ───────────────────────────────

def test_correct_input_succeeds():
    artifact, adata = _fit(seed=16)
    out = apply_preprocessing(adata, artifact)
    assert list(out.var_names) == artifact.gene_list


def test_reordered_input_is_restored_deterministically():
    artifact, adata = _fit(seed=17)
    shuffled = adata[:, list(reversed(list(adata.var_names)))].copy()
    out = apply_preprocessing(shuffled, artifact)
    assert list(out.var_names) == artifact.gene_list


def test_missing_gene_fails_by_default():
    artifact, adata = _fit(seed=18)
    trimmed = adata[:, list(adata.var_names)[1:]].copy()
    with pytest.raises(GeneContractError):
        apply_preprocessing(trimmed, artifact)


def test_duplicate_gene_in_input_fails_by_default():
    artifact, adata = _fit(seed=19)
    dup_names = list(adata.var_names) + [list(adata.var_names)[0]]
    with pytest.raises(GeneContractError):
        verify_compatible(artifact, dup_names)


def test_unexpected_genes_ignored_by_default_but_recorded_in_diagnostics():
    artifact, adata = _fit(seed=20)
    extra = adata.copy()
    extra.var_names = list(extra.var_names)
    padded = ad.concat(
        [extra, ad.AnnData(X=np.zeros((extra.n_obs, 1), dtype="float32"),
                            obs=extra.obs, var=pd.DataFrame(index=["EXTRA_GENE"]))],
        axis=1, join="outer",
    )
    diagnostics = verify_compatible(artifact, list(padded.var_names))
    assert "EXTRA_GENE" in diagnostics["unexpected"]


def test_unexpected_genes_rejected_when_policy_is_error():
    artifact, adata = _fit(seed=21)
    d = artifact.to_dict()
    d["unexpected_gene_policy"] = "error"
    strict = PreprocessingArtifact(**d)
    names = list(adata.var_names) + ["EXTRA_GENE"]
    with pytest.raises(GeneContractError):
        verify_compatible(strict, names)


def test_insufficient_coverage_rejected():
    artifact, adata = _fit(seed=22, n_genes=20)
    d = artifact.to_dict()
    d["missing_gene_policy"] = "zero_fill"
    d["minimum_gene_coverage"] = 0.99
    lenient = PreprocessingArtifact(**d)
    trimmed = adata[:, list(adata.var_names)[3:]].copy()  # drop 3/20 genes -> 85% coverage
    with pytest.raises(GeneContractError):
        apply_preprocessing(trimmed, lenient)


def test_zero_fill_requires_explicit_opt_in_and_is_recorded():
    artifact, adata = _fit(seed=23, n_genes=20)
    d = artifact.to_dict()
    d["missing_gene_policy"] = "zero_fill"
    d["minimum_gene_coverage"] = 0.5
    lenient = PreprocessingArtifact(**d)
    trimmed = adata[:, list(adata.var_names)[1:]].copy()
    out = apply_preprocessing(trimmed, lenient)
    diag = out.uns["preprocessing_compatibility_diagnostics"]
    assert len(diag["missing"]) == 1
    assert out.n_vars == len(lenient.gene_list)


def test_artifact_remains_unchanged_after_repeated_application():
    artifact, adata = _fit(seed=24)
    fp_before = artifact.scientific_fingerprint()
    apply_preprocessing(adata, artifact)
    apply_preprocessing(adata, artifact)
    assert artifact.scientific_fingerprint() == fp_before


def test_repeated_transformation_is_identical():
    artifact, adata = _fit(seed=25)
    out1 = apply_preprocessing(adata, artifact)
    out2 = apply_preprocessing(adata, artifact)
    np.testing.assert_array_equal(np.asarray(out1.X), np.asarray(out2.X))


def test_validation_before_or_after_test_application_gives_identical_per_dataset_results():
    artifact, adata = _fit(seed=26)
    val = adata[:15].copy()
    test = adata[15:30].copy()

    out_val_first = apply_preprocessing(val, artifact)
    out_test_first = apply_preprocessing(test, artifact)

    out_test_second = apply_preprocessing(test, artifact)
    out_val_second = apply_preprocessing(val, artifact)

    np.testing.assert_array_equal(np.asarray(out_val_first.X), np.asarray(out_val_second.X))
    np.testing.assert_array_equal(np.asarray(out_test_first.X), np.asarray(out_test_second.X))


# ─── Inspection report ──────────────────────────────────────────────────────

def test_inspect_report_contains_no_raw_arrays():
    artifact, _ = _fit(seed=27)
    report = artifact.inspect()
    assert "gene_means" not in report
    assert "gene_stds" not in report
    assert report["artifact_fingerprint"] == artifact.scientific_fingerprint()
    assert report["selected_gene_count"] == len(artifact.gene_list)


# ─── Configuration consistency ─────────────────────────────────────────────

def test_config_defaults_match_code_defaults():
    """configs/default.yaml's preprocessing.* gene-contract keys must agree
    with data/preprocessing.py's DEFAULT_* constants — a drift here would
    mean the documented/configured default is not what actually runs."""
    import yaml

    from data.preprocessing import (
        DEFAULT_DUPLICATE_GENE_POLICY,
        DEFAULT_MINIMUM_GENE_COVERAGE,
        DEFAULT_MISSING_GENE_POLICY,
        DEFAULT_UNEXPECTED_GENE_POLICY,
    )

    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    pp = config["preprocessing"]
    assert pp["missing_gene_policy"] == DEFAULT_MISSING_GENE_POLICY
    assert pp["duplicate_gene_policy"] == DEFAULT_DUPLICATE_GENE_POLICY
    assert pp["unexpected_gene_policy"] == DEFAULT_UNEXPECTED_GENE_POLICY
    assert float(pp["minimum_gene_coverage"]) == DEFAULT_MINIMUM_GENE_COVERAGE
