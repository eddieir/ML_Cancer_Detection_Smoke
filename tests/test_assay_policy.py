"""
tests/test_assay_policy.py — GitHub issue #13: bulk/pseudo-bulk samples must
never silently enter the single-cell/hierarchical-MIL pipeline.

Regression coverage for the actual defect (GSE994/GSE123352/GSE307690 have
obs["is_pseudo_bulk"]=True but obs["assay_mode"] stays the single-cell
default, so the pre-issue-13 assay_mode=='bulk_tcga'-only check missed
them) plus adversarial coverage of every enforcement boundary the fix
touches: source loading, merge_sources, fit/apply preprocessing,
CellLevelDataset/bag construction, artifact fingerprinting, bundle
manifests, and the frozen-test guard's non-interference.
"""
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import (
    ASSAY_POLICY_BULK_ONLY,
    ASSAY_POLICY_MULTIMODAL,
    ASSAY_POLICY_SINGLE_CELL_ONLY,
    ASSAY_POLICY_VERSION,
    DEFAULT_ASSAY_POLICY,
)
from data.assay_policy import (
    AssayPolicyError,
    BulkTrainingNotImplementedError,
    MultimodalTrainingNotImplementedError,
    assert_rows_match_policy,
    require_trainable,
    resolve_assay_capabilities,
    validate_assay_policy,
)


def _mini_adata(n=30, g=200, is_pseudo_bulk=False, subject_prefix="s", seed=0):
    rng = np.random.RandomState(seed)
    genes = [f"G{i}" for i in range(g)]
    X = rng.negative_binomial(5, 0.6, (n, g)).astype("float32")
    obs = pd.DataFrame({
        "donor_id":       [f"{subject_prefix}_{i}" for i in range(n)],
        "subject_id":     [f"{subject_prefix}_{i}" for i in range(n)],
        "smoke_type":     rng.randint(0, 6, n),
        "smoke_type_known": True,
        "data_modality":  "microarray" if is_pseudo_bulk else "scrna",
        "is_pseudo_bulk": is_pseudo_bulk,
        "species":        "human",
        "assay_mode":     "human_single_cell",
        "malignancy":     0.0,
        "malignancy_known": False,
        "cell_type_id":   0,
    }, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes))


# ─── Central policy module — basic contract ─────────────────────────────────

def test_default_assay_policy_is_single_cell_only():
    assert DEFAULT_ASSAY_POLICY == ASSAY_POLICY_SINGLE_CELL_ONLY


def test_unknown_assay_policy_rejected():
    with pytest.raises(AssayPolicyError):
        validate_assay_policy("mixed_everything")


def test_multimodal_raises_typed_not_implemented_error():
    with pytest.raises(MultimodalTrainingNotImplementedError):
        resolve_assay_capabilities(ASSAY_POLICY_MULTIMODAL)
    with pytest.raises(MultimodalTrainingNotImplementedError):
        assert_rows_match_policy([False, True], ASSAY_POLICY_MULTIMODAL)


def test_bulk_only_training_raises_typed_not_implemented_error():
    with pytest.raises(BulkTrainingNotImplementedError):
        require_trainable(ASSAY_POLICY_BULK_ONLY)


def test_single_cell_only_is_trainable():
    require_trainable(ASSAY_POLICY_SINGLE_CELL_ONLY)  # must not raise


def test_single_cell_only_rejects_any_pseudo_bulk_row():
    with pytest.raises(AssayPolicyError):
        assert_rows_match_policy([False, False, True], ASSAY_POLICY_SINGLE_CELL_ONLY)


def test_bulk_only_rejects_any_real_cell_row():
    with pytest.raises(AssayPolicyError):
        assert_rows_match_policy([True, True, False], ASSAY_POLICY_BULK_ONLY)


def test_single_cell_only_accepts_all_real_cells():
    assert_rows_match_policy([False, False, False], ASSAY_POLICY_SINGLE_CELL_ONLY)  # no raise


# ─── Default configuration ───────────────────────────────────────────────────

def _load_default_config():
    import yaml
    repo_root = Path(__file__).parents[1]
    with open(repo_root / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def test_default_config_assay_policy_is_single_cell_only():
    cfg = _load_default_config()
    assert cfg["data"]["assay_policy"] == "single_cell_only"


def test_default_single_cell_source_lists_contain_no_bulk_datasets():
    cfg = _load_default_config()
    assert cfg["data"]["microarray_sources"] == []
    for path, *_ in cfg["data"]["scrna_sources"]:
        assert "GSE994" not in path and "GSE123352" not in path and "GSE307690" not in path


def test_gse994_gse123352_canuck_moved_to_disabled_bulk_sources():
    cfg = _load_default_config()
    bulk = cfg["data"]["bulk_sources"]
    assert bulk["enabled"] is False
    names = [row[2] for row in bulk["datasets"]]
    assert set(names) == {"GSE994", "GSE123352", "CANUCK"}


# ─── Regression: the actual pre-issue-13 defect ─────────────────────────────

def test_pseudo_bulk_row_with_default_single_cell_assay_mode_is_rejected():
    """This is the exact shape of the bug: is_pseudo_bulk=True but
    assay_mode stays 'human_single_cell' (GSE994/GSE123352/GSE307690's
    real behaviour — see data/loaders.py::load_microarray). The pre-
    issue-13 assay_mode=='bulk_tcga'-only check let this through;
    row-provenance-based enforcement must not."""
    a = _mini_adata(is_pseudo_bulk=True)
    assert (a.obs["assay_mode"] == "human_single_cell").all()  # confirms the trap
    with pytest.raises(AssayPolicyError):
        assert_rows_match_policy(a.obs["is_pseudo_bulk"].values, ASSAY_POLICY_SINGLE_CELL_ONLY)


@pytest.mark.parametrize("accession", ["GSE994", "GSE123352", "CANUCK", "TCGA-LUAD"])
def test_preprocess_rejects_pseudo_bulk_source_regardless_of_accession_name(accession, tmp_path):
    """Renaming a bulk source cannot bypass the guard — the CSV's file name
    carries the accession, but rejection is driven by is_pseudo_bulk alone."""
    from preprocess import _load_all_sources

    n, g = 20, 50
    df = pd.DataFrame(
        np.random.negative_binomial(5, 0.6, (g, n)).astype("float32"),
        index=[f"G{i}" for i in range(g)],
        columns=[f"sample_{i}" for i in range(n)],
    )
    csv_path = tmp_path / f"{accession}.csv"
    df.to_csv(csv_path)

    cfg = {"microarray_sources": [(str(csv_path), "unknown")]}
    with pytest.raises(AssayPolicyError):
        _load_all_sources(cfg)


def test_renamed_bulk_source_still_rejected_even_under_innocuous_filename(tmp_path):
    """A pseudo-bulk CSV renamed to something with no accession-looking
    substring at all must still be rejected — enforcement never looks at
    the file name."""
    from preprocess import _load_all_sources

    n, g = 15, 40
    df = pd.DataFrame(
        np.random.negative_binomial(5, 0.6, (g, n)).astype("float32"),
        index=[f"G{i}" for i in range(g)], columns=[f"sample_{i}" for i in range(n)],
    )
    csv_path = tmp_path / "totally_innocuous_data.csv"
    df.to_csv(csv_path)
    cfg = {"microarray_sources": [(str(csv_path), "unknown")]}
    with pytest.raises(AssayPolicyError):
        _load_all_sources(cfg)


# ─── Row-level provenance failures ──────────────────────────────────────────

def test_missing_is_pseudo_bulk_column_defaults_to_real_cells_not_a_bypass():
    """A legacy fixture with no is_pseudo_bulk column at all is treated as
    real single cells (documented backward-compatible default), not
    silently rejected outright — but this must never be relied on to
    smuggle real bulk data through; it only applies when the column is
    truly absent."""
    from data.assembly import merge_sources

    a = _mini_adata(n=10)
    del a.obs["is_pseudo_bulk"]
    merged = merge_sources(a, scale=False)
    assert merged.n_obs == 10


def test_mixed_anndata_fails_before_preprocessing_fit():
    from data.preprocessing import fit_preprocessing

    real = _mini_adata(n=20, is_pseudo_bulk=False, subject_prefix="r")
    bulk = _mini_adata(n=5, is_pseudo_bulk=True, subject_prefix="b")
    mixed = ad.concat([real, bulk], axis=0, join="inner")
    mixed.obs["subject_id"] = mixed.obs["subject_id"].astype(str)
    train_subjects = set(mixed.obs["subject_id"])
    with pytest.raises(AssayPolicyError):
        fit_preprocessing(mixed, train_subjects, n_hvgs=20)


def test_mixed_anndata_fails_before_transform():
    from data.preprocessing import apply_preprocessing, fit_preprocessing

    real = _mini_adata(n=20, is_pseudo_bulk=False, subject_prefix="r")
    artifact = fit_preprocessing(real, set(real.obs["subject_id"]), n_hvgs=20)

    bulk = _mini_adata(n=5, is_pseudo_bulk=True, subject_prefix="b", g=200)
    bulk = bulk[:, real.var_names].copy()
    with pytest.raises(AssayPolicyError):
        apply_preprocessing(bulk, artifact)


def test_merge_sources_rejects_pseudo_bulk_under_default_policy():
    from data.assembly import merge_sources

    real = _mini_adata(n=10, is_pseudo_bulk=False)
    bulk = _mini_adata(n=10, is_pseudo_bulk=True, subject_prefix="b")
    with pytest.raises(AssayPolicyError):
        merge_sources(real, bulk, scale=False)


def test_export_cell_dataset_rejects_pseudo_bulk():
    from data.assembly import export_cell_dataset

    bulk = _mini_adata(n=10, is_pseudo_bulk=True)
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(AssayPolicyError):
            export_cell_dataset(bulk, tmp)


def test_assemble_subject_bags_rejects_pseudo_bulk():
    from data.assembly import assemble_subject_bags

    bulk = _mini_adata(n=60, is_pseudo_bulk=True, subject_prefix="b")
    with pytest.raises(AssayPolicyError):
        assemble_subject_bags(bulk, None, min_cells_per_subject=1)


# ─── CellLevelDataset / bag construction ────────────────────────────────────

def test_cell_level_dataset_rejects_pseudo_bulk_rows():
    from train import CellLevelDataset

    n, g = 12, 8
    with pytest.raises(AssayPolicyError):
        CellLevelDataset(
            gene_matrix=np.random.randn(n, g).astype("float32"),
            smoke_labels=np.random.randint(0, 6, n),
            malignancy_labels=np.zeros(n, dtype="float32"),
            cell_type_ids=np.zeros(n, dtype="int64"),
            subject_ids=[f"sub_{i}" for i in range(n)],
            is_pseudo_bulk=np.array([False] * (n - 1) + [True]),
        )


def test_cell_level_dataset_diagnostic_mode_bypasses_assay_check():
    """diagnostic_mode is the documented escape hatch for purely synthetic
    smoke tests — it must still bypass assay-policy validation exactly like
    it bypasses the subject_id checks, so existing synthetic workflows are
    unaffected."""
    from train import CellLevelDataset

    n, g = 6, 4
    ds = CellLevelDataset(
        gene_matrix=np.random.randn(n, g).astype("float32"),
        smoke_labels=np.random.randint(0, 6, n),
        malignancy_labels=np.zeros(n, dtype="float32"),
        cell_type_ids=np.zeros(n, dtype="int64"),
        is_pseudo_bulk=np.array([True] * n),
        diagnostic_mode=True,
    )
    assert len(ds) == n


def test_cell_level_dataset_default_is_pseudo_bulk_is_all_false_legacy():
    """Existing callers that never pass is_pseudo_bulk (the overwhelming
    majority of this codebase's existing synthetic tests) keep working
    unchanged — the default is documented as a legacy backward-compat
    default, not a validated safety claim."""
    from train import CellLevelDataset

    n, g = 6, 4
    ds = CellLevelDataset(
        gene_matrix=np.random.randn(n, g).astype("float32"),
        smoke_labels=np.random.randint(0, 6, n),
        malignancy_labels=np.zeros(n, dtype="float32"),
        cell_type_ids=np.zeros(n, dtype="int64"),
        subject_ids=[f"sub_{i}" for i in range(n)],
    )
    assert not ds.is_pseudo_bulk.any()


def test_subset_by_subjects_preserves_is_pseudo_bulk():
    from train import CellLevelDataset

    n, g = 8, 4
    ds = CellLevelDataset(
        gene_matrix=np.random.randn(n, g).astype("float32"),
        smoke_labels=np.random.randint(0, 6, n),
        malignancy_labels=np.zeros(n, dtype="float32"),
        cell_type_ids=np.zeros(n, dtype="int64"),
        subject_ids=[f"sub_{i % 4}" for i in range(n)],
        is_pseudo_bulk=np.zeros(n, dtype=bool),
    )
    sub = ds.subset_by_subjects(["sub_0", "sub_1"])
    assert len(sub) == 4
    assert sub.assay_policy == ds.assay_policy


# ─── Artifact provenance / fingerprint ──────────────────────────────────────

def test_fit_preprocessing_stamps_assay_provenance_fields():
    from data.preprocessing import fit_preprocessing

    real = _mini_adata(n=20)
    artifact = fit_preprocessing(real, set(real.obs["subject_id"]), n_hvgs=20)
    assert artifact.assay_policy == ASSAY_POLICY_SINGLE_CELL_ONLY
    assert artifact.assay_policy_version == ASSAY_POLICY_VERSION
    assert artifact.pseudo_bulk_rows_present_at_fit is False
    assert artifact.training_data_modality == "single_cell"
    assert artifact.allowed_inference_modality == "single_cell"


def test_artifact_round_trip_preserves_assay_policy(tmp_path):
    from data.preprocessing import PreprocessingArtifact, fit_preprocessing

    real = _mini_adata(n=20)
    artifact = fit_preprocessing(real, set(real.obs["subject_id"]), n_hvgs=20)
    path = tmp_path / "artifact.json"
    artifact.save(path)
    reloaded = PreprocessingArtifact.load(path)
    assert reloaded.assay_policy == artifact.assay_policy
    assert reloaded.scientific_fingerprint() == artifact.scientific_fingerprint()


def test_changing_assay_policy_changes_fingerprint():
    from data.preprocessing import fit_preprocessing

    real = _mini_adata(n=20)
    a1 = fit_preprocessing(real, set(real.obs["subject_id"]), n_hvgs=20,
                            assay_policy=ASSAY_POLICY_SINGLE_CELL_ONLY)
    a2_dict = a1.to_dict()
    a2_dict["assay_policy"] = "bulk_only"
    from data.preprocessing import PreprocessingArtifact
    a2 = PreprocessingArtifact(**a2_dict)
    assert a1.scientific_fingerprint() != a2.scientific_fingerprint()


def test_legacy_artifact_missing_assay_policy_rejected_for_real_context():
    from data.preprocessing import PreprocessingArtifact, assert_real_assay_provenance, CellTypeProvenanceError

    real = _mini_adata(n=20)
    from data.preprocessing import fit_preprocessing
    artifact = fit_preprocessing(real, set(real.obs["subject_id"]), n_hvgs=20)
    d = artifact.to_dict()
    d["assay_policy"] = None
    d["assay_policy_version"] = None
    legacy = PreprocessingArtifact(**d)
    with pytest.raises(CellTypeProvenanceError):
        assert_real_assay_provenance(legacy)


def test_bulk_artifact_rejects_single_cell_input_and_vice_versa():
    from data.preprocessing import apply_preprocessing, fit_preprocessing

    real = _mini_adata(n=20, is_pseudo_bulk=False)
    bulk = _mini_adata(n=20, is_pseudo_bulk=True, subject_prefix="b")

    real_artifact = fit_preprocessing(real, set(real.obs["subject_id"]), n_hvgs=20)
    with pytest.raises(AssayPolicyError):
        apply_preprocessing(bulk[:, real.var_names], real_artifact)

    bulk_artifact = fit_preprocessing(
        bulk, set(bulk.obs["subject_id"]), n_hvgs=20, assay_policy=ASSAY_POLICY_BULK_ONLY,
    )
    with pytest.raises(AssayPolicyError):
        apply_preprocessing(real[:, bulk.var_names], bulk_artifact)


# ─── Bulk-only mode ──────────────────────────────────────────────────────────

def test_bulk_only_mode_accepts_valid_bulk_and_rejects_real_cells():
    from data.preprocessing import fit_preprocessing

    bulk = _mini_adata(n=20, is_pseudo_bulk=True)
    artifact = fit_preprocessing(
        bulk, set(bulk.obs["subject_id"]), n_hvgs=20, assay_policy=ASSAY_POLICY_BULK_ONLY,
    )
    assert artifact.training_data_modality == "bulk"

    real = _mini_adata(n=20, is_pseudo_bulk=False)
    with pytest.raises(AssayPolicyError):
        fit_preprocessing(real, set(real.obs["subject_id"]), n_hvgs=20, assay_policy=ASSAY_POLICY_BULK_ONLY)


# ─── Model bundle binding ────────────────────────────────────────────────────

def test_bundle_records_and_validates_assay_policy(tmp_path):
    from benchmarks.bundle import load_and_validate_bundle, write_model_bundle
    from data.preprocessing import fit_preprocessing
    import torch

    real = _mini_adata(n=20)
    artifact = fit_preprocessing(real, set(real.obs["subject_id"]), n_hvgs=20)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": {}}, ckpt_path)

    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        bundle_dir, ckpt_path, artifact, model_config={}, class_vocabulary=["a", "b"],
        label_policy="verified_only", species_policy="human_only", assay_mode="human_single_cell",
    )
    manifest = load_and_validate_bundle(bundle_dir)
    assert manifest["assay_policy"] == ASSAY_POLICY_SINGLE_CELL_ONLY


def test_bundle_input_modality_mismatch_rejected(tmp_path):
    from benchmarks.bundle import (
        BundleValidationError, load_and_validate_bundle, validate_bundle_input_modality, write_model_bundle,
    )
    from data.preprocessing import fit_preprocessing
    import torch

    real = _mini_adata(n=20)
    artifact = fit_preprocessing(real, set(real.obs["subject_id"]), n_hvgs=20)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": {}}, ckpt_path)
    bundle_dir = tmp_path / "bundle"
    write_model_bundle(
        bundle_dir, ckpt_path, artifact, model_config={}, class_vocabulary=["a", "b"],
        label_policy="verified_only", species_policy="human_only", assay_mode="human_single_cell",
    )
    manifest = load_and_validate_bundle(bundle_dir)
    with pytest.raises(BundleValidationError):
        validate_bundle_input_modality(manifest, [False, True, False])


# ─── Frozen-test guard non-interference ─────────────────────────────────────

def test_assay_policy_enforcement_does_not_touch_frozen_test_guard_module():
    """Sanity check that this change never imported/modified test_guard.py
    — assay-policy validation happens strictly upstream of guard
    acquisition (source loading / preprocessing / dataset construction),
    never inside the guard's own state machine."""
    import benchmarks.test_guard as test_guard_mod
    assert "assay_policy" not in Path(test_guard_mod.__file__).read_text()
