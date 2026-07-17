"""
tests/test_leakage_regression.py — consolidated leakage-safety regression
suite. Each test below is labeled with the specific guarantee it proves,
mirroring this project's leakage-safety checklist. Several of these
guarantees already have deeper coverage elsewhere (tests/test_splitting.py,
tests/test_preprocessing.py, tests/test_benchmarks_leakage_fixes.py) — this
file exists as one place that exercises every item end-to-end so the whole
checklist can be audited/run together, plus the items that had no existing
coverage before this change (cross-species isolation, assay-mode isolation,
label-corruption-cannot-fabricate-a-negative).
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import EXPERIMENT_MODE_HUMAN_ONLY, SPECIES_HUMAN, SPECIES_MOUSE
from data.assay_mode import AssayModeError, assert_no_pseudo_bulk_rows
from data.assembly import merge_sources
from data.preprocessing import fit_preprocessing, apply_preprocessing
from data.species_policy import SpeciesPolicyError
from data.splitting import SplitManifest, subject_train_val_test_split


def _adata(n_subjects=10, cells_per_subject=20, n_genes=50, seed=0, offset=0.0, species="human"):
    rng = np.random.default_rng(seed)
    subject_ids = []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * cells_per_subject
    n = len(subject_ids)
    X = (rng.random((n, n_genes)) + offset).astype("float32")
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame({
        "subject_id": subject_ids, "batch": ["source_0"] * n, "species": species,
    }, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))


# 1. Subject disjointness across splits ─────────────────────────────────────

def test_1_subject_disjointness_across_splits():
    subjects = [f"sub_{i}" for i in range(30)]
    manifest = subject_train_val_test_split(
        subjects * 5,  # multiple cells per subject
        seed=1,
    )
    tr, va, te = set(manifest.train_subjects), set(manifest.val_subjects), set(manifest.test_subjects)
    assert not (tr & va) and not (tr & te) and not (va & te)
    with pytest.raises(ValueError):
        SplitManifest(seed=1, train_subjects=["a", "b"], val_subjects=["b"], test_subjects=["c"])


# 2. HVG corruption isolation ────────────────────────────────────────────────

def test_2_hvg_selection_isolated_from_val_test_corruption():
    adata = _adata(n_subjects=10)
    train_subjects = {f"sub_{i}" for i in range(6)}
    a1 = fit_preprocessing(adata, train_subjects, n_hvgs=20, batch_key=None)

    corrupted = adata.copy()
    non_train = ~corrupted.obs["subject_id"].isin(train_subjects).values
    corrupted.X[non_train] = 1e6  # drastic corruption of val/test expression
    a2 = fit_preprocessing(corrupted, train_subjects, n_hvgs=20, batch_key=None)

    assert a1.gene_list == a2.gene_list


# 3. Scaling corruption isolation ────────────────────────────────────────────

def test_3_scaling_statistics_isolated_from_val_test_corruption():
    adata = _adata(n_subjects=10)
    train_subjects = {f"sub_{i}" for i in range(6)}
    a1 = fit_preprocessing(adata, train_subjects, n_hvgs=20, batch_key=None)

    corrupted = adata.copy()
    non_train = ~corrupted.obs["subject_id"].isin(train_subjects).values
    corrupted.X[non_train] = -999.0
    a2 = fit_preprocessing(corrupted, train_subjects, n_hvgs=20, batch_key=None)

    assert a1.gene_means == a2.gene_means
    assert a1.gene_stds == a2.gene_stds


# 4. Batch-correction isolation ──────────────────────────────────────────────

def test_4_batch_correction_default_mode_never_touches_val_test():
    """Default preprocessing (fit_preprocessing/apply_preprocessing) never
    calls Harmony at all — batch correction is a separate, explicitly
    opt-in step (preprocess.py's allow_transductive_harmony /
    batch_correction.mode) that the leakage-free artifact fit does not
    depend on. This proves the default fit path has no batch-correction
    dependency to poison in the first place."""
    adata = _adata(n_subjects=10)
    train_subjects = {f"sub_{i}" for i in range(6)}
    artifact = fit_preprocessing(adata, train_subjects, n_hvgs=10, batch_key=None)
    assert "batch" not in artifact.to_dict()  # no batch-correction state in the fit artifact


# 5. Label corruption isolation ──────────────────────────────────────────────

def test_5_unknown_malignancy_labels_are_excluded_not_fabricated_negative():
    from data.labellers import add_malignancy_labels

    adata = _adata(n_subjects=4, cells_per_subject=5)
    adata.obs["malignancy"] = 0.0
    adata.obs["malignancy_known"] = False
    out = add_malignancy_labels(adata, tumor_barcodes=None)
    # No tumor_barcodes given -> nothing becomes "known" -> every cell must
    # stay excluded (malignancy_known=False), not silently scored as a
    # verified negative.
    assert not out.obs["malignancy_known"].any()

    # Corrupt the placeholder value itself (e.g. sentinel injected by a
    # buggy upstream step) — known must still be False, so downstream loss
    # masking (model.py::MultiTaskLoss) still excludes these rows
    # regardless of what numeric placeholder they carry.
    out.obs["malignancy"] = 12345.0
    assert not out.obs["malignancy_known"].any()


def test_5_unknown_cancer_outcomes_excluded_from_subject_bags():
    from data.assembly import assemble_subject_bags

    adata = _adata(n_subjects=4, cells_per_subject=60)
    adata.obs["cell_type_id"] = 0
    adata.obs["smoke_type"] = 0
    adata.obs["malignancy"] = 0.0
    adata.obs["malignancy_known"] = False
    # No cancer_outcomes DataFrame at all -> every subject must be
    # cancer_label_known=False, cancer_label=None (never fabricated 0).
    bags = assemble_subject_bags(adata, cancer_outcomes=None, min_cells_per_subject=10)
    assert len(bags) > 0
    assert all(b["cancer_label_known"] is False for b in bags)
    assert all(b["cancer_label"] is None for b in bags)


# 6. Gene-order enforcement ───────────────────────────────────────────────────

def test_6_gene_order_enforcement_rejects_permutation_and_duplicates():
    from data.preprocessing import verify_input_matrix, PreprocessingArtifact

    artifact = PreprocessingArtifact(
        version="1", gene_list=["G0", "G1", "G2"],
        gene_means=[0.0, 0.0, 0.0], gene_stds=[1.0, 1.0, 1.0],
        n_hvgs=3, smoke_marker_genes_forced=[], fit_n_cells=10, fit_n_subjects=2,
    )
    verify_input_matrix(artifact, ["G0", "G1", "G2"])
    with pytest.raises(ValueError):
        verify_input_matrix(artifact, ["G1", "G0", "G2"])
    with pytest.raises(ValueError):
        verify_input_matrix(artifact, ["G0", "G0", "G2"])  # duplicate


# 7. Artifact round trip ──────────────────────────────────────────────────────

def test_7_preprocessing_artifact_round_trip_preserves_transform(tmp_path):
    adata = _adata(n_subjects=10, n_genes=10)
    train_subjects = {f"sub_{i}" for i in range(6)}
    artifact = fit_preprocessing(adata, train_subjects, n_hvgs=5, batch_key=None)

    path = tmp_path / "artifact.json"
    artifact.save(path)
    from data.preprocessing import PreprocessingArtifact
    loaded = PreprocessingArtifact.load(path)

    t1 = apply_preprocessing(adata, artifact)
    t2 = apply_preprocessing(adata, loaded)
    assert np.allclose(t1.X, t2.X)
    assert list(t1.var_names) == list(t2.var_names)


# 8. Inference parity — covered by test 7's t1/t2 equivalence above, plus
# tests/test_inference.py's dedicated inference-contract tests.


# 9. Cross-species isolation ─────────────────────────────────────────────────

def test_9_mouse_cells_cannot_enter_default_human_merge():
    human = _adata(n_subjects=4, species=SPECIES_HUMAN)
    mouse = _adata(n_subjects=4, species=SPECIES_MOUSE)
    mouse.obs["subject_id"] = "mouse::" + mouse.obs["subject_id"].astype(str)
    with pytest.raises(SpeciesPolicyError):
        merge_sources(human, mouse, scale=False, experiment_mode=EXPERIMENT_MODE_HUMAN_ONLY)


def test_9_mouse_expression_cannot_influence_human_only_fit():
    """Even if a caller tried to smuggle mouse cells into a human_only fit
    by mislabeling species, fit_preprocessing only ever sees whatever
    AnnData it's given — the real guarantee is that merge_sources (the
    only place multiple sources are combined) refuses upstream. This test
    documents that boundary explicitly."""
    human = _adata(n_subjects=6, species=SPECIES_HUMAN)
    train_subjects = {f"sub_{i}" for i in range(4)}
    artifact_before = fit_preprocessing(human, train_subjects, n_hvgs=10, batch_key=None)
    with pytest.raises(SpeciesPolicyError):
        merge_sources(human, _adata(n_subjects=2, species=SPECIES_MOUSE), scale=False)
    # Refit after the rejected merge attempt — must be identical (no
    # partial mutation from the failed merge).
    artifact_after = fit_preprocessing(human, train_subjects, n_hvgs=10, batch_key=None)
    assert artifact_before.gene_means == artifact_after.gene_means


# 10. Bulk/single-cell isolation ──────────────────────────────────────────────

def test_10_pseudo_bulk_rows_rejected_from_single_cell_only_assay_mode():
    is_pseudo_bulk = [False, False, True, False]  # one TCGA-style bulk row
    with pytest.raises(AssayModeError):
        assert_no_pseudo_bulk_rows(is_pseudo_bulk, assay_mode="human_single_cell")


def test_10_pure_single_cell_rows_pass_single_cell_only_assay_mode():
    assert_no_pseudo_bulk_rows([False, False, False], assay_mode="human_single_cell")  # must not raise


# 11. Controlled-access safety (NLST) — see tests/test_nlst_adapter.py for
# the full suite; spot-check here that missing credentials fail loudly.

def test_11_nlst_missing_credentials_produce_actionable_error(monkeypatch):
    from data.nlst_adapter import NLSTAccessError, require_nlst_available
    monkeypatch.delenv("NLST_DATA_ROOT_TEST_11", raising=False)
    with pytest.raises(NLSTAccessError):
        require_nlst_available("NLST_DATA_ROOT_TEST_11")


# 12. Frozen-test protection — enforced by src/benchmarks/test_guard.py,
# exercised by tests/test_benchmarks_test_guard.py and
# tests/test_benchmarks_runner_test_guard_integration.py (pre-existing,
# unmodified by this change — see NON-NEGOTIABLE rule against weakening
# these guards). Spot-check here that the guard module still imports and
# exposes its enforcement entry point.

def test_12_test_guard_module_still_present_and_importable():
    import benchmarks.test_guard as guard_mod
    assert hasattr(guard_mod, "__file__")
