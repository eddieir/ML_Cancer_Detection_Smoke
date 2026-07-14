"""preprocess.py::run_pipeline_split_aware — leakage-free end-to-end integration."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import ALL_SMOKE_MARKERS, N_SMOKE_CLASSES


def _synthetic_h5ad_consistent_labels(path, n_subjects=12, cells_per_subject=30, g=500, n_classes=3):
    """Unlike test_pipeline.py's generator, every cell of a given subject
    shares the SAME smoke_type — required for subject-level stratified
    splitting, and realistic (a donor has one smoking status, not a random
    one per cell)."""
    genes = [f"G{i}" for i in range(g)]
    genes[:5] = [f"MT-{i}" for i in range(5)]
    for i, mk in enumerate(ALL_SMOKE_MARKERS[:4]):
        genes[50 + i] = mk

    subject_ids, smoke_types = [], []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * cells_per_subject
        smoke_types  += [i % n_classes] * cells_per_subject
    n = len(subject_ids)

    raw = np.random.negative_binomial(5, 0.7, (n, g)).astype("float32")
    obs = pd.DataFrame({
        "donor_id":        subject_ids,
        "subject_id":      subject_ids,
        "smoke_type":      smoke_types,
        "smoke_type_name": "mixed",
        "data_modality":   "scrna",
        "is_pseudo_bulk":  False,
        "malignancy":      0.0,
        "cell_type_id":    0,
    }, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=sp.csr_matrix(raw), obs=obs, var=pd.DataFrame(index=genes))
    adata.write_h5ad(path)


def test_run_pipeline_split_aware_produces_split_and_artifact():
    from preprocess import run_pipeline_split_aware
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad_consistent_labels(h5ad)
        result = run_pipeline_split_aware({
            "data": {
                "scrna_sources": [(h5ad, "cigarette", "donor_id")],
                "n_hvgs": 50,
                "min_cells_per_subject": 5,
                "out_dir": str(Path(tmp) / "processed"),
                # See tests/test_pipeline.py for why this disclosed,
                # always-degraded diagnostic override is used: this
                # environment's CellTypist model is version-incompatible
                # with the installed scikit-learn, and this test only
                # exercises split-aware pipeline mechanics on synthetic
                # data, not CellTypist compatibility itself.
                "cell_type_allow_diagnostic_fallback": True,
            },
            "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
        })
        assert "cell_data" in result
        assert "bags" in result
        assert "split_manifest" in result
        assert "preprocessing_artifact" in result

        manifest = result["split_manifest"]
        all_subjects = manifest.train_subjects + manifest.val_subjects + manifest.test_subjects
        assert len(all_subjects) == len(set(all_subjects))  # no leakage

        artifact = result["preprocessing_artifact"]
        assert artifact.fit_n_subjects == len(manifest.train_subjects)
        assert len(artifact.gene_list) <= 50


def test_run_pipeline_split_aware_test_cells_do_not_affect_gene_scaling():
    """The core leakage guarantee, exercised end-to-end: refitting after
    corrupting the val/test cells' raw values must not change which genes
    were selected or their scaling statistics."""
    from preprocess import run_pipeline_split_aware
    import anndata as ad_module

    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad_consistent_labels(h5ad)

        cfg = {
            "data": {
                "scrna_sources": [(h5ad, "cigarette", "donor_id")],
                "n_hvgs": 50,
                "min_cells_per_subject": 5,
                "out_dir": str(Path(tmp) / "processed"),
                # See tests/test_pipeline.py for why this disclosed,
                # always-degraded diagnostic override is used: this
                # environment's CellTypist model is version-incompatible
                # with the installed scikit-learn, and this test only
                # exercises split-aware pipeline mechanics on synthetic
                # data, not CellTypist compatibility itself.
                "cell_type_allow_diagnostic_fallback": True,
            },
            "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
        }
        result1 = run_pipeline_split_aware(cfg)
        artifact1 = result1["preprocessing_artifact"]

        # Corrupt the source file's expression matrix, but only for
        # subjects NOT in the training split, then re-run.
        adata = ad_module.read_h5ad(h5ad)
        non_train = ~adata.obs["donor_id"].isin(result1["split_manifest"].train_subjects).values
        adata.X = adata.X.toarray()
        adata.X[non_train] = adata.X[non_train] * 1000 + 5000
        h5ad2 = str(Path(tmp) / "test_corrupted.h5ad")
        adata.write_h5ad(h5ad2)

        cfg2 = dict(cfg)
        cfg2["data"] = dict(cfg["data"])
        cfg2["data"]["scrna_sources"] = [(h5ad2, "cigarette", "donor_id")]
        cfg2["data"]["out_dir"] = str(Path(tmp) / "processed2")
        cfg2["split"] = dict(cfg["split"])
        cfg2["split"]["manifest_path"] = None
        result2 = run_pipeline_split_aware(cfg2)
        artifact2 = result2["preprocessing_artifact"]

        assert artifact1.gene_means == artifact2.gene_means
        assert artifact1.gene_stds == artifact2.gene_stds


def test_run_pipeline_split_aware_returns_explicit_disjoint_per_split_datasets():
    """Section 5: explicit train/val/test cell datasets and bags, not just
    one combined array the caller has to filter themselves."""
    from preprocess import run_pipeline_split_aware
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad_consistent_labels(h5ad, n_subjects=15, cells_per_subject=20)
        result = run_pipeline_split_aware({
            "data": {
                "scrna_sources": [(h5ad, "cigarette", "donor_id")],
                "n_hvgs": 50,
                "min_cells_per_subject": 5,
                "out_dir": str(Path(tmp) / "processed"),
                # See tests/test_pipeline.py for why this disclosed,
                # always-degraded diagnostic override is used: this
                # environment's CellTypist model is version-incompatible
                # with the installed scikit-learn, and this test only
                # exercises split-aware pipeline mechanics on synthetic
                # data, not CellTypist compatibility itself.
                "cell_type_allow_diagnostic_fallback": True,
            },
            "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
        })
        for key in ("train_cell_dataset", "val_cell_dataset", "test_cell_dataset",
                    "train_bags", "val_bags", "test_bags",
                    "rare_class_report", "label_provenance_report",
                    "transductive_batch_correction"):
            assert key in result

        manifest = result["split_manifest"]
        train_ds, val_ds, test_ds = (
            result["train_cell_dataset"], result["val_cell_dataset"], result["test_cell_dataset"]
        )
        assert set(train_ds.subject_ids.tolist()) <= set(manifest.train_subjects)
        assert set(val_ds.subject_ids.tolist())   <= set(manifest.val_subjects)
        assert set(test_ds.subject_ids.tolist())  <= set(manifest.test_subjects)
        assert not (set(train_ds.subject_ids.tolist()) & set(val_ds.subject_ids.tolist()))
        assert not (set(train_ds.subject_ids.tolist()) & set(test_ds.subject_ids.tolist()))
        assert len(train_ds) + len(val_ds) + len(test_ds) == len(result["cell_data"]["gene_matrix"])

        train_bag_subjects = {b["subject_id"] for b in result["train_bags"]}
        test_bag_subjects  = {b["subject_id"] for b in result["test_bags"]}
        assert not (train_bag_subjects & test_bag_subjects)

        assert result["transductive_batch_correction"] is False  # strict mode is the default


def test_run_pipeline_split_aware_batch_correction_skipped_by_default():
    from preprocess import run_pipeline_split_aware
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad_consistent_labels(h5ad)
        result = run_pipeline_split_aware({
            "data": {
                "scrna_sources": [(h5ad, "cigarette", "donor_id")],
                "n_hvgs": 50,
                "min_cells_per_subject": 5,
                "out_dir": str(Path(tmp) / "processed"),
                # See tests/test_pipeline.py for why this disclosed,
                # always-degraded diagnostic override is used: this
                # environment's CellTypist model is version-incompatible
                # with the installed scikit-learn, and this test only
                # exercises split-aware pipeline mechanics on synthetic
                # data, not CellTypist compatibility itself.
                "cell_type_allow_diagnostic_fallback": True,
            },
            "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
            # preprocessing.batch_correction.allow_transductive_harmony omitted -> defaults False
        })
        assert result["transductive_batch_correction"] is False


def test_rare_class_policy_from_config_changes_effective_smoke_labels():
    """Section 7: the YAML-configured rare-class policy must actually
    reassign the class used for splitting/training, not just be available
    as an unused utility."""
    from preprocess import run_pipeline_split_aware
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        # 19 subjects of class 0, 1 subject of class 1 (a "rare" class).
        _synthetic_h5ad_consistent_labels(h5ad, n_subjects=20, cells_per_subject=10, n_classes=1)
        import anndata as ad_module
        adata = ad_module.read_h5ad(h5ad)
        # loaders.py::_attach_standard_obs derives the numeric smoke_type from
        # smoke_type_name (not the raw numeric column) whenever that column is
        # present, so the rare label has to be set via the name, not the id.
        adata.obs["smoke_type_name"] = adata.obs["smoke_type_name"].astype(str)
        rare_mask = adata.obs["donor_id"] == "sub_0"
        adata.obs.loc[rare_mask, "smoke_type_name"] = "cigar"
        adata.write_h5ad(h5ad)

        cfg = {
            "data": {
                "scrna_sources": [(h5ad, "cigarette", "donor_id")],
                "n_hvgs": 50,
                "min_cells_per_subject": 5,
                "out_dir": str(Path(tmp) / "processed"),
                # See tests/test_pipeline.py for why this disclosed,
                # always-degraded diagnostic override is used: this
                # environment's CellTypist model is version-incompatible
                # with the installed scikit-learn, and this test only
                # exercises split-aware pipeline mechanics on synthetic
                # data, not CellTypist compatibility itself.
                "cell_type_allow_diagnostic_fallback": True,
            },
            "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
            "rare_class": {
                "policy": "merge_into_dual_use_or_other",
                "min_subjects_required": 3,
                "target_classes": ["cigar"],
            },
        }
        result = run_pipeline_split_aware(cfg)
        assert result["rare_class_report"]["affected_classes"]["cigar"]["action"] == "merged_into_dual_use"

        # The effective label actually used downstream (exported cell_metadata,
        # bags, split) must reflect the merge — no cell should carry cigar (2)
        # anymore, and smoke_type_raw must still show the original cigar label.
        smoke_type = result["cell_data"]["smoke_labels"]
        assert 2 not in smoke_type
        import pandas as pd
        meta = pd.read_csv(Path(tmp) / "processed" / "cell_metadata.csv")
        assert (meta["smoke_type_raw"] == 2).sum() > 0
        assert (meta["smoke_type"] == 2).sum() == 0


def test_rare_class_merge_produces_contiguous_effective_labels_matching_model_k():
    """Section 1 (effective smoke-class mapping): a merged-away raw class
    must shrink the model's actual output space to a deterministic
    contiguous K, not just be absent from the data while 6 output neurons
    remain."""
    from preprocess import run_pipeline_split_aware
    from model import MultiSmokeCancerNet
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        _synthetic_h5ad_consistent_labels(h5ad, n_subjects=20, cells_per_subject=10, n_classes=1)
        adata = ad.read_h5ad(h5ad)
        adata.obs["smoke_type_name"] = adata.obs["smoke_type_name"].astype(str)
        rare_mask = adata.obs["donor_id"] == "sub_0"
        adata.obs.loc[rare_mask, "smoke_type_name"] = "cigar"
        adata.write_h5ad(h5ad)

        cfg = {
            "data": {
                "scrna_sources": [(h5ad, "cigarette", "donor_id")],
                "n_hvgs": 50,
                "min_cells_per_subject": 5,
                "out_dir": str(Path(tmp) / "processed"),
                # See tests/test_pipeline.py for why this disclosed,
                # always-degraded diagnostic override is used: this
                # environment's CellTypist model is version-incompatible
                # with the installed scikit-learn, and this test only
                # exercises split-aware pipeline mechanics on synthetic
                # data, not CellTypist compatibility itself.
                "cell_type_allow_diagnostic_fallback": True,
            },
            "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
            "rare_class": {
                "policy": "merge_into_dual_use_or_other",
                "min_subjects_required": 3,
                "target_classes": ["cigar"],
            },
        }
        result = run_pipeline_split_aware(cfg)
        mapping = result["label_mapping"]

        # Contiguous 0..K-1, K=5 (6 raw classes minus merged-away cigar).
        assert mapping.k == 5
        assert sorted(mapping.effective_id_to_name) == list(range(5))
        assert "cigar" not in mapping.class_names

        # The actual exported labels are within [0, K) — no dead id beyond K-1.
        smoke_labels = result["cell_data"]["smoke_labels"]
        assert smoke_labels.max() < mapping.k
        assert smoke_labels.min() >= 0

        # Persisted identically in the preprocessing artifact.
        assert result["preprocessing_artifact"].label_mapping == mapping.to_dict()

        # A model built with num_smoke_types=K actually has that output width
        # — no unused/dead output neuron for the merged-away class.
        model = MultiSmokeCancerNet(input_dim=50, embedding_dim=16, attention_dim=8,
                                     num_smoke=mapping.k)
        assert model.num_smoke == 5
        import torch
        _, logits, _ = model.forward_cell(torch.randn(4, 50))
        assert logits.shape == (4, 5)


def test_label_transfer_happens_before_split_changes_effective_class():
    """Section 6 regression test: NLST label transfer must be applied
    BEFORE the subject-level split is computed, so the split (and its
    report) reflects the FINAL label, not the pre-transfer one."""
    from preprocess import run_pipeline_split_aware
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = str(Path(tmp) / "test.h5ad")
        # All subjects start as class 0 (cigarette); NLST will relabel some to
        # class 2 (cigar-only) or 5 (unexposed) via transfer_nlst_labels.
        _synthetic_h5ad_consistent_labels(h5ad, n_subjects=12, cells_per_subject=20, n_classes=1)

        nlst_csv = Path(tmp) / "nlst_screen.csv"
        pd.DataFrame({
            "pid": [f"sub_{i}" for i in range(6)],
            "CIGAR":   [1, 1, 1, 0, 0, 0],
            "CIGSMOK": [0, 0, 0, 0, 0, 0],
        }).to_csv(nlst_csv, index=False)

        cfg = {
            "data": {
                "scrna_sources": [(h5ad, "cigarette", "donor_id")],
                "n_hvgs": 50,
                "min_cells_per_subject": 5,
                "out_dir": str(Path(tmp) / "processed"),
                # See tests/test_pipeline.py for why this disclosed,
                # always-degraded diagnostic override is used: this
                # environment's CellTypist model is version-incompatible
                # with the installed scikit-learn, and this test only
                # exercises split-aware pipeline mechanics on synthetic
                # data, not CellTypist compatibility itself.
                "cell_type_allow_diagnostic_fallback": True,
                "nlst_csv": str(nlst_csv),
            },
            "split": {"seed": 1, "train_frac": 0.5, "val_frac": 0.25, "test_frac": 0.25},
        }
        result = run_pipeline_split_aware(cfg)
        assert result["label_provenance_report"]["nlst_csv_used"] is True
        assert result["label_provenance_report"]["n_subjects_matched"] == 6

        # subjects sub_0..sub_2 (CIGAR=1) must have been relabelled to class 2
        # (cigar) by the time the split report / effective label was computed.
        manifest = result["split_manifest"]
        report = manifest.report["splits"]
        # class "2" (cigar, post-transfer) must appear SOMEWHERE in the split
        # report's class_distribution — impossible if the split had been
        # computed on the pre-transfer (all-class-0) label.
        all_classes = set()
        for split_report in report.values():
            all_classes |= set(split_report["class_distribution"].keys())
        assert "2" in all_classes
