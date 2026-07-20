"""
benchmarks/context.py — ExperimentContext: the single object every benchmark
task, baseline, and report is built from.

Constructed once from run_pipeline_split_aware()'s result (src/preprocess.py)
so every model compared in this framework — baselines and MultiSmokeCancerNet
alike — sees the exact same subjects, features, and labels. Building it here
rather than passing the raw pipeline dict around means a benchmark can never
accidentally read `bags` (the whole, unsplit dataset) when it meant
`train_bags`.
"""

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data.label_mapping import EffectiveLabelMapping
from data.preprocessing import PreprocessingArtifact, validate_cell_type_provenance, assert_real_assay_provenance
from data.splitting import SplitManifest
from train import validate_experiment_partitions, SubjectLevelDataset


def get_git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _dataset_source_summary(cell_dataset) -> Dict[str, int]:
    if cell_dataset is None or len(cell_dataset) == 0:
        return {}
    sources, counts = np.unique(cell_dataset.dataset_source, return_counts=True)
    return {str(s): int(c) for s, c in zip(sources, counts)}


def _validate_context(
    train_cell_dataset, val_cell_dataset, test_cell_dataset,
    train_bags, val_bags, test_bags,
    preprocessing_artifact: PreprocessingArtifact, label_mapping: EffectiveLabelMapping,
    split_manifest: SplitManifest,
) -> None:
    """
    Fail loudly, before returning a usable context, on any of the mismatches
    that would otherwise surface later as a silent wrong-shape model, a
    mislabeled prediction, or a leakage bug discovered only by inspection.
    """
    # 1. cross-task disjointness (cell vs cell, bag vs bag, cell vs bag,
    # across all three splits at once) — reuses train.py's own leakage guard
    # rather than re-implementing a second, possibly-inconsistent version.
    validate_experiment_partitions(
        train_cell_dataset=train_cell_dataset, val_cell_dataset=val_cell_dataset,
        test_cell_dataset=test_cell_dataset,
        train_subject_dataset=SubjectLevelDataset(train_bags, require_known_outcome=False),
        val_subject_dataset=SubjectLevelDataset(val_bags, require_known_outcome=False),
        test_subject_dataset=SubjectLevelDataset(test_bags, require_known_outcome=False),
    )

    # 2. dataset subject sets match the split manifest EXACTLY — reject both
    # unexpected subjects (in the dataset but not the manifest: a leakage-
    # relevant assembly bug) and missing subjects (in the manifest but absent
    # from the dataset: a subject silently dropped somewhere between the
    # split and the exported dataset, which would otherwise surface only as
    # an unexplained smaller train/val/test set).
    for name, ds, manifest_subjects in (
        ("train", train_cell_dataset, split_manifest.train_subjects),
        ("val", val_cell_dataset, split_manifest.val_subjects),
        ("test", test_cell_dataset, split_manifest.test_subjects),
    ):
        if ds is None or getattr(ds, "diagnostic_mode", False):
            continue
        ds_subjects = set(ds.subject_ids.tolist())
        manifest_set = {str(s) for s in manifest_subjects}
        unexpected = ds_subjects - manifest_set
        if unexpected:
            raise ValueError(
                f"ExperimentContext: {name}_cell_dataset contains subject(s) "
                f"{sorted(unexpected)[:5]} not present in split_manifest.{name}_subjects."
            )
        missing = manifest_set - ds_subjects
        if missing:
            raise ValueError(
                f"ExperimentContext: split_manifest.{name}_subjects declares subject(s) "
                f"{sorted(missing)[:5]} that are absent from {name}_cell_dataset — a subject "
                "was silently dropped somewhere between the split and the exported dataset."
            )

    # 2b. no blank/placeholder subject IDs in bags (CellLevelDataset already
    # enforces this for cell datasets at construction time — see train.py —
    # but bags are plain dicts with no equivalent constructor-time guard).
    _PLACEHOLDER_IDS = {"", "unknown", "none", "None", "nan", "NaN"}
    for name, bags in (("train_bags", train_bags), ("val_bags", val_bags), ("test_bags", test_bags)):
        for b in bags:
            sid = str(b.get("subject_id", ""))
            if sid.strip() in _PLACEHOLDER_IDS:
                raise ValueError(
                    f"ExperimentContext: {name} contains a blank/placeholder subject_id "
                    f"({sid!r}) — every bag must carry a real subject identifier."
                )

    # 3. gene count consistency: artifact vs every cell/bag matrix width
    n_genes = len(preprocessing_artifact.gene_list)
    for name, ds in (("train_cell_dataset", train_cell_dataset), ("val_cell_dataset", val_cell_dataset),
                     ("test_cell_dataset", test_cell_dataset)):
        if ds is not None and len(ds) > 0 and ds.X.shape[1] != n_genes:
            raise ValueError(f"ExperimentContext: {name} has {ds.X.shape[1]} genes, "
                              f"preprocessing_artifact has {n_genes}.")
    for name, bags in (("train_bags", train_bags), ("val_bags", val_bags), ("test_bags", test_bags)):
        for b in bags:
            if b["gene_matrix"].shape[1] != n_genes:
                raise ValueError(
                    f"ExperimentContext: {name} subject {b['subject_id']} bag has "
                    f"{b['gene_matrix'].shape[1]} genes, preprocessing_artifact has {n_genes}."
                )

    # 3b. strict cell-type annotation provenance for a real pipeline result
    # — this check only runs here, inside from_pipeline_result, never for a
    # hand-built synthetic context (see build_synthetic_context in
    # runner.py, which constructs ExperimentContext directly and never
    # calls this function). Unlike the historical
    # `getattr(..., "cell_type_annotation_degraded", False)` check this
    # replaced, a MISSING provenance field is never treated as safe — see
    # data.preprocessing.validate_cell_type_provenance for the full policy
    # (accepted modes, fingerprint matching, the explicit pseudo-bulk
    # exemption).
    validate_cell_type_provenance(preprocessing_artifact)

    # 3c. strict assay-policy provenance for a real pipeline result — same
    # fail-closed pattern as validate_cell_type_provenance above (missing
    # is never treated as safe). See data/preprocessing.py::
    # assert_real_assay_provenance / data/assay_policy.py.
    assert_real_assay_provenance(preprocessing_artifact)

    # 3d. the assay policy this artifact was fit under must actually be
    # trainable — this is the single centralized enforcement point for
    # every downstream consumer of a real ExperimentContext (Trainer, CV,
    # OOF, final development fit, source-held-out fitting, domain-
    # robustness ablation all build on a context constructed here). A
    # bulk_only/multimodal artifact must never reach the point where a
    # model/optimizer could be constructed against it.
    from data.assay_policy import require_trainable as _require_assay_trainable
    _require_assay_trainable(preprocessing_artifact.assay_policy)

    # 4. artifact's embedded label mapping matches the context's label_mapping
    if preprocessing_artifact.label_mapping is not None:
        artifact_mapping = EffectiveLabelMapping.from_dict(preprocessing_artifact.label_mapping)
        artifact_mapping.validate_compatible(label_mapping, "preprocessing_artifact", "context.label_mapping")

    # 5. effective smoke labels within [0, K)
    k = label_mapping.k
    for name, ds in (("train_cell_dataset", train_cell_dataset), ("val_cell_dataset", val_cell_dataset),
                     ("test_cell_dataset", test_cell_dataset)):
        if ds is not None and len(ds) > 0:
            bad = ds.smoke[(ds.smoke < 0) | (ds.smoke >= k)]
            if len(bad) > 0:
                raise ValueError(f"ExperimentContext: {name} has smoke label(s) outside [0, {k}): "
                                  f"{sorted(set(bad.tolist()))[:5]}")

    # 6. no non-finite expression values
    for name, ds in (("train_cell_dataset", train_cell_dataset), ("val_cell_dataset", val_cell_dataset),
                     ("test_cell_dataset", test_cell_dataset)):
        if ds is not None and len(ds) > 0 and not torch_isfinite_all(ds.X):
            raise ValueError(f"ExperimentContext: {name} contains non-finite (NaN/Inf) expression values.")


def torch_isfinite_all(tensor) -> bool:
    import torch
    return bool(torch.isfinite(tensor).all())


@dataclass
class ExperimentContext:
    train_cell_dataset: "object"
    val_cell_dataset:   "object"
    test_cell_dataset:  "object"
    train_bags: List[dict]
    val_bags:   List[dict]
    test_bags:  List[dict]
    split_manifest:         SplitManifest
    preprocessing_artifact: PreprocessingArtifact
    label_mapping:          EffectiveLabelMapping
    rare_class_report:      dict
    label_provenance_report: dict
    transductive_batch_correction: bool
    config:  dict
    seed:    int
    dataset_source_summary: Dict[str, Dict[str, int]] = field(default_factory=dict)
    git_sha: Optional[str] = None
    # Full-gene, normalized-but-not-yet-HVG-selected-or-scaled AnnData (see
    # preprocess.py::run_pipeline_split_aware). Required for any grouped-CV
    # or leave-one-source-out run — see fold_preprocessing.py — since
    # reusing the OUTER preprocessing_artifact (fit on ALL original-train
    # subjects) across CV folds leaks an inner-validation subject's
    # influence on scaling/HVG selection into that fold. None only for
    # hand-built contexts that don't intend to run CV (e.g. a context built
    # to test something else entirely) — CV code must fail clearly, not
    # silently fall back to the outer artifact, when this is None.
    normalized_adata_for_refit: Optional["object"] = None

    @property
    def num_smoke_classes(self) -> int:
        return self.label_mapping.k

    @property
    def input_dim(self) -> int:
        return len(self.preprocessing_artifact.gene_list)

    def subjects_for(self, split: str) -> List[str]:
        return self.split_manifest.subjects_for(split)

    def fingerprint(self) -> Optional[str]:
        return self.split_manifest.fingerprint

    @property
    def config_fingerprint(self) -> str:
        """SHA-256 of this context's own (deep-copied, immutable-to-callers)
        config snapshot — lets a checkpoint/result record exactly which
        configuration produced it and detect a mismatch on reload."""
        blob = json.dumps(self.config, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @property
    def label_mapping_fingerprint(self) -> str:
        blob = json.dumps(self.label_mapping.to_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @property
    def preprocessing_artifact_fingerprint(self) -> str:
        from .fold_preprocessing import artifact_fingerprint
        return artifact_fingerprint(self.preprocessing_artifact)

    @property
    def test_membership_fingerprint(self) -> str:
        """SHA-256 of the frozen test-subject membership list, derived
        exclusively from split_manifest.test_subjects — never from
        test_bags/test_cell_dataset. Reading which subject IDs constitute
        the test split is not the same as reading their labels or
        expression, and the split manifest is fixed at split time, so this
        is safe to compute before the frozen-test guard is acquired and
        used as part of the guard's identity (blocker 2)."""
        ids = sorted(str(s) for s in self.split_manifest.test_subjects)
        blob = json.dumps(ids, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def run_identity(self, run_id: str) -> Dict[str, Optional[str]]:
        """
        The full set of fingerprints/identifiers a reproducibility artifact
        or checkpoint must record and a reload must re-verify (section 7):
        manifest identity, preprocessing artifact identity, label-mapping
        identity, configuration identity, and this specific run's id. Two
        runs with identical values here operated on provably identical
        inputs; any mismatch on reload means the checkpoint no longer
        describes the current context and must not be silently reused.
        """
        return {
            "run_id": run_id,
            "git_sha": self.git_sha,
            "split_manifest_fingerprint": self.split_manifest.fingerprint,
            "preprocessing_artifact_fingerprint": self.preprocessing_artifact_fingerprint,
            "label_mapping_fingerprint": self.label_mapping_fingerprint,
            "config_fingerprint": self.config_fingerprint,
        }

    def guard_identity_fingerprint(self, selected_model: str, extra: Optional[Dict] = None) -> str:
        """
        SHA-256 identity key for the durable frozen-test guard (see
        test_guard.py): manifest + preprocessing + label-mapping +
        configuration + the selected final model, combined with whatever
        additional dev-only-derived fingerprints/parameters the caller
        passes via `extra` (e.g. selected hyperparameters, the FINAL
        development-refit preprocessing/model fingerprints, calibration
        parameters, the frozen threshold — see runner.py). Deliberately
        independent of any run_id/run_dir name — two runs given DIFFERENT
        output directory names but the SAME underlying data/config/model/
        hyperparameter selection must resolve to the SAME guard identity,
        so a fresh `--run-id` cannot be used to bypass the one-time
        test-evaluation guard for scientifically identical conditions.
        Changing any selected hyperparameter or the final model fingerprint
        changes this identity, so a genuinely different final candidate is
        never blocked by a guard file recorded for a different one.
        """
        payload = {
            "split_manifest_fingerprint": self.split_manifest.fingerprint,
            "preprocessing_artifact_fingerprint": self.preprocessing_artifact_fingerprint,
            "label_mapping_fingerprint": self.label_mapping_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "selected_model": selected_model,
        }
        if extra:
            payload["extra"] = extra
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def validate_run_identity(self, expected: Dict[str, Optional[str]]) -> None:
        """
        Recompute this context's own run_identity() and compare field-by-
        field against a previously-recorded one (e.g. loaded from a
        checkpoint or a run artifact). Raises loudly on any mismatch —
        never silently proceeds with a stale/incompatible checkpoint.
        """
        current = self.run_identity(expected.get("run_id", ""))
        mismatches = {
            k: (expected.get(k), current.get(k))
            for k in ("split_manifest_fingerprint", "preprocessing_artifact_fingerprint",
                      "label_mapping_fingerprint", "config_fingerprint")
            if expected.get(k) != current.get(k)
        }
        if mismatches:
            raise ValueError(
                f"ExperimentContext.validate_run_identity: {len(mismatches)} field(s) do not match "
                f"the recorded run identity: {mismatches}. Refusing to treat this context as "
                "equivalent to the one the checkpoint/result was produced from."
            )

    @classmethod
    def from_pipeline_result(
        cls, result: dict, config: dict, seed: Optional[int] = None,
    ) -> "ExperimentContext":
        """
        Build from run_pipeline_split_aware()'s return dict. Raises if any
        required key is missing — a caller passing run_pipeline()'s (the
        non-split-aware, legacy) result here is a scientific-validity bug,
        not something to silently paper over with a default.
        """
        required = [
            "train_cell_dataset", "val_cell_dataset", "test_cell_dataset",
            "train_bags", "val_bags", "test_bags", "split_manifest",
            "preprocessing_artifact", "label_mapping", "rare_class_report",
            "label_provenance_report", "transductive_batch_correction",
        ]
        missing = [k for k in required if k not in result]
        if missing:
            raise ValueError(
                f"ExperimentContext.from_pipeline_result: result is missing {missing} — "
                "this must be the dict returned by run_pipeline_split_aware(), not "
                "run_pipeline() (which has no split-aware keys)."
            )
        split_cfg = config.get("split", {}) if isinstance(config, dict) else {}
        resolved_seed = seed if seed is not None else split_cfg.get("seed", 42)

        summary = {
            "train": _dataset_source_summary(result["train_cell_dataset"]),
            "val":   _dataset_source_summary(result["val_cell_dataset"]),
            "test":  _dataset_source_summary(result["test_cell_dataset"]),
        }

        _validate_context(
            result["train_cell_dataset"], result["val_cell_dataset"], result["test_cell_dataset"],
            result["train_bags"], result["val_bags"], result["test_bags"],
            result["preprocessing_artifact"], result["label_mapping"], result["split_manifest"],
        )

        return cls(
            train_cell_dataset=result["train_cell_dataset"],
            val_cell_dataset=result["val_cell_dataset"],
            test_cell_dataset=result["test_cell_dataset"],
            train_bags=result["train_bags"],
            val_bags=result["val_bags"],
            test_bags=result["test_bags"],
            split_manifest=result["split_manifest"],
            preprocessing_artifact=result["preprocessing_artifact"],
            label_mapping=result["label_mapping"],
            rare_class_report=result["rare_class_report"],
            label_provenance_report=result["label_provenance_report"],
            transductive_batch_correction=result["transductive_batch_correction"],
            # Deep-copied so no caller holding a reference to the original
            # config dict can mutate this context's view of it after
            # construction (section 9's immutability requirement).
            config=copy.deepcopy(config),
            seed=resolved_seed,
            dataset_source_summary=summary,
            git_sha=get_git_sha(),
            normalized_adata_for_refit=result.get("normalized_adata_for_refit"),
        )
