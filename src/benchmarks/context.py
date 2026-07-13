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

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data.label_mapping import EffectiveLabelMapping
from data.preprocessing import PreprocessingArtifact
from data.splitting import SplitManifest


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
            config=config,
            seed=resolved_seed,
            dataset_source_summary=summary,
            git_sha=get_git_sha(),
        )
