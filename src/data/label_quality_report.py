"""
data/label_quality_report.py — machine-readable label-quality report,
built from the outputs run_pipeline_split_aware() already produces
(split_manifest, rare_class_report, label_provenance_report) plus the
species/assay checks added in this change, rather than re-deriving any of
those facts independently. Persisted as JSON (full detail) + a compact CSV
summary per split, so a training entry point can refuse to start when a
critical integrity violation is present (see build_and_check below).
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

SCHEMA_VERSION = "1"


@dataclass
class LabelQualityReport:
    schema_version:              str
    dataset_manifest_fingerprint: Optional[str]
    label_policy_fingerprint:     Optional[str]
    split_fingerprint:            Optional[str]
    per_split:                    Dict[str, dict] = field(default_factory=dict)
    flags:                        List[str] = field(default_factory=list)

    @property
    def has_critical_violations(self) -> bool:
        return len(self.flags) > 0

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dataset_manifest_fingerprint": self.dataset_manifest_fingerprint,
            "label_policy_fingerprint": self.label_policy_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "per_split": self.per_split,
            "flags": self.flags,
        }

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def save_csv_summary(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for split_name, s in self.per_split.items():
            rows.append({
                "split": split_name,
                "n_subjects": s.get("n_subjects"),
                "n_cells": s.get("n_cells"),
                "n_unknown_smoke_label": s.get("n_unknown_label"),
                "cancer_outcome_known": s.get("cancer_outcome_known"),
                "cancer_outcome_unknown": s.get("cancer_outcome_unknown"),
                "malignancy_known_cells": s.get("malignancy_known_cells"),
                "malignancy_unknown_cells": s.get("malignancy_unknown_cells"),
            })
        with open(path, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)


def build_label_quality_report(
    pipeline_result: dict,
    dataset_manifest_fingerprint: Optional[str] = None,
    label_policy_fingerprint: Optional[str] = None,
) -> LabelQualityReport:
    """
    Build a LabelQualityReport from preprocess.py::run_pipeline_split_aware's
    return value. Never re-derives known/unknown counts independently —
    reuses split_manifest.report and label_provenance_report exactly as
    produced, so this report cannot drift from what the pipeline itself
    recorded.
    """
    manifest = pipeline_result["split_manifest"]
    prov = pipeline_result.get("label_provenance_report", {})
    split_report = manifest.report.get("splits", {})

    per_split: Dict[str, dict] = {}
    for split_name, s in split_report.items():
        per_split[split_name] = {
            "n_subjects": s.get("n_subjects"),
            "n_cells": s.get("n_cells"),
            "class_distribution": s.get("class_distribution"),
            "n_unknown_label": s.get("n_unknown_label"),
            # Provenance counts are dataset-wide (assemble_subject_bags runs
            # once over the merged, already-split-labeled data) — recorded
            # once under "all", not fabricated per-split.
            "cancer_outcome_known": prov.get("cancer_outcome_known_subjects") if split_name == "all" else None,
            "cancer_outcome_unknown": prov.get("cancer_outcome_unknown_subjects") if split_name == "all" else None,
            "malignancy_known_cells": prov.get("malignancy_known_cells") if split_name == "all" else None,
            "malignancy_unknown_cells": prov.get("malignancy_unknown_cells") if split_name == "all" else None,
            "smoke_verified_known_cells": prov.get("smoke_verified_known_cells") if split_name == "all" else None,
            "smoke_weak_proxy_cells": prov.get("smoke_weak_proxy_cells") if split_name == "all" else None,
            "smoke_unknown_cells": prov.get("smoke_unknown_cells") if split_name == "all" else None,
            "weak_labels_enabled": prov.get("weak_labels_enabled") if split_name == "all" else None,
            "nlst_cells_verified_smoke_label": prov.get("nlst_cells_verified_smoke_label") if split_name == "all" else None,
            "nlst_cells_unknown_smoke_label": prov.get("nlst_cells_unknown_smoke_label") if split_name == "all" else None,
        }
    per_split["all"] = {
        "n_subjects": sum(s.get("n_subjects", 0) for s in split_report.values()),
        "n_cells": sum(s.get("n_cells", 0) for s in split_report.values()),
        "cancer_outcome_known": prov.get("cancer_outcome_known_subjects"),
        "cancer_outcome_unknown": prov.get("cancer_outcome_unknown_subjects"),
        "malignancy_known_cells": prov.get("malignancy_known_cells"),
        "malignancy_unknown_cells": prov.get("malignancy_unknown_cells"),
        "smoke_verified_known_cells": prov.get("smoke_verified_known_cells"),
        "smoke_weak_proxy_cells": prov.get("smoke_weak_proxy_cells"),
        "smoke_unknown_cells": prov.get("smoke_unknown_cells"),
        "weak_labels_enabled": prov.get("weak_labels_enabled"),
        "nlst_cells_verified_smoke_label": prov.get("nlst_cells_verified_smoke_label"),
        "nlst_cells_unknown_smoke_label": prov.get("nlst_cells_unknown_smoke_label"),
    }

    flags: List[str] = []
    # Subjects with zero usable supervised labels anywhere.
    if prov.get("cancer_outcome_known_subjects", 0) == 0:
        flags.append(
            "no_subjects_with_known_cancer_outcome: every subject in this run has an "
            "unknown cancer outcome — cancer-outcome training/evaluation has nothing to "
            "learn or measure from."
        )
    if manifest.report.get("unstratified_classes"):
        flags.append(
            f"unstratified_classes: {manifest.report['unstratified_classes']} could not be "
            "stratified across splits (too few independent subjects)."
        )
    n_weak_proxy = prov.get("smoke_weak_proxy_cells") or 0
    if n_weak_proxy and not prov.get("weak_labels_enabled"):
        flags.append(
            f"weak_smoke_proxy_present_but_disabled: {n_weak_proxy} cell(s) carry a "
            "documented weak smoke-type proxy (e.g. GSE136831's COPD-diagnosis proxy) that "
            "is excluded from smoke supervision under the default verified_only policy — "
            "set data.weak_labels.enabled=true to opt in."
        )
    n_nlst_unknown = prov.get("nlst_cells_unknown_smoke_label") or 0
    if n_nlst_unknown:
        flags.append(
            f"nlst_matched_but_unknown_smoke_label: {n_nlst_unknown} cell(s) matched an NLST "
            "subject_id but CIGSMOK/CIGAR did not parse to a documented positive code (see "
            "data/nlst_smoking.py) — excluded from smoke supervision as unknown, not defaulted "
            "to cigarette/cigar/unexposed."
        )

    return LabelQualityReport(
        schema_version=SCHEMA_VERSION,
        dataset_manifest_fingerprint=dataset_manifest_fingerprint,
        label_policy_fingerprint=label_policy_fingerprint,
        split_fingerprint=manifest.fingerprint,
        per_split=per_split,
        flags=flags,
    )
