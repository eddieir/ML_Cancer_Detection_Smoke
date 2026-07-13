"""
evaluate.py — Model evaluation and interpretability.

Imports Dataset classes from train.py — no duplication.
Three evaluation levels match the three training phases:
  cell_level()       → smoke type F1 + malignancy AUC
  subject_level()    → cancer ROC-AUC, sensitivity/specificity at clinical threshold
  interpretability() → attention by cell type, by smoke type, malignancy correlation

full_report() does ONE forward pass over subject_dataset shared between
subject_level and interpretability — not two.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from constants import CELL_TYPES, N_SMOKE_CLASSES, SMOKE_TYPES
from metrics import multiclass_f1_report
from model import MultiSmokeCancerNet
from train import CellLevelDataset, SubjectLevelDataset, load_checkpoint_into, subject_collate_fn


def describe_split(
    split_name:        str,
    manifest,
    is_held_out:        bool = True,
    data_modality:       str = "single_cell",
    rare_class_policy:   Optional[str] = None,
) -> Dict:
    """
    Provenance metadata for an evaluation output (not a metric itself): which
    split, how many subjects/cells, label distribution, whether it's
    genuinely held out, and whether the underlying data is bulk or single-
    cell. Attach this to any saved evaluation_report.json so a number can't
    be mistaken for held-out test performance when it's actually training-
    set accuracy computed on data the model has seen.
    """
    split_report = manifest.report.get("splits", {}).get(split_name, {})
    return {
        "split_name":         split_name,
        "is_held_out":        is_held_out,
        "data_modality":      data_modality,
        "n_subjects":         split_report.get("n_subjects"),
        "n_cells":            split_report.get("n_cells"),
        "n_unknown_label":    split_report.get("n_unknown_label"),
        "class_distribution": split_report.get("class_distribution"),
        "rare_class_policy":  rare_class_policy,
    }


# ─── DRY metric helpers ───────────────────────────────────────────────────────

def _smoke_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    Full smoke-type multiclass metric set — macro-F1 is the primary
    model-selection metric (see train.py's Phase 1 checkpoint selection),
    but overall accuracy alone is misleading on an imbalanced 6-class
    problem (a model that only ever predicts the majority class scores
    well on accuracy while getting every minority class wrong), so this
    always reports the full breakdown alongside it.
    """
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

    if len(y_true) == 0:
        return {"n": 0, "note": "no labelled cells — metrics undefined"}

    report = classification_report(
        y_true, y_pred,
        labels       = list(range(N_SMOKE_CLASSES)),
        target_names = list(SMOKE_TYPES.values()),
        output_dict  = True,
        zero_division= 0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_SMOKE_CLASSES)))
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)

    # Same shared definition Trainer.phase1 uses for checkpoint selection —
    # classification_report's macro/weighted avg already use the explicit
    # full label list passed above, so these numbers agree with f1_report's
    # by construction; f1_report additionally names which classes had zero
    # true examples so this number isn't mistaken for a full 6-class score.
    f1_report = multiclass_f1_report(y_true, y_pred, N_SMOKE_CLASSES)

    return {
        "n":               len(y_true),
        "accuracy":        round(report["accuracy"], 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1":        round(report["macro avg"]["f1-score"], 4),
        "weighted_f1":      round(report["weighted avg"]["f1-score"], 4),
        "classes_absent_from_targets": f1_report["classes_absent_from_targets"],
        "is_partial":       f1_report["is_partial"],
        "per_class": {
            k: {
                "precision": round(v["precision"], 4),
                "recall":    round(v["recall"], 4),
                "f1":        round(v["f1-score"], 4),
                "support":   int(v["support"]),
            }
            for k, v in report.items() if k in SMOKE_TYPES.values()
        },
        "confusion_matrix":            cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
        "class_labels":                list(SMOKE_TYPES.values()),
    }


def _binary_metrics(
    y_true:    List[float],
    y_prob:    List[float],
    threshold: float = 0.5,
) -> Dict:
    """
    ROC-AUC, PR-AUC, sensitivity, specificity for any binary task.
    Gracefully handles empty input and single-class targets by returning
    None for undefined metrics rather than a misleading number — callers
    must not substitute a default like 0.5 or 1.0 in place of None.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, confusion_matrix
    )
    n = len(y_true)
    if n == 0:
        return {"roc_auc": None, "pr_auc": None, "sensitivity": None,
                "specificity": None, "threshold": threshold, "n": 0,
                "note": "no known labels — metrics undefined"}
    has_both = len(set(y_true)) > 1
    y_pred   = [1 if p >= threshold else 0 for p in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc":     round(roc_auc_score(y_true, y_prob), 4)         if has_both else None,
        "pr_auc":      round(average_precision_score(y_true, y_prob), 4) if has_both else None,
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) > 0 else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else None,
        "threshold":   threshold,
        "n":           n,
        "note":        None if has_both else "single-class target — ROC/PR-AUC undefined",
    }


# ─── Shared subject output type ───────────────────────────────────────────────

# Named fields collected per subject in a single forward pass.
# _SubjectOutputs is a list of these dicts — shared between subject_level and interpretability.
def _empty_subject_record() -> Dict:
    return {
        "prob":       0.0,
        "label":      0.0,
        "attn":       None,   # np.ndarray [N]
        "smoke_pred": None,   # np.ndarray [N]  int (argmax)
        "malig":      None,   # np.ndarray [N]
        "ct_ids":     None,   # np.ndarray [N]  int
    }


# ─── Evaluator ────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Three evaluation levels for MultiSmokeCancerNet.
    Use from_checkpoint() to load a trained model, or pass a model directly.

    full_report() shares ONE subject forward pass across subject_level()
    and interpretability() — not two separate passes.
    """

    def __init__(self, model: MultiSmokeCancerNet, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        config: Union[dict, str, Path],
        phase:  int = 3,
        device: str = "cpu",
    ) -> "Evaluator":
        """Load the best checkpoint from a given training phase."""
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        model    = MultiSmokeCancerNet.from_config(config)
        ckpt_dir = Path(config.get("train", config).get("checkpoint_dir", "checkpoints"))
        if not ckpt_dir.is_absolute():
            ckpt_dir = Path(__file__).parents[1] / ckpt_dir
        load_checkpoint_into(model, ckpt_dir / f"phase{phase}_best.pt", device)
        print(f"[evaluate] loaded phase {phase} checkpoint  ({ckpt_dir}/)")
        return cls(model, device)

    # ── Cell-level ────────────────────────────────────────────────────────────

    def _cell_predictions(
        self,
        cell_dataset: CellLevelDataset,
        batch_size:   int = 512,
    ) -> Dict:
        """Single forward pass over cell_dataset — shared by cell_level() and full_report()."""
        smoke_preds, smoke_true = [], []
        malig_probs, malig_true, malig_known = [], [], []

        self.model.eval()
        with torch.no_grad():
            for batch in DataLoader(cell_dataset, batch_size=batch_size, shuffle=False):
                _, logits, malig = self.model.forward_cell(batch["x"].to(self.device))
                smoke_preds.extend(logits.argmax(1).cpu().tolist())
                smoke_true.extend(batch["smoke_label"].tolist())
                malig_probs.extend(malig.squeeze().cpu().tolist())
                malig_true.extend(batch["malignancy_label"].tolist())
                malig_known.extend(batch["malignancy_known"].tolist())

        return {
            "smoke_pred": smoke_preds, "smoke_true": smoke_true,
            "malig_prob": malig_probs, "malig_true": malig_true, "malig_known": malig_known,
        }

    @staticmethod
    def _known_malignancy_metrics(p: Dict) -> Dict:
        """
        Restrict malignancy metrics to cells with a real label — cells
        stamped with the 0.0 placeholder (unknown) are not verified-normal
        and must not count as evidence the model correctly predicted "not
        malignant" (see labellers.py::add_malignancy_labels).
        """
        known_true = [t for t, k in zip(p["malig_true"], p["malig_known"]) if k]
        known_prob = [pr for pr, k in zip(p["malig_prob"], p["malig_known"]) if k]
        metrics = _binary_metrics(known_true, known_prob, threshold=0.5)
        metrics["n_known_positive"] = int(sum(t == 1.0 for t in known_true))
        metrics["n_known_negative"] = int(sum(t == 0.0 for t in known_true))
        metrics["n_unknown"] = len(p["malig_true"]) - len(known_true)
        return metrics

    def cell_level(
        self,
        cell_dataset: CellLevelDataset,
        batch_size:   int = 512,
    ) -> Dict:
        """Smoke type F1 + malignancy AUC. Corresponds to Phase 1 evaluation."""
        p = self._cell_predictions(cell_dataset, batch_size)
        return {
            "n_cells":    len(p["smoke_true"]),
            "smoke_type": _smoke_metrics(p["smoke_true"], p["smoke_pred"]),
            "malignancy": self._known_malignancy_metrics(p),
        }

    # ── Subject-level ─────────────────────────────────────────────────────────

    def subject_level(
        self,
        subject_dataset: SubjectLevelDataset,
        threshold: float = 0.70,
    ) -> Dict:
        """Cancer probability evaluation. Reports at clinical HIGH RISK threshold (≥ 0.70)."""
        records = self._forward_subjects(subject_dataset)
        return self._subject_metrics_from_records(records, threshold)

    # ── Interpretability ──────────────────────────────────────────────────────

    def interpretability(self, subject_dataset: SubjectLevelDataset) -> Dict:
        """
        Attention weight analysis — reveals which cells drive cancer predictions.
          mean_attention_by_cell_type  : epithelial cells should dominate
          mean_attention_by_smoke_type : which smoke signature gets most focus
          malignancy_attention_correlation : does higher malignancy → higher attention?
        """
        records = self._forward_subjects(subject_dataset)
        return self._interpretability_from_records(records)

    # ── Full report (single subject pass) ────────────────────────────────────

    def full_report(
        self,
        cell_dataset:    CellLevelDataset,
        subject_dataset: SubjectLevelDataset,
        out_dir:         Union[str, Path] = "checkpoints",
        threshold:       float = 0.70,
    ) -> Dict:
        """
        Run all evaluations and save evaluation_report.json.
        Subject dataset is iterated ONCE — outputs shared between
        subject_level and interpretability.
        """
        out = Path(out_dir)
        if not out.is_absolute():
            out = Path(__file__).parents[1] / out
        out.mkdir(parents=True, exist_ok=True)

        # Single subject forward pass
        records = self._forward_subjects(subject_dataset)
        cell_preds = self._cell_predictions(cell_dataset)

        # Raw predictions — needed by visualize.plot_all()
        raw = {
            "y_true_cancer": [r["label"] for r in records],
            "y_prob_cancer":  [r["prob"]  for r in records],
            "y_true_smoke":   cell_preds["smoke_true"],
            "y_pred_smoke":   cell_preds["smoke_pred"],
        }

        report = {
            "cell_level": {
                "n_cells":    len(cell_preds["smoke_true"]),
                "smoke_type": _smoke_metrics(cell_preds["smoke_true"], cell_preds["smoke_pred"]),
                "malignancy": self._known_malignancy_metrics(cell_preds),
            },
            "subject_level":    self._subject_metrics_from_records(records, threshold),
            "interpretability": self._interpretability_from_records(records),
        }

        rpath = out / "evaluation_report.json"
        with open(rpath, "w") as f:
            json.dump(report, f, indent=2)

        self._print_summary(report)
        print(f"\n[evaluate] report → {rpath}")
        return report, raw   # raw contains y_true/y_prob for ROC/PR/calibration plots

    # ── Private: single forward pass ──────────────────────────────────────────

    def _forward_subjects(self, subject_dataset: SubjectLevelDataset) -> List[Dict]:
        """
        One forward pass over all subjects.
        Returns a list of per-subject records used by both
        _subject_metrics_from_records and _interpretability_from_records.
        """
        records = []
        self.model.eval()
        with torch.no_grad():
            for [item] in DataLoader(subject_dataset, batch_size=1,
                                     collate_fn=subject_collate_fn):
                out = self.model.forward_subject(
                    item["gene_matrix"].to(self.device),
                    item["cell_type_ids"].to(self.device),
                )
                records.append({
                    "prob":       out["cancer_probability"].item(),
                    "label":      item["cancer_label"].item(),
                    "attn":       out["attention_weights"].cpu().numpy(),
                    "smoke_pred": out["cell_smoke_probs"].cpu().numpy().argmax(axis=1),
                    "malig":      out["cell_malignancy"].squeeze().cpu().numpy(),
                    "ct_ids":     item["cell_type_ids"].numpy(),
                })
        return records

    # ── Private: compute metrics from cached records ───────────────────────────

    def _subject_metrics_from_records(
        self, records: List[Dict], threshold: float
    ) -> Dict:
        probs  = [r["prob"]  for r in records]
        labels = [r["label"] for r in records]
        return {
            "n_subjects":        len(records),
            "cancer":            _binary_metrics(labels, probs, threshold=threshold),
            "risk_distribution": {
                "high":     sum(p >= 0.70 for p in probs),
                "moderate": sum(0.40 <= p < 0.70 for p in probs),
                "low":      sum(p <  0.40 for p in probs),
            },
        }

    def _interpretability_from_records(self, records: List[Dict]) -> Dict:
        attn_by_cell  = {ct: [] for ct in CELL_TYPES.values()}
        attn_by_smoke = {st: [] for st in SMOKE_TYPES.values()}
        ml_at_pairs:  List[Tuple[float, float]] = []

        for r in records:
            for a, c, s, m in zip(r["attn"], r["ct_ids"],
                                   r["smoke_pred"], r["malig"]):
                attn_by_cell [CELL_TYPES [int(c)]].append(float(a))
                attn_by_smoke[SMOKE_TYPES[int(s)]].append(float(a))
                ml_at_pairs.append((float(m), float(a)))

        ml  = np.array([x[0] for x in ml_at_pairs])
        at  = np.array([x[1] for x in ml_at_pairs])
        corr = float(np.corrcoef(ml, at)[0, 1]) if len(ml) > 1 else 0.0

        return {
            "n_cells_analysed":               len(ml_at_pairs),
            "mean_attention_by_cell_type":    {k: round(float(np.mean(v)), 6) if v else 0.0
                                               for k, v in attn_by_cell.items()},
            "mean_attention_by_smoke_type":   {k: round(float(np.mean(v)), 6) if v else 0.0
                                               for k, v in attn_by_smoke.items()},
            "malignancy_attention_correlation": round(corr, 4),
        }

    def _print_summary(self, report: Dict) -> None:
        cl       = report["cell_level"]
        sl       = report["subject_level"]
        ip       = report["interpretability"]
        top_cell = max(
            ip["mean_attention_by_cell_type"],
            key=ip["mean_attention_by_cell_type"].get,
        )
        print("\n── Evaluation Summary ──────────────────────────")
        print(f"  Cells          : {cl['n_cells']:,}")
        print(f"  Smoke macro-F1 : {cl['smoke_type']['macro_f1']:.3f}")
        print(f"  Malignancy AUC : {cl['malignancy']['roc_auc']}")
        print(f"  Subjects       : {sl['n_subjects']}")
        print(f"  Cancer AUC     : {sl['cancer']['roc_auc']}")
        print(f"  Sensitivity    : {sl['cancer']['sensitivity']}")
        print(f"  Specificity    : {sl['cancer']['specificity']}")
        print(f"  Malig-attn ρ   : {ip['malignancy_attention_correlation']:.4f}")
        print(f"  Top cell type  : {top_cell}")
        print("────────────────────────────────────────────────")


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    from constants import N_CELL_TYPES

    GENES, N_CELLS, N_SUBJ = 2000, 500, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Note on expected metrics ──────────────────────────────────────────────
    # This smoke test uses an UNTRAINED model on RANDOM data.
    # Near-random metrics are correct and expected here:
    #   smoke macro-F1 ≈ 0.04  (random 6-class: 1/6 ≈ 0.17 accuracy but uneven F1)
    #   malignancy AUC ≈ 0.50  (untrained binary classifier)
    #   cancer AUC     ≈ 0.50  (same reason)
    # After real training, expect:  smoke F1 > 0.60,  AUCs > 0.75
    # ─────────────────────────────────────────────────────────────────────────

    CFG = Path(__file__).parents[1] / "configs" / "default.yaml"

    cell_ds = CellLevelDataset(
        gene_matrix       = np.random.randn(N_CELLS, GENES).astype("float32"),
        smoke_labels      = np.random.randint(0, N_SMOKE_CLASSES, N_CELLS),
        malignancy_labels = np.random.randint(0, 2, N_CELLS).astype("float32"),
        cell_type_ids     = np.random.randint(0, N_CELL_TYPES, N_CELLS),
        diagnostic_mode    = True,
    )

    def _bag(n):
        return {
            "gene_matrix":   np.random.randn(n, GENES).astype("float32"),
            "cell_type_ids": np.random.randint(0, N_CELL_TYPES,    n),
            "smoke_labels":  np.random.randint(0, N_SMOKE_CLASSES, n),
            "malig_labels":  np.random.randint(0, 2, n).astype("float32"),
            "cancer_label":  random.randint(0, 1),
        }

    subject_ds = SubjectLevelDataset([
        {"subject_id": f"sub_{i}", **_bag(random.randint(30, 80))}
        for i in range(N_SUBJ)
    ])

    model = MultiSmokeCancerNet.from_config(CFG)
    ev    = Evaluator(model, device)

    # Assertions check CODE correctness, not MODEL performance
    cl = ev.cell_level(cell_ds)
    assert "smoke_type" in cl and "malignancy" in cl and "n_cells" in cl
    assert 0.0 <= cl["smoke_type"]["macro_f1"] <= 1.0
    print(f"cell_level       ✓  smoke_macro_f1={cl['smoke_type']['macro_f1']:.3f}"
          f"  malig_auc={cl['malignancy']['roc_auc']}"
          f"  (untrained → near-random expected)")

    sl = ev.subject_level(subject_ds)
    assert "cancer" in sl and "risk_distribution" in sl
    assert sl["n_subjects"] == N_SUBJ
    print(f"subject_level    ✓  cancer_auc={sl['cancer']['roc_auc']}"
          f"  n={sl['n_subjects']}"
          f"  (untrained → near-random expected)")

    ip = ev.interpretability(subject_ds)
    assert "malignancy_attention_correlation" in ip
    assert set(ip["mean_attention_by_cell_type"].keys()) == set(CELL_TYPES.values())
    assert set(ip["mean_attention_by_smoke_type"].keys()) == set(SMOKE_TYPES.values())
    print(f"interpretability ✓  malig-attn ρ={ip['malignancy_attention_correlation']:.4f}"
          f"  n_cells={ip['n_cells_analysed']}")

    # full_report uses single subject pass — verify it produces the same n_subjects
    report, raw = ev.full_report(cell_ds, subject_ds)
    assert report["subject_level"]["n_subjects"] == N_SUBJ
    assert (Path(__file__).parents[1] / "checkpoints" / "evaluation_report.json").exists()

    # Persist raw predictions too — scripts/generate_plots.py reads this to
    # render ROC/PR/calibration/confusion-matrix plots without rerunning the model.
    raw_path = Path(__file__).parents[1] / "checkpoints" / "evaluation_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f)
    print(f"[evaluate] raw predictions → {raw_path}")

    print("\n=== PASSED ===")