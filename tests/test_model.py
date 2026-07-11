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
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from constants import CELL_TYPES, N_SMOKE_CLASSES, SMOKE_TYPES
from model import MultiSmokeCancerNet
from train import CellLevelDataset, SubjectLevelDataset, subject_collate_fn


# ─── DRY metric helpers ───────────────────────────────────────────────────────

def _smoke_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Per-class and macro F1 for smoke type classification."""
    from sklearn.metrics import classification_report
    report = classification_report(
        y_true, y_pred,
        labels       = list(range(N_SMOKE_CLASSES)),
        target_names = list(SMOKE_TYPES.values()),
        output_dict  = True,
        zero_division= 0,
    )
    return {
        "accuracy":  round(report["accuracy"], 4),
        "macro_f1":  round(report["macro avg"]["f1-score"], 4),
        "per_class": {k: round(v["f1-score"], 4)
                      for k, v in report.items() if k in SMOKE_TYPES.values()},
    }


def _binary_metrics(
    y_true:    List[float],
    y_prob:    List[float],
    threshold: float = 0.5,
) -> Dict:
    """ROC-AUC, PR-AUC, sensitivity, specificity for any binary task."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, confusion_matrix
    )
    has_both = len(set(y_true)) > 1
    y_pred   = [1 if p >= threshold else 0 for p in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc":     round(roc_auc_score(y_true, y_prob), 4)         if has_both else None,
        "pr_auc":      round(average_precision_score(y_true, y_prob), 4) if has_both else None,
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) > 0 else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else None,
        "threshold":   threshold,
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
        model.load_state_dict(
            torch.load(ckpt_dir / f"phase{phase}_best.pt",
                       map_location=device, weights_only=True)
        )
        print(f"[evaluate] loaded phase {phase} checkpoint  ({ckpt_dir}/)")
        return cls(model, device)

    # ── Cell-level ────────────────────────────────────────────────────────────

    def cell_level(
        self,
        cell_dataset: CellLevelDataset,
        batch_size:   int = 512,
    ) -> Dict:
        """Smoke type F1 + malignancy AUC. Corresponds to Phase 1 evaluation."""
        smoke_preds, smoke_true = [], []
        malig_probs, malig_true = [], []

        self.model.eval()
        with torch.no_grad():
            for batch in DataLoader(cell_dataset, batch_size=batch_size, shuffle=False):
                _, logits, malig = self.model.forward_cell(batch["x"].to(self.device))
                smoke_preds.extend(logits.argmax(1).cpu().tolist())
                smoke_true.extend(batch["smoke_label"].tolist())
                malig_probs.extend(malig.squeeze().cpu().tolist())
                malig_true.extend(batch["malignancy_label"].tolist())

        return {
            "n_cells":    len(smoke_true),
            "smoke_type": _smoke_metrics(smoke_true, smoke_preds),
            "malignancy": _binary_metrics(malig_true, malig_probs, threshold=0.5),
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

        # Raw predictions — needed by visualize.plot_all()
        raw = {
            "y_true_cancer": [r["label"] for r in records],
            "y_prob_cancer":  [r["prob"]  for r in records],
        }

        report = {
            "cell_level":       self.cell_level(cell_dataset),
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

    print("\n=== PASSED ===")