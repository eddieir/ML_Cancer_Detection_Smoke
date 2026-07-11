"""
visualize.py — All plots for MultiSmokeCancerNet.
No computation here — pure rendering of pre-computed results.

Three input sources (all pre-computed by other modules):
  train.py    → checkpoints/phase*_history.json
  evaluate.py → checkpoints/evaluation_report.json
  inference.py→ checkpoints/inference_results.json

Entry points:
  plot_training_history(history_dir)
  plot_evaluation(report, y_true, y_prob_cancer, y_true_smoke, y_pred_smoke)
  plot_interpretability(report)
  plot_cell_umap(embeddings, smoke_labels, malignancy_scores)
  plot_patient_profile(result)
  plot_all(...)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np

# ─── DRY save helper ─────────────────────────────────────────────────────────

def _save(fig, name: str, out_dir: Path) -> None:
    """Single place where figures are saved and closed."""
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved → {path.name}")


def _resolve_out(out_dir: Union[str, Path, None]) -> Path:
    default = Path(__file__).parents[1] / "checkpoints" / "plots"
    return Path(out_dir) if out_dir else default


# ─── Training plots ───────────────────────────────────────────────────────────

def plot_training_history(
    history_dir: Union[str, Path, None] = None,
    out_dir:     Union[str, Path, None] = None,
) -> None:
    """
    Reads phase1/2/3_history.json from checkpoints/ and plots
    metric curves for all three training phases in one figure.
    """
    import matplotlib.pyplot as plt

    history_dir = Path(history_dir) if history_dir else Path(__file__).parents[1] / "checkpoints"
    out         = _resolve_out(out_dir)

    phase_cfg = {
        1: ("smoke_acc",  "Smoke Accuracy",    "#4C72B0"),
        2: ("val_auc",    "Subject AUC",        "#DD8452"),
        3: ("val_auc",    "Subject AUC",        "#55A868"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Training Curriculum — All Three Phases", fontsize=13, fontweight="bold")

    for ax, (phase, (metric, label, colour)) in zip(axes, phase_cfg.items()):
        path = history_dir / f"phase{phase}_history.json"
        if not path.exists():
            ax.text(0.5, 0.5, f"phase{phase}_history.json\nnot found",
                    ha="center", va="center", transform=ax.transAxes, color="grey")
            ax.set_title(f"Phase {phase}")
            continue
        data   = json.loads(path.read_text())[f"phase{phase}"]
        epochs = [d["epoch"]  for d in data]
        values = [d[metric]   for d in data]
        ax.plot(epochs, values, marker="o", color=colour, linewidth=2)
        ax.set_title(f"Phase {phase}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, "training_history", out)


# ─── Evaluation plots ─────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: List[float],
    y_prob: List[float],
    title:  str = "Cancer Risk — ROC Curve",
    out_dir: Union[str, Path, None] = None,
) -> None:
    """ROC curve with AUC annotation and clinical 0.70 threshold marked."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Mark point closest to clinical threshold 0.70
    idx   = np.argmin(np.abs(thresholds - 0.70))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random")
    ax.scatter(fpr[idx], tpr[idx], color="#C44E52", zorder=5, s=80,
               label=f"Threshold 0.70\n(sens={tpr[idx]:.2f}, 1-spec={fpr[idx]:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "roc_curve", _resolve_out(out_dir))


def plot_pr_curve(
    y_true: List[float],
    y_prob: List[float],
    out_dir: Union[str, Path, None] = None,
) -> None:
    """Precision-Recall curve with average precision annotation."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#55A868", linewidth=2, label=f"AP = {ap:.3f}")
    baseline = sum(y_true) / len(y_true)
    ax.axhline(baseline, linestyle="--", color="grey", linewidth=1, label=f"Baseline = {baseline:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Cancer Risk — Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "pr_curve", _resolve_out(out_dir))


def plot_confusion_matrix(
    y_true:  List[int],
    y_pred:  List[int],
    out_dir: Union[str, Path, None] = None,
) -> None:
    """Normalised confusion matrix for smoke type classification (6 classes)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from constants import SMOKE_TYPES

    labels = list(SMOKE_TYPES.values())
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))), normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Smoke Type Classification — Confusion Matrix", fontsize=11, fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm[i,j] > 0.5 else "black")
    plt.tight_layout()
    _save(fig, "confusion_matrix_smoke", _resolve_out(out_dir))


def plot_calibration(
    y_true:  List[float],
    y_prob:  List[float],
    n_bins:  int = 10,
    out_dir: Union[str, Path, None] = None,
) -> None:
    """Calibration curve: predicted cancer probability vs observed frequency."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", color="#4C72B0", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Cancer Risk — Calibration Curve", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save(fig, "calibration_curve", _resolve_out(out_dir))


# ─── Interpretability plots ───────────────────────────────────────────────────

def plot_attention_by_cell_type(
    report:  Dict,
    out_dir: Union[str, Path, None] = None,
) -> None:
    """Bar chart: mean attention weight per cell type — shows which cells the model focuses on."""
    import matplotlib.pyplot as plt

    data   = report["interpretability"]["mean_attention_by_cell_type"]
    labels = list(data.keys())
    values = list(data.values())
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.5f", fontsize=8, padding=2)
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Attention by Cell Type\n(higher = model focuses more on this cell type)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "attention_by_cell_type", _resolve_out(out_dir))


def plot_attention_by_smoke_type(
    report:  Dict,
    out_dir: Union[str, Path, None] = None,
) -> None:
    """Bar chart: mean attention weight per smoke type."""
    import matplotlib.pyplot as plt

    data   = report["interpretability"]["mean_attention_by_smoke_type"]
    labels = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color="#8172B2", edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.5f", fontsize=8, padding=2)
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Attention by Smoke Type\n(reveals which exposure signature drives cancer predictions)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "attention_by_smoke_type", _resolve_out(out_dir))


# ─── UMAP ─────────────────────────────────────────────────────────────────────

def plot_cell_umap(
    embeddings:        np.ndarray,   # [N, 256] from forward_cell z
    smoke_labels:      np.ndarray,   # [N] int
    malignancy_scores: np.ndarray,   # [N] float
    out_dir:           Union[str, Path, None] = None,
) -> None:
    """
    UMAP of cell embeddings.
    Left panel:  coloured by smoke type
    Right panel: coloured by malignancy score
    Novel: first UMAP of lung cells across 6 smoke exposure types.
    """
    import matplotlib.pyplot as plt
    from constants import SMOKE_TYPES
    try:
        from umap import UMAP
    except ImportError:
        print("[viz] umap-learn not installed — skipping UMAP. pip install umap-learn")
        return

    print("[viz] computing UMAP (may take a moment)...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords  = reducer.fit_transform(embeddings)

    smoke_colours = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
    smoke_names   = list(SMOKE_TYPES.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: smoke type
    for idx, (name, col) in enumerate(zip(smoke_names, smoke_colours)):
        mask = smoke_labels == idx
        if mask.sum() > 0:
            ax1.scatter(coords[mask, 0], coords[mask, 1], c=col, s=3,
                        alpha=0.6, label=name, rasterized=True)
    ax1.set_title("Cell Embeddings — Smoke Type", fontsize=11, fontweight="bold")
    ax1.legend(markerscale=4, fontsize=8, loc="upper right")
    ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2")
    ax1.axis("off")

    # Right: malignancy score
    sc = ax2.scatter(coords[:, 0], coords[:, 1], c=malignancy_scores,
                     cmap="RdYlGn_r", s=3, alpha=0.6, vmin=0, vmax=1, rasterized=True)
    plt.colorbar(sc, ax=ax2, label="Malignancy Score")
    ax2.set_title("Cell Embeddings — Malignancy Score", fontsize=11, fontweight="bold")
    ax2.set_xlabel("UMAP 1"); ax2.set_ylabel("UMAP 2")
    ax2.axis("off")

    plt.tight_layout()
    _save(fig, "cell_umap", _resolve_out(out_dir))


# ─── Per-patient profile ──────────────────────────────────────────────────────

def plot_patient_profile(
    result:  Dict,
    out_dir: Union[str, Path, None] = None,
) -> None:
    """
    Three-panel figure for one patient inference result:
      1. Cancer risk gauge (probability + flag)
      2. Smoke type profile (bar chart)
      3. Malignancy score percentiles (box-style)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    out     = _resolve_out(out_dir)
    flag    = result["risk_flag"]
    prob    = result["cancer_probability"]
    flag_colour = {"HIGH": "#C44E52", "MODERATE": "#DD8452", "LOW": "#55A868"}[flag]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Patient: {result['subject_id']}", fontsize=13, fontweight="bold")

    # Panel 1: cancer risk
    ax1.barh([""], [prob], color=flag_colour, height=0.4)
    ax1.barh([""], [1 - prob], left=[prob], color="#E0E0E0", height=0.4)
    ax1.axvline(0.70, color="#C44E52", linestyle="--", linewidth=1, alpha=0.7, label="HIGH threshold")
    ax1.axvline(0.40, color="#DD8452", linestyle="--", linewidth=1, alpha=0.7, label="MODERATE threshold")
    ax1.set_xlim(0, 1)
    ax1.set_title("Cancer Probability", fontsize=10, fontweight="bold")
    ax1.text(prob + 0.02, 0, f"{prob:.3f}\n{flag}", va="center", fontsize=10,
             color=flag_colour, fontweight="bold")
    ax1.legend(fontsize=7); ax1.set_yticks([])

    # Panel 2: smoke profile
    smoke  = result["smoke_profile"]
    labels = list(smoke.keys())
    values = list(smoke.values())
    colours= ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
    bars   = ax2.bar(labels, values, color=colours, edgecolor="white")
    ax2.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax2.set_title("Smoke Type Profile", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Cell Fraction"); ax2.set_ylim(0, 1.15)
    ax2.tick_params(axis="x", rotation=35)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: malignancy percentiles
    pct = result["malignancy_percentiles"]
    ax3.boxplot(
        [[pct["p25"], pct["p50"], pct["p75"]]],
        positions=[0], widths=0.4,
        medianprops={"color": "#C44E52", "linewidth": 2},
        boxprops={"color": "#4C72B0"},
        whiskerprops={"color": "#4C72B0"},
        capprops={"color": "#4C72B0"},
    )
    ax3.scatter([0], [pct["p95"]], color="#C44E52", zorder=5, s=60, label="p95")
    ax3.scatter([0], [result["mean_malignancy"]], color="#DD8452", marker="D",
                zorder=5, s=60, label="mean")
    ax3.set_xlim(-0.5, 0.5); ax3.set_ylim(0, 1.05)
    ax3.set_xticks([]); ax3.set_ylabel("Malignancy Score")
    ax3.set_title("Malignancy Distribution\n(p25/p50/p75/p95)", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    sid = result["subject_id"].replace("/", "_")
    _save(fig, f"patient_profile_{sid}", out)


# ─── Master function ──────────────────────────────────────────────────────────

def plot_all(
    report:          Dict,
    y_true_cancer:   Optional[List[float]] = None,
    y_prob_cancer:   Optional[List[float]] = None,
    y_true_smoke:    Optional[List[int]]   = None,
    y_pred_smoke:    Optional[List[int]]   = None,
    history_dir:     Union[str, Path, None] = None,
    out_dir:         Union[str, Path, None] = None,
) -> None:
    """
    Generate all available plots from pre-computed results.
    Pass raw predictions for ROC/PR/calibration curves.
    """
    out = _resolve_out(out_dir)

    plot_training_history(history_dir, out)
    plot_attention_by_cell_type(report, out)
    plot_attention_by_smoke_type(report, out)

    if y_true_cancer and y_prob_cancer and len(set(y_true_cancer)) > 1:
        plot_roc_curve(y_true_cancer, y_prob_cancer, out_dir=out)
        plot_pr_curve(y_true_cancer, y_prob_cancer, out_dir=out)
        plot_calibration(y_true_cancer, y_prob_cancer, out_dir=out)

    if y_true_smoke and y_pred_smoke:
        plot_confusion_matrix(y_true_smoke, y_pred_smoke, out_dir=out)

    print(f"[viz] all plots saved → {out}/")


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    np.random.seed(42)
    random.seed(42)

    from constants import SMOKE_TYPES, N_SMOKE_CLASSES, CELL_TYPES

    print("=== visualize.py smoke test ===\n")
    out = Path(__file__).parents[1] / "checkpoints" / "plots"

    # Synthetic evaluation report (matches evaluate.py output schema)
    N_SUBJ, N_CELLS = 40, 200
    report = {
        "cell_level": {
            "n_cells": N_CELLS,
            "smoke_type": {"accuracy": 0.72, "macro_f1": 0.68, "per_class": {}},
            "malignancy":  {"roc_auc": 0.81},
        },
        "subject_level": {
            "n_subjects": N_SUBJ,
            "cancer": {"roc_auc": 0.78, "sensitivity": 0.74, "specificity": 0.80},
            "risk_distribution": {"high": 8, "moderate": 18, "low": 14},
        },
        "interpretability": {
            "n_cells_analysed": N_CELLS * N_SUBJ,
            "mean_attention_by_cell_type": {k: random.uniform(0.001, 0.01)
                                             for k in CELL_TYPES.values()},
            "mean_attention_by_smoke_type": {k: random.uniform(0.0005, 0.008)
                                              for k in SMOKE_TYPES.values()},
            "malignancy_attention_correlation": 0.41,
        },
    }

    # Synthetic raw predictions for curve plots
    y_true   = [random.randint(0, 1) for _ in range(N_SUBJ)]
    y_prob   = [random.uniform(0.2, 0.9) for _ in range(N_SUBJ)]
    y_t_smoke= [random.randint(0, N_SMOKE_CLASSES - 1) for _ in range(N_CELLS)]
    y_p_smoke= [random.randint(0, N_SMOKE_CLASSES - 1) for _ in range(N_CELLS)]

    # Synthetic patient result
    patient = {
        "subject_id":           "patient_test",
        "cancer_probability":   0.78,
        "risk_flag":            "HIGH",
        "top5_cells":           [12, 45, 3, 99, 7],
        "smoke_profile":        {k: random.uniform(0, 1) for k in SMOKE_TYPES.values()},
        "dominant_smoke_type":  "cigarette",
        "mean_malignancy":      0.62,
        "malignancy_percentiles": {"p25": 0.41, "p50": 0.64, "p75": 0.79, "p95": 0.91},
        "attention_weights":    [random.random() for _ in range(50)],
    }
    # Normalise smoke profile
    total = sum(patient["smoke_profile"].values())
    patient["smoke_profile"] = {k: v / total for k, v in patient["smoke_profile"].items()}

    plot_all(report, y_true, y_prob, y_t_smoke, y_p_smoke, out_dir=out)
    plot_patient_profile(patient, out_dir=out)

    saved = list(out.glob("*.png"))
    assert len(saved) >= 6, f"Expected >= 6 plots, got {len(saved)}"
    for p in sorted(saved):
        print(f"  ✓ {p.name}")

    print("\n=== PASSED ===")