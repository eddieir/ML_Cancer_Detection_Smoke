"""
Renders every plot in src/visualize.py from the artifacts that train.py,
evaluate.py, and inference.py already wrote to checkpoints/ during a demo
run — no re-running the model, purely reading and rendering.

Usage: python3 scripts/generate_plots.py [--out checkpoints/plots]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visualize import plot_all, plot_patient_profile  # noqa: E402

CKPT_DIR = ROOT / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=CKPT_DIR / "plots")
    args = parser.parse_args()

    report_path = CKPT_DIR / "evaluation_report.json"
    raw_path    = CKPT_DIR / "evaluation_raw.json"

    if not report_path.exists():
        print(f"[generate_plots] {report_path} not found — run src/evaluate.py first, skipping.")
        return 0

    report = json.loads(report_path.read_text())
    raw    = json.loads(raw_path.read_text()) if raw_path.exists() else {}

    plot_all(
        report,
        y_true_cancer=raw.get("y_true_cancer"),
        y_prob_cancer=raw.get("y_prob_cancer"),
        y_true_smoke=raw.get("y_true_smoke"),
        y_pred_smoke=raw.get("y_pred_smoke"),
        history_dir=CKPT_DIR,
        out_dir=args.out,
    )

    inference_path = CKPT_DIR / "inference_results.json"
    if inference_path.exists():
        results = json.loads(inference_path.read_text())
        patient = results[0] if isinstance(results, list) else results
        plot_patient_profile(patient, out_dir=args.out)
    else:
        print(f"[generate_plots] {inference_path} not found — skipping patient profile.")

    saved = sorted(args.out.glob("*.png"))
    print(f"[generate_plots] {len(saved)} plot(s) → {args.out}/")
    for p in saved:
        print(f"  - {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
