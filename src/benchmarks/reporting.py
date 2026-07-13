"""
benchmarks/reporting.py — statistical comparison, immutable run artifacts,
and the Markdown/CSV report.

Every artifacts/benchmarks/<run_id>/ directory is written once and never
overwritten (see new_run_dir) — a run_id collision raises rather than
silently clobbering a previous, possibly-audited run.
"""

import csv
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .context import get_git_sha
from .metrics import bootstrap_ci


# ─── Statistical comparison ────────────────────────────────────────────────────

def _paired_values(folds_a: List[dict], folds_b: List[dict], metric_key: str):
    """Returns [(seed, value_a, value_b), ...] for folds where both models
    produced a defined value, matched by (seed, fold) — keeps the seed so
    callers can aggregate to the seed level (see compare_models)."""
    idx_a = {(f["seed"], f["fold"]): f.get(metric_key) for f in folds_a}
    idx_b = {(f["seed"], f["fold"]): f.get(metric_key) for f in folds_b}
    pairs = []
    for key in sorted(set(idx_a) & set(idx_b)):
        va, vb = idx_a[key], idx_b[key]
        if va is not None and vb is not None:
            pairs.append((key[0], va, vb))
    return pairs


def compare_models(
    results: Dict[str, dict], metric_key: str, model_a: str, model_b: str, seed: int = 42,
) -> Dict:
    """
    Paired comparison of model_a vs model_b on metric_key, using only folds
    where BOTH models produced a defined value. Never claims superiority
    from a numerically-higher mean alone — see summarize_comparison for the
    compound criterion the spec requires.

    Two confidence intervals are reported for the paired difference:
      - `ci_diff` (resampling_unit="fold"): bootstraps raw per-fold diffs.
        Folds from repeated seeds over the same subject pool overlap (a
        subject reappears across many folds), so this treats non-independent
        observations as independent — descriptive only, not rigorous.
      - `ci_diff_by_seed` (resampling_unit="seed"): averages each seed's
        paired diffs into one per-seed diff first, then bootstraps across
        seeds — genuinely independent samples, at the cost of very few of
        them (one per seed). Prefer this one; `summarize_comparison` does.
    """
    pairs = _paired_values(results[model_a]["folds"], results[model_b]["folds"], metric_key)
    if not pairs:
        return {"model_a": model_a, "model_b": model_b, "metric": metric_key,
                "n_pairs": 0, "reason": "no fold had a defined value for both models"}

    seeds_seen = [p[0] for p in pairs]
    diffs = [a - b for _, a, b in pairs]
    wins  = sum(1 for d in diffs if d > 1e-9)
    ties  = sum(1 for d in diffs if abs(d) <= 1e-9)
    losses= sum(1 for d in diffs if d < -1e-9)
    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    effect_size = mean_diff / std_diff if std_diff > 0 else None

    by_seed: Dict[int, list] = {}
    for s, d in zip(seeds_seen, diffs):
        by_seed.setdefault(s, []).append(d)
    per_seed_diffs = [float(np.mean(v)) for v in by_seed.values()]
    ci_diff_by_seed = bootstrap_ci(per_seed_diffs, seed=seed) if len(per_seed_diffs) >= 2 else None

    exploratory_p = None
    if len(diffs) >= 5 and any(d != 0 for d in diffs):
        try:
            from scipy.stats import wilcoxon
            _, exploratory_p = wilcoxon(diffs)
            exploratory_p = float(exploratory_p)
        except Exception:
            pass

    return {
        "model_a": model_a, "model_b": model_b, "metric": metric_key,
        "n_pairs": len(pairs), "n_seeds": len(by_seed),
        "wins_a": wins, "ties": ties, "losses_a": losses,
        "mean_diff": mean_diff, "std_diff": std_diff, "effect_size_cohens_d": effect_size,
        "ci_diff": bootstrap_ci(diffs, seed=seed),
        "ci_diff_by_seed": ci_diff_by_seed,
        "exploratory_wilcoxon_p": exploratory_p,
        "exploratory_wilcoxon_note": "does not account for overlapping folds across repeated seeds — exploratory only",
    }


def summarize_comparison(comparison: Dict, min_win_fraction: float = 0.7) -> Dict:
    """
    "Meaningfully better" requires ALL of: enough paired folds, a
    consistent win direction across them, and a CI on the paired difference
    that excludes zero. A higher mean alone never qualifies.

    Prefers `ci_diff_by_seed` (bootstraps independent per-seed diffs, see
    compare_models) over `ci_diff` (bootstraps overlapping raw fold diffs,
    descriptive only) whenever >=2 seeds were run — falls back to the
    fold-level CI, explicitly marked non-rigorous, only when just one seed
    is available (no independent-seed evidence exists yet).
    """
    if comparison.get("n_pairs", 0) < 3:
        return {"meaningfully_better": False, "reason": "fewer than 3 paired folds — not enough evidence"}
    win_fraction = comparison["wins_a"] / comparison["n_pairs"]
    ci = comparison.get("ci_diff_by_seed") or comparison.get("ci_diff")
    ci_is_rigorous = comparison.get("ci_diff_by_seed") is not None
    ci_excludes_zero = ci is not None and (ci["lo"] > 0 or ci["hi"] < 0)
    if ci is not None and not ci_is_rigorous:
        ci_excludes_zero = False  # a single-seed fold-level CI is not sufficient evidence on its own
    if win_fraction >= min_win_fraction and ci_excludes_zero and comparison["mean_diff"] > 0:
        return {"meaningfully_better": True,
                "reason": f"won {win_fraction:.0%} of paired folds, seed-level CI excludes zero "
                          f"({comparison.get('n_seeds', 1)} independent seeds)"}
    if not ci_is_rigorous:
        return {"meaningfully_better": False,
                "reason": f"only 1 seed run — no independent-seed evidence yet (win_fraction={win_fraction:.0%} "
                          "over overlapping folds is not sufficient on its own); run >=2 seeds"}
    return {
        "meaningfully_better": False,
        "reason": f"win_fraction={win_fraction:.0%} (need >={min_win_fraction:.0%}), "
                  f"seed_level_ci_excludes_zero={ci_excludes_zero}",
    }


# ─── Immutable run artifacts ────────────────────────────────────────────────────

def new_run_dir(base_dir: str = "artifacts/benchmarks", run_id: Optional[str] = None) -> Path:
    run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    path = Path(base_dir) / run_id
    if path.exists():
        raise FileExistsError(f"Run directory {path} already exists — run_ids must be immutable/unique.")
    for sub in ("predictions", "metrics", "calibration"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def write_csv_table(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    # Different models' fold records carry different keys (e.g. the neural
    # adapter's cell-capping fields aren't present on baseline records) —
    # using only rows[0]'s keys as the fieldname set crashes DictWriter the
    # moment a later row has a key the first row didn't. The union of every
    # row's keys, in first-seen order, covers all of them; a row missing a
    # given key gets restval (empty), not a crash.
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def cv_results_to_csv_rows(cv_report: Dict) -> List[dict]:
    rows = []
    for model_name, model_result in cv_report["results"].items():
        for fold in model_result["folds"]:
            row = {"model": model_name}
            row.update({k: v for k, v in fold.items() if not isinstance(v, (dict, list))})
            rows.append(row)
    return rows


# ─── Markdown report ────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None:
        return "undefined"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def generate_markdown_report(
    run_dir: Path, context, run_manifest: Dict, eligibility: Dict, cv_reports: Dict,
    comparisons: List[Dict], calibration_report: Optional[Dict], synthetic: bool,
) -> str:
    lines = []
    lines.append(f"# Benchmark report — {run_manifest['run_id']}")
    lines.append("")
    if synthetic:
        lines.append("**⚠ SYNTHETIC RUN — validates software only, not model accuracy. "
                      "Do not cite these numbers as scientific results.**")
    else:
        lines.append("**Real-data run.** See limitations at the end before citing any number.")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(f"- git commit: `{run_manifest.get('git_sha')}`")
    lines.append(f"- seeds: {run_manifest.get('seeds')}")
    lines.append(f"- split fingerprint: `{run_manifest.get('split_fingerprint')}`")
    lines.append(f"- transductive batch correction: {run_manifest.get('transductive_batch_correction')}")
    lines.append(f"- effective label mapping K: {run_manifest.get('num_smoke_classes')}")
    lines.append(f"- dataset/source summary: {run_manifest.get('dataset_source_summary')}")
    lines.append("")

    lines.append("## Eligibility")
    for task, report in eligibility.items():
        d = report.to_dict() if hasattr(report, "to_dict") else report
        lines.append(f"- **{task}**: {d['status']}"
                      + (f" — {'; '.join(d['reasons'])}" if d["reasons"] else ""))
    lines.append("")

    for task, cv_report in cv_reports.items():
        lines.append(f"## {task} — primary metric: `{cv_report['primary_metric']}`")
        lines.append("")
        lines.append("| Model | mean | std | median | 95% CI | n valid | n undefined |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, result in cv_report["results"].items():
            m = result.get(cv_report["primary_metric"], {})
            ci = m.get("ci")
            ci_str = f"[{ci['lo']:.3f}, {ci['hi']:.3f}]" if ci else "undefined"
            lines.append(f"| {name} | {_fmt(m.get('mean'))} | {_fmt(m.get('std'))} | "
                          f"{_fmt(m.get('median'))} | {ci_str} | {m.get('n_valid')} | {m.get('n_undefined')} |")
        lines.append("")

    if comparisons:
        lines.append("## Statistical comparisons")
        lines.append("")
        lines.append("| A | B | metric | n pairs | mean diff | effect size | meaningfully better? |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in comparisons:
            summary = summarize_comparison(c) if "n_pairs" in c and c["n_pairs"] > 0 else {"meaningfully_better": False, "reason": c.get("reason", "")}
            verdict = "YES" if summary["meaningfully_better"] else "no"
            lines.append(f"| {c['model_a']} | {c['model_b']} | {c['metric']} | {c.get('n_pairs', 0)} | "
                          f"{_fmt(c.get('mean_diff'))} | {_fmt(c.get('effect_size_cohens_d'))} | {verdict} — {summary['reason']} |")
        lines.append("")

    if calibration_report:
        lines.append("## Calibration and frozen threshold (test evaluation)")
        lines.append(f"```json\n{json.dumps(calibration_report, indent=2, default=str)}\n```")
        lines.append("")

    lines.append("## Limitations")
    lines.append(
        "- Passing tests does not prove model accuracy.\n"
        "- Synthetic runs validate software only; no clinical claim is made anywhere in this report.\n"
        "- Attention weights are an interpretability aid, not a causal explanation.\n"
        "- Test data was untouched until the single frozen final evaluation above (if present).\n"
        "- MultiSmokeCancerNet is not considered superior to a baseline unless it beats it on "
        "matched subject-level folds with a consistent win rate and a CI excluding zero (see "
        "summarize_comparison).\n"
        "- Grouped CV here searches/fits over the train+val subject pool only; per-fold "
        "HVG/scaling refitting (vs. reusing the context's already train-fit preprocessing "
        "artifact) is not yet implemented — a documented simplification, not a leakage risk, "
        "since the artifact was fit on the ORIGINAL train split, a subset of every CV fold's "
        "train partition."
    )
    return "\n".join(lines)


def write_benchmark_report(
    run_dir: Path, context, run_manifest: Dict, eligibility: Dict, cv_reports: Dict,
    comparisons: List[Dict], calibration_report: Optional[Dict] = None, synthetic: bool = False,
) -> None:
    write_json(run_dir / "run_manifest.json", run_manifest)
    write_json(run_dir / "eligibility.json", {k: v.to_dict() if hasattr(v, "to_dict") else v
                                               for k, v in eligibility.items()})
    write_json(run_dir / "comparisons.json", comparisons)
    summary = {
        task: {name: {k: v for k, v in result.items() if k != "folds"} for name, result in cv["results"].items()}
        for task, cv in cv_reports.items()
    }
    write_json(run_dir / "summary.json", summary)
    if calibration_report:
        write_json(run_dir / "calibration" / "frozen_policy.json", calibration_report)
        if "hyperparameter_search" in calibration_report:
            # Candidates, inner-CV scores, and the final selection for the
            # frozen-test model — see hyperparameter_search.py. Only ever
            # populated for the classical baseline nested-selection path
            # (see README's Benchmarking framework limitations for what
            # this does NOT yet cover: per-fold CV selection, neural/MIL).
            write_json(run_dir / "hyperparameters.json", calibration_report["hyperparameter_search"])

    for task, cv_report in cv_reports.items():
        write_json(run_dir / "metrics" / f"{task}_folds.json", cv_report)
        write_csv_table(run_dir / "metrics" / f"{task}_folds.csv", cv_results_to_csv_rows(cv_report))
        # folds.json: outer subject partitions for every fold this task's CV
        # ran (train/val subject lists + fingerprints) — the per-fold
        # records already carry fit_subject_ids/preprocessing_fingerprint;
        # this file collects just that partition/identity info per fold in
        # one place for audit, without duplicating the full metric report.
        folds_summary = {
            name: [
                {k: f.get(k) for k in (
                    "seed", "fold", "fit_subject_ids", "preprocessing_fingerprint",
                    "gene_list_n", "classes_absent_from_val", "stratified",
                )}
                for f in result.get("folds", [])
            ]
            for name, result in cv_report.get("results", {}).items()
        }
        write_json(run_dir / "metrics" / f"{task}_folds_partitions.json", folds_summary)

    report_md = generate_markdown_report(
        run_dir, context, run_manifest, eligibility, cv_reports, comparisons, calibration_report, synthetic,
    )
    with open(run_dir / "report.md", "w") as f:
        f.write(report_md)
    print(f"[benchmarks] report written -> {run_dir}/report.md")
