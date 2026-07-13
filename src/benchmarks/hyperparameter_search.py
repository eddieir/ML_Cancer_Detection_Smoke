"""
benchmarks/hyperparameter_search.py — leakage-free nested hyperparameter
selection for classical baselines.

Replaces the previously-unused SMOKE_SEARCH_SPACE/CANCER_SEARCH_SPACE
constants in baselines.py (declared but never consulted by any selection
code) with real inner grouped-CV selection. The caller is responsible for
passing X/y/groups that already belong to ONE outer partition (e.g. the
outer-train split, or one CV fold's train subjects) — this module never
sees, and has no way to see, outer validation or test data; it only ever
subdivides whatever it is given into inner folds via grouped_kfold.

A model with no declared search space (e.g. "majority"/"prevalence", which
take no tunable hyperparameters) is recorded as {"no_search_space": True},
never silently skipped without a trace.

Scope: this module selects hyperparameters for the sklearn-based classical
baselines only. Nested selection for the neural/MIL models (which would
mean a full inner-CV training loop per candidate — many times the cost of
one Phase 1 + Phase 2 training run) is NOT implemented here; see
README.md/ARCHITECTURE.md for this explicitly documented as open work.
"""

import itertools
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from data.splitting import grouped_kfold


def build_param_grid(search_space: Dict[str, Sequence]) -> List[Dict]:
    """Cartesian product of a {param_name: [values...]} search space. Empty
    dict in -> empty list out (no candidates to select among)."""
    if not search_space:
        return []
    keys = sorted(search_space.keys())
    value_lists = [search_space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def select_hyperparameters_nested(
    baseline_cls, search_space: Dict[str, Sequence],
    X: np.ndarray, y: np.ndarray, groups: Sequence,
    score_fn: Callable[[object, np.ndarray, np.ndarray], Optional[float]],
    seed: int = 42, n_inner_folds: int = 3,
) -> Dict:
    """
    Select hyperparameters for baseline_cls using inner grouped-CV computed
    ENTIRELY from (X, y, groups) — the caller must never pass outer
    validation/test rows here (see module docstring). score_fn(fitted_model,
    X_inner_val, y_inner_val) -> float | None; None means "undefined for
    this inner fold" (e.g. only one class present) and is excluded from the
    candidate's mean rather than crashing or being coerced to 0.

    Returns:
      no_search_space: bool — True if search_space was empty (model has no
        tunable hyperparameters; this is recorded explicitly, not silently
        skipped)
      selected_params: the chosen candidate's kwargs (empty dict if
        no_search_space)
      candidates: [{"params", "inner_scores", "inner_score_mean"}, ...] for
        every candidate tried, so a run's artifact shows the full search,
        not just the winner
      seed / n_inner_folds: for reproducibility
    """
    candidates = build_param_grid(search_space)
    if not candidates:
        return {"no_search_space": True, "selected_params": {}, "candidates": [],
                "seed": seed, "n_inner_folds": n_inner_folds}

    groups_arr = np.asarray([str(g) for g in groups])
    y_arr = np.asarray(y)
    try:
        inner_folds = grouped_kfold(groups_arr, y_arr, n_folds=n_inner_folds, seed=seed)
    except ValueError as e:
        # Too few independent subjects/groups for inner CV (e.g. a very
        # small fold) — explicitly recorded as untuned, never silently
        # defaulting without a trace.
        return {"no_search_space": False, "selected_params": candidates[0], "candidates": [],
                "seed": seed, "n_inner_folds": n_inner_folds,
                "note": f"inner grouped_kfold unavailable ({e}) — defaulting to the first "
                        "declared candidate, NOT selected by inner-CV evidence."}

    results = []
    for params in candidates:
        fold_scores = []
        for fold in inner_folds:
            train_mask = np.isin(groups_arr, fold["train"])
            val_mask = np.isin(groups_arr, fold["val"])
            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue
            model = baseline_cls(**params).fit(X[train_mask], y_arr[train_mask], seed=seed)
            score = score_fn(model, X[val_mask], y_arr[val_mask])
            if score is not None:
                fold_scores.append(float(score))
        results.append({
            "params": params, "inner_scores": fold_scores,
            "inner_score_mean": float(np.mean(fold_scores)) if fold_scores else None,
        })

    scored = [r for r in results if r["inner_score_mean"] is not None]
    if not scored:
        return {"no_search_space": False, "selected_params": candidates[0], "candidates": results,
                "seed": seed, "n_inner_folds": n_inner_folds,
                "note": "no candidate produced a defined inner score on any inner fold — "
                        "defaulting to the first declared candidate, NOT selected by evidence."}

    best = max(scored, key=lambda r: r["inner_score_mean"])
    return {
        "no_search_space": False, "selected_params": best["params"], "candidates": results,
        "seed": seed, "n_inner_folds": n_inner_folds,
    }
