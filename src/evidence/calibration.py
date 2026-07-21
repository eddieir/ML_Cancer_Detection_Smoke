"""
evidence/calibration.py — Step 14: frozen calibration, thresholding, and
exploratory decision-curve analysis for Phase 7 evidence reports.

This module does not re-implement Platt scaling or isotonic regression —
it wraps the actual fitting code already in benchmarks/calibration.py
(itself sklearn's LogisticRegression / IsotonicRegression) with Phase 7's
identity/fingerprint/freeze semantics: an immutable, sha256-fingerprinted
FrozenCalibrationThresholdArtifact that records exactly which method,
parameters, and development subjects produced it.

Two functions form the entire contract:

  fit_frozen_calibration_and_threshold(development_oof_predictions, ...)
      The ONLY function in this module allowed to fit anything. Its only
      data input is `development_oof_predictions` — a DevelopmentOOFPredictions
      instance built from development out-of-fold predictions. There is no
      parameter for test data anywhere in its signature.

  apply_frozen_calibration_and_threshold(artifact, y_prob)
      The ONLY function in this module allowed to transform probabilities
      into a calibrated probability + decision. It has no fitting
      parameters at all — no y_true, no refit flag — so it is structurally
      impossible to refit or reselect a threshold on new (e.g. test) data
      through this code path.

decision_curve_analysis() is a small, separately-labeled exploratory
helper. Its output must never flow into evidence_level or
clinical_readiness claims — see its docstring.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from benchmarks.calibration import FrozenCalibrator, fit_calibration, select_threshold
from benchmarks.metrics import cancer_metrics_at_threshold

DEFAULT_EVIDENCE_CONFIG_PATH = "configs/evidence.yaml"

VALID_CALIBRATION_METHODS = ("uncalibrated", "platt", "isotonic", "auto")
VALID_THRESHOLD_METHODS = (
    "youden", "balanced_accuracy", "sensitivity_constrained",
    "specificity_constrained", "predeclared_fixed",
)

# Fallback defaults if configs/evidence.yaml cannot be read for any reason
# — never silently permit isotonic below this floor.
_FALLBACK_MIN_ISOTONIC_N = 200
_FALLBACK_MIN_PLATT_N = 20


class CalibrationPolicyError(ValueError):
    """Raised for an unsupported calibration_method/threshold_method, a
    missing required parameter for the requested threshold method (e.g.
    sensitivity_constrained without min_sensitivity), or a malformed
    development_oof_predictions input."""


def load_calibration_policy(config_path: str = DEFAULT_EVIDENCE_CONFIG_PATH) -> Dict[str, Any]:
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        policy = raw.get("calibration_policy") or {}
    except (OSError, yaml.YAMLError):
        policy = {}
    return {
        "min_isotonic_n": int(policy.get("min_isotonic_n", _FALLBACK_MIN_ISOTONIC_N)),
        "min_platt_n": int(policy.get("min_platt_n", _FALLBACK_MIN_PLATT_N)),
    }


@dataclass(frozen=True)
class DevelopmentOOFPredictions:
    """Development out-of-fold predictions — the only data input
    fit_frozen_calibration_and_threshold() is allowed to accept. Building
    this from anything other than genuine development-fold-held-out
    predictions (e.g. in-fold/training predictions) defeats the entire
    point of calibrating and selecting a threshold on unseen-during-fit
    data; this dataclass cannot detect that misuse itself, but naming the
    parameter this way makes the intended contract explicit at every call
    site."""

    subject_ids: Tuple[str, ...]
    y_true: Tuple[int, ...]
    y_prob: Tuple[float, ...]

    def __post_init__(self):
        n = len(self.subject_ids)
        if n == 0:
            raise CalibrationPolicyError("DevelopmentOOFPredictions cannot be empty.")
        if len(self.y_true) != n or len(self.y_prob) != n:
            raise CalibrationPolicyError(
                f"DevelopmentOOFPredictions: subject_ids ({n}), y_true "
                f"({len(self.y_true)}), y_prob ({len(self.y_prob)}) must have equal length."
            )
        if len(set(self.subject_ids)) != n:
            raise CalibrationPolicyError(
                "DevelopmentOOFPredictions: subject_ids contains duplicates — a subject may "
                "appear at most once in the OOF predictions used for calibration/threshold "
                "selection."
            )
        for y in self.y_true:
            if y not in (0, 1):
                raise CalibrationPolicyError(f"DevelopmentOOFPredictions: y_true must be binary 0/1, got {y!r}")
        for p in self.y_prob:
            if not (0.0 <= float(p) <= 1.0):
                raise CalibrationPolicyError(f"DevelopmentOOFPredictions: y_prob must be in [0, 1], got {p!r}")

    def subject_ids_fingerprint(self) -> str:
        blob = json.dumps(sorted(str(s) for s in self.subject_ids)).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def y_true_array(self) -> np.ndarray:
        return np.asarray(self.y_true, dtype=int)

    def y_prob_array(self) -> np.ndarray:
        return np.asarray(self.y_prob, dtype=float)


def build_development_oof_predictions(
    subject_ids: Sequence[str], y_true: Sequence[int], y_prob: Sequence[float],
) -> DevelopmentOOFPredictions:
    return DevelopmentOOFPredictions(
        subject_ids=tuple(str(s) for s in subject_ids),
        y_true=tuple(int(y) for y in y_true),
        y_prob=tuple(float(p) for p in y_prob),
    )


@dataclass(frozen=True)
class FrozenCalibrationThresholdArtifact:
    calibration_method: str
    calibration_parameters: Dict[str, Any]
    calibration_reason: Optional[str]
    threshold_method: str
    threshold_value: float
    threshold_objective: str
    threshold_constraints: Dict[str, Any]
    selection_subject_ids: Tuple[str, ...]
    selection_subject_ids_fingerprint: str
    random_seed: int
    fingerprint: str
    _calibrator: FrozenCalibrator = field(repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_method": self.calibration_method,
            "calibration_parameters": self.calibration_parameters,
            "calibration_reason": self.calibration_reason,
            "threshold_method": self.threshold_method,
            "threshold_value": self.threshold_value,
            "threshold_objective": self.threshold_objective,
            "threshold_constraints": self.threshold_constraints,
            "selection_subject_ids": list(self.selection_subject_ids),
            "selection_subject_ids_fingerprint": self.selection_subject_ids_fingerprint,
            "random_seed": self.random_seed,
            "fingerprint": self.fingerprint,
        }


def _artifact_fingerprint(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _fit_calibrator(
    y_true: np.ndarray, y_prob: np.ndarray, method: str, policy: Dict[str, Any],
) -> Tuple[FrozenCalibrator, str, Dict[str, Any], Optional[str]]:
    """Reuses benchmarks.calibration.fit_calibration/FrozenCalibrator —
    never re-implements Platt/isotonic fitting. Returns
    (calibrator, resolved_method_name, calibration_parameters, reason)."""
    n = len(y_true)

    if method == "uncalibrated":
        calibrator = fit_calibration(y_true, y_prob, method="none")
        return calibrator, "uncalibrated", {}, calibrator.reason

    if method == "platt":
        if n < policy["min_platt_n"]:
            calibrator = FrozenCalibrator(
                "none",
                reason=f"only {n} development OOF subjects (< {policy['min_platt_n']} minimum for platt scaling)",
            )
            return calibrator, "uncalibrated", {}, calibrator.reason
        calibrator = fit_calibration(y_true, y_prob, method="sigmoid")
        resolved = "platt" if calibrator.method == "sigmoid" else "uncalibrated"
        params = {"coef": None, "intercept": None}
        if calibrator.method == "sigmoid" and calibrator.model is not None:
            params = {
                "coef": float(calibrator.model.coef_.ravel()[0]),
                "intercept": float(calibrator.model.intercept_.ravel()[0]),
            }
        return calibrator, resolved, params, calibrator.reason

    if method == "isotonic":
        if n < policy["min_isotonic_n"]:
            calibrator = FrozenCalibrator(
                "none",
                reason=(
                    f"only {n} development OOF subjects (< configured minimum "
                    f"{policy['min_isotonic_n']} for isotonic regression) — isotonic regression "
                    "is not offered below this sample size; falling back to uncalibrated."
                ),
            )
            return calibrator, "uncalibrated", {}, calibrator.reason
        calibrator = fit_calibration(y_true, y_prob, method="isotonic")
        resolved = "isotonic" if calibrator.method == "isotonic" else "uncalibrated"
        params = {}
        if calibrator.method == "isotonic" and calibrator.model is not None:
            params = {"n_thresholds": int(len(calibrator.model.X_thresholds_))}
        return calibrator, resolved, params, calibrator.reason

    if method == "auto":
        if n >= policy["min_isotonic_n"]:
            return _fit_calibrator(y_true, y_prob, "isotonic", policy)
        if n >= policy["min_platt_n"]:
            return _fit_calibrator(y_true, y_prob, "platt", policy)
        return _fit_calibrator(y_true, y_prob, "uncalibrated", policy)

    raise CalibrationPolicyError(
        f"unknown calibration_method={method!r} — must be one of {VALID_CALIBRATION_METHODS}"
    )


def _select_threshold(
    y_true: np.ndarray, y_prob_calibrated: np.ndarray, method: str,
    *, min_sensitivity: Optional[float], min_specificity: Optional[float],
    fixed_threshold: Optional[float],
) -> Tuple[float, str, Dict[str, Any], Optional[str]]:
    """Reuses benchmarks.calibration.select_threshold for youden/
    balanced_accuracy/sensitivity_constrained; implements
    specificity_constrained and predeclared_fixed locally (neither exists
    upstream) using the same threshold-grid pattern and
    benchmarks.metrics.cancer_metrics_at_threshold — never a new metric
    implementation."""
    if method == "predeclared_fixed":
        if fixed_threshold is None:
            raise CalibrationPolicyError("threshold_method='predeclared_fixed' requires fixed_threshold.")
        return (
            float(fixed_threshold), "predeclared_fixed",
            {"fixed_threshold": float(fixed_threshold)},
            "predeclared fixed threshold — no development-data search performed",
        )

    if method in ("youden", "balanced_accuracy"):
        result = select_threshold(y_true, y_prob_calibrated, strategy=method)
        return result["threshold"], method, {}, result["reason"]

    if method == "sensitivity_constrained":
        if min_sensitivity is None:
            raise CalibrationPolicyError("threshold_method='sensitivity_constrained' requires min_sensitivity.")
        result = select_threshold(
            y_true, y_prob_calibrated, strategy="sensitivity_constrained", min_sensitivity=min_sensitivity,
        )
        return result["threshold"], method, {"min_sensitivity": float(min_sensitivity)}, result["reason"]

    if method == "specificity_constrained":
        if min_specificity is None:
            raise CalibrationPolicyError("threshold_method='specificity_constrained' requires min_specificity.")
        if len(set(y_true.tolist())) < 2:
            return (
                0.5, method, {"min_specificity": float(min_specificity)},
                "development OOF set has only one class — threshold selection NOT_EVALUABLE, using fixed 0.5",
            )
        grid = np.unique(np.clip(y_prob_calibrated, 1e-6, 1 - 1e-6))
        best_t, best_score = 0.5, -np.inf
        for t in grid:
            m = cancer_metrics_at_threshold(y_true, y_prob_calibrated, float(t))
            if m["specificity"] is None or m["specificity"] < min_specificity:
                continue
            if m["sensitivity"] is not None and m["sensitivity"] > best_score:
                best_score, best_t = m["sensitivity"], float(t)
        if best_score == -np.inf:
            return (
                0.5, method, {"min_specificity": float(min_specificity)},
                f"no threshold in the development OOF grid achieved specificity >= {min_specificity}",
            )
        return best_t, method, {"min_specificity": float(min_specificity)}, None

    raise CalibrationPolicyError(
        f"unknown threshold_method={method!r} — must be one of {VALID_THRESHOLD_METHODS}"
    )


def fit_frozen_calibration_and_threshold(
    development_oof_predictions: DevelopmentOOFPredictions,
    *,
    calibration_method: str = "auto",
    threshold_method: str = "youden",
    min_sensitivity: Optional[float] = None,
    min_specificity: Optional[float] = None,
    fixed_threshold: Optional[float] = None,
    random_seed: int = 0,
    config_path: str = DEFAULT_EVIDENCE_CONFIG_PATH,
) -> FrozenCalibrationThresholdArtifact:
    """Fits calibration and selects a threshold using ONLY
    `development_oof_predictions` — the sole data input this function
    accepts. Returns an immutable, sha256-fingerprinted artifact; nothing
    about the artifact can be mutated after construction (it is a frozen
    dataclass), and apply_frozen_calibration_and_threshold() below has no
    parameter that could refit it."""
    if not isinstance(development_oof_predictions, DevelopmentOOFPredictions):
        raise CalibrationPolicyError(
            "fit_frozen_calibration_and_threshold requires a DevelopmentOOFPredictions instance "
            "— build one with build_development_oof_predictions()."
        )
    if calibration_method not in VALID_CALIBRATION_METHODS:
        raise CalibrationPolicyError(
            f"unknown calibration_method={calibration_method!r} — must be one of {VALID_CALIBRATION_METHODS}"
        )
    if threshold_method not in VALID_THRESHOLD_METHODS:
        raise CalibrationPolicyError(
            f"unknown threshold_method={threshold_method!r} — must be one of {VALID_THRESHOLD_METHODS}"
        )

    policy = load_calibration_policy(config_path)
    y_true = development_oof_predictions.y_true_array()
    y_prob = development_oof_predictions.y_prob_array()

    calibrator, resolved_method, calib_params, calib_reason = _fit_calibrator(
        y_true, y_prob, calibration_method, policy,
    )
    y_prob_calibrated = calibrator.apply(y_prob)

    threshold_value, resolved_threshold_method, constraints, threshold_reason = _select_threshold(
        y_true, y_prob_calibrated, threshold_method,
        min_sensitivity=min_sensitivity, min_specificity=min_specificity,
        fixed_threshold=fixed_threshold,
    )

    subject_ids = tuple(sorted(str(s) for s in development_oof_predictions.subject_ids))
    payload = {
        "calibration_method": resolved_method,
        "calibration_parameters": calib_params,
        "calibration_reason": calib_reason,
        "threshold_method": resolved_threshold_method,
        "threshold_value": threshold_value,
        "threshold_objective": threshold_method,
        "threshold_constraints": constraints,
        "selection_subject_ids": list(subject_ids),
        "selection_subject_ids_fingerprint": development_oof_predictions.subject_ids_fingerprint(),
        "random_seed": random_seed,
    }
    fingerprint = _artifact_fingerprint(payload)

    return FrozenCalibrationThresholdArtifact(
        calibration_method=resolved_method,
        calibration_parameters=calib_params,
        calibration_reason=calib_reason,
        threshold_method=resolved_threshold_method,
        threshold_value=threshold_value,
        threshold_objective=threshold_method,
        threshold_constraints=constraints,
        selection_subject_ids=subject_ids,
        selection_subject_ids_fingerprint=development_oof_predictions.subject_ids_fingerprint(),
        random_seed=random_seed,
        fingerprint=fingerprint,
        _calibrator=calibrator,
    )


def apply_frozen_calibration_and_threshold(
    artifact: FrozenCalibrationThresholdArtifact, y_prob,
) -> Dict[str, Any]:
    """The only sanctioned way to run new probabilities through a frozen
    artifact. No fitting parameter exists on this function's signature —
    no y_true, no refit flag, nothing that could re-select a threshold —
    so it is structurally impossible to refit on test data through this
    code path. Purely: calibrate probabilities, then apply the frozen
    threshold to produce a hard decision."""
    if not isinstance(artifact, FrozenCalibrationThresholdArtifact):
        raise CalibrationPolicyError(
            "apply_frozen_calibration_and_threshold requires a FrozenCalibrationThresholdArtifact "
            "produced by fit_frozen_calibration_and_threshold()."
        )
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_prob_calibrated = artifact._calibrator.apply(y_prob_arr)
    y_pred = (y_prob_calibrated >= artifact.threshold_value).astype(int)
    return {
        "y_prob_calibrated": y_prob_calibrated.tolist(),
        "y_pred": y_pred.tolist(),
        "threshold": artifact.threshold_value,
        "calibration_method": artifact.calibration_method,
        "artifact_fingerprint": artifact.fingerprint,
    }


# ─── Decision-curve analysis — exploratory only ────────────────────────────

def decision_curve_analysis(
    y_true: Sequence[int], y_prob: Sequence[float],
    threshold_range: Sequence[float] = tuple(round(t, 2) for t in np.arange(0.01, 1.0, 0.01)),
) -> Dict[str, Any]:
    """Computes net benefit across `threshold_range` for three fixed,
    explicit assumptions: treat-all, treat-none, and the model at each
    threshold. Net benefit at threshold pt is the standard Vickers/Elkin
    formula: TP/n - FP/n * (pt / (1 - pt)).

    This is EXPLORATORY ANALYSIS ONLY. It quantifies a hypothetical
    trade-off under stated assumptions about relative harms of a false
    positive vs a false negative at each threshold — it does NOT
    demonstrate clinical utility, and its output must never be used to
    populate an evidence_level or clinical_readiness claim (see
    evidence/evidence_contract.py, evidence/clinical_readiness.py). A
    positive net benefit here says nothing about whether using this model
    changes real clinical decisions or outcomes.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    n = len(y_true_arr)
    if n == 0:
        raise CalibrationPolicyError("decision_curve_analysis: y_true/y_prob must be non-empty.")

    prevalence = float(y_true_arr.mean())
    curve: List[Dict[str, Optional[float]]] = []
    for pt in threshold_range:
        pt = float(pt)
        if not (0.0 < pt < 1.0):
            continue
        y_pred = (y_prob_arr >= pt).astype(float)
        tp = float(np.sum((y_pred == 1) & (y_true_arr == 1)))
        fp = float(np.sum((y_pred == 1) & (y_true_arr == 0)))
        odds = pt / (1.0 - pt)
        net_benefit_model = (tp / n) - (fp / n) * odds
        net_benefit_treat_all = prevalence - (1.0 - prevalence) * odds
        net_benefit_treat_none = 0.0
        curve.append({
            "threshold": pt,
            "net_benefit_model": float(net_benefit_model),
            "net_benefit_treat_all": float(net_benefit_treat_all),
            "net_benefit_treat_none": net_benefit_treat_none,
        })

    return {
        "status": "exploratory_analysis_only",
        "disclaimer": (
            "Decision-curve analysis is exploratory only and does not demonstrate clinical "
            "utility. It must never be used to populate evidence_level or clinical_readiness "
            "claims — see evidence/clinical_readiness.py."
        ),
        "n": n,
        "prevalence": prevalence,
        "curve": curve,
    }
