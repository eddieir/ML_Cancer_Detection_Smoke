"""
benchmarks/model_fingerprint.py — deterministic fingerprints of ACTUAL fitted
model state (weights/parameters), not metadata about how long fitting took.

`fit_seconds`, timestamps, and parameter *counts* are not fingerprints of
fitted weights: two runs with identical data/config/seed produce different
`fit_seconds` every time, and a parameter count is the same for two models
with completely different learned values. Both would make the frozen-test
guard's identity non-deterministic (fit_seconds) or too coarse to detect a
genuinely different fitted model (parameter count) if used directly.

Two entry points:
  * `torch_state_dict_fingerprint(model)` — canonicalizes an nn.Module's
    state_dict (sorted keys, CPU, contiguous, exact tensor bytes) and hashes
    it. Device placement never changes the fingerprint.
  * `sklearn_model_state_fingerprint(model)` — canonicalizes every fitted
    attribute of a fitted estimator or Pipeline, recursively, and hashes it.

Canonicalization is an explicit whitelist of every estimator type this
framework's registered baselines (see baselines.SMOKE_BASELINES /
CANCER_BASELINES) can produce, plus the tree/ensemble internals sklearn
represents through non-Python-object (Cython extension) or non-public
(leading-underscore, not trailing-underscore) attributes:

  * `sklearn.tree._tree.Tree` is a Cython extension type with no `__dict__`
    at all — falling back to `repr()` for it produces a string containing
    the object's memory address (`<sklearn.tree._tree.Tree object at
    0x...>`), which is different every process and would make identically
    fitted random forests hash differently.
  * `sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor` has a
    `__dict__`, but its fitted fields (`nodes`, `raw_left_cat_bitsets`,
    `binned_left_cat_bitsets`) do not follow sklearn's trailing-underscore
    convention, so a generic "collect trailing-underscore attributes" walk
    silently collects nothing for it.
  * `HistGradientBoostingClassifier` itself stores its actual learned trees
    in the private `_predictors` attribute (and `_baseline_prediction`,
    `_bin_mapper`), none of which end in a trailing underscore — its public
    trailing-underscore attributes (`classes_`, `train_score_`, etc.) do not
    include the learned trees at all.

Any fitted value this module does not know how to canonicalize raises
`UnsupportedModelStateError` rather than silently falling back to `repr()`.
"""

import hashlib
import json
from typing import Any, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree as SklearnTree

# Every fitted estimator/container type this framework's registered
# baselines (SMOKE_BASELINES / CANCER_BASELINES in baselines.py) can
# actually produce, including RandomForestClassifier's per-tree
# DecisionTreeClassifier members. Deliberately NOT open-ended: an
# estimator type added to a baseline in the future without a matching
# entry here must fail loudly (UnsupportedModelStateError), not silently
# canonicalize via generic reflection that could miss non-conventional
# fitted state (as happened historically for TreePredictor and
# DummyClassifier.constant — see module docstring).
_EXPLICIT_DICT_REFLECTION_TYPES: Tuple[type, ...] = (
    LogisticRegression,
    StandardScaler,
    RandomForestClassifier,
    MLPClassifier,
    DecisionTreeClassifier,
)


class UnsupportedModelStateError(TypeError):
    """A fitted value had no defined deterministic canonicalization. Never
    caught internally — an unsupported estimator type must fail loudly
    rather than silently fall back to a non-deterministic repr()."""


def _canonicalize_fitted_attrs(obj: Any, extra_private: Tuple[str, ...] = ()) -> dict:
    """Every attribute sklearn's convention marks as fit-derived (trailing
    underscore, not a dunder), plus any explicitly named private attributes
    that hold real learned state under a non-conventional name."""
    fitted_attrs = {k: v for k, v in vars(obj).items()
                     if k.endswith("_") and not k.startswith("__")}
    for name in extra_private:
        if hasattr(obj, name):
            fitted_attrs[name] = getattr(obj, name)
    return {"__class__": type(obj).__qualname__, "fitted": _canonicalize(fitted_attrs)}


def _canonicalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {"__ndarray__": True, "dtype": str(arr.dtype), "shape": list(arr.shape),
                "data": arr.tobytes().hex()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, Pipeline):
        return {"__class__": "sklearn.pipeline.Pipeline",
                "steps": [[name, _canonicalize(step)] for name, step in obj.steps]}
    if isinstance(obj, SklearnTree):
        # A Cython extension type with no __dict__: every field that
        # distinguishes one fitted tree from another must be listed
        # explicitly rather than discovered via reflection.
        return {
            "__class__": "sklearn.tree._tree.Tree",
            "node_count": int(obj.node_count),
            "capacity": int(obj.capacity),
            "max_depth": int(obj.max_depth),
            "n_leaves": int(obj.n_leaves),
            "children_left": _canonicalize(obj.children_left),
            "children_right": _canonicalize(obj.children_right),
            "feature": _canonicalize(obj.feature),
            "threshold": _canonicalize(obj.threshold),
            "impurity": _canonicalize(obj.impurity),
            "n_node_samples": _canonicalize(obj.n_node_samples),
            "weighted_n_node_samples": _canonicalize(obj.weighted_n_node_samples),
            "value": _canonicalize(obj.value),
        }
    if isinstance(obj, TreePredictor):
        # Has a __dict__, but none of its fitted fields end in a trailing
        # underscore, so the generic reflection path below would silently
        # collect nothing for it.
        return {
            "__class__": "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            "nodes": _canonicalize(obj.nodes),
            "raw_left_cat_bitsets": _canonicalize(obj.raw_left_cat_bitsets),
            "binned_left_cat_bitsets": _canonicalize(obj.binned_left_cat_bitsets),
        }
    if isinstance(obj, _BinMapper):
        # HistGradientBoostingClassifier._bin_mapper's actual learned bin
        # thresholds (bin_thresholds_, is_categorical_,
        # missing_values_bin_idx_, n_bins_non_missing_) DO follow the
        # trailing-underscore convention, verified by direct inspection —
        # still listed explicitly rather than left to the generic __dict__
        # fallback, which no longer exists.
        return _canonicalize_fitted_attrs(obj)
    if isinstance(obj, DummyClassifier):
        # strategy="constant"'s predicted value lives in the constructor
        # parameter `constant`, not in any trailing-underscore fitted
        # attribute — two DummyClassifiers predicting different constants
        # would otherwise be indistinguishable by their fitted state alone
        # (this is the actual model used for the single-training-class
        # fallback path in baselines._sklearn_fit / _scaled_sklearn_fit).
        return _canonicalize_fitted_attrs(obj, extra_private=("strategy", "constant"))
    if isinstance(obj, HistGradientBoostingClassifier):
        # The public trailing-underscore attributes (classes_,
        # n_trees_per_iteration_, do_early_stopping_, train_score_,
        # validation_score_, is_categorical_, n_features_in_) do not include
        # the actual learned trees, which live in these private attributes.
        return _canonicalize_fitted_attrs(
            obj,
            extra_private=("_predictors", "_baseline_prediction", "_bin_mapper", "_n_features"),
        )
    if isinstance(obj, _EXPLICIT_DICT_REFLECTION_TYPES):
        # LogisticRegression, StandardScaler, RandomForestClassifier (whose
        # estimators_ are DecisionTreeClassifier instances, each recursing
        # into the explicit sklearn.tree._tree.Tree handling above via its
        # tree_ attribute), MLPClassifier, and DecisionTreeClassifier itself
        # all expose their complete learned state through ordinary
        # trailing-underscore attributes, verified by direct inspection
        # under the pinned scikit-learn version — no non-conventional
        # private state to enumerate, unlike TreePredictor/HGB/DummyClassifier
        # above.
        return _canonicalize_fitted_attrs(obj)
    if hasattr(obj, "__dict__"):
        # Deliberately NOT a supported fallback: an object with a __dict__
        # but no explicit whitelist entry above may hold fitted state under
        # a non-trailing-underscore name (as DummyClassifier.constant and
        # TreePredictor's fields did), or an empty/irrelevant __dict__ that
        # would silently produce an incomplete fingerprint. Fail loudly
        # instead of guessing.
        raise UnsupportedModelStateError(
            f"model_fingerprint: {type(obj).__module__}.{type(obj).__qualname__} has a "
            f"__dict__ but is not in the explicit canonicalization whitelist — add explicit "
            f"handling (verifying which attributes actually hold fitted state under the "
            f"installed scikit-learn version) rather than relying on generic reflection, "
            f"which can silently miss non-trailing-underscore fitted state."
        )
    raise UnsupportedModelStateError(
        f"model_fingerprint: no deterministic canonicalization is defined for "
        f"{type(obj).__module__}.{type(obj).__qualname__}; refusing to fall back "
        f"to repr() because it may embed a non-deterministic memory address or "
        f"other run-specific identity."
    )


def sklearn_model_state_fingerprint(model: Any) -> str:
    """SHA-256 of a canonical representation of every fitted attribute of
    `model` — a fitted sklearn estimator, Pipeline, or DummyClassifier. No
    timestamps, memory addresses, or fit_seconds ever enter this
    representation; only learned numeric state and structure. Raises
    UnsupportedModelStateError if `model` (or any nested fitted value)
    contains a type this module has no explicit deterministic handling for."""
    canonical = _canonicalize(model)
    blob = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def torch_state_dict_fingerprint(model) -> str:
    """SHA-256 of an nn.Module's state_dict: sorted keys, moved to CPU, made
    contiguous, keyed by (name, dtype, shape) with exact tensor bytes.
    Device placement (cpu vs. an equivalent state loaded back on cpu) does
    not change the result; different learned weights do."""
    state_dict = model.state_dict()
    parts = []
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().to("cpu").contiguous()
        header = f"{key}|{str(tensor.dtype)}|{tuple(tensor.shape)}".encode("utf-8")
        parts.append(header + b"|" + tensor.numpy().tobytes())
    blob = b"\x00".join(parts)
    return hashlib.sha256(blob).hexdigest()
