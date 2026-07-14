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
    attribute (scikit-learn's trailing-underscore convention) of a fitted
    estimator or Pipeline, recursively, and hashes it.
"""

import hashlib
import json
from typing import Any

import numpy as np


def _canonicalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {"__ndarray__": True, "dtype": str(arr.dtype), "shape": list(arr.shape),
                "data": arr.tobytes().hex()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    # sklearn-style fitted estimator (BaseEstimator or Pipeline): recurse
    # into every attribute sklearn's convention marks as fit-derived (a
    # trailing underscore, not a dunder) — covers Pipeline.steps' nested
    # estimators automatically since each step's estimator is itself an
    # object with fitted trailing-underscore attributes.
    if hasattr(obj, "__dict__"):
        fitted_attrs = {k: v for k, v in vars(obj).items()
                         if k.endswith("_") and not k.startswith("__")}
        if hasattr(obj, "steps"):  # sklearn.pipeline.Pipeline
            fitted_attrs["__pipeline_steps__"] = [(name, est) for name, est in obj.steps]
        return {"__class__": type(obj).__qualname__, "fitted": _canonicalize(fitted_attrs)}
    return repr(obj)


def sklearn_model_state_fingerprint(model: Any) -> str:
    """SHA-256 of a canonical representation of every fitted (trailing-
    underscore) attribute of `model` — a fitted sklearn estimator, Pipeline,
    or DummyClassifier. No timestamps, memory addresses, or fit_seconds ever
    enter this representation; only learned numeric state and structure."""
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
