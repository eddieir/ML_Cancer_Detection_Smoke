"""
benchmarks/bundle.py — deployable model bundle manifest (Phase 4, Step 10).

A "bundle" is a directory referencing (never copying/duplicating) a model
checkpoint file and a PreprocessingArtifact, plus one coherent
bundle_manifest.json recording everything needed to decide whether the
bundle is safe to load and run inference with, WITHOUT re-deriving any of
it from the original training data:

  - model checkpoint reference (relative path + SHA-256)
  - preprocessing artifact reference (relative path + SHA-256 + its own
    scientific_fingerprint() + selected gene count)
  - model configuration, class vocabulary, label policy, species policy,
    assay mode
  - dataset-manifest fingerprint, split fingerprint
  - calibration state and decision threshold, when applicable
  - an environment snapshot (see benchmarks/reporting.py::
    write_environment_artifact) referenced by hash
  - a single bundle_fingerprint covering every field above (order-
    independent — computed over a canonical, sorted-key JSON encoding)

write_model_bundle() never mutates or moves the checkpoint file it
references — it only hashes it. load_and_validate_bundle() re-hashes and
re-derives every one of the fields above and raises a specific,
non-generic exception the instant anything doesn't match; it never repairs,
warns-and-continues, or silently treats a partial/tampered bundle as usable.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from data.manifest import sha256_of_file
from data.preprocessing import PreprocessingArtifact, PreprocessingArtifactError

from .atomic_io import atomic_write_json

BUNDLE_SCHEMA_VERSION = "1"


class BundleError(RuntimeError):
    """Base class for every error this module raises."""


class BundleCorruptionError(BundleError):
    """Raised when bundle_manifest.json (or a file it references) is
    missing, unparsable, or from an unrecognized schema version."""


class BundleValidationError(BundleError):
    """Raised when a loaded bundle's manifest does not match its
    referenced components (checkpoint hash, artifact hash/fingerprint,
    gene count, class vocabulary against a given model, ...)."""


class LegacyBundleError(BundleError):
    """Raised when a directory has a checkpoint but no bundle_manifest.json
    at all, and the caller did not explicitly pass allow_legacy=True.
    Mirrors inference.py's unsafe_legacy_mode policy: a legacy, manifest-
    less checkpoint is never silently treated as Phase-4-bundle-compatible
    — the caller must opt in explicitly, and that opt-in is never
    represented as equivalent to a real, reproducible bundle."""


def _bundle_fingerprint(manifest: Dict) -> str:
    payload = {k: v for k, v in manifest.items() if k != "bundle_fingerprint"}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_model_bundle(
    bundle_dir: Union[str, Path],
    checkpoint_path: Union[str, Path],
    artifact: PreprocessingArtifact,
    model_config: Dict,
    class_vocabulary: List[str],
    label_policy: str,
    species_policy: str,
    assay_mode: str,
    dataset_manifest_fingerprint: Optional[str] = None,
    split_fingerprint: Optional[str] = None,
    calibration_state: Optional[Dict] = None,
    decision_threshold: Optional[float] = None,
    environment_snapshot: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> Path:
    """
    Write bundle_dir/bundle_manifest.json (and bundle_dir/
    preprocessing_artifact.json, a copy of `artifact`'s own atomic save —
    the ONE file this function does write fresh, since the artifact is
    small, JSON, and needs to live next to the manifest that hashes it).
    `checkpoint_path` must already exist; it is referenced by a bundle-
    relative path + SHA-256, never copied (checkpoints can be large binary
    files and are never committed to source control — see repository
    hygiene rules).
    """
    import os

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"write_model_bundle: checkpoint not found at {checkpoint_path}")

    artifact_path = bundle_dir / "preprocessing_artifact.json"
    artifact.save(artifact_path)

    # Relative to bundle_dir (never an absolute local path — repository
    # hygiene rule, and keeps a bundle portable across machines/checkouts).
    # The checkpoint is NOT required to live inside bundle_dir — Trainer's
    # checkpoints and their bundle directories are siblings under the same
    # checkpoint_dir — so this is computed with os.path.relpath, not
    # assumed to be a bare filename.
    checkpoint_rel = os.path.relpath(checkpoint_path.resolve(), bundle_dir.resolve())

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model_checkpoint": {
            "path": checkpoint_rel,
            "sha256": sha256_of_file(checkpoint_path),
        },
        "preprocessing_artifact": {
            "path": artifact_path.name,
            "sha256": sha256_of_file(artifact_path),
            "fingerprint": artifact.scientific_fingerprint(),
            "gene_count": len(artifact.gene_list),
        },
        "model_config": model_config,
        "class_vocabulary": list(class_vocabulary),
        "label_policy": label_policy,
        "species_policy": species_policy,
        "assay_mode": assay_mode,
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint,
        "split_fingerprint": split_fingerprint,
        "calibration_state": calibration_state,
        "decision_threshold": decision_threshold,
        "environment_snapshot": environment_snapshot,
    }
    if extra:
        manifest.update(extra)
    manifest["bundle_fingerprint"] = _bundle_fingerprint(manifest)
    atomic_write_json(bundle_dir / "bundle_manifest.json", manifest)
    return bundle_dir


def load_and_validate_bundle(bundle_dir: Union[str, Path], allow_legacy: bool = False) -> Dict:
    """
    Load bundle_dir/bundle_manifest.json, re-derive and cross-check every
    component it references, and return the manifest dict (with an extra
    "_artifact" key holding the already-loaded, already-verified
    PreprocessingArtifact, and "_bundle_dir" holding `bundle_dir` — both
    for caller convenience, neither persisted). Raises:

      LegacyBundleError — no bundle_manifest.json at all, and
        allow_legacy=False (the default). Passing allow_legacy=True skips
        every check in this function and returns {} — the caller is
        entirely on its own for a legacy, non-reproducible checkpoint, and
        must not describe the result as Phase-4-bundle-verified.
      BundleCorruptionError — the manifest (or a referenced file) is
        missing, unparsable, or from an unrecognized schema version.
      BundleValidationError — the manifest's own recorded bundle_fingerprint
        doesn't match its content, or a referenced file's hash/fingerprint/
        gene count doesn't match what the manifest recorded.
    """
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        if allow_legacy:
            return {}
        raise LegacyBundleError(
            f"load_and_validate_bundle: no bundle_manifest.json at {manifest_path} — this "
            "looks like a legacy, pre-Phase-4 checkpoint directory with no bundle manifest at "
            "all. Pass allow_legacy=True to load it anyway (never reproducible/verified in the "
            "way a real bundle is), or regenerate a proper bundle for this checkpoint."
        )
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise BundleCorruptionError(
            f"load_and_validate_bundle: {manifest_path} exists but could not be parsed as "
            f"valid JSON ({exc!r})."
        ) from exc

    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleCorruptionError(
            f"load_and_validate_bundle: {manifest_path} has schema_version="
            f"{manifest.get('schema_version')!r}, this code understands "
            f"{BUNDLE_SCHEMA_VERSION!r} only."
        )

    recomputed_fp = _bundle_fingerprint(manifest)
    if recomputed_fp != manifest.get("bundle_fingerprint"):
        raise BundleValidationError(
            f"load_and_validate_bundle: {manifest_path} has been modified since it was "
            "written — recomputed bundle_fingerprint does not match the recorded one. "
            "Refusing to trust a tampered or corrupted manifest."
        )

    ckpt_info = manifest.get("model_checkpoint") or {}
    ckpt_path = bundle_dir / str(ckpt_info.get("path", ""))
    if not ckpt_info.get("path") or not ckpt_path.exists():
        raise BundleCorruptionError(
            f"load_and_validate_bundle: checkpoint referenced by {manifest_path} "
            f"({ckpt_info.get('path')!r}) is missing from {bundle_dir}."
        )
    if sha256_of_file(ckpt_path) != ckpt_info.get("sha256"):
        raise BundleValidationError(
            f"load_and_validate_bundle: checkpoint {ckpt_path} does not match the SHA-256 "
            f"{manifest_path} recorded for it — corrupt, truncated, or swapped checkpoint file."
        )

    art_info = manifest.get("preprocessing_artifact") or {}
    art_path = bundle_dir / str(art_info.get("path", ""))
    if not art_info.get("path") or not art_path.exists():
        raise BundleCorruptionError(
            f"load_and_validate_bundle: preprocessing artifact referenced by {manifest_path} "
            f"({art_info.get('path')!r}) is missing from {bundle_dir}."
        )
    if sha256_of_file(art_path) != art_info.get("sha256"):
        raise BundleValidationError(
            f"load_and_validate_bundle: preprocessing artifact file {art_path} does not match "
            f"the SHA-256 {manifest_path} recorded for it."
        )
    try:
        artifact = PreprocessingArtifact.load(art_path)
    except PreprocessingArtifactError as exc:
        raise BundleValidationError(
            f"load_and_validate_bundle: preprocessing artifact at {art_path} failed its own "
            f"integrity checks ({exc!r}) — refusing to treat this bundle as loadable."
        ) from exc
    if artifact.scientific_fingerprint() != art_info.get("fingerprint"):
        raise BundleValidationError(
            f"load_and_validate_bundle: preprocessing artifact at {art_path} has "
            f"scientific_fingerprint {artifact.scientific_fingerprint()[:16]}..., but "
            f"{manifest_path} recorded {str(art_info.get('fingerprint'))[:16]}... — this is "
            "not the artifact this bundle was built with (e.g. another fold's artifact was "
            "substituted in)."
        )
    if len(artifact.gene_list) != art_info.get("gene_count"):
        raise BundleValidationError(
            f"load_and_validate_bundle: preprocessing artifact at {art_path} selects "
            f"{len(artifact.gene_list)} genes, but {manifest_path} recorded "
            f"gene_count={art_info.get('gene_count')} — the gene panel was altered after this "
            "bundle was built."
        )

    manifest["_artifact"] = artifact
    manifest["_bundle_dir"] = bundle_dir
    return manifest


def validate_bundle_for_model(manifest: Dict, model) -> None:
    """
    Cross-check an already-loaded (load_and_validate_bundle'd) manifest
    against an actual constructed model instance: input width and class
    vocabulary must agree, or the checkpoint in this bundle is not usable
    with this model. Raises BundleValidationError on any mismatch.
    """
    artifact = manifest.get("_artifact")
    if artifact is None:
        raise BundleValidationError(
            "validate_bundle_for_model: manifest has no '_artifact' — call "
            "load_and_validate_bundle() first, not a hand-built manifest dict."
        )
    input_dim = getattr(model, "input_dim", None)
    if input_dim is not None and input_dim != len(artifact.gene_list):
        raise BundleValidationError(
            f"validate_bundle_for_model: model.input_dim={input_dim} does not match this "
            f"bundle's preprocessing artifact gene count ({len(artifact.gene_list)})."
        )
    num_smoke = getattr(model, "num_smoke", None)
    vocab = manifest.get("class_vocabulary") or []
    if num_smoke is not None and vocab and num_smoke != len(vocab):
        raise BundleValidationError(
            f"validate_bundle_for_model: model.num_smoke={num_smoke} does not match this "
            f"bundle's recorded class_vocabulary ({len(vocab)} classes: {vocab})."
        )


def validate_bundle_matches_identity(manifest: Dict, expected: Dict) -> None:
    """
    Resume-safety check (item 39): before treating a persisted bundle as
    reusable for a specific dataset-manifest/split identity, verify its
    recorded dataset_manifest_fingerprint/split_fingerprint match what the
    caller expects. `expected` is a dict with the same two keys (either or
    both may be omitted/None to skip that check). Raises
    BundleValidationError on any mismatch — never silently proceeds with a
    bundle produced under a different dataset or split.
    """
    for key in ("dataset_manifest_fingerprint", "split_fingerprint"):
        if key not in expected or expected[key] is None:
            continue
        if manifest.get(key) != expected[key]:
            raise BundleValidationError(
                f"validate_bundle_matches_identity: bundle's {key}={manifest.get(key)!r} does "
                f"not match the expected value {expected[key]!r} for this resume attempt — "
                "refusing to reuse a bundle built under a different dataset/split identity."
            )
