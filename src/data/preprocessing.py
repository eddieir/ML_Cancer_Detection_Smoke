"""
data/preprocessing.py — leakage-free fit/transform preprocessing artifact.

merge_sources(scale=True) (the legacy default) z-scores the FULL merged
dataset and smoke_aware_hvg() picks highly-variable genes across the FULL
merged dataset — both BEFORE any train/val/test split exists. That means
validation/test cells influence the scaling statistics and which genes
become model features: classic preprocessing leakage, forbidden by this
project's scientific rules (see README.md).

This module separates FIT (train cells only) from TRANSFORM (applied
identically, unchanged, to val/test/inference), and packages the fitted
state into a versioned, JSON-serialisable artifact that can be saved
alongside a checkpoint and reloaded for inference-time compatibility
checks (required genes, gene order, input dimension, artifact version).
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import anndata as ad
import numpy as np
import scanpy as sc

ARTIFACT_VERSION = "1"

# Gene-contract policy defaults — must agree with configs/default.yaml's
# preprocessing.* keys (see tests/test_config_consistency.py, which fails
# the build if these ever drift apart). "error" is the conservative default
# everywhere: silently tolerating a missing/duplicate gene or unlabelled
# coverage shortfall is exactly the kind of silent-recovery behavior Phase 4
# forbids (see this module's own docstring and README.md).
DEFAULT_MISSING_GENE_POLICY = "error"       # "error" | "zero_fill" (explicit opt-in only)
DEFAULT_DUPLICATE_GENE_POLICY = "error"     # "error" only — no aggregation policy implemented
DEFAULT_UNEXPECTED_GENE_POLICY = "ignore"   # "ignore" (recorded in diagnostics) | "error"
DEFAULT_MINIMUM_GENE_COVERAGE = 1.0         # fraction of artifact.gene_list that must be present


class PreprocessingArtifactError(ValueError):
    """Base class for every error this module raises about an artifact's
    identity, integrity, or compatibility with a given input. Subclasses
    ValueError (rather than a bare Exception) so existing callers/tests
    that catch the historical `pytest.raises(ValueError)` for a bad gene
    panel keep working unchanged; callers that want to catch "something is
    wrong with this artifact" specifically should catch this class."""


class GeneContractError(PreprocessingArtifactError):
    """Raised when an input's genes violate the artifact's gene contract:
    a missing required gene under the default error policy, a duplicate
    gene identifier, an unexpected gene under an error policy, or
    insufficient gene coverage against the artifact's configured minimum."""


class ArtifactCompatibilityError(PreprocessingArtifactError):
    """Raised when a model checkpoint and a PreprocessingArtifact do not
    agree on scientific identity (fingerprint mismatch, gene-list mismatch,
    or gene-count mismatch) — see checkpoint_matches_artifact() and
    train.py/inference.py's checkpoint<->artifact binding."""


class LegacyArtifactError(PreprocessingArtifactError):
    """Raised when a checkpoint has no associated PreprocessingArtifact (or
    no recorded artifact fingerprint) and the caller did not explicitly opt
    into legacy/non-reproducible behavior (unsafe_legacy_mode=True)."""


# Scientific-content fields that make up PreprocessingArtifact's fingerprint.
# Deliberately excludes `created_at` (a volatile timestamp — see
# scientific_fingerprint()'s docstring) and `notes` (free-text commentary,
# not scientific state). Every other field is part of the artifact's
# reproducible identity: two artifacts fit from the same training data,
# config, and code produce the same fingerprint regardless of output
# directory, hostname, or wall-clock time.
_FINGERPRINT_FIELDS = (
    "version", "gene_list", "gene_means", "gene_stds", "n_hvgs",
    "smoke_marker_genes_forced", "fit_n_cells", "fit_n_subjects",
    "label_mapping", "expected_input_stage", "cell_type_map_fingerprint",
    "cell_type_annotation_mode", "cell_type_annotation_degraded",
    "cell_type_annotation_compatibility", "missing_gene_policy",
    "duplicate_gene_policy", "unexpected_gene_policy", "minimum_gene_coverage",
    "batch_correction_status",
)

_VALID_BATCH_CORRECTION_STATUSES = frozenset({"disabled", "transductive_diagnostic_only"})

# The only input stages apply_preprocessing()/predict_h5ad() know how to
# handle — see inference.py. "normalized_expression" is what this artifact's
# apply_preprocessing() actually expects: input already QC'd, library-size
# normalized, and log-transformed (data/transforms.py::normalize) upstream
# of this artifact; the artifact itself only stores gene subsetting/ordering
# and train-fit mean/std scaling, NOT QC thresholds, a library-size target,
# or log-transform parameters — so it cannot reproduce a full raw-count
# pipeline, only pick up from already-normalized expression.
EXPECTED_INPUT_STAGE = "normalized_expression"


@dataclass
class PreprocessingArtifact:
    version:                   str
    gene_list:                 List[str]    # ordered HVG gene names, fit on train cells only
    gene_means:                List[float]  # per-gene mean, fit on train cells only
    gene_stds:                 List[float]  # per-gene std,  fit on train cells only (0 -> 1.0)
    n_hvgs:                    int
    smoke_marker_genes_forced: List[str]
    fit_n_cells:                int
    fit_n_subjects:              int
    notes:                     List[str] = field(default_factory=list)
    # Smoke-label effective mapping (data/label_mapping.py), set after fitting
    # once the rare-class policy has run — None only for artifacts fit before
    # this field existed (legacy) or that never had label-mapping context.
    label_mapping:              Optional[Dict] = None
    # What input stage apply_preprocessing() expects to receive — see
    # EXPECTED_INPUT_STAGE above. Stored explicitly (not just documented) so
    # inference can validate a caller's claimed input_stage against what
    # this artifact can actually consume.
    expected_input_stage:       str = EXPECTED_INPUT_STAGE
    # SHA-256 of the fixed cell-type label-name -> ID table (constants.py's
    # CELL_TYPE_MAP) in effect when obs["cell_type_id"] was assigned — see
    # data/transforms.py::cell_type_map_fingerprint(). None for artifacts fit
    # before this field existed, or when cell-type annotation never ran
    # (e.g. pseudo-bulk-only sources). A mismatch on reload means the code's
    # mapping table changed since this artifact was fit, not that different
    # cells were annotated.
    cell_type_map_fingerprint:  Optional[str] = None
    # "inductive_per_cell" (CellTypist majority_voting=False — a pure
    # function of each cell's own expression, independent of which other
    # cells were annotated alongside it) or "majority_voting" (legacy,
    # dataset-dependent) — see annotate_cell_types(). None if annotation
    # never ran.
    cell_type_annotation_mode:  Optional[str] = None
    # True only when annotate_cell_types() fell back to a fixed placeholder
    # label for every cell after CellTypist itself failed (see
    # allow_diagnostic_fallback in data/transforms.py) — never a real,
    # scientifically meaningful per-cell annotation. None/False for a normal
    # artifact. ExperimentContext.from_pipeline_result rejects an artifact
    # with this set to True (see benchmarks/context.py) — a real run must
    # never silently proceed on degraded cell-type labels.
    cell_type_annotation_degraded: Optional[bool] = None
    # CellTypist/scikit-learn version-compatibility provenance for this
    # artifact's cell-type annotation — see data/transforms.py::
    # _load_celltypist_model_checking_sklearn_compatibility. Keys:
    # celltypist_version, model_name, runtime_sklearn_version,
    # serialized_sklearn_versions (list), compatible (bool). None when
    # annotation never ran (pseudo-bulk) or failed to a diagnostic fallback
    # before a model could be loaded, or for artifacts fit before this
    # field existed.
    cell_type_annotation_compatibility: Optional[Dict] = None
    # Gene-contract policies actually enforced by verify_compatible()/
    # apply_preprocessing() below — see the DEFAULT_* constants above.
    # Persisted explicitly (not just implied by code defaults) so a reloaded
    # artifact enforces the SAME policy it was fit under, even if the
    # module-level defaults change in a later code version.
    missing_gene_policy:        str = DEFAULT_MISSING_GENE_POLICY
    duplicate_gene_policy:      str = DEFAULT_DUPLICATE_GENE_POLICY
    unexpected_gene_policy:     str = DEFAULT_UNEXPECTED_GENE_POLICY
    minimum_gene_coverage:      float = DEFAULT_MINIMUM_GENE_COVERAGE
    # Whether the RUN that produced this artifact also opted into batch
    # correction — see data/transforms.py's UnsafeBatchCorrectionError
    # module docstring for why "train_fitted_inductive" is not a value that
    # can occur here (no inductive implementation exists). Note this
    # artifact's OWN fitted state (gene selection, scaling means/stds) never
    # includes Harmony's output either way — preprocess.py runs Harmony
    # AFTER apply_preprocessing, on the already gene-selected/scaled
    # output, never before or during this artifact's own fit. "disabled"
    # (default) means the run never opted into batch correction at all;
    # "transductive_diagnostic_only" means Harmony ran, across the full
    # train+val+test dataset, as a disclosed, non-leakage-free diagnostic
    # step downstream of this artifact — assert_batch_correction_safe()
    # (data/transforms.py) refuses to let a run in this state reach
    # CV/OOF/final-dev-pool/frozen-test evaluation.
    batch_correction_status:    str = "disabled"
    # Wall-clock creation time (time.time()) — informational only, excluded
    # from scientific_fingerprint() (see _FINGERPRINT_FIELDS above). None
    # for artifacts fit before this field existed.
    created_at:                  Optional[float] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = time.time()
        if len(self.gene_list) == 0:
            raise GeneContractError(
                "PreprocessingArtifact: gene_list is empty — an artifact with no selected "
                "genes cannot transform any input and is refused at construction time."
            )
        if len(set(self.gene_list)) != len(self.gene_list):
            dupes = sorted({g for g in self.gene_list if self.gene_list.count(g) > 1})
            raise GeneContractError(
                f"PreprocessingArtifact: gene_list contains duplicate gene identifiers "
                f"{dupes[:10]} — the artifact's own selected-gene panel must be unique."
            )
        if not (len(self.gene_list) == len(self.gene_means) == len(self.gene_stds)):
            raise GeneContractError(
                f"PreprocessingArtifact: gene_list ({len(self.gene_list)}), gene_means "
                f"({len(self.gene_means)}), and gene_stds ({len(self.gene_stds)}) must be "
                "the same length — one scaling statistic per selected gene."
            )
        if self.missing_gene_policy not in ("error", "zero_fill"):
            raise GeneContractError(
                f"PreprocessingArtifact: missing_gene_policy={self.missing_gene_policy!r} "
                "is not one of 'error', 'zero_fill'."
            )
        if self.duplicate_gene_policy != "error":
            raise GeneContractError(
                f"PreprocessingArtifact: duplicate_gene_policy={self.duplicate_gene_policy!r} "
                "is not 'error' — no deterministic duplicate-aggregation policy is implemented, "
                "so 'error' is the only supported value."
            )
        if self.unexpected_gene_policy not in ("ignore", "error"):
            raise GeneContractError(
                f"PreprocessingArtifact: unexpected_gene_policy="
                f"{self.unexpected_gene_policy!r} is not one of 'ignore', 'error'."
            )
        if not (0.0 < self.minimum_gene_coverage <= 1.0):
            raise GeneContractError(
                f"PreprocessingArtifact: minimum_gene_coverage={self.minimum_gene_coverage!r} "
                "must be in (0.0, 1.0]."
            )
        if self.batch_correction_status not in _VALID_BATCH_CORRECTION_STATUSES:
            raise GeneContractError(
                f"PreprocessingArtifact: batch_correction_status="
                f"{self.batch_correction_status!r} is not one of "
                f"{sorted(_VALID_BATCH_CORRECTION_STATUSES)}."
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def scientific_fingerprint(self) -> str:
        """SHA-256 over every scientifically meaningful field of this
        artifact (see _FINGERPRINT_FIELDS) — excludes `created_at` and
        `notes`. Two artifacts fit from the same training subjects, config,
        and code produce the same fingerprint regardless of output
        directory, hostname, or when they were fit; changing the selected
        genes, gene order, scaling statistics, label policy, species/assay
        provenance, or gene-contract policy changes it."""
        payload = {k: getattr(self, k) for k in _FINGERPRINT_FIELDS}
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def inspect(self) -> dict:
        """Read-only, machine-readable summary of this artifact's identity
        and contract — never exposes raw scaling arrays or participant-level
        data, only counts/policies/fingerprints. Safe to print, log, or
        attach to a bundle manifest (see model.py's bundle helpers)."""
        return {
            "schema_version": self.version,
            "artifact_fingerprint": self.scientific_fingerprint(),
            "selected_gene_count": len(self.gene_list),
            "n_hvgs_requested": self.n_hvgs,
            "smoke_marker_genes_forced": len(self.smoke_marker_genes_forced),
            "fit_n_cells": self.fit_n_cells,
            "fit_n_subjects": self.fit_n_subjects,
            "expected_input_stage": self.expected_input_stage,
            "missing_gene_policy": self.missing_gene_policy,
            "duplicate_gene_policy": self.duplicate_gene_policy,
            "unexpected_gene_policy": self.unexpected_gene_policy,
            "minimum_gene_coverage": self.minimum_gene_coverage,
            "batch_correction_status": self.batch_correction_status,
            "cell_type_annotation_mode": self.cell_type_annotation_mode,
            "cell_type_annotation_degraded": self.cell_type_annotation_degraded,
            "cell_type_map_fingerprint": self.cell_type_map_fingerprint,
            "label_mapping_present": self.label_mapping is not None,
            "created_at": self.created_at,
            "notes": list(self.notes),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Atomic write: builds the full JSON in memory, then writes it via
        a temp-file + os.replace() so a reader never observes a partially
        written artifact and a crash mid-write leaves the previous file (or
        nothing), never a truncated one."""
        from benchmarks.atomic_io import atomic_write_json

        path = Path(path)
        atomic_write_json(path, self.to_dict())
        print(f"[preprocessing] artifact saved → {path}  ({len(self.gene_list)} genes, "
              f"fingerprint={self.scientific_fingerprint()[:12]})")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PreprocessingArtifact":
        path = Path(path)
        try:
            with open(path) as f:
                text = f.read()
            d = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise PreprocessingArtifactError(
                f"PreprocessingArtifact.load: {path} exists but could not be parsed as valid "
                f"JSON ({exc!r}) — this looks like a truncated or corrupted write, not a valid "
                "artifact. Refusing to silently recover; regenerate the artifact instead."
            ) from exc
        if d.get("version") != ARTIFACT_VERSION:
            raise LegacyArtifactError(
                f"PreprocessingArtifact.load: {path} has version={d.get('version')!r}, this "
                f"code understands version={ARTIFACT_VERSION!r} only. Loading an artifact "
                "from an unknown/future schema version is refused rather than guessed at; "
                "loading a genuinely legacy (pre-Phase-4) artifact-less checkpoint is a "
                "separate, explicit opt-in — see unsafe_legacy_mode in inference.py."
            )
        # Unknown keys (e.g. a future field this code doesn't know about yet)
        # are rejected rather than silently dropped — cls(**d) already does
        # this via TypeError, but re-raised here as a clearer, artifact-
        # specific error.
        try:
            return cls(**d)
        except TypeError as exc:
            raise PreprocessingArtifactError(
                f"PreprocessingArtifact.load: {path} does not match the fields this code's "
                f"PreprocessingArtifact expects ({exc!r}) — likely a schema mismatch. "
                "Regenerate the artifact with the matching code version."
            ) from exc


def fit_preprocessing(
    adata:              ad.AnnData,
    train_subject_ids:  set,
    n_hvgs:             int = 2000,
    batch_key:          Optional[str] = "batch",
    subject_col:        str = "subject_id",
    missing_gene_policy:    str = DEFAULT_MISSING_GENE_POLICY,
    duplicate_gene_policy:  str = DEFAULT_DUPLICATE_GENE_POLICY,
    unexpected_gene_policy: str = DEFAULT_UNEXPECTED_GENE_POLICY,
    minimum_gene_coverage:  float = DEFAULT_MINIMUM_GENE_COVERAGE,
    batch_correction_status: str = "disabled",
) -> PreprocessingArtifact:
    """
    Fit gene mean/std scaling + smoke-aware HVG selection using ONLY cells
    whose subject_id is in train_subject_ids. Mirrors merge_sources()'s
    z-scoring and transforms.smoke_aware_hvg()'s variance-based selection
    with forced smoke-marker inclusion, but restricted to the train
    partition so val/test statistics never influence what genes are kept
    or how they're scaled.

    The four `*_policy`/`minimum_gene_coverage` arguments are recorded on
    the returned artifact and enforced later by verify_compatible()/
    apply_preprocessing() against every val/test/inference input — see the
    DEFAULT_* module constants and configs/default.yaml's preprocessing.*
    keys, which must agree with these defaults.
    """
    from constants import ALL_SMOKE_MARKERS

    train_mask = adata.obs[subject_col].astype(str).isin(
        {str(s) for s in train_subject_ids}
    ).values
    if train_mask.sum() == 0:
        raise ValueError("fit_preprocessing: no cells matched train_subject_ids")

    train_adata = adata[train_mask].copy()

    X = train_adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    stds[stds == 0] = 1.0  # constant genes: avoid divide-by-zero, contribute nothing either way

    bk = batch_key if (batch_key and batch_key in train_adata.obs.columns) else None
    sc.pp.highly_variable_genes(
        train_adata, n_top_genes=min(n_hvgs, train_adata.n_vars), batch_key=bk
    )
    hvg_mask = train_adata.var["highly_variable"].values.copy()

    present_markers = [g for g in ALL_SMOKE_MARKERS if g in train_adata.var_names]
    forced = [
        g for g in present_markers
        if not hvg_mask[train_adata.var_names.get_loc(g)]
    ]
    if forced:
        non_marker_idx = [
            i for i, hv in enumerate(hvg_mask)
            if hv and train_adata.var_names[i] not in present_markers
        ]
        for i in non_marker_idx[-len(forced):]:
            hvg_mask[i] = False
        for g in forced:
            hvg_mask[train_adata.var_names.get_loc(g)] = True

    gene_list = list(train_adata.var_names[hvg_mask])
    idx = [train_adata.var_names.get_loc(g) for g in gene_list]

    print(f"[preprocessing] fit  {len(gene_list)} genes  "
          f"(forced {len(forced)} smoke markers)  on {train_mask.sum():,} train cells")

    return PreprocessingArtifact(
        version=ARTIFACT_VERSION,
        gene_list=gene_list,
        gene_means=[float(means[i]) for i in idx],
        gene_stds=[float(stds[i]) for i in idx],
        n_hvgs=n_hvgs,
        smoke_marker_genes_forced=forced,
        fit_n_cells=int(train_mask.sum()),
        fit_n_subjects=len(set(train_adata.obs[subject_col].astype(str))),
        # Cell-type annotation provenance is a property of `adata.obs`
        # (annotate_cell_types() ran once, upstream, on the full merged
        # dataset before any fold/OOD/outer-split refit ever calls this
        # function) — never re-derived or defaulted here. Every caller of
        # fit_preprocessing (the outer split in preprocess.py, and every
        # per-fold/per-OOD-reconstruction refit in
        # benchmarks/fold_preprocessing.py) passes the SAME `adata` these
        # fields came from, so propagating them here means a fold-refit
        # artifact carries the identical, correct provenance the outer
        # artifact does — closing the gap where a fold refit silently
        # produced an artifact with unset (None) provenance fields.
        cell_type_map_fingerprint=adata.uns.get("cell_type_map_fingerprint"),
        cell_type_annotation_mode=adata.uns.get("cell_type_annotation_mode"),
        cell_type_annotation_degraded=adata.uns.get("cell_type_annotation_degraded"),
        cell_type_annotation_compatibility=adata.uns.get("cell_type_annotation_compatibility"),
        missing_gene_policy=missing_gene_policy,
        duplicate_gene_policy=duplicate_gene_policy,
        unexpected_gene_policy=unexpected_gene_policy,
        minimum_gene_coverage=minimum_gene_coverage,
        batch_correction_status=batch_correction_status,
        notes=[
            "Batch correction (Harmony) is NOT part of this artifact: Harmony has "
            "no native train-only-fit / apply-to-new-data transform, so it cannot "
            "be included in a leakage-free fit/transform artifact the way scaling "
            "and HVG selection can. batch_correct() must be re-fit independently "
            "per split if used — a documented limitation, not full leakage-free "
            "batch correction. See README.md's preprocessing-leakage section.",
        ],
    )


def apply_preprocessing(adata: ad.AnnData, artifact: PreprocessingArtifact) -> ad.AnnData:
    """
    Transform-only: subset/reorder genes to artifact.gene_list (in that
    exact order) and apply the train-fit mean/std scaling, clipped to
    [-10, 10] to match sc.pp.scale(max_value=10)'s legacy behaviour.

    Never mutates `artifact` and never infers/refits any parameter from
    `adata` — this function's only job is to apply the already-fitted
    scaling/gene-selection to new data. Raises GeneContractError per the
    policies recorded on `artifact` (see verify_compatible()); repeated
    calls with the same inputs produce identical output, and calling this
    on validation data before/after calling it on test data (or vice
    versa) never changes either result.
    """
    diagnostics = verify_compatible(artifact, adata.var_names)

    if diagnostics["missing"]:
        # Only reachable when missing_gene_policy == "zero_fill" (verify_
        # compatible already raised for policy=="error") — pad the missing
        # columns with exact zeros (pre-scaling), which the mean/std
        # scaling step below then transforms like any other gene. This is
        # never presented as observed expression: diagnostics (and the
        # printed warning) always record which genes were synthesized.
        present = adata[:, [g for g in artifact.gene_list if g not in diagnostics["missing"]]].copy()
        X_present = present.X
        if hasattr(X_present, "toarray"):
            X_present = X_present.toarray()
        X_present = np.asarray(X_present, dtype=np.float64)
        present_genes = list(present.var_names)
        full = np.zeros((adata.n_obs, len(artifact.gene_list)), dtype=np.float64)
        present_idx = {g: i for i, g in enumerate(present_genes)}
        for j, g in enumerate(artifact.gene_list):
            if g in present_idx:
                full[:, j] = X_present[:, present_idx[g]]
        X = full
        print(f"[preprocessing] WARNING: zero-filled {len(diagnostics['missing'])} missing "
              f"gene(s) under missing_gene_policy='zero_fill': {diagnostics['missing'][:10]}"
              + (" ..." if len(diagnostics["missing"]) > 10 else ""))
        out = ad.AnnData(X=X.astype(np.float32), obs=adata.obs.copy(),
                          var=adata.var.reindex(artifact.gene_list))
        X = full
    else:
        out = adata[:, artifact.gene_list].copy()
        X = out.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

    means = np.array(artifact.gene_means)
    stds  = np.array(artifact.gene_stds)
    X = np.clip((X - means) / stds, -10, 10)
    out.X = X.astype(np.float32)
    out.uns["preprocessing_compatibility_diagnostics"] = diagnostics
    return out


def verify_compatible(artifact: PreprocessingArtifact, gene_names) -> Dict:
    """
    Enforce this artifact's gene contract against `gene_names` and return a
    diagnostics dict ({"missing": [...], "unexpected": [...], "duplicate":
    [...], "coverage": float, "n_expected": int, "n_present": int}).
    Raises GeneContractError (a ValueError subclass, for backward
    compatibility with existing `pytest.raises(ValueError)` call sites) when:

      * any gene name in `gene_names` is duplicated (always fatal — no
        deterministic aggregation policy is implemented);
      * a required gene is missing and artifact.missing_gene_policy=="error"
        (the default — "zero_fill" is an explicit, non-default opt-in
        applied by apply_preprocessing(), never silently);
      * an unexpected gene is present and
        artifact.unexpected_gene_policy=="error" (default policy is
        "ignore" — unexpected genes are recorded in diagnostics but never
        fatal on their own);
      * gene coverage (fraction of artifact.gene_list actually present)
        falls below artifact.minimum_gene_coverage.

    Called by apply_preprocessing() and by inference code paths that
    receive an already-HVG-selected array instead of an AnnData (see
    inference.py). Never mutates `artifact`.
    """
    gene_names = list(gene_names)

    seen = set()
    duplicate = sorted({g for g in gene_names if g in seen or seen.add(g)})
    if duplicate:
        raise GeneContractError(
            f"Input contains {len(duplicate)} duplicate gene identifier(s): "
            f"{duplicate[:10]}{' ...' if len(duplicate) > 10 else ''}. Duplicate gene "
            "identifiers make gene selection/reordering ambiguous and are always rejected "
            "— deduplicate the input (there is no supported aggregation policy)."
        )

    present = set(gene_names)
    missing = [g for g in artifact.gene_list if g not in present]
    unexpected = sorted(set(gene_names) - set(artifact.gene_list))
    n_expected = len(artifact.gene_list)
    n_present = n_expected - len(missing)
    coverage = n_present / n_expected if n_expected else 0.0

    diagnostics = {
        "missing": missing, "unexpected": unexpected, "duplicate": duplicate,
        "coverage": coverage, "n_expected": n_expected, "n_present": n_present,
    }

    if missing and artifact.missing_gene_policy == "error":
        raise GeneContractError(
            f"Input is missing {len(missing)} gene(s) required by "
            f"PreprocessingArtifact version {artifact.version}: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "") +
            ". Re-run the same conversion/harmonization pipeline used to fit "
            "this artifact, or refit a new artifact for this gene panel. "
            "(missing_gene_policy='error' is the default; 'zero_fill' is an explicit, "
            "documented, non-default opt-in — see PreprocessingArtifact.missing_gene_policy.)"
        )
    if unexpected and artifact.unexpected_gene_policy == "error":
        raise GeneContractError(
            f"Input contains {len(unexpected)} unexpected gene(s) not in this artifact's "
            f"selected panel: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}. "
            "unexpected_gene_policy='error' is configured for this artifact."
        )
    if coverage < artifact.minimum_gene_coverage:
        raise GeneContractError(
            f"Gene coverage {coverage:.4f} is below this artifact's configured minimum "
            f"({artifact.minimum_gene_coverage:.4f}): {n_present}/{n_expected} required "
            "genes present. Refusing to transform an input this incomplete."
        )
    return diagnostics


class CellTypeProvenanceError(ValueError):
    """Raised when a PreprocessingArtifact's cell-type annotation provenance
    is missing, malformed, or records a mode a real (non-synthetic)
    ExperimentContext must not silently accept. Lives here rather than in
    benchmarks/context.py to avoid a circular import (data/transforms.py,
    which defines cell_type_map_fingerprint(), already sits below
    benchmarks/ in the dependency graph, and this module already owns
    PreprocessingArtifact itself)."""


# The only cell-type annotation modes a real (non-synthetic) pipeline result
# may carry. "inductive_per_cell" (data/transforms.py::annotate_cell_types,
# majority_voting=False) requires a matching cell_type_map_fingerprint.
# "pseudo_bulk_no_cell_type_identity" is the explicit, documented policy for
# pseudo-bulk sources: CellTypist annotation never runs on pseudo-bulk data
# (there is no per-cell expression to classify), so no per-cell mapping
# fingerprint applies — this is a deliberate absence of cell-type identity,
# not a degraded/failed annotation, and callers requiring real per-cell
# identity (e.g. a cell-type-aware MIL pooling path) must reject this mode
# themselves rather than have it silently pass as "the same as annotated".
# "majority_voting" (dataset-dependent, non-inductive) and
# "diagnostic_fallback" (every cell given the same placeholder label after
# CellTypist itself failed) are both real annotation modes that exist in
# this codebase but are NEVER accepted for a real ExperimentContext.
_VALID_REAL_CELL_TYPE_ANNOTATION_MODES = frozenset({
    "inductive_per_cell", "pseudo_bulk_no_cell_type_identity",
})


def validate_cell_type_provenance(artifact: "PreprocessingArtifact") -> None:
    """
    Strict, fail-closed validation of a PreprocessingArtifact's cell-type
    annotation provenance for a REAL (non-synthetic) ExperimentContext.
    Unlike the historical `getattr(artifact, "cell_type_annotation_degraded",
    False)` check this replaces, a MISSING provenance field is never treated
    as "not degraded" / safe — every field below must be explicitly present
    and valid, or this raises CellTypeProvenanceError.
    """
    degraded = artifact.cell_type_annotation_degraded
    mode = artifact.cell_type_annotation_mode
    fp = artifact.cell_type_map_fingerprint

    if degraded is not False:
        raise CellTypeProvenanceError(
            f"PreprocessingArtifact.cell_type_annotation_degraded must be exactly False "
            f"for a real pipeline result, got {degraded!r} (missing/None/True are all "
            "refused — a missing field is never treated as 'not degraded')."
        )
    if mode not in _VALID_REAL_CELL_TYPE_ANNOTATION_MODES:
        raise CellTypeProvenanceError(
            f"PreprocessingArtifact.cell_type_annotation_mode={mode!r} is not one of the "
            f"accepted real-run modes {sorted(_VALID_REAL_CELL_TYPE_ANNOTATION_MODES)} — "
            "missing/None/'majority_voting'/'diagnostic_fallback'/any other mode is refused."
        )
    if mode == "inductive_per_cell":
        if not fp or not isinstance(fp, str) or len(fp) != 64 or any(
            c not in "0123456789abcdef" for c in fp
        ):
            raise CellTypeProvenanceError(
                f"PreprocessingArtifact.cell_type_map_fingerprint is missing or malformed "
                f"({fp!r}) for cell_type_annotation_mode='inductive_per_cell' — a real "
                "per-cell annotation must record a valid 64-character hex SHA-256 "
                "fingerprint of the mapping table used."
            )
        from data.transforms import cell_type_map_fingerprint as _current_cell_type_map_fingerprint
        expected = _current_cell_type_map_fingerprint()
        if fp != expected:
            raise CellTypeProvenanceError(
                f"PreprocessingArtifact.cell_type_map_fingerprint {fp!r} does not match the "
                f"current canonical CELL_TYPE_MAP fingerprint {expected!r} — this artifact's "
                "cell-type annotation was produced under a different mapping table than is "
                "currently in effect. Re-run preprocessing rather than silently trusting a "
                "stale mapping; if this is expected (e.g. an intentional constants.py "
                "change), reprocess and refit a new artifact rather than reusing the old one."
            )
    elif mode == "pseudo_bulk_no_cell_type_identity" and fp is not None:
        raise CellTypeProvenanceError(
            f"PreprocessingArtifact.cell_type_map_fingerprint must be None for "
            f"cell_type_annotation_mode='pseudo_bulk_no_cell_type_identity' (no per-cell "
            f"mapping was ever applied), got {fp!r}."
        )


def verify_input_matrix(artifact: PreprocessingArtifact, gene_order: List[str]) -> None:
    """
    Strict check for raw-array inference inputs (no AnnData to reorder):
    gene_order must match artifact.gene_list exactly, in the same order,
    since a raw float32 matrix has no column labels of its own once it
    reaches the model.
    """
    if list(gene_order) != artifact.gene_list:
        raise ValueError(
            f"Input gene order does not match PreprocessingArtifact version "
            f"{artifact.version} (expected {len(artifact.gene_list)} genes in a "
            "fixed order). Feeding a differently-ordered or differently-sized "
            "gene panel to the model silently produces meaningless predictions "
            "— use apply_preprocessing() on an AnnData instead of hand-building "
            "the input array."
        )
