"""
data/transforms.py — pure data transformation. No I/O, no label logic.
Each function takes AnnData, returns AnnData.
"""

import re
from typing import Optional
import numpy as np
import anndata as ad
import scanpy as sc

from constants import ALL_SMOKE_MARKERS


# ─── Gene ID harmonization ────────────────────────────────────────────────────

# Platform-specific probe/gene ID patterns -> BioMart attribute name that maps
# them to human gene symbols. Detected by sniffing var_names, so callers don't
# need to know each source's platform ahead of time.
_ID_PATTERNS: dict[str, "re.Pattern"] = {
    "affy_hg_u133a_2": re.compile(r"^\d+(_[a-z]+)?_at$", re.I),
    "ensembl_gene_id": re.compile(r"^ENSG\d+(\.\d+)?$"),
}


def _detect_platform(var_names) -> Optional[str]:
    sample = list(var_names[:20])
    if not sample:
        return None
    for platform, pattern in _ID_PATTERNS.items():
        if all(pattern.match(str(v)) for v in sample):
            return platform
    return None


def harmonize_gene_ids(adata: ad.AnnData, mapping: Optional[dict] = None) -> ad.AnnData:
    """
    Maps platform-specific probe/gene IDs (Affymetrix HG-U133A probes,
    Ensembl gene IDs) to human gene symbols via BioMart, so merge_sources()
    can find real overlap across heterogeneous sources instead of the zero
    overlap you get when one source uses "1007_s_at" and another uses
    "ENSG00000000003" for the same gene.

    Auto-detects platform from var_names; sources already using gene symbols
    (no pattern match) pass through unchanged. Illumina HumanHT-12 probes
    (e.g. GSE123352's "ILMN_...") are NOT covered here — Ensembl's BioMart
    doesn't expose that array as a queryable attribute, so that mapping
    happens earlier, at conversion time, via GEO's own GPL platform
    annotation file (converters.py::_load_probe_to_symbol_map,
    GSE123352.csv already ships real gene symbols by the time it reaches
    this pipeline) rather than being silently mismapped here.

    `mapping` lets tests (and repeat calls across sources on the same
    platform) skip the live BioMart query.
    """
    platform = _detect_platform(adata.var_names)
    if platform is None:
        return adata  # already gene symbols, or an unrecognised/unsupported platform

    if mapping is None:
        try:
            from pybiomart import Dataset
            ds = Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")
            df = ds.query(attributes=[platform, "external_gene_name"]).dropna()
            mapping = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        except Exception as e:
            print(f"[transform] gene ID mapping unavailable ({e}) — skipping harmonization")
            return adata

    new_names = [mapping.get(g, "") for g in adata.var_names]
    valid = [i for i, n in enumerate(new_names) if n]
    adata = adata[:, valid].copy()
    adata.var_names = [new_names[i] for i in valid]
    adata.var_names_make_unique()
    print(f"[transform] {platform}  {len(valid):,}/{len(new_names):,} probes → gene symbols")
    return adata


def map_mouse_to_human(
    adata: ad.AnnData,
    artifact: Optional[object] = None,
    artifact_path: Optional[str] = None,
) -> ad.AnnData:
    """
    Convert mouse gene symbols → human orthologs using a versioned
    OrthologMappingArtifact (data/ortholog.py) — never a silent "pick the
    first match" for a mouse gene with multiple candidate human orthologs.

    Resolution order:
      1. `artifact` (an OrthologMappingArtifact instance) if passed directly
         — the way tests/fixtures inject a small fixed mapping without any
         network access.
      2. `artifact_path` — a cached artifact previously saved via
         OrthologMappingArtifact.save().
      3. A live PyBiomart query (data/ortholog.py::fetch_live_biomart_pairs),
         resolved through the same one_to_one_only policy and NOT cached to
         disk automatically (callers building a reusable artifact should do
         that once via ortholog.py's CLI/build_ortholog_artifact and pass
         artifact_path from then on).

    Failure (no network, PyBiomart unavailable, and neither artifact nor
    artifact_path given) prints and returns `adata` unchanged, matching
    this function's previous best-effort behaviour — a caller in a
    network-isolated environment should pass a cached artifact_path
    instead of relying on this fallback.

    Novel: enables GSE288003 (only vape scRNA-seq) to train in human gene space.
    """
    from data.ortholog import OrthologMappingArtifact, build_ortholog_artifact, fetch_live_biomart_pairs

    if artifact is None and artifact_path:
        artifact = OrthologMappingArtifact.load(artifact_path)

    if artifact is None:
        try:
            pairs = fetch_live_biomart_pairs()
            artifact = build_ortholog_artifact(pairs, source="ensembl_biomart_live")
        except Exception as e:
            print(f"[transform] ortholog mapping unavailable ({e}) — skipping")
            return adata

    omap = artifact.mapping
    new_names = [omap.get(g, "") for g in adata.var_names]
    valid     = [i for i, n in enumerate(new_names) if n]
    adata     = adata[:, valid].copy()
    adata.var_names = [new_names[i] for i in valid]
    adata.var_names_make_unique()
    adata.uns["ortholog_mapping_fingerprint"] = artifact.fingerprint()
    adata.uns["ortholog_mapping_policy"] = artifact.policy
    print(f"[transform] {len(valid):,}/{len(new_names):,} "
          f"mouse genes → human orthologs retained (policy={artifact.policy}, "
          f"fingerprint {artifact.fingerprint()[:12]}...)")
    return adata


def qc_filter(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
) -> ad.AnnData:
    """Standard QC. Skips per-cell thresholds for pseudo-bulk."""
    if adata.obs["is_pseudo_bulk"].all():
        sc.pp.filter_genes(adata, min_cells=1)
        return adata
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # percent_top defaults to [50,100,200,500] and raises IndexError on any
    # dataset with fewer than 500 genes (e.g. HVG-reduced or small test data);
    # we don't use that metric, so disable it rather than requiring >=500 genes.
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    n_before = adata.n_obs
    adata = adata[adata.obs.n_genes_by_counts >= min_genes].copy()
    adata = adata[adata.obs.pct_counts_mt    <= max_pct_mito].copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"[transform] qc  {n_before:,} → {adata.n_obs:,} cells")
    return adata


def normalize(adata: ad.AnnData) -> ad.AnnData:
    """CPM + log1p. Skips if microarray (already log-transformed)."""
    adata.layers["counts"] = adata.X.copy()
    if adata.obs["data_modality"].iloc[0] == "microarray":
        adata.layers["lognorm"] = adata.X.copy()   # already log-scaled
        return adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()        # preserve before any scaling
    return adata


def smoke_aware_hvg(
    adata: ad.AnnData,
    n_hvgs: int = 2000,
    batch_key: Optional[str] = "batch",
) -> ad.AnnData:
    """
    Variance-based HVG selection with forced inclusion of smoke markers.
    Novel: prevents standard HVG from dropping smoke-discriminative genes
    when one smoke type dominates the dataset distribution.
    """
    bk = batch_key if (batch_key and batch_key in adata.obs.columns) else None
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, batch_key=bk)

    present = [g for g in ALL_SMOKE_MARKERS if g in adata.var_names]
    forced  = [g for g in present if not adata.var.loc[g, "highly_variable"]]

    if forced:
        non_marker_hvgs = [
            g for g in adata.var_names[adata.var["highly_variable"]]
            if g not in present
        ]
        drop = non_marker_hvgs[-len(forced):]
        adata.var.loc[drop,   "highly_variable"] = False
        adata.var.loc[forced, "highly_variable"] = True
        print(f"[transform] hvg  forced {len(forced)} smoke markers in")

    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"[transform] hvg  {adata.n_vars} genes selected")
    return adata


def batch_correct(adata: ad.AnnData, batch_key: str = "batch") -> ad.AnnData:
    """Harmony batch correction on PCA embeddings."""
    if batch_key not in adata.obs.columns or adata.obs[batch_key].nunique() < 2:
        print("[transform] harmony skipped — fewer than 2 batches")
        return adata
    import harmonypy as hm
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)
    ho = hm.run_harmony(
        adata.obsm["X_pca"], adata.obs, batch_key, max_iter_harmony=20
    )
    # harmonypy's Z_corr orientation has flipped across versions (some
    # return n_pcs x n_cells, others n_cells x n_pcs) — orient against the
    # known n_obs rather than assuming a fixed convention.
    z_corr = ho.Z_corr if ho.Z_corr.shape[0] == adata.n_obs else ho.Z_corr.T
    adata.obsm["X_pca_harmony"] = z_corr
    print(f"[transform] harmony  {adata.obs[batch_key].nunique()} batches corrected")
    return adata


def cell_type_map_fingerprint() -> str:
    """
    SHA-256 of the fixed CELL_TYPE_MAP label-name -> ID table (constants.py),
    so a checkpoint/artifact can record and later verify exactly which
    mapping produced its cell_type_id column. The map itself is a static
    dict, never built from data (held-out frequency/order never enters it),
    so this fingerprint is the same for any run of this codebase — a
    mismatch on reload means the code's mapping table itself changed, not
    that different cells were annotated.
    """
    import hashlib
    import json
    from constants import CELL_TYPE_MAP
    blob = json.dumps(CELL_TYPE_MAP, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class CellTypistCompatibilityError(RuntimeError):
    """Raised whenever CellTypist's pretrained model was unpickled with a
    scikit-learn version different from the one installed, UNLESS the
    caller explicitly opted into a diagnostic run (allow_diagnostic_fallback
    =True — see annotate_cell_types). scikit-learn's own model-persistence
    documentation states cross-version unpickling is unsupported and "may
    lead to breaking code or invalid results" — this is an unresolved
    upstream risk (CellTypist's published Immune_All_Low.pkl was serialized
    with scikit-learn 0.24.1; this project has been tested against
    scikit-learn >=1.4, currently 1.9.0 in CI) that this project cannot fix
    by itself, since it does not control CellTypist's published model
    artifact. This is a fail-closed default: a real (non-diagnostic)
    scientific run must never silently treat a version-incompatible
    prediction as valid. There is deliberately no environment variable that
    can weaken this for a real run — the only way to proceed past this
    error is the explicit, always-degraded allow_diagnostic_fallback=True
    path. If no compatible CellTypist model is available, the honest
    resolution is that real preprocessing intentionally fails closed until
    one is: see README.md's documented limitation."""


def _load_celltypist_model_checking_sklearn_compatibility(model_name: str, strict: bool):
    """Loads a CellTypist pretrained model, detecting (never silently
    hiding) a scikit-learn version mismatch between how the model was
    pickled and the scikit-learn version installed here. The
    InconsistentVersionWarning sklearn itself raises during unpickling is
    always re-emitted so it still reaches the caller/CI logs exactly as
    before; strict=True additionally turns it into a hard failure.

    Returns (model, compatibility) where `compatibility` is a JSON-safe dict
    recording every fact needed to judge whether this annotation's results
    can be treated as scientifically valid on reload — never fabricated,
    never silently dropped: celltypist_version, model_name,
    runtime_sklearn_version, serialized_sklearn_versions (one per estimator
    sklearn warned about, deduplicated), compatible (False if any
    InconsistentVersionWarning fired, True if none did), and
    diagnostic_override_used (True only when a mismatch occurred and strict
    was False, i.e. the caller's diagnostic opt-out is the only reason this
    call did not raise)."""
    import warnings

    import celltypist as _celltypist_pkg
    import sklearn as _sklearn_pkg
    from celltypist import models
    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model = models.Model.load(model=model_name)
    version_mismatches = [w for w in caught if issubclass(w.category, InconsistentVersionWarning)]
    for w in caught:
        warnings.warn_explicit(w.message, w.category, w.filename or __file__, w.lineno or 0)

    serialized_versions = sorted({
        getattr(w.message, "original_sklearn_version", None) for w in version_mismatches
        if getattr(w.message, "original_sklearn_version", None) is not None
    })
    compatibility = {
        "celltypist_version": getattr(_celltypist_pkg, "__version__", None),
        "model_name": model_name,
        "runtime_sklearn_version": _sklearn_pkg.__version__,
        "serialized_sklearn_versions": serialized_versions,
        "compatible": not version_mismatches,
        "diagnostic_override_used": bool(version_mismatches) and not strict,
    }

    if version_mismatches and strict:
        raise CellTypistCompatibilityError(
            f"CellTypist pretrained model {model_name!r} (celltypist "
            f"{compatibility['celltypist_version']}) was serialized under "
            f"scikit-learn {serialized_versions}, but this run has scikit-learn "
            f"{_sklearn_pkg.__version__} installed: {version_mismatches[0].message}. "
            "Refusing to continue a real scientific run with a version-inconsistent "
            "estimator. This project cannot fix the mismatch itself — it does not "
            "control CellTypist's published model artifact. Remediation: obtain or "
            "publish a CellTypist model reserialized against a matching scikit-learn "
            "version, or pass allow_diagnostic_fallback=True to annotate_cell_types "
            "for a deliberate, explicitly degraded, non-scientific diagnostic run only "
            "(never for a real benchmark — real ExperimentContext construction rejects "
            "degraded provenance)."
        )
    return model, compatibility


class CellTypeAnnotationError(RuntimeError):
    """Raised when CellTypist annotation fails and no diagnostic fallback
    was explicitly requested. Real benchmark runs must never silently
    continue with missing, stale, or fabricated cell-type labels — a prior
    version of this function printed a warning and returned `adata`
    unchanged on failure, leaving obs["cell_type_id"] either absent or
    stale from a previous call, which then surfaced only much later as a
    confusing downstream KeyError or a silently wrong MIL cell-type input."""


_DIAGNOSTIC_FALLBACK_LABEL = "epithelial"


def annotate_cell_types(
    adata: ad.AnnData, majority_voting: bool = False, allow_diagnostic_fallback: bool = False,
) -> ad.AnnData:
    """
    CellTypist per-cell prediction -> coarse 4-class cell_type_id.

    majority_voting defaults to False (INDUCTIVE): CellTypist's
    majority_voting=True mode over-clusters the supplied cells and
    replaces each cell's raw prediction with its cluster's majority label —
    a step whose output for a given cell depends on which OTHER cells are
    present in the same call. Running it once over train+val+test cells (as
    this pipeline used to) therefore lets held-out validation/test cells
    change a training cell's annotation, and reannotating a different
    subject subset per CV fold would silently reassign cell_type_id for
    cells that didn't change at all.

    majority_voting=False instead returns CellTypist's raw per-cell
    prediction from the fixed pretrained model: a pure function of that
    cell's own expression vector, independent of any other cell supplied in
    the same call. A cell's annotation is therefore identical whether it is
    annotated alone, with its own split, or with the full merged dataset —
    the property every fold/OOD reconstruction in benchmarks/ depends on.
    This is CellTypist's own default (majority_voting=False); this pipeline
    previously opted OUT of that default by passing majority_voting=True.

    The label-name -> ID table (CELL_TYPE_MAP) is a fixed dict in
    constants.py, never built from this dataset's label frequency/order —
    see cell_type_map_fingerprint() for the persisted proof of that.

    Failure behavior: if CellTypist itself fails (missing dependency,
    unreachable model download, malformed input, ...), this function raises
    CellTypeAnnotationError by default — a real scientific run must never
    silently proceed with no/fabricated cell types. allow_diagnostic_fallback
    =True is the ONLY way to continue past that failure; it is meant for a
    deliberate small-scale diagnostic run only (never for a real benchmark:
    src/preprocess.py's real pipeline entry points never pass it). The
    fallback assigns EVERY cell the same fixed placeholder label/ID and
    stamps adata.uns["cell_type_annotation_degraded"] = True plus the
    failure reason, so any downstream consumer (ExperimentContext
    validation in particular — see benchmarks/context.py) can detect and
    reject a degraded annotation rather than silently treating it as real.

    scikit-learn/CellTypist compatibility: CellTypist's published
    Immune_All_Low.pkl model was serialized with an older scikit-learn
    (0.24.1) than this project runs against (>=1.4, currently 1.9.0 in CI);
    loading it always emits scikit-learn's own InconsistentVersionWarning,
    which is never hidden here. A real (allow_diagnostic_fallback=False,
    the default) call FAILS CLOSED on that warning — it raises
    CellTypistCompatibilityError undisturbed, not wrapped in
    CellTypeAnnotationError, so the specific remediation-bearing error
    reaches the caller. There is no environment variable that can weaken
    this for a real run. allow_diagnostic_fallback=True is the only way to
    tolerate the mismatch, and doing so always stamps the result degraded
    (cell_type_annotation_degraded=True) so real ExperimentContext
    construction (benchmarks/context.py, via
    data/preprocessing.py::validate_cell_type_provenance) rejects it.
    """
    from constants import CELL_TYPE_MAP
    if adata.obs["is_pseudo_bulk"].all():
        # Explicit, documented pseudo-bulk policy (never an implicit
        # "missing provenance defaults to safe"): pseudo-bulk sources have
        # no per-cell expression for CellTypist to classify, so no
        # per-cell cell-type mapping is ever applied. This is stamped as
        # its own distinct, non-degraded mode — deliberately NOT
        # "inductive_per_cell" (no annotation ran, so no
        # cell_type_map_fingerprint applies) and NOT "degraded" (this is
        # an intentional, expected skip, not a CellTypist failure). A
        # consumer whose model path requires real per-cell cell-type
        # identity (e.g. cell-type-aware MIL pooling) must reject this
        # mode itself — data/preprocessing.py::validate_cell_type_provenance
        # accepts it as valid PROVENANCE, which is a distinct question from
        # whether a given downstream model can operate without cell-type
        # identity at all.
        adata.uns["cell_type_map_fingerprint"] = None
        adata.uns["cell_type_annotation_mode"] = "pseudo_bulk_no_cell_type_identity"
        adata.uns["cell_type_annotation_degraded"] = False
        print("[transform] celltypist skipped for pseudo-bulk "
              "(cell_type_annotation_mode=pseudo_bulk_no_cell_type_identity)")
        return adata
    try:
        import celltypist

        # CellTypist requires log1p normalized X (CPM → log1p).
        # After merge_sources() X holds z-scores, so we restore lognorm layer.
        ct_input = adata.copy()
        if "lognorm" in adata.layers:
            ct_input.X = adata.layers["lognorm"]

        model, compatibility = _load_celltypist_model_checking_sklearn_compatibility(
            "Immune_All_Low.pkl", strict=not allow_diagnostic_fallback
        )
        pred  = celltypist.annotate(ct_input, model=model, majority_voting=majority_voting)
        labels = (
            pred.predicted_labels.majority_voting
            if majority_voting else
            pred.predicted_labels.predicted_labels
        )
        adata.obs["cell_type_name"] = labels.values
        adata.obs["cell_type_id"]   = (
            adata.obs["cell_type_name"]
            .map(lambda c: CELL_TYPE_MAP.get(c, 0))
            .astype(int)
        )
        adata.uns["cell_type_map_fingerprint"] = cell_type_map_fingerprint()
        adata.uns["cell_type_annotation_mode"] = "majority_voting" if majority_voting else "inductive_per_cell"
        # A diagnostic override that tolerated a real version mismatch must
        # never be indistinguishable from a genuinely compatible run — stamp
        # it degraded so real ExperimentContext construction rejects it
        # (see data/preprocessing.py::validate_cell_type_provenance).
        adata.uns["cell_type_annotation_degraded"] = bool(compatibility["diagnostic_override_used"])
        if compatibility["diagnostic_override_used"]:
            adata.uns["cell_type_fallback_reason"] = (
                "scikit-learn/CellTypist version mismatch tolerated via "
                "allow_diagnostic_fallback=True: " + repr(compatibility)
            )
        # Persisted so a checkpoint/reproducibility artifact can record
        # exactly which CellTypist/scikit-learn combination produced this
        # annotation and whether sklearn itself considers it version-
        # consistent — see _load_celltypist_model_checking_sklearn_compatibility.
        adata.uns["cell_type_annotation_compatibility"] = compatibility
        print(f"[transform] celltypist  {adata.n_obs:,} cells annotated "
              f"({'majority_voting' if majority_voting else 'inductive per-cell'})")
    except CellTypistCompatibilityError:
        # Never wrap this in the generic CellTypeAnnotationError below — its
        # specific remediation guidance (versions involved, model name, how
        # to opt into a diagnostic run) must reach the caller undisturbed.
        raise
    except Exception as e:
        if not allow_diagnostic_fallback:
            raise CellTypeAnnotationError(
                f"CellTypist cell-type annotation failed: {e!r}. Refusing to continue with "
                "missing or fabricated cell-type labels for a real run. Pass "
                "allow_diagnostic_fallback=True only for a deliberate small-scale diagnostic "
                "run — never for a real scientific benchmark."
            ) from e
        adata.obs["cell_type_name"] = _DIAGNOSTIC_FALLBACK_LABEL
        adata.obs["cell_type_id"] = int(CELL_TYPE_MAP.get(_DIAGNOSTIC_FALLBACK_LABEL, 0))
        adata.uns["cell_type_map_fingerprint"] = cell_type_map_fingerprint()
        adata.uns["cell_type_annotation_mode"] = "diagnostic_fallback"
        adata.uns["cell_type_annotation_degraded"] = True
        adata.uns["cell_type_fallback_reason"] = repr(e)
        print(f"[transform] celltypist FAILED ({e!r}) — DIAGNOSTIC FALLBACK: every cell labeled "
              f"{_DIAGNOSTIC_FALLBACK_LABEL!r} (degraded, non-scientific annotation; "
              "allow_diagnostic_fallback=True was explicitly set)")
    return adata
