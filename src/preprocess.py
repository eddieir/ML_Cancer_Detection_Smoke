"""
preprocess.py — Pipeline orchestrator. No logic lives here.
Accepts a config dict or a path to configs/default.yaml.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import tempfile

import numpy as np
import pandas as pd
import yaml

from data.loaders    import load_scrna, load_microarray, load_mouse_scrna
from data.transforms import (
    map_mouse_to_human, harmonize_gene_ids, qc_filter, normalize, smoke_aware_hvg, batch_correct,
    annotate_cell_types, DEFAULT_BATCH_CORRECTION_MODE, DEFAULT_ALLOW_TRANSDUCTIVE_HARMONY,
)
from data.labellers  import transfer_nlst_labels, add_malignancy_labels, compute_smoke_class_weights, apply_weak_smoke_proxies
from data.assembly   import merge_sources, assemble_subject_bags, export_cell_dataset
from data.assay_policy import (
    AssayPolicyError,
    BulkTrainingNotImplementedError,
    MultimodalTrainingNotImplementedError,
    assert_rows_match_policy,
    require_trainable as require_assay_trainable,
    validate_assay_policy,
)
from constants       import N_HVGS_DEFAULT, DEFAULT_ASSAY_POLICY


def load_config(config: Union[dict, str, Path]) -> dict:
    """Accept a plain dict or a path to a YAML config file."""
    if isinstance(config, (str, Path)):
        with open(config) as f:
            return yaml.safe_load(f)
    return config


def _load_all_sources(cfg: dict) -> list:
    """
    Shared source-loading step for run_pipeline() and run_pipeline_split_aware().

    data.experiment_mode (default "human_only" — see constants.py,
    data/species_policy.py) gates whether the mouse source (GSE288003,
    gse288003_path) is loaded at all. human_only (the default) never loads
    it, regardless of whether gse288003_path is configured — a config
    written before species separation existed must not silently start
    mixing mouse cells into a human run just because the path is still
    present in configs/default.yaml. Loading it requires explicitly setting
    experiment_mode to "mouse_only", "cross_species_pretraining", or
    "cross_species_domain_adaptation".

    data.assay_policy (default "single_cell_only" — see constants.py,
    data/assay_policy.py) is the ROW-LEVEL gate for every source loaded
    here: every AnnData returned by load_scrna/load_microarray is checked
    against obs["is_pseudo_bulk"] before it is ever appended to `adatas`.
    This is deliberately NOT a check against assay_mode or a source/
    accession name — see data/assay_policy.py's module docstring for why a
    name-based check (the pre-issue-13 behavior) let GSE994/GSE123352/
    GSE307690 through even though assay_mode='bulk_tcga' rejection was in
    place. A config that lists a pseudo-bulk CSV under scrna_sources/
    microarray_sources under the default single_cell_only policy is
    rejected here regardless of what the file is named.
    """
    from constants import EXPERIMENT_MODE_HUMAN_ONLY
    from data.species_policy import species_allowed, validate_experiment_mode
    from constants import SPECIES_HUMAN, SPECIES_MOUSE

    experiment_mode = validate_experiment_mode(cfg.get("experiment_mode", EXPERIMENT_MODE_HUMAN_ONLY))
    assay_policy = validate_assay_policy(cfg.get("assay_policy", DEFAULT_ASSAY_POLICY))
    adatas = []

    def _exists(path: str) -> bool:
        ok = Path(path).exists()
        if not ok:
            print(f"[preprocess] skip  {path}  (not found — run downloaders.py / converters.py)")
        return ok

    def _enforce_assay_policy(a, path: str):
        """
        Row-provenance assay-policy gate applied to every source this
        function loads (see docstring above). Primarily keyed on
        obs["is_pseudo_bulk"] (the fix for issue #13: GSE994/GSE123352/
        GSE307690 are pseudo-bulk but were never stamped assay_mode=
        'bulk_tcga', so a name/assay_mode-only check missed them). Also
        keeps the original defensive assay_mode=='bulk_tcga' check as a
        second, independent signal — a row explicitly tagged bulk_tcga must
        never enter the single-cell pipeline even if is_pseudo_bulk was
        (incorrectly) left False by whatever produced it.
        """
        if "is_pseudo_bulk" not in a.obs.columns:
            raise AssayPolicyError(
                f"{path!r} loaded with no obs['is_pseudo_bulk'] column — every source must "
                "carry explicit row-level assay provenance before it can enter the pipeline."
            )
        try:
            assert_rows_match_policy(
                a.obs["is_pseudo_bulk"].values, assay_policy,
                context=f"source loading: {path!r}",
            )
        except AssayPolicyError as exc:
            raise AssayPolicyError(
                f"{exc} Configure it under data.bulk_sources / data.tcga.bulk_sources instead "
                "of data.scrna_sources/data.microarray_sources, or set data.assay_policy to "
                "'bulk_only' for a dedicated bulk experiment."
            ) from exc

        if "assay_mode" in a.obs.columns:
            from constants import ASSAY_MODE_BULK_TCGA
            tagged_bulk = (a.obs["assay_mode"] == ASSAY_MODE_BULK_TCGA).values
            if tagged_bulk.any() and assay_policy == "single_cell_only":
                raise AssayPolicyError(
                    f"{path!r} contains {int(tagged_bulk.sum())} assay_mode='bulk_tcga' row(s) — "
                    "bulk TCGA data must never enter the human single-cell pipeline. Remove it "
                    "from data.scrna_sources/data.microarray_sources and configure it under "
                    "data.tcga.bulk_sources instead (see preprocess.py::load_tcga_bulk_dataset)."
                )
        return a

    if species_allowed(experiment_mode, SPECIES_HUMAN):
        for path, stype, scol in cfg.get("scrna_sources", []):
            if _exists(path):
                a = _enforce_assay_policy(load_scrna(path, stype, scol), path)
                adatas.append(normalize(qc_filter(harmonize_gene_ids(a))))

        for path, stype in cfg.get("microarray_sources", []):
            if _exists(path):
                a = _enforce_assay_policy(load_microarray(path, stype), path)
                adatas.append(normalize(harmonize_gene_ids(a)))

    if species_allowed(experiment_mode, SPECIES_MOUSE):
        if cfg.get("gse288003_path") and _exists(cfg["gse288003_path"]):
            a = map_mouse_to_human(load_mouse_scrna(cfg["gse288003_path"]))
            adatas.append(normalize(qc_filter(a)))
    elif cfg.get("gse288003_path"):
        print(f"[preprocess] skip  {cfg['gse288003_path']}  (mouse source — "
              f"experiment_mode={experiment_mode!r} does not include species=mouse; "
              "set data.experiment_mode to mouse_only/cross_species_pretraining/"
              "cross_species_domain_adaptation to include it)")

    if not adatas:
        raise ValueError(
            "No data sources found. Run:\n"
            "  python3 src/data/downloaders.py --all\n"
            "  python3 src/data/converters.py --all\n"
            "or pass a config with paths to already-converted files."
        )
    return adatas


def _load_outcomes(cfg: dict) -> Optional[pd.DataFrame]:
    """Shared cancer-outcome loading step for run_pipeline() and run_pipeline_split_aware()."""
    outcome_sources = []
    if cfg.get("nlst_outcomes_csv") and Path(cfg["nlst_outcomes_csv"]).exists():
        outcome_sources.append(pd.read_csv(cfg["nlst_outcomes_csv"]))
    for path in cfg.get("extra_outcomes_csvs", []):
        if Path(path).exists():
            outcome_sources.append(pd.read_csv(path))

    if not outcome_sources:
        return None
    # A subject appearing in more than one source (e.g. NLST + TCGA) is a
    # cancer positive if any source says so.
    return (
        pd.concat(outcome_sources, ignore_index=True)
        .astype({"subject_id": str})
        .groupby("subject_id", as_index=False)["cancer_label"].max()
    )


def load_tcga_bulk_dataset(config: Union[dict, str, Path], require_trainable: bool = False) -> dict:
    """
    Dedicated bulk_tcga loading path — the ONLY sanctioned way to load
    TCGA's bulk expression matrices in this project. Never call
    load_microarray on a TCGA CSV from the human single-cell pipeline (see
    preprocess.py::_load_all_sources's _enforce_assay_policy guard, which
    exists precisely to catch that mistake).

    config keys (under data.tcga):
      enabled       bool — must be true, or this raises immediately.
      bulk_sources  list[(csv_path, project_id)]

    Returns {"project_id": AnnData} for every configured project whose CSV
    exists, each AnnData carrying assay_mode="bulk_tcga",
    is_pseudo_bulk=True, smoke_type_known=False (see data/converters.py::
    convert_tcga — no verified per-patient smoking history), and a real
    bulk sample_type-derived malignancy field that is explicitly a BULK
    SAMPLE label (tumor vs. solid-tissue-normal), not a per-cell call.

    require_trainable=True additionally raises BulkTrainingNotImplementedError
    — there is no bulk model/training loop in this project yet. This
    function's job is limited to loading/validating the bulk manifest, per
    the documented scope: bulk training stays disabled with an actionable
    error rather than silently reusing the single-cell model on bulk data.
    """
    from constants import ASSAY_MODE_BULK_TCGA
    full_cfg = load_config(config)
    cfg = full_cfg.get("data", full_cfg)
    tcga_cfg = cfg.get("tcga", {})

    if not tcga_cfg.get("enabled", False):
        raise ValueError(
            "load_tcga_bulk_dataset: data.tcga.enabled is false (the default) — TCGA bulk "
            "loading is disabled by default. Set data.tcga.enabled=true to load/validate the "
            "configured data.tcga.bulk_sources. This does not enable bulk TRAINING (see "
            "require_trainable/BulkTrainingNotImplementedError)."
        )

    datasets = {}
    for path, project_id in tcga_cfg.get("bulk_sources", []):
        if not Path(path).exists():
            print(f"[preprocess] skip  {path}  (bulk_tcga source not found — "
                  f"run: python3 src/data/converters.py --tcga {project_id})")
            continue
        adata = load_microarray(path, "unknown")
        if "assay_mode" not in adata.obs.columns or not (adata.obs["assay_mode"] == ASSAY_MODE_BULK_TCGA).all():
            raise ValueError(
                f"{path!r} did not load with assay_mode='bulk_tcga' for every sample — "
                "check that it was produced by data/converters.py::convert_tcga (which writes "
                "the required samples_meta.csv assay_mode column) and not hand-edited."
            )
        datasets[project_id] = adata
        print(f"[preprocess] bulk_tcga  {project_id}  {adata.n_obs:,} samples x {adata.n_vars:,} genes "
              f"(assay_mode=bulk_tcga, smoke_type_known=False)")

    if not datasets:
        raise ValueError(
            "load_tcga_bulk_dataset: data.tcga.enabled=true but no configured bulk_sources "
            "were found on disk. Run: python3 src/data/downloaders.py --tcga --token ... "
            "then python3 src/data/converters.py --tcga <PROJECT>."
        )

    if require_trainable:
        raise BulkTrainingNotImplementedError(
            f"Loaded/validated {len(datasets)} TCGA bulk project(s) "
            f"({sorted(datasets)}), but this project has no bulk RNA-seq model or training "
            "loop implemented — only manifest/query/validation/loading support exists. "
            "Bulk training is intentionally left disabled rather than silently reusing the "
            "single-cell model on bulk expression."
        )
    return datasets


def run_pipeline(config: Union[dict, str, Path]) -> Tuple[dict, list]:
    """
    Full preprocessing pipeline from raw files to training-ready arrays.

    WARNING — preprocessing leakage: this function scales (merge_sources)
    and selects highly-variable genes (smoke_aware_hvg) across the ENTIRE
    merged dataset before any train/val/test split exists, so validation/
    test cells influence those statistics. It has no notion of a split at
    all. Kept only for backward compatibility with existing callers/tests
    and for quick synthetic smoke-testing. For any real train/val/test
    experiment, use run_pipeline_split_aware() instead, which determines a
    subject-level split first and fits scaling/HVG selection on the train
    split only (see data/preprocessing.py, data/splitting.py).

    Parameters
    ----------
    config : dict | str | Path
        Plain dict or path to configs/default.yaml.

    config keys
    -----------
    scrna_sources      list[(h5ad_path, smoke_type, subject_col)]
    microarray_sources list[(csv_path, smoke_type)]
    gse288003_path     str | None   mouse e-cig → triggers ortholog mapping
    nlst_csv           str | None   cigar/dual-use label transfer
    nlst_outcomes_csv  str | None   subject_id + cancer_label for Phase 2/3
    extra_outcomes_csvs list | None additional subject_id + cancer_label sources
                                    (e.g. TCGA tumor/NAT outcomes from converters.py)
    tumor_barcodes     list | None
    n_hvgs             int          default 2000
    out_dir            str

    Returns
    -------
    cell_data   dict  →  CellLevelDataset  (Phase 1)
    bags        list  →  SubjectLevelDataset (Phase 2/3)
    """
    cfg    = load_config(config)
    cfg    = cfg.get("data", cfg)  # configs/default.yaml nests these under "data:"
    adatas = _load_all_sources(cfg)
    assay_policy = validate_assay_policy(cfg.get("assay_policy", DEFAULT_ASSAY_POLICY))

    from constants import EXPERIMENT_MODE_HUMAN_ONLY
    merged = merge_sources(
        *adatas, experiment_mode=cfg.get("experiment_mode", EXPERIMENT_MODE_HUMAN_ONLY),
        assay_policy=assay_policy,
    )
    merged = smoke_aware_hvg(merged, n_hvgs=cfg.get("n_hvgs", N_HVGS_DEFAULT))
    merged = batch_correct(merged)
    # cell_type_allow_diagnostic_fallback defaults to False (fail-closed on
    # any CellTypist failure or scikit-learn/CellTypist version mismatch —
    # see data/transforms.py::annotate_cell_types). No real production
    # config sets this key; it exists only for a deliberate, disclosed
    # diagnostic run, and any output produced with it set is stamped
    # degraded and rejected by real ExperimentContext construction.
    merged = annotate_cell_types(
        merged, allow_diagnostic_fallback=cfg.get("cell_type_allow_diagnostic_fallback", False)
    )

    if cfg.get("nlst_csv"):
        merged = transfer_nlst_labels(merged, cfg["nlst_csv"])

    merged = add_malignancy_labels(merged, cfg.get("tumor_barcodes"))
    merged = apply_weak_smoke_proxies(merged, enabled=cfg.get("weak_labels", {}).get("enabled", False))

    cell_data = export_cell_dataset(merged, cfg.get("out_dir", "data/processed"), assay_policy=assay_policy)

    outcome_sources = []
    if cfg.get("nlst_outcomes_csv") and Path(cfg["nlst_outcomes_csv"]).exists():
        outcome_sources.append(pd.read_csv(cfg["nlst_outcomes_csv"]))
    for path in cfg.get("extra_outcomes_csvs", []):
        if _exists(path):
            outcome_sources.append(pd.read_csv(path))

    outcomes: Optional[pd.DataFrame] = None
    if outcome_sources:
        # A subject appearing in more than one source (e.g. NLST + TCGA) is
        # a cancer positive if any source says so.
        outcomes = (
            pd.concat(outcome_sources, ignore_index=True)
            .astype({"subject_id": str})
            .groupby("subject_id", as_index=False)["cancer_label"].max()
        )

    bags = assemble_subject_bags(
        merged, outcomes,
        min_cells_per_subject=cfg.get("min_cells_per_subject", 50),
        assay_policy=assay_policy,
    )
    return cell_data, bags


def _nlst_join_report(cfg: dict, merged) -> dict:
    """
    How many subjects in this merged dataset actually matched an NLST
    record, independent of transfer_nlst_labels()'s own print — so the
    split-aware pipeline can save this fact in label_provenance_report
    instead of only logging it. NLST provides clinical smoking-category
    labels (cigar/dual-use), NOT a gene-expression-to-outcome link — this
    report exists partly to make that distinction explicit and auditable.
    """
    nlst_csv = cfg.get("nlst_csv")
    if not nlst_csv or not Path(nlst_csv).exists():
        return {
            "nlst_csv_used": False, "n_subjects_matched": 0,
            "note": "No NLST CSV configured/found — no NLST label transfer occurred.",
        }
    nlst = pd.read_csv(nlst_csv, low_memory=False)
    nlst_subjects = set(nlst["pid"].astype(str)) if "pid" in nlst.columns else set()
    data_subjects = set(merged.obs["subject_id"].astype(str))
    matched = nlst_subjects & data_subjects

    # Per-cell known/unknown breakdown among cells transfer_nlst_labels
    # actually touched (identified by smoke_type_source, stamped only by
    # that function — see data/nlst_smoking.py) — a matched subject whose
    # CIGSMOK/CIGAR didn't parse to a documented positive code counts as
    # "matched but unknown", not silently folded into n_subjects_matched
    # as if a label was produced.
    n_cells_verified = n_cells_unknown = 0
    if "smoke_type_source" in merged.obs.columns:
        from data.nlst_smoking import SOURCE_NLST_CIGSMOK_CIGAR
        nlst_touched = merged.obs["smoke_type_source"] == SOURCE_NLST_CIGSMOK_CIGAR
        if nlst_touched.any():
            known = merged.obs.loc[nlst_touched, "smoke_type_known"].astype(bool)
            n_cells_verified = int(known.sum())
            n_cells_unknown = int((~known).sum())

    return {
        "nlst_csv_used": True,
        "n_subjects_matched": len(matched),
        "nlst_cells_verified_smoke_label": n_cells_verified,
        "nlst_cells_unknown_smoke_label": n_cells_unknown,
        "note": (
            None if matched else
            "NLST CSV present but ZERO subject ID overlap with this dataset — no "
            "labels were actually transferred. NLST provides clinical smoking-category "
            "data, not a gene-expression-to-outcome link, and does not apply here."
        ),
    }


def run_pipeline_split_aware(config: Union[dict, str, Path]) -> dict:
    """
    Leakage-free preprocessing pipeline. Fixed order (see README.md /
    ARCHITECTURE.md "Scientific Validity" sections for the rationale):

      1. load sources, harmonize/normalize (no full-dataset fitting)
      2. attach final smoke labels + provenance (NLST label transfer),
         attach malignancy labels + provenance
      3. apply the configured rare-class policy to the FINAL label
         (effective label used by everything downstream; raw label
         preserved as obs["smoke_type_raw"])
      4. determine the subject-level train/val/test split using the FINAL
         effective label (never the pre-transfer label)
      5. fit scaling/HVG selection on the train split only
      6. apply that fit unchanged to every split
      7. batch correction (Harmony) — SKIPPED by default (see
         preprocessing.batch_correction.allow_transductive_harmony);
         Harmony has no train-only-fit/apply-to-new-data mode, so running
         it is a transductive step, not a leakage-free one, and is opt-in
      8. cell type annotation, export

    config keys — same `data:` keys as run_pipeline(), plus:
      split:        train_frac/val_frac/test_frac/seed/manifest_path/
                     force_regenerate (data/splitting.py)
      rare_class:   policy/target_classes/min_subjects_required
                     (data/rare_class.py) — actually applied here, not just
                     available as an unused utility
      preprocessing.batch_correction.allow_transductive_harmony: bool
                     (default False — strict/non-transductive by default)

    Returns a dict with BOTH legacy whole-dataset keys (cell_data, bags —
    every split, for backward compatibility / manual filtering) and
    explicit per-split keys that are the only sanctioned way to build
    Trainer inputs:
      train_cell_dataset / val_cell_dataset / test_cell_dataset : CellLevelDataset
      train_bags / val_bags / test_bags                          : list[dict]
      split_manifest         — data.splitting.SplitManifest
      preprocessing_artifact — data.preprocessing.PreprocessingArtifact
      label_mapping          — data.label_mapping.EffectiveLabelMapping: the
                                deterministic, contiguous (0..K-1) effective
                                smoke-label space actually used by the split,
                                the exported cell dataset, and the bags —
                                also embedded in preprocessing_artifact.label_mapping
      rare_class_report      — data.rare_class.apply_rare_class_policy's report
      label_provenance_report — NLST join count + known/unknown outcome counts
      transductive_batch_correction — bool, whether full-data Harmony ran
    """
    full_cfg  = load_config(config)
    cfg       = full_cfg.get("data", full_cfg)
    split_cfg = full_cfg.get("split", {})
    rare_cfg  = full_cfg.get("rare_class", {})
    pp_cfg    = full_cfg.get("preprocessing", {})

    from data.preprocessing import fit_preprocessing, apply_preprocessing
    from data.rare_class import apply_rare_class_policy
    from data.label_mapping import build_effective_label_mapping
    from data.splitting import load_or_create_split, subject_train_val_test_split
    from train import CellLevelDataset

    from constants import EXPERIMENT_MODE_HUMAN_ONLY
    experiment_mode = cfg.get("experiment_mode", EXPERIMENT_MODE_HUMAN_ONLY)
    assay_policy = validate_assay_policy(cfg.get("assay_policy", DEFAULT_ASSAY_POLICY))
    adatas = _load_all_sources(cfg)
    merged = merge_sources(  # gene intersection + concat only, NOT scaled
        *adatas, scale=False, experiment_mode=experiment_mode, assay_policy=assay_policy,
    )

    # ── 2. Final labels BEFORE anything that depends on them ────────────────
    if cfg.get("nlst_csv"):
        merged = transfer_nlst_labels(merged, cfg["nlst_csv"])
    merged = add_malignancy_labels(merged, cfg.get("tumor_barcodes"))
    weak_labels_enabled = cfg.get("weak_labels", {}).get("enabled", False)
    merged = apply_weak_smoke_proxies(merged, enabled=weak_labels_enabled)
    nlst_report = _nlst_join_report(cfg, merged)

    # ── 3. Rare-class policy on the FINAL label — actually wired in ─────────
    merged.obs["smoke_type_raw"] = merged.obs["smoke_type"].astype(int)  # original, never mutated
    rare_policy = rare_cfg.get("policy", "keep_with_warning")
    effective_ids, keep_mask, rare_report = apply_rare_class_policy(
        merged.obs["smoke_type"].values,
        merged.obs["subject_id"].values,
        policy=rare_policy,
        target_classes=rare_cfg.get("target_classes", []),
        min_subjects_required=rare_cfg.get("min_subjects_required", 3),
    )
    merged.obs["smoke_type"] = effective_ids   # effective label — used by split + everything after
    if not keep_mask.all():
        n_dropped = int((~keep_mask).sum())
        merged = merged[keep_mask].copy()
        print(f"[preprocess] rare_class policy={rare_policy!r} excluded {n_dropped:,} cells")

    # Deterministic contiguous effective label space (0..K-1), built directly
    # from the policy report — never inferred from a particular split. A
    # merged-away or excluded raw class must not leave a dead, unreachable
    # output in the model or an always-zero-support row in every metric.
    label_mapping = build_effective_label_mapping(rare_report)
    merged.obs["smoke_type"] = label_mapping.transform(merged.obs["smoke_type"].values)
    print(f"[preprocess] effective smoke-label space: K={label_mapping.k}  "
          f"classes={label_mapping.class_names}  policy={rare_policy!r}")

    # ── 4. Subject-level split on the FINAL effective label ─────────────────
    # Stratify using None (not the numeric placeholder) for any cell with
    # smoke_type_known=False — a subject whose smoke label is entirely
    # unknown (e.g. GSE136831 under the default verified_only policy) must
    # not be silently counted as a member of whatever placeholder class its
    # numeric smoke_type happens to hold; splitting.py already supports
    # per-subject label=None (pooled, unstratified placement — see
    # subject_train_val_test_split's `unstratified_classes` report), which
    # is the correct treatment here: the subject is still placed into
    # exactly one split, just not used to balance smoke-type class
    # proportions across splits.
    if "smoke_type_known" in merged.obs.columns:
        strat_labels = merged.obs["smoke_type"].astype(object).where(
            merged.obs["smoke_type_known"].astype(bool), None
        ).values
    else:
        strat_labels = merged.obs["smoke_type"].values

    split_kwargs = dict(
        train_frac=split_cfg.get("train_frac", 0.70),
        val_frac=split_cfg.get("val_frac", 0.15),
        test_frac=split_cfg.get("test_frac", 0.15),
        seed=split_cfg.get("seed", 42),
        rare_class_policy=rare_policy,
    )
    manifest_path = split_cfg.get("manifest_path")
    if manifest_path:
        manifest = load_or_create_split(
            manifest_path,
            merged.obs["subject_id"].values, strat_labels,
            force_regenerate=split_cfg.get("force_regenerate", False),
            **split_kwargs,
        )
    else:
        manifest = subject_train_val_test_split(
            merged.obs["subject_id"].values, strat_labels,
            **split_kwargs,
        )

    # Cell-type annotation BEFORE the refit snapshot. CellTypist
    # (annotate_cell_types) is pretrained-model inference, not data-fit: it
    # loads a fixed pretrained classifier and only reads obs["is_pseudo_bulk"]
    # and layers["lognorm"] (restored explicitly since merge_sources leaves
    # .X as z-scores) — it never fits/trains on this dataset, so running it
    # here adds no leakage risk. annotate_cell_types() defaults to
    # majority_voting=False, CellTypist's own inductive mode: each cell's
    # predicted label is a pure function of that cell's own expression
    # vector, independent of which other cells are supplied in the same
    # call. A held-out validation/test cell therefore cannot change a
    # training cell's annotation, and annotating a different subject subset
    # per CV fold reproduces the SAME per-cell labels rather than silently
    # reassigning them — see data/transforms.py::annotate_cell_types for the
    # full rationale. It is still run exactly once here (not once per fold)
    # purely as a performance/consistency convenience — inductive per-cell
    # prediction makes that a choice, not a leakage requirement. Previously
    # this ran AFTER the refit-snapshot line below, which meant every
    # fold-reconstructed cell got cell_type_id=0 (the placeholder) instead
    # of its real annotation — silently destroying cell-type proportions and
    # MIL cell-type-id inputs for every benchmark CV fold and OOD evaluation.
    # See run_pipeline() above for cell_type_allow_diagnostic_fallback's
    # fail-closed-by-default contract — identical here.
    merged = annotate_cell_types(
        merged, allow_diagnostic_fallback=cfg.get("cell_type_allow_diagnostic_fallback", False)
    )

    # Snapshot the full-gene, normalized-but-not-yet-HVG-selected-or-scaled
    # AnnData BEFORE fit_preprocessing/apply_preprocessing run. Neither
    # function mutates `merged` in place (both return new objects), so this
    # is a cheap reference, not a copy — and it's exactly the "pre-feature-
    # selection, normalized/log-transformed expression, WITH real cell-type
    # annotations" a benchmark needs to refit its own PreprocessingArtifact
    # per CV fold (see src/benchmarks/fold_preprocessing.py) instead of
    # reusing this one artifact (fit on ALL original-train subjects) across
    # every fold, which would leak an inner-CV-validation subject's
    # influence on scaling/HVG selection into that same fold's "held-out"
    # evaluation. fit_preprocessing/apply_preprocessing only ever touch .X
    # (gene expression) and never obs["cell_type_id"], so cell-type labels
    # here are exactly what every fold and the outer split will see.
    normalized_adata_for_refit = merged

    # ── Resolve the batch-correction mode BEFORE fitting preprocessing, so
    # the artifact's own batch_correction_status field records the actual
    # resolved mode this run will use, rather than a value patched on after
    # construction (data/preprocessing.py::PreprocessingArtifact is treated
    # as immutable once built — see fit/apply contract notes there). Harmony
    # itself still runs AFTER apply_preprocessing below (unchanged
    # ordering); only the mode validation/lookup moved earlier.
    bc_cfg = pp_cfg.get("batch_correction", {})
    bc_mode = bc_cfg.get("mode", DEFAULT_BATCH_CORRECTION_MODE)
    valid_bc_modes = {"none", "train_fitted_inductive", "transductive_diagnostic_only"}
    if bc_mode not in valid_bc_modes:
        raise ValueError(
            f"preprocessing.batch_correction.mode={bc_mode!r} is not one of {sorted(valid_bc_modes)}."
        )
    if bc_mode == "train_fitted_inductive":
        raise ValueError(
            "preprocessing.batch_correction.mode='train_fitted_inductive' was requested, but "
            "Harmony (this project's only implemented batch-correction method) has no "
            "train-only-fit / apply-to-new-data transform — there is no inductive "
            "implementation to run. Use mode='none' (default, leakage-free) or explicitly "
            "opt into mode='transductive_diagnostic_only' for a disclosed non-leakage-free "
            "diagnostic run outside frozen-test evaluation."
        )
    # Legacy boolean is equivalent to mode='transductive_diagnostic_only'.
    allow_transductive = (
        bc_mode == "transductive_diagnostic_only"
        or bc_cfg.get("allow_transductive_harmony", DEFAULT_ALLOW_TRANSDUCTIVE_HARMONY)
    )

    # ── 5/6. Fit preprocessing on train only, apply to everyone ─────────────
    artifact = fit_preprocessing(
        merged, set(manifest.train_subjects),
        n_hvgs=cfg.get("n_hvgs", N_HVGS_DEFAULT),
        batch_correction_status=(
            "transductive_diagnostic_only" if allow_transductive else "disabled"
        ),
        assay_policy=assay_policy,
    )
    artifact.label_mapping = label_mapping.to_dict()
    # Cell-type annotation provenance (which fixed label-name -> ID table
    # produced obs["cell_type_id"], set by annotate_cell_types above) is
    # already copied onto `artifact` by fit_preprocessing() itself, straight
    # from `merged.uns` — see data/preprocessing.py::fit_preprocessing.
    merged = apply_preprocessing(merged, artifact)

    # ── 7. Batch correction: strict (skipped) unless explicitly opted in ────
    if allow_transductive:
        print("[preprocess] preprocessing.batch_correction.allow_transductive_harmony=True — "
              "running Harmony across the FULL merged dataset (train+val+test). This step is "
              "TRANSDUCTIVE, not leakage-free: held-out expression values influence the batch "
              "correction embedding. Do not describe this experiment as fully leakage-free.")
        merged = batch_correct(merged)
        transductive_used = "batch" in merged.obs.columns and merged.obs["batch"].nunique() >= 2
    else:
        print("[preprocess] batch correction SKIPPED (strict mode, default) — set "
              "preprocessing.batch_correction.allow_transductive_harmony=true to opt into "
              "transductive Harmony correction across the full dataset.")
        transductive_used = False

    # cell-type annotation already ran once, above, before the refit
    # snapshot — see the comment there for why it must not run twice.

    # ── 8. Export + assemble ─────────────────────────────────────────────────
    cell_data = export_cell_dataset(merged, cfg.get("out_dir", "data/processed"), assay_policy=assay_policy)
    outcomes  = _load_outcomes(cfg)
    bags = assemble_subject_bags(
        merged, outcomes,
        min_cells_per_subject=cfg.get("min_cells_per_subject", 50),
        assay_policy=assay_policy,
    )
    n_outcome_known = sum(1 for b in bags if b.get("cancer_label_known"))
    n_outcome_unknown = len(bags) - n_outcome_known

    # ── Persist preprocessing artifact next to the split manifest AND the
    # training checkpoint dir (Predictor.from_config looks for it there) ────
    out_dir = Path(cfg.get("out_dir", "data/processed"))
    artifact.save(out_dir / "preprocessing_artifact.json")
    ckpt_dir_cfg = full_cfg.get("train", {}).get("checkpoint_dir")
    if ckpt_dir_cfg:
        ckpt_dir = Path(ckpt_dir_cfg)
        if not ckpt_dir.is_absolute():
            ckpt_dir = Path(__file__).parents[1] / ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        artifact.save(ckpt_dir / "preprocessing_artifact.json")

    # ── Explicit per-split datasets — the only sanctioned Trainer inputs ────
    full_cell_dataset = CellLevelDataset(
        gene_matrix       = cell_data["gene_matrix"],
        smoke_labels      = cell_data["smoke_labels"],
        malignancy_labels = cell_data["malignancy_labels"],
        cell_type_ids     = cell_data["cell_type_ids"],
        exposure_dose     = cell_data["exposure_dose"],
        malignancy_known  = cell_data["malignancy_known"],
        subject_ids       = merged.obs["subject_id"].astype(str).values,
        is_pseudo_bulk     = merged.obs["is_pseudo_bulk"].values,
        assay_policy       = assay_policy,
    )
    train_cell_dataset = full_cell_dataset.subset_by_subjects(manifest.train_subjects)
    val_cell_dataset   = full_cell_dataset.subset_by_subjects(manifest.val_subjects)
    test_cell_dataset  = full_cell_dataset.subset_by_subjects(manifest.test_subjects)

    bags_by_subject = {str(b["subject_id"]): b for b in bags}
    train_bags = [bags_by_subject[s] for s in manifest.train_subjects if s in bags_by_subject]
    val_bags   = [bags_by_subject[s] for s in manifest.val_subjects   if s in bags_by_subject]
    test_bags  = [bags_by_subject[s] for s in manifest.test_subjects  if s in bags_by_subject]

    return {
        # legacy / whole-dataset (backward compatibility, manual filtering)
        "cell_data": cell_data,
        "bags": bags,
        "split_manifest": manifest,
        "preprocessing_artifact": artifact,
        "label_mapping": label_mapping,
        # explicit per-split datasets — use these for Trainer
        "train_cell_dataset": train_cell_dataset,
        "val_cell_dataset":   val_cell_dataset,
        "test_cell_dataset":  test_cell_dataset,
        "train_bags": train_bags,
        "val_bags":   val_bags,
        "test_bags":  test_bags,
        # provenance / auditability
        "rare_class_report": rare_report,
        "label_provenance_report": {
            **nlst_report,
            "cancer_outcome_known_subjects":   n_outcome_known,
            "cancer_outcome_unknown_subjects": n_outcome_unknown,
            "malignancy_known_cells":   int(cell_data["malignancy_known"].sum()),
            "malignancy_unknown_cells": int((~cell_data["malignancy_known"]).sum()),
            # Verified vs. weak-proxy vs. unknown smoke labels — see
            # data/labellers.py::apply_weak_smoke_proxies and
            # data/converters.py::_load_gse136831_cell_metadata.
            "weak_labels_enabled":       weak_labels_enabled,
            "smoke_verified_known_cells": int(cell_data["smoke_labels_known"].sum()),
            "smoke_unknown_cells":        int((~cell_data["smoke_labels_known"]).sum()),
            "smoke_weak_proxy_cells": (
                int(merged.obs["weak_smoke_proxy_known"].astype(bool).sum())
                if "weak_smoke_proxy_known" in merged.obs.columns else 0
            ),
        },
        "transductive_batch_correction": transductive_used,
        # pre-HVG, pre-scaling normalized AnnData — see the comment at its
        # assignment above. Benchmarks-only; run_pipeline_split_aware's other
        # callers/tests are unaffected by this additional key.
        "normalized_adata_for_refit": normalized_adata_for_refit,
    }


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import anndata as ad
    import scipy.sparse as sp
    from constants import ALL_SMOKE_MARKERS

    print("=== preprocess.py smoke test ===\n")
    np.random.seed(42)
    N, G = 600, 3000

    genes = [f"GENE_{i}" for i in range(G)]
    genes[:10] = [f"MT-{i}" for i in range(10)]
    for i, mk in enumerate(ALL_SMOKE_MARKERS[:8]):
        genes[100 + i] = mk

    raw = np.random.negative_binomial(5, 0.7, (N, G)).astype("float32")
    obs = pd.DataFrame({
        "donor_id":       [f"sub_{i // 20}" for i in range(N)],
        "subject_id":     [f"sub_{i // 20}" for i in range(N)],
        "smoke_type":     np.random.randint(0, 6, N),
        "smoke_type_name": "mixed",
        "data_modality":  "scrna",
        "is_pseudo_bulk": False,
        "malignancy":     0.0,
        "cell_type_id":   0,
    }, index=[f"c{i}" for i in range(N)])

    adata = ad.AnnData(X=sp.csr_matrix(raw), obs=obs, var=pd.DataFrame(index=genes))

    # Save temp h5ad so run_pipeline() is tested end-to-end
    with tempfile.TemporaryDirectory() as tmp:
        h5ad_path = str(Path(tmp) / "test.h5ad")
        out_dir   = str(Path(tmp) / "processed")
        adata.write_h5ad(h5ad_path)

        config = {
            "scrna_sources":          [(h5ad_path, "cigarette", "donor_id")],
            "microarray_sources":     [],
            "n_hvgs":                 200,
            "min_cells_per_subject":  10,
            "out_dir":                out_dir,
        }

        cell_data, bags = run_pipeline(config)

    print(f"\nbags : {len(bags)}")
    print(f"keys : {list(bags[0].keys())}")
    print(f"shape: {bags[0]['gene_matrix'].shape}")
    print("\n=== PASSED ===")