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
from data.transforms import map_mouse_to_human, harmonize_gene_ids, qc_filter, normalize, smoke_aware_hvg, batch_correct, annotate_cell_types
from data.labellers  import transfer_nlst_labels, add_malignancy_labels, compute_smoke_class_weights
from data.assembly   import merge_sources, assemble_subject_bags, export_cell_dataset
from constants       import N_HVGS_DEFAULT


def load_config(config: Union[dict, str, Path]) -> dict:
    """Accept a plain dict or a path to a YAML config file."""
    if isinstance(config, (str, Path)):
        with open(config) as f:
            return yaml.safe_load(f)
    return config


def _load_all_sources(cfg: dict) -> list:
    """Shared source-loading step for run_pipeline() and run_pipeline_split_aware()."""
    adatas = []

    def _exists(path: str) -> bool:
        ok = Path(path).exists()
        if not ok:
            print(f"[preprocess] skip  {path}  (not found — run downloaders.py / converters.py)")
        return ok

    for path, stype, scol in cfg.get("scrna_sources", []):
        if _exists(path):
            adatas.append(normalize(qc_filter(harmonize_gene_ids(load_scrna(path, stype, scol)))))

    for path, stype in cfg.get("microarray_sources", []):
        if _exists(path):
            adatas.append(normalize(harmonize_gene_ids(load_microarray(path, stype))))

    if cfg.get("gse288003_path") and _exists(cfg["gse288003_path"]):
        a = map_mouse_to_human(load_mouse_scrna(cfg["gse288003_path"]))
        adatas.append(normalize(qc_filter(a)))

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
    adatas = []

    def _exists(path: str) -> bool:
        ok = Path(path).exists()
        if not ok:
            print(f"[preprocess] skip  {path}  (not found — run downloaders.py / converters.py)")
        return ok

    for path, stype, scol in cfg.get("scrna_sources", []):
        if _exists(path):
            adatas.append(normalize(qc_filter(harmonize_gene_ids(load_scrna(path, stype, scol)))))

    for path, stype in cfg.get("microarray_sources", []):
        if _exists(path):
            adatas.append(normalize(harmonize_gene_ids(load_microarray(path, stype))))

    if cfg.get("gse288003_path") and _exists(cfg["gse288003_path"]):
        a = map_mouse_to_human(load_mouse_scrna(cfg["gse288003_path"]))
        adatas.append(normalize(qc_filter(a)))

    if not adatas:
        raise ValueError(
            "No data sources found. Run:\n"
            "  python3 src/data/downloaders.py --all\n"
            "  python3 src/data/converters.py --all\n"
            "or pass a config with paths to already-converted files."
        )

    merged = merge_sources(*adatas)
    merged = smoke_aware_hvg(merged, n_hvgs=cfg.get("n_hvgs", N_HVGS_DEFAULT))
    merged = batch_correct(merged)
    merged = annotate_cell_types(merged)

    if cfg.get("nlst_csv"):
        merged = transfer_nlst_labels(merged, cfg["nlst_csv"])

    merged = add_malignancy_labels(merged, cfg.get("tumor_barcodes"))

    cell_data = export_cell_dataset(merged, cfg.get("out_dir", "data/processed"))

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
    return {
        "nlst_csv_used": True,
        "n_subjects_matched": len(matched),
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

    adatas = _load_all_sources(cfg)
    merged = merge_sources(*adatas, scale=False)   # gene intersection + concat only, NOT scaled

    # ── 2. Final labels BEFORE anything that depends on them ────────────────
    if cfg.get("nlst_csv"):
        merged = transfer_nlst_labels(merged, cfg["nlst_csv"])
    merged = add_malignancy_labels(merged, cfg.get("tumor_barcodes"))
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
            merged.obs["subject_id"].values, merged.obs["smoke_type"].values,
            force_regenerate=split_cfg.get("force_regenerate", False),
            **split_kwargs,
        )
    else:
        manifest = subject_train_val_test_split(
            merged.obs["subject_id"].values, merged.obs["smoke_type"].values,
            **split_kwargs,
        )

    # Snapshot the full-gene, normalized-but-not-yet-HVG-selected-or-scaled
    # AnnData BEFORE fit_preprocessing/apply_preprocessing run. Neither
    # function mutates `merged` in place (both return new objects), so this
    # is a cheap reference, not a copy — and it's exactly the "pre-feature-
    # selection, normalized/log-transformed expression" a benchmark needs to
    # refit its own PreprocessingArtifact per CV fold (see
    # src/benchmarks/fold_preprocessing.py) instead of reusing this one
    # artifact (fit on ALL original-train subjects) across every fold, which
    # would leak an inner-CV-validation subject's influence on scaling/HVG
    # selection into that same fold's "held-out" evaluation.
    normalized_adata_for_refit = merged

    # ── 5/6. Fit preprocessing on train only, apply to everyone ─────────────
    artifact = fit_preprocessing(
        merged, set(manifest.train_subjects),
        n_hvgs=cfg.get("n_hvgs", N_HVGS_DEFAULT),
    )
    artifact.label_mapping = label_mapping.to_dict()
    merged = apply_preprocessing(merged, artifact)

    # ── 7. Batch correction: strict (skipped) unless explicitly opted in ────
    allow_transductive = pp_cfg.get("batch_correction", {}).get("allow_transductive_harmony", False)
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

    merged = annotate_cell_types(merged)

    # ── 8. Export + assemble ─────────────────────────────────────────────────
    cell_data = export_cell_dataset(merged, cfg.get("out_dir", "data/processed"))
    outcomes  = _load_outcomes(cfg)
    bags = assemble_subject_bags(
        merged, outcomes,
        min_cells_per_subject=cfg.get("min_cells_per_subject", 50),
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