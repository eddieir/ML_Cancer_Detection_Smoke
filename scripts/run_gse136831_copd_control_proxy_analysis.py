"""
scripts/run_gse136831_copd_control_proxy_analysis.py

Runs a real, end-to-end COPD-vs-Control disease-status PROXY sensitivity
analysis against the real downloaded GSE136831 data (Vanderbilt/Habermann
lung scRNA-seq IPF/COPD/Control atlas) and writes the resulting
EvidenceReport through evidence.artifact_bundle into artifacts/evidence/.

GSE136831 has NO verified per-subject cigarette-exposure field — see
configs/cohorts.yaml's gse136831 source_limitations and
data/converters.py::_load_gse136831_cell_metadata. This script does not
produce smoke-classification evidence: COPD is a clinical diagnosis with
strong smoking association but also documented non-smoking causes, and
Control status does not prove never-smoking. See
evidence.tracks.run_copd_control_proxy_analysis, which this script calls
and which stamps task=TASK_PROXY_ANALYSIS (never TASK_SMOKE) and
verified_label_count=0 unconditionally.

Pipeline (reusing existing repository infrastructure end to end, not a new
bespoke one):
  1. data.converters.convert_accession('GSE136831') -> h5ad (real per-cell
     donor_id/disease_identity join from the metadata table).
  2. data.loaders.load_scrna(...) -> AnnData with standard obs columns.
  3. Drop IPF subjects; keep COPD/Control only.
  4. data.splitting.subject_train_val_test_split(...) for a leakage-free
     subject-level train/val/test split.
  5. data.transforms.qc_filter + normalize (CPM + log1p) — the same
     functions the main pipeline uses.
  6. data.preprocessing.fit_preprocessing on the TRAIN subjects only (HVG
     selection + train-only mean/std scaling), then apply_preprocessing to
     transform every cell consistently.
  7. benchmarks.baselines.SmokeLogisticRegression fit on train cells,
     evaluated on held-out test cells, majority-voted to one prediction per
     subject via benchmarks.metrics.subject_weighted_full_smoke_metrics_report
     (called inside evidence.tracks.run_copd_control_proxy_analysis).

Usage:
    PYTHONPATH=src python3 scripts/run_gse136831_copd_control_proxy_analysis.py

Requires the raw GSE136831 files to already be present under
data/raw/cigarette/GSE136831/. Never downloads anything itself. This is a
real, potentially multi-minute run over ~150k+ real cells — it is not a
fast unit test, and is not run as part of the pytest suite.

Memory note: data.preprocessing.fit_preprocessing/apply_preprocessing
densify the full cell x gene matrix to select HVGs and scale (see
data/preprocessing.py — `X = X.toarray()`). The full COPD/Control subset
here is ~165,755 cells x 45,947 genes, which densifies to ~30GB and does
not fit in a 16GB-RAM environment. --max-cells-per-subject deterministically
subsamples (seeded on --seed) each subject down to at most that many real
cells before qc_filter/preprocessing runs, purely for memory feasibility in
this environment — it does not touch labels, does not change which subjects
are held out, and is stamped into the evidence report's limitations so it
is never silently forgotten.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

RAW_DIR = ROOT / "data" / "raw" / "cigarette" / "GSE136831"
RAW_FILE_NAMES = (
    "GSE136831_RawCounts_Sparse.mtx.gz",
    "GSE136831_AllCells.GeneIDs.txt.gz",
    "GSE136831_AllCells.cellBarcodes.txt.gz",
    "GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz",
)

KEEP_DISEASE_IDENTITIES = ("COPD", "Control")  # IPF dropped — not a smoke-exposure proxy
CLASS_NAMES = ["control_proxy", "copd_proxy"]  # index 0 / 1 — disease status, not smoke exposure


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--n-hvgs", type=int, default=2000)
    parser.add_argument("--run-id", type=str, default="gse136831_copd_control_proxy_v1")
    parser.add_argument(
        "--max-cells-per-subject", type=int, default=300,
        help="Deterministic per-subject cell subsample cap, applied before "
             "qc_filter/preprocessing, so the dense HVG-selection matrix fits "
             "in memory in this environment. See module docstring.",
    )
    args = parser.parse_args(argv)

    raw_files = [RAW_DIR / name for name in RAW_FILE_NAMES]
    missing = [str(p) for p in raw_files if not p.exists()]
    if missing:
        print(f"[run] missing real raw file(s): {missing} — nothing was run.")
        return 1

    import numpy as np
    from data.converters import convert_accession
    from data.loaders import load_scrna
    from data.preprocessing import fit_preprocessing, apply_preprocessing
    from data.splitting import subject_train_val_test_split
    from data.transforms import normalize, qc_filter
    from benchmarks.baselines import SmokeLogisticRegression
    from evidence import tracks
    from evidence.artifact_bundle import write_evidence_run
    from evidence.run_identity import (
        baseline_model_fingerprint,
        build_environment_snapshot,
        build_split_manifest,
        environment_snapshot_fingerprint,
        real_git_commit_sha,
        split_manifest_fingerprint as compute_split_manifest_fingerprint,
    )

    cached_h5ad_path = ROOT / "data" / "processed" / "converted" / "GSE136831.h5ad"
    if cached_h5ad_path.exists():
        print(f"[run] reusing already-converted {cached_h5ad_path} from an earlier real "
              "run of this same conversion (same raw files, same converter) instead of "
              "re-reading the ~2.1GB compressed matrix again.")
        h5ad_path = cached_h5ad_path
    else:
        print("[run] converting real GSE136831 raw files to h5ad (this reads the full "
              "~2.1GB compressed sparse matrix — may take several minutes)...")
        h5ad_path = convert_accession("GSE136831")
        if h5ad_path is None:
            print("[run] convert_accession('GSE136831') returned no path — nothing was run.")
            return 1

    adata = load_scrna(str(h5ad_path), smoke_type="unknown", subject_col="donor_id")

    n_total_subjects = adata.obs["subject_id"].nunique()
    disease = adata.obs["disease_identity"].astype(str)
    n_ipf_subjects = adata.obs.loc[disease == "IPF", "subject_id"].nunique()
    keep_mask = disease.isin(KEEP_DISEASE_IDENTITIES).values
    adata = adata[keep_mask].copy()
    n_kept_subjects = adata.obs["subject_id"].nunique()
    print(f"[run] {n_total_subjects} total subjects; dropped {n_ipf_subjects} IPF "
          f"subject(s); kept {n_kept_subjects} COPD/Control subjects "
          f"({adata.n_obs:,} cells).")

    disease_status = (adata.obs["disease_identity"].astype(str) == "COPD").astype(int).values
    adata.obs["copd_control_disease_status"] = disease_status

    n_cells_before_subsample = adata.n_obs
    if args.max_cells_per_subject is not None:
        rng = np.random.default_rng(args.seed)
        subj_arr = adata.obs["subject_id"].astype(str).values
        keep_positions = []
        for subject in sorted(set(subj_arr.tolist())):
            positions = np.flatnonzero(subj_arr == subject)
            if len(positions) > args.max_cells_per_subject:
                positions = rng.choice(positions, size=args.max_cells_per_subject, replace=False)
            keep_positions.append(positions)
        keep_positions = np.sort(np.concatenate(keep_positions))
        adata = adata[keep_positions].copy()
        print(f"[run] subsampled {n_cells_before_subsample:,} -> {adata.n_obs:,} cells "
              f"(cap={args.max_cells_per_subject} cells/subject, seed={args.seed}) "
              "for dense-matrix memory feasibility in this environment.")
        disease_status = adata.obs["copd_control_disease_status"].values

    subject_ids_per_cell = adata.obs["subject_id"].astype(str).tolist()
    labels_per_cell = disease_status.tolist()

    split = subject_train_val_test_split(
        subject_ids=subject_ids_per_cell, labels=labels_per_cell,
        train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac,
        seed=args.seed,
    )
    print(f"[run] subject split: train={len(split.train_subjects)}  "
          f"val={len(split.val_subjects)}  test={len(split.test_subjects)}")

    adata = qc_filter(adata)
    adata = normalize(adata)

    artifact = fit_preprocessing(
        adata, train_subject_ids=set(split.train_subjects), n_hvgs=args.n_hvgs,
        batch_key=None, subject_col="subject_id",
    )
    transformed = apply_preprocessing(adata, artifact)

    subj = transformed.obs["subject_id"].astype(str).values
    train_mask = np.isin(subj, split.train_subjects)
    test_mask = np.isin(subj, split.test_subjects)

    X = transformed.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = transformed.obs["copd_control_disease_status"].values

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test, subj_test = X[test_mask], y[test_mask], subj[test_mask]

    print(f"[run] fitting logistic regression on {X_train.shape[0]:,} train cells x "
          f"{X_train.shape[1]:,} genes...")
    model = SmokeLogisticRegression(class_weight="balanced").fit(X_train, y_train, seed=args.seed)
    y_pred_test = model.predict(X_test)

    extra_limitations = []
    if args.max_cells_per_subject is not None and adata.n_obs < n_cells_before_subsample:
        extra_limitations.append(
            f"Cells were subsampled from {n_cells_before_subsample:,} to {adata.n_obs:,} "
            f"(cap={args.max_cells_per_subject} cells/subject, seed={args.seed}) before "
            "preprocessing, purely because this environment's available memory (16GB) "
            "cannot hold the full dense HVG-selection matrix; subject-level labels and "
            "the held-out subject split are exact and unaffected — only the per-cell "
            "training/evaluation sample size was reduced."
        )

    commit_sha = real_git_commit_sha(ROOT)
    env_snapshot = build_environment_snapshot(commit_sha, ROOT)
    split_manifest = build_split_manifest(
        task=tracks.TASK_PROXY_ANALYSIS, endpoint="copd_vs_control_disease_status",
        dataset_accession="GSE136831",
        train_subject_ids=split.train_subjects, validation_subject_ids=split.val_subjects,
        development_holdout_subject_ids=split.test_subjects,
        seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac,
        stratification_policy="subject_train_val_test_split (label-stratified subject-level split)",
        class_mapping={name: i for i, name in enumerate(CLASS_NAMES)},
        rare_class_policy="not_applicable_binary_proxy_analysis",
        label_state_policy="weak_disease_status_proxy — never a verified smoke-exposure label",
        weak_label_policy="Disease_Identity=COPD/Control used as a disease-status proxy comparator only",
        subject_identity_field="subject_id (donor_id)",
        grouping_policy="all cells from one subject stay in one partition",
        dataset_manifest_fingerprint=tracks._real_raw_file_fingerprint(raw_files),
    )

    report = tracks.run_copd_control_proxy_analysis(
        y_true=y_test.tolist(), y_pred=y_pred_test.tolist(),
        subject_ids=subj_test.tolist(), num_classes=2,
        raw_file_paths=raw_files, class_names=CLASS_NAMES,
        dataset_accession="GSE136831",
        split_manifest_fingerprint=compute_split_manifest_fingerprint(split_manifest),
        model_fingerprint=baseline_model_fingerprint(
            baseline=model, preprocessing_fingerprint=artifact.scientific_fingerprint(),
            train_subject_ids=split.train_subjects,
        ),
        environment_fingerprint=environment_snapshot_fingerprint(env_snapshot),
        preprocessing_artifact_fingerprint=artifact.scientific_fingerprint(),
        excluded_subject_count=n_ipf_subjects,
        excluded_subject_reason=f"{n_ipf_subjects} IPF subject(s) excluded — not a smoking-relevant comparator.",
        random_seed=args.seed,
        extra_limitations=extra_limitations,
    )

    print("\n[run] GSE136831 COPD-vs-Control disease-status proxy analysis — real held-out subject metrics:")
    print(json.dumps(report["metrics"], indent=2, default=str))
    print(f"\n[run] total subjects in accession: {n_total_subjects}  "
          f"(IPF excluded: {n_ipf_subjects}, COPD/Control kept: {n_kept_subjects})")
    print(f"[run] class counts (all kept cells): "
          f"copd_proxy={int(disease_status.sum())}  control_proxy={int((disease_status == 0).sum())}")
    print(f"[run] train cells: {X_train.shape[0]:,}  test cells: {X_test.shape[0]:,}  "
          f"test subjects: {len(set(subj_test.tolist()))}")

    files = {
        "configuration.json": {
            "accession": "GSE136831",
            "seed": args.seed,
            "train_frac": args.train_frac, "val_frac": args.val_frac, "test_frac": args.test_frac,
            "n_hvgs": args.n_hvgs,
        },
        "predictions/test_predictions.csv": [
            {"subject_id": sid, "y_true": int(yt), "y_pred": int(yp)}
            for sid, yt, yp in zip(subj_test.tolist(), y_test.tolist(), y_pred_test.tolist())
        ],
        "metrics/track_a_copd_control_proxy.json": report,
        "cohort_flow.json": {
            "total_subjects_in_accession": int(n_total_subjects),
            "ipf_subjects_excluded": int(n_ipf_subjects),
            "copd_control_subjects_kept": int(n_kept_subjects),
            "train_subjects": split.train_subjects,
            "val_subjects": split.val_subjects,
            "test_subjects": split.test_subjects,
        },
        "preprocessing/artifact.json": artifact.to_dict(),
        "environment.json": env_snapshot,
        "split_manifest.json": split_manifest,
    }
    run_dir = write_evidence_run(
        ROOT / "artifacts" / "evidence", args.run_id, files,
        extra_manifest_fields={
            "dataset_accession": "GSE136831", "track": "A_proxy_analysis",
            "task": tracks.TASK_PROXY_ANALYSIS,
        },
    )
    print(f"\n[run] wrote evidence artifact bundle to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
