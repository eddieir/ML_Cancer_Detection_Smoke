"""
scripts/run_gse123352_verified_label_evidence.py

Runs a real, end-to-end verified-label smoke-classification evaluation
against the real downloaded GSE123352 data (human bulk microarray, Illumina
HumanHT-12 V4.0, GEO series GSE123352) and writes the resulting
EvidenceReport through evidence.artifact_bundle into artifacts/evidence/.

GSE123352 carries a verified `ever_never_smoker` phenotype field (parsed
directly from the GEO series-matrix sample characteristics — see
data/converters.py::_infer_smoke_column) — unlike GSE136831, this is not a
weak proxy. It is bulk/pseudo-bulk data with no single-cell compatibility,
so this uses the independent path in data/bulk_pipeline.py rather than the
single-cell MIL pipeline (see that module's docstring and
data/assay_policy.py for why the two must never mix).

Usage:
    PYTHONPATH=src python3 scripts/run_gse123352_verified_label_evidence.py

Requires the raw GSE123352 files to already be present under
data/raw/cigarette/GSE123352/ (series matrix, platform annotation,
non-normalized data — see docs/EVIDENCE_PROTOCOL.md). Never downloads
anything itself.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

RAW_DIR = ROOT / "data" / "raw" / "cigarette" / "GSE123352"
CONVERTED_DIR = ROOT / "data" / "processed" / "converted"
RAW_FILE_NAMES = (
    "GSE123352_series_matrix.txt.gz",
    "GSE123352_non-normalized.txt.gz",
    "GPL10558.annot.gz",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--n-top-variance-genes", type=int, default=2000)
    parser.add_argument("--run-id", type=str, default="gse123352_verified_label_smoke_v1")
    args = parser.parse_args(argv)

    raw_files = [RAW_DIR / name for name in RAW_FILE_NAMES]
    missing = [str(p) for p in raw_files if not p.exists()]
    if missing:
        print(f"[run] missing real raw file(s): {missing} — nothing was run.")
        return 1

    from data.converters import convert_accession
    from data import bulk_pipeline
    from evidence import tracks
    from evidence.artifact_bundle import write_evidence_run

    csv_path = convert_accession("GSE123352")
    if csv_path is None:
        print("[run] convert_accession('GSE123352') returned no path — nothing was run.")
        return 1

    result = bulk_pipeline.run_bulk_smoke_classification(
        csv_path,
        seed=args.seed, train_frac=args.train_frac, test_frac=args.test_frac,
        n_top_variance_genes=args.n_top_variance_genes,
    )

    dataset = result["dataset"]
    test_true = result["test"]["y_true"]
    test_pred = result["test"]["y_pred"]
    test_subjects = result["test"]["subject_ids"]

    report = tracks.run_track_a_on_real_verified_label_data(
        y_true=test_true, y_pred=test_pred, subject_ids=test_subjects, num_classes=2,
        raw_file_paths=raw_files,
        class_names=list(bulk_pipeline.SUPPORTED_BULK_CLASSES),
        dataset_accession="GSE123352",
        excluded_subject_count=len(dataset.excluded_sample_ids),
        excluded_subject_reason=(
            f"excluded_sample_ids: {dataset.excluded_reason_counts}" if dataset.excluded_sample_ids else None
        ),
        random_seed=args.seed,
    )

    print("\n[run] GSE123352 verified-label smoke classification — real held-out subject metrics:")
    print(json.dumps(report["metrics"], indent=2, default=str))
    print(f"\n[run] total samples parsed: {dataset.X.shape[0] + len(dataset.excluded_sample_ids)}")
    print(f"[run] verified-label samples kept: {dataset.X.shape[0]}")
    print(f"[run] class counts (kept): "
          f"cigarette={int((dataset.y == 1).sum())}  unexposed={int((dataset.y == 0).sum())}")
    print(f"[run] excluded samples: {len(dataset.excluded_sample_ids)}  reasons={dataset.excluded_reason_counts}")
    print(f"[run] train subjects: {len(result['train']['subject_ids'])}  "
          f"test subjects: {len(test_subjects)}")

    files = {
        "configuration.json": {
            "accession": "GSE123352",
            "seed": args.seed,
            "train_frac": args.train_frac,
            "test_frac": args.test_frac,
            "n_top_variance_genes": args.n_top_variance_genes,
        },
        "predictions/test_predictions.csv": [
            {"subject_id": sid, "y_true": yt, "y_pred": yp}
            for sid, yt, yp in zip(test_subjects, test_true, test_pred)
        ],
        "metrics/track_a_real_verified_label.json": report,
        "cohort_flow.json": {
            "total_samples_parsed": dataset.X.shape[0] + len(dataset.excluded_sample_ids),
            "verified_label_samples_kept": dataset.X.shape[0],
            "excluded_sample_ids": dataset.excluded_sample_ids,
            "excluded_reason_counts": dataset.excluded_reason_counts,
            "train_subjects": result["train"]["subject_ids"],
            "test_subjects": test_subjects,
        },
    }
    run_dir = write_evidence_run(
        ROOT / "artifacts" / "evidence", args.run_id, files,
        extra_manifest_fields={"dataset_accession": "GSE123352", "track": "A"},
    )
    print(f"\n[run] wrote evidence artifact bundle to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
