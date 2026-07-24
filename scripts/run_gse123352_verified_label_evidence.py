"""
scripts/run_gse123352_verified_label_evidence.py

Thin wrapper around evidence.development.run_development for
cohort_id='gse123352', task='smoke_classification' — the same orchestration
`python -m evidence.runner development --cohort gse123352
--task smoke_classification` calls. All real logic (conversion, the bulk
pipeline, real fingerprinting, evidence-report construction, artifact
bundle writing) lives in evidence/development.py; this script exists only
for a convenient standalone entry point and CLI defaults.

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--n-top-variance-genes", type=int, default=2000)
    parser.add_argument("--run-id", type=str, default="gse123352_verified_label_smoke_v1")
    args = parser.parse_args(argv)

    from evidence.cohort_registry import load_cohort_registry
    from evidence.development import run_development
    from evidence.evidence_contract import is_not_evaluable

    cohorts = load_cohort_registry(str(ROOT / "configs" / "cohorts.yaml"))
    result = run_development(
        "gse123352", "smoke_classification", cohorts=cohorts,
        output_root=ROOT / "artifacts" / "evidence", run_id=args.run_id,
        seed=args.seed, train_frac=args.train_frac, test_frac=args.test_frac,
        n_top_variance_genes=args.n_top_variance_genes,
    )

    if is_not_evaluable(result):
        print(f"[run] not_evaluable: {result['reason_code']} — {result['reason']}")
        return 1

    report = result["report"]
    print("\n[run] GSE123352 verified-label smoke classification — real held-out subject metrics:")
    print(json.dumps(report["metrics"], indent=2, default=str))
    print(f"\n[run] wrote evidence artifact bundle to {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
