"""
inference.py — Production prediction interface for MultiSmokeCancerNet.

Three roles, one file, zero duplication:
  Trainer   (train.py)    → trains the model
  Evaluator (evaluate.py) → evaluates with known labels
  Predictor (this file)   → predicts on new unlabelled subjects

Entry points:
  predictor.predict_subject(gene_matrix, cell_type_ids)
  predictor.predict_batch([{"gene_matrix": X, "cell_type_ids": ct, ...}])
  predictor.predict_h5ad("subject.h5ad")
  python3 src/inference.py --h5ad subject.h5ad --out results.json
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import yaml

from constants import CELL_TYPES, N_CELL_TYPES, SMOKE_TYPES
from model import MultiSmokeCancerNet


# ─── DRY output formatter ─────────────────────────────────────────────────────

def _format_result(
    subject_id:   str,
    cancer_prob:  float,
    attn:         np.ndarray,   # [N]
    smoke_probs:  np.ndarray,   # [N, 6]
    malignancy:   np.ndarray,   # [N]
) -> Dict:
    """
    Single source of truth for the prediction output schema.
    Called by every predict_* method — output is always identical in shape.
    """
    dominant_smoke = smoke_probs.argmax(axis=1)
    smoke_profile  = {
        name: round(float((dominant_smoke == idx).mean()), 4)
        for idx, name in SMOKE_TYPES.items()
    }
    return {
        "subject_id":           subject_id,
        "cancer_probability":   round(cancer_prob, 4),
        "risk_flag":            ("HIGH"     if cancer_prob >= 0.70 else
                                 "MODERATE" if cancer_prob >= 0.40 else "LOW"),
        "top5_cells":           attn.argsort()[-5:][::-1].tolist(),
        "smoke_profile":        smoke_profile,
        "dominant_smoke_type":  max(smoke_profile, key=smoke_profile.get),
        "mean_malignancy":      round(float(malignancy.mean()), 4),
        "malignancy_percentiles": {
            "p25": round(float(np.percentile(malignancy, 25)), 4),
            "p50": round(float(np.percentile(malignancy, 50)), 4),
            "p75": round(float(np.percentile(malignancy, 75)), 4),
            "p95": round(float(np.percentile(malignancy, 95)), 4),
        },
        "attention_weights":    attn.tolist(),
    }


# ─── Predictor ────────────────────────────────────────────────────────────────

class Predictor:
    """
    Makes cancer risk predictions on new, unlabelled subjects.

    Expects preprocessed gene matrices (output of preprocess.py run_pipeline).
    For raw H5AD files, call predict_h5ad() which handles loading automatically.
    """

    def __init__(
        self,
        model:                  MultiSmokeCancerNet,
        device:                 str = "cpu",
        preprocessing_artifact: "Optional[object]" = None,
    ):
        """
        preprocessing_artifact, if given (a data.preprocessing.PreprocessingArtifact),
        is used to validate that any raw AnnData passed to predict_h5ad() has
        the exact gene panel/order the model was trained on before running
        inference — see data/preprocessing.py::verify_compatible. Without it,
        a caller can silently feed a mismatched gene panel and get a
        confident-looking but meaningless prediction.
        """
        self.model  = model.to(device).eval()
        self.device = device
        self.preprocessing_artifact = preprocessing_artifact

    @classmethod
    def from_config(
        cls,
        config: Union[dict, str, Path],
        phase:  int = 3,
        device: str = "cpu",
    ) -> "Predictor":
        """
        Load best checkpoint from a given training phase. If
        checkpoint_dir/preprocessing_artifact.json exists (see
        data/preprocessing.py::PreprocessingArtifact.save), it's loaded too
        so predict_h5ad() can validate incoming data's gene panel.
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("[inference] WARNING: CUDA not available — falling back to CPU")
            device = "cpu"
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        model    = MultiSmokeCancerNet.from_config(config)
        ckpt_dir = Path(config.get("train", config).get("checkpoint_dir", "checkpoints"))
        if not ckpt_dir.is_absolute():
            ckpt_dir = Path(__file__).parents[1] / ckpt_dir
        ckpt = ckpt_dir / f"phase{phase}_best.pt"
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True)
        )
        print(f"[inference] loaded phase {phase} checkpoint  ({ckpt})")

        artifact = None
        artifact_path = ckpt_dir / "preprocessing_artifact.json"
        if artifact_path.exists():
            from data.preprocessing import PreprocessingArtifact
            artifact = PreprocessingArtifact.load(artifact_path)
            print(f"[inference] loaded preprocessing artifact  ({artifact_path})")

        return cls(model, device, preprocessing_artifact=artifact)

    # ── Core prediction — all other methods call this ─────────────────────────

    def predict_subject(
        self,
        gene_matrix:   np.ndarray,   # [N, genes] float32, preprocessed
        cell_type_ids: np.ndarray,   # [N] int
        subject_id:    str = "subject",
    ) -> Dict:
        """
        Predict cancer risk for one subject.
        gene_matrix must be preprocessed (log-normalised, HVG-selected, scaled).
        """
        with torch.no_grad():
            out = self.model.forward_subject(
                torch.FloatTensor(gene_matrix).to(self.device),
                torch.LongTensor(cell_type_ids).to(self.device),
            )
        return _format_result(
            subject_id  = subject_id,
            cancer_prob = out["cancer_probability"].item(),
            attn        = out["attention_weights"].cpu().numpy(),
            smoke_probs = out["cell_smoke_probs"].cpu().numpy(),
            malignancy  = out["cell_malignancy"].squeeze().cpu().numpy(),
        )

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch(self, subjects: List[Dict]) -> List[Dict]:
        """
        Predict cancer risk for a list of subjects.

        Each dict must contain:
          gene_matrix   : np.ndarray [N, genes]
          cell_type_ids : np.ndarray [N]
          subject_id    : str (optional, defaults to index)
        """
        return [
            self.predict_subject(
                s["gene_matrix"],
                s["cell_type_ids"],
                s.get("subject_id", f"subject_{i}"),
            )
            for i, s in enumerate(subjects)
        ]

    # ── H5AD prediction ───────────────────────────────────────────────────────

    def predict_h5ad(
        self,
        h5ad_path:     Union[str, Path],
        subject_col:   str = "subject_id",
        cell_type_col: str = "cell_type_id",
    ) -> List[Dict]:
        """
        Predict cancer risk directly from a preprocessed H5AD file.

        Expects the AnnData to have:
          .X               : scaled gene expression [N_cells, N_genes]
          .obs[subject_col]: subject identifier per cell
          .obs[cell_type_col]: integer cell type (0-3)

        Groups cells by subject_id and runs predict_batch.
        Raises ValueError if required obs columns are missing.
        """
        import anndata as ad
        adata = ad.read_h5ad(h5ad_path)
        self._validate_h5ad(adata, subject_col, cell_type_col)
        if self.preprocessing_artifact is not None:
            from data.preprocessing import verify_compatible
            verify_compatible(self.preprocessing_artifact, adata.var_names)

        X = np.array(
            adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray(),
            dtype=np.float32,
        )
        subjects = []
        for sid in adata.obs[subject_col].unique():
            mask = (adata.obs[subject_col] == sid).values
            subjects.append({
                "subject_id":    str(sid),
                "gene_matrix":   X[mask],
                "cell_type_ids": adata.obs[cell_type_col].values[mask].astype(int),
            })

        print(f"[inference] {len(subjects)} subjects from {Path(h5ad_path).name}")
        return self.predict_batch(subjects)

    # ── Save results ──────────────────────────────────────────────────────────

    def save_results(
        self,
        results: Union[Dict, List[Dict]],
        out_path: Union[str, Path],
    ) -> None:
        """Save prediction results to JSON."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[inference] results → {out}")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_h5ad(adata, subject_col: str, cell_type_col: str) -> None:
        missing = [c for c in [subject_col, cell_type_col] if c not in adata.obs.columns]
        if missing:
            raise ValueError(
                f"H5AD missing obs columns: {missing}\n"
                "Run preprocess.py run_pipeline() first to add these columns."
            )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_cli():
    import argparse
    p = argparse.ArgumentParser(description="MultiSmokeCancerNet inference")
    p.add_argument("--config", default="configs/default.yaml",  help="Path to default.yaml")
    p.add_argument("--phase",  type=int, default=3,             help="Checkpoint phase to load (1/2/3)")
    p.add_argument("--h5ad",   type=str, default=None,          help="Preprocessed H5AD file")
    p.add_argument("--out",    type=str, default=None,          help="Output JSON path")
    p.add_argument("--device", type=str, default="cpu",         help="cpu or cuda")
    return p


def _cli_main():
    args = _build_cli().parse_args()
    predictor = Predictor.from_config(args.config, phase=args.phase, device=args.device)

    if args.h5ad:
        results = predictor.predict_h5ad(args.h5ad)
    else:
        _build_cli().print_help()
        return

    for r in results:
        print(f"  {r['subject_id']:<20} P(cancer)={r['cancer_probability']:.4f}"
              f"  {r['risk_flag']:<8} smoke={r['dominant_smoke_type']}")

    if args.out:
        predictor.save_results(results, args.out)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Route to CLI when arguments are provided, smoke test otherwise.
    if len(sys.argv) > 1:
        _cli_main()
        sys.exit(0)
    import random
    torch.manual_seed(42)
    np.random.seed(42)

    CFG   = Path(__file__).parents[1] / "configs" / "default.yaml"
    GENES = 2000

    model     = MultiSmokeCancerNet.from_config(CFG)
    predictor = Predictor(model)   # use untrained model — no checkpoint needed

    # ── predict_subject ───────────────────────────────────────────────────────
    n = random.randint(50, 120)
    r = predictor.predict_subject(
        np.random.randn(n, GENES).astype("float32"),
        np.random.randint(0, N_CELL_TYPES, n),
        subject_id="test_subject",
    )
    assert r["subject_id"]         == "test_subject"
    assert 0.0 <= r["cancer_probability"] <= 1.0
    assert r["risk_flag"]          in ("HIGH", "MODERATE", "LOW")
    assert set(r["smoke_profile"]) == set(SMOKE_TYPES.values())
    assert len(r["top5_cells"])    == 5
    assert len(r["attention_weights"]) == n
    print(f"predict_subject  ✓  P(cancer)={r['cancer_probability']:.4f}"
          f"  flag={r['risk_flag']}  smoke={r['dominant_smoke_type']}")

    # ── predict_batch ─────────────────────────────────────────────────────────
    subjects = [
        {
            "subject_id":    f"batch_sub_{i}",
            "gene_matrix":   (X := np.random.randn(n := random.randint(40, 100), GENES).astype("float32")),
            "cell_type_ids": np.random.randint(0, N_CELL_TYPES, n),
        }
        for i in range(5)
    ]
    results = predictor.predict_batch(subjects)
    assert len(results) == 5
    assert all(r["subject_id"] == f"batch_sub_{i}" for i, r in enumerate(results))
    print(f"predict_batch    ✓  {len(results)} subjects predicted")

    # ── save_results ──────────────────────────────────────────────────────────
    out_path = Path(__file__).parents[1] / "checkpoints" / "inference_results.json"
    predictor.save_results(results, out_path)
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert len(saved) == 5
    print(f"save_results     ✓  {out_path.name}")

    # ── _format_result schema is consistent ───────────────────────────────────
    required_keys = {
        "subject_id", "cancer_probability", "risk_flag",
        "top5_cells", "smoke_profile", "dominant_smoke_type",
        "mean_malignancy", "malignancy_percentiles", "attention_weights",
    }
    assert required_keys.issubset(results[0].keys()), \
        f"Missing keys: {required_keys - results[0].keys()}"
    print(f"output schema    ✓  all {len(required_keys)} keys present")

    print("\n=== PASSED ===")