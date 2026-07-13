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
from data.label_mapping import EffectiveLabelMapping
from data.preprocessing import ARTIFACT_VERSION
from metrics import validate_cell_type_ids
from model import MultiSmokeCancerNet
from train import load_checkpoint_into, read_checkpoint_metadata, _label_mapping_from_checkpoint_meta

# The only input stages predict_h5ad() understands. "raw_counts" is a
# recognized value that is always REJECTED with a clear explanation (see
# predict_h5ad) rather than silently handled — this repository's
# PreprocessingArtifact does not store QC thresholds, a library-size
# normalization target, or log-transform parameters, so it cannot reproduce
# the full raw-count preprocessing chain used at training time.
VALID_INPUT_STAGES = {"model_ready", "normalized_expression", "raw_counts"}


# ─── DRY output formatter ─────────────────────────────────────────────────────

def _format_result(
    subject_id:   str,
    cancer_prob:  float,
    attn:         np.ndarray,   # [N]
    smoke_probs:  np.ndarray,   # [N, K]
    malignancy:   np.ndarray,   # [N]
    class_names:  Optional[List[str]] = None,
) -> Dict:
    """
    Single source of truth for the prediction output schema.
    Called by every predict_* method — output is always identical in shape.

    class_names is indexed by the model's actual effective smoke-label id
    (0..K-1) — defaults to constants.SMOKE_TYPES (six-class, no-merge) only
    when the caller has no EffectiveLabelMapping wired in; using the fixed
    six-class names unconditionally would mislabel predictions whenever a
    rare-class policy shrank the model's output space (data/label_mapping.py).
    """
    class_names = class_names or list(SMOKE_TYPES.values())
    dominant_smoke = smoke_probs.argmax(axis=1)
    smoke_profile  = {
        name: round(float((dominant_smoke == idx).mean()), 4)
        for idx, name in enumerate(class_names)
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
    For H5AD files, call predict_h5ad() which handles loading and validates
    the declared input_stage — see predict_h5ad's docstring. It does NOT
    accept genuinely raw (un-normalized) counts; input_stage="raw_counts" is
    always rejected.
    """

    def __init__(
        self,
        model:                  MultiSmokeCancerNet,
        device:                 str = "cpu",
        preprocessing_artifact: "Optional[object]" = None,
        unsafe_legacy_mode:     bool = False,
        label_mapping:          Optional["EffectiveLabelMapping"] = None,
    ):
        """
        preprocessing_artifact, if given (a data.preprocessing.PreprocessingArtifact),
        is used by predict_h5ad() to reorder/subset genes to the exact panel
        the model was trained on and apply the same train-fit mean/std
        scaling (data/preprocessing.py::apply_preprocessing) before running
        inference. Without it, predict_h5ad() refuses to run unless
        unsafe_legacy_mode=True is explicitly passed — feeding a
        differently-ordered, differently-scaled, or mismatched-panel gene
        matrix to the model produces a confident-looking but meaningless
        prediction, and that failure mode must not be silent.

        label_mapping (data.label_mapping.EffectiveLabelMapping), if given,
        is validated against model.num_smoke and used to name smoke classes
        in every predict_* output — None falls back to the fixed six-class
        constants.SMOKE_TYPES (no-merge / legacy default).
        """
        self.model  = model.to(device).eval()
        self.device = device
        self.preprocessing_artifact = preprocessing_artifact
        self.unsafe_legacy_mode = unsafe_legacy_mode
        if label_mapping is not None and label_mapping.k != self.model.num_smoke:
            raise ValueError(
                f"Predictor: label_mapping has K={label_mapping.k} effective classes but "
                f"model.num_smoke={self.model.num_smoke} — config/checkpoint conflict. The "
                f"model must be constructed with num_smoke_types={label_mapping.k} (see "
                "MultiSmokeCancerNet.from_config's num_smoke_types override)."
            )
        self.label_mapping = label_mapping

    def _class_names(self) -> List[str]:
        return self.label_mapping.class_names if self.label_mapping else list(SMOKE_TYPES.values())

    @classmethod
    def from_config(
        cls,
        config: Union[dict, str, Path],
        phase:  int = 3,
        device: str = "cpu",
        unsafe_legacy_mode: bool = False,
    ) -> "Predictor":
        """
        Load best checkpoint from a given training phase. If
        checkpoint_dir/preprocessing_artifact.json exists (see
        data/preprocessing.py::PreprocessingArtifact.save), it's loaded too
        so predict_h5ad() can reorder/scale incoming data to match training.

        Reconstructs the checkpoint's effective smoke-label mapping (if any)
        BEFORE constructing the model (via read_checkpoint_metadata, which
        peeks at the checkpoint without loading weights) so the model is
        built with the correct num_smoke_types up front. If a preprocessing
        artifact is ALSO loaded and it carries its own label_mapping, the
        two are cross-validated — a mismatch means the checkpoint and the
        artifact came from different rare-class-policy runs and must not be
        used together.
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("[inference] WARNING: CUDA not available — falling back to CPU")
            device = "cpu"
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        ckpt_dir = Path(config.get("train", config).get("checkpoint_dir", "checkpoints"))
        if not ckpt_dir.is_absolute():
            ckpt_dir = Path(__file__).parents[1] / ckpt_dir
        ckpt = ckpt_dir / f"phase{phase}_best.pt"

        label_mapping = None
        if ckpt.exists():
            ckpt_meta = read_checkpoint_metadata(ckpt, device)
            label_mapping = _label_mapping_from_checkpoint_meta(ckpt_meta)

        model = MultiSmokeCancerNet.from_config(
            config, num_smoke_types=label_mapping.k if label_mapping else None,
        )
        load_checkpoint_into(model, ckpt, device)
        print(f"[inference] loaded phase {phase} checkpoint  ({ckpt})"
              + (f"  effective label space K={label_mapping.k}" if label_mapping else ""))

        artifact = None
        artifact_path = ckpt_dir / "preprocessing_artifact.json"
        if artifact_path.exists():
            from data.preprocessing import PreprocessingArtifact
            artifact = PreprocessingArtifact.load(artifact_path)
            print(f"[inference] loaded preprocessing artifact  ({artifact_path})")
            if artifact.label_mapping and label_mapping is not None:
                EffectiveLabelMapping.from_dict(artifact.label_mapping).validate_compatible(
                    label_mapping, self_name="preprocessing_artifact", other_name="checkpoint",
                )
            elif artifact.label_mapping and label_mapping is None:
                label_mapping = EffectiveLabelMapping.from_dict(artifact.label_mapping)
        elif not unsafe_legacy_mode:
            print("[inference] WARNING: no preprocessing_artifact.json found next to the "
                  "checkpoint — predict_h5ad() will refuse raw/unlabelled input unless "
                  "unsafe_legacy_mode=True is explicitly set.")

        return cls(model, device, preprocessing_artifact=artifact,
                   unsafe_legacy_mode=unsafe_legacy_mode, label_mapping=label_mapping)

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
        cell_type_ids = validate_cell_type_ids(
            cell_type_ids, self.model.num_cell_types, n_expected=len(gene_matrix),
        )
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
            class_names = self._class_names(),
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
        h5ad_path:            Union[str, Path],
        subject_col:          str = "subject_id",
        cell_type_col:        str = "cell_type_id",
        input_stage:          str = "normalized_expression",
        already_preprocessed: Optional[bool] = None,
    ) -> List[Dict]:
        """
        Predict cancer risk directly from an H5AD file.

        input_stage declares what state the input .X is ALREADY in — this
        is checked, never trusted on faith:

          "model_ready" — .X is EXACTLY what the model expects: same genes,
            same order, same scaling as training. Verified via exact
            gene-order equality against self.preprocessing_artifact.gene_list
            (data/preprocessing.py::verify_input_matrix), finite-value check,
            and final width against model.input_dim. NO transformation is
            applied — a silent reorder/rescale never happens for input
            declared model-ready.

          "normalized_expression" (default) — .X is already QC'd,
            library-size normalized, and log-transformed the same way
            training data was (see data/transforms.py::normalize) — this is
            NOT raw counts. Genes are reordered/subset to
            self.preprocessing_artifact.gene_list and ONLY the train-fit
            mean/std scaling actually stored in the artifact is applied
            (data/preprocessing.py::apply_preprocessing). Requires a
            preprocessing_artifact unless unsafe_legacy_mode=True was
            explicitly set. Duplicate gene names and non-finite values in
            the transformed output are rejected.

          "raw_counts" — ALWAYS REJECTED with a clear error. This
            repository's PreprocessingArtifact does not store QC
            thresholds, a library-size normalization target, or
            log-transform parameters, so the full raw-count pipeline used
            at training time cannot be reproduced here. Normalize/log-
            transform the data yourself (data/transforms.py::normalize),
            then call this with input_stage="normalized_expression".

        already_preprocessed (bool) is DEPRECATED — kept only for backward
        compatibility, emits a DeprecationWarning, and maps unambiguously:
        True -> "model_ready", False -> "normalized_expression" (False
        never meant genuine raw-count support; it only ever reordered/
        subset/scaled via the fitted artifact, exactly like
        "normalized_expression" today — so mapping it there is not a
        behaviour change, just an honest name). When both are passed,
        already_preprocessed wins and input_stage is ignored, with a
        warning either way.

        Either mode verifies the final input width against
        self.model.input_dim before any forward pass — a shape mismatch
        would otherwise fail deep inside the model with a confusing error,
        or (worse, if dimensions coincidentally matched some other layer)
        silently produce a meaningless prediction.

        Expects the AnnData to have:
          .obs[subject_col]: subject identifier per cell
          .obs[cell_type_col]: integer cell type (0-3)

        Groups cells by subject_id and runs predict_batch.
        Raises ValueError if required obs columns are missing, if input_stage
        is invalid or "raw_counts", if normalized-expression input has no
        preprocessing_artifact and unsafe_legacy_mode is not set, if genes
        are duplicated/missing, if values are non-finite, or if the final
        gene count doesn't match the model's input_dim.
        """
        if already_preprocessed is not None:
            import warnings
            warnings.warn(
                "predict_h5ad(already_preprocessed=...) is deprecated — use "
                "input_stage='model_ready' (was True) or "
                "input_stage='normalized_expression' (was False) instead. "
                "already_preprocessed=False never meant true raw-count support; it only "
                "ever reordered/subset/scaled via the fitted artifact, same as "
                "input_stage='normalized_expression' today.",
                DeprecationWarning, stacklevel=2,
            )
            input_stage = "model_ready" if already_preprocessed else "normalized_expression"

        if input_stage not in VALID_INPUT_STAGES:
            raise ValueError(
                f"predict_h5ad: input_stage must be one of {sorted(VALID_INPUT_STAGES)}, "
                f"got {input_stage!r}."
            )
        if input_stage == "raw_counts":
            raise ValueError(
                "predict_h5ad(input_stage='raw_counts') is not supported: this "
                "repository's PreprocessingArtifact does not store QC thresholds, a "
                "library-size normalization target, or log-transform parameters, so the "
                "full raw-count preprocessing chain used at training time cannot be "
                "reproduced at inference time. Normalize and log-transform your data the "
                "same way training data was processed (see "
                "data/transforms.py::normalize), then call predict_h5ad(..., "
                "input_stage='normalized_expression')."
            )

        import anndata as ad
        adata = ad.read_h5ad(h5ad_path)
        self._validate_h5ad(adata, subject_col, cell_type_col)

        if input_stage == "model_ready":
            X = self._prepare_model_ready(adata)
        else:
            X = self._prepare_normalized_expression(adata)

        if X.shape[1] != self.model.input_dim:
            raise ValueError(
                f"predict_h5ad: final input has {X.shape[1]} genes but the model expects "
                f"input_dim={self.model.input_dim}. This would otherwise silently feed "
                "mismatched features into the model."
            )

        all_cell_type_ids = validate_cell_type_ids(
            adata.obs[cell_type_col].values, self.model.num_cell_types, n_expected=X.shape[0],
        )

        subjects = []
        for sid in adata.obs[subject_col].unique():
            mask = (adata.obs[subject_col] == sid).values
            subjects.append({
                "subject_id":    str(sid),
                "gene_matrix":   X[mask],
                "cell_type_ids": all_cell_type_ids[mask],
            })

        print(f"[inference] {len(subjects)} subjects from {Path(h5ad_path).name}"
              f"  (input_stage={input_stage!r})")
        return self.predict_batch(subjects)

    def _prepare_model_ready(self, adata) -> np.ndarray:
        """
        input_stage="model_ready": caller asserts .X is EXACTLY the model's
        input. Checked, not trusted: exact gene-order equality against the
        artifact (if any), artifact-version match, finite values. No
        transformation is applied.
        """
        import scipy.sparse as sp
        if self.preprocessing_artifact is not None:
            from data.preprocessing import verify_input_matrix
            verify_input_matrix(self.preprocessing_artifact, list(adata.var_names))
            if self.preprocessing_artifact.version != ARTIFACT_VERSION:
                raise ValueError(
                    f"predict_h5ad(input_stage='model_ready'): preprocessing_artifact "
                    f"version {self.preprocessing_artifact.version!r} does not match this "
                    f"code's expected artifact version {ARTIFACT_VERSION!r} — refusing to "
                    "run inference against a schema this Predictor cannot fully validate."
                )
        elif not self.unsafe_legacy_mode:
            raise ValueError(
                "predict_h5ad(input_stage='model_ready') has no preprocessing_artifact to "
                "verify gene order against — cannot confirm this input actually matches "
                "training. Pass unsafe_legacy_mode=True on the Predictor to bypass this "
                "check (not recommended for scientific results)."
            )
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError(
                "predict_h5ad(input_stage='model_ready'): input contains non-finite "
                "values (NaN/Inf) — refusing to run inference on invalid expression data."
            )
        return X

    def _prepare_normalized_expression(self, adata) -> np.ndarray:
        """
        input_stage="normalized_expression": caller asserts .X is already
        QC'd, library-size normalized, and log-transformed the same way
        training data was (data/transforms.py::normalize) — NOT raw counts.
        Reorders/subsets to the fitted artifact's gene panel and applies
        ONLY the train-fit mean/std scaling actually stored in the artifact.
        """
        import scipy.sparse as sp
        if self.preprocessing_artifact is None:
            if not self.unsafe_legacy_mode:
                raise ValueError(
                    "predict_h5ad(input_stage='normalized_expression') has no "
                    "preprocessing_artifact, so genes cannot be safely reordered/scaled "
                    "to match training. Load a Predictor with an artifact (see "
                    "Predictor.from_config), or set unsafe_legacy_mode=True to bypass "
                    "this at your own risk."
                )
            print("[inference] WARNING: unsafe_legacy_mode — running input through the "
                  "model with NO gene reordering/scaling. Predictions are not "
                  "scientifically valid unless this input independently already matches "
                  "training exactly.")
            X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
            return np.asarray(X, dtype=np.float32)

        if self.preprocessing_artifact.expected_input_stage != "normalized_expression":
            raise ValueError(
                "predict_h5ad(input_stage='normalized_expression'): this Predictor's "
                f"preprocessing_artifact expects input_stage="
                f"{self.preprocessing_artifact.expected_input_stage!r}, not "
                "'normalized_expression' — artifact/input_stage schema mismatch."
            )
        if adata.var_names.duplicated().any():
            dupes = sorted(set(adata.var_names[adata.var_names.duplicated()]))
            raise ValueError(
                "predict_h5ad(input_stage='normalized_expression'): input has "
                f"{len(dupes)} duplicate gene name(s) {dupes[:10]} — cannot "
                "unambiguously reorder/subset to the artifact's gene panel."
            )
        from data.preprocessing import apply_preprocessing
        adata = apply_preprocessing(adata, self.preprocessing_artifact)
        assert list(adata.var_names) == self.preprocessing_artifact.gene_list
        X = np.asarray(adata.X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError(
                "predict_h5ad(input_stage='normalized_expression'): transformed input "
                "contains non-finite values (NaN/Inf) after scaling."
            )
        return X

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
    p.add_argument("--h5ad",   type=str, default=None,          help="H5AD file to predict on")
    p.add_argument("--input-stage", type=str, default="normalized_expression",
                   choices=sorted(VALID_INPUT_STAGES),
                   help="What state --h5ad's .X is already in (default: normalized_expression)")
    p.add_argument("--out",    type=str, default=None,          help="Output JSON path")
    p.add_argument("--device", type=str, default="cpu",         help="cpu or cuda")
    return p


def _cli_main():
    args = _build_cli().parse_args()
    predictor = Predictor.from_config(args.config, phase=args.phase, device=args.device)

    if args.h5ad:
        results = predictor.predict_h5ad(args.h5ad, input_stage=args.input_stage)
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