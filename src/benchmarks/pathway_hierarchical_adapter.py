"""
benchmarks/pathway_hierarchical_adapter.py — CV/search/final-fit adapter for
pathway_hierarchical_mil.PathwayHierarchicalMIL, giving it the same
fit()/fit_final()/predict_proba()/metadata()/model_state_fingerprint()
surface benchmarks/neural.py::NeuralCancerAdapter already exposes to
cross_validation.py and final_evaluation.py — see benchmarks/mil_registry.py
for the one place that selects between the two adapter classes by model
name. This keeps run_smoke_cv/run_cancer_cv, the nested hyperparameter
search, and the final-development-fit protocol working through their
existing dispatch rather than growing a second, parallel one.

Unlike NeuralCancerAdapter (which relies on Trainer's three-phase
curriculum), this adapter trains PathwayHierarchicalMIL directly with a
single AdamW loop over MultitaskMaskedLoss — the architecture has no
separate cell-level pretraining phase, so there is nothing for a Phase 1
step to do here.
"""

import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from pathway_hierarchical_mil import (
    GeneModuleCollection,
    GeneModuleError,
    MultitaskMaskedLoss,
    PathwayHierarchicalMIL,
    PathwayHierarchicalMILConfig,
    collate_subject_bags,
)

MODEL_NAME = "pathway_hierarchical_mil"
DEFAULT_EPOCHS = 5
DEFAULT_LR = 5e-3
DEFAULT_GRAD_CLIP = 1.0


class PathwayModuleConfigurationError(ValueError):
    """Raised when this model is selected without a usable gene-module
    source — see build_gene_modules_for_context. Distinct from
    GeneModuleError so callers can tell "no module source configured"
    (a run/config problem) apart from "the module file itself is malformed"
    (a data problem), while both remain ValueError subclasses."""


def build_gene_modules_for_context(context, gene_list: Sequence[str]) -> GeneModuleCollection:
    """
    Build (or load) the GeneModuleCollection this model needs, aligned to
    `gene_list` — always a specific fold's or the final development
    artifact's own ordered gene list, never a cached/outer one, so module
    alignment always matches the exact preprocessing identity the model is
    about to train against.

    Reads context.config["model"]["pathway_hierarchical_mil"]["gene_modules"]:
      - path set -> GeneModuleCollection.from_gmt(path, gene_list, ...)
      - path unset and allow_synthetic_modules=True -> a deterministic,
        clearly-labelled synthetic scheme (test/synthetic workflows only)
      - path unset and allow_synthetic_modules=False (the real-data default)
        -> PathwayModuleConfigurationError, an actionable configuration
        error rather than a silent fallback
    """
    phm_cfg = (context.config.get("model", {}) or {}).get("pathway_hierarchical_mil", {}) or {}
    gm_cfg = phm_cfg.get("gene_modules", {}) or {}
    path = gm_cfg.get("path")
    min_genes = int(gm_cfg.get("minimum_genes_per_module", 3))
    empty_policy = gm_cfg.get("empty_module_policy", "error")
    source_version = gm_cfg.get("source_version", "unspecified")

    if path:
        return GeneModuleCollection.from_gmt(
            path, gene_list, source_version=source_version,
            minimum_genes_per_module=min_genes, empty_module_policy=empty_policy,
        )
    if gm_cfg.get("allow_synthetic_modules", False):
        seed = int(gm_cfg.get("synthetic_seed", 0))
        n_modules = int(gm_cfg.get("synthetic_n_modules", 6))
        genes_per_module = int(gm_cfg.get("synthetic_genes_per_module", 8))
        return GeneModuleCollection.synthetic(
            gene_list, n_modules=n_modules, genes_per_module=genes_per_module, seed=seed,
        )
    raise PathwayModuleConfigurationError(
        "build_gene_modules_for_context: model.pathway_hierarchical_mil.gene_modules.path is not "
        "set and allow_synthetic_modules is not enabled — this model cannot construct without an "
        "explicit gene-module source. Supply a GMT module file path for a real run, or set "
        "gene_modules.allow_synthetic_modules: true for a synthetic/test-only workflow."
    )


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def bags_to_pathway_batch(bags: Sequence[dict]) -> Dict[str, torch.Tensor]:
    """
    Adapt this repository's standard MIL bag-dict shape (subject_id,
    gene_matrix [n,G], cell_type_ids [n], smoke_labels [n] per-cell,
    smoke_known [n] per-cell — see fold_preprocessing.py::
    bags_from_fold_cell_dataset — cancer_label, cancer_label_known) into
    pathway_hierarchical_mil.collate_subject_bags's padded-batch format.

    Subject-level smoke label is the majority vote among cells with a
    KNOWN smoke label; subject-level smoke_known is True only if at least
    one cell carries a known label — a subject with zero known-smoke cells
    contributes no smoke supervision, mirroring the cell-level path's own
    per-cell masking (never a fabricated subject label from unknown cells).
    """
    prepared = []
    for b in bags:
        cell_type_ids = np.asarray(b["cell_type_ids"])
        smoke_labels = np.asarray(b["smoke_labels"])
        smoke_known_cell = np.asarray(b.get("smoke_known", np.ones(len(smoke_labels), dtype=bool)))
        if smoke_known_cell.any():
            known_labels = smoke_labels[smoke_known_cell]
            vals, counts = np.unique(known_labels, return_counts=True)
            subject_smoke_label = int(vals[np.argmax(counts)])
            subject_smoke_known = True
        else:
            subject_smoke_label = 0
            subject_smoke_known = False
        cancer_label = b.get("cancer_label")
        cancer_known = bool(b.get("cancer_label_known", cancer_label is not None))
        prepared.append(dict(
            expression=torch.as_tensor(np.asarray(b["gene_matrix"]), dtype=torch.float32),
            cell_type_ids=torch.as_tensor(cell_type_ids, dtype=torch.long),
            smoke_label=torch.tensor(subject_smoke_label, dtype=torch.long),
            smoke_known=torch.tensor(subject_smoke_known, dtype=torch.bool),
            cancer_label=torch.tensor(float(cancer_label) if cancer_known else 0.0, dtype=torch.float32),
            cancer_known=torch.tensor(cancer_known, dtype=torch.bool),
        ))
    return collate_subject_bags(prepared)


class PathwayHierarchicalAdapter:
    """
    Task A/B adapter — see module docstring. `fit`/`fit_final` intentionally
    mirror NeuralCancerAdapter's signatures (including the unused
    `cell_dataset` positional arguments and the `pretrain_epochs` keyword,
    which this adapter treats as its single training loop's epoch count)
    so benchmarks/mil_registry.py's callers never need to branch on which
    adapter class they are holding.
    """

    name = MODEL_NAME

    def __init__(self, device: str = "cpu", config_overrides: Optional[Dict] = None):
        self.device = device
        self.config_overrides = dict(config_overrides or {})
        self.config: Optional[PathwayHierarchicalMILConfig] = None
        self.modules: Optional[GeneModuleCollection] = None
        self.model: Optional[PathwayHierarchicalMIL] = None
        self.fit_seconds: Optional[float] = None

    def _build(self, context, num_smoke: int) -> None:
        gene_list = list(context.preprocessing_artifact.gene_list)
        self.modules = build_gene_modules_for_context(context, gene_list)
        base = dict((context.config.get("model", {}) or {}).get("pathway_hierarchical_mil", {}) or {})
        base.pop("gene_modules", None)
        base.pop("uncertainty", None)
        base.pop("enabled", None)
        base.update(self.config_overrides)
        base.setdefault("num_cell_type_buckets", context.config.get("model", {}).get("num_cell_types", 4) + 1)
        self.config = PathwayHierarchicalMILConfig.from_dict(base)
        self.model = PathwayHierarchicalMIL(self.modules, self.config, num_smoke=num_smoke).to(self.device)

    def _train_loop(self, bags: Sequence[dict], epochs: int, seed: int) -> None:
        torch.manual_seed(seed)
        batch = bags_to_pathway_batch(bags)
        loss_fn = MultitaskMaskedLoss(
            smoke_loss_weight=self.config.smoke_loss_weight,
            cancer_loss_weight=self.config.cancer_loss_weight,
            empty_batch_policy=self.config.empty_batch_policy,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=DEFAULT_LR)
        self.model.train()
        expression = batch["expression"].to(self.device)
        cell_type_ids = batch["cell_type_ids"].to(self.device)
        cell_mask = batch["cell_mask"].to(self.device)
        for _ in range(max(epochs, 1)):
            optimizer.zero_grad()
            out = self.model(expression, cell_type_ids, cell_mask)
            total, _ = loss_fn(
                out.smoke_logits, batch["smoke_label"].to(self.device), batch["smoke_known"].to(self.device),
                out.cancer_logits, batch["cancer_label"].to(self.device), batch["cancer_known"].to(self.device),
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), DEFAULT_GRAD_CLIP)
            optimizer.step()

    def fit(
        self, context, train_cell_dataset, val_cell_dataset,
        train_subject_dataset, val_subject_dataset,
        seed: int = 42, pretrain_epochs: Optional[int] = None,
    ) -> "PathwayHierarchicalAdapter":
        t0 = time.time()
        self._build(context, num_smoke=context.num_smoke_classes)
        epochs = pretrain_epochs if pretrain_epochs is not None else DEFAULT_EPOCHS
        self._train_loop(train_subject_dataset.bags, epochs=epochs, seed=seed)
        self.fit_seconds = time.time() - t0
        return self

    def fit_final(
        self, context, dev_cell_dataset, dev_subject_dataset,
        seed: int = 42, pretrain_epochs: Optional[int] = None, phase2_epochs: Optional[int] = None,
    ) -> "PathwayHierarchicalAdapter":
        """The one final development-pool fit for the frozen-test protocol:
        every subject in dev_subject_dataset.bags contributes to gradient
        updates, with no internal validation carve-out — mirrors
        NeuralCancerAdapter.fit_final's contract exactly."""
        t0 = time.time()
        self._build(context, num_smoke=context.num_smoke_classes)
        epochs = pretrain_epochs if pretrain_epochs is not None else DEFAULT_EPOCHS
        self._train_loop(dev_subject_dataset.bags, epochs=epochs, seed=seed)
        self.fit_seconds = time.time() - t0
        return self

    def predict_proba(self, subject_dataset) -> np.ndarray:
        """Cancer-risk probability per subject, ordered like
        subject_dataset.bags — same contract as NeuralCancerAdapter."""
        self.model.eval()
        batch = bags_to_pathway_batch(subject_dataset.bags)
        with torch.no_grad():
            out = self.model(
                batch["expression"].to(self.device), batch["cell_type_ids"].to(self.device),
                batch["cell_mask"].to(self.device),
            )
        return torch.sigmoid(out.cancer_logits).cpu().numpy()

    def predict(self, subject_dataset, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(subject_dataset) >= threshold).astype(int)

    def predict_smoke(self, subject_dataset) -> np.ndarray:
        """Predicted smoke-type class per subject, ordered like
        subject_dataset.bags."""
        self.model.eval()
        batch = bags_to_pathway_batch(subject_dataset.bags)
        with torch.no_grad():
            out = self.model(
                batch["expression"].to(self.device), batch["cell_type_ids"].to(self.device),
                batch["cell_mask"].to(self.device),
            )
        return out.smoke_logits.argmax(dim=1).cpu().numpy()

    def known_smoke_labels(self, subject_dataset) -> np.ndarray:
        """Subject-level majority smoke label + known mask, in the SAME
        bag order predict_smoke uses — lets a caller restrict smoke metric
        computation to subjects with verified supervision, exactly like
        every other masked-label consumer in this codebase."""
        batch = bags_to_pathway_batch(subject_dataset.bags)
        return batch["smoke_label"].numpy(), batch["smoke_known"].numpy()

    def metadata(self) -> Dict:
        return {
            "name": self.name,
            "n_parameters": count_parameters(self.model) if self.model else None,
            "fit_seconds": self.fit_seconds,
            "module_fingerprint": self.modules.fingerprint() if self.modules else None,
            "module_source_name": self.modules.source_name if self.modules else None,
            "module_coverage": self.modules.coverage_report() if self.modules else None,
            "config": dict(self.config.__dict__) if self.config else dict(self.config_overrides),
        }

    def model_state_fingerprint(self) -> str:
        """Deterministic SHA-256 of the fitted network weights — same
        helper benchmarks/neural.py's adapters use, so this model's OOF/
        final-fit records carry a fingerprint in exactly the same shape."""
        from .model_fingerprint import torch_state_dict_fingerprint
        return torch_state_dict_fingerprint(self.model)


def validate_pathway_bundle_identity(manifest: Dict, model: "PathwayHierarchicalMIL") -> None:
    """
    Cross-check an already-loaded bundle manifest (benchmarks/bundle.py::
    load_and_validate_bundle) against an actual constructed
    PathwayHierarchicalMIL instance — mirrors bundle.py::
    validate_bundle_for_model's generic input_dim/num_smoke check, but for
    this model's additional gene-module identity: a manifest's recorded
    "model_type" and "module_fingerprint" (written via write_model_bundle's
    `extra` argument) must match this model's own model_type/
    module_fingerprint attributes exactly, or the bundle was built for a
    different architecture or a different gene-module set than the one
    about to load it. Raises BundleValidationError on any mismatch — never
    silently proceeds with a checkpoint trained against a different module
    set than the caller thinks it has.
    """
    from .bundle import BundleValidationError

    recorded_type = manifest.get("model_type")
    if recorded_type is not None and recorded_type != MODEL_NAME:
        raise BundleValidationError(
            f"validate_pathway_bundle_identity: bundle model_type={recorded_type!r} does not "
            f"match {MODEL_NAME!r} — this bundle was not built for this architecture, and "
            "loading its checkpoint into a PathwayHierarchicalMIL instance would silently "
            "misinterpret its weights."
        )
    recorded_fp = manifest.get("module_fingerprint")
    model_fp = getattr(model, "module_fingerprint", None)
    if recorded_fp is not None and model_fp is not None and recorded_fp != model_fp:
        raise BundleValidationError(
            f"validate_pathway_bundle_identity: bundle module_fingerprint="
            f"{str(recorded_fp)[:16]}... does not match this model's module_fingerprint="
            f"{str(model_fp)[:16]}... — the gene-module set was altered (a different GMT file, "
            "or an updated version of the same file) since this bundle was built, or a "
            "different fold's/run's module artifact was substituted in."
        )
