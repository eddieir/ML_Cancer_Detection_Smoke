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

import dataclasses
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

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
from data.manifest import sha256_of_file
from data.source_sampling import SourceBalancedBatchSampler, SourceSubjectIndex

from .bundle import BundleValidationError, load_and_validate_bundle, write_model_bundle
from .domain_losses import (
    DomainClassifierHead,
    DomainLossConfigurationError,
    coral_loss,
    domain_adversarial_loss,
    group_embeddings_by_source,
    mmd_loss,
    resolve_domain_robustness_config,
    validate_source_provenance,
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
    batch = collate_subject_bags(prepared)
    # Development-source label per subject, in the SAME bag order as every
    # other field above — used only by domain-robustness regularizers
    # (coral/mmd/domain_adversarial), never by the primary task heads. A bag
    # with no recorded source (legacy/synthetic caller) reports "unknown",
    # which domain-robustness code must treat as its own explicit source
    # bucket, never silently merged into another one.
    batch["source"] = [str(b.get("source") or "unknown") for b in bags]
    return batch


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

    def __init__(
        self, device: str = "cpu", config_overrides: Optional[Dict] = None,
        domain_robustness_config: Optional[Dict] = None,
    ):
        self.device = device
        self.config_overrides = dict(config_overrides or {})
        self.config: Optional[PathwayHierarchicalMILConfig] = None
        self.modules: Optional[GeneModuleCollection] = None
        self.model: Optional[PathwayHierarchicalMIL] = None
        self.fit_seconds: Optional[float] = None
        # Development-only domain-robustness strategy — see domain_losses.py.
        # Defaults to plain ERM (every regularizer disabled, weight 0.0) when
        # not supplied, so every existing caller of this adapter (CV, OOF,
        # final dev-pool fit, LOSO baselines) is completely unaffected unless
        # it explicitly opts in.
        self.domain_robustness = resolve_domain_robustness_config(domain_robustness_config)
        self.domain_head: Optional[DomainClassifierHead] = None
        self.domain_source_vocabulary: Optional[List[str]] = None
        self.last_loss_components: Dict[str, float] = {}
        # Per-epoch realized source-exposure diagnostics, populated only when
        # strategy == "source_balanced" (see _build_source_sampler) — None
        # for every other strategy, never a fabricated empty-looking dict.
        self.source_sampling_diagnostics: Optional[List[Dict]] = None

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

    def _maybe_build_domain_head(self, sources: Sequence[str]) -> None:
        """Development-source vocabulary is fixed the FIRST time this
        adapter is trained (never rebuilt per epoch, never extended with a
        source seen only later) — see DomainClassifierHead. Only built at
        all when the adversarial strategy is actually enabled, so ERM /
        coral / mmd / source_balanced runs never pay for or persist an
        unused domain head."""
        if self.domain_robustness["strategy"] != "domain_adversarial":
            return
        if not self.domain_robustness["adversarial"]["enabled"]:
            raise DomainLossConfigurationError(
                "domain_robustness.strategy='domain_adversarial' requires "
                "domain_robustness.adversarial.enabled=true (with a weight > 0) — a strategy name "
                "alone does not implicitly enable its regularizer."
            )
        if self.domain_head is None:
            validate_source_provenance(sources)
            unique_sources = sorted(set(str(s) for s in sources))
            self.domain_source_vocabulary = unique_sources
            self.domain_head = DomainClassifierHead(self.config.embedding_dim, unique_sources).to(self.device)

    def _domain_regularizer(
        self, subject_embeddings: torch.Tensor, sources: Sequence[str], epoch: int,
    ) -> "torch.Tensor":
        """Adds whichever development-only regularizer this adapter's
        domain_robustness config selects to the primary task loss. Returns a
        zero (but differentiable, when applicable) tensor for 'erm' and
        'source_balanced' — the latter changes SAMPLING, not the loss
        function, so it has no additional loss term here."""
        strategy = self.domain_robustness["strategy"]
        device = subject_embeddings.device
        zero = torch.zeros((), device=device)
        if strategy in ("erm", "source_balanced"):
            return zero

        if strategy == "coral":
            cfg = self.domain_robustness["coral"]
            if not cfg["enabled"] or cfg["weight"] == 0.0:
                return zero
            validate_source_provenance(sources)
            grouped = group_embeddings_by_source(subject_embeddings, sources)
            loss, meta = coral_loss(grouped)
            self.last_loss_components["coral_loss"] = float(loss.detach().item())
            self.last_loss_components.update({f"coral_{k}": v for k, v in meta.items()})
            return cfg["weight"] * loss

        if strategy == "mmd":
            cfg = self.domain_robustness["mmd"]
            if not cfg["enabled"] or cfg["weight"] == 0.0:
                return zero
            validate_source_provenance(sources)
            grouped = group_embeddings_by_source(subject_embeddings, sources)
            loss, meta = mmd_loss(grouped, kernel=cfg["kernel"])
            self.last_loss_components["mmd_loss"] = float(loss.detach().item())
            self.last_loss_components.update({f"mmd_{k}": v for k, v in meta.items()})
            return cfg["weight"] * loss

        if strategy == "domain_adversarial":
            cfg = self.domain_robustness["adversarial"]
            if not cfg["enabled"] or cfg["weight"] == 0.0 or self.domain_head is None:
                return zero
            validate_source_provenance(sources)
            if epoch < cfg["warmup_epochs"]:
                # Deterministic warm-up: the encoder is never adversarially
                # regularized before warmup_epochs has elapsed, so the
                # primary task heads get a stable representation to start
                # from — same schedule every run for a fixed config, not a
                # random or data-dependent one.
                return zero
            loss, meta = domain_adversarial_loss(
                self.domain_head, subject_embeddings, sources, lambda_=cfg["gradient_reversal_lambda"],
            )
            self.last_loss_components["domain_adversarial_loss"] = float(loss.detach().item())
            self.last_loss_components["domain_accuracy"] = meta["domain_accuracy"]
            return cfg["weight"] * loss

        raise DomainLossConfigurationError(f"Unhandled domain_robustness.strategy={strategy!r}")

    def _build_source_sampler(self, bags: Sequence[dict], seed: int) -> Optional[SourceBalancedBatchSampler]:
        """
        Only constructed for strategy == 'source_balanced' (already validated
        enabled=true by resolve_domain_robustness_config). Each bag IS one
        subject (this adapter's unit of gradient-update input), so
        SourceSubjectIndex is built with exactly one "cell" per subject —
        source -> subject sampling with no further cell-level draw, which is
        the correct degenerate case of the same source -> subject -> cell
        sampler used elsewhere for cell-level training (data/source_sampling.py).
        """
        if self.domain_robustness["strategy"] != "source_balanced":
            return None
        subject_ids = np.array([str(b["subject_id"]) for b in bags])
        sources = np.array([str(b.get("source") or "unknown") for b in bags])
        validate_source_provenance(sources)
        index = SourceSubjectIndex(subject_ids=subject_ids, sources=sources)
        cfg = self.domain_robustness["source_balancing"]
        batch_size = cfg.get("batch_size") or min(len(bags), 8)
        samples_per_epoch = cfg.get("samples_per_epoch") or len(bags)
        return SourceBalancedBatchSampler(
            index, batch_size=batch_size, seed=seed, samples_per_epoch=samples_per_epoch,
        )

    def _train_loop(self, bags: Sequence[dict], epochs: int, seed: int) -> None:
        torch.manual_seed(seed)
        full_batch = bags_to_pathway_batch(bags)
        sources_all = full_batch["source"]
        self._maybe_build_domain_head(sources_all)
        loss_fn = MultitaskMaskedLoss(
            smoke_loss_weight=self.config.smoke_loss_weight,
            cancer_loss_weight=self.config.cancer_loss_weight,
            empty_batch_policy=self.config.empty_batch_policy,
        )
        params = list(self.model.parameters())
        if self.domain_head is not None:
            params += list(self.domain_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=DEFAULT_LR)
        self.model.train()

        def _step(sub_bags: Sequence[dict], epoch: int) -> None:
            batch = bags_to_pathway_batch(sub_bags)
            sources = batch["source"]
            expression = batch["expression"].to(self.device)
            cell_type_ids = batch["cell_type_ids"].to(self.device)
            cell_mask = batch["cell_mask"].to(self.device)
            optimizer.zero_grad()
            out = self.model(expression, cell_type_ids, cell_mask)
            task_loss, _ = loss_fn(
                out.smoke_logits, batch["smoke_label"].to(self.device), batch["smoke_known"].to(self.device),
                out.cancer_logits, batch["cancer_label"].to(self.device), batch["cancer_known"].to(self.device),
            )
            domain_term = self._domain_regularizer(out.subject_embeddings, sources, epoch)
            total = task_loss + domain_term
            self.last_loss_components["task_loss"] = float(task_loss.detach().item())
            self.last_loss_components["total_loss"] = float(total.detach().item())
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, DEFAULT_GRAD_CLIP)
            optimizer.step()

        sampler = self._build_source_sampler(bags, seed)
        if sampler is None:
            # Unchanged ERM/coral/mmd/domain_adversarial behavior: one
            # full-batch gradient step per epoch over every supplied bag.
            for epoch in range(max(epochs, 1)):
                _step(bags, epoch)
        else:
            per_epoch_diagnostics = []
            for epoch in range(max(epochs, 1)):
                for index_batch in sampler:
                    _step([bags[i] for i in index_batch], epoch)
                if sampler.last_realized_diagnostics is not None:
                    per_epoch_diagnostics.append(sampler.last_realized_diagnostics.to_dict())
            self.source_sampling_diagnostics = per_epoch_diagnostics

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

    def predict_smoke_proba(self, subject_dataset) -> np.ndarray:
        """Softmax class-probability matrix ([n_subjects, num_smoke]) per
        subject, ordered like subject_dataset.bags — the multi-class
        analogue of predict_proba, used only for uncertainty/abstention
        diagnostics, never for the primary macro-F1 metric."""
        self.model.eval()
        batch = bags_to_pathway_batch(subject_dataset.bags)
        with torch.no_grad():
            out = self.model(
                batch["expression"].to(self.device), batch["cell_type_ids"].to(self.device),
                batch["cell_mask"].to(self.device),
            )
        return torch.softmax(out.smoke_logits, dim=1).cpu().numpy()

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
            "domain_robustness": self.domain_robustness,
            "domain_source_vocabulary": self.domain_source_vocabulary,
            "last_loss_components": dict(self.last_loss_components),
            "source_sampling_diagnostics": self.source_sampling_diagnostics,
        }

    def model_state_fingerprint(self) -> str:
        """Deterministic SHA-256 of the fitted network weights — same
        helper benchmarks/neural.py's adapters use, so this model's OOF/
        final-fit records carry a fingerprint in exactly the same shape."""
        from .model_fingerprint import torch_state_dict_fingerprint
        return torch_state_dict_fingerprint(self.model)

    def save_bundle(
        self, bundle_dir: Union[str, Path], artifact,
        dataset_manifest_fingerprint: Optional[str] = None, split_fingerprint: Optional[str] = None,
    ) -> Path:
        """
        Persist this fitted adapter as a Phase-4-style bundle (benchmarks/
        bundle.py::write_model_bundle, reused unchanged) — model weights,
        gene modules, and this adapter's FULL domain-robustness state
        (config, fixed development-source vocabulary, and the domain-
        adversarial head's own weights when one was built) all folded into
        one bundle_manifest.json via `extra`, so a domain_adversarial run's
        state round-trips completely through load_bundle, not just the
        primary task head.
        """
        if self.model is None:
            raise ValueError("PathwayHierarchicalAdapter.save_bundle: adapter has not been fit yet.")
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = bundle_dir / "model_checkpoint.pt"
        torch.save(self.model.state_dict(), checkpoint_path)

        gene_modules_path = bundle_dir / "gene_modules.json"
        self.modules.save(gene_modules_path)

        domain_head_checkpoint = None
        if self.domain_head is not None:
            domain_head_path = bundle_dir / "domain_head_checkpoint.pt"
            torch.save(self.domain_head.state_dict(), domain_head_path)
            domain_head_checkpoint = {"path": domain_head_path.name, "sha256": sha256_of_file(domain_head_path)}

        extra = {
            "model_type": MODEL_NAME,
            "module_fingerprint": self.modules.fingerprint(),
            "gene_modules_path": gene_modules_path.name,
            "num_smoke": int(self.model.num_smoke),
            "domain_robustness": self.domain_robustness,
            "domain_source_vocabulary": self.domain_source_vocabulary,
            "domain_head_checkpoint": domain_head_checkpoint,
            "model_state_fingerprint": self.model_state_fingerprint(),
        }
        return write_model_bundle(
            bundle_dir, checkpoint_path, artifact, dataclasses.asdict(self.config),
            class_vocabulary=[str(i) for i in range(int(self.model.num_smoke))],
            label_policy="phase6_source_held_out", species_policy="unspecified", assay_mode="single_cell",
            dataset_manifest_fingerprint=dataset_manifest_fingerprint, split_fingerprint=split_fingerprint,
            extra=extra,
        )

    @classmethod
    def load_bundle(cls, bundle_dir: Union[str, Path], device: str = "cpu") -> "PathwayHierarchicalAdapter":
        """
        Inverse of save_bundle — re-derives every hash load_and_validate_
        bundle already checks (checkpoint/artifact integrity, bundle_
        fingerprint), additionally verifies the gene-module fingerprint and
        the reloaded model's own model_state_fingerprint against what was
        recorded at save time (never proceeds with a silently-altered
        checkpoint), and reconstructs the domain-adversarial head (with its
        OWN weights and the exact fixed source vocabulary) when the bundle
        recorded one. Raises BundleValidationError for any missing/
        corrupted/altered component.
        """
        bundle_dir = Path(bundle_dir)
        manifest = load_and_validate_bundle(bundle_dir)

        gene_modules_path = bundle_dir / manifest["gene_modules_path"]
        modules = GeneModuleCollection.load(gene_modules_path)
        if modules.fingerprint() != manifest["module_fingerprint"]:
            raise BundleValidationError(
                f"PathwayHierarchicalAdapter.load_bundle: gene-module fingerprint at {gene_modules_path} "
                "does not match the bundle manifest's recorded module_fingerprint."
            )

        domain_robustness_config = manifest["domain_robustness"]
        adapter = cls(device=device, domain_robustness_config=domain_robustness_config)
        adapter.modules = modules
        adapter.config = PathwayHierarchicalMILConfig.from_dict(manifest["model_config"])
        adapter.model = PathwayHierarchicalMIL(modules, adapter.config, num_smoke=manifest["num_smoke"]).to(device)

        checkpoint_path = bundle_dir / manifest["model_checkpoint"]["path"]
        state_dict = torch.load(checkpoint_path, map_location=device)
        try:
            adapter.model.load_state_dict(state_dict)
        except RuntimeError as e:
            raise BundleValidationError(
                f"PathwayHierarchicalAdapter.load_bundle: model checkpoint at {checkpoint_path} does "
                f"not match the reconstructed architecture — {e}"
            ) from e

        domain_head_checkpoint = manifest.get("domain_head_checkpoint")
        vocabulary = manifest.get("domain_source_vocabulary")
        if domain_head_checkpoint:
            if not vocabulary:
                raise BundleValidationError(
                    "PathwayHierarchicalAdapter.load_bundle: bundle recorded a domain_head_checkpoint "
                    "but no domain_source_vocabulary — a domain head cannot be reconstructed without "
                    "its fixed vocabulary."
                )
            domain_head_path = bundle_dir / domain_head_checkpoint["path"]
            if sha256_of_file(domain_head_path) != domain_head_checkpoint["sha256"]:
                raise BundleValidationError(
                    f"PathwayHierarchicalAdapter.load_bundle: domain-head checkpoint at "
                    f"{domain_head_path} does not match the bundle manifest's recorded checksum."
                )
            adapter.domain_head = DomainClassifierHead(adapter.config.embedding_dim, vocabulary).to(device)
            domain_head_state = torch.load(domain_head_path, map_location=device)
            try:
                adapter.domain_head.load_state_dict(domain_head_state)
            except RuntimeError as e:
                raise BundleValidationError(
                    f"PathwayHierarchicalAdapter.load_bundle: domain-head checkpoint does not match "
                    f"the reconstructed domain-head architecture (vocabulary size mismatch?) — {e}"
                ) from e
            adapter.domain_source_vocabulary = list(vocabulary)
        elif domain_robustness_config.get("strategy") == "domain_adversarial":
            raise BundleValidationError(
                "PathwayHierarchicalAdapter.load_bundle: domain_robustness.strategy='domain_adversarial' "
                "but the bundle has no domain_head_checkpoint recorded — a domain-adversarial bundle "
                "must never load without its domain head."
            )

        if adapter.model_state_fingerprint() != manifest["model_state_fingerprint"]:
            raise BundleValidationError(
                "PathwayHierarchicalAdapter.load_bundle: reloaded model_state_fingerprint does not "
                "match the bundle manifest's recorded value — the checkpoint or manifest was altered "
                "since save_bundle wrote it."
            )
        return adapter


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
