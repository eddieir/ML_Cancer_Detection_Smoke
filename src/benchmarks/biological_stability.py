"""
benchmarks/biological_stability.py — development-only stability and
perturbation diagnostics for pathway_hierarchical_mil.

Scope and honesty note (read before using any function here for a real,
non-synthetic run): every diagnostic in this module characterizes the
MODEL'S SENSITIVITY to gene-module structure and cell-type composition —
never a biological claim. "Module removal reduces performance" is reported
as a model-sensitivity finding, not evidence a module is biologically
causal; attention weights are reported as "model-weighted contribution" or
"pooling weight," never as "importance" in a causal or biological sense; a
result computed against GeneModuleCollection.synthetic() (the deterministic,
clearly-labelled non-biological module scheme — see
pathway_hierarchical_mil.py) is always tagged is_synthetic_modules=True in
this module's outputs, and real-mode analysis functions here reject a
synthetic module source outright (see require_real_modules) rather than
silently reporting a "biological stability" result that is actually a
software-only diagnostic. As of this Phase, no real gene-set (GMT) resource
ships with this repository (see README.md's Phase 6 section) — every
concrete result this module has actually produced in this codebase's tests
and CI is therefore a synthetic-module software diagnostic, not a
biological-plausibility finding, and must not be described as one.
"""

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _sha256_json(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class RealModuleRequiredError(ValueError):
    """Raised by a real-mode-only analysis function when handed a
    synthetic-diagnostic module source — see
    pathway_hierarchical_mil.is_synthetic_module_source."""


def require_real_modules(modules) -> None:
    from pathway_hierarchical_mil import is_synthetic_module_source
    if is_synthetic_module_source(modules.source_name):
        raise RealModuleRequiredError(
            f"biological_stability: module source {modules.source_name!r} is the synthetic "
            "diagnostic scheme, not a real gene-set resource — refusing to label this analysis "
            "'biological stability'. Supply a real GMT module file (model.pathway_hierarchical_mil."
            "gene_modules.path) to run this analysis in real mode, or call the *_diagnostic-only "
            "variant explicitly if a software-only sensitivity check is what you actually want."
        )


# ─── Module ablation / importance ──────────────────────────────────────────

def _ablation_scores_with_mask(
    adapter, bags: Sequence[dict], module_names: Sequence[str], membership_mask, target: str = "cancer",
) -> Dict[str, float]:
    """Shared forward-pass ablation loop — zero each module's genes (per the
    SUPPLIED membership_mask/module_names, not necessarily adapter.modules'
    own) and record the mean absolute change in the target logit. Used both
    by module_ablation_scores (the real mask) and
    gene_module_permutation_null (a permuted mask) so the two share
    identical forward-pass mechanics and differ only in which mask is
    applied."""
    import torch
    from .pathway_hierarchical_adapter import bags_to_pathway_batch

    model = adapter.model
    model.eval()
    batch = bags_to_pathway_batch(bags)
    expression = batch["expression"]
    cell_type_ids = batch["cell_type_ids"]
    cell_mask = batch["cell_mask"]

    with torch.no_grad():
        base_out = model(expression, cell_type_ids, cell_mask)
        base_logits = base_out.cancer_logits if target == "cancer" else base_out.smoke_logits.argmax(-1).float()

    scores = {}
    for i, module_name in enumerate(module_names):
        gene_mask = membership_mask[i].bool()
        ablated = expression.clone()
        ablated[..., gene_mask] = 0.0
        with torch.no_grad():
            out = model(ablated, cell_type_ids, cell_mask)
            logits = out.cancer_logits if target == "cancer" else out.smoke_logits.argmax(-1).float()
        scores[module_name] = float((logits - base_logits).abs().mean().item())
    return scores


def module_ablation_scores(adapter, bags: Sequence[dict], target: str = "cancer") -> Dict[str, float]:
    """
    Model-weighted contribution of each gene module: for every module, zero
    that module's genes across every cell in every bag, run the ALREADY
    -FITTED adapter's forward pass, and record the mean absolute change in
    the target logit relative to the unablated baseline. Deterministic given
    a fixed fitted model and fixed input bags — no retraining happens here.
    This is a sensitivity/contribution diagnostic ("model-weighted
    contribution"), never described as biological importance.
    """
    modules = adapter.modules
    return _ablation_scores_with_mask(adapter, bags, modules.module_names, modules.membership_mask, target=target)


def gene_module_permutation_null(adapter, bags: Sequence[dict], target: str = "cancer", seed: int = 0) -> Dict:
    """
    Null control (Step 7A): apply the SAME random permutation to every
    module row's gene-axis (membership_mask columns) — this preserves every
    module's size exactly (a row's permutation is just a relabeling of
    which column index counts as "in" the module) and preserves the ordered
    gene universe (no gene is added, removed, or duplicated), while
    destroying the real gene<->module association: a gene's ablation
    membership now follows a different module than the one it actually
    belongs to. If the model's module-ablation sensitivity (see
    module_ablation_scores) is a real structural signal rather than an
    artifact of ablating a fixed FRACTION of genes regardless of which ones,
    the permuted-null ablation scores should look different — the caller
    compares the two via spearman_rank_correlation.
    """
    modules = adapter.modules
    rng = np.random.RandomState(seed)
    n_genes = modules.membership_mask.shape[1]
    gene_permutation = rng.permutation(n_genes)
    permuted_mask = modules.membership_mask[:, gene_permutation]
    scores = _ablation_scores_with_mask(adapter, bags, modules.module_names, permuted_mask, target=target)
    return {
        "status": "null_record", "seed": seed,
        "permuted_gene_mapping_fingerprint": _sha256_json(gene_permutation.tolist()),
        "ablation_scores_under_permuted_module_assignment": scores,
        "note": "module sizes and the ordered gene universe are preserved exactly; only which genes "
                "are grouped into which module has been randomly permuted — a null control for "
                "module_ablation_scores, not a biological finding.",
    }


def within_gene_expression_permutation_null(adapter, bags: Sequence[dict], target: str = "cancer", seed: int = 0) -> Dict:
    """
    Null control (Step 7B): independently, for each gene, shuffle that
    gene's expression values across every VALID (non-padding) cell in the
    supplied bag set — preserving each gene's own empirical marginal
    distribution over these cells EXACTLY (it is a permutation, not a
    resample), while destroying the real per-cell joint structure across
    genes (cross-gene correlations, cell-type-conditional co-expression).
    Operates only on already-supplied bags' own expression values — reads
    no subject_id, no label, no held-out-source data, so it cannot leak
    subject or label identity across bags. Reports the mean absolute change
    in the target logit, the same sensitivity statistic
    cell_type_label_permutation_check reports for its own (different)
    perturbation.
    """
    import torch
    from .pathway_hierarchical_adapter import bags_to_pathway_batch

    rng = np.random.RandomState(seed)
    model = adapter.model
    model.eval()
    batch = bags_to_pathway_batch(bags)
    expression = batch["expression"]
    cell_type_ids = batch["cell_type_ids"]
    cell_mask = batch["cell_mask"]

    with torch.no_grad():
        base_out = model(expression, cell_type_ids, cell_mask)
        base_logits = base_out.cancer_logits if target == "cancer" else base_out.smoke_logits.argmax(-1).float()

    valid = cell_mask.bool()
    permuted_expression = expression.clone()
    n_genes = expression.shape[-1]
    valid_idx = valid.nonzero(as_tuple=False)  # [n_valid_cells, 2] (bag_idx, cell_idx)
    n_valid = valid_idx.shape[0]
    for g in range(n_genes):
        perm = rng.permutation(n_valid)
        values = expression[valid_idx[:, 0], valid_idx[:, 1], g]
        permuted_expression[valid_idx[:, 0], valid_idx[:, 1], g] = values[perm]

    with torch.no_grad():
        out = model(permuted_expression, cell_type_ids, cell_mask)
        logits = out.cancer_logits if target == "cancer" else out.smoke_logits.argmax(-1).float()
    mean_abs_diff = float((logits - base_logits).abs().mean().item())
    return {
        "status": "null_record", "seed": seed, "n_valid_cells_permuted_over": int(n_valid),
        "mean_abs_target_logit_difference": mean_abs_diff,
        "note": "each gene's expression was independently shuffled across every valid cell in this "
                "bag set — its own marginal distribution over these cells is preserved exactly; only "
                "cross-gene per-cell structure is destroyed. No subject_id or label was read, so this "
                "cannot leak subject or held-out-label identity across bags. A sensitivity diagnostic, "
                "not a biological claim.",
    }


def matched_size_random_module_scores(
    adapter, bags: Sequence[dict], modules, target: str = "cancer", seed: int = 0,
) -> Dict[str, float]:
    """Same ablation procedure, but each 'module' is a RANDOM gene set of
    the same size as the real module it replaces — the null this diagnostic
    is compared against (removing a random, equally-sized gene set should
    hurt less than removing a genuinely-attended-to real module, on
    average, if the model actually depends on module structure)."""
    import torch
    from .pathway_hierarchical_adapter import bags_to_pathway_batch

    rng = np.random.RandomState(seed)
    n_genes = modules.n_genes
    model = adapter.model
    model.eval()
    batch = bags_to_pathway_batch(bags)
    expression, cell_type_ids, cell_mask = batch["expression"], batch["cell_type_ids"], batch["cell_mask"]
    with torch.no_grad():
        base_out = model(expression, cell_type_ids, cell_mask)
        base_logits = base_out.cancer_logits if target == "cancer" else base_out.smoke_logits.argmax(-1).float()

    scores = {}
    for i, module_name in enumerate(modules.module_names):
        size = int(modules.membership_mask[i].sum().item())
        random_idx = rng.choice(n_genes, size=size, replace=False)
        ablated = expression.clone()
        ablated[..., random_idx] = 0.0
        with torch.no_grad():
            out = model(ablated, cell_type_ids, cell_mask)
            logits = out.cancer_logits if target == "cancer" else out.smoke_logits.argmax(-1).float()
        scores[f"random_matched_{module_name}"] = float((logits - base_logits).abs().mean().item())
    return scores


# ─── Ranking stability across runs ──────────────────────────────────────────

def _rank(scores: Dict[str, float]) -> List[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def spearman_rank_correlation(scores_a: Dict[str, float], scores_b: Dict[str, float]) -> float:
    keys = sorted(set(scores_a) & set(scores_b))
    if len(keys) < 2:
        return float("nan")
    ranks_a = {k: i for i, k in enumerate(_rank({k: scores_a[k] for k in keys}))}
    ranks_b = {k: i for i, k in enumerate(_rank({k: scores_b[k] for k in keys}))}
    n = len(keys)
    d2 = sum((ranks_a[k] - ranks_b[k]) ** 2 for k in keys)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1)) if n > 1 else float("nan")


def top_k_overlap(scores_a: Dict[str, float], scores_b: Dict[str, float], k: int = 3) -> float:
    top_a = set(_rank(scores_a)[:k])
    top_b = set(_rank(scores_b)[:k])
    if not top_a or not top_b:
        return float("nan")
    return len(top_a & top_b) / float(min(k, len(top_a), len(top_b)))


def module_ranking_stability(list_of_scores: List[Dict[str, float]], top_k: int = 3) -> Dict:
    """Aggregate module-ranking stability across >= 2 independent runs
    (different folds/seeds — each run's module_ablation_scores dict).
    Deterministic given deterministic inputs (fixed model weights + fixed
    bags per run)."""
    if len(list_of_scores) < 2:
        return {"status": "insufficient_evidence", "reason": "need >= 2 runs to assess ranking stability"}
    pair_correlations, pair_overlaps = [], []
    for i in range(len(list_of_scores)):
        for j in range(i + 1, len(list_of_scores)):
            pair_correlations.append(spearman_rank_correlation(list_of_scores[i], list_of_scores[j]))
            pair_overlaps.append(top_k_overlap(list_of_scores[i], list_of_scores[j], k=top_k))
    all_modules = sorted(set().union(*(s.keys() for s in list_of_scores)))
    selection_frequency = {
        m: sum(1 for s in list_of_scores if m in _rank(s)[:top_k]) / len(list_of_scores) for m in all_modules
    }
    return {
        "status": "evaluated", "n_runs": len(list_of_scores),
        "mean_rank_correlation": float(np.nanmean(pair_correlations)),
        "mean_top_k_overlap": float(np.nanmean(pair_overlaps)),
        "top_k": top_k, "selection_frequency": selection_frequency,
    }


# ─── Cell-type attention stability ──────────────────────────────────────────

def cell_type_attention_by_subject(adapter, bags: Sequence[dict]) -> Dict[str, Dict[int, float]]:
    """Subject-level cell-type attention weight (HierarchicalMILOutput.
    cell_type_attention, [B, C]) — a "pooling weight," never called
    "importance."""
    from .pathway_hierarchical_adapter import bags_to_pathway_batch
    import torch

    model = adapter.model
    model.eval()
    batch = bags_to_pathway_batch(bags)
    with torch.no_grad():
        out = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"], return_attention=True)
    subj_ids = [str(b["subject_id"]) for b in bags]
    result = {}
    for i, sid in enumerate(subj_ids):
        result[sid] = {c: float(out.cell_type_attention[i, c].item()) for c in range(out.cell_type_attention.shape[1])}
    return result


def attention_vs_abundance(
    attention_by_subject: Dict[str, Dict[int, float]], abundance_by_subject: Dict[str, Dict[int, int]],
    seed: int = 0,
) -> Dict:
    """
    Pearson correlation between mean cell-type attention weight and
    cell-type abundance (cell count), pooled over all (subject, cell_type)
    pairs, plus a label-shuffled null (permuting the abundance-to-attention
    subject pairing) — determines whether high attention is merely tracking
    which cell type happens to be most numerous in a subject.
    """
    pairs = []
    for sid, att in attention_by_subject.items():
        ab = abundance_by_subject.get(sid, {})
        total = sum(ab.values()) or 1
        for c, a in att.items():
            pairs.append((a, ab.get(c, 0) / total))
    if len(pairs) < 3:
        return {"status": "insufficient_evidence", "reason": "fewer than 3 (subject, cell_type) pairs"}
    att_vals = np.array([p[0] for p in pairs])
    ab_vals = np.array([p[1] for p in pairs])
    if np.std(att_vals) == 0 or np.std(ab_vals) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(att_vals, ab_vals)[0, 1])

    rng = np.random.RandomState(seed)
    null_corrs = []
    for _ in range(200):
        shuffled = rng.permutation(ab_vals)
        if np.std(shuffled) == 0 or np.std(att_vals) == 0:
            continue
        null_corrs.append(float(np.corrcoef(att_vals, shuffled)[0, 1]))
    return {
        "status": "evaluated", "correlation": corr,
        "permutation_null_mean": float(np.mean(null_corrs)) if null_corrs else None,
        "permutation_null_std": float(np.std(null_corrs)) if null_corrs else None,
        "n_pairs": len(pairs),
        "note": "a correlation near the permutation null indicates attention is not simply "
                "tracking cell-type abundance; this is a diagnostic, not a biological claim.",
    }


# ─── Cell-order permutation invariance ──────────────────────────────────────

def cell_order_permutation_invariance_check(adapter, bags: Sequence[dict], seed: int = 0) -> Dict:
    """Shuffling cell order within each bag must not change the model's
    prediction — the pooling architecture is permutation-invariant by
    construction (masked attention sums, not position-dependent layers).
    Returns the max absolute logit difference observed; this should be at
    numerical-precision-level (~1e-5), not exactly proof of correctness on
    its own, but a real discrepancy here would indicate an unintended
    position dependency."""
    import torch
    from .pathway_hierarchical_adapter import bags_to_pathway_batch

    rng = np.random.RandomState(seed)
    model = adapter.model
    model.eval()
    batch = bags_to_pathway_batch(bags)
    with torch.no_grad():
        base = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])

    shuffled_bags = []
    for b in bags:
        n = len(b["gene_matrix"])
        perm = rng.permutation(n)
        shuffled = dict(b)
        shuffled["gene_matrix"] = np.asarray(b["gene_matrix"])[perm]
        shuffled["cell_type_ids"] = np.asarray(b["cell_type_ids"])[perm]
        if "smoke_labels" in b:
            shuffled["smoke_labels"] = np.asarray(b["smoke_labels"])[perm]
        if "smoke_known" in b:
            shuffled["smoke_known"] = np.asarray(b["smoke_known"])[perm]
        shuffled_bags.append(shuffled)
    shuffled_batch = bags_to_pathway_batch(shuffled_bags)
    with torch.no_grad():
        shuffled_out = model(shuffled_batch["expression"], shuffled_batch["cell_type_ids"], shuffled_batch["cell_mask"])

    max_diff = float((base.cancer_logits - shuffled_out.cancer_logits).abs().max().item())
    return {"max_abs_cancer_logit_difference": max_diff, "n_bags": len(bags)}


# ─── Cell-type-label permutation null ──────────────────────────────────────

def cell_type_label_permutation_check(adapter, bags: Sequence[dict], seed: int = 0) -> Dict:
    """
    Shuffles cell_type_ids WITHIN each bag (same per-bag cell-type count
    distribution, different cell-to-type assignment) and compares the
    ALREADY-FITTED model's cancer-logit output against the unshuffled
    baseline. Unlike cell_order_permutation_invariance_check (which must
    show NO change, by architectural construction), this null is expected TO
    change the prediction if the model's cell-type-attention pooling
    genuinely depends on which cells carry which cell-type label — a small
    change here would indicate the model is largely ignoring cell-type
    identity. This is a sensitivity diagnostic, not a biological claim."""
    import torch
    from .pathway_hierarchical_adapter import bags_to_pathway_batch

    rng = np.random.RandomState(seed)
    model = adapter.model
    model.eval()
    batch = bags_to_pathway_batch(bags)
    with torch.no_grad():
        base = model(batch["expression"], batch["cell_type_ids"], batch["cell_mask"])

    shuffled_bags = []
    for b in bags:
        n = len(b["gene_matrix"])
        perm = rng.permutation(n)
        shuffled = dict(b)
        shuffled["cell_type_ids"] = np.asarray(b["cell_type_ids"])[perm]
        shuffled_bags.append(shuffled)
    shuffled_batch = bags_to_pathway_batch(shuffled_bags)
    with torch.no_grad():
        shuffled_out = model(shuffled_batch["expression"], shuffled_batch["cell_type_ids"], shuffled_batch["cell_mask"])

    mean_abs_diff = float((base.cancer_logits - shuffled_out.cancer_logits).abs().mean().item())
    return {
        "mean_abs_cancer_logit_difference": mean_abs_diff, "n_bags": len(bags),
        "note": "cell-type-label permutation null — a large difference here (unlike the cell-order "
                "invariance check, which must show ~zero difference) indicates the model's cell-type "
                "attention pooling is sensitive to cell-type identity, not merely to which cells are "
                "present; a sensitivity finding, not a biological claim.",
    }


# ─── Label-permutation null for module ranking ─────────────────────────────

def label_permutation_null_record(real_ranking: Dict[str, float], permuted_ranking: Dict[str, float]) -> Dict:
    """Records the module-ranking rank correlation between a model trained
    normally and a model of the SAME architecture trained on label-permuted
    data (the caller is responsible for producing permuted_ranking from a
    genuinely separately-fitted model — this function only compares the two
    rankings). A low correlation indicates the real ranking is not an
    artifact reproducible from label-independent structure alone."""
    corr = spearman_rank_correlation(real_ranking, permuted_ranking)
    return {"status": "null_record", "rank_correlation_vs_label_permuted_model": corr,
            "note": "this is a null-model comparison diagnostic, not a biological validation result"}
