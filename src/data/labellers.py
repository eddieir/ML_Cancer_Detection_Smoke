"""
data/labellers.py — assigns smoke type, malignancy, and class weights.
No I/O, no transformation logic.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import anndata as ad

from constants import SMOKE_TYPE_MAP, SMOKE_TYPES, N_SMOKE_CLASSES


def transfer_nlst_labels(
    adata: ad.AnnData,
    nlst_csv: str,
    subject_col: str = "subject_id",
) -> ad.AnnData:
    """
    Transfer cigar and dual-use labels from NLST clinical metadata
    to cells matched by subject_id.

    Novel: first linkage of NLST clinical smoking categories to
    single-cell gene expression data.

    NLST fields: CIGSMOK (cigarette), CIGAR (cigar use flag).
    """
    if not Path(nlst_csv).exists():
        print("[label] NLST CSV not found — skipping label transfer")
        return adata

    nlst = pd.read_csv(nlst_csv, low_memory=False)
    nlst["subject_id"] = nlst["pid"].astype(str)

    def _assign(row) -> int:
        cigar = row.get("CIGAR",   0) == 1
        cig   = row.get("CIGSMOK", 0) in [1, 2]
        if cigar and cig: return 4   # dual-use
        if cigar:         return 2   # cigar only
        if cig:           return 0   # cigarette only
        return 5                     # unexposed

    nlst_map = dict(zip(nlst["subject_id"], nlst.apply(_assign, axis=1)))
    matched  = adata.obs[subject_col].map(nlst_map)
    n        = matched.notna().sum()

    if "smoke_type_known" not in adata.obs.columns:
        adata.obs["smoke_type_known"] = True  # legacy default — see loaders.py::_attach_standard_obs

    if n > 0:
        adata.obs.loc[matched.notna(), "smoke_type"] = matched.dropna().astype(int)
        # NLST CIGSMOK/CIGAR are verified clinical fields for the matched
        # subject — a real measurement, not a proxy — so these cells are
        # marked known.
        adata.obs.loc[matched.notna(), "smoke_type_known"] = True
        print(f"[label] NLST  {n:,} cells relabelled ({n/adata.n_obs:.1%})")
    else:
        print("[label] NLST  no subject ID overlap — labels unchanged")
    return adata


def apply_weak_smoke_proxies(adata: ad.AnnData, enabled: bool = False) -> ad.AnnData:
    """
    Explicit, opt-in-only promotion of a documented weak smoke-type proxy
    (currently: GSE136831's COPD-diagnosis proxy — see
    data/converters.py::_load_gse136831_cell_metadata) into the primary
    smoke_type/smoke_type_name/smoke_type_known fields.

    enabled=False (the default — configs/default.yaml's
    data.weak_labels.enabled) leaves every weak-proxy cell's smoke_type_name
    ="unknown", smoke_type_known=False, exactly as the loader/converter set
    it. Nothing here changes those cells' effective label — this is what
    keeps a weak proxy out of smoke-classification loss/class-weighting/
    subject-balanced sampling/stratification/metrics/model-selection under
    the default verified_only label policy (those all key off
    smoke_type_known, not off whether a weak-proxy VALUE happens to exist).

    enabled=True is a deliberate, disclosed, non-default experiment: for
    every cell with weak_smoke_proxy_known=True, smoke_type_name/smoke_type
    are overwritten with the proxy's value (e.g. "cigarette") and
    smoke_type_known is set True for exactly those cells — never silently,
    only behind this explicit flag, and always reported (see the printed
    count below and label_quality_report.py). Cells with no weak proxy
    available are unaffected either way — they remain smoke_type_known
    =False regardless of this flag, since there is nothing to promote.
    """
    if "weak_smoke_proxy_known" not in adata.obs.columns:
        return adata  # nothing to do — no source in this merge carries a weak proxy column

    proxy_mask = adata.obs["weak_smoke_proxy_known"].astype(bool)
    n_proxy = int(proxy_mask.sum())
    if n_proxy == 0:
        return adata

    if not enabled:
        print(f"[label] weak_smoke_proxy  {n_proxy:,} cell(s) carry a documented weak proxy "
              "but data.weak_labels.enabled=false (default) — left smoke_type_known=False, "
              "excluded from smoke-classification supervision under the verified_only policy.")
        return adata

    proxy_values = adata.obs.loc[proxy_mask, "weak_smoke_proxy_value"].astype(str).str.lower()
    adata.obs.loc[proxy_mask, "smoke_type_name"] = proxy_values.values
    adata.obs.loc[proxy_mask, "smoke_type"] = proxy_values.map(
        lambda s: SMOKE_TYPE_MAP.get(s, 5)
    ).values
    adata.obs.loc[proxy_mask, "smoke_type_known"] = True
    print(f"[label] weak_smoke_proxy  data.weak_labels.enabled=true — promoted {n_proxy:,} "
          "weak-proxy cell(s) into smoke_type_known=True (non-default, disclosed opt-in; "
          "see weak_smoke_proxy_source/weak_smoke_proxy_limitation for provenance).")
    return adata


def add_malignancy_labels(
    adata: ad.AnnData,
    tumor_barcodes: Optional[list] = None,
) -> ad.AnnData:
    """
    Assign per-cell malignancy labels.
    Priority: tumor_barcodes list > values a loader already set in obs
    (e.g. TCGA tumor/NAT via convert_tcga's samples_meta.csv) > 0.0 default.

    0.0 is used as a numeric placeholder for cells with no real malignancy
    label, but it is NOT a verified-normal call — malignancy_known marks
    which cells actually carry ground truth (tumor_barcodes match, or a
    loader-set per-sample label such as TCGA tumor/NAT) versus which are
    just filled with the placeholder so the array has a value everywhere.
    MultiTaskLoss must mask its BCE loss to malignancy_known cells only
    (see model.py::MultiTaskLoss._lm), otherwise every unlabelled cell
    silently trains the model to say "not malignant".
    """
    if "malignancy" not in adata.obs.columns:
        adata.obs["malignancy"] = 0.0
    if "malignancy_known" not in adata.obs.columns:
        adata.obs["malignancy_known"] = False

    if tumor_barcodes:
        mask = adata.obs_names.isin(set(tumor_barcodes))
        adata.obs.loc[mask, "malignancy"] = 1.0
        adata.obs.loc[mask, "malignancy_known"] = True
        print(f"[label] malignancy  {mask.sum():,} tumor cells set to 1.0 (known)")

    adata.obs["malignancy"]       = adata.obs["malignancy"].astype(np.float32)
    adata.obs["malignancy_known"] = adata.obs["malignancy_known"].astype(bool)

    known = adata.obs["malignancy_known"]
    n_pos = int((known & (adata.obs["malignancy"] == 1.0)).sum())
    n_neg = int((known & (adata.obs["malignancy"] == 0.0)).sum())
    n_unk = int((~known).sum())
    print(f"[label] malignancy  known_positive={n_pos:,}  known_negative={n_neg:,}  "
          f"unknown={n_unk:,}")
    return adata


def compute_smoke_class_weights(
    smoke_labels: np.ndarray,
    n_classes: int = N_SMOKE_CLASSES,
    smoke_known: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Inverse-frequency weights for CrossEntropyLoss.
    Cannabis and dual-use are severely under-represented vs cigarette.
    Returns array of shape [n_classes] for direct use in nn.CrossEntropyLoss.

    smoke_known, if given, restricts the class-frequency count to cells
    with a verified (or explicitly opted-in weak-proxy) smoke label —
    cells with smoke_known=False must never influence class weighting,
    the same guarantee train.CellLevelDataset.smoke_class_weights enforces
    for the actual training path. Passing smoke_known=None reproduces the
    previous unmasked behaviour, for callers that already filtered their
    input (or are using fully-synthetic, fully-known labels).
    """
    if smoke_known is not None:
        smoke_known = np.asarray(smoke_known, dtype=bool)
        smoke_labels = np.asarray(smoke_labels)[smoke_known]
    counts  = np.bincount(smoke_labels, minlength=n_classes).astype(np.float32)
    counts  = np.maximum(counts, 1)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.sum() * n_classes   # sum to n_classes

    for i, (w, c) in enumerate(zip(weights, counts)):
        print(f"[label] weight  {SMOKE_TYPES[i]:<12}  count={int(c):>6,}  weight={w:.3f}")
    return weights
