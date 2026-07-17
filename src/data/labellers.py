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
    Transfer cigarette/cigar/dual-use smoke-type evidence from NLST
    clinical metadata to cells matched by subject_id, via the explicit
    field parser in data/nlst_smoking.py.

    Novel: first linkage of NLST clinical smoking categories to
    single-cell gene expression data.

    NLST fields: CIGSMOK (cigarette), CIGAR (cigar use flag) — see
    data/nlst_smoking.py's module docstring for exactly which codes this
    repository trusts as verified evidence, and why every other value
    (missing, blank, null, an undocumented code, or a malformed value)
    stays unknown rather than being coerced into cigarette, cigar, or
    "unexposed". Matching NLST participation alone is never treated as
    cigarette exposure — only a documented positive CIGSMOK/CIGAR code is.

    A matched subject whose CIGSMOK/CIGAR fields don't parse to a
    documented positive code gets smoke_type_known=False,
    smoke_type_name="unknown", and explicit
    smoke_type_source/smoke_type_method/smoke_type_limitation provenance —
    this OVERWRITES whatever smoke_type_known state the cell had before
    (e.g. an upstream accession-level default), because NLST linkage is
    this project's intended source of truth for a scRNA-seq subject's
    smoking status (see data/assembly.py's module docstring) and a failed
    linkage must not silently leave a fabricated upstream label standing
    unlabelled as such. A subject with NO row in the NLST CSV at all is
    entirely untouched — this function only acts on subjects it actually
    matched.

    Raw CIGSMOK/CIGAR values are never written into adata.obs or printed
    per-subject — only aggregate counts are reported, consistent with not
    exposing restricted participant-level fields in downstream artifacts.
    """
    if not Path(nlst_csv).exists():
        print("[label] NLST CSV not found — skipping label transfer")
        return adata

    from data.nlst_smoking import parse_nlst_smoking_row

    nlst = pd.read_csv(nlst_csv, low_memory=False)
    nlst["subject_id"] = nlst["pid"].astype(str)

    records = {}
    for _, row in nlst.iterrows():
        records[row["subject_id"]] = parse_nlst_smoking_row(
            row["CIGSMOK"] if "CIGSMOK" in nlst.columns else None,
            row["CIGAR"] if "CIGAR" in nlst.columns else None,
        )

    matched_subject = adata.obs[subject_col].astype(str).map(lambda s: records.get(s))
    matched_mask = matched_subject.notna()
    n_matched = int(matched_mask.sum())

    for col, default in (
        ("smoke_type_known", True),   # legacy default for cells NLST never touches — see loaders.py
        ("smoke_type_source", None),
        ("smoke_type_method", None),
        ("smoke_type_limitation", None),
    ):
        if col not in adata.obs.columns:
            adata.obs[col] = default

    if n_matched == 0:
        print("[label] NLST  no subject ID overlap — labels unchanged")
        return adata

    matched_idx = adata.obs.index[matched_mask.values]
    recs = [matched_subject.loc[i] for i in matched_idx]
    known_flags = [bool(r.smoke_type_known) for r in recs]
    # No verified label -> smoke_type_name is stamped "unknown" explicitly
    # (never left as a stale numeric-looking name next to
    # smoke_type_known=False) and smoke_type falls back to the same inert
    # placeholder id used elsewhere for unknown cells (see
    # data/converters.py::_load_gse136831_cell_metadata) — smoke_type_known
    # is what every downstream consumer must gate on, not this value.
    names = [r.effective_smoke_type if r.smoke_type_known else "unknown" for r in recs]

    adata.obs.loc[matched_idx, "smoke_type_known"] = known_flags
    adata.obs.loc[matched_idx, "smoke_type_source"] = [r.source for r in recs]
    adata.obs.loc[matched_idx, "smoke_type_method"] = [r.method for r in recs]
    adata.obs.loc[matched_idx, "smoke_type_limitation"] = [r.limitation for r in recs]
    adata.obs.loc[matched_idx, "smoke_type_name"] = names
    adata.obs.loc[matched_idx, "smoke_type"] = [SMOKE_TYPE_MAP.get(n, 5) for n in names]

    n_known = sum(known_flags)
    n_unknown = len(known_flags) - n_known
    print(f"[label] NLST  {n_matched:,} cells matched  "
          f"(verified={n_known:,}  unknown={n_unknown:,}, "
          f"{n_matched/adata.n_obs:.1%} of {adata.n_obs:,} total cells)")
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
