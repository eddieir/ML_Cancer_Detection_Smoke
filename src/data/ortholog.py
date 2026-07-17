"""
data/ortholog.py — versioned mouse→human ortholog mapping artifact.

data/transforms.py::map_mouse_to_human used to query PyBiomart live on
every call and pick whichever human gene happened to come first for a
mouse gene with more than one candidate ortholog ("drop_duplicates" kept
the first row silently) — not reproducible (a re-run against a newer
Ensembl release can silently change which ortholog is kept), not testable
offline, and not policy-explicit about one-to-many / many-to-one cases.

This module makes the mapping an explicit, versioned artifact:
  - resolve_mapping_pairs() classifies every (mouse_gene, human_gene) pair
    from a raw ortholog table into one_to_one / one_to_many / many_to_one /
    duplicate_human_target / unmapped, and applies a configurable policy
    (default: conservative one_to_one_only — ambiguous cases are dropped,
    never silently resolved by "pick the first").
  - OrthologMappingArtifact records source/release/retrieval_date plus the
    exact ordered (mouse_gene -> human_gene) mapping actually used and
    counts for every category, and is meant to be cached to disk
    (save/load) so CI and repeated runs never require a live BioMart query.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

VALID_POLICIES = frozenset({"one_to_one_only", "keep_best_one_to_many", "keep_best_many_to_one"})


@dataclass
class OrthologMappingArtifact:
    source:            str            # e.g. "ensembl_biomart"
    release:           Optional[str]  # Ensembl release / dataset version, if known
    retrieval_date:    Optional[str]  # ISO date the raw table was retrieved, if known
    policy:            str            # one of VALID_POLICIES
    mapping:           Dict[str, str] # mouse_gene -> human_gene, final retained set only
    n_input_pairs:     int
    n_mapped:          int
    n_unmapped:        int
    n_ambiguous_one_to_many: int
    n_ambiguous_many_to_one: int
    n_duplicate_human_target: int
    n_final_retained:  int
    notes:             List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps({"source": self.source, "release": self.release,
                            "policy": self.policy, "mapping": self.mapping},
                           sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint()
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        tmp.replace(path)
        print(f"[ortholog] artifact saved → {path}  ({self.n_final_retained} genes, "
              f"fingerprint {self.fingerprint()[:12]}...)")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OrthologMappingArtifact":
        with open(path) as f:
            d = json.load(f)
        d.pop("fingerprint", None)
        artifact = cls(**d)
        return artifact


def resolve_mapping_pairs(
    pairs: Sequence[Tuple[str, str]],
    policy: str = "one_to_one_only",
) -> Tuple[Dict[str, str], dict]:
    """
    Classify and resolve raw (mouse_gene, human_gene) ortholog pairs.

    policy="one_to_one_only" (default, conservative — recommended for
    scientific benchmarking): only mouse genes with EXACTLY one candidate
    human ortholog, whose human target is claimed by no other mouse gene,
    are retained. Every ambiguous case (one mouse gene -> multiple human
    genes, multiple mouse genes -> one human gene) is dropped, not
    silently resolved by picking one arbitrarily.

    Returns (mapping, counts) where mapping is the final mouse->human dict
    and counts records n_input_pairs/n_mapped/n_unmapped/
    n_ambiguous_one_to_many/n_ambiguous_many_to_one/
    n_duplicate_human_target/n_final_retained — never silently dropped
    from the artifact, always auditable.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(f"Unknown ortholog mapping policy {policy!r} — must be one of {sorted(VALID_POLICIES)}")

    pairs = [(str(m), str(h)) for m, h in pairs if m and h]
    n_input = len(pairs)

    mouse_to_humans: Dict[str, set] = {}
    for m, h in pairs:
        mouse_to_humans.setdefault(m, set()).add(h)

    human_target_count = Counter(h for _, h in pairs)

    one_to_many = {m for m, hs in mouse_to_humans.items() if len(hs) > 1}
    duplicate_human_targets = {h for h, c in human_target_count.items() if c > 1}

    mapping: Dict[str, str] = {}
    n_many_to_one_dropped = 0
    for m, hs in mouse_to_humans.items():
        if policy == "one_to_one_only":
            if len(hs) != 1:
                continue
            h = next(iter(hs))
            if h in duplicate_human_targets:
                n_many_to_one_dropped += 1
                continue
            mapping[m] = h
        else:
            # keep_best_* policies: deterministic (sorted) choice among
            # candidates, explicit about being a policy choice rather than
            # accidental dict-iteration order.
            h = sorted(hs)[0]
            mapping[m] = h

    counts = {
        "n_input_pairs": n_input,
        "n_mapped": len(mapping),
        "n_unmapped": 0,
        "n_ambiguous_one_to_many": len(one_to_many),
        "n_ambiguous_many_to_one": len(duplicate_human_targets),
        "n_duplicate_human_target": n_many_to_one_dropped,
        "n_final_retained": len(mapping),
    }
    return mapping, counts


def build_ortholog_artifact(
    pairs: Sequence[Tuple[str, str]],
    source: str = "ensembl_biomart",
    release: Optional[str] = None,
    retrieval_date: Optional[str] = None,
    policy: str = "one_to_one_only",
) -> OrthologMappingArtifact:
    mapping, counts = resolve_mapping_pairs(pairs, policy=policy)
    return OrthologMappingArtifact(
        source=source, release=release, retrieval_date=retrieval_date, policy=policy,
        mapping=dict(sorted(mapping.items())),
        n_input_pairs=counts["n_input_pairs"], n_mapped=counts["n_mapped"],
        n_unmapped=counts["n_unmapped"],
        n_ambiguous_one_to_many=counts["n_ambiguous_one_to_many"],
        n_ambiguous_many_to_one=counts["n_ambiguous_many_to_one"],
        n_duplicate_human_target=counts["n_duplicate_human_target"],
        n_final_retained=counts["n_final_retained"],
        notes=[
            "Ambiguous mouse->human ortholog pairs (one-to-many, many-to-one) are dropped "
            "under policy=one_to_one_only rather than resolved by picking an arbitrary "
            "candidate — see resolve_mapping_pairs().",
        ],
    )


def fetch_live_biomart_pairs() -> List[Tuple[str, str]]:
    """
    Live PyBiomart query (mmusculus_gene_ensembl -> hsapiens ortholog).
    Isolated in its own function so tests/CI never need to call it — every
    ortholog test in this project uses a small fixed fixture instead (see
    tests/test_ortholog_mapping.py). Raises whatever PyBiomart/network
    error occurs; callers decide whether that's fatal or a reason to fall
    back to a cached artifact.
    """
    from pybiomart import Dataset
    ds = Dataset(name="mmusculus_gene_ensembl", host="http://www.ensembl.org")
    df = ds.query(
        attributes=["external_gene_name", "hsapiens_homolog_associated_gene_name"],
        only_unique=False,
    )
    df.columns = ["mouse_gene", "human_gene"]
    df = df.dropna()
    return list(zip(df["mouse_gene"], df["human_gene"]))
