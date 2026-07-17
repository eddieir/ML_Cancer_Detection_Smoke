"""
data/manifest.py — versioned, machine-readable dataset provenance manifest.

Every dataset this project can ingest (GSE136831, GSE288003, GSE123352,
GSE307690/CANUCK, TCGA-LUAD, TCGA-LUSC, NLST) gets one DatasetManifestEntry
recording where it came from, what it actually is (species/assay/matrix
representation), which identifier fields exist, and — when the raw/
processed files are actually present on disk — their real SHA-256
checksums. A dataset with no local files yet still gets an entry (built
from this project's already-documented registries in data/downloaders.py),
but its checksum fields are explicitly null with `files_present=False`
rather than a fabricated placeholder hash: a manifest entry must never
claim to checksum a file nobody has actually downloaded.

configs/datasets.yaml is the checked-in, human-editable seed for these
entries (accession, URLs, species, identifier fields — facts that don't
depend on which files happen to be present in this environment).
build_dataset_manifest() reads that seed and fills in the checksum/
file-presence facts that DO depend on the local filesystem, producing the
full manifest written to data/processed/dataset_manifest.json.

The manifest's fingerprint (manifest_fingerprint()) is meant to be folded
into every downstream artifact identity (PreprocessingArtifact, split
manifest, experiment context) — see ARCHITECTURE.md's provenance-flow
section. A changed source file, accession, or label-policy version changes
that fingerprint, which is the point: an experiment's identity should not
silently stay the same when what it was trained on changed.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

REQUIRED_FIELDS = (
    "dataset_id", "accession", "source_url", "official_record_url",
    "species", "assay_type", "matrix_representation",
    "subject_identifier_field", "license_or_access_level",
    "controlled_access", "synthetic_or_real",
)

MANIFEST_SCHEMA_VERSION = "1"


class ManifestValidationError(ValueError):
    """Raised when a dataset manifest entry is missing a required
    provenance field, or when a checksum is present without the
    corresponding file actually being on disk (which would mean the
    checksum was fabricated rather than computed)."""


@dataclass
class DatasetManifestEntry:
    dataset_id:                str
    accession:                 str
    source_url:                str
    official_record_url:       str
    species:                   str
    assay_type:                str
    matrix_representation:     str
    subject_identifier_field:  str
    license_or_access_level:   str
    controlled_access:         bool
    synthetic_or_real:         str  # "real" | "synthetic_fixture"
    publication_reference:     Optional[str] = None
    download_date:              Optional[str] = None
    genome_build:               Optional[str] = None
    gene_identifier_type:       Optional[str] = None
    sample_identifier_field:    Optional[str] = None
    cell_identifier_field:      Optional[str] = None
    smoke_metadata_fields:      List[str] = field(default_factory=list)
    cancer_metadata_fields:     List[str] = field(default_factory=list)
    malignancy_metadata_fields: List[str] = field(default_factory=list)
    label_policy_version:       str = "1"
    known_limitations:          List[str] = field(default_factory=list)
    raw_file_names:              List[str] = field(default_factory=list)
    raw_file_checksums:          Dict[str, Optional[str]] = field(default_factory=dict)
    processed_file_checksums:    Dict[str, Optional[str]] = field(default_factory=dict)
    files_present:               bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        d = self.to_dict()
        missing = [f for f in REQUIRED_FIELDS if not d.get(f) and d.get(f) is not False]
        if missing:
            raise ManifestValidationError(
                f"Dataset manifest entry {self.dataset_id!r} is missing required field(s) "
                f"{missing} — dataset processing must refuse to start with incomplete "
                "provenance rather than proceed with an unaudited dataset."
            )
        for bucket_name, bucket in (
            ("raw_file_checksums", self.raw_file_checksums),
            ("processed_file_checksums", self.processed_file_checksums),
        ):
            for fname, checksum in bucket.items():
                if checksum is not None and not self.files_present:
                    raise ManifestValidationError(
                        f"{self.dataset_id!r}.{bucket_name}[{fname!r}] has a checksum but "
                        "files_present=False — a checksum must never be recorded for a file "
                        "that was not actually hashed on disk."
                    )


def sha256_of_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checksum_existing_files(directory: Union[str, Path], filenames: List[str]) -> Dict[str, Optional[str]]:
    """
    Real SHA-256 for each filename actually present under `directory`;
    None (never a fabricated value) for any filename not found. This is
    the only function in this module allowed to produce a checksum.
    """
    directory = Path(directory)
    out: Dict[str, Optional[str]] = {}
    for name in filenames:
        p = directory / name
        out[name] = sha256_of_file(p) if p.exists() else None
    return out


def load_manifest_seed(path: Union[str, Path]) -> List[dict]:
    with open(path) as f:
        seed = yaml.safe_load(f) or {}
    return seed.get("datasets", [])


def build_dataset_manifest(
    seed_path: Union[str, Path],
    raw_root: Optional[Union[str, Path]] = None,
) -> List[DatasetManifestEntry]:
    """
    Build the full manifest from configs/datasets.yaml (accession/URL/
    species/identifier-field facts) plus, for any entry whose `raw_subdir`
    is present under `raw_root` (default data/raw), real checksums of
    whichever `raw_file_names` actually exist there. Never invents a
    checksum for a file it did not find.
    """
    from pathlib import Path as _P
    raw_root = _P(raw_root) if raw_root else _P(__file__).parents[2] / "data" / "raw"

    entries = []
    for d in load_manifest_seed(seed_path):
        d = dict(d)
        raw_subdir = d.pop("raw_subdir", None)
        raw_names = d.get("raw_file_names", [])
        files_present = False
        checksums: Dict[str, Optional[str]] = {n: None for n in raw_names}
        if raw_subdir and raw_names:
            src_dir = raw_root / raw_subdir
            if src_dir.exists():
                checksums = checksum_existing_files(src_dir, raw_names)
                files_present = any(v is not None for v in checksums.values())
        d["raw_file_checksums"] = checksums
        d["files_present"] = files_present
        entry = DatasetManifestEntry(**d)
        entry.validate()
        entries.append(entry)
    return entries


def manifest_fingerprint(entries: List[DatasetManifestEntry]) -> str:
    """
    Deterministic SHA-256 over every entry's provenance-relevant fields
    (excludes download_date, which changes on every re-download without
    the underlying data changing). Meant to be embedded in
    PreprocessingArtifact / ExperimentContext so a changed accession,
    label-policy version, or checksum changes downstream artifact
    identities — see ARCHITECTURE.md.
    """
    payload = []
    for e in sorted(entries, key=lambda e: e.dataset_id):
        d = e.to_dict()
        d.pop("download_date", None)
        payload.append(d)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def save_manifest(entries: List[DatasetManifestEntry], path: Union[str, Path]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = manifest_fingerprint(entries)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "fingerprint": fp,
        "datasets": [e.to_dict() for e in entries],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)
    print(f"[manifest] saved {len(entries)} dataset entries → {path}  (fingerprint {fp[:12]}...)")
    return fp


def load_manifest(path: Union[str, Path]) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build/validate the dataset provenance manifest")
    parser.add_argument("--seed", default="configs/datasets.yaml")
    parser.add_argument("--out", default="data/processed/dataset_manifest.json")
    parser.add_argument("--raw-root", default=None)
    args = parser.parse_args()

    built = build_dataset_manifest(args.seed, args.raw_root)
    for e in built:
        status = "files present" if e.files_present else "no local files (metadata-only entry)"
        print(f"  {e.dataset_id:<16} {e.accession:<14} species={e.species:<6} {status}")
    save_manifest(built, args.out)
