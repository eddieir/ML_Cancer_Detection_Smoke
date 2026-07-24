"""
evidence/clinical_manifest.py — Issue #16 blocker 3: a signed, versioned
clinical-evidence manifest, replacing the caller-controlled boolean that
`clinical_readiness.guard_clinical_claim()` used to accept.

`signed_external_evidence_manifest=True` could be satisfied by any caller
willing to lie about a single flag. This module replaces that with a
concrete, versioned `ClinicalEvidenceManifest` object whose signature is
verified against a configured allow-list of approved public keys, whose
referenced evidence reports are independently validated (schema + content
fingerprint), and whose model/preprocessing/calibration/threshold
identities are cross-checked against those reports — none of which a
caller can satisfy merely by passing a boolean.

No approved public key ships in this repository's default configuration
(see configs/clinical_readiness.yaml — there is no
`approved_reviewer_public_keys` entry there). Without one, ANY manifest —
correctly signed or not — is rejected, so this module cannot manufacture a
clinical-readiness claim on its own; a real external reviewer's public key
would have to be deliberately added to a deployment's configuration first,
and even then a valid signature only proves the named reviewer signed this
exact manifest content — it is not itself regulatory or clinical approval
(see clinical_readiness.py's REQUIRED_DISCLAIMERS).

Uses `cryptography`'s Ed25519 primitives — no hand-rolled cryptography.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .errors import ClinicalReadinessNotEstablishedError
from .evidence_contract import (
    EVIDENCE_LEVELS,
    EXTERNAL_LEVELS,
    report_from_dict,
    validate_report,
)

MANIFEST_SCHEMA_VERSION = "1"

REQUIRED_MANIFEST_FIELDS = (
    "schema_version", "manifest_id", "created_at", "expires_at",
    "intended_use", "target_population", "clinical_setting", "endpoint",
    "prediction_horizon", "model_fingerprint", "preprocessing_fingerprint",
    "label_mapping_fingerprint", "calibration_fingerprint", "threshold_fingerprint",
    "development_evidence_refs", "internal_test_evidence_refs",
    "external_validation_evidence_refs", "prospective_evidence_refs",
    "clinical_utility_evidence_refs", "privacy_security_assessment_ref",
    "human_factors_assessment_ref", "regulatory_review_ref",
    "reviewer_name", "reviewer_role", "approved_public_key_id", "code_commit_sha",
)

# References this schema requires to be non-empty before a manifest can be
# validated at all — a manifest that names zero pieces of evidence in any
# of these categories is structurally incomplete, independent of signature.
NON_EMPTY_LIST_FIELDS = (
    "development_evidence_refs",
    "internal_test_evidence_refs",
    "external_validation_evidence_refs",
    "prospective_evidence_refs",
    "clinical_utility_evidence_refs",
)

PLACEHOLDER_STRINGS = {None, "", "TBD", "TODO", "N/A", "n/a", "unknown", "placeholder"}


class ClinicalManifestValidationError(ClinicalReadinessNotEstablishedError):
    """Raised for any structural, signature, expiry, or cross-reference
    defect in a ClinicalEvidenceManifest — the manifest-specific counterpart
    to ClinicalReadinessNotEstablishedError."""


@dataclass
class ClinicalEvidenceManifest:
    schema_version: str
    manifest_id: str
    created_at: str
    expires_at: str
    intended_use: str
    target_population: str
    clinical_setting: str
    endpoint: str
    prediction_horizon: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    label_mapping_fingerprint: str
    calibration_fingerprint: str
    threshold_fingerprint: str
    development_evidence_refs: List[str]
    internal_test_evidence_refs: List[str]
    external_validation_evidence_refs: List[str]
    prospective_evidence_refs: List[str]
    clinical_utility_evidence_refs: List[str]
    privacy_security_assessment_ref: str
    human_factors_assessment_ref: str
    regulatory_review_ref: str
    reviewer_name: str
    reviewer_role: str
    approved_public_key_id: str
    code_commit_sha: str
    signature_hex: Optional[str] = None

    def canonical_content(self) -> Dict[str, Any]:
        """Every field except the signature itself — this is exactly what
        gets signed and what a signature verification re-derives."""
        d = {f: getattr(self, f) for f in REQUIRED_MANIFEST_FIELDS}
        return d

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.canonical_content(), sort_keys=True, default=str).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        d = self.canonical_content()
        d["signature_hex"] = self.signature_hex
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ClinicalEvidenceManifest":
        kwargs = {f: data.get(f) for f in REQUIRED_MANIFEST_FIELDS}
        return ClinicalEvidenceManifest(signature_hex=data.get("signature_hex"), **kwargs)


def sign_manifest(manifest: ClinicalEvidenceManifest, private_key: Ed25519PrivateKey) -> ClinicalEvidenceManifest:
    """Signs `manifest.canonical_bytes()` with `private_key` and returns a
    new manifest with `signature_hex` populated. Never mutates in place."""
    signature = private_key.sign(manifest.canonical_bytes())
    signed = ClinicalEvidenceManifest.from_dict(manifest.to_dict())
    signed.signature_hex = signature.hex()
    return signed


def _reject_if_placeholder(field_name: str, value: Any) -> None:
    if isinstance(value, (list, dict)):
        return
    if isinstance(value, str) and value.strip() in PLACEHOLDER_STRINGS:
        raise ClinicalManifestValidationError(
            f"clinical evidence manifest field {field_name!r} has placeholder value {value!r}"
        )
    if value in PLACEHOLDER_STRINGS:
        raise ClinicalManifestValidationError(
            f"clinical evidence manifest field {field_name!r} is missing/placeholder ({value!r})"
        )


def _parse_iso(ts: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ClinicalManifestValidationError(f"manifest field {field_name!r} is not a valid ISO-8601 timestamp: {ts!r}") from exc


def validate_manifest_structure(manifest: ClinicalEvidenceManifest, *, now: Optional[datetime] = None) -> None:
    """Structural + expiry checks only — no signature, no cross-referenced
    evidence. Every REQUIRED_MANIFEST_FIELDS entry must be present and
    non-placeholder; every NON_EMPTY_LIST_FIELDS entry must be non-empty."""
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ClinicalManifestValidationError(
            f"unsupported manifest schema_version {manifest.schema_version!r} — this code "
            f"understands {MANIFEST_SCHEMA_VERSION!r} only"
        )
    for f in REQUIRED_MANIFEST_FIELDS:
        _reject_if_placeholder(f, getattr(manifest, f))
    for f in NON_EMPTY_LIST_FIELDS:
        value = getattr(manifest, f)
        if not isinstance(value, list) or not value:
            raise ClinicalManifestValidationError(
                f"manifest field {f!r} must be a non-empty list of evidence references, got {value!r}"
            )

    now = now or datetime.now(timezone.utc)
    created_at = _parse_iso(manifest.created_at, "created_at")
    expires_at = _parse_iso(manifest.expires_at, "expires_at")
    if expires_at <= created_at:
        raise ClinicalManifestValidationError("manifest expires_at must be strictly after created_at")
    if now >= expires_at:
        raise ClinicalManifestValidationError(
            f"manifest expired at {manifest.expires_at} (checked against {now.isoformat()})"
        )

    if not manifest.signature_hex:
        raise ClinicalManifestValidationError("manifest has no signature_hex — an unsigned manifest is rejected")


def verify_manifest_signature(
    manifest: ClinicalEvidenceManifest, *, approved_public_keys: Dict[str, bytes],
) -> None:
    """Verifies `manifest.signature_hex` against the public key named by
    `manifest.approved_public_key_id`, looked up in the CALLER-SUPPLIED
    `approved_public_keys` allow-list (raw Ed25519 public key bytes, keyed
    by key id) — never a key embedded in the manifest itself, which would
    let a manifest vouch for its own signer. An empty allow-list (the
    default repository configuration ships none) rejects every manifest."""
    key_id = manifest.approved_public_key_id
    key_bytes = approved_public_keys.get(key_id)
    if key_bytes is None:
        raise ClinicalManifestValidationError(
            f"manifest names approved_public_key_id={key_id!r}, which is not present in the "
            "configured approved-public-key allow-list — an unrecognized signer cannot unlock "
            "a clinical-readiness claim"
        )
    try:
        public_key = Ed25519PublicKey.from_public_bytes(key_bytes)
        public_key.verify(bytes.fromhex(manifest.signature_hex), manifest.canonical_bytes())
    except (InvalidSignature, ValueError) as exc:
        raise ClinicalManifestValidationError(
            f"manifest signature does not verify against approved_public_key_id={key_id!r}"
        ) from exc


def validate_evidence_references(
    manifest: ClinicalEvidenceManifest, *, evidence_reports: Dict[str, Dict[str, Any]],
) -> None:
    """Cross-checks every evidence reference the manifest names against
    real, independently-validated EvidenceReport dicts supplied by the
    caller (`evidence_reports`, keyed by the same reference strings the
    manifest lists). Raises unless:
      - every referenced report is present in `evidence_reports`;
      - every referenced report round-trip validates (schema + content
        fingerprint — evidence_contract.validate_report);
      - no referenced report has synthetic_flag=True (synthetic evidence
        can never satisfy a clinical evidence requirement);
      - every internal_test_evidence_refs report has
        frozen_test_access_status='acquired_once';
      - every external_validation_evidence_refs report has
        cohort_role='external_validation' and evidence_level in
        EXTERNAL_LEVELS (an internal split relabeled 'external' is rejected
        here, not just by evidence_contract at write time);
      - every referenced report's model_fingerprint /
        preprocessing_artifact_fingerprint agrees with the manifest's own
        model_fingerprint / preprocessing_fingerprint (a manifest cannot
        bind itself to a different fitted model than the evidence it cites).
    """
    all_refs: Dict[str, List[str]] = {
        "development_evidence_refs": manifest.development_evidence_refs,
        "internal_test_evidence_refs": manifest.internal_test_evidence_refs,
        "external_validation_evidence_refs": manifest.external_validation_evidence_refs,
        "prospective_evidence_refs": manifest.prospective_evidence_refs,
        "clinical_utility_evidence_refs": manifest.clinical_utility_evidence_refs,
    }

    for category, refs in all_refs.items():
        for ref in refs:
            if ref not in evidence_reports:
                raise ClinicalManifestValidationError(
                    f"manifest references {ref!r} in {category!r}, but no matching evidence "
                    "report was supplied for independent validation"
                )
            report = report_from_dict(evidence_reports[ref])
            try:
                validate_report(report)
            except Exception as exc:
                raise ClinicalManifestValidationError(
                    f"referenced evidence report {ref!r} ({category!r}) failed independent validation: {exc}"
                ) from exc
            identity = report.identity
            if bool(identity.get("synthetic_flag")):
                raise ClinicalManifestValidationError(
                    f"referenced evidence report {ref!r} ({category!r}) is synthetic_flag=True — "
                    "synthetic evidence can never satisfy a clinical evidence requirement"
                )
            if category == "internal_test_evidence_refs" and identity.get("frozen_test_access_status") != "acquired_once":
                raise ClinicalManifestValidationError(
                    f"referenced internal-test evidence report {ref!r} does not have "
                    "frozen_test_access_status='acquired_once'"
                )
            if category == "external_validation_evidence_refs":
                if identity.get("cohort_role") != "external_validation":
                    raise ClinicalManifestValidationError(
                        f"referenced external-validation evidence report {ref!r} has "
                        f"cohort_role={identity.get('cohort_role')!r} — an internal split relabeled "
                        "'external' is rejected"
                    )
                if identity.get("evidence_level") not in EXTERNAL_LEVELS:
                    raise ClinicalManifestValidationError(
                        f"referenced external-validation evidence report {ref!r} has "
                        f"evidence_level={identity.get('evidence_level')!r}, not one of {EXTERNAL_LEVELS}"
                    )
            for local_field, manifest_field in (
                ("model_fingerprint", manifest.model_fingerprint),
                ("preprocessing_artifact_fingerprint", manifest.preprocessing_fingerprint),
            ):
                report_value = identity.get(local_field)
                if report_value is not None and report_value != manifest_field:
                    raise ClinicalManifestValidationError(
                        f"referenced evidence report {ref!r} ({category!r}) has "
                        f"{local_field}={report_value!r}, which disagrees with the manifest's own "
                        f"{local_field.replace('_artifact', '')}={manifest_field!r} — a manifest "
                        "cannot bind itself to evidence produced by a different fitted model"
                    )


def validate_clinical_evidence_manifest(
    manifest: ClinicalEvidenceManifest, *,
    approved_public_keys: Dict[str, bytes],
    evidence_reports: Dict[str, Dict[str, Any]],
    now: Optional[datetime] = None,
) -> None:
    """The single entry point evidence/clinical_readiness.py's
    guard_clinical_claim() calls. Raises ClinicalManifestValidationError
    (a ClinicalReadinessNotEstablishedError subclass) on the first failure;
    returns None on success. Runs structural validation, signature
    verification, and evidence cross-referencing, in that order."""
    validate_manifest_structure(manifest, now=now)
    verify_manifest_signature(manifest, approved_public_keys=approved_public_keys)
    validate_evidence_references(manifest, evidence_reports=evidence_reports)
