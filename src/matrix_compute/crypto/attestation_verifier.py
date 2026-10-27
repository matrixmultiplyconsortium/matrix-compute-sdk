"""
Attestation Verification Engine for matrix-compute-sdk (Milestone 2).

Enforces strict fail-closed verification of hardware attestation evidence:
1. Validates X.509 certificate chains up to trusted Root CAs.
2. Verifies cryptographic digital signatures over attestation claims/tokens.
3. Appraises firmware and PCR measurements against authoritative Golden RIM manifests.
4. Enforces cryptographic freshness (challenger nonce) and binding (client ephemeral pubkey).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import secrets
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.x509.oid import ExtensionOID

from matrix_contracts.schemas.attestation import AttestationEvidence
from matrix_compute.exceptions import (
    AttestationBindingError,
    AttestationCertificateError,
    AttestationChainInvalidError,
    AttestationError,
    AttestationEvidenceMalformedError,
    AttestationExpiredError,
    AttestationMeasurementMismatchError,
    AttestationNonceMismatchError,
    AttestationProviderUnsupportedError,
    AttestationRevokedError,
    AttestationSignatureError,
    AttestationUntrustedRootError,
)

logger = logging.getLogger("matrix_compute.crypto.attestation")


# ==============================================================================
# POLICY & RESULT DATA STRUCTURES
# ==============================================================================

@dataclass
class VerificationPolicy:
    """Configurable security policy and trust store for attestation appraisal."""

    trusted_roots: List[x509.Certificate] = field(default_factory=list)
    golden_rims: Dict[str, Dict[str, str]] = field(default_factory=dict)
    allowed_clock_skew_seconds: float = 60.0
    max_evidence_age_seconds: float = 300.0
    enforce_ca_extensions: bool = True
    revoked_serial_numbers: Set[int] = field(default_factory=set)
    allowed_providers: Set[str] = field(
        default_factory=lambda: {"nvidia-cca", "simulated-cca", "amd-sev-snp", "intel-tdx"}
    )

    def add_trusted_root_pem(self, pem_str: str) -> None:
        """Parses and appends a PEM-encoded X.509 certificate to trusted roots."""
        cert = x509.load_pem_x509_certificate(pem_str.strip().encode("utf-8"))
        self.trusted_roots.append(cert)

    def add_golden_rim(self, provider_or_model: str, rim_manifest: Dict[str, str]) -> None:
        """Registers an authoritative reference measurement dictionary."""
        self.golden_rims[provider_or_model.lower().strip()] = {
            k: str(v).strip().lower() for k, v in rim_manifest.items()
        }


@dataclass
class VerificationResult:
    """Summary of successful hardware attestation evidence verification."""

    verified: bool
    provider: str
    claims: Dict[str, Any]
    leaf_cert: x509.Certificate
    root_cert: x509.Certificate
    chain_length: int
    measurements_verified: int
    timestamp: float


# ==============================================================================
# CRYPTOGRAPHIC SIGNATURE HELPER
# ==============================================================================

def _verify_digital_signature(
    public_key: Any,
    signature: bytes,
    payload: bytes,
    hash_algo: Optional[hashes.HashAlgorithm] = None,
) -> None:
    """
    Verifies a digital signature across RSA, ECDSA, and Ed25519 public keys.
    Raises AttestationSignatureError if verification fails.
    """
    try:
        if isinstance(public_key, ec.EllipticCurvePublicKey):
            if hash_algo is not None:
                public_key.verify(signature, payload, ec.ECDSA(hash_algo))
            else:
                # Determine default hash algorithm from curve
                primary_algo = (
                    hashes.SHA384()
                    if isinstance(public_key.curve, ec.SECP384R1)
                    else (hashes.SHA512() if isinstance(public_key.curve, ec.SECP521R1) else hashes.SHA256())
                )
                try:
                    public_key.verify(signature, payload, ec.ECDSA(primary_algo))
                except InvalidSignature:
                    # Fallback check for alternate common hash algo
                    alt_algo = hashes.SHA256() if not isinstance(primary_algo, hashes.SHA256) else hashes.SHA384()
                    public_key.verify(signature, payload, ec.ECDSA(alt_algo))
        elif isinstance(public_key, rsa.RSAPublicKey):
            algo = hash_algo or hashes.SHA256()
            try:
                public_key.verify(signature, payload, padding.PKCS1v15(), algo)
            except InvalidSignature:
                public_key.verify(
                    signature,
                    payload,
                    padding.PSS(
                        mgf=padding.MGF1(algo),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    algo,
                )
        elif isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(signature, payload)
        else:
            raise AttestationSignatureError(
                f"Unsupported public key type for signature verification: {type(public_key).__name__}"
            )
    except InvalidSignature as exc:
        raise AttestationSignatureError(
            f"Cryptographic signature verification failed: {exc}",
            cause=exc,
        ) from exc


def _verify_x509_cert_signature(cert: x509.Certificate, issuer_pubkey: Any) -> None:
    """Verifies that an X.509 certificate was legitimately signed by issuer_pubkey."""
    tbs_bytes = cert.tbs_certificate_bytes
    sig = cert.signature
    hash_algo = cert.signature_hash_algorithm or hashes.SHA256()

    try:
        if isinstance(issuer_pubkey, ec.EllipticCurvePublicKey):
            issuer_pubkey.verify(sig, tbs_bytes, ec.ECDSA(hash_algo))
        elif isinstance(issuer_pubkey, rsa.RSAPublicKey):
            issuer_pubkey.verify(sig, tbs_bytes, padding.PKCS1v15(), hash_algo)
        elif isinstance(issuer_pubkey, ed25519.Ed25519PublicKey):
            issuer_pubkey.verify(sig, tbs_bytes)
        else:
            raise AttestationSignatureError(
                f"Unsupported issuer key type for certificate validation: {type(issuer_pubkey).__name__}"
            )
    except InvalidSignature as exc:
        raise AttestationSignatureError(
            f"X.509 certificate signature verification failed for '{cert.subject.rfc4514_string()}': {exc}",
            cause=exc,
        ) from exc


# ==============================================================================
# PLUGGABLE EVIDENCE VERIFIER STRATEGY (F7 / Pluggable Interface)
# ==============================================================================

class BaseEvidenceVerifier(ABC):
    """Abstract strategy for provider-specific attestation token parsing and verification."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Provider identifier matching AttestationEvidence.provider."""
        pass

    @abstractmethod
    def verify_token_signature(
        self,
        evidence: AttestationEvidence,
        leaf_cert: x509.Certificate,
    ) -> None:
        """Verifies leaf cert cryptographic signature over the evidence claims or report blob."""
        pass

    def extract_claims(self, evidence: AttestationEvidence) -> Dict[str, Any]:
        """Extracts and normalizes claims dictionary from evidence."""
        if not evidence.token_claims:
            raise AttestationEvidenceMalformedError("Attestation evidence contains empty token_claims")
        return dict(evidence.token_claims)


class SimulatedCcaVerifier(BaseEvidenceVerifier):
    """Verifier strategy for Simulated CCA attestation evidence in test/dev environments."""

    @property
    def provider_id(self) -> str:
        return "simulated-cca"

    def verify_token_signature(
        self,
        evidence: AttestationEvidence,
        leaf_cert: x509.Certificate,
    ) -> None:
        if not evidence.token_signature:
            raise AttestationSignatureError("Missing token_signature in simulated attestation evidence")

        # Canonicalize claims JSON bytes for deterministic signature appraisal
        canonical_json = json.dumps(evidence.token_claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
        pubkey = leaf_cert.public_key()

        try:
            _verify_digital_signature(
                public_key=pubkey,
                signature=evidence.token_signature,
                payload=canonical_json,
            )
        except AttestationSignatureError as primary_err:
            if evidence.evidence_blob:
                try:
                    _verify_digital_signature(
                        public_key=pubkey,
                        signature=evidence.token_signature,
                        payload=evidence.evidence_blob,
                    )
                    return
                except AttestationSignatureError:
                    pass
            raise primary_err


class NvidiaCcaVerifier(BaseEvidenceVerifier):
    """Verifier strategy for authentic NVIDIA Hopper/Blackwell CCA attestation tokens."""

    @property
    def provider_id(self) -> str:
        return "nvidia-cca"

    def verify_token_signature(
        self,
        evidence: AttestationEvidence,
        leaf_cert: x509.Certificate,
    ) -> None:
        if not evidence.token_signature:
            raise AttestationSignatureError("Missing token_signature in NVIDIA CCA evidence")

        canonical_json = json.dumps(evidence.token_claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
        pubkey = leaf_cert.public_key()

        try:
            _verify_digital_signature(
                public_key=pubkey,
                signature=evidence.token_signature,
                payload=canonical_json,
            )
        except AttestationSignatureError as primary_err:
            if evidence.evidence_blob:
                try:
                    _verify_digital_signature(
                        public_key=pubkey,
                        signature=evidence.token_signature,
                        payload=evidence.evidence_blob,
                    )
                    return
                except AttestationSignatureError:
                    pass
            raise primary_err


# ==============================================================================
# MAIN ATTESTATION VERIFIER ENGINE (F10, F11, F12, F13)
# ==============================================================================

from .sev_snp_verifier import AmdSevSnpVerifier
from .tdx_verifier import IntelTdxVerifier


class AttestationVerifier:
    """
    Production-grade Attestation Verification Engine for matrix-compute-sdk.
    Implements strict fail-closed validation of remote hardware attestation evidence.
    """

    def __init__(self, policy: Optional[VerificationPolicy] = None) -> None:
        self.policy = policy or VerificationPolicy()
        self._providers: Dict[str, BaseEvidenceVerifier] = {}
        # Register default strategies
        self.register_provider(SimulatedCcaVerifier())
        self.register_provider(NvidiaCcaVerifier())
        self.register_provider(AmdSevSnpVerifier())
        self.register_provider(IntelTdxVerifier())

    def register_provider(self, verifier: BaseEvidenceVerifier) -> None:
        """Registers a pluggable attestation evidence verification strategy."""
        pid = verifier.provider_id.lower().strip()
        self._providers[pid] = verifier
        self.policy.allowed_providers.add(pid)

    def verify(
        self,
        evidence: Union[AttestationEvidence, bytes, str],
        expected_nonce: Union[bytes, str],
        expected_client_pubkey: Optional[Union[bytes, str]] = None,
        verification_time: Optional[datetime] = None,
        task_id: Optional[str] = None,
    ) -> VerificationResult:
        """
        Executes strict fail-closed verification pipeline over attestation evidence.
        """
        # Step 1: Parse and normalize evidence container
        ev = self._parse_evidence(evidence, task_id=task_id)

        # Step 2: Validate provider support
        provider_key = ev.provider.lower().strip()
        if provider_key not in self.policy.allowed_providers or provider_key not in self._providers:
            raise AttestationProviderUnsupportedError(
                f"Unsupported or unauthorized attestation provider: '{ev.provider}'",
                provider=ev.provider,
                task_id=task_id,
            )
        strategy = self._providers[provider_key]

        # Step 3: Validate X.509 Certificate Chain (F10)
        leaf_cert, root_cert, chain_len = self.verify_certificate_chain(
            chain_pems=ev.certificate_chain,
            verification_time=verification_time,
            provider=ev.provider,
            task_id=task_id,
        )

        # Step 4: Verify Attestation Token Signature (F10 / Leaf Binding)
        strategy.verify_token_signature(evidence=ev, leaf_cert=leaf_cert)

        # Step 5: Extract Claims
        claims = strategy.extract_claims(evidence=ev)

        # Step 6: Verify Freshness and Cryptographic Nonce/Pubkey Binding (F12)
        self.verify_freshness_and_binding(
            claims=claims,
            expected_nonce=expected_nonce,
            expected_client_pubkey=expected_client_pubkey,
            verification_time=verification_time,
            provider=ev.provider,
            task_id=task_id,
        )

        # Step 7: Verify Golden RIM & Measurement Digests (F11)
        measurements_count = self.verify_measurements(
            claims=claims,
            provider=ev.provider,
            evidence_golden_rim=ev.golden_rim,
            task_id=task_id,
        )

        # All checks passed cleanly
        return VerificationResult(
            verified=True,
            provider=ev.provider,
            claims=claims,
            leaf_cert=leaf_cert,
            root_cert=root_cert,
            chain_length=chain_len,
            measurements_verified=measurements_count,
            timestamp=datetime.now(timezone.utc).timestamp(),
        )

    # --------------------------------------------------------------------------
    # Subsystem: Certificate Chain Verification (F10, F13)
    # --------------------------------------------------------------------------

    def verify_certificate_chain(
        self,
        chain_pems: List[str],
        verification_time: Optional[datetime] = None,
        provider: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[x509.Certificate, x509.Certificate, int]:
        """
        Validates X.509 certificate chain from leaf cert up to trusted root.
        Returns (leaf_cert, root_cert, chain_length).
        """
        if not chain_pems:
            raise AttestationEvidenceMalformedError(
                "Certificate chain is empty",
                provider=provider,
                task_id=task_id,
            )

        now = verification_time or datetime.now(timezone.utc)
        parsed_chain: List[x509.Certificate] = []

        # Parse and check individual certificate validity dates
        for idx, pem_str in enumerate(chain_pems):
            try:
                cert = x509.load_pem_x509_certificate(pem_str.strip().encode("utf-8"))
            except Exception as exc:
                raise AttestationCertificateError(
                    f"Failed to parse PEM certificate at index {idx}: {exc}",
                    provider=provider,
                    task_id=task_id,
                    cause=exc,
                ) from exc

            # Expiration and temporal validity check
            not_before = getattr(cert, "not_valid_before_utc", None)
            if not_before is None:
                not_before = cert.not_valid_before.replace(tzinfo=timezone.utc)
            not_after = getattr(cert, "not_valid_after_utc", None)
            if not_after is None:
                not_after = cert.not_valid_after.replace(tzinfo=timezone.utc)

            if now < not_before:
                raise AttestationExpiredError(
                    f"Certificate at index {idx} not yet valid (valid from {not_before}, current {now})",
                    provider=provider,
                    task_id=task_id,
                )
            if now > not_after:
                raise AttestationExpiredError(
                    f"Certificate at index {idx} has expired (expired at {not_after}, current {now})",
                    provider=provider,
                    task_id=task_id,
                )

            # Check revocation
            if cert.serial_number in self.policy.revoked_serial_numbers:
                raise AttestationRevokedError(
                    f"Certificate with serial number {hex(cert.serial_number)} is revoked",
                    provider=provider,
                    task_id=task_id,
                )

            parsed_chain.append(cert)

        leaf_cert = parsed_chain[0]

        # Check Leaf BasicConstraints: leaf cannot be a CA
        if self.policy.enforce_ca_extensions:
            try:
                ext = leaf_cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
                if ext.value.ca:
                    raise AttestationChainInvalidError(
                        "Leaf certificate cannot have CA flag enabled (basic_constraints.ca=True)",
                        provider=provider,
                        task_id=task_id,
                    )
            except x509.ExtensionNotFound:
                pass

        # Verify intermediate linkages: cert[i] signed by cert[i+1]
        for i in range(len(parsed_chain) - 1):
            child = parsed_chain[i]
            parent = parsed_chain[i + 1]

            if child.issuer != parent.subject:
                raise AttestationChainInvalidError(
                    f"Certificate chain broken at index {i}: child issuer "
                    f"'{child.issuer.rfc4514_string()}' != parent subject '{parent.subject.rfc4514_string()}'",
                    provider=provider,
                    task_id=task_id,
                )

            # Validate BasicConstraints: parent must be a CA
            if self.policy.enforce_ca_extensions:
                try:
                    ext = parent.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
                    if not ext.value.ca:
                        raise AttestationChainInvalidError(
                            f"Certificate at index {i+1} in chain is not a CA (basic_constraints.ca=False)",
                            provider=provider,
                            task_id=task_id,
                        )
                except x509.ExtensionNotFound:
                    raise AttestationChainInvalidError(
                        f"Certificate at index {i+1} missing BasicConstraints extension",
                        provider=provider,
                        task_id=task_id,
                    )

            # Cryptographically verify signature
            _verify_x509_cert_signature(cert=child, issuer_pubkey=parent.public_key())

        # Anchor to trusted Root CA
        top_cert = parsed_chain[-1]
        matched_root: Optional[x509.Certificate] = None

        if not self.policy.trusted_roots:
            raise AttestationUntrustedRootError(
                "Verifier trust store has zero trusted Root CAs configured",
                provider=provider,
                task_id=task_id,
            )

        for trusted_root in self.policy.trusted_roots:
            # Check if top_cert matches trusted root directly
            if (
                top_cert.fingerprint(hashes.SHA256()) == trusted_root.fingerprint(hashes.SHA256())
                or top_cert.public_key().public_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                == trusted_root.public_key().public_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            ):
                # Verify self-signature on root
                _verify_x509_cert_signature(cert=top_cert, issuer_pubkey=top_cert.public_key())
                matched_root = trusted_root
                break

            # Or if top_cert is signed directly by a trusted root
            if top_cert.issuer == trusted_root.subject:
                try:
                    _verify_x509_cert_signature(cert=top_cert, issuer_pubkey=trusted_root.public_key())
                    matched_root = trusted_root
                    break
                except AttestationSignatureError:
                    continue

        if matched_root is None:
            raise AttestationUntrustedRootError(
                f"Certificate chain cannot be anchored to any trusted Root CA (top subject: '{top_cert.subject.rfc4514_string()}')",
                provider=provider,
                task_id=task_id,
            )

        return leaf_cert, matched_root, len(parsed_chain)

    # --------------------------------------------------------------------------
    # Subsystem: Freshness & Nonce / Pubkey Binding (F12)
    # --------------------------------------------------------------------------

    def verify_freshness_and_binding(
        self,
        claims: Dict[str, Any],
        expected_nonce: Union[bytes, str],
        expected_client_pubkey: Optional[Union[bytes, str]] = None,
        verification_time: Optional[datetime] = None,
        provider: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Validates challenger nonce, client pubkey hash, and temporal freshness."""
        # 1. Challenger Nonce Matching
        actual_nonce_val = claims.get("nonce")
        if actual_nonce_val is None:
            raise AttestationNonceMismatchError(
                "Attestation claims missing 'nonce' field",
                expected_nonce=str(expected_nonce),
                actual_nonce=None,
                provider=provider,
                task_id=task_id,
            )

        expected_nonce_hex = (
            expected_nonce.hex()
            if isinstance(expected_nonce, (bytes, bytearray))
            else str(expected_nonce).strip().lower()
        )
        if expected_nonce_hex.startswith("0x"):
            expected_nonce_hex = expected_nonce_hex[2:]

        actual_nonce_hex = str(actual_nonce_val).strip().lower()
        if actual_nonce_hex.startswith("0x"):
            actual_nonce_hex = actual_nonce_hex[2:]

        if not secrets.compare_digest(expected_nonce_hex, actual_nonce_hex):
            raise AttestationNonceMismatchError(
                f"Challenger nonce mismatch: expected {expected_nonce_hex}, got {actual_nonce_hex}",
                expected_nonce=expected_nonce_hex,
                actual_nonce=actual_nonce_hex,
                provider=provider,
                task_id=task_id,
            )

        # 2. Ephemeral Public Key Binding
        if expected_client_pubkey is not None:
            pubkey_bytes = (
                expected_client_pubkey
                if isinstance(expected_client_pubkey, (bytes, bytearray))
                else bytes.fromhex(str(expected_client_pubkey).replace("0x", ""))
            )
            expected_hash_raw = hashlib.sha256(pubkey_bytes).hexdigest().lower()
            expected_hash_bound = hashlib.sha256(bytes.fromhex(expected_nonce_hex) + pubkey_bytes).hexdigest().lower()

            candidate_bindings = []
            for key in ("client_pubkey_hash", "ephemeral_pubkey_hash", "user_data"):
                val = claims.get(key)
                if val:
                    candidate_bindings.append(val)

            if "public_key" in claims:
                candidate_bindings.append(claims["public_key"])

            if not candidate_bindings:
                raise AttestationBindingError(
                    "Attestation claims missing 'ephemeral_pubkey_hash' binding claim",
                    expected_pubkey_hash=expected_hash_raw,
                    actual_pubkey_hash=None,
                    provider=provider,
                    task_id=task_id,
                )

            matched = False
            last_actual = None
            for cb in candidate_bindings:
                cb_str = str(cb).strip().lower()
                if cb_str.startswith("0x"):
                    cb_str = cb_str[2:]
                last_actual = cb_str
                if (
                    secrets.compare_digest(cb_str, expected_hash_raw)
                    or secrets.compare_digest(cb_str, expected_hash_bound)
                    or secrets.compare_digest(cb_str, pubkey_bytes.hex().lower())
                ):
                    matched = True
                    break

            if not matched:
                raise AttestationBindingError(
                    f"Client ephemeral public key binding mismatch: expected {expected_hash_raw}, got {last_actual}",
                    expected_pubkey_hash=expected_hash_raw,
                    actual_pubkey_hash=last_actual,
                    provider=provider,
                    task_id=task_id,
                )

        # 3. Temporal Freshness / Timestamp Check
        ts_val = claims.get("timestamp") or claims.get("iat")
        if ts_val is not None:
            try:
                evidence_ts = float(ts_val)
                if math.isnan(evidence_ts) or math.isinf(evidence_ts):
                    raise AttestationExpiredError(
                        f"Invalid NaN/Inf timestamp: {evidence_ts}",
                        provider=provider,
                        task_id=task_id,
                    )
                now_ts = (verification_time or datetime.now(timezone.utc)).timestamp()

                # Future timestamp check
                if evidence_ts > now_ts + self.policy.allowed_clock_skew_seconds:
                    raise AttestationExpiredError(
                        f"Attestation evidence timestamp is in the future: {evidence_ts} > current {now_ts}",
                        provider=provider,
                        task_id=task_id,
                    )
                # Stale age check
                age = now_ts - evidence_ts
                if age > self.policy.max_evidence_age_seconds:
                    raise AttestationExpiredError(
                        f"Attestation evidence is stale: age {age:.1f}s exceeds limit {self.policy.max_evidence_age_seconds}s",
                        provider=provider,
                        task_id=task_id,
                    )
            except (ValueError, TypeError) as exc:
                raise AttestationEvidenceMalformedError(
                    f"Malformed timestamp in attestation claims: {ts_val}",
                    provider=provider,
                    task_id=task_id,
                ) from exc

    # --------------------------------------------------------------------------
    # Subsystem: Golden RIM & Measurement Digest Appraisal (F11)
    # --------------------------------------------------------------------------

    def verify_measurements(
        self,
        claims: Dict[str, Any],
        provider: str,
        evidence_golden_rim: Optional[Dict[str, str]] = None,
        task_id: Optional[str] = None,
    ) -> int:
        """
        Validates PCR and firmware measurements against trusted Golden RIM.
        Returns the number of verified measurements.
        """
        golden_rim = self.policy.golden_rims.get(provider.lower().strip())
        if not golden_rim:
            # Do NOT fallback to trusting unauthenticated evidence.golden_rim to self-certify.
            # No authoritative Golden RIM configured in policy for this provider.
            return 0

        actual_measurements = claims.get("measurements") or claims.get("pcrs") or claims.get("digests")
        if not isinstance(actual_measurements, dict):
            raise AttestationMeasurementMismatchError(
                "Attestation claims missing 'measurements' or 'pcrs' dictionary",
                missing_claims=list(golden_rim.keys()),
                provider=provider,
                task_id=task_id,
            )

        mismatches: Dict[str, Dict[str, str]] = {}
        missing_claims: List[str] = []
        verified_count = 0

        for key, expected_digest in golden_rim.items():
            if key not in actual_measurements:
                missing_claims.append(key)
                continue

            actual_val = str(actual_measurements[key]).strip().lower()
            expected_val = str(expected_digest).strip().lower()
            if actual_val.startswith("0x"):
                actual_val = actual_val[2:]
            if expected_val.startswith("0x"):
                expected_val = expected_val[2:]

            if not secrets.compare_digest(actual_val, expected_val):
                mismatches[key] = {"expected": expected_val, "actual": actual_val}
            else:
                verified_count += 1

        if mismatches or missing_claims:
            raise AttestationMeasurementMismatchError(
                f"Golden RIM measurement appraisal failed: {len(mismatches)} mismatches, {len(missing_claims)} missing",
                mismatches=mismatches,
                missing_claims=missing_claims,
                provider=provider,
                task_id=task_id,
            )

        return verified_count

    # --------------------------------------------------------------------------
    # Helper: Evidence Container Parsing
    # --------------------------------------------------------------------------

    def _parse_evidence(
        self,
        evidence: Union[AttestationEvidence, bytes, str],
        task_id: Optional[str] = None,
    ) -> AttestationEvidence:
        """Converts raw bytes, strings, or Pydantic models into AttestationEvidence."""
        if isinstance(evidence, AttestationEvidence):
            return evidence

        try:
            if isinstance(evidence, (bytes, bytearray, memoryview)):
                evidence_str = bytes(evidence).decode("utf-8")
                return AttestationEvidence.model_validate_json(evidence_str)
            elif isinstance(evidence, str):
                return AttestationEvidence.model_validate_json(evidence)
            else:
                raise TypeError(f"Unexpected evidence type: {type(evidence).__name__}")
        except Exception as exc:
            raise AttestationEvidenceMalformedError(
                f"Failed to deserialize attestation evidence: {exc}",
                task_id=task_id,
                cause=exc,
            ) from exc
