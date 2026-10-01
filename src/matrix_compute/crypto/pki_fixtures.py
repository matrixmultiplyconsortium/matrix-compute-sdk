"""
Hermetic Synthetic PKI and Attestation Fixtures.
Provides synthetic X.509 Root CA, Intermediate CA, and Leaf certificate hierarchies,
golden Reference Integrity Measurements (RIM), and valid/adversarial AttestationEvidence
fixtures for 100% offline hermetic testing of Confidential Computing attestation.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from matrix_contracts.schemas.attestation import AttestationEvidence

# ============================================================================
# Golden RIM and Measurement Fixture Constants
# ============================================================================

DEFAULT_GOLDEN_RIM: Dict[str, str] = {
    "driver": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
    "vbios": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "firmware": "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
    "uefi": "3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d",
    "cc_mode": "2e7d2c03a9507ae265ecf5b5356885a53393a2029d24139499726b4259fb79c9",
}

DEFAULT_MEASUREMENTS: Dict[str, str] = {
    "driver": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
    "vbios": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "firmware": "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
    "uefi": "3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d",
    "cc_mode": "2e7d2c03a9507ae265ecf5b5356885a53393a2029d24139499726b4259fb79c9",
}

TAMPERED_MEASUREMENTS: Dict[str, str] = {
    "driver": "deadbeef000000000000000000000000000000000000000000000000deadbeef",
    "vbios": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "firmware": "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
    "uefi": "3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d",
    "cc_mode": "2e7d2c03a9507ae265ecf5b5356885a53393a2029d24139499726b4259fb79c9",
}

# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class CertBundle:
    cert: x509.Certificate
    key: Any
    cert_pem: str
    key_pem: str

    @classmethod
    def from_cert_and_key(cls, cert: x509.Certificate, key: Any) -> "CertBundle":
        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode("ascii")
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("ascii")
        return cls(cert=cert, key=key, cert_pem=cert_pem, key_pem=key_pem)


@dataclass
class SyntheticPKI:
    root: CertBundle
    intermediate: CertBundle
    leaf: CertBundle

    @property
    def root_cert_pem(self) -> str:
        return self.root.cert_pem

    @property
    def intermediate_cert_pem(self) -> str:
        return self.intermediate.cert_pem

    @property
    def leaf_cert_pem(self) -> str:
        return self.leaf.cert_pem

    @property
    def certificate_chain(self) -> List[str]:
        """Ordered from leaf to root: [Leaf, Intermediate, Root]."""
        return [self.leaf.cert_pem, self.intermediate.cert_pem, self.root.cert_pem]


# ============================================================================
# Key and Signature Utilities
# ============================================================================

def generate_key(key_type: str = "ecdsa", curve: str = "secp384r1", rsa_bits: int = 2048) -> Any:
    """Generate a private key of specified algorithm."""
    kt = key_type.lower()
    if kt == "ecdsa":
        cv = curve.lower()
        if cv == "secp384r1":
            return ec.generate_private_key(ec.SECP384R1())
        elif cv == "secp256r1":
            return ec.generate_private_key(ec.SECP256R1())
        else:
            raise ValueError(f"Unsupported EC curve: {curve}")
    elif kt == "ed25519":
        return ed25519.Ed25519PrivateKey.generate()
    elif kt == "rsa":
        return rsa.generate_private_key(public_exponent=65537, key_size=rsa_bits)
    else:
        raise ValueError(f"Unsupported key_type: {key_type}")


def canonicalize_claims(claims: Dict[str, Any]) -> bytes:
    """Canonicalize token claims dictionary to UTF-8 bytes for deterministic signing."""
    return json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_claims(claims: Dict[str, Any], signing_key: Any) -> bytes:
    """Sign canonicalized token claims with private key."""
    payload = canonicalize_claims(claims)
    if isinstance(signing_key, ec.EllipticCurvePrivateKey):
        hash_algo = hashes.SHA384() if isinstance(signing_key.curve, ec.SECP384R1) else hashes.SHA256()
        return signing_key.sign(payload, ec.ECDSA(hash_algo))
    elif isinstance(signing_key, ed25519.Ed25519PrivateKey):
        return signing_key.sign(payload)
    elif isinstance(signing_key, rsa.RSAPrivateKey):
        return signing_key.sign(payload, padding.PKCS1v15(), hashes.SHA256())
    else:
        raise TypeError(f"Unsupported signing key type: {type(signing_key)}")


def verify_claims_signature(claims: Dict[str, Any], signature: bytes, public_key: Any) -> bool:
    """Verify claims signature against public key."""
    payload = canonicalize_claims(claims)
    try:
        if isinstance(public_key, ec.EllipticCurvePublicKey):
            hash_algo = hashes.SHA384() if isinstance(public_key.curve, ec.SECP384R1) else hashes.SHA256()
            public_key.verify(signature, payload, ec.ECDSA(hash_algo))
            return True
        elif isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(signature, payload)
            return True
        elif isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(signature, payload, padding.PKCS1v15(), hashes.SHA256())
            return True
        else:
            return False
    except Exception:
        return False


# ============================================================================
# Synthetic PKI Hierarchy Generator
# ============================================================================

def generate_synthetic_pki(
    key_type: str = "ecdsa",
    curve: str = "secp384r1",
    valid_days: int = 365,
    expired: bool = False,
    not_yet_valid: bool = False,
    root_cn: str = "NVIDIA Confidential Computing Root CA",
    intermediate_cn: str = "NVIDIA Hopper Attestation Intermediate CA",
    leaf_cn: str = "NVIDIA H100 Device Attestation",
) -> SyntheticPKI:
    """
    Generates a full 3-tier synthetic X.509 PKI hierarchy (Root CA, Intermediate CA, Leaf).
    Supports valid, expired, and not-yet-valid temporal configurations.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    if expired:
        not_before = now - datetime.timedelta(days=valid_days + 10)
        not_after = now - datetime.timedelta(days=1)
    elif not_yet_valid:
        not_before = now + datetime.timedelta(days=1)
        not_after = now + datetime.timedelta(days=valid_days)
    else:
        not_before = now - datetime.timedelta(days=1)
        not_after = now + datetime.timedelta(days=valid_days)

    # 1. Root CA Key & Self-Signed Cert
    root_key = generate_key(key_type, curve)
    root_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NVIDIA Corporation"),
        x509.NameAttribute(NameOID.COMMON_NAME, root_cn),
    ])
    root_cert_builder = (
        x509.CertificateBuilder()
        .subject_name(root_subject)
        .issuer_name(root_subject)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=2), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
    )
    sig_hash = None if key_type == "ed25519" else (hashes.SHA384() if curve == "secp384r1" else hashes.SHA256())
    root_cert = root_cert_builder.sign(root_key, sig_hash)
    root_bundle = CertBundle.from_cert_and_key(root_cert, root_key)

    # 2. Intermediate CA Key & Cert Signed by Root CA
    inter_key = generate_key(key_type, curve)
    inter_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NVIDIA Corporation"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Confidential Computing"),
        x509.NameAttribute(NameOID.COMMON_NAME, intermediate_cn),
    ])
    inter_cert_builder = (
        x509.CertificateBuilder()
        .subject_name(inter_subject)
        .issuer_name(root_subject)
        .public_key(inter_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(inter_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
    )
    inter_cert = inter_cert_builder.sign(root_key, sig_hash)
    inter_bundle = CertBundle.from_cert_and_key(inter_cert, inter_key)

    # 3. Leaf Key & Cert Signed by Intermediate CA
    leaf_key = generate_key(key_type, curve)
    leaf_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NVIDIA Corporation"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Hardware Security"),
        x509.NameAttribute(NameOID.COMMON_NAME, leaf_cn),
        x509.NameAttribute(NameOID.SERIAL_NUMBER, "GPU-H100-MOCK-998877"),
    ])
    leaf_cert_builder = (
        x509.CertificateBuilder()
        .subject_name(leaf_subject)
        .issuer_name(inter_subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=True,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(inter_key.public_key()), critical=False)
    )
    leaf_cert = leaf_cert_builder.sign(inter_key, sig_hash)
    leaf_bundle = CertBundle.from_cert_and_key(leaf_cert, leaf_key)

    return SyntheticPKI(root=root_bundle, intermediate=inter_bundle, leaf=leaf_bundle)


_DEFAULT_PKI_CACHE: Optional[SyntheticPKI] = None


def get_default_pki() -> SyntheticPKI:
    """Returns cached singleton SyntheticPKI to optimize test execution speed."""
    global _DEFAULT_PKI_CACHE
    if _DEFAULT_PKI_CACHE is None:
        _DEFAULT_PKI_CACHE = generate_synthetic_pki()
    return _DEFAULT_PKI_CACHE


# ============================================================================
# Valid Attestation Evidence Generator
# ============================================================================

def create_valid_attestation_evidence(
    pki: Optional[SyntheticPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    measurements: Optional[Dict[str, str]] = None,
    golden_rim: Optional[Dict[str, str]] = None,
    provider: str = "simulated-cca",
    device_id: str = "GPU-H100-MOCK-001",
    timestamp: Optional[int] = None,
) -> AttestationEvidence:
    """Generates a valid, cryptographically signed AttestationEvidence model."""
    active_pki = pki or get_default_pki()

    if nonce is None:
        nonce_hex = secrets.token_bytes(32).hex()
    elif isinstance(nonce, (bytes, bytearray)):
        nonce_hex = bytes(nonce).hex()
    else:
        nonce_hex = str(nonce)

    ts = timestamp if timestamp is not None else int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    meas = measurements if measurements is not None else DEFAULT_MEASUREMENTS.copy()
    rim = golden_rim if golden_rim is not None else DEFAULT_GOLDEN_RIM.copy()

    token_claims: Dict[str, Any] = {
        "version": "1.0",
        "provider": provider,
        "device_id": device_id,
        "nonce": nonce_hex,
        "timestamp": ts,
        "measurements": meas,
    }

    if worker_ephemeral_pubkey is not None:
        token_claims["ephemeral_pubkey_hash"] = hashlib.sha256(worker_ephemeral_pubkey).hexdigest()

    if client_ephemeral_pubkey is not None:
        token_claims["client_pubkey_hash"] = hashlib.sha256(client_ephemeral_pubkey).hexdigest()
        # Also include client pubkey in user_data or claims if ephemeral_pubkey_hash is expected
        if "ephemeral_pubkey_hash" not in token_claims:
            token_claims["ephemeral_pubkey_hash"] = hashlib.sha256(client_ephemeral_pubkey).hexdigest()

    # User data binding calculation (SHA-256 of nonce + worker_pubkey)
    user_data = None
    if worker_ephemeral_pubkey is not None:
        user_data = hashlib.sha256(bytes.fromhex(nonce_hex) + worker_ephemeral_pubkey).digest()

    token_sig = sign_claims(token_claims, active_pki.leaf.key)

    return AttestationEvidence(
        provider=provider,
        certificate_chain=active_pki.certificate_chain,
        token_claims=token_claims,
        token_signature=token_sig,
        golden_rim=rim,
        user_data=user_data,
    )


# ============================================================================
# Adversarial & Tampered Negative Test Fixture Generators
# ============================================================================

def create_expired_cert_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Certificate chain contains expired certificates."""
    expired_pki = generate_synthetic_pki(expired=True)
    return create_valid_attestation_evidence(
        pki=expired_pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )


def create_not_yet_valid_cert_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Certificate chain contains not-yet-valid certificates."""
    future_pki = generate_synthetic_pki(not_yet_valid=True)
    return create_valid_attestation_evidence(
        pki=future_pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )


def create_untrusted_root_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> Tuple[AttestationEvidence, str]:
    """
    Negative Fixture: Certificate chain issued by an untrusted/rogue Root CA.
    Returns (evidence, rogue_root_cert_pem).
    """
    rogue_pki = generate_synthetic_pki(root_cn="Rogue Untrusted Root CA")
    evidence = create_valid_attestation_evidence(
        pki=rogue_pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
    return evidence, rogue_pki.root_cert_pem


def create_broken_chain_signature_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """
    Negative Fixture: Certificate chain signature validation fails.
    Leaf certificate was signed by a different intermediate key.
    """
    pki1 = generate_synthetic_pki()
    pki2 = generate_synthetic_pki()
    # Chain with leaf from pki2 and intermediate from pki1
    tampered_chain = [pki2.leaf_cert_pem, pki1.intermediate_cert_pem, pki1.root_cert_pem]
    evidence = create_valid_attestation_evidence(
        pki=pki2,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
    evidence.certificate_chain = tampered_chain
    return evidence


def create_tampered_token_signature_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: token_signature bytes corrupted."""
    ev = create_valid_attestation_evidence(
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
    sig = bytearray(ev.token_signature)
    sig[0] ^= 0xFF
    sig[1] ^= 0xFF
    ev.token_signature = bytes(sig)
    return ev


def create_altered_nonce_evidence(
    requested_nonce: bytes,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: token_claims signed with different nonce (replay attack simulation)."""
    stale_nonce = secrets.token_bytes(32)
    while stale_nonce == requested_nonce:
        stale_nonce = secrets.token_bytes(32)
    return create_valid_attestation_evidence(
        nonce=stale_nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )


def create_mismatched_measurements_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    tampered_component: str = "driver",
) -> AttestationEvidence:
    """Negative Fixture: Measured digest does not match Golden RIM."""
    meas = DEFAULT_MEASUREMENTS.copy()
    meas[tampered_component] = TAMPERED_MEASUREMENTS[tampered_component]
    return create_valid_attestation_evidence(
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        measurements=meas,
        golden_rim=DEFAULT_GOLDEN_RIM.copy(),
    )


def create_altered_pubkey_hash_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Bound ephemeral pubkey hash altered (key substitution simulation)."""
    fake_pubkey = secrets.token_bytes(32)
    return create_valid_attestation_evidence(
        nonce=nonce,
        worker_ephemeral_pubkey=fake_pubkey,
        client_ephemeral_pubkey=fake_pubkey,
    )


def create_malformed_pem_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Damaged/corrupted PEM certificate string."""
    ev = create_valid_attestation_evidence(
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
    ev.certificate_chain = [
        "-----BEGIN CERTIFICATE-----\nMALFORMED_INVALID_BASE64_DATA\n-----END CERTIFICATE-----",
        ev.certificate_chain[1],
        ev.certificate_chain[2],
    ]
    return ev


def create_leaf_as_ca_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Leaf certificate has CA:TRUE BasicConstraint violation."""
    now = datetime.datetime.now(datetime.timezone.utc)
    root_key = generate_key("ecdsa", "secp384r1")
    root_sub = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Root CA")])
    root_cert = (
        x509.CertificateBuilder()
        .subject_name(root_sub)
        .issuer_name(root_sub)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(root_key, hashes.SHA384())
    )
    bad_leaf_key = generate_key("ecdsa", "secp384r1")
    bad_leaf_sub = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Bad Leaf CA")])
    bad_leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(bad_leaf_sub)
        .issuer_name(root_sub)
        .public_key(bad_leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)  # VIOLATION: Leaf cannot be CA
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(root_key, hashes.SHA384())
    )
    leaf_bundle = CertBundle.from_cert_and_key(bad_leaf_cert, bad_leaf_key)
    root_bundle = CertBundle.from_cert_and_key(root_cert, root_key)
    pki = SyntheticPKI(root=root_bundle, intermediate=root_bundle, leaf=leaf_bundle)
    return create_valid_attestation_evidence(
        pki=pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
