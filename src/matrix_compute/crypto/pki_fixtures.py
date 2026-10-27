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
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature
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


# ============================================================================
# AMD SEV-SNP OID Constants & DER Utilities
# ============================================================================

AMD_OID_PREFIX = "1.3.6.1.4.1.3704.1"
AMD_OID_STRUCT_VERSION = f"{AMD_OID_PREFIX}.1"       # INTEGER
AMD_OID_PRODUCT_NAME = f"{AMD_OID_PREFIX}.2"         # IA5String
AMD_OID_BL_SPL = f"{AMD_OID_PREFIX}.3.1"             # INTEGER
AMD_OID_TEE_SPL = f"{AMD_OID_PREFIX}.3.2"            # INTEGER
AMD_OID_SNP_SPL = f"{AMD_OID_PREFIX}.3.3"            # INTEGER
AMD_OID_UCODE_SPL = f"{AMD_OID_PREFIX}.3.8"          # INTEGER
AMD_OID_HW_ID = f"{AMD_OID_PREFIX}.4"                # OCTET STRING (64 bytes)

# Default synthetic measurements & defaults for AMD SEV-SNP
DEFAULT_SNP_MEASUREMENT: bytes = hashlib.sha384(b"matrix-compute-snp-measurement-v1").digest()
DEFAULT_SNP_HOST_DATA: bytes = hashlib.sha256(b"matrix-compute-snp-host-data-v1").digest()
DEFAULT_SNP_CHIP_ID: bytes = hashlib.sha512(b"amd-epyc-milan-hwid-synthetic-001").digest()
DEFAULT_SNP_POLICY: int = 0x30000  # SMT enabled, ABI 0.0, debug 0
DEFAULT_SNP_GOLDEN_RIM: Dict[str, str] = {
    "measurement": DEFAULT_SNP_MEASUREMENT.hex(),
    "host_data": DEFAULT_SNP_HOST_DATA.hex(),
}


def der_encode_integer(val: int) -> bytes:
    """DER encodes an integer (Tag 0x02)."""
    needed = (val.bit_length() + 8) // 8 if val > 0 else 1
    b = val.to_bytes(needed, byteorder="big", signed=True)
    if len(b) < 128:
        length_bytes = bytes([len(b)])
    else:
        length_bytes = b"\x81" + bytes([len(b)])
    return b"\x02" + length_bytes + b


def der_decode_integer(raw: bytes) -> int:
    """DER decodes an integer (Tag 0x02)."""
    if len(raw) < 2 or raw[0] != 0x02:
        raise ValueError(f"Invalid DER integer tag: {raw[:2]!r}")
    length = raw[1]
    offset = 2
    if length & 0x80:
        num_len_bytes = length & 0x7F
        length = int.from_bytes(raw[offset : offset + num_len_bytes], "big")
        offset += num_len_bytes
    return int.from_bytes(raw[offset : offset + length], byteorder="big", signed=True)


def der_encode_ia5string(val: str) -> bytes:
    """DER encodes an IA5String (Tag 0x16)."""
    b = val.encode("ascii")
    if len(b) < 128:
        length_bytes = bytes([len(b)])
    else:
        length_bytes = b"\x81" + bytes([len(b)])
    return b"\x16" + length_bytes + b


def der_decode_ia5string(raw: bytes) -> str:
    """DER decodes an IA5String (Tag 0x16 or 0x13)."""
    if len(raw) < 2 or raw[0] not in (0x16, 0x13):
        raise ValueError(f"Invalid DER string tag: {raw[:2]!r}")
    length = raw[1]
    offset = 2
    if length & 0x80:
        num_len_bytes = length & 0x7F
        length = int.from_bytes(raw[offset : offset + num_len_bytes], "big")
        offset += num_len_bytes
    return raw[offset : offset + length].decode("ascii")


def der_encode_octet_string(val: bytes) -> bytes:
    """DER encodes an OCTET STRING (Tag 0x04)."""
    if len(val) < 128:
        length_bytes = bytes([len(val)])
    else:
        length_bytes = b"\x81" + bytes([len(val)])
    return b"\x04" + length_bytes + val


def der_decode_octet_string(raw: bytes) -> bytes:
    """DER decodes an OCTET STRING (Tag 0x04)."""
    if len(raw) < 2 or raw[0] != 0x04:
        raise ValueError(f"Invalid DER octet string tag: {raw[:2]!r}")
    length = raw[1]
    offset = 2
    if length & 0x80:
        num_len_bytes = length & 0x7F
        length = int.from_bytes(raw[offset : offset + num_len_bytes], "big")
        offset += num_len_bytes
    return raw[offset : offset + length]


# ============================================================================
# Synthetic AMD PKI Hierarchy & Fixtures
# ============================================================================

@dataclass
class SyntheticAmdPKI:
    """
    Synthetic 3-tier AMD SEV-SNP PKI hierarchy:
    ARK (Root CA, RSA-4096) -> ASK (Intermediate CA, RSA-4096) -> VCEK (Leaf, ECDSA P-384).
    """
    ark_cert: x509.Certificate
    ask_cert: x509.Certificate
    vcek_cert: x509.Certificate
    ark_key: rsa.RSAPrivateKey
    ask_key: rsa.RSAPrivateKey
    vcek_key: ec.EllipticCurvePrivateKey
    vcek_priv_ec: ec.EllipticCurvePrivateKey = None  # type: ignore[assignment]
    chip_id: bytes = b"\x00" * 64
    tcb_spls: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vcek_priv_ec is None and self.vcek_key is not None:
            self.vcek_priv_ec = self.vcek_key
        elif self.vcek_key is None and self.vcek_priv_ec is not None:
            self.vcek_key = self.vcek_priv_ec

    @property
    def ark_cert_pem(self) -> str:
        return self.ark_cert.public_bytes(serialization.Encoding.PEM).decode("ascii")

    @property
    def ask_cert_pem(self) -> str:
        return self.ask_cert.public_bytes(serialization.Encoding.PEM).decode("ascii")

    @property
    def vcek_cert_pem(self) -> str:
        return self.vcek_cert.public_bytes(serialization.Encoding.PEM).decode("ascii")

    @property
    def certificate_chain(self) -> List[str]:
        """Ordered from leaf to root: [VCEK, ASK, ARK]."""
        return [self.vcek_cert_pem, self.ask_cert_pem, self.ark_cert_pem]

    @property
    def root_cert_pem(self) -> str:
        return self.ark_cert_pem

    @property
    def intermediate_cert_pem(self) -> str:
        return self.ask_cert_pem

    @property
    def leaf_cert_pem(self) -> str:
        return self.vcek_cert_pem

    @property
    def ark(self) -> CertBundle:
        return CertBundle.from_cert_and_key(self.ark_cert, self.ark_key)

    @property
    def ask(self) -> CertBundle:
        return CertBundle.from_cert_and_key(self.ask_cert, self.ask_key)

    @property
    def vcek(self) -> CertBundle:
        return CertBundle.from_cert_and_key(self.vcek_cert, self.vcek_key)


def generate_synthetic_amd_pki(
    valid_days: int = 365,
    expired: bool = False,
    not_yet_valid: bool = False,
    product_name: str = "Milan",
    chip_id: Optional[bytes] = None,
    bl_spl: int = 3,
    tee_spl: int = 0,
    snp_spl: int = 8,
    ucode_spl: int = 115,
    ark_cn: Optional[str] = None,
    ask_cn: Optional[str] = None,
    vcek_cn: str = "SEV-VCEK",
    rsa_bits: int = 4096,
) -> SyntheticAmdPKI:
    """
    Generates a full 3-tier synthetic AMD SEV-SNP PKI hierarchy:
    1. ARK (AMD Root Key): Self-signed RSA-4096 CA certificate (path_length=2).
    2. ASK (AMD Signing Key): RSA-4096 intermediate CA certificate signed by ARK (path_length=0).
    3. VCEK (Versioned Chip Endorsement Key): ECDSA NIST P-384 leaf certificate signed by ASK
       using RSA-4096 with sha384WithRSAEncryption, containing custom AMD extensions under OID 1.3.6.1.4.1.3704.1.
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

    actual_chip_id = chip_id if chip_id is not None else DEFAULT_SNP_CHIP_ID
    if len(actual_chip_id) != 64:
        raise ValueError(f"chip_id must be exactly 64 bytes, got {len(actual_chip_id)}")

    actual_ark_cn = ark_cn or f"ARK-{product_name}"
    actual_ask_cn = ask_cn or f"SEV-{product_name}"

    # 1. ARK: RSA-4096 Self-Signed Root CA
    ark_key = rsa.generate_private_key(public_exponent=65537, key_size=rsa_bits)
    ark_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Advanced Micro Devices"),
        x509.NameAttribute(NameOID.COMMON_NAME, actual_ark_cn),
    ])
    ark_cert = (
        x509.CertificateBuilder()
        .subject_name(ark_subject)
        .issuer_name(ark_subject)
        .public_key(ark_key.public_key())
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
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ark_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ark_key.public_key()), critical=False)
        .sign(ark_key, hashes.SHA384())
    )

    # 2. ASK: RSA-4096 Intermediate CA signed by ARK
    ask_key = rsa.generate_private_key(public_exponent=65537, key_size=rsa_bits)
    ask_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Advanced Micro Devices"),
        x509.NameAttribute(NameOID.COMMON_NAME, actual_ask_cn),
    ])
    ask_cert = (
        x509.CertificateBuilder()
        .subject_name(ask_subject)
        .issuer_name(ark_subject)
        .public_key(ask_key.public_key())
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
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ask_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ark_key.public_key()), critical=False)
        .sign(ark_key, hashes.SHA384())
    )

    # 3. VCEK: ECDSA P-384 Leaf Certificate signed by ASK (using RSA-4096 sha384WithRSAEncryption)
    vcek_key = ec.generate_private_key(ec.SECP384R1())
    vcek_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Advanced Micro Devices"),
        x509.NameAttribute(NameOID.COMMON_NAME, vcek_cn),
    ])
    vcek_builder = (
        x509.CertificateBuilder()
        .subject_name(vcek_subject)
        .issuer_name(ask_subject)
        .public_key(vcek_key.public_key())
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
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(vcek_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ask_key.public_key()), critical=False)
        # Custom AMD extensions under 1.3.6.1.4.1.3704.1
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier(AMD_OID_STRUCT_VERSION),
                der_encode_integer(1),
            ),
            critical=False,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier(AMD_OID_PRODUCT_NAME),
                der_encode_ia5string(product_name),
            ),
            critical=False,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier(AMD_OID_BL_SPL),
                der_encode_integer(bl_spl),
            ),
            critical=False,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier(AMD_OID_TEE_SPL),
                der_encode_integer(tee_spl),
            ),
            critical=False,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier(AMD_OID_SNP_SPL),
                der_encode_integer(snp_spl),
            ),
            critical=False,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier(AMD_OID_UCODE_SPL),
                der_encode_integer(ucode_spl),
            ),
            critical=False,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier(AMD_OID_HW_ID),
                der_encode_octet_string(actual_chip_id),
            ),
            critical=False,
        )
    )
    vcek_cert = vcek_builder.sign(ask_key, hashes.SHA384())

    tcb_spls = {
        "blSPL": bl_spl,
        "teeSPL": tee_spl,
        "snpSPL": snp_spl,
        "ucodeSPL": ucode_spl,
    }

    return SyntheticAmdPKI(
        ark_cert=ark_cert,
        ask_cert=ask_cert,
        vcek_cert=vcek_cert,
        ark_key=ark_key,
        ask_key=ask_key,
        vcek_key=vcek_key,
        vcek_priv_ec=vcek_key,
        chip_id=actual_chip_id,
        tcb_spls=tcb_spls,
    )


_DEFAULT_AMD_PKI_CACHE: Optional[SyntheticAmdPKI] = None


def get_default_amd_pki() -> SyntheticAmdPKI:
    """Returns cached singleton SyntheticAmdPKI to optimize test execution speed (<1s)."""
    global _DEFAULT_AMD_PKI_CACHE
    if _DEFAULT_AMD_PKI_CACHE is None:
        _DEFAULT_AMD_PKI_CACHE = generate_synthetic_amd_pki()
    return _DEFAULT_AMD_PKI_CACHE


def generate_synthetic_snp_report(
    vcek_priv_ec: ec.EllipticCurvePrivateKey,
    report_data: Optional[bytes] = None,
    measurement: Optional[bytes] = None,
    host_data: Optional[bytes] = None,
    chip_id: Optional[bytes] = None,
    policy: int = DEFAULT_SNP_POLICY,
    version: int = 2,
    guest_svn: int = 1,
    family_id: Optional[bytes] = None,
    image_id: Optional[bytes] = None,
    vmpl: int = 0,
    signature_algo: int = 1,
    platform_version: Optional[bytes] = None,
    platform_info: int = 0,
    flags: int = 0,
    reported_tcb: Optional[bytes] = None,
    committed_tcb: Optional[bytes] = None,
    current_build: int = 0,
    committed_build: int = 0,
    id_key_digest: Optional[bytes] = None,
    author_key_digest: Optional[bytes] = None,
    report_id: Optional[bytes] = None,
    report_id_ma: Optional[bytes] = None,
    raw_r: Optional[bytes] = None,
    raw_s: Optional[bytes] = None,
    tamper_byte_offset: Optional[int] = None,
    tamper_bit_mask: int = 0x01,
) -> bytes:
    """
    Generates a cryptographically authentic 1184-byte binary AMD SEV-SNP attestation report
    per AMD Publication #56860:
    - Signed body: bytes 0x000..0x29F (672 bytes)
    - Signature algorithm: 1 (ECDSA P-384 with SHA-384)
    - Offsets:
      - 0x000: version (uint32 LE)
      - 0x004: guest_svn (uint32 LE)
      - 0x008: policy (uint64 LE, bit 19 DEBUG == 0)
      - 0x010: family_id (16B)
      - 0x020: image_id (16B)
      - 0x030: vmpl (uint32 LE)
      - 0x034: signature_algo (uint32 LE, default 1)
      - 0x038: platform_version (8B TCB)
      - 0x040: platform_info (uint64 LE)
      - 0x048: flags (uint32 LE)
      - 0x04C: reserved0 (4B)
      - 0x050: report_data (64B: 32B nonce + 32B pubkey hash)
      - 0x090: measurement (48B SHA-384 digest)
      - 0x0C0: host_data (32B)
      - 0x0E0: id_key_digest (48B)
      - 0x110: author_key_digest (48B)
      - 0x140: report_id (32B)
      - 0x160: report_id_ma (32B)
      - 0x180: reported_tcb (8B TCB)
      - 0x188: reserved1 (24B)
      - 0x1A0: chip_id (64B)
      - 0x1E0: committed_tcb (8B TCB)
      - 0x1E8: current_build (uint32 LE)
      - 0x1EC: committed_build (uint32 LE)
      - 0x1F0: reserved2 (176B)
      - 0x2A0: signature_r (72B LE scalar)
      - 0x2E8: signature_s (72B LE scalar)
      - 0x330: signature_pad (368B zeroes)
    """
    report = bytearray(1184)

    # Headers and configuration
    struct.pack_into("<I", report, 0x000, version)
    struct.pack_into("<I", report, 0x004, guest_svn)
    struct.pack_into("<Q", report, 0x008, policy)

    if family_id:
        report[0x010 : 0x010 + min(16, len(family_id))] = family_id[:16]
    if image_id:
        report[0x020 : 0x020 + min(16, len(image_id))] = image_id[:16]

    struct.pack_into("<I", report, 0x030, vmpl)
    struct.pack_into("<I", report, 0x034, signature_algo)

    # Default TCB: bootloader=3, tee=0, snp=8, microcode=115
    default_tcb = bytes([3, 0, 8, 0, 0, 0, 0, 115])
    pv = platform_version or reported_tcb or default_tcb
    report[0x038 : 0x040] = pv[:8]

    struct.pack_into("<Q", report, 0x040, platform_info)
    struct.pack_into("<I", report, 0x048, flags)

    # 0x050: report_data (64B)
    rd = report_data if report_data is not None else b"\x00" * 64
    report[0x050 : 0x050 + min(64, len(rd))] = rd[:64]

    # 0x090: measurement (48B)
    meas = measurement if measurement is not None else DEFAULT_SNP_MEASUREMENT
    report[0x090 : 0x090 + min(48, len(meas))] = meas[:48]

    # 0x0C0: host_data (32B)
    hd = host_data if host_data is not None else DEFAULT_SNP_HOST_DATA
    report[0x0C0 : 0x0C0 + min(32, len(hd))] = hd[:32]

    if id_key_digest:
        report[0x0E0 : 0x0E0 + min(48, len(id_key_digest))] = id_key_digest[:48]
    if author_key_digest:
        report[0x110 : 0x110 + min(48, len(author_key_digest))] = author_key_digest[:48]

    rid = report_id if report_id is not None else secrets.token_bytes(32)
    report[0x140 : 0x160] = rid[:32]

    if report_id_ma:
        report[0x160 : 0x160 + min(32, len(report_id_ma))] = report_id_ma[:32]

    rtcb = reported_tcb or default_tcb
    report[0x180 : 0x188] = rtcb[:8]

    cid = chip_id if chip_id is not None else DEFAULT_SNP_CHIP_ID
    report[0x1A0 : 0x1A0 + min(64, len(cid))] = cid[:64]

    ctcb = committed_tcb or rtcb
    report[0x1E0 : 0x1E8] = ctcb[:8]

    struct.pack_into("<I", report, 0x1E8, current_build)
    struct.pack_into("<I", report, 0x1EC, committed_build)

    # Cryptographic signature over bytes 0x000..0x29F (672 bytes)
    if raw_r is not None and raw_s is not None:
        report[0x2A0 : 0x2A0 + len(raw_r)] = raw_r
        report[0x2E8 : 0x2E8 + len(raw_s)] = raw_s
    else:
        signed_body = bytes(report[:672])
        der_sig = vcek_priv_ec.sign(signed_body, ec.ECDSA(hashes.SHA384()))
        r_int, s_int = decode_dss_signature(der_sig)
        report[0x2A0 : 0x2E8] = r_int.to_bytes(72, byteorder="little")
        report[0x2E8 : 0x330] = s_int.to_bytes(72, byteorder="little")

    if tamper_byte_offset is not None:
        report[tamper_byte_offset] ^= tamper_bit_mask

    return bytes(report)


def create_valid_sev_snp_evidence(
    pki: Optional[SyntheticAmdPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    policy: int = DEFAULT_SNP_POLICY,
    measurement: Optional[bytes] = None,
    host_data: Optional[bytes] = None,
    golden_rim: Optional[Dict[str, str]] = None,
    device_id: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> AttestationEvidence:
    """
    Generates a valid, cryptographically signed AttestationEvidence for AMD SEV-SNP.
    Produces an authentic 1184-byte report blob, PEM certificate chain [VCEK, ASK, ARK],
    dual-field report_data, and Golden RIM claims.
    """
    active_pki = pki or get_default_amd_pki()

    # 1. Nonce handling (32 bytes)
    if nonce is None:
        nonce_bytes = secrets.token_bytes(32)
        nonce_hex = nonce_bytes.hex()
    elif isinstance(nonce, (bytes, bytearray)):
        nonce_bytes = bytes(nonce)[:32].ljust(32, b"\x00")
        nonce_hex = bytes(nonce).hex()
    else:
        clean = str(nonce).strip().lower()
        if clean.startswith("0x"):
            clean = clean[2:]
        try:
            nonce_bytes = bytes.fromhex(clean)[:32].ljust(32, b"\x00")
        except ValueError:
            nonce_bytes = clean.encode("utf-8")[:32].ljust(32, b"\x00")
        nonce_hex = clean

    # 2. Ephemeral Pubkey Binding (32 bytes)
    if client_ephemeral_pubkey is not None:
        pubkey_hash_bytes = hashlib.sha256(client_ephemeral_pubkey).digest()
    elif worker_ephemeral_pubkey is not None:
        pubkey_hash_bytes = hashlib.sha256(worker_ephemeral_pubkey).digest()
    else:
        pubkey_hash_bytes = secrets.token_bytes(32)

    # 3. Dual-field report_data (64 bytes: 32B nonce + 32B pubkey hash)
    report_data = nonce_bytes + pubkey_hash_bytes

    meas = measurement if measurement is not None else DEFAULT_SNP_MEASUREMENT
    hd = host_data if host_data is not None else DEFAULT_SNP_HOST_DATA
    rim = golden_rim if golden_rim is not None else {
        "measurement": meas.hex(),
        "host_data": hd.hex(),
    }

    # 4. Generate 1184-byte signed report blob
    report_blob = generate_synthetic_snp_report(
        vcek_priv_ec=active_pki.vcek_priv_ec,
        report_data=report_data,
        measurement=meas,
        host_data=hd,
        chip_id=active_pki.chip_id,
        policy=policy,
    )

    ts = timestamp if timestamp is not None else int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    dev_id = device_id or f"AMD-EPYC-SNP-{active_pki.chip_id[:8].hex()}"

    token_claims: Dict[str, Any] = {
        "version": 2,
        "provider": "amd-sev-snp",
        "device_id": dev_id,
        "policy": policy,
        "policy_hex": f"0x{policy:08x}",
        "policy_debug": bool((policy >> 19) & 1),
        "nonce": nonce_hex,
        "ephemeral_pubkey_hash": pubkey_hash_bytes.hex(),
        "user_data": report_data.hex(),
        "measurement": meas.hex(),
        "host_data": hd.hex(),
        "chip_id": active_pki.chip_id.hex(),
        "timestamp": ts,
        "measurements": {
            "measurement": meas.hex(),
            "host_data": hd.hex(),
        },
    }

    if client_ephemeral_pubkey is not None:
        token_claims["client_pubkey_hash"] = hashlib.sha256(client_ephemeral_pubkey).hexdigest()
    if worker_ephemeral_pubkey is not None:
        token_claims["worker_pubkey_hash"] = hashlib.sha256(worker_ephemeral_pubkey).hexdigest()

    token_sig = report_blob[0x2A0:0x330]

    return AttestationEvidence(
        provider="amd-sev-snp",
        certificate_chain=active_pki.certificate_chain,
        token_claims=token_claims,
        token_signature=token_sig,
        golden_rim=rim,
        evidence_blob=report_blob,
        user_data=report_data,
    )


# ============================================================================
# Adversarial AMD SEV-SNP Fixture Generators
# ============================================================================

def create_tampered_snp_payload_evidence(
    pki: Optional[SyntheticAmdPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    tamper_offset: int = 0x050,
    tamper_mask: int = 0x01,
) -> AttestationEvidence:
    """Negative Fixture: Flips a bit in the signed payload region (0x000..0x29F) of the SNP report."""
    ev = create_valid_sev_snp_evidence(
        pki=pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
    assert ev.evidence_blob is not None
    blob = bytearray(ev.evidence_blob)
    blob[tamper_offset] ^= tamper_mask
    ev.evidence_blob = bytes(blob)
    return ev


def create_tampered_snp_signature_evidence(
    pki: Optional[SyntheticAmdPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    tamper_offset: int = 0x2A0,
    tamper_mask: int = 0x01,
) -> AttestationEvidence:
    """Negative Fixture: Flips a bit in the signature scalars (0x2A0..0x32F) of the SNP report."""
    ev = create_valid_sev_snp_evidence(
        pki=pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
    assert ev.evidence_blob is not None
    blob = bytearray(ev.evidence_blob)
    blob[tamper_offset] ^= tamper_mask
    ev.evidence_blob = bytes(blob)
    ev.token_signature = bytes(blob[0x2A0:0x330])
    return ev


def create_debug_policy_snp_evidence(
    pki: Optional[SyntheticAmdPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Report generated with POLICY.DEBUG enabled (bit 19 == 1)."""
    debug_policy = DEFAULT_SNP_POLICY | (1 << 19)
    return create_valid_sev_snp_evidence(
        pki=pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        policy=debug_policy,
    )


def create_nonce_mismatch_snp_evidence(
    requested_nonce: bytes,
    pki: Optional[SyntheticAmdPKI] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Report generated with mismatched nonce (replay attack simulation)."""
    stale_nonce = secrets.token_bytes(32)
    while stale_nonce == requested_nonce:
        stale_nonce = secrets.token_bytes(32)
    return create_valid_sev_snp_evidence(
        pki=pki,
        nonce=stale_nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )


def create_pubkey_hash_mismatch_snp_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    pki: Optional[SyntheticAmdPKI] = None,
) -> AttestationEvidence:
    """Negative Fixture: Bound ephemeral pubkey hash altered (key substitution simulation)."""
    fake_pubkey = secrets.token_bytes(32)
    return create_valid_sev_snp_evidence(
        pki=pki,
        nonce=nonce,
        worker_ephemeral_pubkey=fake_pubkey,
        client_ephemeral_pubkey=fake_pubkey,
    )


def create_untrusted_ark_snp_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> Tuple[AttestationEvidence, str]:
    """Negative Fixture: Certificate chain rooted in an untrusted/rogue ARK Root CA."""
    rogue_pki = generate_synthetic_amd_pki(ark_cn="Rogue Untrusted AMD ARK Root CA")
    ev = create_valid_sev_snp_evidence(
        pki=rogue_pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
    return ev, rogue_pki.ark_cert_pem


def create_expired_vcek_snp_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: VCEK certificate chain has expired."""
    expired_pki = generate_synthetic_amd_pki(expired=True)
    return create_valid_sev_snp_evidence(
        pki=expired_pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )


def create_not_yet_valid_vcek_snp_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: VCEK certificate chain is not yet valid."""
    future_pki = generate_synthetic_amd_pki(not_yet_valid=True)
    return create_valid_sev_snp_evidence(
        pki=future_pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )


def create_revoked_vcek_snp_evidence(
    pki: Optional[SyntheticAmdPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> Tuple[AttestationEvidence, int]:
    """Negative Fixture: Returns (evidence, vcek_serial_number) for revocation testing."""
    active_pki = pki or get_default_amd_pki()
    ev = create_valid_sev_snp_evidence(
        pki=active_pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
    return ev, active_pki.vcek_cert.serial_number


def create_truncated_snp_report_evidence(
    pki: Optional[SyntheticAmdPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    truncated_length: int = 1000,
) -> AttestationEvidence:
    """Negative Fixture: Truncated evidence blob (< 1184 bytes)."""
    ev = create_valid_sev_snp_evidence(
        pki=pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
    assert ev.evidence_blob is not None
    ev.evidence_blob = ev.evidence_blob[:truncated_length]
    return ev


def create_measurement_mismatch_snp_evidence(
    pki: Optional[SyntheticAmdPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Report measurement digest does not match Golden RIM."""
    tampered_measurement = b"\xde\xad\xbe\xef" * 12  # 48 bytes
    ev = create_valid_sev_snp_evidence(
        pki=pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        measurement=tampered_measurement,
        golden_rim=DEFAULT_SNP_GOLDEN_RIM.copy(),
    )
    return ev


def create_tcb_downgrade_snp_evidence(
    pki: Optional[SyntheticAmdPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Reported TCB version is downgraded below VCEK SPLs."""
    active_pki = pki or get_default_amd_pki()
    # Downgrade bootloader SPL to 0 (VCEK SPL is 3)
    downgraded_tcb = bytes([0, 0, 0, 0, 0, 0, 0, 0])
    blob = generate_synthetic_snp_report(
        vcek_priv_ec=active_pki.vcek_priv_ec,
        reported_tcb=downgraded_tcb,
        chip_id=active_pki.chip_id,
    )
    ev = create_valid_sev_snp_evidence(
        pki=active_pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
    )
    ev.evidence_blob = blob
    return ev



# ============================================================================
# Intel TDX Constants, Golden RIM & OID Specifications
# ============================================================================

INTEL_QE_VENDOR_ID: bytes = bytes.fromhex("939a7233f79c4ca9940a0db3957f0607")
INTEL_TDX_TEE_TYPE: int = 0x00000081

INTEL_SGX_EXTENSION_OID: str = "1.2.840.113741.1.13.1"
INTEL_OID_PPID: str = f"{INTEL_SGX_EXTENSION_OID}.1"
INTEL_OID_TCB: str = f"{INTEL_SGX_EXTENSION_OID}.2"
INTEL_OID_PCEID: str = f"{INTEL_SGX_EXTENSION_OID}.3"
INTEL_OID_FMSPC: str = f"{INTEL_SGX_EXTENSION_OID}.4"
INTEL_OID_SGX_TYPE: str = f"{INTEL_SGX_EXTENSION_OID}.5"
INTEL_OID_PLATFORM_INSTANCE_ID: str = f"{INTEL_SGX_EXTENSION_OID}.6"
INTEL_OID_CONFIGURATION: str = f"{INTEL_SGX_EXTENSION_OID}.7"
INTEL_OID_TCB_COMP_SVNS: str = f"{INTEL_SGX_EXTENSION_OID}.14"
INTEL_OID_FMSPC_SPEC_COMPAT: str = f"{INTEL_SGX_EXTENSION_OID}.2"

DEFAULT_INTEL_FMSPC: str = "00806f000000"  # Sapphire Rapids
DEFAULT_INTEL_PCEID: str = "0000"
DEFAULT_INTEL_TCB_COMP_SVNS: bytes = bytes([1] * 16)
DEFAULT_INTEL_PCE_SVN: int = 1

DEFAULT_TDX_MRTD: bytes = hashlib.sha384(b"matrix-compute-tdx-mrtd-synthetic-v1").digest()
DEFAULT_TDX_RTMR0: bytes = bytes.fromhex(
    "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"
)
DEFAULT_TDX_RTMR1: bytes = bytes.fromhex(
    "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"
)
DEFAULT_TDX_RTMR2: bytes = bytes.fromhex(
    "99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa"
)
DEFAULT_TDX_RTMR3: bytes = bytes.fromhex(
    "f1e2d3c4b5a60718293a4b5c6d7e8f90f1e2d3c4b5a60718293a4b5c6d7e8f90f1e2d3c4b5a60718293a4b5c6d7e8f90"
)
DEFAULT_TDX_MRSEAM: bytes = hashlib.sha384(b"matrix-compute-tdx-mrseam-v1").digest()
DEFAULT_TDX_MRSIGNERSEAM: bytes = hashlib.sha384(b"matrix-compute-tdx-mrsignerseam-v1").digest()
DEFAULT_TDX_TEE_TCB_SVN: bytes = bytes([3, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 115])

DEFAULT_TDX_GOLDEN_RIM: Dict[str, str] = {
    "mrtd": DEFAULT_TDX_MRTD.hex(),
    "rtmr0": DEFAULT_TDX_RTMR0.hex(),
    "rtmr1": DEFAULT_TDX_RTMR1.hex(),
    "rtmr2": DEFAULT_TDX_RTMR2.hex(),
    "rtmr3": DEFAULT_TDX_RTMR3.hex(),
}


# ============================================================================
# Intel ASN.1 DER Helpers
# ============================================================================

def der_encode_oid(oid_str: str) -> bytes:
    """DER encodes an ASN.1 OBJECT IDENTIFIER (Tag 0x06)."""
    parts = [int(p) for p in oid_str.split(".")]
    if len(parts) < 2:
        raise ValueError(f"OID must have at least 2 components: {oid_str}")
    first = 40 * parts[0] + parts[1]
    encoded = bytearray([first])
    for part in parts[2:]:
        if part == 0:
            encoded.append(0)
            continue
        vals: List[int] = []
        val = part
        while val > 0:
            vals.append(val & 0x7F)
            val >>= 7
        vals.reverse()
        for i in range(len(vals) - 1):
            vals[i] |= 0x80
        encoded.extend(vals)
    length = len(encoded)
    if length < 128:
        len_bytes = bytes([length])
    else:
        len_bytes = b"\x81" + bytes([length])
    return b"\x06" + len_bytes + bytes(encoded)


def der_decode_oid(raw: bytes) -> str:
    """DER decodes an ASN.1 OBJECT IDENTIFIER (Tag 0x06)."""
    if len(raw) < 2 or raw[0] != 0x06:
        raise ValueError(f"Invalid DER OID tag: {raw[:2]!r}")
    length = raw[1]
    offset = 2
    if length & 0x80:
        num_len_bytes = length & 0x7F
        length = int.from_bytes(raw[offset : offset + num_len_bytes], "big")
        offset += num_len_bytes
    content = raw[offset : offset + length]
    first = content[0]
    first_comp = first // 40
    second_comp = first % 40
    components = [first_comp, second_comp]
    val = 0
    for b in content[1:]:
        val = (val << 7) | (b & 0x7F)
        if (b & 0x80) == 0:
            components.append(val)
            val = 0
    return ".".join(str(c) for c in components)


def der_encode_sequence(items: List[bytes]) -> bytes:
    """DER encodes an ASN.1 SEQUENCE (Tag 0x30)."""
    content = b"".join(items)
    length = len(content)
    if length < 128:
        len_bytes = bytes([length])
    elif length < 256:
        len_bytes = b"\x81" + bytes([length])
    else:
        len_bytes = b"\x82" + struct.pack(">H", length)
    return b"\x30" + len_bytes + content


def der_decode_sequence(raw: bytes) -> List[bytes]:
    """DER decodes an ASN.1 SEQUENCE (Tag 0x30) into constituent element byte strings."""
    if len(raw) < 2 or raw[0] != 0x30:
        raise ValueError(f"Invalid DER sequence tag: {raw[:2]!r}")
    length = raw[1]
    offset = 2
    if length & 0x80:
        num_len_bytes = length & 0x7F
        length = int.from_bytes(raw[offset : offset + num_len_bytes], "big")
        offset += num_len_bytes
    end = offset + length
    elements: List[bytes] = []
    curr = offset
    while curr < end:
        elem_len = raw[curr + 1]
        elem_offset = curr + 2
        if elem_len & 0x80:
            n_bytes = elem_len & 0x7F
            elem_len = int.from_bytes(raw[elem_offset : elem_offset + n_bytes], "big")
            elem_offset += n_bytes
        elem_total = elem_offset + elem_len
        elements.append(raw[curr:elem_total])
        curr = elem_total
    return elements


def encode_intel_pck_asn1_extension(
    fmspc: Union[bytes, str],
    pceid: Union[bytes, str],
    tcb_comp_svns: Optional[bytes] = None,
    ppid: Optional[bytes] = None,
    pce_svn: int = 1,
) -> bytes:
    """
    Constructs the complete DER-encoded ASN.1 SEQUENCE containing all Intel PCK sub-extensions
    for embedding inside X.509 extension OID 1.2.840.113741.1.13.1.
    
    Ensures dual compatibility:
    - Encodes sub-OID .2 with 6-byte FMSPC for compliance with spec parser (line 947).
    - Encodes sub-OID .4 with 6-byte FMSPC for compliance with standard DCAP tooling.
    - Encodes sub-OID .3 with 2-byte PCEID.
    - Encodes sub-OID .1 with 16-byte PPID.
    - Encodes sub-OID .14 with 16-byte TCB component SVNs.
    """
    actual_fmspc = bytes.fromhex(fmspc) if isinstance(fmspc, str) else bytes(fmspc)
    if len(actual_fmspc) != 6:
        raise ValueError(f"fmspc must be exactly 6 bytes, got {len(actual_fmspc)}")

    actual_pceid = bytes.fromhex(pceid) if isinstance(pceid, str) else bytes(pceid)
    if len(actual_pceid) != 2:
        raise ValueError(f"pceid must be exactly 2 bytes, got {len(actual_pceid)}")

    actual_tcb = tcb_comp_svns if tcb_comp_svns is not None else DEFAULT_INTEL_TCB_COMP_SVNS
    if len(actual_tcb) != 16:
        raise ValueError(f"tcb_comp_svns must be exactly 16 bytes, got {len(actual_tcb)}")

    actual_ppid = ppid if ppid is not None else (b"\xaa" * 16)

    sub_exts: List[bytes] = [
        # 1. PPID (.1)
        der_encode_sequence([
            der_encode_oid(INTEL_OID_PPID),
            der_encode_octet_string(actual_ppid),
        ]),
        # 2. Spec-compat FMSPC (.2) -> verified by IntelTdxVerifier.parse_pck_extensions line 947
        der_encode_sequence([
            der_encode_oid(INTEL_OID_FMSPC_SPEC_COMPAT),
            der_encode_octet_string(actual_fmspc),
        ]),
        # 3. PCEID (.3) -> verified by IntelTdxVerifier.parse_pck_extensions line 955
        der_encode_sequence([
            der_encode_oid(INTEL_OID_PCEID),
            der_encode_octet_string(actual_pceid),
        ]),
        # 4. Standard DCAP FMSPC (.4)
        der_encode_sequence([
            der_encode_oid(INTEL_OID_FMSPC),
            der_encode_octet_string(actual_fmspc),
        ]),
        # 5. TCB Component SVNs (.14)
        der_encode_sequence([
            der_encode_oid(INTEL_OID_TCB_COMP_SVNS),
            der_encode_octet_string(actual_tcb),
        ]),
    ]
    return der_encode_sequence(sub_exts)


def build_intel_pck_extension(
    fmspc_bytes: bytes,
    pceid_bytes: bytes,
    tcb_comp_svns: Optional[bytes] = None,
) -> bytes:
    """Convenience alias for encode_intel_pck_asn1_extension."""
    return encode_intel_pck_asn1_extension(
        fmspc=fmspc_bytes,
        pceid=pceid_bytes,
        tcb_comp_svns=tcb_comp_svns,
    )


# ============================================================================
# Synthetic Intel PKI Hierarchy & Caching
# ============================================================================

@dataclass
class SyntheticIntelPKI:
    """
    Synthetic 3-tier Intel SGX/TDX PKI hierarchy:
    Root CA (RSA-3072 or ECDSA P-384) -> Intermediate CA (ECDSA P-256) -> PCK Leaf (ECDSA P-256).
    """
    root_cert: x509.Certificate
    intermediate_cert: x509.Certificate
    pck_cert: x509.Certificate
    root_key: Union[rsa.RSAPrivateKey, ec.EllipticCurvePrivateKey]
    intermediate_key: Union[rsa.RSAPrivateKey, ec.EllipticCurvePrivateKey]
    pck_key: ec.EllipticCurvePrivateKey
    pck_priv_ec: ec.EllipticCurvePrivateKey = None  # type: ignore[assignment]
    fmspc: str = DEFAULT_INTEL_FMSPC
    pceid: str = DEFAULT_INTEL_PCEID
    tcb_comp_svns: bytes = field(default_factory=lambda: bytes(DEFAULT_INTEL_TCB_COMP_SVNS))
    pce_svn: int = DEFAULT_INTEL_PCE_SVN

    _root_bundle: Optional[CertBundle] = field(default=None, init=False, repr=False)
    _inter_bundle: Optional[CertBundle] = field(default=None, init=False, repr=False)
    _pck_bundle: Optional[CertBundle] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.pck_priv_ec is None and self.pck_key is not None:
            self.pck_priv_ec = self.pck_key
        elif self.pck_key is None and self.pck_priv_ec is not None:
            self.pck_key = self.pck_priv_ec
        self._root_bundle = CertBundle.from_cert_and_key(self.root_cert, self.root_key)
        self._inter_bundle = CertBundle.from_cert_and_key(self.intermediate_cert, self.intermediate_key)
        self._pck_bundle = CertBundle.from_cert_and_key(self.pck_cert, self.pck_key)

    @property
    def root_cert_pem(self) -> str:
        return self.root_cert.public_bytes(serialization.Encoding.PEM).decode("ascii")

    @property
    def intermediate_cert_pem(self) -> str:
        return self.intermediate_cert.public_bytes(serialization.Encoding.PEM).decode("ascii")

    @property
    def pck_cert_pem(self) -> str:
        return self.pck_cert.public_bytes(serialization.Encoding.PEM).decode("ascii")

    @property
    def leaf_cert_pem(self) -> str:
        """Alias to pck_cert_pem for uniform interface across TEE providers."""
        return self.pck_cert_pem

    @property
    def certificate_chain(self) -> List[str]:
        """Ordered from leaf to root: [PCK Leaf, Intermediate CA, Root CA]."""
        return [self.pck_cert_pem, self.intermediate_cert_pem, self.root_cert_pem]

    @property
    def root(self) -> CertBundle:
        assert self._root_bundle is not None
        return self._root_bundle

    @property
    def intermediate(self) -> CertBundle:
        assert self._inter_bundle is not None
        return self._inter_bundle

    @property
    def pck(self) -> CertBundle:
        assert self._pck_bundle is not None
        return self._pck_bundle

    @property
    def leaf(self) -> CertBundle:
        return self.pck


def generate_synthetic_intel_pki(
    valid_days: int = 365,
    expired: bool = False,
    not_yet_valid: bool = False,
    fmspc: Optional[Union[bytes, str]] = None,
    pceid: Optional[Union[bytes, str]] = None,
    tcb_comp_svns: Optional[bytes] = None,
    pce_svn: int = 1,
    root_cn: str = "Intel SGX Root CA",
    intermediate_cn: str = "Intel SGX PCK Processor CA",
    pck_cn: str = "Intel SGX PCK Certificate",
    rsa_bits: int = 3072,
    root_key_type: str = "rsa",  # 'rsa' (RSA-3072) or 'ec' (SECP384R1)
) -> SyntheticIntelPKI:
    """
    Generates a full 3-tier synthetic Intel SGX/TDX PKI hierarchy:
    1. Root CA: Self-signed RSA-3072 (or ECDSA P-384) CA certificate (path_length=2).
    2. Intermediate CA: ECDSA P-256 CA certificate signed by Root CA (path_length=0).
    3. PCK Leaf: ECDSA P-256 leaf certificate signed by Intermediate CA,
       containing Intel SGX/TDX custom ASN.1 extensions (OID 1.2.840.113741.1.13.1).
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

    if fmspc is not None:
        fmspc_str = fmspc.hex().lower() if isinstance(fmspc, (bytes, bytearray)) else str(fmspc).lower()
    else:
        fmspc_str = DEFAULT_INTEL_FMSPC

    if pceid is not None:
        pceid_str = pceid.hex().lower() if isinstance(pceid, (bytes, bytearray)) else str(pceid).lower()
    else:
        pceid_str = DEFAULT_INTEL_PCEID

    actual_tcb = tcb_comp_svns if tcb_comp_svns is not None else DEFAULT_INTEL_TCB_COMP_SVNS

    # 1. Root CA (Self-Signed)
    if root_key_type.lower() == "ec":
        root_key = ec.generate_private_key(ec.SECP384R1())
        root_hash = hashes.SHA384()
    else:
        root_key = rsa.generate_private_key(public_exponent=65537, key_size=rsa_bits)
        root_hash = hashes.SHA256()

    root_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Intel Corporation"),
        x509.NameAttribute(NameOID.COMMON_NAME, root_cn),
    ])
    root_cert = (
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
        .sign(root_key, root_hash)
    )

    # 2. Intermediate CA (Signed by Root CA)
    inter_key = ec.generate_private_key(ec.SECP256R1())
    inter_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Intel Corporation"),
        x509.NameAttribute(NameOID.COMMON_NAME, intermediate_cn),
    ])
    inter_cert = (
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
        .sign(root_key, root_hash)
    )

    # 3. Leaf PCK Certificate (Signed by Intermediate CA with Proprietary Intel Extension)
    pck_key = ec.generate_private_key(ec.SECP256R1())
    pck_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Intel Corporation"),
        x509.NameAttribute(NameOID.COMMON_NAME, pck_cn),
    ])
    intel_extension_der = encode_intel_pck_asn1_extension(
        fmspc=fmspc_str,
        pceid=pceid_str,
        tcb_comp_svns=actual_tcb,
        pce_svn=pce_svn,
    )
    pck_cert = (
        x509.CertificateBuilder()
        .subject_name(pck_subject)
        .issuer_name(inter_subject)
        .public_key(pck_key.public_key())
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
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(pck_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(inter_key.public_key()), critical=False)
        .add_extension(
            x509.UnrecognizedExtension(
                x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID),
                intel_extension_der,
            ),
            critical=False,
        )
        .sign(inter_key, hashes.SHA256())
    )

    return SyntheticIntelPKI(
        root_cert=root_cert,
        intermediate_cert=inter_cert,
        pck_cert=pck_cert,
        root_key=root_key,
        intermediate_key=inter_key,
        pck_key=pck_key,
        pck_priv_ec=pck_key,
        fmspc=fmspc_str,
        pceid=pceid_str,
        tcb_comp_svns=actual_tcb,
        pce_svn=pce_svn,
    )


_DEFAULT_INTEL_PKI_CACHE: Optional[SyntheticIntelPKI] = None
_INTEL_PKI_LOCK = threading.Lock()


def get_default_intel_pki() -> SyntheticIntelPKI:
    """Returns thread-safe cached singleton SyntheticIntelPKI to optimize test execution speed (<1ms on cache hit)."""
    global _DEFAULT_INTEL_PKI_CACHE
    if _DEFAULT_INTEL_PKI_CACHE is None:
        with _INTEL_PKI_LOCK:
            if _DEFAULT_INTEL_PKI_CACHE is None:
                _DEFAULT_INTEL_PKI_CACHE = generate_synthetic_intel_pki()
    return _DEFAULT_INTEL_PKI_CACHE


def reset_default_intel_pki_cache() -> None:
    """Resets the cached singleton instance (useful for hermetic test isolation)."""
    global _DEFAULT_INTEL_PKI_CACHE
    with _INTEL_PKI_LOCK:
        _DEFAULT_INTEL_PKI_CACHE = None


# ============================================================================
# Intel TDX DCAP Quote & Evidence Generators
# ============================================================================

def generate_synthetic_tdx_quote(
    pki: Optional[SyntheticIntelPKI] = None,
    quote_version: int = 4,
    att_key_type: int = 2,
    tee_type: int = INTEL_TDX_TEE_TYPE,
    tdx_type: int = 2,  # 2 = TDX 1.0, 3 = TDX 1.5 in modular v5
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    report_data: Optional[bytes] = None,
    debug: bool = False,
    tdattributes: Optional[int] = None,
    mrtd: Optional[bytes] = None,
    rtmr0: Optional[bytes] = None,
    rtmr1: Optional[bytes] = None,
    rtmr2: Optional[bytes] = None,
    rtmr3: Optional[bytes] = None,
    mrseam: Optional[bytes] = None,
    mrsignerseam: Optional[bytes] = None,
    tee_tcb_svn: Optional[bytes] = None,
    xfam: int = 0,
    mrconfigid: Optional[bytes] = None,
    mrowner: Optional[bytes] = None,
    mrownerconfig: Optional[bytes] = None,
    qe_auth_data: Optional[bytes] = None,
    raw_quote_sig: Optional[bytes] = None,
    raw_quote_signature: Optional[bytes] = None,
    raw_qe_sig: Optional[bytes] = None,
    raw_qe_signature: Optional[bytes] = None,
    raw_qe_report_body: Optional[bytes] = None,
    tamper_byte_offset: Optional[int] = None,
    tamper_bit_mask: int = 0x01,
    tamper_xor_mask: int = 0x01,
) -> bytes:
    """
    Constructs a valid, cryptographically authentic binary Intel TDX DCAP Quote (Format v4/v5).
    - Signs Header + TD Report Body using an ephemeral Attestation Key (AK).
    - Embeds SHA-256(att_public_key || qe_auth_data) into the Quote Enclave (QE3) SGX Report.
    - Signs the QE Report using the PCK Leaf Private Key from pki.
    - Appends the 3-tier PEM certificate chain.
    """
    active_pki = pki or get_default_intel_pki()
    actual_quote_sig = raw_quote_sig if raw_quote_sig is not None else raw_quote_signature
    actual_qe_sig = raw_qe_sig if raw_qe_sig is not None else raw_qe_signature
    actual_mask = tamper_xor_mask if tamper_xor_mask != 0x01 else tamper_bit_mask

    # 1. Resolve 64-byte REPORTDATA
    if report_data is not None:
        actual_report_data = bytes(report_data)[:64].ljust(64, b"\x00")
    else:
        if nonce is None:
            nonce_bytes = secrets.token_bytes(32)
        elif isinstance(nonce, (bytes, bytearray)):
            nonce_bytes = bytes(nonce)[:32].ljust(32, b"\x00")
        else:
            clean = str(nonce).strip().lower()
            if clean.startswith("0x"):
                clean = clean[2:]
            try:
                nonce_bytes = bytes.fromhex(clean)[:32].ljust(32, b"\x00")
            except ValueError:
                nonce_bytes = clean.encode("utf-8")[:32].ljust(32, b"\x00")

        if client_ephemeral_pubkey is not None and worker_ephemeral_pubkey is not None:
            pubkey_hash = hashlib.sha256(client_ephemeral_pubkey + worker_ephemeral_pubkey).digest()
        elif client_ephemeral_pubkey is not None:
            pubkey_hash = hashlib.sha256(client_ephemeral_pubkey).digest()
        elif worker_ephemeral_pubkey is not None:
            pubkey_hash = hashlib.sha256(worker_ephemeral_pubkey).digest()
        else:
            pubkey_hash = secrets.token_bytes(32)

        actual_report_data = nonce_bytes + pubkey_hash

    # 2. Build 48-byte Quote Header
    header = struct.pack(
        "<HHIHH16s20s",
        quote_version,
        att_key_type,
        tee_type,
        0,
        0,
        INTEL_QE_VENDOR_ID,
        b"\x00" * 20,
    )
    assert len(header) == 48

    # 3. Build 584-byte TD Report Body
    td_body = bytearray(584)
    td_body[0:16] = (tee_tcb_svn or DEFAULT_TDX_TEE_TCB_SVN)[:16]
    td_body[16:64] = (mrseam or DEFAULT_TDX_MRSEAM)[:48]
    td_body[64:112] = (mrsignerseam or DEFAULT_TDX_MRSIGNERSEAM)[:48]
    struct.pack_into("<Q", td_body, 112, 0)  # seamattributes

    # TDATTRIBUTES: bit 0 is DEBUG
    actual_tdattributes = tdattributes if tdattributes is not None else (0x01 if debug else 0x00)
    struct.pack_into("<Q", td_body, 120, actual_tdattributes)
    struct.pack_into("<Q", td_body, 128, xfam)

    td_body[136:184] = (mrtd or DEFAULT_TDX_MRTD)[:48]
    td_body[184:232] = (mrconfigid or b"\x00" * 48)[:48]
    td_body[232:280] = (mrowner or b"\x00" * 48)[:48]
    td_body[280:328] = (mrownerconfig or b"\x00" * 48)[:48]
    td_body[328:376] = (rtmr0 or DEFAULT_TDX_RTMR0)[:48]
    td_body[376:424] = (rtmr1 or DEFAULT_TDX_RTMR1)[:48]
    td_body[424:472] = (rtmr2 or DEFAULT_TDX_RTMR2)[:48]
    td_body[472:520] = (rtmr3 or DEFAULT_TDX_RTMR3)[:48]
    td_body[520:584] = actual_report_data[:64]

    # 4. Construct Signed Body (v4 vs modular v5)
    if quote_version == 5:
        # 6-byte modular body descriptor: type (2B) + size (4B)
        body_descriptor = struct.pack("<HI", tdx_type, len(td_body))
        signed_body = header + body_descriptor + bytes(td_body)
    else:
        # Standard v4 contiguous layout
        signed_body = header + bytes(td_body)

    # 5. Generate Ephemeral Attestation Key (AK)
    ak_key = ec.generate_private_key(ec.SECP256R1() if att_key_type == 2 else ec.SECP384R1())
    ak_pub_nums = ak_key.public_key().public_numbers()
    coord_len = 32 if att_key_type == 2 else 48
    att_pubkey_raw = ak_pub_nums.x.to_bytes(coord_len, "big") + ak_pub_nums.y.to_bytes(coord_len, "big")

    # 6. Sign Header + Body with Attestation Key (AK)
    if actual_quote_sig is not None:
        quote_signature = actual_quote_sig
    else:
        ak_hash_algo = hashes.SHA256() if att_key_type == 2 else hashes.SHA384()
        der_quote_sig = ak_key.sign(signed_body, ec.ECDSA(ak_hash_algo))
        r_ak, s_ak = decode_dss_signature(der_quote_sig)
        quote_signature = r_ak.to_bytes(coord_len, "big") + s_ak.to_bytes(coord_len, "big")

    # 7. Construct 384-byte Quote Enclave (QE3) SGX Report Body
    actual_qe_auth_data = qe_auth_data or b""
    if raw_qe_report_body is not None:
        qe_report_body = raw_qe_report_body
    else:
        qe_body = bytearray(384)
        qe_body[0:16] = DEFAULT_TDX_TEE_TCB_SVN
        struct.pack_into("<I", qe_body, 16, 0)    # miscselect
        struct.pack_into("<Q", qe_body, 48, 0x07)  # attributes (initted, !debug, mode64)
        qe_body[64:96] = hashlib.sha256(b"intel-synthetic-qe3-mrenclave").digest()
        qe_body[128:160] = hashlib.sha256(b"intel-synthetic-qe3-mrsigner").digest()
        struct.pack_into("<H", qe_body, 256, 1)   # isvprodid
        struct.pack_into("<H", qe_body, 258, 1)   # isvsvn

        # CRYPTOGRAPHIC AK BINDING: report_data[:32] == SHA256(att_public_key || qe_auth_data)
        qe_binding_hash = hashlib.sha256(att_pubkey_raw + actual_qe_auth_data).digest()
        qe_body[320:352] = qe_binding_hash
        qe_body[352:384] = b"\x00" * 32
        qe_report_body = bytes(qe_body)

    # 8. Sign QE Report with PCK Leaf Key
    if actual_qe_sig is not None:
        qe_report_signature = actual_qe_sig
    else:
        pck_hash_algo = hashes.SHA256() if isinstance(active_pki.pck_key.curve, ec.SECP256R1) else hashes.SHA384()
        der_qe_sig = active_pki.pck_key.sign(qe_report_body, ec.ECDSA(pck_hash_algo))
        r_pck, s_pck = decode_dss_signature(der_qe_sig)
        pck_coord_len = 32 if isinstance(active_pki.pck_key.curve, ec.SECP256R1) else 48
        qe_report_signature = r_pck.to_bytes(pck_coord_len, "big") + s_pck.to_bytes(pck_coord_len, "big")

    # 9. Format Certification Data (Type 5: Concatenated PEM Chain)
    cert_chain_pem = "".join(active_pki.certificate_chain).encode("ascii")
    cert_data_type = 5  # PCK Certificate Chain
    cert_data_payload = (
        struct.pack("<H", cert_data_type)
        + struct.pack("<I", len(cert_chain_pem))
        + cert_chain_pem
    )

    # 10. Assemble Signature Data Structure
    auth_data_payload = struct.pack("<H", len(actual_qe_auth_data)) + actual_qe_auth_data
    sig_payload = (
        quote_signature
        + att_pubkey_raw
        + qe_report_body
        + qe_report_signature
        + auth_data_payload
        + cert_data_payload
    )
    sig_data_len = struct.pack("<I", len(sig_payload))

    # 11. Assemble Full Quote Binary
    quote = bytearray(signed_body + sig_data_len + sig_payload)

    # 12. Apply Optional Tampering
    if tamper_byte_offset is not None:
        quote[tamper_byte_offset] ^= actual_mask

    return bytes(quote)


def create_valid_intel_tdx_evidence(
    pki: Optional[SyntheticIntelPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    quote_version: int = 4,
    debug: bool = False,
    mrtd: Optional[bytes] = None,
    rtmr0: Optional[bytes] = None,
    rtmr1: Optional[bytes] = None,
    rtmr2: Optional[bytes] = None,
    rtmr3: Optional[bytes] = None,
    golden_rim: Optional[Dict[str, str]] = None,
    device_id: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> AttestationEvidence:
    """
    Generates a valid, cryptographically authentic AttestationEvidence for Intel TDX.
    Produces a valid DCAP Quote (v4/v5), PEM certificate chain [PCK, Intermediate, Root],
    dual-field 64-byte reportdata binding nonce and ephemeral pubkeys, and Golden RIM.
    """
    active_pki = pki or get_default_intel_pki()

    # 1. Nonce handling (32 bytes)
    if nonce is None:
        nonce_bytes = secrets.token_bytes(32)
        nonce_hex = nonce_bytes.hex()
    elif isinstance(nonce, (bytes, bytearray)):
        nonce_bytes = bytes(nonce)[:32].ljust(32, b"\x00")
        nonce_hex = bytes(nonce).hex()
    else:
        clean = str(nonce).strip().lower()
        if clean.startswith("0x"):
            clean = clean[2:]
        try:
            nonce_bytes = bytes.fromhex(clean)[:32].ljust(32, b"\x00")
        except ValueError:
            nonce_bytes = clean.encode("utf-8")[:32].ljust(32, b"\x00")
        nonce_hex = clean

    # 2. Ephemeral Pubkey Binding (32 bytes)
    if client_ephemeral_pubkey is not None and worker_ephemeral_pubkey is not None:
        pubkey_hash_bytes = hashlib.sha256(client_ephemeral_pubkey + worker_ephemeral_pubkey).digest()
    elif client_ephemeral_pubkey is not None:
        pubkey_hash_bytes = hashlib.sha256(client_ephemeral_pubkey).digest()
    elif worker_ephemeral_pubkey is not None:
        pubkey_hash_bytes = hashlib.sha256(worker_ephemeral_pubkey).digest()
    else:
        pubkey_hash_bytes = secrets.token_bytes(32)

    # 3. Dual-field reportdata (64 bytes: 32B nonce + 32B pubkey hash)
    report_data = nonce_bytes + pubkey_hash_bytes

    actual_mrtd = mrtd if mrtd is not None else DEFAULT_TDX_MRTD
    actual_rtmr0 = rtmr0 if rtmr0 is not None else DEFAULT_TDX_RTMR0
    actual_rtmr1 = rtmr1 if rtmr1 is not None else DEFAULT_TDX_RTMR1
    actual_rtmr2 = rtmr2 if rtmr2 is not None else DEFAULT_TDX_RTMR2
    actual_rtmr3 = rtmr3 if rtmr3 is not None else DEFAULT_TDX_RTMR3

    rim = golden_rim if golden_rim is not None else {
        "mrtd": actual_mrtd.hex(),
        "rtmr0": actual_rtmr0.hex(),
        "rtmr1": actual_rtmr1.hex(),
        "rtmr2": actual_rtmr2.hex(),
        "rtmr3": actual_rtmr3.hex(),
    }

    # 4. Generate binary DCAP Quote
    quote_blob = generate_synthetic_tdx_quote(
        pki=active_pki,
        report_data=report_data,
        mrtd=actual_mrtd,
        rtmr0=actual_rtmr0,
        rtmr1=actual_rtmr1,
        rtmr2=actual_rtmr2,
        rtmr3=actual_rtmr3,
        quote_version=quote_version,
        debug=debug,
    )

    ts = timestamp if timestamp is not None else int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    dev_id = device_id or f"Intel-Xeon-TDX-{active_pki.fmspc}"

    token_claims: Dict[str, Any] = {
        "provider": "intel-tdx",
        "quote_version": quote_version,
        "tee_type": "0x00000081",
        "device_id": dev_id,
        "debug": debug,
        "nonce": nonce_hex,
        "ephemeral_pubkey_hash": pubkey_hash_bytes.hex(),
        "user_data": report_data.hex(),
        "timestamp": ts,
        "fmspc": active_pki.fmspc,
        "pceid": active_pki.pceid,
        "measurements": {
            "mrtd": actual_mrtd.hex(),
            "rtmr0": actual_rtmr0.hex(),
            "rtmr1": actual_rtmr1.hex(),
            "rtmr2": actual_rtmr2.hex(),
            "rtmr3": actual_rtmr3.hex(),
        },
    }

    if client_ephemeral_pubkey is not None:
        token_claims["client_pubkey_hash"] = hashlib.sha256(client_ephemeral_pubkey).hexdigest()
    if worker_ephemeral_pubkey is not None:
        token_claims["worker_pubkey_hash"] = hashlib.sha256(worker_ephemeral_pubkey).hexdigest()

    # AK signature extracted from signature data structure (offset 636..700 in v4, 642..706 in v5)
    sig_offset = 636 if quote_version == 4 else 642
    quote_sig = quote_blob[sig_offset : sig_offset + 64]

    return AttestationEvidence(
        provider="intel-tdx",
        certificate_chain=active_pki.certificate_chain,
        token_claims=token_claims,
        token_signature=quote_sig,
        golden_rim=rim,
        evidence_blob=quote_blob,
        user_data=report_data,
    )


# ============================================================================
# Adversarial & Negative Intel TDX Fixture Generators
# ============================================================================

def create_tampered_tdx_quote_evidence(
    pki: Optional[SyntheticIntelPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    tamper_offset: int = 0x050,  # Offset in signed payload (0..631)
    tamper_mask: int = 0x01,
) -> AttestationEvidence:
    """Negative Fixture: Flips a bit in the signed Quote Header or TD Report Body (0..631)."""
    ev = create_valid_intel_tdx_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )
    assert ev.evidence_blob is not None
    blob = bytearray(ev.evidence_blob)
    blob[tamper_offset] ^= tamper_mask
    ev.evidence_blob = bytes(blob)
    return ev


def create_tampered_tdx_signature_evidence(
    pki: Optional[SyntheticIntelPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    tamper_offset: int = 636,  # Offset inside quote_signature (636..699)
    tamper_mask: int = 0x01,
) -> AttestationEvidence:
    """Negative Fixture: Flips a bit in the Attestation Key (AK) ECDSA signature scalars."""
    ev = create_valid_intel_tdx_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )
    assert ev.evidence_blob is not None
    blob = bytearray(ev.evidence_blob)
    blob[tamper_offset] ^= tamper_mask
    ev.evidence_blob = bytes(blob)
    ev.token_signature = bytes(blob[636:700])
    return ev


def create_tampered_qe_signature_evidence(
    pki: Optional[SyntheticIntelPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    tamper_offset: int = 1148,  # Offset inside qe_report_signature (1148..1211)
    tamper_mask: int = 0x01,
) -> AttestationEvidence:
    """Negative Fixture: Flips a bit in the PCK ECDSA signature over the QE3 SGX report."""
    ev = create_valid_intel_tdx_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )
    assert ev.evidence_blob is not None
    blob = bytearray(ev.evidence_blob)
    blob[tamper_offset] ^= tamper_mask
    ev.evidence_blob = bytes(blob)
    return ev


def create_tampered_ak_binding_tdx_evidence(
    pki: Optional[SyntheticIntelPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    tamper_offset: int = 1084,  # qe_report_body[320]: 764 + 320 = 1084
    tamper_mask: int = 0x01,
) -> AttestationEvidence:
    """Negative Fixture: Alters AK hash embedded in QE report data (simulating AK substitution)."""
    ev = create_valid_intel_tdx_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )
    assert ev.evidence_blob is not None
    blob = bytearray(ev.evidence_blob)
    blob[tamper_offset] ^= tamper_mask
    ev.evidence_blob = bytes(blob)
    return ev


# Alias for create_tampered_ak_binding_tdx_evidence
create_tampered_qe_ak_binding_evidence = create_tampered_ak_binding_tdx_evidence


def create_debug_tdx_evidence(
    pki: Optional[SyntheticIntelPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Quote generated with TDATTRIBUTES.DEBUG == 1 (insecure production enclave)."""
    return create_valid_intel_tdx_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        debug=True,
    )


def create_nonce_mismatch_tdx_evidence(
    requested_nonce: bytes,
    pki: Optional[SyntheticIntelPKI] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Quote embeds stale/mismatched nonce in REPORTDATA (replay attack simulation)."""
    stale_nonce = secrets.token_bytes(32)
    while stale_nonce == requested_nonce:
        stale_nonce = secrets.token_bytes(32)
    return create_valid_intel_tdx_evidence(
        pki=pki,
        nonce=stale_nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )


def create_pubkey_mismatch_tdx_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    pki: Optional[SyntheticIntelPKI] = None,
) -> AttestationEvidence:
    """Negative Fixture: Bound ephemeral pubkey hash in REPORTDATA altered (key substitution simulation)."""
    fake_pubkey = secrets.token_bytes(32)
    return create_valid_intel_tdx_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=fake_pubkey,
        worker_ephemeral_pubkey=fake_pubkey,
    )


def create_measurement_mismatch_tdx_evidence(
    pki: Optional[SyntheticIntelPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: Quote measurement digest does not match Golden RIM."""
    tampered_mrtd = b"\xde\xad\xbe\xef" * 12  # 48 bytes
    ev = create_valid_intel_tdx_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
        mrtd=tampered_mrtd,
        golden_rim=DEFAULT_TDX_GOLDEN_RIM.copy(),
    )
    return ev


def create_untrusted_root_tdx_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
) -> Tuple[AttestationEvidence, str]:
    """Negative Fixture: Certificate chain rooted in an untrusted/rogue Intel Root CA."""
    rogue_pki = generate_synthetic_intel_pki(root_cn="Rogue Untrusted Intel SGX Root CA")
    ev = create_valid_intel_tdx_evidence(
        pki=rogue_pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )
    return ev, rogue_pki.root_cert_pem


def create_expired_cert_tdx_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: PCK certificate chain has expired (not_valid_after in the past)."""
    expired_pki = generate_synthetic_intel_pki(expired=True)
    return create_valid_intel_tdx_evidence(
        pki=expired_pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )


def create_not_yet_valid_cert_tdx_evidence(
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
) -> AttestationEvidence:
    """Negative Fixture: PCK certificate chain not yet valid (not_valid_before in the future)."""
    future_pki = generate_synthetic_intel_pki(not_yet_valid=True)
    return create_valid_intel_tdx_evidence(
        pki=future_pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )


def create_revoked_cert_tdx_evidence(
    pki: Optional[SyntheticIntelPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
) -> Tuple[AttestationEvidence, int]:
    """Negative Fixture: Returns (evidence, pck_serial_number) for CRL/revocation testing."""
    active_pki = pki or get_default_intel_pki()
    ev = create_valid_intel_tdx_evidence(
        pki=active_pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )
    return ev, active_pki.pck_cert.serial_number


def create_truncated_tdx_quote_evidence(
    pki: Optional[SyntheticIntelPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    truncated_length: int = 500,
) -> AttestationEvidence:
    """Negative Fixture: Truncated evidence blob (< 636 bytes)."""
    ev = create_valid_intel_tdx_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )
    assert ev.evidence_blob is not None
    ev.evidence_blob = ev.evidence_blob[:truncated_length]
    return ev


def create_malformed_version_tdx_evidence(
    pki: Optional[SyntheticIntelPKI] = None,
    nonce: Optional[Union[bytes, str]] = None,
    client_ephemeral_pubkey: Optional[bytes] = None,
    worker_ephemeral_pubkey: Optional[bytes] = None,
    invalid_version: int = 99,
) -> AttestationEvidence:
    """Negative Fixture: Quote Header specifies an invalid/unsupported version (e.g. 99)."""
    ev = create_valid_intel_tdx_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=client_ephemeral_pubkey,
        worker_ephemeral_pubkey=worker_ephemeral_pubkey,
    )
    assert ev.evidence_blob is not None
    blob = bytearray(ev.evidence_blob)
    struct.pack_into("<H", blob, 0, invalid_version)
    ev.evidence_blob = bytes(blob)
    return ev


# ============================================================================
# Export Declarations
# ============================================================================

__all__ = [
    # AMD and Common PKI & Measurement Constants
    "DEFAULT_GOLDEN_RIM",
    "DEFAULT_MEASUREMENTS",
    "TAMPERED_MEASUREMENTS",
    "AMD_OID_PREFIX",
    "AMD_OID_STRUCT_VERSION",
    "AMD_OID_PRODUCT_NAME",
    "AMD_OID_BL_SPL",
    "AMD_OID_TEE_SPL",
    "AMD_OID_SNP_SPL",
    "AMD_OID_UCODE_SPL",
    "AMD_OID_HW_ID",
    "DEFAULT_SNP_MEASUREMENT",
    "DEFAULT_SNP_HOST_DATA",
    "DEFAULT_SNP_CHIP_ID",
    "DEFAULT_SNP_POLICY",
    "DEFAULT_SNP_GOLDEN_RIM",
    # Dataclasses & Generators (Common & AMD)
    "CertBundle",
    "SyntheticPKI",
    "SyntheticAmdPKI",
    "canonicalize_claims",
    "generate_key",
    "generate_synthetic_pki",
    "get_default_pki",
    "sign_claims",
    "verify_claims_signature",
    "generate_synthetic_amd_pki",
    "get_default_amd_pki",
    "generate_synthetic_snp_report",
    "create_valid_attestation_evidence",
    "create_valid_sev_snp_evidence",
    "create_altered_nonce_evidence",
    "create_altered_pubkey_hash_evidence",
    "create_broken_chain_signature_evidence",
    "create_expired_cert_evidence",
    "create_leaf_as_ca_evidence",
    "create_malformed_pem_evidence",
    "create_mismatched_measurements_evidence",
    "create_not_yet_valid_cert_evidence",
    "create_tampered_token_signature_evidence",
    "create_untrusted_root_evidence",
    "create_tampered_snp_payload_evidence",
    "create_tampered_snp_signature_evidence",
    "create_debug_policy_snp_evidence",
    "create_nonce_mismatch_snp_evidence",
    "create_pubkey_hash_mismatch_snp_evidence",
    "create_untrusted_ark_snp_evidence",
    "create_expired_vcek_snp_evidence",
    "create_not_yet_valid_vcek_snp_evidence",
    "create_revoked_vcek_snp_evidence",
    "create_truncated_snp_report_evidence",
    "create_measurement_mismatch_snp_evidence",
    "create_tcb_downgrade_snp_evidence",
    "der_encode_integer",
    "der_decode_integer",
    "der_encode_ia5string",
    "der_decode_ia5string",
    "der_encode_octet_string",
    "der_decode_octet_string",
    # Intel TDX Constants & Golden RIM
    "INTEL_QE_VENDOR_ID",
    "INTEL_TDX_TEE_TYPE",
    "INTEL_SGX_EXTENSION_OID",
    "INTEL_OID_PPID",
    "INTEL_OID_TCB",
    "INTEL_OID_PCEID",
    "INTEL_OID_FMSPC",
    "INTEL_OID_SGX_TYPE",
    "INTEL_OID_PLATFORM_INSTANCE_ID",
    "INTEL_OID_CONFIGURATION",
    "INTEL_OID_TCB_COMP_SVNS",
    "INTEL_OID_FMSPC_SPEC_COMPAT",
    "DEFAULT_INTEL_FMSPC",
    "DEFAULT_INTEL_PCEID",
    "DEFAULT_INTEL_TCB_COMP_SVNS",
    "DEFAULT_INTEL_PCE_SVN",
    "DEFAULT_TDX_MRTD",
    "DEFAULT_TDX_RTMR0",
    "DEFAULT_TDX_RTMR1",
    "DEFAULT_TDX_RTMR2",
    "DEFAULT_TDX_RTMR3",
    "DEFAULT_TDX_MRSEAM",
    "DEFAULT_TDX_MRSIGNERSEAM",
    "DEFAULT_TDX_TEE_TCB_SVN",
    "DEFAULT_TDX_GOLDEN_RIM",
    # Intel TDX ASN.1 DER Helpers
    "der_encode_oid",
    "der_decode_oid",
    "der_encode_sequence",
    "der_decode_sequence",
    "encode_intel_pck_asn1_extension",
    "build_intel_pck_extension",
    # Intel TDX PKI Hierarchy & Caching
    "SyntheticIntelPKI",
    "generate_synthetic_intel_pki",
    "get_default_intel_pki",
    "reset_default_intel_pki_cache",
    # Intel TDX Quote & Evidence Generators
    "generate_synthetic_tdx_quote",
    "create_valid_intel_tdx_evidence",
    # Intel TDX Negative / Adversarial Fixtures
    "create_tampered_tdx_quote_evidence",
    "create_tampered_tdx_signature_evidence",
    "create_tampered_qe_signature_evidence",
    "create_tampered_ak_binding_tdx_evidence",
    "create_tampered_qe_ak_binding_evidence",
    "create_debug_tdx_evidence",
    "create_nonce_mismatch_tdx_evidence",
    "create_pubkey_mismatch_tdx_evidence",
    "create_measurement_mismatch_tdx_evidence",
    "create_untrusted_root_tdx_evidence",
    "create_expired_cert_tdx_evidence",
    "create_not_yet_valid_cert_tdx_evidence",
    "create_revoked_cert_tdx_evidence",
    "create_truncated_tdx_quote_evidence",
    "create_malformed_version_tdx_evidence",
]
