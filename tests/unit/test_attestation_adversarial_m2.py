"""
Adversarial Stress Test Suite for matrix-compute-sdk AttestationVerifier (Milestone 2).

Systematically tests fail-closed properties against subtle attacks:
1. 1-bit signature flips across ASN.1 DER structures and certificate signatures.
2. Bit-flipped and truncated Golden RIM measurement digests across all components.
3. Subtle replay attacks with permuted nonces (cyclic shifts, reversals, adjacent swaps, bit flips).
4. MITM ephemeral public key substitutions, 1-bit differences, and missing binding claims.
5. Strict clock skew boundaries (exact threshold, subsecond precision, 1-second past threshold).
6. Certificate chain loops, self-signed rogue roots, ca=False intermediates, and missing extensions.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import secrets
from typing import Any, Dict, List, Optional

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, ExtensionOID, NameOID

from matrix_contracts.schemas.attestation import AttestationEvidence
from matrix_compute.crypto.attestation_verifier import (
    AttestationVerifier,
    VerificationPolicy,
    VerificationResult,
)
from matrix_compute.crypto.pki_fixtures import (
    DEFAULT_GOLDEN_RIM,
    DEFAULT_MEASUREMENTS,
    CertBundle,
    SyntheticPKI,
    canonicalize_claims,
    create_valid_attestation_evidence,
    generate_key,
    generate_synthetic_pki,
    get_default_pki,
    sign_claims,
)
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


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def pki() -> SyntheticPKI:
    return get_default_pki()


@pytest.fixture
def policy(pki: SyntheticPKI) -> VerificationPolicy:
    pol = VerificationPolicy()
    pol.add_trusted_root_pem(pki.root_cert_pem)
    pol.add_golden_rim("simulated-cca", DEFAULT_GOLDEN_RIM)
    pol.add_golden_rim("nvidia-cca", DEFAULT_GOLDEN_RIM)
    pol.allowed_clock_skew_seconds = 60.0
    pol.max_evidence_age_seconds = 300.0
    return pol


@pytest.fixture
def verifier(policy: VerificationPolicy) -> AttestationVerifier:
    return AttestationVerifier(policy=policy)


# ==============================================================================
# DIMENSION 1: SUBTLE CRYPTOGRAPHIC TAMPERING (1-BIT FLIPS & PERMUTATIONS)
# ==============================================================================

def test_adversarial_token_signature_1bit_flip_asn1_header(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: 1-bit flip in byte 0 (ASN.1 DER Sequence tag) of token signature fails closed."""
    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce)

    # Flip bit 0 of first byte (0x30 -> 0x31)
    corrupted_sig = bytearray(ev.token_signature)
    corrupted_sig[0] ^= 0x01
    ev.token_signature = bytes(corrupted_sig)

    with pytest.raises(AttestationSignatureError, match="Cryptographic signature verification failed"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_token_signature_1bit_flip_r_scalar(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: 1-bit flip in the r-scalar value of token signature fails closed."""
    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce)

    # Byte 6 is inside the r-scalar integer payload in SECP384R1 DER signature
    corrupted_sig = bytearray(ev.token_signature)
    corrupted_sig[6] ^= 0x02
    ev.token_signature = bytes(corrupted_sig)

    with pytest.raises(AttestationSignatureError, match="Cryptographic signature verification failed"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_token_signature_1bit_flip_s_scalar(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: 1-bit flip in the s-scalar value (last byte) of token signature fails closed."""
    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce)

    corrupted_sig = bytearray(ev.token_signature)
    corrupted_sig[-1] ^= 0x80
    ev.token_signature = bytes(corrupted_sig)

    with pytest.raises(AttestationSignatureError, match="Cryptographic signature verification failed"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_intermediate_cert_signature_1bit_flip(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: 1-bit flip in intermediate certificate cryptographic signature fails closed."""
    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce)

    # Re-sign intermediate with bad signature bytes
    inter_cert = pki.intermediate.cert
    bad_sig = bytearray(inter_cert.signature)
    bad_sig[10] ^= 0x01

    # Rebuild cert with corrupted signature is represented by signing with wrong key
    wrong_key = generate_key("ecdsa", "secp384r1")
    nvb = getattr(inter_cert, "not_valid_before_utc", getattr(inter_cert, "not_valid_before", None))
    nva = getattr(inter_cert, "not_valid_after_utc", getattr(inter_cert, "not_valid_after", None))
    tampered_inter = (
        x509.CertificateBuilder()
        .subject_name(inter_cert.subject)
        .issuer_name(inter_cert.issuer)
        .public_key(inter_cert.public_key())
        .serial_number(inter_cert.serial_number)
        .not_valid_before(nvb)
        .not_valid_after(nva)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(wrong_key, hashes.SHA384())
    )
    tampered_pem = tampered_inter.public_bytes(serialization.Encoding.PEM).decode("ascii")
    ev.certificate_chain[1] = tampered_pem

    with pytest.raises(AttestationSignatureError, match="certificate signature verification failed"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


@pytest.mark.parametrize("component", ["driver", "vbios", "firmware", "uefi", "cc_mode"])
def test_adversarial_golden_rim_1bit_flip_component_digest(
    verifier: AttestationVerifier,
    pki: SyntheticPKI,
    component: str,
) -> None:
    """Challenge: 1-bit/1-hex alteration in each Golden RIM component digest is detected."""
    nonce = secrets.token_bytes(32)
    meas = DEFAULT_MEASUREMENTS.copy()

    # Flip 1 hex char (last char '9' -> '8' or 'b' -> 'a')
    orig_digest = meas[component]
    flipped_char = "0" if orig_digest[-1] != "0" else "1"
    meas[component] = orig_digest[:-1] + flipped_char

    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce, measurements=meas)

    with pytest.raises(AttestationMeasurementMismatchError, match="Golden RIM measurement appraisal failed") as exc_info:
        verifier.verify(evidence=ev, expected_nonce=nonce)

    assert component in exc_info.value.mismatches


def test_adversarial_golden_rim_truncated_digest(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Truncated digest string (62 instead of 64 hex characters) is detected."""
    nonce = secrets.token_bytes(32)
    meas = DEFAULT_MEASUREMENTS.copy()
    meas["driver"] = meas["driver"][:62]

    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce, measurements=meas)

    with pytest.raises(AttestationMeasurementMismatchError, match="Golden RIM measurement appraisal failed"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_golden_rim_empty_measurements(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Empty measurements dictionary fails closed with all missing claims."""
    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce, measurements={})

    with pytest.raises(AttestationMeasurementMismatchError, match="missing"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_nonce_cyclic_permutation(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Replay attack where nonce bytes are cyclic-shifted by 1 byte fails closed."""
    expected_nonce = secrets.token_bytes(32)
    permuted_nonce = expected_nonce[1:] + expected_nonce[:1]

    ev = create_valid_attestation_evidence(pki=pki, nonce=permuted_nonce)

    with pytest.raises(AttestationNonceMismatchError, match="Challenger nonce mismatch") as exc_info:
        verifier.verify(evidence=ev, expected_nonce=expected_nonce)

    assert exc_info.value.expected_nonce == expected_nonce.hex().lower()
    assert exc_info.value.actual_nonce == permuted_nonce.hex().lower()


def test_adversarial_nonce_reversed(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Replay attack where nonce byte order is reversed fails closed."""
    expected_nonce = secrets.token_bytes(32)
    reversed_nonce = bytes(reversed(expected_nonce))

    ev = create_valid_attestation_evidence(pki=pki, nonce=reversed_nonce)

    with pytest.raises(AttestationNonceMismatchError, match="Challenger nonce mismatch"):
        verifier.verify(evidence=ev, expected_nonce=expected_nonce)


def test_adversarial_nonce_adjacent_byte_swap(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Replay attack with transposed adjacent bytes (bytes 0 and 1 swapped) fails closed."""
    expected_nonce = secrets.token_bytes(32)
    swapped_nonce = bytes([expected_nonce[1], expected_nonce[0]]) + expected_nonce[2:]

    ev = create_valid_attestation_evidence(pki=pki, nonce=swapped_nonce)

    with pytest.raises(AttestationNonceMismatchError, match="Challenger nonce mismatch"):
        verifier.verify(evidence=ev, expected_nonce=expected_nonce)


def test_adversarial_nonce_1bit_flip(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Replay attack where exactly 1 bit is flipped in the 256-bit nonce fails closed."""
    expected_nonce = secrets.token_bytes(32)
    corrupted_nonce = bytearray(expected_nonce)
    corrupted_nonce[0] ^= 0x01

    ev = create_valid_attestation_evidence(pki=pki, nonce=bytes(corrupted_nonce))

    with pytest.raises(AttestationNonceMismatchError, match="Challenger nonce mismatch"):
        verifier.verify(evidence=ev, expected_nonce=expected_nonce)


def test_adversarial_nonce_format_normalization(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Verifier correctly normalizes 0x prefix and uppercase hex without false positives."""
    raw_nonce = secrets.token_bytes(32)
    hex_nonce = raw_nonce.hex()

    # Upper case hex with 0x prefix in expected_nonce
    expected_upper = "0x" + hex_nonce.upper()

    ev = create_valid_attestation_evidence(pki=pki, nonce=raw_nonce)
    result = verifier.verify(evidence=ev, expected_nonce=expected_upper)
    assert result.verified is True


# ==============================================================================
# DIMENSION 2: MITM CLIENT PUBLIC KEY SUBSTITUTIONS
# ==============================================================================

def test_adversarial_mitm_client_pubkey_substitution(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Valid attestation evidence bound to attacker's pubkey is rejected."""
    nonce = secrets.token_bytes(32)
    client_pubkey = secrets.token_bytes(32)
    attacker_pubkey = secrets.token_bytes(32)

    # Attestation evidence legitimately signed by enclave, but binding attacker's pubkey
    ev = create_valid_attestation_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=attacker_pubkey,
    )

    with pytest.raises(AttestationBindingError, match="Client ephemeral public key binding mismatch") as exc_info:
        verifier.verify(
            evidence=ev,
            expected_nonce=nonce,
            expected_client_pubkey=client_pubkey,
        )

    expected_hash = hashlib.sha256(client_pubkey).hexdigest().lower()
    assert exc_info.value.expected_pubkey_hash == expected_hash


def test_adversarial_mitm_client_pubkey_1bit_difference(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: 1-bit difference in expected vs bound client pubkey triggers binding failure."""
    nonce = secrets.token_bytes(32)
    client_pubkey = secrets.token_bytes(32)
    subtle_tampered_pubkey = bytearray(client_pubkey)
    subtle_tampered_pubkey[0] ^= 0x01

    ev = create_valid_attestation_evidence(
        pki=pki,
        nonce=nonce,
        client_ephemeral_pubkey=bytes(subtle_tampered_pubkey),
    )

    with pytest.raises(AttestationBindingError, match="Client ephemeral public key binding mismatch"):
        verifier.verify(
            evidence=ev,
            expected_nonce=nonce,
            expected_client_pubkey=client_pubkey,
        )


def test_adversarial_mitm_missing_binding_claims(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Evidence missing ephemeral_pubkey_hash or user_data binding fails closed."""
    nonce = secrets.token_bytes(32)
    client_pubkey = secrets.token_bytes(32)

    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce)
    ev.token_claims.pop("client_pubkey_hash", None)
    ev.token_claims.pop("ephemeral_pubkey_hash", None)
    ev.token_claims.pop("user_data", None)
    # Re-sign stripped claims
    ev.token_signature = sign_claims(ev.token_claims, pki.leaf.key)

    with pytest.raises(AttestationBindingError, match="missing 'ephemeral_pubkey_hash' binding claim"):
        verifier.verify(
            evidence=ev,
            expected_nonce=nonce,
            expected_client_pubkey=client_pubkey,
        )


# ==============================================================================
# DIMENSION 3: CLOCK SKEW & TEMPORAL BOUNDARIES
# ==============================================================================

def test_adversarial_clock_skew_exact_future_boundary(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Timestamp exactly at now + allowed_clock_skew_seconds (60.0s) is valid."""
    nonce = secrets.token_bytes(32)
    now = datetime.datetime.now(datetime.timezone.utc)
    exact_future_ts = now.timestamp() + 60.0

    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce, timestamp=int(exact_future_ts))
    result = verifier.verify(evidence=ev, expected_nonce=nonce, verification_time=now)
    assert result.verified is True


def test_adversarial_clock_skew_1second_past_future_boundary(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Timestamp at now + 61.0s (1s past allowed skew) raises AttestationExpiredError."""
    nonce = secrets.token_bytes(32)
    now = datetime.datetime.now(datetime.timezone.utc)
    future_ts = now.timestamp() + 61.0

    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce, timestamp=int(future_ts))

    with pytest.raises(AttestationExpiredError, match="timestamp is in the future"):
        verifier.verify(evidence=ev, expected_nonce=nonce, verification_time=now)


def test_adversarial_clock_skew_subsecond_precision(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Sub-second boundary precision (+60.001s fails, +59.999s passes)."""
    nonce = secrets.token_bytes(32)
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. 1 millisecond past threshold
    ts_over = now.timestamp() + 60.001
    ev_over = create_valid_attestation_evidence(pki=pki, nonce=nonce)
    ev_over.token_claims["timestamp"] = ts_over
    ev_over.token_signature = sign_claims(ev_over.token_claims, pki.leaf.key)

    with pytest.raises(AttestationExpiredError, match="timestamp is in the future"):
        verifier.verify(evidence=ev_over, expected_nonce=nonce, verification_time=now)

    # 2. 1 millisecond under threshold
    ts_under = now.timestamp() + 59.999
    ev_under = create_valid_attestation_evidence(pki=pki, nonce=nonce)
    ev_under.token_claims["timestamp"] = ts_under
    ev_under.token_signature = sign_claims(ev_under.token_claims, pki.leaf.key)

    res = verifier.verify(evidence=ev_under, expected_nonce=nonce, verification_time=now)
    assert res.verified is True


def test_adversarial_clock_skew_exact_stale_boundary(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Age exactly at max_evidence_age_seconds (300.0s) is valid."""
    nonce = secrets.token_bytes(32)
    now = datetime.datetime.now(datetime.timezone.utc)
    exact_stale_ts = now.timestamp() - 300.0

    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce, timestamp=exact_stale_ts)
    result = verifier.verify(evidence=ev, expected_nonce=nonce, verification_time=now)
    assert result.verified is True


def test_adversarial_clock_skew_1second_past_stale_boundary(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Age at 301.0s (1s past max age) raises AttestationExpiredError."""
    nonce = secrets.token_bytes(32)
    now = datetime.datetime.now(datetime.timezone.utc)
    stale_ts = now.timestamp() - 301.0

    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce, timestamp=int(stale_ts))

    with pytest.raises(AttestationExpiredError, match="Attestation evidence is stale"):
        verifier.verify(evidence=ev, expected_nonce=nonce, verification_time=now)


def test_adversarial_malformed_timestamp_string(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Non-numeric timestamp in claims raises AttestationEvidenceMalformedError."""
    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce)
    ev.token_claims["timestamp"] = "INVALID_TIMESTAMP_STRING"
    ev.token_signature = sign_claims(ev.token_claims, pki.leaf.key)

    with pytest.raises(AttestationEvidenceMalformedError, match="Malformed timestamp"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


# ==============================================================================
# DIMENSION 4: CERTIFICATE CHAINS, ROGUE ROOTS & BASIC CONSTRAINTS
# ==============================================================================

def test_adversarial_self_signed_rogue_root(verifier: AttestationVerifier) -> None:
    """Challenge: Self-signed rogue root CA not in trust store raises AttestationUntrustedRootError."""
    rogue_pki = generate_synthetic_pki(root_cn="Adversarial Rogue Root CA")
    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(pki=rogue_pki, nonce=nonce)

    with pytest.raises(AttestationUntrustedRootError, match="cannot be anchored to any trusted Root CA"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_rogue_root_mimicking_trusted_subject(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Rogue root mimicking genuine Root Subject name with different key is rejected."""
    genuine_root_subject = pki.root.cert.subject
    now = datetime.datetime.now(datetime.timezone.utc)
    rogue_key = generate_key("ecdsa", "secp384r1")

    # Build rogue root with exact same Subject name but rogue key
    rogue_root = (
        x509.CertificateBuilder()
        .subject_name(genuine_root_subject)
        .issuer_name(genuine_root_subject)
        .public_key(rogue_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=2), critical=True)
        .sign(rogue_key, hashes.SHA384())
    )

    # Rogue intermediate and leaf signed by rogue key
    inter_key = generate_key("ecdsa", "secp384r1")
    inter_sub = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Rogue Inter")])
    rogue_inter = (
        x509.CertificateBuilder()
        .subject_name(inter_sub)
        .issuer_name(genuine_root_subject)
        .public_key(inter_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(rogue_key, hashes.SHA384())
    )

    leaf_key = generate_key("ecdsa", "secp384r1")
    leaf_sub = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Rogue Leaf")])
    rogue_leaf = (
        x509.CertificateBuilder()
        .subject_name(leaf_sub)
        .issuer_name(inter_sub)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(inter_key, hashes.SHA384())
    )

    chain = [
        rogue_leaf.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        rogue_inter.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        rogue_root.public_bytes(serialization.Encoding.PEM).decode("ascii"),
    ]

    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(nonce=nonce)
    ev.certificate_chain = chain
    ev.token_signature = sign_claims(ev.token_claims, leaf_key)

    with pytest.raises(AttestationUntrustedRootError, match="cannot be anchored to any trusted Root CA"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_intermediate_ca_flag_false(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Intermediate certificate configured with ca=False raises AttestationChainInvalidError."""
    now = datetime.datetime.now(datetime.timezone.utc)
    root_bundle = pki.root

    inter_key = generate_key("ecdsa", "secp384r1")
    inter_sub = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Invalid Inter CA False")])
    bad_inter = (
        x509.CertificateBuilder()
        .subject_name(inter_sub)
        .issuer_name(root_bundle.cert.subject)
        .public_key(inter_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)  # ca=False
        .sign(root_bundle.key, hashes.SHA384())
    )

    leaf_key = generate_key("ecdsa", "secp384r1")
    leaf_sub = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Leaf")])
    leaf = (
        x509.CertificateBuilder()
        .subject_name(leaf_sub)
        .issuer_name(inter_sub)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(inter_key, hashes.SHA384())
    )

    chain = [
        leaf.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        bad_inter.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        root_bundle.cert_pem,
    ]

    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(nonce=nonce)
    ev.certificate_chain = chain
    ev.token_signature = sign_claims(ev.token_claims, leaf_key)

    with pytest.raises(AttestationChainInvalidError, match="is not a CA"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_intermediate_missing_basic_constraints(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Intermediate certificate omitting BasicConstraints extension raises AttestationChainInvalidError."""
    now = datetime.datetime.now(datetime.timezone.utc)
    root_bundle = pki.root

    inter_key = generate_key("ecdsa", "secp384r1")
    inter_sub = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Missing BC Inter CA")])
    bad_inter = (
        x509.CertificateBuilder()
        .subject_name(inter_sub)
        .issuer_name(root_bundle.cert.subject)
        .public_key(inter_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        # Omit BasicConstraints entirely
        .sign(root_bundle.key, hashes.SHA384())
    )

    leaf_key = generate_key("ecdsa", "secp384r1")
    leaf_sub = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Leaf")])
    leaf = (
        x509.CertificateBuilder()
        .subject_name(leaf_sub)
        .issuer_name(inter_sub)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(inter_key, hashes.SHA384())
    )

    chain = [
        leaf.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        bad_inter.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        root_bundle.cert_pem,
    ]

    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(nonce=nonce)
    ev.certificate_chain = chain
    ev.token_signature = sign_claims(ev.token_claims, leaf_key)

    with pytest.raises(AttestationChainInvalidError, match="missing BasicConstraints extension"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_chain_loop_repeated_leaf(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Chain with repeated leaf [Leaf, Leaf, Root] raises AttestationChainInvalidError."""
    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce)
    ev.certificate_chain = [pki.leaf_cert_pem, pki.leaf_cert_pem, pki.root_cert_pem]

    with pytest.raises(AttestationChainInvalidError):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_chain_loop_repeated_intermediate(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Chain with duplicate intermediate [Leaf, Inter, Inter, Root] raises AttestationChainInvalidError."""
    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce)
    ev.certificate_chain = [
        pki.leaf_cert_pem,
        pki.intermediate_cert_pem,
        pki.intermediate_cert_pem,
        pki.root_cert_pem,
    ]

    with pytest.raises(AttestationChainInvalidError, match="Certificate chain broken at index 1"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_chain_loop_mutual_ca_cycle(verifier: AttestationVerifier, pki: SyntheticPKI) -> None:
    """Challenge: Mutual intermediate CA cycle [Leaf, CA_A, CA_B, CA_A] raises AttestationUntrustedRootError."""
    now = datetime.datetime.now(datetime.timezone.utc)
    key_a = generate_key("ecdsa", "secp384r1")
    key_b = generate_key("ecdsa", "secp384r1")
    sub_a = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CA Alpha")])
    sub_b = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CA Beta")])

    # CA Alpha signed by CA Beta
    ca_a = (
        x509.CertificateBuilder()
        .subject_name(sub_a)
        .issuer_name(sub_b)
        .public_key(key_a.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .sign(key_b, hashes.SHA384())
    )

    # CA Beta signed by CA Alpha (cycle)
    ca_b = (
        x509.CertificateBuilder()
        .subject_name(sub_b)
        .issuer_name(sub_a)
        .public_key(key_b.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .sign(key_a, hashes.SHA384())
    )

    # Leaf signed by CA Alpha
    leaf_key = generate_key("ecdsa", "secp384r1")
    leaf_sub = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Leaf Cert")])
    leaf = (
        x509.CertificateBuilder()
        .subject_name(leaf_sub)
        .issuer_name(sub_a)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key_a, hashes.SHA384())
    )

    pem_leaf = leaf.public_bytes(serialization.Encoding.PEM).decode("ascii")
    pem_a = ca_a.public_bytes(serialization.Encoding.PEM).decode("ascii")
    pem_b = ca_b.public_bytes(serialization.Encoding.PEM).decode("ascii")

    chain = [pem_leaf, pem_a, pem_b, pem_a]

    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(nonce=nonce)
    ev.certificate_chain = chain
    ev.token_signature = sign_claims(ev.token_claims, leaf_key)

    with pytest.raises(AttestationUntrustedRootError, match="cannot be anchored to any trusted Root CA"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_adversarial_empty_trusted_roots_store(pki: SyntheticPKI) -> None:
    """Challenge: Verifier with empty trust store fails closed immediately."""
    empty_policy = VerificationPolicy()
    empty_verifier = AttestationVerifier(policy=empty_policy)

    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(pki=pki, nonce=nonce)

    with pytest.raises(AttestationUntrustedRootError, match="zero trusted Root CAs configured"):
        empty_verifier.verify(evidence=ev, expected_nonce=nonce)
