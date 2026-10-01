"""
Comprehensive unit test suite for matrix-compute-sdk AttestationVerifier engine (Milestone 2).
Exhaustively tests fail-closed verification, X.509 cert chain validation, Golden RIM appraisal,
replay/MITM protection, expiration/freshness, and pluggable provider strategies.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import secrets
from typing import Any, Dict

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from matrix_contracts.schemas.attestation import AttestationEvidence
from matrix_compute.crypto.attestation_verifier import (
    AttestationVerifier,
    BaseEvidenceVerifier,
    NvidiaCcaVerifier,
    SimulatedCcaVerifier,
    VerificationPolicy,
    VerificationResult,
)
from matrix_compute.crypto.pki_fixtures import (
    DEFAULT_GOLDEN_RIM,
    DEFAULT_MEASUREMENTS,
    TAMPERED_MEASUREMENTS,
    create_altered_nonce_evidence,
    create_altered_pubkey_hash_evidence,
    create_broken_chain_signature_evidence,
    create_expired_cert_evidence,
    create_leaf_as_ca_evidence,
    create_malformed_pem_evidence,
    create_mismatched_measurements_evidence,
    create_not_yet_valid_cert_evidence,
    create_tampered_token_signature_evidence,
    create_untrusted_root_evidence,
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


@pytest.fixture
def default_pki():
    return get_default_pki()


@pytest.fixture
def default_policy(default_pki):
    policy = VerificationPolicy()
    policy.add_trusted_root_pem(default_pki.root_cert_pem)
    policy.add_golden_rim("simulated-cca", DEFAULT_GOLDEN_RIM)
    policy.add_golden_rim("nvidia-cca", DEFAULT_GOLDEN_RIM)
    return policy


@pytest.fixture
def verifier(default_policy):
    return AttestationVerifier(policy=default_policy)


def test_valid_attestation_verification_success(verifier, default_pki) -> None:
    """TC-ATTEST-01: Valid attestation evidence verifies successfully."""
    nonce = secrets.token_bytes(32)
    client_pubkey = secrets.token_bytes(32)
    worker_pubkey = secrets.token_bytes(32)

    evidence = create_valid_attestation_evidence(
        pki=default_pki,
        nonce=nonce,
        worker_ephemeral_pubkey=worker_pubkey,
        client_ephemeral_pubkey=client_pubkey,
        provider="simulated-cca",
    )

    result = verifier.verify(
        evidence=evidence,
        expected_nonce=nonce,
        expected_client_pubkey=client_pubkey,
    )

    assert isinstance(result, VerificationResult)
    assert result.verified is True
    assert result.provider == "simulated-cca"
    assert result.chain_length == 3
    assert result.measurements_verified == len(DEFAULT_GOLDEN_RIM)
    assert result.leaf_cert is not None
    assert result.root_cert is not None


def test_malformed_evidence_bytes(verifier) -> None:
    """TC-ATTEST-02: Non-JSON or truncated bytes raise AttestationEvidenceMalformedError."""
    nonce = secrets.token_bytes(32)
    with pytest.raises(AttestationEvidenceMalformedError):
        verifier.verify(evidence=b"GARBAGE_NON_JSON_DATA", expected_nonce=nonce)


def test_empty_certificate_chain(verifier) -> None:
    """TC-ATTEST-03: Empty certificate chain raises AttestationEvidenceMalformedError."""
    nonce = secrets.token_bytes(32)
    evidence = create_valid_attestation_evidence(nonce=nonce)
    evidence.certificate_chain = []

    with pytest.raises(AttestationEvidenceMalformedError, match="Certificate chain is empty"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_malformed_x509_certificate_pem(verifier) -> None:
    """TC-ATTEST-04: Damaged PEM certificate strings raise AttestationCertificateError."""
    nonce = secrets.token_bytes(32)
    evidence = create_malformed_pem_evidence(nonce=nonce)

    with pytest.raises(AttestationCertificateError, match="Failed to parse PEM certificate"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_broken_chain_issuer_subject_mismatch(verifier) -> None:
    """TC-ATTEST-05: Broken chain where child issuer != parent subject raises AttestationChainInvalidError."""
    nonce = secrets.token_bytes(32)
    evidence = create_broken_chain_signature_evidence(nonce=nonce)

    with pytest.raises((AttestationChainInvalidError, AttestationSignatureError)):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_intermediate_ca_extension_missing(verifier) -> None:
    """TC-ATTEST-06: Intermediate certificate missing CA=True raises AttestationChainInvalidError."""
    # Generate PKI where intermediate has ca=False
    now = datetime.datetime.now(datetime.timezone.utc)
    root_key = generate_key("ecdsa", "secp384r1")
    root_sub = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "Test Root CA")])
    root_cert = (
        x509.CertificateBuilder()
        .subject_name(root_sub)
        .issuer_name(root_sub)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=2), critical=True)
        .sign(root_key, hashes.SHA384())
    )

    bad_inter_key = generate_key("ecdsa", "secp384r1")
    bad_inter_sub = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "Bad Inter CA")])
    bad_inter_cert = (
        x509.CertificateBuilder()
        .subject_name(bad_inter_sub)
        .issuer_name(root_sub)
        .public_key(bad_inter_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)  # ca=False violation
        .sign(root_key, hashes.SHA384())
    )

    leaf_key = generate_key("ecdsa", "secp384r1")
    leaf_sub = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "Leaf Cert")])
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_sub)
        .issuer_name(bad_inter_sub)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(bad_inter_key, hashes.SHA384())
    )

    chain = [
        leaf_cert.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        bad_inter_cert.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        root_cert.public_bytes(serialization.Encoding.PEM).decode("ascii"),
    ]

    nonce = secrets.token_bytes(32)
    ev = create_valid_attestation_evidence(nonce=nonce)
    ev.certificate_chain = chain
    verifier.policy.trusted_roots.append(root_cert)

    with pytest.raises(AttestationChainInvalidError, match="is not a CA"):
        verifier.verify(evidence=ev, expected_nonce=nonce)


def test_leaf_cannot_be_ca(verifier) -> None:
    """TC-ATTEST-07: Leaf certificate configured with CA=True raises AttestationChainInvalidError."""
    nonce = secrets.token_bytes(32)
    evidence = create_leaf_as_ca_evidence(nonce=nonce)

    with pytest.raises(AttestationChainInvalidError, match="Leaf certificate cannot have CA flag enabled"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_untrusted_root_ca(verifier) -> None:
    """TC-ATTEST-08: Certificate chain from an unrecognized root CA raises AttestationUntrustedRootError."""
    nonce = secrets.token_bytes(32)
    evidence, _ = create_untrusted_root_evidence(nonce=nonce)

    with pytest.raises(AttestationUntrustedRootError, match="cannot be anchored to any trusted Root CA"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_expired_certificate_in_chain(verifier) -> None:
    """TC-ATTEST-09: Expired certificate raises AttestationExpiredError."""
    nonce = secrets.token_bytes(32)
    evidence = create_expired_cert_evidence(nonce=nonce)

    with pytest.raises(AttestationExpiredError, match="has expired"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_not_yet_valid_certificate_in_chain(verifier) -> None:
    """TC-ATTEST-10: Not-yet-valid future certificate raises AttestationExpiredError."""
    nonce = secrets.token_bytes(32)
    evidence = create_not_yet_valid_cert_evidence(nonce=nonce)

    with pytest.raises(AttestationExpiredError, match="not yet valid"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_revoked_certificate_serial(verifier, default_pki) -> None:
    """TC-ATTEST-11: Revoked certificate serial raises AttestationRevokedError."""
    nonce = secrets.token_bytes(32)
    evidence = create_valid_attestation_evidence(pki=default_pki, nonce=nonce)

    # Add leaf cert serial number to revocation list
    leaf_cert = default_pki.leaf.cert
    verifier.policy.revoked_serial_numbers.add(leaf_cert.serial_number)

    with pytest.raises(AttestationRevokedError, match="is revoked"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_tampered_certificate_signature(verifier, default_pki) -> None:
    """TC-ATTEST-11b: Tampered certificate signature raises AttestationSignatureError."""
    nonce = secrets.token_bytes(32)
    evidence = create_valid_attestation_evidence(pki=default_pki, nonce=nonce)
    rogue_key = generate_key("ecdsa", "secp384r1")
    leaf_cert = default_pki.leaf.cert
    tampered_leaf = (
        x509.CertificateBuilder()
        .subject_name(leaf_cert.subject)
        .issuer_name(leaf_cert.issuer)
        .public_key(leaf_cert.public_key())
        .serial_number(leaf_cert.serial_number)
        .not_valid_before(leaf_cert.not_valid_before_utc)
        .not_valid_after(leaf_cert.not_valid_after_utc)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(rogue_key, hashes.SHA384())
    )
    tampered_leaf_pem = tampered_leaf.public_bytes(serialization.Encoding.PEM).decode("ascii")
    evidence.certificate_chain[0] = tampered_leaf_pem

    with pytest.raises(AttestationSignatureError, match="signature verification failed"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_tampered_token_signature(verifier, default_pki) -> None:
    """TC-ATTEST-12: Corrupted token signature bytes raise AttestationSignatureError."""
    nonce = secrets.token_bytes(32)
    evidence = create_tampered_token_signature_evidence(nonce=nonce)

    with pytest.raises(AttestationSignatureError, match="Cryptographic signature verification failed"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_nonce_mismatch_replay_attack(verifier, default_pki) -> None:
    """TC-ATTEST-13: Altered nonce (replay attack simulation) raises AttestationNonceMismatchError."""
    expected_nonce = secrets.token_bytes(32)
    evidence = create_altered_nonce_evidence(requested_nonce=expected_nonce)

    with pytest.raises(AttestationNonceMismatchError, match="Challenger nonce mismatch"):
        verifier.verify(evidence=evidence, expected_nonce=expected_nonce)


def test_client_pubkey_binding_mismatch_mitm(verifier, default_pki) -> None:
    """TC-ATTEST-14: Ephemeral public key binding mismatch raises AttestationBindingError."""
    nonce = secrets.token_bytes(32)
    expected_client_pubkey = secrets.token_bytes(32)

    # Evidence generated bound to a different public key
    evidence = create_altered_pubkey_hash_evidence(nonce=nonce)

    with pytest.raises(AttestationBindingError, match="Client ephemeral public key binding mismatch"):
        verifier.verify(
            evidence=evidence,
            expected_nonce=nonce,
            expected_client_pubkey=expected_client_pubkey,
        )


def test_stale_evidence_timestamp(verifier, default_pki) -> None:
    """TC-ATTEST-15: Evidence timestamp older than max_evidence_age raises AttestationExpiredError."""
    nonce = secrets.token_bytes(32)
    now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    stale_ts = now_ts - 600  # 10 minutes old (limit is 300s)

    evidence = create_valid_attestation_evidence(
        pki=default_pki,
        nonce=nonce,
        timestamp=stale_ts,
    )

    with pytest.raises(AttestationExpiredError, match="Attestation evidence is stale"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_future_evidence_timestamp(verifier, default_pki) -> None:
    """TC-ATTEST-16: Evidence timestamp ahead of allowed clock skew raises AttestationExpiredError."""
    nonce = secrets.token_bytes(32)
    now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    future_ts = now_ts + 120  # 2 minutes in future (skew limit is 60s)

    evidence = create_valid_attestation_evidence(
        pki=default_pki,
        nonce=nonce,
        timestamp=future_ts,
    )

    with pytest.raises(AttestationExpiredError, match="timestamp is in the future"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_golden_rim_altered_digest(verifier, default_pki) -> None:
    """TC-ATTEST-17: Measured digest not matching Golden RIM raises AttestationMeasurementMismatchError."""
    nonce = secrets.token_bytes(32)
    evidence = create_mismatched_measurements_evidence(nonce=nonce, tampered_component="driver")

    with pytest.raises(AttestationMeasurementMismatchError, match="Golden RIM measurement appraisal failed"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_golden_rim_missing_measurement_claim(verifier, default_pki) -> None:
    """TC-ATTEST-18: Claims dictionary missing a required RIM key raises AttestationMeasurementMismatchError."""
    nonce = secrets.token_bytes(32)
    meas = DEFAULT_MEASUREMENTS.copy()
    meas.pop("driver")  # Omit required component

    evidence = create_valid_attestation_evidence(
        pki=default_pki,
        nonce=nonce,
        measurements=meas,
    )

    with pytest.raises(AttestationMeasurementMismatchError, match="missing"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_unsupported_provider_rejected(verifier, default_pki) -> None:
    """TC-ATTEST-19: Unknown attestation provider raises AttestationProviderUnsupportedError."""
    nonce = secrets.token_bytes(32)
    evidence = create_valid_attestation_evidence(pki=default_pki, nonce=nonce)
    evidence.provider = "unknown-nonexistent-tee"

    with pytest.raises(AttestationProviderUnsupportedError, match="Unsupported or unauthorized attestation provider"):
        verifier.verify(evidence=evidence, expected_nonce=nonce)


def test_pluggable_custom_verifier_registration(default_policy, default_pki) -> None:
    """TC-ATTEST-20: Custom BaseEvidenceVerifier strategy can be registered and executed."""
    class CustomEnclaveVerifier(BaseEvidenceVerifier):
        @property
        def provider_id(self) -> str:
            return "custom-enclave"

        def verify_token_signature(self, evidence: AttestationEvidence, leaf_cert: x509.Certificate) -> None:
            # Custom token verification
            pass

    verifier = AttestationVerifier(policy=default_policy)
    verifier.register_provider(CustomEnclaveVerifier())

    nonce = secrets.token_bytes(32)
    evidence = create_valid_attestation_evidence(
        pki=default_pki,
        nonce=nonce,
        provider="custom-enclave",
    )

    result = verifier.verify(evidence=evidence, expected_nonce=nonce)
    assert result.verified is True
    assert result.provider == "custom-enclave"
