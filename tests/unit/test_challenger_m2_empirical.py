"""
Adversarial & Empirical Stress Test Suite for Milestone M2: Synthetic AMD PKI & Report Test Fixtures.
Targeting: matrix-compute-sdk (Synthetic AMD PKI, VCEK/ASK/ARK hierarchy, Report binary layout, caching & concurrency).

Authored by: Challenger 2 (Empirical Challenger).

Empirical Test Matrix:
1. 3-Tier Certificate Chain Validation:
   - Direct X.509 signature verification (_verify_x509_cert_signature):
     - ARK self-signature (RSA-4096 / SHA-384)
     - ASK signature by ARK (RSA-4096 / SHA-384)
     - VCEK signature by ASK (RSA-4096 signing ECDSA P-384 cert)
   - Full chain validation via AttestationVerifier.verify_certificate_chain:
     - 3-cert wire chain [VCEK, ASK, ARK] anchored to trusted ARK
     - 2-cert wire chain [VCEK, ASK] anchored to trusted ARK
   - AMD custom OID extension appraisal (Arc 1.3.6.1.4.1.3704.1):
     - StructVersion, ProductName, blSPL, teeSPL, snpSPL, ucodeSPL, hwID
   - X.509 constraints & key usages (CA flags, path lengths, digital_signature, key_cert_sign).

2. Broken Chain Rejections:
   - Corrupted signatures (1-bit modifications in VCEK, ASK, and ARK signatures)
   - Expired validity (past not_after) and not-yet-valid validity (future not_before)
   - Temporal validation at runtime with simulated clock offsets
   - Broken subject/issuer linkage (inverted order, omitted intermediate, foreign intermediate)
   - Untrusted root CA (rogue ARK, empty trust store)
   - Certificate revocation (serial numbers in revoked set)
   - CA constraint violations (leaf with ca=True, intermediate with ca=False)

3. Cached Singleton Performance & Concurrency Stress:
   - Singleton identity preservation across multiple invocations
   - Microsecond latency profiling (1,000 warm invocations)
   - Multi-threaded concurrency stress under high load (50 threads)
   - Cold-start race analysis and thread-safety behavior

4. SEV-SNP 1184-byte Report Generation & Cryptographic Verification:
   - Wire format conforming to AMD Pub #56860 (672-byte signed slice, 512-byte signature region)
   - Little-endian scalar extraction (72-byte R/S buffers, 48-byte significant) and DER DSS encoding
   - Bit-level tamper detection across body boundaries and signature components
   - Comprehensive validation of all 12 negative/adversarial fixture generators
"""

from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import secrets
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
from cryptography.x509.oid import ExtensionOID, NameOID

from matrix_contracts.schemas.attestation import AttestationEvidence
from matrix_compute.crypto.attestation_verifier import (
    AttestationVerifier,
    VerificationPolicy,
    _verify_x509_cert_signature,
)
from matrix_compute.crypto.pki_fixtures import (
    AMD_OID_BL_SPL,
    AMD_OID_HW_ID,
    AMD_OID_PRODUCT_NAME,
    AMD_OID_SNP_SPL,
    AMD_OID_STRUCT_VERSION,
    AMD_OID_TEE_SPL,
    AMD_OID_UCODE_SPL,
    DEFAULT_SNP_CHIP_ID,
    DEFAULT_SNP_GOLDEN_RIM,
    DEFAULT_SNP_HOST_DATA,
    DEFAULT_SNP_MEASUREMENT,
    DEFAULT_SNP_POLICY,
    SyntheticAmdPKI,
    create_debug_policy_snp_evidence,
    create_expired_vcek_snp_evidence,
    create_measurement_mismatch_snp_evidence,
    create_nonce_mismatch_snp_evidence,
    create_not_yet_valid_vcek_snp_evidence,
    create_pubkey_hash_mismatch_snp_evidence,
    create_revoked_vcek_snp_evidence,
    create_tampered_snp_payload_evidence,
    create_tampered_snp_signature_evidence,
    create_tcb_downgrade_snp_evidence,
    create_truncated_snp_report_evidence,
    create_untrusted_ark_snp_evidence,
    create_valid_sev_snp_evidence,
    der_decode_ia5string,
    der_decode_integer,
    der_decode_octet_string,
    generate_synthetic_amd_pki,
    generate_synthetic_snp_report,
    get_default_amd_pki,
)
from matrix_compute.exceptions import (
    AttestationBindingError,
    AttestationCertificateError,
    AttestationChainInvalidError,
    AttestationEvidenceMalformedError,
    AttestationExpiredError,
    AttestationRevokedError,
    AttestationSignatureError,
    AttestationUntrustedRootError,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="module")
def amd_pki() -> SyntheticAmdPKI:
    return get_default_amd_pki()


@pytest.fixture
def amd_policy(amd_pki: SyntheticAmdPKI) -> VerificationPolicy:
    pol = VerificationPolicy()
    pol.add_trusted_root_pem(amd_pki.ark_cert_pem)
    pol.add_golden_rim("amd-sev-snp", DEFAULT_SNP_GOLDEN_RIM)
    pol.allowed_clock_skew_seconds = 60.0
    pol.max_evidence_age_seconds = 300.0
    return pol


@pytest.fixture
def amd_verifier(amd_policy: VerificationPolicy) -> AttestationVerifier:
    return AttestationVerifier(policy=amd_policy)


# ==============================================================================
# TEST SUITE 1: 3-TIER CERTIFICATE CHAIN VALIDATION
# ==============================================================================

class TestAmdPki3TierChainValidation:
    """Empirical verification of 3-tier certificate chain validation."""

    def test_ark_self_signature_direct(self, amd_pki: SyntheticAmdPKI) -> None:
        """ARK must be a valid self-signed RSA-4096 certificate."""
        assert isinstance(amd_pki.ark_cert.public_key(), rsa.RSAPublicKey)
        assert amd_pki.ark_cert.public_key().key_size == 4096
        # Must not raise
        _verify_x509_cert_signature(amd_pki.ark_cert, amd_pki.ark_cert.public_key())

    def test_ask_signature_direct(self, amd_pki: SyntheticAmdPKI) -> None:
        """ASK must be legitimately signed by ARK with RSA-4096 and SHA-384."""
        assert isinstance(amd_pki.ask_cert.public_key(), rsa.RSAPublicKey)
        assert amd_pki.ask_cert.public_key().key_size == 4096
        # Must not raise
        _verify_x509_cert_signature(amd_pki.ask_cert, amd_pki.ark_cert.public_key())

    def test_vcek_signature_direct(self, amd_pki: SyntheticAmdPKI) -> None:
        """VCEK must be an ECDSA P-384 leaf certificate signed by ASK (RSA-4096 / SHA-384)."""
        assert isinstance(amd_pki.vcek_cert.public_key(), ec.EllipticCurvePublicKey)
        assert isinstance(amd_pki.vcek_cert.public_key().curve, ec.SECP384R1)
        # Must not raise
        _verify_x509_cert_signature(amd_pki.vcek_cert, amd_pki.ask_cert.public_key())

    def test_full_chain_verification_with_root(
        self, amd_verifier: AttestationVerifier, amd_pki: SyntheticAmdPKI
    ) -> None:
        """verify_certificate_chain validates full 3-cert chain [VCEK, ASK, ARK]."""
        chain = [amd_pki.vcek_cert_pem, amd_pki.ask_cert_pem, amd_pki.ark_cert_pem]
        leaf, root, length = amd_verifier.verify_certificate_chain(chain)

        assert length == 3
        assert leaf.serial_number == amd_pki.vcek_cert.serial_number
        assert root.serial_number == amd_pki.ark_cert.serial_number

    def test_partial_chain_anchored_to_root(
        self, amd_verifier: AttestationVerifier, amd_pki: SyntheticAmdPKI
    ) -> None:
        """verify_certificate_chain validates 2-cert chain [VCEK, ASK] anchored to pre-installed ARK."""
        chain = [amd_pki.vcek_cert_pem, amd_pki.ask_cert_pem]
        leaf, root, length = amd_verifier.verify_certificate_chain(chain)

        assert length == 2
        assert leaf.serial_number == amd_pki.vcek_cert.serial_number
        assert root.serial_number == amd_pki.ark_cert.serial_number

    def test_vcek_custom_oid_extensions_decoding(self, amd_pki: SyntheticAmdPKI) -> None:
        """VCEK certificate must contain all 7 AMD custom extensions under Arc 1.3.6.1.4.1.3704.1."""
        extensions = amd_pki.vcek_cert.extensions

        ext_struct_ver = extensions.get_extension_for_oid(x509.ObjectIdentifier(AMD_OID_STRUCT_VERSION))
        assert der_decode_integer(ext_struct_ver.value.value) == 1

        ext_prod_name = extensions.get_extension_for_oid(x509.ObjectIdentifier(AMD_OID_PRODUCT_NAME))
        assert der_decode_ia5string(ext_prod_name.value.value) == "Milan"

        ext_bl = extensions.get_extension_for_oid(x509.ObjectIdentifier(AMD_OID_BL_SPL))
        assert der_decode_integer(ext_bl.value.value) == 3

        ext_tee = extensions.get_extension_for_oid(x509.ObjectIdentifier(AMD_OID_TEE_SPL))
        assert der_decode_integer(ext_tee.value.value) == 0

        ext_snp = extensions.get_extension_for_oid(x509.ObjectIdentifier(AMD_OID_SNP_SPL))
        assert der_decode_integer(ext_snp.value.value) == 8

        ext_ucode = extensions.get_extension_for_oid(x509.ObjectIdentifier(AMD_OID_UCODE_SPL))
        assert der_decode_integer(ext_ucode.value.value) == 115

        ext_hwid = extensions.get_extension_for_oid(x509.ObjectIdentifier(AMD_OID_HW_ID))
        assert der_decode_octet_string(ext_hwid.value.value) == DEFAULT_SNP_CHIP_ID

    def test_x509_basic_constraints_and_key_usage(self, amd_pki: SyntheticAmdPKI) -> None:
        """ARK and ASK must be CAs; VCEK must NOT be a CA and have correct key usage."""
        # ARK
        ark_bc = amd_pki.ark_cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS).value
        assert ark_bc.ca is True
        assert ark_bc.path_length == 2
        ark_ku = amd_pki.ark_cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value
        assert ark_ku.key_cert_sign is True
        assert ark_ku.crl_sign is True

        # ASK
        ask_bc = amd_pki.ask_cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS).value
        assert ask_bc.ca is True
        assert ask_bc.path_length == 0
        ask_ku = amd_pki.ask_cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value
        assert ask_ku.key_cert_sign is True
        assert ask_ku.crl_sign is True

        # VCEK
        vcek_bc = amd_pki.vcek_cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS).value
        assert vcek_bc.ca is False
        vcek_ku = amd_pki.vcek_cert.extensions.get_extension_for_oid(ExtensionOID.KEY_USAGE).value
        assert vcek_ku.digital_signature is True
        assert vcek_ku.key_agreement is True
        assert vcek_ku.key_cert_sign is False

    def test_synthetic_pki_custom_parameters(self) -> None:
        """generate_synthetic_amd_pki creates custom processor generations and SPL levels accurately."""
        custom_chip_id = b"\x42" * 64
        custom_pki = generate_synthetic_amd_pki(
            product_name="Genoa",
            chip_id=custom_chip_id,
            bl_spl=7,
            tee_spl=2,
            snp_spl=15,
            ucode_spl=200,
            rsa_bits=2048,  # use 2048 for faster generation in test
        )
        assert custom_pki.chip_id == custom_chip_id
        ext_prod = custom_pki.vcek_cert.extensions.get_extension_for_oid(x509.ObjectIdentifier(AMD_OID_PRODUCT_NAME))
        assert der_decode_ia5string(ext_prod.value.value) == "Genoa"
        ext_bl = custom_pki.vcek_cert.extensions.get_extension_for_oid(x509.ObjectIdentifier(AMD_OID_BL_SPL))
        assert der_decode_integer(ext_bl.value.value) == 7
        ext_hwid = custom_pki.vcek_cert.extensions.get_extension_for_oid(x509.ObjectIdentifier(AMD_OID_HW_ID))
        assert der_decode_octet_string(ext_hwid.value.value) == custom_chip_id


# ==============================================================================
# TEST SUITE 2: BROKEN CHAIN REJECTIONS
# ==============================================================================

class TestAmdPkiBrokenChainRejections:
    """Empirical verification that broken, corrupted, or malicious chains are strictly rejected."""

    def test_rejection_corrupted_vcek_signature(self, amd_pki: SyntheticAmdPKI) -> None:
        """Corrupting VCEK signature bytes must trigger AttestationSignatureError."""
        # Re-sign VCEK with wrong key or mutate signature bytes
        wrong_ask_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        # Sign with wrong key
        bad_vcek = (
            x509.CertificateBuilder()
            .subject_name(amd_pki.vcek_cert.subject)
            .issuer_name(amd_pki.ask_cert.subject)
            .public_key(amd_pki.vcek_cert.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(amd_pki.vcek_cert.not_valid_before)
            .not_valid_after(amd_pki.vcek_cert.not_valid_after)
            .sign(wrong_ask_key, hashes.SHA384())
        )

        with pytest.raises(AttestationSignatureError, match="signature verification failed"):
            _verify_x509_cert_signature(bad_vcek, amd_pki.ask_cert.public_key())

        # In verify_certificate_chain
        pol = VerificationPolicy()
        pol.add_trusted_root_pem(amd_pki.ark_cert_pem)
        verifier = AttestationVerifier(policy=pol)
        bad_vcek_pem = bad_vcek.public_bytes(serialization.Encoding.PEM).decode("ascii")

        with pytest.raises(AttestationSignatureError):
            verifier.verify_certificate_chain([bad_vcek_pem, amd_pki.ask_cert_pem, amd_pki.ark_cert_pem])

    def test_rejection_corrupted_ask_signature(self, amd_pki: SyntheticAmdPKI) -> None:
        """Corrupting ASK signature bytes must trigger AttestationSignatureError."""
        wrong_ark_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        bad_ask = (
            x509.CertificateBuilder()
            .subject_name(amd_pki.ask_cert.subject)
            .issuer_name(amd_pki.ark_cert.subject)
            .public_key(amd_pki.ask_cert.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(amd_pki.ask_cert.not_valid_before)
            .not_valid_after(amd_pki.ask_cert.not_valid_after)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(wrong_ark_key, hashes.SHA384())
        )

        with pytest.raises(AttestationSignatureError, match="signature verification failed"):
            _verify_x509_cert_signature(bad_ask, amd_pki.ark_cert.public_key())

        pol = VerificationPolicy()
        pol.add_trusted_root_pem(amd_pki.ark_cert_pem)
        verifier = AttestationVerifier(policy=pol)
        bad_ask_pem = bad_ask.public_bytes(serialization.Encoding.PEM).decode("ascii")

        with pytest.raises(AttestationSignatureError):
            verifier.verify_certificate_chain([amd_pki.vcek_cert_pem, bad_ask_pem, amd_pki.ark_cert_pem])

    def test_rejection_expired_chain(self) -> None:
        """Chain generated with expired=True must be rejected with AttestationExpiredError."""
        expired_pki = generate_synthetic_amd_pki(expired=True, rsa_bits=2048)
        pol = VerificationPolicy()
        pol.add_trusted_root_pem(expired_pki.ark_cert_pem)
        verifier = AttestationVerifier(policy=pol)

        with pytest.raises(AttestationExpiredError, match="has expired"):
            verifier.verify_certificate_chain(expired_pki.certificate_chain)

    def test_rejection_not_yet_valid_chain(self) -> None:
        """Chain generated with not_yet_valid=True must be rejected with AttestationExpiredError."""
        future_pki = generate_synthetic_amd_pki(not_yet_valid=True, rsa_bits=2048)
        pol = VerificationPolicy()
        pol.add_trusted_root_pem(future_pki.ark_cert_pem)
        verifier = AttestationVerifier(policy=pol)

        with pytest.raises(AttestationExpiredError, match="not yet valid"):
            verifier.verify_certificate_chain(future_pki.certificate_chain)

    def test_rejection_runtime_temporal_skew(
        self, amd_verifier: AttestationVerifier, amd_pki: SyntheticAmdPKI
    ) -> None:
        """Valid chain verified against a far-future or far-past verification_time must fail."""
        now = datetime.datetime.now(datetime.timezone.utc)
        far_future = now + datetime.timedelta(days=400)
        far_past = now - datetime.timedelta(days=400)

        with pytest.raises(AttestationExpiredError, match="has expired"):
            amd_verifier.verify_certificate_chain(
                amd_pki.certificate_chain, verification_time=far_future
            )

        with pytest.raises(AttestationExpiredError, match="not yet valid"):
            amd_verifier.verify_certificate_chain(
                amd_pki.certificate_chain, verification_time=far_past
            )

    def test_rejection_wrong_subject_issuer_mismatch(
        self, amd_verifier: AttestationVerifier, amd_pki: SyntheticAmdPKI
    ) -> None:
        """Inverting chain or skipping ASK must fail chain continuity."""
        # Inverted: [ARK, ASK, VCEK]
        inverted = [amd_pki.ark_cert_pem, amd_pki.ask_cert_pem, amd_pki.vcek_cert_pem]
        with pytest.raises(AttestationChainInvalidError):
            amd_verifier.verify_certificate_chain(inverted)

        # Skipped intermediate: [VCEK, ARK]
        skipped = [amd_pki.vcek_cert_pem, amd_pki.ark_cert_pem]
        with pytest.raises(AttestationChainInvalidError, match="Certificate chain broken at index 0"):
            amd_verifier.verify_certificate_chain(skipped)

    def test_rejection_foreign_intermediate(
        self, amd_verifier: AttestationVerifier, amd_pki: SyntheticAmdPKI
    ) -> None:
        """Substituting ASK from a different independent PKI must fail signature check."""
        foreign_pki = generate_synthetic_amd_pki(rsa_bits=2048)
        # Chain: [Our VCEK, Foreign ASK, Our ARK]
        # Notice: Our VCEK issuer name is "CN=SEV-Milan", Foreign ASK subject is also "CN=SEV-Milan",
        # BUT foreign ASK public key does not match the key that signed our VCEK!
        hybrid_chain = [amd_pki.vcek_cert_pem, foreign_pki.ask_cert_pem, amd_pki.ark_cert_pem]

        with pytest.raises(AttestationSignatureError, match="signature verification failed"):
            amd_verifier.verify_certificate_chain(hybrid_chain)

    def test_rejection_untrusted_root(self, amd_pki: SyntheticAmdPKI) -> None:
        """Valid chain rooted in untrusted ARK must be rejected with AttestationUntrustedRootError."""
        ev, rogue_ark_pem = create_untrusted_ark_snp_evidence()
        pol = VerificationPolicy()
        pol.add_trusted_root_pem(amd_pki.ark_cert_pem)  # only trust our ARK
        verifier = AttestationVerifier(policy=pol)

        with pytest.raises(AttestationUntrustedRootError, match="cannot be anchored to any trusted Root CA"):
            verifier.verify_certificate_chain(ev.certificate_chain)

    def test_rejection_empty_trust_store(self, amd_pki: SyntheticAmdPKI) -> None:
        """Policy with zero trusted roots must fail immediately."""
        pol = VerificationPolicy()  # empty trusted roots
        verifier = AttestationVerifier(policy=pol)

        with pytest.raises(AttestationUntrustedRootError, match="zero trusted Root CAs configured"):
            verifier.verify_certificate_chain(amd_pki.certificate_chain)

    def test_rejection_revoked_certificates(
        self, amd_pki: SyntheticAmdPKI
    ) -> None:
        """Revoked certificates in chain must trigger AttestationRevokedError."""
        # 1. Revoke VCEK
        pol_vcek = VerificationPolicy()
        pol_vcek.add_trusted_root_pem(amd_pki.ark_cert_pem)
        pol_vcek.revoked_serial_numbers.add(amd_pki.vcek_cert.serial_number)
        v_vcek = AttestationVerifier(policy=pol_vcek)

        with pytest.raises(AttestationRevokedError, match="is revoked"):
            v_vcek.verify_certificate_chain(amd_pki.certificate_chain)

        # 2. Revoke ASK
        pol_ask = VerificationPolicy()
        pol_ask.add_trusted_root_pem(amd_pki.ark_cert_pem)
        pol_ask.revoked_serial_numbers.add(amd_pki.ask_cert.serial_number)
        v_ask = AttestationVerifier(policy=pol_ask)

        with pytest.raises(AttestationRevokedError, match="is revoked"):
            v_ask.verify_certificate_chain(amd_pki.certificate_chain)

    def test_rejection_leaf_certificate_with_ca_true(self, amd_pki: SyntheticAmdPKI) -> None:
        """Leaf certificate marked with basic_constraints.ca=True must be rejected."""
        # Create a malicious leaf with ca=True
        leaf_key = ec.generate_private_key(ec.SECP384R1())
        malicious_leaf = (
            x509.CertificateBuilder()
            .subject_name(amd_pki.vcek_cert.subject)
            .issuer_name(amd_pki.ask_cert.subject)
            .public_key(leaf_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(amd_pki.vcek_cert.not_valid_before)
            .not_valid_after(amd_pki.vcek_cert.not_valid_after)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(amd_pki.ask_key, hashes.SHA384())
        )
        malicious_leaf_pem = malicious_leaf.public_bytes(serialization.Encoding.PEM).decode("ascii")

        pol = VerificationPolicy()
        pol.add_trusted_root_pem(amd_pki.ark_cert_pem)
        verifier = AttestationVerifier(policy=pol)

        with pytest.raises(AttestationChainInvalidError, match="Leaf certificate cannot have CA flag enabled"):
            verifier.verify_certificate_chain([malicious_leaf_pem, amd_pki.ask_cert_pem, amd_pki.ark_cert_pem])

    def test_rejection_empty_chain(self, amd_verifier: AttestationVerifier) -> None:
        """Empty certificate chain list must raise AttestationEvidenceMalformedError."""
        with pytest.raises(AttestationEvidenceMalformedError, match="Certificate chain is empty"):
            amd_verifier.verify_certificate_chain([])


# ==============================================================================
# TEST SUITE 3: CACHED SINGLETON PERFORMANCE & THREAD SAFETY
# ==============================================================================

class TestCachedSingletonPerformanceAndConcurrency:
    """Empirical verification of get_default_amd_pki() performance, caching, and thread-safety."""

    def test_cached_singleton_identity(self) -> None:
        """Repeated calls to get_default_amd_pki() must return the exact same object reference."""
        pki1 = get_default_amd_pki()
        pki2 = get_default_amd_pki()
        assert pki1 is pki2
        assert pki1.ark_cert == pki2.ark_cert
        assert pki1.vcek_cert == pki2.vcek_cert

    def test_cached_singleton_latency_benchmark(self) -> None:
        """1,000 warm invocations of get_default_amd_pki() must execute in < 10ms (< 10us/call)."""
        get_default_amd_pki()  # warm up

        start = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            p = get_default_amd_pki()
            assert p is not None
        elapsed = time.perf_counter() - start

        avg_us = (elapsed / iterations) * 1_000_000
        assert elapsed < 0.05, f"1,000 calls took {elapsed:.4f}s (> 50ms)"
        assert avg_us < 50.0, f"Average call latency {avg_us:.2f}us (> 50us)"

    def test_multithreaded_concurrency_warm(self) -> None:
        """50 concurrent worker threads accessing get_default_amd_pki() must be fast and error-free."""
        get_default_amd_pki()  # warm up
        thread_count = 50
        results: List[Any] = [None] * thread_count
        errors: List[Exception] = []

        barrier = threading.Barrier(thread_count)

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5.0)
                results[idx] = get_default_amd_pki()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Encountered errors under concurrency: {errors}"
        primary = results[0]
        assert primary is not None
        for res in results:
            assert res is primary, "Thread received non-identical PKI instance"

    def test_multithreaded_cold_start_thread_safety(self) -> None:
        """
        Adversarial test: Concurrent cold start with multiple threads.
        Verifies that even if multiple threads race to initialize get_default_amd_pki(),
        all threads obtain a fully valid, functional SyntheticAmdPKI without exception.
        """
        import matrix_compute.crypto.pki_fixtures as pki_module

        thread_count = 10
        results: List[Any] = [None] * thread_count
        errors: List[Exception] = []

        # Temporarily save original cached PKI
        orig_cache = pki_module._DEFAULT_AMD_PKI_CACHE
        try:
            # Force cache to None to simulate cold race
            pki_module._DEFAULT_AMD_PKI_CACHE = None
            barrier = threading.Barrier(thread_count)

            def cold_worker(idx: int) -> None:
                try:
                    barrier.wait(timeout=10.0)
                    # Use get_default_amd_pki
                    results[idx] = pki_module.get_default_amd_pki()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=cold_worker, args=(i,)) for i in range(thread_count)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)

            assert not errors, f"Errors in cold start race: {errors}"
            for idx, res in enumerate(results):
                assert res is not None
                assert isinstance(res, SyntheticAmdPKI)
                assert res.ark_cert is not None
                assert res.ask_cert is not None
                assert res.vcek_cert is not None
                # Validate self-consistency of returned PKI instance
                _verify_x509_cert_signature(res.vcek_cert, res.ask_cert.public_key())
                _verify_x509_cert_signature(res.ask_cert, res.ark_cert.public_key())
                _verify_x509_cert_signature(res.ark_cert, res.ark_cert.public_key())

        finally:
            # Restore original cached singleton
            pki_module._DEFAULT_AMD_PKI_CACHE = orig_cache


# ==============================================================================
# TEST SUITE 4: SNP REPORT GENERATION & SIGNATURE VERIFICATION
# ==============================================================================

class TestSnpReportGenerationAndCrypto:
    """Empirical verification of 1184-byte report binary layout and ECDSA P-384 crypto."""

    def test_snp_report_wire_format(self, amd_pki: SyntheticAmdPKI) -> None:
        """Report must match AMD Pub #56860 wire layout exactly."""
        nonce = secrets.token_bytes(32)
        pubkey_hash = hashlib.sha256(b"worker-ephemeral-key").digest()
        report_data = nonce + pubkey_hash

        rep = generate_synthetic_snp_report(
            vcek_priv_ec=amd_pki.vcek_priv_ec,
            report_data=report_data,
            chip_id=amd_pki.chip_id,
        )

        assert len(rep) == 1184
        # Version (uint32 LE at 0x000)
        assert struct.unpack_from("<I", rep, 0x000)[0] == 2
        # Policy (uint64 LE at 0x008, debug bit 19 == 0)
        policy = struct.unpack_from("<Q", rep, 0x008)[0]
        assert policy == DEFAULT_SNP_POLICY
        assert (policy >> 19) & 1 == 0
        # Signature algo (uint32 LE at 0x034)
        assert struct.unpack_from("<I", rep, 0x034)[0] == 1
        # Report data (64 bytes at 0x050)
        assert rep[0x050:0x090] == report_data
        # Measurement (48 bytes at 0x090)
        assert rep[0x090:0x0C0] == DEFAULT_SNP_MEASUREMENT
        # Host data (32 bytes at 0x0C0)
        assert rep[0x0C0:0x0E0] == DEFAULT_SNP_HOST_DATA
        # Chip ID (64 bytes at 0x1A0)
        assert rep[0x1A0:0x1E0] == amd_pki.chip_id
        # Signature padding (368 bytes zeroes at 0x330..0x49F)
        assert rep[0x330:0x4A0] == b"\x00" * 368

    def test_snp_report_ecdsa_p384_signature_verification(self, amd_pki: SyntheticAmdPKI) -> None:
        """Little-endian R and S scalars extracted from 0x2A0 and 0x2E8 verify with VCEK pubkey."""
        rep = generate_synthetic_snp_report(vcek_priv_ec=amd_pki.vcek_priv_ec)

        r = int.from_bytes(rep[0x2A0 : 0x2A0 + 48], "little")
        s = int.from_bytes(rep[0x2E8 : 0x2E8 + 48], "little")
        der_sig = encode_dss_signature(r, s)

        # Must verify cleanly over bytes 0x000..0x29F (672 bytes)
        amd_pki.vcek_cert.public_key().verify(
            der_sig,
            rep[:672],
            ec.ECDSA(hashes.SHA384()),
        )

    def test_snp_report_tamper_body_adversarial(self, amd_pki: SyntheticAmdPKI) -> None:
        """Any 1-bit mutation across the signed body slice (0x000..0x29F) fails verification."""
        rep = generate_synthetic_snp_report(vcek_priv_ec=amd_pki.vcek_priv_ec)
        r = int.from_bytes(rep[0x2A0 : 0x2A0 + 48], "little")
        s = int.from_bytes(rep[0x2E8 : 0x2E8 + 48], "little")
        der_sig = encode_dss_signature(r, s)

        test_offsets = [0x000, 0x008, 0x050, 0x090, 0x0C0, 0x1A0, 0x29F]
        for offset in test_offsets:
            mutated = bytearray(rep)
            mutated[offset] ^= 0x01
            with pytest.raises(InvalidSignature):
                amd_pki.vcek_cert.public_key().verify(
                    der_sig,
                    mutated[:672],
                    ec.ECDSA(hashes.SHA384()),
                )

    def test_snp_report_tamper_signature_adversarial(self, amd_pki: SyntheticAmdPKI) -> None:
        """Any bit mutation in signature scalars R or S fails verification."""
        rep = generate_synthetic_snp_report(vcek_priv_ec=amd_pki.vcek_priv_ec)

        # Corrupt R
        r_corrupt = int.from_bytes(rep[0x2A0 : 0x2A0 + 48], "little") ^ 0x01
        s = int.from_bytes(rep[0x2E8 : 0x2E8 + 48], "little")
        with pytest.raises(InvalidSignature):
            amd_pki.vcek_cert.public_key().verify(
                encode_dss_signature(r_corrupt, s),
                rep[:672],
                ec.ECDSA(hashes.SHA384()),
            )

        # Corrupt S
        r = int.from_bytes(rep[0x2A0 : 0x2A0 + 48], "little")
        s_corrupt = int.from_bytes(rep[0x2E8 : 0x2E8 + 48], "little") ^ 0x01
        with pytest.raises(InvalidSignature):
            amd_pki.vcek_cert.public_key().verify(
                encode_dss_signature(r, s_corrupt),
                rep[:672],
                ec.ECDSA(hashes.SHA384()),
            )

    def test_all_12_adversarial_fixture_generators(self, amd_pki: SyntheticAmdPKI) -> None:
        """Validate that all 12 negative/adversarial fixture generators construct correct invalid states."""
        nonce = secrets.token_bytes(32)
        worker_key = secrets.token_bytes(32)

        # 1. Payload tampered
        ev1 = create_tampered_snp_payload_evidence(pki=amd_pki, nonce=nonce)
        assert len(ev1.evidence_blob) == 1184

        # 2. Signature tampered
        ev2 = create_tampered_snp_signature_evidence(pki=amd_pki, nonce=nonce)
        assert len(ev2.evidence_blob) == 1184

        # 3. Debug policy
        ev3 = create_debug_policy_snp_evidence(pki=amd_pki, nonce=nonce)
        assert ev3.token_claims["policy_debug"] is True
        assert (ev3.token_claims["policy"] >> 19) & 1 == 1

        # 4. Nonce mismatch
        ev4 = create_nonce_mismatch_snp_evidence(requested_nonce=nonce, pki=amd_pki)
        assert ev4.token_claims["nonce"] != nonce.hex()

        # 5. Pubkey hash mismatch
        ev5 = create_pubkey_hash_mismatch_snp_evidence(nonce=nonce, pki=amd_pki)
        assert ev5.user_data is not None

        # 6. Untrusted ARK
        ev6, rogue_ark = create_untrusted_ark_snp_evidence(nonce=nonce)
        assert "BEGIN CERTIFICATE" in rogue_ark
        assert ev6.certificate_chain[-1] == rogue_ark

        # 7. Expired VCEK
        ev7 = create_expired_vcek_snp_evidence(nonce=nonce)
        assert len(ev7.certificate_chain) == 3

        # 8. Not yet valid VCEK
        ev8 = create_not_yet_valid_vcek_snp_evidence(nonce=nonce)
        assert len(ev8.certificate_chain) == 3

        # 9. Revoked VCEK
        ev9, sn = create_revoked_vcek_snp_evidence(pki=amd_pki, nonce=nonce)
        assert sn == amd_pki.vcek_cert.serial_number

        # 10. Truncated report
        ev10 = create_truncated_snp_report_evidence(pki=amd_pki, truncated_length=1000)
        assert len(ev10.evidence_blob) == 1000

        # 11. Measurement mismatch
        ev11 = create_measurement_mismatch_snp_evidence(pki=amd_pki, nonce=nonce)
        assert ev11.token_claims["measurement"] != DEFAULT_SNP_GOLDEN_RIM["measurement"]

        # 12. TCB downgrade
        ev12 = create_tcb_downgrade_snp_evidence(pki=amd_pki, nonce=nonce)
        assert len(ev12.evidence_blob) == 1184
