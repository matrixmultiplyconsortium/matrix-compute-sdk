"""
Unit and Adversarial Test Suite for Intel TDX Attestation Verification (Milestone 2).

100% offline hermetic execution:
1. Valid Quote Format v4 and Modular Format v5 (TDX 1.0 & 1.5) verification end-to-end.
2. Strict TDATTRIBUTES.DEBUG == 0 rejection ((tdattributes & 0x01) == 0).
3. Dual ECDSA signature verification (AK over Header+Body, PCK over QE3 report).
4. Attestation Key cross-binding verification in QE report data.
5. Constant-time freshness (nonce) and ephemeral public key binding validation in REPORTDATA.
6. Reference Golden RIM measurement appraisal (MRTD, RTMR0..3).
7. Truncated, malformed, and unsupported quote versions.
8. Certificate chain validation: untrusted Intel Root CA, expired, not-yet-valid, revoked, and broken chains.
9. Proprietary Intel PCK ASN.1 extension parsing (FMSPC, PCEID, TCB under OID 1.2.840.113741.1.13.1).
10. Registration and dispatch via AttestationVerifier.
"""

from __future__ import annotations

import copy
import hashlib
import secrets
import struct
from typing import Dict, List, Optional

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from matrix_compute.crypto import (
    AttestationVerifier,
    BaseEvidenceVerifier,
    IntelTdxVerifier,
    VerificationPolicy,
    VerificationResult,
    parse_pck_extensions,
    get_default_intel_pki,
    generate_synthetic_intel_pki,
    generate_synthetic_tdx_quote,
    create_valid_intel_tdx_evidence,
    create_tampered_tdx_quote_evidence,
    create_tampered_tdx_signature_evidence,
    create_tampered_qe_signature_evidence,
    create_tampered_ak_binding_tdx_evidence,
    create_tampered_qe_ak_binding_evidence,
    create_debug_tdx_evidence,
    create_nonce_mismatch_tdx_evidence,
    create_pubkey_mismatch_tdx_evidence,
    create_measurement_mismatch_tdx_evidence,
    create_untrusted_root_tdx_evidence,
    create_expired_cert_tdx_evidence,
    create_not_yet_valid_cert_tdx_evidence,
    create_revoked_cert_tdx_evidence,
    create_truncated_tdx_quote_evidence,
    create_malformed_version_tdx_evidence,
    DEFAULT_TDX_GOLDEN_RIM,
    INTEL_TDX_GOLDEN_RIM,
    DEFAULT_TDX_MRTD,
    DEFAULT_TDX_RTMR0,
    DEFAULT_TDX_RTMR1,
    DEFAULT_TDX_RTMR2,
    DEFAULT_TDX_RTMR3,
    DEFAULT_INTEL_FMSPC,
    DEFAULT_INTEL_PCEID,
    INTEL_TDX_TEE_TYPE,
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
    AttestationRevokedError,
    AttestationSignatureError,
    AttestationUntrustedRootError,
)
from matrix_contracts.schemas.attestation import AttestationEvidence


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def intel_pki():
    """Cached singleton Intel PKI hierarchy to guarantee sub-second execution."""
    return get_default_intel_pki()


@pytest.fixture
def default_policy(intel_pki) -> VerificationPolicy:
    """Pre-configured VerificationPolicy trusting the synthetic Intel SGX/TDX Root CA."""
    policy = VerificationPolicy()
    policy.add_trusted_root_pem(intel_pki.root_cert_pem)
    policy.add_golden_rim("intel-tdx", INTEL_TDX_GOLDEN_RIM)
    return policy


@pytest.fixture
def verifier(default_policy) -> AttestationVerifier:
    """AttestationVerifier instance configured with default Intel TDX policy."""
    return AttestationVerifier(policy=default_policy)


@pytest.fixture
def sample_nonce() -> bytes:
    """32-byte challenger nonce."""
    return secrets.token_bytes(32)


@pytest.fixture
def sample_client_pubkey() -> bytes:
    """32-byte client ephemeral public key."""
    return secrets.token_bytes(32)


# ==============================================================================
# TEST SUITE 1: VALID END-TO-END EXECUTION (V4 & V5 MODULAR)
# ==============================================================================

class TestIntelTdxVerifierValidExecution:
    """Verifies valid authentic Intel TDX attestation evidence execution."""

    def test_valid_quote_v4_verification_end_to_end(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """End-to-end verification of authentic Quote Format v4 with 3-tier PCK chain."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            quote_version=4,
        )

        result: VerificationResult = verifier.verify(
            evidence=evidence,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_client_pubkey,
        )

        assert result.verified is True
        assert result.provider == "intel-tdx"
        assert result.chain_length == 3
        assert result.measurements_verified == 5
        assert result.claims["quote_version"] == 4
        assert result.claims["debug"] is False
        assert result.claims["nonce"] == sample_nonce.hex()
        assert result.claims["ephemeral_pubkey_hash"] == hashlib.sha256(sample_client_pubkey).hexdigest()

    @pytest.mark.parametrize("tdx_type", [0x0002, 0x0003])
    def test_valid_quote_v5_modular_descriptor_verification(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, tdx_type
    ):
        """End-to-end verification of Quote Format v5 for TDX 1.0 (0x0002) and TDX 1.5 (0x0003)."""
        pubkey_hash = hashlib.sha256(sample_client_pubkey).digest()
        report_data = sample_nonce + pubkey_hash

        quote_blob = generate_synthetic_tdx_quote(
            pki=intel_pki,
            quote_version=5,
            tdx_type=tdx_type,
            report_data=report_data,
            debug=False,
        )

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            token_claims={},
            token_signature=quote_blob[642:706],
            golden_rim=INTEL_TDX_GOLDEN_RIM.copy(),
            evidence_blob=quote_blob,
            user_data=report_data,
        )

        result = verifier.verify(
            evidence=evidence,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_client_pubkey,
        )

        assert result.verified is True
        assert result.provider == "intel-tdx"
        assert result.chain_length == 3
        assert result.measurements_verified == 5
        assert result.claims["quote_version"] == 5
        assert result.claims["debug"] is False
        assert result.claims["body_type"] == tdx_type
        assert result.claims["tdx_version"] == ("1.5" if tdx_type == 3 else "1.0")

    def test_valid_claims_extraction(self, intel_pki, sample_nonce, sample_client_pubkey):
        """Validates all normalized fields extracted from raw binary TD quote."""
        strategy = IntelTdxVerifier()
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            quote_version=4,
        )

        claims = strategy.extract_claims(evidence)

        assert claims["provider"] == "intel-tdx"
        assert claims["quote_version"] == 4
        assert claims["debug"] is False
        assert claims["nonce"] == sample_nonce.hex()
        assert claims["ephemeral_pubkey_hash"] == hashlib.sha256(sample_client_pubkey).hexdigest()
        assert claims["client_pubkey_hash"] == claims["ephemeral_pubkey_hash"]
        assert claims["measurements"]["mrtd"] == DEFAULT_TDX_MRTD.hex()
        assert claims["measurements"]["rtmr0"] == DEFAULT_TDX_RTMR0.hex()
        assert claims["measurements"]["rtmr1"] == DEFAULT_TDX_RTMR1.hex()
        assert claims["measurements"]["rtmr2"] == DEFAULT_TDX_RTMR2.hex()
        assert claims["measurements"]["rtmr3"] == DEFAULT_TDX_RTMR3.hex()
        assert claims["fmspc"] == DEFAULT_INTEL_FMSPC.lower()
        assert claims["pceid"] == DEFAULT_INTEL_PCEID.lower()

    def test_pck_custom_oid_extensions_parsed(self, intel_pki):
        """Validates parsing of Intel proprietary ASN.1 extension (OID 1.2.840.113741.1.13.1)."""
        exts = parse_pck_extensions(intel_pki.pck_cert)

        assert exts["fmspc"] == DEFAULT_INTEL_FMSPC.lower()
        assert exts["pceid"] == DEFAULT_INTEL_PCEID.lower()
        assert "tcb" in exts
        assert "raw_der_len" in exts

    def test_dual_field_report_data_binding(self, verifier, intel_pki, sample_nonce, sample_client_pubkey):
        """Verifies dual-field packing: REPORTDATA[0..31] = nonce, REPORTDATA[32..63] = pubkey hash."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        assert len(evidence.user_data) == 64
        assert evidence.user_data[:32] == sample_nonce
        assert evidence.user_data[32:64] == hashlib.sha256(sample_client_pubkey).digest()

        result = verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert result.verified is True

    def test_extract_certificate_chain_from_quote_blob(self, intel_pki, sample_nonce, sample_client_pubkey):
        """Verifies IntelTdxVerifier.extract_certificate_chain recovers PEM chain from quote binary."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            quote_version=4,
        )
        pems = IntelTdxVerifier.extract_certificate_chain(evidence.evidence_blob)
        assert len(pems) == 3
        assert "-----BEGIN CERTIFICATE-----" in pems[0]
        assert "-----BEGIN CERTIFICATE-----" in pems[1]
        assert "-----BEGIN CERTIFICATE-----" in pems[2]


# ==============================================================================
# TEST SUITE 2: STRICT DEBUG MODE REJECTION
# ==============================================================================

class TestIntelTdxDebugPolicyRejection:
    """Verifies strict fail-closed rejection of TDATTRIBUTES.DEBUG != 0."""

    def test_reject_debug_quote_v4(self, verifier, intel_pki, sample_nonce, sample_client_pubkey):
        """Quote Format v4 with TDATTRIBUTES.DEBUG == 1 must be rejected with AttestationSignatureError."""
        evidence = create_debug_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationSignatureError, match="DEBUG mode enabled"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_debug_quote_v5(self, verifier, intel_pki, sample_nonce, sample_client_pubkey):
        """Quote Format v5 with TDATTRIBUTES.DEBUG == 1 must be rejected with AttestationSignatureError."""
        pubkey_hash = hashlib.sha256(sample_client_pubkey).digest()
        report_data = sample_nonce + pubkey_hash

        quote_blob = generate_synthetic_tdx_quote(
            pki=intel_pki,
            quote_version=5,
            report_data=report_data,
            debug=True,
        )

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            token_claims={"debug": True},
            token_signature=quote_blob[642:706],
            golden_rim=INTEL_TDX_GOLDEN_RIM.copy(),
            evidence_blob=quote_blob,
            user_data=report_data,
        )

        with pytest.raises(AttestationSignatureError, match="DEBUG mode enabled"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_extract_claims_rejects_debug_quote(self, intel_pki, sample_nonce, sample_client_pubkey):
        """extract_claims must also fail-closed when TDATTRIBUTES.DEBUG == 1."""
        strategy = IntelTdxVerifier()
        evidence = create_debug_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationSignatureError, match="DEBUG mode enabled"):
            strategy.extract_claims(evidence)


# ==============================================================================
# TEST SUITE 3: CRYPTOGRAPHIC SIGNATURE & AK BINDING MUTATIONS
# ==============================================================================

class TestIntelTdxCryptographicMutations:
    """Verifies that 1-bit mutations across signed payload, signatures, and AK binding are caught."""

    @pytest.mark.parametrize(
        "offset,name",
        [
            (0x000, "quote_version"),
            (0x002, "att_key_type"),
            (0x004, "tee_type"),
            (0x00C, "qe_vendor_id"),
            (0x030, "tee_tcb_svn"),
            (0x040, "mrseam"),
            (0x070, "mrsignerseam"),
            (0x0B8, "mrtd_start"),
            (0x0E7, "mrtd_end"),
            (0x178, "rtmr0_start"),
            (0x1A8, "rtmr1_start"),
            (0x1D8, "rtmr2_start"),
            (0x208, "rtmr3_start"),
            (0x238, "reportdata_nonce"),
            (0x258, "reportdata_pubkey"),
            (0x277, "payload_final_byte"),
        ],
    )
    def test_single_bit_mutation_signed_payload(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, offset, name
    ):
        """Any 1-bit mutation in bytes 0..631 must fail AttestationSignatureError or AttestationError."""
        evidence = create_tampered_tdx_quote_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            tamper_offset=offset,
            tamper_mask=0x01,
        )

        with pytest.raises(AttestationError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_tampered_ak_signature(self, verifier, intel_pki, sample_nonce, sample_client_pubkey):
        """Mutation in AK ECDSA signature scalars must fail AttestationSignatureError."""
        evidence = create_tampered_tdx_signature_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            tamper_offset=636,
            tamper_mask=0x01,
        )

        with pytest.raises(AttestationSignatureError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_tampered_qe_signature(self, verifier, intel_pki, sample_nonce, sample_client_pubkey):
        """Mutation in PCK ECDSA signature over QE report must fail AttestationSignatureError."""
        evidence = create_tampered_qe_signature_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            tamper_offset=1148,
            tamper_mask=0x01,
        )

        with pytest.raises(AttestationSignatureError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_tampered_ak_binding_in_qe_report(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Altered AK hash in QE report data must trigger AttestationBindingError."""
        evidence = create_tampered_ak_binding_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            tamper_offset=1084,
            tamper_mask=0x01,
        )

        with pytest.raises(AttestationBindingError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# TEST SUITE 4: FRESHNESS & EPHEMERAL PUBLIC KEY BINDING
# ==============================================================================

class TestIntelTdxFreshnessAndPubkeyBinding:
    """Verifies challenger freshness nonce and client public key binding in REPORTDATA."""

    def test_reject_mismatched_nonce_replay_attack(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Quote embedding a stale/mismatched nonce must raise AttestationNonceMismatchError."""
        evidence = create_nonce_mismatch_tdx_evidence(
            requested_nonce=sample_nonce,
            pki=intel_pki,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationNonceMismatchError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_mismatched_client_pubkey_hash_mitm_attack(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Quote bound to a substituted public key must raise AttestationBindingError."""
        evidence = create_pubkey_mismatch_tdx_evidence(
            nonce=sample_nonce,
            pki=intel_pki,
        )

        with pytest.raises(AttestationBindingError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_nonce_hex_and_bytes_interchangeable(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Nonce can be supplied as raw bytes or hex string and compares cleanly."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        result = verifier.verify(
            evidence=evidence,
            expected_nonce=f"0x{sample_nonce.hex()}",
            expected_client_pubkey=sample_client_pubkey.hex(),
        )
        assert result.verified is True

    def test_missing_nonce_claim_rejected(self, default_policy):
        """Attestation claims missing nonce must raise AttestationNonceMismatchError."""
        verifier = AttestationVerifier(policy=default_policy)
        claims = {"measurements": {}}
        with pytest.raises(AttestationNonceMismatchError):
            verifier.verify_freshness_and_binding(claims=claims, expected_nonce=b"\x01" * 32, provider="intel-tdx")

    def test_missing_pubkey_hash_rejected(self, default_policy):
        """Attestation claims missing pubkey hash when expected must raise AttestationBindingError."""
        verifier = AttestationVerifier(policy=default_policy)
        claims = {"nonce": ("01" * 32)}
        with pytest.raises(AttestationBindingError):
            verifier.verify_freshness_and_binding(
                claims=claims,
                expected_nonce=b"\x01" * 32,
                expected_client_pubkey=b"\x02" * 32,
                provider="intel-tdx",
            )


# ==============================================================================
# TEST SUITE 5: GOLDEN RIM MEASUREMENT APPRAISAL
# ==============================================================================

class TestIntelTdxGoldenRimAppraisal:
    """Verifies MRTD and RTMR0..3 appraisal against reference Golden RIM manifest."""

    def test_golden_rim_valid_measurement_appraisal(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Authentic measurements matching Golden RIM pass appraisal (5/5)."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        result = verifier.verify(
            evidence=evidence,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_client_pubkey,
        )
        assert result.measurements_verified == 5

    def test_reject_tampered_mrtd(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Mismatched MRTD digest must raise AttestationMeasurementMismatchError."""
        evidence = create_measurement_mismatch_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationMeasurementMismatchError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_tampered_rtmr(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Mismatched RTMR digest must raise AttestationMeasurementMismatchError."""
        bad_rtmr = b"\xba\xad" * 24  # 48 bytes
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            rtmr3=bad_rtmr,
        )

        with pytest.raises(AttestationMeasurementMismatchError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_no_golden_rim_skips_appraisal(
        self, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """VerificationPolicy with zero golden rim configured returns measurements_verified == 0."""
        policy = VerificationPolicy()
        policy.add_trusted_root_pem(intel_pki.root_cert_pem)
        no_rim_verifier = AttestationVerifier(policy=policy)

        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        result = no_rim_verifier.verify(
            evidence=evidence,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_client_pubkey,
        )
        assert result.verified is True
        assert result.measurements_verified == 0

    def test_missing_measurement_key_rejected(self, default_policy):
        """Claims missing expected Golden RIM keys raises AttestationMeasurementMismatchError."""
        verifier = AttestationVerifier(policy=default_policy)
        claims = {"measurements": {"mrtd": DEFAULT_TDX_MRTD.hex()}}
        with pytest.raises(AttestationMeasurementMismatchError):
            verifier.verify_measurements(claims=claims, provider="intel-tdx")


# ==============================================================================
# TEST SUITE 6: PKI TRUST CHAIN VALIDATION
# ==============================================================================

class TestIntelTdxPkiTrustChain:
    """Verifies X.509 certificate chain validation across 3-tier Intel PKI."""

    def test_valid_3tier_pck_chain(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Valid PCK -> Intermediate -> Root chain anchors cleanly to trusted Intel Root CA."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        result = verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert result.verified is True
        assert result.chain_length == 3
        assert result.root_cert.subject == intel_pki.root_cert.subject

    def test_reject_untrusted_root(self, verifier, sample_nonce, sample_client_pubkey):
        """Chain rooted in an untrusted rogue Root CA must raise AttestationUntrustedRootError."""
        evidence, rogue_root_pem = create_untrusted_root_tdx_evidence(
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationUntrustedRootError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_expired_pck_cert(self, verifier, sample_nonce, sample_client_pubkey):
        """Expired PCK certificate in chain must raise AttestationExpiredError."""
        evidence = create_expired_cert_tdx_evidence(
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationExpiredError, match="expired"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_not_yet_valid_pck_cert(self, verifier, sample_nonce, sample_client_pubkey):
        """Not-yet-valid PCK certificate in chain must raise AttestationExpiredError."""
        evidence = create_not_yet_valid_cert_tdx_evidence(
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationExpiredError, match="not yet valid"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_revoked_pck_cert(self, intel_pki, sample_nonce, sample_client_pubkey):
        """Revoked PCK certificate serial number must raise AttestationRevokedError."""
        evidence, revoked_serial = create_revoked_cert_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        policy = VerificationPolicy(revoked_serial_numbers={revoked_serial})
        policy.add_trusted_root_pem(intel_pki.root_cert_pem)
        revoked_verifier = AttestationVerifier(policy=policy)

        with pytest.raises(AttestationRevokedError, match="revoked"):
            revoked_verifier.verify(
                evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_client_pubkey,
            )

    def test_reject_broken_intermediate_chain(self, verifier, intel_pki, sample_nonce, sample_client_pubkey):
        """Broken intermediate certificate linkage must raise AttestationChainInvalidError or AttestationSignatureError."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        # Substitute foreign intermediate certificate
        rogue_pki = generate_synthetic_intel_pki()
        evidence.certificate_chain[1] = rogue_pki.intermediate_cert_pem

        with pytest.raises((AttestationChainInvalidError, AttestationSignatureError)):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# TEST SUITE 7: WIRE FORMAT & MALFORMED BLOBS
# ==============================================================================

class TestIntelTdxWireFormatAndBlobIntegrity:
    """Verifies handling of truncated, malformed, and missing evidence blobs."""

    @pytest.mark.parametrize("length", [0, 50, 100, 631, 635])
    def test_reject_truncated_quote_blob(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, length
    ):
        """Quote blob shorter than 636 bytes must raise AttestationEvidenceMalformedError."""
        evidence = create_truncated_tdx_quote_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            truncated_length=length,
        )

        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_malformed_quote_version(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Unsupported quote version (e.g. 99) must raise AttestationEvidenceMalformedError."""
        evidence = create_malformed_version_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            invalid_version=99,
        )

        with pytest.raises(AttestationEvidenceMalformedError, match="version"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_invalid_tee_type(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Quote with invalid TEE type (not 0x00000081) must raise AttestationEvidenceMalformedError."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        struct.pack_into("<I", blob, 4, 0x00000000)  # corrupt tee_type
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationEvidenceMalformedError, match="TEE"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_none_evidence_blob(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """evidence_blob = None must raise AttestationEvidenceMalformedError."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.evidence_blob = None

        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_unsupported_att_key_type(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Quote with unsupported att_key_type (not 2 or 3) must raise AttestationEvidenceMalformedError."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        struct.pack_into("<H", blob, 2, 99)  # corrupt att_key_type
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationEvidenceMalformedError, match="Attestation Key Type"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# TEST SUITE 8: PROVIDER REGISTRATION & DISPATCH
# ==============================================================================

class TestIntelTdxRegistrationAndDispatch:
    """Verifies default registration of IntelTdxVerifier in AttestationVerifier engine."""

    def test_intel_tdx_registered_by_default(self):
        """AttestationVerifier must have 'intel-tdx' registered in providers and policy."""
        verifier = AttestationVerifier()
        assert "intel-tdx" in verifier._providers
        assert "intel-tdx" in verifier.policy.allowed_providers
        assert isinstance(verifier._providers["intel-tdx"], IntelTdxVerifier)

    def test_verify_dispatches_to_intel_tdx_verifier(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """verify() cleanly dispatches provider 'intel-tdx' to IntelTdxVerifier."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        result = verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert result.verified is True
        assert result.provider == "intel-tdx"
