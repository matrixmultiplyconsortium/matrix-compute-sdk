"""
Unit and Adversarial Test Suite for AMD SEV-SNP Attestation Verification (Milestone 4).

100% offline hermetic execution:
1. Valid 1184-byte report verification end-to-end with 3-tier AMD PKI (ARK -> ASK -> VCEK).
2. Strict POLICY.DEBUG == 0 rejection ((policy >> 19) & 0x01 == 0).
3. 1-bit mutations across signed payload (0x000..0x29F) and signature components (0x2A0..0x49F).
4. Zero padding validation in R (0x2D0..0x2E8), S (0x318..0x330), and signature_pad (0x330..0x4A0).
5. Constant-time freshness (nonce) and ephemeral public key binding validation in REPORT_DATA.
6. Reference Golden RIM measurement and host data appraisal.
7. Truncated, oversized, and malformed report blobs.
8. Certificate chain validation: untrusted ARK root, expired, not-yet-valid, revoked, and broken chains.
9. TCB downgrade protection and hardware ID (hwID / chip_id) binding validation against VCEK custom OIDs.
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
    AmdSevSnpVerifier,
    VerificationPolicy,
    VerificationResult,
    parse_vcek_extensions,
    validate_vcek_extensions,
    get_default_amd_pki,
    generate_synthetic_amd_pki,
    generate_synthetic_snp_report,
    create_valid_sev_snp_evidence,
    create_tampered_snp_payload_evidence,
    create_tampered_snp_signature_evidence,
    create_debug_policy_snp_evidence,
    create_nonce_mismatch_snp_evidence,
    create_pubkey_hash_mismatch_snp_evidence,
    create_untrusted_ark_snp_evidence,
    create_expired_vcek_snp_evidence,
    create_not_yet_valid_vcek_snp_evidence,
    create_revoked_vcek_snp_evidence,
    create_truncated_snp_report_evidence,
    create_measurement_mismatch_snp_evidence,
    create_tcb_downgrade_snp_evidence,
    DEFAULT_SNP_GOLDEN_RIM,
    DEFAULT_SNP_MEASUREMENT,
    DEFAULT_SNP_HOST_DATA,
    DEFAULT_SNP_CHIP_ID,
    DEFAULT_SNP_POLICY,
    AMD_OID_PREFIX,
    AMD_OID_STRUCT_VERSION,
    AMD_OID_PRODUCT_NAME,
    AMD_OID_BL_SPL,
    AMD_OID_TEE_SPL,
    AMD_OID_SNP_SPL,
    AMD_OID_UCODE_SPL,
    AMD_OID_HW_ID,
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
def amd_pki():
    """Cached singleton AMD PKI hierarchy to guarantee sub-second execution."""
    return get_default_amd_pki()


@pytest.fixture
def default_policy(amd_pki) -> VerificationPolicy:
    """Pre-configured VerificationPolicy trusting the synthetic ARK Root CA."""
    policy = VerificationPolicy()
    policy.add_trusted_root_pem(amd_pki.ark_cert_pem)
    policy.add_golden_rim("amd-sev-snp", DEFAULT_SNP_GOLDEN_RIM)
    return policy


@pytest.fixture
def verifier(default_policy) -> AttestationVerifier:
    """AttestationVerifier instance with default AMD policy."""
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
# TEST SUITE 1: VALID END-TO-END REPORT & EXTENSIONS
# ==============================================================================

class TestSevSnpVerifierValidExecution:
    """Verifies valid authentic AMD SEV-SNP attestation evidence execution."""

    def test_valid_sev_snp_report_verification_end_to_end(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """End-to-end verification of authentic 1184-byte report with 3-tier PKI."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        result: VerificationResult = verifier.verify(
            evidence=evidence,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_client_pubkey,
        )

        assert result.verified is True
        assert result.provider == "amd-sev-snp"
        assert result.chain_length == 3
        assert result.measurements_verified == 2
        assert result.claims["version"] == 2
        assert result.claims["policy_debug"] is False
        assert result.claims["nonce"] == sample_nonce.hex()
        assert result.claims["ephemeral_pubkey_hash"] == hashlib.sha256(sample_client_pubkey).hexdigest()

    def test_valid_sev_snp_claims_extraction(self, amd_pki, sample_nonce, sample_client_pubkey):
        """Validates all normalized fields extracted from 1184-byte binary report."""
        strategy = AmdSevSnpVerifier()
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        claims = strategy.extract_claims(evidence)

        assert claims["provider"] == "amd-sev-snp"
        assert claims["version"] == 2
        assert claims["guest_svn"] == 1
        assert claims["policy"] == DEFAULT_SNP_POLICY
        assert claims["policy_debug"] is False
        assert claims["vmpl"] == 0
        assert claims["signature_algo"] == 1
        assert claims["nonce"] == sample_nonce.hex()
        assert claims["ephemeral_pubkey_hash"] == hashlib.sha256(sample_client_pubkey).hexdigest()
        assert claims["client_pubkey_hash"] == claims["ephemeral_pubkey_hash"]
        assert claims["measurement"] == DEFAULT_SNP_MEASUREMENT.hex()
        assert claims["host_data"] == DEFAULT_SNP_HOST_DATA.hex()
        assert claims["chip_id"] == amd_pki.chip_id.hex()
        assert claims["measurements"]["measurement"] == DEFAULT_SNP_MEASUREMENT.hex()
        assert claims["measurements"]["host_data"] == DEFAULT_SNP_HOST_DATA.hex()
        assert "vcek_extensions" in claims
        assert claims["vcek_extensions"]["product_name"] == "Milan"

    def test_vcek_custom_oid_extensions_parsed(self, amd_pki):
        """Validates decoding of all 7 AMD custom OIDs under 1.3.6.1.4.1.3704.1."""
        exts = parse_vcek_extensions(amd_pki.vcek_cert)

        assert exts["struct_version"] == 1
        assert exts["product_name"] == "Milan"
        assert exts["blSPL"] == 3
        assert exts["teeSPL"] == 0
        assert exts["snpSPL"] == 8
        assert exts["ucodeSPL"] == 115
        assert exts["hwID"] == amd_pki.chip_id

    def test_scalar_little_endian_roundtrip(self, amd_pki):
        """Verifies mathematical conversion between 48B LE wire scalars and canonical DER DSS."""
        report = generate_synthetic_snp_report(
            vcek_priv_ec=amd_pki.vcek_priv_ec,
            policy=DEFAULT_SNP_POLICY,
        )
        assert len(report) == 1184

        raw_r = report[0x2A0 : 0x2A0 + 48]
        raw_s = report[0x2E8 : 0x2E8 + 48]
        r_int = int.from_bytes(raw_r, byteorder="little")
        s_int = int.from_bytes(raw_s, byteorder="little")

        der_sig = encode_dss_signature(r_int, s_int)
        amd_pki.vcek_cert.public_key().verify(
            der_sig,
            report[:672],
            ec.ECDSA(hashes.SHA384()),
        )

    def test_dual_field_report_data_binding(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Verifies dual-field packing: REPORT_DATA[0..31] = nonce, REPORT_DATA[32..63] = pubkey hash."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        assert len(evidence.user_data) == 64
        assert evidence.user_data[:32] == sample_nonce
        assert evidence.user_data[32:64] == hashlib.sha256(sample_client_pubkey).digest()

        result = verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert result.verified is True


# ==============================================================================
# TEST SUITE 2: STRICT POLICY AND ZERO PADDING REJECTION
# ==============================================================================

class TestSevSnpPolicyAndPadRejection:
    """Verifies strict fail-closed rejection of debug policy and non-zero padding."""

    def test_reject_debug_policy_enabled(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Reports with POLICY.DEBUG == 1 must be immediately rejected with AttestationSignatureError."""
        evidence = create_debug_policy_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationSignatureError, match="DEBUG policy enabled"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_extract_claims_rejects_debug_policy(self, amd_pki, sample_nonce, sample_client_pubkey):
        """extract_claims must also fail-closed on POLICY.DEBUG == 1."""
        strategy = AmdSevSnpVerifier()
        evidence = create_debug_policy_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationSignatureError, match="DEBUG policy enabled"):
            strategy.extract_claims(evidence)

    def test_reject_non_zero_padding_r_scalar(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Corrupted zero-padding in upper 24 bytes of R (0x2D0..0x2E8) must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        # Flip a bit in the R padding region (0x2D0..0x2E7)
        blob[0x2D5] ^= 0x01
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="padding"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_non_zero_padding_s_scalar(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Corrupted zero-padding in upper 24 bytes of S (0x318..0x330) must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        # Flip a bit in the S padding region (0x318..0x32F)
        blob[0x320] ^= 0x01
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="padding"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_non_zero_padding_signature_pad(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Corrupted zero-padding in 368-byte signature pad (0x330..0x4A0) must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        # Flip a bit in signature pad buffer (0x330..0x49F)
        blob[0x350] ^= 0x01
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="padding"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_invalid_signature_algorithm(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Unsupported signature algorithm ID (not 1) must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        # Change signature_algo at 0x034 from 1 to 2
        struct.pack_into("<I", blob, 0x034, 2)
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="signature algorithm"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_zero_scalar_values(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Scalars r=0 or s=0 must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        # Set scalar R to all zeros
        blob[0x2A0:0x2D0] = b"\x00" * 48
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# TEST SUITE 3: 1-BIT MUTATIONS IN SIGNED PAYLOAD & SIGNATURE
# ==============================================================================

class TestSevSnpSignedBodyMutations:
    """Verifies that 1-bit mutations across critical regions are caught."""

    @pytest.mark.parametrize(
        "offset,name",
        [
            (0x000, "version"),
            (0x004, "guest_svn"),
            (0x008, "policy"),
            (0x030, "vmpl"),
            (0x034, "signature_algo"),
            (0x038, "platform_version"),
            (0x040, "platform_info"),
            (0x048, "flags"),
            (0x050, "report_data_nonce_start"),
            (0x06F, "report_data_nonce_end"),
            (0x070, "report_data_pubkey_start"),
            (0x08F, "report_data_pubkey_end"),
            (0x090, "measurement_start"),
            (0x0BF, "measurement_end"),
            (0x0C0, "host_data_start"),
            (0x0DF, "host_data_end"),
            (0x180, "reported_tcb"),
            (0x1A0, "chip_id_start"),
            (0x1DF, "chip_id_end"),
            (0x29F, "payload_final_byte"),
        ],
    )
    def test_single_bit_mutation_signed_payload(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey, offset, name
    ):
        """Any 1-bit mutation in bytes 0x000..0x29F must raise AttestationSignatureError or AttestationError."""
        evidence = create_tampered_snp_payload_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            tamper_offset=offset,
            tamper_mask=0x01,
        )

        with pytest.raises(AttestationError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize(
        "offset,name",
        [
            (0x2A0, "scalar_r_byte_0"),
            (0x2A1, "scalar_r_byte_1"),
            (0x2CF, "scalar_r_byte_47"),
            (0x2E8, "scalar_s_byte_0"),
            (0x2E9, "scalar_s_byte_1"),
            (0x317, "scalar_s_byte_47"),
        ],
    )
    def test_single_bit_mutation_signature_scalars(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey, offset, name
    ):
        """Any 1-bit mutation in R or S scalars must invalidate ECDSA signature."""
        evidence = create_tampered_snp_signature_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            tamper_offset=offset,
            tamper_mask=0x01,
        )

        with pytest.raises(AttestationSignatureError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# TEST SUITE 4: FRESHNESS AND PUBLIC KEY BINDING
# ==============================================================================

class TestSevSnpFreshnessAndPubkeyBinding:
    """Verifies challenger freshness nonce and client public key binding in REPORT_DATA."""

    def test_reject_mismatched_nonce(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Attestation evidence with wrong/replayed nonce must raise AttestationNonceMismatchError."""
        evidence = create_nonce_mismatch_snp_evidence(
            requested_nonce=sample_nonce,
            pki=amd_pki,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationNonceMismatchError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_mismatched_client_pubkey_hash(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Attestation evidence bound to a different public key must raise AttestationBindingError."""
        evidence = create_pubkey_hash_mismatch_snp_evidence(
            nonce=sample_nonce,
            pki=amd_pki,
        )

        with pytest.raises(AttestationBindingError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_nonce_hex_and_bytes_interchangeable(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Nonce can be supplied as raw bytes or hex string and compares cleanly."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        # Pass as hex string with 0x prefix
        result = verifier.verify(
            evidence=evidence,
            expected_nonce=f"0x{sample_nonce.hex()}",
            expected_client_pubkey=sample_client_pubkey.hex(),
        )
        assert result.verified is True


# ==============================================================================
# TEST SUITE 5: GOLDEN RIM APPRAISAL
# ==============================================================================

class TestSevSnpGoldenRimAppraisal:
    """Verifies launch measurement and host data appraisal against reference Golden RIM."""

    def test_golden_rim_valid_measurement_appraisal(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Authentic measurements matching Golden RIM pass appraisal."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        result = verifier.verify(
            evidence=evidence,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_client_pubkey,
        )
        assert result.measurements_verified == 2

    def test_reject_tampered_launch_measurement(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Mismatched launch measurement digest must raise AttestationMeasurementMismatchError."""
        evidence = create_measurement_mismatch_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationMeasurementMismatchError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_client_pubkey,
            )

    def test_reject_tampered_host_data(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Mismatched host data must raise AttestationMeasurementMismatchError."""
        bad_host_data = hashlib.sha256(b"rogue-hypervisor-host-data").digest()
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            host_data=bad_host_data,
        )

        with pytest.raises(AttestationMeasurementMismatchError):
            verifier.verify(
                evidence=evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_client_pubkey,
            )


# ==============================================================================
# TEST SUITE 6: WIRE FORMAT & MALFORMED REPORT BLOBS
# ==============================================================================

class TestSevSnpWireFormatAndBlobIntegrity:
    """Verifies handling of truncated, oversized, and missing evidence blobs."""

    @pytest.mark.parametrize("length", [0, 50, 100, 671, 672, 1183])
    def test_reject_truncated_blob(self, verifier, amd_pki, sample_nonce, sample_client_pubkey, length):
        """Report blob shorter than 1184 bytes must raise AttestationEvidenceMalformedError."""
        evidence = create_truncated_snp_report_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            truncated_length=length,
        )

        with pytest.raises(AttestationEvidenceMalformedError, match="1184 bytes"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_oversized_blob(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Report blob longer than 1184 bytes must raise AttestationEvidenceMalformedError."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        assert evidence.evidence_blob is not None
        evidence.evidence_blob = evidence.evidence_blob + b"\x00"

        with pytest.raises(AttestationEvidenceMalformedError, match="1184 bytes"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_none_evidence_blob(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """evidence_blob = None must raise AttestationEvidenceMalformedError."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.evidence_blob = None

        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# TEST SUITE 7: PKI TRUST CHAIN VALIDATION
# ==============================================================================

class TestSevSnpPkiTrustChain:
    """Verifies X.509 certificate chain validation across the 3-tier AMD hierarchy."""

    def test_valid_3tier_ark_ask_vcek_chain(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Valid ARK -> ASK -> VCEK chain successfully anchors to trusted ARK Root CA."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        result = verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert result.verified is True
        assert result.chain_length == 3
        assert result.root_cert.subject == amd_pki.ark_cert.subject

    def test_reject_untrusted_ark_root(self, verifier, sample_nonce, sample_client_pubkey):
        """Chain rooted in an unknown/rogue ARK Root CA must raise AttestationUntrustedRootError."""
        evidence, rogue_ark_pem = create_untrusted_ark_snp_evidence(
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationUntrustedRootError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_expired_vcek_cert(self, verifier, sample_nonce, sample_client_pubkey):
        """Expired certificate in chain must raise AttestationExpiredError."""
        evidence = create_expired_vcek_snp_evidence(
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationExpiredError, match="expired"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_not_yet_valid_vcek_cert(self, verifier, sample_nonce, sample_client_pubkey):
        """Not-yet-valid certificate in chain must raise AttestationExpiredError."""
        evidence = create_not_yet_valid_vcek_snp_evidence(
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationExpiredError, match="not yet valid"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_revoked_vcek_cert(self, amd_pki, sample_nonce, sample_client_pubkey):
        """Revoked VCEK certificate serial number must raise AttestationRevokedError."""
        evidence, revoked_serial = create_revoked_vcek_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        policy = VerificationPolicy(revoked_serial_numbers={revoked_serial})
        policy.add_trusted_root_pem(amd_pki.ark_cert_pem)
        revoked_verifier = AttestationVerifier(policy=policy)

        with pytest.raises(AttestationRevokedError, match="revoked"):
            revoked_verifier.verify(
                evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_client_pubkey,
            )

    def test_reject_broken_intermediate_chain(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Tampered intermediate ASK signature must raise AttestationSignatureError or AttestationChainInvalidError."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        # Corrupt ASK certificate by substituting a different cert
        rogue_pki = generate_synthetic_amd_pki()
        evidence.certificate_chain[1] = rogue_pki.ask_cert_pem

        with pytest.raises((AttestationChainInvalidError, AttestationSignatureError)):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# TEST SUITE 8: TCB DOWNGRADE & HARDWARE ID BINDING
# ==============================================================================

class TestSevSnpTcbDowngradeAndHardwareMismatch:
    """Verifies appraisal of VCEK custom OID extensions against report body."""

    def test_reject_tcb_downgrade(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Report with reported_tcb lower than VCEK SPLs must raise AttestationCertificateError."""
        evidence = create_tcb_downgrade_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationCertificateError, match="TCB downgrade detected"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_chip_id_hw_id_mismatch(self, verifier, amd_pki, sample_nonce, sample_client_pubkey):
        """Report with chip_id differing from VCEK hwID must raise AttestationCertificateError."""
        rogue_chip_id = hashlib.sha512(b"different-physical-processor-id").digest()
        blob = generate_synthetic_snp_report(
            vcek_priv_ec=amd_pki.vcek_priv_ec,
            chip_id=rogue_chip_id,
        )
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.evidence_blob = blob

        with pytest.raises(AttestationCertificateError, match="Hardware ID mismatch"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
