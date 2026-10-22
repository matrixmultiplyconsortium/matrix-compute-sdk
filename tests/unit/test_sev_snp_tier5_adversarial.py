"""
Tier 5 White-Box Adversarial Hardening Test Suite for AMD SEV-SNP Attestation Verifier.
Validates code paths, branches, curve bounds, zero padding checks, OID decoding edge cases,
and Golden RIM appraisals in matrix_compute.crypto.sev_snp_verifier.
"""

from __future__ import annotations

import copy
import hashlib
import secrets
import struct
from typing import Any, Dict

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.x509.oid import NameOID

from matrix_compute.crypto import (
    AmdSevSnpVerifier,
    AttestationVerifier,
    VerificationPolicy,
    VerificationResult,
    generate_synthetic_amd_pki,
    generate_synthetic_snp_report,
    get_default_amd_pki,
    parse_vcek_extensions,
    validate_vcek_extensions,
    create_valid_sev_snp_evidence,
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
)
from matrix_compute.crypto.sev_snp_verifier import (
    SECP384R1_ORDER,
    der_decode_ia5string,
    der_decode_integer,
    der_decode_octet_string,
)
from matrix_compute.exceptions import (
    AttestationBindingError,
    AttestationCertificateError,
    AttestationChainInvalidError,
    AttestationError,
    AttestationEvidenceMalformedError,
    AttestationMeasurementMismatchError,
    AttestationNonceMismatchError,
    AttestationSignatureError,
    AttestationUntrustedRootError,
)
from matrix_contracts.schemas.attestation import AttestationEvidence


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def amd_pki():
    return get_default_amd_pki()


@pytest.fixture
def default_policy(amd_pki) -> VerificationPolicy:
    policy = VerificationPolicy()
    policy.add_trusted_root_pem(amd_pki.ark_cert_pem)
    policy.add_golden_rim("amd-sev-snp", DEFAULT_SNP_GOLDEN_RIM)
    return policy


@pytest.fixture
def verifier(default_policy) -> AttestationVerifier:
    return AttestationVerifier(policy=default_policy)


@pytest.fixture
def sample_nonce() -> bytes:
    return secrets.token_bytes(32)


@pytest.fixture
def sample_client_pubkey() -> bytes:
    return secrets.token_bytes(32)


# ==============================================================================
# 1. CURVE BOUNDS & MATHEMATICAL EDGE CASES
# ==============================================================================

class TestSevSnpCurveBoundsAdversarial:
    """Tests SECP384R1 curve bounds checking on R and S scalars."""

    def test_secp384r1_curve_order_constant_matches_mathematical_order(self):
        """Verify SECP384R1_ORDER exactly equals NIST P-384 mathematical curve order n."""
        # NIST FIPS 186-4 Curve P-384 order n
        expected_n = int(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
            "C7634D81F4372DDF581A0DB248B0A77AECEC196ACCC52973",
            16,
        )
        assert SECP384R1_ORDER == expected_n
        assert SECP384R1_ORDER.bit_length() == 384

    def test_reject_scalar_r_equal_to_curve_order(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Scalar r == SECP384R1_ORDER must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        # Set R to SECP384R1_ORDER in little endian (48 bytes)
        blob[0x2A0:0x2D0] = SECP384R1_ORDER.to_bytes(48, byteorder="little")
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="out of curve range"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_scalar_s_equal_to_curve_order(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Scalar s == SECP384R1_ORDER must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        blob[0x2E8:0x318] = SECP384R1_ORDER.to_bytes(48, byteorder="little")
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="out of curve range"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_scalar_r_greater_than_curve_order(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Scalar r == SECP384R1_ORDER + 1 must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        blob[0x2A0:0x2D0] = (SECP384R1_ORDER + 1).to_bytes(48, byteorder="little")
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="out of curve range"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_scalar_s_greater_than_curve_order(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Scalar s == SECP384R1_ORDER + 1 must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        blob[0x2E8:0x318] = (SECP384R1_ORDER + 1).to_bytes(48, byteorder="little")
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="out of curve range"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_scalar_r_all_ones(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Scalar r == 2^384 - 1 (all 0xFF bytes) exceeds curve order and must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        blob[0x2A0:0x2D0] = b"\xFF" * 48
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="out of curve range"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_scalar_s_all_ones(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Scalar s == 2^384 - 1 (all 0xFF bytes) exceeds curve order and must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        blob[0x2E8:0x318] = b"\xFF" * 48
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="out of curve range"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_scalar_s_zero(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Scalar s == 0 must be rejected."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        blob[0x2E8:0x318] = b"\x00" * 48
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="out of curve range"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# 2. ZERO PADDING BOUNDARIES ADVERSARIAL
# ==============================================================================

class TestSevSnpZeroPaddingBoundaries:
    """Exhaustive boundary testing of zero padding in R, S, and signature pad."""

    @pytest.mark.parametrize(
        "offset,name",
        [
            (0x2D0, "r_pad_first_byte"),
            (0x2D1, "r_pad_second_byte"),
            (0x2E6, "r_pad_penultimate_byte"),
            (0x2E7, "r_pad_final_byte"),
            (0x318, "s_pad_first_byte"),
            (0x319, "s_pad_second_byte"),
            (0x32E, "s_pad_penultimate_byte"),
            (0x32F, "s_pad_final_byte"),
            (0x330, "sig_pad_first_byte"),
            (0x331, "sig_pad_second_byte"),
            (0x3FF, "sig_pad_middle_byte"),
            (0x49E, "sig_pad_penultimate_byte"),
            (0x49F, "sig_pad_final_byte"),
        ],
    )
    def test_zero_padding_exact_boundary_rejection(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey, offset, name
    ):
        """Any non-zero byte at padding boundary offsets must raise AttestationSignatureError."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        blob[offset] = 0x01
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="padding"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# 3. SIGNATURE ALGORITHM & WIRE FORMAT EDGE CASES
# ==============================================================================

class TestSevSnpWireFormatEdgeCases:
    """Tests wire format validation including signature_algo and blob sizing."""

    @pytest.mark.parametrize("bad_algo", [0, 2, 3, 0xFFFFFFFF])
    def test_reject_non_ecdsa_p384_sig_algo(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey, bad_algo
    ):
        """Unsupported signature algorithms must raise AttestationSignatureError."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(evidence.evidence_blob)
        struct.pack_into("<I", blob, 0x034, bad_algo)
        evidence.evidence_blob = bytes(blob)

        with pytest.raises(AttestationSignatureError, match="signature algorithm"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_report_one_byte_too_short(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """1183 bytes report must raise AttestationEvidenceMalformedError."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.evidence_blob = evidence.evidence_blob[:1183]

        with pytest.raises(AttestationEvidenceMalformedError, match="1184 bytes"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_report_one_byte_too_long(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """1185 bytes report must raise AttestationEvidenceMalformedError."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.evidence_blob = evidence.evidence_blob + b"\x00"

        with pytest.raises(AttestationEvidenceMalformedError, match="1184 bytes"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_empty_report_blob(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """b'' report must raise AttestationEvidenceMalformedError."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.evidence_blob = b""

        with pytest.raises(AttestationEvidenceMalformedError, match="1184 bytes"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# 4. POLICY BIT COMBINATIONS & DEBUG ISOLATION
# ==============================================================================

class TestSevSnpPolicyAdversarial:
    """Tests bitwise isolation of POLICY.DEBUG across various bit patterns."""

    @pytest.mark.parametrize(
        "policy_val",
        [
            1 << 19,                                      # Only DEBUG bit
            DEFAULT_SNP_POLICY | (1 << 19),               # Standard policy with DEBUG bit
            0xFFFFFFFFFFFFFFFF,                           # All 64 bits set
            (1 << 19) | 1,                                # DEBUG with ABI minor
            (1 << 19) | (1 << 16),                        # DEBUG with SMT
            (1 << 19) | (1 << 20),                        # DEBUG with SINGLE_SOCKET
        ],
    )
    def test_policy_debug_rejection_all_bit_combinations(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey, policy_val
    ):
        """Any policy with bit 19 set must be rejected with AttestationSignatureError."""
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            policy=policy_val,
        )

        with pytest.raises(AttestationSignatureError, match="DEBUG policy enabled"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize(
        "policy_val",
        [
            1 << 19,
            0xFFFFFFFFFFFFFFFF,
            DEFAULT_SNP_POLICY | (1 << 19),
        ],
    )
    def test_extract_claims_rejects_debug_policy_all_combinations(
        self, amd_pki, sample_nonce, sample_client_pubkey, policy_val
    ):
        """extract_claims also rejects whenever bit 19 is set."""
        strategy = AmdSevSnpVerifier()
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            policy=policy_val,
        )

        with pytest.raises(AttestationSignatureError, match="DEBUG policy enabled"):
            strategy.extract_claims(evidence)


# ==============================================================================
# 5. ASN.1 DER PARSING UTILITIES EDGE CASES
# ==============================================================================

class TestSevSnpDerDecodingEdgeCases:
    """Tests edge cases and boundary conditions in ASN.1 DER decoding utilities."""

    def test_der_decode_integer_valid_cases(self):
        """Standard valid integers."""
        assert der_decode_integer(b"\x02\x01\x00") == 0
        assert der_decode_integer(b"\x02\x01\x01") == 1
        assert der_decode_integer(b"\x02\x01\x7f") == 127
        assert der_decode_integer(b"\x02\x02\x00\x80") == 128
        assert der_decode_integer(b"\x02\x02\x01\x00") == 256
        assert der_decode_integer(b"\x02\x01\xff") == -1

    def test_der_decode_integer_invalid_tags(self):
        """Non-0x02 tags must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid DER integer tag"):
            der_decode_integer(b"\x04\x01\x00")  # Octet String
        with pytest.raises(ValueError, match="Invalid DER integer tag"):
            der_decode_integer(b"\x16\x01\x00")  # IA5String
        with pytest.raises(ValueError, match="Invalid DER integer tag"):
            der_decode_integer(b"")

    def test_der_decode_integer_truncated_payload(self):
        """Payload length smaller than specified in length byte."""
        with pytest.raises(ValueError, match="Truncated DER integer payload"):
            der_decode_integer(b"\x02\x04\x01\x02")  # claims 4 bytes, only 2 provided

    def test_der_decode_integer_truncated_length_bytes(self):
        """Long-form length with missing length octets."""
        with pytest.raises(ValueError, match="Truncated DER integer length bytes"):
            der_decode_integer(b"\x02\x82\x01")  # claims 2 length octets, only 1 provided

    def test_der_decode_ia5string_valid_and_invalid(self):
        """IA5String and PrintableString decoding."""
        assert der_decode_ia5string(b"\x16\x05Milan") == "Milan"
        assert der_decode_ia5string(b"\x13\x05Milan") == "Milan"

        with pytest.raises(ValueError, match="Invalid DER string tag"):
            der_decode_ia5string(b"\x02\x05Milan")  # Integer tag

        with pytest.raises(ValueError, match="Truncated DER string payload"):
            der_decode_ia5string(b"\x16\x0Ashort")

    def test_der_decode_octet_string_valid_and_invalid(self):
        """OCTET STRING decoding."""
        data = b"\x12\x34\x56\x78"
        assert der_decode_octet_string(b"\x04\x04\x12\x34\x56\x78") == data

        with pytest.raises(ValueError, match="Invalid DER octet string tag"):
            der_decode_octet_string(b"\x02\x04\x12\x34\x56\x78")

        with pytest.raises(ValueError, match="Truncated DER octet string payload"):
            der_decode_octet_string(b"\x04\x10\x12\x34")


# ==============================================================================
# 6. VCEK CUSTOM EXTENSIONS & TCB SPL APPRAISALS
# ==============================================================================

class TestSevSnpVcekExtensionsAdversarial:
    """Tests appraisal of VCEK custom OID extensions against report blob."""

    def test_firmware_upgraded_tcb_spls_accepted(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Higher reported TCB SPLs (firmware upgrade) must be accepted cleanly."""
        # VCEK SPLs: BL=3, TEE=0, SNP=8, ucode=115
        # Upgraded reported TCB: BL=4, TEE=1, SNP=9, ucode=120
        upgraded_tcb = bytes([4, 1, 9, 0, 0, 0, 0, 120])
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob = generate_synthetic_snp_report(
            vcek_priv_ec=amd_pki.vcek_priv_ec,
            report_data=report_data,
            reported_tcb=upgraded_tcb,
            chip_id=amd_pki.chip_id,
        )
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.evidence_blob = blob

        result = verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert result.verified is True

    @pytest.mark.parametrize(
        "bad_tcb,error_match",
        [
            (bytes([2, 0, 8, 0, 0, 0, 0, 115]), "bootloader SPL"),   # BL 2 < 3
            (bytes([3, 0, 7, 0, 0, 0, 0, 115]), "SNP SPL"),          # SNP 7 < 8
            (bytes([3, 0, 8, 0, 0, 0, 0, 114]), "ucode SPL"),        # ucode 114 < 115
        ],
    )
    def test_isolated_single_spl_downgrades(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey, bad_tcb, error_match
    ):
        """Downgrading any single SPL while others match must raise AttestationCertificateError."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob = generate_synthetic_snp_report(
            vcek_priv_ec=amd_pki.vcek_priv_ec,
            report_data=report_data,
            reported_tcb=bad_tcb,
            chip_id=amd_pki.chip_id,
        )
        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.evidence_blob = blob

        with pytest.raises(AttestationCertificateError, match=error_match):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_corrupted_extension_in_vcek_cert_raises_certificate_error(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """Malformed ASN.1 in custom AMD extension must raise AttestationCertificateError."""
        # Build a VCEK cert with same public key but malformed struct_version extension (tag 0x05 NULL instead of 0x02)
        ask_key = amd_pki.ask_key
        builder = (
            x509.CertificateBuilder()
            .subject_name(amd_pki.vcek_cert.subject)
            .issuer_name(amd_pki.ask_cert.subject)
            .public_key(amd_pki.vcek_cert.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(amd_pki.vcek_cert.not_valid_before)
            .not_valid_after(amd_pki.vcek_cert.not_valid_after)
            .add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(AMD_OID_STRUCT_VERSION),
                    b"\x05\x00",  # Tag 0x05 NULL instead of 0x02 INTEGER
                ),
                critical=False,
            )
        )
        bad_vcek = builder.sign(ask_key, hashes.SHA384())
        bad_vcek_pem = bad_vcek.public_bytes(serialization.Encoding.PEM).decode("ascii")

        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.certificate_chain[0] = bad_vcek_pem

        with pytest.raises(AttestationCertificateError, match="Failed to decode custom AMD extension"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# 7. VCEK PUBLIC KEY ALGORITHM CONFORMANCE
# ==============================================================================

class TestSevSnpVcekPublicKeyConformance:
    """Verifies that VCEK public key MUST be NIST P-384."""

    def test_reject_vcek_with_rsa_key(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """VCEK certificate with RSA public key must be rejected."""
        rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        builder = (
            x509.CertificateBuilder()
            .subject_name(amd_pki.vcek_cert.subject)
            .issuer_name(amd_pki.ask_cert.subject)
            .public_key(rsa_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(amd_pki.vcek_cert.not_valid_before)
            .not_valid_after(amd_pki.vcek_cert.not_valid_after)
        )
        rsa_vcek = builder.sign(amd_pki.ask_key, hashes.SHA384())
        rsa_vcek_pem = rsa_vcek.public_bytes(serialization.Encoding.PEM).decode("ascii")

        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.certificate_chain[0] = rsa_vcek_pem

        with pytest.raises(AttestationSignatureError, match="must be ECDSA NIST P-384"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_reject_vcek_with_secp256r1_curve(
        self, verifier, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """VCEK certificate with SECP256R1 (P-256) curve must be rejected."""
        p256_key = ec.generate_private_key(ec.SECP256R1())
        builder = (
            x509.CertificateBuilder()
            .subject_name(amd_pki.vcek_cert.subject)
            .issuer_name(amd_pki.ask_cert.subject)
            .public_key(p256_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(amd_pki.vcek_cert.not_valid_before)
            .not_valid_after(amd_pki.vcek_cert.not_valid_after)
        )
        p256_vcek = builder.sign(amd_pki.ask_key, hashes.SHA384())
        p256_vcek_pem = p256_vcek.public_bytes(serialization.Encoding.PEM).decode("ascii")

        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.certificate_chain[0] = p256_vcek_pem

        with pytest.raises(AttestationSignatureError, match="must be ECDSA NIST P-384"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# 8. GOLDEN RIM APPRAISALS & MISSING CLAIMS
# ==============================================================================

class TestSevSnpGoldenRimEdgeCases:
    """Tests Golden RIM appraisals with extra required claims or format variations."""

    def test_golden_rim_rejects_missing_required_measurement(
        self, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """If Golden RIM requires an unsupported claim (e.g. 'firmware_digest'), verifier must reject."""
        policy = VerificationPolicy()
        policy.add_trusted_root_pem(amd_pki.ark_cert_pem)
        # Require an extra claim that SEV-SNP reports do not provide
        policy.add_golden_rim(
            "amd-sev-snp",
            {
                "measurement": DEFAULT_SNP_MEASUREMENT.hex(),
                "host_data": DEFAULT_SNP_HOST_DATA.hex(),
                "firmware_digest": "deadbeef" * 8,
            },
        )
        strict_verifier = AttestationVerifier(policy=policy)

        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationMeasurementMismatchError, match="missing"):
            strict_verifier.verify(
                evidence,
                expected_nonce=sample_nonce,
                expected_client_pubkey=sample_client_pubkey,
            )

    def test_golden_rim_case_insensitive_matching(
        self, amd_pki, sample_nonce, sample_client_pubkey
    ):
        """UPPERCASE hex in Golden RIM policy matches lowercase report hex."""
        policy = VerificationPolicy()
        policy.add_trusted_root_pem(amd_pki.ark_cert_pem)
        policy.add_golden_rim(
            "amd-sev-snp",
            {
                "measurement": DEFAULT_SNP_MEASUREMENT.hex().upper(),
                "host_data": DEFAULT_SNP_HOST_DATA.hex().upper(),
            },
        )
        case_verifier = AttestationVerifier(policy=policy)

        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        result = case_verifier.verify(
            evidence,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_client_pubkey,
        )
        assert result.verified is True
        assert result.measurements_verified == 2


# ==============================================================================
# 9. EMPIRICALLY CONFIRMED GAPS & CORNER CASES (TIER 5 AUDIT FINDINGS)
# ==============================================================================

class TestSevSnpIdentifiedGapsEmpirical:
    """Empirically reproduces and documents edge-case gaps found during white-box audit."""

    def test_gap_1_missing_hw_id_extension_bypasses_chip_id_check(self, amd_pki):
        """
        GAP 1: If VCEK certificate lacks hwID extension, validate_vcek_extensions
        must reject it fail-closed with AttestationCertificateError.
        """
        # Certificate with blSPL but NO hwID extension
        ask_key = amd_pki.ask_key
        builder = (
            x509.CertificateBuilder()
            .subject_name(amd_pki.vcek_cert.subject)
            .issuer_name(amd_pki.ask_cert.subject)
            .public_key(amd_pki.vcek_cert.public_key())
            .serial_number(99991)
            .not_valid_before(amd_pki.vcek_cert.not_valid_before)
            .not_valid_after(amd_pki.vcek_cert.not_valid_after)
            .add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(AMD_OID_BL_SPL),
                    b"\x02\x01\x03",
                ),
                critical=False,
            )
        )
        cert_without_hwid = builder.sign(ask_key, hashes.SHA384())
        rogue_chip_id = b"\xAA" * 64  # Rogue physical chip ID

        report_blob = generate_synthetic_snp_report(
            vcek_priv_ec=amd_pki.vcek_priv_ec,
            chip_id=rogue_chip_id,
        )

        with pytest.raises(AttestationCertificateError, match="missing required AMD custom extensions or hwID"):
            validate_vcek_extensions(cert_without_hwid, report_blob)

    def test_gap_1_empty_vcek_extensions_bypasses_all_tcb_and_hw_checks(self, amd_pki):
        """
        GAP 1b: If VCEK certificate lacks all custom AMD extensions, validate_vcek_extensions
        must reject it fail-closed with AttestationCertificateError.
        """
        ask_key = amd_pki.ask_key
        builder = (
            x509.CertificateBuilder()
            .subject_name(amd_pki.vcek_cert.subject)
            .issuer_name(amd_pki.ask_cert.subject)
            .public_key(amd_pki.vcek_cert.public_key())
            .serial_number(99992)
            .not_valid_before(amd_pki.vcek_cert.not_valid_before)
            .not_valid_after(amd_pki.vcek_cert.not_valid_after)
        )
        cert_no_exts = builder.sign(ask_key, hashes.SHA384())
        rogue_chip_id = b"\xBB" * 64

        report_blob = generate_synthetic_snp_report(
            vcek_priv_ec=amd_pki.vcek_priv_ec,
            chip_id=rogue_chip_id,
            reported_tcb=b"\x00" * 8,  # All SPLs zero (downgraded)
        )

        with pytest.raises(AttestationCertificateError, match="missing required AMD custom extensions or hwID"):
            validate_vcek_extensions(cert_no_exts, report_blob)

    def test_gap_2_negative_spl_integer_bypasses_tcb_downgrade(self, amd_pki):
        """
        GAP 2: Negative SPL integer in VCEK extension must be rejected fail-closed
        with AttestationCertificateError.
        """
        ask_key = amd_pki.ask_key
        builder = (
            x509.CertificateBuilder()
            .subject_name(amd_pki.vcek_cert.subject)
            .issuer_name(amd_pki.ask_cert.subject)
            .public_key(amd_pki.vcek_cert.public_key())
            .serial_number(99993)
            .not_valid_before(amd_pki.vcek_cert.not_valid_before)
            .not_valid_after(amd_pki.vcek_cert.not_valid_after)
            .add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(AMD_OID_BL_SPL),
                    b"\x02\x01\x80",  # Encodes signed -128
                ),
                critical=False,
            )
            .add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(AMD_OID_HW_ID),
                    b"\x04\x40" + amd_pki.chip_id,
                ),
                critical=False,
            )
        )
        cert_negative_spl = builder.sign(ask_key, hashes.SHA384())

        report_blob = generate_synthetic_snp_report(
            vcek_priv_ec=amd_pki.vcek_priv_ec,
            chip_id=amd_pki.chip_id,
            reported_tcb=bytes([0] * 8),  # Reported bootloader SPL = 0
        )

        with pytest.raises(AttestationCertificateError, match="Invalid SPL value"):
            validate_vcek_extensions(cert_negative_spl, report_blob)

    def test_gap_3_der_decoders_accept_indefinite_length_0x80(self):
        """
        GAP 3: ASN.1 DER decoders must strictly reject 0x80 (indefinite length BER form).
        """
        with pytest.raises(ValueError, match="Indefinite length"):
            der_decode_integer(b"\x02\x80")
        with pytest.raises(ValueError, match="Indefinite length"):
            der_decode_ia5string(b"\x16\x80")
        with pytest.raises(ValueError, match="Indefinite length"):
            der_decode_octet_string(b"\x04\x80")

    def test_gap_3_der_decoders_ignore_trailing_unconsumed_bytes(self):
        """
        GAP 3b: ASN.1 DER decoders must strictly reject unconsumed trailing bytes after the TLV payload.
        """
        with pytest.raises(ValueError, match="Unconsumed trailing bytes"):
            der_decode_integer(b"\x02\x01\x05\xFF\xFF\xFF")
        with pytest.raises(ValueError, match="Unconsumed trailing bytes"):
            der_decode_ia5string(b"\x16\x05Milan\xFF\xFF")
        with pytest.raises(ValueError, match="Unconsumed trailing bytes"):
            der_decode_octet_string(b"\x04\x04\x01\x02\x03\x04\xFF")

    def test_gap_4_golden_rim_fails_with_0x_prefix(self, amd_pki, sample_nonce, sample_client_pubkey):
        """
        GAP 4: verify_measurements in attestation_verifier strips '0x' prefixes
        from Golden RIM manifests, allowing valid appraisals.
        """
        policy = VerificationPolicy()
        policy.add_trusted_root_pem(amd_pki.ark_cert_pem)
        policy.add_golden_rim(
            "amd-sev-snp",
            {
                "measurement": "0x" + DEFAULT_SNP_MEASUREMENT.hex(),
                "host_data": "0x" + DEFAULT_SNP_HOST_DATA.hex(),
            },
        )
        verifier_with_0x = AttestationVerifier(policy=policy)

        evidence = create_valid_sev_snp_evidence(
            pki=amd_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        result = verifier_with_0x.verify(
            evidence,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_client_pubkey,
        )
        assert result.verified is True
        assert result.measurements_verified >= 2

