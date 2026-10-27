"""
Adversarial Stress Test Suite for Intel TDX Attestation Verifier (Milestone 2 Challenge).
Authored by Milestone 2 Challenger 1 (teamwork_preview_challenger).

Empirically challenges:
1. Malformed quote wire buffers:
   - Truncated headers across all lengths 0..47 bytes
   - Truncated bodies (<632 bytes for v4, <638 bytes for v5)
   - Invalid quote versions (0, 1, 2, 3, 6, 7, 255, 65535)
   - Invalid TEE types (0x00 SGX, 0x01, 0x80, 0x82, 0xFF, 0xFFFFFFFF)
   - Invalid modular v5 body types and body sizes
   - Inconsistent signature data lengths and truncated QE payloads
2. Single-bit payload tampering:
   - Bit flips (^ 0x01, ^ 0x04, ^ 0x80) across all signed payload fields
   - Verifies AttestationSignatureError is raised fail-closed
3. Dual signature attacks:
   - Corrupt AK ECDSA signature and out-of-bounds curve scalars
   - Corrupt PCK ECDSA signature and out-of-bounds curve scalars
   - Swapped AK public key coordinates (x <-> y)
   - Substituted foreign AK keypair (piggybacking on PCK cert) caught by QE AK cross-binding
4. DEBUG mode rejection:
   - Unconditional fail-closed rejection of (tdattributes & 0x01) != 0 across all bitmasks
   - Rejection in verify_token_signature, extract_claims, and verifier.verify
5. Freshness nonce replays & MITM ephemeral pubkey substitutions:
   - Nonce mismatches, 1-bit mutations, and irregular lengths
   - Ephemeral public key mismatches, 1-bit mutations, and irregular lengths
   - Token signature field inconsistency against embedded quote signature
"""

from __future__ import annotations

import copy
import hashlib
import secrets
import struct
from typing import List

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from matrix_compute.crypto import (
    AttestationVerifier,
    IntelTdxVerifier,
    VerificationPolicy,
    VerificationResult,
    get_default_intel_pki,
    generate_synthetic_intel_pki,
    generate_synthetic_tdx_quote,
    create_valid_intel_tdx_evidence,
    INTEL_TDX_GOLDEN_RIM,
    INTEL_TDX_TEE_TYPE,
)
from matrix_compute.crypto.tdx_verifier import (
    SECP256R1_ORDER,
    SECP384R1_ORDER,
)
from matrix_compute.exceptions import (
    AttestationBindingError,
    AttestationError,
    AttestationEvidenceMalformedError,
    AttestationNonceMismatchError,
    AttestationSignatureError,
)
from matrix_contracts.schemas.attestation import AttestationEvidence


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def intel_pki():
    return get_default_intel_pki()


@pytest.fixture
def default_policy(intel_pki) -> VerificationPolicy:
    policy = VerificationPolicy()
    policy.add_trusted_root_pem(intel_pki.root_cert_pem)
    policy.add_golden_rim("intel-tdx", INTEL_TDX_GOLDEN_RIM)
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
# 1. MALFORMED QUOTE WIRE BUFFERS & BOUNDARY TESTING
# ==============================================================================

class TestAdversarialWireInputs:
    """Stress-tests wire parsers against arbitrary truncations, corrupt headers, and sizes."""

    @pytest.mark.parametrize("header_len", list(range(0, 48)))
    def test_truncated_headers_all_lengths_0_to_47(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, header_len
    ):
        """Header truncation at any byte from 0 to 47 must fail-closed with AttestationEvidenceMalformedError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        truncated_blob = valid_ev.evidence_blob[:header_len]
        bad_evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=truncated_blob,
        )

        strategy = IntelTdxVerifier()

        # Check strategy.verify_token_signature directly
        with pytest.raises(AttestationEvidenceMalformedError):
            strategy.verify_token_signature(bad_evidence, intel_pki.pck_cert)

        # Check strategy.extract_claims directly
        with pytest.raises(AttestationEvidenceMalformedError):
            strategy.extract_claims(bad_evidence)

        # Check full AttestationVerifier pipeline
        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(bad_evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("body_len", [48, 54, 100, 200, 300, 400, 500, 600, 631, 632, 635])
    def test_truncated_bodies_v4_lengths(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, body_len
    ):
        """Quote v4 truncated before complete signed body + signature length must raise AttestationEvidenceMalformedError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            quote_version=4,
        )
        truncated_blob = valid_ev.evidence_blob[:body_len]
        bad_evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=truncated_blob,
        )
        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(bad_evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("v5_len", [48, 50, 53, 54, 100, 300, 637, 638, 641])
    def test_truncated_bodies_v5_modular_descriptor(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, v5_len
    ):
        """Quote v5 truncated before modular descriptor or full body must raise AttestationEvidenceMalformedError."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob_v5 = generate_synthetic_tdx_quote(
            pki=intel_pki,
            quote_version=5,
            tdx_type=0x0002,
            report_data=report_data,
        )
        truncated_blob = blob_v5[:v5_len]
        bad_evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=truncated_blob,
        )
        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(bad_evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("invalid_version", [0, 1, 2, 3, 6, 7, 255, 65535])
    def test_invalid_quote_versions(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, invalid_version
    ):
        """Unsupported quote versions must raise AttestationEvidenceMalformedError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)
        struct.pack_into("<H", blob, 0, invalid_version)
        bad_evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )
        with pytest.raises(AttestationEvidenceMalformedError, match="version"):
            verifier.verify(bad_evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("invalid_tee", [0x00000000, 0x00000001, 0x00000080, 0x00000082, 0x000000FF, 0xFFFFFFFF])
    def test_invalid_tee_types(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, invalid_tee
    ):
        """Invalid TEE types (e.g. 0x00 SGX or non-TDX) must raise AttestationEvidenceMalformedError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)
        struct.pack_into("<I", blob, 4, invalid_tee)
        bad_evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )
        with pytest.raises(AttestationEvidenceMalformedError, match="TEE"):
            verifier.verify(bad_evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("invalid_body_type", [0x0000, 0x0001, 0x0004, 0xFFFF])
    def test_invalid_v5_body_types(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, invalid_body_type
    ):
        """Unsupported Quote v5 body descriptor type must raise AttestationEvidenceMalformedError."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob_v5 = bytearray(generate_synthetic_tdx_quote(
            pki=intel_pki,
            quote_version=5,
            tdx_type=0x0002,
            report_data=report_data,
        ))
        struct.pack_into("<H", blob_v5, 48, invalid_body_type)
        bad_evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob_v5),
        )
        with pytest.raises(AttestationEvidenceMalformedError, match="Body Type"):
            verifier.verify(bad_evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("invalid_body_size", [0, 500, 583, 585, 1024])
    def test_invalid_v5_body_sizes(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, invalid_body_size
    ):
        """Invalid TD Report Body size in v5 quote descriptor must raise AttestationEvidenceMalformedError."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob_v5 = bytearray(generate_synthetic_tdx_quote(
            pki=intel_pki,
            quote_version=5,
            tdx_type=0x0002,
            report_data=report_data,
        ))
        struct.pack_into("<I", blob_v5, 50, invalid_body_size)
        bad_evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob_v5),
        )
        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(bad_evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_inconsistent_sig_data_len_and_qe_auth_data(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Quote declaring sig_data_len exceeding blob length must fail AttestationEvidenceMalformedError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)
        # Corrupt sig_data_len at offset 632 to huge value
        struct.pack_into("<I", blob, 632, 999999)
        bad_evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )
        with pytest.raises(AttestationEvidenceMalformedError, match="Incomplete Quote Signature"):
            verifier.verify(bad_evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_insufficient_sig_data_len(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Quote declaring sig_data_len smaller than required minimum must raise AttestationEvidenceMalformedError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)
        struct.pack_into("<I", blob, 632, 64)  # insufficient length for sigs + QE report
        bad_evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )
        with pytest.raises(AttestationEvidenceMalformedError, match="Truncated Quote Signature"):
            verifier.verify(bad_evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# 2. SINGLE-BIT PAYLOAD TAMPERING ACROSS SIGNED TD REPORT BODY
# ==============================================================================

class TestAdversarialSingleBitMutations:
    """Stress-tests every field in the signed TD report body with 1-bit mutations."""

    # Offsets in TD Report Body: 48 + offset_in_body
    CRITICAL_BODY_OFFSETS = [
        (48, "TEE_TCB_SVN[0]"),
        (63, "TEE_TCB_SVN[15]"),
        (64, "MRSEAM[0]"),
        (111, "MRSEAM[47]"),
        (112, "MRSIGNERSEAM[0]"),
        (159, "MRSIGNERSEAM[47]"),
        (160, "SEAMATTRIBUTES[0]"),
        (168, "TDATTRIBUTES[0]"),  # Bit 0 flip triggers DEBUG mode rejection, bit 1..7 triggers ECDSA failure
        (175, "TDATTRIBUTES[7]"),
        (176, "XFAM[0]"),
        (184, "MRTD[0]"),
        (231, "MRTD[47]"),
        (232, "MRCONFIGID[0]"),
        (279, "MRCONFIGID[47]"),
        (280, "MROWNER[0]"),
        (327, "MROWNER[47]"),
        (328, "MROWNERCONFIG[0]"),
        (375, "MROWNERCONFIG[47]"),
        (376, "RTMR0[0]"),
        (423, "RTMR0[47]"),
        (424, "RTMR1[0]"),
        (471, "RTMR1[47]"),
        (472, "RTMR2[0]"),
        (519, "RTMR2[47]"),
        (520, "RTMR3[0]"),
        (567, "RTMR3[47]"),
        (568, "REPORTDATA_NONCE[0]"),
        (599, "REPORTDATA_NONCE[31]"),
        (600, "REPORTDATA_PUBKEY[0]"),
        (631, "REPORTDATA_PUBKEY[31]"),
    ]

    @pytest.mark.parametrize("offset,field_name", CRITICAL_BODY_OFFSETS)
    @pytest.mark.parametrize("bit_mask", [0x01, 0x08, 0x80])
    def test_single_bit_flip_in_td_report_body_raises_signature_error(
        self, intel_pki, sample_nonce, sample_client_pubkey, offset, field_name, bit_mask
    ):
        """Any 1-bit mutation across the signed body must unconditionally raise AttestationSignatureError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            quote_version=4,
        )
        blob = bytearray(valid_ev.evidence_blob)
        blob[offset] ^= bit_mask

        tampered_ev = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
            token_signature=valid_ev.token_signature,
            user_data=valid_ev.user_data,
        )

        strategy = IntelTdxVerifier()
        # Direct verify_token_signature must raise AttestationSignatureError
        with pytest.raises(AttestationSignatureError):
            strategy.verify_token_signature(tampered_ev, intel_pki.pck_cert)

    @pytest.mark.parametrize("offset,field_name", [
        (54, "v5_TEE_TCB_SVN"),
        (174, "v5_TDATTRIBUTES"),
        (190, "v5_MRTD"),
        (382, "v5_RTMR0"),
        (574, "v5_REPORTDATA_NONCE"),
        (637, "v5_FINAL_BYTE"),
    ])
    def test_v5_single_bit_flip_in_modular_body(
        self, intel_pki, sample_nonce, sample_client_pubkey, offset, field_name
    ):
        """Single bit mutation in modular v5 quote body must raise AttestationSignatureError."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob_v5 = bytearray(generate_synthetic_tdx_quote(
            pki=intel_pki,
            quote_version=5,
            tdx_type=0x0003,  # TDX 1.5
            report_data=report_data,
        ))
        blob_v5[offset] ^= 0x01

        tampered_ev = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob_v5),
        )

        strategy = IntelTdxVerifier()
        with pytest.raises(AttestationSignatureError):
            strategy.verify_token_signature(tampered_ev, intel_pki.pck_cert)


# ==============================================================================
# 3. DUAL SIGNATURE ATTACKS & KEY SWAPPING
# ==============================================================================

class TestAdversarialDualSignatures:
    """Stress-tests AK signature, PCK signature, and AK public key cross-binding."""

    @pytest.mark.parametrize("offset", [636, 650, 667, 680, 699])
    def test_corrupt_ak_signature_bit_mutations(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, offset
    ):
        """Bit flips in AK quote signature must raise AttestationSignatureError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)
        blob[offset] ^= 0x01
        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )
        with pytest.raises(AttestationSignatureError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize(
        "r_val,s_val",
        [
            (0, 1),
            (1, 0),
            (0, 0),
            (SECP256R1_ORDER, 1),
            (1, SECP256R1_ORDER),
            (SECP256R1_ORDER + 1, 1),
            (1, SECP256R1_ORDER + 100),
        ],
    )
    def test_ak_signature_out_of_bounds_scalars(
        self, intel_pki, sample_nonce, sample_client_pubkey, r_val, s_val
    ):
        """Out-of-range scalars in quote signature must raise AttestationSignatureError without crashing."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)
        # quote_sig is at offset 636, 64 bytes (32 r, 32 s)
        blob[636:668] = r_val.to_bytes(32, "big", signed=False)
        blob[668:700] = s_val.to_bytes(32, "big", signed=False)

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )
        strategy = IntelTdxVerifier()
        with pytest.raises(AttestationSignatureError, match="out of curve range"):
            strategy.verify_token_signature(evidence, intel_pki.pck_cert)

    @pytest.mark.parametrize("offset", [1148, 1160, 1180, 1200, 1211])
    def test_corrupt_pck_signature_bit_mutations(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, offset
    ):
        """Bit flips in PCK signature over QE report must raise AttestationSignatureError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)
        blob[offset] ^= 0x01
        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )
        with pytest.raises(AttestationSignatureError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize(
        "r_val,s_val",
        [
            (0, 1),
            (1, 0),
            (SECP256R1_ORDER, 1),
            (1, SECP256R1_ORDER + 1),
        ],
    )
    def test_pck_signature_out_of_bounds_scalars(
        self, intel_pki, sample_nonce, sample_client_pubkey, r_val, s_val
    ):
        """Out-of-range scalars in PCK QE report signature must raise AttestationSignatureError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)
        # qe_report_sig is at offset 1148, 64 bytes
        blob[1148:1180] = r_val.to_bytes(32, "big", signed=False)
        blob[1180:1212] = s_val.to_bytes(32, "big", signed=False)

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )
        strategy = IntelTdxVerifier()
        with pytest.raises(AttestationSignatureError, match="out of curve range"):
            strategy.verify_token_signature(evidence, intel_pki.pck_cert)

    def test_swapped_ak_public_key_coordinates(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Swapping (x, y) coordinates of AK public key must fail fail-closed."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)
        # AK pubkey at offset 700..764 (32 x, 32 y)
        x_coord = blob[700:732]
        y_coord = blob[732:764]
        # Swap x and y
        blob[700:732] = y_coord
        blob[732:764] = x_coord

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )
        # Must raise AttestationSignatureError (curve check / signature check failure) or AttestationBindingError
        with pytest.raises((AttestationSignatureError, AttestationBindingError)):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_substituted_foreign_ak_keypair_detected_by_qe_binding(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Attacker signs TD body with rogue AK keypair; QE report AK binding mismatch must trigger AttestationBindingError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)

        # Generate rogue EC keypair
        rogue_key = ec.generate_private_key(ec.SECP256R1())
        rogue_pub = rogue_key.public_key()
        rogue_numbers = rogue_pub.public_numbers()
        rogue_pub_raw = rogue_numbers.x.to_bytes(32, "big") + rogue_numbers.y.to_bytes(32, "big")

        # Sign original signed_body (bytes 0..632) with rogue key
        signed_body = bytes(blob[:632])
        der_sig = rogue_key.sign(signed_body, ec.ECDSA(hashes.SHA256()))
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
        r, s = decode_dss_signature(der_sig)
        rogue_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")

        # Overwrite quote_sig and att_pubkey in blob
        blob[636:700] = rogue_sig
        blob[700:764] = rogue_pub_raw

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )

        with pytest.raises(AttestationBindingError, match="Quote Enclave report does not bind Attestation Public Key"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# 4. DEBUG MODE REJECTION
# ==============================================================================

class TestAdversarialDebugPolicyRejection:
    """Stress-tests exhaustive bit patterns of TDATTRIBUTES.DEBUG != 0."""

    @pytest.mark.parametrize(
        "debug_attributes",
        [
            0x0000000000000001,  # DEBUG bit 0 only
            0x0000000000000003,  # DEBUG + bit 1
            0x0000000000000005,  # DEBUG + bit 2
            0x0000000000000011,  # DEBUG + bit 4
            0x0000000000000101,  # DEBUG + bit 8
            0x0000000100000001,  # DEBUG + bit 32
            0x8000000000000001,  # DEBUG + bit 63
            0xFFFFFFFFFFFFFFFF,  # All bits 1
        ],
    )
    def test_debug_mode_exhaustive_bit_combinations(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, debug_attributes
    ):
        """Any tdattributes with bit 0 set must unconditionally raise AttestationSignatureError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        blob = bytearray(valid_ev.evidence_blob)
        # body_offset = 48, tdattributes at body_offset + 120 = 168
        struct.pack_into("<Q", blob, 168, debug_attributes)

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(blob),
        )

        strategy = IntelTdxVerifier()

        # 1. verify_token_signature must reject immediately
        with pytest.raises(AttestationSignatureError, match="DEBUG mode enabled"):
            strategy.verify_token_signature(evidence, intel_pki.pck_cert)

        # 2. extract_claims must reject immediately
        with pytest.raises(AttestationSignatureError, match="DEBUG mode enabled"):
            strategy.extract_claims(evidence)

        # 3. Full verifier.verify must reject immediately
        with pytest.raises(AttestationSignatureError, match="DEBUG mode enabled"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize(
        "non_debug_attributes",
        [
            0x0000000000000002,  # Bit 1 set, bit 0 clear
            0x0000000000000004,  # Bit 2 set, bit 0 clear
            0x0000000000000080,  # Bit 7 set, bit 0 clear
        ],
    )
    def test_non_debug_attributes_do_not_trigger_debug_rejection(
        self, intel_pki, sample_nonce, sample_client_pubkey, non_debug_attributes
    ):
        """Attributes where bit 0 is 0 must NOT trigger DEBUG mode rejection in extract_claims."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob = generate_synthetic_tdx_quote(
            pki=intel_pki,
            quote_version=4,
            report_data=report_data,
            debug=False,
        )
        mod_blob = bytearray(blob)
        struct.pack_into("<Q", mod_blob, 168, non_debug_attributes)

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=intel_pki.certificate_chain,
            evidence_blob=bytes(mod_blob),
        )
        strategy = IntelTdxVerifier()
        # extract_claims should succeed and record debug=False
        claims = strategy.extract_claims(evidence)
        assert claims["debug"] is False
        assert claims["tdattributes"] == non_debug_attributes


# ==============================================================================
# 5. FRESHNESS NONCES & MITM EPHEMERAL PUBKEY BINDINGS
# ==============================================================================

class TestAdversarialNonceAndPubkeyFreshness:
    """Stress-tests nonce replays and ephemeral public key substitutions."""

    def test_replayed_nonce_rejection(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Attacker replays evidence generated for nonce_A to satisfy challenge nonce_B."""
        replayed_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        fresh_challenger_nonce = secrets.token_bytes(32)

        with pytest.raises(AttestationNonceMismatchError):
            verifier.verify(replayed_ev, expected_nonce=fresh_challenger_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("bit_idx", [0, 7, 15, 31])
    def test_single_bit_flip_in_expected_nonce(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, bit_idx
    ):
        """Single bit mutation in challenger nonce must raise AttestationNonceMismatchError."""
        ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        tampered_nonce = bytearray(sample_nonce)
        tampered_nonce[bit_idx] ^= 0x01

        with pytest.raises(AttestationNonceMismatchError):
            verifier.verify(ev, expected_nonce=bytes(tampered_nonce), expected_client_pubkey=sample_client_pubkey)

    def test_mitm_ephemeral_pubkey_substitution(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """Worker proves enclave key K1, but MITM client expected K2; must raise AttestationBindingError."""
        ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        rogue_pubkey = secrets.token_bytes(32)

        with pytest.raises(AttestationBindingError):
            verifier.verify(ev, expected_nonce=sample_nonce, expected_client_pubkey=rogue_pubkey)

    @pytest.mark.parametrize("bit_idx", [0, 7, 15, 31])
    def test_single_bit_flip_in_expected_client_pubkey(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, bit_idx
    ):
        """Single bit mutation in expected client pubkey must raise AttestationBindingError."""
        ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        tampered_pubkey = bytearray(sample_client_pubkey)
        tampered_pubkey[bit_idx] ^= 0x01

        with pytest.raises(AttestationBindingError):
            verifier.verify(ev, expected_nonce=sample_nonce, expected_client_pubkey=bytes(tampered_pubkey))

    @pytest.mark.parametrize("bad_len", [0, 16, 31, 33, 64])
    def test_irregular_length_client_pubkeys(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey, bad_len
    ):
        """Client public key of non-32 byte length must fail binding validation cleanly."""
        ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        bad_pubkey = secrets.token_bytes(bad_len)

        with pytest.raises(AttestationBindingError):
            verifier.verify(ev, expected_nonce=sample_nonce, expected_client_pubkey=bad_pubkey)

    def test_token_signature_mismatch_against_embedded_blob_signature(
        self, verifier, intel_pki, sample_nonce, sample_client_pubkey
    ):
        """evidence.token_signature conflicting with blob quote signature must raise AttestationSignatureError."""
        valid_ev = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        # Corrupt token_signature field while leaving evidence_blob intact
        valid_ev.token_signature = secrets.token_bytes(64)

        with pytest.raises(AttestationSignatureError, match="does not match the quote_signature embedded"):
            verifier.verify(valid_ev, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
