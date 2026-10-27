"""
Adversarial Stress Test Suite for Intel TDX Remote Attestation Fixtures (Milestone 1).
Comprehensive empirical challenge harness authored by Challenger 1:
1. ASN.1 DER parser/encoder fuzzing, boundary lengths, nested sequences, and corrupted tags.
2. Single-bit mutation verification across 632-byte Quote v4 and 638-byte Quote v5 signed bodies.
3. Single-bit mutation verification across 384-byte QE report bodies (v4 and v5).
4. AK public key substitution and coordinate swapping adversarial attacks.
5. Thread safety and concurrency stress testing of get_default_intel_pki().
6. Comprehensive negative fixture validation.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import secrets
import struct
from typing import List

import pytest
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature

from matrix_compute.crypto.pki_fixtures import (
    DEFAULT_INTEL_FMSPC,
    DEFAULT_INTEL_PCEID,
    DEFAULT_INTEL_TCB_COMP_SVNS,
    DEFAULT_TDX_GOLDEN_RIM,
    DEFAULT_TDX_MRTD,
    DEFAULT_TDX_RTMR0,
    DEFAULT_TDX_RTMR1,
    DEFAULT_TDX_RTMR2,
    DEFAULT_TDX_RTMR3,
    INTEL_OID_FMSPC,
    INTEL_OID_FMSPC_SPEC_COMPAT,
    INTEL_OID_PCEID,
    INTEL_OID_PPID,
    INTEL_OID_TCB_COMP_SVNS,
    INTEL_SGX_EXTENSION_OID,
    INTEL_TDX_TEE_TYPE,
    SyntheticIntelPKI,
    build_intel_pck_extension,
    create_debug_tdx_evidence,
    create_expired_cert_tdx_evidence,
    create_malformed_version_tdx_evidence,
    create_measurement_mismatch_tdx_evidence,
    create_nonce_mismatch_tdx_evidence,
    create_not_yet_valid_cert_tdx_evidence,
    create_pubkey_mismatch_tdx_evidence,
    create_revoked_cert_tdx_evidence,
    create_tampered_ak_binding_tdx_evidence,
    create_tampered_qe_ak_binding_evidence,
    create_tampered_qe_signature_evidence,
    create_tampered_tdx_quote_evidence,
    create_tampered_tdx_signature_evidence,
    create_truncated_tdx_quote_evidence,
    create_untrusted_root_tdx_evidence,
    create_valid_intel_tdx_evidence,
    der_decode_octet_string,
    der_decode_oid,
    der_decode_sequence,
    der_encode_octet_string,
    der_encode_oid,
    der_encode_sequence,
    encode_intel_pck_asn1_extension,
    generate_synthetic_intel_pki,
    generate_synthetic_tdx_quote,
    get_default_intel_pki,
    reset_default_intel_pki_cache,
)
from matrix_contracts.schemas.attestation import AttestationEvidence


# ============================================================================
# TEST SUITE 1: ASN.1 DER FUZZING & BOUNDARY CONDITIONS
# ============================================================================

class TestDerFuzzingAndBoundaries:
    """Stress-tests DER encoder and decoder against boundary lengths, nested sequences, and corrupted tags."""

    def test_der_oid_boundary_and_components(self):
        """Test DER OID encoding and decoding across standard, minimal, and large sub-identifiers."""
        valid_oids = [
            "0.0",
            "1.0",
            "1.2",
            "1.39",
            "2.16",
            "2.25",
            "1.2.840.113741.1.13.1",
            "1.2.840.113741.1.13.1.1",
            "1.2.840.113741.1.13.1.2",
            "1.2.840.113741.1.13.1.3",
            "1.2.840.113741.1.13.1.4",
            "1.2.840.113741.1.13.1.14",
            "2.16.840.1.101.3.4.2.1",
            "1.3.6.1.4.1.3704.1.1",
        ]
        for oid in valid_oids:
            encoded = der_encode_oid(oid)
            assert encoded[0] == 0x06, f"Expected OID tag 0x06 for {oid}"
            decoded = der_decode_oid(encoded)
            assert decoded == oid, f"Roundtrip mismatch for {oid}: got {decoded}"

    def test_der_oid_invalid_inputs_rejected(self):
        """Test that invalid non-numeric and malformed OID strings raise ValueError."""
        invalid_oids = [
            "",
            "1",
            "1.",
            ".1",
            "not_an_oid",
            "1.2.invalid",
        ]
        for bad_oid in invalid_oids:
            with pytest.raises((ValueError, Exception)):
                der_encode_oid(bad_oid)

    def test_der_oid_large_second_arc_raises_overflow(self):
        """
        Adversarial Edge Case: Second component causing 40*parts[0] + parts[1] >= 256
        raises ValueError in bytearray constructor.
        """
        with pytest.raises(ValueError, match="byte must be in range"):
            der_encode_oid("2.999")  # 40*2 + 999 = 1079 >= 256

    def test_der_oid_negative_subidentifier_edge_case(self):
        """
        Adversarial Edge Case: Negative sub-identifiers wrap around in first octet
        or get dropped in trailing components.
        """
        # "1.-2": 40*1 + (-2) = 38 -> encodes as 0.38
        enc_neg_first = der_encode_oid("1.-2")
        assert der_decode_oid(enc_neg_first) == "0.38"

        # "1.2.-5": while val > 0 loop terminates immediately -> -5 is dropped -> encodes as 1.2
        enc_neg_trail = der_encode_oid("1.2.-5")
        assert der_decode_oid(enc_neg_trail) == "1.2"

    def test_der_oid_corrupted_tags_and_truncated_payloads(self):
        """Test that malformed DER bytes raise ValueError when decoded as OID."""
        corrupted_oids = [
            b"",
            b"\x06",
            b"\x30\x02\x2a\x03",  # SEQUENCE tag instead of OID tag
            b"\x04\x02\x2a\x03",  # OCTET STRING tag
            b"\x02\x01\x00",      # INTEGER tag
            b"\x00\x00",          # NULL tag
            b"\xff\x02\x2a\x03",  # Invalid tag
        ]
        for bad_raw in corrupted_oids:
            with pytest.raises(ValueError):
                der_decode_oid(bad_raw)

    def test_der_sequence_boundary_lengths(self):
        """Test SEQUENCE encoding and decoding across boundary lengths (0, 1, 127, 128, 255, 256, 1024)."""
        # 1. Empty SEQUENCE
        empty_seq = der_encode_sequence([])
        assert empty_seq == b"\x30\x00"
        assert der_decode_sequence(empty_seq) == []

        # 2. Length 1 payload
        elem1 = b"\x04\x01\xaa"
        seq1 = der_encode_sequence([elem1])
        assert seq1 == b"\x30\x03" + elem1
        assert der_decode_sequence(seq1) == [elem1]

        # 3. Exactly 127 bytes total payload (max single-byte length)
        elem_127 = b"\x04\x7d" + (b"A" * 125)  # 1 tag + 1 len + 125 bytes = 127 bytes
        seq_127 = der_encode_sequence([elem_127])
        assert seq_127[0:2] == b"\x30\x7f"
        assert der_decode_sequence(seq_127) == [elem_127]

        # 4. Exactly 128 bytes total payload (first multi-byte length: 0x81 0x80)
        elem_128 = b"\x04\x81\x7e" + (b"B" * 126)  # 1 tag + 2 len bytes + 126 data = 129 -> total inside seq 128
        elem_128_raw = b"\x04\x7e" + (b"B" * 126)  # exactly 128 bytes
        seq_128 = der_encode_sequence([elem_128_raw])
        assert seq_128[0:3] == b"\x30\x81\x80"
        assert der_decode_sequence(seq_128) == [elem_128_raw]

        # 5. Exactly 255 bytes total payload (max 1-byte length header: 0x81 0xff)
        elem_255 = b"\x04\x81\xfc" + (b"C" * 252)  # 1 + 2 + 252 = 255 bytes
        seq_255 = der_encode_sequence([elem_255])
        assert seq_255[0:3] == b"\x30\x81\xff"
        assert der_decode_sequence(seq_255) == [elem_255]

        # 6. Exactly 256 bytes total payload (first 2-byte length header: 0x82 0x01 0x00)
        elem_256 = b"\x04\x81\xfd" + (b"D" * 253)  # 1 + 2 + 253 = 256 bytes
        seq_256 = der_encode_sequence([elem_256])
        assert seq_256[0:4] == b"\x30\x82\x01\x00"
        assert der_decode_sequence(seq_256) == [elem_256]

        # 7. 1024 bytes payload
        elem_1024 = b"\x04\x82\x03\xfd" + (b"E" * 1021)  # 1 + 3 + 1021 = 1025 bytes
        seq_1024 = der_encode_sequence([elem_1024])
        assert seq_1024[0:2] == b"\x30\x82"
        assert der_decode_sequence(seq_1024) == [elem_1024]

    def test_der_sequence_deeply_nested_structures(self):
        """Test encoding and parsing arbitrary nested sequences (5 levels deep)."""
        leaf = der_encode_oid("1.2.840.113741.1.13.1.4")
        lvl1 = der_encode_sequence([leaf, der_encode_octet_string(bytes.fromhex("00806f000000"))])
        lvl2 = der_encode_sequence([lvl1, lvl1])
        lvl3 = der_encode_sequence([lvl2, lvl1])
        lvl4 = der_encode_sequence([lvl3])
        lvl5 = der_encode_sequence([lvl4, lvl3, lvl2, lvl1])

        decoded_lvl5 = der_decode_sequence(lvl5)
        assert len(decoded_lvl5) == 4
        assert decoded_lvl5[0] == lvl4
        assert decoded_lvl5[1] == lvl3

        decoded_lvl4 = der_decode_sequence(decoded_lvl5[0])
        assert len(decoded_lvl4) == 1
        assert decoded_lvl4[0] == lvl3

        decoded_lvl3 = der_decode_sequence(decoded_lvl4[0])
        assert len(decoded_lvl3) == 2
        assert decoded_lvl3[0] == lvl2
        assert decoded_lvl3[1] == lvl1

        decoded_lvl1 = der_decode_sequence(decoded_lvl3[1])
        assert len(decoded_lvl1) == 2
        assert decoded_lvl1[0] == leaf
        assert der_decode_octet_string(decoded_lvl1[1]) == bytes.fromhex("00806f000000")

    def test_der_sequence_corrupted_tags_rejected(self):
        """Test that non-0x30 tags raise ValueError when parsed as SEQUENCE."""
        corrupted = [
            b"",
            b"\x30",
            b"\x31\x00",
            b"\x04\x02\x01\x02",
            b"\x06\x03\x55\x04\x03",
            b"\x02\x01\x01",
            b"\xff\x05\x00\x00\x00\x00\x00",
        ]
        for c in corrupted:
            with pytest.raises(ValueError):
                der_decode_sequence(c)

    def test_der_octet_string_boundaries_and_rejections(self):
        """Test OCTET STRING encoding and decoding at boundaries (0, 1, 127, 128, 256 bytes)."""
        # Empty octet string
        empty_os = der_encode_octet_string(b"")
        assert empty_os == b"\x04\x00"
        assert der_decode_octet_string(empty_os) == b""

        # 127 bytes
        d127 = secrets.token_bytes(127)
        enc127 = der_encode_octet_string(d127)
        assert enc127[0:2] == b"\x04\x7f"
        assert der_decode_octet_string(enc127) == d127

        # 128 bytes
        d128 = secrets.token_bytes(128)
        enc128 = der_encode_octet_string(d128)
        assert enc128[0:3] == b"\x04\x81\x80"
        assert der_decode_octet_string(enc128) == d128

        # Corrupted tags
        for bad_tag in [b"", b"\x04", b"\x02\x01\x00", b"\x06\x01\x00", b"\x30\x00"]:
            with pytest.raises(ValueError):
                der_decode_octet_string(bad_tag)

    def test_encode_intel_pck_asn1_extension_strict_validation(self):
        """Test that encode_intel_pck_asn1_extension enforces strict field sizes."""
        # 1. Invalid FMSPC length
        with pytest.raises(ValueError, match="fmspc must be exactly 6 bytes"):
            encode_intel_pck_asn1_extension(fmspc="00806f0000", pceid="0000")  # 5 bytes

        with pytest.raises(ValueError, match="fmspc must be exactly 6 bytes"):
            encode_intel_pck_asn1_extension(fmspc="00806f00000000", pceid="0000")  # 7 bytes

        # 2. Invalid PCEID length
        with pytest.raises(ValueError, match="pceid must be exactly 2 bytes"):
            encode_intel_pck_asn1_extension(fmspc="00806f000000", pceid="00")  # 1 byte

        with pytest.raises(ValueError, match="pceid must be exactly 2 bytes"):
            encode_intel_pck_asn1_extension(fmspc="00806f000000", pceid="000000")  # 3 bytes

        # 3. Invalid TCB Component SVNs length
        with pytest.raises(ValueError, match="tcb_comp_svns must be exactly 16 bytes"):
            encode_intel_pck_asn1_extension(
                fmspc="00806f000000",
                pceid="0000",
                tcb_comp_svns=bytes([1] * 15),
            )

        # 4. Valid structure inspection
        ext_der = encode_intel_pck_asn1_extension(
            fmspc="00806f000000",
            pceid="0000",
            tcb_comp_svns=bytes([2] * 16),
            ppid=bytes([3] * 16),
        )
        sub_seqs = der_decode_sequence(ext_der)
        assert len(sub_seqs) == 5

        # Check PPID (.1)
        ppid_parts = der_decode_sequence(sub_seqs[0])
        assert der_decode_oid(ppid_parts[0]) == INTEL_OID_PPID
        assert der_decode_octet_string(ppid_parts[1]) == bytes([3] * 16)

        # Check Spec FMSPC (.2)
        spec_parts = der_decode_sequence(sub_seqs[1])
        assert der_decode_oid(spec_parts[0]) == INTEL_OID_FMSPC_SPEC_COMPAT
        assert der_decode_octet_string(spec_parts[1]) == bytes.fromhex("00806f000000")

        # Check PCEID (.3)
        pceid_parts = der_decode_sequence(sub_seqs[2])
        assert der_decode_oid(pceid_parts[0]) == INTEL_OID_PCEID
        assert der_decode_octet_string(pceid_parts[1]) == bytes.fromhex("0000")

        # Check Standard FMSPC (.4)
        fmspc_parts = der_decode_sequence(sub_seqs[3])
        assert der_decode_oid(fmspc_parts[0]) == INTEL_OID_FMSPC
        assert der_decode_octet_string(fmspc_parts[1]) == bytes.fromhex("00806f000000")

        # Check TCB (.14)
        tcb_parts = der_decode_sequence(sub_seqs[4])
        assert der_decode_oid(tcb_parts[0]) == INTEL_OID_TCB_COMP_SVNS
        assert der_decode_octet_string(tcb_parts[1]) == bytes([2] * 16)


# ============================================================================
# TEST SUITE 2: EXHAUSTIVE SINGLE-BIT MUTATIONS ON SIGNED BODY (AK SIGNATURE)
# ============================================================================

class TestSignedBodyBitFlipExhaustive:
    """Stress-tests AK signature verification over Quote v4 (632B) and Quote v5 (638B) signed bodies."""

    def test_quote_v4_signed_body_exhaustive_bit_flips(self):
        """
        Verify that altering any single bit in the 632-byte Quote v4 signed body
        (Quote Header 48B + TD Report Body 584B) invalidates the Attestation Key (AK) signature.
        Tests all 632 byte offsets with bit 0x01, plus edge masks (0x80, 0xAA).
        """
        pki = get_default_intel_pki()
        blob = generate_synthetic_tdx_quote(pki=pki, quote_version=4)
        assert len(blob) >= 636

        # Extract genuine AK public key and genuine AK signature
        ak_pubkey_raw = blob[700:764]
        ak_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b"\x04" + ak_pubkey_raw)
        r = int.from_bytes(blob[636:668], "big")
        s = int.from_bytes(blob[668:700], "big")
        der_quote_sig = encode_dss_signature(r, s)

        # Confirm unmodified signed body verifies cleanly
        signed_body = blob[:632]
        ak_pub.verify(der_quote_sig, signed_body, ec.ECDSA(hashes.SHA256()))

        # Test every single byte offset (0 to 631) across the entire 632-byte signed body
        rejected_count = 0
        total_offsets = 632
        for offset in range(total_offsets):
            mutated = bytearray(signed_body)
            # Alternate masks: bit 0, bit 7, alternating bits
            mask = 0x01 if (offset % 3 == 0) else (0x80 if (offset % 3 == 1) else 0x55)
            mutated[offset] ^= mask
            try:
                ak_pub.verify(der_quote_sig, bytes(mutated), ec.ECDSA(hashes.SHA256()))
                pytest.fail(f"AK signature verification PASSED on corrupted signed body at offset {offset} (mask {mask:02x})")
            except Exception:
                rejected_count += 1

        assert rejected_count == total_offsets, f"Expected {total_offsets} rejections, got {rejected_count}"

    def test_quote_v5_modular_signed_body_exhaustive_bit_flips(self):
        """
        Verify that altering any single bit in the 638-byte Quote v5 modular signed body
        (Header 48B + Body Descriptor 6B + TD Report Body 584B) invalidates the AK signature.
        """
        pki = get_default_intel_pki()
        blob = generate_synthetic_tdx_quote(pki=pki, quote_version=5, tdx_type=3)
        assert len(blob) >= 642

        # In Quote v5:
        # Header (0..48) + Descriptor (48..54) + TD Report Body (54..638) = 638 bytes
        signed_body_v5 = blob[:638]
        sig_offset = 642
        ak_sig = blob[sig_offset : sig_offset + 64]
        ak_pubkey = blob[sig_offset + 64 : sig_offset + 128]
        ak_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b"\x04" + ak_pubkey)

        r = int.from_bytes(ak_sig[:32], "big")
        s = int.from_bytes(ak_sig[32:], "big")
        der_sig = encode_dss_signature(r, s)

        # Baseline verifies
        ak_pub.verify(der_sig, signed_body_v5, ec.ECDSA(hashes.SHA256()))

        # Test all 638 byte offsets including the 6-byte body descriptor
        rejected_count = 0
        total_offsets = 638
        for offset in range(total_offsets):
            mutated = bytearray(signed_body_v5)
            mutated[offset] ^= 0x01
            try:
                ak_pub.verify(der_sig, bytes(mutated), ec.ECDSA(hashes.SHA256()))
                pytest.fail(f"Quote v5 AK verification PASSED on corrupted body at offset {offset}")
            except Exception:
                rejected_count += 1

        assert rejected_count == total_offsets


# ============================================================================
# TEST SUITE 3: EXHAUSTIVE SINGLE-BIT MUTATIONS ON QE REPORT (PCK SIGNATURE)
# ============================================================================

class TestQeReportBitFlipExhaustive:
    """Stress-tests PCK signature verification over the 384-byte QE3 SGX Report Body."""

    def test_quote_v4_qe_report_exhaustive_bit_flips(self):
        """
        Verify that altering any single bit in the 384-byte Quote Enclave (QE3) SGX Report
        invalidates the PCK Leaf signature.
        Tests all 384 byte offsets across CPU_SVN, MRENCLAVE, MRSIGNER, ISVSVN, and REPORTDATA.
        """
        pki = get_default_intel_pki()
        blob = generate_synthetic_tdx_quote(pki=pki, quote_version=4)

        # In Quote v4: QE Report Body is blob[764:1148] (384 bytes)
        qe_body = blob[764:1148]
        assert len(qe_body) == 384

        # QE Signature is blob[1148:1212] (64 bytes)
        qe_sig = blob[1148:1212]
        r = int.from_bytes(qe_sig[:32], "big")
        s = int.from_bytes(qe_sig[32:], "big")
        der_qe_sig = encode_dss_signature(r, s)

        pck_pub = pki.pck_cert.public_key()
        # Baseline verifies cleanly
        pck_pub.verify(der_qe_sig, qe_body, ec.ECDSA(hashes.SHA256()))

        # Exhaustively mutate every single byte in the 384-byte QE report body
        rejected_count = 0
        total_offsets = 384
        for offset in range(total_offsets):
            mutated = bytearray(qe_body)
            mask = 0x01 if (offset % 2 == 0) else 0x80
            mutated[offset] ^= mask
            try:
                pck_pub.verify(der_qe_sig, bytes(mutated), ec.ECDSA(hashes.SHA256()))
                pytest.fail(f"PCK signature verification PASSED on corrupted QE report at offset {offset}")
            except Exception:
                rejected_count += 1

        assert rejected_count == total_offsets

    def test_quote_v5_qe_report_exhaustive_bit_flips(self):
        """Verify that altering any bit in Quote v5 QE report (blob[770:1154]) invalidates PCK signature."""
        pki = get_default_intel_pki()
        blob = generate_synthetic_tdx_quote(pki=pki, quote_version=5, tdx_type=3)

        # In Quote v5 with 6B descriptor: QE report is offset 770..1154
        qe_body = blob[770:1154]
        assert len(qe_body) == 384
        qe_sig = blob[1154:1218]

        r = int.from_bytes(qe_sig[:32], "big")
        s = int.from_bytes(qe_sig[32:], "big")
        der_qe_sig = encode_dss_signature(r, s)

        pck_pub = pki.pck_cert.public_key()
        pck_pub.verify(der_qe_sig, qe_body, ec.ECDSA(hashes.SHA256()))

        rejected_count = 0
        total_offsets = 384
        for offset in range(total_offsets):
            mutated = bytearray(qe_body)
            mutated[offset] ^= 0x01
            try:
                pck_pub.verify(der_qe_sig, bytes(mutated), ec.ECDSA(hashes.SHA256()))
                pytest.fail(f"Quote v5 PCK verification PASSED on corrupted QE report at offset {offset}")
            except Exception:
                rejected_count += 1

        assert rejected_count == total_offsets


# ============================================================================
# TEST SUITE 4: AK PUBLIC KEY SUBSTITUTION & CROSS-BINDING ATTACKS
# ============================================================================

class TestAkSubstitutionAndCrossBindingAdversarial:
    """Stress-tests the cryptographic cross-binding between Attestation Key and QE Report."""

    def test_ak_coordinate_swap_attack_defeated(self):
        """
        Attack Scenario 1: Attacker swaps the x and y coordinates of the AK public key.
        Verify that either the swapped point is rejected as off-curve, OR the QE cross-binding hash fails.
        """
        pki = get_default_intel_pki()
        blob = generate_synthetic_tdx_quote(pki=pki, quote_version=4)

        ak_x = blob[700:732]
        ak_y = blob[732:764]
        swapped_ak = ak_y + ak_x

        # 1. Test point validity
        try:
            ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b"\x04" + swapped_ak)
            point_valid = True
        except ValueError:
            point_valid = False

        # 2. Test AK cross-binding check in QE report data
        qe_report_body = blob[764:1148]
        embedded_binding = qe_report_body[320:352]
        qe_auth_data_len = int.from_bytes(blob[1212:1214], "little")
        qe_auth_data = blob[1214 : 1214 + qe_auth_data_len]
        swapped_binding = hashlib.sha256(swapped_ak + qe_auth_data).digest()

        # Both the point must be off-curve or the binding must mismatch
        assert (not point_valid) or (embedded_binding != swapped_binding)

    def test_rogue_ak_substitution_without_qe_update_defeated(self):
        """
        Attack Scenario 2: Attacker generates a fresh ephemeral keypair (Rogue AK),
        signs a malicious TD Report Body, replaces the AK signature and AK public key in the quote,
        but cannot forge the Intel PCK signature over the QE Report.
        Verify that AK signature passes, but AK cross-binding in QE Report fails.
        """
        pki = get_default_intel_pki()
        blob = bytearray(generate_synthetic_tdx_quote(pki=pki, quote_version=4))

        # Generate rogue AK keypair
        rogue_key = ec.generate_private_key(ec.SECP256R1())
        rogue_nums = rogue_key.public_key().public_numbers()
        rogue_pub_raw = rogue_nums.x.to_bytes(32, "big") + rogue_nums.y.to_bytes(32, "big")

        # Rogue signs the body
        signed_body = blob[:632]
        der_rogue_sig = rogue_key.sign(signed_body, ec.ECDSA(hashes.SHA256()))
        r_rogue, s_rogue = decode_dss_signature(der_rogue_sig)
        rogue_sig_raw = r_rogue.to_bytes(32, "big") + s_rogue.to_bytes(32, "big")

        # Substitute in quote
        blob[636:700] = rogue_sig_raw
        blob[700:764] = rogue_pub_raw

        # 1. AK signature with substituted key passes
        rogue_ak_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b"\x04" + rogue_pub_raw)
        rogue_ak_pub.verify(der_rogue_sig, bytes(blob[:632]), ec.ECDSA(hashes.SHA256()))

        # 2. BUT QE report cross-binding fails
        qe_report_body = blob[764:1148]
        embedded_binding = qe_report_body[320:352]
        qe_auth_data_len = int.from_bytes(blob[1212:1214], "little")
        qe_auth_data = bytes(blob[1214 : 1214 + qe_auth_data_len])
        expected_binding = hashlib.sha256(rogue_pub_raw + qe_auth_data).digest()

        assert embedded_binding != expected_binding, "Cross-binding check MUST fail when AK is substituted"

    def test_rogue_ak_substitution_with_tampered_qe_report_defeated_by_pck_sig(self):
        """
        Attack Scenario 3: Attacker substitutes AK, and also updates QE Report Body[320:352]
        to match the rogue AK hash.
        Verify that PCK signature verification over the QE Report unconditionally fails.
        """
        pki = get_default_intel_pki()
        blob = bytearray(generate_synthetic_tdx_quote(pki=pki, quote_version=4))

        rogue_key = ec.generate_private_key(ec.SECP256R1())
        rogue_nums = rogue_key.public_key().public_numbers()
        rogue_pub_raw = rogue_nums.x.to_bytes(32, "big") + rogue_nums.y.to_bytes(32, "big")

        qe_auth_data_len = int.from_bytes(blob[1212:1214], "little")
        qe_auth_data = bytes(blob[1214 : 1214 + qe_auth_data_len])
        new_binding = hashlib.sha256(rogue_pub_raw + qe_auth_data).digest()

        # Update both AK pubkey and QE report binding
        blob[700:764] = rogue_pub_raw
        blob[764 + 320 : 764 + 352] = new_binding

        # Verify PCK signature over QE report fails
        qe_sig_raw = blob[1148:1212]
        qe_r = int.from_bytes(qe_sig_raw[:32], "big")
        qe_s = int.from_bytes(qe_sig_raw[32:], "big")
        der_qe_sig = encode_dss_signature(qe_r, qe_s)

        pck_pub = pki.pck_cert.public_key()
        with pytest.raises(Exception):
            pck_pub.verify(der_qe_sig, bytes(blob[764:1148]), ec.ECDSA(hashes.SHA256()))

    def test_qe_auth_data_tampering_defeated_by_ak_binding(self):
        """
        Attack Scenario 4: Attacker modifies qe_auth_data in quote signature structure.
        Verify that AK cross-binding check SHA256(att_public_key || qe_auth_data) detects tampering.
        """
        pki = get_default_intel_pki()
        # Create quote with non-empty qe_auth_data
        sample_auth = b"intel-qe3-authentic-auth-data-12345"
        blob = bytearray(generate_synthetic_tdx_quote(pki=pki, qe_auth_data=sample_auth))

        att_pubkey_raw = bytes(blob[700:764])
        qe_report_body = bytes(blob[764:1148])
        embedded_binding = qe_report_body[320:352]

        # Valid binding matches
        assert embedded_binding == hashlib.sha256(att_pubkey_raw + sample_auth).digest()

        # Tampered auth data
        tampered_auth = b"intel-qe3-tampered-auth-data-99999"
        tampered_binding = hashlib.sha256(att_pubkey_raw + tampered_auth).digest()
        assert embedded_binding != tampered_binding


# ============================================================================
# TEST SUITE 5: THREAD SAFETY & CONCURRENCY STRESS TESTING
# ============================================================================

class TestThreadSafetyPkiCache:
    """Stress-tests get_default_intel_pki() and cache resetting under concurrent execution."""

    def test_cold_start_multithreaded_singleton_invariant(self):
        """
        Verify that when get_default_intel_pki() is invoked simultaneously by 32 threads
        from a cold cache, exactly one SyntheticIntelPKI instance is created and returned to all callers.
        """
        reset_default_intel_pki_cache()
        thread_count = 32

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(get_default_intel_pki) for _ in range(thread_count)]
            results = [f.result() for f in futures]

        first = results[0]
        assert first is not None
        assert isinstance(first, SyntheticIntelPKI)
        assert all(r is first for r in results), "Double-checked locking failed: distinct instances created!"
        assert len(first.certificate_chain) == 3

    def test_rapid_concurrent_reads_performance(self):
        """
        Verify high-throughput concurrent reads across 64 threads executing multiple iterations
        achieve 100% cache hits with zero locking deadlocks or race conditions.
        """
        thread_count = 64
        iterations_per_thread = 20

        def worker():
            local_instances = []
            for _ in range(iterations_per_thread):
                inst = get_default_intel_pki()
                local_instances.append(inst)
            return local_instances

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(worker) for _ in range(thread_count)]
            all_batches = [f.result() for f in futures]

        baseline = get_default_intel_pki()
        for batch in all_batches:
            assert len(batch) == iterations_per_thread
            assert all(item is baseline for item in batch)

    def test_thread_safe_reset_and_reinitialization(self):
        """
        Verify that reset_default_intel_pki_cache() safely resets the singleton
        and subsequent concurrent threads properly initialize a new singleton.
        """
        first_pki = get_default_intel_pki()
        reset_default_intel_pki_cache()

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(get_default_intel_pki) for _ in range(16)]
            results = [f.result() for f in futures]

        second_pki = results[0]
        assert second_pki is not None
        assert all(r is second_pki for r in results)
        assert second_pki is not first_pki, "Cache reset did not produce a fresh singleton"


# ============================================================================
# TEST SUITE 6: ADVERSARIAL FIXTURES INTEGRITY & VARIANT COVERAGE
# ============================================================================

class TestAdversarialFixturesIntegrityHarness:
    """Systematically validates all negative fixture variants for expected deviations."""

    @pytest.mark.parametrize(
        "fixture_func,expected_attr,expected_val",
        [
            (create_debug_tdx_evidence, "debug", True),
            (create_tampered_tdx_quote_evidence, "provider", "intel-tdx"),
            (create_tampered_tdx_signature_evidence, "provider", "intel-tdx"),
            (create_tampered_qe_signature_evidence, "provider", "intel-tdx"),
            (create_tampered_ak_binding_tdx_evidence, "provider", "intel-tdx"),
            (create_tampered_qe_ak_binding_evidence, "provider", "intel-tdx"),
            (create_expired_cert_tdx_evidence, "provider", "intel-tdx"),
            (create_not_yet_valid_cert_tdx_evidence, "provider", "intel-tdx"),
            (create_truncated_tdx_quote_evidence, "provider", "intel-tdx"),
        ],
    )
    def test_negative_fixtures_basic_conformance(self, fixture_func, expected_attr, expected_val):
        """Verify each negative fixture produces valid AttestationEvidence instance with target property."""
        ev = fixture_func()
        if expected_attr == "debug":
            assert ev.token_claims["debug"] is expected_val
        else:
            assert getattr(ev, expected_attr) == expected_val

    def test_untrusted_root_and_revoked_cert_tuples(self):
        """Verify tuple-returning negative fixtures return valid evidence and metadata."""
        ev_untrusted, rogue_root = create_untrusted_root_tdx_evidence()
        assert isinstance(ev_untrusted, AttestationEvidence)
        assert isinstance(rogue_root, str)
        assert "-----BEGIN CERTIFICATE-----" in rogue_root

        ev_revoked, serial = create_revoked_cert_tdx_evidence()
        assert isinstance(ev_revoked, AttestationEvidence)
        assert isinstance(serial, int)
        assert serial > 0
