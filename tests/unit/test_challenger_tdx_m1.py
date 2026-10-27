"""
Adversarial Stress Test Suite for Milestone 1 Intel TDX Fixtures & Contracts.
Authored by Challenger 2 (teamwork_preview_challenger).

Empirically challenges:
1. Modular Quote v5 layout: descriptor offsets, body types 0x0002 (TDX 1.0) and 0x0003 (TDX 1.5),
   dual ECDSA signatures, AK cross-binding in QE3 report data.
2. Complete negative fixture fidelity across all 14 adversarial mutations.
3. Edge-case input resilience: zero nonces, max-length nonces, hex string variants, arbitrary pubkeys.
4. Contract schema boundary testing across matrix-contracts and AttestationEvidence.
"""

from __future__ import annotations

import datetime
import hashlib
import secrets
import struct
from typing import Dict, List, Optional

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature

import matrix_contracts.proto.compute_service_pb2 as pb2
from matrix_contracts.schemas.attestation import AttestationEvidence
from matrix_contracts.schemas.discovery import HardwareSpecs, MatchedWorker, WorkerQueryRequest
from matrix_compute.crypto.pki_fixtures import (
    DEFAULT_INTEL_FMSPC,
    DEFAULT_INTEL_PCEID,
    DEFAULT_TDX_GOLDEN_RIM,
    DEFAULT_TDX_MRTD,
    DEFAULT_TDX_RTMR0,
    DEFAULT_TDX_RTMR1,
    DEFAULT_TDX_RTMR2,
    DEFAULT_TDX_RTMR3,
    INTEL_QE_VENDOR_ID,
    INTEL_TDX_TEE_TYPE,
    SyntheticIntelPKI,
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
    generate_synthetic_tdx_quote,
    get_default_intel_pki,
)


@pytest.fixture(scope="session")
def pki() -> SyntheticIntelPKI:
    return get_default_intel_pki()


# ============================================================================
# 1. MODULAR QUOTE V5 DESCRIPTOR & OFFSET STRESS TESTS
# ============================================================================

class TestModularQuoteV5Offsets:
    """Exhaustive offset and cryptographic verification of DCAP Quote Format v5."""

    @pytest.mark.parametrize("tdx_type", [0x0002, 0x0003])
    def test_v5_modular_descriptor_and_field_offsets(self, pki, tdx_type):
        """
        Verify exact byte boundaries and offsets in Quote Format v5:
        - Header: 0..48
        - Body Descriptor: 48..54 (type: uint16, size: uint32)
        - TD Report Body: 54..638 (584 bytes)
        - Sig Data Length: 638..642 (uint32)
        - Quote Signature: 642..706 (64 bytes)
        - Attestation Pubkey: 706..770 (64 bytes)
        - QE Report Body: 770..1154 (384 bytes)
        - QE Report Signature: 1154..1218 (64 bytes)
        - QE Auth Data Size: 1218..1220 (uint16)
        """
        nonce = secrets.token_bytes(32)
        pubkey = secrets.token_bytes(32)
        report_data = nonce + hashlib.sha256(pubkey).digest()

        quote = generate_synthetic_tdx_quote(
            pki=pki,
            quote_version=5,
            tdx_type=tdx_type,
            report_data=report_data,
            debug=False,
        )

        assert len(quote) >= 1220

        # 1. Header (48B)
        version, att_key_type, tee_type = struct.unpack_from("<HHI", quote, 0)
        assert version == 5
        assert att_key_type == 2  # ECDSA-256
        assert tee_type == INTEL_TDX_TEE_TYPE
        assert quote[12:28] == INTEL_QE_VENDOR_ID

        # 2. Body Descriptor (6B: 48..54)
        desc_type, desc_size = struct.unpack_from("<HI", quote, 48)
        assert desc_type == tdx_type
        assert desc_size == 584

        # 3. TD Report Body (584B: 54..638)
        # Check seamattributes at 54 + 112 = 166
        # TDATTRIBUTES at 54 + 120 = 174
        tdattributes = struct.unpack_from("<Q", quote, 174)[0]
        assert (tdattributes & 0x01) == 0  # not debug

        # Check MRTD at 54 + 136 = 190
        mrtd = quote[190 : 190 + 48]
        assert mrtd == DEFAULT_TDX_MRTD

        # Check RTMR0..3 at 54 + 328, 376, 424, 472
        assert quote[382 : 382 + 48] == DEFAULT_TDX_RTMR0
        assert quote[430 : 430 + 48] == DEFAULT_TDX_RTMR1
        assert quote[478 : 478 + 48] == DEFAULT_TDX_RTMR2
        assert quote[526 : 526 + 48] == DEFAULT_TDX_RTMR3

        # Check REPORTDATA at 54 + 520 = 574 (64 bytes: 574..638)
        embedded_report_data = quote[574:638]
        assert embedded_report_data == report_data

        # 4. Signature Data Length (4B: 638..642)
        sig_data_len = struct.unpack_from("<I", quote, 638)[0]
        assert len(quote) == 642 + sig_data_len

        # 5. Quote Signature (64B: 642..706)
        quote_sig = quote[642:706]
        assert len(quote_sig) == 64

        # 6. Attestation Public Key (64B: 706..770)
        att_pubkey_raw = quote[706:770]
        assert len(att_pubkey_raw) == 64

        # Verify AK signature over Header + Descriptor + TD Report (bytes 0..638)
        signed_body = quote[:638]
        ak_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b"\x04" + att_pubkey_raw)
        r_ak = int.from_bytes(quote_sig[:32], "big")
        s_ak = int.from_bytes(quote_sig[32:], "big")
        der_ak_sig = encode_dss_signature(r_ak, s_ak)
        ak_pub.verify(der_ak_sig, signed_body, ec.ECDSA(hashes.SHA256()))

        # 7. QE Report Body (384B: 770..1154)
        qe_body = quote[770:1154]
        assert len(qe_body) == 384

        # 8. QE Report Signature (64B: 1154..1218)
        qe_sig = quote[1154:1218]
        assert len(qe_sig) == 64
        r_qe = int.from_bytes(qe_sig[:32], "big")
        s_qe = int.from_bytes(qe_sig[32:], "big")
        der_qe_sig = encode_dss_signature(r_qe, s_qe)
        pki.pck_cert.public_key().verify(der_qe_sig, qe_body, ec.ECDSA(hashes.SHA256()))

        # 9. AK Cross-Binding in QE Report (offset 320 in QE body -> 770 + 320 = 1090)
        qe_auth_data_len = struct.unpack_from("<H", quote, 1218)[0]
        auth_data = quote[1220 : 1220 + qe_auth_data_len]
        expected_ak_hash = hashlib.sha256(att_pubkey_raw + auth_data).digest()
        assert qe_body[320:352] == expected_ak_hash

    def test_v5_evidence_generation_and_token_signature_offset(self, pki):
        """Verify create_valid_intel_tdx_evidence with quote_version=5 extracts correct AK signature."""
        nonce = secrets.token_bytes(32)
        ev = create_valid_intel_tdx_evidence(pki=pki, nonce=nonce, quote_version=5)
        assert ev.provider == "intel-tdx"
        assert ev.token_claims["quote_version"] == 5
        assert len(ev.token_signature) == 64

        # Verify that token_signature matches quote_blob[642:706] for v5
        assert ev.token_signature == ev.evidence_blob[642:706]

        # Verify REPORTDATA matches user_data at offset 574..638
        assert ev.evidence_blob[574:638] == ev.user_data


# ============================================================================
# 2. ADVERSARIAL STRESS TESTING ACROSS ALL 14 NEGATIVE FIXTURES
# ============================================================================

class TestNegativeFixtureFidelity:
    """
    Empirically verifies that each of the 14 negative fixture functions
    produces an authentically invalid attestation state that causes the expected
    security or cryptographic verification check to fail.
    """

    def test_mutation_01_tampered_tdx_quote_fails_ak_signature(self, pki):
        """1. Tampering quote body breaks Attestation Key signature verification."""
        ev = create_tampered_tdx_quote_evidence(pki=pki)
        blob = ev.evidence_blob
        ak_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b"\x04" + blob[700:764])
        r = int.from_bytes(blob[636:668], "big")
        s = int.from_bytes(blob[668:700], "big")
        der_sig = encode_dss_signature(r, s)

        with pytest.raises(Exception):
            ak_pub.verify(der_sig, blob[:632], ec.ECDSA(hashes.SHA256()))

    def test_mutation_02_tampered_tdx_sig_fails_ak_signature(self, pki):
        """2. Tampering AK signature scalars breaks quote verification."""
        ev = create_tampered_tdx_signature_evidence(pki=pki)
        blob = ev.evidence_blob
        ak_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b"\x04" + blob[700:764])
        r = int.from_bytes(ev.token_signature[:32], "big")
        s = int.from_bytes(ev.token_signature[32:], "big")
        der_sig = encode_dss_signature(r, s)

        with pytest.raises(Exception):
            ak_pub.verify(der_sig, blob[:632], ec.ECDSA(hashes.SHA256()))

    def test_mutation_03_tampered_qe_sig_fails_pck_signature(self, pki):
        """3. Tampering QE signature breaks PCK verification over QE report."""
        ev = create_tampered_qe_signature_evidence(pki=pki)
        blob = ev.evidence_blob
        r = int.from_bytes(blob[1148:1180], "big")
        s = int.from_bytes(blob[1180:1212], "big")
        der_sig = encode_dss_signature(r, s)

        with pytest.raises(Exception):
            pki.pck_cert.public_key().verify(der_sig, blob[764:1148], ec.ECDSA(hashes.SHA256()))

    def test_mutation_04_tampered_ak_binding_fails_cross_binding(self, pki):
        """4. Tampering AK hash in QE report breaks cross-binding."""
        ev = create_tampered_ak_binding_tdx_evidence(pki=pki)
        blob = ev.evidence_blob
        att_pubkey = blob[700:764]
        expected_hash = hashlib.sha256(att_pubkey).digest()
        actual_hash = blob[764 + 320 : 764 + 352]
        assert actual_hash != expected_hash

    def test_mutation_04_alias_qe_ak_binding(self, pki):
        """4b. Verify create_tampered_qe_ak_binding_evidence alias behaves identically."""
        ev = create_tampered_qe_ak_binding_evidence(pki=pki)
        blob = ev.evidence_blob
        att_pubkey = blob[700:764]
        expected_hash = hashlib.sha256(att_pubkey).digest()
        actual_hash = blob[764 + 320 : 764 + 352]
        assert actual_hash != expected_hash

    def test_mutation_05_debug_quote_fails_security_policy(self, pki):
        """5. Debug enclave quote has TDATTRIBUTES.DEBUG == 1."""
        ev = create_debug_tdx_evidence(pki=pki)
        tdattributes = struct.unpack_from("<Q", ev.evidence_blob, 168)[0]
        assert (tdattributes & 0x01) == 1
        assert ev.token_claims["debug"] is True

    def test_mutation_06_nonce_mismatch_fails_freshness(self, pki):
        """6. Replay attack: REPORTDATA embeds stale nonce."""
        genuine_nonce = secrets.token_bytes(32)
        ev = create_nonce_mismatch_tdx_evidence(requested_nonce=genuine_nonce, pki=pki)
        embedded_nonce = ev.evidence_blob[568:600]
        assert embedded_nonce != genuine_nonce
        assert ev.token_claims["nonce"] != genuine_nonce.hex()

    def test_mutation_07_pubkey_mismatch_fails_session_binding(self, pki):
        """7. Key substitution: REPORTDATA embeds wrong pubkey hash."""
        genuine_client_pubkey = secrets.token_bytes(32)
        genuine_hash = hashlib.sha256(genuine_client_pubkey).digest()
        ev = create_pubkey_mismatch_tdx_evidence(pki=pki)
        embedded_hash = ev.evidence_blob[600:632]
        assert embedded_hash != genuine_hash

    def test_mutation_08_measurement_mismatch_fails_golden_rim(self, pki):
        """8. Software deviation: MRTD in quote does not match Golden RIM."""
        ev = create_measurement_mismatch_tdx_evidence(pki=pki)
        quote_mrtd = ev.evidence_blob[184:232].hex()
        assert quote_mrtd != ev.golden_rim["mrtd"]

    def test_mutation_09_untrusted_root_fails_pki_chain(self):
        """9. Rogue PKI: certificate chain rooted in an untrusted root CA."""
        default_pki = get_default_intel_pki()
        ev, rogue_root_pem = create_untrusted_root_tdx_evidence()
        assert ev.certificate_chain[-1] == rogue_root_pem
        assert rogue_root_pem != default_pki.root_cert_pem

    def test_mutation_10_expired_cert_fails_temporal_check(self):
        """10. Expired PCK certificate has not_valid_after in the past."""
        ev = create_expired_cert_tdx_evidence()
        pck_cert = x509.load_pem_x509_certificate(ev.certificate_chain[0].encode("utf-8"))
        now = datetime.datetime.now(datetime.timezone.utc)
        cert_expiry = pck_cert.not_valid_after_utc if hasattr(pck_cert, "not_valid_after_utc") else pck_cert.not_valid_after
        if cert_expiry.tzinfo is None:
            now = datetime.datetime.utcnow()
        assert cert_expiry < now

    def test_mutation_11_future_cert_fails_temporal_check(self):
        """11. Not-yet-valid PCK certificate has not_valid_before in the future."""
        ev = create_not_yet_valid_cert_tdx_evidence()
        pck_cert = x509.load_pem_x509_certificate(ev.certificate_chain[0].encode("utf-8"))
        now = datetime.datetime.now(datetime.timezone.utc)
        cert_start = pck_cert.not_valid_before_utc if hasattr(pck_cert, "not_valid_before_utc") else pck_cert.not_valid_before
        if cert_start.tzinfo is None:
            now = datetime.datetime.utcnow()
        assert cert_start > now

    def test_mutation_12_revoked_cert_provides_serial(self, pki):
        """12. Revoked certificate fixture provides matching serial number."""
        ev, serial_num = create_revoked_cert_tdx_evidence(pki=pki)
        pck_cert = x509.load_pem_x509_certificate(ev.certificate_chain[0].encode("utf-8"))
        assert serial_num == pck_cert.serial_number

    def test_mutation_13_truncated_quote_fails_length_check(self, pki):
        """13. Truncated quote is under minimum size (636 bytes)."""
        ev = create_truncated_tdx_quote_evidence(pki=pki, truncated_length=500)
        assert len(ev.evidence_blob) == 500
        assert len(ev.evidence_blob) < 636

    def test_mutation_14_malformed_version_fails_version_check(self, pki):
        """14. Quote with version 99 is neither v4 nor v5."""
        ev = create_malformed_version_tdx_evidence(pki=pki, invalid_version=99)
        version = struct.unpack_from("<H", ev.evidence_blob, 0)[0]
        assert version == 99
        assert version not in (4, 5)


# ============================================================================
# 3. BOUNDARY EDGE CASE STRESS TESTS
# ============================================================================

class TestEdgeCaseInputs:
    """Stress tests boundary edge cases for nonces, public keys, and measurements."""

    def test_all_zero_nonce_32_bytes(self, pki):
        """Zero nonces (32 zero bytes) must bind correctly into REPORTDATA."""
        zero_nonce = b"\x00" * 32
        ev = create_valid_intel_tdx_evidence(pki=pki, nonce=zero_nonce)
        assert ev.evidence_blob[568:600] == zero_nonce
        assert ev.user_data[:32] == zero_nonce
        assert ev.token_claims["nonce"] == zero_nonce.hex()

    def test_short_nonce_is_padded_to_32_bytes(self, pki):
        """Shorter nonce (e.g. 16 bytes) is cleanly right-padded to 32 bytes."""
        short_nonce = b"\xde\xad\xbe\xef" * 4  # 16 bytes
        ev = create_valid_intel_tdx_evidence(pki=pki, nonce=short_nonce)
        expected_padded = short_nonce + b"\x00" * 16
        assert ev.evidence_blob[568:600] == expected_padded

    def test_oversized_nonce_is_truncated_to_32_bytes(self, pki):
        """Oversized nonce (64 bytes) is safely truncated to 32 bytes."""
        oversized_nonce = b"\xaa" * 64
        ev = create_valid_intel_tdx_evidence(pki=pki, nonce=oversized_nonce)
        assert ev.evidence_blob[568:600] == b"\xaa" * 32

    def test_hex_string_nonce_with_and_without_0x(self, pki):
        """Hex string nonces with '0x' or without '0x' parse equivalently."""
        raw_nonce = secrets.token_bytes(32)
        hex_with_0x = "0x" + raw_nonce.hex()
        hex_without_0x = raw_nonce.hex()

        ev1 = create_valid_intel_tdx_evidence(pki=pki, nonce=hex_with_0x)
        ev2 = create_valid_intel_tdx_evidence(pki=pki, nonce=hex_without_0x)

        assert ev1.evidence_blob[568:600] == raw_nonce
        assert ev2.evidence_blob[568:600] == raw_nonce

    def test_dual_ephemeral_pubkey_hash_binding(self, pki):
        """When both client and worker pubkeys are provided, SHA256(client || worker) is bound."""
        client_pk = secrets.token_bytes(32)
        worker_pk = secrets.token_bytes(32)
        expected_hash = hashlib.sha256(client_pk + worker_pk).digest()

        ev = create_valid_intel_tdx_evidence(
            pki=pki,
            client_ephemeral_pubkey=client_pk,
            worker_ephemeral_pubkey=worker_pk,
        )
        assert ev.evidence_blob[600:632] == expected_hash
        assert ev.token_claims["ephemeral_pubkey_hash"] == expected_hash.hex()
        assert ev.token_claims["client_pubkey_hash"] == hashlib.sha256(client_pk).hexdigest()
        assert ev.token_claims["worker_pubkey_hash"] == hashlib.sha256(worker_pk).hexdigest()

    def test_arbitrary_uncompressed_ec_pubkey_65_bytes(self, pki):
        """Handles 65-byte uncompressed elliptic curve public keys (0x04 || X || Y)."""
        uncompressed_pk = b"\x04" + secrets.token_bytes(64)
        ev = create_valid_intel_tdx_evidence(
            pki=pki,
            client_ephemeral_pubkey=uncompressed_pk,
        )
        expected_hash = hashlib.sha256(uncompressed_pk).digest()
        assert ev.evidence_blob[600:632] == expected_hash

    def test_extreme_measurements_all_zero_and_all_ff(self, pki):
        """Supports boundary measurement digests (all 0x00 and all 0xFF)."""
        mrtd_zero = b"\x00" * 48
        mrtd_ff = b"\xff" * 48

        ev_zero = create_valid_intel_tdx_evidence(pki=pki, mrtd=mrtd_zero)
        assert ev_zero.evidence_blob[184:232] == mrtd_zero
        assert ev_zero.golden_rim["mrtd"] == mrtd_zero.hex()

        ev_ff = create_valid_intel_tdx_evidence(pki=pki, mrtd=mrtd_ff)
        assert ev_ff.evidence_blob[184:232] == mrtd_ff
        assert ev_ff.golden_rim["mrtd"] == mrtd_ff.hex()


# ============================================================================
# 4. CONTRACT SCHEMAS ADVERSARIAL STRESS TESTS
# ============================================================================

class TestContractsIntegration:
    """Stress tests protocol buffers and discovery schemas with intel-tdx."""

    def test_protobuf_tile_task_handshake_with_intel_tdx(self):
        """Verify TileTaskRequest and TileTaskAck serialize and deserialize with intel-tdx."""
        req = pb2.TileTaskRequest(
            task_id="tdx_task_001",
            require_confidential_compute=True,
            required_tee_type="intel-tdx",
            attestation_nonce=b"\x11" * 32,
            client_ephemeral_pubkey=b"\x22" * 32,
        )
        wire_req = req.SerializeToString()
        deserialized_req = pb2.TileTaskRequest()
        deserialized_req.ParseFromString(wire_req)
        assert deserialized_req.required_tee_type == "intel-tdx"
        assert deserialized_req.require_confidential_compute is True

        ack = pb2.TileTaskAck(
            task_id="tdx_task_001",
            is_ready=True,
            tee_type="intel-tdx",
            attestation_evidence=b"\x33" * 128,
            worker_ephemeral_pubkey=b"\x44" * 32,
        )
        wire_ack = ack.SerializeToString()
        deserialized_ack = pb2.TileTaskAck()
        deserialized_ack.ParseFromString(wire_ack)
        assert deserialized_ack.tee_type == "intel-tdx"
        assert deserialized_ack.is_ready is True

    def test_hardware_specs_auto_infer_and_serialization(self):
        """Verify HardwareSpecs auto-infers tee=True when tee_type='intel-tdx'."""
        specs = HardwareSpecs(
            gpu_model="Intel Xeon Platinum 8480+",
            vram_mb=65536,
            peak_fp32_tflops=32.0,
            attestation_proof="tdx_quote_proof",
            tee_type="intel-tdx",
        )
        assert specs.tee is True
        assert specs.tee_type == "intel-tdx"

        # JSON roundtrip
        json_str = specs.model_dump_json()
        restored = HardwareSpecs.model_validate_json(json_str)
        assert restored.tee is True
        assert restored.tee_type == "intel-tdx"
        assert restored == specs

    def test_worker_query_request_filtering_for_intel_tdx(self):
        """Verify WorkerQueryRequest properly filters for intel-tdx workers."""
        query = WorkerQueryRequest(
            min_vram_mb=32768,
            max_price_sats_per_gflop=0.002,
            tee_type="intel-tdx",
        )
        assert query.require_tee is True
        assert query.required_tee_type == "intel-tdx"
        assert query.tee is True
        assert query.tee_type == "intel-tdx"

        # JSON roundtrip
        json_str = query.model_dump_json()
        restored = WorkerQueryRequest.model_validate_json(json_str)
        assert restored.require_tee is True
        assert restored.required_tee_type == "intel-tdx"

    def test_matched_worker_schema_carrying_intel_tdx(self):
        """Verify MatchedWorker correctly reflects intel-tdx metadata."""
        worker = MatchedWorker(
            worker_pubkey="02tdxworkerpubkey",
            p2p_multiaddr="/ip4/192.168.1.100/tcp/50051",
            lightning_routing_pubkey="03lightningpubkey",
            sats_per_gflop=0.001,
            tee_type="intel-tdx",
        )
        assert worker.tee is True
        assert worker.tee_type == "intel-tdx"

    def test_attestation_evidence_full_roundtrip_fidelity(self, pki):
        """Verify AttestationEvidence preserving all TDX evidence attributes through binary roundtrip."""
        ev = create_valid_intel_tdx_evidence(pki=pki)
        raw_bytes = ev.to_bytes()
        restored = AttestationEvidence.from_bytes(raw_bytes)

        assert restored.provider == "intel-tdx"
        assert restored.tee_type == "intel-tdx"
        assert restored.evidence_blob == ev.evidence_blob
        assert restored.user_data == ev.user_data
        assert restored.token_signature == ev.token_signature
        assert restored.certificate_chain == ev.certificate_chain
        assert restored.token_claims == ev.token_claims
        assert restored.golden_rim == ev.golden_rim
        assert restored == ev
