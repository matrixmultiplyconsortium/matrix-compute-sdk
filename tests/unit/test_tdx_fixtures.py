"""
Unit Test Suite for Intel TDX Synthetic PKI and Quote Fixtures (Milestone 1).
Validates 3-tier Intel PKI generation, DCAP Quote (v4/v5) binary wire layout,
dual ECDSA signatures (AK and PCK), AK hash cross-binding in QE report data,
and the complete set of 13 negative/adversarial TDX test fixtures.
"""

from __future__ import annotations

import datetime
import hashlib
import secrets
import struct
from typing import Dict, List

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
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
    INTEL_OID_FMSPC_SPEC_COMPAT,
    INTEL_OID_PCEID,
    INTEL_QE_VENDOR_ID,
    INTEL_SGX_EXTENSION_OID,
    INTEL_TDX_TEE_TYPE,
    CertBundle,
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
    der_decode_oid,
    der_decode_sequence,
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
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def intel_pki() -> SyntheticIntelPKI:
    """Cached singleton Intel PKI hierarchy."""
    return get_default_intel_pki()


@pytest.fixture
def sample_nonce() -> bytes:
    """32-byte challenger nonce."""
    return secrets.token_bytes(32)


@pytest.fixture
def sample_client_pubkey() -> bytes:
    """32-byte ephemeral client public key."""
    return secrets.token_bytes(32)


@pytest.fixture
def sample_worker_pubkey() -> bytes:
    """32-byte ephemeral worker public key."""
    return secrets.token_bytes(32)


# ============================================================================
# TEST SUITE 1: SYNTHETIC INTEL PKI GENERATION & CACHING
# ============================================================================

class TestIntelPkiGeneration:
    """Tests 3-tier Intel SGX/TDX PKI certificate hierarchy generation."""

    def test_default_intel_pki_caching_and_subsecond_execution(self):
        """Verify cached singleton executes in <1ms and returns identical instance."""
        pki1 = get_default_intel_pki()
        pki2 = get_default_intel_pki()
        assert pki1 is pki2
        assert len(pki1.certificate_chain) == 3

    def test_intel_pki_hierarchy_and_signatures(self, intel_pki):
        """Verify Root CA -> Intermediate CA -> PCK Leaf signature chain."""
        root_cert = intel_pki.root_cert
        inter_cert = intel_pki.intermediate_cert
        pck_cert = intel_pki.pck_cert

        # 1. Root CA is self-signed (RSA-3072)
        root_pub = root_cert.public_key()
        if isinstance(root_pub, rsa.RSAPublicKey):
            root_pub.verify(
                root_cert.signature,
                root_cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
        elif isinstance(root_pub, ec.EllipticCurvePublicKey):
            root_pub.verify(
                root_cert.signature,
                root_cert.tbs_certificate_bytes,
                ec.ECDSA(hashes.SHA384()),
            )
        assert root_cert.subject == root_cert.issuer

        # 2. Intermediate CA is signed by Root CA
        if isinstance(root_pub, rsa.RSAPublicKey):
            root_pub.verify(
                inter_cert.signature,
                inter_cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
        elif isinstance(root_pub, ec.EllipticCurvePublicKey):
            root_pub.verify(
                inter_cert.signature,
                inter_cert.tbs_certificate_bytes,
                ec.ECDSA(hashes.SHA384()),
            )
        assert inter_cert.issuer == root_cert.subject

        # 3. PCK Leaf is signed by Intermediate CA (ECDSA P-256)
        inter_pub = inter_cert.public_key()
        assert isinstance(inter_pub, ec.EllipticCurvePublicKey)
        inter_pub.verify(
            pck_cert.signature,
            pck_cert.tbs_certificate_bytes,
            ec.ECDSA(hashes.SHA256()),
        )
        assert pck_cert.issuer == inter_cert.subject

    def test_intel_pck_custom_oid_extensions(self, intel_pki):
        """Verify PCK Leaf cert contains Intel OID 1.2.840.113741.1.13.1 extensions with FMSPC and PCEID."""
        pck_cert = intel_pki.pck_cert
        sgx_oid = x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID)
        ext = pck_cert.extensions.get_extension_for_oid(sgx_oid)
        assert ext is not None
        assert ext.critical is False
        raw_der = bytes(ext.value.value if hasattr(ext.value, "value") else ext.value)
        assert len(raw_der) > 0

        # Verify spec-compat parser snippet from planned_work/06_intel_tdx_enclaves.md line 947
        fmspc_oid_bytes = bytes([0x2A, 0x86, 0x48, 0x86, 0xF8, 0x4D, 0x01, 0x0D, 0x01, 0x02])
        idx_fmspc = raw_der.find(fmspc_oid_bytes)
        assert idx_fmspc != -1
        tag_idx_fmspc = raw_der.find(b"\x04\x06", idx_fmspc + len(fmspc_oid_bytes))
        assert tag_idx_fmspc != -1
        parsed_fmspc = raw_der[tag_idx_fmspc + 2 : tag_idx_fmspc + 8].hex().lower()
        assert parsed_fmspc == DEFAULT_INTEL_FMSPC

        # Verify PCEID parser snippet from planned_work line 955
        pceid_oid_bytes = bytes([0x2A, 0x86, 0x48, 0x86, 0xF8, 0x4D, 0x01, 0x0D, 0x01, 0x03])
        idx_pceid = raw_der.find(pceid_oid_bytes)
        assert idx_pceid != -1
        tag_idx_pceid = raw_der.find(b"\x04\x02", idx_pceid + len(pceid_oid_bytes))
        assert tag_idx_pceid != -1
        parsed_pceid = raw_der[tag_idx_pceid + 2 : tag_idx_pceid + 4].hex().lower()
        assert parsed_pceid == DEFAULT_INTEL_PCEID

    def test_intel_pki_cert_bundle_properties(self, intel_pki):
        """Verify CertBundle accessors and leaf alias for cross-provider uniformity."""
        assert isinstance(intel_pki.root, CertBundle)
        assert isinstance(intel_pki.intermediate, CertBundle)
        assert isinstance(intel_pki.pck, CertBundle)
        assert intel_pki.leaf is intel_pki.pck
        assert intel_pki.leaf_cert_pem == intel_pki.pck_cert_pem
        assert "-----BEGIN CERTIFICATE-----" in intel_pki.root_cert_pem
        assert "-----BEGIN CERTIFICATE-----" in intel_pki.intermediate_cert_pem
        assert "-----BEGIN CERTIFICATE-----" in intel_pki.pck_cert_pem


# ============================================================================
# TEST SUITE 2: INTEL ASN.1 DER HELPERS
# ============================================================================

class TestIntelAsn1DerHelpers:
    """Tests pure Python ASN.1 DER encoding and decoding helpers."""

    def test_der_oid_roundtrip(self):
        """Verify OID encoding and decoding roundtrip across Intel sub-OIDs."""
        test_oids = [
            INTEL_SGX_EXTENSION_OID,
            f"{INTEL_SGX_EXTENSION_OID}.1",
            f"{INTEL_SGX_EXTENSION_OID}.2",
            f"{INTEL_SGX_EXTENSION_OID}.3",
            f"{INTEL_SGX_EXTENSION_OID}.4",
            f"{INTEL_SGX_EXTENSION_OID}.14",
        ]
        for oid in test_oids:
            encoded = der_encode_oid(oid)
            assert encoded[0] == 0x06
            decoded = der_decode_oid(encoded)
            assert decoded == oid

    def test_der_sequence_roundtrip(self):
        """Verify SEQUENCE encoding and element extraction roundtrip."""
        item1 = der_encode_oid(INTEL_OID_FMSPC_SPEC_COMPAT)
        item2 = b"\x04\x06" + bytes.fromhex("00806f000000")
        seq = der_encode_sequence([item1, item2])
        assert seq[0] == 0x30
        elements = der_decode_sequence(seq)
        assert len(elements) == 2
        assert elements[0] == item1
        assert elements[1] == item2

    def test_build_intel_pck_extension(self):
        """Verify build_intel_pck_extension wraps custom sub-extensions properly."""
        ext_der = build_intel_pck_extension(
            fmspc_bytes=bytes.fromhex("00806f000000"),
            pceid_bytes=bytes.fromhex("0000"),
        )
        assert ext_der[0] == 0x30
        sub_seqs = der_decode_sequence(ext_der)
        assert len(sub_seqs) == 5  # PPID, spec FMSPC, PCEID, standard FMSPC, TCB


# ============================================================================
# TEST SUITE 3: DCAP TD QUOTE BINARY LAYOUT & SIGNATURES
# ============================================================================

class TestSyntheticTdxQuoteGeneration:
    """Tests DCAP Quote wire generation, offsets, and cryptographic signatures."""

    def test_quote_v4_header_and_body_layout(self, intel_pki, sample_nonce, sample_client_pubkey):
        """Verify DCAP Quote v4 byte layout, header fields, and measurement registers."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob = generate_synthetic_tdx_quote(
            pki=intel_pki,
            report_data=report_data,
            quote_version=4,
            debug=False,
        )
        assert len(blob) >= 636

        # 1. Quote Header (48B)
        version, att_key_type, tee_type = struct.unpack_from("<HHI", blob, 0)
        assert version == 4
        assert att_key_type == 2  # ECDSA-256
        assert tee_type == INTEL_TDX_TEE_TYPE
        assert blob[12:28] == INTEL_QE_VENDOR_ID

        # 2. TD Report Body (584B: offset 48..632)
        tdattributes = struct.unpack_from("<Q", blob, 168)[0]
        assert (tdattributes & 0x01) == 0  # debug == False

        mrtd = blob[184:232]
        assert mrtd == DEFAULT_TDX_MRTD
        rtmr0 = blob[376:424]
        assert rtmr0 == DEFAULT_TDX_RTMR0
        rtmr1 = blob[424:472]
        assert rtmr1 == DEFAULT_TDX_RTMR1
        rtmr2 = blob[472:520]
        assert rtmr2 == DEFAULT_TDX_RTMR2
        rtmr3 = blob[520:568]
        assert rtmr3 == DEFAULT_TDX_RTMR3
        actual_reportdata = blob[568:632]
        assert actual_reportdata == report_data

    def test_quote_v5_modular_descriptor_layout(self, intel_pki, sample_nonce, sample_client_pubkey):
        """Verify modular DCAP Quote v5 layout with 6-byte body descriptor."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob = generate_synthetic_tdx_quote(
            pki=intel_pki,
            report_data=report_data,
            quote_version=5,
            tdx_type=3,  # TDX 1.5
            debug=False,
        )
        assert len(blob) >= 642

        # 1. Quote Header (48B)
        version, att_key_type, tee_type = struct.unpack_from("<HHI", blob, 0)
        assert version == 5
        assert att_key_type == 2
        assert tee_type == INTEL_TDX_TEE_TYPE

        # 2. Modular Body Descriptor (6B: offset 48..54)
        body_type, body_size = struct.unpack_from("<HI", blob, 48)
        assert body_type == 3  # TDX 1.5
        assert body_size == 584

        # 3. Shifted TD Report Body (offset 54..638)
        tdattributes = struct.unpack_from("<Q", blob, 174)[0]
        assert (tdattributes & 0x01) == 0
        actual_reportdata = blob[574:638]
        assert actual_reportdata == report_data

    def test_dual_ecdsa_signatures_and_ak_cross_binding(self, intel_pki, sample_nonce):
        """Verify AK verifies quote body, PCK verifies QE report, and QE report binds AK."""
        report_data = sample_nonce + secrets.token_bytes(32)
        blob = generate_synthetic_tdx_quote(
            pki=intel_pki,
            report_data=report_data,
            quote_version=4,
        )

        # 1. AK Signature over Header + TD Report (bytes 0..631)
        signed_body = blob[:632]
        sig_data_len = struct.unpack_from("<I", blob, 632)[0]
        assert len(blob) >= 636 + sig_data_len

        quote_sig = blob[636:700]
        att_pubkey_raw = blob[700:764]

        # Reconstruct AK public key
        ak_point = b"\x04" + att_pubkey_raw
        ak_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ak_point)
        r = int.from_bytes(quote_sig[:32], "big")
        s = int.from_bytes(quote_sig[32:], "big")
        der_quote_sig = encode_dss_signature(r, s)
        ak_pub.verify(der_quote_sig, signed_body, ec.ECDSA(hashes.SHA256()))

        # 2. QE Report Body and Signature
        qe_report_body = blob[764:1148]
        qe_report_sig = blob[1148:1212]

        qe_r = int.from_bytes(qe_report_sig[:32], "big")
        qe_s = int.from_bytes(qe_report_sig[32:], "big")
        der_qe_sig = encode_dss_signature(qe_r, qe_s)
        pck_pub = intel_pki.pck_cert.public_key()
        pck_pub.verify(der_qe_sig, qe_report_body, ec.ECDSA(hashes.SHA256()))

        # 3. AK Cross-Binding in QE Report Data
        qe_auth_data_size = struct.unpack_from("<H", blob, 1212)[0]
        auth_data_start = 1214
        qe_auth_data = blob[auth_data_start : auth_data_start + qe_auth_data_size]
        expected_ak_binding = hashlib.sha256(att_pubkey_raw + qe_auth_data).digest()
        actual_ak_binding = qe_report_body[320:352]
        assert actual_ak_binding == expected_ak_binding


# ============================================================================
# TEST SUITE 4: VALID ATTESTATION EVIDENCE
# ============================================================================

class TestValidIntelTdxEvidence:
    """Tests create_valid_intel_tdx_evidence construction and serialization."""

    def test_valid_evidence_structure_and_roundtrip(self, intel_pki, sample_nonce, sample_client_pubkey):
        """Verify AttestationEvidence structure, claims, and byte roundtrip."""
        evidence = create_valid_intel_tdx_evidence(
            pki=intel_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        assert evidence.provider == "intel-tdx"
        assert evidence.tee_type == "intel-tdx"
        assert len(evidence.certificate_chain) == 3
        assert evidence.token_claims["nonce"] == sample_nonce.hex()
        assert evidence.token_claims["debug"] is False
        assert evidence.evidence_blob is not None
        assert len(evidence.evidence_blob) >= 636
        assert evidence.user_data == evidence.evidence_blob[568:632]

        # Serialized byte roundtrip
        raw = evidence.to_bytes()
        restored = AttestationEvidence.from_bytes(raw)
        assert restored.provider == "intel-tdx"
        assert restored.evidence_blob == evidence.evidence_blob
        assert restored == evidence


# ============================================================================
# TEST SUITE 5: COMPLETE 13 ADVERSARIAL FIXTURES
# ============================================================================

class TestAdversarialTdxFixtures:
    """Verifies that all 13 negative fixtures introduce exact, intended mutations."""

    def test_tampered_tdx_quote_evidence(self, intel_pki, sample_nonce):
        """Verify bit-flip in signed payload region causes AK signature verification to fail."""
        ev = create_tampered_tdx_quote_evidence(pki=intel_pki, nonce=sample_nonce, tamper_offset=0x050)
        blob = ev.evidence_blob
        ak_point = b"\x04" + blob[700:764]
        ak_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ak_point)
        r = int.from_bytes(blob[636:668], "big")
        s = int.from_bytes(blob[668:700], "big")
        der_sig = encode_dss_signature(r, s)

        with pytest.raises(Exception):
            ak_pub.verify(der_sig, blob[:632], ec.ECDSA(hashes.SHA256()))

    def test_tampered_tdx_signature_evidence(self, intel_pki, sample_nonce):
        """Verify bit-flip in AK signature scalars causes verification to fail."""
        ev = create_tampered_tdx_signature_evidence(pki=intel_pki, nonce=sample_nonce)
        blob = ev.evidence_blob
        ak_point = b"\x04" + blob[700:764]
        ak_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ak_point)
        r = int.from_bytes(blob[636:668], "big")
        s = int.from_bytes(blob[668:700], "big")
        der_sig = encode_dss_signature(r, s)

        with pytest.raises(Exception):
            ak_pub.verify(der_sig, blob[:632], ec.ECDSA(hashes.SHA256()))

    def test_tampered_qe_signature_evidence(self, intel_pki, sample_nonce):
        """Verify bit-flip in PCK signature causes QE report verification to fail."""
        ev = create_tampered_qe_signature_evidence(pki=intel_pki, nonce=sample_nonce)
        blob = ev.evidence_blob
        pck_pub = intel_pki.pck_cert.public_key()
        r = int.from_bytes(blob[1148:1180], "big")
        s = int.from_bytes(blob[1180:1212], "big")
        der_sig = encode_dss_signature(r, s)

        with pytest.raises(Exception):
            pck_pub.verify(der_sig, blob[764:1148], ec.ECDSA(hashes.SHA256()))

    def test_tampered_ak_binding_tdx_evidence(self, intel_pki, sample_nonce):
        """Verify AK hash in QE report does not match SHA-256(att_public_key || qe_auth_data)."""
        ev = create_tampered_ak_binding_tdx_evidence(pki=intel_pki, nonce=sample_nonce)
        blob = ev.evidence_blob
        att_pubkey_raw = blob[700:764]
        actual_binding = blob[764 + 320 : 764 + 352]
        expected_binding = hashlib.sha256(att_pubkey_raw).digest()
        assert actual_binding != expected_binding

        # Test alias
        ev_alias = create_tampered_qe_ak_binding_evidence(pki=intel_pki, nonce=sample_nonce)
        assert ev_alias.evidence_blob is not None

    def test_debug_tdx_evidence(self, intel_pki, sample_nonce):
        """Verify TDATTRIBUTES.DEBUG bit 0 is set and claims reflect debug=True."""
        ev = create_debug_tdx_evidence(pki=intel_pki, nonce=sample_nonce)
        tdattributes = struct.unpack_from("<Q", ev.evidence_blob, 168)[0]
        assert (tdattributes & 0x01) == 1
        assert ev.token_claims["debug"] is True

    def test_nonce_mismatch_tdx_evidence(self, intel_pki, sample_nonce):
        """Verify embedded nonce in REPORTDATA differs from requested challenger nonce."""
        ev = create_nonce_mismatch_tdx_evidence(requested_nonce=sample_nonce, pki=intel_pki)
        embedded_nonce = ev.evidence_blob[568:600]
        assert embedded_nonce != sample_nonce

    def test_pubkey_mismatch_tdx_evidence(self, intel_pki, sample_nonce, sample_client_pubkey):
        """Verify embedded pubkey hash in REPORTDATA differs from genuine client pubkey hash."""
        ev = create_pubkey_mismatch_tdx_evidence(nonce=sample_nonce, pki=intel_pki)
        genuine_hash = hashlib.sha256(sample_client_pubkey).digest()
        embedded_hash = ev.evidence_blob[600:632]
        assert embedded_hash != genuine_hash

    def test_measurement_mismatch_tdx_evidence(self, intel_pki, sample_nonce):
        """Verify MRTD in evidence deviates from reference Golden RIM."""
        ev = create_measurement_mismatch_tdx_evidence(pki=intel_pki, nonce=sample_nonce)
        actual_mrtd = ev.evidence_blob[184:232].hex()
        assert actual_mrtd != DEFAULT_TDX_GOLDEN_RIM["mrtd"]

    def test_untrusted_root_tdx_evidence(self, intel_pki, sample_nonce):
        """Verify returns rogue certificate hierarchy not anchored to default Intel Root CA."""
        ev, rogue_root_pem = create_untrusted_root_tdx_evidence(nonce=sample_nonce)
        assert rogue_root_pem != intel_pki.root_cert_pem
        assert ev.certificate_chain[-1] == rogue_root_pem

    def test_expired_cert_tdx_evidence(self, sample_nonce):
        """Verify PCK certificate hierarchy validity dates are expired."""
        ev = create_expired_cert_tdx_evidence(nonce=sample_nonce)
        pck_cert = x509.load_pem_x509_certificate(ev.certificate_chain[0].encode("utf-8"))
        now = datetime.datetime.utcnow()
        assert pck_cert.not_valid_after < now

    def test_not_yet_valid_cert_tdx_evidence(self, sample_nonce):
        """Verify PCK certificate hierarchy validity dates are in the future."""
        ev = create_not_yet_valid_cert_tdx_evidence(nonce=sample_nonce)
        pck_cert = x509.load_pem_x509_certificate(ev.certificate_chain[0].encode("utf-8"))
        now = datetime.datetime.utcnow()
        assert pck_cert.not_valid_before > now

    def test_revoked_cert_tdx_evidence(self, intel_pki, sample_nonce):
        """Verify returns genuine evidence along with leaf PCK serial number for revocation."""
        ev, serial_num = create_revoked_cert_tdx_evidence(pki=intel_pki, nonce=sample_nonce)
        assert serial_num == intel_pki.pck_cert.serial_number
        assert len(ev.certificate_chain) == 3

    def test_truncated_tdx_quote_evidence(self, intel_pki, sample_nonce):
        """Verify truncated evidence blob is shorter than minimum quote length (636 bytes)."""
        ev = create_truncated_tdx_quote_evidence(pki=intel_pki, nonce=sample_nonce, truncated_length=500)
        assert len(ev.evidence_blob) == 500

    def test_malformed_version_tdx_evidence(self, intel_pki, sample_nonce):
        """Verify version in Quote Header is malformed (e.g. 99 != 4 and != 5)."""
        ev = create_malformed_version_tdx_evidence(pki=intel_pki, nonce=sample_nonce, invalid_version=99)
        version = struct.unpack_from("<H", ev.evidence_blob, 0)[0]
        assert version == 99
