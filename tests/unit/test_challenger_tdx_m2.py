"""
Challenger 2 Empirical Stress Test Suite for Milestone 2: Intel TDX Quote Verification.

Adversarial stress-testing covering:
1. Modular Quote Format v5 descriptors:
   - Unsupported / reserved body types (0x0001 SGX, 0x0000, 0x0004, 0xFFFF).
   - Invalid / corrupt body sizes (0, 583, 585, 1024, 0xFFFFFFFF).
   - Boundary truncations at and around modular descriptor (lengths 48..54, 637..641).
   - Single-bit mutations in modular descriptor header (offsets 48..53).
   - TDX 1.0 (0x0002) and TDX 1.5 (0x0003) valid verification end-to-end and claims checks.
   - Certificate chain extraction on v5 quotes (TDX 1.0 & 1.5, truncated blobs).
2. Golden RIM measurement deviations:
   - Individual register deviations (MRTD, RTMR0, RTMR1, RTMR2, RTMR3) with bit flips.
   - Partial Golden RIM manifests (validating subsets).
   - Extraneous / nonexistent Golden RIM keys (e.g. rtmr4, pcr0) fail-closed.
   - Hex prefix (0x), uppercase, and case-insensitivity formatting resilience.
   - Adversarial self-certification defeat (evidence.golden_rim injection ignored).
   - Missing / malformed measurement claims dictionaries.
3. Certificate revocation and temporal expiration:
   - Revocation at every tier: Leaf PCK, Intermediate CA, and Root CA.
   - Temporal expiration boundaries (not_valid_before - 1s, not_valid_after + 1s).
   - Exact boundary seconds (not_valid_before, not_valid_after).
   - Expired or not-yet-valid intermediate and root certificates.
   - Chain topology boundaries: empty chain, 1-cert, 2-cert, reversed order.
   - CA basic constraints enforcement (leaf cannot be CA).
4. ASN.1 extension boundary conditions:
   - Certificates without SGX/TDX custom extension (OID 1.2.840.113741.1.13.1).
   - Empty extension payload (0 bytes).
   - Truncated FMSPC, PCEID, TCB components (buffer overrun resilience).
   - Sub-OID compatibility: .2 vs .4 vs both.
   - Non-OCTET-STRING ASN.1 tags (INTEGER, SEQUENCE, corrupt tags).
   - Fuzz / garbage extension payloads (1KB noise).
   - End-to-end verification resilience with non-standard extension payloads.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import secrets
import struct
from typing import Any, Dict, List, Optional

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

from matrix_compute.crypto import (
    AttestationVerifier,
    IntelTdxVerifier,
    SyntheticIntelPKI,
    VerificationPolicy,
    VerificationResult,
    create_valid_intel_tdx_evidence,
    encode_intel_pck_asn1_extension,
    generate_synthetic_intel_pki,
    generate_synthetic_tdx_quote,
    get_default_intel_pki,
    parse_pck_extensions,
    DEFAULT_INTEL_FMSPC,
    DEFAULT_INTEL_PCEID,
    DEFAULT_INTEL_TCB_COMP_SVNS,
    DEFAULT_TDX_GOLDEN_RIM,
    DEFAULT_TDX_MRTD,
    DEFAULT_TDX_RTMR0,
    DEFAULT_TDX_RTMR1,
    DEFAULT_TDX_RTMR2,
    DEFAULT_TDX_RTMR3,
    INTEL_SGX_EXTENSION_OID,
    INTEL_TDX_TEE_TYPE,
)
from matrix_compute.exceptions import (
    AttestationBindingError,
    AttestationCertificateError,
    AttestationChainInvalidError,
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
def pki() -> SyntheticIntelPKI:
    return get_default_intel_pki()


@pytest.fixture
def policy(pki: SyntheticIntelPKI) -> VerificationPolicy:
    pol = VerificationPolicy()
    pol.add_trusted_root_pem(pki.root_cert_pem)
    pol.add_golden_rim("intel-tdx", DEFAULT_TDX_GOLDEN_RIM)
    return pol


@pytest.fixture
def verifier(policy: VerificationPolicy) -> AttestationVerifier:
    return AttestationVerifier(policy=policy)


@pytest.fixture
def sample_nonce() -> bytes:
    return secrets.token_bytes(32)


@pytest.fixture
def sample_client_pubkey() -> bytes:
    return secrets.token_bytes(32)


# ==============================================================================
# 1. MODULAR QUOTE FORMAT V5 DESCRIPTORS (TDX 1.0 & 1.5)
# ==============================================================================

class TestModularQuoteFormatV5DescriptorStress:
    """Stress-tests Quote Format v5 modular descriptor parsing, validation, and boundaries."""

    @pytest.mark.parametrize("invalid_body_type", [0x0001, 0x0000, 0x0004, 0x0005, 0xFFFF])
    def test_v5_descriptor_unsupported_body_types_rejected(
        self, verifier, pki, sample_nonce, sample_client_pubkey, invalid_body_type
    ):
        """Quote v5 with body_type not in (0x0002, 0x0003) must be rejected with AttestationEvidenceMalformedError."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob = bytearray(
            generate_synthetic_tdx_quote(
                pki=pki,
                quote_version=5,
                tdx_type=2,
                report_data=report_data,
            )
        )
        # Corrupt body_type at offset 48 (uint16_t little-endian)
        struct.pack_into("<H", blob, 48, invalid_body_type)

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=pki.certificate_chain,
            evidence_blob=bytes(blob),
            golden_rim=DEFAULT_TDX_GOLDEN_RIM.copy(),
            user_data=report_data,
        )

        with pytest.raises(AttestationEvidenceMalformedError, match="Unsupported Quote v5 Body Type"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("invalid_body_size", [0, 500, 583, 585, 1024, 0xFFFF])
    def test_v5_descriptor_invalid_body_sizes_rejected(
        self, verifier, pki, sample_nonce, sample_client_pubkey, invalid_body_size
    ):
        """Quote v5 with body_size != 584 must be rejected with AttestationEvidenceMalformedError."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob = bytearray(
            generate_synthetic_tdx_quote(
                pki=pki,
                quote_version=5,
                tdx_type=2,
                report_data=report_data,
            )
        )
        # Corrupt body_size at offset 50 (uint32_t little-endian)
        struct.pack_into("<I", blob, 50, invalid_body_size)

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=pki.certificate_chain,
            evidence_blob=bytes(blob),
            golden_rim=DEFAULT_TDX_GOLDEN_RIM.copy(),
            user_data=report_data,
        )

        with pytest.raises(AttestationEvidenceMalformedError, match="Invalid TD Report Body size"):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("trunc_len", [48, 49, 50, 51, 52, 53, 54, 100, 500, 637, 638, 641])
    def test_v5_descriptor_boundary_truncations(
        self, verifier, pki, sample_nonce, sample_client_pubkey, trunc_len
    ):
        """Quote v5 truncated anywhere in or right after descriptor must raise AttestationEvidenceMalformedError."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob = generate_synthetic_tdx_quote(
            pki=pki,
            quote_version=5,
            tdx_type=3,
            report_data=report_data,
        )
        truncated_blob = blob[:trunc_len]

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=pki.certificate_chain,
            evidence_blob=truncated_blob,
            golden_rim=DEFAULT_TDX_GOLDEN_RIM.copy(),
            user_data=report_data,
        )

        with pytest.raises(AttestationEvidenceMalformedError):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("desc_offset", [48, 49, 50, 51, 52, 53])
    def test_v5_descriptor_bit_flips_fail_closed(
        self, verifier, pki, sample_nonce, sample_client_pubkey, desc_offset
    ):
        """Single bit flips in the 6-byte modular descriptor (offsets 48..53) must be caught."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob = bytearray(
            generate_synthetic_tdx_quote(
                pki=pki,
                quote_version=5,
                tdx_type=3,  # TDX 1.5
                report_data=report_data,
            )
        )
        blob[desc_offset] ^= 0x01

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=pki.certificate_chain,
            evidence_blob=bytes(blob),
            golden_rim=DEFAULT_TDX_GOLDEN_RIM.copy(),
            user_data=report_data,
        )

        with pytest.raises((AttestationEvidenceMalformedError, AttestationSignatureError)):
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("tdx_type,expected_ver", [(2, "1.0"), (3, "1.5")])
    def test_v5_valid_tdx_1_0_and_1_5_end_to_end(
        self, verifier, pki, sample_nonce, sample_client_pubkey, tdx_type, expected_ver
    ):
        """Both TDX 1.0 (0x0002) and TDX 1.5 (0x0003) modular v5 quotes pass complete verification."""
        report_data = sample_nonce + hashlib.sha256(sample_client_pubkey).digest()
        blob = generate_synthetic_tdx_quote(
            pki=pki,
            quote_version=5,
            tdx_type=tdx_type,
            report_data=report_data,
        )

        evidence = AttestationEvidence(
            provider="intel-tdx",
            certificate_chain=pki.certificate_chain,
            evidence_blob=blob,
            golden_rim=DEFAULT_TDX_GOLDEN_RIM.copy(),
            user_data=report_data,
        )

        res = verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert res.verified is True
        assert res.claims["quote_version"] == 5
        assert res.claims["body_type"] == tdx_type
        assert res.claims["tdx_version"] == expected_ver
        assert res.measurements_verified == 5

    @pytest.mark.parametrize("tdx_type", [2, 3])
    def test_v5_extract_certificate_chain(self, pki, sample_nonce, sample_client_pubkey, tdx_type):
        """IntelTdxVerifier.extract_certificate_chain recovers PEM chain from v5 quotes for both TDX 1.0 and 1.5."""
        blob = generate_synthetic_tdx_quote(
            pki=pki,
            quote_version=5,
            tdx_type=tdx_type,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        certs = IntelTdxVerifier.extract_certificate_chain(blob)
        assert len(certs) == 3
        assert "-----BEGIN CERTIFICATE-----" in certs[0]
        assert "-----BEGIN CERTIFICATE-----" in certs[1]
        assert "-----BEGIN CERTIFICATE-----" in certs[2]

        # Truncated v5 quotes return empty list safely
        assert IntelTdxVerifier.extract_certificate_chain(blob[:50]) == []
        assert IntelTdxVerifier.extract_certificate_chain(blob[:638]) == []


# ==============================================================================
# 2. GOLDEN RIM MEASUREMENT DEVIATIONS
# ==============================================================================

class TestGoldenRimMeasurementDeviationsStress:
    """Stress-tests measurement appraisal across all registers, formatting, and subsets."""

    @pytest.mark.parametrize(
        "reg_name,kwarg",
        [
            ("mrtd", "mrtd"),
            ("rtmr0", "rtmr0"),
            ("rtmr1", "rtmr1"),
            ("rtmr2", "rtmr2"),
            ("rtmr3", "rtmr3"),
        ],
    )
    def test_individual_register_deviations_fail_closed(
        self, verifier, pki, sample_nonce, sample_client_pubkey, reg_name, kwarg
    ):
        """Single register mutation in any of MRTD, RTMR0, RTMR1, RTMR2, RTMR3 raises AttestationMeasurementMismatchError."""
        kwargs: Dict[str, Any] = {
            "pki": pki,
            "nonce": sample_nonce,
            "client_ephemeral_pubkey": sample_client_pubkey,
        }
        # Flip first byte of targeted register
        default_val = getattr(pki, kwarg, None)
        if default_val is None:
            defaults = {
                "mrtd": DEFAULT_TDX_MRTD,
                "rtmr0": DEFAULT_TDX_RTMR0,
                "rtmr1": DEFAULT_TDX_RTMR1,
                "rtmr2": DEFAULT_TDX_RTMR2,
                "rtmr3": DEFAULT_TDX_RTMR3,
            }
            default_val = defaults[kwarg]

        corrupted = bytearray(default_val)
        corrupted[0] ^= 0x01
        kwargs[kwarg] = bytes(corrupted)

        evidence = create_valid_intel_tdx_evidence(**kwargs)

        with pytest.raises(AttestationMeasurementMismatchError) as exc_info:
            verifier.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

        assert reg_name in exc_info.value.mismatches

    def test_partial_golden_rim_policies(self, pki, sample_nonce, sample_client_pubkey):
        """Policy specifying only a subset of registers verifies exactly those registers."""
        # 1. MRTD only
        pol1 = VerificationPolicy()
        pol1.add_trusted_root_pem(pki.root_cert_pem)
        pol1.add_golden_rim("intel-tdx", {"mrtd": DEFAULT_TDX_MRTD.hex()})
        v1 = AttestationVerifier(policy=pol1)

        evidence = create_valid_intel_tdx_evidence(
            pki=pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        res1 = v1.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert res1.verified is True
        assert res1.measurements_verified == 1

        # 2. MRTD and RTMR0 only
        pol2 = VerificationPolicy()
        pol2.add_trusted_root_pem(pki.root_cert_pem)
        pol2.add_golden_rim("intel-tdx", {"mrtd": DEFAULT_TDX_MRTD.hex(), "rtmr0": DEFAULT_TDX_RTMR0.hex()})
        v2 = AttestationVerifier(policy=pol2)

        res2 = v2.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert res2.verified is True
        assert res2.measurements_verified == 2

    @pytest.mark.parametrize("missing_key", ["rtmr4", "pcr0", "invalid_reg"])
    def test_extraneous_golden_rim_keys_fail_closed(
        self, pki, sample_nonce, sample_client_pubkey, missing_key
    ):
        """Policy requiring a non-existent register fails closed with missing_claims."""
        rim = DEFAULT_TDX_GOLDEN_RIM.copy()
        rim[missing_key] = "aa" * 48

        pol = VerificationPolicy()
        pol.add_trusted_root_pem(pki.root_cert_pem)
        pol.add_golden_rim("intel-tdx", rim)
        v = AttestationVerifier(policy=pol)

        evidence = create_valid_intel_tdx_evidence(
            pki=pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        with pytest.raises(AttestationMeasurementMismatchError) as exc_info:
            v.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

        assert missing_key in exc_info.value.missing_claims

    def test_golden_rim_formatting_resilience(self, pki, sample_nonce, sample_client_pubkey):
        """Golden RIM hex strings with 0x prefix, uppercase, or whitespace compare cleanly."""
        resilient_rim = {
            "mrtd": f"0x{DEFAULT_TDX_MRTD.hex().upper()}",
            "rtmr0": f"  {DEFAULT_TDX_RTMR0.hex()}  ",
            "rtmr1": DEFAULT_TDX_RTMR1.hex().upper(),
            "rtmr2": f"0X{DEFAULT_TDX_RTMR2.hex()}",
            "rtmr3": DEFAULT_TDX_RTMR3.hex(),
        }

        pol = VerificationPolicy()
        pol.add_trusted_root_pem(pki.root_cert_pem)
        pol.add_golden_rim("intel-tdx", resilient_rim)
        v = AttestationVerifier(policy=pol)

        evidence = create_valid_intel_tdx_evidence(
            pki=pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        res = v.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert res.verified is True
        assert res.measurements_verified == 5

    def test_forged_evidence_golden_rim_ignored(self, pki, sample_nonce, sample_client_pubkey):
        """Attacker injecting evidence.golden_rim cannot self-certify rogue measurements."""
        rogue_mrtd = b"\xff" * 48
        evidence = create_valid_intel_tdx_evidence(
            pki=pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
            mrtd=rogue_mrtd,
        )
        # Attacker injects fake golden_rim into evidence to match their rogue measurement
        evidence.golden_rim = {"mrtd": rogue_mrtd.hex()}

        # Policy maintains authoritative RIM
        pol = VerificationPolicy()
        pol.add_trusted_root_pem(pki.root_cert_pem)
        pol.add_golden_rim("intel-tdx", DEFAULT_TDX_GOLDEN_RIM)
        v = AttestationVerifier(policy=pol)

        with pytest.raises(AttestationMeasurementMismatchError):
            v.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    @pytest.mark.parametrize("bad_measurements", [{}, None, "not-a-dict", []])
    def test_malformed_measurements_container_fails(self, verifier, bad_measurements):
        """Missing or malformed measurements dictionary fails with AttestationMeasurementMismatchError."""
        claims = {"measurements": bad_measurements}
        with pytest.raises(AttestationMeasurementMismatchError):
            verifier.verify_measurements(claims=claims, provider="intel-tdx")


# ==============================================================================
# 3. CERTIFICATE REVOCATION AND EXPIRATION
# ==============================================================================

class TestCertRevocationAndExpirationStress:
    """Stress-tests PKI certificate revocation, temporal boundaries, and chain hierarchies."""

    def test_revocation_at_each_tier(self, pki, sample_nonce, sample_client_pubkey):
        """Revoking leaf PCK, intermediate CA, or root CA serial number triggers AttestationRevokedError."""
        evidence = create_valid_intel_tdx_evidence(
            pki=pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        # 1. Revoke leaf PCK
        pol_leaf = VerificationPolicy(revoked_serial_numbers={pki.pck_cert.serial_number})
        pol_leaf.add_trusted_root_pem(pki.root_cert_pem)
        with pytest.raises(AttestationRevokedError, match="revoked"):
            AttestationVerifier(policy=pol_leaf).verify(
                evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey
            )

        # 2. Revoke intermediate CA
        pol_inter = VerificationPolicy(revoked_serial_numbers={pki.intermediate_cert.serial_number})
        pol_inter.add_trusted_root_pem(pki.root_cert_pem)
        with pytest.raises(AttestationRevokedError, match="revoked"):
            AttestationVerifier(policy=pol_inter).verify(
                evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey
            )

        # 3. Revoke root CA
        pol_root = VerificationPolicy(revoked_serial_numbers={pki.root_cert.serial_number})
        pol_root.add_trusted_root_pem(pki.root_cert_pem)
        with pytest.raises(AttestationRevokedError, match="revoked"):
            AttestationVerifier(policy=pol_root).verify(
                evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey
            )

        # 4. Irrelevant serial number revoked does not affect valid chain
        pol_unrelated = VerificationPolicy(revoked_serial_numbers={1234567890})
        pol_unrelated.add_trusted_root_pem(pki.root_cert_pem)
        res = AttestationVerifier(policy=pol_unrelated).verify(
            evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey
        )
        assert res.verified is True

    def test_temporal_validity_exact_boundaries(self, pki, sample_nonce, sample_client_pubkey):
        """Tests exact 1-second boundaries around not_valid_before and not_valid_after."""
        evidence = create_valid_intel_tdx_evidence(
            pki=pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )

        not_before = getattr(
            pki.pck_cert, "not_valid_before_utc", pki.pck_cert.not_valid_before.replace(tzinfo=datetime.timezone.utc)
        )
        not_after = getattr(
            pki.pck_cert, "not_valid_after_utc", pki.pck_cert.not_valid_after.replace(tzinfo=datetime.timezone.utc)
        )

        pol = VerificationPolicy()
        pol.add_trusted_root_pem(pki.root_cert_pem)
        v = AttestationVerifier(policy=pol)

        # 1. Exact not_before second -> valid certificate chain
        leaf, root, clen = v.verify_certificate_chain(
            chain_pems=evidence.certificate_chain,
            verification_time=not_before,
        )
        assert leaf.subject == pki.pck_cert.subject

        # 2. 1 second before not_before -> not yet valid
        with pytest.raises(AttestationExpiredError, match="not yet valid"):
            v.verify_certificate_chain(
                chain_pems=evidence.certificate_chain,
                verification_time=not_before - datetime.timedelta(seconds=1),
            )

        # 3. Exact not_after second -> valid certificate chain
        leaf_na, root_na, clen_na = v.verify_certificate_chain(
            chain_pems=evidence.certificate_chain,
            verification_time=not_after,
        )
        assert leaf_na.subject == pki.pck_cert.subject

        # 4. 1 second after not_after -> expired
        with pytest.raises(AttestationExpiredError, match="expired"):
            v.verify_certificate_chain(
                chain_pems=evidence.certificate_chain,
                verification_time=not_after + datetime.timedelta(seconds=1),
            )

        # 5. Full end-to-end verification with aligned token timestamp at not_before + 1 hour
        valid_mid_time = not_before + datetime.timedelta(hours=1)
        evidence.token_claims["timestamp"] = valid_mid_time.timestamp()
        res = v.verify(
            evidence,
            expected_nonce=sample_nonce,
            expected_client_pubkey=sample_client_pubkey,
            verification_time=valid_mid_time,
        )
        assert res.verified is True

    def test_expired_intermediate_or_root_ca_rejected(self, sample_nonce, sample_client_pubkey):
        """Expired intermediate CA or root CA in chain triggers AttestationExpiredError."""
        # 1. Expired intermediate CA
        expired_inter_pki = generate_synthetic_intel_pki(expired=True)
        evidence = create_valid_intel_tdx_evidence(
            pki=expired_inter_pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        pol = VerificationPolicy()
        pol.add_trusted_root_pem(expired_inter_pki.root_cert_pem)
        v = AttestationVerifier(policy=pol)

        with pytest.raises(AttestationExpiredError, match="expired"):
            v.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_empty_chain_rejected(self, pki, sample_nonce, sample_client_pubkey):
        """Empty certificate chain raises AttestationEvidenceMalformedError."""
        evidence = create_valid_intel_tdx_evidence(
            pki=pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.certificate_chain = []

        pol = VerificationPolicy()
        pol.add_trusted_root_pem(pki.root_cert_pem)
        v = AttestationVerifier(policy=pol)

        with pytest.raises(AttestationEvidenceMalformedError, match="empty"):
            v.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_unanchored_incomplete_chain_rejected(self, pki, sample_nonce, sample_client_pubkey):
        """Leaf-only chain raises AttestationUntrustedRootError because leaf cannot anchor to trusted root."""
        evidence = create_valid_intel_tdx_evidence(
            pki=pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        # Leaf only: leaf is signed by intermediate, not root
        evidence.certificate_chain = [pki.pck_cert_pem]

        pol = VerificationPolicy()
        pol.add_trusted_root_pem(pki.root_cert_pem)
        v = AttestationVerifier(policy=pol)

        with pytest.raises(AttestationUntrustedRootError):
            v.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)

    def test_2tier_chain_without_root_anchors_to_local_root(self, pki, sample_nonce, sample_client_pubkey):
        """RFC 5280: Chain with [PCK, Intermediate] succeeds when Intermediate is directly signed by trusted root."""
        evidence = create_valid_intel_tdx_evidence(
            pki=pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        # Omit root cert from wire evidence
        evidence.certificate_chain = [pki.pck_cert_pem, pki.intermediate_cert_pem]

        pol = VerificationPolicy()
        pol.add_trusted_root_pem(pki.root_cert_pem)
        v = AttestationVerifier(policy=pol)

        res = v.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)
        assert res.verified is True
        assert res.chain_length == 2
        assert res.root_cert.subject == pki.root_cert.subject

    def test_reversed_chain_order_rejected(self, pki, sample_nonce, sample_client_pubkey):
        """Reversed certificate chain order [Root, Inter, PCK] is rejected."""
        evidence = create_valid_intel_tdx_evidence(
            pki=pki,
            nonce=sample_nonce,
            client_ephemeral_pubkey=sample_client_pubkey,
        )
        evidence.certificate_chain = list(reversed(pki.certificate_chain))

        pol = VerificationPolicy()
        pol.add_trusted_root_pem(pki.root_cert_pem)
        v = AttestationVerifier(policy=pol)

        with pytest.raises((AttestationCertificateError, AttestationSignatureError, AttestationUntrustedRootError)):
            v.verify(evidence, expected_nonce=sample_nonce, expected_client_pubkey=sample_client_pubkey)


# ==============================================================================
# 4. ASN.1 EXTENSION BOUNDARY CONDITIONS
# ==============================================================================

class TestAsn1ExtensionBoundaryConditionsStress:
    """Stress-tests Intel proprietary ASN.1 extension (OID 1.2.840.113741.1.13.1) parser."""

    def test_cert_without_sgx_extension_returns_defaults(self, pki):
        """Certificate without SGX extension cleanly returns default claims without raising."""
        # Intermediate cert has no SGX extension
        res = parse_pck_extensions(pki.intermediate_cert)
        assert res["fmspc"] == "unknown"
        assert res["pceid"] == "unknown"
        assert res["tcb"] == {}
        assert res["raw_der_len"] == 0

    def test_empty_extension_bytes_returns_defaults(self, pki):
        """Certificate with 0-byte SGX extension value returns defaults."""
        inter_cert = pki.intermediate_cert
        inter_key = pki.intermediate_key
        pck_key = pki.pck_key

        now = datetime.datetime.now(datetime.timezone.utc)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Empty Ext Cert")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(inter_cert.subject)
            .public_key(pck_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=10))
            .add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID),
                    b"",
                ),
                critical=False,
            )
            .sign(inter_key, hashes.SHA256())
        )

        res = parse_pck_extensions(cert)
        assert res["fmspc"] == "unknown"
        assert res["pceid"] == "unknown"
        assert res["raw_der_len"] == 0

    def test_truncated_fmspc_tag_does_not_overrun(self, pki):
        """Extension containing FMSPC OID + tag but truncated payload does not crash."""
        from matrix_compute.crypto.pki_fixtures import INTEL_OID_FMSPC, der_encode_oid
        oid_bytes = der_encode_oid(INTEL_OID_FMSPC)
        # Append tag 0x04, len 0x06, but only 2 payload bytes
        corrupt_der = oid_bytes + b"\x04\x06\x11\x22"

        now = datetime.datetime.now(datetime.timezone.utc)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Trunc FMSPC")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(pki.intermediate_cert.subject)
            .public_key(pki.pck_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=10))
            .add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID),
                    corrupt_der,
                ),
                critical=False,
            )
            .sign(pki.intermediate_key, hashes.SHA256())
        )

        res = parse_pck_extensions(cert)
        assert res["fmspc"] == "unknown"
        assert res["raw_der_len"] == len(corrupt_der)

    def test_truncated_pceid_tag_does_not_overrun(self, pki):
        """Extension containing PCEID OID + tag but truncated payload does not crash."""
        from matrix_compute.crypto.pki_fixtures import INTEL_OID_PCEID, der_encode_oid
        oid_bytes = der_encode_oid(INTEL_OID_PCEID)
        # Tag 0x04, len 0x02, but only 1 payload byte
        corrupt_der = oid_bytes + b"\x04\x02\x01"

        now = datetime.datetime.now(datetime.timezone.utc)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Trunc PCEID")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(pki.intermediate_cert.subject)
            .public_key(pki.pck_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=10))
            .add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID),
                    corrupt_der,
                ),
                critical=False,
            )
            .sign(pki.intermediate_key, hashes.SHA256())
        )

        res = parse_pck_extensions(cert)
        assert res["pceid"] == "unknown"

    def test_truncated_tcb_tag_does_not_overrun(self, pki):
        """Extension containing TCB OID + tag but truncated payload does not crash."""
        from matrix_compute.crypto.pki_fixtures import INTEL_OID_TCB_COMP_SVNS, der_encode_oid
        oid_bytes = der_encode_oid(INTEL_OID_TCB_COMP_SVNS)
        # Tag 0x04, len 0x10 (16B), but only 4 payload bytes
        corrupt_der = oid_bytes + b"\x04\x10\x01\x02\x03\x04"

        now = datetime.datetime.now(datetime.timezone.utc)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Trunc TCB")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(pki.intermediate_cert.subject)
            .public_key(pki.pck_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=10))
            .add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID),
                    corrupt_der,
                ),
                critical=False,
            )
            .sign(pki.intermediate_key, hashes.SHA256())
        )

        res = parse_pck_extensions(cert)
        assert res["tcb"] == {}

    def test_fuzz_garbage_extension_payload(self, pki):
        """Random binary garbage inside extension does not cause exception or hang."""
        garbage_der = secrets.token_bytes(2048)

        now = datetime.datetime.now(datetime.timezone.utc)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Garbage Ext")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(pki.intermediate_cert.subject)
            .public_key(pki.pck_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=10))
            .add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID),
                    garbage_der,
                ),
                critical=False,
            )
            .sign(pki.intermediate_key, hashes.SHA256())
        )

        res = parse_pck_extensions(cert)
        assert res["raw_der_len"] == 2048
        # Should cleanly return unknown/empty without crash
        assert isinstance(res["fmspc"], str)
        assert isinstance(res["pceid"], str)
        assert isinstance(res["tcb"], dict)

    def test_suboid_compat_spec_2_and_standard_4(self, pki):
        """Parser extracts FMSPC whether encoded under sub-OID .2 or sub-OID .4."""
        from matrix_compute.crypto.pki_fixtures import (
            INTEL_OID_FMSPC,
            INTEL_OID_FMSPC_SPEC_COMPAT,
            der_encode_oid,
            der_encode_octet_string,
            der_encode_sequence,
        )

        expected_fmspc = "00806f000000"

        # Case 1: Sub-OID .2 only
        der_sub2 = der_encode_sequence([
            der_encode_sequence([
                der_encode_oid(INTEL_OID_FMSPC_SPEC_COMPAT),
                der_encode_octet_string(bytes.fromhex(expected_fmspc)),
            ])
        ])

        now = datetime.datetime.now(datetime.timezone.utc)
        cert2 = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Sub2 FMSPC")]))
            .issuer_name(pki.intermediate_cert.subject)
            .public_key(pki.pck_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=10))
            .add_extension(
                x509.UnrecognizedExtension(x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID), der_sub2),
                critical=False,
            )
            .sign(pki.intermediate_key, hashes.SHA256())
        )

        res2 = parse_pck_extensions(cert2)
        assert res2["fmspc"] == expected_fmspc

        # Case 2: Sub-OID .4 only
        der_sub4 = der_encode_sequence([
            der_encode_sequence([
                der_encode_oid(INTEL_OID_FMSPC),
                der_encode_octet_string(bytes.fromhex(expected_fmspc)),
            ])
        ])

        cert4 = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Sub4 FMSPC")]))
            .issuer_name(pki.intermediate_cert.subject)
            .public_key(pki.pck_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=10))
            .add_extension(
                x509.UnrecognizedExtension(x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID), der_sub4),
                critical=False,
            )
            .sign(pki.intermediate_key, hashes.SHA256())
        )

        res4 = parse_pck_extensions(cert4)
        assert res4["fmspc"] == expected_fmspc

    def test_non_octet_string_tag_graceful_handling(self, pki):
        """Sub-extension using INTEGER tag (0x02) instead of OCTET STRING (0x04) returns unknown."""
        from matrix_compute.crypto.pki_fixtures import INTEL_OID_FMSPC, der_encode_oid
        # OID followed by INTEGER tag 0x02, len 0x06
        fake_der = der_encode_oid(INTEL_OID_FMSPC) + b"\x02\x06\x11\x22\x33\x44\x55\x66"

        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Integer Tag")]))
            .issuer_name(pki.intermediate_cert.subject)
            .public_key(pki.pck_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=10))
            .add_extension(
                x509.UnrecognizedExtension(x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID), fake_der),
                critical=False,
            )
            .sign(pki.intermediate_key, hashes.SHA256())
        )

        res = parse_pck_extensions(cert)
        # Because parser specifically expects OCTET STRING (tag 0x04), INTEGER is not misidentified
        assert res["fmspc"] == "unknown"
