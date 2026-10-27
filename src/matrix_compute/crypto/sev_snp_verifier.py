"""
AMD SEV-SNP Attestation Evidence Verifier for matrix-compute-sdk.

Conforms to AMD SEV-SNP Firmware ABI Specification (AMD Pub. #56860)
and AMD SEV-SNP Attestation: VCEK Certificate and KDS Interface Specification (AMD Pub. #57230).

Implements strict fail-closed verification of:
1. Exact 1184-byte SEV-SNP report wire format.
2. Guest policy invariants, specifically rejecting POLICY.DEBUG == 1.
3. Zero padding in signature component buffers (high 24 bytes of R/S and 368-byte signature pad).
4. Little-endian 48-byte scalar extraction and canonical DER DSS ECDSA signature conversion.
5. Cryptographic signature verification over bytes 0x000..0x29F using VCEK ECDSA P-384 public key.
6. Validation of custom AMD X.509 extensions (1.3.6.1.4.1.3704.1) against report chip_id and TCB SPLs.
7. Claims extraction for freshness (nonce), binding (ephemeral pubkey hash), and Golden RIM appraisal.
"""

from __future__ import annotations

import logging
import secrets
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from matrix_compute.exceptions import (
    AttestationCertificateError,
    AttestationEvidenceMalformedError,
    AttestationSignatureError,
)
from matrix_contracts.schemas.attestation import AttestationEvidence
from .attestation_verifier import BaseEvidenceVerifier

logger = logging.getLogger("matrix_compute.crypto.attestation.sev_snp")

# ==============================================================================
# AMD SEV-SNP WIRE CONSTANTS (AMD Pub. #56860 & #57230)
# ==============================================================================

SNP_REPORT_LEN: int = 1184
SNP_SIGNED_PORTION_LEN: int = 672  # Bytes 0x000 through 0x29F inclusive

SNP_SIG_ALGO_ECDSA_P384_SHA384: int = 1
SNP_POLICY_DEBUG_BIT_POSITION: int = 19
SNP_POLICY_DEBUG_MASK: int = 1 << SNP_POLICY_DEBUG_BIT_POSITION

# Signature component offsets and sizes
SNP_SIG_R_OFFSET: int = 0x2A0
SNP_SIG_S_OFFSET: int = 0x2E8
SNP_SCALAR_LEN: int = 48  # NIST P-384 384-bit scalar = 48 bytes
SNP_COMPONENT_LEN: int = 72  # 48 bytes scalar + 24 bytes zero padding
SNP_PADDING_LEN: int = 24  # 72 - 48 = 24 bytes
SNP_SIG_PAD_OFFSET: int = 0x330
SNP_SIG_PAD_LEN: int = 368  # 1184 - 0x330 = 368 bytes

# SECP384R1 curve order n for bounds checking
SECP384R1_ORDER: int = int(
    "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
    "C7634D81F4372DDF581A0DB248B0A77AECEC196ACCC52973",
    16,
)

# AMD Custom X.509 OID Arc (1.3.6.1.4.1.3704.1)
AMD_OID_PREFIX: str = "1.3.6.1.4.1.3704.1"
AMD_OID_STRUCT_VERSION: str = f"{AMD_OID_PREFIX}.1"       # INTEGER
AMD_OID_PRODUCT_NAME: str = f"{AMD_OID_PREFIX}.2"         # IA5String
AMD_OID_BL_SPL: str = f"{AMD_OID_PREFIX}.3.1"             # INTEGER
AMD_OID_TEE_SPL: str = f"{AMD_OID_PREFIX}.3.2"            # INTEGER
AMD_OID_SNP_SPL: str = f"{AMD_OID_PREFIX}.3.3"            # INTEGER
AMD_OID_UCODE_SPL: str = f"{AMD_OID_PREFIX}.3.8"          # INTEGER
AMD_OID_HW_ID: str = f"{AMD_OID_PREFIX}.4"                # OCTET STRING (64 bytes)


# ==============================================================================
# ASN.1 DER PARSING UTILITIES FOR CUSTOM AMD EXTENSIONS
# ==============================================================================

def der_decode_integer(raw: bytes) -> int:
    """DER decodes an ASN.1 INTEGER (Tag 0x02) with strict length validation."""
    if len(raw) < 2 or raw[0] != 0x02:
        raise ValueError(f"Invalid DER integer tag: {raw[:2]!r}")
    length = raw[1]
    offset = 2
    if length & 0x80:
        num_len_bytes = length & 0x7F
        if num_len_bytes == 0:
            raise ValueError("Indefinite length (0x80) is forbidden in DER")
        if len(raw) < offset + num_len_bytes:
            raise ValueError("Truncated DER integer length bytes")
        length = int.from_bytes(raw[offset : offset + num_len_bytes], "big")
        offset += num_len_bytes
    if len(raw) < offset + length:
        raise ValueError("Truncated DER integer payload bytes")
    if len(raw) > offset + length:
        raise ValueError(
            f"Unconsumed trailing bytes in DER integer: expected {offset + length} bytes, got {len(raw)}"
        )
    return int.from_bytes(raw[offset : offset + length], byteorder="big", signed=True)


def der_decode_ia5string(raw: bytes) -> str:
    """DER decodes an ASN.1 IA5String (Tag 0x16) or PrintableString (Tag 0x13)."""
    if len(raw) < 2 or raw[0] not in (0x16, 0x13):
        raise ValueError(f"Invalid DER string tag: {raw[:2]!r}")
    length = raw[1]
    offset = 2
    if length & 0x80:
        num_len_bytes = length & 0x7F
        if num_len_bytes == 0:
            raise ValueError("Indefinite length (0x80) is forbidden in DER")
        if len(raw) < offset + num_len_bytes:
            raise ValueError("Truncated DER string length bytes")
        length = int.from_bytes(raw[offset : offset + num_len_bytes], "big")
        offset += num_len_bytes
    if len(raw) < offset + length:
        raise ValueError("Truncated DER string payload bytes")
    if len(raw) > offset + length:
        raise ValueError(
            f"Unconsumed trailing bytes in DER string: expected {offset + length} bytes, got {len(raw)}"
        )
    return raw[offset : offset + length].decode("ascii")


def der_decode_octet_string(raw: bytes) -> bytes:
    """DER decodes an ASN.1 OCTET STRING (Tag 0x04) with strict length validation."""
    if len(raw) < 2 or raw[0] != 0x04:
        raise ValueError(f"Invalid DER octet string tag: {raw[:2]!r}")
    length = raw[1]
    offset = 2
    if length & 0x80:
        num_len_bytes = length & 0x7F
        if num_len_bytes == 0:
            raise ValueError("Indefinite length (0x80) is forbidden in DER")
        if len(raw) < offset + num_len_bytes:
            raise ValueError("Truncated DER octet string length bytes")
        length = int.from_bytes(raw[offset : offset + num_len_bytes], "big")
        offset += num_len_bytes
    if len(raw) < offset + length:
        raise ValueError("Truncated DER octet string payload bytes")
    if len(raw) > offset + length:
        raise ValueError(
            f"Unconsumed trailing bytes in DER octet string: expected {offset + length} bytes, got {len(raw)}"
        )
    return raw[offset : offset + length]


def parse_vcek_extensions(cert: x509.Certificate) -> Dict[str, Any]:
    """
    Parses custom AMD X.509 extensions under OID arc 1.3.6.1.4.1.3704.1 from a VCEK certificate.
    Returns a dictionary of parsed extensions: struct_version, product_name, blSPL, teeSPL,
    snpSPL, ucodeSPL, hwID.
    """
    extensions_dict: Dict[str, Any] = {}
    for ext in cert.extensions:
        oid_str = ext.oid.dotted_string
        if not oid_str.startswith(AMD_OID_PREFIX):
            continue
        val_bytes = getattr(ext.value, "value", None)
        if val_bytes is None or not isinstance(val_bytes, (bytes, bytearray)):
            continue
        try:
            if oid_str == AMD_OID_STRUCT_VERSION:
                extensions_dict["struct_version"] = der_decode_integer(val_bytes)
            elif oid_str == AMD_OID_PRODUCT_NAME:
                extensions_dict["product_name"] = der_decode_ia5string(val_bytes)
            elif oid_str == AMD_OID_BL_SPL:
                extensions_dict["blSPL"] = der_decode_integer(val_bytes)
            elif oid_str == AMD_OID_TEE_SPL:
                extensions_dict["teeSPL"] = der_decode_integer(val_bytes)
            elif oid_str == AMD_OID_SNP_SPL:
                extensions_dict["snpSPL"] = der_decode_integer(val_bytes)
            elif oid_str == AMD_OID_UCODE_SPL:
                extensions_dict["ucodeSPL"] = der_decode_integer(val_bytes)
            elif oid_str == AMD_OID_HW_ID:
                extensions_dict["hwID"] = der_decode_octet_string(val_bytes)
        except Exception as exc:
            raise AttestationCertificateError(
                f"Failed to decode custom AMD extension {oid_str}: {exc}",
                cause=exc,
            ) from exc
    return extensions_dict


def validate_vcek_extensions(cert: x509.Certificate, report_blob: bytes) -> Dict[str, Any]:
    """
    Validates custom AMD extensions on VCEK against report blob fields:
    - Enforces mandatory presence of custom AMD extensions and hwID
    - hwID matches chip_id at 0x1A0 (64 bytes)
    - TCB SPLs (blSPL, teeSPL, snpSPL, ucodeSPL) are non-negative uint8 and <= reported_tcb at 0x180 (8 bytes)
    """
    exts = parse_vcek_extensions(cert)
    if not exts or "hwID" not in exts or exts.get("hwID") is None:
        raise AttestationCertificateError(
            "VCEK leaf certificate missing required AMD custom extensions or hwID"
        )

    # Check hwID against report chip_id
    hw_id = exts["hwID"]
    report_chip_id = bytes(report_blob[0x1A0:0x1E0])
    if not secrets.compare_digest(hw_id, report_chip_id):
        raise AttestationCertificateError(
            f"Hardware ID mismatch: VCEK hwID ({hw_id.hex()}) != report chip_id ({report_chip_id.hex()})"
        )

    # Check TCB SPLs against reported_tcb (byte 0: BL, byte 1: TEE, byte 2: SNP, byte 7: ucode)
    reported_tcb = report_blob[0x180:0x188]
    spl_specs = (
        ("blSPL", 0, "bootloader SPL"),
        ("teeSPL", 1, "TEE SPL"),
        ("snpSPL", 2, "SNP SPL"),
        ("ucodeSPL", 7, "ucode SPL"),
    )
    for spl_key, tcb_idx, spl_desc in spl_specs:
        val = exts.get(spl_key)
        if val is not None:
            if not isinstance(val, int) or val < 0 or val > 255:
                raise AttestationCertificateError(
                    f"Invalid SPL value in VCEK extension {spl_key}: {val} (must be non-negative uint8 0..255)"
                )
            if reported_tcb[tcb_idx] < val:
                raise AttestationCertificateError(
                    f"TCB downgrade detected: reported {spl_desc} ({reported_tcb[tcb_idx]}) < VCEK SPL ({val})"
                )

    return exts


# ==============================================================================
# AMD SEV-SNP EVIDENCE VERIFIER STRATEGY
# ==============================================================================

class AmdSevSnpVerifier(BaseEvidenceVerifier):
    """
    Pluggable Attestation Evidence Verifier for AMD SEV-SNP Attestation Reports.
    Validates ECDSA P-384 hardware signature, VCEK certificate chain,
    non-debug guest policy, zero padding, and cryptographic nonce/pubkey binding.
    """

    @property
    def provider_id(self) -> str:
        return "amd-sev-snp"

    def verify_token_signature(
        self,
        evidence: AttestationEvidence,
        leaf_cert: x509.Certificate,
    ) -> None:
        """
        Verifies ECDSA P-384 signature over bytes 0x000..0x29F of the 1184-byte report
        using the public key of the VCEK leaf certificate.
        Enforces strict length check (1184B), policy debug rejection, and zero padding.
        """
        blob = evidence.evidence_blob
        if not blob or len(blob) != SNP_REPORT_LEN:
            raise AttestationEvidenceMalformedError(
                f"AMD SEV-SNP evidence blob length must be exactly {SNP_REPORT_LEN} bytes, got {len(blob) if blob else 0}"
            )

        # 1. Enforce strict POLICY.DEBUG == 0 check (Offset 0x008, uint64 little-endian)
        policy = struct.unpack_from("<Q", blob, 0x008)[0]
        if (policy >> SNP_POLICY_DEBUG_BIT_POSITION) & 0x01 != 0:
            raise AttestationSignatureError(
                f"AMD SEV-SNP Enclave has DEBUG policy enabled (POLICY.DEBUG == 1, policy=0x{policy:08x})"
            )

        # 2. Verify Signature Algorithm (0x034): Must be 1 (ECDSA P-384 with SHA-384)
        sig_algo = struct.unpack_from("<I", blob, 0x034)[0]
        if sig_algo != SNP_SIG_ALGO_ECDSA_P384_SHA384:
            raise AttestationSignatureError(
                f"Unsupported AMD SEV-SNP signature algorithm: {sig_algo} (expected {SNP_SIG_ALGO_ECDSA_P384_SHA384})"
            )

        # 3. Validate zero padding in signature buffers
        # High 24 bytes of R: 0x2D0..0x2E8
        if blob[0x2D0:0x2E8] != b"\x00" * SNP_PADDING_LEN:
            raise AttestationSignatureError("Non-zero padding detected in R scalar buffer (0x2D0..0x2E8)")

        # High 24 bytes of S: 0x318..0x330
        if blob[0x318:0x330] != b"\x00" * SNP_PADDING_LEN:
            raise AttestationSignatureError("Non-zero padding detected in S scalar buffer (0x318..0x330)")

        # Signature pad: 0x330..0x4A0 (368 bytes)
        if blob[SNP_SIG_PAD_OFFSET : SNP_SIG_PAD_OFFSET + SNP_SIG_PAD_LEN] != b"\x00" * SNP_SIG_PAD_LEN:
            raise AttestationSignatureError("Non-zero padding detected in signature pad buffer (0x330..0x4A0)")

        # 4. Extract 48-byte significant scalars R and S and convert to integers
        raw_r = blob[SNP_SIG_R_OFFSET : SNP_SIG_R_OFFSET + SNP_SCALAR_LEN]
        raw_s = blob[SNP_SIG_S_OFFSET : SNP_SIG_S_OFFSET + SNP_SCALAR_LEN]

        r_int = int.from_bytes(raw_r, byteorder="little")
        s_int = int.from_bytes(raw_s, byteorder="little")

        if r_int <= 0 or r_int >= SECP384R1_ORDER or s_int <= 0 or s_int >= SECP384R1_ORDER:
            raise AttestationSignatureError(
                f"ECDSA scalar values out of curve range: r={r_int}, s={s_int}"
            )

        try:
            der_signature = encode_dss_signature(r_int, s_int)
        except Exception as exc:
            raise AttestationSignatureError(
                f"Failed to encode canonical DER DSS signature from (r, s): {exc}"
            ) from exc

        # 5. Extract VCEK Public Key and verify ECDSA P-384 signature
        vcek_pubkey = leaf_cert.public_key()
        if not isinstance(vcek_pubkey, ec.EllipticCurvePublicKey) or not isinstance(
            vcek_pubkey.curve, ec.SECP384R1
        ):
            raise AttestationSignatureError(
                f"VCEK leaf certificate public key must be ECDSA NIST P-384 (SECP384R1), got {type(vcek_pubkey).__name__}"
            )

        signed_body = bytes(blob[:SNP_SIGNED_PORTION_LEN])
        try:
            vcek_pubkey.verify(der_signature, signed_body, ec.ECDSA(hashes.SHA384()))
        except InvalidSignature as exc:
            raise AttestationSignatureError(
                f"AMD SEV-SNP ECDSA P-384 report signature verification failed: {exc}",
                cause=exc,
            ) from exc
        except Exception as exc:
            raise AttestationSignatureError(
                f"AMD SEV-SNP signature verification encountered error: {exc}",
                cause=exc,
            ) from exc

        # 6. Validate custom VCEK extensions against report fields (hwID and TCB SPLs)
        validate_vcek_extensions(leaf_cert, blob)

    def extract_claims(self, evidence: AttestationEvidence) -> Dict[str, Any]:
        """
        Extracts normalized claims dictionary from raw 1184-byte report binary.
        Parses all header, policy, measurement, host data, and TCB fields.
        Enforces POLICY.DEBUG == 0 check fail-closed.
        """
        blob = evidence.evidence_blob
        if not blob or len(blob) != SNP_REPORT_LEN:
            raise AttestationEvidenceMalformedError(
                f"AMD SEV-SNP evidence blob must be exactly {SNP_REPORT_LEN} bytes, got {len(blob) if blob else 0}"
            )

        # Parse header fields
        version = struct.unpack_from("<I", blob, 0x000)[0]
        guest_svn = struct.unpack_from("<I", blob, 0x004)[0]
        policy = struct.unpack_from("<Q", blob, 0x008)[0]
        family_id = blob[0x010:0x020].hex()
        image_id = blob[0x020:0x030].hex()
        vmpl = struct.unpack_from("<I", blob, 0x030)[0]
        signature_algo = struct.unpack_from("<I", blob, 0x034)[0]
        platform_version = blob[0x038:0x040].hex()
        platform_info = struct.unpack_from("<Q", blob, 0x040)[0]
        flags = struct.unpack_from("<I", blob, 0x048)[0]

        # Enforce POLICY.DEBUG == 0
        policy_debug = bool((policy >> SNP_POLICY_DEBUG_BIT_POSITION) & 0x01)
        if policy_debug:
            raise AttestationSignatureError(
                f"AMD SEV-SNP Enclave has DEBUG policy enabled (POLICY.DEBUG == 1, policy=0x{policy:08x})"
            )

        # Data and Measurement fields
        report_data = blob[0x050:0x090]
        nonce = report_data[:32].hex()
        ephemeral_pubkey_hash = report_data[32:64].hex()

        measurement = blob[0x090:0x0C0].hex()
        host_data = blob[0x0C0:0x0E0].hex()
        id_key_digest = blob[0x0E0:0x110].hex()
        author_key_digest = blob[0x110:0x140].hex()
        report_id = blob[0x140:0x160].hex()
        report_id_ma = blob[0x160:0x180].hex()
        reported_tcb = blob[0x180:0x188].hex()
        chip_id = blob[0x1A0:0x1E0].hex()
        committed_tcb = blob[0x1E0:0x1E8].hex()
        current_build = struct.unpack_from("<I", blob, 0x1E8)[0]
        committed_build = struct.unpack_from("<I", blob, 0x1EC)[0]

        claims: Dict[str, Any] = {
            "provider": self.provider_id,
            "version": version,
            "guest_svn": guest_svn,
            "policy": policy,
            "policy_hex": f"0x{policy:08x}",
            "policy_debug": policy_debug,
            "family_id": family_id,
            "image_id": image_id,
            "vmpl": vmpl,
            "signature_algo": signature_algo,
            "platform_version": platform_version,
            "platform_info": platform_info,
            "flags": flags,
            "report_data": report_data.hex(),
            "nonce": nonce,
            "ephemeral_pubkey_hash": ephemeral_pubkey_hash,
            "client_pubkey_hash": ephemeral_pubkey_hash,
            "user_data": report_data.hex(),
            "measurement": measurement,
            "host_data": host_data,
            "measurements": {
                "measurement": measurement,
                "host_data": host_data,
            },
            "id_key_digest": id_key_digest,
            "author_key_digest": author_key_digest,
            "report_id": report_id,
            "report_id_ma": report_id_ma,
            "reported_tcb": reported_tcb,
            "chip_id": chip_id,
            "committed_tcb": committed_tcb,
            "current_build": current_build,
            "committed_build": committed_build,
        }

        # Preserve metadata from evidence.token_claims if available
        if evidence.token_claims:
            for k in ("timestamp", "device_id", "worker_pubkey_hash"):
                if k in evidence.token_claims and k not in claims:
                    claims[k] = evidence.token_claims[k]

        # Extract VCEK extensions if certificate chain is attached
        if evidence.certificate_chain:
            try:
                leaf_pem = evidence.certificate_chain[0]
                leaf_cert = x509.load_pem_x509_certificate(leaf_pem.strip().encode("utf-8"))
                vcek_exts = parse_vcek_extensions(leaf_cert)
                if vcek_exts:
                    claims["vcek_extensions"] = vcek_exts
                    claims["tcb_spls"] = {
                        "blSPL": vcek_exts.get("blSPL"),
                        "teeSPL": vcek_exts.get("teeSPL"),
                        "snpSPL": vcek_exts.get("snpSPL"),
                        "ucodeSPL": vcek_exts.get("ucodeSPL"),
                    }
            except Exception:
                pass

        return claims
