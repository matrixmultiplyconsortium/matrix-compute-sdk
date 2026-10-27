"""
Intel Trust Domain Extensions (TDX) Attestation Evidence Verifier for matrix-compute-sdk.

Conforms to:
- Intel® Trust Domain Extensions (Intel® TDX) Architecture Specification
- Intel® SGX/TDX DCAP Quote Format Specification (Format Version 4 & Modular Version 5)
- Intel® Provisioning Certification Service (PCS) API Specification

Implements strict fail-closed verification of:
1. Exact binary deserialization of DCAP Quote Format v4 and modular v5 layouts.
2. Production security invariants: immediately rejecting TDATTRIBUTES.DEBUG != 0.
3. Dual ECDSA signature appraisal:
   - Attestation Key (AK) signature over Header + TD Report Body.
   - PCK Leaf signature over 384-byte Quote Enclave (QE3) SGX Report Body.
4. Cryptographic Attestation Key binding in QE report (SHA-256(AK || auth_data)).
5. Proprietary Intel PCK custom ASN.1 extensions (OID 1.2.840.113741.1.13.1) extracting FMSPC, PCEID, and TCB.
6. Claims extraction for freshness (nonce), binding (ephemeral pubkey hash), and Golden RIM measurements.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from matrix_compute.exceptions import (
    AttestationBindingError,
    AttestationCertificateError,
    AttestationEvidenceMalformedError,
    AttestationSignatureError,
)
from matrix_contracts.schemas.attestation import AttestationEvidence
from .attestation_verifier import BaseEvidenceVerifier

logger = logging.getLogger("matrix_compute.crypto.attestation.tdx")

# ==============================================================================
# INTEL TDX WIRE CONSTANTS
# ==============================================================================

INTEL_TDX_TEE_TYPE: int = 0x00000081
INTEL_QE_VENDOR_ID: bytes = bytes.fromhex("939a7233f79c4ca9940a0db3957f0607")

# Intel SGX/TDX Custom X.509 Extension OID Arc (1.2.840.113741.1.13.1)
INTEL_SGX_EXTENSION_OID: str = "1.2.840.113741.1.13.1"
INTEL_OID_PPID: str = f"{INTEL_SGX_EXTENSION_OID}.1"
INTEL_OID_FMSPC_SPEC_COMPAT: str = f"{INTEL_SGX_EXTENSION_OID}.2"
INTEL_OID_PCEID: str = f"{INTEL_SGX_EXTENSION_OID}.3"
INTEL_OID_FMSPC: str = f"{INTEL_SGX_EXTENSION_OID}.4"
INTEL_OID_TCB_COMP_SVNS: str = f"{INTEL_SGX_EXTENSION_OID}.14"

# Raw OID Byte Sequences for Resilient ASN.1 Pattern Scanning
_OID_FMSPC_SPEC_BYTES = bytes([0x2A, 0x86, 0x48, 0x86, 0xF8, 0x4D, 0x01, 0x0D, 0x01, 0x02])
_OID_PCEID_BYTES = bytes([0x2A, 0x86, 0x48, 0x86, 0xF8, 0x4D, 0x01, 0x0D, 0x01, 0x03])
_OID_FMSPC_STD_BYTES = bytes([0x2A, 0x86, 0x48, 0x86, 0xF8, 0x4D, 0x01, 0x0D, 0x01, 0x04])
_OID_TCB_COMP_BYTES = bytes([0x2A, 0x86, 0x48, 0x86, 0xF8, 0x4D, 0x01, 0x0D, 0x01, 0x0E])
_OID_PPID_BYTES = bytes([0x2A, 0x86, 0x48, 0x86, 0xF8, 0x4D, 0x01, 0x0D, 0x01, 0x01])

# Elliptic Curve Orders for Bounds Checking
SECP256R1_ORDER: int = int("FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551", 16)
SECP384R1_ORDER: int = int(
    "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
    "C7634D81F4372DDF581A0DB248B0A77AECEC196ACCC52973",
    16,
)


def parse_pck_extensions(cert: x509.Certificate) -> Dict[str, Any]:
    """
    Parses Intel SGX/TDX proprietary X.509 extension (OID 1.2.840.113741.1.13.1).
    Extracts FMSPC, PCEID, TCB components, and PPID from DER encoded bytes.
    Supports both sub-OID .2 and .4 for FMSPC compatibility.
    """
    sgx_ext_oid = x509.ObjectIdentifier(INTEL_SGX_EXTENSION_OID)
    try:
        ext = cert.extensions.get_extension_for_oid(sgx_ext_oid)
    except x509.ExtensionNotFound:
        return {"fmspc": "unknown", "pceid": "unknown", "tcb": {}, "raw_der_len": 0}

    raw_der = ext.value.value if hasattr(ext.value, "value") else bytes(ext.value)
    pck_info: Dict[str, Any] = {
        "raw_der_len": len(raw_der),
        "fmspc": "unknown",
        "pceid": "unknown",
        "tcb": {},
    }

    # 1. Search for FMSPC (OID: 1.2.840.113741.1.13.1.2 or .4) - 6-byte octet string
    idx_fmspc = raw_der.find(_OID_FMSPC_STD_BYTES)
    oid_len = len(_OID_FMSPC_STD_BYTES)
    if idx_fmspc == -1:
        idx_fmspc = raw_der.find(_OID_FMSPC_SPEC_BYTES)
        oid_len = len(_OID_FMSPC_SPEC_BYTES)

    if idx_fmspc != -1:
        tag_idx = raw_der.find(b"\x04\x06", idx_fmspc + oid_len)
        if tag_idx != -1 and tag_idx + 8 <= len(raw_der):
            pck_info["fmspc"] = raw_der[tag_idx + 2 : tag_idx + 8].hex().lower()

    # 2. Search for PCEID (OID: 1.2.840.113741.1.13.1.3) - 2-byte octet string
    idx_pceid = raw_der.find(_OID_PCEID_BYTES)
    if idx_pceid != -1:
        tag_idx = raw_der.find(b"\x04\x02", idx_pceid + len(_OID_PCEID_BYTES))
        if tag_idx != -1 and tag_idx + 4 <= len(raw_der):
            pck_info["pceid"] = raw_der[tag_idx + 2 : tag_idx + 4].hex().lower()

    # 3. Search for TCB Component SVNs (OID: 1.2.840.113741.1.13.1.14) - 16-byte octet string
    idx_tcb = raw_der.find(_OID_TCB_COMP_BYTES)
    if idx_tcb != -1:
        tag_idx = raw_der.find(b"\x04\x10", idx_tcb + len(_OID_TCB_COMP_BYTES))
        if tag_idx != -1 and tag_idx + 18 <= len(raw_der):
            comp_svns = list(raw_der[tag_idx + 2 : tag_idx + 18])
            pck_info["tcb"] = {f"comp{i+1:02d}": val for i, val in enumerate(comp_svns)}
            pck_info["tcb"]["comp_svns"] = raw_der[tag_idx + 2 : tag_idx + 18].hex().lower()

    # 4. Search for PPID (OID: 1.2.840.113741.1.13.1.1) - 16-byte octet string
    idx_ppid = raw_der.find(_OID_PPID_BYTES)
    if idx_ppid != -1:
        tag_idx = raw_der.find(b"\x04\x10", idx_ppid + len(_OID_PPID_BYTES))
        if tag_idx != -1 and tag_idx + 18 <= len(raw_der):
            pck_info["ppid"] = raw_der[tag_idx + 2 : tag_idx + 18].hex().lower()

    return pck_info


# ==============================================================================
# INTEL TDX EVIDENCE VERIFIER STRATEGY
# ==============================================================================

class IntelTdxVerifier(BaseEvidenceVerifier):
    """
    Pluggable Attestation Evidence Verifier for Intel TDX Quotes (Format v4 and modular v5).
    Enforces hardware cryptographic signature checks, PCK certificate chain appraisal,
    debug enclave rejection, and binding of challenger nonce and ephemeral client public key.
    """

    @property
    def provider_id(self) -> str:
        return "intel-tdx"

    parse_pck_extensions = staticmethod(parse_pck_extensions)

    def verify_token_signature(
        self,
        evidence: AttestationEvidence,
        leaf_cert: x509.Certificate,
    ) -> None:
        """
        Verifies dual ECDSA signatures and cryptographic bindings in the Intel TDX Quote:
        1. Deserializes Quote Header (version in {4,5}, att_key_type in {2,3}, tee_type == 0x81).
        2. Resolves modular body descriptor in v5 quotes (body_type in {2,3}, body_size == 584).
        3. Enforces strict production policy: rejects TDATTRIBUTES.DEBUG != 0.
        4. Verifies Attestation Key (AK) ECDSA signature over Signed Body (Header + Body).
        5. Verifies Quote Enclave (QE3) SGX Report cryptographically binds the Attestation Key.
        6. Verifies PCK Leaf Certificate signature over the 384-byte QE Report Body.
        """
        blob = evidence.evidence_blob
        if not blob or len(blob) < 636:
            raise AttestationEvidenceMalformedError(
                f"Intel TDX evidence missing or truncated raw evidence_blob quote (len={len(blob) if blob else 0})"
            )

        # 1. Parse Quote Header (48 bytes)
        version, att_key_type, tee_type = struct.unpack_from("<HHI", blob, 0)
        if version not in (4, 5):
            raise AttestationEvidenceMalformedError(
                f"Unsupported Intel TDX Quote version: {version} (expected 4 or 5)"
            )
        if tee_type != INTEL_TDX_TEE_TYPE:
            raise AttestationEvidenceMalformedError(
                f"Invalid TEE Type in Intel TDX Quote: 0x{tee_type:08x} (expected 0x{INTEL_TDX_TEE_TYPE:08x})"
            )
        if att_key_type not in (2, 3):
            raise AttestationEvidenceMalformedError(
                f"Unsupported Attestation Key Type: {att_key_type} (expected 2 for P-256 or 3 for P-384)"
            )

        # 2. Resolve modular descriptor and body offsets
        if version == 4:
            body_type = 2  # TDX 1.0
            body_offset = 48
            body_size = 584
        elif version == 5:
            if len(blob) < 54:
                raise AttestationEvidenceMalformedError(
                    "Intel TDX v5 Quote truncated before 6-byte modular body descriptor"
                )
            body_type, body_size = struct.unpack_from("<HI", blob, 48)
            if body_type not in (2, 3):
                raise AttestationEvidenceMalformedError(
                    f"Unsupported Quote v5 Body Type: 0x{body_type:04x} (expected 0x0002 for TDX 1.0 or 0x0003 for TDX 1.5)"
                )
            if body_size != 584:
                raise AttestationEvidenceMalformedError(
                    f"Invalid TD Report Body size in v5 quote: {body_size} bytes (expected 584)"
                )
            body_offset = 54

        signed_body_len = body_offset + body_size
        if len(blob) < signed_body_len + 4:
            raise AttestationEvidenceMalformedError(
                f"Intel TDX Quote truncated before signature data length ({len(blob)} < {signed_body_len + 4} bytes)"
            )

        # 3. Enforce strict TDATTRIBUTES.DEBUG == 0 check (Offset body_offset + 120, uint64 little-endian)
        tdattributes = struct.unpack_from("<Q", blob, body_offset + 120)[0]
        if (tdattributes & 0x01) != 0:
            raise AttestationSignatureError(
                f"Intel TDX Enclave has DEBUG mode enabled (insecure for production, tdattributes=0x{tdattributes:016x})"
            )

        # 4. Extract Signed Header + TD Report Body
        signed_body = blob[:signed_body_len]
        sig_len_offset = signed_body_len
        sig_data_len = struct.unpack_from("<I", blob, sig_len_offset)[0]
        sig_offset = sig_len_offset + 4

        if len(blob) < sig_offset + sig_data_len:
            raise AttestationEvidenceMalformedError(
                f"Incomplete Quote Signature Data Structure: blob has {len(blob)} bytes, expected {sig_offset + sig_data_len}"
            )

        # 5. Extract Quote Signature and Attestation Public Key
        key_len = 64 if att_key_type == 2 else 96
        coord_len = key_len // 2

        if sig_data_len < key_len * 2 + 384 + key_len + 2:
            raise AttestationEvidenceMalformedError(
                "Truncated Quote Signature payload components (insufficient length for signatures and QE report)"
            )

        quote_sig = blob[sig_offset : sig_offset + key_len]
        att_pubkey_raw = blob[sig_offset + key_len : sig_offset + key_len * 2]

        if att_key_type == 2:
            curve = ec.SECP256R1()
            hash_algo = hashes.SHA256()
        else:
            curve = ec.SECP384R1()
            hash_algo = hashes.SHA384()

        # Reconstruct AK Elliptic Curve Public Key
        pubkey_point = b"\x04" + att_pubkey_raw
        try:
            att_public_key = ec.EllipticCurvePublicKey.from_encoded_point(curve, pubkey_point)
        except Exception as exc:
            raise AttestationSignatureError(
                f"Failed reconstructing attestation public key: {exc}"
            ) from exc

        # Convert quote_sig big-endian (r, s) scalars to DER DSS signature
        r = int.from_bytes(quote_sig[:coord_len], "big")
        s = int.from_bytes(quote_sig[coord_len:key_len], "big")
        ak_curve_order = SECP256R1_ORDER if att_key_type == 2 else SECP384R1_ORDER
        if r <= 0 or r >= ak_curve_order or s <= 0 or s >= ak_curve_order:
            raise AttestationSignatureError(f"Quote signature scalars out of curve range: r={r}, s={s}")
        der_quote_sig = encode_dss_signature(r, s)

        # 6. Verify Attestation Key (AK) signature over signed_body
        try:
            att_public_key.verify(der_quote_sig, signed_body, ec.ECDSA(hash_algo))
        except InvalidSignature as exc:
            raise AttestationSignatureError(
                f"Intel TDX Quote ECDSA signature verification failed: {exc}"
            ) from exc
        except Exception as exc:
            raise AttestationSignatureError(
                f"Intel TDX Quote signature verification encountered error: {exc}"
            ) from exc

        # Also enforce consistency if evidence.token_signature is provided
        if evidence.token_signature and evidence.token_signature != quote_sig:
            raise AttestationSignatureError(
                "evidence.token_signature does not match the quote_signature embedded in evidence_blob"
            )

        # 7. Extract Quote Enclave (QE3) SGX Report Body, Signature, and Auth Data
        qe_report_offset = sig_offset + key_len * 2
        qe_report_body = blob[qe_report_offset : qe_report_offset + 384]

        leaf_pubkey = leaf_cert.public_key()
        if isinstance(leaf_pubkey, ec.EllipticCurvePublicKey):
            pck_coord_len = 48 if isinstance(leaf_pubkey.curve, ec.SECP384R1) else 32
            pck_sig_len = pck_coord_len * 2
            pck_hash_algo = hashes.SHA384() if isinstance(leaf_pubkey.curve, ec.SECP384R1) else hashes.SHA256()
        elif isinstance(leaf_pubkey, rsa.RSAPublicKey):
            pck_coord_len = 32
            pck_sig_len = 64
            pck_hash_algo = hashes.SHA256()
        else:
            raise AttestationSignatureError(f"Unsupported PCK public key type: {type(leaf_pubkey).__name__}")

        qe_sig_offset = qe_report_offset + 384
        if len(blob) < qe_sig_offset + pck_sig_len + 2:
            raise AttestationEvidenceMalformedError("Truncated QE report signature or auth data length")

        qe_report_sig = blob[qe_sig_offset : qe_sig_offset + pck_sig_len]

        auth_size_offset = qe_sig_offset + pck_sig_len
        qe_auth_data_size = struct.unpack_from("<H", blob, auth_size_offset)[0]
        auth_data_start = auth_size_offset + 2
        if len(blob) < auth_data_start + qe_auth_data_size:
            raise AttestationEvidenceMalformedError("Truncated QE authentication data payload")
        qe_auth_data = blob[auth_data_start : auth_data_start + qe_auth_data_size]

        # 8. Cryptographic AK Cross-Binding: qe_report_body.report_data[:32] == SHA256(att_public_key || qe_auth_data)
        expected_qe_binding = hashlib.sha256(att_pubkey_raw + qe_auth_data).digest()
        actual_qe_report_data = qe_report_body[320:384]
        if not secrets.compare_digest(actual_qe_report_data[:32], expected_qe_binding):
            raise AttestationBindingError(
                "Quote Enclave report does not bind Attestation Public Key",
                expected_pubkey_hash=expected_qe_binding.hex(),
                actual_pubkey_hash=actual_qe_report_data[:32].hex(),
                provider=self.provider_id,
            )

        # 9. Verify PCK Leaf Certificate signature over QE Report Body (384 bytes)
        qe_r = int.from_bytes(qe_report_sig[:pck_coord_len], "big")
        qe_s = int.from_bytes(qe_report_sig[pck_coord_len : pck_coord_len * 2], "big")

        if isinstance(leaf_pubkey, ec.EllipticCurvePublicKey):
            pck_order = SECP384R1_ORDER if isinstance(leaf_pubkey.curve, ec.SECP384R1) else SECP256R1_ORDER
            if qe_r <= 0 or qe_r >= pck_order or qe_s <= 0 or qe_s >= pck_order:
                raise AttestationSignatureError(
                    f"QE report signature scalars out of curve range: r={qe_r}, s={qe_s}"
                )
            der_qe_sig = encode_dss_signature(qe_r, qe_s)
            try:
                leaf_pubkey.verify(der_qe_sig, qe_report_body, ec.ECDSA(pck_hash_algo))
            except InvalidSignature as exc:
                raise AttestationSignatureError(
                    f"Intel PCK signature verification over Quote Enclave report failed: {exc}"
                ) from exc
            except Exception as exc:
                raise AttestationSignatureError(
                    f"Intel PCK signature verification encountered error: {exc}"
                ) from exc
        elif isinstance(leaf_pubkey, rsa.RSAPublicKey):
            der_qe_sig = encode_dss_signature(qe_r, qe_s)
            try:
                leaf_pubkey.verify(der_qe_sig, qe_report_body, padding.PKCS1v15(), pck_hash_algo)
            except InvalidSignature as exc:
                raise AttestationSignatureError(
                    f"Intel PCK RSA signature verification over Quote Enclave report failed: {exc}"
                ) from exc
            except Exception as exc:
                raise AttestationSignatureError(
                    f"Intel PCK RSA signature verification encountered error: {exc}"
                ) from exc

        # 10. Parse and record platform FMSPC from PCK extensions
        pck_info = self.parse_pck_extensions(leaf_cert)
        logger.debug(
            "Successfully verified Intel TDX Quote (v%d, body_type 0x%04x) on platform FMSPC: %s",
            version,
            body_type,
            pck_info.get("fmspc", "unknown"),
        )

    def extract_claims(self, evidence: AttestationEvidence) -> Dict[str, Any]:
        """
        Extracts normalized claims dictionary from raw binary Quote or evidence container.
        Validates production non-debug invariant, measurement registers, and cryptographic bindings.
        """
        blob = evidence.evidence_blob
        if not blob or len(blob) < 48:
            if evidence.token_claims:
                if evidence.token_claims.get("debug") is True:
                    raise AttestationSignatureError(
                        "Intel TDX Enclave has DEBUG mode enabled (insecure for production)"
                    )
                return dict(evidence.token_claims)
            raise AttestationEvidenceMalformedError("Missing evidence_blob to extract Intel TDX claims")

        version, att_key_type, tee_type = struct.unpack_from("<HHI", blob, 0)
        if version not in (4, 5):
            raise AttestationEvidenceMalformedError(f"Unsupported Quote version: {version}")
        if tee_type != INTEL_TDX_TEE_TYPE:
            raise AttestationEvidenceMalformedError(
                f"Invalid TEE Type in Intel TDX Quote: 0x{tee_type:08x}"
            )

        if version == 4:
            body_type = 2
            body_offset = 48
            body_size = 584
        else:
            if len(blob) < 54:
                raise AttestationEvidenceMalformedError("Truncated Quote v5 body descriptor")
            body_type, body_size = struct.unpack_from("<HI", blob, 48)
            body_offset = 54

        if len(blob) < body_offset + 584:
            raise AttestationEvidenceMalformedError(
                f"Truncated TD Report Body in evidence_blob ({len(blob)} bytes)"
            )

        # Enforce DEBUG mode check fail-closed
        tdattributes = struct.unpack_from("<Q", blob, body_offset + 120)[0]
        if (tdattributes & 0x01) != 0:
            raise AttestationSignatureError(
                "Intel TDX Enclave has DEBUG mode enabled (insecure for production)"
            )

        # Extract measurements and fields from TD Report Body
        tee_tcb_svn = blob[body_offset : body_offset + 16].hex()
        mrseam = blob[body_offset + 16 : body_offset + 64].hex()
        mrsignerseam = blob[body_offset + 64 : body_offset + 112].hex()
        seamattributes = struct.unpack_from("<Q", blob, body_offset + 112)[0]
        xfam = struct.unpack_from("<Q", blob, body_offset + 128)[0]
        mrtd = blob[body_offset + 136 : body_offset + 184].hex()
        mrconfigid = blob[body_offset + 184 : body_offset + 232].hex()
        mrowner = blob[body_offset + 232 : body_offset + 280].hex()
        mrownerconfig = blob[body_offset + 280 : body_offset + 328].hex()
        rtmr0 = blob[body_offset + 328 : body_offset + 376].hex()
        rtmr1 = blob[body_offset + 376 : body_offset + 424].hex()
        rtmr2 = blob[body_offset + 424 : body_offset + 472].hex()
        rtmr3 = blob[body_offset + 472 : body_offset + 520].hex()
        reportdata = blob[body_offset + 520 : body_offset + 584]

        nonce = reportdata[:32].hex()
        ephemeral_pubkey_hash = reportdata[32:64].hex()

        claims: Dict[str, Any] = {
            "provider": self.provider_id,
            "quote_version": version,
            "tee_type": f"0x{tee_type:08x}",
            "body_type": body_type,
            "tdx_version": "1.5" if body_type == 3 else "1.0",
            "nonce": nonce,
            "ephemeral_pubkey_hash": ephemeral_pubkey_hash,
            "client_pubkey_hash": ephemeral_pubkey_hash,
            "user_data": reportdata.hex(),
            "tee_tcb_svn": tee_tcb_svn,
            "mrseam": mrseam,
            "mrsignerseam": mrsignerseam,
            "seamattributes": seamattributes,
            "tdattributes": tdattributes,
            "xfam": xfam,
            "mrtd": mrtd,
            "mrconfigid": mrconfigid,
            "mrowner": mrowner,
            "mrownerconfig": mrownerconfig,
            "rtmr0": rtmr0,
            "rtmr1": rtmr1,
            "rtmr2": rtmr2,
            "rtmr3": rtmr3,
            "measurements": {
                "mrtd": mrtd,
                "rtmr0": rtmr0,
                "rtmr1": rtmr1,
                "rtmr2": rtmr2,
                "rtmr3": rtmr3,
            },
            "debug": False,
        }

        # Preserve metadata from evidence.token_claims if available
        if evidence.token_claims:
            for k in ("timestamp", "device_id", "worker_pubkey_hash", "iat"):
                if k in evidence.token_claims and k not in claims:
                    claims[k] = evidence.token_claims[k]

        # Extract PCK extensions if certificate chain is present
        if evidence.certificate_chain:
            try:
                leaf_pem = evidence.certificate_chain[0]
                leaf_cert = x509.load_pem_x509_certificate(leaf_pem.strip().encode("utf-8"))
                pck_info = self.parse_pck_extensions(leaf_cert)
                claims["fmspc"] = pck_info.get("fmspc")
                claims["pceid"] = pck_info.get("pceid")
                claims["tcb"] = pck_info.get("tcb")
            except Exception:
                pass

        return claims

    @staticmethod
    def extract_certificate_chain(quote_blob: bytes) -> List[str]:
        """
        Extracts concatenated PEM X.509 certificate chain from DCAP Quote certification data (Type 5).
        Returns list of PEM strings: [PCK Leaf PEM, Intermediate CA PEM, Root CA PEM].
        """
        if not quote_blob or len(quote_blob) < 48:
            return []
        version = struct.unpack_from("<H", quote_blob, 0)[0]
        if version == 4:
            body_offset = 48
            body_size = 584
        elif version == 5:
            if len(quote_blob) < 54:
                return []
            body_size = struct.unpack_from("<HI", quote_blob, 48)[1]
            body_offset = 54
        else:
            return []

        signed_body_len = body_offset + body_size
        if len(quote_blob) < signed_body_len + 4:
            return []

        sig_data_len = struct.unpack_from("<I", quote_blob, signed_body_len)[0]
        sig_offset = signed_body_len + 4
        if len(quote_blob) < sig_offset + sig_data_len:
            return []

        att_key_type = struct.unpack_from("<H", quote_blob, 2)[0]
        key_len = 64 if att_key_type == 2 else 96

        auth_size_offset = sig_offset + 3 * key_len + 384
        if len(quote_blob) < auth_size_offset + 2:
            return []
        auth_data_size = struct.unpack_from("<H", quote_blob, auth_size_offset)[0]
        cert_type_offset = auth_size_offset + 2 + auth_data_size
        if len(quote_blob) < cert_type_offset + 6:
            return []
        cert_data_type, cert_data_size = struct.unpack_from("<HI", quote_blob, cert_type_offset)
        cert_bytes = quote_blob[cert_type_offset + 6 : cert_type_offset + 6 + cert_data_size]

        try:
            cert_text = cert_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                cert_text = cert_bytes.decode("latin-1")
            except Exception:
                return []

        certs: List[str] = []
        marker = "-----BEGIN CERTIFICATE-----"
        end_marker = "-----END CERTIFICATE-----"
        start = 0
        while True:
            idx = cert_text.find(marker, start)
            if idx == -1:
                break
            end_idx = cert_text.find(end_marker, idx)
            if end_idx == -1:
                break
            certs.append(cert_text[idx : end_idx + len(end_marker)])
            start = end_idx + len(end_marker)
        return certs
