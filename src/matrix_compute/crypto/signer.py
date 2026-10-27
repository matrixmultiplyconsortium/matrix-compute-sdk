"""Secp256k1 Message Signer, Low-S Normalization, and Signature Verifier."""

from __future__ import annotations

import hashlib
import secrets
from typing import Any, Optional, Union

from .keys import (
    SECP256K1_HALF_ORDER,
    SECP256K1_ORDER,
    privkey_to_pubkey,
    pubkey_to_npub,
    pubkey_to_xonly,
    validate_private_key_hex,
)

try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import (
        decode_dss_signature,
        encode_dss_signature,
    )
    from cryptography.hazmat.primitives import hashes
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False
    try:
        import ecdsa
    except ImportError:
        ecdsa = None


# ============================================================================
# Hashing Utilities
# ============================================================================

def sha256_digest(data: Union[str, bytes, bytearray, memoryview]) -> str:
    """Computes lowercase hex SHA-256 digest over string or raw bytes."""
    if isinstance(data, str):
        payload = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        payload = bytes(data)
    else:
        raise TypeError(f"Expected str or bytes-like object, got {type(data).__name__}")
    return hashlib.sha256(payload).hexdigest()


def hash_tensor(tensor_data: Any) -> str:
    """Computes SHA256 hex digest of raw tensor buffer or numpy array."""
    if hasattr(tensor_data, "tobytes"):
        b = tensor_data.tobytes()
    elif isinstance(tensor_data, (bytes, bytearray, memoryview)):
        b = bytes(tensor_data)
    else:
        raise TypeError("Tensor data must provide tobytes() or be bytes-like")
    return hashlib.sha256(b).hexdigest()


def format_tile_result_payload(task_id: str, result_hash: str) -> bytes:
    """Constructs canonical UTF-8 payload for TileResult signature: f'{task_id}:{result_hash}'."""
    return f"{task_id.strip()}:{result_hash.strip()}".encode("utf-8")


# ============================================================================
# Functional Signing and Verification Interface
# ============================================================================

def sign_message(private_key_hex: str, message: Union[str, bytes]) -> str:
    """
    Computes SHA256(message) and produces a 64-byte compact hex signature (128 hex chars)
    with canonical low-S normalization (BIP-62).
    """
    if not validate_private_key_hex(private_key_hex):
        raise ValueError("Invalid private key hex or out of valid range")

    if isinstance(message, str):
        data_bytes = message.encode("utf-8")
    elif isinstance(message, (bytes, bytearray)):
        data_bytes = bytes(message)
    else:
        raise TypeError("Message to sign must be str or bytes")

    priv_int = int(private_key_hex.strip(), 16)

    if _HAS_CRYPTOGRAPHY:
        priv_key = ec.derive_private_key(priv_int, ec.SECP256K1())
        der_sig = priv_key.sign(data_bytes, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der_sig)
    elif ecdsa is not None:
        digest = hashlib.sha256(data_bytes).digest()
        sk = ecdsa.SigningKey.from_secret_exponent(priv_int, curve=ecdsa.SECP256k1)
        sig_bytes = sk.sign_digest_deterministic(digest, hashfunc=hashlib.sha256)
        r = int.from_bytes(sig_bytes[:32], "big")
        s = int.from_bytes(sig_bytes[32:], "big")
    else:
        raise RuntimeError("No cryptographic backend available")

    # Enforce canonical low-S normalization (BIP-62)
    if s > SECP256K1_HALF_ORDER:
        s = SECP256K1_ORDER - s

    r_bytes = r.to_bytes(32, "big")
    s_bytes = s.to_bytes(32, "big")
    return (r_bytes + s_bytes).hex()


def verify_signature(public_key_hex: str, message: Union[str, bytes], signature_hex: str) -> bool:
    """
    Verifies signature over message against public_key_hex.
    Supports 33-byte compressed, 65-byte uncompressed, and 32-byte X-only Nostr public keys.
    Handles both 64-byte compact hex and DER signatures.
    Returns False safely on any validation or parsing error without raising exceptions.
    """
    if isinstance(message, str):
        data_bytes = message.encode("utf-8")
    elif isinstance(message, (bytes, bytearray)):
        data_bytes = bytes(message)
    else:
        return False

    try:
        sig_str = str(signature_hex).strip()
        sig_bytes = bytes.fromhex(sig_str)
        pub_str = str(public_key_hex).strip()
        pub_bytes = bytes.fromhex(pub_str)
    except Exception:
        return False

    # Candidate public keys
    candidate_pubs = []
    if len(pub_bytes) == 32:
        # Nostr X-only: evaluate both even (02) and odd (03) parity prefixes
        candidate_pubs = [b"\x02" + pub_bytes, b"\x03" + pub_bytes]
    elif len(pub_bytes) in (33, 65):
        candidate_pubs = [pub_bytes]
    else:
        return False

    for cand_pub in candidate_pubs:
        if _verify_single_pubkey(data_bytes, sig_bytes, cand_pub):
            return True

    return False


def _verify_single_pubkey(data_bytes: bytes, sig_bytes: bytes, pub_bytes: bytes) -> bool:
    """Helper to verify against a specific 33-byte or 65-byte encoded curve point."""
    if _HAS_CRYPTOGRAPHY:
        try:
            peer_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), pub_bytes)

            if len(sig_bytes) == 64:
                r = int.from_bytes(sig_bytes[:32], "big")
                s = int.from_bytes(sig_bytes[32:], "big")
                if not (1 <= r < SECP256K1_ORDER and 1 <= s < SECP256K1_ORDER):
                    return False
                s_candidates = [s]
                if s != 0:
                    s_candidates.append(SECP256K1_ORDER - s)
                for s_cand in s_candidates:
                    try:
                        der_bytes = encode_dss_signature(r, s_cand)
                        peer_pub.verify(der_bytes, data_bytes, ec.ECDSA(hashes.SHA256()))
                        return True
                    except Exception:
                        continue
                return False
            else:
                peer_pub.verify(sig_bytes, data_bytes, ec.ECDSA(hashes.SHA256()))
                return True
        except Exception:
            return False
    elif ecdsa is not None:
        try:
            vk = ecdsa.VerifyingKey.from_string(pub_bytes, curve=ecdsa.SECP256k1)
            digest = hashlib.sha256(data_bytes).digest()
            if len(sig_bytes) == 64:
                return vk.verify_digest(sig_bytes, digest)
            else:
                return vk.verify_digest(sig_bytes, digest, sigdecode=ecdsa.util.sigdecode_der)
        except Exception:
            return False
    return False


def verify_tile_result_signature(
    worker_pubkey: str,
    task_id: str,
    result_bytes: bytes,
    signature_hex: str,
) -> bool:
    """Verifies that worker_signature on TileResult is valid over f'{task_id}:{sha256(result_bytes)}'."""
    c_hash = sha256_digest(result_bytes)
    payload = format_tile_result_payload(task_id, c_hash)
    return verify_signature(worker_pubkey, payload, signature_hex)


# ============================================================================
# ClientSigner Class
# ============================================================================

class ClientSigner:
    """
    Consumer / Client Secp256k1 Key Manager and Signer.
    Manages consumer identity, Nostr npub/nsec, task request signing, and result validation.
    """

    def __init__(self, privkey_hex: Optional[str] = None):
        if privkey_hex is None or not str(privkey_hex).strip():
            self._privkey_int = secrets.randbelow(SECP256K1_ORDER - 1) + 1
        else:
            clean = str(privkey_hex).strip().lower()
            if len(clean) != 64:
                raise ValueError(f"Private key hex must be 64 characters (32 bytes), got {len(clean)}")
            try:
                self._privkey_int = int(clean, 16)
            except ValueError:
                raise ValueError("Private key contains invalid hexadecimal characters")
            if not (1 <= self._privkey_int < SECP256K1_ORDER):
                raise ValueError("Private key scalar out of valid range [1, SECP256K1_ORDER - 1]")

        self._privkey_bytes = self._privkey_int.to_bytes(32, "big")
        self._privkey_hex = self._privkey_bytes.hex()
        self._pubkey_hex_compressed = privkey_to_pubkey(self._privkey_hex, compressed=True)
        self._pubkey_hex_uncompressed = privkey_to_pubkey(self._privkey_hex, compressed=False)
        self._xonly_pubkey_hex = pubkey_to_xonly(self._pubkey_hex_compressed)
        self._npub = pubkey_to_npub(self._xonly_pubkey_hex)

    @classmethod
    def generate(cls) -> "ClientSigner":
        """Generates a fresh random ClientSigner instance."""
        return cls()

    def get_privkey_hex(self) -> str:
        """Returns 32-byte hex private key (64 chars)."""
        return self._privkey_hex

    def get_pubkey_hex(self) -> str:
        """Returns 33-byte compressed public key hex (66 chars)."""
        return self._pubkey_hex_compressed

    def get_uncompressed_pubkey_hex(self) -> str:
        """Returns 65-byte uncompressed public key hex (130 chars)."""
        return self._pubkey_hex_uncompressed

    def get_nostr_pubkey(self) -> str:
        """Returns 32-byte X-only Nostr public key hex (64 chars)."""
        return self._xonly_pubkey_hex

    def get_nostr_npub(self) -> str:
        """Returns Bech32 'npub1...' string (63 chars)."""
        return self._npub

    def get_npub(self) -> str:
        """Alias for get_nostr_npub."""
        return self._npub

    def sign_message(self, message: Union[str, bytes]) -> str:
        """Signs message with client's private key, returning 64-byte compact hex signature."""
        return sign_message(self._privkey_hex, message)

    sign = sign_message

    def sign_result(self, task_id: str, result_hash_or_bytes: Union[str, bytes]) -> str:
        """Convenience method signing (task_id, SHA256(result_bytes)) tuple."""
        if isinstance(result_hash_or_bytes, (bytes, bytearray, memoryview)):
            c_hash = sha256_digest(result_hash_or_bytes)
        else:
            c_hash = str(result_hash_or_bytes).strip()
        payload = format_tile_result_payload(task_id, c_hash)
        return self.sign_message(payload)

    def verify_signature(
        self,
        message: Union[str, bytes],
        signature_hex: str,
        pubkey_hex: Optional[str] = None,
    ) -> bool:
        """Verifies signature over message against this client's or specified public key."""
        target_pub = pubkey_hex if pubkey_hex is not None else self._pubkey_hex_compressed
        return verify_signature(target_pub, message, signature_hex)

    @classmethod
    def verify_peer_signature(
        cls,
        data: Union[str, bytes],
        signature_hex: str,
        pubkey_hex: str,
    ) -> bool:
        """Classmethod matching WorkerSigner.verify_peer_signature."""
        return verify_signature(pubkey_hex, data, signature_hex)

    def nip04_encrypt(self, recipient_pubkey: Union[str, bytes], plaintext: Union[str, bytes]) -> str:
        """Encrypts message for recipient public key using NIP-04 ECDH AES-256-CBC."""
        from .nip04 import nip04_encrypt
        return nip04_encrypt(self._privkey_hex, recipient_pubkey, plaintext)

    def nip04_decrypt(self, sender_pubkey: Union[str, bytes], payload: str) -> str:
        """Decrypts NIP-04 payload from sender public key."""
        from .nip04 import nip04_decrypt
        return nip04_decrypt(self._privkey_hex, sender_pubkey, payload)

    def compute_nip04_shared_secret(self, peer_pubkey: Union[str, bytes]) -> bytes:
        """Computes 32-byte Secp256k1 ECDH unhashed shared secret with peer."""
        from .nip04 import compute_nip04_shared_secret
        return compute_nip04_shared_secret(self._privkey_hex, peer_pubkey)
