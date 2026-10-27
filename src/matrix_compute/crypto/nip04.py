"""
NIP-04: Secp256k1 ECDH AES-256-CBC Encrypted Direct Messages (Milestone M2.2).

Implements:
1. Symmetrical Secp256k1 ECDH key exchange deriving unhashed 32-byte X-coordinate AES keys.
2. AES-256-CBC encryption with fresh 16-byte random IVs and PKCS#7 padding.
3. Wire framing: <base64_ciphertext>?iv=<base64_iv>.
4. Robust parsing and decryption with strict DecryptionError validation.
5. Class Nip04Cipher and functional interfaces.
"""

from __future__ import annotations

import base64
import os
import secrets
from typing import Any, Optional, Tuple, Union

from .keys import (
    SECP256K1_ORDER,
    npub_to_pubkey_hex,
    nsec_to_privkey_hex,
    pubkey_to_xonly,
    validate_private_key_hex,
    validate_public_key_hex,
)
from .session import DecryptionError, EncryptionError, KeyExchangeError

# Cryptographic backend resolution
try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding, serialization
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False
    try:
        import ecdsa
    except ImportError:
        ecdsa = None

# ==============================================================================
# CONSTANTS
# ==============================================================================

AES_BLOCK_SIZE: int = 16
IV_BYTES: int = 16
AES_KEY_BYTES: int = 32
NIP04_DELIMITER: str = "?iv="


# ==============================================================================
# KEY NORMALIZATION HELPERS
# ==============================================================================

def _normalize_private_key_to_int(privkey: Any) -> int:
    """Normalizes private key from hex, nsec, int, bytes, or ClientSigner to valid integer scalar."""
    if hasattr(privkey, "get_privkey_hex"):
        # ClientSigner or similar object
        priv_hex = str(privkey.get_privkey_hex()).strip()
        return _hex_to_priv_int(priv_hex)
    elif isinstance(privkey, str):
        clean = privkey.strip()
        if clean.lower().startswith("nsec1"):
            clean = nsec_to_privkey_hex(clean)
        return _hex_to_priv_int(clean)
    elif isinstance(privkey, (bytes, bytearray)):
        if len(privkey) != 32:
            raise ValueError(f"Private key bytes must be exactly 32 bytes, got {len(privkey)}")
        priv_int = int.from_bytes(privkey, "big")
        if not (1 <= priv_int < SECP256K1_ORDER):
            raise ValueError("Private key scalar out of valid range [1, SECP256K1_ORDER - 1]")
        return priv_int
    elif isinstance(privkey, int):
        if not (1 <= privkey < SECP256K1_ORDER):
            raise ValueError(f"Private key integer out of valid range [1, SECP256K1_ORDER - 1]: {privkey}")
        return privkey
    raise TypeError(f"Unsupported private key type: {type(privkey).__name__}")


def _hex_to_priv_int(priv_hex: str) -> int:
    clean = priv_hex.lower()
    if len(clean) != 64:
        raise ValueError(f"Private key hex must be 64 characters (32 bytes), got {len(clean)}")
    try:
        val = int(clean, 16)
    except ValueError as exc:
        raise ValueError(f"Private key contains invalid hexadecimal characters: {exc}") from exc
    if not (1 <= val < SECP256K1_ORDER):
        raise ValueError("Private key scalar out of valid range [1, SECP256K1_ORDER - 1]")
    return val


def _normalize_public_key_point_bytes(pubkey: Any) -> bytes:
    """
    Normalizes public key from X-only hex, npub, compressed/uncompressed hex, or bytes
    into a valid 33-byte or 65-byte Secp256k1 encoded point on the curve.
    """
    if hasattr(pubkey, "public_bytes"):
        # ec.EllipticCurvePublicKey
        return pubkey.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.CompressedPoint,
        )

    if isinstance(pubkey, str):
        clean = pubkey.strip()
        if clean.lower().startswith("npub1"):
            clean = npub_to_pubkey_hex(clean)
        try:
            raw_bytes = bytes.fromhex(clean)
        except ValueError as exc:
            raise KeyExchangeError(f"Public key contains invalid hex: {exc}", cause=exc) from exc
    elif isinstance(pubkey, (bytes, bytearray)):
        raw_bytes = bytes(pubkey)
    else:
        raise TypeError(f"Unsupported public key type: {type(pubkey).__name__}")

    # Standardize point format
    if len(raw_bytes) == 32:
        # BIP-340 X-only: prepend 0x02 even Y parity
        point_bytes = b"\x02" + raw_bytes
    elif len(raw_bytes) == 33:
        if raw_bytes[0] not in (2, 3):
            raise KeyExchangeError(f"Invalid compressed public key prefix: 0x{raw_bytes[0]:02x} (must be 0x02 or 0x03)")
        point_bytes = raw_bytes
    elif len(raw_bytes) == 65:
        if raw_bytes[0] != 4:
            raise KeyExchangeError(f"Invalid uncompressed public key prefix: 0x{raw_bytes[0]:02x} (must be 0x04)")
        point_bytes = raw_bytes
    else:
        raise KeyExchangeError(f"Invalid public key byte length: {len(raw_bytes)} (expected 32, 33, or 65 bytes)")

    # Validate point lies on curve
    if _HAS_CRYPTOGRAPHY:
        try:
            ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), point_bytes)
        except ValueError as exc:
            raise KeyExchangeError(f"Public key point does not lie on secp256k1 curve: {exc}", cause=exc) from exc
    elif ecdsa is not None:
        try:
            ecdsa.VerifyingKey.from_string(point_bytes, curve=ecdsa.SECP256k1)
        except Exception as exc:
            raise KeyExchangeError(f"Public key point does not lie on secp256k1 curve: {exc}", cause=exc) from exc

    return point_bytes


# ==============================================================================
# ECDH SHARED SECRET DERIVATION
# ==============================================================================

def compute_nip04_shared_secret(private_key: Any, public_key: Any) -> bytes:
    """
    Computes Secp256k1 ECDH shared secret point S = d * P and extracts the raw
    32-byte big-endian X-coordinate without hashing, per NIP-04 specification.
    """
    priv_int = _normalize_private_key_to_int(private_key)
    pub_point_bytes = _normalize_public_key_point_bytes(public_key)

    if _HAS_CRYPTOGRAPHY:
        try:
            priv_ec = ec.derive_private_key(priv_int, ec.SECP256K1())
            pub_ec = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), pub_point_bytes)
            shared_secret = priv_ec.exchange(ec.ECDH(), pub_ec)
            if len(shared_secret) != AES_KEY_BYTES:
                raise KeyExchangeError(f"Unexpected shared secret length: {len(shared_secret)} bytes")
            return shared_secret
        except KeyExchangeError:
            raise
        except Exception as exc:
            raise KeyExchangeError(f"Secp256k1 ECDH exchange failed: {exc}", cause=exc) from exc
    elif ecdsa is not None:
        try:
            sk = ecdsa.SigningKey.from_secret_exponent(priv_int, curve=ecdsa.SECP256k1)
            vk = ecdsa.VerifyingKey.from_string(pub_point_bytes, curve=ecdsa.SECP256k1)
            shared_point = vk.pubkey.point * sk.privkey.secret_multiplier
            return int(shared_point.x()).to_bytes(AES_KEY_BYTES, "big")
        except Exception as exc:
            raise KeyExchangeError(f"Secp256k1 ECDH exchange failed: {exc}", cause=exc) from exc
    else:
        raise RuntimeError("No cryptographic backend available")


# ==============================================================================
# NIP-04 ENCRYPTION & DECRYPTION
# ==============================================================================

def nip04_encrypt(
    private_key: Any,
    public_key: Any,
    plaintext: Union[str, bytes],
    iv: Optional[bytes] = None,
) -> str:
    """
    Encrypts plaintext for recipient using NIP-04 specification:
    1. Computes Secp256k1 ECDH shared secret X-coordinate as 32-byte AES key.
    2. Generates fresh random 16-byte IV (or uses explicit iv if provided).
    3. Pads plaintext with PKCS#7 to 16-byte blocks.
    4. Encrypts using AES-256-CBC.
    5. Returns '<ciphertext_b64>?iv=<iv_b64>'.
    """
    shared_key = compute_nip04_shared_secret(private_key, public_key)

    if isinstance(plaintext, str):
        data = plaintext.encode("utf-8")
    elif isinstance(plaintext, (bytes, bytearray)):
        data = bytes(plaintext)
    else:
        raise TypeError(f"Plaintext must be str or bytes, got {type(plaintext).__name__}")

    if iv is None:
        iv_bytes = secrets.token_bytes(IV_BYTES)
    else:
        iv_bytes = bytes(iv)
        if len(iv_bytes) != IV_BYTES:
            raise ValueError(f"IV must be exactly {IV_BYTES} bytes, got {len(iv_bytes)}")

    if not _HAS_CRYPTOGRAPHY:
        raise RuntimeError("cryptography package is required for AES-256-CBC encryption")

    try:
        # PKCS#7 Pad
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()

        # AES-256-CBC Encrypt
        cipher = Cipher(algorithms.AES(shared_key), modes.CBC(iv_bytes))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        ct_b64 = base64.b64encode(ciphertext).decode("ascii")
        iv_b64 = base64.b64encode(iv_bytes).decode("ascii")
        return f"{ct_b64}{NIP04_DELIMITER}{iv_b64}"
    except Exception as exc:
        raise EncryptionError(f"NIP-04 encryption failed: {exc}", cause=exc) from exc


def nip04_decrypt(
    private_key: Any,
    public_key: Any,
    payload: str,
) -> str:
    """
    Decrypts NIP-04 payload and returns UTF-8 string.
    Raises DecryptionError on malformed format, invalid IV, corrupted ciphertext,
    wrong key, or invalid UTF-8 string.
    """
    raw_bytes = nip04_decrypt_raw(private_key, public_key, payload)
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DecryptionError(f"Decrypted plaintext is not valid UTF-8: {exc}", cause=exc) from exc


def nip04_decrypt_raw(
    private_key: Any,
    public_key: Any,
    payload: str,
) -> bytes:
    """
    Decrypts NIP-04 payload and returns unpadded raw bytes.
    Raises DecryptionError on any cryptographic, padding, framing, or format error.
    """
    if not isinstance(payload, str):
        raise DecryptionError(f"Payload must be string, got {type(payload).__name__}")

    if NIP04_DELIMITER not in payload:
        raise DecryptionError("Invalid NIP-04 payload format: missing '?iv=' delimiter")

    parts = payload.split(NIP04_DELIMITER)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise DecryptionError("Invalid NIP-04 payload format: expected non-empty <ciphertext_b64>?iv=<iv_b64>")

    ct_b64 = parts[0].strip()
    iv_b64 = parts[1].strip()

    try:
        ciphertext = base64.b64decode(ct_b64, validate=True)
        iv = base64.b64decode(iv_b64, validate=True)
    except Exception as exc:
        raise DecryptionError(f"Invalid Base64 encoding in NIP-04 payload: {exc}", cause=exc) from exc

    if len(iv) != IV_BYTES:
        raise DecryptionError(f"Invalid IV length: expected {IV_BYTES} bytes, got {len(iv)}")

    if len(ciphertext) == 0 or len(ciphertext) % AES_BLOCK_SIZE != 0:
        raise DecryptionError(
            f"Invalid ciphertext length: must be a positive multiple of {AES_BLOCK_SIZE} bytes, got {len(ciphertext)}"
        )

    shared_key = compute_nip04_shared_secret(private_key, public_key)

    if not _HAS_CRYPTOGRAPHY:
        raise RuntimeError("cryptography package is required for AES-256-CBC decryption")

    try:
        cipher = Cipher(algorithms.AES(shared_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    except Exception as exc:
        raise DecryptionError(f"AES-256-CBC decryption failed: {exc}", cause=exc) from exc

    try:
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_data) + unpadder.finalize()
        return plaintext
    except (ValueError, Exception) as exc:
        raise DecryptionError(
            "NIP-04 decryption failed: invalid PKCS#7 padding (ciphertext corrupted or incorrect key)",
            cause=exc,
        ) from exc


# ==============================================================================
# Nip04Cipher CLASS
# ==============================================================================

class Nip04Cipher:
    """Encapsulates a client private key for stateful NIP-04 message exchange."""

    def __init__(self, private_key: Any) -> None:
        self._privkey_int = _normalize_private_key_to_int(private_key)
        self._privkey_hex = hex(self._privkey_int)[2:].zfill(64)

    def shared_secret(self, peer_pubkey: Any) -> bytes:
        """Computes 32-byte shared secret with peer."""
        return compute_nip04_shared_secret(self._privkey_int, peer_pubkey)

    def encrypt(self, recipient_pubkey: Any, plaintext: Union[str, bytes], iv: Optional[bytes] = None) -> str:
        """Encrypts plaintext for recipient."""
        return nip04_encrypt(self._privkey_int, recipient_pubkey, plaintext, iv=iv)

    def decrypt(self, sender_pubkey: Any, payload: str) -> str:
        """Decrypts NIP-04 payload from sender to UTF-8 string."""
        return nip04_decrypt(self._privkey_int, sender_pubkey, payload)

    def decrypt_raw(self, sender_pubkey: Any, payload: str) -> bytes:
        """Decrypts NIP-04 payload from sender to raw bytes."""
        return nip04_decrypt_raw(self._privkey_int, sender_pubkey, payload)
