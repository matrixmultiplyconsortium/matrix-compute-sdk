"""Secp256k1 Key Management and Nostr Key Encodings."""

from __future__ import annotations

import secrets
from typing import List, Optional, Tuple, Union

# BIP-173 Bech32 Constants
BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP256K1_HALF_ORDER = SECP256K1_ORDER // 2
SECP256K1_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP256K1_GX_HEX = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
SECP256K1_GY_HEX = "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"

# Cryptographic backend resolution
try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False
    try:
        import ecdsa
    except ImportError:
        ecdsa = None


# ============================================================================
# Pure-Python BIP-173 Bech32 Implementation (Encode & Decode)
# ============================================================================

def _bech32_polymod(values: List[int]) -> int:
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ value
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str) -> List[int]:
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _convertbits(data: Union[bytes, List[int]], frombits: int, tobits: int, pad: bool = True) -> Optional[List[int]]:
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    max_acc = (1 << (frombits + tobits - 1)) - 1
    for value in data:
        if value < 0 or (value >> frombits):
            return None
        acc = ((acc << frombits) | value) & max_acc
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret


def bech32_encode(hrp: str, data_bytes: bytes) -> str:
    """Encodes arbitrary bytes into BIP-173 Bech32 format with given human-readable part."""
    data5 = _convertbits(data_bytes, 8, 5, True)
    if data5 is None:
        raise ValueError("Failed to convert bits for Bech32 encoding")
    values = _bech32_hrp_expand(hrp) + data5
    polymod = _bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp.lower() + "1" + "".join(BECH32_CHARSET[d] for d in (data5 + checksum))


def bech32_decode(bech_str: str) -> Tuple[Optional[str], Optional[bytes]]:
    """
    Decodes a BIP-173 Bech32 string.
    Returns (hrp, data_bytes) or (None, None) on checksum/format failure.
    """
    if not isinstance(bech_str, str):
        return None, None
    if any(ord(x) < 33 or ord(x) > 126 for x in bech_str):
        return None, None
    if bech_str.lower() != bech_str and bech_str.upper() != bech_str:
        return None, None  # Mixed case rejected per BIP-173

    clean = bech_str.lower()
    pos = clean.rfind("1")
    if pos < 1 or pos + 7 > len(clean):
        return None, None

    hrp = clean[:pos]
    data5 = []
    for c in clean[pos + 1:]:
        d = BECH32_CHARSET.find(c)
        if d == -1:
            return None, None
        data5.append(d)

    values = _bech32_hrp_expand(hrp) + data5
    if _bech32_polymod(values) != 1:
        return None, None

    payload5 = data5[:-6]
    data_bytes_list = _convertbits(payload5, 5, 8, pad=False)
    if data_bytes_list is None:
        return None, None

    return hrp, bytes(data_bytes_list)


# ============================================================================
# Key Validation and Conversion Functions
# ============================================================================

def validate_private_key_int(priv_int: int) -> bool:
    """Returns True if integer is a valid Secp256k1 private key scalar [1, ORDER - 1]."""
    return 1 <= priv_int < SECP256K1_ORDER


def validate_private_key_hex(priv_hex: str) -> bool:
    """Validates private key hex string format and scalar range."""
    if not isinstance(priv_hex, str):
        return False
    clean = priv_hex.strip().lower()
    if len(clean) != 64:
        return False
    try:
        val = int(clean, 16)
        return validate_private_key_int(val)
    except ValueError:
        return False


def validate_public_key_hex(pub_hex: str) -> bool:
    """Validates public key hex format (32-byte X-only, 33-byte compressed, or 65-byte uncompressed)."""
    if not isinstance(pub_hex, str):
        return False
    clean = pub_hex.strip().lower()
    try:
        b = bytes.fromhex(clean)
    except Exception:
        return False

    if len(b) == 32:
        return True
    elif len(b) == 33:
        return b[0] in (2, 3)
    elif len(b) == 65:
        return b[0] == 4
    return False


def generate_keypair() -> Tuple[str, str]:
    """
    Generates a cryptographically secure Secp256k1 keypair.
    Returns:
        (private_key_hex, public_key_hex)
        - private_key_hex: 32-byte hex string (64 characters)
        - public_key_hex: 33-byte compressed public key hex string (66 characters)
    """
    priv_int = secrets.randbelow(SECP256K1_ORDER - 1) + 1
    priv_bytes = priv_int.to_bytes(32, "big")

    if _HAS_CRYPTOGRAPHY:
        priv_key = ec.derive_private_key(priv_int, ec.SECP256K1())
        pub_bytes = priv_key.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.CompressedPoint,
        )
    elif ecdsa is not None:
        sk = ecdsa.SigningKey.from_secret_exponent(priv_int, curve=ecdsa.SECP256k1)
        pub_bytes = sk.verifying_key.to_string("compressed")
    else:
        raise RuntimeError("No cryptographic backend available")

    return priv_bytes.hex(), pub_bytes.hex()


def privkey_to_pubkey(privkey_hex: str, compressed: bool = True) -> str:
    """Derives compressed (33-byte) or uncompressed (65-byte) public key hex from private key."""
    if not validate_private_key_hex(privkey_hex):
        raise ValueError("Invalid private key hex or out of valid range")
    priv_int = int(privkey_hex.strip(), 16)

    if _HAS_CRYPTOGRAPHY:
        priv_key = ec.derive_private_key(priv_int, ec.SECP256K1())
        fmt = (
            serialization.PublicFormat.CompressedPoint
            if compressed
            else serialization.PublicFormat.UncompressedPoint
        )
        return priv_key.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=fmt,
        ).hex()
    elif ecdsa is not None:
        sk = ecdsa.SigningKey.from_secret_exponent(priv_int, curve=ecdsa.SECP256k1)
        fmt_str = "compressed" if compressed else "uncompressed"
        return sk.verifying_key.to_string(fmt_str).hex()
    raise RuntimeError("No cryptographic backend available")


def pubkey_to_xonly(pubkey_hex: str) -> str:
    """Converts 33-byte compressed or 65-byte uncompressed public key hex to 32-byte X-only Nostr hex."""
    clean = pubkey_hex.strip().lower()
    try:
        b = bytes.fromhex(clean)
    except Exception:
        raise ValueError("Public key is not a valid hex string")

    if len(b) == 32:
        return clean
    elif len(b) in (33, 65):
        return b[1:33].hex()
    raise ValueError(f"Invalid public key byte length: {len(b)}")


def pubkey_to_npub(pubkey_hex: str) -> str:
    """Converts public key hex (32-byte, 33-byte, or 65-byte) to Bech32 'npub1...' string."""
    xonly_hex = pubkey_to_xonly(pubkey_hex)
    return bech32_encode("npub", bytes.fromhex(xonly_hex))


def npub_to_pubkey_hex(npub: str) -> str:
    """Decodes Bech32 'npub1...' string to 32-byte X-only public key hex (64 hex chars)."""
    hrp, data = bech32_decode(npub)
    if hrp != "npub" or data is None or len(data) != 32:
        raise ValueError("Invalid npub string or checksum failure")
    return data.hex()


def privkey_to_nsec(privkey_hex: str) -> str:
    """Converts 32-byte private key hex to Bech32 'nsec1...' string."""
    if not validate_private_key_hex(privkey_hex):
        raise ValueError("Invalid private key hex")
    return bech32_encode("nsec", bytes.fromhex(privkey_hex.strip().lower()))


def nsec_to_privkey_hex(nsec: str) -> str:
    """Decodes Bech32 'nsec1...' string to 32-byte private key hex (64 hex chars)."""
    hrp, data = bech32_decode(nsec)
    if hrp != "nsec" or data is None or len(data) != 32:
        raise ValueError("Invalid nsec string or checksum failure")
    hex_str = data.hex()
    if not validate_private_key_hex(hex_str):
        raise ValueError("Decoded nsec scalar out of valid curve range")
    return hex_str
