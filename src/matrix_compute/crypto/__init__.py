"""Cryptographic utilities, key management, and ECDSA signing for matrix-compute-sdk."""

from .keys import (
    SECP256K1_HALF_ORDER,
    SECP256K1_ORDER,
    generate_keypair,
    npub_to_pubkey_hex,
    nsec_to_privkey_hex,
    privkey_to_nsec,
    privkey_to_pubkey,
    pubkey_to_npub,
    pubkey_to_xonly,
    validate_private_key_hex,
    validate_private_key_int,
    validate_public_key_hex,
)
from .signer import (
    ClientSigner,
    format_tile_result_payload,
    hash_tensor,
    sha256_digest,
    sign_message,
    verify_signature,
    verify_tile_result_signature,
)

__all__ = [
    # Key operations
    "generate_keypair",
    "privkey_to_pubkey",
    "pubkey_to_xonly",
    "pubkey_to_npub",
    "npub_to_pubkey_hex",
    "privkey_to_nsec",
    "nsec_to_privkey_hex",
    "validate_private_key_hex",
    "validate_private_key_int",
    "validate_public_key_hex",
    # Signer & Verification
    "ClientSigner",
    "sign_message",
    "verify_signature",
    "verify_tile_result_signature",
    "format_tile_result_payload",
    "sha256_digest",
    "hash_tensor",
    # Constants
    "SECP256K1_ORDER",
    "SECP256K1_HALF_ORDER",
]
