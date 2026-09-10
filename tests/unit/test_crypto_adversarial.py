"""
Adversarial Stress Test Suite for matrix-compute-sdk Cryptographic Engine.

Challenge Objectives:
1. Exhaustively stress-test Secp256k1 key generation, scalar boundaries (0, 1, n-1, n, n+1),
   and invalid hex inputs.
2. Verify canonical low-S normalization under random message batches: confirm that NO generated
   signature has s > floor(n / 2).
3. Test bit-flip sweeps: inject single-bit corruptions across compact 64-byte signatures and
   verify that verify_signature rejects 100% of mutated signatures.
4. Test malleability resilience: verify that verify_signature accepts signatures whether
   normalized or inverted, while sign_message only generates low-S.
5. Test multi-key format verification: compressed (33 bytes), uncompressed (65 bytes),
   and X-only Nostr (32 bytes) with both even and odd parity curve points.
6. Stress test BIP-173 Bech32 codec against independent oracle and single-character corruptions.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from typing import Optional, Tuple

import pytest

from matrix_compute.crypto.keys import (
    BECH32_CHARSET,
    SECP256K1_GX_HEX,
    SECP256K1_GY_HEX,
    SECP256K1_HALF_ORDER,
    SECP256K1_ORDER,
    SECP256K1_P,
    bech32_decode,
    bech32_encode,
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
from matrix_compute.crypto.signer import (
    ClientSigner,
    format_tile_result_payload,
    hash_tensor,
    sha256_digest,
    sign_message,
    verify_signature,
    verify_tile_result_signature,
)


# ============================================================================
# Independent Reference BIP-173 Decoder Oracle
# ============================================================================

def _oracle_polymod(values: list[int]) -> int:
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ value
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _oracle_hrp_expand(hrp: str) -> list[int]:
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _oracle_verify_checksum(hrp: str, data: list[int]) -> bool:
    return _oracle_polymod(_oracle_hrp_expand(hrp) + data) == 1


def bech32_decode_oracle(bech_str: str) -> Tuple[Optional[str], Optional[bytes]]:
    """Independent BIP-173 decoder oracle for adversarial validation."""
    if not isinstance(bech_str, str):
        return None, None
    if any(ord(x) < 33 or ord(x) > 126 for x in bech_str):
        return None, None
    if bech_str.lower() != bech_str and bech_str.upper() != bech_str:
        return None, None

    bech_clean = bech_str.lower()
    pos = bech_clean.rfind("1")
    if pos < 1 or pos + 7 > len(bech_clean):
        return None, None

    hrp = bech_clean[:pos]
    data5: list[int] = []
    for c in bech_clean[pos + 1:]:
        d = BECH32_CHARSET.find(c)
        if d == -1:
            return None, None
        data5.append(d)

    if not _oracle_verify_checksum(hrp, data5):
        return None, None

    payload5 = data5[:-6]
    acc = 0
    bits = 0
    out_bytes: list[int] = []
    max_acc = (1 << (5 + 8 - 1)) - 1
    for val in payload5:
        acc = ((acc << 5) | val) & max_acc
        bits += 5
        while bits >= 8:
            bits -= 8
            out_bytes.append((acc >> bits) & 0xFF)

    if bits >= 5 or ((acc << (8 - bits)) & 0xFF) != 0:
        return None, None

    return hrp, bytes(out_bytes)


# ============================================================================
# 1. Secp256k1 Scalar Boundaries & Key Generation Stress Tests
# ============================================================================

class TestSecp256k1ScalarBoundaries:
    """Stress tests integer and hex scalar validation at exact boundary values."""

    def test_integer_scalar_boundaries(self):
        """Scalar range must be strictly [1, n - 1]."""
        # Lower boundaries
        assert validate_private_key_int(-100) is False
        assert validate_private_key_int(-1) is False
        assert validate_private_key_int(0) is False
        assert validate_private_key_int(1) is True
        assert validate_private_key_int(2) is True

        # Upper boundaries
        assert validate_private_key_int(SECP256K1_ORDER - 2) is True
        assert validate_private_key_int(SECP256K1_ORDER - 1) is True
        assert validate_private_key_int(SECP256K1_ORDER) is False
        assert validate_private_key_int(SECP256K1_ORDER + 1) is False
        assert validate_private_key_int((1 << 256) - 1) is False
        assert validate_private_key_int(1 << 256) is False

    def test_hex_scalar_boundaries(self):
        """Hex strings representing boundary scalars must be validated accurately."""
        # Scalar 0
        assert validate_private_key_hex("00" * 32) is False

        # Scalar 1 (Generator G)
        assert validate_private_key_hex("01".zfill(64)) is True

        # Scalar n - 1 (-G)
        hex_n_minus_1 = hex(SECP256K1_ORDER - 1)[2:].zfill(64)
        assert validate_private_key_hex(hex_n_minus_1) is True

        # Scalar n (Curve order)
        hex_n = hex(SECP256K1_ORDER)[2:].zfill(64)
        assert validate_private_key_hex(hex_n) is False

        # Scalar n + 1
        hex_n_plus_1 = hex(SECP256K1_ORDER + 1)[2:].zfill(64)
        assert validate_private_key_hex(hex_n_plus_1) is False

        # Maximum 256-bit value (2^256 - 1)
        assert validate_private_key_hex("ff" * 32) is False

    def test_client_signer_scalar_boundary_exceptions(self):
        """ClientSigner constructor must reject invalid scalar bounds with ValueError."""
        # Scalar 0
        with pytest.raises(ValueError, match="range"):
            ClientSigner("00" * 32)

        # Scalar n
        hex_n = hex(SECP256K1_ORDER)[2:].zfill(64)
        with pytest.raises(ValueError, match="range"):
            ClientSigner(hex_n)

        # Scalar n + 1
        hex_n_plus_1 = hex(SECP256K1_ORDER + 1)[2:].zfill(64)
        with pytest.raises(ValueError, match="range"):
            ClientSigner(hex_n_plus_1)

        # 2^256 - 1
        with pytest.raises(ValueError, match="range"):
            ClientSigner("ff" * 32)

        # Scalar 1 and n-1 must succeed
        signer_1 = ClientSigner("01".zfill(64))
        assert signer_1.get_privkey_hex() == "01".zfill(64)

        hex_n_minus_1 = hex(SECP256K1_ORDER - 1)[2:].zfill(64)
        signer_n_minus_1 = ClientSigner(hex_n_minus_1)
        assert signer_n_minus_1.get_privkey_hex() == hex_n_minus_1

    def test_malformed_hex_inputs(self):
        """Invalid hex strings (lengths, characters, prefixes) must be rejected."""
        invalid_hex_cases = [
            "0" * 63,                 # 31.5 bytes (odd length)
            "0" * 65,                 # 32.5 bytes (odd length)
            "11" * 31,                # 31 bytes
            "11" * 33,                # 33 bytes
            "0x" + "11" * 32,         # Prefixed with 0x (66 chars)
            "gg" * 32,                # Non-hex characters
            "11" * 31 + "ZZ",         # Trailing non-hex
            "11" * 16 + "  " + "11" * 15,  # Embedded spaces
        ]
        for bad_input in invalid_hex_cases:
            assert validate_private_key_hex(bad_input) is False
            with pytest.raises(ValueError):
                ClientSigner(bad_input)

        # Empty, None, whitespace, and non-string types must fail validate_private_key_hex
        assert validate_private_key_hex("") is False
        assert validate_private_key_hex("   ") is False
        assert validate_private_key_hex(None) is False
        assert validate_private_key_hex(12345) is False
        assert validate_private_key_hex(b"11" * 32) is False

        # ClientSigner with None or empty string acts as default key generator
        signer_none = ClientSigner(None)
        assert len(signer_none.get_privkey_hex()) == 64
        assert validate_private_key_hex(signer_none.get_privkey_hex()) is True

        signer_empty = ClientSigner("")
        assert len(signer_empty.get_privkey_hex()) == 64
        assert validate_private_key_hex(signer_empty.get_privkey_hex()) is True

    def test_privkey_to_pubkey_boundary_exceptions(self):
        """privkey_to_pubkey must reject out-of-bounds scalars."""
        with pytest.raises(ValueError):
            privkey_to_pubkey("00" * 32)

        hex_n = hex(SECP256K1_ORDER)[2:].zfill(64)
        with pytest.raises(ValueError):
            privkey_to_pubkey(hex_n)

        with pytest.raises(ValueError):
            privkey_to_pubkey("ff" * 32)


# ============================================================================
# 2. Canonical Low-S Normalization Battery
# ============================================================================

class TestCanonicalLowSNormalization:
    """Stress tests that 100% of generated signatures strictly satisfy s <= floor(n/2)."""

    def test_random_batch_low_s_invariant(self):
        """Generates 100 signatures across random keys and payloads; asserts 0 high-S occurrences."""
        signer = ClientSigner()
        high_s_count = 0
        total_tests = 100

        for i in range(total_tests):
            # Random payload of variable size
            payload = secrets.token_bytes(secrets.randbelow(128) + 1)
            sig_hex = signer.sign_message(payload)
            sig_bytes = bytes.fromhex(sig_hex)
            assert len(sig_bytes) == 64

            r = int.from_bytes(sig_bytes[:32], "big")
            s = int.from_bytes(sig_bytes[32:], "big")

            assert 1 <= r < SECP256K1_ORDER, f"Invalid r scalar at iter {i}: {r}"
            assert 1 <= s < SECP256K1_ORDER, f"Invalid s scalar at iter {i}: {s}"

            if s > SECP256K1_HALF_ORDER:
                high_s_count += 1

            # Verify that signature verifies
            assert signer.verify_signature(payload, sig_hex) is True

        assert high_s_count == 0, f"Detected {high_s_count}/{total_tests} non-low-S signatures!"

    def test_multi_signer_low_s_invariant(self):
        """Generates signatures across 20 distinct random signers."""
        for _ in range(20):
            s = ClientSigner()
            msg = b"independent signer low-s test"
            sig = s.sign_message(msg)
            s_val = int.from_bytes(bytes.fromhex(sig)[32:], "big")
            assert s_val <= SECP256K1_HALF_ORDER


# ============================================================================
# 3. Bit-Flip Sweeps Across 64-Byte Compact Signatures
# ============================================================================

class TestBitFlipSweeps:
    """Adversarial corruption tests verifying 100% rejection across all bit flips."""

    def test_exhaustive_64_byte_signature_bit_flip_sweep(self):
        """
        Exhaustively inverts every single bit in a 64-byte compact signature (512 total mutations).
        verify_signature must reject 100% (512/512) of corrupted signatures.
        """
        signer = ClientSigner()
        message = b"adversarial bit flip benchmark payload"
        valid_sig_hex = signer.sign_message(message)
        valid_sig_bytes = bytes.fromhex(valid_sig_hex)
        assert len(valid_sig_bytes) == 64

        pubkey = signer.get_pubkey_hex()
        rejected_count = 0
        total_mutations = 64 * 8  # 512 single-bit flips

        for byte_idx in range(64):
            for bit_idx in range(8):
                mutated = bytearray(valid_sig_bytes)
                mutated[byte_idx] ^= (1 << bit_idx)

                # Verification MUST return False
                res = verify_signature(pubkey, message, bytes(mutated).hex())
                if res is False:
                    rejected_count += 1

        assert rejected_count == total_mutations == 512, (
            f"Bit flip sweep failed: only {rejected_count}/{total_mutations} were rejected!"
        )

    def test_payload_bit_flip_sweep(self):
        """Every single-bit corruption in the message payload must cause verification failure."""
        signer = ClientSigner()
        message = bytearray(b"0123456789abcdef0123456789abcdef")  # 32 bytes
        sig_hex = signer.sign_message(bytes(message))
        pubkey = signer.get_pubkey_hex()

        rejected = 0
        total = len(message) * 8
        for byte_idx in range(len(message)):
            for bit_idx in range(8):
                mutated = bytearray(message)
                mutated[byte_idx] ^= (1 << bit_idx)
                if verify_signature(pubkey, bytes(mutated), sig_hex) is False:
                    rejected += 1

        assert rejected == total == 256

    def test_signature_truncation_and_extension_sweep(self):
        """Signatures with byte lengths != 64 (unless valid DER) must return False."""
        signer = ClientSigner()
        message = b"length variation payload"
        valid_sig = bytes.fromhex(signer.sign_message(message))
        pubkey = signer.get_pubkey_hex()

        aberrant_lengths = [0, 1, 2, 16, 31, 32, 33, 48, 63, 65, 66, 70, 80, 100, 128]
        for length in aberrant_lengths:
            if length < len(valid_sig):
                test_sig = valid_sig[:length]
            else:
                test_sig = valid_sig + b"\x00" * (length - len(valid_sig))

            assert verify_signature(pubkey, message, test_sig.hex()) is False


# ============================================================================
# 4. Malleability Resilience (Normalized vs Inverted Signatures)
# ============================================================================

class TestMalleabilityResilience:
    """Verifies that verify_signature accepts both low-S and high-S, while sign_message only emits low-S."""

    def test_inverted_high_s_signature_acceptance(self):
        """
        Given authentic (r, s), the inverted signature (r, n - s) is a mathematically valid
        ECDSA point. verify_signature must accept it for resilience, while sign_message never emits it.
        """
        signer = ClientSigner()
        msg = b"malleability resilience test payload"
        canonical_sig_hex = signer.sign_message(msg)
        canonical_sig_bytes = bytes.fromhex(canonical_sig_hex)

        r = int.from_bytes(canonical_sig_bytes[:32], "big")
        s = int.from_bytes(canonical_sig_bytes[32:], "big")

        # Canonical signature is low-S
        assert s <= SECP256K1_HALF_ORDER

        # Create inverted high-S signature: (r, n - s)
        s_high = SECP256K1_ORDER - s
        assert s_high > SECP256K1_HALF_ORDER

        inverted_sig_bytes = r.to_bytes(32, "big") + s_high.to_bytes(32, "big")
        inverted_sig_hex = inverted_sig_bytes.hex()

        # Both compressed and X-only public keys must accept the inverted signature
        assert verify_signature(signer.get_pubkey_hex(), msg, inverted_sig_hex) is True
        assert verify_signature(signer.get_nostr_pubkey(), msg, inverted_sig_hex) is True
        assert verify_signature(signer.get_uncompressed_pubkey_hex(), msg, inverted_sig_hex) is True

    def test_degenerate_scalars_rejected(self):
        """Signatures with r=0, s=0, r=n, or s=n must be rejected."""
        signer = ClientSigner()
        msg = b"degenerate scalar test"
        valid_sig = bytes.fromhex(signer.sign_message(msg))
        r = int.from_bytes(valid_sig[:32], "big")
        s = int.from_bytes(valid_sig[32:], "big")
        pub = signer.get_pubkey_hex()

        zero_r = (0).to_bytes(32, "big") + s.to_bytes(32, "big")
        zero_s = r.to_bytes(32, "big") + (0).to_bytes(32, "big")
        order_r = SECP256K1_ORDER.to_bytes(32, "big") + s.to_bytes(32, "big")
        order_s = r.to_bytes(32, "big") + SECP256K1_ORDER.to_bytes(32, "big")

        assert verify_signature(pub, msg, zero_r.hex()) is False
        assert verify_signature(pub, msg, zero_s.hex()) is False
        assert verify_signature(pub, msg, order_r.hex()) is False
        assert verify_signature(pub, msg, order_s.hex()) is False


# ============================================================================
# 5. Multi-Key Format Verification (Even/Odd Parity, Compressed, Uncompressed, X-Only)
# ============================================================================

class TestMultiKeyFormatVerification:
    """Verifies support for 33B compressed, 65B uncompressed, and 32B X-only Nostr with even/odd parities."""

    def test_even_parity_curve_point_formats(self):
        """
        Scalar 1 derives generator point G:
        Gx = 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
        Gy = 483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8 (even, prefix 02)
        """
        priv_1 = "01".zfill(64)
        signer_1 = ClientSigner(priv_1)
        msg = b"message signed by scalar 1 (even parity G)"
        sig = signer_1.sign_message(msg)

        # 33-byte compressed key (starts with 02)
        pub_comp = signer_1.get_pubkey_hex()
        assert pub_comp.startswith("02")
        assert len(pub_comp) == 66
        assert verify_signature(pub_comp, msg, sig) is True

        # 65-byte uncompressed key (starts with 04)
        pub_uncomp = signer_1.get_uncompressed_pubkey_hex()
        assert pub_uncomp.startswith("04")
        assert len(pub_uncomp) == 130
        assert verify_signature(pub_uncomp, msg, sig) is True

        # 32-byte X-only Nostr pubkey (exact Gx coordinate)
        pub_xonly = signer_1.get_nostr_pubkey()
        assert len(pub_xonly) == 64
        assert pub_xonly.lower() == SECP256K1_GX_HEX.lower()
        assert verify_signature(pub_xonly, msg, sig) is True

    def test_odd_parity_curve_point_formats(self):
        """
        Scalar n - 1 derives point -G:
        x = Gx
        y = p - Gy (odd, prefix 03)
        """
        priv_n_minus_1 = hex(SECP256K1_ORDER - 1)[2:].zfill(64)
        signer_neg = ClientSigner(priv_n_minus_1)
        msg = b"message signed by scalar n - 1 (odd parity -G)"
        sig = signer_neg.sign_message(msg)

        # 33-byte compressed key (starts with 03)
        pub_comp = signer_neg.get_pubkey_hex()
        assert pub_comp.startswith("03")
        assert len(pub_comp) == 66
        assert verify_signature(pub_comp, msg, sig) is True

        # 65-byte uncompressed key (starts with 04)
        pub_uncomp = signer_neg.get_uncompressed_pubkey_hex()
        assert pub_uncomp.startswith("04")
        assert len(pub_uncomp) == 130
        assert verify_signature(pub_uncomp, msg, sig) is True

        # 32-byte X-only Nostr pubkey (exact Gx coordinate, but odd parity point!)
        pub_xonly = signer_neg.get_nostr_pubkey()
        assert len(pub_xonly) == 64
        assert pub_xonly.lower() == SECP256K1_GX_HEX.lower()
        # Dual parity resolution in verify_signature MUST accept this odd parity point!
        assert verify_signature(pub_xonly, msg, sig) is True

    def test_pairwise_cross_key_isolation(self):
        """Signatures from Signer i must never verify under Signer j's public key."""
        signers = [ClientSigner() for _ in range(5)]
        msg = b"cross-key isolation test payload"

        for i, signer_a in enumerate(signers):
            sig_a = signer_a.sign_message(msg)
            for j, signer_b in enumerate(signers):
                if i == j:
                    assert verify_signature(signer_b.get_pubkey_hex(), msg, sig_a) is True
                    assert verify_signature(signer_b.get_nostr_pubkey(), msg, sig_a) is True
                else:
                    assert verify_signature(signer_b.get_pubkey_hex(), msg, sig_a) is False
                    assert verify_signature(signer_b.get_nostr_pubkey(), msg, sig_a) is False
                    assert verify_signature(signer_b.get_uncompressed_pubkey_hex(), msg, sig_a) is False

    def test_invalid_pubkey_formats_and_points(self):
        """Malformed, off-curve, and wrong-length public keys must return False without crashing."""
        signer = ClientSigner()
        msg = b"test payload"
        sig = signer.sign_message(msg)

        invalid_pubkeys = [
            "",                        # Empty
            "nothex",                  # Non-hex
            "11" * 16,                 # 16 bytes
            "11" * 31,                 # 31 bytes
            "11" * 34,                 # 34 bytes
            "11" * 64,                 # 64 bytes
            "11" * 66,                 # 66 bytes
            "00" + "11" * 32,          # Invalid compressed prefix 00
            "01" + "11" * 32,          # Invalid compressed prefix 01
            "04" + "11" * 32,          # Prefix 04 with only 33 bytes
            "05" + "11" * 32,          # Invalid prefix 05
            "02" + "11" * 64,          # Prefix 02 with 65 bytes
            "04" + "00" * 32 + "00" * 32,  # Point at infinity / (0,0) off-curve
            signer.get_nostr_npub(),   # Bech32 npub string passed as pubkey hex
        ]

        for bad_pub in invalid_pubkeys:
            assert verify_signature(bad_pub, msg, sig) is False


# ============================================================================
# 6. BIP-173 Bech32 Codec Oracle & Fuzzing
# ============================================================================

class TestBech32CodecOracle:
    """Stress tests Bech32 encoding/decoding against independent oracle and mutations."""

    def test_npub_roundtrip_against_oracle(self):
        """Verifies 25 randomly generated ClientSigner npubs against independent oracle."""
        for _ in range(25):
            signer = ClientSigner()
            npub = signer.get_nostr_npub()
            assert npub.startswith("npub1")
            assert len(npub) == 63

            # Decode using oracle
            hrp, data = bech32_decode_oracle(npub)
            assert hrp == "npub"
            assert data is not None
            assert len(data) == 32
            assert data.hex() == signer.get_nostr_pubkey()

            # Decode using library
            lib_hrp, lib_data = bech32_decode(npub)
            assert lib_hrp == hrp
            assert lib_data == data

    def test_nsec_roundtrip_against_oracle(self):
        """Verifies 25 randomly generated private keys converted to nsec against oracle."""
        for _ in range(25):
            priv_hex, _ = generate_keypair()
            nsec = privkey_to_nsec(priv_hex)
            assert nsec.startswith("nsec1")
            assert len(nsec) == 63

            hrp, data = bech32_decode_oracle(nsec)
            assert hrp == "nsec"
            assert data is not None
            assert len(data) == 32
            assert data.hex() == priv_hex

            # Roundtrip helper
            assert nsec_to_privkey_hex(nsec) == priv_hex

    def test_checksum_mutation_detection(self):
        """Mutating any character in an npub string must cause decode failure."""
        signer = ClientSigner()
        npub = signer.get_nostr_npub()

        for idx in range(len(npub)):
            orig_char = npub[idx]
            # Pick a different valid character
            replacement = "q" if orig_char != "q" else "p"
            mutated = npub[:idx] + replacement + npub[idx + 1:]
            hrp, data = bech32_decode(mutated)
            # If mutated is still valid, it must be because hrp didn't change and checksum failed
            # Bech32 guarantee: 100% of single-character errors are detected by BCH checksum
            assert (hrp, data) == (None, None), f"Failed to detect single-char corruption at {idx}"

    def test_bech32_malformed_inputs(self):
        """Non-Bech32 strings, illegal characters, and mixed cases must return (None, None)."""
        bad_cases = [
            "",
            "not_bech32",
            "npub1" + "b" * 10,   # 'b' not in Bech32 charset
            "npub1" + "i" * 10,   # 'i' not in Bech32 charset
            "npub1" + "o" * 10,   # 'o' not in Bech32 charset
            "npub1" + "1" * 10,   # '1' cannot appear in data part
            "Npub1MixedCase12345",
            None,
            12345,
        ]
        for bad in bad_cases:
            assert bech32_decode(bad) == (None, None)


# ============================================================================
# 7. TileResult Result Signing & Replay Resistance
# ============================================================================

class TestTileResultSigningAndCrossReplay:
    """Stress tests TileResult cryptographic verification and cross-task replay resistance."""

    def test_tile_result_signature_tamper_detection(self):
        """Altering task_id, tensor bytes, signature, or pubkey causes verification rejection."""
        signer = ClientSigner()
        task_id = "task-matrix-block-001"
        result_bytes = b"matrix_tensor_output_data_fp32" * 16

        sig = signer.sign_result(task_id, result_bytes)
        pub = signer.get_pubkey_hex()
        nostr_pub = signer.get_nostr_pubkey()

        # Valid baseline
        assert verify_tile_result_signature(pub, task_id, result_bytes, sig) is True
        assert verify_tile_result_signature(nostr_pub, task_id, result_bytes, sig) is True

        # Tampered task_id
        assert verify_tile_result_signature(pub, "task-matrix-block-002", result_bytes, sig) is False

        # Tampered result bytes
        tampered_bytes = bytearray(result_bytes)
        tampered_bytes[0] ^= 0xFF
        assert verify_tile_result_signature(pub, task_id, bytes(tampered_bytes), sig) is False

        # Tampered signature
        tampered_sig = bytearray(bytes.fromhex(sig))
        tampered_sig[10] ^= 0x01
        assert verify_tile_result_signature(pub, task_id, result_bytes, tampered_sig.hex()) is False

        # Tampered worker pubkey
        other_signer = ClientSigner()
        assert verify_tile_result_signature(other_signer.get_pubkey_hex(), task_id, result_bytes, sig) is False
        assert verify_tile_result_signature(other_signer.get_nostr_pubkey(), task_id, result_bytes, sig) is False

    def test_cross_task_result_isolation(self):
        """Signature on Task A cannot be replayed on Task B even with identical tensor results."""
        signer = ClientSigner()
        common_tensor = b"\x00\x00\x80\x3f" * 64  # Array of 1.0s in FP32
        sig_a = signer.sign_result("task-tile-A", common_tensor)

        # Verified on task A
        assert verify_tile_result_signature(signer.get_pubkey_hex(), "task-tile-A", common_tensor, sig_a) is True
        # Replay attempt on task B must fail
        assert verify_tile_result_signature(signer.get_pubkey_hex(), "task-tile-B", common_tensor, sig_a) is False

