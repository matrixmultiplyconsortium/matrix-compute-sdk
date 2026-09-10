"""Comprehensive unit tests for Secp256k1 crypto engine and Nostr key management."""

import hashlib
import numpy as np
import pytest

from matrix_compute.crypto.keys import (
    SECP256K1_GX_HEX,
    SECP256K1_HALF_ORDER,
    SECP256K1_ORDER,
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


def test_keypair_generation_formats():
    """TC-CRY-01: Verifies generated key formats, lengths, and prefixes."""
    signer = ClientSigner()
    priv = signer.get_privkey_hex()
    pub = signer.get_pubkey_hex()
    uncompressed = signer.get_uncompressed_pubkey_hex()
    xonly = signer.get_nostr_pubkey()
    npub = signer.get_nostr_npub()

    assert len(priv) == 64
    assert len(pub) == 66
    assert pub[:2] in ("02", "03")
    assert len(uncompressed) == 130
    assert uncompressed[:2] == "04"
    assert len(xonly) == 64
    assert xonly == pub[2:]
    assert npub.startswith("npub1")
    assert len(npub) == 63


def test_known_generator_g_test_vector():
    """TC-CRY-02: Verifies scalar 1 derives known generator G coordinate."""
    priv_one_hex = "01".zfill(64)
    derived_pub = privkey_to_pubkey(priv_one_hex, compressed=True)
    assert derived_pub.lower() == f"02{SECP256K1_GX_HEX}".lower()
    xonly = pubkey_to_xonly(derived_pub)
    assert xonly.lower() == SECP256K1_GX_HEX.lower()


def test_known_point_negation_vector():
    """TC-CRY-03: Verifies scalar n - 1 derives point -G with prefix 03 and coordinate Gx."""
    priv_n_minus_1 = hex(SECP256K1_ORDER - 1)[2:].zfill(64)
    derived_pub = privkey_to_pubkey(priv_n_minus_1, compressed=True)
    assert derived_pub.startswith("03")
    assert derived_pub[2:].lower() == SECP256K1_GX_HEX.lower()


def test_bech32_encode_decode_roundtrip():
    """TC-CRY-04: Verifies bidirectional encoding and decoding of arbitrary byte payloads."""
    raw_payload = b"\xde\xad\xbe\xef" * 8
    encoded = bech32_encode("test", raw_payload)
    assert encoded.startswith("test1")
    hrp, decoded = bech32_decode(encoded)
    assert hrp == "test"
    assert decoded == raw_payload


def test_bech32_checksum_mutation_rejected():
    """TC-CRY-05: Mutating any character in Bech32 string causes decode failure."""
    raw = b"\xca\xfe\xba\xbe" * 8
    encoded = bech32_encode("npub", raw)
    # Mutate last character
    mutated = encoded[:-1] + ("q" if encoded[-1] != "q" else "p")
    hrp, decoded = bech32_decode(mutated)
    assert hrp is None
    assert decoded is None


def test_bech32_malformed_inputs():
    """TC-CRY-06: Malformed Bech32 inputs return (None, None) safely."""
    assert bech32_decode("") == (None, None)
    assert bech32_decode("notbech32") == (None, None)
    assert bech32_decode("Npub1MixedCase12345") == (None, None)
    assert bech32_decode("npub1" + "b" * 10) == (None, None)  # 'b' is not in bech32 charset


def test_nsec_and_npub_helpers():
    """TC-CRY-07: Tests privkey_to_nsec, nsec_to_privkey_hex, pubkey_to_npub, npub_to_pubkey_hex."""
    priv_hex = "11" * 32
    nsec = privkey_to_nsec(priv_hex)
    assert nsec.startswith("nsec1")
    assert nsec_to_privkey_hex(nsec) == priv_hex

    pub_hex = privkey_to_pubkey(priv_hex, compressed=True)
    npub = pubkey_to_npub(pub_hex)
    assert npub.startswith("npub1")
    assert npub_to_pubkey_hex(npub) == pub_hex[2:]


def test_scalar_zero_rejected():
    """TC-CRY-08: Scalar 0 raises ValueError in ClientSigner and validate_private_key_hex."""
    assert validate_private_key_int(0) is False
    assert validate_private_key_hex("00" * 32) is False
    with pytest.raises(ValueError):
        ClientSigner("00" * 32)


def test_scalar_order_rejected():
    """TC-CRY-09: Scalar n (curve order) raises ValueError."""
    order_hex = hex(SECP256K1_ORDER)[2:].zfill(64)
    assert validate_private_key_int(SECP256K1_ORDER) is False
    assert validate_private_key_hex(order_hex) is False
    with pytest.raises(ValueError):
        ClientSigner(order_hex)


def test_scalar_greater_than_order_rejected():
    """TC-CRY-10: Scalar n + 1 and 2^256 - 1 raise ValueError."""
    over_order = hex(SECP256K1_ORDER + 1)[2:].zfill(64)
    max_u256 = "ff" * 32
    assert validate_private_key_hex(over_order) is False
    assert validate_private_key_hex(max_u256) is False
    with pytest.raises(ValueError):
        ClientSigner(over_order)
    with pytest.raises(ValueError):
        ClientSigner(max_u256)


def test_malformed_private_key_strings():
    """TC-CRY-11: Rejects odd lengths, non-hex characters, empty strings, and prefixes."""
    assert validate_private_key_hex("1234") is False
    assert validate_private_key_hex("0x" + "11" * 32) is False
    assert validate_private_key_hex("g" * 64) is False
    assert validate_private_key_hex("") is False


def test_sign_and_verify_nominal():
    """TC-CRY-12: Basic signature creation over string and bytes payloads."""
    signer = ClientSigner()
    # String payload
    sig1 = signer.sign_message("distributed compute task #1")
    assert signer.verify_signature("distributed compute task #1", sig1) is True

    # Raw bytes payload
    data_bytes = b"\x01\x02\x03\x04\x05"
    sig2 = signer.sign_message(data_bytes)
    assert signer.verify_signature(data_bytes, sig2) is True


def test_functional_api_parity():
    """TC-CRY-13: Top-level sign_message and verify_signature produce identical behavior."""
    priv, pub = generate_keypair()
    msg = b"hello distributed matrix"
    sig = sign_message(priv, msg)
    assert verify_signature(pub, msg, sig) is True


def test_canonical_low_s_normalization():
    """TC-CRY-14: Signs 50 distinct randomized messages, asserting s <= n/2 for 100% of signatures."""
    signer = ClientSigner()
    for i in range(50):
        msg = f"random test message {i}".encode("utf-8")
        sig = signer.sign_message(msg)
        sig_bytes = bytes.fromhex(sig)
        s = int.from_bytes(sig_bytes[32:], "big")
        assert s <= SECP256K1_HALF_ORDER, f"Signature {i} failed low-S normalization: s={s}"


def test_signature_compact_format():
    """TC-CRY-15: Verifies compact signature is exactly 64 bytes (128 hex characters)."""
    signer = ClientSigner()
    sig = signer.sign_message("compact format test")
    assert len(sig) == 128
    sig_bytes = bytes.fromhex(sig)
    assert len(sig_bytes) == 64
    r = int.from_bytes(sig_bytes[:32], "big")
    s = int.from_bytes(sig_bytes[32:], "big")
    assert 1 <= r < SECP256K1_ORDER
    assert 1 <= s <= SECP256K1_HALF_ORDER


def test_tampered_message_detected():
    """TC-CRY-16: Single-byte modification to signed payload causes verification failure."""
    signer = ClientSigner()
    msg = b"original uncorrupted payload"
    sig = signer.sign_message(msg)
    tampered = b"original uncorrupted payloae"
    assert signer.verify_signature(tampered, sig) is False


def test_signature_bit_flip_detection():
    """TC-CRY-17: Flips individual bits in r and s coordinates; verifies rejection."""
    signer = ClientSigner()
    msg = b"bit flip test"
    sig = signer.sign_message(msg)
    sig_bytes = bytearray(bytes.fromhex(sig))

    # Flip bit in r
    sig_bytes_r = bytearray(sig_bytes)
    sig_bytes_r[0] ^= 0x01
    assert signer.verify_signature(msg, sig_bytes_r.hex()) is False

    # Flip bit in s
    sig_bytes_s = bytearray(sig_bytes)
    sig_bytes_s[32] ^= 0x01
    assert signer.verify_signature(msg, sig_bytes_s.hex()) is False


def test_truncated_extended_signature():
    """TC-CRY-18: Truncated (< 64 bytes) or padded (> 64 bytes) signatures return False."""
    signer = ClientSigner()
    msg = b"length test"
    sig = signer.sign_message(msg)
    assert signer.verify_signature(msg, sig[:120]) is False
    assert signer.verify_signature(msg, sig + "00") is False


def test_cross_key_isolation():
    """TC-CRY-19: Signatures generated by Signer A are rejected when verified against Signer B."""
    signer_a = ClientSigner()
    signer_b = ClientSigner()
    msg = b"secret consumer instruction"
    sig_a = signer_a.sign_message(msg)
    assert signer_b.verify_signature(msg, sig_a) is False


def test_malformed_signature_inputs():
    """TC-CRY-20: Non-hex strings, numbers, None, and empty strings return False safely."""
    signer = ClientSigner()
    msg = b"malformed test"
    assert signer.verify_signature(msg, "") is False
    assert signer.verify_signature(msg, "nothex") is False
    assert signer.verify_signature(msg, None) is False
    assert signer.verify_signature(msg, 12345) is False


def test_verify_compressed_pubkey_33b():
    """TC-CRY-21: Verifies signature against 33-byte compressed public key hex."""
    signer = ClientSigner()
    msg = b"compressed pubkey test"
    sig = signer.sign_message(msg)
    assert verify_signature(signer.get_pubkey_hex(), msg, sig) is True


def test_verify_uncompressed_pubkey_65b():
    """TC-CRY-22: Verifies signature against 65-byte uncompressed public key hex."""
    signer = ClientSigner()
    msg = b"uncompressed pubkey test"
    sig = signer.sign_message(msg)
    assert verify_signature(signer.get_uncompressed_pubkey_hex(), msg, sig) is True


def test_verify_xonly_pubkey_32b():
    """TC-CRY-23: Verifies signature against 32-byte X-only Nostr public key hex."""
    signer = ClientSigner()
    msg = b"nostr x-only pubkey test"
    sig = signer.sign_message(msg)
    assert verify_signature(signer.get_nostr_pubkey(), msg, sig) is True


def test_invalid_pubkey_lengths():
    """TC-CRY-24: Public keys of lengths != 32, 33, or 65 bytes return False."""
    signer = ClientSigner()
    msg = b"invalid length pubkey"
    sig = signer.sign_message(msg)
    assert verify_signature("11" * 16, msg, sig) is False
    assert verify_signature("11" * 40, msg, sig) is False


def test_invalid_pubkey_prefixes():
    """TC-CRY-25: Compressed keys with invalid prefixes return False."""
    signer = ClientSigner()
    msg = b"invalid prefix"
    sig = signer.sign_message(msg)
    pub = signer.get_pubkey_hex()
    bad_pub = "00" + pub[2:]
    assert verify_signature(bad_pub, msg, sig) is False


def test_off_curve_pubkey_points():
    """TC-CRY-26: Points not on Secp256k1 curve return False safely."""
    # (x=0, y=0) is off curve (0^2 != 0^3 + 7)
    off_curve_pub = "04" + "00" * 32 + "00" * 32
    signer = ClientSigner()
    sig = signer.sign_message(b"test")
    assert verify_signature(off_curve_pub, b"test", sig) is False


def test_npub_passed_to_verify_returns_false():
    """TC-CRY-27: Passing npub1... string directly to verify_signature returns False safely."""
    signer = ClientSigner()
    sig = signer.sign_message(b"test")
    assert verify_signature(signer.get_nostr_npub(), b"test", sig) is False


def test_worker_to_client_cross_verify():
    """TC-CRY-28: Mock/reference WorkerSigner signature is verified by SDK verify_signature."""
    priv, pub = generate_keypair()
    task_id = "task-cross-1"
    result_bytes = np.ones((4, 4), dtype=np.float32).tobytes()
    c_hash = hashlib.sha256(result_bytes).hexdigest()
    payload = f"{task_id}:{c_hash}".encode("utf-8")
    sig = sign_message(priv, payload)

    # SDK verifies with compressed pubkey
    assert verify_tile_result_signature(pub, task_id, result_bytes, sig) is True
    # SDK verifies with 32-byte X-only pubkey
    assert verify_tile_result_signature(pub[2:], task_id, result_bytes, sig) is True


def test_client_to_worker_cross_verify():
    """TC-CRY-29: SDK ClientSigner signature is verified by verify_peer_signature classmethod."""
    signer = ClientSigner()
    data = b"handshake auth payload"
    sig = signer.sign_message(data)
    assert ClientSigner.verify_peer_signature(data, sig, signer.get_pubkey_hex()) is True
    assert ClientSigner.verify_peer_signature(data, sig, signer.get_nostr_pubkey()) is True


def test_sign_result_convenience():
    """TC-CRY-30: ClientSigner.sign_result verifies over raw tensor bytes and precomputed SHA256."""
    signer = ClientSigner()
    task_id = "task-gemm-42"
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # Sign using array/bytes
    sig1 = signer.sign_result(task_id, arr.tobytes())
    assert verify_tile_result_signature(signer.get_pubkey_hex(), task_id, arr.tobytes(), sig1) is True

    # Sign using precomputed hash
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    sig2 = signer.sign_result(task_id, digest)
    assert verify_tile_result_signature(signer.get_pubkey_hex(), task_id, arr.tobytes(), sig2) is True


def test_cross_task_result_isolation():
    """TC-CRY-31: Result signature on task A cannot be replayed on task B."""
    signer = ClientSigner()
    arr = np.zeros((4, 4), dtype=np.float32)
    sig = signer.sign_result("task-A", arr.tobytes())
    assert verify_tile_result_signature(signer.get_pubkey_hex(), "task-A", arr.tobytes(), sig) is True
    assert verify_tile_result_signature(signer.get_pubkey_hex(), "task-B", arr.tobytes(), sig) is False


def test_sha256_and_tensor_hashing():
    """TC-CRY-32: Tests sha256_digest and hash_tensor with numpy array and memoryviews."""
    arr = np.arange(16, dtype=np.float32)
    expected_hash = hashlib.sha256(arr.tobytes()).hexdigest()

    assert hash_tensor(arr) == expected_hash
    assert hash_tensor(memoryview(arr.tobytes())) == expected_hash
    assert sha256_digest("test string") == hashlib.sha256(b"test string").hexdigest()
    assert sha256_digest(b"test bytes") == hashlib.sha256(b"test bytes").hexdigest()
