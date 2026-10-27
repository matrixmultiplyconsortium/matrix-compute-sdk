"""
Unit tests for NIP-04 Secp256k1 ECDH AES-256-CBC cipher in matrix-compute-sdk.
Executes 100% offline with zero external network access.
"""

import base64
import json
import secrets
import pytest

from matrix_compute.crypto.nip04 import (
    Nip04Cipher,
    compute_nip04_shared_secret,
    nip04_decrypt,
    nip04_decrypt_raw,
    nip04_encrypt,
    AES_BLOCK_SIZE,
    IV_BYTES,
)
from matrix_compute.crypto.keys import (
    SECP256K1_ORDER,
    privkey_to_pubkey,
    pubkey_to_xonly,
    privkey_to_nsec,
    pubkey_to_npub,
)
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.crypto.session import DecryptionError, KeyExchangeError


@pytest.fixture
def alice_bob_keys():
    alice_priv_hex = "1111111111111111111111111111111111111111111111111111111111111111"
    bob_priv_hex = "2222222222222222222222222222222222222222222222222222222222222222"

    alice_pub_compressed = privkey_to_pubkey(alice_priv_hex, compressed=True)
    alice_pub_xonly = pubkey_to_xonly(alice_pub_compressed)

    bob_pub_compressed = privkey_to_pubkey(bob_priv_hex, compressed=True)
    bob_pub_xonly = pubkey_to_xonly(bob_pub_compressed)

    return {
        "alice_priv": alice_priv_hex,
        "alice_pub_comp": alice_pub_compressed,
        "alice_pub_xonly": alice_pub_xonly,
        "bob_priv": bob_priv_hex,
        "bob_pub_comp": bob_pub_compressed,
        "bob_pub_xonly": bob_pub_xonly,
    }


def test_nip04_deterministic_test_vector():
    """Validates Alice d=1, Bob d=2, S=X_2G, fixed IV ciphertext matches reference Base64."""
    d_alice = "0000000000000000000000000000000000000000000000000000000000000001"
    d_bob = "0000000000000000000000000000000000000000000000000000000000000002"

    # Alice pubkey G
    p_alice_comp = privkey_to_pubkey(d_alice, compressed=True)
    assert p_alice_comp == "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
    p_alice_xonly = pubkey_to_xonly(p_alice_comp)

    # Bob pubkey 2G
    p_bob_comp = privkey_to_pubkey(d_bob, compressed=True)
    assert p_bob_comp == "02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"
    p_bob_xonly = pubkey_to_xonly(p_bob_comp)

    # Verify ECDH shared secret is X_2G
    shared_alice = compute_nip04_shared_secret(d_alice, p_bob_xonly)
    shared_bob = compute_nip04_shared_secret(d_bob, p_alice_xonly)
    assert shared_alice == shared_bob
    assert shared_alice.hex() == "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"

    # Plaintext and fixed IV
    plaintext = "Hello Nostr Wallet Connect!"
    fixed_iv = b"0123456789abcdef"

    encrypted = nip04_encrypt(d_alice, p_bob_xonly, plaintext, iv=fixed_iv)
    expected_wire = "8be56amhaJ1rACia4zKGh+F6CVDqHdxzmA6/vi0udW4=?iv=MDEyMzQ1Njc4OWFiY2RlZg=="
    assert encrypted == expected_wire

    # Decrypt with Bob's private key and Alice's public key
    decrypted = nip04_decrypt(d_bob, p_alice_xonly, encrypted)
    assert decrypted == plaintext


def test_nip04_roundtrip_ascii_string(alice_bob_keys):
    plaintext = "Standard ASCII encryption payload for Nostr test."
    payload = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], plaintext)
    assert "?iv=" in payload
    decrypted = nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)
    assert decrypted == plaintext


def test_nip04_roundtrip_json_payload(alice_bob_keys):
    rpc_req = {
        "method": "pay_invoice",
        "params": {
            "invoice": "lnbc100n1pj0mock_invoice_for_test",
            "amount": 100000,
        },
    }
    plaintext = json.dumps(rpc_req)
    payload = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], plaintext)
    decrypted_str = nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)
    decrypted_json = json.loads(decrypted_str)
    assert decrypted_json == rpc_req


def test_nip04_roundtrip_raw_bytes(alice_bob_keys):
    binary_data = bytes([0x00, 0xFF, 0x10, 0x20, 0x05, 0x00, 0x7F, 0x80])
    payload = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], binary_data)
    decrypted = nip04_decrypt_raw(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)
    assert decrypted == binary_data


def test_nip04_bidirectional_symmetry(alice_bob_keys):
    msg_a_to_b = "Alice to Bob message"
    ct_a_to_b = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], msg_a_to_b)
    assert nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], ct_a_to_b) == msg_a_to_b

    msg_b_to_a = "Bob to Alice response"
    ct_b_to_a = nip04_encrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], msg_b_to_a)
    assert nip04_decrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], ct_b_to_a) == msg_b_to_a


def test_nip04_client_signer_methods(alice_bob_keys):
    signer_alice = ClientSigner(alice_bob_keys["alice_priv"])
    signer_bob = ClientSigner(alice_bob_keys["bob_priv"])

    msg = "Testing ClientSigner NIP-04 integration"
    enc = signer_alice.nip04_encrypt(signer_bob.get_nostr_pubkey(), msg)
    dec = signer_bob.nip04_decrypt(signer_alice.get_nostr_pubkey(), enc)
    assert dec == msg

    ss1 = signer_alice.compute_nip04_shared_secret(signer_bob.get_nostr_pubkey())
    ss2 = signer_bob.compute_nip04_shared_secret(signer_alice.get_nostr_pubkey())
    assert ss1 == ss2


def test_nip04_cipher_class_interface(alice_bob_keys):
    cipher_alice = Nip04Cipher(alice_bob_keys["alice_priv"])
    cipher_bob = Nip04Cipher(alice_bob_keys["bob_priv"])

    msg = "Cipher class interface test"
    enc = cipher_alice.encrypt(alice_bob_keys["bob_pub_xonly"], msg)
    dec = cipher_bob.decrypt(alice_bob_keys["alice_pub_xonly"], enc)
    assert dec == msg

    raw_data = b"raw binary test"
    enc_raw = cipher_alice.encrypt(alice_bob_keys["bob_pub_xonly"], raw_data)
    dec_raw = cipher_bob.decrypt_raw(alice_bob_keys["alice_pub_xonly"], enc_raw)
    assert dec_raw == raw_data


def test_nip04_key_format_interoperability(alice_bob_keys):
    # nsec and npub
    nsec_alice = privkey_to_nsec(alice_bob_keys["alice_priv"])
    npub_bob = pubkey_to_npub(alice_bob_keys["bob_pub_xonly"])

    msg = "Key format interop"
    payload = nip04_encrypt(nsec_alice, npub_bob, msg)

    # Decrypt with raw bytes privkey and compressed pubkey
    bob_priv_bytes = bytes.fromhex(alice_bob_keys["bob_priv"])
    alice_pub_comp = alice_bob_keys["alice_pub_comp"]
    dec = nip04_decrypt(bob_priv_bytes, alice_pub_comp, payload)
    assert dec == msg


def test_nip04_fresh_iv_uniqueness(alice_bob_keys):
    plaintext = "Constant Plaintext"
    ivs = set()
    ciphertexts = set()

    for _ in range(100):
        enc = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], plaintext)
        parts = enc.split("?iv=")
        ciphertexts.add(parts[0])
        ivs.add(parts[1])

    assert len(ivs) == 100
    assert len(ciphertexts) == 100


def test_nip04_empty_string_padding(alice_bob_keys):
    empty = ""
    payload = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], empty)
    decrypted = nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)
    assert decrypted == empty


def test_nip04_unicode_and_emojis(alice_bob_keys):
    unicode_msg = "⚡ Lightning Fast! 日本語テスト 🚀 Special chars: café, naïve, ñ, 𝄞"
    payload = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], unicode_msg)
    decrypted = nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)
    assert decrypted == unicode_msg


def test_nip04_missing_delimiter_rejected(alice_bob_keys):
    with pytest.raises(DecryptionError, match="missing '\\?iv='"):
        nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], "corruptpayloadwithoutiv")


def test_nip04_empty_components_rejected(alice_bob_keys):
    with pytest.raises(DecryptionError, match="expected non-empty"):
        nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], "?iv=MDEyMzQ1Njc4OWFiY2RlZg==")

    with pytest.raises(DecryptionError, match="expected non-empty"):
        nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], "8be56amhaJ1rACia4zKGh+F6CVDqHdxzmA6/vi0udW4=?iv=")


def test_nip04_invalid_base64_rejected(alice_bob_keys):
    # Non-base64 characters in ciphertext
    with pytest.raises(DecryptionError, match="Invalid Base64"):
        nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], "bad!!!base64?iv=MDEyMzQ1Njc4OWFiY2RlZg==")

    # Non-base64 in IV
    with pytest.raises(DecryptionError, match="Invalid Base64"):
        nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], "8be56amhaJ1rACia4zKGh+F6CVDqHdxzmA6/vi0udW4=?iv=bad!!!iv")


def test_nip04_short_iv_rejected(alice_bob_keys):
    # 15-byte IV
    short_iv = base64.b64encode(b"0" * 15).decode()
    payload = f"8be56amhaJ1rACia4zKGh+F6CVDqHdxzmA6/vi0udW4=?iv={short_iv}"
    with pytest.raises(DecryptionError, match="Invalid IV length"):
        nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)


def test_nip04_long_iv_rejected(alice_bob_keys):
    # 17-byte IV
    long_iv = base64.b64encode(b"0" * 17).decode()
    payload = f"8be56amhaJ1rACia4zKGh+F6CVDqHdxzmA6/vi0udW4=?iv={long_iv}"
    with pytest.raises(DecryptionError, match="Invalid IV length"):
        nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)


def test_nip04_unaligned_ciphertext_rejected(alice_bob_keys):
    # Ciphertext of 15 bytes
    unaligned_ct = base64.b64encode(b"x" * 15).decode()
    valid_iv = base64.b64encode(b"0" * 16).decode()
    payload = f"{unaligned_ct}?iv={valid_iv}"
    with pytest.raises(DecryptionError, match="Invalid ciphertext length"):
        nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)


def test_nip04_tampered_ciphertext_fails_padding(alice_bob_keys):
    payload = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], "Secret message")
    parts = payload.split("?iv=")
    ct_bytes = bytearray(base64.b64decode(parts[0]))
    # Flip one bit in ciphertext
    ct_bytes[0] ^= 0x01
    tampered_payload = f"{base64.b64encode(ct_bytes).decode()}?iv={parts[1]}"

    with pytest.raises(DecryptionError, match="invalid PKCS#7 padding"):
        nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], tampered_payload)


def test_nip04_tampered_iv_fails(alice_bob_keys):
    payload = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], "Secret message")
    parts = payload.split("?iv=")
    iv_bytes = bytearray(base64.b64decode(parts[1]))
    # Flip bit in IV
    iv_bytes[0] ^= 0x01
    tampered_payload = f"{parts[0]}?iv={base64.b64encode(iv_bytes).decode()}"

    # In CBC mode, IV tampering corrupts block 0; if block 0 is also the last block (short message),
    # padding may fail, or result is corrupted.
    try:
        dec = nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], tampered_payload)
        assert dec != "Secret message"
    except DecryptionError:
        pass  # Expected if corrupted padding


def test_nip04_wrong_recipient_key_fails(alice_bob_keys):
    charlie_priv = "3333333333333333333333333333333333333333333333333333333333333333"
    payload = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], "For Bob only")

    with pytest.raises(DecryptionError, match="invalid PKCS#7 padding"):
        nip04_decrypt(charlie_priv, alice_bob_keys["alice_pub_xonly"], payload)


def test_nip04_scalar_zero_private_key_rejected(alice_bob_keys):
    zero_key = "0000000000000000000000000000000000000000000000000000000000000000"
    with pytest.raises(ValueError, match="out of valid range"):
        nip04_encrypt(zero_key, alice_bob_keys["bob_pub_xonly"], "Hello")


def test_nip04_scalar_order_overflow_rejected(alice_bob_keys):
    overflow_key = hex(SECP256K1_ORDER)[2:].zfill(64)
    with pytest.raises(ValueError, match="out of valid range"):
        nip04_encrypt(overflow_key, alice_bob_keys["bob_pub_xonly"], "Hello")


def test_nip04_off_curve_public_key_rejected(alice_bob_keys):
    # Invalid pubkey point
    off_curve_pub = "00" * 32
    with pytest.raises(KeyExchangeError):
        nip04_encrypt(alice_bob_keys["alice_priv"], off_curve_pub, "Hello")


def test_nip04_non_utf8_plaintext_handling(alice_bob_keys):
    # Non-UTF8 raw bytes
    invalid_utf8 = b"\x80\x81\x82\x83\xfe\xff"
    payload = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], invalid_utf8)

    # nip04_decrypt should fail with DecryptionError
    with pytest.raises(DecryptionError, match="not valid UTF-8"):
        nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)

    # nip04_decrypt_raw should succeed
    raw = nip04_decrypt_raw(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)
    assert raw == invalid_utf8


def test_nip04_large_multiblock_payload(alice_bob_keys):
    large_text = "NIP-04 Multiblock Payload Test Line.\n" * 200  # ~7.4 KB
    payload = nip04_encrypt(alice_bob_keys["alice_priv"], alice_bob_keys["bob_pub_xonly"], large_text)
    decrypted = nip04_decrypt(alice_bob_keys["bob_priv"], alice_bob_keys["alice_pub_xonly"], payload)
    assert decrypted == large_text
