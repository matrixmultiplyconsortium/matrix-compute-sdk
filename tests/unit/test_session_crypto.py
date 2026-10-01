"""
Unit tests for Session Crypto & Wire AES-256-GCM Framing (Features F14, F15, F17, F18).
"""

from __future__ import annotations

import secrets
from cryptography.hazmat.primitives.asymmetric import x25519
import numpy as np
import pytest

from matrix_compute.crypto.session import (
    AES_GCM_IV_BYTES,
    AES_GCM_TAG_BYTES,
    AES_KEY_BYTES,
    HKDF_INFO_LABEL,
    NONCE_BYTES,
    X25519_KEY_BYTES,
    AuthenticationTagError,
    CryptoSession,
    DecryptionError,
    EncryptedPayload,
    EncryptionError,
    FramingError,
    KeyExchangeError,
    SecurityPolicyViolationError,
    SessionContext,
    SessionCrypto,
    SessionKeys,
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    canonical_chunk_aad,
    canonical_result_aad,
    compute_chunk_aad,
    compute_ecdh_shared_secret,
    compute_result_aad,
    decrypt_chunk,
    decrypt_result,
    derive_session_keys,
    encrypt_chunk,
    encrypt_result,
    generate_challenger_nonce,
    generate_ephemeral_keypair,
    generate_ephemeral_x25519_keypair,
)


# ==============================================================================
# Tier 1: Ephemeral X25519 ECDH Key Exchange Tests (F14)
# ==============================================================================

def test_x25519_keypair_generation():
    priv1, pub1 = generate_ephemeral_keypair()
    priv2, pub2 = generate_ephemeral_x25519_keypair()

    assert isinstance(priv1, x25519.X25519PrivateKey)
    assert isinstance(priv2, x25519.X25519PrivateKey)
    assert len(pub1) == X25519_KEY_BYTES
    assert len(pub2) == X25519_KEY_BYTES
    assert pub1 != pub2


def test_challenger_nonce_generation():
    n1 = generate_challenger_nonce()
    n2 = generate_challenger_nonce()
    assert len(n1) == NONCE_BYTES
    assert len(n2) == NONCE_BYTES
    assert n1 != n2


def test_ecdh_shared_secret_symmetry():
    client_priv, client_pub = generate_ephemeral_keypair()
    worker_priv, worker_pub = generate_ephemeral_keypair()

    z_client = compute_ecdh_shared_secret(client_priv, worker_pub)
    z_worker = compute_ecdh_shared_secret(worker_priv, client_pub)

    assert len(z_client) == X25519_KEY_BYTES
    assert z_client == z_worker


def test_ecdh_shared_secret_third_party_differs():
    client_priv, client_pub = generate_ephemeral_keypair()
    worker_priv, worker_pub = generate_ephemeral_keypair()
    eve_priv, eve_pub = generate_ephemeral_keypair()

    z_legit = compute_ecdh_shared_secret(client_priv, worker_pub)
    z_eve = compute_ecdh_shared_secret(eve_priv, worker_pub)

    assert z_legit != z_eve


def test_ecdh_invalid_public_key_rejected():
    client_priv, _ = generate_ephemeral_keypair()

    # Wrong length
    with pytest.raises(KeyExchangeError):
        compute_ecdh_shared_secret(client_priv, b"\x00" * 31)

    with pytest.raises(KeyExchangeError):
        compute_ecdh_shared_secret(client_priv, b"\x00" * 33)

    # Invalid type
    with pytest.raises(KeyExchangeError):
        compute_ecdh_shared_secret(client_priv, "not_bytes")  # type: ignore


def test_ecdh_low_order_point_rejected():
    client_priv, _ = generate_ephemeral_keypair()
    # 0 point produces all-zero shared secret
    zero_point = b"\x00" * 32
    with pytest.raises(KeyExchangeError):
        compute_ecdh_shared_secret(client_priv, zero_point)


# ==============================================================================
# Tier 2: HKDF-SHA256 Session Key Derivation Tests (F15)
# ==============================================================================

def test_hkdf_directional_keys_derived():
    shared_secret = secrets.token_bytes(32)
    nonce_c = secrets.token_bytes(32)
    nonce_w = secrets.token_bytes(32)

    keys = derive_session_keys(shared_secret, nonce_c, nonce_w)
    assert isinstance(keys, SessionKeys)
    assert len(keys.k_c2w) == AES_KEY_BYTES
    assert len(keys.k_w2c) == AES_KEY_BYTES
    assert keys.k_c2w != keys.k_w2c
    assert keys.shared_secret == shared_secret
    assert keys.salt == nonce_c + nonce_w
    assert keys.info == HKDF_INFO_LABEL


def test_hkdf_single_nonce_salt():
    shared_secret = secrets.token_bytes(32)
    nonce_c = secrets.token_bytes(32)

    keys = derive_session_keys(shared_secret, nonce_c)
    assert len(keys.k_c2w) == AES_KEY_BYTES
    assert len(keys.k_w2c) == AES_KEY_BYTES
    assert keys.k_c2w != keys.k_w2c
    assert keys.salt == nonce_c


def test_hkdf_private_key_direct_derivation():
    c_priv, c_pub = generate_ephemeral_keypair()
    w_priv, w_pub = generate_ephemeral_keypair()
    nonce = secrets.token_bytes(32)

    client_keys = derive_session_keys(c_priv, w_pub, nonce)
    worker_keys = derive_session_keys(w_priv, c_pub, nonce)

    assert client_keys.k_c2w == worker_keys.k_c2w
    assert client_keys.k_w2c == worker_keys.k_w2c


def test_hkdf_avalanche_effect():
    shared_secret = secrets.token_bytes(32)
    nonce_c1 = secrets.token_bytes(32)
    # Flip one bit in nonce
    nonce_c2 = bytes([nonce_c1[0] ^ 0x01]) + nonce_c1[1:]

    keys1 = derive_session_keys(shared_secret, nonce_c1)
    keys2 = derive_session_keys(shared_secret, nonce_c2)

    assert keys1.k_c2w != keys2.k_c2w
    assert keys1.k_w2c != keys2.k_w2c


def test_session_keys_tuple_unpacking():
    shared_secret = secrets.token_bytes(32)
    nonce = secrets.token_bytes(32)
    keys = derive_session_keys(shared_secret, nonce)

    # Tuple unpacking
    k1, k2 = keys
    assert k1 == keys.k_c2w
    assert k2 == keys.k_w2c

    # Indexing
    assert keys[0] == keys.k_c2w
    assert keys[1] == keys.k_w2c
    with pytest.raises(IndexError):
        _ = keys[2]


# ==============================================================================
# Tier 3: AES-256-GCM AEAD Wire Framing Tests (F17, F18)
# ==============================================================================

def test_aes_gcm_roundtrip_raw():
    key = secrets.token_bytes(32)
    plaintext = b"DistributedMatrixMultiply: Confidential Compute Milestone 3 Payload"
    aad = b"canonical-task-aad:001"

    payload = aes_gcm_encrypt(key, plaintext, aad=aad)
    assert isinstance(payload, EncryptedPayload)
    assert len(payload.iv) == AES_GCM_IV_BYTES
    assert len(payload.tag) == AES_GCM_TAG_BYTES
    assert payload.ciphertext != plaintext
    assert len(payload.ciphertext) == len(plaintext)

    decrypted = aes_gcm_decrypt(key, payload.ciphertext, payload.iv, payload.tag, aad=aad)
    assert decrypted == plaintext


def test_encrypted_payload_unpacking():
    key = secrets.token_bytes(32)
    plaintext = b"Payload for unpacking test"
    payload = aes_gcm_encrypt(key, plaintext)

    ct, iv, tag = payload
    assert ct == payload.ciphertext
    assert iv == payload.iv
    assert tag == payload.tag
    assert payload[0] == ct
    assert payload[1] == iv
    assert payload[2] == tag
    with pytest.raises(IndexError):
        _ = payload[3]


def test_aes_gcm_fresh_iv_uniqueness():
    key = secrets.token_bytes(32)
    plaintext = b"Fixed plaintext to test IV uniqueness across 500 calls"
    ivs = set()
    for _ in range(500):
        p = aes_gcm_encrypt(key, plaintext)
        assert p.iv not in ivs
        ivs.add(p.iv)
    assert len(ivs) == 500


def test_aes_gcm_ciphertext_tamper_fails():
    key = secrets.token_bytes(32)
    plaintext = b"Sensitive matrix payload"
    aad = b"task:aad"

    payload = aes_gcm_encrypt(key, plaintext, aad=aad)
    # Flip bit in ciphertext
    tampered_ct = bytes([payload.ciphertext[0] ^ 0x01]) + payload.ciphertext[1:]

    with pytest.raises(AuthenticationTagError):
        aes_gcm_decrypt(key, tampered_ct, payload.iv, payload.tag, aad=aad)


def test_aes_gcm_tag_tamper_fails():
    key = secrets.token_bytes(32)
    plaintext = b"Sensitive matrix payload"
    aad = b"task:aad"

    payload = aes_gcm_encrypt(key, plaintext, aad=aad)
    # Flip bit in tag
    tampered_tag = bytes([payload.tag[0] ^ 0x01]) + payload.tag[1:]

    with pytest.raises(AuthenticationTagError):
        aes_gcm_decrypt(key, payload.ciphertext, payload.iv, tampered_tag, aad=aad)


def test_aes_gcm_aad_tamper_fails():
    key = secrets.token_bytes(32)
    plaintext = b"Sensitive matrix payload"
    valid_aad = b"task:aad:seq:1"
    tampered_aad = b"task:aad:seq:2"

    payload = aes_gcm_encrypt(key, plaintext, aad=valid_aad)

    with pytest.raises(AuthenticationTagError):
        aes_gcm_decrypt(key, payload.ciphertext, payload.iv, payload.tag, aad=tampered_aad)


def test_aes_gcm_invalid_iv_or_tag_length():
    key = secrets.token_bytes(32)
    payload = aes_gcm_encrypt(key, b"data")

    # Invalid IV length
    with pytest.raises(FramingError):
        aes_gcm_decrypt(key, payload.ciphertext, b"\x00" * 11, payload.tag)
    with pytest.raises(FramingError):
        aes_gcm_decrypt(key, payload.ciphertext, b"\x00" * 13, payload.tag)

    # Invalid Tag length
    with pytest.raises(FramingError):
        aes_gcm_decrypt(key, payload.ciphertext, payload.iv, b"\x00" * 15)
    with pytest.raises(FramingError):
        aes_gcm_decrypt(key, payload.ciphertext, payload.iv, b"\x00" * 17)


# ==============================================================================
# Tier 4: Chunk & Result Helpers and SessionCrypto Bidirectional Tests
# ==============================================================================

def test_chunk_and_result_encryption_convenience_helpers():
    key = secrets.token_bytes(32)
    data = np.random.randn(16, 16).astype(np.float32).tobytes()

    # Chunk helper
    ct, iv, tag = encrypt_chunk(key, data, task_id="task-100", sequence_number=0, is_matrix_a=True, is_final_chunk=False)
    dec = decrypt_chunk(key, ct, iv, tag, task_id="task-100", sequence_number=0, is_matrix_a=True, is_final_chunk=False)
    assert dec == data

    # Result helper
    ct_r, iv_r, tag_r = encrypt_result(key, data, task_id="task-100", rows=16, cols=16)
    dec_r = decrypt_result(key, ct_r, iv_r, tag_r, task_id="task-100", rows=16, cols=16)
    assert dec_r == data


def test_full_session_crypto_bidirectional_exchange():
    # 1. Key exchange
    c_priv, c_pub = generate_ephemeral_keypair()
    w_priv, w_pub = generate_ephemeral_keypair()
    nonce = generate_challenger_nonce()

    c_keys = derive_session_keys(c_priv, w_pub, nonce)
    w_keys = derive_session_keys(w_priv, c_pub, nonce)

    # 2. Initialize SessionCrypto on both sides
    client_crypto = SessionCrypto(c_keys, role="client", task_id="task-gemm-42")
    worker_crypto = CryptoSession(w_keys, role="worker", task_id="task-gemm-42")

    # 3. Client encrypts Matrix A chunk -> Worker decrypts
    a_tile = np.random.randn(32, 32).astype(np.float32)
    a_bytes = a_tile.tobytes()

    enc_chunk = client_crypto.encrypt_chunk(a_bytes, is_matrix_a=True, sequence_number=0, is_final_chunk=False)
    assert enc_chunk.ciphertext != a_bytes

    dec_a_bytes = worker_crypto.decrypt_chunk(
        enc_chunk.ciphertext, enc_chunk.iv, enc_chunk.tag, is_matrix_a=True, sequence_number=0, is_final_chunk=False
    )
    assert dec_a_bytes == a_bytes
    a_recovered = np.frombuffer(dec_a_bytes, dtype=np.float32).reshape((32, 32))
    np.testing.assert_array_equal(a_recovered, a_tile)

    # 4. Worker executes GEMM C = A @ B and encrypts result -> Client decrypts
    b_tile = np.random.randn(32, 32).astype(np.float32)
    c_tile = a_tile @ b_tile
    c_bytes = c_tile.tobytes()

    enc_result = worker_crypto.encrypt_result(c_bytes, rows=32, cols=32)
    assert enc_result.ciphertext != c_bytes

    dec_c_bytes = client_crypto.decrypt_result(enc_result.ciphertext, enc_result.iv, enc_result.tag, rows=32, cols=32)
    assert dec_c_bytes == c_bytes
    c_recovered = np.frombuffer(dec_c_bytes, dtype=np.float32).reshape((32, 32))
    np.testing.assert_allclose(c_recovered, c_tile, rtol=1e-5)


def test_session_crypto_invalid_role():
    keys = derive_session_keys(secrets.token_bytes(32), secrets.token_bytes(32))
    with pytest.raises(ValueError, match="Role must be 'client' or 'worker'"):
        SessionCrypto(keys, role="observer")
