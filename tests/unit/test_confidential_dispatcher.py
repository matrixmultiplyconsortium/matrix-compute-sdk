"""
Unit tests for Confidential Computing Dispatcher (Features F14, F17, F18, F19).

Tests:
1. Ephemeral X25519 handshake and challenger nonce generation in RequestTileTask.
2. Hardware attestation evidence verification fail-closed integration.
3. Transparent outbound chunk encryption with fresh 12-byte IVs and authenticated AAD.
4. Transparent inbound result decryption and Secp256k1 signature validation.
5. Strict security policy enforcement (unencrypted responses rejected in TEE mode).
6. Clean unencrypted fallback when require_confidential_compute=False.
7. Outbound ciphertext wire entropy verification (Shannon entropy H >= 7.8).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock
import numpy as np
import pytest

from matrix_compute.config import ClientConfig
from matrix_compute.crypto.attestation_verifier import AttestationVerifier
from matrix_compute.crypto.pki_fixtures import (
    create_broken_chain_signature_evidence,
    create_valid_attestation_evidence,
)
from matrix_compute.crypto.session import (
    AES_GCM_IV_BYTES,
    AES_GCM_TAG_BYTES,
    DecryptionError,
    KeyExchangeError,
    SecurityPolicyViolationError,
    SessionContext,
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    canonical_chunk_aad,
    canonical_result_aad,
    derive_session_keys,
    generate_ephemeral_keypair,
)
from matrix_compute.crypto.signer import ClientSigner, sha256_digest, sign_message
from matrix_compute.exceptions import (
    AttestationError,
    AttestationEvidenceMalformedError,
    SecurityPolicyViolationError,
    WorkerSignatureError,
    WorkerTaskRejectedError,
)
from matrix_compute.p2p.channel_pool import ChannelPool
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher

try:
    from matrix_contracts.v1 import pb2
except ImportError:
    from matrix_contracts.proto import compute_service_pb2 as pb2


def calculate_shannon_entropy(data: bytes) -> float:
    """Calculates Shannon entropy in bits per byte [0.0 to 8.0]."""
    if not data:
        return 0.0
    freq: dict[int, int] = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    total = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


# ==============================================================================
# Handshake & Attestation Tests
# ==============================================================================

def test_confidential_handshake_sends_keys_and_nonce():
    # Setup mock channel and stub
    mock_channel_pool = MagicMock(spec=ChannelPool)
    mock_stub = MagicMock()
    mock_channel_pool.get_stub.return_value = mock_stub

    dispatcher = GrpcTaskDispatcher(
        channel_pool=mock_channel_pool,
        require_confidential_compute=True,
        required_tee_type="simulated-cca",
    )

    # Prepare worker response
    w_priv, w_pub = generate_ephemeral_keypair()
    captured_request = []

    def mock_request_tile_task(req, timeout=None):
        captured_request.append(req)
        # Create valid attestation evidence bound to challenger nonce and worker pubkey
        evidence = create_valid_attestation_evidence(
            nonce=req.attestation_nonce,
            public_key=w_pub,
        )
        return pb2.TileTaskAck(
            task_id=req.task_id,
            is_ready=True,
            estimated_gflops=1.0,
            worker_pubkey="worker_secp_pubkey",
            message="Ready",
            worker_ephemeral_pubkey=w_pub,
            attestation_evidence=evidence.to_bytes(),
            tee_type="simulated-cca",
        )

    mock_stub.RequestTileTask.side_effect = mock_request_tile_task

    ack = dispatcher.request_tile_task(
        worker_or_target="/ip4/127.0.0.1/tcp/50051",
        task_id="task-tee-01",
        m=64,
        k=64,
        n=64,
        dtype="fp32",
    )

    assert ack.is_ready is True
    assert len(captured_request) == 1
    req = captured_request[0]
    assert req.require_confidential_compute is True
    assert len(req.client_ephemeral_pubkey) == 32
    assert len(req.attestation_nonce) == 32
    assert req.required_tee_type == "simulated-cca"

    # Verify session context is cached and keys are derived
    with dispatcher._sessions_lock:
        session = dispatcher._sessions.get("task-tee-01")
        assert session is not None
        assert session.is_confidential is True
        assert session.k_c2w is not None
        assert session.k_w2c is not None
        assert session.k_c2w != session.k_w2c


def test_confidential_handshake_rejects_missing_pubkey():
    mock_channel_pool = MagicMock(spec=ChannelPool)
    mock_stub = MagicMock()
    mock_channel_pool.get_stub.return_value = mock_stub

    dispatcher = GrpcTaskDispatcher(
        channel_pool=mock_channel_pool,
        require_confidential_compute=True,
    )

    mock_stub.RequestTileTask.return_value = pb2.TileTaskAck(
        task_id="task-err",
        is_ready=True,
        worker_ephemeral_pubkey=b"",  # missing pubkey
        attestation_evidence=b"evidence",
    )

    with pytest.raises(KeyExchangeError, match="Worker did not provide 32-byte ephemeral public key"):
        dispatcher.request_tile_task(
            worker_or_target="/ip4/127.0.0.1/tcp/50051",
            task_id="task-err",
            m=16,
            k=16,
            n=16,
        )


def test_confidential_handshake_rejects_missing_evidence():
    mock_channel_pool = MagicMock(spec=ChannelPool)
    mock_stub = MagicMock()
    mock_channel_pool.get_stub.return_value = mock_stub

    dispatcher = GrpcTaskDispatcher(
        channel_pool=mock_channel_pool,
        require_confidential_compute=True,
    )

    mock_stub.RequestTileTask.return_value = pb2.TileTaskAck(
        task_id="task-err-ev",
        is_ready=True,
        worker_ephemeral_pubkey=b"\x01" * 32,
        attestation_evidence=b"",  # missing evidence
    )

    with pytest.raises(AttestationEvidenceMalformedError, match="empty attestation evidence"):
        dispatcher.request_tile_task(
            worker_or_target="/ip4/127.0.0.1/tcp/50051",
            task_id="task-err-ev",
            m=16,
            k=16,
            n=16,
        )


def test_confidential_handshake_attestation_failure_fails_closed():
    mock_channel_pool = MagicMock(spec=ChannelPool)
    mock_stub = MagicMock()
    mock_channel_pool.get_stub.return_value = mock_stub

    dispatcher = GrpcTaskDispatcher(
        channel_pool=mock_channel_pool,
        require_confidential_compute=True,
    )

    w_priv, w_pub = generate_ephemeral_keypair()

    def mock_request(req, timeout=None):
        # Return broken chain signature evidence
        bad_evidence = create_broken_chain_signature_evidence()
        return pb2.TileTaskAck(
            task_id=req.task_id,
            is_ready=True,
            worker_ephemeral_pubkey=w_pub,
            attestation_evidence=bad_evidence.to_bytes(),
        )

    mock_stub.RequestTileTask.side_effect = mock_request

    with pytest.raises(AttestationError):
        dispatcher.request_tile_task(
            worker_or_target="/ip4/127.0.0.1/tcp/50051",
            task_id="task-tampered-attest",
            m=16,
            k=16,
            n=16,
        )


# ==============================================================================
# Chunk Streaming & Result Decryption Tests
# ==============================================================================

def test_stream_tile_chunks_encrypted_with_high_entropy():
    dispatcher = GrpcTaskDispatcher(chunk_size_bytes=512)
    k_c2w = secrets.token_bytes(32)
    session = SessionContext(
        task_id="task-chunks-01",
        client_ephemeral_priv=None,
        client_ephemeral_pub_bytes=b"",
        attestation_nonce=b"",
        k_c2w=k_c2w,
        is_confidential=True,
    )

    a = np.random.randn(32, 32).astype(np.float32)
    b = np.random.randn(32, 32).astype(np.float32)

    chunks = list(dispatcher.stream_tile_chunks("task-chunks-01", a, b, dtype="fp32", session=session))
    assert len(chunks) > 0

    all_ivs = set()
    for chunk in chunks:
        assert chunk.is_encrypted is True
        assert len(chunk.iv) == AES_GCM_IV_BYTES
        assert len(chunk.tag) == AES_GCM_TAG_BYTES
        assert chunk.iv not in all_ivs
        all_ivs.add(chunk.iv)

        # High entropy check (AES-GCM ciphertext should have Shannon entropy >= 7.8 bits/byte)
        if len(chunk.chunk_bytes) >= 128:
            entropy = calculate_shannon_entropy(chunk.chunk_bytes)
            assert entropy >= 7.8, f"Ciphertext entropy {entropy:.2f} is too low"


def test_collect_tile_results_decrypts_and_verifies_signature():
    dispatcher = GrpcTaskDispatcher(require_confidential_compute=True)
    k_w2c = secrets.token_bytes(32)
    session = SessionContext(
        task_id="task-res-01",
        client_ephemeral_priv=None,
        client_ephemeral_pub_bytes=b"",
        attestation_nonce=b"",
        k_w2c=k_w2c,
        is_confidential=True,
    )

    m, n = 16, 16
    expected_c = np.random.randn(m, n).astype(np.float32)
    c_raw = expected_c.tobytes()

    # Worker encrypts result using K_w2c
    aad = canonical_result_aad("task-res-01", m, n)
    payload = aes_gcm_encrypt(k_w2c, c_raw, aad=aad)

    # Worker signs result
    worker_signer = ClientSigner()
    c_hash = sha256_digest(c_raw)
    worker_sig = worker_signer.sign(f"task-res-01:{c_hash}".encode("utf-8"))

    tile_result = pb2.TileResult(
        task_id="task-res-01",
        rows=m,
        cols=n,
        result_bytes=payload.ciphertext,
        worker_signature=worker_sig,
        execution_time_ms=10,
        is_encrypted=True,
        iv=payload.iv,
        tag=payload.tag,
    )

    c_matrix, raw_res = dispatcher.collect_tile_results(
        tile_result=tile_result,
        task_id="task-res-01",
        m=m,
        n=n,
        dtype="fp32",
        worker_pubkey=worker_signer.get_pubkey_hex(),
        session=session,
    )

    np.testing.assert_allclose(c_matrix, expected_c, rtol=1e-5)
    assert raw_res.task_id == "task-res-01"


def test_collect_tile_results_rejects_unencrypted_when_confidential():
    dispatcher = GrpcTaskDispatcher(require_confidential_compute=True)
    session = SessionContext(
        task_id="task-unenc-err",
        client_ephemeral_priv=None,
        client_ephemeral_pub_bytes=b"",
        attestation_nonce=b"",
        k_w2c=secrets.token_bytes(32),
        is_confidential=True,
    )

    tile_result = pb2.TileResult(
        task_id="task-unenc-err",
        rows=8,
        cols=8,
        result_bytes=np.ones((8, 8), dtype=np.float32).tobytes(),
        is_encrypted=False,  # Unencrypted!
    )

    with pytest.raises(SecurityPolicyViolationError, match="Confidential computing was required"):
        dispatcher.collect_tile_results(
            tile_result=tile_result,
            task_id="task-unenc-err",
            m=8,
            n=8,
            session=session,
        )


def test_collect_tile_results_rejects_tampered_ciphertext():
    dispatcher = GrpcTaskDispatcher(require_confidential_compute=True)
    k_w2c = secrets.token_bytes(32)
    session = SessionContext(
        task_id="task-tamper-res",
        client_ephemeral_priv=None,
        client_ephemeral_pub_bytes=b"",
        attestation_nonce=b"",
        k_w2c=k_w2c,
        is_confidential=True,
    )

    m, n = 8, 8
    c_raw = np.ones((m, n), dtype=np.float32).tobytes()
    aad = canonical_result_aad("task-tamper-res", m, n)
    payload = aes_gcm_encrypt(k_w2c, c_raw, aad=aad)

    # Flip bit in ciphertext
    bad_ct = bytes([payload.ciphertext[0] ^ 0x01]) + payload.ciphertext[1:]

    tile_result = pb2.TileResult(
        task_id="task-tamper-res",
        rows=m,
        cols=n,
        result_bytes=bad_ct,
        is_encrypted=True,
        iv=payload.iv,
        tag=payload.tag,
    )

    with pytest.raises(DecryptionError, match="tag verification failed"):
        dispatcher.collect_tile_results(
            tile_result=tile_result,
            task_id="task-tamper-res",
            m=m,
            n=n,
            session=session,
        )


# ==============================================================================
# Unencrypted Fallback Tests (F19)
# ==============================================================================

def test_dispatcher_unencrypted_fallback():
    mock_channel_pool = MagicMock(spec=ChannelPool)
    mock_stub = MagicMock()
    mock_channel_pool.get_stub.return_value = mock_stub

    dispatcher = GrpcTaskDispatcher(
        channel_pool=mock_channel_pool,
        require_confidential_compute=False,
    )

    captured_request = []

    def mock_req(req, timeout=None):
        captured_request.append(req)
        return pb2.TileTaskAck(
            task_id=req.task_id,
            is_ready=True,
            worker_pubkey="mock_worker_pubkey",
            message="Ready",
        )

    mock_stub.RequestTileTask.side_effect = mock_req

    ack = dispatcher.request_tile_task(
        worker_or_target="/ip4/127.0.0.1/tcp/50051",
        task_id="task-plain-01",
        m=16,
        k=16,
        n=16,
    )

    assert ack.is_ready is True
    assert captured_request[0].require_confidential_compute is False
    assert captured_request[0].client_ephemeral_pubkey == b""

    # Verify stream_tile_chunks yields unencrypted chunks
    a = np.ones((16, 16), dtype=np.float32)
    b = np.ones((16, 16), dtype=np.float32)

    chunks = list(dispatcher.stream_tile_chunks("task-plain-01", a, b, dtype="fp32", session=None))
    for chunk in chunks:
        assert chunk.is_encrypted is False
        assert chunk.iv == b""
        assert chunk.tag == b""
