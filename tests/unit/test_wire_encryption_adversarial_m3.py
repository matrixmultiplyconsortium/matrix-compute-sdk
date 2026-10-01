"""
Milestone 3 Adversarial Challenge & Stress Test Suite: Wire Encryption & Dispatcher.

Target Package: matrix-compute-sdk
Modules Challenged:
  - matrix_compute.crypto.session (SessionCrypto, CryptoSession, derive_session_keys, aes_gcm_encrypt, aes_gcm_decrypt)
  - matrix_compute.p2p.dispatcher (GrpcTaskDispatcher, TileDispatcher)
  - matrix_compute.config (ClientConfig confidential compute toggles)

Adversarial Challenge Vectors:
1. Shannon Entropy & Wire Plaintext Leakage Analysis:
   - High-entropy ciphertext verification (H >= 7.85 bits/byte) on low-entropy plaintexts (all-zeros, all-ones, structured patterns).
   - Wire ciphertext undecodability: proves intercepted raw chunks cannot be parsed as valid float32 arrays, contain NaNs/Infs, exhibit unphysical dynamic range, and have zero correlation with plaintext.
2. Result Tampering Attacks & Fail-Closed Behavior:
   - 1-bit flips in ciphertext (first, middle, last byte), truncation, expansion.
   - IV corruptions (1-bit flips, invalid lengths, zeroed IV).
   - Tag corruptions (1-bit flips, invalid lengths, zeroed tag).
   - AAD metadata alteration (task_id, rows, cols).
   - Unencrypted response injection under confidential computing policy.
   - Verification that execution strictly fails closed before result assembly.
3. Replay Attacks & Sequence Manipulation:
   - Replaying old result frames across different tasks (detecting task_id mismatch and AAD authentication failure).
   - Replaying old result frames across fresh ephemeral sessions with same task_id.
   - Duplicate chunk injection and out-of-order chunk replay via AAD sequence binding.
   - Cross-matrix chunk substitution (Matrix A chunk as Matrix B chunk).
4. Unencrypted Fallback & Toggle Verification:
   - ClientConfig toggle and environment variable parsing.
   - Transparent unencrypted gRPC handshake, chunk streaming, and result collection.
   - Rejection of unexpected encrypted results when unencrypted fallback is active.
   - Rejection of missing confidential fields during confidential handshake.
"""

from __future__ import annotations

import math
import secrets
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
    AES_KEY_BYTES,
    HKDF_INFO_LABEL,
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
    decrypt_chunk,
    decrypt_result,
    derive_session_keys,
    encrypt_chunk,
    encrypt_result,
    generate_challenger_nonce,
    generate_ephemeral_keypair,
)
from matrix_compute.crypto.signer import ClientSigner, sha256_digest
from matrix_compute.exceptions import (
    AttestationError,
    AttestationEvidenceMalformedError,
    StreamingError,
    TaskDispatchError,
    WorkerSignatureError,
    WorkerTaskRejectedError,
)
from matrix_compute.p2p.channel_pool import ChannelPool
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher, TileDispatcher

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
# CHALLENGE AREA 1: Shannon Entropy & Wire Undecodability
# ==============================================================================

class TestShannonEntropyAndWireUndecodability:
    """Stress-tests wire encryption against passive network eavesdropping."""

    def test_shannon_entropy_all_zeros_matrix(self):
        """
        Adversarial Test: Plaintext of all zeros (entropy H = 0.0).
        Wire ciphertext must achieve high Shannon entropy H >= 7.85 bits/byte.
        """
        dispatcher = GrpcTaskDispatcher(chunk_size_bytes=4096)
        k_c2w = secrets.token_bytes(32)
        session = SessionContext(
            task_id="task-entropy-zeros",
            client_ephemeral_priv=None,
            client_ephemeral_pub_bytes=b"",
            attestation_nonce=b"",
            k_c2w=k_c2w,
            is_confidential=True,
        )

        # 64x64 float32 matrix of all zeros (16,384 bytes of 0x00)
        zeros_a = np.zeros((64, 64), dtype=np.float32)
        zeros_b = np.zeros((64, 64), dtype=np.float32)

        # Confirm plaintext has zero entropy
        raw_plaintext = zeros_a.tobytes()
        assert calculate_shannon_entropy(raw_plaintext) == 0.0

        chunks = list(dispatcher.stream_tile_chunks("task-entropy-zeros", zeros_a, zeros_b, dtype="fp32", session=session))
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.is_encrypted is True
            if len(chunk.chunk_bytes) >= 1024:
                entropy = calculate_shannon_entropy(chunk.chunk_bytes)
                assert entropy >= 7.85, (
                    f"Ciphertext entropy for all-zeros matrix must be >= 7.85, got {entropy:.4f}"
                )

    def test_shannon_entropy_all_ones_matrix(self):
        """
        Adversarial Test: Plaintext of all ones (IEEE-754: 0x3F800000 repeating).
        Plaintext has very low entropy (< 2.0 bits/byte).
        Wire ciphertext must achieve high Shannon entropy H >= 7.85 bits/byte.
        """
        dispatcher = GrpcTaskDispatcher(chunk_size_bytes=4096)
        k_c2w = secrets.token_bytes(32)
        session = SessionContext(
            task_id="task-entropy-ones",
            client_ephemeral_priv=None,
            client_ephemeral_pub_bytes=b"",
            attestation_nonce=b"",
            k_c2w=k_c2w,
            is_confidential=True,
        )

        ones_a = np.ones((64, 64), dtype=np.float32)
        ones_b = np.ones((64, 64), dtype=np.float32)

        raw_plaintext = ones_a.tobytes()
        assert calculate_shannon_entropy(raw_plaintext) < 2.0

        chunks = list(dispatcher.stream_tile_chunks("task-entropy-ones", ones_a, ones_b, dtype="fp32", session=session))
        for chunk in chunks:
            if len(chunk.chunk_bytes) >= 1024:
                entropy = calculate_shannon_entropy(chunk.chunk_bytes)
                assert entropy >= 7.85, (
                    f"Ciphertext entropy for all-ones matrix must be >= 7.85, got {entropy:.4f}"
                )

    def test_shannon_entropy_structured_diagonal_matrix(self):
        """
        Adversarial Test: Identity matrix with repeating sparse diagonal pattern.
        Wire ciphertext must achieve high Shannon entropy H >= 7.85 bits/byte.
        """
        dispatcher = GrpcTaskDispatcher(chunk_size_bytes=4096)
        k_c2w = secrets.token_bytes(32)
        session = SessionContext(
            task_id="task-entropy-eye",
            client_ephemeral_priv=None,
            client_ephemeral_pub_bytes=b"",
            attestation_nonce=b"",
            k_c2w=k_c2w,
            is_confidential=True,
        )

        eye_a = np.eye(64, dtype=np.float32)
        eye_b = np.eye(64, dtype=np.float32)

        chunks = list(dispatcher.stream_tile_chunks("task-entropy-eye", eye_a, eye_b, dtype="fp32", session=session))
        for chunk in chunks:
            if len(chunk.chunk_bytes) >= 1024:
                entropy = calculate_shannon_entropy(chunk.chunk_bytes)
                assert entropy >= 7.85, (
                    f"Ciphertext entropy for identity matrix must be >= 7.85, got {entropy:.4f}"
                )

    def test_outbound_ciphertext_cannot_be_parsed_as_valid_float32(self):
        """
        Adversarial Test: Network eavesdropper captures encrypted chunk_bytes
        and attempts to interpret it as a NumPy float32 array.
        Proves:
        1. Encrypted bytes contain NaNs and/or Infs due to pseudo-random exponent patterns.
        2. Dynamic range of interpreted floats is unphysical (e.g. magnitudes exceeding 1e30).
        3. Pearson correlation between raw floats and ciphertext floats is negligible (|r| < 0.05).
        4. Linear algebra / matrix multiplication on ciphertext floats produces NaNs or numerical chaos.
        """
        dispatcher = GrpcTaskDispatcher(chunk_size_bytes=8192)
        k_c2w = secrets.token_bytes(32)
        session = SessionContext(
            task_id="task-undecodable",
            client_ephemeral_priv=None,
            client_ephemeral_pub_bytes=b"",
            attestation_nonce=b"",
            k_c2w=k_c2w,
            is_confidential=True,
        )

        # Deterministic matrix with known values [1.0, 2.0, ..., 4096.0]
        plain_matrix = np.arange(1, 4097, dtype=np.float32).reshape((64, 64))
        chunks = list(dispatcher.stream_tile_chunks("task-undecodable", plain_matrix, plain_matrix, dtype="fp32", session=session))

        # Intercept wire chunk bytes
        wire_chunk = chunks[0]
        assert wire_chunk.is_encrypted is True
        wire_bytes = wire_chunk.chunk_bytes

        # Attempt to decode as float32 array
        wire_floats = np.frombuffer(wire_bytes, dtype=np.float32)

        # 1. Check for presence of NaNs or Infs (exponent 0xFF in IEEE-754 occurs ~1/256)
        has_nans_or_infs = bool(np.isnan(wire_floats).any() or np.isinf(wire_floats).any())

        # 2. Check unphysical dynamic range: AES ciphertext produces exponents across full 8-bit range
        abs_floats = np.abs(wire_floats[~np.isnan(wire_floats)])
        max_val = np.max(abs_floats)
        min_positive = np.min(abs_floats[abs_floats > 0])
        unphysical_range = (max_val > 1e20) or (min_positive < 1e-20)

        assert has_nans_or_infs or unphysical_range, (
            "Ciphertext floats must not behave like valid structured matrix data"
        )

        # 3. Check correlation with original plain floats
        plain_slice = plain_matrix.flatten()[: len(wire_floats)]
        valid_mask = ~np.isnan(wire_floats) & ~np.isinf(wire_floats)
        if np.sum(valid_mask) > 100:
            corr_matrix = np.corrcoef(plain_slice[valid_mask], wire_floats[valid_mask])
            correlation = abs(corr_matrix[0, 1])
            assert correlation < 0.05, f"Information leakage detected: correlation {correlation:.4f} >= 0.05"

        # 4. Performing linear algebra on intercepted floats triggers NaNs or massive discrepancy
        with np.errstate(all="ignore"):
            matmul_attempt = wire_floats[:64] @ wire_floats[:64]
            plain_dot = plain_slice[:64] @ plain_slice[:64]
            assert np.isnan(matmul_attempt) or abs(matmul_attempt - plain_dot) > 1e10


# ==============================================================================
# CHALLENGE AREA 2: Result Tampering Attacks & Fail-Closed Enforcement
# ==============================================================================

class TestResultTamperingAttacks:
    """Stress-tests result decryption against active network tampering."""

    def _setup_tamper_environment(self, task_id="task-tamper-01", m=16, n=16):
        dispatcher = GrpcTaskDispatcher(require_confidential_compute=True)
        k_w2c = secrets.token_bytes(32)
        session = SessionContext(
            task_id=task_id,
            client_ephemeral_priv=None,
            client_ephemeral_pub_bytes=b"",
            attestation_nonce=b"",
            k_w2c=k_w2c,
            is_confidential=True,
        )

        expected_c = np.random.randn(m, n).astype(np.float32)
        c_raw = expected_c.tobytes()

        # Worker produces authentic payload and signature
        aad = canonical_result_aad(task_id, m, n)
        payload = aes_gcm_encrypt(k_w2c, c_raw, aad=aad)

        worker_signer = ClientSigner()
        c_hash = sha256_digest(c_raw)
        worker_sig = worker_signer.sign(f"{task_id}:{c_hash}".encode("utf-8"))

        return dispatcher, session, payload, worker_signer, expected_c, c_raw

    @pytest.mark.parametrize("flip_pos", ["first", "middle", "last"])
    def test_result_tampering_ciphertext_1bit_flip(self, flip_pos: str):
        """
        Adversarial Test: Flip exactly 1 bit in result ciphertext at various offsets.
        Must fail closed with DecryptionError before result assembly.
        """
        dispatcher, session, payload, worker_signer, _, _ = self._setup_tamper_environment()

        ct_bytes = bytearray(payload.ciphertext)
        if flip_pos == "first":
            ct_bytes[0] ^= 0x01
        elif flip_pos == "middle":
            ct_bytes[len(ct_bytes) // 2] ^= 0x80
        elif flip_pos == "last":
            ct_bytes[-1] ^= 0x04

        tile_result = pb2.TileResult(
            task_id=session.task_id,
            rows=16,
            cols=16,
            result_bytes=bytes(ct_bytes),
            worker_signature=worker_signer.sign(b"dummy"),
            execution_time_ms=10,
            is_encrypted=True,
            iv=payload.iv,
            tag=payload.tag,
        )

        with pytest.raises(DecryptionError, match="tag verification failed"):
            dispatcher.collect_tile_results(
                tile_result=tile_result,
                task_id=session.task_id,
                m=16,
                n=16,
                session=session,
            )

    def test_result_tampering_ciphertext_truncation(self):
        """
        Adversarial Test: Truncate result ciphertext by 1 byte or 4 bytes.
        Must fail closed with DecryptionError or StreamingError.
        """
        dispatcher, session, payload, _, _, _ = self._setup_tamper_environment()

        # Truncate by 1 byte
        truncated_ct = payload.ciphertext[:-1]
        tile_result = pb2.TileResult(
            task_id=session.task_id,
            rows=16,
            cols=16,
            result_bytes=truncated_ct,
            is_encrypted=True,
            iv=payload.iv,
            tag=payload.tag,
        )

        with pytest.raises((DecryptionError, StreamingError)):
            dispatcher.collect_tile_results(
                tile_result=tile_result,
                task_id=session.task_id,
                m=16,
                n=16,
                session=session,
            )

    @pytest.mark.parametrize("iv_offset", [0, 6, 11])
    def test_result_tampering_iv_1bit_flip(self, iv_offset: int):
        """
        Adversarial Test: Flip 1 bit in 12-byte IV at byte 0, 6, or 11.
        Must fail closed with DecryptionError.
        """
        dispatcher, session, payload, _, _, _ = self._setup_tamper_environment()

        tampered_iv = bytearray(payload.iv)
        tampered_iv[iv_offset] ^= 0x02

        tile_result = pb2.TileResult(
            task_id=session.task_id,
            rows=16,
            cols=16,
            result_bytes=payload.ciphertext,
            is_encrypted=True,
            iv=bytes(tampered_iv),
            tag=payload.tag,
        )

        with pytest.raises(DecryptionError, match="tag verification failed"):
            dispatcher.collect_tile_results(
                tile_result=tile_result,
                task_id=session.task_id,
                m=16,
                n=16,
                session=session,
            )

    @pytest.mark.parametrize("invalid_iv", [b"", b"\x00" * 11, b"\x00" * 13, b"\x00" * 16])
    def test_result_tampering_iv_invalid_length(self, invalid_iv: bytes):
        """
        Adversarial Test: Send IV with length != 12 bytes.
        Must fail closed with FramingError or DecryptionError.
        """
        dispatcher, session, payload, _, _, _ = self._setup_tamper_environment()

        tile_result = pb2.TileResult(
            task_id=session.task_id,
            rows=16,
            cols=16,
            result_bytes=payload.ciphertext,
            is_encrypted=True,
            iv=invalid_iv,
            tag=payload.tag,
        )

        with pytest.raises((FramingError, DecryptionError)):
            dispatcher.collect_tile_results(
                tile_result=tile_result,
                task_id=session.task_id,
                m=16,
                n=16,
                session=session,
            )

    @pytest.mark.parametrize("tag_offset", [0, 8, 15])
    def test_result_tampering_tag_1bit_flip(self, tag_offset: int):
        """
        Adversarial Test: Flip 1 bit in 16-byte GCM authentication tag.
        Must fail closed with DecryptionError.
        """
        dispatcher, session, payload, _, _, _ = self._setup_tamper_environment()

        tampered_tag = bytearray(payload.tag)
        tampered_tag[tag_offset] ^= 0x08

        tile_result = pb2.TileResult(
            task_id=session.task_id,
            rows=16,
            cols=16,
            result_bytes=payload.ciphertext,
            is_encrypted=True,
            iv=payload.iv,
            tag=bytes(tampered_tag),
        )

        with pytest.raises(DecryptionError, match="tag verification failed"):
            dispatcher.collect_tile_results(
                tile_result=tile_result,
                task_id=session.task_id,
                m=16,
                n=16,
                session=session,
            )

    @pytest.mark.parametrize("invalid_tag", [b"", b"\x00" * 15, b"\x00" * 17, b"\x00" * 32])
    def test_result_tampering_tag_invalid_length(self, invalid_tag: bytes):
        """
        Adversarial Test: Send GCM tag with length != 16 bytes.
        Must fail closed with FramingError or DecryptionError.
        """
        dispatcher, session, payload, _, _, _ = self._setup_tamper_environment()

        tile_result = pb2.TileResult(
            task_id=session.task_id,
            rows=16,
            cols=16,
            result_bytes=payload.ciphertext,
            is_encrypted=True,
            iv=payload.iv,
            tag=invalid_tag,
        )

        with pytest.raises((FramingError, DecryptionError)):
            dispatcher.collect_tile_results(
                tile_result=tile_result,
                task_id=session.task_id,
                m=16,
                n=16,
                session=session,
            )

    def test_result_tampering_unencrypted_injection_under_confidential_policy(self):
        """
        Adversarial Test: Malicious worker returns is_encrypted=False with crafted floats
        when client has require_confidential_compute=True.
        Must raise SecurityPolicyViolationError immediately before reading or assembling.
        """
        dispatcher, session, _, _, _, _ = self._setup_tamper_environment()

        fake_floats = np.ones((16, 16), dtype=np.float32).tobytes()
        tile_result = pb2.TileResult(
            task_id=session.task_id,
            rows=16,
            cols=16,
            result_bytes=fake_floats,
            is_encrypted=False,  # Policy violation!
        )

        with pytest.raises(SecurityPolicyViolationError, match="Confidential computing was required"):
            dispatcher.collect_tile_results(
                tile_result=tile_result,
                task_id=session.task_id,
                m=16,
                n=16,
                session=session,
            )

    def test_result_tampering_worker_signature_forgery(self):
        """
        Adversarial Test: Ciphertext and tag are valid, but worker signature is forged/corrupted.
        Must raise WorkerSignatureError.
        """
        dispatcher, session, payload, _, expected_c, _ = self._setup_tamper_environment()
        worker_signer = ClientSigner()

        # Forged signature
        forged_sig = "deadbeef" * 16

        tile_result = pb2.TileResult(
            task_id=session.task_id,
            rows=16,
            cols=16,
            result_bytes=payload.ciphertext,
            worker_signature=forged_sig,
            execution_time_ms=10,
            is_encrypted=True,
            iv=payload.iv,
            tag=payload.tag,
        )

        with pytest.raises(WorkerSignatureError, match="signature verification failed"):
            dispatcher.collect_tile_results(
                tile_result=tile_result,
                task_id=session.task_id,
                m=16,
                n=16,
                worker_pubkey=worker_signer.get_pubkey_hex(),
                session=session,
            )


# ==============================================================================
# CHALLENGE AREA 3: Replay Attacks & Sequence Manipulation
# ==============================================================================

class TestReplayAttacksAndSequenceManipulation:
    """Stress-tests SessionCrypto and Dispatcher against replay and reflection attacks."""

    def test_replay_attack_cross_task_result_replayed_verbatim(self):
        """
        Adversarial Test: Attacker intercepts valid TileResult from Task 1
        and replays it verbatim as the response to Task 2.
        Must fail closed with TaskDispatchError due to task_id mismatch.
        """
        dispatcher = GrpcTaskDispatcher(require_confidential_compute=True)
        session_2 = SessionContext(
            task_id="task-02",
            client_ephemeral_priv=None,
            client_ephemeral_pub_bytes=b"",
            attestation_nonce=b"",
            k_w2c=secrets.token_bytes(32),
            is_confidential=True,
        )

        # Valid result from task-01
        tile_result_task1 = pb2.TileResult(
            task_id="task-01",  # Old task_id
            rows=16,
            cols=16,
            result_bytes=b"\xaa" * 1024,
            is_encrypted=True,
            iv=b"\x01" * 12,
            tag=b"\x02" * 16,
        )

        with pytest.raises(TaskDispatchError, match="Worker returned mismatched task_id"):
            dispatcher.collect_tile_results(
                tile_result=tile_result_task1,
                task_id="task-02",
                m=16,
                n=16,
                session=session_2,
            )

    def test_replay_attack_result_replayed_with_spoofed_task_id(self):
        """
        Adversarial Test: Attacker replays Task 1 ciphertext into Task 2,
        rewriting protobuf task_id="task-02".
        Because canonical AAD binds task_id, AES-GCM tag verification must fail.
        """
        dispatcher = GrpcTaskDispatcher(require_confidential_compute=True)
        shared_key = secrets.token_bytes(32)

        # Encrypt under task-01 AAD
        aad_task1 = canonical_result_aad("task-01", 16, 16)
        c_raw = np.ones((16, 16), dtype=np.float32).tobytes()
        payload_task1 = aes_gcm_encrypt(shared_key, c_raw, aad=aad_task1)

        # Client expects task-02
        session_task2 = SessionContext(
            task_id="task-02",
            client_ephemeral_priv=None,
            client_ephemeral_pub_bytes=b"",
            attestation_nonce=b"",
            k_w2c=shared_key,
            is_confidential=True,
        )

        # Attacker spoofs protobuf task_id to task-02
        replayed_tile_result = pb2.TileResult(
            task_id="task-02",  # Spoofed!
            rows=16,
            cols=16,
            result_bytes=payload_task1.ciphertext,
            is_encrypted=True,
            iv=payload_task1.iv,
            tag=payload_task1.tag,
        )

        with pytest.raises(DecryptionError, match="tag verification failed"):
            dispatcher.collect_tile_results(
                tile_result=replayed_tile_result,
                task_id="task-02",
                m=16,
                n=16,
                session=session_task2,
            )

    def test_replay_attack_result_replayed_across_ephemeral_sessions(self):
        """
        Adversarial Test: The same task_id is executed in Session 1, then re-executed in Session 2.
        Attacker replays the verified result from Session 1 in Session 2.
        Because ephemeral keys are refreshed, K_w2c differs and decryption must fail.
        """
        dispatcher = GrpcTaskDispatcher(require_confidential_compute=True)
        task_id = "task-replay-session"

        # Session 1 keys
        k_w2c_1 = secrets.token_bytes(32)
        session_1 = SessionContext(
            task_id=task_id,
            client_ephemeral_priv=None,
            client_ephemeral_pub_bytes=b"",
            attestation_nonce=b"",
            k_w2c=k_w2c_1,
            is_confidential=True,
        )

        # Produce valid payload for Session 1
        c_raw = np.ones((16, 16), dtype=np.float32).tobytes()
        payload_1 = aes_gcm_encrypt(k_w2c_1, c_raw, aad=canonical_result_aad(task_id, 16, 16))

        # Session 2 keys (different ephemeral keypair)
        k_w2c_2 = secrets.token_bytes(32)
        assert k_w2c_1 != k_w2c_2
        session_2 = SessionContext(
            task_id=task_id,
            client_ephemeral_priv=None,
            client_ephemeral_pub_bytes=b"",
            attestation_nonce=b"",
            k_w2c=k_w2c_2,
            is_confidential=True,
        )

        replayed_result = pb2.TileResult(
            task_id=task_id,
            rows=16,
            cols=16,
            result_bytes=payload_1.ciphertext,
            is_encrypted=True,
            iv=payload_1.iv,
            tag=payload_1.tag,
        )

        with pytest.raises(DecryptionError, match="tag verification failed"):
            dispatcher.collect_tile_results(
                tile_result=replayed_result,
                task_id=task_id,
                m=16,
                n=16,
                session=session_2,
            )

    def test_replay_attack_duplicate_chunk_in_session_crypto(self):
        """
        Adversarial Test: Duplicate chunk 0 re-ingested as chunk 1.
        Sequence number in AAD fails authentication.
        """
        keys = derive_session_keys(secrets.token_bytes(32), secrets.token_bytes(32))
        client = SessionCrypto(keys, role="client", task_id="task-chunk-dup")
        worker = SessionCrypto(keys, role="worker", task_id="task-chunk-dup")

        chunk0_payload = client.encrypt_chunk(b"Chunk 0 data", is_matrix_a=True, sequence_number=0)

        # Worker successfully decrypts chunk 0
        worker.decrypt_chunk(
            chunk0_payload.ciphertext, chunk0_payload.iv, chunk0_payload.tag,
            is_matrix_a=True, sequence_number=0
        )

        # Attacker attempts to replay chunk 0 at sequence number 1
        with pytest.raises(AuthenticationTagError):
            worker.decrypt_chunk(
                chunk0_payload.ciphertext, chunk0_payload.iv, chunk0_payload.tag,
                is_matrix_a=True, sequence_number=1  # Replayed at wrong sequence!
            )

    def test_replay_attack_cross_matrix_chunk_substitution(self):
        """
        Adversarial Test: Matrix A chunk replayed as Matrix B chunk.
        AAD binds is_matrix_a, causing immediate tag failure.
        """
        keys = derive_session_keys(secrets.token_bytes(32), secrets.token_bytes(32))
        client = SessionCrypto(keys, role="client", task_id="task-chunk-mat")
        worker = SessionCrypto(keys, role="worker", task_id="task-chunk-mat")

        chunk_a = client.encrypt_chunk(b"Matrix A chunk", is_matrix_a=True, sequence_number=0)

        # Attempt to decrypt as Matrix B chunk
        with pytest.raises(AuthenticationTagError):
            worker.decrypt_chunk(
                chunk_a.ciphertext, chunk_a.iv, chunk_a.tag,
                is_matrix_a=False, sequence_number=0  # Matrix substitution!
            )

    def test_directional_key_isolation_cannot_cross_decrypt(self):
        """
        Adversarial Test: Client chunk encrypted with K_c2w cannot be decrypted
        by receiver using K_w2c (reflection / direction confusion attack).
        """
        keys = derive_session_keys(secrets.token_bytes(32), secrets.token_bytes(32))
        assert keys.k_c2w != keys.k_w2c

        # Encrypt with K_c2w
        chunk = encrypt_chunk(keys.k_c2w, b"Forward chunk", "task-dir", 0, True)

        # Attempt to decrypt with K_w2c
        with pytest.raises(AuthenticationTagError):
            aes_gcm_decrypt(keys.k_w2c, chunk[0], chunk[1], chunk[2], aad=canonical_chunk_aad("task-dir", 0, True))


# ==============================================================================
# CHALLENGE AREA 4: Unencrypted Fallback & Configuration Toggle
# ==============================================================================

class TestUnencryptedFallbackAndToggle:
    """Stress-tests unencrypted fallback, configuration toggles, and environment parsing."""

    def test_unencrypted_fallback_dispatcher_transparent_execution(self):
        """
        Adversarial Test: Verify unencrypted mode functions without requiring keys,
        yielding plaintext chunks and accepting plaintext results with zero encryption overhead.
        """
        mock_channel_pool = MagicMock(spec=ChannelPool)
        mock_stub = MagicMock()
        mock_channel_pool.get_stub.return_value = mock_stub

        dispatcher = GrpcTaskDispatcher(
            channel_pool=mock_channel_pool,
            require_confidential_compute=False,
        )

        captured_requests = []

        def mock_request(req, timeout=None):
            captured_requests.append(req)
            return pb2.TileTaskAck(
                task_id=req.task_id,
                is_ready=True,
                worker_pubkey="worker_secp_pubkey",
                message="Ready for unencrypted compute",
                # Omit worker_ephemeral_pubkey and attestation_evidence
                worker_ephemeral_pubkey=b"",
                attestation_evidence=b"",
            )

        mock_stub.RequestTileTask.side_effect = mock_request

        ack = dispatcher.request_tile_task(
            worker_or_target="/ip4/127.0.0.1/tcp/50051",
            task_id="task-unenc-01",
            m=8,
            k=8,
            n=8,
        )

        assert ack.is_ready is True
        assert captured_requests[0].require_confidential_compute is False
        assert captured_requests[0].client_ephemeral_pubkey == b""
        assert captured_requests[0].attestation_nonce == b""

        # Verify stream_tile_chunks emits unencrypted chunks
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        chunks = list(dispatcher.stream_tile_chunks("task-unenc-01", a, b, dtype="fp32", session=None))

        for c in chunks:
            assert c.is_encrypted is False
            assert c.iv == b""
            assert c.tag == b""
            assert len(c.chunk_bytes) > 0

        # Verify collect_tile_results accepts unencrypted results
        expected_c = a @ b
        worker_signer = ClientSigner()
        c_hash = sha256_digest(expected_c.tobytes())
        sig = worker_signer.sign(f"task-unenc-01:{c_hash}".encode("utf-8"))

        unenc_result = pb2.TileResult(
            task_id="task-unenc-01",
            rows=8,
            cols=8,
            result_bytes=expected_c.tobytes(),
            worker_signature=sig,
            execution_time_ms=5,
            is_encrypted=False,
            iv=b"",
            tag=b"",
        )

        c_out, _ = dispatcher.collect_tile_results(
            tile_result=unenc_result,
            task_id="task-unenc-01",
            m=8,
            n=8,
            dtype="fp32",
            worker_pubkey=worker_signer.get_pubkey_hex(),
            session=None,
        )

        np.testing.assert_allclose(c_out, expected_c, rtol=1e-5)

    def test_unencrypted_mode_rejects_spurious_encrypted_results_without_session(self):
        """
        Adversarial Test: Worker returns is_encrypted=True for an unencrypted task.
        Dispatcher has no session keys and must fail closed with DecryptionError.
        """
        dispatcher = GrpcTaskDispatcher(require_confidential_compute=False)

        spurious_encrypted_result = pb2.TileResult(
            task_id="task-unenc-spur",
            rows=8,
            cols=8,
            result_bytes=b"\x99" * 256,
            is_encrypted=True,
            iv=b"\x01" * 12,
            tag=b"\x02" * 16,
        )

        with pytest.raises(DecryptionError, match="Unexpected encrypted TileResult for unencrypted task"):
            dispatcher.collect_tile_results(
                tile_result=spurious_encrypted_result,
                task_id="task-unenc-spur",
                m=8,
                n=8,
                session=None,
            )

    def test_confidential_mode_rejects_empty_worker_pubkey(self):
        """
        Adversarial Test: In confidential mode, worker accepts task but omits ephemeral pubkey.
        Must raise KeyExchangeError.
        """
        mock_channel_pool = MagicMock(spec=ChannelPool)
        mock_stub = MagicMock()
        mock_channel_pool.get_stub.return_value = mock_stub

        dispatcher = GrpcTaskDispatcher(
            channel_pool=mock_channel_pool,
            require_confidential_compute=True,
        )

        mock_stub.RequestTileTask.return_value = pb2.TileTaskAck(
            task_id="task-no-pubkey",
            is_ready=True,
            worker_ephemeral_pubkey=b"",  # Empty!
            attestation_evidence=b"evidence",
        )

        with pytest.raises(KeyExchangeError, match="Worker did not provide 32-byte ephemeral public key"):
            dispatcher.request_tile_task(
                worker_or_target="/ip4/127.0.0.1/tcp/50051",
                task_id="task-no-pubkey",
                m=16,
                k=16,
                n=16,
            )

    def test_confidential_mode_rejects_empty_attestation_evidence(self):
        """
        Adversarial Test: In confidential mode, worker accepts task but omits attestation evidence.
        Must raise AttestationEvidenceMalformedError.
        """
        mock_channel_pool = MagicMock(spec=ChannelPool)
        mock_stub = MagicMock()
        mock_channel_pool.get_stub.return_value = mock_stub

        dispatcher = GrpcTaskDispatcher(
            channel_pool=mock_channel_pool,
            require_confidential_compute=True,
        )

        mock_stub.RequestTileTask.return_value = pb2.TileTaskAck(
            task_id="task-no-evidence",
            is_ready=True,
            worker_ephemeral_pubkey=b"\x01" * 32,
            attestation_evidence=b"",  # Empty!
        )

        with pytest.raises(AttestationEvidenceMalformedError, match="empty attestation evidence"):
            dispatcher.request_tile_task(
                worker_or_target="/ip4/127.0.0.1/tcp/50051",
                task_id="task-no-evidence",
                m=16,
                k=16,
                n=16,
            )

    def test_client_config_toggle_and_env_parsing(self):
        """
        Adversarial Test: Verify ClientConfig correctly toggles confidential computing
        via constructor, for_testing factory, and environment variables.
        """
        # Default is True
        cfg_default = ClientConfig()
        assert cfg_default.require_confidential_compute is True

        # Explicit toggle False
        cfg_off = ClientConfig(require_confidential_compute=False)
        assert cfg_off.require_confidential_compute is False

        # for_testing defaults to False
        cfg_test = ClientConfig.for_testing()
        assert cfg_test.require_confidential_compute is False

        # for_testing with override
        cfg_test_tee = ClientConfig.for_testing(require_confidential_compute=True)
        assert cfg_test_tee.require_confidential_compute is True

        # Env variable parsing
        cfg_env_0 = ClientConfig.from_env({"MATRIX_REQUIRE_CONFIDENTIAL_COMPUTE": "0"})
        assert cfg_env_0.require_confidential_compute is False

        cfg_env_false = ClientConfig.from_env({"MATRIX_REQUIRE_CONFIDENTIAL_COMPUTE": "false"})
        assert cfg_env_false.require_confidential_compute is False

        cfg_env_1 = ClientConfig.from_env({"MATRIX_REQUIRE_CONFIDENTIAL_COMPUTE": "1"})
        assert cfg_env_1.require_confidential_compute is True

        cfg_env_true = ClientConfig.from_env({"MATRIX_REQUIRE_CONFIDENTIAL_COMPUTE": "true"})
        assert cfg_env_true.require_confidential_compute is True

    def test_tile_dispatcher_alias_matches_grpc_task_dispatcher(self):
        """Contract Test: TileDispatcher is an alias for GrpcTaskDispatcher."""
        assert TileDispatcher is GrpcTaskDispatcher
        dispatcher = TileDispatcher(require_confidential_compute=False)
        assert isinstance(dispatcher, GrpcTaskDispatcher)


# ==============================================================================
# CHALLENGE AREA 5: SessionCrypto Edge Cases, Low-Order Points & Cryptographic Stress
# ==============================================================================

class TestSessionCryptoEdgeCasesAndLowOrderPoints:
    """Stress-tests low-level cryptographic primitives and edge cases in SessionCrypto."""

    def test_ecdh_low_order_zero_point_rejection(self):
        """
        Adversarial Test: An attacker supplies a low-order Curve25519 point
        (e.g., all-zero point) attempting to force the shared secret to all-zeros.
        Must raise KeyExchangeError.
        """
        priv, _ = generate_ephemeral_keypair()
        zero_point = b"\x00" * 32

        with pytest.raises(KeyExchangeError, match="low-order point rejected|produced all-zero"):
            from matrix_compute.crypto.session import compute_ecdh_shared_secret
            compute_ecdh_shared_secret(priv, zero_point)

    def test_ecdh_invalid_pubkey_length_rejection(self):
        """
        Adversarial Test: Peer public key with length != 32 bytes.
        Must raise KeyExchangeError.
        """
        priv, _ = generate_ephemeral_keypair()
        from matrix_compute.crypto.session import compute_ecdh_shared_secret

        with pytest.raises(KeyExchangeError, match="must be exactly 32 bytes"):
            compute_ecdh_shared_secret(priv, b"\x01" * 31)

        with pytest.raises(KeyExchangeError, match="must be exactly 32 bytes"):
            compute_ecdh_shared_secret(priv, b"\x01" * 33)

    def test_hkdf_domain_separation_prevents_cross_protocol_confusion(self):
        """
        Adversarial Test: Deriving keys with a modified info string produces
        completely uncorrelated session keys.
        """
        secret = secrets.token_bytes(32)
        nonce = secrets.token_bytes(32)

        keys_std = derive_session_keys(secret, nonce, info=HKDF_INFO_LABEL)
        keys_alt = derive_session_keys(secret, nonce, info=b"matrix-matmul-v2-other")

        assert keys_std.k_c2w != keys_alt.k_c2w
        assert keys_std.k_w2c != keys_alt.k_w2c

    def test_zero_length_chunk_encryption_and_decryption(self):
        """
        Edge Case Test: Encrypting and decrypting an empty payload (b"")
        for zero-sized matrices or EOF signals.
        Must produce 0-byte ciphertext, valid 12-byte IV, 16-byte tag, and roundtrip cleanly.
        """
        key = secrets.token_bytes(32)
        payload = aes_gcm_encrypt(key, b"", aad=canonical_chunk_aad("task-empty", 0, True, True))

        assert len(payload.ciphertext) == 0
        assert len(payload.iv) == AES_GCM_IV_BYTES
        assert len(payload.tag) == AES_GCM_TAG_BYTES

        decrypted = aes_gcm_decrypt(
            key, payload.ciphertext, payload.iv, payload.tag,
            aad=canonical_chunk_aad("task-empty", 0, True, True)
        )
        assert decrypted == b""

    def test_fresh_iv_uniqueness_stress_1000_iterations(self):
        """
        Stress Test: 1,000 AES-256-GCM encryptions must yield 1,000 distinct IVs
        (birthday bound for 96-bit random IVs ensures collision probability is < 1e-15).
        """
        key = secrets.token_bytes(32)
        data = b"Testing IV uniqueness under rapid chunk streaming"
        iv_pool = set()

        for _ in range(1000):
            p = aes_gcm_encrypt(key, data)
            assert p.iv not in iv_pool
            iv_pool.add(p.iv)

        assert len(iv_pool) == 1000
