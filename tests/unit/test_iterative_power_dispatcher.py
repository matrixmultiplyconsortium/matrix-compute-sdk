"""
Unit and integration tests for stateful iterative P2P gRPC Task Dispatcher methods.
Tests:
1. GrpcTaskDispatcher.lease_matrix (FP32, FP16, BF16, rejection, timeouts).
2. GrpcTaskDispatcher.step_iterative (plaintext, signature verification, omitted signatures,
   IterativeStepResponseWrapper.result_vec property, tuple unpacking, confidential AES-GCM mode).
3. GrpcTaskDispatcher.release_lease (explicit memory reclamation, zeroization).
4. GrpcTaskDispatcher._verify_iterative_step_signature (canonical formats, tampering, edge cases).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Generator, Optional
import numpy as np
import pytest

try:
    from matrix_contracts.proto import compute_service_pb2 as pb2
except ImportError:
    from matrix_contracts.v1 import pb2

from matrix_compute.crypto.session import (
    DecryptionError,
    SecurityPolicyViolationError,
)
from matrix_compute.exceptions import (
    StreamingError,
    WorkerConnectionError,
    WorkerSignatureError,
    WorkerTaskRejectedError,
)
from matrix_compute.p2p.dispatcher import (
    GrpcTaskDispatcher,
    IterativeSessionContext,
    IterativeStepResponseWrapper,
)
from matrix_compute.test_doubles.mock_worker import (
    MockWorkerNode,
    _InternalWorkerSigner,
)
from matrix_compute.tiling.precision import truncate_to_bf16


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_worker() -> Generator[MockWorkerNode, None, None]:
    """Yields a running mock worker node with stateful session leasing support."""
    worker = MockWorkerNode(vram_mb=8192)
    worker.start()
    yield worker
    worker.stop()


@pytest.fixture
def dispatcher() -> GrpcTaskDispatcher:
    """Yields a task dispatcher instance."""
    return GrpcTaskDispatcher(
        consumer_pubkey="02" + "11" * 32,
        connect_timeout=2.0,
        task_timeout=5.0,
    )


# ==============================================================================
# Suite 1: IterativeStepResponseWrapper Property & Unpacking Tests
# ==============================================================================

class TestIterativeStepResponseWrapper:
    """Verifies that IterativeStepResponseWrapper exposes .result_vec without AttributeError."""

    def test_result_vec_property_access(self):
        """Asserts .result_vec property returns underlying np.ndarray."""
        res_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        resp = pb2.IterativeStepResponse(
            lease_id="test_lease",
            step_index=0,
            result_vector_bytes=res_arr.tobytes(),
            vector_length=4,
            worker_signature="sig123",
            execution_time_ms=5,
        )
        wrapper = IterativeStepResponseWrapper(response=resp, result_vec=res_arr)

        # Direct property access: MUST NOT raise AttributeError
        assert hasattr(wrapper, "result_vec")
        np.testing.assert_array_equal(wrapper.result_vec, res_arr)

    def test_attribute_delegation(self):
        """Asserts protobuf attributes are transparently delegated via __getattr__."""
        res_arr = np.array([5.0, 6.0], dtype=np.float32)
        resp = pb2.IterativeStepResponse(
            lease_id="test_lease_42",
            step_index=7,
            result_vector_bytes=res_arr.tobytes(),
            vector_length=2,
            worker_signature="sig_valid_hex",
            execution_time_ms=12,
        )
        wrapper = IterativeStepResponseWrapper(response=resp, result_vec=res_arr)

        assert wrapper.lease_id == "test_lease_42"
        assert wrapper.step_index == 7
        assert wrapper.vector_length == 2
        assert wrapper.worker_signature == "sig_valid_hex"
        assert wrapper.execution_time_ms == 12

    def test_tuple_unpacking_and_indexing(self):
        """Asserts wrapper can be unpacked as (vec, resp) tuple and supports 0/1 indexing."""
        res_arr = np.array([10.0, 20.0], dtype=np.float32)
        resp = pb2.IterativeStepResponse(
            lease_id="lease_tuple",
            step_index=1,
            result_vector_bytes=res_arr.tobytes(),
            vector_length=2,
        )
        wrapper = IterativeStepResponseWrapper(response=resp, result_vec=res_arr)

        # Tuple unpacking
        vec, raw_resp = wrapper
        np.testing.assert_array_equal(vec, res_arr)
        assert raw_resp.lease_id == "lease_tuple"

        # Indexing
        np.testing.assert_array_equal(wrapper[0], res_arr)
        assert wrapper[1] is resp

        with pytest.raises(IndexError):
            _ = wrapper[2]


# ==============================================================================
# Suite 2: Dispatcher lease_matrix Tests
# ==============================================================================

class TestDispatcherLeaseMatrix:
    """Verifies GrpcTaskDispatcher.lease_matrix over live gRPC loopback."""

    def test_lease_matrix_fp32_success(self, mock_worker: MockWorkerNode, dispatcher: GrpcTaskDispatcher):
        """Tests successful FP32 matrix lease negotiation."""
        n = 8
        mat = np.random.randn(n, n).astype(np.float32)
        lease_id = "lease_fp32_01"

        resp = dispatcher.lease_matrix(
            worker_or_target=mock_worker.multiaddr,
            lease_id=lease_id,
            matrix_a=mat,
            dtype="fp32",
            ttl_seconds=1800,
            worker_pubkey=mock_worker.pubkey,
        )

        assert resp.lease_id == lease_id
        assert resp.is_accepted is True
        assert resp.expires_at_unix_ms > int(time.time() * 1000)
        assert resp.worker_pubkey == mock_worker.pubkey
        assert mock_worker.servicer.lease_calls == 1

    def test_lease_matrix_fp16_and_bf16(self, mock_worker: MockWorkerNode, dispatcher: GrpcTaskDispatcher):
        """Tests FP16 and BF16 matrix lease upload."""
        n = 4
        # FP16
        mat16 = (np.random.randn(n, n) * 0.1).astype(np.float16)
        resp16 = dispatcher.lease_matrix(
            worker_or_target=mock_worker.multiaddr,
            lease_id="lease_fp16",
            matrix_a=mat16,
            dtype="fp16",
        )
        assert resp16.is_accepted is True

        # BF16
        mat_bf = truncate_to_bf16((np.random.randn(n, n) * 0.1).astype(np.float32))
        resp_bf = dispatcher.lease_matrix(
            worker_or_target=mock_worker.multiaddr,
            lease_id="lease_bf16",
            matrix_a=mat_bf,
            dtype="bf16",
        )
        assert resp_bf.is_accepted is True

    def test_lease_matrix_rejection(self, mock_worker: MockWorkerNode, dispatcher: GrpcTaskDispatcher):
        """Tests worker returning is_accepted=False raises WorkerTaskRejectedError."""
        mock_worker.servicer.reject_lease = True
        mock_worker.servicer.lease_reject_message = "VRAM exhausted"

        mat = np.eye(4, dtype=np.float32)
        with pytest.raises(WorkerTaskRejectedError) as exc_info:
            dispatcher.lease_matrix(
                worker_or_target=mock_worker.multiaddr,
                lease_id="lease_rejected",
                matrix_a=mat,
                dtype="fp32",
            )
        assert "VRAM exhausted" in str(exc_info.value)
        assert exc_info.value.task_id == "lease_rejected"

    def test_lease_matrix_connection_error(self, dispatcher: GrpcTaskDispatcher):
        """Tests unreachable endpoint raises WorkerConnectionError."""
        mat = np.eye(4, dtype=np.float32)
        with pytest.raises(WorkerConnectionError):
            dispatcher.lease_matrix(
                worker_or_target="127.0.0.1:59998",
                lease_id="lease_unreachable",
                matrix_a=mat,
                dtype="fp32",
                timeout=0.5,
            )


# ==============================================================================
# Suite 3: Dispatcher step_iterative Tests
# ==============================================================================

class TestDispatcherStepIterative:
    """Verifies GrpcTaskDispatcher.step_iterative streaming operations."""

    def test_step_iterative_nominal_execution(self, mock_worker: MockWorkerNode, dispatcher: GrpcTaskDispatcher):
        """Tests standard step execution, signature verification, and result_vec property."""
        n = 8
        mat = np.random.randn(n, n).astype(np.float32)
        v0 = np.random.randn(n).astype(np.float32)
        lease_id = "lease_step_nominal"

        # 1. Lease matrix
        lease_resp = dispatcher.lease_matrix(
            worker_or_target=mock_worker.multiaddr,
            lease_id=lease_id,
            matrix_a=mat,
            dtype="fp32",
            worker_pubkey=mock_worker.pubkey,
        )
        assert lease_resp.is_accepted is True

        # 2. Step 0
        step_resp = dispatcher.step_iterative(
            worker_or_target=mock_worker.multiaddr,
            lease_id=lease_id,
            step_index=0,
            vector_v=v0,
            dtype="fp32",
            worker_pubkey=mock_worker.pubkey,
        )

        # Verify IterativeStepResponseWrapper
        assert isinstance(step_resp, IterativeStepResponseWrapper)
        assert hasattr(step_resp, "result_vec")
        expected_v1 = np.matmul(mat, v0)
        np.testing.assert_allclose(step_resp.result_vec, expected_v1, rtol=1e-5, atol=1e-5)
        assert step_resp.step_index == 0
        assert bool(step_resp.worker_signature) is True

    def test_step_iterative_omitted_signature_rejected_fail_closed(
        self,
        mock_worker: MockWorkerNode,
        dispatcher: GrpcTaskDispatcher,
    ):
        """
        Tests that when worker_pubkey is set, an empty worker_signature raises
        WorkerSignatureError fail-closed rather than bypassing cryptographic verification.
        """
        n = 4
        mat = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)
        lease_id = "lease_omitted_sig"

        dispatcher.lease_matrix(
            worker_or_target=mock_worker.multiaddr,
            lease_id=lease_id,
            matrix_a=mat,
            dtype="fp32",
        )

        # Inject omitted signature by intercepting servicer response
        mock_worker.servicer.tamper_step_signature = True
        mock_worker.servicer.tamper_step_index = 0

        # Replace signer to yield empty signature
        original_sign = mock_worker.signer.sign_message
        mock_worker.signer.sign_message = lambda data: ""

        try:
            with pytest.raises(WorkerSignatureError) as exc_info:
                dispatcher.step_iterative(
                    worker_or_target=mock_worker.multiaddr,
                    lease_id=lease_id,
                    step_index=0,
                    vector_v=v0,
                    dtype="fp32",
                    worker_pubkey=mock_worker.pubkey,
                )
            assert "omitted" in str(exc_info.value).lower()
        finally:
            mock_worker.signer.sign_message = original_sign

    def test_step_iterative_forged_signature_rejected(
        self,
        mock_worker: MockWorkerNode,
        dispatcher: GrpcTaskDispatcher,
    ):
        """Tests that a forged signature from an attacker key raises WorkerSignatureError."""
        n = 4
        mat = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)
        lease_id = "lease_forged_sig"

        dispatcher.lease_matrix(
            worker_or_target=mock_worker.multiaddr,
            lease_id=lease_id,
            matrix_a=mat,
            dtype="fp32",
        )

        # Tamper signature with garbage hex
        mock_worker.servicer.tamper_step_signature = True
        mock_worker.servicer.tamper_step_index = 0

        with pytest.raises(WorkerSignatureError) as exc_info:
            dispatcher.step_iterative(
                worker_or_target=mock_worker.multiaddr,
                lease_id=lease_id,
                step_index=0,
                vector_v=v0,
                dtype="fp32",
                worker_pubkey=mock_worker.pubkey,
            )
        assert "failed" in str(exc_info.value).lower()

    def test_step_iterative_byte_count_mismatch(
        self,
        mock_worker: MockWorkerNode,
        dispatcher: GrpcTaskDispatcher,
    ):
        """Tests vector byte count mismatch against expected length raises StreamingError."""
        n = 4
        mat = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)
        lease_id = "lease_len_mismatch"

        dispatcher.lease_matrix(
            worker_or_target=mock_worker.multiaddr,
            lease_id=lease_id,
            matrix_a=mat,
            dtype="fp32",
        )

        # Mock the stub to return truncated vector bytes
        stub = mock_worker.get_stub()
        orig_step = stub.StepIterative

        def fake_step(req, **kwargs):
            resp = orig_step(req, **kwargs)
            # Truncate result bytes to half size
            return pb2.IterativeStepResponse(
                lease_id=resp.lease_id,
                step_index=resp.step_index,
                result_vector_bytes=resp.result_vector_bytes[:4],  # 1 float instead of 4
                vector_length=n,
                worker_signature=resp.worker_signature,
            )

        mock_worker.get_stub = lambda: type("FakeStub", (), {"StepIterative": staticmethod(fake_step)})()

        # Channel pool cache must be bypassed or channel used directly
        # Test directly with mock channel pool or stub
        pool_stub = dispatcher.channel_pool.get_stub(mock_worker.target)
        original_pool_step = pool_stub.StepIterative
        pool_stub.StepIterative = fake_step

        try:
            with pytest.raises(StreamingError) as exc_info:
                dispatcher.step_iterative(
                    worker_or_target=mock_worker.multiaddr,
                    lease_id=lease_id,
                    step_index=0,
                    vector_v=v0,
                    dtype="fp32",
                )
            assert "byte count" in str(exc_info.value).lower()
        finally:
            pool_stub.StepIterative = original_pool_step


# ==============================================================================
# Suite 4: Dispatcher release_lease Tests
# ==============================================================================

class TestDispatcherReleaseLease:
    """Verifies GrpcTaskDispatcher.release_lease memory deallocation."""

    def test_release_lease_success(self, mock_worker: MockWorkerNode, dispatcher: GrpcTaskDispatcher):
        """Tests successful lease deallocation and zeroization."""
        n = 8
        mat = np.eye(n, dtype=np.float32)
        lease_id = "lease_release_success"

        dispatcher.lease_matrix(
            worker_or_target=mock_worker.multiaddr,
            lease_id=lease_id,
            matrix_a=mat,
            dtype="fp32",
        )
        assert lease_id in mock_worker.servicer._pinned_sessions

        rel_resp = dispatcher.release_lease(
            worker_or_target=mock_worker.multiaddr,
            lease_id=lease_id,
            reason="completed",
        )

        assert rel_resp.is_released is True
        assert rel_resp.bytes_freed > 0
        assert mock_worker.servicer.release_calls == 1
        assert lease_id not in mock_worker.servicer._pinned_sessions


# ==============================================================================
# Suite 5: Dispatcher _verify_iterative_step_signature Unit Tests
# ==============================================================================

class TestDispatcherSignatureVerification:
    """Verifies Secp256k1 cryptographic signature validation in dispatcher."""

    def test_canonical_format_1_hex_digest(self, dispatcher: GrpcTaskDispatcher):
        """Tests signature verification on payload f'{lease_id}:{step_index}:{sha256_hex}'."""
        signer = _InternalWorkerSigner()
        pubkey = signer.get_pubkey_hex()
        lease_id = "lease_sig_1"
        step_index = 3
        res_bytes = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()

        r_hash_hex = hashlib.sha256(res_bytes).hexdigest()
        payload = f"{lease_id}:{step_index}:{r_hash_hex}".encode("utf-8")
        sig = signer.sign_message(hashlib.sha256(payload).digest())

        valid = dispatcher._verify_iterative_step_signature(
            worker_pubkey=pubkey,
            lease_id=lease_id,
            step_index=step_index,
            result_bytes=res_bytes,
            signature=sig,
        )
        assert valid is True

    def test_canonical_format_2_binary_digest(self, dispatcher: GrpcTaskDispatcher):
        """Tests signature verification on payload f'{lease_id}:{step_index}:' + raw_sha256_bytes."""
        signer = _InternalWorkerSigner()
        pubkey = signer.get_pubkey_hex()
        lease_id = "lease_sig_2"
        step_index = 5
        res_bytes = np.array([4.0, 5.0, 6.0], dtype=np.float32).tobytes()

        r_hash_digest = hashlib.sha256(res_bytes).digest()
        payload = f"{lease_id}:{step_index}:".encode("utf-8") + r_hash_digest
        sig = signer.sign_message(payload)

        valid = dispatcher._verify_iterative_step_signature(
            worker_pubkey=pubkey,
            lease_id=lease_id,
            step_index=step_index,
            result_bytes=res_bytes,
            signature=sig,
        )
        assert valid is True

    def test_signature_tampered_bytes_rejection(self, dispatcher: GrpcTaskDispatcher):
        """Tests that modifying result bytes invalidates signature."""
        signer = _InternalWorkerSigner()
        pubkey = signer.get_pubkey_hex()
        lease_id = "lease_tampered"
        step_index = 1
        res_bytes = np.array([1.0, 2.0], dtype=np.float32).tobytes()

        r_hash_hex = hashlib.sha256(res_bytes).hexdigest()
        payload = f"{lease_id}:{step_index}:{r_hash_hex}".encode("utf-8")
        sig = signer.sign_message(hashlib.sha256(payload).digest())

        # Alter result bytes
        bad_bytes = np.array([1.0, 2.0001], dtype=np.float32).tobytes()
        valid = dispatcher._verify_iterative_step_signature(
            worker_pubkey=pubkey,
            lease_id=lease_id,
            step_index=step_index,
            result_bytes=bad_bytes,
            signature=sig,
        )
        assert valid is False

    def test_signature_empty_parameters(self, dispatcher: GrpcTaskDispatcher):
        """Tests empty pubkey or signature returns False."""
        res_bytes = b"sample_bytes"
        assert dispatcher._verify_iterative_step_signature("", "l1", 0, res_bytes, "sig") is False
        assert dispatcher._verify_iterative_step_signature("02" + "00" * 32, "l1", 0, res_bytes, "") is False

    def test_confidential_step_signature_verified_over_plaintext(self, dispatcher: GrpcTaskDispatcher):
        """
        Tests that in confidential mode, where worker returns encrypted ciphertext (iv, tag, ciphertext),
        dispatcher decrypts the ciphertext and validates worker_signature over the decrypted plaintext,
        rather than attempting to validate over the ciphertext.
        """
        import secrets
        from matrix_compute.crypto.session import aes_gcm_encrypt

        signer = _InternalWorkerSigner()
        pubkey = signer.get_pubkey_hex()
        lease_id = "lease_confidential_sig"
        step_index = 0

        # Create session with shared keys
        key_c2w = secrets.token_bytes(32)
        key_w2c = secrets.token_bytes(32)
        session = IterativeSessionContext(
            lease_id=lease_id,
            rows=3,
            cols=3,
            dtype="fp32",
            is_confidential=True,
            k_c2w=key_c2w,
            k_w2c=key_w2c,
        )
        with dispatcher._sessions_lock:
            dispatcher._iterative_sessions[lease_id] = session

        # Plaintext result
        plaintext_vec = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        plaintext_bytes = plaintext_vec.tobytes()

        # Worker signs plaintext
        r_hash_hex = hashlib.sha256(plaintext_bytes).hexdigest()
        payload = f"{lease_id}:{step_index}:{r_hash_hex}".encode("utf-8")
        worker_sig = signer.sign_message(hashlib.sha256(payload).digest())

        # Encrypt with worker-to-client key
        step_resp_aad = f"{lease_id}:step:{step_index}".encode("utf-8")
        enc = aes_gcm_encrypt(key=key_w2c, plaintext=plaintext_bytes, aad=step_resp_aad)

        # Mock stub
        mock_response = pb2.IterativeStepResponse(
            lease_id=lease_id,
            step_index=step_index,
            result_vector_bytes=enc.ciphertext,  # AES-GCM ciphertext
            vector_length=3,
            worker_signature=worker_sig,
            execution_time_ms=10,
            is_encrypted=True,
            iv=enc.iv,
            tag=enc.tag,
        )

        fake_stub = type("FakeStub", (), {"StepIterative": lambda *args, **kwargs: mock_response})()
        orig_get_stub = dispatcher.channel_pool.get_stub
        dispatcher.channel_pool.get_stub = lambda target: fake_stub

        try:
            wrapper = dispatcher.step_iterative(
                worker_or_target="127.0.0.1:50051",
                lease_id=lease_id,
                step_index=step_index,
                vector_v=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                dtype="fp32",
                worker_pubkey=pubkey,
            )

            assert isinstance(wrapper, IterativeStepResponseWrapper)
            np.testing.assert_array_equal(wrapper.result_vec, plaintext_vec)
        finally:
            dispatcher.channel_pool.get_stub = orig_get_stub
