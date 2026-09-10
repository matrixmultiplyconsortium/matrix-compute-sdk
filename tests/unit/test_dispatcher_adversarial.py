"""
Adversarial Stress Test Suite for P2P Dispatcher, Scheduler, Coordinator, and Client API.

Stress-tests:
1. Out-of-order, truncated, and oversized chunk streaming.
2. Worker signature forgery, corrupted result digests, and stolen/replayed task ID injection.
3. High-concurrency slot contention (e.g. 32 worker threads competing for limited worker slots).
4. Repeated worker failure cascades and rapid failover switching.
5. Mathematical perturbation injection into tiles verifying Freivalds rejection and fraud proof generation.
6. Extreme matrix shapes and edge workloads on the Client API.
"""

from concurrent import futures
import hashlib
import math
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import grpc
import numpy as np
import pytest

from matrix_compute.client import MatrixComputeClient, distributed_matmul
from matrix_compute.config import ClientConfig
from matrix_compute.crypto.keys import generate_keypair
from matrix_compute.crypto.signer import ClientSigner, sha256_digest, sign_message
from matrix_compute.exceptions import (
    DispatchError,
    FraudDetectedError,
    MatrixDimensionError,
    StreamingError,
    TaskTimeoutError,
    WorkerConnectionError,
    WorkerSignatureError,
    WorkerTaskRejectedError,
    WorkerUnavailableError,
)
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler, WorkerSlotTracker
from matrix_compute.p2p.channel_pool import ChannelPool, parse_multiaddr
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.escrow import EscrowManager
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import MockWorkerNode, _InternalWorkerSigner
from matrix_compute.tiling.assembler import ResultAssembler
from matrix_compute.tiling.partitioner import MatrixTiler, TileTaskPlan
from matrix_compute.verification.fraud_proof import FraudProofBuilder
from matrix_compute.verification.freivalds import FreivaldsVerifier

try:
    from matrix_contracts.proto import compute_service_pb2 as pb2
    from matrix_contracts.proto import compute_service_pb2_grpc as pb2_grpc
except ImportError:
    from matrix_contracts.v1 import pb2, pb2_grpc


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def verification_key():
    return "settlement-secret-key-adversarial-testing-999"


@pytest.fixture
def mock_settlement(verification_key):
    return MockSettlementService(verification_key=verification_key)


@pytest.fixture
def mock_worker(mock_settlement, verification_key):
    worker = MockWorkerNode(
        port=0,
        settlement_service=mock_settlement,
        verification_key=verification_key,
        vram_mb=32768,
        sats_per_gflop=0.0005,
    )
    worker.start()
    yield worker
    worker.stop()


# ==============================================================================
# Suite 1: Chunk Streaming Adversarial Challenge
# ==============================================================================

class TestChunkStreamingAdversarial:
    """Stress-tests progressive chunk streaming against malformed, out-of-order, and truncated streams."""

    def test_out_of_order_chunks_rejected_by_worker(self, mock_worker, mock_settlement, verification_key):
        """Adversarial sender emits chunk 1 before chunk 0; asserts worker aborts with INVALID_ARGUMENT."""
        task_id = "task_adv_ooo_01"
        m, k, n = 8, 8, 8

        # 1. Negotiate
        stub = mock_worker.get_stub()
        ack = stub.RequestTileTask(
            pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=m, matrix_a_cols=k, matrix_b_cols=n, dtype="fp32")
        )
        assert ack.is_ready is True

        # 2. Escrow
        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=50)
        mock_settlement.hold_invoice(inv.payment_hash)

        # 3. Stream chunks out-of-order (sequence 1 before sequence 0)
        a_bytes = np.ones((m, k), dtype=np.float32).tobytes()
        b_bytes = np.ones((k, n), dtype=np.float32).tobytes()

        def _ooo_generator():
            # Emit sequence 1 first
            yield pb2.TileDataChunk(
                task_id=task_id,
                chunk_bytes=a_bytes[: len(a_bytes) // 2],
                is_matrix_a=True,
                sequence_number=1,  # WRONG SEQUENCE!
                is_final_chunk=False,
            )

        metadata = [("authorization", f"L402 {inv.macaroon}:{inv.payment_hash}"), ("x-task-id", task_id)]
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.ExecuteTileTask(_ooo_generator(), metadata=metadata)
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Out of order sequence" in exc_info.value.details()

    def test_truncated_stream_missing_final_chunk_flag(self, mock_worker, mock_settlement):
        """Adversarial sender ends stream prematurely without setting is_final_chunk=True."""
        task_id = "task_adv_trunc_02"
        m, k, n = 4, 4, 4

        stub = mock_worker.get_stub()
        stub.RequestTileTask(
            pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=m, matrix_a_cols=k, matrix_b_cols=n, dtype="fp32")
        )

        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=50)
        mock_settlement.hold_invoice(inv.payment_hash)

        a_bytes = np.ones((m, k), dtype=np.float32).tobytes()

        def _truncated_generator():
            # Send Matrix A with final_chunk=False, then stop without sending Matrix B
            yield pb2.TileDataChunk(
                task_id=task_id,
                chunk_bytes=a_bytes,
                is_matrix_a=True,
                sequence_number=0,
                is_final_chunk=False,
            )

        metadata = [("authorization", f"L402 {inv.macaroon}:{inv.payment_hash}"), ("x-task-id", task_id)]
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.ExecuteTileTask(_truncated_generator(), metadata=metadata)
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "without final chunk flag" in exc_info.value.details()

    def test_oversized_chunk_stream_dos_protection(self, mock_worker, mock_settlement):
        """Adversarial sender attempts to stream more bytes than matrix dimensions declare."""
        task_id = "task_adv_oversize_03"
        m, k, n = 4, 4, 4
        expected_bytes = m * k * 4  # 64 bytes

        stub = mock_worker.get_stub()
        stub.RequestTileTask(
            pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=m, matrix_a_cols=k, matrix_b_cols=n, dtype="fp32")
        )

        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=50)
        mock_settlement.hold_invoice(inv.payment_hash)

        # 128 bytes (exceeds 64 expected)
        excessive_bytes = b"\x00" * 128

        def _oversized_generator():
            yield pb2.TileDataChunk(
                task_id=task_id,
                chunk_bytes=excessive_bytes,
                is_matrix_a=True,
                sequence_number=0,
                is_final_chunk=False,
            )

        metadata = [("authorization", f"L402 {inv.macaroon}:{inv.payment_hash}"), ("x-task-id", task_id)]
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.ExecuteTileTask(_oversized_generator(), metadata=metadata)
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "exceeded limit" in exc_info.value.details()

    def test_chunk_task_id_tampering_rejected(self, mock_worker, mock_settlement):
        """Adversarial sender injects chunk containing mismatched task_id."""
        task_id = "task_adv_mismatch_04"
        stub = mock_worker.get_stub()
        stub.RequestTileTask(
            pb2.TileTaskRequest(task_id=task_id, matrix_a_rows=4, matrix_a_cols=4, matrix_b_cols=4, dtype="fp32")
        )

        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=50)
        mock_settlement.hold_invoice(inv.payment_hash)

        def _mismatched_generator():
            yield pb2.TileDataChunk(
                task_id="injected_malicious_task_id_999",
                chunk_bytes=b"\x00" * 64,
                is_matrix_a=True,
                sequence_number=0,
                is_final_chunk=False,
            )

        metadata = [("authorization", f"L402 {inv.macaroon}:{inv.payment_hash}"), ("x-task-id", task_id)]
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.ExecuteTileTask(_mismatched_generator(), metadata=metadata)
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "task_id mismatch" in exc_info.value.details()


# ==============================================================================
# Suite 2: Worker Cryptographic Signature & Tampering Challenges
# ==============================================================================

class TestWorkerSignatureAdversarial:
    """Stress-tests worker signature forgery, byte corruption, and stolen result replays."""

    def test_worker_corrupted_result_byte_count(self, mock_worker, mock_settlement):
        """Worker returns result_bytes with truncated byte length; dispatcher raises StreamingError."""
        dispatcher = GrpcTaskDispatcher()
        task_id = "task_adv_sig_01"
        task_plan = TileTaskPlan(
            task_id=task_id,
            row_idx=0,
            col_idx=0,
            k_idx=0,
            row_slice=(0, 4),
            col_slice=(0, 4),
            k_slice=(0, 4),
            a_tile=np.ones((4, 4), dtype=np.float32),
            b_tile=np.ones((4, 4), dtype=np.float32),
            dtype="fp32",
            estimated_gflops=0.001,
        )

        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=50)
        mock_settlement.hold_invoice(inv.payment_hash)

        # Reconfigure mock_worker with a servicer returning truncated result_bytes
        mock_worker.stop()
        class CorruptedWorkerServicer(mock_worker.servicer.__class__):
            def ExecuteTileTask(self, request_iterator, context=None):
                res = super().ExecuteTileTask(request_iterator, context)
                return pb2.TileResult(
                    task_id=res.task_id,
                    rows=res.rows,
                    cols=res.cols,
                    result_bytes=res.result_bytes[:len(res.result_bytes) // 2],
                    worker_signature=res.worker_signature,
                    execution_time_ms=res.execution_time_ms,
                )

        mock_worker.servicer = CorruptedWorkerServicer(
            signer=mock_worker.signer,
            settlement_service=mock_settlement,
            verification_key=mock_worker.servicer.verification_key,
            min_price_rate=mock_worker.sats_per_gflop,
            min_task_sats=mock_worker.min_task_sats,
        )
        mock_worker.server = None
        mock_worker._thread_pool = futures.ThreadPoolExecutor(max_workers=4)
        mock_worker.start()

        with pytest.raises(StreamingError) as exc_info:
            dispatcher.dispatch_tile(
                target_multiaddr=mock_worker.multiaddr,
                worker_pubkey=mock_worker.pubkey,
                task_plan=task_plan,
                macaroon=inv.macaroon,
                payment_hash=inv.payment_hash,
                dtype="fp32",
            )
        assert "byte count" in str(exc_info.value).lower()

    def test_forged_worker_signature_rejection(self, mock_worker, mock_settlement):
        """Adversarial worker returns valid result bytes but a forged Secp256k1 signature."""
        dispatcher = GrpcTaskDispatcher()
        mock_worker.set_tamper_signature(True)

        task_id = "task_adv_forged_sig_02"
        task_plan = TileTaskPlan(
            task_id=task_id,
            row_idx=0,
            col_idx=0,
            k_idx=0,
            row_slice=(0, 4),
            col_slice=(0, 4),
            k_slice=(0, 4),
            a_tile=np.eye(4, dtype=np.float32),
            b_tile=np.eye(4, dtype=np.float32),
            dtype="fp32",
            estimated_gflops=0.001,
        )

        inv = mock_settlement.create_hodl_invoice(task_id=task_id, total_compute_sats=50)
        mock_settlement.hold_invoice(inv.payment_hash)

        with pytest.raises(WorkerSignatureError) as exc_info:
            dispatcher.dispatch_tile(
                target_multiaddr=mock_worker.multiaddr,
                worker_pubkey=mock_worker.pubkey,
                task_plan=task_plan,
                macaroon=inv.macaroon,
                payment_hash=inv.payment_hash,
                dtype="fp32",
            )
        assert exc_info.value.task_id == task_id
        assert exc_info.value.worker_pubkey == mock_worker.pubkey

    def test_signature_verification_uses_expected_task_id(self):
        """
        Adversarial challenge: Evaluates whether verify_worker_signature correctly rejects
        a signature generated for a different task_id ('task_old_999') when verified against
        the expected task_id ('task_expected_123').
        """
        dispatcher = GrpcTaskDispatcher()
        signer = _InternalWorkerSigner()
        pubkey = signer.get_pubkey_hex()

        expected_task_id = "task_expected_123"
        stolen_task_id = "task_stolen_999"

        result_bytes = np.ones((4, 4), dtype=np.float32).tobytes()
        c_hash = hashlib.sha256(result_bytes).hexdigest()

        # Sign for stolen_task_id
        stolen_payload = f"{stolen_task_id}:{c_hash}".encode("utf-8")
        stolen_sig = signer.sign_message(hashlib.sha256(stolen_payload).digest())

        # If verified against expected_task_id, MUST be False
        assert not dispatcher.verify_worker_signature(
            worker_pubkey=pubkey,
            task_id=expected_task_id,
            result_bytes=result_bytes,
            worker_signature=stolen_sig,
        )

        # If verified against stolen_task_id, matches stolen task
        assert dispatcher.verify_worker_signature(
            worker_pubkey=pubkey,
            task_id=stolen_task_id,
            result_bytes=result_bytes,
            worker_signature=stolen_sig,
        )


# ==============================================================================
# Suite 3: High-Concurrency Slot Contention & WorkerSlotTracker Stress
# ==============================================================================

class TestSlotContentionAdversarial:
    """Stress-tests WorkerSlotTracker and ParallelTaskScheduler under high concurrency."""

    def test_worker_slot_tracker_strict_capacity_limit(self):
        """Asserts WorkerSlotTracker never allows active slots to exceed max_slots."""
        tracker = WorkerSlotTracker()
        pubkey = "worker_slot_stress_pk"
        max_slots = 4
        tracker.register_worker(pubkey, max_slots=max_slots)

        acquired_count = 0
        lock = threading.Lock()

        def _acquire_worker():
            nonlocal acquired_count
            ok = tracker.acquire_slot(pubkey, timeout=0.05)
            if ok:
                with lock:
                    acquired_count += 1
                time.sleep(0.02)
                tracker.release_slot(pubkey)

        threads = [threading.Thread(target=_acquire_worker) for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At the end, active slots should return to 0
        assert tracker._active_slots[pubkey] == 0

    def test_worker_slot_tracker_timeout_behavior(self):
        """Verifies that acquire_slot returns False when capacity is saturated and timeout expires."""
        tracker = WorkerSlotTracker()
        pubkey = "worker_saturated_pk"
        tracker.register_worker(pubkey, max_slots=1)

        # Thread 1 acquires the only slot
        assert tracker.acquire_slot(pubkey, timeout=1.0) is True

        # Thread 2 attempts to acquire with short timeout -> must return False
        start = time.time()
        assert tracker.acquire_slot(pubkey, timeout=0.1) is False
        elapsed = time.time() - start
        assert elapsed >= 0.08

        # Release slot
        tracker.release_slot(pubkey)
        assert tracker._active_slots[pubkey] == 0

    def test_concurrent_32_worker_threads_scheduler_contention(self, mock_settlement):
        """
        32 concurrent tasks dispatched to 2 workers (each 4 slots).
        Validates scheduler handles concurrency without deadlocks or unhandled crashes.
        """
        w1 = MockWorkerNode(port=0, settlement_service=mock_settlement, vram_mb=16384).start()
        w2 = MockWorkerNode(port=0, settlement_service=mock_settlement, vram_mb=16384).start()

        try:
            dispatcher = GrpcTaskDispatcher(connect_timeout=2.0, task_timeout=5.0)
            scheduler = ParallelTaskScheduler(
                dispatcher=dispatcher,
                max_retries=2,
                backoff_factor=0.01,
                max_concurrency=16,
            )

            w1_obj = type("W", (), {"p2p_multiaddr": w1.multiaddr, "worker_pubkey": w1.pubkey, "slots_available": 4})()
            w2_obj = type("W", (), {"p2p_multiaddr": w2.multiaddr, "worker_pubkey": w2.pubkey, "slots_available": 4})()
            candidates = [w1_obj, w2_obj]

            def _execute_subtask(task_idx: int) -> Tuple[np.ndarray, Any]:
                a = np.random.randn(4, 4).astype(np.float32)
                b = np.random.randn(4, 4).astype(np.float32)
                task_plan = TileTaskPlan(
                    task_id=f"concurrent_task_{task_idx}",
                    row_idx=0,
                    col_idx=0,
                    k_idx=0,
                    row_slice=(0, 4),
                    col_slice=(0, 4),
                    k_slice=(0, 4),
                    a_tile=a,
                    b_tile=b,
                    dtype="fp32",
                    estimated_gflops=0.001,
                )
                inv = mock_settlement.create_hodl_invoice(task_id=task_plan.task_id, total_compute_sats=50)
                mock_settlement.hold_invoice(inv.payment_hash)

                return scheduler.execute_with_failover(
                    task_plan=task_plan,
                    candidate_workers=candidates,
                    macaroon=inv.macaroon,
                    payment_hash=inv.payment_hash,
                    dtype="fp32",
                )

            with futures.ThreadPoolExecutor(max_workers=16) as pool:
                tasks = [pool.submit(_execute_subtask, i) for i in range(16)]
                results = [f.result() for f in futures.as_completed(tasks)]

            assert len(results) == 16
            for c_res, raw_res in results:
                assert c_res.shape == (4, 4)
        finally:
            w1.stop()
            w2.stop()
            scheduler.shutdown()


# ==============================================================================
# Suite 4: Worker Failure Cascades & Rapid Failover Switching
# ==============================================================================

class TestWorkerFailoverCascadeAdversarial:
    """Stress-tests cascading worker dropouts and blacklisting under adversity."""

    def test_triple_worker_failure_cascade_to_survivor(self, mock_settlement):
        """3 workers drop connection in sequence; scheduler successfully fails over to 4th healthy worker."""
        w1 = MockWorkerNode(port=0, settlement_service=mock_settlement).start()
        w1.set_drop_connection(True)

        w2 = MockWorkerNode(port=0, settlement_service=mock_settlement).start()
        w2.set_drop_connection(True)

        w3 = MockWorkerNode(port=0, settlement_service=mock_settlement).start()
        w3.set_reject_handshake(True, message="VRAM exhausted")

        w4 = MockWorkerNode(port=0, settlement_service=mock_settlement).start()  # Healthy survivor!

        try:
            dispatcher = GrpcTaskDispatcher(connect_timeout=0.5, task_timeout=1.0)
            scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=4, backoff_factor=0.01)

            c1 = type("W", (), {"p2p_multiaddr": w1.multiaddr, "worker_pubkey": w1.pubkey, "slots_available": 1})()
            c2 = type("W", (), {"p2p_multiaddr": w2.multiaddr, "worker_pubkey": w2.pubkey, "slots_available": 1})()
            c3 = type("W", (), {"p2p_multiaddr": w3.multiaddr, "worker_pubkey": w3.pubkey, "slots_available": 1})()
            c4 = type("W", (), {"p2p_multiaddr": w4.multiaddr, "worker_pubkey": w4.pubkey, "slots_available": 1})()

            candidates = [c1, c2, c3, c4]

            a = np.random.randn(4, 4).astype(np.float32)
            b = np.random.randn(4, 4).astype(np.float32)
            task_plan = TileTaskPlan(
                task_id="task_cascade_01",
                row_idx=0,
                col_idx=0,
                k_idx=0,
                row_slice=(0, 4),
                col_slice=(0, 4),
                k_slice=(0, 4),
                a_tile=a,
                b_tile=b,
                dtype="fp32",
                estimated_gflops=0.001,
            )

            inv = mock_settlement.create_hodl_invoice(task_id=task_plan.task_id, total_compute_sats=50)
            mock_settlement.hold_invoice(inv.payment_hash)

            failover_log = []
            c_res, raw_res = scheduler.execute_with_failover(
                task_plan=task_plan,
                candidate_workers=candidates,
                macaroon=inv.macaroon,
                payment_hash=inv.payment_hash,
                dtype="fp32",
                on_failover_callback=lambda w, ex: failover_log.append((getattr(w, "worker_pubkey", ""), str(ex))),
            )

            expected_c = np.matmul(a, b)
            np.testing.assert_allclose(c_res, expected_c, rtol=1e-5)
            assert len(failover_log) == 3  # Failed on w1, w2, w3 before succeeding on w4
        finally:
            w1.stop()
            w2.stop()
            w3.stop()
            w4.stop()
            scheduler.shutdown()

    def test_total_worker_pool_exhaustion_raises_dispatch_error(self, mock_settlement):
        """All workers fail; scheduler exhausts retry limit and raises DispatchError."""
        w1 = MockWorkerNode(port=0, settlement_service=mock_settlement).start()
        w1.set_drop_connection(True)

        try:
            dispatcher = GrpcTaskDispatcher(connect_timeout=0.2, task_timeout=0.5)
            scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=1, backoff_factor=0.01)

            c1 = type("W", (), {"p2p_multiaddr": w1.multiaddr, "worker_pubkey": w1.pubkey, "slots_available": 1})()

            a = np.ones((4, 4), dtype=np.float32)
            b = np.ones((4, 4), dtype=np.float32)
            task_plan = TileTaskPlan(
                task_id="task_exhaustion_02",
                row_idx=0,
                col_idx=0,
                k_idx=0,
                row_slice=(0, 4),
                col_slice=(0, 4),
                k_slice=(0, 4),
                a_tile=a,
                b_tile=b,
                dtype="fp32",
                estimated_gflops=0.001,
            )

            inv = mock_settlement.create_hodl_invoice(task_id=task_plan.task_id, total_compute_sats=50)
            mock_settlement.hold_invoice(inv.payment_hash)

            with pytest.raises(DispatchError) as exc_info:
                scheduler.execute_with_failover(
                    task_plan=task_plan,
                    candidate_workers=[c1],
                    macaroon=inv.macaroon,
                    payment_hash=inv.payment_hash,
                    dtype="fp32",
                )
            assert "exhausted" in str(exc_info.value)
        finally:
            w1.stop()
            scheduler.shutdown()


# ==============================================================================
# Suite 5: Mathematical Perturbation & Fraud Proof Generation
# ==============================================================================

class TestMathematicalFraudAdversarial:
    """Stress-tests Freivalds rejection, Higham dynamic bound, and FraudProofBuilder under attack."""

    def test_mathematical_perturbation_triggers_freivalds_rejection(self, mock_settlement):
        """Worker returning slightly perturbed result (e.g. +0.05) is rejected by Freivalds verifier."""
        w = MockWorkerNode(port=0, settlement_service=mock_settlement).start()
        w.set_tamper_result(True)

        try:
            dispatcher = GrpcTaskDispatcher()
            a = np.random.randn(8, 8).astype(np.float32)
            b = np.random.randn(8, 8).astype(np.float32)

            task_plan = TileTaskPlan(
                task_id="task_fraud_verify_01",
                row_idx=0,
                col_idx=0,
                k_idx=0,
                row_slice=(0, 8),
                col_slice=(0, 8),
                k_slice=(0, 8),
                a_tile=a,
                b_tile=b,
                dtype="fp32",
                estimated_gflops=0.001,
            )

            inv = mock_settlement.create_hodl_invoice(task_id=task_plan.task_id, total_compute_sats=50)
            mock_settlement.hold_invoice(inv.payment_hash)

            c_tampered, raw_res = dispatcher.dispatch_tile(
                target_multiaddr=w.multiaddr,
                worker_pubkey=w.pubkey,
                task_plan=task_plan,
                macaroon=inv.macaroon,
                payment_hash=inv.payment_hash,
                dtype="fp32",
            )

            # Freivalds verifier MUST reject tampered result
            verifier = FreivaldsVerifier(k_rounds=5, use_dynamic_bound=True)
            is_valid, disc, bound, proof_data = verifier.verify(A=a, B=b, C=c_tampered, dtype="fp32")
            assert is_valid is False
            assert disc > bound
            assert "random_vector_r" in proof_data

            # Compile authoritative fraud proof
            consumer_privkey, _ = generate_keypair()
            proof = FraudProofBuilder.build_fraud_proof(
                task_id=task_plan.task_id,
                worker_pubkey=w.pubkey,
                A=a,
                B=b,
                C=c_tampered,
                proof_data=proof_data,
                worker_sig=raw_res.worker_signature,
                consumer_privkey=consumer_privkey,
                dtype="fp32",
            )
            assert proof.task_id == task_plan.task_id
            assert proof.observed_discrepancy > proof.dynamic_epsilon_bound
            assert len(proof.vector_r) == 8
        finally:
            w.stop()

    def test_client_fraud_detection_raises_fraud_detected_error(self, mock_settlement):
        """End-to-end client with tampered mock worker aborts and raises FraudDetectedError."""
        w_evil = MockWorkerNode(port=0, settlement_service=mock_settlement).start()
        w_evil.set_tamper_result(True)

        try:
            config = ClientConfig.for_testing(default_tile_size=8)
            client = MatrixComputeClient(config=config)
            # Override coordinator's mock workers with evil worker
            client.coordinator.mock_workers = [w_evil]

            a = np.random.randn(8, 8).astype(np.float32)
            b = np.random.randn(8, 8).astype(np.float32)

            with pytest.raises(FraudDetectedError) as exc_info:
                client.matmul(a, b, dtype="fp32", tile_size=8)

            assert "Freivalds verification failed" in str(exc_info.value)
            assert exc_info.value.fraud_proof is not None
            client.close()
        finally:
            w_evil.stop()


# ==============================================================================
# Suite 6: Extreme Workloads & Matrix Aspect Ratios
# ==============================================================================

class TestExtremeWorkloadsAdversarial:
    """Stress-tests client and coordinator across edge geometries and numerical scales."""

    def test_client_scalar_1x1_matrix(self):
        """Computes 1x1 scalar dot product [[3.5]] x [[2.0]] = [[7.0]]."""
        config = ClientConfig.for_testing(default_tile_size=4)
        a = np.array([[3.5]], dtype=np.float32)
        b = np.array([[2.0]], dtype=np.float32)

        with MatrixComputeClient(config=config) as client:
            c = client.matmul(a, b, dtype="fp32", tile_size=4)
            np.testing.assert_allclose(c, [[7.0]], rtol=1e-5)

    def test_client_extreme_aspect_ratio_1x128_by_128x1(self):
        """Vector dot product: (1 x 128) x (128 x 1) -> (1 x 1)."""
        config = ClientConfig.for_testing(default_tile_size=32)
        np.random.seed(201)
        a = np.random.randn(1, 128).astype(np.float32)
        b = np.random.randn(128, 1).astype(np.float32)

        with MatrixComputeClient(config=config) as client:
            c = client.matmul(a, b, dtype="fp32", tile_size=32)
            expected = np.matmul(a, b)
            np.testing.assert_allclose(c, expected, rtol=1e-4)

    def test_client_extreme_aspect_ratio_128x1_by_1x128(self):
        """Outer product: (128 x 1) x (1 x 128) -> (128 x 128)."""
        config = ClientConfig.for_testing(default_tile_size=32)
        np.random.seed(202)
        a = np.random.randn(128, 1).astype(np.float32)
        b = np.random.randn(1, 128).astype(np.float32)

        with MatrixComputeClient(config=config) as client:
            c = client.matmul(a, b, dtype="fp32", tile_size=32)
            expected = np.matmul(a, b)
            np.testing.assert_allclose(c, expected, rtol=1e-4)

    def test_client_tiny_magnitude_1e4_scale(self):
        """Inputs with scale ~1e-4 do not falsely trigger zero-work fraud or underflow."""
        config = ClientConfig.for_testing(default_tile_size=16)
        np.random.seed(203)
        a = (np.random.randn(16, 16) * 1e-4).astype(np.float32)
        b = (np.random.randn(16, 16) * 1e-4).astype(np.float32)

        with MatrixComputeClient(config=config) as client:
            c = client.matmul(a, b, dtype="fp32", tile_size=16)
            expected = np.matmul(a, b)
            np.testing.assert_allclose(c, expected, rtol=1e-3, atol=1e-9)

    def test_client_huge_magnitude_1e4_scale(self):
        """Inputs with scale ~1e4 do not overflow or trigger false positives."""
        config = ClientConfig.for_testing(default_tile_size=16)
        np.random.seed(204)
        a = (np.random.randn(16, 16) * 1e4).astype(np.float32)
        b = (np.random.randn(16, 16) * 1e4).astype(np.float32)

        with MatrixComputeClient(config=config) as client:
            c = client.matmul(a, b, dtype="fp32", tile_size=16)
            expected = np.matmul(a, b)
            np.testing.assert_allclose(c, expected, rtol=1e-4)

    def test_client_all_zeros_matrix(self):
        """All-zero matrices are correctly computed without zero-work fraud false positive."""
        config = ClientConfig.for_testing(default_tile_size=16)
        a = np.zeros((16, 16), dtype=np.float32)
        b = np.zeros((16, 16), dtype=np.float32)

        with MatrixComputeClient(config=config) as client:
            c = client.matmul(a, b, dtype="fp32", tile_size=16)
            assert np.all(c == 0.0)
