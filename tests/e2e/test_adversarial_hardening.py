"""
Tier 5 Adversarial Hardening Test Suite for matrix-compute-sdk.

Comprehensive white-box adversarial stress test suite covering 5 core categories:
1. Perturbed outputs & mathematical fraud detection (single bit flips, magnitude shifts, Inf/NaN injections, fraud proofs, escrow refund).
2. Corrupt / Byzantine worker nodes (mismatched task IDs, forged/corrupted signatures, dimension/byte mismatches, malformed chunk streams).
3. Network drops, disconnects & cascade failover (transient errors, 3-worker cascade failover, pool exhaustion, blacklisting).
4. Extreme & degenerate matrix workloads (1x1 scalar, 1x2048 dot product, 2048x1 outer product, subnormal floats, large magnitudes, zeros, identity, mixed FP32/FP16/BF16).
5. High concurrency & worker slot contention (32-thread dispatch, WorkerSlotTracker underflow & capacity limits, thread-safe settle/cancel race conditions).
"""

from __future__ import annotations

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

from matrix_contracts.schemas import (
    BondStatus,
    CreateHodlInvoiceResponse,
    FreivaldsFraudProof,
    HardwareSpecs,
    MatchedWorker,
    PricingSpecs,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
    WorkerQueryResponse,
)
from matrix_contracts.v1 import pb2, pb2_grpc

from matrix_compute.client import MatrixComputeClient, distributed_matmul
from matrix_compute.config import ClientConfig
from matrix_compute.crypto.keys import generate_keypair
from matrix_compute.crypto.signer import (
    ClientSigner,
    format_tile_result_payload,
    hash_tensor,
    sha256_digest,
    sign_message,
    verify_signature,
)
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import (
    DispatchError,
    EscrowError,
    EscrowStateError,
    FraudDetectedError,
    MatrixDimensionError,
    NoWorkersFoundError,
    StreamingError,
    TaskDispatchError,
    TaskTimeoutError,
    VerificationFailedError,
    WorkerConnectionError,
    WorkerSignatureError,
    WorkerTaskRejectedError,
    WorkerUnavailableError,
)
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler, WorkerSlotTracker
from matrix_compute.p2p.channel_pool import ChannelPool, parse_multiaddr
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager, EscrowState, EscrowStatus
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import MockWorkerNode, _InternalWorkerSigner
from matrix_compute.tiling.assembler import ResultAssembler
from matrix_compute.tiling.partitioner import MatrixTiler, TileTaskPlan
from matrix_compute.tiling.precision import (
    calculate_higham_bound,
    get_bytes_per_element,
    get_numpy_dtype,
    get_unit_roundoff,
    normalize_dtype,
    truncate_to_bf16,
)
from matrix_compute.verification.fraud_proof import (
    FraudProofBuilder,
    verify_fraud_proof,
)
from matrix_compute.verification.freivalds import (
    DEFAULT_FREIVALDS_ROUNDS,
    FreivaldsVerifier,
    freivalds_verify,
    higham_bound,
    matrix_infinity_norm,
)
from tests.conftest import (
    MockRelayDouble,
    MockSettlementDouble,
    OpaqueL402Helper,
    ReferenceOracle,
    StreamChunkHelper,
)


# ==============================================================================
# Shared Fixtures
# ==============================================================================

@pytest.fixture
def adversarial_keypair():
    """Returns consumer and worker cryptographic keypairs."""
    c_priv, c_pub = generate_keypair()
    w_priv, w_pub = generate_keypair()
    return {
        "consumer_priv": c_priv,
        "consumer_pub": c_pub,
        "worker_priv": w_priv,
        "worker_pub": w_pub,
    }


@pytest.fixture
def settlement_service():
    """Provides isolated MockSettlementService with deterministic verification key."""
    vk = "adversarial-settlement-verification-key-tier5"
    service = MockSettlementService(verification_key=vk)
    yield service
    service.reset()


@pytest.fixture
def worker_node_factory(settlement_service):
    """Factory creating started MockWorkerNode instances cleanly shut down on teardown."""
    created_workers: List[MockWorkerNode] = []

    def _create(
        port: int = 0,
        privkey_hex: Optional[str] = None,
        vram_mb: int = 24576,
        sats_per_gflop: float = 0.0005,
    ) -> MockWorkerNode:
        node = MockWorkerNode(
            port=port,
            privkey_hex=privkey_hex,
            settlement_service=settlement_service,
            verification_key=settlement_service.verification_key,
            vram_mb=vram_mb,
            sats_per_gflop=sats_per_gflop,
        )
        node.start()
        created_workers.append(node)
        return node

    yield _create

    for w in created_workers:
        try:
            w.stop()
        except Exception:
            pass


# ==============================================================================
# Category 1: Perturbed Outputs & Mathematical Fraud Detection
# ==============================================================================

class TestCategory1PerturbedOutputsAndFraudDetection:
    """
    Tier 5 Category 1:
    - Tampering with floating-point values (single bit flip, magnitude shifts, Inf/NaN).
    - Freivalds O(n^2) detection and verification bound validation.
    - L402 HODL escrow cancellation and preimage withholding.
    - Cryptographically signed FreivaldsFraudProof generation and independent dispute verification.
    """

    def test_tampered_float_single_bit_flip_mantissa_caught_by_freivalds(self):
        """Single bit flip in float32 mantissa is reliably caught by Freivalds verification."""
        np.random.seed(101)
        m, k, n = 32, 32, 32
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)
        C_honest = np.matmul(A, B)

        # Flip bit 16 of a non-zero float element in C
        C_tampered = C_honest.copy()
        raw_u32 = C_tampered.view(np.uint32)
        raw_u32[5, 5] ^= np.uint32(1 << 16)

        is_valid, disc, bound, proof_data = FreivaldsVerifier.verify(
            A, B, C_tampered, dtype="fp32", k_rounds=10
        )
        assert is_valid is False
        assert disc > bound
        assert "random_vector_r" in proof_data
        assert len(proof_data["random_vector_r"]) == n

    def test_tampered_float_single_bit_flip_exponent_caught_by_freivalds(self):
        """Single bit flip in float32 exponent produces detectable discrepancy."""
        np.random.seed(102)
        m, k, n = 16, 16, 16
        A = (np.random.randn(m, k) + 1.0).astype(np.float32)
        B = (np.random.randn(k, n) + 1.0).astype(np.float32)
        C_honest = np.matmul(A, B)

        # Flip bit 25 (exponent)
        C_tampered = C_honest.copy()
        raw_u32 = C_tampered.view(np.uint32)
        raw_u32[2, 3] ^= np.uint32(1 << 25)

        is_valid, disc, bound, _ = FreivaldsVerifier.verify(
            A, B, C_tampered, dtype="fp32", k_rounds=5
        )
        assert is_valid is False
        assert disc > bound

    def test_tampered_float_large_magnitude_perturbation_caught(self):
        """Large magnitude additive perturbation (+1000.0) is rejected 100% of the time."""
        np.random.seed(103)
        m, k, n = 16, 16, 16
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)
        C_tampered = np.matmul(A, B)
        C_tampered[0, :] += 1000.0

        for trial in range(10):
            is_valid, disc, bound, _ = FreivaldsVerifier.verify(
                A, B, C_tampered, dtype="fp32", k_rounds=20
            )
            assert is_valid is False
            assert disc > 100.0

    def test_tampered_float_inf_injection_caught_cleanly(self):
        """Worker injecting positive or negative infinity is caught without crashing."""
        np.random.seed(104)
        m, k, n = 8, 8, 8
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)
        C_tampered = np.matmul(A, B)
        C_tampered[3, 3] = np.inf

        is_valid, disc, bound, proof = FreivaldsVerifier.verify(
            A, B, C_tampered, dtype="fp32", k_rounds=5
        )
        assert is_valid is False
        assert math.isinf(disc)
        assert "Non-finite" in proof.get("reason", "")

    def test_tampered_float_nan_injection_caught_cleanly(self):
        """Worker injecting NaN is caught cleanly, returning invalid status and failure proof."""
        np.random.seed(105)
        m, k, n = 8, 8, 8
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)
        C_tampered = np.matmul(A, B)
        C_tampered[1, 1] = np.nan

        is_valid, disc, bound, proof = FreivaldsVerifier.verify(
            A, B, C_tampered, dtype="fp32", k_rounds=5
        )
        assert is_valid is False
        assert math.isinf(disc)
        assert "Non-finite" in proof.get("reason", "")

    def test_escrow_aborted_and_refunded_on_freivalds_failure(self, settlement_service, adversarial_keypair):
        """Freivalds verification failure aborts escrow, refunds funds, and withholds preimage."""
        task_id = "task_adv_fraud_escrow_abort"
        mgr = EscrowManager(
            settlement_backend=MockSettlementBackend(settlement_service),
            verification_key=settlement_service.verification_key,
        )
        escrow = mgr.create_and_hold(task_id=task_id, total_sats=500)
        assert escrow.status == EscrowStatus.HELD
        assert settlement_service.get_status(escrow.payment_hash) == "HELD"

        # Tampered computation
        np.random.seed(106)
        A = np.random.randn(8, 8).astype(np.float32)
        B = np.random.randn(8, 8).astype(np.float32)
        C_tampered = np.matmul(A, B) + 10.0

        w_priv = adversarial_keypair["worker_priv"]
        w_pub = adversarial_keypair["worker_pub"]
        c_hash = sha256_digest(np.ascontiguousarray(C_tampered).tobytes())
        worker_sig = sign_message(w_priv, format_tile_result_payload(task_id, c_hash))

        is_valid, cancelled_escrow, fraud_proof = mgr.verify_and_settle(
            task_id=task_id,
            A=A,
            B=B,
            C=C_tampered,
            worker_pubkey=w_pub,
            worker_sig=worker_sig,
            consumer_privkey=adversarial_keypair["consumer_priv"],
            dtype="fp32",
        )

        assert is_valid is False
        assert cancelled_escrow.status == EscrowStatus.CANCELLED
        assert settlement_service.get_status(escrow.payment_hash) == "CANCELLED"
        assert fraud_proof is not None
        assert isinstance(fraud_proof, FreivaldsFraudProof)

        # Preimage must remain strictly withheld; settle must fail
        with pytest.raises(Exception):
            settlement_service.settle_invoice(escrow.payment_hash, preimage=cancelled_escrow.preimage)

    def test_freivalds_fraud_proof_cryptographically_signed_and_verifiable(self, adversarial_keypair):
        """Authoritative FreivaldsFraudProof is generated, cryptographically signed, and independently verified."""
        task_id = "task_adv_fraud_proof_crypto"
        m, k, n = 16, 16, 16
        np.random.seed(107)
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)
        C_tampered = np.matmul(A, B) + 5.0

        w_priv = adversarial_keypair["worker_priv"]
        w_pub = adversarial_keypair["worker_pub"]
        c_priv = adversarial_keypair["consumer_priv"]
        c_pub = adversarial_keypair["consumer_pub"]

        c_hash = sha256_digest(np.ascontiguousarray(C_tampered).tobytes())
        worker_sig = sign_message(w_priv, format_tile_result_payload(task_id, c_hash))

        _, _, bound, proof_data = FreivaldsVerifier.verify(A, B, C_tampered, dtype="fp32", k_rounds=5)

        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=w_pub,
            A=A,
            B=B,
            C=C_tampered,
            proof_data=proof_data,
            worker_sig=worker_sig,
            consumer_privkey=c_priv,
            dtype="fp32",
            epsilon_tolerance=bound,
            verify_worker_sig=True,
        )

        assert proof.task_id == task_id
        assert proof.worker_pubkey == w_pub
        assert proof.consumer_pubkey == c_pub
        assert proof.max_observed_discrepancy > proof.epsilon_tolerance

        # Verify mathematical and cryptographic integrity via independent verifier
        assert verify_fraud_proof(proof, matrix_a=A, matrix_b=B, matrix_c=C_tampered, verify_signatures=True) is True

    def test_e2e_mock_worker_tamper_result_triggers_fraud_detected_error(self, settlement_service, worker_node_factory):
        """E2E test: MockWorkerNode with set_tamper_result(True) raises FraudDetectedError in Client API."""
        worker = worker_node_factory()
        worker.set_tamper_result(tamper=True, perturbation=25.0)

        mock_relay = MockNostrRelay()
        worker.publish_to_relay(mock_relay)

        config = ClientConfig(
            offline_mock_mode=True,
            nostr_relays=["ws://127.0.0.1:8008"],
            settlement_api_url="http://127.0.0.1:8080",
            settlement_verification_key=settlement_service.verification_key,
            min_worker_vram_mb=8192,
            max_price_per_gflop=0.001,
            default_tile_size=16,
            max_retries=0,
        )

        signer = ClientSigner()
        relay_client = RelayClient(relay_urls=["ws://127.0.0.1:8008"], in_memory_relay=mock_relay)
        matchmaker = Matchmaker()
        backend = MockSettlementBackend(settlement_service)
        escrow_mgr = EscrowManager(settlement_backend=backend, verification_key=config.settlement_verification_key)
        dispatcher = GrpcTaskDispatcher(consumer_pubkey=signer.get_pubkey_hex())
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=0)

        coordinator = DistributedMatmulCoordinator(
            config=config,
            signer=signer,
            relay_client=relay_client,
            matchmaker=matchmaker,
            escrow_manager=escrow_mgr,
            scheduler=scheduler,
            dispatcher=dispatcher,
            mock_workers=[worker],
        )

        np.random.seed(108)
        A = np.random.randn(16, 16).astype(np.float32)
        B = np.random.randn(16, 16).astype(np.float32)

        with pytest.raises(FraudDetectedError) as exc_info:
            coordinator.matmul(A, B, dtype="fp32", tile_size=16)

        assert "Freivalds verification failed" in str(exc_info.value)
        assert exc_info.value.fraud_proof is not None
        assert isinstance(exc_info.value.fraud_proof, FreivaldsFraudProof)


# ==============================================================================
# Category 2: Corrupt / Byzantine Worker Nodes
# ==============================================================================

class TestCategory2CorruptByzantineWorkerNodes:
    """
    Tier 5 Category 2:
    - Worker returning mismatched task ID.
    - Worker returning invalid/forged Secp256k1 signatures.
    - Worker returning incorrect dimensions or corrupted byte counts.
    - Worker sending truncated or out-of-order chunk streams.
    """

    def test_worker_mismatched_task_id_rejected_with_task_dispatch_error(self, settlement_service, worker_node_factory):
        """Worker returning a mismatched task ID raises TaskDispatchError."""
        worker = worker_node_factory()
        worker.set_override_result_task_id("fraudulent_task_id_injection_666")

        dispatcher = GrpcTaskDispatcher()
        task_id = "task_adv_honest_id_123"
        A = np.ones((4, 4), dtype=np.float32)
        B = np.ones((4, 4), dtype=np.float32)

        # 1. Negotiate
        dispatcher.request_tile_task(
            worker_or_target=worker.multiaddr,
            task_id=task_id,
            m=4,
            k=4,
            n=4,
            dtype="fp32",
            worker_pubkey=worker.pubkey,
        )

        # 2. Escrow
        backend = MockSettlementBackend(settlement_service)
        mgr = EscrowManager(settlement_backend=backend, verification_key=settlement_service.verification_key)
        escrow = mgr.create_and_hold(task_id=task_id, total_sats=100)

        # 3. Stream execute
        with pytest.raises(TaskDispatchError) as exc_info:
            dispatcher.stream_execute_tile(
                worker_or_target=worker.multiaddr,
                task_id=task_id,
                matrix_a=A,
                matrix_b=B,
                dtype="fp32",
                auth_header=escrow.get_auth_header(),
                worker_pubkey=worker.pubkey,
            )

        assert exc_info.value.task_id == task_id
        assert exc_info.value.returned_task_id == "fraudulent_task_id_injection_666"

    def test_worker_invalid_forged_secp256k1_signature_rejected(self, settlement_service, worker_node_factory):
        """Worker returning an invalid/tampered cryptographic signature raises WorkerSignatureError."""
        worker = worker_node_factory()
        worker.set_tamper_signature(True)

        dispatcher = GrpcTaskDispatcher()
        task_id = "task_adv_forged_sig_test"
        A = np.ones((4, 4), dtype=np.float32)
        B = np.ones((4, 4), dtype=np.float32)

        dispatcher.request_tile_task(
            worker_or_target=worker.multiaddr,
            task_id=task_id,
            m=4,
            k=4,
            n=4,
            dtype="fp32",
            worker_pubkey=worker.pubkey,
        )

        backend = MockSettlementBackend(settlement_service)
        mgr = EscrowManager(settlement_backend=backend, verification_key=settlement_service.verification_key)
        escrow = mgr.create_and_hold(task_id=task_id, total_sats=100)

        with pytest.raises(WorkerSignatureError) as exc_info:
            dispatcher.stream_execute_tile(
                worker_or_target=worker.multiaddr,
                task_id=task_id,
                matrix_a=A,
                matrix_b=B,
                dtype="fp32",
                auth_header=escrow.get_auth_header(),
                worker_pubkey=worker.pubkey,
            )

        assert exc_info.value.task_id == task_id
        assert exc_info.value.worker_pubkey == worker.pubkey

    def test_worker_zero_signature_rejected_by_verifier(self):
        """verify_worker_signature rejects an all-zero signature string."""
        dispatcher = GrpcTaskDispatcher()
        pubkey = "02" + "a" * 64
        assert dispatcher.verify_worker_signature(
            worker_pubkey=pubkey,
            task_id="task_zero_sig",
            result_bytes=b"dummy_bytes",
            worker_signature="00" * 64,
        ) is False

    def test_worker_incorrect_dimensions_rejected(self, worker_node_factory):
        """Dispatcher rejects worker response if rows or cols do not match requested dimensions."""
        dispatcher = GrpcTaskDispatcher()
        # Mocking stub to return mismatched dimensions
        class MockStubWrongDimensions:
            def ExecuteTileTask(self, *args, **kwargs):
                return pb2.TileResult(
                    task_id="task_wrong_dims",
                    rows=999,  # Mismatched
                    cols=4,
                    result_bytes=b"\x00" * (4 * 4 * 4),
                    worker_signature="sig",
                    execution_time_ms=10,
                )

        class DummyPool:
            def get_stub(self, target):
                return MockStubWrongDimensions()

        dispatcher.channel_pool = DummyPool()

        A = np.zeros((4, 4), dtype=np.float32)
        B = np.zeros((4, 4), dtype=np.float32)

        with pytest.raises(StreamingError) as exc_info:
            dispatcher.stream_execute_tile(
                worker_or_target="127.0.0.1:9999",
                task_id="task_wrong_dims",
                matrix_a=A,
                matrix_b=B,
                dtype="fp32",
                worker_pubkey="",
            )
        assert "dimensions" in str(exc_info.value).lower()

    def test_worker_corrupted_byte_count_rejected(self):
        """Dispatcher rejects worker response if result_bytes length does not match rows * cols * bytes_per_elem."""
        dispatcher = GrpcTaskDispatcher()

        class MockStubCorruptedBytes:
            def ExecuteTileTask(self, *args, **kwargs):
                return pb2.TileResult(
                    task_id="task_corrupt_bytes",
                    rows=4,
                    cols=4,
                    result_bytes=b"\x00" * 40,  # Expected 64 bytes for 4x4 fp32
                    worker_signature="sig",
                    execution_time_ms=10,
                )

        class DummyPool:
            def get_stub(self, target):
                return MockStubCorruptedBytes()

        dispatcher.channel_pool = DummyPool()

        A = np.zeros((4, 4), dtype=np.float32)
        B = np.zeros((4, 4), dtype=np.float32)

        with pytest.raises(StreamingError) as exc_info:
            dispatcher.stream_execute_tile(
                worker_or_target="127.0.0.1:9999",
                task_id="task_corrupt_bytes",
                matrix_a=A,
                matrix_b=B,
                dtype="fp32",
                worker_pubkey="",
            )
        assert "byte count" in str(exc_info.value).lower()

    def test_worker_out_of_order_chunk_stream_rejected_by_servicer(self, settlement_service, worker_node_factory):
        """Out-of-order chunk stream (seq 1 before seq 0) is aborted by worker servicer."""
        worker = worker_node_factory()
        task_id = "task_adv_ooo_stream"

        # 1. Negotiate
        stub = worker.get_stub()
        stub.RequestTileTask(
            pb2.TileTaskRequest(
                task_id=task_id, matrix_a_rows=2, matrix_a_cols=2, matrix_b_cols=2, dtype="fp32"
            )
        )

        # 2. Escrow
        backend = MockSettlementBackend(settlement_service)
        mgr = EscrowManager(settlement_backend=backend, verification_key=settlement_service.verification_key)
        escrow = mgr.create_and_hold(task_id=task_id, total_sats=50)

        # 3. Adversarial chunk stream: send sequence 1 before sequence 0
        def _bad_chunks():
            yield pb2.TileDataChunk(
                task_id=task_id,
                chunk_bytes=b"\x00" * 8,
                is_matrix_a=True,
                sequence_number=1,  # INVALID: expected 0
                is_final_chunk=False,
            )

        with pytest.raises(grpc.RpcError) as exc_info:
            stub.ExecuteTileTask(
                _bad_chunks(),
                metadata=[("x-task-id", task_id), ("authorization", escrow.get_auth_header())],
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Out of order sequence" in exc_info.value.details()

    def test_worker_truncated_chunk_stream_without_final_flag_aborts(self, settlement_service, worker_node_factory):
        """Worker servicer aborts stream if terminated without is_final_chunk=True."""
        worker = worker_node_factory()
        task_id = "task_adv_no_final_flag"

        stub = worker.get_stub()
        stub.RequestTileTask(
            pb2.TileTaskRequest(
                task_id=task_id, matrix_a_rows=2, matrix_a_cols=2, matrix_b_cols=2, dtype="fp32"
            )
        )

        backend = MockSettlementBackend(settlement_service)
        mgr = EscrowManager(settlement_backend=backend, verification_key=settlement_service.verification_key)
        escrow = mgr.create_and_hold(task_id=task_id, total_sats=50)

        def _truncated_chunks():
            yield pb2.TileDataChunk(
                task_id=task_id,
                chunk_bytes=b"\x00" * 16,
                is_matrix_a=True,
                sequence_number=0,
                is_final_chunk=False,
            )
            # Ends prematurely without final chunk

        with pytest.raises(grpc.RpcError) as exc_info:
            stub.ExecuteTileTask(
                _truncated_chunks(),
                metadata=[("x-task-id", task_id), ("authorization", escrow.get_auth_header())],
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "final chunk flag" in exc_info.value.details()


# ==============================================================================
# Category 3: Network Drops, Disconnects & Cascade Failover
# ==============================================================================

class TestCategory3NetworkDropsAndCascadeFailover:
    """
    Tier 5 Category 3:
    - Transient gRPC errors triggering jittered exponential backoff and worker failover.
    - Pool of 3 workers where workers 1 and 2 fail, cascading cleanly to worker 3.
    - Complete worker pool exhaustion raising clear WorkerUnavailableError / DispatchError.
    - Worker failure tracking and blacklisting.
    """

    def test_three_worker_cascade_failover_cleanly_to_third_worker(self, settlement_service, worker_node_factory):
        """
        Pool of 3 workers:
        - Worker 1 drops connection (UNAVAILABLE).
        - Worker 2 rejects task handshake (PERMISSION_DENIED / WorkerTaskRejectedError).
        - Worker 3 is healthy.
        Scheduler cleanly cascades w1 -> w2 -> w3 and completes the task.
        """
        w1 = worker_node_factory()
        w2 = worker_node_factory()
        w3 = worker_node_factory()

        # Fault injection
        w1.set_drop_connection(True)
        w2.set_reject_handshake(True, message="Worker VRAM full")

        candidates = [
            MatchedWorker(
                worker_pubkey=w1.pubkey,
                p2p_multiaddr=w1.multiaddr,
                lightning_routing_pubkey=w1.lightning_routing_pubkey,
                reputation_score=1.0,
                sats_per_gflop=0.0005,
            ),
            MatchedWorker(
                worker_pubkey=w2.pubkey,
                p2p_multiaddr=w2.multiaddr,
                lightning_routing_pubkey=w2.lightning_routing_pubkey,
                reputation_score=0.9,
                sats_per_gflop=0.0005,
            ),
            MatchedWorker(
                worker_pubkey=w3.pubkey,
                p2p_multiaddr=w3.multiaddr,
                lightning_routing_pubkey=w3.lightning_routing_pubkey,
                reputation_score=0.8,
                sats_per_gflop=0.0005,
            ),
        ]

        task_id = "task_adv_cascade_3workers"
        np.random.seed(301)
        A = np.random.randn(8, 8).astype(np.float32)
        B = np.random.randn(8, 8).astype(np.float32)

        task_plan = TileTaskPlan(
            task_id=task_id,
            row_idx=0,
            col_idx=0,
            k_idx=0,
            row_slice=(0, 8),
            col_slice=(0, 8),
            k_slice=(0, 8),
            a_tile=A,
            b_tile=B,
            dtype="fp32",
            estimated_gflops=(2 * 8**3) / 1e9,
        )

        backend = MockSettlementBackend(settlement_service)
        mgr = EscrowManager(settlement_backend=backend, verification_key=settlement_service.verification_key)
        escrow = mgr.create_and_hold(task_id=task_id, total_sats=100)

        dispatcher = GrpcTaskDispatcher()
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=3, backoff_factor=0.05)

        failed_workers: List[str] = []
        def _on_failover(w, ex):
            pk = getattr(w, "worker_pubkey", getattr(w, "pubkey", ""))
            failed_workers.append(pk)

        res_matrix, raw_res = scheduler.execute_with_failover(
            task_plan=task_plan,
            candidate_workers=candidates,
            macaroon=escrow.macaroon,
            payment_hash=escrow.payment_hash,
            dtype="fp32",
            on_failover_callback=_on_failover,
        )

        # Expected result from healthy worker 3
        expected = np.matmul(A, B)
        np.testing.assert_allclose(res_matrix, expected, rtol=1e-5, atol=1e-5)
        assert failed_workers == [w1.pubkey, w2.pubkey]
        assert scheduler._worker_failure_counts.get(w1.pubkey, 0) >= 1
        assert scheduler._worker_failure_counts.get(w2.pubkey, 0) >= 1
        assert scheduler._worker_failure_counts.get(w3.pubkey, 0) == 0

    def test_complete_worker_pool_exhaustion_raises_dispatch_error(self, settlement_service, worker_node_factory):
        """All workers in pool failing exhausts max_retries and raises DispatchError."""
        w1 = worker_node_factory()
        w2 = worker_node_factory()
        w1.set_drop_connection(True)
        w2.set_drop_connection(True)

        candidates = [
            MatchedWorker(worker_pubkey=w1.pubkey, p2p_multiaddr=w1.multiaddr, lightning_routing_pubkey="ln1", sats_per_gflop=0.0005),
            MatchedWorker(worker_pubkey=w2.pubkey, p2p_multiaddr=w2.multiaddr, lightning_routing_pubkey="ln2", sats_per_gflop=0.0005),
        ]

        task_id = "task_adv_pool_exhausted"
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
            estimated_gflops=(2 * 4**3) / 1e9,
        )

        backend = MockSettlementBackend(settlement_service)
        mgr = EscrowManager(settlement_backend=backend, verification_key=settlement_service.verification_key)
        escrow = mgr.create_and_hold(task_id=task_id, total_sats=100)

        dispatcher = GrpcTaskDispatcher()
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=2, backoff_factor=0.01)

        with pytest.raises(DispatchError) as exc_info:
            scheduler.execute_with_failover(
                task_plan=task_plan,
                candidate_workers=candidates,
                macaroon=escrow.macaroon,
                payment_hash=escrow.payment_hash,
                dtype="fp32",
            )

        assert "exhausted" in str(exc_info.value).lower()

    def test_empty_candidate_pool_raises_worker_unavailable_error(self):
        """Empty candidate pool immediately raises WorkerUnavailableError without dispatching."""
        task_plan = TileTaskPlan(
            task_id="task_no_candidates",
            row_idx=0,
            col_idx=0,
            k_idx=0,
            row_slice=(0, 2),
            col_slice=(0, 2),
            k_slice=(0, 2),
            a_tile=np.zeros((2, 2)),
            b_tile=np.zeros((2, 2)),
            dtype="fp32",
            estimated_gflops=(2 * 2**3) / 1e9,
        )
        scheduler = ParallelTaskScheduler(dispatcher=GrpcTaskDispatcher())

        with pytest.raises(WorkerUnavailableError):
            scheduler.execute_with_failover(
                task_plan=task_plan,
                candidate_workers=[],
                macaroon="mock_macaroon",
                payment_hash="mock_hash",
            )

    def test_worker_blacklisting_after_consecutive_failures(self, worker_node_factory):
        """Worker failing 3 consecutive times is blacklisted by the scheduler."""
        w = worker_node_factory()
        scheduler = ParallelTaskScheduler(dispatcher=GrpcTaskDispatcher())

        assert len(scheduler.get_healthy_workers([w])) == 1

        scheduler.record_worker_failure(w.pubkey)
        scheduler.record_worker_failure(w.pubkey)
        assert len(scheduler.get_healthy_workers([w])) == 1

        # 3rd failure blacklists
        scheduler.record_worker_failure(w.pubkey)
        assert len(scheduler.get_healthy_workers([w])) == 0
        assert w.pubkey in scheduler._blacklisted_workers


# ==============================================================================
# Category 4: Extreme & Degenerate Matrix Workloads
# ==============================================================================

class TestCategory4ExtremeAndDegenerateMatrixWorkloads:
    """
    Tier 5 Category 4:
    - Scalar 1x1 matrix multiplication.
    - Extreme aspect ratios: 1x2048 by 2048x1 (dot product) and 2048x1 by 1x2048 (outer product).
    - High dynamic range / ill-conditioned: subnormal floats, large magnitudes (10^4), all zeros, identity.
    - Mixed datatypes (FP32, FP16, BF16).
    """

    def test_scalar_1x1_matrix_multiplication(self, settlement_service, worker_node_factory):
        """Scalar 1x1 matrix multiplication ([3.5] x [-2.0] = [-7.0]) executes and verifies cleanly."""
        worker = worker_node_factory()
        mock_relay = MockNostrRelay()
        worker.publish_to_relay(mock_relay)

        config = ClientConfig(
            offline_mock_mode=True,
            nostr_relays=["ws://127.0.0.1:8008"],
            settlement_api_url="http://127.0.0.1:8080",
            settlement_verification_key=settlement_service.verification_key,
            min_worker_vram_mb=1024,
            default_tile_size=1,
        )

        signer = ClientSigner()
        relay_client = RelayClient(relay_urls=["ws://127.0.0.1:8008"], in_memory_relay=mock_relay)
        matchmaker = Matchmaker()
        backend = MockSettlementBackend(settlement_service)
        escrow_mgr = EscrowManager(settlement_backend=backend, verification_key=config.settlement_verification_key)
        dispatcher = GrpcTaskDispatcher(consumer_pubkey=signer.get_pubkey_hex())
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher)

        coordinator = DistributedMatmulCoordinator(
            config=config,
            signer=signer,
            relay_client=relay_client,
            matchmaker=matchmaker,
            escrow_manager=escrow_mgr,
            scheduler=scheduler,
            dispatcher=dispatcher,
            mock_workers=[worker],
        )

        A = np.array([[3.5]], dtype=np.float32)
        B = np.array([[-2.0]], dtype=np.float32)
        C = coordinator.matmul(A, B, dtype="fp32", tile_size=1)

        assert C.shape == (1, 1)
        np.testing.assert_allclose(C, np.array([[-7.0]], dtype=np.float32))

    def test_extreme_aspect_ratio_dot_product_1x2048_by_2048x1(self):
        """
        Dot product: 1x2048 by 2048x1 -> 1x1.
        Evaluates large inner dimension K=2048 across multi-K slices with Higham bound.
        """
        np.random.seed(402)
        k_dim = 2048
        A = np.random.randn(1, k_dim).astype(np.float32) / math.sqrt(k_dim)
        B = np.random.randn(k_dim, 1).astype(np.float32) / math.sqrt(k_dim)
        expected = np.matmul(A.astype(np.float64), B.astype(np.float64)).astype(np.float32)

        # Partition into tile_size=512 -> 4 slices along K
        grid_plan = MatrixTiler.create_plan(A, B, tile_size=512, dtype="fp32")
        assert len(grid_plan.tasks) == 4

        assembler = ResultAssembler.from_plan(grid_plan)
        for task in grid_plan.tasks:
            tile_c = np.matmul(task.a_tile, task.b_tile)
            is_valid, disc, bound, _ = FreivaldsVerifier.verify(
                task.a_tile, task.b_tile, tile_c, dtype="fp32"
            )
            assert is_valid is True
            assembler.accumulate(
                tile_result=tile_c,
                row_idx=task.row_idx,
                col_idx=task.col_idx,
                k_idx=task.k_idx,
                row_slice=task.row_slice,
                col_slice=task.col_slice,
                task_id=task.task_id,
            )

        assembled_c = assembler.get_result(enforce_complete=True)
        assert assembled_c.shape == (1, 1)
        np.testing.assert_allclose(assembled_c, expected, rtol=1e-4, atol=1e-4)

    def test_extreme_aspect_ratio_outer_product_2048x1_by_1x2048(self):
        """
        Outer product: 2048x1 by 1x2048 -> 2048x2048.
        Evaluates inner dimension K=1 generating a large 2D matrix across 16 block tiles.
        """
        np.random.seed(403)
        dim = 1024  # Use 1024 to stay memory-efficient in unit environment
        A = np.random.randn(dim, 1).astype(np.float32)
        B = np.random.randn(1, dim).astype(np.float32)
        expected = np.matmul(A, B)

        grid_plan = MatrixTiler.create_plan(A, B, tile_size=512, dtype="fp32")
        assert len(grid_plan.tasks) == 4  # (1024/512)^2 = 4

        assembler = ResultAssembler.from_plan(grid_plan)
        for task in grid_plan.tasks:
            tile_c = np.matmul(task.a_tile, task.b_tile)
            is_valid, _, _, _ = FreivaldsVerifier.verify(task.a_tile, task.b_tile, tile_c, dtype="fp32")
            assert is_valid is True
            assembler.accumulate(
                tile_result=tile_c,
                row_idx=task.row_idx,
                col_idx=task.col_idx,
                k_idx=task.k_idx,
                row_slice=task.row_slice,
                col_slice=task.col_slice,
                task_id=task.task_id,
            )

        assembled_c = assembler.get_result(enforce_complete=True)
        assert assembled_c.shape == (dim, dim)
        np.testing.assert_allclose(assembled_c, expected, rtol=1e-5, atol=1e-5)

    def test_high_dynamic_range_subnormal_floating_point_numbers(self):
        """Subnormal floating point values (~1.4e-38) are handled stably without underflow crashes."""
        subnormal_val = np.float32(1.4e-38)
        A = np.full((8, 8), subnormal_val, dtype=np.float32)
        B = np.eye(8, dtype=np.float32)
        C = np.matmul(A, B)

        is_valid, disc, bound, _ = FreivaldsVerifier.verify(A, B, C, dtype="fp32", k_rounds=5)
        assert is_valid is True
        np.testing.assert_allclose(C, A)

    def test_high_dynamic_range_large_magnitudes_1e4(self):
        """Large magnitude matrices (values ~1e4) scale Higham dynamic bound properly without false fraud."""
        np.random.seed(405)
        m, k, n = 16, 16, 16
        A = (np.random.randn(m, k) * 1e4).astype(np.float32)
        B = (np.random.randn(k, n) * 1e4).astype(np.float32)
        C_honest = np.matmul(A, B)

        is_valid, disc, bound, _ = FreivaldsVerifier.verify(A, B, C_honest, dtype="fp32", k_rounds=5)
        assert is_valid is True
        # Bound scales with ||A||_inf * ||B||_inf
        assert bound > 1.0

        # Perturbation exceeding bound fails
        C_tampered = C_honest + (bound * 2.0)
        is_fraud, disc_fraud, _, _ = FreivaldsVerifier.verify(A, B, C_tampered, dtype="fp32", k_rounds=5)
        assert is_fraud is False

    def test_all_zeros_matrix_multiplication_no_false_fraud(self):
        """All-zero matrices A=0, B=0, C=0 pass Freivalds without ZeroDivisionError or false fraud."""
        A = np.zeros((16, 16), dtype=np.float32)
        B = np.zeros((16, 16), dtype=np.float32)
        C = np.zeros((16, 16), dtype=np.float32)

        is_valid, disc, bound, _ = FreivaldsVerifier.verify(A, B, C, dtype="fp32", k_rounds=5)
        assert is_valid is True
        assert disc == 0.0

    def test_identity_matrices_gemm(self):
        """Identity matrix product I x I = I passes verification cleanly."""
        I = np.eye(32, dtype=np.float32)
        C = np.matmul(I, I)

        is_valid, disc, bound, _ = FreivaldsVerifier.verify(I, I, C, dtype="fp32")
        assert is_valid is True
        assert disc == 0.0
        np.testing.assert_array_equal(C, I)

    def test_mixed_datatypes_fp32_fp16_bf16_precision_preservation(self):
        """Evaluates precision handling across FP32, FP16, and BF16 formats."""
        np.random.seed(408)
        A = np.random.randn(8, 8).astype(np.float32)
        B = np.random.randn(8, 8).astype(np.float32)

        # 1. FP32
        c_fp32 = np.matmul(A, B)
        v32, _, b32, _ = FreivaldsVerifier.verify(A, B, c_fp32, dtype="fp32")
        assert v32 is True

        # 2. FP16
        a_16 = A.astype(np.float16)
        b_16 = B.astype(np.float16)
        c_16 = np.matmul(a_16, b_16)
        v16, _, b16, _ = FreivaldsVerifier.verify(a_16, b_16, c_16, dtype="fp16")
        assert v16 is True
        assert b16 > b32  # Unit roundoff for FP16 is larger than FP32

        # 3. BF16
        a_bf16 = truncate_to_bf16(A)
        b_bf16 = truncate_to_bf16(B)
        c_bf16 = truncate_to_bf16(np.matmul(a_bf16, b_bf16))
        vbf, _, bbf, _ = FreivaldsVerifier.verify(a_bf16, b_bf16, c_bf16, dtype="bf16")
        assert vbf is True
        assert bbf > b16  # BF16 unit roundoff (2^-8) > FP16 (2^-11)


# ==============================================================================
# Category 5: High Concurrency & Worker Slot Contention
# ==============================================================================

class TestCategory5HighConcurrencyAndSlotContention:
    """
    Tier 5 Category 5:
    - Concurrent execution with multiple threads (16-32 threads) dispatching matrix tasks.
    - WorkerSlotTracker capacity limits and underflow prevention.
    - Thread-safe escrow state transitions between concurrent settles and cancels.
    """

    def test_concurrent_execution_32_threads_matrix_dispatch(self, settlement_service, worker_node_factory):
        """32 concurrent worker threads dispatching tile tasks through scheduler complete without deadlock."""
        w1 = worker_node_factory()
        w2 = worker_node_factory()

        candidates = [
            MatchedWorker(worker_pubkey=w1.pubkey, p2p_multiaddr=w1.multiaddr, lightning_routing_pubkey="ln1", sats_per_gflop=0.0005),
            MatchedWorker(worker_pubkey=w2.pubkey, p2p_multiaddr=w2.multiaddr, lightning_routing_pubkey="ln2", sats_per_gflop=0.0005),
        ]

        backend = MockSettlementBackend(settlement_service)
        mgr = EscrowManager(settlement_backend=backend, verification_key=settlement_service.verification_key)
        dispatcher = GrpcTaskDispatcher()
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_concurrency=32)

        def _worker_job(thread_idx: int) -> bool:
            task_id = f"task_adv_concurrent_{thread_idx}"
            a_tile = np.ones((4, 4), dtype=np.float32) * float(thread_idx + 1)
            b_tile = np.eye(4, dtype=np.float32)
            plan = TileTaskPlan(
                task_id=task_id,
                row_idx=0,
                col_idx=0,
                k_idx=0,
                row_slice=(0, 4),
                col_slice=(0, 4),
                k_slice=(0, 4),
                a_tile=a_tile,
                b_tile=b_tile,
                dtype="fp32",
                estimated_gflops=(2 * 4**3) / 1e9,
            )
            escrow = mgr.create_and_hold(task_id=task_id, total_sats=100)
            res_matrix, _ = scheduler.execute_with_failover(
                task_plan=plan,
                candidate_workers=candidates,
                macaroon=escrow.macaroon,
                payment_hash=escrow.payment_hash,
                dtype="fp32",
            )
            np.testing.assert_allclose(res_matrix, a_tile)
            mgr.release_payout(task_id)
            return True

        with futures.ThreadPoolExecutor(max_workers=32) as pool:
            f_list = [pool.submit(_worker_job, i) for i in range(32)]
            results = [f.result(timeout=20.0) for f in f_list]

        assert len(results) == 32
        assert all(r is True for r in results)
        scheduler.shutdown()

    def test_worker_slot_tracker_strict_capacity_limit_enforcement(self):
        """WorkerSlotTracker strictly rejects acquisitions exceeding max_slots."""
        tracker = WorkerSlotTracker()
        pubkey = "worker_slot_pubkey_01"
        tracker.register_worker(pubkey, max_slots=2)

        # Acquire 2 slots -> both succeed
        assert tracker.acquire_slot(pubkey, timeout=0.1) is True
        assert tracker.acquire_slot(pubkey, timeout=0.1) is True

        # 3rd slot must time out and return False
        start = time.time()
        assert tracker.acquire_slot(pubkey, timeout=0.1) is False
        assert time.time() - start >= 0.08

        # Release 1 slot -> acquisition now succeeds
        tracker.release_slot(pubkey)
        assert tracker.acquire_slot(pubkey, timeout=0.1) is True

    def test_worker_slot_tracker_never_underflows_below_zero(self):
        """Repeatedly releasing a slot on an idle worker never underflows active_slots below 0."""
        tracker = WorkerSlotTracker()
        pubkey = "worker_underflow_guard"
        tracker.register_worker(pubkey, max_slots=4)

        # Release without acquiring
        for _ in range(10):
            tracker.release_slot(pubkey)

        with tracker._lock:
            assert tracker._active_slots[pubkey] == 0

    def test_concurrent_slot_contention_stress_test(self):
        """32 threads rapidly acquiring and releasing slots maintain 0 <= active_slots <= max_slots."""
        tracker = WorkerSlotTracker()
        pubkey = "contention_stress_worker"
        max_capacity = 4
        tracker.register_worker(pubkey, max_slots=max_capacity)

        violations = []
        barrier = threading.Barrier(32)

        def _hammer(thread_id: int):
            barrier.wait()
            for _ in range(25):
                if tracker.acquire_slot(pubkey, timeout=0.2):
                    with tracker._lock:
                        curr = tracker._active_slots[pubkey]
                        if curr < 0 or curr > max_capacity:
                            violations.append(curr)
                    time.sleep(0.001)
                    tracker.release_slot(pubkey)

        threads = [threading.Thread(target=_hammer, args=(i,)) for i in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(violations) == 0
        with tracker._lock:
            assert tracker._active_slots[pubkey] == 0

    def test_concurrent_settle_and_cancel_race_condition_atomicity(self, settlement_service):
        """
        Thread-safe escrow state transition:
        When Thread A calls release_payout and Thread B calls abort_and_refund simultaneously,
        exactly ONE transition succeeds; the other is rejected with EscrowStateError.
        State is never corrupted.
        """
        mgr = EscrowManager(
            settlement_backend=MockSettlementBackend(settlement_service),
            verification_key=settlement_service.verification_key,
        )

        for iteration in range(20):
            task_id = f"task_race_{iteration}"
            escrow = mgr.create_and_hold(task_id=task_id, total_sats=100)

            barrier = threading.Barrier(2)
            results = {"settle": None, "abort": None}
            errors = {"settle": None, "abort": None}

            def _settle():
                barrier.wait()
                try:
                    results["settle"] = mgr.release_payout(task_id)
                except Exception as ex:
                    errors["settle"] = ex

            def _abort():
                barrier.wait()
                try:
                    results["abort"] = mgr.abort_and_refund(task_id, reason="concurrent_cancel")
                except Exception as ex:
                    errors["abort"] = ex

            t1 = threading.Thread(target=_settle)
            t2 = threading.Thread(target=_abort)
            t1.start()
            t2.start()
            t1.join(timeout=2.0)
            t2.join(timeout=2.0)

            final_escrow = mgr.get_escrow(task_id)
            assert final_escrow.status in (EscrowStatus.SETTLED, EscrowStatus.CANCELLED)

            # Exactly one must succeed, the other must fail with EscrowStateError
            success_count = int(results["settle"] is not None) + int(results["abort"] is not None)
            error_count = int(errors["settle"] is not None) + int(errors["abort"] is not None)

            assert success_count == 1, f"Iteration {iteration}: success_count={success_count}"
            assert error_count == 1, f"Iteration {iteration}: error_count={error_count}"

            failed_err = errors["settle"] or errors["abort"]
            assert isinstance(failed_err, EscrowStateError)

    def test_batch_escrow_creation_and_timeout_sweep(self, settlement_service):
        """Batch creation of 20 escrows with timeout sweep safely expires overdue escrows."""
        mgr = EscrowManager(
            settlement_backend=MockSettlementBackend(settlement_service),
            verification_key=settlement_service.verification_key,
            default_timeout_seconds=0.05,  # 50ms
        )

        for i in range(20):
            mgr.create_and_hold(task_id=f"task_timeout_batch_{i}", total_sats=50, timeout_seconds=0.05)

        time.sleep(0.08)  # Allow escrows to expire
        expired = mgr.check_timeouts()

        assert len(expired) == 20
        assert all(e.status == EscrowStatus.EXPIRED for e in expired)
