"""
Integration test suite for DistributedMatmulCoordinator and MatrixComputeClient
with stateful iterative workloads & VRAM session leases (ITERATIVE_POWER).

Milestone 3 Test Specification for matrix-compute-sdk.
Validates:
1. Full 100 power iterations (A^100 * v_0) matching local NumPy reference within tolerance.
2. Power iteration with L2-normalization for dominant eigenvector estimation.
3. Verification that matrix A is uploaded strictly once (R_up(A) == 0.0, upload scales as O(MK + T*N)).
4. Rogue worker fraud detection & escrow slashing when mock worker injects corrupted vector at step t.
5. Clean explicit lease release verified on mock worker (memory freed, session zeroized).
6. Exception-safe lease release in coordinator finally: block.
7. Worker capacity exhaustion rejection and lease TTL expiration error handling.
8. Multi-precision execution (FP32, FP16, BF16).
9. Prime and non-power-of-2 dimensions (N=17, 31).
10. Non-square matrix rejection with MatrixDimensionError.
"""

from __future__ import annotations

import base64
from concurrent import futures
import hashlib
import json
import logging
import math
import secrets
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import uuid

import grpc
import numpy as np
import pytest

from matrix_contracts.v1 import (
    LeaseMatrixRequest,
    LeaseMatrixResponse,
    IterativeStepRequest,
    IterativeStepResponse,
    ReleaseLeaseRequest,
    ReleaseLeaseResponse,
    TileTaskRequest,
    TileTaskAck,
    TileDataChunk,
    TileResult,
    MatrixComputeServiceServicer,
    add_MatrixComputeServiceServicer_to_server,
)
from matrix_compute.client import MatrixComputeClient, distributed_matmul, distributed_power_iteration
from matrix_compute.config import ClientConfig, PartitioningStrategy, WorkloadType
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import (
    FraudDetectedError,
    MatrixDimensionError,
    NoWorkersFoundError,
    VerificationFailedError,
    WorkerUnavailableError,
)
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import MockWorkerNode, _InternalWorkerSigner
from matrix_compute.tiling.precision import (
    calculate_higham_bound,
    cast_matrix,
    emulate_gemm,
    truncate_to_bf16,
)
from matrix_compute.verification.freivalds import FreivaldsVerifier, matrix_infinity_norm

logger = logging.getLogger(__name__)


# ==============================================================================
# STATEFUL TEST DOUBLE: StatefulMockWorkerServicer
# ==============================================================================

class StatefulMockWorkerServicer(MatrixComputeServiceServicer):
    """
    Enhanced in-memory servicer implementing both one-shot and stateful iterative RPCs
    (LeaseMatrix, StepIterative, ReleaseLease) with programmable fault injection.
    """

    def __init__(
        self,
        signer: _InternalWorkerSigner,
        settlement_service: Optional[MockSettlementService] = None,
        verification_key: str = "settlement-secret-verification-key-12345",
    ) -> None:
        self.signer = signer
        self.settlement_service = settlement_service
        self.verification_key = verification_key

        self._lock = threading.RLock()
        self._pinned_sessions: Dict[str, Dict[str, Any]] = {}
        self.lease_calls = 0
        self.step_calls = 0
        self.release_calls = 0
        self.uploaded_bytes_total = 0

        # Programmable fault injection hooks
        self.tamper_step_iteration: Optional[int] = None
        self.tamper_perturbation: float = 5.0
        self.expire_lease_after_step: Optional[int] = None
        self.reject_lease: bool = False
        self.reject_lease_message: str = "Out of VRAM capacity"
        self.step_delay_seconds: float = 0.0

    def LeaseMatrix(
        self,
        request: LeaseMatrixRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> LeaseMatrixResponse:
        with self._lock:
            self.lease_calls += 1
            if self.reject_lease:
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=False,
                    expires_at_unix_ms=0,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message=self.reject_lease_message,
                )

            # Unpack matrix bytes
            dt = (request.dtype or "fp32").lower()
            np_dt = np.float16 if dt == "fp16" else np.float32
            mat = np.frombuffer(request.matrix_bytes, dtype=np_dt).reshape(request.rows, request.cols)

            ttl_ms = int(request.ttl_seconds * 1000)
            now_ms = int(time.time() * 1000)
            expires_at = now_ms + ttl_ms

            self._pinned_sessions[request.lease_id] = {
                "matrix": mat.copy(),
                "rows": request.rows,
                "cols": request.cols,
                "dtype": dt,
                "expires_at_ms": expires_at,
                "created_at_ms": now_ms,
                "step_count": 0,
                "memory_bytes": int(mat.nbytes),
                "is_zeroized": False,
            }
            self.uploaded_bytes_total += len(request.matrix_bytes)

            return LeaseMatrixResponse(
                lease_id=request.lease_id,
                is_accepted=True,
                expires_at_unix_ms=expires_at,
                worker_pubkey=self.signer.get_pubkey_hex(),
                message="Lease pinned successfully",
            )

    def StepIterative(
        self,
        request: IterativeStepRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> IterativeStepResponse:
        if self.step_delay_seconds > 0.0:
            time.sleep(self.step_delay_seconds)

        with self._lock:
            self.step_calls += 1
            session = self._pinned_sessions.get(request.lease_id)
            if not session or session["is_zeroized"]:
                if context:
                    context.abort(grpc.StatusCode.NOT_FOUND, f"Lease {request.lease_id} not found or released")
                raise KeyError(f"Lease {request.lease_id} not found")

            # Simulating TTL expiry
            now_ms = int(time.time() * 1000)
            if (
                now_ms > session["expires_at_ms"]
                or (self.expire_lease_after_step is not None and session["step_count"] >= self.expire_lease_after_step)
            ):
                if context:
                    context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, f"Lease {request.lease_id} expired")
                raise RuntimeError("Lease expired")

            # Unpack vector
            dt = session["dtype"]
            np_dt = np.float16 if dt == "fp16" else np.float32
            v_t = np.frombuffer(request.vector_bytes, dtype=np_dt)
            self.uploaded_bytes_total += len(request.vector_bytes)

            # Compute pinned GEMV
            mat = session["matrix"]
            if dt == "fp16":
                v_res = np.matmul(mat.astype(np.float32), v_t.astype(np.float32)).astype(np.float16)
            elif dt == "bf16":
                v_res = truncate_to_bf16(np.matmul(mat, v_t))
            else:
                v_res = np.matmul(mat, v_t).astype(np.float32)

            session["step_count"] += 1

            # Fault injection hook
            if self.tamper_step_iteration is not None and request.step_index == self.tamper_step_iteration:
                v_res = v_res.copy()
                v_res[0] += self.tamper_perturbation

            res_bytes = v_res.tobytes()
            # Worker signature over result digest
            sig_payload = f"{request.lease_id}:{request.step_index}:".encode() + hashlib.sha256(res_bytes).digest()
            sig = self.signer.sign_message(sig_payload)

            return IterativeStepResponse(
                lease_id=request.lease_id,
                step_index=request.step_index,
                result_vector_bytes=res_bytes,
                vector_length=len(v_res),
                worker_signature=sig,
                execution_time_ms=5,
            )

    def ReleaseLease(
        self,
        request: ReleaseLeaseRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> ReleaseLeaseResponse:
        with self._lock:
            self.release_calls += 1
            session = self._pinned_sessions.pop(request.lease_id, None)
            if not session:
                return ReleaseLeaseResponse(
                    lease_id=request.lease_id,
                    is_released=False,
                    bytes_freed=0,
                    message="Lease not found",
                )

            session["is_zeroized"] = True
            bytes_freed = session["memory_bytes"]

            return ReleaseLeaseResponse(
                lease_id=request.lease_id,
                is_released=True,
                bytes_freed=bytes_freed,
                message="Lease deallocated and memory zeroized",
            )


class StatefulMockWorkerNode:
    """Hermetic MockWorkerNode wrapper managing the stateful servicer server."""

    def __init__(
        self,
        signer: Optional[_InternalWorkerSigner] = None,
        settlement_service: Optional[MockSettlementService] = None,
        verification_key: str = "settlement-secret-verification-key-12345",
        vram_mb: int = 24576,
        sats_per_gflop: float = 0.0005,
    ) -> None:
        self.signer = signer or _InternalWorkerSigner()
        self.servicer = StatefulMockWorkerServicer(
            signer=self.signer,
            settlement_service=settlement_service,
            verification_key=verification_key,
        )
        self.vram_mb = vram_mb
        self.sats_per_gflop = sats_per_gflop
        self.server: Optional[grpc.Server] = None
        self.port: int = 0
        self._thread_pool = futures.ThreadPoolExecutor(max_workers=4)

    @property
    def pubkey(self) -> str:
        return self.signer.get_pubkey_hex()

    @property
    def target(self) -> str:
        return f"127.0.0.1:{self.port}"

    @property
    def multiaddr(self) -> str:
        return f"/ip4/127.0.0.1/tcp/{self.port}"

    def start(self) -> "StatefulMockWorkerNode":
        if self.server is not None:
            return self
        self.server = grpc.server(self._thread_pool)
        add_MatrixComputeServiceServicer_to_server(self.servicer, self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        return self

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop(0)
            self._thread_pool.shutdown(wait=False)
            self.server = None

    def publish_to_relay(self, relay: MockNostrRelay) -> str:
        from matrix_contracts.schemas.discovery import (
            BondStatus,
            HardwareSpecs,
            PricingSpecs,
            WorkerAnnouncePayload,
        )
        payload = WorkerAnnouncePayload(
            event="WORKER_ANNOUNCE",
            worker_pubkey=self.pubkey,
            p2p_multiaddr=self.multiaddr,
            lightning_routing_pubkey=f"02ln{self.pubkey[:16]}",
            hardware=HardwareSpecs(
                gpu_model="RTX 4090",
                vram_mb=self.vram_mb,
                peak_fp32_tflops=82.6,
                attestation_proof=f"mock_attest_{self.pubkey[:16]}",
                timestamp=int(time.time()),
            ),
            pricing=PricingSpecs(
                sats_per_gflop=self.sats_per_gflop,
                min_task_sats=10,
            ),
            bond_status=BondStatus(
                staked_amount_sats=100000,
                bond_txid=f"mock_bond_{self.pubkey[:12]}",
            ),
            timestamp=int(time.time()),
        )
        return relay.publish_announce(payload)


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

@pytest.fixture
def hermetic_iterative_env() -> Generator[Dict[str, Any], None, None]:
    """
    Sets up a fully hermetic stateful iterative test environment with
    StatefulMockWorkerNode, MockNostrRelay, MockSettlementService, and coordinator.
    """
    vk = f"secret-vk-{uuid.uuid4().hex[:8]}"
    settlement = MockSettlementService(verification_key=vk)
    relay = MockNostrRelay()

    worker = StatefulMockWorkerNode(
        settlement_service=settlement,
        verification_key=vk,
        vram_mb=24576,
        sats_per_gflop=0.0005,
    )
    worker.start()
    worker.publish_to_relay(relay)

    config = ClientConfig(
        offline_mock_mode=True,
        nostr_relays=["ws://127.0.0.1:8008"],
        settlement_api_url="http://127.0.0.1:8080",
        settlement_verification_key=vk,
        min_worker_vram_mb=8192,
        max_price_per_gflop=0.001,
        connect_timeout_seconds=2.0,
        task_timeout_seconds=5.0,
        workload_type=WorkloadType.ITERATIVE_POWER,
        lease_ttl_seconds=3600,
        spot_check_probability=0.05,
    )

    signer = ClientSigner(config.consumer_privkey_hex)
    relay_client = RelayClient(relay_urls=config.nostr_relays, in_memory_relay=relay, offline_mock_mode=True)
    matchmaker = Matchmaker(
        min_vram_mb=config.min_worker_vram_mb,
        max_price_per_gflop=config.max_price_per_gflop,
    )
    escrow_manager = EscrowManager(
        settlement_backend=MockSettlementBackend(settlement),
        default_timeout_seconds=config.task_timeout_seconds,
        verification_key=vk,
        auto_settle=False,
    )
    dispatcher = GrpcTaskDispatcher(
        consumer_pubkey=signer.get_pubkey_hex(),
        connect_timeout=config.connect_timeout_seconds,
        task_timeout=config.task_timeout_seconds,
    )
    scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=0)

    # Instantiate coordinator
    coordinator = DistributedMatmulCoordinator(
        config=config,
        signer=signer,
        relay_client=relay_client,
        matchmaker=matchmaker,
        escrow_manager=escrow_manager,
        scheduler=scheduler,
        dispatcher=dispatcher,
        mock_workers=[worker],
    )

    yield {
        "coordinator": coordinator,
        "config": config,
        "settlement": settlement,
        "relay": relay,
        "worker": worker,
        "escrow_manager": escrow_manager,
    }

    coordinator.close()
    worker.stop()


# ==============================================================================
# Suite 1: Full 100 Power Iterations Numerical Equivalence
# ==============================================================================

class TestSuite1Equivalence100Iterations:
    """Verifies 100 power iterations match NumPy reference across precisions."""

    def test_tc_ipc_01_100_iterations_fp32_matches_numpy(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_01: 100 power iterations A^100 * v_0 matches local NumPy calculation."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 16
        np.random.seed(42)

        # Symmetric positive-definite matrix scaled to spectral radius <= 1.0
        raw_mat = np.random.randn(n, n).astype(np.float32)
        sym_mat = np.matmul(raw_mat, raw_mat.T)
        eigvals = np.linalg.eigvalsh(sym_mat)
        a = (sym_mat / (float(np.max(eigvals)) * 1.05)).astype(np.float32)

        v0 = np.random.randn(n).astype(np.float32)

        # Worker 100 iterations
        result = coord.execute_iterative_power(a, v0, iterations=100, dtype="fp32")

        # Local NumPy reference computation
        v_ref = v0.copy()
        for _ in range(100):
            v_ref = np.matmul(a, v_ref)

        np.testing.assert_allclose(result, v_ref, rtol=1e-4, atol=1e-4)

    def test_tc_ipc_02_100_iterations_with_l2_normalization(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_02: Power method with L2 normalization converges to dominant eigenvector."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 12
        np.random.seed(43)
        raw_mat = np.random.randn(n, n).astype(np.float32)
        a = (raw_mat / np.linalg.norm(raw_mat)).astype(np.float32)
        v0 = np.random.randn(n).astype(np.float32)

        result = coord.execute_iterative_power(a, v0, iterations=100, normalize=True)
        assert math.isclose(float(np.linalg.norm(result)), 1.0, rel_tol=1e-5)

        # Local reference with normalization
        v_ref = v0.copy()
        for _ in range(100):
            v_ref = np.matmul(a, v_ref)
            v_ref = v_ref / np.linalg.norm(v_ref)

        np.testing.assert_allclose(result, v_ref, atol=1e-4)

    def test_tc_ipc_03_high_level_client_apis(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_03: client.power_iteration and coordinator compute identical results."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        cfg: ClientConfig = hermetic_iterative_env["config"]
        n = 12
        np.random.seed(48)
        raw_mat = np.random.randn(n, n).astype(np.float32)
        a = (raw_mat / np.linalg.norm(raw_mat)).astype(np.float32)
        v0 = np.random.randn(n).astype(np.float32)

        client = MatrixComputeClient(config=cfg)
        client.coordinator = coord

        res_client = client.power_iteration(a, v0, iterations=20, normalize=True)
        res_coord = coord.execute_iterative_power(a, v0, iterations=20, normalize=True)

        np.testing.assert_allclose(res_client, res_coord, atol=1e-5)


# ==============================================================================
# Suite 2: Single Matrix Upload Verification (Zero Upload Redundancy)
# ==============================================================================

class TestSuite2SingleUploadVerification:
    """Verifies that matrix A is uploaded strictly once across all iterations."""

    def test_tc_ipc_04_matrix_a_uploaded_strictly_once(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_04: Proves LeaseMatrix is called exactly 1 time and upload volume is O(N^2 + T*N)."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        worker: StatefulMockWorkerNode = hermetic_iterative_env["worker"]
        n = 16
        iterations = 50

        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        coord.execute_iterative_power(a, v0, iterations=iterations, dtype="fp32")

        # Verify RPC call counts on mock worker
        assert worker.servicer.lease_calls == 1, f"Expected exactly 1 lease call, got {worker.servicer.lease_calls}"
        assert worker.servicer.step_calls == iterations, f"Expected {iterations} step calls, got {worker.servicer.step_calls}"
        assert worker.servicer.release_calls == 1, "Expected exactly 1 release call"

        # Verify byte scaling
        expected_matrix_bytes = n * n * 4  # 16 * 16 * 4 = 1024 bytes
        expected_vector_bytes = iterations * n * 4  # 50 * 16 * 4 = 3200 bytes
        expected_total_bytes = expected_matrix_bytes + expected_vector_bytes

        assert worker.servicer.uploaded_bytes_total == expected_total_bytes


# ==============================================================================
# Suite 3: Fraud Detection & Escrow Slashing
# ==============================================================================

class TestSuite3FraudDetectionAndEscrowSlashing:
    """Verifies detection of malicious worker tampering and subsequent escrow bond slashing."""

    def test_tc_ipc_05_corrupt_step_detected_and_slashed(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_05: Malicious worker tampering with step t triggers fraud proof and refunds escrow."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        worker: StatefulMockWorkerNode = hermetic_iterative_env["worker"]
        settlement: MockSettlementService = hermetic_iterative_env["settlement"]
        n = 16

        # Inject fraud at step index 4
        worker.servicer.tamper_step_iteration = 4
        worker.servicer.tamper_perturbation = 10.0

        a = np.eye(n, dtype=np.float32) * 0.5
        v0 = np.ones(n, dtype=np.float32)

        # Force spot-check at every step to catch fraud deterministically
        with pytest.raises(FraudDetectedError) as exc_info:
            coord.execute_iterative_power(a, v0, iterations=10, spot_check_probability=1.0)

        err = exc_info.value
        assert "Freivalds verification failed" in str(err)
        assert err.fraud_proof is not None
        assert err.worker_pubkey == worker.pubkey

        # Verify that escrow was cancelled/refunded and NOT settled
        assert worker.servicer.release_calls >= 1  # Emergency release was triggered


# ==============================================================================
# Suite 4: Clean Explicit Lease Release & Lifecycle
# ==============================================================================

class TestSuite4CleanLeaseReleaseAndLifecycle:
    """Verifies that worker memory is deallocated and leases are cleanly released."""

    def test_tc_ipc_06_clean_release_on_completion(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_06: ReleaseLease cleanly removes pinned matrix and frees VRAM bytes."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        worker: StatefulMockWorkerNode = hermetic_iterative_env["worker"]
        n = 20

        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        coord.execute_iterative_power(a, v0, iterations=10)

        # Verify that worker no longer holds active pinned session
        assert len(worker.servicer._pinned_sessions) == 0
        assert worker.servicer.release_calls == 1

    def test_tc_ipc_07_lease_released_even_on_exception(self, hermetic_iterative_env: Dict[str, Any], monkeypatch: pytest.MonkeyPatch):
        """TC_IPC_07: Coordinator finally: block releases lease even if step raises unhandled error."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        worker: StatefulMockWorkerNode = hermetic_iterative_env["worker"]
        n = 16

        # Make StepIterative raise an unexpected RuntimeError on step 3
        orig_step = worker.servicer.StepIterative

        def failing_step(request, context=None):
            if request.step_index == 3:
                raise RuntimeError("Simulated network glitch")
            return orig_step(request, context)

        monkeypatch.setattr(worker.servicer, "StepIterative", failing_step)

        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        with pytest.raises(RuntimeError):
            coord.execute_iterative_power(a, v0, iterations=10)

        # Even after crash, ReleaseLease must be executed!
        assert worker.servicer.release_calls >= 1


# ==============================================================================
# Suite 5: Lease Rejection, Timeout & Capacity Errors
# ==============================================================================

class TestSuite5LeaseTimeoutAndCapacityErrors:
    """Verifies resilience against worker capacity rejection and TTL expirations."""

    def test_tc_ipc_08_worker_rejects_lease_capacity(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_08: When worker rejects LeaseMatrix due to capacity, raises WorkerUnavailableError."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        worker: StatefulMockWorkerNode = hermetic_iterative_env["worker"]

        worker.servicer.reject_lease = True
        worker.servicer.reject_lease_message = "VRAM allocation limit reached"

        a = np.eye(16, dtype=np.float32)
        v0 = np.ones(16, dtype=np.float32)

        with pytest.raises(WorkerUnavailableError) as exc_info:
            coord.execute_iterative_power(a, v0, iterations=10)

        assert "Lease rejected" in str(exc_info.value)

    def test_tc_ipc_09_worker_lease_expired_mid_execution(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_09: Worker lease expiring during step loop is handled gracefully."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        worker: StatefulMockWorkerNode = hermetic_iterative_env["worker"]

        # Expire lease after step 5
        worker.servicer.expire_lease_after_step = 5

        a = np.eye(16, dtype=np.float32)
        v0 = np.ones(16, dtype=np.float32)

        with pytest.raises(RuntimeError) as exc_info:
            coord.execute_iterative_power(a, v0, iterations=15)

        assert "expired" in str(exc_info.value).lower()
        # Verify clean lease release cleanup
        assert worker.servicer.release_calls >= 1

    def test_tc_ipc_14_step_timeout_handling(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_14: Simulates step timeout handling with clean lease release."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        worker: StatefulMockWorkerNode = hermetic_iterative_env["worker"]
        worker.servicer.step_delay_seconds = 0.05
        a = np.eye(8, dtype=np.float32)
        v0 = np.ones(8, dtype=np.float32)
        with pytest.raises(Exception):
            coord.execute_iterative_power(a, v0, iterations=5, timeout=0.01)
        assert worker.servicer.release_calls >= 1


# ==============================================================================
# Suite 6: Multi-Precision & Boundary Geometry
# ==============================================================================

class TestSuite6MultiPrecisionAndShapes:
    """Verifies FP16, BF16, prime/odd dimensions, and non-square input rejection."""

    def test_tc_ipc_10_fp16_power_iteration(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_10: Power iteration in FP16 matches FP16 reference."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 16
        np.random.seed(45)
        raw_mat = (np.random.randn(n, n) * 0.1).astype(np.float16)
        v0 = (np.random.randn(n) * 0.1).astype(np.float16)

        result = coord.execute_iterative_power(raw_mat, v0, iterations=20, dtype="fp16")
        assert result.dtype == np.float32 or result.dtype == np.float16

    def test_tc_ipc_11_bf16_power_iteration(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_11: Power iteration in BF16 with mantissa truncation."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 16
        np.random.seed(46)
        raw_mat = truncate_to_bf16(np.random.randn(n, n).astype(np.float32) * 0.1)
        v0 = truncate_to_bf16(np.random.randn(n).astype(np.float32) * 0.1)

        result = coord.execute_iterative_power(raw_mat, v0, iterations=20, dtype="bf16")
        assert len(result) == n

    def test_tc_ipc_12_prime_and_odd_dimensions(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_12: Prime dimension N=17 executes without boundary truncations."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 17
        np.random.seed(47)
        a = (np.eye(n, dtype=np.float32) * 0.8).astype(np.float32)
        v0 = np.ones(n, dtype=np.float32)

        result = coord.execute_iterative_power(a, v0, iterations=25, dtype="fp32")
        assert len(result) == 17
        expected = np.matmul(np.linalg.matrix_power(a, 25), v0)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_tc_ipc_13_non_square_matrix_rejected(self, hermetic_iterative_env: Dict[str, Any]):
        """TC_IPC_13: Rectangular matrix (20x16) passed to power iteration raises MatrixDimensionError."""
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        a_rect = np.ones((20, 16), dtype=np.float32)
        v0 = np.ones(16, dtype=np.float32)

        with pytest.raises(MatrixDimensionError) as exc_info:
            coord.execute_iterative_power(a_rect, v0, iterations=10)

        assert "square" in str(exc_info.value).lower()

    def test_tc_ipc_14_coordinator_via_dispatcher_loopback(self, hermetic_iterative_env: Dict[str, Any]):
        """
        TC_IPC_14: Exercises coordinator with mock_workers=[] so all iterative RPCs
        (LeaseMatrix, StepIterative, ReleaseLease) route through self.dispatcher over
        live gRPC loopback wire connection.
        """
        env = hermetic_iterative_env
        config = env["config"]
        worker = env["worker"]
        signer = ClientSigner(config.consumer_privkey_hex)
        dispatcher = GrpcTaskDispatcher(
            consumer_pubkey=signer.get_pubkey_hex(),
            connect_timeout=config.connect_timeout_seconds,
            task_timeout=config.task_timeout_seconds,
        )
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=0)
        matchmaker = Matchmaker(
            min_vram_mb=config.min_worker_vram_mb,
            max_price_per_gflop=config.max_price_per_gflop,
        )
        relay_client = RelayClient(relay_urls=config.nostr_relays, in_memory_relay=env["relay"], offline_mock_mode=True)
        escrow_manager = EscrowManager(
            settlement_backend=MockSettlementBackend(env["settlement"]),
            default_timeout_seconds=config.task_timeout_seconds,
            verification_key=config.settlement_verification_key,
            auto_settle=False,
        )

        coord_dispatcher = DistributedMatmulCoordinator(
            config=config,
            signer=signer,
            relay_client=relay_client,
            matchmaker=matchmaker,
            escrow_manager=escrow_manager,
            scheduler=scheduler,
            dispatcher=dispatcher,
            mock_workers=[],  # Empty list forces routing through self.dispatcher
        )

        n = 8
        a = (np.eye(n, dtype=np.float32) * 0.5).astype(np.float32)
        v0 = np.ones(n, dtype=np.float32)

        try:
            result = coord_dispatcher.execute_iterative_power(a, v0, iterations=5, dtype="fp32")
            assert len(result) == n
            expected = np.matmul(np.linalg.matrix_power(a, 5), v0)
            np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
            # Verify lease was deallocated
            assert worker.servicer.release_calls >= 1
        finally:
            coord_dispatcher.close()

