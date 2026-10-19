"""
Adversarial Stress Testing Suite for Consumer SDK Stateful Session Coordinator (ITERATIVE_POWER).
Milestone 3 Security, Fault Tolerance, and Edge Case Verification.

Focus Areas:
1. Malicious Worker Simulation & Escrow Slashing:
   - Corrupted step vector at arbitrary step t in [0, T-1].
   - Forged worker Secp256k1 signature on step t.
   - Authoritative FreivaldsFraudProof schema and invariant validation.
   - L402 escrow abort and refund verification (status CANCELLED, not SETTLED).
   - Guaranteed ReleaseLease execution in coordinator finally: block.
2. Lifecycle Edge Cases:
   - Worker lease rejection handling (is_accepted=False).
   - Worker lease expiration mid-loop (DEADLINE_EXCEEDED or session expired).
   - High-iteration loop (100 steps) numerical equivalence against NumPy reference.
3. Security Boundary & Vulnerability Probes:
   - Signature omission bypass probe.
   - Pinned VRAM zeroization and session deallocation.
   - Matrix upload single-shot invariance (R_up(A) == 0.0).
"""

from __future__ import annotations

import copy
import hashlib
import math
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
import uuid

import grpc
import numpy as np
import pytest

from matrix_contracts.schemas.fraud import FreivaldsFraudProof
from matrix_contracts.v1 import (
    IterativeStepRequest,
    IterativeStepResponse,
    LeaseMatrixRequest,
    LeaseMatrixResponse,
    MatrixComputeServiceServicer,
    ReleaseLeaseRequest,
    ReleaseLeaseResponse,
    add_MatrixComputeServiceServicer_to_server,
)
from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig, WorkloadType
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import (
    FraudDetectedError,
    MatrixDimensionError,
    WorkerSignatureError,
    WorkerUnavailableError,
)
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator, IterativeResult
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import _InternalWorkerSigner
from matrix_compute.tiling.precision import truncate_to_bf16


# ==============================================================================
# ADVERSARIAL TEST DOUBLE: AdversarialMockWorkerServicer
# ==============================================================================

class AdversarialMockWorkerServicer(MatrixComputeServiceServicer):
    """
    Advanced programmable test double servicer for simulating rogue workers,
    cryptographic forgery, stateful session leaks, and protocol violations.
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

        # Fault injection hooks
        self.reject_lease: bool = False
        self.reject_lease_message: str = "Out of VRAM capacity"
        self.expire_lease_after_step: Optional[int] = None
        self.step_delay_seconds: float = 0.0

        # Vector corruption hooks
        self.corrupt_steps: Dict[int, Dict[str, Any]] = {}

        # Signature manipulation hooks
        self.forged_signature_steps: Set[int] = set()
        self.omit_signature_steps: Set[int] = set()
        self.attacker_signer: Optional[_InternalWorkerSigner] = None

    def inject_vector_corruption(
        self,
        step_index: int,
        mode: str = "additive",
        perturbation: float = 10.0,
    ) -> None:
        """Configures vector corruption for a specific step index."""
        self.corrupt_steps[step_index] = {
            "mode": mode,
            "perturbation": perturbation,
        }

    def inject_forged_signature(self, step_index: int) -> None:
        """Injects a Secp256k1 signature created by an untrusted attacker key."""
        self.forged_signature_steps.add(step_index)
        if self.attacker_signer is None:
            self.attacker_signer = _InternalWorkerSigner()

    def inject_omitted_signature(self, step_index: int) -> None:
        """Strips the signature completely (empty string) on step_index."""
        self.omit_signature_steps.add(step_index)

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

            # Check simulated lease expiration
            now_ms = int(time.time() * 1000)
            if (
                now_ms > session["expires_at_ms"]
                or (self.expire_lease_after_step is not None and session["step_count"] >= self.expire_lease_after_step)
            ):
                if context:
                    context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, f"Lease {request.lease_id} expired")
                raise RuntimeError("Lease expired")

            dt = session["dtype"]
            np_dt = np.float16 if dt == "fp16" else np.float32
            v_t = np.frombuffer(request.vector_bytes, dtype=np_dt)
            self.uploaded_bytes_total += len(request.vector_bytes)

            mat = session["matrix"]
            if dt == "fp16":
                v_res = np.matmul(mat.astype(np.float32), v_t.astype(np.float32)).astype(np.float16)
            elif dt == "bf16":
                v_res = truncate_to_bf16(np.matmul(mat, v_t))
            else:
                v_res = np.matmul(mat, v_t).astype(np.float32)

            session["step_count"] += 1

            # Vector corruption attack injection
            if request.step_index in self.corrupt_steps:
                spec = self.corrupt_steps[request.step_index]
                mode = spec["mode"]
                v_res = v_res.copy()
                if mode == "additive":
                    v_res[0] += spec["perturbation"]
                elif mode == "zero":
                    v_res[:] = 0.0
                elif mode == "sign_flip":
                    v_res = -v_res
                elif mode == "nan":
                    v_res[0] = np.nan
                elif mode == "inf":
                    v_res[0] = np.inf

            res_bytes = v_res.tobytes()

            # Signature generation / tampering
            if request.step_index in self.omit_signature_steps:
                sig = ""
            elif request.step_index in self.forged_signature_steps:
                sig_payload = f"{request.lease_id}:{request.step_index}:".encode() + hashlib.sha256(res_bytes).digest()
                sig = self.attacker_signer.sign_message(sig_payload)
            else:
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


class AdversarialMockWorkerNode:
    """Hermetic node wrapper managing AdversarialMockWorkerServicer."""

    def __init__(
        self,
        signer: Optional[_InternalWorkerSigner] = None,
        settlement_service: Optional[MockSettlementService] = None,
        verification_key: str = "settlement-secret-verification-key-12345",
        vram_mb: int = 24576,
        sats_per_gflop: float = 0.0005,
    ) -> None:
        self.signer = signer or _InternalWorkerSigner()
        self.servicer = AdversarialMockWorkerServicer(
            signer=self.signer,
            settlement_service=settlement_service,
            verification_key=verification_key,
        )
        self.vram_mb = vram_mb
        self.sats_per_gflop = sats_per_gflop

    @property
    def pubkey(self) -> str:
        return self.signer.get_pubkey_hex()

    @property
    def target(self) -> str:
        return "127.0.0.1:50051"

    @property
    def multiaddr(self) -> str:
        return "/ip4/127.0.0.1/tcp/50051"

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
# FIXTURES
# ==============================================================================

@pytest.fixture
def adversarial_env() -> Generator[Dict[str, Any], None, None]:
    """Sets up hermetic environment for adversarial stress testing."""
    vk = f"secret-vk-{uuid.uuid4().hex[:8]}"
    settlement = MockSettlementService(verification_key=vk)
    relay = MockNostrRelay()

    worker = AdversarialMockWorkerNode(
        settlement_service=settlement,
        verification_key=vk,
        vram_mb=24576,
        sats_per_gflop=0.0005,
    )
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
    relay_client = RelayClient(relay_urls=config.nostr_relays, in_memory_relay=relay)
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


# ==============================================================================
# 1. MALICIOUS WORKER SIMULATION & ESCROW SLASHING
# ==============================================================================

class TestMaliciousWorkerSimulationAndSlashing:
    """
    Stress-tests coordinator response to malicious vector corruption at arbitrary steps:
    asserts FraudDetectedError, valid FreivaldsFraudProof, escrow abort/refund, and ReleaseLease.
    """

    @pytest.mark.parametrize("corrupt_step, total_steps", [
        (0, 10),   # Earliest step t = 0
        (1, 10),   # Early step t = 1
        (5, 10),   # Mid-loop step t = 5
        (9, 10),   # Final step t = T - 1
    ])
    def test_corrupted_step_vector_at_arbitrary_step_triggers_fraud_and_escrow_refund(
        self,
        adversarial_env: Dict[str, Any],
        corrupt_step: int,
        total_steps: int,
    ) -> None:
        """
        Injects corrupted step vector at arbitrary step t in [0, T-1].
        Asserts:
        1. FraudDetectedError is raised.
        2. FreivaldsFraudProof is strictly valid and populated.
        3. L402 escrow is aborted and refunded (status CANCELLED, not SETTLED).
        4. ReleaseLease RPC is executed in coordinator finally: block.
        """
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        worker: AdversarialMockWorkerNode = adversarial_env["worker"]
        escrow_mgr: EscrowManager = adversarial_env["escrow_manager"]

        n = 16
        a = (np.eye(n, dtype=np.float32) * 0.8).astype(np.float32)
        v0 = np.ones(n, dtype=np.float32)

        # Configure worker to inject gross mathematical fraud at corrupt_step
        worker.servicer.inject_vector_corruption(step_index=corrupt_step, mode="additive", perturbation=15.0)

        # Force spot-check at corrupt_step
        with pytest.raises(FraudDetectedError) as exc_info:
            coord.execute_iterative_power(
                a,
                v0,
                iterations=total_steps,
                forced_spot_check_indices={corrupt_step},
            )

        err = exc_info.value
        assert f"step {corrupt_step}" in str(err)
        assert err.fraud_proof is not None

        # Verify FreivaldsFraudProof invariants
        proof: FreivaldsFraudProof = err.fraud_proof
        assert isinstance(proof, FreivaldsFraudProof)
        assert proof.dimension_m == n
        assert proof.dimension_k == n
        assert proof.dimension_n == 1
        assert proof.random_vector_r == [1]
        assert proof.max_observed_discrepancy > proof.epsilon_tolerance
        assert proof.worker_pubkey == worker.pubkey
        assert f"step_{corrupt_step}" in proof.task_id

        # Verify escrow status: must be CANCELLED, never SETTLED
        active_escrows = list(escrow_mgr._escrows.values())
        assert len(active_escrows) == 1
        escrow = active_escrows[0]
        assert escrow.status == EscrowStatus.CANCELLED
        assert escrow.status != EscrowStatus.SETTLED

        # Verify ReleaseLease RPC was executed in finally: block
        assert worker.servicer.release_calls >= 1
        assert len(worker.servicer._pinned_sessions) == 0

    @pytest.mark.parametrize("fraud_mode", ["zero", "sign_flip"])
    def test_adversarial_zero_work_and_sign_flip_fraud(
        self,
        adversarial_env: Dict[str, Any],
        fraud_mode: str,
    ) -> None:
        """Tests zero-work vector (v_res = 0) and sign-inversion attacks."""
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        worker: AdversarialMockWorkerNode = adversarial_env["worker"]
        escrow_mgr: EscrowManager = adversarial_env["escrow_manager"]

        n = 12
        a = (np.eye(n, dtype=np.float32) * 0.5).astype(np.float32)
        v0 = np.ones(n, dtype=np.float32)

        worker.servicer.inject_vector_corruption(step_index=2, mode=fraud_mode)

        with pytest.raises(FraudDetectedError) as exc_info:
            coord.execute_iterative_power(
                a,
                v0,
                iterations=5,
                forced_spot_check_indices={2},
            )

        proof = exc_info.value.fraud_proof
        assert proof.max_observed_discrepancy > proof.epsilon_tolerance

        # Escrow must be cancelled
        escrow = list(escrow_mgr._escrows.values())[0]
        assert escrow.status == EscrowStatus.CANCELLED
        assert worker.servicer.release_calls >= 1

    def test_forged_worker_secp256k1_signature_rejection(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """
        Injects a forged Secp256k1 signature created by an untrusted key on step t.
        Asserts:
        1. WorkerSignatureError is raised.
        2. Escrow is aborted and refunded (CANCELLED).
        3. ReleaseLease RPC is executed in finally: block.
        """
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        worker: AdversarialMockWorkerNode = adversarial_env["worker"]
        escrow_mgr: EscrowManager = adversarial_env["escrow_manager"]

        n = 16
        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        # Worker forges signature on step 3
        worker.servicer.inject_forged_signature(step_index=3)

        with pytest.raises(WorkerSignatureError) as exc_info:
            coord.execute_iterative_power(a, v0, iterations=10)

        err = exc_info.value
        assert "Secp256k1 signature validation failed" in str(err)
        assert err.worker_pubkey == worker.pubkey

        # Assert escrow aborted and refunded
        escrow = list(escrow_mgr._escrows.values())[0]
        assert escrow.status == EscrowStatus.CANCELLED
        assert escrow.status != EscrowStatus.SETTLED

        # Assert ReleaseLease executed in finally:
        assert worker.servicer.release_calls >= 1
        assert len(worker.servicer._pinned_sessions) == 0

    def test_corrupted_signature_hex_payload(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """Worker returns garbage signature payload; asserts WorkerSignatureError and refund."""
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        worker: AdversarialMockWorkerNode = adversarial_env["worker"]
        escrow_mgr: EscrowManager = adversarial_env["escrow_manager"]

        n = 16
        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        orig_step = worker.servicer.StepIterative

        def corrupt_sig_step(req: IterativeStepRequest, ctx=None):
            resp = orig_step(req, ctx)
            if req.step_index == 2:
                resp.worker_signature = "deadbeef" * 8
            return resp

        worker.servicer.StepIterative = corrupt_sig_step

        with pytest.raises(WorkerSignatureError):
            coord.execute_iterative_power(a, v0, iterations=5)

        escrow = list(escrow_mgr._escrows.values())[0]
        assert escrow.status == EscrowStatus.CANCELLED
        assert worker.servicer.release_calls >= 1


# ==============================================================================
# 2. LIFECYCLE EDGE CASES
# ==============================================================================

class TestLifecycleEdgeCases:
    """
    Stress-tests worker lease rejection, mid-loop lease expiration,
    and 100-step numerical equivalence against NumPy reference.
    """

    def test_worker_lease_rejection_handling(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """
        Simulates worker rejecting lease negotiation (is_accepted=False).
        Asserts:
        1. WorkerUnavailableError is raised with rejection reason.
        2. Escrow is aborted and refunded (CANCELLED).
        3. No lingering sessions allocated on worker.
        """
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        worker: AdversarialMockWorkerNode = adversarial_env["worker"]
        escrow_mgr: EscrowManager = adversarial_env["escrow_manager"]

        worker.servicer.reject_lease = True
        worker.servicer.reject_lease_message = "Insufficient VRAM for requested dimensions"

        n = 16
        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        with pytest.raises(WorkerUnavailableError) as exc_info:
            coord.execute_iterative_power(a, v0, iterations=10)

        assert "Insufficient VRAM" in str(exc_info.value)

        # Escrow must be refunded
        escrow = list(escrow_mgr._escrows.values())[0]
        assert escrow.status == EscrowStatus.CANCELLED
        assert escrow.status != EscrowStatus.SETTLED

        # No pinned sessions on worker
        assert len(worker.servicer._pinned_sessions) == 0

    def test_lease_expiration_mid_loop_triggers_release(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """
        Worker lease expires mid-loop (e.g. at step 4).
        Asserts:
        1. Exception is propagated.
        2. ReleaseLease is executed in finally: block.
        3. Worker deallocates pinned session.
        """
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        worker: AdversarialMockWorkerNode = adversarial_env["worker"]

        worker.servicer.expire_lease_after_step = 4

        n = 16
        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        with pytest.raises(RuntimeError) as exc_info:
            coord.execute_iterative_power(a, v0, iterations=10)

        assert "expired" in str(exc_info.value).lower()
        # ReleaseLease MUST be called in finally: block
        assert worker.servicer.release_calls >= 1
        assert len(worker.servicer._pinned_sessions) == 0

    def test_high_iteration_100_steps_numerical_equivalence_fp32(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """
        Executes 100 power iteration steps (v_100 = A^100 * v_0).
        Asserts numerical equivalence against pure NumPy calculation within rtol=1e-4, atol=1e-4.
        """
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        n = 16
        np.random.seed(301)

        # Construct symmetric positive definite matrix with spectral radius <= 1.0
        g = np.random.randn(n, n).astype(np.float32)
        sym = np.matmul(g, g.T)
        max_eig = float(np.max(np.linalg.eigvalsh(sym)))
        a = (sym / (max_eig * 1.05)).astype(np.float32)

        v0 = np.random.randn(n).astype(np.float32)

        result: IterativeResult = coord.execute_iterative_power(
            a,
            v0,
            iterations=100,
            dtype="fp32",
        )

        assert result.iterations_completed == 100

        # Local reference computation using pure NumPy loop
        v_ref = v0.copy()
        for _ in range(100):
            v_ref = np.matmul(a, v_ref)

        np.testing.assert_allclose(np.asarray(result), v_ref, rtol=1e-4, atol=1e-4)

    def test_high_iteration_100_steps_with_l2_normalization(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """
        Executes 100 power iterations with L2-normalization enabled.
        Asserts convergence to dominant eigenvector and exact match with normalized NumPy reference.
        """
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        n = 16
        np.random.seed(302)

        g = np.random.randn(n, n).astype(np.float32)
        a = (g / np.linalg.norm(g)).astype(np.float32)
        v0 = np.random.randn(n).astype(np.float32)

        result = coord.execute_iterative_power(
            a,
            v0,
            iterations=100,
            normalize=True,
        )

        # Assert unit length
        norm_res = float(np.linalg.norm(result))
        assert math.isclose(norm_res, 1.0, rel_tol=1e-5)

        # Compare with local normalized reference
        v_ref = v0.copy()
        for _ in range(100):
            v_ref = np.matmul(a, v_ref)
            v_ref = v_ref / np.linalg.norm(v_ref)

        # Eigenvector direction check: |<v_res, v_ref>| should be ~ 1.0
        dot_product = abs(float(np.dot(np.asarray(result), v_ref)))
        assert math.isclose(dot_product, 1.0, rel_tol=1e-4)

    @pytest.mark.parametrize("dim", [17, 31])
    def test_high_iteration_prime_dimensions(
        self,
        adversarial_env: Dict[str, Any],
        dim: int,
    ) -> None:
        """Verifies 100 iterations with prime matrix dimensions N=17, 31."""
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        np.random.seed(dim)

        a = (np.eye(dim, dtype=np.float32) * 0.9).astype(np.float32)
        v0 = np.ones(dim, dtype=np.float32)

        result = coord.execute_iterative_power(a, v0, iterations=100, dtype="fp32")
        assert len(result) == dim

        expected = v0 * (0.9 ** 100)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-4, atol=1e-4)


# ==============================================================================
# 3. SECURITY BOUNDARY & PROTOCOL INVARIANT PROBES
# ==============================================================================

class TestSecurityBoundaryProbes:
    """Probes coordinator security invariants, upload bounds, and edge defenses."""

    def test_single_upload_invariance_across_100_iterations(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """
        Verifies matrix A is uploaded strictly once across 100 iterations.
        LeaseMatrix called 1 time, StepIterative called 100 times.
        """
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        worker: AdversarialMockWorkerNode = adversarial_env["worker"]

        n = 16
        iterations = 100
        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        coord.execute_iterative_power(a, v0, iterations=iterations, dtype="fp32")

        assert worker.servicer.lease_calls == 1
        assert worker.servicer.step_calls == iterations
        assert worker.servicer.release_calls == 1

        expected_matrix_bytes = n * n * 4
        expected_vector_bytes = iterations * n * 4
        assert worker.servicer.uploaded_bytes_total == expected_matrix_bytes + expected_vector_bytes

    def test_spot_check_fractional_sampling_pass_rate(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """
        When worker is honest, all spot-checks performed pass successfully.
        """
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        n = 16
        a = (np.eye(n, dtype=np.float32) * 0.5).astype(np.float32)
        v0 = np.ones(n, dtype=np.float32)

        result = coord.execute_iterative_power(
            a,
            v0,
            iterations=50,
            spot_check_probability=0.2,
        )

        assert result.spot_checks_performed > 0
        assert result.spot_checks_passed == result.spot_checks_performed

    def test_non_square_matrix_rejected_fail_closed(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """Coordinator rejects non-square matrix before any network or escrow activity."""
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        worker: AdversarialMockWorkerNode = adversarial_env["worker"]
        escrow_mgr: EscrowManager = adversarial_env["escrow_manager"]

        a_bad = np.ones((16, 12), dtype=np.float32)
        v0 = np.ones(12, dtype=np.float32)

        with pytest.raises(MatrixDimensionError):
            coord.execute_iterative_power(a_bad, v0, iterations=10)

        # Ensure no network calls or escrows created
        assert worker.servicer.lease_calls == 0
        assert len(escrow_mgr._escrows) == 0

    def test_audit_mid_loop_expiration_escrow_refund_gap(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """
        AUDIT PROBE (Finding 1):
        Investigates escrow state after worker lease expires mid-loop.
        Verifies that coordinator cleans up remote VRAM lease in finally: block,
        and strictly calls abort_and_refund on escrow_manager when StepIterative fails,
        transitioning consumer escrow to EscrowStatus.CANCELLED.
        """
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        worker: AdversarialMockWorkerNode = adversarial_env["worker"]
        escrow_mgr: EscrowManager = adversarial_env["escrow_manager"]

        worker.servicer.expire_lease_after_step = 3
        a = np.eye(16, dtype=np.float32)
        v0 = np.ones(16, dtype=np.float32)

        with pytest.raises(RuntimeError):
            coord.execute_iterative_power(a, v0, iterations=10)

        # ReleaseLease is executed in finally: block
        assert worker.servicer.release_calls >= 1

        # Check escrow state:
        escrow = list(escrow_mgr._escrows.values())[0]
        # abort_and_refund is called on mid-loop error, transitioning escrow to CANCELLED.
        assert escrow.status == EscrowStatus.CANCELLED
        assert escrow.cancellation_reason == "iteration_loop_failed"

    def test_audit_omitted_signature_bypass_probe(
        self,
        adversarial_env: Dict[str, Any],
    ) -> None:
        """
        AUDIT PROBE (Finding 2):
        Investigates coordinator behavior when worker returns empty signature ("").
        Enforces fail-closed rejection with WorkerSignatureError and escrow cancellation.
        """
        coord: DistributedMatmulCoordinator = adversarial_env["coordinator"]
        worker: AdversarialMockWorkerNode = adversarial_env["worker"]
        escrow_mgr: EscrowManager = adversarial_env["escrow_manager"]

        worker.servicer.inject_omitted_signature(step_index=2)
        a = (np.eye(16, dtype=np.float32) * 0.5).astype(np.float32)
        v0 = np.ones(16, dtype=np.float32)

        # Worker omits required signature on step 2:
        # Coordinator must reject fail-closed with WorkerSignatureError and refund escrow
        with pytest.raises(WorkerSignatureError) as exc_info:
            coord.execute_iterative_power(a, v0, iterations=5, spot_check_probability=0.0)

        assert "omitted" in str(exc_info.value).lower()
        escrow = list(escrow_mgr._escrows.values())[0]
        assert escrow.status == EscrowStatus.CANCELLED

