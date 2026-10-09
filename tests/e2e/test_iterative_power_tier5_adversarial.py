"""
Milestone 5 Phase 2: Tier 5 Adversarial Coverage Hardening Suite
Subsystem: matrix-compute-sdk & matrix-compute-node Stateful Iterative Interactions.

Adversarial Coverage Matrix:
Category 1: TestCategory1IllConditionedAndCancellationBoundaries (5 tests)
  - test_ill_conditioned_matrix_verification: kappa(A) >= 1e7 numerical stability
  - test_alternating_sign_cancellation_vector: Catastrophic cancellation vector without false alarm
  - test_cancellation_vector_with_adversarial_bias: Fraud detection on cancelled vectors
  - test_sharp_higham_sub_bound_acceptance: Acceptance at 0.95 * B_eff
  - test_sharp_higham_super_bound_rejection: Strict rejection at 1.05 * B_eff

Category 2: TestCategory2BF16CeilingAndSubtleFraud (4 tests)
  - test_bf16_gamma_k_saturation_above_half: BF16 K=128 gamma_K >= 0.50 saturation
  - test_bf16_sub_ceiling_relative_discrepancy_acceptance: 0.45 relative discrepancy accepted
  - test_bf16_super_ceiling_relative_discrepancy_rejection: 0.55 relative discrepancy rejected (0.50 cap)
  - test_bf16_zero_work_fraud_defense: Zero-work fraud rejection in BF16

Category 3: TestCategory3LifecycleStepAsymmetry (5 tests)
  - test_step_0_worker_drop_fail_closed: Step 0 drop fails closed and refunds escrow
  - test_step_t_minus_1_worker_drop_fail_closed: Final step T-1 drop fails closed and refunds escrow
  - test_targeted_t_minus_1_fraud_caught_via_forced_spot_check: Step T-1 cheat caught via forced check
  - test_step_0_fraud_caught_via_forced_spot_check: Step 0 cheat caught via forced check
  - test_rapid_client_abort_escrow_refund_and_cleanup: Client abort triggers ReleaseLease and refund

Category 4: TestCategory4LeaseRejectionAndDoubleFaultRecovery (4 tests)
  - test_lease_matrix_rejection_triggers_escrow_refund: Worker rejection raises WorkerUnavailableError
  - test_lease_matrix_transport_error_triggers_escrow_refund: Transport error raises WorkerUnavailableError
  - test_double_fault_release_lease_rpc_failure_tolerance: ReleaseLease RPC failure during cleanup handled
  - test_matchmaker_empty_worker_pool_clean_rejection: Empty worker pool raises NoWorkersFoundError

Category 5: TestCategory5PolymorphicVectorsAndOmittedSignatures (9 tests)
  - test_omitted_empty_string_worker_signature: Empty string signature raises WorkerSignatureError
  - test_whitespace_worker_signature: Whitespace signature raises WorkerSignatureError
  - test_spot_check_probability_near_zero: p = 1e-12 sampling extreme runs cleanly
  - test_spot_check_probability_near_one: p = 1 - 1e-12 sampling extreme runs cleanly
  - test_out_of_bounds_forced_spot_check_indices: Out-of-bounds forced indices ignored without error
  - test_polymorphic_nested_list_vector_input: Nested list input normalized to ndarray
  - test_non_vector_2d_shape_rejection: (N, 2) shape vector raises MatrixDimensionError
  - test_vector_length_mismatch_rejection: Length mismatch raises MatrixDimensionError
  - test_iterative_result_duck_typing: IterativeResult array protocols and properties verified

Suite 1: TestTier5NodeSessionManagerRacesAndMemoryBounds (5 tests)
  - test_reaper_eviction_race_during_step_iterative: Concurrent reaper vs step execution
  - test_exact_memory_quota_boundary_and_overflow_defense: Exact byte quota boundary
  - test_bf16_wire_vs_memory_footprint_quota_accounting: BF16 16-bit to 32-bit quota accounting
  - test_concurrent_double_release_idempotency_saturation: 20-thread release saturation
  - test_zeroization_view_permanence_and_non_revocability: In-place buffer zeroization view permanence

Suite 2: TestTier5NodeConfidentialEnclaveCryptoAdversarial (5 tests)
  - test_malformed_iv_framing_rejection: 11-byte and 13-byte IVs rejected
  - test_truncated_aead_tag_framing_rejection: 15-byte and 0-byte tags rejected
  - test_bit_flipped_ciphertext_decryption_failure: Bit-flipped ciphertext rejection
  - test_low_order_x25519_client_pubkey_rejection: Low-order point b"\\x00"*32 rejected
  - test_confidential_mode_plaintext_matrix_policy_enforcement: Confidential mode framing enforced

Suite 3: TestTier5CrossSubsystemSignaturesAndSettlementAdversarial (5 tests)
  - test_omitted_worker_signature_rejection: Worker omitted signature on step 1 rejected
  - test_rogue_signer_forged_signature_rejection: Rogue key signature rejected
  - test_perturbed_result_vector_signature_mismatch: Tampered vector with old sig rejected
  - test_cross_step_and_cross_lease_signature_replay_attack: Replay across steps/leases rejected
  - test_mid_loop_fault_settlement_cancellation_and_resource_reclamation: Fail-closed settlement

Total: 42 Tier 5 Adversarial Test Cases (27 Category 1-5 + 15 Suites 1-3).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import hmac
import json
import logging
from pathlib import Path
import secrets
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519
import grpc
import numpy as np
import pytest

# Ensure all workspace modules are resolvable
_TESTS_DIR = Path(__file__).resolve().parent
_SDK_DIR = _TESTS_DIR.parents[1]
_ROOT_DIR = _SDK_DIR.parent
_CONTRACTS_DIR = _ROOT_DIR / "matrix-contracts"
_NODE_DIR = _ROOT_DIR / "matrix-compute-node"
_SDK_SRC_DIR = _SDK_DIR / "src"

for _p in [
    str(_CONTRACTS_DIR),
    str(_SDK_DIR),
    str(_SDK_SRC_DIR),
    str(_NODE_DIR),
    str(_NODE_DIR / "src"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Interface contracts & protobuf stubs
try:
    from matrix_contracts.v1 import (
        IterativeStepRequest,
        IterativeStepResponse,
        LeaseMatrixRequest,
        LeaseMatrixResponse,
        MatrixComputeServiceServicer,
        ReleaseLeaseRequest,
        ReleaseLeaseResponse,
        pb2,
        pb2_grpc,
    )
except ImportError:
    from matrix_contracts.proto import (
        IterativeStepRequest,
        IterativeStepResponse,
        LeaseMatrixRequest,
        LeaseMatrixResponse,
        MatrixComputeServiceServicer,
        ReleaseLeaseRequest,
        ReleaseLeaseResponse,
        compute_service_pb2 as pb2,
        compute_service_pb2_grpc as pb2_grpc,
    )

from matrix_contracts.schemas import FreivaldsFraudProof, MatchedWorker

# SDK Components
from matrix_compute.config import ClientConfig, WorkloadType
from matrix_compute.crypto.signer import ClientSigner, verify_signature
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.exceptions import (
    FraudDetectedError,
    MatrixDimensionError,
    NoWorkersFoundError,
    WorkerSignatureError,
    WorkerUnavailableError,
)
from matrix_compute.orchestrator.coordinator import (
    DistributedMatmulCoordinator,
    IterativeResult,
)
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import MockWorkerNode
from matrix_compute.tiling.precision import truncate_to_bf16
from matrix_compute.verification.freivalds import (
    FreivaldsVerifier,
    matrix_infinity_norm,
    vector_infinity_norm,
    verify_iterative_step,
)

# Compute Node Components
from src.auth.l402_gate import L402PreAllocationGate
from src.crypto.signer import WorkerSigner
from src.engine.compute_core import ComputeEngine
from src.engine.enclave_compute import (
    EnclaveComputeCore,
    EnclaveDecryptionError,
    EnclaveFramingError,
    EnclaveKeyExchangeError,
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    derive_session_keys,
)
from src.engine.session_manager import (
    InvalidMatrixDimensionsError,
    MemoryBoundError,
    PinnedTensorSession,
    SessionCapacityError,
    SessionExpiredError,
    SessionManager,
    SessionNotFoundError,
)
from src.p2p.servicer import MatrixComputeServicer

logger = logging.getLogger(__name__)


# ==============================================================================
# TEST FIXTURES & DOUBLES
# ==============================================================================

class MockServicerGrpcContext:
    """Mock gRPC ServicerContext capturing abort calls, status codes, and metadata."""

    def __init__(self, metadata: Optional[List[Tuple[str, str]]] = None) -> None:
        self._metadata = metadata or []
        self.aborted = False
        self.code: Optional[grpc.StatusCode] = None
        self.details: Optional[str] = None

    def invocation_metadata(self) -> List[Tuple[str, str]]:
        return self._metadata

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.aborted = True
        self.code = code
        self.details = details
        raise RuntimeError(f"ABORT: {code}: {details}")


@pytest.fixture
def compute_engine() -> ComputeEngine:
    return ComputeEngine(force_cpu=True)


@pytest.fixture
def worker_signer() -> WorkerSigner:
    return WorkerSigner()


@pytest.fixture
def rogue_signer() -> WorkerSigner:
    return WorkerSigner()


@pytest.fixture
def client_signer() -> ClientSigner:
    return ClientSigner()


@pytest.fixture
def mock_settlement() -> MockSettlementService:
    return MockSettlementService(verification_key="tier5-secret-key-99999")


@pytest.fixture
def mock_relay() -> MockNostrRelay:
    return MockNostrRelay()


def _make_test_coordinator(
    mock_node: MockWorkerNode,
    client_signer: ClientSigner,
    mock_settlement: MockSettlementService,
    mock_relay: MockNostrRelay,
    config: Optional[ClientConfig] = None,
) -> Tuple[DistributedMatmulCoordinator, EscrowManager]:
    cfg = config or ClientConfig(
        lease_ttl_seconds=60,
        spot_check_probability=0.0,
        max_price_per_gflop=5.0,
    )
    escrow_mgr = EscrowManager(backend=mock_settlement, signer=client_signer)
    matchmaker = Matchmaker(config=cfg)
    scheduler = ParallelTaskScheduler(max_concurrency=2)
    dispatcher = GrpcTaskDispatcher(timeout_seconds=5.0)

    coord = DistributedMatmulCoordinator(
        config=cfg,
        signer=client_signer,
        relay_client=mock_relay,
        matchmaker=matchmaker,
        escrow_manager=escrow_mgr,
        scheduler=scheduler,
        dispatcher=dispatcher,
        mock_workers=[mock_node],
    )
    return coord, escrow_mgr


# ==============================================================================
# CATEGORY 1: ILL-CONDITIONED MATRICES & CANCELLATION BOUNDARIES (5 Tests)
# ==============================================================================

class TestCategory1IllConditionedAndCancellationBoundaries:
    """
    Mathematical boundary stress tests for verify_iterative_step:
    Ill-conditioned matrices, cancellation vectors, and sharp Higham bound transitions.
    """

    def test_ill_conditioned_matrix_verification(self) -> None:
        """
        Category 1: Ill-conditioned matrix with kappa(A) >= 1e7.
        Verifies verify_iterative_step does not generate false fraud positives.
        """
        # Singular values [1.0, 0.1, 0.01, 1e-7] -> kappa(A) = 1e7
        s = np.array([1.0, 0.1, 0.01, 1e-7], dtype=np.float32)
        A = np.diag(s)

        # Dominant projection
        vt_dom = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float32)
        vnext_dom = np.matmul(A, vt_dom)
        res_dom = verify_iterative_step(A=A, v_t=vt_dom, v_next=vnext_dom, dtype="fp32")
        assert res_dom.is_valid is True
        assert res_dom.discrepancy <= res_dom.effective_bound

        # Subdominant projection along smallest singular vector
        vt_sub = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        vnext_sub = np.matmul(A, vt_sub)
        res_sub = verify_iterative_step(A=A, v_t=vt_sub, v_next=vnext_sub, dtype="fp32")
        assert res_sub.is_valid is True

    def test_alternating_sign_cancellation_vector(self) -> None:
        """
        Category 1: Catastrophic cancellation vector where A * v_t ~= 0 while ||A|| * ||v_t|| >> 0.
        Verifies genuine cancellation passes without underflow crash or false fraud alarm.
        """
        A = np.array([
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [2.0, -2.0, 2.0, -2.0],
            [-2.0, 2.0, -2.0, 2.0],
        ], dtype=np.float32)
        vt = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        vnext = np.matmul(A, vt)
        assert np.allclose(vnext, 0.0)

        res = verify_iterative_step(A=A, v_t=vt, v_next=vnext, dtype="fp32")
        assert res.is_valid is True

    def test_cancellation_vector_with_adversarial_bias(self) -> None:
        """
        Category 1: Adversarial worker injects unauthorized bias into a cancelled result.
        Verifies that verify_iterative_step detects the fraud.
        """
        A = np.array([
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [2.0, -2.0, 2.0, -2.0],
            [-2.0, 2.0, -2.0, 2.0],
        ], dtype=np.float32)
        vt = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        v_tampered = np.matmul(A, vt)
        v_tampered[0] += 0.5  # Unauthorized bias injected into cancelled zero

        res = verify_iterative_step(A=A, v_t=vt, v_next=v_tampered, dtype="fp32")
        assert res.is_valid is False
        assert res.failure_proof is not None

    def test_sharp_higham_sub_bound_acceptance(self) -> None:
        """
        Category 1: Near-boundary perturbation at 0.95 * B_eff (sub-bound) is accepted
        within the forward rounding uncertainty envelope.
        """
        N = 8
        A = np.eye(N, dtype=np.float32) * 2.0
        vt = np.ones(N, dtype=np.float32)
        v_true = np.matmul(A, vt)

        baseline = verify_iterative_step(A=A, v_t=vt, v_next=v_true, dtype="fp32")
        b_eff = baseline.effective_bound

        # Inject perturbation strictly within bound
        v_pert = v_true.copy()
        v_pert[0] += float(0.95 * b_eff)

        res = verify_iterative_step(A=A, v_t=vt, v_next=v_pert, dtype="fp32")
        assert res.is_valid is True

    def test_sharp_higham_super_bound_rejection(self) -> None:
        """
        Category 1: Near-boundary perturbation at 1.05 * B_eff (super-bound) is strictly rejected.
        """
        N = 8
        A = np.eye(N, dtype=np.float32) * 2.0
        vt = np.ones(N, dtype=np.float32)
        v_true = np.matmul(A, vt)

        baseline = verify_iterative_step(A=A, v_t=vt, v_next=v_true, dtype="fp32")
        b_eff = baseline.effective_bound

        # Inject perturbation strictly breaching bound
        v_pert = v_true.copy()
        v_pert[0] += float(1.05 * b_eff)

        res = verify_iterative_step(A=A, v_t=vt, v_next=v_pert, dtype="fp32")
        assert res.is_valid is False
        assert res.discrepancy > res.effective_bound


# ==============================================================================
# CATEGORY 2: BF16 CEILING & SUBTLE FRAUD DEFENSE (4 Tests)
# ==============================================================================

class TestCategory2BF16CeilingAndSubtleFraud:
    """
    Verifies BF16 gamma_K bounds, relative discrepancy 0.50 ceiling enforcement,
    and zero-work fraud defense.
    """

    def test_bf16_gamma_k_saturation_above_half(self) -> None:
        """
        Category 2: In BF16 precision with K=128, gamma_K >= 0.50.
        Verifies gamma_K calculation and saturation.
        """
        K = 128
        A = np.eye(K, dtype=np.float32)
        vt = np.ones(K, dtype=np.float32)
        vnext = np.matmul(A, vt)

        res = verify_iterative_step(A=A, v_t=vt, v_next=vnext, dtype="bf16")
        assert res.is_valid is True
        assert res.effective_bound >= 0.5

    def test_bf16_sub_ceiling_relative_discrepancy_acceptance(self) -> None:
        """
        Category 2: In BF16 with K=128, a relative discrepancy of 0.45 is below the 0.50 ceiling
        and is accepted when dynamic bound allows.
        """
        K = 128
        A = np.eye(K, dtype=np.float32)
        vt = np.ones(K, dtype=np.float32)
        v_true = np.matmul(A, vt)

        v_pert = v_true.copy()
        v_pert[0] -= 0.45  # Downward perturbation preserves max(||Av||, ||v_next||) = 1.0 -> rel_diff = 0.45 / 1.0 = 0.45 <= 0.50
        res = verify_iterative_step(A=A, v_t=vt, v_next=v_pert, dtype="bf16", epsilon=0.46)
        assert res.is_valid is True

    def test_bf16_super_ceiling_relative_discrepancy_rejection(self) -> None:
        """
        Category 2: In BF16 with K=128, a relative discrepancy of 0.55 exceeds the 0.50 ceiling
        and is strictly flagged as fraud.
        """
        K = 128
        A = np.eye(K, dtype=np.float32)
        vt = np.ones(K, dtype=np.float32)
        v_true = np.matmul(A, vt)

        v_pert = v_true.copy()
        v_pert[0] -= 0.55  # Downward perturbation preserves max(||Av||, ||v_next||) = 1.0 -> rel_diff = 0.55 / 1.0 = 0.55 > 0.50
        res = verify_iterative_step(A=A, v_t=vt, v_next=v_pert, dtype="bf16", epsilon=1.0)
        assert res.is_valid is False
        assert "exceeded tolerance" in res.failure_proof.get("reason", "").lower()

    def test_bf16_zero_work_fraud_defense(self) -> None:
        """
        Category 2: Malicious worker returns an all-zero vector for BF16 workload.
        Verifies that zero-work fraud is detected.
        """
        K = 64
        A = np.eye(K, dtype=np.float32) * 5.0
        vt = np.ones(K, dtype=np.float32)
        v_zero = np.zeros(K, dtype=np.float32)

        res = verify_iterative_step(A=A, v_t=vt, v_next=v_zero, dtype="bf16")
        assert res.is_valid is False
        assert res.failure_proof is not None


# ==============================================================================
# CATEGORY 3: LIFECYCLE STEP ASYMMETRY & MID-LOOP RESILIENCE (5 Tests)
# ==============================================================================

class TestCategory3LifecycleStepAsymmetry:
    """
    Lifecycle step asymmetry: Step 0 drop vs Step T-1 drop, targeted cheating on T-1,
    and client-side abort cleanup.
    """

    def test_step_0_worker_drop_fail_closed(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 3: Worker drops immediately on step 0.
        Verifies coordinator fails closed, refunds escrow, and releases lease.
        """
        class Step0DropServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer
                self.released = False

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                raise RuntimeError("Worker node crashed immediately on step 0")

            def ReleaseLease(self, request, context=None):
                self.released = True
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = Step0DropServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_step0_drop", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, escrow_mgr = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        with pytest.raises(RuntimeError, match="Worker node crashed immediately on step 0"):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=5,
                candidate_workers=[node],
            )

        assert servicer.released is True
        for escrow in escrow_mgr._escrows.values():
            assert escrow.status == EscrowStatus.CANCELLED

    def test_step_t_minus_1_worker_drop_fail_closed(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 3: Worker behaves honestly for T-1 steps, then drops on the final step T-1.
        Verifies coordinator does not return truncated output, releases lease, and refunds escrow.
        """
        class StepTMinus1DropServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer
                self.released = False
                self.completed_steps = 0

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                if request.step_index >= 4:
                    raise RuntimeError("Worker node crashed on final step T-1")
                self.completed_steps += 1
                res = np.ones((request.vector_length,), dtype=np.float32).tobytes()
                r_hash = hashlib.sha256(res).hexdigest()
                payload = f"{request.lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload)
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                self.released = True
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = StepTMinus1DropServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_step_tm1_drop", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, escrow_mgr = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        with pytest.raises(RuntimeError, match="Worker node crashed on final step T-1"):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=5,
                candidate_workers=[node],
            )

        assert servicer.completed_steps == 4
        assert servicer.released is True
        for escrow in escrow_mgr._escrows.values():
            assert escrow.status == EscrowStatus.CANCELLED

    def test_targeted_t_minus_1_fraud_caught_via_forced_spot_check(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 3: Worker cheats strictly on the final step T-1.
        Forced spot checking on {T-1} catches the fraud and generates a fraud proof.
        """
        class TargetedFraudServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer
                self.released = False

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                dt = np.float32
                vec = np.frombuffer(request.vector_bytes, dtype=dt)
                res_vec = vec.copy()
                if request.step_index == 4:
                    res_vec[0] += 5.0  # Gross perturbation on step T-1

                res_bytes = res_vec.tobytes()
                r_hash = hashlib.sha256(res_bytes).hexdigest()
                payload = f"{request.lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload)
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res_bytes,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                self.released = True
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = TargetedFraudServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_t_minus_1_fraud", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, escrow_mgr = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        with pytest.raises(FraudDetectedError) as exc_info:
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=5,
                candidate_workers=[node],
                spot_check_probability=0.0,
                forced_spot_check_indices={4},
            )

        assert "step_4" in str(exc_info.value.fraud_proof.task_id)
        assert servicer.released is True

    def test_step_0_fraud_caught_via_forced_spot_check(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 3: Worker cheats on initial step 0.
        Forced spot checking on {0} catches the fraud on the very first iteration.
        """
        class Step0FraudServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer
                self.released = False

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                res_vec = np.ones(request.vector_length, dtype=np.float32)
                if request.step_index == 0:
                    res_vec[0] += 5.0  # Fraud on step 0
                res_bytes = res_vec.tobytes()
                r_hash = hashlib.sha256(res_bytes).hexdigest()
                payload = f"{request.lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload)
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res_bytes,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                self.released = True
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = Step0FraudServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_step_0_fraud", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, escrow_mgr = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        with pytest.raises(FraudDetectedError) as exc_info:
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=5,
                candidate_workers=[node],
                spot_check_probability=0.0,
                forced_spot_check_indices={0},
            )

        assert "step_0" in str(exc_info.value.fraud_proof.task_id)
        assert servicer.released is True

    def test_rapid_client_abort_escrow_refund_and_cleanup(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 3: Simulates client-side abortion or interrupt mid-loop.
        Verifies ReleaseLease executes and escrow is refunded.
        """
        class HonestServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer
                self.released = False

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                res = np.ones((request.vector_length,), dtype=np.float32).tobytes()
                r_hash = hashlib.sha256(res).hexdigest()
                payload = f"{request.lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload)
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                self.released = True
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = HonestServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_client_abort", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, escrow_mgr = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        # Monkeypatch step signature verification to simulate client cancellation
        def abort_client(*args, **kwargs):
            raise KeyboardInterrupt("Simulated client abort / cancellation")

        coord._verify_step_signature = abort_client

        with pytest.raises(KeyboardInterrupt):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=5,
                candidate_workers=[node],
            )

        assert servicer.released is True
        for escrow in escrow_mgr._escrows.values():
            assert escrow.status == EscrowStatus.CANCELLED


# ==============================================================================
# CATEGORY 4: LEASE REJECTION & DOUBLE-FAULT TOLERANCE (4 Tests)
# ==============================================================================

class TestCategory4LeaseRejectionAndDoubleFaultRecovery:
    """
    Verifies error recovery on LeaseMatrix rejection, transport failures,
    double-fault during ReleaseLease, and empty matchmaking pools.
    """

    def test_lease_matrix_rejection_triggers_escrow_refund(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 4: Worker explicitly returns is_accepted=False on LeaseMatrix.
        Verifies coordinator raises WorkerUnavailableError and refunds escrow.
        """
        class RejectServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=False,
                    expires_at_unix_ms=0,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OUT_OF_VRAM",
                )

            def ReleaseLease(self, request, context=None):
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=0)

        servicer = RejectServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_reject", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, escrow_mgr = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        with pytest.raises(WorkerUnavailableError, match="Lease rejected"):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=5,
                candidate_workers=[node],
            )

        for escrow in escrow_mgr._escrows.values():
            assert escrow.status == EscrowStatus.CANCELLED

    def test_lease_matrix_transport_error_triggers_escrow_refund(
        self,
        client_signer: ClientSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 4: Dispatcher raises transport exception during LeaseMatrix RPC.
        Verifies coordinator catches error, refunds escrow, and raises WorkerUnavailableError.
        """
        node = MockWorkerNode(worker_id="w_remote_stub", pubkey="02" + "22" * 32)
        coord, escrow_mgr = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        class FailingDispatcher(GrpcTaskDispatcher):
            def lease_matrix(self, *args, **kwargs):
                raise ConnectionError("Transport connection refused")

        coord.dispatcher = FailingDispatcher()

        remote_worker = MatchedWorker(
            worker_id="remote_node_1",
            worker_pubkey="02" + "33" * 32,
            multiaddr="/ip4/127.0.0.1/tcp/50051",
            sats_per_gflop=1.0,
            supported_dtypes=["fp32"],
        )

        with pytest.raises(WorkerUnavailableError, match="Lease rejected"):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=3,
                candidate_workers=[remote_worker],
            )

        for escrow in escrow_mgr._escrows.values():
            assert escrow.status == EscrowStatus.CANCELLED

    def test_double_fault_release_lease_rpc_failure_tolerance(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 4: Worker fails during StepIterative, and ReleaseLease in finally: also fails.
        Verifies double-fault is logged without suppressing original exception, and escrow refund proceeds.
        """
        class DoubleFaultServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                raise RuntimeError("Primary step error")

            def ReleaseLease(self, request, context=None):
                raise RuntimeError("Double-fault: release RPC broken pipe")

        servicer = DoubleFaultServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_double_fault", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, escrow_mgr = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        with pytest.raises(RuntimeError, match="Primary step error"):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=3,
                candidate_workers=[node],
            )

        for escrow in escrow_mgr._escrows.values():
            assert escrow.status == EscrowStatus.CANCELLED

    def test_matchmaker_empty_worker_pool_clean_rejection(
        self,
        client_signer: ClientSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 4: Matchmaker returns 0 matching candidates.
        Verifies NoWorkersFoundError is raised without attempting escrow lock.
        """
        dummy_node = MockWorkerNode(worker_id="w_empty", pubkey="02" + "44" * 32)
        coord, escrow_mgr = _make_test_coordinator(dummy_node, client_signer, mock_settlement, mock_relay)

        class EmptyMatchmaker(Matchmaker):
            def query_and_match(self, *args, **kwargs):
                return []

        coord.matchmaker = EmptyMatchmaker(config=coord.config)
        coord.mock_workers = []

        with pytest.raises(NoWorkersFoundError):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=5,
                candidate_workers=[],
            )

        assert len(escrow_mgr._escrows) == 0


# ==============================================================================
# CATEGORY 5: POLYMORPHIC VECTORS & OMITTED SIGNATURES (9 Tests)
# ==============================================================================

class TestCategory5PolymorphicVectorsAndOmittedSignatures:
    """
    Verifies input polymorphic normalization, dimension validations, signature edge cases,
    spot check sampling extremes, and IterativeResult duck-typing.
    """

    def test_omitted_empty_string_worker_signature(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 5: Worker returns an empty string ("") as signature.
        Verifies coordinator raises WorkerSignatureError with explicit reason.
        """
        class EmptySigServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                res = np.ones((request.vector_length,), dtype=np.float32).tobytes()
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res,
                    vector_length=request.vector_length,
                    worker_signature="",
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = EmptySigServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_empty_sig", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, _ = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        with pytest.raises(WorkerSignatureError, match="omitted required Secp256k1 signature"):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=3,
                candidate_workers=[node],
            )

    def test_whitespace_worker_signature(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 5: Worker returns whitespace ("   ") as signature.
        Verifies coordinator rejects as WorkerSignatureError.
        """
        class WhitespaceSigServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                res = np.ones((request.vector_length,), dtype=np.float32).tobytes()
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res,
                    vector_length=request.vector_length,
                    worker_signature="   ",
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = WhitespaceSigServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_ws_sig", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, _ = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        with pytest.raises(WorkerSignatureError, match="omitted required Secp256k1 signature"):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones(4, dtype=np.float32),
                iterations=3,
                candidate_workers=[node],
            )

    def test_spot_check_probability_near_zero(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 5: spot_check_probability = 1e-12 boundary condition executes cleanly.
        """
        class HonestServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                dt = np.float32
                vec = np.frombuffer(request.vector_bytes, dtype=dt)
                res_bytes = vec.tobytes()
                r_hash = hashlib.sha256(res_bytes).hexdigest()
                payload = f"{request.lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload)
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res_bytes,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = HonestServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_honest_p0", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, _ = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        res = coord.execute_iterative_power(
            A=np.eye(4, dtype=np.float32),
            v0=np.ones(4, dtype=np.float32),
            iterations=5,
            candidate_workers=[node],
            spot_check_probability=1e-12,
        )
        assert res.iterations_completed == 5

    def test_spot_check_probability_near_one(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 5: spot_check_probability = 1.0 - 1e-12 boundary condition executes cleanly.
        """
        class HonestServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                dt = np.float32
                vec = np.frombuffer(request.vector_bytes, dtype=dt)
                res_bytes = vec.tobytes()
                r_hash = hashlib.sha256(res_bytes).hexdigest()
                payload = f"{request.lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload)
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res_bytes,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = HonestServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_honest_p1", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, _ = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        res = coord.execute_iterative_power(
            A=np.eye(4, dtype=np.float32),
            v0=np.ones(4, dtype=np.float32),
            iterations=5,
            candidate_workers=[node],
            spot_check_probability=1.0 - 1e-12,
        )
        assert res.iterations_completed == 5
        assert res.spot_checks_performed > 0

    def test_out_of_bounds_forced_spot_check_indices(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 5: Out-of-bounds indices in forced_spot_check_indices ({-5, 999}) are ignored without error.
        """
        class HonestServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                dt = np.float32
                vec = np.frombuffer(request.vector_bytes, dtype=dt)
                res_bytes = vec.tobytes()
                r_hash = hashlib.sha256(res_bytes).hexdigest()
                payload = f"{request.lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload)
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res_bytes,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = HonestServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_oob_check", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, _ = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        res = coord.execute_iterative_power(
            A=np.eye(4, dtype=np.float32),
            v0=np.ones(4, dtype=np.float32),
            iterations=3,
            candidate_workers=[node],
            forced_spot_check_indices={-5, 999, 1000},
        )
        assert res.iterations_completed == 3

    def test_polymorphic_nested_list_vector_input(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 5: Client provides initial vector as nested list ([[1.0], [2.0], [3.0], [4.0]]).
        Verifies coordinator normalizes input and runs power iteration cleanly.
        """
        class HonestServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                dt = np.float32
                vec = np.frombuffer(request.vector_bytes, dtype=dt)
                res_bytes = vec.tobytes()
                r_hash = hashlib.sha256(res_bytes).hexdigest()
                payload = f"{request.lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload)
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res_bytes,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = HonestServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_polymorphic", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, _ = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        v_list = [[1.0], [2.0], [3.0], [4.0]]
        res = coord.execute_iterative_power(
            A=np.eye(4, dtype=np.float32),
            v0=v_list,
            iterations=3,
            candidate_workers=[node],
        )
        assert res.iterations_completed == 3

    def test_non_vector_2d_shape_rejection(
        self,
        client_signer: ClientSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 5: Client provides non-vector 2D matrix shape (4, 2) as initial vector.
        Verifies MatrixDimensionError is raised.
        """
        node = MockWorkerNode(worker_id="w_dummy", pubkey="02" + "55" * 32)
        coord, _ = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        with pytest.raises(MatrixDimensionError):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones((4, 2), dtype=np.float32),
                iterations=3,
                candidate_workers=[node],
            )

    def test_vector_length_mismatch_rejection(
        self,
        client_signer: ClientSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 5: Vector length does not match square matrix dimension (5 != 4).
        Verifies MatrixDimensionError is raised.
        """
        node = MockWorkerNode(worker_id="w_dummy", pubkey="02" + "66" * 32)
        coord, _ = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        with pytest.raises(MatrixDimensionError):
            coord.execute_iterative_power(
                A=np.eye(4, dtype=np.float32),
                v0=np.ones((5,), dtype=np.float32),
                iterations=3,
                candidate_workers=[node],
            )

    def test_iterative_result_duck_typing(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        Category 5: IterativeResult exhibits seamless array duck-typing, indexing, slicing,
        arithmetic operations, and property getters.
        """
        class HonestServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                dt = np.float32
                vec = np.frombuffer(request.vector_bytes, dtype=dt)
                res_bytes = vec.tobytes()
                r_hash = hashlib.sha256(res_bytes).hexdigest()
                payload = f"{request.lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
                sig = self.signer.sign_message(payload)
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res_bytes,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        servicer = HonestServicer(worker_signer)
        node = MockWorkerNode(worker_id="w_duck_typing", pubkey=worker_signer.get_pubkey_hex())
        node.servicer = servicer

        coord, _ = _make_test_coordinator(node, client_signer, mock_settlement, mock_relay)

        v_init = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        res = coord.execute_iterative_power(
            A=np.eye(4, dtype=np.float32),
            v0=v_init,
            iterations=3,
            candidate_workers=[node],
        )

        assert isinstance(res, IterativeResult)
        assert len(res) == 4
        assert res.shape == (4,)
        assert res.ndim == 1
        assert res.dtype == np.float32
        assert res[0] == 1.0
        assert np.allclose(res[:2], [1.0, 2.0])
        assert np.allclose(np.asarray(res), res.final_vector)
        assert np.allclose(res * 2.0, res.final_vector * 2.0)
        assert res.iterations_completed == 3
        assert res.lease_id.startswith("lease_")
        assert res.total_compute_time_ms >= 0


# ==============================================================================
# SUITE 1: SESSION MANAGER RACES, QUOTA BOUNDS & ZEROIZATION PERMANENCE (5 Tests)
# ==============================================================================

class TestTier5NodeSessionManagerRacesAndMemoryBounds:
    """
    White-box stress tests for matrix-compute-node SessionManager:
    concurrency races, byte quota boundaries, and zeroization view permanence.
    """

    def test_reaper_eviction_race_during_step_iterative(self, compute_engine: ComputeEngine) -> None:
        """
        GAP-01: Simulates background reaper thread evicting an expiring session
        concurrently while worker threads attempt touch(), record_step(), and get_session().
        Verifies thread safety, absence of deadlocks, and clean exception handling.
        """
        mgr = SessionManager(
            max_sessions=10,
            max_memory_bytes=1024 * 1024,
            reaper_interval_seconds=0.01,
            compute_engine=compute_engine,
            auto_start_reaper=True,
        )
        try:
            n = 16
            mat = np.ones((n, n), dtype=np.float32)
            session = mgr.lease_matrix(
                lease_id="race_lease_01",
                matrix=mat,
                rows=n,
                cols=n,
                dtype="fp32",
                ttl_seconds=0.04,
            )
            assert session is not None
            assert mgr.active_session_count == 1

            step_recorded_count = 0
            evicted_detected_count = 0
            errors: List[Exception] = []

            def worker_step_loop():
                nonlocal step_recorded_count, evicted_detected_count
                for _ in range(50):
                    time.sleep(0.002)
                    sess = mgr.get_session("race_lease_01")
                    if sess is None:
                        evicted_detected_count += 1
                        continue
                    try:
                        sess.record_step()
                        step_recorded_count += 1
                    except SessionExpiredError:
                        evicted_detected_count += 1
                    except Exception as exc:
                        errors.append(exc)

            threads = [threading.Thread(target=worker_step_loop) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=3.0)

            assert not errors, f"Unexpected exceptions during reaper race: {errors}"
            assert evicted_detected_count > 0, "Reaper eviction was not observed by worker threads"

            time.sleep(0.05)
            assert mgr.get_session("race_lease_01") is None
            assert mgr.active_session_count == 0
            assert mgr.total_memory_bytes == 0
        finally:
            mgr.shutdown()

    def test_exact_memory_quota_boundary_and_overflow_defense(self, compute_engine: ComputeEngine) -> None:
        """
        GAP-02: Boundary condition allocating exactly at max_memory_bytes, followed
        by a 1-byte overflow attempt. Verifies MemoryBoundError, lack of partial
        registration, and non-leakage of tracked memory.
        """
        exact_quota_bytes = 4 * 4 * 4  # 4x4 float32 = 64 bytes
        mgr = SessionManager(
            max_sessions=5,
            max_memory_bytes=exact_quota_bytes,
            max_matrix_bytes=exact_quota_bytes,
            compute_engine=compute_engine,
            auto_start_reaper=False,
        )
        try:
            mat = np.eye(4, dtype=np.float32)
            assert mat.nbytes == exact_quota_bytes

            sess1 = mgr.lease_matrix(
                lease_id="exact_boundary_lease",
                matrix=mat,
                rows=4,
                cols=4,
                dtype="fp32",
                ttl_seconds=3600,
            )
            assert sess1 is not None
            assert mgr.total_memory_bytes == exact_quota_bytes
            assert mgr.active_session_count == 1

            # Allocation of even 1 element overflow must fail
            mat_overflow = np.ones((1, 1), dtype=np.float32)  # 4 bytes
            with pytest.raises(MemoryBoundError):
                mgr.lease_matrix(
                    lease_id="overflow_lease",
                    matrix=mat_overflow,
                    rows=1,
                    cols=1,
                    dtype="fp32",
                    ttl_seconds=3600,
                )

            assert mgr.total_memory_bytes == exact_quota_bytes
            assert mgr.active_session_count == 1
            assert mgr.get_session("overflow_lease") is None
        finally:
            mgr.shutdown()

    def test_bf16_wire_vs_memory_footprint_quota_accounting(self, compute_engine: ComputeEngine) -> None:
        """
        GAP-03: Multi-precision wire bytes (16-bit uint16) vs in-memory representation (float32).
        Verifies that SessionManager memory quota accounts for expanded 32-bit container bytes.
        """
        n = 8
        wire_elements = n * n
        wire_u16 = np.ones(wire_elements, dtype=np.uint16)
        wire_bytes = wire_u16.tobytes()  # 64 elements * 2 = 128 bytes
        expected_fp32_nbytes = wire_elements * 4  # 256 bytes

        mgr = SessionManager(
            max_sessions=5,
            max_memory_bytes=1024,
            compute_engine=compute_engine,
            auto_start_reaper=False,
        )
        try:
            sess = mgr.lease_matrix(
                lease_id="bf16_quota_lease",
                matrix=wire_bytes,
                rows=n,
                cols=n,
                dtype="bf16",
                ttl_seconds=3600,
            )
            assert sess.memory_bytes == expected_fp32_nbytes
            assert mgr.total_memory_bytes == expected_fp32_nbytes
        finally:
            mgr.shutdown()

    def test_concurrent_double_release_idempotency_saturation(self, compute_engine: ComputeEngine) -> None:
        """
        GAP-04: Concurrent multi-threaded release saturation: 20 threads simultaneously
        calling release_lease on the exact same lease_id.
        Verifies exactly one thread receives non-zero freed bytes and no double decrements.
        """
        mgr = SessionManager(
            max_sessions=5,
            max_memory_bytes=1024 * 1024,
            compute_engine=compute_engine,
            auto_start_reaper=False,
        )
        try:
            mat = np.ones((8, 8), dtype=np.float32)  # 256 bytes
            sess = mgr.lease_matrix(
                lease_id="concurrent_release_lease",
                matrix=mat,
                rows=8,
                cols=8,
                dtype="fp32",
                ttl_seconds=3600,
            )
            initial_mem = mgr.total_memory_bytes
            assert initial_mem == 256

            results: List[int] = []
            results_lock = threading.Lock()

            def do_release():
                freed = mgr.release_lease("concurrent_release_lease")
                with results_lock:
                    results.append(freed)

            threads = [threading.Thread(target=do_release) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=3.0)

            assert len(results) == 20
            non_zero_releases = [r for r in results if r > 0]
            assert len(non_zero_releases) == 1, f"Expected exactly 1 non-zero release, got {len(non_zero_releases)}"
            assert non_zero_releases[0] == 256
            assert sum(results) == 256
            assert mgr.total_memory_bytes == 0
            assert mgr.active_session_count == 0
        finally:
            mgr.shutdown()

    def test_zeroization_view_permanence_and_non_revocability(self, compute_engine: ComputeEngine) -> None:
        """
        GAP-05: In-place buffer zeroization view permanence: external views into the matrix
        must be permanently zeroized and the session rendered non-revocable.
        """
        mgr = SessionManager(
            max_sessions=5,
            max_memory_bytes=1024 * 1024,
            compute_engine=compute_engine,
            auto_start_reaper=False,
        )
        try:
            orig_data = np.ones((4, 4), dtype=np.float32) * 42.0
            sess = mgr.lease_matrix(
                lease_id="zeroize_view_lease",
                matrix=orig_data,
                rows=4,
                cols=4,
                dtype="fp32",
                ttl_seconds=3600,
            )
            retained_view = sess.matrix.view()
            assert retained_view[0, 0] == 42.0

            freed = mgr.release_lease("zeroize_view_lease")
            assert freed > 0
            assert np.all(retained_view == 0.0), "Array view was not zeroized in-place"
            assert sess.matrix is None

            with pytest.raises(SessionExpiredError):
                sess.touch()
            with pytest.raises(SessionExpiredError):
                sess.record_step()
        finally:
            mgr.shutdown()


# ==============================================================================
# SUITE 2: CONFIDENTIAL ENCLAVE CRYPTO BOUNDARY FAILURES (5 Tests)
# ==============================================================================

class TestTier5NodeConfidentialEnclaveCryptoAdversarial:
    """
    Cryptographic stress tests for AEAD framing, tampering, low-order ECDH keys,
    and confidential mode policy enforcement in matrix-compute-node.
    """

    def test_malformed_iv_framing_rejection(
        self,
        compute_engine: ComputeEngine,
        worker_signer: WorkerSigner,
    ) -> None:
        """
        GAP-06: Malformed IV length (11 bytes and 13 bytes) in StepIterative.
        Verifies rejection with StatusCode.INVALID_ARGUMENT.
        """
        servicer = MatrixComputeServicer(
            signer=worker_signer,
            engine=compute_engine,
        )
        ctx = MockServicerGrpcContext()

        lease_resp = servicer.LeaseMatrix(
            LeaseMatrixRequest(
                lease_id="iv_framing_lease",
                rows=4,
                cols=4,
                dtype="fp32",
                matrix_bytes=np.eye(4, dtype=np.float32).tobytes(),
                ttl_seconds=60,
                require_confidential_compute=False,
            ),
            context=ctx,
        )
        assert lease_resp.is_accepted is True

        # Test 11-byte IV
        req_11b = IterativeStepRequest(
            lease_id="iv_framing_lease",
            step_index=0,
            vector_bytes=np.ones(4, dtype=np.float32).tobytes(),
            vector_length=4,
            dtype="fp32",
            is_encrypted=True,
            iv=b"\x01" * 11,
            tag=b"\x02" * 16,
        )
        with pytest.raises(RuntimeError):
            servicer.StepIterative(req_11b, context=ctx)
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "iv must be 12 bytes" in ctx.details.lower()

        # Test 13-byte IV
        ctx2 = MockServicerGrpcContext()
        req_13b = IterativeStepRequest(
            lease_id="iv_framing_lease",
            step_index=0,
            vector_bytes=np.ones(4, dtype=np.float32).tobytes(),
            vector_length=4,
            dtype="fp32",
            is_encrypted=True,
            iv=b"\x01" * 13,
            tag=b"\x02" * 16,
        )
        with pytest.raises(RuntimeError):
            servicer.StepIterative(req_13b, context=ctx2)
        assert ctx2.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "iv must be 12 bytes" in ctx2.details.lower()

    def test_truncated_aead_tag_framing_rejection(
        self,
        compute_engine: ComputeEngine,
        worker_signer: WorkerSigner,
    ) -> None:
        """
        GAP-07: Truncated or empty AEAD tag framing (15 bytes and empty 0 bytes).
        Verifies rejection with StatusCode.INVALID_ARGUMENT.
        """
        servicer = MatrixComputeServicer(
            signer=worker_signer,
            engine=compute_engine,
        )
        ctx = MockServicerGrpcContext()

        servicer.LeaseMatrix(
            LeaseMatrixRequest(
                lease_id="tag_framing_lease",
                rows=4,
                cols=4,
                dtype="fp32",
                matrix_bytes=np.eye(4, dtype=np.float32).tobytes(),
                ttl_seconds=60,
                require_confidential_compute=False,
            ),
            context=ctx,
        )

        # 15-byte truncated tag
        req_15b = IterativeStepRequest(
            lease_id="tag_framing_lease",
            step_index=0,
            vector_bytes=np.ones(4, dtype=np.float32).tobytes(),
            vector_length=4,
            dtype="fp32",
            is_encrypted=True,
            iv=b"\x01" * 12,
            tag=b"\x02" * 15,
        )
        with pytest.raises(RuntimeError):
            servicer.StepIterative(req_15b, context=ctx)
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "tag must be 16 bytes" in ctx.details.lower()

        # Empty 0-byte tag
        ctx2 = MockServicerGrpcContext()
        req_0b = IterativeStepRequest(
            lease_id="tag_framing_lease",
            step_index=0,
            vector_bytes=np.ones(4, dtype=np.float32).tobytes(),
            vector_length=4,
            dtype="fp32",
            is_encrypted=True,
            iv=b"\x01" * 12,
            tag=b"",
        )
        with pytest.raises(RuntimeError):
            servicer.StepIterative(req_0b, context=ctx2)
        assert ctx2.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "tag must be 16 bytes" in ctx2.details.lower()

    def test_bit_flipped_ciphertext_decryption_failure(
        self,
        compute_engine: ComputeEngine,
        worker_signer: WorkerSigner,
    ) -> None:
        """
        GAP-08: Bit-flipped ciphertext authentication failure in StepIterative.
        Verifies that AES-GCM tag verification failure aborts with INVALID_ARGUMENT.
        """
        servicer = MatrixComputeServicer(
            signer=worker_signer,
            engine=compute_engine,
        )
        ctx = MockServicerGrpcContext()

        client_priv = x25519.X25519PrivateKey.generate()
        client_pub = client_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        lease_resp = servicer.LeaseMatrix(
            LeaseMatrixRequest(
                lease_id="bitflip_lease",
                rows=4,
                cols=4,
                dtype="fp32",
                matrix_bytes=np.eye(4, dtype=np.float32).tobytes(),
                ttl_seconds=60,
                require_confidential_compute=False,
                client_ephemeral_pubkey=client_pub,
            ),
            context=ctx,
        )

        sess = servicer.session_manager.get_session("bitflip_lease")
        salt = hashlib.sha256("iterative:bitflip_lease".encode("utf-8")).digest()
        keys = derive_session_keys(
            worker_priv=client_priv,
            client_pub_bytes=lease_resp.worker_ephemeral_pubkey,
            client_nonce=salt,
        )
        sess.session_keys = keys
        sess.require_confidential_compute = True

        raw_vec = np.ones(4, dtype=np.float32).tobytes()
        ct, iv, tag = aes_gcm_encrypt(key=keys.k_c2w, plaintext=raw_vec)

        # Flip 1 bit in ciphertext
        ct_tampered = bytearray(ct)
        ct_tampered[0] ^= 0x01
        ct_tampered = bytes(ct_tampered)

        req_tampered = IterativeStepRequest(
            lease_id="bitflip_lease",
            step_index=0,
            vector_bytes=ct_tampered,
            vector_length=4,
            dtype="fp32",
            is_encrypted=True,
            iv=iv,
            tag=tag,
        )

        with pytest.raises(RuntimeError):
            servicer.StepIterative(req_tampered, context=ctx)
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "decryption failed" in ctx.details.lower()

    def test_low_order_x25519_client_pubkey_rejection(
        self,
        compute_engine: ComputeEngine,
        worker_signer: WorkerSigner,
    ) -> None:
        """
        GAP-09: Low-order curve point submission (all zeros b"\\x00" * 32).
        Verifies rejection during ECDH key exchange with StatusCode.INTERNAL.
        """
        servicer = MatrixComputeServicer(
            signer=worker_signer,
            engine=compute_engine,
        )
        ctx = MockServicerGrpcContext()

        low_order_pub = b"\x00" * 32
        req = LeaseMatrixRequest(
            lease_id="low_order_lease",
            rows=4,
            cols=4,
            dtype="fp32",
            matrix_bytes=np.eye(4, dtype=np.float32).tobytes(),
            ttl_seconds=60,
            require_confidential_compute=True,
            client_ephemeral_pubkey=low_order_pub,
        )

        with pytest.raises(RuntimeError):
            servicer.LeaseMatrix(req, context=ctx)
        assert ctx.code == grpc.StatusCode.INTERNAL
        assert "ecdh key exchange failed" in ctx.details.lower()

    def test_confidential_mode_plaintext_matrix_policy_enforcement(
        self,
        compute_engine: ComputeEngine,
        worker_signer: WorkerSigner,
    ) -> None:
        """
        GAP-10: In LeaseMatrix with require_confidential_compute=True:
        1. Client transmits invalid client_ephemeral_pubkey (16B instead of 32B) -> INVALID_ARGUMENT.
        2. Client transmits plaintext matrix without 28B encrypted framing -> INVALID_ARGUMENT.
        """
        servicer = MatrixComputeServicer(
            signer=worker_signer,
            engine=compute_engine,
        )

        # Part 1: Invalid pubkey length
        ctx1 = MockServicerGrpcContext()
        req1 = LeaseMatrixRequest(
            lease_id="confidential_fail_lease",
            rows=4,
            cols=4,
            dtype="fp32",
            matrix_bytes=np.eye(4, dtype=np.float32).tobytes(),
            ttl_seconds=60,
            require_confidential_compute=True,
            client_ephemeral_pubkey=b"\x00" * 16,
        )
        with pytest.raises(RuntimeError):
            servicer.LeaseMatrix(req1, context=ctx1)
        assert ctx1.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "client_ephemeral_pubkey must be exactly 32 bytes" in ctx1.details.lower()

        # Part 2: Plaintext matrix bypass attempt with valid pubkey
        client_priv = x25519.X25519PrivateKey.generate()
        client_pub = client_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        ctx2 = MockServicerGrpcContext()
        req2 = LeaseMatrixRequest(
            lease_id="confidential_plaintext_bypass",
            rows=4,
            cols=4,
            dtype="fp32",
            matrix_bytes=np.eye(4, dtype=np.float32).tobytes(),  # 64 bytes instead of 64 + 28 = 92 bytes
            ttl_seconds=60,
            require_confidential_compute=True,
            client_ephemeral_pubkey=client_pub,
        )
        with pytest.raises(RuntimeError):
            servicer.LeaseMatrix(req2, context=ctx2)
        assert ctx2.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "confidential compute requires 28-byte encrypted framing" in ctx2.details.lower()


# ==============================================================================
# SUITE 3: SECP256K1 SIGNATURES & SETTLEMENT FAULT ISOLATION (5 Tests)
# ==============================================================================

class TestTier5CrossSubsystemSignaturesAndSettlementAdversarial:
    """
    Cross-subsystem adversarial tests exercising Secp256k1 signature validation,
    cross-step/cross-lease replay attacks, and mid-loop settlement fail-closed invariants.
    """

    def test_omitted_worker_signature_rejection(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        GAP-11: Worker omits required Secp256k1 signature on an iterative step (returns "").
        Verifies that coordinator raises WorkerSignatureError, refunds escrow, and releases lease.
        """
        class OmittedSigServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer
                self.released = False

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                sig = "" if request.step_index == 1 else self.signer.sign_message(f"{request.lease_id}:{request.step_index}:res".encode())
                res = np.ones((request.vector_length,), dtype=np.float32).tobytes()
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                self.released = True
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        omitted_servicer = OmittedSigServicer(worker_signer)
        worker_node = MockWorkerNode(
            worker_id="worker_omitted_sig",
            pubkey=worker_signer.get_pubkey_hex(),
            sats_per_gflop=1.0,
        )
        worker_node.servicer = omitted_servicer

        coord, _ = _make_test_coordinator(worker_node, client_signer, mock_settlement, mock_relay)

        mat_a = np.eye(4, dtype=np.float32)
        v0 = np.ones((4,), dtype=np.float32)

        with pytest.raises(WorkerSignatureError, match="omitted required Secp256k1 signature"):
            coord.execute_iterative_power(
                A=mat_a,
                v0=v0,
                iterations=3,
                candidate_workers=[worker_node],
            )

        assert omitted_servicer.released is True

    def test_rogue_signer_forged_signature_rejection(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        rogue_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        GAP-11: Worker signs step result using a rogue/unregistered private key.
        Verifies that coordinator detects signature verification failure and raises WorkerSignatureError.
        """
        class RogueSigServicer(MatrixComputeServiceServicer):
            def __init__(self, legitimate_signer: WorkerSigner, rogue_signer: WorkerSigner):
                self.legitimate_signer = legitimate_signer
                self.rogue_signer = rogue_signer
                self.released = False

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.legitimate_signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                res = np.ones((request.vector_length,), dtype=np.float32).tobytes()
                r_hash = hashlib.sha256(res).hexdigest()
                payload = f"{request.lease_id}:{request.step_index}:{r_hash}".encode("utf-8")
                forged_sig = self.rogue_signer.sign_message(payload)

                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res,
                    vector_length=request.vector_length,
                    worker_signature=forged_sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                self.released = True
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        rogue_servicer = RogueSigServicer(worker_signer, rogue_signer)
        worker_node = MockWorkerNode(
            worker_id="worker_rogue_sig",
            pubkey=worker_signer.get_pubkey_hex(),
            sats_per_gflop=1.0,
        )
        worker_node.servicer = rogue_servicer

        coord, _ = _make_test_coordinator(worker_node, client_signer, mock_settlement, mock_relay)

        mat_a = np.eye(4, dtype=np.float32)
        v0 = np.ones((4,), dtype=np.float32)

        with pytest.raises(WorkerSignatureError, match="Secp256k1 signature validation failed"):
            coord.execute_iterative_power(
                A=mat_a,
                v0=v0,
                iterations=2,
                candidate_workers=[worker_node],
            )

        assert rogue_servicer.released is True

    def test_perturbed_result_vector_signature_mismatch(
        self,
        worker_signer: WorkerSigner,
    ) -> None:
        """
        GAP-11: Worker computes valid signature over true result vector v_t, but alters
        a single byte of the result vector before sending.
        Verifies that signature verification fails due to SHA256 digest discrepancy.
        """
        lease_id = "lease_tamper_vec"
        step_index = 3
        true_vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        res_bytes = true_vector.tobytes()

        r_hash = hashlib.sha256(res_bytes).hexdigest()
        payload = f"{lease_id}:{step_index}:{r_hash}".encode("utf-8")
        sig = worker_signer.sign_message(payload)

        assert verify_signature(worker_signer.get_pubkey_hex(), payload, sig) is True

        tampered_vector = true_vector.copy()
        tampered_vector[0] += 0.001
        tampered_bytes = tampered_vector.tobytes()
        tampered_r_hash = hashlib.sha256(tampered_bytes).hexdigest()
        tampered_payload = f"{lease_id}:{step_index}:{tampered_r_hash}".encode("utf-8")

        assert verify_signature(worker_signer.get_pubkey_hex(), tampered_payload, sig) is False

    def test_cross_step_and_cross_lease_signature_replay_attack(
        self,
        worker_signer: WorkerSigner,
    ) -> None:
        """
        GAP-12: Verifies that valid signatures from step t=0 or lease L1 cannot be replayed
        onto step t=1 or lease L2 due to explicit lease_id and step_index binding in payload.
        """
        res_bytes = np.ones((4,), dtype=np.float32).tobytes()
        r_hash = hashlib.sha256(res_bytes).hexdigest()

        # Step 0 signature on Lease 1
        p_l1_s0 = f"lease_1:0:{r_hash}".encode("utf-8")
        sig_l1_s0 = worker_signer.sign_message(p_l1_s0)

        # 1. Step replay attack: replay sig_l1_s0 on step 1 of Lease 1
        p_l1_s1 = f"lease_1:1:{r_hash}".encode("utf-8")
        assert verify_signature(worker_signer.get_pubkey_hex(), p_l1_s1, sig_l1_s0) is False

        # 2. Lease replay attack: replay sig_l1_s0 on step 0 of Lease 2
        p_l2_s0 = f"lease_2:0:{r_hash}".encode("utf-8")
        assert verify_signature(worker_signer.get_pubkey_hex(), p_l2_s0, sig_l1_s0) is False

    def test_mid_loop_fault_settlement_cancellation_and_resource_reclamation(
        self,
        client_signer: ClientSigner,
        worker_signer: WorkerSigner,
        mock_settlement: MockSettlementService,
        mock_relay: MockNostrRelay,
    ) -> None:
        """
        GAP-13: Worker experiences network crash or unhandled failure mid-loop (step 2 of 5).
        Verifies fail-closed financial invariant:
        - Escrow is aborted and cancelled/refunded (zero payment to worker).
        - Worker payout is NEVER released.
        - ReleaseLease is guaranteed to execute in finally: block.
        """
        class DroppingWorkerServicer(MatrixComputeServiceServicer):
            def __init__(self, signer: WorkerSigner):
                self.signer = signer
                self.released = False
                self.step_count = 0

            def LeaseMatrix(self, request, context=None):
                return LeaseMatrixResponse(
                    lease_id=request.lease_id,
                    is_accepted=True,
                    expires_at_unix_ms=int(time.time() * 1000) + 3600000,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message="OK",
                )

            def StepIterative(self, request, context=None):
                self.step_count += 1
                if request.step_index >= 2:
                    raise RuntimeError("Worker node crashed mid-loop during iterative step 2")

                res = np.ones((request.vector_length,), dtype=np.float32).tobytes()
                r_hash = hashlib.sha256(res).hexdigest()
                sig = self.signer.sign_message(f"{request.lease_id}:{request.step_index}:{r_hash}".encode())
                return IterativeStepResponse(
                    lease_id=request.lease_id,
                    step_index=request.step_index,
                    result_vector_bytes=res,
                    vector_length=request.vector_length,
                    worker_signature=sig,
                    execution_time_ms=5,
                )

            def ReleaseLease(self, request, context=None):
                self.released = True
                return ReleaseLeaseResponse(lease_id=request.lease_id, is_released=True, bytes_freed=64)

        drop_servicer = DroppingWorkerServicer(worker_signer)
        worker_node = MockWorkerNode(
            worker_id="worker_dropping",
            pubkey=worker_signer.get_pubkey_hex(),
            sats_per_gflop=1.0,
        )
        worker_node.servicer = drop_servicer

        coord, escrow_mgr = _make_test_coordinator(worker_node, client_signer, mock_settlement, mock_relay)

        mat_a = np.eye(4, dtype=np.float32)
        v0 = np.ones((4,), dtype=np.float32)

        with pytest.raises(RuntimeError, match="Worker node crashed mid-loop"):
            coord.execute_iterative_power(
                A=mat_a,
                v0=v0,
                iterations=5,
                candidate_workers=[worker_node],
            )

        # ReleaseLease executed in finally: block
        assert drop_servicer.released is True

        # Escrows created under this job are cancelled/refunded, never settled
        for escrow in escrow_mgr._escrows.values():
            assert escrow.status == EscrowStatus.CANCELLED
            assert escrow.status != EscrowStatus.SETTLED
