"""
Adversarial Stress Test Suite for Milestone 3: Consumer SDK Stateful Session Coordinator.
Targeting: matrix-compute-sdk (ITERATIVE_POWER, verify_iterative_step, and VRAM Session Lease Coordinator).

Authored by: Adversarial Challenger 1 (teamwork_preview_challenger_m3_1).

Challenge Dimensions:
1. Single-Step Freivalds Verification (verify_iterative_step):
   - Boundary values: subnormal vectors (10^-35), zero vectors, large norms (10^6), extreme matrix condition numbers.
   - Subnormal underflow guards and zero-work fraud detection.
   - Precision formats: FP32, FP16, and BF16 with Higham bounds.
   - Relative discrepancy check dead-code analysis and bound saturation under large gamma_K / epsilon.
2. Network Upload Volume & Zero-Redundancy Invariant:
   - For any N-step iterative run, LeaseMatrix is called exactly 1 time (R_up(A) == 0.0).
   - Upload bytes scale strictly as O(N^2 + T * N) without re-transmitting matrix A.
   - Lifecycle guarantees: ReleaseLease executed under normal completion, fraud, and unhandled errors.
"""

from __future__ import annotations

import collections
import hashlib
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from matrix_contracts.v1 import (
    IterativeStepRequest,
    IterativeStepResponse,
    LeaseMatrixRequest,
    LeaseMatrixResponse,
    ReleaseLeaseRequest,
    ReleaseLeaseResponse,
)
from matrix_compute.config import ClientConfig, WorkloadType
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import (
    FraudDetectedError,
    MatrixDimensionError,
    NoWorkersFoundError,
    WorkerUnavailableError,
)
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator, IterativeResult
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import (
    MockWorkerNode,
    MockWorkerServicer,
    _InternalWorkerSigner,
)
from matrix_compute.tiling.precision import (
    cast_matrix,
    get_unit_roundoff,
    normalize_dtype,
    truncate_to_bf16,
)
from matrix_compute.verification.freivalds import (
    DEFAULT_RELATIVE_TOLERANCE_FLOOR,
    UNIT_ROUNDOFFS,
    FreivaldsVerifier,
    higham_bound,
    matrix_infinity_norm,
    vector_infinity_norm,
    verify_iterative_step,
)


# ==============================================================================
# Suite 1: Boundary Values & Extremes in verify_iterative_step
# ==============================================================================

class TestChallengerFreivaldsBoundaryExtremes:
    """Stress tests boundary values: subnormals, zeros, large norms, extreme condition numbers."""

    def test_adv_subnormal_vectors_genuine_computation_passes(self):
        """
        Tests subnormal vector inputs (~1e-35) in FP32.
        Genuine matrix-vector multiplication must pass without false alarms.
        """
        n = 8
        a = np.eye(n, dtype=np.float32)
        v_subnormal = (np.ones(n, dtype=np.float32) * 1e-35).astype(np.float32)
        v_next = np.matmul(a, v_subnormal)

        is_valid, disc, bound, proof = verify_iterative_step(a, v_subnormal, v_next, dtype="fp32")
        assert is_valid is True, f"Subnormal genuine computation was falsely rejected: {proof}"
        assert math.isfinite(bound)
        assert bound > 0.0

    def test_adv_subnormal_vectors_corrupted_computation_detected(self):
        """
        Tests that perturbations above underflow threshold on subnormal vectors
        are strictly flagged as fraud.
        """
        n = 8
        a = np.eye(n, dtype=np.float32)
        v_subnormal = (np.ones(n, dtype=np.float32) * 1e-35).astype(np.float32)
        v_corrupt = np.matmul(a, v_subnormal)
        # Inject perturbation of 1.0 (gross fraud)
        v_corrupt[0] += 1.0

        is_valid, disc, bound, proof = verify_iterative_step(a, v_subnormal, v_corrupt, dtype="fp32")
        assert is_valid is False
        assert disc >= 1.0 - 1e-5
        assert len(proof) > 0

    def test_adv_subnormal_underflow_guard_legitimate_hardware_flush(self):
        """
        Tests that when products underflow hardware limits (<1e-30) and flush to 0.0,
        the subnormal underflow guard correctly classifies it as legitimate hardware flush.
        """
        n = 4
        # Two matrices with tiny magnitudes whose product is ~1e-50 (below FP32 subnormal ~1.4e-45)
        a_tiny = np.eye(n, dtype=np.float32) * 1e-25
        v_tiny = (np.ones(n, dtype=np.float32) * 1e-25).astype(np.float32)
        # Hardware GEMM legitimately produces zeros
        v_flushed = np.zeros(n, dtype=np.float32)

        is_valid, disc, bound, proof = verify_iterative_step(a_tiny, v_tiny, v_flushed, dtype="fp32")
        assert is_valid is True, "Genuine subnormal float flushing was falsely flagged as fraud"
        assert len(proof) == 0

    def test_adv_zero_vector_input_honest_passes(self):
        """
        Tests behavior when state vector v_t is identically zero.
        Honest worker returning v_next = 0 must pass without division by zero.
        """
        n = 16
        a = np.random.randn(n, n).astype(np.float32)
        v_zero = np.zeros(n, dtype=np.float32)
        v_next_zero = np.zeros(n, dtype=np.float32)

        is_valid, disc, bound, proof = verify_iterative_step(a, v_zero, v_next_zero, dtype="fp32")
        assert is_valid is True
        assert disc == 0.0
        assert math.isfinite(bound)

    def test_adv_zero_vector_input_fraudulent_nonzero_detected(self):
        """
        Tests behavior when v_t = 0 but malicious worker returns non-zero v_next.
        Must be strictly flagged as fraud.
        """
        n = 16
        a = np.random.randn(n, n).astype(np.float32)
        v_zero = np.zeros(n, dtype=np.float32)
        v_fake = np.zeros(n, dtype=np.float32)
        v_fake[0] = 1.0  # Gross fraud

        is_valid, disc, bound, proof = verify_iterative_step(a, v_zero, v_fake, dtype="fp32")
        assert is_valid is False
        assert disc >= 1.0
        assert len(proof) > 0

    def test_adv_zero_matrix_input_honest_passes_and_fraud_caught(self):
        """
        Tests behavior when matrix A is identically zero.
        Honest v_next = 0 passes; fraudulent non-zero v_next is caught.
        """
        n = 12
        a_zero = np.zeros((n, n), dtype=np.float32)
        v_t = np.ones(n, dtype=np.float32)

        # Honest
        is_valid_h, disc_h, _, _ = verify_iterative_step(a_zero, v_t, np.zeros(n, dtype=np.float32))
        assert is_valid_h is True
        assert disc_h == 0.0

        # Fraudulent
        v_bad = np.ones(n, dtype=np.float32)
        is_valid_f, disc_f, _, proof_f = verify_iterative_step(a_zero, v_t, v_bad)
        assert is_valid_f is False
        assert disc_f >= 1.0

    def test_adv_large_norm_inputs_scale_1e6(self):
        """
        Tests large norm inputs (||A|| ~ 1e6, ||v|| ~ 1e6, scale ~ 1e12).
        Verifies Higham dynamic bound scales linearly and avoids false alarms.
        """
        n = 16
        np.random.seed(901)
        a = (np.random.randn(n, n) * 1e5).astype(np.float32)
        v_t = (np.random.randn(n) * 1e5).astype(np.float32)
        v_next = np.matmul(a, v_t)

        is_valid, disc, bound, proof = verify_iterative_step(a, v_t, v_next, dtype="fp32")
        assert is_valid is True, f"Large norm genuine computation failed: {proof}"
        assert math.isfinite(bound)
        assert bound > 1e4  # Bound scales proportionally with 1e10 - 1e12 scale

        # Corrupt with perturbation exceeding bound
        v_corrupt = v_next.copy()
        v_corrupt[0] += bound * 3.0
        is_valid_c, disc_c, _, proof_c = verify_iterative_step(a, v_t, v_corrupt, dtype="fp32")
        assert is_valid_c is False
        assert disc_c > bound

    def test_adv_extreme_matrix_condition_number_forward_error(self):
        """
        Tests matrix with extreme condition number kappa(A) = 1e12.
        Confirms that Higham forward error bound holds for honest matrix-vector multiplication,
        because matrix multiplication is forward-stable regardless of conditioning.
        """
        n = 8
        # Diagonal matrix with condition number 1e12
        s = np.geomspace(1e6, 1e-6, n).astype(np.float32)
        a_ill = np.diag(s)
        v_t = np.ones(n, dtype=np.float32)
        v_next = np.matmul(a_ill, v_t)

        is_valid, disc, bound, proof = verify_iterative_step(a_ill, v_t, v_next, dtype="fp32")
        assert is_valid is True
        assert disc <= bound
        assert len(proof) == 0

        # Tampering with dominant component is caught
        v_corrupt = v_next.copy()
        v_corrupt[0] += bound * 2.0
        is_valid_c, _, _, proof_c = verify_iterative_step(a_ill, v_t, v_corrupt, dtype="fp32")
        assert is_valid_c is False
        assert len(proof_c) > 0


# ==============================================================================
# Suite 2: Multi-Precision Formats & Higham Rounding Hierarchy
# ==============================================================================

class TestChallengerPrecisionHierarchy:
    """Tests FP32, FP16, and BF16 precision formats and Higham unit roundoff properties."""

    def test_adv_unit_roundoff_constants(self):
        """Verifies exact unit roundoffs u = 2^-24 (FP32), 2^-11 (FP16), 2^-8 (BF16)."""
        assert UNIT_ROUNDOFFS["fp32"] == 2.0 ** -24
        assert UNIT_ROUNDOFFS["fp16"] == 2.0 ** -11
        assert UNIT_ROUNDOFFS["bf16"] == 2.0 ** -8
        assert UNIT_ROUNDOFFS["fp64"] == 2.0 ** -53

    def test_adv_higham_bound_hierarchy_bf16_fp16_fp32(self):
        """Verifies bound ordering: Bound_BF16 > Bound_FP16 > Bound_FP32 for fixed dimensions."""
        n = 32
        norm_a, norm_v = 10.0, 5.0
        b_fp32 = higham_bound(n, norm_a, norm_v, "fp32")
        b_fp16 = higham_bound(n, norm_a, norm_v, "fp16")
        b_bf16 = higham_bound(n, norm_a, norm_v, "bf16")

        assert b_bf16 > b_fp16 > b_fp32
        # Ratio BF16 / FP16 should reflect 2^-8 / 2^-11 = 8x
        assert 7.0 < (b_bf16 / b_fp16) < 9.0
        # Ratio FP16 / FP32 should reflect 2^-11 / 2^-24 = 8192x
        assert 7000.0 < (b_fp16 / b_fp32) < 9500.0

    def test_adv_genuine_step_across_all_precisions(self):
        """Verifies genuine step evaluation passes across FP32, FP16, and BF16."""
        np.random.seed(902)
        n = 16

        # FP32
        a32 = np.random.randn(n, n).astype(np.float32) * 0.1
        v32 = np.random.randn(n).astype(np.float32) * 0.1
        v32_next = np.matmul(a32, v32)
        assert verify_iterative_step(a32, v32, v32_next, dtype="fp32").is_valid

        # FP16
        a16 = cast_matrix(a32, "fp16")
        v16 = cast_matrix(v32, "fp16")
        v16_next = cast_matrix(np.matmul(a16.astype(np.float32), v16.astype(np.float32)), "fp16")
        assert verify_iterative_step(a16, v16, v16_next, dtype="fp16").is_valid

        # BF16
        a_bf = truncate_to_bf16(a32)
        v_bf = truncate_to_bf16(v32)
        v_bf_next = truncate_to_bf16(np.matmul(a_bf, v_bf))
        assert verify_iterative_step(a_bf, v_bf, v_bf_next, dtype="bf16").is_valid

    def test_adv_adversarial_tampering_caught_in_all_precisions(self):
        """Verifies that an injected error of +1.0 is caught in FP32, FP16, and BF16."""
        np.random.seed(903)
        n = 16
        a = (np.random.randn(n, n) * 0.1).astype(np.float32)
        v = (np.random.randn(n) * 0.1).astype(np.float32)

        for dt in ("fp32", "fp16", "bf16"):
            v_next = np.matmul(a, v)
            v_bad = v_next.copy()
            v_bad[3] += 1.0  # perturbation

            res = verify_iterative_step(a, v, v_bad, dtype=dt)
            assert res.is_valid is False, f"Tampering +1.0 was not caught in {dt}"
            assert res.discrepancy >= 0.99


# ==============================================================================
# Suite 3: Forensic Audit of Relative Discrepancy Defense Defect
# ==============================================================================

class TestChallengerRelativeDiscrepancyForensicAudit:
    """
    Forensic stress tests uncovering structural dead-code and blind spots in
    verify_iterative_step relative discrepancy defense.
    """

    def test_adv_dead_code_discovery_in_relative_discrepancy_check(self):
        """
        Remediation Verification: In verify_iterative_step, relative discrepancy defense
        must detect zero-work fraud when discrepancy <= effective_bound under loose bounds.
        """
        # Construct scenario where relative difference is 100% (zero work),
        # but discrepancy <= effective_bound due to an explicit epsilon_tolerance.
        n = 4
        a = np.eye(n, dtype=np.float32)
        v_t = np.ones(n, dtype=np.float32)
        # Worker returns all zeros (zero work fraud)
        v_zero = np.zeros(n, dtype=np.float32)

        # High bound override: epsilon_tolerance = 2.0
        # discrepancy = ||A v_t - 0|| = 1.0
        # discrepancy (1.0) <= effective_bound (2.0)
        # Relative error = 1.0 / 1.0 = 1.0 (100% fraud!)
        res = verify_iterative_step(
            a,
            v_t,
            v_zero,
            dtype="fp32",
            epsilon_tolerance=2.0,
            relative_discrepancy_threshold=0.05,
        )

        assert res.is_valid is False, "Relative discrepancy check must reject zero-work fraud"
        assert "relative discrepancy" in res.failure_proof.get("reason", "").lower()
        assert res.effective_bound >= 2.0

    def test_adv_bf16_large_k_bound_saturation_zero_work(self):
        """
        Audit Finding: In BF16, u = 2^-8 = 1/256.
        When K >= 128, Higham gamma_K = (K*u)/(1 - K*u) >= 1.0.
        When K >= 256, gamma_K jumps to K*u*1000 >= 1000.0.
        For K >= 128, bound saturation no longer admits zero-work fraud because
        MAX_RELATIVE_DISCREPANCY_CEILING (0.50) caps allowed relative discrepancy.
        """
        k = 128
        a = np.eye(k, dtype=np.float32)
        v_t = np.ones(k, dtype=np.float32)
        v_zero = np.zeros(k, dtype=np.float32)

        # In BF16 with K=128:
        # gamma_128 = (128 * 2^-8) / (1 - 128 * 2^-8) = 0.5 / 0.5 = 1.0
        # effective_bound = 1.0 * scale = 1.0
        # discrepancy = 1.0
        # Relative discrepancy check catches 100% zero-work fraud!
        res = verify_iterative_step(a, v_t, v_zero, dtype="bf16")
        assert res.is_valid is False, (
            "BF16 Higham bound >= 1.0 zero-work fraud must be rejected by relative discrepancy ceiling"
        )


# ==============================================================================
# Suite 4: Network Upload Volume & Zero-Redundancy Invariant R_up(A) == 0.0
# ==============================================================================

class TestChallengerUploadVolumeAndZeroRedundancy:
    """
    Adversarial verification of network upload volume and zero-redundancy invariant:
    - LeaseMatrix called exactly 1 time across N iterations: R_up(A) == 0.0.
    - Upload bytes scale strictly as O(N^2 + T * N).
    - Lifecycle deallocation guarantees in coordinator finally: block.
    """

    @pytest.fixture
    def test_coordinator_env(self) -> Dict[str, Any]:
        """Hermetic coordinator test environment with tracking mock worker."""
        vk = "test-verification-key-m3-challenger"
        settlement = MockSettlementService(verification_key=vk)
        relay = MockNostrRelay()

        # Internal signer
        signer = _InternalWorkerSigner()
        servicer = MockWorkerServicer(signer=signer, settlement_service=settlement, verification_key=vk)

        # MockWorkerNode double
        worker = MockWorkerNode(
            settlement_service=settlement,
            verification_key=vk,
            vram_mb=16384,
            sats_per_gflop=0.0005,
        )
        worker.servicer = servicer
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

        client_signer = ClientSigner(config.consumer_privkey_hex)
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
            consumer_pubkey=client_signer.get_pubkey_hex(),
            connect_timeout=config.connect_timeout_seconds,
            task_timeout=config.task_timeout_seconds,
        )
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=0)

        coordinator = DistributedMatmulCoordinator(
            config=config,
            signer=client_signer,
            relay_client=relay_client,
            matchmaker=matchmaker,
            escrow_manager=escrow_manager,
            scheduler=scheduler,
            dispatcher=dispatcher,
            mock_workers=[worker],
        )

        return {
            "coordinator": coordinator,
            "worker": worker,
            "servicer": servicer,
        }

    def test_adv_upload_zero_redundancy_across_iteration_counts(self, test_coordinator_env: Dict[str, Any]):
        """
        TC_ADV_UP_01: Verifies that for T in {1, 5, 25, 50} iterations:
        1. LeaseMatrix is called exactly 1 time: lease_calls == 1.
        2. R_up(A) = (lease_calls - 1) / lease_calls == 0.0.
        3. StepIterative is called exactly T times: step_calls == T.
        4. Uploaded bytes strictly equal N^2 * B + T * N * B.
        """
        coord: DistributedMatmulCoordinator = test_coordinator_env["coordinator"]
        servicer: MockWorkerServicer = test_coordinator_env["servicer"]

        n = 16
        bytes_per_elem = 4  # FP32
        matrix_bytes = n * n * bytes_per_elem  # 1024 bytes
        vector_bytes_per_step = n * bytes_per_elem  # 64 bytes

        test_iterations = [1, 5, 20, 50]

        for iters in test_iterations:
            # Reset counters
            servicer.lease_calls = 0
            servicer.step_calls = 0
            servicer.release_calls = 0
            servicer.uploaded_bytes_total = 0

            a = np.eye(n, dtype=np.float32)
            v0 = np.ones(n, dtype=np.float32)

            res = coord.execute_iterative_power(a, v0, iterations=iters, dtype="fp32")
            assert len(res) == n

            # 1. LeaseMatrix called strictly once
            assert servicer.lease_calls == 1, f"Expected 1 lease call for T={iters}, got {servicer.lease_calls}"
            r_up_a = (servicer.lease_calls - 1.0) / float(servicer.lease_calls)
            assert r_up_a == 0.0, f"Upload redundancy ratio R_up(A) must be 0.0, got {r_up_a}"

            # 2. StepIterative called exactly T times
            assert servicer.step_calls == iters, f"Expected {iters} steps, got {servicer.step_calls}"

            # 3. ReleaseLease called strictly once
            assert servicer.release_calls == 1, "Expected 1 release call"

            # 4. Strict byte scaling: O(N^2 + T * N)
            expected_total_bytes = matrix_bytes + (iters * vector_bytes_per_step)
            assert servicer.uploaded_bytes_total == expected_total_bytes, (
                f"Byte mismatch for T={iters}: expected {expected_total_bytes}, got {servicer.uploaded_bytes_total}"
            )

    def test_adv_upload_volume_scaling_fp16_2bytes(self, test_coordinator_env: Dict[str, Any]):
        """
        TC_ADV_UP_02: Verifies upload volume scaling for FP16 (2 bytes per element):
        uploaded_bytes == N^2 * 2 + T * N * 2.
        """
        coord: DistributedMatmulCoordinator = test_coordinator_env["coordinator"]
        servicer: MockWorkerServicer = test_coordinator_env["servicer"]

        n = 20
        iters = 10
        bytes_per_elem = 2  # FP16

        servicer.lease_calls = 0
        servicer.step_calls = 0
        servicer.release_calls = 0
        servicer.uploaded_bytes_total = 0

        a = cast_matrix(np.eye(n, dtype=np.float32), "fp16")
        v0 = cast_matrix(np.ones(n, dtype=np.float32), "fp16")

        coord.execute_iterative_power(a, v0, iterations=iters, dtype="fp16")

        assert servicer.lease_calls == 1
        assert servicer.step_calls == iters
        expected_bytes = (n * n * bytes_per_elem) + (iters * n * bytes_per_elem)
        assert servicer.uploaded_bytes_total == expected_bytes

    def test_adv_lease_released_in_finally_on_fraud_slashing(self, test_coordinator_env: Dict[str, Any]):
        """
        TC_ADV_UP_03: Verifies that when fraud is detected mid-execution,
        ReleaseLease is guaranteed to execute in finally: block, freeing memory.
        """
        coord: DistributedMatmulCoordinator = test_coordinator_env["coordinator"]
        servicer: MockWorkerServicer = test_coordinator_env["servicer"]

        n = 16
        servicer.tamper_step_index = 2
        servicer.tamper_step_perturbation = 20.0

        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        with pytest.raises(FraudDetectedError):
            coord.execute_iterative_power(a, v0, iterations=10, spot_check_probability=1.0)

        # ReleaseLease was called despite FraudDetectedError
        assert servicer.release_calls >= 1
        # Pinned session was evicted and zeroized
        assert len(servicer._pinned_sessions) == 0
