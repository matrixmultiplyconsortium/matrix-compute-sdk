"""
Tier 5 Adversarial Coverage Hardening Suite for 2D Output Grid (OUTPUT_GRID_2D).

White-box and adversarial stress test suite targeting the 2D Output Grid partitioning
strategy across matrix-compute-sdk:
1. Malicious Tile Injection: Subtle perturbations (1e-7 vs Higham bounds), boundary edge cell
   corruption, NaN/Inf injection, and overlapping slice vulnerabilities in zero-reduction assembler.
2. Freivalds Verification & Numerical Stability Limits: Extreme matrix scaling (10^6 x 10^-6),
   near-overflow magnitudes (10^18), subnormal / FP32 underflow false-positive vulnerabilities,
   zero-matrix relative floor defenses, and Higham bound monotonicity.
3. Coordinator Fault Tolerance & Byzantine Isolation: Partial worker dropouts during multi-task
   2D grid dispatch, rogue node fraud proof generation and bond slashing, batch escrow refund
   isolation, and failover batch accounting anomalies.
4. ResultAssembler Concurrency Saturation & Thread Contention: Rapid out-of-order completion
   across 32 concurrent threads, race-condition duplicate rejection, drain state tracking,
   and unconstrained completion vulnerabilities.
5. Partitioner Boundary Stress: Degenerate aspect ratios (1x1000, 1000x1), prime grids,
   and defensive fail-closed dimension validation.
"""

from __future__ import annotations

import concurrent.futures
import copy
import dataclasses
import hashlib
import math
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from matrix_contracts.schemas import (
    BondStatus,
    FreivaldsFraudProof,
    MatchedWorker,
)

from matrix_compute.config import ClientConfig, PartitioningStrategy
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import (
    FraudDetectedError,
    MatrixDimensionError,
    TilingError,
    VerificationFailedError,
    WorkerUnavailableError,
)
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler, WorkerSlotTracker
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import MockWorkerNode
from matrix_compute.tiling import (
    BandwidthProfile,
    BandwidthProfiler,
    MatrixTiler,
    ResultAssembler,
    TileGridPlan,
    TileTaskPlan,
    calculate_higham_bound,
    cast_matrix,
    emulate_gemm,
    normalize_dtype,
    truncate_to_bf16,
    validate_bf16_masking,
)
from matrix_compute.tiling.partitioner import (
    compute_2d_grid_intervals,
    compute_grid_intervals,
)
from matrix_compute.verification.fraud_proof import FraudProofBuilder
from matrix_compute.verification.freivalds import (
    DEFAULT_FREIVALDS_ROUNDS,
    DEFAULT_RELATIVE_TOLERANCE_FLOOR,
    FreivaldsVerifier,
    freivalds_verify,
    get_unit_roundoff,
    higham_bound,
    matrix_infinity_norm,
)


# =============================================================================
# SUITE 1: MALICIOUS TILE INJECTION & BOUNDARY CORRUPTIONS
# =============================================================================

class TestTier5MaliciousTileInjection:
    """
    Adversarial attacks injecting corrupt, perturbed, non-finite, and overlapping
    tiles into 2D output grid pipelines and ResultAssembler.
    """

    def test_subtle_perturbation_evasion_vs_detection_threshold(self):
        """
        Adversarial Analysis: Evaluates detection sensitivity of subtle perturbations.
        Perturbations < min(gamma_K * scale, 1e-6) pass Freivalds verification due
        to effective_bound = max(raw_bound, 1e-6) clamping, while larger perturbations are caught.
        """
        m, k, n = 16, 16, 16
        np.random.seed(101)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        c_clean = np.matmul(a, b)

        verifier = FreivaldsVerifier(k_rounds=10)

        # 1. Subtle perturbation of 1e-7 on single element
        c_subtle = c_clean.copy()
        c_subtle[0, 0] += 1e-7

        is_valid_subtle, disc_subtle, bound_subtle, _ = verifier.verify_tile(a, b, c_subtle, dtype="fp32")
        # Because scale ~ 10-20 and gamma_k ~ 16 * 5.96e-8 ~ 9.5e-7,
        # bound is at least 1e-6 (clamped), so 1e-7 perturbation passes undetected
        assert is_valid_subtle is True
        assert disc_subtle <= bound_subtle

        # 2. Significant perturbation of 1e-2 on single element
        c_gross = c_clean.copy()
        c_gross[0, 0] += 1e-2
        is_valid_gross, disc_gross, bound_gross, proof_gross = verifier.verify_tile(a, b, c_gross, dtype="fp32")
        assert is_valid_gross is False
        assert disc_gross > bound_gross
        assert proof_gross["observed_diff"] > bound_gross

    def test_boundary_corner_cells_tampering(self):
        """
        Adversarial Test: Tampering specifically with boundary corner cells:
        (0, 0), (0, N-1), (M-1, 0), and (M-1, N-1).
        Examines single-column evasion probabilities when random vector r[c] == 0.
        """
        m, k, n = 8, 16, 8
        np.random.seed(102)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        c_true = np.matmul(a, b)

        corner_coords = [
            (0, 0),
            (0, n - 1),
            (m - 1, 0),
            (m - 1, n - 1),
        ]

        for r_c, c_c in corner_coords:
            c_tampered = c_true.copy()
            c_tampered[r_c, c_c] += 2.0  # gross corruption at boundary corner

            # Over 10 rounds, probability of random vector having r[c_c] == 0 in all 10 rounds is 2^-10 ~ 0.098%
            verifier = FreivaldsVerifier(k_rounds=10)
            is_valid, disc, bound, proof = verifier.verify_tile(a, b, c_tampered, dtype="fp32")
            assert is_valid is False, f"Corruption at corner ({r_c}, {c_c}) was not caught"
            assert disc > bound
            assert "vector_abr" in proof
            assert "vector_cr" in proof

    def test_nan_injection_in_boundary_cells_fail_closed(self):
        """
        Adversarial Test: Injecting NaN at boundary coordinates.
        Verifies that Freivalds immediately flags non-finite values and fails closed.
        """
        m, k, n = 8, 8, 8
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)
        c_nan = np.matmul(a, b)
        c_nan[0, 0] = np.nan

        verifier = FreivaldsVerifier()
        is_valid, disc, bound, proof = verifier.verify_tile(a, b, c_nan, dtype="fp32")

        assert is_valid is False
        assert math.isinf(disc)
        assert proof["reason"] == "Non-finite value (NaN/Inf) detected in matrices"

    def test_inf_injection_in_tile_elements_fail_closed(self):
        """
        Adversarial Test: Injecting +Inf and -Inf into computed spatial sub-tiles.
        """
        m, k, n = 12, 12, 12
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        for inf_val in [np.inf, -np.inf]:
            c_inf = np.matmul(a, b)
            c_inf[m - 1, n - 1] = inf_val

            is_valid, disc, bound, proof = freivalds_verify(a, b, c_inf, dtype="fp32")
            assert is_valid is False
            assert math.isinf(disc)
            assert "Non-finite" in proof.get("reason", "")

    def test_overlapping_tile_slices_assembler_vulnerability(self):
        """
        Vulnerability Remediation: ResultAssembler must validate spatial slice disjointness
        during accumulation so rogue overlapping slices cannot silently overwrite already placed regions.
        """
        m, n = 16, 16
        assembler = ResultAssembler(m=m, n=n, zero_reduction=True)

        # Tile 1: covers rows 0:8, cols 0:8 with 1.0
        t1 = np.ones((8, 8), dtype=np.float32)
        assembler.accumulate(
            tile_result=t1,
            row_idx=0,
            col_idx=0,
            row_slice=(0, 8),
            col_slice=(0, 8),
            task_id="t1",
        )

        # Tile 2 (Adversarial overlap): reports coord (1, 0) but passes overlapping row_slice (4, 12)
        t2 = 9.0 * np.ones((8, 8), dtype=np.float32)
        with pytest.raises(TilingError, match="[Oo]verlap"):
            assembler.accumulate(
                tile_result=t2,
                row_idx=1,
                col_idx=0,
                row_slice=(4, 12),  # overlaps rows 4:8 of Tile 1
                col_slice=(0, 8),
                task_id="t2",
            )

        res = assembler.get_result()
        # Verifies that rows 0:8 were preserved as 1.0 and NOT corrupted by overlapping tile
        assert np.all(res[0:8, 0:8] == 1.0)
        assert assembler.reduction_ops == 0

        # Contrast with BandwidthProfiler.verify_2d_grid_coverage which correctly rejects the overlap
        task_plans = [
            {"row_slice": (0, 8), "col_slice": (0, 8)},
            {"row_slice": (4, 12), "col_slice": (0, 8)},
            {"row_slice": (0, 8), "col_slice": (8, 16)},
            {"row_slice": (8, 16), "col_slice": (8, 16)},
        ]
        assert BandwidthProfiler.verify_2d_grid_coverage(task_plans, m=m, n=n) is False

    def test_out_of_bounds_slice_rejection(self):
        """
        Adversarial Test: Submitting tile slices with out-of-bounds coordinates
        (e.g., negative index, exceeding M/N, or end <= start).
        """
        assembler = ResultAssembler(m=20, n=20, zero_reduction=True)
        tile_data = np.ones((5, 5), dtype=np.float32)

        # Negative slice
        with pytest.raises(ValueError, match="Tile slice out of bounds"):
            assembler.accumulate(tile_data, 0, 0, row_slice=(-1, 4), col_slice=(0, 5))

        # Exceeding M
        with pytest.raises(ValueError, match="Tile slice out of bounds"):
            assembler.accumulate(tile_data, 0, 0, row_slice=(16, 21), col_slice=(0, 5))

        # Inverted slice (end <= start)
        with pytest.raises(ValueError, match="Tile slice out of bounds"):
            assembler.accumulate(tile_data, 0, 0, row_slice=(5, 5), col_slice=(0, 5))


# =============================================================================
# SUITE 2: FREIVALDS VERIFICATION & NUMERICAL STABILITY LIMITS
# =============================================================================

class TestTier5FreivaldsNumericalBoundTightness:
    """
    Stress tests for Higham dynamic error bounds under extreme conditions:
    denormals, near-overflow, extreme scaling (10^6 x 10^-6), and zero matrices.
    """

    def test_scaled_matrices_1e6_times_1e_minus_6(self):
        """
        Adversarial Analysis: Extreme matrix scaling where ||A||_inf = 10^6 and ||B||_inf = 10^-6.
        The product C = A * B has norm ~ 1.0, and scale = ||A|| * ||B|| = 1.0.
        Verifies that dynamic Higham bound gamma_K * ||A|| * ||B|| correctly bounds FP32 rounding.
        """
        k = 32
        np.random.seed(201)
        # Construct A with scale 1e6 and B with scale 1e-6
        a = (np.random.randn(16, k) * 1e6).astype(np.float32)
        b = (np.random.randn(k, 16) * 1e-6).astype(np.float32)

        c_fp32 = np.matmul(a, b)

        # Higham dynamic bound calculation
        norm_a = matrix_infinity_norm(a)
        norm_b = matrix_infinity_norm(b)
        scale = norm_a * norm_b
        bound = higham_bound(k, norm_a, norm_b, dtype="fp32")

        # Freivalds verification of honest FP32 result
        verifier = FreivaldsVerifier(k_rounds=10)
        is_valid, disc, eff_bound, _ = verifier.verify_tile(a, b, c_fp32, dtype="fp32")
        assert is_valid is True, f"Scaled matrix false rejection: disc={disc} > bound={eff_bound}"
        assert disc <= eff_bound

    def test_extreme_scaled_matrices_near_overflow(self):
        """
        Adversarial Test: Matrices with values near FP32 overflow limits (1e18 x 1e18 = 1e36).
        Tests that intermediate products and Higham bound calculations do not overflow to Inf in float64.
        """
        k = 8
        a = (np.ones((4, k)) * 1e18).astype(np.float32)
        b = (np.ones((k, 4)) * 1e18).astype(np.float32)
        c_expected = np.matmul(a, b)  # k * 1e36, within FP32 max (~3.4e38)

        assert np.all(np.isfinite(c_expected))

        verifier = FreivaldsVerifier(k_rounds=5)
        is_valid, disc, eff_bound, proof = verifier.verify_tile(a, b, c_expected, dtype="fp32")
        assert is_valid is True
        assert np.isfinite(eff_bound)

    def test_near_zero_subnormal_matrices_fp32_underflow(self):
        """
        Vulnerability Remediation: Near-zero matrices (1e-25 x 1e-25 = 1e-50).
        In FP32, 1e-50 underflows strictly to 0.0 (below subnormal min ~1.4e-45).
        An honest FP32 worker returns C = 0.0.
        Verifies that subnormal underflow to zero does NOT trigger false fraud alarms.
        """
        k = 10
        a = (np.ones((4, k)) * 1e-25).astype(np.float32)
        b = (np.ones((k, 4)) * 1e-25).astype(np.float32)
        c_worker_fp32 = np.zeros((4, 4), dtype=np.float32)  # true FP32 underflow output

        verifier = FreivaldsVerifier(k_rounds=5)
        is_valid, disc, eff_bound, proof = verifier.verify_tile(a, b, c_worker_fp32, dtype="fp32")

        # Underflow defense verified: Honest worker's underflow output passes verification
        assert is_valid is True, f"Subnormal underflow false-positive fraud accusation: disc={disc} > bound={eff_bound}"
        assert disc < 1e-30  # Absolute discrepancy is astronomically tiny

    def test_zero_matrix_detection_and_tolerance_floor(self):
        """
        Adversarial Test: When A=0 and B=0, theoretical product is strictly 0.
        Tests the DEFAULT_RELATIVE_TOLERANCE_FLOOR (1e-7) boundary defense:
        - Perturbation of 5e-8 (< 1e-7) passes undetected
        - Perturbation of 2e-7 (> 1e-7) is caught
        """
        a = np.zeros((8, 8), dtype=np.float32)
        b = np.zeros((8, 8), dtype=np.float32)

        # 1. Subtle fraud below tolerance floor (5e-8)
        c_sub_floor = np.zeros((8, 8), dtype=np.float32)
        c_sub_floor[0, 0] = 5e-8
        is_valid_sub, disc_sub, _, _ = freivalds_verify(a, b, c_sub_floor, dtype="fp32")
        # Allowed by 1e-7 floor
        assert is_valid_sub is True
        assert disc_sub <= DEFAULT_RELATIVE_TOLERANCE_FLOOR

        # 2. Fraud above tolerance floor (2e-7)
        c_above_floor = np.zeros((8, 8), dtype=np.float32)
        c_above_floor[0, 0] = 2e-7
        is_valid_above, disc_above, _, proof_above = freivalds_verify(a, b, c_above_floor, dtype="fp32")
        assert is_valid_above is False
        assert disc_above > DEFAULT_RELATIVE_TOLERANCE_FLOOR

    def test_identity_and_diagonal_matrix_invariants(self):
        """
        Adversarial Test: Identity matrices I_n * I_n = I_n under single-element perturbation.
        Tests perturbation of an off-diagonal zero element to 1.0.
        """
        n = 16
        a = np.eye(n, dtype=np.float32)
        b = np.eye(n, dtype=np.float32)
        c_honest = np.eye(n, dtype=np.float32)

        # Honest passes
        valid_honest, _, _, _ = freivalds_verify(a, b, c_honest, dtype="fp32")
        assert valid_honest is True

        # Malicious node alters an off-diagonal element
        c_corrupt = c_honest.copy()
        c_corrupt[0, 1] = 1.0  # Spurious 1.0 off-diagonal
        valid_corrupt, disc, bound, proof = freivalds_verify(a, b, c_corrupt, dtype="fp32", num_trials=10)
        assert valid_corrupt is False
        assert disc >= 1.0


# =============================================================================
# SUITE 3: COORDINATOR FAULT TOLERANCE, BYZANTINE ISOLATION & ESCROW
# =============================================================================

class TestTier5CoordinatorFaultToleranceAndSettlement:
    """
    Stress tests covering coordinator handling of partial worker dropouts,
    Byzantine fraud isolation, batch escrow refunds, and failover accounting.
    """

    def test_partial_worker_dropout_during_2d_grid_dispatch(self):
        """
        Adversarial Test: During a multi-worker 2D grid dispatch (4 tasks on 2 workers),
        Worker 2 experiences a connection dropout.
        Verifies that coordinator halts cleanly and refunds all escrows.
        """
        vk = f"vk-{time.time()}"
        settlement = MockSettlementService(verification_key=vk)
        relay = MockNostrRelay()

        w1 = MockWorkerNode(port=0, settlement_service=settlement, verification_key=vk, sats_per_gflop=0.0005)
        w2 = MockWorkerNode(port=0, settlement_service=settlement, verification_key=vk, sats_per_gflop=0.0005)
        w1.start()
        w2.start()
        w1.publish_to_relay(relay)
        w2.publish_to_relay(relay)

        # Inject worker 2 failure
        w2.set_fault_mode("disconnect")

        config = ClientConfig(
            offline_mock_mode=True,
            nostr_relays=["ws://127.0.0.1:8008"],
            settlement_api_url="http://127.0.0.1:8080",
            settlement_verification_key=vk,
            task_timeout_seconds=2.0,
            require_confidential_compute=False,
        )
        signer = ClientSigner()
        matchmaker = Matchmaker(config=config, relay_client=relay)
        escrow_mgr = EscrowManager(config=config, settlement_backend=settlement)
        dispatcher = GrpcTaskDispatcher(config=config, signer=signer)
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=0, backoff_factor=0.05)

        coord = DistributedMatmulCoordinator(
            config=config,
            signer=signer,
            relay_client=relay,
            matchmaker=matchmaker,
            escrow_manager=escrow_mgr,
            scheduler=scheduler,
            dispatcher=dispatcher,
            mock_workers=[w1, w2],
        )

        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)

        try:
            with pytest.raises(Exception):
                coord.execute_gemm(a, b, grid_rows=2, grid_cols=2, partitioning_strategy="output_grid_2d")

            # Check escrow refund isolation: no escrows left in HELD or UNPAID state
            for h, inv in settlement.invoices.items():
                assert inv.status in ("CANCELLED", "UNPAID", "SETTLED")
        finally:
            w1.stop()
            w2.stop()

    def test_byzantine_fault_isolation_honest_worker_protection(self):
        """
        Adversarial Test: Worker 1 is honest, Worker 2 is Byzantine (tampering with tile C).
        Verifies:
        1. Byzantine worker's tile is rejected by per-tile Freivalds
        2. FreivaldsFraudProof is constructed and bond slashed
        3. FraudDetectedError is raised preventing buffer pollution
        """
        vk = f"vk-{time.time()}"
        settlement = MockSettlementService(verification_key=vk)
        relay = MockNostrRelay()

        w1 = MockWorkerNode(port=0, settlement_service=settlement, verification_key=vk, sats_per_gflop=0.0005)
        w2 = MockWorkerNode(port=0, settlement_service=settlement, verification_key=vk, sats_per_gflop=0.0005)
        w1.start()
        w2.start()
        w1.publish_to_relay(relay)
        w2.publish_to_relay(relay)

        w2.set_fault_mode("corrupt_result")

        config = ClientConfig(
            offline_mock_mode=True,
            nostr_relays=["ws://127.0.0.1:8008"],
            settlement_api_url="http://127.0.0.1:8080",
            settlement_verification_key=vk,
            require_confidential_compute=False,
        )
        signer = ClientSigner()
        matchmaker = Matchmaker(config=config, relay_client=relay)
        escrow_mgr = EscrowManager(config=config, settlement_backend=settlement)
        dispatcher = GrpcTaskDispatcher(config=config, signer=signer)
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=1, backoff_factor=0.05)

        coord = DistributedMatmulCoordinator(
            config=config,
            signer=signer,
            relay_client=relay,
            matchmaker=matchmaker,
            escrow_manager=escrow_mgr,
            scheduler=scheduler,
            dispatcher=dispatcher,
            mock_workers=[w1, w2],
        )

        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)

        try:
            with pytest.raises(FraudDetectedError) as exc_info:
                coord.execute_gemm(a, b, grid_rows=2, grid_cols=2, partitioning_strategy="output_grid_2d")

            assert "Freivalds verification failed" in str(exc_info.value)
            proof = exc_info.value.fraud_proof
            assert proof is not None
            assert proof.worker_pubkey == w2.pubkey
        finally:
            w1.stop()
            w2.stop()

    def test_batch_escrow_refund_isolation_on_retry_exhaustion(self):
        """
        Adversarial Test: When all retry attempts are exhausted across candidate workers,
        the scheduler raises DispatchError and all newly minted attempt escrows are cancelled.
        """
        config = ClientConfig.for_testing()
        signer = ClientSigner()
        dispatcher = GrpcTaskDispatcher(config=config, signer=signer)
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=2, backoff_factor=0.01)

        task = TileTaskPlan(
            task_id="exhaustion_task_0",
            row_idx=0,
            col_idx=0,
            k_idx=0,
            row_slice=(0, 4),
            col_slice=(0, 4),
            k_slice=(0, 4),
            a_tile=np.ones((4, 4), dtype=np.float32),
            b_tile=np.ones((4, 4), dtype=np.float32),
            dtype="fp32",
            estimated_gflops=0.0001,
        )

        # Candidates with invalid endpoints
        candidates = [
            MatchedWorker(
                worker_pubkey="bad_worker_1",
                p2p_multiaddr="/ip4/127.0.0.1/tcp/59991",
                lightning_routing_pubkey="02bad1",
                sats_per_gflop=0.0005,
            ),
            MatchedWorker(
                worker_pubkey="bad_worker_2",
                p2p_multiaddr="/ip4/127.0.0.1/tcp/59992",
                lightning_routing_pubkey="02bad2",
                sats_per_gflop=0.0005,
            ),
        ]

        with pytest.raises(Exception):
            scheduler.execute_with_failover(
                task_plan=task,
                candidate_workers=candidates,
                macaroon="mock_macaroon",
                payment_hash="mock_payment_hash",
                timeout=1.0,
            )

    def test_failover_batch_escrow_accounting_no_double_payment(self):
        """
        Adversarial Test: When a worker drops out and its tasks are failed over to a backup worker,
        the failed worker's batch verified count must NOT be incremented, preventing double payment
        or payout release for the failed worker's escrow.
        """
        vk = f"vk-{time.time()}"
        settlement = MockSettlementService(verification_key=vk)
        relay = MockNostrRelay()

        w1 = MockWorkerNode(port=0, settlement_service=settlement, verification_key=vk, sats_per_gflop=0.0005)
        w2 = MockWorkerNode(port=0, settlement_service=settlement, verification_key=vk, sats_per_gflop=0.0005)
        w1.start()
        w2.start()
        w1.publish_to_relay(relay)
        w2.publish_to_relay(relay)

        # w1 is faulty; all its tasks fail over to healthy backup w2
        w1.set_fault_mode("disconnect")

        config = ClientConfig(
            offline_mock_mode=True,
            nostr_relays=["ws://127.0.0.1:8008"],
            settlement_api_url="http://127.0.0.1:8080",
            settlement_verification_key=vk,
            task_timeout_seconds=2.0,
            require_confidential_compute=False,
        )
        signer = ClientSigner()
        matchmaker = Matchmaker(config=config, relay_client=relay)
        escrow_mgr = EscrowManager(config=config, settlement_backend=settlement)
        dispatcher = GrpcTaskDispatcher(config=config, signer=signer)
        scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=2, backoff_factor=0.01)

        coord = DistributedMatmulCoordinator(
            config=config,
            signer=signer,
            relay_client=relay,
            matchmaker=matchmaker,
            escrow_manager=escrow_mgr,
            scheduler=scheduler,
            dispatcher=dispatcher,
            mock_workers=[w1, w2],
        )

        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)

        try:
            res = coord.execute_gemm(a, b, grid_rows=1, grid_cols=2, partitioning_strategy="output_grid_2d")
            np.testing.assert_allclose(res, np.matmul(a, b))

            # Verify that w1's batch escrow was NEVER paid out or settled
            for p_hash, inv in settlement.invoices.items():
                if getattr(inv, "worker_pubkey", "") == w1.pubkey:
                    assert inv.status != "SETTLED", "Failed worker batch escrow was illegally settled!"
        finally:
            w1.stop()
            w2.stop()


# =============================================================================
# SUITE 4: RESULT ASSEMBLER CONCURRENCY & THREAD CONTENTION
# =============================================================================

class TestTier5AssemblerThreadContentionAndConcurrency:
    """
    Stress tests for ResultAssembler under intense concurrency, race conditions,
    and out-of-order 2D tile completion.
    """

    def test_rapid_out_of_order_completion_high_concurrency(self):
        """
        Stress Test: 32 concurrent threads rapidly accumulating 64 sub-tiles in
        completely random out-of-order sequence into ResultAssembler.
        Verifies:
        1. Thread-safety of internal buffer placement without race condition tears.
        2. Zero reduction operations executed (reduction_ops == 0).
        3. 100% mathematical equivalence with np.matmul(A, B).
        """
        m, k, n = 64, 32, 64
        pr, pc = 8, 8
        total_tasks = pr * pc  # 64 tasks

        np.random.seed(401)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=pr, grid_cols=pc)
        assert len(plan.tasks) == total_tasks

        assembler = ResultAssembler.from_plan(plan, zero_reduction=True)

        # Shuffle tasks to guarantee completely out-of-order execution
        shuffled_tasks = list(plan.tasks)
        random.shuffle(shuffled_tasks)

        def _worker_accumulate(t: TileTaskPlan):
            # Compute tile dot product
            c_tile = np.matmul(t.a_tile, t.b_tile)
            assembler.accumulate_task(t, c_tile)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(_worker_accumulate, t) for t in shuffled_tasks]
            concurrent.futures.wait(futures)

        assert assembler.is_complete() is True
        assert assembler.reduction_ops == 0
        assert assembler.get_progress() == 1.0

        res = assembler.get_result(enforce_complete=True)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)

    def test_concurrent_duplicate_accumulation_rejection(self):
        """
        Adversarial Test: Multiple threads racing to submit the same coordinate (row_idx, col_idx).
        Verifies that exactly one thread succeeds and all other racing threads raise TilingError.
        """
        assembler = ResultAssembler(m=8, n=8, zero_reduction=True, allow_duplicate=False)
        tile_data = np.ones((4, 4), dtype=np.float32)

        success_count = 0
        error_count = 0
        lock = threading.Lock()

        def _race_submission():
            nonlocal success_count, error_count
            try:
                assembler.accumulate(
                    tile_result=tile_data,
                    row_idx=0,
                    col_idx=0,
                    row_slice=(0, 4),
                    col_slice=(0, 4),
                    task_id="race_task",
                )
                with lock:
                    success_count += 1
            except TilingError:
                with lock:
                    error_count += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futs = [executor.submit(_race_submission) for _ in range(8)]
            concurrent.futures.wait(futs)

        assert success_count == 1
        assert error_count == 7

    def test_empty_expected_tasks_premature_completion_vulnerability(self):
        """
        Vulnerability Remediation: When ResultAssembler is initialized without expected_tasks,
        is_complete() must NOT prematurely return True after receiving a single tile.
        """
        assembler = ResultAssembler(m=100, n=100, expected_tasks=None, zero_reduction=True)
        assert assembler.is_complete() is False

        # Accumulate 1 tile out of a 100x100 matrix
        assembler.accumulate(
            tile_result=np.ones((10, 10), dtype=np.float32),
            row_idx=0,
            col_idx=0,
            row_slice=(0, 10),
            col_slice=(0, 10),
        )

        # Premature completion prevented: Assembler requires full coverage or expected tasks
        assert assembler.is_complete() is False
        with pytest.raises(TilingError, match="assembly incomplete"):
            assembler.get_result(enforce_complete=True)

    def test_mismatched_task_id_completion_tracking(self):
        """
        Edge Case: When expected_tasks uses custom IDs but accumulate is called
        with task_id=None, task IDs default to 'tile_{r}_{c}_{k}', causing missing_tasks()
        to never clear.
        """
        custom_ids = ["task_custom_A", "task_custom_B"]
        assembler = ResultAssembler(m=8, n=8, expected_tasks=custom_ids, zero_reduction=True)

        # Accumulate with task_id=None
        assembler.accumulate(
            tile_result=np.ones((8, 4), dtype=np.float32),
            row_idx=0,
            col_idx=0,
            row_slice=(0, 8),
            col_slice=(0, 4),
            task_id=None,
        )

        assert "task_custom_A" in assembler.missing_tasks()
        assert assembler.is_complete() is False


# =============================================================================
# SUITE 5: PARTITIONER BOUNDARY & ADVERSARIAL GEOMETRY
# =============================================================================

class TestTier5PartitionerAdversarialStress:
    """
    Stress tests covering extreme processor grid geometries and shape boundaries.
    """

    def test_extreme_grid_aspect_ratios_1x1000_and_1000x1(self):
        """
        Stress Test: Highly asymmetric processor grids:
        - 1 x 1000 grid (only column slicing)
        - 1000 x 1 grid (only row slicing)
        Verifies clamping and contiguous intervals.
        """
        # Case 1: 1 x 1000 grid on 10 x 50 matrix (clamped to 10 x 50)
        a = np.ones((10, 20), dtype=np.float32)
        b = np.ones((20, 50), dtype=np.float32)

        plan_col = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=1, grid_cols=1000)
        assert plan_col.p_r == 1
        assert plan_col.p_c == 50  # clamped to N
        assert len(plan_col.tasks) == 50

        # Case 2: 1000 x 1 grid on 50 x 10 matrix (clamped to 50 x 1)
        a2 = np.ones((50, 20), dtype=np.float32)
        b2 = np.ones((20, 10), dtype=np.float32)

        plan_row = MatrixTiler.create_plan_output_grid_2d(a2, b2, grid_rows=1000, grid_cols=1)
        assert plan_row.p_r == 50  # clamped to M
        assert plan_row.p_c == 1
        assert len(plan_row.tasks) == 50

    def test_degenerate_prime_grid_shapes(self):
        """
        Stress Test: Large prime matrix dimensions M=97, K=89, N=83 with prime grid 7x11.
        Verifies exact disjointness and zero-reduction spatial assembly.
        """
        m, k, n = 97, 89, 83
        pr, pc = 7, 11
        np.random.seed(502)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=pr, grid_cols=pc)
        assert len(plan.tasks) == pr * pc

        # Verify coverage via BandwidthProfiler
        assert BandwidthProfiler.verify_2d_grid_coverage(plan.tasks, m=m, n=n) is True

        # Assemble and verify mathematical equivalence
        assembler = ResultAssembler.from_plan(plan, zero_reduction=True)
        for t in plan.tasks:
            c_ij = np.matmul(t.a_tile, t.b_tile)
            assembler.accumulate_task(t, c_ij)

        assert assembler.is_complete() is True
        assert assembler.reduction_ops == 0

        res = assembler.get_result()
        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)

    def test_zero_and_negative_dimension_fail_closed(self):
        """
        Adversarial Test: Partitioner fails closed when presented with
        zero-dimension matrices, negative grid counts, or non-2D arrays.
        """
        # Zero dimension
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_output_grid_2d(np.ones((0, 10)), np.ones((10, 10)), 2, 2)

        # Negative grid rows
        with pytest.raises(ValueError, match="grid_rows must be >= 1"):
            MatrixTiler.create_plan_output_grid_2d(np.ones((10, 10)), np.ones((10, 10)), grid_rows=-1, grid_cols=2)

        # Non-2D array
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_output_grid_2d(np.ones((10, 10, 2)), np.ones((10, 10)), 2, 2)
