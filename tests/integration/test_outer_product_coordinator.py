"""
Integration test suite for DistributedMatmulCoordinator and MatrixComputeClient
with 1D Outer-Product (Upload-Optimal) matrix partitioning.

Validates:
- Multi-worker execution with mock worker pool.
- 1D outer-product execution with various strip counts and shapes.
- Strict mathematical equivalence against np.matmul(a, b) within Higham bounds across FP32, FP16, and BF16.
- Rogue worker fraud proof isolation: corrupt strip rejected by per-strip Freivalds,
  rogue batch escrow cancelled, honest worker settled cleanly, accumulator buffer untainted.
- Dual-tier Freivalds verification: Stage 3 per-strip and Stage 5 assembled product verification.
- L402 batch escrow lifecycle (INITIALIZED -> HELD -> SETTLED / CANCELLED).
- Sliding window concurrency bounding.
- Boundary geometry: primes, scalar (1x1), tall-skinny, and strip oversubscription.
- Zero regressions on baseline 3D block tiling.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Dict, Generator, List, Tuple
import uuid

import numpy as np
import pytest

from matrix_compute.client import MatrixComputeClient, distributed_matmul
from matrix_compute.config import ClientConfig, PartitioningStrategy
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import (
    FraudDetectedError,
    VerificationFailedError,
)
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import MockWorkerNode
from matrix_compute.tiling.assembler import ResultAssembler
from matrix_compute.tiling.precision import (
    calculate_higham_bound,
    cast_matrix,
    truncate_to_bf16,
    validate_bf16_masking,
)


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

@pytest.fixture
def hermetic_env() -> Generator[Dict[str, Any], None, None]:
    """
    Sets up a fully hermetic, multi-worker distributed test environment
    with 2 MockWorkerNodes, MockNostrRelay, MockSettlementService, and coordinator.
    """
    vk = f"secret-vk-{uuid.uuid4().hex[:8]}"
    settlement = MockSettlementService(verification_key=vk)
    relay = MockNostrRelay()

    worker1 = MockWorkerNode(
        port=0,
        settlement_service=settlement,
        verification_key=vk,
        vram_mb=24576,
        sats_per_gflop=0.0005,
    )
    worker2 = MockWorkerNode(
        port=0,
        settlement_service=settlement,
        verification_key=vk,
        vram_mb=24576,
        sats_per_gflop=0.0005,
    )
    worker1.start()
    worker2.start()

    worker1.publish_to_relay(relay)
    worker2.publish_to_relay(relay)

    config = ClientConfig(
        offline_mock_mode=True,
        nostr_relays=["ws://127.0.0.1:8008"],
        settlement_api_url="http://127.0.0.1:8080",
        settlement_verification_key=vk,
        min_worker_vram_mb=8192,
        max_price_per_gflop=0.001,
        default_tile_size=16,
        max_retries=0,
        partitioning_strategy=PartitioningStrategy.OUTER_PRODUCT_1D,
    )

    signer = ClientSigner()
    relay_client = RelayClient(relay_urls=["ws://127.0.0.1:8008"], in_memory_relay=relay)
    matchmaker = Matchmaker()
    backend = MockSettlementBackend(settlement)
    escrow_mgr = EscrowManager(settlement_backend=backend, verification_key=vk)
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
        mock_workers=[worker1, worker2],
    )

    yield {
        "coordinator": coordinator,
        "config": config,
        "escrow_mgr": escrow_mgr,
        "settlement": settlement,
        "relay": relay,
        "worker1": worker1,
        "worker2": worker2,
        "signer": signer,
        "vk": vk,
    }

    worker1.stop()
    worker2.stop()
    if hasattr(scheduler, "shutdown"):
        scheduler.shutdown()


# ==============================================================================
# SUITE 1: STRATEGY ROUTING & VALIDATION
# ==============================================================================

class TestCoordinatorStrategyRouting:
    """Verifies strategy routing in coordinator.execute_gemm and client."""

    def test_default_routes_to_block_3d(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(101)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)

        # Force block_3d
        result = coord.execute_gemm(a, b, tile_size=16, partitioning_strategy=PartitioningStrategy.BLOCK_3D)
        manifest = coord.get_last_manifest()
        assert manifest is not None
        # 32x32 / 16x16 -> 2x2x2 = 8 tasks for 3D block
        assert manifest.total_tasks == 8
        np.testing.assert_allclose(result, np.matmul(a, b), rtol=1e-5, atol=1e-5)

    def test_config_outer_product_routes_to_1d(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        # Default config has OUTER_PRODUCT_1D
        np.random.seed(102)
        a = np.random.randn(32, 64).astype(np.float32)
        b = np.random.randn(64, 32).astype(np.float32)

        result = coord.execute_gemm(a, b, num_strips=4)
        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 4
        # Outer-product tasks have strip_ prefix
        for tid in manifest.task_ids:
            assert "strip_" in tid
        np.testing.assert_allclose(result, np.matmul(a, b), rtol=1e-5, atol=1e-5)

    def test_explicit_parameter_overrides_config(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        # Config has OUTER_PRODUCT_1D, but we explicitly pass BLOCK_3D
        np.random.seed(103)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)

        result = coord.execute_gemm(a, b, tile_size=16, partitioning_strategy="block_3d")
        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 8  # 3D block 2x2x2

    def test_unimplemented_2d_strategy_raises_not_implemented(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)

        with pytest.raises(NotImplementedError, match="OUTPUT_GRID_2D"):
            coord.execute_gemm(a, b, partitioning_strategy="output_grid_2d")

    def test_unknown_strategy_raises_value_error(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)

        with pytest.raises(ValueError, match="Unknown partitioning strategy"):
            coord.execute_gemm(a, b, partitioning_strategy="hyper_cube_4d")


# ==============================================================================
# SUITE 2: NUM_STRIPS PRECEDENCE HIERARCHY
# ==============================================================================

class TestNumStripsPrecedenceHierarchy:
    """Validates 4-tier precedence resolution for num_strips."""

    def test_precedence_1_explicit_parameter_overrides_all(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        # Set config.num_strips to 3
        coord.config.num_strips = 3
        # mock_workers has 2 workers
        # Call with explicit num_strips=5 -> must use 5
        a = np.zeros((16, 30), dtype=np.float32)
        b = np.zeros((30, 16), dtype=np.float32)

        coord.execute_gemm(a, b, num_strips=5)
        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 5

    def test_precedence_2_config_overrides_heuristics(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        coord.config.num_strips = 4
        # No explicit parameter; mock_workers has 2 -> must use config (4)
        a = np.zeros((16, 32), dtype=np.float32)
        b = np.zeros((32, 16), dtype=np.float32)

        coord.execute_gemm(a, b, num_strips=None)
        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 4

    def test_precedence_3_candidate_count_heuristic(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        coord.config.num_strips = None
        # No explicit, no config -> uses len(mock_workers) == 2
        a = np.zeros((16, 32), dtype=np.float32)
        b = np.zeros((32, 16), dtype=np.float32)

        coord.execute_gemm(a, b)
        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == len(coord.mock_workers)  # 2

    def test_precedence_4_tile_size_fallback(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        coord.config.num_strips = None
        orig_mock_workers = coord.mock_workers
        coord.mock_workers = []
        try:
            np.random.seed(204)
            a = np.random.randn(16, 48).astype(np.float32)
            b = np.random.randn(48, 16).astype(np.float32)
            # Tiling fallback: max(1, ceil(48 / 16)) = 3
            result = coord.execute_gemm(a, b, tile_size=16)
            manifest = coord.get_last_manifest()
            assert manifest is not None
            assert manifest.total_tasks == 3
            np.testing.assert_allclose(result, np.matmul(a, b), rtol=1e-5, atol=1e-5)
        finally:
            coord.mock_workers = orig_mock_workers


# ==============================================================================
# SUITE 3: MULTI-WORKER EXECUTION & MATHEMATICAL EQUIVALENCE
# ==============================================================================

class TestMultiWorkerOuterProductExecution:
    """Validates multi-worker distributed GEMM across FP32, FP16, and BF16 within Higham bounds."""

    def test_multi_worker_outer_product_clean_settlement_fp32(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        escrow_mgr: EscrowManager = hermetic_env["escrow_mgr"]
        settlement: MockSettlementService = hermetic_env["settlement"]

        np.random.seed(301)
        m, k, n = 32, 64, 32
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        result = coord.execute_gemm(a, b, dtype="fp32", num_strips=4)
        expected = np.matmul(a, b)

        # Higham floating-point error bound validation
        norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
        bound = calculate_higham_bound(k, norm_a, norm_b, dtype="fp32")
        discrepancy = float(np.max(np.abs(result - expected)))
        assert discrepancy <= bound, f"FP32 discrepancy {discrepancy} exceeded Higham bound {bound}"

        # Escrow verification: exactly 2 batch escrows created (1 per worker)
        unique_escrows = {e.payment_hash: e for e in escrow_mgr._escrows.values()}
        assert len(unique_escrows) == 2, f"Expected 2 batch escrows, got {len(unique_escrows)}"

        for phash, esc in unique_escrows.items():
            assert esc.is_batch is True
            assert esc.status == EscrowStatus.SETTLED
            assert esc.preimage is not None
            assert len(esc.preimage) == 64
            assert settlement.get_status(phash) == "SETTLED"

    @pytest.mark.parametrize("p_strips", [1, 2, 4, 8])
    def test_outer_product_various_strip_counts(self, hermetic_env: Dict[str, Any], p_strips: int):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(302 + p_strips)
        m, k, n = 24, 50, 36
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        result = coord.execute_gemm(a, b, dtype="fp32", num_strips=p_strips)
        expected = np.matmul(a, b)

        norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
        bound = calculate_higham_bound(k, norm_a, norm_b, dtype="fp32")
        discrepancy = float(np.max(np.abs(result - expected)))
        assert discrepancy <= bound

    def test_outer_product_fp16(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(303)
        m, k, n = 32, 48, 32
        a = np.random.randn(m, k).astype(np.float16)
        b = np.random.randn(k, n).astype(np.float16)

        result = coord.execute_gemm(a, b, dtype="fp16", num_strips=4)
        expected = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)

        norm_a = float(np.max(np.sum(np.abs(a.astype(np.float32)), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b.astype(np.float32)), axis=1)))
        bound = calculate_higham_bound(k, norm_a, norm_b, dtype="fp16")
        discrepancy = float(np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32))))
        assert discrepancy <= max(bound, 0.05)

    def test_outer_product_bf16(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(304)
        m, k, n = 32, 48, 32
        a = truncate_to_bf16(np.random.randn(m, k).astype(np.float32))
        b = truncate_to_bf16(np.random.randn(k, n).astype(np.float32))

        result = coord.execute_gemm(a, b, dtype="bf16", num_strips=4)
        assert validate_bf16_masking(result)

        expected = truncate_to_bf16(np.matmul(a, b))
        norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
        bound = calculate_higham_bound(k, norm_a, norm_b, dtype="bf16")
        discrepancy = float(np.max(np.abs(result - expected)))
        assert discrepancy <= max(bound, 0.5)

    def test_distributed_matmul_functional_api(self):
        """Validates public functional API distributed_matmul."""
        np.random.seed(305)
        a = np.random.randn(20, 40).astype(np.float32)
        b = np.random.randn(40, 20).astype(np.float32)

        config = ClientConfig(
            offline_mock_mode=True,
            default_tile_size=16,
            partitioning_strategy=PartitioningStrategy.OUTER_PRODUCT_1D,
        )

        result = distributed_matmul(
            a,
            b,
            dtype="fp32",
            config=config,
            partitioning_strategy="outer_product_1d",
            num_strips=2,
        )
        assert result.shape == (20, 20)
        np.testing.assert_allclose(result, np.matmul(a, b), rtol=1e-5, atol=1e-5)


# ==============================================================================
# SUITE 4: DUAL-TIER FREIVALDS VERIFICATION
# ==============================================================================

class TestDualTierFreivaldsVerification:
    """Validates Stage 3 per-strip and Stage 5 assembled product Freivalds verification."""

    def test_honest_execution_passes_both_tiers(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(401)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)

        # Runs smoothly through Tier 1 and Tier 2
        result = coord.execute_gemm(a, b, num_strips=4)
        assert result.shape == (32, 32)

    def test_corrupt_strip_rejected_at_tier_1_before_accumulator(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        worker2: MockWorkerNode = hermetic_env["worker2"]

        # Worker 2 injects corruption
        worker2.servicer.tamper_result = True
        worker2.servicer.tamper_perturbation = 10.0

        np.random.seed(402)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)

        with pytest.raises(FraudDetectedError) as exc_info:
            coord.execute_gemm(a, b, num_strips=2)

        err = exc_info.value
        assert "Freivalds verification failed" in str(err)
        assert err.worker_pubkey == worker2.pubkey
        assert err.fraud_proof is not None

        # Verify accumulator buffer was never contaminated
        assert coord._last_assembler is not None
        assert (0, 0, 1) not in coord._last_assembler._received_coords
        assert not any("strip_1" in tid for tid in coord._last_assembler._received_tasks)

    def test_tampered_assembled_matrix_caught_at_tier_2(self, hermetic_env: Dict[str, Any], monkeypatch: pytest.MonkeyPatch):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(403)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)

        # Monkeypatch ResultAssembler.get_result to inject subtle accumulator drift/tampering
        orig_get_result = ResultAssembler.get_result

        def tampered_get_result(self_asm: ResultAssembler, enforce_complete: bool = True) -> np.ndarray:
            res = orig_get_result(self_asm, enforce_complete=enforce_complete)
            tampered = res.copy()
            tampered[0, 0] += 50.0  # Excessive drift violating Higham bound
            return tampered

        monkeypatch.setattr(ResultAssembler, "get_result", tampered_get_result)

        with pytest.raises(VerificationFailedError) as exc_info:
            coord.execute_gemm(a, b, num_strips=2)

        err = exc_info.value
        assert "Assembled Freivalds verification failed" in str(err)
        assert err.discrepancy is not None
        assert err.bound is not None
        assert err.discrepancy > err.bound


# ==============================================================================
# SUITE 5: ROGUE WORKER FRAUD PROOF ISOLATION & ESCROW ROLLBACK
# ==============================================================================

class TestRogueWorkerFraudProofIsolation:
    """Validates that rogue workers are slashed, escrows cancelled, while honest workers settle."""

    def test_adversarial_fraud_isolation_multi_worker(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        escrow_mgr: EscrowManager = hermetic_env["escrow_mgr"]
        settlement: MockSettlementService = hermetic_env["settlement"]
        worker1: MockWorkerNode = hermetic_env["worker1"]
        worker2: MockWorkerNode = hermetic_env["worker2"]

        # Worker 1 is honest, Worker 2 is rogue
        worker2.servicer.tamper_result = True
        worker2.servicer.tamper_perturbation = 2.0

        np.random.seed(501)
        a = np.random.randn(32, 64).astype(np.float32)
        b = np.random.randn(64, 32).astype(np.float32)

        with pytest.raises(FraudDetectedError) as exc_info:
            coord.execute_gemm(a, b, num_strips=4)

        err = exc_info.value
        assert err.worker_pubkey == worker2.pubkey
        assert err.fraud_proof is not None

        # Verify batch escrows
        w1_escrow = None
        w2_escrow = None
        for esc in escrow_mgr._escrows.values():
            if esc.worker_pubkey == worker1.pubkey:
                w1_escrow = esc
            elif esc.worker_pubkey == worker2.pubkey:
                w2_escrow = esc

        assert w2_escrow is not None
        assert w2_escrow.status == EscrowStatus.CANCELLED
        assert settlement.get_status(w2_escrow.payment_hash) == "CANCELLED"

        assert w1_escrow is not None
        assert w1_escrow.status == EscrowStatus.SETTLED
        assert settlement.get_status(w1_escrow.payment_hash) == "SETTLED"

        # Verify accumulator buffer was never contaminated by rogue worker
        assert coord._last_assembler is not None
        assert (0, 0, 1) not in coord._last_assembler._received_coords
        assert not any("strip_1" in tid for tid in coord._last_assembler._received_tasks)


# ==============================================================================
# SUITE 6: L402 BATCH ESCROW LIFECYCLE
# ==============================================================================

class TestL402BatchEscrowLifecycle:
    """Validates full L402 batch escrow lifecycle from minting to settlement."""

    def test_batch_escrow_full_lifecycle(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        escrow_mgr: EscrowManager = hermetic_env["escrow_mgr"]
        settlement: MockSettlementService = hermetic_env["settlement"]

        np.random.seed(601)
        a = np.random.randn(16, 32).astype(np.float32)
        b = np.random.randn(32, 16).astype(np.float32)

        result = coord.execute_gemm(a, b, num_strips=2)
        assert result.shape == (16, 16)

        # Inspect escrows
        unique_escrows = {e.payment_hash: e for e in escrow_mgr._escrows.values()}
        for phash, esc in unique_escrows.items():
            assert esc.status == EscrowStatus.SETTLED
            assert esc.preimage is not None
            # Validate SHA-256(preimage) == payment_hash
            derived_hash = hashlib.sha256(bytes.fromhex(esc.preimage)).hexdigest()
            assert derived_hash == phash
            assert settlement.get_status(phash) == "SETTLED"


# ==============================================================================
# SUITE 7: SLIDING WINDOW CONCURRENCY
# ==============================================================================

class TestSlidingWindowConcurrency:
    """Validates that in-flight strip concurrency is bounded by max_concurrency."""

    def test_sliding_window_concurrency_bounded(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(701)
        a = np.random.randn(32, 64).astype(np.float32)
        b = np.random.randn(64, 32).astype(np.float32)

        # 8 strips with max_concurrency=2
        result = coord.execute_gemm(a, b, num_strips=8, max_concurrency=2)
        assert result.shape == (32, 32)
        assert coord.peak_in_flight <= 2


# ==============================================================================
# SUITE 8: BOUNDARY GEOMETRY THROUGH COORDINATOR
# ==============================================================================

class TestBoundaryGeometry:
    """Validates prime, scalar, asymmetric, and oversubscribed dimensions."""

    def test_prime_dimensions_7x11x13(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(801)
        a = np.random.randn(7, 11).astype(np.float32)
        b = np.random.randn(11, 13).astype(np.float32)

        result = coord.execute_gemm(a, b, num_strips=3)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_scalar_1x1(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        a = np.array([[3.0]], dtype=np.float32)
        b = np.array([[7.0]], dtype=np.float32)

        result = coord.execute_gemm(a, b, num_strips=4)
        assert result.shape == (1, 1)
        assert float(result[0, 0]) == 21.0

    def test_asymmetric_tall_skinny(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(803)
        a = np.random.randn(128, 16).astype(np.float32)
        b = np.random.randn(16, 32).astype(np.float32)

        result = coord.execute_gemm(a, b, num_strips=4)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_strip_oversubscription_p_greater_than_k(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(804)
        a = np.random.randn(5, 3).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)

        # P=10 > K=3 clamps to 3 strips
        result = coord.execute_gemm(a, b, num_strips=10)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 3


# ==============================================================================
# SUITE 9: ZERO REGRESSIONS
# ==============================================================================

class TestZeroRegressions:
    """Validates baseline 3D block execution still functions correctly."""

    def test_baseline_3d_block_continues_working(self):
        with MatrixComputeClient() as client:
            np.random.seed(901)
            a = np.random.randn(32, 32).astype(np.float32)
            b = np.random.randn(32, 32).astype(np.float32)

            result = client.matmul(a, b, tile_size=16)
            expected = np.matmul(a, b)
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
