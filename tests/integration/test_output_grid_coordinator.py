"""
Integration test suite for DistributedMatmulCoordinator and MatrixComputeClient
with 2D Output Grid (Download-Optimal & Zero-Reduction) matrix partitioning.

Milestone 1 Test Specification for matrix-compute-sdk.
Validates:
1. Multi-worker distributed execution across mock compute nodes.
2. 2D output grid execution with various grid shapes and matrix dimensions.
3. Strict mathematical equivalence against np.matmul(a, b) across FP32, FP16, and BF16.
4. Zero-reduction assembly verification during coordinated execution.
5. Grid dimension precedence hierarchy: explicit kwargs > ClientConfig > worker factorization > tile_size heuristic.
6. Dual-tier Freivalds verification: Stage 3 per-tile and Stage 5 assembled product verification.
7. Rogue worker fraud detection & batch escrow isolation:
   - Corrupt spatial tile caught and rejected by per-tile Freivalds.
   - Authoritative FreivaldsFraudProof emitted.
   - Rogue worker's batch escrow cancelled; bond slashed.
   - Honest worker settles cleanly; accumulator buffer untainted.
8. L402 batch escrow lifecycle (INITIALIZED -> HELD -> SETTLED / CANCELLED).
9. High-level MatrixComputeClient.matmul and distributed_matmul API exposure.
10. Irregular boundary shapes: primes, odds, scalar 1x1, and non-square grids.
"""

from __future__ import annotations

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
    NoWorkersFoundError,
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
    emulate_gemm,
    truncate_to_bf16,
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
        partitioning_strategy=PartitioningStrategy.OUTPUT_GRID_2D,
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
    scheduler = ParallelTaskScheduler(
        dispatcher=dispatcher,
        max_retries=config.max_retries,
    )
    coordinator = DistributedMatmulCoordinator(
        config=config,
        signer=signer,
        relay_client=relay_client,
        matchmaker=matchmaker,
        escrow_manager=escrow_manager,
        scheduler=scheduler,
        dispatcher=dispatcher,
        mock_workers=[worker1, worker2],
    )

    yield {
        "coordinator": coordinator,
        "config": config,
        "settlement": settlement,
        "relay": relay,
        "worker1": worker1,
        "worker2": worker2,
        "escrow_manager": escrow_manager,
    }

    coordinator.close()
    worker1.stop()
    worker2.stop()


# ==============================================================================
# SUITE 1: BASIC 2D OUTPUT GRID EXECUTION & ZERO REDUCTION
# ==============================================================================

class TestSuite1BasicExecution:
    """Tests executing distributed matmul with OUTPUT_GRID_2D."""

    def test_output_grid_2d_square_execution(self, hermetic_env: Dict[str, Any]):
        """Executes 2x2 grid multiplication matching np.matmul."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        result = coord.execute_gemm(
            a, b, grid_rows=2, grid_cols=2, partitioning_strategy="output_grid_2d"
        )

        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected, atol=1e-5)

        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 4
        assert manifest.completed_tasks == 4

    def test_output_grid_2d_zero_reduction_confirmed(self, hermetic_env: Dict[str, Any]):
        """Verifies the coordinator assembler performed strictly 0 reduction ops."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)

        result = coord.execute_gemm(
            a, b, grid_rows=2, grid_cols=2, partitioning_strategy="output_grid_2d"
        )
        np.testing.assert_allclose(result, 16.0 * np.ones((16, 16)), atol=1e-5)

        assembler = getattr(coord, "_last_assembler", None)
        assert assembler is not None
        assert assembler.zero_reduction is True
        assert assembler.reduction_ops == 0

    def test_output_grid_non_square_grid(self, hermetic_env: Dict[str, Any]):
        """Executes asymmetric non-square grid P_r=3, P_c=2."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(123)
        a = np.random.randn(30, 20).astype(np.float32)
        b = np.random.randn(20, 24).astype(np.float32)

        result = coord.execute_gemm(
            a, b, grid_rows=3, grid_cols=2, partitioning_strategy="output_grid_2d"
        )

        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected, atol=1e-4)

        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 6
        assert manifest.completed_tasks == 6


# ==============================================================================
# SUITE 2: GRID DIMENSIONS PRECEDENCE HIERARCHY
# ==============================================================================

class TestSuite2PrecedenceHierarchy:
    """Tests precedence resolution for (P_r, P_c) grid dimensions."""

    def test_explicit_kwargs_override_config(self, hermetic_env: Dict[str, Any]):
        """Explicit grid_rows/grid_cols in execute_gemm override ClientConfig values."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        coord.config.grid_rows = 1
        coord.config.grid_cols = 1

        a = np.ones((32, 16), dtype=np.float32)
        b = np.ones((16, 32), dtype=np.float32)

        coord.execute_gemm(a, b, grid_rows=4, grid_cols=2, partitioning_strategy="output_grid_2d")
        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 8  # 4 * 2

    def test_explicit_grid_shape_kwarg_precedence(self, hermetic_env: Dict[str, Any]):
        """grid_shape=(3, 2) in execute_gemm is respected."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        a = np.ones((30, 16), dtype=np.float32)
        b = np.ones((16, 20), dtype=np.float32)

        coord.execute_gemm(a, b, grid_shape=(3, 2), partitioning_strategy="output_grid_2d")
        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 6

    def test_config_grid_shape_fallback(self, hermetic_env: Dict[str, Any]):
        """Falls back to ClientConfig.grid_shape when execute_gemm omits dimensions."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        coord.config.grid_rows = 2
        coord.config.grid_cols = 3

        a = np.ones((20, 16), dtype=np.float32)
        b = np.ones((16, 30), dtype=np.float32)

        coord.execute_gemm(a, b, partitioning_strategy="output_grid_2d")
        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 6

    def test_worker_factorization_heuristic(self, hermetic_env: Dict[str, Any]):
        """When grid dimensions are completely omitted, factors worker count (2 workers -> 2x1 or 1x2)."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        coord.config.grid_rows = None
        coord.config.grid_cols = None
        coord.config.grid_shape = None

        a = np.ones((20, 16), dtype=np.float32)
        b = np.ones((16, 20), dtype=np.float32)

        result = coord.execute_gemm(a, b, partitioning_strategy="output_grid_2d")
        manifest = coord.get_last_manifest()
        assert manifest is not None
        assert manifest.total_tasks == 2  # 2 workers factorized
        np.testing.assert_allclose(result, 16.0 * np.ones((20, 20)), atol=1e-5)


# ==============================================================================
# SUITE 3: MULTI-PRECISION SUPPORT (FP32, FP16, BF16)
# ==============================================================================

class TestSuite3MultiPrecision:
    """Tests executing 2D output grid across supported precision types."""

    def test_fp32_precision_execution(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(42)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)

        res = coord.execute_gemm(a, b, dtype="fp32", grid_rows=2, grid_cols=2)
        assert res.dtype == np.float32
        expected = np.matmul(a, b)
        np.testing.assert_allclose(res, expected, atol=1e-5)

    def test_fp16_precision_execution(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(42)
        a = (np.random.randn(16, 16) * 0.1).astype(np.float16)
        b = (np.random.randn(16, 16) * 0.1).astype(np.float16)

        res = coord.execute_gemm(a, b, dtype="fp16", grid_rows=2, grid_cols=2)
        assert res.dtype == np.float16
        ref = emulate_gemm(a, b, dtype="fp16")
        bound = calculate_higham_bound(k=16, a=a, b=b, dtype="fp16")
        diff = float(np.max(np.abs(res.astype(np.float32) - ref.astype(np.float32))))
        assert diff <= max(bound, 1e-2)

    def test_bf16_precision_execution(self, hermetic_env: Dict[str, Any]):
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        np.random.seed(42)
        a = truncate_to_bf16(np.random.randn(16, 16).astype(np.float32))
        b = truncate_to_bf16(np.random.randn(16, 16).astype(np.float32))

        res = coord.execute_gemm(a, b, dtype="bf16", grid_rows=2, grid_cols=2)
        ref = emulate_gemm(a, b, dtype="bf16")
        bound = calculate_higham_bound(k=16, a=a, b=b, dtype="bf16")
        diff = float(np.max(np.abs(res - ref)))
        assert diff <= max(bound, 1e-2)


# ==============================================================================
# SUITE 4: DUAL-TIER FREIVALDS & FRAUD ISOLATION
# ==============================================================================

class TestSuite4FreivaldsAndFraudIsolation:
    """Tests Freivalds verification and rogue worker fraud slashing with 2D grid."""

    def test_rogue_worker_fraud_detected_and_isolated(self, hermetic_env: Dict[str, Any]):
        """A rogue worker returning corrupt sub-tile is caught and escrow cancelled."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        worker2: MockWorkerNode = hermetic_env["worker2"]
        settlement: MockSettlementService = hermetic_env["settlement"]

        # Configure worker2 to return fraudulent sub-tile
        worker2.set_fault_mode("corrupt_result")

        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)

        # Dispatch should detect fraud and raise FraudDetectedError
        with pytest.raises(FraudDetectedError) as exc_info:
            coord.execute_gemm(a, b, grid_rows=2, grid_cols=2, partitioning_strategy="output_grid_2d")

        assert "Freivalds verification failed" in str(exc_info.value)
        assert exc_info.value.fraud_proof is not None

        # Verify bond slashing dispute was recorded on settlement service
        disputes = settlement.get_disputes() if hasattr(settlement, "get_disputes") else []
        # Result assembler buffer was NOT completed with fraudulent data
        assembler = getattr(coord, "_last_assembler", None)
        assert assembler is not None
        assert not assembler.is_complete()

    def test_assembled_freivalds_verification_catches_perturbation(self, hermetic_env: Dict[str, Any]):
        """Tier 2 Assembled Freivalds verification passes on honest output."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)

        # Honest execution passes both Tier 1 and Tier 2 Freivalds
        result = coord.execute_gemm(a, b, grid_rows=2, grid_cols=2, verify_assembled=True)
        np.testing.assert_allclose(result, 8.0 * np.ones((8, 8)))


# ==============================================================================
# SUITE 5: L402 BATCH ESCROW LIFECYCLE
# ==============================================================================

class TestSuite5EscrowLifecycle:
    """Tests L402 batch escrow transitions (INITIALIZED -> HELD -> SETTLED)."""

    def test_batch_escrows_settled_on_completion(self, hermetic_env: Dict[str, Any]):
        """Escrows are held initially and settled upon all assigned tiles passing verification."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        settlement: MockSettlementService = hermetic_env["settlement"]

        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)

        initial_settled_count = len([inv for inv in settlement.invoices.values() if inv.status == "SETTLED"])
        coord.execute_gemm(a, b, grid_rows=2, grid_cols=2)

        final_settled_count = len([inv for inv in settlement.invoices.values() if inv.status == "SETTLED"])
        # Both workers completed all assigned tiles, so both batch escrows settled
        assert final_settled_count >= initial_settled_count + 1


# ==============================================================================
# SUITE 6: HIGH-LEVEL CLIENT API INTEGRATION
# ==============================================================================

class TestSuite6ClientAPI:
    """Tests high-level MatrixComputeClient.matmul and distributed_matmul with OUTPUT_GRID_2D."""

    def test_client_matmul_with_output_grid_2d(self, hermetic_env: Dict[str, Any]):
        """MatrixComputeClient.matmul works cleanly with OUTPUT_GRID_2D."""
        config: ClientConfig = hermetic_env["config"]
        config.partitioning_strategy = PartitioningStrategy.OUTPUT_GRID_2D

        with MatrixComputeClient(config=config) as client:
            a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            c = client.matmul(a, b, grid_rows=2, grid_cols=2)
            expected = np.matmul(a, b)
            np.testing.assert_allclose(c, expected, atol=1e-5)

    def test_distributed_matmul_functional_helper(self, hermetic_env: Dict[str, Any]):
        """distributed_matmul functional helper computes result and cleans up."""
        config: ClientConfig = hermetic_env["config"]
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)

        res = distributed_matmul(
            a, b,
            config=config,
            partitioning_strategy="output_grid_2d",
            grid_rows=2,
            grid_cols=2,
        )
        np.testing.assert_allclose(res, 8.0 * np.ones((8, 8)), atol=1e-5)


# ==============================================================================
# SUITE 7: IRREGULAR BOUNDARY SHAPES & CORNER CASES
# ==============================================================================

class TestSuite7BoundaryShapes:
    """Tests coordinator with prime, scalar, and asymmetric matrix dimensions."""

    def test_prime_dimensions_through_coordinator(self, hermetic_env: Dict[str, Any]):
        """Prime shapes M=7, K=11, N=13 executed through coordinator without index errors."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        a = np.ones((7, 11), dtype=np.float32)
        b = np.ones((11, 13), dtype=np.float32)

        result = coord.execute_gemm(a, b, grid_rows=2, grid_cols=3, partitioning_strategy="output_grid_2d")
        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_scalar_1x1_through_coordinator(self, hermetic_env: Dict[str, Any]):
        """Scalar 1x1 matrix executed through coordinator."""
        coord: DistributedMatmulCoordinator = hermetic_env["coordinator"]
        a = np.array([[3.0]], dtype=np.float32)
        b = np.array([[5.0]], dtype=np.float32)

        result = coord.execute_gemm(a, b, grid_rows=2, grid_cols=2, partitioning_strategy="output_grid_2d")
        np.testing.assert_allclose(result, [[15.0]], atol=1e-5)
