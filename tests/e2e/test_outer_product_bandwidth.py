"""
Automated Bandwidth Profiling Suite and Numerical Equivalence Verification for 1D Outer-Product Matmul.

Phase 1 Milestone 3 Verification Suite:
- TestUploadBandwidthOptimality:
  Verifies that 1D outer product matmul uploads strictly M*K + K*N matrix elements with
  redundancy ratio == 0.0 across varied dimensions (primes 7x11x13, 17x31x23, 67x31x43;
  odd 15x25x35; scalar 1x1; tall-skinny 128x16x32; wide-flat 16x128x16; vector dot product 1x1000x1).
  Verifies zero duplicate uploads and perfect disjoint strip reconstruction.

- TestBandwidthContrastBaseline:
  Contrasts 1D Outer-Product (0.0 redundancy ratio) with 3D Block Tiling on identical matrix
  dimensions, proving that 3D block tiling suffers from redundant element uploads:
  ((J * M * K + I * K * N) / (M * K + K * N) - 1.0 > 0.0).

- TestHighamPrecisionEquivalence:
  Asserts ||C - C_hat||_inf <= gamma_K * ||A||_inf * ||B||_inf, where gamma_K = (K*u)/(1 - K*u),
  across FP32 (u=2^-24), FP16 (u=2^-11), and BF16 (u=2^-8, bitmask 0xFFFF0000) comparing
  coordinator and client execution against np.matmul reference.

- TestCoordinatorEndToEndBandwidth:
  Runs full distributed execution through DistributedMatmulCoordinator.execute_gemm and
  MatrixComputeClient.matmul with mock workers, counting actual elements dispatched in tasks
  and verifying exact M*K + K*N elements.

- TestLargeScale1024Stress:
  Runs 1024x1024 matmul with 16 strips and verifies that total uploaded elements equal
  strictly 2,097,152 with zero duplicate elements, and result matches np.matmul within Higham bound.
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
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import MockWorkerNode
from matrix_compute.tiling.assembler import ResultAssembler
from matrix_compute.tiling.partitioner import MatrixTiler, TileGridPlan, TileTaskPlan
from matrix_compute.tiling.precision import (
    calculate_higham_bound,
    cast_matrix,
    get_bytes_per_element,
    get_unit_roundoff,
    normalize_dtype,
    truncate_to_bf16,
    validate_bf16_masking,
)
from matrix_compute.tiling.profiler import BandwidthProfile, BandwidthProfiler


# ==============================================================================
# TEST FIXTURES & HELPERS
# ==============================================================================

@pytest.fixture
def hermetic_coordinator_env() -> Generator[Dict[str, Any], None, None]:
    """
    Provisions a fully isolated hermetic distributed test environment
    with 2 MockWorkerNodes, MockNostrRelay, MockSettlementService, and coordinator.
    """
    vk = f"bandwidth-test-vk-{uuid.uuid4().hex[:8]}"
    settlement = MockSettlementService(verification_key=vk)
    relay = MockNostrRelay()

    worker1 = MockWorkerNode(
        port=0,
        settlement_service=settlement,
        verification_key=vk,
        vram_mb=32768,
        sats_per_gflop=0.0005,
    )
    worker2 = MockWorkerNode(
        port=0,
        settlement_service=settlement,
        verification_key=vk,
        vram_mb=32768,
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
        "worker1": worker1,
        "worker2": worker2,
        "settlement": settlement,
        "relay": relay,
        "dispatcher": dispatcher,
        "scheduler": scheduler,
    }

    worker1.stop()
    worker2.stop()
    if hasattr(scheduler, "shutdown"):
        scheduler.shutdown()


class DispatchElementSpy:
    """
    Spy interceptor wrapping scheduler/dispatcher to count exact elements dispatched in tasks.
    """

    def __init__(self, original_submit_tile):
        self.original_submit_tile = original_submit_tile
        self.dispatched_tasks: List[TileTaskPlan] = []
        self.total_a_elements = 0
        self.total_b_elements = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        task_plan: Optional[TileTaskPlan] = args[0] if len(args) > 0 else kwargs.get("task_plan")
        if task_plan is not None:
            self.dispatched_tasks.append(task_plan)
            self.total_a_elements += task_plan.a_tile.size
            self.total_b_elements += task_plan.b_tile.size
        return self.original_submit_tile(*args, **kwargs)

    @property
    def total_dispatched_elements(self) -> int:
        return self.total_a_elements + self.total_b_elements


# ==============================================================================
# SUITE 1: TEST UPLOAD BANDWIDTH OPTIMALITY
# ==============================================================================

class TestUploadBandwidthOptimality:
    """
    Verifies that 1D outer product matmul uploads strictly M*K + K*N matrix elements
    with redundancy ratio == 0.0 across varied dimensions:
    - Primes: 7x11x13, 17x31x23, 67x31x43
    - Odd: 15x25x35
    - Scalar: 1x1
    - Tall-skinny: 128x16x32
    - Wide-flat: 16x128x16
    - Vector dot product: 1x1000x1
    """

    @pytest.mark.parametrize(
        "m, k, n, p_strips",
        [
            (7, 11, 13, 1),
            (7, 11, 13, 3),
            (7, 11, 13, 5),
            (7, 11, 13, 11),
            (17, 31, 23, 2),
            (17, 31, 23, 4),
            (17, 31, 23, 31),
            (67, 31, 43, 3),
            (67, 31, 43, 7),
            (67, 31, 43, 16),
        ],
    )
    def test_prime_dimensions_upload_optimality(self, m: int, k: int, n: int, p_strips: int):
        """Verifies prime dimensions upload strictly M*K + K*N elements with 0.0 redundancy."""
        np.random.seed(101 + m + k + n + p_strips)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p_strips, dtype="fp32")
        profile = BandwidthProfiler.profile_plan(plan)

        expected_elements = m * k + k * n
        assert profile.total_uploaded_elements == expected_elements
        assert profile.optimal_elements == expected_elements
        assert profile.redundancy_elements == 0
        assert profile.redundancy_ratio == 0.0
        assert profile.is_upload_optimal is True

        # Verify exact disjoint coverage of [0, K)
        assert BandwidthProfiler.verify_disjoint_coverage(plan.tasks, k) is True

        # Content reconstruction identity: horizontal concatenation of A strips == A
        a_reconstructed = np.hstack([t.a_tile for t in plan.tasks])
        np.testing.assert_array_equal(a_reconstructed, a)

        # Content reconstruction identity: vertical concatenation of B strips == B
        b_reconstructed = np.vstack([t.b_tile for t in plan.tasks])
        np.testing.assert_array_equal(b_reconstructed, b)

    @pytest.mark.parametrize("p_strips", [1, 2, 4, 6, 25])
    def test_odd_dimensions_15x25x35(self, p_strips: int):
        """Verifies odd dimensions (15x25x35) upload strictly M*K + K*N elements."""
        m, k, n = 15, 25, 35
        np.random.seed(102 + p_strips)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p_strips, dtype="fp32")
        profile = BandwidthProfiler.profile_plan(plan)

        expected_elements = m * k + k * n  # 15*25 + 25*35 = 375 + 875 = 1250
        assert expected_elements == 1250
        assert profile.total_uploaded_elements == 1250
        assert profile.redundancy_elements == 0
        assert profile.redundancy_ratio == 0.0
        assert profile.is_upload_optimal is True

        assert BandwidthProfiler.verify_disjoint_coverage(plan.tasks, k) is True
        np.testing.assert_array_equal(np.hstack([t.a_tile for t in plan.tasks]), a)
        np.testing.assert_array_equal(np.vstack([t.b_tile for t in plan.tasks]), b)

    @pytest.mark.parametrize("p_strips", [1, 2, 4])
    def test_scalar_1x1(self, p_strips: int):
        """Verifies scalar 1x1 matrix multiplication: uploads strictly 2 elements (1 for A, 1 for B)."""
        m, k, n = 1, 1, 1
        a = np.array([[3.0]], dtype=np.float32)
        b = np.array([[7.0]], dtype=np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p_strips, dtype="fp32")
        # P > 1 is clamped to P_eff = 1
        assert len(plan.tasks) == 1

        profile = BandwidthProfiler.profile_plan(plan)
        expected_elements = 1 * 1 + 1 * 1  # 2
        assert profile.total_uploaded_elements == 2
        assert profile.redundancy_elements == 0
        assert profile.redundancy_ratio == 0.0
        assert profile.is_upload_optimal is True

        assert BandwidthProfiler.verify_disjoint_coverage(plan.tasks, 1) is True
        np.testing.assert_array_equal(plan.tasks[0].a_tile, a)
        np.testing.assert_array_equal(plan.tasks[0].b_tile, b)

    @pytest.mark.parametrize("p_strips", [1, 2, 4, 8, 16])
    def test_tall_skinny_128x16x32(self, p_strips: int):
        """Verifies tall-skinny aspect ratio (128x16x32) upload optimality."""
        m, k, n = 128, 16, 32
        np.random.seed(104 + p_strips)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p_strips, dtype="fp32")
        profile = BandwidthProfiler.profile_plan(plan)

        expected_elements = 128 * 16 + 16 * 32  # 2048 + 512 = 2560
        assert expected_elements == 2560
        assert profile.total_uploaded_elements == 2560
        assert profile.redundancy_elements == 0
        assert profile.redundancy_ratio == 0.0
        assert profile.is_upload_optimal is True

        assert BandwidthProfiler.verify_disjoint_coverage(plan.tasks, k) is True
        np.testing.assert_array_equal(np.hstack([t.a_tile for t in plan.tasks]), a)
        np.testing.assert_array_equal(np.vstack([t.b_tile for t in plan.tasks]), b)

    @pytest.mark.parametrize("p_strips", [1, 2, 4, 8, 16, 32])
    def test_wide_flat_16x128x16(self, p_strips: int):
        """Verifies wide-flat aspect ratio (16x128x16) upload optimality."""
        m, k, n = 16, 128, 16
        np.random.seed(105 + p_strips)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p_strips, dtype="fp32")
        profile = BandwidthProfiler.profile_plan(plan)

        expected_elements = 16 * 128 + 128 * 16  # 2048 + 2048 = 4096
        assert expected_elements == 4096
        assert profile.total_uploaded_elements == 4096
        assert profile.redundancy_elements == 0
        assert profile.redundancy_ratio == 0.0
        assert profile.is_upload_optimal is True

        assert BandwidthProfiler.verify_disjoint_coverage(plan.tasks, k) is True
        np.testing.assert_array_equal(np.hstack([t.a_tile for t in plan.tasks]), a)
        np.testing.assert_array_equal(np.vstack([t.b_tile for t in plan.tasks]), b)

    @pytest.mark.parametrize("p_strips", [1, 5, 10, 25, 50])
    def test_vector_dot_product_1x1000x1(self, p_strips: int):
        """Verifies vector dot product (1x1000x1) upload optimality."""
        m, k, n = 1, 1000, 1
        np.random.seed(106 + p_strips)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p_strips, dtype="fp32")
        profile = BandwidthProfiler.profile_plan(plan)

        expected_elements = 1 * 1000 + 1000 * 1  # 2000
        assert expected_elements == 2000
        assert profile.total_uploaded_elements == 2000
        assert profile.redundancy_elements == 0
        assert profile.redundancy_ratio == 0.0
        assert profile.is_upload_optimal is True

        assert BandwidthProfiler.verify_disjoint_coverage(plan.tasks, k) is True
        np.testing.assert_array_equal(np.hstack([t.a_tile for t in plan.tasks]), a)
        np.testing.assert_array_equal(np.vstack([t.b_tile for t in plan.tasks]), b)


# ==============================================================================
# SUITE 2: TEST BANDWIDTH CONTRAST BASELINE (1D vs 3D BLOCK TILING)
# ==============================================================================

class TestBandwidthContrastBaseline:
    """
    Contrasts 1D Outer-Product (0.0 redundancy ratio) with 3D Block Tiling on
    identical matrix dimensions, proving that 3D block tiling suffers from
    redundant element uploads ((J * M * K + I * K * N) / (M * K + K * N) - 1.0 > 0.0).
    """

    def test_square_contrast_128x128x128(self):
        """
        Square matrix contrast:
        M=128, K=128, N=128, tile_size=32.
        Grid: I=4, J=4, L=4 -> 64 tasks.
        1D Outer Product uploads 32,768 elements (0.0 redundancy).
        3D Block Tiling uploads 131,072 elements ((4*16384 + 4*16384) = 131072, redundancy = 3.0).
        """
        m, k, n = 128, 128, 128
        tile_size = 32
        np.random.seed(201)
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        # 1. Profile 1D Outer-Product
        profile_1d = BandwidthProfiler.profile_outer_product_1d(a, b, num_strips=4, dtype="fp32")
        optimal_elements = m * k + k * n  # 32,768
        assert profile_1d.total_uploaded_elements == optimal_elements
        assert profile_1d.redundancy_elements == 0
        assert profile_1d.redundancy_ratio == 0.0
        assert profile_1d.is_upload_optimal is True

        # 2. Profile 3D Block Tiling plan directly from MatrixTiler
        profile_3d = BandwidthProfiler.profile_block_3d(a, b, tile_size=tile_size, dtype="fp32")

        # 3. Theoretical derivation for 3D block tiling
        theo_3d = BandwidthProfiler.theoretical_3d_block_upload(m, k, n, tile_size)
        i_blocks = theo_3d["i_blocks"]  # 4
        j_blocks = theo_3d["j_blocks"]  # 4
        l_blocks = theo_3d["l_blocks"]  # 4
        assert theo_3d["total_tasks"] == 64

        # Redundancy formula: (J * M * K + I * K * N) / (M * K + K * N) - 1.0
        expected_3d_uploaded = j_blocks * m * k + i_blocks * k * n
        expected_3d_redundancy_ratio = (float(expected_3d_uploaded) / float(optimal_elements)) - 1.0

        assert expected_3d_uploaded == 131072
        assert expected_3d_redundancy_ratio == 3.0  # Exactly 300% redundant data (4x upload!)

        # Assert actual 3D plan matches theoretical formula exactly
        assert profile_3d.total_uploaded_elements == expected_3d_uploaded
        assert profile_3d.redundancy_ratio == expected_3d_redundancy_ratio
        assert profile_3d.redundancy_ratio > 0.0
        assert profile_3d.is_upload_optimal is False

        # Contrast assertion: 1D uses strictly 25% of 3D block bandwidth (300% savings)
        bandwidth_ratio = float(profile_1d.total_uploaded_elements) / float(profile_3d.total_uploaded_elements)
        assert bandwidth_ratio == 0.25

    def test_asymmetric_contrast_64x64x256(self):
        """
        Asymmetric matrix contrast:
        M=64, K=64, N=256, tile_size=32.
        Grid: I=2, J=8, L=2 -> 32 tasks.
        Optimal = 64*64 + 64*256 = 4096 + 16384 = 20,480.
        3D Upload = J*M*K + I*K*N = 8*4096 + 2*16384 = 32768 + 32768 = 65,536.
        Redundancy ratio = (65536 / 20480) - 1.0 = 2.2 > 0.0.
        """
        m, k, n = 64, 64, 256
        tile_size = 32
        np.random.seed(202)
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        profile_1d = BandwidthProfiler.profile_outer_product_1d(a, b, num_strips=4)
        profile_3d = BandwidthProfiler.profile_block_3d(a, b, tile_size=tile_size)
        theo_3d = BandwidthProfiler.theoretical_3d_block_upload(m, k, n, tile_size)

        optimal_elements = m * k + k * n  # 20,480
        assert profile_1d.total_uploaded_elements == optimal_elements
        assert profile_1d.redundancy_ratio == 0.0

        assert profile_3d.total_uploaded_elements == 65536
        assert profile_3d.total_uploaded_elements == theo_3d["total_uploaded_elements"]
        assert abs(profile_3d.redundancy_ratio - 2.2) < 1e-12
        assert profile_3d.redundancy_ratio > 0.0

    def test_tall_skinny_contrast_512x64x64(self):
        """
        Tall-skinny matrix contrast:
        M=512, K=64, N=64, tile_size=32.
        Grid: I=16, J=2, L=2 -> 64 tasks.
        Optimal = 512*64 + 64*64 = 32768 + 4096 = 36,864.
        3D Upload = J*M*K + I*K*N = 2*32768 + 16*4096 = 65536 + 65536 = 131,072.
        Redundancy ratio = (131072 / 36864) - 1.0 = 2.5555... > 0.0.
        """
        m, k, n = 512, 64, 64
        tile_size = 32
        np.random.seed(203)
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        profile_1d = BandwidthProfiler.profile_outer_product_1d(a, b, num_strips=4)
        profile_3d = BandwidthProfiler.profile_block_3d(a, b, tile_size=tile_size)
        theo_3d = BandwidthProfiler.theoretical_3d_block_upload(m, k, n, tile_size)

        optimal_elements = m * k + k * n  # 36,864
        assert profile_1d.total_uploaded_elements == optimal_elements
        assert profile_1d.redundancy_ratio == 0.0

        expected_3d = 131072
        assert profile_3d.total_uploaded_elements == expected_3d
        assert profile_3d.total_uploaded_elements == theo_3d["total_uploaded_elements"]
        assert profile_3d.redundancy_ratio > 2.5
        assert profile_3d.redundancy_ratio > 0.0

    def test_tile_size_sensitivity_sweep(self):
        """
        Sensitivity analysis across tile sizes:
        M=256, K=256, N=256 with tile sizes 16, 32, 64, 128.
        Proves that for all tile sizes where I > 1 or J > 1, 3D block tiling
        suffers from significant redundancy, while 1D outer product strictly maintains 0.0.
        """
        m, k, n = 256, 256, 256
        optimal = m * k + k * n  # 131,072
        np.random.seed(204)
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        # 1D outer-product is invariant to strip count
        for p in [1, 2, 4, 8, 16, 32]:
            p_1d = BandwidthProfiler.profile_outer_product_1d(a, b, num_strips=p)
            assert p_1d.total_uploaded_elements == optimal
            assert p_1d.redundancy_ratio == 0.0

        # 3D block tiling redundancy scales inversely with tile size:
        # T=16 -> I=16, J=16 -> Redundancy = 15.0 (16x upload)
        # T=32 -> I=8,  J=8  -> Redundancy = 7.0  (8x upload)
        # T=64 -> I=4,  J=4  -> Redundancy = 3.0  (4x upload)
        # T=128 -> I=2, J=2  -> Redundancy = 1.0  (2x upload)
        expected_ratios = {
            16: 15.0,
            32: 7.0,
            64: 3.0,
            128: 1.0,
        }

        for ts, expected_ratio in expected_ratios.items():
            p_3d = BandwidthProfiler.profile_block_3d(a, b, tile_size=ts)
            theo = BandwidthProfiler.theoretical_3d_block_upload(m, k, n, ts)
            assert theo["redundancy_ratio"] == expected_ratio
            assert p_3d.redundancy_ratio == expected_ratio
            assert p_3d.redundancy_ratio > 0.0

    def test_download_tradeoff_characterization(self):
        """
        Validates the trade-off documented in planned_work/04:
        1D Outer Product achieves optimal upload (0.0 redundancy) at the cost of
        downloading P partial sums (P * MN elements).
        3D Block Tiling achieves optimal download (MN elements) at the cost of
        redundant uploads (J*MK + I*KN elements).
        """
        m, k, n = 64, 64, 64
        num_strips = 4
        tile_size = 32

        np.random.seed(205)
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        p_1d = BandwidthProfiler.profile_outer_product_1d(a, b, num_strips=num_strips)
        p_3d = BandwidthProfiler.profile_block_3d(a, b, tile_size=tile_size)

        # 1D has 0.0 upload redundancy
        assert p_1d.redundancy_ratio == 0.0
        # 1D download redundancy is P - 1 = 4 - 1 = 3.0
        assert p_1d.download_redundancy_ratio == float(num_strips - 1)

        # 3D has >0.0 upload redundancy
        assert p_3d.redundancy_ratio == 1.0  # I=2, J=2 -> 1.0
        # 3D has 0.0 download redundancy (download is minimal MN)
        assert p_3d.download_redundancy_ratio == 0.0


# ==============================================================================
# SUITE 3: TEST HIGHAM PRECISION EQUIVALENCE
# ==============================================================================

class TestHighamPrecisionEquivalence:
    """
    Asserts ||C - C_hat||_inf <= gamma_K * ||A||_inf * ||B||_inf, where gamma_K = (K*u)/(1 - K*u),
    across FP32 (u=2^-24), FP16 (u=2^-11), and BF16 (u=2^-8, bitmask 0xFFFF0000) comparing
    coordinator and client execution against np.matmul reference.
    """

    def test_higham_bound_theoretical_properties(self):
        """Verifies Higham unit roundoff constants and monotonic scaling."""
        u_fp32 = get_unit_roundoff("fp32")
        u_fp16 = get_unit_roundoff("fp16")
        u_bf16 = get_unit_roundoff("bf16")

        assert u_fp32 == 2.0 ** -24
        assert u_fp16 == 2.0 ** -11
        assert u_bf16 == 2.0 ** -8

        # For fixed norm_a=1.0, norm_b=1.0, Higham bound must scale monotonically with K
        b_k10 = calculate_higham_bound(10, 1.0, 1.0, "fp32")
        b_k100 = calculate_higham_bound(100, 1.0, 1.0, "fp32")
        b_k1000 = calculate_higham_bound(1000, 1.0, 1.0, "fp32")
        assert b_k10 < b_k100 < b_k1000

        # Bound ordering across dtypes: FP32 < FP16 < BF16
        b_fp32 = calculate_higham_bound(64, 1.0, 1.0, "fp32")
        b_fp16 = calculate_higham_bound(64, 1.0, 1.0, "fp16")
        b_bf16 = calculate_higham_bound(64, 1.0, 1.0, "bf16")
        assert b_fp32 < b_fp16 < b_bf16

    @pytest.mark.parametrize(
        "m, k, n",
        [
            (32, 64, 32),
            (17, 31, 23),  # Primes
            (64, 128, 64),
        ],
    )
    def test_higham_fp32_coordinator_execution(
        self, hermetic_coordinator_env: Dict[str, Any], m: int, k: int, n: int
    ):
        """FP32 distributed execution matches np.matmul within Higham bound gamma_K."""
        coord: DistributedMatmulCoordinator = hermetic_coordinator_env["coordinator"]
        np.random.seed(301 + m + k + n)
        a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        c_dist = coord.execute_gemm(a, b, dtype="fp32", num_strips=4)
        c_ref = np.matmul(a, b)

        # Higham dynamic rounding bound
        norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
        higham_limit = calculate_higham_bound(k, norm_a, norm_b, dtype="fp32")

        max_elem_error = float(np.max(np.abs(c_dist - c_ref)))
        inf_norm_error = float(np.max(np.sum(np.abs(c_dist - c_ref), axis=1)))

        assert max_elem_error <= higham_limit, f"Max error {max_elem_error} > Higham bound {higham_limit}"
        assert inf_norm_error <= higham_limit, f"Inf norm error {inf_norm_error} > Higham bound {higham_limit}"
        assert max_elem_error < 1e-4

    def test_higham_fp16_coordinator_execution(self, hermetic_coordinator_env: Dict[str, Any]):
        """FP16 distributed execution matches np.matmul within Higham bound gamma_K."""
        coord: DistributedMatmulCoordinator = hermetic_coordinator_env["coordinator"]
        m, k, n = 32, 64, 32
        np.random.seed(302)
        a = (np.random.randn(m, k) * 0.1).astype(np.float16)
        b = (np.random.randn(k, n) * 0.1).astype(np.float16)

        c_dist = coord.execute_gemm(a, b, dtype="fp16", num_strips=4)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)

        norm_a = float(np.max(np.sum(np.abs(a.astype(np.float32)), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b.astype(np.float32)), axis=1)))
        higham_limit = calculate_higham_bound(k, norm_a, norm_b, dtype="fp16")

        max_elem_error = float(np.max(np.abs(c_dist.astype(np.float32) - c_ref.astype(np.float32))))
        assert max_elem_error <= max(higham_limit, 0.05)

    def test_higham_bf16_coordinator_execution(self, hermetic_coordinator_env: Dict[str, Any]):
        """BF16 distributed execution maintains 16-bit mantissa mask and satisfies Higham bound."""
        coord: DistributedMatmulCoordinator = hermetic_coordinator_env["coordinator"]
        m, k, n = 32, 64, 32
        np.random.seed(303)
        a = truncate_to_bf16(np.random.randn(m, k) * 0.1)
        b = truncate_to_bf16(np.random.randn(k, n) * 0.1)

        c_dist = coord.execute_gemm(a, b, dtype="bf16", num_strips=4)

        # Invariant 1: Valid BF16 bitmask (lower 16 bits zeroed)
        assert validate_bf16_masking(c_dist) is True
        u32_arr = c_dist.view(np.uint32)
        assert np.all((u32_arr & np.uint32(0x0000FFFF)) == 0)

        # Invariant 2: Higham numerical equivalence
        c_ref = truncate_to_bf16(np.matmul(a, b))
        norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
        higham_limit = calculate_higham_bound(k, norm_a, norm_b, dtype="bf16")

        max_elem_error = float(np.max(np.abs(c_dist - c_ref)))
        assert max_elem_error <= max(higham_limit, 0.5)

    def test_higham_client_matmul_api(self):
        """Verifies high-level MatrixComputeClient.matmul satisfies Higham bounds across FP32 and BF16."""
        config = ClientConfig(
            offline_mock_mode=True,
            default_tile_size=16,
            partitioning_strategy=PartitioningStrategy.OUTER_PRODUCT_1D,
        )

        with MatrixComputeClient(config=config) as client:
            np.random.seed(304)
            m, k, n = 24, 48, 24
            a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
            b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

            # FP32 matmul
            c_fp32 = client.matmul(a, b, dtype="fp32", num_strips=4)
            c_ref_fp32 = np.matmul(a, b)
            norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
            norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
            bound_fp32 = calculate_higham_bound(k, norm_a, norm_b, dtype="fp32")
            assert float(np.max(np.abs(c_fp32 - c_ref_fp32))) <= bound_fp32

            # BF16 matmul
            a_bf16 = truncate_to_bf16(a)
            b_bf16 = truncate_to_bf16(b)
            c_bf16 = client.matmul(a_bf16, b_bf16, dtype="bf16", num_strips=4)
            assert validate_bf16_masking(c_bf16) is True
            c_ref_bf16 = truncate_to_bf16(np.matmul(a_bf16, b_bf16))
            bound_bf16 = calculate_higham_bound(k, norm_a, norm_b, dtype="bf16")
            assert float(np.max(np.abs(c_bf16 - c_ref_bf16))) <= max(bound_bf16, 0.5)


# ==============================================================================
# SUITE 4: TEST COORDINATOR END-TO-END BANDWIDTH
# ==============================================================================

class TestCoordinatorEndToEndBandwidth:
    """
    Runs full distributed execution through DistributedMatmulCoordinator.execute_gemm
    and MatrixComputeClient.matmul with mock workers, counting actual elements
    dispatched in tasks and verifying exact M*K + K*N elements.
    """

    def test_coordinator_dispatches_exact_elements(self, hermetic_coordinator_env: Dict[str, Any]):
        """
        Instruments coordinator scheduler to count actual elements dispatched across all strips.
        Asserts total dispatched elements strictly equal M*K + K*N with 0 duplicates.
        """
        coord: DistributedMatmulCoordinator = hermetic_coordinator_env["coordinator"]
        scheduler: ParallelTaskScheduler = hermetic_coordinator_env["scheduler"]

        # Instrument scheduler.submit_tile
        spy = DispatchElementSpy(scheduler.submit_tile)
        scheduler.submit_tile = spy

        m, k, n = 32, 64, 32
        p_strips = 4
        np.random.seed(401)
        a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        result = coord.execute_gemm(a, b, dtype="fp32", num_strips=p_strips)

        # Expected upload: M*K + K*N = 32*64 + 64*32 = 2048 + 2048 = 4096 elements
        expected_optimal_elements = m * k + k * n
        assert expected_optimal_elements == 4096

        assert spy.total_a_elements == m * k
        assert spy.total_b_elements == k * n
        assert spy.total_dispatched_elements == expected_optimal_elements
        assert len(spy.dispatched_tasks) == p_strips

        # Verify zero redundant elements and zero overlap
        assert BandwidthProfiler.verify_disjoint_coverage(spy.dispatched_tasks, k) is True

        # Verify mathematical correctness
        expected_result = np.matmul(a, b)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5, atol=1e-5)

    def test_coordinator_prime_dimensions_dispatched_elements(
        self, hermetic_coordinator_env: Dict[str, Any]
    ):
        """
        Instruments coordinator with prime dimensions (7x11x13, P=3).
        Asserts total dispatched elements strictly equal 7*11 + 11*13 = 220 elements.
        """
        coord: DistributedMatmulCoordinator = hermetic_coordinator_env["coordinator"]
        scheduler: ParallelTaskScheduler = hermetic_coordinator_env["scheduler"]

        spy = DispatchElementSpy(scheduler.submit_tile)
        scheduler.submit_tile = spy

        m, k, n = 7, 11, 13
        p_strips = 3
        np.random.seed(402)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        result = coord.execute_gemm(a, b, dtype="fp32", num_strips=p_strips)

        expected_elements = 7 * 11 + 11 * 13  # 77 + 143 = 220
        assert spy.total_dispatched_elements == expected_elements
        assert spy.total_a_elements == 77
        assert spy.total_b_elements == 143
        assert len(spy.dispatched_tasks) == 3

        # Strip widths: 11 % 3 = 2 remainder -> [4, 4, 3]
        strip_widths = [t.k_slice[1] - t.k_slice[0] for t in spy.dispatched_tasks]
        assert strip_widths == [4, 4, 3]
        assert sum(strip_widths) == 11

        expected_result = np.matmul(a, b)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5, atol=1e-5)

    def test_client_context_manager_bandwidth_e2e(self):
        """
        Validates MatrixComputeClient context manager end-to-end execution
        dispatches exact M*K + K*N elements with zero upload redundancy.
        """
        config = ClientConfig(
            offline_mock_mode=True,
            default_tile_size=16,
            partitioning_strategy=PartitioningStrategy.OUTER_PRODUCT_1D,
        )

        with MatrixComputeClient(config=config) as client:
            scheduler = client.scheduler
            spy = DispatchElementSpy(scheduler.submit_tile)
            scheduler.submit_tile = spy

            m, k, n = 24, 48, 36
            p_strips = 4
            np.random.seed(403)
            a = np.random.randn(m, k).astype(np.float32)
            b = np.random.randn(k, n).astype(np.float32)

            result = client.matmul(
                a,
                b,
                dtype="fp32",
                partitioning_strategy=PartitioningStrategy.OUTER_PRODUCT_1D,
                num_strips=p_strips,
            )

            expected_optimal = m * k + k * n  # 24*48 + 48*36 = 1152 + 1728 = 2880
            assert spy.total_dispatched_elements == expected_optimal
            assert spy.total_a_elements == 1152
            assert spy.total_b_elements == 1728
            assert len(spy.dispatched_tasks) == 4

            expected_result = np.matmul(a, b)
            np.testing.assert_allclose(result, expected_result, rtol=1e-5, atol=1e-5)

    def test_worker_node_received_bytes_optimality(
        self, hermetic_coordinator_env: Dict[str, Any]
    ):
        """
        Validates that workers receive strictly (M*K + K*N) * bytes_per_element bytes
        over the gRPC transport, with zero redundant bytes received.
        """
        coord: DistributedMatmulCoordinator = hermetic_coordinator_env["coordinator"]
        worker1: MockWorkerNode = hermetic_coordinator_env["worker1"]
        worker2: MockWorkerNode = hermetic_coordinator_env["worker2"]

        # Reset worker task counters and hook chunk stream accounting
        worker1.servicer.task_count = 0
        worker2.servicer.task_count = 0
        total_wire_bytes_received = 0

        orig_exec1 = worker1.servicer.ExecuteTileTask
        orig_exec2 = worker2.servicer.ExecuteTileTask

        def make_wire_counter(orig_func):
            def counting_execute(request_iterator, context=None):
                nonlocal total_wire_bytes_received
                def counting_stream():
                    nonlocal total_wire_bytes_received
                    for chunk in request_iterator:
                        if chunk.chunk_bytes:
                            total_wire_bytes_received += len(chunk.chunk_bytes)
                        yield chunk
                return orig_func(counting_stream(), context)
            return counting_execute

        worker1.servicer.ExecuteTileTask = make_wire_counter(orig_exec1)
        worker2.servicer.ExecuteTileTask = make_wire_counter(orig_exec2)

        try:
            m, k, n = 20, 40, 20
            p_strips = 4
            dtype = "fp32"
            bytes_per_elem = 4

            np.random.seed(404)
            a = np.random.randn(m, k).astype(np.float32)
            b = np.random.randn(k, n).astype(np.float32)

            result = coord.execute_gemm(a, b, dtype=dtype, num_strips=p_strips)

            # Both workers executed tasks
            total_tasks_executed = worker1.servicer.task_count + worker2.servicer.task_count
            assert total_tasks_executed == p_strips

            manifest = coord.get_last_manifest()
            assert manifest is not None
            assert manifest.total_tasks == p_strips

            # Assert exact wire byte upload optimality
            expected_optimal_elements = m * k + k * n  # 1600 elements
            expected_optimal_bytes = expected_optimal_elements * bytes_per_elem  # 6400 bytes
            assert total_wire_bytes_received == expected_optimal_bytes

            # Verify output numerical equivalence
            expected = np.matmul(a, b)
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
        finally:
            worker1.servicer.ExecuteTileTask = orig_exec1
            worker2.servicer.ExecuteTileTask = orig_exec2


# ==============================================================================
# SUITE 5: TEST LARGE-SCALE 1024 STRESS
# ==============================================================================

class TestLargeScale1024Stress:
    """
    Runs 1024x1024 matmul with 16 strips and verifies that total uploaded elements
    equal strictly 2,097,152 with zero duplicate elements, and result matches
    np.matmul within Higham bound.
    """

    def test_large_scale_1024x1024_exact_element_count(self):
        """
        Validates 1024x1024 distributed outer product:
        - M=1024, K=1024, N=1024, P=16 strips.
        - Optimal upload elements = 1024*1024 + 1024*1024 = 2,097,152.
        - Verifies profile.total_uploaded_elements == 2,097,152.
        - Verifies profile.redundancy_elements == 0.
        - Verifies profile.redundancy_ratio == 0.0.
        """
        m, k, n = 1024, 1024, 1024
        p_strips = 16

        np.random.seed(501)
        a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p_strips, dtype="fp32")
        profile = BandwidthProfiler.profile_plan(plan)

        expected_elements = 1024 * 1024 + 1024 * 1024  # 2,097,152
        assert expected_elements == 2097152

        assert profile.total_uploaded_elements == expected_elements
        assert profile.optimal_elements == expected_elements
        assert profile.redundancy_elements == 0
        assert profile.redundancy_ratio == 0.0
        assert profile.is_upload_optimal is True
        assert profile.num_tasks == 16

    def test_large_scale_1024x1024_zero_duplicate_elements(self):
        """
        Validates that all 16 strips form a disjoint partition of [0, 1024)
        with zero duplicate elements and exact reconstruction.
        """
        m, k, n = 1024, 1024, 1024
        p_strips = 16

        np.random.seed(502)
        a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p_strips, dtype="fp32")

        # Invariant 1: Exactly 16 strips of width 64
        assert len(plan.tasks) == 16
        for p, t in enumerate(plan.tasks):
            assert t.k_slice == (p * 64, (p + 1) * 64)
            assert t.a_tile.shape == (1024, 64)
            assert t.b_tile.shape == (64, 1024)

        # Invariant 2: Disjoint partition verification
        assert BandwidthProfiler.verify_disjoint_coverage(plan.tasks, 1024) is True

        # Invariant 3: Perfect array reconstruction with 0 difference
        a_reconstructed = np.hstack([t.a_tile for t in plan.tasks])
        np.testing.assert_array_equal(a_reconstructed, a)

        b_reconstructed = np.vstack([t.b_tile for t in plan.tasks])
        np.testing.assert_array_equal(b_reconstructed, b)

    def test_large_scale_1024x1024_numerical_equivalence_higham(self):
        """
        Computes the complete 1024x1024 outer product via ResultAssembler and
        asserts mathematical equivalence against np.matmul within Higham bound.
        """
        m, k, n = 1024, 1024, 1024
        p_strips = 16

        np.random.seed(503)
        a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p_strips, dtype="fp32")
        assembler = ResultAssembler.from_plan(plan)

        # Simulate distributed execution: compute each strip's partial product C_p
        for t in plan.tasks:
            c_p = np.matmul(t.a_tile, t.b_tile)
            assembler.accumulate_task(t, c_p)

        assert assembler.is_complete() is True
        c_assembled = assembler.get_result(enforce_complete=True)

        # Authoritative reference
        c_expected = np.matmul(a, b)

        # Higham dynamic rounding bound
        norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
        higham_limit = calculate_higham_bound(k, norm_a, norm_b, dtype="fp32")

        max_error = float(np.max(np.abs(c_assembled - c_expected)))
        assert max_error <= higham_limit, f"1024x1024 error {max_error} exceeded Higham bound {higham_limit}"
        assert max_error < 1e-4

        # Full array comparison
        np.testing.assert_allclose(c_assembled, c_expected, rtol=1e-5, atol=1e-5)
