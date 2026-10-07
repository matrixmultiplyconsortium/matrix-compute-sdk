"""
Comprehensive Unit Test Suite for 2D Output Grid (Download-Optimal & Zero-Reduction) Matrix Partitioning.

Milestone 1 Test Specification for matrix-compute-sdk.
Validates:
1. 2D spatial grid interval calculation: balanced div-mod remainder distribution and boundary clamping.
2. Two-sided boundary clamping: P_r > M, P_c > N (oversubscription), and single-tile grids.
3. Plan and task geometry: TileGridPlan metadata (p_r, p_c, k_blocks, grid_shape), TileTaskPlan attributes, and dict access.
4. Unsliced inner dimension invariant: K is strictly unsliced (num_k_blocks == 1, k_slice == (0, K)).
5. Zero-reduction spatial assembly: direct coordinate placement with reduction_ops == 0.
6. Multi-precision preservation & IEEE bitmasking across FP32, FP16, and BF16.
7. Bandwidth profiler metrics: strictly M*N downloaded elements, download redundancy ratio == 0.0, is_download_optimal flag.
8. Spatial grid coverage verification: exact disjoint cover of [0, M) x [0, N) with zero gaps and zero overlaps.
9. Mathematical equivalence & Higham error bounds against reference GEMM.
10. Shape variation stress tests: prime, odd, scalar (1x1), and asymmetric (M != N != K) dimensions.
11. Defensive validation: dimension mismatches, empty matrices, non-2D arrays, invalid grid parameters.
"""

from __future__ import annotations

import concurrent.futures
import math
import random
import threading
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from matrix_compute.config import ClientConfig, PartitioningStrategy
from matrix_compute.exceptions import MatrixDimensionError, TilingError
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


# =============================================================================
# HELPER ORACLES & VALIDATORS
# =============================================================================

def assert_2d_plan_invariants(
    plan: TileGridPlan,
    a: np.ndarray,
    b: np.ndarray,
    requested_pr: int,
    requested_pc: int,
    expected_dtype: str = "fp32",
    expected_prefix: str = "tile",
) -> None:
    """
    Comprehensive oracle verifying all structural and geometric invariants of a 2D TileGridPlan.
    """
    m, k = a.shape
    _, n = b.shape
    expected_pr = min(max(1, int(requested_pr)), m)
    expected_pc = min(max(1, int(requested_pc)), n)
    expected_total_tasks = expected_pr * expected_pc

    # 1. Grid Plan Metadata
    assert isinstance(plan, TileGridPlan), f"Expected TileGridPlan, got {type(plan)}"
    assert plan.m == m, f"Plan M ({plan.m}) != a.shape[0] ({m})"
    assert plan.k == k, f"Plan K ({plan.k}) != a.shape[1] ({k})"
    assert plan.n == n, f"Plan N ({plan.n}) != b.shape[1] ({n})"
    assert plan.num_m_blocks == expected_pr, f"num_m_blocks ({plan.num_m_blocks}) != {expected_pr}"
    assert plan.num_n_blocks == expected_pc, f"num_n_blocks ({plan.num_n_blocks}) != {expected_pc}"
    assert plan.num_k_blocks == 1, f"num_k_blocks must be strictly 1, got {plan.num_k_blocks}"
    assert plan.p_r == expected_pr, f"p_r property ({plan.p_r}) != {expected_pr}"
    assert plan.p_c == expected_pc, f"p_c property ({plan.p_c}) != {expected_pc}"
    assert plan.k_blocks == 1, f"k_blocks property ({plan.k_blocks}) != 1"
    assert plan.grid_shape == (expected_pr, expected_pc), f"grid_shape != ({expected_pr}, {expected_pc})"
    assert plan.total_tasks == expected_total_tasks, f"total_tasks != {expected_total_tasks}"
    assert len(plan.tasks) == expected_total_tasks, f"len(plan.tasks) != {expected_total_tasks}"
    assert plan.strategy == "output_grid_2d"
    assert plan.dtype == normalize_dtype(expected_dtype)

    # 2. Total GFLOPs Conservation: 2 * M * K * N / 10^9
    expected_total_gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9
    assert math.isclose(plan.total_gflops, expected_total_gflops, rel_tol=1e-7), (
        f"Plan total_gflops {plan.total_gflops} != expected {expected_total_gflops}"
    )
    task_gflops_sum = sum(t.estimated_gflops for t in plan.tasks)
    assert math.isclose(task_gflops_sum, expected_total_gflops, rel_tol=1e-7), (
        f"Sum of task GFLOPs {task_gflops_sum} != {expected_total_gflops}"
    )

    # 3. Disjoint Coverage of [0, M) x [0, N)
    assert BandwidthProfiler.verify_2d_grid_coverage(plan.tasks, m=m, n=n) is True

    # 4. Individual Task Properties
    for task in plan.tasks:
        assert isinstance(task, TileTaskPlan)
        assert task.k_idx == 0, f"Task {task.task_id} k_idx must be 0"
        assert task.k_slice == (0, k), f"Task {task.task_id} k_slice must be (0, {k})"
        assert 0 <= task.row_idx < expected_pr
        assert 0 <= task.col_idx < expected_pc
        assert task.a_tile.shape == (task.m_i, k)
        assert task.b_tile.shape == (k, task.n_k)
        assert task.shape_c == (task.m_i, task.n_k)


# =============================================================================
# SUITE 1: 2D GRID INTERVAL CALCULATION & REMAINDER DISTRIBUTION
# =============================================================================

class TestSuite1GridIntervalCalculation:
    """Tests for compute_2d_grid_intervals algorithm."""

    def test_even_division(self):
        """Even division produces uniform block sizes."""
        intervals = compute_2d_grid_intervals(dimension=64, num_blocks=4)
        assert len(intervals) == 4
        assert intervals == [(0, 16), (16, 32), (32, 48), (48, 64)]
        assert all(end - start == 16 for start, end in intervals)

    def test_balanced_remainder_distribution(self):
        """Remainder elements are distributed 1-by-1 to the first remainder intervals."""
        # 10 / 3 -> q=3, r=1 -> [4, 3, 3]
        intervals_m = compute_2d_grid_intervals(dimension=10, num_blocks=3)
        assert len(intervals_m) == 3
        lengths_m = [end - start for start, end in intervals_m]
        assert lengths_m == [4, 3, 3]
        assert intervals_m == [(0, 4), (4, 7), (7, 10)]

        # 11 / 3 -> q=3, r=2 -> [4, 4, 3]
        intervals_n = compute_2d_grid_intervals(dimension=11, num_blocks=3)
        assert len(intervals_n) == 3
        lengths_n = [end - start for start, end in intervals_n]
        assert lengths_n == [4, 4, 3]
        assert intervals_n == [(0, 4), (4, 8), (8, 11)]

    def test_clamping_oversubscription(self):
        """Requesting more blocks than dimension clamps to dimension with length 1 each."""
        # Request 10 blocks for dimension 3 -> clamped to 3 blocks of length 1
        intervals = compute_2d_grid_intervals(dimension=3, num_blocks=10)
        assert len(intervals) == 3
        assert intervals == [(0, 1), (1, 2), (2, 3)]

    def test_clamping_zero_or_negative_blocks(self):
        """Requesting 0 or negative blocks clamps to 1 block."""
        for bad_p in (0, -1, -10):
            intervals = compute_2d_grid_intervals(dimension=16, num_blocks=bad_p)
            assert len(intervals) == 1
            assert intervals == [(0, 16)]

    def test_single_block_partition(self):
        """Requesting 1 block returns single interval [0, dimension)."""
        intervals = compute_2d_grid_intervals(dimension=42, num_blocks=1)
        assert intervals == [(0, 42)]

    def test_invalid_dimension_raises_value_error(self):
        """Dimension < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Dimension must be >= 1"):
            compute_2d_grid_intervals(dimension=0, num_blocks=2)
        with pytest.raises(ValueError, match="Dimension must be >= 1"):
            compute_2d_grid_intervals(dimension=-5, num_blocks=2)

    def test_alias_function_parity(self):
        """compute_grid_intervals alias returns identical results to compute_2d_grid_intervals."""
        assert compute_grid_intervals(50, 4) == compute_2d_grid_intervals(50, 4)


# =============================================================================
# SUITE 2: PLAN CREATION, TASK GEOMETRY & PROPERTIES
# =============================================================================

class TestSuite2PlanCreationAndGeometry:
    """Tests for MatrixTiler.create_plan_output_grid_2d and plan metadata."""

    def test_square_matrix_square_grid(self):
        """Standard square matrix with square grid."""
        a = np.ones((64, 32), dtype=np.float32)
        b = np.ones((32, 64), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        assert_2d_plan_invariants(plan, a, b, requested_pr=2, requested_pc=2)
        assert plan.grid_shape == (2, 2)
        assert plan.total_tasks == 4

    def test_asymmetric_non_square_grid(self):
        """Asymmetric matrix with asymmetric grid (P_r != P_c)."""
        a = np.ones((30, 40), dtype=np.float32)
        b = np.ones((40, 50), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=3, grid_cols=5)
        assert_2d_plan_invariants(plan, a, b, requested_pr=3, requested_pc=5)
        assert plan.p_r == 3
        assert plan.p_c == 5
        assert plan.k_blocks == 1
        assert plan.total_tasks == 15

    def test_grid_shape_parameter_alias(self):
        """grid_shape=(P_r, P_c) tuple resolves correctly."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_shape=(2, 4))
        assert plan.p_r == 2
        assert plan.p_c == 4
        assert plan.grid_shape == (2, 4)

    def test_tile_size_fallback_heuristic(self):
        """When grid_rows/cols omitted, derives P_r and P_c from tile_size."""
        a = np.ones((64, 32), dtype=np.float32)
        b = np.ones((32, 48), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, tile_size=16)
        # 64 / 16 = 4 rows, 48 / 16 = 3 cols -> 12 tasks
        assert plan.p_r == 4
        assert plan.p_c == 3
        assert plan.total_tasks == 12

    def test_tile_task_plan_dict_emulation(self):
        """TileTaskPlan and TileGridPlan support dictionary subscription and keys."""
        a = np.ones((10, 20), dtype=np.float32)
        b = np.ones((20, 30), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=3)

        assert plan["m"] == 10
        assert plan["k"] == 20
        assert plan["n"] == 30
        assert plan["strategy"] == "output_grid_2d"
        assert plan["p_r"] == 2
        assert plan["p_c"] == 3
        assert plan["grid_shape"] == (2, 3)

        task = plan.tasks[0]
        assert task["row_idx"] == 0
        assert task["col_idx"] == 0
        assert task["k_idx"] == 0
        assert "a_tile" in task
        assert "b_tile" in task

    def test_partition_output_grid_2d_returns_tasks(self):
        """partition_output_grid_2d returns flat list of TileTaskPlans."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        tasks = MatrixTiler.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        assert isinstance(tasks, list)
        assert len(tasks) == 4
        assert all(isinstance(t, TileTaskPlan) for t in tasks)

    def test_create_plan_routing_output_grid_2d(self):
        """MatrixTiler.create_plan routes strategy='output_grid_2d' and aliases."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        for strat in ("output_grid_2d", "output_grid", "2d_output_grid", "grid_2d", "2d"):
            plan = MatrixTiler.create_plan(a, b, strategy=strat, grid_rows=2, grid_cols=2)
            assert plan.strategy == "output_grid_2d"
            assert plan.num_k_blocks == 1
            assert plan.total_tasks == 4


# =============================================================================
# SUITE 3: ZERO-REDUCTION SPATIAL ASSEMBLY
# =============================================================================

class TestSuite3ZeroReductionAssembly:
    """Tests verifying direct spatial placement and reduction_ops == 0."""

    def test_direct_placement_zero_reduction_ops(self):
        """Assembling 2D output grid tasks executes strictly 0 reduction additions."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=2)

        assembler = ResultAssembler.from_plan(plan)
        assert assembler.zero_reduction is True
        assert assembler.reduction_ops == 0

        # Accumulate all tasks with computed dot products
        for task in plan.tasks:
            tile_res = np.matmul(task.a_tile, task.b_tile)
            assembler.accumulate_task(task, tile_res)

        assert assembler.is_complete() is True
        # STRICT REQUIREMENT: zero floating-point reduction additions!
        assert assembler.reduction_ops == 0

        result = assembler.get_result(enforce_complete=True)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(result, expected)

    def test_assembler_from_plan_auto_detects_zero_reduction(self):
        """from_plan auto-detects zero_reduction=True when num_k_blocks == 1."""
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        plan_2d = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        assembler = ResultAssembler.from_plan(plan_2d)
        assert assembler.zero_reduction is True

        plan_3d = MatrixTiler.create_plan(a, b, tile_size=4, strategy="block_3d")
        assembler_3d = ResultAssembler.from_plan(plan_3d)
        assert assembler_3d.zero_reduction is False

    def test_reentrancy_duplicate_rejection(self):
        """ResultAssembler rejects duplicate arrival of the same coordinate."""
        assembler = ResultAssembler(m=4, n=4, k_blocks=1, zero_reduction=True)
        tile = np.ones((2, 2), dtype=np.float32)
        assembler.accumulate(tile, row_idx=0, col_idx=0, k_idx=0, row_slice=(0, 2), col_slice=(0, 2))

        with pytest.raises(TilingError, match="Duplicate accumulation rejected"):
            assembler.accumulate(tile, row_idx=0, col_idx=0, k_idx=0, row_slice=(0, 2), col_slice=(0, 2))

    def test_out_of_bounds_slice_rejection(self):
        """ResultAssembler rejects out-of-bounds row or column slices."""
        assembler = ResultAssembler(m=4, n=4, zero_reduction=True)
        tile = np.ones((2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="out of bounds"):
            assembler.accumulate(tile, row_idx=0, col_idx=0, row_slice=(0, 5), col_slice=(0, 2))

    def test_reset_clears_reduction_ops(self):
        """reset() resets reduction_ops and buffers."""
        assembler = ResultAssembler(m=4, n=4, zero_reduction=False)
        tile = np.ones((2, 2), dtype=np.float32)
        assembler.accumulate(tile, row_idx=0, col_idx=0, row_slice=(0, 2), col_slice=(0, 2))
        assert assembler.reduction_ops > 0
        assembler.reset()
        assert assembler.reduction_ops == 0

    def test_multithreaded_concurrent_assembly(self):
        """Concurrent worker threads assemble disjoint slices safely without race conditions."""
        m, n = 32, 32
        plan = MatrixTiler.create_plan_output_grid_2d(
            np.ones((m, 16), dtype=np.float32),
            np.ones((16, n), dtype=np.float32),
            grid_rows=4,
            grid_cols=4,
        )
        assembler = ResultAssembler.from_plan(plan)

        def worker_task(t: TileTaskPlan):
            res = np.matmul(t.a_tile, t.b_tile)
            assembler.accumulate_task(t, res)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(worker_task, plan.tasks))

        assert assembler.is_complete() is True
        assert assembler.reduction_ops == 0
        final_mat = assembler.get_result(enforce_complete=True)
        assert final_mat.shape == (m, n)
        np.testing.assert_allclose(final_mat, 16.0 * np.ones((m, n)))


# =============================================================================
# SUITE 4: BANDWIDTH PROFILING & DOWNLOAD OPTIMALITY
# =============================================================================

class TestSuite4BandwidthProfiling:
    """Tests verifying strictly M*N download elements and download redundancy ratio == 0.0."""

    def test_download_bandwidth_optimality_strict_mn(self):
        """Downloaded elements strictly equal M*N with zero redundancy."""
        shapes = [
            (32, 16, 32, 2, 2),
            (64, 32, 128, 4, 8),
            (7, 11, 13, 2, 3),   # Prime dimensions
            (15, 25, 35, 3, 4),  # Odd dimensions
        ]
        for m, k, n, pr, pc in shapes:
            a = np.ones((m, k), dtype=np.float32)
            b = np.ones((k, n), dtype=np.float32)
            profile = BandwidthProfiler.profile_output_grid_2d(a, b, grid_rows=pr, grid_cols=pc)

            # STRICT INVARIANTS:
            assert profile.download_elements == m * n
            assert profile.optimal_download_elements == m * n
            assert profile.download_redundancy_ratio == 0.0
            assert profile.is_download_optimal is True

    def test_upload_volume_formula(self):
        """Uploaded elements equal Pc * M * K + Pr * K * N."""
        m, k, n, pr, pc = 32, 16, 48, 2, 3
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)
        profile = BandwidthProfiler.profile_output_grid_2d(a, b, grid_rows=pr, grid_cols=pc)

        expected_upload = pc * (m * k) + pr * (k * n)
        assert profile.total_uploaded_elements == expected_upload
        assert profile.total_a_elements == pc * (m * k)
        assert profile.total_b_elements == pr * (k * n)

    def test_theoretical_2d_output_grid_bandwidth(self):
        """theoretical_2d_output_grid_bandwidth calculates correct theoretical profile."""
        theo = BandwidthProfiler.theoretical_2d_output_grid_bandwidth(m=64, n=64, k=32, p_r=4, p_c=4)
        assert theo.download_elements == 64 * 64
        assert theo.download_redundancy_ratio == 0.0
        assert theo.is_download_optimal is True
        # On square grid: 4 * (64 * 32) + 4 * (32 * 64) = 4 * (2048 + 2048) = 16384
        assert theo.total_uploaded_elements == 16384

    def test_verify_2d_grid_coverage_valid_and_invalid(self):
        """verify_2d_grid_coverage correctly checks contiguous disjoint partition."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=2)

        # Valid plan passes
        assert BandwidthProfiler.verify_2d_grid_coverage(plan.tasks, m=16, n=16) is True

        # Incomplete task list fails
        incomplete = plan.tasks[:-1]
        assert BandwidthProfiler.verify_2d_grid_coverage(incomplete, m=16, n=16) is False

        # Wrong matrix dimensions fail
        assert BandwidthProfiler.verify_2d_grid_coverage(plan.tasks, m=32, n=16) is False


# =============================================================================
# SUITE 5: MULTI-PRECISION PRESERVATION & HIGHAM EQUIVALENCE
# =============================================================================

class TestSuite5PrecisionAndMathematicalEquivalence:
    """Tests verifying FP32, FP16, and BF16 precision integrity and Higham error bounds."""

    def test_fp32_higham_equivalence(self):
        """FP32 output grid multiplication matches reference within Higham bound."""
        np.random.seed(42)
        m, k, n = 32, 64, 32
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=2, dtype="fp32")
        assembler = ResultAssembler.from_plan(plan)
        for t in plan.tasks:
            assembler.accumulate_task(t, np.matmul(t.a_tile, t.b_tile))

        res = assembler.get_result(enforce_complete=True)
        ref = emulate_gemm(a, b, dtype="fp32")
        bound = calculate_higham_bound(k=k, a=a, b=b, dtype="fp32")
        diff = float(np.max(np.abs(res - ref)))
        assert diff <= max(bound, 1e-5)

    def test_fp16_higham_equivalence(self):
        """FP16 output grid multiplication matches reference within FP16 Higham bound."""
        np.random.seed(42)
        m, k, n = 16, 32, 16
        a = (np.random.randn(m, k) * 0.1).astype(np.float16)
        b = (np.random.randn(k, n) * 0.1).astype(np.float16)

        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=2, dtype="fp16")
        assembler = ResultAssembler.from_plan(plan)
        for t in plan.tasks:
            assembler.accumulate_task(t, np.matmul(t.a_tile.astype(np.float32), t.b_tile.astype(np.float32)).astype(np.float16))

        res = assembler.get_result(enforce_complete=True)
        assert res.dtype == np.float16
        ref = emulate_gemm(a, b, dtype="fp16")
        bound = calculate_higham_bound(k=k, a=a, b=b, dtype="fp16")
        diff = float(np.max(np.abs(res.astype(np.float32) - ref.astype(np.float32))))
        assert diff <= max(bound, 1e-2)

    def test_bf16_masking_and_nan_inf_preservation(self):
        """BF16 preserves 16-bit masked mantissa (0xFFFF0000) and special values."""
        a = np.array([[1.5, -2.25], [float("inf"), float("nan")]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        a_bf16 = truncate_to_bf16(a)
        validate_bf16_masking(a_bf16)

        plan = MatrixTiler.create_plan_output_grid_2d(a_bf16, b, grid_rows=1, grid_cols=1, dtype="bf16")
        assert plan.dtype == "bf16"


# =============================================================================
# SUITE 6: BOUNDARY SHAPES & CORNER CASES
# =============================================================================

class TestSuite6BoundaryShapesAndCornerCases:
    """Tests verifying prime, odd, scalar, and asymmetric dimension handling."""

    def test_prime_dimensions_7x11x13(self):
        """Prime dimensions M=7, K=11, N=13 with grid 2x3."""
        a = np.ones((7, 11), dtype=np.float32)
        b = np.ones((11, 13), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=3)
        assert_2d_plan_invariants(plan, a, b, requested_pr=2, requested_pc=3)

    def test_scalar_1x1_matrix(self):
        """Scalar 1x1 matrix multiplication with grid 2x2 clamped to 1x1."""
        a = np.array([[3.0]], dtype=np.float32)
        b = np.array([[7.0]], dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        assert plan.p_r == 1
        assert plan.p_c == 1
        assert plan.total_tasks == 1
        res = ResultAssembler.assemble_tiles(1, 1, plan.tasks)
        np.testing.assert_allclose(res, [[21.0]])

    def test_matrix_vector_and_vector_matrix(self):
        """Matrix-vector (M x K x 1) and vector-matrix (1 x K x N)."""
        # Matrix-vector: N=1 -> P_c clamped to 1
        a_mv = np.ones((32, 16), dtype=np.float32)
        b_mv = np.ones((16, 1), dtype=np.float32)
        plan_mv = MatrixTiler.create_plan_output_grid_2d(a_mv, b_mv, grid_rows=4, grid_cols=4)
        assert plan_mv.p_r == 4
        assert plan_mv.p_c == 1
        assert plan_mv.total_tasks == 4

        # Vector-matrix: M=1 -> P_r clamped to 1
        a_vm = np.ones((1, 16), dtype=np.float32)
        b_vm = np.ones((16, 32), dtype=np.float32)
        plan_vm = MatrixTiler.create_plan_output_grid_2d(a_vm, b_vm, grid_rows=4, grid_cols=4)
        assert plan_vm.p_r == 1
        assert plan_vm.p_c == 4
        assert plan_vm.total_tasks == 4

    def test_extreme_contraction_k_unsliced(self):
        """Extreme inner dimension K=5000 is strictly unsliced."""
        a = np.ones((2, 5000), dtype=np.float32)
        b = np.ones((5000, 2), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        assert plan.num_k_blocks == 1
        assert plan.k_blocks == 1
        for t in plan.tasks:
            assert t.k_slice == (0, 5000)


# =============================================================================
# SUITE 7: CONFIGURATION & DEFENSIVE VALIDATION
# =============================================================================

class TestSuite7ConfigAndDefensiveValidation:
    """Tests for ClientConfig grid settings and defensive rejection."""

    def test_client_config_grid_fields_and_aliases(self):
        """ClientConfig parses grid_rows, grid_cols, grid_shape, and aliases."""
        cfg = ClientConfig(grid_rows=3, grid_cols=4)
        assert cfg.grid_rows == 3
        assert cfg.grid_cols == 4
        assert cfg.grid_shape == (3, 4)
        assert cfg.p_r == 3
        assert cfg.p_c == 4

        cfg2 = ClientConfig(grid_shape=(2, 5))
        assert cfg2.grid_rows == 2
        assert cfg2.grid_cols == 5
        assert cfg2.grid_shape == (2, 5)

    def test_client_config_conflicting_grid_shape_raises_error(self):
        """Conflicting grid_rows and grid_shape raises ValueError."""
        with pytest.raises(ValueError, match="Conflicting configuration"):
            ClientConfig(grid_rows=3, grid_shape=(2, 4))

    def test_client_config_invalid_grid_dims_raise_error(self):
        """grid_rows < 1 or grid_cols < 1 raises ValidationError."""
        with pytest.raises(Exception):
            ClientConfig(grid_rows=0)
        with pytest.raises(Exception):
            ClientConfig(grid_shape=(0, 2))

    def test_defensive_dimension_mismatch(self):
        """Inner dimension mismatch raises MatrixDimensionError."""
        a = np.ones((4, 8), dtype=np.float32)
        b = np.ones((6, 4), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="Dimension mismatch"):
            MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=2)

    def test_defensive_empty_or_non_2d_matrices(self):
        """Empty or 1D/3D arrays raise MatrixDimensionError."""
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_output_grid_2d(np.ones((0, 4)), np.ones((4, 4)))
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_output_grid_2d(np.ones((4,)), np.ones((4, 4)))
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_output_grid_2d(np.ones((2, 2, 2)), np.ones((2, 2)))
