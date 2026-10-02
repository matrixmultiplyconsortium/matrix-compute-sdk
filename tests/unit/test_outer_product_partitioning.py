"""
Comprehensive Unit Test Suite for 1D Outer-Product (Upload-Optimal) Matrix Partitioning.

Milestone 1 Test Specification for matrix-compute-sdk.
Validates:
1. Strip calculation algorithm: divisibility, balanced remainder distribution, and clamping.
2. Two-sided boundary clamping: P > K (oversubscription), P = 1, and P <= 0.
3. Plan and task geometry: TileGridPlan metadata, TileTaskPlan attributes, and dict access.
4. Exact slice continuity: pairwise disjoint intervals covering [0, K) with zero gap and zero overlap.
5. Mathematical equivalence: sum of partial products matches np.matmul across FP32, FP16, BF16.
6. Floating-point error bounds: Higham bound compliance across all supported dtypes.
7. Shape variation stress tests: prime, odd, scalar (1x1), and asymmetric (M != N != K) dimensions.
8. ResultAssembler integration: in-place accumulation at (0, 0, p), completion tracking, and duplicate rejection.
9. Defensive validation: dimension mismatches, non-2D matrices, empty matrices, and invalid types.
10. Strategy routing & backward compatibility in MatrixTiler.create_plan.
"""

from __future__ import annotations

import concurrent.futures
import math
import random
import threading
from typing import List, Tuple

import numpy as np
import pytest

from matrix_compute.config import PartitioningStrategy
from matrix_compute.exceptions import MatrixDimensionError, TilingError
from matrix_compute.tiling import (
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
from matrix_compute.tiling.partitioner import compute_1d_strips


# =============================================================================
# HELPER ORACLES & VALIDATORS
# =============================================================================

def assert_plan_invariants(
    plan: TileGridPlan,
    a: np.ndarray,
    b: np.ndarray,
    requested_p: int,
    expected_dtype: str = "fp32",
    expected_prefix: str = "strip",
) -> None:
    """
    Comprehensive oracle verifying all structural and geometric invariants of a TileGridPlan.
    """
    m, k = a.shape
    _, n = b.shape
    expected_p = min(max(1, int(requested_p)), k)

    # 1. Grid Plan Metadata
    assert isinstance(plan, TileGridPlan), f"Expected TileGridPlan, got {type(plan)}"
    assert plan.m == m, f"Plan M ({plan.m}) != a.shape[0] ({m})"
    assert plan.k == k, f"Plan K ({plan.k}) != a.shape[1] ({k})"
    assert plan.n == n, f"Plan N ({plan.n}) != b.shape[1] ({n})"
    assert plan.num_m_blocks == 1, f"Expected num_m_blocks=1, got {plan.num_m_blocks}"
    assert plan.num_n_blocks == 1, f"Expected num_n_blocks=1, got {plan.num_n_blocks}"
    assert plan.num_k_blocks == expected_p, f"Expected num_k_blocks={expected_p}, got {plan.num_k_blocks}"
    assert plan.total_tasks == expected_p, f"Expected total_tasks={expected_p}, got {plan.total_tasks}"
    assert len(plan.tasks) == expected_p, f"len(plan.tasks) ({len(plan.tasks)}) != {expected_p}"
    assert plan.dtype == normalize_dtype(expected_dtype)

    # 2. Total GFLOPs Conservation
    expected_total_gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9
    assert math.isclose(plan.total_gflops, expected_total_gflops, rel_tol=1e-7), (
        f"Plan total_gflops {plan.total_gflops} != expected {expected_total_gflops}"
    )
    task_gflops_sum = sum(t.estimated_gflops for t in plan.tasks)
    assert math.isclose(task_gflops_sum, expected_total_gflops, rel_tol=1e-7), (
        f"Sum of task GFLOPs {task_gflops_sum} != {expected_total_gflops}"
    )

    # 3. Individual Task Invariants
    curr_k = 0
    for p, task in enumerate(plan.tasks):
        assert isinstance(task, TileTaskPlan)
        assert task.row_idx == 0, f"Task {p} row_idx must be 0"
        assert task.col_idx == 0, f"Task {p} col_idx must be 0"
        assert task.k_idx == p, f"Task {p} k_idx must be {p}"
        assert task.row_slice == (0, m), f"Task {p} row_slice must be (0, {m})"
        assert task.col_slice == (0, n), f"Task {p} col_slice must be (0, {n})"

        k_s, k_e = task.k_slice
        assert k_s == curr_k, f"Task {p} k_start ({k_s}) != expected ({curr_k}) - Gap or overlap detected"
        assert k_e > k_s, f"Task {p} strip length must be >= 1, got {k_e - k_s}"
        k_p = k_e - k_s
        curr_k = k_e

        # Shape consistency
        assert task.m_i == m
        assert task.k_j == k_p
        assert task.n_k == n
        assert task.shape_a == (m, k_p)
        assert task.shape_b == (k_p, n)
        assert task.shape_c == (m, n)
        assert task.a_tile.shape == (m, k_p)
        assert task.b_tile.shape == (k_p, n)

        # Slice data matches input matrix slices
        np.testing.assert_array_equal(task.a_tile, a[:, k_s:k_e])
        np.testing.assert_array_equal(task.b_tile, b[k_s:k_e, :])

        # Task GFLOPs
        expected_task_gflops = (2.0 * float(m) * float(k_p) * float(n)) / 1e9
        assert math.isclose(task.estimated_gflops, expected_task_gflops, rel_tol=1e-7)

    assert curr_k == k, f"Final strip ended at {curr_k} != inner dimension K ({k})"


# =============================================================================
# SUITE 1: 1D Strip Partitioning Algorithm Tests
# =============================================================================

class TestOuterProduct1DStripAlgorithm:
    """Tests the low-level compute_1d_strips(k, p) function directly."""

    @pytest.mark.parametrize(
        "k, p, expected_strips",
        [
            # Divisible cases
            (64, 4, [(0, 16), (16, 32), (32, 48), (48, 64)]),
            (12, 3, [(0, 4), (4, 8), (8, 12)]),
            (12, 12, [(i, i + 1) for i in range(12)]),
            # Non-divisible remainder distribution cases (first r get q+1)
            (65, 4, [(0, 17), (17, 33), (33, 49), (49, 65)]),
            (10, 4, [(0, 3), (3, 6), (6, 8), (8, 10)]),
            (23, 5, [(0, 5), (5, 10), (10, 15), (15, 19), (19, 23)]),
            (11, 4, [(0, 3), (3, 6), (6, 9), (9, 11)]),
            # Prime K
            (17, 4, [(0, 5), (5, 9), (9, 13), (13, 17)]),
            (31, 7, [(0, 5), (5, 10), (10, 15), (15, 19), (19, 23), (23, 27), (27, 31)]),
            # Single strip
            (100, 1, [(0, 100)]),
            (1, 1, [(0, 1)]),
            # Oversubscription (P > K) clamped to K strips of length 1
            (4, 10, [(0, 1), (1, 2), (2, 3), (3, 4)]),
            (1, 50, [(0, 1)]),
            (7, 100, [(i, i + 1) for i in range(7)]),
            # Zero / negative P clamped to 1 strip
            (16, 0, [(0, 16)]),
            (16, -5, [(0, 16)]),
        ],
    )
    def test_strip_computation_exact_intervals(self, k: int, p: int, expected_strips: List[Tuple[int, int]]):
        """Verifies strip intervals match the exact expected mathematical intervals."""
        strips = compute_1d_strips(k, p)
        assert strips == expected_strips

    @pytest.mark.parametrize("k, p", [(10, 3), (25, 4), (100, 7), (97, 8), (3, 2), (500, 13)])
    def test_strip_computation_remainder_balance_invariant(self, k: int, p: int):
        """
        Oracle property test:
        1. Lengths differ by at most 1: max(lengths) - min(lengths) <= 1.
        2. Exactly r strips have length q + 1, and p_eff - r strips have length q.
        3. Larger strips are placed first (index < r).
        """
        strips = compute_1d_strips(k, p)
        p_eff = min(max(1, p), k)
        q = k // p_eff
        r = k % p_eff

        lengths = [end - start for start, end in strips]
        assert len(lengths) == p_eff
        assert sum(lengths) == k
        assert max(lengths) - min(lengths) <= 1

        for idx, length in enumerate(lengths):
            if idx < r:
                assert length == q + 1, f"Strip {idx} length {length} != {q + 1}"
            else:
                assert length == q, f"Strip {idx} length {length} != {q}"

    def test_strip_computation_invalid_k_error(self):
        """Dimension k < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            compute_1d_strips(k=0, p=4)
        with pytest.raises(ValueError, match="must be >= 1"):
            compute_1d_strips(k=-10, p=4)


# =============================================================================
# SUITE 2: Plan & Task Geometry Tests
# =============================================================================

class TestOuterProduct1DPlanGeometry:
    """Tests MatrixTiler.create_plan_outer_product_1d manifest structure and task plans."""

    def test_plan_outer_product_even_divisible(self):
        """Partitions 64x64 matrices with num_strips=4 -> 4 strip tasks of 64x16."""
        a = np.ones((64, 64), dtype=np.float32)
        b = np.ones((64, 64), dtype=np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=4)
        assert_plan_invariants(plan, a, b, requested_p=4)

        for task in plan.tasks:
            assert task.a_tile.shape == (64, 16)
            assert task.b_tile.shape == (16, 64)
            assert task.shape_c == (64, 64)

    def test_plan_outer_product_non_divisible(self):
        """Partitions (32x50) x (50x48) with num_strips=4 -> strips 13, 13, 12, 12."""
        a = np.arange(32 * 50, dtype=np.float32).reshape((32, 50))
        b = np.arange(50 * 48, dtype=np.float32).reshape((50, 48))
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=4)
        assert_plan_invariants(plan, a, b, requested_p=4)

        lengths = [t.k_j for t in plan.tasks]
        assert lengths == [13, 13, 12, 12]

    def test_plan_outer_product_task_id_formatting(self):
        """Task IDs default to 'strip_p' or '{job_id}_strip_p'."""
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)

        # Default prefix
        plan_default = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=2)
        assert plan_default.tasks[0].task_id == "strip_0"
        assert plan_default.tasks[1].task_id == "strip_1"

        # Explicit job_id
        plan_job = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=2, job_id="job_matmul_99")
        assert plan_job.tasks[0].task_id == "job_matmul_99_strip_0"
        assert plan_job.tasks[1].task_id == "job_matmul_99_strip_1"

    def test_plan_outer_product_dictionary_subscription(self):
        """Verifies dual access: attribute access and dictionary subscription on TileTaskPlan."""
        a = np.ones((16, 8), dtype=np.float32)
        b = np.ones((8, 16), dtype=np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=2)
        t = plan.tasks[0]

        assert t["task_id"] == t.task_id
        assert t["row_idx"] == 0
        assert t["col_idx"] == 0
        assert t["k_idx"] == 0
        assert t["row_slice"] == (0, 16)
        assert t["col_slice"] == (0, 16)
        assert t["k_slice"] == (0, 4)
        assert np.array_equal(t["a_tile"], t.a_tile)
        assert np.array_equal(t["b_tile"], t.b_tile)
        assert t.get("estimated_gflops") == t.estimated_gflops
        assert "a_tile" in t
        assert "b_tile" in t
        assert dict(t.items())["task_id"] == t.task_id

    def test_partition_outer_product_1d_flat_list_helper(self):
        """MatrixTiler.partition_outer_product_1d returns flat List[TileTaskPlan]."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        tasks = MatrixTiler.partition_outer_product_1d(a, b, num_strips=4)
        assert isinstance(tasks, list)
        assert len(tasks) == 4
        assert all(isinstance(t, TileTaskPlan) for t in tasks)


# =============================================================================
# SUITE 3: Exact Slice Continuity & Non-Overlap Invariants
# =============================================================================

class TestOuterProduct1DIndexContinuity:
    """Oracle tests asserting zero gaps, zero overlaps, and exact array reconstruction."""

    @pytest.mark.parametrize(
        "m, k, n, p",
        [
            (32, 64, 32, 4),
            (17, 23, 19, 5),       # Small primes
            (1, 1, 1, 1),          # Scalar
            (1, 1, 1, 8),          # Scalar oversubscribed
            (64, 4, 64, 10),       # Oversubscription
            (128, 16, 4, 3),       # Tall-skinny
            (4, 16, 128, 3),       # Short-wide
            (1, 256, 1, 8),        # Vector dot
            (64, 1, 64, 1),        # Vector outer
            (37, 79, 53, 6),       # Asymmetric prime
            (100, 100, 100, 33),   # Large odd
        ],
    )
    def test_slice_continuity_and_full_matrix_reconstruction(self, m: int, k: int, n: int, p: int):
        """
        Oracle verification:
        1. Union of k_slice covers [0, k) with zero gaps and zero overlaps.
        2. Horizontal stacking of a_tile reconstructs original matrix A: hstack([t.a_tile]) == A.
        3. Vertical stacking of b_tile reconstructs original matrix B: vstack([t.b_tile]) == B.
        4. Total uploaded elements strictly equal M*K + K*N (zero redundant elements).
        """
        np.random.seed(42)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p)
        assert_plan_invariants(plan, a, b, requested_p=p)

        # Reconstruct A along columns
        reconstructed_a = np.hstack([task.a_tile for task in plan.tasks])
        np.testing.assert_array_equal(reconstructed_a, a)

        # Reconstruct B along rows
        reconstructed_b = np.vstack([task.b_tile for task in plan.tasks])
        np.testing.assert_array_equal(reconstructed_b, b)

        # Assert strict upload element count: strictly M*K + K*N
        total_upload_elements = sum(t.a_tile.size + t.b_tile.size for t in plan.tasks)
        assert total_upload_elements == (m * k + k * n)


# =============================================================================
# SUITE 4: Boundary & Irregular Dimension Stress Tests
# =============================================================================

class TestOuterProduct1DShapeVariations:
    """Stress tests across extreme, irregular, prime, and scalar geometries."""

    def test_shape_scalar_1x1x1(self):
        """1x1 matrices with P=1 produces 1 task and exact scalar multiplication."""
        a = np.array([[7.0]], dtype=np.float32)
        b = np.array([[3.0]], dtype=np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=1)
        assert len(plan.tasks) == 1
        t = plan.tasks[0]
        c_sub = np.matmul(t.a_tile, t.b_tile)
        assert c_sub[0, 0] == 21.0

    def test_shape_scalar_oversubscribed(self):
        """1x1 matrices with requested P=100 clamps to P_eff=1 without crashing."""
        a = np.array([[4.0]], dtype=np.float32)
        b = np.array([[5.0]], dtype=np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=100)
        assert len(plan.tasks) == 1
        assert plan.num_k_blocks == 1

    def test_shape_prime_dimensions(self):
        """All-prime dimensions (17x23) x (23x19) with P=5."""
        np.random.seed(17)
        a = np.random.randn(17, 23).astype(np.float32)
        b = np.random.randn(23, 19).astype(np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=5)
        assert_plan_invariants(plan, a, b, requested_p=5)

    def test_shape_large_primes(self):
        """Large primes (67x31) x (31x43) with P=7."""
        np.random.seed(67)
        a = np.random.randn(67, 31).astype(np.float32)
        b = np.random.randn(31, 43).astype(np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=7)
        assert_plan_invariants(plan, a, b, requested_p=7)

    def test_shape_vector_inner_product(self):
        """Dot product: row vector (1x256) x col vector (256x1) with P=8 -> scalar (1x1)."""
        a = np.ones((1, 256), dtype=np.float32)
        b = np.ones((256, 1), dtype=np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=8)
        assert_plan_invariants(plan, a, b, requested_p=8)

        accumulated = np.zeros((1, 1), dtype=np.float32)
        for t in plan.tasks:
            accumulated += np.matmul(t.a_tile, t.b_tile)
        assert accumulated[0, 0] == 256.0

    def test_shape_vector_outer_product(self):
        """Rank-1 outer product: col vector (64x1) x row vector (1x64) with P=1 (and P=4 clamped to 1)."""
        a = np.arange(64, dtype=np.float32).reshape((64, 1))
        b = np.arange(64, dtype=np.float32).reshape((1, 64))

        # Requesting P=4 on K=1 clamps to 1
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=4)
        assert len(plan.tasks) == 1
        assert plan.num_k_blocks == 1

        c_expected = np.matmul(a, b)
        c_task = np.matmul(plan.tasks[0].a_tile, plan.tasks[0].b_tile)
        np.testing.assert_array_equal(c_task, c_expected)

    def test_shape_extreme_oversubscription(self):
        """Inner dimension K=4 partitioned into P=20 strips clamps to 4 strips of size 1."""
        a = np.ones((10, 4), dtype=np.float32)
        b = np.ones((4, 10), dtype=np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=20)
        assert len(plan.tasks) == 4
        for t in plan.tasks:
            assert t.k_j == 1


# =============================================================================
# SUITE 5: Exact Mathematical Identity & Floating-Point Error Bounds
# =============================================================================

class TestOuterProduct1DMathematicalIdentity:
    """Verifies sum_{p} A_p B_p == A x B across integer arithmetic and FP32/FP16/BF16 Higham bounds."""

    def test_mathematical_identity_exact_integers(self):
        """Integer arithmetic has zero rounding: sum_{p} A_p B_p == A B bit-for-bit."""
        np.random.seed(101)
        a = np.random.randint(-5, 5, size=(24, 37)).astype(np.float32)
        b = np.random.randint(-5, 5, size=(37, 19)).astype(np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=6)

        c_accumulated = np.zeros((24, 19), dtype=np.float32)
        for t in plan.tasks:
            c_accumulated += np.matmul(t.a_tile, t.b_tile)

        c_expected = np.matmul(a, b)
        np.testing.assert_array_equal(c_accumulated, c_expected)

    def test_mathematical_identity_identity_matrix(self):
        """Multiplication with Identity matrix I_K reconstructs original matrix exactly."""
        np.random.seed(102)
        a = np.random.randn(20, 20).astype(np.float32)
        identity = np.eye(20, dtype=np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, identity, num_strips=5)
        c_acc = np.zeros((20, 20), dtype=np.float32)
        for t in plan.tasks:
            c_acc += np.matmul(t.a_tile, t.b_tile)

        np.testing.assert_allclose(c_acc, a, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize(
        "m, k, n, p",
        [
            (32, 64, 32, 4),
            (17, 23, 19, 5),
            (48, 33, 64, 7),
            (10, 4, 10, 2),
            (67, 31, 43, 8),
        ],
    )
    def test_mathematical_identity_fp32_higham_bound(self, m: int, k: int, n: int, p: int):
        """Verifies FP32 outer-product sum against Higham rounding error bound."""
        np.random.seed(201)
        a = np.random.uniform(-2.0, 2.0, size=(m, k)).astype(np.float32)
        b = np.random.uniform(-2.0, 2.0, size=(k, n)).astype(np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p, dtype="fp32")
        c_accumulated = np.zeros((m, n), dtype=np.float32)
        for t in plan.tasks:
            c_accumulated += np.matmul(t.a_tile, t.b_tile)

        c_expected = np.matmul(a, b)
        np.testing.assert_allclose(c_accumulated, c_expected, rtol=1e-5, atol=1e-5)

        # Higham bound validation
        norm_a = float(np.max(np.sum(np.abs(a.astype(np.float64)), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b.astype(np.float64)), axis=1)))
        bound = max(calculate_higham_bound(k, norm_a, norm_b, dtype="fp32"), 1e-5)
        max_diff = float(np.max(np.abs(c_accumulated.astype(np.float64) - c_expected.astype(np.float64))))
        assert max_diff <= bound, f"FP32 discrepancy {max_diff} exceeded Higham bound {bound}"

    @pytest.mark.parametrize(
        "m, k, n, p",
        [
            (24, 32, 24, 4),
            (17, 23, 19, 5),
            (32, 16, 32, 3),
        ],
    )
    def test_mathematical_identity_fp16_higham_bound(self, m: int, k: int, n: int, p: int):
        """Verifies FP16 outer-product sum against Higham bound."""
        np.random.seed(202)
        # Scaled inputs to prevent intermediate half-precision overflow
        a = np.random.uniform(-0.5, 0.5, size=(m, k)).astype(np.float16)
        b = np.random.uniform(-0.5, 0.5, size=(k, n)).astype(np.float16)

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p, dtype="fp16")
        assert plan.dtype == "fp16"

        c_accumulated = np.zeros((m, n), dtype=np.float16)
        for t in plan.tasks:
            assert t.a_tile.dtype == np.float16
            assert t.b_tile.dtype == np.float16
            c_accumulated += np.matmul(t.a_tile, t.b_tile)

        c_ref = emulate_gemm(a, b, dtype="fp16")
        np.testing.assert_allclose(c_accumulated, c_ref, rtol=1e-2, atol=1e-2)

        norm_a = float(np.max(np.sum(np.abs(a.astype(np.float64)), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b.astype(np.float64)), axis=1)))
        bound = max(calculate_higham_bound(k, norm_a, norm_b, dtype="fp16"), 1e-3)
        max_diff = float(np.max(np.abs(c_accumulated.astype(np.float64) - c_ref.astype(np.float64))))
        assert max_diff <= bound, f"FP16 discrepancy {max_diff} exceeded Higham bound {bound}"

    @pytest.mark.parametrize(
        "m, k, n, p",
        [
            (24, 32, 24, 4),
            (17, 23, 19, 5),
            (32, 16, 32, 3),
        ],
    )
    def test_mathematical_identity_bf16_higham_bound(self, m: int, k: int, n: int, p: int):
        """Verifies IEEE BF16 bit-masking emulation and Higham bound compliance."""
        np.random.seed(203)
        a = truncate_to_bf16(np.random.uniform(-1.0, 1.0, size=(m, k)).astype(np.float32))
        b = truncate_to_bf16(np.random.uniform(-1.0, 1.0, size=(k, n)).astype(np.float32))

        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=p, dtype="bf16")
        assert plan.dtype == "bf16"

        c_accumulated = np.zeros((m, n), dtype=np.float32)
        for t in plan.tasks:
            assert validate_bf16_masking(t.a_tile) is True
            assert validate_bf16_masking(t.b_tile) is True
            partial = np.matmul(t.a_tile, t.b_tile)
            c_accumulated += truncate_to_bf16(partial)

        c_bf16 = truncate_to_bf16(c_accumulated)
        assert validate_bf16_masking(c_bf16) is True

        c_ref = emulate_gemm(a, b, dtype="bf16")
        np.testing.assert_allclose(c_bf16, c_ref, rtol=2e-2, atol=2e-2)

        norm_a = float(np.max(np.sum(np.abs(a.astype(np.float64)), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b.astype(np.float64)), axis=1)))
        bound = max(calculate_higham_bound(k, norm_a, norm_b, dtype="bf16"), 1e-2)
        max_diff = float(np.max(np.abs(c_bf16.astype(np.float64) - c_ref.astype(np.float64))))
        assert max_diff <= bound, f"BF16 discrepancy {max_diff} exceeded Higham bound {bound}"


# =============================================================================
# SUITE 6: Result Accumulator Integration Tests
# =============================================================================

class TestOuterProduct1DAssemblerIntegration:
    """Tests seamless in-place accumulation into ResultAssembler."""

    def test_assembler_from_plan_sequential_accumulation(self):
        """ResultAssembler.from_plan correctly configures buffer and accumulates 1D strip tasks."""
        np.random.seed(301)
        a = np.random.randn(32, 48).astype(np.float32)
        b = np.random.randn(48, 64).astype(np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=4)

        assembler = ResultAssembler.from_plan(plan)
        assert assembler.is_complete() is False
        assert assembler.get_progress() == 0.0

        for t in plan.tasks:
            c_p = np.matmul(t.a_tile, t.b_tile)
            assembler.accumulate_task(t, c_p)

        assert assembler.is_complete() is True
        assert assembler.get_progress() == 1.0
        assert len(assembler.missing_tasks()) == 0

        assembled = assembler.get_result(enforce_complete=True)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(assembled, expected, rtol=1e-5, atol=1e-5)

    def test_assembler_assemble_tiles_drop_in(self):
        """ResultAssembler.assemble_tiles works as drop-in reference accumulator."""
        np.random.seed(302)
        a = np.random.randn(20, 30).astype(np.float32)
        b = np.random.randn(30, 25).astype(np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=3)

        assembled = ResultAssembler.assemble_tiles(20, 25, plan.tasks, dtype="fp32")
        expected = np.matmul(a, b)
        np.testing.assert_allclose(assembled, expected, rtol=1e-5, atol=1e-5)

    def test_assembler_out_of_order_accumulation(self):
        """Randomly shuffled strip arrival order reconstructs identical matrix."""
        np.random.seed(303)
        a = np.random.randn(30, 40).astype(np.float32)
        b = np.random.randn(40, 30).astype(np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=5)

        # Forward
        asm_fwd = ResultAssembler.from_plan(plan)
        for t in plan.tasks:
            asm_fwd.accumulate_task(t, np.matmul(t.a_tile, t.b_tile))
        res_fwd = asm_fwd.get_result(enforce_complete=True)

        # Shuffled
        shuffled = list(plan.tasks)
        random.seed(42)
        random.shuffle(shuffled)
        asm_shuf = ResultAssembler.from_plan(plan)
        for t in shuffled:
            asm_shuf.accumulate_task(t, np.matmul(t.a_tile, t.b_tile))
        res_shuf = asm_shuf.get_result(enforce_complete=True)

        np.testing.assert_allclose(res_fwd, res_shuf, rtol=1e-6, atol=1e-5)

    def test_assembler_duplicate_rejection(self):
        """Re-accumulating task at same coordinate (0, 0, p) raises TilingError."""
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=2)
        assembler = ResultAssembler.from_plan(plan)

        t0 = plan.tasks[0]
        c0 = np.matmul(t0.a_tile, t0.b_tile)
        assembler.accumulate_task(t0, c0)

        # Duplicate attempt
        with pytest.raises(TilingError, match="Duplicate accumulation rejected"):
            assembler.accumulate_task(t0, c0)

    def test_assembler_incomplete_enforcement_error(self):
        """Calling get_result(enforce_complete=True) on partial accumulation raises TilingError."""
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=4)
        assembler = ResultAssembler.from_plan(plan)

        # Accumulate only 2 out of 4 tasks
        for t in plan.tasks[:2]:
            assembler.accumulate_task(t, np.matmul(t.a_tile, t.b_tile))

        assert assembler.is_complete() is False
        with pytest.raises(TilingError, match="Cannot extract result: assembly incomplete"):
            assembler.get_result(enforce_complete=True)

    def test_assembler_thread_safe_concurrent_accumulation(self):
        """Multiple threads concurrently accumulating strip tasks into ResultAssembler."""
        np.random.seed(304)
        a = np.random.randn(32, 64).astype(np.float32)
        b = np.random.randn(64, 32).astype(np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=8)
        assembler = ResultAssembler.from_plan(plan)

        def worker_accumulate(task: TileTaskPlan):
            c_p = np.matmul(task.a_tile, task.b_tile)
            assembler.accumulate_task(task, c_p)

        threads = [threading.Thread(target=worker_accumulate, args=(t,)) for t in plan.tasks]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5.0)

        assert assembler.is_complete() is True
        assembled = assembler.get_result(enforce_complete=True)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(assembled, expected, rtol=1e-5, atol=1e-5)


# =============================================================================
# SUITE 7: Defensive Validation & Error Handling
# =============================================================================

class TestOuterProduct1DDefensiveValidation:
    """Tests defensive parameter validation, dimension checking, and error contracts."""

    def test_dimension_mismatch_error(self):
        """Matrices with inner dimension mismatch (A cols != B rows) raise MatrixDimensionError."""
        a = np.ones((16, 32), dtype=np.float32)
        b = np.ones((24, 16), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="Dimension mismatch"):
            MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=4)

    def test_non_2d_matrix_error(self):
        """1D or 3D matrices raise MatrixDimensionError."""
        a_1d = np.ones(16, dtype=np.float32)
        b_2d = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="2-dimensional"):
            MatrixTiler.create_plan_outer_product_1d(a_1d, b_2d, num_strips=2)

        a_3d = np.ones((2, 16, 16), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="2-dimensional"):
            MatrixTiler.create_plan_outer_product_1d(a_3d, b_2d, num_strips=2)

    def test_zero_dimension_error(self):
        """Matrices with dimension 0 raise MatrixDimensionError."""
        a = np.ones((0, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="non-zero"):
            MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=2)

    def test_non_ndarray_type_error(self):
        """Non-numpy array arguments raise MatrixDimensionError."""
        with pytest.raises(MatrixDimensionError, match="numpy.ndarray"):
            MatrixTiler.create_plan_outer_product_1d([[1, 2], [3, 4]], np.ones((2, 2)), num_strips=2)

    def test_unsupported_dtype_error(self):
        """Unsupported precision string raises ValueError."""
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported precision dtype"):
            MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=2, dtype="int32")


# =============================================================================
# SUITE 8: Strategy Routing & Backward Compatibility
# =============================================================================

class TestOuterProduct1DStrategyRouting:
    """Verifies MatrixTiler.create_plan routing, strategy enums, and backward compatibility."""

    def test_routing_with_partitioning_strategy_enum(self):
        """Passing strategy=PartitioningStrategy.OUTER_PRODUCT_1D routes to 1D outer product."""
        a = np.ones((32, 64), dtype=np.float32)
        b = np.ones((64, 32), dtype=np.float32)
        plan = MatrixTiler.create_plan(
            a,
            b,
            strategy=PartitioningStrategy.OUTER_PRODUCT_1D,
            num_strips=4,
        )
        assert plan.num_m_blocks == 1
        assert plan.num_n_blocks == 1
        assert plan.num_k_blocks == 4
        assert plan.total_tasks == 4

    def test_routing_with_case_insensitive_string(self):
        """Passing strategy='outer_product_1d' or 'OUTER_PRODUCT_1D' routes correctly."""
        a = np.ones((16, 32), dtype=np.float32)
        b = np.ones((32, 16), dtype=np.float32)

        plan_lower = MatrixTiler.create_plan(a, b, strategy="outer_product_1d", num_strips=2)
        assert plan_lower.num_k_blocks == 2
        assert plan_lower.total_tasks == 2

        plan_upper = MatrixTiler.create_plan(a, b, strategy="OUTER_PRODUCT_1D", num_strips=2)
        assert plan_upper.num_k_blocks == 2
        assert plan_upper.total_tasks == 2

    def test_backward_compatibility_default_block_3d(self):
        """Calling create_plan without strategy maintains 100% backward compatible 3D block tiling."""
        a = np.ones((32, 32), dtype=np.float32)
        b = np.ones((32, 32), dtype=np.float32)

        # Baseline 3D call: tile_size=16 -> 2x2x2 = 8 tasks
        plan = MatrixTiler.create_plan(a, b, tile_size=16)
        assert plan.num_m_blocks == 2
        assert plan.num_k_blocks == 2
        assert plan.num_n_blocks == 2
        assert plan.total_tasks == 8

    def test_routing_heuristic_num_strips_fallback(self):
        """When strategy=OUTER_PRODUCT_1D and num_strips is None, uses tile_size heuristic."""
        a = np.ones((32, 100), dtype=np.float32)
        b = np.ones((100, 32), dtype=np.float32)

        # tile_size=32 on K=100 -> ceil(100 / 32) = 4 strips
        plan = MatrixTiler.create_plan(
            a,
            b,
            strategy=PartitioningStrategy.OUTER_PRODUCT_1D,
            tile_size=32,
        )
        assert plan.num_k_blocks == 4
        assert plan.total_tasks == 4
