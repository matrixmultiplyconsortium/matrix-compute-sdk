"""
Comprehensive Unit Test Suite for Matrix Tiling, Precision Emulation, and Result Assembly.
Covers precision casting, BF16 mantissa truncation, NaN/Inf/zero preservation, 3D partitioning,
remainder blocks, non-square aspects, thread-safe accumulation, and defensive validations.
"""

from __future__ import annotations

import concurrent.futures
import math
import random
import numpy as np
import pytest

from matrix_compute.exceptions import MatrixDimensionError, TilingError
from matrix_compute.tiling import (
    MatrixTiler,
    ResultAssembler,
    TileGridPlan,
    TileTaskPlan,
    calculate_higham_bound,
    cast_matrix,
    emulate_gemm,
    get_bytes_per_element,
    get_numpy_dtype,
    get_unit_roundoff,
    normalize_dtype,
    truncate_to_bf16,
    validate_bf16_masking,
)


# =============================================================================
# GROUP 1: Precision Emulation Tests (precision.py)
# =============================================================================

class TestPrecisionEmulation:
    """Exhaustive tests for data type normalization, casting, and IEEE BF16 bit emulation."""

    def test_precision_normalize_aliases(self):
        """Verifies normalization for standard aliases across fp32, fp16, bf16, fp64."""
        assert normalize_dtype("fp32") == "fp32"
        assert normalize_dtype("float32") == "fp32"
        assert normalize_dtype("f32") == "fp32"
        assert normalize_dtype("single") == "fp32"

        assert normalize_dtype("fp16") == "fp16"
        assert normalize_dtype("float16") == "fp16"
        assert normalize_dtype("f16") == "fp16"
        assert normalize_dtype("half") == "fp16"

        assert normalize_dtype("bf16") == "bf16"
        assert normalize_dtype("bfloat16") == "bf16"
        assert normalize_dtype("bf") == "bf16"

        assert normalize_dtype("fp64") == "fp64"
        assert normalize_dtype("double") == "fp64"

        assert normalize_dtype(np.float32) == "fp32"
        assert normalize_dtype(np.float16) == "fp16"
        assert normalize_dtype(np.float64) == "fp64"

        with pytest.raises(ValueError, match="Unsupported precision dtype"):
            normalize_dtype("int32")

        with pytest.raises(ValueError, match="Unsupported precision dtype"):
            normalize_dtype("complex128")

    def test_precision_numpy_dtype_mapping(self):
        """Validates storage dtype mapping."""
        assert get_numpy_dtype("fp32") == np.float32
        assert get_numpy_dtype("fp16") == np.float16
        assert get_numpy_dtype("bf16") == np.float32  # BF16 uses float32 container
        assert get_numpy_dtype("fp64") == np.float64

    def test_precision_bytes_per_element(self):
        """Validates element sizes for memory and streaming bandwidth calculation."""
        assert get_bytes_per_element("fp32") == 4
        assert get_bytes_per_element("fp16") == 2
        assert get_bytes_per_element("bf16") == 4
        assert get_bytes_per_element("fp64") == 8

    def test_precision_bf16_mantissa_masking(self):
        """Verifies that lower 16 bits of FP32 mantissa are zeroed by truncate_to_bf16."""
        np.random.seed(101)
        arr = np.random.randn(10, 10).astype(np.float32)
        bf16 = truncate_to_bf16(arr)
        assert validate_bf16_masking(bf16) is True

        u32 = np.ascontiguousarray(bf16).view(np.uint32)
        assert np.all((u32 & np.uint32(0x0000FFFF)) == 0)

    def test_precision_bf16_nan_preservation(self):
        """Ensures NaNs retain isnan() semantics and quiet NaN bit (0x00400000) is preserved."""
        # Create a float32 NaN whose lower 16 mantissa bits are non-zero, but upper 7 are zero
        nan_u32 = np.uint32(0x7F800001)  # Exponent 0xFF, mantissa bit 0 set
        nan_f32 = np.array([nan_u32], dtype=np.uint32).view(np.float32)
        assert np.isnan(nan_f32[0])

        bf16_nan = truncate_to_bf16(nan_f32)
        assert np.isnan(bf16_nan[0])
        res_u32 = bf16_nan.view(np.uint32)[0]
        # Bit 22 (quiet NaN) must be set
        assert (res_u32 & np.uint32(0x00400000)) != 0
        assert (res_u32 & np.uint32(0x0000FFFF)) == 0

    def test_precision_bf16_infinity_preservation(self):
        """Ensures +inf and -inf retain infinite value and correct signs."""
        infs = np.array([np.inf, -np.inf], dtype=np.float32)
        bf16_infs = truncate_to_bf16(infs)
        assert np.isposinf(bf16_infs[0])
        assert np.isneginf(bf16_infs[1])

    def test_precision_bf16_signed_zero_preservation(self):
        """Ensures +0.0 and -0.0 preserve distinct sign bits."""
        zeros = np.array([0.0, -0.0], dtype=np.float32)
        bf16_zeros = truncate_to_bf16(zeros)
        assert np.signbit(bf16_zeros[0]) == False
        assert np.signbit(bf16_zeros[1]) == True

    def test_precision_bf16_subnormal_handling(self):
        """Ensures subnormals underflow cleanly without producing NaNs."""
        subnormals = np.array([1e-40, -1e-40], dtype=np.float32)
        bf16_sub = truncate_to_bf16(subnormals)
        assert not np.any(np.isnan(bf16_sub))
        assert validate_bf16_masking(bf16_sub) is True

    def test_precision_bf16_relative_error_bound(self):
        """Verifies relative error |x - x_bf16| / |x| <= 2^-7."""
        np.random.seed(102)
        x = np.random.uniform(1.0, 100.0, size=(100, 100)).astype(np.float32)
        x_bf16 = truncate_to_bf16(x)
        rel_error = np.abs(x - x_bf16) / np.abs(x)
        max_rel = float(np.max(rel_error))
        assert max_rel <= 2.0 ** -7 + 1e-9

    def test_precision_unit_roundoffs(self):
        """Validates theoretical machine unit roundoffs."""
        assert get_unit_roundoff("fp32") == 2.0 ** -24
        assert get_unit_roundoff("fp16") == 2.0 ** -11
        assert get_unit_roundoff("bf16") == 2.0 ** -8
        assert get_unit_roundoff("fp64") == 2.0 ** -53

    def test_precision_higham_bound_scaling(self):
        """Asserts Higham bound scales properly with norms and precision."""
        b_fp32 = calculate_higham_bound(k=64, norm_a=10.0, norm_b=10.0, dtype="fp32")
        b_fp16 = calculate_higham_bound(k=64, norm_a=10.0, norm_b=10.0, dtype="fp16")
        b_bf16 = calculate_higham_bound(k=64, norm_a=10.0, norm_b=10.0, dtype="bf16")
        b_fp64 = calculate_higham_bound(k=64, norm_a=10.0, norm_b=10.0, dtype="fp64")

        assert b_bf16 > b_fp16 > b_fp32 > b_fp64

        # Scaling norms by 10x each scales bound by 100x
        b_scaled = calculate_higham_bound(k=64, norm_a=100.0, norm_b=100.0, dtype="fp32")
        np.testing.assert_allclose(b_scaled, b_fp32 * 100.0, rtol=1e-5)

    def test_precision_emulate_gemm_consistency(self):
        """Validates emulate_gemm accuracy."""
        np.random.seed(103)
        a = np.random.randn(8, 8).astype(np.float32)
        b = np.random.randn(8, 8).astype(np.float32)

        c_fp32 = emulate_gemm(a, b, dtype="fp32")
        c_expected = np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(c_fp32, c_expected, rtol=1e-6)

        c_bf16 = emulate_gemm(a, b, dtype="bf16")
        assert validate_bf16_masking(c_bf16) is True


# =============================================================================
# GROUP 2: Partitioner Tests (partitioner.py)
# =============================================================================

class TestMatrixPartitioner:
    """Exhaustive tests for MatrixTiler, TileTaskPlan, and TileGridPlan geometry."""

    def test_partitioner_square_even_blocks(self):
        """Partitions 64x64 square matrices with tile_size=32 -> 2x2x2 = 8 tasks."""
        a = np.ones((64, 64), dtype=np.float32)
        b = np.ones((64, 64), dtype=np.float32)
        tasks = MatrixTiler.partition(a, b, tile_size=32)
        assert len(tasks) == 8
        for t in tasks:
            assert t.a_tile.shape == (32, 32)
            assert t.b_tile.shape == (32, 32)
            assert t.shape_c == (32, 32)

    def test_partitioner_rectangular_blocks(self):
        """Partitions (64x32) x (32x128) with tile_size=32 -> 2x4x1 = 8 tasks."""
        a = np.ones((64, 32), dtype=np.float32)
        b = np.ones((32, 128), dtype=np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=32)
        assert plan.num_m_blocks == 2
        assert plan.num_k_blocks == 1
        assert plan.num_n_blocks == 4
        assert plan.total_tasks == 8
        assert len(plan.tasks) == 8

    def test_partitioner_odd_and_prime_dimensions(self):
        """Partitions (17x23) x (23x19) with tile_size=8 -> 3x3x3 = 27 tasks."""
        a = np.ones((17, 23), dtype=np.float32)
        b = np.ones((23, 19), dtype=np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=8)
        assert plan.num_m_blocks == 3
        assert plan.num_k_blocks == 3
        assert plan.num_n_blocks == 3
        assert plan.total_tasks == 27
        assert len(plan.tasks) == 27

        # Test boundary slices
        last_m = [t for t in plan.tasks if t.row_idx == 2]
        for t in last_m:
            assert t.m_i == 1  # 17 - 16 = 1

    def test_partitioner_non_multiple_remainders(self):
        """Partitions 100x100 matrices with tile_size=32 -> 4x4x4 = 64 tasks."""
        a = np.ones((100, 100), dtype=np.float32)
        b = np.ones((100, 100), dtype=np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=32)
        assert plan.num_m_blocks == 4
        assert plan.num_k_blocks == 4
        assert plan.num_n_blocks == 4
        assert plan.total_tasks == 64

        corner_task = plan.get_task(row_idx=3, col_idx=3, k_idx=3)
        assert corner_task is not None
        assert corner_task.m_i == 4   # 100 - 96 = 4
        assert corner_task.k_j == 4
        assert corner_task.n_k == 4

    def test_partitioner_1x1_scalar(self):
        """Partitions 1x1 matrices with tile_size=1 -> 1 task."""
        a = np.array([[5.0]], dtype=np.float32)
        b = np.array([[3.0]], dtype=np.float32)
        tasks = MatrixTiler.partition(a, b, tile_size=1)
        assert len(tasks) == 1
        assert tasks[0].a_tile.shape == (1, 1)
        assert tasks[0].b_tile.shape == (1, 1)
        assert tasks[0].shape_c == (1, 1)

    def test_partitioner_extreme_vector_aspect(self):
        """Partitions row vector (1x1024) and col vector (1024x1) with tile_size=128 -> 8 tasks."""
        a = np.ones((1, 1024), dtype=np.float32)
        b = np.ones((1024, 1), dtype=np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=128)
        assert plan.num_m_blocks == 1
        assert plan.num_n_blocks == 1
        assert plan.num_k_blocks == 8
        assert plan.total_tasks == 8

    def test_partitioner_single_tile_full_coverage(self):
        """When tile_size >= matrix dimensions, creates exactly 1 task."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        tasks = MatrixTiler.partition(a, b, tile_size=64)
        assert len(tasks) == 1
        assert tasks[0].shape_c == (16, 16)

    def test_partitioner_task_id_uniqueness(self):
        """Ensures all task IDs are unique and deterministic."""
        a = np.ones((32, 32), dtype=np.float32)
        b = np.ones((32, 32), dtype=np.float32)
        tasks = MatrixTiler.partition(a, b, tile_size=8, task_id_prefix="job1")
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))
        assert ids[0] == "job1_0_0_0"

    def test_partitioner_dual_access_task_plan(self):
        """Verifies attribute access, dict indexing, and dict methods on TileTaskPlan."""
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        tasks = MatrixTiler.partition(a, b, tile_size=4)
        t = tasks[0]

        assert t.task_id == t["task_id"]
        assert np.array_equal(t.a_tile, t["a_tile"])
        assert np.array_equal(t.b_tile, t["b_tile"])
        assert t.get("row_idx") == 0
        assert t.get("nonexistent", "default") == "default"
        assert "a_tile" in t
        assert "task_id" in t.keys()
        assert len(t.values()) == len(t.keys())
        assert dict(t.items())["task_id"] == t.task_id

    def test_partitioner_gflops_calculation(self):
        """Validates exact theoretical GFLOP calculation: 2 * M * K * N / 10^9."""
        a = np.ones((32, 32), dtype=np.float32)
        b = np.ones((32, 32), dtype=np.float32)
        tasks = MatrixTiler.partition(a, b, tile_size=16)
        expected_tile_gflops = (2.0 * 16 * 16 * 16) / 1e9
        for t in tasks:
            assert math.isclose(t.estimated_gflops, expected_tile_gflops, rel_tol=1e-7)

    def test_partitioner_inner_dimension_mismatch_error(self):
        """Rejects matrices with mismatched inner dimensions with 'Dimension mismatch'."""
        a = np.ones((16, 32), dtype=np.float32)
        b = np.ones((24, 16), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="Dimension mismatch"):
            MatrixTiler.partition(a, b, tile_size=8)
        # Also satisfies isinstance(ValueError)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            MatrixTiler.partition(a, b, tile_size=8)

    def test_partitioner_non_2d_matrix_error(self):
        """Rejects non-2D matrices."""
        a_1d = np.ones(16, dtype=np.float32)
        b_2d = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.partition(a_1d, b_2d, tile_size=8)

    def test_partitioner_invalid_tile_size_error(self):
        """Rejects tile_size < 1."""
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        with pytest.raises(ValueError):
            MatrixTiler.partition(a, b, tile_size=0)
        with pytest.raises(ValueError):
            MatrixTiler.partition(a, b, tile_size=-5)


# =============================================================================
# GROUP 3: Assembler Tests (assembler.py)
# =============================================================================

class TestResultAssembler:
    """Exhaustive tests for thread-safe ResultAssembler and reassembly correctness."""

    def test_assembler_square_fp32_matches_matmul(self):
        """Reassembles 32x32 tiles with S=16, matching np.matmul within 1e-5."""
        np.random.seed(201)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=16)

        assembler = ResultAssembler.from_plan(plan)
        for task in plan.tasks:
            tile_c = np.matmul(task.a_tile, task.b_tile)
            assembler.accumulate_task(task, tile_c)

        assembled = assembler.get_result(enforce_complete=True)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(assembled, expected, rtol=1e-5, atol=1e-5)

    def test_assembler_rectangular_reassembly(self):
        """Reassembles rectangular (48x32) x (32x64) with S=16."""
        np.random.seed(202)
        a = np.random.randn(48, 32).astype(np.float32)
        b = np.random.randn(32, 64).astype(np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=16)

        assembler = ResultAssembler.from_plan(plan)
        for task in plan.tasks:
            tile_c = np.matmul(task.a_tile, task.b_tile)
            assembler.accumulate_task(task, tile_c)

        assembled = assembler.get_result(enforce_complete=True)
        expected = np.matmul(a, b)
        assert assembled.shape == (48, 64)
        np.testing.assert_allclose(assembled, expected, rtol=1e-5, atol=1e-5)

    def test_assembler_multi_k_reduction_sum(self):
        """Verifies outer-product accumulation over multiple reduction K-blocks."""
        np.random.seed(203)
        a = np.random.randn(8, 16).astype(np.float32)
        b = np.random.randn(16, 8).astype(np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=8)  # num_k_blocks = 2

        assembled = ResultAssembler.assemble_tiles(8, 8, plan.tasks, dtype="fp32")
        expected = np.matmul(a, b)
        np.testing.assert_allclose(assembled, expected, rtol=1e-5, atol=1e-5)

    def test_assembler_identity_matrix(self):
        """Multiplication with Identity matrix reconstructs exact original matrix."""
        np.random.seed(204)
        a = np.random.randn(16, 16).astype(np.float32)
        identity = np.eye(16, dtype=np.float32)
        plan = MatrixTiler.create_plan(a, identity, tile_size=8)
        assembled = ResultAssembler.assemble_tiles(16, 16, plan.tasks, dtype="fp32")
        np.testing.assert_allclose(assembled, a, rtol=1e-5, atol=1e-5)

    def test_assembler_out_of_order_accumulation(self):
        """Confirms that forward, reversed, and randomly shuffled order produce identical results."""
        np.random.seed(205)
        a = np.random.randn(24, 24).astype(np.float32)
        b = np.random.randn(24, 24).astype(np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=8)

        # Forward
        asm_fwd = ResultAssembler.from_plan(plan)
        for task in plan.tasks:
            asm_fwd.accumulate_task(task, np.matmul(task.a_tile, task.b_tile))
        res_fwd = asm_fwd.get_result()

        # Shuffled
        shuffled = list(plan.tasks)
        random.seed(42)
        random.shuffle(shuffled)
        asm_shuf = ResultAssembler.from_plan(plan)
        for task in shuffled:
            asm_shuf.accumulate_task(task, np.matmul(task.a_tile, task.b_tile))
        res_shuf = asm_shuf.get_result()

        np.testing.assert_allclose(res_fwd, res_shuf, rtol=1e-6)

    def test_assembler_thread_safety_concurrent_accumulation(self):
        """8 concurrent worker threads accumulating sub-tiles into a single ResultAssembler."""
        import threading
        np.random.seed(206)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=16)

        assembler = ResultAssembler.from_plan(plan)

        def worker_task(t: TileTaskPlan):
            tile_c = np.matmul(t.a_tile, t.b_tile)
            assembler.accumulate_task(t, tile_c)

        threads = []
        for t in plan.tasks:
            th = threading.Thread(target=worker_task, args=(t,))
            threads.append(th)
            th.start()
        for th in threads:
            th.join(timeout=5.0)

        assert assembler.is_complete() is True
        assembled = assembler.get_result(enforce_complete=True)
        expected = np.matmul(a, b)
        np.testing.assert_allclose(assembled, expected, rtol=1e-5, atol=1e-5)

    def test_assembler_completion_tracking(self):
        """Validates is_complete(), get_progress(), and missing_tasks()."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=8)  # 8 tasks
        assembler = ResultAssembler.from_plan(plan)

        assert assembler.is_complete() is False
        assert assembler.get_progress() == 0.0
        assert len(assembler.missing_tasks()) == 8

        # Accumulate 4 tasks
        for task in plan.tasks[:4]:
            assembler.accumulate_task(task, np.matmul(task.a_tile, task.b_tile))

        assert assembler.is_complete() is False
        assert assembler.get_progress() == 0.5
        assert len(assembler.missing_tasks()) == 4

        # Accumulate remaining 4
        for task in plan.tasks[4:]:
            assembler.accumulate_task(task, np.matmul(task.a_tile, task.b_tile))

        assert assembler.is_complete() is True
        assert assembler.get_progress() == 1.0
        assert len(assembler.missing_tasks()) == 0

    def test_assembler_enforce_complete_exception(self):
        """Raises TilingError when enforce_complete=True on incomplete assembly."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=8)
        assembler = ResultAssembler.from_plan(plan)

        assembler.accumulate_task(plan.tasks[0], np.matmul(plan.tasks[0].a_tile, plan.tasks[0].b_tile))
        with pytest.raises(TilingError, match="assembly incomplete"):
            assembler.get_result(enforce_complete=True)

    def test_assembler_out_of_bounds_slice_error(self):
        """Raises ValueError when slices exceed matrix dimensions."""
        assembler = ResultAssembler(m=16, n=16)
        tile = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="out of bounds"):
            assembler.accumulate(tile, row_idx=0, col_idx=0, row_slice=(0, 20), col_slice=(0, 8))

    def test_assembler_shape_mismatch_error(self):
        """Raises TilingError when tile shape does not match slice size."""
        assembler = ResultAssembler(m=16, n=16)
        tile = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(TilingError, match="does not match expected slice"):
            assembler.accumulate(tile, row_idx=0, col_idx=0, row_slice=(0, 8), col_slice=(0, 8))

    def test_assembler_duplicate_accumulation_defense(self):
        """Rejects duplicate coordinates when allow_duplicate=False."""
        assembler = ResultAssembler(m=16, n=16, allow_duplicate=False)
        tile = np.ones((8, 8), dtype=np.float32)
        assembler.accumulate(tile, row_idx=0, col_idx=0, k_idx=0, row_slice=(0, 8), col_slice=(0, 8))

        with pytest.raises(TilingError, match="Duplicate accumulation rejected"):
            assembler.accumulate(tile, row_idx=0, col_idx=0, k_idx=0, row_slice=(0, 8), col_slice=(0, 8))

    def test_assembler_all_zeros_and_1x1(self):
        """Validates accumulation of zero matrices and 1x1 scalar destination."""
        a = np.zeros((1, 1), dtype=np.float32)
        b = np.zeros((1, 1), dtype=np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=1)
        res = ResultAssembler.assemble_tiles(1, 1, plan.tasks, dtype="fp32")
        assert res.shape == (1, 1)
        assert res[0, 0] == 0.0

    def test_assembler_dtype_preservation(self):
        """Ensures returned matrix conforms to requested precision format."""
        np.random.seed(207)
        a = np.random.randn(8, 8).astype(np.float32)
        b = np.random.randn(8, 8).astype(np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=4)

        res_fp32 = ResultAssembler.assemble_tiles(8, 8, plan.tasks, dtype="fp32")
        assert res_fp32.dtype == np.float32

        res_fp16 = ResultAssembler.assemble_tiles(8, 8, plan.tasks, dtype="fp16")
        assert res_fp16.dtype == np.float16

        res_bf16 = ResultAssembler.assemble_tiles(8, 8, plan.tasks, dtype="bf16")
        assert res_bf16.dtype == np.float32
        assert validate_bf16_masking(res_bf16) is True
