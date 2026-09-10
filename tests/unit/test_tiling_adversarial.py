"""
Adversarial Stress Test Suite for Matrix Tiling, Precision Emulation, and Result Assembly.

Authored by Challenger M2 1 (teamwork_preview_challenger_sdk_m2_1).
Empirically stress-tests:
1. Matrix partitioning across extreme shapes (1x1, 1x1024, 1024x1, prime dimensions 17x23, 67x31),
   remainder slicing, non-overlap, full index coverage, and boundary GFLOPs accuracy.
2. Precision and BF16 emulation: IEEE special floats (NaN, +Inf, -Inf, +0.0, -0.0, subnormals),
   bitwise masking (u32 & 0xFFFF0000) and bit 22 quiet NaN restoration.
3. Result Assembler: concurrent out-of-order accumulation across 20+ threads, duplicate rejection,
   multi-thread race duplicate rejection, and numerical accuracy against NumPy reference for FP32, FP16, and BF16.
"""

from __future__ import annotations

import concurrent.futures
import math
import random
import threading
from typing import List, Set, Tuple

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
# SUITE 1: Matrix Partitioning, Slicing & GFLOPs Boundary Stress Tests
# =============================================================================

class TestPartitioningAdversarial:
    """Stress-tests partitioning across extreme geometries and asserts exact coverage invariants."""

    @pytest.mark.parametrize(
        "m, k, n, tile_size",
        [
            (1, 1, 1, 1),
            (1, 1, 1, 1024),
            (1, 1024, 1, 128),      # Vector dot product
            (1024, 1, 1, 128),      # Column slice
            (1, 1, 1024, 128),      # Row slice
            (1024, 1, 1024, 256),   # Outer product
            (17, 23, 31, 7),        # Small primes
            (67, 31, 43, 11),       # Primes with irregular tile
            (100, 100, 100, 33),    # Odd tile size with remainders
            (500, 300, 700, 128),   # Large non-multiples
            (3, 17, 9, 5),          # Narrow rectangular primes
        ],
    )
    def test_partitioning_exact_coverage_no_overlap(self, m: int, k: int, n: int, tile_size: int):
        """
        Oracle verification:
        1. Union of row slices covers [0, m) exactly without gap or overlap.
        2. Union of col slices covers [0, n) exactly without gap or overlap.
        3. Union of k slices covers [0, k) exactly without gap or overlap.
        4. Every element of A(m, k) and B(k, n) is correctly accessed.
        5. Total tasks == num_m_blocks * num_k_blocks * num_n_blocks.
        """
        a = np.arange(m * k, dtype=np.float32).reshape((m, k))
        b = np.arange(k * n, dtype=np.float32).reshape((k, n))

        plan = MatrixTiler.create_plan(a, b, tile_size=tile_size)

        num_m = math.ceil(m / tile_size)
        num_k = math.ceil(k / tile_size)
        num_n = math.ceil(n / tile_size)

        assert plan.num_m_blocks == num_m
        assert plan.num_k_blocks == num_k
        assert plan.num_n_blocks == num_n
        assert plan.total_tasks == num_m * num_k * num_n
        assert len(plan.tasks) == plan.total_tasks

        # Verify row partition partition coverage
        row_intervals = sorted(list({t.row_slice for t in plan.tasks}))
        assert len(row_intervals) == num_m
        curr = 0
        for start, end in row_intervals:
            assert start == curr, f"Gap or overlap in row slice: expected {curr}, got {start}"
            assert end <= m
            curr = end
        assert curr == m, f"Row coverage did not reach m={m}, ended at {curr}"

        # Verify col partition coverage
        col_intervals = sorted(list({t.col_slice for t in plan.tasks}))
        assert len(col_intervals) == num_n
        curr = 0
        for start, end in col_intervals:
            assert start == curr, f"Gap or overlap in col slice: expected {curr}, got {start}"
            assert end <= n
            curr = end
        assert curr == n, f"Col coverage did not reach n={n}, ended at {curr}"

        # Verify k partition coverage
        k_intervals = sorted(list({t.k_slice for t in plan.tasks}))
        assert len(k_intervals) == num_k
        curr = 0
        for start, end in k_intervals:
            assert start == curr, f"Gap or overlap in k slice: expected {curr}, got {start}"
            assert end <= k
            curr = end
        assert curr == k, f"K coverage did not reach k={k}, ended at {curr}"

        # Verify element-level fidelity: every tile matches original slice
        for task in plan.tasks:
            rs, re = task.row_slice
            cs, ce = task.col_slice
            ks, ke = task.k_slice
            np.testing.assert_array_equal(task.a_tile, a[rs:re, ks:ke])
            np.testing.assert_array_equal(task.b_tile, b[ks:ke, cs:ce])

    @pytest.mark.parametrize(
        "m, k, n, tile_size",
        [
            (1, 1, 1, 1),
            (17, 23, 31, 7),
            (67, 31, 43, 13),
            (128, 128, 128, 32),
            (300, 200, 500, 64),
        ],
    )
    def test_gflops_conservation_law(self, m: int, k: int, n: int, tile_size: int):
        """
        Mathematical invariant:
        Sum of individual boundary tile GFLOPs MUST strictly conserve total GFLOPs:
        sum_i sum_j sum_p (2 * M_i * K_p * N_j) / 10^9 == (2 * M * K * N) / 10^9.
        """
        a = np.zeros((m, k), dtype=np.float32)
        b = np.zeros((k, n), dtype=np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=tile_size)

        expected_total_gflops = (2.0 * float(m) * float(k) * float(n)) / 1e9

        summed_task_gflops = sum(t.estimated_gflops for t in plan.tasks)
        assert math.isclose(plan.total_gflops, expected_total_gflops, rel_tol=1e-9)
        assert math.isclose(summed_task_gflops, expected_total_gflops, rel_tol=1e-9)

        # Also verify individual task GFLOPs formula
        for t in plan.tasks:
            expected_task_gflops = (2.0 * float(t.m_i) * float(t.k_j) * float(t.n_k)) / 1e9
            assert math.isclose(t.estimated_gflops, expected_task_gflops, rel_tol=1e-9)


# =============================================================================
# SUITE 2: Precision Emulation & BF16 Quiet NaN Preservation Stress Tests
# =============================================================================

class TestPrecisionAdversarial:
    """Stress-tests IEEE 754 float32 and bfloat16 bitwise masking edge cases."""

    def test_bf16_quiet_nan_restoration_exhaustive_bit_sweep(self):
        """
        Adversarial attack on naive masking:
        Construct NaNs where mantissa bits 22..16 are 0, but bits 15..0 have arbitrary 1s.
        A naive u32 & 0xFFFF0000 clears the entire mantissa, falsely converting NaN to +/- Infinity.
        truncate_to_bf16 MUST restore quiet NaN bit 22 (0x00400000) for ALL cases.
        """
        # Test positive NaNs with varying lower mantissa bits
        test_patterns = [
            0x7F800001,  # bit 0
            0x7F800002,  # bit 1
            0x7F8000FF,  # lower 8 bits
            0x7F808000,  # bit 15
            0x7F80FFFF,  # all lower 16 bits set, bits 22..16 zero
            0xFF800001,  # negative NaN bit 0
            0xFF800010,  # negative NaN bit 4
            0xFF80FFFF,  # negative NaN all lower 16 bits set
        ]

        for raw_u32 in test_patterns:
            f32_arr = np.array([raw_u32], dtype=np.uint32).view(np.float32)
            assert np.isnan(f32_arr[0]), f"Test pattern {hex(raw_u32)} is not NaN"

            bf16_res = truncate_to_bf16(f32_arr)
            # Must remain NaN!
            assert np.isnan(bf16_res[0]), f"Failed NaN preservation for {hex(raw_u32)} -> {hex(bf16_res.view(np.uint32)[0])}"

            res_u32 = bf16_res.view(np.uint32)[0]
            # Bit 22 (0x00400000) must be set
            assert (res_u32 & np.uint32(0x00400000)) != 0, f"Quiet NaN bit 22 not set for {hex(raw_u32)}"
            # Lower 16 bits must be 0
            assert (res_u32 & np.uint32(0x0000FFFF)) == 0, f"Lower 16 bits not zero for {hex(raw_u32)}"

            # Sign bit must be preserved
            expected_sign = bool(raw_u32 & 0x80000000)
            actual_sign = bool(res_u32 & 0x80000000)
            assert actual_sign == expected_sign, f"Sign bit mutated for {hex(raw_u32)}"

    def test_bf16_preexisting_quiet_nan_intact(self):
        """Pre-existing quiet NaNs (bit 22 already set) remain quiet NaNs without mutation."""
        pre_quiet_patterns = [
            0x7FC00000,  # Standard positive quiet NaN
            0x7FE01234,  # Quiet NaN with additional bits
            0xFFC00000,  # Standard negative quiet NaN
            0xFFF12345,  # Negative quiet NaN
        ]
        for raw_u32 in pre_quiet_patterns:
            f32_arr = np.array([raw_u32], dtype=np.uint32).view(np.float32)
            bf16_res = truncate_to_bf16(f32_arr)
            assert np.isnan(bf16_res[0])
            res_u32 = bf16_res.view(np.uint32)[0]
            assert (res_u32 & np.uint32(0x00400000)) != 0
            assert (res_u32 & np.uint32(0x0000FFFF)) == 0

    def test_bf16_infinity_isolation(self):
        """Positive and negative infinities remain infinities and do not become NaNs."""
        infs = np.array([np.inf, -np.inf, float("inf"), float("-inf")], dtype=np.float32)
        bf16_infs = truncate_to_bf16(infs)

        assert np.isposinf(bf16_infs[0])
        assert np.isneginf(bf16_infs[1])
        assert np.isposinf(bf16_infs[2])
        assert np.isneginf(bf16_infs[3])

        u32 = bf16_infs.view(np.uint32)
        assert u32[0] == 0x7F800000
        assert u32[1] == 0xFF800000

    def test_bf16_signed_zeros(self):
        """+0.0 and -0.0 maintain signbit distinction and zero value."""
        zeros = np.array([0.0, -0.0], dtype=np.float32)
        bf16_zeros = truncate_to_bf16(zeros)

        assert bf16_zeros[0] == 0.0
        assert bf16_zeros[1] == 0.0
        assert np.signbit(bf16_zeros[0]) is False or np.signbit(bf16_zeros[0]) == 0
        assert np.signbit(bf16_zeros[1]) is True or np.signbit(bf16_zeros[1]) == 1

        u32 = bf16_zeros.view(np.uint32)
        assert u32[0] == 0x00000000
        assert u32[1] == 0x80000000

    def test_bf16_subnormal_underflow_no_nan(self):
        """Subnormal numbers near machine epsilon underflow cleanly to zero or valid subnormal."""
        subnormals = np.array([1e-40, -1e-40, 1e-44, 1e-45, -1e-45], dtype=np.float32)
        bf16_res = truncate_to_bf16(subnormals)

        assert not np.any(np.isnan(bf16_res))
        assert validate_bf16_masking(bf16_res) is True


# =============================================================================
# SUITE 3: Result Assembler Multi-Threaded & Concurrency Stress Tests
# =============================================================================

class TestAssemblerAdversarial:
    """Stress-tests thread safety, duplicate rejection, and high concurrency (20+ threads)."""

    def test_assembler_concurrent_accumulation_32_threads(self):
        """
        Stress test: 32 concurrent worker threads accumulating out-of-order sub-tiles.
        Matrix shape (64, 64) with tile_size=16 -> 4x4x4 = 64 tasks.
        Threads randomly pick tasks and accumulate concurrently.
        """
        np.random.seed(301)
        a = np.random.randn(64, 64).astype(np.float32)
        b = np.random.randn(64, 64).astype(np.float32)
        plan = MatrixTiler.create_plan(a, b, tile_size=16)
        assert plan.total_tasks == 64

        assembler = ResultAssembler.from_plan(plan)

        # Precompute individual tile GEMMs
        computed_tiles = [
            (task, np.matmul(task.a_tile.astype(np.float64), task.b_tile.astype(np.float64)).astype(np.float32))
            for task in plan.tasks
        ]

        # Shuffle execution order
        random.seed(999)
        random.shuffle(computed_tiles)

        def worker_task(item):
            t, res = item
            assembler.accumulate_task(t, res)

        # Run with 32 concurrent threads in ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(worker_task, item) for item in computed_tiles]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # Re-raises any exception

        assert assembler.is_complete() is True
        assert assembler.get_progress() == 1.0
        assert len(assembler.missing_tasks()) == 0

        assembled = assembler.get_result(enforce_complete=True)
        expected = np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(assembled, expected, rtol=1e-5, atol=1e-5)

    def test_assembler_concurrent_duplicate_rejection_race(self):
        """
        Stress test: 20 threads simultaneously race to accumulate the EXACT same tile coordinate.
        Invariant: Exactly 1 thread must succeed; the other 19 threads MUST raise TilingError.
        """
        assembler = ResultAssembler(m=32, n=32, allow_duplicate=False)
        tile = np.ones((16, 16), dtype=np.float32)

        barrier = threading.Barrier(20)
        successes: List[int] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def racer(thread_id: int):
            barrier.wait()  # Synchronize start so all 20 hit assemble at the exact same instant
            try:
                assembler.accumulate(
                    tile_result=tile,
                    row_idx=0,
                    col_idx=0,
                    k_idx=0,
                    row_slice=(0, 16),
                    col_slice=(0, 16),
                    task_id=f"race_task_{thread_id}",
                )
                with lock:
                    successes.append(thread_id)
            except TilingError as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=racer, args=(i,)) for i in range(20)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5.0)

        assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"
        assert len(errors) == 19, f"Expected 19 duplicate rejections, got {len(errors)}"
        for err in errors:
            assert "Duplicate accumulation rejected" in str(err)

    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_assembler_end_to_end_precision_tolerances(self, dtype: str):
        """
        End-to-end verification:
        Tiling -> Sub-tile GEMM -> ResultAssembler -> Accuracy check against NumPy reference.
        Validates FP32, FP16, and BF16 numerical tolerance.
        """
        np.random.seed(401)
        m, k, n = 32, 48, 32
        a = np.random.uniform(0.1, 2.0, size=(m, k)).astype(np.float32)
        b = np.random.uniform(0.1, 2.0, size=(k, n)).astype(np.float32)

        plan = MatrixTiler.create_plan(a, b, tile_size=16, dtype=dtype)
        assembler = ResultAssembler.from_plan(plan)

        for task in plan.tasks:
            # Emulate worker GEMM in target precision
            tile_c = emulate_gemm(task.a_tile, task.b_tile, dtype=dtype)
            assembler.accumulate_task(task, tile_c)

        result = assembler.get_result(enforce_complete=True)

        # Reference GEMM
        ref = emulate_gemm(a, b, dtype=dtype)

        if dtype == "fp32":
            np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-5)
            assert result.dtype == np.float32
        elif dtype == "fp16":
            np.testing.assert_allclose(result, ref, rtol=5e-3, atol=5e-3)
            assert result.dtype == np.float16
        elif dtype == "bf16":
            np.testing.assert_allclose(result, ref, rtol=2e-2, atol=2e-2)
            assert result.dtype == np.float32
            assert validate_bf16_masking(result) is True
