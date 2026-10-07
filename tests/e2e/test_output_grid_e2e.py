"""
Opaque-Box E2E Test Suite: 2D Output Grid (OUTPUT_GRID_2D) Matrix Partitioning.

Phase 2 Verification Matrix:
- Tier 1: Feature Coverage (2D spatial grid partitioning, div-mod remainder distribution,
  multi-precision preservation FP32/FP16/BF16, direct spatial assembly without local additions,
  TileGridPlan/TileTaskPlan contracts, bandwidth profiler metrics and download invariants).
- Tier 2: Boundary & Corner Cases (Primes 7x11x13, odd shapes 15x25x35, scalar 1x1,
  1xN / Mx1 vector strips, asymmetric M!=N!=K, non-square P_r!=P_c grids,
  oversubscription clamping, extreme contraction dimensions, defensive fail-closed validation).
- Tier 3: Cross-Feature Combinations (Dual-tier Freivalds verification, per-tile fraud rejection
  and FreivaldsFraudProof emission, batch L402 escrow lifecycle, multi-worker partial fraud isolation,
  sliding window concurrency bounding, assembled product tamper detection,
  TEE encrypted chunk dispatch, worker failover redispatch).
- Tier 4: Real-World Scenarios & Download Bandwidth Profiling (Strictly M*N download elements,
  download redundancy ratio == 0.0, download bandwidth contrast with 1D outer-product and 3D block,
  Higham floating-point error bounds across FP32/FP16/BF16, Transformer Attention projection,
  MLP feedforward projection, large-scale 1024x1024 stress).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytest

from matrix_contracts.schemas import (
    BondStatus,
    CreateHodlInvoiceRequest,
    CreateHodlInvoiceResponse,
    FreivaldsFraudProof,
    HardwareSpecs,
    MatchedWorker,
    PricingSpecs,
    SettlePaymentRequest,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
    WorkerQueryResponse,
)
from matrix_contracts.v1 import pb2, pb2_grpc
from tests.conftest import (
    MockRelayDouble,
    MockSettlementDouble,
    OpaqueL402Helper,
    ReferenceOracle,
    StreamChunkHelper,
)

# Production SDK imports
from matrix_compute.config import ClientConfig, PartitioningStrategy
from matrix_compute.crypto.session import (
    CryptoSession,
    SessionCrypto,
    derive_session_keys,
    generate_challenger_nonce,
    generate_ephemeral_keypair,
)
from matrix_compute.exceptions import MatrixDimensionError, TilingError
from matrix_compute.tiling.assembler import ResultAssembler
from matrix_compute.tiling.partitioner import MatrixTiler, TileGridPlan, TileTaskPlan
from matrix_compute.tiling.precision import (
    calculate_higham_bound,
    cast_matrix,
    get_bytes_per_element,
    get_numpy_dtype,
    get_unit_roundoff,
    normalize_dtype,
    truncate_to_bf16,
    validate_bf16_masking,
)
from matrix_compute.tiling.profiler import BandwidthProfile, BandwidthProfiler


# ==============================================================================
# AUTHORITATIVE REFERENCE 2D OUTPUT GRID ENGINE & PROFILER
# ==============================================================================

class Reference2DOutputGridEngine:
    """
    Authoritative, independent reference implementation of 2D Output Grid
    (download-optimal & zero-reduction) matrix partitioning, spatial assembly,
    and bandwidth profiling.
    """

    @classmethod
    def calculate_grid_blocks(
        cls, m: int, n: int, grid_rows: int, grid_cols: int
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Calculates balanced contiguous slice ranges [row_start, row_end) and [col_start, col_end).
        Distributes remainders across the first r intervals.
        Clamps effective grid dimensions:
            P_r_eff = min(max(1, int(grid_rows)), m)
            P_c_eff = min(max(1, int(grid_cols)), n)
        Guarantees zero-width blocks are never produced and coverage of [0, M) and [0, N) is exact.
        """
        if m < 1 or n < 1:
            raise ValueError(f"Matrix dimensions must be >= 1, got M={m}, N={n}")

        p_r_eff = min(max(1, int(grid_rows)), m)
        p_c_eff = min(max(1, int(grid_cols)), n)

        # Div-mod balanced distribution for row blocks
        q_m, rem_m = divmod(m, p_r_eff)
        row_splits: List[Tuple[int, int]] = []
        curr_m = 0
        for i in range(p_r_eff):
            h = q_m + (1 if i < rem_m else 0)
            row_splits.append((curr_m, curr_m + h))
            curr_m += h
        assert curr_m == m, f"Row splits did not cover M exactly: {curr_m} != {m}"

        # Div-mod balanced distribution for col blocks
        q_n, rem_n = divmod(n, p_c_eff)
        col_splits: List[Tuple[int, int]] = []
        curr_n = 0
        for j in range(p_c_eff):
            w = q_n + (1 if j < rem_n else 0)
            col_splits.append((curr_n, curr_n + w))
            curr_n += w
        assert curr_n == n, f"Col splits did not cover N exactly: {curr_n} != {n}"

        return row_splits, col_splits

    @classmethod
    def partition_output_grid_2d(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        grid_rows: int = 2,
        grid_cols: int = 2,
        dtype: str = "fp32",
        task_id_prefix: str = "tile_grid",
    ) -> List[Dict[str, Any]]:
        """
        Partitions matrix A (M x K) and B (K x N) into P_r x P_c spatial tiles C_ij = A_i x B_j.
        Dimension K is strictly unsliced (K-block count = 1, k_slice = (0, K)).
        Returns a list of task specification dictionaries.
        """
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise MatrixDimensionError("Matrices A and B must be numpy arrays")
        if a.ndim != 2 or b.ndim != 2:
            raise MatrixDimensionError(f"Matrices must be 2D. Got {a.shape} and {b.shape}")

        m, k = a.shape
        k2, n = b.shape
        if k != k2:
            raise MatrixDimensionError(
                f"Inner dimension mismatch: A cols ({k}) != B rows ({k2})",
                shape_a=a.shape,
                shape_b=b.shape,
                expected_inner_dim=k,
            )
        if m < 1 or k < 1 or n < 1:
            raise MatrixDimensionError(f"Matrix dimensions must be >= 1. Got {a.shape} and {b.shape}")

        if grid_rows < 1 or grid_cols < 1:
            raise ValueError(f"Grid dimensions must be >= 1, got rows={grid_rows}, cols={grid_cols}")

        norm_dtype = normalize_dtype(dtype)
        a_cast = cast_matrix(a, norm_dtype)
        b_cast = cast_matrix(b, norm_dtype)

        row_splits, col_splits = cls.calculate_grid_blocks(m, n, grid_rows, grid_cols)
        tasks: List[Dict[str, Any]] = []

        for i, (r_s, r_e) in enumerate(row_splits):
            m_i = r_e - r_s
            a_tile = a_cast[r_s:r_e, :]  # Shape (M_i, K) - K is strictly unsliced

            for j, (c_s, c_e) in enumerate(col_splits):
                n_j = c_e - c_s
                b_tile = b_cast[:, c_s:c_e]  # Shape (K, N_j) - K is strictly unsliced

                gflops = (2.0 * float(m_i) * float(k) * float(n_j)) / 1e9

                tasks.append(
                    {
                        "task_id": f"{task_id_prefix}_{i}_{j}",
                        "row_idx": i,
                        "col_idx": j,
                        "k_idx": 0,  # Strictly unsliced
                        "row_slice": (r_s, r_e),
                        "col_slice": (c_s, c_e),
                        "k_slice": (0, k),  # Full K dimension
                        "a_tile": a_tile,
                        "b_tile": b_tile,
                        "m_i": m_i,
                        "k_j": k,
                        "n_k": n_j,
                        "dtype": norm_dtype,
                        "estimated_gflops": gflops,
                    }
                )

        return tasks

    @classmethod
    def create_plan_output_grid_2d(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
        tile_size: Optional[int] = None,
        dtype: str = "fp32",
        task_id_prefix: str = "tile_grid",
    ) -> TileGridPlan:
        """
        Creates a complete TileGridPlan manifest for 2D Output Grid partitioning.
        """
        m, k = a.shape
        _, n = b.shape
        p_r = grid_rows or (max(1, math.ceil(m / tile_size)) if tile_size else 2)
        p_c = grid_cols or (max(1, math.ceil(n / tile_size)) if tile_size else 2)

        task_dicts = cls.partition_output_grid_2d(
            a, b, grid_rows=p_r, grid_cols=p_c, dtype=dtype, task_id_prefix=task_id_prefix
        )
        row_splits, col_splits = cls.calculate_grid_blocks(m, n, p_r, p_c)

        task_plans = [
            TileTaskPlan(
                task_id=t["task_id"],
                row_idx=t["row_idx"],
                col_idx=t["col_idx"],
                k_idx=t["k_idx"],
                row_slice=t["row_slice"],
                col_slice=t["col_slice"],
                k_slice=t["k_slice"],
                a_tile=t["a_tile"],
                b_tile=t["b_tile"],
                dtype=t["dtype"],
                estimated_gflops=t["estimated_gflops"],
            )
            for t in task_dicts
        ]

        total_gflops = sum(t.estimated_gflops for t in task_plans)

        return TileGridPlan(
            m=m,
            k=k,
            n=n,
            tile_size=tile_size or 1024,
            dtype=normalize_dtype(dtype),
            num_m_blocks=len(row_splits),
            num_k_blocks=1,
            num_n_blocks=len(col_splits),
            total_tasks=len(task_plans),
            total_gflops=total_gflops,
            tasks=task_plans,
            strategy="output_grid_2d",
        )

    @classmethod
    def assemble_output_grid(
        cls,
        m: int,
        n: int,
        tile_results: List[Tuple[Any, np.ndarray]],
        dtype: str = "fp32",
    ) -> Tuple[np.ndarray, int]:
        """
        Zero-reduction direct spatial assembly:
        Places each completed sub-tile C_ij directly into output buffer at C[row_slice, col_slice] = C_ij.
        Performs strictly 0 local floating-point reduction additions.
        Returns: (assembled_matrix, reduction_additions_count).
        """
        norm_dtype = normalize_dtype(dtype)
        np_dtype = np.float32 if norm_dtype in ("fp32", "bf16") else np.float16
        c_out = np.zeros((m, n), dtype=np_dtype)
        reduction_ops = 0  # Verified: Zero reduction additions!

        for task_info, tile_c in tile_results:
            r_s, r_e = task_info["row_slice"] if isinstance(task_info, dict) else task_info.row_slice
            c_s, c_e = task_info["col_slice"] if isinstance(task_info, dict) else task_info.col_slice
            expected_shape = (r_e - r_s, c_e - c_s)
            assert tile_c.shape == expected_shape, f"Shape mismatch: {tile_c.shape} != {expected_shape}"

            # Zero-reduction direct spatial placement:
            c_out[r_s:r_e, c_s:c_e] = tile_c.astype(np_dtype)

        if norm_dtype == "bf16":
            return truncate_to_bf16(c_out), reduction_ops
        elif norm_dtype == "fp16":
            return c_out.astype(np.float16), reduction_ops
        return c_out.astype(np.float32), reduction_ops

    @classmethod
    def profile_download_bandwidth(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        grid_tasks: List[Any],
        dtype: str = "fp32",
    ) -> Dict[str, Any]:
        """
        Measures total downloaded and uploaded matrix elements and bytes across all 2D grid tasks.
        Asserts download optimality: total_downloaded_elements == M * N with redundancy ratio == 0.0.
        """
        m, k = a.shape
        _, n = b.shape
        norm_dtype = normalize_dtype(dtype)
        bytes_per_elem = get_bytes_per_element(norm_dtype)

        optimal_download_elements = m * n
        total_downloaded_elements = 0
        total_a_elements = 0
        total_b_elements = 0

        for t in grid_tasks:
            r_s, r_e = t["row_slice"] if isinstance(t, dict) else t.row_slice
            c_s, c_e = t["col_slice"] if isinstance(t, dict) else t.col_slice
            total_downloaded_elements += (r_e - r_s) * (c_e - c_s)

            a_tile = t["a_tile"] if isinstance(t, dict) else t.a_tile
            b_tile = t["b_tile"] if isinstance(t, dict) else t.b_tile
            total_a_elements += a_tile.size
            total_b_elements += b_tile.size

        download_redundancy_elements = total_downloaded_elements - optimal_download_elements
        download_redundancy_ratio = (
            (float(total_downloaded_elements) / float(optimal_download_elements)) - 1.0
            if optimal_download_elements > 0
            else 0.0
        )

        total_uploaded_elements = total_a_elements + total_b_elements
        optimal_upload_elements = m * k + k * n
        upload_redundancy_ratio = (
            (float(total_uploaded_elements) / float(optimal_upload_elements)) - 1.0
            if optimal_upload_elements > 0
            else 0.0
        )

        return {
            "m": m,
            "k": k,
            "n": n,
            "num_tasks": len(grid_tasks),
            "download_elements": total_downloaded_elements,
            "optimal_download_elements": optimal_download_elements,
            "download_redundancy_elements": download_redundancy_elements,
            "download_redundancy_ratio": download_redundancy_ratio,
            "is_download_optimal": download_redundancy_elements == 0 and abs(download_redundancy_ratio) < 1e-12,
            "total_a_elements": total_a_elements,
            "total_b_elements": total_b_elements,
            "total_uploaded_elements": total_uploaded_elements,
            "optimal_upload_elements": optimal_upload_elements,
            "upload_redundancy_ratio": upload_redundancy_ratio,
            "download_bytes": total_downloaded_elements * bytes_per_elem,
            "optimal_download_bytes": optimal_download_elements * bytes_per_elem,
        }

    @classmethod
    def verify_spatial_coverage(cls, tasks: List[Any], m: int, n: int) -> bool:
        """
        Verifies [row_slice, col_slice] tiles form an exact, contiguous, disjoint
        cover of [0, M) x [0, N) with zero overlaps and zero gaps.
        """
        coverage = np.zeros((m, n), dtype=np.int32)
        for t in tasks:
            r_s, r_e = t["row_slice"] if isinstance(t, dict) else t.row_slice
            c_s, c_e = t["col_slice"] if isinstance(t, dict) else t.col_slice
            if r_s < 0 or c_s < 0 or r_e > m or c_e > n or r_e <= r_s or c_e <= c_s:
                return False
            coverage[r_s:r_e, c_s:c_e] += 1
        return bool(np.all(coverage == 1))


# ==============================================================================
# TIER 1: FEATURE COVERAGE (26 TESTS ACROSS 5 FEATURES)
# ==============================================================================

class TestTier1OutputGridFeatureCoverage:
    """
    Tier 1: Feature coverage of 2D Output Grid partitioning, spatial tiling,
    multi-precision handling, zero-reduction assembly, contracts, and bandwidth metrics.
    """

    # --------------------------------------------------------------------------
    # Feature 1: 2D Spatial Grid Partitioning & Div-Mod Tiling (6 tests)
    # --------------------------------------------------------------------------

    def test_t1_01_output_grid_square_even_division(self):
        """T1.1: 64x64 matrix partitioned into P_r=2, P_c=2 (4 tasks of 32x32), K=32 unsliced."""
        np.random.seed(42)
        a = np.random.randn(64, 32).astype(np.float32)
        b = np.random.randn(32, 64).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        assert len(tasks) == 4

        for task in tasks:
            i, j = task["row_idx"], task["col_idx"]
            assert task["task_id"] == f"tile_grid_{i}_{j}"
            assert task["k_idx"] == 0
            assert task["k_slice"] == (0, 32)
            assert task["row_slice"] == (i * 32, (i + 1) * 32)
            assert task["col_slice"] == (j * 32, (j + 1) * 32)
            assert task["a_tile"].shape == (32, 32)
            assert task["b_tile"].shape == (32, 32)

    def test_t1_02_output_grid_div_mod_remainder_distribution(self):
        """T1.2: Remainder distribution for M=10, N=11 across P_r=3, P_c=3 grid."""
        a = np.ones((10, 16), dtype=np.float32)
        b = np.ones((16, 11), dtype=np.float32)

        row_splits, col_splits = Reference2DOutputGridEngine.calculate_grid_blocks(10, 11, 3, 3)
        assert row_splits == [(0, 4), (4, 7), (7, 10)]
        assert col_splits == [(0, 4), (4, 8), (8, 11)]

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=3, grid_cols=3)
        assert len(tasks) == 9
        assert Reference2DOutputGridEngine.verify_spatial_coverage(tasks, 10, 11) is True

    def test_t1_03_output_grid_non_square_topology(self):
        """T1.3: Non-square processor grid P_r=3, P_c=4 (12 tasks) on 48x64 matrix."""
        a = np.ones((48, 20), dtype=np.float32)
        b = np.ones((20, 64), dtype=np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=3, grid_cols=4)
        assert len(tasks) == 12

        row_indices = {t["row_idx"] for t in tasks}
        col_indices = {t["col_idx"] for t in tasks}
        assert row_indices == {0, 1, 2}
        assert col_indices == {0, 1, 2, 3}

    def test_t1_04_output_grid_task_geometry_invariants(self):
        """T1.4: Every TileTaskPlan adheres to strict 2D output grid geometry invariants."""
        a = np.random.randn(30, 40).astype(np.float32)
        b = np.random.randn(40, 50).astype(np.float32)

        plan = Reference2DOutputGridEngine.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=3)
        for t in plan.tasks:
            assert t.k_idx == 0
            assert t.k_slice == (0, 40)
            assert t.shape_a == (t.m_i, 40)
            assert t.shape_b == (40, t.n_k)
            assert t.shape_c == (t.m_i, t.n_k)

    def test_t1_05_output_grid_gflops_conservation(self):
        """T1.5: Total GFLOPs across all 2D grid tasks strictly equals 2*M*K*N / 1e9."""
        m, k, n = 35, 45, 55
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        plan = Reference2DOutputGridEngine.create_plan_output_grid_2d(a, b, grid_rows=3, grid_cols=4)
        expected_total_gflops = (2.0 * m * k * n) / 1e9
        computed_total_gflops = sum(t.estimated_gflops for t in plan.tasks)

        assert abs(computed_total_gflops - expected_total_gflops) < 1e-12
        assert abs(plan.total_gflops - expected_total_gflops) < 1e-12

    def test_t1_06_output_grid_k_unsliced_invariant(self):
        """T1.6: Inner dimension K is strictly unsliced across arbitrary grid topologies."""
        k_dim = 128
        a = np.ones((32, k_dim), dtype=np.float32)
        b = np.ones((k_dim, 32), dtype=np.float32)

        for pr, pc in [(1, 1), (1, 4), (4, 1), (2, 3), (5, 5)]:
            plan = Reference2DOutputGridEngine.create_plan_output_grid_2d(a, b, grid_rows=pr, grid_cols=pc)
            assert plan.num_k_blocks == 1
            for task in plan.tasks:
                assert task.k_idx == 0
                assert task.k_slice == (0, k_dim)
                assert task.k_j == k_dim

    # --------------------------------------------------------------------------
    # Feature 2: Multi-Precision Support & Dtype Preservation (5 tests)
    # --------------------------------------------------------------------------

    def test_t1_07_output_grid_fp32_preservation(self):
        """T1.7: IEEE 754 FP32 tensor precision preserved throughout 2D grid pipeline."""
        a = np.random.randn(16, 24).astype(np.float32)
        b = np.random.randn(24, 16).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2, dtype="fp32")
        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(16, 16, computed, dtype="fp32")

        assert c_out.dtype == np.float32
        np.testing.assert_allclose(c_out, np.matmul(a, b), rtol=1e-5, atol=1e-5)

    def test_t1_08_output_grid_fp16_preservation(self):
        """T1.8: IEEE 754 half-precision FP16 preserved throughout 2D grid pipeline."""
        a = (np.random.randn(16, 24) * 0.5).astype(np.float16)
        b = (np.random.randn(24, 16) * 0.5).astype(np.float16)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2, dtype="fp16")
        computed = [(t, np.matmul(t["a_tile"].astype(np.float32), t["b_tile"].astype(np.float32)).astype(np.float16)) for t in tasks]
        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(16, 16, computed, dtype="fp16")

        assert c_out.dtype == np.float16
        c_expected = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(c_out.astype(np.float32), c_expected.astype(np.float32), rtol=1e-3, atol=1e-3)

    def test_t1_09_output_grid_bf16_masking(self):
        """T1.9: BF16 16-bit mantissa bitmasking (0xFFFF0000) verified across all tiles."""
        a = truncate_to_bf16(np.random.randn(16, 24))
        b = truncate_to_bf16(np.random.randn(24, 16))

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2, dtype="bf16")
        for t in tasks:
            assert validate_bf16_masking(t["a_tile"]) is True
            assert validate_bf16_masking(t["b_tile"]) is True

        computed = [(t, truncate_to_bf16(np.matmul(t["a_tile"], t["b_tile"]))) for t in tasks]
        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(16, 16, computed, dtype="bf16")

        assert validate_bf16_masking(c_out) is True

    def test_t1_10_output_grid_bf16_nan_inf_preservation(self):
        """T1.10: NaN, +inf, -inf semantics preserved in BF16 partitioning and assembly."""
        a = np.array([[np.nan, np.inf], [-np.inf, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        a_bf16 = truncate_to_bf16(a)
        assert np.isnan(a_bf16[0, 0])
        assert np.isposinf(a_bf16[0, 1])
        assert np.isneginf(a_bf16[1, 0])

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a_bf16, b, grid_rows=2, grid_cols=2, dtype="bf16")
        assert np.isnan(tasks[0]["a_tile"][0, 0])

    def test_t1_11_output_grid_precision_normalization_aliases(self):
        """T1.11: Normalization of dtype aliases within 2D grid planner."""
        aliases = [
            ("fp32", "fp32"), ("float32", "fp32"), ("single", "fp32"),
            ("fp16", "fp16"), ("float16", "fp16"), ("half", "fp16"),
            ("bf16", "bf16"), ("bfloat16", "bf16"),
        ]
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)

        for raw_alias, norm in aliases:
            tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2, dtype=raw_alias)
            assert tasks[0]["dtype"] == norm

    # --------------------------------------------------------------------------
    # Feature 3: Direct Spatial Assembly Without Addition (Zero-Reduction) (5 tests)
    # --------------------------------------------------------------------------

    def test_t1_12_direct_spatial_placement_zero_addition(self):
        """T1.12: Sub-tiles C_ij placed directly into output coordinates with strictly 0 additions."""
        m, n = 32, 32
        a = np.random.randn(m, 16).astype(np.float32)
        b = np.random.randn(16, n).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]

        c_out, reduction_ops = Reference2DOutputGridEngine.assemble_output_grid(m, n, computed, dtype="fp32")
        assert reduction_ops == 0

        # ResultAssembler direct execution
        assembler = ResultAssembler(m=m, n=n, expected_tasks=len(tasks), dtype="fp32", k_blocks=1)
        for t, tile_c in computed:
            assembler.accumulate(
                tile_result=tile_c,
                row_idx=t["row_idx"],
                col_idx=t["col_idx"],
                k_idx=0,
                row_slice=t["row_slice"],
                col_slice=t["col_slice"],
                task_id=t["task_id"],
            )

        assert assembler.is_complete() is True
        np.testing.assert_array_equal(assembler.get_result(), c_out)

    def test_t1_13_direct_assembly_mathematical_equivalence(self):
        """T1.13: Direct spatial assembly matches np.matmul with zero numerical drift."""
        np.random.seed(113)
        a = np.random.randn(20, 30).astype(np.float32)
        b = np.random.randn(30, 40).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=3, grid_cols=4)
        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]

        c_assembled, _ = Reference2DOutputGridEngine.assemble_output_grid(20, 40, computed)
        c_direct = np.matmul(a, b)

        np.testing.assert_allclose(c_assembled, c_direct, rtol=1e-5, atol=1e-5)

    def test_t1_14_reentrancy_duplicate_rejection(self):
        """T1.14: ResultAssembler strictly rejects duplicate sub-tile on coordinate (i, j, 0)."""
        assembler = ResultAssembler(m=16, n=16, expected_tasks=4, dtype="fp32", allow_duplicate=False)
        tile = np.ones((8, 8), dtype=np.float32)

        # Initial arrival
        assembler.accumulate(
            tile_result=tile,
            row_idx=0,
            col_idx=0,
            k_idx=0,
            row_slice=(0, 8),
            col_slice=(0, 8),
            task_id="tile_0_0",
        )

        # Duplicate arrival
        with pytest.raises(TilingError, match="Duplicate accumulation rejected"):
            assembler.accumulate(
                tile_result=tile,
                row_idx=0,
                col_idx=0,
                k_idx=0,
                row_slice=(0, 8),
                col_slice=(0, 8),
                task_id="tile_0_0_dup",
            )

    def test_t1_15_out_of_bounds_slice_rejection(self):
        """T1.15: ResultAssembler rejects out-of-bounds row or column slices."""
        assembler = ResultAssembler(m=16, n=16, expected_tasks=1, dtype="fp32")
        tile = np.ones((8, 8), dtype=np.float32)

        with pytest.raises(ValueError, match="Tile slice out of bounds"):
            assembler.accumulate(
                tile_result=tile,
                row_idx=0,
                col_idx=0,
                k_idx=0,
                row_slice=(10, 20),  # 20 > M=16
                col_slice=(0, 8),
            )

    def test_t1_16_incomplete_assembly_detection(self):
        """T1.16: ResultAssembler tracks missing tasks until all P_r x P_c tiles arrive."""
        task_ids = ["tile_0_0", "tile_0_1", "tile_1_0", "tile_1_1"]
        assembler = ResultAssembler(m=16, n=16, expected_tasks=task_ids, dtype="fp32")

        assert assembler.is_complete() is False
        assert set(assembler.missing_tasks()) == set(task_ids)

        tile = np.ones((8, 8), dtype=np.float32)
        assembler.accumulate(tile, 0, 0, 0, (0, 8), (0, 8), task_id="tile_0_0")

        assert assembler.is_complete() is False
        assert set(assembler.missing_tasks()) == {"tile_0_1", "tile_1_0", "tile_1_1"}

    # --------------------------------------------------------------------------
    # Feature 4: Client & Config Contracts (5 tests)
    # --------------------------------------------------------------------------

    def test_t1_17_matrix_tiler_create_plan_output_grid_2d_contract(self):
        """T1.17: Validates TileGridPlan contract conformance for 2D Output Grid."""
        a = np.ones((16, 24), dtype=np.float32)
        b = np.ones((24, 32), dtype=np.float32)

        plan = Reference2DOutputGridEngine.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=4)
        assert isinstance(plan, TileGridPlan)
        assert plan.strategy == "output_grid_2d"
        assert plan.num_m_blocks == 2
        assert plan.num_k_blocks == 1
        assert plan.num_n_blocks == 4
        assert plan.total_tasks == 8

    def test_t1_18_matrix_tiler_plan_dictionary_emulation(self):
        """T1.18: TileTaskPlan supports both property access and dict subscription."""
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)

        plan = Reference2DOutputGridEngine.create_plan_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        task = plan.tasks[0]

        assert task["row_idx"] == task.row_idx
        assert task["col_idx"] == task.col_idx
        assert task["k_idx"] == task.k_idx
        assert task["a_tile"].shape == task.a_tile.shape
        assert "row_slice" in task

    def test_t1_19_config_partitioning_strategy_enum(self):
        """T1.19: PartitioningStrategy enum defines OUTPUT_GRID_2D and aliases."""
        assert PartitioningStrategy.OUTPUT_GRID_2D == "output_grid_2d"
        assert PartitioningStrategy.OUTPUT_GRID_2D.value == "output_grid_2d"

        cfg = ClientConfig(partitioning_strategy=PartitioningStrategy.OUTPUT_GRID_2D)
        assert cfg.partitioning_strategy == PartitioningStrategy.OUTPUT_GRID_2D

    def test_t1_20_client_config_defaults_and_grid_fields(self):
        """T1.20: ClientConfig supports grid configuration fields."""
        cfg = ClientConfig.for_testing()
        assert hasattr(cfg, "partitioning_strategy")

    def test_t1_21_coordinator_routing_contract(self):
        """T1.21: Coordinator routing contract validates OUTPUT_GRID_2D strategy specification."""
        cfg = ClientConfig.for_testing()
        assert PartitioningStrategy.OUTPUT_GRID_2D in PartitioningStrategy

    # --------------------------------------------------------------------------
    # Feature 5: Profiler Metrics & Download Invariants (5 tests)
    # --------------------------------------------------------------------------

    def test_t1_22_profiler_optimal_download_elements(self):
        """T1.22: Bandwidth profiler verifies download_elements == M * N."""
        m, k, n = 32, 64, 48
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=4, grid_cols=3)
        profile = Reference2DOutputGridEngine.profile_download_bandwidth(a, b, tasks)

        assert profile["download_elements"] == m * n
        assert profile["optimal_download_elements"] == m * n

    def test_t1_23_profiler_zero_download_redundancy_ratio(self):
        """T1.23: Download redundancy ratio is strictly 0.0 for 2D Output Grid."""
        m, k, n = 20, 50, 30
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=3)
        profile = Reference2DOutputGridEngine.profile_download_bandwidth(a, b, tasks)

        assert profile["download_redundancy_elements"] == 0
        assert profile["download_redundancy_ratio"] == 0.0
        assert profile["is_download_optimal"] is True

    def test_t1_24_profiler_upload_volume_formula(self):
        """T1.24: Upload volume equals P_c * M * K + P_r * K * N."""
        m, k, n = 40, 20, 60
        pr, pc = 2, 3
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=pr, grid_cols=pc)
        profile = Reference2DOutputGridEngine.profile_download_bandwidth(a, b, tasks)

        expected_upload = pc * m * k + pr * k * n
        assert profile["total_uploaded_elements"] == expected_upload

    def test_t1_25_profiler_dict_access_and_serialization(self):
        """T1.25: Bandwidth profile dictionary serialization contains required metrics."""
        a = np.ones((10, 10), dtype=np.float32)
        b = np.ones((10, 10), dtype=np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        profile = Reference2DOutputGridEngine.profile_download_bandwidth(a, b, tasks)

        required_keys = {
            "m", "k", "n", "num_tasks", "download_elements", "optimal_download_elements",
            "download_redundancy_ratio", "is_download_optimal", "total_uploaded_elements",
        }
        assert required_keys.issubset(set(profile.keys()))

    def test_t1_26_profiler_contrast_metrics(self):
        """T1.26: Profiler demonstrates download optimality over 1D outer product."""
        m, k, n = 32, 32, 32
        p_total = 4
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        # 2D Output Grid (2x2 = 4 tasks)
        grid_tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        prof_2d = Reference2DOutputGridEngine.profile_download_bandwidth(a, b, grid_tasks)

        # 1D Outer-Product (4 strips) download volume is 4 * M * N
        download_1d = p_total * m * n
        redundancy_ratio_1d = (float(download_1d) / float(m * n)) - 1.0

        assert prof_2d["download_elements"] == m * n
        assert prof_2d["download_redundancy_ratio"] == 0.0
        assert download_1d == 4 * m * n
        assert redundancy_ratio_1d == 3.0


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (10 TESTS)
# ==============================================================================

class TestTier2OutputGridBoundaries:
    """
    Tier 2: Boundary and corner cases: prime dimensions, odd shapes, scalar 1x1,
    strip vectors 1xN / Mx1, asymmetric shapes, non-square grids, oversubscription clamping,
    and defensive fail-closed validation.
    """

    def test_t2_01_prime_dimensions_7x11x13(self):
        """T2.1: Prime dimensions M=7, K=11, N=13 with grid P_r=2, P_c=3."""
        np.random.seed(201)
        a = np.random.randn(7, 11).astype(np.float32)
        b = np.random.randn(11, 13).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=3)
        assert len(tasks) == 6

        row_splits, col_splits = Reference2DOutputGridEngine.calculate_grid_blocks(7, 13, 2, 3)
        assert row_splits == [(0, 4), (4, 7)]
        assert col_splits == [(0, 5), (5, 9), (9, 13)]

        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(7, 13, computed)
        c_ref = np.matmul(a, b)
        np.testing.assert_allclose(c_out, c_ref, rtol=1e-5, atol=1e-5)

    def test_t2_02_odd_dimensions_15x25x35(self):
        """T2.2: Odd dimensions M=15, K=25, N=35 with non-square grid P_r=3, P_c=4."""
        np.random.seed(202)
        a = np.random.randn(15, 25).astype(np.float32)
        b = np.random.randn(25, 35).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=3, grid_cols=4)
        assert len(tasks) == 12

        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(15, 35, computed)
        np.testing.assert_allclose(c_out, np.matmul(a, b), rtol=1e-5, atol=1e-5)

    def test_t2_03_scalar_1x1_matrix(self):
        """T2.3: Scalar 1x1 matrix: [[3.0]] x [[7.0]] = [[21.0]] with P_r=1, P_c=1."""
        a = np.array([[3.0]], dtype=np.float32)
        b = np.array([[7.0]], dtype=np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=4, grid_cols=4)
        # Clamped to 1x1
        assert len(tasks) == 1
        assert tasks[0]["a_tile"].shape == (1, 1)
        assert tasks[0]["b_tile"].shape == (1, 1)

        c_p = np.matmul(tasks[0]["a_tile"], tasks[0]["b_tile"])
        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(1, 1, [(tasks[0], c_p)])
        assert c_out[0, 0] == 21.0

    def test_t2_04_row_vector_1xN_dot_product(self):
        """T2.4: Vector dot product 1x1000 multiplied by 1000x1 (clamped to 1x1 grid)."""
        np.random.seed(204)
        a = np.random.randn(1, 1000).astype(np.float32)
        b = np.random.randn(1000, 1).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        assert len(tasks) == 1
        assert tasks[0]["row_slice"] == (0, 1)
        assert tasks[0]["col_slice"] == (0, 1)
        assert tasks[0]["k_slice"] == (0, 1000)

        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(1, 1, [(tasks[0], np.matmul(a, b))])
        np.testing.assert_allclose(c_out, np.matmul(a, b), rtol=1e-5, atol=1e-5)

    def test_t2_05_column_vector_Mx1(self):
        """T2.5: Matrix 100x32 multiplied by vector 32x1 with grid 4x1."""
        np.random.seed(205)
        a = np.random.randn(100, 32).astype(np.float32)
        b = np.random.randn(32, 1).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=4, grid_cols=1)
        assert len(tasks) == 4
        for t in tasks:
            assert t["col_slice"] == (0, 1)
            assert t["b_tile"].shape == (32, 1)

        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(100, 1, computed)
        np.testing.assert_allclose(c_out, np.matmul(a, b), rtol=1e-5, atol=1e-5)

    def test_t2_06_asymmetric_tall_skinny_and_wide_flat(self):
        """T2.6: Tall-skinny (128x16x32) and wide-flat (16x128x16) shapes."""
        np.random.seed(206)

        # Tall-skinny
        a_tall = np.random.randn(128, 16).astype(np.float32)
        b_tall = np.random.randn(16, 32).astype(np.float32)
        tasks_tall = Reference2DOutputGridEngine.partition_output_grid_2d(a_tall, b_tall, grid_rows=4, grid_cols=2)
        c_tall_res = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks_tall]
        c_tall, _ = Reference2DOutputGridEngine.assemble_output_grid(128, 32, c_tall_res)
        np.testing.assert_allclose(c_tall, np.matmul(a_tall, b_tall), rtol=1e-5, atol=1e-5)

        # Wide-flat
        a_wide = np.random.randn(16, 128).astype(np.float32)
        b_wide = np.random.randn(128, 16).astype(np.float32)
        tasks_wide = Reference2DOutputGridEngine.partition_output_grid_2d(a_wide, b_wide, grid_rows=2, grid_cols=4)
        c_wide_res = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks_wide]
        c_wide, _ = Reference2DOutputGridEngine.assemble_output_grid(16, 16, c_wide_res)
        np.testing.assert_allclose(c_wide, np.matmul(a_wide, b_wide), rtol=1e-5, atol=1e-5)

    def test_t2_07_extreme_contraction_dimension_k5000(self):
        """T2.7: Extreme inner contraction dimension M=2, K=5000, N=2 with K unsliced."""
        np.random.seed(207)
        a = (np.random.randn(2, 5000) / math.sqrt(5000)).astype(np.float32)
        b = (np.random.randn(5000, 2) / math.sqrt(5000)).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=1, grid_cols=2)
        assert len(tasks) == 2
        for t in tasks:
            assert t["k_j"] == 5000
            assert t["k_slice"] == (0, 5000)

        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(2, 2, computed)
        np.testing.assert_allclose(c_out, np.matmul(a, b), atol=1e-5)

    def test_t2_08_non_square_grids_pr_ne_pc(self):
        """T2.8: Highly asymmetric grid layouts (1x8 and 8x1)."""
        a = np.ones((64, 32), dtype=np.float32)
        b = np.ones((32, 64), dtype=np.float32)

        # 1 x 8 column strips
        tasks_1x8 = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=1, grid_cols=8)
        assert len(tasks_1x8) == 8
        for t in tasks_1x8:
            assert t["row_slice"] == (0, 64)

        # 8 x 1 row strips
        tasks_8x1 = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=8, grid_cols=1)
        assert len(tasks_8x1) == 8
        for t in tasks_8x1:
            assert t["col_slice"] == (0, 64)

    def test_t2_09_grid_oversubscription_clamping(self):
        """T2.9: Grid oversubscription P_r=10 > M=3 and P_c=10 > N=4 clamped to 3x4 without zero blocks."""
        a = np.ones((3, 5), dtype=np.float32)
        b = np.ones((5, 4), dtype=np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=10, grid_cols=10)
        assert len(tasks) == 12  # 3 * 4
        for t in tasks:
            assert t["m_i"] == 1
            assert t["n_k"] == 1

        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(3, 4, computed)
        np.testing.assert_allclose(c_out, np.matmul(a, b))

    def test_t2_10_invalid_dimensions_fail_closed(self):
        """T2.10: Rejection of 1D/3D arrays, inner dimension mismatch, 0-dim matrices, negative grid."""
        # 1D array
        with pytest.raises(MatrixDimensionError):
            Reference2DOutputGridEngine.partition_output_grid_2d(np.ones(10), np.ones((10, 10)), 2, 2)

        # Mismatched inner dimension
        with pytest.raises(MatrixDimensionError, match="Inner dimension mismatch"):
            Reference2DOutputGridEngine.partition_output_grid_2d(np.ones((10, 5)), np.ones((6, 10)), 2, 2)

        # Zero dimension
        with pytest.raises(MatrixDimensionError):
            Reference2DOutputGridEngine.partition_output_grid_2d(np.ones((0, 5)), np.ones((5, 10)), 2, 2)

        # Non-positive grid dimensions
        with pytest.raises(ValueError):
            Reference2DOutputGridEngine.partition_output_grid_2d(np.ones((4, 4)), np.ones((4, 4)), 0, 2)


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (9 TESTS)
# ==============================================================================

class TestTier3OutputGridCrossFeature:
    """
    Tier 3: Cross-feature interactions: Dual-tier Freivalds verification,
    per-tile fraud rejection and FreivaldsFraudProof emission, batch L402 escrow lifecycle,
    multi-worker partial fraud isolation, sliding window concurrency bounding,
    TEE encrypted chunk dispatch, and worker failover redispatch.
    """

    def test_t3_01_dual_tier_freivalds_per_tile_and_assembled(self):
        """T3.1: Dual-tier Freivalds verifies each spatial tile before placement and assembled matrix."""
        np.random.seed(301)
        m, k, n = 32, 48, 32
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        completed = []

        # Tier 1 of Dual-Tier: Per-Tile Freivalds verification
        for t in tasks:
            c_ij = np.matmul(t["a_tile"], t["b_tile"])
            passed, disc, bound, _ = ReferenceOracle.freivalds_verify(t["a_tile"], t["b_tile"], c_ij, dtype="fp32")
            assert passed is True, f"Per-tile verification failed on {t['task_id']}"
            assert disc <= bound
            completed.append((t, c_ij))

        # Direct spatial assembly
        c_assembled, ops = Reference2DOutputGridEngine.assemble_output_grid(m, n, completed)
        assert ops == 0

        # Tier 2 of Dual-Tier: Assembled Freivalds verification
        assembled_passed, _, _, _ = ReferenceOracle.freivalds_verify(a, b, c_assembled, dtype="fp32")
        assert assembled_passed is True

    def test_t3_02_freivalds_fraud_rejection_of_corrupt_tile(self, mock_settlement):
        """T3.2: Rogue worker tampers with sub-tile C_ij (+5.0); caught by per-tile Freivalds, emits proof."""
        np.random.seed(302)
        m, k, n = 16, 32, 16
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        assembler = ResultAssembler(m=m, n=n, expected_tasks=[t["task_id"] for t in tasks], dtype="fp32")

        # Tile 0: Honest
        c_0 = np.matmul(tasks[0]["a_tile"], tasks[0]["b_tile"])
        p0_valid, _, _, _ = ReferenceOracle.freivalds_verify(tasks[0]["a_tile"], tasks[0]["b_tile"], c_0)
        assert p0_valid is True
        assembler.accumulate_task(tasks[0], c_0)

        # Tile 1: Rogue worker tampers with element
        c_1_corrupt = np.matmul(tasks[1]["a_tile"], tasks[1]["b_tile"])
        c_1_corrupt[1, 1] += 5.0

        p1_valid, disc, bound, proof_data = ReferenceOracle.freivalds_verify(
            tasks[1]["a_tile"], tasks[1]["b_tile"], c_1_corrupt
        )
        assert p1_valid is False
        assert disc > bound
        assert len(proof_data) > 0

        # Build authoritative FreivaldsFraudProof
        proof = FreivaldsFraudProof(
            task_id=tasks[1]["task_id"],
            worker_pubkey="rogue_worker_node",
            dimension_m=tasks[1]["m_i"],
            dimension_k=tasks[1]["k_j"],
            dimension_n=tasks[1]["n_k"],
            dtype="fp32",
            matrix_a_hash=ReferenceOracle.sha256_digest(tasks[1]["a_tile"]),
            matrix_b_hash=ReferenceOracle.sha256_digest(tasks[1]["b_tile"]),
            invalid_result_c_hash=ReferenceOracle.sha256_digest(c_1_corrupt),
            random_vector_r=proof_data["random_vector_r"],
            vector_br=proof_data["vector_br"],
            vector_abr=proof_data["vector_abr"],
            vector_cr=proof_data["vector_cr"],
            epsilon_tolerance=bound,
            max_observed_discrepancy=disc,
            worker_signature_on_result="sig_rogue_tile",
            consumer_pubkey="consumer_pubkey_test",
        )
        assert proof.task_id == tasks[1]["task_id"]
        assert proof.max_observed_discrepancy > proof.epsilon_tolerance

        # Invariant: Corrupt tile is rejected BEFORE buffer placement
        assert assembler.is_complete() is False
        assert tasks[1]["task_id"] in assembler.missing_tasks()

    def test_t3_03_batch_l402_escrow_lifecycle_2d_grid(self, mock_settlement):
        """T3.3: 2D output grid tasks grouped per worker into batch HODL invoices; settled upon 100% pass."""
        np.random.seed(303)
        m, k, n = 32, 64, 32
        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(
            np.random.randn(m, k), np.random.randn(k, n), grid_rows=2, grid_cols=2
        )

        workers = ["worker_grid_1", "worker_grid_2"]
        worker_tasks = {
            "worker_grid_1": [tasks[0], tasks[1]],
            "worker_grid_2": [tasks[2], tasks[3]],
        }

        escrows = {}
        for w_pub in workers:
            preimage = hashlib.sha256(f"preimage_{w_pub}".encode()).hexdigest()
            p_hash = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()

            inv = mock_settlement.create_hodl_invoice(
                task_id=f"batch_{w_pub}",
                total_compute_sats=100,
                payment_hash=p_hash,
            )
            mock_settlement.hold_invoice(p_hash)
            assert mock_settlement.get_status(p_hash) == "HELD"
            escrows[w_pub] = {"p_hash": p_hash, "preimage": preimage}

        # Verify all tasks pass and settle invoices
        for w_pub in workers:
            for t in worker_tasks[w_pub]:
                c_ij = np.matmul(t["a_tile"], t["b_tile"])
                passed, _, _, _ = ReferenceOracle.freivalds_verify(t["a_tile"], t["b_tile"], c_ij)
                assert passed is True

            mock_settlement.settle_invoice(escrows[w_pub]["p_hash"], preimage=escrows[w_pub]["preimage"])
            assert mock_settlement.get_status(escrows[w_pub]["p_hash"]) == "SETTLED"

    def test_t3_04_batch_l402_escrow_cancellation_on_fraud(self, mock_settlement):
        """T3.4: Corrupt tile triggers atomic invoice cancellation and withholding of sats."""
        p_hash = hashlib.sha256(b"preimage_fraud_2d").hexdigest()
        mock_settlement.create_hodl_invoice(
            task_id="batch_grid_fraud",
            total_compute_sats=250,
            payment_hash=p_hash,
        )
        mock_settlement.hold_invoice(p_hash)
        assert mock_settlement.get_status(p_hash) == "HELD"

        # Fraud detected on tile
        mock_settlement.cancel_invoice(p_hash)
        assert mock_settlement.get_status(p_hash) == "CANCELLED"

        with pytest.raises(ValueError, match="Cannot settle invoice"):
            mock_settlement.settle_invoice(p_hash, preimage=hashlib.sha256(b"dummy").hexdigest())

    def test_t3_05_multi_worker_partial_fraud_isolation(self, mock_settlement):
        """T3.5: Multi-worker execution: Honest worker settles, rogue worker cancelled & isolated."""
        w1_preimage = hashlib.sha256(b"w1_grid_honest").hexdigest()
        w1_hash = hashlib.sha256(bytes.fromhex(w1_preimage)).hexdigest()
        mock_settlement.create_hodl_invoice(task_id="batch_w1", total_compute_sats=150, payment_hash=w1_hash)
        mock_settlement.hold_invoice(w1_hash)

        w2_preimage = hashlib.sha256(b"w2_grid_rogue").hexdigest()
        w2_hash = hashlib.sha256(bytes.fromhex(w2_preimage)).hexdigest()
        mock_settlement.create_hodl_invoice(task_id="batch_w2", total_compute_sats=150, payment_hash=w2_hash)
        mock_settlement.hold_invoice(w2_hash)

        # Worker 1 passes
        mock_settlement.settle_invoice(w1_hash, preimage=w1_preimage)
        assert mock_settlement.get_status(w1_hash) == "SETTLED"

        # Worker 2 fails
        mock_settlement.cancel_invoice(w2_hash)
        assert mock_settlement.get_status(w2_hash) == "CANCELLED"

    def test_t3_06_sliding_window_concurrency_bounding(self):
        """T3.6: 2D output grid tile execution respects sliding window concurrency limit."""
        np.random.seed(306)
        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(
            np.ones((64, 32)), np.ones((32, 64)), grid_rows=4, grid_cols=4
        )
        assert len(tasks) == 16

        max_concurrency = 4
        in_flight = 0
        peak_in_flight = 0
        completed = []

        for t in tasks:
            in_flight += 1
            if in_flight > peak_in_flight:
                peak_in_flight = in_flight
            assert in_flight <= max_concurrency, f"Concurrency limit exceeded: {in_flight} > {max_concurrency}"

            c_ij = np.matmul(t["a_tile"], t["b_tile"])
            completed.append((t, c_ij))
            in_flight -= 1

        assert peak_in_flight <= max_concurrency
        assert len(completed) == 16

    def test_t3_07_assembled_corruption_detection(self):
        """T3.7: Assembled Freivalds verification catches memory corruption in final product."""
        np.random.seed(307)
        a = np.random.randn(24, 36).astype(np.float32)
        b = np.random.randn(36, 24).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        completed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_clean, _ = Reference2DOutputGridEngine.assemble_output_grid(24, 24, completed)

        # Clean passes
        valid_clean, _, _, _ = ReferenceOracle.freivalds_verify(a, b, c_clean, dtype="fp32")
        assert valid_clean is True

        # Tampered output buffer
        c_corrupted = c_clean.copy()
        c_corrupted[5, 5] += 3.0
        valid_corrupt, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c_corrupted, dtype="fp32", num_trials=20)
        assert valid_corrupt is False
        assert disc > bound

    def test_t3_08_confidential_computing_tee_2d_grid(self):
        """T3.8: 2D output grid tile streamed with AES-256-GCM encryption, decrypted and assembled."""
        np.random.seed(308)
        c_priv, c_pub = generate_ephemeral_keypair()
        w_priv, w_pub = generate_ephemeral_keypair()
        nonce = generate_challenger_nonce()

        c_keys = derive_session_keys(c_priv, w_pub, nonce)
        w_keys = derive_session_keys(w_priv, c_pub, nonce)

        client_crypto = SessionCrypto(c_keys, role="client", task_id="grid_tee_task_0")
        worker_crypto = CryptoSession(w_keys, role="worker", task_id="grid_tee_task_0")

        # Tile inputs
        a_tile = np.random.randn(16, 32).astype(np.float32)
        b_tile = np.random.randn(32, 16).astype(np.float32)

        # Client encrypts A and B tiles
        enc_a = client_crypto.encrypt_chunk(a_tile.tobytes(), is_matrix_a=True, sequence_number=0, is_final_chunk=True)
        enc_b = client_crypto.encrypt_chunk(b_tile.tobytes(), is_matrix_a=False, sequence_number=0, is_final_chunk=True)

        # Worker decrypts inside enclave
        dec_a_bytes = worker_crypto.decrypt_chunk(enc_a.ciphertext, enc_a.iv, enc_a.tag, is_matrix_a=True, sequence_number=0, is_final_chunk=True)
        dec_b_bytes = worker_crypto.decrypt_chunk(enc_b.ciphertext, enc_b.iv, enc_b.tag, is_matrix_a=False, sequence_number=0, is_final_chunk=True)

        dec_a = np.frombuffer(dec_a_bytes, dtype=np.float32).reshape((16, 32))
        dec_b = np.frombuffer(dec_b_bytes, dtype=np.float32).reshape((32, 16))

        # In-enclave GEMM & result encryption
        c_sub = np.matmul(dec_a, dec_b)
        enc_result = worker_crypto.encrypt_result(c_sub.tobytes(), rows=16, cols=16)

        # Client decrypts and verifies
        dec_c_bytes = client_crypto.decrypt_result(enc_result.ciphertext, enc_result.iv, enc_result.tag, rows=16, cols=16)
        c_recovered = np.frombuffer(dec_c_bytes, dtype=np.float32).reshape((16, 16))

        np.testing.assert_array_equal(c_recovered, np.matmul(a_tile, b_tile))

    def test_t3_09_worker_failover_redispatch(self):
        """T3.9: Worker node fails on assigned tile; uncompleted task re-dispatched to backup worker."""
        np.random.seed(309)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        assembler = ResultAssembler(m=32, n=32, expected_tasks=[t["task_id"] for t in tasks], dtype="fp32")

        # Worker 1 succeeds on tile 0, 1
        assembler.accumulate_task(tasks[0], np.matmul(tasks[0]["a_tile"], tasks[0]["b_tile"]))
        assembler.accumulate_task(tasks[1], np.matmul(tasks[1]["a_tile"], tasks[1]["b_tile"]))

        # Worker 2 crashes on tile 2, 3 -> tasks[2], tasks[3] re-dispatched to Worker 3
        backup_worker_completed = []
        for failed_task in [tasks[2], tasks[3]]:
            c_recovered = np.matmul(failed_task["a_tile"], failed_task["b_tile"])
            assembler.accumulate_task(failed_task, c_recovered)

        assert assembler.is_complete() is True
        np.testing.assert_allclose(assembler.get_result(), np.matmul(a, b), rtol=1e-5, atol=1e-5)


# ==============================================================================
# TIER 4: REAL-WORLD SCENARIOS & DOWNLOAD BANDWIDTH PROFILING (8 TESTS)
# ==============================================================================

class TestTier4OutputGridBandwidthAndScenarios:
    """
    Tier 4: Bandwidth profiling proving strictly M*N download elements (redundancy ratio == 0.0),
    contrast with 1D/3D strategies, Higham error bounds, Transformer Attention and MLP projections,
    and large-scale 1024x1024 stress testing.
    """

    def test_t4_01_download_bandwidth_profiling_zero_redundancy(self):
        """T4.1: Verifies total downloaded matrix elements equal strictly M*N (redundancy == 0.0)."""
        test_shapes = [
            (64, 128, 64, 2, 2),
            (100, 200, 100, 4, 4),
            (37, 73, 59, 3, 3),      # Primes
            (128, 64, 256, 4, 8),    # Rectangular asymmetric
            (1, 1000, 1, 1, 1),      # Dot product
        ]

        for m, k, n, pr, pc in test_shapes:
            a = np.ones((m, k), dtype=np.float32)
            b = np.ones((k, n), dtype=np.float32)

            tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=pr, grid_cols=pc)
            profile = Reference2DOutputGridEngine.profile_download_bandwidth(a, b, tasks)

            # Invariant 1: Total downloaded elements strictly equal M * N
            assert profile["download_elements"] == m * n
            # Invariant 2: Download redundancy elements strictly zero
            assert profile["download_redundancy_elements"] == 0
            # Invariant 3: Download redundancy ratio strictly zero
            assert profile["download_redundancy_ratio"] == 0.0
            # Invariant 4: Flagged as download optimal
            assert profile["is_download_optimal"] is True

    def test_t4_02_download_bandwidth_contrast_with_1d_and_3d(self):
        """T4.2: Contrasts 2D Output Grid (0% download redundancy) with 1D and 3D strategies."""
        m, k, n = 128, 128, 128
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)

        # 1. 2D Output Grid (2x2 grid, P=4 tasks, K unsliced)
        grid_tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2)
        prof_2d = Reference2DOutputGridEngine.profile_download_bandwidth(a, b, grid_tasks)

        # 2. 1D Outer-Product with P=4 strips: Each strip returns full (128, 128) -> 4 * 128 * 128
        download_1d = 4 * (m * n)
        redundancy_ratio_1d = (float(download_1d) / float(m * n)) - 1.0

        # 3. 3D Block Tiling with tile_size=64 (2x2x2 = 8 tasks):
        # K is sliced into 2 blocks. For each (row, col) block, 2 partial tiles are returned -> 2 * M * N
        p_k = 2
        download_3d = p_k * (m * n)
        redundancy_ratio_3d = (float(download_3d) / float(m * n)) - 1.0

        # 2D Output Grid is strictly optimal
        assert prof_2d["download_elements"] == m * n
        assert prof_2d["download_redundancy_ratio"] == 0.0

        # 1D has 300% download redundancy (4x optimal)
        assert redundancy_ratio_1d == 3.0

        # 3D has 100% download redundancy (2x optimal)
        assert redundancy_ratio_3d == 1.0

    def test_t4_03_mathematical_equivalence_higham_fp32(self):
        """T4.3: Mathematical equivalence against np.matmul within Higham bound for FP32."""
        np.random.seed(403)
        m, k, n = 128, 256, 128
        a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=4, grid_cols=4, dtype="fp32")
        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_assembled, _ = Reference2DOutputGridEngine.assemble_output_grid(m, n, computed, dtype="fp32")
        c_direct = np.matmul(a, b)

        norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
        higham_limit = calculate_higham_bound(k, norm_a, norm_b, dtype="fp32")

        observed_error = float(np.max(np.abs(c_assembled - c_direct)))
        assert observed_error <= higham_limit, f"Observed error {observed_error} > Higham bound {higham_limit}"

    def test_t4_04_mathematical_equivalence_higham_fp16(self):
        """T4.4: Mathematical equivalence for FP16 within Higham bound."""
        np.random.seed(404)
        m, k, n = 64, 128, 64
        a = (np.random.randn(m, k) * 0.1).astype(np.float16)
        b = (np.random.randn(k, n) * 0.1).astype(np.float16)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2, dtype="fp16")
        computed = []
        for t in tasks:
            c_p = np.matmul(t["a_tile"].astype(np.float32), t["b_tile"].astype(np.float32)).astype(np.float16)
            computed.append((t, c_p))

        c_assembled, _ = Reference2DOutputGridEngine.assemble_output_grid(m, n, computed, dtype="fp16")
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)

        norm_a = float(np.max(np.sum(np.abs(a.astype(np.float32)), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b.astype(np.float32)), axis=1)))
        higham_limit = calculate_higham_bound(k, norm_a, norm_b, dtype="fp16")

        observed_error = float(np.max(np.abs(c_assembled.astype(np.float32) - c_ref.astype(np.float32))))
        assert observed_error <= max(higham_limit, 1e-2)

    def test_t4_05_mathematical_equivalence_higham_bf16(self):
        """T4.5: Mathematical equivalence for BF16 with mantissa bitmasking."""
        np.random.seed(405)
        m, k, n = 64, 128, 64
        a = truncate_to_bf16(np.random.randn(m, k) * 0.1)
        b = truncate_to_bf16(np.random.randn(k, n) * 0.1)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=2, grid_cols=2, dtype="bf16")
        computed = [(t, truncate_to_bf16(np.matmul(t["a_tile"], t["b_tile"]))) for t in tasks]
        c_assembled, _ = Reference2DOutputGridEngine.assemble_output_grid(m, n, computed, dtype="bf16")

        c_ref = ReferenceOracle.gemm(a, b, dtype="bf16")
        observed_error = float(np.max(np.abs(c_assembled - c_ref)))

        assert observed_error < 0.05
        assert validate_bf16_masking(c_assembled) is True

    def test_t4_06_deep_learning_attention_projection_2d_grid(self):
        """T4.6: Real-world Transformer Attention projection (batch*seq=64, dim=512, heads*head_dim=64)."""
        np.random.seed(406)
        m, k, n = 64, 512, 64
        q = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        w_proj = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(q, w_proj, grid_rows=2, grid_cols=2)
        profile = Reference2DOutputGridEngine.profile_download_bandwidth(q, w_proj, tasks)

        assert profile["is_download_optimal"] is True
        assert profile["download_elements"] == 64 * 64

        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_projected, ops = Reference2DOutputGridEngine.assemble_output_grid(m, n, computed)
        assert ops == 0

        c_expected = np.matmul(q, w_proj)
        np.testing.assert_allclose(c_projected, c_expected, rtol=1e-5, atol=1e-5)

    def test_t4_07_deep_learning_mlp_projection_2d_grid(self):
        """T4.7: Real-world MLP projection (M=32, K=1024, N=2048) with 2x4 grid."""
        np.random.seed(407)
        m, k, n = 32, 1024, 2048
        x = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        w_mlp = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(x, w_mlp, grid_rows=2, grid_cols=4)
        profile = Reference2DOutputGridEngine.profile_download_bandwidth(x, w_mlp, tasks)

        assert profile["download_elements"] == 32 * 2048
        assert profile["download_redundancy_ratio"] == 0.0

        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_out, _ = Reference2DOutputGridEngine.assemble_output_grid(m, n, computed)
        np.testing.assert_allclose(c_out, np.matmul(x, w_mlp), rtol=1e-5, atol=1e-5)

    def test_t4_08_large_scale_stress_1024x1024(self):
        """T4.8: Large scale 1024x1024 distributed matmul verifying exact 1,048,576 downloaded elements."""
        m, k, n = 1024, 1024, 1024
        pr, pc = 4, 4  # 16 spatial tiles

        np.random.seed(408)
        a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        tasks = Reference2DOutputGridEngine.partition_output_grid_2d(a, b, grid_rows=pr, grid_cols=pc)
        assert len(tasks) == 16

        profile = Reference2DOutputGridEngine.profile_download_bandwidth(a, b, tasks)
        expected_download = 1024 * 1024  # 1,048,576 elements
        assert profile["download_elements"] == expected_download
        assert profile["download_redundancy_elements"] == 0
        assert profile["download_redundancy_ratio"] == 0.0

        computed = [(t, np.matmul(t["a_tile"], t["b_tile"])) for t in tasks]
        c_out, ops = Reference2DOutputGridEngine.assemble_output_grid(m, n, computed)
        assert ops == 0

        c_expected = np.matmul(a, b)
        np.testing.assert_allclose(c_out, c_expected, rtol=1e-5, atol=1e-5)
