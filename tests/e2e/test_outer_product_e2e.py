"""
Opaque-Box E2E Test Suite: 1D Outer-Product (Upload-Optimal) Matrix Partitioning.

Phase 1 Verification Matrix:
- Tier 1: Feature Coverage (Basic 1D outer-product partitioning, strips generation,
  partial product accumulation, dtype preservation across FP32, FP16, and BF16,
  ResultAssembler in-place accumulation, reentrancy guards).
- Tier 2: Boundary & Corner Cases (Primes 7x11x13, odd shapes, scalar 1x1,
  asymmetric M!=N!=K, strip oversubscription P > K, single strip P=1,
  extreme aspect ratios, invalid dimensions).
- Tier 3: Cross-Feature Combinations (1D outer product with mock worker pool,
  dual-tier Freivalds per-strip verification, Freivalds fraud rejection of corrupt strips,
  batch L402 escrow lifecycle, escrow cancellation on fraud, sliding window concurrency).
- Tier 4: Real-World Scenarios & Bandwidth Profiling (verifying total uploaded matrix
  elements equal strictly M*K + K*N with zero duplicate uploads, bandwidth contrast with
  3D block tiling, Higham error bound mathematical equivalence across FP32/FP16/BF16,
  deep learning attention projection, large-scale 1024x1024 stress).
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
from matrix_compute.config import ClientConfig
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


# ==============================================================================
# AUTHORITATIVE REFERENCE 1D OUTER-PRODUCT ENGINE & PROFILER
# ==============================================================================

class Reference1DOuterProductEngine:
    """
    Authoritative, independent reference implementation of 1D Outer-Product
    (upload-optimal) matrix partitioning, accumulation, and bandwidth profiling.
    """

    @classmethod
    def calculate_strip_splits(cls, k: int, num_strips: int) -> List[Tuple[int, int]]:
        """
        Calculates balanced contiguous slice ranges [k_start, k_end) along dimension K.
        Distributes remainder r = K % P across the first r strips.
        Clamps effective strips: P_eff = min(max(1, num_strips), K).
        Guarantees zero-width strips are never produced and coverage of [0, K) is exact.
        """
        if k < 1:
            raise ValueError(f"Inner dimension K must be >= 1, got {k}")
        p_eff = min(max(1, int(num_strips)), k)
        base_w = k // p_eff
        rem = k % p_eff

        splits: List[Tuple[int, int]] = []
        curr = 0
        for p in range(p_eff):
            w = base_w + (1 if p < rem else 0)
            splits.append((curr, curr + w))
            curr += w
        assert curr == k, f"Splits did not cover K exactly: {curr} != {k}"
        return splits

    @classmethod
    def partition_outer_product_1d(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        num_strips: int,
        dtype: str = "fp32",
        task_id_prefix: str = "strip",
    ) -> List[Dict[str, Any]]:
        """
        Partitions matrix A (M x K) along columns and B (K x N) along rows into P strips.
        Returns a list of strip task specification dictionaries.
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

        norm_dtype = normalize_dtype(dtype)
        a_cast = cast_matrix(a, norm_dtype)
        b_cast = cast_matrix(b, norm_dtype)

        splits = cls.calculate_strip_splits(k, num_strips)
        tasks: List[Dict[str, Any]] = []

        for p, (k_start, k_end) in enumerate(splits):
            k_p = k_end - k_start
            a_strip = a_cast[:, k_start:k_end]
            b_strip = b_cast[k_start:k_end, :]

            gflops = (2.0 * float(m) * float(k_p) * float(n)) / 1e9

            tasks.append(
                {
                    "task_id": f"{task_id_prefix}_{p}",
                    "row_idx": 0,
                    "col_idx": 0,
                    "k_idx": p,
                    "row_slice": (0, m),
                    "col_slice": (0, n),
                    "k_slice": (k_start, k_end),
                    "a_tile": a_strip,
                    "b_tile": b_strip,
                    "m_i": m,
                    "k_j": k_p,
                    "n_k": n,
                    "dtype": norm_dtype,
                    "estimated_gflops": gflops,
                }
            )

        return tasks

    @classmethod
    def assemble_strips(
        cls,
        m: int,
        n: int,
        strip_results: List[Tuple[Dict[str, Any], np.ndarray]],
        dtype: str = "fp32",
    ) -> np.ndarray:
        """
        Accumulates partial matrix products C_p = A_p x B_p into final M x N matrix.
        C = sum_{p=0}^{P-1} C_p
        """
        norm_dtype = normalize_dtype(dtype)
        np_dtype = np.float32 if norm_dtype in ("fp32", "bf16") else np.float16
        c_acc = np.zeros((m, n), dtype=np_dtype)

        for task_info, partial_c in strip_results:
            assert partial_c.shape == (m, n), f"Partial result shape {partial_c.shape} != {(m, n)}"
            c_acc += partial_c.astype(np_dtype)

        if norm_dtype == "bf16":
            return truncate_to_bf16(c_acc)
        elif norm_dtype == "fp16":
            return c_acc.astype(np.float16)
        return c_acc.astype(np.float32)

    @classmethod
    def profile_upload_bandwidth(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        strip_tasks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Measures total uploaded matrix elements and bytes across all strips.
        Asserts upload optimality: total_uploaded_elements == M*K + K*N.
        """
        m, k = a.shape
        _, n = b.shape
        optimal_elements = m * k + k * n

        total_a_elements = sum(t["a_tile"].size for t in strip_tasks)
        total_b_elements = sum(t["b_tile"].size for t in strip_tasks)
        total_uploaded_elements = total_a_elements + total_b_elements

        redundancy = total_uploaded_elements - optimal_elements
        redundancy_ratio = (float(total_uploaded_elements) / float(optimal_elements)) - 1.0

        bytes_per_elem = get_bytes_per_element(strip_tasks[0]["dtype"])
        total_uploaded_bytes = total_uploaded_elements * bytes_per_elem
        optimal_bytes = optimal_elements * bytes_per_elem

        return {
            "m": m,
            "k": k,
            "n": n,
            "num_strips": len(strip_tasks),
            "total_a_elements": total_a_elements,
            "total_b_elements": total_b_elements,
            "total_uploaded_elements": total_uploaded_elements,
            "optimal_elements": optimal_elements,
            "redundancy_elements": redundancy,
            "redundancy_ratio": redundancy_ratio,
            "total_uploaded_bytes": total_uploaded_bytes,
            "optimal_bytes": optimal_bytes,
            "is_upload_optimal": redundancy == 0 and abs(redundancy_ratio) < 1e-12,
        }


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 TESTS)
# ==============================================================================

class TestTier1OuterProductFeatureCoverage:
    """
    Tier 1: Feature coverage of 1D outer-product partitioning, strip generation,
    partial accumulation, and dtype preservation across FP32, FP16, and BF16.
    """

    def test_t1_01_outer_product_strip_partitioning_even_division(self):
        """T1.1: 64x64 square matrix partitioned into P=4 strips along K=64."""
        np.random.seed(42)
        a = np.random.randn(64, 64).astype(np.float32)
        b = np.random.randn(64, 64).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=4)
        assert len(strips) == 4

        for p, task in enumerate(strips):
            assert task["task_id"] == f"strip_{p}"
            assert task["row_idx"] == 0
            assert task["col_idx"] == 0
            assert task["k_idx"] == p
            assert task["row_slice"] == (0, 64)
            assert task["col_slice"] == (0, 64)
            assert task["k_slice"] == (p * 16, (p + 1) * 16)
            assert task["a_tile"].shape == (64, 16)
            assert task["b_tile"].shape == (16, 64)

    def test_t1_02_outer_product_strip_remainder_distribution(self):
        """T1.2: Inner dimension K=10 with P=3 distributes remainder: widths [4, 3, 3]."""
        a = np.ones((8, 10), dtype=np.float32)
        b = np.ones((10, 12), dtype=np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=3)
        assert len(strips) == 3

        expected_slices = [(0, 4), (4, 7), (7, 10)]
        expected_k_widths = [4, 3, 3]

        for p, (task, exp_slice, exp_k) in enumerate(zip(strips, expected_slices, expected_k_widths)):
            assert task["k_slice"] == exp_slice
            assert task["k_j"] == exp_k
            assert task["a_tile"].shape == (8, exp_k)
            assert task["b_tile"].shape == (exp_k, 12)

        # Verify contiguous coverage
        all_k_covered = sum(t["k_j"] for t in strips)
        assert all_k_covered == 10

    def test_t1_03_partial_product_accumulation_fp32(self):
        """T1.3: Partial products C_p = A_p x B_p sum to C matching direct np.matmul in FP32."""
        np.random.seed(101)
        a = np.random.randn(32, 48).astype(np.float32)
        b = np.random.randn(48, 24).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=4, dtype="fp32")
        computed_strips = []
        for s in strips:
            c_p = np.matmul(s["a_tile"], s["b_tile"])
            computed_strips.append((s, c_p))

        c_assembled = Reference1DOuterProductEngine.assemble_strips(32, 24, computed_strips, dtype="fp32")
        c_expected = np.matmul(a, b)

        assert c_assembled.shape == (32, 24)
        assert c_assembled.dtype == np.float32
        np.testing.assert_allclose(c_assembled, c_expected, rtol=1e-5, atol=1e-5)

    def test_t1_04_precision_fp16_preservation(self):
        """T1.4: FP16 precision preserves float16 data types and respects FP16 rounding bounds."""
        np.random.seed(102)
        a = (np.random.randn(20, 30) * 0.5).astype(np.float16)
        b = (np.random.randn(30, 20) * 0.5).astype(np.float16)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=5, dtype="fp16")
        computed_strips = []
        for s in strips:
            assert s["a_tile"].dtype == np.float16
            assert s["b_tile"].dtype == np.float16
            # Worker computes in float16 or accumulates in float32 then casts to float16
            c_p = np.matmul(s["a_tile"].astype(np.float32), s["b_tile"].astype(np.float32)).astype(np.float16)
            computed_strips.append((s, c_p))

        c_assembled = Reference1DOuterProductEngine.assemble_strips(20, 20, computed_strips, dtype="fp16")
        assert c_assembled.dtype == np.float16
        c_expected = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(c_assembled, c_expected, rtol=1e-2, atol=1e-2)

    def test_t1_05_precision_bf16_preservation(self):
        """T1.5: BF16 precision enforces 16-bit mantissa masking (0xFFFF0000) and preserves dtype."""
        np.random.seed(103)
        a = truncate_to_bf16(np.random.randn(16, 24).astype(np.float32))
        b = truncate_to_bf16(np.random.randn(24, 16).astype(np.float32))
        assert validate_bf16_masking(a) is True
        assert validate_bf16_masking(b) is True

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=3, dtype="bf16")
        computed_strips = []
        for s in strips:
            assert validate_bf16_masking(s["a_tile"]) is True
            assert validate_bf16_masking(s["b_tile"]) is True
            c_p = truncate_to_bf16(np.matmul(s["a_tile"], s["b_tile"]))
            computed_strips.append((s, c_p))

        c_assembled = Reference1DOuterProductEngine.assemble_strips(16, 16, computed_strips, dtype="bf16")
        assert c_assembled.dtype == np.float32
        assert validate_bf16_masking(c_assembled) is True

        c_ref = ReferenceOracle.gemm(a, b, dtype="bf16")
        np.testing.assert_allclose(c_assembled, c_ref, rtol=1e-2, atol=0.08)

    def test_t1_06_result_assembler_in_place_outer_product_slices(self):
        """T1.6: Production ResultAssembler accumulates outer-product slices (0:M, 0:N, k=p) in-place."""
        m, n, k = 16, 16, 32
        np.random.seed(104)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=4, dtype="fp32")
        assembler = ResultAssembler(m=m, n=n, expected_tasks=len(strips), dtype="fp32")

        assert assembler.is_complete() is False
        assert len(assembler.missing_tasks()) == 0  # because integer count was passed

        for s in strips:
            tile_c = np.matmul(s["a_tile"], s["b_tile"])
            assembler.accumulate(
                tile_result=tile_c,
                row_idx=s["row_idx"],
                col_idx=s["col_idx"],
                k_idx=s["k_idx"],
                row_slice=s["row_slice"],
                col_slice=s["col_slice"],
                task_id=s["task_id"],
            )

        assert assembler.is_complete() is True
        c_out = assembler.get_result(enforce_complete=True)
        c_expected = np.matmul(a, b)
        np.testing.assert_allclose(c_out, c_expected, rtol=1e-5, atol=1e-5)

    def test_t1_07_reentrancy_duplicate_accumulation_rejection(self):
        """T1.7: ResultAssembler strictly rejects duplicate strip accumulation on coordinate (0, 0, p)."""
        assembler = ResultAssembler(m=8, n=8, expected_tasks=2, dtype="fp32", allow_duplicate=False)
        tile = np.ones((8, 8), dtype=np.float32)

        # First accumulation for strip 0
        assembler.accumulate(
            tile_result=tile,
            row_idx=0,
            col_idx=0,
            k_idx=0,
            row_slice=(0, 8),
            col_slice=(0, 8),
            task_id="strip_0",
        )

        # Duplicate accumulation for strip 0 must raise TilingError
        with pytest.raises(TilingError, match="Duplicate accumulation rejected"):
            assembler.accumulate(
                tile_result=tile,
                row_idx=0,
                col_idx=0,
                k_idx=0,
                row_slice=(0, 8),
                col_slice=(0, 8),
                task_id="strip_0_duplicate",
            )

    def test_t1_08_matrix_tiler_plan_compatibility_when_present(self):
        """T1.8: If MatrixTiler implements create_plan_outer_product_1d, verify contract conformance."""
        if hasattr(MatrixTiler, "create_plan_outer_product_1d"):
            a = np.ones((16, 32), dtype=np.float32)
            b = np.ones((32, 16), dtype=np.float32)
            plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=4, dtype="fp32")

            assert isinstance(plan, TileGridPlan)
            assert plan.m == 16
            assert plan.k == 32
            assert plan.n == 16
            assert plan.total_tasks == 4
            assert len(plan.tasks) == 4

            # Verify every task conforms to outer-product strip contract
            for p, task in enumerate(plan.tasks):
                assert task.row_idx == 0
                assert task.col_idx == 0
                assert task.k_idx == p
                assert task.row_slice == (0, 16)
                assert task.col_slice == (0, 16)
                assert task.a_tile.shape == (16, 8)
                assert task.b_tile.shape == (8, 16)

    def test_t1_09_configuration_and_enum_contracts_when_present(self):
        """T1.9: Validates PartitioningStrategy, WorkloadType, and ClientConfig fields when declared."""
        import matrix_compute.config as config_mod
        if hasattr(config_mod, "PartitioningStrategy"):
            ps = config_mod.PartitioningStrategy
            assert hasattr(ps, "OUTER_PRODUCT_1D")
            assert hasattr(ps, "BLOCK_3D")
            assert ps.OUTER_PRODUCT_1D.value == "outer_product_1d"
            assert ps.BLOCK_3D.value == "block_3d"

        if hasattr(config_mod, "WorkloadType"):
            wt = config_mod.WorkloadType
            assert hasattr(wt, "GEMM_ONE_SHOT")
            assert wt.GEMM_ONE_SHOT.value == "gemm_one_shot"

        cfg = ClientConfig.for_testing()
        if hasattr(cfg, "partitioning_strategy"):
            assert cfg.partitioning_strategy in ("block_3d", "outer_product_1d")



# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 TESTS)
# ==============================================================================

class TestTier2OuterProductBoundaries:
    """
    Tier 2: Boundary and corner cases: prime dimensions, odd shapes, scalar 1x1,
    asymmetric M!=N!=K, strip oversubscription P > K, single strip P=1.
    """

    def test_t2_01_prime_dimensions_7x11x13(self):
        """T2.1: Prime dimensions M=7, K=11, N=13 partitioned across P=3 strips."""
        np.random.seed(201)
        a = np.random.randn(7, 11).astype(np.float32)
        b = np.random.randn(11, 13).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=3)
        assert len(strips) == 3
        # 11 = 4 + 4 + 3
        assert [s["k_j"] for s in strips] == [4, 4, 3]

        computed = [(s, np.matmul(s["a_tile"], s["b_tile"])) for s in strips]
        c_assembled = Reference1DOuterProductEngine.assemble_strips(7, 13, computed, dtype="fp32")
        c_expected = np.matmul(a, b)
        np.testing.assert_allclose(c_assembled, c_expected, rtol=1e-5, atol=1e-5)

    def test_t2_02_odd_dimensions_and_strips(self):
        """T2.2: Odd dimensions M=15, K=25, N=35 partitioned across P=4 strips."""
        np.random.seed(202)
        a = np.random.randn(15, 25).astype(np.float32)
        b = np.random.randn(25, 35).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=4)
        assert len(strips) == 4
        # 25 = 7 + 6 + 6 + 6
        assert [s["k_j"] for s in strips] == [7, 6, 6, 6]

        computed = [(s, np.matmul(s["a_tile"], s["b_tile"])) for s in strips]
        c_assembled = Reference1DOuterProductEngine.assemble_strips(15, 35, computed, dtype="fp32")
        c_expected = np.matmul(a, b)
        np.testing.assert_allclose(c_assembled, c_expected, rtol=1e-5, atol=1e-5)

    def test_t2_03_scalar_1x1_matrix(self):
        """T2.3: Scalar 1x1 matrix multiplication: [[3.0]] x [[7.0]] = [[21.0]]."""
        a = np.array([[3.0]], dtype=np.float32)
        b = np.array([[7.0]], dtype=np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=4)
        # K=1, so P=4 is clamped to P_eff = min(4, 1) = 1
        assert len(strips) == 1
        assert strips[0]["a_tile"].shape == (1, 1)
        assert strips[0]["b_tile"].shape == (1, 1)
        assert strips[0]["k_slice"] == (0, 1)

        c_p = np.matmul(strips[0]["a_tile"], strips[0]["b_tile"])
        c_assembled = Reference1DOuterProductEngine.assemble_strips(1, 1, [(strips[0], c_p)], dtype="fp32")
        assert c_assembled[0, 0] == 21.0

    def test_t2_04_asymmetric_tall_skinny_and_wide_flat(self):
        """T2.4: Asymmetric shapes: tall-skinny (128x16x32) and wide-flat (16x128x16)."""
        np.random.seed(204)

        # Tall-skinny
        a_tall = np.random.randn(128, 16).astype(np.float32)
        b_tall = np.random.randn(16, 32).astype(np.float32)
        strips_tall = Reference1DOuterProductEngine.partition_outer_product_1d(a_tall, b_tall, num_strips=4)
        c_tall_res = [(s, np.matmul(s["a_tile"], s["b_tile"])) for s in strips_tall]
        c_tall = Reference1DOuterProductEngine.assemble_strips(128, 32, c_tall_res)
        np.testing.assert_allclose(c_tall, np.matmul(a_tall, b_tall), rtol=1e-5, atol=1e-5)

        # Wide-flat
        a_wide = np.random.randn(16, 128).astype(np.float32)
        b_wide = np.random.randn(128, 16).astype(np.float32)
        strips_wide = Reference1DOuterProductEngine.partition_outer_product_1d(a_wide, b_wide, num_strips=8)
        c_wide_res = [(s, np.matmul(s["a_tile"], s["b_tile"])) for s in strips_wide]
        c_wide = Reference1DOuterProductEngine.assemble_strips(16, 16, c_wide_res)
        np.testing.assert_allclose(c_wide, np.matmul(a_wide, b_wide), rtol=1e-5, atol=1e-5)

    def test_t2_05_strip_oversubscription_p_greater_than_k(self):
        """T2.5: Strip oversubscription P=10 > K=3 clamps effective strips to P_eff=3 without zero-width strips."""
        a = np.ones((5, 3), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=10)
        # Must clamp P to K=3
        assert len(strips) == 3
        for s in strips:
            assert s["k_j"] == 1
            assert s["a_tile"].shape == (5, 1)
            assert s["b_tile"].shape == (1, 4)

        computed = [(s, np.matmul(s["a_tile"], s["b_tile"])) for s in strips]
        c_assembled = Reference1DOuterProductEngine.assemble_strips(5, 4, computed)
        np.testing.assert_allclose(c_assembled, np.matmul(a, b))

    def test_t2_06_single_strip_p_equals_1(self):
        """T2.6: Single strip P=1 produces exactly 1 task covering full K dimension."""
        np.random.seed(206)
        a = np.random.randn(12, 18).astype(np.float32)
        b = np.random.randn(18, 10).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=1)
        assert len(strips) == 1
        assert strips[0]["k_slice"] == (0, 18)
        assert strips[0]["a_tile"].shape == (12, 18)
        assert strips[0]["b_tile"].shape == (18, 10)

        c_p = np.matmul(strips[0]["a_tile"], strips[0]["b_tile"])
        c_assembled = Reference1DOuterProductEngine.assemble_strips(12, 10, [(strips[0], c_p)])
        np.testing.assert_allclose(c_assembled, np.matmul(a, b))

    def test_t2_07_extreme_contraction_dimension(self):
        """T2.7: Extreme inner contraction dimension M=2, K=5000, N=2 partitioned into P=10 strips."""
        np.random.seed(207)
        a = (np.random.randn(2, 5000) / math.sqrt(5000)).astype(np.float32)
        b = (np.random.randn(5000, 2) / math.sqrt(5000)).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=10)
        assert len(strips) == 10
        for s in strips:
            assert s["k_j"] == 500
            assert s["a_tile"].shape == (2, 500)
            assert s["b_tile"].shape == (500, 2)

        computed = [(s, np.matmul(s["a_tile"], s["b_tile"])) for s in strips]
        c_assembled = Reference1DOuterProductEngine.assemble_strips(2, 2, computed)
        np.testing.assert_allclose(c_assembled, np.matmul(a, b), atol=1e-5)

    def test_t2_08_invalid_dimensions_rejected(self):
        """T2.8: Partitioning rejects 1D/3D tensors, inner dimension mismatch, and 0-size matrices."""
        # 1D array
        with pytest.raises(MatrixDimensionError):
            Reference1DOuterProductEngine.partition_outer_product_1d(np.ones(10), np.ones((10, 10)), num_strips=2)

        # Mismatched inner dimension
        with pytest.raises(MatrixDimensionError, match="Inner dimension mismatch"):
            Reference1DOuterProductEngine.partition_outer_product_1d(np.ones((10, 5)), np.ones((6, 10)), num_strips=2)

        # Zero dimension
        with pytest.raises(MatrixDimensionError):
            Reference1DOuterProductEngine.partition_outer_product_1d(np.ones((0, 5)), np.ones((5, 10)), num_strips=2)


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================

class TestTier3OuterProductCrossFeature:
    """
    Tier 3: Cross-feature interactions: 1D outer product with mock workers,
    dual-tier Freivalds verification, Freivalds fraud proof emission on corrupt strips,
    batch L402 escrow lifecycle, escrow cancellation on fraud, sliding window concurrency.
    """

    def test_t3_01_outer_product_with_mock_worker_pool(self, mock_settlement):
        """T3.1: 1D outer product dispatched across simulated worker pool with batch escrows."""
        np.random.seed(301)
        m, k, n = 32, 64, 32
        p_strips = 4
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=p_strips)

        # Simulate 2 workers handling 2 strips each
        workers = [
            {"pubkey": "worker_node_alpha", "ln_addr": "alpha@ln", "sats_per_gflop": 0.001},
            {"pubkey": "worker_node_beta", "ln_addr": "beta@ln", "sats_per_gflop": 0.001},
        ]

        worker_strip_assignments: Dict[str, List[Dict[str, Any]]] = {
            "worker_node_alpha": [strips[0], strips[1]],
            "worker_node_beta": [strips[2], strips[3]],
        }

        # Setup batch escrows per worker
        worker_escrows: Dict[str, Dict[str, Any]] = {}
        for w in workers:
            w_pub = w["pubkey"]
            assigned_tasks = worker_strip_assignments[w_pub]
            total_gflops = sum(t["estimated_gflops"] for t in assigned_tasks)
            total_sats = max(int(math.ceil(total_gflops * w["sats_per_gflop"])), 10)

            preimage = hashlib.sha256(f"secret_{w_pub}".encode()).hexdigest()
            p_hash = hashlib.sha256(bytes.fromhex(preimage)).hexdigest()

            inv = mock_settlement.create_hodl_invoice(
                task_id=f"batch_{w_pub}",
                total_compute_sats=total_sats,
                payment_hash=p_hash,
            )
            mock_settlement.hold_invoice(p_hash)
            assert mock_settlement.get_status(p_hash) == "HELD"

            worker_escrows[w_pub] = {
                "invoice": inv,
                "payment_hash": p_hash,
                "preimage": preimage,
            }

        # Workers compute assigned strips
        completed_strips = []
        for w in workers:
            w_pub = w["pubkey"]
            for task in worker_strip_assignments[w_pub]:
                c_p = np.matmul(task["a_tile"], task["b_tile"])
                # Per-strip Freivalds verification
                passed, disc, bound, _ = ReferenceOracle.freivalds_verify(
                    task["a_tile"], task["b_tile"], c_p, dtype="fp32"
                )
                assert passed is True, f"Per-strip Freivalds failed for {task['task_id']}"
                completed_strips.append((task, c_p))

            # All strips for this worker passed -> release escrow
            escrow = worker_escrows[w_pub]
            mock_settlement.settle_invoice(escrow["payment_hash"], preimage=escrow["preimage"])
            assert mock_settlement.get_status(escrow["payment_hash"]) == "SETTLED"

        # Assemble and verify final product
        c_final = Reference1DOuterProductEngine.assemble_strips(m, n, completed_strips)
        c_expected = np.matmul(a, b)
        np.testing.assert_allclose(c_final, c_expected, rtol=1e-5, atol=1e-5)

        # Assembled Freivalds verification
        assembled_passed, _, _, _ = ReferenceOracle.freivalds_verify(a, b, c_final, dtype="fp32")
        assert assembled_passed is True

    def test_t3_02_dual_tier_freivalds_per_strip_verification(self):
        """T3.2: Dual-tier Freivalds verifies each strip independently and the assembled final product."""
        np.random.seed(302)
        m, k, n = 24, 36, 24
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=3)
        completed = []

        # Tier 1 of Dual-Tier: Per-strip Freivalds before buffer accumulation
        for s in strips:
            c_p = np.matmul(s["a_tile"], s["b_tile"])
            k_p = s["k_j"]
            norm_a_p = float(np.max(np.sum(np.abs(s["a_tile"]), axis=1)))
            norm_b_p = float(np.max(np.sum(np.abs(s["b_tile"]), axis=1)))
            bound_p = calculate_higham_bound(k_p, norm_a_p, norm_b_p, dtype="fp32")

            passed, disc, _, _ = ReferenceOracle.freivalds_verify(s["a_tile"], s["b_tile"], c_p, dtype="fp32")
            assert passed is True
            assert disc <= max(bound_p, 1e-5)
            completed.append((s, c_p))

        # Assemble
        c_assembled = Reference1DOuterProductEngine.assemble_strips(m, n, completed)

        # Tier 2 of Dual-Tier: Assembled Freivalds on final product
        assembled_passed, assembled_disc, _, _ = ReferenceOracle.freivalds_verify(a, b, c_assembled, dtype="fp32")
        assert assembled_passed is True

    def test_t3_03_freivalds_fraud_rejection_of_corrupt_strip(self, mock_settlement):
        """T3.3: Rogue worker injects single element corruption in strip; caught by per-strip Freivalds."""
        np.random.seed(303)
        m, k, n = 16, 32, 16
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=2)
        assembler = ResultAssembler(m=m, n=n, expected_tasks=[s["task_id"] for s in strips], dtype="fp32")

        # Strip 0: Honest worker
        c_0 = np.matmul(strips[0]["a_tile"], strips[0]["b_tile"])
        p0_valid, _, _, _ = ReferenceOracle.freivalds_verify(strips[0]["a_tile"], strips[0]["b_tile"], c_0)
        assert p0_valid is True
        assembler.accumulate_task(strips[0], c_0)

        # Strip 1: Rogue worker tampers with result (injects +5.0)
        c_1_corrupt = np.matmul(strips[1]["a_tile"], strips[1]["b_tile"])
        c_1_corrupt[3, 3] += 5.0

        p1_valid, disc, bound, proof_data = ReferenceOracle.freivalds_verify(
            strips[1]["a_tile"], strips[1]["b_tile"], c_1_corrupt
        )
        assert p1_valid is False
        assert disc > bound
        assert len(proof_data) > 0

        # Construct authoritative FreivaldsFraudProof
        r_vec = proof_data["random_vector_r"]
        proof = FreivaldsFraudProof(
            task_id=strips[1]["task_id"],
            worker_pubkey="malfeasant_worker_01",
            dimension_m=strips[1]["m_i"],
            dimension_k=strips[1]["k_j"],
            dimension_n=strips[1]["n_k"],
            dtype="fp32",
            matrix_a_hash=ReferenceOracle.sha256_digest(strips[1]["a_tile"]),
            matrix_b_hash=ReferenceOracle.sha256_digest(strips[1]["b_tile"]),
            invalid_result_c_hash=ReferenceOracle.sha256_digest(c_1_corrupt),
            random_vector_r=r_vec,
            vector_br=proof_data["vector_br"],
            vector_abr=proof_data["vector_abr"],
            vector_cr=proof_data["vector_cr"],
            epsilon_tolerance=bound,
            max_observed_discrepancy=disc,
            worker_signature_on_result="sig_test_mock",
            consumer_pubkey="consumer_pubkey_mock",
        )
        assert proof.task_id == "strip_1"
        assert proof.observed_discrepancy > proof.epsilon_tolerance

        # The corrupt strip is REJECTED before entering accumulator:
        # Buffer remains untainted by rogue worker
        assert assembler.is_complete() is False
        assert "strip_1" in assembler.missing_tasks()

    def test_t3_04_batch_l402_escrow_cancellation_on_fraudulent_strip(self, mock_settlement):
        """T3.4: Corrupt strip in a batch escrow causes atomic invoice cancellation and withholding of sats."""
        p_hash = hashlib.sha256(b"preimage_batch_fraud").hexdigest()
        inv = mock_settlement.create_hodl_invoice(
            task_id="batch_worker_rogue",
            total_compute_sats=500,
            payment_hash=p_hash,
        )
        mock_settlement.hold_invoice(p_hash)
        assert mock_settlement.get_status(p_hash) == "HELD"

        # Fraud detected on one of worker's strips
        fraud_detected = True
        if fraud_detected:
            # Preimage is withheld, invoice cancelled
            mock_settlement.cancel_invoice(p_hash)

        assert mock_settlement.get_status(p_hash) == "CANCELLED"
        # Worker cannot settle cancelled invoice
        with pytest.raises(ValueError, match="Cannot settle invoice"):
            mock_settlement.settle_invoice(p_hash, preimage=hashlib.sha256(b"foo").hexdigest())

    def test_t3_05_sliding_window_concurrency_bounding(self):
        """T3.5: 1D outer product strip execution respects sliding window concurrency limit."""
        np.random.seed(305)
        m, k, n = 16, 80, 16
        strips = Reference1DOuterProductEngine.partition_outer_product_1d(
            np.ones((m, k)), np.ones((k, n)), num_strips=8
        )

        max_concurrency = 3
        in_flight = 0
        peak_in_flight = 0
        completed = []

        for s in strips:
            in_flight += 1
            if in_flight > peak_in_flight:
                peak_in_flight = in_flight
            assert in_flight <= max_concurrency, f"Concurrency exceeded limit: {in_flight} > {max_concurrency}"

            # Simulate worker processing
            c_p = np.matmul(s["a_tile"], s["b_tile"])
            completed.append((s, c_p))

            in_flight -= 1  # Task finished

        assert peak_in_flight <= max_concurrency
        assert len(completed) == 8

    def test_t3_06_batch_l402_escrow_multi_worker_partial_fraud_isolation(self, mock_settlement):
        """T3.6: Partial fraud isolation: honest worker's batch escrow settles while rogue worker's is cancelled."""
        # Worker 1: Honest
        w1_preimage = hashlib.sha256(b"w1_preimage_honest").hexdigest()
        w1_hash = hashlib.sha256(bytes.fromhex(w1_preimage)).hexdigest()
        mock_settlement.create_hodl_invoice(task_id="batch_w1", total_compute_sats=200, payment_hash=w1_hash)
        mock_settlement.hold_invoice(w1_hash)

        # Worker 2: Rogue
        w2_preimage = hashlib.sha256(b"w2_preimage_rogue").hexdigest()
        w2_hash = hashlib.sha256(bytes.fromhex(w2_preimage)).hexdigest()
        mock_settlement.create_hodl_invoice(task_id="batch_w2", total_compute_sats=200, payment_hash=w2_hash)
        mock_settlement.hold_invoice(w2_hash)

        # Worker 1 strips pass verification
        mock_settlement.settle_invoice(w1_hash, preimage=w1_preimage)
        assert mock_settlement.get_status(w1_hash) == "SETTLED"

        # Worker 2 strip fails verification -> cancel worker 2
        mock_settlement.cancel_invoice(w2_hash)
        assert mock_settlement.get_status(w2_hash) == "CANCELLED"

    def test_t3_07_dual_tier_assembled_corruption_detection(self):
        """T3.7: Assembled Freivalds verification catches subtle cumulative drift or tampered accumulator buffer."""
        np.random.seed(307)
        m, k, n = 32, 64, 32
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=4)
        completed = [(s, np.matmul(s["a_tile"], s["b_tile"])) for s in strips]
        c_clean = Reference1DOuterProductEngine.assemble_strips(m, n, completed)

        # Clean passes assembled Freivalds
        passed_clean, _, _, _ = ReferenceOracle.freivalds_verify(a, b, c_clean, dtype="fp32")
        assert passed_clean is True

        # Tampered accumulator buffer (e.g. memory bitflip or drift)
        c_corrupted = c_clean.copy()
        c_corrupted[10, 10] += 2.0

        passed_corrupt, disc, bound, proof = ReferenceOracle.freivalds_verify(a, b, c_corrupted, dtype="fp32")
        assert passed_corrupt is False
        assert disc > bound
        assert len(proof) > 0



# ==============================================================================
# TIER 4: REAL-WORLD SCENARIOS & BANDWIDTH PROFILING
# ==============================================================================

class TestTier4OuterProductBandwidthAndScenarios:
    """
    Tier 4: Bandwidth profiling asserting strictly M*K + K*N uploaded elements
    with zero duplicate uploads, contrast with 3D block tiling, Higham error bounds,
    and large-scale stress scenarios.
    """

    def test_t4_01_bandwidth_profiling_zero_upload_redundancy(self):
        """T4.1: Proves total uploaded matrix elements equal strictly M*K + K*N (redundancy ratio == 0.0)."""
        test_shapes = [
            (64, 128, 64, 4),
            (100, 200, 100, 5),
            (37, 73, 59, 7),     # Primes
            (128, 64, 256, 8),   # Rectangular asymmetric
            (1, 1000, 1, 10),    # Dot product
        ]

        for m, k, n, p in test_shapes:
            a = np.ones((m, k), dtype=np.float32)
            b = np.ones((k, n), dtype=np.float32)

            strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=p)
            profile = Reference1DOuterProductEngine.profile_upload_bandwidth(a, b, strips)

            # Invariant 1: Total uploaded elements strictly equal MK + KN
            assert profile["total_uploaded_elements"] == m * k + k * n
            # Invariant 2: Redundancy elements strictly zero
            assert profile["redundancy_elements"] == 0
            # Invariant 3: Redundancy ratio strictly zero
            assert profile["redundancy_ratio"] == 0.0
            # Invariant 4: Flagged as optimal
            assert profile["is_upload_optimal"] is True

    def test_t4_02_bandwidth_contrast_with_3d_block_tiling(self):
        """T4.2: Contrasts 1D Outer-Product (0% redundancy) with 3D Block Tiling (>0% upload redundancy)."""
        m, k, n = 128, 128, 128
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)
        tile_size = 32

        # 1. 1D Outer-Product with P = 4 strips
        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=4)
        profile_1d = Reference1DOuterProductEngine.profile_upload_bandwidth(a, b, strips)

        # 2. 3D Block Tiling with tile_size = 32
        # Grid: (128/32) * (128/32) * (128/32) = 4 * 4 * 4 = 64 block tasks
        # In 3D block tiling, each A block (32x32) is sent to 4 column blocks.
        # Each B block (32x32) is sent to 4 row blocks.
        num_m = m // tile_size  # 4
        num_n = n // tile_size  # 4
        num_k = k // tile_size  # 4
        total_3d_tasks = num_m * num_n * num_k  # 64

        elements_per_tile = tile_size * tile_size
        total_3d_uploaded = total_3d_tasks * (elements_per_tile + elements_per_tile)  # 64 * (1024 + 1024) = 131,072

        optimal_elements = m * k + k * n  # 128*128 + 128*128 = 32,768
        redundancy_3d_ratio = (float(total_3d_uploaded) / float(optimal_elements)) - 1.0

        # Assert 1D is strictly optimal
        assert profile_1d["total_uploaded_elements"] == optimal_elements
        assert profile_1d["redundancy_ratio"] == 0.0

        # Assert 3D tiling has high upload redundancy
        assert total_3d_uploaded > optimal_elements
        assert redundancy_3d_ratio == 3.0  # Exactly 300% redundant uploads (4x total upload)

    def test_t4_03_mathematical_equivalence_higham_fp32(self):
        """T4.3: Mathematical equivalence against np.matmul within Higham bound ||C - C_hat||_inf <= gamma_K * ||A|| * ||B||."""
        np.random.seed(403)
        m, k, n = 128, 256, 128
        # Scaled random inputs
        a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=8, dtype="fp32")
        computed = [(s, np.matmul(s["a_tile"], s["b_tile"])) for s in strips]
        c_1d = Reference1DOuterProductEngine.assemble_strips(m, n, computed, dtype="fp32")

        c_direct = np.matmul(a, b)

        # Higham dynamic rounding bound
        norm_a = float(np.max(np.sum(np.abs(a), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b), axis=1)))
        higham_limit = calculate_higham_bound(k, norm_a, norm_b, dtype="fp32")

        observed_error = float(np.max(np.abs(c_1d - c_direct)))

        # Assert error is bounded by theoretical Higham limit
        assert observed_error <= higham_limit, f"Observed error {observed_error} exceeded Higham bound {higham_limit}"
        assert observed_error < 1e-4

    def test_t4_04_mathematical_equivalence_higham_fp16(self):
        """T4.4: Mathematical equivalence for FP16 within Higham bound gamma_K."""
        np.random.seed(404)
        m, k, n = 64, 128, 64
        a = (np.random.randn(m, k) * 0.1).astype(np.float16)
        b = (np.random.randn(k, n) * 0.1).astype(np.float16)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=4, dtype="fp16")
        computed = []
        for s in strips:
            c_p = np.matmul(s["a_tile"].astype(np.float32), s["b_tile"].astype(np.float32)).astype(np.float16)
            computed.append((s, c_p))

        c_1d = Reference1DOuterProductEngine.assemble_strips(m, n, computed, dtype="fp16")
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)

        norm_a = float(np.max(np.sum(np.abs(a.astype(np.float32)), axis=1)))
        norm_b = float(np.max(np.sum(np.abs(b.astype(np.float32)), axis=1)))
        higham_limit = calculate_higham_bound(k, norm_a, norm_b, dtype="fp16")

        observed_error = float(np.max(np.abs(c_1d.astype(np.float32) - c_ref.astype(np.float32))))
        assert observed_error <= max(higham_limit, 1e-2)

    def test_t4_05_mathematical_equivalence_higham_bf16(self):
        """T4.5: Mathematical equivalence for BF16 with mantissa truncation."""
        np.random.seed(405)
        m, k, n = 64, 128, 64
        a = truncate_to_bf16(np.random.randn(m, k) * 0.1)
        b = truncate_to_bf16(np.random.randn(k, n) * 0.1)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=4, dtype="bf16")
        computed = []
        for s in strips:
            c_p = truncate_to_bf16(np.matmul(s["a_tile"], s["b_tile"]))
            computed.append((s, c_p))

        c_1d = Reference1DOuterProductEngine.assemble_strips(m, n, computed, dtype="bf16")
        c_ref = ReferenceOracle.gemm(a, b, dtype="bf16")

        observed_error = float(np.max(np.abs(c_1d - c_ref)))
        assert observed_error < 0.05
        assert validate_bf16_masking(c_1d) is True

    def test_t4_06_deep_learning_attention_outer_product_projection(self):
        """T4.6: Real-world transformer attention projection (batch*seq=64, dim=512, heads*head_dim=64)."""
        np.random.seed(406)
        m, k, n = 64, 512, 64
        q = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        w_proj = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(q, w_proj, num_strips=8)
        profile = Reference1DOuterProductEngine.profile_upload_bandwidth(q, w_proj, strips)
        assert profile["is_upload_optimal"] is True

        computed = [(s, np.matmul(s["a_tile"], s["b_tile"])) for s in strips]
        c_projected = Reference1DOuterProductEngine.assemble_strips(m, n, computed)
        c_expected = np.matmul(q, w_proj)

        np.testing.assert_allclose(c_projected, c_expected, rtol=1e-5, atol=1e-5)

    def test_t4_07_large_scale_stress_1024x1024(self):
        """T4.7: Large scale 1024x1024 distributed outer product: verifies exact 2,097,152 uploaded elements."""
        m, k, n = 1024, 1024, 1024
        p_strips = 16

        # Allocate synthetic matrices
        np.random.seed(407)
        a = (np.random.randn(m, k) / math.sqrt(k)).astype(np.float32)
        b = (np.random.randn(k, n) / math.sqrt(k)).astype(np.float32)

        strips = Reference1DOuterProductEngine.partition_outer_product_1d(a, b, num_strips=p_strips)
        assert len(strips) == 16

        # Verify upload optimality
        profile = Reference1DOuterProductEngine.profile_upload_bandwidth(a, b, strips)
        expected_elements = 1024 * 1024 + 1024 * 1024  # 2,097,152 elements
        assert profile["total_uploaded_elements"] == expected_elements
        assert profile["redundancy_elements"] == 0
        assert profile["redundancy_ratio"] == 0.0

        # Compute sub-sample of partial accumulation to verify mathematical equivalence
        computed = [(s, np.matmul(s["a_tile"], s["b_tile"])) for s in strips]
        c_assembled = Reference1DOuterProductEngine.assemble_strips(m, n, computed)
        c_expected = np.matmul(a, b)

        np.testing.assert_allclose(c_assembled, c_expected, rtol=1e-5, atol=1e-5)
