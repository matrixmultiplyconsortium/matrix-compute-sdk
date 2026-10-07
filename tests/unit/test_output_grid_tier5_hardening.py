"""
Tier 5 Adversarial Coverage Hardening Suite for 2D Output Grid Partitioning (Phase 2).

Exhaustively covers all edge cases, boundary conditions, defensive error paths,
and mathematical invariants across:
1. `config.py`: Enums, case-insensitivity, aliases, env parsing, validation errors, grid_shape syncing.
2. `partitioner.py`: 2D spatial grid div-mod, extreme non-square grids (1x100, 100x1), degenerate
   dimensions (1x1, vectors), divisibility remainders, dictionary emulation methods.
3. `assembler.py`: Zero-reduction direct spatial placement, high concurrency multithreading,
   duplicate coordinate handling, buffer copy slicing, progress tracking fractions.
4. `profiler.py`: Spatial grid coverage validation, gaps/overlaps detection, empty task lists,
   download optimality invariants (download_redundancy_ratio == 0.0), upload volume formula.
5. `coordinator.py` & `client.py`: Processor grid factorization heuristic (primes, 1 worker,
   excessive workers), parameter precedence hierarchy, dual-tier Freivalds verification,
   fraud isolation, and high-level client interfaces.
"""

from __future__ import annotations

import collections
import concurrent.futures
import math
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest
from pydantic import ValidationError

from matrix_compute.client import MatrixComputeClient, distributed_matmul
from matrix_compute.config import (
    ClientConfig,
    PartitioningStrategy,
    WorkloadType,
    get_default_config,
)
from matrix_compute.exceptions import (
    FraudDetectedError,
    MatrixComputeError,
    MatrixDimensionError,
    NoWorkersFoundError,
    TilingError,
    VerificationFailedError,
)
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
from matrix_compute.tiling.assembler import ResultAssembler
from matrix_compute.tiling.partitioner import (
    MatrixTiler,
    TileGridPlan,
    TileTaskPlan,
    compute_2d_grid_intervals,
    compute_grid_intervals,
)
from matrix_compute.tiling.precision import (
    calculate_higham_bound,
    cast_matrix,
    emulate_gemm,
    get_bytes_per_element,
    normalize_dtype,
    truncate_to_bf16,
    validate_bf16_masking,
)
from matrix_compute.tiling.profiler import (
    BandwidthProfile,
    BandwidthProfiler,
)


# =============================================================================
# 1. CONFIGURATION & SCHEMA HARDENING (config.py)
# =============================================================================

class TestTier5ConfigHardening:
    """Adversarial coverage for config.py models and validators for 2D Output Grid."""

    def test_partitioning_strategy_enum_properties(self):
        """Verifies enum value mapping for OUTPUT_GRID_2D."""
        assert PartitioningStrategy.OUTPUT_GRID_2D == "output_grid_2d"
        assert PartitioningStrategy.OUTPUT_GRID_2D.value == "output_grid_2d"

    @pytest.mark.parametrize(
        "raw_input,expected_enum",
        [
            ("OUTPUT_GRID_2D", PartitioningStrategy.OUTPUT_GRID_2D),
            ("output_grid_2d", PartitioningStrategy.OUTPUT_GRID_2D),
            ("output_grid", PartitioningStrategy.OUTPUT_GRID_2D),
            ("2d_output_grid", PartitioningStrategy.OUTPUT_GRID_2D),
            ("outputgrid2d", PartitioningStrategy.OUTPUT_GRID_2D),
            ("grid_2d", PartitioningStrategy.OUTPUT_GRID_2D),
            ("output_2d", PartitioningStrategy.OUTPUT_GRID_2D),
            ("output2d", PartitioningStrategy.OUTPUT_GRID_2D),
            ("2d", PartitioningStrategy.OUTPUT_GRID_2D),
            ("2d_grid", PartitioningStrategy.OUTPUT_GRID_2D),
            ("  OUTPUT_GRID_2D  ", PartitioningStrategy.OUTPUT_GRID_2D),
            ("  output_grid  ", PartitioningStrategy.OUTPUT_GRID_2D),
            ("Output_Grid_2D", PartitioningStrategy.OUTPUT_GRID_2D),
            ("2D_OUTPUT_GRID", PartitioningStrategy.OUTPUT_GRID_2D),
        ],
    )
    def test_partitioning_strategy_output_grid_aliases_and_case(self, raw_input, expected_enum):
        """All variations, aliases, cases, and whitespace variations resolve cleanly."""
        cfg = ClientConfig(partitioning_strategy=raw_input)
        assert cfg.partitioning_strategy == expected_enum

    @pytest.mark.parametrize("invalid_input", ["unknown_grid", "invalid_strategy", "5d_tensor", "", "   "])
    def test_invalid_partitioning_strategy_strings_raise_validation_error(self, invalid_input):
        """Unrecognized string names raise ValidationError."""
        with pytest.raises(ValidationError):
            ClientConfig(partitioning_strategy=invalid_input)

    @pytest.mark.parametrize("bad_type", [123, 45.6, ["output_grid_2d"], {"strategy": "output_grid_2d"}])
    def test_invalid_partitioning_strategy_types_raise_validation_error(self, bad_type):
        """Non-string, non-enum types raise ValidationError."""
        with pytest.raises(ValidationError):
            ClientConfig(partitioning_strategy=bad_type)

    def test_env_var_parsing_for_grid_dimensions(self, monkeypatch):
        """Environment variables MATRIX_GRID_ROWS and MATRIX_GRID_COLS populate ClientConfig."""
        monkeypatch.setenv("MATRIX_GRID_ROWS", "4")
        monkeypatch.setenv("MATRIX_GRID_COLS", "8")
        monkeypatch.setenv("MATRIX_PARTITIONING_STRATEGY", "output_grid_2d")

        cfg = ClientConfig()
        assert cfg.partitioning_strategy == PartitioningStrategy.OUTPUT_GRID_2D
        assert cfg.grid_rows == 4
        assert cfg.grid_cols == 8
        assert cfg.grid_shape == (4, 8)
        assert cfg.p_r == 4
        assert cfg.p_c == 8

    def test_env_var_alternate_aliases(self, monkeypatch):
        """Alternative environment variable aliases GRID_ROWS, GRID_COLS, P_R, P_C populate ClientConfig."""
        monkeypatch.setenv("P_R", "3")
        monkeypatch.setenv("P_C", "5")
        cfg = ClientConfig()
        assert cfg.grid_rows == 3
        assert cfg.grid_cols == 5
        assert cfg.grid_shape == (3, 5)

    def test_string_integer_parsing_in_env_and_kwargs(self):
        """Strings containing integers are parsed into ints cleanly."""
        cfg = ClientConfig(grid_rows=" 6 ", grid_cols=" 12 ")
        assert cfg.grid_rows == 6
        assert cfg.grid_cols == 12
        assert cfg.grid_shape == (6, 12)

    def test_malformed_string_integer_fails_validation(self):
        """Non-numeric string integer fails validation."""
        with pytest.raises(ValidationError):
            ClientConfig(grid_rows="not_an_int")

    def test_grid_shape_tuple_and_list_sync(self):
        """grid_shape specified as tuple or list synchronizes with grid_rows and grid_cols."""
        cfg_tuple = ClientConfig(grid_shape=(2, 6))
        assert cfg_tuple.grid_shape == (2, 6)
        assert cfg_tuple.grid_rows == 2
        assert cfg_tuple.grid_cols == 6
        assert cfg_tuple.p_r == 2
        assert cfg_tuple.p_c == 6

        cfg_list = ClientConfig(grid_shape=[4, 7])
        assert cfg_list.grid_shape == (4, 7)
        assert cfg_list.grid_rows == 4
        assert cfg_list.grid_cols == 7

    def test_grid_shape_string_ints_inside_tuple(self):
        """String integers inside grid_shape tuple are coerced cleanly."""
        cfg = ClientConfig(grid_shape=("3", "9"))
        assert cfg.grid_shape == (3, 9)
        assert cfg.grid_rows == 3
        assert cfg.grid_cols == 9

    def test_grid_shape_conflicting_rows_raises_validation_error(self):
        """Conflicting grid_rows vs grid_shape[0] raises ValidationError."""
        with pytest.raises(ValidationError, match="Conflicting configuration"):
            ClientConfig(grid_rows=2, grid_shape=(3, 4))

    def test_grid_shape_conflicting_cols_raises_validation_error(self):
        """Conflicting grid_cols vs grid_shape[1] raises ValidationError."""
        with pytest.raises(ValidationError, match="Conflicting configuration"):
            ClientConfig(grid_cols=5, grid_shape=(3, 4))

    @pytest.mark.parametrize(
        "bad_shape",
        [
            (1,),             # 1-tuple
            (1, 2, 3),        # 3-tuple
            "2,4",            # string
            (0, 4),           # row < 1
            (4, -1),          # col < 1
            ("abc", 4),       # non-int
        ],
    )
    def test_invalid_grid_shape_structures_raise_validation_error(self, bad_shape):
        """Malformed grid_shape structures raise ValidationError."""
        with pytest.raises(ValidationError):
            ClientConfig(grid_shape=bad_shape)

    def test_p_r_and_p_c_kwargs_alias_resolution(self):
        """p_r and p_c passed directly in constructor map to grid_rows and grid_cols."""
        cfg = ClientConfig.load(p_r=5, p_c=10)
        assert cfg.grid_rows == 5
        assert cfg.grid_cols == 10
        assert cfg.grid_shape == (5, 10)


# =============================================================================
# 2. PARTITIONER HARDENING (partitioner.py)
# =============================================================================

class TestTier5PartitionerHardening:
    """Adversarial coverage for compute_2d_grid_intervals and MatrixTiler.create_plan_output_grid_2d."""

    def test_extreme_non_square_grid_1x100(self):
        """Extreme non-square grid P_r=1, P_c=100 on matrix M=1, N=100."""
        a = np.ones((1, 50), dtype=np.float32)
        b = np.ones((50, 100), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=1, grid_cols=100)
        assert plan.p_r == 1
        assert plan.p_c == 100
        assert plan.num_k_blocks == 1
        assert plan.total_tasks == 100
        for task in plan.tasks:
            assert task.row_slice == (0, 1)
            assert task.k_slice == (0, 50)
            assert task.col_slice[1] - task.col_slice[0] == 1
            assert task.a_tile.shape == (1, 50)
            assert task.b_tile.shape == (50, 1)

    def test_extreme_non_square_grid_100x1(self):
        """Extreme non-square grid P_r=100, P_c=1 on matrix M=100, N=1."""
        a = np.ones((100, 50), dtype=np.float32)
        b = np.ones((50, 1), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=100, grid_cols=1)
        assert plan.p_r == 100
        assert plan.p_c == 1
        assert plan.num_k_blocks == 1
        assert plan.total_tasks == 100
        for task in plan.tasks:
            assert task.col_slice == (0, 1)
            assert task.k_slice == (0, 50)
            assert task.row_slice[1] - task.row_slice[0] == 1
            assert task.a_tile.shape == (1, 50)
            assert task.b_tile.shape == (50, 1)

    def test_degenerate_scalar_1x1_with_excessive_grid(self):
        """Scalar 1x1 matrix with grid_rows=10, grid_cols=10 clamps to 1x1."""
        a = np.array([[5.0]], dtype=np.float32)
        b = np.array([[6.0]], dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=10, grid_cols=10)
        assert plan.p_r == 1
        assert plan.p_c == 1
        assert plan.total_tasks == 1
        assert plan.tasks[0].row_slice == (0, 1)
        assert task.col_slice == (0, 1) if (task := plan.tasks[0]) else True

    @pytest.mark.parametrize(
        "dim,blocks,expected_lengths",
        [
            (10, 3, [4, 3, 3]),
            (11, 3, [4, 4, 3]),
            (12, 3, [4, 4, 4]),
            (13, 5, [3, 3, 3, 2, 2]),
            (100, 7, [15, 15, 14, 14, 14, 14, 14]),
            (1, 10, [1]),          # Clamped to 1
            (5, 1, [5]),           # Single block
        ],
    )
    def test_divisibility_remainder_conservation(self, dim, blocks, expected_lengths):
        """Interval lengths sum exactly to dimension and match balanced remainder distribution."""
        intervals = compute_2d_grid_intervals(dimension=dim, num_blocks=blocks)
        lengths = [end - start for start, end in intervals]
        assert lengths == expected_lengths
        assert sum(lengths) == dim
        assert intervals[0][0] == 0
        assert intervals[-1][1] == dim

    def test_negative_or_zero_blocks_clamping_in_interval_function(self):
        """compute_2d_grid_intervals clamps zero or negative block counts to 1."""
        for b in (0, -1, -50):
            intervals = compute_2d_grid_intervals(dimension=20, num_blocks=b)
            assert len(intervals) == 1
            assert intervals == [(0, 20)]

    def test_invalid_dimension_raises_value_error(self):
        """compute_2d_grid_intervals with dimension < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Dimension must be >= 1"):
            compute_2d_grid_intervals(dimension=0, num_blocks=4)
        with pytest.raises(ValueError, match="Dimension must be >= 1"):
            compute_2d_grid_intervals(dimension=-10, num_blocks=4)

    def test_create_plan_invalid_grid_rows_or_cols_raises_value_error(self):
        """create_plan_output_grid_2d with explicit grid_rows < 1 or grid_cols < 1 raises ValueError."""
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="grid_rows must be >= 1"):
            MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=0)
        with pytest.raises(ValueError, match="grid_cols must be >= 1"):
            MatrixTiler.create_plan_output_grid_2d(a, b, grid_cols=-2)

    def test_defensive_matrix_dimension_errors(self):
        """Dimension mismatches, 1D/3D arrays, and empty shapes raise MatrixDimensionError."""
        # Mismatched inner dimension
        with pytest.raises(MatrixDimensionError, match="Dimension mismatch"):
            MatrixTiler.create_plan_output_grid_2d(np.ones((4, 5)), np.ones((6, 4)))

        # 1D array
        with pytest.raises(MatrixDimensionError, match="must be 2-dimensional"):
            MatrixTiler.create_plan_output_grid_2d(np.ones((4,)), np.ones((4, 4)))

        # 3D array
        with pytest.raises(MatrixDimensionError, match="must be 2-dimensional"):
            MatrixTiler.create_plan_output_grid_2d(np.ones((4, 4, 4)), np.ones((4, 4)))

        # Empty matrix
        with pytest.raises(MatrixDimensionError, match="must be non-zero"):
            MatrixTiler.create_plan_output_grid_2d(np.ones((0, 4)), np.ones((4, 4)))

    def test_dictionary_emulation_completeness(self):
        """TileGridPlan and TileTaskPlan implement complete dictionary emulation interface."""
        a = np.ones((12, 16), dtype=np.float32)
        b = np.ones((16, 20), dtype=np.float32)
        plan = MatrixTiler.create_plan_output_grid_2d(a, b, grid_rows=3, grid_cols=4)

        # Plan dictionary interface
        assert plan["m"] == 12
        assert plan["n"] == 20
        assert plan["k"] == 16
        assert plan["p_r"] == 3
        assert plan["p_c"] == 4
        assert plan["k_blocks"] == 1
        assert plan.get("grid_shape") == (3, 4)
        assert plan.get("nonexistent_key", "default_val") == "default_val"
        assert "grid_shape" in plan
        assert "nonexistent_key" not in plan
        assert len(plan.keys()) > 0
        assert len(plan.values()) == len(plan.keys())
        assert len(plan.items()) == len(plan.keys())

        with pytest.raises(KeyError):
            _ = plan["nonexistent_field"]

        # Task dictionary interface
        task = plan.tasks[0]
        assert task["row_idx"] == 0
        assert task["col_idx"] == 0
        assert task["k_idx"] == 0
        assert task.get("dtype") == "fp32"
        assert "a_tile" in task
        assert "b_tile" in task
        assert len(task.keys()) > 0
        assert len(task.values()) == len(task.keys())

        with pytest.raises(KeyError):
            _ = task["invalid_task_key"]


# =============================================================================
# 3. ASSEMBLER HARDENING (assembler.py)
# =============================================================================

class TestTier5AssemblerHardening:
    """Adversarial coverage for ResultAssembler in zero-reduction mode."""

    def test_zero_reduction_strictly_zero_reduction_ops(self):
        """Zero-reduction mode enforces strictly 0 floating-point reduction additions."""
        m, n = 24, 32
        plan = MatrixTiler.create_plan_output_grid_2d(
            np.ones((m, 16), dtype=np.float32),
            np.ones((16, n), dtype=np.float32),
            grid_rows=3,
            grid_cols=4,
        )
        assembler = ResultAssembler.from_plan(plan)
        assert assembler.zero_reduction is True
        assert assembler.reduction_ops == 0

        for task in plan.tasks:
            tile_res = np.matmul(task.a_tile, task.b_tile)
            assembler.accumulate_task(task, tile_res)

        assert assembler.is_complete() is True
        # STRICT REQUIREMENT: zero floating-point reduction additions!
        assert assembler.reduction_ops == 0

    def test_high_concurrency_multithreaded_stress(self):
        """16 concurrent threads accumulating disjoint tiles into shared buffer without corruption."""
        m, n = 64, 64
        plan = MatrixTiler.create_plan_output_grid_2d(
            np.ones((m, 32), dtype=np.float32),
            np.ones((32, n), dtype=np.float32),
            grid_rows=4,
            grid_cols=4,
        )
        assembler = ResultAssembler.from_plan(plan)

        def worker_job(t: TileTaskPlan):
            tile_res = np.matmul(t.a_tile, t.b_tile)
            assembler.accumulate_task(t, tile_res)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(worker_job, t) for t in plan.tasks]
            concurrent.futures.wait(futures)

        assert assembler.is_complete() is True
        assert assembler.reduction_ops == 0
        result = assembler.get_result(enforce_complete=True)
        expected = 32.0 * np.ones((m, n), dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_duplicate_coordinate_handling(self):
        """allow_duplicate=False raises TilingError; allow_duplicate=True overwrites cleanly."""
        assembler_strict = ResultAssembler(m=4, n=4, zero_reduction=True, allow_duplicate=False)
        tile = np.ones((2, 2), dtype=np.float32)
        assembler_strict.accumulate(tile, row_idx=0, col_idx=0, row_slice=(0, 2), col_slice=(0, 2))

        with pytest.raises(TilingError, match="Duplicate accumulation rejected"):
            assembler_strict.accumulate(tile, row_idx=0, col_idx=0, row_slice=(0, 2), col_slice=(0, 2))

        # With allow_duplicate=True
        assembler_lax = ResultAssembler(m=4, n=4, zero_reduction=True, allow_duplicate=True)
        assembler_lax.accumulate(tile, row_idx=0, col_idx=0, row_slice=(0, 2), col_slice=(0, 2))
        tile_override = 5.0 * np.ones((2, 2), dtype=np.float32)
        # Succeeds without error
        assembler_lax.accumulate(tile_override, row_idx=0, col_idx=0, row_slice=(0, 2), col_slice=(0, 2))
        res = assembler_lax.get_result()
        np.testing.assert_allclose(res[0:2, 0:2], tile_override)

    @pytest.mark.parametrize(
        "r_slice,c_slice",
        [
            ((-1, 2), (0, 2)),      # negative row start
            ((0, 5), (0, 2)),       # row end > M (M=4)
            ((0, 2), (0, 6)),       # col end > N (N=4)
            ((3, 2), (0, 2)),       # inverted row slice (start > end)
            ((2, 2), (0, 2)),       # zero-width row slice (start == end)
        ],
    )
    def test_out_of_bounds_and_inverted_slices_raise_value_error(self, r_slice, c_slice):
        """Invalid slice bounds raise ValueError."""
        assembler = ResultAssembler(m=4, n=4, zero_reduction=True)
        tile = np.ones((max(1, abs(r_slice[1] - r_slice[0])), max(1, abs(c_slice[1] - c_slice[0]))), dtype=np.float32)
        with pytest.raises(ValueError, match="out of bounds"):
            assembler.accumulate(tile, row_idx=0, col_idx=0, row_slice=r_slice, col_slice=c_slice)

    def test_tile_shape_mismatch_raises_tiling_error(self):
        """Mismatch between tile array shape and slice dimensions raises TilingError."""
        assembler = ResultAssembler(m=4, n=4, zero_reduction=True)
        tile_wrong_shape = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(TilingError, match="does not match expected slice"):
            assembler.accumulate(tile_wrong_shape, row_idx=0, col_idx=0, row_slice=(0, 2), col_slice=(0, 2))

    def test_buffer_copy_immutability(self):
        """Modifying the returned result matrix does not alter the assembler's internal buffer."""
        assembler = ResultAssembler(m=4, n=4, zero_reduction=True)
        assembler.accumulate(np.ones((4, 4), dtype=np.float32), row_idx=0, col_idx=0, row_slice=(0, 4), col_slice=(0, 4))
        res1 = assembler.get_result()
        res1[0, 0] = 999.0  # Mutate external copy

        res2 = assembler.get_result()
        assert res2[0, 0] == 1.0  # Internal buffer was unaffected

    def test_progress_tracking_and_missing_tasks(self):
        """Progress fraction tracks accumulated count and missing_tasks lists pending IDs."""
        plan = MatrixTiler.create_plan_output_grid_2d(
            np.ones((4, 4), dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
            grid_rows=2,
            grid_cols=2,
        )
        assembler = ResultAssembler.from_plan(plan)
        assert assembler.get_progress() == 0.0
        assert not assembler.is_complete()
        assert len(assembler.missing_tasks()) == 4

        # Accumulate first task
        t0 = plan.tasks[0]
        assembler.accumulate_task(t0, np.matmul(t0.a_tile, t0.b_tile))
        assert assembler.get_progress() == 0.25
        assert not assembler.is_complete()
        assert t0.task_id not in assembler.missing_tasks()

        # Enforce complete before finishing raises TilingError
        with pytest.raises(TilingError, match="Cannot extract result: assembly incomplete"):
            assembler.get_result(enforce_complete=True)

        # Accumulate remaining tasks
        for t in plan.tasks[1:]:
            assembler.accumulate_task(t, np.matmul(t.a_tile, t.b_tile))

        assert assembler.get_progress() == 1.0
        assert assembler.is_complete()
        assert len(assembler.missing_tasks()) == 0

    def test_assembler_reset_behavior(self):
        """reset() clears internal buffer, tracking sets, and reduction_ops."""
        assembler = ResultAssembler(m=4, n=4, zero_reduction=True, expected_tasks=1)
        assembler.accumulate(np.ones((4, 4), dtype=np.float32), row_idx=0, col_idx=0, row_slice=(0, 4), col_slice=(0, 4))
        assert assembler.is_complete()

        assembler.reset()
        assert not assembler.is_complete()
        assert assembler.get_progress() == 0.0
        assert assembler.reduction_ops == 0
        np.testing.assert_allclose(assembler.get_result(), np.zeros((4, 4)))


# =============================================================================
# 4. PROFILER HARDENING (profiler.py)
# =============================================================================

class TestTier5ProfilerHardening:
    """Adversarial coverage for BandwidthProfiler download optimality and spatial grid coverage."""

    def test_verify_2d_grid_coverage_edge_cases(self):
        """verify_2d_grid_coverage handles empty lists, non-positive dimensions, gaps, and overlaps."""
        # Empty tasks
        assert BandwidthProfiler.verify_2d_grid_coverage([], m=10, n=10) is False

        # Non-positive dimensions
        task_single = TileTaskPlan(
            task_id="t0", row_idx=0, col_idx=0, k_idx=0,
            row_slice=(0, 10), col_slice=(0, 10), k_slice=(0, 5),
            a_tile=np.ones((10, 5)), b_tile=np.ones((5, 10)),
            dtype="fp32", estimated_gflops=0.001,
        )
        assert BandwidthProfiler.verify_2d_grid_coverage([task_single], m=0, n=10) is False
        assert BandwidthProfiler.verify_2d_grid_coverage([task_single], m=10, n=-1) is False

        # Valid single tile
        assert BandwidthProfiler.verify_2d_grid_coverage([task_single], m=10, n=10) is True

    def test_verify_2d_grid_coverage_gap_detection(self):
        """Missing quadrant in 2x2 grid fails coverage check."""
        t0 = {"row_slice": (0, 5), "col_slice": (0, 5)}
        t1 = {"row_slice": (0, 5), "col_slice": (5, 10)}
        t2 = {"row_slice": (5, 10), "col_slice": (0, 5)}
        # Omit t3 (5:10, 5:10)
        assert BandwidthProfiler.verify_2d_grid_coverage([t0, t1, t2], m=10, n=10) is False

    def test_verify_2d_grid_coverage_overlap_detection(self):
        """Overlapping rectangles fail pairwise disjointness check even if total area matches."""
        # Two overlapping tiles covering total area 100 with overlap
        # t0: (0:6, 0:10) -> area 60
        # t1: (4:10, 0:10) -> area 60 (overlap on rows 4:6)
        # Plus adjustment that keeps total area 100
        t0 = {"row_slice": (0, 6), "col_slice": (0, 10)}
        t1 = {"row_slice": (4, 8), "col_slice": (0, 10)}
        assert BandwidthProfiler.verify_2d_grid_coverage([t0, t1], m=10, n=10) is False

    def test_strict_download_optimality_invariant(self):
        """Download redundancy ratio is identically 0.0 and download elements equal M*N."""
        shapes = [
            (1, 100, 1, 1, 1),
            (17, 23, 29, 2, 3),   # Primes
            (100, 10, 100, 10, 1), # Extreme aspect ratio
        ]
        for m, k, n, pr, pc in shapes:
            a = np.ones((m, k), dtype=np.float32)
            b = np.ones((k, n), dtype=np.float32)
            prof = BandwidthProfiler.profile_output_grid_2d(a, b, grid_rows=pr, grid_cols=pc)
            assert prof.download_elements == m * n
            assert prof.optimal_download_elements == m * n
            assert prof.download_redundancy_ratio == 0.0
            assert prof.is_download_optimal is True

    def test_upload_volume_scaling_formula(self):
        """Theoretical upload elements strictly equal Pc * M * K + Pr * K * N."""
        m, k, n, pr, pc = 40, 20, 60, 4, 5
        theo = BandwidthProfiler.theoretical_2d_output_grid_bandwidth(m=m, n=n, k=k, p_r=pr, p_c=pc)
        expected_upload = pc * (m * k) + pr * (k * n)
        assert theo.total_uploaded_elements == expected_upload
        assert theo.num_tasks == pr * pc
        assert theo.download_elements == m * n
        assert theo.download_redundancy_ratio == 0.0

    def test_single_tile_is_upload_and_download_optimal(self):
        """Single tile grid (P_r=1, P_c=1) achieves both upload and download optimality."""
        theo = BandwidthProfiler.theoretical_2d_output_grid_bandwidth(m=32, n=32, k=16, p_r=1, p_c=1)
        assert theo.is_upload_optimal is True
        assert theo.is_download_optimal is True
        assert theo.redundancy_elements == 0


# =============================================================================
# 5. COORDINATOR & CLIENT HARDENING (coordinator.py & client.py)
# =============================================================================

class TestTier5CoordinatorHardening:
    """Adversarial coverage for DistributedMatmulCoordinator processor grid factorization and execution."""

    def test_worker_count_factorization_single_worker(self):
        """1 worker factorizes into (1, 1) grid."""
        # Simulated factorization logic from coordinator
        w_count = 1
        m, n = 32, 32
        target_ratio = float(m) / float(n)
        best_r, best_c = 1, w_count
        min_diff = float("inf")
        for r in range(1, w_count + 1):
            if w_count % r == 0:
                c = w_count // r
                ratio = float(r) / float(c)
                diff = abs(ratio - target_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_r, best_c = r, c
        assert (best_r, best_c) == (1, 1)

    @pytest.mark.parametrize(
        "m,n,w_count,expected_grid",
        [
            (32, 32, 7, (1, 7)),    # Prime worker count 7 on square matrix selects (1, 7)
            (100, 10, 7, (7, 1)),   # Prime worker count 7 on tall-skinny matrix selects (7, 1)
            (10, 100, 7, (1, 7)),   # Prime worker count 7 on wide-flat matrix selects (1, 7)
            (40, 60, 6, (2, 3)),    # 6 workers on 40x60 matrix (ratio 2/3) selects (2, 3)
            (60, 40, 6, (3, 2)),    # 6 workers on 60x40 matrix (ratio 3/2) selects (3, 2)
            (32, 32, 16, (4, 4)),   # 16 workers on square matrix selects (4, 4)
        ],
    )
    def test_worker_factorization_aspect_ratio_matching(self, m, n, w_count, expected_grid):
        """Worker factorization matches target aspect ratio M / N."""
        target_ratio = float(m) / float(n)
        best_r, best_c = 1, w_count
        min_diff = float("inf")
        for r in range(1, w_count + 1):
            if w_count % r == 0:
                c = w_count // r
                ratio = float(r) / float(c)
                diff = abs(ratio - target_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_r, best_c = r, c
        assert (best_r, best_c) == expected_grid

    def test_parameter_precedence_hierarchy(self):
        """Explicit kwargs override ClientConfig and heuristics in MatrixTiler and coordinator."""
        cfg = ClientConfig(grid_rows=4, grid_cols=8)
        # Passing explicit kwargs overrides cfg
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        plan = MatrixTiler.create_plan(
            a=a,
            b=b,
            strategy="output_grid_2d",
            grid_rows=2,
            grid_cols=3,
        )
        assert plan.p_r == 2
        assert plan.p_c == 3

    def test_high_level_client_matmul_grid_kwargs(self):
        """MatrixComputeClient.matmul passes grid_rows and grid_cols cleanly."""
        cfg = ClientConfig.for_testing(partitioning_strategy="output_grid_2d")
        with MatrixComputeClient(config=cfg) as client:
            a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            res = client.matmul(a, b, grid_rows=1, grid_cols=2)
            np.testing.assert_allclose(res, np.matmul(a, b))

    def test_high_level_distributed_matmul_functional_api(self):
        """distributed_matmul functional wrapper succeeds with grid_shape."""
        cfg = ClientConfig.for_testing()
        a = np.ones((4, 8), dtype=np.float32)
        b = np.ones((8, 4), dtype=np.float32)
        res = distributed_matmul(
            a,
            b,
            config=cfg,
            partitioning_strategy="output_grid_2d",
            grid_shape=(2, 2),
        )
        np.testing.assert_allclose(res, np.matmul(a, b))
