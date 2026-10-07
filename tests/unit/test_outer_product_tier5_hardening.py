"""
Tier 5 Adversarial Coverage Hardening Suite for Configurable Matrix Partitioning (Phase 1).

Exhaustively covers all edge cases, boundary conditions, defensive error paths,
and mathematical invariants across:
1. `config.py`: Enums, case-insensitivity, aliases, env parsing, validation errors.
2. `partitioner.py`: 1D strip div-mod, clamping, plan manifests, dictionary access, strategy routing.
3. `assembler.py`: In-place accumulation, slice boundaries, duplicates, completion states, dtypes.
4. `profiler.py`: Upload/download redundancy ratios, dict tasks, disjoint coverage verification.
5. `precision.py`: IEEE 754 half-precision, NaN/Inf preservation, Higham bounds, BF16 masking.
6. `coordinator.py` & `client.py`: Strategy validation, strip precedence, closed client guards, dimension checks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np
import pytest
from pydantic import ValidationError

from matrix_compute.config import (
    ClientConfig,
    PartitioningStrategy,
    WorkloadType,
)
from matrix_compute.exceptions import (
    MatrixComputeError,
    MatrixDimensionError,
    TilingError,
    VerificationFailedError,
)
from matrix_compute.tiling.assembler import ResultAssembler
from matrix_compute.tiling.partitioner import (
    MatrixTiler,
    TileGridPlan,
    TileTaskPlan,
    compute_1d_strips,
)
from matrix_compute.tiling.precision import (
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
from matrix_compute.tiling.profiler import (
    BandwidthProfile,
    BandwidthProfiler,
)


# =============================================================================
# 1. CONFIGURATION & SCHEMA HARDENING (config.py)
# =============================================================================

class TestTier5ConfigHardening:
    """Adversarial coverage for config.py models and validators."""

    def test_partitioning_strategy_enum_properties(self):
        assert PartitioningStrategy.BLOCK_3D == "block_3d"
        assert PartitioningStrategy.OUTER_PRODUCT_1D == "outer_product_1d"
        assert PartitioningStrategy.OUTPUT_GRID_2D == "output_grid_2d"

    def test_workload_type_enum_properties(self):
        assert WorkloadType.GEMM_ONE_SHOT == "gemm_one_shot"
        assert WorkloadType.ITERATIVE_POWER == "iterative_power"
        assert WorkloadType.BATCH_STREAM == "batch_stream"

    @pytest.mark.parametrize(
        "raw_input,expected_enum",
        [
            ("OUTER_PRODUCT_1D", PartitioningStrategy.OUTER_PRODUCT_1D),
            ("outer_product_1d", PartitioningStrategy.OUTER_PRODUCT_1D),
            ("outer_1d", PartitioningStrategy.OUTER_PRODUCT_1D),
            ("1d_outer_product", PartitioningStrategy.OUTER_PRODUCT_1D),
            ("outer_product", PartitioningStrategy.OUTER_PRODUCT_1D),
            ("outerproduct1d", PartitioningStrategy.OUTER_PRODUCT_1D),
            ("OUTPUT_GRID_2D", PartitioningStrategy.OUTPUT_GRID_2D),
            ("output_grid", PartitioningStrategy.OUTPUT_GRID_2D),
            ("2d_output_grid", PartitioningStrategy.OUTPUT_GRID_2D),
            ("outputgrid2d", PartitioningStrategy.OUTPUT_GRID_2D),
            ("grid_2d", PartitioningStrategy.OUTPUT_GRID_2D),
            ("BLOCK_3D", PartitioningStrategy.BLOCK_3D),
            ("3d_block", PartitioningStrategy.BLOCK_3D),
            ("block3d", PartitioningStrategy.BLOCK_3D),
            ("block", PartitioningStrategy.BLOCK_3D),
        ],
    )
    def test_partitioning_strategy_aliases_and_case(self, raw_input, expected_enum):
        cfg = ClientConfig(partitioning_strategy=raw_input)
        assert cfg.partitioning_strategy == expected_enum

    @pytest.mark.parametrize(
        "raw_input,expected_enum",
        [
            ("GEMM_ONE_SHOT", WorkloadType.GEMM_ONE_SHOT),
            ("gemm_one_shot", WorkloadType.GEMM_ONE_SHOT),
            ("gemm", WorkloadType.GEMM_ONE_SHOT),
            ("one_shot", WorkloadType.GEMM_ONE_SHOT),
            ("gemm_oneshot", WorkloadType.GEMM_ONE_SHOT),
            ("ITERATIVE_POWER", WorkloadType.ITERATIVE_POWER),
            ("iterative_power", WorkloadType.ITERATIVE_POWER),
            ("iterative", WorkloadType.ITERATIVE_POWER),
            ("power", WorkloadType.ITERATIVE_POWER),
            ("power_iteration", WorkloadType.ITERATIVE_POWER),
            ("BATCH_STREAM", WorkloadType.BATCH_STREAM),
            ("batch_stream", WorkloadType.BATCH_STREAM),
            ("batch", WorkloadType.BATCH_STREAM),
            ("stream", WorkloadType.BATCH_STREAM),
            ("streaming", WorkloadType.BATCH_STREAM),
        ],
    )
    def test_workload_type_aliases_and_case(self, raw_input, expected_enum):
        cfg = ClientConfig(workload_type=raw_input)
        assert cfg.workload_type == expected_enum

    def test_partitioning_strategy_invalid_types_and_values(self):
        with pytest.raises(ValidationError):
            ClientConfig(partitioning_strategy=12345)

        with pytest.raises(ValidationError) as exc:
            ClientConfig(partitioning_strategy="invalid_unknown_strategy")
        assert "Invalid partitioning_strategy" in str(exc.value)

    def test_workload_type_invalid_types_and_values(self):
        with pytest.raises(ValidationError):
            ClientConfig(workload_type=object())

        with pytest.raises(ValidationError) as exc:
            ClientConfig(workload_type="invalid_unknown_workload")
        assert "Invalid workload_type" in str(exc.value)

    def test_numeric_bounds_validation(self):
        with pytest.raises(ValidationError):
            ClientConfig(num_strips=0)

        with pytest.raises(ValidationError):
            ClientConfig(num_strips=-1)

        with pytest.raises(ValidationError):
            ClientConfig(lease_ttl_seconds=0)

        with pytest.raises(ValidationError):
            ClientConfig(spot_check_probability=-0.0001)

        with pytest.raises(ValidationError):
            ClientConfig(spot_check_probability=1.0001)

    def test_env_population_non_dict_passthrough(self):
        result = ClientConfig.populate_from_environment("not_a_dict")
        assert result == "not_a_dict"

    def test_env_population_nostr_relays_parsing(self, monkeypatch):
        # Malformed bracketed JSON falling back to comma-split
        monkeypatch.setenv("MATRIX_NOSTR_RELAYS", "[ws://relay1:8008, ws://relay2:8008]")
        cfg = ClientConfig()
        assert cfg.nostr_relays == ["ws://relay1:8008", "ws://relay2:8008"]

        # Plain comma list without brackets
        monkeypatch.setenv("MATRIX_NOSTR_RELAYS", "ws://relayA:8008, ws://relayB:8008")
        cfg2 = ClientConfig()
        assert cfg2.nostr_relays == ["ws://relayA:8008", "ws://relayB:8008"]

    def test_boolean_string_env_parsing(self, monkeypatch):
        for truthy in ("1", "true", "yes", "on", "TRUE", "Yes"):
            monkeypatch.setenv("MATRIX_OFFLINE_MOCK_MODE", truthy)
            cfg = ClientConfig()
            assert cfg.offline_mock_mode is True

        for falsy in ("0", "false", "no", "off", "FALSE", "No"):
            monkeypatch.setenv("MATRIX_USE_DYNAMIC_FREIVALDS_BOUND", falsy)
            cfg = ClientConfig()
            assert cfg.use_dynamic_freivalds_bound is False

    def test_settlement_url_validation_rejections(self):
        for invalid_url in ("ftp://node.local:8080", "ws://node.local:8080", "not_a_url", "http://"):
            with pytest.raises(ValidationError):
                ClientConfig(settlement_api_url=invalid_url)


# =============================================================================
# 2. PARTITIONER & GEOMETRY HARDENING (partitioner.py)
# =============================================================================

class TestTier5PartitionerHardening:
    """Adversarial coverage for compute_1d_strips, MatrixTiler, and TileGridPlan."""

    def test_compute_1d_strips_invalid_k(self):
        with pytest.raises(ValueError) as exc:
            compute_1d_strips(k=0, p=4)
        assert "Inner dimension k must be >= 1" in str(exc.value)

        with pytest.raises(ValueError):
            compute_1d_strips(k=-5, p=4)

    def test_compute_1d_strips_clamping_extremes(self):
        # p <= 0 clamps to 1
        assert compute_1d_strips(k=10, p=0) == [(0, 10)]
        assert compute_1d_strips(k=10, p=-5) == [(0, 10)]

        # p > k clamps to k (strips of width 1)
        assert compute_1d_strips(k=3, p=10) == [(0, 1), (1, 2), (2, 3)]

        # Scalar matrix (k=1)
        assert compute_1d_strips(k=1, p=100) == [(0, 1)]

    def test_compute_1d_strips_quotient_remainder_distribution(self):
        # 17 partitioned into 5 strips -> 17 % 5 = 2. First 2 have length 4, last 3 have length 3.
        strips = compute_1d_strips(k=17, p=5)
        lengths = [e - s for s, e in strips]
        assert lengths == [4, 4, 3, 3, 3]
        assert sum(lengths) == 17
        assert strips[0] == (0, 4)
        assert strips[-1] == (14, 17)

    def test_matrix_tiler_init_validation(self):
        with pytest.raises(ValueError):
            MatrixTiler(default_tile_size=0)
        with pytest.raises(ValueError):
            MatrixTiler(default_tile_size=-10)

    def test_create_plan_outer_product_1d_type_and_dimension_validation(self):
        valid = np.ones((4, 4), dtype=np.float32)

        # Non-ndarray
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_outer_product_1d([[1, 2], [3, 4]], valid, num_strips=2)
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_outer_product_1d(valid, "invalid", num_strips=2)

        # Non-2D
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_outer_product_1d(np.ones((4,)), valid, num_strips=2)
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_outer_product_1d(valid, np.ones((4, 4, 4)), num_strips=2)

        # Dimension mismatch
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_outer_product_1d(np.ones((4, 5)), np.ones((6, 4)), num_strips=2)

        # Zero dimensions
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_outer_product_1d(np.ones((0, 4)), np.ones((4, 4)), num_strips=2)
        with pytest.raises(MatrixDimensionError):
            MatrixTiler.create_plan_outer_product_1d(np.ones((4, 4)), np.ones((4, 0)), num_strips=2)

    def test_create_plan_outer_product_1d_custom_params(self):
        a = np.ones((8, 16), dtype=np.float32)
        b = np.ones((16, 8), dtype=np.float32)

        plan = MatrixTiler.create_plan_outer_product_1d(
            a=a,
            b=b,
            num_strips=4,
            dtype="fp16",
            job_id="custom_job",
            tile_size=64,
        )
        assert plan.tile_size == 64
        assert plan.dtype == "fp16"
        assert len(plan.tasks) == 4
        assert plan.tasks[0].task_id.startswith("custom_job_strip_")

    def test_partition_outer_product_1d_returns_task_list(self):
        a = np.ones((4, 8), dtype=np.float32)
        b = np.ones((8, 4), dtype=np.float32)
        tasks = MatrixTiler.partition_outer_product_1d(a, b, num_strips=2)
        assert isinstance(tasks, list)
        assert len(tasks) == 2
        assert all(isinstance(t, TileTaskPlan) for t in tasks)

    def test_create_plan_strategy_routing(self):
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)

        # Fully implemented 2D grid strategy routes cleanly
        for grid_strat in ("output_grid_2d", "output_grid", "2d_output_grid", "grid_2d"):
            plan_2d = MatrixTiler.create_plan(a, b, strategy=grid_strat)
            assert plan_2d.strategy == "output_grid_2d"
            assert plan_2d.num_k_blocks == 1

        # Unknown strategy
        with pytest.raises(ValueError) as exc:
            MatrixTiler.create_plan(a, b, strategy="unknown_random_strategy")
        assert "Unknown partitioning strategy" in str(exc.value)

        # Outer product with num_strips fallback heuristic
        plan = MatrixTiler.create_plan(a, b, strategy="outer_product_1d", num_strips=None, tile_size=4)
        assert plan.num_k_blocks == 4  # ceil(16 / 4) == 4 strips

    def test_tile_grid_plan_and_task_plan_dict_interface(self):
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=2)

        # Plan dictionary access across all primary keys
        assert plan["m"] == 8
        assert plan["k"] == 8
        assert plan["n"] == 8
        assert plan["tile_size"] == 4
        assert plan["dtype"] == "fp32"
        assert plan["num_m_blocks"] == 1
        assert plan["num_k_blocks"] == 2
        assert plan["num_n_blocks"] == 1
        assert plan["total_tasks"] == 2
        assert plan["total_gflops"] == plan.total_gflops
        assert len(plan["tasks"]) == 2

        # Plan dictionary aliases and strategy metadata (SCOPE.md contract)
        assert plan["p"] == 2
        assert plan["strategy"] == "outer_product_1d"

        # Plan dictionary helper methods
        assert plan.get("m") == 8
        assert plan.get("p") == 2
        assert plan.get("strategy") == "outer_product_1d"
        assert plan.get("nonexistent_key", "default_val") == "default_val"
        assert plan.get("nonexistent_key") is None
        assert "m" in plan
        assert "p" in plan
        assert "strategy" in plan
        assert "tasks" in plan
        assert "nonexistent_key" not in plan

        # Plan dictionary collection views
        assert "m" in plan.keys()
        assert "p" in plan.keys()
        assert "strategy" in plan.keys()
        assert len(plan.values()) == len(plan.keys())
        assert len(plan.items()) == len(plan.keys())
        assert dict(plan.items())["m"] == 8
        assert dict(plan.items())["p"] == 2
        assert dict(plan.items())["strategy"] == "outer_product_1d"

        with pytest.raises(KeyError, match="TileGridPlan has no field 'nonexistent_key'"):
            _ = plan["nonexistent_key"]

        # Plan task coordinate lookup
        task_found = plan.get_task(row_idx=0, col_idx=0, k_idx=1)
        assert task_found is not None
        assert task_found.k_idx == 1

        task_missing = plan.get_task(row_idx=1, col_idx=0, k_idx=0)
        assert task_missing is None

        # Task dictionary access and methods
        task = plan.tasks[0]
        assert task["task_id"] == task.task_id
        assert task["row_idx"] == 0
        assert task["col_idx"] == 0
        assert task["k_idx"] == 0
        assert task["row_slice"] == (0, 8)
        assert task["col_slice"] == (0, 8)
        assert task["k_slice"] == (0, 4)
        assert np.array_equal(task["a_tile"], task.a_tile)
        assert np.array_equal(task["b_tile"], task.b_tile)
        assert task.get("k_idx") == 0
        assert task.get("nonexistent_task_key", 999) == 999
        assert "task_id" in task
        assert "a_tile" in task
        assert "nonexistent_task_key" not in task
        assert "task_id" in task.keys()
        assert len(task.values()) == len(task.keys())
        assert dict(task.items())["task_id"] == task.task_id

        with pytest.raises(KeyError, match="TileTaskPlan has no field 'nonexistent_task_key'"):
            _ = task["nonexistent_task_key"]


# =============================================================================
# 3. ASSEMBLER & BUFFER HARDENING (assembler.py)
# =============================================================================

class TestTier5AssemblerHardening:
    """Adversarial coverage for ResultAssembler buffer lifecycle and error paths."""

    def test_assembler_init_dimension_validation(self):
        with pytest.raises(ValueError):
            ResultAssembler(m=0, n=10)
        with pytest.raises(ValueError):
            ResultAssembler(m=10, n=0)
        with pytest.raises(ValueError):
            ResultAssembler(m=-4, n=-4)

    def test_assembler_expected_tasks_variants(self):
        # 1. Expected count as int
        asm1 = ResultAssembler(m=4, n=4, expected_tasks=3)
        assert asm1._expected_count == 3

        # 2. Expected tasks as list of dicts
        asm2 = ResultAssembler(m=4, n=4, expected_tasks=[{"task_id": "t1"}, {"task_id": "t2"}])
        assert asm2._expected_count == 2
        assert asm2._expected_task_ids == {"t1", "t2"}

        # 3. Expected tasks as set of strings
        asm3 = ResultAssembler(m=4, n=4, expected_tasks={"s1", "s2", "s3"})
        assert asm3._expected_count == 3
        assert asm3._expected_task_ids == {"s1", "s2", "s3"}

        # 4. Expected tasks as list of objects with task_id
        task_obj = TileTaskPlan(
            task_id="obj_t1",
            row_idx=0, col_idx=0, k_idx=0,
            row_slice=(0, 4), col_slice=(0, 4), k_slice=(0, 2),
            a_tile=np.ones((4, 2)), b_tile=np.ones((2, 4)),
            dtype="fp32", estimated_gflops=0.1,
        )
        asm4 = ResultAssembler(m=4, n=4, expected_tasks=[task_obj])
        assert asm4._expected_count == 1
        assert asm4._expected_task_ids == {"obj_t1"}

    def test_accumulate_slice_boundary_violations(self):
        asm = ResultAssembler(m=10, n=10)
        tile = np.ones((5, 5), dtype=np.float32)

        # Missing slices
        with pytest.raises(ValueError):
            asm.accumulate(tile, row_idx=0, col_idx=0, row_slice=None, col_slice=(0, 5))

        with pytest.raises(ValueError):
            asm.accumulate(tile, row_idx=0, col_idx=0, row_slice=(0, 5), col_slice=None)

        # Out of bounds: negative indices
        with pytest.raises(ValueError):
            asm.accumulate(tile, row_idx=0, col_idx=0, row_slice=(-1, 4), col_slice=(0, 5))

        # Out of bounds: exceeding buffer dimensions
        with pytest.raises(ValueError):
            asm.accumulate(tile, row_idx=0, col_idx=0, row_slice=(0, 11), col_slice=(0, 5))

        # Out of bounds: inverted slice
        with pytest.raises(ValueError):
            asm.accumulate(tile, row_idx=0, col_idx=0, row_slice=(5, 3), col_slice=(0, 5))

    def test_accumulate_tile_shape_mismatch(self):
        asm = ResultAssembler(m=10, n=10)
        wrong_tile = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(TilingError) as exc:
            asm.accumulate(wrong_tile, row_idx=0, col_idx=0, row_slice=(0, 5), col_slice=(0, 5))
        assert "does not match expected slice" in str(exc.value)

    def test_duplicate_rejection_vs_allow_duplicate(self):
        tile = np.ones((4, 4), dtype=np.float32)

        # allow_duplicate=False (default): raises TilingError
        asm_strict = ResultAssembler(m=4, n=4, allow_duplicate=False)
        asm_strict.accumulate(tile, row_idx=0, col_idx=0, k_idx=0, row_slice=(0, 4), col_slice=(0, 4))
        with pytest.raises(TilingError) as exc:
            asm_strict.accumulate(tile, row_idx=0, col_idx=0, k_idx=0, row_slice=(0, 4), col_slice=(0, 4))
        assert "Duplicate accumulation rejected" in str(exc.value)

        # allow_duplicate=True: adds in-place
        asm_permissive = ResultAssembler(m=4, n=4, allow_duplicate=True)
        asm_permissive.accumulate(tile, row_idx=0, col_idx=0, k_idx=0, row_slice=(0, 4), col_slice=(0, 4))
        asm_permissive.accumulate(tile, row_idx=0, col_idx=0, k_idx=0, row_slice=(0, 4), col_slice=(0, 4))
        result = asm_permissive.get_result()
        np.testing.assert_allclose(result, np.full((4, 4), 2.0, dtype=np.float32))

    def test_completion_progress_and_enforcement(self):
        asm = ResultAssembler(m=4, n=4, expected_tasks={"t1", "t2"})
        assert asm.get_progress() == 0.0
        assert asm.is_complete() is False
        assert asm.missing_tasks() == ["t1", "t2"]

        # Incomplete result extraction raises TilingError when enforce_complete=True
        with pytest.raises(TilingError) as exc:
            asm.get_result(enforce_complete=True)
        assert "assembly incomplete" in str(exc.value)

        # Incomplete result extraction succeeds when enforce_complete=False
        res_partial = asm.get_result(enforce_complete=False)
        assert res_partial.shape == (4, 4)

        # Accumulate first task
        tile = np.ones((4, 4), dtype=np.float32)
        asm.accumulate(tile, row_idx=0, col_idx=0, k_idx=0, row_slice=(0, 4), col_slice=(0, 4), task_id="t1")
        assert asm.get_progress() == 0.5
        assert asm.missing_tasks() == ["t2"]
        assert asm.is_complete() is False

        # Accumulate second task
        asm.accumulate(tile, row_idx=0, col_idx=0, k_idx=1, row_slice=(0, 4), col_slice=(0, 4), task_id="t2")
        assert asm.get_progress() == 1.0
        assert asm.missing_tasks() == []
        assert asm.is_complete() is True
        res_complete = asm.get_result(enforce_complete=True)
        np.testing.assert_allclose(res_complete, np.full((4, 4), 2.0, dtype=np.float32))

    def test_assembler_reset_lifecycle(self):
        asm = ResultAssembler(m=4, n=4, expected_tasks=1)
        tile = np.ones((4, 4), dtype=np.float32)
        asm.accumulate(tile, row_idx=0, col_idx=0, k_idx=0, row_slice=(0, 4), col_slice=(0, 4), task_id="t1")
        assert asm.is_complete() is True

        asm.reset()
        assert asm.is_complete() is False
        assert asm.get_progress() == 0.0
        assert np.all(asm._buffer == 0)

    def test_assemble_tiles_classmethod_helper(self):
        task1 = {
            "task_id": "strip_0",
            "row_idx": 0, "col_idx": 0, "k_idx": 0,
            "row_slice": (0, 2), "col_slice": (0, 2),
            "a_tile": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "b_tile": np.array([[5.0, 6.0], [7.0, 8.0]]),
        }
        res = ResultAssembler.assemble_tiles(m=2, n=2, tile_tasks=[task1], dtype="fp32")
        expected = np.matmul(task1["a_tile"], task1["b_tile"])
        np.testing.assert_allclose(res, expected)


# =============================================================================
# 4. BANDWIDTH PROFILER HARDENING (profiler.py)
# =============================================================================

class TestTier5ProfilerHardening:
    """Adversarial coverage for BandwidthProfile and BandwidthProfiler."""

    def test_bandwidth_profile_dictionary_interface(self):
        prof = BandwidthProfile(
            m=10, k=20, n=30, num_tasks=4, strategy="outer_product_1d", dtype="fp32",
            total_a_elements=200, total_b_elements=600, total_uploaded_elements=800,
            optimal_elements=800, redundancy_elements=0, redundancy_ratio=0.0,
            bytes_per_element=4, total_uploaded_bytes=3200, optimal_bytes=3200,
            redundancy_bytes=0, is_upload_optimal=True,
        )

        assert prof["m"] == 10
        assert prof["total_uploaded_elements"] == 800
        assert prof["is_upload_optimal"] is True
        assert prof.get("strategy") == "outer_product_1d"
        assert prof.get("nonexistent", "fallback") == "fallback"
        assert prof.num_strips == 4

        with pytest.raises(KeyError):
            _ = prof["invalid_profile_key"]

        keys = prof.keys()
        assert "optimal_elements" in keys
        assert "num_strips" in keys
        d = prof.to_dict()
        assert isinstance(d, dict)
        assert d["m"] == 10

    def test_bandwidth_profiler_with_dict_task_formats(self):
        # Format 1: dict with a_tile, b_tile ndarrays
        tasks_fmt1 = [
            {
                "a_tile": np.ones((8, 4)),
                "b_tile": np.ones((4, 8)),
                "row_slice": (0, 8),
                "col_slice": (0, 8),
            }
        ]
        p1 = BandwidthProfiler.profile_tasks(tasks_fmt1, m=8, k=4, n=8)
        assert p1.total_uploaded_elements == 32 + 32
        assert p1.is_upload_optimal is True

        # Format 2: dict with m_i, k_j, n_k dimensions
        tasks_fmt2 = [
            {"m_i": 8, "k_j": 4, "n_k": 8, "row_slice": (0, 8), "col_slice": (0, 8)}
        ]
        p2 = BandwidthProfiler.profile_tasks(tasks_fmt2, m=8, k=4, n=8)
        assert p2.total_uploaded_elements == 64
        assert p2.is_upload_optimal is True

        # Format 3: dict with shape_a, shape_b
        tasks_fmt3 = [
            {"shape_a": (8, 4), "shape_b": (4, 8), "m_i": 8, "n_k": 8}
        ]
        p3 = BandwidthProfiler.profile_tasks(tasks_fmt3, m=8, k=4, n=8)
        assert p3.total_uploaded_elements == 64

    def test_verify_disjoint_coverage_adversarial_paths(self):
        # Empty task list
        assert BandwidthProfiler.verify_disjoint_coverage([], k=10) is False

        # Missing slice fields
        assert BandwidthProfiler.verify_disjoint_coverage([{"invalid": 1}], k=10) is False

        # Gap between strips: [0, 4) and [5, 10) (missing 4)
        tasks_gap = [
            {"k_slice": (0, 4)},
            {"k_slice": (5, 10)},
        ]
        assert BandwidthProfiler.verify_disjoint_coverage(tasks_gap, k=10) is False

        # Overlap between strips: [0, 6) and [4, 10) (overlapping 4..6)
        tasks_overlap = [
            {"k_slice": (0, 6)},
            {"k_slice": (4, 10)},
        ]
        assert BandwidthProfiler.verify_disjoint_coverage(tasks_overlap, k=10) is False

        # Zero width strip: [0, 5) and [5, 5) and [5, 10)
        tasks_zero_width = [
            {"k_slice": (0, 5)},
            {"k_slice": (5, 5)},
            {"k_slice": (5, 10)},
        ]
        assert BandwidthProfiler.verify_disjoint_coverage(tasks_zero_width, k=10) is False

        # Incomplete termination: [0, 5) and [5, 9) when k=10
        tasks_incomplete = [
            {"k_slice": (0, 5)},
            {"k_slice": (5, 9)},
        ]
        assert BandwidthProfiler.verify_disjoint_coverage(tasks_incomplete, k=10) is False

        # Perfectly valid disjoint coverage
        tasks_valid = [
            {"k_slice": (5, 10)},
            {"k_slice": (0, 5)},
        ]
        assert BandwidthProfiler.verify_disjoint_coverage(tasks_valid, k=10) is True


# =============================================================================
# 5. PRECISION & NUMERICAL HARDENING (precision.py)
# =============================================================================

class TestTier5PrecisionHardening:
    """Adversarial coverage for precision normalization, IEEE bfloat16, and Higham bounds."""

    @pytest.mark.parametrize(
        "dtype_input,expected_norm",
        [
            ("fp32", "fp32"),
            ("float32", "fp32"),
            ("single", "fp32"),
            ("f32", "fp32"),
            (np.float32, "fp32"),
            ("fp16", "fp16"),
            ("float16", "fp16"),
            ("half", "fp16"),
            ("f16", "fp16"),
            (np.float16, "fp16"),
            ("bf16", "bf16"),
            ("bfloat16", "bf16"),
            ("bf", "bf16"),
            ("fp64", "fp64"),
            ("float64", "fp64"),
            ("double", "fp64"),
            ("f64", "fp64"),
            (np.float64, "fp64"),
        ],
    )
    def test_normalize_dtype_aliases(self, dtype_input, expected_norm):
        assert normalize_dtype(dtype_input) == expected_norm

    def test_normalize_dtype_unsupported(self):
        for unsupported in ("int32", "complex64", "uint8", "invalid"):
            with pytest.raises(ValueError):
                normalize_dtype(unsupported)

    def test_get_numpy_dtype_and_bytes(self):
        assert get_numpy_dtype("fp16") == np.float16
        assert get_numpy_dtype("fp32") == np.float32
        assert get_numpy_dtype("bf16") == np.float32  # stored in float32 container
        assert get_numpy_dtype("fp64") == np.float64

        assert get_bytes_per_element("fp16") == 2
        assert get_bytes_per_element("fp32") == 4
        assert get_bytes_per_element("bf16") == 4
        assert get_bytes_per_element("fp64") == 8

    def test_truncate_to_bf16_ieee_special_values(self):
        # 1. NaN preservation with quiet bit restoration
        nan_val = np.array([np.nan], dtype=np.float32)
        bf16_nan = truncate_to_bf16(nan_val)
        assert np.isnan(bf16_nan[0])
        assert validate_bf16_masking(bf16_nan)

        # 2. +/- Infinity preservation
        inf_vals = np.array([np.inf, -np.inf], dtype=np.float32)
        bf16_inf = truncate_to_bf16(inf_vals)
        assert np.isinf(bf16_inf[0]) and bf16_inf[0] > 0
        assert np.isinf(bf16_inf[1]) and bf16_inf[1] < 0
        assert validate_bf16_masking(bf16_inf)

        # 3. Signed zeros (+0.0 and -0.0)
        zeros = np.array([0.0, -0.0], dtype=np.float32)
        bf16_zeros = truncate_to_bf16(zeros)
        assert bf16_zeros[0] == 0.0
        assert bf16_zeros[1] == 0.0
        assert validate_bf16_masking(bf16_zeros)

    def test_validate_bf16_masking_rejection(self):
        # Non-float32 array returns False
        f16_arr = np.array([1.0, 2.0], dtype=np.float16)
        assert validate_bf16_masking(f16_arr) is False

        # Float32 array with unmasked lower 16 bits returns False
        u32_raw = np.array([0x3F800001], dtype=np.uint32)
        f32_unmasked = u32_raw.view(np.float32)
        assert validate_bf16_masking(f32_unmasked) is False

        # Float32 array with masked lower 16 bits returns True
        u32_masked = np.array([0x3F800000], dtype=np.uint32)
        f32_masked = u32_masked.view(np.float32)
        assert validate_bf16_masking(f32_masked) is True

    def test_calculate_higham_bound_branches(self):
        # Standard condition: K * u < 0.999
        b1 = calculate_higham_bound(k=100, norm_a=1.0, norm_b=1.0, dtype="fp32")
        assert b1 > 0.0

        # Extreme contraction condition: K * u >= 0.999 (e.g. k=1000 with bf16 where u = 2^-8 ~ 0.0039)
        # 1000 * 0.00390625 = 3.90625 >= 0.999
        b_extreme = calculate_higham_bound(k=1000, norm_a=1.0, norm_b=1.0, dtype="bf16")
        assert b_extreme > 0.0
        assert math.isclose(b_extreme, (1000.0 * 2.0**-8) * 1000.0 * 1.0 * 1.0)

    def test_emulate_gemm_across_dtypes(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        c_fp32 = emulate_gemm(a, b, dtype="fp32")
        assert c_fp32.dtype == np.float32

        c_fp16 = emulate_gemm(a, b, dtype="fp16")
        assert c_fp16.dtype == np.float16

        c_bf16 = emulate_gemm(a, b, dtype="bf16")
        assert c_bf16.dtype == np.float32
        assert validate_bf16_masking(c_bf16)

        c_fp64 = emulate_gemm(a, b, dtype="fp64")
        assert c_fp64.dtype == np.float64


# =============================================================================
# 6. COORDINATOR & CLIENT HARDENING (coordinator.py, client.py)
# =============================================================================

class TestTier5CoordinatorClientHardening:
    """Adversarial coverage for client lifecycle, coordinator validation, and routing."""

    def test_client_validation_closed_state(self):
        from matrix_compute.client import MatrixComputeClient
        client = MatrixComputeClient(offline_mock_mode=True)
        client.close()

        # Calling close multiple times is safe and idempotent
        client.close()

        # Calling matmul on closed client raises MatrixComputeError
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(MatrixComputeError) as exc:
            client.matmul(a, b)
        assert "MatrixComputeClient is closed" in str(exc.value)

    def test_client_matmul_defensive_dimension_checks(self):
        from matrix_compute.client import MatrixComputeClient
        client = MatrixComputeClient(offline_mock_mode=True)
        try:
            # 1. Non-ndarray inputs
            with pytest.raises(MatrixDimensionError):
                client.matmul([[1, 2], [3, 4]], np.ones((2, 2)))

            # 2. 1D array input
            with pytest.raises(MatrixDimensionError):
                client.matmul(np.ones((4,)), np.ones((4, 4)))

            # 3. 3D array input
            with pytest.raises(MatrixDimensionError):
                client.matmul(np.ones((4, 4, 4)), np.ones((4, 4)))

            # 4. Zero dimension input
            with pytest.raises(MatrixDimensionError):
                client.matmul(np.ones((0, 4)), np.ones((4, 4)))

            # 5. Inner dimension mismatch
            with pytest.raises(MatrixDimensionError):
                client.matmul(np.ones((4, 5)), np.ones((6, 4)))
        finally:
            client.close()
