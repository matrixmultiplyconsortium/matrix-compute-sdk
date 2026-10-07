"""
Thread-safe result matrix accumulator and assembler for matrix-compute-sdk.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from matrix_compute.exceptions import TilingError
from matrix_compute.tiling.partitioner import TileGridPlan, TileTaskPlan
from matrix_compute.tiling.precision import cast_matrix, normalize_dtype, truncate_to_bf16

logger = logging.getLogger(__name__)


class ResultAssembler:
    """
    Thread-safe accumulator that collects sub-matrix dot products C_{i,k} = sum_j A_{i,j} B_{j,k}
    into a full M x N output matrix.
    """

    def __init__(
        self,
        m: int,
        n: int,
        k_blocks: Optional[int] = None,
        expected_tasks: Optional[Union[int, List[TileTaskPlan], Set[str], List[Any]]] = None,
        dtype: str = "fp32",
        allow_duplicate: bool = False,
        zero_reduction: Optional[bool] = None,
    ) -> None:
        if m < 1 or n < 1:
            raise ValueError(f"Matrix dimensions must be >= 1. Got m={m}, n={n}")

        self._m = m
        self._n = n
        self._k_blocks = k_blocks
        self._dtype = normalize_dtype(dtype)
        self._allow_duplicate = allow_duplicate

        if zero_reduction is not None:
            self._zero_reduction = bool(zero_reduction)
        elif k_blocks is not None and k_blocks == 1:
            self._zero_reduction = True
        else:
            self._zero_reduction = False

        self._reduction_ops: int = 0

        # Internal accumulation buffer (stored in float32 for fp32 and bf16, float16 for fp16, float64 for fp64)
        if self._dtype == "fp16":
            self._buf_dtype = np.float16
        elif self._dtype == "fp64":
            self._buf_dtype = np.float64
        else:
            self._buf_dtype = np.float32

        self._buffer = np.zeros((m, n), dtype=self._buf_dtype)

        # Thread synchronization
        self._lock = threading.RLock()

        # Task completion tracking
        self._received_tasks: Set[str] = set()
        self._received_coords: Set[Tuple[int, int, int]] = set()

        # Spatial slice disjointness tracking: (r_s, r_e, c_s, c_e, k_idx, coord)
        self._placed_slices: List[Tuple[int, int, int, int, int, Tuple[int, int, int]]] = []
        self._covered_elements: int = 0

        # Expected tasks tracking
        self._expected_task_ids: Optional[Set[str]] = None
        self._expected_count: Optional[int] = None

        if isinstance(expected_tasks, int):
            self._expected_count = expected_tasks
        elif isinstance(expected_tasks, (list, tuple)):
            self._expected_task_ids = set()
            for t in expected_tasks:
                if isinstance(t, dict):
                    self._expected_task_ids.add(t.get("task_id", ""))
                elif hasattr(t, "task_id"):
                    self._expected_task_ids.add(getattr(t, "task_id"))
                else:
                    self._expected_task_ids.add(str(t))
            self._expected_count = len(self._expected_task_ids)
        elif isinstance(expected_tasks, set):
            self._expected_task_ids = set(str(t) for t in expected_tasks)
            self._expected_count = len(self._expected_task_ids)

    @property
    def reduction_ops(self) -> int:
        """Count of local floating-point reduction addition operations executed during assembly."""
        return self._reduction_ops

    @property
    def zero_reduction(self) -> bool:
        """Whether zero-reduction direct coordinate placement is enabled."""
        return self._zero_reduction

    @classmethod
    def from_plan(
        cls,
        plan: TileGridPlan,
        allow_duplicate: bool = False,
        zero_reduction: Optional[bool] = None,
    ) -> ResultAssembler:
        """Creates a ResultAssembler pre-configured with the full TileGridPlan manifest."""
        eff_zero_reduction = (
            zero_reduction
            if zero_reduction is not None
            else (plan.strategy == "output_grid_2d" or plan.num_k_blocks == 1)
        )
        return cls(
            m=plan.m,
            n=plan.n,
            k_blocks=plan.num_k_blocks,
            expected_tasks=plan.tasks,
            dtype=plan.dtype,
            allow_duplicate=allow_duplicate,
            zero_reduction=eff_zero_reduction,
        )

    def accumulate(
        self,
        tile_result: np.ndarray,
        row_idx: int,
        col_idx: int,
        k_idx: int = 0,
        row_slice: Optional[Tuple[int, int]] = None,
        col_slice: Optional[Tuple[int, int]] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """
        Accumulates a computed tile dot product into the target matrix block.

        Raises:
            ValueError: If slices are out of bounds.
            TilingError: If shapes mismatch or duplicate task is received.
        """
        if row_slice is None or col_slice is None:
            raise ValueError("row_slice and col_slice must be explicitly provided")

        r_s, r_e = row_slice
        c_s, c_e = col_slice

        # Bounds checking
        if r_s < 0 or c_s < 0 or r_e > self._m or c_e > self._n or r_e <= r_s or c_e <= c_s:
            raise ValueError(
                f"Tile slice out of bounds: rows {r_s}:{r_e} vs {self._m}, cols {c_s}:{c_e} vs {self._n}"
            )

        expected_shape = (r_e - r_s, c_e - c_s)
        if tile_result.shape != expected_shape:
            raise TilingError(
                f"Tile result shape {tile_result.shape} does not match expected slice {expected_shape}"
            )

        coord = (row_idx, col_idx, k_idx)
        eff_task_id = task_id or f"tile_{row_idx}_{col_idx}_{k_idx}"

        with self._lock:
            if coord in self._received_coords and not self._allow_duplicate:
                raise TilingError(
                    f"Duplicate accumulation rejected for coordinate (r={row_idx}, c={col_idx}, k={k_idx})"
                )

            # Spatial slice disjointness validation:
            # Slices must be mutually disjoint in 2D spatial coordinate space to prevent rogue
            # overlapping slices from silently overwriting or corrupting already placed regions.
            # In zero-reduction mode, all slices must be mutually disjoint across the entire grid.
            # In reduction mode, slices sharing the same k_idx must be mutually disjoint.
            is_exact_rewrite = False
            for p_rs, p_re, p_cs, p_ce, p_k, p_coord in self._placed_slices:
                if self._zero_reduction or p_k == k_idx:
                    # Check 2D bounding rectangle intersection: [r_s, r_e) x [c_s, c_e)
                    has_overlap = max(r_s, p_rs) < min(r_e, p_re) and max(c_s, p_cs) < min(c_e, p_ce)
                    if has_overlap:
                        if self._allow_duplicate and coord == p_coord and (r_s, r_e, c_s, c_e) == (p_rs, p_re, p_cs, p_ce):
                            is_exact_rewrite = True
                        else:
                            raise TilingError(
                                f"Overlapping spatial slice detected: rows {r_s}:{r_e}, cols {c_s}:{c_e} "
                                f"overlaps with previously placed slice rows {p_rs}:{p_re}, cols {p_cs}:{p_ce} (k={p_k})"
                            )

            # Perform direct placement if zero_reduction, otherwise in-place accumulation
            if self._zero_reduction:
                self._buffer[r_s:r_e, c_s:c_e] = tile_result.astype(self._buf_dtype)
            else:
                self._buffer[r_s:r_e, c_s:c_e] += tile_result.astype(self._buf_dtype)
                self._reduction_ops += tile_result.size

            if not is_exact_rewrite:
                self._placed_slices.append((r_s, r_e, c_s, c_e, k_idx, coord))
                if self._zero_reduction:
                    self._covered_elements += (r_e - r_s) * (c_e - c_s)

            self._received_coords.add(coord)
            self._received_tasks.add(eff_task_id)

    def accumulate_task(self, task: Union[TileTaskPlan, Dict[str, Any]], tile_result: np.ndarray) -> None:
        """Accumulates result using metadata from a TileTaskPlan or dictionary."""
        r_idx = task["row_idx"] if isinstance(task, dict) else task.row_idx
        c_idx = task["col_idx"] if isinstance(task, dict) else task.col_idx
        k_idx = task["k_idx"] if isinstance(task, dict) else task.k_idx
        r_slice = task["row_slice"] if isinstance(task, dict) else task.row_slice
        c_slice = task["col_slice"] if isinstance(task, dict) else task.col_slice
        t_id = task["task_id"] if isinstance(task, dict) else task.task_id

        self.accumulate(
            tile_result=tile_result,
            row_idx=r_idx,
            col_idx=c_idx,
            k_idx=k_idx,
            row_slice=r_slice,
            col_slice=c_slice,
            task_id=t_id,
        )

    def is_complete(self) -> bool:
        """Returns True if all expected tile tasks have been accumulated."""
        with self._lock:
            if self._expected_task_ids is not None:
                return (
                    len(self._expected_task_ids) > 0
                    and self._expected_task_ids.issubset(self._received_tasks)
                )
            if self._expected_count is not None:
                return self._expected_count > 0 and len(self._received_tasks) >= self._expected_count
            if self._zero_reduction:
                return self._covered_elements >= (self._m * self._n)
            return False

    def get_progress(self) -> float:
        """Returns progress fraction from 0.0 to 1.0."""
        with self._lock:
            if self._expected_count and self._expected_count > 0:
                return min(1.0, len(self._received_tasks) / float(self._expected_count))
            if self._zero_reduction:
                total_elem = self._m * self._n
                return min(1.0, float(self._covered_elements) / float(total_elem))
            return 0.0 if not self._received_tasks else 1.0

    def missing_tasks(self) -> List[str]:
        """Returns list of expected task IDs that have not yet been accumulated."""
        with self._lock:
            if self._expected_task_ids is None:
                return []
            return sorted(list(self._expected_task_ids - self._received_tasks))

    def get_result(self, enforce_complete: bool = False) -> np.ndarray:
        """
        Extracts the final assembled matrix in the target precision format.

        Raises:
            TilingError: If enforce_complete is True and the assembly is incomplete.
        """
        with self._lock:
            if enforce_complete and not self.is_complete():
                missing = self.missing_tasks()
                raise TilingError(
                    f"Cannot extract result: assembly incomplete (missing {len(missing)} tasks)"
                )

            res = self._buffer.copy()

        # Apply precision-specific formatting
        if self._dtype == "bf16":
            return truncate_to_bf16(res)
        elif self._dtype == "fp16":
            return res.astype(np.float16)
        elif self._dtype == "fp64":
            return res.astype(np.float64)
        return res.astype(np.float32)

    def reset(self) -> None:
        """Clears the accumulator buffer and tracking sets."""
        with self._lock:
            self._buffer.fill(0)
            self._received_tasks.clear()
            self._received_coords.clear()
            self._placed_slices.clear()
            self._covered_elements = 0
            self._reduction_ops = 0

    @classmethod
    def assemble_tiles(
        cls,
        m: int,
        n: int,
        tile_tasks: List[Union[TileTaskPlan, Dict[str, Any]]],
        dtype: str = "fp32",
        zero_reduction: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Drop-in reference compatible accumulator that iterates over tasks and sums their products.
        """
        assembler = cls(
            m=m,
            n=n,
            expected_tasks=len(tile_tasks),
            dtype=dtype,
            zero_reduction=zero_reduction,
        )
        for task in tile_tasks:
            r_s, r_e = task["row_slice"]
            c_s, c_e = task["col_slice"]
            a_tile = task["a_tile"]
            b_tile = task["b_tile"]
            tile_res = cast_matrix(np.matmul(a_tile.astype(np.float64), b_tile.astype(np.float64)), dtype)
            assembler.accumulate(
                tile_result=tile_res,
                row_idx=task["row_idx"],
                col_idx=task["col_idx"],
                k_idx=task["k_idx"],
                row_slice=(r_s, r_e),
                col_slice=(c_s, c_e),
                task_id=task["task_id"],
            )
        return assembler.get_result(enforce_complete=False)
