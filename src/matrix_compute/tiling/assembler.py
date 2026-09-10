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
    ) -> None:
        if m < 1 or n < 1:
            raise ValueError(f"Matrix dimensions must be >= 1. Got m={m}, n={n}")

        self._m = m
        self._n = n
        self._k_blocks = k_blocks
        self._dtype = normalize_dtype(dtype)
        self._allow_duplicate = allow_duplicate

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

    @classmethod
    def from_plan(cls, plan: TileGridPlan, allow_duplicate: bool = False) -> ResultAssembler:
        """Creates a ResultAssembler pre-configured with the full TileGridPlan manifest."""
        return cls(
            m=plan.m,
            n=plan.n,
            k_blocks=plan.num_k_blocks,
            expected_tasks=plan.tasks,
            dtype=plan.dtype,
            allow_duplicate=allow_duplicate,
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

            # Perform in-place accumulation
            self._buffer[r_s:r_e, c_s:c_e] += tile_result.astype(self._buf_dtype)
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
                return self._expected_task_ids.issubset(self._received_tasks)
            if self._expected_count is not None:
                return len(self._received_tasks) >= self._expected_count
            return len(self._received_tasks) > 0

    def get_progress(self) -> float:
        """Returns progress fraction from 0.0 to 1.0."""
        with self._lock:
            if not self._expected_count:
                return 0.0 if not self._received_tasks else 1.0
            return min(1.0, len(self._received_tasks) / float(self._expected_count))

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

    @classmethod
    def assemble_tiles(
        cls,
        m: int,
        n: int,
        tile_tasks: List[Union[TileTaskPlan, Dict[str, Any]]],
        dtype: str = "fp32",
    ) -> np.ndarray:
        """
        Drop-in reference compatible accumulator that iterates over tasks and sums their products.
        """
        assembler = cls(m=m, n=n, expected_tasks=len(tile_tasks), dtype=dtype)
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
