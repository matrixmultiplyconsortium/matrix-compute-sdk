"""
Matrix tiling and sub-block task partitioning for matrix-compute-sdk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from matrix_compute.exceptions import MatrixDimensionError, TilingError
from matrix_compute.tiling.precision import cast_matrix, normalize_dtype


@dataclass
class TileTaskPlan:
    """
    Execution plan for an individual sub-matrix tile multiplication task.
    Represents the computation C_sub = A_tile x B_tile where:
    - A_tile has shape (M_i, K_j)
    - B_tile has shape (K_j, N_k)
    - C_sub has shape (M_i, N_k)

    Supports both object attribute access and dictionary subscription (plan['a_tile']).
    """

    task_id: str
    row_idx: int
    col_idx: int
    k_idx: int
    row_slice: Tuple[int, int]
    col_slice: Tuple[int, int]
    k_slice: Tuple[int, int]
    a_tile: np.ndarray
    b_tile: np.ndarray
    dtype: str
    estimated_gflops: float

    @property
    def m_i(self) -> int:
        return self.row_slice[1] - self.row_slice[0]

    @property
    def k_j(self) -> int:
        return self.k_slice[1] - self.k_slice[0]

    @property
    def n_k(self) -> int:
        return self.col_slice[1] - self.col_slice[0]

    @property
    def shape_a(self) -> Tuple[int, int]:
        return (self.m_i, self.k_j)

    @property
    def shape_b(self) -> Tuple[int, int]:
        return (self.k_j, self.n_k)

    @property
    def shape_c(self) -> Tuple[int, int]:
        return (self.m_i, self.n_k)

    # Dictionary emulation for backwards and reference compatibility
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"TileTaskPlan has no field '{key}'")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        return [
            "task_id",
            "row_idx",
            "col_idx",
            "k_idx",
            "row_slice",
            "col_slice",
            "k_slice",
            "a_tile",
            "b_tile",
            "dtype",
            "estimated_gflops",
        ]

    def values(self) -> List[Any]:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


@dataclass
class TileGridPlan:
    """Complete manifest describing a tiled matrix multiplication grid."""

    m: int
    k: int
    n: int
    tile_size: int
    dtype: str
    num_m_blocks: int
    num_k_blocks: int
    num_n_blocks: int
    total_tasks: int
    total_gflops: float
    tasks: List[TileTaskPlan]

    def iter_tasks(self) -> Iterator[TileTaskPlan]:
        return iter(self.tasks)

    def get_task(self, row_idx: int, col_idx: int, k_idx: int) -> Optional[TileTaskPlan]:
        for t in self.tasks:
            if t.row_idx == row_idx and t.col_idx == col_idx and t.k_idx == k_idx:
                return t
        return None


class MatrixTiler:
    """
    Production-grade matrix tiling engine.
    Partitions arbitrary 2D matrices A (M x K) and B (K x N) into sub-blocks.
    """

    def __init__(self, default_tile_size: int = 1024, default_dtype: str = "fp32") -> None:
        if default_tile_size < 1:
            raise ValueError(f"default_tile_size must be >= 1, got {default_tile_size}")
        self.default_tile_size = default_tile_size
        self.default_dtype = normalize_dtype(default_dtype)

    @staticmethod
    def calculate_gflops(m: int, k: int, n: int) -> float:
        """Calculates exact theoretical GFLOPs: (2 * M * K * N) / 10^9."""
        return (2.0 * float(m) * float(k) * float(n)) / 1e9

    @classmethod
    def partition(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        tile_size: Optional[int] = None,
        dtype: Optional[str] = None,
        task_id_prefix: Optional[str] = None,
    ) -> List[TileTaskPlan]:
        """
        Partitions matrices A (M x K) and B (K x N) into a flat list of TileTaskPlans.

        Raises:
            MatrixDimensionError: If shapes are invalid or inner dimensions mismatch.
            ValueError / TilingError: If tile_size is invalid.
        """
        plan = cls.create_plan(
            a=a,
            b=b,
            tile_size=tile_size,
            dtype=dtype,
            task_id_prefix=task_id_prefix,
        )
        return plan.tasks

    @classmethod
    def create_plan(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        tile_size: Optional[int] = None,
        dtype: Optional[str] = None,
        task_id_prefix: Optional[str] = None,
    ) -> TileGridPlan:
        """
        Creates a comprehensive TileGridPlan manifest for matrices A and B.
        """
        # Validate 2D dimensions
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise MatrixDimensionError(
                "Matrices A and B must be numpy.ndarray instances",
                shape_a=getattr(a, "shape", None),
                shape_b=getattr(b, "shape", None),
            )

        if a.ndim != 2 or b.ndim != 2:
            raise MatrixDimensionError(
                f"Matrices must be 2-dimensional. Got shapes {a.shape} and {b.shape}",
                shape_a=a.shape,
                shape_b=b.shape,
            )

        m, k = a.shape
        k2, n = b.shape

        if k != k2:
            raise MatrixDimensionError(
                f"Dimension mismatch: A columns ({k}) != B rows ({k2})",
                shape_a=a.shape,
                shape_b=b.shape,
                expected_inner_dim=k,
            )

        if m == 0 or k == 0 or n == 0:
            raise MatrixDimensionError(
                f"Matrix dimensions must be non-zero. Got {a.shape} and {b.shape}",
                shape_a=a.shape,
                shape_b=b.shape,
            )

        effective_tile_size = int(tile_size) if tile_size is not None else 1024
        if effective_tile_size < 1:
            raise ValueError(f"Tile size must be >= 1, got {effective_tile_size}")

        norm_dtype = normalize_dtype(dtype or "fp32")
        a_cast = cast_matrix(a, norm_dtype)
        b_cast = cast_matrix(b, norm_dtype)

        num_m = math.ceil(m / effective_tile_size)
        num_k = math.ceil(k / effective_tile_size)
        num_n = math.ceil(n / effective_tile_size)

        prefix = task_id_prefix or "tile"
        tasks: List[TileTaskPlan] = []
        total_gflops = 0.0

        for i in range(num_m):
            row_start = i * effective_tile_size
            row_end = min((i + 1) * effective_tile_size, m)
            m_i = row_end - row_start

            for j in range(num_n):
                col_start = j * effective_tile_size
                col_end = min((j + 1) * effective_tile_size, n)
                n_k = col_end - col_start

                for p in range(num_k):
                    k_start = p * effective_tile_size
                    k_end = min((p + 1) * effective_tile_size, k)
                    k_j = k_end - k_start

                    a_tile = a_cast[row_start:row_end, k_start:k_end]
                    b_tile = b_cast[k_start:k_end, col_start:col_end]

                    gflops = cls.calculate_gflops(m_i, k_j, n_k)
                    total_gflops += gflops

                    task_id = f"{prefix}_{i}_{j}_{p}"

                    tasks.append(
                        TileTaskPlan(
                            task_id=task_id,
                            row_idx=i,
                            col_idx=j,
                            k_idx=p,
                            row_slice=(row_start, row_end),
                            col_slice=(col_start, col_end),
                            k_slice=(k_start, k_end),
                            a_tile=a_tile,
                            b_tile=b_tile,
                            dtype=norm_dtype,
                            estimated_gflops=gflops,
                        )
                    )

        return TileGridPlan(
            m=m,
            k=k,
            n=n,
            tile_size=effective_tile_size,
            dtype=norm_dtype,
            num_m_blocks=num_m,
            num_k_blocks=num_k,
            num_n_blocks=num_n,
            total_tasks=len(tasks),
            total_gflops=total_gflops,
            tasks=tasks,
        )
