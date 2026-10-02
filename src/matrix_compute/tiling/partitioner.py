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
    strategy: Optional[str] = None

    def __post_init__(self) -> None:
        if self.strategy is None:
            if self.num_m_blocks == 1 and self.num_n_blocks == 1:
                self.strategy = "outer_product_1d"
            else:
                self.strategy = "block_3d"

    @property
    def p(self) -> int:
        """Alias for num_k_blocks (the number of strips / partitions along dimension K)."""
        return self.num_k_blocks

    def iter_tasks(self) -> Iterator[TileTaskPlan]:
        return iter(self.tasks)

    def get_task(self, row_idx: int, col_idx: int, k_idx: int) -> Optional[TileTaskPlan]:
        for t in self.tasks:
            if t.row_idx == row_idx and t.col_idx == col_idx and t.k_idx == k_idx:
                return t
        return None

    # Dictionary emulation for backwards and reference compatibility
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"TileGridPlan has no field '{key}'")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        return [
            "m",
            "k",
            "n",
            "tile_size",
            "dtype",
            "num_m_blocks",
            "num_k_blocks",
            "num_n_blocks",
            "total_tasks",
            "total_gflops",
            "tasks",
            "strategy",
            "p",
        ]

    def values(self) -> List[Any]:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def compute_1d_strips(k: int, p: int) -> List[Tuple[int, int]]:
    """
    Computes 1D strip intervals [k_start, k_end) along dimension k.

    Clamps p_effective = min(max(1, int(p)), k) ensuring 1 <= p_effective <= k.
    Distributes remainder r = k % p_effective across the first r strips:
    - First r strips have length q + 1
    - Remaining p_effective - r strips have length q

    Args:
        k: Inner dimension length (must be >= 1).
        p: Target number of strips.

    Returns:
        List of (k_start, k_end) tuples forming a contiguous, disjoint partition of [0, k).

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"Inner dimension k must be >= 1, got {k}")

    p_effective = min(max(1, int(p)), k)
    q = k // p_effective
    r = k % p_effective

    strips: List[Tuple[int, int]] = []
    curr = 0
    for i in range(p_effective):
        strip_len = q + 1 if i < r else q
        strips.append((curr, curr + strip_len))
        curr += strip_len

    return strips


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
    def create_plan_outer_product_1d(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        num_strips: int,
        dtype: Optional[str] = "fp32",
        job_id: Optional[str] = None,
        task_id_prefix: Optional[str] = None,
        tile_size: Optional[int] = None,
    ) -> TileGridPlan:
        """
        Creates a TileGridPlan partitioning inner dimension K into P strips (OUTER_PRODUCT_1D).

        Args:
            a: Input matrix A of shape (M, K).
            b: Input matrix B of shape (K, N).
            num_strips: Desired number of strips.
            dtype: Precision identifier ("fp32", "fp16", "bf16").
            job_id: Optional job identifier prefix.
            task_id_prefix: Optional task id prefix alias.
            tile_size: Optional tile size override descriptor.

        Returns:
            TileGridPlan manifest with P_effective TileTaskPlans.
        """
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

        norm_dtype = normalize_dtype(dtype or "fp32")
        a_cast = cast_matrix(a, norm_dtype)
        b_cast = cast_matrix(b, norm_dtype)

        strips = compute_1d_strips(k, num_strips)
        p_effective = len(strips)

        prefix = job_id or task_id_prefix
        tasks: List[TileTaskPlan] = []
        total_gflops = 0.0

        for p, (k_start, k_end) in enumerate(strips):
            k_p = k_end - k_start
            a_tile = a_cast[:, k_start:k_end]
            b_tile = b_cast[k_start:k_end, :]

            gflops = cls.calculate_gflops(m, k_p, n)
            total_gflops += gflops

            task_id = f"{prefix}_strip_{p}" if prefix else f"strip_{p}"

            tasks.append(
                TileTaskPlan(
                    task_id=task_id,
                    row_idx=0,
                    col_idx=0,
                    k_idx=p,
                    row_slice=(0, m),
                    col_slice=(0, n),
                    k_slice=(k_start, k_end),
                    a_tile=a_tile,
                    b_tile=b_tile,
                    dtype=norm_dtype,
                    estimated_gflops=gflops,
                )
            )

        eff_tile_size = int(tile_size) if tile_size is not None else math.ceil(k / p_effective)

        return TileGridPlan(
            m=m,
            k=k,
            n=n,
            tile_size=eff_tile_size,
            dtype=norm_dtype,
            num_m_blocks=1,
            num_k_blocks=p_effective,
            num_n_blocks=1,
            total_tasks=p_effective,
            total_gflops=total_gflops,
            tasks=tasks,
            strategy="outer_product_1d",
        )

    @classmethod
    def partition_outer_product_1d(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        num_strips: int,
        dtype: Optional[str] = "fp32",
        job_id: Optional[str] = None,
        task_id_prefix: Optional[str] = None,
        tile_size: Optional[int] = None,
    ) -> List[TileTaskPlan]:
        """Partitions matrices into a flat list of 1D outer-product strip tasks."""
        plan = cls.create_plan_outer_product_1d(
            a=a,
            b=b,
            num_strips=num_strips,
            dtype=dtype,
            job_id=job_id,
            task_id_prefix=task_id_prefix,
            tile_size=tile_size,
        )
        return plan.tasks

    @classmethod
    def partition(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        tile_size: Optional[int] = None,
        dtype: Optional[str] = None,
        task_id_prefix: Optional[str] = None,
        strategy: Optional[Union[str, Any]] = None,
        num_strips: Optional[int] = None,
        **kwargs: Any,
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
            strategy=strategy,
            num_strips=num_strips,
            **kwargs,
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
        strategy: Optional[Union[str, Any]] = None,
        num_strips: Optional[int] = None,
        **kwargs: Any,
    ) -> TileGridPlan:
        """
        Creates a comprehensive TileGridPlan manifest for matrices A and B.
        Routes execution based on partitioning strategy:
        - BLOCK_3D (default): Standard 3D cubic sub-block tiling across M, N, and K.
        - OUTER_PRODUCT_1D: Upload-optimal 1D strip partitioning across inner dimension K.
        """
        raw_strat = strategy or kwargs.get("partitioning_strategy") or "block_3d"
        strat_str = raw_strat.value if hasattr(raw_strat, "value") else str(raw_strat)
        strat_norm = strat_str.lower().strip()

        if strat_norm in (
            "outer_product_1d",
            "outer_1d",
            "1d_outer_product",
            "outer_product",
            "outerproduct1d",
        ):
            if num_strips is not None:
                strips_count = int(num_strips)
            elif kwargs.get("num_strips") is not None:
                strips_count = int(kwargs["num_strips"])
            elif isinstance(a, np.ndarray) and a.ndim == 2:
                k_val = a.shape[1]
                if tile_size is not None:
                    strips_count = max(1, math.ceil(k_val / int(tile_size)))
                else:
                    strips_count = max(1, math.ceil(k_val / 1024))
            else:
                strips_count = 1

            job_id = kwargs.get("job_id") or task_id_prefix
            return cls.create_plan_outer_product_1d(
                a=a,
                b=b,
                num_strips=strips_count,
                dtype=dtype,
                job_id=job_id,
                tile_size=tile_size,
            )

        if strat_norm in (
            "output_grid_2d",
            "output_grid",
            "2d_output_grid",
            "outputgrid2d",
            "grid_2d",
            "output_2d",
        ):
            raise NotImplementedError("OUTPUT_GRID_2D partitioning is planned for a future milestone")

        if strat_norm not in ("block_3d", "block", "3d", "3d_block", "block3d"):
            raise ValueError(f"Unknown partitioning strategy: '{raw_strat}'")

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
            strategy="block_3d",
        )
