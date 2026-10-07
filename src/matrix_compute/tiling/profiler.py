"""
Bandwidth profiling and network transmission analysis for matrix partitioning strategies.

Provides BandwidthProfile and BandwidthProfiler to measure and verify upload/download
optimality, redundancy ratios, and element/byte volume across 1D Outer-Product,
3D Block Tiling, and other partitioning topologies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from matrix_compute.tiling.partitioner import TileGridPlan, TileTaskPlan, MatrixTiler
from matrix_compute.tiling.precision import get_bytes_per_element, normalize_dtype


@dataclass
class BandwidthProfile:
    """
    Detailed bandwidth profile metrics for a partitioned matrix multiplication plan.

    Supports both object attribute access (profile.total_uploaded_elements)
    and dictionary subscription (profile["total_uploaded_elements"]).
    """

    m: int
    k: int
    n: int
    num_tasks: int
    strategy: str
    dtype: str
    total_a_elements: int
    total_b_elements: int
    total_uploaded_elements: int
    optimal_elements: int
    redundancy_elements: int
    redundancy_ratio: float
    bytes_per_element: int
    total_uploaded_bytes: int
    optimal_bytes: int
    redundancy_bytes: int
    is_upload_optimal: bool
    download_elements: int = 0
    optimal_download_elements: int = 0
    download_redundancy_ratio: float = 0.0
    is_download_optimal: bool = False

    # Backwards compatibility alias
    @property
    def num_strips(self) -> int:
        return self.num_tasks

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"BandwidthProfile has no field '{key}'")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        return [
            "m",
            "k",
            "n",
            "num_tasks",
            "num_strips",
            "strategy",
            "dtype",
            "total_a_elements",
            "total_b_elements",
            "total_uploaded_elements",
            "optimal_elements",
            "redundancy_elements",
            "redundancy_ratio",
            "bytes_per_element",
            "total_uploaded_bytes",
            "optimal_bytes",
            "redundancy_bytes",
            "is_upload_optimal",
            "download_elements",
            "optimal_download_elements",
            "download_redundancy_ratio",
            "is_download_optimal",
        ]

    def items(self) -> List[Tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.keys()}


class BandwidthProfiler:
    """
    Bandwidth profiler analyzing matrix partitioning plans and task sets.
    """

    @classmethod
    def profile_plan(cls, plan: TileGridPlan, strategy: Optional[str] = None) -> BandwidthProfile:
        """
        Profiles a TileGridPlan by summing the exact sizes of all task tiles.
        """
        m, k, n = plan.m, plan.k, plan.n
        dtype = plan.dtype
        tasks = plan.tasks
        if strategy is not None:
            strat = strategy
        elif plan.strategy is not None:
            strat = plan.strategy
        elif plan.num_m_blocks == 1 and plan.num_n_blocks == 1:
            strat = "outer_product_1d"
        elif plan.num_k_blocks == 1:
            strat = "output_grid_2d"
        else:
            strat = "block_3d"
        return cls.profile_tasks(tasks, m=m, k=k, n=n, dtype=dtype, strategy=strat)

    @classmethod
    def profile_tasks(
        cls,
        tasks: List[Union[TileTaskPlan, Dict[str, Any]]],
        m: int,
        k: int,
        n: int,
        dtype: str = "fp32",
        strategy: str = "outer_product_1d",
    ) -> BandwidthProfile:
        """
        Profiles a list of TileTaskPlan objects or task dictionaries.
        """
        norm_dtype = normalize_dtype(dtype)
        bytes_per_elem = get_bytes_per_element(norm_dtype)

        optimal_elements = m * k + k * n
        optimal_bytes = optimal_elements * bytes_per_elem

        total_a_elements = 0
        total_b_elements = 0
        download_elements = 0

        for t in tasks:
            if isinstance(t, TileTaskPlan):
                total_a_elements += t.a_tile.size
                total_b_elements += t.b_tile.size
                download_elements += (t.row_slice[1] - t.row_slice[0]) * (t.col_slice[1] - t.col_slice[0])
            elif isinstance(t, dict):
                a_tile = t.get("a_tile")
                b_tile = t.get("b_tile")
                if a_tile is not None and hasattr(a_tile, "size"):
                    total_a_elements += a_tile.size
                elif "m_i" in t and "k_j" in t:
                    total_a_elements += t["m_i"] * t["k_j"]
                elif "shape_a" in t:
                    total_a_elements += t["shape_a"][0] * t["shape_a"][1]

                if b_tile is not None and hasattr(b_tile, "size"):
                    total_b_elements += b_tile.size
                elif "k_j" in t and "n_k" in t:
                    total_b_elements += t["k_j"] * t["n_k"]
                elif "shape_b" in t:
                    total_b_elements += t["shape_b"][0] * t["shape_b"][1]

                if "row_slice" in t and "col_slice" in t:
                    rs = t["row_slice"]
                    cs = t["col_slice"]
                    download_elements += (rs[1] - rs[0]) * (cs[1] - cs[0])
                elif "m_i" in t and "n_k" in t:
                    download_elements += t["m_i"] * t["n_k"]
            else:
                a_tile = getattr(t, "a_tile", None)
                b_tile = getattr(t, "b_tile", None)
                if a_tile is not None and hasattr(a_tile, "size"):
                    total_a_elements += a_tile.size
                if b_tile is not None and hasattr(b_tile, "size"):
                    total_b_elements += b_tile.size

        total_uploaded_elements = total_a_elements + total_b_elements
        total_uploaded_bytes = total_uploaded_elements * bytes_per_elem

        redundancy_elements = total_uploaded_elements - optimal_elements
        redundancy_ratio = (
            (float(total_uploaded_elements) / float(optimal_elements)) - 1.0
            if optimal_elements > 0
            else 0.0
        )
        redundancy_bytes = total_uploaded_bytes - optimal_bytes

        optimal_download = m * n
        download_redundancy_ratio = (
            (float(download_elements) / float(optimal_download)) - 1.0
            if optimal_download > 0
            else 0.0
        )

        is_upload_optimal = (redundancy_elements == 0) and (abs(redundancy_ratio) < 1e-12)
        is_download_optimal = (download_elements == optimal_download) and (abs(download_redundancy_ratio) < 1e-12)

        return BandwidthProfile(
            m=m,
            k=k,
            n=n,
            num_tasks=len(tasks),
            strategy=strategy,
            dtype=norm_dtype,
            total_a_elements=total_a_elements,
            total_b_elements=total_b_elements,
            total_uploaded_elements=total_uploaded_elements,
            optimal_elements=optimal_elements,
            redundancy_elements=redundancy_elements,
            redundancy_ratio=redundancy_ratio,
            bytes_per_element=bytes_per_elem,
            total_uploaded_bytes=total_uploaded_bytes,
            optimal_bytes=optimal_bytes,
            redundancy_bytes=redundancy_bytes,
            is_upload_optimal=is_upload_optimal,
            download_elements=download_elements,
            optimal_download_elements=optimal_download,
            download_redundancy_ratio=download_redundancy_ratio,
            is_download_optimal=is_download_optimal,
        )

    @classmethod
    def profile_outer_product_1d(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        num_strips: int,
        dtype: str = "fp32",
    ) -> BandwidthProfile:
        """
        Profiles a 1D Outer-Product partitioning plan directly from matrices A and B.
        """
        plan = MatrixTiler.create_plan_outer_product_1d(a, b, num_strips=num_strips, dtype=dtype)
        return cls.profile_plan(plan, strategy="outer_product_1d")

    @classmethod
    def profile_output_grid_2d(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        p_r: Optional[int] = None,
        p_c: Optional[int] = None,
        tile_size: Optional[int] = None,
        dtype: str = "fp32",
    ) -> BandwidthProfile:
        """
        Profiles a 2D Output Grid plan directly from matrices A and B.
        """
        plan = MatrixTiler.create_plan_output_grid_2d(
            a=a,
            b=b,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            grid_shape=grid_shape,
            p_r=p_r,
            p_c=p_c,
            tile_size=tile_size,
            dtype=dtype,
        )
        return cls.profile_plan(plan, strategy="output_grid_2d")

    @classmethod
    def theoretical_2d_output_grid_bandwidth(
        cls,
        m: int,
        n: int,
        k: int,
        p_r: int = 1,
        p_c: int = 1,
        itemsize: Optional[int] = None,
        dtype: str = "fp32",
        **kwargs: Any,
    ) -> BandwidthProfile:
        """
        Calculates theoretical bandwidth metrics for 2D Output Grid partitioning.
        Formula:
        Total tasks: P_r * P_c
        Upload volume: P_c * M * K + P_r * K * N
        Download volume: M * N (zero download redundancy, optimal theoretical bound)
        """
        eff_m = int(kwargs.get("m", m))
        eff_n = int(kwargs.get("n", n))
        eff_k = int(kwargs.get("k", k))
        eff_pr = int(kwargs.get("p_r", kwargs.get("grid_rows", p_r)))
        eff_pc = int(kwargs.get("p_c", kwargs.get("grid_cols", p_c)))

        norm_dtype = normalize_dtype(dtype)
        bytes_per_elem = itemsize if itemsize is not None else get_bytes_per_element(norm_dtype)

        optimal_elements = eff_m * eff_k + eff_k * eff_n
        total_a = eff_pc * (eff_m * eff_k)
        total_b = eff_pr * (eff_k * eff_n)
        total_uploaded = total_a + total_b

        redundancy_elements = total_uploaded - optimal_elements
        redundancy_ratio = (
            (float(total_uploaded) / float(optimal_elements)) - 1.0
            if optimal_elements > 0
            else 0.0
        )

        optimal_download = eff_m * eff_n
        download_elements = eff_m * eff_n
        download_redundancy_ratio = 0.0

        is_upload_optimal = (eff_pr == 1 and eff_pc == 1)
        is_download_optimal = True

        return BandwidthProfile(
            m=eff_m,
            k=eff_k,
            n=eff_n,
            num_tasks=eff_pr * eff_pc,
            strategy="output_grid_2d",
            dtype=norm_dtype,
            total_a_elements=total_a,
            total_b_elements=total_b,
            total_uploaded_elements=total_uploaded,
            optimal_elements=optimal_elements,
            redundancy_elements=redundancy_elements,
            redundancy_ratio=redundancy_ratio,
            bytes_per_element=bytes_per_elem,
            total_uploaded_bytes=total_uploaded * bytes_per_elem,
            optimal_bytes=optimal_elements * bytes_per_elem,
            redundancy_bytes=redundancy_elements * bytes_per_elem,
            is_upload_optimal=is_upload_optimal,
            download_elements=download_elements,
            optimal_download_elements=optimal_download,
            download_redundancy_ratio=download_redundancy_ratio,
            is_download_optimal=is_download_optimal,
        )

    @classmethod
    def profile_block_3d(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        tile_size: int,
        dtype: str = "fp32",
    ) -> BandwidthProfile:
        """
        Profiles a 3D Block Tiling plan directly from matrices A and B.
        """
        plan = MatrixTiler.create_plan(a, b, tile_size=tile_size, dtype=dtype, strategy="block_3d")
        return cls.profile_plan(plan, strategy="block_3d")

    @classmethod
    def theoretical_3d_block_upload(
        cls,
        m: int,
        k: int,
        n: int,
        tile_size: int,
    ) -> Dict[str, Any]:
        """
        Calculates theoretical upload element volume and redundancy ratio for 3D block tiling:
        Grid dimensions: I = ceil(M / tile_size), J = ceil(N / tile_size), L = ceil(K / tile_size).
        Formula: total_uploaded = J * M * K + I * K * N.
        Redundancy ratio = (J * M * K + I * K * N) / (M * K + K * N) - 1.0.
        """
        i_blocks = math.ceil(m / tile_size)
        j_blocks = math.ceil(n / tile_size)
        l_blocks = math.ceil(k / tile_size)
        total_tasks = i_blocks * j_blocks * l_blocks

        optimal_elements = m * k + k * n
        # In 3D tiling, A tiles are reused across J column blocks; B tiles reused across I row blocks
        total_uploaded_elements = (j_blocks * m * k) + (i_blocks * k * n)
        redundancy_elements = total_uploaded_elements - optimal_elements
        redundancy_ratio = (
            (float(total_uploaded_elements) / float(optimal_elements)) - 1.0
            if optimal_elements > 0
            else 0.0
        )

        return {
            "m": m,
            "k": k,
            "n": n,
            "tile_size": tile_size,
            "i_blocks": i_blocks,
            "j_blocks": j_blocks,
            "l_blocks": l_blocks,
            "total_tasks": total_tasks,
            "optimal_elements": optimal_elements,
            "total_uploaded_elements": total_uploaded_elements,
            "redundancy_elements": redundancy_elements,
            "redundancy_ratio": redundancy_ratio,
        }

    @classmethod
    def verify_disjoint_coverage(
        cls,
        tasks: List[Union[TileTaskPlan, Dict[str, Any]]],
        k: int,
    ) -> bool:
        """
        Verifies that strip intervals [k_start, k_end) form a strict, contiguous,
        disjoint partition of [0, k) with zero gaps, zero overlaps, and zero empty strips.
        """
        if not tasks:
            return False

        # Extract (k_start, k_end)
        intervals: List[Tuple[int, int]] = []
        for t in tasks:
            if isinstance(t, TileTaskPlan):
                intervals.append((t.k_slice[0], t.k_slice[1]))
            elif isinstance(t, dict):
                ks = t.get("k_slice")
                if ks is not None:
                    intervals.append((ks[0], ks[1]))
                elif "k_start" in t and "k_end" in t:
                    intervals.append((t["k_start"], t["k_end"]))
                else:
                    return False
            else:
                ks = getattr(t, "k_slice", None)
                if ks is not None:
                    intervals.append((ks[0], ks[1]))
                else:
                    return False

        intervals.sort(key=lambda x: x[0])

        current = 0
        for start, end in intervals:
            if start != current:
                return False  # Gap or overlap
            if end <= start:
                return False  # Zero or negative width
            current = end

        return current == k

    @classmethod
    def verify_2d_grid_coverage(
        cls,
        tasks: List[Union[TileTaskPlan, Dict[str, Any]]],
        m: int,
        n: int,
    ) -> bool:
        """
        Verifies that [row_slice, col_slice] blocks across all tasks form
        an exact, contiguous, disjoint cover of [0, M) x [0, N) with zero gaps,
        zero overlaps, and zero out-of-bounds tiles.
        """
        if not tasks or m < 1 or n < 1:
            return False

        rectangles: List[Tuple[int, int, int, int]] = []
        total_area = 0

        for t in tasks:
            if isinstance(t, TileTaskPlan):
                rs, re = t.row_slice
                cs, ce = t.col_slice
            elif isinstance(t, dict):
                rs, re = t.get("row_slice", (0, 0))
                cs, ce = t.get("col_slice", (0, 0))
            else:
                rs, re = getattr(t, "row_slice", (0, 0))
                cs, ce = getattr(t, "col_slice", (0, 0))

            if rs < 0 or cs < 0 or re > m or ce > n or re <= rs or ce <= cs:
                return False

            rectangles.append((rs, re, cs, ce))
            total_area += (re - rs) * (ce - cs)

        if total_area != m * n:
            return False

        # Check pairwise disjointness
        for i in range(len(rectangles)):
            r1_s, r1_e, c1_s, c1_e = rectangles[i]
            for j in range(i + 1, len(rectangles)):
                r2_s, r2_e, c2_s, c2_e = rectangles[j]
                if max(r1_s, r2_s) < min(r1_e, r2_e) and max(c1_s, c2_s) < min(c1_e, c2_e):
                    return False

        return True

    verify_spatial_grid_coverage = verify_2d_grid_coverage
