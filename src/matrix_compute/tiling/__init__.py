"""Matrix tiling, partitioning, precision, and result assembly subsystem for matrix-compute-sdk."""

from .precision import (
    SUPPORTED_DTYPES,
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
from .partitioner import (
    MatrixTiler,
    TileGridPlan,
    TileTaskPlan,
    compute_1d_strips,
)
from .assembler import (
    ResultAssembler,
)

__all__ = [
    # Precision
    "SUPPORTED_DTYPES",
    "normalize_dtype",
    "get_numpy_dtype",
    "get_bytes_per_element",
    "truncate_to_bf16",
    "cast_matrix",
    "validate_bf16_masking",
    "get_unit_roundoff",
    "calculate_higham_bound",
    "emulate_gemm",
    # Partitioning
    "TileTaskPlan",
    "TileGridPlan",
    "MatrixTiler",
    "compute_1d_strips",
    # Assembly
    "ResultAssembler",
]
