"""
Re-export Freivalds verification symbols for matrix_compute.tiling namespace.
"""

from matrix_compute.verification.freivalds import (
    DEFAULT_FREIVALDS_ROUNDS,
    DEFAULT_RELATIVE_TOLERANCE_FLOOR,
    UNIT_ROUNDOFFS,
    FreivaldsVerifier,
    freivalds_verify,
    generate_random_binary_vector,
    get_unit_roundoff,
    higham_bound,
    matrix_infinity_norm,
    normalize_dtype,
)

__all__ = [
    "DEFAULT_FREIVALDS_ROUNDS",
    "DEFAULT_RELATIVE_TOLERANCE_FLOOR",
    "UNIT_ROUNDOFFS",
    "FreivaldsVerifier",
    "freivalds_verify",
    "generate_random_binary_vector",
    "get_unit_roundoff",
    "higham_bound",
    "matrix_infinity_norm",
    "normalize_dtype",
]
