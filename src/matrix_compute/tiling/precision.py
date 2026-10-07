"""
Precision normalization, casting, and IEEE bfloat16 emulation for matrix-compute-sdk.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Canonical supported datatype identifiers
SUPPORTED_DTYPES: Set[str] = {"fp32", "fp16", "bf16"}

# Common aliases mapped to canonical identifiers
DTYPE_ALIASES: Dict[str, str] = {
    "fp32": "fp32",
    "float32": "fp32",
    "f32": "fp32",
    "single": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
    "f16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "bf": "bf16",
    "fp64": "fp64",
    "float64": "fp64",
    "f64": "fp64",
    "double": "fp64",
}


def normalize_dtype(dtype: Union[str, np.dtype, type]) -> str:
    """
    Normalizes string, NumPy dtype, or Python type into canonical string ('fp32', 'fp16', 'bf16').

    Raises:
        ValueError: If dtype is unrecognized or unsupported.
    """
    if isinstance(dtype, (np.dtype, type)):
        if dtype == np.float32:
            return "fp32"
        elif dtype == np.float16:
            return "fp16"
        elif dtype == np.float64:
            return "fp64"
        dtype_str = getattr(dtype, "name", str(dtype))
    else:
        dtype_str = str(dtype)

    norm = dtype_str.lower().strip()
    if norm in DTYPE_ALIASES:
        canonical = DTYPE_ALIASES[norm]
        if canonical not in SUPPORTED_DTYPES and canonical != "fp64":
            raise ValueError(
                f"Unsupported precision dtype '{dtype_str}'. Supported dtypes: {sorted(SUPPORTED_DTYPES)}"
            )
        return canonical

    raise ValueError(
        f"Unsupported precision dtype '{dtype_str}'. Supported dtypes: {sorted(SUPPORTED_DTYPES)}"
    )


def get_numpy_dtype(dtype: Union[str, np.dtype, type]) -> np.dtype:
    """
    Returns the underlying NumPy storage dtype for a canonical precision identifier.
    Note: 'bf16' is stored as np.float32 with masked mantissa.
    """
    norm = normalize_dtype(dtype)
    if norm == "fp16":
        return np.dtype(np.float16)
    elif norm == "fp64":
        return np.dtype(np.float64)
    else:
        # fp32 and bf16 both store in float32
        return np.dtype(np.float32)


def get_bytes_per_element(dtype: Union[str, np.dtype, type]) -> int:
    """
    Returns bytes per element for buffer calculation and streaming bandwidth estimation.
    'fp32' -> 4, 'fp16' -> 2, 'bf16' -> 4 (FP32 container format), 'fp64' -> 8.
    """
    norm = normalize_dtype(dtype)
    if norm == "fp16":
        return 2
    elif norm == "fp64":
        return 8
    return 4


def truncate_to_bf16(tensor: np.ndarray) -> np.ndarray:
    """
    Truncates / masks float32 mantissa to 7 bits (IEEE bfloat16 emulation).
    Clears the lowest 16 bits of 32-bit float representation (mask 0xFFFF0000).

    Guarantees:
    - Contiguous memory layout.
    - Preservation of IEEE NaN semantics (restores quiet NaN bit 0x00400000 if lower-16-bit
      masking cleared the mantissa entirely).
    - Preservation of IEEE +/- Infinity semantics.
    - Preservation of signed zeros (+0.0 and -0.0).
    - Subnormal underflow without generating NaNs.
    """
    f32 = np.ascontiguousarray(tensor, dtype=np.float32)
    u32 = f32.view(np.uint32)
    masked_u32 = u32 & np.uint32(0xFFFF0000)

    # Preserve NaN semantics: if input was NaN and lower-16-bit truncation cleared
    # the entire mantissa to 0 (which would otherwise falsely convert NaN to Infinity),
    # restore the quiet NaN bit (bit 22, 0x00400000).
    is_nan = np.isnan(f32)
    if np.any(is_nan):
        lost_mantissa = is_nan & ((masked_u32 & np.uint32(0x007F0000)) == 0)
        if np.any(lost_mantissa):
            masked_u32 = masked_u32.copy()
            masked_u32[lost_mantissa] |= np.uint32(0x00400000)

    return masked_u32.view(np.float32)


def cast_matrix(tensor: np.ndarray, dtype: Union[str, np.dtype, type] = "fp32") -> np.ndarray:
    """
    Casts an input NumPy array into the specified precision format.

    - 'fp32': returns contiguous np.float32 array.
    - 'fp16': returns contiguous np.float16 array.
    - 'bf16': returns contiguous np.float32 array with 16-bit masked mantissa.
    - 'fp64': returns contiguous np.float64 array.
    """
    norm = normalize_dtype(dtype)
    if norm == "fp16":
        return np.ascontiguousarray(tensor, dtype=np.float16)
    elif norm == "bf16":
        return truncate_to_bf16(tensor)
    elif norm == "fp64":
        return np.ascontiguousarray(tensor, dtype=np.float64)
    else:
        return np.ascontiguousarray(tensor, dtype=np.float32)


def validate_bf16_masking(tensor: np.ndarray) -> bool:
    """
    Validates that every element in an FP32 array has its lower 16 bits zeroed (valid BF16).
    """
    if tensor.dtype != np.float32:
        return False
    u32 = np.ascontiguousarray(tensor).view(np.uint32)
    return bool(np.all((u32 & np.uint32(0x0000FFFF)) == 0))


def get_unit_roundoff(dtype: Union[str, np.dtype, type] = "fp32") -> float:
    """
    Returns IEEE 754 unit roundoff u for the specified precision:
    - fp16: 2^-11 (~4.8828e-4)
    - bf16: 2^-8  (~3.90625e-3)
    - fp32: 2^-24 (~5.9605e-8)
    - fp64: 2^-53 (~1.1102e-16)
    """
    dt = normalize_dtype(dtype)
    if dt == "fp16":
        return 2.0 ** -11
    elif dt == "bf16":
        return 2.0 ** -8
    elif dt == "fp64":
        return 2.0 ** -53
    else:
        return 2.0 ** -24


def calculate_higham_bound(
    k: int,
    norm_a: Optional[Union[float, np.ndarray]] = None,
    norm_b: Optional[Union[float, np.ndarray]] = None,
    dtype: Union[str, np.dtype, type] = "fp32",
    *,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
) -> float:
    """
    Calculates Higham's matrix multiplication rounding bound:
    gamma_K = (K * u) / (1 - K * u)  [or 1000 * K * u if K * u >= 0.999]
    bound = gamma_K * ||A||_inf * ||B||_inf
    """
    if a is not None:
        norm_a_val = float(np.max(np.sum(np.abs(a.astype(np.float64)), axis=1))) if a.ndim == 2 and a.size > 0 else 0.0
    elif isinstance(norm_a, np.ndarray):
        norm_a_val = float(np.max(np.sum(np.abs(norm_a.astype(np.float64)), axis=1))) if norm_a.ndim == 2 and norm_a.size > 0 else 0.0
    elif norm_a is not None:
        norm_a_val = float(norm_a)
    else:
        norm_a_val = 1.0

    if b is not None:
        norm_b_val = float(np.max(np.sum(np.abs(b.astype(np.float64)), axis=1))) if b.ndim == 2 and b.size > 0 else 0.0
    elif isinstance(norm_b, np.ndarray):
        norm_b_val = float(np.max(np.sum(np.abs(norm_b.astype(np.float64)), axis=1))) if norm_b.ndim == 2 and norm_b.size > 0 else 0.0
    elif norm_b is not None:
        norm_b_val = float(norm_b)
    else:
        norm_b_val = 1.0

    u = get_unit_roundoff(dtype)
    ku = float(k) * u
    gamma_k = (ku / (1.0 - ku)) if ku < 0.999 else (ku * 1000.0)
    return gamma_k * norm_a_val * norm_b_val


def emulate_gemm(a: np.ndarray, b: np.ndarray, dtype: Union[str, np.dtype, type] = "fp32") -> np.ndarray:
    """
    Authoritative reference GEMM matrix multiplication C = A x B computed in float64
    and cast / truncated to the target precision.
    """
    norm = normalize_dtype(dtype)
    if norm == "bf16":
        a_bf16 = truncate_to_bf16(a)
        b_bf16 = truncate_to_bf16(b)
        c = np.matmul(a_bf16.astype(np.float64), b_bf16.astype(np.float64)).astype(np.float32)
        return truncate_to_bf16(c)

    a_f64 = a.astype(np.float64)
    b_f64 = b.astype(np.float64)
    c_f64 = np.matmul(a_f64, b_f64)

    if norm == "fp16":
        return c_f64.astype(np.float16)
    elif norm == "fp64":
        return c_f64.astype(np.float64)
    return c_f64.astype(np.float32)
