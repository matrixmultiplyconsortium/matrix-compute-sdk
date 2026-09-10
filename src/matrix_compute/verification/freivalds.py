"""
Freivalds Mathematical Verification Engine with Higham Dynamic Error Bounds.

Implements randomized O(n^2) verification (ABr == Cr) over k independent trials
with dynamic error bounds scaled by input infinity norms and precision unit roundoffs,
incorporating relative discrepancy defenses against zero-work fraud.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from matrix_compute.config import ClientConfig
from matrix_compute.exceptions import MatrixDimensionError

# -----------------------------------------------------------------------------
# Precision Constants & Unit Roundoffs u
# -----------------------------------------------------------------------------

UNIT_ROUNDOFFS: Dict[str, float] = {
    "fp32": 2.0 ** -24,       # ~5.960464477539063e-8
    "float32": 2.0 ** -24,
    "single": 2.0 ** -24,
    "fp16": 2.0 ** -11,       # ~4.8828125e-4
    "float16": 2.0 ** -11,
    "half": 2.0 ** -11,
    "bf16": 2.0 ** -8,        # ~3.90625e-3
    "bfloat16": 2.0 ** -8,
    "fp64": 2.0 ** -53,       # ~1.1102230246251565e-16
    "float64": 2.0 ** -53,
    "double": 2.0 ** -53,
}

DEFAULT_FREIVALDS_ROUNDS: int = 5
DEFAULT_RELATIVE_TOLERANCE_FLOOR: float = 1e-7


def normalize_dtype(
    dtype: Optional[Union[str, np.dtype]],
    fallback_tensor: Optional[np.ndarray] = None,
) -> str:
    """
    Normalizes dtype string or numpy dtype to canonical format ('fp32', 'fp16', 'bf16', 'fp64').
    """
    if dtype is not None:
        dt_str = str(dtype).lower().strip()
        if dt_str in ("fp16", "float16", "half"):
            return "fp16"
        elif dt_str in ("bf16", "bfloat16", "bf"):
            return "bf16"
        elif dt_str in ("fp64", "float64", "double"):
            return "fp64"
        elif dt_str in ("fp32", "float32", "single", "f32"):
            return "fp32"

    if fallback_tensor is not None:
        if fallback_tensor.dtype == np.float16:
            return "fp16"
        elif fallback_tensor.dtype == np.float64:
            return "fp64"

    return "fp32"


def get_unit_roundoff(dtype: Union[str, np.dtype] = "fp32") -> float:
    """
    Returns the IEEE 754 unit roundoff u for the specified precision format.
    """
    norm_dt = normalize_dtype(dtype)
    return UNIT_ROUNDOFFS.get(norm_dt, 2.0 ** -24)


def matrix_infinity_norm(tensor: np.ndarray) -> float:
    """
    Calculates matrix infinity norm ||A||_inf (maximum absolute row sum).
    """
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        return 0.0
    return float(np.max(np.sum(np.abs(tensor.astype(np.float64)), axis=1)))


def higham_bound(k: int, norm_a: float, norm_b: float, dtype: str = "fp32") -> float:
    """
    Calculates Higham's matrix multiplication rounding bound:
    gamma_K = (K * u) / (1 - K * u)  [or 1000 * K * u if K * u >= 0.999]
    bound = gamma_K * ||A||_inf * ||B||_inf
    """
    u = get_unit_roundoff(dtype)
    ku = float(k) * u
    gamma_k = (ku / (1.0 - ku)) if ku < 0.999 else (ku * 1000.0)
    return gamma_k * float(norm_a) * float(norm_b)


def generate_random_binary_vector(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generates a random binary column vector r in {0, 1}^N with shape (N, 1).
    Guarantees that r is not identically zero.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    r = rng.randint(0, 2, size=(n, 1)).astype(np.float64)
    if not np.any(r):
        r[rng.randint(0, n), 0] = 1.0
    return r


class FreivaldsVerifier:
    """
    Scale-aware randomized Freivalds verification engine for distributed matrix tiles.
    """

    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        default_rounds: int = DEFAULT_FREIVALDS_ROUNDS,
        use_dynamic_bound: bool = True,
        relative_threshold: float = 1e-3,
        baseline_tolerance: float = 1e-4,
        **kwargs: Any,
    ):
        if "k_rounds" in kwargs and kwargs["k_rounds"] is not None:
            default_rounds = kwargs["k_rounds"]
        if "epsilon_tolerance" in kwargs and kwargs["epsilon_tolerance"] is not None:
            baseline_tolerance = kwargs["epsilon_tolerance"]
        if "relative_discrepancy_threshold" in kwargs and kwargs["relative_discrepancy_threshold"] is not None:
            relative_threshold = kwargs["relative_discrepancy_threshold"]

        if config is not None:
            self.default_rounds = config.freivalds_rounds
            self.use_dynamic_bound = config.use_dynamic_freivalds_bound
            self.relative_threshold = config.relative_discrepancy_threshold
            self.baseline_tolerance = config.verification_tolerance
        else:
            self.default_rounds = default_rounds
            self.use_dynamic_bound = use_dynamic_bound
            self.relative_threshold = relative_threshold
            self.baseline_tolerance = baseline_tolerance

    @classmethod
    def verify(
        cls,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        dtype: Optional[str] = None,
        k_rounds: int = DEFAULT_FREIVALDS_ROUNDS,
        seed: Optional[int] = None,
        epsilon: Optional[float] = None,
        config: Optional[ClientConfig] = None,
        **kwargs: Any,
    ) -> Tuple[bool, float, float, Dict[str, Any]]:
        """
        Classmethod entrypoint for Freivalds verification.
        Returns: (is_valid, max_discrepancy, dynamic_bound, proof_data)
        """
        if a is None:
            a = kwargs.get("A", kwargs.get("A_tile", kwargs.get("a_tile")))
        if b is None:
            b = kwargs.get("B", kwargs.get("B_tile", kwargs.get("b_tile")))
        if c is None:
            c = kwargs.get("C", kwargs.get("C_tile", kwargs.get("c_tile")))
        if "k_rounds" in kwargs and kwargs["k_rounds"] is not None:
            k_rounds = kwargs["k_rounds"]

        verifier = cls(config=config, default_rounds=k_rounds)
        return verifier.verify_tile(
            a=a,
            b=b,
            c=c,
            dtype=dtype,
            k_rounds=k_rounds,
            seed=seed,
            epsilon=epsilon,
        )

    def verify_tile(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        dtype: Optional[str] = None,
        k_rounds: Optional[int] = None,
        seed: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> Tuple[bool, float, float, Dict[str, Any]]:
        """
        Executes randomized Freivalds verification over k_rounds.
        """
        # 1. Dimension & Shape Validation
        if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
            raise MatrixDimensionError(
                f"Matrices must be 2D: A is {a.ndim}D, B is {b.ndim}D, C is {c.ndim}D",
                shape_a=getattr(a, "shape", None),
                shape_b=getattr(b, "shape", None),
            )

        m, k = a.shape
        k_b, n = b.shape
        if k != k_b:
            raise MatrixDimensionError(
                f"Inner dimension mismatch: A cols ({k}) != B rows ({k_b})",
                shape_a=a.shape,
                shape_b=b.shape,
                expected_inner_dim=k,
            )

        if c.shape != (m, n):
            raise MatrixDimensionError(
                f"Result matrix shape mismatch: C {c.shape} != expected ({m}, {n})",
                shape_a=a.shape,
                shape_b=b.shape,
            )

        norm_dtype = normalize_dtype(dtype, fallback_tensor=a)
        rounds = k_rounds if k_rounds is not None else self.default_rounds

        # 2. Check Finiteness (NaN / Inf)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
            r = generate_random_binary_vector(n, seed=seed)
            # Create diagnostic failure proof
            with np.errstate(all="ignore"):
                br = np.matmul(b.astype(np.float64), r)
                abr = np.matmul(a.astype(np.float64), br)
                cr = np.matmul(c.astype(np.float64), r)

            failure_proof = {
                "random_vector_r": r.flatten().astype(int).tolist(),
                "vector_br": br.flatten().astype(float).tolist(),
                "vector_abr": abr.flatten().astype(float).tolist(),
                "vector_cr": cr.flatten().astype(float).tolist(),
                "observed_diff": float("inf"),
                "dynamic_bound": 1e-6,
                "max_observed_discrepancy": float("inf"),
                "dimension_m": int(m),
                "dimension_k": int(k),
                "dimension_n": int(n),
                "dtype": norm_dtype,
                "reason": "Non-finite value (NaN/Inf) detected in matrices",
            }
            return False, float("inf"), 1e-6, failure_proof

        # 3. Compute Higham Dynamic Error Bound
        norm_a = matrix_infinity_norm(a)
        norm_b = matrix_infinity_norm(b)
        scale = norm_a * norm_b

        u = get_unit_roundoff(norm_dtype)
        ku = float(k) * u
        gamma_k = (ku / (1.0 - ku)) if ku < 0.999 else (ku * 1000.0)

        raw_dynamic_bound = gamma_k * scale

        # Calculate effective bound accounting for optional epsilon override
        if epsilon is not None:
            if scale >= 1.0:
                effective_bound = max(raw_dynamic_bound, epsilon)
            else:
                effective_bound = max(raw_dynamic_bound, epsilon * scale, 1e-6)
        else:
            effective_bound = max(raw_dynamic_bound, 1e-6)

        # 4. Randomized Freivalds Trials
        if seed is not None:
            np.random.seed(seed)

        max_discrepancy = 0.0
        best_failure_proof: Dict[str, Any] = {}
        is_fraud = False

        a_f64 = a.astype(np.float64)
        b_f64 = b.astype(np.float64)
        c_f64 = c.astype(np.float64)

        for trial_idx in range(rounds):
            r = generate_random_binary_vector(n)
            br = np.matmul(b_f64, r)
            abr = np.matmul(a_f64, br)
            cr = np.matmul(c_f64, r)

            diff_vec = np.abs(abr - cr)
            diff = float(np.max(diff_vec))

            if np.isnan(diff) or np.isinf(diff):
                is_fraud = True
                max_discrepancy = float("inf")
                best_failure_proof = {
                    "random_vector_r": r.flatten().astype(int).tolist(),
                    "vector_br": br.flatten().astype(float).tolist(),
                    "vector_abr": abr.flatten().astype(float).tolist(),
                    "vector_cr": cr.flatten().astype(float).tolist(),
                    "observed_diff": float("inf"),
                    "dynamic_bound": float(effective_bound),
                    "max_observed_discrepancy": float("inf"),
                    "dimension_m": int(m),
                    "dimension_k": int(k),
                    "dimension_n": int(n),
                    "dtype": norm_dtype,
                }
                break

            if diff > max_discrepancy:
                max_discrepancy = diff

            # Relative discrepancy check to defend against zero-work fraud
            abr_norm = float(np.max(np.abs(abr)))
            cr_norm = float(np.max(np.abs(cr)))
            denom = max(scale, abr_norm, cr_norm)

            trial_failed = False
            if denom > 0.0:
                rel_diff = diff / denom
                allowed_rel = max(gamma_k, epsilon if epsilon is not None else 0.0, DEFAULT_RELATIVE_TOLERANCE_FLOOR)
                if diff > effective_bound or rel_diff > allowed_rel:
                    trial_failed = True
            else:
                # scale == 0 (A=0 or B=0) -> A*B == 0. C must be 0.
                if diff > DEFAULT_RELATIVE_TOLERANCE_FLOOR:
                    trial_failed = True

            if trial_failed:
                is_fraud = True
                # Record or update failure proof with the highest discrepancy observed
                if not best_failure_proof or diff > best_failure_proof.get("observed_diff", -1.0):
                    best_failure_proof = {
                        "random_vector_r": r.flatten().astype(int).tolist(),
                        "vector_br": br.flatten().astype(float).tolist(),
                        "vector_abr": abr.flatten().astype(float).tolist(),
                        "vector_cr": cr.flatten().astype(float).tolist(),
                        "observed_diff": float(diff),
                        "dynamic_bound": float(effective_bound),
                        "max_observed_discrepancy": float(diff),
                        "dimension_m": int(m),
                        "dimension_k": int(k),
                        "dimension_n": int(n),
                        "dtype": norm_dtype,
                    }

        # 5. Return Outcome
        if is_fraud:
            reported_disc = best_failure_proof.get("observed_diff", max_discrepancy)
            return False, reported_disc, effective_bound, best_failure_proof
        else:
            return True, max_discrepancy, effective_bound, {}


def freivalds_verify(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    dtype: str = "fp32",
    num_trials: int = DEFAULT_FREIVALDS_ROUNDS,
    seed: Optional[int] = None,
    epsilon: Optional[float] = None,
) -> Tuple[bool, float, float, Dict[str, Any]]:
    """
    Functional convenience wrapper for Freivalds verification.
    """
    return FreivaldsVerifier.verify(
        a=a,
        b=b,
        c=c,
        dtype=dtype,
        k_rounds=num_trials,
        seed=seed,
        epsilon=epsilon,
    )
