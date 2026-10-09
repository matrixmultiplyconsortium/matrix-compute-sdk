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
MAX_RELATIVE_DISCREPANCY_CEILING: float = 0.50


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


def vector_infinity_norm(vec: np.ndarray) -> float:
    """Calculates vector infinity norm ||v||_inf = max(|v_i|)."""
    if vec.size == 0:
        return 0.0
    return float(np.max(np.abs(vec.astype(np.float64))))


class IterativeVerificationResult(tuple):
    """
    Result of single-step iterative Freivalds verification.
    Subclasses tuple (is_valid, discrepancy, effective_bound, failure_proof)
    so it can be unpacked as a 4-tuple or evaluated directly as a boolean.
    """

    def __new__(
        cls,
        is_valid: bool,
        discrepancy: float,
        effective_bound: float,
        failure_proof: Dict[str, Any],
    ) -> "IterativeVerificationResult":
        return super().__new__(
            cls, (bool(is_valid), float(discrepancy), float(effective_bound), failure_proof)
        )

    @property
    def is_valid(self) -> bool:
        return bool(self[0])

    @property
    def discrepancy(self) -> float:
        return float(self[1])

    @property
    def effective_bound(self) -> float:
        return float(self[2])

    @property
    def failure_proof(self) -> Dict[str, Any]:
        return self[3]

    def __bool__(self) -> bool:
        return bool(self[0])


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

        # Calculate effective bound accounting for optional epsilon override.
        # For normalized matrices, maintain Higham precision bound without
        # artificial 1e-6 clamping that broadens fraud detection blind spots.
        min_bound_floor = u * scale if scale > 0 else u
        if epsilon is not None:
            if scale >= 1.0:
                effective_bound = max(raw_dynamic_bound, epsilon)
            else:
                effective_bound = max(raw_dynamic_bound, epsilon * scale, min_bound_floor)
        else:
            effective_bound = max(raw_dynamic_bound, min_bound_floor)

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

            # Subnormal underflow guard for floating point hardware:
            # When matrices have magnitudes near or below FP32 subnormal underflow thresholds (e.g. 1e-25 x 1e-25 = 1e-50),
            # standard hardware GEMM legitimately flushes products to 0.0 (below IEEE 754 float32 min subnormal ~1.4e-45).
            # When promoted to float64, abr is non-zero (~1e-49), creating a relative difference of 1.0 against c=0.
            # When diff is within subnormal underflow range (< 1e-30) and scale or denom indicates underflow,
            # this is genuine hardware float flushing and must not trigger false fraud alarms.
            underflow_threshold = 1e-30 if norm_dtype in ("fp32", "float32", "fp16", "float16", "bf16", "bfloat16") else 1e-100
            is_underflow_regime = scale < underflow_threshold or denom < underflow_threshold

            trial_failed = False
            if scale == 0.0 or norm_a == 0.0 or norm_b == 0.0:
                # When A=0 or B=0, theoretical product is identically 0.
                # Evaluates strictly against absolute tolerance floor without relative normalization.
                if diff > DEFAULT_RELATIVE_TOLERANCE_FLOOR:
                    trial_failed = True
            elif is_underflow_regime and diff < underflow_threshold:
                # Legitimate hardware subnormal underflow to zero; not fraud
                trial_failed = False
            elif denom > 0.0:
                rel_diff = diff / denom
                allowed_rel = max(gamma_k, epsilon if epsilon is not None else 0.0, DEFAULT_RELATIVE_TOLERANCE_FLOOR)
                if diff > effective_bound or rel_diff > allowed_rel:
                    trial_failed = True
            else:
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

    def verify_step(
        self,
        A: np.ndarray,
        v_t: np.ndarray,
        v_next: np.ndarray,
        dtype: Optional[str] = None,
        epsilon: Optional[float] = None,
        **kwargs: Any,
    ) -> IterativeVerificationResult:
        """Instance method shortcut for single-step iterative verification."""
        eff_eps = epsilon if epsilon is not None else getattr(self, "baseline_tolerance", None)
        eff_rel = kwargs.get("relative_discrepancy_threshold", getattr(self, "relative_threshold", None))
        return verify_iterative_step(
            A=A,
            v_t=v_t,
            v_next=v_next,
            dtype=dtype,
            epsilon=eff_eps,
            relative_discrepancy_threshold=eff_rel,
            **kwargs,
        )

    @classmethod
    def verify_iterative_step(
        cls,
        A: np.ndarray,
        v_t: np.ndarray,
        v_next: np.ndarray,
        dtype: Optional[str] = None,
        epsilon: Optional[float] = None,
        **kwargs: Any,
    ) -> IterativeVerificationResult:
        """Classmethod entrypoint for single-step iterative verification."""
        return verify_iterative_step(A=A, v_t=v_t, v_next=v_next, dtype=dtype, epsilon=epsilon, **kwargs)


def verify_iterative_step(
    A: np.ndarray,
    v_t: np.ndarray,
    v_next: np.ndarray,
    dtype: Optional[str] = "fp32",
    epsilon: Optional[float] = None,
    epsilon_tolerance: Optional[float] = None,
    relative_discrepancy_threshold: Optional[float] = None,
    failure_proof_dict: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> IterativeVerificationResult:
    """
    Verifies a single-step iterative state transition v_{t+1} == A * v_t
    using Higham floating-point forward error bounds.

    Returns:
        IterativeVerificationResult(is_valid, discrepancy, effective_bound, failure_proof).
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise MatrixDimensionError(
            f"Matrix A must be a 2D numpy array, got {getattr(A, 'ndim', type(A))}D",
            shape_a=getattr(A, "shape", None),
        )

    if not isinstance(v_t, np.ndarray) or v_t.ndim not in (1, 2):
        raise MatrixDimensionError(
            f"State vector v_t must be a 1D or 2D column vector, got {getattr(v_t, 'ndim', type(v_t))}D",
        )

    if not isinstance(v_next, np.ndarray) or v_next.ndim not in (1, 2):
        raise MatrixDimensionError(
            f"Result vector v_next must be a 1D or 2D column vector, got {getattr(v_next, 'ndim', type(v_next))}D",
        )

    m, k = A.shape
    v_t_arr = np.asarray(v_t)
    v_next_arr = np.asarray(v_next)
    vt_flat = v_t_arr.flatten().astype(np.float64)
    vnext_flat = v_next_arr.flatten().astype(np.float64)

    if vt_flat.size != k:
        raise MatrixDimensionError(
            f"Inner dimension mismatch: Matrix A columns ({k}) != vector v_t length ({vt_flat.size})",
            shape_a=A.shape,
            expected_inner_dim=k,
        )

    if vnext_flat.size != m:
        raise MatrixDimensionError(
            f"Result dimension mismatch: Matrix A rows ({m}) != vector v_next length ({vnext_flat.size})",
            shape_a=A.shape,
            expected_inner_dim=m,
        )

    norm_dt = normalize_dtype(dtype, fallback_tensor=A)

    # Check Finiteness (NaN / Inf Detection)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(vt_flat)) and np.all(np.isfinite(vnext_flat))):
        proof = {
            "vector_vt": vt_flat.tolist() if np.all(np.isfinite(vt_flat)) else [],
            "vector_av": [],
            "vector_v_next": vnext_flat.tolist() if np.all(np.isfinite(vnext_flat)) else [],
            "observed_diff": float("inf"),
            "dynamic_bound": 1e-6,
            "max_observed_discrepancy": float("inf"),
            "dimension_m": int(m),
            "dimension_k": int(k),
            "dimension_n": int(k),
            "dtype": norm_dt,
            "reason": "Non-finite value (NaN/Inf) detected in matrix A or vectors",
        }
        if failure_proof_dict is not None and isinstance(failure_proof_dict, dict):
            failure_proof_dict.update(proof)
        return IterativeVerificationResult(False, float("inf"), 1e-6, proof)

    # Compute Higham Dynamic Error Bound for GEMV
    norm_a = matrix_infinity_norm(A)
    norm_vt = vector_infinity_norm(vt_flat)
    scale = norm_a * norm_vt

    u = get_unit_roundoff(norm_dt)
    ku = float(k) * u
    gamma_k = (ku / (1.0 - ku)) if ku < 0.999 else (ku * 1000.0)

    raw_dynamic_bound = gamma_k * scale
    min_bound_floor = u * scale if scale > 0 else u

    eff_eps = epsilon_tolerance if epsilon_tolerance is not None else epsilon
    if eff_eps is not None:
        effective_bound = max(raw_dynamic_bound, float(eff_eps) if scale >= 1.0 else float(eff_eps) * scale, min_bound_floor)
    else:
        effective_bound = max(raw_dynamic_bound, min_bound_floor)

    # Reference Verification in float64
    a_f64 = A.astype(np.float64)
    av_ref = np.matmul(a_f64, vt_flat)
    diff_vec = np.abs(av_ref - vnext_flat)
    discrepancy = float(np.max(diff_vec))

    if np.isnan(discrepancy) or np.isinf(discrepancy):
        proof = {
            "vector_vt": vt_flat.tolist(),
            "vector_av": av_ref.tolist(),
            "vector_v_next": vnext_flat.tolist(),
            "observed_diff": float("inf"),
            "dynamic_bound": float(effective_bound),
            "max_observed_discrepancy": float("inf"),
            "dimension_m": int(m),
            "dimension_k": int(k),
            "dimension_n": int(k),
            "dtype": norm_dt,
            "reason": "Non-finite difference encountered during verification",
        }
        if failure_proof_dict is not None and isinstance(failure_proof_dict, dict):
            failure_proof_dict.update(proof)
        return IterativeVerificationResult(False, float("inf"), float(effective_bound), proof)

    # Underflow & Relative Discrepancy Defense
    av_norm = vector_infinity_norm(av_ref)
    vnext_norm = vector_infinity_norm(vnext_flat)
    denom = max(scale, av_norm, vnext_norm)

    underflow_threshold = 1e-30 if norm_dt in ("fp32", "float32", "fp16", "float16", "bf16", "bfloat16") else 1e-100
    is_underflow_regime = scale < underflow_threshold or denom < underflow_threshold

    is_fraud = False
    failure_reason = ""
    if scale == 0.0 or norm_a == 0.0 or norm_vt == 0.0:
        if discrepancy > DEFAULT_RELATIVE_TOLERANCE_FLOOR:
            is_fraud = True
            failure_reason = f"Observed discrepancy {discrepancy:.6e} exceeded dynamic Higham bound {effective_bound:.6e}"
    elif is_underflow_regime and discrepancy < underflow_threshold:
        is_fraud = False
    elif discrepancy <= effective_bound:
        is_fraud = False
    else:
        is_fraud = True
        failure_reason = f"Observed discrepancy {discrepancy:.6e} exceeded dynamic Higham bound {effective_bound:.6e}"

    out_norm = max(av_norm, vnext_norm)
    out_denom = max(out_norm, min_bound_floor, underflow_threshold)
    rel_diff = discrepancy / out_denom

    if relative_discrepancy_threshold is not None:
        allowed_rel = float(relative_discrepancy_threshold)
    else:
        allowed_rel = min(max(gamma_k, DEFAULT_RELATIVE_TOLERANCE_FLOOR), MAX_RELATIVE_DISCREPANCY_CEILING)

    if not is_fraud and out_denom > underflow_threshold:
        if rel_diff > allowed_rel:
            is_fraud = True
            failure_reason = f"Observed relative discrepancy {rel_diff:.6e} exceeded tolerance {allowed_rel:.6e} (discrepancy {discrepancy:.6e}, output norm {out_denom:.6e})"

    if is_fraud:
        proof = {
            "vector_vt": vt_flat.tolist(),
            "vector_av": av_ref.tolist(),
            "vector_v_next": vnext_flat.tolist(),
            "observed_diff": float(discrepancy),
            "dynamic_bound": float(effective_bound),
            "max_observed_discrepancy": float(discrepancy),
            "dimension_m": int(m),
            "dimension_k": int(k),
            "dimension_n": int(k),
            "dtype": norm_dt,
            "vector_vt_norm": norm_vt,
            "matrix_a_norm": norm_a,
            "reason": (
                failure_reason
                if failure_reason
                else f"Observed discrepancy {discrepancy:.6e} exceeded dynamic Higham bound {effective_bound:.6e}"
            ),
        }
        if failure_proof_dict is not None and isinstance(failure_proof_dict, dict):
            failure_proof_dict.update(proof)
        return IterativeVerificationResult(False, float(discrepancy), float(effective_bound), proof)

    return IterativeVerificationResult(True, float(discrepancy), float(effective_bound), {})


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
