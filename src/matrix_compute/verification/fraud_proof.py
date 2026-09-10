"""
Authoritative Freivalds Fraud Proof Builder and Independent Verifier.
Implements fraud proof construction, Higham error bound invariant validation,
cryptographic Secp256k1 attestation binding, and dispute payload verification.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from matrix_contracts.schemas.fraud import FreivaldsFraudProof
from matrix_compute.crypto.keys import (
    privkey_to_pubkey,
    validate_private_key_hex,
    validate_public_key_hex,
)
from matrix_compute.crypto.signer import (
    ClientSigner,
    format_tile_result_payload,
    hash_tensor,
    sha256_digest,
    sign_message,
    verify_signature,
)
from matrix_compute.exceptions import (
    FraudDetectedError,
    MatrixDimensionError,
    VerificationFailedError,
    WorkerSignatureError,
)
from matrix_compute.tiling.precision import calculate_higham_bound, get_unit_roundoff, normalize_dtype

logger = logging.getLogger(__name__)


def _compute_higham_bound(k: int, norm_a: float, norm_b: float, dtype: str = "fp32") -> float:
    """Computes dynamic scale-aware Higham bound gamma_k * ||A||_inf * ||B||_inf."""
    u = get_unit_roundoff(dtype)
    ku = float(k) * u
    gamma_k = (ku / (1.0 - ku)) if ku < 0.999 else (ku * 1000.0)
    return max(gamma_k * float(norm_a) * float(norm_b), 1e-6)


def _find_witness_vector(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    epsilon_tolerance: float,
    max_trials: int = 50,
) -> Tuple[List[int], List[float], List[float], List[float], float]:
    """Finds a binary vector r proving max(|ABr - Cr|) > epsilon_tolerance."""
    m, k = A.shape
    _, n = B.shape
    a_f64 = A.astype(np.float64)
    b_f64 = B.astype(np.float64)
    c_f64 = C.astype(np.float64)

    best_disc = -1.0
    best_tuple: Optional[Tuple[List[int], List[float], List[float], List[float], float]] = None

    # Try standard basis vectors first
    for j in range(n):
        r = np.zeros((n, 1), dtype=np.float64)
        r[j, 0] = 1.0
        br = np.matmul(b_f64, r)
        abr = np.matmul(a_f64, br)
        cr = np.matmul(c_f64, r)
        disc = float(np.max(np.abs(abr - cr)))
        if disc > best_disc:
            best_disc = disc
            best_tuple = (
                r.flatten().astype(int).tolist(),
                br.flatten().astype(float).tolist(),
                abr.flatten().astype(float).tolist(),
                cr.flatten().astype(float).tolist(),
                disc,
            )
        if disc > epsilon_tolerance:
            return best_tuple

    # Try random binary vectors
    for _ in range(max_trials):
        r = np.random.randint(0, 2, size=(n, 1)).astype(np.float64)
        if not np.any(r):
            r[np.random.randint(0, n), 0] = 1.0
        br = np.matmul(b_f64, r)
        abr = np.matmul(a_f64, br)
        cr = np.matmul(c_f64, r)
        disc = float(np.max(np.abs(abr - cr)))
        if disc > best_disc:
            best_disc = disc
            best_tuple = (
                r.flatten().astype(int).tolist(),
                br.flatten().astype(float).tolist(),
                abr.flatten().astype(float).tolist(),
                cr.flatten().astype(float).tolist(),
                disc,
            )
        if disc > epsilon_tolerance:
            return best_tuple

    if best_tuple is not None and best_disc > epsilon_tolerance:
        return best_tuple

    raise ValueError(
        f"Observed discrepancy ({best_disc}) does not exceed epsilon tolerance ({epsilon_tolerance}). "
        "Result cannot be proven fraudulent."
    )


class FraudProofBuilder:
    """Authoritative builder for FreivaldsFraudProof dispute payloads."""

    @classmethod
    def build_fraud_proof(
        cls,
        task_id: str,
        worker_pubkey: str,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        proof_data: Optional[Dict[str, Any]] = None,
        worker_sig: str = "",
        consumer_privkey: Optional[str] = None,
        consumer_pubkey: Optional[str] = None,
        dtype: Optional[str] = None,
        epsilon_tolerance: Optional[float] = None,
        verify_worker_sig: bool = True,
    ) -> FreivaldsFraudProof:
        """Constructs an authoritative FreivaldsFraudProof Pydantic payload."""
        if not task_id or not str(task_id).strip():
            raise ValueError("task_id must be a non-empty string")

        worker_pk = str(worker_pubkey).strip()

        # Resolve consumer public key
        if consumer_pubkey is not None and str(consumer_pubkey).strip():
            c_pub = str(consumer_pubkey).strip()
        elif consumer_privkey is not None and str(consumer_privkey).strip():
            c_priv = str(consumer_privkey).strip()
            if not validate_private_key_hex(c_priv):
                raise ValueError("Invalid consumer_privkey format or scalar range")
            c_pub = privkey_to_pubkey(c_priv, compressed=True)
        else:
            raise ValueError("Either consumer_pubkey or consumer_privkey must be provided")

        # Validate input matrix shapes
        if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray) or not isinstance(C, np.ndarray):
            raise MatrixDimensionError("Matrices A, B, and C must be numpy ndarrays")

        if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
            raise MatrixDimensionError("Matrices A, B, and C must be 2D arrays")
        m, k = A.shape
        k_b, n = B.shape
        if k != k_b:
            raise MatrixDimensionError(
                f"Dimension mismatch: A columns ({k}) != B rows ({k_b})",
                shape_a=A.shape,
                shape_b=B.shape,
                expected_inner_dim=k,
            )
        if C.shape != (m, n):
            raise MatrixDimensionError(
                f"Result matrix shape mismatch: C {C.shape} != expected ({m}, {n})",
                shape_a=A.shape,
                shape_b=B.shape,
            )

        # Datatype resolution
        if dtype is None:
            if A.dtype == np.float16 or C.dtype == np.float16:
                dtype_clean = "fp16"
            else:
                dtype_clean = "fp32"
        else:
            dtype_clean = normalize_dtype(dtype)

        # Dynamic Higham bound
        if epsilon_tolerance is None or epsilon_tolerance <= 0.0:
            if proof_data and (proof_data.get("dynamic_bound") or proof_data.get("epsilon_tolerance")):
                epsilon_tolerance = float(proof_data.get("dynamic_bound") or proof_data.get("epsilon_tolerance"))
            else:
                norm_a = float(np.max(np.sum(np.abs(A.astype(np.float64)), axis=1))) if m > 0 else 0.0
                norm_b = float(np.max(np.sum(np.abs(B.astype(np.float64)), axis=1))) if k > 0 else 0.0
                epsilon_tolerance = _compute_higham_bound(k, norm_a, norm_b, dtype=dtype_clean)

        # Hashes
        hash_a = hash_tensor(np.ascontiguousarray(A))
        hash_b = hash_tensor(np.ascontiguousarray(B))
        hash_c = hash_tensor(np.ascontiguousarray(C))

        # Witness vectors
        r_list, br_list, abr_list, cr_list = None, None, None, None
        max_disc = 0.0

        if proof_data and ("random_vector_r" in proof_data or "r" in proof_data):
            raw_r = proof_data.get("random_vector_r") if "random_vector_r" in proof_data else proof_data.get("r")
            if hasattr(raw_r, "flatten"):
                r_list = [int(x) for x in raw_r.flatten().tolist()]
            else:
                r_list = [int(x) for x in np.array(raw_r).flatten().tolist()]

            if len(r_list) != n:
                raise ValueError(f"random_vector_r length ({len(r_list)}) must match dimension_n ({n})")

            # Check for other vectors
            for key, attr in [("vector_br", "br"), ("vector_abr", "abr"), ("vector_cr", "cr")]:
                val = proof_data.get(key) if key in proof_data else proof_data.get(attr)
                if val is not None:
                    if hasattr(val, "flatten"):
                        lst = val.flatten().tolist()
                    else:
                        lst = np.array(val).flatten().tolist()
                    if key == "vector_br":
                        br_list = [float(x) for x in lst]
                    elif key == "vector_abr":
                        abr_list = [float(x) for x in lst]
                    elif key == "vector_cr":
                        cr_list = [float(x) for x in lst]

            # Recompute any missing intermediate vector
            r_arr = np.array(r_list, dtype=np.float64).reshape((n, 1))
            if br_list is None:
                br_list = np.matmul(B.astype(np.float64), r_arr).flatten().astype(float).tolist()
            if abr_list is None:
                br_arr = np.array(br_list, dtype=np.float64).reshape((k, 1))
                abr_list = np.matmul(A.astype(np.float64), br_arr).flatten().astype(float).tolist()
            if cr_list is None:
                cr_list = np.matmul(C.astype(np.float64), r_arr).flatten().astype(float).tolist()

            max_disc = float(np.max(np.abs(np.array(abr_list) - np.array(cr_list))))
            if max_disc <= epsilon_tolerance:
                raise ValueError(
                    f"Provided proof_data discrepancy ({max_disc}) does not exceed epsilon_tolerance ({epsilon_tolerance})"
                )
        else:
            r_list, br_list, abr_list, cr_list, max_disc = _find_witness_vector(A, B, C, epsilon_tolerance)

        # Worker signature check
        if verify_worker_sig:
            if not worker_sig:
                raise WorkerSignatureError(
                    f"Missing worker signature for task '{task_id}'",
                    task_id=task_id,
                    worker_pubkey=worker_pk,
                )
            if not validate_public_key_hex(worker_pk):
                raise WorkerSignatureError(
                    f"Invalid worker public key format '{worker_pk}'",
                    task_id=task_id,
                    worker_pubkey=worker_pk,
                )
            payload = format_tile_result_payload(task_id, hash_c)
            if not verify_signature(worker_pk, payload, worker_sig):
                raise WorkerSignatureError(
                    f"Invalid worker cryptographic signature on invalid result for task '{task_id}'",
                    task_id=task_id,
                    worker_pubkey=worker_pk,
                    signature=worker_sig,
                )

        return FreivaldsFraudProof(
            task_id=task_id,
            worker_pubkey=worker_pk,
            dimension_m=m,
            dimension_k=k,
            dimension_n=n,
            dtype=dtype_clean,
            matrix_a_hash=hash_a,
            matrix_b_hash=hash_b,
            invalid_result_c_hash=hash_c,
            random_vector_r=r_list,
            vector_br=br_list,
            vector_abr=abr_list,
            vector_cr=cr_list,
            epsilon_tolerance=float(epsilon_tolerance),
            max_observed_discrepancy=float(max_disc),
            worker_signature_on_result=str(worker_sig),
            consumer_pubkey=c_pub,
        )


def verify_fraud_proof(
    proof: FreivaldsFraudProof,
    matrix_a: Optional[np.ndarray] = None,
    matrix_b: Optional[np.ndarray] = None,
    matrix_c: Optional[np.ndarray] = None,
    verify_signatures: bool = True,
) -> bool:
    """Independently verifies an authoritative FreivaldsFraudProof dispute payload."""
    if not isinstance(proof, FreivaldsFraudProof):
        return False

    # Metadata & dimension sanity
    m = proof.dimension_m
    k = proof.dimension_k
    n = proof.dimension_n
    if m <= 0 or k <= 0 or n <= 0:
        return False

    # Tolerances sanity
    if proof.epsilon_tolerance < 0.0 or not math.isfinite(proof.epsilon_tolerance):
        return False
    if proof.max_observed_discrepancy <= proof.epsilon_tolerance or not math.isfinite(proof.max_observed_discrepancy):
        return False

    # Vector dimensions
    if len(proof.random_vector_r) != n:
        return False
    if len(proof.vector_br) != k:
        return False
    if len(proof.vector_abr) != m:
        return False
    if len(proof.vector_cr) != m:
        return False

    # Binary vector r validation
    if not any(x == 1 for x in proof.random_vector_r):
        return False  # All-zero vector cannot prove discrepancy
    for x in proof.random_vector_r:
        if x not in (0, 1):
            return False

    # Floating point finiteness
    for vec in (proof.vector_br, proof.vector_abr, proof.vector_cr):
        for val in vec:
            if not math.isfinite(val):
                return False

    # Mathematical violation verification
    abr = np.array(proof.vector_abr, dtype=np.float64)
    cr = np.array(proof.vector_cr, dtype=np.float64)
    diff = float(np.max(np.abs(abr - cr)))

    if diff <= proof.epsilon_tolerance:
        return False
    if not math.isclose(diff, proof.max_observed_discrepancy, rel_tol=1e-5, abs_tol=1e-7):
        return False

    # Cryptographic attestation verification
    if verify_signatures:
        if not validate_public_key_hex(proof.worker_pubkey):
            return False
        if not validate_public_key_hex(proof.consumer_pubkey):
            return False
        payload = format_tile_result_payload(proof.task_id, proof.invalid_result_c_hash)
        if not verify_signature(proof.worker_pubkey, payload, proof.worker_signature_on_result):
            return False

    # Optional full witness matrix cross-verification
    if matrix_a is not None:
        if matrix_a.shape != (m, k) or hash_tensor(np.ascontiguousarray(matrix_a)) != proof.matrix_a_hash:
            return False
        br_arr = np.array(proof.vector_br, dtype=np.float64).reshape((k, 1))
        calc_abr = np.matmul(matrix_a.astype(np.float64), br_arr).flatten()
        if np.max(np.abs(calc_abr - abr)) > max(proof.epsilon_tolerance, 1e-4):
            return False

    if matrix_b is not None:
        if matrix_b.shape != (k, n) or hash_tensor(np.ascontiguousarray(matrix_b)) != proof.matrix_b_hash:
            return False
        r_arr = np.array(proof.random_vector_r, dtype=np.float64).reshape((n, 1))
        calc_br = np.matmul(matrix_b.astype(np.float64), r_arr).flatten()
        if np.max(np.abs(calc_br - np.array(proof.vector_br, dtype=np.float64))) > max(proof.epsilon_tolerance, 1e-4):
            return False

    if matrix_c is not None:
        if matrix_c.shape != (m, n) or hash_tensor(np.ascontiguousarray(matrix_c)) != proof.invalid_result_c_hash:
            return False
        r_arr = np.array(proof.random_vector_r, dtype=np.float64).reshape((n, 1))
        calc_cr = np.matmul(matrix_c.astype(np.float64), r_arr).flatten()
        if np.max(np.abs(calc_cr - cr)) > max(proof.epsilon_tolerance, 1e-4):
            return False

    return True


def create_fraud_dispute(
    task_id: str,
    worker_pubkey: str,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    proof_data: Optional[Dict[str, Any]],
    worker_sig: str,
    consumer_signer: ClientSigner,
    dtype: str = "fp32",
) -> FreivaldsFraudProof:
    """Convenience helper constructing an authoritative fraud proof using ClientSigner."""
    return FraudProofBuilder.build_fraud_proof(
        task_id=task_id,
        worker_pubkey=worker_pubkey,
        A=A,
        B=B,
        C=C,
        proof_data=proof_data,
        worker_sig=worker_sig,
        consumer_pubkey=consumer_signer.get_pubkey_hex(),
        dtype=dtype,
        verify_worker_sig=True,
    )


__all__ = [
    "FraudProofBuilder",
    "verify_fraud_proof",
    "create_fraud_dispute",
]
