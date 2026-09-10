"""
Adversarial Stress Test Suite for Freivalds Verification Engine & Fraud Proof Builder.
Targeting: Milestone 2 (M2) in matrix-compute-sdk.

Authored by teamwork_preview_challenger_sdk_m2_2 (Freivalds & Fraud Proof Challenger).

Challenge Dimensions:
1. Forged and Tampered Result Verification:
   - Single-element perturbations (+1e-3, +1.0)
   - Single row corruptions (+0.5 bias)
   - Sign flips (C = -A x B)
   - Transposed outputs (C = (A x B)^T)
   - Zero-matrix returns (C = 0)
   - 100+ trials per attack vector confirming 100% detection rate.
2. Dynamic Scale Extremes & Precision Hierarchy:
   - Tiny magnitude matrices (||A||, ||B|| ~ 1e-4, 1e-5) with zero-work fraud defense.
   - Huge magnitude matrices (||A||, ||B|| ~ 1e4) confirming zero false positives.
   - Asymmetric scale extremes (||A|| ~ 1e4, ||B|| ~ 1e-4).
   - FP32, FP16, and BF16 precision adaptations.
3. Large-Scale Matrix Stability:
   - Matrix dimensions N = 64, 128, 256, 512.
   - Zero false negatives on honest GEMM products.
   - Large-scale perturbation detection.
4. Fraud Proof Builder Authoritative Invariants:
   - Authoritative FreivaldsFraudProof generation with vectors (r, Br, ABr, Cr).
   - Invariant check: max |ABr - Cr| > epsilon_tolerance.
   - Honest computation rejection: building proof on valid GEMM raises ValueError.
   - Automatic witness vector search when proof_data is None.
5. Independent Dispute Payload Verifier (verify_fraud_proof):
   - Tampered witness vectors (r, Br, ABr, Cr).
   - Non-binary or all-zero vector r.
   - Falsified discrepancy metadata.
   - Dimension and vector length mismatches.
   - Forged, replayed, or corrupted worker signatures.
   - Witness matrix cross-verification and hash integrity.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from matrix_contracts.schemas.fraud import FreivaldsFraudProof
from matrix_compute.config import ClientConfig
from matrix_compute.crypto.keys import generate_keypair
from matrix_compute.crypto.signer import (
    ClientSigner,
    format_tile_result_payload,
    hash_tensor,
    sign_message,
)
from matrix_compute.exceptions import MatrixDimensionError, WorkerSignatureError
from matrix_compute.tiling.precision import truncate_to_bf16
from matrix_compute.verification.fraud_proof import (
    FraudProofBuilder,
    create_fraud_dispute,
    verify_fraud_proof,
)
from matrix_compute.verification.freivalds import (
    DEFAULT_FREIVALDS_ROUNDS,
    DEFAULT_RELATIVE_TOLERANCE_FLOOR,
    UNIT_ROUNDOFFS,
    FreivaldsVerifier,
    freivalds_verify,
    generate_random_binary_vector,
    higham_bound,
    matrix_infinity_norm,
    normalize_dtype,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def crypto_keys():
    """Generates distinct worker and consumer Secp256k1 keypairs."""
    w_priv, w_pub = generate_keypair()
    c_priv, c_pub = generate_keypair()
    return {
        "w_priv": w_priv,
        "w_pub": w_pub,
        "c_priv": c_priv,
        "c_pub": c_pub,
    }


# =============================================================================
# Suite 1: Forged & Tampered Products (100+ Trials Rejection Rate)
# =============================================================================

class TestAdversarialFreivaldsPerturbations:
    """
    Exhaustively tests detection of corrupted result matrices across 100+ trials per attack vector.
    Confirms 100% rejection rate with randomized binary Freivalds vectors.
    """

    def test_single_element_perturbation_1e3_exhaustive_100_trials(self):
        """
        Attack: Worker injects subtle +1e-3 perturbation at a random single coordinate (i, j).
        Confidence: k=15 rounds ensures error probability < 2^-15 (~3.05e-5).
        Expected: 100/100 trials rejected (100% detection rate).
        """
        rng = np.random.RandomState(1001)
        trials = 100
        rejected_count = 0

        for trial in range(trials):
            m, k, n = 16, 16, 16
            a = rng.randn(m, k).astype(np.float32)
            b = rng.randn(k, n).astype(np.float32)
            c = np.matmul(a, b)

            # Perturb a random single element by +1e-3
            i_corrupt = rng.randint(0, m)
            j_corrupt = rng.randint(0, n)
            c[i_corrupt, j_corrupt] += 1e-3

            passed, disc, bound, proof = freivalds_verify(
                a, b, c, dtype="fp32", num_trials=15
            )
            if not passed:
                rejected_count += 1
                assert disc > bound or disc > 0.0
                assert "random_vector_r" in proof
                assert proof["dimension_m"] == m
                assert proof["dimension_n"] == n

        assert rejected_count == trials, f"Failed: only {rejected_count}/{trials} rejected!"

    def test_single_element_perturbation_unit_exhaustive_100_trials(self):
        """
        Attack: Worker injects macro +1.0 perturbation at a random coordinate (i, j).
        Expected: 100/100 trials rejected.
        """
        rng = np.random.RandomState(1002)
        trials = 100
        rejected_count = 0

        for trial in range(trials):
            m, k, n = 20, 20, 20
            a = rng.randn(m, k).astype(np.float32)
            b = rng.randn(k, n).astype(np.float32)
            c = np.matmul(a, b)

            i_corrupt = rng.randint(0, m)
            j_corrupt = rng.randint(0, n)
            c[i_corrupt, j_corrupt] += 1.0

            passed, disc, bound, proof = freivalds_verify(
                a, b, c, dtype="fp32", num_trials=15
            )
            if not passed:
                rejected_count += 1
                assert disc > bound

        assert rejected_count == trials

    def test_single_row_corruption_exhaustive_100_trials(self):
        """
        Attack: Worker injects +0.5 bias across an entire row.
        Expected: 100/100 trials rejected.
        """
        rng = np.random.RandomState(1003)
        trials = 100
        rejected_count = 0

        for trial in range(trials):
            m, k, n = 16, 24, 16
            a = rng.randn(m, k).astype(np.float32)
            b = rng.randn(k, n).astype(np.float32)
            c = np.matmul(a, b)

            row_corrupt = rng.randint(0, m)
            c[row_corrupt, :] += 0.5

            passed, disc, bound, proof = freivalds_verify(
                a, b, c, dtype="fp32", num_trials=15
            )
            if not passed:
                rejected_count += 1
                assert disc > bound

        assert rejected_count == trials

    def test_sign_flip_corruption_exhaustive_100_trials(self):
        """
        Attack: Worker negates the result matrix C = -(A x B).
        Expected: 100/100 trials rejected.
        """
        rng = np.random.RandomState(1004)
        trials = 100
        rejected_count = 0

        for trial in range(trials):
            m, k, n = 16, 16, 16
            a = rng.randn(m, k).astype(np.float32)
            b = rng.randn(k, n).astype(np.float32)
            c = -np.matmul(a, b)  # Negation

            passed, disc, bound, proof = freivalds_verify(
                a, b, c, dtype="fp32", num_trials=10
            )
            if not passed:
                rejected_count += 1
                assert disc > bound

        assert rejected_count == trials

    def test_transposed_output_corruption_exhaustive_100_trials(self):
        """
        Attack: Worker returns transposed matrix C = (A x B)^T for square asymmetric GEMM.
        Expected: 100/100 trials rejected.
        """
        rng = np.random.RandomState(1005)
        trials = 100
        rejected_count = 0

        for trial in range(trials):
            n = 16
            a = rng.randn(n, n).astype(np.float32)
            b = rng.randn(n, n).astype(np.float32)
            c = np.matmul(a, b).T  # Transposed

            passed, disc, bound, proof = freivalds_verify(
                a, b, c, dtype="fp32", num_trials=15
            )
            if not passed:
                rejected_count += 1
                assert disc > bound

        assert rejected_count == trials

    def test_zero_matrix_return_corruption_exhaustive_100_trials(self):
        """
        Attack: Lazy worker performs zero work and returns C = 0 for normal non-zero inputs.
        Expected: 100/100 trials rejected.
        """
        rng = np.random.RandomState(1006)
        trials = 100
        rejected_count = 0

        for trial in range(trials):
            m, k, n = 16, 16, 16
            a = rng.randn(m, k).astype(np.float32)
            b = rng.randn(k, n).astype(np.float32)
            c = np.zeros((m, n), dtype=np.float32)  # Zero work

            passed, disc, bound, proof = freivalds_verify(
                a, b, c, dtype="fp32", num_trials=10
            )
            if not passed:
                rejected_count += 1
                assert disc > bound

        assert rejected_count == trials


# =============================================================================
# Suite 2: Scale Extremes & Zero-Work Fraud Defenses
# =============================================================================

class TestAdversarialScaleExtremes:
    """
    Tests dynamic Higham error bound behavior across extreme dynamic ranges:
    Tiny magnitude (1e-4, 1e-5), huge magnitude (1e4), asymmetric scales, and zero-work fraud.
    """

    def test_tiny_magnitude_honest_no_false_positives_50_trials(self):
        """
        Inputs ||A||, ||B|| ~ 1e-4.
        Honest GEMM product C must pass Freivalds verification with 0 false positives.
        """
        rng = np.random.RandomState(2001)
        for _ in range(50):
            m, k, n = 16, 16, 16
            a = (rng.randn(m, k) * 1e-4).astype(np.float32)
            b = (rng.randn(k, n) * 1e-4).astype(np.float32)
            c = np.matmul(a, b)

            passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
            assert passed is True, f"Honest tiny input failed: disc={disc}, bound={bound}"
            assert disc <= bound
            assert len(proof) == 0

    def test_tiny_magnitude_zero_work_caught_100_trials(self):
        """
        Inputs ||A||, ||B|| ~ 1e-4. Lazy worker returns C = 0.
        Although absolute discrepancy ~ 1e-7 may be below default absolute bound (1e-6),
        the relative discrepancy defense must trigger and catch 100/100 trials.
        """
        rng = np.random.RandomState(2002)
        trials = 100
        caught = 0

        for _ in range(trials):
            m, k, n = 16, 16, 16
            a = (rng.randn(m, k) * 1e-4).astype(np.float32)
            b = (rng.randn(k, n) * 1e-4).astype(np.float32)
            c_zero = np.zeros((m, n), dtype=np.float32)

            passed, disc, bound, proof = freivalds_verify(
                a, b, c_zero, dtype="fp32", num_trials=10
            )
            if not passed:
                caught += 1
                assert "random_vector_r" in proof

        assert caught == trials, f"Zero-work fraud missed: only {caught}/{trials} caught!"

    def test_ultratiny_magnitude_1e5_zero_work_caught(self):
        """
        Inputs ||A||, ||B|| ~ 1e-5. Lazy worker returns C = 0.
        Confirms relative discrepancy threshold catches ultra-tiny zero-work fraud.
        """
        rng = np.random.RandomState(2003)
        for _ in range(25):
            a = (rng.randn(12, 12) * 1e-5).astype(np.float32)
            b = (rng.randn(12, 12) * 1e-5).astype(np.float32)
            c_zero = np.zeros((12, 12), dtype=np.float32)

            passed, disc, bound, proof = freivalds_verify(a, b, c_zero, dtype="fp32")
            assert passed is False
            assert "random_vector_r" in proof

    def test_huge_magnitude_honest_no_false_positives_50_trials(self):
        """
        Inputs ||A||, ||B|| ~ 1e4.
        Dynamic Higham bound must scale proportionally to gamma_K * ||A|| * ||B|| (~1e5),
        preventing false positive rejection of honest computations.
        """
        rng = np.random.RandomState(2004)
        for _ in range(50):
            m, k, n = 16, 16, 16
            a = (rng.randn(m, k) * 1e4).astype(np.float32)
            b = (rng.randn(k, n) * 1e4).astype(np.float32)
            c = np.matmul(a, b)

            passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
            assert passed is True, f"Honest huge input rejected: disc={disc}, bound={bound}"
            assert bound > 1.0  # Dynamic bound appropriately scaled
            assert disc <= bound

    def test_huge_magnitude_tampered_caught_50_trials(self):
        """
        Inputs ||A||, ||B|| ~ 1e4. Worker injects perturbation > dynamic bound.
        """
        rng = np.random.RandomState(2005)
        for _ in range(50):
            m, k, n = 16, 16, 16
            a = (rng.randn(m, k) * 1e4).astype(np.float32)
            b = (rng.randn(k, n) * 1e4).astype(np.float32)
            c = np.matmul(a, b)

            # Higham bound is ~ 16 * 6e-8 * (16*1e4)^2 ~ 2.5e4
            # Add perturbation of 1e6
            c[0, 0] += 1e6

            passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32", num_trials=10)
            assert passed is False
            assert disc > bound

    def test_asymmetric_scales_honest_and_tampered(self):
        """
        ||A|| ~ 1e4, ||B|| ~ 1e-4 -> scale ||A||*||B|| ~ 1.0.
        Honest passes; perturbation fails.
        """
        rng = np.random.RandomState(2006)
        a = (rng.randn(16, 16) * 1e4).astype(np.float32)
        b = (rng.randn(16, 16) * 1e-4).astype(np.float32)
        c_honest = np.matmul(a, b)

        passed, disc, bound, _ = freivalds_verify(a, b, c_honest, dtype="fp32")
        assert passed is True
        assert disc <= bound

        # Tampered
        c_tampered = c_honest.copy()
        c_tampered[2, 3] += 5.0
        passed_t, disc_t, bound_t, proof_t = freivalds_verify(a, b, c_tampered, dtype="fp32", num_trials=10)
        assert passed_t is False
        assert disc_t > bound_t

    def test_precision_hierarchy_bounds_honest_fp16_bf16(self):
        """
        Tests honest computations across FP16 and BF16.
        Confirms bound hierarchy: bound(BF16) > bound(FP16) > bound(FP32).
        """
        rng = np.random.RandomState(2007)
        a_f32 = rng.randn(16, 16).astype(np.float32)
        b_f32 = rng.randn(16, 16).astype(np.float32)

        # FP32
        c_f32 = np.matmul(a_f32, b_f32)
        p32, d32, b32, _ = freivalds_verify(a_f32, b_f32, c_f32, dtype="fp32")
        assert p32 is True

        # FP16
        a_f16 = a_f32.astype(np.float16)
        b_f16 = b_f32.astype(np.float16)
        c_f16 = np.matmul(a_f16.astype(np.float32), b_f16.astype(np.float32)).astype(np.float16)
        p16, d16, b16, _ = freivalds_verify(a_f16, b_f16, c_f16, dtype="fp16")
        assert p16 is True

        # BF16
        a_bf16 = truncate_to_bf16(a_f32)
        b_bf16 = truncate_to_bf16(b_f32)
        c_bf16 = truncate_to_bf16(np.matmul(a_bf16, b_bf16))
        pbf, dbf, bbf, _ = freivalds_verify(a_bf16, b_bf16, c_bf16, dtype="bf16")
        assert pbf is True

        assert bbf > b16 > b32


# =============================================================================
# Suite 3: Large-Scale Matrices (N = 64, 128, 256, 512)
# =============================================================================

class TestAdversarialLargeMatrices:
    """
    Tests randomized Freivalds verification scalability across large matrix sizes:
    N = 64, 128, 256, 512.
    Confirms zero false negatives on honest GEMM and robust tampering rejection.
    """

    def test_honest_large_matrix_n64(self):
        """Honest N=64 FP32 GEMM passes with empty proof."""
        rng = np.random.RandomState(3001)
        a = rng.randn(64, 64).astype(np.float32)
        b = rng.randn(64, 64).astype(np.float32)
        c = np.matmul(a, b)

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound
        assert len(proof) == 0

    def test_honest_large_matrix_n128(self):
        """Honest N=128 FP32 GEMM passes without false negative."""
        rng = np.random.RandomState(3002)
        a = rng.randn(128, 128).astype(np.float32)
        b = rng.randn(128, 128).astype(np.float32)
        c = np.matmul(a, b)

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound
        assert len(proof) == 0

    def test_honest_large_matrix_n256(self):
        """Honest N=256 FP32 GEMM passes with dynamic Higham bound."""
        rng = np.random.RandomState(3003)
        a = rng.randn(256, 256).astype(np.float32)
        b = rng.randn(256, 256).astype(np.float32)
        c = np.matmul(a, b)

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound
        assert len(proof) == 0

    def test_honest_large_matrix_n512(self):
        """Honest N=512 FP32 GEMM passes without numerical instability."""
        rng = np.random.RandomState(3004)
        a = rng.randn(512, 512).astype(np.float32)
        b = rng.randn(512, 512).astype(np.float32)
        c = np.matmul(a, b)

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound
        assert len(proof) == 0

    def test_large_matrix_n256_single_element_tampering_caught(self):
        """N=256 matrix with single element perturbed by +1.0 is caught."""
        rng = np.random.RandomState(3005)
        a = rng.randn(256, 256).astype(np.float32)
        b = rng.randn(256, 256).astype(np.float32)
        c = np.matmul(a, b)
        c[120, 210] += 1.0

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32", num_trials=15)
        assert passed is False
        assert disc > bound
        assert len(proof["random_vector_r"]) == 256
        assert len(proof["vector_br"]) == 256
        assert len(proof["vector_abr"]) == 256
        assert len(proof["vector_cr"]) == 256

    def test_large_rectangular_matrices(self):
        """Rectangular dimensions: A is (64, 256), B is (256, 128)."""
        rng = np.random.RandomState(3006)
        a = rng.randn(64, 256).astype(np.float32)
        b = rng.randn(256, 128).astype(np.float32)
        c_honest = np.matmul(a, b)

        # Honest passes
        passed_h, disc_h, bound_h, _ = freivalds_verify(a, b, c_honest, dtype="fp32")
        assert passed_h is True

        # Tampered fails
        c_bad = c_honest.copy()
        c_bad[10, 20] += 5.0
        passed_b, disc_b, bound_b, proof_b = freivalds_verify(a, b, c_bad, dtype="fp32", num_trials=10)
        assert passed_b is False
        assert proof_b["dimension_m"] == 64
        assert proof_b["dimension_k"] == 256
        assert proof_b["dimension_n"] == 128


# =============================================================================
# Suite 4: Fraud Proof Builder Invariants & Rejection Rules
# =============================================================================

class TestAdversarialFraudProofBuilder:
    """
    Stress-tests FraudProofBuilder for authoritative payload construction,
    vector invariant verification, and rejection of honest products.
    """

    def test_builder_produces_authoritative_proof_for_invalid_product(self, crypto_keys):
        """
        Invalid product results in an authoritative FreivaldsFraudProof instance
        where max |ABr - Cr| > epsilon_tolerance.
        """
        rng = np.random.RandomState(4001)
        a = rng.randn(12, 16).astype(np.float32)
        b = rng.randn(16, 14).astype(np.float32)
        c_bad = np.matmul(a, b)
        c_bad[3, 5] += 2.0  # Fraud

        task_id = "task_adv_fraud_01"
        c_hash = hash_tensor(np.ascontiguousarray(c_bad))
        sig = sign_message(crypto_keys["w_priv"], format_tile_result_payload(task_id, c_hash))

        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=crypto_keys["w_pub"],
            A=a,
            B=b,
            C=c_bad,
            worker_sig=sig,
            consumer_privkey=crypto_keys["c_priv"],
            verify_worker_sig=True,
        )

        assert isinstance(proof, FreivaldsFraudProof)
        assert proof.task_id == task_id
        assert proof.worker_pubkey == crypto_keys["w_pub"]
        assert proof.consumer_pubkey == crypto_keys["c_pub"]
        assert proof.dimension_m == 12
        assert proof.dimension_k == 16
        assert proof.dimension_n == 14
        assert proof.dtype == "fp32"
        assert len(proof.random_vector_r) == 14
        assert len(proof.vector_br) == 16
        assert len(proof.vector_abr) == 12
        assert len(proof.vector_cr) == 12
        assert proof.max_observed_discrepancy > proof.epsilon_tolerance

        # Mathematically verify vectors
        abr = np.array(proof.vector_abr, dtype=np.float64)
        cr = np.array(proof.vector_cr, dtype=np.float64)
        calc_disc = float(np.max(np.abs(abr - cr)))
        assert math.isclose(calc_disc, proof.max_observed_discrepancy, rel_tol=1e-6)

    def test_builder_rejects_honest_computation_with_value_error(self, crypto_keys):
        """
        Attempting to build a fraud proof on honest/valid GEMM must fail with ValueError.
        """
        rng = np.random.RandomState(4002)
        a = rng.randn(8, 8).astype(np.float32)
        b = rng.randn(8, 8).astype(np.float32)
        c_honest = np.matmul(a, b)

        task_id = "task_honest_rejection"
        c_hash = hash_tensor(np.ascontiguousarray(c_honest))
        sig = sign_message(crypto_keys["w_priv"], format_tile_result_payload(task_id, c_hash))

        # 1. With proof_data=None (triggers witness search)
        with pytest.raises(ValueError, match="cannot be proven fraudulent"):
            FraudProofBuilder.build_fraud_proof(
                task_id=task_id,
                worker_pubkey=crypto_keys["w_pub"],
                A=a,
                B=b,
                C=c_honest,
                worker_sig=sig,
                consumer_privkey=crypto_keys["c_priv"],
                proof_data=None,
            )

        # 2. With passing proof_data (discrepancy <= tolerance)
        passed, disc, bound, proof_data = freivalds_verify(a, b, c_honest, dtype="fp32")
        assert passed is True
        with pytest.raises(ValueError, match="cannot be proven fraudulent"):
            FraudProofBuilder.build_fraud_proof(
                task_id=task_id,
                worker_pubkey=crypto_keys["w_pub"],
                A=a,
                B=b,
                C=c_honest,
                worker_sig=sig,
                consumer_privkey=crypto_keys["c_priv"],
                proof_data=proof_data,
            )

    def test_builder_rejects_sub_epsilon_perturbation(self, crypto_keys):
        """
        Perturbation smaller than dynamic Higham bound cannot be proven fraudulent.
        """
        rng = np.random.RandomState(4003)
        a = rng.randn(8, 8).astype(np.float32)
        b = rng.randn(8, 8).astype(np.float32)
        c_sub = np.matmul(a, b)
        c_sub[0, 0] += 1e-12  # Sub-epsilon

        task_id = "task_sub_epsilon"
        sig = sign_message(crypto_keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c_sub))))

        with pytest.raises(ValueError, match="cannot be proven fraudulent"):
            FraudProofBuilder.build_fraud_proof(
                task_id=task_id,
                worker_pubkey=crypto_keys["w_pub"],
                A=a,
                B=b,
                C=c_sub,
                worker_sig=sig,
                consumer_privkey=crypto_keys["c_priv"],
            )

    def test_builder_auto_witness_search_finds_single_column_corruption(self, crypto_keys):
        """
        When proof_data is None, auto-witness search isolates corrupted column via standard basis.
        """
        rng = np.random.RandomState(4004)
        a = rng.randn(10, 10).astype(np.float32)
        b = rng.randn(10, 10).astype(np.float32)
        c_corrupt = np.matmul(a, b)
        # Corrupt column 7
        c_corrupt[4, 7] += 3.5

        task_id = "task_auto_witness"
        sig = sign_message(crypto_keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c_corrupt))))

        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=crypto_keys["w_pub"],
            A=a,
            B=b,
            C=c_corrupt,
            proof_data=None,
            worker_sig=sig,
            consumer_privkey=crypto_keys["c_priv"],
        )

        assert proof.max_observed_discrepancy >= 3.0
        assert verify_fraud_proof(proof) is True

    def test_builder_dtype_support_fp16_and_bf16(self, crypto_keys):
        """
        Builder properly respects explicit fp16 and bf16 datatypes.
        """
        rng = np.random.RandomState(4005)
        a = rng.randn(8, 8).astype(np.float16)
        b = rng.randn(8, 8).astype(np.float16)
        c_bad = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
        c_bad[1, 1] += np.float16(10.0)

        task_id = "task_fp16_proof"
        sig = sign_message(crypto_keys["w_priv"], format_tile_result_payload(task_id, hash_tensor(np.ascontiguousarray(c_bad))))

        proof_fp16 = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=crypto_keys["w_pub"],
            A=a,
            B=b,
            C=c_bad,
            dtype="fp16",
            worker_sig=sig,
            consumer_privkey=crypto_keys["c_priv"],
        )
        assert proof_fp16.dtype == "fp16"
        assert verify_fraud_proof(proof_fp16) is True


# =============================================================================
# Suite 5: Independent Dispute Verifier (verify_fraud_proof) Stress Tests
# =============================================================================

class TestAdversarialVerifyFraudProof:
    """
    Stress-tests verify_fraud_proof against tampered vectors, dimension mismatches,
    falsified discrepancies, and cryptographic forgery.
    """

    def _make_valid_proof(self, crypto_keys, task_id="task_base_test") -> Tuple[FreivaldsFraudProof, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(5001)
        a = rng.randn(8, 8).astype(np.float32)
        b = rng.randn(8, 8).astype(np.float32)
        c = np.matmul(a, b)
        c[2, 2] += 5.0  # Fraudulent
        c_hash = hash_tensor(np.ascontiguousarray(c))
        sig = sign_message(crypto_keys["w_priv"], format_tile_result_payload(task_id, c_hash))

        proof = FraudProofBuilder.build_fraud_proof(
            task_id=task_id,
            worker_pubkey=crypto_keys["w_pub"],
            A=a,
            B=b,
            C=c,
            worker_sig=sig,
            consumer_privkey=crypto_keys["c_priv"],
        )
        return proof, a, b, c

    def test_verify_valid_proof_passes_both_modes(self, crypto_keys):
        """Nominal valid fraud proof passes both standalone and full-witness modes."""
        proof, a, b, c = self._make_valid_proof(crypto_keys, "task_pass_both")
        # Standalone
        assert verify_fraud_proof(proof) is True
        # Full witness
        assert verify_fraud_proof(proof, matrix_a=a, matrix_b=b, matrix_c=c) is True

    def test_verify_rejects_non_binary_vector_r(self, crypto_keys):
        """r contains non-binary value 2 -> rejected."""
        proof, _, _, _ = self._make_valid_proof(crypto_keys, "task_non_binary")
        proof.random_vector_r[0] = 2
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_all_zero_vector_r(self, crypto_keys):
        """r is all zeros -> cannot prove discrepancy, rejected."""
        proof, _, _, _ = self._make_valid_proof(crypto_keys, "task_all_zero_r")
        proof.random_vector_r = [0] * len(proof.random_vector_r)
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_tampered_vector_abr(self, crypto_keys):
        """Attacker modifies vector_abr to match vector_cr -> rejected (diff <= tolerance)."""
        proof, _, _, _ = self._make_valid_proof(crypto_keys, "task_tamper_abr")
        proof.vector_abr = list(proof.vector_cr)
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_falsified_max_observed_discrepancy(self, crypto_keys):
        """Discrepancy value in metadata does not match calculated max |ABr - Cr| -> rejected."""
        proof, _, _, _ = self._make_valid_proof(crypto_keys, "task_falsified_disc")
        proof.max_observed_discrepancy = proof.max_observed_discrepancy + 10.0
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_mismatched_vector_length_br(self, crypto_keys):
        """vector_br length != dimension_k -> rejected."""
        proof, _, _, _ = self._make_valid_proof(crypto_keys, "task_len_br")
        proof.vector_br.append(1.0)
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_mismatched_vector_length_cr(self, crypto_keys):
        """vector_cr length != dimension_m -> rejected."""
        proof, _, _, _ = self._make_valid_proof(crypto_keys, "task_len_cr")
        proof.vector_cr.pop()
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_non_finite_vector_entries(self, crypto_keys):
        """NaN or Inf in witness vectors -> rejected."""
        proof, _, _, _ = self._make_valid_proof(crypto_keys, "task_nan_vec")
        proof.vector_abr[0] = float("nan")
        assert verify_fraud_proof(proof) is False

        proof2, _, _, _ = self._make_valid_proof(crypto_keys, "task_inf_vec")
        proof2.vector_cr[0] = float("inf")
        assert verify_fraud_proof(proof2) is False

    def test_verify_rejects_forged_worker_signature(self, crypto_keys):
        """Attacker submits fraud proof with forged worker signature -> rejected."""
        proof, _, _, _ = self._make_valid_proof(crypto_keys, "task_forged_sig")
        # Sign with consumer private key instead of worker private key
        payload = format_tile_result_payload(proof.task_id, proof.invalid_result_c_hash)
        proof.worker_signature_on_result = sign_message(crypto_keys["c_priv"], payload)
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_signature_task_id_mismatch(self, crypto_keys):
        """Signature belongs to task_01, but proof specifies task_02 -> rejected."""
        proof, _, _, _ = self._make_valid_proof(crypto_keys, "task_original")
        proof.task_id = "task_replayed_substitution"
        assert verify_fraud_proof(proof) is False

    def test_verify_rejects_invalid_pubkey_hex(self, crypto_keys):
        """Malformed worker or consumer public key -> rejected."""
        proof, _, _, _ = self._make_valid_proof(crypto_keys, "task_bad_pk")
        proof.worker_pubkey = "not_a_valid_hex_key"
        assert verify_fraud_proof(proof) is False

        proof2, _, _, _ = self._make_valid_proof(crypto_keys, "task_bad_cpk")
        proof2.consumer_pubkey = "1234abcd"
        assert verify_fraud_proof(proof2) is False

    def test_verify_with_witness_matrix_cross_verification_failures(self, crypto_keys):
        """Full matrix witness cross-verification catches forged matrix inputs."""
        proof, a, b, c = self._make_valid_proof(crypto_keys, "task_cross_verify")

        # 1. Swapped matrix A
        a_corrupt = a.copy()
        a_corrupt[0, 0] += 1.0
        assert verify_fraud_proof(proof, matrix_a=a_corrupt, matrix_b=b, matrix_c=c) is False

        # 2. Swapped matrix B
        b_corrupt = b.copy()
        b_corrupt[0, 0] += 1.0
        assert verify_fraud_proof(proof, matrix_a=a, matrix_b=b_corrupt, matrix_c=c) is False

        # 3. Honest matrix C passed as witness for invalid proof
        c_honest = np.matmul(a, b)
        assert verify_fraud_proof(proof, matrix_a=a, matrix_b=b, matrix_c=c_honest) is False
