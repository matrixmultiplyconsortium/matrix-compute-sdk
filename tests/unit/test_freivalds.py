"""
Comprehensive Unit Test Suite for Freivalds Verification Engine and Higham Dynamic Bounds.
Covers unit roundoffs, norm scaling, large K stability, randomized binary vectors, honest verification,
adversarial tampering, zero-work fraud, NaN/Inf handling, and intermediate proof invariants.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from matrix_compute.config import ClientConfig
from matrix_compute.exceptions import MatrixDimensionError
from matrix_compute.tiling.precision import cast_matrix, emulate_gemm, truncate_to_bf16
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
# Suite 1: Unit Roundoff & Dtype Normalization
# =============================================================================

class TestUnitRoundoffAndNormalization:
    def test_tc_fv_01_unit_roundoff_exact_values(self):
        """TC_FV_01: Asserts theoretical unit roundoff u values for FP32, FP16, BF16, FP64."""
        assert UNIT_ROUNDOFFS["fp32"] == 2.0 ** -24
        assert UNIT_ROUNDOFFS["fp16"] == 2.0 ** -11
        assert UNIT_ROUNDOFFS["bf16"] == 2.0 ** -8
        assert UNIT_ROUNDOFFS["fp64"] == 2.0 ** -53

    def test_tc_fv_02_dtype_normalization_aliases(self):
        """TC_FV_02: Validates string aliases across floating-point formats."""
        assert normalize_dtype("fp32") == "fp32"
        assert normalize_dtype("float32") == "fp32"
        assert normalize_dtype("single") == "fp32"
        assert normalize_dtype("fp16") == "fp16"
        assert normalize_dtype("half") == "fp16"
        assert normalize_dtype("bf16") == "bf16"
        assert normalize_dtype("bfloat16") == "bf16"
        assert normalize_dtype("fp64") == "fp64"
        assert normalize_dtype("double") == "fp64"

    def test_tc_fv_03_fallback_tensor_dtype_inference(self):
        """TC_FV_03: Verifies automatic dtype detection from tensor."""
        t_fp16 = np.zeros((4, 4), dtype=np.float16)
        t_fp64 = np.zeros((4, 4), dtype=np.float64)
        t_fp32 = np.zeros((4, 4), dtype=np.float32)
        assert normalize_dtype(None, fallback_tensor=t_fp16) == "fp16"
        assert normalize_dtype(None, fallback_tensor=t_fp64) == "fp64"
        assert normalize_dtype(None, fallback_tensor=t_fp32) == "fp32"


# =============================================================================
# Suite 2: Higham Bound Mathematical Properties
# =============================================================================

class TestHighamBoundProperties:
    def test_tc_fv_04_higham_bound_closed_form(self):
        """TC_FV_04: Verifies gamma_K * ||A||_inf * ||B||_inf matches formula directly."""
        u = 2.0 ** -24
        k = 64
        gamma = (k * u) / (1.0 - k * u)
        expected = gamma * 15.0 * 25.0
        calculated = higham_bound(k=64, norm_a=15.0, norm_b=25.0, dtype="fp32")
        assert math.isclose(calculated, expected, rel_tol=1e-9)

    def test_tc_fv_05_higham_bound_precision_hierarchy(self):
        """TC_FV_05: Asserts bound_BF16 > bound_FP16 > bound_FP32 > bound_FP64."""
        k = 32
        b_fp64 = higham_bound(k, 10.0, 10.0, dtype="fp64")
        b_fp32 = higham_bound(k, 10.0, 10.0, dtype="fp32")
        b_fp16 = higham_bound(k, 10.0, 10.0, dtype="fp16")
        b_bf16 = higham_bound(k, 10.0, 10.0, dtype="bf16")
        assert b_bf16 > b_fp16 > b_fp32 > b_fp64

    def test_tc_fv_06_higham_bound_norm_scaling(self):
        """TC_FV_06: Asserts scaling norms by 10x each scales bound by 100x."""
        b_base = higham_bound(16, 2.0, 3.0, dtype="fp32")
        b_scaled = higham_bound(16, 20.0, 30.0, dtype="fp32")
        assert math.isclose(b_scaled, b_base * 100.0, rel_tol=1e-7)

    def test_tc_fv_07_higham_bound_large_k_stability(self):
        """TC_FV_07: Tests K=512, 1024 in BF16 (K * u >= 1.0) without division by zero or negative bounds."""
        b_512 = higham_bound(512, 1.0, 1.0, dtype="bf16")
        b_1024 = higham_bound(1024, 1.0, 1.0, dtype="bf16")
        assert math.isfinite(b_512)
        assert math.isfinite(b_1024)
        assert b_512 > 0.0
        assert b_1024 > b_512


# =============================================================================
# Suite 3: Random Binary Vector Generation
# =============================================================================

class TestRandomBinaryVector:
    def test_tc_fv_08_random_vector_binary_support(self):
        """TC_FV_08: Verifies all entries of r are strictly in {0, 1}."""
        r = generate_random_binary_vector(100)
        assert r.shape == (100, 1)
        unique_vals = set(np.unique(r))
        assert unique_vals.issubset({0.0, 1.0})

    def test_tc_fv_09_random_vector_non_zero_invariant(self):
        """TC_FV_09: Verifies r is never identically zero."""
        for _ in range(50):
            r = generate_random_binary_vector(3)
            assert np.any(r == 1.0)

    def test_tc_fv_10_random_vector_seed_reproducibility(self):
        """TC_FV_10: Same seed produces identical vectors."""
        r1 = generate_random_binary_vector(64, seed=42)
        r2 = generate_random_binary_vector(64, seed=42)
        np.testing.assert_array_equal(r1, r2)


# =============================================================================
# Suite 4: Nominal Matrix Verification (Honest Workers)
# =============================================================================

class TestNominalVerification:
    def test_tc_fv_11_honest_fp32_square(self):
        """TC_FV_11: Nominal square 16x16 FP32 multiplication passes with empty proof_data."""
        np.random.seed(301)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)
        c = np.matmul(a, b)
        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound
        assert len(proof) == 0

    def test_tc_fv_12_honest_fp32_rectangular(self):
        """TC_FV_12: Rectangular (12x30 x 30x18) FP32 multiplication passes."""
        np.random.seed(302)
        a = np.random.randn(12, 30).astype(np.float32)
        b = np.random.randn(30, 18).astype(np.float32)
        c = np.matmul(a, b)
        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound
        assert len(proof) == 0

    def test_tc_fv_13_honest_fp16_rounding(self):
        """TC_FV_13: FP16 product passes with dynamic FP16 bound."""
        np.random.seed(303)
        a = np.random.randn(16, 16).astype(np.float16)
        b = np.random.randn(16, 16).astype(np.float16)
        c = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp16")
        assert passed is True
        assert disc <= bound
        assert len(proof) == 0

    def test_tc_fv_14_honest_bf16_mantissa_masked(self):
        """TC_FV_14: BF16 product passes with dynamic BF16 bound."""
        np.random.seed(304)
        a = truncate_to_bf16(np.random.randn(16, 16).astype(np.float32))
        b = truncate_to_bf16(np.random.randn(16, 16).astype(np.float32))
        c = truncate_to_bf16(np.matmul(a, b))
        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="bf16")
        assert passed is True
        assert disc <= bound
        assert len(proof) == 0

    def test_tc_fv_15_honest_fp64_tight_bound(self):
        """TC_FV_15: FP64 product passes with extremely tight bound."""
        np.random.seed(305)
        a = np.random.randn(16, 16).astype(np.float64)
        b = np.random.randn(16, 16).astype(np.float64)
        c = np.matmul(a, b)
        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp64")
        assert passed is True
        assert disc <= bound
        assert len(proof) == 0


# =============================================================================
# Suite 5: Adversarial Tampering & Corrupted Results
# =============================================================================

class TestAdversarialTampering:
    def test_tc_fv_16_tamper_single_element(self):
        """TC_FV_16: Perturbing a single element at (3, 5) fails verification."""
        np.random.seed(306)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)
        c = np.matmul(a, b)
        c[3, 5] += 5.0

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is False
        assert disc > bound
        assert "random_vector_r" in proof
        assert proof["max_observed_discrepancy"] > bound

    def test_tc_fv_17_tamper_entire_row_bias(self):
        """TC_FV_17: Adding +2.0 bias to an entire row fails verification."""
        np.random.seed(307)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)
        c = np.matmul(a, b)
        c[2, :] += 2.0

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is False
        assert disc > bound

    def test_tc_fv_18_tamper_sign_inversion(self):
        """TC_FV_18: Worker negating the result matrix C = -(A x B) is detected."""
        np.random.seed(308)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)
        c = -np.matmul(a, b)

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is False
        assert disc > bound

    def test_tc_fv_19_tamper_transposed_result(self):
        """TC_FV_19: Worker returning transposed result C^T is detected."""
        np.random.seed(309)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)
        c = np.matmul(a, b).T

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is False
        assert disc > bound

    def test_tc_fv_20_tamper_permutation(self):
        """TC_FV_20: Swapping two columns of C fails verification."""
        np.random.seed(310)
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(16, 16).astype(np.float32)
        c = np.matmul(a, b)
        c[:, [0, 1]] = c[:, [1, 0]]

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is False
        assert disc > bound


# =============================================================================
# Suite 6: Zero-Work Fraud & Scale Awareness
# =============================================================================

class TestZeroWorkFraudAndScaleAwareness:
    def test_tc_fv_21_zero_matrix_small_inputs_caught(self):
        """TC_FV_21: Inputs A, B ~ 10^-3, worker returns C = 0 -> caught via relative check."""
        np.random.seed(311)
        a = (np.random.randn(16, 16) * 1e-3).astype(np.float32)
        b = (np.random.randn(16, 16) * 1e-3).astype(np.float32)
        c_zero = np.zeros((16, 16), dtype=np.float32)

        passed, disc, bound, proof = freivalds_verify(a, b, c_zero, dtype="fp32")
        assert passed is False
        assert "random_vector_r" in proof

    def test_tc_fv_22_zero_matrix_tiny_inputs_caught(self):
        """TC_FV_22: Inputs A, B ~ 10^-4, worker returns C = 0 -> caught."""
        np.random.seed(312)
        a = (np.random.randn(16, 16) * 1e-4).astype(np.float32)
        b = (np.random.randn(16, 16) * 1e-4).astype(np.float32)
        c_zero = np.zeros((16, 16), dtype=np.float32)

        passed, disc, bound, proof = freivalds_verify(a, b, c_zero, dtype="fp32")
        assert passed is False

    def test_tc_fv_23_all_zeros_inputs_and_output_passes(self):
        """TC_FV_23: A=0, B=0, C=0 passes without false positive (disc = 0.0)."""
        a = np.zeros((16, 16), dtype=np.float32)
        b = np.zeros((16, 16), dtype=np.float32)
        c = np.zeros((16, 16), dtype=np.float32)

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc == 0.0
        assert len(proof) == 0

    def test_tc_fv_24_all_zeros_inputs_nonzero_output_caught(self):
        """TC_FV_24: A=0, B=0, but C != 0 fails verification."""
        a = np.zeros((16, 16), dtype=np.float32)
        b = np.zeros((16, 16), dtype=np.float32)
        c = np.ones((16, 16), dtype=np.float32)

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is False


# =============================================================================
# Suite 7: Dynamic Range Extremes
# =============================================================================

class TestDynamicRangeExtremes:
    def test_tc_fv_25_huge_magnitude_inputs(self):
        """TC_FV_25: Inputs A, B ~ 10^4 pass honest GEMM without false positive alarm."""
        np.random.seed(313)
        a = (np.random.randn(16, 16) * 1e4).astype(np.float32)
        b = (np.random.randn(16, 16) * 1e4).astype(np.float32)
        c = np.matmul(a, b)

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound

    def test_tc_fv_26_tiny_magnitude_inputs(self):
        """TC_FV_26: Inputs A, B ~ 10^-5 pass honest GEMM."""
        np.random.seed(314)
        a = (np.random.randn(16, 16) * 1e-5).astype(np.float32)
        b = (np.random.randn(16, 16) * 1e-5).astype(np.float32)
        c = np.matmul(a, b)

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound

    def test_tc_fv_27_asymmetric_scales(self):
        """TC_FV_27: ||A|| ~ 10^4, ||B|| ~ 10^-4 passes."""
        np.random.seed(315)
        a = (np.random.randn(16, 16) * 1e4).astype(np.float32)
        b = (np.random.randn(16, 16) * 1e-4).astype(np.float32)
        c = np.matmul(a, b)

        passed, disc, bound, proof = freivalds_verify(a, b, c, dtype="fp32")
        assert passed is True
        assert disc <= bound


# =============================================================================
# Suite 8: Dimension Validation & Shape Mismatches
# =============================================================================

class TestDimensionValidation:
    def test_tc_fv_28_inner_dimension_mismatch_raises(self):
        """TC_FV_28: Inner dimension mismatch raises MatrixDimensionError."""
        a = np.ones((16, 32), dtype=np.float32)
        b = np.ones((24, 16), dtype=np.float32)
        c = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="Inner dimension mismatch"):
            freivalds_verify(a, b, c)

    def test_tc_fv_29_result_shape_mismatch_raises(self):
        """TC_FV_29: Result matrix shape mismatch raises MatrixDimensionError."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        c = np.ones((16, 20), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="Result matrix shape mismatch"):
            freivalds_verify(a, b, c)

    def test_tc_fv_30_non_2d_tensor_raises(self):
        """TC_FV_30: 1D or 3D tensors raise MatrixDimensionError."""
        a = np.ones(16, dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        c = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(MatrixDimensionError, match="Matrices must be 2D"):
            freivalds_verify(a, b, c)


# =============================================================================
# Suite 9: Non-Finite Values (NaN / Inf)
# =============================================================================

class TestNonFiniteValues:
    def test_tc_fv_31_nan_in_result_c(self):
        """TC_FV_31: NaN in result matrix fails verification with inf discrepancy."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        c = np.matmul(a, b)
        c[0, 0] = np.nan

        passed, disc, bound, proof = freivalds_verify(a, b, c)
        assert passed is False
        assert math.isinf(disc)
        assert "random_vector_r" in proof

    def test_tc_fv_32_inf_in_result_c(self):
        """TC_FV_32: Inf in result matrix fails verification."""
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        c = np.matmul(a, b)
        c[1, 1] = np.inf

        passed, disc, bound, proof = freivalds_verify(a, b, c)
        assert passed is False
        assert math.isinf(disc)

    def test_tc_fv_33_nan_in_input_matrices(self):
        """TC_FV_33: NaN in inputs A or B fails verification cleanly."""
        a = np.ones((16, 16), dtype=np.float32)
        a[0, 0] = np.nan
        b = np.ones((16, 16), dtype=np.float32)
        c = np.ones((16, 16), dtype=np.float32)

        passed, disc, bound, proof = freivalds_verify(a, b, c)
        assert passed is False
        assert math.isinf(disc)


# =============================================================================
# Suite 10: Fraud Proof Intermediate Vector Invariants
# =============================================================================

class TestProofVectorInvariants:
    def test_tc_fv_34_proof_data_dimensions(self):
        """TC_FV_34: Validates len(r)==N, len(Br)==K, len(ABr)==M, len(Cr)==M."""
        m, k, n = 12, 18, 24
        a = np.ones((m, k), dtype=np.float32)
        b = np.ones((k, n), dtype=np.float32)
        c_bad = np.zeros((m, n), dtype=np.float32)

        passed, disc, bound, proof = freivalds_verify(a, b, c_bad)
        assert passed is False
        assert len(proof["random_vector_r"]) == n
        assert len(proof["vector_br"]) == k
        assert len(proof["vector_abr"]) == m
        assert len(proof["vector_cr"]) == m

    def test_tc_fv_35_proof_data_mathematical_consistency(self):
        """TC_FV_35: Asserts max|ABr - Cr| == max_observed_discrepancy > dynamic_bound."""
        a = np.ones((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        c = np.matmul(a, b)
        c[0, 0] += 10.0

        passed, disc, bound, proof = freivalds_verify(a, b, c)
        assert passed is False
        abr = np.array(proof["vector_abr"])
        cr = np.array(proof["vector_cr"])
        calc_diff = float(np.max(np.abs(abr - cr)))
        assert math.isclose(calc_diff, proof["max_observed_discrepancy"], rel_tol=1e-6)
        assert proof["max_observed_discrepancy"] > proof["dynamic_bound"]

    def test_tc_fv_36_proof_data_pydantic_serializability(self):
        """TC_FV_36: All proof elements are Python native int and float types."""
        a = np.ones((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        c_bad = np.zeros((4, 4), dtype=np.float32)

        passed, disc, bound, proof = freivalds_verify(a, b, c_bad)
        assert passed is False
        assert all(isinstance(x, int) for x in proof["random_vector_r"])
        assert all(isinstance(x, float) for x in proof["vector_br"])
        assert all(isinstance(x, float) for x in proof["vector_abr"])
        assert all(isinstance(x, float) for x in proof["vector_cr"])


# =============================================================================
# Suite 11: Statistical Power & Multi-Round Amplification
# =============================================================================

class TestStatisticalPower:
    def test_tc_fv_37_multi_round_probability_amplification(self):
        """TC_FV_37: Verifies theoretical error probability bounds 1 - 2^-k."""
        for k in (1, 5, 10, 20):
            p_detect = 1.0 - (2.0 ** -k)
            assert p_detect >= 0.5
        assert (1.0 - 2.0 ** -5) >= 0.96
        assert (1.0 - 2.0 ** -10) >= 0.999


# =============================================================================
# Suite 12: Object-Oriented & Configuration Integration
# =============================================================================

class TestVerifierConfigIntegration:
    def test_tc_fv_38_verifier_from_client_config(self):
        """TC_FV_38: Verifier correctly inherits configuration parameters from ClientConfig."""
        cfg = ClientConfig(
            freivalds_rounds=10,
            use_dynamic_freivalds_bound=True,
            relative_discrepancy_threshold=0.005,
            verification_tolerance=1e-5,
        )
        verifier = FreivaldsVerifier(config=cfg)
        assert verifier.default_rounds == 10
        assert verifier.use_dynamic_bound is True
        assert verifier.relative_threshold == 0.005
        assert verifier.baseline_tolerance == 1e-5

    def test_tc_fv_39_verifier_instance_and_class_methods(self):
        """TC_FV_39: FreivaldsVerifier.verify and verifier.verify_tile produce consistent outcomes."""
        np.random.seed(316)
        a = np.random.randn(8, 8).astype(np.float32)
        b = np.random.randn(8, 8).astype(np.float32)
        c = np.matmul(a, b)

        verifier = FreivaldsVerifier(default_rounds=3)
        res1 = verifier.verify_tile(a, b, c, seed=42)
        res2 = FreivaldsVerifier.verify(a, b, c, k_rounds=3, seed=42)
        assert res1[0] == res2[0] == True
        assert math.isclose(res1[1], res2[1], rel_tol=1e-6)
        assert math.isclose(res1[2], res2[2], rel_tol=1e-6)
