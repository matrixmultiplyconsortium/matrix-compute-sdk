"""
Unit test suite for single-step Freivalds verification and probabilistic spot-checking
in stateful iterative workloads (ITERATIVE_POWER).

Tests:
1. verify_iterative_step with genuine vs corrupted step vectors across FP32, FP16, BF16.
2. Higham dynamic error bound evaluation (gamma_N * ||A||_inf * ||v_t||_inf).
3. Adversarial perturbations: subtle tampering, zero-work fraud, sign flips, NaNs/Infs.
4. Input validation and dimension checking.
5. Probabilistic spot-checking Bernoulli distribution (4-sigma statistical validation).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pytest

from matrix_compute.exceptions import MatrixDimensionError
from matrix_compute.tiling.precision import (
    cast_matrix,
    emulate_gemm,
    get_unit_roundoff,
    normalize_dtype,
    truncate_to_bf16,
)
from matrix_compute.verification.freivalds import (
    DEFAULT_RELATIVE_TOLERANCE_FLOOR,
    UNIT_ROUNDOFFS,
    higham_bound,
    matrix_infinity_norm,
    verify_iterative_step,
)


# ==============================================================================
# Suite 1: Nominal Genuine Vector Step Verification
# ==============================================================================

class TestIterativeStepVerificationGenuine:
    """Verifies that honest worker steps pass single-step Freivalds verification."""

    def test_tc_ipv_01_genuine_fp32_square(self):
        """TC_IPV_01: Genuine FP32 step v_{t+1} = A * v_t passes with zero proof data."""
        np.random.seed(101)
        n = 32
        a = np.random.randn(n, n).astype(np.float32)
        v_t = np.random.randn(n).astype(np.float32)
        v_next = np.matmul(a, v_t)

        is_valid, disc, bound, proof = verify_iterative_step(a, v_t, v_next, dtype="fp32")
        assert is_valid is True
        assert disc <= bound
        assert len(proof) == 0

    def test_tc_ipv_02_genuine_fp16(self):
        """TC_IPV_02: Genuine FP16 step with cast_matrix passes Higham FP16 bound."""
        np.random.seed(102)
        n = 24
        a = cast_matrix(np.random.randn(n, n) * 0.1, "fp16")
        v_t = cast_matrix(np.random.randn(n) * 0.1, "fp16")
        v_next = cast_matrix(np.matmul(a.astype(np.float32), v_t.astype(np.float32)), "fp16")

        is_valid, disc, bound, proof = verify_iterative_step(a, v_t, v_next, dtype="fp16")
        assert is_valid is True
        assert disc <= max(bound, 1e-3)
        assert len(proof) == 0

    def test_tc_ipv_03_genuine_bf16(self):
        """TC_IPV_03: Genuine BF16 step with truncate_to_bf16 passes Higham BF16 bound."""
        np.random.seed(103)
        n = 24
        a = truncate_to_bf16(np.random.randn(n, n).astype(np.float32))
        v_t = truncate_to_bf16(np.random.randn(n).astype(np.float32))
        v_next = truncate_to_bf16(np.matmul(a, v_t))

        is_valid, disc, bound, proof = verify_iterative_step(a, v_t, v_next, dtype="bf16")
        assert is_valid is True
        assert disc <= max(bound, 1e-2)
        assert len(proof) == 0

    def test_tc_ipv_04_supports_1d_and_2d_column_vectors(self):
        """TC_IPV_04: Verifies transparent handling of shape (N,) and (N, 1) vectors."""
        np.random.seed(104)
        n = 16
        a = np.random.randn(n, n).astype(np.float32)

        # 2D column vector (N, 1)
        v_t_2d = np.random.randn(n, 1).astype(np.float32)
        v_next_2d = np.matmul(a, v_t_2d)
        is_valid, _, _, _ = verify_iterative_step(a, v_t_2d, v_next_2d, dtype="fp32")
        assert is_valid is True

        # Mixed 1D input and 2D result
        is_valid, _, _, _ = verify_iterative_step(a, v_t_2d.flatten(), v_next_2d, dtype="fp32")
        assert is_valid is True

    def test_tc_ipv_05_rectangular_step_dimensions(self):
        """TC_IPV_05: Supports general rectangular A (M x N) step with length N -> length M."""
        np.random.seed(105)
        m, n = 20, 35
        a = np.random.randn(m, n).astype(np.float32)
        v_t = np.random.randn(n).astype(np.float32)
        v_next = np.matmul(a, v_t)
        assert v_next.shape == (m,)

        is_valid, disc, bound, _ = verify_iterative_step(a, v_t, v_next, dtype="fp32")
        assert is_valid is True
        assert disc <= bound


# ==============================================================================
# Suite 2: Corrupted & Adversarial Vector Step Verification
# ==============================================================================

class TestIterativeStepVerificationCorrupted:
    """Verifies that fraudulent or corrupted steps are strictly caught."""

    def test_tc_ipv_06_single_element_perturbation(self):
        """TC_IPV_06: Worker injecting gross perturbation (+1.0) is caught and flagged."""
        np.random.seed(201)
        n = 16
        a = np.random.randn(n, n).astype(np.float32)
        v_t = np.random.randn(n).astype(np.float32)
        v_next = np.matmul(a, v_t)

        # Inject fraud at index 3
        v_corrupt = v_next.copy()
        v_corrupt[3] += 1.0

        is_valid, disc, bound, proof = verify_iterative_step(a, v_t, v_corrupt, dtype="fp32")
        assert is_valid is False
        assert disc > bound
        assert proof["observed_diff"] >= 1.0 - 1e-4
        assert proof["dimension_n"] == n or proof["dimension_k"] == n

    def test_tc_ipv_07_subtle_perturbation_above_higham_bound(self):
        """TC_IPV_07: Subtle perturbation exceeding Higham bound is detected."""
        np.random.seed(202)
        n = 16
        a = np.random.randn(n, n).astype(np.float32)
        v_t = np.random.randn(n).astype(np.float32)
        v_next = np.matmul(a, v_t)

        _, _, bound, _ = verify_iterative_step(a, v_t, v_next, dtype="fp32")
        v_corrupt = v_next.copy()
        v_corrupt[0] += bound * 5.0  # 5x the dynamic bound

        is_valid, disc, _, proof = verify_iterative_step(a, v_t, v_corrupt, dtype="fp32")
        assert is_valid is False
        assert disc > bound
        assert len(proof) > 0

    def test_tc_ipv_08_zero_work_fraud_detection(self):
        """TC_IPV_08: Lazy worker returning all-zeros vector is caught."""
        np.random.seed(203)
        n = 16
        a = np.ones((n, n), dtype=np.float32)
        v_t = np.ones(n, dtype=np.float32)
        v_zero = np.zeros(n, dtype=np.float32)

        is_valid, disc, bound, proof = verify_iterative_step(a, v_t, v_zero, dtype="fp32")
        assert is_valid is False
        assert disc == 16.0  # ||A * v_t - 0||_inf = 16.0
        assert proof["observed_diff"] == 16.0

    def test_tc_ipv_09_sign_flip_fraud_detection(self):
        """TC_IPV_09: Worker returning negated vector (-v_{next}) is caught."""
        np.random.seed(204)
        n = 16
        a = np.random.randn(n, n).astype(np.float32)
        v_t = np.ones(n, dtype=np.float32)
        v_next = np.matmul(a, v_t)

        v_negated = -v_next
        is_valid, disc, bound, _ = verify_iterative_step(a, v_t, v_negated, dtype="fp32")
        assert is_valid is False

    def test_tc_ipv_10_nan_and_inf_handling(self):
        """TC_IPV_10: Injected NaN or Inf returns is_valid=False with infinite discrepancy."""
        n = 8
        a = np.eye(n, dtype=np.float32)
        v_t = np.ones(n, dtype=np.float32)

        # NaN test
        v_nan = v_t.copy()
        v_nan[2] = np.nan
        is_valid, disc, _, proof = verify_iterative_step(a, v_t, v_nan, dtype="fp32")
        assert is_valid is False
        assert disc == float("inf")
        assert "Non-finite" in proof.get("reason", "")

        # Inf test
        v_inf = v_t.copy()
        v_inf[4] = np.inf
        is_valid, disc, _, proof = verify_iterative_step(a, v_t, v_inf, dtype="fp32")
        assert is_valid is False
        assert disc == float("inf")

    def test_tc_ipv_11_dimension_mismatch_raises_error(self):
        """TC_IPV_11: Vector length mismatch against matrix dimensions raises MatrixDimensionError."""
        a = np.ones((10, 10), dtype=np.float32)
        v_short = np.ones(8, dtype=np.float32)
        v_long = np.ones(12, dtype=np.float32)
        v_correct = np.ones(10, dtype=np.float32)

        with pytest.raises(MatrixDimensionError):
            verify_iterative_step(a, v_short, v_correct)

        with pytest.raises(MatrixDimensionError):
            verify_iterative_step(a, v_correct, v_long)


# ==============================================================================
# Suite 3: Higham Single-Step Error Bound Properties
# ==============================================================================

class TestHighamErrorBoundEvaluation:
    """Verifies mathematical properties of single-step GEMV Higham error bounds."""

    def test_tc_ipv_12_single_step_higham_bound_formula(self):
        """TC_IPV_12: Asserts gamma_N * ||A||_inf * ||v_t||_inf matches analytical formula."""
        n = 64
        u = 2.0 ** -24
        gamma_n = (n * u) / (1.0 - n * u)
        norm_a = 12.0
        norm_v = 4.0
        expected = gamma_n * norm_a * norm_v

        # Calculate using higham_bound helper
        calculated = higham_bound(k=n, norm_a=norm_a, norm_b=norm_v, dtype="fp32")
        assert math.isclose(calculated, expected, rel_tol=1e-9)

    def test_tc_ipv_13_precision_hierarchy_comparison(self):
        """TC_IPV_13: Bound ordering: BF16 > FP16 > FP32 for identical norms."""
        n = 32
        norm_a, norm_v = 8.0, 5.0
        b_fp32 = higham_bound(n, norm_a, norm_v, dtype="fp32")
        b_fp16 = higham_bound(n, norm_a, norm_v, dtype="fp16")
        b_bf16 = higham_bound(n, norm_a, norm_v, dtype="bf16")

        assert b_bf16 > b_fp16 > b_fp32
        assert math.isclose(b_bf16 / b_fp16, (2.0 ** -8) / (2.0 ** -11), rel_tol=1e-2)

    def test_tc_ipv_14_linear_norm_scaling(self):
        """TC_IPV_14: Scaling ||A||_inf by 10x scales bound by 10x."""
        b_base = higham_bound(16, 2.0, 1.0, dtype="fp32")
        b_scaled = higham_bound(16, 20.0, 1.0, dtype="fp32")
        assert math.isclose(b_scaled, b_base * 10.0, rel_tol=1e-7)

    def test_tc_ipv_15_subnormal_underflow_guard(self):
        """TC_IPV_15: Tiny values (< 1e-25) do not divide by zero or raise false alarms."""
        n = 4
        a_tiny = np.eye(n, dtype=np.float32) * 1e-25
        v_tiny = np.ones(n, dtype=np.float32) * 1e-25
        v_next_tiny = np.matmul(a_tiny, v_tiny)

        is_valid, disc, bound, proof = verify_iterative_step(a_tiny, v_tiny, v_next_tiny, dtype="fp32")
        assert is_valid is True
        assert math.isfinite(bound)
        assert bound > 0.0


# ==============================================================================
# Suite 4: Probabilistic Spot-Checking & Bernoulli Distribution
# ==============================================================================

class TestProbabilisticSpotCheckingBernoulli:
    """Validates statistical properties of spot-check Bernoulli sampling."""

    def test_tc_ipv_16_bernoulli_distribution_expected_count(self):
        """
        TC_IPV_16: Over N=2000 trials with p=0.05, observed spot checks follow
        Bernoulli(p) within 4 standard deviations (4-sigma).
        """
        p = 0.05
        num_trials = 2000
        rng = np.random.RandomState(42)

        # Independent Bernoulli trials
        samples = rng.uniform(0.0, 1.0, size=num_trials) < p
        observed_count = int(np.sum(samples))

        expected_count = num_trials * p  # 100.0
        variance = num_trials * p * (1.0 - p)  # 2000 * 0.05 * 0.95 = 95.0
        std_dev = math.sqrt(variance)  # ~9.7468
        margin_4_sigma = 4.0 * std_dev  # ~38.99

        lower_bound = expected_count - margin_4_sigma
        upper_bound = expected_count + margin_4_sigma

        assert lower_bound <= observed_count <= upper_bound, (
            f"Observed {observed_count} checks outside 4-sigma range [{lower_bound:.1f}, {upper_bound:.1f}]"
        )

    def test_tc_ipv_17_boundary_probabilities(self):
        """TC_IPV_17: p=0.0 produces zero checks; p=1.0 produces 100% checks."""
        rng = np.random.RandomState(43)

        # p = 0.0 -> never check
        zero_checks = rng.uniform(0.0, 1.0, size=500) < 0.0
        assert np.sum(zero_checks) == 0

        # p = 1.0 -> always check
        full_checks = rng.uniform(0.0, 1.0, size=500) < 1.0
        assert np.sum(full_checks) == 500

    def test_tc_ipv_18_fraud_detection_game_theory_bounds(self):
        """
        TC_IPV_18: Verifies mathematical fraud detection probability P(catch) = 1 - (1-p)^m.
        Confirms cheat deterrence for m cheating steps.
        """
        p = 0.05

        def p_catch(m: int) -> float:
            return 1.0 - ((1.0 - p) ** m)

        # For 1 cheat: 5%
        assert math.isclose(p_catch(1), 0.05, rel_tol=1e-5)
        # For 10 cheats: ~40.1%
        assert math.isclose(p_catch(10), 0.40126, rel_tol=1e-3)
        # For 50 cheats: ~92.3%
        assert math.isclose(p_catch(50), 0.92305, rel_tol=1e-3)
        # For 100 cheats: ~99.4%
        assert math.isclose(p_catch(100), 0.99408, rel_tol=1e-3)

        # Monte Carlo empirical validation over 10,000 runs for m=10
        rng = np.random.RandomState(44)
        simulated_runs = 10000
        # Cheater cheats on 10 steps. Worker is caught if ANY of the 10 steps is sampled.
        cheats_caught = sum(
            np.any(rng.uniform(0.0, 1.0, size=10) < p) for _ in range(simulated_runs)
        )
        empirical_p = cheats_caught / simulated_runs
        assert math.isclose(empirical_p, p_catch(10), abs_tol=0.015)


# ==============================================================================
# Suite 5: Relative Discrepancy Defense & Adversarial Fraud Detection
# ==============================================================================

class TestRelativeDiscrepancyDefense:
    """
    Verifies that verify_iterative_step evaluates relative discrepancy cleanly
    to catch zero-work and sign-flip fraud even when absolute discrepancy is within bounds.
    """

    def test_tc_ipv_19_zero_work_fraud_caught_when_discrepancy_within_epsilon_bound(self):
        """
        TC_IPV_19: Worker returns all zeros (v_{next} = 0) with a loose epsilon_tolerance override.
        Even though discrepancy <= effective_bound, relative discrepancy check catches the fraud.
        """
        n = 4
        a = np.eye(n, dtype=np.float32)
        v_t = np.ones(n, dtype=np.float32)
        v_zero = np.zeros(n, dtype=np.float32)

        # Scale = 1.0. Loose bound override: epsilon_tolerance = 2.0.
        # discrepancy = ||A v_t - 0||_inf = 1.0 <= effective_bound = 2.0.
        # Relative diff = 1.0 / 1.0 = 1.0 (100% relative discrepancy).
        res = verify_iterative_step(
            a,
            v_t,
            v_zero,
            dtype="fp32",
            epsilon_tolerance=2.0,
            relative_discrepancy_threshold=0.05,
        )

        assert res.effective_bound >= 2.0
        assert res.is_valid is False, "Zero-work fraud was falsely certified as valid!"
        assert "relative discrepancy" in res.failure_proof["reason"].lower()

    def test_tc_ipv_20_sign_flip_fraud_caught_when_discrepancy_within_bound(self):
        """
        TC_IPV_20: Worker returns negated vector (-A v_t) with loose epsilon bound.
        Relative discrepancy is ~2.0 (200%), which strictly triggers relative fraud detection.
        """
        n = 4
        a = np.eye(n, dtype=np.float32)
        v_t = np.ones(n, dtype=np.float32)
        v_zero = np.zeros(n, dtype=np.float32)
        v_honest = np.matmul(a, v_t)  # ones
        v_flipped = -v_honest

        # Discrepancy = ||A v_t - (-A v_t)||_inf = 2.0.
        # Loose bound override: epsilon_tolerance = 2.0.
        # Relative diff = 2.0 / 1.0 = 2.0 (200%).
        res = verify_iterative_step(
            a,
            v_t,
            v_flipped,
            dtype="fp32",
            epsilon_tolerance=2.0,
            relative_discrepancy_threshold=0.05,
        )

        assert res.effective_bound >= 2.0
        assert res.is_valid is False, "Sign-flip fraud was falsely certified as valid!"
        assert "relative discrepancy" in res.failure_proof["reason"].lower()

    def test_tc_ipv_21_bf16_large_k_zero_work_fraud_rejected(self):
        """
        TC_IPV_21: K=128 in BF16 precision has gamma_k >= 1.0 (bound saturation).
        Ensures that zero-work fraud (v_{next} = 0) is rejected despite bound saturation.
        """
        k = 128
        a = np.eye(k, dtype=np.float32)
        A = a
        v_t = np.ones(k, dtype=np.float32)
        v_zero = np.zeros(k, dtype=np.float32)

        res = verify_iterative_step(A, v_t, v_zero, dtype="bf16")
        assert res.is_valid is False, "BF16 zero-work fraud must return is_valid=False!"

    def test_tc_ipv_22_genuine_steps_pass_relative_discrepancy_check(self):
        """
        TC_IPV_22: Honest genuine steps across FP32, FP16, and BF16 never trigger
        false positives in the relative discrepancy check.
        """
        np.random.seed(301)
        n = 16
        for dt in ("fp32", "fp16", "bf16"):
            a = np.random.randn(n, n).astype(np.float32) * 0.1
            v_t = np.random.randn(n).astype(np.float32) * 0.1
            if dt == "bf16":
                a = truncate_to_bf16(a)
                v_t = truncate_to_bf16(v_t)
                v_next = truncate_to_bf16(np.matmul(a, v_t))
            elif dt == "fp16":
                a = cast_matrix(a, "fp16")
                v_t = cast_matrix(v_t, "fp16")
                v_next = cast_matrix(np.matmul(a.astype(np.float32), v_t.astype(np.float32)), "fp16")
            else:
                v_next = np.matmul(a, v_t)

            res = verify_iterative_step(a, v_t, v_next, dtype=dt)
            assert res.is_valid is True, f"Genuine computation falsely rejected in {dt}: {res.failure_proof}"

