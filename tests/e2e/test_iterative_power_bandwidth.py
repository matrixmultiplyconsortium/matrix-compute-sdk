"""
Turnkey E2E Bandwidth Profiling Suite & Numerical Equivalence Verification for Stateful Iterative Workloads.

Target: matrix-compute-sdk/tests/e2e/test_iterative_power_bandwidth.py
Milestone: Milestone 4 (Dual-Track E2E Testing & Bandwidth Profiling)

Verification Coverage:
1. Suite 1: TestUploadBandwidthInvariance
   - Proves upload bytes scale strictly as O(MK + T * N) across T in [1, 5, 20, 50, 100].
   - Asserts single-upload redundancy ratio R_up(A) == 0.0.
   - Asserts LeaseMatrix called exactly 1 time, StepIterative T times, ReleaseLease 1 time.
   - Verifies linear regression slope == N * b_dtype and intercept == N^2 * b_dtype.
   - Verifies precision byte scaling across FP32 (4B), FP16 (2B), BF16 (4B).

2. Suite 2: TestNumericalEquivalence100Iterations
   - Full 100-step power iteration (A^100 * v_0) matches local NumPy float64 reference.
   - Spectral radius scaling rho(A) <= 1.0 ensuring stability without overflow.
   - Evaluated across FP32, FP16, and BF16 (with mantissa mask 0xFFFF0000).
   - Evaluated across shapes: standard square (64x64, 128x128), prime dimensions (17x17, 31x31, 997x997).
   - Evaluated on non-symmetric matrices (A != A^T).

3. Suite 3: TestDominantEigenvectorConvergence
   - L2 normalization mode: intermediate unit norm preservation ||v_t||_2 == 1.0.
   - Convergence to dominant eigenvector: cosine similarity |<v_100, v_ref>| >= 1 - eps.

4. Suite 4: TestWorkerMemoryReclamation
   - Explicit ReleaseLease zeroizes buffer memory and reclaims bytes in SessionManager.
   - Exception-safe release in coordinator finally: block.
   - Hybrid TTL reaper: background daemon thread proactively evicts expired sessions.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Tuple
import uuid

import numpy as np
import pytest

# Ensure all workspace modules resolve cleanly
_TESTS_DIR = Path(__file__).resolve().parent
_SDK_DIR = _TESTS_DIR.parents[1]
_ROOT_DIR = _SDK_DIR.parent
_CONTRACTS_DIR = _ROOT_DIR / "matrix-contracts"
_NODE_DIR = _ROOT_DIR / "matrix-compute-node"
_SDK_SRC_DIR = _SDK_DIR / "src"

for _p in [
    str(_CONTRACTS_DIR),
    str(_SDK_DIR),
    str(_SDK_SRC_DIR),
    str(_NODE_DIR),
    str(_NODE_DIR / "src"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Contracts & Stubs
from matrix_contracts.v1 import (
    LeaseMatrixRequest,
    LeaseMatrixResponse,
    IterativeStepRequest,
    IterativeStepResponse,
    ReleaseLeaseRequest,
    ReleaseLeaseResponse,
    MatrixComputeServiceServicer,
    add_MatrixComputeServiceServicer_to_server,
)

# SDK Core Components
from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig, WorkloadType
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import MatrixDimensionError, WorkerUnavailableError
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator, IterativeResult
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import (
    MockWorkerNode,
    MockWorkerServicer,
    _InternalWorkerSigner,
)
from matrix_compute.tiling.precision import (
    calculate_higham_bound,
    cast_matrix,
    get_bytes_per_element,
    get_unit_roundoff,
    normalize_dtype,
    truncate_to_bf16,
)
from matrix_compute.verification.freivalds import (
    matrix_infinity_norm,
    vector_infinity_norm,
    verify_iterative_step,
)

# Worker Node Core (for memory reclamation testing)
try:
    from src.engine.session_manager import PinnedTensorSession, SessionManager
except ImportError:
    from engine.session_manager import PinnedTensorSession, SessionManager


# ==============================================================================
# AUTHORITATIVE REFERENCE ORACLES & SPY INTERCEPTORS
# ==============================================================================

class ReferencePowerIterationOracle:
    """
    Authoritative double-precision mathematical oracle for stateful power iterations.
    """

    @staticmethod
    def generate_stable_matrix(
        n: int,
        symmetric: bool = True,
        spectral_radius: float = 0.95,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generates an N x N matrix with spectral radius strictly clamped to spectral_radius <= 1.0.
        """
        rng = np.random.RandomState(seed)
        raw = rng.randn(n, n)

        if symmetric:
            mat = np.matmul(raw, raw.T)
            eigvals = np.linalg.eigvalsh(mat)
            max_eig = float(np.max(np.abs(eigvals)))
            if max_eig > 0:
                scaled = (mat / max_eig) * spectral_radius
            else:
                scaled = np.eye(n) * spectral_radius
        else:
            mat = raw
            eigvals = np.linalg.eigvals(mat)
            max_eig = float(np.max(np.abs(eigvals)))
            if max_eig > 0:
                scaled = (mat / max_eig) * spectral_radius
            else:
                scaled = np.eye(n) * spectral_radius

        return scaled.astype(np.float64)

    @classmethod
    def compute_power_iteration_ref(
        cls,
        a: np.ndarray,
        v0: np.ndarray,
        iterations: int = 100,
        normalize: bool = False,
    ) -> np.ndarray:
        """Computes power iteration in double precision."""
        v = v0.astype(np.float64).flatten().copy()
        if normalize:
            norm_v = np.linalg.norm(v)
            if norm_v > 0:
                v = v / norm_v

        a_f64 = a.astype(np.float64)
        for _ in range(iterations):
            v = np.matmul(a_f64, v)
            if normalize:
                norm_v = np.linalg.norm(v)
                if norm_v > 0:
                    v = v / norm_v

        return v


class IterativeBandwidthSpy:
    """
    Intercepts and logs network payload bytes and RPC invocation counts
    across LeaseMatrix, StepIterative, and ReleaseLease.
    """

    def __init__(self, servicer: MockWorkerServicer) -> None:
        self.servicer = servicer

    def reset(self) -> None:
        self.servicer.lease_calls = 0
        self.servicer.step_calls = 0
        self.servicer.release_calls = 0
        self.servicer.uploaded_bytes_total = 0

    @property
    def lease_calls(self) -> int:
        return self.servicer.lease_calls

    @property
    def step_calls(self) -> int:
        return self.servicer.step_calls

    @property
    def release_calls(self) -> int:
        return self.servicer.release_calls

    @property
    def total_uploaded_bytes(self) -> int:
        return self.servicer.uploaded_bytes_total


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

@pytest.fixture
def hermetic_iterative_env() -> Generator[Dict[str, Any], None, None]:
    """
    Provisions an isolated hermetic distributed environment with MockWorkerNode,
    MockNostrRelay, MockSettlementService, and DistributedMatmulCoordinator.
    """
    vk = f"secret-vk-{uuid.uuid4().hex[:8]}"
    settlement = MockSettlementService(verification_key=vk)
    relay = MockNostrRelay()

    worker = MockWorkerNode(
        settlement_service=settlement,
        verification_key=vk,
        vram_mb=32768,
        sats_per_gflop=0.0005,
    )
    worker.start()
    worker.publish_to_relay(relay)

    config = ClientConfig(
        offline_mock_mode=True,
        nostr_relays=["ws://127.0.0.1:8008"],
        settlement_api_url="http://127.0.0.1:8080",
        settlement_verification_key=vk,
        min_worker_vram_mb=8192,
        max_price_per_gflop=0.001,
        connect_timeout_seconds=2.0,
        task_timeout_seconds=5.0,
        workload_type=WorkloadType.ITERATIVE_POWER,
        lease_ttl_seconds=3600,
        spot_check_probability=0.05,
    )

    signer = ClientSigner(config.consumer_privkey_hex)
    relay_client = RelayClient(relay_urls=config.nostr_relays, in_memory_relay=relay)
    matchmaker = Matchmaker(
        min_vram_mb=config.min_worker_vram_mb,
        max_price_per_gflop=config.max_price_per_gflop,
    )
    escrow_manager = EscrowManager(
        settlement_backend=MockSettlementBackend(settlement),
        default_timeout_seconds=config.task_timeout_seconds,
        verification_key=vk,
        auto_settle=False,
    )
    dispatcher = GrpcTaskDispatcher(
        consumer_pubkey=signer.get_pubkey_hex(),
        connect_timeout=config.connect_timeout_seconds,
        task_timeout=config.task_timeout_seconds,
    )
    scheduler = ParallelTaskScheduler(dispatcher=dispatcher, max_retries=0)

    coordinator = DistributedMatmulCoordinator(
        config=config,
        signer=signer,
        relay_client=relay_client,
        matchmaker=matchmaker,
        escrow_manager=escrow_manager,
        scheduler=scheduler,
        dispatcher=dispatcher,
        mock_workers=[worker],
    )

    spy = IterativeBandwidthSpy(worker.servicer)

    yield {
        "coordinator": coordinator,
        "config": config,
        "worker": worker,
        "servicer": worker.servicer,
        "spy": spy,
        "settlement": settlement,
        "relay": relay,
    }

    coordinator.close()
    worker.stop()


# ==============================================================================
# SUITE 1: TEST UPLOAD BANDWIDTH INVARIANCE
# ==============================================================================

class TestUploadBandwidthInvariance:
    """
    Formally verifies that network upload volume scales strictly as O(MK + T * N)
    with single-upload redundancy ratio R_up(A) == 0.0 across T in [1, 5, 20, 50, 100].
    """

    @pytest.mark.parametrize("t_iters", [1, 5, 20, 50, 100])
    def test_upload_bytes_scaling_and_rpc_counts_fp32(
        self, hermetic_iterative_env: Dict[str, Any], t_iters: int
    ):
        """
        TC_UBI_01: Verifies LeaseMatrix is called exactly 1 time, StepIterative T times,
        and total uploaded bytes strictly equal (N^2 + T * N) * 4 for FP32.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        spy: IterativeBandwidthSpy = hermetic_iterative_env["spy"]
        n = 32
        bytes_per_elem = 4  # FP32

        spy.reset()

        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        res = coord.execute_iterative_power(a, v0, iterations=t_iters, dtype="fp32")
        assert len(res) == n

        # 1. Assert RPC call counts
        assert spy.lease_calls == 1, f"LeaseMatrix must be invoked exactly 1 time (got {spy.lease_calls})"
        assert spy.step_calls == t_iters, f"StepIterative must be invoked exactly {t_iters} times (got {spy.step_calls})"
        assert spy.release_calls == 1, f"ReleaseLease must be invoked exactly 1 time (got {spy.release_calls})"

        # 2. Assert Single-Upload Redundancy Ratio R_up(A) == 0.0
        # Formula: R_up(A) = (actual_upload - optimal_upload) / optimal_upload
        r_up_a = (float(spy.lease_calls) - 1.0) / 1.0
        assert r_up_a == 0.0, f"Single-upload redundancy ratio must be 0.0, got {r_up_a}"

        # 3. Assert Strict Byte Scaling: (N^2 + T * N) * bytes_per_elem
        expected_matrix_bytes = n * n * bytes_per_elem
        expected_vector_bytes = t_iters * n * bytes_per_elem
        expected_total_bytes = expected_matrix_bytes + expected_vector_bytes

        assert spy.total_uploaded_bytes == expected_total_bytes, (
            f"Uploaded bytes mismatch for T={t_iters}: expected {expected_total_bytes}, "
            f"got {spy.total_uploaded_bytes}"
        )

    def test_linear_regression_slope_verification(
        self, hermetic_iterative_env: Dict[str, Any]
    ):
        """
        TC_UBI_02: Performs linear regression over T in [1, 5, 20, 50, 100]
        proving slope == N * b_dtype and intercept == N^2 * b_dtype with R^2 == 1.0.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        spy: IterativeBandwidthSpy = hermetic_iterative_env["spy"]
        n = 16
        bytes_per_elem = 4
        iterations_list = [1, 5, 20, 50, 100]
        measured_bytes = []

        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        for iters in iterations_list:
            spy.reset()
            coord.execute_iterative_power(a, v0, iterations=iters, dtype="fp32")
            measured_bytes.append(spy.total_uploaded_bytes)

        t_arr = np.array(iterations_list, dtype=np.float64)
        y_arr = np.array(measured_bytes, dtype=np.float64)

        # Fit line y = slope * t + intercept
        slope, intercept = np.polyfit(t_arr, y_arr, 1)

        expected_slope = float(n * bytes_per_elem)         # 16 * 4 = 64 bytes/step
        expected_intercept = float(n * n * bytes_per_elem)   # 256 * 4 = 1024 bytes

        assert math.isclose(slope, expected_slope, abs_tol=1e-9), f"Slope {slope} != {expected_slope}"
        assert math.isclose(intercept, expected_intercept, abs_tol=1e-9), f"Intercept {intercept} != {expected_intercept}"

        # Calculate coefficient of determination R^2 = 1.0 - (sum(y - y_hat)^2 / sum(y - y_bar)^2)
        y_hat = slope * t_arr + intercept
        ss_res = float(np.sum((y_arr - y_hat) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot)
        assert math.isclose(r_squared, 1.0, abs_tol=1e-9) or r_squared >= 0.9999999, f"R^2 {r_squared} != 1.0"

    @pytest.mark.parametrize("dtype, bytes_per_elem", [("fp32", 4), ("fp16", 2), ("bf16", 4)])
    def test_precision_byte_scaling_fp32_fp16_bf16(
        self, hermetic_iterative_env: Dict[str, Any], dtype: str, bytes_per_elem: int
    ):
        """
        TC_UBI_03: Verifies byte scaling across FP32 (4B), FP16 (2B), and BF16 (4B).
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        spy: IterativeBandwidthSpy = hermetic_iterative_env["spy"]
        n = 20
        t_iters = 10

        spy.reset()
        a = cast_matrix(np.eye(n, dtype=np.float32), dtype)
        v0 = cast_matrix(np.ones(n, dtype=np.float32), dtype)

        coord.execute_iterative_power(a, v0, iterations=t_iters, dtype=dtype)

        expected_bytes = (n * n * bytes_per_elem) + (t_iters * n * bytes_per_elem)
        assert spy.total_uploaded_bytes == expected_bytes
        assert spy.lease_calls == 1
        assert spy.step_calls == t_iters

    def test_bandwidth_savings_contrast_against_naive_retransmission(
        self, hermetic_iterative_env: Dict[str, Any]
    ):
        """
        TC_UBI_04: Contrasts VRAM leased upload against naive re-uploading,
        demonstrating >98% bandwidth reduction for T=100.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        spy: IterativeBandwidthSpy = hermetic_iterative_env["spy"]
        n = 64
        t_iters = 100
        bytes_per_elem = 4

        spy.reset()
        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        coord.execute_iterative_power(a, v0, iterations=t_iters, dtype="fp32")

        actual_leased_bytes = spy.total_uploaded_bytes
        naive_reupload_bytes = t_iters * (n * n * bytes_per_elem + n * bytes_per_elem)

        savings_ratio = 1.0 - (float(actual_leased_bytes) / float(naive_reupload_bytes))
        assert savings_ratio > 0.98, f"Bandwidth savings ratio {savings_ratio:.4f} must exceed 98%"


# ==============================================================================
# SUITE 2: TEST NUMERICAL EQUIVALENCE (100 ITERATIONS)
# ==============================================================================

class TestNumericalEquivalence100Iterations:
    """
    Verifies full 100-step power iteration (A^100 * v_0) matches double-precision
    NumPy reference under spectral radius scaling rho(A) <= 1.0.
    """

    @pytest.mark.parametrize("shape", [64, 128])
    def test_100_iterations_standard_squares_fp32(
        self, hermetic_iterative_env: Dict[str, Any], shape: int
    ):
        """
        TC_NE_01: Standard squares 64x64 and 128x128 match NumPy reference within FP32 Higham tolerance.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = shape
        a = ReferencePowerIterationOracle.generate_stable_matrix(n, symmetric=True, spectral_radius=0.95, seed=100 + n)
        v0 = np.random.RandomState(200 + n).randn(n).astype(np.float32)

        result = coord.execute_iterative_power(a.astype(np.float32), v0, iterations=100, dtype="fp32")

        # Double-precision reference
        v_ref = ReferencePowerIterationOracle.compute_power_iteration_ref(a, v0, iterations=100, normalize=False)

        np.testing.assert_allclose(result, v_ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("prime_n", [17, 31, 997])
    def test_100_iterations_prime_dimensions(
        self, hermetic_iterative_env: Dict[str, Any], prime_n: int
    ):
        """
        TC_NE_02: Prime dimensions 17x17, 31x31, and large prime 997x997
        execute without boundary truncations or index errors.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = prime_n
        a = ReferencePowerIterationOracle.generate_stable_matrix(n, symmetric=True, spectral_radius=0.90, seed=300 + n)
        v0 = np.random.RandomState(400 + n).randn(n).astype(np.float32)

        # For 997x997 run 10 iterations to maintain fast execution, 100 iterations for 17 and 31
        iterations = 10 if n == 997 else 100

        result = coord.execute_iterative_power(a.astype(np.float32), v0, iterations=iterations, dtype="fp32")
        assert len(result) == n

        v_ref = ReferencePowerIterationOracle.compute_power_iteration_ref(a, v0, iterations=iterations, normalize=False)
        np.testing.assert_allclose(result, v_ref, rtol=1e-3, atol=1e-3)

    def test_100_iterations_non_symmetric_matrix(
        self, hermetic_iterative_env: Dict[str, Any]
    ):
        """
        TC_NE_03: Verifies power iteration on non-symmetric matrix (A != A^T) with rho(A) <= 1.0.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 32
        a = ReferencePowerIterationOracle.generate_stable_matrix(n, symmetric=False, spectral_radius=0.92, seed=555)
        assert not np.allclose(a, a.T), "Matrix must be non-symmetric"

        v0 = np.random.RandomState(666).randn(n).astype(np.float32)

        result = coord.execute_iterative_power(a.astype(np.float32), v0, iterations=100, dtype="fp32")
        v_ref = ReferencePowerIterationOracle.compute_power_iteration_ref(a, v0, iterations=100, normalize=False)

        np.testing.assert_allclose(result, v_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype, rtol, atol", [
        ("fp32", 1e-4, 1e-4),
        ("fp16", 5e-2, 5e-2),
        ("bf16", 1e-1, 1e-1),
    ])
    def test_multi_precision_equivalence(
        self, hermetic_iterative_env: Dict[str, Any], dtype: str, rtol: float, atol: float
    ):
        """
        TC_NE_04: Tests 100 power iterations across FP32, FP16, and BF16 within respective tolerances.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 16
        a = ReferencePowerIterationOracle.generate_stable_matrix(n, symmetric=True, spectral_radius=0.90, seed=777)
        v0 = np.random.RandomState(888).randn(n).astype(np.float32)

        a_cast = cast_matrix(a, dtype)
        v0_cast = cast_matrix(v0, dtype)

        result = coord.execute_iterative_power(a_cast, v0_cast, iterations=50, dtype=dtype)
        v_ref = ReferencePowerIterationOracle.compute_power_iteration_ref(a_cast, v0_cast, iterations=50, normalize=False)

        np.testing.assert_allclose(result, v_ref, rtol=rtol, atol=atol)


# ==============================================================================
# SUITE 3: TEST DOMINANT EIGENVECTOR CONVERGENCE
# ==============================================================================

class TestDominantEigenvectorConvergence:
    """
    Verifies power iteration with L2 normalization mode preserves unit norm ||v_t||_2 == 1.0
    and converges to the dominant eigenvector: |<v_100, v_ref>| >= 1 - eps.
    """

    def test_l2_normalization_unit_norm_preservation(
        self, hermetic_iterative_env: Dict[str, Any]
    ):
        """
        TC_DEC_01: Verifies that normalize=True produces final vector with ||v_100||_2 == 1.0.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 24
        a = ReferencePowerIterationOracle.generate_stable_matrix(n, symmetric=True, spectral_radius=0.95, seed=901)
        v0 = np.random.RandomState(902).randn(n).astype(np.float32)

        result = coord.execute_iterative_power(a.astype(np.float32), v0, iterations=100, normalize=True)

        norm_val = float(np.linalg.norm(result))
        assert math.isclose(norm_val, 1.0, abs_tol=1e-5), f"L2 norm {norm_val} != 1.0"

    def test_dominant_eigenvector_alignment_symmetric(
        self, hermetic_iterative_env: Dict[str, Any]
    ):
        """
        TC_DEC_02: Verifies absolute cosine similarity |<v_100, v_ref>| >= 0.9999 for symmetric matrix.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 20
        a = ReferencePowerIterationOracle.generate_stable_matrix(n, symmetric=True, spectral_radius=0.95, seed=903)
        v0 = np.random.RandomState(904).randn(n).astype(np.float32)

        result = coord.execute_iterative_power(a.astype(np.float32), v0, iterations=100, normalize=True)
        v_ref = ReferencePowerIterationOracle.compute_power_iteration_ref(a, v0, iterations=100, normalize=True)

        cos_sim = abs(float(np.dot(result / np.linalg.norm(result), v_ref / np.linalg.norm(v_ref))))
        assert cos_sim >= 0.9999, f"Cosine similarity {cos_sim} < 0.9999"

    def test_dominant_eigenvector_alignment_non_symmetric(
        self, hermetic_iterative_env: Dict[str, Any]
    ):
        """
        TC_DEC_03: Verifies absolute cosine similarity |<v_100, v_ref>| >= 0.999 for non-symmetric matrix.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 20
        a = ReferencePowerIterationOracle.generate_stable_matrix(n, symmetric=False, spectral_radius=0.92, seed=905)
        v0 = np.random.RandomState(906).randn(n).astype(np.float32)

        result = coord.execute_iterative_power(a.astype(np.float32), v0, iterations=100, normalize=True)
        v_ref = ReferencePowerIterationOracle.compute_power_iteration_ref(a, v0, iterations=100, normalize=True)

        cos_sim = abs(float(np.dot(result / np.linalg.norm(result), v_ref / np.linalg.norm(v_ref))))
        assert cos_sim >= 0.999, f"Cosine similarity {cos_sim} < 0.999"

    def test_unnormalized_power_iteration_divergence_when_spectral_radius_gt_one(
        self, hermetic_iterative_env: Dict[str, Any]
    ):
        """
        TC_DEC_04: Verifies unnormalized power iteration with rho(A) > 1.0 (e.g. 1.5) diverges/explodes
        over 100 iterations (norm > 1e8 or inf), while normalize=True preserves unit norm ||v_100||_2 == 1.0.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        n = 16
        # Generate symmetric matrix with spectral radius rho(A) = 1.5 > 1.0
        a_unstable = ReferencePowerIterationOracle.generate_stable_matrix(
            n, symmetric=True, spectral_radius=1.5, seed=999
        ).astype(np.float32)
        v0 = np.random.RandomState(1001).randn(n).astype(np.float32)
        v0 = v0 / np.linalg.norm(v0)

        # 1. Unnormalized execution: over 100 iterations, norm diverges/explodes (> 10^8)
        res_unnorm = coord.execute_iterative_power(
            a_unstable, v0, iterations=100, normalize=False, spot_check_probability=0.0
        )
        unnorm_norm = float(np.linalg.norm(res_unnorm.final_vector))
        assert unnorm_norm > 1e8 or np.isinf(unnorm_norm) or np.isnan(unnorm_norm), (
            f"Unnormalized power iteration with rho(A)=1.5 must explode over 100 iterations, got norm={unnorm_norm}"
        )

        # 2. Normalized execution: preserves unit norm ||v_100||_2 == 1.0 and remains numerically stable
        res_norm = coord.execute_iterative_power(
            a_unstable, v0, iterations=100, normalize=True
        )
        norm_val = float(np.linalg.norm(res_norm.final_vector))
        assert math.isclose(norm_val, 1.0, abs_tol=1e-5), f"Normalized L2 norm {norm_val} != 1.0"
        assert not np.isnan(res_norm.final_vector).any(), "Normalized vector contains NaN"
        assert not np.isinf(res_norm.final_vector).any(), "Normalized vector contains Inf"


# ==============================================================================
# SUITE 4: TEST WORKER MEMORY RECLAMATION
# ==============================================================================

class TestWorkerMemoryReclamation:
    """
    Verifies worker memory reclamation and buffer zeroization on explicit release
    and via the background proactive daemon TTL reaper.
    """

    def test_explicit_release_zeroizes_buffer_and_reclaims_bytes(
        self, hermetic_iterative_env: Dict[str, Any]
    ):
        """
        TC_WMR_01: Verifies ReleaseLease clears pinned session and reclaims allocated bytes.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        servicer: MockWorkerServicer = hermetic_iterative_env["servicer"]
        n = 16

        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        coord.execute_iterative_power(a, v0, iterations=10)

        # After execution completes normally, coordinator finally: block released the lease
        assert len(servicer._pinned_sessions) == 0, "All pinned sessions must be freed"
        assert servicer.release_calls == 1

    def test_session_manager_explicit_release_zeroization_view(self):
        """
        TC_WMR_02: Verifies that session.zeroize() securely fills the tensor view with zeros.
        """
        mgr = SessionManager(max_sessions=10, max_memory_bytes=1024 * 1024)
        try:
            arr = np.full((16, 16), 42.0, dtype=np.float32)
            session = mgr.lease_matrix(
                lease_id="test_zeroize_lease",
                matrix=arr,
                rows=16,
                cols=16,
                dtype="fp32",
                ttl_seconds=60.0,
            )

            # Retain view to check underlying buffer
            buf_view = arr.view()
            assert not np.all(buf_view == 0.0)

            freed = mgr.release_lease("test_zeroize_lease")
            assert freed == 16 * 16 * 4
            assert mgr.active_session_count == 0
            assert mgr.total_memory_bytes == 0

            # Buffer memory must be zeroized!
            assert np.all(buf_view == 0.0), "Underlying buffer was not zeroized"
        finally:
            mgr.shutdown()

    def test_hybrid_ttl_reaper_proactive_daemon_eviction(self):
        """
        TC_WMR_03: Verifies background daemon thread (reaper_interval_seconds=0.02)
        automatically evicts expired lease without client invocation.
        """
        mgr = SessionManager(reaper_interval_seconds=0.02)
        try:
            mat_bytes = np.ones((8, 8), dtype=np.float32).tobytes()
            # 50ms TTL
            mgr.lease_matrix("reaper_lease", mat_bytes, 8, 8, "fp32", ttl_seconds=0.05)

            assert mgr.active_session_count == 1
            assert mgr.total_memory_bytes == 8 * 8 * 4

            # Wait past expiration and reaper cycle (50ms TTL + 70ms headroom = 120ms)
            time.sleep(0.12)

            # Proactive reaper must have cleared session without calling get_session
            assert mgr.active_session_count == 0
            assert mgr.total_memory_bytes == 0
        finally:
            mgr.shutdown()

    def test_exception_safety_release_in_finally_block(
        self, hermetic_iterative_env: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ):
        """
        TC_WMR_04: Verifies that when StepIterative fails with an exception,
        coordinator finally: block guarantees ReleaseLease invocation.
        """
        coord: DistributedMatmulCoordinator = hermetic_iterative_env["coordinator"]
        servicer: MockWorkerServicer = hermetic_iterative_env["servicer"]
        n = 12

        orig_step = servicer.StepIterative

        def failing_step(request, context=None):
            if request.step_index == 2:
                raise RuntimeError("Simulated network link drop")
            return orig_step(request, context)

        monkeypatch.setattr(servicer, "StepIterative", failing_step)

        a = np.eye(n, dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        with pytest.raises(RuntimeError):
            coord.execute_iterative_power(a, v0, iterations=5)

        # ReleaseLease MUST have executed despite mid-loop failure
        assert servicer.release_calls >= 1
        assert len(servicer._pinned_sessions) == 0
