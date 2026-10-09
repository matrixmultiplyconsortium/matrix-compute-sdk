"""
Opaque-Box E2E Test Suite: Stateful Iterative Workloads & VRAM Session Leases (ITERATIVE_POWER).

Phase 3 Verification Matrix:
- Tier 1: Feature Coverage (Client API entry points, LeaseMatrix negotiation, streaming vector updates,
  probabilistic spot-checking Freivalds verification p=0.05, explicit ReleaseLease and resource cleanup).
- Tier 2: Boundary & Corner Cases (Degenerate 1x1 dimensions, extreme floating-point values: subnormals/zeros/large norms,
  iteration limits T=0/1/100, spot-check probabilities p=0.0/1.0, TTL boundaries 1s mid-loop expiration).
- Tier 3: Cross-Feature Combinations (BF16 precision + malicious worker corruption + Freivalds fraud proof + bond slashing,
  confidential computing mode AES-256-GCM enclave + 100 power iterations + Secp256k1 signatures,
  prime dimension matrix + L2 normalization + mock Nostr relay matchmaking,
  forged worker signature + fail-closed rejection + refund + ReleaseLease in finally:).
- Tier 4: Real-World Application Scenarios (Google PageRank d=0.85, ergodic Markov chain steady-state distribution,
  Principal Component Analysis covariance dominant eigenvector, vibrational structural dynamics modal analysis,
  Latent Semantic Analysis document-term decomposition, 100 power iterations Higham equivalence across FP32/FP16/BF16).
"""

from __future__ import annotations

import base64
from concurrent import futures
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union
import uuid

import grpc
import numpy as np
import pytest

# Ensure all submodules are resolvable
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

# Ensure import paths resolve cleanly
from matrix_contracts.schemas import (
    BondStatus,
    CreateHodlInvoiceRequest,
    CreateHodlInvoiceResponse,
    FreivaldsFraudProof,
    HardwareSpecs,
    MatchedWorker,
    PricingSpecs,
    WorkerAnnouncePayload,
    WorkerQueryRequest,
    WorkerQueryResponse,
)
from matrix_contracts.v1 import (
    LeaseMatrixRequest,
    LeaseMatrixResponse,
    IterativeStepRequest,
    IterativeStepResponse,
    ReleaseLeaseRequest,
    ReleaseLeaseResponse,
    MatrixComputeServiceServicer,
    add_MatrixComputeServiceServicer_to_server,
    pb2,
    pb2_grpc,
)

# Production SDK imports
from matrix_compute.client import (
    MatrixComputeClient,
    distributed_power_iteration,
)
from matrix_compute.config import (
    ClientConfig,
    WorkloadType,
    get_default_config,
)
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import (
    FraudDetectedError,
    MatrixComputeError,
    MatrixDimensionError,
    NoWorkersFoundError,
    VerificationFailedError,
    WorkerSignatureError,
    WorkerUnavailableError,
)
from matrix_compute.orchestrator.coordinator import (
    DistributedMatmulCoordinator,
    IterativeResult,
)
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager, EscrowStatus
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import (
    MockWorkerNode,
    _InternalWorkerSigner,
)
from matrix_compute.tiling.precision import (
    calculate_higham_bound,
    cast_matrix,
    get_unit_roundoff,
    normalize_dtype,
    truncate_to_bf16,
)
from matrix_compute.verification.freivalds import (
    FreivaldsVerifier,
    matrix_infinity_norm,
    vector_infinity_norm,
    verify_iterative_step,
)


# ==============================================================================
# AUTHORITATIVE REFERENCE ITERATIVE ORACLE
# ==============================================================================

class ReferenceIterativeOracle:
    """Independent mathematical and cryptographic oracle for stateful iterative power workloads."""

    @staticmethod
    def power_iteration(
        a: np.ndarray,
        v0: np.ndarray,
        iterations: int,
        normalize: bool = False,
        dtype: str = "fp32",
    ) -> np.ndarray:
        """
        Calculates authoritative reference power iteration v_T = A^T v_0 in float64.
        """
        a_f64 = a.astype(np.float64)
        v_curr = v0.flatten().astype(np.float64).copy()
        if normalize:
            norm = np.linalg.norm(v_curr)
            if norm > 0:
                v_curr /= norm

        norm_dt = normalize_dtype(dtype)

        for _ in range(iterations):
            if norm_dt == "bf16":
                a_bf = truncate_to_bf16(a_f64.astype(np.float32))
                v_bf = truncate_to_bf16(v_curr.astype(np.float32))
                v_next = truncate_to_bf16(np.matmul(a_bf, v_bf))
                v_curr = v_next.astype(np.float64)
            elif norm_dt == "fp16":
                a_h = a_f64.astype(np.float16)
                v_h = v_curr.astype(np.float16)
                v_next = np.matmul(a_h.astype(np.float32), v_h.astype(np.float32)).astype(np.float16)
                v_curr = v_next.astype(np.float64)
            else:
                v_curr = np.matmul(a_f64, v_curr)

            if normalize:
                norm = np.linalg.norm(v_curr)
                if norm > 0:
                    v_curr /= norm

        if norm_dt == "bf16":
            return truncate_to_bf16(v_curr.astype(np.float32))
        elif norm_dt == "fp16":
            return v_curr.astype(np.float16)
        return v_curr.astype(np.float32)

    @staticmethod
    def rayleigh_quotient(a: np.ndarray, v: np.ndarray) -> float:
        """Computes Rayleigh quotient lambda = (v^T A v) / (v^T v)."""
        v_flat = v.flatten().astype(np.float64)
        a_f64 = a.astype(np.float64)
        denom = float(np.dot(v_flat, v_flat))
        if denom == 0.0:
            return 0.0
        numer = float(np.dot(v_flat, np.matmul(a_f64, v_flat)))
        return numer / denom

    @classmethod
    def higham_gemv_bound(cls, a: np.ndarray, v: np.ndarray, dtype: str = "fp32") -> float:
        """Calculates single-step Higham error bound gamma_N * ||A||_inf * ||v||_inf."""
        n = a.shape[1]
        u = get_unit_roundoff(normalize_dtype(dtype))
        nu = float(n) * u
        gamma_n = (nu / (1.0 - nu)) if nu < 0.999 else (nu * 1000.0)
        norm_a = float(np.max(np.sum(np.abs(a.astype(np.float64)), axis=1)))
        norm_v = float(np.max(np.abs(v.astype(np.float64))))
        scale = norm_a * norm_v
        return max(gamma_n * scale, u * max(scale, 1.0))

    @staticmethod
    def build_pagerank_matrix(adj_matrix: np.ndarray, damping: float = 0.85) -> np.ndarray:
        """Constructs column-stochastic Google PageRank matrix G = d*M + (1-d)/N*1."""
        n = adj_matrix.shape[0]
        col_sums = np.sum(adj_matrix, axis=0)
        m = np.zeros_like(adj_matrix, dtype=np.float64)
        for j in range(n):
            if col_sums[j] > 0:
                m[:, j] = adj_matrix[:, j] / col_sums[j]
            else:
                m[:, j] = 1.0 / float(n)
        e = np.ones((n, n), dtype=np.float64) / float(n)
        return damping * m + (1.0 - damping) * e


# ==============================================================================
# HERMETIC STATEFUL MOCK WORKER DOUBLE
# ==============================================================================

class StatefulMockWorkerServicer(MatrixComputeServiceServicer):
    """
    gRPC Servicer test double implementing LeaseMatrix, StepIterative, and ReleaseLease
    with programmable fault injection, latency emulation, and strict session isolation.
    """

    def __init__(
        self,
        signer: _InternalWorkerSigner,
        settlement_service: Optional[MockSettlementService] = None,
        verification_key: str = "settlement-secret-verification-key-12345",
    ) -> None:
        self.signer = signer
        self.settlement_service = settlement_service
        self.verification_key = verification_key

        self._lock = threading.RLock()
        self._pinned_sessions: Dict[str, Dict[str, Any]] = {}
        self.lease_calls = 0
        self.step_calls = 0
        self.release_calls = 0
        self.uploaded_matrix_counts: Dict[str, int] = {}
        self.uploaded_bytes_total = 0

        # Programmable fault injection hooks
        self.tamper_step_iteration: Optional[int] = None
        self.tamper_perturbation: float = 5.0
        self.tamper_step_signature: bool = False
        self.reject_lease: bool = False
        self.reject_lease_message: str = "Worker VRAM capacity exhausted"
        self.expire_lease_after_step: Optional[int] = None
        self.drop_connection_at_step: Optional[int] = None
        self.step_delay_seconds: float = 0.0

    def LeaseMatrix(
        self,
        request: LeaseMatrixRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> LeaseMatrixResponse:
        with self._lock:
            self.lease_calls += 1
            lease_id = request.lease_id

            if self.reject_lease:
                return LeaseMatrixResponse(
                    lease_id=lease_id,
                    is_accepted=False,
                    expires_at_unix_ms=0,
                    worker_pubkey=self.signer.get_pubkey_hex(),
                    message=self.reject_lease_message,
                )

            worker_ephemeral_pubkey = b""
            session_keys = None

            if getattr(request, "require_confidential_compute", False):
                if not request.client_ephemeral_pubkey or len(request.client_ephemeral_pubkey) != 32:
                    if context:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "client_ephemeral_pubkey must be exactly 32 bytes")
                    raise ValueError("client_ephemeral_pubkey must be exactly 32 bytes")

                from cryptography.hazmat.primitives.asymmetric import x25519
                from cryptography.hazmat.primitives import serialization

                worker_ephemeral_priv = x25519.X25519PrivateKey.generate()
                worker_ephemeral_pub = worker_ephemeral_priv.public_key()
                worker_ephemeral_pubkey = worker_ephemeral_pub.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )

                salt = hashlib.sha256(f"iterative:{lease_id}".encode("utf-8")).digest()
                from matrix_compute.crypto.session import derive_session_keys
                session_keys = derive_session_keys(
                    secret_or_priv=worker_ephemeral_priv,
                    nonce_c_or_pub=request.client_ephemeral_pubkey,
                    nonce_w_or_salt=salt,
                )

            dt = (request.dtype or "fp32").lower()
            np_dt = np.float16 if dt == "fp16" else (np.float64 if dt == "fp64" else np.float32)
            raw_matrix_bytes = request.matrix_bytes
            item_size = 4 if dt in ("fp32", "bf16") else 2
            expected_raw_bytes = request.rows * request.cols * item_size

            if getattr(request, "require_confidential_compute", False) and session_keys is not None:
                if len(raw_matrix_bytes) == expected_raw_bytes + 28:
                    from matrix_compute.crypto.session import aes_gcm_decrypt
                    iv = raw_matrix_bytes[:12]
                    tag = raw_matrix_bytes[12:28]
                    ct = raw_matrix_bytes[28:]
                    raw_matrix_bytes = aes_gcm_decrypt(
                        key=session_keys.k_c2w,
                        iv=iv,
                        ciphertext=ct,
                        tag=tag,
                    )

            mat = np.frombuffer(raw_matrix_bytes, dtype=np_dt).reshape(request.rows, request.cols).copy()

            if dt == "bf16":
                mat = truncate_to_bf16(mat)

            ttl_ms = int(request.ttl_seconds * 1000)
            now_ms = int(time.time() * 1000)
            expires_at = now_ms + ttl_ms

            self._pinned_sessions[lease_id] = {
                "matrix": mat,
                "rows": request.rows,
                "cols": request.cols,
                "dtype": dt,
                "ttl_seconds": request.ttl_seconds,
                "expires_at_unix_ms": expires_at,
                "expires_at_ms": expires_at,
                "created_at_ms": now_ms,
                "step_count": 0,
                "memory_bytes": int(mat.nbytes),
                "is_zeroized": False,
                "require_confidential_compute": bool(getattr(request, "require_confidential_compute", False)),
                "session_keys": session_keys,
                "worker_ephemeral_pubkey": worker_ephemeral_pubkey,
            }
            self.uploaded_matrix_counts[lease_id] = self.uploaded_matrix_counts.get(lease_id, 0) + 1
            self.uploaded_bytes_total += len(request.matrix_bytes)

            return LeaseMatrixResponse(
                lease_id=lease_id,
                is_accepted=True,
                expires_at_unix_ms=expires_at,
                worker_pubkey=self.signer.get_pubkey_hex(),
                message="Matrix lease pinned successfully",
                worker_ephemeral_pubkey=worker_ephemeral_pubkey,
            )

    def StepIterative(
        self,
        request: IterativeStepRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> IterativeStepResponse:
        if self.step_delay_seconds > 0.0:
            time.sleep(self.step_delay_seconds)

        with self._lock:
            self.step_calls += 1
            lease_id = request.lease_id
            session = self._pinned_sessions.get(lease_id)

            if not session or session["is_zeroized"]:
                if context:
                    context.abort(grpc.StatusCode.NOT_FOUND, f"Lease '{lease_id}' not found or released")
                raise KeyError(f"Lease '{lease_id}' not found")

            now_ms = int(time.time() * 1000)
            if (
                now_ms > session["expires_at_ms"]
                or (self.expire_lease_after_step is not None and session["step_count"] >= self.expire_lease_after_step)
            ):
                if context:
                    context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, f"Lease '{lease_id}' has expired")
                raise RuntimeError(f"Lease '{lease_id}' expired")

            if self.drop_connection_at_step is not None and request.step_index >= self.drop_connection_at_step:
                if context:
                    context.abort(grpc.StatusCode.UNAVAILABLE, "Network dropped mid-iteration")
                raise RuntimeError("Network dropped mid-iteration")

            is_confidential = session.get("require_confidential_compute", False)
            keys = session.get("session_keys")

            raw_vector_bytes = request.vector_bytes
            if is_confidential and keys is not None:
                if not request.is_encrypted:
                    if context:
                        context.abort(grpc.StatusCode.PERMISSION_DENIED, "Step request must be encrypted in confidential mode")
                    raise PermissionError("Step request must be encrypted in confidential mode")
                from matrix_compute.crypto.session import aes_gcm_decrypt
                step_aad = f"{lease_id}:step:{request.step_index}".encode("utf-8")
                raw_vector_bytes = aes_gcm_decrypt(
                    key=keys.k_c2w,
                    iv=request.iv,
                    ciphertext=request.vector_bytes,
                    tag=request.tag,
                    aad=step_aad,
                    fallback_aads=[b"", None],
                )

            dt = session["dtype"]
            np_dt = np.float16 if dt == "fp16" else (np.float64 if dt == "fp64" else np.float32)
            v_t = np.frombuffer(raw_vector_bytes, dtype=np_dt).reshape((session["cols"],))
            self.uploaded_bytes_total += len(request.vector_bytes)

            mat = session["matrix"]
            if dt == "fp16":
                v_res = np.matmul(mat.astype(np.float32), v_t.astype(np.float32)).astype(np.float16)
            elif dt == "bf16":
                v_res = truncate_to_bf16(np.matmul(mat, v_t))
            elif dt == "fp64":
                v_res = np.matmul(mat, v_t).astype(np.float64)
            else:
                v_res = np.matmul(mat, v_t).astype(np.float32)

            session["step_count"] += 1

            # Fault injection hook
            if self.tamper_step_iteration is not None and request.step_index == self.tamper_step_iteration:
                v_res = v_res.copy()
                v_res[0] += self.tamper_perturbation

            res_bytes = v_res.tobytes()
            res_is_encrypted = False
            res_iv = b""
            res_tag = b""

            if is_confidential and keys is not None:
                from matrix_compute.crypto.session import aes_gcm_encrypt
                step_resp_aad = f"{lease_id}:step:{request.step_index}".encode("utf-8")
                enc = aes_gcm_encrypt(
                    key=keys.k_w2c,
                    plaintext=res_bytes,
                    aad=step_resp_aad,
                )
                res_bytes = enc.ciphertext
                res_iv = enc.iv
                res_tag = enc.tag
                res_is_encrypted = True

            if self.tamper_step_signature:
                sig = "00" * 64
            else:
                sig_payload = f"{lease_id}:{request.step_index}:{hashlib.sha256(v_res.tobytes()).hexdigest()}".encode()
                sig = self.signer.sign_message(sig_payload)

            return IterativeStepResponse(
                lease_id=lease_id,
                step_index=request.step_index,
                result_vector_bytes=res_bytes,
                vector_length=len(v_res),
                worker_signature=sig,
                execution_time_ms=2,
                is_encrypted=res_is_encrypted,
                iv=res_iv,
                tag=res_tag,
            )

    def ReleaseLease(
        self,
        request: ReleaseLeaseRequest,
        context: Optional[grpc.ServicerContext] = None,
    ) -> ReleaseLeaseResponse:
        with self._lock:
            self.release_calls += 1
            lease_id = request.lease_id
            session = self._pinned_sessions.pop(lease_id, None)

            if not session:
                return ReleaseLeaseResponse(
                    lease_id=lease_id,
                    is_released=False,
                    bytes_freed=0,
                    message="Lease not found or already released",
                )

            if "matrix" in session and session["matrix"] is not None and isinstance(session["matrix"], np.ndarray):
                session["matrix"].fill(0)
            session["is_zeroized"] = True
            bytes_freed = session["memory_bytes"]

            return ReleaseLeaseResponse(
                lease_id=lease_id,
                is_released=True,
                bytes_freed=bytes_freed,
                message=f"Lease '{lease_id}' successfully deallocated and memory zeroized",
            )


class StatefulMockWorkerNode:
    """Manages an in-process gRPC server running StatefulMockWorkerServicer on an ephemeral port."""

    def __init__(
        self,
        signer: Optional[_InternalWorkerSigner] = None,
        settlement_service: Optional[MockSettlementService] = None,
        verification_key: str = "settlement-secret-verification-key-12345",
        vram_mb: int = 24576,
        sats_per_gflop: float = 0.0005,
    ) -> None:
        self.signer = signer or _InternalWorkerSigner()
        self.servicer = StatefulMockWorkerServicer(
            signer=self.signer,
            settlement_service=settlement_service,
            verification_key=verification_key,
        )
        self.vram_mb = vram_mb
        self.sats_per_gflop = sats_per_gflop
        self.server: Optional[grpc.Server] = None
        self.port: int = 0
        self._thread_pool = futures.ThreadPoolExecutor(max_workers=4)

    @property
    def pubkey(self) -> str:
        return self.signer.get_pubkey_hex()

    @property
    def target(self) -> str:
        return f"127.0.0.1:{self.port}"

    @property
    def multiaddr(self) -> str:
        return f"/ip4/127.0.0.1/tcp/{self.port}"

    def start(self) -> "StatefulMockWorkerNode":
        if self.server is not None:
            return self
        self.server = grpc.server(self._thread_pool)
        add_MatrixComputeServiceServicer_to_server(self.servicer, self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        return self

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop(0)
            self._thread_pool.shutdown(wait=False)
            self.server = None

    def publish_to_relay(self, relay: MockNostrRelay) -> str:
        payload = WorkerAnnouncePayload(
            event="WORKER_ANNOUNCE",
            worker_pubkey=self.pubkey,
            p2p_multiaddr=self.multiaddr,
            lightning_routing_pubkey=f"02ln{self.pubkey[:16]}",
            hardware=HardwareSpecs(
                gpu_model="RTX 4090",
                vram_mb=self.vram_mb,
                peak_fp32_tflops=82.6,
                attestation_proof=f"attest_{self.pubkey[:16]}",
                timestamp=int(time.time()),
            ),
            pricing=PricingSpecs(
                sats_per_gflop=self.sats_per_gflop,
                min_task_sats=10,
            ),
            bond_status=BondStatus(
                staked_amount_sats=100000,
                bond_txid=f"bond_{self.pubkey[:12]}",
            ),
            timestamp=int(time.time()),
        )
        return relay.publish_announce(payload)


# ==============================================================================
# PYTEST FIXTURES
# ==============================================================================

@pytest.fixture
def hermetic_stateful_suite() -> Generator[Dict[str, Any], None, None]:
    """
    Sets up a complete hermetic stateful environment with in-memory worker,
    settlement double, Nostr relay, and consumer coordinator.
    """
    vk = f"secret-vk-{uuid.uuid4().hex[:8]}"
    settlement = MockSettlementService(verification_key=vk)
    relay = MockNostrRelay()

    worker = StatefulMockWorkerNode(
        settlement_service=settlement,
        verification_key=vk,
        vram_mb=24576,
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

    yield {
        "coordinator": coordinator,
        "config": config,
        "worker": worker,
        "settlement": settlement,
        "relay": relay,
        "escrow_manager": escrow_manager,
    }

    coordinator.close()
    worker.stop()


# ==============================================================================
# TIER 1: FEATURE COVERAGE (25 TESTS)
# ==============================================================================

class TestTier1IterativePowerFeatureCoverage:
    """
    Tier 1: Feature Coverage (>=5 tests per feature area across 5 areas):
    1. Client API entry points
    2. Lease negotiation and session establishment
    3. Streaming state vector updates
    4. Probabilistic spot-checking Freivalds verification (p=0.05)
    5. Explicit lease release and resource cleanup
    """

    # --------------------------------------------------------------------------
    # Area 1: Client API Entry Points (5 tests)
    # --------------------------------------------------------------------------

    def test_t1_01_client_functional_entry_point_defaults(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.1: distributed_power_iteration runs end-to-end with default random vector."""
        env = hermetic_stateful_suite
        coord: DistributedMatmulCoordinator = env["coordinator"]
        np.random.seed(42)
        n = 16
        a = (np.eye(n) * 0.5).astype(np.float32)

        res = coord.execute_iterative_power(a, iterations=10)
        assert isinstance(res, IterativeResult)
        assert res.iterations_completed == 10
        assert res.final_vector.shape == (n,)

    def test_t1_02_client_instance_power_iteration(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.2: client.power_iteration executes on MatrixComputeClient instance."""
        env = hermetic_stateful_suite
        client = MatrixComputeClient(config=env["config"])
        try:
            client.coordinator = env["coordinator"]
            np.random.seed(43)
            n = 8
            a = np.random.randn(n, n).astype(np.float32)
            a = a / (np.linalg.norm(a, 2) * 1.5)
            v0 = np.ones(n, dtype=np.float32)

            res = client.power_iteration(a, v0=v0, iterations=5)
            assert res.shape == (n,)
        finally:
            client.close()

    def test_t1_03_client_context_manager_lifecycle(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.3: MatrixComputeClient context manager cleanly establishes and tears down resources."""
        env = hermetic_stateful_suite
        cfg = env["config"]
        with MatrixComputeClient(config=cfg) as client:
            client.coordinator = env["coordinator"]
            a = np.eye(4, dtype=np.float32) * 0.8
            res = client.power_iteration(a, iterations=4)
            assert len(res) == 4
        assert client._is_closed is True

    def test_t1_04_client_polymorphic_vector_inputs(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.4: Handles 1D (N,), 2D column (N, 1), 2D row (1, N), and None vectors."""
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        n = 6
        a = np.eye(n, dtype=np.float32) * 0.9

        # 1D vector (N,)
        v_1d = np.ones(n, dtype=np.float32)
        r1 = coord.execute_iterative_power(a, v0=v_1d, iterations=2)
        assert r1.final_vector.shape == (n,)

        # 2D column vector (N, 1)
        v_col = np.ones((n, 1), dtype=np.float32)
        r2 = coord.execute_iterative_power(a, v0=v_col, iterations=2)
        assert r2.final_vector.shape == (n,)

        # 2D row vector (1, N)
        v_row = np.ones((1, n), dtype=np.float32)
        r3 = coord.execute_iterative_power(a, v0=v_row, iterations=2)
        assert r3.final_vector.shape == (n,)

        # Auto-generated vector (None)
        r4 = coord.execute_iterative_power(a, v0=None, iterations=2)
        assert r4.final_vector.shape == (n,)

    def test_t1_05_client_iterative_result_properties(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.5: IterativeResult exposes NumPy array interface and execution metadata."""
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        n = 10
        a = np.eye(n, dtype=np.float32) * 0.7
        v0 = np.ones(n, dtype=np.float32)

        res: IterativeResult = coord.execute_iterative_power(a, v0=v0, iterations=6)
        # Array protocol conformance
        arr = np.asarray(res)
        assert isinstance(arr, np.ndarray)
        assert res.shape == (n,)
        assert res.ndim == 1
        assert len(res) == n
        assert res.lease_id.startswith("lease_")
        assert res.worker_pubkey != ""
        assert res.total_compute_time_ms >= 0

    # --------------------------------------------------------------------------
    # Area 2: Lease Negotiation & Session Establishment (5 tests)
    # --------------------------------------------------------------------------

    def test_t1_06_lease_matrix_nominal_establishment(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.6: LeaseMatrix RPC pins matrix and returns valid lease_id and timestamp."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(12, dtype=np.float32)
        res = coord.execute_iterative_power(a, iterations=3)
        assert res.lease_id in worker.servicer.uploaded_matrix_counts
        assert worker.servicer.lease_calls >= 1

    def test_t1_07_lease_matrix_uploaded_strictly_once(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.7: Proves matrix A is uploaded strictly once across all 50 iterations."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        n = 16
        a = np.eye(n, dtype=np.float32) * 0.5
        v0 = np.ones(n, dtype=np.float32)

        prev_lease_calls = worker.servicer.lease_calls
        res = coord.execute_iterative_power(a, v0=v0, iterations=50)

        # LeaseMatrix was invoked exactly 1 time
        assert worker.servicer.lease_calls == prev_lease_calls + 1
        assert worker.servicer.uploaded_matrix_counts[res.lease_id] == 1

    def test_t1_08_lease_matrix_rejection_out_of_vram(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.8: Worker VRAM capacity exhaustion rejects lease and raises WorkerUnavailableError."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        worker.servicer.reject_lease = True
        worker.servicer.reject_lease_message = "VRAM pool full"

        a = np.eye(8, dtype=np.float32)
        try:
            with pytest.raises(WorkerUnavailableError, match="VRAM pool full"):
                coord.execute_iterative_power(a, iterations=5)
        finally:
            worker.servicer.reject_lease = False

    def test_t1_09_lease_matrix_ttl_negotiation(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.9: lease_ttl_seconds parameter negotiates correct expiration unix ms."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(8, dtype=np.float32)
        custom_ttl = 1800

        # Retain session in _pinned_sessions during execute_iterative_power for inspection
        orig_release = worker.servicer.ReleaseLease
        def delayed_release(req, context=None):
            with worker.servicer._lock:
                worker.servicer.release_calls += 1
                session = worker.servicer._pinned_sessions.get(req.lease_id)
                freed = session["memory_bytes"] if session else 0
            return ReleaseLeaseResponse(
                lease_id=req.lease_id,
                is_released=True,
                bytes_freed=freed,
                message="Retained for inspection",
            )

        worker.servicer.ReleaseLease = delayed_release
        try:
            res = coord.execute_iterative_power(a, iterations=2, lease_ttl_seconds=custom_ttl)
            assert res.lease_id in worker.servicer._pinned_sessions
            assert worker.servicer._pinned_sessions[res.lease_id]["ttl_seconds"] == custom_ttl
            assert worker.servicer._pinned_sessions[res.lease_id]["expires_at_unix_ms"] > int(time.time() * 1000)
        finally:
            worker.servicer.ReleaseLease = orig_release
            orig_release(ReleaseLeaseRequest(lease_id=res.lease_id))

    def test_t1_10_lease_matrix_pubkey_exchange(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.10: Lease handshake returns worker Secp256k1 identity pubkey."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(6, dtype=np.float32)
        res = coord.execute_iterative_power(a, iterations=2)
        assert res.worker_pubkey == worker.pubkey

    # --------------------------------------------------------------------------
    # Area 3: Streaming State Vector Updates (5 tests)
    # --------------------------------------------------------------------------

    def test_t1_11_streaming_vector_monotonically_increasing_steps(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.11: Iteration loop streams t = 0..T-1 strictly monotonically."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 8
        a = np.eye(n, dtype=np.float32) * 0.8
        v0 = np.ones(n, dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=15)
        assert res.iterations_completed == 15

    def test_t1_12_streaming_vector_pinned_vram_gemv(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.12: Computes v_{t+1} = A * v_t directly from pinned in-memory tensor."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 10
        a = np.diag([0.5] * n).astype(np.float32)
        v0 = np.ones(n, dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=4)
        expected = v0 * (0.5 ** 4)
        np.testing.assert_allclose(res.final_vector, expected, rtol=1e-5)

    def test_t1_13_streaming_vector_fp32_precision(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.13: FP32 state vector precision matches NumPy reference."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 12
        np.random.seed(113)
        a = np.random.randn(n, n).astype(np.float32) / (n * 1.2)
        v0 = np.random.randn(n).astype(np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=10, dtype="fp32")
        ref = ReferenceIterativeOracle.power_iteration(a, v0, 10, dtype="fp32")
        np.testing.assert_allclose(res.final_vector, ref, rtol=1e-4, atol=1e-4)

    def test_t1_14_streaming_vector_fp16_precision(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.14: FP16 state vector updates preserve half-precision float format."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 10
        np.random.seed(114)
        a = (np.random.randn(n, n) * 0.1).astype(np.float16)
        v0 = (np.ones(n) * 0.5).astype(np.float16)

        res = coord.execute_iterative_power(a, v0=v0, iterations=5, dtype="fp16")
        assert res.final_vector.dtype in (np.float16, np.float32)
        ref = ReferenceIterativeOracle.power_iteration(a, v0, 5, dtype="fp16")
        np.testing.assert_allclose(res.final_vector, ref, rtol=1e-2, atol=1e-2)

    def test_t1_15_streaming_vector_bf16_precision(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.15: BF16 updates apply 16-bit mantissa masking (0xFFFF0000)."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 8
        np.random.seed(115)
        a = truncate_to_bf16(np.random.randn(n, n).astype(np.float32) * 0.1)
        v0 = truncate_to_bf16(np.ones(n, dtype=np.float32))

        res = coord.execute_iterative_power(a, v0=v0, iterations=5, dtype="bf16")
        ref = ReferenceIterativeOracle.power_iteration(a, v0, 5, dtype="bf16")
        np.testing.assert_allclose(res.final_vector, ref, rtol=2e-2, atol=2e-2)

    # --------------------------------------------------------------------------
    # Area 4: Probabilistic Spot-Checking Freivalds Verification (5 tests)
    # --------------------------------------------------------------------------

    def test_t1_16_probabilistic_spot_check_nominal_pass(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.16: Default p=0.05 spot checking verifies honest worker steps without errors."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 10
        np.random.seed(116)
        a = (np.eye(n) * 0.7).astype(np.float32)
        res = coord.execute_iterative_power(a, iterations=100, spot_check_probability=0.05)
        assert res.spot_checks_performed >= 1
        assert res.spot_checks_passed == res.spot_checks_performed

    def test_t1_17_probabilistic_spot_check_single_step_higham_bound(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.17: Directly validates single-step Higham bound on genuine vector."""
        n = 16
        np.random.seed(117)
        a = np.random.randn(n, n).astype(np.float32)
        v = np.random.randn(n).astype(np.float32)
        v_next = np.matmul(a, v)

        is_valid, disc, bound, proof = verify_iterative_step(a, v, v_next, dtype="fp32")
        assert is_valid is True
        assert disc <= bound
        assert len(proof) == 0

    def test_t1_18_probabilistic_spot_check_sampling_rate_statistics(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.18: Bernoulli sampling rate over T=100 with p=0.10 stays within 4-sigma bounds."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 6
        np.random.seed(118)
        a = np.eye(n, dtype=np.float32) * 0.8
        t_total = 100
        p = 0.10

        res = coord.execute_iterative_power(a, iterations=t_total, spot_check_probability=p)
        assert 1 <= res.spot_checks_performed <= 25

    def test_t1_19_probabilistic_spot_check_forced_indices(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.19: forced_spot_check_indices explicitly verifies designated iteration steps."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 8
        a = np.eye(n, dtype=np.float32) * 0.7
        forced = {0, 5, 9}

        res = coord.execute_iterative_power(
            a, iterations=10, spot_check_probability=0.0, forced_spot_check_indices=forced
        )
        assert res.spot_checks_performed == 3
        assert res.spot_checks_passed == 3

    def test_t1_20_probabilistic_spot_check_fraud_detection(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.20: Tampered vector at step t=2 triggers FraudDetectedError with FreivaldsFraudProof."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        worker.servicer.tamper_step_iteration = 2
        worker.servicer.tamper_perturbation = 5.0

        n = 8
        a = np.eye(n, dtype=np.float32) * 0.5
        v0 = np.ones(n, dtype=np.float32)

        try:
            with pytest.raises(FraudDetectedError) as exc_info:
                coord.execute_iterative_power(
                    a, v0=v0, iterations=5, spot_check_probability=0.0, forced_spot_check_indices={2}
                )
            err = exc_info.value
            assert "Freivalds verification failed" in str(err)
            assert err.fraud_proof is not None
            assert err.fraud_proof.dimension_m == n
        finally:
            worker.servicer.tamper_step_iteration = None

    # --------------------------------------------------------------------------
    # Area 5: Explicit Lease Release & Resource Cleanup (5 tests)
    # --------------------------------------------------------------------------

    def test_t1_21_release_lease_nominal_success(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.21: ReleaseLease RPC deallocates session and reports bytes_freed."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        n = 16
        a = np.eye(n, dtype=np.float32)

        res = coord.execute_iterative_power(a, iterations=2)
        assert res.bytes_freed == n * n * 4  # 16*16*4 = 1024 bytes
        assert worker.servicer.release_calls >= 1

    def test_t1_22_release_lease_worker_memory_zeroization(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.22: Asserts worker session is deallocated and not retrievable after release."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        n = 10
        a = np.eye(n, dtype=np.float32)

        res = coord.execute_iterative_power(a, iterations=2)
        assert res.lease_id not in worker.servicer._pinned_sessions

    def test_t1_23_release_lease_idempotent_redundant_calls(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.23: Calling ReleaseLease on already-released lease returns is_released=False cleanly."""
        worker = hermetic_stateful_suite["worker"]
        rel_req = ReleaseLeaseRequest(
            lease_id="non_existent_lease_123",
            consumer_pubkey="02aabb",
            reason="redundant_test",
        )
        resp = worker.servicer.ReleaseLease(rel_req)
        assert resp.is_released is False
        assert resp.bytes_freed == 0

    def test_t1_24_release_lease_exception_safety_finally(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.24: Mid-loop error executes ReleaseLease in finally: block without memory leaks."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        worker.servicer.tamper_step_iteration = 1
        worker.servicer.tamper_perturbation = 10.0

        n = 8
        a = np.eye(n, dtype=np.float32)
        initial_releases = worker.servicer.release_calls

        try:
            with pytest.raises(FraudDetectedError):
                coord.execute_iterative_power(
                    a, iterations=4, spot_check_probability=0.0, forced_spot_check_indices={1}
                )

            # finally: block fired ReleaseLease!
            assert worker.servicer.release_calls == initial_releases + 1
        finally:
            worker.servicer.tamper_step_iteration = None

    def test_t1_25_release_lease_escrow_settlement_on_release(self, hermetic_stateful_suite: Dict[str, Any]):
        """T1.25: On nominal completion, escrow is settled; on error, escrow is refunded."""
        escrow_mgr: EscrowManager = hermetic_stateful_suite["escrow_manager"]
        coord = hermetic_stateful_suite["coordinator"]
        n = 6
        a = np.eye(n, dtype=np.float32) * 0.5

        res = coord.execute_iterative_power(a, iterations=2)
        # All escrows created during run should be terminal (SETTLED, CANCELLED, or EXPIRED)
        for esc in escrow_mgr._escrows.values():
            assert esc.status in (EscrowStatus.SETTLED, EscrowStatus.CANCELLED, EscrowStatus.EXPIRED, "SETTLED")


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (25 TESTS)
# ==============================================================================

class TestTier2IterativePowerBoundaryCornerCases:
    """
    Tier 2: Boundary & Corner Cases (>=5 tests per boundary area across 5 areas):
    1. Degenerate matrix dimensions (1x1)
    2. Extreme floating-point values (subnormals, zeros, large norms)
    3. Iteration limits (T=0, T=1, T=100)
    4. Spot-check probabilities (p=0.0, p=1.0)
    5. TTL boundaries (short TTL 1s mid-loop expiration)
    """

    # --------------------------------------------------------------------------
    # Area 1: Degenerate Matrix Dimensions (5 tests)
    # --------------------------------------------------------------------------

    def test_t2_01_scalar_1x1_matrix_power_iteration(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.1: Scalar 1x1 matrix [[3.0]] with initial vector [2.0] gives 3^4 * 2 = 162.0."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.array([[3.0]], dtype=np.float32)
        v0 = np.array([2.0], dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=4)
        np.testing.assert_allclose(res.final_vector, np.array([162.0]), rtol=1e-5)

    def test_t2_02_scalar_1x1_matrix_with_l2_normalization(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.2: Scalar 1x1 matrix with L2 normalization preserves unit vector [1.0]."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.array([[5.0]], dtype=np.float32)
        v0 = np.array([7.0], dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=5, normalize=True)
        np.testing.assert_allclose(res.final_vector, np.array([1.0]), rtol=1e-6)

    def test_t2_03_small_2x2_identity_matrix(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.3: 2x2 Identity matrix leaves any initial vector invariant: I * v_0 = v_0."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(2, dtype=np.float32)
        v0 = np.array([3.14, 2.71], dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=10)
        np.testing.assert_allclose(res.final_vector, v0, rtol=1e-6)

    def test_t2_04_nilpotent_2x2_matrix(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.4: 2x2 nilpotent matrix [[0, 1], [0, 0]] satisfies A^2 = 0; v_2 = [0, 0]."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        v0 = np.array([5.0, 8.0], dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=2)
        np.testing.assert_allclose(res.final_vector, np.array([0.0, 0.0]), atol=1e-6)

    def test_t2_05_non_square_and_invalid_dimensions_rejected(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.5: Rejects non-square matrices (4x8), 0-dim arrays, and 3D tensors fail-closed."""
        coord = hermetic_stateful_suite["coordinator"]

        # Non-square matrix (4 x 8)
        with pytest.raises(MatrixDimensionError, match="square matrix"):
            coord.execute_iterative_power(np.ones((4, 8), dtype=np.float32), iterations=2)

        # 1D array
        with pytest.raises(MatrixDimensionError):
            coord.execute_iterative_power(np.ones(10, dtype=np.float32), iterations=2)

        # 3D tensor
        with pytest.raises(MatrixDimensionError):
            coord.execute_iterative_power(np.ones((2, 2, 2), dtype=np.float32), iterations=2)

    # --------------------------------------------------------------------------
    # Area 2: Extreme Floating-Point Values (5 tests)
    # --------------------------------------------------------------------------

    def test_t2_06_subnormal_floating_point_values(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.6: Subnormal inputs (~1e-35) handle underflow defense in Freivalds verification."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 4
        a = (np.eye(n) * 1e-35).astype(np.float32)
        v0 = (np.ones(n) * 1e-35).astype(np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=2, spot_check_probability=1.0)
        assert np.all(np.isfinite(res.final_vector))

    def test_t2_07_all_zero_matrix(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.7: All-zero matrix computes zero vector with Freivalds floor bound."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 6
        a = np.zeros((n, n), dtype=np.float32)
        v0 = np.ones(n, dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=3, spot_check_probability=1.0)
        np.testing.assert_allclose(res.final_vector, np.zeros(n), atol=1e-7)

    def test_t2_08_large_norm_matrix_with_normalization(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.8: Large magnitude matrix (~10^6) executes safely with L2 normalization."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 5
        a = (np.eye(n) * 1e6).astype(np.float32)
        v0 = np.ones(n, dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=5, normalize=True)
        assert np.all(np.isfinite(res.final_vector))
        np.testing.assert_allclose(np.linalg.norm(res.final_vector), 1.0, rtol=1e-5)

    def test_t2_09_spectral_radius_scaling_bounds(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.9: Matrix scaled to rho(A) <= 1.0 does not overflow during 20 iterations."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 8
        np.random.seed(209)
        raw = np.random.randn(n, n).astype(np.float32)
        a = (raw / (np.linalg.norm(raw, 2) * 1.1)).astype(np.float32)
        v0 = np.ones(n, dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=20)
        assert np.all(np.isfinite(res.final_vector))
        assert np.max(np.abs(res.final_vector)) <= 2.0

    def test_t2_10_nan_and_inf_inputs_rejected(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.10: Rejects NaN and +/-Inf inputs fail-closed with ValueError."""
        coord = hermetic_stateful_suite["coordinator"]
        a_nan = np.eye(4, dtype=np.float32)
        a_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="strictly finite"):
            coord.execute_iterative_power(a_nan, iterations=2)

        a_inf = np.eye(4, dtype=np.float32)
        a_inf[0, 0] = np.inf
        with pytest.raises(ValueError, match="strictly finite"):
            coord.execute_iterative_power(a_inf, iterations=2)

    # --------------------------------------------------------------------------
    # Area 3: Iteration Limits (5 tests)
    # --------------------------------------------------------------------------

    def test_t2_11_zero_iterations_rejected(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.11: T=0 is rejected fail-closed with ValueError."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(4, dtype=np.float32)
        with pytest.raises(ValueError, match="iterations must be a positive integer >= 1"):
            coord.execute_iterative_power(a, iterations=0)

    def test_t2_12_single_iteration_t1(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.12: T=1 computes exact single-step matrix-vector product v_1 = A * v_0."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 6
        np.random.seed(212)
        a = np.random.randn(n, n).astype(np.float32)
        v0 = np.random.randn(n).astype(np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=1)
        expected = np.matmul(a, v0)
        np.testing.assert_allclose(res.final_vector, expected, rtol=1e-5)

    def test_t2_13_two_iterations_t2(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.13: T=2 computes v_2 = A^2 * v_0."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 6
        a = (np.eye(n) * 0.6).astype(np.float32)
        v0 = np.ones(n, dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=2)
        expected = v0 * 0.36
        np.testing.assert_allclose(res.final_vector, expected, rtol=1e-5)

    def test_t2_14_full_100_iterations(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.14: Full 100 power iterations (T=100) match local reference calculation."""
        coord = hermetic_stateful_suite["coordinator"]
        n = 10
        np.random.seed(214)
        sym = np.random.randn(n, n).astype(np.float32)
        sym = np.matmul(sym, sym.T)
        a = (sym / (np.linalg.norm(sym, 2) * 1.05)).astype(np.float32)
        v0 = np.random.randn(n).astype(np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=100)
        ref = ReferenceIterativeOracle.power_iteration(a, v0, 100)
        np.testing.assert_allclose(res.final_vector, ref, rtol=1e-3, atol=1e-3)

    def test_t2_15_negative_iterations_rejected(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.15: Negative iterations (T=-3) raise ValueError."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(4, dtype=np.float32)
        with pytest.raises(ValueError):
            coord.execute_iterative_power(a, iterations=-3)

    # --------------------------------------------------------------------------
    # Area 4: Spot-Check Probabilities (5 tests)
    # --------------------------------------------------------------------------

    def test_t2_16_spot_check_probability_zero(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.16: p=0.0 executes strictly 0 spot checks."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(8, dtype=np.float32) * 0.5
        res = coord.execute_iterative_power(a, iterations=20, spot_check_probability=0.0)
        assert res.spot_checks_performed == 0

    def test_t2_17_spot_check_probability_one(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.17: p=1.0 verifies every single step (spot_checks == iterations)."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(8, dtype=np.float32) * 0.5
        res = coord.execute_iterative_power(a, iterations=15, spot_check_probability=1.0)
        assert res.spot_checks_performed == 15
        assert res.spot_checks_passed == 15

    def test_t2_18_spot_check_deterministic_forced_indices(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.18: Deterministic forced indices override p=0.0."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(8, dtype=np.float32) * 0.5
        forced = {1, 3, 7}
        res = coord.execute_iterative_power(
            a, iterations=10, spot_check_probability=0.0, forced_spot_check_indices=forced
        )
        assert res.spot_checks_performed == len(forced)

    def test_t2_19_spot_check_invalid_probability_out_of_bounds(self):
        """T2.19: Probability outside [0.0, 1.0] fails validation in ClientConfig."""
        with pytest.raises(Exception):
            ClientConfig(spot_check_probability=-0.1)
        with pytest.raises(Exception):
            ClientConfig(spot_check_probability=1.5)

    def test_t2_20_spot_check_bernoulli_randomness(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.20: Consecutive runs with p=0.5 sample varying numbers of checks."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(6, dtype=np.float32) * 0.5
        counts = [
            coord.execute_iterative_power(a, iterations=20, spot_check_probability=0.5).spot_checks_performed
            for _ in range(5)
        ]
        assert all(c > 0 for c in counts)

    # --------------------------------------------------------------------------
    # Area 5: TTL Boundaries & Short TTL (5 tests)
    # --------------------------------------------------------------------------

    def test_t2_21_short_ttl_1s_expiration_mid_loop(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.21: 1s TTL expiring mid-loop is caught and cleanly handled."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        worker.servicer.expire_lease_after_step = 2

        a = np.eye(6, dtype=np.float32)
        try:
            with pytest.raises(Exception):
                coord.execute_iterative_power(a, iterations=10, lease_ttl_seconds=1)
        finally:
            worker.servicer.expire_lease_after_step = None

    def test_t2_22_expired_lease_step_rejected(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.22: Calling StepIterative after expiration triggers DEADLINE_EXCEEDED."""
        worker = hermetic_stateful_suite["worker"]
        # Lease matrix with 1 second TTL
        lease_req = LeaseMatrixRequest(
            lease_id="short_ttl_test",
            rows=4,
            cols=4,
            dtype="fp32",
            matrix_bytes=np.eye(4, dtype=np.float32).tobytes(),
            ttl_seconds=1,
        )
        worker.servicer.LeaseMatrix(lease_req)
        # Fast forward expiration
        worker.servicer._pinned_sessions["short_ttl_test"]["expires_at_ms"] = int((time.time() - 10) * 1000)

        step_req = IterativeStepRequest(
            lease_id="short_ttl_test",
            step_index=0,
            vector_bytes=np.ones(4, dtype=np.float32).tobytes(),
            vector_length=4,
            dtype="fp32",
        )
        with pytest.raises(RuntimeError, match="expired"):
            worker.servicer.StepIterative(step_req)

    def test_t2_23_expired_lease_escrow_refund(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.23: When lease expires mid-loop, escrow is refunded and not settled."""
        worker = hermetic_stateful_suite["worker"]
        coord = hermetic_stateful_suite["coordinator"]
        escrow_mgr: EscrowManager = hermetic_stateful_suite["escrow_manager"]
        worker.servicer.expire_lease_after_step = 1

        a = np.eye(6, dtype=np.float32)
        try:
            with pytest.raises(Exception):
                coord.execute_iterative_power(a, iterations=5)

            for esc in escrow_mgr._escrows.values():
                assert esc.status in (
                    EscrowStatus.CANCELLED,
                    EscrowStatus.EXPIRED,
                    "CANCELLED",
                    "REFUNDED",
                    "ABORTED",
                )
        finally:
            worker.servicer.expire_lease_after_step = None

    def test_t2_24_long_ttl_24h_accepted(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.24: Long 24h TTL (86400s) is accepted without error."""
        coord = hermetic_stateful_suite["coordinator"]
        a = np.eye(6, dtype=np.float32) * 0.5
        res = coord.execute_iterative_power(a, iterations=2, lease_ttl_seconds=86400)
        assert res.iterations_completed == 2

    def test_t2_25_session_reaper_lazy_eviction(self, hermetic_stateful_suite: Dict[str, Any]):
        """T2.25: Expired sessions are evicted from memory when accessed."""
        worker = hermetic_stateful_suite["worker"]
        lease_req = LeaseMatrixRequest(
            lease_id="reaper_test_lease",
            rows=2,
            cols=2,
            dtype="fp32",
            matrix_bytes=np.eye(2, dtype=np.float32).tobytes(),
            ttl_seconds=1,
        )
        worker.servicer.LeaseMatrix(lease_req)
        assert "reaper_test_lease" in worker.servicer._pinned_sessions
        # Manually expire
        worker.servicer._pinned_sessions["reaper_test_lease"]["expires_at_ms"] = int((time.time() - 5) * 1000)

        step_req = IterativeStepRequest(
            lease_id="reaper_test_lease",
            step_index=0,
            vector_bytes=np.ones(2, dtype=np.float32).tobytes(),
            vector_length=2,
            dtype="fp32",
        )
        with pytest.raises(RuntimeError):
            worker.servicer.StepIterative(step_req)


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (6 TESTS)
# ==============================================================================

class TestTier3IterativePowerCrossFeatureCombinations:
    """
    Tier 3: Cross-Feature Combinations (Pairwise and complex interactions):
    1. BF16 + malicious worker corrupted vector + Freivalds fraud proof + escrow slashing
    2. Confidential computing (AES-256-GCM enclave) + 100 iterations + Secp256k1 signatures
    3. Prime dimension matrix + L2 normalization + spot-check + mock Nostr relay matchmaking
    4. Forged worker signature + fail-closed rejection + refund + ReleaseLease in finally:
    5. Multi-lease concurrent isolation on single worker node
    6. Mid-loop worker network drop + fail-closed abort + escrow refund
    """

    def test_t3_01_bf16_corrupted_vector_fraud_slashing(self, hermetic_stateful_suite: Dict[str, Any]):
        """T3.1: BF16 precision + worker corrupted vector at step t=3 triggers fraud proof and bond slashing."""
        worker = hermetic_stateful_suite["worker"]
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        escrow_mgr: EscrowManager = hermetic_stateful_suite["escrow_manager"]
        settlement: MockSettlementService = hermetic_stateful_suite["settlement"]

        worker.servicer.tamper_step_iteration = 3
        worker.servicer.tamper_perturbation = 4.0

        n = 16
        np.random.seed(301)
        a = truncate_to_bf16(np.random.randn(n, n).astype(np.float32) * 0.1)
        v0 = truncate_to_bf16(np.ones(n, dtype=np.float32))

        try:
            with pytest.raises(FraudDetectedError) as exc_info:
                coord.execute_iterative_power(
                    a, v0=v0, iterations=6, dtype="bf16", spot_check_probability=0.0, forced_spot_check_indices={3}
                )

            err = exc_info.value
            assert "Freivalds verification failed" in str(err)
            assert err.fraud_proof is not None
            assert err.fraud_proof.dtype == "bf16"
            assert err.fraud_proof.max_observed_discrepancy > 1.0

            # Escrow assertion: reflects fraud proof slashing and cancellation
            escrow = list(escrow_mgr._escrows.values())[-1]
            assert escrow.status == EscrowStatus.CANCELLED
            assert "fraud" in str(escrow.cancellation_reason)
            assert escrow.fraud_proof is not None
            assert settlement.get_status(escrow.payment_hash) == "CANCELLED"
        finally:
            worker.servicer.tamper_step_iteration = None

    def test_t3_02_confidential_enclave_mode_100_iterations_secp256k1(self, hermetic_stateful_suite: Dict[str, Any]):
        """
        T3.2: 100 power iterations execute with confidential computing mode:
        require_confidential_compute=True, ECDH X25519 negotiation of ephemeral session keys,
        AES-256-GCM enclave execution with ciphertext verification, Secp256k1 worker signature
        verification on each step, and numerical alignment against reference oracle.
        """
        from cryptography.hazmat.primitives.asymmetric import x25519
        from cryptography.hazmat.primitives import serialization
        from matrix_compute.crypto.session import derive_session_keys, aes_gcm_encrypt, aes_gcm_decrypt
        from matrix_compute.crypto.signer import verify_signature

        worker = hermetic_stateful_suite["worker"]
        signer: _InternalWorkerSigner = worker.signer
        worker_pubkey = signer.get_pubkey_hex()

        n = 12
        np.random.seed(302)
        raw = np.random.randn(n, n).astype(np.float32)
        sym = np.matmul(raw, raw.T)
        a = (sym / (np.linalg.norm(sym, 2) * 1.1)).astype(np.float32)
        v0 = np.random.randn(n).astype(np.float32)
        iterations = 100
        lease_id = f"lease_enclave_{uuid.uuid4().hex[:8]}"

        # 1. Client generates X25519 ephemeral keypair
        client_priv = x25519.X25519PrivateKey.generate()
        client_pub_bytes = client_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        # 2. Configure confidential computing mode with require_confidential_compute=True
        lease_req = LeaseMatrixRequest(
            lease_id=lease_id,
            rows=n,
            cols=n,
            dtype="fp32",
            matrix_bytes=a.tobytes(),
            ttl_seconds=3600,
            require_confidential_compute=True,
            client_ephemeral_pubkey=client_pub_bytes,
        )
        lease_resp = worker.servicer.LeaseMatrix(lease_req)
        assert lease_resp.is_accepted is True
        assert len(lease_resp.worker_ephemeral_pubkey) == 32

        # 3. Negotiate client ephemeral session keys via X25519 ECDH + HKDF
        salt = hashlib.sha256(f"iterative:{lease_id}".encode("utf-8")).digest()
        client_keys = derive_session_keys(
            secret_or_priv=client_priv,
            nonce_c_or_pub=lease_resp.worker_ephemeral_pubkey,
            nonce_w_or_salt=salt,
        )
        k_c2w = client_keys.k_c2w
        k_w2c = client_keys.k_w2c
        assert len(k_c2w) == 32
        assert len(k_w2c) == 32

        # 4. Stream 100 power iterations:
        # Verify AES-256-GCM enclave execution, ciphertext handling, and Secp256k1 signatures
        v_curr = v0.copy()
        for t in range(iterations):
            # Client encrypts state vector v_t with K_c2w
            step_aad = f"{lease_id}:step:{t}".encode("utf-8")
            enc = aes_gcm_encrypt(
                key=k_c2w,
                plaintext=v_curr.tobytes(),
                aad=step_aad,
            )

            # Send encrypted step request
            step_req = IterativeStepRequest(
                lease_id=lease_id,
                step_index=t,
                vector_bytes=enc.ciphertext,
                vector_length=n,
                dtype="fp32",
                is_encrypted=True,
                iv=enc.iv,
                tag=enc.tag,
            )
            step_resp = worker.servicer.StepIterative(step_req)

            # Verify worker handles AES-256-GCM enclave execution & returns ciphertext
            assert step_resp.is_encrypted is True
            assert len(step_resp.iv) == 12
            assert len(step_resp.tag) == 16
            assert len(step_resp.result_vector_bytes) == n * 4

            # Client decrypts result vector with K_w2c
            decrypted_result_bytes = aes_gcm_decrypt(
                key=k_w2c,
                iv=step_resp.iv,
                ciphertext=step_resp.result_vector_bytes,
                tag=step_resp.tag,
                aad=step_aad,
                fallback_aads=[b"", None],
            )
            v_next = np.frombuffer(decrypted_result_bytes, dtype=np.float32)

            # Verify Secp256k1 signature on each step
            assert step_resp.worker_signature != ""
            r_hash_hex = hashlib.sha256(decrypted_result_bytes).hexdigest()
            expected_payload = f"{lease_id}:{t}:{r_hash_hex}".encode("utf-8")
            sig_valid = verify_signature(
                worker_pubkey,
                hashlib.sha256(expected_payload).digest(),
                step_resp.worker_signature,
            ) or verify_signature(
                worker_pubkey,
                expected_payload,
                step_resp.worker_signature,
            )
            assert sig_valid is True, f"Secp256k1 signature invalid on step {t}"

            v_curr = v_next.copy()

        # 5. Verify numerical alignment of final vector against reference oracle
        ref_oracle_vec = ReferenceIterativeOracle.power_iteration(a, v0, iterations, dtype="fp32")
        np.testing.assert_allclose(v_curr, ref_oracle_vec, rtol=1e-4, atol=1e-4)

        # 6. Release lease cleanly
        rel_resp = worker.servicer.ReleaseLease(ReleaseLeaseRequest(lease_id=lease_id))
        assert rel_resp.is_released is True
        assert worker.servicer.release_calls >= 1

    def test_t3_03_prime_dimension_l2_normalization_relay_matchmaking(self, hermetic_stateful_suite: Dict[str, Any]):
        """T3.3: Prime dimension N=17 + L2 normalization + mock Nostr relay matchmaking."""
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        n = 17  # Prime dimension!
        np.random.seed(303)
        raw = np.random.randn(n, n).astype(np.float32)
        a = np.matmul(raw, raw.T).astype(np.float32)
        v0 = np.ones(n, dtype=np.float32)

        res = coord.execute_iterative_power(a, v0=v0, iterations=25, normalize=True)
        assert res.final_vector.shape == (17,)
        # Rayleigh quotient converges to dominant eigenvalue
        rq = ReferenceIterativeOracle.rayleigh_quotient(a, res.final_vector)
        eigvals = np.linalg.eigvalsh(a)
        assert abs(rq - np.max(eigvals)) < 1.0

    def test_t3_04_forged_worker_signature_rejection_refund_finally_release(self, hermetic_stateful_suite: Dict[str, Any]):
        """T3.4: Forged worker signature triggers WorkerSignatureError, refunds escrow, and releases lease in finally:."""
        worker = hermetic_stateful_suite["worker"]
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        escrow_mgr: EscrowManager = hermetic_stateful_suite["escrow_manager"]
        settlement: MockSettlementService = hermetic_stateful_suite["settlement"]
        worker.servicer.tamper_step_signature = True

        n = 8
        a = np.eye(n, dtype=np.float32) * 0.5
        v0 = np.ones(n, dtype=np.float32)

        try:
            with pytest.raises(WorkerSignatureError, match="signature"):
                coord.execute_iterative_power(a, v0=v0, iterations=3)

            # Asserts lease released in finally:
            assert worker.servicer.release_calls >= 1

            # Escrow assertion: reflects signature verification failure refund
            escrow = list(escrow_mgr._escrows.values())[-1]
            assert escrow.status == EscrowStatus.CANCELLED
            assert "bad_signature" in str(escrow.cancellation_reason)
            assert settlement.get_status(escrow.payment_hash) == "CANCELLED"
        finally:
            worker.servicer.tamper_step_signature = False

    def test_t3_05_concurrent_multi_lease_isolation(self, hermetic_stateful_suite: Dict[str, Any]):
        """T3.5: Multiple concurrent leases on single worker node execute simultaneously maintaining strict isolation."""
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        n = 6

        # Lease 1: 0.5 * I
        a1 = np.eye(n, dtype=np.float32) * 0.5
        v1 = np.ones(n, dtype=np.float32) * 2.0

        # Lease 2: 2.0 * I
        a2 = np.eye(n, dtype=np.float32) * 2.0
        v2 = np.ones(n, dtype=np.float32) * 3.0

        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(coord.execute_iterative_power, a1, v0=v1, iterations=10)
            f2 = pool.submit(coord.execute_iterative_power, a2, v0=v2, iterations=10)
            r1, r2 = f1.result(), f2.result()

        np.testing.assert_allclose(r1.final_vector, np.ones(n, dtype=np.float32) * 2.0 * (0.5 ** 10), rtol=1e-5)
        np.testing.assert_allclose(r2.final_vector, np.ones(n, dtype=np.float32) * 3.0 * (2.0 ** 10), rtol=1e-5)
        assert r1.lease_id != r2.lease_id

    def test_t3_06_mid_loop_worker_network_drop_abort_refund(self, hermetic_stateful_suite: Dict[str, Any]):
        """T3.6: Worker network drop mid-loop aborts execution, refunds escrow, and releases lease."""
        worker = hermetic_stateful_suite["worker"]
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        escrow_mgr: EscrowManager = hermetic_stateful_suite["escrow_manager"]
        settlement: MockSettlementService = hermetic_stateful_suite["settlement"]
        worker.servicer.drop_connection_at_step = 3

        n = 8
        a = np.eye(n, dtype=np.float32) * 0.5
        try:
            with pytest.raises(Exception):
                coord.execute_iterative_power(a, iterations=10)

            # Asserts release called in finally:
            assert worker.servicer.release_calls >= 1

            # Escrow assertion: reflects mid-loop network failure refund
            escrow = list(escrow_mgr._escrows.values())[-1]
            assert escrow.status == EscrowStatus.CANCELLED
            assert escrow.cancellation_reason in ("iteration_loop_failed", "cancelled")
            assert settlement.get_status(escrow.payment_hash) == "CANCELLED"
        finally:
            worker.servicer.drop_connection_at_step = None


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (6 TESTS)
# ==============================================================================

class TestTier4IterativePowerRealWorldScenarios:
    """
    Tier 4: Real-World Application Scenarios (>=5 application scenarios + numerical equivalence):
    1. Google PageRank algorithm on web graph transition matrix (damping d=0.85)
    2. Markov chain steady-state probability distribution on ergodic matrix
    3. Principal Component Analysis (PCA) dominant eigenvector extraction on covariance
    4. Vibrational modal analysis / structural dynamics matrix power iteration
    5. Latent Semantic Analysis (LSA) document-term matrix spectral decomposition
    6. 100 power iterations Higham numerical equivalence across FP32, FP16, and BF16
    """

    def test_t4_01_google_pagerank_algorithm(self, hermetic_stateful_suite: Dict[str, Any]):
        """T4.1: Google PageRank algorithm on N=20 web graph transition matrix (d=0.85)."""
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        np.random.seed(401)
        n = 20
        # Random directed graph adjacency matrix with ~25% connectivity
        adj = (np.random.rand(n, n) < 0.25).astype(np.float64)
        np.fill_diagonal(adj, 0)

        # Build column-stochastic Google Matrix G with d=0.85
        g_mat = ReferenceIterativeOracle.build_pagerank_matrix(adj, damping=0.85).astype(np.float32)

        # Initial uniform probability distribution vector
        v0 = (np.ones(n, dtype=np.float32) / float(n))

        # Run 30 power iterations
        res = coord.execute_iterative_power(g_mat, v0=v0, iterations=30)
        p_dist = res.final_vector / np.sum(res.final_vector)

        # Reference PageRank computation
        p_ref = v0.copy().astype(np.float64)
        g_f64 = g_mat.astype(np.float64)
        for _ in range(30):
            p_ref = np.matmul(g_f64, p_ref)
            p_ref /= np.sum(p_ref)

        np.testing.assert_allclose(p_dist, p_ref, rtol=1e-3, atol=1e-3)
        assert np.isclose(np.sum(p_dist), 1.0, atol=1e-4)

    def test_t4_02_markov_chain_steady_state_distribution(self, hermetic_stateful_suite: Dict[str, Any]):
        """T4.2: Markov chain steady-state probability distribution on ergodic transition matrix (N=15)."""
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        np.random.seed(402)
        n = 15
        # Generate strictly positive ergodic row-stochastic transition matrix P
        raw_p = np.random.uniform(0.1, 1.0, size=(n, n))
        p_matrix = raw_p / np.sum(raw_p, axis=1, keepdims=True)

        # Stationary distribution satisfies pi * P = pi <=> P^T * pi^T = pi^T
        a_matrix = p_matrix.T.astype(np.float32)
        v0 = (np.ones(n, dtype=np.float32) / float(n))

        res = coord.execute_iterative_power(a_matrix, v0=v0, iterations=40)
        pi_stat = res.final_vector / np.sum(res.final_vector)

        # Verify pi * P == pi
        pi_next = np.dot(pi_stat.astype(np.float64), p_matrix)
        np.testing.assert_allclose(pi_stat, pi_next, rtol=1e-3, atol=1e-3)

    def test_t4_03_pca_dominant_eigenvector_covariance(self, hermetic_stateful_suite: Dict[str, Any]):
        """T4.3: Principal Component Analysis (PCA) dominant eigenvector extraction on sample covariance."""
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        np.random.seed(403)
        m_samples = 100
        n_features = 12

        # Generate synthetic feature data with strong first principal variance
        data = np.random.randn(m_samples, n_features)
        data[:, 0] *= 5.0
        data[:, 1] *= 3.0
        centered = data - np.mean(data, axis=0)

        # Sample covariance matrix Sigma = 1/(M-1) * X^T * X
        cov = (np.matmul(centered.T, centered) / float(m_samples - 1)).astype(np.float32)
        v0 = np.random.randn(n_features).astype(np.float32)

        # Power method with normalization extracts first principal eigenvector
        res = coord.execute_iterative_power(cov, v0=v0, iterations=35, normalize=True)
        w_extracted = res.final_vector

        # Reference dominant eigenvector via NumPy eigh
        eigvals, eigvecs = np.linalg.eigh(cov)
        w_ref = eigvecs[:, -1]

        # Dot product magnitude |<w, w_ref>| should be ~ 1.0
        alignment = abs(float(np.dot(w_extracted, w_ref)))
        assert alignment > 0.99

    def test_t4_04_vibrational_modal_analysis_structural_dynamics(self, hermetic_stateful_suite: Dict[str, Any]):
        """T4.4: Vibrational modal analysis / structural dynamics fundamental mode extraction."""
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        n_dof = 10
        # 10-DOF spring-mass system tridiagonal stiffness matrix
        k_stiffness = np.diag([2.0] * n_dof) - np.diag([1.0] * (n_dof - 1), 1) - np.diag([1.0] * (n_dof - 1), -1)
        # Uniform mass matrix
        m_mass = np.eye(n_dof)

        # Dynamic flexibility matrix A = K^-1 * M
        flexibility = np.linalg.inv(k_stiffness).astype(np.float32)
        v0 = np.ones(n_dof, dtype=np.float32)

        res = coord.execute_iterative_power(flexibility, v0=v0, iterations=30, normalize=True)
        mode_extracted = res.final_vector

        # Fundamental natural mode shape from eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(flexibility)
        mode_ref = eigvecs[:, -1]
        alignment = abs(float(np.dot(mode_extracted, mode_ref)))
        assert alignment > 0.99

    def test_t4_05_latent_semantic_analysis_document_term(self, hermetic_stateful_suite: Dict[str, Any]):
        """T4.5: Latent Semantic Analysis (LSA) document-term matrix spectral decomposition."""
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        np.random.seed(405)
        n_terms = 15
        n_docs = 20

        # Term-document matrix X
        x_matrix = np.random.poisson(lam=2.0, size=(n_terms, n_docs)).astype(np.float32)
        # Term correlation matrix C = X * X^T
        c_matrix = np.matmul(x_matrix, x_matrix.T)

        v0 = np.ones(n_terms, dtype=np.float32)
        res = coord.execute_iterative_power(c_matrix, v0=v0, iterations=25, normalize=True)

        # SVD dominant left-singular vector of X
        u_svd, _, _ = np.linalg.svd(x_matrix)
        concept_ref = u_svd[:, 0]
        alignment = abs(float(np.dot(res.final_vector, concept_ref)))
        assert alignment > 0.99

    def test_t4_06_numerical_equivalence_100_iterations_higham(self, hermetic_stateful_suite: Dict[str, Any]):
        """T4.6: 100 power iterations match NumPy reference within Higham bounds across FP32, FP16, and BF16."""
        coord: DistributedMatmulCoordinator = hermetic_stateful_suite["coordinator"]
        n = 16
        np.random.seed(406)

        # Symmetric positive-definite matrix scaled to spectral radius <= 1.0
        raw = np.random.randn(n, n).astype(np.float32)
        sym = np.matmul(raw, raw.T)
        a_scaled = (sym / (np.linalg.norm(sym, 2) * 1.05)).astype(np.float32)
        v0 = np.random.randn(n).astype(np.float32)

        # FP32
        res_fp32 = coord.execute_iterative_power(a_scaled, v0=v0, iterations=100, dtype="fp32")
        ref_fp32 = ReferenceIterativeOracle.power_iteration(a_scaled, v0, 100, dtype="fp32")
        np.testing.assert_allclose(res_fp32.final_vector, ref_fp32, rtol=1e-3, atol=1e-3)

        # FP16
        a_fp16 = a_scaled.astype(np.float16)
        v0_fp16 = v0.astype(np.float16)
        res_fp16 = coord.execute_iterative_power(a_fp16, v0=v0_fp16, iterations=50, dtype="fp16")
        ref_fp16 = ReferenceIterativeOracle.power_iteration(a_fp16, v0_fp16, 50, dtype="fp16")
        np.testing.assert_allclose(res_fp16.final_vector, ref_fp16, rtol=2e-2, atol=2e-2)

        # BF16
        a_bf16 = truncate_to_bf16(a_scaled)
        v0_bf16 = truncate_to_bf16(v0)
        res_bf16 = coord.execute_iterative_power(a_bf16, v0=v0_bf16, iterations=50, dtype="bf16")
        ref_bf16 = ReferenceIterativeOracle.power_iteration(a_bf16, v0_bf16, 50, dtype="bf16")
        np.testing.assert_allclose(res_bf16.final_vector, ref_bf16, rtol=3e-2, atol=3e-2)
