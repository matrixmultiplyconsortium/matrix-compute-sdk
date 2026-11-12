"""
High-level Consumer Client API for matrix-compute-sdk.

Provides `MatrixComputeClient` and `distributed_matmul` for distributed matrix
multiplication across decentralized compute nodes. Handles matrix tiling, worker
discovery, L402 trustless escrow, gRPC task dispatch, dynamic Freivalds verification,
and result assembly.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from matrix_compute.config import ClientConfig, PartitioningStrategy, get_default_config
from matrix_compute.crypto.attestation_verifier import AttestationVerifier, VerificationPolicy
from matrix_compute.crypto.hashcash import solve_hashcash
from matrix_compute.crypto.pki_fixtures import (
    DEFAULT_GOLDEN_RIM,
    DEFAULT_SNP_GOLDEN_RIM,
    DEFAULT_TDX_GOLDEN_RIM,
    get_default_amd_pki,
    get_default_intel_pki,
    get_default_pki,
)
from matrix_compute.crypto.signer import ClientSigner
from matrix_compute.discovery.matchmaker import Matchmaker
from matrix_compute.discovery.relay_client import RelayClient
from matrix_compute.exceptions import (
    ConfigurationError,
    MatrixComputeError,
    MatrixDimensionError,
)
from matrix_compute.orchestrator.coordinator import DistributedMatmulCoordinator
from matrix_compute.orchestrator.scheduler import ParallelTaskScheduler
from matrix_compute.p2p.channel_pool import ChannelPool
from matrix_compute.p2p.dispatcher import GrpcTaskDispatcher
from matrix_compute.payment.backend import HttpSettlementBackend, MockSettlementBackend
from matrix_compute.payment.escrow import EscrowManager
from matrix_compute.payment.lnd_payer import LndPayer
from matrix_compute.test_doubles.mock_relay import MockNostrRelay
from matrix_compute.test_doubles.mock_settlement import MockSettlementService
from matrix_compute.test_doubles.mock_worker import MockWorkerNode
from matrix_compute.tiling.precision import normalize_dtype

logger = logging.getLogger(__name__)


class MatrixComputeClient:
    """
    High-level consumer client for distributed matrix multiplication.

    Supports context manager usage (`with MatrixComputeClient() as client:`)
    and automatic hermetic offline test double provisioning.
    """

    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        nwc_connection_uri: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the MatrixComputeClient.

        Args:
            config: Optional ClientConfig instance. If omitted, loaded from environment and kwargs.
            nwc_connection_uri: Optional Nostr Wallet Connect connection URI.
            **kwargs: Configuration overrides passed to ClientConfig.load().
        """
        if nwc_connection_uri is not None:
            kwargs["nwc_uri"] = nwc_connection_uri

        if config is not None:
            if kwargs:
                cfg_dict = config.model_dump()
                cfg_dict.update(kwargs)
                self.config = ClientConfig.load(**cfg_dict)
            else:
                self.config = config
        else:
            self.config = ClientConfig.load(**kwargs)

        self._lock = threading.RLock()
        self._is_closed = False

        # Consumer cryptographic signer identity
        self.signer = ClientSigner(self.config.consumer_privkey_hex)

        # Test doubles / mock resources
        self._mock_relay: Optional[MockNostrRelay] = None
        self._mock_settlement: Optional[MockSettlementService] = None
        self._mock_workers: List[MockWorkerNode] = []

        # Trial & execution telemetry
        self.last_trial_macaroon: Optional[str] = None
        self.last_hashcash_stamp: Optional[str] = None
        self.last_hashcash_solve_ms: float = 0.0
        self.last_manifest: Optional[Any] = None
        self.last_assembler: Optional[Any] = None

        # Internal subsystems
        self._init_subsystems()

    def _init_subsystems(self) -> None:
        """Initialize discovery, escrow, gRPC channel pool, dispatcher, and orchestrator."""
        # 1. Test doubles & offline hermetic mode
        if self.config.offline_mock_mode:
            self._setup_offline_mock_environment()

        # 2. Discovery subsystem
        if self.config.offline_mock_mode and self._mock_relay is not None:
            self.relay_client = RelayClient(
                relay_urls=["mock://relay"],
                timeout_seconds=self.config.connect_timeout_seconds,
                in_memory_relay=self._mock_relay,
            )
        else:
            self.relay_client = RelayClient(
                relay_urls=self.config.nostr_relays,
                timeout_seconds=self.config.connect_timeout_seconds,
            )

        self.matchmaker = Matchmaker(
            min_vram_mb=self.config.min_worker_vram_mb,
            max_price_per_gflop=self.config.max_price_per_gflop,
        )

        # 3. Settlement & Escrow subsystem
        if self.config.offline_mock_mode and self._mock_settlement is not None:
            backend = MockSettlementBackend(self._mock_settlement)
        elif self.config.mock_settlement:
            self._mock_settlement = MockSettlementService(
                verification_key=self.config.settlement_verification_key
            )
            backend = MockSettlementBackend(self._mock_settlement)
        else:
            backend = HttpSettlementBackend(
                base_url=self.config.settlement_api_url,
                verification_key=self.config.settlement_verification_key,
                timeout_seconds=self.config.connect_timeout_seconds,
            )

        lnd_payer = None
        if self.config.lnd_payer_enabled or self.config.consumer_lnd_rest_url:
            lnd_payer = LndPayer(
                rest_url=self.config.consumer_lnd_rest_url,
                macaroon_hex=self.config.consumer_lnd_macaroon_hex,
                macaroon_path=self.config.consumer_lnd_macaroon_path,
                tls_cert=self.config.consumer_lnd_tls_cert,
                tls_path=self.config.consumer_lnd_tls_path,
                timeout_seconds=self.config.task_timeout_seconds,
            )

        nwc_client = None
        if getattr(self.config, "nwc_uri", None):
            try:
                from matrix_compute.payment.nwc_client import NwcClient, NwcConnectionParams
                nwc_client = NwcClient(
                    params=NwcConnectionParams.from_uri(self.config.nwc_uri),
                    timeout_seconds=self.config.task_timeout_seconds,
                )
            except Exception as ex:
                logger.warning("Failed initializing NwcClient in MatrixComputeClient: %s", ex)

        self.escrow_manager = EscrowManager(
            settlement_backend=backend,
            default_timeout_seconds=self.config.task_timeout_seconds,
            verification_key=self.config.settlement_verification_key,
            auto_settle=False,
            lnd_payer=lnd_payer,
            nwc_client=nwc_client,
            config=self.config,
        )

        # 4. P2P gRPC subsystem
        self.channel_pool = ChannelPool(
            max_message_bytes=self.config.grpc_max_message_bytes,
            ssl_enabled=self.config.ssl_enabled,
        )
        attestation_verifier = None
        if (self.config.offline_mock_mode or self.config.required_tee_type == "simulated-cca") and self.config.require_confidential_compute:
            policy = VerificationPolicy()
            policy.add_trusted_root_pem(get_default_pki().root_cert_pem)
            policy.add_trusted_root_pem(get_default_amd_pki().root_cert_pem)
            policy.add_trusted_root_pem(get_default_intel_pki().root_cert_pem)
            policy.add_golden_rim("nvidia-cca", DEFAULT_GOLDEN_RIM)
            policy.add_golden_rim("simulated-cca", DEFAULT_GOLDEN_RIM)
            policy.add_golden_rim("amd-sev-snp", DEFAULT_SNP_GOLDEN_RIM)
            policy.add_golden_rim("intel-tdx", DEFAULT_TDX_GOLDEN_RIM)
            attestation_verifier = AttestationVerifier(policy=policy)

        self.dispatcher = GrpcTaskDispatcher(
            channel_pool=self.channel_pool,
            consumer_pubkey=self.signer.get_pubkey_hex(),
            connect_timeout=self.config.connect_timeout_seconds,
            task_timeout=self.config.task_timeout_seconds,
            chunk_size_bytes=self.config.chunk_size_bytes,
            require_confidential_compute=self.config.require_confidential_compute,
            required_tee_type=self.config.required_tee_type,
            attestation_verifier=attestation_verifier,
        )

        # 5. Orchestrator & Scheduler
        self.scheduler = ParallelTaskScheduler(
            dispatcher=self.dispatcher,
            max_retries=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_factor,
        )
        self.coordinator = DistributedMatmulCoordinator(
            config=self.config,
            signer=self.signer,
            relay_client=self.relay_client,
            matchmaker=self.matchmaker,
            escrow_manager=self.escrow_manager,
            scheduler=self.scheduler,
            dispatcher=self.dispatcher,
            mock_workers=self._mock_workers if self.config.offline_mock_mode else None,
        )

    def _setup_offline_mock_environment(self) -> None:
        """Provisions hermetic in-process Nostr relay, settlement service, and worker pool."""
        self._mock_relay = MockNostrRelay()
        self._mock_settlement = MockSettlementService(
            verification_key=self.config.settlement_verification_key
        )

        # Launch 2 in-process MockWorkerNode instances on ephemeral ports
        for i in range(2):
            worker = MockWorkerNode(
                port=0,
                settlement_service=self._mock_settlement,
                verification_key=self.config.settlement_verification_key,
                vram_mb=24576,
                sats_per_gflop=0.0005,
            )
            worker.start()
            worker.publish_to_relay(self._mock_relay)
            self._mock_workers.append(worker)

    @property
    def mock_relay(self) -> Optional[MockNostrRelay]:
        """Accessor for in-memory MockNostrRelay when running in mock mode."""
        return self._mock_relay

    @property
    def mock_settlement(self) -> Optional[MockSettlementService]:
        """Accessor for in-memory MockSettlementService when running in mock mode."""
        return self._mock_settlement

    @property
    def mock_workers(self) -> List[MockWorkerNode]:
        """Accessor for in-memory MockWorkerNode instances when running in mock mode."""
        return list(self._mock_workers)

    def matmul(
        self,
        A: np.ndarray,
        B: np.ndarray,
        dtype: Optional[str] = None,
        tile_size: Optional[int] = None,
        partitioning_strategy: Optional[Union[str, PartitioningStrategy]] = None,
        num_strips: Optional[int] = None,
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Computes the distributed matrix product C = A x B.

        Args:
            A: 2D numpy array of shape (M, K).
            B: 2D numpy array of shape (K, N).
            dtype: Target precision format ('fp32', 'fp16', 'bf16'). Defaults to config.dtype_preference.
            tile_size: Block size for matrix partitioning. Defaults to config.default_tile_size.
            partitioning_strategy: Partitioning strategy (BLOCK_3D, OUTER_PRODUCT_1D, OUTPUT_GRID_2D). Defaults to config.
            num_strips: Strip count for 1D outer product. Defaults to worker count or tile_size heuristic.
            grid_rows: Row partition count P_r for 2D output grid.
            grid_cols: Column partition count P_c for 2D output grid.
            grid_shape: (P_r, P_c) partition shape for 2D output grid.
            **kwargs: Additional options passed to the coordinator.

        Returns:
            np.ndarray of shape (M, N) containing the assembled matrix product.

        Raises:
            MatrixDimensionError: If shapes are invalid, empty, or inner dimensions mismatch.
            MatrixComputeError: If distributed dispatch, verification, or assembly fails.
        """
        with self._lock:
            if self._is_closed:
                raise MatrixComputeError("Cannot execute matmul: MatrixComputeClient is closed")

        # 1. Type & Dimension Validation
        if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
            raise MatrixDimensionError(
                "Input matrices A and B must be numpy.ndarray instances",
                shape_a=getattr(A, "shape", None),
                shape_b=getattr(B, "shape", None),
            )

        if A.ndim != 2 or B.ndim != 2:
            raise MatrixDimensionError(
                f"Matrices must be 2-dimensional. Got shapes A={A.shape} and B={B.shape}",
                shape_a=A.shape,
                shape_b=B.shape,
            )

        m, k = A.shape
        k2, n = B.shape

        if m == 0 or k == 0 or n == 0 or k2 == 0:
            raise MatrixDimensionError(
                f"Matrix dimensions must be non-zero. Got shapes A={A.shape} and B={B.shape}",
                shape_a=A.shape,
                shape_b=B.shape,
            )

        if k != k2:
            raise MatrixDimensionError(
                f"Inner dimension mismatch: A columns ({k}) != B rows ({k2})",
                shape_a=A.shape,
                shape_b=B.shape,
                expected_inner_dim=k,
            )

        # 1b. Zero-Drain Trial Sandbox guardrails (mode="trial")
        if self.config.is_trial_mode:
            if m > 4096 or k > 4096 or n > 4096:
                raise MatrixDimensionError(
                    f"PERMISSION_DENIED: Trial macaroon dimension exceeds 4096 cap (got A={A.shape}, B={B.shape}). "
                    "Pass nwc_uri='nostr+walletconnect://...' in mode='mainnet' for unbounded dimensions.",
                    shape_a=A.shape,
                    shape_b=B.shape,
                )
            pow_t0 = time.perf_counter()
            self.last_hashcash_stamp = solve_hashcash(
                resource=self.signer.get_pubkey_hex()[:16],
                bits=self.config.trial_hashcash_bits,
            )
            self.last_hashcash_solve_ms = (time.perf_counter() - pow_t0) * 1000.0
            trial_caveats = {
                "mode": "trial",
                "target_workers": [w.pubkey for w in self._mock_workers] or ["seed-worker-01", "seed-worker-02"],
                "max_matrix_dim": 4096,
                "max_gflops": (2.0 * 4096.0 * 4096.0 * 4096.0) / 1e9,
                "locked_sats": 0,
                "expiry": int(time.time()) + 300,
                "hashcash": self.last_hashcash_stamp,
            }
            caveat_json = json.dumps(trial_caveats, sort_keys=True)
            sig = hmac.new(
                self.config.settlement_verification_key.encode("utf-8"),
                caveat_json.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            token_b64 = base64.urlsafe_b64encode(
                json.dumps({"caveats": trial_caveats, "signature": sig}).encode("utf-8")
            ).decode("ascii")
            self.last_trial_macaroon = f"mc_trial_{token_b64}"

        # 2. Resolve target dtype
        target_dtype = normalize_dtype(dtype or self.config.dtype_preference)
        eff_tile_size = int(tile_size) if tile_size is not None else self.config.default_tile_size
        if self.config.is_trial_mode and tile_size is None and max(m, k, n) >= 1024:
            eff_tile_size = max(eff_tile_size, 2048)

        # 3. Delegate to distributed coordinator
        result = self.coordinator.execute_gemm(
            A=A,
            B=B,
            dtype=target_dtype,
            tile_size=eff_tile_size,
            partitioning_strategy=partitioning_strategy,
            num_strips=num_strips,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            grid_shape=grid_shape,
            **kwargs,
        )
        self.last_manifest = getattr(self.coordinator, "last_manifest", getattr(self.coordinator, "_last_manifest", None))
        self.last_assembler = getattr(self.coordinator, "last_assembler", getattr(self.coordinator, "_last_assembler", None))
        if self.config.is_trial_mode and self.last_manifest is not None:
            self.last_manifest.total_cost_sats = 0
        return result

    def power_iteration(
        self,
        A: np.ndarray,
        v0: Optional[np.ndarray] = None,
        iterations: int = 100,
        dtype: Optional[str] = None,
        lease_ttl_seconds: Optional[int] = None,
        spot_check_probability: Optional[float] = None,
        normalize: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Executes distributed power iteration v_{iterations} = A^{iterations} v_0
        using remote VRAM session leasing and probabilistic Freivalds verification.

        Returns:
            np.ndarray of shape (N,) containing the final state vector.
        """
        result = self.coordinator.execute_iterative_power(
            A=A,
            v0=v0,
            iterations=iterations,
            dtype=dtype,
            lease_ttl_seconds=lease_ttl_seconds,
            spot_check_probability=spot_check_probability,
            normalize=normalize,
            **kwargs,
        )
        if hasattr(result, "final_vector"):
            return result.final_vector
        return result

    def close(self) -> None:
        """
        Gracefully releases all client resources, channels, thread pools, and mock workers.
        Idempotent and safe to call multiple times.
        """
        with self._lock:
            if self._is_closed:
                return
            self._is_closed = True

            # Stop mock workers
            for worker in self._mock_workers:
                try:
                    worker.stop()
                except Exception as ex:
                    logger.debug("Error stopping mock worker: %s", ex)
            self._mock_workers.clear()

            # Shutdown scheduler thread pools
            if hasattr(self, "scheduler") and self.scheduler is not None:
                try:
                    self.scheduler.shutdown()
                except Exception as ex:
                    logger.debug("Error shutting down scheduler: %s", ex)

            # Close gRPC channel pool
            if hasattr(self, "channel_pool") and self.channel_pool is not None:
                try:
                    self.channel_pool.close()
                except Exception as ex:
                    logger.debug("Error closing channel pool: %s", ex)

            # Clean up relay client
            if hasattr(self, "relay_client") and self.relay_client is not None:
                try:
                    self.relay_client.close()
                except Exception as ex:
                    logger.debug("Error closing relay client: %s", ex)

            # Close escrow manager and payment clients
            if hasattr(self, "escrow_manager") and self.escrow_manager is not None:
                try:
                    self.escrow_manager.close()
                except Exception as ex:
                    logger.debug("Error closing escrow manager: %s", ex)

    def __enter__(self) -> "MatrixComputeClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def distributed_matmul(
    A: np.ndarray,
    B: np.ndarray,
    dtype: str = "fp32",
    config: Optional[ClientConfig] = None,
    tile_size: Optional[int] = None,
    partitioning_strategy: Optional[Union[str, PartitioningStrategy]] = None,
    num_strips: Optional[int] = None,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Convenience functional interface for distributed matrix multiplication.

    Instantiates a temporary MatrixComputeClient, computes C = A x B, and closes cleanly.
    """
    client_kwargs = dict(kwargs)
    if partitioning_strategy is not None and "partitioning_strategy" not in client_kwargs:
        client_kwargs["partitioning_strategy"] = partitioning_strategy
    if num_strips is not None and "num_strips" not in client_kwargs:
        client_kwargs["num_strips"] = num_strips
    if grid_rows is not None and "grid_rows" not in client_kwargs:
        client_kwargs["grid_rows"] = grid_rows
    if grid_cols is not None and "grid_cols" not in client_kwargs:
        client_kwargs["grid_cols"] = grid_cols
    if grid_shape is not None and "grid_shape" not in client_kwargs:
        client_kwargs["grid_shape"] = grid_shape

    matmul_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ("partitioning_strategy", "num_strips", "grid_rows", "grid_cols", "grid_shape")
    }

    with MatrixComputeClient(config=config, **client_kwargs) as client:
        return client.matmul(
            A=A,
            B=B,
            dtype=dtype,
            tile_size=tile_size,
            partitioning_strategy=partitioning_strategy,
            num_strips=num_strips,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            grid_shape=grid_shape,
            **matmul_kwargs,
        )


def distributed_power_iteration(
    A: np.ndarray,
    v0: Optional[np.ndarray] = None,
    iterations: int = 100,
    dtype: str = "fp32",
    config: Optional[ClientConfig] = None,
    lease_ttl_seconds: Optional[int] = None,
    spot_check_probability: Optional[float] = None,
    normalize: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """
    Convenience top-level functional wrapper executing distributed power iteration.

    Instantiates a temporary MatrixComputeClient, executes power iteration, and closes cleanly.
    """
    with MatrixComputeClient(config=config) as client:
        return client.power_iteration(
            A=A,
            v0=v0,
            iterations=iterations,
            dtype=dtype,
            lease_ttl_seconds=lease_ttl_seconds,
            spot_check_probability=spot_check_probability,
            normalize=normalize,
            **kwargs,
        )


def main() -> None:
    """CLI entrypoint for matrix-compute console script."""
    parser = argparse.ArgumentParser(
        prog="matrix-compute",
        description="Distributed Matrix Multiplication SDK Client",
    )
    parser.add_argument("--version", action="version", version="matrix-compute-sdk 0.1.0")
    parser.add_argument("--rows-a", type=int, default=32, help="Number of rows for matrix A")
    parser.add_argument("--cols-a", type=int, default=32, help="Number of columns for matrix A")
    parser.add_argument("--cols-b", type=int, default=32, help="Number of columns for matrix B")
    parser.add_argument("--tile-size", type=int, default=16, help="Tile size block dimension")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--mock-mode", action="store_true", default=True, help="Run in hermetic mock mode")

    args = parser.parse_args()

    print(f"Initializing distributed GEMM: ({args.rows_a}x{args.cols_a}) x ({args.cols_a}x{args.cols_b}) [{args.dtype}]")
    np.random.seed(42)
    a = np.random.randn(args.rows_a, args.cols_a).astype(np.float32)
    b = np.random.randn(args.cols_a, args.cols_b).astype(np.float32)

    config = ClientConfig.load(
        offline_mock_mode=args.mock_mode,
        default_tile_size=args.tile_size,
        dtype_preference=args.dtype,
    )

    with MatrixComputeClient(config=config) as client:
        c = client.matmul(a, b, dtype=args.dtype, tile_size=args.tile_size)
        print(f"Computation successful! Result shape: {c.shape}, dtype: {c.dtype}")


if __name__ == "__main__":
    main()
