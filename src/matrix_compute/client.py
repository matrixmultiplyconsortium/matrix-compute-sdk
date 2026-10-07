"""
High-level Consumer Client API for matrix-compute-sdk.

Provides `MatrixComputeClient` and `distributed_matmul` for distributed matrix
multiplication across decentralized compute nodes. Handles matrix tiling, worker
discovery, L402 trustless escrow, gRPC task dispatch, dynamic Freivalds verification,
and result assembly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from matrix_compute.config import ClientConfig, PartitioningStrategy, get_default_config
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

    def __init__(self, config: Optional[ClientConfig] = None, **kwargs: Any) -> None:
        """
        Initialize the MatrixComputeClient.

        Args:
            config: Optional ClientConfig instance. If omitted, loaded from environment and kwargs.
            **kwargs: Configuration overrides passed to ClientConfig.load().
        """
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

        self.escrow_manager = EscrowManager(
            settlement_backend=backend,
            default_timeout_seconds=self.config.task_timeout_seconds,
            verification_key=self.config.settlement_verification_key,
            auto_settle=False,
        )

        # 4. P2P gRPC subsystem
        self.channel_pool = ChannelPool(
            max_message_bytes=self.config.grpc_max_message_bytes,
            ssl_enabled=self.config.ssl_enabled,
        )
        self.dispatcher = GrpcTaskDispatcher(
            channel_pool=self.channel_pool,
            consumer_pubkey=self.signer.get_pubkey_hex(),
            connect_timeout=self.config.connect_timeout_seconds,
            task_timeout=self.config.task_timeout_seconds,
            chunk_size_bytes=self.config.chunk_size_bytes,
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

        # 2. Resolve target dtype
        target_dtype = normalize_dtype(dtype or self.config.dtype_preference)
        eff_tile_size = int(tile_size) if tile_size is not None else self.config.default_tile_size

        # 3. Delegate to distributed coordinator
        return self.coordinator.execute_gemm(
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
