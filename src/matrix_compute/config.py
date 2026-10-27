"""Configuration manager for matrix-compute-sdk client."""

from __future__ import annotations

from enum import Enum
import json
import os
import secrets
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


class PartitioningStrategy(str, Enum):
    """
    Mathematical matrix partitioning strategy for distributed matrix multiplication.

    - BLOCK_3D: Standard 3D sub-block tiling along M, N, and K dimensions (baseline).
    - OUTER_PRODUCT_1D: Slices only inner dimension K into strips (upload-optimal, 0% upload redundancy).
    - OUTPUT_GRID_2D: Slices M and N into a 2D spatial grid with K unsliced (download-optimal, zero local reduction).
    """

    BLOCK_3D = "block_3d"
    OUTER_PRODUCT_1D = "outer_product_1d"
    OUTPUT_GRID_2D = "output_grid_2d"

    def __str__(self) -> str:
        return str(self.value)


class WorkloadType(str, Enum):
    """
    Execution workload profile for matrix computations.

    - GEMM_ONE_SHOT: Single stateless one-off matrix multiplication (C = A x B).
    - ITERATIVE_POWER: Iterative computation where matrix A is pinned in remote VRAM across vector steps.
    - BATCH_STREAM: Streaming continuous sequence of matrix multiply tasks.
    """

    GEMM_ONE_SHOT = "gemm_one_shot"
    ITERATIVE_POWER = "iterative_power"
    BATCH_STREAM = "batch_stream"

    def __str__(self) -> str:
        return str(self.value)


class ClientConfig(BaseModel):
    """
    Configuration model for MatrixComputeClient and distributed dispatch.
    Resolves settings with priority: Explicit arguments > Environment variables > Defaults.
    """

    # Consumer Identity
    consumer_privkey_hex: str = Field(
        default_factory=lambda: hex(secrets.randbelow(SECP256K1_ORDER - 1) + 1)[2:].zfill(64),
        description="32-byte Secp256k1 private key hex for consumer identity and dispute signing",
    )

    # Relay & Discovery Settings
    nostr_relays: List[str] = Field(
        default_factory=lambda: ["ws://127.0.0.1:8008"],
        description="Nostr discovery relay WebSocket URLs for worker announcements",
    )
    min_worker_vram_mb: int = Field(4096, ge=512, description="Minimum worker VRAM in MB")
    max_price_per_gflop: float = Field(0.001, ge=0.0, description="Max acceptable price in sats/GFLOP")
    max_budget_sats: Optional[int] = Field(None, ge=0, description="Maximum total satoshi budget allowance")

    # Settlement & L402 Settings
    settlement_api_url: str = Field(
        "http://127.0.0.1:8080", description="Settlement layer base HTTP URL"
    )
    settlement_verification_key: str = Field(
        "settlement_secret_key_dev", description="L402 Macaroon HMAC verification key"
    )
    lnd_payer_enabled: bool = Field(
        False, description="Enable genuine LND wallet payments for HODL invoices"
    )
    consumer_lnd_rest_url: Optional[str] = Field(
        None, description="Consumer LND REST API base URL (e.g. https://127.0.0.1:8081)"
    )
    consumer_lnd_macaroon_hex: Optional[str] = Field(
        None, description="Hex-encoded Consumer LND admin macaroon"
    )
    consumer_lnd_macaroon_path: Optional[str] = Field(
        None, description="File path to Consumer LND admin.macaroon"
    )
    consumer_lnd_tls_cert: Optional[str] = Field(
        None, description="Consumer LND TLS cert PEM string"
    )
    consumer_lnd_tls_path: Optional[str] = Field(
        None, description="File path to Consumer LND tls.cert"
    )
    nwc_uri: Optional[str] = Field(
        default=None,
        description="Nostr Wallet Connect URI (nostr+walletconnect://<wallet_pubkey>?relay=<relay_url>&secret=<client_secret>)",
    )

    # Network Timeouts & Retries
    connect_timeout_seconds: float = Field(5.0, ge=0.1, description="Worker gRPC connection timeout")
    task_timeout_seconds: float = Field(30.0, ge=0.5, description="Worker tile execution timeout")
    max_retries: int = Field(3, ge=0, description="Maximum worker retry attempts on failure")
    retry_backoff_factor: float = Field(0.5, ge=0.0, description="Exponential retry backoff multiplier")

    # Tiling & Mathematical Constraints
    default_tile_size: int = Field(1024, ge=1, description="Default square block dimension for tiling")
    dtype_preference: str = Field("fp32", description="Precision format: 'fp32', 'fp16', or 'bf16'")

    # Partitioning & Workload Strategy
    partitioning_strategy: PartitioningStrategy = Field(
        default=PartitioningStrategy.BLOCK_3D,
        description="Mathematical matrix partitioning strategy (BLOCK_3D, OUTER_PRODUCT_1D, OUTPUT_GRID_2D)",
    )
    workload_type: WorkloadType = Field(
        default=WorkloadType.GEMM_ONE_SHOT,
        description="Workload profile for compute execution (GEMM_ONE_SHOT, ITERATIVE_POWER, BATCH_STREAM)",
    )
    num_strips: Optional[int] = Field(
        default=None,
        ge=1,
        description="Explicit strip count for 1D outer-product partitioning; defaults to worker count or tile_size heuristic",
    )
    grid_rows: Optional[int] = Field(
        default=None,
        ge=1,
        description="Explicit row partition count P_r for 2D output grid; defaults to worker count factor or tile_size heuristic",
    )
    grid_cols: Optional[int] = Field(
        default=None,
        ge=1,
        description="Explicit column partition count P_c for 2D output grid; defaults to worker count factor or tile_size heuristic",
    )
    grid_shape: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Explicit (P_r, P_c) partition shape for 2D output grid",
    )
    lease_ttl_seconds: int = Field(
        default=3600,
        ge=1,
        description="Remote GPU VRAM session lease duration in seconds for stateful iterative workloads",
    )
    spot_check_probability: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Sampling probability for probabilistic Freivalds verification in iterative workloads",
    )
    default_iterations: int = Field(
        default=100,
        ge=1,
        description="Default step count for iterative power loops",
    )

    # Freivalds Mathematical Verification Parameters
    verification_tolerance: float = Field(
        1e-4, ge=0.0, description="Baseline epsilon tolerance for Freivalds verification"
    )
    use_dynamic_freivalds_bound: bool = Field(
        True, description="Enable Higham dynamic error bound gamma_K * ||A||_inf * ||B||_inf"
    )
    freivalds_rounds: int = Field(
        5, ge=1, description="Number of randomized Freivalds test iterations (error prob <= 2^-k)"
    )
    relative_discrepancy_threshold: float = Field(
        1e-3, ge=0.0, description="Relative discrepancy threshold preventing zero-work fraud on small inputs"
    )

    # gRPC P2P Streaming Parameters
    chunk_size_bytes: int = Field(1024 * 1024, ge=1024, description="Size of streamed tensor chunks (bytes)")
    grpc_max_message_bytes: int = Field(100 * 1024 * 1024, ge=1024 * 1024, description="Max gRPC message size")
    ssl_enabled: bool = Field(False, description="Whether worker gRPC channel uses TLS")

    # Offline / Hermetic Test Doubles Controls
    offline_mock_mode: bool = Field(
        False, description="Master switch enabling in-memory test doubles without network daemons"
    )
    mock_relay: bool = Field(False, description="Route discovery to in-memory MockNostrRelay")
    mock_settlement: bool = Field(False, description="Route escrow to in-memory MockSettlementService")
    mock_workers: bool = Field(False, description="Dispatch tasks to in-memory MockWorkerNode pool")

    # Confidential Computing (TEE) & End-to-End Transport Encryption
    require_confidential_compute: bool = Field(
        True,
        description="Enforce TEE hardware attestation verification and AES-256-GCM end-to-end transport encryption",
    )
    required_tee_type: Optional[str] = Field(
        None,
        description="Specific TEE architecture requirement (e.g. 'nvidia-cca', 'simulated-cca'). None accepts any valid TEE.",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_from_environment(cls, data: Any) -> Any:
        """Enrich configuration dictionary with MATRIX_* and standard environment variables."""
        if not isinstance(data, dict):
            return data

        env_mappings = {
            "consumer_privkey_hex": ["MATRIX_CONSUMER_PRIVKEY", "CONSUMER_PRIVKEY", "MATRIX_PRIVKEY"],
            "nostr_relays": ["MATRIX_NOSTR_RELAYS", "NOSTR_RELAYS"],
            "settlement_api_url": ["MATRIX_SETTLEMENT_API_URL", "MATRIX_SETTLEMENT_URL", "SETTLEMENT_API_URL", "SETTLEMENT_URL"],
            "settlement_verification_key": ["MATRIX_SETTLEMENT_VERIFICATION_KEY", "SETTLEMENT_VERIFICATION_KEY"],
            "min_worker_vram_mb": ["MATRIX_MIN_WORKER_VRAM_MB", "MIN_WORKER_VRAM_MB", "MIN_VRAM_MB"],
            "max_price_per_gflop": ["MATRIX_MAX_PRICE_PER_GFLOP", "MAX_PRICE_PER_GFLOP", "MATRIX_MAX_PRICE_SATS_PER_GFLOP"],
            "connect_timeout_seconds": ["MATRIX_CONNECT_TIMEOUT", "CONNECT_TIMEOUT"],
            "task_timeout_seconds": ["MATRIX_TASK_TIMEOUT", "TASK_TIMEOUT"],
            "max_retries": ["MATRIX_MAX_RETRIES", "MAX_RETRIES"],
            "retry_backoff_factor": ["MATRIX_RETRY_BACKOFF_FACTOR"],
            "default_tile_size": ["MATRIX_DEFAULT_TILE_SIZE", "TILE_SIZE"],
            "dtype_preference": ["MATRIX_DTYPE_PREFERENCE", "DTYPE_PREFERENCE"],
            "partitioning_strategy": [
                "MATRIX_PARTITIONING_STRATEGY",
                "PARTITIONING_STRATEGY",
                "MATRIX_PARTITION_STRATEGY",
                "PARTITION_STRATEGY",
            ],
            "workload_type": [
                "MATRIX_WORKLOAD_TYPE",
                "WORKLOAD_TYPE",
                "MATRIX_WORKLOAD_PROFILE",
                "WORKLOAD_PROFILE",
            ],
            "num_strips": [
                "MATRIX_NUM_STRIPS",
                "NUM_STRIPS",
                "MATRIX_STRIPS",
            ],
            "grid_rows": [
                "MATRIX_GRID_ROWS",
                "GRID_ROWS",
                "MATRIX_P_R",
                "P_R",
            ],
            "grid_cols": [
                "MATRIX_GRID_COLS",
                "GRID_COLS",
                "MATRIX_P_C",
                "P_C",
            ],
            "lease_ttl_seconds": [
                "MATRIX_COMPUTE_LEASE_TTL_SECONDS",
                "MATRIX_LEASE_TTL_SECONDS",
                "LEASE_TTL_SECONDS",
            ],
            "spot_check_probability": [
                "MATRIX_COMPUTE_SPOT_CHECK_PROBABILITY",
                "MATRIX_SPOT_CHECK_PROBABILITY",
                "SPOT_CHECK_PROBABILITY",
            ],
            "default_iterations": [
                "MATRIX_COMPUTE_DEFAULT_ITERATIONS",
                "MATRIX_DEFAULT_ITERATIONS",
                "DEFAULT_ITERATIONS",
                "MATRIX_ITERATIONS",
                "ITERATIONS",
            ],
            "verification_tolerance": ["MATRIX_VERIFICATION_TOLERANCE", "VERIFICATION_TOLERANCE"],
            "use_dynamic_freivalds_bound": ["MATRIX_USE_DYNAMIC_FREIVALDS_BOUND"],
            "freivalds_rounds": ["MATRIX_FREIVALDS_ROUNDS", "FREIVALDS_ROUNDS"],
            "relative_discrepancy_threshold": ["MATRIX_RELATIVE_DISCREPANCY_THRESHOLD"],
            "chunk_size_bytes": ["MATRIX_CHUNK_SIZE_BYTES"],
            "grpc_max_message_bytes": ["MATRIX_GRPC_MAX_MESSAGE_BYTES"],
            "ssl_enabled": ["MATRIX_SSL_ENABLED"],
            "offline_mock_mode": ["MATRIX_OFFLINE_MOCK_MODE", "MATRIX_MOCK_MODE", "OFFLINE_MOCK_MODE", "OFFLINE_HERMETIC"],
            "mock_relay": ["MATRIX_MOCK_RELAY"],
            "mock_settlement": ["MATRIX_MOCK_SETTLEMENT"],
            "mock_workers": ["MATRIX_MOCK_WORKERS"],
            "require_confidential_compute": ["MATRIX_REQUIRE_CONFIDENTIAL_COMPUTE", "REQUIRE_CONFIDENTIAL_COMPUTE"],
            "required_tee_type": ["MATRIX_REQUIRED_TEE_TYPE", "REQUIRED_TEE_TYPE"],
            "lnd_payer_enabled": ["MATRIX_LND_PAYER_ENABLED", "LND_PAYER_ENABLED"],
            "consumer_lnd_rest_url": ["CONSUMER_LND_REST_URL", "MATRIX_CONSUMER_LND_REST_URL", "LND_REST_URL"],
            "consumer_lnd_macaroon_hex": ["CONSUMER_LND_MACAROON_HEX", "MATRIX_CONSUMER_LND_MACAROON_HEX", "LND_MACAROON_HEX"],
            "consumer_lnd_macaroon_path": ["CONSUMER_LND_MACAROON_PATH", "MATRIX_CONSUMER_LND_MACAROON_PATH", "LND_MACAROON_PATH"],
            "consumer_lnd_tls_cert": ["CONSUMER_LND_TLS_CERT", "MATRIX_CONSUMER_LND_TLS_CERT", "LND_TLS_CERT"],
            "consumer_lnd_tls_path": ["CONSUMER_LND_TLS_PATH", "MATRIX_CONSUMER_LND_TLS_PATH", "LND_TLS_PATH"],
            "nwc_uri": ["MATRIX_NWC_URI", "NWC_URI", "NOSTR_WALLET_CONNECT_URI"],
        }

        enriched = dict(data)
        for field_name, env_vars in env_mappings.items():
            if field_name not in enriched or enriched[field_name] is None:
                for env_var in env_vars:
                    if env_var in os.environ and os.environ[env_var] != "":
                        val = os.environ[env_var]
                        enriched[field_name] = val
                        break

        # Parse string boolean representations
        bool_fields = [
            "use_dynamic_freivalds_bound",
            "ssl_enabled",
            "offline_mock_mode",
            "mock_relay",
            "lnd_payer_enabled",
            "mock_settlement",
            "mock_workers",
            "require_confidential_compute",
        ]
        for bf in bool_fields:
            if bf in enriched and isinstance(enriched[bf], str):
                enriched[bf] = enriched[bf].strip().lower() in ("true", "1", "yes", "on")

        # Automatically cascade master offline_mock_mode to child mock flags
        if enriched.get("offline_mock_mode") is True:
            enriched.setdefault("mock_relay", True)
            enriched.setdefault("mock_settlement", True)
            enriched.setdefault("mock_workers", True)

        # Support p_r and p_c input aliases
        if "p_r" in enriched and ("grid_rows" not in enriched or enriched["grid_rows"] is None):
            enriched["grid_rows"] = enriched["p_r"]
        if "p_c" in enriched and ("grid_cols" not in enriched or enriched["grid_cols"] is None):
            enriched["grid_cols"] = enriched["p_c"]

        # Support input aliases for iterative parameters
        if "iterations" in enriched and ("default_iterations" not in enriched or enriched["default_iterations"] is None):
            enriched["default_iterations"] = enriched["iterations"]
        if "ttl_seconds" in enriched and ("lease_ttl_seconds" not in enriched or enriched["lease_ttl_seconds"] is None):
            enriched["lease_ttl_seconds"] = enriched["ttl_seconds"]
        if "spot_check_prob" in enriched and ("spot_check_probability" not in enriched or enriched["spot_check_probability"] is None):
            enriched["spot_check_probability"] = enriched["spot_check_prob"]

        # Parse string integer representations
        for int_field in ("grid_rows", "grid_cols", "num_strips", "lease_ttl_seconds", "default_iterations"):
            if int_field in enriched and isinstance(enriched[int_field], str):
                try:
                    enriched[int_field] = int(enriched[int_field].strip())
                except ValueError:
                    pass

        # Parse string float representations
        for float_field in ("spot_check_probability", "verification_tolerance", "relative_discrepancy_threshold", "max_price_per_gflop"):
            if float_field in enriched and isinstance(enriched[float_field], str):
                try:
                    enriched[float_field] = float(enriched[float_field].strip())
                except ValueError:
                    pass

        # Parse nostr_relays list from string or JSON
        if "nostr_relays" in enriched and isinstance(enriched["nostr_relays"], str):
            raw = enriched["nostr_relays"].strip()
            if raw.startswith("[") and raw.endswith("]"):
                try:
                    enriched["nostr_relays"] = json.loads(raw)
                except Exception:
                    enriched["nostr_relays"] = [r.strip() for r in raw[1:-1].split(",") if r.strip()]
            else:
                enriched["nostr_relays"] = [r.strip() for r in raw.split(",") if r.strip()]

        return enriched

    @field_validator("settlement_api_url")
    @classmethod
    def validate_settlement_url(cls, v: str) -> str:
        cleaned = str(v).strip()
        parsed = urlparse(cleaned)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"settlement_api_url must be a valid http or https URL, got '{v}'")
        return cleaned

    @field_validator("partitioning_strategy", mode="before")
    @classmethod
    def validate_partitioning_strategy(cls, v: Any) -> PartitioningStrategy:
        """Parse partitioning strategy with case-insensitivity and alias support."""
        if isinstance(v, PartitioningStrategy):
            return v
        if not isinstance(v, str):
            raise ValueError(
                f"partitioning_strategy must be a PartitioningStrategy or string, got {type(v).__name__}"
            )
        cleaned = v.strip().lower()
        for member in PartitioningStrategy:
            if cleaned == member.value.lower() or cleaned == member.name.lower():
                return member

        aliases = {
            "outer_1d": PartitioningStrategy.OUTER_PRODUCT_1D,
            "1d_outer_product": PartitioningStrategy.OUTER_PRODUCT_1D,
            "outer_product": PartitioningStrategy.OUTER_PRODUCT_1D,
            "outerproduct1d": PartitioningStrategy.OUTER_PRODUCT_1D,
            "output_grid": PartitioningStrategy.OUTPUT_GRID_2D,
            "2d_output_grid": PartitioningStrategy.OUTPUT_GRID_2D,
            "outputgrid2d": PartitioningStrategy.OUTPUT_GRID_2D,
            "grid_2d": PartitioningStrategy.OUTPUT_GRID_2D,
            "output_2d": PartitioningStrategy.OUTPUT_GRID_2D,
            "output2d": PartitioningStrategy.OUTPUT_GRID_2D,
            "2d": PartitioningStrategy.OUTPUT_GRID_2D,
            "2d_grid": PartitioningStrategy.OUTPUT_GRID_2D,
            "3d_block": PartitioningStrategy.BLOCK_3D,
            "block3d": PartitioningStrategy.BLOCK_3D,
            "block": PartitioningStrategy.BLOCK_3D,
        }
        if cleaned in aliases:
            return aliases[cleaned]

        valid_options = [e.value for e in PartitioningStrategy]
        raise ValueError(
            f"Invalid partitioning_strategy '{v}'. Must be one of {valid_options}"
        )

    @field_validator("workload_type", mode="before")
    @classmethod
    def validate_workload_type(cls, v: Any) -> WorkloadType:
        """Parse workload type with case-insensitivity and alias support."""
        if isinstance(v, WorkloadType):
            return v
        if not isinstance(v, str):
            raise ValueError(
                f"workload_type must be a WorkloadType or string, got {type(v).__name__}"
            )
        cleaned = v.strip().lower()
        for member in WorkloadType:
            if cleaned == member.value.lower() or cleaned == member.name.lower():
                return member

        aliases = {
            "gemm": WorkloadType.GEMM_ONE_SHOT,
            "one_shot": WorkloadType.GEMM_ONE_SHOT,
            "gemm_oneshot": WorkloadType.GEMM_ONE_SHOT,
            "iterative": WorkloadType.ITERATIVE_POWER,
            "power": WorkloadType.ITERATIVE_POWER,
            "power_iteration": WorkloadType.ITERATIVE_POWER,
            "iterative_power": WorkloadType.ITERATIVE_POWER,
            "iterativepower": WorkloadType.ITERATIVE_POWER,
            "power_iter": WorkloadType.ITERATIVE_POWER,
            "poweriter": WorkloadType.ITERATIVE_POWER,
            "poweriteration": WorkloadType.ITERATIVE_POWER,
            "batch": WorkloadType.BATCH_STREAM,
            "stream": WorkloadType.BATCH_STREAM,
            "streaming": WorkloadType.BATCH_STREAM,
        }
        if cleaned in aliases:
            return aliases[cleaned]

        valid_options = [e.value for e in WorkloadType]
        raise ValueError(
            f"Invalid workload_type '{v}'. Must be one of {valid_options}"
        )

    @field_validator("dtype_preference")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"fp32", "fp16", "bf16"}
        cleaned = str(v).lower().strip()
        if cleaned not in allowed:
            raise ValueError(f"dtype_preference must be one of {allowed}, got '{v}'")
        return cleaned

    @field_validator("consumer_privkey_hex")
    @classmethod
    def validate_privkey(cls, v: str) -> str:
        clean = str(v).strip().lower()
        if not clean:
            return secrets.token_hex(32)
        if len(clean) != 64:
            raise ValueError(f"consumer_privkey_hex must be 64 hex characters (32 bytes), got length {len(clean)}")
        try:
            scalar = int(clean, 16)
        except ValueError:
            raise ValueError("consumer_privkey_hex contains non-hex characters")
        if not (1 <= scalar < SECP256K1_ORDER):
            raise ValueError("consumer_privkey_hex scalar out of valid Secp256k1 range")
        return clean

    @field_validator("nostr_relays")
    @classmethod
    def validate_relays(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("nostr_relays list must contain at least one relay URL")
        for r in v:
            parsed = urlparse(r)
            if parsed.scheme not in ("ws", "wss", "http", "https"):
                raise ValueError(f"Invalid relay URL scheme in '{r}'; expected ws, wss, http, or https")
        return v

    @field_validator("nwc_uri")
    @classmethod
    def validate_nwc_uri(cls, v: Optional[str]) -> Optional[str]:
        """Validates NWC URI format and parameters upon configuration initialization."""
        if v is None:
            return None
        cleaned = str(v).strip()
        if not cleaned:
            return None
        from matrix_compute.payment.nwc_params import NwcConnectionParams

        # Raises ValueError if URI is malformed, missing required params, or has bad hex/schemes
        NwcConnectionParams.from_uri(cleaned)
        return cleaned

    @model_validator(mode="after")
    def validate_and_sync_grid_shape(self) -> "ClientConfig":
        """Validates and synchronizes grid_rows, grid_cols, and grid_shape."""
        if self.grid_shape is not None:
            if not (isinstance(self.grid_shape, (tuple, list)) and len(self.grid_shape) == 2):
                raise ValueError(
                    f"grid_shape must be a 2-tuple of integers (grid_rows, grid_cols), got '{self.grid_shape}'"
                )
            r, c = self.grid_shape[0], self.grid_shape[1]
            if not isinstance(r, int) or not isinstance(c, int):
                try:
                    r, c = int(r), int(c)
                except (ValueError, TypeError):
                    raise ValueError(f"grid_shape dimensions must be integers, got {self.grid_shape}")
            if r < 1 or c < 1:
                raise ValueError(f"grid_shape dimensions must be >= 1, got {self.grid_shape}")

            if self.grid_rows is not None and self.grid_rows != r:
                raise ValueError(
                    f"Conflicting configuration: grid_rows={self.grid_rows} conflicts with grid_shape={self.grid_shape}"
                )
            if self.grid_cols is not None and self.grid_cols != c:
                raise ValueError(
                    f"Conflicting configuration: grid_cols={self.grid_cols} conflicts with grid_shape={self.grid_shape}"
                )
            self.grid_rows = r
            self.grid_cols = c
            self.grid_shape = (r, c)
        elif self.grid_rows is not None and self.grid_cols is not None:
            self.grid_shape = (self.grid_rows, self.grid_cols)

        return self

    @property
    def p_r(self) -> Optional[int]:
        """Alias for grid_rows (row partitions)."""
        return self.grid_rows

    @property
    def p_c(self) -> Optional[int]:
        """Alias for grid_cols (column partitions)."""
        return self.grid_cols

    @property
    def is_iterative_workload(self) -> bool:
        """Returns True if configured for stateful iterative workloads (ITERATIVE_POWER)."""
        return self.workload_type == WorkloadType.ITERATIVE_POWER

    @property
    def is_gemm_workload(self) -> bool:
        """Returns True if configured for one-shot GEMM workloads (GEMM_ONE_SHOT)."""
        return self.workload_type == WorkloadType.GEMM_ONE_SHOT

    @property
    def nwc_params(self) -> Optional[Any]:
        """Returns parsed NwcConnectionParams if nwc_uri is configured, else None."""
        if not self.nwc_uri:
            return None
        from matrix_compute.payment.nwc_params import NwcConnectionParams
        return NwcConnectionParams.from_uri(self.nwc_uri)

    @property
    def is_nwc_enabled(self) -> bool:
        """Returns True if a valid NWC URI is configured."""
        return bool(self.nwc_uri and self.nwc_uri.strip())

    @classmethod
    def load(cls, **kwargs: Any) -> "ClientConfig":
        """Alias constructor resolving environment variables and explicit overrides."""
        return cls(**kwargs)

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "ClientConfig":
        """Construct ClientConfig explicitly using the given environment mapping."""
        saved_env = os.environ.copy() if env is not None else None
        try:
            if env is not None:
                os.environ.clear()
                os.environ.update(env)
            return cls()
        finally:
            if saved_env is not None:
                os.environ.clear()
                os.environ.update(saved_env)

    @classmethod
    def for_testing(cls, **kwargs: Any) -> "ClientConfig":
        """
        Factory producing a hermetic, fast-failing configuration preset for unit/integration tests.
        Sets offline_mock_mode=True and fast timeouts.
        """
        params: Dict[str, Any] = {
            "offline_mock_mode": True,
            "mock_relay": True,
            "mock_settlement": True,
            "mock_workers": True,
            "require_confidential_compute": kwargs.get("require_confidential_compute", False),
            "connect_timeout_seconds": 1.0,
            "task_timeout_seconds": 3.0,
            "max_retries": 1,
            "default_tile_size": 256,
            "lease_ttl_seconds": 60,
            "spot_check_probability": 0.05,
            "default_iterations": 10,
        }
        params.update(kwargs)
        return cls(**params)


def get_default_config() -> ClientConfig:
    """Convenience helper returning standard ClientConfig."""
    return ClientConfig.load()
