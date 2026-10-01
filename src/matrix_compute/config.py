"""Configuration manager for matrix-compute-sdk client."""

from __future__ import annotations

import json
import os
import secrets
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


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

    # Settlement & L402 Settings
    settlement_api_url: str = Field(
        "http://127.0.0.1:8080", description="Settlement layer base HTTP URL"
    )
    settlement_verification_key: str = Field(
        "settlement_secret_key_dev", description="L402 Macaroon HMAC verification key"
    )

    # Network Timeouts & Retries
    connect_timeout_seconds: float = Field(5.0, ge=0.1, description="Worker gRPC connection timeout")
    task_timeout_seconds: float = Field(30.0, ge=0.5, description="Worker tile execution timeout")
    max_retries: int = Field(3, ge=0, description="Maximum worker retry attempts on failure")
    retry_backoff_factor: float = Field(0.5, ge=0.0, description="Exponential retry backoff multiplier")

    # Tiling & Mathematical Constraints
    default_tile_size: int = Field(1024, ge=1, description="Default square block dimension for tiling")
    dtype_preference: str = Field("fp32", description="Precision format: 'fp32', 'fp16', or 'bf16'")

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
        }
        params.update(kwargs)
        return cls(**params)


def get_default_config() -> ClientConfig:
    """Convenience helper returning standard ClientConfig."""
    return ClientConfig.load()
