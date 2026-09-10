"""Unit tests for ClientConfig and configuration loader in matrix-compute-sdk."""

import os
import pytest
from pydantic import ValidationError

from matrix_compute.config import ClientConfig, get_default_config


def test_default_config():
    """Asserts production default values."""
    config = ClientConfig()
    assert config.nostr_relays == ["ws://127.0.0.1:8008"]
    assert config.min_worker_vram_mb == 4096
    assert config.max_price_per_gflop == 0.001
    assert config.settlement_api_url == "http://127.0.0.1:8080"
    assert config.settlement_verification_key == "settlement_secret_key_dev"
    assert config.connect_timeout_seconds == 5.0
    assert config.task_timeout_seconds == 30.0
    assert config.max_retries == 3
    assert config.default_tile_size == 1024
    assert config.dtype_preference == "fp32"
    assert config.verification_tolerance == 1e-4
    assert config.use_dynamic_freivalds_bound is True
    assert config.freivalds_rounds == 5
    assert config.chunk_size_bytes == 1024 * 1024
    assert config.grpc_max_message_bytes == 100 * 1024 * 1024
    assert config.ssl_enabled is False
    assert config.offline_mock_mode is False
    assert config.mock_relay is False
    assert config.mock_settlement is False
    assert config.mock_workers is False
    assert len(config.consumer_privkey_hex) == 64


def test_get_default_config():
    cfg = get_default_config()
    assert isinstance(cfg, ClientConfig)
    assert cfg.task_timeout_seconds == 30.0


def test_from_env_override(monkeypatch):
    """Verifies that environment variables override defaults."""
    monkeypatch.setenv("MATRIX_TASK_TIMEOUT", "45.0")
    monkeypatch.setenv("MATRIX_CONNECT_TIMEOUT", "2.5")
    monkeypatch.setenv("MATRIX_MIN_WORKER_VRAM_MB", "8192")
    monkeypatch.setenv("MATRIX_NOSTR_RELAYS", "ws://relay1.local:8008,ws://relay2.local:8008")
    monkeypatch.setenv("MATRIX_OFFLINE_MOCK_MODE", "true")
    monkeypatch.setenv("MATRIX_DTYPE_PREFERENCE", "fp16")

    config = ClientConfig()
    assert config.task_timeout_seconds == 45.0
    assert config.connect_timeout_seconds == 2.5
    assert config.min_worker_vram_mb == 8192
    assert config.nostr_relays == ["ws://relay1.local:8008", "ws://relay2.local:8008"]
    assert config.offline_mock_mode is True
    assert config.mock_relay is True
    assert config.mock_settlement is True
    assert config.mock_workers is True
    assert config.dtype_preference == "fp16"


def test_json_nostr_relays_parsing(monkeypatch):
    """Verifies JSON-formatted relay list parsing from environment variable."""
    monkeypatch.setenv("MATRIX_NOSTR_RELAYS", '["ws://node1:8008", "ws://node2:8008"]')
    config = ClientConfig()
    assert config.nostr_relays == ["ws://node1:8008", "ws://node2:8008"]


def test_boolean_string_variations(monkeypatch):
    """Verifies parsing of various truthy and falsy boolean string representations."""
    monkeypatch.setenv("MATRIX_SSL_ENABLED", "yes")
    monkeypatch.setenv("MATRIX_USE_DYNAMIC_FREIVALDS_BOUND", "0")
    config = ClientConfig()
    assert config.ssl_enabled is True
    assert config.use_dynamic_freivalds_bound is False


def test_url_validation():
    """Verifies ValueError when settlement_api_url has an invalid scheme."""
    with pytest.raises(ValidationError) as excinfo:
        ClientConfig(settlement_api_url="ftp://settlement.local")
    assert "valid http or https URL" in str(excinfo.value)

    with pytest.raises(ValidationError):
        ClientConfig(settlement_api_url="not_a_url")


def test_dtype_validation():
    """Verifies ValueError when given an unsupported data type."""
    with pytest.raises(ValidationError) as excinfo:
        ClientConfig(dtype_preference="int32")
    assert "dtype_preference must be one of" in str(excinfo.value)


def test_privkey_validation():
    """Verifies validation of consumer_privkey_hex."""
    # Invalid length
    with pytest.raises(ValidationError):
        ClientConfig(consumer_privkey_hex="1234abcd")

    # Non-hex characters
    with pytest.raises(ValidationError):
        ClientConfig(consumer_privkey_hex="z" * 64)

    # Scalar 0 (out of range)
    with pytest.raises(ValidationError):
        ClientConfig(consumer_privkey_hex="0" * 64)

    # Valid key
    valid_key = "01" * 32
    config = ClientConfig(consumer_privkey_hex=valid_key)
    assert config.consumer_privkey_hex == valid_key


def test_relays_validation():
    """Verifies that empty relay list or invalid URL schemes are rejected."""
    with pytest.raises(ValidationError):
        ClientConfig(nostr_relays=[])

    with pytest.raises(ValidationError):
        ClientConfig(nostr_relays=["tcp://127.0.0.1:8008"])


def test_for_testing_factory():
    """Verifies that for_testing() returns a configured mock environment."""
    config = ClientConfig.for_testing()
    assert config.offline_mock_mode is True
    assert config.mock_relay is True
    assert config.mock_settlement is True
    assert config.mock_workers is True
    assert config.connect_timeout_seconds == 1.0
    assert config.task_timeout_seconds == 3.0
    assert config.max_retries == 1
    assert config.default_tile_size == 256

    # Verify custom overrides in for_testing
    custom = ClientConfig.for_testing(default_tile_size=512, max_retries=2)
    assert custom.default_tile_size == 512
    assert custom.max_retries == 2


def test_from_env_isolated_method():
    """Verifies ClientConfig.from_env() operates in isolated environment."""
    custom_env = {
        "MATRIX_TASK_TIMEOUT": "99.0",
        "MATRIX_MIN_WORKER_VRAM_MB": "16384",
    }
    config = ClientConfig.from_env(custom_env)
    assert config.task_timeout_seconds == 99.0
    assert config.min_worker_vram_mb == 16384
    # Ensure environment was cleaned up
    assert "MATRIX_TASK_TIMEOUT" not in os.environ
