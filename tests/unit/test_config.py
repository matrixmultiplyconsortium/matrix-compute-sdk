"""Unit tests for ClientConfig and configuration loader in matrix-compute-sdk."""

import os
import pytest
from pydantic import ValidationError

from matrix_compute.config import ClientConfig, get_default_config


def test_default_config():
    """Asserts production default values."""
    config = ClientConfig()
    assert config.nostr_relays == ["wss://relay.matrix-compute.net"]
    assert config.min_worker_vram_mb == 4096
    assert config.max_price_per_gflop == 0.001
    assert config.settlement_api_url == "https://settlement.matrix-compute.net"
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


from matrix_compute import PartitioningStrategy, WorkloadType


def test_default_config_partitioning_and_workload():
    """Asserts production default values for new Milestone 1 partitioning fields."""
    config = ClientConfig()
    assert config.partitioning_strategy == PartitioningStrategy.BLOCK_3D
    assert config.partitioning_strategy.value == "block_3d"
    assert config.workload_type == WorkloadType.GEMM_ONE_SHOT
    assert config.workload_type.value == "gemm_one_shot"
    assert config.num_strips is None
    assert config.lease_ttl_seconds == 3600
    assert config.spot_check_probability == 0.05


def test_partitioning_strategy_enum():
    """Verifies PartitioningStrategy enum values and string subclass behavior."""
    assert PartitioningStrategy.BLOCK_3D == "block_3d"
    assert PartitioningStrategy.OUTER_PRODUCT_1D == "outer_product_1d"
    assert PartitioningStrategy.OUTPUT_GRID_2D == "output_grid_2d"
    assert isinstance(PartitioningStrategy.OUTER_PRODUCT_1D, str)


def test_workload_type_enum():
    """Verifies WorkloadType enum values and string subclass behavior."""
    assert WorkloadType.GEMM_ONE_SHOT == "gemm_one_shot"
    assert WorkloadType.ITERATIVE_POWER == "iterative_power"
    assert WorkloadType.BATCH_STREAM == "batch_stream"
    assert isinstance(WorkloadType.GEMM_ONE_SHOT, str)


def test_case_insensitive_partitioning_strategy_parsing():
    """Verifies case-insensitive parsing and alias resolution for partitioning strategies."""
    cfg1 = ClientConfig(partitioning_strategy="OUTER_PRODUCT_1D")
    assert cfg1.partitioning_strategy == PartitioningStrategy.OUTER_PRODUCT_1D

    cfg2 = ClientConfig(partitioning_strategy="outer_product_1d")
    assert cfg2.partitioning_strategy == PartitioningStrategy.OUTER_PRODUCT_1D

    cfg3 = ClientConfig(partitioning_strategy="1d_outer_product")
    assert cfg3.partitioning_strategy == PartitioningStrategy.OUTER_PRODUCT_1D

    cfg4 = ClientConfig(partitioning_strategy="outer_1d")
    assert cfg4.partitioning_strategy == PartitioningStrategy.OUTER_PRODUCT_1D

    cfg5 = ClientConfig(partitioning_strategy="output_grid_2d")
    assert cfg5.partitioning_strategy == PartitioningStrategy.OUTPUT_GRID_2D

    cfg6 = ClientConfig(partitioning_strategy="block_3d")
    assert cfg6.partitioning_strategy == PartitioningStrategy.BLOCK_3D


def test_case_insensitive_workload_type_parsing():
    """Verifies case-insensitive parsing and alias resolution for workload types."""
    cfg1 = ClientConfig(workload_type="ITERATIVE_POWER")
    assert cfg1.workload_type == WorkloadType.ITERATIVE_POWER

    cfg2 = ClientConfig(workload_type="iterative_power")
    assert cfg2.workload_type == WorkloadType.ITERATIVE_POWER

    cfg3 = ClientConfig(workload_type="power")
    assert cfg3.workload_type == WorkloadType.ITERATIVE_POWER

    cfg4 = ClientConfig(workload_type="BATCH_STREAM")
    assert cfg4.workload_type == WorkloadType.BATCH_STREAM

    cfg5 = ClientConfig(workload_type="gemm_one_shot")
    assert cfg5.workload_type == WorkloadType.GEMM_ONE_SHOT


def test_partitioning_and_workload_validation_errors():
    """Verifies validation errors for invalid enum strings and boundary bounds."""
    # Invalid strategy string
    with pytest.raises(ValidationError) as exc:
        ClientConfig(partitioning_strategy="nonexistent_strategy")
    assert "Invalid partitioning_strategy" in str(exc.value)

    # Invalid workload type
    with pytest.raises(ValidationError) as exc:
        ClientConfig(workload_type="invalid_workload")
    assert "Invalid workload_type" in str(exc.value)

    # num_strips ge=1 violation
    with pytest.raises(ValidationError):
        ClientConfig(num_strips=0)

    with pytest.raises(ValidationError):
        ClientConfig(num_strips=-4)

    # lease_ttl_seconds ge=1 violation
    with pytest.raises(ValidationError):
        ClientConfig(lease_ttl_seconds=0)

    # spot_check_probability range [0.0, 1.0] violations
    with pytest.raises(ValidationError):
        ClientConfig(spot_check_probability=-0.01)

    with pytest.raises(ValidationError):
        ClientConfig(spot_check_probability=1.05)


def test_environment_variable_override_new_fields(monkeypatch):
    """Verifies that environment variables correctly override partitioning fields."""
    monkeypatch.setenv("MATRIX_PARTITIONING_STRATEGY", "outer_product_1d")
    monkeypatch.setenv("MATRIX_WORKLOAD_TYPE", "iterative_power")
    monkeypatch.setenv("MATRIX_NUM_STRIPS", "8")
    monkeypatch.setenv("MATRIX_LEASE_TTL_SECONDS", "7200")
    monkeypatch.setenv("MATRIX_SPOT_CHECK_PROBABILITY", "0.20")

    config = ClientConfig()
    assert config.partitioning_strategy == PartitioningStrategy.OUTER_PRODUCT_1D
    assert config.workload_type == WorkloadType.ITERATIVE_POWER
    assert config.num_strips == 8
    assert config.lease_ttl_seconds == 7200
    assert config.spot_check_probability == 0.20


def test_environment_variable_fallback_aliases(monkeypatch):
    """Verifies fallback environment variable names (e.g. PARTITIONING_STRATEGY, WORKLOAD_PROFILE)."""
    monkeypatch.setenv("PARTITIONING_STRATEGY", "OUTER_1D")
    monkeypatch.setenv("WORKLOAD_PROFILE", "batch_stream")
    monkeypatch.setenv("NUM_STRIPS", "4")

    config = ClientConfig()
    assert config.partitioning_strategy == PartitioningStrategy.OUTER_PRODUCT_1D
    assert config.workload_type == WorkloadType.BATCH_STREAM
    assert config.num_strips == 4


def test_public_api_exports():
    """Verifies top-level exports from matrix_compute."""
    import matrix_compute

    assert hasattr(matrix_compute, "PartitioningStrategy")
    assert hasattr(matrix_compute, "WorkloadType")
    assert matrix_compute.PartitioningStrategy.BLOCK_3D == "block_3d"
    assert matrix_compute.WorkloadType.GEMM_ONE_SHOT == "gemm_one_shot"


def test_nwc_uri_default():
    """Verifies nwc_uri defaults to None and computed helpers return inactive state."""
    config = ClientConfig()
    assert config.nwc_uri is None
    assert config.is_nwc_enabled is False
    assert config.nwc_params is None


def test_nwc_uri_explicit():
    """Verifies explicit nwc_uri is accepted and parsed into nwc_params."""
    valid_uri = (
        "nostr+walletconnect://0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        "?relay=wss://relay.damus.io&secret=fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
    )
    config = ClientConfig(nwc_uri=valid_uri)
    assert config.nwc_uri == valid_uri
    assert config.is_nwc_enabled is True
    assert config.nwc_params is not None
    assert config.nwc_params.wallet_pubkey == "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def test_nwc_uri_environment_override_priority(monkeypatch):
    """Verifies MATRIX_NWC_URI, NWC_URI, and NOSTR_WALLET_CONNECT_URI precedence."""
    uri1 = "nostr+walletconnect://1111111111111111111111111111111111111111111111111111111111111111?relay=wss://r1.com&secret=1111111111111111111111111111111111111111111111111111111111111111"
    uri2 = "nostr+walletconnect://2222222222222222222222222222222222222222222222222222222222222222?relay=wss://r2.com&secret=2222222222222222222222222222222222222222222222222222222222222222"
    uri3 = "nostr+walletconnect://3333333333333333333333333333333333333333333333333333333333333333?relay=wss://r3.com&secret=3333333333333333333333333333333333333333333333333333333333333333"

    # Test lowest priority: NOSTR_WALLET_CONNECT_URI
    monkeypatch.setenv("NOSTR_WALLET_CONNECT_URI", uri3)
    cfg3 = ClientConfig()
    assert cfg3.nwc_uri == uri3

    # Test middle priority: NWC_URI overrides NOSTR_WALLET_CONNECT_URI
    monkeypatch.setenv("NWC_URI", uri2)
    cfg2 = ClientConfig()
    assert cfg2.nwc_uri == uri2

    # Test top priority: MATRIX_NWC_URI overrides NWC_URI
    monkeypatch.setenv("MATRIX_NWC_URI", uri1)
    cfg1 = ClientConfig()
    assert cfg1.nwc_uri == uri1


def test_nwc_uri_validation_error_on_invalid_uri():
    """Verifies that invalid nwc_uri raises Pydantic ValidationError on instantiation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        ClientConfig(nwc_uri="ftp://bad_scheme")
    assert "nostr+walletconnect://" in str(excinfo.value)
