"""
Unit test suite for ClientConfig iterative workload configuration in matrix-compute-sdk.

Tests:
1. WorkloadType.ITERATIVE_POWER enum values and string behavior.
2. ClientConfig default values and helper property `is_iterative_workload`.
3. Case-insensitive alias parsing ("iterative", "power", "power_iteration").
4. Bounds validation for lease_ttl_seconds (ge=1) and spot_check_probability (ge=0.0, le=1.0).
5. Environment variable overrides (MATRIX_* and fallback names).
6. ClientConfig.from_env() isolated loader.
7. ClientConfig.for_testing() factory presets and kwargs overrides.
"""

from __future__ import annotations

import os
import pytest
from pydantic import ValidationError

import matrix_compute
from matrix_compute.config import (
    ClientConfig,
    WorkloadType,
    get_default_config,
)


# ==============================================================================
# 1. WorkloadType Enum Verification
# ==============================================================================

class TestWorkloadTypeEnum:
    """Verifies WorkloadType enum values, string inheritance, and membership."""

    def test_tc_cfg_01_enum_values_and_types(self):
        """TC_CFG_01: Asserts WorkloadType values and string subclass behavior."""
        assert WorkloadType.GEMM_ONE_SHOT == "gemm_one_shot"
        assert WorkloadType.ITERATIVE_POWER == "iterative_power"
        assert WorkloadType.BATCH_STREAM == "batch_stream"

        assert isinstance(WorkloadType.ITERATIVE_POWER, str)
        assert isinstance(WorkloadType.ITERATIVE_POWER, WorkloadType)
        assert str(WorkloadType.ITERATIVE_POWER) == "iterative_power"

    def test_tc_cfg_02_enum_members_contain_iterative(self):
        """TC_CFG_02: Confirms ITERATIVE_POWER is present in enum member set."""
        members = set(WorkloadType.__members__.keys())
        expected = {"GEMM_ONE_SHOT", "ITERATIVE_POWER", "BATCH_STREAM"}
        assert expected.issubset(members)


# ==============================================================================
# 2. ClientConfig Defaults & Helper Properties
# ==============================================================================

class TestClientConfigDefaultsAndHelpers:
    """Verifies default values and property helpers on ClientConfig."""

    def test_tc_cfg_03_default_configuration_values(self):
        """TC_CFG_03: Asserts standard production defaults for iterative fields."""
        config = ClientConfig()
        assert config.workload_type == WorkloadType.GEMM_ONE_SHOT
        assert config.lease_ttl_seconds == 3600
        assert config.spot_check_probability == 0.05
        assert config.is_iterative_workload is False

    def test_tc_cfg_04_is_iterative_workload_property(self):
        """TC_CFG_04: Verifies is_iterative_workload property returns True only for ITERATIVE_POWER."""
        cfg_iter = ClientConfig(workload_type=WorkloadType.ITERATIVE_POWER)
        assert cfg_iter.is_iterative_workload is True

        cfg_gemm = ClientConfig(workload_type=WorkloadType.GEMM_ONE_SHOT)
        assert cfg_gemm.is_iterative_workload is False

        cfg_batch = ClientConfig(workload_type=WorkloadType.BATCH_STREAM)
        assert cfg_batch.is_iterative_workload is False

    def test_tc_cfg_05_get_default_config_iterative_defaults(self):
        """TC_CFG_05: Verifies get_default_config() helper reflects standard defaults."""
        config = get_default_config()
        assert isinstance(config, ClientConfig)
        assert config.lease_ttl_seconds == 3600
        assert config.spot_check_probability == 0.05
        assert config.is_iterative_workload is False


# ==============================================================================
# 3. Alias Parsing & Normalization
# ==============================================================================

class TestIterativeAliasParsing:
    """Verifies case-insensitive parsing and alias mapping for WorkloadType."""

    @pytest.mark.parametrize(
        "raw_input,expected_enum",
        [
            ("iterative_power", WorkloadType.ITERATIVE_POWER),
            ("ITERATIVE_POWER", WorkloadType.ITERATIVE_POWER),
            ("Iterative_Power", WorkloadType.ITERATIVE_POWER),
            ("iterative", WorkloadType.ITERATIVE_POWER),
            ("ITERATIVE", WorkloadType.ITERATIVE_POWER),
            ("power", WorkloadType.ITERATIVE_POWER),
            ("POWER", WorkloadType.ITERATIVE_POWER),
            ("power_iteration", WorkloadType.ITERATIVE_POWER),
            ("POWER_ITERATION", WorkloadType.ITERATIVE_POWER),
            ("Power_Iteration", WorkloadType.ITERATIVE_POWER),
            ("gemm", WorkloadType.GEMM_ONE_SHOT),
            ("one_shot", WorkloadType.GEMM_ONE_SHOT),
            ("batch", WorkloadType.BATCH_STREAM),
            ("stream", WorkloadType.BATCH_STREAM),
        ],
    )
    def test_tc_cfg_06_case_insensitive_aliases(self, raw_input: str, expected_enum: WorkloadType):
        """TC_CFG_06: Validates resolution of diverse string representations."""
        config = ClientConfig(workload_type=raw_input)
        assert config.workload_type == expected_enum

    def test_tc_cfg_07_invalid_workload_type_rejected(self):
        """TC_CFG_07: Invalid workload type strings or types raise ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            ClientConfig(workload_type="invalid_workload_type_xyz")
        assert "Invalid workload_type" in str(excinfo.value)

        with pytest.raises(ValidationError):
            ClientConfig(workload_type=12345)


# ==============================================================================
# 4. Bounds & Range Validation
# ==============================================================================

class TestIterativeBoundsValidation:
    """Verifies mathematical boundary constraints on iterative settings."""

    def test_tc_cfg_08_lease_ttl_seconds_bounds(self):
        """TC_CFG_08: lease_ttl_seconds must be a positive integer (ge=1)."""
        # Valid boundaries
        assert ClientConfig(lease_ttl_seconds=1).lease_ttl_seconds == 1
        assert ClientConfig(lease_ttl_seconds=86400).lease_ttl_seconds == 86400

        # Invalid bounds: 0 or negative
        with pytest.raises(ValidationError):
            ClientConfig(lease_ttl_seconds=0)

        with pytest.raises(ValidationError):
            ClientConfig(lease_ttl_seconds=-3600)

    def test_tc_cfg_09_spot_check_probability_bounds(self):
        """TC_CFG_09: spot_check_probability must be within [0.0, 1.0]."""
        # Valid boundaries
        assert ClientConfig(spot_check_probability=0.0).spot_check_probability == 0.0
        assert ClientConfig(spot_check_probability=1.0).spot_check_probability == 1.0
        assert ClientConfig(spot_check_probability=0.5).spot_check_probability == 0.5

        # Invalid bounds: < 0.0 or > 1.0
        with pytest.raises(ValidationError):
            ClientConfig(spot_check_probability=-0.0001)

        with pytest.raises(ValidationError):
            ClientConfig(spot_check_probability=1.0001)

        with pytest.raises(ValidationError):
            ClientConfig(spot_check_probability=5.0)


# ==============================================================================
# 5. Environment Variable Overrides
# ==============================================================================

class TestEnvironmentVariableOverrides:
    """Verifies environment variable resolution and cascading."""

    def test_tc_cfg_10_primary_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        """TC_CFG_10: MATRIX_* environment variables override defaults."""
        monkeypatch.setenv("MATRIX_WORKLOAD_TYPE", "iterative_power")
        monkeypatch.setenv("MATRIX_LEASE_TTL_SECONDS", "7200")
        monkeypatch.setenv("MATRIX_SPOT_CHECK_PROBABILITY", "0.20")

        config = ClientConfig()
        assert config.workload_type == WorkloadType.ITERATIVE_POWER
        assert config.is_iterative_workload is True
        assert config.lease_ttl_seconds == 7200
        assert config.spot_check_probability == 0.20

    def test_tc_cfg_11_fallback_aliases_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        """TC_CFG_11: Fallback environment variables (WORKLOAD_TYPE, LEASE_TTL_SECONDS) are honored."""
        monkeypatch.setenv("WORKLOAD_PROFILE", "power_iteration")
        monkeypatch.setenv("LEASE_TTL_SECONDS", "1800")
        monkeypatch.setenv("SPOT_CHECK_PROBABILITY", "0.10")

        config = ClientConfig()
        assert config.workload_type == WorkloadType.ITERATIVE_POWER
        assert config.is_iterative_workload is True
        assert config.lease_ttl_seconds == 1800
        assert config.spot_check_probability == 0.10

    def test_tc_cfg_12_from_env_isolated_loader(self):
        """TC_CFG_12: ClientConfig.from_env() parses isolated mapping without polluting os.environ."""
        custom_env = {
            "MATRIX_WORKLOAD_TYPE": "power",
            "MATRIX_LEASE_TTL_SECONDS": "900",
            "MATRIX_SPOT_CHECK_PROBABILITY": "0.15",
        }
        config = ClientConfig.from_env(custom_env)
        assert config.workload_type == WorkloadType.ITERATIVE_POWER
        assert config.is_iterative_workload is True
        assert config.lease_ttl_seconds == 900
        assert config.spot_check_probability == 0.15
        assert "MATRIX_WORKLOAD_TYPE" not in os.environ


# ==============================================================================
# 6. Test Double Factory Presets (`for_testing`)
# ==============================================================================

class TestForTestingFactoryPresets:
    """Verifies ClientConfig.for_testing() factory presets."""

    def test_tc_cfg_13_for_testing_standard_presets(self):
        """TC_CFG_13: for_testing() preserves offline mock flags and fast timeouts."""
        config = ClientConfig.for_testing()
        assert config.offline_mock_mode is True
        assert config.mock_workers is True
        assert config.mock_settlement is True
        assert config.mock_relay is True
        assert config.connect_timeout_seconds == 1.0
        assert config.task_timeout_seconds == 3.0

    def test_tc_cfg_14_for_testing_iterative_kwargs_override(self):
        """TC_CFG_14: for_testing() accepts iterative overrides."""
        config = ClientConfig.for_testing(
            workload_type=WorkloadType.ITERATIVE_POWER,
            lease_ttl_seconds=60,
            spot_check_probability=1.0,
        )
        assert config.is_iterative_workload is True
        assert config.lease_ttl_seconds == 60
        assert config.spot_check_probability == 1.0

    def test_tc_cfg_15_for_testing_with_alias_string(self):
        """TC_CFG_15: for_testing() correctly parses alias string in kwargs."""
        config = ClientConfig.for_testing(workload_type="power")
        assert config.workload_type == WorkloadType.ITERATIVE_POWER
        assert config.is_iterative_workload is True


# ==============================================================================
# 7. Public API Export Check
# ==============================================================================

def test_tc_cfg_16_public_api_exports():
    """TC_CFG_16: Asserts WorkloadType and ClientConfig exported from top-level package."""
    assert hasattr(matrix_compute, "WorkloadType")
    assert hasattr(matrix_compute, "ClientConfig")
    assert matrix_compute.WorkloadType.ITERATIVE_POWER == "iterative_power"
