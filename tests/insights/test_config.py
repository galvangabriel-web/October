"""
Unit Tests for Configuration Management

Tests InsightsConfig class:
- Default configuration
- Custom configuration
- Environment variable override
- Configuration validation
- Configuration serialization
"""

import pytest
import os
from src.insights import InsightsConfig, DEFAULT_CONFIG


class TestDefaultConfiguration:
    """Test default configuration values."""

    def test_default_config_exists(self):
        """Test default config is available."""
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, InsightsConfig)

    def test_default_brake_thresholds(self):
        """Test default brake pressure thresholds."""
        assert DEFAULT_CONFIG.hard_brake_threshold == 100.0
        assert DEFAULT_CONFIG.trail_brake_threshold == 0.3
        assert DEFAULT_CONFIG.brake_consistency_threshold == 0.15

    def test_default_corner_thresholds(self):
        """Test default corner detection thresholds."""
        assert DEFAULT_CONFIG.min_corner_duration == 1.0
        assert DEFAULT_CONFIG.lateral_g_threshold == 0.5
        assert DEFAULT_CONFIG.speed_drop_threshold == 10.0

    def test_default_consistency_thresholds(self):
        """Test default consistency thresholds."""
        assert DEFAULT_CONFIG.outlier_threshold == 2.0
        assert DEFAULT_CONFIG.min_lap_count == 3

    def test_default_performance_settings(self):
        """Test default performance settings."""
        assert DEFAULT_CONFIG.max_memory_mb == 4096
        assert DEFAULT_CONFIG.chunk_size == 100000
        assert DEFAULT_CONFIG.enable_caching is True


class TestCustomConfiguration:
    """Test custom configuration creation."""

    def test_create_custom_config(self):
        """Test creating config with custom values."""
        config = InsightsConfig(
            hard_brake_threshold=120.0,
            trail_brake_threshold=0.4,
            min_corner_duration=1.5
        )

        assert config.hard_brake_threshold == 120.0
        assert config.trail_brake_threshold == 0.4
        assert config.min_corner_duration == 1.5

    def test_partial_override(self):
        """Test overriding only some values."""
        config = InsightsConfig(
            hard_brake_threshold=110.0
        )

        # Overridden value
        assert config.hard_brake_threshold == 110.0

        # Default values still present
        assert config.trail_brake_threshold == 0.3
        assert config.min_corner_duration == 1.0

    def test_all_parameters_override(self):
        """Test overriding all configurable parameters."""
        config = InsightsConfig(
            hard_brake_threshold=115.0,
            trail_brake_threshold=0.35,
            brake_consistency_threshold=0.2,
            min_corner_duration=1.2,
            lateral_g_threshold=0.6,
            speed_drop_threshold=15.0,
            outlier_threshold=2.5,
            min_lap_count=5,
            throttle_smooth_threshold=0.25,
            steering_smooth_threshold=0.3,
            apex_speed_factor=0.85,
            max_memory_mb=8192,
            chunk_size=50000,
            enable_caching=False,
            enable_performance_logging=True,
            log_level="DEBUG"
        )

        assert config.hard_brake_threshold == 115.0
        assert config.trail_brake_threshold == 0.35
        assert config.brake_consistency_threshold == 0.2
        assert config.min_corner_duration == 1.2
        assert config.lateral_g_threshold == 0.6
        assert config.speed_drop_threshold == 15.0
        assert config.outlier_threshold == 2.5
        assert config.min_lap_count == 5
        assert config.throttle_smooth_threshold == 0.25
        assert config.steering_smooth_threshold == 0.3
        assert config.apex_speed_factor == 0.85
        assert config.max_memory_mb == 8192
        assert config.chunk_size == 50000
        assert config.enable_caching is False
        assert config.enable_performance_logging is True
        assert config.log_level == "DEBUG"


class TestEnvironmentVariableOverride:
    """Test configuration from environment variables."""

    def test_env_var_override_brake_threshold(self):
        """Test brake threshold from environment."""
        os.environ['INSIGHTS_HARD_BRAKE_THRESHOLD'] = '125.0'

        config = InsightsConfig()

        assert config.hard_brake_threshold == 125.0

        # Cleanup
        del os.environ['INSIGHTS_HARD_BRAKE_THRESHOLD']

    def test_env_var_override_multiple(self):
        """Test multiple env var overrides."""
        os.environ['INSIGHTS_HARD_BRAKE_THRESHOLD'] = '125.0'
        os.environ['INSIGHTS_MIN_CORNER_DURATION'] = '1.8'
        os.environ['INSIGHTS_OUTLIER_THRESHOLD'] = '3.0'

        config = InsightsConfig()

        assert config.hard_brake_threshold == 125.0
        assert config.min_corner_duration == 1.8
        assert config.outlier_threshold == 3.0

        # Cleanup
        del os.environ['INSIGHTS_HARD_BRAKE_THRESHOLD']
        del os.environ['INSIGHTS_MIN_CORNER_DURATION']
        del os.environ['INSIGHTS_OUTLIER_THRESHOLD']

    def test_env_var_boolean_values(self):
        """Test boolean env var parsing."""
        os.environ['INSIGHTS_ENABLE_CACHING'] = 'false'
        os.environ['INSIGHTS_ENABLE_PERFORMANCE_LOGGING'] = 'true'

        config = InsightsConfig()

        assert config.enable_caching is False
        assert config.enable_performance_logging is True

        # Cleanup
        del os.environ['INSIGHTS_ENABLE_CACHING']
        del os.environ['INSIGHTS_ENABLE_PERFORMANCE_LOGGING']

    def test_env_var_integer_values(self):
        """Test integer env var parsing."""
        os.environ['INSIGHTS_MAX_MEMORY_MB'] = '16384'
        os.environ['INSIGHTS_CHUNK_SIZE'] = '200000'

        config = InsightsConfig()

        assert config.max_memory_mb == 16384
        assert config.chunk_size == 200000

        # Cleanup
        del os.environ['INSIGHTS_MAX_MEMORY_MB']
        del os.environ['INSIGHTS_CHUNK_SIZE']

    def test_constructor_overrides_env_var(self):
        """Test constructor parameter takes precedence over env var."""
        os.environ['INSIGHTS_HARD_BRAKE_THRESHOLD'] = '125.0'

        config = InsightsConfig(hard_brake_threshold=130.0)

        # Constructor value should win
        assert config.hard_brake_threshold == 130.0

        # Cleanup
        del os.environ['INSIGHTS_HARD_BRAKE_THRESHOLD']


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_valid_config_values(self):
        """Test validation passes for valid values."""
        # Should not raise
        config = InsightsConfig(
            hard_brake_threshold=100.0,
            trail_brake_threshold=0.3,
            min_corner_duration=1.0
        )

        assert config.hard_brake_threshold > 0
        assert 0 <= config.trail_brake_threshold <= 1
        assert config.min_corner_duration > 0

    def test_negative_threshold_warning(self):
        """Test that negative thresholds are handled."""
        # Configuration should handle this gracefully
        # (may log warning but should not crash)
        config = InsightsConfig(hard_brake_threshold=-10.0)

        # Value should be set (even if negative)
        assert config.hard_brake_threshold == -10.0

    def test_zero_values(self):
        """Test zero values in configuration."""
        config = InsightsConfig(
            min_corner_duration=0.0,
            lateral_g_threshold=0.0
        )

        assert config.min_corner_duration == 0.0
        assert config.lateral_g_threshold == 0.0


class TestConfigurationSerialization:
    """Test configuration serialization and comparison."""

    def test_config_equality(self):
        """Test two identical configs are equal."""
        config1 = InsightsConfig(hard_brake_threshold=110.0)
        config2 = InsightsConfig(hard_brake_threshold=110.0)

        # Should have same values
        assert config1.hard_brake_threshold == config2.hard_brake_threshold

    def test_config_inequality(self):
        """Test different configs are not equal."""
        config1 = InsightsConfig(hard_brake_threshold=110.0)
        config2 = InsightsConfig(hard_brake_threshold=120.0)

        assert config1.hard_brake_threshold != config2.hard_brake_threshold

    def test_config_repr(self):
        """Test string representation of config."""
        config = InsightsConfig(hard_brake_threshold=110.0)

        repr_str = repr(config)

        assert 'InsightsConfig' in repr_str
        assert '110' in repr_str or '110.0' in repr_str


class TestConfigurationUsage:
    """Test configuration in real-world scenarios."""

    def test_config_in_profiler(self, sample_telemetry_df, sample_lap_times_df, custom_config):
        """Test using custom config with DriverProfiler."""
        from src.insights import DriverProfiler

        # Should accept config parameter
        profiler = DriverProfiler(config=custom_config)

        assert profiler.config == custom_config
        assert profiler.config.hard_brake_threshold == 120.0

    def test_config_in_corner_analyzer(self, custom_config):
        """Test using custom config with CornerAnalyzer."""
        from src.insights import CornerAnalyzer

        analyzer = CornerAnalyzer(config=custom_config)

        assert analyzer.config == custom_config
        assert analyzer.config.min_corner_duration == 1.5

    def test_config_in_consistency_tracker(self, custom_config):
        """Test using custom config with ConsistencyTracker."""
        from src.insights import ConsistencyTracker

        tracker = ConsistencyTracker(config=custom_config)

        assert tracker.config == custom_config
        assert tracker.config.outlier_threshold == 2.5

    def test_default_config_when_none_provided(self):
        """Test default config is used when none provided."""
        from src.insights import DriverProfiler

        profiler = DriverProfiler()  # No config parameter

        assert profiler.config is not None
        # Should use default values
        assert profiler.config.hard_brake_threshold == 100.0


class TestConfigurationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_values(self):
        """Test config with extreme values."""
        config = InsightsConfig(
            hard_brake_threshold=1000.0,
            trail_brake_threshold=0.99,
            min_corner_duration=0.001,
            max_memory_mb=1048576  # 1 TB
        )

        assert config.hard_brake_threshold == 1000.0
        assert config.trail_brake_threshold == 0.99
        assert config.min_corner_duration == 0.001
        assert config.max_memory_mb == 1048576

    def test_floating_point_precision(self):
        """Test floating point values are preserved."""
        config = InsightsConfig(
            trail_brake_threshold=0.333333333,
            apex_speed_factor=0.777777777
        )

        # Values should be close (within floating point precision)
        assert abs(config.trail_brake_threshold - 0.333333333) < 1e-9
        assert abs(config.apex_speed_factor - 0.777777777) < 1e-9

    def test_config_immutability(self):
        """Test that config values can be modified after creation."""
        config = InsightsConfig(hard_brake_threshold=100.0)

        # Dataclass should allow modification (mutable by default)
        config.hard_brake_threshold = 120.0

        assert config.hard_brake_threshold == 120.0

    def test_unknown_env_var(self):
        """Test that unknown env vars are ignored."""
        os.environ['INSIGHTS_NONEXISTENT_PARAM'] = '999'

        # Should not crash
        config = InsightsConfig()

        # Cleanup
        del os.environ['INSIGHTS_NONEXISTENT_PARAM']

    def test_invalid_env_var_type(self):
        """Test handling of invalid env var type conversion."""
        os.environ['INSIGHTS_HARD_BRAKE_THRESHOLD'] = 'not_a_number'

        # Should handle gracefully (may use default or raise)
        try:
            config = InsightsConfig()
            # If it succeeds, should use default
            assert config.hard_brake_threshold == 100.0
        except (ValueError, TypeError):
            # Or it might raise - either is acceptable
            pass

        # Cleanup
        del os.environ['INSIGHTS_HARD_BRAKE_THRESHOLD']
