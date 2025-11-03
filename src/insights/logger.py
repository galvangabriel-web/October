"""
Production Logging Infrastructure for Racing Insights Module

Enterprise-grade logging system providing:
- Structured logging with loguru
- Log rotation and retention
- Performance tracking
- Context-aware logging
- Production-safe configuration

Design Pattern: Singleton Logger Pattern
- Centralized logger configuration
- Consistent log formatting
- Performance monitoring decorators
- Thread-safe operation

Log Levels:
- DEBUG: Detailed diagnostic information
- INFO: General operational information
- WARNING: Warning messages for unexpected behavior
- ERROR: Error events that might still allow continued execution
- CRITICAL: Severe error events that may cause premature termination

Author: Production Engineering Team
Version: 1.0.0
License: GR Cup 2025 Hackathon
"""

import sys
import time
import functools
from pathlib import Path
from typing import Callable, Any, Optional
from loguru import logger

from .config import InsightsConfig, DEFAULT_CONFIG


# ==============================================================================
# LOGGER CONFIGURATION
# ==============================================================================

# Module-level flag to track if logger is configured
_logger_configured = False


def configure_logger(
    config: Optional[InsightsConfig] = None,
    log_dir: str = "logs",
    enable_console: bool = True
) -> None:
    """
    Configure production logger with rotation and retention.

    Should be called once at application startup. Subsequent calls are ignored
    to prevent duplicate logging.

    Parameters:
        config: Configuration object (uses DEFAULT_CONFIG if None)
        log_dir: Directory for log files
        enable_console: Whether to log to console (stdout)

    Example:
        # Configure with defaults
        configure_logger()

        # Custom configuration
        config = InsightsConfig(log_level="DEBUG")
        configure_logger(config, log_dir="custom_logs")
    """
    global _logger_configured

    if _logger_configured:
        return  # Already configured

    cfg = config or DEFAULT_CONFIG

    # Remove default handler
    logger.remove()

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Console handler (if enabled)
    if enable_console:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=cfg.log_level,
            colorize=True
        )

    # File handler - main log
    logger.add(
        log_path / "insights_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=cfg.log_level,
        rotation=cfg.log_rotation_size,
        retention=f"{cfg.log_retention_days} days",
        compression="zip",
        encoding="utf-8"
    )

    # File handler - errors only
    logger.add(
        log_path / "insights_errors_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        level="ERROR",
        rotation=cfg.log_rotation_size,
        retention=f"{cfg.log_retention_days} days",
        compression="zip",
        encoding="utf-8"
    )

    _logger_configured = True
    logger.info("Insights logger configured successfully")


# Initialize logger with defaults
configure_logger()


# ==============================================================================
# PERFORMANCE TRACKING DECORATOR
# ==============================================================================

def log_performance(func: Callable) -> Callable:
    """
    Decorator to log function execution time and success/failure.

    Automatically logs:
    - Function entry with parameters
    - Execution time
    - Success/failure status
    - Exception details on failure

    Example:
        @log_performance
        def create_profile(telemetry_df, lap_times_df, vehicle_number):
            # ... implementation ...
            return profile

        # Logs:
        # INFO: Starting create_profile | vehicle_number=5
        # INFO: Completed create_profile in 1.23s
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Build parameter string
        params = []
        if args:
            params.extend([str(arg)[:50] for arg in args[:2]])  # First 2 args only
        if kwargs:
            params.extend([f"{k}={v}" for k, v in list(kwargs.items())[:3]])  # First 3 kwargs
        param_str = ", ".join(params) if params else "no params"

        logger.debug(f"Starting {func.__name__} | {param_str}")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {duration:.2f}s: {str(e)}")
            raise

    return wrapper


def log_method_performance(method: Callable) -> Callable:
    """
    Decorator for class methods (handles 'self' parameter).

    Example:
        class DriverProfiler:
            @log_method_performance
            def create_profile(self, telemetry_df, lap_times_df, vehicle_number):
                # ... implementation ...
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Build parameter string (skip 'self')
        params = []
        if args:
            params.extend([str(arg)[:50] for arg in args[:2]])
        if kwargs:
            params.extend([f"{k}={v}" for k, v in list(kwargs.items())[:3]])
        param_str = ", ".join(params) if params else "no params"

        class_name = self.__class__.__name__
        logger.debug(f"Starting {class_name}.{method.__name__} | {param_str}")

        start_time = time.time()
        try:
            result = method(self, *args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Completed {class_name}.{method.__name__} in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed {class_name}.{method.__name__} after {duration:.2f}s: {str(e)}")
            raise

    return wrapper


# ==============================================================================
# CONTEXT LOGGING UTILITIES
# ==============================================================================

def log_data_load(data_type: str, record_count: int, vehicle_number: Optional[int] = None) -> None:
    """
    Log data loading operation.

    Parameters:
        data_type: Type of data loaded (telemetry, lap_times, etc.)
        record_count: Number of records loaded
        vehicle_number: Vehicle ID (if applicable)

    Example:
        log_data_load("telemetry", len(df), vehicle_number=5)
        # INFO: Loaded 50000 telemetry records for vehicle 5
    """
    if vehicle_number is not None:
        logger.info(f"Loaded {record_count:,} {data_type} records for vehicle {vehicle_number}")
    else:
        logger.info(f"Loaded {record_count:,} {data_type} records")


def log_analysis_start(analysis_type: str, **context) -> None:
    """
    Log start of analysis operation.

    Parameters:
        analysis_type: Type of analysis being performed
        **context: Additional context (vehicle_number, session_id, etc.)

    Example:
        log_analysis_start("driver profiling", vehicle_number=5, track="sebring")
        # INFO: Starting driver profiling | vehicle_number=5, track=sebring
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items()) if context else "no context"
    logger.info(f"Starting {analysis_type} | {context_str}")


def log_metric_calculation(metric_name: str, value: Any) -> None:
    """
    Log metric calculation result.

    Parameters:
        metric_name: Name of metric
        value: Calculated value

    Example:
        log_metric_calculation("consistency_score", 85.3)
        # DEBUG: Calculated consistency_score = 85.3
    """
    logger.debug(f"Calculated {metric_name} = {value}")


def log_data_quality_warning(issue: str, **context) -> None:
    """
    Log data quality warning.

    Parameters:
        issue: Description of data quality issue
        **context: Additional context

    Example:
        log_data_quality_warning("Low lap count", vehicle_number=5, lap_count=2)
        # WARNING: Low lap count | vehicle_number=5, lap_count=2
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items()) if context else ""
    logger.warning(f"{issue} | {context_str}")


def log_corner_detection(corner_count: int, threshold: float) -> None:
    """
    Log corner detection result.

    Parameters:
        corner_count: Number of corners detected
        threshold: Speed threshold used

    Example:
        log_corner_detection(8, 120.0)
        # INFO: Detected 8 corners (threshold: 120.0 km/h)
    """
    logger.info(f"Detected {corner_count} corners (threshold: {threshold:.1f} km/h)")


# ==============================================================================
# EXCEPTION LOGGING UTILITIES
# ==============================================================================

def log_exception_with_context(exc: Exception, context: Optional[dict] = None) -> None:
    """
    Log exception with additional context.

    Parameters:
        exc: Exception to log
        context: Additional context information

    Example:
        try:
            # ... some operation ...
        except Exception as e:
            log_exception_with_context(e, context={"vehicle_number": 5})
            raise
    """
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.exception(f"{exc.__class__.__name__}: {exc} | Context: {context_str}")
    else:
        logger.exception(f"{exc.__class__.__name__}: {exc}")


# ==============================================================================
# EXPORT LOGGER INSTANCE
# ==============================================================================

# Export logger instance for direct use
__all__ = [
    'logger',
    'configure_logger',
    'log_performance',
    'log_method_performance',
    'log_data_load',
    'log_analysis_start',
    'log_metric_calculation',
    'log_data_quality_warning',
    'log_corner_detection',
    'log_exception_with_context'
]
