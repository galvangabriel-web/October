"""
Configuration module for managing dataset paths and loading strategies.

This module provides per-tab dataset configuration, allowing different
dashboard tabs to use dedicated datasets for data independence.
"""

from .dataset_config import (
    DatasetConfig,
    DatasetLoader,
    DatasetValidationError,
    get_default_config,
)

__all__ = [
    "DatasetConfig",
    "DatasetLoader",
    "DatasetValidationError",
    "get_default_config",
]
