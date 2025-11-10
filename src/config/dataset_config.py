"""
Dataset configuration and loading module.

This module provides:
- Per-tab dataset configuration
- Dataset loading with validation
- Fallback to global datasets
- YAML-based configuration management

Architecture:
- Each tab can have a dedicated dataset for data independence
- Configuration stored in dataset_config.yaml
- Automatic validation against data dictionary requirements
- Graceful fallback to global dataset if tab-specific not found
"""

import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


class DatasetConfig:
    """
    Manages dataset configuration for dashboard tabs.

    Supports per-tab dataset paths to ensure data independence
    and enable tab-specific optimizations.

    Configuration Format (YAML):
        global:
          dataset_path: "data/master_racing_data.csv"
          fallback_enabled: true

        tabs:
          winning_edge:
            dataset_path: "data/winning_edge_dataset.csv"
            description: "Dedicated dataset for Winning Edge tab"
            enabled: true

          post_race:
            dataset_path: "data/post_race_dataset.csv"
            enabled: false  # Falls back to global

    Usage:
        config = DatasetConfig()
        path = config.get_dataset_path("winning_edge")  # Tab-specific
        path = config.get_dataset_path()  # Global dataset
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize dataset configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default location.
        """
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config()
        logger.info(f"Loaded dataset config from {self.config_path}")

    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """Resolve configuration file path."""
        if config_path:
            return Path(config_path)

        # Try multiple locations in priority order
        candidates = [
            Path("src/config/dataset_config.yaml"),  # Development
            Path("config/dataset_config.yaml"),  # Production
            Path("/home/tactical/racing_analytics/config/dataset_config.yaml"),  # Absolute prod
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Return default location (will be created if doesn't exist)
        return Path("src/config/dataset_config.yaml")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._get_default_config()

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Validate config structure
            self._validate_config(config)
            return config

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config YAML: {e}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure."""
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        if 'global' not in config:
            raise ValueError("Config missing 'global' section")

        if 'dataset_path' not in config['global']:
            raise ValueError("Config missing 'global.dataset_path'")

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'global': {
                'dataset_path': 'master_racing_data.csv',
                'fallback_enabled': True,
                'cache_enabled': False,
            },
            'tabs': {}
        }

    def get_dataset_path(self, tab_name: Optional[str] = None) -> str:
        """
        Get dataset path for a specific tab or global dataset.

        Args:
            tab_name: Dashboard tab name (e.g., "winning_edge", "post_race").
                     If None, returns global dataset path.

        Returns:
            Absolute path to dataset file.

        Resolution Order:
            1. Tab-specific dataset (if tab_name provided and enabled)
            2. Global dataset (fallback)

        Example:
            >>> config = DatasetConfig()
            >>> config.get_dataset_path("winning_edge")
            'data/winning_edge_dataset.csv'
            >>> config.get_dataset_path()
            'master_racing_data.csv'
        """
        # If no tab specified, return global dataset
        if not tab_name:
            return self._resolve_path(self.config['global']['dataset_path'])

        # Check if tab has dedicated dataset
        tabs_config = self.config.get('tabs', {})

        if tab_name in tabs_config:
            tab_config = tabs_config[tab_name]

            # Check if tab-specific dataset is enabled
            if tab_config.get('enabled', True):
                tab_path = tab_config.get('dataset_path')

                if tab_path:
                    resolved_path = self._resolve_path(tab_path)

                    # Check if file exists
                    if Path(resolved_path).exists():
                        logger.info(f"Using tab-specific dataset for '{tab_name}': {resolved_path}")
                        return resolved_path
                    else:
                        logger.warning(f"Tab-specific dataset not found: {resolved_path}")

        # Fallback to global dataset
        fallback_enabled = self.config['global'].get('fallback_enabled', True)

        if fallback_enabled:
            global_path = self._resolve_path(self.config['global']['dataset_path'])
            logger.info(f"Using global dataset for '{tab_name}': {global_path}")
            return global_path
        else:
            raise DatasetValidationError(
                f"No dataset configured for tab '{tab_name}' and fallback disabled"
            )

    def _resolve_path(self, path: str) -> str:
        """
        Resolve relative paths to absolute paths.

        Relative paths are resolved from project root.
        """
        path_obj = Path(path)

        if path_obj.is_absolute():
            return str(path_obj)

        # Try to find project root
        project_root = self._find_project_root()

        if project_root:
            return str(project_root / path_obj)

        return str(path_obj)

    def _find_project_root(self) -> Optional[Path]:
        """Find project root directory by looking for marker files."""
        current = Path.cwd()

        # Look for typical project root markers
        markers = ['requirements.txt', 'setup.py', 'pyproject.toml', '.git']

        for parent in [current] + list(current.parents):
            if any((parent / marker).exists() for marker in markers):
                return parent

        return None

    def get_tab_config(self, tab_name: str) -> Dict[str, Any]:
        """Get full configuration for a specific tab."""
        tabs_config = self.config.get('tabs', {})
        return tabs_config.get(tab_name, {})

    def is_cache_enabled(self, tab_name: Optional[str] = None) -> bool:
        """Check if dataset caching is enabled."""
        if tab_name:
            tab_config = self.get_tab_config(tab_name)
            if 'cache_enabled' in tab_config:
                return tab_config['cache_enabled']

        return self.config['global'].get('cache_enabled', False)


class DatasetLoader:
    """
    Loads and validates datasets according to data dictionary requirements.

    Features:
    - Automatic validation of required columns
    - Data type checking
    - Value range validation
    - Performance metrics logging
    - Caching support

    Usage:
        loader = DatasetLoader()
        df = loader.load_dataset("winning_edge")
        # Returns validated DataFrame ready for analysis
    """

    # Data dictionary validation rules
    REQUIRED_COLUMNS = [
        'vehicle_number',
        'lap',
        'timestamp',
        'telemetry_name',
        'telemetry_value'
    ]

    CRITICAL_SENSORS = ['speed', 'pbrake_f']  # Timestamp is a column, not a sensor

    OPTIONAL_COLUMNS = ['track', 'race', 'session']

    # Value ranges for validation
    VALUE_RANGES = {
        'speed': (0, 350),  # km/h
        'pbrake_f': (0, 200),  # bar
        'pbrake_r': (0, 200),
        'aps': (0, 100),  # %
        'Steering_Angle': (-900, 900),  # degrees
        'gear': (-1, 8),  # -1 = neutral, 0 = reverse, 1-8 = forward gears
        'nmot': (0, 20000),  # RPM
        'accx_can': (-5, 5),  # g
        'accy_can': (-5, 5),
    }

    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        Initialize dataset loader.

        Args:
            config: DatasetConfig instance. If None, creates default.
        """
        self.config = config or DatasetConfig()
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_dataset(
        self,
        tab_name: Optional[str] = None,
        validate: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load and optionally validate dataset for a dashboard tab.

        Args:
            tab_name: Dashboard tab name (e.g., "winning_edge").
                     If None, loads global dataset.
            validate: Whether to validate dataset against data dictionary.
            use_cache: Whether to use cached dataset if available.

        Returns:
            Validated pandas DataFrame ready for analysis.

        Raises:
            DatasetValidationError: If validation fails.
            FileNotFoundError: If dataset file not found.

        Example:
            >>> loader = DatasetLoader()
            >>> df = loader.load_dataset("winning_edge")
            >>> print(df.shape)
            (71000, 8)
        """
        # Check cache first
        cache_key = tab_name or 'global'

        if use_cache and self.config.is_cache_enabled(tab_name):
            if cache_key in self._cache:
                logger.info(f"Using cached dataset for '{cache_key}'")
                return self._cache[cache_key].copy()

        # Get dataset path
        dataset_path = self.config.get_dataset_path(tab_name)

        # Load dataset
        logger.info(f"Loading dataset from: {dataset_path}")
        df = self._load_file(dataset_path)

        # Validate if requested
        if validate:
            self._validate_dataset(df, tab_name)

        # Cache if enabled
        if self.config.is_cache_enabled(tab_name):
            self._cache[cache_key] = df.copy()

        logger.info(f"Successfully loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _load_file(self, path: str) -> pd.DataFrame:
        """Load CSV or Parquet file."""
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        try:
            if path_obj.suffix == '.parquet':
                return pd.read_parquet(path)
            elif path_obj.suffix == '.csv':
                return pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path_obj.suffix}")

        except Exception as e:
            raise DatasetValidationError(f"Failed to load dataset: {e}")

    def _validate_dataset(self, df: pd.DataFrame, tab_name: Optional[str] = None) -> None:
        """
        Validate dataset against data dictionary requirements.

        Checks:
        1. Required columns present
        2. Critical sensors available
        3. Data types correct
        4. Value ranges valid
        5. Minimum row count
        """
        errors = []

        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check minimum row count
        if len(df) < 100:
            errors.append(f"Dataset too small: {len(df)} rows (minimum 100)")

        # Check critical sensors present in telemetry_name
        if 'telemetry_name' in df.columns:
            available_sensors = df['telemetry_name'].unique()
            missing_sensors = set(self.CRITICAL_SENSORS) - set(available_sensors)

            if missing_sensors:
                errors.append(f"Missing critical sensors: {missing_sensors}")

        # Check data types
        if 'vehicle_number' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['vehicle_number']):
                errors.append("vehicle_number must be numeric")

        if 'lap' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['lap']):
                errors.append("lap must be numeric")

        if 'timestamp' in df.columns:
            # Accept timestamp as numeric (Unix) or string (ISO datetime)
            is_numeric = pd.api.types.is_numeric_dtype(df['timestamp'])
            is_string = pd.api.types.is_string_dtype(df['timestamp']) or pd.api.types.is_object_dtype(df['timestamp'])
            if not (is_numeric or is_string):
                errors.append("timestamp must be numeric or string (datetime)")

        if 'telemetry_value' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['telemetry_value']):
                errors.append("telemetry_value must be numeric")

        # Check value ranges (sample-based for performance)
        if 'telemetry_name' in df.columns and 'telemetry_value' in df.columns:
            for sensor, (min_val, max_val) in self.VALUE_RANGES.items():
                sensor_data = df[df['telemetry_name'] == sensor]

                if len(sensor_data) > 0:
                    out_of_range = (
                        (sensor_data['telemetry_value'] < min_val) |
                        (sensor_data['telemetry_value'] > max_val)
                    )

                    if out_of_range.any():
                        count = out_of_range.sum()
                        errors.append(
                            f"Sensor '{sensor}' has {count} values out of range "
                            f"[{min_val}, {max_val}]"
                        )

        # Raise if validation failed
        if errors:
            error_msg = f"Dataset validation failed for tab '{tab_name}':\n" + "\n".join(errors)
            raise DatasetValidationError(error_msg)

        logger.info(f"Dataset validation passed for tab '{tab_name}'")

    def clear_cache(self, tab_name: Optional[str] = None) -> None:
        """
        Clear cached datasets.

        Args:
            tab_name: Clear specific tab cache. If None, clears all.
        """
        if tab_name:
            cache_key = tab_name
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info(f"Cleared cache for '{tab_name}'")
        else:
            self._cache.clear()
            logger.info("Cleared all cached datasets")

    def get_dataset_info(self, tab_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata about a dataset without loading it.

        Returns:
            Dictionary with file size, path, modification time, etc.
        """
        dataset_path = self.config.get_dataset_path(tab_name)
        path_obj = Path(dataset_path)

        if not path_obj.exists():
            return {'exists': False, 'path': dataset_path}

        stats = path_obj.stat()

        return {
            'exists': True,
            'path': str(path_obj),
            'size_bytes': stats.st_size,
            'size_mb': round(stats.st_size / (1024 * 1024), 2),
            'modified': stats.st_mtime,
            'format': path_obj.suffix,
        }


def get_default_config() -> DatasetConfig:
    """Create default DatasetConfig instance."""
    return DatasetConfig()


def create_default_config_file(output_path: Optional[str] = None) -> Path:
    """
    Create a default dataset_config.yaml file.

    Args:
        output_path: Where to save config file. If None, uses default location.

    Returns:
        Path to created config file.
    """
    if output_path:
        config_path = Path(output_path)
    else:
        config_path = Path("src/config/dataset_config.yaml")

    # Create default configuration
    default_config = {
        'global': {
            'dataset_path': 'master_racing_data.csv',
            'fallback_enabled': True,
            'cache_enabled': False,
        },
        'tabs': {
            'winning_edge': {
                'dataset_path': 'data/winning_edge_dataset.csv',
                'description': 'Dedicated dataset for Winning Edge tab - corner analysis and performance gaps',
                'enabled': True,
                'cache_enabled': False,
            },
            'post_race': {
                'dataset_path': 'data/post_race_dataset.csv',
                'description': 'Dedicated dataset for Post-Race Analysis tab',
                'enabled': False,  # Not yet implemented
                'cache_enabled': False,
            },
            'weather': {
                'dataset_path': 'data/weather_dataset.csv',
                'description': 'Dedicated dataset for Weather Analysis tab',
                'enabled': False,
                'cache_enabled': False,
            },
        }
    }

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created default config file: {config_path}")
    return config_path
