"""
Telemetry Analysis Service - Sprint 2 Task 2
============================================

Wraps the CubeAnalysisEngine to provide:
- Clean interface for analyzing telemetry DataFrames
- Caching to avoid re-analysis of same data
- Error handling and logging
- Pattern format conversion for dashboard widgets
"""

import pandas as pd
import logging
import hashlib
import tempfile
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import cube analysis engine
from demo_cube_analysis import CubeAnalysisEngine, DrivingPattern

logger = logging.getLogger(__name__)


class TelemetryAnalyzer:
    """
    Service for analyzing racing telemetry data using cube analysis
    """

    def __init__(self):
        self._cache = {}  # Cache analysis results by data hash
        self._temp_dir = Path(tempfile.gettempdir()) / 'racing_analytics'
        self._temp_dir.mkdir(exist_ok=True)

    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """
        Generate hash of DataFrame for caching

        Args:
            df: Telemetry DataFrame

        Returns:
            MD5 hash string
        """
        # Create hash from DataFrame shape and first/last rows
        hash_input = f"{df.shape}_{df.head(5).to_json()}_{df.tail(5).to_json()}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def analyze_telemetry(
        self,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Analyze telemetry data and detect driving patterns

        Args:
            df: Telemetry DataFrame in long format (telemetry_name, telemetry_value columns)
            use_cache: Whether to use cached results if available

        Returns:
            List of pattern dictionaries suitable for pattern_analysis_widget

        Raises:
            ValueError: If DataFrame is invalid or missing required columns
            Exception: If analysis fails
        """
        try:
            # Validate input
            self._validate_telemetry_data(df)

            # Check cache
            data_hash = self._get_data_hash(df)
            if use_cache and data_hash in self._cache:
                logger.info(f"Using cached analysis for data hash: {data_hash[:8]}")
                return self._cache[data_hash]['patterns']

            logger.info(f"Running cube analysis on {len(df):,} rows...")

            # Save DataFrame to temporary CSV for CubeAnalysisEngine
            temp_csv_path = self._temp_dir / f'telemetry_{data_hash[:8]}.csv'
            df.to_csv(temp_csv_path, index=False)

            try:
                # Initialize and run cube analysis engine
                engine = CubeAnalysisEngine(str(temp_csv_path))
                engine.load_data()
                engine.detect_corners()
                patterns = engine.detect_driving_patterns()

                # Convert patterns to widget format
                patterns_data = self._convert_patterns_to_dict(patterns)

                # Cache results (including engine for corner analysis)
                self._cache[data_hash] = {
                    'patterns': patterns_data,
                    'engine': engine  # Store engine for later corner analysis
                }

                logger.info(f"Analysis complete: {len(patterns_data)} patterns detected")

                return patterns_data

            finally:
                # Clean up temporary file
                if temp_csv_path.exists():
                    temp_csv_path.unlink()

        except Exception as e:
            logger.error(f"Telemetry analysis failed: {str(e)}", exc_info=True)
            # Return empty list on error (dashboard will handle gracefully)
            return []

    def analyze_corners(
        self,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Analyze corner performance from telemetry data

        Args:
            df: Telemetry DataFrame in long format
            use_cache: Whether to use cached results if available

        Returns:
            List of corner analysis dictionaries suitable for corner_analysis_widget

        Raises:
            ValueError: If DataFrame is invalid or missing required columns
            Exception: If analysis fails
        """
        try:
            # Validate input
            self._validate_telemetry_data(df)

            # Check cache
            data_hash = self._get_data_hash(df)
            if use_cache and data_hash in self._cache:
                logger.info(f"Using cached corner analysis for data hash: {data_hash[:8]}")
                engine = self._cache[data_hash].get('engine')
                if engine and hasattr(engine, 'corners'):
                    # Convert corner analyses from engine
                    return self._convert_corners_to_dict(engine)

            logger.info(f"Running corner analysis on {len(df):,} rows...")

            # Save DataFrame to temporary CSV for CubeAnalysisEngine
            temp_csv_path = self._temp_dir / f'telemetry_{data_hash[:8]}.csv'
            df.to_csv(temp_csv_path, index=False)

            try:
                # Initialize and run cube analysis engine
                engine = CubeAnalysisEngine(str(temp_csv_path))
                engine.load_data()
                corners = engine.detect_corners()

                if not corners:
                    logger.warning("No corners detected in telemetry data")
                    return []

                # Analyze performance for each corner
                corner_analyses = []
                for corner in corners[:10]:  # Limit to first 10 corners for performance
                    analysis = engine.analyze_corner_performance(corner)
                    if analysis:
                        corner_analyses.append(analysis)

                logger.info(f"Corner analysis complete: {len(corner_analyses)} corners analyzed")

                # Convert to widget format
                corners_data = self._convert_corner_analyses_to_dict(corner_analyses)

                # Update cache
                if data_hash in self._cache:
                    self._cache[data_hash]['corners'] = corners_data
                    self._cache[data_hash]['engine'] = engine
                else:
                    self._cache[data_hash] = {
                        'patterns': [],
                        'corners': corners_data,
                        'engine': engine
                    }

                return corners_data

            finally:
                # Clean up temporary file
                if temp_csv_path.exists():
                    temp_csv_path.unlink()

        except Exception as e:
            logger.error(f"Corner analysis failed: {str(e)}", exc_info=True)
            # Return empty list on error (dashboard will handle gracefully)
            return []

    def _convert_corners_to_dict(self, engine) -> List[Dict]:
        """
        Convert corners from cached engine to dictionary format

        Args:
            engine: CubeAnalysisEngine with detected corners

        Returns:
            List of corner analysis dictionaries
        """
        if not engine.corners:
            return []

        corner_analyses = []
        for corner in engine.corners[:10]:
            analysis = engine.analyze_corner_performance(corner)
            if analysis:
                corner_analyses.append(analysis)

        return self._convert_corner_analyses_to_dict(corner_analyses)

    def _convert_corner_analyses_to_dict(self, corner_analyses: List) -> List[Dict]:
        """
        Convert CornerAnalysis objects to dictionaries for widget

        Args:
            corner_analyses: List of CornerAnalysis objects

        Returns:
            List of corner analysis dictionaries
        """
        corners_data = []

        for corner in corner_analyses:
            # Convert dataclass to dict
            corner_dict = asdict(corner)

            # Ensure all expected fields are present for corner_analysis_widget
            corner_dict.setdefault('corner_number', 0)
            corner_dict.setdefault('corner_name', 'Unknown Corner')
            corner_dict.setdefault('entry_speed_avg', 0.0)
            corner_dict.setdefault('entry_speed_best', 0.0)
            corner_dict.setdefault('apex_speed_avg', 0.0)
            corner_dict.setdefault('apex_speed_best', 0.0)
            corner_dict.setdefault('brake_pressure_avg', 0.0)
            corner_dict.setdefault('brake_pressure_max', 0.0)
            corner_dict.setdefault('time_delta', 0.0)
            corner_dict.setdefault('opportunities', [])

            corners_data.append(corner_dict)

        return corners_data

    def _validate_telemetry_data(self, df: pd.DataFrame) -> None:
        """
        Validate telemetry DataFrame has required columns

        Args:
            df: Telemetry DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty")

        required_columns = ['telemetry_name', 'telemetry_value', 'timestamp', 'lap']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for minimum data
        if len(df) < 100:
            raise ValueError(f"Insufficient data: only {len(df)} rows (need at least 100)")

        # Check for required sensors
        sensors = df['telemetry_name'].unique()
        if len(sensors) < 3:
            raise ValueError(f"Insufficient sensors: only {len(sensors)} (need at least 3)")

    def _convert_patterns_to_dict(self, patterns: List[DrivingPattern]) -> List[Dict]:
        """
        Convert DrivingPattern dataclasses to dictionaries for widget

        Args:
            patterns: List of DrivingPattern objects

        Returns:
            List of pattern dictionaries
        """
        patterns_data = []

        for pattern in patterns:
            # Convert dataclass to dict
            pattern_dict = asdict(pattern)

            # Ensure all expected fields are present
            pattern_dict.setdefault('pattern_name', 'Unknown Pattern')
            pattern_dict.setdefault('severity', 'Medium')
            pattern_dict.setdefault('impact_seconds', 0.0)
            pattern_dict.setdefault('what_metrics', [])
            pattern_dict.setdefault('where_corners', [])
            pattern_dict.setdefault('when_laps', [])
            pattern_dict.setdefault('coaching', 'No coaching available')

            patterns_data.append(pattern_dict)

        return patterns_data

    def clear_cache(self) -> None:
        """Clear the analysis cache"""
        self._cache.clear()
        logger.info("Analysis cache cleared")

    def get_cache_size(self) -> int:
        """Get number of cached analyses"""
        return len(self._cache)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Create global instance for use across dashboard
_analyzer_instance = None

def get_telemetry_analyzer() -> TelemetryAnalyzer:
    """
    Get singleton TelemetryAnalyzer instance

    Returns:
        Global TelemetryAnalyzer instance
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TelemetryAnalyzer()
        logger.info("Created TelemetryAnalyzer singleton instance")
    return _analyzer_instance


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Test the telemetry analyzer service"""
    import sys

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("🧪 Testing TelemetryAnalyzer Service")
    print("=" * 80)

    # Test with master_racing_data.csv
    csv_path = 'master_racing_data.csv'

    if not Path(csv_path).exists():
        print(f"❌ Test file not found: {csv_path}")
        sys.exit(1)

    print(f"\n📂 Loading test data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df):,} rows")

    # Test analysis
    analyzer = get_telemetry_analyzer()

    print("\n🔬 Running first analysis (no cache)...")
    patterns = analyzer.analyze_telemetry(df)
    print(f"✅ Detected {len(patterns)} patterns")

    for i, pattern in enumerate(patterns, 1):
        print(f"\n  Pattern {i}: {pattern['pattern_name']}")
        print(f"    Severity: {pattern['severity']}")
        print(f"    Impact: {pattern['impact_seconds']:.2f}s/lap")

    print(f"\n💾 Cache size: {analyzer.get_cache_size()}")

    print("\n🔬 Running second analysis (should use cache)...")
    patterns2 = analyzer.analyze_telemetry(df)
    print(f"✅ Retrieved {len(patterns2)} patterns from cache")

    print("\n" + "=" * 80)
    print("✅ TelemetryAnalyzer Service Test Complete!")
    print("=" * 80)
