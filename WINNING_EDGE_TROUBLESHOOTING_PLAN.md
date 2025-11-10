# Winning Edge Dashboard Troubleshooting Plan
## Root Cause Analysis & Fix Implementation Guide

**Date**: 2025-11-10
**Issue**: Plots not showing in Winning Edge dashboard
**Status**: Root cause identified ✓
**Severity**: High (Dashboard completely non-functional)

---

## Executive Summary

The Winning Edge dashboard is unable to display any plots despite successfully loading data (582,035 samples, 1 vehicle, 10 laps from winning_edge_dataset.csv).

**Root Cause Identified**: JSON serialization error in data processing callbacks
- **Error**: `Object of type int64 is not JSON serializable`
- **Location**: `src/dashboard/winning_edge_callback_helpers.py`
- **Impact**: All visualizations fail silently (heatmap, spider chart, correlation, etc.)

---

## Diagnostic Results Summary

### Test Results (from diagnose_winning_edge_issue.py)

| Test | Status | Details |
|------|--------|---------|
| **FILES** | ✓ PASS | All dataset files exist and accessible |
| **LOADER** | ✓ PASS | DatasetLoader successfully loads 1M rows |
| **STRUCTURE** | ✓ PASS | All required columns and sensors present |
| **CORNERS** | ✓ PASS | Corner detection logic finds potential corners |
| **HELPERS** | ✗ FAIL | **JSON serialization error in callback helpers** |
| **FLOW** | ✗ FAIL | Dashboard flow broken due to helper failure |
| **CONFIG** | ✓ PASS | YAML configuration correct |

### Key Findings

1. **Data Loading**: Working perfectly
   - Dataset: `/data/barber_winning_edge_dataset.csv` (154.3 MB)
   - Successfully loaded: 1,000,000 rows
   - Vehicles found: [0, 78]
   - Laps: Vehicle 0 has 28 laps, Vehicle 78 has 19 laps
   - All required sensors present: speed, pbrake_f, pbrake_r, aps, Steering_Angle

2. **Corner Detection**: Working
   - 12,216 speed samples for vehicle 0
   - Speed range: 64.7 - 186.9 km/h
   - 4,046 potential corner points detected

3. **Critical Failure Point**: Callback helpers
   - Error location: `process_corner_data_for_heatmap()` function
   - Error type: JSON serialization of numpy int64/float64 types
   - Result: Empty corner data dict returned (`{}`)
   - Consequence: All visualizations show "No data" or empty plots

---

## Root Cause Deep Dive

### Technical Explanation

The data processing pipeline works as follows:

```
1. DataFrame → WinningEdgeDataProcessor → CornerMetrics (dataclasses)
2. CornerMetrics → calculate_real_time_gaps() → PerformanceGap (dataclasses)
3. PerformanceGap → process_corner_data_for_heatmap() → Dict[str, Dict]
4. Dict[str, Dict] → JSON serialization (for dcc.Store)
5. JSON → Dash callback → Visualization
```

**The bug occurs at step 4**: When the dataclass fields (which contain numpy types from DataFrame operations) are extracted and placed into a Python dict, they retain their numpy types (np.int64, np.float64). When Dash attempts to serialize this dict to JSON for the dcc.Store component, it fails because JSON encoder doesn't know how to handle numpy types.

### Example of the Issue

```python
# In process_corner_data_for_heatmap():
corner_data[corner_name] = {
    'time_loss': gap.total_time_gap,  # This might be np.float64
    'pct_of_total': (gap.total_time_gap / total_loss * 100)  # np.float64
}

# When Dash tries to serialize:
json.dumps(corner_data)  # ERROR: Object of type int64 is not JSON serializable
```

### Why Silent Failure?

The error is caught by the try/except block in the helper functions:

```python
except Exception as e:
    logger.error(f"Error processing corner data for heatmap: {e}")
    return {}  # Empty dict returned instead of raising error
```

This causes:
1. No error shown to user
2. Callbacks receive empty data
3. Visualizations display "No data available" or empty plots
4. User sees dashboard but no plots

---

## Solution Strategy

### Fix Approach: Convert Numpy Types to Native Python Types

**Strategy**: Add type conversion layer in callback helpers before returning data.

**Implementation Points**:
1. `process_corner_data_for_heatmap()` - Convert before returning dict
2. `process_speed_gap_data_for_spider()` - Convert list of floats
3. `process_brake_exit_correlation_data()` - Convert tuples/lists
4. All other helper functions returning data for JSON serialization

**Methods**:
- Use `.item()` on numpy scalars: `np_value.item()` → `python_value`
- Use `float()` / `int()` constructors: `float(np_value)` → `python_float`
- Use recursive conversion for nested structures

---

## Fix Implementation Plan

### Phase 1: Create Type Conversion Utility (PRIORITY 1)

**File**: `src/dashboard/winning_edge_callback_helpers.py`

Add helper function at the top of the file:

```python
def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Any object that might contain numpy types

    Returns:
        Object with all numpy types converted to native Python types
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
```

### Phase 2: Fix Each Helper Function (PRIORITY 1)

**1. process_corner_data_for_heatmap()**

Current (line 87-93):
```python
corner_data[corner_name] = {
    'time_loss': gap.total_time_gap,
    'pct_of_total': (gap.total_time_gap / total_loss * 100) if total_loss > 0 else 0
}
```

Fixed:
```python
corner_data[corner_name] = {
    'time_loss': float(gap.total_time_gap),
    'pct_of_total': float((gap.total_time_gap / total_loss * 100) if total_loss > 0 else 0)
}
```

**OR** use the utility at the end:
```python
return convert_numpy_types(corner_data)
```

**2. process_speed_gap_data_for_spider()**

Check and apply same fix pattern.

**3. process_brake_exit_correlation_data()**

Check and apply same fix pattern.

**4. All other helper functions**

Audit and fix each function that returns data for JSON serialization.

### Phase 3: Test Fixes (PRIORITY 2)

Run diagnostic script again:
```bash
python3 diagnose_winning_edge_issue.py
```

Expected output: All tests PASS

### Phase 4: Restart Dashboard (PRIORITY 2)

```bash
# Stop current dashboard process
pkill -f "python.*src/dashboard/app.py"

# Restart dashboard
python3 src/dashboard/app.py
```

---

## Verification Checklist

After implementing fixes:

- [ ] Diagnostic script shows all tests passing
- [ ] Dashboard starts without errors
- [ ] Vehicle dropdown populates with options
- [ ] Data info shows correct statistics
- [ ] **Tab 1: Race Winner's Dashboard**
  - [ ] Time Loss Heatmap displays with colors
  - [ ] Speed Gap Spider Chart displays with traces
- [ ] **Tab 2: Correlation Dashboard**
  - [ ] Brake-Exit Correlation chart displays
  - [ ] Speed Cascade waterfall displays
  - [ ] Consistency Matrix displays
- [ ] **Tab 3-7: All other tabs**
  - [ ] Action cards display
  - [ ] Race simulation charts display
  - [ ] Timeline displays
  - [ ] Session guides display
  - [ ] Summary dashboard displays

---

## Alternative Solutions (If Fix Doesn't Work)

### Option A: Use Custom JSON Encoder

Create custom JSON encoder that handles numpy types:

```python
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
```

Use when converting to JSON:
```python
data_json = json.dumps(data, cls=NumpyEncoder)
```

### Option B: Fix at DataFrame Level

Convert DataFrame columns to native types before processing:

```python
def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all numeric columns to native Python types."""
    for col in df.columns:
        if df[col].dtype in ['int64', 'int32']:
            df[col] = df[col].astype(int)
        elif df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].astype(float)
    return df
```

### Option C: Fix at Dataclass Level

Add `to_dict()` method to dataclasses that handles conversion:

```python
@dataclass
class PerformanceGap:
    # ... fields ...

    def to_dict(self):
        """Convert to dict with native Python types."""
        return {
            'corner_id': int(self.corner_id),
            'corner_name': str(self.corner_name),
            'total_time_gap': float(self.total_time_gap),
            # ... etc
        }
```

---

## Prevention Measures

### Code Review Checklist

When adding new callback helpers:
- [ ] Check all returned values for numpy types
- [ ] Add explicit type conversion for dict values
- [ ] Test with `json.dumps()` before returning
- [ ] Add unit test that verifies JSON serializability

### Testing Protocol

Add automated test:

```python
def test_callback_helpers_json_serializable():
    """Ensure all callback helper outputs are JSON serializable."""
    from src.dashboard.winning_edge_callback_helpers import (
        process_corner_data_for_heatmap,
        process_speed_gap_data_for_spider,
        # ... all helpers
    )
    import json

    # Load test data
    df = load_test_telemetry()
    vehicle = df['vehicle_number'].iloc[0]

    # Test each helper
    corner_data = process_corner_data_for_heatmap(df, vehicle)
    json.dumps(corner_data)  # Should not raise error

    speed_gaps = process_speed_gap_data_for_spider(df, vehicle)
    json.dumps(speed_gaps)  # Should not raise error

    # etc...
```

---

## Timeline Estimate

| Phase | Effort | Duration |
|-------|--------|----------|
| Add type conversion utility | Low | 5 minutes |
| Fix all helper functions | Medium | 15 minutes |
| Test fixes | Low | 5 minutes |
| Restart dashboard | Low | 2 minutes |
| Verify all tabs | Medium | 10 minutes |
| **Total** | | **~40 minutes** |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Fix doesn't resolve issue | Low | High | Use alternative solutions |
| Fix breaks other functionality | Low | Medium | Test all tabs thoroughly |
| New numpy type appears later | Medium | Low | Add comprehensive type conversion |
| Performance impact | Low | Low | Conversion is fast, negligible impact |

---

## Success Criteria

**Definition of Done**:
1. All diagnostic tests pass (7/7)
2. All 7 tabs display visualizations correctly
3. No console errors related to JSON serialization
4. Vehicle selection triggers plot updates
5. Data flows through entire callback chain successfully

**Performance Targets**:
- Dashboard loads in < 5 seconds
- Plot renders in < 2 seconds after vehicle selection
- No memory leaks or performance degradation

---

## References

### Files Modified
- `src/dashboard/winning_edge_callback_helpers.py` - Add type conversion

### Files to Review
- `src/dashboard/winning_edge_widget.py` - Callbacks using the helpers
- `src/dashboard/winning_edge_data_processor.py` - Dataclass definitions

### Related Documentation
- Dash JSON serialization: https://dash.plotly.com/sharing-data-between-callbacks
- Numpy type conversion: https://numpy.org/doc/stable/user/basics.types.html
- Python JSON encoder: https://docs.python.org/3/library/json.html

---

## Contact & Support

**Issue Severity**: P0 - Critical
**Component**: Dashboard / Winning Edge Widget
**Owner**: Racing Analytics Team
**Status**: Fix Ready for Implementation

---

## Appendix A: Diagnostic Command Reference

```bash
# Run full diagnostic
python3 diagnose_winning_edge_issue.py

# Check specific dataset
python3 -c "import pandas as pd; df = pd.read_csv('data/barber_winning_edge_dataset.csv'); print(df.info())"

# Test JSON serialization
python3 -c "import json; import numpy as np; d = {'val': np.int64(5)}; json.dumps(d)"  # Will fail

python3 -c "import json; import numpy as np; d = {'val': int(np.int64(5))}; json.dumps(d)"  # Will succeed

# Check dashboard logs
tail -f ~/racing_analytics/logs/app.log  # Production
tail -f logs/dashboard.log  # Development

# Restart dashboard
pkill -f "python.*app.py" && python3 src/dashboard/app.py
```

---

## Appendix B: Error Trace Analysis

```
ERROR: Error processing corner data for heatmap: Object of type int64 is not JSON serializable
  File: src/dashboard/winning_edge_callback_helpers.py
  Function: process_corner_data_for_heatmap()
  Line: 91-93 (dict construction)

  Stack trace breakdown:
  1. DataFrame operations produce numpy types
  2. WinningEdgeDataProcessor extracts metrics → dataclasses with numpy types
  3. calculate_real_time_gaps() → PerformanceGap with numpy types
  4. Helper function creates dict → values still numpy types
  5. Dash callback returns dict → dcc.Store tries to serialize
  6. json.dumps() called internally → FAILS on numpy types
  7. Exception caught → returns empty dict
  8. Empty dict propagates → plots show "No data"
```

---

**End of Troubleshooting Plan**
