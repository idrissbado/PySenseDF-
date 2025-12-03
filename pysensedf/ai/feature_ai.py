"""
Auto Feature Engineering
=========================

One-line feature engineering: df.autofeatures(target="label")
"""

from typing import Optional, List, Dict, Any
import datetime


def auto_generate_features(df, target: Optional[str] = None):
    """
    Automatically generate features
    
    Creates:
    - Date/time features (year, month, day, hour, dayofweek)
    - Aggregation features (per-group stats)
    - Ratio features
    - Interaction features
    - Lag features
    """
    from ..core.dataframe import DataFrame
    
    new_data = df._data.copy()
    
    # 1. Date features
    for col in df.columns():
        if _is_date_column(df._data[col]):
            date_features = _create_date_features(df._data[col], col)
            new_data.update(date_features)
    
    # 2. Numeric aggregations
    numeric_cols = _get_numeric_columns(df)
    if numeric_cols:
        # Create ratio features
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                ratio_col = f"{col1}_div_{col2}"
                new_data[ratio_col] = _safe_divide(df._data[col1], df._data[col2])
    
    # 3. Interaction features (for small datasets)
    if len(numeric_cols) <= 5 and df.shape()[0] < 10000:
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                interact_col = f"{col1}_times_{col2}"
                new_data[interact_col] = _multiply(df._data[col1], df._data[col2])
    
    return DataFrame(new_data)


def _is_date_column(values: List[Any]) -> bool:
    """Check if column contains dates"""
    non_empty = [v for v in values[:50] if v and v != '']
    if not non_empty:
        return False
    
    date_count = 0
    for v in non_empty:
        try:
            if isinstance(v, datetime.datetime):
                date_count += 1
            elif isinstance(v, str) and ('/' in v or '-' in v) and len(v) >= 8:
                # Try parsing
                datetime.datetime.fromisoformat(v.replace('/', '-')[:10])
                date_count += 1
        except:
            pass
    
    return date_count / len(non_empty) > 0.5


def _create_date_features(values: List[Any], col_name: str) -> Dict[str, List]:
    """Create date-based features"""
    features = {
        f"{col_name}_year": [],
        f"{col_name}_month": [],
        f"{col_name}_day": [],
        f"{col_name}_dayofweek": [],
        f"{col_name}_quarter": [],
    }
    
    for v in values:
        try:
            if isinstance(v, datetime.datetime):
                dt = v
            elif isinstance(v, str):
                dt = datetime.datetime.fromisoformat(v.replace('/', '-')[:10])
            else:
                dt = None
            
            if dt:
                features[f"{col_name}_year"].append(dt.year)
                features[f"{col_name}_month"].append(dt.month)
                features[f"{col_name}_day"].append(dt.day)
                features[f"{col_name}_dayofweek"].append(dt.weekday())
                features[f"{col_name}_quarter"].append((dt.month - 1) // 3 + 1)
            else:
                for k in features.keys():
                    features[k].append(None)
        except:
            for k in features.keys():
                features[k].append(None)
    
    return features


def _get_numeric_columns(df) -> List[str]:
    """Get list of numeric columns"""
    numeric = []
    for col in df.columns():
        values = df._data[col]
        try:
            # Try to convert first 10 non-empty values
            sample = [v for v in values[:50] if v and v != ''][:10]
            if sample:
                [float(v) for v in sample]
                numeric.append(col)
        except:
            pass
    
    return numeric


def _safe_divide(col1: List[Any], col2: List[Any]) -> List[Any]:
    """Safely divide two columns"""
    result = []
    for v1, v2 in zip(col1, col2):
        try:
            num1 = float(v1)
            num2 = float(v2)
            if num2 != 0:
                result.append(num1 / num2)
            else:
                result.append(None)
        except:
            result.append(None)
    
    return result


def _multiply(col1: List[Any], col2: List[Any]) -> List[Any]:
    """Multiply two columns"""
    result = []
    for v1, v2 in zip(col1, col2):
        try:
            result.append(float(v1) * float(v2))
        except:
            result.append(None)
    
    return result


def create_lag_features(values: List[Any], lags: List[int] = [1, 7, 30]) -> Dict[str, List]:
    """Create lag features for time series"""
    features = {}
    
    for lag in lags:
        lagged = [None] * lag + values[:-lag] if lag < len(values) else [None] * len(values)
        features[f"lag_{lag}"] = lagged
    
    return features


def create_rolling_features(values: List[Any], windows: List[int] = [7, 30]) -> Dict[str, List]:
    """Create rolling window features"""
    features = {}
    
    for window in windows:
        rolling_mean = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_values = values[start:i+1]
            try:
                numeric = [float(v) for v in window_values if v is not None]
                mean = sum(numeric) / len(numeric) if numeric else None
                rolling_mean.append(mean)
            except:
                rolling_mean.append(None)
        
        features[f"rolling_mean_{window}"] = rolling_mean
    
    return features
