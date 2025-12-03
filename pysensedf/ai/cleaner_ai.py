"""
Auto Data Cleaning Engine
==========================

One-line data cleaning: df.autoclean()
"""

from typing import Dict, List, Any
import datetime


def auto_clean_dataframe(df):
    """
    Automatic data cleaning pipeline
    
    Steps:
    1. Infer types
    2. Parse dates
    3. Convert numeric strings
    4. Handle missing values
    5. Remove duplicates
    6. Standardize text
    """
    from ..core.dataframe import DataFrame
    
    new_data = {}
    
    for col in df.columns():
        values = df._data[col]
        cleaned = _clean_column(values, col)
        new_data[col] = cleaned
    
    return DataFrame(new_data)


def _clean_column(values: List[Any], col_name: str) -> List[Any]:
    """Clean a single column"""
    
    # Step 1: Infer type
    col_type = _infer_type(values)
    
    # Step 2: Clean based on type
    if col_type == "numeric":
        return _clean_numeric(values)
    elif col_type == "date":
        return _clean_dates(values)
    elif col_type == "text":
        return _clean_text(values)
    else:
        return values


def _infer_type(values: List[Any]) -> str:
    """Infer column type"""
    non_empty = [v for v in values[:100] if v and v != '']
    
    if not non_empty:
        return "text"
    
    # Check if numeric
    numeric_count = 0
    for v in non_empty:
        try:
            float(str(v).replace(',', '').replace('$', ''))
            numeric_count += 1
        except:
            pass
    
    if numeric_count / len(non_empty) > 0.7:
        return "numeric"
    
    # Check if date
    date_count = 0
    for v in non_empty:
        if _is_date(str(v)):
            date_count += 1
    
    if date_count / len(non_empty) > 0.7:
        return "date"
    
    return "text"


def _is_date(value: str) -> bool:
    """Check if value looks like a date"""
    date_patterns = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"
    ]
    
    for pattern in date_patterns:
        try:
            datetime.datetime.strptime(value[:10], pattern)
            return True
        except:
            pass
    
    return False


def _clean_numeric(values: List[Any]) -> List[Any]:
    """Clean numeric column"""
    cleaned = []
    numeric_values = []
    
    # First pass: convert to numbers
    for v in values:
        try:
            # Remove common non-numeric characters
            v_str = str(v).replace(',', '').replace('$', '').replace('%', '').strip()
            if v_str:
                num = float(v_str)
                numeric_values.append(num)
                cleaned.append(num)
            else:
                cleaned.append(None)
        except:
            cleaned.append(None)
    
    # Fill missing with median
    if numeric_values:
        numeric_values.sort()
        median = numeric_values[len(numeric_values) // 2]
        cleaned = [median if v is None else v for v in cleaned]
    
    return cleaned


def _clean_dates(values: List[Any]) -> List[Any]:
    """Clean date column"""
    date_patterns = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y",
        "%Y-%m-%d %H:%M:%S"
    ]
    
    cleaned = []
    for v in values:
        if not v or v == '':
            cleaned.append(None)
            continue
        
        parsed = None
        for pattern in date_patterns:
            try:
                parsed = datetime.datetime.strptime(str(v)[:19], pattern)
                break
            except:
                pass
        
        cleaned.append(parsed.isoformat() if parsed else str(v))
    
    return cleaned


def _clean_text(values: List[Any]) -> List[Any]:
    """Clean text column"""
    cleaned = []
    for v in values:
        if v and v != '':
            # Trim whitespace
            text = str(v).strip()
            cleaned.append(text)
        else:
            cleaned.append('')
    
    return cleaned


def detect_outliers(values: List[float], method: str = "iqr") -> List[int]:
    """Detect outlier indices"""
    if not values or len(values) < 4:
        return []
    
    numeric = [v for v in values if v is not None]
    if not numeric:
        return []
    
    numeric.sort()
    n = len(numeric)
    
    if method == "iqr":
        q1 = numeric[n // 4]
        q3 = numeric[3 * n // 4]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outliers = []
        for i, v in enumerate(values):
            if v is not None and (v < lower or v > upper):
                outliers.append(i)
        
        return outliers
    
    return []
