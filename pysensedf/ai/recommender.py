"""
DataFrame Profiler and Recommender
===================================

Smart profiling with recommendations
"""

from typing import Dict, Any, List


def profile_dataframe(df) -> Dict[str, Any]:
    """
    Comprehensive DataFrame profiling
    
    Returns:
    - Shape information
    - Column types
    - Missing values
    - Unique values
    - Recommendations
    """
    rows, cols = df.shape()
    
    profile = {
        "shape": {"rows": rows, "columns": cols},
        "columns": {},
        "recommendations": []
    }
    
    for col in df.columns():
        col_profile = _profile_column(df._data[col], col)
        profile["columns"][col] = col_profile
        
        # Add recommendations
        if col_profile["missing_pct"] > 5:
            profile["recommendations"].append(
                f"Column '{col}' has {col_profile['missing_pct']:.1f}% missing values - consider imputation"
            )
        
        if col_profile["unique_count"] == rows and col != "id":
            profile["recommendations"].append(
                f"Column '{col}' has all unique values - might be an identifier"
            )
    
    # Print pretty output
    _print_profile(profile)
    
    return profile


def _profile_column(values: List[Any], col_name: str) -> Dict[str, Any]:
    """Profile a single column"""
    n = len(values)
    
    # Count missing
    missing = sum(1 for v in values if v == '' or v is None)
    missing_pct = (missing / n * 100) if n > 0 else 0
    
    # Count unique
    unique = len(set(str(v) for v in values if v != '' and v is not None))
    
    # Infer type
    col_type = _infer_column_type(values)
    
    # Get sample values
    non_empty = [v for v in values if v != '' and v is not None]
    sample = non_empty[:5] if non_empty else []
    
    return {
        "type": col_type,
        "missing": missing,
        "missing_pct": missing_pct,
        "unique_count": unique,
        "sample_values": sample
    }


def _infer_column_type(values: List[Any]) -> str:
    """Infer column type"""
    non_empty = [v for v in values[:100] if v and v != '']
    
    if not non_empty:
        return "empty"
    
    # Check numeric
    numeric_count = 0
    for v in non_empty:
        try:
            float(str(v).replace(',', ''))
            numeric_count += 1
        except:
            pass
    
    if numeric_count / len(non_empty) > 0.8:
        # Check if integer
        int_count = sum(1 for v in non_empty if '.' not in str(v))
        if int_count / len(non_empty) > 0.9:
            return "integer"
        return "float"
    
    # Check boolean
    bool_values = {'true', 'false', 'yes', 'no', '0', '1', 't', 'f'}
    if all(str(v).lower() in bool_values for v in non_empty):
        return "boolean"
    
    return "string"


def _print_profile(profile: Dict[str, Any]):
    """Pretty print profile"""
    print("\n" + "=" * 60)
    print(f"📊 DataFrame Profile")
    print("=" * 60)
    print(f"Shape: {profile['shape']['rows']} rows × {profile['shape']['columns']} columns\n")
    
    print("Columns:")
    print("-" * 60)
    for col, info in list(profile['columns'].items())[:10]:  # Show first 10
        missing_str = f"{info['missing_pct']:.1f}% missing" if info['missing_pct'] > 0 else "complete"
        print(f"  {col:20} | {info['type']:10} | {info['unique_count']:6} unique | {missing_str}")
    
    if len(profile['columns']) > 10:
        print(f"  ... and {len(profile['columns']) - 10} more columns")
    
    if profile['recommendations']:
        print("\n" + "⚠️  Recommendations:")
        print("-" * 60)
        for rec in profile['recommendations'][:5]:
            print(f"  • {rec}")
    
    print("=" * 60 + "\n")
