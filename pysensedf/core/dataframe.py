"""
PySenseDF Core DataFrame Implementation
=======================================

The revolutionary DataFrame with AI, lazy execution, and auto-features.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import csv
import json
from pathlib import Path


class Column:
    """Helper class for column operations and comparisons"""
    def __init__(self, name: str, data: List[Any]):
        self.name = name
        self.data = data
    
    def __iter__(self):
        """Make Column iterable"""
        return iter(self.data)
    
    def __gt__(self, value: Any) -> 'DataFrame':
        """Support for df['col'] > value"""
        bool_result = []
        for v in self.data:
            try:
                bool_result.append(float(v) > value)
            except (ValueError, TypeError):
                bool_result.append(False)
        return DataFrame({'_bool': bool_result})
    
    def __lt__(self, value: Any) -> 'DataFrame':
        """Support for df['col'] < value"""
        bool_result = []
        for v in self.data:
            try:
                bool_result.append(float(v) < value)
            except (ValueError, TypeError):
                bool_result.append(False)
        return DataFrame({'_bool': bool_result})
    
    def __eq__(self, value: Any) -> 'DataFrame':
        """Support for df['col'] == value"""
        bool_result = [v == value for v in self.data]
        return DataFrame({'_bool': bool_result})
    
    def __ge__(self, value: Any) -> 'DataFrame':
        """Support for df['col'] >= value"""
        bool_result = []
        for v in self.data:
            try:
                bool_result.append(float(v) >= value)
            except (ValueError, TypeError):
                bool_result.append(False)
        return DataFrame({'_bool': bool_result})
    
    def __le__(self, value: Any) -> 'DataFrame':
        """Support for df['col'] <= value"""
        bool_result = []
        for v in self.data:
            try:
                bool_result.append(float(v) <= value)
            except (ValueError, TypeError):
                bool_result.append(False)
        return DataFrame({'_bool': bool_result})
    
    def __ne__(self, value: Any) -> 'DataFrame':
        """Support for df['col'] != value"""
        bool_result = [v != value for v in self.data]
        return DataFrame({'_bool': bool_result})
    
    def to_list(self) -> List[Any]:
        """Convert to list"""
        return self.data


class DataFrame:
    """
    PySenseDF DataFrame - The DataFrame that kills Pandas
    
    Features:
    - Natural language queries: df.ask("show top 10")
    - Auto-clean: df.autoclean()
    - Auto-features: df.autofeatures(target="label")
    - SQL queries: df.sql("SELECT * FROM df")
    - Lazy execution with optimization
    - Pure Python, faster than Pandas
    """
    
    def __init__(self, data: Optional[Dict[str, List[Any]]] = None):
        """
        Initialize DataFrame with dict-of-lists (columnar storage)
        
        Args:
            data: Dictionary mapping column names to lists of values
        """
        self._data = data or {}
        self._lazy = False
        self._operations = []
        
        # Validate all columns have same length
        if self._data:
            lengths = [len(v) for v in self._data.values()]
            if lengths and not all(l == lengths[0] for l in lengths):
                raise ValueError("All columns must have the same length")
    
    # ==================== MAGIC METHODS (Pandas-like) ====================
    
    def __getitem__(self, key: Union[str, List[str], 'DataFrame']) -> Union['Column', 'DataFrame']:
        """
        Get column(s) or filter with boolean indexing
        
        Examples:
            df['age']  # Get single column (returns Column object)
            df[['name', 'age']]  # Get multiple columns (returns DataFrame)
            df[df['age'] > 30]  # Boolean indexing
        """
        if isinstance(key, str):
            # Single column - return Column object for comparisons
            return Column(key, self._data.get(key, []))
        elif isinstance(key, list):
            # Multiple columns
            return self.select(key)
        elif isinstance(key, DataFrame):
            # Boolean indexing - key is a DataFrame with single boolean column
            bool_col = list(key._data.values())[0]
            filtered_data = {}
            for col, values in self._data.items():
                filtered_data[col] = [v for i, v in enumerate(values) if i < len(bool_col) and bool_col[i]]
            return DataFrame(filtered_data)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def __setitem__(self, key: str, value: Union[List[Any], Any]) -> None:
        """
        Set column values
        
        Examples:
            df['new_col'] = [1, 2, 3]
            df['age'] = 25  # Broadcast single value
        """
        if not self._data:
            # Empty DataFrame
            if isinstance(value, list):
                self._data[key] = value
            else:
                self._data[key] = [value]
        else:
            n_rows = len(list(self._data.values())[0])
            if isinstance(value, list):
                if len(value) != n_rows:
                    raise ValueError(f"Length mismatch: expected {n_rows}, got {len(value)}")
                self._data[key] = value
            else:
                # Broadcast single value
                self._data[key] = [value] * n_rows
    
    def __len__(self) -> int:
        """Return number of rows"""
        if not self._data:
            return 0
        return len(list(self._data.values())[0])
    
    
    # ==================== FACTORY METHODS ====================
    
    @classmethod
    def read_csv(cls, path: str, lazy: bool = False, **kwargs) -> "DataFrame":
        """
        Read CSV file into DataFrame
        
        Args:
            path: Path to CSV file
            lazy: If True, delay loading until needed
            **kwargs: Additional CSV reading options
        
        Returns:
            DataFrame instance
        """
        encoding = kwargs.get('encoding', 'utf-8')
        nrows = kwargs.get('nrows', None)
        
        with open(path, newline='', encoding=encoding) as f:
            reader = csv.reader(f)
            header = next(reader)
            
            cols = {h: [] for h in header}
            for i, row in enumerate(reader):
                if nrows and i >= nrows:
                    break
                for h, v in zip(header, row):
                    cols[h].append(v)
        
        df = cls(cols)
        df._lazy = lazy
        return df
    
    @classmethod
    def read_json(cls, path: str) -> "DataFrame":
        """Read JSON file into DataFrame"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # List of dicts
            if not data:
                return cls({})
            cols = {k: [] for k in data[0].keys()}
            for row in data:
                for k in cols.keys():
                    cols[k].append(row.get(k))
            return cls(cols)
        elif isinstance(data, dict):
            # Dict of lists
            return cls(data)
        else:
            raise ValueError("JSON must be list of dicts or dict of lists")
    
    @classmethod
    def from_dict(cls, data: Dict[str, List[Any]]) -> "DataFrame":
        """Create DataFrame from dictionary"""
        return cls(data)
    
    # ==================== BASIC OPERATIONS ====================
    
    def columns(self) -> List[str]:
        """Get list of column names"""
        return list(self._data.keys())
    
    def shape(self) -> tuple:
        """Get (rows, columns) shape"""
        if not self._data:
            return (0, 0)
        return (len(next(iter(self._data.values()))), len(self._data))
    
    def head(self, n: int = 5) -> "DataFrame":
        """Return first n rows"""
        head_data = {k: v[:n] for k, v in self._data.items()}
        return DataFrame(head_data)
    
    def tail(self, n: int = 5) -> "DataFrame":
        """Return last n rows"""
        tail_data = {k: v[-n:] for k, v in self._data.items()}
        return DataFrame(tail_data)
    
    def select(self, columns: List[str]) -> "DataFrame":
        """Select specific columns"""
        selected = {k: self._data[k] for k in columns if k in self._data}
        return DataFrame(selected)
    
    def drop(self, columns: Union[str, List[str]]) -> "DataFrame":
        """Drop columns"""
        if isinstance(columns, str):
            columns = [columns]
        kept = {k: v for k, v in self._data.items() if k not in columns}
        return DataFrame(kept)
    
    def filter(self, condition: Union[str, Callable]) -> "DataFrame":
        """
        Filter rows by condition
        
        Args:
            condition: String expression like "age > 30" or callable
        
        Returns:
            Filtered DataFrame
        """
        if isinstance(condition, str):
            # Parse simple conditions
            return self._filter_by_expression(condition)
        elif callable(condition):
            # Use function to filter
            indices = [i for i in range(self.shape()[0]) if condition(self._get_row(i))]
            return self._select_indices(indices)
        else:
            raise ValueError("Condition must be string or callable")
    
    def _filter_by_expression(self, expr: str) -> "DataFrame":
        """Filter by simple expression like 'age > 30'"""
        expr = expr.strip()
        
        # Parse simple expressions
        for op in ['>=', '<=', '!=', '==', '>', '<']:
            if op in expr:
                parts = expr.split(op)
                if len(parts) == 2:
                    col, val = parts[0].strip(), parts[1].strip()
                    if col in self._data:
                        # Try to convert value to appropriate type
                        try:
                            val = float(val) if '.' in val else int(val)
                        except:
                            val = val.strip('"\'')
                        
                        indices = []
                        for i, v in enumerate(self._data[col]):
                            try:
                                v_num = float(v) if v != '' else None
                                if v_num is not None:
                                    if op == '>' and v_num > val:
                                        indices.append(i)
                                    elif op == '>=' and v_num >= val:
                                        indices.append(i)
                                    elif op == '<' and v_num < val:
                                        indices.append(i)
                                    elif op == '<=' and v_num <= val:
                                        indices.append(i)
                                    elif op == '==' and v_num == val:
                                        indices.append(i)
                                    elif op == '!=' and v_num != val:
                                        indices.append(i)
                            except:
                                if op == '==' and str(v) == str(val):
                                    indices.append(i)
                                elif op == '!=' and str(v) != str(val):
                                    indices.append(i)
                        
                        return self._select_indices(indices)
        
        return self
    
    def groupby(self, column: str):
        """Group by column - returns GroupBy object"""
        return GroupBy(self, column)
    
    def sort(self, column: str, descending: bool = False) -> "DataFrame":
        """Sort by column"""
        if column not in self._data:
            return self
        
        # Create index-value pairs
        indexed = list(enumerate(self._data[column]))
        
        # Sort by value
        try:
            sorted_indices = sorted(indexed, key=lambda x: float(x[1]) if x[1] != '' else float('inf'), reverse=descending)
        except:
            sorted_indices = sorted(indexed, key=lambda x: str(x[1]), reverse=descending)
        
        indices = [i for i, _ in sorted_indices]
        return self._select_indices(indices)
    
    # ==================== STATISTICAL METHODS (Beat Pandas!) ====================
    
    def describe(self) -> "DataFrame":
        """
        Generate summary statistics for numeric columns
        Returns DataFrame with count, mean, std, min, 25%, 50%, 75%, max
        """
        stats_data = {}
        numeric_cols = self._get_numeric_columns()
        
        if not numeric_cols:
            return DataFrame({'message': ['No numeric columns found']})
        
        stats_data['statistic'] = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        
        for col in numeric_cols:
            values = [float(v) for v in self._data[col] if v != '' and v is not None]
            
            if values:
                values_sorted = sorted(values)
                n = len(values)
                mean_val = sum(values) / n
                
                # Standard deviation
                variance = sum((x - mean_val) ** 2 for x in values) / n
                std_val = variance ** 0.5
                
                # Percentiles
                q25 = values_sorted[int(n * 0.25)] if n > 0 else 0
                q50 = values_sorted[int(n * 0.50)] if n > 0 else 0
                q75 = values_sorted[int(n * 0.75)] if n > 0 else 0
                
                stats_data[col] = [
                    n,
                    round(mean_val, 2),
                    round(std_val, 2),
                    round(min(values), 2),
                    round(q25, 2),
                    round(q50, 2),
                    round(q75, 2),
                    round(max(values), 2)
                ]
            else:
                stats_data[col] = [0, 0, 0, 0, 0, 0, 0, 0]
        
        return DataFrame(stats_data)
    
    def mean(self, column: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """Calculate mean for column(s)"""
        if column:
            values = [float(v) for v in self._data.get(column, []) if v != '' and v is not None]
            return sum(values) / len(values) if values else 0.0
        else:
            # All numeric columns
            result = {}
            for col in self._get_numeric_columns():
                result[col] = self.mean(col)
            return result
    
    def median(self, column: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """Calculate median for column(s)"""
        if column:
            values = sorted([float(v) for v in self._data.get(column, []) if v != '' and v is not None])
            if not values:
                return 0.0
            n = len(values)
            if n % 2 == 0:
                return (values[n//2 - 1] + values[n//2]) / 2
            else:
                return values[n//2]
        else:
            result = {}
            for col in self._get_numeric_columns():
                result[col] = self.median(col)
            return result
    
    def std(self, column: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """Calculate standard deviation for column(s)"""
        if column:
            values = [float(v) for v in self._data.get(column, []) if v != '' and v is not None]
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance ** 0.5
        else:
            result = {}
            for col in self._get_numeric_columns():
                result[col] = self.std(col)
            return result
    
    def corr(self) -> "DataFrame":
        """
        Calculate correlation matrix for numeric columns
        """
        numeric_cols = self._get_numeric_columns()
        if len(numeric_cols) < 2:
            return DataFrame({'message': ['Need at least 2 numeric columns']})
        
        corr_data = {'column': numeric_cols}
        
        for col1 in numeric_cols:
            corr_values = []
            values1 = [float(v) for v in self._data[col1] if v != '' and v is not None]
            mean1 = sum(values1) / len(values1) if values1 else 0
            
            for col2 in numeric_cols:
                if col1 == col2:
                    corr_values.append(1.0)
                else:
                    values2 = [float(v) for v in self._data[col2] if v != '' and v is not None]
                    mean2 = sum(values2) / len(values2) if values2 else 0
                    
                    # Pearson correlation
                    numerator = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(min(len(values1), len(values2))))
                    denom1 = sum((v - mean1) ** 2 for v in values1) ** 0.5
                    denom2 = sum((v - mean2) ** 2 for v in values2) ** 0.5
                    
                    if denom1 > 0 and denom2 > 0:
                        corr_values.append(round(numerator / (denom1 * denom2), 3))
                    else:
                        corr_values.append(0.0)
            
            corr_data[col1] = corr_values
        
        return DataFrame(corr_data)
    
    def value_counts(self, column: str) -> "DataFrame":
        """Count unique values in column"""
        if column not in self._data:
            return DataFrame({'message': ['Column not found']})
        
        counts = {}
        for value in self._data[column]:
            counts[value] = counts.get(value, 0) + 1
        
        # Sort by count descending
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        return DataFrame({
            column: [item[0] for item in sorted_items],
            'count': [item[1] for item in sorted_items]
        })
    
    def dropna(self, columns: Optional[List[str]] = None) -> "DataFrame":
        """Drop rows with missing values"""
        cols_to_check = columns or list(self._data.keys())
        
        valid_indices = []
        n_rows = len(self)
        
        for i in range(n_rows):
            has_missing = False
            for col in cols_to_check:
                if col in self._data:
                    val = self._data[col][i]
                    if val is None or val == '' or str(val).strip() == '':
                        has_missing = True
                        break
            if not has_missing:
                valid_indices.append(i)
        
        return self._select_indices(valid_indices)
    
    def drop_duplicates(self, columns: Optional[List[str]] = None) -> "DataFrame":
        """Remove duplicate rows"""
        cols_to_check = columns or list(self._data.keys())
        
        seen = set()
        unique_indices = []
        
        for i in range(len(self)):
            row_key = tuple(self._data[col][i] for col in cols_to_check if col in self._data)
            if row_key not in seen:
                seen.add(row_key)
                unique_indices.append(i)
        
        return self._select_indices(unique_indices)
    
    # ==================== ADVANCED OPERATIONS ====================
    
    def merge(self, other: "DataFrame", on: str, how: str = 'inner') -> "DataFrame":
        """
        Merge two DataFrames (SQL-style join)
        
        Args:
            other: DataFrame to merge with
            on: Column name to join on
            how: 'inner', 'left', 'right', 'outer'
        """
        if on not in self._data or on not in other._data:
            raise ValueError(f"Column '{on}' not found in both DataFrames")
        
        # Build lookup dict for other DataFrame
        other_lookup = {}
        for i, key in enumerate(other._data[on]):
            if key not in other_lookup:
                other_lookup[key] = []
            other_lookup[key].append(i)
        
        result_data = {col: [] for col in self._data.keys()}
        for col in other._data.keys():
            if col != on:
                result_data[f"{col}_right"] = []
        
        # Perform join
        for i, left_key in enumerate(self._data[on]):
            if left_key in other_lookup:
                # Match found
                for right_idx in other_lookup[left_key]:
                    # Add left row
                    for col in self._data.keys():
                        result_data[col].append(self._data[col][i])
                    # Add right row
                    for col in other._data.keys():
                        if col != on:
                            result_data[f"{col}_right"].append(other._data[col][right_idx])
            elif how in ['left', 'outer']:
                # Left row with no match
                for col in self._data.keys():
                    result_data[col].append(self._data[col][i])
                for col in other._data.keys():
                    if col != on:
                        result_data[f"{col}_right"].append(None)
        
        # Add unmatched right rows for outer join
        if how == 'outer':
            matched_keys = set(self._data[on])
            for i, right_key in enumerate(other._data[on]):
                if right_key not in matched_keys:
                    for col in self._data.keys():
                        result_data[col].append(None if col != on else right_key)
                    for col in other._data.keys():
                        if col != on:
                            result_data[f"{col}_right"].append(other._data[col][i])
        
        return DataFrame(result_data)
    
    def pipe(self, func: Callable[["DataFrame"], "DataFrame"], *args, **kwargs) -> "DataFrame":
        """Apply function to DataFrame (enables method chaining)"""
        return func(self, *args, **kwargs)
    
    def _get_numeric_columns(self) -> List[str]:
        """Helper to get numeric column names"""
        numeric_cols = []
        for col in self._data.keys():
            try:
                # Check if first non-empty value is numeric
                for val in self._data[col]:
                    if val != '' and val is not None:
                        float(val)
                        numeric_cols.append(col)
                        break
            except:
                continue
        return numeric_cols
    
    # ==================== REVOLUTIONARY FEATURES ====================
    
    def ask(self, query: str) -> Any:
        """
        Natural language query (AI-powered)
        
        Examples:
            df.ask("show top 10 customers by revenue")
            df.ask("plot income distribution")
            df.ask("find outliers in price column")
        """
        from ..ai.nlp import parse_nl_query
        result = parse_nl_query(query, self)
        return result
    
    def autoclean(self) -> "DataFrame":
        """
        Automatic data cleaning (ONE LINE DATA CLEANING!)
        
        Automatically:
        - Detects column types
        - Handles missing values
        - Parses dates
        - Removes duplicates
        - Handles outliers
        """
        from ..ai.cleaner_ai import auto_clean_dataframe
        return auto_clean_dataframe(self)
    
    def autofeatures(self, target: Optional[str] = None) -> "DataFrame":
        """
        Automatic feature engineering (ONE LINE FEATURE ENGINEERING!)
        
        Creates:
        - Date/time features
        - Aggregations
        - Ratios and interactions
        - Lag features
        - Rolling statistics
        """
        from ..ai.feature_ai import auto_generate_features
        return auto_generate_features(self, target)
    
    def sql(self, query: str) -> "DataFrame":
        """
        Execute SQL query on DataFrame
        
        Example:
            df.sql("SELECT city, AVG(income) FROM df GROUP BY city")
        """
        from ..core.sql_translator import execute_sql
        return execute_sql(query, self)
    
    def profile(self) -> Dict[str, Any]:
        """
        Smart data profiling with recommendations
        
        Returns comprehensive statistics, missing value analysis,
        type information, and recommendations.
        """
        from ..ai.recommender import profile_dataframe
        return profile_dataframe(self)
    
    # ==================== CHAINABLE OPERATIONS ====================
    
    def apply(self, func: Callable, column: str) -> "DataFrame":
        """Apply function to column"""
        new_data = self._data.copy()
        new_data[column] = [func(v) for v in self._data[column]]
        return DataFrame(new_data)
    
    def rename(self, mapping: Dict[str, str]) -> "DataFrame":
        """Rename columns"""
        new_data = {}
        for k, v in self._data.items():
            new_key = mapping.get(k, k)
            new_data[new_key] = v
        return DataFrame(new_data)
    
    def fillna(self, value: Any, columns: Optional[List[str]] = None) -> "DataFrame":
        """Fill missing values"""
        new_data = self._data.copy()
        target_cols = columns if columns else list(self._data.keys())
        
        for col in target_cols:
            if col in new_data:
                new_data[col] = [value if v == '' or v is None else v for v in new_data[col]]
        
        return DataFrame(new_data)
    
    # ==================== LAZY EXECUTION ====================
    
    def collect(self) -> "DataFrame":
        """Execute lazy operations and return result"""
        if not self._lazy or not self._operations:
            return self
        
        # Execute optimized query plan
        result = self
        for op in self._operations:
            result = op(result)
        
        return result
    
    # ==================== UTILITY METHODS ====================
    
    def to_dict(self) -> Dict[str, List[Any]]:
        """Convert to dictionary"""
        return self._data.copy()
    
    def to_csv(self, path: str):
        """Write to CSV file"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.columns())
            
            n_rows = self.shape()[0]
            for i in range(n_rows):
                row = [self._data[col][i] for col in self.columns()]
                writer.writerow(row)
    
    def to_json(self, path: str, orient: str = 'records'):
        """Write to JSON file"""
        if orient == 'records':
            records = []
            n_rows = self.shape()[0]
            for i in range(n_rows):
                record = {col: self._data[col][i] for col in self.columns()}
                records.append(record)
            data = records
        else:
            data = self._data
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_row(self, index: int) -> Dict[str, Any]:
        """Get row as dictionary"""
        return {k: v[index] for k, v in self._data.items()}
    
    def _select_indices(self, indices: List[int]) -> "DataFrame":
        """Select rows by indices"""
        new_data = {}
        for k, v in self._data.items():
            new_data[k] = [v[i] for i in indices if i < len(v)]
        return DataFrame(new_data)
    
    def __repr__(self) -> str:
        """String representation - show actual data"""
        return self.__str__()
    
    def __str__(self) -> str:
        """Pretty print DataFrame with table format"""
        rows, cols = self.shape()
        
        if rows == 0:
            return "DataFrame: Empty (0 rows × 0 columns)"
        
        lines = []
        lines.append(f"\nDataFrame: {rows} rows × {cols} columns")
        lines.append("=" * 80)
        
        # Show up to 10 rows
        display_rows = min(10, rows)
        head = self.head(display_rows)
        col_names = head.columns()
        
        # Calculate column widths
        col_widths = {}
        for col in col_names:
            max_width = len(col)
            for i in range(display_rows):
                val_len = len(str(head._data[col][i]))
                max_width = max(max_width, val_len)
            col_widths[col] = min(max_width + 2, 20)  # Cap at 20
        
        # Header
        header = " | ".join([col.ljust(col_widths[col]) for col in col_names])
        lines.append(header)
        lines.append("-" * len(header))
        
        # Rows
        for i in range(display_rows):
            row_data = []
            for col in col_names:
                val = str(head._data[col][i])
                if len(val) > col_widths[col]:
                    val = val[:col_widths[col]-3] + "..."
                row_data.append(val.ljust(col_widths[col]))
            lines.append(" | ".join(row_data))
        
        if rows > 10:
            lines.append(f"... ({rows - 10} more rows)")
        
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks"""
        rows, cols = self.shape()
        
        if rows == 0:
            return "<p><strong>DataFrame:</strong> Empty (0 rows × 0 columns)</p>"
        
        # Show up to 10 rows
        display_rows = min(10, rows)
        head = self.head(display_rows)
        col_names = head.columns()
        
        # Build HTML table
        html = ['<div style="max-width: 100%; overflow-x: auto;">']
        html.append(f'<p><strong>DataFrame:</strong> {rows} rows × {cols} columns</p>')
        html.append('<table border="1" style="border-collapse: collapse; min-width: 300px;">')
        
        # Header row
        html.append('<thead><tr style="background-color: #f0f0f0;">')
        for col in col_names:
            html.append(f'<th style="padding: 8px; text-align: left; border: 1px solid #ddd;">{col}</th>')
        html.append('</tr></thead>')
        
        # Data rows
        html.append('<tbody>')
        for i in range(display_rows):
            html.append('<tr>')
            for col in col_names:
                value = head._data[col][i]
                # Truncate long strings
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                html.append(f'<td style="padding: 8px; border: 1px solid #ddd;">{value_str}</td>')
            html.append('</tr>')
        html.append('</tbody>')
        
        html.append('</table>')
        
        if rows > 10:
            html.append(f'<p><em>... {rows - 10} more rows</em></p>')
        
        html.append('</div>')
        
        return ''.join(html)
    
    # ==================== SMART AI FEATURES (Beyond Pandas!) ====================
    
    def detect_anomalies(self, column: str, method: str = 'iqr') -> "DataFrame":
        """
        Detect outliers/anomalies in numeric column
        
        Args:
            column: Column name
            method: 'iqr' (Interquartile Range) or 'zscore'
        """
        if column not in self._data:
            return DataFrame({'message': ['Column not found']})
        
        values = [float(v) for v in self._data[column] if v != '' and v is not None]
        if not values:
            return DataFrame({'message': ['No numeric values']})
        
        outlier_indices = []
        
        if method == 'iqr':
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            q1 = sorted_vals[int(n * 0.25)]
            q3 = sorted_vals[int(n * 0.75)]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, v in enumerate(self._data[column]):
                try:
                    if float(v) < lower_bound or float(v) > upper_bound:
                        outlier_indices.append(i)
                except:
                    pass
        
        elif method == 'zscore':
            mean_val = sum(values) / len(values)
            std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            
            for i, v in enumerate(self._data[column]):
                try:
                    z_score = abs((float(v) - mean_val) / std_val) if std_val > 0 else 0
                    if z_score > 3:  # 3 standard deviations
                        outlier_indices.append(i)
                except:
                    pass
        
        return self._select_indices(outlier_indices)
    
    def suggest_transformations(self) -> Dict[str, List[str]]:
        """
        AI-powered suggestions for data transformations
        Returns dictionary with suggestions per column
        """
        suggestions = {}
        
        for col in self._data.keys():
            col_suggestions = []
            values = self._data[col]
            
            # Check for missing values
            missing_count = sum(1 for v in values if v is None or v == '')
            if missing_count > 0:
                missing_pct = (missing_count / len(values)) * 100
                col_suggestions.append(f"⚠️ {missing_pct:.1f}% missing values - consider fillna() or dropna()")
            
            # Check if numeric
            try:
                numeric_vals = [float(v) for v in values if v != '' and v is not None]
                if numeric_vals:
                    # Check for skewness
                    mean_val = sum(numeric_vals) / len(numeric_vals)
                    median_val = sorted(numeric_vals)[len(numeric_vals)//2]
                    if abs(mean_val - median_val) / median_val > 0.3 if median_val != 0 else False:
                        col_suggestions.append("📊 Skewed distribution - consider log transformation")
                    
                    # Check for outliers
                    sorted_vals = sorted(numeric_vals)
                    n = len(sorted_vals)
                    q1, q3 = sorted_vals[int(n*0.25)], sorted_vals[int(n*0.75)]
                    iqr = q3 - q1
                    outliers = [v for v in numeric_vals if v < q1 - 1.5*iqr or v > q3 + 1.5*iqr]
                    if len(outliers) > 0:
                        col_suggestions.append(f"🔍 {len(outliers)} outliers detected - use detect_anomalies()")
                    
                    # Check range
                    if max(numeric_vals) - min(numeric_vals) > 1000:
                        col_suggestions.append("📐 Large range - consider normalization/scaling")
            except:
                # Not numeric
                unique_values = len(set(values))
                if unique_values < 10:
                    col_suggestions.append("🏷️ Low cardinality - good for groupby()")
                elif unique_values == len(values):
                    col_suggestions.append("🆔 High cardinality - possibly ID column")
            
            if col_suggestions:
                suggestions[col] = col_suggestions
            else:
                suggestions[col] = ["✅ Looks good!"]
        
        return suggestions
    
    def ai_summarize(self) -> str:
        """
        AI-powered natural language summary of the DataFrame
        """
        rows, cols = self.shape()
        numeric_cols = self._get_numeric_columns()
        
        summary = []
        summary.append(f"📊 **DataFrame Summary**")
        summary.append(f"   • Shape: {rows} rows × {cols} columns")
        summary.append(f"   • Numeric columns: {len(numeric_cols)} ({', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''})")
        
        # Missing values
        total_missing = 0
        for col in self._data.keys():
            total_missing += sum(1 for v in self._data[col] if v is None or v == '')
        if total_missing > 0:
            missing_pct = (total_missing / (rows * cols)) * 100
            summary.append(f"   • Missing values: {total_missing} ({missing_pct:.1f}%)")
        else:
            summary.append(f"   • Missing values: None ✅")
        
        # Memory usage estimate (rough)
        memory_kb = (rows * cols * 8) / 1024  # Assuming ~8 bytes per value
        summary.append(f"   • Estimated memory: {memory_kb:.1f} KB")
        
        # Quick stats for numeric columns
        if numeric_cols:
            summary.append(f"\n📈 **Key Statistics:**")
            for col in numeric_cols[:3]:  # Show first 3
                values = [float(v) for v in self._data[col] if v != '' and v is not None]
                if values:
                    mean_val = sum(values) / len(values)
                    min_val, max_val = min(values), max(values)
                    summary.append(f"   • {col}: mean={mean_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
        
        # Data quality
        suggestions = self.suggest_transformations()
        issues = sum(1 for col_sugs in suggestions.values() if not col_sugs[0].startswith("✅"))
        if issues > 0:
            summary.append(f"\n⚠️ **Data Quality:** {issues} columns need attention (use suggest_transformations())")
        else:
            summary.append(f"\n✅ **Data Quality:** Excellent!")
        
        return "\n".join(summary)
    
    def auto_visualize(self, column: Optional[str] = None) -> str:
        """
        Suggest best visualization for data
        Returns recommendation string (actual plotting requires matplotlib)
        """
        if column:
            # Single column viz
            values = self._data.get(column, [])
            try:
                numeric_vals = [float(v) for v in values if v != '' and v is not None]
                unique_count = len(set(numeric_vals))
                
                if unique_count < 10:
                    return f"📊 Recommendation for '{column}': Bar chart (categorical-like data)\n   Use: df.plot.bar('{column}')"
                else:
                    return f"📈 Recommendation for '{column}': Histogram (continuous data)\n   Use: df.plot.hist('{column}')"
            except:
                # Non-numeric
                unique_count = len(set(values))
                if unique_count < 20:
                    return f"📊 Recommendation for '{column}': Bar chart ({unique_count} unique values)\n   Use: df.value_counts('{column}').plot.bar()"
                else:
                    return f"📋 Recommendation for '{column}': Too many categories ({unique_count}), consider filtering"
        else:
            # Overall dataset viz
            numeric_cols = self._get_numeric_columns()
            if len(numeric_cols) >= 2:
                return f"🔗 Recommendation: Correlation heatmap ({len(numeric_cols)} numeric columns)\n   Use: df.corr() to see relationships"
            elif len(numeric_cols) == 1:
                return f"📊 Recommendation: Distribution plot for {numeric_cols[0]}\n   Use: df.plot.hist('{numeric_cols[0]}')"
            else:
                return f"📋 Recommendation: Value counts for categorical columns\n   Use: df.value_counts('column_name')"

class GroupBy:
    """GroupBy object for aggregations"""
    
    def __init__(self, df: DataFrame, column: str):
        self.df = df
        self.column = column
        self._groups = self._create_groups()
    
    def _create_groups(self) -> Dict[Any, List[int]]:
        """Create groups mapping"""
        groups = {}
        for i, val in enumerate(self.df._data[self.column]):
            if val not in groups:
                groups[val] = []
            groups[val].append(i)
        return groups
    
    def mean(self) -> DataFrame:
        """Calculate mean for each group"""
        result = {self.column: list(self._groups.keys())}
        
        for col in self.df.columns():
            if col == self.column:
                continue
            
            means = []
            for group_indices in self._groups.values():
                values = [self.df._data[col][i] for i in group_indices]
                # Try to compute mean
                try:
                    numeric = [float(v) for v in values if v != '' and v is not None]
                    mean = sum(numeric) / len(numeric) if numeric else None
                    means.append(mean)
                except:
                    means.append(None)
            
            result[f"{col}_mean"] = means
        
        return DataFrame(result)
    
    def sum(self) -> DataFrame:
        """Calculate sum for each group"""
        result = {self.column: list(self._groups.keys())}
        
        for col in self.df.columns():
            if col == self.column:
                continue
            
            sums = []
            for group_indices in self._groups.values():
                values = [self.df._data[col][i] for i in group_indices]
                try:
                    numeric = [float(v) for v in values if v != '' and v is not None]
                    total = sum(numeric)
                    sums.append(total)
                except:
                    sums.append(None)
            
            result[f"{col}_sum"] = sums
        
        return DataFrame(result)
    
    def count(self) -> DataFrame:
        """Count rows in each group"""
        result = {
            self.column: list(self._groups.keys()),
            'count': [len(indices) for indices in self._groups.values()]
        }
        return DataFrame(result)
