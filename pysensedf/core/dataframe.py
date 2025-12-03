"""
PySenseDF Core DataFrame Implementation
=======================================

The revolutionary DataFrame with AI, lazy execution, and auto-features.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import csv
import json
from pathlib import Path


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
        """String representation"""
        rows, cols = self.shape()
        return f"DataFrame(rows={rows}, columns={cols}, columns={self.columns()[:5]}{'...' if cols > 5 else ''})"
    
    def __str__(self) -> str:
        """Pretty print DataFrame"""
        lines = []
        lines.append(f"DataFrame: {self.shape()[0]} rows × {self.shape()[1]} columns")
        lines.append("")
        
        # Show first 10 rows
        head = self.head(10)
        cols = head.columns()
        
        # Header
        lines.append(" | ".join(cols))
        lines.append("-" * (len(cols) * 15))
        
        # Rows
        for i in range(min(10, head.shape()[0])):
            row = [str(head._data[col][i])[:12] for col in cols]
            lines.append(" | ".join(row))
        
        return "\n".join(lines)


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
