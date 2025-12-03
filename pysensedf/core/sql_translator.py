"""
SQL Query Translator
====================

Execute SQL queries on DataFrames
"""

import re
from typing import Dict, Any


def execute_sql(query: str, df) -> "DataFrame":
    """
    Execute SQL query on DataFrame
    
    Supports:
    - SELECT columns
    - WHERE conditions
    - GROUP BY
    - ORDER BY
    - LIMIT
    
    Example:
        SELECT city, AVG(income) as avg_income
        FROM df
        WHERE age > 25
        GROUP BY city
        ORDER BY avg_income DESC
        LIMIT 10
    """
    query = query.strip()
    
    # Parse SELECT
    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE)
    if not select_match:
        return df
    
    select_part = select_match.group(1).strip()
    
    # Parse WHERE
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP|\s+ORDER|\s+LIMIT|$)', query, re.IGNORECASE)
    if where_match:
        where_condition = where_match.group(1).strip()
        df = df.filter(where_condition)
    
    # Parse GROUP BY
    groupby_match = re.search(r'GROUP\s+BY\s+(\w+)', query, re.IGNORECASE)
    if groupby_match:
        group_col = groupby_match.group(1)
        
        # Check for aggregation function in SELECT
        if 'AVG' in select_part.upper() or 'MEAN' in select_part.upper():
            df = df.groupby(group_col).mean()
        elif 'SUM' in select_part.upper():
            df = df.groupby(group_col).sum()
        elif 'COUNT' in select_part.upper():
            df = df.groupby(group_col).count()
        else:
            df = df.groupby(group_col).count()
    
    # Parse ORDER BY
    orderby_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', query, re.IGNORECASE)
    if orderby_match:
        order_col = orderby_match.group(1)
        order_dir = orderby_match.group(2)
        descending = order_dir and order_dir.upper() == 'DESC'
        
        if order_col in df.columns():
            df = df.sort(order_col, descending=descending)
    
    # Parse LIMIT
    limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
    if limit_match:
        n = int(limit_match.group(1))
        df = df.head(n)
    
    # Parse SELECT columns
    if select_part != '*':
        columns = [c.strip().split(' as ')[0].strip() for c in select_part.split(',')]
        columns = [c for c in columns if not any(agg in c.upper() for agg in ['AVG', 'SUM', 'COUNT', 'MIN', 'MAX'])]
        
        if columns and all(c in df.columns() for c in columns):
            df = df.select(columns)
    
    return df
