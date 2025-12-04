"""
Natural Language Query Parser
==============================

Convert natural language to DataFrame operations.

Examples:
- "show top 10 customers" → df.head(10)
- "filter by age > 30" → df.filter("age > 30")
- "group by city" → df.groupby("city")
"""

from typing import Any
import re


def parse_nl_query(query: str, df) -> Any:
    """
    Parse natural language query and execute on DataFrame
    
    Supports:
    - Top N: "show top 10", "get first 5"
    - Filter: "filter by age > 30", "where status is active"
    - Sort: "sort by revenue", "order by age descending"
    - Group: "group by city"
    - Select: "show only name and city"
    - Stats: "count", "average revenue", "sum of sales"
    """
    query_lower = query.lower().strip()
    
    # 1. Top/Head queries
    if any(word in query_lower for word in ['top', 'first', 'head']):
        match = re.search(r'(?:top|first|head)\s+(\d+)', query_lower)
        if match:
            n = int(match.group(1))
            print(f"✅ Showing top {n} rows")
            result = df.head(n)
            print(result)
            return result
        else:
            print(f"✅ Showing top 10 rows (default)")
            result = df.head(10)
            print(result)
            return result
    
    # 2. Bottom/Tail queries
    if any(word in query_lower for word in ['bottom', 'last', 'tail']):
        match = re.search(r'(?:bottom|last|tail)\s+(\d+)', query_lower)
        if match:
            n = int(match.group(1))
            print(f"✅ Showing last {n} rows")
            result = df.tail(n)
            print(result)
            return result
    
    # 3. Filter queries
    if any(word in query_lower for word in ['filter', 'where', 'only show']):
        # Extract condition
        patterns = [
            r'(?:filter|where)\s+(?:by\s+)?(.+)',
            r'only show\s+(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                condition = match.group(1).strip()
                
                # Try to parse condition
                if '>' in condition or '<' in condition or '==' in condition or '!=' in condition:
                    print(f"✅ Filtering by: {condition}")
                    result = df.filter(condition)
                    print(result)
                    return result
                elif ' is ' in condition:
                    # Convert "status is active" to "status == 'active'"
                    parts = condition.split(' is ')
                    if len(parts) == 2:
                        col, val = parts[0].strip(), parts[1].strip()
                        condition_str = f"{col} == '{val}'"
                        print(f"✅ Filtering by: {condition_str}")
    # 4. Sort queries
    if any(word in query_lower for word in ['sort', 'order']):
        match = re.search(r'(?:sort|order)\s+by\s+(\w+)', query_lower)
        if match:
            col = match.group(1)
            descending = 'descending' in query_lower or 'desc' in query_lower or 'highest' in query_lower
            print(f"✅ Sorting by {col} ({'descending' if descending else 'ascending'})")
            result = df.sort(col, descending=descending)
            print(result)
            return result
            col = match.group(1)
            descending = 'descending' in query_lower or 'desc' in query_lower or 'highest' in query_lower
            print(f"✅ Sorting by {col} ({'descending' if descending else 'ascending'})")
    # 5. Select columns
    if 'show only' in query_lower or 'select only' in query_lower:
        # Extract column names
        match = re.search(r'(?:show|select)\s+only\s+(.+)', query_lower)
        if match:
            cols_str = match.group(1)
            # Split by "and" or ","
            cols = [c.strip() for c in re.split(r',|\s+and\s+', cols_str)]
            print(f"✅ Selecting columns: {cols}")
            result = df.select(cols)
            print(result)
            return result
    
    # 6. Count queries
    if query_lower.startswith('count') or 'how many' in query_lower:
        count = df.shape()[0]
        print(f"✅ Count: {count} rows")
        return count
    
    # 7. Average/Mean queries
    if any(word in query_lower for word in ['average', 'mean']):
        match = re.search(r'(?:average|mean)\s+(?:of\s+)?(\w+)', query_lower)
        if match:
            col = match.group(1)
            try:
                values = [float(v) for v in df._data.get(col, []) if v and str(v).strip()]
                avg = sum(values) / len(values) if values else 0
                print(f"✅ Average {col}: {avg:.2f}")
                return avg
            except:
                print(f"❌ Cannot calculate average for column: {col}")
    
    # 8. Sum queries
    if 'sum' in query_lower or 'total' in query_lower:
        match = re.search(r'(?:sum|total)\s+(?:of\s+)?(\w+)', query_lower)
        if match:
            col = match.group(1)
            try:
                values = [float(v) for v in df._data.get(col, []) if v and str(v).strip()]
                total = sum(values)
                print(f"✅ Total {col}: {total:.2f}")
                return total
            except:
                print(f"❌ Cannot calculate sum for column: {col}")
    
    # 9. Unique/Distinct values
    if 'unique' in query_lower or 'distinct' in query_lower:
        match = re.search(r'(?:unique|distinct)\s+(\w+)', query_lower)
        if match:
            col = match.group(1)
            if col in df._data:
                unique_vals = list(set(df._data[col]))
                print(f"✅ Unique values in {col}: {len(unique_vals)}")
                return unique_vals
    
    # 10. Show all / Display
    if any(phrase in query_lower for phrase in ['show all', 'display all', 'show everything']):
        print(f"✅ Showing entire DataFrame")
        print(df)
        return df
    
    # Default: show what we understood
    print(f"🤖 Understood query: '{query}'")
    print(f"💡 Supported commands:")
    print(f"   • 'show top 10' / 'first 5'")
    print(f"   • 'filter by age > 30' / 'where status is active'")
    print(f"   • 'sort by revenue descending'")
    print(f"   • 'count' / 'how many rows'")
    print(f"   • 'average revenue' / 'sum of sales'")
    return df
