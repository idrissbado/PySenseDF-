"""
Natural Language Query Parser
==============================

Converts natural language to DataFrame operations.
"""

from typing import Dict, Any


def parse_nl_query(query: str, df) -> Any:
    """
    Parse natural language query and execute
    
    Examples:
        "show top 10" → df.head(10)
        "show top 10 by revenue" → df.sort("revenue", descending=True).head(10)
        "filter age > 30" → df.filter("age > 30")
        "group by city and count" → df.groupby("city").count()
    """
    query = query.lower().strip()
    
    # Top N queries
    if "top" in query:
        try:
            n = int([word for word in query.split() if word.isdigit()][0])
        except:
            n = 10
        
        # Check for "by column"
        if " by " in query:
            parts = query.split(" by ")
            if len(parts) > 1:
                column = parts[1].strip().split()[0]
                if column in df.columns():
                    return df.sort(column, descending=True).head(n)
        
        return df.head(n)
    
    # Filter queries
    if "filter" in query or "where" in query:
        # Extract condition
        for word in ["filter", "where"]:
            if word in query:
                condition = query.split(word)[1].strip()
                return df.filter(condition)
    
    # Group by queries
    if "group by" in query or "groupby" in query:
        words = query.replace("group by", "groupby").split("groupby")[1].strip().split()
        if words:
            column = words[0]
            if column in df.columns():
                grouped = df.groupby(column)
                
                # Check aggregation
                if "count" in query:
                    return grouped.count()
                elif "sum" in query:
                    return grouped.sum()
                elif "mean" in query or "average" in query or "avg" in query:
                    return grouped.mean()
                else:
                    return grouped.count()
    
    # Sort queries
    if "sort" in query:
        words = query.split()
        for i, word in enumerate(words):
            if word in ["by", "sort"] and i + 1 < len(words):
                column = words[i + 1]
                if column in df.columns():
                    descending = "desc" in query or "descending" in query
                    return df.sort(column, descending=descending)
    
    # Default: show head
    return df.head(10)


def suggest_query(query: str) -> str:
    """Suggest corrections for query"""
    suggestions = {
        "mean": ["average", "avg", "mean"],
        "count": ["count", "number", "total"],
        "top": ["top", "first", "head"],
    }
    return query
