"""
PySenseDF Quickstart Example
=============================

Demonstrates the revolutionary features that kill Pandas
"""

from pysensedf import DataFrame

def main():
    print("🚀 PySenseDF - The DataFrame That Kills Pandas\n")
    print("=" * 60)
    
    # Create sample data
    data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry"],
        "age": ["25", "30", "35", "28", "32", "45", "29", "38"],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago", "LA"],
        "income": ["50000", "75000", "60000", "55000", "80000", "95000", "58000", "72000"],
        "date_joined": ["2023-01-15", "2023-02-20", "2023-01-10", "2023-03-05", 
                       "2023-02-18", "2023-01-25", "2023-03-12", "2023-02-28"]
    }
    
    df = DataFrame.from_dict(data)
    
    # 1. Basic operations
    print("\n1️⃣  Basic DataFrame")
    print(df)
    
    # 2. Natural language query
    print("\n2️⃣  Natural Language Query: 'show top 3 by income'")
    result = df.ask("show top 3 by income")
    print(result)
    
    # 3. Auto-clean (type inference, etc.)
    print("\n3️⃣  Auto-Clean (one line!)")
    df_clean = df.autoclean()
    print(f"✓ Cleaned! Types inferred automatically")
    
    # 4. Auto-features
    print("\n4️⃣  Auto-Features (one line!)")
    df_features = df_clean.autofeatures(target="income")
    print(f"✓ Generated {len(df_features.columns()) - len(df.columns())} new features!")
    print(f"New columns: {[c for c in df_features.columns() if c not in df.columns()][:5]}...")
    
    # 5. Group by operations
    print("\n5️⃣  Group By City")
    grouped = df_clean.groupby("city").mean()
    print(grouped)
    
    # 6. SQL query
    print("\n6️⃣  SQL Query")
    sql_result = df_clean.sql("SELECT city, COUNT(*) FROM df GROUP BY city")
    print(sql_result)
    
    # 7. Chainable operations
    print("\n7️⃣  Chainable Operations")
    result = (df_clean
        .filter("age > 28")
        .sort("income", descending=True)
        .select(["name", "city", "income"])
        .head(3)
    )
    print(result)
    
    # 8. Profile
    print("\n8️⃣  Smart Profiling")
    profile = df_clean.profile()
    
    # 9. Save to file
    print("\n9️⃣  Save to CSV")
    df_clean.to_csv("output.csv")
    print("✓ Saved to output.csv")
    
    print("\n" + "=" * 60)
    print("🎉 PySenseDF Demo Complete!")
    print("\n💡 Remember:")
    print("   • df.ask() for natural language")
    print("   • df.autoclean() for one-line cleaning")
    print("   • df.autofeatures() for one-line features")
    print("   • df.sql() for SQL queries")
    print("   • df.profile() for smart insights")
    print("\n🚀 100 lines of Pandas → 3 lines of PySenseDF!")
    print("=" * 60)


if __name__ == "__main__":
    main()
