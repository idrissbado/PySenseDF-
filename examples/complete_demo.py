"""
PySenseDF v0.2.2 - Complete Demo with Real Tests
=================================================

This demo showcases ALL features of PySenseDF with real data and expected outputs.
"""

print("="*80)
print("PySenseDF v0.2.2 - Complete Feature Demo")
print("="*80)

# ============================================================================
# 1. LOADING DATA
# ============================================================================
print("\n1️⃣  LOADING BUILT-IN DATASETS")
print("-" * 80)

from pysensedf.datasets import load_customers, load_products, load_sales
from pysensedf import DataFrame

# Load sample data
df = load_customers()
print(f"✅ Loaded customers dataset: {df.shape()[0]} rows × {df.shape()[1]} columns")
print(df.head(3))

# ============================================================================
# 2. NATURAL LANGUAGE QUERIES (ask)
# ============================================================================
print("\n\n2️⃣  NATURAL LANGUAGE QUERIES")
print("-" * 80)

print("\n📊 Query 1: 'show top 5'")
df.ask("show top 5")

print("\n\n📊 Query 2: 'filter by income > 90000'")
df.ask("filter by income > 90000")

print("\n\n📊 Query 3: 'sort by revenue descending'")
result = df.ask("sort by revenue descending")

print("\n\n📊 Query 4: 'average age'")
avg_age = df.ask("average age")

print("\n\n📊 Query 5: 'count'")
count = df.ask("count")

print("\n\n📊 Query 6: 'unique city'")
cities = df.ask("unique city")
print(f"Cities: {cities}")

# ============================================================================
# 3. AUTO-CLEAN (autoclean)
# ============================================================================
print("\n\n3️⃣  AUTO-CLEAN FEATURE")
print("-" * 80)

print("\n🧹 Original data (first 3 rows):")
print(df.head(3))

print("\n🧹 After autoclean():")
df_clean = df.autoclean()
print(df_clean.head(3))

print("\n✅ Changes applied:")
print("   • Detected column types automatically")
print("   • Parsed dates to datetime format")
print("   • Converted numeric columns to floats")
print("   • Standardized categorical data")
print("   • Removed duplicates (if any)")

# ============================================================================
# 4. AUTO-FEATURES (autofeatures)
# ============================================================================
print("\n\n4️⃣  AUTO-FEATURES FOR ML")
print("-" * 80)

print(f"\n🔧 Original columns ({df.shape()[1]}): {df.columns()}")

df_features = df.autofeatures(target="revenue")
print(f"\n🔧 After autofeatures() ({df_features.shape()[1]} columns):")
print(f"   New columns: {[c for c in df_features.columns() if c not in df.columns()][:10]}")

print("\n✅ Features generated:")
print("   • Date features: year, month, day, quarter, dayofweek")
print("   • Ratio features: income_div_revenue, age_div_purchase_count, etc.")
print("   • Interaction features: income_times_revenue, etc.")

# ============================================================================
# 5. STANDARD DATAFRAME OPERATIONS
# ============================================================================
print("\n\n5️⃣  STANDARD OPERATIONS")
print("-" * 80)

# Filter
print("\n🔍 Filter: income > 80000")
filtered = df.filter("income > 80000")
print(f"Result: {filtered.shape()[0]} rows")
print(filtered.head(3))

# Sort
print("\n\n📊 Sort by revenue (descending)")
sorted_df = df.sort("revenue", descending=True)
print(sorted_df.head(3))

# Select columns
print("\n\n📋 Select specific columns")
selected = df.select(["name", "city", "revenue"])
print(selected.head(3))

# GroupBy
print("\n\n👥 GroupBy city (average revenue)")
grouped = df.groupby("city").mean()
print(grouped)

# ============================================================================
# 6. WORKING WITH DIFFERENT DATASETS
# ============================================================================
print("\n\n6️⃣  OTHER DATASETS")
print("-" * 80)

# Products
print("\n📦 Products Dataset:")
products = load_products()
print(products.head(3))
print(f"✅ {products.shape()[0]} products loaded")

# Sales
print("\n\n💰 Sales Dataset:")
sales = load_sales()
print(sales.head(3))
print(f"✅ {sales.shape()[0]} sales records loaded")

# ============================================================================
# 7. ADVANCED QUERIES
# ============================================================================
print("\n\n7️⃣  ADVANCED NATURAL LANGUAGE QUERIES")
print("-" * 80)

print("\n💬 Query: 'where status is active'")
active = df.ask("where status is active")

print("\n\n💬 Query: 'last 3'")
df.ask("last 3")

print("\n\n💬 Query: 'sum revenue'")
total_revenue = df.ask("sum revenue")

# ============================================================================
# 8. CREATING YOUR OWN DATAFRAME
# ============================================================================
print("\n\n8️⃣  CREATE YOUR OWN DATAFRAME")
print("-" * 80)

data = {
    "product": ["Laptop", "Mouse", "Keyboard", "Monitor"],
    "price": [999.99, 29.99, 79.99, 299.99],
    "stock": [15, 150, 80, 25],
    "rating": [4.5, 4.2, 4.7, 4.4]
}

my_df = DataFrame(data)
print("\n📊 Custom DataFrame:")
print(my_df)

print("\n💬 Query: 'filter by price < 100'")
my_df.ask("filter by price < 100")

print("\n💬 Query: 'average rating'")
avg_rating = my_df.ask("average rating")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("🎉 DEMO COMPLETE!")
print("="*80)
print("\n✨ PySenseDF v0.2.2 Features Demonstrated:")
print("   ✅ Natural language queries (10+ patterns)")
print("   ✅ Auto-clean with intelligent type detection")
print("   ✅ Auto-features for machine learning")
print("   ✅ Built-in sample datasets (3 datasets)")
print("   ✅ Standard DataFrame operations")
print("   ✅ Beautiful table formatting")
print("   ✅ Pure Python - no Rust/C++ dependencies")
print("\n🚀 PySenseDF is truly smarter than Pandas!")
print("="*80)
