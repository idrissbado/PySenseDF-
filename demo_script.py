"""
🚀 PySenseDF v0.3.0 - Quick Demo Script
Run this in any Python environment!

Install: pip install pysensedf==0.3.0
"""

from pysensedf import DataFrame
from pysensedf.datasets import load_customers

print("="*70)
print("🚀 PySenseDF v0.3.0 - Revolutionary Features Demo")
print("="*70)

# Load data
df = load_customers()
print(f"\n✅ Loaded {len(df)} customers")
print(f"📊 Columns: {len(df.columns())}")

# 1. Magic Methods
print("\n" + "="*70)
print("1️⃣  MAGIC METHODS (Pandas-like Syntax)")
print("="*70)

# Get column
ages = df['age']
print(f"\n📌 df['age'] returns: {type(ages)}")
print(f"   First 5: {ages.to_list()[:5]}")

# Boolean filtering
high_income = df[df['income'] > 90000]
print(f"\n📌 df[df['income'] > 90000]:")
print(f"   Found {len(high_income)} high-income customers")

# Add new column
df['age_group'] = ['Young' if int(age) < 35 else 'Senior' for age in df['age']]
print(f"\n📌 Added 'age_group' column")
print(f"   Total columns now: {len(df.columns())}")

# 2. Statistics
print("\n" + "="*70)
print("2️⃣  STATISTICAL METHODS (Pure Python - No NumPy!)")
print("="*70)

print(f"\n📊 Average Age: {df.mean('age'):.1f} years")
print(f"📊 Median Income: ${df.median('income'):,.0f}")
print(f"📊 Std Dev (Revenue): ${df.std('revenue'):,.2f}")

print("\n📊 Summary Statistics:")
print(df.describe())

print("\n🔗 Correlation Matrix:")
corr = df.corr()
print(corr)

# 3. Data Cleaning
print("\n" + "="*70)
print("3️⃣  DATA CLEANING")
print("="*70)

df_clean = df.dropna()
print(f"\n🧹 After dropna(): {len(df_clean)} rows (no missing values)")

df_unique = df.drop_duplicates()
print(f"🧹 After drop_duplicates(): {len(df_unique)} unique rows")

# 4. Advanced Operations
print("\n" + "="*70)
print("4️⃣  ADVANCED OPERATIONS")
print("="*70)

# Merge
city_data = DataFrame({
    'city': ['New York', 'Los Angeles', 'Chicago'],
    'timezone': ['EST', 'PST', 'CST']
})

merged = df.merge(city_data, on='city', how='left')
print(f"\n🔗 Merged with city data: {len(merged.columns())} columns")

# Method chaining
top5 = df.pipe(lambda x: x.sort('revenue', descending=True)).head(5)
print(f"\n⛓️  Method chaining: Top 5 by revenue")
print(top5.select(['name', 'revenue', 'city']))

# 5. Smart AI Features
print("\n" + "="*70)
print("5️⃣  SMART AI FEATURES (Beyond Pandas!)")
print("="*70)

# AI Summary
print("\n🤖 AI Summary:")
print(df.ai_summarize())

# Anomaly Detection
print("\n🔍 Anomaly Detection (IQR Method):")
outliers = df.detect_anomalies('revenue', method='iqr')
print(f"   Found {len(outliers)} outliers")

# Smart Suggestions
print("\n💡 Smart Suggestions (first 3 columns):")
suggestions = df.suggest_transformations()
for col, tips in list(suggestions.items())[:3]:
    status = tips if tips else ['✅ Looks good!']
    print(f"   {col}: {status[0]}")

# Visualization Recommendations
print("\n📈 Visualization Recommendation:")
print(df.auto_visualize('income'))

# 6. Natural Language
print("\n" + "="*70)
print("6️⃣  NATURAL LANGUAGE QUERIES")
print("="*70)

result = df.ask("show me customers from New York")
print(f"\n💬 Asked: 'show me customers from New York'")
print(f"   Result: {len(result)} customers")
print(result.select(['name', 'city', 'income']).head(3))

# Summary
print("\n" + "="*70)
print("🎉 DEMO COMPLETE!")
print("="*70)
print("""
✅ PySenseDF v0.3.0 Features Demonstrated:
   • Magic methods (df['col'], boolean indexing)
   • Statistical methods (describe, mean, median, std, corr)
   • Data cleaning (dropna, drop_duplicates)
   • Advanced operations (merge, pipe)
   • Smart AI features (anomalies, suggestions, summaries, viz)
   • Natural language queries

🚀 Why PySenseDF Beats Pandas:
   • Pure Python (no numpy/pandas dependencies)
   • More efficient (no overhead)
   • AI-powered insights
   • Natural language queries
   • Same familiar syntax

📦 Install: pip install pysensedf==0.3.0
🔗 PyPI: https://pypi.org/project/pysensedf/0.3.0/
""")
