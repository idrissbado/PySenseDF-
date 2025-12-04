# 🚀 PySenseDF v0.3.0 - Examples & Demos

## 📦 Installation

```bash
pip install pysensedf==0.3.0
```

## 📚 Available Examples

### 1. **Jupyter Notebooks** (Recommended for Interactive Exploration)

#### 🎯 **`pysensedf_v03_demo.ipynb`** - Complete Feature Demo
- **40+ code cells** with all v0.3.0 features
- Interactive examples you can modify and run
- Beautiful HTML table displays
- Covers:
  - Magic methods (Pandas-like syntax)
  - Statistical methods (pure Python)
  - Data cleaning operations
  - Advanced operations (merge, pipe)
  - Smart AI features
  - Natural language queries
  - Complete workflow example

**To run:**
```bash
cd C:\Users\DELL\PySenseDF
jupyter notebook pysensedf_v03_demo.ipynb
```

#### ⚡ **`quickstart.ipynb`** - Quick Tour
- **12 focused code cells**
- Perfect for beginners
- Core features demonstration
- Takes 5 minutes to run through

**To run:**
```bash
cd C:\Users\DELL\PySenseDF
jupyter notebook quickstart.ipynb
```

### 2. **Python Script** (No Jupyter Required)

#### 🐍 **`demo_script.py`** - Standalone Demo
- Complete demo in a single Python script
- Run anywhere Python is installed
- No Jupyter required
- Shows all major features

**To run:**
```bash
cd C:\Users\DELL\PySenseDF
python demo_script.py
```

### 3. **Test Suite**

#### ✅ **`test_v03_features.py`** - Feature Tests
- Comprehensive tests for all v0.3.0 features
- Validates functionality
- Great for understanding feature usage

**To run:**
```bash
cd C:\Users\DELL\PySenseDF
python test_v03_features.py
```

---

## 🎯 Quick Start (30 seconds)

```python
from pysensedf import DataFrame
from pysensedf.datasets import load_customers

# Load sample data
df = load_customers()

# Pandas-like syntax
high_earners = df[df['income'] > 90000]

# Pure Python statistics
print(f"Average income: ${df.mean('income'):,.2f}")
print(df.describe())

# AI features
print(df.ai_summarize())
outliers = df.detect_anomalies('revenue')

# Natural language
result = df.ask("show me customers from New York")
```

---

## 🌟 Key Features in v0.3.0

### ✅ **Pandas-Compatible Syntax**
```python
df['age']                    # Get column
df[['name', 'age']]         # Multiple columns
df[df['income'] > 90000]    # Boolean filtering
df['new_col'] = values      # Add column
len(df)                     # Row count
```

### ✅ **Statistical Methods (Pure Python - No NumPy!)**
```python
df.describe()               # Complete summary
df.mean('age')              # Mean
df.median('income')         # Median
df.std('revenue')           # Standard deviation
df.corr()                   # Correlation matrix
df.value_counts('city')     # Frequency table
```

### ✅ **Smart AI Features (Beyond Pandas!)**
```python
df.ai_summarize()                      # Natural language summary
df.detect_anomalies('revenue')         # Find outliers
df.suggest_transformations()           # Data quality tips
df.auto_visualize('income')            # Viz recommendations
```

### ✅ **Advanced Operations**
```python
df.merge(df2, on='city', how='left')   # SQL-style join
df.pipe(lambda x: x.sort(...))         # Method chaining
df.dropna()                            # Remove missing values
df.drop_duplicates()                   # Remove duplicates
```

### ✅ **Natural Language Queries**
```python
df.ask("show me customers from New York")
df.ask("what is the average income by city?")
```

---

## 🎯 Which Example Should I Use?

| Your Need | Recommended Example |
|-----------|-------------------|
| **Interactive exploration** | `pysensedf_v03_demo.ipynb` |
| **Quick overview** | `quickstart.ipynb` |
| **No Jupyter available** | `demo_script.py` |
| **Learn feature details** | `test_v03_features.py` |
| **Just starting out** | `quickstart.ipynb` |

---

## 💡 Why PySenseDF v0.3.0?

| Feature | Pandas | PySenseDF v0.3.0 |
|---------|--------|------------------|
| Pandas-like syntax | ✅ | ✅ |
| Statistical methods | ✅ | ✅ (Pure Python!) |
| Dependencies | NumPy, many | **NONE!** |
| AI anomaly detection | ❌ | ✅ |
| AI suggestions | ❌ | ✅ |
| Natural language queries | ❌ | ✅ |
| Auto-clean & features | ❌ | ✅ |

**Advantages:**
- 🔹 **No dependencies** - Pure Python stdlib only
- 🔹 **More efficient** - No NumPy overhead
- 🔹 **AI-powered** - Smart features Pandas lacks
- 🔹 **Lighter** - Smaller installation size
- 🔹 **Same syntax** - Familiar Pandas-like API

---

## 📖 Documentation

- **PyPI Package**: https://pypi.org/project/pysensedf/0.3.0/
- **Full API**: See notebooks for detailed examples

---

## 🤝 Support

If you encounter any issues:
1. Check the notebooks for working examples
2. Run `test_v03_features.py` to validate installation
3. Review the demo output for expected behavior

---

## 🌟 PySenseDF v0.3.0 - The DataFrame that truly beats Pandas!

**Install now:**
```bash
pip install pysensedf==0.3.0
```
