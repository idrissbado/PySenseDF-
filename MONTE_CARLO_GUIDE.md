# 🎲 Monte Carlo Simulation Guide - PySenseDF v0.4.0

Complete guide to Monte Carlo simulation, risk analysis, and financial modeling in PySenseDF.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Monte Carlo Methods](#monte-carlo-methods)
3. [Risk Analysis](#risk-analysis)
4. [Portfolio Simulation](#portfolio-simulation)
5. [Scenario Analysis](#scenario-analysis)
6. [Stress Testing](#stress-testing)
7. [Sensitivity Analysis](#sensitivity-analysis)
8. [PipelineScript Integration](#pipelinescript-integration)
9. [Real-World Examples](#real-world-examples)

---

## 🚀 Quick Start

### Basic Monte Carlo Simulation

```python
from pysensedf import DataFrame

# Load stock price data
df = DataFrame.from_csv('stock_prices.csv')

# Run 10,000 simulations
results = df.monte_carlo(
    value_column='close_price',
    n_simulations=10000,
    time_periods=252,  # 1 year of trading days
    method='geometric_brownian'
)

# View results
print(f"Initial Value: ${results['statistics']['initial_value']:.2f}")
print(f"Expected Value (1 year): ${results['statistics']['mean_final']:.2f}")
print(f"95% Confidence Interval: ${results['percentiles'][5][-1]:.2f} - ${results['percentiles'][95][-1]:.2f}")
print(f"95% VaR: ${results['var'][0.95]:.2f}")
print(f"Probability of Profit: {results['statistics']['probability_positive']:.1%}")
```

**Output:**
```
Initial Value: $150.00
Expected Value (1 year): $165.30
95% Confidence Interval: $98.50 - $275.80
95% VaR: $45.20
Probability of Profit: 67.3%
```

---

## 🎲 Monte Carlo Methods

### 1. Geometric Brownian Motion (GBM)

**Best for:** Stock prices, asset values, anything that can't go negative

```python
results = df.monte_carlo(
    'stock_price',
    n_simulations=10000,
    method='geometric_brownian',  # Default method
    time_periods=252
)

# GBM formula: dS = μS dt + σS dW
# - μ: drift (mean return)
# - σ: volatility (standard deviation)
# - dW: Wiener process (random walk)
```

**Use cases:**
- Stock price forecasting
- Real estate value projections
- Cryptocurrency predictions
- Asset allocation

### 2. Arithmetic Brownian Motion

**Best for:** Values that can go negative (interest rates, temperatures)

```python
results = df.monte_carlo(
    'interest_rate',
    method='arithmetic',
    n_simulations=10000
)

# ABM formula: dS = μ dt + σ dW
# Allows negative values
```

**Use cases:**
- Interest rate modeling
- Temperature forecasting
- Commodity price spreads
- Economic indicators

### 3. Jump Diffusion Model

**Best for:** Assets with sudden jumps (earnings announcements, market crashes)

```python
results = df.monte_carlo(
    'stock_price',
    method='jump_diffusion',
    n_simulations=10000
)

# Merton Jump Diffusion: GBM + Poisson jumps
# Captures sudden market events
```

**Use cases:**
- Options pricing
- Crypto volatility modeling
- Event-driven trading
- Credit risk

### 4. Historical Simulation

**Best for:** When you want to preserve actual historical patterns

```python
results = df.monte_carlo(
    'returns',
    method='historical',
    n_simulations=10000
)

# Bootstrap from actual historical returns
# No distributional assumptions
```

**Use cases:**
- Backtesting strategies
- Non-normal distributions
- Fat-tail risk assessment
- Stress testing

---

## 📊 Risk Analysis

### Value at Risk (VaR)

Measures the maximum expected loss at a given confidence level.

```python
results = df.monte_carlo(
    'portfolio_value',
    n_simulations=10000,
    confidence_levels=[0.90, 0.95, 0.99]
)

# VaR interpretation:
# "95% chance we won't lose more than $X"
print(f"1-day 95% VaR: ${results['var'][0.95]:,.2f}")
print(f"1-day 99% VaR: ${results['var'][0.99]:,.2f}")

# Annual VaR (scaling rule)
annual_var_95 = results['var'][0.95] * (252 ** 0.5)
print(f"Annual 95% VaR: ${annual_var_95:,.2f}")
```

### Conditional Value at Risk (CVaR / Expected Shortfall)

Average loss in worst-case scenarios beyond VaR.

```python
# CVaR is already calculated
print(f"95% CVaR: ${results['cvar'][0.95]:,.2f}")
print(f"Expected loss if we breach VaR: ${results['cvar'][0.95] - results['var'][0.95]:,.2f}")

# CVaR > VaR indicates tail risk
```

### Risk Metrics

```python
stats = results['statistics']

print(f"Expected Return: {(stats['mean_final'] / stats['initial_value'] - 1) * 100:.2f}%")
print(f"Volatility (Std Dev): ${stats['std_final']:,.2f}")
print(f"Downside Risk: ${stats['expected_loss']:,.2f}")
print(f"Upside Potential: ${stats['expected_gain']:,.2f}")
print(f"Risk/Reward Ratio: {stats['expected_gain'] / stats['expected_loss']:.2f}")
```

---

## 💼 Portfolio Simulation

### Multi-Asset Portfolio

```python
# Simulate portfolio with 3 assets
results = df.portfolio_monte_carlo(
    asset_columns=['stock_a', 'stock_b', 'bond_c'],
    weights=[0.50, 0.30, 0.20],  # 50% stock A, 30% stock B, 20% bonds
    n_simulations=10000,
    time_periods=252
)

# Portfolio statistics
stats = results['statistics']
print(f"Initial Portfolio Value: ${stats['initial_value']:,.2f}")
print(f"Expected Value (1 year): ${stats['mean_final']:,.2f}")
print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
print(f"95% VaR: ${stats['var_95']:,.2f}")
print(f"99% VaR: ${stats['var_99']:,.2f}")
```

### Optimal Allocation (Efficient Frontier)

```python
# Test different weight combinations
allocations = [
    [0.6, 0.3, 0.1],
    [0.5, 0.3, 0.2],
    [0.4, 0.4, 0.2],
    [0.3, 0.5, 0.2],
]

best_sharpe = -999
best_allocation = None

for weights in allocations:
    results = df.portfolio_monte_carlo(
        ['stock_a', 'stock_b', 'bond_c'],
        weights=weights,
        n_simulations=5000
    )
    
    sharpe = results['statistics']['sharpe_ratio']
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_allocation = weights

print(f"Optimal Allocation: {best_allocation}")
print(f"Sharpe Ratio: {best_sharpe:.3f}")
```

---

## 🎯 Scenario Analysis

Compare specific economic scenarios.

```python
scenarios = {
    'bull_market': {
        'mean': 0.15,        # 15% annual return
        'std': 0.10,         # 10% volatility
        'simulations': 5000
    },
    'bear_market': {
        'mean': -0.10,       # -10% annual return
        'std': 0.25,         # 25% volatility
        'simulations': 5000
    },
    'base_case': {
        'mean': 0.07,        # 7% annual return
        'std': 0.15,         # 15% volatility
        'simulations': 5000
    },
    'recession': {
        'mean': -0.20,       # -20% annual return
        'std': 0.35,         # 35% volatility
        'simulations': 5000
    },
    'stagflation': {
        'mean': 0.02,        # 2% return (high inflation, low growth)
        'std': 0.20,         # 20% volatility
        'simulations': 5000
    }
}

results = df.scenario_analysis('portfolio_value', scenarios, time_periods=252)

# Compare scenarios
for scenario_name, scenario_results in results.items():
    print(f"\n{scenario_name.upper()}:")
    print(f"  Expected Return: {scenario_results['expected_return']:.1%}")
    print(f"  Probability of Profit: {scenario_results['probability_positive']:.1%}")
    print(f"  5th Percentile: ${scenario_results['percentile_5']:,.2f}")
    print(f"  95th Percentile: ${scenario_results['percentile_95']:,.2f}")
```

**Output:**
```
BULL_MARKET:
  Expected Return: 15.2%
  Probability of Profit: 85.3%
  5th Percentile: $145,230
  95th Percentile: $295,840

BEAR_MARKET:
  Expected Return: -9.8%
  Probability of Profit: 38.2%
  5th Percentile: $65,420
  95th Percentile: $175,630
...
```

---

## ⚠️ Stress Testing

Test portfolio resilience under extreme conditions.

```python
stress_scenarios = [
    {
        'name': '2008 Financial Crisis',
        'shock': -0.50,              # 50% immediate drop
        'volatility_multiplier': 3   # 3x normal volatility
    },
    {
        'name': '1987 Black Monday',
        'shock': -0.25,              # 25% immediate drop
        'volatility_multiplier': 4   # 4x normal volatility
    },
    {
        'name': 'COVID-19 Crash (March 2020)',
        'shock': -0.35,              # 35% immediate drop
        'volatility_multiplier': 5   # 5x normal volatility
    },
    {
        'name': 'Dot-com Bubble Burst',
        'shock': -0.75,              # 75% immediate drop
        'volatility_multiplier': 2   # 2x normal volatility
    },
    {
        'name': 'Black Swan Event',
        'shock': -0.90,              # 90% immediate drop
        'volatility_multiplier': 10  # 10x normal volatility
    }
]

results = df.stress_test('portfolio_value', stress_scenarios, n_simulations=5000)

# Analyze results
for scenario_name, scenario_results in results.items():
    print(f"\n{scenario_name}:")
    print(f"  Initial Shock: {scenario_results['initial_shock']:.1f}%")
    print(f"  Shocked Value: ${scenario_results['shocked_value']:,.2f}")
    print(f"  Mean Recovery (30 days): ${scenario_results['mean_recovery']:,.2f}")
    print(f"  Probability of Full Recovery: {scenario_results['probability_full_recovery']:.1%}")
    print(f"  Worst Case: ${scenario_results['worst_case']:,.2f}")
    print(f"  Expected Loss: ${scenario_results['expected_loss']:,.2f}")
```

**Output:**
```
2008 Financial Crisis:
  Initial Shock: -50.0%
  Shocked Value: $500,000
  Mean Recovery (30 days): $525,000
  Probability of Full Recovery: 15.3%
  Worst Case: $325,000
  Expected Loss: $475,000
```

---

## 🔬 Sensitivity Analysis

Test how parameters affect outcomes.

```python
parameter_ranges = {
    'mean': [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20],
    'std': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
}

base_params = {
    'mean': 0.10,  # Base case: 10% return
    'std': 0.15    # Base case: 15% volatility
}

results = df.sensitivity_analysis(
    'portfolio_value',
    parameter_ranges,
    base_params,
    n_simulations=1000,
    time_periods=252
)

# Plot sensitivity to mean return
print("\nSENSITIVITY TO MEAN RETURN:")
for result in results['mean']:
    param_value = result['parameter_value']
    mean_final = result['mean_final']
    prob_positive = result['probability_positive']
    print(f"  Return {param_value:+.0%}: Mean=${mean_final:,.2f}, P(profit)={prob_positive:.1%}")

# Plot sensitivity to volatility
print("\nSENSITIVITY TO VOLATILITY:")
for result in results['std']:
    param_value = result['parameter_value']
    mean_final = result['mean_final']
    std_final = result['std_final']
    print(f"  Vol {param_value:.0%}: Mean=${mean_final:,.2f}, StdDev=${std_final:,.2f}")
```

---

## 🔗 PipelineScript Integration

Combine Monte Carlo simulation with ML pipelines.

### Installation

```bash
pip install pipelinescript
```

### Quick ML Pipeline

```python
from pysensedf import DataFrame
from pysensedf.integrations.pipelinescript_integration import quick_ml_pipeline

# Load data
df = DataFrame.from_csv('data.csv')

# Run complete ML pipeline in one line
results = quick_ml_pipeline(
    df,
    target='price',
    model='xgboost',
    task='regression'
)

print(f"R² Score: {results['metrics']['r2']:.4f}")
print(f"RMSE: {results['metrics']['rmse']:.2f}")
```

### Monte Carlo + ML Pipeline

```python
from pysensedf.integrations.pipelinescript_integration import monte_carlo_pipeline

# Run Monte Carlo simulation + train ML model
results = monte_carlo_pipeline(
    df,
    value_column='stock_price',
    pipeline_script='''
    clean missing
    encode
    split 80/20 --target future_return
    train xgboost
    evaluate
    ''',
    n_simulations=5000,
    time_periods=252
)

# Monte Carlo results
mc_stats = results['monte_carlo']['statistics']
print(f"Expected Price: ${mc_stats['mean_final']:.2f}")

# ML Pipeline results
ml_results = results['pipeline']
print(f"ML Model Accuracy: {ml_results['metrics']['accuracy']:.2%}")
```

### PipelineScript DSL

```python
# Execute PipelineScript commands directly
result, df_output = df.execute_psl('''
    clean missing
    clean outliers
    encode
    scale
    split 75/25 --target price
    train xgboost
    evaluate
    export model.pkl
''', target='price')

if result.success:
    print(f"Pipeline completed in {result.duration:.2f}s")
    print(f"Metrics: {result.context.metrics}")
```

---

## 💡 Real-World Examples

### Example 1: Stock Portfolio Optimization

```python
from pysensedf import DataFrame

# Load historical prices
df = DataFrame.from_csv('portfolio.csv')

# Test different allocations
best_sharpe = -999
best_weights = None

for stock_weight in [0.3, 0.4, 0.5, 0.6, 0.7]:
    bond_weight = 1 - stock_weight
    
    results = df.portfolio_monte_carlo(
        ['stocks', 'bonds'],
        weights=[stock_weight, bond_weight],
        n_simulations=10000,
        time_periods=252
    )
    
    sharpe = results['statistics']['sharpe_ratio']
    var_95 = results['statistics']['var_95']
    
    print(f"Allocation {stock_weight:.0%}/{bond_weight:.0%}: Sharpe={sharpe:.3f}, VaR=${var_95:,.2f}")
    
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_weights = [stock_weight, bond_weight]

print(f"\nOptimal Allocation: {best_weights[0]:.0%} stocks, {best_weights[1]:.0%} bonds")
print(f"Sharpe Ratio: {best_sharpe:.3f}")
```

### Example 2: Real Estate Investment Risk

```python
# Load property values
df = DataFrame.from_csv('real_estate.csv')

# Run scenarios
scenarios = {
    'boom': {'mean': 0.12, 'std': 0.08},
    'normal': {'mean': 0.05, 'std': 0.10},
    'bust': {'mean': -0.08, 'std': 0.15}
}

results = df.scenario_analysis('property_value', scenarios, time_periods=120)  # 10 years

for scenario, data in results.items():
    print(f"\n{scenario.upper()} SCENARIO:")
    print(f"  5-year value: ${data['percentile_50']:,.2f}")
    print(f"  Probability of profit: {data['probability_positive']:.1%}")
```

### Example 3: Cryptocurrency Volatility

```python
# Load Bitcoin prices
df = DataFrame.from_csv('bitcoin.csv')

# Jump diffusion model (crypto has sudden jumps)
results = df.monte_carlo(
    'btc_price',
    method='jump_diffusion',
    n_simulations=10000,
    time_periods=365
)

# Calculate risk metrics
stats = results['statistics']
var_95 = results['var'][0.95]
cvar_95 = results['cvar'][0.95]

print(f"Current Price: ${stats['initial_value']:,.2f}")
print(f"1-year Expected: ${stats['mean_final']:,.2f}")
print(f"95% VaR: ${var_95:,.2f}")
print(f"95% CVaR: ${cvar_95:,.2f}")
print(f"Probability of 2x: {sum(1 for v in results['final_values'] if v >= stats['initial_value']*2) / len(results['final_values']):.1%}")
```

### Example 4: Options Pricing

```python
# Black-Scholes Monte Carlo
df = DataFrame.from_csv('stock.csv')

strike_price = 100
results = df.monte_carlo(
    'stock_price',
    method='geometric_brownian',
    n_simulations=100000,
    time_periods=252
)

# Calculate call option value
call_payoffs = [max(0, final_price - strike_price) for final_price in results['final_values']]
call_value = sum(call_payoffs) / len(call_payoffs)

print(f"Call Option Value (Strike ${strike_price}): ${call_value:.2f}")

# Put option value
put_payoffs = [max(0, strike_price - final_price) for final_price in results['final_values']]
put_value = sum(put_payoffs) / len(put_payoffs)

print(f"Put Option Value (Strike ${strike_price}): ${put_value:.2f}")
```

---

## 📚 Best Practices

### 1. Choose Appropriate Simulation Count

```python
# Small datasets, quick tests
n_simulations=1000

# Standard analysis
n_simulations=10000  # Recommended

# High-precision (options pricing, VaR)
n_simulations=100000

# Extreme precision (academic research)
n_simulations=1000000
```

### 2. Select Right Time Horizon

```python
# Day trading
time_periods=1

# Short-term (1 month)
time_periods=21

# Medium-term (1 quarter)
time_periods=63

# Annual (1 year)
time_periods=252  # Trading days

# Long-term (5 years)
time_periods=1260
```

### 3. Model Selection

| Asset Type | Best Method | Reason |
|------------|-------------|--------|
| **Stocks** | Geometric Brownian | Prevents negative prices |
| **Bonds** | Arithmetic | Can model negative yields |
| **Crypto** | Jump Diffusion | Captures sudden moves |
| **Options** | Geometric Brownian | Industry standard |
| **Interest Rates** | Arithmetic | Can go negative |
| **Commodities** | Historical | Non-normal returns |

### 4. Interpreting Results

```python
results = df.monte_carlo('price', n_simulations=10000)
stats = results['statistics']

# Good signal: High Sharpe, high probability of profit
if stats['sharpe_ratio'] > 1.0 and stats['probability_positive'] > 0.7:
    print("✅ Attractive risk/reward")

# Warning signal: High VaR relative to expected gain
if results['var'][0.95] > stats['expected_gain']:
    print("⚠️ High downside risk")

# Red flag: Low probability of profit
if stats['probability_positive'] < 0.5:
    print("❌ More likely to lose than gain")
```

---

## 🔧 Performance Tips

### 1. Use NumPy Backend (27-92x faster!)

```python
# Enable NumPy for large simulations
df = DataFrame(data, backend='numpy')

results = df.monte_carlo('price', n_simulations=100000)
# With NumPy: ~2 seconds
# Without NumPy: ~180 seconds
```

### 2. Enable Caching

```python
# Cache results for repeated analysis
df = DataFrame(data, enable_cache=True)

results1 = df.monte_carlo('price', n_simulations=10000)  # 5 seconds
results2 = df.monte_carlo('price', n_simulations=10000)  # 0.001 seconds (cached!)
```

### 3. Parallel Processing

```python
# Use all CPU cores
df = DataFrame(data, n_jobs=-1)

results = df.monte_carlo('price', n_simulations=50000)
# Uses all 8 cores → 8x faster
```

---

## 📖 Additional Resources

- [PySenseDF Documentation](https://github.com/idrissbado/PySenseDF)
- [PipelineScript Documentation](https://github.com/idrissbado/PipelineScript)
- [Monte Carlo Methods in Financial Engineering](https://www.springer.com/gp/book/9780387004518)
- [Risk Management and Simulation](https://www.amazon.com/dp/0470371897)

---

## 🤝 Contributing

Have ideas for new Monte Carlo features? Open an issue or PR!

---

**Built with ❤️ by [Idriss Bado](https://github.com/idrissbado)**

*Making quantitative finance accessible to everyone.*
