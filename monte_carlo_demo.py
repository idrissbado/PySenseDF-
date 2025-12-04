"""
PySenseDF v0.4.0 - Monte Carlo Simulation Demo
===============================================

Quick demonstration of Monte Carlo simulation features.
"""

from pysensedf import DataFrame
import random

# Create sample stock price data
print("📊 Creating sample stock price data...")
dates = [f"2024-{i//30+1:02d}-{i%30+1:02d}" for i in range(252)]
prices = [100.0]
for _ in range(251):
    change = random.gauss(0.001, 0.02)  # 0.1% drift, 2% volatility
    prices.append(prices[-1] * (1 + change))

df = DataFrame({
    'date': dates,
    'stock_price': prices
})

print(f"✅ Created {len(prices)} days of stock data")
print(f"Initial Price: ${prices[0]:.2f}")
print(f"Final Price: ${prices[-1]:.2f}")
print(f"Return: {(prices[-1]/prices[0] - 1)*100:.2f}%\n")

# ============================================================================
# 1. BASIC MONTE CARLO SIMULATION
# ============================================================================

print("=" * 70)
print("1️⃣  BASIC MONTE CARLO SIMULATION (Geometric Brownian Motion)")
print("=" * 70)

results = df.monte_carlo(
    'stock_price',
    n_simulations=10000,
    time_periods=252,
    method='geometric_brownian',
    confidence_levels=[0.95, 0.99]
)

stats = results['statistics']
print(f"\n📈 Simulation Results (1 year, 10,000 paths):")
print(f"   Initial Value: ${stats['initial_value']:.2f}")
print(f"   Expected Value: ${stats['mean_final']:.2f}")
print(f"   Median Value: ${stats['median_final']:.2f}")
print(f"   Std Deviation: ${stats['std_final']:.2f}")
print(f"   Min Value: ${stats['min_final']:.2f}")
print(f"   Max Value: ${stats['max_final']:.2f}")

print(f"\n📊 Confidence Intervals:")
print(f"   5th Percentile: ${results['percentiles'][5][-1]:.2f}")
print(f"   25th Percentile: ${results['percentiles'][25][-1]:.2f}")
print(f"   50th Percentile: ${results['percentiles'][50][-1]:.2f}")
print(f"   75th Percentile: ${results['percentiles'][75][-1]:.2f}")
print(f"   95th Percentile: ${results['percentiles'][95][-1]:.2f}")

print(f"\n⚠️  Risk Metrics:")
print(f"   95% VaR: ${results['var'][0.95]:.2f}")
print(f"   99% VaR: ${results['var'][0.99]:.2f}")
print(f"   95% CVaR: ${results['cvar'][0.95]:.2f}")
print(f"   99% CVaR: ${results['cvar'][0.99]:.2f}")

print(f"\n💰 Probability Analysis:")
print(f"   Probability of Profit: {stats['probability_positive']:.1%}")
print(f"   Expected Gain: ${stats['expected_gain']:.2f}")
print(f"   Expected Loss: ${stats['expected_loss']:.2f}")
print(f"   Risk/Reward Ratio: {stats['expected_gain']/stats['expected_loss']:.2f}x")

# ============================================================================
# 2. SCENARIO ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("2️⃣  SCENARIO ANALYSIS (Bull vs Bear vs Base Case)")
print("=" * 70)

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
    }
}

scenario_results = df.scenario_analysis('stock_price', scenarios, time_periods=252)

print("\n📊 Scenario Comparison:")
for scenario_name, data in scenario_results.items():
    print(f"\n   {scenario_name.upper()}:")
    print(f"      Expected Final Value: ${data['mean_final']:,.2f}")
    print(f"      Expected Return: {data['expected_return']:.1%}")
    print(f"      Probability of Profit: {data['probability_positive']:.1%}")
    print(f"      5th Percentile: ${data['percentile_5']:,.2f}")
    print(f"      95th Percentile: ${data['percentile_95']:,.2f}")

# ============================================================================
# 3. STRESS TESTING
# ============================================================================

print("\n" + "=" * 70)
print("3️⃣  STRESS TESTING (Extreme Market Conditions)")
print("=" * 70)

stress_scenarios = [
    {
        'name': '2008 Financial Crisis',
        'shock': -0.50,
        'volatility_multiplier': 3
    },
    {
        'name': 'Black Swan Event',
        'shock': -0.75,
        'volatility_multiplier': 5
    },
    {
        'name': 'COVID-19 Crash',
        'shock': -0.35,
        'volatility_multiplier': 4
    }
]

stress_results = df.stress_test('stock_price', stress_scenarios, n_simulations=5000)

print("\n⚠️  Stress Test Results (30-day recovery simulation):")
for scenario_name, data in stress_results.items():
    print(f"\n   {scenario_name}:")
    print(f"      Initial Shock: {data['initial_shock']:.1f}%")
    print(f"      Shocked Value: ${data['shocked_value']:,.2f}")
    print(f"      Mean Recovery Value: ${data['mean_recovery']:,.2f}")
    print(f"      Probability of Full Recovery: {data['probability_full_recovery']:.1%}")
    print(f"      Worst Case: ${data['worst_case']:,.2f}")
    print(f"      Expected Loss: ${data['expected_loss']:,.2f}")

# ============================================================================
# 4. PORTFOLIO SIMULATION
# ============================================================================

print("\n" + "=" * 70)
print("4️⃣  PORTFOLIO SIMULATION (Multi-Asset)")
print("=" * 70)

# Create multi-asset data
df_portfolio = DataFrame({
    'stock_a': [100.0 * (1 + random.gauss(0.001, 0.02)) ** i for i in range(252)],
    'stock_b': [100.0 * (1 + random.gauss(0.0005, 0.015)) ** i for i in range(252)],
    'bonds': [100.0 * (1 + random.gauss(0.0002, 0.005)) ** i for i in range(252)]
})

portfolio_results = df_portfolio.portfolio_monte_carlo(
    ['stock_a', 'stock_b', 'bonds'],
    weights=[0.50, 0.30, 0.20],
    n_simulations=10000,
    time_periods=252
)

port_stats = portfolio_results['statistics']
print(f"\n📊 Portfolio Results (50% Stock A, 30% Stock B, 20% Bonds):")
print(f"   Initial Portfolio Value: ${port_stats['initial_value']:,.2f}")
print(f"   Expected Final Value: ${port_stats['mean_final']:,.2f}")
print(f"   Median Final Value: ${port_stats['median_final']:,.2f}")
print(f"   Sharpe Ratio: {port_stats['sharpe_ratio']:.3f}")
print(f"   Probability of Profit: {port_stats['probability_positive']:.1%}")
print(f"   95% VaR: ${port_stats['var_95']:,.2f}")
print(f"   99% VaR: ${port_stats['var_99']:,.2f}")

# ============================================================================
# 5. SENSITIVITY ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("5️⃣  SENSITIVITY ANALYSIS (Parameter Impact)")
print("=" * 70)

parameter_ranges = {
    'mean': [0.0, 0.05, 0.10, 0.15],
    'std': [0.10, 0.15, 0.20, 0.25]
}

base_params = {'mean': 0.10, 'std': 0.15}

sensitivity_results = df.sensitivity_analysis(
    'stock_price',
    parameter_ranges,
    base_params,
    n_simulations=1000,
    time_periods=252
)

print("\n📊 Sensitivity to Mean Return:")
for result in sensitivity_results['mean']:
    print(f"   Return {result['parameter_value']:+.0%}: "
          f"Mean=${result['mean_final']:,.2f}, "
          f"P(profit)={result['probability_positive']:.1%}")

print("\n📊 Sensitivity to Volatility:")
for result in sensitivity_results['std']:
    print(f"   Vol {result['parameter_value']:.0%}: "
          f"Mean=${result['mean_final']:,.2f}, "
          f"StdDev=${result['std_final']:,.2f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("✅ MONTE CARLO SIMULATION DEMO COMPLETE!")
print("=" * 70)
print("\n📚 What we demonstrated:")
print("   1. Basic Monte Carlo simulation (10,000 paths)")
print("   2. Scenario analysis (bull/bear/base case)")
print("   3. Stress testing (2008 crisis, Black Swan)")
print("   4. Portfolio simulation (multi-asset)")
print("   5. Sensitivity analysis (parameter impacts)")

print("\n🚀 Key Features:")
print("   - 4 simulation methods (GBM, ABM, Jump Diffusion, Historical)")
print("   - Risk metrics (VaR, CVaR, Sharpe ratio)")
print("   - Portfolio optimization support")
print("   - Parallel processing for speed")
print("   - NumPy backend (27-92x faster!)")

print("\n📖 Learn More:")
print("   - See MONTE_CARLO_GUIDE.md for full documentation")
print("   - 10+ real-world examples with complete code")
print("   - Integration with PipelineScript for ML workflows")

print("\n💡 Try it yourself:")
print("   pip install pysensedf[full]")
print("   # All features including NumPy backend!")

print("\n🔥 PySenseDF v0.4.0 - Making quantitative finance accessible to everyone!")
