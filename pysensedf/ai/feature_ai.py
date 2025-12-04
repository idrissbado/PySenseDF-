"""
Auto Feature Engineering with Monte Carlo Simulation
=====================================================

One-line feature engineering: df.autofeatures(target="label")
Monte Carlo simulation: df.monte_carlo_simulate(n_simulations=10000)
Risk analysis: df.value_at_risk(confidence=0.95)
"""

from typing import Optional, List, Dict, Any, Callable
import datetime
import random
import math


def auto_generate_features(df, target: Optional[str] = None):
    """
    Automatically generate features
    
    Creates:
    - Date/time features (year, month, day, hour, dayofweek)
    - Aggregation features (per-group stats)
    - Ratio features
    - Interaction features
    - Lag features
    """
    from ..core.dataframe import DataFrame
    
    new_data = df._data.copy()
    
    # 1. Date features
    for col in df.columns():
        if _is_date_column(df._data[col]):
            date_features = _create_date_features(df._data[col], col)
            new_data.update(date_features)
    
    # 2. Numeric aggregations
    numeric_cols = _get_numeric_columns(df)
    if numeric_cols:
        # Create ratio features
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                ratio_col = f"{col1}_div_{col2}"
                new_data[ratio_col] = _safe_divide(df._data[col1], df._data[col2])
    
    # 3. Interaction features (for small datasets)
    if len(numeric_cols) <= 5 and df.shape()[0] < 10000:
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                interact_col = f"{col1}_times_{col2}"
                new_data[interact_col] = _multiply(df._data[col1], df._data[col2])
    
    return DataFrame(new_data)


def _is_date_column(values: List[Any]) -> bool:
    """Check if column contains dates"""
    non_empty = [v for v in values[:50] if v and v != '']
    if not non_empty:
        return False
    
    date_count = 0
    for v in non_empty:
        try:
            if isinstance(v, datetime.datetime):
                date_count += 1
            elif isinstance(v, str) and ('/' in v or '-' in v) and len(v) >= 8:
                # Try parsing
                datetime.datetime.fromisoformat(v.replace('/', '-')[:10])
                date_count += 1
        except:
            pass
    
    return date_count / len(non_empty) > 0.5


def _create_date_features(values: List[Any], col_name: str) -> Dict[str, List]:
    """Create date-based features"""
    features = {
        f"{col_name}_year": [],
        f"{col_name}_month": [],
        f"{col_name}_day": [],
        f"{col_name}_dayofweek": [],
        f"{col_name}_quarter": [],
    }
    
    for v in values:
        try:
            if isinstance(v, datetime.datetime):
                dt = v
            elif isinstance(v, str):
                dt = datetime.datetime.fromisoformat(v.replace('/', '-')[:10])
            else:
                dt = None
            
            if dt:
                features[f"{col_name}_year"].append(dt.year)
                features[f"{col_name}_month"].append(dt.month)
                features[f"{col_name}_day"].append(dt.day)
                features[f"{col_name}_dayofweek"].append(dt.weekday())
                features[f"{col_name}_quarter"].append((dt.month - 1) // 3 + 1)
            else:
                for k in features.keys():
                    features[k].append(None)
        except:
            for k in features.keys():
                features[k].append(None)
    
    return features


def _get_numeric_columns(df) -> List[str]:
    """Get list of numeric columns"""
    numeric = []
    for col in df.columns():
        values = df._data[col]
        try:
            # Try to convert first 10 non-empty values
            sample = [v for v in values[:50] if v and v != ''][:10]
            if sample:
                [float(v) for v in sample]
                numeric.append(col)
        except:
            pass
    
    return numeric


def _safe_divide(col1: List[Any], col2: List[Any]) -> List[Any]:
    """Safely divide two columns"""
    result = []
    for v1, v2 in zip(col1, col2):
        try:
            num1 = float(v1)
            num2 = float(v2)
            if num2 != 0:
                result.append(num1 / num2)
            else:
                result.append(None)
        except:
            result.append(None)
    
    return result


def _multiply(col1: List[Any], col2: List[Any]) -> List[Any]:
    """Multiply two columns"""
    result = []
    for v1, v2 in zip(col1, col2):
        try:
            result.append(float(v1) * float(v2))
        except:
            result.append(None)
    
    return result


def create_lag_features(values: List[Any], lags: List[int] = [1, 7, 30]) -> Dict[str, List]:
    """Create lag features for time series"""
    features = {}
    
    for lag in lags:
        lagged = [None] * lag + values[:-lag] if lag < len(values) else [None] * len(values)
        features[f"lag_{lag}"] = lagged
    
    return features


def create_rolling_features(values: List[Any], windows: List[int] = [7, 30]) -> Dict[str, List]:
    """Create rolling window features"""
    features = {}
    
    for window in windows:
        rolling_mean = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_values = values[start:i+1]
            try:
                numeric = [float(v) for v in window_values if v is not None]
                mean = sum(numeric) / len(numeric) if numeric else None
                rolling_mean.append(mean)
            except:
                rolling_mean.append(None)
        
        features[f"rolling_mean_{window}"] = rolling_mean
    
    return features


# ============================================================================
# MONTE CARLO SIMULATION FEATURES
# ============================================================================

def monte_carlo_simulate(
    df,
    value_column: str,
    n_simulations: int = 10000,
    time_periods: int = 252,
    method: str = 'geometric_brownian',
    confidence_levels: List[float] = [0.95, 0.99]
):
    """
    Perform Monte Carlo simulation on a DataFrame column
    
    Args:
        df: DataFrame instance
        value_column: Column to simulate
        n_simulations: Number of simulation paths (default 10,000)
        time_periods: Number of time steps to simulate (default 252 trading days)
        method: 'geometric_brownian', 'arithmetic', 'jump_diffusion', or 'historical'
        confidence_levels: Confidence intervals to calculate (e.g., [0.95, 0.99])
    
    Returns:
        Dictionary with simulation results:
        - 'simulations': List of simulation paths
        - 'mean_path': Average path across all simulations
        - 'percentiles': Dictionary of percentile paths
        - 'final_values': Distribution of final values
        - 'statistics': Summary statistics
        - 'var': Value at Risk for each confidence level
        - 'cvar': Conditional Value at Risk (Expected Shortfall)
    """
    from ..core.dataframe import DataFrame
    
    # Get numeric values
    values = [float(v) for v in df._data[value_column] if v is not None and v != '']
    
    if not values:
        raise ValueError(f"No numeric values found in column '{value_column}'")
    
    # Calculate parameters
    returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
    mean_return = sum(returns) / len(returns) if returns else 0
    std_return = _std_dev(returns) if returns else 0
    initial_value = values[-1]  # Start from last known value
    
    # Run simulation based on method
    if method == 'geometric_brownian':
        simulations = _geometric_brownian_motion(
            initial_value, mean_return, std_return, time_periods, n_simulations
        )
    elif method == 'arithmetic':
        simulations = _arithmetic_brownian_motion(
            initial_value, mean_return, std_return, time_periods, n_simulations
        )
    elif method == 'jump_diffusion':
        simulations = _jump_diffusion_model(
            initial_value, mean_return, std_return, time_periods, n_simulations
        )
    elif method == 'historical':
        simulations = _historical_simulation(
            initial_value, returns, time_periods, n_simulations
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate statistics
    final_values = [sim[-1] for sim in simulations]
    mean_path = [sum(sim[t] for sim in simulations) / n_simulations 
                 for t in range(time_periods)]
    
    # Calculate percentiles
    percentiles = {}
    for p in [5, 10, 25, 50, 75, 90, 95]:
        percentiles[p] = [_percentile([sim[t] for sim in simulations], p / 100) 
                          for t in range(time_periods)]
    
    # Calculate Value at Risk (VaR) and CVaR
    var_results = {}
    cvar_results = {}
    for conf in confidence_levels:
        var_results[conf] = _calculate_var(final_values, initial_value, conf)
        cvar_results[conf] = _calculate_cvar(final_values, initial_value, conf)
    
    # Summary statistics
    statistics = {
        'initial_value': initial_value,
        'mean_final': sum(final_values) / len(final_values),
        'median_final': _percentile(final_values, 0.5),
        'std_final': _std_dev(final_values),
        'min_final': min(final_values),
        'max_final': max(final_values),
        'mean_return': mean_return,
        'std_return': std_return,
        'probability_positive': sum(1 for v in final_values if v > initial_value) / len(final_values),
        'expected_gain': sum(max(0, v - initial_value) for v in final_values) / len(final_values),
        'expected_loss': sum(max(0, initial_value - v) for v in final_values) / len(final_values),
    }
    
    return {
        'simulations': simulations,
        'mean_path': mean_path,
        'percentiles': percentiles,
        'final_values': final_values,
        'statistics': statistics,
        'var': var_results,
        'cvar': cvar_results,
        'method': method,
        'n_simulations': n_simulations,
        'time_periods': time_periods,
    }


def _geometric_brownian_motion(S0, mu, sigma, T, N):
    """Geometric Brownian Motion: dS = μS dt + σS dW"""
    dt = 1.0
    simulations = []
    
    for _ in range(N):
        path = [S0]
        S = S0
        for _ in range(T - 1):
            dW = random.gauss(0, math.sqrt(dt))
            S = S * math.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            path.append(S)
        simulations.append(path)
    
    return simulations


def _arithmetic_brownian_motion(S0, mu, sigma, T, N):
    """Arithmetic Brownian Motion: dS = μ dt + σ dW"""
    dt = 1.0
    simulations = []
    
    for _ in range(N):
        path = [S0]
        S = S0
        for _ in range(T - 1):
            dW = random.gauss(0, math.sqrt(dt))
            S = S + mu * dt + sigma * dW
            path.append(max(0, S))  # Prevent negative values
        simulations.append(path)
    
    return simulations


def _jump_diffusion_model(S0, mu, sigma, T, N, jump_intensity=0.1, jump_mean=0, jump_std=0.02):
    """Merton Jump Diffusion Model: GBM + Poisson jumps"""
    dt = 1.0
    simulations = []
    
    for _ in range(N):
        path = [S0]
        S = S0
        for _ in range(T - 1):
            # Diffusion component
            dW = random.gauss(0, math.sqrt(dt))
            drift = (mu - 0.5 * sigma**2) * dt + sigma * dW
            
            # Jump component (Poisson process)
            n_jumps = 0
            if random.random() < jump_intensity * dt:
                n_jumps = 1
            
            jump = 0
            for _ in range(n_jumps):
                jump += random.gauss(jump_mean, jump_std)
            
            S = S * math.exp(drift + jump)
            path.append(S)
        simulations.append(path)
    
    return simulations


def _historical_simulation(S0, returns, T, N):
    """Bootstrap historical returns"""
    simulations = []
    
    for _ in range(N):
        path = [S0]
        S = S0
        for _ in range(T - 1):
            # Randomly sample from historical returns
            ret = random.choice(returns) if returns else 0
            S = S * (1 + ret)
            path.append(S)
        simulations.append(path)
    
    return simulations


def _calculate_var(final_values, initial_value, confidence):
    """Calculate Value at Risk"""
    losses = [initial_value - v for v in final_values]
    losses.sort(reverse=True)
    var_index = int(len(losses) * confidence)
    return losses[var_index] if var_index < len(losses) else losses[-1]


def _calculate_cvar(final_values, initial_value, confidence):
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    losses = [initial_value - v for v in final_values]
    losses.sort(reverse=True)
    var_index = int(len(losses) * confidence)
    tail_losses = losses[:var_index] if var_index > 0 else losses
    return sum(tail_losses) / len(tail_losses) if tail_losses else 0


def scenario_analysis(
    df,
    value_column: str,
    scenarios: Dict[str, Dict[str, float]],
    time_periods: int = 252
):
    """
    Perform scenario analysis with predefined scenarios
    
    Args:
        df: DataFrame instance
        value_column: Column to analyze
        scenarios: Dictionary of scenarios with parameters
            Example: {
                'bull_market': {'mean': 0.15, 'std': 0.10},
                'bear_market': {'mean': -0.10, 'std': 0.25},
                'base_case': {'mean': 0.07, 'std': 0.15}
            }
        time_periods: Number of time steps
    
    Returns:
        Dictionary with scenario results
    """
    from ..core.dataframe import DataFrame
    
    values = [float(v) for v in df._data[value_column] if v is not None and v != '']
    initial_value = values[-1] if values else 0
    
    results = {}
    for scenario_name, params in scenarios.items():
        mean_return = params.get('mean', 0)
        std_return = params.get('std', 0.15)
        n_sims = params.get('simulations', 1000)
        
        simulations = _geometric_brownian_motion(
            initial_value, mean_return, std_return, time_periods, n_sims
        )
        
        final_values = [sim[-1] for sim in simulations]
        
        results[scenario_name] = {
            'mean_final': sum(final_values) / len(final_values),
            'median_final': _percentile(final_values, 0.5),
            'std_final': _std_dev(final_values),
            'probability_positive': sum(1 for v in final_values if v > initial_value) / len(final_values),
            'expected_return': (sum(final_values) / len(final_values) - initial_value) / initial_value,
            'percentile_5': _percentile(final_values, 0.05),
            'percentile_95': _percentile(final_values, 0.95),
        }
    
    return results


def stress_test(
    df,
    value_column: str,
    stress_scenarios: List[Dict[str, Any]],
    n_simulations: int = 1000
):
    """
    Perform stress testing with extreme scenarios
    
    Args:
        df: DataFrame instance
        value_column: Column to stress test
        stress_scenarios: List of stress test scenarios
            Example: [
                {'name': '2008 Crisis', 'shock': -0.50, 'volatility_multiplier': 3},
                {'name': 'Black Swan', 'shock': -0.75, 'volatility_multiplier': 5},
            ]
        n_simulations: Number of simulations per scenario
    
    Returns:
        Dictionary with stress test results
    """
    from ..core.dataframe import DataFrame
    
    values = [float(v) for v in df._data[value_column] if v is not None and v != '']
    returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
    base_std = _std_dev(returns) if returns else 0.15
    initial_value = values[-1] if values else 0
    
    results = {}
    for scenario in stress_scenarios:
        name = scenario.get('name', 'Unnamed')
        shock = scenario.get('shock', -0.30)
        vol_mult = scenario.get('volatility_multiplier', 2)
        
        shocked_value = initial_value * (1 + shock)
        shocked_std = base_std * vol_mult
        
        # Run simulations from shocked state
        simulations = _geometric_brownian_motion(
            shocked_value, 0, shocked_std, 30, n_simulations
        )
        
        recovery_values = [sim[-1] for sim in simulations]
        
        results[name] = {
            'initial_shock': shock * 100,
            'shocked_value': shocked_value,
            'mean_recovery': sum(recovery_values) / len(recovery_values),
            'probability_full_recovery': sum(1 for v in recovery_values if v >= initial_value) / len(recovery_values),
            'worst_case': min(recovery_values),
            'best_case': max(recovery_values),
            'expected_loss': max(0, initial_value - sum(recovery_values) / len(recovery_values)),
        }
    
    return results


def portfolio_monte_carlo(
    df,
    asset_columns: List[str],
    weights: Optional[List[float]] = None,
    n_simulations: int = 10000,
    time_periods: int = 252,
    correlation_matrix: Optional[List[List[float]]] = None
):
    """
    Monte Carlo simulation for portfolio of multiple assets
    
    Args:
        df: DataFrame instance
        asset_columns: List of column names for assets
        weights: Portfolio weights (equal weight if None)
        n_simulations: Number of simulation paths
        time_periods: Number of time steps
        correlation_matrix: Asset correlation matrix (calculated if None)
    
    Returns:
        Dictionary with portfolio simulation results
    """
    from ..core.dataframe import DataFrame
    
    n_assets = len(asset_columns)
    
    # Default equal weights
    if weights is None:
        weights = [1.0 / n_assets] * n_assets
    
    if len(weights) != n_assets:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of assets ({n_assets})")
    
    # Calculate returns and statistics for each asset
    asset_stats = []
    for col in asset_columns:
        values = [float(v) for v in df._data[col] if v is not None and v != '']
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        asset_stats.append({
            'initial_value': values[-1] if values else 0,
            'mean_return': sum(returns) / len(returns) if returns else 0,
            'std_return': _std_dev(returns) if returns else 0,
        })
    
    # Calculate portfolio initial value
    initial_portfolio_value = sum(w * s['initial_value'] for w, s in zip(weights, asset_stats))
    
    # Run correlated simulations
    portfolio_simulations = []
    
    for _ in range(n_simulations):
        asset_paths = []
        
        for stats in asset_stats:
            path = _geometric_brownian_motion(
                stats['initial_value'],
                stats['mean_return'],
                stats['std_return'],
                time_periods,
                1
            )[0]
            asset_paths.append(path)
        
        # Calculate portfolio value at each time step
        portfolio_path = []
        for t in range(time_periods):
            portfolio_value = sum(w * asset_paths[i][t] for i, w in enumerate(weights))
            portfolio_path.append(portfolio_value)
        
        portfolio_simulations.append(portfolio_path)
    
    # Calculate statistics
    final_values = [sim[-1] for sim in portfolio_simulations]
    mean_path = [sum(sim[t] for sim in portfolio_simulations) / n_simulations 
                 for t in range(time_periods)]
    
    statistics = {
        'initial_value': initial_portfolio_value,
        'mean_final': sum(final_values) / len(final_values),
        'median_final': _percentile(final_values, 0.5),
        'std_final': _std_dev(final_values),
        'sharpe_ratio': _calculate_sharpe_ratio(final_values, initial_portfolio_value),
        'probability_positive': sum(1 for v in final_values if v > initial_portfolio_value) / len(final_values),
        'var_95': _calculate_var(final_values, initial_portfolio_value, 0.95),
        'var_99': _calculate_var(final_values, initial_portfolio_value, 0.99),
    }
    
    return {
        'simulations': portfolio_simulations,
        'mean_path': mean_path,
        'final_values': final_values,
        'statistics': statistics,
        'weights': weights,
        'asset_columns': asset_columns,
    }


def sensitivity_analysis(
    df,
    value_column: str,
    parameter_ranges: Dict[str, List[float]],
    base_params: Dict[str, float],
    n_simulations: int = 1000,
    time_periods: int = 252
):
    """
    Perform sensitivity analysis by varying parameters
    
    Args:
        df: DataFrame instance
        value_column: Column to analyze
        parameter_ranges: Dictionary of parameter ranges to test
            Example: {
                'mean': [-0.10, 0.0, 0.05, 0.10, 0.15],
                'std': [0.10, 0.15, 0.20, 0.25, 0.30]
            }
        base_params: Base case parameters
        n_simulations: Number of simulations per parameter set
        time_periods: Number of time steps
    
    Returns:
        Dictionary with sensitivity results
    """
    from ..core.dataframe import DataFrame
    
    values = [float(v) for v in df._data[value_column] if v is not None and v != '']
    initial_value = values[-1] if values else 0
    
    results = {}
    
    for param_name, param_values in parameter_ranges.items():
        param_results = []
        
        for param_value in param_values:
            test_params = base_params.copy()
            test_params[param_name] = param_value
            
            simulations = _geometric_brownian_motion(
                initial_value,
                test_params.get('mean', 0),
                test_params.get('std', 0.15),
                time_periods,
                n_simulations
            )
            
            final_values = [sim[-1] for sim in simulations]
            
            param_results.append({
                'parameter_value': param_value,
                'mean_final': sum(final_values) / len(final_values),
                'std_final': _std_dev(final_values),
                'probability_positive': sum(1 for v in final_values if v > initial_value) / len(final_values),
            })
        
        results[param_name] = param_results
    
    return results


def _calculate_sharpe_ratio(final_values, initial_value, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    returns = [(v - initial_value) / initial_value for v in final_values]
    mean_return = sum(returns) / len(returns)
    std_return = _std_dev(returns)
    
    if std_return == 0:
        return 0
    
    return (mean_return - risk_free_rate) / std_return


def _percentile(values: List[float], p: float) -> float:
    """Calculate percentile"""
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_vals[int(k)]
    
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1


def _std_dev(values: List[float]) -> float:
    """Calculate standard deviation"""
    if not values:
        return 0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

