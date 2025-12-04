"""
PipelineScript Integration for PySenseDF
=========================================

Seamless integration with PipelineScript for ML pipeline orchestration.
https://pypi.org/project/pipelinescript/

Features:
- Convert PySenseDF DataFrames to PipelineScript pipelines
- Execute PipelineScript commands on PySenseDF data
- Combine PySenseDF's speed with PipelineScript's readability
- Support for ML workflows with Monte Carlo simulation

Installation:
    pip install pipelinescript

Usage:
    from pysensedf import DataFrame
    from pysensedf.integrations.pipelinescript_integration import to_pipeline
    
    # Create DataFrame
    df = DataFrame({'data': data})
    
    # Convert to PipelineScript pipeline
    pipeline = to_pipeline(df, target='label')
    
    # Execute pipeline
    result = pipeline.run()
"""

from typing import Optional, Dict, Any, List, Union
from ..core.dataframe import DataFrame


def to_pipeline(
    df: DataFrame,
    target: Optional[str] = None,
    model: str = 'auto',
    test_size: float = 0.2,
    clean_missing: bool = True,
    encode: bool = True,
    scale: bool = True,
    export_path: Optional[str] = None
):
    """
    Convert PySenseDF DataFrame to PipelineScript pipeline
    
    Args:
        df: PySenseDF DataFrame
        target: Target column name
        model: Model type ('auto', 'xgboost', 'random_forest', 'logistic', 'linear')
        test_size: Test set size (0.2 = 20%)
        clean_missing: Whether to clean missing values
        encode: Whether to encode categorical variables
        scale: Whether to scale features
        export_path: Path to export trained model (optional)
    
    Returns:
        PipelineScript Pipeline object
    
    Example:
        df = DataFrame.from_csv('data.csv')
        pipeline = to_pipeline(df, target='price', model='xgboost')
        result = pipeline.run()
        print(f"Accuracy: {result.context.metrics['accuracy']:.4f}")
    """
    try:
        from pipelinescript import Pipeline
    except ImportError:
        raise ImportError(
            "PipelineScript not installed. Install with: pip install pipelinescript"
        )
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Load data from DataFrame (convert to in-memory representation)
    pipeline._context.data = _to_pandas_compatible(df)
    
    # Add cleaning steps
    if clean_missing:
        pipeline.clean_missing()
    
    # Add encoding
    if encode:
        pipeline.encode()
    
    # Add scaling
    if scale:
        pipeline.scale()
    
    # Add train/test split
    if target:
        train_size = 1.0 - test_size
        pipeline.split(train_size, target=target)
    
    # Add training
    if model == 'auto':
        pipeline.train('auto')
    else:
        pipeline.train(model)
    
    # Add evaluation
    pipeline.evaluate()
    
    # Add export if specified
    if export_path:
        pipeline.export(export_path)
    
    return pipeline


def from_pipeline_result(result) -> DataFrame:
    """
    Convert PipelineScript execution result to PySenseDF DataFrame
    
    Args:
        result: PipelineScript execution result
    
    Returns:
        PySenseDF DataFrame with predictions and metrics
    
    Example:
        result = pipeline.run()
        df_results = from_pipeline_result(result)
        print(df_results.head())
    """
    data = {}
    
    # Extract predictions
    if hasattr(result.context, 'predictions') and result.context.predictions is not None:
        data['predictions'] = list(result.context.predictions)
    
    # Extract metrics as single-row columns
    if hasattr(result.context, 'metrics') and result.context.metrics:
        for metric_name, metric_value in result.context.metrics.items():
            data[f'metric_{metric_name}'] = [metric_value]
    
    # Extract test labels if available
    if hasattr(result.context, 'y_test') and result.context.y_test is not None:
        data['actual'] = list(result.context.y_test)
    
    return DataFrame(data)


def execute_psl_script(df: DataFrame, script: str, target: Optional[str] = None):
    """
    Execute a PipelineScript (.psl) script on a DataFrame
    
    Args:
        df: PySenseDF DataFrame
        script: PipelineScript commands as string
        target: Target column name (required if script uses split/train)
    
    Returns:
        Tuple of (result, output_dataframe)
    
    Example:
        script = '''
        clean missing
        encode
        split 80/20 --target price
        train xgboost
        evaluate
        '''
        result, df_out = execute_psl_script(df, script, target='price')
    """
    try:
        from pipelinescript import run
    except ImportError:
        raise ImportError(
            "PipelineScript not installed. Install with: pip install pipelinescript"
        )
    
    # Save DataFrame to temporary CSV
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        df.to_csv(temp_path)
    
    try:
        # Prepend load command to script
        full_script = f"load {temp_path}\n{script}"
        
        # Execute script
        result = run(full_script)
        
        # Convert result to DataFrame
        output_df = from_pipeline_result(result) if result.success else DataFrame({})
        
        return result, output_df
    
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_path)
        except:
            pass


def quick_ml_pipeline(
    df: DataFrame,
    target: str,
    model: str = 'xgboost',
    task: str = 'auto'
) -> Dict[str, Any]:
    """
    Quick ML pipeline using PipelineScript's quick builders
    
    Args:
        df: PySenseDF DataFrame
        target: Target column name
        model: Model type
        task: 'classification', 'regression', or 'auto'
    
    Returns:
        Dictionary with results
    
    Example:
        df = DataFrame.from_csv('data.csv')
        results = quick_ml_pipeline(df, target='species', task='classification')
        print(f"Accuracy: {results['accuracy']:.4f}")
    """
    try:
        from pipelinescript.pipeline import quick_classification, quick_regression, quick_train
    except ImportError:
        raise ImportError(
            "PipelineScript not installed. Install with: pip install pipelinescript"
        )
    
    # Save to temporary file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        df.to_csv(temp_path)
    
    try:
        # Determine task type
        if task == 'auto':
            # Auto-detect from target column
            unique_values = len(set(df._data.get(target, [])))
            task = 'classification' if unique_values < 20 else 'regression'
        
        # Execute appropriate pipeline
        if task == 'classification':
            result = quick_classification(temp_path, target, model)
        elif task == 'regression':
            result = quick_regression(temp_path, target, model)
        else:
            result = quick_train(temp_path, target, f"{temp_path}.model.pkl")
        
        # Extract metrics
        if result.success:
            return {
                'success': True,
                'metrics': result.context.metrics,
                'duration': result.duration,
                'predictions': list(result.context.predictions) if hasattr(result.context, 'predictions') else None,
            }
        else:
            return {
                'success': False,
                'errors': result.errors,
            }
    
    finally:
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass


def monte_carlo_pipeline(
    df: DataFrame,
    value_column: str,
    pipeline_script: Optional[str] = None,
    n_simulations: int = 10000,
    **monte_carlo_kwargs
) -> Dict[str, Any]:
    """
    Combine Monte Carlo simulation with PipelineScript ML pipeline
    
    Runs Monte Carlo simulation first, then trains ML model on simulated data.
    
    Args:
        df: PySenseDF DataFrame
        value_column: Column to simulate
        pipeline_script: Optional PipelineScript commands to run on simulated data
        n_simulations: Number of Monte Carlo simulations
        **monte_carlo_kwargs: Additional arguments for monte_carlo()
    
    Returns:
        Dictionary with Monte Carlo results and ML pipeline results
    
    Example:
        results = monte_carlo_pipeline(
            df,
            'stock_price',
            pipeline_script='train xgboost\\nevaluate',
            n_simulations=5000,
            time_periods=252
        )
    """
    # Run Monte Carlo simulation
    mc_results = df.monte_carlo(value_column, n_simulations=n_simulations, **monte_carlo_kwargs)
    
    # If no pipeline script, return MC results only
    if not pipeline_script:
        return {
            'monte_carlo': mc_results,
            'pipeline': None
        }
    
    # Create DataFrame from simulated paths
    simulated_data = {}
    
    # Add mean path
    simulated_data['mean_path'] = mc_results['mean_path']
    
    # Add percentile paths
    for p, path in mc_results['percentiles'].items():
        simulated_data[f'percentile_{p}'] = path
    
    # Add time index
    simulated_data['time_step'] = list(range(len(mc_results['mean_path'])))
    
    df_simulated = DataFrame(simulated_data)
    
    # Execute pipeline on simulated data
    try:
        pipeline_result, df_output = execute_psl_script(df_simulated, pipeline_script)
        
        return {
            'monte_carlo': mc_results,
            'pipeline': {
                'success': pipeline_result.success,
                'metrics': pipeline_result.context.metrics if hasattr(pipeline_result.context, 'metrics') else {},
                'predictions': df_output,
            }
        }
    except Exception as e:
        return {
            'monte_carlo': mc_results,
            'pipeline': {
                'success': False,
                'error': str(e)
            }
        }


def _to_pandas_compatible(df: DataFrame):
    """Convert PySenseDF DataFrame to pandas-compatible format"""
    try:
        import pandas as pd
        return pd.DataFrame(df._data)
    except ImportError:
        # Return dict format if pandas not available
        return df._data


# Extend DataFrame with PipelineScript methods
def add_pipelinescript_methods():
    """
    Add PipelineScript integration methods to DataFrame class
    
    Usage:
        from pysensedf.integrations.pipelinescript_integration import add_pipelinescript_methods
        add_pipelinescript_methods()
        
        # Now you can use:
        df.to_pipeline(target='label')
        df.quick_ml(target='price', model='xgboost')
    """
    DataFrame.to_pipeline = lambda self, **kwargs: to_pipeline(self, **kwargs)
    DataFrame.execute_psl = lambda self, script, target=None: execute_psl_script(self, script, target)
    DataFrame.quick_ml = lambda self, target, model='xgboost', task='auto': quick_ml_pipeline(self, target, model, task)
    DataFrame.monte_carlo_pipeline = lambda self, value_column, pipeline_script=None, n_simulations=10000, **kwargs: monte_carlo_pipeline(self, value_column, pipeline_script, n_simulations, **kwargs)


# Auto-register methods when module is imported
try:
    add_pipelinescript_methods()
except:
    pass  # Silently fail if DataFrame not available
