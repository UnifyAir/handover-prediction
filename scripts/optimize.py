#!/usr/bin/env python3
"""
Command-line script for hyperparameter optimization of the mobility prediction model.
"""

import os
import argparse
import yaml
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import ParameterGrid

# Import from project modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_utils import load_mobility_data
from src.preprocessing import prepare_sequences
from src.trainer import prepare_train_val_test_split, train_model, evaluate_model

# Add the project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimize model hyperparameters')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the mobility data file (.parquet, .pkl, .csv, or .npy)')
    parser.add_argument('--config', type=str, default='configs/optimization_config.yaml',
                        help='Path to optimization configuration file')
    parser.add_argument('--output', type=str, default='models/optimized',
                        help='Directory to save optimization results')
    parser.add_argument('--n-trials', type=int, help='Number of optimization trials')
    parser.add_argument('--cv-folds', type=int, help='Number of cross-validation folds')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    # Convert relative path to absolute if needed
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_hyperparameter_combinations(param_grid, max_trials=None):
    """
    Generate hyperparameter combinations for grid search.
    
    Args:
        param_grid: Dictionary with parameter names as keys and lists of parameter values
        max_trials: Maximum number of random combinations to try (None for full grid search)
        
    Returns:
        List of parameter dictionaries
    """
    full_grid = list(ParameterGrid(param_grid))
    
    if max_trials is None or max_trials >= len(full_grid):
        return full_grid
    
    # Randomly sample if max_trials is less than the full grid size
    indices = np.random.choice(len(full_grid), max_trials, replace=False)
    return [full_grid[i] for i in indices]


def run_optimization_trial(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model_params, training_params, base_config
):
    """
    Run a single optimization trial with specific hyperparameters.
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits
        model_params: Model hyperparameters for this trial
        training_params: Training hyperparameters for this trial
        base_config: Base configuration for non-optimized parameters
        
    Returns:
        Dictionary with trial results
    """
    # Merge parameters with base configurations
    model_config = base_config.get('model', {}).copy()
    model_config.update(model_params)
    
    training_config = base_config.get('training', {}).copy()
    training_config.update(training_params)
    
    # Set verbose to 0 to reduce output during optimization
    training_config['verbose'] = 0
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, model_config, training_config)
    
    # Evaluate on validation set (last epoch metrics)
    val_metrics = {f"val_{k}": v[-1] for k, v in history.history.items() 
                  if k.startswith('val_')}
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, X_test, y_test)
    
    # Combine parameters and results
    result = {
        'model_params': model_params,
        'training_params': training_params,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_epoch': np.argmin(history.history['val_loss']) + 1,
        'total_epochs': len(history.history['val_loss'])
    }
    
    return result


def main():
    """Main entry point for hyperparameter optimization script."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data}")
    mobility_data = load_mobility_data(args.data)
    print(f"Loaded data with {len(mobility_data):,} samples")
    
    # Prepare sequences for LSTM
    print("Preparing sequences for training...")
    X, y, scaler, feature_names = prepare_sequences(
        mobility_data, 
        sequence_length=config['model']['sequence_length'],
        prediction_horizon=config['model']['prediction_horizon']
    )
    
    print(f"Created {len(X):,} sequences with shape {X.shape}")
    
    # Split data once for all trials
    print("Splitting data into train/validation/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
        X, y,
        validation_split=config['training'].get('validation_split', 0.15),
        test_split=config['training'].get('test_split', 0.15),
        random_seed=config['training'].get('random_seed', 42)
    )
    
    # Generate hyperparameter combinations
    model_param_grid = config['model_params']
    training_param_grid = config['training_params']
    
    # If learning_rate is in training params, add it to the optimizer config
    if 'learning_rate' in training_param_grid:
        lr_values = training_param_grid.pop('learning_rate')
        training_param_grid['optimizer_config'] = [{'learning_rate': lr} for lr in lr_values]
    
    param_combinations = []
    for model_params in generate_hyperparameter_combinations(model_param_grid):
        for training_params in generate_hyperparameter_combinations(training_param_grid):
            param_combinations.append((model_params, training_params))
    
    # Limit number of trials if specified
    max_trials = config['optimization'].get('max_trials')
    if max_trials and max_trials < len(param_combinations):
        np.random.shuffle(param_combinations)
        param_combinations = param_combinations[:max_trials]
    
    print(f"Running {len(param_combinations)} optimization trials...")
    
    # Run optimization trials
    results = []
    for i, (model_params, training_params) in enumerate(param_combinations):
        print(f"\nTrial {i+1}/{len(param_combinations)}")
        print(f"Model params: {model_params}")
        print(f"Training params: {training_params}")
        
        try:
            result = run_optimization_trial(
                X_train, y_train, X_val, y_val, X_test, y_test,
                model_params, training_params, config
            )
            
            # Extract metric of interest
            metric_name = config['optimization'].get('metric', 'val_loss')
            metric_mode = config['optimization'].get('mode', 'min')
            
            if metric_name in result['val_metrics']:
                metric_value = result['val_metrics'][metric_name]
            elif metric_name in result['test_metrics']:
                metric_value = result['test_metrics'][metric_name]
            else:
                metric_value = None
                
            result['metric_value'] = metric_value
            results.append(result)
            
            # Print trial results
            print(f"Trial completed: {metric_name} = {metric_value:.4f}")
            print(f"Best epoch: {result['best_epoch']}/{result['total_epochs']}")
            
        except Exception as e:
            print(f"Error in trial: {e}")
    
    # Sort results by optimization metric
    metric_mode = config['optimization'].get('mode', 'min')
    results.sort(key=lambda x: x['metric_value'], reverse=(metric_mode == 'max'))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"optimization_results_{timestamp}.pkl")
    joblib.dump(results, results_file)
    
    # Save summary as YAML
    summary = {
        'optimization_config': config,
        'best_trial': results[0] if results else None,
        'timestamp': timestamp,
        'num_trials': len(results),
        'best_params': {
            'model': results[0]['model_params'] if results else None,
            'training': results[0]['training_params'] if results else None
        }
    }
    
    summary_file = os.path.join(output_dir, f"optimization_summary_{timestamp}.yaml")
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f)
    
    # Print best results
    print("\nOptimization complete!")
    print(f"Results saved to {results_file}")
    print(f"Summary saved to {summary_file}")
    
    if results:
        print("\nBest hyperparameters:")
        print(f"Model parameters: {results[0]['model_params']}")
        print(f"Training parameters: {results[0]['training_params']}")
        print(f"Metric ({config['optimization'].get('metric', 'val_loss')}): {results[0]['metric_value']:.4f}")
        print(f"Test metrics: {results[0]['test_metrics']}")
    

if __name__ == "__main__":
    main()