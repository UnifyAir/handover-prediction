#!/usr/bin/env python3
"""
Command-line script for training the mobility prediction model.
"""

import os
import argparse
import yaml
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_utils import load_mobility_data
from src.preprocessing import prepare_sequences
from src.trainer import (
    prepare_train_val_test_split, 
    train_model, 
    evaluate_model, 
    save_model_artifacts
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train mobility prediction model')
    
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to the mobility data file (.pkl or .csv)')
    parser.add_argument('--config', type=str, default='../configs/model_config.yaml',
                        help='Path to model configuration file')
    parser.add_argument('--output', type=str, default='../models/saved',
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point for model training script."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration")
        config = {
            'model': {
                'sequence_length': 20,
                'prediction_horizon': 5,
                'lstm_units': 64,
                'dropout_rate': 0.3
            },
            'training': {
                'epochs': 50,
                'batch_size': 64,
                'validation_split': 0.15,
                'test_split': 0.15,
                'early_stopping_patience': 10,
                'lr_reduction_factor': 0.5,
                'lr_patience': 5,
                'min_lr': 0.0001,
                'random_seed': 42
            }
        }
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Add output directory to config
    config['training']['checkpoint_dir'] = os.path.join(args.output, 'checkpoints')
    config['training']['log_dir'] = os.path.join(args.output, 'logs')
    
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
    
    # Split data
    print("Splitting data into train/validation/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
        X, y,
        validation_split=config['training']['validation_split'],
        test_split=config['training']['test_split'],
        random_seed=config['training']['random_seed']
    )
    
    print(f"Data splits:")
    print(f"  Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Train model
    print("\nStarting model training...")
    model, history = train_model(
        X_train, y_train,
        X_val, y_val,
        config['model'],
        config['training']
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_results = evaluate_model(model, X_test, y_test)
    
    print("\nTest results:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model artifacts
    print("\nSaving model artifacts...")
    artifacts = save_model_artifacts(
        model, 
        history, 
        scaler, 
        feature_names, 
        test_results, 
        config, 
        args.output
    )
    
    print("\nTraining complete!")
    print(f"All model artifacts saved to {args.output}")


if __name__ == "__main__":
    main()