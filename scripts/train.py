#!/usr/bin/env python3
"""
Command-line script for training the mobility prediction model.
"""

import os
import argparse
import yaml
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add the project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.data.data_utils import load_mobility_data, prepare_data_for_inference
from src.models.trainer import (
    prepare_train_val_test_split, 
    train_model, 
    evaluate_model, 
    save_model_artifacts
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train mobility prediction model')
    
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to the mobility data file (.parquet, .pkl, .csv, or .npy)')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--output', type=str, default='models/saved',
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use (for memory efficiency)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    # Convert relative path to absolute if needed
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    
    # Create default config
    default_config = create_default_config()
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}. Using default configuration.")
        return default_config
    
    # Load config from file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure all required fields are present
        if 'model' not in config:
            config['model'] = default_config['model']
        if 'training' not in config:
            config['training'] = default_config['training']
        
        # Ensure model fields are present
        for key in default_config['model']:
            if key not in config['model']:
                config['model'][key] = default_config['model'][key]
        
        # Ensure training fields are present
        for key in default_config['training']:
            if key not in config['training']:
                config['training'][key] = default_config['training'][key]
        
        return config
    except Exception as e:
        print(f"Error loading config file: {e}. Using default configuration.")
        return default_config


def create_default_config():
    """Create a default configuration."""
    return {
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


def main():
    """Main entry point for model training script."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Add output directory to config
    config['training']['checkpoint_dir'] = os.path.join(output_dir, 'checkpoints')
    config['training']['log_dir'] = os.path.join(output_dir, 'logs')
    
    # Load data
    print(f"Loading data from {args.data}")
    mobility_data = load_mobility_data(args.data)
    print(f"Loaded data with {len(mobility_data):,} samples")
    
    # Prepare sequences for LSTM
    print("Preparing sequences for training...")
    X, timestamps, user_ids, connected_cells = prepare_data_for_inference(
        mobility_data, 
        sequence_length=config['model']['sequence_length']
    )
    
    # Create target sequences (you'll need to implement this based on your needs)
    y = X[:, 1:, :]  # Using next timestep as target, adjust based on your needs
    
    # Create scaler and feature names for model artifacts
    scaler = StandardScaler()
    # Reshape X to 2D for scaling
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler.fit(X_reshaped)
    # Apply scaling to X
    X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
    X = X_scaled
    
    # Define feature names based on the actual data columns
    feature_names = ['x', 'y', 'velocity', 'heading']
    
    print(f"Created {len(X):,} sequences with shape {X.shape}")
    
    # Split data
    print("Splitting data into train/validation/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
        X, y,
        validation_split=config['training']['validation_split'],
        test_split=config['training']['test_split'],
        random_seed=config['training']['random_seed'],
        max_samples=args.max_samples
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
        output_dir
    )
    
    print("\nTraining complete!")
    print(f"All model artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()