import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import argparse

# Add the project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from data.synthetic.generator import generate_synthetic_mobility_data, visualize_trajectories
from data.processed.processor import add_network_conditions, prepare_sequences, create_balanced_dataset

def load_config(config_path):
    """Load configuration from YAML file."""
    # Convert relative path to absolute if needed
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic mobility data')
    parser.add_argument('--num-users', type=int, help='Number of users to generate')
    parser.add_argument('--days', type=int, help='Number of days to generate')
    parser.add_argument('--sampling-rate', type=int, help='Sampling rate in seconds')
    parser.add_argument('--config', type=str, default='configs/generation_config.yaml',
                        help='Path to generation configuration file')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    num_users = args.num_users if args.num_users is not None else config['generator']['num_users']
    days = args.days if args.days is not None else config['generator']['days']
    sampling_rate = args.sampling_rate if args.sampling_rate is not None else config['generator']['sampling_rate_seconds']
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(PROJECT_ROOT, 'data/raw'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'data/processed'), exist_ok=True)
    
    # Generate synthetic data
    print("Generating synthetic mobility data...")
    raw_data = generate_synthetic_mobility_data(
        num_users=num_users,
        days=days,
        sampling_rate_seconds=sampling_rate,
        grid_config=config['generator']['grid'],
        signal_config=config['generator']['signal'],
        patterns_config=config['generator']['patterns'],
        handover_config=config['generator']['handover']
    )
    
    # Save raw data
    raw_data_path = os.path.join(PROJECT_ROOT, config['output']['raw_data'])
    raw_data.to_parquet(raw_data_path)
    print(f"Saved raw data to {raw_data_path}")
    
    # Visualize trajectories
    print("\nVisualizing sample trajectories...")
    visualize_trajectories(raw_data, grid_config=config['generator']['grid'])
    
    # Process data
    print("\nProcessing data...")
    processed_data = add_network_conditions(raw_data)
    
    # Create balanced dataset
    balanced_data = create_balanced_dataset(processed_data)
    
    # Prepare sequences for training
    X, y = prepare_sequences(processed_data)
    
    # Save processed data
    processed_data_path = os.path.join(PROJECT_ROOT, config['output']['processed_data'])
    processed_data.to_parquet(processed_data_path)
    print(f"Saved processed data to {processed_data_path}")
    
    # Save balanced dataset
    balanced_data_path = os.path.join(PROJECT_ROOT, config['output']['balanced_data'])
    balanced_data.to_parquet(balanced_data_path)
    print(f"Saved balanced dataset to {balanced_data_path}")
    
    # Save sequence data
    sequence_data = {
        'X': X,
        'y': y
    }
    sequence_data_path = os.path.join(PROJECT_ROOT, config['output']['sequence_data'])
    np.save(sequence_data_path, sequence_data)
    print(f"Saved sequence data to {sequence_data_path}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Balanced dataset shape: {balanced_data.shape}")
    print(f"Sequence data shape: X: {X.shape}, y: {y.shape}")
    print(f"Time range: {raw_data['timestamp'].min()} to {raw_data['timestamp'].max()}")
    print(f"Number of unique users: {raw_data['user_id'].nunique()}")
    print(f"Number of handovers: {raw_data['handover_needed'].sum()}")

if __name__ == "__main__":
    main() 