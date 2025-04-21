"""
Data utilities for loading and processing mobility data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import pickle


def load_mobility_data(file_path: str) -> pd.DataFrame:
    """
    Load mobility data from a file.
    
    Args:
        file_path: Path to the data file (.pkl, .csv, or .parquet)
        
    Returns:
        DataFrame containing mobility data
    """
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def prepare_data_for_inference(
    data: pd.DataFrame,
    sequence_length: int = 20,
    scaler: Optional[object] = None
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """
    Prepare data for inference by creating sequences and applying scaling.
    
    Args:
        data: DataFrame containing mobility data
        sequence_length: Length of sequences to create
        scaler: Optional scaler to apply to features
        
    Returns:
        Tuple containing:
        - X: Feature sequences
        - timestamps: List of timestamps
        - user_ids: List of user IDs
        - connected_cells: List of connected cells
    """
    # Sort data by timestamp
    data = data.sort_values('timestamp')
    
    # Initialize lists for sequences
    sequences = []
    timestamps = []
    user_ids = []
    connected_cells = []
    
    # Group by user
    for user_id, user_data in data.groupby('user_id'):
        if len(user_data) < sequence_length:
            continue
            
        # Create sequences
        for i in range(len(user_data) - sequence_length + 1):
            sequence = user_data.iloc[i:i+sequence_length]
            
            # Extract features (using the actual columns in the data)
            features = sequence[['x', 'y', 'velocity', 'heading']].values
            
            # Apply scaling if provided
            if scaler is not None:
                features = scaler.transform(features)
            
            sequences.append(features)
            timestamps.append(sequence.iloc[-1]['timestamp'])
            user_ids.append(user_id)
            connected_cells.append(sequence.iloc[-1]['connected_cell'])
    
    return np.array(sequences), timestamps, user_ids, connected_cells 