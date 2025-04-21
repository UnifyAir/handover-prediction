"""
Data preprocessing utilities for mobility prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler


def create_feature_scaler(data: pd.DataFrame) -> StandardScaler:
    """
    Create and fit a scaler for mobility features.
    
    Args:
        data: DataFrame containing mobility data
        
    Returns:
        Fitted StandardScaler
    """
    scaler = StandardScaler()
    features = data[['x', 'y', 'velocity', 'heading']].values
    scaler.fit(features)
    return scaler


def save_scaler(scaler: StandardScaler, file_path: str) -> None:
    """
    Save a fitted scaler to a file.
    
    Args:
        scaler: Fitted scaler to save
        file_path: Path to save the scaler
    """
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(file_path: str) -> StandardScaler:
    """
    Load a saved scaler from a file.
    
    Args:
        file_path: Path to the saved scaler
        
    Returns:
        Loaded StandardScaler
    """
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f) 