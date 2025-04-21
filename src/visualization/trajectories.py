"""
Visualization utilities for prediction trajectories.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def plot_prediction_trajectory(data: pd.DataFrame, 
                             predictions: List[Dict[str, Any]], 
                             output_path: str,
                             figsize: tuple = (12, 8)) -> None:
    """
    Plot the trajectory with handover predictions.
    
    Args:
        data: DataFrame containing mobility data
        predictions: List of prediction results
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot trajectory
    plt.plot(data['x'], data['y'], 'b-', alpha=0.5, label='Trajectory')
    
    # Plot points with predictions
    for pred in predictions:
        x, y = pred['x'], pred['y']
        prob = pred['handover_probability']
        decision = pred['decision']
        
        if decision == 'initiate_handover':
            color = 'red'
            marker = '^'
        else:
            color = 'green'
            marker = 'o'
        
        plt.scatter(x, y, c=color, marker=marker, s=100, alpha=0.6)
        plt.annotate(f'{prob:.2f}', (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.title('Mobility Trajectory with Handover Predictions')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(['Trajectory', 'Handover', 'No Handover'])
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
