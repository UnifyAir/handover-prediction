import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_network_conditions(df):
    # Add time-of-day based network congestion
    df['hour'] = df['timestamp'].dt.hour
    
    # Peak hours have more congestion
    peak_hours_morning = (7 <= df['hour']) & (df['hour'] <= 9)
    peak_hours_evening = (16 <= df['hour']) & (df['hour'] <= 19)
    
    # Base network load (30-50%)
    df['network_load'] = np.random.uniform(0.3, 0.5, size=len(df))
    
    # Increase during peak hours (60-90%)
    df.loc[peak_hours_morning | peak_hours_evening, 'network_load'] = \
        np.random.uniform(0.6, 0.9, size=len(df[peak_hours_morning | peak_hours_evening]))
    
    # Add random network quality metrics
    df['sinr'] = df['signal_strength'] - np.random.uniform(-5, 5, size=len(df)) - 10 * df['network_load']
    df['throughput_mbps'] = 10 * (1 + np.log10(1 + df['sinr'])) * (1 - df['network_load']*0.7)
    
    # Add UE capabilities
    device_categories = ['5G_basic', '5G_advanced', '5G_premium']
    df['device_type'] = np.random.choice(device_categories, size=len(df))
    
    # Add handover performance metrics
    df['handover_latency'] = np.random.uniform(10, 30, size=len(df))  # ms
    
    # Some percentage of handovers fail
    df['handover_success'] = np.random.choice([True, False], size=len(df), p=[0.95, 0.05])
    
    return df

def prepare_sequences(df, sequence_length=20):
    # Group by user
    grouped = df.groupby('user_id')

    sequence_data = []
    labels = []

    for user_id, user_data in grouped:
        user_data = user_data.sort_values('timestamp')

        # Create sequences
        for i in range(len(user_data) - sequence_length):
            sequence = user_data.iloc[i:i+sequence_length]
            target = user_data.iloc[i+sequence_length]

            # Feature vector (normalize numeric features)
            features = sequence[['x', 'y', 'velocity', 'heading', 'signal_strength', 
                               'sinr', 'network_load', 'throughput_mbps']].values

            # Min-max normalize 
            min_vals = features.min(axis=0, keepdims=True)
            max_vals = features.max(axis=0, keepdims=True)
            eps = 1e-8  # Avoid division by zero
            normalized_features = (features - min_vals) / (max_vals - min_vals + eps)

            # Target - will the user need a handover in the next step?
            handover_label = 1 if target['handover_needed'] else 0

            sequence_data.append(normalized_features)
            labels.append(handover_label)

    return np.array(sequence_data), np.array(labels)

def create_balanced_dataset(df):
    """Create a balanced dataset with a 1:3 ratio of handover to non-handover samples"""
    handover_samples = df[df['handover_needed'] == True]
    non_handover_samples = df[df['handover_needed'] == False].sample(
        n=len(handover_samples)*3,  # 1:3 ratio
        random_state=42
    )
    return pd.concat([handover_samples, non_handover_samples]).sort_values('timestamp') 