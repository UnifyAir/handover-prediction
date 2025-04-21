"""
Model architecture definition for mobility prediction.
"""

import tensorflow as tf

def build_mobility_prediction_model(input_shape, lstm_units=64, dropout_rate=0.3):
    """
    Build and compile the mobility prediction model.
    
    Args:
        input_shape: Shape of input sequences (sequence_length, n_features)
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            units=lstm_units,
            input_shape=input_shape,
            return_sequences=False
        ),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(input_shape[1] * 5)  # 5 timesteps prediction
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mse',
        metrics=['mae']
    )
    
    return model 