"""
Core implementation of model training functionality.
"""

import os
import tensorflow as tf
from datetime import datetime
import pickle
import yaml
from sklearn.model_selection import train_test_split


def train_model(X_train, y_train, X_val, y_val, model_config, training_config):
    """
    Train a mobility prediction model with the given data and configuration.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_config: Model architecture configuration
        training_config: Training hyperparameter configuration
        
    Returns:
        Trained model and training history
    """
    from src.model import build_mobility_prediction_model
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_mobility_prediction_model(
        input_shape=input_shape,
        lstm_units=model_config.get('lstm_units', 64),
        dropout_rate=model_config.get('dropout_rate', 0.3)
    )
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=training_config.get('early_stopping_patience', 10),
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=training_config.get('lr_reduction_factor', 0.5),
            patience=training_config.get('lr_patience', 5),
            min_lr=training_config.get('min_lr', 0.0001)
        )
    ]
    
    # Add model checkpoint callback if output_dir is provided
    if 'checkpoint_dir' in training_config:
        os.makedirs(training_config['checkpoint_dir'], exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(training_config['checkpoint_dir'], 'model_checkpoint.h5'),
                save_best_only=True
            )
        )
    
    # Add TensorBoard callback if log_dir is provided
    if 'log_dir' in training_config:
        log_dir = os.path.join(
            training_config['log_dir'], 
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(log_dir, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=training_config.get('epochs', 50),
        batch_size=training_config.get('batch_size', 64),
        callbacks=callbacks,
        verbose=training_config.get('verbose', 1)
    )
    
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Evaluate on test set
    test_results = model.evaluate(X_test, y_test, verbose=1)
    
    # Create results dictionary
    results = {name: float(value) for name, value in zip(model.metrics_names, test_results)}
    
    return results


def prepare_train_val_test_split(X, y, validation_split=0.15, test_split=0.15, random_seed=42):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X: Feature data
        y: Target data
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split off the test set
    test_val_size = validation_split + test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_size, random_state=random_seed
    )
    
    # Then split the remaining data into validation and test
    val_ratio = validation_split / test_val_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_ratio), random_state=random_seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_model_artifacts(model, history, scaler, feature_names, test_results, config, output_dir):
    """
    Save all model artifacts.
    
    Args:
        model: Trained model
        history: Training history
        scaler: Feature scaler
        feature_names: List of feature names
        test_results: Test evaluation results
        config: Training configuration
        output_dir: Directory to save artifacts
    
    Returns:
        Dictionary of file paths for saved artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts = {}
    
    # Save Keras H5 model
    model_path = os.path.join(output_dir, f'mobility_prediction_model_{timestamp}.h5')
    model.save(model_path)
    artifacts['model_h5_timestamped'] = model_path
    
    # Also save a version without timestamp for easier reference
    standard_model_path = os.path.join(output_dir, 'mobility_prediction_model.h5')
    model.save(standard_model_path)
    artifacts['model_h5'] = standard_model_path
    
    # Save TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(output_dir, f'mobility_prediction_model_{timestamp}.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    artifacts['model_tflite_timestamped'] = tflite_path
    
    # Also save a version without timestamp
    standard_tflite_path = os.path.join(output_dir, 'mobility_prediction_model.tflite')
    with open(standard_tflite_path, 'wb') as f:
        f.write(tflite_model)
    artifacts['model_tflite'] = standard_tflite_path
    
    # Save scaler for preprocessing new data
    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    artifacts['scaler'] = scaler_path
    
    # Save feature names for reference
    feature_names_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    artifacts['feature_names'] = feature_names_path
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    artifacts['history'] = history_path
    
    # Save test results
    results_path = os.path.join(output_dir, 'test_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(test_results, f)
    artifacts['test_results'] = results_path
    
    # Save training configuration for reference
    config_path = os.path.join(output_dir, f'training_config_{timestamp}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    artifacts['config'] = config_path
    
    return artifacts