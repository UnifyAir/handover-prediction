"""
Core implementation for making predictions with the trained mobility prediction model.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime


def load_model(model_path):
    """
    Load a mobility prediction model from file.
    
    Args:
        model_path: Path to the model file (.h5 or .tflite)
        
    Returns:
        Loaded model and model type ('h5' or 'tflite')
    """
    if model_path.endswith('.h5'):
        return tf.keras.models.load_model(model_path), 'h5'
    elif model_path.endswith('.tflite'):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter, 'tflite'
    else:
        raise ValueError(f"Unsupported model format: {model_path}")


def predict_handovers(model, X, model_type='h5'):
    """
    Make handover predictions using the model.
    
    Args:
        model: Loaded model (H5 or TFLite interpreter)
        X: Input features (preprocessed sequences)
        model_type: Type of model ('h5' or 'tflite')
        
    Returns:
        Array of prediction probabilities
    """
    if model_type == 'h5':
        return predict_with_h5_model(model, X)
    elif model_type == 'tflite':
        return predict_with_tflite_model(model, X)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def predict_with_h5_model(model, X):
    """
    Make predictions using H5 model.
    
    Args:
        model: Loaded Keras model
        X: Input features
        
    Returns:
        Array of prediction probabilities
    """
    return model.predict(X)


def predict_with_tflite_model(interpreter, X):
    """
    Make predictions using TFLite model.
    
    Args:
        interpreter: TFLite interpreter
        X: Input features
        
    Returns:
        Array of prediction probabilities
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    
    # TFLite models process one sample at a time
    for i in range(len(X)):
        input_data = X[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])
    
    return np.array(predictions)


def format_predictions(predictions, timestamps, user_ids, connected_cells, threshold=0.5):
    """
    Format raw predictions into structured results.
    
    Args:
        predictions: Array of prediction probabilities
        timestamps: List of timestamps for each prediction
        user_ids: List of user IDs for each prediction
        connected_cells: List of connected cells for each prediction
        threshold: Threshold for handover decision
        
    Returns:
        List of dictionaries with prediction details
    """
    results = []
    
    for i, pred in enumerate(predictions):
        prob = float(pred[0]) if len(pred.shape) > 0 else float(pred)
        decision = "initiate_handover" if prob > threshold else "no_handover"
        
        results.append({
            "timestamp": timestamps[i].isoformat() if isinstance(timestamps[i], datetime) else timestamps[i],
            "user_id": user_ids[i],
            "connected_cell": connected_cells[i],
            "handover_probability": prob,
            "decision": decision,
            "threshold": threshold
        })
    
    return results


def predict_single_user(model, model_type, user_data, sequence_length, scaler=None, threshold=0.5):
    """
    Make predictions for a single user's recent data.
    
    Args:
        model: Loaded model
        model_type: Type of model ('h5' or 'tflite')
        user_data: DataFrame with user's recent mobility data
        sequence_length: Length of input sequence for the model
        scaler: Optional scaler for feature normalization
        threshold: Threshold for handover decision
        
    Returns:
        Dictionary with prediction results
    """
    from src.preprocessing import prepare_data_for_inference
    
    # Ensure we have enough data
    if len(user_data) < sequence_length:
        return {
            "error": "insufficient_data",
            "message": f"Need at least {sequence_length} data points for prediction",
            "available_points": len(user_data)
        }
    
    # Prepare data for inference
    X, timestamps, user_ids, connected_cells = prepare_data_for_inference(
        user_data,
        sequence_length=sequence_length,
        scaler=scaler,
        return_latest_only=True  # Only predict for the most recent sequence
    )
    
    if len(X) == 0:
        return {
            "error": "preprocessing_error",
            "message": "Failed to create valid sequence for prediction"
        }
    
    # Make prediction
    predictions = predict_handovers(model, X, model_type)
    
    # Format result
    results = format_predictions(
        predictions, 
        timestamps, 
        user_ids, 
        connected_cells, 
        threshold
    )
    
    if results:
        return results[0]  # Return the single prediction
    else:
        return {
            "error": "prediction_error",
            "message": "Failed to generate prediction"
        }


def real_time_prediction_service(model, model_type, sequence_buffer, sequence_length, scaler=None, threshold=0.5):
    """
    Streaming prediction service for real-time handover decisions.
    
    This function is designed to be called each time new measurement data arrives
    for a user, updating a buffer and making predictions when enough data is available.
    
    Args:
        model: Loaded model
        model_type: Type of model ('h5' or 'tflite')
        sequence_buffer: DataFrame containing the most recent measurements for a user
        sequence_length: Length of input sequence for the model
        scaler: Optional scaler for feature normalization
        threshold: Threshold for handover decision
        
    Returns:
        Dictionary with prediction result or None if more data is needed
    """
    # Check if we have enough data for a prediction
    if len(sequence_buffer) < sequence_length:
        return {
            "status": "buffering",
            "message": f"Collecting data: {len(sequence_buffer)}/{sequence_length} points"
        }
    
    # Keep only the most recent data points needed for the prediction
    if len(sequence_buffer) > sequence_length:
        sequence_buffer = sequence_buffer.iloc[-sequence_length:]
    
    # Make a prediction
    result = predict_single_user(
        model, 
        model_type, 
        sequence_buffer, 
        sequence_length, 
        scaler, 
        threshold
    )
    
    return result


def batch_prediction(model, model_type, mobility_data, sequence_length, scaler=None, threshold=0.5):
    """
    Generate predictions for a batch of mobility data.
    
    Args:
        model: Loaded model
        model_type: Type of model ('h5' or 'tflite')
        mobility_data: DataFrame with mobility data for multiple users/timepoints
        sequence_length: Length of input sequence for the model
        scaler: Optional scaler for feature normalization
        threshold: Threshold for handover decision
        
    Returns:
        List of dictionaries with prediction results
    """
    from src.preprocessing import prepare_data_for_inference
    
    # Prepare data for inference
    X, timestamps, user_ids, connected_cells = prepare_data_for_inference(
        mobility_data,
        sequence_length=sequence_length,
        scaler=scaler
    )
    
    if len(X) == 0:
        return []
    
    # Make predictions
    predictions = predict_handovers(model, X, model_type)
    
    # Format results
    results = format_predictions(
        predictions, 
        timestamps, 
        user_ids, 
        connected_cells, 
        threshold
    )
    
    return results


def compute_prediction_metrics(predictions, actual_handovers):
    """
    Compute performance metrics for predictions.
    
    Args:
        predictions: List of prediction dictionaries
        actual_handovers: Series or array of actual handover events (True/False)
        
    Returns:
        Dictionary with performance metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Extract predictions
    y_pred = [p['decision'] == 'initiate_handover' for p in predictions]
    y_prob = [p['handover_probability'] for p in predictions]
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(actual_handovers, y_pred),
        'precision': precision_score(actual_handovers, y_pred),
        'recall': recall_score(actual_handovers, y_pred),
        'f1_score': f1_score(actual_handovers, y_pred),
        'auc': roc_auc_score(actual_handovers, y_prob)
    }
    
    return metrics


def amf_handover_decision(prediction_result, current_network_state=None):
    """
    Make an AMF handover decision based on the prediction and network state.
    
    This function combines the ML prediction with current network information
    to make the final handover decision that would be implemented by the AMF.
    
    Args:
        prediction_result: Dictionary with prediction details
        current_network_state: Optional dictionary with current network load info
        
    Returns:
        Dictionary with handover decision details
    """
    # If the prediction doesn't recommend a handover, simply return that
    if prediction_result['decision'] != 'initiate_handover':
        return {
            'action': 'no_handover',
            'user_id': prediction_result['user_id'],
            'current_cell': prediction_result['connected_cell'],
            'reason': 'prediction_below_threshold',
            'probability': prediction_result['handover_probability']
        }
    
    # If we have network state info, we can make a more informed decision
    if current_network_state:
        # Get current cell load
        current_cell = prediction_result['connected_cell']
        current_load = current_network_state.get(current_cell, {}).get('load', 0.5)
        
        # Find best target cell based on prediction and network load
        candidate_cells = current_network_state.get('candidate_cells', {})
        
        if candidate_cells:
            # Find cell with lowest load
            best_cell = min(candidate_cells.items(), key=lambda x: x[1]['load'])
            target_cell = best_cell[0]
            target_load = best_cell[1]['load']
            
            # Only handover if target cell has lower load or prediction is very confident
            if target_load < current_load or prediction_result['handover_probability'] > 0.8:
                return {
                    'action': 'initiate_handover',
                    'user_id': prediction_result['user_id'],
                    'source_cell': current_cell,
                    'target_cell': target_cell,
                    'reason': 'predicted_mobility_and_load_optimization',
                    'probability': prediction_result['handover_probability'],
                    'current_cell_load': current_load,
                    'target_cell_load': target_load
                }
            else:
                return {
                    'action': 'no_handover',
                    'user_id': prediction_result['user_id'],
                    'current_cell': current_cell,
                    'reason': 'target_cell_not_optimal',
                    'probability': prediction_result['handover_probability']
                }
    
    # Without network state, use simple prediction-based decision
    return {
        'action': 'initiate_handover',
        'user_id': prediction_result['user_id'],
        'source_cell': prediction_result['connected_cell'],
        'reason': 'predicted_mobility',
        'probability': prediction_result['handover_probability']
    }