#!/usr/bin/env python3
"""
Command-line script for making handover predictions with the trained model.
"""

import os
import argparse
import yaml
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime

# Import from project modules
import sys
# Add the project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from data.data_utils import load_mobility_data
from src.preprocessing import prepare_data_for_inference


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make handover predictions using the trained model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file (.h5 or .tflite)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data file (.pkl or .csv)')
    parser.add_argument('--scaler', type=str, default=None,
                        help='Path to feature scaler file (.pkl)')
    parser.add_argument('--config', type=str, default='configs/inference_config.yaml',
                        help='Path to inference configuration file')
    parser.add_argument('--output', type=str, default='predictions',
                        help='Directory to save prediction results')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold (overrides config)')
    parser.add_argument('--user_id', type=str, default=None,
                        help='Filter predictions for specific user')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    # Convert relative path to absolute if needed
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_path):
    """Load either H5 or TFLite model."""
    if model_path.endswith('.h5'):
        return tf.keras.models.load_model(model_path)
    elif model_path.endswith('.tflite'):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    else:
        raise ValueError(f"Unsupported model format: {model_path}")


def predict_with_h5_model(model, X):
    """Make predictions using H5 model."""
    return model.predict(X)


def predict_with_tflite_model(interpreter, X):
    """Make predictions using TFLite model."""
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


def main():
    """Main entry point for prediction script."""
    # Parse command line arguments
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration")
        config = {
            'inference': {
                'threshold': 0.5,
                'sequence_length': 20
            }
        }
    
    # Override config with command line arguments
    if args.threshold is not None:
        config['inference']['threshold'] = args.threshold
    
    # Load model
    print(f"Loading model from {args.model}")
    model = load_model(args.model)
    model_type = 'h5' if args.model.endswith('.h5') else 'tflite'
    
    # Load data
    print(f"Loading data from {args.data}")
    mobility_data = load_mobility_data(args.data)
    
    # Filter for specific user if requested
    if args.user_id:
        mobility_data = mobility_data[mobility_data['user_id'] == args.user_id]
        if len(mobility_data) == 0:
            print(f"No data found for user {args.user_id}")
            return
        print(f"Filtered data for user {args.user_id}: {len(mobility_data)} samples")
    
    # Load scaler if provided
    scaler = None
    if args.scaler:
        print(f"Loading feature scaler from {args.scaler}")
        with open(args.scaler, 'rb') as f:
            scaler = pickle.load(f)
    
    # Prepare data for inference
    print("Preparing data for inference...")
    sequence_length = config['inference'].get('sequence_length', 20)
    X, timestamps, user_ids, connected_cells = prepare_data_for_inference(
        mobility_data,
        sequence_length=sequence_length,
        scaler=scaler
    )
    
    if len(X) == 0:
        print("No valid sequences found in the data")
        return
    
    print(f"Prepared {len(X)} sequences for prediction")
    
    # Make predictions
    print("Making predictions...")
    if model_type == 'h5':
        predictions = predict_with_h5_model(model, X)
    else:
        predictions = predict_with_tflite_model(model, X)
    
    # Format predictions
    threshold = config['inference']['threshold']
    results = []
    
    for i, pred in enumerate(predictions):
        prob = float(pred[0]) if len(pred.shape) > 0 else float(pred)
        decision = "initiate_handover" if prob > threshold else "no_handover"
        
        results.append({
            "timestamp": timestamps[i].isoformat(),
            "user_id": user_ids[i],
            "connected_cell": connected_cells[i],
            "handover_probability": prob,
            "decision": decision,
            "threshold": threshold
        })
    
    # Save results
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.user_id:
        output_file = os.path.join(output_dir, f"predictions_{args.user_id}_{timestamp_str}.json")
    else:
        output_file = os.path.join(output_dir, f"predictions_{timestamp_str}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Predictions saved to {output_file}")
    
    # Print summary
    handover_count = sum(1 for r in results if r['decision'] == 'initiate_handover')
    print(f"\nPrediction Summary:")
    print(f"Total predictions: {len(results)}")
    print(f"Handovers predicted: {handover_count} ({handover_count/len(results):.2%})")
    print(f"Average handover probability: {np.mean([r['handover_probability'] for r in results]):.4f}")


if __name__ == "__main__":
    main()