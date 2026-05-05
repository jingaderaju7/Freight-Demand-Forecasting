"""
==============================================================================
Freight Demand Forecasting System - Prediction Module
==============================================================================

This script handles:
  1. Loading a saved model and scaler from disk
  2. Accepting raw feature values (single prediction)
  3. Preprocessing input (scale, reshape)
  4. Returning a formatted prediction

Author: AI Engineer
==============================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")


def load_artifacts():
    """
    Load the saved model, scaler, and feature names from disk.

    Returns
    -------
    tuple : (model, scaler, feature_names)

    Raises
    ------
    FileNotFoundError if model or scaler files are missing.
    """
    # Try to load the best model (try Random Forest first as it typically wins)
    model_files = [
        os.path.join(MODEL_DIR, "best_model_random_forest.pkl"),
        os.path.join(MODEL_DIR, "best_model_linear_regression.pkl"),
        os.path.join(MODEL_DIR, "random_forest_model.pkl"),
    ]

    model = None
    model_path = None
    for path in model_files:
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            model_path = path
            break

    if model is None:
        raise FileNotFoundError(
            f"No trained model found in {MODEL_DIR}. "
            "Run train_model.py first to train and save a model."
        )

    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Run data_preprocessing.py first."
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load feature names
    feature_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Feature names not found at {feature_path}. "
            "Run data_preprocessing.py first."
        )
    with open(feature_path, "rb") as f:
        feature_names = pickle.load(f)

    print(f"[INFO] Loaded model from:      {model_path}")
    print(f"[INFO] Loaded scaler from:     {scaler_path}")
    print(f"[INFO] Feature names ({len(feature_names)}): {feature_names}")

    return model, scaler, feature_names


def predict_single(model, scaler, feature_names: list, input_dict: dict) -> float:
    """
    Make a single prediction from a dictionary of feature values.

    Parameters
    ----------
    model         : trained sklearn model
    scaler        : fitted StandardScaler
    feature_names : list of str, expected feature order
    input_dict    : dict, {feature_name: value}

    Returns
    -------
    float : predicted order volume
    """
    # Build the feature vector in the correct order
    try:
        features = [float(input_dict[name]) for name in feature_names]
    except KeyError as e:
        missing = e.args[0]
        raise ValueError(f"Missing required feature: '{missing}'. "
                         f"Expected features: {feature_names}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid feature value. All features must be numeric. Error: {e}")

    # Build a DataFrame with the same feature order and names used during scaler fitting.
    X = pd.DataFrame([features], columns=feature_names)
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]

    return float(prediction)


def format_prediction(value: float) -> str:
    """
    Format the predicted value for user-friendly display.

    Parameters
    ----------
    value : float, predicted order volume

    Returns
    -------
    str : formatted prediction string
    """
    # Round to nearest integer since order volume is a count
    rounded = round(value)
    lower = round(value * 0.92)   # approximate lower bound
    upper = round(value * 1.08)   # approximate upper bound

    return (f"Predicted Freight Demand: {rounded:,} orders\n"
            f"Estimated Range: {lower:,} — {upper:,} orders")


# ---------------------------------------------------------------------------
# Entry point: demo prediction with sample input
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("  PREDICTION DEMO")
    print("=" * 50)

    model, scaler, feature_names = load_artifacts()

    # Sample input matching a typical logistics day
    sample_input = {
        "holiday_flag": 0,
        "promo_flag": 0,
        "delivery_count": 400,
        "inventory_level": 15000,
        "lead_time_avg": 2.5,
        "warehouse_utilization": 0.65,
        "vehicle_availability": 30,
        "fuel_price_index": 85.0,
        "economic_index": 65.5,
        "temperature": 22.0,
        "rainfall_mm": 3.0,
        "lag_1_order": 420,
        "lag_7_order": 410,
        "lag_30_order": 395,
        "day": 15,
        "month": 6,
        "year": 2025,
        "day_of_week": 2,
        "is_weekend": 0
    }

    print(f"\nSample Input:")
    for k, v in sample_input.items():
        print(f"  {k:<25}: {v}")

    prediction = predict_single(model, scaler, feature_names, sample_input)
    print(f"\n{format_prediction(prediction)}")
