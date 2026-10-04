"""
predict.py
----------
Loads the saved model and scaler, then exposes a single
`predict_price()` function used by both the Flask API and
any ad-hoc scripts.

Usage (standalone test):
    python src/predict.py
"""

import os
import sys

import joblib
import numpy as np

# Allow imports from the project root when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import FEATURE_NAMES

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "house_price_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def _load_artefacts():
    """Load (and cache) the model and scaler from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run  python src/train.py  first to train and save the model."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_price(features: dict) -> float:
    """
    Predict the median house value for one sample.

    Parameters
    ----------
    features : dict
        Keys must match FEATURE_NAMES (case-sensitive):
            MedInc, HouseAge, AveRooms, AveBedrms,
            Population, AveOccup, Latitude, Longitude

    Returns
    -------
    float
        Predicted median house value in US dollars.
    """
    model, scaler = _load_artefacts()

    # Build a 1-row array in the correct feature order
    row = np.array([[features[name] for name in FEATURE_NAMES]])

    # Scale using the same scaler fitted during training
    row_scaled = scaler.transform(row)

    # Model predicts in units of $100,000 — convert to full dollars
    prediction_100k = model.predict(row_scaled)[0]
    return float(prediction_100k * 100_000)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "MedInc":     8.3252,
        "HouseAge":   41.0,
        "AveRooms":   6.984,
        "AveBedrms":  1.024,
        "Population": 322.0,
        "AveOccup":   2.555,
        "Latitude":   37.88,
        "Longitude": -122.23,
    }
    price = predict_price(sample)
    print(f"Sample prediction: ${price:,.0f}")
