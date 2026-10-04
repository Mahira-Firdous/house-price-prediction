"""
train.py
--------
Trains two regression models on the California Housing dataset,
evaluates them, prints a comparison table, then saves the best
model (Random Forest) and the fitted scaler to the model/ directory.

Usage:
    python src/train.py
"""

import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow imports from the project root when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import get_processed_data

# Directory where the trained artefacts will be saved
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return a dict of evaluation metrics for a fitted model."""
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def print_metrics(results: list[dict]) -> None:
    """Pretty-print a comparison table of model metrics."""
    header = f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(f"{r['Model']:<25} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f}")
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train():
    print("Loading and preprocessing data ...")
    X_train, X_test, y_train, y_test, scaler, _ = get_processed_data()

    # ── 1. Baseline: Linear Regression ──────────────────────────────────────
    print("Training Linear Regression (baseline) ...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_metrics = evaluate("Linear Regression", lr, X_test, y_test)

    # ── 2. Main model: Random Forest Regressor ───────────────────────────────
    print("Training Random Forest Regressor ...")
    rf = RandomForestRegressor(
        n_estimators=100,   # 100 trees — good balance of speed vs accuracy
        random_state=42,
        n_jobs=-1,          # use all CPU cores
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test)

    # ── 3. Print comparison ──────────────────────────────────────────────────
    print_metrics([lr_metrics, rf_metrics])

    # ── 4. Save the best model and scaler ────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"[OK] Model saved  -> {MODEL_PATH}")
    print(f"[OK] Scaler saved -> {SCALER_PATH}")

    # Return for testing purposes
    return rf, scaler


if __name__ == "__main__":
    train()
