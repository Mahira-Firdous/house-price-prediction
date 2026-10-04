"""
preprocess.py
-------------
Loads the California Housing dataset, builds a clean DataFrame,
scales the features with StandardScaler, and splits into train/test sets.

Run this file directly to print a quick sanity-check summary:
    python src/preprocess.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Human-readable column names
FEATURE_NAMES = [
    "MedInc",      # Median income in block group
    "HouseAge",    # Median house age in block group
    "AveRooms",    # Average number of rooms per household
    "AveBedrms",   # Average number of bedrooms per household
    "Population",  # Block group population
    "AveOccup",    # Average number of household members
    "Latitude",    # Block group latitude
    "Longitude",   # Block group longitude
]

TARGET_NAME = "MedHouseVal"  # Median house value (in $100,000s)


def load_data() -> pd.DataFrame:
    """
    Fetch the California Housing dataset from sklearn and return it
    as a single Pandas DataFrame (features + target column).
    """
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=FEATURE_NAMES)
    df[TARGET_NAME] = housing.target
    return df


def get_processed_data(test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline:
      1. Load raw data
      2. Separate features (X) from target (y)
      3. Split into train / test sets
      4. Fit a StandardScaler on the training features only
      5. Transform both train and test features

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, df
      - df is the full raw DataFrame (useful for EDA)
      - scaler is the fitted StandardScaler (save it alongside the model)
    """
    df = load_data()

    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, df = get_processed_data()
    print("Dataset shape     :", df.shape)
    print("Training samples  :", X_train.shape[0])
    print("Test samples      :", X_test.shape[0])
    print("Missing values    :", df.isnull().sum().sum())
    print("\nFeature stats (raw):")
    print(df[FEATURE_NAMES].describe().round(2))
