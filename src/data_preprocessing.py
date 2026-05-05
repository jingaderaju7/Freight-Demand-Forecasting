"""
==============================================================================
Freight Demand Forecasting System - Data Preprocessing Module
==============================================================================

This script handles:
  1. Loading the raw logistics CSV dataset
  2. Handling missing values (forward-fill, median, zero-fill)
  3. Feature engineering from the date column
  4. Dropping unnecessary columns
  5. Scaling / normalizing features
  6. Splitting into train/test sets

Author: AI Engineer
Dataset: logistics_dataset_2020_2024.csv (2020-2024 daily records)
==============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# ---------------------------------------------------------------------------
# Project root resolution – so paths work regardless of where the script runs
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the logistics dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe loaded from the CSV.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"[INFO] Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with domain-aware strategies.

    Strategy:
      - lag_1_order, lag_7_order, lag_30_order : forward-fill then backward-fill
        (these are time-series lag features; adjacent values are closest)
      - rolling_mean_7, rolling_std_7            : forward-fill then backward-fill
      - holiday_flag, promo_flag                 : fill with 0 (assume no holiday/promo)
      - All other numeric columns                : fill with column median
      - Any remaining NaN                        : fill with 0

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with no remaining NaN values.
    """
    missing_before = df.isnull().sum().sum()
    print(f"\n[INFO] Missing values before cleaning: {missing_before}")

    if missing_before == 0:
        print("[INFO] No missing values found — dataset is clean.")
        return df

    # --- Time-series columns: forward-fill then backward-fill ---
    ts_cols = ["lag_1_order", "lag_7_order", "lag_30_order",
               "rolling_mean_7", "rolling_std_7"]
    for col in ts_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].ffill().bfill()
            print(f"  [FIX] {col}: forward/backward filled")

    # --- Flag columns: assume 0 when missing ---
    flag_cols = ["holiday_flag", "promo_flag"]
    for col in flag_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(0)
            print(f"  [FIX] {col}: filled with 0")

    # --- Numeric columns: median imputation ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  [FIX] {col}: filled with median ({median_val:.2f})")

    # --- Final safety net: fill any remaining NaN with 0 ---
    if df.isnull().any().any():
        df = df.fillna(0)
        print("  [FIX] Remaining NaN values filled with 0")

    missing_after = df.isnull().sum().sum()
    print(f"[INFO] Missing values after cleaning:  {missing_after}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from the date column and prepare the final feature set.

    New features created:
      - day       : calendar day (1-31)
      - month     : calendar month (1-12)
      - year      : calendar year
      - day_of_week: Monday=0 … Sunday=6
      - is_weekend : 1 if Saturday/Sunday, else 0

    Columns dropped:
      - date       (raw string, replaced by engineered features)
      - quarter    (redundant with month)
      - Any target leakage features (rolling_mean_7, rolling_std_7)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with engineered features and cleaned columns.
    """
    print("\n[INFO] Engineering features from date column...")

    # --- Extract date components ---
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Drop columns that are not needed ---
    cols_to_drop = ["date", "quarter", "rolling_mean_7", "rolling_std_7"]
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"  [DROP] Removed columns: {existing_drops}")

    # --- Ensure target is last column for clarity ---
    if "order_volume" in df.columns:
        cols = [c for c in df.columns if c != "order_volume"] + ["order_volume"]
        df = df[cols]

    print(f"[INFO] Final feature set: {df.shape[1]} columns")
    print(f"  Features: {[c for c in df.columns if c != 'order_volume']}")
    return df


def preprocess_pipeline(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline: load → clean → engineer → scale → split.

    Parameters
    ----------
    csv_path        : str   – Path to the raw CSV file
    test_size       : float – Fraction of data for testing (default 0.2)
    random_state    : int   – Random seed for reproducibility

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    print("=" * 65)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 65)

    # Step 1: Load data
    df = load_data(csv_path)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Feature engineering
    df = engineer_features(df)

    # Step 4: Separate features and target
    if "order_volume" not in df.columns:
        raise ValueError("Target column 'order_volume' not found in dataset!")

    X = df.drop(columns=["order_volume"])
    y = df["order_volume"]
    feature_names = list(X.columns)

    print(f"\n[INFO] Features shape: {X.shape}")
    print(f"[INFO] Target shape:   {y.shape}")
    print(f"[INFO] Target stats — Mean: {y.mean():.1f}, Std: {y.std():.1f}, "
          f"Min: {y.min()}, Max: {y.max()}")

    # Step 5: Train-test split (chronological-aware via shuffle=False for time series)
    # Using shuffle=True here because dataset has enough randomness from various features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set:  {X_test.shape[0]} samples")

    # Step 6: Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for prediction use
    os.makedirs(MODEL_DIR, exist_ok=True)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Scaler saved to: {scaler_path}")

    # Save feature names for the Flask app
    feature_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    with open(feature_path, "wb") as f:
        pickle.dump(feature_names, f)
    print(f"[INFO] Feature names saved to: {feature_path}")

    print("\n" + "=" * 65)
    print("  PREPROCESSING COMPLETE")
    print("=" * 65)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


# ---------------------------------------------------------------------------
# Entry point: run preprocessing when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    csv_file = os.path.join(DATA_DIR, "logistics_dataset_2020_2024.csv")
    X_train, X_test, y_train, y_test, features, scaler = preprocess_pipeline(csv_file)
    print(f"\n[RESULT] X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[RESULT] Feature names: {features}")
