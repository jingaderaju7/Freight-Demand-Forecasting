"""
==============================================================================
Freight Demand Forecasting System - Model Training Module
==============================================================================

This script handles:
  1. Training two regression models (Linear Regression & Random Forest)
  2. Evaluating using MAE, RMSE, and R² score
  3. Comparing model performance
  4. Saving the best model using pickle
  5. Generating visualizations (actual vs predicted, feature importance)

Author: AI Engineer
==============================================================================
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Font configuration for charts
try:
    matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/english/calibri-italic.ttf')
except FileNotFoundError:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Calibri', 'FreeSans']
plt.rcParams['axes.unicode_minus'] = False


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
    y_train : array-like, shape (n_samples,)

    Returns
    -------
    trained LinearRegression model
    """
    print("\n--- Training Linear Regression ---")
    model = LinearRegression()
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"  Training time: {elapsed:.4f} seconds")
    print(f"  Coefficients: {model.coef_[:5]}... (showing first 5)")
    print(f"  Intercept: {model.intercept_:.2f}")
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 200, random_state: int = 42):
    """
    Train a Random Forest Regressor model.

    Parameters
    ----------
    X_train      : array-like, shape (n_samples, n_features)
    y_train      : array-like, shape (n_samples,)
    n_estimators  : int, number of trees (default 200)
    random_state  : int, random seed

    Returns
    -------
    trained RandomForestRegressor model
    """
    print("\n--- Training Random Forest Regressor ---")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores for faster training
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"  Training time: {elapsed:.4f} seconds")
    print(f"  Number of trees: {n_estimators}")
    print(f"  Feature importances range: {model.feature_importances_.min():.4f} - "
          f"{model.feature_importances_.max():.4f}")
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a regression model and return metrics.

    Parameters
    ----------
    model      : trained sklearn regressor
    X_test     : array-like, test features
    y_test     : array-like, test target
    model_name : str, name for display

    Returns
    -------
    dict with keys: mae, rmse, r2, predictions
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {model_name} Evaluation ---")
    print(f"  MAE  (Mean Absolute Error):  {mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  R² Score:                    {r2:.4f}")
    print(f"  Prediction range: {y_pred.min():.1f} - {y_pred.max():.1f}")
    print(f"  Actual range:      {y_test.min():.1f} - {y_test.max():.1f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": y_pred,
        "model_name": model_name
    }


def compare_models(results: list) -> str:
    """
    Compare model results and select the best one based on RMSE.

    Parameters
    ----------
    results : list of dicts from evaluate_model()

    Returns
    -------
    str : name of the best performing model
    """
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON")
    print("=" * 65)
    print(f"{'Model':<30} {'MAE':<12} {'RMSE':<12} {'R²':<10}")
    print("-" * 65)

    best_model = None
    best_rmse = float("inf")

    for r in results:
        print(f"{r['model_name']:<30} {r['mae']:<12.4f} {r['rmse']:<12.4f} {r['r2']:<10.4f}")
        if r["rmse"] < best_rmse:
            best_rmse = r["rmse"]
            best_model = r["model_name"]

    print("-" * 65)
    print(f"  >>> Best Model: {best_model} (RMSE = {best_rmse:.4f})")
    print("=" * 65)
    return best_model


def save_model(model, model_name: str):
    """
    Save a trained model to disk using pickle.

    Parameters
    ----------
    model      : trained sklearn model
    model_name : str, filename (without .pkl extension)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[INFO] Model saved to: {path}")
    return path


def plot_actual_vs_predicted(y_test, y_pred, model_name: str, save_path: str):
    """
    Generate an Actual vs Predicted scatter/line plot.

    Parameters
    ----------
    y_test     : array-like, actual target values
    y_pred     : array-like, predicted target values
    model_name : str, name for the plot title
    save_path  : str, where to save the PNG
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Scatter plot of actual vs predicted
    ax.scatter(y_test, y_pred, alpha=0.4, color="#2196F3", edgecolors="#1565C0",
               s=30, label="Predictions")

    # Perfect prediction line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label="Perfect Prediction (y=x)")

    ax.set_xlabel("Actual Order Volume", fontsize=13, fontweight="bold")
    ax.set_ylabel("Predicted Order Volume", fontsize=13, fontweight="bold")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Actual vs Predicted plot saved: {save_path}")


def plot_prediction_timeline(y_test, y_pred, model_name: str, save_path: str):
    """
    Generate a timeline plot showing actual vs predicted over test samples.

    Parameters
    ----------
    y_test     : array-like, actual values
    y_pred     : array-like, predicted values
    model_name : str, name for title
    save_path  : str, file save location
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    x_axis = range(len(y_test))
    ax.plot(x_axis, y_test.values if hasattr(y_test, 'values') else y_test,
            color="#333333", linewidth=1, alpha=0.8, label="Actual")
    ax.plot(x_axis, y_pred, color="#E91E63", linewidth=1, alpha=0.8, label="Predicted")

    ax.set_xlabel("Test Sample Index", fontsize=13, fontweight="bold")
    ax.set_ylabel("Order Volume", fontsize=13, fontweight="bold")
    ax.set_title(f"Prediction Timeline — {model_name}", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Timeline plot saved: {save_path}")


def plot_feature_importance(model, feature_names: list, save_path: str):
    """
    Generate a horizontal bar chart of Random Forest feature importances.

    Parameters
    ----------
    model         : trained RandomForestRegressor (must have feature_importances_)
    feature_names : list of str, column names
    save_path     : str, where to save the PNG
    """
    if not hasattr(model, "feature_importances_"):
        print("[WARN] Model has no feature_importances_ attribute — skipping plot.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    importances = model.feature_importances_
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Color gradient from light to dark
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_names)))

    bars = ax.barh(range(len(sorted_names)), sorted_values[::-1], color=colors[::-1],
                   edgecolor="#333333", linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=11)

    ax.set_xlabel("Importance Score", fontsize=13, fontweight="bold")
    ax.set_title("Random Forest — Feature Importance", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, sorted_values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature importance plot saved: {save_path}")


def plot_model_comparison(results: list, save_path: str):
    """
    Generate a grouped bar chart comparing MAE and RMSE across models.

    Parameters
    ----------
    results   : list of dicts with 'model_name', 'mae', 'rmse'
    save_path : str, where to save the PNG
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    names = [r["model_name"] for r in results]
    mae_vals = [r["mae"] for r in results]
    rmse_vals = [r["rmse"] for r in results]

    x = np.arange(len(names))
    width = 0.3

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, mae_vals, width, label="MAE",
                   color="#42A5F5", edgecolor="#1565C0", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, rmse_vals, width, label="RMSE",
                   color="#EF5350", edgecolor="#B71C1C", linewidth=0.8)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Error", fontsize=13, fontweight="bold")
    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Model comparison plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def training_pipeline(X_train, X_test, y_train, y_test, feature_names: list):
    """
    Full training pipeline: train models → evaluate → compare → save → visualize.

    Parameters
    ----------
    X_train       : scaled training features
    X_test        : scaled test features
    y_train       : training target
    y_test        : test target
    feature_names : list of feature column names
    """
    print("\n" + "=" * 65)
    print("  MODEL TRAINING PIPELINE")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Train Linear Regression
    # ------------------------------------------------------------------
    lr_model = train_linear_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression")

    # ------------------------------------------------------------------
    # Step 2: Train Random Forest
    # ------------------------------------------------------------------
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # ------------------------------------------------------------------
    # Step 3: Compare models and select best
    # ------------------------------------------------------------------
    all_results = [lr_results, rf_results]
    best_name = compare_models(all_results)

    # Select the best model object
    models = {"Linear Regression": lr_model, "Random Forest": rf_model}
    best_model = models[best_name]

    # ------------------------------------------------------------------
    # Step 4: Save the best model
    # ------------------------------------------------------------------
    safe_name = best_name.lower().replace(" ", "_")
    model_path = save_model(best_model, f"best_model_{safe_name}")

    # Also save the Random Forest separately for feature importance in the app
    rf_path = save_model(rf_model, "random_forest_model")

    # ------------------------------------------------------------------
    # Step 5: Generate visualizations
    # ------------------------------------------------------------------
    print("\n[INFO] Generating visualizations...")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Actual vs Predicted for the best model
    plot_actual_vs_predicted(
        y_test, best_model.predict(X_test), best_name,
        os.path.join(OUTPUTS_DIR, "actual_vs_predicted_best_model.png")
    )

    # Prediction timeline for the best model
    plot_prediction_timeline(
        y_test, best_model.predict(X_test), best_name,
        os.path.join(OUTPUTS_DIR, "prediction_timeline_best_model.png")
    )

    # Feature importance (Random Forest specific)
    plot_feature_importance(
        rf_model, feature_names,
        os.path.join(OUTPUTS_DIR, "feature_importance.png")
    )

    # Model comparison bar chart
    plot_model_comparison(
        all_results,
        os.path.join(OUTPUTS_DIR, "model_comparison.png")
    )

    # Also generate plots for each individual model
    for r in all_results:
        safe = r["model_name"].lower().replace(" ", "_")
        plot_actual_vs_predicted(
            y_test, r["predictions"], r["model_name"],
            os.path.join(OUTPUTS_DIR, f"actual_vs_predicted_{safe}.png")
        )

    print("\n" + "=" * 65)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 65)
    print(f"\n  Best model: {best_name}")
    print(f"  Saved to:   {model_path}")
    print(f"  Outputs:    {OUTPUTS_DIR}/")

    return best_model, best_name, all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline

    csv_file = os.path.join(PROJECT_ROOT, "data",
                            "logistics_dataset_2020_2024.csv")
    X_train, X_test, y_train, y_test, features, scaler = preprocess_pipeline(csv_file)
    best_model, best_name, results = training_pipeline(
        X_train, X_test, y_train, y_test, features
    )
