"""
model_evaluation.py
-------------------
Loads the trained models and the test split, generates evaluation
plots (actual vs predicted, residuals, feature importance), and
saves a final performance summary to reports/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

FEATURES_PATH  = "data/crypto_features.csv"
MODELS_DIR     = "models"
FIGURES_PATH   = "reports/figures"
os.makedirs(FIGURES_PATH, exist_ok=True)

FEATURE_COLS = [
    "log_return", "volume_change", "bb_bandwidth", "atr",
    "liquidity_ratio", "close_to_ma7", "close_to_ma30",
    "volatility_lag_1", "volatility_lag_3", "volatility_lag_7"
]
TARGET_COL = "volatility"
TEST_SIZE  = 0.2
sns.set_theme(style="whitegrid")


def load_test_data():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    split_idx  = int(len(X) * (1 - TEST_SIZE))
    X_test_raw = X[split_idx:]
    y_test     = y[split_idx:]

    scaler    = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    X_test_sc = scaler.transform(X_test_raw)
    return X_test_sc, y_test


def metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE" : mean_absolute_error(y_true, y_pred),
        "R2"  : r2_score(y_true, y_pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Actual vs Predicted plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_true, y_pred, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set(xlabel="Actual Volatility", ylabel="Predicted Volatility",
                title=f"{model_name} — Actual vs Predicted")
    axes[0].legend()

    # Time-series line (first 500 test points)
    n = min(500, len(y_true))
    axes[1].plot(y_true[:n],  label="Actual",    linewidth=1.2, alpha=0.9)
    axes[1].plot(y_pred[:n],  label="Predicted", linewidth=1.2, alpha=0.8, linestyle="--")
    axes[1].set(xlabel="Test Sample Index", ylabel="Volatility",
                title=f"{model_name} — First {n} Predictions")
    axes[1].legend()

    fig.tight_layout()
    fname = f"{FIGURES_PATH}/eval_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Residuals plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_residuals(y_true, y_pred, model_name: str) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color="coral")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Predicted", ylabel="Residual",
                title=f"{model_name} — Residuals")

    axes[1].hist(residuals, bins=100, color="coral", edgecolor="none", density=True)
    axes[1].set(xlabel="Residual", ylabel="Density",
                title="Residual Distribution")

    fig.tight_layout()
    fname = f"{FIGURES_PATH}/residuals_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance() -> None:
    model = xgb.XGBRegressor()
    model.load_model(f"{MODELS_DIR}/xgboost.json")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": FEATURE_COLS, "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color="mediumseagreen")
    ax.set(title="XGBoost Feature Importances", xlabel="Importance Score")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/feature_importance_xgb.png", dpi=150)
    plt.close()
    print("Saved feature_importance_xgb.png")


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison() -> None:
    comp_path = f"{MODELS_DIR}/model_comparison.csv"
    if not os.path.exists(comp_path):
        print("model_comparison.csv not found — run model_training.py first")
        return
    df = pd.read_csv(comp_path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        colors = ["tomato" if metric in ["RMSE", "MAE"] else "steelblue"] * len(df)
        ax.bar(df["model"], df[metric], color=colors, edgecolor="none")
        ax.set(title=metric, xlabel="Model", ylabel=metric)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/model_comparison.png", dpi=150)
    plt.close()
    print("Saved model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run evaluation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_test, y_test = load_test_data()

    all_metrics = []

    # Random Forest
    rf = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
    y_pred_rf = rf.predict(X_test)
    m = metrics(y_test, y_pred_rf); m["model"] = "Random Forest"; all_metrics.append(m)
    plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest")
    plot_residuals(y_test, y_pred_rf, "Random Forest")

    # XGBoost
    xgb_model = xgb.XGBRegressor(); xgb_model.load_model(f"{MODELS_DIR}/xgboost.json")
    y_pred_xgb = xgb_model.predict(X_test)
    m = metrics(y_test, y_pred_xgb); m["model"] = "XGBoost"; all_metrics.append(m)
    plot_actual_vs_predicted(y_test, y_pred_xgb, "XGBoost")
    plot_residuals(y_test, y_pred_xgb, "XGBoost")
    plot_feature_importance()

    plot_model_comparison()

    summary = pd.DataFrame(all_metrics)[["model", "RMSE", "MAE", "R2"]]
    print("\n=== Final Evaluation Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv("reports/model_performance_summary.csv", index=False)
