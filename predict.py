"""
predict.py
----------
Inference utility. Loads the best trained model and scaler,
accepts a feature dictionary or DataFrame, and returns a
volatility prediction.

Usage (command line):
    python src/predict.py --symbol BTC --date 2024-01-15

Usage (as module):
    from src.predict import predict_volatility
    result = predict_volatility(feature_dict)
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

MODELS_DIR   = "models"
FEATURES_PATH = "data/crypto_features.csv"

FEATURE_COLS = [
    "log_return", "volume_change", "bb_bandwidth", "atr",
    "liquidity_ratio", "close_to_ma7", "close_to_ma30",
    "volatility_lag_1", "volatility_lag_3", "volatility_lag_7"
]


def load_model_and_scaler():
    """Load the best available XGBoost model and scaler."""
    import os
    model_path = (
        f"{MODELS_DIR}/xgboost_tuned.json"
        if os.path.exists(f"{MODELS_DIR}/xgboost_tuned.json")
        else f"{MODELS_DIR}/xgboost.json"
    )
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    print(f"Loaded model from {model_path}")
    return model, scaler


def predict_volatility(features: dict) -> float:
    """
    Predict annualised volatility from a feature dictionary.

    Parameters
    ----------
    features : dict
        Keys must match FEATURE_COLS.

    Returns
    -------
    float
        Predicted annualised volatility.
    """
    model, scaler = load_model_and_scaler()
    row = np.array([[features[c] for c in FEATURE_COLS]])
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    return float(pred)


def predict_from_history(symbol: str, date: str) -> float:
    """Look up a historical record by symbol and date and return prediction."""
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    row = df[(df["symbol"].str.upper() == symbol.upper()) &
             (df["date"] == pd.to_datetime(date))]
    if row.empty:
        raise ValueError(f"No data found for symbol={symbol}, date={date}")
    features = row[FEATURE_COLS].iloc[0].to_dict()
    return predict_volatility(features)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict cryptocurrency volatility")
    parser.add_argument("--symbol", type=str, required=True, help="Crypto symbol e.g. BTC")
    parser.add_argument("--date",   type=str, required=True, help="Date YYYY-MM-DD")
    args = parser.parse_args()

    pred = predict_from_history(args.symbol, args.date)
    print(f"\nPredicted Annualised Volatility for {args.symbol.upper()} on {args.date}: {pred:.6f}")
