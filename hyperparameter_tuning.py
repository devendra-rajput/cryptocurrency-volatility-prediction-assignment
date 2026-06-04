"""
hyperparameter_tuning.py
------------------------
Optimises the XGBoost model using Optuna (Bayesian optimisation).
The best parameters are saved to models/best_xgb_params.json.
"""

import json
import numpy as np
import pandas as pd
import joblib
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_PATH = "data/crypto_features.csv"
MODELS_DIR    = "models"
TEST_SIZE     = 0.2

FEATURE_COLS = [
    "log_return", "volume_change", "bb_bandwidth", "atr",
    "liquidity_ratio", "close_to_ma7", "close_to_ma30",
    "volatility_lag_1", "volatility_lag_3", "volatility_lag_7"
]
TARGET_COL = "volatility"
N_TRIALS   = 50


def load_data():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler    = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators",    100, 600),
        "max_depth"       : trial.suggest_int("max_depth",         3,  10),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"       : trial.suggest_float("subsample",     0.5,  1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight",  1,   10),
        "gamma"           : trial.suggest_float("gamma",          0.0,  5.0),
        "objective"       : "reg:squarederror",
        "random_state"    : 42,
        "verbosity"       : 0,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    best = study.best_params
    print(f"\nBest RMSE : {study.best_value:.6f}")
    print("Best params:", best)

    with open(f"{MODELS_DIR}/best_xgb_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved best params to {MODELS_DIR}/best_xgb_params.json")

    # Retrain with best params
    best["objective"]    = "reg:squarederror"
    best["random_state"] = 42
    best["verbosity"]    = 0
    final_model = xgb.XGBRegressor(**best)
    final_model.fit(X_train, y_train)
    final_model.save_model(f"{MODELS_DIR}/xgboost_tuned.json")
    print("Saved tuned XGBoost model to models/xgboost_tuned.json")
