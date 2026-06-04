"""
model_training.py
-----------------
Trains and compares three models for cryptocurrency volatility prediction:
  1. Random Forest Regressor
  2. XGBoost Regressor
  3. LSTM (deep learning time-series model)

The best model (by validation RMSE) is saved to models/.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

FEATURES_PATH = "data/crypto_features.csv"
MODELS_DIR    = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Features used for training
FEATURE_COLS = [
    "log_return", "volume_change", "bb_bandwidth", "atr",
    "liquidity_ratio", "close_to_ma7", "close_to_ma30",
    "volatility_lag_1", "volatility_lag_3", "volatility_lag_7"
]
TARGET_COL = "volatility"

RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare(path: str = FEATURES_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Chronological split (no shuffle — time-series)
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler


def evaluate(name: str, y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n[{name}]  RMSE={rmse:.5f}  MAE={mae:.5f}  R²={r2:.4f}")
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Random Forest
# ─────────────────────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    print("\n--- Training Random Forest ---")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")
    return evaluate("Random Forest", y_test, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — XGBoost
# ─────────────────────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    print("\n--- Training XGBoost ---")
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=RANDOM_STATE,
        eval_metric="rmse", early_stopping_rounds=20,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    y_pred = model.predict(X_test)
    model.save_model(f"{MODELS_DIR}/xgboost.json")
    return evaluate("XGBoost", y_test, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — LSTM
# ─────────────────────────────────────────────────────────────────────────────
def build_sequences(X, y, seq_len: int = 30):
    """Reshape flat feature matrix into (samples, timesteps, features) for LSTM."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train, X_test, y_test, seq_len: int = 30) -> dict:
    print("\n--- Training LSTM ---")
    X_tr_seq, y_tr_seq = build_sequences(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = build_sequences(X_test,  y_test,  seq_len)

    n_features = X_tr_seq.shape[2]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_tr_seq, y_tr_seq,
        epochs=100, batch_size=64,
        validation_split=0.1,
        callbacks=[es],
        verbose=1
    )
    y_pred = model.predict(X_te_seq).flatten()
    model.save(f"{MODELS_DIR}/lstm_model.h5")
    return evaluate("LSTM", y_te_seq, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Run all models and save results
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = load_and_prepare()

    results = []
    results.append(train_random_forest(X_train, y_train, X_test, y_test))
    results.append(train_xgboost(X_train, y_train, X_test, y_test))
    results.append(train_lstm(X_train, y_train, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{MODELS_DIR}/model_comparison.csv", index=False)

    best = results_df.iloc[0]["model"]
    print(f"\nBest model: {best}")
