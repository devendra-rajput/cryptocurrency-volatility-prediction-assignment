"""
app.py
------
Streamlit web app for Cryptocurrency Volatility Prediction.

Run:
    streamlit run deployment/app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Volatility Predictor",
    page_icon="📈",
    layout="wide"
)

MODELS_DIR    = "models"
FEATURES_PATH = "data/crypto_features.csv"

FEATURE_COLS = [
    "log_return", "volume_change", "bb_bandwidth", "atr",
    "liquidity_ratio", "close_to_ma7", "close_to_ma30",
    "volatility_lag_1", "volatility_lag_3", "volatility_lag_7"
]

# ─── Load model (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    path = (
        f"{MODELS_DIR}/xgboost_tuned.json"
        if os.path.exists(f"{MODELS_DIR}/xgboost_tuned.json")
        else f"{MODELS_DIR}/xgboost.json"
    )
    model.load_model(path)
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    return model, scaler


@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH, parse_dates=["date"])


# ─── Header ──────────────────────────────────────────────────────────────────
st.title("📈 Cryptocurrency Volatility Predictor")
st.markdown(
    "Predict **annualised rolling volatility** for any cryptocurrency "
    "in the dataset using a trained XGBoost model."
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Settings")

try:
    df = load_features()
    model, scaler = load_model()

    symbols  = sorted(df["symbol"].unique().tolist())
    selected = st.sidebar.selectbox("Select Cryptocurrency", symbols, index=0)

    coin_df = df[df["symbol"] == selected].sort_values("date")

    min_date = coin_df["date"].min().date()
    max_date = coin_df["date"].max().date()
    selected_date = st.sidebar.date_input(
        "Select Date", value=max_date, min_value=min_date, max_value=max_date
    )

    # ─── Main area ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    row = coin_df[coin_df["date"] == pd.to_datetime(selected_date)]

    if row.empty:
        st.warning(f"No data available for {selected} on {selected_date}.")
    else:
        feat_vals = row[FEATURE_COLS].iloc[0].to_dict()
        feat_arr  = scaler.transform([[feat_vals[c] for c in FEATURE_COLS]])
        pred_vol  = float(model.predict(feat_arr)[0])
        actual_vol = float(row["volatility"].iloc[0])

        col1.metric("📊 Predicted Volatility", f"{pred_vol:.4f}")
        col2.metric("📌 Actual Volatility",    f"{actual_vol:.4f}")
        col3.metric("💰 Close Price (USD)",    f"${row['close'].iloc[0]:,.2f}")

    # ─── Historical volatility chart ─────────────────────────────────────────
    st.subheader(f"Historical Volatility — {selected}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(coin_df["date"], coin_df["volatility"], linewidth=1.2, color="steelblue", label="Actual")

    # Overlay predictions for all historical rows
    valid = coin_df.dropna(subset=FEATURE_COLS)
    if len(valid) > 0:
        X_hist = scaler.transform(valid[FEATURE_COLS].values)
        y_hist = model.predict(X_hist)
        ax.plot(valid["date"], y_hist, linewidth=1, color="tomato",
                linestyle="--", alpha=0.7, label="Predicted")

    ax.axvline(pd.to_datetime(selected_date), color="gold", linestyle=":", linewidth=1.5,
               label="Selected date")
    ax.set(xlabel="Date", ylabel="Annualised Volatility")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # ─── Feature table ───────────────────────────────────────────────────────
    with st.expander("🔍 Input Features for Selected Date"):
        feat_df = pd.DataFrame(feat_vals, index=["Value"]).T
        st.dataframe(feat_df.style.format("{:.6f}"))

    # ─── Price + Bollinger Bands ──────────────────────────────────────────────
    st.subheader("Price & Bollinger Bands")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(coin_df["date"], coin_df["close"],    color="gold",      linewidth=1.2, label="Close")
    ax2.plot(coin_df["date"], coin_df["bb_upper"], color="tomato",    linewidth=0.7, linestyle="--", label="BB Upper")
    ax2.plot(coin_df["date"], coin_df["bb_lower"], color="royalblue", linewidth=0.7, linestyle="--", label="BB Lower")
    ax2.fill_between(coin_df["date"], coin_df["bb_lower"], coin_df["bb_upper"],
                     alpha=0.08, color="gray")
    ax2.set(xlabel="Date", ylabel="Price (USD)")
    ax2.legend()
    st.pyplot(fig2)
    plt.close()

except FileNotFoundError as e:
    st.error(
        f"Required file not found: {e}\n\n"
        "Please run the full pipeline first:\n"
        "```\npython src/pipeline.py\n```"
    )
