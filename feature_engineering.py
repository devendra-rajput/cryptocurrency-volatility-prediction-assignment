"""
feature_engineering.py
-----------------------
Creates all model features from the cleaned dataset:
  - Daily log return and rolling volatility (target)
  - Moving averages (7-day, 14-day, 30-day)
  - Bollinger Bands (upper, lower, bandwidth)
  - Average True Range (ATR)
  - Liquidity ratio (volume / market_cap)
  - Lag features for the target variable
"""

import pandas as pd
import numpy as np

CLEAN_DATA_PATH    = "data/crypto_cleaned.csv"
FEATURES_DATA_PATH = "data/crypto_features.csv"

VOLATILITY_WINDOW = 14   # days of rolling std for volatility target
ATR_WINDOW        = 14
BB_WINDOW         = 20
BB_STD            = 2


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features per cryptocurrency symbol.
    Operates on a sorted (symbol, date) DataFrame.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    results = []

    for symbol, grp in df.groupby("symbol"):
        grp = grp.copy().reset_index(drop=True)

        # --- Daily log return ---
        grp["log_return"] = np.log(grp["close"] / grp["close"].shift(1))

        # --- Rolling volatility (target) ---
        grp["volatility"] = (
            grp["log_return"]
            .rolling(VOLATILITY_WINDOW)
            .std()
            * np.sqrt(252)   # annualise
        )

        # --- Moving averages ---
        for w in [7, 14, 30]:
            grp[f"ma_{w}"] = grp["close"].rolling(w).mean()

        # --- Price relative to MAs ---
        grp["close_to_ma7"]  = grp["close"] / grp["ma_7"]
        grp["close_to_ma30"] = grp["close"] / grp["ma_30"]

        # --- Bollinger Bands ---
        bb_mid             = grp["close"].rolling(BB_WINDOW).mean()
        bb_std             = grp["close"].rolling(BB_WINDOW).std()
        grp["bb_upper"]    = bb_mid + BB_STD * bb_std
        grp["bb_lower"]    = bb_mid - BB_STD * bb_std
        grp["bb_bandwidth"] = (grp["bb_upper"] - grp["bb_lower"]) / bb_mid

        # --- Average True Range (ATR) ---
        prev_close         = grp["close"].shift(1)
        tr1                = grp["high"] - grp["low"]
        tr2                = (grp["high"] - prev_close).abs()
        tr3                = (grp["low"]  - prev_close).abs()
        grp["true_range"]  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        grp["atr"]         = grp["true_range"].rolling(ATR_WINDOW).mean()

        # --- Liquidity ratio ---
        grp["liquidity_ratio"] = grp["volume"] / grp["market_cap"].replace(0, np.nan)

        # --- Volume change ---
        grp["volume_change"] = grp["volume"].pct_change()

        # --- Lag features for target ---
        for lag in [1, 3, 7]:
            grp[f"volatility_lag_{lag}"] = grp["volatility"].shift(lag)

        results.append(grp)

    df_feat = pd.concat(results, ignore_index=True)

    # Drop rows missing target or key features
    df_feat = df_feat.dropna(subset=["volatility"])
    df_feat = df_feat.reset_index(drop=True)

    print(f"Feature dataset: {len(df_feat):,} rows, {df_feat.shape[1]} columns")
    return df_feat


def save_features(df: pd.DataFrame, path: str = FEATURES_DATA_PATH) -> None:
    df.to_csv(path, index=False)
    print(f"Saved feature dataset to {path}")


if __name__ == "__main__":
    df_clean = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date"])
    df_feat  = compute_features(df_clean)
    save_features(df_feat)
