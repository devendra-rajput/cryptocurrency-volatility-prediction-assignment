"""
data_loader.py
--------------
Loads the Cryptocurrency Historical Prices Dataset from a CSV file,
performs initial inspection, and saves a cleaned copy for downstream use.
"""

import pandas as pd
import numpy as np
import os

RAW_DATA_PATH  = "data/crypto_historical_prices.csv"
CLEAN_DATA_PATH = "data/crypto_cleaned.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV data and parse the date column."""
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns from {path}")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick overview of the dataset."""
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- Columns ---")
    print(df.columns.tolist())
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Sample Records ---")
    print(df.head(5))
    print("\n--- Basic Statistics ---")
    print(df.describe())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, remove duplicates, ensure correct data types,
    and filter out rows with non-positive prices or volume.
    """
    # Sort by symbol and date
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Drop fully duplicate rows
    before = len(df)
    df = df.drop_duplicates(subset=["date", "symbol"])
    print(f"Removed {before - len(df)} duplicate rows")

    # Drop rows where essential price columns are all missing
    essential_cols = ["open", "high", "low", "close", "volume"]
    df = df.dropna(subset=essential_cols)

    # Fill remaining missing market_cap with per-symbol median
    df["market_cap"] = df.groupby("symbol")["market_cap"].transform(
        lambda x: x.fillna(x.median())
    )

    # Remove rows with non-positive close prices or zero volume
    df = df[(df["close"] > 0) & (df["volume"] > 0)]

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume", "market_cap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.reset_index(drop=True)

    print(f"Clean dataset: {len(df):,} rows remaining")
    return df


def save_clean_data(df: pd.DataFrame, path: str = CLEAN_DATA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to {path}")


if __name__ == "__main__":
    df_raw   = load_raw_data()
    inspect_data(df_raw)
    df_clean = clean_data(df_raw)
    save_clean_data(df_clean)
