"""
eda.py
------
Exploratory Data Analysis for the Cryptocurrency Volatility project.
Generates and saves all EDA plots to reports/figures/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

FEATURES_PATH = "data/crypto_features.csv"
FIGURES_PATH  = "reports/figures"

os.makedirs(FIGURES_PATH, exist_ok=True)

TOP_N_SYMBOLS = 6   # number of cryptos to highlight in multi-line plots

# ── Palette ──────────────────────────────────────────────────────────────────
PALETTE = "tab10"
sns.set_theme(style="whitegrid", palette=PALETTE)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows for EDA")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset statistics summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Dataset Summary ===")
    print(f"Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Symbols    : {df['symbol'].nunique()}")
    print(f"Total rows : {len(df):,}")
    print("\n--- Numeric Statistics ---")
    print(df[["close", "volume", "market_cap", "volatility", "log_return"]].describe().round(4))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Closing price over time for top symbols
# ─────────────────────────────────────────────────────────────────────────────
def plot_price_trend(df: pd.DataFrame) -> None:
    top_syms = (
        df.groupby("symbol")["market_cap"].median()
        .nlargest(TOP_N_SYMBOLS).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    for sym in top_syms:
        subset = df[df["symbol"] == sym]
        ax.plot(subset["date"], subset["close"], label=sym, linewidth=1.4)
    ax.set(title="Closing Price Over Time (Top Cryptos by Market Cap)",
           xlabel="Date", ylabel="Close Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/01_price_trend.png", dpi=150)
    plt.close()
    print("Saved 01_price_trend.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rolling volatility over time
# ─────────────────────────────────────────────────────────────────────────────
def plot_volatility_trend(df: pd.DataFrame) -> None:
    top_syms = (
        df.groupby("symbol")["market_cap"].median()
        .nlargest(TOP_N_SYMBOLS).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    for sym in top_syms:
        subset = df[df["symbol"] == sym]
        ax.plot(subset["date"], subset["volatility"], label=sym, linewidth=1.2, alpha=0.8)
    ax.set(title="Annualised Rolling Volatility Over Time",
           xlabel="Date", ylabel="Volatility (annualised)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/02_volatility_trend.png", dpi=150)
    plt.close()
    print("Saved 02_volatility_trend.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Distribution of daily log returns
# ─────────────────────────────────────────────────────────────────────────────
def plot_return_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["log_return"].dropna(), bins=200, color="steelblue",
            edgecolor="none", alpha=0.8, density=True)
    from scipy.stats import norm
    mu, std = df["log_return"].mean(), df["log_return"].std()
    x = np.linspace(df["log_return"].quantile(0.001),
                    df["log_return"].quantile(0.999), 300)
    ax.plot(x, norm.pdf(x, mu, std), "r-", linewidth=2, label="Normal fit")
    ax.set(title="Distribution of Daily Log Returns",
           xlabel="Log Return", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/03_return_distribution.png", dpi=150)
    plt.close()
    print("Saved 03_return_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Correlation heatmap of key features
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = [
        "volatility", "log_return", "volume_change",
        "bb_bandwidth", "atr", "liquidity_ratio",
        "close_to_ma7", "close_to_ma30", "volatility_lag_1"
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/04_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved 04_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Average volatility per symbol (bar chart)
# ─────────────────────────────────────────────────────────────────────────────
def plot_avg_volatility_by_symbol(df: pd.DataFrame) -> None:
    avg_vol = (
        df.groupby("symbol")["volatility"].mean()
        .sort_values(ascending=False)
        .head(20)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    avg_vol.plot(kind="bar", ax=ax, color="coral", edgecolor="none")
    ax.set(title="Average Annualised Volatility by Cryptocurrency (Top 20)",
           xlabel="Symbol", ylabel="Mean Volatility")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/05_avg_volatility_by_symbol.png", dpi=150)
    plt.close()
    print("Saved 05_avg_volatility_by_symbol.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Volume vs Volatility scatter
# ─────────────────────────────────────────────────────────────────────────────
def plot_volume_vs_volatility(df: pd.DataFrame) -> None:
    sample = df[["volume", "volatility"]].dropna().sample(
        min(20000, len(df)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(np.log1p(sample["volume"]), sample["volatility"],
               alpha=0.15, s=8, color="teal")
    ax.set(title="Log Volume vs Annualised Volatility",
           xlabel="log(1 + Volume)", ylabel="Volatility")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/06_volume_vs_volatility.png", dpi=150)
    plt.close()
    print("Saved 06_volume_vs_volatility.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Bollinger Band width over time for Bitcoin
# ─────────────────────────────────────────────────────────────────────────────
def plot_bollinger_bandwidth(df: pd.DataFrame) -> None:
    btc = df[df["symbol"].str.upper() == "BTC"].copy()
    if btc.empty:
        btc = df[df["symbol"] == df["symbol"].value_counts().idxmax()].copy()

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(btc["date"], btc["close"],      color="gold",   linewidth=1.2, label="Close")
    axes[0].plot(btc["date"], btc["bb_upper"],   color="tomato", linewidth=0.8, linestyle="--", label="BB Upper")
    axes[0].plot(btc["date"], btc["bb_lower"],   color="royalblue", linewidth=0.8, linestyle="--", label="BB Lower")
    axes[0].fill_between(btc["date"], btc["bb_lower"], btc["bb_upper"], alpha=0.1, color="gray")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Bollinger Bands & Bandwidth — {btc['symbol'].iloc[0].upper()}")

    axes[1].plot(btc["date"], btc["bb_bandwidth"], color="purple", linewidth=1)
    axes[1].set_ylabel("BB Bandwidth")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/07_bollinger_bands.png", dpi=150)
    plt.close()
    print("Saved 07_bollinger_bands.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print_summary(df)
    plot_price_trend(df)
    plot_volatility_trend(df)
    plot_return_distribution(df)
    plot_correlation_heatmap(df)
    plot_avg_volatility_by_symbol(df)
    plot_volume_vs_volatility(df)
    plot_bollinger_bandwidth(df)
    print("\nAll EDA plots saved to reports/figures/")
