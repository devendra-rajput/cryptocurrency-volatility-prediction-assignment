"""
pipeline.py
-----------
End-to-end pipeline that runs all stages in sequence:
  1. Data loading & cleaning
  2. Feature engineering
  3. EDA (optional)
  4. Model training
  5. Model evaluation

Run:
    python src/pipeline.py [--skip-eda]
"""

import argparse
import time

# ─── Stage helpers ────────────────────────────────────────────────────────────

def run_data_loading():
    from src.data_loader import load_raw_data, clean_data, save_clean_data
    print("\n" + "="*55)
    print("STAGE 1 — Data Loading & Cleaning")
    print("="*55)
    df_raw   = load_raw_data()
    df_clean = clean_data(df_raw)
    save_clean_data(df_clean)


def run_feature_engineering():
    import pandas as pd
    from src.feature_engineering import compute_features, save_features
    print("\n" + "="*55)
    print("STAGE 2 — Feature Engineering")
    print("="*55)
    df_clean = pd.read_csv("data/crypto_cleaned.csv", parse_dates=["date"])
    df_feat  = compute_features(df_clean)
    save_features(df_feat)


def run_eda():
    print("\n" + "="*55)
    print("STAGE 3 — Exploratory Data Analysis")
    print("="*55)
    import src.eda as eda_module
    import pandas as pd
    df = eda_module.load_data()
    eda_module.print_summary(df)
    eda_module.plot_price_trend(df)
    eda_module.plot_volatility_trend(df)
    eda_module.plot_return_distribution(df)
    eda_module.plot_correlation_heatmap(df)
    eda_module.plot_avg_volatility_by_symbol(df)
    eda_module.plot_volume_vs_volatility(df)
    eda_module.plot_bollinger_bandwidth(df)


def run_training():
    from src.model_training import (
        load_and_prepare, train_random_forest,
        train_xgboost, train_lstm
    )
    import pandas as pd
    print("\n" + "="*55)
    print("STAGE 4 — Model Training")
    print("="*55)
    X_train, X_test, y_train, y_test, _ = load_and_prepare()
    results = []
    results.append(train_random_forest(X_train, y_train, X_test, y_test))
    results.append(train_xgboost(X_train, y_train, X_test, y_test))
    results.append(train_lstm(X_train, y_train, X_test, y_test))
    df_res = pd.DataFrame(results).sort_values("RMSE")
    df_res.to_csv("models/model_comparison.csv", index=False)
    print("\n=== Model Comparison ===")
    print(df_res.to_string(index=False))


def run_evaluation():
    import src.model_evaluation as ev
    print("\n" + "="*55)
    print("STAGE 5 — Model Evaluation")
    print("="*55)
    import xgboost as xgb, joblib
    import numpy as np
    X_test, y_test = ev.load_test_data()

    rf = joblib.load("models/random_forest.pkl")
    ev.plot_actual_vs_predicted(y_test, rf.predict(X_test),    "Random Forest")
    ev.plot_residuals(y_test,           rf.predict(X_test),    "Random Forest")

    xgb_m = xgb.XGBRegressor(); xgb_m.load_model("models/xgboost.json")
    ev.plot_actual_vs_predicted(y_test, xgb_m.predict(X_test), "XGBoost")
    ev.plot_residuals(y_test,           xgb_m.predict(X_test), "XGBoost")
    ev.plot_feature_importance()
    ev.plot_model_comparison()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip the EDA stage (saves time)")
    args = parser.parse_args()

    t0 = time.time()
    run_data_loading()
    run_feature_engineering()
    if not args.skip_eda:
        run_eda()
    run_training()
    run_evaluation()

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"{'='*55}")
