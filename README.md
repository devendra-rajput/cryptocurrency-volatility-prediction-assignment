# Cryptocurrency Volatility Prediction

A complete machine learning pipeline to predict cryptocurrency volatility using historical OHLCV price data.

---

## Project Structure

```
crypto_volatility/
├── data/                          Raw and processed datasets
├── src/
│   ├── data_loader.py             Data ingestion and cleaning
│   ├── feature_engineering.py     Feature creation (MA, BB, ATR, lag features)
│   ├── eda.py                     Exploratory Data Analysis plots
│   ├── model_training.py          Train Random Forest, XGBoost, LSTM
│   ├── model_evaluation.py        Evaluation plots and metrics
│   ├── hyperparameter_tuning.py   Optuna-based XGBoost tuning
│   ├── predict.py                 Inference utility (CLI + module)
│   └── pipeline.py                End-to-end runner
├── notebooks/
│   └── crypto_volatility_pipeline.ipynb
├── models/                        Saved model artefacts
├── deployment/
│   └── app.py                     Streamlit web app
├── reports/
│   ├── figures/                   EDA and evaluation plots
│   ├── HLD_Document.docx          High-Level Design document
│   ├── LLD_Document.docx          Low-Level Design document
│   └── Final_Report.docx          Final project report
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone or download the project
cd crypto_volatility

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
# Download the Cryptocurrency Historical Prices Dataset from Kaggle
# and save it as: data/crypto_historical_prices.csv
```

---

## Running the Pipeline

### Option A — Full pipeline (recommended)
```bash
python src/pipeline.py
```

This runs all stages in sequence: data cleaning → feature engineering → EDA → model training → evaluation.

To skip the EDA stage:
```bash
python src/pipeline.py --skip-eda
```

### Option B — Individual stages
```bash
python src/data_loader.py
python src/feature_engineering.py
python src/eda.py
python src/model_training.py
python src/model_evaluation.py
```

### Option C — Hyperparameter tuning (optional, takes ~10 min)
```bash
python src/hyperparameter_tuning.py
```

---

## Deployment

```bash
streamlit run deployment/app.py
```

Open http://localhost:8501 in your browser.

---

## Making Predictions

```bash
# Command line
python src/predict.py --symbol BTC --date 2023-06-01

# In Python
from src.predict import predict_volatility
pred = predict_volatility({
    "log_return": 0.012,
    "volume_change": 0.05,
    "bb_bandwidth": 0.18,
    "atr": 850.0,
    "liquidity_ratio": 0.0042,
    "close_to_ma7": 1.02,
    "close_to_ma30": 0.98,
    "volatility_lag_1": 0.72,
    "volatility_lag_3": 0.68,
    "volatility_lag_7": 0.65,
})
print(f"Predicted volatility: {pred:.4f}")
```

---

## Models

| Model         | RMSE    | MAE     | R²    |
|---------------|---------|---------|-------|
| Random Forest | ~0.042  | ~0.031  | ~0.87 |
| XGBoost       | ~0.035  | ~0.025  | ~0.91 |
| LSTM          | ~0.038  | ~0.028  | ~0.89 |

XGBoost is used for deployment. Exact values depend on the dataset version.

---

## Deliverables

- `src/` — all source code, well-commented
- `notebooks/crypto_volatility_pipeline.ipynb` — full walkthrough notebook
- `reports/figures/` — 7 EDA plots + evaluation charts
- `reports/HLD_Document.docx` — system architecture overview
- `reports/LLD_Document.docx` — module-level implementation details
- `reports/Final_Report.docx` — findings, metrics, insights, future work
- `deployment/app.py` — Streamlit prediction interface
