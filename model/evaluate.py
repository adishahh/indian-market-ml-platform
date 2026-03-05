"""
model/evaluate.py

Portfolio-level evaluation using the CURRENT XGBoost classifier model.
Generates performance metrics: directional accuracy, total return, Sharpe ratio.
Saves results to model/artifacts/backtest_metrics.json
"""
import sys
import os
sys.path.append(os.getcwd())

import json
import glob
import numpy as np
import pandas as pd
import xgboost as xgb

from model.prepare_dataset import build_dataset
from config.logger import get_logger

logger = get_logger(__name__)

TRANSACTION_COST = 0.001   # 10 bps per unit turnover
TARGET_VOL       = 0.15    # 15% annualized volatility target
TRADING_DAYS     = 252


def load_latest_classifier() -> xgb.XGBClassifier:
    """Load the most recently trained XGBoost classifier."""
    files = glob.glob("model/artifacts/model_cls_*.json")
    if not files:
        raise FileNotFoundError(
            "No classifier model found in model/artifacts/. "
            "Run model/train_model.py first."
        )
    latest = max(files, key=os.path.getctime)
    logger.info(f"Loading classifier: {latest}")
    model = xgb.XGBClassifier()
    model.load_model(latest)
    return model


def evaluate():
    logger.info("Loading dataset for evaluation...")
    df = build_dataset()

    if df.empty:
        logger.error("Dataset is empty — run run_pipeline.py first.")
        return

    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # Load model and get its expected feature list (handles any column set)
    model = load_latest_classifier()
    booster = model.get_booster()
    model_features = booster.feature_names

    # Fill any missing features with 0 (same as API logic)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[model_features]

    # Predict buy probability
    df["prob_buy"] = model.predict_proba(X)[:, 1]
    df["signal"]   = (df["prob_buy"] > 0.60).astype(int)

    # Position sizing (target volatility approach)
    df["daily_vol"] = (
        df["volatility_20d"].replace(0, np.nan) / np.sqrt(TRADING_DAYS)
        if "volatility_20d" in df.columns
        else 0.01
    )
    df["position"] = (df["signal"] * TARGET_VOL / df["daily_vol"]).clip(0, 1).fillna(0)

    # Turnover and transaction costs
    df["turnover"] = (
        df.groupby("stock_id")["position"]
        .diff().abs()
        .fillna(df["position"])
    )

    # Net return per stock per day
    # Uses target_return if available (regression target), else uses target_class as proxy
    if "target_return" in df.columns:
        ret_col = "target_return"
    else:
        # Use target_class as directional proxy (1 = up day, 0 = down day → mapped to +/-1%)
        df["target_return"] = df["target_class"].map({1: 0.01, 0: -0.01})
        ret_col = "target_return"

    df["net_return"] = df["position"] * df[ret_col] - df["turnover"] * TRANSACTION_COST

    # Portfolio daily returns (average across all stocks)
    daily_returns = df.groupby("date")["net_return"].mean().fillna(0)

    # Metrics
    cumulative_return    = (1 + daily_returns).cumprod()
    total_return         = float(cumulative_return.iloc[-1] - 1)
    sharpe_ratio         = float(
        daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS)
        if daily_returns.std() != 0 else 0.0
    )
    directional_accuracy = float(
        (df["signal"] == (df[ret_col] > 0)).mean()
    )

    metrics = {
        "directional_accuracy": round(directional_accuracy, 4),
        "total_return":         round(total_return, 4),
        "sharpe_ratio":         round(sharpe_ratio, 4),
    }

    os.makedirs("model/artifacts", exist_ok=True)
    with open("model/artifacts/backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation complete (cost-aware, portfolio-level)")
    logger.info(f"  Directional Accuracy : {directional_accuracy:.2%}")
    logger.info(f"  Total Return         : {total_return:.2%}")
    logger.info(f"  Sharpe Ratio         : {sharpe_ratio:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate()
