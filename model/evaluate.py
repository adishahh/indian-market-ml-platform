import json
import numpy as np
import pandas as pd
import xgboost as xgb

from model.prepare_dataset import build_dataset

# ---------------- CONFIG ----------------
FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi_14",
    "volatility_20d",
]

TRANSACTION_COST = 0.001   # 10 bps per unit turnover
TARGET_VOL = 0.15          # 15% annualized volatility
TRADING_DAYS = 252
# ---------------------------------------


def evaluate():
    # ---------------- LOAD DATA ----------------
    df = build_dataset()

    df = (
        df.sort_values(["stock_id", "date"])
        .reset_index(drop=True)
    )

    # ---------------- LOAD MODEL ----------------
    model = xgb.XGBRegressor()
    model.load_model("model/artifacts/model.json")

    # ---------------- PREDICTIONS ----------------
    X = df[FEATURES]
    df["predicted_return"] = model.predict(X)

    # Long-only signal
    df["signal"] = (df["predicted_return"] > 0).astype(int)

    # ---------------- POSITION SIZING ----------------
    df["daily_vol"] = (
        df["volatility_20d"]
        .replace(0, np.nan)
        / np.sqrt(TRADING_DAYS)
    )

    df["position"] = (
        df["signal"] * TARGET_VOL / df["daily_vol"]
    )

    # Enforce constraints
    df["position"] = (
        df["position"]
        .clip(0, 1)
        .fillna(0)
    )

    # ---------------- TURNOVER & COSTS ----------------
    df["turnover"] = (
        df.groupby("stock_id")["position"]
        .diff()
        .abs()
        .fillna(df["position"])
    )

    # ---------------- NET STRATEGY RETURN ----------------
    df["net_return"] = (
        df["position"] * df["target_return"]
        - df["turnover"] * TRANSACTION_COST
    )

    # ---------------- PORTFOLIO AGGREGATION ----------------
    daily_returns = (
        df.groupby("date")["net_return"]
        .mean()
        .fillna(0)
    )

    # ---------------- PERFORMANCE METRICS ----------------
    cumulative_return = (1 + daily_returns).cumprod()
    total_return = cumulative_return.iloc[-1] - 1

    sharpe_ratio = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS)
        if daily_returns.std() != 0
        else 0.0
    )

    directional_accuracy = (
        (df["signal"] == (df["target_return"] > 0)).mean()
    )

    metrics = {
        "directional_accuracy": float(directional_accuracy),
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
    }

    # ---------------- SAVE ARTIFACTS ----------------
    with open("model/artifacts/backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------------- OUTPUT ----------------
    print("Backtest completed (cost-aware portfolio-level)")
    print("Directional accuracy:", directional_accuracy)
    print("Total return:", total_return)
    print("Sharpe ratio:", sharpe_ratio)


if __name__ == "__main__":
    evaluate()
