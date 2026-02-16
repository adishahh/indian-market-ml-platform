import json
import numpy as np
import pandas as pd
import xgboost as xgb
from model.prepare_dataset import build_dataset

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


def evaluate():
    df = build_dataset()

    X = df[FEATURES]
    y_true = df["target_return"]

    # Load trained model
    model = xgb.XGBRegressor()
    model.load_model("model/artifacts/model.json")

    # Predictions
    df["predicted_return"] = model.predict(X)

    # Direction (up/down)
    df["actual_direction"] = (y_true > 0).astype(int)
    df["predicted_direction"] = (df["predicted_return"] > 0).astype(int)

    directional_accuracy = (
        df["actual_direction"] == df["predicted_direction"]
    ).mean()

    # Strategy: long if prediction > 0, else flat
    df["strategy_return"] = df["predicted_direction"] * y_true

    # Performance metrics
    cumulative_return = (1 + df["strategy_return"]).prod() - 1
    sharpe_ratio = (
        np.mean(df["strategy_return"]) /
        np.std(df["strategy_return"]) * np.sqrt(252)
        if np.std(df["strategy_return"]) != 0 else 0
    )

    metrics = {
        "directional_accuracy": directional_accuracy,
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio,
    }

    # Save metrics
    with open("model/artifacts/backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Backtest completed")
    print("Directional accuracy:", directional_accuracy)
    print("Cumulative return:", cumulative_return)
    print("Sharpe ratio:", sharpe_ratio)


if __name__ == "__main__":
    evaluate()
