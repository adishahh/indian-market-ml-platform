import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import timedelta

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

TRAIN_YEARS = 3
TEST_MONTHS = 6


def walk_forward_backtest():
    df = build_dataset()

    # FORCE pandas datetime (CRITICAL FIX)
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)


    start_date = pd.to_datetime(df["date"].min())
    end_date = pd.to_datetime(df["date"].max())


    results = []

    train_start = start_date

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        if test_end > end_date:
            break

        train_df = df[
            (df["date"] >= train_start) &
            (df["date"] < train_end)
        ]

        test_df = df[
            (df["date"] >= train_end) &
            (df["date"] < test_end)
        ]

        if len(train_df) < 1000 or len(test_df) < 200:
            train_start += pd.DateOffset(months=6)
            continue

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

        model.fit(
            train_df[FEATURES],
            train_df["target_return"]
        )

        test_df = test_df.copy()
        test_df["prediction"] = model.predict(test_df[FEATURES])
        test_df["signal"] = (test_df["prediction"] > 0).astype(int)

        test_df["strategy_return"] = (
            test_df["signal"] * test_df["target_return"]
        )

        daily_returns = (
            test_df.groupby("date")["strategy_return"]
            .mean()
            .fillna(0)
        )

        sharpe = (
            daily_returns.mean() /
            daily_returns.std() * np.sqrt(252)
            if daily_returns.std() != 0 else 0
        )

        results.append({
            "train_start": train_start.date(),
            "train_end": train_end.date(),
            "test_start": train_end.date(),
            "test_end": test_end.date(),
            "sharpe": sharpe,
            "mean_daily_return": daily_returns.mean(),
        })

        train_start += pd.DateOffset(months=6)

    results_df = pd.DataFrame(results)

    print("\nWALK-FORWARD RESULTS")
    print(results_df)
    print("\nAverage Sharpe:", results_df["sharpe"].mean())

    return results_df


if __name__ == "__main__":
    walk_forward_backtest()
