import pandas as pd
from sqlalchemy import text
from config.database import engine


# ---------------- CONFIG ----------------
TARGET_HORIZON = 5  # days ahead to predict
# --------------------------------------


def build_dataset():
    query = f"""
    SELECT
        f.stock_id,
        f.date,
        f.return_1d,
        f.return_5d,
        f.return_20d,
        f.sma_20,
        f.sma_50,
        f.ema_20,
        f.rsi_14,
        f.volatility_20d,
        p.close AS close_today,
        LEAD(p.close, {TARGET_HORIZON})
            OVER (PARTITION BY p.stock_id ORDER BY p.date) AS close_future
    FROM features_daily f
    JOIN prices p
        ON f.stock_id = p.stock_id
       AND f.date = p.date
    ORDER BY f.stock_id, f.date
    """

    df = pd.read_sql(text(query), engine)

    # Forward return (multi-day)
    df["target_return"] = (
        df["close_future"] / df["close_today"] - 1
    )

    # Drop rows where future price is unavailable
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print("Dataset shape:", df.shape)
    print("Target horizon:", TARGET_HORIZON, "days")
