import pandas as pd
from sqlalchemy import text
from config.database import engine


def build_dataset():
    query = """
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
        LEAD(p.close) OVER (PARTITION BY p.stock_id ORDER BY p.date) AS close_next
    FROM features_daily f
    JOIN prices p
        ON f.stock_id = p.stock_id
       AND f.date = p.date
    ORDER BY f.stock_id, f.date
    """

    df = pd.read_sql(text(query), engine)

    df["target_return"] = (df["close_next"] - df["close_today"]) / df["close_today"]

    df = df.dropna()

    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print("Dataset shape:", df.shape)
