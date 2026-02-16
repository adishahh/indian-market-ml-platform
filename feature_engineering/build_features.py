import pandas as pd
import numpy as np
from sqlalchemy import text
from config.database import engine


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_price_data():
    query = """
        SELECT stock_id, date, close
        FROM prices
        ORDER BY stock_id, date
    """
    return pd.read_sql(query, engine)


def build_features():
    df = fetch_price_data()

    feature_rows = []

    for stock_id, stock_df in df.groupby("stock_id"):
        stock_df = stock_df.sort_values("date").copy()

        stock_df["return_1d"] = stock_df["close"].pct_change(1)
        stock_df["return_5d"] = stock_df["close"].pct_change(5)
        stock_df["return_20d"] = stock_df["close"].pct_change(20)

        stock_df["sma_20"] = stock_df["close"].rolling(20).mean()
        stock_df["sma_50"] = stock_df["close"].rolling(50).mean()
        stock_df["ema_20"] = stock_df["close"].ewm(span=20, adjust=False).mean()

        stock_df["rsi_14"] = compute_rsi(stock_df["close"], 14)
        stock_df["volatility_20d"] = stock_df["return_1d"].rolling(20).std()

        stock_df = stock_df.dropna()

        for _, row in stock_df.iterrows():
            feature_rows.append({
                "stock_id": stock_id,
                "date": row["date"],
                "return_1d": row["return_1d"],
                "return_5d": row["return_5d"],
                "return_20d": row["return_20d"],
                "sma_20": row["sma_20"],
                "sma_50": row["sma_50"],
                "ema_20": row["ema_20"],
                "rsi_14": row["rsi_14"],
                "volatility_20d": row["volatility_20d"]
            })

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE features_daily"))

        conn.execute(
            text("""
                INSERT INTO features_daily (
                    stock_id, date,
                    return_1d, return_5d, return_20d,
                    sma_20, sma_50, ema_20,
                    rsi_14, volatility_20d
                )
                VALUES (
                    :stock_id, :date,
                    :return_1d, :return_5d, :return_20d,
                    :sma_20, :sma_50, :ema_20,
                    :rsi_14, :volatility_20d
                )
            """),
            feature_rows
        )

    print("Feature engineering completed successfully")


if __name__ == "__main__":
    build_features()
