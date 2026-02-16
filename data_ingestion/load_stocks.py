from config.database import engine
from sqlalchemy import text
import pandas as pd


def load_stocks(csv_path: str):
    df = pd.read_csv(csv_path)

    with engine.begin() as conn:
        for symbol in df["symbol"].unique():
            conn.execute(
                text(
                    """
                    INSERT INTO stocks (symbol)
                    VALUES (:symbol)
                    ON CONFLICT (symbol) DO NOTHING
                    """
                ),
                {"symbol": symbol},
            )

    print("Stocks loaded successfully")


if __name__ == "__main__":
    load_stocks("data/raw/nifty50.csv")
