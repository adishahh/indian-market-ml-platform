import pandas as pd
from config.database import Database
from sqlalchemy import text

def load_stocks(csv_path: str):
    db = Database()
    engine = db.get_engine()

    df = pd.read_csv(csv_path)

    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                INSERT INTO stocks (symbol, name, exchange, sector)
                VALUES (:symbol, :name, :exchange, :sector)
                ON CONFLICT (symbol) DO NOTHING
                """),
                {
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "exchange": "NSE",
                    "sector": row.get("sector")
                }
            )

if __name__ == "__main__":
    load_stocks("data/nifty50.csv")
