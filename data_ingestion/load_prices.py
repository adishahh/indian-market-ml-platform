import yfinance as yf
from sqlalchemy import text
from config.database import engine


def fetch_stock_symbols():
    query = "SELECT stock_id, symbol FROM stocks"
    with engine.connect() as conn:
        return conn.execute(text(query)).fetchall()


def load_prices(start_date="2018-01-01", end_date=None):
    stocks = fetch_stock_symbols()

    for stock_id, symbol in stocks:
        yahoo_symbol = symbol + ".NS"
        print(f"Fetching data for {yahoo_symbol}")

        df = yf.download(
            yahoo_symbol,
            start=start_date,
            end=end_date,
            progress=False
        )

        if df.empty:
            print(f"No data found for {symbol}")
            continue

        # ✅ RESET INDEX FIRST
        df.reset_index(inplace=True)

        # ✅ FLATTEN MULTI-INDEX COLUMNS (CRITICAL FIX)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        with engine.begin() as conn:
            for row in df.itertuples(index=False):
                conn.execute(
                    text(
                        """
                        INSERT INTO prices (stock_id, date, open, close, volume)
                        VALUES (:stock_id, :date, :open, :close, :volume)
                        ON CONFLICT (stock_id, date) DO NOTHING
                        """
                    ),
                    {
                        "stock_id": stock_id,
                        "date": row.Date.date(),   # ✅ NOW THIS WORKS
                        "open": float(row.Open),
                        "close": float(row.Close),
                        "volume": int(row.Volume),
                    },
                )

        print(f"Loaded prices for {symbol}")


if __name__ == "__main__":
    load_prices()
