import logging
from sqlalchemy import text
from config.database import engine
# Import the new function from market_data.py
from data_ingestion.market_data import fetch_market_data_with_retry, DataIngestionError

# Use the same logger config
logger = logging.getLogger(__name__)

def fetch_stock_symbols():
    query = "SELECT stock_id, symbol FROM stocks"
    with engine.connect() as conn:
        return conn.execute(text(query)).fetchall()


def load_prices(start_date="2018-01-01", end_date=None):
    try:
        stocks = fetch_stock_symbols()
    except Exception as e:
        logger.critical(f"Database connection failed: {e}")
        return

    for stock_id, symbol in stocks:
        yahoo_symbol = symbol + ".NS"
        
        try:
            # Use the robust fetch function
            df = fetch_market_data_with_retry(yahoo_symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Skipping {symbol} after all retries failed. Error: {e}")
            continue

        # In-memory transformation
        df.reset_index(inplace=True)
        
        # Ensure column names match schema exactly if needed, or map them below
        # Note: fetch_market_data_with_retry already flattens columns

        try:
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
                            "date": row.Date.date(),
                            "open": float(row.Open),
                            "close": float(row.Close),
                            "volume": int(row.Volume),
                        },
                    )
            logger.info(f"Successfully loaded prices for {symbol} into DB")
            
        except Exception as db_err:
            logger.error(f"Database insertion failed for {symbol}: {db_err}")


if __name__ == "__main__":
    load_prices()
