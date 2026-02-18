import yfinance as yf
import pandas as pd
import logging
from sqlalchemy import text
from config.database import engine

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDICES = {
    "^NSEI": "Nifty 50",
    "^NSEBANK": "Nifty Bank",
    "INR=X": "USD/INR",
    "CL=F": "Crude Oil",
    "GC=F": "Gold"
}

def load_index_data(start_date="2018-01-01"):
    for symbol, name in INDICES.items():
        logger.info(f"Processing {name} ({symbol})...")
        
        # 1. Ensure Index exists in Master Table
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO indices (symbol, name)
                    VALUES (:symbol, :name)
                    ON CONFLICT (symbol) DO NOTHING
                """),
                {"symbol": symbol, "name": name}
            )
            
            # Get Index ID
            result = conn.execute(
                text("SELECT index_id FROM indices WHERE symbol = :symbol"),
                {"symbol": symbol}
            )
            index_id = result.fetchone()[0]

        # 2. Fetch Data
        try:
            df = yf.download(symbol, start=start_date, progress=False, auto_adjust=True)
            if df.empty:
                logger.error(f"No data for {symbol}")
                continue
                
            df.reset_index(inplace=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Quick validation
            if 'Date' not in df.columns or 'Close' not in df.columns:
                logger.error(f"Missing required columns in index data for {symbol}: {df.columns}")
                continue
                
            # 3. Insert into DB
            with engine.begin() as conn:
                for row in df.itertuples(index=False):
                    conn.execute(
                        text("""
                            INSERT INTO index_prices (index_id, date, close)
                            VALUES (:index_id, :date, :close)
                            ON CONFLICT (index_id, date) DO NOTHING
                        """),
                        {
                            "index_id": index_id,
                            "date": row.Date.date(),
                            "close": float(row.Close)
                        }
                    )
            logger.info(f"Loaded {len(df)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed {symbol}: {e}")

if __name__ == "__main__":
    load_index_data()
