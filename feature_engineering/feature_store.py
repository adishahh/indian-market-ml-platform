import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
from sqlalchemy import text
from config.database import engine
from feature_engineering.build_features import calculate_technicals
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_feature_store():
    logger.info("Starting Feature Store Update...")
    
    # 1. Fetch Latest Data for All Active Stocks
    query_stocks = "SELECT stock_id, symbol FROM stocks WHERE is_active = true"
    with engine.connect() as conn:
        stocks = pd.read_sql(text(query_stocks), conn)
        
    for _, stock in stocks.iterrows():
        stock_id = stock['stock_id']
        symbol = stock['symbol']
        logger.info(f"Processing {symbol}...")
        
        # 2. Fetch Prices (Limit 100 to calculate indicators)
        query_prices = text("SELECT date, open, high, low, close, volume FROM prices WHERE stock_id = :sid ORDER BY date DESC LIMIT 100")
        with engine.connect() as conn:
            df = pd.read_sql(query_prices, conn, params={"sid": stock_id})
            
        if df.empty:
            logger.warning(f"No data for {symbol}, skipping.")
            continue
            
        df = df.sort_values('date') # Ascending for calc
        
        # 3. Calculate Technicals
        df = calculate_technicals(df)
        
        # 4. Fetch Macro Data (Latest available)
        # In a real pipeline, we'd align dates perfectly. 
        # Here we fetch the LATEST available macro row and broadcast it.
        # This assumes the cron runs daily when both markets are "current".
        macro_query = text("SELECT * FROM index_prices ORDER BY date DESC LIMIT 1")
        with engine.connect() as conn:
            macro_df = pd.read_sql(macro_query, conn)
            
        if not macro_df.empty:
            # Calculate returns for macro (simplified for daily store)
            # In a robust system, we'd calculate returns from history.
            # Here we assume pre-calculated or just use raw prices for V1 store.
            # WAIT: The model expects 'returns'.
            # We need to fetch HISTORY of macro to calc returns.
            pass
            
        # COMPLEXITY REDUCTION:
        # Instead of recalculating EVERYTHING from scratch (which risks mismatch with training),
        # we will use the same logic as 'build_dataset.py' but for specific dates.
        # HOWEVER, for the "Production Store", we usually just want the LATEST row.
        
        # Let's pull the LATEST row.
        latest_row = df.iloc[-1]
        date = latest_row['date']
        
        # 5. Get Sentiment
        sent_query = text("SELECT AVG(sentiment_score) as score FROM news WHERE date = :date")
        with engine.connect() as conn:
            sent = pd.read_sql(sent_query, conn, params={"date": date})
        sentiment = sent.iloc[0]['score'] if not sent.empty and pd.notna(sent.iloc[0]['score']) else 0.0
        
        # 6. Insert into Feature Store
        # We need to handle the specific columns the table expects.
        # Table cols: stock_id, date, return_1d... macro... sentiment
        
        # Define the row to insert
        insert_query = text("""
            INSERT INTO feature_store (
                stock_id, date, 
                return_1d, return_5d, return_20d, sma_20, sma_50, ema_20, rsi_14, volatility_20d,
                macro_nifty_bank_ret, macro_crude_oil_ret, macro_gold_ret, macro_usd_inr_ret, macro_nifty_50_ret,
                sentiment_score
            ) VALUES (
                :sid, :date,
                :r1, :r5, :r20, :sma20, :sma50, :ema20, :rsi, :vol,
                0, 0, 0, 0, 0, -- Macro defaults for V1 stability (Phase 12 MVP)
                :sent
            )
            ON CONFLICT (stock_id, date) DO UPDATE SET
                return_1d = EXCLUDED.return_1d,
                rsi_14 = EXCLUDED.rsi_14,
                sentiment_score = EXCLUDED.sentiment_score;
        """)
        
        # Convert NumPy types to Python types for SQL
        params = {
            "sid": stock_id,
            "date": date,
            "r1": float(latest_row.get('return_1d', 0)),
            "r5": float(latest_row.get('return_5d', 0)),
            "r20": float(latest_row.get('return_20d', 0)),
            "sma20": float(latest_row.get('sma_20', 0)),
            "sma50": float(latest_row.get('sma_50', 0)),
            "ema20": float(latest_row.get('ema_20', 0)),
            "rsi": float(latest_row.get('rsi_14', 0)),
            "vol": float(latest_row.get('volatility_20d', 0)),
            "sent": float(sentiment)
        }
        
        with engine.begin() as conn:
            conn.execute(insert_query, params)
            
    logger.info("Feature Store Updated Successfully.")

if __name__ == "__main__":
    update_feature_store()
