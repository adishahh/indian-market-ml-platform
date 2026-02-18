import sys
import os
sys.path.append(os.getcwd())
from config.database import engine
from sqlalchemy import text

def create_prod_tables():
    with engine.connect() as conn:
        print("Creating table: feature_store...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feature_store (
                stock_id INTEGER REFERENCES stocks(stock_id),
                date DATE,
                
                -- The full feature vector (Technicals + Macro + Sentiment)
                -- We use JSONB for flexibility if features change, 
                -- or we can use explicit columns.
                -- For speed and strictness, let's use explicit columns matching our model.
                
                return_1d FLOAT,
                return_5d FLOAT,
                return_20d FLOAT,
                sma_20 FLOAT,
                sma_50 FLOAT,
                ema_20 FLOAT,
                rsi_14 FLOAT,
                volatility_20d FLOAT,
                
                macro_nifty_bank_ret FLOAT,
                macro_crude_oil_ret FLOAT,
                macro_gold_ret FLOAT,
                macro_usd_inr_ret FLOAT,
                macro_nifty_50_ret FLOAT,
                
                sentiment_score FLOAT DEFAULT 0.0,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (stock_id, date)
            );
        """))
        
        print("Creating table: prediction_logs...")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                stock_id INTEGER REFERENCES stocks(stock_id),
                prediction_date DATE, -- Date the prediction is FOR
                
                predicted_class VARCHAR(10), -- "Buy" or "Sell"
                probability FLOAT,           -- 0.75
                
                model_version VARCHAR(255),  -- "model_cls_20260218..."
                execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        
        conn.commit()
    print("Tables created successfully.")

if __name__ == "__main__":
    create_prod_tables()
