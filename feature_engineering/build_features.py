import pandas as pd
import numpy as np
import yaml
import logging
from sqlalchemy import text
from config.database import engine
from feature_engineering.scaler import fit_and_save_scaler

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

FEAT_CONFIG = config.get("features", {})

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_price_data():
    query = """
        SELECT stock_id, date, close, volume, open
        FROM prices
        ORDER BY stock_id, date
    """
    return pd.read_sql(query, engine)

def build_features():
    logger.info("Fetching raw price data...")
    df = fetch_price_data()
    
    if df.empty:
        logger.warning("No price data found in DB!")
        return

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    feature_rows = []
    
def calculate_technicals(stock_df):
    stock_df = stock_df.sort_values("date").copy()
    
    # 1. Handle Missing Data
    stock_df[['close', 'open', 'volume']] = stock_df[['close', 'open', 'volume']].ffill()

    # 2. Key Technical Indicators
    # Returns
    stock_df["return_1d"] = stock_df["close"].pct_change(1)
    stock_df["return_5d"] = stock_df["close"].pct_change(5)
    stock_df["return_20d"] = stock_df["close"].pct_change(20)

    # SMAs
    for w in FEAT_CONFIG.get("sma_windows", [20, 50]):
        stock_df[f"sma_{w}"] = stock_df["close"].rolling(w).mean()

    # EMAs
    for w in FEAT_CONFIG.get("ema_windows", [20]):
        stock_df[f"ema_{w}"] = stock_df["close"].ewm(span=w, adjust=False).mean()

    # RSI
    rsi_window = FEAT_CONFIG.get("rsi_window", 14)
    stock_df[f"rsi_{rsi_window}"] = compute_rsi(stock_df["close"], rsi_window)

    # Volatility
    vol_window = FEAT_CONFIG.get("volatility_window", 20)
    stock_df[f"volatility_{vol_window}d"] = stock_df["return_1d"].rolling(vol_window).std()

    # 3. Lag Features
    for lag in FEAT_CONFIG.get("lag_days", []):
        stock_df[f"close_lag_{lag}"] = stock_df["close"].shift(lag)

    # Drop rows that have NaNs due to rolling windows (optional, handled by caller if needed)
    # stock_df = stock_df.dropna()
    
    return stock_df

def build_features():
    logger.info("Fetching raw price data...")
    df = fetch_price_data()
    
    if df.empty:
        logger.warning("No price data found in DB!")
        return

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    feature_rows = []
    
    # Process per stock
    for stock_id, stock_df in df.groupby("stock_id"):
        # Use common logic
        stock_df = calculate_technicals(stock_df)
        
        # Drop rows that have NaNs due to rolling windows
        stock_df = stock_df.dropna()
        
        # Append to main list
        feature_rows.extend(stock_df.to_dict('records'))

    if not feature_rows:
        logger.warning("No features generated (maybe not enough history?)")
        return

    # Create DataFrame for Scaling
    final_df = pd.DataFrame(feature_rows)
    
    # 4. Feature Scaling (Fit on everything for now - in production separate train/test fit)
    # Identify numeric feature columns (exclude ID and Date)
    exclude_cols = ['stock_id', 'date']
    feature_cols = [c for c in final_df.columns if c not in exclude_cols]
    
    # Save the scaler for inference usage
    fit_and_save_scaler(final_df, feature_cols)
    
    # (Optional) We could apply the scaling here, but usually libraries handle unscaled features ok. 
    # For neural networks, we would replace values with scaled ones. 
    # For now, we just save the scaler.

    logger.info(f"Generated {len(final_df)} feature rows. Saving to DB...")

    # 5. Database Save (Truncate & Insert)
    # Note: efficient bulk insert requires handling columns dynamically if they changed
    # For simplicity/robustness, we might want to drop/recreate table or use pandas to_sql
    
    with engine.begin() as conn:
        # Clear old features
        conn.execute(text("TRUNCATE TABLE features_daily"))
        
        # Using pandas to_sql is cleaner for writing large DF with dynamic columns
        final_df.to_sql(
            'features_daily', 
            con=conn, 
            if_exists='append', 
            index=False,
            method='multi',
            chunksize=1000 # Batch size
        )

    logger.info("Feature engineering completed successfully!")

if __name__ == "__main__":
    build_features()
