import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import yaml
import logging
from sqlalchemy import text
from config.database import engine
from config.logger import get_logger
from feature_engineering.scaler import fit_and_save_scaler

# Use centralized logger
logger = get_logger(__name__)


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

    # 2. Returns
    stock_df["return_1d"] = stock_df["close"].pct_change(1)
    stock_df["return_5d"] = stock_df["close"].pct_change(5)
    stock_df["return_20d"] = stock_df["close"].pct_change(20)

    # 3. SMAs
    for w in FEAT_CONFIG.get("sma_windows", [20, 50]):
        stock_df[f"sma_{w}"] = stock_df["close"].rolling(w).mean()

    # 4. EMAs
    for w in FEAT_CONFIG.get("ema_windows", [20]):
        stock_df[f"ema_{w}"] = stock_df["close"].ewm(span=w, adjust=False).mean()

    # 5. RSI
    rsi_window = FEAT_CONFIG.get("rsi_window", 14)
    stock_df[f"rsi_{rsi_window}"] = compute_rsi(stock_df["close"], rsi_window)

    # 6. Volatility
    vol_window = FEAT_CONFIG.get("volatility_window", 20)
    stock_df[f"volatility_{vol_window}d"] = stock_df["return_1d"].rolling(vol_window).std()

    # 7. Lag Features
    for lag in FEAT_CONFIG.get("lag_days", []):
        stock_df[f"close_lag_{lag}"] = stock_df["close"].shift(lag)

    # --- NEW FEATURES ---

    # 8. MACD (12/26/9)
    ema_12 = stock_df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = stock_df["close"].ewm(span=26, adjust=False).mean()
    stock_df["macd"] = ema_12 - ema_26
    stock_df["macd_signal"] = stock_df["macd"].ewm(span=9, adjust=False).mean()
    stock_df["macd_hist"] = stock_df["macd"] - stock_df["macd_signal"]

    # 9. Volume Ratio (today's volume vs 20d avg — spikes = momentum)
    stock_df["volume_ratio"] = stock_df["volume"] / stock_df["volume"].rolling(20).mean()

    # 10. OBV (On-Balance Volume)
    obv = [0]
    for i in range(1, len(stock_df)):
        if stock_df["close"].iloc[i] > stock_df["close"].iloc[i - 1]:
            obv.append(obv[-1] + stock_df["volume"].iloc[i])
        elif stock_df["close"].iloc[i] < stock_df["close"].iloc[i - 1]:
            obv.append(obv[-1] - stock_df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    stock_df["obv"] = obv
    stock_df["obv_ma"] = stock_df["obv"].rolling(20).mean()

    # 11. Price vs SMA20 distance (how extended is price from its mean)
    stock_df["price_to_sma20"] = (stock_df["close"] - stock_df["sma_20"]) / stock_df["sma_20"]

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
    
    # CLEANING: Replace Inf with NaN and drop
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(inplace=True)
    
    # 4. Feature Scaling — fit ONLY on training data (pre-2023) to prevent data leakage
    # This is critical for time-series ML: scaler must not see future (test) data
    exclude_cols = ['stock_id', 'date']
    feature_cols = [c for c in final_df.columns if c not in exclude_cols]

    # Use pre-2023 as train split for scaler fitting (consistent with model train/test split)
    TRAIN_CUTOFF = pd.Timestamp("2023-01-01")
    train_only_df = final_df[final_df['date'] < TRAIN_CUTOFF]

    if len(train_only_df) > 100:
        fit_and_save_scaler(train_only_df, feature_cols)
        logger.info(f"Scaler fitted on {len(train_only_df)} training rows (pre-2023 only — no leakage).")
    else:
        # Fallback: not enough pre-2023 data, fit on all (shouldn't happen with real data)
        fit_and_save_scaler(final_df, feature_cols)
        logger.warning("Not enough pre-2023 data — scaler fitted on full dataset (check your data range).")


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
            if_exists='replace', 
            index=False,
            method='multi',
            chunksize=1000 # Batch size
        )

    logger.info("Feature engineering completed successfully!")

if __name__ == "__main__":
    build_features()
