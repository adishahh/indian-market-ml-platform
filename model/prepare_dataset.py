import pandas as pd
import yaml
from sqlalchemy import text
from config.database import engine

# ---------------- CONFIG ----------------
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MARKET_CONFIG = config.get("market", {})
# Default to 5 days if not set
TARGET_HORIZON = MARKET_CONFIG.get("prediction_horizon_days", 5) 
# --------------------------------------


def build_dataset():
    # 1. Fetch Stock Features
    # We grab the raw prices too for target calculation
    stock_query = """
    SELECT 
        f.stock_id, 
        f.date,
        f.return_1d, f.return_5d, f.return_20d,
        f.sma_20, f.sma_50, f.ema_20,
        f.rsi_14, f.volatility_20d,
        p.close AS stock_close_today
    FROM features_daily f
    JOIN prices p ON f.stock_id = p.stock_id AND f.date = p.date
    ORDER BY f.date, f.stock_id
    """
    df = pd.read_sql(text(stock_query), engine)
    
    # 2. Fetch Index Data (Pivot it)
    index_query = """
    SELECT i.symbol, ip.date, ip.close
    FROM index_prices ip
    JOIN indices i ON ip.index_id = i.index_id
    ORDER BY ip.date
    """
    df_idx = pd.read_sql(text(index_query), engine)
    
    if df_idx.empty:
        print("Warning: No index data found. Skipping macro features.")
        df_macro = pd.DataFrame() # Empty
    else:
        # Pivot: date | CL=F | GC=F | INR=X | ^NSEBANK | ^NSEI
        df_macro = df_idx.pivot(index='date', columns='symbol', values='close')
        
        # Calculate Returns for features (past info) -> pct_change(1)
        df_macro_ret = df_macro.pct_change()
        df_macro_ret = df_macro_ret.add_suffix('_ret')
        
        # We also need the Nifty 50 Future Close for the TARGET calculation
        df_macro_raw = df_macro[['^NSEI']].copy()
        df_macro_raw = df_macro_raw.rename(columns={'^NSEI': 'index_close_today'})
        
        # Merge returns and raw price
        df_macro_final = pd.concat([df_macro_ret, df_macro_raw], axis=1)
        df_macro_final = df_macro_final.reset_index()
        
        # Ensure date is datetime
        df_macro_final['date'] = pd.to_datetime(df_macro_final['date'])
        df['date'] = pd.to_datetime(df['date'])

        # 3. Merge Macro Features
        df = pd.merge(df, df_macro_final, on='date', how='left')
        
        # Forward fill macro data
        df = df.ffill()
        df = df.fillna(0)

    # 3.5 Merge Sentiment Data
    sentiment_query = """
    SELECT date, AVG(sentiment_score) as daily_sentiment
    FROM news
    WHERE sentiment_score IS NOT NULL
    GROUP BY date
    ORDER BY date
    """
    df_sent = pd.read_sql(text(sentiment_query), engine)
    
    if not df_sent.empty:
        df_sent['date'] = pd.to_datetime(df_sent['date'])
        df = pd.merge(df, df_sent, on='date', how='left')
        # Fill missing sentiment with 0 (Neutral)
        df['daily_sentiment'] = df['daily_sentiment'].fillna(0)
    else:
        df['daily_sentiment'] = 0
    
    # 4. Target Calculation (Alpha)
    # We need FUTURE Nifty 50 return
    # We already have stock_close_future logic in python now
    
    # Calculate Stock Future manually in Pandas since we dropped the window function in SQL
    # Sort by stock and date to shift correctly
    df = df.sort_values(by=['stock_id', 'date'])
    
    df['stock_close_future'] = df.groupby('stock_id')['stock_close_today'].shift(-TARGET_HORIZON)
    
    # Calculate Index Future Return
    # We need to shift the 'index_close_today' column BUT we can't group by stock_id 
    # because the index is the same for all stocks on the same date.
    # However, since we merged on date, 'index_close_today' is repeated for every stock.
    # So valid approach: groupby stock_id and shift index_close_today too.
    df['index_close_future'] = df.groupby('stock_id')['index_close_today'].shift(-TARGET_HORIZON)

    # 5. Drop rows where future is NaN (last 5 days)
    df = df.dropna(subset=['stock_close_future', 'index_close_future'])

    # 6. Calculate Returns
    df["stock_return"] = df["stock_close_future"] / df["stock_close_today"] - 1
    df["index_return"] = df["index_close_future"] / df["index_close_today"] - 1
    
    # 7. Excess Return
    df["excess_return"] = df["stock_return"] - df["index_return"]
    
    # 8. Target Class
    df["target_class"] = (df["excess_return"] > 0).astype(int)
    
    # Cleanup
    drop_cols = [
        "stock_close_today", "stock_close_future", 
        "index_close_today", "index_close_future",
        "stock_return", "index_return", "excess_return"
    ]
    df = df.drop(columns=drop_cols)
    
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
