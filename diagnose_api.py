from config.database import engine
from sqlalchemy import text
import pandas as pd

def diagnose():
    print("Checking database for TCS.NS...")
    
    with engine.connect() as conn:
        # 1. Check if stock exists
        stock = pd.read_sql(text("SELECT * FROM stocks WHERE symbol = 'TCS.NS'"), conn)
        if stock.empty:
            print("ERROR: TCS.NS not found in 'stocks' table!")
            return
        
        stock_id = stock.iloc[0]['stock_id']
        print(f"Found TCS.NS with stock_id: {stock_id}")
        
        # 2. Check features
        features = pd.read_sql(text(f"SELECT * FROM features_daily WHERE stock_id = {stock_id} ORDER BY date DESC LIMIT 5"), conn)
        
        if features.empty:
            print(f"ERROR: No data in 'features_daily' for stock_id {stock_id}!")
        else:
            print(f"Found {len(features)} rows in features_daily. Latest date: {features.iloc[0]['date']}")
            print(features.head())

if __name__ == "__main__":
    diagnose()
