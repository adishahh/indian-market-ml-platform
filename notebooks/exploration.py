import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sys
import os

# Connect to Database
start_dir = os.getcwd()
sys.path.append(start_dir)
from config.database import DATABASE_URL

print("Connecting to Database...")
engine = create_engine(DATABASE_URL)

def show_chart(symbol="TCS"):
    # 1. Fetch Data
    query = f"""
    SELECT f.date, f.rsi_14, p.close 
    FROM features_daily f
    JOIN stocks s ON f.stock_id = s.stock_id
    JOIN prices p ON f.stock_id = p.stock_id AND f.date = p.date
    WHERE s.symbol = '{symbol}'
    ORDER BY f.date DESC LIMIT 100
    """
    
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 2. Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color=color)
    ax1.plot(df['date'], df['close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:red'
    ax2.set_ylabel('RSI (14)', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['date'], df['rsi_14'], color=color, linestyle='--')
    ax2.axhline(30, color='green', linewidth=1, label="Oversold")
    ax2.axhline(70, color='red', linewidth=1, label="Overbought")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"{symbol} Price vs RSI")
    fig.tight_layout()
    plt.savefig(f"chart_{symbol}.png")
    print(f"Chart saved to chart_{symbol}.png")
    plt.show()

if __name__ == "__main__":
    show_chart("TCS")
    # show_chart("INFY")
