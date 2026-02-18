import pandas as pd
import xgboost as xgb
import glob
import os
import matplotlib.pyplot as plt
from sqlalchemy import text
from config.database import engine
from feature_engineering.feature_store import calculate_technicals 
# Re-using logic where possible, but for backtest we often load bulk data.

def load_latest_model():
    files = glob.glob("model/artifacts/model_cls_*.json")
    if not files:
        raise FileNotFoundError("No model found!")
    latest = max(files, key=os.path.getctime)
    print(f"Loading Model: {latest}")
    model = xgb.XGBClassifier()
    model.load_model(latest)
    return model

def run_backtest(symbol="TCS", initial_capital=100000, trade_amount=20000, threshold=0.60):
    print(f"--- Starting Backtest for {symbol} ---")
    
    # 1. Fetch Data (Price + Features)
    # We need the columns matching the model training.
    # In a perfect world, we'd use 'feature_store' history.
    # But 'feature_store' is new. So we reconstruct features from 'features_daily' 
    # and merge with 'prices' for execution prices.
    
    query = text("""
        SELECT f.*, p.close as price_close, p.open as price_open, p.date as trade_date
        FROM features_daily f
        JOIN stocks s ON f.stock_id = s.stock_id
        JOIN prices p ON f.stock_id = p.stock_id AND f.date = p.date
        WHERE s.symbol = :symbol
        ORDER BY f.date ASC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol})
        
    if df.empty:
        print("No data found.")
        return

    # 2. Prepare Feature Vector for Model
    model = load_latest_model()
    booster = model.get_booster()
    model_features = booster.feature_names
    
    # Fill macro/sentiment defaults if missing (same as API logic for V1 consistency)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[model_features]
    
    # FILTER FOR 2024-2026
    start_filter = pd.Timestamp("2024-01-01")
    df['trade_date'] = pd.to_datetime(df['trade_date']) # Convert to datetime for comparison
    df = df[df['trade_date'] >= start_filter].copy()
    X = X.loc[df.index]
    
    start_date = df['trade_date'].min()
    end_date = df['trade_date'].max()
    print(f"Backtesting Interval: {start_date} to {end_date}")
    
    # 3. Predict on WHOLE history (Vectorized is faster, but loop simulates reality)
    print("Running Predictions...")
    probs = model.predict_proba(X)[:, 1] # Prob of Buy
    df['probability'] = probs
    df['prediction'] = (probs > threshold).astype(int)
    
    # 4. Simulate Trades
    capital = initial_capital
    position = 0 # Shares held
    entry_price = 0
    trades = []
    equity_curve = []
    
    for i, row in df.iterrows():
        price = row['price_close']
        date = row['trade_date']
        
        # Sell Rule: Exit after 5 days OR if Prob drops (Simplified: Exit if we hold and prediction is weak)
        # Strategy: Buy on Strong Signal, Hold for 5 days fixed (Simple)
        
        # Check if we need to sell (Fixed Hold Period Logic)
        # ... For simplicity, let's say: 
        # Buy if Prob > 0.6. Sell if RSI > 70 OR Prob < 0.4.
        
        target_shares = 0
        signal = "HOLD"
        
        if position == 0:
            if row['probability'] > threshold and row['rsi_14'] < 70:
                signal = "BUY"
        elif position > 0:
             # Sell if signal lost or overbought
             if row['rsi_14'] > 70 or row['probability'] < 0.4:
                 signal = "SELL"
                 
        # Execute
        if signal == "BUY" and position == 0:
            position = int(trade_amount / price)
            entry_price = price
            capital -= position * price
            trades.append({"date": date, "type": "BUY", "price": price, "shares": position})
            
        elif signal == "SELL" and position > 0:
            revenue = position * price
            profit = revenue - (position * entry_price)
            capital += revenue
            trades.append({"date": date, "type": "SELL", "price": price, "shares": position, "profit": profit})
            position = 0
            
        # Markup Daily Value
        current_val = capital + (position * price)
        equity_curve.append({"date": date, "equity": current_val})
        
    # 5. Results
    final_equity = equity_curve[-1]['equity']
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    win_trades = [t for t in trades if t.get('profit', 0) > 0]
    loss_trades = [t for t in trades if t.get('profit', 0) <= 0 and t['type'] == 'SELL']
    
    print(f"\n--- Results for {symbol} ---")
    print(f"Initial: ₹{initial_capital}")
    print(f"Final:   ₹{final_equity:.2f}")
    print(f"Return:  {total_return:.2f}%")
    print(f"Trades:  {len(trades)//2} Round trips")
    if len(win_trades) + len(loss_trades) > 0:
        win_rate = len(win_trades) / (len(win_trades) + len(loss_trades)) * 100
        print(f"Win Rate: {win_rate:.1f}%")
    
    # Save Plot
    edf = pd.DataFrame(equity_curve)
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(edf['date']), edf['equity'], label='Strategy')
    plt.title(f"Backtest: {symbol} (Initial ₹1L)")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.savefig(f"backtest_{symbol}.png")
    print(f"Equity Curve saved to backtest_{symbol}.png")

if __name__ == "__main__":
    run_backtest("TCS")
