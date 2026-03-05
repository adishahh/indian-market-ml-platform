from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import glob
import os
import yaml
from sqlalchemy import text
from config.database import engine
from config.logger import get_logger

logger = get_logger("api")
app = FastAPI(title="Indian Market Standard ML API")


# Load Config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Global Model
model = None

class PredictionRequest(BaseModel):
    symbol: str = "TCS.NS"

@app.on_event("startup")
def load_model():
    global model
    try:
        # Find latest model
        artifacts_dir = "model/artifacts"
        list_of_files = glob.glob(f"{artifacts_dir}/model_cls_*.json")
        if not list_of_files:
            logger.error("No model_cls_*.json found in model/artifacts/ — predictions will fail!")
            return
            
        latest_file = max(list_of_files, key=os.path.getctime)
        
        logger.info(f"Loading model: {latest_file}")
        model = xgb.XGBClassifier()
        model.load_model(latest_file)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_symbol(req: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Clean symbol: "TCS.NS" -> "TCS"
    clean_symbol = req.symbol.replace(".NS", "").upper()
    
    # 1. Fetch latest PRE-CALCULATED features from Feature Store
    # This table now contains Technicals + Macro + Sentiment all merged.
    query = text("""
        SELECT * FROM feature_store 
        WHERE stock_id = (SELECT stock_id FROM stocks WHERE symbol = :symbol LIMIT 1)
        ORDER BY date DESC LIMIT 1
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"symbol": clean_symbol})
            
        if df.empty:
            raise HTTPException(status_code=404, detail="No feature data found in store. Run feature_store.py first.")
            
        row = df.iloc[0]
        date = row['date']
        
        # 2. Prepare Feature Vector for Model
        # The feature_store columns match the model needs (mostly).
        # We just need to align them.
        
        booster = model.get_booster()
        model_features = booster.feature_names
        
        # Create input DF
        X_input = pd.DataFrame([row])
        
        # Ensure all model columns exist (defensive programming)
        for col in model_features:
            if col not in X_input.columns:
                logger.warning(f"Missing feature column '{col}', filling with 0")
                X_input[col] = 0.0

                
        # Filter to exact model features in order
        X_input = X_input[model_features]
        
        # 3. Predict
        prediction_cls = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1] # Prob of class 1 (Buy)
        
        predicted_label = "Buy" if prediction_cls == 1 else "Sell"
        
        # 4. LOG THE PREDICTION (Phase 12 Requirement)
        log_query = text("""
            INSERT INTO prediction_logs (stock_id, prediction_date, predicted_class, probability, model_version)
            VALUES (
                (SELECT stock_id FROM stocks WHERE symbol = :symbol),
                :date,
                :cls,
                :prob,
                'v1_xgboost'
            )
        """)
        
        with engine.begin() as conn:
            conn.execute(log_query, {
                "symbol": clean_symbol,
                "date": date,
                "cls": predicted_label,
                "prob": float(probability)
            })
        
        return {
            "symbol": req.symbol,
            "date": str(date),
            "rsi": float(row['rsi_14']),
            "sentiment": float(row.get('sentiment_score', 0.0)),
            "prediction": predicted_label,
            "probability": float(probability),
            "note": "Production Inference (Feature Store + Logging Active)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate_positions")
def evaluate_positions():
    """
    Evaluates all OPEN positions in the portfolio.
    Returns a list of stocks that hit the Target Profit or Stop Loss,
    triggering a "Sell" signal.
    """
    # Load targets from config
    raw_take_profit = config.get('backtest', {}).get('take_profit', 0.15)
    raw_stop_loss = config.get('backtest', {}).get('stop_loss', -0.05)
    
    # Allow for configs written as "-5%" or integer 15 instead of float
    try: take_profit = float(raw_take_profit)
    except: take_profit = 0.15
    try: stop_loss = float(raw_stop_loss)
    except: stop_loss = -0.05
    
    sell_signals = []
    
    try:
        with engine.connect() as conn:
            # 1. Fetch all OPEN positions
            open_positions_query = text("""
                SELECT p.id, p.stock_id, s.symbol, p.buy_price 
                FROM portfolio_positions p
                JOIN stocks s ON p.stock_id = s.stock_id
                WHERE p.status = 'OPEN'
            """)
            positions_df = pd.read_sql(open_positions_query, conn)
            
            if positions_df.empty:
                return {"message": "No open positions to evaluate", "sell_signals": []}
            
            # 2. Iterate through each position and check current price
            for _, pos in positions_df.iterrows():
                symbol = pos['symbol']
                buy_price = pos['buy_price']
                
                # Fetch latest closing price (from most recent data ingestion)
                # In a real ultra-live intraday system, this would call an external API.
                price_query = text("""
                    SELECT close as current_price 
                    FROM prices 
                    WHERE stock_id = :stock_id 
                    ORDER BY date DESC LIMIT 1
                """)
                price_df = pd.read_sql(price_query, conn, params={"stock_id": pos['stock_id']})
                
                if price_df.empty:
                    continue # No price data found for this stock, skip
                    
                current_price = price_df.iloc[0]['current_price']
                
                # 3. Calculate Profit/Loss %
                profit_pct = (current_price - buy_price) / buy_price
                
                # 4. Evaluate against rules
                signal_reason = None
                
                if profit_pct >= take_profit:
                    signal_reason = f"TARGET REACHED (+{(profit_pct*100):.2f}%)"
                elif profit_pct <= stop_loss:
                    signal_reason = f"STOP LOSS HIT ({(profit_pct*100):.2f}%)"
                    
                # Optional: Check if the main XGBoost model is screaming SELL for this stock today
                # This could be added as a third check if desired.
                    
                if signal_reason:
                    sell_signals.append({
                        "symbol": symbol,
                        "buy_price": float(buy_price),
                        "current_price": float(current_price),
                        "profit_percentage": float(profit_pct * 100),
                        "reason": signal_reason
                    })
                    
        return {
            "message": f"Evaluated {len(positions_df)} open positions.",
            "take_profit_threshold": take_profit * 100,
            "stop_loss_threshold": stop_loss * 100,
            "sell_signals": sell_signals
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating positions: {str(e)}")


@app.post("/retrain")
def trigger_retrain():
    """
    Triggers the full automated retraining pipeline:
       1. Ingest fresh data
       2. Rebuild features
       3. Retrain model
       4. Deploy ONLY if performance improves
    Useful to call from n8n on a weekly schedule.
    """
    try:
        logger.info("Retraining triggered via API endpoint.")
        from automation.retrain_pipeline import run_auto_retrain
        result = run_auto_retrain()
        return result
    except Exception as e:
        logger.error(f"Retrain endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
