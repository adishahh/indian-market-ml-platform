from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import glob
import os
import yaml
from sqlalchemy import text
from config.database import engine

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
            print("No model found!")
            return
            
        latest_file = max(list_of_files, key=os.path.getctime)
        
        print(f"Loading model: {latest_file}")
        model = xgb.XGBClassifier()
        model.load_model(latest_file)
    except Exception as e:
        print(f"Error loading model: {e}")

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
                print(f"Warning: Missing column {col}, filling 0")
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
