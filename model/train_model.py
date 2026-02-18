import xgboost as xgb
import pandas as pd
import json
import yaml
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from model.prepare_dataset import build_dataset
from model.metrics import get_classification_metrics

# Load Config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_CONFIG = config.get("model", {})
PARAMS = MODEL_CONFIG.get("params", {})
TARGET_COL = MODEL_CONFIG.get("target", "target_class") # "target_class"

# Define paths
ARTIFACTS_DIR = "model/artifacts"
EXPERIMENTS_DIR = "model/experiments"
HISTORY_PATH = os.path.join(EXPERIMENTS_DIR, "history.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

def feature_selection(df):
    exclude = ['stock_id', 'date', 'target_return', 'target_class', 'excess_return']
    return [c for c in df.columns if c not in exclude]

def save_experiment_log(timestamp, params, metrics, model_path):
    log_entry = {
        "timestamp": timestamp,
        **params,
        **metrics,
        "model_path": model_path
    }
    df_log = pd.DataFrame([log_entry])
    if not os.path.exists(HISTORY_PATH):
        df_log.to_csv(HISTORY_PATH, index=False)
    else:
        df_log.to_csv(HISTORY_PATH, mode='a', header=False, index=False)
    print(f"Experiment logged to {HISTORY_PATH}")

def train():
    print("Loading dataset...")
    df = build_dataset()
    
    if df.empty:
        print("Dataset is empty.")
        return

    feature_cols = feature_selection(df)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {TARGET_COL}")

    X = df[feature_cols]
    y = df[TARGET_COL]

    # Time-series split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"Training XGBoost Classifier parameters: {PARAMS}")
    
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    
    # Evaluation
    metrics = get_classification_metrics(y_test, preds)
    print(f"Evaluation Metrics: {metrics}")

    # Versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_cls_{timestamp}.json"
    model_path = os.path.join(ARTIFACTS_DIR, model_filename)
    
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save Metrics
    metrics_path = os.path.join(ARTIFACTS_DIR, f"metrics_cls_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    save_experiment_log(timestamp, PARAMS, metrics, model_path)

if __name__ == "__main__":
    train()
