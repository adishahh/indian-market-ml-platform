import sys
import os
sys.path.append(os.getcwd())
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
from config.logger import get_logger

logger = get_logger(__name__)


# Load Config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_CONFIG = config.get("model", {})
TARGET_COL = MODEL_CONFIG.get("target", "target_class")

# Load best params from best_params.yaml (has the tuned hyperparams)
with open("config/best_params.yaml", "r") as f:
    best_config = yaml.safe_load(f)

PARAMS = best_config.get("model", {}).get("params", {})

# Define paths
ARTIFACTS_DIR = "model/artifacts"
EXPERIMENTS_DIR = "model/experiments"
HISTORY_PATH = os.path.join(EXPERIMENTS_DIR, "history.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# Fixed column schema for history.csv — always write these columns
HISTORY_COLUMNS = [
    "timestamp",
    "max_depth", "learning_rate", "n_estimators",
    "subsample", "colsample_bytree", "min_child_weight",
    "reg_alpha", "reg_lambda", "random_state",
    "objective", "eval_metric",
    "accuracy", "precision", "recall", "f1_score",
    "model_path"
]

def save_experiment_log(timestamp, params, metrics, model_path):
    row = {"timestamp": timestamp, "model_path": model_path}
    # Fill in known param columns (default to empty string if not present)
    for col in HISTORY_COLUMNS:
        if col not in row:
            row[col] = params.get(col, metrics.get(col, ""))

    df_log = pd.DataFrame([row], columns=HISTORY_COLUMNS)

    if not os.path.exists(HISTORY_PATH):
        df_log.to_csv(HISTORY_PATH, index=False)
    else:
        df_log.to_csv(HISTORY_PATH, mode='a', header=False, index=False)
    print(f"Experiment logged to {HISTORY_PATH}")


def feature_selection(df):
    exclude = ['stock_id', 'date', 'target_return', 'target_class', 'excess_return']
    return [c for c in df.columns if c not in exclude]


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

    # --- Class Imbalance Fix ---
    # Indian markets trend upward most of the time, causing far more
    # "Sell/Hold" labels (class 0) than "Buy" labels (class 1).
    # scale_pos_weight = count(class 0) / count(class 1) tells XGBoost
    # to penalise misclassifying Buy signals more heavily.
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Class balance — Sell/Hold: {neg_count} | Buy: {pos_count} | scale_pos_weight: {scale_pos_weight:.3f}")

    model = xgb.XGBClassifier(**PARAMS, scale_pos_weight=scale_pos_weight)
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
