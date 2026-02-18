import optuna
import xgboost as xgb
import pandas as pd
import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, log_loss
from model.prepare_dataset import build_dataset

# Load Base Config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_CONFIG = config.get("model", {})
TARGET_COL = MODEL_CONFIG.get("target", "target_class")

def objective(trial):
    # 2. Suggest Hyperparameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "n_jobs": -1
    }

    # 3. Train
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # 4. Evaluate
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    return accuracy

if __name__ == "__main__":
    print("Loading dataset...")
    df = build_dataset()
    
    if df.empty:
        print("Dataset empty.")
        exit()

    # Feature Selection
    exclude = ['stock_id', 'date', 'target_return', 'target_class', 'excess_return']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"Starting optimization with {len(X)} rows...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=600) # 50 trials or 10 mins

    print("\nBest trial:")
    trial = study.best_trial

    print(f"  Value (Accuracy): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best params to a yaml snippet
    best_params = {
        "model": {
            "params": trial.params
        }
    }
    # Add defaults that aren't tuned
    best_params["model"]["params"]["objective"] = "binary:logistic"
    best_params["model"]["params"]["eval_metric"] = "logloss"
    best_params["model"]["params"]["random_state"] = 42
    
    with open("config/best_params.yaml", "w") as f:
        yaml.dump(best_params, f)
        
    print("\nSaved best params to config/best_params.yaml")
