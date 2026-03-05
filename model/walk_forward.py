import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import accuracy_score, f1_score
from model.prepare_dataset import build_dataset

# Load best params
with open("config/best_params.yaml", "r") as f:
    best_config = yaml.safe_load(f)
PARAMS = best_config.get("model", {}).get("params", {})

TRAIN_YEARS = 3
TEST_MONTHS = 6
TARGET_COL = "target_class"


def feature_selection(df):
    exclude = ['stock_id', 'date', 'target_return', 'target_class', 'excess_return']
    return [c for c in df.columns if c not in exclude]


def walk_forward_backtest():
    print("Loading dataset for walk-forward validation...")
    df = build_dataset()

    if df.empty:
        print("Dataset is empty.")
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = feature_selection(df)
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    start_date = df["date"].min()
    end_date = df["date"].max()

    results = []
    train_start = start_date

    fold = 0
    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        if test_end > end_date:
            break

        train_df = df[(df["date"] >= train_start) & (df["date"] < train_end)]
        test_df  = df[(df["date"] >= train_end) & (df["date"] < test_end)]

        if len(train_df) < 500 or len(test_df) < 100:
            train_start += pd.DateOffset(months=6)
            continue

        fold += 1
        print(f"\nFold {fold}: Train [{train_start.date()} → {train_end.date()}] | Test [{train_end.date()} → {test_end.date()}]")
        print(f"  Train rows: {len(train_df)} | Test rows: {len(test_df)}")

        # Train classifier with tuned params
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(train_df[feature_cols], train_df[TARGET_COL])

        # Evaluate
        preds = model.predict(test_df[feature_cols])
        acc = accuracy_score(test_df[TARGET_COL], preds)
        f1  = f1_score(test_df[TARGET_COL], preds, zero_division=0)

        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f}")

        results.append({
            "fold": fold,
            "train_start": train_start.date(),
            "train_end": train_end.date(),
            "test_start": train_end.date(),
            "test_end": test_end.date(),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
        })

        train_start += pd.DateOffset(months=6)

    results_df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    print(f"\nAverage Accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"Average F1 Score: {results_df['f1_score'].mean():.4f}")
    print(f"Min Accuracy:     {results_df['accuracy'].min():.4f}")
    print(f"Max Accuracy:     {results_df['accuracy'].max():.4f}")

    # Save results
    os.makedirs("model/experiments", exist_ok=True)
    results_df.to_csv("model/experiments/walk_forward_results.csv", index=False)
    print("\nResults saved to model/experiments/walk_forward_results.csv")

    return results_df


if __name__ == "__main__":
    walk_forward_backtest()
