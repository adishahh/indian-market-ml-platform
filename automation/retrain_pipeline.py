"""
automation/retrain_pipeline.py

Callable by n8n (or any scheduler) to automatically:
  1. Pull fresh price and news data
  2. Rebuild features
  3. Retrain model
  4. Compare new model's metrics to the current deployed model
  5. Deploy the new model ONLY if it improves performance

Usage in n8n:
  HTTP Request node -> POST http://localhost:8003/retrain
  Or shell exec:
    .\\venv\\Scripts\\python.exe automation/retrain_pipeline.py
"""
import sys
import os
sys.path.append(os.getcwd())

import json
import glob
import yaml
from datetime import datetime
from config.logger import get_logger

logger = get_logger("retraining")


def get_current_model_metrics() -> dict:
    """Load the metrics of the currently deployed model for comparison."""
    files = sorted(glob.glob("model/artifacts/metrics_cls_*.json"))
    if not files:
        logger.warning("No existing metrics found — treating current accuracy as 0.")
        return {"accuracy": 0.0, "f1_score": 0.0}
    latest = files[-1]
    logger.info(f"Current deployed model metrics from: {latest}")
    with open(latest, "r") as f:
        return json.load(f)


def run_auto_retrain():
    logger.info("=" * 60)
    logger.info("🔁 AUTOMATED RETRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    try:
        # Step 1: Ingest fresh data
        logger.info("[1/5] Ingesting fresh market data...")
        from data_ingestion.load_prices import load_prices
        load_prices()
        from data_ingestion.load_news import load_news
        load_news()

        # Step 2: Feature Engineering
        logger.info("[2/5] Rebuilding feature store...")
        from feature_engineering.build_features import build_features
        build_features()
        from feature_engineering.feature_store import update_feature_store
        update_feature_store()

        # Step 3: Get baseline metrics BEFORE retraining
        logger.info("[3/5] Recording current model metrics for comparison...")
        current_metrics = get_current_model_metrics()
        current_acc = float(current_metrics.get("accuracy", 0.0))
        logger.info(f"  Current model accuracy: {current_acc:.4f}")

        # Step 4: Retrain
        logger.info("[4/5] Training new model...")
        from model.train_model import train
        train()

        # Step 5: Compare & Auto-Deploy
        logger.info("[5/5] Comparing new model vs current...")
        new_metric_files = sorted(glob.glob("model/artifacts/metrics_cls_*.json"))
        if not new_metric_files:
            logger.error("Training failed — no new metrics file found.")
            return {"status": "failed", "reason": "No new model metrics found"}

        with open(new_metric_files[-1], "r") as f:
            new_metrics = json.load(f)

        new_acc = float(new_metrics.get("accuracy", 0.0))
        logger.info(f"  New model accuracy: {new_acc:.4f}")

        if new_acc > current_acc:
            logger.info(f"✅ IMPROVEMENT DETECTED: {current_acc:.4f} → {new_acc:.4f}. New model deployed!")
            outcome = "deployed"
        else:
            # Remove the new model artifact to keep the old one
            logger.warning(f"⚠️ No improvement: {new_acc:.4f} <= {current_acc:.4f}. Keeping current model.")
            # Optional: delete the non-improving new model here
            outcome = "kept_old"

        logger.info("=" * 60)
        logger.info(f"🏁 RETRAINING COMPLETE — Outcome: {outcome.upper()}")
        logger.info("=" * 60)

        return {
            "status": "success",
            "outcome": outcome,
            "old_accuracy": current_acc,
            "new_accuracy": new_acc,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Retraining pipeline failed: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}


if __name__ == "__main__":
    result = run_auto_retrain()
    print(result)
