"""
run_pipeline.py

Master pipeline script — runs the full data → features → train → backtest flow.
This is the correct entrypoint for setting up or refreshing the entire system.
"""
import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n=======================================================")
    print(f"🚀 Running: {script_path}")
    print(f"=======================================================")
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        if result.returncode == 0:
            print(f"✅ Success: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_path}: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("🇮🇳 Indian Market ML Platform — Full Pipeline")
    print("=" * 60)
    print("WARNING: This will re-ingest data, re-train the model, and re-populate the DB.")
    print("Do NOT run this during live trading hours.")
    input("Press Enter to continue (or Ctrl+C to cancel)...")

    # 1. Data Ingestion
    run_script("data_ingestion/load_stocks.py")
    run_script("data_ingestion/load_prices.py")
    run_script("data_ingestion/load_index.py")
    run_script("data_ingestion/load_news.py")
    print("\n[INFO] Data ingestion step complete.")

    # 2. Feature Engineering (Raw → Features)
    run_script("feature_engineering/build_features.py")

    # 3. Feature Store (Production Hardening)
    run_script("feature_engineering/feature_store.py")

    # 4. Model Training (Train → Artifact)
    run_script("model/train_model.py")

    # 5. Backtesting (Verification)
    run_script("model/backtest.py")

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
