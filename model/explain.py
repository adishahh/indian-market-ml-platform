import xgboost as xgb
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from model.prepare_dataset import build_dataset

def get_latest_model():
    # Find latest model file
    files = glob.glob("model/artifacts/model_cls_*.json")
    if not files:
        raise FileNotFoundError("No model file found in model/artifacts/")
    latest_model = max(files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")
    return latest_model

def explain_model():
    # 1. Load Model
    model_path = get_latest_model()
    model = xgb.XGBClassifier() # Changing to Classifier since we switched in Phase 6
    model.load_model(model_path)
    
    # 2. Get Features from Model
    booster = model.get_booster()
    model_features = booster.feature_names
    print(f"Model expects features: {model_features}")
    
    # 3. Load Data
    print("Building dataset...")
    df = build_dataset()
    
    # Ensure all columns exist
    X = df[model_features]
    
    # 4. SHAP Explanation
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 5. Save Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    print("SHAP summary plot saved to shap_summary.png")
    
    # 6. Textual Importance (Feature Importance)
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": model_features, "Importance": importance})
    feat_imp = feat_imp.sort_values("Importance", ascending=False)
    
    print("\n=== TOP 10 DRIVERS OF PREDICTION ===")
    print(feat_imp.head(10))

if __name__ == "__main__":
    explain_model()
