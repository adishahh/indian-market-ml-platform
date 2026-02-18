import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

SCALER_PATH = "model/artifacts/scaler.pkl"

def fit_and_save_scaler(df: pd.DataFrame, feature_columns: list):
    """
    Fits a StandardScaler on the given features and saves it.
    """
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    
    scaler = StandardScaler()
    scaler.fit(df[feature_columns])
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    return scaler

def load_scaler():
    """Loads the saved scaler."""
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None
