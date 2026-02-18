import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculates the percentage of times the predicted direction (up/down) matches the actual direction.
    Assumes y_true and y_pred are returns.
    """
    # Direction is 1 if return > 0, else -1 (or 0)
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    
    # Accuracy: fraction where directions match
    return np.mean(true_dir == pred_dir)

def get_model_metrics(y_true, y_pred):
    """
    Returns a dictionary of all relevant metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    dir_acc = calculate_directional_accuracy(y_true, y_pred)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_acc
    }

def get_classification_metrics(y_true, y_pred):
    """
    Returns dictionary of classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }
