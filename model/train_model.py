import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model.prepare_dataset import build_dataset


FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi_14",
    "volatility_20d",
]


def train():
    df = build_dataset()

    X = df[FEATURES]
    y = df["target_return"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    model.save_model("model/artifacts/model.json")

    metrics = {"mse": mse}
    with open("model/artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training completed")
    print("MSE:", mse)


if __name__ == "__main__":
    train()
