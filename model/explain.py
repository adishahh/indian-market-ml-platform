# import xgboost as xgb
# import shap
# import pandas as pd

# from model.prepare_dataset import build_dataset

# FEATURES = [
#     "return_1d",
#     "return_5d",
#     "return_20d",
#     "sma_20",
#     "sma_50",
#     "ema_20",
#     "rsi_14",
#     "volatility_20d",
# ]


# def explain_model():
#     df = build_dataset()
#     df["date"] = pd.to_datetime(df["date"])

#     X = df[FEATURES]

#     model = xgb.XGBRegressor()
#     model.load_model("model/artifacts/model.json")

#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)

#     shap.summary_plot(shap_values, X)


# if __name__ == "__main__":
#     explain_model()
import xgboost as xgb
import shap
import pandas as pd
import matplotlib.pyplot as plt

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


def explain_model():
    # Load dataset
    df = build_dataset()
    df["date"] = pd.to_datetime(df["date"])

    X = df[FEATURES]

    # Load trained model
    model = xgb.XGBRegressor()
    model.load_model("model/artifacts/model.json")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ---- SAVE SHAP SUMMARY PLOT ----
    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X,
        show=False,      # IMPORTANT: required to save
        plot_type="dot"  # use "dot" if you want scatter style
    )

    plt.tight_layout()
    plt.savefig(
        "model/artifacts/shap_summary.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    print("SHAP summary plot saved to model/artifacts/shap_summary.png")


if __name__ == "__main__":
    explain_model()
