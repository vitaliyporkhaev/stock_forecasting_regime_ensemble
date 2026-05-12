import pandas as pd


def build_comparison_table(results: dict):
    """
    results = {
        "Naive": {...},
        "ARIMA": {...},
        "LightGBM": {...},
        "LSTM": {...},
        "Meta": {...},
        "BuyHold": {...}
    }
    """

    rows = []

    for name, r in results.items():
        rows.append({
            "Model": name,
            "RMSE": r.get("rmse"),
            "MAE": r.get("mae"),
            "Direction Acc": r.get("direction_acc"),
            "Sharpe": r.get("sharpe"),
            "Max DD": r.get("max_drawdown"),
            "Return": r.get("return")
        })

    return pd.DataFrame(rows)


def save_table(df, path="results/model_comparison.csv"):
    df.to_csv(path, index=False)
