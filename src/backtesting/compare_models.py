import pandas as pd


def compare_models(results_dict):
    """
    results_dict:
    {
        "LightGBM": {...},
        "LSTM": {...},
        "Meta": {...},
        "BuyHold": {...}
    }
    """

    table = []

    for name, res in results_dict.items():
        table.append({
            "Model": name,
            "Sharpe": res["sharpe"],
            "MaxDD": res["max_drawdown"],
            "Return": res["cumulative_return"]
        })

    return pd.DataFrame(table)
