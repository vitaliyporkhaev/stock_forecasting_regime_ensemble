import pandas as pd


def compare_models(results_dict):
    table = []

    for name, res in results_dict.items():
        table.append({
            "Model": name,
            "Sharpe": res["sharpe"],
            "MaxDD": res["max_drawdown"],
            "Return": res["cumulative_return"]
        })

    return pd.DataFrame(table)
