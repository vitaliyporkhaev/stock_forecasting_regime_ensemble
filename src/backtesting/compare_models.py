import pandas as pd

def build_comparison_table(results):

    rows = []

    for name, metrics in results.items():

        rows.append({
            "Model": name,
            "Sharpe": metrics.get("sharpe"),
            "Max DD": metrics.get("max_drawdown"),
            "Return": metrics.get("cumulative_return")
        })

    return pd.DataFrame(rows)
