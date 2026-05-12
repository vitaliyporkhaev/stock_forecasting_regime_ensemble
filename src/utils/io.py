import os
import pandas as pd


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def load_csv(path: str):
    return pd.read_csv(path, index_col=0, parse_dates=True)
