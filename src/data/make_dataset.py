import pandas as pd
import numpy as np
import os


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["target"] = df["log_return"].shift(-1)

    df = df.dropna()

    return df


def save_processed(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "NVDA", "SPY"]

    all_data = []

    for t in tickers:
        df = load_data(f"../../data/raw/{t}.csv")

        df = create_target(df)
        df["ticker"] = t

        all_data.append(df)

    final_df = pd.concat(all_data)

    save_processed(final_df, "../../data/processed/dataset.csv")

    print("Dataset shape:", final_df.shape)
