import yfinance as yf
import pandas as pd
import os


def download_ticker(ticker: str,
                    start: str = "2010-01-01",
                    end: str = None,
                    interval: str = "1d") -> pd.DataFrame:

    df = yf.download(ticker, start=start, end=end, interval=interval)

    df = df.dropna()
    df.columns = [c.lower() for c in df.columns]

    return df


def save_raw_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "NVDA", "SPY"]

    for t in tickers:
        df = download_ticker(t)

        save_raw_data(
            df,
            f"../../data/raw/{t}.csv"
        )

        print(f"{t} downloaded: {df.shape}")
