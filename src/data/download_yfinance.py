import yfinance as yf
import pandas as pd


def download_ticker(
    ticker: str,
    start: str = "2010-01-01",
    end: str = None,
    interval: str = "1d"
) -> pd.DataFrame:

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]

    if "adj close" in df.columns:
        df["close"] = df["adj close"]

    if "close" not in df.columns:
        raise ValueError(f"'close' column not found. Columns: {df.columns}")

    df = df.reset_index()

    df.columns = [str(c).lower() for c in df.columns]

    return df
