import pandas as pd
import numpy as np

def log_returns(close: pd.Series):
    return np.log(close / close.shift(1))


def simple_returns(close: pd.Series):
    return close.pct_change()

def rolling_features(series: pd.Series, window: int = 20):
    return pd.DataFrame({
        "rolling_mean": series.rolling(window).mean(),
        "rolling_std": series.rolling(window).std(),
        "rolling_min": series.rolling(window).min(),
        "rolling_max": series.rolling(window).max(),
        "rolling_skew": series.rolling(window).skew(),
        "rolling_kurt": series.rolling(window).kurt()
    })
