import numpy as np
import pandas as pd


def naive_forecast(df: pd.DataFrame):
    return df["log_return"].shift(1)


def moving_average_signal(df: pd.DataFrame, window: int = 5):
    return df["log_return"].rolling(window).mean()


def buy_and_hold_returns(df: pd.DataFrame):
    return (1 + df["log_return"]).cumprod()
