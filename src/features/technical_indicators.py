import pandas as pd
import numpy as np

# def rsi(series: pd.Series, period: int = 14):
#     delta = series.diff()

#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

#     rs = gain / (loss + 1e-9)
#     return 100 - (100 / (1 + rs))

def add_rsi(df, period=14):
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-8)

    df["rsi"] = 100 - (100 / (1 + rs))

    return df

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int):
    return series.rolling(window=window).mean()

def macd(close: pd.Series):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)

    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal

    return macd_line, signal, hist

def bollinger_bands(close: pd.Series, window: int = 20):
    ma = sma(close, window)
    std = close.rolling(window).std()

    upper = ma + 2 * std
    lower = ma - 2 * std

    return upper, lower
