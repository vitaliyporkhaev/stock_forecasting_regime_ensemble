import pandas as pd

from src.features.technical_indicators import (
    rsi, ema, sma, macd, bollinger_bands
)

from src.features.statistical_features import (
    log_returns, simple_returns, rolling_features
)

from src.features.volatility_features import realized_volatility

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_return"] = log_returns(df["close"])
    df["simple_return"] = simple_returns(df["close"])

    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

    df["rsi_14"] = rsi(df["close"])

    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    df["sma_20"] = sma(df["close"], 20)

    macd_line, signal, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    upper, lower = bollinger_bands(df["close"])
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_width"] = upper - lower

    df["volatility_20"] = realized_volatility(df["log_return"])

    roll = rolling_features(df["log_return"])
    df = pd.concat([df, roll], axis=1)

    df = df.dropna()

    return df
