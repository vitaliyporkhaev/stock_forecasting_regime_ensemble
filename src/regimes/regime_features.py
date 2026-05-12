import pandas as pd
import numpy as np


def encode_regimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    regime_dummies = pd.get_dummies(df["regime"], prefix="regime")
    df = pd.concat([df, regime_dummies], axis=1)
    return df


def add_regime_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regime_volatility"] = df["regime"] * df["volatility"]
    df["regime_return"] = df["regime"] * df["log_return"]
    return df


def add_regime_statistics(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()

    df["regime_mean_return"] = df.groupby("regime")["log_return"].transform(
        lambda x: x.rolling(window).mean()
    )

    df["regime_std_return"] = df.groupby("regime")["log_return"].transform(
        lambda x: x.rolling(window).std()
    )

    return df


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = encode_regimes(df)
    df = add_regime_interactions(df)
    df = add_regime_statistics(df)

    return df.dropna()
