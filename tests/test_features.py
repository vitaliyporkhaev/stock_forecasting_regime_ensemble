import pandas as pd
import numpy as np

from src.features.feature_pipeline import build_features


def test_build_features_not_empty():
    df = pd.DataFrame({
        "close": np.random.rand(100) * 100 + 50,
        "volume": np.random.randint(1000, 5000, 100)
    })

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna()

    features = build_features(df)

    assert features is not None
    assert len(features) > 0
    assert "rsi" in features.columns
