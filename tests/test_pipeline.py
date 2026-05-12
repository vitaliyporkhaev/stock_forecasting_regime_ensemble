import pandas as pd
import numpy as np

from src.features.feature_pipeline import build_features
from src.regimes.clustering_regime import add_cluster_regimes


def test_pipeline_integration():
    df = pd.DataFrame({
        "close": np.random.rand(200) * 100 + 50,
        "volume": np.random.randint(1000, 5000, 200)
    })

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna()

    features = build_features(df)
    features = add_cluster_regimes(features)

    assert "regime" in features.columns
    assert len(features) > 0
