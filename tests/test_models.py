import numpy as np
import pandas as pd

from src.models.lightgbm_model import LightGBMModel


def test_lightgbm_fit_predict():
    X = pd.DataFrame(np.random.randn(100, 5))
    y = np.random.randn(100)

    model = LightGBMModel()
    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)
    assert np.isfinite(preds).all()
