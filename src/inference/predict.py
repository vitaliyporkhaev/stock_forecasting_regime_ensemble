import numpy as np
import pandas as pd


def predict_model(model, X):
    """
    Унифицированный predict для любых моделей
    (LightGBM, LSTM, ARIMA через interface)
    """
    return model.predict(X)
