import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, train_series: pd.Series):
        self.model = ARIMA(train_series, order=self.order)
        self.fitted_model = self.model.fit()
        return self

    def predict(self, steps: int):
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
