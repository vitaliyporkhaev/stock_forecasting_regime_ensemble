import numpy as np
from sklearn.linear_model import LinearRegression


class StackingModel:
    def __init__(self):
        self.meta_model = LinearRegression()

    def fit(self, preds: np.ndarray, y: np.ndarray):
        self.meta_model.fit(preds, y)
        return self

    def predict(self, preds: np.ndarray):
        return self.meta_model.predict(preds)
