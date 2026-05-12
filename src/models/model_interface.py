import numpy as np


class ModelInterface:

    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type

    def fit(self, X, y=None):
        if self.model_type in ["lgb", "lstm"]:
            self.model.fit(X, y)
        elif self.model_type == "arima":
            self.model.fit(X)
        return self

    def predict(self, X):

        if self.model_type == "lgb":
            return self.model.predict(X)

        if self.model_type == "lstm":
            return self.model.predict(X).reshape(-1)

        if self.model_type == "arima":
            return self.model.predict(len(X))

        raise ValueError("Unknown model type")
