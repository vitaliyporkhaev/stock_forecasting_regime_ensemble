import pandas as pd
from src.models.lightgbm_model import LightGBMModel


def train_lightgbm(X_train, y_train):
    model = LightGBMModel()
    model.fit(X_train, y_train)
    return model
