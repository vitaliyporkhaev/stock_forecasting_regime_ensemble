import numpy as np
from src.models.stacking_model import StackingModel


def train_stacking(oof_preds: np.ndarray, y_true: np.ndarray):
    model = StackingModel()
    model.fit(oof_preds, y_true)
    return model
