import numpy as np
from src.models.stacking_model import StackingModel


def train_stacking(oof_preds: np.ndarray, y_true: np.ndarray):
    """
    oof_preds shape: (n_samples, n_models)
    """
    model = StackingModel()
    model.fit(oof_preds, y_true)
    return model
