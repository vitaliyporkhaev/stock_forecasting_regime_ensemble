import numpy as np
import pandas as pd


def prediction_to_signal(pred: np.ndarray,
                         threshold: float = 0.0):
    """
    1 - long
    -1 - short
    0 - no position (optional if threshold used)
    """

    signals = np.where(pred > threshold, 1,
               np.where(pred < -threshold, -1, 0))

    return signals


def continuous_position(pred: np.ndarray):
    return np.tanh(pred)
