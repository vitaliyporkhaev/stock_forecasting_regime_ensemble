import numpy as np
import pandas as pd

def prediction_to_signal(pred: np.ndarray, threshold: float = 0.0):
    pred = np.asarray(pred)

    signals = np.where(pred > threshold, 1, -1)  # long / short

    return signals


def continuous_position(pred: np.ndarray):
    return np.tanh(pred)
