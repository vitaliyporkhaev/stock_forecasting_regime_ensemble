import numpy as np


def ensemble_predict(predictions: np.ndarray, weights=None):

    if weights is None:
        weights = np.ones(predictions.shape[1]) / predictions.shape[1]

    return np.dot(predictions, weights)
