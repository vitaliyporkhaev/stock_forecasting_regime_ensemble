import numpy as np


def ensemble_predict(predictions: np.ndarray, weights=None):
    """
    predictions shape: (n_samples, n_models)
    """

    if weights is None:
        weights = np.ones(predictions.shape[1]) / predictions.shape[1]

    return np.dot(predictions, weights)
