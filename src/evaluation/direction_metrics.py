import numpy as np


def direction_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.sign(y_true) == np.sign(y_pred))


def f1_direction(y_true, y_pred):

    y_true = (np.array(y_true) > 0).astype(int)
    y_pred = (np.array(y_pred) > 0).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return 2 * precision * recall / (precision + recall + 1e-9)
