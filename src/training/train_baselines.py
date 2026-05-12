import numpy as np


def train_naive(y_train):
    return y_train.shift(1)


def train_mean(y_train, window=5):
    return y_train.rolling(window).mean()
