import numpy as np


class WalkForwardCV:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X):
        n = len(X)
        step = n // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = step * (i + 1)
            test_end = step * (i + 2)

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)

            yield train_idx, test_idx
