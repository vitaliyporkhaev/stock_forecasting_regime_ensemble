import pandas as pd


class TimeSeriesSplit:

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, df: pd.DataFrame):
        df = df.sort_index()

        total_size = len(df)
        fold_size = total_size // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_end = fold_size * (i + 2)

            train = df.iloc[:train_end]
            test = df.iloc[train_end:test_end]

            yield train, test


def train_test_split_time(df: pd.DataFrame, train_ratio: float = 0.8):
    split_idx = int(len(df) * train_ratio)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    return train, test
